#include "system-one.h"

#include "chat.h"                 // pulls in the jinja engine the chat templates use
#include "json.h"

#include <algorithm>
#include <cmath>
#include <cstring>

using json = common_json;

namespace system_one {

// ---------------------------------------------------------------- model metadata

static bool meta_str(const llama_model * model, const char * key, std::string & out) {
    const int32_t n = llama_model_meta_val_str(model, key, nullptr, 0);
    if (n < 0) return false;
    std::vector<char> buf(n + 1);
    if (llama_model_meta_val_str(model, key, buf.data(), buf.size()) < 0) return false;
    out.assign(buf.data());
    return true;
}

static void meta_int(const llama_model * model, const char * key, int & out) {
    std::string s;
    if (meta_str(model, key, s) && !s.empty()) out = std::atoi(s.c_str());
}

static void meta_float(const llama_model * model, const char * key, float & out) {
    std::string s;
    if (meta_str(model, key, s) && !s.empty()) out = std::strtof(s.c_str(), nullptr);
}

static void meta_bool(const llama_model * model, const char * key, bool & out) {
    std::string s;
    if (meta_str(model, key, s) && !s.empty()) out = (s == "true" || s == "1");
}

bool so_config::from_model(const llama_model * model, so_config & out, std::string & err) {
    if (!meta_str(model, "system_one.template", out.template_src) || out.template_src.empty()) {
        err = "model has no system_one template (system_one.template); "
              "convert the checkpoint with its system_one.json sidecar, or pass a template in the request";
        return false;
    }

    std::string s;
    if (meta_str(model, "system_one.template.segment_separator", s) && !s.empty()) out.segment_separator = s;
    if (meta_str(model, "system_one.slot",              s) && !s.empty()) out.slot_rule = s;
    if (meta_str(model, "system_one.readout",           s) && !s.empty()) out.readout   = s;
    if (meta_str(model, "system_one.letters",           s) && !s.empty()) out.letters   = s;
    if (meta_str(model, "system_one.kind_tag.noul",     s) && !s.empty()) out.tag_noul   = s;
    if (meta_str(model, "system_one.kind_tag.choice",   s) && !s.empty()) out.tag_choice = s;
    if (meta_str(model, "system_one.kind_tag.score",    s) && !s.empty()) out.tag_score  = s;
    meta_str(model, "system_one.truncation",        out.truncation);
    // noul_options is a kv array; llama.cpp stringifies it, so accept the common forms
    if (meta_str(model, "system_one.noul_options", s) && !s.empty()) {
        std::vector<std::string> parsed;
        std::string cur;
        for (char ch : s) {
            if (ch == '[' || ch == ']' || ch == '"' || ch == ' ') continue;
            if (ch == ',') { if (!cur.empty()) parsed.push_back(cur); cur.clear(); continue; }
            cur += ch;
        }
        if (!cur.empty()) parsed.push_back(cur);
        if (parsed.size() >= 2) out.noul_options = parsed;
    }
    meta_str(model, "system_one.truncation_marker", out.truncation_marker);
    meta_int  (model, "system_one.max_state_tokens", out.max_state_tokens);
    meta_bool (model, "system_one.bos",              out.add_bos);
    meta_float(model, "system_one.calibration_temperature", out.calibration_temperature);
    meta_bool (model, "system_one.calibration_folded",      out.calibration_folded);

    if (meta_str(model, "system_one.mask_token", s) && !s.empty()) out.mask_text = s;
    {
        int id = -1;
        meta_int(model, "system_one.mask_token_id", id);
        out.mask_token = (llama_token) id;
    }

    const bool letter = out.readout == "letter_slot"  && out.slot_rule == "last_token_of_question_segment";
    const bool masked = out.readout == "masked_slot"  && out.slot_rule == "mask_token_per_question";
    if (!letter && !masked) {
        err = "unsupported System One readout/slot combination: " + out.readout + " / " + out.slot_rule;
        return false;
    }
    if (masked && out.mask_token < 0) {
        err = "masked_slot needs system_one.mask_token_id";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- rendering

static const std::string & tag_of(const so_config & cfg, kind k) {
    switch (k) {
        case kind::choice: return cfg.tag_choice;
        case kind::score:  return cfg.tag_score;
        default:           return cfg.tag_noul;
    }
}

static const char * kind_name(kind k) {
    switch (k) {
        case kind::choice: return "choice";
        case kind::score:  return "score";
        default:           return "noul";
    }
}

bool render_segments(const so_config & cfg,
                     const std::string & state,
                     const std::vector<question> & qs,
                     std::vector<std::string> & segments,
                     std::string & err) {
    if (cfg.segment_separator.empty()) {
        err = "system_one.template.segment_separator is empty";
        return false;
    }

    json questions = json::array();
    for (size_t i = 0; i < qs.size(); i++) {
        const question & q = qs[i];
        json options = json::array();
        for (size_t j = 0; j < q.options.size(); j++) {
            const bool has_desc = j < q.descs.size() && !q.descs[j].empty();
            options.push_back(json{
                {"letter", j < cfg.letters.size() ? std::string(1, cfg.letters[j]) : std::string("?")},
                {"option", q.options[j]},
                {"desc",   has_desc ? json(q.descs[j]) : json(nullptr)},
            });
        }
        questions.push_back(json{
            {"k",       (int) i + 1},          // 1-based, as the template expects
            {"kind",    kind_name(q.k)},
            {"tag",     tag_of(cfg, q.k)},
            {"text",    q.text},
            {"options", options},
        });
    }

    std::string rendered;
    try {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(cfg.template_src);
        jinja::program prog = jinja::parse_from_tokens(lexer_res);

        jinja::context ctx(lexer_res.source);
        const json inp = json{
            {"state",     state},
            {"questions", questions},
            {"sep",       cfg.segment_separator},
            {"mask",      cfg.mask_text},     // empty unless the format places a mask token
        };
        jinja::global_from_json(ctx, inp, false);

        jinja::runtime runtime(ctx);
        const jinja::value results = runtime.execute(prog);
        auto parts = jinja::runtime::gather_string_parts(results);
        rendered = parts->as_string().str();
    } catch (const std::exception & e) {
        err = std::string("system_one template failed to render: ") + e.what();
        return false;
    }

    segments.clear();
    size_t pos = 0;
    while (true) {
        const size_t next = rendered.find(cfg.segment_separator, pos);
        if (next == std::string::npos) {
            segments.push_back(rendered.substr(pos));
            break;
        }
        segments.push_back(rendered.substr(pos, next - pos));
        pos = next + cfg.segment_separator.size();
    }
    // a separator at the very start produces a leading empty piece; drop empties, they
    // tokenize to nothing anyway and would only confuse the slot indexing
    segments.erase(std::remove_if(segments.begin(), segments.end(),
                                  [](const std::string & s) { return s.empty(); }),
                   segments.end());

    if (segments.size() < qs.size()) {
        err = "template produced " + std::to_string(segments.size()) + " segments for " +
              std::to_string(qs.size()) + " questions: the separator must precede every question block";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- tokenization

static std::vector<llama_token> encode(const llama_vocab * vocab, const std::string & text,
                                       bool add_special, bool parse_special = false) {
    if (text.empty()) return {};
    int n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, add_special, parse_special);
    std::vector<llama_token> out(n);
    n = llama_tokenize(vocab, text.data(), (int32_t) text.size(), out.data(), n, add_special, parse_special);
    out.resize(std::max(0, n));
    return out;
}

std::vector<llama_token> tokenize_whole(const llama_vocab * vocab, const std::string & text, bool add_special) {
    return encode(vocab, text, add_special, true);
}

std::string truncate_state(const llama_vocab * vocab, const so_config & cfg, const std::string & state) {
    if (cfg.max_state_tokens <= 0) return state;

    const auto ids = encode(vocab, state, false);
    if ((int) ids.size() <= cfg.max_state_tokens) return state;

    const int head = (int) (cfg.max_state_tokens * cfg.head_fraction);
    const int tail = cfg.max_state_tokens - head - cfg.tail_reserve;

    // round-trips through the tokenizer, so the cut text may not be byte-identical to a
    // substring of the original -- acceptable, and only on this path
    auto detok = [&](const llama_token * p, int n) {
        if (n <= 0) return std::string();
        int need = -llama_detokenize(vocab, p, n, nullptr, 0, false, false);
        std::vector<char> buf(need + 1);
        const int got = llama_detokenize(vocab, p, n, buf.data(), buf.size(), false, false);
        return got > 0 ? std::string(buf.data(), got) : std::string();
    };

    std::string out = detok(ids.data(), head) + cfg.truncation_marker;
    if (tail > 0) out += detok(ids.data() + ids.size() - tail, tail);
    return out;
}

bool tokenize_segments(const llama_vocab * vocab,
                       const so_config & cfg,
                       const std::vector<std::string> & segments,
                       size_t n_questions,
                       tokenized & out,
                       std::string & err) {
    if (segments.size() < n_questions) {
        err = "fewer segments than questions";
        return false;
    }

    out.ids.clear();
    out.slots.clear();
    if (cfg.add_bos) out.ids.push_back(llama_vocab_bos(vocab));

    // the mask token is written into the template as text, so it has to be parsed back into
    // its id; HF's tokenizers split on added tokens regardless of add_special_tokens, so this
    // matches the reference. The letter-slot path keeps parse_special off, as validated.
    const bool masked = cfg.slot_rule == "mask_token_per_question";

    // the question blocks are the trailing segments
    const size_t first_question = segments.size() - n_questions;
    for (size_t i = 0; i < segments.size(); i++) {
        const auto ids = encode(vocab, segments[i], false, masked);
        if (ids.empty()) {
            err = "segment " + std::to_string(i) + " tokenized to nothing";
            return false;
        }
        const size_t base = out.ids.size();
        out.ids.insert(out.ids.end(), ids.begin(), ids.end());

        if (i < first_question) continue;

        if (!masked) {
            out.slots.push_back((int) out.ids.size() - 1);
            continue;
        }

        // the answer is read at the mask token itself; take the last one in the segment
        int slot = -1;
        for (size_t j = ids.size(); j-- > 0; ) {
            if (ids[j] == cfg.mask_token) { slot = (int) (base + j); break; }
        }
        if (slot < 0) {
            err = "question segment " + std::to_string(i - first_question) +
                  " has no mask token (" + std::to_string(cfg.mask_token) + ") after tokenization";
            return false;
        }
        out.slots.push_back(slot);
    }
    return true;
}

bool letter_tokens(const llama_vocab * vocab, const std::string & letters,
                   std::vector<llama_token> & out) {
    const int n_vocab = llama_vocab_n_tokens(vocab);
    out.assign(letters.size(), -1);
    for (llama_token id = 0; id < n_vocab; id++) {
        const char * txt = llama_vocab_get_text(vocab, id);
        if (!txt || strlen(txt) != 1) continue;
        const size_t pos = letters.find(txt[0]);
        if (pos != std::string::npos && out[pos] < 0) out[pos] = id;
    }
    return std::find(out.begin(), out.end(), -1) == out.end();
}

// ---------------------------------------------------------------- readout

static std::vector<float> softmax(const std::vector<float> & x) {
    const float mx = *std::max_element(x.begin(), x.end());
    std::vector<float> p(x.size());
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); i++) { p[i] = std::exp(x[i] - mx); sum += p[i]; }
    for (auto & v : p) v /= sum;
    return p;
}

answer answer_from_logits(const float * row, const std::vector<llama_token> & letter_ids, int n_options) {
    answer a;
    a.logits.resize(n_options);
    for (int j = 0; j < n_options; j++) a.logits[j] = row[letter_ids[j]];
    a.probs  = softmax(a.logits);
    a.choice = (int) (std::max_element(a.probs.begin(), a.probs.end()) - a.probs.begin());

    // confidence = 1 - H(p)/ln K, as the Jev-compatible servers report it
    if (n_options > 1) {
        double h = 0.0;
        for (float p : a.probs) if (p > 0.0f) h -= (double) p * std::log((double) p);
        a.confidence = (float) (1.0 - h / std::log((double) n_options));
    } else {
        a.confidence = 1.0f;
    }
    return a;
}

float score_expectation(const answer & a) {
    double e = 0.0;
    for (size_t i = 0; i < a.probs.size(); i++) e += (double) i * (double) a.probs[i];
    return (float) e;
}

bool read_slots(llama_context * ctx,
                const tokenized & t,
                const std::vector<int> & n_options,
                const std::vector<llama_token> & letter_ids,
                std::vector<answer> & out,
                std::string & err) {
    if (t.slots.size() != n_options.size()) {
        err = "slot count does not match question count";
        return false;
    }

    llama_batch batch = llama_batch_init((int32_t) t.ids.size(), 0, 1);
    batch.n_tokens = (int32_t) t.ids.size();
    for (size_t i = 0; i < t.ids.size(); i++) {
        batch.token[i]     = t.ids[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]    = 0;
    }
    for (int slot : t.slots) {
        if (slot < 0 || slot >= (int) t.ids.size()) {
            llama_batch_free(batch);
            err = "answer slot outside the sequence";
            return false;
        }
        batch.logits[slot] = 1;
    }

    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        err = "llama_decode failed with " + std::to_string(rc);
        return false;
    }

    out.clear();
    for (size_t i = 0; i < t.slots.size(); i++) {
        const int n = n_options[i];
        if (n < 1 || n > (int) letter_ids.size()) {
            err = "option count " + std::to_string(n) + " is outside the label set";
            return false;
        }
        const float * lg = llama_get_logits_ith(ctx, t.slots[i]);
        if (!lg) { err = "no logits at an answer slot"; return false; }

        out.push_back(answer_from_logits(lg, letter_ids, n));
    }
    return true;
}

} // namespace system_one
