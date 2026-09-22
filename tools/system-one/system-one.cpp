#include "system-one.h"

#include "chat.h"                 // pulls in the jinja engine the chat templates use
#include "common.h"               // common_tokenize, string_split
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

static std::vector<std::string> split_array(const std::string & s) {
    // llama.cpp stringifies a kv array as [a, b, c]
    std::vector<std::string> out;
    for (auto & piece : string_split(s, ",")) {
        std::string cur;
        for (char c : piece) {
            if (c != '[' && c != ']' && c != '"' && c != ' ') cur += c;
        }
        if (!cur.empty()) out.push_back(cur);
    }
    return out;
}

bool so_config::from_model(const llama_model * model, so_config & out, std::string & err) {
    const char * tmpl = llama_model_chat_template(model, "system_one");
    if (tmpl == nullptr || tmpl[0] == '\0') {
        err = "model has no System One template (tokenizer.chat_template.system_one); "
              "convert it with its template, or pass one in the request";
        return false;
    }
    out.template_src = tmpl;

    std::string s;
    if (meta_str(model, "system_one.readout", s) && !s.empty()) out.readout = s;
    if (meta_str(model, "system_one.segment_separator", s) && !s.empty()) out.segment_separator = s;
    if (meta_str(model, "system_one.labels", s) && !s.empty()) out.labels = split_array(s);

    if (out.labels.empty()) {
        // the usual alphabet, so a plain letter format needs no key at all
        for (char c = 'A'; c <= 'Z'; c++) out.labels.push_back(std::string(1, c));
        for (char c = 'a'; c <= 'z'; c++) out.labels.push_back(std::string(1, c));
    }

    if (out.readout != "letter_slot" && out.readout != "masked_slot" && out.readout != "rank_head") {
        err = "unsupported system_one.readout: " + out.readout;
        return false;
    }

    // the rest is ordinary model metadata
    const llama_vocab * vocab = llama_model_get_vocab(model);

    auto piece = [&](llama_token id) {
        const char * t = id >= 0 ? llama_vocab_get_text(vocab, id) : nullptr;
        return t ? std::string(t) : std::string();
    };
    out.bos_text = piece(llama_vocab_bos(vocab));
    out.eos_text = piece(llama_vocab_eos(vocab));

    if (out.masked()) {
        out.mask_token = llama_vocab_mask(vocab);
        if (out.mask_token < 0) {
            err = "masked_slot needs the model's tokenizer.ggml.mask_token_id";
            return false;
        }
        const char * piece = llama_vocab_get_text(vocab, out.mask_token);
        out.mask_text = piece ? piece : "";
    }

    return true;
}

// ---------------------------------------------------------------- rendering

static const char * kind_name(kind k) {
    switch (k) {
        case kind::choice: return "choice";
        case kind::score:  return "score";
        default:           return "noul";
    }
}

static bool render_and_split(const so_config & cfg, const json & inp,
                             std::vector<std::string> & segments, std::string & err) {
    std::string rendered;
    try {
        jinja::lexer lexer;
        auto lexer_res = lexer.tokenize(cfg.template_src);
        jinja::program prog = jinja::parse_from_tokens(lexer_res);

        jinja::context ctx(lexer_res.source);
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
    segments.erase(std::remove_if(segments.begin(), segments.end(),
                                  [](const std::string & s) { return s.empty(); }),
                   segments.end());
    return true;
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
                {"label",  j < cfg.labels.size() ? cfg.labels[j] : std::string("?")},
                {"option", q.options[j]},
                {"desc",   has_desc ? json(q.descs[j]) : json(nullptr)},
            });
        }
        questions.push_back(json{
            {"k",       (int) i + 1},          // 1-based, as templates count questions
            {"kind",    kind_name(q.k)},
            {"text",    q.text},
            {"options", options},
        });
    }

    const json inp = json{
        {"state",     state},
        {"questions", questions},
        {"sep",       cfg.segment_separator},
        {"mask",      cfg.mask_text},      // empty unless the format places a mask token
        {"bos_token", cfg.bos_text},       // as a chat template gets them: the prompt says
        {"eos_token", cfg.eos_text},       // where they go, no flag decides it
    };

    if (!render_and_split(cfg, inp, segments, err)) return false;

    if (segments.size() < qs.size()) {
        err = "template produced " + std::to_string(segments.size()) + " segments for " +
              std::to_string(qs.size()) + " questions: the separator must precede every question block";
        return false;
    }
    return true;
}

bool render_pair_segments(const so_config & cfg,
                          const std::string & state,
                          const question & q,
                          size_t option_index,
                          std::vector<std::string> & segments,
                          std::string & err) {
    if (option_index >= q.options.size()) {
        err = "option index out of range";
        return false;
    }
    const bool has_desc = option_index < q.descs.size() && !q.descs[option_index].empty();

    const json inp = json{
        {"state", state},
        {"question", json{
            {"kind", kind_name(q.k)},
            {"text", q.text},
        }},
        {"option", json{
            {"label",  option_index < cfg.labels.size() ? cfg.labels[option_index] : std::string("?")},
            {"option", q.options[option_index]},
            {"desc",   has_desc ? json(q.descs[option_index]) : json(nullptr)},
            {"index",  (int) option_index},
        }},
        {"sep",       cfg.segment_separator},
        {"mask",      cfg.mask_text},
        {"bos_token", cfg.bos_text},
        {"eos_token", cfg.eos_text},
    };

    if (!render_and_split(cfg, inp, segments, err)) return false;
    if (segments.empty()) {
        err = "template rendered nothing for this question/option pair";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- tokenization

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

    // Special tokens -- BOS, a mask -- are written into the template as text and have to parse
    // back to their ids. HF's tokenizers split on added and special tokens regardless of
    // add_special_tokens, so parsing them is what matches the reference.
    const bool masked = cfg.masked();

    // the question blocks are the trailing segments
    const size_t first_question = segments.size() - n_questions;
    for (size_t i = 0; i < segments.size(); i++) {
        const auto ids = common_tokenize(vocab, segments[i], false, true);
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

bool label_tokens(const llama_vocab * vocab, const std::vector<std::string> & labels,
                  std::vector<llama_token> & out) {
    // Tokenize each label rather than matching vocab text: a label is read at the position
    // where the template would have written it, so it has to be what the tokenizer produces
    // there. This also makes leading spaces work -- a format whose answers are " A", " B"
    // reads the tokens a BPE vocab spells "ĠA", "ĠB", which no text comparison would find.
    out.assign(labels.size(), -1);
    for (size_t i = 0; i < labels.size(); i++) {
        const auto ids = common_tokenize(vocab, labels[i], false, false);
        if (ids.size() == 1) out[i] = ids[0];
    }
    return std::find(out.begin(), out.end(), -1) == out.end();
}

// ---------------------------------------------------------------- planning

bool build_plan(const so_config & cfg,
                const llama_vocab * vocab,
                const std::string & state,
                const std::vector<question> & qs_in,
                plan & out,
                std::string & err) {
    out = plan();
    if (qs_in.empty()) {
        err = "a request needs at least one question";
        return false;
    }

    // The two sides of a yes/no question are a property of the format, not of the caller:
    // every front end would otherwise have to know to spell them "no" and "yes". A caller
    // that wants them worded differently still may -- this only fills in the blank.
    std::vector<question> qs;
    qs.reserve(qs_in.size());
    for (auto q : qs_in) {
        if (q.k == kind::noul && q.options.empty()) {
            q.options = {"no", "yes"};
        }
        qs.push_back(std::move(q));
    }

    for (const auto & q : qs) {
        if (q.options.size() < 2) {
            err = "every question needs at least two options";
            return false;
        }
        out.n_options.push_back((int) q.options.size());
    }

    if (cfg.ranked()) {
        // the head scores a whole sequence, so each option gets one of its own
        out.rank_pooling = true;
        out.prefix_reuse = false;   // the sequences differ from the first token onwards

        for (size_t qi = 0; qi < qs.size(); qi++) {
            for (size_t oi = 0; oi < qs[qi].options.size(); oi++) {
                std::vector<std::string> segments;
                if (!render_pair_segments(cfg, state, qs[qi], oi, segments, err)) return false;

                plan::sequence seq;
                if (!tokenize_segments(vocab, cfg, segments, 0, seq.tok, err)) return false;
                seq.question = qi;
                seq.option   = oi;
                out.sequences.push_back(std::move(seq));
            }
        }
        return true;
    }

    // a slot readout answers every question from one sequence
    if (!label_tokens(vocab, cfg.labels, out.labels)) {
        err = "a label in system_one.labels is not a single token for this tokenizer";
        return false;
    }
    for (const auto & q : qs) {
        if (q.options.size() > cfg.labels.size()) {
            err = "a question has " + std::to_string(q.options.size()) + " options but this "
                  "readout has only " + std::to_string(cfg.labels.size()) + " labels; a model "
                  "with a scoring head (system_one.readout = rank_head) has no such limit";
            return false;
        }
    }

    std::vector<std::string> segments;
    if (!render_segments(cfg, state, qs, segments, err)) return false;

    plan::sequence seq;
    if (!tokenize_segments(vocab, cfg, segments, qs.size(), seq.tok, err)) return false;
    out.sequences.push_back(std::move(seq));

    // a bidirectional readout reads from inside the canvas, so no prefix survives a change
    out.prefix_reuse = !cfg.masked();
    return true;
}

bool answers_from_scores(const plan & p, const std::vector<float> & scores,
                         std::vector<answer> & out, std::string & err) {
    if (scores.size() != p.sequences.size()) {
        err = "expected one score per planned sequence";
        return false;
    }

    out.assign(p.n_options.size(), answer());
    std::vector<std::vector<float>> per_question(p.n_options.size());
    for (size_t i = 0; i < p.sequences.size(); i++) {
        const auto & seq = p.sequences[i];
        if (seq.question >= per_question.size()) {
            err = "a planned sequence refers to a question that is not in the request";
            return false;
        }
        per_question[seq.question].push_back(scores[i]);
    }

    for (size_t qi = 0; qi < per_question.size(); qi++) {
        if ((int) per_question[qi].size() != p.n_options[qi]) {
            err = "a question did not get one score per option";
            return false;
        }
        out[qi] = answer_from_scores(per_question[qi]);
    }
    return true;
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

answer answer_from_scores(const std::vector<float> & scores) {
    answer a;
    a.logits = scores;                 // the head's raw output, as the caller gets it
    a.probs  = softmax(scores);
    a.choice = (int) (std::max_element(a.probs.begin(), a.probs.end()) - a.probs.begin());

    // confidence = 1 - H(p)/ln K, as the Jev-compatible servers report it
    if (scores.size() > 1) {
        double h = 0.0;
        for (float p : a.probs) if (p > 0.0f) h -= (double) p * std::log((double) p);
        a.confidence = (float) (1.0 - h / std::log((double) scores.size()));
    } else {
        a.confidence = 1.0f;
    }
    return a;
}

answer answer_from_logits(const float * row, const std::vector<llama_token> & letter_ids, int n_options) {
    // the only difference between the two readouts is where the numbers come from
    std::vector<float> scores(n_options);
    for (int j = 0; j < n_options; j++) scores[j] = row[letter_ids[j]];
    return answer_from_scores(scores);
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
