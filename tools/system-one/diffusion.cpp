// llama-system-one-diffusion: answer a System One request on a DiffusionGemma checkpoint.
//
// This is a separate command on purpose. DiffusionGemma is an encoder-decoder -- a causal
// encoder over the prompt, then a canvas of `diffusion.canvas_length` positions denoised
// bidirectionally against the encoder's K,V -- and its arch is not upstream. The ordinary
// `/v1/systemone` route issues one plain llama_decode, which lands in the arch's UNIFIED
// branch, where the region split is positional:
//
//     P = n_tokens - canvas_length
//
// so the boundary falls wherever the prompt happens to end, usually inside the state, and an
// answer mask can end up on the causal-prompt side of it. Nothing there is read from the
// region the model denoises. Rather than teach the shared route about a phase that only one
// unmerged arch has, this drives the model's own two-phase path and keeps it out of the way.
//
// The layout follows the model's own use (razorback16/openjev): the state and the question
// text are the prompt, and the canvas holds one labelled, masked answer slot per question --
// `q1: <mask>`. The template supplies both, separated by a segment that is exactly the canvas
// marker; everything before it is the prompt, everything after is the canvas.
//
//   --mode prefill   PREFILL the prompt, DECODE the canvas (default; the cached path)
//   --mode unified   one [prompt | canvas] batch, which is only correct when the canvas is
//                    exactly canvas_length tokens -- so it always is here
//   --requests F     answer a JSONL of requests instead of scoring a golden, and write the answers
//                    as JSONL to --out. Same contract as llama-system-one-batch, so the evaluation
//                    suite can drive this arch through the same path as every other one.
//   --dump-ids PFX   write PFX.prompt.i32 and PFX.canvas.i32 (raw little-endian int32) for the
//                    first item, which is what llama-diffusion-gemma-eval takes -- so the same
//                    layout can be run through the arch's own harness, DG_CACHED=0 and 1.
//   --pad            pad the canvas to canvas_length with the mask token. Implied by `unified`
//                    and `both`, which split on the hparam and cannot do without it. It is NOT
//                    the default for `prefill`, because padding is not neutral: measured against
//                    HF on the same prompt, padding a 30-token canvas out to 256 moves the
//                    model's own answer probabilities by up to .386 (argmax unchanged). DECODE
//                    takes C from the batch, so the rendered length is the faithful one.
//   --dry-run        load the vocabulary only and print the layout: how the template split
//                    into prompt and canvas, and where the answer slots landed. No weights,
//                    so it is the tokenizer check for this command.
//   --control        with --mode both, also run each path a second time and report how far it
//                    differs from itself. A path that does not reproduce its own numbers makes
//                    the cross-path figure meaningless, so this is measured rather than assumed.
//   --mode both      run both and report the largest disagreement at the answer slots. The
//                    arch specifies they agree to f32 round-off, so this is a real check on
//                    the layout without needing a 26B HF reference.
//
// usage: llama-system-one-diffusion <model.gguf> <golden.json> --template-file <f>
//        [--limit N] [--threads N] [--mode prefill|unified|both] [--ngl N] [--dump out.json]

#include "system-one.h"

#include "llama.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ordered_json: the request's key order is part of the question (see batch.cpp)
using json = nlohmann::ordered_json;
using clk  = std::chrono::steady_clock;

static double ms_since(clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

enum pkv_phase { PKV_UNIFIED = 0, PKV_PREFILL = 1, PKV_DECODE = 2 };

static system_one_kind kind_of(const std::string & s) {
    if (s == "choice") return SYSTEM_ONE_KIND_CHOICE;
    if (s == "score")  return SYSTEM_ONE_KIND_SCORE;
    return SYSTEM_ONE_KIND_NOUL;
}

// The prompt and the canvas, already tokenized, with the answer slots given as offsets into
// the canvas. The canvas is padded to canvas_length with the mask token -- an unfilled canvas
// position is exactly what a mask is, and both modes need the same one to be comparable.
struct layout {
    std::vector<llama_token> prompt;
    std::vector<llama_token> canvas;
    std::vector<int>         slots;    // relative to the canvas
    size_t                   n_canvas_used = 0;
};

static layout build_layout(const llama_vocab * vocab, const system_one_params & cfg,
                           const std::string & state, const std::vector<system_one_question> & qs,
                           const std::string & marker, int canvas_length) {
    const auto segments = system_one_render_segments(cfg, state, qs);

    size_t at = segments.size();
    for (size_t i = 0; i < segments.size(); i++) {
        if (segments[i] == marker) { at = i; break; }
    }
    if (at == segments.size()) {
        throw std::invalid_argument("the template never wrote the canvas marker \"" + marker +
              "\" as a segment of its own; this command needs a canvas template, not the "
              "inline-mask one");
    }

    const std::vector<std::string> prompt_segments(segments.begin(), segments.begin() + at);
    const std::vector<std::string> canvas_segments(segments.begin() + at + 1, segments.end());
    if (canvas_segments.empty()) {
        throw std::invalid_argument("the canvas is empty: nothing follows the marker");
    }

    layout out;
    out.prompt = system_one_tokenize_segments(vocab, cfg, prompt_segments, 0).ids;

    // slots relative to the canvas, because DECODE indexes logits within its own batch
    const system_one_tokenized c = system_one_resolve_question_slots(vocab, cfg, canvas_segments, 0);
    out.canvas        = c.ids;
    out.slots         = c.slots;
    out.n_canvas_used = c.ids.size();

    if (canvas_length < 0) {
        return out;   // --no-pad: DECODE takes C from the batch, so the rendered length stands
    }
    if ((int) out.canvas.size() > canvas_length) {
        throw std::invalid_argument("the canvas needs " + std::to_string(out.canvas.size()) +
              " tokens but diffusion.canvas_length is " + std::to_string(canvas_length) +
              "; ask fewer questions per request");
    }
    out.canvas.resize(canvas_length, cfg.mask_token);
    return out;
}

// One decode of [prompt | canvas] with no phase, which the arch splits at n_tokens - canvas_length.
// Correct only because the canvas is padded to exactly that, which is why the padding is not optional.
static std::vector<std::vector<float>> run_unified(llama_context * ctx, llama_model * model,
                                                   const layout & L, int n_vocab) {
    llama_diffusion_set_phase(model, PKV_UNIFIED, 0, 0);

    const int P = (int) L.prompt.size();
    const int N = P + (int) L.canvas.size();
    llama_batch b = llama_batch_init(N, 0, 1);
    b.n_tokens = N;
    for (int i = 0; i < N; i++) {
        b.token[i]     = i < P ? L.prompt[i] : L.canvas[i - P];
        b.pos[i]       = i;
        b.n_seq_id[i]  = 1;
        b.seq_id[i][0] = 0;
        b.logits[i]    = 0;
    }
    for (int s : L.slots) { b.logits[P + s] = 1; }

    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) { throw std::runtime_error("unified decode failed with " + std::to_string(rc)); }

    std::vector<std::vector<float>> rows;
    for (int s : L.slots) {
        const float * r = llama_get_logits_ith(ctx, P + s);
        if (!r) { throw std::runtime_error("no logits at a unified answer slot"); }
        rows.emplace_back(r, r + n_vocab);
    }
    return rows;
}

// The model's own path: the prompt written causally into the K,V store, then the canvas read
// bidirectionally against it. Here the canvas length is the decode batch, not the hparam.
static std::vector<std::vector<float>> run_prefill_decode(llama_context * ctx, llama_model * model,
                                                          const layout & L, int n_vocab) {
    const int P = (int) L.prompt.size();
    const int C = (int) L.canvas.size();

    llama_diffusion_set_phase(model, PKV_PREFILL, P, 0);
    {
        llama_batch b = llama_batch_init(P, 0, 1);
        b.n_tokens = P;
        for (int i = 0; i < P; i++) {
            b.token[i]     = L.prompt[i];
            b.pos[i]       = i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = i == P - 1;   // unused, but n_outputs must not be zero
        }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { throw std::runtime_error("PREFILL failed with " + std::to_string(rc)); }
    }

    llama_diffusion_set_phase(model, PKV_DECODE, P, 0);
    std::vector<std::vector<float>> rows;
    {
        llama_batch b = llama_batch_init(C, 0, 1);
        b.n_tokens = C;
        for (int i = 0; i < C; i++) {
            b.token[i]     = L.canvas[i];
            b.pos[i]       = P + i;
            b.n_seq_id[i]  = 1;
            b.seq_id[i][0] = 0;
            b.logits[i]    = 0;
        }
        for (int s : L.slots) { b.logits[s] = 1; }
        const int rc = llama_decode(ctx, b);
        llama_batch_free(b);
        if (rc != 0) { throw std::runtime_error("DECODE failed with " + std::to_string(rc)); }

        for (int s : L.slots) {
            const float * r = llama_get_logits_ith(ctx, s);
            if (!r) { throw std::runtime_error("no logits at a decoded answer slot"); }
            rows.emplace_back(r, r + n_vocab);
        }
    }
    llama_diffusion_set_phase(model, PKV_UNIFIED, 0, 0);   // leave the model as it was found
    return rows;
}

static int run(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <golden.json> --template-file <f> [--limit N] "
                        "[--threads N] [--mode prefill|unified|both] [--ngl N] [--dump out.json]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    std::string golden_path;   // positional, and only needed when scoring rather than answering

    int         limit = -1, ngl = 0;
    int         nthreads = std::max(1u, std::thread::hardware_concurrency() / 2);
    std::string mode = "prefill", tmpl_path, dump_path, ids_prefix, marker = "<|canvas|>";
    std::string req_path, out_path;
    bool dry_run = false, control = false, pad = false;
    for (int i = 3; i < argc; i++) {
        if (strcmp(argv[i], "--dry-run") == 0) { dry_run = true; }
        if (strcmp(argv[i], "--pad")     == 0) { pad     = true; }
        if (strcmp(argv[i], "--control") == 0) { control = true; }
    }
    for (int i = 2; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (golden_path.empty()) { golden_path = argv[i]; }
            continue;
        }
        if (i + 1 >= argc) { break; }
        if      (strcmp(argv[i], "--limit")         == 0) limit     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--threads")       == 0) nthreads  = atoi(argv[++i]);
        else if (strcmp(argv[i], "--ngl")           == 0) ngl       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--mode")          == 0) mode      = argv[++i];
        else if (strcmp(argv[i], "--template-file") == 0) tmpl_path = argv[++i];
        else if (strcmp(argv[i], "--dump")          == 0) dump_path = argv[++i];
        else if (strcmp(argv[i], "--canvas-marker") == 0) marker    = argv[++i];
        else if (strcmp(argv[i], "--dump-ids")      == 0) ids_prefix = argv[++i];
        else if (strcmp(argv[i], "--requests")      == 0) req_path   = argv[++i];
        else if (strcmp(argv[i], "--out")           == 0) out_path   = argv[++i];
    }
    if (mode != "prefill") {
        pad = true;   // UNIFIED splits on the hparam, so it has no choice
    }
    if (req_path.empty() && golden_path.empty()) {
        fprintf(stderr, "pass a golden to score, or --requests FILE to answer\n");
        return 1;
    }
    if (mode != "prefill" && mode != "unified" && mode != "both") {
        fprintf(stderr, "--mode must be prefill, unified or both\n");
        return 1;
    }

    // A request and a golden item hold the same thing in different clothes, so the wire format is
    // reshaped into the item the loop below already walks and only the output differs.
    json items = json::array();
    std::vector<json>                     req_ids;
    std::vector<std::vector<std::string>> req_keys;
    const bool answering = !req_path.empty();

    if (answering) {
        std::ifstream f(req_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", req_path.c_str()); return 1; }
        std::string line;
        size_t n = 0;
        while (std::getline(f, line)) {
            if (line.empty()) { continue; }
            const json r = json::parse(line);
            req_ids.push_back(r.contains("id") ? r.at("id") : json((int64_t) n));
            json item = { {"state", r.at("state").is_string() ? r.at("state").get<std::string>()
                                                              : r.at("state").dump()},
                          {"questions", json::array()} };
            std::vector<std::string> keys;
            for (const auto & e : r.at("questions").items()) {
                const json & q = e.value();
                const std::string kind = q.at("type").get<std::string>();
                json out = { {"kind", kind}, {"text", q.value("instructions", std::string())} };
                json opts = json::array(), descs = json::array();
                if (kind == "noul") {
                    opts = json::array({"no", "yes"});
                } else if (q.contains("criteria") && q.at("criteria").is_object()) {
                    for (const auto & c : q.at("criteria").items()) {
                        opts.push_back(c.key());
                        descs.push_back(c.value().is_string() ? c.value().get<std::string>() : "");
                    }
                } else if (q.contains("criteria")) {
                    for (const auto & v : q.at("criteria")) { opts.push_back(v); descs.push_back(""); }
                }
                out["options"] = opts;
                if (!descs.empty()) { out["descs"] = descs; }
                item["questions"].push_back(out);
                keys.push_back(e.key());
            }
            req_keys.push_back(std::move(keys));
            items.push_back(std::move(item));
            n++;
        }
        if (items.empty()) { fprintf(stderr, "%s holds no requests\n", req_path.c_str()); return 1; }
    } else {
        json g;
        std::ifstream f(golden_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", golden_path.c_str()); return 1; }
        f >> g;
        items = g.at("items");
    }
    if (limit > 0 && (int) items.size() > limit) { items.erase(items.begin() + limit, items.end()); }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    mp.vocab_only   = dry_run;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    char buf[32] = {};
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", buf, sizeof(buf)) < 0) {
        fprintf(stderr, "%s has no diffusion.canvas_length: this command is for DiffusionGemma\n",
                model_path.c_str());
        return 1;
    }
    const int canvas_length = atoi(buf);

    system_one_params cfg;
    try {
        cfg = system_one_params_from_model(model);
    } catch (const std::exception & e) {
        if (tmpl_path.empty()) { fprintf(stderr, "%s\n", e.what()); return 1; }
    }
    if (!tmpl_path.empty()) {
        std::ifstream f(tmpl_path);
        if (!f) { fprintf(stderr, "cannot open %s\n", tmpl_path.c_str()); return 1; }
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        while (!s.empty() && s.back() == '\n') { s.pop_back(); }
        cfg.template_src = s;
    }
    if (cfg.readout != SYSTEM_ONE_READOUT_MASKED_SLOT) {
        fprintf(stderr, "this model derives readout=%s; the canvas is read at mask tokens\n",
                system_one_readout_name(cfg.readout));
        return 1;
    }

    // The whole sequence has to be one ubatch: UNIFIED splits on the ubatch's own length, so a
    // prompt cut in two would have each half compute its own region boundary.
    const int n_ctx_need = 8192;
    llama_context * ctx = nullptr;
    if (!dry_run) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = n_ctx_need;
    cp.n_batch     = n_ctx_need;
    cp.n_ubatch    = n_ctx_need;
    cp.n_threads   = nthreads;
    cp.n_threads_batch = nthreads;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create a context\n"); return 1; }
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    const std::vector<llama_token> label_ids =
        system_one_label_tokens(vocab, cfg.labels, cfg.labels_are_default);

    std::ofstream answers_out;
    if (answering) {
        if (out_path.empty()) { fprintf(stderr, "--requests needs --out\n"); return 1; }
        answers_out.open(out_path);
        if (!answers_out) { fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    }

    printf("model: %s\ncanvas_length: %d   mode: %s   threads: %d   marker: %s\n",
           model_path.c_str(), canvas_length, mode.c_str(), nthreads, marker.c_str());

    int n_q = 0, n_correct = 0, n_scored = 0, n_slots_in_canvas = 0;
    double worst_mode_gap = 0.0, worst_label_logit = 0.0, worst_label_prob = 0.0;
    double self_prefill = 0.0, self_unified = 0.0;
    int    n_mode_pairs = 0, n_mode_argmax_same = 0;
    json dump = json::array();

    for (size_t it = 0; it < items.size(); it++) {
        const auto & item = items[it];
        std::vector<system_one_question> qs;
        for (const auto & q : item.at("questions")) {
            system_one_question sq;
            sq.kind = kind_of(q.at("kind").get<std::string>());
            sq.text = q.at("text").get<std::string>();
            sq.options = q.at("options").get<std::vector<std::string>>();
            if (q.contains("descs") && !q.at("descs").is_null()) {
                sq.descs = q.at("descs").get<std::vector<std::string>>();
            }
            qs.push_back(std::move(sq));
        }

        layout L;
        try {
            L = build_layout(vocab, cfg, item.at("state").get<std::string>(), qs, marker,
                             pad ? canvas_length : -1);
        } catch (const std::exception & e) {
            fprintf(stderr, "item %zu: %s\n", it, e.what());
            return 1;
        }
        if (L.slots.size() != qs.size()) {
            fprintf(stderr, "item %zu: %zu answer slots for %zu questions\n", it, L.slots.size(), qs.size());
            return 1;
        }
        n_slots_in_canvas += (int) L.slots.size();   // by construction: they are canvas offsets

        if (!ids_prefix.empty() && it == 0) {
            const auto write_i32 = [](const std::string & path, const std::vector<llama_token> & v) {
                std::ofstream f(path, std::ios::binary);
                for (const llama_token t : v) {
                    const int32_t x = (int32_t) t;
                    f.write((const char *) &x, sizeof(x));
                }
            };
            write_i32(ids_prefix + ".prompt.i32", L.prompt);
            write_i32(ids_prefix + ".canvas.i32", L.canvas);
            printf("ids: %s.prompt.i32 (%zu) %s.canvas.i32 (%zu)\n",
                   ids_prefix.c_str(), L.prompt.size(), ids_prefix.c_str(), L.canvas.size());
        }

        if (dry_run) {
            printf("item %3zu: prompt %4zu   canvas %3zu/%d   slots", it, L.prompt.size(),
                   L.n_canvas_used, canvas_length);
            for (int s : L.slots) { printf(" %d", s); }
            printf("   (%zu questions)\n", qs.size());
            n_q += (int) qs.size();
            continue;
        }

        const auto t_req = clk::now();
        std::vector<std::vector<float>> rows;
        if (mode == "unified") {
            rows = run_unified(ctx, model, L, n_vocab);
        } else {
            rows = run_prefill_decode(ctx, model, L, n_vocab);
            if (mode == "both") {
                const auto u = run_unified(ctx, model, L, n_vocab);
                if (control) {
                    const auto p2 = run_prefill_decode(ctx, model, L, n_vocab);
                    const auto u2 = run_unified(ctx, model, L, n_vocab);
                    for (size_t r = 0; r < rows.size(); r++) {
                        const int k = (int) qs[r].options.size();
                        const auto a  = system_one_answer_from_logits(rows[r].data(), label_ids, k);
                        const auto a2 = system_one_answer_from_logits(p2[r].data(),   label_ids, k);
                        const auto b  = system_one_answer_from_logits(u[r].data(),    label_ids, k);
                        const auto b2 = system_one_answer_from_logits(u2[r].data(),   label_ids, k);
                        for (int o = 0; o < k; o++) {
                            self_prefill = std::max(self_prefill, (double) std::fabs(a.probs[o] - a2.probs[o]));
                            self_unified = std::max(self_unified, (double) std::fabs(b.probs[o] - b2.probs[o]));
                        }
                    }
                }
                for (size_t r = 0; r < rows.size(); r++) {
                    for (int v = 0; v < n_vocab; v++) {
                        worst_mode_gap = std::max(worst_mode_gap, (double) std::fabs(rows[r][v] - u[r][v]));
                    }
                    // the whole vocabulary is not what the readout uses: a big gap on a token no
                    // answer can ever be is not a layout problem. Judge on this question's labels.
                    const int k = (int) qs[r].options.size();
                    const auto a = system_one_answer_from_logits(rows[r].data(), label_ids, k);
                    const auto b = system_one_answer_from_logits(u[r].data(),    label_ids, k);
                    for (int o = 0; o < k; o++) {
                        worst_label_logit = std::max(worst_label_logit, (double) std::fabs(a.logits[o] - b.logits[o]));
                        worst_label_prob  = std::max(worst_label_prob,  (double) std::fabs(a.probs[o]  - b.probs[o]));
                    }
                    n_mode_pairs++;
                    n_mode_argmax_same += a.choice == b.choice;
                }
            }
        }

        if (answering) {
            json ja;
            for (size_t qi = 0; qi < qs.size(); qi++) {
                const auto a = system_one_answer_from_logits(rows[qi].data(), label_ids,
                                                            (int) qs[qi].options.size());
                json one;
                switch (qs[qi].kind) {
                    case SYSTEM_ONE_KIND_NOUL:
                        one["noul"] = a.probs.size() > 1 ? a.probs[1] : 0.0f;
                        break;
                    case SYSTEM_ONE_KIND_CHOICE: {
                        one["choice"] = qs[qi].options[a.choice];
                        json pr;
                        for (size_t o = 0; o < qs[qi].options.size(); o++) {
                            pr[qs[qi].options[o]] = a.probs[o];
                        }
                        one["probabilities"] = pr;
                        break;
                    }
                    case SYSTEM_ONE_KIND_SCORE:
                        one["score"] = system_one_score_expectation(a);
                        one["probabilities"] = a.probs;
                        break;
                }
                one["confidence"] = a.confidence;
                one["logits"]     = a.logits;
                ja[req_keys[it][qi]] = one;
            }
            answers_out << json{{"id", req_ids[it]}, {"answers", ja}, {"ms", ms_since(t_req)},
                                {"n_tokens", (int) (L.prompt.size() + L.canvas.size())},
                                {"n_sequences", 1}}.dump() << "\n";
            continue;
        }

        json ditem = { {"item", it}, {"n_prompt", L.prompt.size()}, {"n_canvas_used", L.n_canvas_used},
                       {"slots", L.slots}, {"answers", json::array()} };
        for (size_t qi = 0; qi < qs.size(); qi++) {
            const auto a = system_one_answer_from_logits(rows[qi].data(), label_ids,
                                                         (int) qs[qi].options.size());
            ditem["answers"].push_back({ {"choice", a.choice}, {"confidence", a.confidence}, {"probs", a.probs} });
            n_q++;
            const auto & q = item.at("questions")[qi];
            if (q.contains("gold") && !q.at("gold").is_null() && q.at("gold").get<int>() >= 0) {
                n_scored++;
                n_correct += a.choice == q.at("gold").get<int>();
            }
        }
        dump.push_back(std::move(ditem));

        if (it == 0) {
            printf("layout: prompt %zu tokens, canvas %zu used of %d (%s), slots at",
                   L.prompt.size(), L.n_canvas_used, canvas_length,
                   pad ? "padded to canvas_length" : "as rendered");
            for (int s : L.slots) { printf(" %d", s); }
            printf(" (canvas-relative; every one is inside the canvas by construction)\n");
        }
    }

    printf("\nitems %zu   questions %d   answer slots in the canvas %d/%d\n",
           items.size(), n_q, n_slots_in_canvas, n_q);
    if (n_scored) {
        printf("accuracy vs gold: %.4f (%d/%d scored)\n", (double) n_correct / n_scored, n_correct, n_scored);
    }
    if (mode == "both") {
        printf("unified vs prefill+decode, at the answer slots:\n");
        printf("  over this question's labels : |dlogit| %.3e   |dp| %.3e\n", worst_label_logit, worst_label_prob);
        printf("  over the whole vocabulary   : |dlogit| %.3e   (includes tokens no answer can be)\n", worst_mode_gap);
        printf("  argmax                      : %d/%d agree\n", n_mode_argmax_same, n_mode_pairs);
        if (control) {
            printf("  control, each path vs itself: prefill+decode |dp| %.3e   unified |dp| %.3e\n",
                   self_prefill, self_unified);
        }
        printf("verdict: %s\n", (worst_label_prob <= 1e-3 && n_mode_argmax_same == n_mode_pairs)
               ? "AGREE -- the two paths read the same answer, so the layout is the one the model expects"
               : "DISAGREE -- the paths do not read the same answer");
    }

    if (!dump_path.empty()) {
        json out = { {"model", model_path}, {"mode", mode}, {"canvas_length", canvas_length},
                     {"threads", nthreads}, {"items", dump} };
        if (mode == "both") {
            out["unified_vs_cached"] = { {"vocab_max_dlogit", worst_mode_gap},
                                         {"label_max_dlogit", worst_label_logit},
                                         {"label_max_dp", worst_label_prob},
                                         {"argmax_same", n_mode_argmax_same}, {"pairs", n_mode_pairs} };
        }
        std::ofstream f(dump_path);
        f << out.dump(2) << "\n";
        printf("dump: %s\n", dump_path.c_str());
    }

    if (ctx) { llama_free(ctx); }
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
