// e2e-check: the whole product path against a golden dump.
//
// text -> model's own system_one template -> segments -> ids -> one decode -> label
// logits at each answer slot -> probabilities. Nothing is taken from the reference except
// the state and the questions; in particular the ids are ours, not the golden's.
//
// Thresholds (agreed with the reference side):
//   prob  max |diff| <= 1e-3
//   logit max |diff| <= 1e-4   (only meaningful on an exact-GELU build)
//   argmax equal everywhere, except where the reference's own top-2 logit gap <= 1e-2,
//   which is a tie and is printed rather than failed.
//
// usage: llama-system-one-e2e-check <model.gguf> <golden.json>
//        [--limit N] [--threads N] [--kv f32|f16] [--fa on|off] [--dump out.json]

#include "system-one.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace system_one;

// Is the linked ggml computing GELU exactly in f32, or through the fp16 lookup table
// (ggml-cpu/vec.h: GGML_GELU_FP16)? The table is a systematic ~5e-4 relative error per
// activation and dominates any comparison against a HF fp32 reference.
static bool ggml_gelu_is_exact() {
    const int n = 1024;
    std::vector<float> xs(n);
    for (int i = 0; i < n; i++) xs[i] = -8.0f + 16.0f * (float) i / (float) (n - 1);

    const size_t mem = 4u * 1024 * 1024;
    std::vector<uint8_t> buf(mem);
    ggml_init_params ip = { mem, buf.data(), false };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x  = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * y  = ggml_gelu(ctx, x);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    memcpy(x->data, xs.data(), n * sizeof(float));
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    const float * out = (const float *) y->data;
    double max_abs = 0.0;
    for (int i = 0; i < n; i++) {
        const double k   = 0.79788456080286535587989211986876;
        const double ref = 0.5 * xs[i] * (1.0 + std::tanh(k * xs[i] * (1.0 + 0.044715 * xs[i] * xs[i])));
        max_abs = std::max(max_abs, std::fabs(out[i] - ref));
    }
    ggml_free(ctx);
    return max_abs < 1e-6;
}

static kind kind_of(const std::string & s) {
    if (s == "choice") return kind::choice;
    if (s == "score")  return kind::score;
    return kind::noul;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <golden.json> [--limit N] [--threads N] "
                        "[--kv f32|f16] [--fa on|off] [--dump out.json]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const std::string golden_path = argv[2];

    int  limit = -1, nthreads = 32;      // threads are pinned: they set the reduction order
    bool kv_f32 = true, use_fa = false;
    std::string dump_path;
    for (int i = 3; i < argc - 1; i++) {
        if      (strcmp(argv[i], "--limit")   == 0) limit    = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--threads") == 0) nthreads = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--kv")      == 0) kv_f32   = strcmp(argv[i+1], "f32") == 0;
        else if (strcmp(argv[i], "--fa")      == 0) use_fa   = strcmp(argv[i+1], "on")  == 0;
        else if (strcmp(argv[i], "--dump")    == 0) dump_path = argv[i+1];
    }

    json golden;
    { std::ifstream f(golden_path); if (!f) { fprintf(stderr, "cannot open %s\n", golden_path.c_str()); return 1; } f >> golden; }
    const auto & items = golden.at("items");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    so_config   cfg;
    std::string err;
    if (!so_config::from_model(model, cfg, err)) { fprintf(stderr, "error: %s\n", err.c_str()); return 1; }

    std::vector<llama_token> label_ids;
    if (!label_tokens(vocab, cfg.labels, label_ids)) {
        fprintf(stderr, "error: a label in system_one.labels is not a single token\n");
        return 1;
    }

    // longest prompt decides the context size; render everything once up front
    struct prepared { tokenized t; std::vector<int> n_options; const json * qs; };
    std::vector<prepared> prep;
    const size_t n_items = (limit > 0 && (size_t) limit < items.size()) ? (size_t) limit : items.size();
    size_t max_len = 0;
    for (size_t ii = 0; ii < n_items; ii++) {
        const auto & it = items[ii];
        std::vector<question> qs;
        std::vector<int>      n_options;
        for (const auto & q : it.at("questions")) {
            question o;
            o.k       = kind_of(q.value("kind", "noul"));
            o.text    = q.at("text").get<std::string>();
            o.options = q.at("options").get<std::vector<std::string>>();
            if (q.contains("descs") && !q.at("descs").is_null()) o.descs = q.at("descs").get<std::vector<std::string>>();
            n_options.push_back((int) q.at("logits").size());
            qs.push_back(std::move(o));
        }

        const std::string state = it.at("state").get<std::string>();
        std::vector<std::string> segs;
        if (!render_segments(cfg, state, qs, segs, err)) { fprintf(stderr, "item %zu: %s\n", ii, err.c_str()); return 1; }

        prepared p;
        if (!tokenize_segments(vocab, cfg, segs, qs.size(), p.t, err)) { fprintf(stderr, "item %zu: %s\n", ii, err.c_str()); return 1; }
        p.n_options = std::move(n_options);
        p.qs        = &it.at("questions");
        max_len     = std::max(max_len, p.t.ids.size());
        prep.push_back(std::move(p));
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = max_len + 8;
    cp.n_batch         = max_len + 8;
    cp.n_ubatch        = max_len + 8;
    cp.n_seq_max       = 1;
    cp.n_threads       = nthreads;
    cp.n_threads_batch = nthreads;
    cp.flash_attn_type = use_fa ? LLAMA_FLASH_ATTN_TYPE_ENABLED : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.type_k          = kv_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.type_v          = kv_f32 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const bool gelu_exact = ggml_gelu_is_exact();
    printf("build: gelu=%s  kv=%s  flash_attn=%s  threads=%d\n",
           gelu_exact ? "exact-f32" : "fp16-table", kv_f32 ? "f32" : "f16", use_fa ? "on" : "off", nthreads);
    printf("readout: %s, %zu labels, attention from the model\n", cfg.readout.c_str(), cfg.labels.size());
    printf("ids come from our own tokenizer (the golden's input_ids are not used)\n");

    double sum_dl = 0, max_dl = 0, sum_dp = 0, max_dp = 0;
    long   n_cmp = 0, n_slots = 0, n_flip = 0, n_tie = 0;
    json   slots_out = json::array(), flips = json::array();

    for (size_t ii = 0; ii < prep.size(); ii++) {
        llama_memory_clear(llama_get_memory(ctx), true);

        std::vector<answer> answers;
        if (!read_slots(ctx, prep[ii].t, prep[ii].n_options, label_ids, answers, err)) {
            fprintf(stderr, "item %zu: %s\n", ii, err.c_str());
            return 1;
        }

        for (size_t qi = 0; qi < answers.size(); qi++) {
            const auto & q    = (*prep[ii].qs)[qi];
            const auto   lref = q.at("logits").get<std::vector<float>>();
            const auto   pref = q.contains("probs") ? q.at("probs").get<std::vector<float>>() : std::vector<float>();
            const auto & a    = answers[qi];

            for (size_t j = 0; j < lref.size(); j++) {
                const double dl = std::fabs(a.logits[j] - lref[j]);
                sum_dl += dl; max_dl = std::max(max_dl, dl); n_cmp++;
                if (!pref.empty()) {
                    const double dp = std::fabs(a.probs[j] - pref[j]);
                    sum_dp += dp; max_dp = std::max(max_dp, dp);
                }
            }

            const size_t am_ref = std::max_element(lref.begin(), lref.end()) - lref.begin();
            if ((size_t) a.choice != am_ref) {
                std::vector<float> s = lref;
                std::sort(s.begin(), s.end(), std::greater<float>());
                const double gap_ref = s.size() > 1 ? s[0] - s[1] : 0.0;
                const bool   tie     = gap_ref <= 1e-2;
                n_flip++; n_tie += tie;
                printf("  ARGMAX %s item %zu q %zu kind=%s: ref %zu vs ours %d (ref top-2 gap %.3e)\n",
                       tie ? "TIE " : "FLIP", ii, qi, q.value("kind", "?").c_str(), am_ref, a.choice, gap_ref);
                flips.push_back({{"item", ii}, {"q", qi}, {"kind", q.value("kind", "?")},
                                 {"argmax_ref", am_ref}, {"argmax_mine", a.choice},
                                 {"gap_ref", gap_ref}, {"tie", tie},
                                 {"logits_ref", lref}, {"logits_mine", a.logits}});
                fflush(stdout);
            }

            slots_out.push_back({{"item", ii}, {"q", qi}, {"slot_pos", prep[ii].t.slots[qi]},
                                 {"kind", q.value("kind", "?")}, {"probs_mine", a.probs},
                                 {"confidence", a.confidence}});
            n_slots++;
        }
        if ((ii + 1) % 50 == 0) { printf("  ... %zu/%zu items\n", ii + 1, prep.size()); fflush(stdout); }
    }

    printf("\ncompared %ld slots / %ld logit values over %zu items\n", n_slots, n_cmp, prep.size());
    printf("  logit |diff|: max %.3e  mean %.3e   (threshold 1e-4, exact-GELU builds)\n", max_dl, sum_dl / (double) n_cmp);
    printf("  prob  |diff|: max %.3e  mean %.3e   (threshold 1e-3)\n", max_dp, sum_dp / (double) n_cmp);
    printf("  argmax differences: %ld (%ld of them ties)\n", n_flip, n_tie);

    const bool pass = max_dp <= 1e-3 && (n_flip == n_tie) && (!gelu_exact || max_dl <= 1e-4);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    if (!dump_path.empty()) {
        json out = {
            {"model", model_path}, {"golden", golden_path},
            {"build", gelu_exact ? "exact-f32-gelu" : "stock-fp16-gelu-table"},
            {"gelu_exact", gelu_exact}, {"threads", nthreads},
            {"kv_f32", kv_f32}, {"flash_attn", use_fa},
            {"ids_source", "llama.cpp (system_one template)"},
            {"n_items", prep.size()}, {"n_slots", n_slots},
            {"logit_diff_max", max_dl}, {"logit_diff_mean", sum_dl / (double) n_cmp},
            {"prob_diff_max", max_dp},  {"prob_diff_mean", sum_dp / (double) n_cmp},
            {"argmax_differences", n_flip}, {"argmax_ties", n_tie},
            {"pass", pass}, {"flips", flips}, {"slots", slots_out},
        };
        std::ofstream f(dump_path);
        f << out.dump(1);
        printf("wrote %s\n", dump_path.c_str());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return pass ? 0 : 2;
}
