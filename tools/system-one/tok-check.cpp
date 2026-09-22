// tok-check: does llama.cpp's tokenizer reproduce the reference implementation's ids?
//
// Renders the model's own `system_one.template` (Jinja, from the GGUF) for every state in
// a golden dump, then compares against the reference:
//   1. rendered text vs the golden's prompt_text   (is the template + our renderer right?)
//   2. segment-wise ids vs the golden's input_ids  (does our tokenizer agree per segment?)
//   3. single-pass ids vs the golden's input_ids   (what does the segment split cost?)
//
// usage: llama-system-one-tok-check <model.gguf> <golden.json> [--limit N] [--template-file F]
//
// --template-file overrides the model's template, which is the request-level escape hatch
// used for experiments; without it the model's own GGUF template is authoritative.

#include "system-one.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace system_one;

static kind kind_of(const std::string & s) {
    if (s == "choice") return kind::choice;
    if (s == "score")  return kind::score;
    return kind::noul;
}

static std::string piece(const llama_vocab * vocab, llama_token id) {
    const char * t = llama_vocab_get_text(vocab, id);
    return t ? std::string(t) : std::string("<null>");
}

// index of the first differing element, or -1 when equal
static int first_diff(const std::vector<llama_token> & a, const std::vector<llama_token> & b) {
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) return (int) i;
    return a.size() == b.size() ? -1 : (int) n;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <golden.json> [--limit N]\n", argv[0]);
        return 1;
    }
    int limit = -1;
    std::string template_file;
    for (int i = 3; i < argc - 1; i++) {
        if      (strcmp(argv[i], "--limit")         == 0) limit = atoi(argv[i+1]);
        else if (strcmp(argv[i], "--template-file") == 0) template_file = argv[i+1];
    }

    json golden;
    { std::ifstream f(argv[2]); if (!f) { fprintf(stderr, "cannot open %s\n", argv[2]); return 1; } f >> golden; }
    const auto & items = golden.at("items");

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;                    // tokenizer parity needs no weights
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", argv[1]); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    so_config   cfg;
    std::string err;
    if (!so_config::from_model(model, cfg, err)) {
        fprintf(stderr, "error: %s\n", err.c_str());
        return 1;
    }
    if (!template_file.empty()) {
        std::ifstream f(template_file);
        if (!f) { fprintf(stderr, "cannot open %s\n", template_file.c_str()); return 1; }
        std::string src((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        while (!src.empty() && (src.back() == '\n' || src.back() == '\r')) src.pop_back();
        cfg.template_src = src;
        printf("template overridden from %s\n", template_file.c_str());
    }
    printf("template: %zu bytes, sep=U+%04X, slot=%s, readout=%s, bos=%d, max_state_tokens=%d\n",
           cfg.template_src.size(), (unsigned) (unsigned char) cfg.segment_separator[0],
           cfg.slot_rule.c_str(), cfg.readout.c_str(), (int) cfg.add_bos, cfg.max_state_tokens);
    printf("calibration: T=%.4f folded=%d\n", cfg.calibration_temperature, (int) cfg.calibration_folded);

    std::vector<llama_token> letter_ids;
    printf("letters: %s\n", letter_tokens(vocab, cfg.letters, letter_ids)
           ? "all are single tokens" : "MISSING single-token pieces");

    const size_t n = (limit > 0 && (size_t) limit < items.size()) ? (size_t) limit : items.size();
    int ok_text = 0, ok_seg = 0, ok_whole = 0, ok_slots = 0;
    int shown = 0;
    // how far does the single-pass tokenization drift? (the segment split is not free)
    int whole_len_delta_min = 1 << 30, whole_len_delta_max = -(1 << 30);
    double whole_len_delta_sum = 0;
    int whole_first_diff_min = 1 << 30, whole_first_diff_max = -1;

    for (size_t ii = 0; ii < n; ii++) {
        const auto & it = items[ii];
        std::vector<question> qs;
        std::vector<int>      slots_ref;
        for (const auto & q : it.at("questions")) {
            question o;
            o.k    = kind_of(q.value("kind", "noul"));
            o.text = q.at("text").get<std::string>();
            o.options = q.at("options").get<std::vector<std::string>>();
            if (q.contains("descs") && !q.at("descs").is_null()) {
                o.descs = q.at("descs").get<std::vector<std::string>>();
            }
            qs.push_back(std::move(o));
            slots_ref.push_back(q.at("slot_pos").get<int>());
        }

        const std::string state = truncate_state(vocab, cfg, it.at("state").get<std::string>());
        std::vector<std::string> segs;
        if (!render_segments(cfg, state, qs, segs, err)) {
            fprintf(stderr, "error on item %zu: %s\n", ii, err.c_str());
            return 1;
        }
        std::string built;
        for (const auto & seg : segs) built += seg;

        // the golden's prompt_text renders the BOS token; compare the rest
        std::string ref_text = it.at("prompt_text").get<std::string>();
        const std::string bos_piece = piece(vocab, llama_vocab_bos(vocab));
        if (ref_text.rfind(bos_piece, 0) == 0) ref_text.erase(0, bos_piece.size());
        const bool text_eq = (built == ref_text);
        ok_text += text_eq;

        const auto ids_ref = it.at("input_ids").get<std::vector<llama_token>>();
        tokenized t_seg;
        if (!tokenize_segments(vocab, cfg, segs, qs.size(), t_seg, err)) {
            fprintf(stderr, "error on item %zu: %s\n", ii, err.c_str());
            return 1;
        }
        const auto t_whole = tokenize_whole(vocab, built, cfg.add_bos);

        const int d_seg   = first_diff(t_seg.ids, ids_ref);
        const int d_whole = first_diff(t_whole,   ids_ref);
        ok_seg   += (d_seg   < 0);
        ok_whole += (d_whole < 0);
        ok_slots += (t_seg.slots == slots_ref);

        if (d_whole >= 0) {
            const int dl = (int) t_whole.size() - (int) ids_ref.size();
            whole_len_delta_min = std::min(whole_len_delta_min, dl);
            whole_len_delta_max = std::max(whole_len_delta_max, dl);
            whole_len_delta_sum += dl;
            whole_first_diff_min = std::min(whole_first_diff_min, d_whole);
            whole_first_diff_max = std::max(whole_first_diff_max, d_whole);
        }

        if ((!text_eq || d_seg >= 0) && shown < 3) {
            shown++;
            printf("\nitem %zu: text_eq=%d  seg_ids=%s  whole_ids=%s  slots=%s\n", ii, (int) text_eq,
                   d_seg   < 0 ? "match" : "differ", d_whole < 0 ? "match" : "differ",
                   t_seg.slots == slots_ref ? "match" : "differ");
            if (!text_eq) {
                size_t i = 0;
                while (i < built.size() && i < ref_text.size() && built[i] == ref_text[i]) i++;
                printf("  text diverges at byte %zu\n    built: %.60s\n    ref:   %.60s\n",
                       i, built.c_str() + i, ref_text.c_str() + i);
            }
            if (d_seg >= 0) {
                printf("  seg ids diverge at %d (len %zu vs ref %zu)\n", d_seg, t_seg.ids.size(), ids_ref.size());
                for (int k = std::max(0, d_seg - 2); k < std::min((int) t_seg.ids.size(), d_seg + 3); k++) {
                    printf("    ours[%d] = %6d %-16s", k, t_seg.ids[k], piece(vocab, t_seg.ids[k]).c_str());
                    if (k < (int) ids_ref.size()) printf("  ref = %6d %s", ids_ref[k], piece(vocab, ids_ref[k]).c_str());
                    printf("\n");
                }
            }
            if (t_seg.slots != slots_ref) {
                printf("  slots ours:");
                for (int s : t_seg.slots) printf(" %d", s);
                printf("\n  slots ref :");
                for (int s : slots_ref)   printf(" %d", s);
                printf("\n");
            }
        }
    }

    printf("\nover %zu states:\n", n);
    printf("  rendered text == golden prompt_text: %d / %zu\n", ok_text,  n);
    printf("  segment-wise ids == golden ids     : %d / %zu\n", ok_seg,   n);
    printf("  single-pass  ids == golden ids     : %d / %zu\n", ok_whole, n);
    printf("  answer slots == golden slot_pos    : %d / %zu\n", ok_slots, n);
    if (ok_whole < (int) n) {
        printf("  single-pass drift: length delta %+d..%+d (mean %+.2f), first divergence at token %d..%d\n",
               whole_len_delta_min, whole_len_delta_max, whole_len_delta_sum / (double) (n - ok_whole),
               whole_first_diff_min, whole_first_diff_max);
    }

    llama_model_free(model);
    llama_backend_free();
    return ok_seg == (int) n ? 0 : 2;
}
