// System One (Jev-compatible) typed decision readout.
//
// One forward pass over [state | question 1 .. question n], reading the label-token
// logits at each question's answer slot. No decoding, no grammar: the answer is a
// distribution over a small label set, which is what makes the output typed by
// construction.
//
// The prompt format is a property of the checkpoint, not of this code. It is a Jinja
// template stored in the GGUF as `system_one.template`, exactly like
// `tokenizer.chat_template`, and rendered with llama.cpp's own Jinja engine. There is
// deliberately NO built-in default: different System One lines (letter slots, a scoring
// head taking one option per sequence, per-option mask tokens) need different prompts,
// so a model without a template is an error rather than an assumption.
//
// Segment boundaries matter as much as the text. The reference implementation encodes
// the state and each question block separately and concatenates the ids, which differs
// from a single-pass tokenization at the seams. The template marks those seams with
// `system_one.template.segment_separator` (U+001E); this side splits on it and
// tokenizes each piece on its own.

#pragma once

#include "llama.h"

#include <string>
#include <vector>

namespace system_one {

enum class kind { noul, choice, score };

struct question {
    kind                     k = kind::noul;
    std::string              text;
    std::vector<std::string> options;   // noul: {"no","yes"}; score: levels in order
    std::vector<std::string> descs;     // empty, or one per option
};

// Everything the checkpoint declares about its own prompt and readout.
struct so_config {
    std::string template_src;                                   // system_one.template (required)
    std::string segment_separator = "\x1e";

    // where the answer sits in a question's segment, and what is read there:
    //   last_token_of_question_segment + letter_slot  the next-token distribution after "("
    //   mask_token_per_question        + masked_slot  the distribution at a mask token,
    //                                                 which needs bidirectional attention
    std::string slot_rule         = "last_token_of_question_segment";
    std::string readout           = "letter_slot";
    llama_token mask_token        = -1;     // system_one.mask_token_id
    std::string mask_text;                  // system_one.mask_token, as the template writes it
    std::string letters           = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

    std::vector<std::string> noul_options = {"no", "yes"};   // P(true) is the last one

    std::string tag_noul   = "yes/no";
    std::string tag_choice = "choice";
    std::string tag_score  = "score";

    int         max_state_tokens  = 0;    // 0 = never truncate
    std::string truncation;               // e.g. "head0.7_tail_marker"
    std::string truncation_marker;
    float       head_fraction     = 0.7f;
    int         tail_reserve      = 8;

    bool        add_bos           = true;
    float       calibration_temperature = 1.0f;   // provenance only: already in the weights
    bool        calibration_folded      = false;

    // Reads `system_one.*` kv. Fails when the model carries no template.
    static bool from_model(const llama_model * model, so_config & out, std::string & err);
};

// Render the template, then split on the separator. The last `n_questions` segments are
// the question blocks. `mask` is exposed to the template so a masked-slot format can place
// its mask token, and the answer slot is then that token rather than the segment's end.
bool render_segments(const so_config & cfg,
                     const std::string & state,
                     const std::vector<question> & qs,
                     std::vector<std::string> & segments,
                     std::string & err);

struct tokenized {
    std::vector<llama_token> ids;
    std::vector<int>         slots;   // one per question: position of its "(" token
};

// One BOS (when the model asks for it), then every segment with add_special = false.
bool tokenize_segments(const llama_vocab * vocab,
                       const so_config & cfg,
                       const std::vector<std::string> & segments,
                       size_t n_questions,
                       tokenized & out,
                       std::string & err);

// Cuts the state to `max_state_tokens` (head, marker, tail) before rendering, so the
// rule stays independent of where the template puts the state. Returns the state
// unchanged when it fits, which is the usual case.
std::string truncate_state(const llama_vocab * vocab, const so_config & cfg, const std::string & state);

// Single-pass tokenization of the whole prompt, for measuring what the split costs.
std::vector<llama_token> tokenize_whole(const llama_vocab * vocab, const std::string & text, bool add_special);

// Token ids of the single-character label pieces, in `letters` order.
bool letter_tokens(const llama_vocab * vocab, const std::string & letters,
                   std::vector<llama_token> & out);

struct answer {
    int                choice = 0;      // argmax over the labels
    float              confidence = 0;  // 1 - H(p)/ln K
    std::vector<float> logits;
    std::vector<float> probs;
};

// Turn one row of vocab logits into a typed answer: gather the first `n_options` label
// tokens, softmax over just those, argmax, and the entropy-normalised confidence.
// Shared by the standalone tools and the server route so the math has one home.
answer answer_from_logits(const float * row, const std::vector<llama_token> & letter_ids, int n_options);

// Expected value over an ordered rubric: sum(i * p_i), the `score` question type.
float score_expectation(const answer & a);

// One llama_decode over `t.ids`, then the label logits at every slot.
bool read_slots(llama_context * ctx,
                const tokenized & t,
                const std::vector<int> & n_options,
                const std::vector<llama_token> & letter_ids,
                std::vector<answer> & out,
                std::string & err);

} // namespace system_one
