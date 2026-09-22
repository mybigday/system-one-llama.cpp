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

// What the checkpoint says about its own prompt and readout. Three of these come from
// `system_one.*` keys; the rest are standard GGUF metadata, because a mask token, a leading BOS
// and an attention direction all have homes of their own already.
struct so_config {
    std::string template_src;                             // tokenizer.chat_template.system_one
    std::string segment_separator = "\x1e";               // system_one.segment_separator

    // system_one.readout: letter_slot reads the next-token distribution at the end of a
    // question's segment, masked_slot reads the distribution at the mask token inside it
    std::string readout = "letter_slot";

    // system_one.labels: option i is labelled labels[i] and that token is what gets read
    std::vector<std::string> labels;

    // from the standard metadata
    llama_token mask_token = -1;    // tokenizer.ggml.mask_token_id
    std::string mask_text;          // its piece, so a template can write it
    std::string bos_text;           // likewise for BOS: the template writes it, as chat
    std::string eos_text;           // templates do, instead of a flag saying "prepend one"

    bool masked() const { return readout == "masked_slot"; }

    // Reads the metadata. Fails when the model carries no System One template.
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

// Single-pass tokenization of the whole prompt, for measuring what the split costs.
std::vector<llama_token> tokenize_whole(const llama_vocab * vocab, const std::string & text, bool add_special);

// Token ids of the label pieces, in `labels` order. Every label must be a single token, since
// the answer is read as one position's distribution over them.
bool label_tokens(const llama_vocab * vocab, const std::vector<std::string> & labels,
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
