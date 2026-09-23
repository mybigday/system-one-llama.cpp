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

// This header builds the input and reads the output. Running the model is deliberately not
// here: like common/chat.h, which renders a prompt and parses a reply but never calls
// llama_decode, this is the format layer, and inference belongs to whoever owns the batching
// -- the server's scheduler, or a CLI's own loop.

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

    // Derived from the model's capabilities in from_model(), not required in the GGUF:
    // a classification head means rank_head, a bidirectional model means masked_slot, and a
    // causal one means letter_slot. system_one.readout overrides it when present.
    //   letter_slot  the next-token distribution at the end of a question's segment
    //   masked_slot  the distribution at a mask token inside it (bidirectional models)
    //   rank_head    one sequence per option, scored by the model's classification head --
    //                the reranker shape, and the only readout that costs K forward passes
    std::string readout = "letter_slot";

    // system_one.labels: option i is labelled labels[i] and that token is what gets read
    std::vector<std::string> labels;

    // from the standard metadata
    // {arch}.attention.causal, and what the readout is derived from. It is also the whole of
    // whether a prefix's KV can be reused: see plan::sequence::n_reusable.
    bool causal = true;

    llama_token mask_token = -1;    // tokenizer.ggml.mask_token_id
    std::string mask_text;          // its piece, so a template can write it
    std::string bos_text;           // likewise for BOS: the template writes it, as chat
    std::string eos_text;           // templates do, instead of a flag saying "prepend one"

    bool masked() const { return readout == "masked_slot"; }
    bool ranked() const { return readout == "rank_head"; }

    // Reads the metadata. Fails when the model carries no System One template.
    static bool from_model(const llama_model * model, so_config & out, std::string & err);
};

// The stages below are exposed so the pipeline can be checked one step at a time against a
// reference -- the regression harnesses compare the rendered text and the segment-wise ids
// separately, which build_plan() alone could not tell apart.

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

// What a request turns into before anything is evaluated: which sequences to run, where the
// answers are in them, and what the numbers coming back mean. Every decision that depends on
// the readout is made here, so a caller only has to run sequences and hand the numbers back.
// A request, resolved. build_plan() fills in everything but the tokens -- which sequences to
// run, what each renders to, where its answers will be read -- and tokenize_plan() fills in the
// rest. They are separate because a caller may have to encode part of the input itself: a state
// carrying images or audio is tokenized by whoever holds the mtmd context, not here.
struct plan {
    struct sequence {
        // stage one: what this sequence renders to. The trailing n_question_segments are the
        // question blocks; everything before them is the state.
        std::vector<std::string> segments;
        size_t n_question_segments = 0;

        size_t question = 0;   // rank_head: the question this sequence scores
        size_t option   = 0;   // rank_head: the option it scores

        // stage two: what that tokenized to, and where the answers sit in it
        tokenized tok;

        // A prefix's KV may be reused only where the prefix's representation does not depend
        // on what follows it -- that is, only on a causal model. Both numbers are already zero
        // when it does not hold, so a caller never has to ask again.

        // How far a reused prefix may extend into this sequence. Short of the whole sequence
        // when an answer is read from inside it: a slot covered by the reused part is never
        // decoded and has no logits.
        size_t n_reusable  = 0;

        // How many leading tokens this sequence has in common with an earlier one of the same
        // question, and which that is -- the state and the question, for a readout that scores
        // one sequence per option. Decode that once and copy its KV rather than repeating it.
        size_t shares_with = 0;
        size_t n_shared    = 0;
    };

    std::vector<sequence>    sequences;
    std::vector<int>         n_options;    // per question, in request order
    std::vector<llama_token> labels;       // label token ids; empty for rank_head

    bool rank_pooling = false;  // sequences are scored by the model's head, not read at a slot
};

// Stage one: what the request is, before anything is tokenized.
bool build_plan(const so_config & cfg,
                const llama_vocab * vocab,
                const std::string & state,
                const std::vector<question> & qs_in,
                plan & out,
                std::string & err);

// Stage two, for a caller whose input is all text.
bool tokenize_plan(const llama_vocab * vocab, const so_config & cfg, plan & p, std::string & err);

bool resolve_question_slots(const llama_vocab * vocab,
                            const so_config & cfg,
                            const std::vector<std::string> & question_segments,
                            size_t prefix_positions,
                            std::vector<llama_token> & ids_out,
                            std::vector<int> & slots_out,
                            std::string & err);


// One BOS (when the model asks for it), then every segment with add_special = false.
bool tokenize_segments(const llama_vocab * vocab,
                       const so_config & cfg,
                       const std::vector<std::string> & segments,
                       size_t n_questions,
                       tokenized & out,
                       std::string & err);


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

// Softmax over one question's option scores, with the same confidence measure the slot
// readouts report. The scores come from the model's classification head, one per sequence.
answer answer_from_scores(const std::vector<float> & scores);

// Assemble a request's answers from one score per planned sequence, in plan order.
bool answers_from_scores(const plan & p, const std::vector<float> & scores,
                         std::vector<answer> & out, std::string & err);

// What the readout requires of a context -- not how big to make it, which is the caller's
// business, but the part the checkpoint dictates: a rank_head answer is a classification
// head's output, which is reached as a pooled embedding.
struct context_needs {
    bool                    embeddings   = false;
    enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED;
};

context_needs required_context(const so_config & cfg);

} // namespace system_one
