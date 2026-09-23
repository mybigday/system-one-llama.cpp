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
    // {arch}.attention.causal, and what the readout is derived from. Also decides whether a
    // shared prefix is really shared: see plan::share_prefix.
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

// Render the template, then split on the separator. The last `n_questions` segments are
// the question blocks. `mask` is exposed to the template so a masked-slot format can place
// its mask token, and the answer slot is then that token rather than the segment's end.
bool render_segments(const so_config & cfg,
                     const std::string & state,
                     const std::vector<question> & qs,
                     std::vector<std::string> & segments,
                     std::string & err);

// A rank_head prompt is built per (question, option) pair rather than once for the whole
// request: the template sees `state`, `question` and `option` instead of `questions`.
bool render_pair_segments(const so_config & cfg,
                          const std::string & state,
                          const question & q,
                          size_t option_index,
                          std::vector<std::string> & segments,
                          std::string & err);


struct tokenized {
    std::vector<llama_token> ids;
    std::vector<int>         slots;   // one per question: position of its "(" token
};

// What a request turns into before anything is evaluated: which sequences to run, where the
// answers are in them, and what the numbers coming back mean. Every decision that depends on
// the readout is made here, so a caller only has to run sequences and hand the numbers back.
struct plan {
    struct sequence {
        tokenized tok;              // ids, and the answer slots when the readout has any
        size_t    question = 0;     // rank_head: the question this sequence scores
        size_t    option   = 0;     // rank_head: the option it scores

        // How many leading tokens this sequence has in common with the first sequence of its
        // question, and which that is. A caller with a causal model can decode that prefix
        // once and copy its KV to the rest instead of recomputing it per option; see
        // plan::share_prefix for when that is allowed.
        size_t    shares_with = 0;
        size_t    n_shared    = 0;
    };

    std::vector<sequence>    sequences;
    std::vector<int>         n_options;    // per question, in request order
    std::vector<llama_token> labels;       // label token ids; empty for rank_head

    bool rank_pooling = false;  // sequences are scored by the model's head, not read at a slot
    bool prefix_reuse = true;   // false when attention is bidirectional: nothing is reusable

    // Whether the sequences' shared prefixes may actually be shared. Bidirectional attention
    // makes a prefix's representation depend on what follows it, so the same tokens are not
    // the same computation and n_shared must be ignored; a causal model has no such problem.
    bool share_prefix = false;
};

// A noul question with no options declared gets {"no","yes"}: which words the two sides of
// a yes/no answer are spelled with is a property of the format, not something each front end
// should have to know. A caller that wants them worded differently still sets them.
// What a request renders to, before anything is tokenized. The trailing `n_question_segments`
// of each sequence are the question blocks; everything before them is the state.
//
// This exists so a caller can encode the state itself -- with mtmd, when it carries images or
// audio -- and still get the question slots placed by the same rule as the text-only path.
// Splitting a request is phase one, turning it into tokens is phase two, and build_plan() is
// simply the two composed for a caller that has nothing but text.
struct layout {
    struct sequence {
        std::vector<std::string> segments;
        size_t n_question_segments = 0;
        size_t question = 0;   // rank_head: the question this sequence scores
        size_t option   = 0;   // rank_head: the option it scores
    };

    std::vector<sequence>    sequences;
    std::vector<int>         n_options;
    std::vector<llama_token> labels;
    bool rank_pooling = false;
    bool prefix_reuse = true;
    bool share_prefix = false;
};

bool build_layout(const so_config & cfg,
                  const llama_vocab * vocab,
                  const std::string & state,
                  const std::vector<question> & qs_in,
                  layout & out,
                  std::string & err);

// Place the answer slots of the question segments, given how many positions the state already
// occupies. The state may be anything -- plain text, or text with media chunks the caller
// encoded itself -- because a question segment is always text, and the slot rule only needs to
// know where the question blocks begin.
bool resolve_question_slots(const llama_vocab * vocab,
                            const so_config & cfg,
                            const std::vector<std::string> & question_segments,
                            size_t prefix_positions,
                            std::vector<llama_token> & ids_out,
                            std::vector<int> & slots_out,
                            std::string & err);

bool build_plan(const so_config & cfg,
                const llama_vocab * vocab,
                const std::string & state,
                const std::vector<question> & qs_in,
                plan & out,
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

// What a request needs of the context that will run it. This only reports; applying it is the
// caller's, so nothing it owns changes behind its back and it stays free to refuse, to clamp,
// or to keep a value the user asked for.
struct context_needs {
    bool               embeddings   = false;                          // rank_head answers through the head
    enum llama_pooling_type pooling_type = LLAMA_POOLING_TYPE_UNSPECIFIED; // ... pooled as a rank score
    bool               share_prefix = false; // the sequences want to share a prefix's KV, which
                                             // a partial seq_cp() can only do within one stream
    size_t             n_seq        = 1;  // sequences that would go into one decode
    size_t             n_tokens     = 0;  // ... and how many tokens that is
};

// Capped, because a 255-option question should not demand a 255-sequence micro-batch.
// The caller then fits whatever context it actually created.
constexpr size_t MAX_BATCHED_SEQS = 32;

context_needs required_context(const so_config & cfg, const plan & p);

} // namespace system_one
