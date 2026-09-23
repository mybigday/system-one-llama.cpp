#ifndef SYSTEM_ONE_H
#define SYSTEM_ONE_H

#include "llama.h"

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
#include <memory>
#include <string>
#include <vector>
#endif

/**
 * libsystem-one: System One (Jev-compatible) typed decision readout.
 *
 * One forward pass over [state | question 1 .. question n], reading the label-token logits at
 * each question's answer slot. No decoding, no grammar: the answer is a distribution over a
 * small label set, which is what makes the output typed by construction.
 *
 * This library builds the input and reads the output. Running the model is deliberately not
 * here: like common/chat.h, which renders a prompt and parses a reply but never calls
 * llama_decode, this is the format layer, and inference belongs to whoever owns the batching
 * -- the server's scheduler, or a CLI's own loop.
 *
 * The prompt format is a property of the checkpoint, not of this code. It is a Jinja template
 * stored in the GGUF as the named chat template `system_one`, and rendered with llama.cpp's
 * own Jinja engine. There is deliberately NO built-in default: different System One lines
 * (letter slots, a scoring head taking one option per sequence, per-option mask tokens) need
 * different prompts, so a model without a template is an error rather than an assumption.
 *
 * Segment boundaries matter as much as the text. The reference implementations encode the
 * state and each question block separately and concatenate the ids, which differs from a
 * single-pass tokenization at the seams. The template marks those seams with
 * `system_one.segment_separator` (U+001E); this side splits on it and tokenizes each piece on
 * its own.
 *
 * For contributors:
 * - Keep the C API aligned with the libllama C API (as in llama.h)
 * - Anything that owns a container is opaque; plain value bags are PODs
 * - Every fallible call ends with (char * err, size_t err_len); err may be NULL
 */

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define SYSTEM_ONE_API __declspec(dllexport)
#        else
#            define SYSTEM_ONE_API __declspec(dllimport)
#        endif
#    else
#        define SYSTEM_ONE_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define SYSTEM_ONE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

enum system_one_kind {
    SYSTEM_ONE_KIND_NOUL,
    SYSTEM_ONE_KIND_CHOICE,
    SYSTEM_ONE_KIND_SCORE,
};

// Where an answer is read, and what is read there. A closed set: every readout needs its own
// slot rule and its own way of turning numbers into a distribution, so a value outside it has
// no implementation to reach.
//   LETTER_SLOT  the next-token distribution at the end of a question's segment
//   MASKED_SLOT  the distribution at a mask token inside it (bidirectional models)
//   RANK_HEAD    one sequence per option, scored by the model's classification head -- the
//                reranker shape, and the only readout that costs K forward passes
enum system_one_readout {
    SYSTEM_ONE_READOUT_LETTER_SLOT,
    SYSTEM_ONE_READOUT_MASKED_SLOT,
    SYSTEM_ONE_READOUT_RANK_HEAD,
};

// opaque types
struct system_one_params;
struct system_one_plan;
struct system_one_sequence;
struct system_one_tokenized;
struct system_one_answer;
struct system_one_answers;

typedef struct system_one_params    system_one_params;
typedef struct system_one_plan      system_one_plan;
typedef struct system_one_sequence  system_one_sequence;
typedef struct system_one_tokenized system_one_tokenized;
typedef struct system_one_answer    system_one_answer;
typedef struct system_one_answers   system_one_answers;

// One question, as the caller states it. Pure input: the strings are borrowed for the
// duration of the call that takes them.
//   noul   options may be left empty; the readout supplies {"no","yes"} in that order, so
//          the affirmative side is last and P(true) is the last probability
//   choice options are the criteria keys
//   score  options are the rubric levels, in order
// descs is optional and, when given, has one entry per option (an entry may be NULL).
struct system_one_question {
    enum system_one_kind kind;
    const char *         text;
    const char * const * options;
    const char * const * descs;
    size_t               n_options;
};

// What the readout requires of a context -- not how big to make it, which is the caller's
// business, but the part the checkpoint dictates: a rank_head answer is a classification
// head's output, which is reached as a pooled embedding.
struct system_one_context_needs {
    bool                    embeddings;
    enum llama_pooling_type pooling_type;
};

SYSTEM_ONE_API const char * system_one_readout_name(enum system_one_readout readout);

//
// system_one_params -- what the checkpoint says about its own prompt and readout
//

// The built-in defaults and no template -- for a caller that supplies the whole format itself,
// which is how a model carrying no System One metadata is still asked a question.
SYSTEM_ONE_API system_one_params * system_one_params_init(void);

// Reads the model's metadata. Returns NULL when the model carries no System One template.
SYSTEM_ONE_API system_one_params * system_one_params_init_from_model(const struct llama_model * model,
                                                                     char * err, size_t err_len);
SYSTEM_ONE_API void system_one_params_free(system_one_params * params);

// Derived from the model's capabilities, not required in the GGUF: a classification head means
// RANK_HEAD, a bidirectional model means MASKED_SLOT, and a causal one means LETTER_SLOT.
// `system_one.readout` overrides it, and so does the setter.
SYSTEM_ONE_API enum system_one_readout system_one_params_get_readout(const system_one_params * params);
SYSTEM_ONE_API void                    system_one_params_set_readout(system_one_params * params,
                                                                     enum system_one_readout readout);

SYSTEM_ONE_API const char * system_one_params_get_template(const system_one_params * params);
SYSTEM_ONE_API void         system_one_params_set_template(system_one_params * params, const char * jinja);

SYSTEM_ONE_API const char * system_one_params_get_segment_separator(const system_one_params * params);

// The BOS piece, so a caller can tell whether the template writes one.
SYSTEM_ONE_API const char * system_one_params_get_bos_text(const system_one_params * params);

// Option i is labelled label i, and that token is what gets read. Defaults to A-Za-z when the
// checkpoint says nothing, which is why a failure to resolve one says where the label came
// from -- the caller may never have chosen it. A template override usually needs labels too:
// a format that writes (1)(2)(3) has to say so, or the answer is read at A, B, C instead.
SYSTEM_ONE_API size_t       system_one_params_get_n_labels(const system_one_params * params);
SYSTEM_ONE_API const char * system_one_params_get_label   (const system_one_params * params, size_t i);
SYSTEM_ONE_API void         system_one_params_set_labels  (system_one_params * params,
                                                           const char * const * labels, size_t n_labels);

// {arch}.attention.causal. It is also the whole of whether a prefix's KV can be reused: see
// system_one_sequence_get_n_reusable().
SYSTEM_ONE_API bool system_one_params_get_causal(const system_one_params * params);

SYSTEM_ONE_API struct system_one_context_needs system_one_params_get_context_needs(const system_one_params * params);

//
// system_one_plan -- a request, resolved
//
// Which sequences to run, where the answers are in them, and what the numbers coming back
// mean. Every decision that depends on the readout is made here, so a caller only has to run
// the sequences and hand the numbers back.
//
// Two stages, because a caller may have to encode part of the input itself: a state carrying
// images or audio is tokenized by whoever holds the mtmd context, not here. Stage one renders
// and plans; system_one_plan_tokenize() is stage two, for a caller whose input is all text.
//

// Returns NULL on failure.
SYSTEM_ONE_API system_one_plan * system_one_plan_init(const system_one_params * params,
                                                      const struct llama_vocab * vocab,
                                                      const char * state,
                                                      const struct system_one_question * questions,
                                                      size_t n_questions,
                                                      char * err, size_t err_len);
SYSTEM_ONE_API void system_one_plan_free(system_one_plan * plan);

// Returns 0 on success.
SYSTEM_ONE_API int32_t system_one_plan_tokenize(system_one_plan * plan,
                                                const struct llama_vocab * vocab,
                                                const system_one_params * params,
                                                char * err, size_t err_len);

SYSTEM_ONE_API size_t  system_one_plan_get_n_sequences(const system_one_plan * plan);
SYSTEM_ONE_API size_t  system_one_plan_get_n_questions(const system_one_plan * plan);
SYSTEM_ONE_API int32_t system_one_plan_get_n_options  (const system_one_plan * plan, size_t question);

// True when the sequences are scored by the model's head rather than read at a slot.
SYSTEM_ONE_API bool system_one_plan_get_rank_pooling(const system_one_plan * plan);

// The label token ids, in label order. NULL for a rank_head plan, which reads no slots.
SYSTEM_ONE_API const llama_token * system_one_plan_get_labels(const system_one_plan * plan, size_t * n_out);

// Valid until the plan is freed. NULL when i is out of range.
SYSTEM_ONE_API const system_one_sequence * system_one_plan_get_sequence(const system_one_plan * plan, size_t i);

//
// system_one_sequence -- one sequence the request turns into
//
// A slot readout plans exactly one; a rank_head readout plans one per option.
//

// The trailing n_question_segments are the question blocks; everything before them is the
// state. A caller that tokenizes the state itself renders from these.
SYSTEM_ONE_API size_t       system_one_sequence_get_n_segments         (const system_one_sequence * seq);
SYSTEM_ONE_API const char * system_one_sequence_get_segment            (const system_one_sequence * seq, size_t i);
SYSTEM_ONE_API size_t       system_one_sequence_get_n_question_segments(const system_one_sequence * seq);

// rank_head: which question this sequence scores, and which of its options.
SYSTEM_ONE_API size_t system_one_sequence_get_question(const system_one_sequence * seq);
SYSTEM_ONE_API size_t system_one_sequence_get_option  (const system_one_sequence * seq);

// Empty until system_one_plan_tokenize().
SYSTEM_ONE_API const system_one_tokenized * system_one_sequence_get_tokenized(const system_one_sequence * seq);

// A prefix's KV may be reused only where the prefix's representation does not depend on what
// follows it -- that is, only on a causal model. Both numbers are already zero when that does
// not hold, so a caller never has to ask again.

// How far a reused prefix may extend into this sequence. Short of the whole sequence when an
// answer is read from inside it: a slot covered by the reused part is never decoded and has no
// logits.
SYSTEM_ONE_API size_t system_one_sequence_get_n_reusable(const system_one_sequence * seq);

// How many leading tokens this sequence has in common with an earlier one of the same question,
// and which that is -- the state and the question, for a readout that scores one sequence per
// option. Decode that once and copy its KV rather than repeating it.
SYSTEM_ONE_API size_t system_one_sequence_get_shares_with(const system_one_sequence * seq);
SYSTEM_ONE_API size_t system_one_sequence_get_n_shared   (const system_one_sequence * seq);

//
// system_one_tokenized -- ids, and the answer slots in them
//

SYSTEM_ONE_API system_one_tokenized * system_one_tokenized_init(void);
SYSTEM_ONE_API void                   system_one_tokenized_free(system_one_tokenized * tok);

SYSTEM_ONE_API const llama_token * system_one_tokenized_get_tokens(const system_one_tokenized * tok, size_t * n_out);
// One slot per question: the position whose distribution carries that question's answer.
SYSTEM_ONE_API const int32_t *     system_one_tokenized_get_slots (const system_one_tokenized * tok, size_t * n_out);

// For a caller that tokenized the state itself (media in the state, say): place the question
// blocks after however many positions the state occupied, and report their answer slots
// against the whole sequence. Returns 0 on success.
SYSTEM_ONE_API int32_t system_one_resolve_question_slots(const struct llama_vocab * vocab,
                                                         const system_one_params * params,
                                                         const char * const * question_segments,
                                                         size_t n_segments,
                                                         size_t prefix_positions,
                                                         system_one_tokenized * out,
                                                         char * err, size_t err_len);

//
// system_one_answer -- one question's distribution, and what follows from it
//

// Gather the first n_options label tokens out of one row of vocab logits, softmax over just
// those, argmax, and the entropy-normalised confidence. Returns NULL on bad arguments.
SYSTEM_ONE_API system_one_answer * system_one_answer_init_from_logits(const float * row,
                                                                      const llama_token * labels,
                                                                      size_t n_labels,
                                                                      int32_t n_options);

// The same, from one score per option -- what a classification head produces.
SYSTEM_ONE_API system_one_answer * system_one_answer_init_from_scores(const float * scores, size_t n_scores);

SYSTEM_ONE_API void system_one_answer_free(system_one_answer * answer);

SYSTEM_ONE_API int32_t       system_one_answer_get_choice    (const system_one_answer * answer);
SYSTEM_ONE_API float         system_one_answer_get_confidence(const system_one_answer * answer);  // 1 - H(p)/ln K
SYSTEM_ONE_API const float * system_one_answer_get_probs     (const system_one_answer * answer, size_t * n_out);
SYSTEM_ONE_API const float * system_one_answer_get_logits    (const system_one_answer * answer, size_t * n_out);

// Expected value over an ordered rubric: sum(i * p_i), the `score` question type.
SYSTEM_ONE_API float system_one_answer_get_score_expectation(const system_one_answer * answer);

// Rescale by a temperature and recompute everything that follows from it. The model ships with
// its own T already folded into its weights, so this is for a caller recalibrating on a
// distribution of its own; T = 1 is the common case and costs nothing, which is why it is
// checked rather than applied. One T per request, not per question.
SYSTEM_ONE_API void system_one_answer_apply_temperature(system_one_answer * answer, float t);

//
// system_one_answers -- a request's answers, in question order
//

SYSTEM_ONE_API system_one_answers *      system_one_answers_init(void);
SYSTEM_ONE_API void                      system_one_answers_free(system_one_answers * answers);
SYSTEM_ONE_API size_t                    system_one_answers_size(const system_one_answers * answers);
SYSTEM_ONE_API const system_one_answer * system_one_answers_get (const system_one_answers * answers, size_t i);

// Append, taking ownership of `answer` -- which is how a caller that reads the slots itself
// builds the list, the slot readouts' case. `system_one_answers_from_scores()` is the rank_head
// counterpart, where the library assembles the list from one score per sequence.
SYSTEM_ONE_API void system_one_answers_add(system_one_answers * answers, system_one_answer * answer);

// Assemble a request's answers from one score per planned sequence, in plan order.
// Returns 0 on success.
SYSTEM_ONE_API int32_t system_one_answers_from_scores(const system_one_plan * plan,
                                                      const float * scores, size_t n_scores,
                                                      system_one_answers * out,
                                                      char * err, size_t err_len);

// Token ids of the label pieces, in label order, into a caller array of n_labels entries.
// Every label must be a single token, since the answer is read as one position's distribution
// over them. Returns 0 on success; on failure `err` names the offending label.
SYSTEM_ONE_API int32_t system_one_label_tokens(const struct llama_vocab * vocab,
                                               const char * const * labels, size_t n_labels,
                                               llama_token * out,
                                               char * err, size_t err_len);

#ifdef __cplusplus
} // extern "C"
#endif

//
// C++ wrappers
//

#ifdef __cplusplus

namespace system_one {

struct params_deleter {
    void operator()(system_one_params * val) { system_one_params_free(val); }
};
using params_ptr = std::unique_ptr<system_one_params, params_deleter>;

struct plan_deleter {
    void operator()(system_one_plan * val) { system_one_plan_free(val); }
};
using plan_ptr = std::unique_ptr<system_one_plan, plan_deleter>;

struct tokenized_deleter {
    void operator()(system_one_tokenized * val) { system_one_tokenized_free(val); }
};
using tokenized_ptr = std::unique_ptr<system_one_tokenized, tokenized_deleter>;

struct answer_deleter {
    void operator()(system_one_answer * val) { system_one_answer_free(val); }
};
using answer_ptr = std::unique_ptr<system_one_answer, answer_deleter>;

struct answers_deleter {
    void operator()(system_one_answers * val) { system_one_answers_free(val); }
};
using answers_ptr = std::unique_ptr<system_one_answers, answers_deleter>;

// Error text, for the calls that take (char * err, size_t err_len).
struct error_buffer {
    char buf[512] = {0};
    char * data() { return buf; }
    size_t size() const { return sizeof(buf); }
    std::string str() const { return std::string(buf); }
};

// A question stated in C++ terms. The C struct borrows pointers, so the strings have to
// outlive the call that takes them; question_list owns them and hands out the array.
struct question {
    enum system_one_kind     kind = SYSTEM_ONE_KIND_NOUL;
    std::string              text;
    std::vector<std::string> options;   // noul: may be left empty
    std::vector<std::string> descs;     // empty, or one per option
};

struct question_list {
    std::vector<question> entries;

    size_t size() const { return entries.size(); }

    // Valid until entries changes or the list is destroyed.
    const system_one_question * c_ptr() {
        c_questions.clear();
        c_options.assign(entries.size(), {});
        c_descs.assign(entries.size(), {});
        c_questions.reserve(entries.size());
        for (size_t i = 0; i < entries.size(); i++) {
            const question & q = entries[i];
            for (const auto & o : q.options) c_options[i].push_back(o.c_str());
            for (const auto & d : q.descs)   c_descs[i].push_back(d.c_str());

            system_one_question c;
            c.kind      = q.kind;
            c.text      = q.text.c_str();
            c.options   = c_options[i].empty() ? nullptr : c_options[i].data();
            c.descs     = c_descs[i].size() == q.options.size() && !c_descs[i].empty() ? c_descs[i].data() : nullptr;
            c.n_options = q.options.size();
            c_questions.push_back(c);
        }
        return c_questions.empty() ? nullptr : c_questions.data();
    }

  private:
    std::vector<system_one_question>       c_questions;
    std::vector<std::vector<const char *>> c_options;
    std::vector<std::vector<const char *>> c_descs;
};

inline std::vector<llama_token> tokens_of(const system_one_tokenized * tok) {
    size_t n = 0;
    const llama_token * p = system_one_tokenized_get_tokens(tok, &n);
    return p ? std::vector<llama_token>(p, p + n) : std::vector<llama_token>();
}

inline std::vector<int32_t> slots_of(const system_one_tokenized * tok) {
    size_t n = 0;
    const int32_t * p = system_one_tokenized_get_slots(tok, &n);
    return p ? std::vector<int32_t>(p, p + n) : std::vector<int32_t>();
}

inline std::vector<float> probs_of(const system_one_answer * a) {
    size_t n = 0;
    const float * p = system_one_answer_get_probs(a, &n);
    return p ? std::vector<float>(p, p + n) : std::vector<float>();
}

inline std::vector<float> logits_of(const system_one_answer * a) {
    size_t n = 0;
    const float * p = system_one_answer_get_logits(a, &n);
    return p ? std::vector<float>(p, p + n) : std::vector<float>();
}

} // namespace system_one

#endif

#endif // SYSTEM_ONE_H
