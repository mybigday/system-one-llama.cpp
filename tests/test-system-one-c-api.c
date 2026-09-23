// The point of this file is that it is C. If it stops compiling, system-one.h has stopped
// being a C header -- which is the only way to find that out, since everything that includes
// it in-tree is C++.
//
// The checks are a macro rather than assert() on purpose: this is built in Release too, where
// NDEBUG would turn every assert into nothing and leave a test that only proves the header
// parses.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "system-one.h"

static int n_failed = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);      \
            n_failed++;                                                          \
        }                                                                        \
    } while (0)

int main(void) {
    printf("\n\nTesting libsystem-one C API...\n");
    printf("--------\n\n");

    // the readout names, which are the only place the GGUF's spelling still shows through
    CHECK(strcmp(system_one_readout_name(SYSTEM_ONE_READOUT_LETTER_SLOT), "letter_slot") == 0);
    CHECK(strcmp(system_one_readout_name(SYSTEM_ONE_READOUT_MASKED_SLOT), "masked_slot") == 0);
    CHECK(strcmp(system_one_readout_name(SYSTEM_ONE_READOUT_RANK_HEAD),   "rank_head")   == 0);
    printf("readout names: %s / %s / %s\n",
           system_one_readout_name(SYSTEM_ONE_READOUT_LETTER_SLOT),
           system_one_readout_name(SYSTEM_ONE_READOUT_MASKED_SLOT),
           system_one_readout_name(SYSTEM_ONE_READOUT_RANK_HEAD));

    // the question POD, built the way a C caller would
    {
        const char * options[] = { "burger", "fries", "drink" };
        const char * descs[]   = { "a burger", "fries", "a drink" };
        struct system_one_question q;
        q.kind      = SYSTEM_ONE_KIND_CHOICE;
        q.text      = "What is the main item?";
        q.options   = options;
        q.descs     = descs;
        q.n_options = 3;
        CHECK(q.n_options == 3);
        CHECK(strcmp(q.options[0], "burger") == 0);
        printf("question: kind %d, %zu options, first \"%s\"\n", (int) q.kind, q.n_options, q.options[0]);
    }

    // an answer read from one row of logits, over three labels
    {
        const float       row[8]   = { 0.0f, 1.0f, 3.0f, 2.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        const llama_token labels[] = { 1, 2, 3 };
        size_t n = 0;
        const float * p;
        float before;

        system_one_answer * a = system_one_answer_init_from_logits(row, labels, 3, 3);
        CHECK(a != NULL);
        if (a == NULL) {
            return 1;
        }

        // the largest of the three gathered logits is label 1 (token 2, value 3.0)
        CHECK(system_one_answer_get_choice(a) == 1);

        p = system_one_answer_get_probs(a, &n);
        CHECK(p != NULL);
        CHECK(n == 3);
        printf("probs: %.4f %.4f %.4f, confidence %.4f\n",
               (double) p[0], (double) p[1], (double) p[2],
               (double) system_one_answer_get_confidence(a));

        p = system_one_answer_get_logits(a, &n);
        CHECK(p != NULL);
        CHECK(n == 3);
        CHECK(p[0] == 1.0f);
        CHECK(p[1] == 3.0f);
        CHECK(p[2] == 2.0f);

        // T = 1 is the no-op fast path; a larger T flattens the distribution
        before = system_one_answer_get_confidence(a);
        system_one_answer_apply_temperature(a, 1.0f);
        CHECK(system_one_answer_get_confidence(a) == before);
        system_one_answer_apply_temperature(a, 4.0f);
        printf("confidence at T=4: %.4f (was %.4f)\n",
               (double) system_one_answer_get_confidence(a), (double) before);
        CHECK(system_one_answer_get_confidence(a) < before);

        system_one_answer_free(a);

        // bad arguments give NULL rather than a crash
        CHECK(system_one_answer_init_from_logits(NULL, labels, 3, 3) == NULL);
        CHECK(system_one_answer_init_from_logits(row, labels, 3, 0)  == NULL);
        CHECK(system_one_answer_init_from_logits(row, labels, 2, 3)  == NULL);  // more options than labels
    }

    // the score readout: an expectation over an ordered rubric
    {
        const float scores[] = { 0.0f, 0.0f, 10.0f };   // all the mass on the last level
        system_one_answer * a = system_one_answer_init_from_scores(scores, 3);
        CHECK(a != NULL);
        if (a != NULL) {
            CHECK(system_one_answer_get_choice(a) == 2);
            printf("score expectation: %.4f (expect close to 2)\n",
                   (double) system_one_answer_get_score_expectation(a));
            CHECK(system_one_answer_get_score_expectation(a) > 1.9f);
            system_one_answer_free(a);
        }

        CHECK(system_one_answer_init_from_scores(NULL, 3)   == NULL);
        CHECK(system_one_answer_init_from_scores(scores, 0) == NULL);
    }

    // the containers
    {
        system_one_tokenized * tok     = system_one_tokenized_init();
        system_one_answers   * answers = system_one_answers_init();
        size_t n = 12345;

        CHECK(tok != NULL);
        if (tok != NULL) {
            CHECK(system_one_tokenized_get_tokens(tok, &n) == NULL);
            CHECK(n == 0);
            n = 12345;
            CHECK(system_one_tokenized_get_slots(tok, &n) == NULL);
            CHECK(n == 0);
            system_one_tokenized_free(tok);
        }

        CHECK(answers != NULL);
        if (answers != NULL) {
            CHECK(system_one_answers_size(answers) == 0);
            CHECK(system_one_answers_get(answers, 0) == NULL);
            system_one_answers_free(answers);
        }
    }

    // every fallible call refuses a null argument with a code, and logs why
    {
        CHECK(system_one_label_tokens(NULL, NULL, 0, NULL) != 0);
        CHECK(system_one_answers_from_scores(NULL, NULL, 0, NULL) != 0);
        CHECK(system_one_answers_from_logits(NULL, NULL, 0, NULL) != 0);
        CHECK(system_one_resolve_question_slots(NULL, NULL, NULL, 0, 0, NULL) != 0);
        CHECK(system_one_params_init_from_model(NULL) == NULL);
        CHECK(system_one_plan_init(NULL, NULL, "x", NULL, 0) == NULL);
        CHECK(system_one_plan_tokenize(NULL, NULL, NULL) != 0);
        printf("null-argument guards: ok\n");
    }

    // an answer handed to system_one_answers_add() is owned by it either way -- including
    // when there is no list to store it in, which must not leak
    {
        const float scores[] = { 1.0f, 2.0f };
        system_one_answers_add(NULL, system_one_answer_init_from_scores(scores, 2));
        system_one_answers_add(NULL, NULL);
        printf("answers_add ownership on a null list: ok\n");
    }

    if (n_failed > 0) {
        fprintf(stderr, "\n%d check(s) failed\n", n_failed);
        return 1;
    }

    printf("\n\nDONE: test libsystem-one C API...\n");

    return 0;
}
