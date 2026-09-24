#include "models.h"

#include <vector>

// The score of a candidate is the dot product of two projections: the marked position's, and the
// pooled text's. The pooled text is one vector per sequence, so it is projected once per sequence
// and then handed to that sequence's tokens -- two gathers rather than a projection per position.
class llama_model_gliclass::llm_graph_input_seq_head : public llm_graph_input_i {
public:
    llm_graph_input_seq_head() = default;
    ~llm_graph_input_seq_head() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (!head_pos || !tok_seq) {
            return;
        }

        GGML_ASSERT(ggml_backend_buffer_is_host(head_pos->buffer));
        GGML_ASSERT(ggml_backend_buffer_is_host(tok_seq->buffer));

        // the pooled text is the sequence's first position; find it by the smallest pos present,
        // not by pos == 0, so a ubatch that does not start at the beginning still resolves
        std::vector<llama_pos> best(ubatch->n_seqs_unq, -1);
        int32_t * head = (int32_t *) head_pos->data;
        std::fill(head, head + ubatch->n_seqs_unq, 0);

        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
                const int32_t seq_idx = ubatch->seq_idx[ubatch->seq_id[i][s]];
                if (best[seq_idx] < 0 || ubatch->pos[i] < best[seq_idx]) {
                    best[seq_idx] = ubatch->pos[i];
                    head[seq_idx] = (int32_t) i;
                }
            }
        }

        int32_t * tok = (int32_t *) tok_seq->data;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            tok[i] = ubatch->n_seq_id[i] > 0 ? ubatch->seq_idx[ubatch->seq_id[i][0]] : 0;
        }
    }

    ggml_tensor * head_pos = nullptr;  // I32 [n_seqs_unq]  the sequence's first token, in the batch
    ggml_tensor * tok_seq  = nullptr;  // I32 [n_tokens]    each token's sequence
};

void llama_model_gliclass::load_arch_hparams(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_hparams(ml);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gliclass::load_arch_tensors(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    class_proj_1   = create_tensor(tn(LLM_TENSOR_CLASS_PROJ_1, "weight"), {n_embd, n_embd}, 0);
    class_proj_1_b = create_tensor(tn(LLM_TENSOR_CLASS_PROJ_1, "bias"),   {n_embd}, 0);
    class_proj_2   = create_tensor(tn(LLM_TENSOR_CLASS_PROJ_2, "weight"), {n_embd, n_embd}, 0);
    class_proj_2_b = create_tensor(tn(LLM_TENSOR_CLASS_PROJ_2, "bias"),   {n_embd}, 0);

    text_proj_1    = create_tensor(tn(LLM_TENSOR_TEXT_PROJ_1,  "weight"), {n_embd, n_embd}, 0);
    text_proj_1_b  = create_tensor(tn(LLM_TENSOR_TEXT_PROJ_1,  "bias"),   {n_embd}, 0);
    text_proj_2    = create_tensor(tn(LLM_TENSOR_TEXT_PROJ_2,  "weight"), {n_embd, n_embd}, 0);
    text_proj_2_b  = create_tensor(tn(LLM_TENSOR_TEXT_PROJ_2,  "bias"),   {n_embd}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_gliclass::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gliclass::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_no_cache();

    // the scores are pointwise, so the narrowing to output rows waits until after them
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    ggml_tensor * cur;
    ggml_tensor * inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    inpL = build_norm(inpL, model.tok_norm, NULL, LLM_NORM, -1);
    cb(inpL, "inp_norm", -1);

    for (int il = 0; il < n_layer; ++il) {
        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        cur = inpL;

        if (model.layers[il].attn_norm) {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM, il);
            cb(cur, "attn_norm", il);
        }

        auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur, n_embd_head, n_head, n_head_kv, il);

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);
        Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                             freq_base_l, freq_scale_l, ext_factor, attn_factor, beta_fast, beta_slow);

        cur = build_attn(inp_attn, model.layers[il].wo, nullptr, model.layers[il].wo_s,
                         Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), il);
        cb(cur, "kqv_out", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                NULL,                      NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL, hparams.llm_ffn_op, LLM_FFN_SEQ, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        inpL = cur;
    }

    cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM, -1);
    cb(cur, "final_norm_out", -1);

    // FeaturesProjector: linear -> GELU -> linear, both with bias. nn.GELU() is the exact erf form.
    auto project = [&](ggml_tensor * x, ggml_tensor * w1, ggml_tensor * b1,
                                        ggml_tensor * w2, ggml_tensor * b2) {
        x = ggml_add(ctx0, build_lora_mm(w1, x, nullptr), b1);
        x = ggml_gelu_erf(ctx0, x);
        return ggml_add(ctx0, build_lora_mm(w2, x, nullptr), b2);
    };

    ggml_tensor * txt;
    {
        auto inp = std::make_unique<llm_graph_input_seq_head>();

        inp->head_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ubatch.n_seqs_unq);
        ggml_set_input(inp->head_pos);
        ggml_set_name(inp->head_pos, "seq_head_pos");

        inp->tok_seq = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->tok_seq);
        ggml_set_name(inp->tok_seq, "tok_seq");

        // project the pooled text once per sequence, then hand each token its own
        ggml_tensor * pooled = ggml_get_rows(ctx0, cur, inp->head_pos);
        txt = project(pooled, model.text_proj_1, model.text_proj_1_b,
                              model.text_proj_2, model.text_proj_2_b);
        txt = ggml_get_rows(ctx0, txt, inp->tok_seq);
        cb(txt, "text_proj_out", -1);

        res->add_input(std::move(inp));
    }

    cur = project(cur, model.class_proj_1, model.class_proj_1_b,
                       model.class_proj_2, model.class_proj_2_b);
    cb(cur, "class_proj_out", -1);

    // ScorerDot: one number per position, and the caller reads the marked ones
    cur = ggml_sum_rows(ctx0, ggml_mul(ctx0, cur, txt));
    cb(cur, "scorer_out", -1);

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
