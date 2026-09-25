#include "models.h"

#include <vector>

// The question type is not carried by the batch and does not have to be: laya writes the word
// "choice" / "score" / "noul" into the prompt itself. Which token that is depends on the
// checkpoint's tokenizer, so the three ids and the position are declared as arch metadata and
// matched here rather than guessed from text.
class llama_model_laya::llm_graph_input_qtype : public llm_graph_input_i {
public:
    llm_graph_input_qtype(const llama_hparams & hparams) : hparams(hparams) {}
    ~llm_graph_input_qtype() = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (!qtype) {
            return;
        }

        GGML_ASSERT(ggml_backend_buffer_is_host(qtype->buffer));

        // First which type each sequence asks, then that answer for each of its tokens. Per token
        // and not per sequence: several sequences share a ubatch, and a per-sequence row could
        // only reach their tokens by a broadcast that does not know where one sequence ends.
        std::vector<int32_t> by_seq(ubatch->n_seqs_unq, 0);

        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            if ((uint32_t) ubatch->pos[i] != hparams.qtype_token_index) {
                continue;
            }

            for (size_t t = 0; t < hparams.qtype_token_ids.size(); ++t) {
                if (ubatch->token[i] != hparams.qtype_token_ids[t]) {
                    continue;
                }
                for (int32_t s = 0; s < ubatch->n_seq_id[i]; ++s) {
                    by_seq[ubatch->seq_idx[ubatch->seq_id[i][s]]] = (int32_t) t;
                }
                break;
            }
        }

        int32_t * data = (int32_t *) qtype->data;
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            data[i] = ubatch->n_seq_id[i] > 0 ? by_seq[ubatch->seq_idx[ubatch->seq_id[i][0]]] : 0;
        }
    }

    ggml_tensor * qtype = nullptr;  // I32 [n_tokens]

    const llama_hparams & hparams;
};

void llama_model_laya::load_arch_hparams(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DECISION_HEAD_BLOCK_COUNT,       hparams.n_dhead_layer);
    ml.get_key(LLM_KV_DECISION_HEAD_QTYPE_TOKEN_INDEX, hparams.qtype_token_index);
    std::vector<int32_t> qtype_ids;
    ml.get_arr(LLM_KV_DECISION_HEAD_QTYPE_TOKEN_IDS, qtype_ids);
    if (qtype_ids.size() != hparams.qtype_token_ids.size()) {
        throw std::runtime_error(format("%s: expected %zu question-type token ids, got %zu",
                    __func__, hparams.qtype_token_ids.size(), qtype_ids.size()));
    }
    std::copy(qtype_ids.begin(), qtype_ids.end(), hparams.qtype_token_ids.begin());

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_laya::load_arch_tensors(llama_model_loader & ml) {
    llama_model_modern_bert::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t n_dhead = hparams.n_dhead_layer;
    const int64_t n_ff_dh = 4 * n_embd;   // nn.TransformerEncoderLayer(d, ..., 4*d, ...)

    dhead_layers.resize(n_dhead);

    for (int64_t i = 0; i < n_dhead; ++i) {
        auto & layer = dhead_layers[i];

        layer.attn_norm   = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.attn_norm_b = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_NORM, "bias",   i), {n_embd}, 0);

        layer.wqkv        = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_QKV, "weight", i), {n_embd, 3 * n_embd}, 0);
        layer.wqkv_b      = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_QKV, "bias",   i), {3 * n_embd}, 0);
        layer.wo          = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);
        layer.wo_b        = create_tensor(tn(LLM_TENSOR_DHEAD_ATTN_OUT, "bias",   i), {n_embd}, 0);

        layer.ffn_norm    = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm_b  = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_NORM, "bias",   i), {n_embd}, 0);

        layer.ffn_up      = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_UP,   "weight", i), {n_embd, n_ff_dh}, 0);
        layer.ffn_up_b    = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_UP,   "bias",   i), {n_ff_dh}, 0);
        layer.ffn_down    = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_DOWN, "weight", i), {n_ff_dh, n_embd}, 0);
        layer.ffn_down_b  = create_tensor(tn(LLM_TENSOR_DHEAD_FFN_DOWN, "bias",   i), {n_embd}, 0);
    }

    qtype_embd    = create_tensor(tn(LLM_TENSOR_QTYPE_EMBD,  "weight"), {n_embd, 3}, 0);

    scorer_norm   = create_tensor(tn(LLM_TENSOR_SCORER_NORM, "weight"), {n_embd}, 0);
    scorer_norm_b = create_tensor(tn(LLM_TENSOR_SCORER_NORM, "bias"),   {n_embd}, 0);
    scorer        = create_tensor(tn(LLM_TENSOR_SCORER,      "weight"), {n_embd, n_embd}, 0);
    scorer_b      = create_tensor(tn(LLM_TENSOR_SCORER,      "bias"),   {n_embd}, 0);
    scorer_out    = create_tensor(tn(LLM_TENSOR_SCORER_OUT,  "weight"), {n_embd, 1}, 0);
    scorer_out_b  = create_tensor(tn(LLM_TENSOR_SCORER_OUT,  "bias"),   {1}, 0);
}

std::unique_ptr<llm_graph_context> llama_model_laya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_laya::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const int64_t n_dhead     = hparams.n_dhead_layer;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_no_cache();

    // The narrowing to output rows that the encoder normally does on its last layer is left to the
    // very end here: the decision head attends over the whole sequence, so it needs every position.
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
        cb(ffn_inp, "ffn_inp", il);

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

    // one of three learned vectors, chosen by the question's type, added to every position
    {
        auto inp = std::make_unique<llm_graph_input_qtype>(hparams);
        inp->qtype = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        ggml_set_input(inp->qtype);
        ggml_set_name(inp->qtype, "qtype");

        ggml_tensor * qt = ggml_get_rows(ctx0, model.qtype_embd, inp->qtype);
        cb(qt, "qtype_embd", -1);

        cur = ggml_add(ctx0, cur, qt);
        cb(cur, "qtype_added", -1);

        res->add_input(std::move(inp));
    }

    // the decision head: plain pre-norm transformer layers, ReLU feed-forward, biases throughout,
    // and no positional encoding -- none of which the encoder's layers use
    for (int64_t i = 0; i < n_dhead; ++i) {
        const auto & layer = model.dhead_layers[i];
        // The head sits after the last encoder layer and runs where it runs, so it borrows that
        // layer's index: a graph node is placed by `model.dev_layer(il)`, and a fused one is
        // asserted to have a real index. What these layers genuinely do not share with the
        // encoder is its sliding-window pattern and its rope -- and neither is taken from `il`
        // here: the mask is passed explicitly below and there is no positional encoding at all.
        const int    il    = n_layer - 1;

        ggml_tensor * inp = cur;

        cur = build_norm(inp, layer.attn_norm, layer.attn_norm_b, LLM_NORM, il);
        cb(cur, "dhead_attn_norm", il);

        ggml_tensor * qkv = ggml_add(ctx0, build_lora_mm(layer.wqkv, cur, nullptr), layer.wqkv_b);
        cb(qkv, "dhead_wqkv", il);

        ggml_tensor * Qcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens,
                ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], 0);
        ggml_tensor * Kcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens,
                ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], ggml_row_size(qkv->type, n_embd));
        ggml_tensor * Vcur = ggml_view_3d(ctx0, qkv, n_embd_head, n_head, n_tokens,
                ggml_row_size(qkv->type, n_embd_head), qkv->nb[1], ggml_row_size(qkv->type, 2 * n_embd));

        // the mask is taken explicitly rather than through build_attn(), which would pick it from
        // hparams.is_swa(il): these layers are not part of the encoder's sliding-window pattern
        // and attend over the whole sequence
        ggml_build_forward_expand(gf, Kcur);
        ggml_build_forward_expand(gf, Vcur);

        cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, inp_attn->get_kq_mask(), nullptr, nullptr,
                             0, 1.0f/sqrtf(float(n_embd_head)), il);
        cur = ggml_add(ctx0, build_lora_mm(layer.wo, cur, nullptr), layer.wo_b);
        cb(cur, "dhead_kqv_out", il);

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inp);

        cur = build_norm(ffn_inp, layer.ffn_norm, layer.ffn_norm_b, LLM_NORM, il);
        cb(cur, "dhead_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   layer.ffn_up_b,   NULL,
                NULL,           NULL,             NULL,
                layer.ffn_down, layer.ffn_down_b, NULL,
                NULL, LLM_FFN_RELU, LLM_FFN_SEQ, il);

        cur = ggml_add(ctx0, cur, ffn_inp);
    }

    // the scorer is pointwise, so it runs on every position and the caller reads the rows it asked
    // for -- which is why the answer slots never have to reach the graph
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.scorer_norm, model.scorer_norm_b, LLM_NORM, -1);
    cur = ggml_add(ctx0, build_lora_mm(model.scorer, cur, nullptr), model.scorer_b);
    cur = ggml_gelu_erf(ctx0, cur);   // nn.GELU() defaults to the exact erf form, not the tanh one
    cur = ggml_add(ctx0, build_lora_mm(model.scorer_out, cur, nullptr), model.scorer_out_b);
    cb(cur, "scorer_out", -1);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
