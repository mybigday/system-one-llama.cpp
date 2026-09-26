#include "models.h"

// kev (github.com/jaredpalmer/kev) answers a typed question with a pointer head rather than with
// a vocabulary: one row per question, the options marked by `</opt>`, the query at the final
// `<decide>` token, and
//
//     logit_j = dot( k(h[</opt>_j]) , q(h[<decide>]) ) / sqrt(dp) / T
//
// (kev/model.py, class PointerHead). The backbone is Qwen3.5 untouched, so everything here is
// what that formula adds: two linear maps of the same hidden state.
//
// The graph emits both, concatenated, instead of the dot itself. The dot needs two positions at
// once, and in a causal row the query is the *last* token -- so a graph that computed it would
// only be correct while a whole row fits in one ubatch, and would go quietly wrong on a long
// state. Emitting [k ; q] per position is pointwise, so it holds for any split, and the one
// multiply-accumulate left over is the caller reading its own output.
//
// The scale and the fitted temperature are folded into `q` at conversion (the dot is linear in
// q), so what comes out is already kev's calibrated logit and nothing here reads a constant.

void llama_model_kev::load_arch_hparams(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_hparams(ml);

    ml.get_key(LLM_KV_DECISION_HEAD_POINTER_DIM, hparams.n_pointer_dim);

    if (hparams.n_pointer_dim == 0) {
        throw std::runtime_error("kev: decision_head.pointer_dim must be non-zero");
    }

    // Derived, not declared: an output row is one key followed by one query, so its width is
    // twice the pointer dimension and the file cannot disagree with itself about it.
    hparams.n_embd_out_impl = 2 * hparams.n_pointer_dim;
}

void llama_model_kev::load_arch_tensors(llama_model_loader & ml) {
    llama_model_qwen35::load_arch_tensors(ml);

    LLAMA_LOAD_LOCALS;

    const int64_t n_dp = hparams.n_pointer_dim;

    pointer_q   = create_tensor(tn(LLM_TENSOR_POINTER_Q, "weight"), {n_embd, n_dp}, 0);
    pointer_q_b = create_tensor(tn(LLM_TENSOR_POINTER_Q, "bias"),   {n_dp},         0);
    pointer_k   = create_tensor(tn(LLM_TENSOR_POINTER_K, "weight"), {n_embd, n_dp}, 0);
    pointer_k_b = create_tensor(tn(LLM_TENSOR_POINTER_K, "bias"),   {n_dp},         0);
}

std::unique_ptr<llm_graph_context> llama_model_kev::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_kev::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llama_model_qwen35::graph(model, params) {
    // The backbone graph has already narrowed to the requested rows and set `t_embd` to the
    // final-norm hidden state -- which is exactly what kev feeds its head: `self.lm` is the base
    // Qwen3.5 model and `last_hidden_state` is post-norm (kev/model.py, hidden_batch).
    ggml_tensor * h = res->t_embd;
    GGML_ASSERT(h && "kev: the backbone graph produced no hidden state to score");

    ggml_tensor * k = ggml_add(ctx0, build_lora_mm(model.pointer_k, h, nullptr), model.pointer_k_b);
    cb(k, "pointer_k", -1);

    ggml_tensor * q = ggml_add(ctx0, build_lora_mm(model.pointer_q, h, nullptr), model.pointer_q_b);
    cb(q, "pointer_q", -1);

    ggml_tensor * cur = ggml_concat(ctx0, k, q, 0);
    cb(cur, "pointer_out", -1);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}
