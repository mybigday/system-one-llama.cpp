from __future__ import annotations

from typing import Callable, Iterable

from torch import Tensor

import gguf

from .base import ModelBase
from .bert import ModernBertModel


@ModelBase.register("LayaDecisionModel")
@ModelBase.example("convaiinnovations/laya")
class LayaDecisionModel(ModernBertModel):
    """convaiinnovations/laya: a ModernBERT backbone with a trained decision head on top.

    The head is two ordinary pre-norm transformer layers over the whole sequence, a per-question-
    type embedding added before them, and a pointwise scorer that turns each position into one
    scalar. The answer for an option is that scalar at the option's mask token, so the model's
    output is one value per token rather than a distribution over the vocabulary.
    """

    model_arch = gguf.MODEL_ARCH.LAYA

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        # the "escalate" head and the fitted temperature table are not part of the answer
        if name.startswith("act_head.") or name == "temperature":
            return None

        if name.startswith("encoder."):
            return super().filter_tensors((name[8:], gen))

        # nn.MultiheadAttention stores its fused projection as in_proj_weight / in_proj_bias,
        # with no separator before the suffix the tensor map splits on
        name = name.replace("self_attn.in_proj_weight", "self_attn.in_proj.weight")
        name = name.replace("self_attn.in_proj_bias", "self_attn.in_proj.bias")
        return (name, gen)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # one scalar per token, which is what makes this a decision model rather than an encoder
        self.gguf_writer.add_embedding_length_out(1)

        self.gguf_writer.add_decision_head_block_count(self.hparams["decision_head_layers"])
        self.gguf_writer.add_decision_head_qtype_token_ids(self.hparams["qtype_token_ids"])
        self.gguf_writer.add_decision_head_qtype_token_index(self.hparams["qtype_token_index"])
        self.gguf_writer.add_decision_head_calibration_temperature(self.hparams["calibration_temperature"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith("head.") or name.startswith("scorer.") or name.startswith("type_emb"):
            yield self.map_tensor_name(name), data_torch
            return

        yield from super().modify_tensors(data_torch, name, bid)
