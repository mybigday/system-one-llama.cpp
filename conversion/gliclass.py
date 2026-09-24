from __future__ import annotations

from typing import Callable, Iterable

from torch import Tensor

import gguf

from .base import ModelBase
from .bert import ModernBertModel


@ModelBase.register("GLiClassModel")
@ModelBase.example("heman10x/rlcd-modernbert-151m")
class GLiClassModel(ModernBertModel):
    """GLiClass uni-encoder: an encoder with a decision head that scores marked positions.

    Each candidate is marked in the prompt by the class token, and its score is the dot product of
    two projections -- the marked position's, and the pooled text's. So the model answers with one
    number per token, and the answer to a question is the numbers at its own marks.
    """

    model_arch = gguf.MODEL_ARCH.GLICLASS

    def __init__(self, dir_model, *args, **kwargs):
        # The backbone's shape is under encoder_config. Lift it before the base class reads the
        # block count, the way a text_config is lifted, so everything downstream -- rope, the
        # tensor map, the layer loop -- sees an ordinary ModernBERT config.
        if kwargs.get("hparams") is None:
            outer = ModelBase.load_hparams(dir_model, False)
            kwargs["hparams"] = {**outer, **outer["encoder_config"], "gliclass_config": outer}
        super().__init__(dir_model, *args, **kwargs)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.startswith("model."):
            name = name[6:]

        # only used when normalize_features is set, and this checkpoint does not set it
        if name == "logit_scale":
            return None

        if name.startswith("encoder_model."):
            return super().filter_tensors((name[len("encoder_model."):], gen))

        return (name, gen)

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # one scalar per token: the dot product of the two projections
        self.gguf_writer.add_embedding_length_out(1)

        # where an answer is read. The mark is this architecture's, not the tokenizer's mask.
        self.gguf_writer.add_decision_head_slot_token_id(self.hparams["gliclass_config"]["class_token_index"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.startswith(("classes_projector.", "text_projector.")):
            yield self.map_tensor_name(name), data_torch
            return

        yield from super().modify_tensors(data_torch, name, bid)
