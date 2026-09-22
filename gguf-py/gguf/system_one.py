from __future__ import annotations

import logging
import re
from typing import Any, Iterator

from .constants import Keys

logger = logging.getLogger(__name__)

# System One metadata: what a runtime must know about a Jev-style typed-decision checkpoint and
# cannot get from the weights or from the standard GGUF keys.
#
# Deliberately small. Anything the standard metadata already carries is read from there, and
# anything about how the prompt *looks* belongs in the template rather than in a key:
#
#   the prompt template   tokenizer.chat_template.system_one  (a named chat template)
#   the mask token        tokenizer.ggml.mask_token_id
#   the leading BOS       tokenizer.ggml.add_bos_token
#   bidirectional or not  {arch}.attention.causal
#
# See docs/SYSTEM_ONE_GGUF_SPEC.md.

TEMPLATE_NAME = "system_one"
TEMPLATE_KEY = Keys.Tokenizer.CHAT_TEMPLATE_N.format(name=TEMPLATE_NAME)

SCHEMA: dict[str, str] = {
    "system_one.readout":           "str",    # letter_slot | masked_slot -- this also says where
                                              # the answer sits, so there is no second key for it
    "system_one.labels":            "array",  # the answer alphabet: option i is labelled
                                              # labels[i], and that token is what gets read
    "system_one.segment_separator": "str",    # marks the tokenizer seams in the render;
                                              # defaults to U+001E when absent
}

# Nothing is required: the readout is derived from the model's own capabilities (a
# classification head means rank_head, a bidirectional model means masked_slot, otherwise
# letter_slot), and labels and segment_separator override their defaults. A template still has
# to be there, but it is a named chat template rather than a system_one.* key.
REQUIRED: tuple[str, ...] = ()

_ESCAPES = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[nrt0\\])")


def unescape(value: str) -> str:
    """Decode the escapes a shell cannot type, leaving everything else (UTF-8 included) alone."""
    def sub(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok[0] in "xu":
            return chr(int(tok[1:], 16))
        return {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\"}[tok]
    return _ESCAPES.sub(sub, value)


def flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(f"{prefix}.{k}" if prefix else str(k), v)
    else:
        yield prefix, value


def coerce(key: str, value: Any, *, from_text: bool = False) -> Any:
    kind = SCHEMA.get(key, "str")
    if not isinstance(value, str):
        return value
    if from_text:
        value = unescape(value)
    if kind == "array":
        return [v for v in value.split(",") if v]
    return value


def write(writer, values: dict[str, Any], *, from_text: bool = False) -> int:
    """Write System One metadata onto a GGUFWriter; returns how many keys were written.

    `system_one.template` is accepted as the input spelling and stored where prompt templates
    belong: the named chat template `system_one`.
    """
    values = dict(values)
    template = values.pop("system_one.template", None)

    if template is None and not any(k in SCHEMA for k in values):
        return 0

    missing = [k for k in REQUIRED if k not in values]
    if missing:
        raise ValueError(f"System One metadata is incomplete, missing: {', '.join(missing)}")

    written = 0
    if template is not None:
        # writes tokenizer.chat_template.system_one and lists the name in
        # tokenizer.chat_templates, without touching the model's default chat template
        writer.add_chat_template([{"name": TEMPLATE_NAME, "template": template}])
        written += 1

    for key, val in values.items():
        if not key.startswith("system_one."):
            logger.warning(f"system_one: skipping key outside the namespace: {key}")
            continue
        if key not in SCHEMA:
            logger.warning(
                f"system_one: {key} is not part of the spec and the runtime ignores it. "
                f"Prompt wording belongs in the template; the mask token, BOS and attention "
                f"direction have standard keys of their own."
            )
            continue

        val = coerce(key, val, from_text=from_text)
        if isinstance(val, str):
            writer.add_string(key, val)
        elif isinstance(val, list) and all(isinstance(v, str) for v in val):
            writer.add_array(key, val)
        else:
            logger.warning(f"system_one: unsupported value type for {key}: {type(val).__name__}")
            continue
        written += 1
        logger.debug(f"system_one: {key} = {val!r}")

    return written
