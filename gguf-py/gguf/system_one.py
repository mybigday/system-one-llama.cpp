from __future__ import annotations

import logging
import re
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# System One metadata: the keys a Jev-style typed-decision checkpoint carries in its GGUF, so
# that a runtime can build the prompt and read the answers without being told out of band.
#
# The values can come from a `system_one.json` sidecar (our export convention), from
# convert_hf_to_gguf.py's --system-one KEY=VALUE, or be written into an existing GGUF with
# gguf_set_system_one.py. The runtime only ever reads the kv; the sidecar is never required.
#
# The full semantics are in docs/SYSTEM_ONE_GGUF_SPEC.md.
SCHEMA: dict[str, str] = {
    "system_one.version":                    "int",
    "system_one.template":                   "str",    # Jinja, rendered with trim_blocks and
                                                       # lstrip_blocks (the chat-template convention)
    "system_one.template.segment_separator": "str",    # marks the tokenizer seams in the render
    "system_one.readout":                    "str",    # letter_slot | masked_slot
    "system_one.slot":                       "str",    # last_token_of_question_segment |
                                                       # mask_token_per_question
    "system_one.letters":                    "str",
    "system_one.mask_token_id":              "int",    # required by masked_slot
    "system_one.mask_token":                 "str",    # how the template writes it
    "system_one.attention":                  "str",    # "bidirectional" also sets attention.causal
    "system_one.noul_options":               "array",
    "system_one.kind_tag.choice":            "str",
    "system_one.kind_tag.noul":              "str",
    "system_one.kind_tag.score":             "str",
    "system_one.max_state_tokens":           "int",
    "system_one.truncation":                 "str",
    "system_one.truncation_marker":          "str",
    "system_one.bos":                        "bool",
    "system_one.calibration_temperature":    "float",  # provenance: already folded into the weights
    "system_one.calibration_folded":         "bool",
    "system_one.source_run":                 "str",
}

# describe the format for a reader, mean nothing to a runtime, and are not written
DOC_ONLY = {"system_one.template.render", "system_one.template.vars"}

REQUIRED = ("system_one.template", "system_one.readout", "system_one.slot")

_ESCAPES = re.compile(r"\\(x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|[nrt0\\])")


def unescape(value: str) -> str:
    """Decode the escapes a shell cannot type, leaving everything else (UTF-8 included) alone."""
    def sub(m: re.Match[str]) -> str:
        tok = m.group(1)
        if tok[0] == "x" or tok[0] == "u":
            return chr(int(tok[1:], 16))
        return {"n": "\n", "r": "\r", "t": "\t", "0": "\0", "\\": "\\"}[tok]
    return _ESCAPES.sub(sub, value)


def flatten(prefix: str, value: Any) -> Iterator[tuple[str, Any]]:
    """A sidecar may nest (kind_tag: {choice: ...}); the kv namespace is flat."""
    if isinstance(value, dict):
        for k, v in value.items():
            yield from flatten(f"{prefix}.{k}" if prefix else str(k), v)
    else:
        yield prefix, value


def coerce(key: str, value: Any, *, from_text: bool = False) -> Any:
    kind = SCHEMA.get(key)
    if kind is None:
        logger.warning(f"system_one: {key} is not in the spec, treating it as a string")
        kind = "str"
    if not isinstance(value, str):
        return value
    if from_text:
        value = unescape(value)
    if kind == "int":
        return int(value, 0)
    if kind == "float":
        return float(value)
    if kind == "bool":
        low = value.lower()
        if low in ("true", "1", "yes", "on"):
            return True
        if low in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"{key}: expected a boolean, got {value!r}")
    if kind == "array":
        return [v for v in value.split(",") if v]
    return value


def check_required(values: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if k not in values]
    if missing:
        raise ValueError(f"System One metadata is incomplete, missing: {', '.join(missing)}")


def write(writer, values: dict[str, Any], *, from_text: bool = False) -> int:
    """Write the kv onto a GGUFWriter. Returns how many keys were written.

    A bidirectional readout also sets the standard attention.causal key, because that is what
    the loader reads -- system_one.attention alone would leave the model running causally.
    """
    check_required(values)

    if values.get("system_one.attention") == "bidirectional":
        logger.info("system_one: bidirectional attention -> attention.causal = False")
        writer.add_causal_attention(False)

    written = 0
    for key, val in values.items():
        if key in DOC_ONLY:
            continue
        if not key.startswith("system_one."):
            logger.warning(f"system_one: skipping key outside the namespace: {key}")
            continue

        val = coerce(key, val, from_text=from_text)
        if isinstance(val, bool):
            writer.add_bool(key, val)
        elif isinstance(val, int):
            writer.add_uint32(key, val)
        elif isinstance(val, float):
            writer.add_float32(key, val)
        elif isinstance(val, str):
            writer.add_string(key, val)
        elif isinstance(val, list) and all(isinstance(v, str) for v in val):
            writer.add_array(key, val)
        else:
            logger.warning(f"system_one: unsupported value type for {key}: {type(val).__name__}")
            continue
        written += 1
        logger.debug(f"system_one: {key} = {val!r}")

    return written
