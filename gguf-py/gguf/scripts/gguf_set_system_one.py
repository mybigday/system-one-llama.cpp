#!/usr/bin/env python3
"""Write System One metadata into an existing GGUF.

For checkpoints that were converted before their System One metadata existed, or that came from
somewhere else entirely (Laya, jwenv, a scoring-head export). The runtime reads only the kv, so
a file patched this way is indistinguishable from one converted with a sidecar.

    gguf_set_system_one.py in.gguf out.gguf \\
        --system-one-template template.jinja \\
        --set system_one.readout=letter_slot \\
        --set system_one.slot=last_token_of_question_segment \\
        --set 'system_one.template.segment_separator=\\x1e'

or copy the values from a sidecar:

    gguf_set_system_one.py in.gguf out.gguf --json system_one.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

if "NO_LOCAL_GGUF" not in __import__("os").environ:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import gguf  # noqa: E402

logger = logging.getLogger("gguf-set-system-one")


def copy_file_with_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter, values: dict[str, Any]) -> int:
    for field in reader.fields.values():
        # GGUF.* are the reader's synthetic header fields and general.architecture is written by
        # the writer itself; system_one.* are what this tool replaces
        if field.name.startswith(("GGUF.", "system_one.")) or field.name == "general.architecture":
            continue
        writer.add_key_value(field.name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == gguf.GGUFValueType.ARRAY else None)

    n = gguf.system_one.write(writer, values, from_text=True)

    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, raw_dtype=tensor.tensor_type)
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="write System One metadata into an existing GGUF")
    parser.add_argument("input", type=Path, help="GGUF to read")
    parser.add_argument("output", type=Path, help="GGUF to write")
    parser.add_argument("--json", type=Path, help="sidecar to take the values from")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="set one key; repeatable, and overrides --json")
    parser.add_argument("--system-one-template", type=Path, metavar="FILE",
                        help="file holding the Jinja template (system_one.template)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    values: dict[str, Any] = {}
    if args.json:
        with open(args.json, encoding="utf-8") as f:
            values.update(dict(gguf.system_one.flatten("", json.load(f))))
    for item in args.set:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        values[k.strip()] = v
    if args.system_one_template:
        values["system_one.template"] = args.system_one_template.read_text(encoding="utf-8").rstrip("\n")

    if not values:
        raise SystemExit("nothing to write: pass --json, --set or --system-one-template")

    reader = gguf.GGUFReader(args.input, "r")
    arch = reader.fields["general.architecture"].contents()
    writer = gguf.GGUFWriter(args.output, arch, endianess=reader.endianess)

    n = copy_file_with_metadata(reader, writer, values)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    writer.close()

    logger.info(f"wrote {n} System One key(s) into {args.output}")


if __name__ == "__main__":
    main()
