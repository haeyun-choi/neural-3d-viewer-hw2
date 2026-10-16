#!/usr/bin/env python3
"""Repack a single-file Spark RAD asset into the chunked .rad + .radc layout.

Spark can page a level-of-detail tree in two ways (see docs/hosted-scenes.md):

* a single ``scene-lod.rad`` read with HTTP Range requests, and
* a small ``scene-lod.rad`` header beside ``scene-lod-0.radc``, ``-1.radc`` …
  chunk files, which needs no Range support at all.

``build-lod --rad-chunked`` produces the second layout directly and is the
supported way to create one. This script exists for the case where you already
have the single-file asset and the host cannot serve byte ranges: it rewrites
only the header's chunk table and copies the chunk payloads out byte for byte,
so the splat data is exactly what ``build-lod`` produced.

Container layout, from Spark's rust/spark-lib/src/rad.rs::

    u32  magic        "RAD0" little-endian (0x30444152)
    u32  meta_bytes   length of the JSON metadata that follows
    ...  meta         JSON, padded with zeros to an 8-byte boundary
    ...  chunks       concatenated chunk payloads, each starting with "RADC"

Usage::

    python scripts/rad_chunk.py scene-lod.rad --output out/
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

RAD_MAGIC = b"RAD0"
RADC_MAGIC = b"RADC"
COPY_BLOCK = 8 * 1024 * 1024


def read_header(handle) -> tuple[dict, int]:
    """Return the decoded metadata and the offset the chunk payloads start at."""
    prefix = handle.read(8)
    if len(prefix) < 8 or prefix[:4] != RAD_MAGIC:
        raise SystemExit("Not a Spark RAD asset: the RAD0 signature is missing.")
    meta_bytes = int.from_bytes(prefix[4:8], "little")
    meta = json.loads(handle.read(meta_bytes))
    padded = (meta_bytes + 7) & ~7
    return meta, 8 + padded


def write_header(path: pathlib.Path, meta: dict) -> None:
    payload = json.dumps(meta, indent=2).encode() + b"\n"
    padding = ((len(payload) + 7) & ~7) - len(payload)
    with path.open("wb") as out:
        out.write(RAD_MAGIC)
        out.write(len(payload).to_bytes(4, "little"))
        out.write(payload)
        out.write(b"\0" * padding)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", type=pathlib.Path, help="single-file *-lod.rad produced by build-lod")
    parser.add_argument("--output", type=pathlib.Path, required=True, help="directory to write the chunked layout into")
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    stem = arguments.asset.stem
    with arguments.asset.open("rb") as source:
        meta, chunks_start = read_header(source)
        chunks = meta.get("chunks") or []
        if not chunks:
            raise SystemExit("This RAD asset declares no chunks; nothing to split.")
        if chunks[0].get("filename"):
            raise SystemExit("This RAD asset is already in the chunked layout.")

        for index, chunk in enumerate(chunks):
            name = f"{stem}-{index}.radc"
            source.seek(chunks_start + int(chunk["offset"]))
            remaining = int(chunk["bytes"])
            target = arguments.output / name
            with target.open("wb") as out:
                first = source.read(min(COPY_BLOCK, remaining))
                if first[:4] != RADC_MAGIC:
                    raise SystemExit(f"Chunk {index} does not start with the RADC signature; the header is inconsistent.")
                out.write(first)
                remaining -= len(first)
                while remaining > 0:
                    block = source.read(min(COPY_BLOCK, remaining))
                    if not block:
                        raise SystemExit(f"Chunk {index} is truncated in the source asset.")
                    out.write(block)
                    remaining -= len(block)
            chunk["filename"] = name
            chunk["offset"] = 0

    write_header(arguments.output / f"{stem}.rad", meta)
    print(f"{len(chunks)} chunks written to {arguments.output}/ beside {stem}.rad")
    return 0


if __name__ == "__main__":
    sys.exit(main())
