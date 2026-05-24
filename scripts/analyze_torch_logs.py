#!/usr/bin/env python3
"""Summarize common torch.compile log signals."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "graph_break_mentions": re.compile(r"graph break|Graph break|graph_break", re.I),
    "recompile_mentions": re.compile(r"recompil|Recompil|guard failure|guard_fail", re.I),
    "dynamic_shape_mentions": re.compile(r"dynamic|mark_dynamic|symbolic", re.I),
    "inductor_mentions": re.compile(r"inductor|TorchInductor", re.I),
    "triton_mentions": re.compile(r"triton|Triton", re.I),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", help="Torch log file")
    parser.add_argument("--out", default=None, help="Optional JSON output")
    parser.add_argument("--markdown", default=None, help="Optional markdown output")
    args = parser.parse_args()

    text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    result = {
        "log": args.log,
        "lines": text.count("\n") + 1,
        "counts": {name: len(pattern.findall(text)) for name, pattern in PATTERNS.items()},
    }

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        lines = ["# Torch Compile Log Summary", "", "| Signal | Count |", "|---|---:|"]
        for key, value in result["counts"].items():
            lines.append(f"| {key} | {value} |")
        lines.append("")
        Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
