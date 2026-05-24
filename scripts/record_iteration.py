#!/usr/bin/env python3
"""Append an optimization iteration to the run ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def esc(value: object) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def calc_delta(before: float, after: float, direction: str) -> float:
    if before == 0:
        return 0.0
    if direction in {"throughput", "higher-is-better"}:
        return (after / before - 1.0) * 100.0
    if direction in {"latency", "lower-is-better"}:
        return (before / after - 1.0) * 100.0 if after != 0 else 0.0
    raise ValueError(f"unknown direction: {direction}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iter", required=True, type=int)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--changed", default="")
    parser.add_argument("--files", default="")
    parser.add_argument("--correctness", default="unknown")
    parser.add_argument("--before", required=True, type=float)
    parser.add_argument("--after", required=True, type=float)
    parser.add_argument("--unit", default="")
    parser.add_argument(
        "--direction",
        default="throughput",
        choices=["throughput", "latency", "higher-is-better", "lower-is-better"],
    )
    parser.add_argument("--retained", default="unknown", choices=["yes", "no", "optional", "unknown"])
    parser.add_argument("--evidence", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    delta = calc_delta(args.before, args.after, args.direction)
    before_text = f"{args.before:g} {args.unit}".strip()
    after_text = f"{args.after:g} {args.unit}".strip()
    row = (
        f"| {args.iter} | {esc(args.layer)} | {esc(args.hypothesis)} | {esc(args.changed)} | "
        f"{esc(args.files)} | {esc(args.correctness)} | {esc(before_text)} | {esc(after_text)} | "
        f"{delta:+.2f}% | {esc(args.retained)} | {esc(args.evidence)} |\n"
    )
    table_path = run_dir / "iteration_table.md"
    if not table_path.exists():
        table_path.write_text(
            "| Iter | Layer | Hypothesis | Code/config changed | Files changed | Correctness | E2E metric before | E2E metric after | Delta | Retained | Evidence |\n"
            "|---:|---|---|---|---|---|---:|---:|---:|---|---|\n",
            encoding="utf-8",
        )
    with table_path.open("a", encoding="utf-8") as handle:
        handle.write(row)

    state_path = run_dir / "state.json"
    state = {}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    state.setdefault("iterations", []).append(
        {
            "iter": args.iter,
            "layer": args.layer,
            "hypothesis": args.hypothesis,
            "changed": args.changed,
            "files": args.files,
            "correctness": args.correctness,
            "before": args.before,
            "after": args.after,
            "unit": args.unit,
            "direction": args.direction,
            "delta_percent": delta,
            "retained": args.retained,
            "evidence": args.evidence,
        }
    )
    if args.retained == "yes":
        best = state.get("best")
        if best is None or delta >= float(best.get("delta_percent", -1e9)):
            state["best"] = state["iterations"][-1]
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{delta:+.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
