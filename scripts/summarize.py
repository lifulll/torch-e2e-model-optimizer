#!/usr/bin/env python3
"""Generate final_summary.md from a torch E2E optimization run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def calc_total_gain(iterations: list[dict[str, object]]) -> tuple[float, dict[str, object]] | None:
    retained = [item for item in iterations if item.get("retained") == "yes"]
    if not retained:
        return None
    first = iterations[0]
    last = retained[-1]
    if first.get("direction") != last.get("direction") or first.get("unit") != last.get("unit"):
        return None
    before = float(first.get("before", 0.0))
    after = float(last.get("after", 0.0))
    if before == 0:
        return None
    if first.get("direction") in {"throughput", "higher-is-better"}:
        gain = (after / before - 1.0) * 100.0
    elif first.get("direction") in {"latency", "lower-is-better"}:
        gain = (before / after - 1.0) * 100.0 if after else 0.0
    else:
        return None
    return gain, last


def trailing_no_gain_count(iterations: list[dict[str, object]]) -> int:
    count = 0
    for item in reversed(iterations):
        if item.get("material") is True:
            break
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--stop-reason", default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    state_path = run_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    table = (run_dir / "iteration_table.md").read_text(encoding="utf-8") if (run_dir / "iteration_table.md").exists() else ""
    iterations = state.get("iterations", [])
    retained = [item for item in iterations if item.get("retained") == "yes"]
    rejected = [item for item in iterations if item.get("retained") == "no"]
    sub_threshold = [item for item in iterations if item.get("material") is False]
    no_gain_tail = trailing_no_gain_count(iterations)
    total_gain = calc_total_gain(iterations)
    best = state.get("best")

    lines = [
        "# Final Summary",
        "",
        f"- Entry command: `{state.get('entry_cmd', '')}`",
        f"- Cleanup command: `{state.get('cleanup_cmd', '')}`",
        f"- Objective: {state.get('objective', '')}",
        f"- Iterations recorded: {len(iterations)}",
        f"- Retained changes: {len(retained)}",
        f"- Rejected changes: {len(rejected)}",
        f"- Sub-threshold iterations: {len(sub_threshold)}",
        f"- Consecutive no-gain iterations at end: {no_gain_tail}",
    ]
    if best:
        lines.extend(
            [
                f"- Best retained layer: {best.get('layer')}",
                f"- Best retained gain: {best.get('delta_percent', 0):+.2f}%",
                f"- Best after metric: {best.get('after')} {best.get('unit', '')}".rstrip(),
            ]
        )
    if total_gain:
        gain, last = total_gain
        lines.extend(
            [
                f"- Total E2E gain from first measured baseline: {gain:+.2f}%",
                f"- Final retained metric: {last.get('after')} {last.get('unit', '')}".rstrip(),
            ]
        )
    if args.stop_reason:
        lines.append(f"- Stop reason: {args.stop_reason}")
    lines.extend(["", "## Iteration Table", "", table.strip(), ""])

    out = Path(args.out) if args.out else run_dir / "final_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
