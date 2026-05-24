#!/usr/bin/env python3
"""Generate final_summary.md from a torch E2E optimization run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    best = state.get("best")

    lines = [
        "# Final Summary",
        "",
        f"- Entry command: `{state.get('entry_cmd', '')}`",
        f"- Objective: {state.get('objective', '')}",
        f"- Iterations recorded: {len(iterations)}",
        f"- Retained changes: {len(retained)}",
        f"- Rejected changes: {len(rejected)}",
    ]
    if best:
        lines.extend(
            [
                f"- Best retained layer: {best.get('layer')}",
                f"- Best retained gain: {best.get('delta_percent', 0):+.2f}%",
                f"- Best after metric: {best.get('after')} {best.get('unit', '')}".rstrip(),
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
