#!/usr/bin/env python3
"""Run a stale-process cleanup command before launching the model command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_shell(command: str, *, cwd: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, shell=True, text=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cleanup-cmd", required=True, help="Command that kills stale model processes")
    parser.add_argument("--entry-cmd", required=True, help="Model benchmark/training/inference command")
    parser.add_argument("--cwd", default=None, help="Optional working directory for both commands")
    parser.add_argument(
        "--strict-cleanup",
        action="store_true",
        help="Fail when the cleanup command returns nonzero. Default allows pkill-style no-match returns.",
    )
    args = parser.parse_args()

    cwd = str(Path(args.cwd).resolve()) if args.cwd else None
    print(f"[cleanup] {args.cleanup_cmd}", flush=True)
    cleanup = run_shell(args.cleanup_cmd, cwd=cwd)
    print(f"[cleanup] exit={cleanup.returncode}", flush=True)
    if args.strict_cleanup and cleanup.returncode != 0:
        return cleanup.returncode

    print(f"[entry] {args.entry_cmd}", flush=True)
    entry = run_shell(args.entry_cmd, cwd=cwd)
    return entry.returncode


if __name__ == "__main__":
    sys.exit(main())
