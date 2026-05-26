#!/usr/bin/env python3
"""Archive scoped git diffs for an optimization iteration."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run_git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def list_untracked(repo: Path, pathspec: list[str]) -> list[str]:
    args = ["ls-files", "--others", "--exclude-standard"]
    if pathspec:
        args.extend(["--", *pathspec])
    proc = run_git(repo, args)
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def untracked_patch(repo: Path, files: list[str]) -> str:
    chunks = []
    for file in files:
        proc = run_git(repo, ["diff", "--no-index", "--", "/dev/null", file])
        if proc.stdout:
            chunks.append(proc.stdout)
    return "\n".join(chunks)


def safe_label(value: str) -> str:
    keep = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "."}:
            keep.append(char)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "changes"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo", default=".")
    parser.add_argument("--iter", type=int, default=None)
    parser.add_argument("--layer", default="")
    parser.add_argument("--decision", default="unknown", choices=["retained", "rejected", "optional", "blocked", "unknown"])
    parser.add_argument("--label", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--pathspec", action="append", default=[], help="Limit archive to this pathspec; repeatable")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    repo = Path(args.repo).resolve()
    patch_dir = run_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    if args.label:
        label = safe_label(args.label)
    elif args.iter is not None:
        label = safe_label(f"iter{args.iter:02d}_{args.layer or 'layer'}_{args.decision}")
    else:
        label = safe_label(f"{args.layer or 'changes'}_{args.decision}")

    pathspec = ["--", *args.pathspec] if args.pathspec else []
    diff = run_git(repo, ["diff", *pathspec])
    files = run_git(repo, ["diff", "--name-only", *pathspec])
    untracked = list_untracked(repo, args.pathspec)
    full_patch = diff.stdout
    if untracked:
        full_patch = "\n".join(part for part in [full_patch, untracked_patch(repo, untracked)] if part)

    patch_path = patch_dir / f"{label}.patch"
    files_path = patch_dir / f"{label}.files.txt"
    meta_path = patch_dir / f"{label}.json"

    patch_path.write_text(full_patch, encoding="utf-8")
    changed_files = [line for line in files.stdout.splitlines() if line.strip()]
    files_path.write_text("\n".join([*changed_files, *untracked]) + ("\n" if changed_files or untracked else ""), encoding="utf-8")
    metadata = {
        "label": label,
        "iter": args.iter,
        "layer": args.layer,
        "decision": args.decision,
        "note": args.note,
        "repo": str(repo),
        "pathspec": args.pathspec,
        "patch": str(patch_path),
        "files": str(files_path),
        "untracked_files": untracked,
        "git_diff_returncode": diff.returncode,
        "git_diff_stderr": diff.stderr.strip(),
        "git_files_returncode": files.returncode,
        "git_files_stderr": files.stderr.strip(),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(patch_path)
    return 0 if diff.returncode == 0 and files.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
