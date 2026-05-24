#!/usr/bin/env python3
"""Collect a compact PyTorch/accelerator environment snapshot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], timeout: int = 10) -> dict[str, object]:
    if shutil.which(cmd[0]) is None:
        return {"available": False}
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        return {"available": True, "error": repr(exc)}
    return {
        "available": True,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip()[-12000:],
        "stderr": proc.stderr.strip()[-4000:],
    }


def collect() -> dict[str, object]:
    data: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "env": {
            key: os.environ.get(key)
            for key in [
                "CUDA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "TORCH_LOGS",
                "TORCH_TRACE",
                "TORCH_COMPILE_DEBUG",
                "TORCHINDUCTOR_CACHE_DIR",
                "TRITON_CACHE_DIR",
            ]
            if os.environ.get(key) is not None
        },
    }
    try:
        import torch

        data["torch"] = {
            "version": torch.__version__,
            "cuda_version": getattr(torch.version, "cuda", None),
            "hip_version": getattr(torch.version, "hip", None),
            "debug": torch.version.debug,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        if torch.cuda.is_available():
            devices = []
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                devices.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory": props.total_memory,
                        "major": props.major,
                        "minor": props.minor,
                        "multi_processor_count": props.multi_processor_count,
                    }
                )
            data["torch"]["cuda_devices"] = devices
        try:
            import torch._inductor.config as inductor_config

            data["torch_inductor"] = {
                "max_autotune": getattr(inductor_config, "max_autotune", None),
                "triton_cudagraphs": getattr(getattr(inductor_config, "triton", object()), "cudagraphs", None),
            }
        except Exception as exc:
            data["torch_inductor"] = {"error": repr(exc)}
    except Exception as exc:
        data["torch"] = {"error": repr(exc)}

    try:
        import triton

        data["triton"] = {"version": getattr(triton, "__version__", "unknown")}
    except Exception as exc:
        data["triton"] = {"error": repr(exc)}

    data["tools"] = {
        "nvidia-smi": run_cmd(["nvidia-smi", "-L"]),
        "rocm-smi": run_cmd(["rocm-smi", "--showproductname"]),
        "rocminfo": run_cmd(["rocminfo"], timeout=15),
        "hipprof": run_cmd(["hipprof", "--version"]),
        "nsys": run_cmd(["nsys", "--version"]),
        "ncu": run_cmd(["ncu", "--version"]),
    }
    return data


def write_markdown(data: dict[str, object], path: Path) -> None:
    torch_info = data.get("torch", {})
    lines = [
        "# Environment",
        "",
        f"- Python: `{data.get('python', '').splitlines()[0]}`",
        f"- Platform: `{data.get('platform')}`",
        f"- Torch: `{torch_info.get('version') if isinstance(torch_info, dict) else 'unknown'}`",
        f"- CUDA: `{torch_info.get('cuda_version') if isinstance(torch_info, dict) else None}`",
        f"- HIP: `{torch_info.get('hip_version') if isinstance(torch_info, dict) else None}`",
        f"- Triton: `{(data.get('triton') or {}).get('version') if isinstance(data.get('triton'), dict) else 'unknown'}`",
        "",
        "## Raw JSON",
        "",
        "```json",
        json.dumps(data, indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="env.json", help="JSON output path")
    parser.add_argument("--markdown", default=None, help="Optional markdown output path")
    args = parser.parse_args()

    data = collect()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        write_markdown(data, Path(args.markdown))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
