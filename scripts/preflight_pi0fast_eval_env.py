#!/usr/bin/env python3
"""Preflight checks for real PI0-FAST LIBERO eval runs."""

from __future__ import annotations

import argparse
import ctypes.util
from importlib import metadata
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Callable


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def distribution_available(name: str) -> bool:
    try:
        metadata.version(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _version_tuple(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups(default="0"))


def _lt_version(value: str | None, threshold: tuple[int, ...]) -> bool:
    parsed = _version_tuple(value)
    return parsed is not None and parsed < threshold


def _ge_version(value: str | None, threshold: tuple[int, ...]) -> bool:
    parsed = _version_tuple(value)
    return parsed is not None and parsed >= threshold


def dependency_available(
    name: str,
    *,
    module_checker: Callable[[str], bool] = module_available,
    distribution_checker: Callable[[str], bool] = distribution_available,
) -> bool:
    if name.startswith("dist:"):
        return bool(distribution_checker(name.split(":", 1)[1]))
    return bool(module_checker(name))


def torch_cuda_summary(device: str) -> dict[str, Any]:
    if not module_available("torch"):
        return {"required": device.startswith("cuda"), "available": False, "device_count": 0, "error": "torch missing"}
    try:
        import torch

        return {
            "required": device.startswith("cuda"),
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "devices": [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())],
        }
    except Exception as exc:  # noqa: BLE001
        return {"required": device.startswith("cuda"), "available": False, "device_count": 0, "error": repr(exc)}


def collect_preflight(
    *,
    device: str,
    modules: list[str],
    eval_script: Path,
    gate_script: Path,
    require_hf_token: bool,
    module_checker: Callable[[str], bool] = module_available,
    distribution_checker: Callable[[str], bool] = distribution_available,
    version_provider: Callable[[str], str | None] = distribution_version,
    library_finder: Callable[[str], str | None] = ctypes.util.find_library,
    cuda_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cuda = cuda_summary if cuda_summary is not None else torch_cuda_summary(device)
    module_rows = {
        name: dependency_available(
            name,
            module_checker=module_checker,
            distribution_checker=distribution_checker,
        )
        for name in modules
    }
    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    mujoco_gl = os.environ.get("MUJOCO_GL", "")
    runtime_versions = {
        name: version_provider(name)
        for name in ("numpy", "networkx", "mujoco", "robosuite")
    }
    gl_library = ""
    mujoco_gl_check = True
    if mujoco_gl.lower() == "osmesa":
        gl_library = library_finder("OSMesa") or ""
        mujoco_gl_check = bool(gl_library)
    elif mujoco_gl.lower() == "egl":
        gl_library = library_finder("EGL") or ""
        mujoco_gl_check = bool(gl_library)
    networkx_numpy_compat = not (
        _ge_version(runtime_versions.get("numpy"), (1, 24, 0))
        and _lt_version(runtime_versions.get("networkx"), (3, 0, 0))
    )
    mujoco_robosuite_compat = not (
        _lt_version(runtime_versions.get("robosuite"), (1, 5, 0))
        and _ge_version(runtime_versions.get("mujoco"), (3, 0, 0))
    )
    checks = {
        "eval_script_exists": eval_script.exists(),
        "gate_script_exists": gate_script.exists(),
        "required_modules": all(module_rows.values()),
        "cuda": (not cuda.get("required")) or (bool(cuda.get("available")) and int(cuda.get("device_count", 0)) > 0),
        "hf_token": hf_token_present or not require_hf_token,
        "mujoco_gl": mujoco_gl_check,
        "networkx_numpy_compat": networkx_numpy_compat,
        "mujoco_robosuite_compat": mujoco_robosuite_compat,
    }
    warnings = []
    if not mujoco_gl:
        warnings.append("MUJOCO_GL is not set; headless LIBERO runs usually use MUJOCO_GL=osmesa")
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "modules": module_rows,
        "cuda": cuda,
        "hf_token_present": hf_token_present,
        "mujoco_gl": mujoco_gl,
        "mujoco_gl_library": gl_library,
        "runtime_versions": runtime_versions,
        "warnings": warnings,
    }


def format_text(summary: dict[str, Any]) -> str:
    status = "PASS" if summary["passed"] else "FAIL"
    lines = [f"PI0-FAST eval preflight: {status}", "", "Checks:"]
    for name, passed in summary["checks"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.append("")
    lines.append("Modules:")
    for name, available in summary["modules"].items():
        lines.append(f"- {name}: {'ok' if available else 'missing'}")
    cuda = summary["cuda"]
    lines.append("")
    lines.append(f"CUDA: available={cuda.get('available')} count={cuda.get('device_count')}")
    if cuda.get("devices"):
        for idx, name in enumerate(cuda["devices"]):
            lines.append(f"- cuda:{idx}: {name}")
    versions = summary.get("runtime_versions") or {}
    if versions:
        lines.append("")
        lines.append("Runtime versions:")
        for name, version in versions.items():
            lines.append(f"- {name}: {version or 'unknown'}")
    if summary.get("mujoco_gl"):
        lines.append("")
        lines.append(f"MUJOCO_GL: {summary.get('mujoco_gl')} library={summary.get('mujoco_gl_library') or 'not found'}")
    if summary["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in summary["warnings"])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether this environment can run PI0-FAST LIBERO evals.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-script", type=Path, default=Path("scripts/run_pi0fast_chunk_eval.py"))
    parser.add_argument("--gate-script", type=Path, default=Path("scripts/gate_pi0fast_target_eos.py"))
    parser.add_argument(
        "--modules",
        default="torch,lerobot,libero,robosuite,mujoco,networkx,dist:hf_libero",
        help="Comma-separated import modules, with dist:NAME for installed distribution checks.",
    )
    parser.add_argument("--require-hf-token", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    modules = [part.strip() for part in args.modules.split(",") if part.strip()]
    summary = collect_preflight(
        device=args.device,
        modules=modules,
        eval_script=args.eval_script,
        gate_script=args.gate_script,
        require_hf_token=args.require_hf_token,
    )
    print(json.dumps(summary, indent=2) if args.json else format_text(summary))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
