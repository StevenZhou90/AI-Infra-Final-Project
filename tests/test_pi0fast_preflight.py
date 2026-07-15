from __future__ import annotations

import os
from pathlib import Path

from scripts.preflight_pi0fast_eval_env import collect_preflight


def test_preflight_passes_with_modules_and_cuda(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")

    summary = collect_preflight(
        device="cuda",
        modules=["torch", "lerobot", "libero"],
        eval_script=eval_script,
        gate_script=gate_script,
        require_hf_token=False,
        module_checker=lambda _name: True,
        cuda_summary={"required": True, "available": True, "device_count": 1, "devices": ["GPU"]},
    )

    assert summary["passed"] is True
    assert all(summary["checks"].values())


def test_preflight_fails_missing_cuda_and_module(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")

    summary = collect_preflight(
        device="cuda",
        modules=["torch", "lerobot"],
        eval_script=eval_script,
        gate_script=gate_script,
        require_hf_token=False,
        module_checker=lambda name: name == "torch",
        cuda_summary={"required": True, "available": False, "device_count": 0},
    )

    assert summary["passed"] is False
    assert summary["checks"]["required_modules"] is False
    assert summary["checks"]["cuda"] is False


def test_preflight_accepts_distribution_checks(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")

    summary = collect_preflight(
        device="cuda",
        modules=["torch", "dist:hf_libero"],
        eval_script=eval_script,
        gate_script=gate_script,
        require_hf_token=False,
        module_checker=lambda name: name == "torch",
        distribution_checker=lambda name: name == "hf_libero",
        cuda_summary={"required": True, "available": True, "device_count": 1},
    )

    assert summary["passed"] is True
    assert summary["modules"]["dist:hf_libero"] is True


def test_preflight_fails_missing_osmesa_library(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")
    old = os.environ.get("MUJOCO_GL")
    os.environ["MUJOCO_GL"] = "osmesa"
    try:
        summary = collect_preflight(
            device="cuda",
            modules=[],
            eval_script=eval_script,
            gate_script=gate_script,
            require_hf_token=False,
            cuda_summary={"required": True, "available": True, "device_count": 1},
            library_finder=lambda _name: None,
            version_provider=lambda _name: None,
        )
    finally:
        if old is None:
            os.environ.pop("MUJOCO_GL", None)
        else:
            os.environ["MUJOCO_GL"] = old

    assert summary["passed"] is False
    assert summary["checks"]["mujoco_gl"] is False


def test_preflight_fails_incompatible_runtime_versions(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")
    versions = {
        "numpy": "1.26.4",
        "networkx": "2.4",
        "mujoco": "3.10.0",
        "robosuite": "1.4.0",
    }

    summary = collect_preflight(
        device="cuda",
        modules=[],
        eval_script=eval_script,
        gate_script=gate_script,
        require_hf_token=False,
        cuda_summary={"required": True, "available": True, "device_count": 1},
        version_provider=lambda name: versions.get(name),
    )

    assert summary["passed"] is False
    assert summary["checks"]["networkx_numpy_compat"] is False
    assert summary["checks"]["mujoco_robosuite_compat"] is False


def test_preflight_cpu_does_not_require_cuda(tmp_path: Path) -> None:
    eval_script = tmp_path / "eval.py"
    gate_script = tmp_path / "gate.py"
    eval_script.write_text("")
    gate_script.write_text("")

    summary = collect_preflight(
        device="cpu",
        modules=[],
        eval_script=eval_script,
        gate_script=gate_script,
        require_hf_token=False,
        module_checker=lambda _name: False,
        cuda_summary={"required": False, "available": False, "device_count": 0},
    )

    assert summary["passed"] is True
    assert summary["checks"]["cuda"] is True
