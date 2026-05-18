#!/usr/bin/env python3
"""Generate LIBERO supervised data for one-head or two-head trajectory drafts."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_libero_specvla_mirror import (  # noqa: E402
    SUITE_STEP_CAPS,
    center_crop_openvla_image,
    ensure_openvla_action_prompt_token,
)
from serving.trajectory_draft_head import token_ids_to_bins  # noqa: E402
from serving.trajectory_phase import label_phase  # noqa: E402
from serving.trajectory_speculative_decoder import (  # noqa: E402
    TrajectorySpeculativeDecoder,
    capture_prefill_hidden_openvla,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_libero_trajectory_head_data")


def parse_ints(raw: str | None, default: list[int]) -> list[int]:
    if not raw:
        return default
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_model_cfg(args: argparse.Namespace, suite_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_family="openvla",
        pretrained_checkpoint=args.pretrained,
        load_in_8bit=False,
        load_in_4bit=False,
        center_crop=True,
        use_spec=False,
        parallel_draft=False,
        accept_threshold=0,
        spec_checkpoint="",
        task_suite_name=suite_name,
        num_steps_wait=args.num_steps_wait,
        num_trials_per_task=0,
        seed=args.seed,
        unnorm_key=suite_name,
    )


def add_external_paths(specvla_repo: Path) -> None:
    openvla_pkg_root = specvla_repo / "openvla"
    for path in (specvla_repo, openvla_pkg_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


@torch.no_grad()
def teacher_action(
    *,
    model,
    processor,
    image,
    task_description: str,
    unnorm_key: str,
    device: str,
    dtype: torch.dtype,
    capture_prefill: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
    prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=dtype)
    inputs = ensure_openvla_action_prompt_token(inputs, device)
    action_dim = model.get_action_dim(unnorm_key)
    prefill_hidden = None

    if capture_prefill:
        prefill_kw = {
            key: value
            for key, value in {
                "pixel_values": inputs.get("pixel_values"),
                "attention_mask": inputs.get("attention_mask"),
            }.items()
            if value is not None
        }
        prefill, prefill_h = capture_prefill_hidden_openvla(
            model,
            input_ids=inputs["input_ids"],
            device=torch.device(device),
            **prefill_kw,
        )
        llm = model.language_model
        past = prefill.past_key_values
        last_logits = prefill.logits[:, -1:, :]
        generated: list[torch.Tensor] = []
        for _ in range(action_dim):
            next_id = last_logits.argmax(dim=-1)
            generated.append(next_id)
            out = llm(input_ids=next_id, past_key_values=past, use_cache=True)
            past = out.past_key_values
            last_logits = out.logits[:, -1:, :]
        action_ids = torch.cat(generated, dim=1)[0].detach().cpu()
        prefill_hidden = prefill_h[0].detach().float().cpu()
    else:
        out = model.generate(
            inputs["input_ids"],
            pixel_values=inputs.get("pixel_values"),
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=action_dim,
            do_sample=False,
            use_cache=True,
        )
        action_ids = out[0, -action_dim:].detach().cpu()

    return action_ids, prefill_hidden, inputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LIBERO trajectory-head data with phase labels")
    parser.add_argument("--suite", default="libero_goal")
    parser.add_argument("--task-ids", default="0,1,3,5,7,9")
    parser.add_argument("--trials-per-task", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None, help="Optional cap below the LIBERO suite step limit.")
    parser.add_argument("--out-dir", default="artifacts/data/libero_goal/two_head_r1")
    parser.add_argument("--history-size", type=int, default=4)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--pretrained", default="openvla/openvla-7b-finetuned-libero-goal")
    parser.add_argument("--specvla-repo-path", default=".tmp_specvla")
    parser.add_argument("--no-prefill-hidden", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    add_external_paths(Path(args.specvla_repo_path).resolve())

    try:
        from libero.libero import benchmark  # type: ignore
    except ModuleNotFoundError:
        from libero import benchmark  # type: ignore
    from experiments.robot.libero.libero_utils import (  # type: ignore
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        quat2axisangle,
    )
    from experiments.robot.robot_utils import (  # type: ignore
        get_image_resize_size,
        get_model,
        invert_gripper_action,
        normalize_gripper_action,
        set_seed_everywhere,
    )
    from experiments.robot.openvla_utils import get_processor  # type: ignore
    from PIL import Image
    import numpy as np

    dtype = getattr(torch, args.dtype)
    model_cfg = build_model_cfg(args, args.suite)
    set_seed_everywhere(model_cfg.seed)
    model = get_model(model_cfg)
    processor = get_processor(model_cfg)
    resize_size = get_image_resize_size(model_cfg)
    decoder = TrajectorySpeculativeDecoder(model=model, device=args.device)

    task_suite = benchmark.get_benchmark_dict()[args.suite]()
    task_ids = parse_ints(args.task_ids, list(range(task_suite.n_tasks)))
    max_steps = min(SUITE_STEP_CAPS[args.suite], args.max_steps) if args.max_steps else SUITE_STEP_CAPS[args.suite]
    examples = []
    rollouts = []
    phase_counts = {"smooth": 0, "complex": 0}
    t_start = time.perf_counter()

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, model_cfg.model_family, resolution=256)
        for trial in range(args.trials_per_task):
            env.reset()
            obs = env.set_init_state(initial_states[trial])
            history_bins: list[torch.Tensor] = []
            done = False
            steps = 0
            logger.info("suite=%s task=%d trial=%d desc=%s", args.suite, task_id, trial, task_description)

            for t in range(max_steps + args.num_steps_wait):
                if t < args.num_steps_wait:
                    obs, _, _, _ = env.step(get_libero_dummy_action(model_cfg.model_family))
                    continue

                img = get_libero_image(obs, resize_size)
                image = Image.fromarray(img).convert("RGB")
                image = center_crop_openvla_image(image, enabled=True)
                action_ids, prefill_h, _inputs = teacher_action(
                    model=model,
                    processor=processor,
                    image=image,
                    task_description=task_description,
                    unnorm_key=model_cfg.unnorm_key,
                    device=args.device,
                    dtype=dtype,
                    capture_prefill=not args.no_prefill_hidden,
                )
                bins = token_ids_to_bins(action_ids, model.vocab_size).cpu()

                if len(history_bins) >= args.history_size:
                    hist = torch.stack(history_bins[-args.history_size :]).short()
                    phase = label_phase(hist)
                    rec = {
                        "history_bins": hist,
                        "target_bins": bins.short(),
                        "phase_label": phase,
                        "suite": args.suite,
                        "task_id": task_id,
                        "task": task_description,
                        "trial": trial,
                        "timestep": steps,
                    }
                    if prefill_h is not None:
                        rec["prefill_hidden"] = prefill_h.to(torch.bfloat16)
                    examples.append(rec)
                    phase_counts[phase] += 1

                history_bins.append(bins)
                action = decoder.decode_action_ids(action_ids, model_cfg.unnorm_key)
                action = normalize_gripper_action(action, binarize=True)
                action = invert_gripper_action(action)
                obs, _, done, _ = env.step(action.tolist())
                steps += 1
                if done:
                    break

            rollouts.append(
                {
                    "suite": args.suite,
                    "task_id": task_id,
                    "trial": trial,
                    "task": task_description,
                    "success": bool(done),
                    "steps": steps,
                }
            )
        env.close()

    payload = {
        "examples": examples,
        "history_size": args.history_size,
        "rollouts": rollouts,
        "unnorm_key": args.suite,
        "use_prefill_hidden": not args.no_prefill_hidden,
        "llm_hidden_size": int(model.language_model.config.hidden_size),
        "data_source": "libero_teacher_rollouts",
        "phase_counts": phase_counts,
    }
    torch.save(payload, out_dir / "dataset.pt")
    summary = {
        "examples": len(examples),
        "phase_counts": phase_counts,
        "rollouts": len(rollouts),
        "successes": sum(1 for r in rollouts if r["success"]),
        "elapsed_s": time.perf_counter() - t_start,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Saved %d examples to %s", len(examples), out_dir / "dataset.pt")


if __name__ == "__main__":
    main()
