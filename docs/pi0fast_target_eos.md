# pi0fast target eos

We tested exact early stop for pi0fast FAST-token decode. The baseline decodes
to the fixed FAST token budget. `target_eos` stops as soon as the generated FAST
action text reaches `|`. LeRobot detokenization ignores everything after `|`, so
this should preserve the continuous action chunk.

## setup

Requires Hugging Face auth with access to `google/paligemma-3b-pt-224`.

```bash
python -m venv --system-site-packages .venv-pi
.venv-pi/bin/python -m pip install -U pip setuptools wheel
.venv-pi/bin/python -m pip install -e . --no-deps
.venv-pi/bin/python -m pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.4"
.venv-pi/bin/python -m pip install hf-libero==0.1.3 --no-deps
.venv-pi/bin/python -m pip install hydra-core robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 easydict thop mujoco tensorboardX imageio-ffmpeg egl_probe numba jupytext pytest
.venv-pi/bin/python -m pip install "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12" "matplotlib>=3.5.3" hf-egl-probe
```

## exactness run

This run is not for speed. It runs full native decode and early-stop decode on
the same observation at every chunk refresh.

```bash
for suite in libero_object libero_spatial libero_goal; do
  MUJOCO_GL=osmesa .venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
    --task "$suite" --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes 3 --steps 300 \
    --modes target_eos_validate \
    --enable-fast-token-hooks \
    --output-dir "outputs/pi0fast_target_eos_validate_30task3ep_batched/${suite}" \
    --device cuda --dtype bfloat16 \
    --smooth-position-delta 0.06 --smooth-rotation-delta 0.22
done
```

Observed:

| suite | episodes | success | exact verifies | max action diff |
| --- | ---: | ---: | ---: | ---: |
| libero_object | 30 | 30/30 | 472 | 0.0 |
| libero_spatial | 30 | 24/30 | 458 | 0.0 |
| libero_goal | 30 | 27/30 | 420 | 0.0 |
| overall | 90 | 81/90 | 1350 | 0.0 |

## speed run

Run baseline and target-eos as separate sweeps. Do not run
`--modes baseline,target_eos` in one process for this comparison; LIBERO reset
ordering made that noisy during testing.

```bash
for mode in baseline target_eos; do
  for suite in libero_object libero_spatial libero_goal; do
    MUJOCO_GL=osmesa .venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
      --task "$suite" --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --episodes 3 --steps 300 \
      --modes "$mode" \
      --enable-fast-token-hooks \
      --output-dir "outputs/pi0fast_target_eos_speed_90_separate/${mode}/${suite}" \
      --device cuda --dtype bfloat16 \
      --smooth-position-delta 0.06 --smooth-rotation-delta 0.22
  done
done
```

Summarize:

```bash
.venv-pi/bin/python scripts/summarize_pi0fast_target_eos.py \
  outputs/pi0fast_target_eos_speed_90_separate
```

Observed:

| suite | baseline | target_eos | baseline ms | target_eos ms | speedup | drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | 81/90 | 81/90 | 634.3 | 259.5 | 2.44x | 0.0% |
| libero_object | 30/30 | 30/30 | 579.0 | 200.4 | 2.89x | 0.0% |
| libero_spatial | 24/30 | 24/30 | 671.6 | 306.4 | 2.19x | 0.0% |
| libero_goal | 27/30 | 27/30 | 652.2 | 271.7 | 2.40x | 0.0% |

No task had a success-count delta between baseline and target-eos.
