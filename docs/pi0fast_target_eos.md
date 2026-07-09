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
sudo apt-get install -y libosmesa6 libegl1 libgl1-mesa-dri libglx-mesa0
.venv-pi/bin/python -m pip install hydra-core robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 easydict thop mujoco==2.3.7 "networkx>=3.2,<4" tensorboardX imageio-ffmpeg egl_probe numba jupytext pytest
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

## 120-eval gate

For the requested claim, use a 120 matched-evaluation gate rather than a
hand-counted table. This keeps the claim strict:

- every `target_eos` row must pair with the same suite, task id, episode, and
  seed as a `baseline` row
- success drop must be `0.0%`
- no baseline-success episode may regress, even if another episode improves
- average control-step speedup must be at least `2.0x`
- exact validation must cover the same suite, task id, episode, and seed keys
  as the speed eval set
- every validation row must run at least one exact fixed-budget comparison, and
  the maximum action difference must be `0.0`

This command set evaluates 3 LIBERO suites x 10 tasks x 4 seeds = 120 matched
evaluations for speed, plus 120 exact-action validation episodes.

```bash
export PI0FAST_EVAL_ROOT="outputs/pi0fast_target_eos_120"

for mode in baseline target_eos; do
  for suite in libero_object libero_spatial libero_goal; do
    MUJOCO_GL=osmesa .venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
      --task "$suite" --task-ids 0,1,2,3,4,5,6,7,8,9 \
      --episodes 4 --steps 300 \
      --modes "$mode" \
      --enable-fast-token-hooks \
      --output-dir "$PI0FAST_EVAL_ROOT/speed/$mode/$suite" \
      --device cuda --dtype bfloat16 \
      --smooth-position-delta 0.06 --smooth-rotation-delta 0.22
  done
done

for suite in libero_object libero_spatial libero_goal; do
  MUJOCO_GL=osmesa .venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
    --task "$suite" --task-ids 0,1,2,3,4,5,6,7,8,9 \
    --episodes 4 --steps 300 \
    --modes target_eos_validate \
    --enable-fast-token-hooks \
    --output-dir "$PI0FAST_EVAL_ROOT/validate/$suite" \
    --device cuda --dtype bfloat16 \
    --smooth-position-delta 0.06 --smooth-rotation-delta 0.22
done
```

Summarize and gate:

```bash
.venv-pi/bin/python scripts/summarize_pi0fast_target_eos.py \
  "$PI0FAST_EVAL_ROOT/speed"

.venv-pi/bin/python scripts/gate_pi0fast_target_eos.py \
  "$PI0FAST_EVAL_ROOT/speed" \
  --validation-root "$PI0FAST_EVAL_ROOT/validate" \
  --min-pairs 120 \
  --min-speedup 2.0 \
  --max-success-drop 0.0 \
  --max-baseline-success-regressions 0 \
  --min-validation-episodes 120 \
  --min-exact-verifies 1 \
  --max-action-diff 0.0 \
  --output "$PI0FAST_EVAL_ROOT/gate.json" \
  --markdown
```

Only claim the 120-eval result when the gate prints `PI0-FAST speedup gate:
PASS` and writes `gate_passed: true` in `gate.json`.

The same layout can be generated by the orchestration wrapper. With
`--run-preflight`, it checks CUDA, imports, scripts, and optional Hugging Face
auth before launching the expensive eval shards:

```bash
.venv-pi/bin/python scripts/run_pi0fast_100_eval_gate.py \
  --root "$PI0FAST_EVAL_ROOT" \
  --speed-modes baseline,target_eos \
  --candidate-mode target_eos \
  --run-preflight \
  --require-hf-token \
  --run-synthetic \
  --run-final-audit \
  --render-result-card \
  --skip-existing
```

For a checkpoint-free speculative PI0-FAST candidate, use `pattern_sd_direct`.
This is an exact FAST-token draft/verify mode: it drafts from robot-specific
regularities in the narrow action-token stream, such as repeated gripper tokens,
periodic action-dimension structure, and constant per-dimension token velocity.
The target PI0-FAST model verifies drafted tokens before they are emitted.

Include both `target_eos` and the candidate in the speed modes. The runner will
compare the candidate to the fixed-budget baseline and report the candidate
against the `target_eos` early-stop reference. With validation left at the
default `auto`, this also runs both `pattern_sd_validate` and
`target_eos_validate`; both validation sets must match the speed rows with
`max_action_diff=0.0`:

The wrapper intentionally rejects speculative PI0-FAST candidate runs that omit
`target_eos` or set a different reference mode. For speculative candidates it
also defaults the reference thresholds to the main gate thresholds, so the
candidate must clear the speedup, success-drop, and regression checks against
stop-token early stop, not fixed-budget decode alone.

```bash
.venv-pi/bin/python scripts/run_pi0fast_100_eval_gate.py \
  --root "outputs/pi0fast_pattern_sd_120" \
  --speed-modes baseline,target_eos,pattern_sd_direct \
  --candidate-mode pattern_sd_direct \
  --reference-mode target_eos \
  --min-reference-speedup 2.0 \
  --max-reference-success-drop 0.0 \
  --max-reference-success-regressions 0 \
  --run-preflight \
  --require-hf-token \
  --run-synthetic \
  --run-final-audit \
  --render-result-card \
  --skip-existing \
  -- --pattern-lookahead 8 --pattern-action-dim 7 --pattern-reuse-full-blocks --pattern-emit-bonus-token
```

For a learned candidate such as `block_sd_direct`, use the same layout but pass
its checkpoint arguments after `--` and keep `--reference-mode target_eos`.

Before launching a full 120-eval candidate run, score candidate pattern settings
on saved PI0-FAST token traces:

For new PI0-FAST pattern candidates, prefer collecting those traces
from the `target_eos` early-stop path so the offline comparison models the same
stop-token baseline used by the final gate. The trace shards carry the resolved
FAST action-end token, and offline sweeps infer `--stop-token-ids` from that
metadata unless an override is supplied. This wrapper collects per-suite
`target_eos` trace shards, runs the heldout pattern sweep, then stages the
120-eval wrapper in dry-run mode with `--reference-mode target_eos`:

```bash
python scripts/run_pi0fast_pattern_candidate_pipeline.py \
  --root outputs/pi0fast_pattern_candidate \
  --gate-dry-run
```

Use `--no-gate-dry-run` only when you are ready to launch the actual 120-task
simulator gate. The pipeline passes `--pattern-sweep-rank 0` and
`--pattern-auto-source-min-metrics` to the gate runner by default, so the final
gate auto-selects the first ranked row that passes train, heldout, task-count,
and per-row source-usage checks. For lower-level trace collection,
`run_pi0fast_chunk_eval.py` can write the sweep format directly
with `--modes target_eos`,
`--token-trace-output-dir`, and `--token-trace-modes target_eos`.
The pipeline's default offline sweep is bounded with `--max-enabled-sources 4`
and `--max-sweep-configs 4096`, which keeps the search focused on small
combinations of robot priors rather than overfit rows that enable every source
at once. The sweep prunes over-budget source combinations before evaluation and
varies source settings fastest, so the cap covers many robot-prior ideas before
deeper runtime variants. Sweep JSON includes `evaluated_source_counts`,
`evaluated_source_coverage`, and `evaluated_enabled_source_count_histogram` so
capped runs can prove which source families were actually evaluated. Use
`--max-enabled-sources 0 --max-sweep-configs 0` only when an exhaustive research
sweep is intentional.

```bash
python scripts/sweep_pi0fast_pattern_offline.py \
  --data-dir outputs/pi0fast_eagle_tasks0_4_20trace_data \
  --split all \
  --heldout-split task \
  --lookaheads 4,8,12 \
  --action-dims 7 \
  --max-periods 8,16 \
  --repeat-token-min-runs 2,3 \
  --reuse-full-blocks both \
  --emit-bonus-token both \
  --second-order-action-extrapolation both \
  --second-order-max-accels 4,8 \
  --action-context-tree both \
  --action-context-tree-max-contexts 2,3 \
  --action-context-tree-top-ks 2 \
  --action-transition-histogram both \
  --action-transition-history-sizes 2,4 \
  --action-transition-top-ks 2,3 \
  --action-delta-histogram both \
  --action-delta-histories 4,8 \
  --action-delta-top-ks 2,3 \
  --action-delta-min-counts 1,2 \
  --chunk-position-delta both \
  --chunk-position-delta-min-counts 1,2 \
  --chunk-delta-template both \
  --chunk-delta-template-max-delta-mismatch-values 0 \
  --action-token-neighborhood both \
  --action-token-neighborhood-radii 1 \
  --action-token-neighborhood-top-ks 3 \
  --action-dimension-mode both \
  --action-dimension-mode-history-sizes 2,4 \
  --action-dimension-mode-top-ks 2,3 \
  --hold-action-token both \
  --position-mode-histogram both \
  --position-mode-history-sizes 2,4 \
  --position-mode-top-ks 2,3 \
  --global-position-mode both \
  --global-position-history-sizes 16 \
  --global-position-top-ks 3 \
  --global-position-min-counts 3 \
  --ngram-continuation both \
  --ngram-min-contexts 2,3 \
  --ngram-max-contexts 8,16 \
  --min-source-agreements 1,2 \
  --source-priority-modes default,lookup_first,smooth_first \
  --source-cooldown both \
  --source-cooldown-steps 1,2 \
  --dynamic-lookahead both \
  --min-lookaheads 1,2 \
  --lookahead-shrinks 2,4 \
  --tree-widths 1 \
  --top-k 8 \
  --markdown \
  --output outputs/pi0fast_pattern_sd_offline/sweep.json
```

The sweep reports exact draft-prefix acceptance and modeled target-forward
reduction on trace tokens. Let the gate runner auto-select with
`--pattern-sweep-rank 0`, or manually inspect the first row that clears the
train and heldout thresholds, before choosing `--pattern-lookahead`,
`--pattern-action-dim`, period, repeat, full-block reuse, bonus-token, and
dynamic-lookahead settings for the real `pattern_sd_direct` run. The optional
source grids are conditional: if a prior is disabled, its history/top-k/context
settings collapse to the first supplied value so disabled configs do not
multiply equivalent rows. The optional
second-order extrapolator drafts from smooth per-action-dimension acceleration,
but remains exact because PI0-FAST verifies every emitted token. The optional
previous-chunk position prior drafts token `i` from token `i` of recent verified
chunks in the same task or episode; this targets robot rollout repetition while
remaining exact under the same verifier. When enabled through the 120-eval
wrapper, candidate `trace_stats` also report previous-chunk drafted and
accepted token counts so the final gate artifact shows whether the prior was
actually used. Offline scans reset this history on `(task_id, seed)` changes by
default, matching the real runner's per-episode reset and avoiding cross-reset
trace leakage. The optional position-mode histogram prior proposes the most
common next FAST token at the same chunk position among recent verified chunks
that share the current prefix. It is a conservative repetition prior for PI0-FAST
traces, and the online verifier still rejects any token that does not
match the target model. The optional global-position prior removes the prefix
match and proposes modal non-stop tokens at the same absolute FAST position from
bounded recent history, which is useful when robot actions are low entropy across
nearby control steps but the first token of the chunk changes. The target model
still verifies every emitted token. The optional n-gram continuation prior is a
prompt-lookup style drafter: it finds the longest suffix of already verified FAST tokens in the
current prefix or recent verified chunks and proposes that matched continuation.
This adapts recent prompt-lookup speculative decoding to the smaller robotics
token distribution without fitting a task-specific drafter. When enabled through
the 120-eval wrapper, the gate also requires n-gram drafted and accepted token
stats in candidate traces. A single setting can still be inspected in more
detail with
`scripts/eval_pi0fast_pattern_offline.py`.

Use `--action-context-tree both` to test a same-action-dimension context-tree
histogram. It learns short verified histories for the current action dimension,
backs off from deeper to shallower contexts, and supplies top-k candidates to
the exact chain or tree verifier. This is the robotics analogue of retrieval
and tree-candidate speculative decoding: exploit the smaller action-token
distribution, but still let PI0-FAST verify every emitted token. The
120-eval wrapper requires `action_context_tree_*` trace stats when the prior
is enabled.

Use `--source-priority-modes` to rank cheap proposal sources without changing
verification semantics. `lookup_first` prioritizes prompt-lookup/n-gram and
previous-chunk reuse, while `smooth_first` prioritizes second-order and linear
action-token extrapolation. The selected sweep row converts to
`--pattern-source-priority` for the real runner.

Use `--source-cooldown both` to screen an acceptance-aware source scheduler.
When a source proposes the first rejected verified token, the drafter skips that
source for the next few speculative blocks and falls back to the next configured
source. This adapts dynamic drafter scheduling ideas to robotics without
changing exactness: the target PI0-FAST model still verifies every
emitted token. The runtime reports `source_cooldown_events` and
`source_cooldown_skipped_sources`; the 120-eval wrapper requires those
trace-stat fields when `--pattern-source-cooldown` is enabled.

Use `--action-transition-histogram both` to test a dimension-aware transition
prior. For the next FAST token's action dimension, it builds a small transition
histogram such as `1 -> 2` from verified same-dimension tokens in the current
prefix and recent verified chunks. This targets cyclic low-entropy robot-token
patterns while keeping PI0-FAST as the exact verifier. The 120-eval wrapper
requires `action_transition_histogram_*` trace stats when the prior is enabled.

Use `--action-delta-histogram both` to test a same-action-dimension delta
histogram prior. It proposes tokens from frequent per-dimension FAST token
deltas in the current prefix and recent verified chunks, which is useful when
robot motion repeats a velocity that is more stable than the most recent single
delta. `--action-delta-min-counts` can require the same proposed delta token to
appear more than once before it is proposed. The online verifier remains exact,
and the 120-eval wrapper requires `action_delta_histogram_*` trace stats when
the prior is enabled.

Use `--chunk-position-delta both` to test a per-absolute-position delta prior.
It predicts the next token by applying recent verified same-position action
deltas to the current same-dimension token. `--chunk-position-delta-min-counts`
can require a delta to repeat across chunks before it is proposed, which is a
less task-memorizing variant of copying absolute token positions. The online
verifier remains exact, and the 120-eval wrapper requires
`chunk_position_delta_*` trace stats when the prior is enabled.

Use `--chunk-delta-template both` to test a verified chunk-delta template
prior. It compares the current chunk's already verified same-dimension deltas
with recent chunks, then applies the matching chunk's next delta to the current
same-dimension token. The default exact-match setting
`--chunk-delta-template-max-delta-mismatch-values 0` keeps this as a conservative
robot-motion template rather than a task-specific absolute-token copy. The
120-eval wrapper requires `chunk_delta_template_*` trace stats when the prior is
enabled.
The candidate pipeline also forwards `--pattern-required-source-coverage auto`
to the gate runner, so source families such as `chunk_delta_template` must appear
in the sweep-level `evaluated_source_counts`/`evaluated_source_coverage` fields
before a capped sweep row can be selected.

Use `--action-trend-regression both` to test a bounded same-action-dimension
trend prior. It fits a short regression line over verified prefix tokens for the
current action dimension and proposes the projected next token; PI0-FAST still
verifies every emitted token. The 120-eval wrapper requires
`action_trend_regression_*` trace stats when the prior is enabled.

Use `--action-prefix-lookup both` to test an intra-action-vector lookup prior.
Once the target model has emitted the first dimensions of the current action,
the drafter looks for prior verified actions with the same prefix and proposes
the next dimension token. This adapts prompt-lookup/tree-candidate speculative
decoding to the smaller robotics action-token distribution while keeping exact
PI0-FAST verification. The 120-eval wrapper requires
`action_prefix_lookup_*` trace stats when the prior is enabled.

Use `--action-vector-suffix-lookup both` to test a stricter full-action-vector
suffix lookup prior. Once the target model has emitted part of the current
action vector, the drafter proposes the next remaining dimension from prior full
action vectors whose prefix matches within
`--action-vector-suffix-max-prefix-delta-values`. This adapts prompt-lookup
speculation to low-entropy robot action vectors while preserving exact PI0-FAST
PI0-FAST verification. The 120-eval wrapper requires
`action_vector_suffix_lookup_*` trace stats when the prior is enabled.

Use `--action-vector-transition both` to test a full-action-vector transition
prior. The drafter matches the previous verified action vector against prior
verified vector transitions, then requires the current partial vector prefix to
match the candidate next vector before proposing the next dimension token. This
targets repeated low-entropy robot action transitions while preserving exact
PI0-FAST verification. The 120-eval wrapper requires
`action_vector_transition_*` trace stats when the prior is enabled.

Use `--action-repeat-vector both` to test a repeated-action-vector prior. After
the last verified full action vector has repeated for
`--action-repeat-min-repeats-values`, the drafter copies the same dimension from
that vector only while the current partial action prefix still matches it. This
targets held-pose segments without enabling unconditional same-dimension holds.
Verification remains exact, and the 120-eval wrapper requires
`action_repeat_vector_*` trace stats when the prior is enabled.

Use `--chunk-length-stop both` to test a verified chunk-length stop prior. When
recent verified chunks ended at the current prefix length, the drafter proposes
the FAST stop token so the exact verifier can accept the `target_eos` token in
the speculative block instead of spending a separate final step. The 120-eval
wrapper requires `chunk_length_stop_*` trace stats when the prior is enabled.
When traces were exported through `target_eos`, the offline sweep uses the
recorded action-end token for this prior automatically.

Use `--action-token-neighborhood both` to test a tree-friendly numeric
neighborhood around smooth action-token extrapolation, recent verified chunk
positions, and other enabled robot-prior centers. For a predicted token `x`,
the drafter can propose nearby quantized candidates such as `x`, `x-1`, and
`x+1`; the exact tree verifier batches those candidates and commits only tokens
selected by PI0-FAST. The 120-eval wrapper requires
`action_token_neighborhood_*` trace stats when the prior is enabled.

Use `--action-dimension-mode both` to test a same-action-dimension modal-token
prior. It proposes the most common verified FAST token for the next action
dimension from the current prefix and recent verified chunks, targeting held
joints and gripper states in the smaller robotics token distribution. It remains
exact because PI0-FAST verifies the token before emission, and the
120-eval wrapper requires `action_dimension_mode_*` trace stats when the prior
is enabled.

Use `--hold-action-token both` to test the cheapest same-dimension repetition
prior. It proposes the token from the same action dimension in the previous
action step, which targets held joints and gripper states without any fitted
drafter. Verification stays exact, and the 120-eval wrapper requires
`hold_action_token_*` trace stats when the prior is enabled.

Use `--min-source-agreements 1,2` to screen a source-agreement prior. With
threshold `2`, the drafter labels a token as `source_agreement` only when at
least two configured sources propose the same next FAST token; PI0-FAST still
verifies the token before emission. The sweep and runtime stats report
`source_agreement_drafted_tokens` and `source_agreement_accepted_tokens`, and
the 120-eval wrapper requires those trace-stat fields when
`--pattern-min-source-agreement > 1`.

For research screening and candidate selection, the same sweep can include
`--tree-widths 1,4` to model exact tree verification over several robot-prior
candidates per step. The candidate pipeline defaults to this compact tree
screen, with `--dynamic-tree-width both`, `--tree-anchor-target-token both`,
`--tree-anchor-target-continuation both`, `--action-trend-regression both`,
`--action-delta-ngram both`, `--chunk-delta-template both`, and
`--source-acceptance-bias both`; pass
`--tree-widths 1` for chain-only
screening. Rows with `tree_width > 1` convert to `--pattern-tree-width` and use
the online exact tree verifier. That verifier batches several candidates in one
target pass, then commits only the selected target-matching prefix. Keep the
full validation gate enabled before treating a tree row as final evidence. Add
`--dynamic-tree-width both` to adapt the
verified candidate count from recent exact acceptance, following dynamic
draft-tree speculative decoding while preserving target verification. Add
`--tree-anchor-target-token both` to evaluate an exact fallback that anchors on
the target greedy token after a tree first-token miss and verifies drafted
continuations after that anchor. Add
`--tree-anchor-target-continuation both` to always anchor on the already-known
target greedy token and verify only drafted futures after it. The default
candidate sweep includes `--chunk-prefix-retrieval both` to test a retrieval
prior for near-repeated verified robot chunks whose partial prefixes have a
small bounded mismatch. Add `--action-token-neighborhood both` with tree
rows to check nearby quantized FAST-token candidates around smooth,
recent-chunk, and enabled-prior centers. When
`--pattern-tree-width > 1` is selected from a
sweep or passed manually through `run_pi0fast_100_eval_gate.py`, the gate
command automatically requires candidate `trace_stats` to report the selected
tree width, nonzero tree verification, dynamic-tree stats when enabled, and
zero unverified pattern tokens.

When `--heldout-split` is set, the sweep ranks on the complement of that split
and attaches heldout metrics to the top rows. Use `task` for the canonical
proof path so selected pattern settings have task-disjoint heldout evidence.
`task_seed` remains useful for episode-style previous-chunk experiments, but it
can leave the same task in the ranking and heldout groups when several seeds are
present. Sweep JSON records rank/heldout task-key overlap, and the canonical
proof wrapper rejects overlap by default. Trace shards that include a suite/task
name use that
namespace in per-task metrics and history-reset keys, so `libero_goal:0` and
`libero_spatial:0` remain separate even though both have numeric task id 0.
When no explicit `--heldout-task-id` / `--heldout-seed` is provided, grouped
heldout splits sample `--heldout-val-fraction` of the task, seed, or task/seed
groups rather than a single sorted group. For `--heldout-split task`, records
with suite metadata are sampled per suite so the default multi-suite sweep keeps
heldout coverage across all requested suites.
For pattern proof runs, the canonical wrapper also requires the sweep's heldout
split to include every requested suite by default. Rerun the sweep from
per-suite `target_eos` traces if `selection.heldout_suite_keys` does not cover
the planned `--suites` list. Direct `run_pi0fast_100_eval_gate.py` final-audit
pattern runs apply the same precheck before launching simulator shards.
Pass the heldout thresholds to
`run_pi0fast_100_eval_gate.py` so the manifest rejects trace-overfit settings
before launching the real simulator run.

Then pass the sweep JSON directly to the real candidate run. The runner records
the selected sweep row and appended `--pattern-*` args in `run_manifest.json`:

```bash
.venv-pi/bin/python scripts/run_pi0fast_100_eval_gate.py \
  --root "outputs/pi0fast_pattern_sd_120" \
  --speed-modes baseline,target_eos,pattern_sd_direct \
  --candidate-mode pattern_sd_direct \
  --reference-mode target_eos \
  --min-reference-speedup 2.0 \
  --max-reference-success-drop 0.0 \
  --max-reference-success-regressions 0 \
  --run-preflight \
  --require-hf-token \
  --run-synthetic \
  --run-final-audit \
  --render-result-card \
  --pattern-sweep-json outputs/pi0fast_pattern_sd_offline/sweep.json \
  --pattern-sweep-rank 0 \
  --pattern-min-modeled-speedup 2.0 \
  --pattern-min-forward-reduction 2.0 \
  --pattern-min-task-forward-reduction 1.25 \
  --pattern-min-task-acceptance-rate 0.30 \
  --pattern-min-heldout-modeled-speedup 1.25 \
  --pattern-min-heldout-forward-reduction 1.25 \
  --pattern-min-heldout-task-forward-reduction 1.10 \
  --pattern-min-heldout-task-acceptance-rate 0.25 \
  --pattern-auto-source-min-metrics \
  --skip-existing \
  --
```

If you need a shell argument string instead, use
`scripts/pattern_sweep_to_eval_args.py` on the same sweep JSON.
The candidate pipeline wrapper derives `--pattern-min-task-count` and
`--pattern-min-heldout-task-count` from the planned suite/task-id coverage by
default; for the default 3-suite x 10-task layout with a 0.2 heldout fraction,
that resolves to 24 ranking task groups and 6 heldout task groups. Pass explicit
values only when intentionally running a smaller smoke gate.

`--pattern-auto-source-min-metrics` rejects an offline row before simulator
shards launch when an enabled optional source prior has no accepted tokens or no
`min_task_*` accepted-token coverage. It mirrors those checks onto heldout
metrics when the selected row contains heldout data. It also requires
`tree_anchor_accepted_tokens=1` and
`min_task_tree_anchor_accepted_tokens=1` when the selected row enables a
target-anchored tree mode. You can also use
`--pattern-min-metric NAME=VALUE` and `--pattern-min-heldout-metric NAME=VALUE`
for manual source-specific evidence such as
`ngram_continuation_accepted_tokens=1` or
`previous_chunk_position_accepted_tokens=1`,
`position_mode_histogram_accepted_tokens=1`,
`global_position_mode_accepted_tokens=1`,
`action_dimension_mode_accepted_tokens=1`,
`hold_action_token_accepted_tokens=1`,
`action_repeat_vector_accepted_tokens=1`,
`action_vector_suffix_lookup_accepted_tokens=1`,
`action_vector_transition_accepted_tokens=1`,
`chunk_length_stop_accepted_tokens=1`,
`source_agreement_accepted_tokens=1`,
`action_transition_histogram_accepted_tokens=1`, or
`action_delta_histogram_accepted_tokens=1`, or
`action_prefix_lookup_accepted_tokens=1`; heldout thresholds automatically
check the selected row's `heldout_`-prefixed metrics. Prefer min-task source
metrics such as `min_task_ngram_continuation_accepted_tokens=1`,
`min_task_source_agreement_accepted_tokens=1`,
`min_task_action_transition_histogram_accepted_tokens=1`, or
`min_task_action_delta_histogram_accepted_tokens=1`,
`min_task_action_prefix_lookup_accepted_tokens=1`,
`min_task_action_vector_suffix_lookup_accepted_tokens=1`,
`min_task_action_vector_transition_accepted_tokens=1`,
`min_task_action_repeat_vector_accepted_tokens=1`,
`min_task_chunk_length_stop_accepted_tokens=1`,
`min_task_global_position_mode_accepted_tokens=1`,
`min_task_action_dimension_mode_accepted_tokens=1`, or
`min_task_hold_action_token_accepted_tokens=1` when a prior should work
across every task in the offline split, not just in aggregate.
When the 120-eval wrapper selected a pattern sweep row and `--run-final-audit`
is enabled, it passes `run_manifest.json` to the audit and adds
`--require-pattern-source-coverage`. The audit then requires every source in
`pattern_sweep_selection.required_source_coverage` to have a positive
`required_source_counts` entry.

This protocol is PI0-FAST-specific today. LeRobot PI0.5 uses flow-action
sampling in this runner and does not expose the PI0-FAST FAST-token decode
hooks used by `target_eos`, `target_cutoff`, or `pattern_sd`; those modes now
fail early when launched with `--policy-kind pi05`. If the protocol is adapted
to PI0.5 later, compare speculative candidates against a PI0.5-specific
stop-token reference baseline, not against a fixed-budget decode alone.

The offline sweep is only a tuning proxy. Verify the final 0-drop / 2x claim
only with the real `run_pi0fast_100_eval_gate.py` gate above.

After the real gate and the synthetic draft-verify check have both produced
JSON artifacts, run the final objective audit:

```bash
python scripts/audit_robotics_spec_goal.py \
  --pi0fast-gate "$PI0FAST_EVAL_ROOT/gate.json" \
  --synthetic-gate "$PI0FAST_EVAL_ROOT/synthetic_gate.json" \
  --markdown
```

The final objective should only be treated as proven when this audit prints
`Robotics speculative decoding objective audit: PASS`.
For sweep-selected pattern runs, `--require-pattern-heldout-evidence` now also
requires `run_manifest.json` to preserve task-disjoint heldout metadata from the
sweep JSON, zero overlap between ranking and heldout task keys, and heldout
suite coverage for the manifest's requested suites.

To render a compact paper/report table from a passing audit:

```bash
python scripts/render_robotics_spec_result_card.py \
  "$PI0FAST_EVAL_ROOT/objective_audit.json" \
  --output "$PI0FAST_EVAL_ROOT/result_card.md"
```

OpenVLA can satisfy the real-eval half of the same audit through
`scripts/run_openvla_120_eval_gate.py`. Its default OpenVLA gate requires
matched AR/spec control-step counts plus episode-level `spec_stats`, and rejects
unverified fast-draft actions, chunk-buffer actions, relaxed fast-draft
acceptances, and tree verification depth above 1. Use
`--allow-unverified-spec-shortcuts` only for exploratory
research runs that should not be treated as the final 0-drop / 2x claim; the
wrapper requires `--skip-audit --skip-result-card` with that flag.
The final objective audit enforces the same strict OpenVLA `quality` counters
and thresholds by default.
