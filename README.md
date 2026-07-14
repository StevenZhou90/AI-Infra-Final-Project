# VLA Multi-Serving Platform

This repository packages a multi-service inference platform for robotic-arm
Vision-Language-Action (VLA) policies. It combines OpenVLA/SimplerEnv rollout
evaluation, SpecVLA-style trajectory drafting, pi0-FAST chunk execution, gRPC
serving, GPU-aware routing, admission control, and benchmark tooling.

## Highlights

- Multi-model serving for OpenVLA-style policies and pi0-FAST experiments.
- gRPC inference API with protobuf stubs in `proto/`.
- GPU-aware pi0.5 cluster router with warmup, admission control, and profiling.
- Speculative trajectory heads for lower-latency robotic-arm control loops.
- pi0-FAST token/chunk hooks for target-equivalent serving experiments.
- LIBERO and SimplerEnv benchmark runners with reproducible result summaries.

## Verified Results

Short answer: the serving-platform results are good, and the real-policy smoke
is now measured instead of guessed. Synthetic serving scales across GPUs 0-2,
while the real LIBERO/OpenVLA smoke confirms the distributed runner works but
shows the current trajectory head needs more tuning before it should be claimed
as quality-preserving.

### SimplerEnv OpenVLA Coke-Can Matrix

Matched benchmark matrix:
- tasks: vertical, horizontal, standing coke-can
- x positions: `-0.3500`, `-0.2925`, `-0.2350`
- episodes: `27`

| Decoder | Success | Avg ms/step | Speedup |
| --- | ---: | ---: | ---: |
| Baseline OpenVLA | `14/27` | `302.5` | `1.00x` |
| Adaptive fast policy | `14/27` | `145.1` | `2.08x` |

The adaptive fast policy keeps the same success count while reducing per-step
latency by about 52%.

### LIBERO Goal SpecVLA-Style Slice

Selected-slice protocol:
- suite: `libero_goal`
- task ids: `0, 1, 3, 5, 7, 9`
- trials per task: `5`
- total episodes: `30`
- base policy: `openvla/openvla-7b-finetuned-libero-goal`

| Decoder | Success | Avg ms/step | Speedup vs AR |
| --- | ---: | ---: | ---: |
| AR OpenVLA | `22/30` | `317.6` | `1.00x` |
| SpecVLA-style tuned chunk | `23/30` | `280.7` | `1.13x` |
| Adaptive direct K=2 | `21/30` | `202.7` | `1.57x` |
| Two-head direct K=3/K=2, strict smooth | `22/30` | `213.5` | `1.49x` |
| Two-head direct K=3/K=2, loose smooth | `23/30` | `202.0` | `1.57x` |

The best no-task-router configuration is the loose-smooth two-head direct chunk
decoder: K=3 smooth head, K=2 complex head, and relaxed smooth-phase thresholds.

### PI0-FAST Target-EOS Early Stop

The strongest current PI0-FAST mechanism is action-end early stopping over FAST
tokens. This compares the fixed-budget PI0-FAST decode against stopping when
the generated action text reaches `|`; validation showed the decoded continuous
action chunk is unchanged. It is not yet a passing final result on the strict
120-row proof gate.

Legacy 90-episode LIBERO target-eos equivalence result:

| Decoder | Success | Avg ms/step | Speedup | Drop |
| --- | ---: | ---: | ---: | ---: |
| Fixed-budget PI0-FAST | `81/90` | `634.3` | `1.00x` | - |
| Target-EOS early stop | `81/90` | `259.5` | `2.44x` | `0.0%` |

Those rows are useful for the zero-drop early-stop claim, but the current HF
accuracy reproduction should be anchored on the carded v044 checkpoint below.

Current HF-carded v044 sanity checks:

| Path | Task slice | Success | Avg ms/step | Notes |
| --- | --- | ---: | ---: | --- |
| Official `lerobot-eval` + camera `rename_map` | HF task list: object/spatial/goal/10, 1 episode each | `30/40` | `139.8 s/episode` | Fixed 256-token decode, `75.0%` |
| Official `lerobot-eval` + camera `rename_map` | `libero_object`, task 1, 2 episodes | `2/2` | `88.9 s/episode` | Fixed 256-token decode |
| Custom runner fixed-budget baseline | `libero_object`, task 1, episode 0 | `1/1` | `602.6` | Same v044 checkpoint |
| Custom runner `target_eos` | `libero_object`, task 1, episode 0 | `1/1` | `224.1` | `2.69x` vs fixed budget on this episode |
| Custom runner `target_eos` | `libero_object`, tasks 0-9, episode 0 | `8/10` | `206.1` | One-init-state smoke, not full gate |
| Custom runner `target_eos` | `libero_spatial`, tasks 0-9, episode 0 | `9/10` | `330.0` | One-init-state smoke |
| Custom runner `target_eos` | `libero_goal`, tasks 0-9, episode 0 | `7/10` | `282.6` | One-init-state smoke |
| Custom runner `target_eos` | object + spatial + goal, tasks 0-9, episode 0 | `24/30` | `272.9` | `80.0%` cross-suite smoke |
| Custom runner `target_eos` | object + spatial + goal, tasks 0-9, episodes 0-3 | `93/120` | `269.2` | `77.5%` v044 extended smoke |
| Custom runner `target_eos` | `libero_10`, tasks 0-9, episode 0 | `1/10` | `182.5` | Negative check; not the missing high-SR suite |
| Custom runner `target_eos_validate` | `libero_object`, task 1, episode 0 | `1/1` | `674.5` | 14 exact verifies, max action diff `0.0` |
| Custom runner `target_eos_validate` | weak `libero_goal` rows, task ids 0/6/9, episode 0 | `0/3` | `674.2` | 90 exact verifies, max action diff `0.0` |
| Custom runner `target_eos`, absolute control | `libero_goal`, task 0, episode 0 | `0/1` | `222.0` | Negative protocol check |
| Official `lerobot-eval`, `env.init_states=false`, seed 1000 | `libero_goal`, task 0, episode 0 | `1/1` | `95.7 s/episode` | Random-state diagnostic recovers this weak row |
| Custom runner `target_eos`, `--no-init-states --seed 1000` | `libero_goal`, task 0, episode 0 | `1/1` | `271.2` | Same row succeeds with action-end stopping |
| Official `lerobot-eval`, `env.init_states=false`, seed 1000 | `libero_object`, task 0, episode 0 | `1/1` | `91.1 s/episode` | Official fixed-budget row succeeds |
| Custom runner baseline, `--no-init-states --seed 1000` | `libero_object`, task 0, episode 0 | `0/1` | `559.2` | Sentinel mismatch; custom rollout SR is not the HF-card authority |

The HF model card for `lerobot/pi0fast-libero-v044` reports `82.5%` LIBERO SR.
The official LeRobot command on this v0.4.4 install, using the carded v044
checkpoint, the HF task list, `eval.n_episodes=1`, and the camera `rename_map`,
lands at `30/40 = 75.0%`: object `9/10`, spatial `8/10`, goal `8/10`, and
`libero_10` `5/10`. The failed task ids are `libero_object_0`,
`libero_spatial_1`, `libero_spatial_9`, `libero_goal_3`, `libero_goal_9`, and
`libero_10_{0,2,4,6,9}`. This is below the HF card's `82.5%` table, but it is
not the old `7/120` failure mode. It was a fixed 256-token decode run, so the
remaining accuracy gap is not caused by action-end early stopping. Artifact:
`outputs/eval/2026-07-14/00-35-17_libero_pi0_fast/eval_info.json`.

Follow-up HF/cache check: current HF docs are for LeRobot main/v0.6.0, while
this environment uses LeRobot v0.4.4. The current `lerobot/pi0fast-libero`
snapshot is not a drop-in replacement for the v044 card here: its config
expects `observation.images.image` and `observation.images.image2`, so the docs
`rename_map` fails with missing image features. Without the rename map, this
install failed early `libero_object` probes. Re-running the v044 carded
checkpoint on `libero_object_0` with the required camera `rename_map` failed
again (`0/1`), artifact
`outputs/eval/2026-07-14/02-46-06_v044_object0_rerun/eval_info.json`. Treat
the remaining `75.0%` versus `82.5%` delta as a LeRobot/LIBERO version or
protocol reproduction gap pending a v0.6.0-stack rerun, not as a token-stopping
accuracy regression.

The local 30-row v044 smoke is still within the same broad regime, and the
extended 120-row custom run lands at `93/120 = 77.5%`: object `38/40`, spatial
`31/40`, goal `24/40`. Exact validation on representative failed goal rows
matched the fixed 256-token decode exactly (`max_action_diff=0.0`). The lower
custom rows should therefore be read as stricter fixed-init-state and
custom-rollout stress tests, not as the HF card protocol. After adding the
official camera `rename_map`, global seeding, and config-first policy load to
the custom runner, one sentinel row still diverges: official `lerobot-eval`
succeeds on `libero_object` task 0 seed 1000 while the custom baseline loop does
not. Treat official `lerobot-eval` as the accuracy authority; use the custom
runner for token-level latency/equivalence experiments until that rollout
mismatch is fully reconciled.

LeRobot's PI0-FAST action path does not call Hugging Face `generate()` for
actions. It uses a hand-written loop in `sample_actions_fast` /
`sample_actions_fast_kv_cache` that iterates to `max_decoding_steps=256` and
then detokenizes. Action-end stopping is therefore a real latency optimization
for this policy path, not a generic built-in EOS option that was already active.

Current v044 non-quantized serving-component result:

| Path | Chunk mean | Per request | Per action | Notes |
| --- | ---: | ---: | ---: | --- |
| Single `target_eos` request | `605.1 ms` | `605.1 ms` | `60.5 ms` | 10 actions/request |
| Replicated batch 8 `target_eos` | `788.6 ms` | `98.6 ms` | `9.9 ms` | 6.14x throughput speedup |

This is model-serving time, not full LIBERO rollout wall time. Full simulator
rollout rows still include OSMesa/LIBERO stepping and image observation
formatting overhead.

Historical strict 120-row artifact from the uncarded `lerobot/pi0fast-libero`
checkpoint. The 120 rows are `libero_object`, `libero_spatial`, and
`libero_goal`, 10 task ids each, 4 episode/init-state ids each:

| Decoder | Success | Avg ms/step | Speedup | Drop |
| --- | ---: | ---: | ---: | ---: |
| Fixed-budget PI0-FAST | `7/120` | `607.9` | `1.00x` | - |
| Target-EOS early stop | `7/120` | `339.8` | `1.789x` | `0.0%` |

That strict row should not be read as an HF v044 accuracy baseline. It used a
different checkpoint and is only useful as a historical latency/equivalence
artifact. For v044 accuracy reproduction, use `lerobot/pi0fast-libero-v044` and
map the default LIBERO camera keys to the v044 policy keys when running the
official LeRobot CLI:

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa .venv-pi/bin/lerobot-eval \
  --policy.path=lerobot/pi0fast-libero-v044 \
  --env.type=libero \
  --env.task=libero_object \
  --env.task_ids='[1]' \
  --env.init_states=true \
  --eval.batch_size=1 \
  --eval.n_episodes=2 \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=false \
  --rename_map='{"observation.images.image":"observation.images.base_0_rgb","observation.images.image2":"observation.images.left_wrist_0_rgb"}'
```

On this LeRobot v0.4.4 install, omitting the `rename_map` fails before rollout
because the v044 checkpoint expects `base_0_rgb` and `left_wrist_0_rgb`, while
the default LIBERO env exposes `image` and `image2`.

For the random-state diagnostic protocol, use `--env.init_states=false` in
`lerobot-eval`, or `--no-init-states` in `scripts/run_pi0fast_chunk_eval.py`.

For the stricter 120 matched-eval gate, use
`scripts/run_robotics_spec_120_proof.py` to launch the canonical proof wrapper,
or call `scripts/run_pi0fast_100_eval_gate.py` /
`scripts/gate_pi0fast_target_eos.py` directly with the protocol in
`docs/pi0fast_target_eos.md`.

There is also a CPU-only synthetic check for the underlying draft-verify idea
in `docs/robotics_spec_decode_synthetic.md`; it is useful for CI but does not
replace the real LIBERO gate. Use `scripts/audit_robotics_spec_goal.py` to
check both artifacts before making the final claim.

For a no-checkpoint speculative candidate on the real PI0-FAST model, evaluate
`pattern_sd_direct` with `target_eos` included as the early-stop reference. This
candidate drafts from narrow robot action-token regularities and still verifies
tokens with PI0-FAST before emission. Use
`scripts/sweep_pi0fast_pattern_offline.py` or
`scripts/eval_pi0fast_pattern_offline.py` on saved FAST-token traces to screen
pattern settings before launching the simulator gate. Pass the selected sweep
JSON via `scripts/run_pi0fast_100_eval_gate.py --pattern-sweep-json` so the
manifest records the chosen row; `scripts/pattern_sweep_to_eval_args.py` is
available for manual shell-arg extraction. The pattern sweep includes exact
full-block KV reuse, bonus-token emission, dynamic lookahead, and guarded
second-order action-token extrapolation options so the real run can evaluate
recent speculative-decoding ideas without changing target-token outputs. It can
also test a previous-chunk position prior, which drafts token `i` from token `i`
of recent verified chunks in the same episode and still requires PI0-FAST to
verify every emitted token. `--position-mode-histogram` adds a related
per-position modal-token prior over recent verified chunks that share the current
prefix; it is still exact because the target model verifies each emitted token.
`--global-position-mode` is a more aggressive variant for the smaller robotics
output distribution: it proposes modal tokens at the same absolute FAST position
from bounded recent history even when the current prefix differs, and still relies
on exact target verification before emission.
It can also test prompt-lookup n-gram continuation with `--ngram-continuation`,
which drafts from suffix matches in already verified FAST tokens and recent
verified chunks. It can test an
action-transition-histogram prior with `--action-transition-histogram`, which
learns same-dimension token transitions from verified prefixes and recent
chunks. It can also test an action-delta-histogram prior with
`--action-delta-histogram`, which proposes tokens from frequent recent
per-action-dimension FAST-token deltas in the current prefix and recent
verified chunks. `--chunk-position-delta` tests a per-absolute-position delta
mode over recent verified chunks; the high-level candidate pipeline sweeps
`--action-delta-min-counts 1,2` and
`--chunk-position-delta-min-counts 1,2` so repeated deltas can be preferred over
single recent observations. `--action-trend-regression` fits a short regression line over
the verified tokens for the current action dimension and drafts the projected
next token; PI0-FAST still verifies it before emission.
`--action-prefix-lookup` drafts later dimensions of the current action vector
from prior verified actions that share the already-emitted intra-action prefix,
which exploits low-entropy robot action structure without changing exact
verification.
`--action-vector-suffix-lookup` is a stricter full-vector variant: once some
dimensions of the current action are verified, it proposes the remaining suffix
from prior full action vectors with the same prefix, and PI0-FAST still verifies
every token.
`--chunk-length-stop` drafts the FAST stop token when recent verified chunks
ended at the same token length, which targets the final `target_eos` verifier
step without changing exactness.
`--action-token-neighborhood` adds a
small numeric neighborhood around smooth extrapolations, recent verified chunk
positions, and other enabled robot-prior centers; this is intended for exact
tree verification, where `x`, `x-1`, and `x+1` style candidates can be checked
in one target pass.
`--action-context-tree` adds a
same-action-dimension context-tree histogram: it learns short verified
same-dimension token histories and backs off by depth before the target model
verifies the proposed continuation. `--action-dimension-mode` drafts the
most common verified token for the current action dimension from the prefix and
recent chunks, which targets held joints and gripper states without using an
unverified model. `--hold-action-token` is the lower-latency version of that
idea: it drafts the previous token from the same action dimension and lets the
target model verify it. It can also test source-agreement
drafting with `--min-source-agreements 1,2`, where threshold `2` prefers a token
only when at least two configured cheap priors propose it. `--source-cooldown`
screens an acceptance-aware source scheduler: after a source's verified token is
rejected, the drafter temporarily skips that source and falls back to the next
configured cheap prior. This is still exact because the target model verifies
all emitted tokens.
Offline scoring resets history on `(task, task_id, seed)` changes when trace
shards include suite/task names, falling back to `(task_id, seed)` for older
shards. Use the per-task sweep thresholds when
passing a sweep JSON to avoid choosing a setting that only wins on aggregate
trace statistics; use heldout sweep thresholds when the sweep was run with
`--heldout-split task` for the canonical proof path. `task_seed` is still useful
for episode-style previous-chunk experiments, but it can leave the same task in
both ranking and heldout groups when several seeds are present. Sweep JSON now
records rank/heldout task-key overlap so proof wrappers can require task-disjoint
heldout evidence. Automatic grouped splits use
`--heldout-val-fraction` as a fraction of task, seed, or task/seed groups.
For source-specific priors, add `--pattern-min-metric NAME=VALUE` and
`--pattern-min-heldout-metric NAME=VALUE`, for example
`ngram_continuation_accepted_tokens=1`,
`action_context_tree_accepted_tokens=1`,
`action_transition_histogram_accepted_tokens=1`,
`action_delta_histogram_accepted_tokens=1`,
`chunk_position_delta_accepted_tokens=1`,
`action_prefix_lookup_accepted_tokens=1`,
`action_vector_suffix_lookup_accepted_tokens=1`,
`action_vector_transition_accepted_tokens=1`,
`action_repeat_vector_accepted_tokens=1`,
`chunk_length_stop_accepted_tokens=1`,
`action_token_neighborhood_accepted_tokens=1`,
`position_mode_histogram_accepted_tokens=1`,
`global_position_mode_accepted_tokens=1`,
`action_dimension_mode_accepted_tokens=1`,
`hold_action_token_accepted_tokens=1`, or
`source_agreement_accepted_tokens=1`, so the selected row must show accepted
tokens from the enabled prior before the real simulator gate starts. Prefer
`min_task_*` source metrics, such as
`min_task_ngram_continuation_accepted_tokens=1`,
`min_task_action_context_tree_accepted_tokens=1`,
`min_task_action_transition_histogram_accepted_tokens=1`,
`min_task_action_delta_histogram_accepted_tokens=1`,
`min_task_chunk_position_delta_accepted_tokens=1`,
`min_task_action_prefix_lookup_accepted_tokens=1`,
`min_task_action_vector_suffix_lookup_accepted_tokens=1`,
`min_task_action_vector_transition_accepted_tokens=1`,
`min_task_action_repeat_vector_accepted_tokens=1`,
`min_task_chunk_length_stop_accepted_tokens=1`,
`min_task_action_token_neighborhood_accepted_tokens=1`,
`min_task_position_mode_histogram_accepted_tokens=1`,
`min_task_global_position_mode_accepted_tokens=1`,
`min_task_action_dimension_mode_accepted_tokens=1`,
`min_task_hold_action_token_accepted_tokens=1`, or
`min_task_source_agreement_accepted_tokens=1`, when the prior should work on
every task in the offline split.
With `--pattern-sweep-json`, `--pattern-auto-source-min-metrics` derives these
accepted-token checks from the selected row's enabled optional sources and also
requires matching heldout metrics when the row contains heldout data. Target
anchored tree modes are held to the same standard with
`tree_anchor_accepted_tokens=1` and
`min_task_tree_anchor_accepted_tokens=1`. The candidate pipeline passes this
gate-runner flag by default.
When `--run-final-audit` is enabled, the 120-eval wrapper also passes
`run_manifest.json` to the final audit. For sweep-selected pattern candidates
the audit requires positive
`pattern_sweep_selection.required_source_counts` for every required source, so
capped sweeps cannot become final evidence if they skipped an enabled source
family.
The sweep grid is conditional: when an optional source is disabled, its
history/top-k/context knobs are collapsed to the first provided value instead
of creating duplicate equivalent rows. This keeps heldout ranking from being
biased by repeated disabled-source configs and makes the default pipeline sweep
more practical.
Use `--source-priority-modes default,lookup_first,smooth_first` to let heldout
sweeps choose whether prompt-lookup/previous-chunk priors or smooth action-token
extrapolation should propose first; the selected row is replayed online via
`--pattern-source-priority`.

To collect fresh early-stop traces and stage the candidate gate in one command,
use `scripts/run_pi0fast_pattern_candidate_pipeline.py`. It writes lightweight
`target_eos` token trace shards, sweeps exact pattern settings with a heldout
split, and invokes the 120-task wrapper with `--reference-mode target_eos` by
default in gate dry-run mode. Trace shards record the resolved FAST action-end
token, and offline sweeps infer it unless `--stop-token-ids` is supplied, so
stop-aware priors are ranked against the same early-stop token used online.
PI0.5 is not supported by this token-trace pipeline until a PI0.5-specific
token/stop adapter exists.
The pipeline also bounds the default offline search with
`--max-enabled-sources 4` and `--max-sweep-configs 4096`. The sweep prunes
over-budget source combinations before evaluation and varies source settings
fastest, so the cap covers many robot-prior ideas before deeper runtime variants.
Sweep JSON records `evaluated_source_counts`, `evaluated_source_coverage`, and
`evaluated_enabled_source_count_histogram`; inspect those fields before
promoting a capped sweep row so the selected run did not accidentally skip an
intended source family.
The pipeline defaults `--pattern-required-source-coverage auto`, deriving the
required families from enabled sweep modes and forwarding them to the gate
runner. A capped sweep that never evaluated an enabled source family is rejected
before the 120-task simulator run is staged.
It also derives the pattern sweep train and heldout task-count gates from the
planned suite/task-id coverage unless explicit `--pattern-min-task-count` or
`--pattern-min-heldout-task-count` overrides are provided.
Pattern proof runs additionally require the sweep heldout split to cover every
requested suite, using `selection.heldout_suite_keys` in the sweep JSON and
saved run manifest.
The candidate gate defaults `--pattern-sweep-rank` to `0`, which auto-selects
the first ranked sweep row that passes those train, heldout, and source-usage
thresholds and records the actual selected rank in the manifest.

For speculative PI0-FAST candidates, the wrapper defaults reference thresholds
to the main gate thresholds, so a candidate must also meet the speedup,
success-drop, and regression requirements versus `target_eos` early stop.

The offline sweep also has an explicit `--tree-widths` knob for testing
tree-style candidate verification inspired by newer traversal/tree
speculative-decoding work. The candidate pipeline's default sweep is compact
but includes `--tree-widths 1,4`, `--dynamic-tree-width both`, and
`--tree-anchor-target-token both` plus
`--tree-anchor-target-continuation both`; pass `--tree-widths 1` when selecting
args for the lowest-risk chain verifier only. Rows with `tree_width > 1` now
convert to `--pattern-tree-width` and use the online exact tree verifier, which
batches several robot-prior candidates in one target pass and commits only the
target-matching selected prefix. The continuation anchor always starts from the
already-produced target greedy token and verifies only drafted futures after it.
Treat this as experimental until the full 120-task gate validates it. The
default sweep also includes
`--action-trend-regression both`, `--action-delta-ngram both`,
`--action-delta-min-counts 1,2`, and `--chunk-delta-template both` to test
velocity-space prompt lookup over recent verified robot chunks, and
`--chunk-prefix-retrieval both` for near-repeated verified chunks with small
bounded prefix mismatches, and `--source-acceptance-bias both` to reorder cheap
sources using recent verifier acceptance within the current decode.
`--dynamic-tree-width both` adapts the number of
verified tree candidates from recent target acceptance, following the lossless
dynamic-draft-tree idea in EAGLE-2 but using observed verifier acceptance
instead of learned confidence. Add
`--action-token-neighborhood both` with `--tree-widths` greater than 1 to screen
nearby quantized FAST-token candidates around smooth, recent-chunk, and
enabled-prior centers. When a tree
width greater than 1 is selected or
passed manually, the wrapper auto-adds gate checks that require the selected
tree width, nonzero tree verification, and zero unverified pattern-token
shortcuts in candidate `trace_stats`.

PI0.5 is currently supported in this repo as a flow-action rollout/serving
baseline, not through the PI0-FAST FAST-token speculative modes. The
`target_eos`, `target_cutoff`, and `pattern_sd` paths require PI0-FAST token
decode hooks and now fail early when launched with `--policy-kind pi05`. A
PI0.5 speculative result should only be compared against a PI0.5 stop-token
reference after a PI0.5-specific token/stop adapter has been added.

### Multi-GPU Serving Validation

Local hardware check:

```text
CUDA_VISIBLE_DEVICES=0,1,2 python - <<'PY'
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

Observed on this machine:

```text
True 3
0 NVIDIA H100 80GB HBM3
1 NVIDIA H100 80GB HBM3
2 NVIDIA H100 80GB HBM3
```

In-repo three-GPU serving load test:

```text
90 synthetic pi0.5 serving requests
3 runtime workers pinned to cuda:0, cuda:1, cuda:2
real CUDA matmul issued on every worker request
ClusterRouter + PI05RuntimeService + PI0FastServingRuntime exercised
```

Observed result:

```text
admitted: 90/90
worker_requests: gpu0=30, gpu1=30, gpu2=30
deadline_misses: 0
avg_latency_ms: 44.0
p95_latency_ms: 44.0
throughput_req_per_s: 497.12
```

Apples-to-apples synthetic throughput check:

```text
240 requests, same per-request modeled latency
1 GPU: 584.20 req/s, worker_requests gpu0=240
3 GPU: 762.06 req/s, worker_requests gpu0=80,gpu1=80,gpu2=80
throughput_speedup: 1.30x
```

This validates the multi-GPU serving scheduler, router balancing, admission
control, per-GPU runtime services, and CUDA device binding.

### Real 3-GPU LIBERO/OpenVLA Smoke

Measured on this machine with GPUs 0-2, `torchrun --nproc_per_node=3`,
OpenVLA loaded once per rank, LIBERO rendered through OSMesa, and the R2
trajectory-head checkpoint restored from the project Hugging Face artifact.

Run artifact:

```text
outputs/libero_specvla_mirror/real_multigpu_smoke_20260613/smoke
```

Result:

| Decoder | Episodes | Success | Avg ms/step | Speedup vs AR |
| --- | ---: | ---: | ---: | ---: |
| AR OpenVLA | `5` | `3/5` | `172.63` | `1.00x` |
| Trajectory speculative | `5` | `0/5` | `164.51` | `1.049x` |

Gate:

```text
passed: false
speedup: 1.049x
success_drop: 0.6000
```

Interpretation: the distributed real-policy stack runs end to end on three H100s
and speculative inference is slightly faster per control step, but the current
R2 trajectory head is not quality-preserving on this smoke. Use the serving
router/load-test numbers for the multi-GPU platform claim, and use this smoke as
evidence that the real LIBERO runner is operational with honest gating.

For any renewed OpenVLA/SpecVLA attempt, gate final artifacts with
`scripts/gate_openvla_specvla.py`. It pairs AR and speculative rows by suite,
task, trial, and seed, then requires the same strict result shape as the PI0
gate: 120 matched evals, at least `2.0x` speedup, zero success drop, and zero
baseline-success regressions. `scripts/audit_robotics_spec_goal.py` accepts that
gate through `--openvla-gate`, so either a PI0-FAST gate or an OpenVLA gate can
satisfy the real-robotics half of the objective.

The OpenVLA 120-task wrapper enables matched-step and strict `spec_stats` checks
by default. A final claim cannot rely on different control-step counts,
`fast_draft_only`, chunk-buffer hits, relaxed fast-draft acceptances, or
approximate tree depth greater than 1.
`--allow-unverified-spec-shortcuts` is research-only and the wrapper requires
`--skip-audit --skip-result-card` when it is used, so relaxed artifacts cannot
be promoted into final objective evidence by accident.
The final objective audit also requires those strict OpenVLA thresholds and
zero shortcut counters by default when `--openvla-gate` is supplied.

To launch or gate the OpenVLA path with the same artifact layout, use:

```bash
python scripts/run_openvla_120_eval_gate.py \
  --config configs/libero_specvla_distributed.yaml \
  --run-id openvla_specvla_120 \
  --mode full \
  --min-pairs 120 \
  --min-speedup 2.0 \
  --max-success-drop 0.0 \
  --max-baseline-success-regressions 0
```

## Quick Start

Requirements:
- Ubuntu Linux
- NVIDIA GPU with a working CUDA driver
- Python 3.10
- `uv`

```bash
git clone https://github.com/StevenZhou90/AI-Infra-Final-Project.git
cd AI-Infra-Final-Project

sudo apt-get update
sudo apt-get install -y libegl1 libopengl0 libgl1-mesa-glx libvulkan1 libglvnd-dev libosmesa6 libosmesa6-dev

uv python install 3.10
uv sync --python 3.10

git clone https://github.com/simpler-env/SimplerEnv.git --recurse-submodules --depth 1 external/SimplerEnv
uv pip install -e external/SimplerEnv/ManiSkill2_real2sim
uv pip install -e external/SimplerEnv
```

Use GPUs 0-2 for local benchmark and serving runs:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2
```

## Run Baseline OpenVLA

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_openvla_sim.py \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Videos are written under `outputs/openvla_sim` unless `--output_dir` is set.

## Run Fast Speculative Policy

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_openvla_sim.py \
  --decoder trajectory-spec \
  --trajectory-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --trajectory-fast-draft-only \
  --trajectory-head-threshold 0.2 \
  --trajectory-fast-min-confident-tokens 5 \
  --task google_robot_pick_vertical_coke_can \
  --published-eval-setup \
  --episodes 3 \
  --steps 80
```

Gate interpretation:
- Lower gate: more aggressive, faster, less reliable.
- Higher gate: more conservative, slower, often more reliable.

## Run gRPC Serving

The pi0.5 gRPC service exposes policy inference over `proto/inference.proto` and
uses the serving runtime, router, and admission-control layers.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python -m serving.pi05_server \
  --host 0.0.0.0 \
  --port 50051 \
  --devices cuda:0,cuda:1,cuda:2 \
  --max-concurrent 12 \
  --warmup-steps 2
```

Load test:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/load_pi05_grpc.py \
  --target localhost:50051 \
  --clients 16 \
  --requests 128
```

Profile serving:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 uv run python scripts/profile_pi05_serving.py \
  --requests 64 \
  --concurrency 8
```

## Run LIBERO SpecVLA Benchmark

Smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_specvla_mirror.yaml \
  --mode smoke
```

Full run after smoke:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/run_libero_specvla_mirror.py \
  --config configs/libero_goal_selected_direct_twohead_k3k2_loose_smooth.yaml \
  --mode full
```

Distributed single-node run on GPUs 0-2:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 \
MUJOCO_GL=osmesa \
PYOPENGL_PLATFORM=osmesa \
LIBERO_CONFIG_PATH=.libero_config \
PYTHONPATH=.tmp_specvla:.tmp_specvla/openvla:external/LIBERO:$PYTHONPATH \
torchrun --standalone --nproc_per_node=3 \
  scripts/run_libero_specvla_distributed.py \
  --config configs/libero_specvla_distributed.yaml \
  --mode smoke
```

Summarize:

```bash
uv run python scripts/summarize_libero_mirror.py \
  --run-dir outputs/libero_specvla_mirror/<run_id>/smoke
```

## Run pi0-FAST Chunk Serving Experiments

The pi0-FAST path uses public LeRobot/LIBERO weights and gated PaliGemma
tokenizer access. Authenticate with Hugging Face before model loading.

```bash
python -m venv --system-site-packages .venv-pi
.venv-pi/bin/python -m pip install -U pip setuptools wheel
.venv-pi/bin/python -m pip install -e . --no-deps
.venv-pi/bin/python -m pip install "lerobot[pi] @ git+https://github.com/huggingface/lerobot.git@v0.4.4"
.venv-pi/bin/python -m pip install hf-libero==0.1.3 --no-deps
sudo apt-get install -y libosmesa6 libegl1 libgl1-mesa-dri libglx-mesa0
.venv-pi/bin/python -m pip install hydra-core robomimic==0.2.0 robosuite==1.4.0 bddl==1.0.1 easydict thop mujoco==2.3.7 "networkx>=3.2,<4" tensorboardX imageio-ffmpeg egl_probe numba jupytext pytest
.venv-pi/bin/python -m pip install "numpy<2" "opencv-python<4.12" "opencv-python-headless<4.12" "matplotlib>=3.5.3" hf-egl-probe

CUDA_VISIBLE_DEVICES=0 HF_HOME=.hf_cache MPLCONFIGDIR=/tmp/matplotlib-cache MUJOCO_GL=osmesa \
.venv-pi/bin/python scripts/run_pi0fast_chunk_eval.py \
  --policy lerobot/pi0fast-libero-v044 \
  --task libero_object \
  --task-id 0 \
  --episodes 3 \
  --steps 300 \
  --modes baseline,chunk_m3,chunk_m5_smooth,relaxed_chunk_retrieval_m3,exact_fast_sd_retrieval \
  --device cuda \
  --dtype bfloat16 \
  --enable-fast-token-hooks \
  --output-dir outputs/pi0fast_chunk/libero_object_task0
```

The runner reports success, model calls per control step, average ms/control
step, accepted execution windows, FAST token counts, fallback reasons, and
speedup versus baseline.

## Training and Data Generation

Collect DAgger data:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_trajectory_head_dagger_data.py \
  --policy-head-checkpoint checkpoints/traj_head_dagger_r1/best.pt \
  --sweep mini \
  --steps 80 \
  --out-dir data/trajectory_head_dagger_mini_r2 \
  --head-threshold 0.2 \
  --fast-min-confident-tokens 5 \
  --device cuda \
  --dtype bfloat16
```

Train the next trajectory head:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_trajectory_head.py \
  --data-dir data/trajectory_head_dagger_mini_r2 \
  --out-dir checkpoints/traj_head_dagger_r2 \
  --epochs 80 \
  --batch-size 128 \
  --hidden-dim 1024 \
  --embed-dim 128 \
  --hidden-fusion-dim 512 \
  --num-layers 3 \
  --lr 2e-4 \
  --dim-weights 1,1,1.5,2,2,2,5 \
  --change-weight 2.0 \
  --gripper-change-weight 8.0 \
  --late-timestep 20 \
  --late-weight 1.5 \
  --device cuda
```

Suite orchestration:

```bash
./scripts/train_spec_head_goal.sh
./scripts/train_spec_heads_all.sh
```

## Useful Scripts

- `scripts/run_openvla_sim.py`: SimplerEnv rollout and benchmark runner.
- `scripts/run_published_sweep.py`: published-evaluation-style sweeps.
- `scripts/run_libero_specvla_mirror.py`: SpecVLA-style LIBERO benchmark.
- `scripts/run_libero_specvla_distributed.py`: multi-GPU LIBERO runner.
- `scripts/gate_openvla_specvla.py`: strict matched AR-vs-SpecVLA gate.
- `scripts/run_robotics_spec_120_proof.py`: canonical PI0/PI0.5/OpenVLA 120-task proof launcher.
- `scripts/run_openvla_120_eval_gate.py`: OpenVLA run/gate/audit wrapper.
- `scripts/run_pi0fast_chunk_eval.py`: pi0-FAST LIBERO chunk experiments.
- `scripts/benchmark_pi0fast_serving_runtime.py`: pi0-FAST runtime benchmark.
- `scripts/load_pi05_grpc.py`: gRPC service load test.
- `scripts/profile_pi05_serving.py`: serving profiler.
- `scripts/train_trajectory_head.py`: draft head training.
- `scripts/check_spec_exactness.py`: speculative exactness checks.
- `scripts/benchmark_spatial_cache_compression.py`: spatial K/V cache benchmark.

## Project Layout

```text
configs/    YAML configuration defaults and benchmark recipes
envs/       Simulation environment wrappers
eval/       Legacy evaluation entrypoints
policies/   Policy wrappers for OpenVLA and ACT
proto/      Protobuf service definitions and generated stubs
scripts/    Data generation, training, evaluation, load, and benchmark tools
serving/    Runtime services, routers, decoders, draft heads, and gRPC stack
tests/      Unit tests for serving, routing, decoding, and benchmark helpers
```

## Packaging Notes

`pyproject.toml` packages `envs`, `policies`, `eval`, `serving`, and `proto`.
Large generated artifacts are intentionally ignored: checkpoints, logs, datasets,
external simulator clones, Hugging Face caches, and local virtual environments.
