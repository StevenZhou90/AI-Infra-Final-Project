# PI0-FAST Speculative / Prefix Speedup Handoff

## Objective

Find a robotics analogue of speculative decoding for PI0-FAST or OpenVLA that
gives at least `2.0x` end-to-end control latency speedup with `0.0%` success
drop on the strict 120 matched-evaluation gate.

The current experiments use `lerobot/pi0fast-libero` in LIBERO object tasks with OSMesa rendering. The model is PI0-FAST, so actions are represented as FAST discrete action tokens. We can exploit this by stopping action-token generation early, appending the action-end token, detokenizing a partial action chunk, and executing the resulting action chunk.

## Current Best Result

Update: there is not yet a final passing strict 120-row proof because full
exact validation for adaptive prefix cutoff is still outstanding. The strongest
strict speed evidence is still speed-only: the suite-conditional adaptive
prefix hybrid at
`outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_hybrid/gate.json`
passes the strict 120-row speed gate at `302.5 ms/control` (`2.01x`) with
matched steps and zero success drop. It is not a final candidate as-is because
focused exact validation found another object failure. A selective late
target-EOS fallback fixed the known bad rows but was too slow in a partial
object speed shard. The current next step is a narrower safety gate for late
stability stops, ideally using scalar features from the stop candidate rather
than falling back on every late stability stop. The strict best fully validated
artifact remains exact target-EOS early stop, documented in
`docs/pi0fast_target_eos.md`, but it is short of the required `2.0x` speedup.
PI0-FAST's fixed-budget decoder keeps generating after the FAST action-end
marker (`|`), while LeRobot detokenization ignores the tail. Stopping when the
target model emits action-end preserves the continuous action chunk while
removing wasted target decode steps. The best current candidate for closing the
remaining gap is adaptive prefix cutoff: detokenize target prefixes at fixed
checkpoints, append the action-end marker, and stop only when the continuous
action snapshot is stable.

Legacy 90-episode LIBERO result:

| Eval | Mode | Success | ms/control | Speedup | Drop |
|---|---:|---:|---:|---:|---:|
| 3 suites x 10 tasks x 3 seeds | Full PI0-FAST fixed-budget baseline | `81/90` | `634.3` | `1.00x` | - |
| 3 suites x 10 tasks x 3 seeds | Target-EOS early stop | `81/90` | `259.5` | `2.44x` | `0.0%` |

Exactness validation over the same 90-episode scale showed `1350` refreshes with
`max_action_diff=0.0` against fixed-budget decode. That result is superseded for
final claims by the strict 120 matched-eval gate:

| Artifact | Mode | Success | ms/control | Speedup | Status |
|---|---:|---:|---:|---:|---:|
| `outputs/robotics_spec_120_proof/pi0fast_target_eos/speed_only_gate.json` | Fixed-budget baseline | `7/120` | `607.9` | `1.00x` | reference |
| `outputs/robotics_spec_120_proof/pi0fast_target_eos/speed_only_gate.json` | Target-EOS early stop | `7/120` | `339.8` | `1.789x` | fails `2.0x` |

This artifact has zero success drop and zero regressions, but it must get below
`303.9 ms/control` against the same baseline to pass `2.0x`.

Recent 2026-07-09 probes on the matched task-id/seed subset:

| Artifact | Candidate | Rows | Success | ms/control | Speedup | Takeaway |
|---|---:|---:|---:|---:|---:|---|
| `outputs/pi0fast_target_eos_fastpath_probe_rq` | current no-logits target-EOS | `6` | `1 -> 1` | `331.5` | `1.83x` | object reaches `2.04x`, spatial/goal stay near `1.74x` |
| `outputs/pi0fast_target_eos_no_gc_probe_rq` | target-EOS with `--disable-gradient-checkpointing` | `6` | `1 -> 1` | `333.3` | `1.82x` | neutral-to-worse; do not promote |
| `outputs/pi0fast_constrained_struct_margin1_validate_6x50_rq` | constrained head, margin `1.0`, object rows only | `2` | exact actions/tokens | `416.5`, `469.5` | slower than target-EOS | exact but too many full-head fallbacks |
| `outputs/pi0fast_empirical_vocab_smoke/shrunk_candidate_argmax` | 2-row empirical vocab, 1k restricted head | `1` | not exact | `186.8` | `2.12x` vs target-EOS smoke | fast but action diff; do not promote |
| `outputs/pi0fast_empirical_vocab_smoke/shrunk_candidate_prefix32_margin001` | same-slice empirical vocab, full-head prefix 32 | `1` | exact tokens/actions | `187.5` | `2.06x` vs target-EOS smoke | overfit smoke only; useful mechanism check |
| `outputs/pi0fast_empirical_vocab_probe_3task/heldout_task3_prefix32` | tasks 0-2 calibration, task 3 heldout | `1` | not exact | `209.1` | `1.85x` vs target-EOS smoke | heldout mismatch after prefix; needs broader calibration/adaptive fallback |
| `outputs/pi0fast_empirical_vocab_probe_3task/heldout_task3_prefix64` | tasks 0-2 calibration, task 3 heldout, prefix 64 | `1` | exact tokens/actions | `210.0` | `1.85x` vs target-EOS smoke | exact because restricted tail unused on this slice; not proof of generalization |
| `outputs/pi0fast_empirical_vocab_probe_3task/heldout_task3_prefix32_expand8` | tasks 0-2 vocab plus high-token radius 8, task 3 heldout | `1` | not exact | `211.4` | `1.84x` vs target-EOS smoke | neighborhood expansion alone did not fix action diff |
| `outputs/pi0fast_empirical_vocab_probe_3task/heldout_task3_prefix32_expand8_action4096` | expanded vocab, 4096 high-action band, task 3 heldout | `1` | `max_action_diff=0.0`, token mismatch in one chunk | `210.7` | `1.85x` vs target-EOS smoke | promising action-exact candidate; scale before trusting |
| `outputs/pi0fast_empirical_vocab_cross_suite/heldout_task3/libero_object_prefix24_action4096` | object/spatial/goal tasks 0-2 calibration, object task 3 heldout | `1` | `max_action_diff=0.0`, token mismatch in one chunk | `207.9` | `1.88x` vs target-EOS smoke | broader vocab preserves object heldout action equality |
| `outputs/pi0fast_empirical_vocab_cross_suite/heldout_task3/libero_spatial_prefix24_action4096` | same cross-suite vocab, spatial task 3 heldout | `1` | `max_action_diff=0.0`, token exact in validation | `312.8` | `1.07x` vs target-EOS smoke | exact but speed collapses because long token generations remain |
| `outputs/pi0fast_empirical_vocab_cross_suite/heldout_task3/libero_goal_prefix24_action4096` | same cross-suite vocab, goal task 3 heldout | `1` | `max_action_diff=0.0`, token exact in validation | `463.8` | `1.01x` vs target-EOS smoke | exact but no useful speedup |
| `outputs/pi0fast_empirical_vocab_cross_suite/heldout_task3/libero_goal_prefix24_action4096_charplateau32_2` | constrained vocab + `min_chars=32`, plateau 2, goal task 3 heldout | `1` | `max_action_diff=0.0`, token exact in validation | `460.0` | `1.00x` vs target-EOS smoke | plateau never fired; one chunk ran to the 256-token ceiling with only 3 decoded action chars |
| `outputs/pi0fast_adaptive_prefix_cross_suite_task3_ck32_224_stable1` | adaptive prefix checkpoints `32..224`, stable checks 1, object task 3 heldout | `1` | `max_action_diff=0.0` vs target-EOS | `213.5` | `1.86x` vs target-EOS smoke | action-stability cutoff recovers object speed without static vocab |
| `outputs/pi0fast_adaptive_prefix_cross_suite_task3_ck32_224_stable1` | same adaptive setting, spatial task 3 heldout | `1` | `max_action_diff=0.0` vs target-EOS | `294.9` | `1.14x` vs target-EOS smoke | exact but only modest gain because target-EOS already emitted shorter chunks |
| `outputs/pi0fast_adaptive_prefix_cross_suite_task3_ck32_224_stable1` | same adaptive setting, goal task 3 heldout | `1` | `max_action_diff=0.0` vs target-EOS | `298.8` | `1.54x` vs target-EOS smoke | fixes the no-stop-token 256-token goal chunk; best candidate to scale |
| `outputs/robotics_spec_120_proof_mini_adaptive_rq/pi0fast_adaptive/gate.json` | adaptive mini matched gate, task 3 across object/spatial/goal | `3` | `0 -> 0`, `max_action_diff=0.0` in validation | `283.3` | `2.19x` vs baseline, `1.42x` vs target-EOS | passes mini gate with exact validation; not final 120 proof because success thresholds were disabled for this tiny slice |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive/gate.json` | strict 120 adaptive speed-only, stable checks `1` | `120` | `7 -> 8` | `276.8` | `2.20x` vs baseline, `1.23x` vs target-EOS | fast but rejected: one matched-step mismatch on `libero_object` task `7`, episode `1` |
| `outputs/pi0fast_adaptive_mismatch_object7_ep1_stable3_rq` | focused bad-row validation, stable checks `3` | `1` | `0 -> 0`, `max_action_diff=0.0` | `246.3` speed / `897.7` validate | diagnostic only | fixes the stable-checks `1`/`2` action diffs on the bad row |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive_stable3/gate.json` | strict 120 adaptive speed-only, stable checks `3` | `120` | `7 -> 7` | `301.0` | `2.02x` vs baseline, `1.13x` vs target-EOS | passes strict speed-only gate; full exact validation still required |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive_stable3/validate/target_eos_adaptive_validate/libero_object/metrics.jsonl` | strict validation start, stable checks `3` | `2` | row 0 exact, row 1 not exact | `779.8` validate on bad row | blocks final proof | `libero_object` task `0`, episode `1`, seed `43` had `max_action_diff=0.07857602834701538` |
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_early4_128_rq` | focused bad-row validation, checkpoint `152` plus early checks `4<=128` | `1` | `0 -> 0`, `max_action_diff=0.0` | `257.5` speed / `796.0` validate | diagnostic only | cheapest exact focused setting found; checkpoints `144` and `148` still failed |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_hybrid/gate.json` | strict 120 hybrid speed-only: object checkpoint `152` with early checks `4<=128`, spatial/goal stable checks `3` | `120` | `7 -> 7` | `302.4` | `2.01x` vs baseline, `1.12x` vs target-EOS | speed-only passed, but object validation failed on task `1`, episode `1` with `max_action_diff=0.3119494318962097` |
| `outputs/pi0fast_adaptive_mismatch_object1_ep1_ck152_stable4_rq` | focused bad-row validation, object checkpoint `152`, stable checks `4` | `1` | `0 -> 0`, `max_action_diff=0.0` | `245.6` speed / `835.8` validate | diagnostic only | fixes the second object validation failure |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_hybrid/gate.json` | strict 120 hybrid speed-only: object checkpoint `152` with stable checks `4`, spatial/goal stable checks `3` | `120` | `7 -> 7` | `302.5` | `2.01x` vs baseline, `1.12x` vs target-EOS | speed-only passed, but exact validation failed on object task `0`, episode `1` with `max_action_diff=0.3084411323070526` |
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_stable4_trace_rq` | traced failing row, object checkpoint `152`, stable checks `4` | `1` | one bad refresh out of `30` | `879.0` validate | diagnostic only | bad refresh was step `210`, stability stop at checkpoint `152`, emitted `153` tokens while target fallback had `197` tokens |
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_stable4_late200_once_rq` | object checkpoint `152`, stable checks `4`, one late stability target-EOS fallback after step `200` | `1` | `0 -> 0`, `max_action_diff=0.0` | `270.6` speed / `798.9` validate | diagnostic only | fixes object task `0`, episode `1`; one fallback per episode kept speed much lower than uncapped fallback |
| `outputs/pi0fast_adaptive_mismatch_object1_ep1_ck152_stable4_late200_once_rq` | same once-capped late fallback | `1` | `0 -> 0`, `max_action_diff=0.0` | `263.1` speed / `852.1` validate | diagnostic only | preserves exactness on the task `1`, episode `1` row that killed the first hybrid |
| `outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_late200_once_hybrid/speed/target_eos_adaptive/libero_object/metrics.jsonl` | partial object speed shard, one late target-EOS fallback after step `200` | `6` | running shard stopped early | `268.6` | too slow | exceeded the object budget `255.8 ms/control`; do not finish this candidate as a global object policy |
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_stable4_continue200_once_rq` | one late stability continuation to action-end using existing KV cache | `1` | `0 -> 0`, `max_action_diff=0.0` | `264.7` speed / `807.7` validate | diagnostic only | exact and cheaper than target-EOS fallback, but still too slow unless a narrower trigger is found |
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_stable4_confirm192_rq` | disallow stops at checkpoints `152` and `160`, first stability stop at `192` | `1` | `0 -> 0`, `max_action_diff=0.0` | `279.3` speed / `801.2` validate | diagnostic only | exact but slower than action-end continuation; not speed-viable globally |

The empirical-vocab path now has two important correctness fixes in
`serving/pi0fast_token_hooks.py`: the sliced restricted LM head includes
`lm_head.bias`, and selected tokens use `argmax` over sorted candidate IDs
instead of `topk` tie ordering. These are required for parity with the full
head, but the heldout probe shows whitelist coverage is still the limiting
factor.

`scripts/build_pi0fast_empirical_vocab.py` can now expand observed token IDs by
a bounded neighborhood, e.g. `--expand-radius 8 --expand-min-token-id 240000
--expand-max-token-id 257151`. This is a training-free robotics analogue of
dynamic-vocabulary speculation: keep the active vocabulary compact, but cover
nearby FAST quantization bins around target-observed action tokens. The
`action4096` heldout probe suggests action equivalence may be recoverable with
a wider high-action band even when exact token identity differs.

A broader 2026-07-09 calibration used target-EOS traces from tasks 0-2 across
`libero_object`, `libero_spatial`, and `libero_goal` (`18` chunks, `3` shards).
The expanded whitelist had `1623` extra IDs. It validated action equality on
heldout task 3 for all three suites, but only object got a meaningful speedup.
Spatial and goal remained slow because the constrained decode still emitted
long FAST token sequences. Conclusion: static active vocab can address
exactness/generalization, but it is not enough for the final 120-task speed
target unless paired with a safe early-execution/char-stop verifier or another
mechanism that reduces target forward count.

The adaptive prefix path does not rely on an empirical vocabulary. On the
task-3 heldout sweep above, it reduced average target-EOS latency from
`398.2` to `269.1 ms/control` across object/spatial/goal with
`max_action_diff=0.0` in validation. Relative to the existing strict fixed
baseline average (`607.9 ms/control`), that heldout mean would be `2.26x`, but
this is not a final claim until the strict 120 matched gate passes.

A follow-up mini matched gate using the strict proof wrapper and the `.env` HF
token passed on task 3 across `libero_object`, `libero_spatial`, and
`libero_goal`. The artifact is
`outputs/robotics_spec_120_proof_mini_adaptive_rq/pi0fast_adaptive/gate.json`.
It measured baseline `619.5 ms/control`, target-EOS `401.2 ms/control`, and
adaptive `283.3 ms/control`, for `2.19x` vs fixed-budget baseline and `1.42x`
vs target-EOS. Validation covered `3` adaptive rows and `3` target-EOS rows with
`18` exact verifies total and `max_action_diff=0.0`. This run used
`--min-baseline-successes 0 --min-suite-baseline-successes 0`, so it is a speed
and exactness sanity check only; the strict 120 gate below is still required.

The first strict 120 adaptive speed-only run used stable checks `1` and was not
acceptable even though it was fast: it measured baseline `607.9 ms/control` and
adaptive `276.8 ms/control` (`2.20x`), but one row stopped at `227` steps while
the baseline and target-EOS rows ran `300` steps. Focused validation on
`libero_object` task `7`, episode `1`, seed `43` showed the issue was a real
action difference (`max_action_diff=0.5489519238471985`). Stable checks `2`
restored the `300` steps but still had `max_action_diff=0.2679973542690277`.
Stable checks `3` restored `300` steps and `max_action_diff=0.0` on that bad
row, with the speed row at `246.3 ms/control`.

The strict 120 adaptive speed-only rerun with stable checks `3` passed:
`outputs/robotics_spec_120_proof/pi0fast_adaptive_stable3/gate.json` has
`120` matched rows, `7 -> 7` successes, zero regressions, zero matched-step
mismatches, and `301.0 ms/control` (`2.02x`) versus the same `607.9 ms/control`
baseline. This is not the final claim: exact validation was started and the
second object row failed (`libero_object` task `0`, episode `1`, seed `43`) with
`max_action_diff=0.07857602834701538`.

The cheapest focused fix found for that row was adding checkpoint `152` and
using `--adaptive-early-stable-checks 4 --adaptive-early-max-stable-checkpoint
128`. Checkpoints `144` and `148` still had the same action diff; checkpoint
`152` had `max_action_diff=0.0`. Running that setting globally was too slow in
the early speed sample, but a suite-conditional hybrid is still viable:
object uses the checkpoint-`152` conservative setting, while spatial and goal
reuse stable checks `3`.

The first hybrid strict 120 speed-only diagnostic passed:
`outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_hybrid/gate.json`
has `120` matched rows, `7 -> 7` successes, zero regressions, zero matched-step
mismatches, and `302.4 ms/control` (`2.01x`) versus the `607.9 ms/control`
baseline. It is not usable as the final result: exact validation then failed on
`libero_object` task `1`, episode `1`, seed `43` with
`max_action_diff=0.3119494318962097`.

Stable checks `4` with checkpoint `152` fixed that second object failure in the
focused diagnostic
`outputs/pi0fast_adaptive_mismatch_object1_ep1_ck152_stable4_rq`:
`target_eos_adaptive_validate` had `60` exact verifies and
`max_action_diff=0.0`. A second hybrid speed-only root using that conservative
object setting passed:
`outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_hybrid/gate.json`
has `120` matched rows, `7 -> 7` successes, zero regressions, zero matched-step
mismatches, and `302.5 ms/control` (`2.01x`) versus the `607.9 ms/control`
baseline. Suite-level speeds were object `251.5 ms/control` (`2.25x`), spatial
`341.2 ms/control` (`1.87x`), and goal `314.8 ms/control` (`1.97x`). This was
also not usable as-is: focused validation under the same object setting failed
on `libero_object` task `0`, episode `1`, seed `43` with
`max_action_diff=0.3084411323070526`.

Trace export for that row showed exactly one bad refresh out of `30`: control
step `210`, a stability stop at checkpoint `152` that emitted `153` tokens; the
target-EOS fallback had `197` tokens. This motivated the new runner knobs in
`scripts/run_pi0fast_chunk_eval.py`:

- `--adaptive-checkpoint-stable-checks CHECKPOINT=CHECKS` for checkpoint-level
  stability thresholds.
- `--adaptive-stability-target-eos-after-step STEP` for target-EOS fallback on
  late stability stops.
- `--adaptive-stability-target-eos-max-fallbacks N` to cap those fallbacks per
  episode.
- `--adaptive-stability-continue-to-action-end-after-step STEP` to continue a
  risky stability stop to the action-end token using the existing KV cache.
- `--adaptive-stability-continue-to-action-end-max-fallbacks N` to cap those
  continuations per episode.
- `--adaptive-validate-label-all-stability-stops` to attach
  `fallback_action_max_diff` labels to every stability-stop candidate in
  `target_eos_adaptive_validate` traces, including exact matches. This is for
  gate-data collection only, not speed measurement.

Stability-stop feature export now exists in
`serving/pi0fast_token_hooks.py`. Stop candidates record scalar
`stability_stop_*` fields for checkpoint, token count, forced-EOS status,
logprob/entropy summaries, stable-check metadata, and action-shape features.
`scripts/extract_pi0fast_stability_gate_rows.py` turns token-trace shards into
JSONL rows filtered to stability stops.

Two focused label-all traces have been collected:

| Artifact | Task row | Stability rows | Safe / unsafe | Key result |
|---|---:|---:|---:|---|
| `outputs/pi0fast_adaptive_mismatch_object0_ep1_ck152_stable4_feature_trace_labeled_rq/stability_gate_rows.jsonl` | object task `0`, episode `1`, seed `43` | `8` | `7 / 1` | reproduces the bad step `210` row with `fallback_action_max_diff=0.3084411323070526` |
| `outputs/pi0fast_adaptive_mismatch_object1_ep1_ck152_stable4_feature_trace_labeled_rq/stability_gate_rows.jsonl` | object task `1`, episode `1`, seed `43` | `6` | `6 / 0` | all stability stops exact under the same checkpoint-`152`, stable-checks-`4` policy |

The combined labeled set is `13` safe rows and `1` unsafe row. A broad hand
threshold is not defensible yet: the unsafe step-`210` row is close to a safe
step-`220` row on the exported confidence/action features (`action_abs_max`
matches, entropy/logprob are nearby). Do not hard-code a one-negative rule as a
final safety gate without more labeled heldout rows.

The checkpoint-level threshold probe (`152=5`) did not fix task `0`; it delayed
the false positive to checkpoint `160`. The more promising focused candidate is
the once-capped late fallback: object checkpoint `152`, stable checks `4`,
`--adaptive-stability-target-eos-after-step 200`, and
`--adaptive-stability-target-eos-max-fallbacks 1`. It fixed object task `0`,
episode `1` with `max_action_diff=0.0` at `270.6 ms/control`, and preserved
object task `1`, episode `1` exactness with `max_action_diff=0.0` at
`263.1 ms/control`. It is not viable as a global object policy: the partial
object speed shard under
`outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_late200_once_hybrid`
was stopped after `6` rows because it averaged `268.6 ms/control`, above the
`255.8 ms/control` object budget needed for a `2.0x` 120-row hybrid.

The KV-cache continuation probe is directionally better than from-scratch
target-EOS fallback. On object task `0`, episode `1`, it had
`max_action_diff=0.0` at `264.7 ms/control`; forcing stability confirmation to
checkpoint `192` was also exact but slower at `279.3 ms/control`. Both are too
slow if applied broadly. The next required evidence step is not a full 120-row
run; it is a narrower late-stability safety gate. Recommended next moves:

- Use the existing feature export and extractor to build a larger labeled
  stop-candidate table:
  `--adaptive-validate-label-all-stability-stops` plus
  `scripts/extract_pi0fast_stability_gate_rows.py`.
- Collect more label-all stability-stop traces on heldout object tasks and the
  spatial/goal rows that supply the hybrid speed margin.
- Train or hand-check a conservative gate on stop candidates, with heldout
  tasks, that rejects the one bad late stability stop while keeping most safe
  checkpoint-`152` stops.
- Only after a gate keeps the object speed estimate below `255.8 ms/control`,
  rerun the full object speed shard and then the 120-row speed gate.

If a new gate passes speed, object validation should use the same
suite-conditional settings as the speed artifact. A template for the current
KV-cache continuation diagnostic is:

```bash
for mode in target_eos_adaptive_validate target_eos_validate; do
python scripts/run_pi0fast_chunk_eval.py \
  --task libero_object \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episodes 4 \
  --steps 300 \
  --modes "$mode" \
  --seed 42 \
  --output-dir outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_continue200_once_hybrid/validate/"$mode"/libero_object \
  --device cuda \
  --dtype bfloat16 \
  --smooth-position-delta 0.06 \
  --smooth-rotation-delta 0.22 \
  --enable-fast-token-hooks \
  --adaptive-prefix-checkpoints 32,64,96,128,152,160,192,224 \
  --adaptive-stable-checks 4 \
  --adaptive-stable-tolerance 0.0 \
  --adaptive-stability-continue-to-action-end-after-step 200 \
  --adaptive-stability-continue-to-action-end-max-fallbacks 1
done
```

Spatial/goal validation should use the stable-checks `3` setting:

```bash
for suite in libero_spatial libero_goal; do
for mode in target_eos_adaptive_validate target_eos_validate; do
python scripts/run_pi0fast_chunk_eval.py \
  --task "$suite" \
  --task-ids 0,1,2,3,4,5,6,7,8,9 \
  --episodes 4 \
  --steps 300 \
  --modes "$mode" \
  --seed 42 \
  --output-dir outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_continue200_once_hybrid/validate/"$mode"/"$suite" \
  --device cuda \
  --dtype bfloat16 \
  --smooth-position-delta 0.06 \
  --smooth-rotation-delta 0.22 \
  --enable-fast-token-hooks \
  --adaptive-prefix-checkpoints 32,64,96,128,160,192,224 \
  --adaptive-stable-checks 3 \
  --adaptive-stable-tolerance 0.0
done
done
```

After all validation shards exist, run the gate/final audit against
`outputs/robotics_spec_120_proof/pi0fast_adaptive_object_ck152_stable4_continue200_once_hybrid`.

For a fresh canonical `pi0fast-adaptive` run, use a new root or remove stale
stable-checks `1` adaptive speed shards before combining `--skip-existing` with
the wrapper:

```bash
python scripts/run_robotics_spec_120_proof.py \
  --path pi0fast-adaptive \
  --root outputs/robotics_spec_120_proof \
  --run-preflight \
  --require-hf-token \
  --run-synthetic \
  --run-final-audit \
  --render-result-card \
  --skip-existing
```

That wrapper expands to `baseline,target_eos,target_eos_adaptive`, uses
`target_eos` as the early-stop reference, and validates both
`target_eos_adaptive_validate` and `target_eos_validate`. The baked candidate
settings are:

```bash
--adaptive-prefix-checkpoints 32,64,96,128,160,192,224 \
--adaptive-stable-checks 3 \
--adaptive-stable-tolerance 0.0
```

The target-EOS strict gate remains useful as the reference/baseline:

```bash
python scripts/run_pi0fast_100_eval_gate.py \
  --root outputs/pi0fast_target_eos_120 \
  --speed-modes baseline,target_eos \
  --candidate-mode target_eos \
  --run-preflight \
  --require-hf-token \
  --run-synthetic \
  --run-final-audit \
  --render-result-card \
  --skip-existing
```

For a checkpoint-free speculative candidate, try `pattern_sd_direct`. It drafts
from repeated, periodic, and constant-velocity structure in the FAST action
token stream, then lets PI0-FAST verify drafted tokens before emission. The
runner's default validation mode is `auto`, so this run validates
both `pattern_sd_validate` and `target_eos_validate`:

```bash
python scripts/run_pi0fast_100_eval_gate.py \
  --root outputs/pi0fast_pattern_sd_120 \
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

For non-`target_eos_*` speculative candidates, include `target_eos` in
`--speed-modes` and use `--reference-mode target_eos` so the result is compared
against early stop, not only against fixed-budget decode. The wrapper defaults
those speculative candidate reference thresholds to the main gate thresholds, so
pattern/block candidates must also clear speedup, success-drop, and regression
checks versus `target_eos`. The adaptive path is a `target_eos_*` variant, so
its hard `2.0x` speed gate is against the fixed-budget baseline while
validation checks action equality against `target_eos`.

To screen pattern settings before the expensive simulator run, sweep saved
PI0-FAST FAST-token traces:

For fresh PI0-FAST evidence, collect the traces from `target_eos`
early-stop decode, not from a fixed token budget. The new pipeline collects
per-suite `target_eos` shard files, runs the heldout pattern sweep, and stages
the 120-task gate with `--reference-mode target_eos`:

```bash
python scripts/run_pi0fast_pattern_candidate_pipeline.py \
  --root outputs/pi0fast_pattern_candidate \
  --gate-dry-run
```

Pass `--no-gate-dry-run` only when launching the real 120-task simulator gate.
The pipeline passes `--pattern-sweep-rank 0` and
`--pattern-auto-source-min-metrics` to the gate runner by default, so the gate
auto-selects the first ranked row that passes train, heldout, task-count, and
per-row source-usage checks before the full simulator run starts.
It also defaults `--pattern-required-source-coverage auto`, deriving required
source families from enabled sweep modes and rejecting capped sweeps whose
summary never evaluated one of those families.

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
  --action-prefix-lookup both \
  --action-prefix-history-sizes 4 \
  --action-prefix-top-ks 3 \
  --action-prefix-min-prefixes 1 \
  --action-prefix-max-mismatches-values 0 \
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
  --chunk-position-delta both \
  --chunk-position-delta-min-counts 1,2 \
  --chunk-delta-template both \
  --chunk-delta-template-max-delta-mismatch-values 0 \
  --min-source-agreements 1,2 \
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

This trace sweep is only a tuning proxy. The 120-eval gate remains the
evidence needed for the final 0-drop / 2x claim.
For capped sweeps, check `evaluated_source_counts`,
`evaluated_source_coverage`, and `evaluated_enabled_source_count_histogram` in
the sweep JSON before promoting a row; those fields show which robot-prior
families were actually evaluated before `--max-configs` truncated the search.
The sweep grid is conditional: disabled optional sources keep only the first
provided history/top-k/context setting, so equivalent inactive configs do not
inflate ranking or waste offline trace evaluation.

The sweep now includes exact-verification latency knobs inspired by recent LLM
speculative decoding successes: reuse fully verified blocks in the target KV
cache, emit the verifier's bonus token after full-block acceptance, and adapt
the draft window from the previous verified acceptance. It also sweeps guarded
second-order action-token extrapolation for smooth robot segments, plus a
prompt-lookup n-gram continuation prior that drafts from suffix matches in
already verified FAST tokens and recent verified chunks, an action-delta
histogram prior over same-action-dimension token deltas in the current prefix
and recent verified chunks, an action-trend regression prior that projects the
next same-dimension token from a short verified-prefix trend, an
intra-action-prefix lookup prior that predicts later dimensions from prior
verified action vectors with the same already-emitted prefix, an
action-token-neighborhood prior that proposes a small numeric band around
smooth extrapolations, recent verified chunk positions, and other enabled-prior
centers for exact tree verification, a
same-action-dimension context-tree prior that learns short verified
same-dimension histories with depth backoff, a dimension-aware action-transition
histogram prior, a per-position modal-token
histogram over recent verified chunks that share the current prefix, a
same-action-dimension modal-token prior for held robot joints and gripper
states, a direct hold-action-token prior that repeats the previous token from
the same action dimension, and a source-agreement prior that prefers tokens
proposed by at least two configured cheap sources. It can also sweep
`--source-cooldown both`, an acceptance-aware scheduler that temporarily skips a
source after verified rejections and falls back to the next cheap source.
These preserve target-token exactness; they only change how aggressively the
cheap pattern drafter proposes future FAST action tokens.

When promoting a sweep row into the real 120-eval wrapper, use named metric
thresholds for any source-specific prior you enable. For example,
`--pattern-min-metric ngram_continuation_accepted_tokens=1` and
`--pattern-min-heldout-metric ngram_continuation_accepted_tokens=1` require the
selected row to show accepted prompt-lookup tokens on both ranking and heldout
splits before simulator shards are launched. For agreement rows, use
`source_agreement_accepted_tokens=1` and, when coverage across tasks matters,
`min_task_source_agreement_accepted_tokens=1`. For action-delta rows, use
`action_delta_histogram_accepted_tokens=1` or the min-task variant; higher
`action_delta_min_count` values require repeated same-dimension deltas rather
than a singleton recent velocity. For
action-prefix-lookup rows, use `action_prefix_lookup_accepted_tokens=1` or
`min_task_action_prefix_lookup_accepted_tokens=1`. For action-vector-suffix
lookup rows, use `action_vector_suffix_lookup_accepted_tokens=1` or
`min_task_action_vector_suffix_lookup_accepted_tokens=1`. For action-vector-transition
rows, use `action_vector_transition_accepted_tokens=1` or
`min_task_action_vector_transition_accepted_tokens=1`. For chunk-length stop rows,
use `chunk_length_stop_accepted_tokens=1` or
`min_task_chunk_length_stop_accepted_tokens=1` so the selected row actually
accepts verified `target_eos` stop tokens from the prior. For
action-token-neighborhood rows, use
`action_token_neighborhood_accepted_tokens=1` or the min-task variant. For
action-context-tree rows, use `action_context_tree_accepted_tokens=1` or
`min_task_action_context_tree_accepted_tokens=1`. For
target-anchored tree rows, require `tree_anchor_accepted_tokens=1` or
`min_task_tree_anchor_accepted_tokens=1` so the anchor path accepted verified
continuation tokens in the offline split. For
transition rows, use `action_transition_histogram_accepted_tokens=1` or the
min-task variant. For the position-mode row, use
`position_mode_histogram_accepted_tokens=1` and
`min_task_position_mode_histogram_accepted_tokens=1` when every offline task
should exercise that prior. For action-dimension-mode rows, use
`action_dimension_mode_accepted_tokens=1` and
`min_task_action_dimension_mode_accepted_tokens=1`. For global-position rows, use
`global_position_mode_accepted_tokens=1` and
`min_task_global_position_mode_accepted_tokens=1` to prove the prefix-free
per-position prior accepted verified tokens. For hold-token rows, use
`hold_action_token_accepted_tokens=1` and
`min_task_hold_action_token_accepted_tokens=1`.
For chunk-position-delta rows, use `chunk_position_delta_accepted_tokens=1`
and `min_task_chunk_position_delta_accepted_tokens=1`; higher
`chunk_position_delta_min_count` values require repeated same-position deltas
instead of a single recent chunk.
Use `--pattern-auto-source-min-metrics` to have the gate runner derive those
accepted-token and `min_task_*` source checks from the selected sweep row; it
also mirrors them onto heldout metrics when the row contains heldout data.
When `--run-final-audit` is used, the wrapper passes `run_manifest.json` to the
audit and adds `--require-pattern-source-coverage` for sweep-selected pattern
candidates. Positive required-source counts are therefore part of the final
evidence, not just staging metadata.

Also sweep source ordering with
`--source-priority-modes default,lookup_first,smooth_first`. This lets heldout
traces choose whether lookup-style priors or smooth action-token extrapolation
draft first, while online evaluation replays the chosen order with
`--pattern-source-priority`. Pair this with `--source-cooldown both` when a
source is useful on some phases but should back off immediately after verified
mismatches.

There is also an exact tree screen via `--tree-widths 1,4`. It builds a small
beam from the same robot priors and simulates exact tree verification, which is
the robotics analogue of recent dynamic/traversal tree speculative decoding.
The candidate pipeline now includes this compact tree screen by default, along
with `--action-trend-regression both`, `--action-delta-ngram both`, and
`--chunk-delta-template both` for
velocity-space prompt lookup over recent verified chunks, and
`--source-acceptance-bias both` for verifier-feedback source ordering. Rows
with `tree_width > 1` now convert to
`--pattern-tree-width` and use the online PI0-FAST tree verifier. That path
batches several candidates in one target pass and commits only the selected
target-matching prefix, but it still needs the full 120-task validation gate
before any final claim. Add
`--dynamic-tree-width both` with `--tree-widths 1,4` to screen an EAGLE-2-style
dynamic draft tree adapted to robotics: the candidate count grows after full
verified acceptance and shrinks after misses or partial acceptance. Add
`--tree-anchor-target-token both` to screen an exact target-token anchored
fallback: if every tree row starts with the wrong token, the verifier anchors on
the target greedy token and only drafts continuations after it. Add
`--tree-anchor-target-continuation both` to always anchor on that already-known
target greedy token and verify only drafted futures after it. The default
candidate pipeline now includes `--chunk-prefix-retrieval both` to screen a
retrieval prior for near-repeated robot chunks with a small bounded prefix
mismatch. Add
`--action-token-neighborhood both` with tree rows to screen nearby quantized
FAST-token candidates around smooth, recent-chunk, and enabled-prior centers.
The wrapper
auto-adds candidate `trace_stats` gate checks for tree rows: selected tree
width, nonzero tree verification, dynamic-tree stats when enabled, and zero
unverified pattern-token shortcuts.

Pass the sweep JSON directly to the 120-eval runner so the selected row is
recorded in `run_manifest.json`:

```bash
python scripts/run_pi0fast_100_eval_gate.py \
  --root outputs/pi0fast_pattern_sd_120 \
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

`scripts/pattern_sweep_to_eval_args.py` remains available if a shell argument
string is needed for manual runs.

The per-task sweep thresholds are deliberately weaker than the final 120-task
gate. They are a guard against selecting a row whose aggregate speed is carried
by a small subset of trace tasks. The heldout thresholds add a second guard
against selecting a row that only works on the trace subset used for ranking.
With `--pattern-sweep-rank 0`, these thresholds are applied while scanning the
ranked rows, and the manifest records both the requested rank and the actual
selected rank.

The older strongest prefix-cutoff / learned-gate result remains useful for
more aggressive early execution, but it is no longer the primary 2x / 0-drop
path.

If the work pivots to PI0.5, do not reuse the PI0-FAST FAST-token modes as-is.
LeRobot PI0.5 uses flow-action sampling in this runner, and a smoke on
`libero_object` task 0 showed:

- `... --policy-kind pi05 --pi05-disable-compile --num-inference-steps 4 --modes baseline --steps 2`
  reached rollout and produced a baseline row.
- Adding `target_eos` failed because `PI05Config` has no PI0-FAST token decode
  fields such as `temperature`.

The runner and 120-eval wrappers now fail early for PI0.5 plus PI0-FAST
FAST-token modes (`target_eos`, `target_cutoff`, `pattern_sd`, etc.). A PI0.5
speculative result should still be compared against a PI0.5 stop-token
early-stop reference, but that needs a PI0.5-specific stop-token/token adapter
first.

For an OpenVLA/SpecVLA pivot, use `scripts/run_openvla_120_eval_gate.py` and
keep its default matched-step and strict `spec_stats` checks enabled. The
OpenVLA gate now fails final runs with mismatched AR/spec control-step counts,
unverified `fast_draft_only` actions, chunk-buffer actions, relaxed fast-draft
acceptances, or approximate tree depth above 1. The
`--allow-unverified-spec-shortcuts` flag is for research sweeps only, not for
the final 0-drop / 2x evidence; the wrapper requires
`--skip-audit --skip-result-card` when that flag is used.
The final objective audit now independently requires those strict OpenVLA
thresholds and zero shortcut counters when `--openvla-gate` is supplied.

| Eval | Mode | Success | ms/control | Avg FAST tokens | Speedup |
|---|---:|---:|---:|---:|---:|
| Tasks `0-4`, one episode each | Full PI0-FAST EOS baseline | `1/5` | `299.1` | `101.9` | `1.00x` |
| Tasks `0-4`, one episode each | DAgger prefix gate adaptive | `1/5` | `169.9` | `27.3` | `1.76x` |
| Task `2`, one episode | Full PI0-FAST EOS baseline | `1/1` | `322.1` | `95.2` | `1.00x` |
| Task `2`, one episode | Fixed `target_cutoff16` | `1/1` | `180.7` | `17.0` | `1.78x` |
| Task `2`, one episode | DAgger prefix gate adaptive | `1/1` | `192.1` | `19.0` | `1.68x` |

Important caveat: the 5-task validation has weak accuracy evidence because the baseline itself only solved `1/5`. On the one task where baseline succeeded, both `cutoff16` and the DAgger gate preserved success.

## What The Current Method Does

Baseline PI0-FAST:

1. Generate FAST action tokens to the fixed decoding budget.
2. Detokenize full FAST token sequence into a continuous action chunk.
3. Execute the chunk.

Target-EOS early stop:

1. Generate FAST action tokens until the target model emits action-end (`|`).
2. Stop without sampling the ignored fixed-budget tail.
3. Detokenize the same action text into a continuous action chunk.
4. Execute the chunk.

Prefix cutoff method:

1. Generate only the first `N` FAST tokens.
2. Append action-end token.
3. Detokenize the partial FAST sequence into an action chunk.
4. Execute the chunk.

Adaptive gate method:

1. Try checkpoints like `24,32,40,48,56,64` tokens.
2. For each candidate prefix, append action-end and detokenize.
3. Compute token/action features:
   - cutoff length
   - token logprobs / entropy
   - action magnitude
   - max step delta
   - jerk
   - gripper change
   - position/rotation span
4. A small MLP predicts whether the partial decoded action chunk matches the full target chunk closely enough.
5. Stop at the first checkpoint passing threshold.

This is not classic LLM speculative decoding with a draft model and exact token verification. It is closer to learned early-exit / robot-aware speculative prefix execution over FAST action tokens.

## Results From Failed Faster Paths

| Mode | Success | ms/control | Avg FAST tokens | Approx speed | Takeaway |
|---|---:|---:|---:|---:|---|
| `target_cutoff8` on task 2 | `0/1` | `137.6` | `9.0` | `~2.34x` | Fast enough, not correct |
| `target_cutoff12` on task 2 | `0/1` | `144.3` | `13.0` | `~2.23x` | Fast enough, not correct |
| Early learned gate on task 2 | `0/1` | `138.7` | `9.0` | `2.36x` | Over-accepted 8-token prefixes |
| Early gate threshold `0.9999` | `0/1` | `178.6` | `31.9` | `~1.8x` | Still failed despite tighter threshold |
| `cutoff12_warmup1` | `0/1` | `155.0` | `21.1` | fast | Warmup did not fix drift |
| `cutoff12_warmup2` | `0/1` | `173.6` | `29.2` | fast | Warmup did not fix drift |
| `cutoff8_warmup2` | `0/1` | `165.1` | `25.5` | fast | Warmup did not fix drift |

Interpretation: `8-12` FAST tokens is where `2x+` speed lives, but those prefixes are too lossy for the current verifier. `16` tokens is the current fastest successful cutoff on task 2.

## Other Paths Tried Earlier

### Token-Level / N-Gram SD

FAST-token n-gram speculative path produced strong decode-only speed in isolation, around `2.45x`, but it diverged from target tokens by position 3. In rollout it failed and became slower due guard rejections.

Conclusion: exact FAST-token SD with simple n-gram drafting is not enough.

### Medusa / Learned Head

A learned Medusa-style trajectory/token head was trained and tested. Offline it had some apparent acceptance, but online rollout was not useful. It predicted lower-error tails than naive methods, but they were not smooth enough and did not produce robust end-to-end speedups.

Conclusion: learned heads are still plausible, but the current version needs better targets and a stronger verification/guard story.

### EAGLE-Style Drafting

EAGLE-style attempts did not produce a useful online speedup in this setup. The main issue is that action-token mistakes early in the FAST sequence decode into materially different continuous actions, so partial token correctness is not enough unless verification is very strong.

## Most Likely Paths To Success

### 1. Calibrated DAgger Prefix Gate Over `16/24/32/...`

Most likely to produce a real `1.5x-1.8x` success-preserving result quickly.

Why:

- Already got `1.76x` on the 5-task batch with no measured success drop.
- Already got `1.68x-1.78x` on the solved held-out task.
- It uses target PI0-FAST itself as the generator, just stops early.
- No draft-model synchronization or KV-cache correctness issue.

What to do next:

- Generate more DAgger data from adaptive rollouts, not only baseline states.
- Use multiple seeds and all LIBERO object tasks.
- Hold out entire tasks, not just random rows.
- Train cutoff-specific calibrated gates, especially for `16`, `24`, `32`.
- Avoid `8/12` until the verifier has explicit negative examples from failed online rollouts.
- Report success only on tasks where baseline has nonzero success.

Expected upside:

- Realistic near-term target: `1.5x-1.8x` with low success drop.
- `2x` is unlikely without accepting riskier prefixes or changing execution horizon.

### 2. Fixed `cutoff16` As A Strong Baseline

Good simple baseline, not yet proven general.

Why:

- On task 2, `cutoff16` was fastest successful: `1.78x`, no success drop.
- No learned model required.

Risk:

- Could be task/seed lucky.
- `cutoff8/12` failed, so there is a sharp quality boundary.

What to do next:

- Run `target_eos` vs `target_cutoff16` on all tasks/seeds where baseline succeeds.
- If `cutoff16` holds across tasks, it becomes the cleanest speedup story.

### 3. Learned Gate With Conservative Online False-Positive Penalty

Could eventually push toward `2x`, but current early gate failed.

Why:

- `8/12` prefixes are fast enough for `2x+`.
- Some `8/12/16` prefixes are exact in offline labels.

Current failure:

- The early gate had validation precision `1.0` on random held-out rows, but failed task 2 online by over-accepting unsafe early prefixes.

Fix:

- Train on online failure states, not just baseline states.
- Add a high-cost false-positive loss.
- Calibrate separately per cutoff.
- Require a conservative fallback: never accept `8/12` unless both token confidence and action smoothness are extreme.
- Use a rollout-level validation metric, not row-level accuracy.

Expected upside:

- If it works, this is the path to `2x`.
- Higher risk than `cutoff16` / DAgger gate.

### 4. Real Draft-Then-Verify Speculative Decoding

Scientifically closer to LLM SD, but less likely to give short-term wins in this setup.

Why it is hard:

- FAST token prefixes do not map cleanly to control-action prefixes.
- Wrong early FAST tokens decode into wrong continuous action chunks.
- Exact verification saves correctness but often kills speed due target verification cost.
- Relaxed verification gets speed but risks rollout drift.

What might make it work:

- Draft only after stable context positions.
- Verify decoded action chunk, not only token equality.
- Use target logits plus action-space checks.
- Train the draft model on rollout states induced by its own drafts.

Expected upside:

- More publishable if it works.
- Higher engineering cost and more uncertain than prefix cutoff/gating.

## Recommended Next Experiment

Run a larger validation focused on the two promising modes:

- `target_eos`
- `target_cutoff16`
- DAgger `target_eos_adaptive` with checkpoints `24,32,40,48,56,64`

Use:

- LIBERO object tasks `0-9`
- multiple seeds
- enough episodes to separate baseline failures from acceleration failures
- only compute success-drop on the subset where baseline succeeds

Primary table:

| Metric | Baseline | `cutoff16` | DAgger gate |
|---|---:|---:|---:|
| Success rate on all episodes | | | |
| Success rate on baseline-success subset | | | |
| ms/control | | | |
| model call ms | | | |
| avg FAST tokens | | | |
| model calls/step | | | |
| speedup | | | |

Decision rule:

- If `cutoff16` keeps success within `5%`, use it as the speed baseline and build verifier/gate around it.
- If `cutoff16` drops success but DAgger gate holds, continue DAgger gate.
- If both fail broadly, the current prefix-cutoff approach is task-specific and needs online DAgger plus action-space verification before scaling.

## File / Output Pointers

Key outputs:

- `outputs/pi0fast_adaptive_gateonly98_object0_4_clean/separate_summary.json`
- `outputs/pi0fast_fast_push_task2_cutoffs/separate_summary.json`
- `outputs/pi0fast_fast_push_task2_cutoffs_8_12/separate_summary.json`
- `outputs/pi0fast_early_gate_object0134_task2_eval/separate_summary.json`
- `outputs/pi0fast_early_gate_object0134_task2_thr9999/separate_summary.json`
- `outputs/pi0fast_task2_aggressive_warmups/separate_summary.json`

Key code:

- `scripts/run_pi0fast_chunk_eval.py`
- `scripts/run_pi0fast_separate_process_eval.py`
- `scripts/generate_pi0fast_prefix_gate_data.py`
- `scripts/train_pi0fast_prefix_gate.py`
- `serving/pi0fast_prefix_gate.py`
- `serving/pi0fast_token_hooks.py`

## 2026-07 Deferred-Correction Pattern Probe

Implemented an opt-in `pattern_sd` correction-token deferral path:

- Online flag: `--pattern-defer-correction-token`
- Offline flag: `--defer-correction-token`
- Sweep/pipeline flag: `--defer-correction-token`

The change emits a rejected-block correction token but leaves it pending so the
next verifier pass can process the correction and future draft together. This
is exact in the fake-model tests and improves offline target-forward accounting.

Results:

- Existing 60-row target-eos trace mini, tree width 4 + anchor continuation:
  - `outputs/pi0fast_pattern_offline_existing_target_rows/tree_anchor_cont_defercorr.json`
  - `target_forward_reduction = 2.7485x`
  - `min_task_target_forward_reduction = 1.9818x`
  - `deferred_correction_tokens = 1487`
- Same mini, tree width 2 + anchor continuation:
  - `outputs/pi0fast_pattern_offline_existing_target_rows/tree2_anchor_cont_defercorr.json`
  - `target_forward_reduction = 2.7100x`
  - `min_task_target_forward_reduction = 1.9739x`
- Same mini, no tree:
  - `outputs/pi0fast_pattern_offline_existing_target_rows/multiprior_defercorr.json`
  - `target_forward_reduction = 1.5003x`
  - `min_task_target_forward_reduction = 1.0187x`
- Live 20-step object0 smoke, tree width 4 + anchor continuation:
  - `outputs/pi0fast_pattern_tree_anchor_defercorr_validate_object0_20_rq_osmesa/summary.json`
  - `pattern_sd_validate = 8111.2 ms/control`
  - `target_eos = 188.0 ms/control`
  - Conclusion: the tree verifier's wall-clock overhead dominates despite good offline forward counts.

Takeaway:

- Correction deferral is worth keeping as an exact SD primitive.
- Current tree-pattern speculation is not a viable speed path unless verifier
  batching becomes much cheaper or candidate count is gated aggressively.
- Non-tree pattern speculation remains below the 2x target on the current trace mini.

## Bottom Line

The most likely near-term success is not a full LLM-style draft model. It is a conservative FAST-prefix early-stop verifier:

- `cutoff16` already hits `1.78x` on the solved held-out task.
- DAgger gate already hits `1.76x` on the 5-task batch with no measured success drop.
- `8/12` tokens can hit `2x+`, but current verification is not good enough and causes failure.

Best next move: validate `cutoff16` and DAgger gate on a larger baseline-success subset, then train the gate with online DAgger failure states to safely reclaim some `8/12`-token opportunities.
