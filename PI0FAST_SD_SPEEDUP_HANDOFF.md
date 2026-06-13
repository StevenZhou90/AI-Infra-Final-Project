# PI0-FAST Speculative / Prefix Speedup Handoff

## Objective

Find a robotics analogue of speculative decoding for PI0-FAST that gives roughly `>=1.5x` end-to-end control latency speedup with at most `~5%` success loss.

The current experiments use `lerobot/pi0fast-libero` in LIBERO object tasks with OSMesa rendering. The model is PI0-FAST, so actions are represented as FAST discrete action tokens. We can exploit this by stopping action-token generation early, appending the action-end token, detokenizing a partial action chunk, and executing the resulting action chunk.

## Current Best Result

The strongest usable result is a FAST-token prefix cutoff / learned gate path.

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

1. Generate FAST action tokens until the model emits action-end / EOS.
2. Detokenize full FAST token sequence into a continuous action chunk.
3. Execute the chunk.

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

## Bottom Line

The most likely near-term success is not a full LLM-style draft model. It is a conservative FAST-prefix early-stop verifier:

- `cutoff16` already hits `1.78x` on the solved held-out task.
- DAgger gate already hits `1.76x` on the 5-task batch with no measured success drop.
- `8/12` tokens can hit `2x+`, but current verification is not good enough and causes failure.

Best next move: validate `cutoff16` and DAgger gate on a larger baseline-success subset, then train the gate with online DAgger failure states to safely reclaim some `8/12`-token opportunities.
