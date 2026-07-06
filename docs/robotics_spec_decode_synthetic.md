# Synthetic robotics speculative decoding check

This is a CI-friendly check for the algorithmic idea behind adapting LLM
speculative decoding to robot action tokens.

The benchmark models each robot action as a small vector of discrete tokens.
Tasks are generated with long smooth motion phases and occasional phase changes.
A cheap robotics drafter predicts future tokens with a constant-velocity action
prior, then an exact target verifier accepts matching draft tokens and replaces
mismatches with target tokens. Because every emitted token comes from the target
stream or an exactly verified draft, the decoded action-token sequence is
identical to baseline autoregressive generation.

Run:

```bash
python scripts/benchmark_robotics_spec_decode_synthetic.py \
  --num-tasks 120 \
  --min-speedup 2.0 \
  --max-accuracy-drop 0.0 \
  --output outputs/robotics_spec_synthetic/gate.json \
  --markdown
```

Observed in the local CPU-only environment:

| metric | value |
| --- | ---: |
| tasks | 120 |
| exact token match | true |
| baseline success | 120/120 |
| spec success | 120/120 |
| accuracy drop | 0.00% |
| speedup | 5.44x |
| target forward reduction | 6.40x |
| acceptance rate | 61.60% |
| rejected blocks | 1321 |

Interpretation:

- This shows the LLM draft-verify pattern can exploit the narrower robot
  action-token distribution while preserving exact target outputs.
- The real PI0-FAST checkpoint-free analogue is `pattern_sd_direct`, which
  drafts from repeated, periodic, and constant-velocity FAST-token structure
  and still verifies drafted tokens with PI0-FAST before emission.
- This is not the final robotics-model claim. The authoritative PI0-FAST claim
  still requires `scripts/run_pi0fast_100_eval_gate.py` to pass on 120 real
  LIBERO matched evaluations with `target_eos` validation.
- Once both artifacts exist, use `scripts/audit_robotics_spec_goal.py` to check
  the full objective evidence in one place.
