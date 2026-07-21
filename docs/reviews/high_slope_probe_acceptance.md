# High-slope sampling probe: independent acceptance report

## Scope and immutable decision rule

This report belongs to the independent Acceptance/Decision Agent.  The agent
does not modify production code, start PPO, or use the GPU.  PRE-GPU was run
from clean baseline `b102a4273cad1e6077ec89bd8c6ef1015597b196` on branch
`agent/highslope-probe-acceptance` on 2026-07-21.

The candidate is either **ACCEPT** or **REJECT** after the complete post-training
matrix.  Thresholds below are locked before candidate metrics are inspected and
must not be relaxed.  A missing, non-finite, unmatched, or malformed artifact is
a failed gate, not an ignorable sample.

## PRE-GPU result: PASS

The repository and CPU contracts are training-ready:

- Worktree started clean and exactly at `b102a42`.
- `python -m compileall -q scripts src tests`: PASS.
- Full `unittest` discovery: 321 PASS and one known unrelated placeholder skip.
- High-slope targeted contracts: 82/82 PASS.  These cover the sampler and runner
  state, matched geometry and lifecycle, evaluator JSON/metric invariants,
  controller-headroom identity, and attribution rules.
- Task registry/config/RL config/runner load: PASS.  The registered task is
  `Unitree-Go2-Rough-V7-HighSlopeProbe`, its runner is
  `VelocityOnPolicyRunner`, and the first reset event is
  `high_slope_sampling`.
- The task-aware CLI invocation
  `python scripts/train.py Unitree-Go2-Rough-V7-HighSlopeProbe --help`: PASS.
  The CLI requires the positional task before `--help`; invoking bare
  `scripts/train.py --help` is not a valid project CLI contract.
- `git diff --check b102a42`: PASS.
- V7 checkpoint exists, is 7,077,833 bytes, and has SHA-256
  `73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`.
- Final process check found no train/evaluate/play/RSL-RL process.  The GPU was
  idle at 0 MiB and 0% utilization; it was queried only for status.

`pytest` is not installed in the conda environment, so the repository's actual
standard-library `unittest` entry point was used.  This is not a code failure.

## Single-variable audit

The probe derives from `unitree_go2_rough_v7_env_cfg()` and adds only:

1. a reset event targeting
   `H = slope_up levels 8/9 + slope_down level 9` at ratio `0.10`; and
2. sampler telemetry metrics.

The contract test removes those additions and compares the resulting config to
V7.  Rewards, commands, terminations, observations, actions, original events,
original metrics, curriculum, and terrain config are equal.  Play config is
unchanged V7.  The sampler runs before `reset_base`, preserves the conditional
slot distributions inside H and non-H, changes only the required membership
mismatches, relocates terrain origins, and persists RNG/quota/count/histogram
state.  Old V7 state rebases the sampler after terrain restoration.

This satisfies the declared single variable: aggregate high-slope reset exposure
from nominal 3% (V7 snapshot 64/2048 = 3.125%) to target 10%.  Reward, command
distribution, terrain geometry, termination, gait, randomization, height scan,
observations, network, and PPO parameters remain frozen.

## Evaluator artifact contract

Both formal baseline A/B JSON files parse, are recursively finite, and pass the
production `validate_headroom_result` contract:

- clean SHA-256
  `be2b583d6b72ca24c1df8464d0883022d73e52a2f864814ea68b259166527e5a`;
- randomized SHA-256
  `31e80f3812b2bb9c6c9766d65d724ffbfe55617b48ffce028038907018e9c578`.

They identify V7 `model_13600.pt`, task `Unitree-Go2-Rough-V7`, seed 42, the
correct clean/randomized profile, fresh environment per route kind, strict A/B
identity, explicit active-sample denominators, and frozen-attempt lifecycle.
The first failed pre-fix rollout without JSON remains inadmissible.

Every post-training result must preserve the same checkpoint-independent
identity: evaluator schema, task/play configuration, seed, profile,
randomization, route kind and slot order, terrain geometry, radius 2.5 m, speed
0.5 m/s, 2400-step horizon, controller settings, completion tolerances, active
sample denominator, contact definitions, and attempt-freeze semantics.  Only
checkpoint identity and resulting dynamic rollout metrics may differ.

Required JSON fields include completion/progress; commanded and actual vx/vy/wz;
response gain; cross-track and heading RMS/P95/max/final; slip and action
acceleration mean/P95/max with denominators; fell/base/upper-leg/calf contacts
and terminations; first failure reason; terrain level/type; relocation error;
and sampler reset count, hard count, ratio, and slot histogram.  Completed rows
must have null failure reason; failed rows must have an explicit real reason.
Reset ends the original attempt and no subsequent episode may add progress.

## Locked post-training gates

The candidate must pass every gate:

1. Clean high-slope completion is at least 0.70 for each of straight, arc, and
   S; randomized completion is at least 0.60 for each route.
2. Each route and profile improves completion by at least +0.20 absolute over
   the exactly matched V7 baseline.
3. High-slope mean forward response gain is at least 0.80.  No retained suite's
   forward gain may be more than 0.05 below its matched V7 value.
4. Cross-track and heading must satisfy the evaluator's predeclared completion
   tolerances (0.30 m and 20 degrees) and all progress/error values must be
   finite.  The evaluator configuration and horizons cannot be changed.
5. For each matched suite, slip and action-acceleration mean/P95/max may not
   exceed 1.2 times V7.  Denominators must cover the same original-attempt scope.
6. No aggregate suite may increase catastrophic fell/base/upper-leg/calf
   termination count or rate versus its matched V7 baseline, and no new
   catastrophic termination class is permitted.  Non-terminating contact counts
   and rates must not materially increase; any apparent increase is treated as
   FAIL unless the predeclared same-attempt evidence shows it is no larger than
   baseline measurement granularity.
7. Level-9 stairs, seeds 42/43/44, must retain at least 2/3 up and 2/3 down,
   introduce no new base/upper-leg/fall failure, and not exceed V7's total of two
   calf failures across the six direction/seed attempts.
8. Patch regression remains 112/112 clean and 112/112 randomized.  Continuous
   regression remains at least 12/12 clean and 10/12 randomized, without a new
   failure/contact class.
9. Cumulative sampler exposure must be auditable and close to 10%; integer
   reset/hard counts, histogram, RNG/quota persistence, terrain origins, and
   checkpoint round-trip must agree.  The first full 2048 reset contract is
   exactly 204 hard cases (9.96094%) with only the theoretical minimum slot
   membership changes from the restored population.
10. Candidate selection must follow the frozen order: clean high-slope
    completion, worst-route completion, mean forward gain, contact/fall count,
    then slip/action acceleration.  A single favorable metric cannot override a
    failed gate.

If any gate fails, the decision is **REJECT**, V7 `model_13600.pt` remains the
default, and this experiment does not authorize a second variable or additional
training.  PRE-GPU PASS authorizes only the fixed 2048-env, 400-iteration,
seed-42 training run; it does not pre-accept any resulting checkpoint.

## Pending POST-training audit

No candidate checkpoint or post-training JSON was available during PRE-GPU.
The final ACCEPT/REJECT section will be appended only after the integration HEAD,
training telemetry, candidate-selection evidence, and complete matched matrices
are supplied for independent review.
