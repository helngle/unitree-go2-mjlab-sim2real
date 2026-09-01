# V7 local-tangent loaded-stance slip shaping: training design

Date: 2026-07-24

## Decision

The formal friction source/sham/probe v13 acceptance is `CONTACT_CAUSAL` and
`training_ready=true`.  The first controlled training task is implemented as
`Unitree-Go2-Rough-V7-StanceSlip` and has passed CPU contracts plus a real
2048-environment, full-resume, no-learning GPU preflight.  The training design
status is **TRAINING-READY**.  PPO training has not been started by this work.

The locked warm start remains:

```text
checkpoint: logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256: 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

V7 stays the default model until one candidate passes every post-training gate.
The rejected `model_13900.pt`, `model_13999.pt`, and `model_14099.pt` are not
eligible warm starts or defaults.

## Single-variable reward definition

For foot `i`, select the nearest valid yaw-aligned terrain ray in horizontal
distance, requiring distance at most `0.25 m`.  Let its upward-oriented unit
normal be `n_i`, foot world velocity be `v_i`, and the contact sensor force be
`F_i`.  The configured foot-primary/terrain-secondary sensor reports net force
on the terrain, so supporting normal load is:

```text
Fn_i = max(0, -dot(F_i, n_i))
```

Loaded stance is registered only when all are true:

```text
foot contact is present
nearest terrain ray and normal are valid
Fn_i >= 15 N
```

Terrain-relative tangent slip and bounded per-foot cost are:

```text
vt_i  = v_i - dot(v_i, n_i) * n_i
s_i   = norm(vt_i)
phi_i = min(4, (max(s_i - 0.03 m/s, 0) / 0.10 m/s)^2)
```

For moving commands (`norm(command_xy) + abs(command_yaw) > 0.1`), the raw
per-environment cost is the normal-load-weighted mean over loaded feet:

```text
C = sum_i(loaded_i * Fn_i * phi_i) / sum_i(loaded_i * Fn_i)
```

`C=0` when no valid loaded foot exists.  The new reward rate is:

```text
r_stance_slip = -0.05 * C
```

The deadband protects normal stance micro-motion, local-tangent projection does
not misclassify velocity along the slope normal as slip, clipping prevents rare
contact spikes from dominating PPO, and load normalization prevents the scale
from changing merely because one or two feet are supporting the robot.

The **only experimental variable** is the new reward weight, preregistered at
`-0.05` for this first run.  The projection, thresholds, deadband, scale, clip,
sensor mapping, and command gate are frozen and must not be tuned during the run.
There is no multi-weight sweep in the first round.

## Code and runtime telemetry

Implementation points:

```text
src/tasks/velocity/mdp/rewards.py
  terrain_relative_loaded_stance_slip_cost
  terrain_relative_loaded_stance_slip

src/tasks/velocity/config/go2/env_cfgs.py
  unitree_go2_rough_v7_stance_slip_env_cfg

src/tasks/velocity/config/go2/__init__.py
  Unitree-Go2-Rough-V7-StanceSlip
```

The foot site configuration uses `preserve_order=True`, and contact sensor slots
are explicitly permuted from MuJoCo XML order into `FR/FL/RR/RL`.  The reward
logs:

```text
Metrics/terrain_tangent_stance_slip_mean
Metrics/terrain_tangent_loaded_fraction
Metrics/terrain_tangent_ray_valid_fraction
Metrics/terrain_tangent_normal_force_mean
Metrics/terrain_tangent_slip_cost_mean
Episode_Reward/terrain_tangent_stance_slip
```

No actor or critic observation is added.  Actor/critic shapes remain `234/261`.
The reward may use privileged contact force and ray normal during training, but
the deployed actor still receives the unchanged V7 observations.

## Frozen training contract

Relative to `Unitree-Go2-Rough-V7`, the new task adds exactly one reward term.
All of the following remain unchanged:

- terrain generator, terrain proportions, curriculum, geometry and initial levels;
- high-slope reset exposure; the rejected 10% sampler is not present;
- foot friction randomization and all other dynamics randomization;
- termination definitions and thresholds;
- command modes, probabilities, ranges and resampling;
- fixed gait phase reward and clearance/contact rewards;
- actor and critic observations, network dimensions and action definition;
- PPO optimizer, learning rate, epochs, mini-batches, clip, entropy, gamma and lambda;
- 2048 environments, seed 42, checkpoint resume state and 400 iterations.

The existing V7 `foot_slip` term is retained unchanged.  The new local-tangent,
loaded-stance term adds the causal shaping signal without silently rewriting the
baseline reward.

## Pre-training verification

CPU contract:

```text
tests/test_go2_stance_slip_reward.py: 9 PASS
stance-slip plus causal evaluators: 25 PASS
full unittest discovery: 397 PASS, 1 intentional skip
```

It covers flat math, slope-normal projection, loaded/contact/ray gating,
finite zero behavior, yaw invariance, clipping/parameter validation, exact
single-reward config diff, task registration, unchanged runner and zero-weight
V7 equivalence after removal of the inert term.

No-learning runtime tool:

```text
scripts/preflight_go2_stance_slip_training.py
```

32-env calibration artifact:

```text
stance_slip_training_preflight_seed42_32env_64steps_v2.json
SHA256 a7dca995a5acd8e24f8e15668c3831e97af305a92b1c62eff5a91888bd8cffd6
raw cost mean/p95/max: 0.1567 / 0.8462 / 4.0
weighted reward-rate mean: -0.00784
last-step tangent slip: 0.0330 m/s
```

Formal warm-start preflight artifact:

```text
stance_slip_training_preflight_seed42_2048env_8steps_fullresume_v3.json
SHA256 348377defa91a940d4683d9b2188642dc423e757b6c3fae36d12f0b495e9bc9d
strict full resume: PASS
restored iteration: 13600
restored terrain mean level: 5.275 (runtime log)
actor/critic/action/reward finite: PASS
learn_called: false
```

The first calibration attempt exposed a site/contact permutation mismatch.  It
was corrected with `preserve_order=True`; the invalid v1 artifact was deleted and
is not evidence.  The corrected v2/v3 artifacts are the only preflight results.

## Preregistered training matrix

There is one trained arm and one fixed existing reference:

| Arm | Checkpoint/start | New reward weight | PPO continuation |
| --- | --- | ---: | ---: |
| Reference | V7 `model_13600.pt` | absent / equivalent to 0 | none |
| Candidate | V7 `model_13600.pt` full resume | `-0.05` | 400 iterations |

Expected candidate checkpoints are `model_13700.pt`, `model_13800.pt`,
`model_13900.pt`, and final `model_13999.pt` under the new run.  Names matching
previous rejected checkpoints do not transfer any status across run directories;
every checkpoint is identified by full path and SHA256.

Fixed command:

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab

python scripts/train.py Unitree-Go2-Rough-V7-StanceSlip \
  --env.scene.num-envs=2048 \
  --env.seed=42 \
  --agent.seed=42 \
  --agent.resume=True \
  --agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter \
  --agent.load-checkpoint=model_13600.pt \
  --agent.max-iterations=400 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.run-name=go2_rough_v7_local_tangent_stance_slip_2048env_400iter
```

Stop and reject the invocation on NaN/Inf, OOM, unrecoverable simulator error,
checkpoint/resume mismatch, missing reward telemetry, or evidence that any frozen
configuration differs from the registered contract.  Do not repair a failed run
by changing another training factor.

## Candidate selection

Do not select a checkpoint from TensorBoard reward alone.  Evaluate each saved
stage with the same fixed clean/randomized high-slope matched screen.  Discard a
stage that violates any `1.2x` safety guardrail.  Among remaining stages, select
lexicographically by:

1. highest minimum clean high-slope completion across line/arc/S;
2. highest minimum randomized high-slope completion across line/arc/S;
3. highest minimum forward response gain;
4. lowest terrain-tangent loaded-stance slip;
5. earlier checkpoint if still tied.

Only the selected stage enters the full acceptance suite.  The final checkpoint
is not automatically preferred.

## Post-training acceptance gates

All candidate/reference comparisons must be same-scene and matched wherever the
evaluator supports it.  Required suites are:

- clean and randomized high-slope straight/arc/S, both `vx=0.3/0.5`;
- flat and ordinary rough/obstacle retained patches;
- continuous stairs and slopes, plus randomized level-9 stairs seeds 42/43/44;
- flat line/arc/S and terrain line/arc/S path regression.

The candidate is accepted only if all are true:

- clean high-slope completion is at least `12/16` per route kind and randomized
  completion at least `10/16` per route kind;
- completion improves by at least `+0.20` absolute versus V7 per high-slope route
  kind, and high-slope mean forward gain reaches `0.80`;
- retained-scene forward gain is no more than `0.05` below V7;
- flat, rough, obstacle, stairs and line/arc/S completion do not regress;
- slip, action acceleration, base pitch, base contact, upper-leg contact, calf
  contact and failure risk are each no greater than the matched V7 value times
  `1.2`;
- metrics, lifecycle, placement and output JSON are finite and provenance-complete.

Any failed gate rejects the candidate and keeps V7 `model_13600.pt` as default.
No default path or deployment configuration may be changed before full PASS.

## Current state

Training preparation is complete and the registered task is ready for the fixed
command above.  GPU and relevant process state must be checked once more at the
actual launch boundary.  This document does not itself authorize changing the
preregistered weight or adding a second intervention.
