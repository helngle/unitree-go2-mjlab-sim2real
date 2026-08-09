# Go2 V8 Privileged Linear-Velocity Teacher Probe

Date: 2026-08-04 (Asia/Shanghai)

## Status

```text
FORMAL_TRAINING_STARTED=false
DEFAULT_MODEL_REPLACED=false
STUDENT_TRAINING_ALLOWED=false
```

The objective is an accepted complex-terrain simulation teacher. Student
distillation, DAgger, student PPO, and Sim2Real changes remain out of scope
until this teacher arm passes all registered gates.

## Locked Source

```text
checkpoint: logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
sha256: 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
source actor / critic / action: 234 / 261 / 12
```

## Matched Arms

Both arms use seed 42, 2048 environments, 400 PPO updates, a fresh optimizer,
and the same transferred V7 actor, critic, normalizers, terrain assignment, and
common environment step. The only arm difference is the appended actor term.

```text
control task:   Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher-Control
control actor:  234-D V7 observation
candidate task: Unitree-Go2-Rough-V8-PrivilegedLinVelTeacher
candidate actor: 234-D V7 observation + base_lin_vel(3) = 237-D
critic:         261-D in both arms
action:         12-D in both arms
```

`base_lin_vel` occupies actor indices `[234:237]` in `vx, vy, vz` order. It is
the noise-free MuJoCo velocimeter output at the offset IMU site, in m/s and the
base-aligned IMU-site local frame. It is not world-frame velocity and is not
strictly base-COM velocity.

No foot contact/force/height/air-time, mass, COM, friction, motor-strength, or
`isSlope` observation may be added. Rewards, termination, terrain geometry,
sampling, curriculum, commands, gait phase, action interface, actuators, PPO,
student observations, and distillation remain unchanged.

## Transfer Contract

- Copy all original 234 actor input columns and all compatible actor tensors.
- Initialize the three new first-layer columns to exact zero.
- Copy the new normalizer mean/variance/std from the locked source critic at
  `[234:237]`; actor and critic normalizer counts must match.
- Copy the complete compatible 261-D critic.
- Do not restore optimizer or learning iteration.
- Restore the source `terrain_levels`, `terrain_types`, and
  `common_step_counter=326664`; formal 2048-env restoration must be exact.
- Reset Python, NumPy, CPU Torch, and CUDA RNG streams to seed 42 after model
  construction and transfer so the wider candidate build cannot shift draws.
- Before learning, deterministic control/candidate action parity must have
  maximum absolute error at most `1e-6` on matched nonzero observations.

The probe runner uses logical update labels 1 through 400, so `model_100.pt`,
`model_200.pt`, `model_300.pt`, and `model_400.pt` represent exactly 100, 200,
300, and 400 completed updates. Pre-learning transfer parity is a separate
preflight artifact and cannot participate in checkpoint selection.

## Preflight Gates

Formal training is forbidden until all of the following pass:

- schema/order/unit/frame and information-isolation tests;
- source path/SHA, tensor mapping, zero-column, normalizer, critic, and RNG tests;
- stable semantic equality of the control environment and V7;
- `git diff --check`;
- 32-env optimizer smoke marked non-candidate;
- 2048-env no-learning control/candidate preflight with exact environment-state
  restoration, finite observations/actions/rewards, no optimizer step, and
  matched parity;
- no duplicate run or related GPU process.

NaN/Inf, OOM, missing telemetry, provenance mismatch, simulator failure, or
non-exact formal environment restoration stops the experiment. Only a purely
technical retry with an identical contract may be considered.

## Screening And Selection

All four checkpoints from each arm first run finite/provenance/action-target
screening and clean/randomized high-slope line/arc/S evaluation. One checkpoint
per arm is selected by hard gates first, then:

1. highest minimum clean route completion;
2. highest minimum randomized route completion;
3. highest mean forward gain;
4. lowest terrain-tangent loaded-stance slip;
5. earliest checkpoint.

Weighted post-hoc scores are forbidden. The selected control, selected
candidate, and original V7 then run the same complete retained-scene matrix.

## Candidate ACCEPT Gates

Every gate is mandatory:

```text
clean high-slope line/arc/S completion:      each >= 12/16
randomized high-slope line/arc/S completion: each >= 10/16
high-slope mean forward gain:                >= 0.80
six-cell high-slope macro completion:        >= selected control + 0.10
each high-slope route/profile completion:    >= matched original V7
each retained scene/route/profile:           >= matched original V7 - 0.05
action and processed joint-target fault rate: <= 1.2 * matched original V7
```

For every matched group, failure risk, terrain-tangent loaded-stance slip,
action acceleration, absolute pitch, base contact, upper-leg contact, and calf
contact must be no greater than `1.2 * matched V7`. A zero V7 reference remains
an exact zero limit; no unregistered epsilon is allowed. All tensors,
TensorBoard values, and evaluation JSON values must be finite, and placement,
lifecycle, provenance, coverage, and matched-scene identity must pass.

The relative action/target-fault rule is required because the frozen legacy V7
action interface already has nonzero high-slope fault steps; an absolute-zero
gate would contradict the registered prohibition on changing that interface.

An ACCEPT promotes only the fully identified teacher checkpoint after complete
evaluation. A REJECT does not authorize a 241-D arm automatically: first audit
the new columns' weights, gradients, and counterfactual action sensitivity,
then preregister `foot_contact(4)` as a separate experiment if supported.

## Reference Boundary

The design follows the information split used by Kumar et al., *RMA: Rapid
Motor Adaptation for Legged Robots*, and its official implementation:
<https://github.com/antonilo/rl_locomotion>. RMA supports privileged velocity
and dynamics information and defaults to `includeGRF: false`; it does not prove
that this three-dimensional Go2 intervention must succeed.
