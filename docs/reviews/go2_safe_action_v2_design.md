# Go2 safe-action V2 design

Date: 2026-07-30

## Scope

V2 changes the action interface because actual V1 rollouts confirmed both
`abs(action) > 4` and processed joint-target limit violations. Observation,
reward, terrain, curriculum, commands, dynamics randomization, actuator gains,
network size, PPO hyperparameters, seeds and schedules remain the V1 baseline.

The interface identity is:

```text
bounded_asymmetric_per_joint_v2
```

V1 checkpoints are baselines only. They cannot be resumed into V2 because their
output and previous-action semantics differ.

## Joint bounds

For joint `i`, nominal position is `q0_i`, action scale is `s_i=0.25 rad`, and
the MJCF range is `[qmin_i,qmax_i]`:

```text
a_low_i  = max(-4, (qmin_i-q0_i)/s_i)
a_high_i = min( 4, (qmax_i-q0_i)/s_i)
```

| Joint | q0 | qmin | qmax | a_low | a_high |
|---|---:|---:|---:|---:|---:|
| FL_hip_joint | -0.10000 | -1.04720 | 1.04720 | -3.78880 | 4.00000 |
| FL_thigh_joint | 0.90000 | -1.57080 | 3.49070 | -4.00000 | 4.00000 |
| FL_calf_joint | -1.80000 | -2.72270 | -0.83776 | -3.69080 | 3.84896 |
| FR_hip_joint | 0.10000 | -1.04720 | 1.04720 | -4.00000 | 3.78880 |
| FR_thigh_joint | 0.90000 | -1.57080 | 3.49070 | -4.00000 | 4.00000 |
| FR_calf_joint | -1.80000 | -2.72270 | -0.83776 | -3.69080 | 3.84896 |
| RL_hip_joint | -0.10000 | -1.04720 | 1.04720 | -3.78880 | 4.00000 |
| RL_thigh_joint | 0.90000 | -0.52360 | 4.53790 | -4.00000 | 4.00000 |
| RL_calf_joint | -1.80000 | -2.72270 | -0.83776 | -3.69080 | 3.84896 |
| RR_hip_joint | 0.10000 | -1.04720 | 1.04720 | -4.00000 | 3.78880 |
| RR_thigh_joint | 0.90000 | -0.52360 | 4.53790 | -4.00000 | 4.00000 |
| RR_calf_joint | -1.80000 | -2.72270 | -0.83776 | -3.69080 | 3.84896 |

The canonical V2 schema, not this copied table, is the executable source of
truth. Startup must compare its SHA256 with checkpoint and ONNX metadata.

## Transform

Let the actor MLP output a finite `m_raw_i`. The Gaussian mean is
`m_i=5*tanh(m_raw_i/5)`, its sampled policy variable is `z_i`, and
`u_i=tanh(z_i)`. The applied
normalized residual is:

```text
T_i(z_i) = u_i*a_high_i       if u_i >= 0
T_i(z_i) = (-u_i)*a_low_i     if u_i < 0
q_target_i = q0_i + 0.25*T_i(z_i)
```

Thus `T(0)=0`, every applied residual is within `[-4,4]`, and every processed
target is inside its per-joint MJCF range. NaN/Inf is an explicit failure. The
observation at control step `t` contains the applied `T(z)` from `t-1`; reset
uses zeros and the first finite frame backfills history according to the frozen
425-D schema.

Python training, evaluation, exported ONNX and C++ deployment must implement
the same transform. A deployment-only clamp is forbidden.

## Teacher distillation

The frozen V7 teacher emits old-interface raw action `z_teacher`. V2 constructs
exactly one target and exactly one environment action:

```text
teacher_label = T(z_teacher)
teacher_applied_action = T(z_teacher)
```

The student loss compares the student's applied output with this transformed
label. The rollout returns the same transformed action to the environment and
feeds it back as previous action. Comparing a V2 latent value directly with the
old raw V7 action, or applying `T` twice, is invalid. Training telemetry records
raw teacher `z`, bounded `u`, applied action, target-limit margin and saturation
statistics.

The handoff contains only the complete V2 student actor. PPO starts with a fresh
critic, optimizer and iteration zero; no teacher group or imitation loss remains.

## PPO change of variable

For raw density `p_Z`, V2 treats the executed variable as `a=T(z)`. Away from
the measure-zero branch at zero:

```text
log p_A(a) = log p_Z(z) - log(scale_i) - log(1-tanh(z_i)^2)
scale_i = a_high_i when z_i >= 0, else abs(a_low_i)
```

The implementation must use stable `log(1-tanh(z)^2)` evaluation and sum the
per-joint Jacobian terms. Sampling, stored PPO actions, recomputed log
probabilities, deterministic inference and entropy reporting must use one
declared variable convention. It is not valid to execute `T(z)` while treating
that value as an unsquashed Gaussian sample. A latent-space PPO ratio is
mathematically equivalent only when both old and new policies evaluate the same
stored `z` and the fixed Jacobian cancels; any such implementation must still
export `T(z)`, feed back `T(z)`, and test applied-space log-probability parity.

The finite mean parameterization is part of the V2 probability semantics. It
does not change the registered applied-action bounds or the residual joint
target mapping. It prevents the Gaussian mean from entering the float32 tanh
saturation regime where applied-space inversion loses the sampled latent and
PPO likelihood ratios can diverge. The bound `5.0` is frozen before the
replacement formal run; it leaves deterministic `|u|` capacity above 0.9999.

## Fixed baseline

V2 retains V1 actor/critic dimensions `425/261`, ten-frame 50 Hz proprioceptive
history, V7 rewards, terrain and commands, seven randomized dynamics events,
2048 environments, env/agent seed 42, 24 steps per environment, networks
`512/256/128` with ELU and observation normalization, 300 distillation
iterations and 4000 fresh PPO iterations. The V7 teacher is fixed by full path
and SHA256 in the training contract.

## Required implementation tests

- exact 12-joint name/order and bound derivation;
- `z=0`, large positive/negative values and at least 100000 random finite inputs;
- finite applied output, `abs(action)<=4`, and processed target in MJCF range;
- transform continuity, monotonicity and stable Jacobian near saturation;
- teacher label equals teacher applied action and transform is applied once;
- previous-action history contains the applied action and resets to zero;
- stochastic/deterministic PPO log-probability consistency;
- 2048-environment shape, dtype and device behavior;
- eager/ONNX/C++ elementwise parity;
- checkpoint schema/action-interface/source-manifest provenance.
