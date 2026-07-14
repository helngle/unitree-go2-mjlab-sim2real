# Lateral-conditioned hip pose tolerance: independent analysis

Baseline reviewed: `e8a7eeef647861d674410ef9f5ad5f7e68f3d337` (`e8a7eee`).
This review is read-only with respect to implementation and training code.

## Conclusion

The proposed probe is technically sound and is a defensible single-variable
experiment if it is enabled only by a new task derived directly from V7. The
reviewed formula is:

```text
competitor = max(abs(vx), yaw_scale * abs(wz))
alpha = clamp((abs(vy) - competitor) / lateral_scale, 0, 1)
hip_std = baseline_hip_std + alpha * (target_hip_std - baseline_hip_std)
```

with `yaw_scale=1.0`, `lateral_scale=0.30`, and `target_hip_std=0.30 rad`.
For V7's sampled command modes, it changes only pure-lateral commands. It does
not require an actor, critic, observation, or action shape change, so V7
`model_13600.pt` is warm-start compatible.

## Baseline reward math and tensor contract

- `variable_posture` obtains the command from `get_command("twist")`. The
  command term exposes `vel_command_b`, a `[num_envs, 3]` tensor ordered as
  `[vx, vy, wz]`. Linear entries are in m/s and yaw is in rad/s.
- The current regime scalar is
  `norm(command[:, :2], dim=1) + abs(command[:, 2])`, shape `[B]`.
  `walking_threshold=0.1` and `running_threshold=1.5` are supplied by the task
  config. The function signature's default of `0.5` is not active for Go2.
- The three masks have shape `[B]`; `unsqueeze(1)` combines them with a std
  vector of shape `[J]`, producing `[B, J]` by broadcasting.
- Joint error is `[B, J]`, and the reward is exactly
  `exp(-mean((q - q_default)^2 / std^2, dim=1))`, producing `[B]`.
- Go2's standing hip std is `0.05 rad`. Its walking and running hip std are
  both `0.15 rad`. Walking/running thigh and calf std values are `0.35` and
  `0.50 rad`; these must remain unchanged by this probe.

### Joint ordering

The compiled Go2 articulation order is:

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

`asset.find_joints(".*")`, `SceneEntityCfg.resolve`, and
`resolve_matching_names_values` all use target/model order when
`preserve_order=False` (the current setting). Consequently, the three std
vectors and `joint_pos[:, asset_cfg.joint_ids]` are aligned. A new hip mask
must be built in this same resolved `joint_names` order. It must select exactly
indices `[0, 3, 6, 9]` for the current model.

The implementation should build an environment-specific std tensor without
in-place mutation of the persistent `[J]` regime vectors. A safe broadcast
contract is `alpha[:, None]` (`[B,1]`) with a hip mask `[J]`, yielding `[B,J]`.
The result must retain the command/std device and floating dtype.

## V7 command distribution

The registered `Unitree-Go2-Rough-V7` config instantiates successfully and has:

```text
general / lateral / yaw / high-speed = 0.40 / 0.25 / 0.15 / 0.20
pure-lateral magnitude range = 0.10..0.30 m/s
lateral_speed_stages = ()
standing probability = 0.02
command curriculum entry = absent
```

The lateral sampler sets `vx=0`, `wz=0`, and samples signed `vy`, so it is
genuinely pure lateral. The standard command update subsequently zeros the 2%
standing environments; standing therefore remains a zero command even though
a locomotion mode ID was sampled first.

The 40/25/15/20 figures are base probabilities, not universal per-terrain
probabilities. On configured focus terrain columns at level 7 or above,
high-speed probability becomes 0.45 and the other modes are scaled by 0.6875:

```text
general / lateral / yaw / high-speed =
0.275 / 0.171875 / 0.103125 / 0.45
```

This is existing V7 behavior and must remain frozen. Acceptance comparisons
should not assert an unconditional 25% lateral fraction across a terrain-mixed
batch; instead assert the config fields and test ordinary versus focus
probability rows separately.

V7.1 is not the correct parent for the probe: it overrides the proportions to
30/45/10/15 and adds a staged lateral range. A new config must call V7 directly.

## Formula assessment

At the walking/running hip baseline of `0.15 rad`, the reviewed formula gives:

| Command `[vx, vy, wz]` | alpha | Hip std |
| --- | ---: | ---: |
| `[0, 0, 0]` | 0 | existing standing `0.05 rad` |
| `[0.6, 0, 0]` | 0 | `0.15 rad` |
| `[0, 0.1, 0]` | 1/3 | `0.20 rad` |
| `[0, 0.2, 0]` | 2/3 | `0.25 rad` |
| `[0, 0.3, 0]` | 1 | `0.30 rad` |
| `[0, 0, 0.7]` | 0 | `0.15 rad` |
| `[0.1, 0.2, 0]` | 1/3 | `0.20 rad` |
| `[0.2, 0.2, 0]` | 0 | `0.15 rad` |
| `[0, 0.3, 0.3]` | 0 | `0.15 rad` |

The alpha function is continuous, bounded, sign-symmetric, and has no
data-dependent denominator. Its derivative has corners at `abs`, `max`, and
`clamp` boundaries, which is irrelevant because commands do not require
gradients in reward evaluation. `lateral_scale` must nevertheless be validated
as strictly positive to prevent division by zero from a bad config.

The comparison between m/s and rad/s is dimensionally meaningful only through
an explicit conversion coefficient. `yaw_scale=1.0` should be documented as
having effective units of metres per radian, even if represented as a scalar.
Keeping it configurable also makes the intended semantics testable.

The interpolation is based on lateral *dominance margin*, not lateral speed
alone. Equality with the strongest competing axis gives no relaxation. This is
consistent with the stated lateral-dominant requirement and avoids affecting
mixed/yaw commands merely because `vy` is large.

Within V7's actual mode ranges this is cleanly isolated:

- General has `vx >= 0.15` and `abs(vy) <= 0.10`, so alpha is always zero.
- Pure lateral has `vx=wz=0`, so alpha spans `1/3..1`.
- Pure yaw has `vy=0`, so alpha is zero.
- High-speed has `vx >= 0.8` and `abs(vy) <= 0.05`, so alpha is zero.
- Standing is the zero command, so alpha is zero and its original `0.05 rad`
  hip std remains intact.

The phrase "0.15 to 0.30" is anchored at zero lateral command. Since V7 starts
lateral sampling at `0.10 m/s`, sampled lateral tolerance starts at `0.20 rad`,
not `0.15 rad`. This matches the supplied formula and must be explicit in the
experiment record.

## Warm-start and registration review

`model_13600.pt` exists and contains actor, critic, optimizer, iteration, and
environment state. It records iteration 13600, common step 326664, and terrain
state for 2048 environments. With a 2048-env probe, environment state restoration
can restore terrain levels/types directly.

Changing reward calculation and registering a new task does not change actor or
critic inputs, action size, model layers, or optimizer parameter shapes. Strict
checkpoint loading is therefore compatible provided the probe inherits the V7
environment and the existing Go2 PPO runner config unchanged.

Registration risks to check:

1. Export/import the new factory in `src/tasks/velocity/config/go2/__init__.py`
   and register unique train and play configs under a unique task ID.
2. Derive from `unitree_go2_rough_v7_env_cfg`, not V7.1.
3. Apply new reward parameters only in that derived config. Defaults in shared
   `variable_posture` must reproduce the old reward exactly for every existing
   task.
4. Keep V7 command cfg, terrain, curriculum, events, terminations, reward weights,
   gait phase, observations, actions, and PPO config unchanged.
5. Do not add the conditioning signal to observations: the command is already
   present, and changing observation shape would break strict warm start.

## Prioritized risks

### High

1. **Shared reward regression.** Adding mandatory parameters or enabling the
   behavior by default in `variable_posture` would alter every robot/task using
   the shared term. Require an explicit disabled default and an old-versus-new
   equivalence test with conditioning disabled.
2. **Wrong experiment parent.** Deriving from V7.1 silently retains 45% lateral
   sampling and staged speed, invalidating the single-variable claim.
3. **Hip/std mis-broadcast or persistent mutation.** A `[B]` alpha multiplied
   directly with `[J]` fails unless `B==J`, and in-place edits to `self.std_*`
   can leak one batch's commands into later steps. Require `[B,J]` construction
   and exact per-joint assertions.

### Medium

4. **Incorrect standing baseline.** Standing hip tolerance is `0.05`, not
   `0.15`. Zero command must remain exactly on the standing vector; only active
   lateral commands should interpolate from their selected regime baseline.
5. **Misreported sampler mix.** Focus terrains intentionally change the base
   40/25/15/20 distribution. Comparing aggregate sampled counts to 25% without
   accounting for focus rows can produce a false failure.
6. **Implicit units.** Treating yaw rate numerically as linear speed without
   naming/documenting `yaw_scale` obscures the experiment definition.

### Low

7. **Threshold interpretation.** At exactly `0.10 m/s`, the existing regime
   comparison selects walking (`>= walking_threshold`), giving `0.20 rad` after
   interpolation. Boundary tests must use the same inclusive comparison.
8. **Non-smooth derivative.** Alpha is continuous but not differentiable at its
   piecewise boundaries. This is not a training blocker because reward gradients
   do not flow through sampled commands.

## Acceptance recommendations

Before training, require all of the following:

1. Config/import check shows the new task and confirms its command config equals
   V7 field-for-field, including 40/25/15/20, `(0.1, 0.3)`, no stages, and no
   command curriculum.
2. Boundary test verifies pure lateral `0.1/0.2/0.3` maps hip std to
   `0.20/0.25/0.30 rad`, with both signs producing identical results.
3. Command batch test covers zero, pure forward, pure yaw, general, high-speed,
   lateral-dominant mixed, equality boundary, and yaw-dominant mixed commands.
4. Joint test verifies only four hip columns change and all thigh/calf columns
   exactly match the selected baseline std.
5. Shape test uses `B != J` (for example `B=5`, `J=12`) and checks finite `[B]`
   rewards, finite `[B,J]` std, correct device/dtype, and no mutation between calls.
6. Validation test rejects zero/negative `lateral_scale`, malformed command last
   dimension, negative `yaw_scale`, and a target hip std below the active baseline
   if that configuration is unsupported.
7. Backward-compatibility test compares the old formula with the conditioned
   implementation disabled over randomized commands/joints and requires exact or
   tight floating-point equality.
8. Strict-load smoke check loads `model_13600.pt` into the new task at 2048 envs,
   verifies restored common step/terrain state, runs a few environment steps, and
   confirms finite observations, actions, rewards, and losses before any long run.
9. Diff audit confirms the only intentional production behavior change is the
   new task's command-conditioned hip tolerance. Existing V7 and V7.1 instantiated
   configs must remain unchanged.

Only after these checks pass should the 300--500 iteration, 2048-env probe begin.
Post-training acceptance remains comparison against V7 `model_13600.pt` with the
corrected evaluator and lateral gait diagnostic, including lateral/forward gain,
foot lateral excursion, terrain curriculum, contacts/falls, action acceleration,
and foot slip.
