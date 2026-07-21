# Final high-slope attribution and training decision

Date: 2026-07-21
Scope: simulation-only Go2 locomotion; no vision, sim-to-real, reward redesign, or
long training in this attribution round.

## Final decision

The clean and randomized controller-headroom A/B both identify
`sustained_slope_locomotion_limited` with HIGH attribution confidence.  At
controller scale 1.5, clean straight/arc/S completion remains `0/4, 0/4, 0/4`,
with mean `vx` gain `0.607/0.546/0.464` and maximum saturation
`0.0101/0/0`.  Randomized completion also remains `0/4, 0/4, 0/4`, with gain
`0.488/0.514/0.508` and maximum saturation `0.0076/0/0`.  Removing lateral/yaw
controller saturation therefore does not recover any route; physical contact,
fall, and low-progress failures remain.

The training design gate is `TRAINING-READY`.  The independent task and sampler
described below are implemented and CPU-tested, but the 2048-env, 400-iteration
probe has **not** been started in this round.  V7 `model_13600.pt` remains the
default model until a trained checkpoint passes the unchanged acceptance matrix.

## Baseline terrain distribution (from the actual V7 configuration)

V7 is built by `unitree_go2_rough_v6_env_cfg()` and then changes only the command
term in `unitree_go2_rough_v7_env_cfg()`.  In training mode the terrain generator
uses `curriculum=True`, `num_rows=10` (difficulty rows 0--9), `num_cols=20`, and
these proportions:

| terrain column family | configured proportion | normalized columns |
| --- | ---: | ---: |
| flat | 0.15 | 3/20 |
| pyramid stairs up | 0.15 | 3/20 |
| pyramid stairs down | 0.15 | 3/20 |
| pyramid slope down (`hf_pyramid_slope`) | 0.10 | 2/20 |
| pyramid slope up (`hf_pyramid_slope_inv`) | 0.10 | 2/20 |
| random rough | 0.15 | 3/20 |
| discrete obstacles | 0.20 | 4/20 |

The proportions sum to `1.0`; the two slope families together occupy 20% of
terrain columns.  Curriculum mode assigns a terrain family to a column and
increases difficulty along rows; the proportions do **not** directly sample a
particular difficulty row at reset.

For the hard-case definition used here:

```text
H = { slope_up, level 8 or 9 }
    union { slope_down, level 9 }
```

If terrain rows were uniformly represented, the nominal V7 mass of `H` would be

```text
q_H = 0.10 * (2/10) + 0.10 * (1/10) = 0.03  (3.0% of all slots).
```

The saved V7 environment state provides an empirical checkpoint snapshot:
`64/2048 = 3.125%` of environments are in `H`; all four slope columns account for
`410/2048 = 20.02%`, and all slope level-8/9 environments account for
`91/2048 = 4.44%`.  The difference between 3.0% and 3.125% is expected because
the checkpoint stores the current curriculum state rather than a stationary
uniform row distribution.  Training logs must report the measured reset fraction,
not infer it from a single checkpoint.

The original V7 task has no field that targets `H` specifically.  Changing a
`SubTerrainCfg.proportion` changes every difficulty row of that terrain family;
it cannot, by itself, implement “slope-up high/extreme plus slope-down extreme”.
The independent probe therefore uses a level-aware reset event while keeping the
generated geometry unchanged.

## Required controller-headroom A/B

The A/B must use the same V7 checkpoint, seed, terrain assignment, matched slot,
route, horizon, and fresh-environment policy rollout.  The only changed values are
the closed-loop controller limits:

```text
scale 1.0: max_lateral_speed, max_yaw_rate unchanged
scale 1.5: both limits multiplied by 1.5
```

Cross-track/heading gains, tolerances, route geometry, terrain randomization,
policy inputs, and all reward/training configuration remain frozen.  The evaluator
must record commanded and actual `vx/vy/wz`, response gains, saturation fraction,
completion/progress, cross-track and heading errors, reset/first-failure reason,
contacts, slip, and action acceleration.  The `r=4.0` straight case remains a
pre-GPU geometry rejection because its scan footprint exceeds the 18 m patch by
about `0.17758 m`; it is never a policy failure.

## Final attribution rules

### A. `controller_limited`

Use this result only when all of the following are observed in the affected
matched slots:

1. Scale 1.0 has material controller saturation (the declared gate is 10% of
   control steps or more) in the failures.
2. Scale 1.5 materially reduces saturation (preferably below 10%) and restores
   arc/S completion or progress, while straight completion does not materially
   worsen.
3. The policy's forward response under the same command tape does not show a new
   locomotion collapse; the improvement is explained by command headroom rather
   than a changed checkpoint.

Decision: `NO-GO` for PPO.  Fix the controller limit/command generation with the
smallest controller-only change, rerun the identical A/B matrix, and do not alter
reward, terrain sampling, termination, gait, or network during that fix.

### B. `sustained_slope_locomotion_limited`

Use this result only when the following joint evidence is present:

1. Scale 1.0 **and** scale 1.5 remain poor across straight, arc, and S on the
   same high/extreme slope slots (completion at or below the declared 0.50 failure
   gate and mean `vx` gain below 0.80, with route-to-route gain spread no larger
   than 0.15).
2. Saturation is not a common explanation after scale 1.5: either it is below the
   10% gate, or high-saturation slots fail at the same rate as non-saturated slots
   and the extra headroom does not recover them.
3. Failures are concentrated in slope-up high/extreme and/or slope-down extreme,
   with physical contact/fall or sustained under-speed evidence; geometry,
   placement, finite-value, and horizon contracts all pass.

Decision: `TRAINING-READY` for exactly one next probe, subject to the Acceptance
Agent's final PASS and a clean integration worktree.  This is an authorization to
prepare/start the specified probe in the **next** round, not permission to train
while this A/B round is incomplete.

### C. `inconclusive_no_training`

Use this result for every other outcome: saturation remains ambiguous, only one
route kind fails, completion lies in the gray zone, A/B slots are not matched,
geometry or horizon is invalid, or the scale change alters anything besides the
two declared limits.

Decision: `NO-GO`.  Record the smallest additional evaluation-only diagnostic;
do not change reward, terrain proportions, command sampling, termination, gait, or
network merely to force a training decision.

## Implemented training probe for result B

The new task `Unitree-Go2-Rough-V7-HighSlopeProbe` derives from V7 without
mutating the registered V7 config or checkpoint.  Its explicit configuration is:

```python
cfg.events["high_slope_sampling"].params = {
    "target_hard_case_ratio": 0.10,
    "slope_up_levels": (8, 9),
    "slope_down_levels": (9,),
    "seed_offset": 700,
}
```

The event runs after the original `terrain_levels` curriculum computation and
before `reset_base`.  It preserves every post-curriculum candidate whose hard/non-
hard membership already matches the target quota and changes only the minimum
number of mismatches.  Donor slots use the current group's conditional distribution,
with the nominal V7 distribution only as a zero-mass fallback.  Telemetry records
candidate, changed-slot, batch, cumulative-reset, and population hard-case ratios.
Terrain meshes, randomization, height scan, rewards, terminations, gait phase,
command ranges, observations, actions, and network shape remain unchanged.

Let `q(s)` be the measured V7 baseline probability of a reset slot and `H` the set
above.  Let `q_H = sum(q(s) for s in H)`.  For the recommended probe target
`p_H = 0.10`, reweight slots as:

```text
P_probe(s) = q(s) * (p_H/q_H)                  for s in H
P_probe(s) = q(s) * ((1-p_H)/(1-q_H))          for s not in H
```

This preserves the relative mixture within `H` and outside `H`, changes only the
aggregate hard-case ratio, and yields a measured target near 10% (with a logged
confidence interval).  At the nominal V7 distribution this is a 3.0% -> 10.0%
change, about 3.3x more hard-case exposure; the exact baseline denominator must be
reported from the same-seed 2048-env reset audit.  No second sampling knob may be
changed in the probe.

### Fixed warm start and command (next round only)

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab

python scripts/train.py Unitree-Go2-Rough-V7-HighSlopeProbe \
  --env.scene.num-envs=2048 \
  --env.seed=42 \
  --agent.seed=42 \
  --agent.resume=True \
  --agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter \
  --agent.load-checkpoint=model_13600.pt \
  --agent.max-iterations=400 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.run-name=go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

The only experimental variable is `target_hard_case_ratio=0.10`; use the V7
checkpoint's optimizer and terrain state as the warm start.  With a 400-iteration
continuation from iteration 13600, the expected periodic checkpoints are
`model_13700.pt`, `model_13800.pt`, and `model_13900.pt`, with the final checkpoint
`model_13999.pt` under the new run directory.  CPU contracts measure
`204/2048 = 9.9609%` on the first full reset, verify that only the required
membership mismatches change, preserve H/non-H conditional probabilities, and
verify terrain-origin relocation before the base reset.

## Post-training acceptance

Run the exact pre-training matrices without changing route, seed, horizon, or
metrics (the same acceptance protocol):

- clean and randomized high-slope matched straight/arc/S (`r=2.5`, 2400 steps);
- randomized level-9 stairs, up/down, seeds 42/43/44;
- V7 flat/rough/obstacle and continuous straight regression suites.

Accept a new checkpoint only if all are true:

- high-slope clean and randomized completion improve materially over V7 (recommended
  minimum: +0.20 absolute per route kind and no route below 0.70 clean / 0.60
  randomized);
- mean forward response reaches the 0.80 gate and is not more than 0.05 below V7
  on any retained baseline suite;
- progress, cross-track/heading error, slip, action acceleration, and contacts are
  finite and within the predeclared same-scene limits; slip/action acceleration do
  not exceed V7 by 1.2x;
- calf/base/upper-leg/fall terminations do not materially increase;
- stairs and all flat/rough/obstacle regression gates do not regress.

If any gate fails, reject the probe, keep V7 `model_13600.pt` as the default model,
and do not append another training variable.  Every result must be recorded in
`docs/PROJECT_JOURNAL.md` and summarized in `docs/HANDOFF.md` with the exact
checkpoint, measured hard-case ratio, JSON paths, metrics, and PASS/FAIL decision.

## Current authorization

The final A/B satisfies result B and the training design status is
**TRAINING-READY** for the single 10% hard-case sampling probe.  This status does
not accept a new model in advance: no training was started in this round, V7
`model_13600.pt` remains default, and any next checkpoint must pass the complete
post-training acceptance matrix above.
