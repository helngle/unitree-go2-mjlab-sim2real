# High-slope probe training telemetry contract

Date: 2026-07-21

Scope: read-only preflight and training telemetry for
`Unitree-Go2-Rough-V7-HighSlopeProbe`.  This report does not authorize a second
probe, a configuration change, or concurrent GPU work.

## Preflight verdict

**PASS / ready for the Integration Agent's single fixed training run.**  The
probe is a direct V7 derivative.  After removing the new reset event and seven
telemetry terms, its rewards, commands, terminations, observations, actions,
existing events, existing metrics, curriculum, and terrain configuration are
equal to V7.  The registered V7 and probe tasks use the same RL configuration
and `VelocityOnPolicyRunner` class.

The sole behavior-changing field is:

```text
events.high_slope_sampling.params.target_hard_case_ratio = 0.10
H = slope_up levels 8/9 + slope_down level 9
```

The other new event parameters define that fixed set and a deterministic RNG
stream (`slope_up_levels=(8,9)`, `slope_down_levels=(9,)`, `seed_offset=700`);
they are implementation constants for the one sampling variable, not additional
experimental knobs.  The event is ordered after curriculum computation and
before `reset_base`.  It changes the minimum number of hard/non-hard membership
mismatches and does not replace `terrain_levels_vel`.

The V7 source checkpoint was inspected read-only:

```text
path: /home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/
      2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/
      model_13600.pt
size: 6.8 MiB
sha256: 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
iter: 13600
common_step_counter: 326664
terrain_levels: shape=(2048,), range=0..9
terrain_types: shape=(2048,), range=0..19
hard population snapshot: 64/2048 = 3.125%
```

It contains actor, critic, optimizer, iteration, and V7 terrain state.  As
expected, it has no `high_slope_sampling` state because it predates the probe.
`runner.load()` defaults to `strict=True`; the probe preserves actor/critic
input, output, and network configuration.  After strict algorithm load, the
runner restores V7 terrain state and calls `sampler.rebase()`, so preload-reset
statistics are cleared.  This path previously passed the independent real
2048-environment strict-load audit; the CPU persistence contracts pass again in
this preflight.

## Fixed command review

The declared CLI names are present in `scripts/train.py --help`.  The command is
valid as written:

```bash
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

`max_iterations` means **400 additional iterations** after resume: RSL-RL uses
`total_it = loaded_iteration + num_learning_iterations`.  The loop labels are
`13600..13999`; periodic saves occur at `13600`, `13700`, `13800`, and `13900`,
and the final save is `13999`.  The new run therefore also contains a
`model_13600.pt` after its first update.  It must never be confused with the V7
source `model_13600.pt`; all reports must use full paths.  The planned selection
set remains `13700/13800/13900/13999`.

## TensorBoard telemetry

Use the timestamped directory ending in
`_go2_rough_v7_high_slope_sampling_probe_2048env_400iter` supplied by the
Integration Agent as `RUN_DIR`.  Read-only UI:

```bash
tensorboard --logdir "$RUN_DIR" --port 6006
```

Required existing tags:

```text
Train/mean_reward
Train/mean_episode_length
Curriculum/terrain_levels
Episode_Reward/track_linear_velocity
Episode_Reward/track_angular_velocity
Episode_Reward/pose
Episode_Reward/foot_slip
Episode_Metrics/mean_action_acc
Episode_Termination/fell_over
Episode_Termination/illegal_base_contact
Episode_Termination/illegal_upper_leg_contact
Episode_Termination/illegal_calf_contact
Metrics/slip_velocity_mean
Metrics/base_contact_contact_rate
Metrics/upper_leg_contact_contact_rate
Metrics/calf_contact_contact_rate
Loss/value
Loss/surrogate
Loss/entropy
Loss/learning_rate
Policy/mean_std
Perf/total_fps
```

Required new tags:

```text
Episode_Metrics/candidate_hard_ratio
Episode_Metrics/changed_slot_ratio
Episode_Metrics/hard_case_batch_ratio
Episode_Metrics/hard_case_reset_ratio
Episode_Metrics/hard_case_population_ratio
Episode_Metrics/total_reset_count
Episode_Metrics/total_hard_count
```

Important semantic limitation: `MetricsManager` accumulates a metric at every
control step and emits the per-episode step average on reset.  Consequently the
seven TensorBoard values are useful trend signals, but `total_reset_count` and
`total_hard_count` there are **not exact latest integer counters**.  Likewise,
the plotted reset ratio is an episode-time average of cumulative snapshots.
Formal sampling audits must read checkpoint state, not reconstruct counts from
TensorBoard.

A non-GPU tag/finite summary after an event file exists:

```bash
RUN_DIR=/absolute/path/to/run
/home/jensen/anaconda3/envs/unitree_rl_mjlab/bin/python - "$RUN_DIR" <<'PY'
import math, sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1]
ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
for tag in sorted(ea.Tags().get("scalars", [])):
  events = ea.Scalars(tag)
  values = [float(e.value) for e in events]
  status = "finite" if all(math.isfinite(v) for v in values) else "NONFINITE"
  tail = values[-20:]
  mean20 = sum(tail) / len(tail) if tail else None
  print(f"{tag}\tcount={len(values)}\tlast={values[-1] if values else None}\tmean20={mean20}\t{status}")
PY
```

## Exact checkpoint audit

Only inspect a checkpoint after its size and modification time have stopped
changing.  Probe checkpoints must contain
`infos.env_state.high_slope_sampling` with:

```text
schema_version
target_hard_case_ratio
quota_residual
total_reset_count
total_hard_count
sampled_slot_histogram
generator_state
candidate_hard_ratio
changed_slot_ratio
hard_case_batch_ratio
hard_case_reset_ratio
hard_case_population_ratio
```

For each `model_13700.pt`, `model_13800.pt`, `model_13900.pt`, and
`model_13999.pt`, record the iteration, exact counters, exact cumulative ratio,
latest candidate/batch/changed/population ratios, histogram total, hard
histogram total, RNG-state length, quota residual, and optimizer presence.
The invariant checks are:

```text
target_hard_case_ratio == 0.10
0 <= quota_residual < 1
sum(sampled_slot_histogram) == total_reset_count
sum(sampled_slot_histogram[H]) == total_hard_count
hard_case_reset_ratio == total_hard_count / total_reset_count
abs(hard_case_reset_ratio - 0.10) <= 1 / total_reset_count
changed_slot_ratio == abs(hard_case_batch_ratio - candidate_hard_ratio)
all saved scalar values finite
generator_state is present and nonempty
optimizer_state_dict is present
```

The last equality is the sampler's minimum membership-change contract.  Small
floating serialization tolerance (`1e-6`) is allowed.  Population ratio is
recorded but is not required to equal 10% at every instant because resets are
asynchronous and curriculum continues to move slots.

Resume persistence is accepted only if loading a probe checkpoint restores the
same counter, histogram, quota residual, and generator state before the next
real reset.  The next matched reset must then produce the same sampled slots as
an uninterrupted sampler stream.  CPU tests cover this exact round trip; a
post-save smoke must not silently rebase an existing probe state.

## Stop and warning policy

Immediate stop and preserve the last complete checkpoint if any of these occur:

1. traceback, CUDA OOM, process nonzero exit, or simulator fatal error;
2. NaN/Inf in observations/rewards (the runner's built-in guard), loss, reward,
   terrain, sampler, or checkpoint scalar state;
3. after at least 2048 audited resets, exact cumulative hard ratio differs from
   0.10 by more than 0.005, or any counter/histogram/quota/RNG invariant fails;
4. a scheduled checkpoint is unreadable, lacks optimizer or sampler state, or
   cannot strict-resume with the same task;
5. the event file and checkpoint iteration stop advancing for 15 minutes while
   the training process is still expected to be active (Integration Agent must
   then check process/GPU state; this telemetry Agent does not touch the GPU).

Warnings requiring investigation, but not an automatic configuration change:

- rolling-20 reward or episode length drops by more than 20% from the first
  stable rolling-20 window;
- terrain level drops by more than 1.0 and does not recover for 50 iterations;
- termination/contact tags rise materially, action acceleration or slip exceeds
  the V7 training trend, or forward tracking falls;
- latest changed ratio violates the minimum-change identity, even if cumulative
  reset ratio remains close to target;
- throughput collapses without an explicit error.

Warnings do not authorize retuning, restart with a different seed, extra
iterations, or a second probe.  Performance is decided later by the fixed
evaluation gates, not by training reward alone.

## CPU verification performed

```text
CUDA_VISIBLE_DEVICES='' python -m unittest
  tests.test_go2_high_slope_training_probe
  tests.test_go2_final_slope_acceptance
Result: 29/29 PASS

python scripts/train.py Unitree-Go2-Rough-V7-HighSlopeProbe --help
Result: all fixed command flags present

checkpoint read-only schema/SHA audit
Result: PASS; V7 has the expected 2048 terrain state and no preexisting sampler state
```

No training, evaluator, simulator environment, or GPU process was started by
this Agent.  It does not poll or claim GPU ownership.

## Live run audit: stages 13700 and 13800

The Integration Agent supplied the stable read-only run directory:

```text
/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/
2026-07-21_16-21-46_go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

No simulator, runner, evaluator, or GPU context was created for this audit.
TensorBoard event data and fully written checkpoint files were read on CPU.

### Checkpoint state

| checkpoint | SHA256 | iter | common step | reset / hard | exact ratio | quota residual | population H |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_13700.pt` | `43a3823b3a26c9a7491332b931128653d410bd85d085e86c146dd8e3122c5d2f` | 13700 | 329088 | 5011 / 501 | 0.09998004 | 0.10000000 | 0.09765625 |
| `model_13800.pt` | `6312ca929edb9d33953a4edcf4da66733711c825ddf147489e5113b089f34101` | 13800 | 331488 | 10040 / 1004 | 0.10000000 | approximately `4.6e-14` | 0.09814453 |

Both files are 7,080,459 bytes and contain actor, critic, optimizer, iteration,
V7 terrain state, and the complete sampler state.  For both checkpoints:

```text
schema_version == 1
target_hard_case_ratio == 0.10
histogram sum == total_reset_count
hard histogram sum == total_hard_count
saved ratio == hard count / reset count
target error <= 1 / total_reset_count
0 <= quota_residual < 1
generator state exists, dtype uint8, length 16
all saved sampler scalars finite
changed ratio == abs(batch ratio - candidate ratio)
terrain level/type arrays remain shape (2048,), ranges 0..9 / 0..19
```

The last reset represented in a checkpoint may be a very small partial-reset
batch.  Thus `model_13700` legitimately records candidate/batch/changed ratios
`0.0/0.5/0.5`, while `model_13800` records
`0.3333/0.3333/0.0`.  These latest-batch values do not replace the exact
cumulative counts above.

The common-step offsets also confirm the RSL-RL iteration-label convention:
relative to V7 step 326664, `model_13700` is `+2424 = 101 * 24` control steps
and `model_13800` is `+4824 = 201 * 24`.  This is consistent with the new-run
`model_13600` being the first update and the final `model_13999` representing
all 400 requested updates.

### TensorBoard snapshot through iteration 13817

All 63 scalar tags are finite.  All 25 required core/sampler tags are present;
none are missing.  The latest 20-iteration means are:

| metric | probe mean20 | V7 pre-resume mean20 at 13600 | observation |
| --- | ---: | ---: | --- |
| mean reward | 49.293 | 51.901 | lower, but not a stop condition |
| mean episode length | 978.25 | 990.30 | small decrease |
| terrain level | 5.605 | 5.225 | increased |
| linear tracking reward | 0.8136 | 0.8407 | about 3.2% lower |
| angular tracking reward | 0.8999 | 0.9124 | about 1.4% lower |
| pose reward | 0.8439 | 0.8600 | about 1.9% lower |
| slip velocity | 0.08185 | 0.07649 | about 7.0% higher |
| action acceleration | 0.7666 | 0.7211 | about 6.3% higher |
| fell termination | 0.01875 | 0.00417 | warning: increased |
| base termination | 0.01875 | 0.01250 | warning: increased |
| upper-leg termination | 0.01875 | 0.01458 | warning: increased |
| calf termination | 0.06042 | 0.02708 | warning: increased |
| total FPS | 17409 | not used as model gate | training still advancing |

Loss/value, surrogate, entropy, learning rate, and policy standard deviation are
finite.  The latest 20 loss/value values are `0.0097..0.0364`, with no divergence.
The latest TensorBoard sampler reset-ratio mean is `0.099951`; checkpoint counts,
not this averaged plot, remain the formal ratio source.

Status at this snapshot: **CONTINUE / HEALTHY STATE PERSISTENCE**, with an
acceptance risk warning for increased slip, action acceleration, and physical
terminations.  The increases do not meet a declared automatic-stop condition
and may reflect the intended extra hard-slope exposure.  They must be judged by
the fixed post-training route/contact regression gates; training reward or
terrain level alone must not select the model.
