# High-slope probe evaluation contract

Date: 2026-07-21
Status: PRE-GPU command audit complete; no GPU evaluation has been started by this Agent.

## Scope and immutable evaluation identity

This document fixes checkpoint ranking and post-training evaluation for the single-variable
`Unitree-Go2-Rough-V7-HighSlopeProbe` run.  The training task may contain the 10% reset sampler,
but every policy evaluation below deliberately uses `Unitree-Go2-Rough-V7`.  This keeps the
evaluation terrain assignment, rewards, terminations, commands, observations, randomization and
network identical to the V7 baseline; only the actor weights come from the trained checkpoint.

Baseline checkpoint:

```text
/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

Expected training checkpoints are `model_13700.pt`, `model_13800.pt`, `model_13900.pt`, and the
final `model_13999.pt`.  Missing or unloadable checkpoints are recorded as unavailable; they are
never silently replaced by another iteration.  All GPU commands are sequential.  No evaluator may
run while PPO owns the GPU.

## Evaluator audit

| Requirement | Script and exact interface | Audit result |
| --- | --- | --- |
| Matched high-slope straight/arc/S | `scripts/evaluate_go2_high_slope_matched.py` | Fresh same-seed environment per route kind; matched slot order; active-attempt freeze; P95/max slip/action/contact; r=2.5 is valid and r=4 is rejected by scan preflight. |
| Level-9 stairs seeds 42/43/44 | `scripts/evaluate_go2_routes.py --terrain-suite continuous` | Exact prior two-attempt approach-feature-exit stairs matrix. It records completion, reset/failure and mean slip/action. |
| Flat/rough/obstacle path regression | `scripts/evaluate_go2_routes.py --terrain-suite patch` | Correct terrain relocation and route-attempt freeze. Use the exact prior 2.5 m, 0.4 m/s, 700-step matrix. |
| Continuous terrain regression | `scripts/evaluate_go2_terrain_boundary.py --suite continuous_straight` | Formal metric wrapper adds cross-track/heading P95, slip/action P95/max and non-terminating body contact rates. Use the exact prior six-case, levels 7/9 matrix. |
| Fixed-command tracking regression | `scripts/evaluate_go2_rough.py` | Corrected terrain relocation; provides response gain, cross-axis velocity, slip/action mean and terminations by command, level and terrain type. It does not freeze after reset, so it is a fixed-duration tracking regression, not route completion evidence. |

CPU verification in this worktree:

```text
python -m unittest -v \
  tests.test_go2_high_slope_matched \
  tests.test_go2_route_scenarios \
  tests.test_go2_terrain_boundary_scenarios \
  tests.test_go2_high_slope_acceptance

54/54 PASS
python -m py_compile on all four evaluator scripts: PASS
all four --help CLI contracts: PASS
git diff --check before this document: PASS
```

`pytest` is not installed in the `unitree_rl_mjlab` conda environment; the repository's unittest
entry points were used instead.

## Authoritative V7 baseline artifacts

`BASE` below means:

```bash
BASE=/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter
```

The following artifacts are immutable comparison sources:

| Matrix | File | SHA256 / baseline result |
| --- | --- | --- |
| High-slope clean | `high_slope_matched_clean_r2p5_seed42_2400steps_v2.json` | `adf8d73eda47a11fe16a7bb4e1e5cf0795ff01daec4e3b0c961dc319798562ac`; straight/arc/S `4/16, 3/16, 3/16`, mean vx gain about `0.504/0.516/0.526`. |
| High-slope randomized | `high_slope_matched_randomized_r2p5_seed42_2400steps.json` | `19bd9292d596bf7ec3a889114cd618335b27f3568cba1d9817a72cf6dcdc5259`; `5/16, 4/16, 4/16`, gain about `0.505/0.566/0.557`. |
| Stairs randomized seed42 | `high_slope_stairs_level9_randomized_seed42_2env_2400steps.json` | `e5ea6197379cedc944baf4b0a0983d620487a0bb254e8b9f3aad6393f376f5cc`. |
| Stairs randomized seed43 | `high_slope_stairs_level9_randomized_seed43_2env_2400steps.json` | `fc3ff5b109573ae616ba02e3aa1795cdafe429ab3ef3ef8175bfdcc39b32240b`. |
| Stairs randomized seed44 | `high_slope_stairs_level9_randomized_seed44_2env_2400steps.json` | `79963d6221f95d68caee5b11ea5bc3b934fd1b9a671021e53088112cba50599a`; aggregate up/down are each `2/3`, with one calf failure in different seeds. |
| Patch clean | `route_baseline_patch_matrix_line_follow_clean_seed42.json` | `7299aad534866f959d7b2f9fadb8509c10603302072c825fb17bb8b89dffbf3a`; full seven-terrain matrix `112/112`. |
| Patch randomized | `route_baseline_patch_matrix_line_follow_randomized_seed42.json` | `9b4f19e3d40fd64aa155c58d5e002e1c3e616b38cab76bc346f92eb2c75f43e8`; `112/112`. |
| Continuous clean | `route_boundary_continuous_straight_clean_levels7_9_metricsv2_seed42_12env_2400steps.json` | `1b203568c4181adddd987f9c2f661010efd1c3d534847c88a8aacfaf5bea9751`; `12/12`. |
| Continuous randomized | `route_boundary_continuous_straight_randomized_levels7_9_metricsv2_seed42_12env_2400steps.json` | `f08e2cc90f2d98f757a81553e0aec049e0a711da5b0318dd787e5b723efd3b46`; `10/12`, with the two known level-9 stairs calf resets. |

The old non-`v2` clean high-slope JSON and the 1800-step continuous JSON are audit-only and must
not be used for model selection.  For fixed-command regression, rerun V7 and the candidate in one
command with the current corrected evaluator; do not use pre-relocation terrain labels.

## Fixed stage-checkpoint ranking

The ranking screen is clean only and intentionally half the size of the final matrix:

```text
slope_up + slope_down
high + extreme (levels 0/1 in this evaluation-only generator)
r=2.5 m, v=0.5 m/s, left/right, one repeat
8 matched slots per route kind, 24 route attempts per checkpoint
2400 control steps, settle=10, seed42
```

After training has exited and the GPU is free, set the actual run directory and execute
sequentially:

```bash
source /home/jensen/anaconda3/etc/profile.d/conda.sh
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab

RUN_DIR=/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/<ACTUAL_HIGH_SLOPE_RUN>

for TAG in 13700 13800 13900 13999; do
  CKPT="$RUN_DIR/model_${TAG}.pt"
  test -f "$CKPT" || { echo "UNAVAILABLE $CKPT"; continue; }
  python scripts/evaluate_go2_high_slope_matched.py \
    --checkpoint "$CKPT" \
    --task-id Unitree-Go2-Rough-V7 \
    --profiles clean \
    --slope-directions slope_up slope_down \
    --levels 0 1 \
    --radii 2.5 \
    --speeds 0.5 \
    --turn-signs 1 -1 \
    --repeats 1 \
    --steps 2400 \
    --settle-steps 10 \
    --seed 42 \
    --output-file "$RUN_DIR/stage_rank_high_slope_clean_seed42_r2p5_v0p5_8slots_2400steps_model_${TAG}.json"
done
```

For each checkpoint calculate, from its three `route_results`, the following lexicographic score:

```text
1. total completed attempts across straight + arc + S                 higher wins
2. minimum completed count among straight, arc and S                  higher wins
3. sample-count-weighted mean non-null response_gain.vx               higher wins
4. sum of fell/base/upper-leg/calf termination counts                 lower wins
5. checkpoint iteration                                               lower wins exact ties
```

No cross-track, heading, slip or action metric may override a checkpoint that loses an earlier
rank item.  They remain veto checks: non-finite JSON, placement error above `1e-4`, invalid matched
identity, or slip/action above `1.2x` the matching V7 scene disqualifies a checkpoint.  This rule
prevents selecting a visually attractive arc while straight or S remains collapsed.

## Completed stage ranking

The four fixed clean stage screens completed under:

```text
/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-21_16-21-46_go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

Independent CPU-only review loaded every JSON with the standard library JSON parser, recursively
checked every float for finiteness, ran production
`assert_recursive_json_finite()` and `validate_matched_result_invariants()`, and independently
checked the declared config, task, checkpoint, matched slot order, route identities, geometry,
placement, completion/failure lifecycle, sample count and termination counters.  All four files
passed.  Maximum terrain and route placement errors were below `1e-4`.

The independently recomputed score is:

| checkpoint | straight/arc/S completed | total | minimum route | weighted vx gain | four termination counts | rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `model_13700.pt` | `0/1/3` | 4 | 0 | 0.507963 | 18 | 4 |
| `model_13800.pt` | `4/2/3` | 9 | 2 | 0.452512 | 12 | 2 |
| `model_13900.pt` | `3/3/3` | 9 | 3 | 0.548234 | 13 | **1** |
| `model_13999.pt` | `3/2/3` | 8 | 2 | 0.458741 | 13 | 3 |

The strict lexicographic order is therefore:

```text
model_13900.pt > model_13800.pt > model_13999.pt > model_13700.pt
```

`model_13900.pt` is the nominated full-matrix candidate.  This is not an ACCEPT decision: its
stage screen is only `9/24` total and every route is `3/8`, so it must still pass the unchanged
clean/randomized full matrix and all regressions below.

Artifact SHA256:

```text
model_13700 stage JSON  82bec06d1e8a3a84728756ea3f19ce0687a45ec03b0b6d2b4d8d913d80adb903
model_13800 stage JSON  a63eec5ca4de51350cb9dc75cd14eaea1ccecbc67123837553741e1770faac82
model_13900 stage JSON  c22708ec7c9a9cac7eb13f90c173dc40d30c8ceee8654def618a79b6db865782
model_13999 stage JSON  fa1b0d782c57cfddbd13c2a2eeafb26d4c3523c912d5091d2d4a1403f2f74e90
```

The `model_13700` JSON records git HEAD `1225a57`; the other three record `30dd6d9`.  The exact
commit diff between those revisions changes only
`docs/reviews/high_slope_probe_training_monitor.md`; no evaluator, environment, policy, config or
test code changed.  The executable evaluation identity is therefore unchanged, but the provenance
difference remains explicitly recorded rather than hidden.

## Full post-training commands

Set these after stage ranking:

```bash
V7=/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
RUN_DIR=/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/<ACTUAL_HIGH_SLOPE_RUN>
CANDIDATE="$RUN_DIR/model_<SELECTED_ITERATION>.pt"
FINAL="$RUN_DIR/model_13999.pt"

EVAL_TAGS=(candidate)
EVAL_CKPTS=("$CANDIDATE")
if test "$FINAL" != "$CANDIDATE"; then
  EVAL_TAGS+=(final)
  EVAL_CKPTS+=("$FINAL")
fi
```

If `CANDIDATE` and `FINAL` are different, run every checkpoint-specific route matrix for both.
If they are identical, write one set and record that candidate equals final.  Use a distinct `TAG`
in every filename.

### 1. Full clean and randomized high-slope matched matrices

```bash
for INDEX in "${!EVAL_TAGS[@]}"; do
  TAG=${EVAL_TAGS[$INDEX]}; CKPT=${EVAL_CKPTS[$INDEX]}
  for PROFILE in clean randomized; do
    python scripts/evaluate_go2_high_slope_matched.py \
      --checkpoint "$CKPT" \
      --task-id Unitree-Go2-Rough-V7 \
      --profiles "$PROFILE" \
      --slope-directions slope_up slope_down \
      --levels 0 1 \
      --radii 2.5 \
      --speeds 0.3 0.5 \
      --turn-signs 1 -1 \
      --repeats 1 \
      --steps 2400 \
      --settle-steps 10 \
      --seed 42 \
      --output-file "$RUN_DIR/post_${TAG}_high_slope_matched_${PROFILE}_seed42_r2p5_v0p3_0p5_16slots_2400steps.json"
  done
done
```

### 2. Randomized level-9 stairs, seeds 42/43/44

This intentionally uses `evaluate_go2_routes.py`, matching the existing three baseline files.

```bash
for INDEX in "${!EVAL_TAGS[@]}"; do
  TAG=${EVAL_TAGS[$INDEX]}; CKPT=${EVAL_CKPTS[$INDEX]}
  for SEED in 42 43 44; do
    python scripts/evaluate_go2_routes.py \
      --checkpoints "$CKPT" \
      --task-id Unitree-Go2-Rough-V7 \
      --mode line_follow \
      --terrain-suite continuous \
      --transition-cases stairs_up stairs_down \
      --levels 9 \
      --repeats 1 \
      --cross-track-offsets 0.0 \
      --yaw-offsets 0.0 \
      --route-length 6.0 \
      --target-speed 0.5 \
      --steps 2400 \
      --seed "$SEED" \
      --profile randomized \
      --output-file "$RUN_DIR/post_${TAG}_stairs_level9_randomized_seed${SEED}_2env_2400steps.json"
  done
done
```

### 3. Flat/random-rough/discrete-obstacle patch path regression

The required subset is 3 terrain families x 4 levels x 4 repeats = 48 attempts per checkpoint.
It is taken from the prior seven-family 112-attempt matrix without changing geometry or controller.

```bash
for INDEX in "${!EVAL_TAGS[@]}"; do
  TAG=${EVAL_TAGS[$INDEX]}; CKPT=${EVAL_CKPTS[$INDEX]}
  for PROFILE in clean randomized; do
    python scripts/evaluate_go2_routes.py \
      --checkpoints "$CKPT" \
      --task-id Unitree-Go2-Rough-V7 \
      --mode line_follow \
      --terrain-suite patch \
      --terrain-types flat random_rough discrete_obstacles \
      --levels 0 3 5 7 \
      --repeats 4 \
      --cross-track-offsets 0.0 \
      --yaw-offsets 0.0 \
      --route-length 2.5 \
      --target-speed 0.4 \
      --steps 700 \
      --seed 42 \
      --profile "$PROFILE" \
      --output-file "$RUN_DIR/post_${TAG}_patch_flat_rough_obstacle_${PROFILE}_seed42_48env_700steps.json"
  done
done
```

### 4. Six-case continuous straight regression

```bash
for INDEX in "${!EVAL_TAGS[@]}"; do
  TAG=${EVAL_TAGS[$INDEX]}; CKPT=${EVAL_CKPTS[$INDEX]}
  for PROFILE in clean randomized; do
    python scripts/evaluate_go2_terrain_boundary.py \
      --checkpoint "$CKPT" \
      --task-id Unitree-Go2-Rough-V7 \
      --suite continuous_straight \
      --route-kind straight \
      --transition-cases slope_up slope_down stairs_up stairs_down random_rough discrete_obstacle \
      --transition-levels 7 9 \
      --speeds 0.5 \
      --cross-track-offsets 0.0 \
      --yaw-offsets 0.0 \
      --repeats 1 \
      --steps 2400 \
      --seed 42 \
      --profile "$PROFILE" \
      --output-file "$RUN_DIR/post_${TAG}_continuous_straight_${PROFILE}_levels7_9_seed42_12env_2400steps.json"
  done
done
```

### 5. Corrected fixed-command tracking regression

Run V7 and each distinct candidate/final in the same invocation so the evaluator code, task,
matrix, seed and profile are identical.  This is the source for retained-scene response gain,
cross-axis velocity and by-terrain-type tracking comparisons.

```bash
CHECKPOINTS=("$V7" "$CANDIDATE")
if test "$FINAL" != "$CANDIDATE"; then CHECKPOINTS+=("$FINAL"); fi

for PROFILE in clean randomized; do
  python scripts/evaluate_go2_rough.py \
    --checkpoints "${CHECKPOINTS[@]}" \
    --task-id Unitree-Go2-Rough-V7 \
    --levels 3 5 7 9 \
    --repeats 2 \
    --steps 1000 \
    --seed 42 \
    --command-cases forward_0.3 forward_0.6 forward_0.9 lateral_left lateral_right yaw_left yaw_right \
    --profile "$PROFILE" \
    --output-file "$RUN_DIR/post_v7_candidate_final_tracking_${PROFILE}_seed42_1120env_1000steps.json"
done
```

## Acceptance and reporting rules

Stage ranking only nominates a candidate; it never accepts the model.  The selected candidate and
final checkpoint are evaluated independently against the same predeclared gate.  The final
checkpoint is not accepted merely because it is later, and its failure does not erase a passing
stage candidate:

- clean high-slope: each route kind at least `0.70` completion (at least `12/16`) and at least
  `+0.20` absolute over its V7 route baseline;
- randomized high-slope: each route kind at least `0.60` (at least `10/16`) and at least `+0.20`
  absolute over V7;
- full-matrix high-slope mean forward response gain at least `0.80`;
- every retained-scene forward gain in the fixed-command comparison is no more than `0.05` below
  its V7 value;
- same-scene slip and action acceleration are no more than `1.2x` V7; all values are finite;
- no material increase in fall/base/upper-leg/calf terminations or contact rates, and no new
  catastrophic failure class;
- stairs up and down each remain at least `2/3` over seeds 42/43/44, with no more than the existing
  two aggregate calf failures and no new fall/base/upper-leg failure;
- flat/random-rough/discrete-obstacle patch completion remains `48/48` in both profiles;
- continuous clean remains `12/12`; randomized remains at least `10/12`, with slope/rough/obstacle
  unchanged and no worse failure severity than the two known stairs calf resets;
- every result passes schema, recursive finite, matched identity, placement (`<=1e-4`) and
  original-attempt freeze checks.

The Acceptance Agent gives the final PASS/FAIL.  Any failed critical gate means `REJECT`, V7
`model_13600.pt` remains default, and this round must not start a second PPO probe or change a
second variable.

## Known risks

1. The evaluation worktree does not contain ignored `logs/`; use the absolute main-repository
   checkpoint/run paths shown above.
2. `evaluate_go2_routes.py` provides only mean slip/action values.  Formal P95/max/contact-rate
   gates come from high-slope matched and terrain-boundary outputs.
3. `evaluate_go2_rough.py` continues sampling after automatic reset.  Use it only for fixed-duration
   tracking regression, never for path completion or first-failure claims.
4. A 2400-step high-slope screen is intentionally expensive because shorter horizons previously
   caused false `step_limit` failures.  Do not shorten it during candidate selection.
5. The clean and randomized high-slope V7 files were generated in separate fresh processes.  Static
   matrix identity is authoritative; dynamic placement floats are not compared by exact dictionary
   equality.
