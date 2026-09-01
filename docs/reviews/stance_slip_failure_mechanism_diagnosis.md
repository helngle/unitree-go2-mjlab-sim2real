# Stance-slip training failure mechanism diagnosis

Date: 2026-07-27 (Asia/Shanghai)

## Verdict

```text
new_gpu_rollout = true (evaluation-only; no PPO learning)
REWARD_AVOIDANCE = INCONCLUSIVE
  reduced-speed/short-step avoidance = SUGGESTIVE at model_13700 only
  unloading avoidance = NOT SUPPORTED
OBJECTIVE_CONFLICT = INCONCLUSIVE
PHYSICAL_AUTHORITY_LIMIT = SUPPORTED for V7 under the evaluator's MuJoCo contact model
  candidate-specific persistence = INCONCLUSIVE
next_training_action = DO_NOT_TRAIN
default_model_replaced = false
```

The rejected `-0.05` reward produced no safe checkpoint. The new diagnosis
strengthens a transient conservative-policy explanation at `model_13700.pt`,
but it does not establish the required time ordering and the pattern disappears
at later checkpoints. Existing friction v13 remains strong causal evidence that
V7 is traction-limited in this MuJoCo evaluator. It is not a candidate-specific
friction experiment for the three trained policies, so it cannot prove that
physical authority is the sole cause of this training failure.

No new reward, terrain, friction, termination, command, observation, network,
actuator, or PPO setting was introduced. Friction `1.2` was not used in the new
rollout. V7 remains the only default model.

## Identity and provenance

Branch `exp/high-slope-probe-integration`, HEAD
`0a204b645a2325cb06264725c58cc5745da64a43`; the existing dirty worktree was
preserved. Compared checkpoints were:

```text
/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff

/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/model_13700.pt
02f9e821739babb844598b735da5aaac4d42d8a88f203431d28689d06519f2fc

/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/model_13900.pt
4ab8740c7170b25923d4130b850fc77407f365923ad1634bb96a92ebf2eb8dea

/home/jensen/projects/unitree_rl_mjlab/logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/model_13999.pt
db46dcc1272cb0a722b695568c8cdf4d086af1075cd4d0b53da7a75a643563e3
```

Formal diagnosis artifact:

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/diagnostics/stance_slip_failure_mechanism_clean_seed42_r4_2400steps_v1.json
SHA256 27c9042870c065731aebf9038ea93ed55970bff97e93dbc1ef9bc4d03601655d
```

The artifact is recursively finite and stores full checkpoint paths/hashes,
branch/HEAD/dirty-state fingerprints, source hashes, seed, placement errors,
lifecycle, metric definitions, and the SHA256 of each one-checkpoint chunk.
Evaluator and dependency hashes at execution were:

```text
scripts/diagnose_go2_stance_slip_failure_mechanism.py 2b75af6f18b73e948c4d186949e50a95fba2552a6b159da525ca233c2f881ce8
scripts/diagnose_go2_high_slope_gait.py               084ad11c8dcbdab357cfe977e0d4b868dbfb733551747219e6ee2678d004443b
scripts/audit_go2_high_slope_actuators.py             07a957ffe8dc01178932c5993006201b315b44e5ca9b287e793d2755842dcc76
src/tasks/velocity/mdp/rewards.py                     7496518b0e8ce50a5488cf190441fab369b9fae0b8fbc1a81cbe4ada03e70b86
```

## Evidence audit

The formal acceptance artifacts are finite and matched across V7 and all four
training stages: clean/randomized, straight/arc/S, 16 slots per route, identical
terrain/route/placement identities. The selection result is unchanged:

```text
selection_status = NO_SAFE_SURVIVOR
violations: 13700=9, 13800=13, 13900=7, 13999=25
selection artifact SHA256 = 52e42466b10be55e55a74df1c0d368902eba118b6fd59b2f41eb08ed8f9a88bd
```

Across all 96 formal cells, V7 versus `13700` was:

| Metric | V7 | 13700 | Direction |
|---|---:|---:|---|
| Progress ratio | 0.5201 | 0.5266 | no global loss |
| Forward response gain | 0.5475 | 0.4609 | -15.8% |
| Exact 15 N tangent slip | 0.06136 | 0.06073 | -1.0% |
| Load-normalized slip cost | 0.3506 | 0.3358 | -4.2% |
| Exact 15 N loaded fraction | 0.5604 | 0.5727 | increased |
| Active steps | 992.3 | 1184.4 | increased |

This rejects the simple claim that `13700` unloaded its feet or merely ended
earlier to reduce the penalty. A stronger local pattern exists in the formal
clean slope-up high `vx=0.3`, matched-slot 1 cell: straight/arc/S all have equal
2400-step lifecycles, while `13700` reduces both gain and exact 15 N slip. For
example, straight changes gain `0.1285 -> 0.1112` and slip
`0.0591 -> 0.0204`; loaded fraction increases `0.575 -> 0.608`.

The new per-foot straight-command diagnostic used clean slope-up high/extreme,
`vx=0.3/0.5`, four matched repeats, seed 42, 100 warmup and the full 2400-step
horizon. Its separate gait-state mask is explicitly `20 N on / 10 N off`; it
does not replace the exact signed 15 N reward metric above.

At extreme `vx=0.3`, V7 and `13700` both ran all four attempts for 2400 steps:

| Metric | V7 | 13700 |
|---|---:|---:|
| Forward gain | 0.07418 | 0.07223 |
| Hysteretic loaded slip | 0.01572 | 0.01453 |
| Loaded fraction | 0.6283 | 0.5974 |
| Absolute step length | 0.01530 | 0.01383 |
| Stance duration (s) | 0.3928 | 0.3572 |
| Duty factor | 0.6250 | 0.5975 |

All `4/4` equal-full-horizon pairs meet the diagnostic's slip-down,
gain-down, and conservative-gait direction pattern. However, high has no
equal-full-horizon V7/13700 pairs, and the pattern is absent from the available
equal-horizon `13900` pairs and all later-stage aggregates. The gait output is
attempt-level rather than an onset trace, so it cannot prove that the
conservative change precedes progress loss or failure. That missing time order
is decisive under the preregistered rule.

## Objective conflict

The new term's TensorBoard contribution at `13700/13900/13999` is approximately
`-0.0147/-0.0166/-0.0167`, similar to the old foot-slip contribution and far
smaller than action-rate (`-0.273/-0.299/-0.306`) or linear tracking
(`+0.817/+0.822/+0.843`). This does not show reward-scale domination.

The required repeated inverse relation is also absent. `13700` has local cells
where slip improves while gain worsens, but later checkpoints generally worsen
slip, action acceleration, pitch, and contacts together. There is no zero-new-
reward continuation control, per-scenario reward-gradient proxy, or registered
onset sequence. Therefore `OBJECTIVE_CONFLICT` remains `INCONCLUSIVE`, not
`SUPPORTED` and not disproved.

## Physical authority

The new pre-reset actuator capture is valid for true terminal windows. On clean
slope-up high, failures were V7 `7/8`, 13700 `8/8`, 13900 `3/8`, and 13999
`7/8`. Every model had effort-saturation samples in most eligible terminal or
stable windows, and mean maximum foot tangent force was approximately
`28.1/33.2/30.7/32.2 N`. This confirms sustained control/contact demand but
does not isolate the limiting authority.

Prior causal evidence supplies that isolation for V7. The registered v13
friction `0.6/0.6/1.2` source/sham/probe has 64 matched triplets per speed:

| Cell | Completion source/sham/probe | Probe gain vs sham | Failure-risk ratio |
|---|---:|---:|---:|
| `vx=0.3` | 39 / 42 / 45 | +30.6% | 0.864 |
| `vx=0.5` | 11 / 10 / 49 | +74.1% | 0.278 |

Slip, cone utilization, gain, onset, and all 1.2x side-effect gates passed.
Conversely, 1.25x actuator headroom reduced persistent saturation by 98.2%
without recovering slope-up completion and introduced risk regressions. Thus
`PHYSICAL_AUTHORITY_LIMIT` is supported for V7 under this evaluator's MuJoCo
contact model, while actuator saturation is de-prioritized as the sole cause.
No same-scene friction triplet was run for the three stance-slip checkpoints;
candidate-specific persistence remains inconclusive.

## Execution note and limitations

The first monolithic invocation stopped before writing an artifact after
repeated MuJoCo-Warp environment construction produced CUDA graph-capture and
device-allocation errors. No training was active and no partial JSON existed.
The exact same evaluation matrix was rerun as one checkpoint per process so the
CUDA context was released between checkpoints. The CPU merger rejected any
config, identity, contract, or checkpoint-set mismatch and bound all four chunk
hashes into the final artifact. No experimental parameter changed.

The existing formal matched evaluator samples some mechanical telemetry after
automatic reset; this can slightly bias a terminating sample and is not used
for onset attribution here. The new actuator path captures the true pre-reset
terminal state, but the gait path still reports aggregate intervals. Randomized
physics values are deterministically reconstructed from seed but are not stored
per slot in the old acceptance JSON. These limits do not alter the original
REJECT decision, but they prevent stronger mechanism claims.

## Recommendation

Do not start another PPO run and do not add a second reward. There is not yet a
defensible next single training variable: the `-0.05` reward shows a transient
conservative response, later optimization does not preserve slip improvement,
and the strongest causal evidence points to a MuJoCo contact-authority limit
that reward shaping cannot change.

If candidate-specific attribution is required before redesign, the next action
is evaluation-only: apply the already registered `0.6/0.6/1.2` friction
source/sham/probe to the four fixed checkpoints on clean straight slope-up high
at both speeds, without changing training configuration. That experiment is not
authorization to use friction `1.2` in training or deployment.

Final default remains V7 `model_13600.pt`, SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`.
