# Go2 proprioceptive action rollout safety diagnostic (2026-07-29)

## Scope

This is an evaluation-only diagnostic of the rejected proprioceptive PPO final
checkpoint. It does not change the task, reward, action mapping, checkpoint or
default model.

Candidate:

```text
logs/rsl_rl/go2_velocity/2026-07-29_10-10-14_go2_sim2real_proprio_v1_ppo_2048env_4000iter_exact_resume_1501_3999/model_3999.pt
SHA256 d48d08188c0823e42610a9ffd5de4cead2093af2cc9171517bb7099c40bb4760
```

Matched simulation reference:

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

The action fault contract is unchanged: a control step faults if any normalized
action has absolute value greater than `4.0`, or if
`default_joint_position + 0.25 * action` exceeds that joint's MJCF position
range. Counts below are active original-attempt control steps where at least one
of the twelve joints faults.

## Coverage

The candidate was evaluated serially on GPU with the frozen acceptance
evaluators. Existing matched V7 raw JSON was reused without rerunning it.

- flat line, arc and S-curve, clean and full randomized;
- continuous ordinary rough, discrete obstacle, stairs up and stairs down at
  levels 3/5/7, clean and randomized;
- registered high-slope line, arc and S-curve, clean and randomized, levels
  0/1, speeds 0.3/0.5 m/s and radius 2.5 m.

All JSON is recursive finite and binds checkpoint/evaluator/dependency
identities. No GPU evaluators ran concurrently.

## Results

| Suite | Profile | Model | Completed | Fault scenarios | Fault steps | Maximum `abs(action)` |
|---|---|---|---:|---:|---:|---:|
| flat line/arc/S | clean | candidate | 48/54 | 0/54 | 0 | 2.4586 |
| flat line/arc/S | randomized | candidate | 50/54 | 0/54 | 0 | 2.7794 |
| continuous retained | clean | candidate | 11/12 | 1/12 | 6 | 4.3208 |
| continuous retained | randomized | candidate | 11/12 | 3/12 | 23 | 7.4655 |
| high slope line/arc/S | clean | candidate | 26/48 | 25/48 | 933 | 8.5521 |
| high slope line/arc/S | randomized | candidate | 26/48 | 28/48 | 973 | 10.7027 |
| flat line/arc/S | clean | V7 | 54/54 | 0/54 | 0 | 2.4540 |
| flat line/arc/S | randomized | V7 | 53/54 | 0/54 | 0 | 3.1692 |
| continuous retained | clean | V7 | 10/12 | 1/12 | 5 | 4.1861 |
| continuous retained | randomized | V7 | 11/12 | 4/12 | 18 | 5.0236 |
| high slope line/arc/S | clean | V7 | 9/48 | 34/48 | 2489 | 27.1537 |
| high slope line/arc/S | randomized | V7 | 7/48 | 43/48 | 3178 | 54.8979 |

Across the registered diagnostic suites, the candidate totals are:

```text
clean:      85/114 completed, 26/114 fault scenarios, 939 fault steps,
            max abs(action)=8.5521
randomized: 87/114 completed, 31/114 fault scenarios, 996 fault steps,
            max abs(action)=10.7027
```

The matched V7 totals are:

```text
clean:      73/114 completed, 35/114 fault scenarios, 2494 fault steps,
            max abs(action)=27.1537
randomized: 71/114 completed, 47/114 fault scenarios, 3196 fault steps,
            max abs(action)=54.8979
```

The continuous candidate faults are localized to difficult stairs: clean
level-7 stairs down has six fault steps; randomized level-7 stairs up has 17,
level-5 stairs down has one and level-7 stairs down has five. Ordinary rough,
discrete obstacle and all registered flat routes have zero candidate action
faults in this diagnostic.

Some high-slope scenarios report a fault even when their maximum normalized
action is below `4.0`. This confirms that processed per-joint target-limit faults
also occur; the evidence is not limited to the global normalized-action bound.

## Interpretation

```text
ACTUAL_ROLLOUT_ACTION_FAULT = CONFIRMED
SYNTHETIC_ONLY_EXPLANATION = REJECTED
FLAT_ACTION_SAFETY = SUPPORTED within registered coverage
COMPLEX_TERRAIN_ACTION_SAFETY = FAILED
```

The candidate is materially better than V7 on high-slope completion and action
safety, but relative improvement cannot replace the preregistered absolute
zero-fault deployment gate. V7 remains a simulation default, not a
hardware-ready action-safety reference. This diagnostic supports the existing
`TRAINING_REJECTED` decision and does not select or promote `model_3999.pt`.

The next training arm should not repeat the same unconstrained action contract.
It should preregister one action-interface change that is identical in training,
ONNX and C++ deployment: a bounded output with an asymmetric per-joint safe
mapping derived from nominal pose, action scale and MJCF joint limits. Applied
actions must be fed back as previous-action observations. A deployment-only
silent clamp is not an acceptable fix because it creates a train/deploy
mismatch.

## Raw evidence

```text
logs/rsl_rl/go2_velocity/action_rollout_diagnostic_20260729/model_3999_flat_matched_routes.json
SHA256 8d6157eda676ce23b82eb824dc444c8244e770576b1e41aa6dada1b98fa5ea65

logs/rsl_rl/go2_velocity/action_rollout_diagnostic_20260729/model_3999_continuous_retained_clean.json
SHA256 571bacde4f199a59b40f0aa55b76eb20f66f1c1c797c0980d672d7e43f039b4a

logs/rsl_rl/go2_velocity/action_rollout_diagnostic_20260729/model_3999_continuous_retained_randomized.json
SHA256 89f149fc1d44fad20f3d488a38c5e4dad6a6955d5dd02daa25553ff33402df3b

logs/rsl_rl/go2_velocity/action_rollout_diagnostic_20260729/model_3999_high_slope_matched.json
SHA256 3d3974bbb4f127abd3c9f5932af6bc5e4e34b45cbab93bd88556e22119db9df7
```
