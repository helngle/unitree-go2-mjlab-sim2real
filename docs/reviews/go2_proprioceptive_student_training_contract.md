# Go2 proprioceptive student frozen training contract

Date: 2026-07-27

## Scope

This is one fixed two-stage arm: deterministic V7 teacher-rollout BC followed
by pure PPO.  Formal training was not started during readiness work.  Once
started, do not change task, observations, history, rewards, terrain,
curriculum, commands, dynamics randomization, network, action, actuator, PPO,
seeds, environment count, schedule or initialization.

## Unique command

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab
python scripts/train_go2_proprioceptive.py
```

The orchestrator first verifies the exact V7 teacher SHA, acquires a
full-duration nonblocking file lock and refuses a duplicate matching run.  It
generates a canonical source manifest whose SHA is persisted in every
checkpoint.  The manifest records all training-critical file hashes and embeds
the complete bytes of untracked training source; a verbatim copy is installed
in each run.  Any source drift between stages aborts the handoff.  It then
executes these frozen stages:

```text
Stage 1 task: Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill
envs/seeds: 2048, env=42, agent=42
schedule: 300 iterations, 24 steps/env, save interval 100
teacher: exact V7 model_13600.pt path/SHA
terrain: retained levels 0-6, no terrain-level advancement
rollout: deterministic teacher actions
objective: elementwise mean Huber over all 24 batches
run suffix: go2_sim2real_proprio_v1_v7_teacher_distill_2048env_300iter

Stage 2 task: Unitree-Go2-Rough-Sim2Real-Proprio-V1
envs/seeds: 2048, env=42, agent=42
schedule: pure PPO 4000 iterations, 24 steps/env, save interval 250
initialization: complete distilled actor only
fresh state: critic, optimizer and PPO iteration zero
teacher loss/group: absent
run suffix: go2_sim2real_proprio_v1_ppo_2048env_4000iter
```

The stage-1 handoff is exactly the newly created `model_299.pt`.  It must contain
`proprioceptive_stage=distillation`, the registered teacher SHA and student
schema SHA.  The orchestrator resolves exactly one new distillation directory;
regex ambiguity, missing output or any provenance mismatch aborts stage 2.

Expected resource envelope on the current RTX 5060 Laptop 8 GB is approximately
`5.5-7.5 GB` during learning and `4-6 hours` total.  These are planning estimates,
not gates; actual peak memory, start/end times and duration must be recorded.

## Stop rules

Stop the active stage and preserve logs on NaN/Inf, OOM, simulator failure,
teacher/schema/checkpoint mismatch, missing telemetry, optimizer ownership
violation, absent expected checkpoint, or duplicate run discovery.  Do not
repair an experimental failure by changing a frozen factor.  A rerun may be
considered only for a purely technical failure with byte-identical config,
inputs and command after the failed run is retained and identified.

## Checkpoint screening

Every persisted PPO checkpoint is identified by full path and SHA256.  The final
checkpoint is not preferred automatically, and TensorBoard reward is not a
selection criterion by itself.

Apply hard gates before ranking:

1. provenance, lifecycle, finite JSON/tensors and exact 425-D schema;
2. PyTorch/ONNX parity at max absolute error `<=1e-5`, one actor input and no
   teacher/critic/height truth;
3. no NaN/Inf, reset storm, simulator placement error or action-limit fault;
4. clean retained flat/ordinary rough/obstacle/stairs and line/arc/S completion
   at least `0.80` in every registered group;
5. randomized retained completion at least `0.70` in every registered group;
6. clean complex slope/stairs completion at least `0.65` per group and
   randomized complex completion at least `0.55` per group;
7. mean forward response gain at least `0.75` on moving forward profiles;
8. slip, absolute pitch, action acceleration, effort/energy, base contact,
   upper-leg contact, calf contact and failure risk each no greater than the
   same-scene V7 safety reference times `1.2`.

V7 has privileged height scan and therefore is an upper/safety reference, not
an observation-matched student baseline.  Ability is judged by the absolute
gates above; V7 comparisons are used for safety and retained-capability context,
not to hide student failure behind a relative score.

Among hard-gate survivors, select lexicographically:

1. highest minimum randomized complex-terrain completion;
2. highest minimum clean complex-terrain completion;
3. highest minimum retained-scene completion;
4. highest minimum line/arc/S completion;
5. highest minimum forward response gain, then best command tracking;
6. lowest failure risk, slip, action acceleration, effort/energy and pitch;
7. earlier checkpoint if still tied.

Do not create a post-hoc weighted total score.

## Required acceptance matrix

Run same-scene clean and randomized evaluation on flat, ordinary rough,
discrete obstacle, continuous slopes, continuous stairs and the registered
high-slope/stairs cases.  Include randomized level-9 stairs seeds 42/43/44,
forward speeds represented in training and line/arc/S paths.  Preserve raw JSON,
commands, seeds, scene placement, checkpoint path/SHA, evaluator/dependency
hashes and recursive-finite checks.

The selected checkpoint must then pass:

```text
PyTorch -> ONNX numerical parity
canonical schema metadata and static 1x425 -> 1x12 shapes
C++ observation history/action round-trip
official SDK2 headless unitree_mujoco bridge cycle
graphical unitree_mujoco closed-loop run with the trained ONNX
timeout/stale/nonfinite/action-limit -> Passive fallback
```

Any hard-gate failure rejects that checkpoint.  If there is no survivor, the
training arm is `REJECT`; do not add a second objective in the same experiment.

## Promotion boundary

V7 `model_13600.pt` remains the simulation default until a student checkpoint
passes the complete simulation matrix.  Even then, the student is only a
simulation candidate.  Physical promotion additionally requires a specific Go2
variant and firmware low-level mode, measured state/action latency and jitter,
encoder zero and IMU calibration, torque/thermal/power checks, suspension and
low-speed trials, working E-stop/fallback, then staged terrain expansion.
