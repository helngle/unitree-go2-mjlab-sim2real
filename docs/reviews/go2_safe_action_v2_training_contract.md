# Go2 safe-action V2 frozen training and acceptance contract

Date: 2026-07-30

## Experiment identity

This is one two-stage training arm whose only new training variable relative to
the formal proprioceptive V1 arm is action interface
`bounded_asymmetric_per_joint_v2`. Safety remains a permanent constraint; a
later arm may add one evidence-supported performance variable if V2 is safe but
fails ability gates.

Tasks:

```text
Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill
Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction
```

Unique command after all preflight gates pass:

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab
python scripts/train_go2_proprioceptive_safe_action.py
```

The orchestrator holds the shared full-duration nonblocking training lock,
rejects any existing matching V2 distillation or PPO run, verifies the teacher,
and binds every training-critical source in one canonical manifest. It validates
the manifest between stages and installs an identical copy in each run.

## Locked initialization and schedule

Teacher:

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

Stage 1:

```text
task: Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction-Distill
envs: 2048
env seed: 42
agent seed: 42
iterations: 300 (0..299)
steps/env: 24
save interval: 100
terrain: levels 0-6, no terrain-level advancement
teacher rollout: deterministic V7 raw z transformed once to T(z)
loss: elementwise mean Huber in V2 applied-action space
run suffix: go2_sim2real_proprio_v2_safe_action_meanbound5_v7_teacher_distill_2048env_300iter
```

Stage 2:

```text
task: Unitree-Go2-Rough-Sim2Real-Proprio-V2-SafeAction
envs: 2048
env seed: 42
agent seed: 42
iterations: fresh PPO 4000 (0..3999)
steps/env: 24
save interval: 250
initialization: complete V2 distilled actor only
fresh state: critic, optimizer, iteration and curriculum
teacher group/loss: absent
run suffix: go2_sim2real_proprio_v2_safe_action_meanbound5_ppo_2048env_4000iter
```

Observation, reward, terrain geometry/proportions, curriculum, commands,
randomization, actuator/PD settings, network and PPO parameters are byte-audited
against formal V1 configuration. V1 `model_3999.pt` is rejected and must not be
resumed; rejected stance-slip checkpoints are also forbidden.

Every V2 checkpoint must contain the canonical safe schema SHA in
`student_schema_sha256`, `action_interface=bounded_asymmetric_per_joint_v2`,
stage identity and source-manifest SHA. Distillation final iteration must be 299;
PPO final iteration must be 3999.

The first V2 PPO attempt and its `model_250.pt` recovery line are rejected.
Synchronous CUDA diagnosis reproduced a non-finite actor latent mean after the
applied-space likelihood ratio became unstable. The replacement run freezes
the Gaussian mean parameterization to `5*tanh(raw_mean/5)` while retaining the
same action interface, task variables, PPO hyperparameters and acceptance
gates. No checkpoint from the rejected attempt may seed the replacement run;
distillation and PPO are rerun under a new source manifest and the unique
`safe_action_meanbound5` run suffix.

## Stop and retry rules

Stop and preserve all evidence on NaN/Inf, OOM, simulator failure, missing
telemetry, duplicate run, source/schema/action-interface mismatch, wrong
checkpoint iteration, teacher leakage into PPO, action/target fault, or PPO
probability-semantic inconsistency. Only a purely technical pre-learning or
external interruption may be retried with identical code, config, inputs and
command; the failed run must remain registered. Experimental failure never
authorizes an in-place parameter change.

## Checkpoint schedule and screening

Screen every persisted PPO checkpoint:

```text
model_0.pt
model_250.pt through model_3750.pt at interval 250
model_3999.pt
```

Hard gates precede ranking:

1. exact provenance, lifecycle, finite tensors/JSON and 425-D actor schema;
2. one static ONNX actor input `[1,425]`, output `[1,12]`, no privileged input,
   and PyTorch/ONNX max absolute error `<=1e-5`;
3. zero reset storm and placement fault;
4. zero nonfinite action, zero applied-action fault and zero processed-target
   limit fault on screening and every rollout;
5. clean retained flat, ordinary rough, obstacle, stairs and line/arc/S
   completion at least `0.80` in every registered group;
6. randomized retained completion at least `0.70` in every group;
7. clean complex slope/stairs completion at least `0.65` per group;
8. randomized complex completion at least `0.55` per group;
9. mean forward response gain at least `0.75` on moving-forward profiles;
10. terrain-tangent slip, absolute pitch, action acceleration, effort, power,
    energy, failure risk, base contact, upper-leg contact and calf contact each
    no greater than the same-scene V7 value times `1.2`.

Body-contact metrics are hard 1.2x guardrails. They enter lexicographic ranking
only as final tie-breakers after every previously registered ability and safety
key is exactly tied. Stand/basic tracking is diagnostic-only in V2 because no
quantitative preregistered baseline exists; results must be reported, but no
threshold may be invented after seeing checkpoints.

## Exact acceptance inventory

For every candidate checkpoint and V7 reference, execute the existing serial
14-invocation matrix without deleting or substituting an item:

1. high-slope matched, jointly producing clean and randomized line coverage;
2. flat matched line/arc/S, jointly producing clean and randomized groups;
3. continuous retained clean;
4. continuous retained randomized;
5. terrain-curves arc clean;
6. terrain-curves arc randomized;
7. terrain-curves S-curve clean;
8. terrain-curves S-curve randomized;
9. level-9 stairs seed 42 clean;
10. level-9 stairs seed 43 clean;
11. level-9 stairs seed 44 clean;
12. level-9 stairs seed 42 randomized;
13. level-9 stairs seed 43 randomized;
14. level-9 stairs seed 44 randomized.

The retained continuous invocation must include ordinary rough, discrete
obstacle, stairs up and stairs down at registered levels. Every artifact binds
checkpoint path/SHA, source manifest, evaluator/dependency hashes, profile,
scene, route, seed, scenario identity and recursive-finite status. GPU execution
is serial. The exact inventory is verified before bundle assembly; missing,
extra or duplicated entries fail closed.

## Selection

Among hard-gate survivors select lexicographically:

1. highest minimum randomized complex-terrain completion;
2. highest minimum clean complex-terrain completion;
3. highest minimum retained-scene completion;
4. highest minimum line/arc/S completion;
5. highest minimum forward response gain, then best command tracking;
6. lowest failure risk, terrain-tangent slip, action acceleration, effort,
   power/energy and absolute pitch;
7. lowest body-contact metrics only when all preceding keys are exactly tied;
8. earlier checkpoint if still tied.

No post-hoc weighted score is allowed. Final checkpoint and training reward have
no automatic preference.

## Promotion boundary

A selected checkpoint must additionally pass eager/ONNX/C++ action parity,
history/action round-trip, trained-policy headless and graphical
`unitree_mujoco` closed loop, and timeout/stale/nonfinite/action-fault to Passive
fallback. V7 remains the simulation default until the complete matrix passes.

Without a physical Go2 the maximum result is:

```text
SIMULATION_ACCEPTED
DEPLOYMENT_BUNDLE_READY
HARDWARE_READY=false
HARDWARE_PENDING
```

Physical promotion still requires the specific Go2/firmware low-level mode,
encoder and IMU calibration, measured latency/jitter, motor torque/thermal/power
checks, working E-stop/fallback, suspended trials, low-speed flat trials and
staged terrain expansion.
