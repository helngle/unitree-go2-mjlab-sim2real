# Go2 proprioceptive student training readiness

Date: 2026-07-27

## Decision

```text
PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY=true
HARDWARE_READY=false
FORMAL_TRAINING_STARTED=false
DEFAULT_MODEL_CHANGED=false
```

The deployable proprioceptive student arm is ready for its single registered
training command.  This decision authorizes simulation training only.  It does
not authorize deployment on a physical Go2, and it does not promote an
untrained student over the current V7 simulation default.

The locked teacher and simulation default remain:

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

No checkpoint from the rejected stance-slip run is an initializer, candidate,
or default.

## Multi-agent resolution

The architecture/provenance audit selected a fixed ten-frame feed-forward
history and found the C++ gait phase was call-count dependent.  The main branch
now computes phase as a pure function of integer `policy_tick`, and the schema
freezes term order, units, frames, reset behavior, scales, action mapping and
ONNX metadata.

The distillation audit found that the initial runner used the wrong inheritance,
executed random student actions, discarded 9 of 24 rollout batches, and did not
fully lock teacher provenance.  The final implementation uses a real
`DistillationRunner`, executes deterministic teacher actions, consumes all 24
batches with an elementwise mean Huber loss, locks path/SHA, freezes teacher
parameters and normalization, supports registered distillation resume, and
transfers only the complete student actor into fresh PPO state.

The dynamics/authority audit confirmed that actuator IDs `[0,1,2]` are three
actuator groups covering all 12 joints, not three motors.  It also found an
encoder-bias mismatch and incomplete link robustness.  The student now observes
biased joint position, and the new task alone freezes foot friction `[0.3,1.2]`,
all-actuator effort `[0.9,1.1]`, Kp `[0.9,1.1]`, Kd `[0.8,1.2]`, and limb
pseudo-inertia mass scale `[0.95,1.05]`.  Existing base payload, COM, observation
noise, command, push and terrain distributions remain inherited from V7.

Nonzero observation/action timing randomization is not guessed without a Go2
measurement.  Its nominal value is frozen at zero in this arm; the ten-frame
history provides temporal context, but real SDK latency and jitter remain a
hardware calibration gate and may require a later registered robustness arm.

## Frozen model contract

The actor input is term-major.  Every history term is oldest-to-newest:

| term | per-frame dim | frames | flattened dim |
| --- | ---: | ---: | ---: |
| base angular velocity | 3 | 10 | 30 |
| projected gravity | 3 | 10 | 30 |
| command | 3 | 1 | 3 |
| gait phase | 2 | 1 | 2 |
| joint position relative to nominal | 12 | 10 | 120 |
| joint velocity | 12 | 10 | 120 |
| previous normalized action | 12 | 10 | 120 |
| **total** | | | **425** |

Ten 50 Hz samples have a `0.18 s` endpoint span and represent a nominal `0.20 s`
sampled context.  Reset backfills all history slots with the first finite frame;
the previous action resets to zero.  The actor has no height scan, contact truth,
base linear velocity, raycast truth or dynamics truth.

The privileged critic remains `261` dimensions and may use current height scan,
base linear velocity, foot height/air time/contact/contact force.  It produces
only value and is absent from the actor export.  The frozen V7 teacher input is
`234` dimensions and the action is `12` dimensions.  The teacher is a scan-based
upper reference, not an observation-matched baseline.

Actor and teacher MLPs use `512/256/128` ELU layers with observation
normalization.  PPO uses the same actor structure and a fresh `512/256/128`
critic.  The action contract is a 12-joint position target at `20 ms`, scale
`0.25 rad`, fixed nominal offsets, training-to-SDK map
`[3,4,5,0,1,2,9,10,11,6,7,8]`, normalized runtime limit `abs(action)<=4`,
and the twelve MJCF joint-position limits.  A normalized action that is within
the global bound but produces an out-of-range processed target fails closed.
Nominal PD is hip/thigh `20/1` and calf `40/2`; effort limits are
`23.5/23.5/45 Nm` per leg.  No mechanical capability was raised.

Canonical schema identity:

```text
379d982c61c839286fe7a566fee40160f599831752041250dbd804709a6e4b10
```

The deploy YAML SHA256 is
`3f4dcd0247f9cfa8a30ca7a85fe459f211b68d0b667a3a53520d49d478663ede`;
the schema artifact file SHA256 is
`9be31dfefa9a2fb0517695bd20d5ccbde802023ed2fa0ef088b37cffefb8e582`.
The latter contains the canonical schema hash above as a field.

## Initialization and runtime evidence

Stage 1 is 300 iterations of teacher-rollout behavior cloning on terrain levels
0-6 only.  The environment executes deterministic V7 teacher actions; the
student receives the matched proprioceptive state and learns one mean Huber
objective over all 24 rollout steps.  The teacher is never updated.

Stage 2 transfers only `student_state_dict` into the PPO actor.  The critic,
optimizer and PPO iteration start fresh at zero.  PPO then runs without a
teacher group or auxiliary imitation loss for 4000 iterations.  This prevents
known V7 failures on the hardest terrain from being permanently imposed through
hard imitation.

The formal 2048-environment, seed-42, eight-step no-learning artifact is:

```text
docs/reviews/go2_proprioceptive_student_preflight_2048env.json
SHA256 fff95311e3e0335ab244594c98891e7cc1d0a1569ebb3758857554bc251560a0
```

It verifies `425/261/234/12` shapes, finite observations/actions/rewards and
actor/critic forward paths, zero resets, no `learn`, no optimizer step, no
checkpoint write, exact deterministic teacher action, bitwise unchanged teacher
state and normalizer, single-input `[1,425]` ONNX export, `[1,12]` output,
schema metadata, and PyTorch/ONNX max absolute error
`4.842877388000488e-08`.  Peak GPU allocation/reservation was
`326579712/367001600` bytes.

Python targeted tests are `17/17 PASS`; provenance/locking and registered-retry
tests are `9/9 PASS`; full discovery is `429 PASS, 2 skip`.
The standalone C++ history test and the official-SDK2 headless
`unitree_mujoco` bridge test pass.  With official SDK2 HEAD
`21d0a3b2c46ee48c8fdf2783becb6be3beb0a59b`, both `unitree_mujoco` and
`go2_ctrl` compile in an isolated `/tmp` dependency prefix.  The headless bridge
loads `scene_go2.xml`, applies LowCmd to all 12 MuJoCo controls, publishes finite
motor/IMU LowState and advances the SDK tick.

The graphical simulator cannot create a window in the current noninteractive
session.  This does not invalidate the headless bridge cycle; visual simulator
and trained ONNX closed-loop behavior remain post-training acceptance work.

## Gate decision

| gate | state | evidence |
| --- | --- | --- |
| G0 provenance | PASS | branch/HEAD/worktrees recorded; teacher SHA exact; canonical source manifest embeds untracked source bytes; full-duration nonblocking lock; finite JSON |
| G1 student deployability | PASS | canonical 425-D schema; Python/YAML/C++ order and history tests |
| G2 critic/teacher isolation | PASS | PPO has only actor/critic; ONNX has one actor input; teacher frozen/unchanged |
| G3 initialization integrity | PASS | deterministic teacher rollout; full 24-step loss; actor-only PPO handoff; levels 0-6 BC |
| G4 action/runtime identity | PASS | 12-joint mock round-trip, fixed 20 ms/scale/PD/limits, processed-target guard, first-action latch and synchronized C++ runtime |
| G5 robustness contract | PASS | scoped ranges and all-12 coverage tested; finite nominal/randomized preflight; timing unknown explicitly frozen |
| G6 training contract | PASS | two task IDs construct; one arm/command; duplicate and provenance guards; selection preregistered |
| G7 export/deploy/preflight | PASS (interface only) | fresh-actor ONNX parity, C++ history, official SDK2 builds, bridge-only headless smoke and 2048-env preflight; trained ONNX/controller closed loop remains post-training |
| physical Go2 validation | HARDWARE_PENDING | model variant/firmware mode, measured latency, calibration, thermal/power and real E-stop |

## Source state

```text
branch exp/high-slope-probe-integration
HEAD   0a204b645a2325cb06264725c58cc5745da64a43
```

The worktree was already dirty and all prior user changes were retained.  No
reset, checkout, clean, branch switch or commit was performed.  Detailed
training and checkpoint acceptance rules are in
`docs/reviews/go2_proprioceptive_student_training_contract.md`.
