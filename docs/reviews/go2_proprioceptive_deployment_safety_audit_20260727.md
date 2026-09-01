# Go2 proprioceptive deployment safety audit

Date: 2026-07-27

This is a CPU-only deployment-interface audit performed while the formal
two-stage training was active. It did not start a trainer, simulator evaluator,
or GPU workload, and it did not modify a training-critical source file.

## Verified interface contract

- The deploy YAML declares one static `425`-element actor input, one
  `12`-element normalized action, a `0.02 s` policy period, ten-sample
  oldest-to-newest histories, the frozen training-to-SDK joint map, nominal
  position offsets, action scale, PD gains, and per-joint position limits.
- Python schema validation rejects history, SDK joint-map, PD/action, schema
  identity, and processed-target-limit mismatches.
- The C++ ONNX runner rejects a schema metadata mismatch, dynamic/wrong input
  shape, multiple/wrong output shape, wrong input vector size, and nonfinite
  input/output.
- The C++ environment rejects nonfinite robot state, inference exceptions,
  wrong action dimension, nonfinite action, normalized action beyond the
  absolute limit, and processed position targets beyond the frozen joint
  limits. A valid action is required before `action_ready` becomes true.
- RL-state entry resets the readiness latch synchronously and installs a
  measured-position hold in SDK motor order. Position, Kp, Kd, zero velocity,
  and zero feed-forward torque are mapped together; malformed or duplicate
  mappings fail before any partial hold is written.
- The action cache is mutex-protected, the policy thread stop flag is atomic,
  and SDK quaternion input is normalized before projected gravity is formed.
- LowState timeout and `runtime_fault` are registered to transition the FSM to
  `Passive`.

The new executable test
`tests/go2_proprio_runtime_safety_smoke.cpp` exercises the real C++ observation,
action-manager, and environment headers with a mock articulation and inference
algorithm. It verifies the 425-D term-major history, reset backfill, previous
action timing, moving-command phase at 20 ms, processed targets, first-action
latch, and all listed fail-closed injections. This complements the existing
standalone history test and Python SDK-order round trip.

## CPU evidence

```text
Python targeted deploy/training-interface tests: 17/17 PASS
C++ history smoke: PASS
C++ runtime safety/fault-injection smoke: PASS
go2_ctrl rebuild: PASS
git diff --check for the new test: PASS
```

Build artifact identities in this session:

```text
go2_ctrl
e61ea9c68361ddb52be7adbf1bf9f997c5a717ad138abc8b05928de7264ed2f5

go2_proprio_history_smoke
877d50d8ed69e3b4db0c0b60695151f1d4ff9d0c1d2ad225f96235d807acdadb

go2_proprio_runtime_safety_smoke
83e72d90c07ab0a401546f8ebbc419f54aee7da8ac8ef4729d28dfe307e577cb
```

The tests used the already isolated official SDK2 dependency tree under
`/tmp/go2-sdk2-RIALJj`; no dependency was installed into the workspace.

## Open deployment gates

These results are interface evidence only. They do not make the untrained
candidate, a future checkpoint, or a physical Go2 ready for use.

1. There is no selected trained checkpoint or trained ONNX yet. Checkpoint
   provenance, export metadata, static `[1,425] -> [1,12]` shape, and
   PyTorch/ONNX numerical parity must be rerun on the selected checkpoint.
2. LowState timeout is covered, but there is no policy-inference deadline or
   stale-action watchdog. If ONNX inference blocks after one valid action, the
   1 kHz publisher can continue sending the last valid target instead of
   transitioning to `Passive`.
3. C++ checks the ONNX schema hash against the hash declared by `deploy.yaml`,
   but it does not cryptographically bind all deploy YAML bytes to the ONNX.
   The packaging gate must therefore run the canonical Python YAML validator
   and record both artifact SHA256 values.
4. The complete same-sample Python/C++ observation comparison, controller plus
   trained ONNX plus DDS/MuJoCo closed loop, and injected FSM transition to
   `Passive` remain post-training tests. The existing headless bridge test has
   no trained policy in its loop.
5. Graphical `unitree_mujoco` remains unavailable in the current headless
   session. Physical calibration, actual SDK timing/jitter, thermal/power
   limits, firmware control mode, and hardware emergency stop remain
   `HARDWARE_PENDING`.

Before a simulation acceptance decision, items 1, 2, and 4 must pass. Item 3
must be enforced by the immutable deployment artifact manifest. No result in
this audit supports `HARDWARE_READY`.

The registered watchdog design is a host-monotonic timestamp updated only after
successful inference and all action/processed-target checks. Once
`action_ready` is true, the 1 kHz FSM checks that age against a frozen
`policy_action_timeout_s`; expiration sets `runtime_fault` and transitions to
`Passive`. Tests must cover pre-first-action behavior, a blocked inference,
deadline boundary, clock monotonicity, recovery only through state re-entry,
and no stale target publication after the transition. The timeout value must be
part of the canonical deploy schema rather than a C++ literal. That schema is a
formal-training source, so the watchdog is deliberately deferred until the
active training has ended and its source manifest is no longer being guarded.
