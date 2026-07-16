# V7 S-Curve Training Decision

This review is limited to pure-simulation locomotion. It does not change the
production command sampler, terrain curriculum, rewards, task registration, or
policy. The default checkpoint is V7 `model_13600.pt`.

## Decision

**NO-GO: do not start either S-curve training probe.**

The clean baseline does not show an S-curve-specific locomotion or command
bandwidth failure. The fixed command tape misses its ideal endpoint, but its yaw
reversal is fast and stable while forward response has the same general
under-gain already measured on straight and arc commands. With feedback and
enough horizon, the same V7 policy completes all 18 clean S-curves with very
small path and heading error.

Randomized flat also passes the route gate after correcting one horizon false
failure. Complex terrain S-curves are still untested because the current
evaluator explicitly reports `rough_curves=false` and
`terrain_transitions=false`.

## Evidence

The valid clean results are:

```text
route_baseline_s_curve_command_tape_clean_seed42_18env_1600steps.json
route_baseline_s_curve_closed_loop_clean_seed42_18env_2000steps.json
```

Both use V7 `model_13600.pt`, seed 42, zero initial cross-track/yaw error, radii
`{1.5, 2.5, 4.0} m`, speeds `{0.3, 0.5, 0.6} m/s`, and both first-turn signs.
Four cases are OOD for V7 general mode: `r=1.5 m` with `v={0.5,0.6} m/s` in
both directions, because `|wz|>0.3 rad/s`. The other 14 are ID.

The prior valid arc tape provides the steady reference: ID `vx` gain `0.8108`
and `wz` gain `0.9525`, with arc closed loop `18/18` after a sufficient
horizon. The matched-speed flat forward reference is about `0.8540` versus ID
coupled arc `0.8490`. S-curve segment gains below are consistent with that
existing forward under-response rather than a new yaw-coupling collapse.

### Fixed command tape

| metric | ID (n=14) | OOD (n=4) |
| --- | ---: | ---: |
| completion | 0/14 | 0/4 |
| progress ratio | 0.8271 | 0.8534 |
| segment-1 vx gain | 0.7902 | 0.7790 |
| segment-2 vx gain | 0.8209 | 0.8518 |
| segment-1 wz gain | 0.9408 | 0.9249 |
| segment-2 wz gain | 0.9357 | 0.9045 |
| yaw sign-switch latency | 0.0457 s | 0.0400 s |
| controller saturation | 0 | 0 |
| slip | 0.0273 | 0.0357 |
| action acceleration | 0.0728 | 0.0895 |

There are zero resets and zero fell/base/upper-leg/calf contacts. Tape
completion is not a valid standalone training gate here: the tape changes
segments and stops at ideal step indices, so a persistent `vx` gain near 0.8
necessarily leaves endpoint progress near 0.8 even when yaw reversal works.

### Closed loop

Closed loop completes `18/18`, with progress ratio 1.0, placement error 0,
mean lateral RMS `0.00659 m`, worst lateral max `0.03174 m`, mean heading RMS
`1.423 deg`, and worst heading max `4.658 deg`. It has zero resets, zero listed
catastrophic/contact terminations, mean slip `0.02858`, and mean action
acceleration `0.07524`.

For the 14 ID cases, segment `vx` gains are `0.7970/0.8214`, segment `wz` gains
are `0.9456/0.9388`, and mean sign-switch latency is `0.0314 s`. Controller
saturation is zero in every case. The largest ID closed-loop per-step `wz`
delta is `0.4820 rad/s`; this intentional reversal is abrupt, but V7 changes
yaw sign in roughly 1.6 control steps. There is no command-bandwidth shortfall.

The online gain uses least-squares `sum(command*actual)/sum(command^2)` and
returns null for near-zero command energy. This fixes the earlier meaningless
near-zero-`vy` ratio spike. Closed-loop settling remains descriptive only:
its target changes continuously, so it must not be interpreted like settling
to a fixed step target.

### Randomized flat closed loop

The 2000-step randomized matrix completes `17/18` (`94.44%`), with mean
progress `0.99939`, mean lateral RMS `0.02653 m`, worst lateral max
`0.12909 m`, mean heading RMS `1.890 deg`, and worst heading max `6.485 deg`.
It has placement error 0, zero resets, zero listed catastrophic/contact
terminations, zero controller saturation, mean slip `0.03674`, and mean action
acceleration `0.23258`.

The only unfinished case is ID `r=4.0 m, v=0.3 m/s`, first turn right. It
reaches about 98.9% and reports `step_limit`. An otherwise identical 2400-step
retry completes `1/1`, with final position error `0.0256 m` and zero
reset/contact. This is an evaluator-horizon false failure, not a policy
shortfall.

Randomized action acceleration is about 3.1 times the clean mean. The profile
also enables observation corruption, startup dynamics randomization, and
pushes, so clean is not a matched attribution baseline. Until a randomized
straight/arc reference exists, this scalar remains a robustness risk to track;
it neither proves an S-specific deficit nor authorizes an S sampler.

## What V7 Training Covered

V7 holds each sampled command for a uniformly random `3--8 s`. Its base modes
are general/lateral/yaw/high-speed = `40/25/15/20%`. General mode independently
samples:

```text
vx in [0.15, 0.8] m/s
vy in [-0.1, 0.1] m/s
wz in [-0.3, 0.3] rad/s
```

Consequently, ordinary resampling does expose the policy to command steps and
some forward commands followed by the opposite yaw sign. Under independent
mode/sign sampling, a boundary has probability
`0.4 * 0.4 * 0.5 = 0.08` of being general-to-general with an opposite nonzero
yaw sign. It does **not** deliberately cover an S-curve sequence: consecutive
yaw magnitudes are almost surely unequal, `vx` changes independently, `vy` is
not fixed to zero, and there is no constraint `|wz|=vx/r`.

For the evaluator's `pi/3` arc segments, ideal dwell is `(pi/3)*r/v`. Only five
of the nine radius/speed pairs lie inside `3--8 s`. `r=1.5,v=0.6` is shorter;
`(r,v)={(2.5,0.3),(4.0,0.3),(4.0,0.5)}` are longer. Thus V7 also lacks exact
dwell coverage for four pairs per turn direction.

This distinction matters for causal claims. V7 cannot be described as having
explicitly trained on S-curves, but the clean baseline empirically demonstrates
that its incidental command-step exposure generalizes to the required ID yaw
reversal.

## Failure Classification

| candidate cause | current classification | evidence / next action |
| --- | --- | --- |
| evaluator bug | horizon false failure identified | r4/v0.3/right completes at 2400 steps; step-index tape, reset freeze, placement 0 and robust gains otherwise valid |
| controller bug | not supported on flat | clean 18/18; randomized 17/18 plus 1/1 horizon retry; small errors and zero saturation |
| command bandwidth | not supported | ID yaw switch 0.031--0.046 s and wz gain about 0.94 |
| locomotion policy | general forward under-gain only | vx about 0.79--0.82; no extra S/yaw degradation versus prior straight/arc evidence |
| terrain geometry | untested, not failed | current evaluator is flat-only; validate corridor/scan/relocation before complex-terrain claims |
| training curriculum | unsuitable if left active | endpoint-displacement curriculum is not a curve tracking metric and can change assignment differently between control/probe |

## Mutually Exclusive Future Probes

These are designs, not authorizations. Select exactly one only after a matched
baseline demonstrates its corresponding deficit. Both retain V7 rewards,
terrain geometry, termination, gait, randomization, network, optimizer, and
observation definitions.

### Probe A: steady correlated curve mode

Use only if ID steady coupled response is materially worse than matched pure
forward/yaw response.

- Replace 15 percentage points of general mode: general/curve becomes
  `25/15%`; lateral/yaw/high-speed remain `25/15/20%`.
- Sample `vx in [0.3,0.6] m/s`, `vy=0`, turn sign equiprobably, and radius
  conditionally in `[max(1.5, vx/0.3), 4.0] m`; set `wz=sign*vx/r`.
- This probe is ID-only: `|wz|<=0.3 rad/s`. Keep the four higher-curvature
  matrix cases as OOD evaluation, not training evidence.
- Preserve V7 dwell exactly: one piecewise-constant command for uniform
  `3--8 s`.
- Preserve V7 boundary behavior: an instantaneous one-control-step change,
  with no new slew limiter.

The only changed distribution is the 15% correlated `(vx,wz)` quota. Do not
also change resampling cadence or add transition state.

### Probe B: controlled yaw-transition sequence

Use only if steady coupled tracking is matched but ID sign-switch latency,
overshoot, IAE, or failure rate is materially worse than matched command steps.

- Reserve 15% of rollout command fragments for a two-segment transition
  sequence; the remaining 85% uses the unmodified V7 sampler. Report achieved
  time-weighted exposure, not only selection counts.
- Sample `vx in [0.3,0.6] m/s`, `vy=0`, `|wz| in [0.075,0.3] rad/s`, and the
  first sign equiprobably. Segment 2 keeps the same `vx` and `|wz|` and flips
  only the yaw sign.
- Draw one dwell uniformly in `3--8 s` and use it for both segments, avoiding
  a duration asymmetry confound.
- Switch in one control step with no slew limiter. If deployment later requires
  a ramp, that is a separate controller experiment and cannot be combined with
  this probe.
- Keep all transition training ID. Evaluate `|wz|>0.3` separately as OOD.

Probe A and Probe B must never be enabled together. Probe A changes steady
component correlation; Probe B changes temporal correlation. Combining them
would make attribution impossible.

## Matched Control / Probe Contract

If a future baseline authorizes one probe, run an original-V7 control and the
selected probe independently:

```text
checkpoint: model_13600.pt for both (never chain control -> probe)
num_envs: 2048 for both
iterations: 300 for both
seed: 42 for both
terrain_levels curriculum term: removed for both
terrain assignment: restored once, then frozen for the full run
```

Before rollout 1, require exact equality of the checkpoint's 2048-element
`terrain_levels` and `terrain_types` tensors in both runs. Preserve element
order; equal histograms are insufficient. Emit for both tensors:

- shape, dtype, min/max, and histogram;
- an order-sensitive SHA-256 over a canonical little-endian int64 encoding;
- start and end hashes, which must be identical within each run and across
  control/probe;
- an assertion that every `env_origin` equals
  `terrain_origins[terrain_level, terrain_type]` after restoration.

Remove only the `terrain_levels` curriculum term after config construction.
Do not modify production `terrain_levels_vel()`. Assert assignments at startup,
after checkpoint restore, periodically during training, and before the final
save. Any mismatch invalidates the matched experiment.

The present runner can restore checkpoint `terrain_levels/types` and relocate
the roots consistently, but there is not yet an authorized production API that
enforces the periodic/start/end freeze assertions and hashes above. That missing
wiring is an additional training gate, not permission to approximate the match
with equal histograms.

## Reopen / GO Criteria

Training can be reconsidered only after all of the following are true:

1. Final integration acceptance passes, including JSON finite checks.
2. A valid complex-terrain or new matched response matrix shows an ID policy
   deficit not present in the current clean/randomized flat results.
3. The deficit is not ordinary forward under-gain, controller saturation,
   evaluator horizon, route geometry, or terrain relocation.
4. Matched pure-axis/coupled or steady/transition evidence selects exactly one
   probe above.
5. The frozen terrain assignment interface and start/end hash checks exist.
6. There is enough time for both 300-iteration control and probe plus the
   unchanged post-training matrix.

Until then: **NO-GO, no PPO training, and V7 `model_13600.pt` remains the
default model.**
