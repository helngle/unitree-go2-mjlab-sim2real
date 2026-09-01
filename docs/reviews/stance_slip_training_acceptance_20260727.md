# V7 stance-slip single-variable training acceptance

Date: 2026-07-27 (Asia/Shanghai)

## Verdict

```text
training_completed = true
checkpoint_selection = NO_SAFE_SURVIVOR
candidate = none
verdict = REJECT
default_model_replaced = false
```

The registered 2048-environment, 400-iteration training completed without NaN,
Inf, OOM, resume mismatch, telemetry loss, or simulator failure. All four saved
stages failed the preregistered high-slope safety/ability screen, so none entered
the retained-scene/stairs/path full suite. V7 `model_13600.pt` remains default.

## Training provenance

Start: `2026-07-27T09:57:40+08:00`
Final artifact time: `2026-07-27T10:16:04+08:00`

```text
run: logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter
task: Unitree-Go2-Rough-V7-StanceSlip
environments: 2048 on cuda:0
iterations: 400 (13600 -> 13999; 9600 common environment steps)
seed: env=42, agent=42
only intervention: terrain_tangent_stance_slip weight=-0.05
```

Warm start:

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

The resumed run also saved its own same-named initial snapshot. It is not the
source file above and is distinguished by full path and hash:

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/model_13600.pt
SHA256 8e513953771454d5efaa0d25bd1f5c73682c7137a5ee9fd255c5c11d83cf7afe
```

The 2048-env no-learning preflight restored iteration `13600` and terrain mean
level `5.275`:

```text
stance_slip_training_preflight_seed42_2048env_8steps_fullresume_20260727_v4.json
SHA256 21c4c5fda987b1cc1f6332d7a3268e5b9b7156c8bd1bbb2f3f9a17af1a101331
```

TensorBoard contains `62` scalar tags and `24800` scalar values. All values are
finite; observed step range is `3..13999`. Final telemetry includes mean reward
`50.28495`, mean episode length `990.5`, tangent stance-slip `0.072481 m/s`,
load-normalized cost `0.330645`, and terrain-ray valid fraction `1.0`.

## Checkpoint identity

```text
model_13700.pt  02f9e821739babb844598b735da5aaac4d42d8a88f203431d28689d06519f2fc
model_13800.pt  5c0b909232e2df6e5b0616731acecdee567e00c7bda4842ccbe99ae650ab04bd
model_13900.pt  4ab8740c7170b25923d4130b850fc77407f365923ad1634bb96a92ebf2eb8dea
model_13999.pt  db46dcc1272cb0a722b695568c8cdf4d086af1075cd4d0b53da7a75a643563e3
```

Every checkpoint contains actor, critic, optimizer, iteration, and
`infos.env_state`. The run also contains final ONNX, YAML configs, TensorBoard
event data, and captured source diff.

## Evaluation evidence

The high-slope evidence layer now records the frozen local-terrain-tangent
loaded-stance formula and base pitch while retaining the old world-XY slip
metric. Twenty-nine evaluator/reward CPU contracts and a one-slot GPU smoke
passed before formal evaluation. Formal coverage was clean plus randomized,
straight/arc/S, both `vx=0.3/0.5`, 16 matched slots per route, seed 42, radius
2.5, 2400 steps, and 10 settle steps.

Dirty-worktree source provenance used by all formal JSONs:

```text
scripts/evaluate_go2_high_slope_matched.py
  SHA256 63e5e12c18dbd19add4a19e070a06325660433de3a5f07d91ccc52784561ae46
src/tasks/velocity/evaluation/terrain_rollout_metrics.py
  SHA256 6d648b522e290f91ea8847262c978a2c894eca8f44f560010bd8286c3f2022e4
scripts/select_go2_stance_slip_checkpoint.py
  SHA256 9efcf179ee14d08e39d70aab0e233500532a3a3ee98f2fe3a860803358dc08f9
```

Completion counts (straight / arc / S):

| Checkpoint | Clean | Randomized |
|---|---:|---:|
| V7 reference | 4 / 2 / 3 | 6 / 4 / 4 |
| 13700 | 5 / 2 / 4 | 4 / 2 / 4 |
| 13800 | 4 / 4 / 4 | 5 / 3 / 4 |
| 13900 | 4 / 4 / 4 | 6 / 4 / 4 |
| 13999 | 4 / 4 / 4 | 5 / 4 / 5 |

No stage approached clean `12/16` or randomized `10/16`. Minimum weighted
forward gain across the six profile/route groups was `0.3691`, `0.2480`,
`0.2486`, and `0.3990`, all below the required `0.80`.

Hard `1.2x` guardrails were applied per profile and route before ranking:

| Checkpoint | Violations | Violating objectives |
|---|---:|---|
| 13700 | 9 | base/upper-leg/calf contact, failure risk |
| 13800 | 13 | slip, action acceleration, pitch, base/upper-leg/calf contact |
| 13900 | 7 | slip, action acceleration, pitch, base/calf contact |
| 13999 | 25 | slip, action acceleration, pitch, base/upper-leg/calf contact, failure risk |

Therefore `survivor_count=0`; no lexicographic checkpoint selection was
possible. Strict selection artifact:

```text
acceptance/stance_slip_checkpoint_selection.json
SHA256 52e42466b10be55e55a74df1c0d368902eba118b6fd59b2f41eb08ed8f9a88bd
```

## Gate status

| Objective / gate | Status | Evidence |
|---|---|---|
| Training lifecycle and finite telemetry | PASS | Complete run; finite TensorBoard |
| Clean high-slope completion | FAIL | Every stage below 12/16 on every route |
| Randomized high-slope completion | FAIL | Every stage below 10/16 on every route |
| Completion improvement vs V7 | FAIL | No stage reaches +0.20 on every route |
| High-slope forward gain | FAIL | Stage minima 0.2480..0.3990, required 0.80 |
| Terrain-tangent loaded-stance slip | FAIL | 13800/13900/13999 violate 1.2x |
| Action acceleration | FAIL | 13800/13900/13999 violate 1.2x |
| Base pitch | FAIL | 13800/13900/13999 violate 1.2x |
| Body contact safety | FAIL | Every stage violates at least one contact gate |
| Failure risk | FAIL | 13700 and 13999 violate 1.2x |
| Placement/lifecycle/finite/provenance | PASS | Strict matched JSON validation |
| Retained flat/rough/obstacle/stairs/path suite | NOT ENTERED | No safe stage survived selection |
| Default replacement | FAIL / NO | V7 remains default |

Any failed gate rejects the candidate and forbids adding a second training
variable. No additional training, tuning, deployment change, or default
replacement was performed. At final inspection the GPU was idle and no related
train/evaluate/audit/play/TensorBoard process remained. Final targeted CPU
verification was `31 PASS`; compileall and `git diff --check` passed.
