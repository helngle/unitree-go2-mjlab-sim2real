# Unitree Go2 MJLab Sim-to-Real

[简体中文](README_zh.md)

A research-oriented reinforcement learning stack for Unitree Go2 locomotion,
from GPU-accelerated MuJoCo training to auditable evaluation and ONNX-based
deployment.

This repository started from
[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)
and now focuses on Go2 rough-terrain locomotion, complex-route evaluation,
proprioceptive policies, and sim-to-real safety. The inherited multi-robot
assets remain in the tree, but the custom research workflow documented here is
Go2-specific.

> [!WARNING]
> This is research software, not a production-ready robot controller. A policy
> that works in simulation is not automatically safe on hardware. Validate the
> observation schema, action mapping, ONNX parity, runtime limits, suspended
> operation, and emergency-stop procedure before any real-robot trial.

## What is included

- Go2 flat- and rough-terrain PPO environments built on MJLab and MuJoCo Warp.
- Complex terrain and route coverage: slopes, stairs, arcs, S-curves, and
  continuous terrain transitions.
- Matched-seed evaluators and diagnostics for tracking, contact, slip,
  actuator headroom, and failure attribution.
- Privileged-teacher and contact-force-teacher research workflows.
- Proprioceptive sim-to-real student tasks, observation schemas, bounded action
  paths, preflight checks, checkpoint screening, and acceptance tooling.
- C++ Go2 deployment runtime with ONNX Runtime and joint-command safety checks.
- Versioned design decisions, experiment journals, acceptance contracts, and
  compact result summaries under `docs/`.

Model checkpoints, raw rollout tensors, generated ONNX screening files, and
training logs are intentionally not stored in Git. Reproduce or provide the
required artifacts locally before running playback, evaluation, or deployment.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/tasks/velocity/config/go2/` | Go2 task, terrain, observation, reward, and runner configuration |
| `src/tasks/velocity/evaluation/` | Reusable route, terrain, contact, and acceptance metrics |
| `scripts/` | Training, playback, evaluation, diagnosis, screening, and selection entry points |
| `deploy/robots/go2/` | Go2 C++ controller and deployment configuration |
| `simulate/` | Integrated Unitree MuJoCo deployment simulator |
| `tests/` | Python contract tests and C++ deployment smoke sources |
| `docs/` | Project plans, journals, reviews, and auditable experiment decisions |

## Installation

Recommended environment: Ubuntu 22.04, Python 3.11, an NVIDIA GPU, and a recent
NVIDIA driver. See the [full installation guide](doc/setup_en.md) for system
packages and environment setup.

```bash
git clone https://github.com/helngle/unitree-go2-mjlab-sim2real.git
cd unitree-go2-mjlab-sim2real
pip install -e .
```

List the registered Go2 tasks after installation:

```bash
python scripts/list_envs.py --keyword Go2
```

## Basic workflow

### Train a baseline

Start with a registered baseline task and a smaller environment count while
checking a new installation:

```bash
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=256
```

For a normal single-GPU run, increase the environment count as resources allow:

```bash
python scripts/train.py Unitree-Go2-Rough \
  --gpu-ids 0 \
  --env.scene.num-envs=4096
```

Training outputs are written below `logs/rsl_rl/go2_velocity/`. The `logs/`
directory is local-only by design.

### Play a checkpoint

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iteration>.pt
```

To inspect a deterministic route with a fixed forward command:

```bash
python scripts/play.py Unitree-Go2-Rough-V7 \
  --checkpoint-file /path/to/model.pt \
  --terrain-demo stairs_up_down \
  --fixed-vx 0.5
```

### Advanced Go2 workflows

The advanced tasks are controlled research variants, not interchangeable
presets. Read their contracts and recorded decisions before training or
evaluation:

- [Complex-locomotion project plan](docs/V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md)
- [Current Go2 project journal](docs/V10_GO2_PROJECT_JOURNAL.md)
- [Proprioceptive student readiness](docs/reviews/go2_proprioceptive_student_readiness.md)
- [Safe-action V2 design](docs/reviews/go2_safe_action_v2_design.md)
- [Deployment safety audit](docs/reviews/go2_proprioceptive_deployment_safety_audit_20260727.md)
- [Project history](docs/PROJECT_JOURNAL.md)

Do not infer acceptance from the presence of a task or script. The corresponding
review artifact is the source of truth for whether a candidate was accepted,
rejected, or still awaiting evaluation.

## Go2 deployment

The deployment runtime requires Cyclone DDS, Unitree SDK2, Eigen, yaml-cpp,
Boost, fmt, and the bundled architecture-specific ONNX Runtime libraries.

1. Export a compatible policy to ONNX.
2. Keep its `deploy.yaml` and `observation_schema.json` together with the model.
3. Place the model at
   `deploy/robots/go2/config/policy/velocity/<version>/exported/policy.onnx`.
4. Select or verify the intended policy directory in
   `deploy/robots/go2/config/config.yaml`.
5. Build the controller:

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j
```

Test with the integrated simulator before considering hardware:

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j
./simulate/build/unitree_mujoco
./deploy/robots/go2/build/go2_ctrl --network=lo
```

The checked-in `simulate/config.yaml` selects the Go2 scene by default.

Real-robot execution requires an explicit safety review and the correct network
interface; it is deliberately not presented as a copy-paste quick-start.

## Reproducibility and repository policy

- Keep source, schemas, contracts, decisions, and compact summaries in Git.
- Keep `logs/`, checkpoints, raw rollout tensors, and generated screening models
  outside Git.
- Record checkpoint hashes and task IDs in evaluation artifacts.
- Treat new observations, rewards, architectures, curricula, training
  mechanisms, and acceptance mechanisms as gated research changes. See
  [AGENTS.md](AGENTS.md) for the mandatory reference and approval policy.

## Upstream and license

The framework builds on
[MJLab](https://github.com/mujocolab/mjlab),
[Isaac Lab](https://github.com/isaac-sim/IsaacLab),
[RSL-RL](https://github.com/leggedrobotics/rsl_rl),
[MuJoCo](https://github.com/google-deepmind/mujoco),
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), and
[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2).

The repository is distributed under the [Apache License 2.0](LICENCE).
