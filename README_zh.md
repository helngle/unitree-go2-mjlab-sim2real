# Unitree Go2 MJLab Sim-to-Real

[English](README.md)

面向 Unitree Go2 运动控制研究的强化学习工程：覆盖 GPU 加速 MuJoCo 训练、可审计评估，
以及基于 ONNX 的仿真到实机部署。

本仓库源自
[unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab)，
目前重点是 Go2 复杂地形运动、连续路线评估、本体感知策略和 sim-to-real 安全。仓库仍保留
上游的多机器人资产，但这里新增并记录的研究流程以 Go2 为主。

> [!WARNING]
> 这是研究代码，不是可直接上实机的成品控制器。仿真中可运行的策略不等于实机安全。
> 任何实机试验前，都必须核对观测 schema、动作映射、ONNX 一致性、运行时限幅、吊装流程
> 和急停方案。

## 仓库包含什么

- 基于 MJLab 和 MuJoCo Warp 的 Go2 平地与粗糙地形 PPO 环境。
- 坡地、楼梯、圆弧、S 弯和连续地形过渡等复杂路线覆盖。
- 使用 matched seed 的跟踪、接触、打滑、执行器余量和失败归因工具。
- privileged teacher 与 contact-force teacher 研究流程。
- 面向 sim-to-real 的纯本体感知 student task、观测 schema、有界动作路径、preflight、
  checkpoint 筛选和验收工具。
- 使用 ONNX Runtime、包含关节命令安全检查的 Go2 C++ 部署运行时。
- `docs/` 中版本化的设计决策、项目日志、验收合同和精简结果汇总。

模型 checkpoint、原始 rollout 张量、筛选过程中生成的 ONNX 和训练日志不会提交到 Git。
运行回放、评估或部署前，需要在本机复现或提供对应产物。

## 目录结构

| 路径 | 用途 |
| --- | --- |
| `src/tasks/velocity/config/go2/` | Go2 task、地形、观测、奖励与 runner 配置 |
| `src/tasks/velocity/evaluation/` | 路线、地形、接触与验收指标 |
| `scripts/` | 训练、回放、评估、诊断、筛选与选择入口 |
| `deploy/robots/go2/` | Go2 C++ 控制器和部署配置 |
| `simulate/` | 集成的 Unitree MuJoCo 部署模拟器 |
| `tests/` | Python 合同测试和 C++ 部署 smoke 源码 |
| `docs/` | 项目计划、日志、评审和可审计实验结论 |

## 安装

推荐使用 Ubuntu 22.04、Python 3.11、NVIDIA GPU 和较新的 NVIDIA 驱动。系统依赖与
环境配置见[完整安装指南](doc/setup_zh.md)。

```bash
git clone https://github.com/helngle/unitree-go2-mjlab-sim2real.git
cd unitree-go2-mjlab-sim2real
pip install -e .
```

安装后列出已注册的 Go2 任务：

```bash
python scripts/list_envs.py --keyword Go2
```

## 基本流程

### 训练基线

检查新环境时，先用已注册的基线任务和较少的并行环境：

```bash
python scripts/train.py Unitree-Go2-Flat --env.scene.num-envs=256
```

正常单 GPU 训练时，可根据显存增加环境数量：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --gpu-ids 0 \
  --env.scene.num-envs=4096
```

训练输出位于 `logs/rsl_rl/go2_velocity/`；`logs/` 按设计只保留在本机。

### 回放 checkpoint

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_<iteration>.pt
```

使用固定前进命令检查确定性路线：

```bash
python scripts/play.py Unitree-Go2-Rough-V7 \
  --checkpoint-file /path/to/model.pt \
  --terrain-demo stairs_up_down \
  --fixed-vx 0.5
```

### Go2 高级研究流程

高级 task 是受合同约束的研究变体，不是可以随意互换的 preset。训练或评估前请阅读对应
合同与已有结论：

- [复杂运动项目计划](docs/V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md)
- [当前 Go2 项目日志](docs/V10_GO2_PROJECT_JOURNAL.md)
- [本体感知 student 就绪评审](docs/reviews/go2_proprioceptive_student_readiness.md)
- [Safe-action V2 设计](docs/reviews/go2_safe_action_v2_design.md)
- [部署安全审计](docs/reviews/go2_proprioceptive_deployment_safety_audit_20260727.md)
- [完整项目历史](docs/PROJECT_JOURNAL.md)

某个 task 或脚本存在，不代表相应候选策略已经通过验收。是否接受、拒绝或仍待评估，以对应
review artifact 为准。

## Go2 部署

部署运行时依赖 Cyclone DDS、Unitree SDK2、Eigen、yaml-cpp、Boost、fmt，以及仓库内按
架构提供的 ONNX Runtime 动态库。

1. 将兼容策略导出为 ONNX。
2. 保证模型与对应的 `deploy.yaml`、`observation_schema.json` 一致。
3. 将模型放到
   `deploy/robots/go2/config/policy/velocity/<version>/exported/policy.onnx`。
4. 在 `deploy/robots/go2/config/config.yaml` 中确认要加载的策略目录。
5. 编译控制器：

```bash
cmake -S deploy/robots/go2 -B deploy/robots/go2/build
cmake --build deploy/robots/go2/build -j
```

考虑实机前，先使用集成模拟器验证：

```bash
cmake -S simulate -B simulate/build
cmake --build simulate/build -j
./simulate/build/unitree_mujoco
./deploy/robots/go2/build/go2_ctrl --network=lo
```

仓库中的 `simulate/config.yaml` 默认选择 Go2 场景。

实机运行必须先完成独立安全审查并确认正确网卡，因此这里不提供可直接复制执行的实机
quick-start 命令。

## 可复现性与仓库规则

- 源码、schema、合同、决策和精简汇总进入 Git。
- `logs/`、checkpoint、原始 rollout 张量和临时筛选模型保留在 Git 之外。
- 评估产物记录 checkpoint hash 和 task ID。
- 新观测、奖励、网络、课程、训练机制和验收机制均属于受控研究变更；必须遵守
  [AGENTS.md](AGENTS.md) 中的文献依据与显式批准规则。

## 上游与许可证

本项目建立在
[MJLab](https://github.com/mujocolab/mjlab)、
[Isaac Lab](https://github.com/isaac-sim/IsaacLab)、
[RSL-RL](https://github.com/leggedrobotics/rsl_rl)、
[MuJoCo](https://github.com/google-deepmind/mujoco)、
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) 和
[Unitree SDK2](https://github.com/unitreerobotics/unitree_sdk2) 之上。

仓库使用 [Apache License 2.0](LICENCE)。
