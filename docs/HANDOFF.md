# 新聊天窗口交接摘要

当前项目：`/home/jensen/projects/unitree_rl_mjlab`

目标：沿 Unitree 官方 `unitree_rl_mjlab` 路线，在 MuJoCo/MJLab 里训练和回放 Go2。现在 `Unitree-Go2-Flat` 已经训练通，最新 baseline 效果不错。

## 已完成

- 保留旧项目：`/home/jensen/projects/quadruped_mujoco_rl`
- 使用官方仓库：`/home/jensen/projects/unitree_rl_mjlab`
- 当前官方 HEAD：`1425b15 Fix the warnings during rough-terrain training.`
- Conda 环境：`unitree_rl_mjlab`
- 关键依赖修正完成：
  - `mujoco==3.5.0`
  - `mujoco-warp==3.5.0`
  - `mjlab==1.2.0`
  - `warp-lang==1.12.0`
  - `scipy`
- `python scripts/list_envs.py` 已可用。
- `Unitree-Go2-Flat` smoke test、512 env 训练、1024 env 训练均已完成。
- TensorBoard 可用：

```bash
tensorboard --logdir logs/rsl_rl/go2_velocity
```

## 当前最佳模型

最新推荐 baseline：

```text
logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt
logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/policy.onnx
```

评价指标：

```text
Train/mean_reward: 52.08
Train/mean_episode_length: 990.10
Episode_Termination/fell_over: 0
Episode_Termination/illegal_contact: 0
Episode_Reward/track_linear_velocity: 0.808
Episode_Reward/track_angular_velocity: 0.924
Metrics/slip_velocity_mean: 0.0786
Episode_Metrics/mean_action_acc: 0.753
```

结论：Go2 Flat 已经稳定跑通，1024 env 版本优于 512 env 版本，建议作为当前 baseline。

## 回放命令

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt \
  --num-envs 1
```

如果 viewer 有问题：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt \
  --num-envs 1 \
  --viewer viser
```

## 已知注意点

- 训练命令使用 `--agent.max-iterations`，不是 `--runner.max-iterations`。
- 训练建议显式加 `--agent.logger tensorboard`，避免默认 W&B 登录问题。
- `model_999.pt` 用于回放/继续训练；`policy.onnx` 是部署推理用 actor，不是完整训练 checkpoint。
- `play.py` 里速度命令随机采样，所以轨迹看起来随机；不是回放固定路线。
- `Unitree-Go2-Flat` 平地任务较轻，1024 env 在 RTX 5060 Laptop 8GB 上可以跑。

## 下一步

1. 如果继续优化 Flat，跑 3000 iter：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_flat_1024env_3000iter \
  --agent.logger tensorboard
```

2. Flat 稳定后，再尝试 `Unitree-Go2-Rough`。
3. 如果想控制固定速度，优先研究 `--viewer viser` 的 joystick 面板，或修改 Go2 play 模式下 `twist` command 的采样逻辑。
