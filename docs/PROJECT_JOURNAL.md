# Unitree RL Mjlab 本地实验日志

最后更新：2026-06-25

## 当前目标

在 `/home/jensen/projects/unitree_rl_mjlab` 下走 Unitree 官方路线，先让 Go2 在 MuJoCo/MJLab 里完成训练、回放和 ONNX 导出。当前策略是先跑通 `Unitree-Go2-Flat`，建立稳定 baseline，再考虑 `Unitree-Go2-Rough`、更长训练和部署。

## 仓库与环境

- 官方仓库：`https://github.com/unitreerobotics/unitree_rl_mjlab.git`
- 本地路径：`/home/jensen/projects/unitree_rl_mjlab`
- 当前分支：`main`
- 当前官方 HEAD：`1425b15 Fix the warnings during rough-terrain training.`
- Conda 环境：`unitree_rl_mjlab`
- 关键依赖修正：
  - `mujoco==3.5.0`
  - `mujoco-warp==3.5.0`
  - `mjlab==1.2.0`
  - `warp-lang==1.12.0`
  - `scipy`

## 已解决的问题

1. `python scripts/list_envs.py` 初始因 MuJoCo 版本不匹配失败，降到 `mujoco==3.5.0` 后解决。
2. 训练初始因缺少 `scipy` 失败，安装后解决。
3. 训练初始因 `warp-lang==1.14.0` 缺少 `warp.context` 失败，降到 `warp-lang==1.12.0` 后解决。
4. 默认 W&B logger 需要登录，后续训练统一使用 `--agent.logger tensorboard`。
5. `--runner.max-iterations` 不是有效参数，应使用 `--agent.max-iterations`。

## 训练记录

### Smoke test

命令：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=64 \
  --agent.max-iterations=2 \
  --agent.run-name smoke \
  --agent.logger tensorboard
```

结果：训练链路跑通，能生成 TensorBoard 日志和 `policy.onnx`。

### Go2 Flat 512 envs / 1000 iter

命令：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=512 \
  --agent.max-iterations=1000 \
  --agent.run-name go2_flat_512env_1000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-06-25_15-48-27_go2_flat_512env_1000iter/model_999.pt
logs/rsl_rl/go2_velocity/2026-06-25_15-48-27_go2_flat_512env_1000iter/policy.onnx
```

关键结果：

```text
Train/mean_reward: 47.14
Train/mean_episode_length: 961.65
Episode_Termination/fell_over: 0
Episode_Termination/illegal_contact: 0
Episode_Reward/track_linear_velocity: 0.765
Episode_Reward/track_angular_velocity: 0.906
Metrics/slip_velocity_mean: 0.089
Episode_Metrics/mean_action_acc: 0.851
```

评价：平地回放可用，Go2 运动效果肉眼看起来不错，作为第一版 baseline 成立。

### Go2 Flat 1024 envs / 1000 iter

命令：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=1000 \
  --agent.run-name go2_flat_1024env_1000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt
logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/policy.onnx
```

关键结果：

```text
Train/mean_reward: 52.08
Train/mean_episode_length: 990.10
Episode_Termination/fell_over: 0
Episode_Termination/illegal_contact: 0
Episode_Reward/track_linear_velocity: 0.808
Episode_Reward/track_angular_velocity: 0.924
Episode_Reward/pose: 0.820
Metrics/slip_velocity_mean: 0.0786
Metrics/landing_force_mean: 76.17
Metrics/angular_momentum_mean: 0.238
Episode_Metrics/mean_action_acc: 0.753
Policy/mean_std: 0.378
Loss/value: 0.013
```

评价：比 512 env 版本更好。episode length 更接近上限，reward 更高，速度跟踪更好，打滑更少，动作更平滑。当前建议把它作为新的 Go2 Flat baseline。

## 回放命令

当前推荐回放最新 baseline：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt \
  --num-envs 1
```

如果 native viewer 有问题：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-25_16-35-33_go2_flat_1024env_1000iter/model_999.pt \
  --num-envs 1 \
  --viewer viser
```

## 重要理解

- `model_999.pt` 是训练 checkpoint，可用于回放、继续训练和调试。
- `policy.onnx` 是从当前 actor policy 导出的部署推理文件，不包含完整训练状态、critic 或 optimizer。
- `play.py` 回放的是训练后的当前策略，不是训练过程录像。
- `Unitree-Go2-Flat` 是平地速度跟踪任务；`Unitree-Go2-Rough` 是复杂地形任务。
- Go2 回放时运动轨迹看起来随机，是因为 `twist` 速度命令会随机采样；它不是在跟踪固定路线。

## 下一步建议

1. 用最新 `1024env_1000iter` 模型做几次回放，确认肉眼效果优于 512 env baseline。
2. 如果继续优化 Flat，跑更长训练：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_flat_1024env_3000iter \
  --agent.logger tensorboard
```

3. Flat 稳定后，再开始尝试 `Unitree-Go2-Rough`。
4. 后续如果要控制固定速度，可优先研究 `--viewer viser` 里的 joystick 面板，或修改 play 模式的 command 采样逻辑。
