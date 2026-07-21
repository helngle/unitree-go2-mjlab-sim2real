# Unitree RL Mjlab 本地实验日志

最后更新：2026-07-21

## 当前目标

在 `/home/jensen/projects/unitree_rl_mjlab` 下优化统一 Go2 rough-terrain locomotion policy。当前默认模型仍为 V7 `model_13600.pt`。10% high-slope hard-case sampling probe 已完成 2048-env、400-iteration 正式训练和完整验收；sampler/训练管线 PASS，但 candidate/final 的 high-slope completion、forward gain 和 tail-risk gate FAIL，final 另有楼梯退化，因此两个新 checkpoint 均已 REJECT。本轮不追加 PPO；下一步先诊断持续高坡落脚/步长不足，再决定新的单变量机制。

## 记录约定

每个阶段结束后都要把实验思路、遇到的问题、改用的新方法、关键指标和下一步建议记录到本文档；`docs/HANDOFF.md` 同步保留一份更短的新窗口交接摘要。

## 仓库与环境

- 官方仓库：`https://github.com/unitreerobotics/unitree_rl_mjlab.git`
- 本地路径：`/home/jensen/projects/unitree_rl_mjlab`
- 当前分支：`exp/high-slope-probe-integration`
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

### Go2 Flat 2048 envs resume / plus 1000 iter

从 `1024env_1000iter/model_999.pt` 继续训练，环境数改为 2048。

命令核心参数：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=2048 \
  --agent.resume=True \
  --agent.load-run 2026-06-25_16-35-33_go2_flat_1024env_1000iter \
  --agent.load-checkpoint model_999.pt \
  --agent.max-iterations=1000 \
  --agent.run-name go2_flat_2048env_resume999_plus1000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt
logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/policy.onnx
```

关键结果：

```text
Train/mean_reward: 54.71
Train/mean_episode_length: 994.19
Episode_Reward/track_linear_velocity: 0.841
Episode_Reward/track_angular_velocity: 0.937
Episode_Reward/pose: 0.841
Episode_Termination/fell_over: 0
Episode_Termination/illegal_contact: 0
Metrics/slip_velocity_mean: 0.072
Episode_Metrics/mean_action_acc: 0.667
Policy/mean_std: 0.337
Loss/value: 0.003
```

评价：这是目前最好的 Flat baseline。平地慢速/中速速度跟踪稳定，回放效果较好。高速度“跑步”仍然不应视为已经学好，因为当前课程在约 5000 PPO iterations 前主要训练较温和速度，且 gait/reward 没有为高速奔跑单独设计。

### Flat capacity tests

为了确认 RTX 5060 Laptop 8GB 的并行训练上限，做过 Flat 快速容量测试：

```text
4096 env: 10 iter 成功，约 48k-53k steps/s
6144 env: 10 iter 成功，约 52k steps/s
8192 env: 10 iter 成功，约 50k-51k steps/s，显存约 5.3GB
10240 env: 未完成初始化/无有效 run dir
```

结论：Flat 可运行上限大约是 8192，但长训更建议 4096 或 6144；Rough 更重，初始训练仍建议 1024。

### Go2 Rough 1024 envs / 3000 iter

命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_rough_1024env_3000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-06_10-04-23_go2_rough_1024env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-06_10-04-23_go2_rough_1024env_3000iter/policy.onnx
```

关键结果：

```text
Tail100 reward: 39.74
Tail100 episode length: 943
Tail100 terrain level: 1.11
Tail100 track linear velocity: 0.667
Tail100 track angular velocity: 0.818
Tail100 fell_over: 0.007
Tail100 illegal_contact: 0.185
Tail100 slip velocity: 0.111
Tail100 mean_action_acc: 1.076
Best reward: 42.57 around iteration 2337
```

评价：能形成 low-level rough baseline，但没有学到高难度复杂地形。terrain level 长期只在约 1.1 附近。

### Go2 Rough resume 2800 / plus 1500 iter

命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-06_10-04-23_go2_rough_1024env_3000iter \
  --agent.load-checkpoint model_2800.pt \
  --agent.max-iterations=1500 \
  --agent.run-name go2_rough_resume2800_plus1500iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-06_13-40-01_go2_rough_resume2800_plus1500iter/model_4299.pt
```

关键结果：

```text
Tail100 reward: 40.48
Tail100 episode length: 944
Tail100 terrain level: 1.25
Tail100 track linear velocity: 0.684
Tail100 track angular velocity: 0.822
Tail100 fell_over: 0.003
Tail100 illegal_contact: 0.181
Tail100 slip velocity: 0.110
Tail100 mean_action_acc: 1.066
```

评价：相比第一段 rough 有小幅提升，但仍停留在低难度 terrain。继续原配置训练的收益开始变低。

### Go2 Rough resume 4299 / plus 4500 iter

命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-06_13-40-01_go2_rough_resume2800_plus1500iter \
  --agent.load-checkpoint model_4299.pt \
  --agent.max-iterations=4500 \
  --agent.run-name go2_rough_resume4299_plus4500iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/model_8798.pt
logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/policy.onnx
```

关键结果：

```text
Final reward: 37.84
Tail100 reward: 38.24
Tail100 episode length: 964.75
Tail100 terrain level: 0.148
Tail100 track linear velocity: 0.288
Tail100 track angular velocity: 0.860
Tail100 fell_over: 0.0004
Tail100 illegal_contact: 0.116
Tail100 slip velocity: 0.092
Tail100 mean_action_acc: 0.949
Best reward: 44.39 around iteration 4914
```

分段观察：

```text
4800-4999: reward 41.64, terrain 1.41, linear tracking 0.695
5000-5199: reward 31.01, terrain 1.48, linear tracking 0.319
5800-5999: reward 37.31, terrain 0.05, linear tracking 0.257
7800-7999: reward 38.75, terrain 0.13, linear tracking 0.279
8599-8798: reward 38.39, terrain 0.15, linear tracking 0.288
```

评价：5000 PPO iterations 附近发生明显退化。原因很可能是 `command_vel` curriculum 在 `5000 * 24` steps 后把速度范围从温和范围扩到更大范围，导致策略从“尝试跟踪速度和爬地形”退化为“保守存活、少动、在低 terrain level 附近打转”。最后的 `model_8798.pt` 不推荐作为 rough 最佳模型；若要回放这一条训练线，优先看 `model_4900.pt` 附近。

## 回放命令

当前推荐回放最新 Flat baseline：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt \
  --num-envs 1
```

如果 native viewer 有问题：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt \
  --num-envs 1 \
  --viewer viser
```

当前推荐 rough 回放检查点：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/model_4900.pt \
  --num-envs 1 \
  --viewer viser
```

多个机器人同时展示：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/model_4900.pt \
  --num-envs 4 \
  --viewer viser
```

## 重要理解

- `.pt` 文件是训练 checkpoint，可用于回放、继续训练和调试。
- `policy.onnx` 是从当前 actor policy 导出的部署推理文件，不包含完整训练状态、critic 或 optimizer。
- `play.py` 回放的是训练后的当前策略，不是训练过程录像。
- `Unitree-Go2-Flat` 是平地速度跟踪任务；`Unitree-Go2-Rough` 是复杂地形任务。
- Go2 回放时运动轨迹看起来随机，是因为 `twist` 速度命令会随机采样；它不是在跟踪固定路线。
- Rough 的 `terrain_levels` 不是高度米数，而是地形难度等级。
- Rough 使用 height scan/raycast 地形高度观测，不是相机视觉。
- 当前训练动作为 12 维关节位置目标，底层是 MuJoCo 内置 position actuator/PD，而不是直接 torque policy。
- 当前 rough 最后退化不是关节映射已经明显错了；更像是课程设计导致策略选择保守存活。

## Rough 训练诊断与下一步路线

外部论文和开源项目常见路线不是“直接把 flat 换成 rough 然后一直训”，而是：

1. Flat prior / warmstart：先学会平地基础 gait，再迁移到 rough。
2. Game-inspired terrain curriculum：像游戏闯关一样，走得好升难度，走不好降难度。
3. Velocity curriculum：速度范围不要突然扩张，应随能力逐步扩大。
4. Asymmetric PPO / teacher-student：actor 只看可部署观测，critic 或 teacher 训练时可看 privileged terrain/dynamics 信息。

对当前项目，最优先处理：

1. 调整 `command_vel` 速度课程。
   - 现状：`src/tasks/velocity/velocity_env_cfg.py` 里 `command_vel` 在 `5000 * 24` 后扩到 `lin_vel_x=(-1.0, 2.0)`、`lin_vel_y=(-1.0, 1.0)`。
   - 问题：实际日志显示 5000 iter 附近 linear tracking 和 terrain level 崩掉。
   - 建议：先延后扩速，或拆成更平滑的阶段；早期只训练低速前进和小角速度。

2. 做 rough-compatible flat prior。
   - 现状：普通 Flat 删除了 `height_scan`，actor 约 47 维；Rough actor 约 234 维，不能直接 resume 普通 Flat checkpoint。
   - 建议：新建一个“平地但保留 rough 观测结构”的训练阶段，先在 plane terrain 上训练，再切到 rough。这样 checkpoint 结构兼容，可以 warmstart。

3. 重新思考 actor/critic 观测边界。
   - 现状：Rough actor 和 critic 都有 height scan，部署时如果真机没有对应地形估计，会产生落差。
   - 建议：后续更接近 sim2real 的路线是 asymmetric PPO：actor 用 IMU、关节、上一帧动作、命令等可部署观测；critic 在训练时用 height scan、接触、摩擦、质量扰动等 privileged 信息。

4. 检查 `foot_clearance` 在 rough terrain 上是否合理。
   - 当前实现可能使用 foot world z 与固定目标高度比较。
   - 在台阶/高低地形上，更合理的是相对地面高度或局部 terrain height，否则可能给出误导性奖励。

建议优先级：

```text
第一步：改速度课程，保留 rough 原观测结构，先让低速前进 rough 学稳。
第二步：做 rough-compatible flat prior，再从该 prior warmstart rough。
第三步：再考虑 asymmetric PPO / teacher-student / deployable actor 观测。
```

### Go2 Rough low-speed curriculum v1

2026-07-07 已实现第一步最小改动实验：只改 Go2 Rough 的课程，不动 reward、观测结构、网络结构和其它机器人配置。

代码位置：

```text
src/tasks/velocity/config/go2/env_cfgs.py
```

改动内容：

```text
max_init_terrain_level: 5 -> 2
rel_heading_envs: 1.0 -> 0.0
initial lin_vel_x: (0.0, 0.8)
initial lin_vel_y: (-0.2, 0.2)
initial ang_vel_z: (-0.5, 0.5)
8000 iter 后扩到: lin_vel_x=(-0.2, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.7, 0.7)
12000 iter 后扩到: lin_vel_x=(-0.5, 1.2), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8)
```

注意：`Unitree-Go2-Flat` 因为继承自 rough cfg，已经额外恢复为原来的 Flat 速度课程，避免被这次 rough v1 实验误伤。

验证：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=128 \
  --agent.max-iterations=2 \
  --agent.run-name go2_rough_low_speed_curriculum_smoke \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-07_10-01-31_go2_rough_low_speed_curriculum_smoke
```

结果：smoke 训练跑通，Rough actor/critic 维度保持 `234/261`，说明观测结构未变。

建议正式训练命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_rough_low_speed_curriculum_v1_1024env_3000iter \
  --agent.logger tensorboard
```

本轮重点观察：

```text
Curriculum/terrain_levels 是否能稳定上升
Episode_Reward/track_linear_velocity 是否能维持在 0.6 以上
Episode_Termination/fell_over 是否接近 0
Episode_Termination/illegal_contact 是否比前几次更低
Metrics/slip_velocity_mean 与 Episode_Metrics/mean_action_acc 是否不要明显恶化
```

正式训练结果：

```text
logs/rsl_rl/go2_velocity/2026-07-07_10-11-16_go2_rough_low_speed_curriculum_v1_1024env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-07_10-11-16_go2_rough_low_speed_curriculum_v1_1024env_3000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 40.71
Train/mean_episode_length: 926.86
Curriculum/terrain_levels: 2.82
Episode_Reward/track_linear_velocity: 0.740
Episode_Reward/track_angular_velocity: 0.806
Episode_Termination/fell_over: 0.015
Episode_Termination/illegal_contact: 0.226
Metrics/slip_velocity_mean: 0.103
Episode_Metrics/mean_action_acc: 1.054
```

对比原始 rough：

```text
原始 rough 3000 iter tail100: reward 39.74, terrain 1.11, linear tracking 0.667, illegal 0.185
低速 v1 3000 iter tail100: reward 40.71, terrain 2.82, linear tracking 0.740, illegal 0.226
```

结论：v1 方向有效。它明显提升了 terrain level 和线速度跟踪，说明“降低速度课程难度、限制随机 heading”确实让 Go2 更愿意往前走 rough 地形。代价是 illegal contact 和 fell_over 略高，说明它更敢走之后，身体/非足端接触还需要进一步压住。

建议回放 checkpoint：

```text
model_2000.pt: reward/illegal contact 比较好，适合看稳定性
model_2999.pt: terrain level 更高，适合看最终策略
```

### Go2 Rough low-speed curriculum v1 resume / plus 3000 iter

在用户回放 `model_2000.pt` / `model_2999.pt` 后，肉眼观察“看着还行”，因此继续沿 v1 路线巩固低速 rough，不急着扩速。

命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-07_10-11-16_go2_rough_low_speed_curriculum_v1_1024env_3000iter \
  --agent.load-checkpoint model_2999.pt \
  --agent.max-iterations=3000 \
  --agent.run-name go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/model_5998.pt
logs/rsl_rl/go2_velocity/2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 43.76
Train/mean_episode_length: 942.84
Curriculum/terrain_levels: 2.97
Episode_Reward/track_linear_velocity: 0.765
Episode_Reward/track_angular_velocity: 0.839
Episode_Termination/fell_over: 0.002
Episode_Termination/illegal_contact: 0.182
Metrics/slip_velocity_mean: 0.094
Episode_Metrics/mean_action_acc: 0.982
```

对比上一段 v1：

```text
v1 0-2999 tail100: reward 40.71, terrain 2.82, linear tracking 0.740, illegal 0.226, fell_over 0.015
v1 2999-5998 tail100: reward 43.76, terrain 2.97, linear tracking 0.765, illegal 0.182, fell_over 0.002
```

结论：继续低速 v1 是有效的。terrain level 没有大幅继续冲高，但保持在约 3；线速度跟踪、跌倒率、illegal contact、slip 和动作平滑度都有改善。当前策略比第一段 v1 更稳。

建议回放 checkpoint：

```text
model_5800.pt: 近邻窗口 reward 高、illegal contact 低，适合看稳定表现
model_5998.pt: 最终 checkpoint，适合代表当前训练结果
```

### Go2 Rough forward curriculum v2

用户判断：v1 resume 已经接近 terrain level 3，但还没有明显突破 3，说明低速课程有效，但继续原样硬训可能遇到新瓶颈。下一步应调整方法，集中训练“向前通过 rough 地形”。

参考思路：

```text
legged_gym / Learning to Walk in Minutes: 使用地形课程和命令课程，让机器人逐步进入更难 terrain。
Go2 rough 开源路线: 常见做法是先用更可学的 locomotion prior / focused curriculum，再进入更复杂 rough 策略。
```

改动位置：

```text
src/tasks/velocity/config/go2/env_cfgs.py
```

v2 改动：

```text
max_init_terrain_level: 保持 2
rel_standing_envs: 0.05 -> 0.02
rel_heading_envs: 保持 0.0
initial lin_vel_x: (0.2, 0.8)
initial lin_vel_y: (-0.1, 0.1)
initial ang_vel_z: (-0.3, 0.3)
10000 iter 后扩到: lin_vel_x=(-0.2, 1.0), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5)
14000 iter 后扩到: lin_vel_x=(-0.5, 1.2), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.7, 0.7)
```

设计目的：

```text
少站立
少横移
少随机转向
更多练向前通过 rough
尝试让 terrain level 从约 3 突破到 3.2/3.5 以上
```

注意：由于 `Unitree-Go2-Flat` 是从 rough cfg 派生出来再删地形，因此 Flat 分支中已显式恢复原来的 `rel_standing_envs=0.05`、`rel_heading_envs=1.0` 和原速度课程，避免被 v2 误伤。

验证：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=128 \
  --agent.max-iterations=2 \
  --agent.run-name go2_rough_forward_curriculum_v2_smoke \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-09_10-20-43_go2_rough_forward_curriculum_v2_smoke
```

结果：smoke 训练跑通，Rough actor/critic 维度保持 `234/261`，说明网络输入未变。

建议正式训练命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter \
  --agent.load-checkpoint model_5998.pt \
  --agent.max-iterations=2000 \
  --agent.run-name go2_rough_forward_curriculum_v2_resume5998_plus2000iter \
  --agent.logger tensorboard
```

本轮重点观察：

```text
Curriculum/terrain_levels 是否突破 3.2/3.5
Episode_Reward/track_linear_velocity 是否保持 0.75 左右或更高
Episode_Termination/illegal_contact 是否不反弹
Episode_Termination/fell_over 是否保持接近 0
Metrics/slip_velocity_mean 与 Episode_Metrics/mean_action_acc 是否保持稳定
```

正式训练结果：

```text
logs/rsl_rl/go2_velocity/2026-07-09_10-32-58_go2_rough_forward_curriculum_v2_resume5998_plus2000iter/model_7997.pt
logs/rsl_rl/go2_velocity/2026-07-09_10-32-58_go2_rough_forward_curriculum_v2_resume5998_plus2000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 35.30
Train/mean_episode_length: 890.51
Curriculum/terrain_levels: 4.76
Episode_Reward/track_linear_velocity: 0.644
Episode_Reward/track_angular_velocity: 0.778
Episode_Termination/fell_over: 0.0025
Episode_Termination/illegal_contact: 0.304
Metrics/slip_velocity_mean: 0.110
Episode_Metrics/mean_action_acc: 1.103
```

对比 v1 resume：

```text
v1 resume tail100: reward 43.76, terrain 2.97, linear tracking 0.765, illegal 0.182, slip 0.094
v2 forward tail100: reward 35.30, terrain 4.76, linear tracking 0.644, illegal 0.304, slip 0.110
```

结论：v2 成功突破 terrain level 3，说明“更强制向前通过地形”的课程确实能把 terrain curriculum 推上去；但它不是更好的最终策略。代价是速度跟踪下降、非足端接触增加、打滑和动作加速度变差。当前 v2 更像是“高地形探索策略”，不是展示或部署候选。

建议回放 checkpoint：

```text
model_7000.pt: illegal contact 相对低一些，适合看 v2 中段表现
model_7997.pt: 最终 checkpoint，适合看高 terrain level 策略
```

下一步判断：

```text
如果目标是展示当前最好效果：优先使用 v1 resume 的 model_5800.pt 或 model_5998.pt。
如果目标是继续研究突破 terrain：从 v2 学到的经验是，terrain 能上去，但需要 v3 加强接触/抬脚/稳定性约束。
```

### Go2 Rough contact-clean curriculum v3

用户确认继续做 v3。v3 的目标不是继续硬冲 terrain level，而是在 v2 已经能到高地形的基础上，让动作更干净：

```text
保留 v2 的高地形/前进通过能力
降低非足端接触
稍微鼓励更高脚 clearance
增强姿态和动作平滑约束
```

改动位置：

```text
src/tasks/velocity/config/go2/env_cfgs.py
```

v3 改动：

```text
initial lin_vel_x: (0.2, 0.8) -> (0.15, 0.8)
initial lin_vel_y: 保持 (-0.1, 0.1)
initial ang_vel_z: 保持 (-0.3, 0.3)
body_orientation_l2 weight: -1.0 -> -1.2
action_rate_l2 weight: -0.05 -> -0.07
foot_clearance weight: -1.0 -> -1.2
foot_clearance target_height: 0.10 -> 0.12
新增 nonfoot_contact reward: weight=-2.0, force_threshold=5.0
```

`nonfoot_contact` 使用已有的 `mdp.self_collision_cost` 函数，但传入的是 `nonfoot_ground_touch` 传感器；实际作用是惩罚身体/大腿/小腿等非足端碰地。Flat 分支已恢复默认 reward 权重并移除 `nonfoot_contact`，避免被 rough v3 误伤。

验证：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=128 \
  --agent.max-iterations=2 \
  --agent.run-name go2_rough_contact_clean_v3_smoke \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-09_13-04-58_go2_rough_contact_clean_v3_smoke
```

结果：smoke 训练跑通，Rough actor/critic 维度保持 `234/261`；RewardManager 中出现 `nonfoot_contact`，说明 v3 接触惩罚已生效。

建议正式训练命令：

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-09_10-32-58_go2_rough_forward_curriculum_v2_resume5998_plus2000iter \
  --agent.load-checkpoint model_7997.pt \
  --agent.max-iterations=2000 \
  --agent.run-name go2_rough_contact_clean_v3_resume7997_plus2000iter \
  --agent.logger tensorboard
```

本轮重点观察：

```text
Curriculum/terrain_levels 是否仍能保持 > 4
Episode_Termination/illegal_contact 是否从 0.304 降到 0.20 附近或以下
Episode_Reward/track_linear_velocity 是否从 0.644 回升到 0.70 左右
Metrics/slip_velocity_mean 是否从 0.110 降回 0.10 附近
Episode_Metrics/mean_action_acc 是否不要继续升高
```

正式训练结果：

```text
logs/rsl_rl/go2_velocity/2026-07-09_14-36-38_go2_rough_contact_clean_v3_resume7997_plus2000iter/model_9996.pt
logs/rsl_rl/go2_velocity/2026-07-09_14-36-38_go2_rough_contact_clean_v3_resume7997_plus2000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 39.45
Train/mean_episode_length: 918.85
Curriculum/terrain_levels: 4.36
Episode_Reward/track_linear_velocity: 0.667
Episode_Reward/track_angular_velocity: 0.826
Episode_Termination/fell_over: 0.0004
Episode_Termination/illegal_contact: 0.204
Metrics/slip_velocity_mean: 0.086
Episode_Metrics/mean_action_acc: 0.881
```

对比 v2：

```text
v2 forward tail100: reward 35.30, terrain 4.76, linear tracking 0.644, illegal 0.304, slip 0.110, action_acc 1.103
v3 contact tail100: reward 39.45, terrain 4.36, linear tracking 0.667, illegal 0.204, slip 0.086, action_acc 0.881
```

结论：v3 有用，但不是肉眼质变。它明显降低了 illegal contact、slip 和动作加速度，并让 reward 回升；代价是 terrain level 从 v2 的约 4.76 降到约 4.36，线速度跟踪只小幅改善，仍没有回到 v1 resume 的 0.765。v3 适合继续作为“高 terrain、更干净”的研究分支，但继续硬训 v3 预计边际收益不高。

建议优先回放 checkpoint：

```text
model_9000.pt: 窗口表现较平衡，reward 40.43, terrain 4.45, linear tracking 0.680, illegal 0.188
model_9400.pt: 中后段稳定性较好
model_9996.pt: 最终 checkpoint
```

### Go2 Flat-RoughObs prior

用户确认按下一阶段方案调整，并要求记录每一步计划。当前新增任务：

```text
Unitree-Go2-Flat-RoughObs
```

设计目的：

```text
训练一个“平地但保留 rough 观测结构”的 gait prior
actor/critic 输入维度保持和 Unitree-Go2-Rough 一致
后续可从该 prior checkpoint warmstart rough，避免普通 Flat actor 约 47 维、Rough actor 约 234 维不兼容的问题
```

代码改动：

```text
src/tasks/velocity/config/go2/env_cfgs.py
  新增 unitree_go2_flat_rough_obs_env_cfg()
src/tasks/velocity/config/go2/__init__.py
  注册 Unitree-Go2-Flat-RoughObs
```

配置要点：

```text
terrain_type="plane"
terrain_generator=None
保留 terrain_scan sensor
保留 actor/critic 的 height_scan
保留 feet_ground_contact / nonfoot_ground_touch
移除 terrain_levels curriculum
保留 command_vel curriculum
初始速度课程: lin_vel_x=(0.0, 0.9), lin_vel_y=(-0.25, 0.25), ang_vel_z=(-0.5, 0.5)
5000 iter 后扩到: lin_vel_x=(-0.3, 1.2), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8)
```

验证：

```bash
python scripts/list_envs.py
python scripts/train.py Unitree-Go2-Flat-RoughObs \
  --env.scene.num-envs=128 \
  --agent.max-iterations=2 \
  --agent.run-name go2_flat_roughobs_prior_smoke \
  --agent.logger tensorboard
```

结果：

```text
logs/rsl_rl/go2_velocity/2026-07-09_15-52-41_go2_flat_roughobs_prior_smoke
actor shape: 234
critic shape: 261
```

结论：任务注册成功，plane + height_scan 组合能训练，actor/critic 形状与 Rough 一致。smoke 初期 illegal_contact 高是随机初始策略现象，不作为性能判断。

建议正式训练命令：

```bash
python scripts/train.py Unitree-Go2-Flat-RoughObs \
  --env.scene.num-envs=2048 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_flat_roughobs_prior_2048env_3000iter \
  --agent.logger tensorboard
```

本轮重点观察：

```text
Train/mean_reward 是否接近或超过普通 Flat 早期水平
Episode_Reward/track_linear_velocity 是否稳步上升
Episode_Termination/fell_over 是否接近 0
Episode_Termination/illegal_contact 是否明显下降
Metrics/slip_velocity_mean 是否接近 Flat baseline
Episode_Metrics/mean_action_acc 是否不要明显高于 v1/v3 rough
```

正式训练结果：

```text
logs/rsl_rl/go2_velocity/2026-07-09_16-04-12_go2_flat_roughobs_prior_2048env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-09_16-04-12_go2_flat_roughobs_prior_2048env_3000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 55.96
Train/mean_episode_length: 997.37
Episode_Reward/track_linear_velocity: 0.884
Episode_Reward/track_angular_velocity: 0.946
Episode_Termination/fell_over: 0.0075
Episode_Termination/illegal_contact: 0.0079
Metrics/slip_velocity_mean: 0.057
Episode_Metrics/mean_action_acc: 0.563
Policy/mean_std: 0.287
Perf/total_fps: 35509
```

对比普通 Flat 2048 baseline：

```text
普通 Flat tail100: reward 54.76, linear tracking 0.836, angular tracking 0.933, slip 0.071, action_acc 0.671
Flat-RoughObs prior tail100: reward 55.96, linear tracking 0.884, angular tracking 0.946, slip 0.057, action_acc 0.563
```

结论：`Unitree-Go2-Flat-RoughObs` 训练成功，而且不仅 shape 兼容 Rough，平地指标也优于当前普通 Flat baseline。它是当前最合适的 rough warmstart prior。

### Go2 Rough from Flat-RoughObs prior / 2048 envs

从 `Unitree-Go2-Flat-RoughObs` 的 `model_2999.pt` warmstart 到 `Unitree-Go2-Rough`，环境数使用 2048。

```bash
python scripts/train.py Unitree-Go2-Rough \
  --env.scene.num-envs=2048 \
  --agent.resume=True \
  --agent.load-run 2026-07-09_16-04-12_go2_flat_roughobs_prior_2048env_3000iter \
  --agent.load-checkpoint model_2999.pt \
  --agent.max-iterations=3000 \
  --agent.run-name go2_rough_from_flat_roughobs_prior_2048env_plus3000iter \
  --agent.logger tensorboard
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-10_11-56-36_go2_rough_from_flat_roughobs_prior_2048env_plus3000iter/model_5998.pt
logs/rsl_rl/go2_velocity/2026-07-10_11-56-36_go2_rough_from_flat_roughobs_prior_2048env_plus3000iter/policy.onnx
```

Tail100 结果：

```text
Train/mean_reward: 39.39
Train/mean_episode_length: 918.23
Curriculum/terrain_levels: 4.37
Episode_Reward/track_linear_velocity: 0.666
Episode_Reward/track_angular_velocity: 0.822
Episode_Termination/fell_over: 0.0096
Episode_Termination/illegal_contact: 0.317
Metrics/slip_velocity_mean: 0.0866
Episode_Metrics/mean_action_acc: 0.874
Policy/mean_std: 0.453
```

对比：

```text
v1 resume tail100: reward 43.76, terrain 2.97, linear 0.765, illegal 0.182, slip 0.094, action_acc 0.982
v3 contact tail100: reward 39.45, terrain 4.36, linear 0.667, illegal 0.204, slip 0.086, action_acc 0.881
prior warmstart tail100: reward 39.39, terrain 4.37, linear 0.666, illegal 0.317, slip 0.087, action_acc 0.874
```

结论：prior warmstart 能直接把 terrain level 推到 v3 同档，并保持较好的 slip 和动作平滑度，但没有带来预期的 contact 改善；`illegal_contact` 明显高于 v3。它不是新的最佳 rough 策略，更像是 v3 同类的高地形分支。继续沿这条线硬训的优先级不高。

中间 checkpoint 观察：

```text
model_5600.pt: reward 39.59, terrain 4.47, linear 0.685, illegal 0.270, slip 0.088, action_acc 0.892
model_5998.pt: reward 39.49, terrain 4.36, linear 0.668, illegal 0.349, slip 0.086, action_acc 0.861
```

建议回放优先级：

```text
model_5600.pt: 这轮中相对最平衡，适合先看肉眼效果
model_5998.pt: 最终 checkpoint，适合和 v3 final 对比
v1 model_5998.pt: 仍是低 terrain 下最稳的展示候选
v3 model_9000.pt / model_9996.pt: 仍是高 terrain、较干净接触的主要候选
```

### Go2 Rough V4: terrain-relative clearance / graded contact

本轮新增独立任务：

```text
Unitree-Go2-Rough-V4
```

V4 从现有 v3 配置派生，不覆盖 `Unitree-Go2-Rough`，以便继续复现已有
checkpoint。actor/critic 观测维度保持 `234/261`，可直接 warmstart v3 或
Flat-RoughObs checkpoint。

实现内容：

```text
foot_clearance:
  改为脚端世界高度减去脚下最近有效 terrain-scan 点高度
  仅使用水平距离 0.2 m 内的有效地形点

nonfoot_contact:
  5 N 以下不惩罚
  超过 5 N 后按接触力连续增加，force_scale=20 N
  每个仿真子步只取所有非足端 geom 中的最大接触力
  单子步 cost 截断到 2.0，reward weight=-1.5

illegal_contact termination:
  force_threshold=35 N
  必须连续至少 2 个仿真子步超过阈值
```

排查相对地形高度时发现：原 `terrain_scan.include_geom_groups=(0, 1, 2)`，
而 Go2 visual mesh 使用 group 2。脚附近的 ray 会命中机器人脚部 visual mesh，
实测得到约 `-0.221 m` 的错误 clearance。V4 将 terrain scan 限制为 group 0；
修正后相同初始状态的脚端中心相对地形高度为约 `+0.047 m`，最近采样点水平
距离约 `0.0165 m`。旧任务保持原配置，以免破坏既有实验的可复现性。

同时恢复了曾被注册但在当前工作区中缺失的
`unitree_go2_flat_rough_obs_env_cfg()`；恢复内容来自已完成训练 run 保存的 git
diff。修复前 `scripts/list_envs.py` 会因 import error 退出。

验证：

```text
python scripts/list_envs.py: Unitree-Go2-Rough-V4 和 Flat-RoughObs 均注册成功
数值回归: terrain-relative clearance、graded force cost、连续接触终止通过
随机策略 smoke: 128 env / 2 iter 通过
v3 model_9996 warmstart smoke: 128 env / 2 iter 通过
actor shape: 234
critic shape: 261
```

有效 warmstart smoke run：

```text
logs/rsl_rl/go2_velocity/2026-07-10_14-48-57_go2_rough_v4_from_v3_9996_smoke
```

该 smoke 只验证接线和 checkpoint 兼容性，窗口过短，不能用于判断 V4 性能。
建议下一段正式实验从 contact-clean v3 final 开始：

```bash
python scripts/train.py Unitree-Go2-Rough-V4 \
  --env.scene.num-envs=1024 \
  --agent.resume=True \
  --agent.load-run 2026-07-09_14-36-38_go2_rough_contact_clean_v3_resume7997_plus2000iter \
  --agent.load-checkpoint model_9996.pt \
  --agent.max-iterations=2000 \
  --agent.run-name go2_rough_v4_relative_clearance_contact_from_v3_9996_plus2000iter \
  --agent.logger tensorboard
```

重点比较 v3 tail100 的 terrain `4.36`、linear `0.667`、illegal contact
`0.204`、slip `0.086` 和 action acceleration `0.881`，并新增观察
`Metrics/nonfoot_contact_force_mean`。由于 V4 同时修正了 height scan 内容，加载
v3 observation normalizer 后前期可能有短暂适应过程，不应只看最初几十次迭代。

正式训练已于 2026-07-13 完成，实际使用 2048 env：

```text
logs/rsl_rl/go2_velocity/2026-07-13_11-14-09_go2_rough_v4_relative_clearance_contact_2048env_plus2000iter/model_11995.pt
训练耗时约 1.47 小时
```

V4 tail100：

```text
Train/mean_reward: 40.61
Train/mean_episode_length: 938.18
Curriculum/terrain_levels: 4.15
Episode_Reward/track_linear_velocity: 0.731
Episode_Reward/track_angular_velocity: 0.819
Episode_Termination/fell_over: 0.0079
Episode_Termination/illegal_contact: 0.301
Metrics/slip_velocity_mean: 0.100
Episode_Metrics/mean_action_acc: 0.971
Metrics/nonfoot_contact_force_mean: 0.0287
```

与 v3 tail100 对比：

```text
v3: terrain 4.36, linear 0.667, angular 0.826, fell 0.0004, illegal 0.204, slip 0.086, action_acc 0.881
v4: terrain 4.15, linear 0.731, angular 0.819, fell 0.0079, illegal 0.301, slip 0.100, action_acc 0.971
```

V4 的 reward 定义与 v3 不同，因此 mean reward 不能直接横向比较。illegal contact
定义也不同：v3 是任一子步超过 10 N，V4 是连续两个子步超过 35 N。V4 使用了
明显更宽松的终止条件，数值却从 `0.204` 上升到 `0.301`，说明高 terrain 下的
严重、持续非足端接触实际增加了，不是单纯统计口径造成的表面变化。

分段结果：

```text
model_10200 附近 tail100: terrain 2.31, linear 0.770, illegal 0.197, slip 0.0865, action_acc 0.834
model_10800 附近 tail100: terrain 3.87, linear 0.736, illegal 0.294, slip 0.100, action_acc 0.962
model_11995 tail100: terrain 4.15, linear 0.731, illegal 0.301, slip 0.100, action_acc 0.971
```

结论：V4 明显改善了线速度跟踪，但没有成为新的综合 best。随着 terrain level
上升，illegal contact、slip 和动作粗糙度同步恶化，最终 terrain 还略低于 v3。
这说明修正 height scan 和相对地形 clearance 是正确的基础修复，但当前 contact
reward/termination 组合没有让策略在高难度地形上学会更干净的跨越动作。高 terrain
展示仍优先使用 v3；V4 回放主要看 `model_10200.pt` 和 `model_11995.pt`，分别代表
低中 terrain 的平滑快速策略和最终高 terrain 策略。不建议原样继续 V4 硬训。

### Go2 Rough V5 设计：按身体部位拆分接触

V4 结果说明统一的 `nonfoot_contact` 仍然不能区分“机身砸地”和“小腿擦台阶”。
因此 V5 不再使用单一 `nonfoot_ground_touch`，改为三个接触传感器：

```text
base_ground_contact:
  base1/base2/base3 collision geoms
  强惩罚，20 N 且连续 2 个子步终止

upper_leg_ground_contact:
  所有 hip/thigh collision geoms
  中等偏强惩罚，35 N 且连续 2 个子步终止

calf_ground_contact:
  所有 calf1/calf2 collision geoms
  较弱连续惩罚，60 N 且连续 3 个子步才终止
```

这样设计的理由：base/hip/thigh 触地通常表示姿态或落差处理失败；calf 在 rough
台阶边缘出现短暂擦碰是可恢复事件，不应与机身撞地同样处理。每类 reward 都记录
三项独立指标：`contact_rate`、`force_mean` 和 `force_when_active`，避免当前 V4
总平均力包含大量零值、难以解释的问题。

V5 还把 terrain-relative foot clearance 限制到摆动脚（`feet_ground_contact`
显示离地）上，支撑脚不再被要求达到摆动高度。V4 已修正的 terrain group 0 scan
保持不变，actor/critic 维度仍为 `234/261`。

探针训练计划：从 V4 的 `model_10200.pt` 开始，用 2048 env 运行 500 iter：

```bash
python scripts/train.py Unitree-Go2-Rough-V5 \
  --env.scene.num-envs=2048 \
  --agent.resume=True \
  --agent.load-run 2026-07-13_11-14-09_go2_rough_v4_relative_clearance_contact_2048env_plus2000iter \
  --agent.load-checkpoint model_10200.pt \
  --agent.max-iterations=500 \
  --agent.run-name go2_rough_v5_bodypart_contact_probe_2048env_500iter \
  --agent.logger tensorboard
```

判断标准不是只看 mean reward：若 terrain level 持续上升，同时 linear tracking
保持约 `0.72` 以上、slip 不超过 `0.09`、action acceleration 不超过 `0.90`，
且 base/upper-leg 的 contact rate 或 force_when_active 下降，则继续扩展到 2000
iter；若接触分部指标没有改善，停止训练并重新调整部位阈值，不再盲目加 iteration。

V5 smoke 和正式探针均已完成：

```text
smoke:
logs/rsl_rl/go2_velocity/2026-07-13_13-16-45_go2_rough_v5_bodypart_contact_smoke

2048 env / 500 iter probe:
logs/rsl_rl/go2_velocity/2026-07-13_13-17-21_go2_rough_v5_bodypart_contact_probe_2048env_500iter/model_10699.pt
训练耗时约 0.376 小时（22.6 分钟）
```

V5 tail100：

```text
Train/mean_reward: 44.53
Train/mean_episode_length: 951.54
Curriculum/terrain_levels: 3.558
Episode_Reward/track_linear_velocity: 0.758
Episode_Reward/track_angular_velocity: 0.847
Episode_Termination/fell_over: 0.0142
Episode_Termination/illegal_base_contact: 0.0142
Episode_Termination/illegal_upper_leg_contact: 0.0158
Episode_Termination/illegal_calf_contact: 0.1817
Metrics/slip_velocity_mean: 0.0950
Episode_Metrics/mean_action_acc: 0.884
```

分部接触 tail100：

```text
base:  threshold-exceedance rate 0.000006, force_when_active 12.0 N
upper: threshold-exceedance rate 0.000008, force_when_active 7.6 N
calf:  threshold-exceedance rate 0.000480, force_when_active 50.0 N
```

与 V4 在相近 terrain 阶段比较：

```text
V4 model_10600 窗口: terrain 3.51, linear 0.734, illegal 0.300, slip 0.101, action_acc 0.971
V5 model_10699 窗口: terrain 3.56, linear 0.758, 分部 illegal 合计约 0.212, slip 0.095, action_acc 0.884
```

两个版本的 termination 定义不同，因此 illegal 数值不是严格等口径；但 V5 在几乎
相同 terrain 下同时改善了 linear、slip 和 action acceleration，说明按部位拆分
接触和摆动脚 clearance 是有效方向。V5 没有完全达到预设继续条件：linear 和
action 达标，terrain 正常上升，但 slip `0.095` 高于目标 `0.09`；同时 calf
termination 随 terrain 上升并占绝大多数，base/upper-leg 已经很少。

因此按预先约定停止在 500 iter，不自动追加 1500 iter。下一步先回放
`model_10400.pt`（其前一完整窗口 terrain 2.31、linear 0.792、slip 0.0837、action 0.787）和
`model_10699.pt`，确认 calf contact 是可恢复的台阶擦碰还是影响稳定性的撞击。
若主要是可恢复擦碰，V5.1 可适度放宽 calf termination，同时保留 calf soft
penalty；若是明显失稳，则不放宽终止，改进摆动轨迹或 clearance 目标。

### V5 calf contact 自动事件诊断

为避免完全依赖肉眼回放，新增：

```text
scripts/diagnose_calf_contacts.py
```

诊断使用 512 个并行环境、固定 `0.6 m/s` 前进指令、相同 seed 和相同 play
terrain。关闭内置 termination 后，每个环境记录第一次超过 10 N 的 calf contact，
继续模拟 0.5 s，并统计：触发 calf geom、峰值力、持续子步、接触前/后的机身倾角、
速度误差增量和随后是否达到 70° fall angle。每个环境只取第一次事件，避免不同
checkpoint 因 reset 时刻不同而进一步分叉。

同时精确模拟 V5 当前 `60 N + 连续 3 子步` 的 calf termination，并单独分析
“事件发生前机身倾角不超过 35°”的稳定起始子集，避免把已经失稳后才出现的 calf
接触误判为跌倒原因。完整 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-13_13-17-21_go2_rough_v5_bodypart_contact_probe_2048env_500iter/calf_diagnostics_512env_1500steps.json
```

结果：

```text
model_10400:
  completed events 146, coverage 28.5%
  stable-at-onset events 125
  stable + current termination trigger events 51
  其中 recoverable 92.2%, destabilizing 5.9%, fall 2.0%

model_10699:
  completed events 188, coverage 36.7%
  stable-at-onset events 153
  stable + current termination trigger events 50
  其中 recoverable 66.0%, destabilizing 16.0%, fall 18.0%
```

这说明完全删除 calf termination 不安全，但当前无条件 `60 N × 3 substeps`
会截断大量本来可以恢复的事件，尤其是较早的 `model_10400`。进一步比较触发瞬间
姿态门控：对稳定起始事件，只有在当前 force/duration 条件满足且机身倾角超过
15° 时终止：

```text
model_10400: terminated 2，bad-event precision 50%，recoverable false positive 1
model_10699: terminated 10，bad-event precision 100%，recoverable false positive 0
```

25° 门控虽然在本批样本中 precision 也是 100%，但只保留 2 个终止事件，bad-event
recall 太低；35°/45° 几乎不再触发。因此 V5.1 推荐保留 calf soft penalty，并把
calf termination 改为复合条件：`>60 N` 连续 3 个仿真子步，且触发瞬间机身倾角
`>15°`。该结果来自单一 seed、固定 0.6 m/s 和一套 play terrain，正式训练前应先
实现 V5.1 并用短探针验证，不能把这次诊断当作跨所有速度/地形的最终定论。

### Go2 Rough V5.1 训练计划

新增独立任务 `Unitree-Go2-Rough-V5.1`，V5 保持不变。V5.1 只修改 calf
termination：保留 `60 N × 连续 3 子步`，并增加触发瞬间机身倾角必须超过 15°。
base/upper-leg termination、三类 soft contact penalty、terrain-relative swing-foot
clearance 和观测维度均不变。

从 V5 `model_10699.pt` 运行 2048 env / 500 iter 探针：

```bash
python scripts/train.py Unitree-Go2-Rough-V5.1 \
  --env.scene.num-envs=2048 \
  --agent.resume=True \
  --agent.load-run 2026-07-13_13-17-21_go2_rough_v5_bodypart_contact_probe_2048env_500iter \
  --agent.load-checkpoint model_10699.pt \
  --agent.max-iterations=500 \
  --agent.run-name go2_rough_v5_1_orientation_gated_calf_probe_2048env_500iter \
  --agent.logger tensorboard
```

继续标准：terrain 目标超过 `3.8`，linear tracking 至少 `0.74`，slip 不超过
`0.095`，action acceleration 不超过 `0.90`，calf termination 明显下降且
fell-over 不明显增加。未达到则停在 500 iter，不自动追加训练。

### Go2 Rough V5.1 训练结果

2048 env / 500 iter 探针已完成，耗时约 22 分 10 秒：

```text
logs/rsl_rl/go2_velocity/2026-07-13_15-06-34_go2_rough_v5_1_orientation_gated_calf_probe_2048env_500iter/model_11198.pt
```

V5.1 tail100：

```text
Train/mean_reward: 48.921
Train/mean_episode_length: 976.36
Curriculum/terrain_levels: 3.766
Episode_Reward/track_linear_velocity: 0.799
Episode_Reward/track_angular_velocity: 0.888
Metrics/slip_velocity_mean: 0.0845
Episode_Metrics/mean_action_acc: 0.774
Episode_Termination/fell_over: 0.0333
Episode_Termination/illegal_base_contact: 0.0229
Episode_Termination/illegal_upper_leg_contact: 0.0308
Episode_Termination/illegal_calf_contact: 0.0367
```

tail100 的 calf threshold-exceedance rate 为 `0.001539`，active force 为
`64.1 N`。与 V5 相比，calf 接触事件更常被策略经历并恢复，但真正触发 calf
termination 的比例从 `0.1817` 降到 `0.0367`，约下降 80%。linear、angular、
slip 和 action acceleration 也都优于 V5 tail100。

为了排除 terrain 难度差异，另取 V5.1 中 terrain 均值最接近 V5 tail100
`3.558` 的连续 100 iter 窗口（`11037..11136`）：

```text
terrain 3.560, reward 49.34, linear 0.801, angular 0.892
slip 0.0838, action_acc 0.765
fell_over 0.0167, base termination 0.0275
upper-leg termination 0.0333, calf termination 0.0354
```

该窗口的 fell-over 与 V5 的 `0.0142` 接近，说明姿态门控没有在相同难度下明显
增加直接跌倒；V5.1 尾段 `0.0333` 的 fell-over 增长主要出现在 curriculum 推进
到更难地形后。最后 50 iter 的 terrain 均值为 `3.842`，最终点为 `3.922`，但
最后 50 iter 的 fell-over 也升到 `0.0458`。

结论：`15°` 姿态门控的机制验证通过，V5.1 是当前 V5 分支中更好的综合策略。
不过 tail100 terrain `3.766` 略低于预设 `3.8`，且高 terrain 尾段跌倒率仍在
上升，因此本次按计划停在 500 iter，不自动追加 1500 iter。若继续，应从
`model_11198.pt` 只增加 500 iter，并持续观察 rolling-100 terrain、fell-over、
base/upper-leg termination；如果 terrain 不再上升或 fell-over 持续高于 `0.04`，
停止续训并调整高难 terrain 课程，而不是继续放宽 calf 条件。

### Go2 Rough V5.1 受控 +500 与 curriculum resume 限制

按上述计划，在完全冻结 V5.1 配置的前提下，从 `model_11198.pt` 继续运行 2048
env / 500 iter，耗时约 22 分 17 秒：

```text
logs/rsl_rl/go2_velocity/2026-07-13_15-36-37_go2_rough_v5_1_orientation_gated_calf_controlled_2048env_plus500iter/model_11697.pt
```

本次 tail100：

```text
Train/mean_reward: 48.094
Train/mean_episode_length: 981.09
Curriculum/terrain_levels: 3.788
Episode_Reward/track_linear_velocity: 0.795
Episode_Reward/track_angular_velocity: 0.887
Metrics/slip_velocity_mean: 0.0858
Episode_Metrics/mean_action_acc: 0.817
Episode_Termination/fell_over: 0.0075
Episode_Termination/illegal_base_contact: 0.0146
Episode_Termination/illegal_upper_leg_contact: 0.0329
Episode_Termination/illegal_calf_contact: 0.0346
```

与上一轮 V5.1 tail100（terrain `3.766`）几乎等难度，可直接比较：fell-over 从
`0.0333` 降到 `0.0075`，base termination 从 `0.0229` 降到 `0.0146`，calf
termination 基本持平；linear/ angular 基本持平，slip 从 `0.0845` 略升到
`0.0858`，action acceleration 从 `0.774` 升到 `0.817`。续训主要改善了稳定性，
没有明显提高跟踪或地形能力，且动作平滑度有所回退，已出现平台期信号。

本次还确认了一个此前评估需要修正的限制：训练 checkpoint 会恢复 policy、critic、
optimizer 等学习状态，但不会恢复各并行环境的 terrain level。新进程首个记录点
terrain 为 `1.969`，随后最低到 `1.563`，再自动爬升到最终点 `3.910`；它没有从
上一进程最终的 `3.922` 接着训练。因此：

1. 单次 run 内 terrain level 确实由表现自动控制：走过半张地形升级，未达到指令
   距离则降级；不是人工逐级切换。
2. V1 到 V5.1 的 reward、termination 和课程范围修改属于人工训练问题设计。
3. 多次短 resume 会反复经历低 level，不能等同于一段不中断的长训练；前 300 多
   iter 主要是在重新爬已经掌握的课程。

因此停止继续追加 PPO。若目标是验证真正的自主持续学习，下一步应先解决 terrain
curriculum 跨进程持久化，或直接做一次不中断长训练；随后冻结奖励和终止条件，使用
多个 seed 对照。当前 checkpoint 选择：优先稳定性用 `model_11697.pt`，优先动作
平滑度用 `model_11198.pt`。

### Go2 Rough V6：可恢复 curriculum 与固定评估

新增独立任务 `Unitree-Go2-Rough-V6`，V5.1 保持不变。V6 不修改网络、观测、
reward 或 termination，只修改训练分布和训练基础设施：

1. `VelocityOnPolicyRunner` 在 checkpoint 中保存每个环境的 `terrain_levels`、
   `terrain_types` 和 `common_step_counter`。相同环境数恢复时同步 env origin，并
   平移机器人到恢复后的 terrain patch；actor-only play/evaluation 不恢复训练状态。
2. 新增 `scripts/evaluate_go2_rough.py`：固定 seed、速度、terrain level/column，
   输出按 level、column 和 terrain type 聚合的 JSON 指标。
3. terrain 配比为 15% flat、30% 上下楼梯、20% 上下坡、15% random rough、
   20% heightfield discrete obstacles。rough 高度 `1–6 cm`，离散障碍 `2–10 cm`，
   坡度范围 `0–0.4`。
4. 增加 base payload `-1..+3 kg` 和 actuator effort capacity `0.9..1.1` 随机化；
   push 从每 `5–6 s` 的 3D/旋转扰动改成每 `10–15 s` 的水平 `±0.5 m/s`。
5. V6 初次从旧 checkpoint 加载时没有可恢复的 terrain 数组，因此
   `max_init_terrain_level=7`；V6 产生的后续 checkpoint 均可精确恢复课程分布。

curriculum persistence 往返 smoke：32 env checkpoint 保存的平均 level 为 `4.469`，
再次启动后打印并恢复为 `4.469`。2048 env 性能测试中，primitive boxes/stepping
stones 只有约 `1760 FPS`，因此放弃该 MuJoCo Warp 下的高开销表达；改为 heightfield
discrete obstacles 后恢复到约 `17.5–18k FPS`。

固定评估基线使用 V5.1 `model_11697.pt`、320 env、levels `3/5/7/9`、固定
`0.6 m/s`、1000 steps：

```text
overall: linear error 0.146, yaw error 0.056, slip 0.049, action_acc 0.100
overall term/env: fell 0.0063, base 0.0031, upper 0.0031, calf 0.0250
up stairs: linear error 0.238, calf term/env 0.0208
down stairs: linear error 0.193, calf term/env 0.1250
down slope: linear error 0.153, fell/env 0.0625
random rough: linear error 0.135, no failure termination
discrete obstacles: linear error 0.101, no failure termination
```

完整基线：

```text
logs/rsl_rl/go2_velocity/2026-07-13_15-36-37_go2_rough_v5_1_orientation_gated_calf_controlled_2048env_plus500iter/v6_hfield_fixed_eval_baseline_320env_1000steps.json
```

正式训练已从 V5.1 `model_11697.pt` 启动，2048 env / 2000 iter，不中途修改配置：

```text
logs/rsl_rl/go2_velocity/2026-07-13_16-56-07_go2_rough_v6_curriculum_persistent_hfield_dr_2048env_2000iter
```

用户下班前要求在阶段点停止，因此训练在完整的 `model_13000.pt` checkpoint 后安全
终止，实际完成约 1303/2000 iter，耗时约 57 分钟。checkpoint 内保存的 2048 个
terrain level 平均为 `5.306`，后续可直接恢复，不会重新从低 level 开始。

`model_13000.pt` 前一完整 tail100：

```text
reward 50.884, episode length 990.01, terrain 5.326
linear 0.837, angular 0.911, slip 0.0775, action_acc 0.741
fell_over 0.0079, base term 0.0058, upper term 0.0204, calf term 0.0271
```

使用与训练前完全相同的 320 env / levels `3/5/7/9` / 0.6 m/s / 1000 steps
矩阵复评：

```text
overall linear error: 0.146 -> 0.127
overall yaw error: 0.056 -> 0.053
overall slip: 0.0492 -> 0.0470
overall action_acc: 0.100 -> 0.104
overall fell/env: 0.0063 -> 0.0031
overall base term/env: 0.0031 -> 0.0000
overall upper term/env: 0.0031 -> 0.0094
overall calf term/env: 0.0250 -> 0.0031

up stairs linear error: 0.238 -> 0.142, calf term/env: 0.0208 -> 0
down stairs linear error: 0.193 -> 0.123, calf term/env: 0.125 -> 0
down slope linear error: 0.153 -> 0.121, fell/env: 0.0625 -> 0.0312
flat linear error: 0.092 -> 0.116
random rough linear error: 0.135 -> 0.147
discrete obstacles linear error: 0.101 -> 0.124
```

阶段结论：V6 显著改善楼梯、下坡和总体接触稳定性，代价是 flat、random rough、
discrete obstacles 的线速度误差小幅增加，以及 overall action acceleration 从
`0.100` 升到 `0.104`。`model_13000.pt` 是目前更适合复杂地形的 checkpoint，但
不是所有地形上的全面支配策略。完整复评：

```text
logs/rsl_rl/go2_velocity/2026-07-13_16-56-07_go2_rough_v6_curriculum_persistent_hfield_dr_2048env_2000iter/fixed_eval_model_13000_320env_1000steps.json
```

原计划剩余 697 iter 已于 2026-07-14 完成。续训时日志明确显示
`Restored terrain curriculum: mean level 5.306`，并从 iteration `13000/13697`
开始，证明没有重置课程。续训耗时约 30 分 18 秒，最终 checkpoint 为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/model_13696.pt
```

`model_13696.pt` checkpoint 内 2048 个 terrain level 平均为 `5.240`。最终
tail100：

```text
reward 49.473, terrain 5.240, linear 0.829, angular 0.900
slip 0.0806, action_acc 0.788, fell_over 0.0058
base term 0.0075, upper term 0.0138, calf term 0.0458
```

与 `model_13000.pt` 的训练 tail100 相比，最终模型的跟踪、动作平滑度和
calf termination 有所回退。固定 seed 42 评估也显示 `model_13696.pt`
的 overall linear error 从 `0.127` 升到 `0.134`，697 iter 的最终点不是
最佳 checkpoint。

为避免只比较起点和终点，对 `model_13100.pt` 到 `model_13600.pt`
的每个 100-iter checkpoint 都运行了相同的 320-env 固定矩阵。
`model_13500.pt` 是 seed 42 上的最佳综合点：

```text
model_13000: linear 0.1268, yaw 0.0528, slip 0.0470, action_acc 0.1041
model_13500: linear 0.1058, yaw 0.0474, slip 0.0485, action_acc 0.1011
model_13696: linear 0.1342, yaw 0.0546, slip 0.0459, action_acc 0.0978

model_13500 term/env: fell 0.0031, base 0, upper 0, calf 0
```

再用 seed `42/43/44` 对 `model_13000.pt` 和 `model_13500.pt` 做三组复评，
每组均为 320 env / levels `3/5/7/9` / 0.6 m/s / 1000 steps。三 seed
平均：

```text
                         model_13000  model_13500
linear error                0.1270       0.1062
yaw error                   0.0527       0.0474
slip                        0.0469       0.0485
action_acc                  0.1043       0.1015
fell/env                    0.0063       0.0010
base term/env               0.0000       0.0000
upper term/env              0.0083       0.0010
calf term/env               0.0063       0.0052
```

960 个环境合计，跌倒从 `6` 次降到 `1` 次，upper-leg termination 从
`8` 次降到 `1` 次，calf termination 从 `6` 次降到 `5` 次。linear
error 平均改善约 16.4%，各 terrain type 的三 seed 平均 linear error 也全部
优于 `model_13000.pt`。代价是 slip 从 `0.0469` 升到 `0.0485`，约增加
3.5%。

结论：V6 推荐 checkpoint 为 `model_13500.pt`。`model_13696.pt` 作为完整
训练终点保留，但不作为当前最佳策略。继续原样追加 PPO 已出现跟踪回退，因此在此
停止 V6 续训。评估 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/fixed_eval_intermediate_13100_to_13600_320env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/fixed_eval_model_13696_320env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/fixed_eval_finalists_seed43_320env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/fixed_eval_finalists_seed44_320env_1000steps.json
```

### Go2 Rough V6：多命令鲁棒性矩阵

固定 `0.6 m/s` 的 clean 评估只能证明单一工况。为检查命令泛化和训练随机化下的
表现，扩展 `scripts/evaluate_go2_rough.py`：

1. `--command-cases` 可在同一批环境中并行评估 `forward_0.3/0.6/0.9`、
   `lateral_left/right` 和 `yaw_left/right`。
2. `--profile clean` 关闭观测噪声、startup randomization 和 push；
   `--profile dynamics` 只保留物理随机化；`--profile randomized` 使用训练时的
   观测噪声、摩擦、encoder bias、COM、payload、电机强度以及 10–15 s 水平 push。
3. JSON 新增按 command、command × level、command × terrain type 的交叉汇总；
   旧的 `--command-x/y/yaw` 单命令接口保持兼容。

正式矩阵比较 `model_13000.pt` 和 `model_13500.pt`，profile 为 clean/randomized，
seed 为 `42/43/44`。每次 checkpoint 评估包含：

```text
7 commands × 4 levels × 20 terrain columns × 2 repeats = 1120 env
1000 steps = 20 s，randomized profile 中每个环境至少经历一次 interval push
总计 2 checkpoints × 2 profiles × 3 seeds × 1120 = 13440 env instances
```

三 seed 的全命令平均如下。termination 是 3360 个环境中的 term flag 总数；同一
episode 可能同时满足多个 term，因此这些计数不能直接相加为唯一失败数。

```text
profile     model   linear    yaw     slip   action_acc   fell base upper calf
clean       13000   0.1215  0.0568  0.0366    0.0797       5    1     9    9
clean       13500   0.1101  0.0508  0.0367    0.0784       5    0     6   19
randomized  13000   0.1411  0.0755  0.0459    0.2373       9    5    22   27
randomized  13500   0.1373  0.0758  0.0458    0.2390       4    3    19   32
```

`model_13500.pt` 从 clean 到 randomized 的 linear error 增加 24.7%，yaw error
增加 49.1%，slip 增加 24.9%，action acceleration 约变为 3.05 倍。后者包含策略
响应 push 的动作变化，不能与无 push 的 clean 数值等价解释。randomized 下与
`model_13000.pt` 相比，`model_13500.pt` 的 linear error 仍低 2.7%、跌倒和
upper-leg termination 更少，但 yaw/action_acc 基本相同，calf termination 从
`27` 增至 `32`。

`model_13500.pt` 的分命令结果：

```text
command          clean primary error   randomized primary error   randomized terms
forward_0.3             0.0816                  0.1096             0/0/1/12
forward_0.6             0.1060                  0.1365             1/2/3/11
forward_0.9             0.1355                  0.1752             2/1/13/8
lateral_left            0.2009                  0.2113             0/0/2/0
lateral_right           0.1974                  0.2142             1/0/0/1
yaw_left                0.0527                  0.0893             0/0/0/0
yaw_right               0.0469                  0.0767             0/0/0/0
termination order: fell/base/upper/calf
```

主要薄弱项：

1. 横移是最明显的能力缺口。目标为 `±0.3 m/s`，randomized linear error 仍为
   `0.211–0.214 m/s`；策略大体保持稳定，但没有充分执行横向指令。当前 checkpoint
   尚未进入 `±0.3 m/s` lateral curriculum，且 uniform command 很少产生纯横移。
2. `0.9 m/s` 的高难 inverted slope 是主要严重失稳组合：randomized 三 seed 中
   primary error `0.224`，出现 `1/0/9/3` 个 fell/base/upper/calf term flag。
   上楼梯在 `0.3/0.6 m/s` 下分别出现 `11/7` 个 calf term flag。
3. 失败几乎集中在 level 9。`model_13500.pt` randomized 的 level 9 有
   `4/2/14/27` 个 fell/base/upper/calf flag；level 3/5 基本稳定。
4. random rough 的横移误差最高，左右分别约 `0.243/0.242`，但很少终止，属于
   “走不准而不是直接摔倒”。

结论：`model_13500.pt` 仍作为默认综合 checkpoint；如果只重视 `0.9 m/s` 高速
复杂地形，`model_13000.pt` 是更保守的备选。下一步建立独立 V7，V6 保持不变：

1. 从 `model_13500.pt` warm start，先冻结 reward 和 termination。
2. command sampler 显式分配纯前进、纯横移、纯 yaw 和高速前进模式，而不是只用
   连续 uniform box；横移从 `±0.1` 逐步扩到 `±0.3 m/s`。
3. 增加 `0.8–1.0 m/s` 与 level 7–9 上楼梯、下坡组合的采样，但不整体抬高所有
   terrain 难度。
4. 保留 V6 的 dynamics randomization 和 push，先跑 500 iter 探针。继续标准：
   randomized lateral error 至少降低 15%，`0.9 m/s` inverted-slope upper-leg
   term 明显下降，且 overall fell/calf 不高于 V6 基线。
5. 只有在命令/地形分布修正后 randomized 与 clean 的差距仍很大，再引入
   asymmetric adaptation / RMA，避免同时改变数据分布和网络后无法归因。

完整结果：

```text
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_clean_seed42_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_clean_seed43_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_clean_seed44_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_randomized_seed42_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_randomized_seed43_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/robustness_randomized_seed44_1120env_1000steps.json
```

### Go2 Rough V7：显式命令模式 500 iter 探针

V7 保持 V6 的 terrain、reward、termination、观测维度和 dynamics randomization
不变，只把 Uniform command 替换为显式模式采样，并移除旧的 uniform
command_vel curriculum：

~~~text
general forward: 40%
pure lateral: 25%, |v_y| = 0.1..0.3 m/s
pure yaw: 15%, |w_z| = 0.2..0.7 rad/s
high-speed forward: 20%, v_x = 0.8..1.0 m/s
~~~

当 terrain level >= 7 且 terrain 为 pyramid_stairs 或 hf_pyramid_slope_inv
时，high-speed mode 概率提升到 45%。512-env 分布 smoke 实测普通/focus terrain
的 mode 比例分别为：

~~~text
普通: 40.1% / 25.0% / 15.0% / 19.8%
focus: 27.7% / 17.3% / 10.3% / 44.7%
~~~

横移模式的 x/yaw 指令严格为零，yaw 模式的线速度严格为零。任务注册为
Unitree-Go2-Rough-V7，actor/critic shape 仍为 234/261。

从 V6 model_13500.pt warm start，2048 env / 500 iter 已完成，耗时约 22 分
28 秒：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13999.pt
~~~

最终 checkpoint 的 terrain level 平均为 5.648。训练 tail100：

~~~text
reward 50.071, terrain 5.650, linear 0.829, angular 0.901
slip 0.0795, action_acc 0.766, fell_over 0.0054
base term 0.0071, upper term 0.0142, calf term 0.0554
mode general/lateral/yaw/high-speed/standing:
0.386 / 0.230 / 0.143 / 0.224 / 0.018
~~~

对 V7 model_13500.pt 和阶段点 model_13600.pt 做了 clean/randomized、
seed 42/43/44 的相同 1120-env 矩阵复评。三 seed 全命令平均如下，termination
是 3360 个环境中的 term flag 总数：

~~~text
profile     model   linear    yaw     slip   action_acc   fell/base/upper/calf
clean       13500   0.1159  0.0555  0.0375    0.0799       1/1/26/38
clean       13600   0.1173  0.0500  0.0347    0.0783       2/1/6/10
randomized  13500   0.1410  0.0813  0.0465    0.2424       4/1/58/53
randomized  13600   0.1392  0.0754  0.0440    0.2411       0/5/19/24
~~~

model_13600.pt 的稳定性改善明确：randomized 下 fell 4 -> 0、upper
termination 58 -> 19、calf termination 53 -> 24，yaw/slip/action_acc
也改善。主要命令结果：

~~~text
randomized primary error       model_13500  model_13600
forward_0.3 (linear)              0.1177       0.1191
forward_0.6 (linear)              0.1492       0.1415
forward_0.9 (linear)              0.1845       0.1776
lateral_left (linear)             0.2039       0.2138
lateral_right (linear)            0.2145       0.2098
yaw_left (yaw)                    0.0832       0.0845
yaw_right (yaw)                   0.0836       0.0839
~~~

V7 没有达到原定横移目标：左右横移平均约 0.212 m/s，相比 baseline 没有
改善 15%。0.9 m/s + hf_pyramid_slope_inv 的 error 也从 0.2104 变为
0.2156，但该组合的 upper/calf term flag 从 5/3 降为 2/4；上楼梯
0.6 m/s 的 error 从 0.2056 降到 0.1513，calf flag 从 11 降到 2。

阶段结论：V7 证明显式模式采样能明显降低复杂地形接触失稳，但单纯增加横移
样本仍不足以学会横向跟踪。推荐保留 model_13600.pt 作为随机扰动下更稳的
V7 候选，同时保留 V6 model_13500.pt 作为固定 forward benchmark。停止继续
追加当前 V7 配置；下一轮应把 lateral mode 提高到约 40–50%，做相对
±0.1 -> ±0.3 m/s 的阶段 curriculum，再重新评估，暂不同时修改 reward 或
引入 RMA。

V7 评估结果：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_stage_randomized_seed42_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_finalists_randomized_seed43_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_finalists_randomized_seed44_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_finalists_clean_seed42_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_finalists_clean_seed43_1120env_1000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/robustness_finalists_clean_seed44_1120env_1000steps.json
~~~

### Go2 Rough V7.1：lateral-heavy 两阶段探针

V7.1 从 V7 model_13600.pt warm start，继续冻结 reward、termination、terrain
和 dynamics randomization。只修改 command 分布：

~~~text
general / lateral / yaw / high-speed = 30% / 45% / 10% / 15%
前 250 iter lateral |v_y| = 0.10..0.20 m/s
后 250 iter lateral |v_y| = 0.10..0.30 m/s
~~~

model_13600.pt 保存的 common_step_counter 为 326664，阶段切换使用相对偏移
6000 steps，即 250 × 24。smoke 验证 offset 5999 仍使用 0.20 上限，offset
6000 精确切换到 0.30。普通 terrain 的实测 mode 比例为
29.9% / 45.2% / 9.9% / 14.9%，focus terrain 的 high-speed 比例为 44.6%。

2048 env / 500 iter 正式训练已完成，耗时约 22 分钟：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter/model_14099.pt
~~~

最终 checkpoint terrain level 平均为 4.803。tail100：

~~~text
reward 52.621, terrain 4.798, linear 0.844, angular 0.918
slip 0.0708, action_acc 0.722
fell 0.0058, base 0.0046, upper 0.0192, calf 0.0271
mode general/lateral/yaw/high-speed/standing:
0.288 / 0.438 / 0.096 / 0.168 / 0.018
~~~

训练指标改善，但 terrain 从 warm-start checkpoint 的 5.275 降到 4.80，因此
不能只按 reward 选模型。使用 randomized seed42 的 1120-env 矩阵筛选
model_13700 到 model_14099，横移最好的点是最终 model_14099：

~~~text
                           model_13600  model_14099
overall linear                0.1382       0.1408
overall yaw                   0.0749       0.0744
slip                          0.0442       0.0451
action_acc                    0.2403       0.2371
fell/base/upper/calf          0/4/8/6      3/2/5/9
forward_0.3 error             0.1167       0.1328
forward_0.6 error             0.1408       0.1533
forward_0.9 error             0.1739       0.1795
lateral_left error            0.2134       0.2039
lateral_right error           0.2108       0.2039
~~~

左右横移平均从 0.2121 降到 0.2039，只改善约 3.9%，没有达到 15% gate。
forward_0.6 退化约 8.9%，超过允许的 5%；fell 从 0 增至 3，calf flag 从
6 增至 9，terrain 也低于 5.5。多个停止条件同时失败，因此不再运行 seed43/44，
也不继续追加 V7.1。

横移按地形拆分后呈现小幅但一致的学习：

~~~text
flat: 约改善 5%
hf_pyramid_slope: 约改善 9–11%
random rough: 约改善 4–5%
stairs / discrete obstacles: 多数改善 2–5%
inverted slope: 左侧约改善 1%，右侧退化约 2%
各 level: 约改善 3–6%
~~~

结论：增加 lateral 样本有边际效果，但收益不足，并开始牺牲 forward 和 terrain
能力。默认模型仍为 V7 model_13600.pt，拒绝 V7.1 model_14099.pt。下一步不应
继续提高 lateral 概率，而应先扩展诊断指标：

1. 将线速度误差拆成 x/y 分量和 lateral response gain，判断是单纯欠跟踪还是
   出现错误方向运动。
2. 在 flat、stairs、inverted slope 上记录 lateral 模式的足端轨迹、接触相位和
   12 关节动作。
3. 检查固定 trot gait reward 是否阻碍横向步态；确认后再做 command-conditioned
   gait/foot-placement 修改。
4. 在诊断前不修改 reward，不引入 RMA，也不继续追加 PPO。

完整 seed42 阶段结果：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter/robustness_stage_randomized_seed42_1120env_1000steps.json
~~~

### 固定 terrain 重定位修复与 lateral gait 诊断

在实现横移足端诊断时发现固定评估器存在 terrain assignment 缺陷：脚本更新了
terrain_levels、terrain_types 和 env_origins，但没有把已经初始化的 robot root
平移到新 patch。最小验证显示 env_origins 改变后 root position 的变化为零。

修复方式与 VelocityOnPolicyRunner 的 curriculum restore 相同：

1. 保存旧 env origin 和 root pose。
2. 设置目标 level/type/origin。
3. 将 root position 平移 new_origin - old_origin。
4. 写回仿真并 forward/sense。
5. 验证 root 相对 patch 的偏移保持不变；正式结果误差约 1.2e-7 m。

影响说明：旧 JSON 的 overall 同模型相对比较仍有参考价值，因为模型经历相同的
初始化分布；但旧 by_level/by_terrain_type 标签与实际 patch 不一致，之前所有
按楼梯、坡道等类型作出的定量归因应视为已废弃。评估器现在输出
terrain_assignment_position_error_max，并新增：

~~~text
x/y absolute velocity error
actual vx/vy
linear command response gain
cross-axis velocity
~~~

使用修复后的 randomized seed42 矩阵重跑三个关键模型：

~~~text
model       linear   yaw     slip   action_acc   fell/base/upper/calf
13500       0.1398  0.0815  0.0466    0.2425       1/1/12/15
13600       0.1393  0.0755  0.0443    0.2412       0/1/4/6
14099       0.1410  0.0743  0.0451    0.2371       0/1/1/9
~~~

修复后仍支持原结论：V7 model_13600.pt 是默认稳定模型，V7.1 model_14099.pt
横移小幅改善但 overall/forward 回退，不采用。修正结果：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter/robustness_key_models_relocated_randomized_seed42_1120env_1000steps.json
~~~

新增 scripts/diagnose_go2_lateral.py，使用正确重定位后的 clean 固定 patch，
对 model_13600.pt 和 model_14099.pt 运行：

~~~text
3 commands × 2 levels × 3 terrain types × 8 repeats = 144 env
100 warmup steps + 800 sample steps
terrain: flat / pyramid_stairs / hf_pyramid_slope_inv
~~~

诊断输出速度分量、response gain、方向正确率、cross-axis drift、固定 trot
匹配率、四足接触 pattern、12-bin 足端相位轨迹、12 关节动作和 termination。
完整结果：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter/lateral_gait_diagnostics_clean_seed42_144env_900steps.json
~~~

V7 model_13600.pt 的核心证据：

~~~text
command          response gain   direction correct   fixed-trot match
forward_0.6          0.818             99.1%              95.3%
lateral_left         0.349             95.7%              94.0%
lateral_right        0.366             94.2%              91.0%

flat lateral gain: 0.45–0.50
level-9 stairs lateral gain: 0.25–0.26
level-9 inverted-slope lateral gain: 0.28–0.30
~~~

横移不是方向控制错误，而是幅度严重不足。动作/步态证据：

~~~text
                                  forward      lateral
phase foot-path main-axis range   15–16 cm      3–4 cm
all-four-feet contact fraction       5%        13–14%
thigh action std                    0.652       0.225
calf action std                     1.195       0.75–0.79
hip action std                      0.407       0.42
~~~

横移仍主要使用正确的 diagonal trot pattern，但增加了多足同时着地，腿部屈伸
明显不足；它是在小幅摆髋并侧向挪动，不是充分侧步。V7.1 model_14099.pt 将
lateral-left gain 提到约 0.401，但 forward gain 从 0.818 降到 0.761，说明增加
横移样本只是沿同一 reward trade-off 移动，没有消除它。

根因位于 reward trade-off。pose reward 与 tracking reward 权重均为 +1，
walking hip tolerance 只有 0.15 rad。使用记录的 joint RMS 和速度误差近似重建
两项 reward：

~~~text
command          approximate pose   approximate tracking   sum
forward_0.6            0.861               0.938           1.799
lateral_left           0.949               0.854           1.802
lateral_right          0.947               0.860           1.807
~~~

策略通过减少 thigh/calf motion、保持默认姿态并欠跟踪横移，获得了与正常前进
几乎相同的 pose+tracking 总收益。foot_gait 只奖励固定接触时序，不奖励步长或
命令方向；横移已经能得到较高 gait reward，因此它强化了低幅度 shuffle，但没有
证据表明 diagonal trot 相位本身错误。

下一步应做单变量 reward probe，而不是继续改采样或立刻改 gait：

1. 从 V7 model_13600.pt warm start，恢复 V7 的 25% lateral 分布。
2. 只对 lateral-dominant command 放宽 hip pose tolerance，例如从 0.15 插值到
   0.30 rad；forward/yaw 的 pose tolerance 保持原值。
3. foot_gait、其他 reward、termination、terrain 和 randomization 全部冻结。
4. 先跑 300–500 iter，使用修复后的 evaluator 验收 response gain、forward
   regression、terrain 和 contact stability。
5. 若 hip tolerance probe 仍不能增加足端横向摆幅，再引入
   command-conditioned foot-placement；暂不改 trot phase，不引入 RMA。

### Go2 Rough V7 lateral-conditioned hip pose tolerance 单变量探针

本轮按 1 个主 Agent + 3 个子 Agent、独立 Git worktree 执行。开始前将原 dirty
工作区完整固化为 baseline commit：

```text
branch: exp/lateral-pose-integration
baseline commit: e8a7eee chore: checkpoint rough terrain v7 baseline
reward Agent commits: e7009c1, bd85bcd
analysis Agent commit: f3342b4
test Agent rebased/final integration HEAD before experiment: 5071764
```

worktree：

```text
/home/jensen/projects/worktrees/go2-reward
/home/jensen/projects/worktrees/go2-analysis
/home/jensen/projects/worktrees/go2-test
```

#### 实验目的与唯一变量

任务注册为 `Unitree-Go2-Rough-V7-LateralPose`，直接从 V7 派生。V7 的普通
terrain 模式概率保持 general/lateral/yaw/high-speed=`40/25/15/20%`，pure
lateral 仍为 `|vy|=0.1..0.3 m/s`，没有 V7.1 的 45% lateral 或 staged range。
focus terrain 原有 high-speed 重加权、terrain、随机化、termination、foot gait、
其他 reward 权重、观测、动作和 PPO 配置全部冻结。

唯一行为变化为 `variable_posture` 的 hip std。先按原逻辑由
`||command_xy|| + |wz|` 选择 standing/walking/running std，再计算：

```text
dominance_margin = |vy| - max(|vx|, yaw_scale * |wz|)
alpha = clamp(dominance_margin / full_lateral_command, 0, 1)
effective_hip_std = base_hip_std + alpha * (max_hip_std - base_hip_std)

yaw_scale = 1.0
full_lateral_command = 0.30
max_hip_std = 0.30 rad
```

只有四个 hip joint 使用插值；thigh/calf 不变。pure lateral
`0.1/0.2/0.3 m/s` 对应 walking hip std `0.20/0.25/0.30 rad`；pure forward、
pure yaw 不变，zero standing command 继续使用 standing hip std `0.05 rad`。
公式在 dominance 边界连续，固定正分母避免除零；helper 还验证 command 最后一维
严格为 3、full scale 为正、yaw scale 非负。`yaw_scale=1.0` 是显式的数值换算
系数，因为 `vy` 和 `wz` 的物理单位不同。

#### 训练前验收

Reward Agent 的 7 项 unit test 和 Test Agent 的 9 项独立 acceptance test 共 16 项
全部通过，同时通过 Python 编译、`git diff --check`、任务注册和语义配置 diff。
Test Agent 确认新任务相对 V7 恰好只增加四个 pose 参数；actor/critic shape 仍为
`234/261`。

smoke：

```text
32-env random / 2 iter:
logs/rsl_rl/go2_velocity/2026-07-14_15-52-07_go2_v7_lateral_pose_32env_random_2iter_integration_acceptance/model_1.pt

128-env V7 model_13600 warm-start / 2 iter:
logs/rsl_rl/go2_velocity/2026-07-14_15-54-53_go2_v7_lateral_pose_128env_warmstart13600_2iter_integration_acceptance/model_13601.pt
```

两次 smoke 均无 traceback、missing/unexpected key，TensorBoard 的 56 个 scalar
tags / 112 values 全部有限。128 env 因环境数与 checkpoint 的 2048 不同，按预期
只跳过 terrain 数组恢复；正式 2048-env run 随后成功将保存的 terrain mean
`5.275` 精确恢复。

#### 正式训练

实验名称和目的：V7 lateral-conditioned hip pose tolerance probe；验证放宽横向
主导命令下的 hip pose tolerance 能否消除低幅度侧挪，同时保持 forward 和 rough
terrain 能力。

唯一变量：上述 command-conditioned hip std；其余配置冻结。

```bash
python scripts/train.py Unitree-Go2-Rough-V7-LateralPose \
  --env.scene.num-envs=2048 \
  --env.seed=42 \
  --agent.seed=42 \
  --agent.resume=True \
  --agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter \
  --agent.load-checkpoint=model_13600.pt \
  --agent.max-iterations=500 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.run-name=go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter
```

输出：

```text
logs/rsl_rl/go2_velocity/2026-07-14_15-58-33_go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter
warm start: V7 model_13600.pt
num_envs: 2048
iterations: 500
seed: 42
耗时: 约 22 分钟
final checkpoint: model_14099.pt
stage checkpoints: model_13700.pt / model_13800.pt / model_13900.pt / model_14000.pt
```

正式 run tail100：

```text
Train/mean_reward: 50.912
Train/mean_episode_length: 987.681
Curriculum/terrain_levels: 5.479
Episode_Reward/track_linear_velocity: 0.8352
Episode_Reward/track_angular_velocity: 0.9086
Episode_Reward/pose: 0.8605
Metrics/slip_velocity_mean: 0.07735
Episode_Metrics/mean_action_acc: 0.74996
Episode_Termination/fell_over: 0.00667
Episode_Termination/illegal_base_contact: 0.00833
Episode_Termination/illegal_upper_leg_contact: 0.01792
Episode_Termination/illegal_calf_contact: 0.03417
mode general/lateral/yaw/high-speed/standing:
0.3882 / 0.2357 / 0.1455 / 0.2187 / 0.02135
```

训练课程没有退化：baseline checkpoint 保存 mean level `5.275`，横移最佳阶段
`model_13900.pt` 保存 `5.516`，final 保存约 `5.498`。

#### 修正 evaluator 阶段筛选

使用修复 terrain relocation 后的 evaluator，对原始 V7 baseline 和
`13700/13800/13900/14000/14099` 全部运行 randomized seed42、1000 steps、
7 commands、levels `3/5/7/9`。当前 terrain generator 有 20 columns，因此
`4 repeats` 的实际规模是：

```text
7 commands x 4 levels x 20 columns x 4 repeats = 2240 env
terrain_assignment_position_error_max = 1.19e-7 m
```

阶段总结果：

```text
logs/rsl_rl/go2_velocity/2026-07-14_15-58-33_go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter/robustness_stage_randomized_seed42_2240env_1000steps.json
```

横移最佳点为 `model_13900.pt`：

```text
metric                              model_13600   model_13900
overall linear error                   0.1390        0.1352
overall yaw error                      0.0753        0.0754
forward_0.6 gain                       0.8068        0.8054
forward_0.6 error                      0.1435        0.1440
lateral left/right gain             0.309/0.326   0.363/0.390
lateral mean gain                      0.3171        0.3766
lateral mean error                     0.2132        0.1984
slip                                   0.0434        0.0453
action acceleration                    0.2412        0.2414
fell/base/upper/calf flags            1/1/15/9      4/2/22/16
```

randomized mean lateral gain 改善约 18.8%，mean lateral error 改善约 6.9%；
forward 基本不变，overall linear error 小幅改善。代价是 slip 增加约 4.3%，
action acceleration 基本持平，四类 failure/contact flags 均有增加。

final `model_14099.pt` 的 lateral mean gain 为 `0.3451`，低于阶段最佳；尽管
overall linear error `0.1357`、slip `0.0442`、action acceleration `0.2374`，
仍不作为横移 probe 候选。

#### Lateral gait diagnostic 与最终验收

对 baseline 和 `model_13900.pt` 运行 clean、144 env、900 steps 诊断：

```text
logs/rsl_rl/go2_velocity/2026-07-14_15-58-33_go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter/lateral_gait_diagnostics_clean_seed42_144env_900steps.json
```

```text
metric                              model_13600   model_13900
forward gain                            0.8272        0.8273
lateral left/right gain             0.349/0.366   0.383/0.399
lateral mean gain                       0.3573        0.3913
forward phase foot range                15.90 cm      15.65 cm
lateral left/right phase range        3.14/3.34 cm  3.43/3.85 cm
lateral mean phase range                 3.24 cm       3.64 cm
all-four contact left/right          12.7/14.4%    10.2/13.0%
left action std hip/thigh/calf       .418/.224/.787 .401/.295/.822
right action std hip/thigh/calf      .425/.226/.746 .393/.292/.815
```

forward gain 和 forward 足端摆幅保持不变；横移 thigh/calf motion 增加，四足同时
接触比例下降，说明 tolerance probe 确实推动了更积极的横移动作。但 clean mean
lateral gain 只有 `0.391`，未达到预设 `>=0.40`；横向相位足端摆幅仍只有
`3.64 cm`，未达到 `>=5 cm`，没有形成充分横向跨步。左右 lateral diagnostic
没有 termination；forward fell fraction 从 `0.0417` 降到 0，upper-leg fraction
保持 `0.0208`，但 randomized 大矩阵 failure flags 增加。

Test/Acceptance Agent 最终判定：**FAIL**。拒绝 `model_13900.pt` 和 final
`model_14099.pt` 作为部署模型，继续默认使用：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

实验结论：command-conditioned hip tolerance 有可测的横移收益，且没有破坏
forward/terrain，但不足以把 shuffle 变成充分侧步，并带来 randomized contact/
failure 增加。下一步进入先前约定的 command-conditioned foot-placement/
step-length reward 单变量设计；不继续放宽 pose tolerance，不增加 lateral 采样，
不先改 trot phase，也不引入 RMA/深度图/Transformer。

## 下一步建议

1. 如果要展示 Flat，优先用 `2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt`。
2. 如果要展示 Rough v1，优先回放 `2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/model_5800.pt` 和 `model_5998.pt` 做肉眼对比。
3. 如果要展示 Rough 高 terrain 分支，优先对比 v3 的 `model_9000.pt`、`model_9400.pt`、`model_9996.pt`。
4. 新的 prior warmstart 2048 env 结果不优于 v3；可回放 `model_5600.pt` 做肉眼确认，但不建议继续原样硬训。
5. V4 正式训练已完成：linear tracking 提升，但 terrain、illegal contact、slip 和 action acceleration 不如 v3；不建议原样继续训练。
6. V5 的 2048 env / 500 iter 探针已完成：相同 terrain 下优于 V4，但 slip 未达到预设线，且 calf termination 占主要部分，因此已按计划停止，没有自动追加训练。
7. V5.1 受控 +500 已完成：稳定性改善，但跟踪/terrain 基本持平且动作平滑度回退，停止继续追加 PPO。
8. V6 原计划剩余 697 iter 已完成，curriculum 从平均 `5.306` 正确恢复，最终点为 `model_13696.pt`。
9. 全部阶段点固定评估和 seed `42/43/44` 复评后，V6 推荐使用 `model_13500.pt`；它相对 `model_13000.pt` 的三 seed 平均 linear error 改善约 16.4%，跌倒从 6/960 降到 1/960，但 slip 增加约 3.5%。
10. `model_13696.pt` 跟踪回退，不是新的 best；停止原样追加 V6 PPO。
11. V7.1 lateral-heavy 500 iter 已完成：横移只改善约 3.9%，forward_0.6 退化约 8.9%，terrain 降到 4.80，拒绝 model_14099 并停止续训。
12. Lateral-conditioned hip pose tolerance 单变量探针已完成：`model_13900.pt` 的横移 gain 有改善，但 clean mean 仅 `0.391`，横向足端摆幅仅 `3.64 cm`，且 randomized failure/contact flags 增加；Test Agent 判定 FAIL，继续默认使用 V7 `model_13600.pt`。
13. 下一步设计 command-conditioned foot-placement/step-length reward 单变量探针；保持 V7 25% lateral、terrain、randomization、termination 和 trot phase 不变，不继续放宽 pose tolerance。
14. 如果想控制固定速度，优先研究 `--viewer viser` 的 joystick 面板，或修改 Go2 play 模式的 command 采样逻辑。

## 2026-07-15：复杂路径第一阶段无训练 baseline

### 目标与多 Agent 集成

本阶段将优先级从横移专项改为统一 policy 的局部路径执行闭环：世界路径由局部
controller 转为 body-frame `vx/vy/yaw`，V7 policy 负责执行命令并适应地形。
本阶段不训练，不修改 reward、command 采样、terrain、termination 或网络。

使用 1 个 Integration Agent 和 3 个独立 worktree Agent：

```text
Training Design Agent: e5275a0
Scenario Agent:        1f53a52 + 8aee1bb + b969175
Acceptance Agent:      133e1a1

integration commits:
c14c18e  training design review
7ef2fb5  independent acceptance tests
cb9a9d5  parameterized route evaluator
0038222  route placement after wrapper reset
351c6ea  preserve rollout step index
```

Training Design Agent 确认 V7 actor 只观察 body-frame twist、proprioception、phase
和局部 height scan，不观察世界位置、waypoint 或累计路径误差。因此 V7 是局部速度
executor，路径跟踪必须由外部 controller 闭环完成。完整审查见：

```text
docs/reviews/complex_path_training_design.md
```

### Route evaluator

新增：

```text
scripts/evaluate_go2_routes.py
src/tasks/velocity/evaluation/routes.py
tests/test_go2_route_scenarios.py
tests/test_go2_route_acceptance.py
```

支持两种诊断模式：

- `open_loop`：固定 body-frame forward command tape，隔离 locomotion 执行能力。
- `line_follow`：由世界系 cross-track/heading error 生成 body-frame `vx/vy/yaw`，
  检查 path controller 与 locomotion 的组合闭环。

每个 env 是一次 route attempt；首次 completion 或 reset/termination 后冻结，自动
reset 的新 episode 不会继续累计前进距离。JSON 记录 completion、progress、位置/
航向误差、commanded/actual velocity、route-normal cross-axis velocity、slip、action
acceleration、reset、首次失败原因、接触终止和 terrain relocation error。

GPU smoke 发现并修复两个只在真实 rollout 暴露的问题：

1. `RslRlVecEnvWrapper` 构造时会 reset，最初在 wrapper 前设置的 route pose 被覆盖。
   现在 relocation/route placement 在 wrapper/runner 初始化后执行，并断言初始
   progress、cross-track 和 heading error。
2. rollout 循环索引 `_` 被 `step()` 返回的 extras 字典覆盖，首次 completion 时
   `steps_to_completion` 抛 TypeError。现在使用明确的 `step_index`。

最终 21 项 route tests、Python 编译、V7 task registration/import、CLI、
`git diff --check` 和短 GPU smoke 均通过。smoke 的 relocation error 为 0，route
start 与机器人实际初始位置一致，初始 heading 约 0，32 步无 reset。Acceptance
Agent 在最终 integration HEAD `351c6ea` 上复验并给出 PASS；无残留训练/评估进程。

### V7 无训练 baseline

固定 checkpoint：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
task: Unitree-Go2-Rough-V7
seed: 42
```

Flat open-loop clean，2 m，0.4 m/s，16 attempts：

```text
completion                         16/16 = 100%
mean progress ratio               1.0016
lateral RMS / max                 0.0074 / 0.0162 m
heading RMS / max                 0.54 / 0.89 deg
mean final position error         0.0079 m
forward response gain             0.771
reset / fell/base/upper/calf      0 / 0/0/0/0
slip / action acceleration        0.0225 / 0.0696
terrain relocation max error      3.81e-6 m
```

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_flat_open_loop_clean_seed42.json
```

Flat line-follow 使用 cross-track `-0.2/0/+0.2 m`、yaw `-0.2/0/+0.2 rad`，
4 levels x 4 repeats x 9 offsets = 144 attempts：

```text
profile       completion   lateral RMS   final |lateral|   heading RMS   reset
clean         144/144      0.0505 m      0.0034 m          2.30 deg      0
randomized    144/144      0.0574 m      0.0176 m          2.68 deg      0
```

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_flat_line_follow_clean_seed42.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_flat_line_follow_randomized_seed42.json
```

独立 terrain patch 零初始误差矩阵使用 7 terrain types x 4 levels x 4 repeats =
112 attempts，route 2.5 m：

```text
mode/profile             completion   reset/contact failure
open_loop clean          109/112      0
line_follow clean        112/112      0
line_follow randomized   112/112      0
```

open-loop 的 3 个 `step_limit` 都已越过终点，但横向/航向没有收敛到 completion
tolerance；line-follow 全部纠正完成。因此这三个失败属于缺少路径闭环，不是 policy
在地形上摔倒。

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_patch_matrix_open_loop_clean_seed42.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_patch_matrix_line_follow_clean_seed42.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_patch_matrix_line_follow_randomized_seed42.json
```

最高难度重点矩阵使用 levels `3/7`、4 repeats、cross-track/yaw 各
`-0.2/0/+0.2`，共 504 clean attempts：

```text
overall completion                503/504 = 99.80%
flat/slope/inv-slope/rough/obstacle/up-stage completion 100%
pyramid_stairs down-stage         71/72 = 98.61%
唯一失败                          level 7, yaw +0.2 rad, illegal_calf_contact
overall reset                     1/504
terrain relocation max error      3.81e-6 m
```

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_patch_matrix_line_follow_offsets_clean_seed42.json
```

### 结论与限制

第一阶段结论：flat 直线路径 evaluator 的坐标、body command 和 attempt 生命周期已
由测试和真实 GPU rollout 验证；V7 在现有独立 patch 的 2.5 m 直线闭环上表现良好，
没有证据支持此时修改 reward 或启动 PPO。默认部署模型继续是 `model_13600.pt`。

当前不能宣称“完整直线上/下楼梯”或“平地到楼梯/坡地到平地连续过渡”已通过：
现有 V6/V7 terrain 是相互独立的 8 m x 8 m patch，没有 inter-patch transition
geometry；楼梯 origin 位于中央平台，2.5 m outward route 只覆盖平台退出和一段台阶，
不是从外侧入口到另一侧出口的完整楼梯。JSON 已明确标记
`continuous_inter_patch_transitions=false`。

下一步应先实现/校准连续 transition terrain 和合法入口 pose（包括 terrain-aware
root z），用相同 evaluator 口径完成完整直线楼梯和 transition gate。该 gate 通过
前不增加圆弧、S 弯或启动 2048-env 训练；若届时出现明确、可复现的单一 locomotion
短板，再设计单变量 probe。

## 2026-07-15：连续地形完整直线路径 gate

### Evaluation-only geometry

继续使用同一 1+3 Agent/worktree 结构。实现只进入 evaluation 层，不注册新 task，
不修改 V7 的 reward、command sampler、reset、terrain training distribution、
termination、网络或 checkpoint：

```text
Terrain Agent:     1c65c58 + 6cda2fb + ddd3748
Scenario Agent:    e1140f9 + e8b1d94
Acceptance Agent:  ca922bf + 8beeec9

integration:
5214ef0  continuous route terrain profiles
5a66776  independent continuous acceptance
092d24f  continuous route evaluator
30b5c4a  keep terrain scan inside patch
76d20f1  ray-test scan footprint heights
440484e  enforce scan margins in acceptance
3b2b471  report scan footprint margins
```

新增 `src/tasks/velocity/evaluation/route_terrains.py`。evaluator 从 registry 加载 V7
配置深拷贝后，仅为本次评估替换 terrain generator；V7 actor/critic shape 仍为
`234/261`，`model_13600.pt` strict load。四类 geometry：

```text
stairs_up / stairs_down / slope_up / slope_down
patch: 8.0 x 4.0 m
route axis: local +x
start: x=1.0 m
feature: x=2.0..4.4 m
end: x=7.0 m
route length: 6.0 m
stairs: 8 x 0.30 m, step height 0.02..0.12 m by difficulty
slope: 0.0..0.4 gradient by difficulty
```

每条 route 包含 approach flat、完整 feature 和 exit flat。TerrainOutput origin 位于
入口 surface，wrapper reset 后按该 origin 重新 placement，保持 Go2 root clearance
`0.32 m`。自定义 stairs boxes 使 MuJoCo 初始 contact 数超过 V7 的 `nconmax=35`，
因此 continuous evaluation copy 单独将 `nconmax` 提升到 128；patch/training 配置不变。

### GPU 发现的 scan 边界缺陷

第一版使用 start `0.75`、end `7.25`、route `6.5 m`。虽然 CPU geometry/ray tests
通过，但 GPU baseline 在 slope_up level 5/7 于 progress `<=0.22 m` 提前发生
upper/base contact。open-loop 和 0.4 m/s 复验同样失败，最初看似 locomotion 短板。

进一步检查发现 terrain scan x footprint 为 `+-0.8 m`：start `0.75 m` 会让后向
ray 跨到相邻 curriculum row。相邻 profile 的入口/出口高度不同，初始观测被 patch
边界高度墙污染。该问题属于 evaluator geometry，不是模型失败。

修复后 contract 为 start `1.0`、end `7.0`、route `6.0 m`：

```text
start scan range: [0.2, 1.8] m，全部位于 entry flat
end scan range:   [6.2, 7.8] m，全部位于 exit flat
feature range:    [2.0, 4.4] m
residual patch boundary clearance: 0.2 m
```

实际 MuJoCo rays 和独立 acceptance 均验证四类 profile 的 scan footprint 高度。修复
后原来 0/8 的 slope_up level 5/7 诊断变为 8/8、零 reset/contact，确认旧失败完全
来自 scan 跨 patch。以下旧 JSON 无效，不得用于模型结论：

```text
route_baseline_continuous_line_follow_clean_seed42.json
route_baseline_continuous_slope_up_open_loop_clean_seed42.json
route_diagnostic_continuous_slope_up_v0_4_clean_seed42.json
```

### 最终 V7 baseline

共同设置：V7 `model_13600.pt`、`Unitree-Go2-Rough-V7`、seed 42、body forward
command `0.6 m/s`、最多 1000 steps、line-follow，route `6.0 m`。

Clean：4 cases x 4 levels (`0/3/5/7`) x 4 repeats = 64 attempts：

```text
case          completion   lateral RMS   heading RMS   final pos err   slip    action acc
stairs_up       16/16        0.0529 m       0.54 deg      0.0074 m     0.0430    0.1108
stairs_down     16/16        0.0221 m       1.26 deg      0.0070 m     0.0414    0.0956
slope_up        16/16        0.0255 m       0.29 deg      0.0086 m     0.0320    0.1002
slope_down      16/16        0.0094 m       1.08 deg      0.0096 m     0.0470    0.0896
overall         64/64
reset / fell/base/upper/calf: 0 / 0/0/0/0
terrain_assignment_position_error_max: 0
```

Randomized 同一 64 attempts：`64/64`，零 reset/termination；各 case final position
error 约 `0.029..0.033 m`，slip `0.040..0.050`，action acceleration
`0.247..0.265`。

参数化初始误差 clean：levels `3/7`、2 repeats、cross-track/yaw 各
`-0.2/0/+0.2`，共 144 attempts：`144/144`，零 reset/termination。各 case lateral
RMS `0.0395..0.0763 m`，heading RMS `1.72..2.46 deg`，final position error
`0.0073..0.0100 m`。

最终有效 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_continuous_line_follow_scanfixed_clean_seed42.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_continuous_line_follow_scanfixed_randomized_seed42.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_continuous_line_follow_offsets_scanfixed_clean_seed42.json
```

Test Agent 在最终 integration HEAD `3b2b471` 给出 PASS：41/41 route/terrain tests、
py_compile、CLI contract、V7 registration/import、diff-check、GPU metrics 和 process
检查全部通过；所有 worktree clean，无残留训练/评估进程。

结论：完整直线 stairs/slope approach->feature->exit gate 已通过，且没有暴露需要
训练的单一 locomotion 短板。因此不启动 2048-env PPO，继续默认部署 V7
`model_13600.pt`。Coverage 是真实的 intra-patch continuous transition，不宣称
inter-patch 世界连续性。下一阶段可按原顺序增加圆弧、S 弯、forward+yaw、
stop-and-go 和急转恢复 evaluator；先评估，再决定是否存在训练变量。

### Viser 固定地形网页演示

为解决原 `play.py` 随机采样 `vx/vy/yaw` 且随机选择 terrain 的问题，新增回放参数：

```text
--fixed-vx VX --fixed-vy VY --fixed-yaw-rate YAW_RATE
--terrain-demo default|stairs_up|stairs_down|slope_up|slope_down
--terrain-level 0..9
```

指定 terrain demo 时只修改本次 play 配置：使用 evaluation-only continuous terrain、
单环境、固定入口和 yaw、clean observation/events、`nconmax=128`，并移除
`randomize_terrain`，不会修改训练 task 或 checkpoint。V7 ModeVelocityCommand 被
锁定到 100% general mode，重采样时间设为 `1e9 s`。

GPU strict-load 50-step smoke：stairs_up level 5、command `(0.4,0,0)`，terrain
column/level/origin 保持不变，placement error `0`，初始 root clearance `0.32 m`，
无 reset，actor observation `(1,234)`，action 全部有限。4 项配置 unittest、CLI
显式速度参数解析、py_compile 和 diff-check 通过。

新增 `stairs_up_down` 网页 demo：12 m x 4 m 单 patch，start `x=1`，8 级上楼
`x=2.0..4.4`，顶部平台 `x=4.4..7.6`，8 级下楼至 `x=10`，出口平地至
`x=11`。level 5、固定 `(0.4,0,0)` 的完整 GPU rollout 在 1473 steps 到达下楼
出口，progress `9.20 m`，base relative z 从 `0.32 m` 上升到最高 `0.955 m` 后
回到 `0.319 m`；无 reset/termination，lateral final `0.097 m`。几何 ray、CLI、
配置和完整 policy rollout 均通过。

### 2026-07-15 参数化圆弧 baseline（未训练）

目的：验证统一 V7 policy 在固定半径圆弧上的 `body-frame vx, vy, yaw` 路径闭环，
并区分固定命令时序的 locomotion 执行能力与 closed-loop path controller 的补偿能力。
实现已整合到 integration 代码/测试基线 `e95a4bc`（关键实现 `b4f48c6`，acceptance
`e95a4bc`）；测试 worktree 独立提交为 `dacad5c`。唯一评估模型为 V7：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

实现契约：左曲率为正、`wz = sign * vx / radius`；command-tape 按理想时间
`ceil(length / (speed * control_dt))` 发送命令，运动段后发送零命令并等待 settle；
S 弯按 step index 切换曲率，不依赖实际位置；所有 radius/speed/sign 场景在一个
batched environment 中执行。reset 后 attempt 进度冻结，
`terrain_assignment_position_error_max` 纳入 JSON。此前继续按实际 progress 发 tape
命令的 JSON `.../route_baseline_curved_arc_command_tape_clean_seed42_72.json`
判定为无效，不得用于模型结论。

验证结果：

```text
纯几何/场景/acceptance 测试：34/34 PASS
py_compile、CLI、V7 registration/import、非法 steps、git diff --check：PASS
GPU smoke：placement error 0、reset 0、JSON finite；生命周期正确冻结为 tape_end
```

有效 clean arc command-tape 矩阵（18 场景，steps=1200）保存为
`logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_command_tape_clean_seed42_18env_1200steps.json`。
completion `0/18`，progress ratio
约 `0.812..0.922`，无 reset；其中 required yaw 在 V7 general 分布内的场景也未能
在理想时长内走完，说明固定时序下存在实际前进速度不足。该结果是诊断，不是网页
或 evaluator 生命周期错误。

同矩阵 closed-loop 保存为
`logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_closed_loop_clean_seed42_18env_1200steps.json`：
`16/18` 在 1200 steps 完成，剩余 `r=4.0,v=0.3` 两项仅因步数上限为
`step_limit`；将这两项单独增加到 1600 steps 后为 `2/2` 完成，JSON 为
`logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_closed_loop_r4v03_clean_seed42_2env_1600steps.json`。
其闭环误差约
`0.002..0.019 m` lateral RMS、`0.014..0.051 rad` heading RMS，无 reset。

结论：V7 的 closed-loop 圆弧执行在足够时长下通过，command-tape 暴露的欠速尚
不足以证明需要改 policy；当前不满足“tape 与 closed-loop 均对 ID 命令失败”的
训练授权条件。未启动 2048-env PPO，未产生新 checkpoint；继续默认使用
`model_13600.pt`，下一步先跑 S 弯和 randomized/rough 扩展，再决定是否存在单一
训练变量。

### 2026-07-16 Curve pre-training 审计（NO-GO，未训练）

目的：在实现 15% correlated forward+yaw sampler 之前，验证 terrain curriculum、
pure-axis/coupled response 和 matched control/probe 契约。integration 分支为
`exp/curve-pretrain-integration`；三个独立 Agent source commits 为 curriculum
`f431598`、command diagnostics `36f68b9`、acceptance `a9704c2`，对应 integration
commits 为 `0cd1fff`、`6671bd2`、`5b4e436`。

本轮没有修改生产 `curriculums.py`、`mode_velocity_command.py`、Go2 env config、
reward、terrain、termination、gait 或网络；没有实现 curve sampler，也没有启动 GPU
训练。

#### Terrain curriculum 审计

当前生产公式等价于：

```text
d_net = ||root_xy(T) - terrain_origin_xy||
v_last = ||command_xy(T)||
move_up = d_net > patch_length/2
move_down = d_net < v_last * episode_length * 0.5 and not move_up
```

它使用终点净位移而不是累计路程，并且降级阈值只使用 3--8 秒重采样序列中的最后
一个平移命令。独立测试确认：完整圆完美回到起点会降级；纯 yaw 失败不能降级，
但漂移超过 4 m 可升级；换向抵消和最后进入 zero-command settle 会改变判断；相同
累计路程可因路线几何得到相反 level 决策。因此保持当前 curriculum 开启并增加
curve quota 不是单变量实验。

只读检查 `model_13600.pt`：checkpoint 保存 `2048` 个 terrain levels/types，
`common_step_counter=326664`，mean level `5.275390625`，level `0..9`、type `0..19`。
未来若重新授权 matched 实验，control/probe 都必须使用 2048 env 从同一 checkpoint
恢复完全相同 assignment，然后移除 `terrain_levels` term 并在起止比较逐元素值或
hash；不得同时修改 command-aware curriculum。

详细审计：`docs/reviews/curve_curriculum_audit.md`。

#### Command response 诊断

只解析修复后的 scheduled tape 和既有有效 JSON。旧的
`route_baseline_curved_arc_command_tape_clean_seed42_72.json` 仍无效，新增脚本会拒绝
其 schema。

```text
subset   n    vx gain   wz gain   progress   reset/contact
ID       14    0.8108    0.9525     0.8558       0
OOD       4    0.8290    0.9450     0.8884       0
```

OOD 仅为 `r=1.5,v=0.5/0.6` 左右四项。clean flat pure forward `0.6 m/s` gain
为 `0.8540`，同速且 ID 的 coupled gain 为 `0.8490`，差约 `-0.6%`；补足 horizon
后 closed-loop 合并结果为 `18/18`、零 reset。证据支持通用 forward under-gain，
不支持额外 forward+yaw 耦合退化。

离线可复现输出：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/curve_command_diagnostics_offline_seed42.json
```

历史 JSON 只保存 attempt aggregate，没有逐 control-step command/response；rise time、
overshoot、settling time 当前不可恢复。pure-yaw 文件也没有 signed actual yaw mean，
因此诊断明确输出 unavailable，而没有用 absolute error 猜测 gain。详细结论见
`docs/reviews/curve_command_diagnostics.md`。

#### 验收与决策

integration 首轮统一运行 57 项测试：56 PASS、1 intentional skip。skip 对应尚未授权
实现的 production curve sampler/frozen-terrain wiring；不能标记为已经完成。其余
py_compile、V7 registration/import、CLI、真实 JSON 离线解析和 `git diff --check`
均通过。

最终决策：**NO-GO，不实现 curve sampler，不启动 matched control/probe 或 PPO**。
原因是训练 gate 要求 coupled gain 明显弱于 pure forward/pure yaw 或 closed-loop 失败，
而现有证据相反。下一步先完成 clean S 弯、randomized/rough 曲线，或增加带逐步时序
的严格 matched command-response 小诊断；只有发现额外 coupled locomotion 短板后，
才重新评估 15% curve quota。V7 `model_13600.pt` 继续作为默认模型。

### 2026-07-16 S 弯逐步瞬态与 randomized flat baseline（NO-GO，未训练）

目的：补齐逐控制步 command-response 指标，并使用 V7 验证两段反向曲率的 clean 与
randomized flat S 弯，区分普通前进欠速、yaw 换向、closed-loop controller 和
locomotion policy 问题。本轮 integration 分支为 `exp/s-curve-transient-integration`；
实现 Agent source commit 为 `d18bf26`，整合后提交为 `042ce00`，主 Agent 的 near-zero
gain 修复为 `3a68ec0`，Acceptance Agent source commit 为 `e0b4edf`（整合后
`18e7669`），Training Decision Agent source commit 为 `6148829`（整合后
`2550236`）。

唯一模型仍为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

实现新增逐 segment 的 commanded/actual `vx/vy/wz`、least-squares response gain、
IAE、90% rise time、overshoot、进入并保持 +/-10% band 的 settling time、yaw
sign-switch latency、command delta、controller saturation、slip 和 action acceleration。
S command-tape 两段按严格 step index 切换，不依赖实际 pose；reset 后 attempt 和指标
冻结。gain 分母使用 `sum(command^2)`，命令能量低于阈值时输出 JSON `null`，避免把
接近零的 `vy` 噪声误报为几十倍 gain。closed-loop 的 target 每步变化，因此其
rise/overshoot/settling 只能解释为对 instantaneous command 的响应，不能当作固定
阶跃稳态指标。

#### Clean command-tape

正式命令：

```text
python scripts/evaluate_go2_curved_routes.py \
  --checkpoint <V7 model_13600.pt> \
  --route-kind s_curve --mode command_tape \
  --radii 1.5 2.5 4.0 --speeds 0.3 0.5 0.6 --turn-signs 1 -1 \
  --steps 1600 --settle-steps 10 --profile clean --seed 42
```

JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_command_tape_clean_seed42_18env_1600steps.json
```

固定理想时长下 completion 为 `0/18`，但零 reset、零 fell/base/upper-leg/calf contact，
terrain placement error 为 `0`。它是前进欠速诊断，不作为唯一训练 gate：

```text
subset  n   progress  vx gain seg1/seg2  wz gain seg1/seg2  yaw switch
ID      14   0.8271       0.7902/0.8209       0.9408/0.9357      0.0457 s
OOD      4   0.8534       0.7790/0.8518       0.9249/0.9045      0.0400 s
```

ID slip/action acceleration/cross-axis velocity 分别为 `0.0273/0.0728/0.0651`；yaw
换向快速且方向正确，没有 S 弯特有的 command bandwidth 失败。

#### Clean closed-loop

同一 18 场景矩阵使用 `--mode closed_loop --steps 2000 --profile clean`。JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_clean_seed42_18env_2000steps.json
```

结果为 `18/18` completion、mean progress `1.0`、零 reset/termination、controller
saturation `0`、placement error `0`。lateral RMS mean `0.00659 m`、lateral max
`0.03174 m`、heading RMS mean `1.42 deg`、heading max `4.66 deg`、slip mean
`0.02858`、action acceleration mean `0.07524`。ID yaw sign-switch latency mean
`0.0314 s`。clean closed-loop gate 全部通过。

#### Randomized flat

randomized profile 保留 observation corruption、friction、encoder bias、COM/payload、
motor strength 和 push，仍使用相同 flat S 矩阵、seed 和 V7 checkpoint。正式 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_randomized_flat_seed42_18env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_r4v03_right_randomized_seed42_1env_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_command_tape_randomized_flat_seed42_18env_1600steps.json
```

closed-loop 2000-step 矩阵为 `17/18`，mean progress `0.99939`，completion `94.4%`，
零 reset/termination。唯一 `r=4.0,v=0.3,right` 达到 `98.9%` 后因 `step_limit`
结束；保持相同 seed/profile 单独延长到 2400 steps 后 `1/1` 完成，最终位置误差
`0.0256 m`，证明是 horizon false failure。原矩阵 lateral RMS mean `0.02653 m`、
lateral max `0.12909 m`、heading RMS mean `1.89 deg`、heading max `6.48 deg`，均通过
randomized gate；slip mean `0.03674`，action acceleration mean `0.23258`。

randomized command-tape 仍为固定时长 `0/18`，但零 reset/contact。ID progress
`0.8063`、两段 vx gain `0.7798/0.8038`、wz gain `0.8844/0.9413`、yaw switch
`0.0486 s`。因此随机化下仍是普通 forward under-gain，未出现明显换向带宽或左右
不对称问题。五个正式 JSON 均通过递归 finite 检查；允许的缺失瞬态值使用 JSON
`null`，没有 NaN/Inf。

randomized action acceleration 约为 clean 的 `3.1x`，但当前没有完全相同随机化条件的
straight/arc matched reference，不能把它归因于 S 弯，也不能用 clean 的 `1.2x`
阈值直接授权训练。后续应先补同 profile 的 matched route reference。

#### 训练决策与覆盖边界

Training Decision Agent 判定 **NO-GO**：clean closed-loop `18/18`，randomized 的唯一
未完成项补足 horizon 后通过；yaw 换向约 `0.03--0.05 s`，controller saturation 为
`0`，coupled/sign-switch 没有明显弱于既有 pure-axis baseline。V7 的 3--8 s 随机
重采样只会偶然覆盖 general-to-general 的反向 yaw，不精确等价于 S tape，但当前证据
仍不足以授权 15% curve mode 或 transition-sequence sampler。

本轮没有修改 reward、production sampler、terrain、termination、gait、network 或
checkpoint；没有启动 matched control/probe 或 PPO，没有新模型。现 evaluator 明确
输出 `flat_curves=true`、`rough_curves=false`、`terrain_transitions=false`，因此没有
运行或宣称 slope/rough/obstacle 曲线能力；后续必须先实现兼容 corridor、scan footprint
和 relocation 的复杂地形曲线 evaluator。

Integration HEAD 的全量 Go2 unittest 为 `132 PASS + 1 intentional skip`；skip 仍是
尚未授权的 production curve sampler/frozen-terrain wiring。相关 `py_compile`、CLI
help、V7 task registration/load、五个正式 JSON finite 检查和 `git diff --check` 均
通过。Test Agent 在包含本文档的 integration tree `54b320e` 上最终判定 **PASS**：
90 项 curved/S/transient/curriculum 测试通过、1 项同一 intentional skip；1-env/32-step
GPU smoke 的 placement/reset 均为 `0`，transient schema 和 finite 检查通过。四个本轮
worktree clean，无残留 train/evaluate/play 或 GPU compute 进程。

最终结论：**NO-GO，未启动训练；V7 `model_13600.pt` 继续作为默认模型。**

### 2026-07-16 Matched randomization 与复杂地形曲线 smoke（NO-GO，未训练）

本阶段使用 1 个 Integration Agent 和 3 个独立 Agent/worktree。Matched Reference
Agent commit `8dc0caf` 整合为 `524fc5b`；Terrain Curve Agent commit `c1579ef`
整合为 `0b8b2ac`；Acceptance commits `a7209ab/21173f7` 整合为
`2e6712e/f36b55f`。Acceptance 首轮发现并阻止了三个 matched contract 缺陷：120 度
arc corridor x 上界低估、`settle_steps` 只存在于 metadata、JSON 没有 command
energy。Integration 修复 commit 为 `b058f13`：arc bounds 使用 90 度处真实极值，
completion 后执行固定零命令 settle window，并输出每轴离散/积分 command energy。

#### 严格 matched flat 对照

straight、120 度 arc、两个反向 60 度 arc 组成的 S 使用统一路长 `2*pi*r/3`，相同
checkpoint、seed、matched slot、speed、steps、controller limits 和 profile；每个
route kind 在 fresh env 中用同 seed 重建。正式修复后 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_matched_straight_arc_s_clean_full_settlefixed_seed42_18slots_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_matched_straight_arc_s_clean_full_settlefixed_diagnostics_seed42.json
```

结果：

```text
profile          route       completion   action_acc   slip
clean            straight      18/18       0.07435    0.02833
clean            arc           18/18       0.07521    0.02875
clean            S             18/18       0.07587    0.02896
full randomized  straight      17/18       0.23202    0.03577
full randomized  arc           17/18       0.23257    0.03717
full randomized  S             17/18       0.23292    0.03723
```

randomized 三类唯一未完成项均是同一 `r=4,v=0.3,right` slot，progress 约
`0.982/0.987/0.989`、零 reset，属于共同 horizon 边界。S action acceleration 在
clean 下仅比 arc/straight 高 `0.87%/2.04%`，randomized 下仅高 `0.15%/0.39%`；
S slip randomized 比 arc 高 `0.14%`、比 straight 高 `4.08%`。所有 1.2x/1.3x、
slip 和 catastrophic termination gates 通过，明确归因 `not_s_curve_specific`。

#### Randomization 因素归因

pre-fix 但同矩阵的 ablation JSON 仅用于因素诊断；10-step settle 修复不改变主结论：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_matched_straight_arc_s_core_ablation_seed42_18slots_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_matched_straight_arc_s_observation_subfactors_seed42_18slots_2000steps.json
```

S action acceleration：clean `0.07524`，dynamics-only `0.07705` (`1.024x`)，
push-only `0.07563` (`1.005x`)，observation-only `0.22738` (`3.022x`)，
full-randomized `0.23265` (`3.092x`)。进一步拆分：actor corruption only
`0.22810`，encoder bias only `0.07622`。因此动作加速度升高几乎完全由 actor
observation corruption 主导，不是 friction/COM/payload/motor/push、encoder bias 或
S 曲线本身。该结论是评估风险归因，不授权修改训练观测或启动 PPO。

#### 复杂地形曲线 smoke

新增 evaluation-only 18 x 18 m terrain curve patch，route start `(9,9)`；精确计算
0.4 m corridor 和 yaw-aligned 1.6 x 1.0 m height-scan footprint。`r=4` S 的最小
scan margin 仍约 `1.2718 m`。旧 8 x 4 m transition patch 明确拒绝且不缩放；
continuous approach-feature-exit 和 stairs curves 未实现、未声称覆盖。Terrain 使用
V7 slope up/down、random rough、discrete obstacle primitives，low/medium 对应 level
0/1。random rough primitive 不使用 difficulty，JSON 明确标记该限制。

有效 smoke JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_arc_clean_low_medium_seed42_64env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_s_curve_clean_low_medium_seed42_64env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_s_curve_slope_up_low_r4v03_left_retry_seed42_1env_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_arc_randomized_low_medium_seed42_64env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_s_curve_randomized_low_medium_seed42_64env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_terrain_curves_s_curve_randomized_r4v03_retry_seed42_12env_2400steps.json
```

Clean arc 为 `64/64`，clean S 为 `63/64`，唯一 slow `r=4,v=0.3` 增加 horizon 后
`1/1`；randomized arc `64/64`，randomized S `59/64`，五项均为 slow `r=4,v=0.3`
step-limit，相关 2400-step 子矩阵 `12/12`。所有 rollout 零 reset/termination，terrain
assignment error `5.96e-8`、route placement error `0`。Randomized arc/S mean action
acceleration 为 `0.23888/0.23911`，再次表明曲线类型不是动作粗糙度来源。

这些结果只能标记为可信 low/medium smoke：复用 rollout 当前只保留 mean slip/action
acceleration 和 catastrophic contact terminations，没有 P95/max 或非终止 body-part
contact rate；不能标记为完整 formal complex-terrain gate，也不覆盖 high difficulty、
continuous transitions 或 stairs curves。

结论：没有发现 controller 或 locomotion policy 的单一训练短板。**NO-GO，未启动
训练；V7 `model_13600.pt` 继续作为默认模型。** 后续优先把 terrain rollout 接入
matched P95/max 和非终止 contact 聚合，再扩大 high difficulty/transition 覆盖；不修改
reward、sampler 或网络。

最终 Acceptance Agent 在 integration `f7e7ceb` 上给出 **PASS**：全量 200 tests
PASS、1 个未来 production sampler 接线测试按设计 skip；独立 matched/terrain 47/47
PASS；编译、CLI、V7 注册、8 份关键 JSON finite 检查、worktree/process/GPU 清洁检查
全部通过。Terrain coverage 仍严格保持 smoke-only 表述。

### 2026-07-16 High terrain boundary 与完整 rollout metrics（NO-GO，未训练）

本阶段从 clean baseline `5d233f4` 创建 `exp/terrain-boundary-integration`，使用 1 个
Integration Agent 和 3 个独立 worktree Agent：

```text
Terrain Metrics Agent:   7c1ec54 -> integration edd2bc9
Boundary Scenario Agent: c24f166 -> integration cbb3292
Acceptance Agent:        f0a48ff -> integration ce6632f
Integration fixes:       72eb5ff, 17663c8, d16a784, 308e974, 2a43523
```

#### 指标和 evaluator 修复

新增 `OnlineTerrainRolloutMetrics`，按每个环境原 route attempt 的 active control-step
保留 action acceleration、foot slip 和 body-part contact。终止控制步计入分母，之后
自动 reset 的新 episode 不计入；空样本/缺失传感器使用 `null + reason`，JSON 使用
`allow_nan=False`。action acceleration 保持原离散定义：

```text
mean(abs(action[t] - 2*action[t-1] + action[t-2]))
```

输出新增 action/slip mean、P95、max；base/upper-leg/calf 非终止与全部 contact
count/rate；catastrophic termination 独立统计。Straight/curve route 同时补齐
cross-track/heading P95、逐 scenario progress ratio 和 steps_sampled。

首个 GPU 回归在 JSON 拼装前 fail-fast：底层 curved scenario 缺少 `steps_sampled`，
修复为 `308e974`。Continuous 初次使用 `--steps=1800` 时又发现 straight evaluator
没有延长 20 s episode timeout；level-9 stairs_up 因内置 time_out 被误判。`2a43523`
使 effective episode length 覆盖 requested horizon，并增加 CPU contract。旧文件：

```text
route_boundary_continuous_straight_clean_levels7_9_metricsv2_seed42_12env_1800steps.json
```

只保留为 evaluator 缺陷记录，不得用于模型失败结论；正式 clean 结论使用 2400-step
完整重跑。

#### 参数化边界 geometry

High curve evaluator 使用 evaluation-only 18 x 18 m patch，V7 primitive 的 high/
extreme difficulty 固定为 `0.8/1.0`。Slope gradient 对应约 `0.32/0.40`，discrete
obstacle height 对应约 `0.084/0.10 m`。V7 random rough primitive 忽略 difficulty，
因此 high/extreme 只是相同分布的独立样本，JSON 明确禁止声称 extreme 更难。

Continuous straight 使用单一 8 x 4 m approach-feature-exit surface，覆盖 slope/stairs
up/down、random rough 和 discrete obstacle。Route 为 `x=1..7 m`，feature 为
`x=2..4.4 m`，height scan 边界余量 `0.2 m`。它是真实 intra-patch continuous
transition，不声称 inter-patch 连续；旧 patch 不缩放为曲线，continuous curves 和
stairs curves 明确拒绝。

#### 正式 GPU JSON 与结果

共同 checkpoint/task/seed：

```text
checkpoint: logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
task: Unitree-Go2-Rough-V7
seed: 42
```

正式 JSON 均位于上述 V7 run 目录：

```text
route_terrain_curves_arc_clean_low_medium_metricsv2_seed42_64env_2000steps.json
route_boundary_high_extreme_arc_clean_metricsv2_seed42_64env_2400steps.json
route_boundary_high_extreme_s_curve_clean_metricsv2_seed42_64env_2400steps.json
route_boundary_high_extreme_arc_randomized_slopes_seed42_32env_2400steps.json
route_boundary_continuous_straight_clean_levels7_9_metricsv2_seed42_12env_2400steps.json
route_boundary_continuous_straight_randomized_levels7_9_metricsv2_seed42_12env_2400steps.json
```

统一汇总：

```text
matrix                         done   reset  progress  action  action P95max  slip   slip P95max  base/upper/calf samples
low/medium arc clean           64/64     0    1.0000   0.076      0.279      0.036     0.185          0/0/0
high/extreme arc clean         37/64    14    0.7412   0.112      0.975      0.055     0.611          0/4/8
high/extreme S clean           37/64    18    0.7025   0.106      0.956      0.055     0.615          0/3/11
high/extreme arc randomized     7/32    19    0.5867   0.299      1.011      0.068     0.785          0/1/9
continuous straight clean      12/12     0    1.0008   0.096      0.333      0.045     0.230          0/0/11
continuous straight randomized 10/12     2    0.8807   0.258      0.513      0.051     0.228          0/0/11
```

Low/medium regression 64/64、零 reset/termination，assignment error `5.96e-8`、
route placement error `0`，说明新指标没有改变旧结论。

High clean arc：slope_up high/extreme 均 `0/8`；slope_down high `6/8`、extreme
`0/8`。High clean S 的对应结果为 `0/8, 0/8, 7/8, 0/8`。Random rough 两种
route 均 `16/16`；non-slope 不是主要失败来源。Arc/S slope failure 的 controller
saturation 分别只有约 `2.8%/1.1%`，mean commanded vx 约 `0.394/0.392 m/s`，mean
actual vx 仅 `0.160/0.167 m/s`；corridor/scan 最小 margin 为 `4.2/1.27 m`。
Clean termination totals：arc fell/base/upper/calf=`1/1/9/6`，S=`5/0/9/4`；另有
低进度 step-limit `13/9`。因此这是 sustained high/extreme pyramid slope 下可复现的
locomotion 执行边界，不是 controller saturation、horizon 或 scan 越界。

High randomized slope arc 为 `7/32`，与同子集 clean `6/32` 同级，说明随机化不是
主要根因。Continuous clean levels 7/9 在修复 timeout 后为 `12/12`、零 reset；
randomized 为 `10/12`，level-9 stairs_up/down 各因 illegal calf contact reset 一次，
其余 slope/rough/obstacle `8/8`。

#### 验收和下一步

Acceptance Agent 最终判定 **Evaluator Gate PASS**：239 tests PASS、1 个既有
intentional skip；compileall、4 个 CLI、V7 registry、diff-check、6 份正式 JSON 的
recursive finite/schema/sample freeze/P95/max/contact denominator/placement/margin/
coverage 全部通过。无残留 train/evaluate/play 进程，GPU 空闲。

Model Gate 单独记录为部分 FAIL：high/extreme sustained slope curves 和 randomized
level-9 stairs 暴露能力边界，但不反向否定 evaluator。本轮固定 **NO-GO，未启动训练，
没有新 checkpoint；V7 `model_13600.pt` 继续作为默认模型**，同时明确不宣称已通过
上述边界场景。

下一步先做同一 18 m high/extreme pyramid patch 的 matched straight/arc/S，区分
“长坡持续暴露”与“坡地 forward+yaw/横坡耦合”。再对 randomized level-9 stairs
使用 seed 43/44 复现。只有归因稳定后才定义单变量 probe：若 straight 也失败，只
提高 high/extreme slope hard-case terrain exposure；若只曲线失败，只增加高坡上的
parameterized forward+yaw hard-case sampling。两种方案都冻结 reward、termination、
gait、network 和其余通用 terrain 分布。

### 2026-07-21 High-slope matched attribution（NO-GO，未训练）

本阶段完成 strict matched high-slope evaluator、离线归因和 level-9 stairs 多 seed 复验。
最终 integration HEAD 为 `8aba90d`；子任务提交为 evaluator `a766432`、归因
`e85d0e0`、PRE-GPU acceptance `c4f1bbf`、failure-reason 修复 `174c19a`、验收回归
`609eb06`。最终 PRE-GPU：全量 276 tests PASS（1 skipped）、compileall、V7
registry/import/config load、CLI、diff-check、worktree/process/GPU clean 均 PASS。

正式 V7 checkpoint：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

合法 r=2.5 clean matched JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_matched_clean_r2p5_seed42_2400steps_v2.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_matched_clean_r2p5_seed42_2400steps_v2_attribution.json
```

矩阵固定为 18×18 m patch、r=2.5、high/extreme slope up/down、v=0.3/0.5、左右
slot、seed42、2400 steps、每个 route kind fresh environment。结果为 straight
`4/16`、arc `3/16`、S `3/16`；平均 forward gain 约 `0.504/0.516/0.526`，
progress ratio 约 `0.517/0.461/0.464`。方向误差较小，但出现 fell、upper-leg、base
和 calf termination。arc/S controller saturation max 约 `0.384/0.352`，超过归因
阈值 `0.1`，故严格归因输出 `inconclusive_no_training`，不授权训练。

Matched slot 交叉审查进一步显示：clean 有 `12/16` 三路线共同失败，randomized 有
`11/16` 共同失败；绝大多数失败 slot saturation <= 0.1，少数高 saturation slot 反而
完成。`slope_up` high/extreme 两个 profile、三路线全部失败；`slope_down high` 基本
通过，而 `slope_down extreme` 基本失败。2400 steps 高于理想最低约 883 steps，且无
near-end retry candidate。定性证据更支持 sustained slope/contact-stability 短板，
不支持普遍曲率耦合；正式 analyzer 仍按预声明规则保持 LOW/INCONCLUSIVE。

randomized matched JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_matched_randomized_r2p5_seed42_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_randomized_r2p5_seed42_stairs_seed42_43_44_attribution.json
```

randomized completion 为 straight `5/16`、arc `4/16`、S `4/16`，forward gain
约 `0.505/0.566/0.557`；同样因 saturation/归因规则不满足而保持 NO-GO。

Level-9 continuous randomized stairs（0.5 m/s、2400 steps）JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_stairs_level9_randomized_seed42_2env_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_stairs_level9_randomized_seed43_2env_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_stairs_level9_randomized_seed44_2env_2400steps.json
```

上楼梯只有 seed42 calf termination（1/3），下楼梯只有 seed43 calf termination
（1/3）；其余通过，归因为 `heterogeneous_failures_require_more_diagnosis`，不是
稳定的同方向楼梯风险。

r=4 straight 的 scan footprint 在 18×18 m 中心 patch 越界约 0.1776 m，正式 evaluator
如实拒绝，不能作为策略失败。failure-reason schema 已修复：completed 使用 JSON null，
失败使用非空真实原因；首轮错误 JSON 仅留作审计。全阶段固定 **NO-GO，未启动训练，
没有新 checkpoint**；V7 `model_13600.pt` 继续为默认模型。下一步先对 controller
saturation/command 生成做 command-tape 对照，只有确认是 policy 执行短板后才设计单变量
probe。

POST-GPU Acceptance：Evaluator Gate **PASS**，所有正式 JSON finite/schema/identity/
matched invariants 通过，重算 attribution 与保存 JSON 逐字段一致；Model Gate
**FAIL**，因果归因置信度 LOW/INCONCLUSIVE。无残留 train/evaluate/play/rsl_rl 或
GPU compute process。下一项只允许 evaluation-only controller-headroom A/B：聚焦
`slope_up high` 与 `slope_down extreme`，保持 policy/seed/terrain/route/horizon 不变，
只统一缩放 lateral/yaw controller limit 并记录分轴 saturation。若三路线仍共同欠速，
再把 sustained slope 作为候选单变量训练方向；否则优先修 controller/curvature coupling。

### 2026-07-21 Final high-slope attribution 与 training-ready probe（未训练）

本阶段使用 1 个 Integration Agent 和 3 个独立 worktree Agent，integration 分支为
`exp/final-slope-diagnosis-integration`。关键 integration commits：

```text
e1d75f6  final slope acceptance contracts
6775908  final high-slope training decision contract
0a9176f  controller-headroom A/B evaluator
90a4701  separate static A/B identity from dynamic rollout values
d40dfce  identity regression tests
19bf43b  V7 high-slope sampling probe implementation
```

默认 checkpoint 全程固定为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

#### Controller-headroom A/B

唯一 evaluation 变量是 closed-loop controller 的 `max_lateral_speed` 和
`max_yaw_rate` 同时从 scale `1.0` 放宽到 `1.5`。Checkpoint、seed42、terrain slot、
route、r=2.5、v=0.5、2400-step horizon、policy 和 randomization profile 均严格匹配。
首轮 GPU 运行因 validator 错把 fresh-env placement 动态浮点值纳入 exact static identity
而 fail-fast，未产生 JSON，不用于结论；修复后 dynamic rollout fields 不参与静态身份，
静态 GPU float 使用 `abs_tol=1e-5`，真实 mismatch 报完整字段路径。

正式 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/final_high_slope_headroom_clean_seed42_r2p5_v0p5_2400steps.json
SHA256 be2b583d6b72ca24c1df8464d0883022d73e52a2f864814ea68b259166527e5a

logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/final_high_slope_headroom_randomized_seed42_r2p5_v0p5_2400steps.json
SHA256 31e80f3812b2bb9c6c9766d65d724ffbfe55617b48ffce028038907018e9c578
```

结果：

```text
profile     scale  straight completion/gain  arc completion/gain  S completion/gain
clean       1.0       0/4 / 0.587               0/4 / 0.462         0/4 / 0.704
clean       1.5       0/4 / 0.607               0/4 / 0.546         0/4 / 0.464
randomized  1.0       1/4 / 0.629               0/4 / 0.517         0/4 / 0.531
randomized  1.5       0/4 / 0.488               0/4 / 0.515         0/4 / 0.508
```

Scale 1.5 后 clean/randomized route gain spread 分别为 `0.143/0.027`，且 randomized
三路线 maximum saturation 均低于 `0.01`，但 completion 没有恢复。失败仍包含
fall/base/upper-leg/calf contact 和 low-progress step-limit。因此最终归因为
`sustained_slope_locomotion_limited`，controller-vs-policy 因果置信度 HIGH；这不是跨
seed 的统计置信区间。Evaluator/Artifact Gate PASS，Model Gate FAIL，Training Design
Gate 为 `TRAINING-READY`。r=4 straight 仍因 scan margin `-0.1775804 m` 在 GPU 前拒绝。

Level-9 stairs seeds 42/43/44 维持既有结论：up/down 各 `2/3`，分别只有一个不同 seed
的 calf failure，属于异质、低置信回归风险，不合并进训练变量。V7 regression 保持：
patch matrix clean/randomized 均 `112/112`；continuous clean `12/12`、randomized
`10/12`（既有两个 calf failures）。

#### 唯一训练变量和实现

Hard-case 集合固定为：

```text
H = slope_up levels 8/9 + slope_down level 9
nominal V7 ratio = 3.0%
model_13600 snapshot = 64/2048 = 3.125%
probe target = 10.0%
```

新增任务 `Unitree-Go2-Rough-V7-HighSlopeProbe`。Sampler 在原
`terrain_levels_vel` curriculum 之后、`reset_base` 之前执行；只改变达到 10% membership
quota 所需的最少 slot，membership 已符合的 post-curriculum candidate 原样保留。H 和
non-H donor 各自按当前条件分布采样，nominal V7 仅作 zero-mass fallback。Reward、command
distribution、terrain geometry、termination、gait、randomization、observation、height
scan、network 和 V7 task 均未改变。

Sampler telemetry/state 包含 candidate/changed/batch/cumulative/population ratio、整数
reset/hard count、sampled slot histogram、RNG 和 quota residual。Runner 在 probe
checkpoint 中保存/恢复完整 sampler state；从旧 V7 checkpoint warm start 时，terrain
state 恢复后重新 rebase，wrapper 的 preload reset 不计入正式统计。

最终独立验收：PRE-GPU targeted `70/70` PASS；全量 `321` PASS、1 个既有无关 skip；
compileall、CLI、task registry、RL cfg、runner load 和 diff-check 均 PASS。真实 GPU
smoke 使用 probe task、2048 env、seed42 严格加载 `model_13600.pt`，不调用 `learn`：

```text
loaded iteration                         13600
restore hard population                  64/2048 = 3.125%
restore sampler reset/hard/hist count    0 / 0 / 0
restore root-relative error max          3.81e-6
first real full-reset candidate H        25/2048
first real full-reset sampled H          204/2048 = 9.96094%
changed slots                            179 (理论最小差额)
non-H conditional max abs / TV           0.001848 / 0.03974
terrain origin error                     0
observations                             finite
sampler state round-trip                 PASS
```

GPU 验收后无残留 train/evaluate/play/rsl_rl 进程，GPU `0 MiB/0%`。本阶段没有调用
PPO learn、没有新训练 run、没有新 checkpoint；V7 `model_13600.pt` 继续为默认模型。

#### 下一轮固定训练命令

开始前再次确认 worktree/process/GPU clean，并把命令原样记录。只允许：

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab

python scripts/train.py Unitree-Go2-Rough-V7-HighSlopeProbe \
  --env.scene.num-envs=2048 \
  --env.seed=42 \
  --agent.seed=42 \
  --agent.resume=True \
  --agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter \
  --agent.load-checkpoint=model_13600.pt \
  --agent.max-iterations=400 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.run-name=go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

训练后必须使用完全相同的 clean/randomized high-slope matched matrix、stairs seeds
42/43/44 和 V7 regression matrix。接受条件包括 high-slope completion 明显提高、mean
forward gain 达到 `0.8`、保留场景 forward gain 不得比 V7 低 `0.05` 以上、slip/action
acceleration 不超过同场景 V7 的 `1.2x`、contact/fall 不增加且 stairs/flat/rough/obstacle
不退化。任一 gate 失败即拒绝新模型、保留 V7，不追加第二变量。

### 2026-07-21 High-slope 10% sampling probe 正式训练启动记录

启动时间：`2026-07-21 16:09:52 +0800`。Integration 分支：
`exp/high-slope-probe-integration`，起始 HEAD `b102a42`。本轮二选一终点固定为
ACCEPT 新 checkpoint 或 REJECT 并保留 V7；第一次 probe 失败后不补训、不增加变量。

唯一变量：

```text
target_hard_case_ratio: nominal V7 3.0% / checkpoint 3.125% -> probe 10.0%
H: slope_up levels 8/9 + slope_down level 9
```

Warm start：

```text
checkpoint: logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256: 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
size: 7077833 bytes
task: Unitree-Go2-Rough-V7-HighSlopeProbe
num_envs: 2048
iterations: 400
seed: 42
logger: tensorboard
```

Reward、command distribution、terrain geometry、termination、gait、randomization、
observations/height scan、actor/critic network、PPO 参数全部冻结。正式命令：

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab

python scripts/train.py Unitree-Go2-Rough-V7-HighSlopeProbe \
  --env.scene.num-envs=2048 \
  --env.seed=42 \
  --agent.seed=42 \
  --agent.resume=True \
  --agent.load-run=2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter \
  --agent.load-checkpoint=model_13600.pt \
  --agent.max-iterations=400 \
  --agent.save-interval=100 \
  --agent.logger=tensorboard \
  --agent.run-name=go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

训练前主 Agent targeted tests `29/29 PASS`；独立 Acceptance PRE-GPU 结果在实际训练
开始前追加。训练 run 目录、checkpoint、sampler telemetry、TensorBoard tail、正式 JSON
和最终 ACCEPT/REJECT 将在本节续写。

#### 正式训练结果

PRE-GPU 独立验收 PASS：全量 `321 PASS + 1` 个既有无关 skip，high-slope targeted
`82/82 PASS`，compileall、task-aware CLI、registry/config/runner、单变量 diff、V7
checkpoint 和两份 baseline JSON 合约全部通过。正式 run：

```text
logs/rsl_rl/go2_velocity/2026-07-21_16-21-46_go2_rough_v7_high_slope_sampling_probe_2048env_400iter
耗时: 18m37s
model_13700.pt SHA256 43a3823b3a26c9a7491332b931128653d410bd85d085e86c146dd8e3122c5d2f
model_13800.pt SHA256 6312ca929edb9d33953a4edcf4da66733711c825ddf147489e5113b089f34101
model_13900.pt SHA256 52aa46dc1a7c1ac6ef54b89c777732a5fed93e2c396a75e13f4a22db20004eb5
model_13999.pt SHA256 76c8f9017d7200fdf1477b89f35100a7dce9b8db3295ae424e1e6adfef339f4d
```

所有 checkpoint 均含 actor、critic、optimizer、terrain state 及 sampler
RNG/quota/counter/histogram。Final common-step delta 为 `9600 = 400 * 24`，确认完整跑完
400 iterations；final sampler 为 `1996/19966 = 9.996995%`，目标误差在 deterministic
quota bound 内。TensorBoard 400 个 samples、63 tags 全部 finite，无 OOM、NaN、stall
或持续 loss divergence。Final tail100：

```text
reward 49.612, episode length 985.85, terrain 5.676
linear tracking 0.8239, angular tracking 0.9005
slip 0.08124, action acceleration 0.7792
fell/base/upper/calf 0.00500/0.01667/0.02083/0.03667
```

训练健康 gate PASS，但相对 V7 source window，tracking 约低 2%，slip 高 6.2%，action
acceleration 高 8.1%，base/upper/calf termination 偏高，必须由固定 evaluator 决定。

#### 阶段 checkpoint 筛选

固定 clean r=2.5、v=0.5、8 slots/route、2400 steps 的 lexicographic 结果：

```text
checkpoint  total complete  min route  weighted vx gain  physical terms  routes straight/arc/S
13700           4/24            0          0.508              18             0/1/3
13800           9/24            2          0.453              12             4/2/3
13900           9/24            3          0.548              13             3/3/3
13999           8/24            2          0.459              13             3/2/3
```

按预声明顺序选择 `model_13900.pt` 为 candidate；final `model_13999.pt` 仍独立跑完整
矩阵。四份 stage JSON 均通过 finite/schema/matched/geometry/placement/lifecycle 验证，
文件前缀为 `stage_rank_high_slope_clean_...`。

#### 完整 high-slope 结果

正式 JSON：

```text
post_candidate_high_slope_matched_clean_seed42_r2p5_v0p3_0p5_16slots_2400steps.json
post_candidate_high_slope_matched_randomized_seed42_r2p5_v0p3_0p5_16slots_2400steps.json
post_final_high_slope_matched_clean_seed42_r2p5_v0p3_0p5_16slots_2400steps.json
post_final_high_slope_matched_randomized_seed42_r2p5_v0p3_0p5_16slots_2400steps.json
```

Completion / mean scenario vx gain / physical termination flags：

```text
profile     route       V7                    candidate 13900          final 13999
clean       straight    4/16 / .500 / 12       5/16 / .528 / 4        2/16 / .558 / 4
clean       arc         3/16 / .516 / 9        3/16 / .634 / 8        4/16 / .566 / 10
clean       S           3/16 / .526 / 10       5/16 / .693 / 9        4/16 / .605 / 9
randomized  straight    5/16 / .499 / 10       3/16 / .461 / 8        4/16 / .495 / 6
randomized  arc         4/16 / .546 / 10       4/16 / .515 / 9        3/16 / .498 / 11
randomized  S           4/16 / .555 / 12       4/16 / .522 / 9        4/16 / .509 / 10
```

Candidate 要求 clean 每路线至少 `12/16`、randomized 至少 `10/16`、gain `>=0.80`，
实际全部失败；相对 V7 也未达到每路线 `+0.20`。Candidate clean 的 matched-slot slip
P95 在 straight/arc/S 分别有 `4/7/9` 个 slot 超过 V7 `1.2x`，action P95 有
`3/3/4` 个违反；final clean weighted slip 已是 V7 的 `1.24x/1.28x/1.35x`。
High-slope Model Gate 对 candidate 和 final 均 FAIL。

#### 楼梯、通用地形和 tracking 回归

所有正式 JSON 位于上述 run 目录，命名组为：

```text
post_{candidate,final}_stairs_level9_randomized_seed{42,43,44}_2env_2400steps.json
post_{candidate,final}_patch_flat_rough_obstacle_{clean,randomized}_seed42_48env_700steps.json
post_{candidate,final}_continuous_straight_{clean,randomized}_levels7_9_seed42_12env_2400steps.json
post_v7_candidate_final_tracking_{clean,randomized}_seed42_1120env_1000steps.json
```

回归结果：

```text
matrix                              V7 baseline        candidate 13900       final 13999
stairs up seeds42/43/44                 2/3                 3/3                  2/3
stairs down seeds42/43/44               2/3                 3/3                  1/3
patch clean flat/rough/obstacle         48/48               48/48                48/48
patch randomized                       48/48               48/48                48/48
continuous clean                       12/12               12/12                12/12
continuous randomized                  10/12               12/12                12/12
```

Candidate 楼梯六次全部通过；final 出现 seed42/44 两次 stairs_down calf failure 和
seed43 stairs_up base failure，违反 no-new-base 和 stairs-down `>=2/3` gate。普通 patch
与 continuous 能力没有遗忘。

Fixed-command tracking（V7 / candidate / final）：

```text
profile     linear error       overall gain       yaw error          slip              action acc
clean       .1170/.1219/.1190  .6253/.6153/.6162  .0498/.0530/.0497  .0346/.0361/.0356  .0781/.0762/.0788
randomized  .1387/.1431/.1425  .5985/.5815/.5857  .0754/.0786/.0753  .0442/.0447/.0450  .2412/.2353/.2399
```

Aggregate command gain/slip/action 看似接近 V7，但按锁定的 retained-scene cell gate，
randomized `forward_0.3` stairs 从 `0.681 -> 0.526`，下降 `0.155`；candidate clean
tracking 的 yaw-left/flat slip 也达到 V7 `1.50x`。Continuous completion 虽为 `12/12`，
candidate non-terminating calf contact 从 V7 clean `11/10014` 升到 `34/9414` active steps，
randomized 从 `11/8769` 升到 `37/9708`，因此 tracking/contact safety 也 FAIL。

#### 最终决定

独立 POST Acceptance：repository/artifact integrity PASS；24 份 stage/post JSON 全部通过
strict JSON、finite、schema、identity、placement 和 lifecycle；模型 gate FAIL。
**REJECT `model_13900.pt` 和 `model_13999.pt`。** 10% high-slope exposure sampler 与
训练管线本身验证正确，但该单变量不足以解决 sustained high-slope locomotion，且 final
出现明确 stairs regression。不得追加训练或提高 hard-case ratio；默认部署模型继续是
V7 `model_13600.pt`。下一步应先分析为何高坡 exposure 增加后 straight forward gain
仍低于约 0.56，再决定新的单变量机制；优先候选是 command/terrain-conditioned
foot-placement 或 step-length shaping，但必须先做离线/短诊断，不直接改 reward 并训练。
