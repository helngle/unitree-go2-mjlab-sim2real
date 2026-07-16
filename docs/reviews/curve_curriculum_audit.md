# Go2 曲线命令 Terrain Curriculum 审计

本报告只审查 `terrain_levels_vel()`、V7 命令模式及正式 curve control/probe 的
terrain 配对方案。没有修改生产 curriculum、reward、terrain、termination、gait、
policy 网络或训练配置，也没有启动训练。

默认 checkpoint：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

## 结论

当前 `terrain_levels_vel()` **不适合直接用于含圆弧、闭合曲线、纯 yaw 或 episode 内
多次换向的训练实验**。它没有评价路径长度、速度 tracking、yaw tracking 或命令历史，
只评价 episode 终点相对 terrain origin 的二维净位移；降级阈值又只读取 episode 结束
时的最后一个平移命令。

因此，如果在保持该 curriculum 开启的情况下把 15% general quota 换成 curve mode，
terrain level 分布也会随轨迹几何和终止时刻隐式变化。“唯一变量是 curve command
sampling”不成立，训练结果无法可靠归因。

首轮正式实验不应在本轮同时重写 command-aware curriculum。推荐方案是：control 和
probe 都从同一个 `model_13600.pt` 恢复完全相同的 2048 个 `terrain_levels` 和
`terrain_types`，然后移除 `terrain_levels` curriculum term，整个 300-iteration probe
期间冻结环境到相同的 level/type assignment。这样唯一变化才是 correlated
forward+yaw command sampling。

## 当前公式

生产实现等价于：

```text
d_net = ||root_xy(T) - terrain_origin_xy||
v_last = ||command_xy(T)||

move_up   = d_net > terrain_patch_length / 2
move_down = d_net < v_last * episode_length_seconds * 0.5
move_down = move_down and not move_up
```

V7 terrain patch 为 `8 m x 8 m`，episode 为 `20 s`，所以升级只要求：

```text
d_net > 4 m
```

这里有四个重要事实：

1. `d_net` 是起点到终点的弦长/净位移，不是累计走过的路程。
2. `v_last` 只取终止时 command tensor 的 `vx,vy`；不累计 3--8 秒分段命令历史。
3. `wz` 完全不进入升级或降级公式。
4. 一旦 `d_net > 4 m`，`move_up` 会覆盖 `move_down`，即使只完成高速命令理论距离的
   一小部分也仍然升级。

## 参数化轨迹的数学行为

对半径 `r`、转角 `theta` 的理想圆弧：

```text
实际路径长度 s = r * theta
当前公式测量值 d_net = 2 * r * sin(theta/2)
d_net / s = 2*sin(theta/2)/theta
```

| 轨迹 | `d_net / s` | 当前行为 |
| --- | ---: | --- |
| 直线 | `1.000` | 只在命令方向和 episode 内基本一致时近似合理 |
| 90 度圆弧 | `0.900` | 用弦长代替弧长；半径决定能否跨过 4 m 升级线 |
| 180 度圆弧 | `0.637` | 已完成整个半圆仍只计约 64% 路程 |
| 360 度圆弧 | `0.000` | 完美回到起点会被当成没有前进并降级 |
| 两段反向 60 度 S 弯 | `3/pi = 0.955` | 仍只看终点；`d_net=2r`，半径而非 tracking 质量主导升级 |

具体而言，理想两段 60 度 S 弯在 `r=1.5/2.5/4.0 m` 时净位移分别为
`3/5/8 m`。三者 tracking 可以同样完美，但在 8 m patch 上，`r=1.5` 不升级，
`r=2.5/4.0` 升级。这个结果衡量的是路线几何，不是 locomotion 能力。

独立 contract test 还验证了：

- 直线和纯横移在方向保持、无重采样时表现相同；完成不到最后命令理论距离的一半会降级。
- 纯 yaw 的 `v_last=0`，所以即使 yaw 完全不响应也不能降级；反而横向漂移超过 4 m
  会升级。
- 同样累计行走 5 m，直线终点在 5 m 时升级；闭合轨迹回到起点时降级。
- 前进后后退或左右命令抵消，累计路程可以很长，净位移为零时仍会降级。
- 如果 evaluator/命令在 episode 结束前进入零命令 settle，降级阈值也变为零；同一个
  失败终点会从“降级”变为“不变”。
- 终点同为 2 m、episode 同为 10 s 时，最后命令 `0.2 m/s` 不降级，最后命令
  `1.0 m/s` 降级。此前执行过的命令完全不参与判断。

## 对 V7 现有模式的影响

V7 基础 mode 概率和命令范围为：

| mode | 基础概率 | 命令特征 | curriculum 偏差 |
| --- | ---: | --- | --- |
| general | 40% | `vx>0`，小 `vy`，独立 `wz` | yaw 造成的曲线/换向只通过终点净位移间接体现；最后一次命令决定降级阈值 |
| lateral | 25% | 纯正/负 `vy` | 3--8 秒重采样换符号时位移会抵消，更容易降级 |
| yaw | 15% | `vx=vy=0` | yaw tracking 不参与公式，失败不能降级，漂移可升级 |
| high_speed | 20% | `vx=0.8--1.0` | 20 秒理论距离约 16--20 m，但净位移刚超过 4 m 就无条件升级 |

standing command 同样不会降级，因为平移阈值为零。

此外，V7 在指定高难 terrain 且 level >= 7 时，会将 high-speed mode 从 20% 重加权到
45%。因此 terrain level 的几何偏差不只改变 terrain 难度，还会反馈到后续命令 mode
分布：

```text
轨迹几何/最后命令
-> terrain level 升降偏差
-> 是否进入 level>=7 focus 区域
-> high-speed mode 概率变化
-> 下一轮 locomotion 数据分布变化
```

所以 V7 历史 `terrain_levels` 不能解释为所有 mode 上统一可比的通过能力。它仍可作为
训练状态和粗略难度指标，但不能单独证明 yaw、曲线或换向能力。

## `model_13600.pt` 的可恢复状态

本轮以只读方式检查 checkpoint，确认 `infos.env_state` 包含：

```text
common_step_counter: 326664
terrain_levels: shape=(2048,), dtype=int64, mean=5.275390625, range=[0,9]
terrain_types:  shape=(2048,), dtype=int64, range=[0,19]
iteration: 13600
```

level histogram：

```text
[71, 101, 155, 194, 249, 271, 261, 271, 278, 197]
```

20 个 terrain type 每列各有 102 或 103 个环境。现有
`VelocityOnPolicyRunner._restore_environment_state()` 会在环境数一致时恢复 levels/types，
根据新 origin 平移 root pose，并刷新仿真和传感器。它在 curriculum term 被移除时仍会
执行恢复，因为 runner 保存/恢复逻辑直接读取 terrain entity，而不是 curriculum manager。

限制：checkpoint terrain state 只有 2048 个元素。正式 matched control/probe 必须都使用
`num_envs=2048`；若环境数不同，runner 会明确跳过恢复，不能声称 terrain distribution
已经配对。

## 推荐的 matched control/probe 契约

首轮优先使用冻结 terrain，而不是同时上线新 command-aware curriculum。

### 共同设置

```text
warm start: model_13600.pt
num_envs: 2048
iterations: first 300
seed: 42
terrain geometry/proportions: V7 unchanged
terrain levels/types: restore exact checkpoint arrays
terrain_levels curriculum term: removed after config construction
terrain generator: keep unchanged so checkpoint columns retain identical meaning
reward/termination/randomization/gait/network/resampling cadence: frozen
```

执行顺序必须是：

1. 从 V7 config 派生专用 control/probe task。
2. 仅移除 `cfg.curriculum["terrain_levels"]`；不要修改生产 `terrain_levels_vel()`。
3. 创建 2048-env 环境并通过 `VelocityOnPolicyRunner` strict-load 同一个 checkpoint。
4. 在第一个 rollout 前断言实际 `terrain_levels/types` 与 checkpoint tensor 完全相等。
5. control/probe 分别运行，禁止串接续训；两者都必须从原始 V7 `model_13600.pt` 开始。
6. 训练结束再次断言 level/type tensor 与起始值完全相等。
7. JSON/训练日志记录 level/type histogram 及 hash，证明两组 terrain assignment 相同。

Control：

```text
curve_probability=0
V7 general/lateral/yaw/high-speed = 40/25/15/20
```

Probe：

```text
curve_probability=0.15
general: 40% -> 25%
lateral/yaw/high-speed: 25/15/20 unchanged
curve: 15%, correlated vx and wz=sign*vx/r
```

为了保持真正单变量，curve mode 在 focus terrain 上如何被重加权必须预先写死并测试。
最保守的定义是 curve 只替换 general quota，focus 重加权时与 general 使用相同 scale；
不得因 level/type 不同额外提高 curve quota。control 和 probe 均需输出
`mode x terrain_type x terrain_level` 直方图。

## 为什么首轮不推荐立即改 command-aware curriculum

长期正确方向是基于 episode 累计量，例如：

```text
linear_tracking_score = mean(exp(-||v_actual_xy-v_command_xy||^2/sigma_v^2))
yaw_tracking_score = mean(exp(-(wz_actual-wz_command)^2/sigma_w^2))
survival/contact gates
optional commanded arc-length progress
```

但选择权重、standing/yaw/lateral 特例、termination 截断和 terrain-specific threshold 都是
新的实验变量。若本轮同时修改 curriculum 与 command sampler，无法判断结果来自曲线暴露
还是难度升降策略。冻结 checkpoint terrain assignment 能以最小改动隔离问题；待 curve
probe 归因完成后，再将 command-aware curriculum 作为独立实验。

## GO / NO-GO 建议

当前结论是 **对“直接保持 terrain curriculum 开启并训练 curve sampler”给出 NO-GO**。

只有以下条件通过后，才可对 matched curve control/probe 给出 GO：

- 专用 task 恢复并冻结 checkpoint 的 2048 个 level/type；
- control/probe 起止 histogram 和 assignment hash 完全一致；
- `curve_probability=0` 与 V7 sampler 等价；
- focus terrain 上的 curve/general 重加权契约有独立测试；
- command response 诊断证明 coupled forward+yaw 存在额外 locomotion 短板；
- Test Agent 在最终 integration HEAD 给出 PASS。

在这些条件满足前，不应启动 2048-env 正式训练。

## 风险和未完成项

- 本审计没有实现冻结 task；该文件归属 Command/Integration Agent，不能由本 Agent改
  `env_cfgs.py`。
- 本审计没有设计生产级 command-aware curriculum，只给出后续独立实验的指标方向。
- 同 seed 和相同 level/type 仍要求 terrain generator 配置、列含义和随机化配置完全一致；
  任一变化都会破坏配对。
- matched control/probe 各自的 command 随机序列会因 sampler 分支不同而分叉，这是目标
  变量的自然结果；terrain assignment 和非命令配置仍必须严格匹配。
- 固定 terrain 会移除训练中的自适应难度变化，因此结果只能回答“相关曲线命令采样在
  固定 V7 terrain 分布上是否有效”，不能直接回答新 curriculum 的长期收敛能力。
