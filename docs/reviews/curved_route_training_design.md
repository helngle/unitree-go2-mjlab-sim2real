# V7 Curved-Route Training Design Review

本报告只审查 V7 当前的命令采样、观测接口和曲线路径训练决策，不修改 reward、训练配置、
terrain 或 policy 网络。默认 checkpoint 是 `model_13600.pt`。在曲线路径 baseline 和
Acceptance Agent 通过前，不授权 PPO 长训练。

## 结论

V7 不是路径跟随策略，而是局部身体坐标系 twist 执行器。actor 能看到三维
`(vx, vy, wz)` 命令、本体状态、步态 phase 和局部 height scan，但看不到 waypoint、
世界位置、路径切线或累计 cross-track error。因此圆弧/S 弯必须由外部 path controller
把路径转换成逐控制步的 body-frame 命令；训练只能改善 policy 对这些局部命令的执行。

当前 general mode 有独立的 forward 和 yaw 采样，数值范围部分覆盖圆弧需要的曲率，但没有
保证 `wz` 与 `vx` 的相关性。曲线路径不能仅凭已有 general/yaw 比例宣称已训练覆盖。

## V7 命令和时序审查

`ModeVelocityCommandCfg` 的基础 mode 概率为：

| mode | 概率 | 采样范围 |
| --- | ---: | --- |
| general | 0.40 | `vx=(0.15,0.8)`, `vy=(-0.1,0.1)`, `wz=(-0.3,0.3)`，三个分量独立均匀采样 |
| lateral | 0.25 | `vx=0`, `vy=+- (0.1,0.3)`, `wz=0` |
| yaw | 0.15 | `vx=0`, `vy=0`, `wz=+- (0.2,0.7)` |
| high_speed | 0.20 | `vx=(0.8,1.0)`, `vy=(-0.05,0.05)`, `wz=(-0.15,0.15)` |

每个 command 由基类以 `resampling_time_range=(3,8)` 秒 piecewise-constant 保持；仿真
控制步为 `0.02` 秒。V7 关闭 heading command，且 `unitree_go2_rough_v7_env_cfg()`
移除了旧的 `command_vel` curriculum。因而 `commands_vel` 不会改变 V7 mode 内部的
`general_*`、`high_speed_*` 等范围；focus terrain 只把高难度指定 terrain 上的
high-speed 概率重加权到 0.45。

actor observation 的 command term 直接读取 command manager 的三维身体坐标命令；没有
路线状态。critic 额外有真实 base linear velocity，但这不会在部署 actor 中提供路径信息。

## 圆弧需求是否在当前支持范围

对恒速圆弧，若机器人朝向保持路径切线，理想命令为：

```text
vx = v, vy = 0, wz = sign * v / r
```

本阶段参数 `r={1.5,2.5,4.0} m`、`v={0.3,0.5,0.6} m/s` 对应的 `|wz|` 为：

| radius | v=0.3 | v=0.5 | v=0.6 |
| ---: | ---: | ---: | ---: |
| 1.5 m | 0.200 | 0.333 | 0.400 |
| 2.5 m | 0.120 | 0.200 | 0.240 |
| 4.0 m | 0.075 | 0.125 | 0.150 |

除 `r=1.5,v>=0.5` 外，这些值在 general 的 `|wz|<=0.3` 内；但 general 是三个
分量独立采样，恰好满足 `wz≈v/r` 的概率很低。`r=1.5,v=0.6` 还超出 general 的 yaw
上限，只能落入 yaw/high-speed 的边缘范围，后二者又没有 forward+yaw 耦合。现有命令
保持时间也远长于曲线 controller 应有的更新周期；3--8 秒内只能执行少数几个转弯段，
不能代表连续曲率变化。

因此当前 V7 可能具备部分 forward+yaw 能力，但不能把曲线覆盖归因于现有 sampler；必须
先用固定 command tape 测量实际执行。

## `command_tape` 与 `closed_loop` 的归因协议

1. **数学/虚拟执行器检查**：用已知初始 pose 对 world tangent 生成 command tape，验证
   yaw 符号、`wz=v/r`、body/world 旋转、angle wrap、曲率和 route length。虚拟积分器若
   不能到终点，归因 evaluator/controller，停止训练。
2. **policy command tape（open-loop）**：每个控制步直接写入预先计算的 body command，
   绕过 3--8 秒随机重采样；记录实际 world pose、body `vx/vy`、`wz`、heading 和
   termination。这样能隔离 policy 对 forward+yaw 序列的执行。
3. **closed-loop line/arc follow**：用当前 cross-track 和 heading error 生成 body
   command，并记录命令饱和、更新延迟、期望/实际曲率。如果 tape 通过而 closed-loop
   失败，优先修 controller 或命令时序，不训练 policy；两者都失败且命令未饱和、直线
   baseline 通过，才可归因 locomotion。

诊断必须区分：

* `wz` tracking 差而 `vx` 正常：forward+yaw 耦合短板；
* tape 通过、closed-loop cross-track 差：path controller/坐标变换短板；
* 只有 stairs/rough 失败且出现 contact/slip：terrain exposure 或起点几何问题；
* 只有 `|wz|>0.3` 失败：命令超出 V7 分布，不能称为泛化失败，先标记 OOD。

## 唯一变量训练 probe（仅在 baseline 明确为 locomotion 失败时）

唯一改变 `forward+yaw` 参数化圆弧命令采样分布，建议初始总概率 **15%**。保持
`lateral=25%`、`yaw=15%`、`high_speed=20%` 不变，将 `general` 从 40% 调为 25%，
并把这 15% general quota 改为相关采样：均匀采样 `v∈[0.3,0.6]`、`r∈[1.5,4.0]`、
左右符号各半，设置 `vx=v, vy` 为小范围（例如 `[-0.05,0.05]`），`wz=sign*v/r`。
这使总概率仍为 1，且保留 25% 独立 general 命令作为通用能力锚点。不要同时改变
reward、terrain 比例/难度、termination、randomization、gait phase、网络或 lateral
比例；不要继续使用 pose tolerance probe。

15% 在 2048 env 下每次 resample 约有 300 个 curve samples，足以提供左右半径和速度的
覆盖，同时比 30--45% 更不容易遗忘直线、楼梯和横移。若 15% 仍不足，下一轮只能把
**同一变量**提升到 20%，不能同时改其他项。probe 采用固定概率，不使用额外 staged
schedule，以保持归因清晰。

训练建议：warm start V7 `model_13600.pt`，2048 env，seed 42，300--500 iterations。
训练前后必须运行完全相同的直线 patch、continuous stairs/slope、圆弧和 S 弯矩阵。

## 训练 gates 和“不训练”条件

接受 probe 必须同时满足：clean curve completion >=90%、randomized >=80%、arc-length
progress >=0.90、cross-track RMS <=0.15 m、P95/max <=0.30 m、heading RMS <=10°、
P95/max <=20°；reset/catastrophic termination 不增加，slip/action acceleration
不超过同场景 V7 baseline 的 1.2 倍。直线 clean/randomized、已有上下楼梯和坡地 gate
不得回归，且 forward gain 不得明显低于 V7 约 0.818。

以下任一情况都不训练：虚拟 tape 数学失败；flat straight/open-loop 失败；closed-loop
失败但 policy tape 通过；命令被饱和或超过明确的 V7 支持范围；失败来自 terrain relocation、
route corridor 或 reset 统计污染；Test Agent 尚未在最终 integration HEAD 给出 PASS。

## 风险

* 直接把 `wz=v/r` tape 当成 V7 训练分布会掩盖 3--8 秒 resampling 与 policy command
  latency 差异；evaluator 必须报告每步 command 和实际响应。
* `r=1.5,v=0.6` 的 `wz=0.4` 超过 general 上限，应在 baseline 中单独标记 OOD，不得与
  in-distribution 失败混合统计。
* 只在 flat 上训练曲线命令可能造成 terrain/straight 遗忘；训练 probe 后必须完整复测
  rough terrain 矩阵。
* 当前 terrain 是独立 patch；不能把单 patch 内曲线通过外推为跨 patch 无缝路径能力。
