# Go2 Complex-Path Training Design Review

本报告是对 V7 `model_13600.pt` 后续复杂路径工作的训练设计审查。它只分析当前实现和
评估边界，不修改生产配置、reward、command sampler 或 terrain。结论先行：当前策略训练的是
局部速度命令执行，不是路径跟随；当前 terrain patch 也没有提供真正的 flat-to-stairs 连续
场景。第一阶段应先完成不训练的直线 route baseline，确认命令坐标、起点和路线指标正确，
再决定唯一训练变量。

## 1. 当前 V7 的可观测输入和命令时序

`make_velocity_env_cfg()` 的 actor observation 包含：IMU angular velocity、projected
gravity、`twist` 三维身体坐标速度命令、0.6 s phase、关节位置/速度、上一动作和
1.6 m x 1.0 m、0.1 m resolution 的 yaw-aligned height scan。critic 额外有真实 base
linear velocity、foot height/contact/air-time/force。actor 没有世界位置、目标 waypoint、
路线切向量或累计路径误差；因此在现有接口下它不能直接知道“要走哪条全局路线”。

V7 使用 `ModeVelocityCommand`：general/lateral/yaw/high-speed 的基础概率是
40/25/15/20%，命令每次按基类的 `resampling_time_range=(3, 8)` 秒重采样，纯 mode 内部
保持一个身体坐标系速度。V7 禁用旧的 `command_vel` curriculum，focus terrain 只会将
high-speed 概率重加权到 45%。这适合学习局部 steady-state velocity tracking，不覆盖
waypoint 产生的连续 `vx, vy, yaw` 序列、加减速、转弯耦合或路径误差反馈。

默认 reset event `reset_root_state_uniform` 只在相对 origin 的 `x,y=(-0.5,0.5) m`、
`yaw=(-pi,pi)` 随机，速度为零；浮动根状态随后加上 `scene.env_origins`。这会让单次训练
episode 从 patch 中心附近、任意朝向开始，不能产生指定的“楼梯外缘 -> 中心”或“中心 ->
外缘”起点。reset 后环境会重新计 episode length 和 root state，路线评估必须把它视为该
route 的失败，而不能继续累加新 episode 的位移。

## 2. Terrain 几何和 origin 语义

V7 从 V6 继承 8 m x 8 m patch、10 rows x 20 columns、20 m 全局 border，并配置：

| 类型 | 比例 | 关键范围 |
| --- | ---: | --- |
| flat | 15% | 平面 |
| pyramid stairs | 15% | step height 0--0.12 m（V6 覆盖后），step width 0.3 m，center platform 3 m |
| inverted pyramid stairs | 15% | 同上，中心为低处 |
| pyramid slope | 10% | slope 0--0.4 |
| inverted pyramid slope | 10% | slope 0--0.4，反向 |
| random rough | 15% | height noise 0.01--0.06 m |
| discrete obstacles | 20% | obstacle 0.30--0.80 m、height 0.02--0.10 m |

terrain generator 在 curriculum 模式下按列固定 terrain type、按行增加 difficulty；每个
8 m patch 独立生成，patch 之间不是几何连续的 flat/stairs/slope 链。`terrain_origins[row,
col]` 是该 patch 生成器返回的 spawn origin（通常是中心平台及其顶面），不是 patch 的
几何外缘。`env_origins` 只选择这个 origin 并平移机器人；修正后的 evaluator 也只保持
机器人相对于 origin 的 offset，不会自动把机器人放到路线入口。

楼梯方向具体取决于几何：

* normal `BoxPyramidStairsTerrainCfg` 的中心平台是最高处，向外走是下楼；若要测上楼，
  应从内边界外侧沿 x 或 y 朝中心走。
* inverted pyramid 的中心平台是最低处，中心向外是上楼；要测完整往返，必须区分上楼
  段和下楼段，不能只用一个“stairs”标签。
* V6/V7 的 3 m 中心平台和 1 m stairs border 使 patch 中心附近约有一段平地。默认
  `x,y=±0.5 m` 起点通常仍在平台上，不能代表从第一阶前开始的上/下楼。

因此，已有固定命令 evaluator 中“在 stairs patch 上施加 `vx=0.6`”只能回答中心附近
的局部速度跟踪；它不能证明从楼梯入口走到另一端，也不能区分走向是上楼还是下楼。

## 3. 如何分离 path-controller 和 locomotion 失败

第一阶段必须同时保留两种模式，但先做 open-loop：

1. **Open-loop command tape**：给定路线切线和期望世界速度，离线将其变换成每个控制
   step 的身体坐标 `(vx, vy, yaw_rate)`，固定写入 command，不用机器人实际位置反馈。
   先用一个几何/恒速虚拟执行器验证坐标变换、符号、yaw wrap、命令限幅和 route length。
   若虚拟执行器都不能到达终点，失败是 path generator/evaluator，不是 policy。
2. **Policy open-loop**：对同一 command tape 运行 V7，记录实际世界 root pose、body-frame
   velocity、命令、terrain origin 和每个 termination。若命令 tape 数学正确而机器人出现
   速度/姿态/接触失败，归因于 locomotion；若实际命令执行正常但路线目标定义不一致，归因
   于控制器或坐标接口。
3. **Line-follow**：之后才用当前位置和 heading 误差闭环产生命令。分别记录 command
   saturation、command update latency、cross-track error 和 actual response。line-follow
   成功而 open-loop 失败，说明路径跟随反馈在补偿 locomotion；两者都失败且命令本身合理，
   才值得设计 locomotion 训练 probe。

不要把 `linear response gain` 当作 path success：它只比较身体坐标速度与固定 command，
不包含位置、航向或 route completion。

## 4. 直线 route baseline 的起点和方向

每个 8 m patch 先选 route 方向为 patch local +x，中心线为 `y=0`，起始 yaw=0；同时
运行 yaw=pi 的反向样本，避免把几何方向误认为策略能力。起点应写成相对于目标
`terrain_origin` 的 offset，并将 root z 保持在对应 origin z + Go2 default root height
附近：

* flat、rough、obstacle：`x=-2.5 m -> +2.5 m`，`y` 采样 `{-0.15, 0, +0.15}`，
  route length 5 m。起点须避开 3 m center platform 的特殊语义，但不能越过 1 m border。
* normal pyramid stairs：上楼 `x=-2.8 m -> +1.8 m`（由外向中心），下楼反向；终点
  至少落在中心平台内，另测从中心向外的完整下楼段。实际可用长度应由生成器的
  `step_width/platform_width/border_width` 计算，而不是硬编码“stairs=8 m”。
* inverted pyramid stairs：上楼为中心向外，使用 `x=0 -> +2.8 m`；下楼反向。若要
  测完整 patch 穿越，需要分段标记中心平台，避免把上楼和下楼合成一个 gain。
* slope：从低边朝高边和从高边朝低边分别测试；仅用 terrain type 名称不能确定方向。

起始横向偏移、yaw 偏差、速度（建议 0.3/0.6 m/s）和扰动应作为 case 参数。评估器需要
在初始化时写入目标 offset 和 yaw，并在 `sim.forward()/sense()` 后检查实际 root offset；
不能只改 `terrain.env_origins`。路线终点应有明确的几何容差和最大时间，不能依赖 20 s
episode 到时后“仍然有前进距离”来判成功。

## 5. 直线场景验收 gates

先采用 32 个 episode/场景的 clean smoke，再采用至少 64 个 episode/场景的固定 seed
randomized 矩阵。阈值按 Go2 身体宽度、5 m route 和 V7 当前约 0.8 forward gain 设置，
并同时报告均值、P95 和失败原因：

* completion rate：clean >= 90%，randomized >= 80%；完成要求 progress >= 0.90、
  未发生 reset/termination，并在时间预算内到达终点。
* forward progress ratio >= 0.90；末端纵向误差 <= 0.30 m。
* cross-track RMS <= 0.15 m、P95/max <= 0.30 m；这是约半个 Go2 身宽的可解释容差。
* heading RMS <= 10 deg、P95/max <= 20 deg；终点 heading error <= 15 deg。
* 速度执行作为诊断 gate：forward response gain >= 0.75，或 absolute forward velocity
  error <= 0.15 m/s；straight route cross-axis velocity P95 <= 0.10 m/s。
* clean 中 fell/base/upper-leg/calf termination 应为 0；randomized catastrophic
  termination 总率 <= 5%，并单列每种 contact，而不把一次 reset 后的新 episode 计入原路线。
* slip velocity 和 action acceleration 不得超过 V7 同 terrain/command baseline 的
  1.2 倍；若 baseline 本身超过该线，则先记录基线，不用训练结果掩盖它。

这些是“直线 baseline gate”，不是宣称 V7 必须已经通过的结果。若 V7 不达标，应先报告
每个场景的首次失败时间、位置、命令和 termination；不能直接开始曲线或横移 reward。

## 6. 统一策略的后续训练分布

原则是一个 policy、参数化分布，不是每个 terrain 一个 checkpoint。后续可分三阶段：

1. **基线阶段（不训练）**：固定 V7 checkpoint，跑上述 straight open-loop 矩阵和一小组
   line-follow 对照，确认评估器、route 起点、方向和坐标。
2. **场景暴露阶段**：如果失败确实集中在某一类几何，应只增加该类参数化 patch/入口方向
   的采样概率，同时保留 flat、stairs、slope、rough、obstacle 的混合和原有 dynamics
   randomization。课程变量应是 terrain difficulty、入口偏移和 route command coherence，
   不是按场景拆模型。
3. **hard-case replay 阶段**：按首次失败类型（上楼、下楼、过渡、横向偏移、扰动）回放
   固定比例 hard cases，例如 20--30%，其余保持通用分布；每次只改变一个采样变量并以
   同一完整 route 矩阵复测，避免灾难性遗忘。

训练时可参数化的维度包括 terrain difficulty、stairs step height/width、slope sign、
入口 x/y offset、初始 yaw、恒速段的加减速和短暂 `vx+vy+yaw` 耦合。应让 command tape
   与实际路径 controller 的更新周期一致，而不是依赖现有 3--8 s 随机重采样。

## 7. 单变量 probe 的决策树

在 baseline 结果之前不授权任何 PPO 训练：

* flat straight 失败或虚拟 command tape 失败：修评估器/路径控制坐标，不训练。
* flat 通过、stairs 失败且实际命令正确：优先考虑 stairs 入口方向/连续过渡暴露这一单一
  采样变量；冻结 reward、termination、网络和其他 terrain 比例。
* route 位置误差高但速度/接触稳定：先检查 line-follow 控制器和命令延迟；若确定是
  locomotion 的落脚短板，再设计 command-conditioned foot-placement/step-length reward，
  不先放宽 hip pose tolerance。
* forward+yaw 失败而直线 terrain 通过：先增加命令时间序列/曲率的参数化暴露；不要把
  横移比例、pose tolerance、gait phase 同时修改。
* 只有在证据显示观测不足、且高度图/本体状态无法区分难例时，才讨论 RMA、深度图或
  Transformer；当前阶段明确不引入。

因此，lateral pose probe 的收益不能作为复杂路径成功证据，也不应成为下一轮默认变量。
默认部署模型仍为 V7 `model_13600.pt`，直到同一 route matrix 上出现通过且无 forward、
terrain、contact 或 slip 回归的单变量实验。

## 8. 当前优先风险

1. **评估器任务 ID 风险**：`evaluate_go2_rough.py` 默认 task id 是 `Unitree-Go2-Rough-V6`；
   V7 baseline 运行时必须显式传 `Unitree-Go2-Rough-V7`，否则会误测 V6 配置。
2. **起点/方向风险**：terrain relocation 只修复 patch assignment，不等于 route entrance
   placement；stairs normal/inverted 的上、下方向必须在 case 中显式声明。
3. **连续过渡缺失**：当前 patch 是单一 terrain type，不能从这些 patch 推导 flat-to-stairs
   或 stairs-to-slope 完成率；需要后续专门生成连续几何或明确的多 patch route。
4. **reset 污染风险**：现有固定矩阵 evaluator 累积每步速度统计但没有路径完成状态和
   reset 分段；route evaluator 必须在首次 termination 时冻结该 route 的统计。
5. **命令接口风险**：现有 command manager 的局部 command 是 piecewise constant，不能
   代替 waypoint/local path controller；必须同时输出 commanded world tangent 和转换后的
   body command，才能定位坐标/控制器问题。

在这些风险消除、Test/Acceptance Agent 对 route evaluator 给出 PASS 前，不启动 2048-env
长训练。
