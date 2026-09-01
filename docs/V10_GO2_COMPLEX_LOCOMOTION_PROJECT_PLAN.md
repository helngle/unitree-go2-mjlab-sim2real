# Unitree Go2 V10 复杂地形行走、奔跑与 Sim2Real 项目方案

版本：`1.0`

建立日期：`2026-08-10`

当前状态：`PLAN_APPROVED_FOR_DESIGN / TRAINING_NOT_STARTED`

实时日志：[`V10_GO2_PROJECT_JOURNAL.md`](V10_GO2_PROJECT_JOURNAL.md)

历史日志：[`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md)，自本方案建立后作为只读历史证据库使用。

## 1. 项目最终目标

训练并最终部署一套适用于原厂机械能力 Unitree Go2 的分阶段四足运动系统：

1. 在平地和普通粗糙地形稳定行走、站立、转向和跟踪速度命令；
2. 在连续起伏、坡地、楼梯、离散障碍、低摩擦以及 line/arc/S 路线中稳定行走；
3. 在保留低速行走能力的基础上，在仿真中逐渐获得平地奔跑和复杂地形奔跑能力；
4. 将通过验收的行走+奔跑特权 Teacher 蒸馏为只使用实机可获得观测的最终 Student；
5. 所有仿真能力和 Student 验收完成后，最后执行导出、C++运行时和真实 Go2 的 Sim2Real；
6. 跳跃、沟壑和大型障碍等 parkour 能力作为独立的后续扩展，不插入本轮主路线。

论文和开源仓库是设计证据，不是复现目标。最终判断只由 Go2 的能力、安全性、可部署性和实机结果决定。

## 2. 当前项目边界与历史封存

### 2.1 历史基线

当前仿真基线仍为 V7：

```text
logs/rsl_rl/go2_velocity/
2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/
model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

V7 Actor 为 234 维：

```text
base angular velocity                 3
projected gravity                     3
velocity command                      3
global gait phase                     2
joint position                       12
joint velocity                       12
last action                          12
global height scan                  187
total                               234
```

V7 Critic 为 261 维，在 Actor 输入之外还包含 base linear velocity、foot height、air time、contact state 和 contact force。

V7、V8 privileged-linear-velocity、contact-force、stance-slip、foot-contact observability 和旧 proprioceptive Student 工作全部保留。它们只允许用于：

- 对照评估；
- 复用已经验证正确的基础设施；
- 查找失败机制；
- 提供验收基准；
- 避免重复已失败实验。

它们不得在没有新合同的情况下自动成为 V10 warm start、默认模型或继续训练入口。

### 2.2 为什么新建 V10 路线

既有证据表明，V7 是一次筛选得到的局部 Pareto checkpoint，而不是继续 PPO 就会单调改善的基础模型。扩展 Actor 输入的实现没有发现系统性切片、归一化或零初始化错误；共同退化更符合以下机制：

- 已筛选策略继续优化后发生策略漂移；
- fresh optimizer 丢失 Adam 状态并带来较强初始优化冲击；
- 训练分布的期望 reward 与高坡长路线、尾部安全硬门并不一致；
- V7 高坡失败还受到接触、摩擦、打滑和执行器饱和等物理边界影响；
- 单次 seed 和 MuJoCo-Warp 接触非确定性限制了单臂归因。

因此 V10 不再围绕 V7 逐个追加观测，而是重新共同设计观测、网络、地形课程、奖励、执行器模型、Teacher/Student 接口和验收体系。

## 3. 强制研究与批准规则

任何创新性 observation、预处理、reward、网络、记忆、估计器、loss、curriculum、Teacher/Student 接口、训练机制或验收机制，必须在实施前完成：

1. 至少一篇直接相关的原始论文或公开可审计 GitHub 实现；
2. 写清参考方法的输入、输出、时间语义和算法；
3. 写清本项目对 Go2、MJLab、MuJoCo、动作接口和传感器的所有实质差异；
4. GitHub 参考必须记录 URL、commit/tag、许可证和访问日期；
5. 获得用户对参考和差异的明确批准；
6. 然后才允许代码修改、GPU smoke、正式评估或训练。

找不到直接参考时，必须记录：

```text
NO_DIRECT_REFERENCE_DO_NOT_IMPLEMENT
```

项目日志和失败诊断可以触发文献检索，但不能替代参考证据。

## 4. 参考方法与项目映射

以下文献构成本项目的首版参考集。每个实际组件在实现前还要冻结准确版本和详细映射。

### 4.1 复杂地形 Teacher 与感知结构

Miki et al., *Learning robust perceptive locomotion for quadrupedal robots in the wild*：

- 论文：https://arxiv.org/abs/2201.08117
- 直接参考组件：proprioception、exteroception 和 privileged information 分支；Teacher/Student；时间历史；地形感知退化下的鲁棒运动。
- 参考输入：命令、机体状态、关节状态、动作历史、步态相位、每足多尺度地形高度和接触/摩擦/外力等特权量。
- 参考输出：关节级策略动作，经低层控制器执行。
- V10 差异：机器人改为 Go2；仿真改为本项目 MJLab/MuJoCo；动作固定为 12 维关节目标位置；地形编码的准确采样点、维数和传感器来源必须重新冻结。

Lee et al., *Learning quadrupedal locomotion over challenging terrain*：

- 论文：https://arxiv.org/abs/2010.11251
- 直接参考组件：特权 Teacher、接触/地形/摩擦信息、Student 适应、由易到难地形课程。
- 参考输出：关节目标，经关节控制器转为执行器命令。
- V10 差异：不直接复用其机器人、网络维数或 terrain representation；Go2 的接触力、执行器限制和部署传感器必须由本项目运行时合同确定。

### 4.2 大规模 PPO 与地形课程

Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning*：

- 论文：https://arxiv.org/abs/2109.11978
- GitHub：https://github.com/leggedrobotics/legged_gym
- 直接参考组件：大量并行环境、PPO、基于表现的地形课程和从简单运动到困难地形的推进。
- V10 差异：使用当前 MJLab/MuJoCo runner；课程不仅按地形难度，还必须约束 retained easy terrain 比例、路线类型、速度区间和安全回归。

### 4.3 Student 动力学适应

Kumar et al., *RMA: Rapid Motor Adaptation for Legged Robots*：

- 论文：https://arxiv.org/abs/2107.04034
- 直接参考组件：Teacher/base policy 使用环境特权变量；Student/adaptation module 根据近期本体历史估计环境潜变量；策略在质量、摩擦和电机变化下适应。
- 参考时间语义：部署时只使用截至当前时刻的因果历史，不得使用未来状态。
- V10 差异：环境潜变量集合、历史长度和编码维数重新预注册；Student 输出仍为与 Teacher 完全相同的 12 维关节目标；不默认继承论文的网络尺寸。

### 4.4 奔跑与可控步态

Margolis and Agrawal, *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior*：

- 论文：https://arxiv.org/abs/2212.03238
- 直接参考组件：用行为参数调节步态、抬脚、身体姿态和速度，使一个策略覆盖多种行走与奔跑行为。
- V10 使用阶段：V10-Walk Teacher 在仿真中通过后即可启用，不等待 Student 或 Sim2Real。
- V10 差异：行为参数及范围必须按 Go2 执行器能力、控制频率和项目地形重新设计；不能只提高速度命令冒充奔跑课程。

Wang et al., *Learning Robust, Agile, Natural Legged Locomotion Skills in the Wild*：

- 论文：https://arxiv.org/abs/2304.10888
- 参考用途：后续动态步态和运动风格学习，不作为首版 V10-Walk 的必需 loss。
- 边界：如果未来引入 motion prior、mentor 或模仿损失，必须再次单独通过参考门和用户批准。

### 4.5 Parkour 后续阶段

Cheng et al., *Extreme Parkour with Legged Robots*：https://arxiv.org/abs/2309.14341

Hoeller et al., *ANYmal Parkour*：https://arxiv.org/abs/2306.14874

这两篇只作为跳跃、沟壑、攀爬、多技能或层级策略阶段的直接候选参考。在行走和奔跑基础能力通过前，不引入其 parkour 训练目标。

## 5. 系统总体架构

```text
仿真真值 ───────────────> privileged encoder ───────┐
                                                    │
当前本体观测 ───────────> proprio encoder ──────────┼─> Teacher policy ─> 12 joint targets
                                                    │
地形真值/高度表示 ──────> terrain encoder ──────────┘

因果本体历史 ───────────> adaptation encoder ─> privileged-latent estimate ─┐
实机当前本体观测 ───────> proprio encoder ───────────────────────────────────┼─> Student policy
真实地形传感器（若存在）> deployable terrain encoder ───────────────────────┘

Teacher/Student action ─> scale/clip/safety ─> target joint position ─> PD/actuator ─> Go2
```

关键约束：

- Teacher 和 Student 的动作含义、关节顺序、缩放、控制周期和 PD 接口完全一致；
- Student Actor 不得读取仿真真值、接触真值或未在实机实现的 raycast；
- Critic 可以使用特权信息，但不得进入导出的 Student Actor；
- Teacher 的特权 latent 必须是 Student 能通过因果历史或真实传感器估计的目标；
- 所有编码器接口、维数和时序在正式训练前机器可读冻结。

## 6. V10-Walk Teacher 设计

### 6.1 动作接口

首版统一输出：

```text
12-D normalized action
    -> per-joint scale
    -> default joint position offset
    -> joint target position [rad]
    -> PD / actuator model
```

不把 Actor 输出改为机体位姿或关节速度。这样保持当前 Go2 训练、导出、C++部署和低层接口的一致性。

### 6.2 Actor 信息分组

正式维数尚未冻结，先冻结语义边界：

1. **deployable proprioception**
   - 用户速度和转向命令；
   - IMU角速度；
   - projected gravity 或等价姿态量；
   - 12个关节位置；
   - 12个关节速度；
   - 上一时刻12维动作；
   - 参考论文支持且经批准的步态相位/行为命令。
2. **causal history**
   - 只包含当前及过去时刻；
   - 用于估计动力学、接触变化和外部扰动；
   - 历史长度、采样频率和网络类型必须在实现前找到直接参考并冻结。
3. **terrain representation**
   - Teacher 可使用仿真地形真值；
   - 优先研究以四足落脚区域为中心的多尺度表示；
   - Student 是否使用深度相机/雷达，取决于目标 Go2 SKU 和真实传感器管线审计；
   - 没有可复现真实等价管线时，仿真 height truth 不得进入 Student Actor。
4. **privileged environment information**
   - base linear velocity；
   - 足端接触状态、接触力和地形法向；
   - 摩擦、质量、质心、惯量和电机参数；
   - 外力、外力矩、air time 和非足端身体碰撞；
   - 仅允许出现在 Teacher/critic 或被 Student 估计的 latent 中。

### 6.3 网络原则

- 各信息组先独立编码，再融合进入策略 MLP；
- Teacher actor、Student actor 和 critic 分离，不共享可能泄漏特权信息的张量；
- 新模型默认从随机初始化训练，不继承 V7 optimizer；
- 是否迁移 V7 的个别编码层只能作为以后独立、引用充分的初始化实验；
- 使用 asymmetric actor-critic：critic 可拥有完整仿真状态；
- 网络尺寸、激活、初始化、PPO超参数和归一化在预检前冻结。

## 7. 地形、命令和能力课程

### Stage W0：基础运动

- 站立；
- 平地前进、后退、横移和转向；
- 低速命令；
- 轻微外部扰动；
- 目标是动作接口、步态和基本稳定性正确。

### Stage W1：普通复杂地形

- ordinary rough；
- 低到中等连续坡；
- 小台阶和离散障碍；
- clean 与 physically grounded randomized profiles；
- 始终保留平地和低难度回放，避免遗忘。

### Stage W2：复杂路线与高难地形

- 高坡与连续起伏；
- 楼梯；
- line、arc、S-curve；
- 摩擦变化和外部扰动；
- 组合地形，但不能在未完成单场景能力定位前用混合分布掩盖失败。

### Stage W3：Teacher 稳定化

- 训练变量冻结；
- 只保存预登记 checkpoint；
- 多 seed 筛选；
- 只允许在所有 hard gate 通过的 checkpoint 中词典序选择；
- final checkpoint 和 TensorBoard reward 都没有自动优先权。

课程推进应由成功率、前进增益和安全状态共同决定。不能只按 reward 或 terrain level 提升难度，也不能通过提高极端地形比例让策略遗忘常规场景。

## 8. Reward 设计原则

首版 reward 由已有文献组件和项目已验证指标构成，具体公式和权重必须在训练合同中冻结：

- 速度/角速度命令跟踪；
- 路线前进和有效 progress；
- 身体姿态与方向稳定；
- 足端滑移；
- 非足端身体碰撞；
- 关节位置、速度、加速度和目标变化；
- 执行器 effort、energy、torque-speed/饱和边界；
- 存活和终止；
- 参考支持的 gait/foot-air-time 项。

设计约束：

- 防止策略以几乎不前进换取低风险；
- 防止策略以冲撞和高打滑换取短期 progress；
- 高频惩罚不能压制必要的动态步态；
- 稀有高坡和尾部安全必须进入训练采样或约束，而不能只存在于验收；
- 每项 reward 都要记录激活条件、量纲、裁剪、权重和预期副作用；
- 新 reward 不得只凭当前项目日志直接实现。

## 9. 物理模型与 Domain Randomization

Nominal robot 必须保持 stock Go2 能力。训练前审计并冻结：

- 12个关节的名称、顺序、轴、符号、零位和范围；
- link质量、质心、惯量和碰撞体；
- 足端几何、接触参数和 IMU frame；
- joint effort/velocity limit；
- torque-speed、PD、动作scale和控制周期；
- SDK joint mapping 和 LowCmd 语义。

Randomization 候选包括：

- 地面摩擦；
- link质量、payload、质心和惯量；
- 12个电机强度、位置偏置和 PD；
- IMU与编码器噪声、bias和丢帧；
- observation/action latency、action hold和jitter；
- 外力与外力矩；
- 地形测量噪声和传感器 dropout。

每一个范围必须有官方资料、测量结果或直接论文/开源实现依据，包含 nominal 值，且必须测试确实作用到正确的全部 actuator/body。随机化不能掩盖错误的 nominal asset。

## 10. Teacher 验收与 V10-Walk 晋级条件

正式数值将在 evaluator 审计后写入机器合同。当前顶层 hard gate 为：

### 10.1 基础与保留能力

- flat、ordinary rough、continuous terrain、obstacle、stairs、standing、turning 均有 clean/randomized 分组；
- retained scene 不得低于锁定 V7 同场景完成率超过 `0.05`；
- 不允许通过牺牲站立、转向或普通地形换取高坡能力。

### 10.2 高坡和路线能力

- line、arc、S-curve 分开统计；
- clean 每路线完成至少 `12/16`；
- randomized 每路线完成至少 `10/16`；
- moving-forward mean forward gain 至少 `0.80`；
- high-slope completion 不低于锁定 V7，且必须有绝对能力提升。

### 10.3 安全

每个正式 group 检查：

- slip；
- absolute pitch；
- action acceleration；
- effort/energy和饱和；
- base、upper-leg、calf contact；
- fall/failure risk；
- action fault、joint-target fault和non-finite。

候选相对 V7 的同场景 safety 指标原则上不得超过 `1.2x`；V7为零的 fault 指标仍要求精确为零。

### 10.4 选择规则

先过 hard gate，再按以下词典序选择：

1. 最低 randomized complex-terrain completion；
2. 最低 clean complex-terrain completion；
3. 最低 retained-scene completion；
4. 最低 line/arc/S completion；
5. 最低 forward gain；
6. failure、slip、contact、effort、action acceleration和pitch；
7. 完全相同时选择更早 checkpoint。

禁止训练后修改阈值、挑 final、跨 update 拼接或构造事后加权总分。

## 11. 奔跑升级路线

V10-Walk Teacher 在完整仿真验收中通过后，即可开始 V10-Run Teacher；不等待 Student 蒸馏，也不进行任何实机 Sim2Real。

### R0：平地动态步态 Teacher

- 从已通过的 V10-Walk Teacher 能力出发；
- 保留低速行走命令和行走地形回放；
- 引入有直接参考的步频、步态、抬脚高度、身体高度或行为latent；
- 速度逐级从行走区间增加；
- 加强 torque-speed、impact、slip、energy和热风险监控。

### R1：复杂地形奔跑 Teacher

- ordinary rough；
- 坡地；
- line、arc和S-curve；
- 最后才加入障碍组合；
- 每个阶段继续执行 retained walking capability gate。

### R2：行走+奔跑统一 Teacher 验收

- 低速站立、行走、转向不得遗忘；
- 平地奔跑和复杂地形奔跑分别通过能力与安全门；
- 只在完整仿真验收后冻结最终 Teacher；
- 训练阶段不执行真实机器人部署。

每个奔跑阶段都必须回放低速行走并执行 retained capability gate，禁止以遗忘行走换取高速。

## 12. 最终 Teacher-Student 路线

默认只在行走+奔跑统一 Teacher 完整通过后进行最终 Student 蒸馏，避免分别为行走和奔跑重复建立两套部署模型。若以后需要中间 Walking Student，它只能作为单独批准的仿真诊断分支，不改变主路线。

### 12.1 Student 输入边界

Student 只使用目标实机可实时获得的：

- IMU；
- 关节位置和速度；
- 当前命令；
- last action；
- 因果历史；
- 经真实管线验证的深度/雷达地形表示（若目标硬件具备）。

接触真值、仿真 raycast、精确摩擦、外力、质量和电机真值均不得进入部署 Actor。

### 12.2 蒸馏过程

预期分为：

1. 覆盖行走与奔跑任务的 Teacher rollout supervised distillation；
2. privileged/terrain latent reconstruction；
3. Student 自己闭环运行时由 Teacher 重新标注的 DAgger 式迭代；
4. 同一任务分布下的受控 RL fine-tuning；
5. Student 对行走、奔跑和全部 retained ability 的独立仿真验收。

上述每一种 loss 和数据采样机制都要在实现前完成直接参考与用户批准。不能因为 Teacher 通过就假定 Student 自动通过。

### 12.3 Student 验收

- 与 Teacher 完全相同的12维动作接口；
- Actor导出不包含 privileged tensor；
- 同场景 Teacher-Student performance gap 受控；
- walking、running、retained、complex和safety hard gate独立通过；
- 仿真中的传感噪声、延迟和dropout鲁棒性通过；
- 通过后冻结唯一最终 Student，随后才进入 Sim2Real。

## 13. 最终 Sim2Real 路线

Sim2Real 是本项目主路线的最后一步。只有行走+奔跑 Teacher 和最终 Student 的全部仿真验收通过后，才开始部署链和真实 Go2 测试。

为了避免最终无法部署，训练设计阶段仍会冻结可部署观测、动作、关节顺序和控制周期；但这只是接口兼容设计，不提前进行实机迁移或以实机结果阻塞奔跑 Teacher 训练。

### 13.1 最终一致性

- stock Go2 asset identity；
- train/eval/export/deploy observation schema 单一来源；
- 12关节 mapping、sign、zero、scale、PD和limit round-trip；
- 20 ms 或最终冻结控制周期全链路一致；
- observation/action timestamp和last-action语义一致；
- torque、velocity、temperature和power边界保守处理。

### 13.2 软件链验证

```text
final Student PyTorch policy
  -> ONNX export and metadata
  -> C++ observation builder
  -> inference
  -> action safety
  -> Unitree SDK2 LowCmd
  -> unitree_mujoco closed loop
```

必须覆盖 stale state、通信超时、NaN/Inf、动作越界、姿态异常、关节异常和人工急停，并进入 damping/passive fallback。

### 13.3 实机逐级开放

1. 无负载软件回放和命令映射；
2. 吊架/台架站立；
3. 低速平地直行；
4. 停止、转向和扰动恢复；
5. 轻微起伏与低坡；
6. 普通粗糙地形；
7. 楼梯和复杂路线；
8. 平地低速奔跑；
9. 平地目标速度奔跑；
10. 复杂地形奔跑。

任何阶段出现关节、热、功率、通信、跌倒或观测异常，都退回上一阶段。没有真实 Go2 测试时，只能给出 `SIMULATION_ACCEPTED` 或 `SIM2REAL_TRAINING_READY`，不得标记 `HARDWARE_READY`。

## 14. 项目阶段与交付物

### M0：方案与历史封存

- 本方案；
- 新实时日志；
- 历史方案只读；
- 状态：本文件建立时完成。

### M1：V10 训练前技术设计

- 官方 Go2 asset/SDK身份审计；
- 目标硬件和传感器清单；
- 文献/GitHub版本与许可证登记；
- observation/action/encoder schema；
- terrain、reward、curriculum和randomization合同；
- 验收矩阵和资源预算；
- 用户批准。

### M2：实现与无学习预检

- 新 task 和新 config；
- 定向单元测试；
- 32-env optimizer smoke；
- planned-env no-learning preflight；
- finite、shape、timing、足序和provenance通过；
- 不产生正式 checkpoint。

### M3：V10-Walk Teacher 训练与验收

- 从随机初始化开始；
- 多阶段 curriculum；
- 固定 checkpoint schedule；
- 多 seed 正式筛选与完整 acceptance；
- 选出一个 Teacher 或得到 `NO_TEACHER_SURVIVOR`。

### M4：V10-Run Teacher 训练与验收

- 从已通过的V10-Walk能力升级；
- 平地动态步态；
- 复杂地形奔跑；
- retained walking + running acceptance；
- 冻结行走+奔跑统一Teacher。

### M5：最终 Student 蒸馏与仿真验收

- 使用冻结的行走+奔跑Teacher；
- 蒸馏、因果适应和必要微调；
- 行走、奔跑、retained与安全矩阵全部通过；
- 得到唯一final Student simulation candidate。

### M6：最后执行 Go2 Sim2Real

- final Student导出与软件链验证；
- 实机配置和校准；
- 安全状态机；
- 从站立、行走到奔跑逐级实机测试；
- 最终得到hardware-validated walking/running policy。

## 15. 训练和实验停止条件

以下任一情况立即停止当前阶段并保留全部证据：

- 参考门未通过；
- 用户尚未批准实质差异；
- source/config/schema/checkpoint SHA不一致；
- observation/action顺序、frame、单位或时序不明；
- NaN/Inf、OOM、simulator错误或reset storm；
- optimizer/checkpoint/environment状态不符合合同；
- 训练源码或依赖在正式run中漂移；
- telemetry、TensorBoard或checkpoint缺失；
- 并发训练或GPU资源冲突；
- Student读取不可部署特权信息；
- 实机安全fallback、急停或通信freshness未验证。

纯技术失败只有在配置和合同字节级一致时才允许重试。模型能力失败不得通过临时改reward、追加输入、补训或挑选未登记checkpoint修复。

## 16. 日志、版本和 Git 规则

- 所有新工作记录在 `docs/V10_GO2_PROJECT_JOURNAL.md`；
- 日志按时间追加，不重写失败记录；
- 每次训练记录命令、branch、HEAD、git status、配置SHA、checkpoint、GPU、开始/结束时间和结论；
- 每个正式 artifact 不覆盖写入并登记SHA256；
- 旧 `docs/PROJECT_JOURNAL.md` 作为历史档案保留；
- 需要引用旧结果时，在新日志记录旧文件路径、用途和结论边界；
- 工作区现有未提交修改归用户所有，不允许reset、checkout、clean或擅自覆盖；
- Git提交只纳入经过确认的相关文件，禁止把大型raw/checkpoint和无关脏改动混入同一提交。

## 17. 当前立即执行顺序

本方案建立后，下一阶段不是直接开始长训练，而是：

1. 只读冻结目标 Go2 硬件、官方 asset、SDK和传感器能力；
2. 为 V10-Walk 选择每个组件的直接论文/GitHub参考并登记版本；
3. 输出准确 Actor/Teacher/Critic/Student 输入维度、网络和时间历史；
4. 输出 reward、terrain、curriculum、randomization和PPO机器合同；
5. 输出训练预算、checkpoint schedule和V10 acceptance矩阵；
6. 向用户说明全部实质差异并取得批准；
7. 才开始实现、smoke、preflight和正式训练。

截至 `2026-08-10`，本项目状态为：

```text
M0_COMPLETE
M1_DESIGN_READY_FOR_USER_APPROVAL
FORMAL_TRAINING_NOT_STARTED
DEFAULT_MODEL_UNCHANGED_V7
```
