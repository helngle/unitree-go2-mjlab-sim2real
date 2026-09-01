# Unitree Go2 Sim2Real 训练就绪多智能体协同指令

下面整段可直接作为新一轮主智能体指令使用。

---
你是主智能体，负责把当前 Unitree Go2 rough-terrain locomotion 项目推进到
**SIM2REAL_TRAINING_READY**。本轮不追求证明旧 stance-slip reward 有效，也不启动正式
PPO；本轮的唯一终点是：下一步可以按一个冻结、可复现、面向真实原版 Go2 的命令开始
训练。必须真正组织子智能体并行审计，在达到终点或留下不可绕过的硬阻塞证据前不得停止。

## 1. 工作区和环境

```text
工作区：/home/jensen/projects/unitree_rl_mjlab
Conda：conda activate unitree_rl_mjlab
当前分支：exp/high-slope-probe-integration
已知 HEAD：0a204b645a2325cb06264725c58cc5745da64a43
```

工作区已有大量用户未提交修改和多个 worktree。必须保留全部现状；禁止 `reset`、
`checkout`、`clean`、切换分支、删除用户文件或擅自 commit。开始和结束都要记录当前
branch、HEAD、`git status`、`git diff --check`、全部 worktree、GPU 和相关进程。发现并发
训练或评估时，不得启动重复任务。

当前仿真基线：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

该模型只可作为仿真基线或 teacher 候选，不能称为 hardware-ready。其 actor 为 234 维，
包含约 187 维 MuJoCo raycast `height_scan`；当前 Go2 部署 YAML 只有 47 维本体感知输入，
二者不能直接部署对接。

已拒绝的 stance-slip run：

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter
selection_status = NO_SAFE_SURVIVOR
```

不得把其中 `model_13700.pt`、`model_13800.pt`、`model_13900.pt` 或 `model_13999.pt`
恢复为默认模型、部署模型或新训练 warm start。旧诊断中的 `DO_NOT_TRAIN` 是对继续追加
stance-slip 变量的结论，不是禁止设计一个新的、明确标记为 sim2real 系统基线的任务。

## 2. 最终目标和能力边界

最终目标是把训练策略部署到一台保持原厂机械能力的 Unitree Go2，在复杂环境中稳定行走。
当前没有实机，因此本轮最多给出：

```text
SIM2REAL_TRAINING_READY
```

绝不能给出 `HARDWARE_READY`、`REAL_ROBOT_VALIDATED` 或类似结论。只有拿到明确的 Go2
型号、固件、低层控制权限并完成台架和实机渐进测试后，才可能解除该边界。

官方开源模型和参数是可信基线。不得为了通过仿真而降低质量、放大足端、提高额定力矩/
速度/功率、固定异常高摩擦或改造 stock Go2 的机械能力。允许修复的是资产转换、碰撞体、
关节/坐标约定、接触求解、SDK 映射、可部署观测、时延/噪声建模和安全运行时的一致性。

这轮是系统设计和训练就绪工程，不是单变量因果实验。可以为新的 sim2real task 同时完善
观测边界、随机化和运行时，但必须逐项登记来源与理由、做消融/单元测试，并在正式训练前
整体冻结。不得把多项系统改动伪装成一个 reward 的因果结论。

## 3. 多智能体组织

主智能体必须先创建 3 个子智能体。并发槽不足时采用两批，但职责不得省略。子智能体默认
只读审计并提交带文件/行号、命令输出摘要和证据强度的报告；主智能体统一实施代码修改，
防止共享脏工作区冲突。确需让子智能体编辑时，只能分配互不重叠的文件，并在编辑前声明
所有权。

### 子智能体 A：官方资产与物理身份审计

对照以下官方仓库的当前可复现 commit，而不是只看博客或二手配置：

```text
https://github.com/unitreerobotics/unitree_mujoco
https://github.com/unitreerobotics/unitree_rl_gym
https://github.com/unitreerobotics/unitree_rl_lab
https://github.com/unitreerobotics/unitree_sdk2
https://github.com/unitreerobotics/unitree_sdk2_python
https://github.com/unitreerobotics/unitree_ros2
```

记录 URL、commit SHA、访问时间和许可证。逐项比较本仓库 Go2 资产与官方来源：link/joint
名称、SDK 顺序、轴和符号、零位、range、质量、质心、惯量、碰撞 geom、足端 site、gear/
effort/velocity 限制、默认姿态、IMU frame、重力和坐标系。区分“官方明确值”“由格式转换
推导”“本项目经验配置”“官方未公开”。不得臆造完整电机动力学、固件时延或安全行为。

输出：差异表、风险等级、建议修复项、禁止修改项，以及 stock-Go2 identity manifest。

### 子智能体 B：可部署观测、动作与 SDK 运行时审计

建立从训练 tensor 到 `unitree_sdk2` LowCmd 的端到端合同，至少覆盖：

- actor 每个 observation term 的名称、顺序、维数、单位、frame、scale、clip、noise、历史；
- 训练 234 维 V7、部署 YAML 47 维和 ONNX 实际输入 shape 的机器可读对比；
- `height_scan` 在真实机器人上的来源。若没有经过验证的真实传感器等价管线，不得进入
  actor；可保留给 critic/teacher 作为 privileged observation；
- 12 维 action 的 joint 顺序、SDK `joint_ids_map`、正负号、默认零位、`0.25 rad` scale、
  clip、position target、20 ms control step、PD 增益和 effort limit；
- LowState 时间戳/新鲜度、IMU 定义、关节编码器、last-action 语义、命令来源、推理频率、
  action hold/slew、丢包和通信时延；
- PyTorch policy、导出 ONNX、C++ inference、`unitree_mujoco` bridge 的数值一致性；
- 安全状态机：mode handoff、姿态/关节/速度/动作限制、NaN/Inf、观测过期、通信超时、
  emergency damping/passive fallback 和人工急停。

在当前没有实机和型号信息时，默认实现与额外地形传感器无关的 proprioceptive actor；
`height_scan` 只给 asymmetric critic/teacher。只有找到官方、可复现且能在目标 Go2 SKU
实时生成相同空间语义的管线，才允许改选 perceptive actor，并必须把该硬件依赖写进 gate。

输出：observation/action schema、映射断言、ONNX parity 方案、mock SDK/仿真部署测试方案和
未拿到实机前不能关闭的风险。

### 子智能体 C：训练鲁棒性、课程与多目标验收设计

参考官方 Unitree 训练仓库以及有可追溯实现的 legged-locomotion sim2real 工作，审计现有
randomization 和 curriculum。重点检查 foot friction、各 link mass/inertia/COM、payload、
全部 12 个电机 strength/offset、PD、encoder/IMU noise、observation/action latency、action
hold、外力扰动和地形分布；核对当前 `motor_strength` 是否错误地只覆盖少数 actuator。

随机化范围必须有物理依据并覆盖 nominal，不能通过过宽随机化掩盖配置错误。摩擦 `1.2`
曾是 MuJoCo evaluation-only 因果探针，不得未经新合同审查直接固定为训练/部署值。

设计一个冻结后可执行的训练阶段和 checkpoint 验收。多目标采用“硬约束优先、再词典序
选择”，不得事后构造加权总分。目标至少包括：

1. 平地速度跟踪、站立和转向；
2. ordinary rough、连续起伏、离散障碍、坡地和楼梯；
3. line、arc、S-curve 路径保持；
4. fall/base/upper-leg/calf contact、关节/力矩/速度/动作边界；
5. terrain-tangent slip、pitch、action acceleration、energy/effort 和 failure risk；
6. 对时延、摩擦、质量、COM、电机强度、传感噪声的 randomized robustness；
7. PyTorch/ONNX/C++/unitree_mujoco 一致性和完整 provenance。

输出：初始化策略比较（从头训练、V7 teacher-student/distillation、兼容层部分迁移），明确
选择一种；给出固定 task、seed、env 数、iterations、save interval、训练命令、阶段 gate、
checkpoint 选择规则和预计资源。新 actor 输入维数不同，禁止假装可以 full-resume V7 的
actor/optimizer；若做权重迁移，必须精确登记哪些 tensor 被加载、哪些重新初始化。

## 4. 主智能体集成决策

收齐三份报告后，主智能体必须交叉核验，不得机械拼接。至少做以下决定并说明证据：

1. **Robot identity**：冻结哪个官方 commit/asset 为 nominal；本地差异是正确适配、需修复
   还是信息未知。
2. **Actor observation**：默认应是 47 维左右的 deployable proprioception；critic 可继续
   使用 height scan/contact/base velocity 等 privileged 信息。实际维数以运行时导出为准，
   不得硬编码猜测。
3. **Initialization**：新 actor shape 与 V7 不兼容时，选择从头训练或有严格映射的
   teacher-student/partial initialization；禁止静默跳过 shape mismatch。
4. **Action/runtime**：冻结 joint mapping、frame/sign/zero、action scale、PD、limits、control
   dt 和超时行为；nominal 必须保持 stock Go2 能力。
5. **Randomization**：只保留有来源、通过实现覆盖测试且不改变 nominal robot identity 的
   参数；所有 12 个关节和正确 body 必须覆盖。
6. **Training task**：创建一个新名字，避免污染 `Unitree-Go2-Rough-V7` 和已拒绝的
   `V7-StanceSlip`。冻结所有配置后生成唯一正式训练命令。

若三方证据冲突，主智能体先用最小测试或官方源解决；无法解决时采用更保守、无需额外
硬件的训练方案，并把未知项作为后续 hardware gate，而不是无限期阻塞仿真训练。

## 5. 必须实施的训练前产物

本轮不能只写分析报告。主智能体必须完成必要代码和测试，使新 task 真正可运行：

- 一个独立注册的 sim2real rough task；
- deployable actor 与 privileged critic 的明确 observation 配置；
- 与 actor 完全一致的 Go2 deploy YAML/schema；
- 训练/部署共享或自动比对的 joint/action metadata，避免双份配置静默漂移；
- physically grounded domain randomization，包含必要的 latency/hold/noise，并验证覆盖；
- policy export 与 ONNX metadata/schema；
- C++/mock LowState 到 observation、action 到 LowCmd 的无硬件测试；
- `unitree_mujoco` 中的 nominal sim-deploy smoke；
- 一个不调用 `learn` 的真实 GPU preflight；
- 训练合同、验收合同和机器可读 readiness artifact。

不得修改旧 checkpoint。不得改变默认部署模型。不得在本轮启动正式 PPO。允许编译、导出
临时未训练 policy、运行 CPU/GPU smoke 和 `unitree_mujoco` 仿真部署；临时产物必须明确
标记，不能冒充训练模型。

## 6. 硬门槛

只有以下全部通过，才可宣告 `SIM2REAL_TRAINING_READY=true`：

### G0 Provenance

- branch/HEAD/diff/worktree、官方 repo commit、asset/config/source SHA 完整；
- 默认 V7 checkpoint SHA 与登记值一致；
- 被拒绝 checkpoint 未被引用为默认或 warm start；
- JSON 严格 finite，命令、环境、时间、硬件和软件版本已记录。

### G1 Stock Go2 identity

- 12 DOF link/joint/order/axis/sign/zero/range 与官方来源逐项核对；
- mass/COM/inertia/collision/foot/IMU/actuator nominal 差异都有解释；
- 未提高机械能力，未知固件/动力学明确标注而非猜测。

### G2 Deployable observation

- actor 不依赖 MuJoCo-only state、contact truth、raycast truth 或未实现的真实 height map；
- train/eval/export/deploy 的 term、顺序、shape、scale、frame、history 完全一致；
- critic privileged 信息不会进入 actor ONNX；
- observation schema 自动校验和故意错序/缺项的负向测试通过。

### G3 SDK-compatible action/runtime

- 12 关节 mapping、sign、zero、scale、PD、limit、20 ms step 做 round-trip 测试；
- nominal action 在 joint/effort/velocity 安全范围内；
- stale state、NaN/Inf、timeout、越界动作和 mode transition 均进入安全 fallback；
- PyTorch 与 ONNX 输出在预登记 tolerance 内，C++ 处理前后无未解释漂移。

### G4 Physically defensible robustness

- 每个 randomization 的来源、nominal、范围、分布和作用对象已记录；
- 全 12 actuator 覆盖、body/link 覆盖和实际 runtime sampling 已测试；
- observation/action delay、noise、action hold 不改变 20 ms 外部控制合同；
- nominal 与 randomized smoke 均 finite，无 reset storm 或配置失效。

### G5 Task and optimization contract

- task 可通过 registry/config/CLI 构造；actor/critic/action shapes 与合同一致；
- 初始化方式真实可执行，checkpoint/optimizer shape mismatch 被显式处理；
- 训练期间所有变量冻结，checkpoint 选择和多目标 guardrail 预先登记；
- 不以 TensorBoard reward 或 final checkpoint 单独选模型。

### G6 No-learning full-scale preflight

- 使用计划中的 env 数、seed、task、初始化路径构造真实 runner；
- 完成 reset 和至少 8 个 control steps，observation/action/reward/metrics 全 finite；
- `learn_called=false`，无 OOM、shape、export、resume 或 telemetry 错误；
- preflight artifact、日志和 SHA 完整，退出后 GPU 无残留任务。

### G7 Sim-deploy and safety

- 未训练/临时策略只能用于 wiring smoke，并明确标记；
- mock SDK 与 `unitree_mujoco` 至少验证 observation/action 周期、mapping、timeout 和 fallback；
- 没有实机时相关 gate 只能标记 `HARDWARE_PENDING`，不能伪造通过；
- `HARDWARE_PENDING` 不阻止仿真训练，但必须阻止真实机器人部署。

## 7. 验证范围

根据改动运行并记录：

```text
定向 unit tests
相关完整 test suite
compileall / C++ build（若涉及）
task registry/config/CLI smoke
observation/action schema tests
PyTorch -> ONNX parity
mock SDK tests
unitree_mujoco nominal sim-deploy smoke
GPU no-learning full-scale preflight
recursive finite / SHA / provenance checks
git diff --check
```

GPU/MuJoCo-Warp 出现资源累积时，保持配置不变并采用每个独立场景一个进程，不能把不同
batch size、seed 或配置拼接成 matched 证据。遇到 NaN/Inf、OOM、错误映射、导出漂移或
安全 fallback 失效时，停止对应运行、保留证据、修复后从独立 invocation 重跑。

## 8. 训练命令和后续验收合同

本轮结束前必须登记一条唯一训练命令，但不执行。命令必须使用新 task，明确：

```text
num-envs、env/agent seed、初始化方式、iterations、save interval、logger、run name
```

若选择从头训练，命令中不得出现伪 resume；若选择 teacher-student 或 partial init，先提供
专用、已测试且写明 tensor 映射的加载入口。正式训练开始后不得临时改 reward、terrain、
curriculum、randomization、observation、network、action、actuator 或 PPO。

预登记 checkpoint 选择必须先执行硬安全约束，再按以下词典序：

1. retained flat/rough/obstacle/stairs/line/arc/S 无回归；
2. complex-terrain completion；
3. randomized completion；
4. forward/command tracking；
5. slip、pitch、action acceleration、effort/energy；
6. 若仍相同，选择更早 checkpoint。

新模型在完成仿真多目标验收、ONNX/C++ sim-deploy 验收和将来实机渐进验收前，不得替换
V7 仿真默认模型，更不得称为实机部署默认模型。

## 9. 记录和交付

至少创建或更新：

```text
docs/reviews/go2_sim2real_training_readiness.md
docs/reviews/go2_sim2real_training_contract.md
docs/PROJECT_JOURNAL.md
docs/HANDOFF.md
```

并生成机器可读 readiness JSON。最终报告必须明确：

- 3 个子智能体各自结论以及主智能体如何解决冲突；
- 官方资产与本地模型是否一致，哪些仍未知；
- actor/critic 最终 observation schema 和维数；
- joint/action/SDK/PD/control-rate 合同；
- randomization 和安全层的最终冻结内容；
- 初始化策略、唯一训练命令和预计训练资源；
- G0-G7 每项 PASS/FAIL/HARDWARE_PENDING 及证据路径/SHA；
- `SIM2REAL_TRAINING_READY` 是否为 true；
- 为什么仍不能声称 `HARDWARE_READY`；
- 默认模型是否变化（本轮应为否）；
- GPU 和相关进程最终状态。

## 10. 停止条件

只允许以下两种结束：

```text
A. SIM2REAL_TRAINING_READY=true
   G0-G6 全 PASS；G7 的无硬件部分 PASS，实机部分 HARDWARE_PENDING；
   新 task、代码、测试、preflight、唯一训练命令和验收合同均已落地。

B. BLOCKED_WITH_EVIDENCE
   存在无法用仓库、官方源、mock SDK 或 unitree_mujoco 解决的真实硬阻塞；
   已至少验证两条安全替代路径均不可行，并给出最小所需用户/硬件输入。
```

不能因为报告写完、子智能体返回、测试较多或需要做工程修改而提前停止。也不能为了按时
结束降低 gate。没有实机本身不是训练阻塞，只是 `HARDWARE_READY` 的硬边界。

---
