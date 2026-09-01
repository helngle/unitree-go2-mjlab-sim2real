# Go2 本体感知 Student / Privileged Critic / V7 Teacher 多智能体指令

下面整段可直接交给新窗口中的主智能体执行。它是
`go2_sim2real_training_readiness_multi_agent_instruction.md` 的聚焦版本；若两者有歧义，
以本文件对 actor/critic/teacher 的约束为准。

---
你是主智能体，负责组织多个子智能体，把 Unitree Go2 rough-terrain locomotion 项目推进到
**PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY**。本轮必须实现并验证新的训练架构，但不启动正式
PPO。下一轮应能够直接运行一条冻结命令训练：

```text
deployable proprioceptive-history student actor
+ privileged asymmetric critic
+ frozen V7 height-scan teacher
```

不得停留在论文调研或计划阶段。只有训练任务、teacher/student 数据链、导出部署 schema、
测试和 full-scale no-learning preflight 全部落地后才能停止。

## 一、工作区与当前事实

```text
工作区：/home/jensen/projects/unitree_rl_mjlab
Conda：conda activate unitree_rl_mjlab
分支：exp/high-slope-probe-integration
已知 HEAD：0a204b645a2325cb06264725c58cc5745da64a43
```

工作区已有用户未提交修改和多个 worktree。必须全部保留。禁止 `reset`、`checkout`、
`clean`、切换分支、删除用户文件或擅自 commit。开始和结束都记录：

```text
git status / git diff / branch / HEAD
git worktree list
GPU 状态
train/evaluate/audit/play/TensorBoard/unitree_mujoco 进程
```

当前可靠仿真基线与唯一 teacher 候选：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
actor shape = 234
critic shape = 261
action shape = 12
```

V7 actor 输入包含 187 维 MuJoCo raycast `height_scan`。它可以作为训练期 teacher 和带感知
仿真上界，但不能称为实机可部署模型。

以下 stance-slip run 已正式拒绝：

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter
selection_status = NO_SAFE_SURVIVOR
```

不得将其中任何 checkpoint 用作 warm start、teacher、候选或默认模型。旧
`next_training_action=DO_NOT_TRAIN` 仅禁止继续追加 stance-slip 变量，不禁止创建新的
proprioceptive sim2real task。

## 二、冻结的系统边界

### 1. Student actor：真机可部署

Student actor 禁止读取以下信息：

- `height_scan` 或任何 MuJoCo raycast/terrain ground truth；
- 真实 base linear velocity；
- 真实 foot contact/contact force；
- terrain id、difficulty、friction、mass、COM、motor strength 等仿真真值；
- future command、future state 或 evaluator route truth。

Student 只允许读取能由 Go2 SDK 或上层命令实时构造的信息：

```text
当前 base angular velocity                 3
当前 projected gravity                    3
当前 vx/vy/yaw command                    3
当前 gait phase sin/cos                   2
当前 12 joint position relative to nominal 12
当前 12 joint velocity                    12
当前 previous action                      12
--------------------------------------------
单帧基础 schema                           47
```

允许加入过去若干控制帧的 deployable proprioceptive history，但 history 中也不得混入任何
privileged term。历史窗口长度、是否只堆叠 42 维本体项或完整 47 维、flatten 顺序和时间方向
必须由证据决定后冻结，并在 Python、ONNX、C++ deploy YAML 中使用同一机器可读 schema。

不得把“直接删除 187 维后保持单帧 MLP”当作默认答案。至少比较以下两类可部署方案：

```text
A. feed-forward actor + fixed proprioceptive frame stack
B. recurrent/adaptation encoder + fixed-length proprioceptive history
```

比较只允许使用部署复杂度、参考实现、参数量、推理时间、schema 风险和 no-learning/synthetic
smoke；不得为了选择架构先跑多组正式 PPO。选择最小、可稳定导出并足以覆盖控制时延与短时
接触响应的方案。最终训练只能保留一个 student 架构。

### 2. Privileged critic：只在训练中存在

Critic 可读取 student 当前输入，并可额外读取：

```text
height_scan
base linear velocity
foot height / air time / contact / contact forces
terrain/friction/mass/COM/motor-strength/latency randomization truth
```

每项是否采用必须登记用途和维数。Critic 只能输出 value，不得向 actor ONNX 泄漏 privileged
tensor、latent 或 normalization state。导出后必须证明 ONNX graph 不含这些输入。

### 3. Frozen V7 teacher：只提供训练指导

Teacher 固定为上述 V7 `model_13600.pt` 和登记 SHA。Teacher：

- 读取原 V7 的 234 维输入，包括 height scan；
- 使用 deterministic inference action；
- 不更新参数、不恢复 optimizer、不改变 normalization；
- 不直接成为 student actor 的输入分支；
- 不出现在最终 ONNX/C++ runtime 中。

Teacher 的作用是提供 retained terrain 上的动作/latent 行为参考和更好的训练初始方向，不能
成为硬行为上限。V7 在持续高坡仍有明显缺陷，因此禁止用过强的全程 imitation loss 把
student 锁死在 V7。主智能体必须在正式训练前选择并冻结一种方式：

```text
A. teacher rollout dataset -> student behavior-cloning initialization -> PPO
B. PPO early-stage auxiliary distillation with preregistered decay -> pure PPO
C. 先 A 后短期 B，但必须证明不是两个可自由调参的实验 arm
```

不得把 V7 checkpoint 作为 shape 不兼容 student 的假 full-resume。任何部分权重迁移都要
逐 tensor 登记 loaded/reinitialized/skipped，并证明没有静默 mismatch。

### 4. 输出与 stock Go2 能力

Student 输出仍为 12 维 normalized joint-position action，仿真顺序冻结为：

```text
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf
```

动作语义继续是：

```text
joint_target = default_joint_position + 0.25 * action
control_dt = 0.02 s
hip/thigh PD = 20/1, effort limit = 23.5 Nm
calf PD = 40/2, effort limit = 45 Nm
```

先与 Unitree 官方资产和 SDK2 核验，再冻结 nominal。禁止为通过仿真而增加额定力矩/速度/
功率、降低质量、放大足端、固定异常高摩擦或改变 stock Go2 机械能力。

## 三、多智能体组织

主智能体必须创建 3 个子智能体。子智能体第一阶段只读审计并回传证据，主智能体统一修改
共享工作区；确需并行编辑时必须分配互不重叠文件并声明所有权。

### 子智能体 A：Student observation/history 与部署 schema

职责：

1. 逐项确认当前 V7 actor 234 维布局和 Go2 deploy YAML 47 维布局。
2. 审计 SDK2 LowState 能否构造每个 student term，包括单位、frame、sign、scale、noise、
   timestamp 和 initialization。
3. 比较 frame stack 与 recurrent/adaptation encoder，选择一个最终方案和固定 history window。
4. 定义 history reset、warmup、丢帧、重复帧、时序方向、last-action 和 command change 语义。
5. 实现或设计一个训练/导出/C++ 共用的 machine-readable observation schema。
6. 设计故意错序、缺帧、过期状态、维数漂移和 normalization drift 的负向测试。
7. 给出 actor 参数量、50 Hz 推理预算和 ONNX 导出风险。

参考至少包括：

```text
unitreerobotics/unitree_rl_mjlab
leggedrobotics/legged_gym
RMA: arXiv 2107.04034
DreamWaQ: arXiv 2301.10602
```

输出：`student_schema_report`、推荐架构、准确 actor 输入 shape 和证据强度。

### 子智能体 B：Asymmetric critic 与 teacher distillation

职责：

1. 载入并锁定 V7 teacher，复核 checkpoint SHA、actor normalizer、234 维 term 顺序和 action
   顺序。
2. 定义同一仿真状态下 teacher observation 与 student history observation 的构造时点，禁止
   one-step/off-by-one 泄漏。
3. 比较 BC initialization、decaying auxiliary distillation 和组合方案，选择一个固定流程。
4. 预登记 teacher loss 的作用阶段、归一化、mask、schedule 和停止条件；不得训练中临时调。
5. 明确哪些 terrain/command 使用 teacher，避免在 V7 已知失败的高坡上强制模仿坏动作。
6. 定义 privileged critic schema，并证明 actor/ONNX 不接收 privileged tensor。
7. 运行最小 forward/backward synthetic smoke，验证 teacher frozen、student/critic gradient
   ownership、loss finite 和无 optimizer state 污染；不得产出候选模型。

输出：`teacher_critic_report`、固定初始化/蒸馏合同、critic shape 和泄漏负向测试。

### 子智能体 C：Sim2real robustness、多目标验收与训练命令

职责：

1. 审计现有 terrain curriculum、command distribution、reward、termination 和 domain
   randomization。
2. 重点核验 friction、全 link mass/inertia/COM、payload、全部 12 actuator strength/offset、
   PD、encoder/IMU noise、observation/action latency、action hold、丢包和 push。
3. 发现当前 randomization 只覆盖部分 actuator/body 时给出最小修复，不得用过宽范围掩盖
   nominal 配置错误。
4. 为新的 proprioceptive task 设计一次冻结训练，不做多权重/多 history/PPO sweep。
5. 预登记 checkpoint 硬约束和词典序选择；V7 是带 scan 的 upper reference，不得伪装成
   observation-matched baseline。
6. 给出唯一 task、seed、env 数、iterations、save interval、logger、run name、预计显存和
   耗时。

多目标至少覆盖：

```text
flat stand/velocity/turn
ordinary rough / continuous terrain / discrete obstacles
slope / stairs
line / arc / S-curve
completion / forward gain / tracking
slip / pitch / action acceleration / effort
base / upper-leg / calf contact / fall / failure risk
clean and randomized robustness
PyTorch / ONNX / C++ / unitree_mujoco parity
```

输出：`robustness_training_report`、冻结 training/acceptance contract 和唯一命令。

## 四、主智能体实施顺序

### 阶段 0：启动边界

- 记录 Git/worktree/GPU/process/checkpoint SHA。
- 确认没有相同 readiness 工作或正式训练正在运行。
- 保留全部用户修改。

### 阶段 1：三方并行只读审计

- 创建并行子智能体 A/B/C。
- 主智能体同时阅读训练、export、deploy、SDK bridge 和现有测试。
- 子智能体报告必须有 repo URL/commit、文件/行号和已验证/推断/未知分类。

### 阶段 2：冲突解决和预登记

主智能体收齐报告后必须冻结：

```text
student architecture and history length
student observation schema and exact shape
critic privileged schema and exact shape
teacher observation/action timing
distillation/initialization method and schedule
action/SDK mapping and control timing
domain randomization
task name and PPO configuration
checkpoint selection and acceptance gates
```

若证据冲突，先做最小 schema/runtime test；无法确定时选择更简单、无额外硬件依赖、可导出
的方案。不得通过正式 PPO sweep 来替代工程判断。

### 阶段 3：统一实施

主智能体必须实现：

- 新 task：建议注册名 `Unitree-Go2-Rough-Sim2Real-Proprio-V1`；
- student-only actor observation group；
- privileged critic observation group；
- frozen teacher loader 和明确的 BC/distillation data path；
- history buffer/reset/warmup/dropout 语义；
- 全 12 actuator 和正确 body 的 domain randomization；
- observation/action schema artifact；
- 与 student 完全一致的 Go2 deploy YAML；
- ONNX metadata 和 schema hash；
- mock LowState -> history -> actor -> action -> LowCmd 链路；
- timeout/NaN/stale observation/action limit/fallback 安全层；
- training readiness/preflight 脚本和定向测试。

不得修改旧 V7 task 语义或 checkpoint。不得改变当前默认模型。

### 阶段 4：训练前验证

至少执行：

```text
targeted unit tests
相关完整 Python test suite
compileall
C++ deploy build/tests（若涉及）
task registry/config/CLI smoke
teacher SHA/strict-load/deterministic/frozen tests
student history ordering/reset/warmup/drop-frame tests
privileged-leakage negative tests
single-batch forward/backward ownership/finite smoke
PyTorch -> ONNX numerical parity
mock SDK observation/action round-trip
unitree_mujoco nominal wiring smoke
full-scale GPU no-learning preflight
recursive finite/provenance/SHA checks
git diff --check
```

Full-scale preflight 必须使用正式计划中的 task、num-envs 和 seeds，构造 student、critic、
teacher、environment、runner，完成 reset 和至少 8 个 control steps。可以验证 teacher/student
forward 和 loss 构造，但必须满足：

```text
learn_called = false
optimizer_step_called = false
candidate_checkpoint_written = false
all observations/actions/rewards/loss inputs finite
GPU exit clean
```

## 五、训练前硬门槛

只有全部通过，才可设置 `PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY=true`：

### G0 Provenance

- Git/worktree/source/config/checkpoint/official-reference SHA 完整。
- V7 teacher SHA 与登记值一致。
- 被拒绝 checkpoint 未被引用。

### G1 Student deployability

- actor 无 height scan、contact truth、base linear velocity 或 dynamics truth。
- history 每个 term 都能由 SDK LowState/command/previous action 构造。
- Python/ONNX/C++ term、顺序、shape、scale、normalization、history 完全一致。

### G2 Critic/teacher isolation

- privileged critic 只输出 value，未进入 actor graph。
- teacher 参数和 normalization 冻结，teacher action timing 与 student state matched。
- 最终 actor ONNX 只有 student schema 输入，无 teacher/critic/terrain truth。

### G3 Initialization integrity

- 不假 full-resume shape 不兼容 V7 actor/optimizer。
- BC/distillation/partial-load 的 tensor 和 schedule 全部登记。
- teacher 已知失败场景不会通过硬 imitation 锁定 student。

### G4 Action/runtime identity

- 12 joint order、SDK map、sign、zero、0.25 scale、PD、limit、20 ms 合同 round-trip PASS。
- timeout、stale、NaN/Inf、越界和 mode handoff 进入安全 fallback。
- 未提高 stock Go2 机械能力。

### G5 Robustness contract

- randomization nominal/range/source/target 完整且覆盖全部正确对象。
- latency/noise/hold/dropout 在部署可重现的语义内。
- nominal/randomized smoke finite，无 reset storm。

### G6 Training contract

- 新 task 独立注册并可构造。
- student/critic/teacher shape 和 gradient ownership 正确。
- 只有一个冻结训练 arm 和一条正式命令。
- checkpoint 先过安全 gate，再按预登记词典序选择，不按 final/reward 单独选择。

### G7 Export/deploy/preflight

- PyTorch/ONNX parity、schema hash、mock SDK、unitree_mujoco wiring PASS。
- 正式规模 no-learning preflight PASS，无 OOM/NaN/shape/telemetry/residual process。
- 无实机只能标记 `HARDWARE_PENDING`，不能声称 `HARDWARE_READY`。

## 六、训练后验收原则

V7 带 height scan，因此只能作为 teacher、仿真 upper reference 和安全参考，不能称为 matched
student baseline。新 student 的接受标准必须同时包括绝对能力 gate 和安全 guardrail：

1. flat/stand/basic tracking 不得失败；
2. rough/continuous/obstacle/slope/stairs 达到预登记绝对 completion/gain；
3. line/arc/S 达到预登记路径 gate；
4. randomized robustness 达到预登记绝对 gate；
5. slip、pitch、action acceleration、effort、body contact 和 failure risk 不得违反 safety
   guardrail；
6. ONNX/C++/unitree_mujoco 行为与 PyTorch 一致；
7. 若 checkpoint 仍相同，选择更早者。

禁止为了掩盖某一项回归构造事后加权总分。未通过全部 gate 前，V7 仍是仿真默认模型；新
student 也不得称为实机默认模型。

## 七、记录和交付

至少创建或更新：

```text
docs/reviews/go2_proprioceptive_student_readiness.md
docs/reviews/go2_proprioceptive_student_training_contract.md
docs/PROJECT_JOURNAL.md
docs/HANDOFF.md
```

并生成 strict finite JSON readiness artifact。最终必须报告：

- 子智能体 A/B/C 的结论和主智能体解决的冲突；
- student 当前项、history 项、准确顺序、shape 和窗口时长；
- critic privileged terms 和准确 shape；
- teacher 完整路径、SHA、加载方式、action timing 和蒸馏 schedule；
- action/SDK/PD/control-rate/safety 合同；
- randomization、task、network、PPO 和初始化方式；
- 唯一正式训练命令、预计耗时和显存；
- G0-G7 的 PASS/FAIL/HARDWARE_PENDING 与证据路径/SHA；
- `PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY` 是否为 true；
- 默认模型是否变化（本轮应为否）；
- GPU 和相关进程最终状态。

## 八、停止条件

只能以以下状态之一结束：

```text
A. PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY=true
   G0-G7 的无硬件部分全部 PASS；实机部分 HARDWARE_PENDING；
   新 task、student/critic/teacher、schema、tests、preflight、唯一训练命令和验收合同已落地。

B. BLOCKED_WITH_EVIDENCE
   存在不能通过官方源、仓库实现、mock SDK 或 unitree_mujoco 解决的硬阻塞；
   已验证至少两条安全替代路径不可行，并给出解除阻塞所需的最小输入。
```

不能因为子智能体报告完成、需要修改网络、测试数量多或没有真机而提前停止。没有真机只
阻止 `HARDWARE_READY`，不阻止本轮训练就绪。

---
