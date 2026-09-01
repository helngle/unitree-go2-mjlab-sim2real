# Go2 复杂环境 Teacher → 奔跑 Teacher 两阶段多智能体协同指令

下面整段可直接交给新窗口中的主智能体执行。

---

你是主智能体，负责组织三个子智能体，把
`/home/jensen/projects/unitree_rl_mjlab` 中的 Unitree Go2 项目推进到：

```text
GO2_COMPLEX_WALKING_TEACHER_TRAINING_READY=true
FORMAL_TRAINING_STARTED=false
STUDENT_WORK_OUT_OF_SCOPE=true
HARDWARE_READY=false
```

本轮不是继续旧 V7、重跑 V8、直接训练 student，也不是把复杂地形和高速奔跑一次性混在
同一训练 arm。本轮唯一目标是：设计并实现第一阶段 privileged complex-walking teacher，
先提升 `0.3-1.0 m/s` 正常步态下的复杂环境能力，完成代码、测试、冻结训练合同和
full-scale no-learning preflight，使下一轮可以用一条唯一命令启动正式第一阶段 teacher
训练。

第一阶段 teacher 通过完整验收后，才允许单独预登记第二阶段 running teacher；第二阶段
计划在保留第一阶段能力的前提下扩展到约 `1.5-2.5 m/s` fast trot。Student 蒸馏、DAgger、
student PPO 和 student 部署不属于当前计划，除非用户以后重新授权。不得仅写调研报告；
任务、配置、schema、runner/初始化路径、验收器和 preflight 必须真实落地，但不得在本轮
执行正式长训。

## 一、工作区与已知事实

```text
工作区：/home/jensen/projects/unitree_rl_mjlab
Conda Python：/home/jensen/anaconda3/envs/unitree_rl_mjlab/bin/python
分支：exp/high-slope-probe-integration
已知 HEAD：0a204b645a2325cb06264725c58cc5745da64a43
```

工作区已有大量用户未提交修改和训练证据。必须全部保留。禁止：

```text
git reset
git checkout -- <path>
git clean
切换分支
删除或覆盖既有日志/JSON/checkpoint
擅自 commit、push 或替换默认模型
```

开始和结束必须记录：branch、HEAD、`git status --short`、`git diff --check`、worktree、
磁盘、GPU，以及 train/evaluate/play/audit/TensorBoard/unitree_mujoco 进程。发现同类正式
任务正在运行时，不得启动重复任务。

当前唯一保留的综合仿真基线：

```text
checkpoint:
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256:
73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
actor / critic / action: 234 / 261 / 12
```

V7 actor 包含 187 维 MuJoCo `height_scan`，只能作为仿真 baseline/transfer source，不能
直接部署。V7 的 `high_speed` 仅为约 `0.8-1.0 m/s`，固定 `0.6 s` 对角 trot，不能冒充
已经训练好的奔跑策略。

以下结论已经正式成立，禁止推翻或绕过：

```text
V1 proprioceptive student:
  TRAINING_REJECTED
  selection_status=NO_SAFE_SURVIVOR
  原因包括真实 rollout action/processed-target faults

V2 safe-action meanbound5 student:
  252/252 formal GPU invocations complete
  17/17 checkpoints fail completion and/or safety gates
  selection_status=NO_SAFE_SURVIVOR
  bounded action 本身通过，但 locomotion/safety 不通过

V8 privileged base_lin_vel teacher:
  234-D control 与 237-D candidate 均完成 400 updates
  selection_status=NO_CAUSAL_SURVIVOR
  decision=REJECT
  base_lin_vel 新列确实被学习，失败不是 unused-input 或 transfer bug
```

对应证据：

```text
docs/reviews/go2_v2_meanbound5_formal_selection.json
docs/reviews/go2_privileged_teacher_high_slope_selection.json
docs/reviews/go2_privileged_linvel_teacher_acceptance.json
docs/HANDOFF.md
docs/PROJECT_JOURNAL.md
```

旧文档中曾提到的 241-D `foot_contact(4)` arm 已被后续审计纠正：相对 V7 的干净单变量
实验必须是 `234 + foot_contact(4) = 238-D`；241-D 会叠加已拒绝的 `base_lin_vel(3)`。
当前推荐先执行
`docs/reviews/go2_foot_contact_teacher_probe_multi_agent_instruction.md` 中的训练前诊断和 preflight，
不得自动启动正式 PPO。

## 二、本轮推荐的两阶段系统边界

### 1. 先训练复杂环境正常步态，再扩展奔跑

固定顺序：

```text
Stage A: complex-walking privileged teacher
  command: 0.3-1.0 m/s 为主
  terrain: flat retained + ordinary rough + obstacle + stairs + continuous slope
  objective: 落脚、接触、滑移、姿态、forward gain 和 retained route 稳定

Stage B: running privileged teacher（Stage A 验收通过后另行预登记）
  command: 1.5-2.5 m/s proposed
  terrain: flat + low/ordinary rough 起步
  objective: fast trot、高速 tracking、步频/占空比、功率和安全
```

Stage A 必须先关闭当前 V7 的正常速度复杂环境短板，尤其是持续坡地、复杂接触、落脚/步长
不足和失败前恢复。不得要求 Stage A 同时解决 `2.5 m/s`，也不得因为 Stage B 最终需要高速
而提前修改多个 gait 机制。

### 2. 第二阶段奔跑目标的预备定义

若用户没有另行指定，主智能体应把下面方案作为推荐默认值写入设计稿，但必须明确标记为
`PROPOSED_FOR_USER_APPROVAL`，在正式训练前获得确认：

```text
gait family: speed-conditioned fast diagonal trot
primary surface: flat + low/ordinary rough
target forward command: 1.5-2.5 m/s
evaluation points: 1.5, 2.0, 2.5 m/s
lateral/yaw during first running teacher: small nonzero stabilization envelope
control rate: 50 Hz
stock Go2 actuator/PD/limits: unchanged
```

第二阶段 teacher 不默认追求 gallop、bound、极限速度、高坡高速和复杂路径高速同时通过。
这些目标耦合过强，应在 Stage A 和基础 running gait 通过后分阶段扩展。若证据表明 `2.5 m/s` 超出当前
MuJoCo stock actuator/接触模型的可信能力，必须给出测得的可行上限和 headroom 证据，不能
通过提高力矩、摩擦或 PD 来伪造成功。

奔跑定义至少必须冻结：

```text
速度范围与命令分布
步态族与允许的自然相变
是否要求 flight phase，以及如何测量
步频/占空比是否随速度变化
平地与 rough 的范围
tracking/completion/safety/energy 硬门槛
起步、稳态、减速和 command transition 的窗口
```

### 3. Teacher 可以 privileged，但必须保持清晰的信息边界

Teacher actor 可使用训练期 privileged observation，但每一项都必须说明：维数、单位、
frame、时序和用途。当前不要求设计或训练 student，也不得为了假设中的 student 提前削弱
teacher；但必须保持 actor/critic 信息边界清晰，便于解释和复现实验。

允许审查的候选信息包括：

```text
height scan
noise-free base linear velocity
foot contact state
foot contact force / normal load
foot height / air time
terrain normal or local height features
randomized dynamics truth
```

不得把所有 privileged 信息一次性堆入 actor。必须选择最小必要集合，并通过 input ablation、
zero-column transfer parity、gradient/weight norm、counterfactual action sensitivity 证明它被
正确使用。critic 可以更宽，但只能输出 value。

V8 已证明“只追加 base_lin_vel(3)”不足以达到高坡 teacher gate；不得原样重跑。新的 teacher
可以从 V7 做严格映射的 transfer，也可以重新训练，但必须比较二者风险并只选择一种正式
初始化方式。禁止恢复 V7 optimizer 或把 iteration 伪装成 continuation。

### 4. 动作接口必须安全一致，但不得混入输入因果探针

当前 234/238-D foot-contact 输入因果探针必须冻结 V7 action interface，不能同时改变输入和
动作语义；因此它只回答输入问题，不能推广为部署候选。探针若通过，后续正式两阶段 teacher
路线仍必须单独冻结训练、评估、ONNX、C++ 一致的安全动作语义。优先复用已经测试的：

```text
bounded_asymmetric_per_joint_v2
q_target = q0 + 0.25 * a_applied
per-joint bounds derived from MJCF limits
```

但不得假定 V2 student 的失败证明 bounded action 错误；也不得假定它一定适合高速奔跑。
子智能体必须审计 saturation、可达步幅、关节速度、effort、mechanical power 和高速恢复能力。
若提出新动作参数化，它必须是独立、预登记的单一接口变更，并先完成概率语义、ONNX 和 C++
一致性测试。部署端静默 clip 禁止作为训练修复。

### 5. 机械能力与硬件边界

不得提高 stock Go2 的 effort、velocity、range、摩擦、质量比或接触能力来通过仿真。没有
真实机器人测试时，最终状态始终为：

```text
HARDWARE_READY=false
HARDWARE_PENDING
```

本轮不生成或推广正式部署 policy。Go2 部署目录当前没有已选中候选的
`exported/policy.onnx`；不得用未验收 checkpoint 填充该目录。

## 三、多智能体组织

主智能体必须创建三个子智能体并行工作。第一轮默认只读审计；主智能体负责最终冲突解决和
共享工作区集成。确需并行编辑时，只能分配互不重叠的文件，并在消息中声明文件所有权。

### 子智能体 A：复杂环境正常步态、命令与 reward 设计

职责：

1. 审计当前 V7 command sampler、`phase`、`feet_gait`、posture、clearance、slip、landing、
   action-rate、termination 和 terrain curriculum。
2. 证明当前 `0.8-1.0 m/s + 0.6 s fixed trot` 为什么不足以定义奔跑。
3. 先定位 `0.3-1.0 m/s` 下持续坡地、楼梯、障碍、复杂接触和落脚/步长的主要失败机制，
   为 Stage A 选择一个最小训练干预。
4. 保持当前 gait 机制冻结，除非证据证明它直接阻塞 Stage A；不得提前引入高速步频变化。
5. 定义复杂地形 line/arc/S、上下坡、楼梯、起步、稳态和恢复场景。
6. 设计 Stage A telemetry，并为 Stage B 预留 running telemetry：actual speed、response gain、stride frequency、duty factor、
   diagonal phase error、flight fraction、step length、foot clearance、landing impulse、slip、
   pitch/roll、action acceleration、effort、power、saturation、fall/contact。
7. 给出唯一推荐的 Stage A task 配置；同时给出 Stage B 的边界草案，但不实现正式 Stage B
   训练 arm，不运行 PPO sweep。

输出：

```text
docs/reviews/go2_complex_walking_teacher_design.md
机器可读 Stage A acceptance schema 草案
推荐 task ID、精确 command/gait/reward/terrain 差异表
```

### 子智能体 B：Privileged teacher 架构与初始化

职责：

1. 锁定 V7 checkpoint/path/SHA，复核 234/261/12 schema、normalizer 和 environment state。
2. 复盘 V8 234/237 两臂的 9 个 high-slope artifacts 与 selection，不得只读总结结论。
3. 比较最小 privileged actor 候选：V7 scan、base velocity、contact/air-time、局部地形信息，
   并区分对 Stage A 正常速度复杂地形有用的信息与只对 Stage B 高速有用的信息。
4. 设计最小输入干预，禁止一次增加多个无法归因的 term；对未选候选写明拒绝理由。
5. 比较 strict V7 transfer 与 fresh teacher，选择一个正式初始化；定义 tensor mapping、
   zero initialization、normalizer source、fresh optimizer、RNG 和 environment restoration。
6. 确认 bounded action 与 teacher transfer 的兼容方式，防止 transform twice 或 label/action
   语义不一致。
7. 设计 optimizer smoke、2048-env no-learning preflight、checkpoint provenance 和停止规则。

输出：

```text
docs/reviews/go2_complex_walking_teacher_design.md
teacher actor/critic/action 精确维数与 term order
冻结 transfer/initialization contract
```

### 子智能体 C：验收、物理能力与 Stage B 奔跑边界

职责：

1. 审计 stock Go2 actuator effort/range、PD、action scale、控制频率和高速所需 headroom。
2. 复用现有 route/terrain/safety evaluator，设计 Stage A 正常速度复杂地形的
   clean/randomized 矩阵，不重复制造不兼容指标。
3. 为 teacher 预登记硬门槛和词典序 checkpoint 选择，禁止事后 weighted score。
4. 定义 retained-capability guardrail，保证新 teacher 不因追求复杂地形而破坏站立、平地、
   低速、转向和动作安全。
5. 审计 Stage A observation/action/domain-randomization 对 Stage B 高速扩展的兼容性；student
   不在当前职责范围内。
6. 明确 PyTorch/ONNX/C++ 只做接口 smoke 的范围，以及没有 selected teacher 时禁止进行的
   部署动作。
7. 给出训练资源、checkpoint schedule、串行评估预算和 fail-closed artifact 规则。

输出：

```text
docs/reviews/go2_complex_walking_teacher_acceptance_contract.md
机器可读 acceptance plan
Stage B running boundary report
```

## 四、主智能体集成顺序

### 阶段 0：现状冻结

- 记录 Git/GPU/process/disk/checkpoint identities。
- 阅读最新 `HANDOFF`、`PROJECT_JOURNAL`、V1/V2/V8 正式结论。
- 验证 V7 SHA，不修改任何旧 checkpoint 或 raw acceptance artifact。

### 阶段 1：三个子智能体并行只读审计

- A/B/C 必须给出文件和行号、artifact path/SHA、已验证/推断/未知分类。
- 主智能体同时检查 task registry、runner、export、deploy 和测试入口。
- 子智能体不得擅自启动正式训练或大规模 GPU evaluator。

### 阶段 2：冲突解决与用户确认点

主智能体交叉核验三份报告并冻结一份设计候选。以下内容若用户此前没有明确指定，必须作为
一个集中确认点提交，而不是在训练中猜测：

```text
Stage A 复杂地形范围与正常速度上限
Stage A 选定的单一训练干预
teacher 是否允许 contact/terrain privileged actor inputs
Stage B proposed gait family、速度上限和 flight-phase 定义
```

在等待确认期间仍可完成不依赖这些选择的 schema、测试和 evaluator 工程，但不得启动正式
teacher 训练。

### 阶段 3：统一实施

主智能体实现一个全新、不会污染 V7/V8 的 Stage A 任务，建议命名：

```text
Unitree-Go2-Complex-Walk-Teacher-V1
```

`Unitree-Go2-Run-Teacher-V1` 仅保留为未来 Stage B 建议名；Stage A 验收通过前不得注册成
可启动正式训练的第二 arm。

必须实现或更新：

```text
独立 env cfg 与 task registration
独立 runner cfg
normal-speed complex-terrain command/curriculum
选定的单一复杂环境训练干预
privileged actor/critic schema
安全动作接口
严格 transfer 或 fresh initialization loader
Stage A telemetry 与 evaluator
checkpoint provenance/source manifest
training contract 与 acceptance contract
定向单测和负向测试
```

不得修改旧 V7/V8 task 的语义，不得把新 task 注册成旧名字，不得更换默认模型。

### 阶段 4：训练前验证

至少执行并记录：

```text
targeted unit tests
完整相关 Python unittest
compileall
task registry/config/CLI smoke
reward/command/gait metric synthetic tests
teacher transfer/fresh-init strict tests
normalizer/RNG/environment-state tests
bounded action probability/target-limit tests
checkpoint/source-manifest tests
32-env optimizer smoke（明确 non-candidate）
2048-env、正式 task/seed 的至少 8 control-step no-learning preflight
recursive finite checks
git diff --check
GPU/process cleanup verification
```

2048-env preflight 必须满足：

```text
learn_called=false
optimizer_step_called=false
candidate_checkpoint_written=false
observation/action/reward/metrics finite
actor/critic/action shapes exact
action and processed target within registered limits
source/transfer/schema identities exact
```

### 阶段 5：冻结下一轮唯一训练命令

只登记、不执行一条正式命令。命令必须固定：

```text
task ID
num_envs
env seed / agent seed
initialization source and SHA
iterations
save interval
logger
run name
source manifest
exclusive run lock
```

正式训练开始后不得修改 command、gait、reward、terrain、randomization、observation、action、
network、PPO、seed 或 selection gate。技术失败重试必须保留失败 run，并证明配置和源码身份
一致。

## 五、Stage A Teacher 验收原则

具体数值由三方审计后预登记，但结构必须固定为“硬门槛优先，随后词典序选择”。至少覆盖：

### Complex-walking ability hard gates

```text
0.3/0.5/0.8/1.0 m/s steady-state tracking
起步、速度切换、减速和停止
clean/randomized flat、ordinary rough、obstacle、stairs、continuous slope
line/arc/S 正常速度闭环
足够的 full-horizon survival/completion
forward response gain
foot placement/step length/contact/slip/recovery 指标可用且 finite
```

### Safety hard gates

```text
zero action-interface and processed-target faults
joint position/velocity within registered limits
effort/power/saturation 不超过 stock 能力合同
fall/base/upper-leg/calf contact 受控
slip、pitch/roll、landing impulse、action acceleration 受控
无 NaN/Inf、reset storm、placement 或 lifecycle failure
```

### Retained capability guardrails

```text
stand
0.3/0.5/0.8/1.0 m/s tracking
lateral stabilization
yaw/turning
ordinary rough/obstacle/stairs retained matrix
line/arc/S low-speed route capability
```

所有 matched safety 指标必须与 V7 或明确登记的 control 同场景比较。V7 为 legacy simulation
reference，不是硬件安全真值；绝对动作/target fault gate 不能被相对改善替代。

硬门槛幸存者之间按以下词典序选择：

1. 最高的最差复杂地形/profile completion/survival；
2. 最低的最差速度跟踪误差；
3. 最高 randomized complex-terrain completion；
4. 最佳 retained-capability 下界；
5. 最低 failure risk、slip、contact、pitch、landing impact；
6. 最低 effort/power/action acceleration；
7. 若仍相同，选择更早 checkpoint。

禁止用训练 reward、最终 checkpoint 或事后加权总分直接选模型。

## 六、硬门槛

只有以下全部成立，才能给出 `GO2_COMPLEX_WALKING_TEACHER_TRAINING_READY=true`：

```text
G0 provenance/worktree/checkpoint identities PASS
G1 Stage A target、normal-speed command 和 terrain semantics frozen
G2 stock-Go2 physical/action contract PASS
G3 teacher observation and privileged boundary PASS
G4 initialization/transfer/optimizer ownership PASS
G5 task/reward/command/curriculum implementation PASS
G6 Stage A evaluator/telemetry/acceptance contract PASS
G7 32-env optimizer smoke PASS, non-candidate
G8 2048-env no-learning preflight PASS
G9 tests/compile/diff/process cleanup PASS
```

`GO2_COMPLEX_WALKING_TEACHER_TRAINING_READY=true` 只表示可以开始 Stage A 仿真 teacher 训练，不代表 teacher
已经合格。必须同时报告：

```text
FORMAL_TRAINING_STARTED=false
TEACHER_ACCEPTED=false
STUDENT_WORK_OUT_OF_SCOPE=true
DEFAULT_MODEL_CHANGED=false
DEPLOYMENT_BUNDLE_READY=false
HARDWARE_READY=false
HARDWARE_PENDING
```

## 七、交付物

至少生成或更新：

```text
docs/reviews/go2_complex_walking_teacher_design.md
docs/reviews/go2_complex_walking_teacher_acceptance_contract.md
docs/reviews/go2_complex_walking_teacher_preflight_2048env.json
docs/reviews/go2_complex_walking_teacher_training_contract.md
docs/reviews/go2_running_stage_b_boundary.md
docs/PROJECT_JOURNAL.md
docs/HANDOFF.md
```

最终报告必须说明：

```text
三个子智能体各自结论
主智能体如何解决冲突
用户确认了什么，哪些仍是假设
Stage A 速度/terrain/单一干预定义及 Stage B 边界
teacher actor/critic/action schema
动作与物理能力合同
初始化方式
唯一正式训练命令
验收矩阵与 checkpoint 选择规则
G0-G9 逐项状态及证据路径/SHA
默认模型是否变化（必须为否）
student 状态（必须为 OUT_OF_SCOPE）
GPU 和相关进程最终状态
```

## 八、停止条件

只允许两种结束：

```text
A. GO2_COMPLEX_WALKING_TEACHER_TRAINING_READY=true
   所有代码、测试、preflight、合同和唯一训练命令均已落地；
   Stage A 正式训练尚未开始；Stage B 未启动；student 不在当前范围内。

B. BLOCKED_WITH_EVIDENCE
   存在无法由仓库、已有 artifact、官方来源或安全 smoke 解决的真实阻塞；
   已验证至少两条合理替代路径仍不可行；
   明确给出解除阻塞所需的最小用户选择、数据或硬件信息。
```

不得因为子智能体返回、报告完成、测试较多或预计训练耗时较长而提前停止，也不得降低 gate
来制造 `READY=true`。

---
