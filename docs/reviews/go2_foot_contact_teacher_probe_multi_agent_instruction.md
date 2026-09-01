# Go2 Foot-contact Teacher 单变量探针：多智能体协同指令

下面整段可以直接交给新窗口中的主智能体执行。

---

你是主智能体。请在
`/home/jensen/projects/unitree_rl_mjlab` 中组织三个子智能体，完成 Go2 complex-walking
teacher 的下一项输入干预准备。最终目标是：

```text
SELECTED_INPUT=foot_contact(4)
CONTROL_ACTOR_DIM=234
CANDIDATE_ACTOR_DIM=238
OBSERVABILITY_DIAGNOSTIC_PASSED=true
GO2_FOOT_CONTACT_TEACHER_PROBE_READY=true
FORMAL_TRAINING_STARTED=false
DEFAULT_MODEL_REPLACED=false
HARDWARE_READY=false
```

本轮只完成：只读复核、evaluation-only 增量可观测性诊断、238-D schema/transfer/task 实现、
测试、32-env optimizer smoke、2048-env no-learning preflight、训练合同和日志。不得调用正式
400-update PPO `learn`。若可观测性诊断不通过，则停止在：

```text
OBSERVABILITY_DIAGNOSTIC_PASSED=false
GO2_FOOT_CONTACT_TEACHER_PROBE_READY=false
DECISION=INCONCLUSIVE_DO_NOT_TRAIN
```

不得为了得到 ready 结果而放宽诊断、改换输入或叠加变量。

## 1. 锁定基线与实验边界

唯一 source：

```text
checkpoint:
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256:
73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
actor / critic / action: 234 / 261 / 12
```

唯一允许的候选差异：

```text
control_234: V7 actor terms unchanged
candidate_238: V7 actor terms + foot_contact(4) at [234:238]
contact runtime sensor order: FL, FR, RL, RR
contact value/unit/frame: {0,1} / boolean / frame-independent
contact timing/rate: current policy observation / 50 Hz
normalizer source: V7 critic foot_contact [245:249]
both arms: critic 261-D and action 12-D unchanged
```

严禁：

- 做 241-D；它会叠加已拒绝的 `base_lin_vel(3)`。
- 从 V8 237-D checkpoint 或 optimizer 初始化。
- 同时加入 foot height、air time、contact force、terrain normal、clearance 或新 scan。
- 修改 reward、terrain、command、curriculum、gait、termination、network width、PPO、actuator、
  action mapping 或控制频率。
- 启动 student、running teacher、正式长训、部署导出或硬件测试。
- reset/checkout/clean、切换分支、删除证据、覆盖 JSON/checkpoint、擅自 commit/push。

已有 dirty worktree 全部属于用户，必须保留。开始和结束记录 branch、HEAD、status、diff-check、
磁盘、GPU 和 train/evaluate/play/audit/TensorBoard/unitree_mujoco 进程；不得重复启动已有任务。

先读：

```text
docs/reviews/go2_foot_contact_teacher_input_diagnosis_20260809.md
docs/reviews/go2_privileged_linvel_teacher_acceptance.json
docs/reviews/go2_privileged_teacher_high_slope_selection.json
docs/HANDOFF.md
docs/PROJECT_JOURNAL.md
src/tasks/velocity/privileged_teacher_schema.py
src/tasks/velocity/velocity_env_cfg.py
src/tasks/velocity/mdp/observations.py
```

## 2. 三个子智能体的职责

第一轮全部只读。主智能体拥有集成权；如需编辑，先声明互不重叠的文件所有权。

### 子智能体 A：证据与可观测性诊断

1. 复核 friction/contact、scan-mask、foot-placement、contact-stabilizer、stance-slip 和 V8 证据。
2. 校验 `foot_contact` 的来源、阈值、单位、FL/FR/RL/RR 运行时槽序、采样时刻、reset、chatter
   与 finite。
3. 实现 evaluation-only 数据记录：同一 pre-action 时间戳的 V7 obs234、contact4 和一个唯一
   定义的 terrain-relative foot-clearance4 对照。若 clearance 来自 site canonical
   `FR/FL/RR/RL`，必须显式重排为 contact 的 `FL/FR/RL/RR` 后再比较。
4. 数据按 matched seed/route/slot 分组切分；禁止逐 step 泄漏。预测 horizon 为未来
   10--50 steps，目标至少含 slip onset、unexpected contact transition、failure 和 progress。
5. 比较 `234-only`、`234+contact4`、`234+clearance4` 的 out-of-sample 增益，输出 paired
   bootstrap CI，并分别报告 clean/randomized 与 `vx=.3/.5`。
6. contact4 只有在方向一致且 CI 排除无增益时才判通过。coverage、时序、足序、provenance、
   finite 任一不成立，必须 `INCONCLUSIVE_DO_NOT_TRAIN`。

输出必须含 raw artifact、机器可读 summary、命令、SHA256 和人类可读结论。critic sensitivity
只能作为辅助，不得替代 actor 信息增益证据。

### 子智能体 B：238-D schema、迁移与任务

1. 新建独立 frozen contact schema；不得修改 V8 schema/hash 的既有语义。
2. 从实际 ObservationGroup 计算并断言 term order/dim，actor 只追加 contact4，critic 保持 261。
3. source actor 旧列/bias、critic、actor/critic normalizer 和 environment state 严格复制。
4. candidate 第一层 `[234:238]` 精确为零；新 normalizer 只能从 source critic `[245:249]`
   gather。现有 V8 mapper 硬编码 `[234:237]`，不得简单改目标维数后误复制
   `base_lin_vel + foot_height[0]`。
5. optimizer 必须 fresh；构建两臂后重新设置 Python/NumPy/Torch CPU/CUDA RNG。
6. 注册明确的 `control_234` 和 `candidate_238` task ID；不得更改两臂的其他配置。
7. 实现 term/foot order、runtime actor-contact=critic-contact、finite/binary、tensor mapping、
   normalizer mapping、zero-column、iteration-0 action parity、provenance 和错误 schema 拒绝测试。
   必须区分 sensor observation 的自然序 `FL/FR/RL/RR` 与 evaluator site canonical
   `FR/FL/RR/RL`；已有 gait artifact 的对应 permutation 是 `[1,0,3,2]`。

### 子智能体 C：验收、停止条件与审计

1. 复用 V8 fail-closed selector，不得只比较均值或 final checkpoint。
2. 冻结 update `100/200/300/400` 的同号 paired screening；两臂都必须通过 checkpoint
   provenance、recursive finite、optimizer/env state 与 TensorBoard 完整性检查。
3. 高坡 screening 使用同一 V7 baseline 和 clean/randomized line/arc/S matched slots。
4. hard gates：每 route clean completion `>=12/16`、randomized `>=10/16`、mean forward
   gain `>=.80`、每 route completion 不低于 matched V7。
5. 两臂均满足相对 V7 `1.2x` safety guardrail：terrain-tangent slip、action acceleration、
   base pitch、base/upper-leg/calf contact、failure risk、action fault、joint-target fault；V7 为
   零时仍要求精确零。
6. 只有同一 update 两臂都通过 hard gates，且 candidate-control macro completion
   `>=+0.10`，才是 causal survivor。排序只用预登记 lexicographic 规则，不用 weighted score。
7. 只设计后续 42-invocation full acceptance，不得在本轮执行。screening/causal gate 失败即
   `NO_CAUSAL_SURVIVOR`，禁止自动补训或追加输入。

## 3. 主智能体执行顺序

```text
只读基线审计
  -> evaluation-only 可观测性诊断
  -> 若失败：写日志并停止
  -> 若通过：冻结 234/238 schema 与训练合同
  -> 实现 matched tasks 与 strict transfer
  -> 单元测试 / compileall
  -> 32-env optimizer smoke
  -> 2048-env no-learning preflight
  -> 生成唯一正式训练命令但不执行
  -> 更新 PROJECT_JOURNAL 与 HANDOFF
```

32-env smoke 只能验证 optimizer state、梯度、新列学习通路与 finite，不能判断步态性能。
2048-env preflight 必须不学习，证明两臂 shape、state restoration、RNG、reward/terrain/config
equality 和 pre-learning action parity。任何技术失败必须修复并写入新的、不可覆盖的 artifact；
不能把技术失败解释成行为失败。

## 4. 必须冻结的正式训练合同（只生成，不执行）

```text
arms: control_234, candidate_238
source: locked V7 model_13600.pt
num_envs: 2048 each
seed: 42
optimizer: fresh
updates: 400
checkpoints: 100, 200, 300, 400 only
actor intervention: append foot_contact(4) only in candidate
critic/action/reward/terrain/command/gait/PPO: identical
```

每个 checkpoint 必须记录 candidate 新列 gradient norm、weight norm、合法 `0<->1` contact
counterfactual action sensitivity、config/schema/provenance SHA。新列非零不构成通过；最终以
hard behavior/safety 与 causal delta 为准。

## 5. 停止与交付

下列任一项立即 fail closed：

```text
diagnostic no incremental contact value
foot order/timing/coverage/provenance ambiguous
pre-learning action parity > 1e-6
nonzero new actor columns before learning
old tensor or normalizer mismatch
NaN/Inf/OOM/simulator failure
state restoration or RNG mismatch
reward/terrain/action/config differs between arms
```

技术重试只能保持相同合同并生成新 artifact。禁止放宽 gate、跨 update 配对、挑“最少违规”
checkpoint、补训、加第二输入、替换 V7 默认模型或宣传 training/hardware ready。

最终交付：诊断 raw/summary/SHA、238-D schema、matched task、测试与 compileall 结果、32-env
smoke、2048-env preflight、冻结训练合同、唯一未执行命令、PROJECT_JOURNAL 与 HANDOFF 更新。
最终报告必须区分：contact/traction 是已支持的失败机制；contact bit 的 actor 增量价值仍需
诊断证明；probe ready 不等于 trained；sim teacher 不等于 hardware ready。
