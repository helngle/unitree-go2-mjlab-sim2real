# Unitree Go2 V10 项目实时日志

项目方案：[`V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md`](V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md)

历史档案：[`PROJECT_JOURNAL.md`](PROJECT_JOURNAL.md)

## 日志规则

- 本文件从 `2026-08-10` 开始，按时间顺序追加；
- 不删除失败、不覆盖旧结论、不把 partial 结果改写成 pass；
- 每次代码、测试、评估和训练都记录状态、命令、输入、输出、SHA和结论；
- 引用历史实验时记录准确路径及其允许用途；
- `STARTED`、`RUNNING`、`PAUSED`、`FAILED`、`REJECTED`、`PASSED` 和 `COMPLETE` 必须按真实状态使用；
- 没有真实启动进程时不得写“正在训练”。

## 2026-08-10：建立 V10 总方案与独立日志

### 用户确认的最终目标

```text
Unitree Go2复杂地形稳定行走
  -> V10-Walk Teacher
  -> Student蒸馏
  -> Sim2Real
  -> 平地奔跑
  -> 复杂地形奔跑
```

论文用于提供直接设计依据，目标不是逐篇复现。允许重新设计现有训练系统，但最终能力、安全和可部署性是唯一终点。

用户同时确认：

- 每项创新机制至少需要一篇直接相关论文或公开GitHub实现；
- Teacher设计必须从一开始考虑后续Student和Sim2Real；
- 旧方案和旧日志暂时封存，但以后允许只读查看和按需引用；
- 旧证据不得未经新合同自动触发训练。

### 本次文档变更

新增：

```text
docs/V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md
docs/V10_GO2_PROJECT_JOURNAL.md
```

旧 `docs/PROJECT_JOURNAL.md` 不删除、不移动，只增加新路线导航并进入历史档案状态。

### 建档时仓库状态

```text
workspace: /home/jensen/projects/unitree_rl_mjlab
branch: exp/high-slope-probe-integration
HEAD: 3ab0ad164dddc095f9ca084e60662597eb73d4a1
HEAD subject: feat: preserve privileged teacher acceptance workflow
worktree: dirty，存在大量用户历史实验、评估和部署改动
GPU compute process: none reported by nvidia-smi
training/evaluation process: none detected
```

现有脏工作区全部视为用户资产。本次不修改旧实验代码、checkpoint、raw artifact或部署实现。

### 历史基线状态

V7继续作为只读仿真基线：

```text
checkpoint:
logs/rsl_rl/go2_velocity/
2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/
model_13600.pt

SHA256:
73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

不得把已拒绝或partial的 privileged-linear-velocity、contact-force、stance-slip、foot-contact或旧Student实验标记为V10基础模型。

### 当前决策

```text
project_phase: M0
project_status: M0_COMPLETE
formal_training_status: NOT_STARTED
evaluation_status: NOT_STARTED
default_model: V7 model_13600.pt (unchanged)
new_teacher_initialization: planned random initialization
student_status: DESIGN_RESERVED / NOT_STARTED
sim2real_status: DESIGN_RESERVED / HARDWARE_NOT_VALIDATED
running_status: FUTURE_PHASE / NOT_STARTED
```

### 参考路线

首版项目参考集：

- Lee et al. 2020：复杂地形Teacher、privileged information和课程；
- Miki et al. 2022：proprio/extero/privileged分支和Teacher-Student感知运动；
- RMA 2021：使用因果历史估计环境latent；
- Rudin et al. 2021：大规模并行PPO和地形课程；
- Walk These Ways 2022：后续步态参数化和奔跑；
- AMP in the Wild 2023：后续动态步态候选；
- Extreme Parkour、ANYmal Parkour：更晚的障碍与parkour阶段。

这些参考只授权继续完成设计映射。实际代码修改前仍需登记准确输入、输出、时序、差异；GitHub需要commit/tag和许可证，并再次取得用户批准。

### 未关闭问题

- 目标Go2具体SKU、固件和低层控制权限；
- 实机是否有可用且能实时复现训练语义的深度/地形传感器；
- stock Go2 asset、执行器、SDK joint order和控制周期的最终identity；
- Teacher terrain representation和精确维数；
- Student历史长度、encoder结构和latent维数；
- reward公式、权重和课程推进条件；
- domain randomization的物理范围；
- PPO网络、超参数、seed、env数、iterations和资源预算；
- V10正式验收矩阵的机器实现与provenance。

### 下一步

进入 `M1 V10训练前技术设计`。在用户批准完整的输入、网络、reward、课程、randomization、训练预算和验收合同前，不启动GPU训练。

当前没有后台训练或评估任务。

## 2026-08-10：用户调整主路线，Sim2Real移至最后一步

用户明确要求不在奔跑训练之前执行 Sim2Real。项目主路线立即修订为：

```text
V10-Walk Teacher（仿真）
  -> V10-Run Teacher（仿真）
  -> 行走+奔跑统一Teacher验收
  -> final Student蒸馏与完整仿真验收
  -> Sim2Real与真实Go2逐级测试（最后一步）
```

实施边界：

- V10-Run Teacher只需等待V10-Walk Teacher仿真验收，不等待Student或实机结果；
- 默认只蒸馏一次final Student，使其同时保留行走和奔跑能力；
- 训练设计阶段继续保持观测和动作可部署，避免最后无法迁移；
- ONNX/C++/SDK/真实Go2闭环和实机测试全部放到final Student仿真通过之后；
- 当前仍未启动训练、评估或后台任务。

状态保持：

```text
M0_COMPLETE
M1_NOT_STARTED
FORMAL_TRAINING_NOT_STARTED
```

## 2026-08-10 14:51 CST：按用户要求启动多智能体训练准备

用户授权按当前V10路线训练一版，并要求继续采用多智能体协同。依据仓库根目录
`AGENTS.md` 的 Mandatory Reference Gate，本阶段先启动 `M1` 精确设计冻结；在每个创新组件的
直接参考、输入输出、时序和项目差异尚未完成登记并得到明确批准前，不允许修改训练代码、
执行GPU smoke或启动PPO。

并行只读任务：

```text
A: 原始论文/GitHub/commit/license与V10组件的直接映射
B: 当前task/observation/reward/terrain/curriculum/runner复用边界和准确维数
C: 从随机初始化的训练、课程、验收、资源、停止和provenance合同
Root: 交叉核验、M1审批文件与V10日志集成
```

启动检查：

```text
time: 2026-08-10 14:51:07 CST
GPU compute process: none reported by nvidia-smi
training/evaluation process: none detected
formal_training_status: NOT_STARTED
```

下一状态只能是：

```text
M1_DESIGN_READY_FOR_USER_APPROVAL
```

用户批准精确M1合同后，才依次进入实现、测试、32-env optimizer smoke、planned-env
no-learning preflight和正式V10-Walk Teacher训练。

## 2026-08-10：V10-Walk Teacher V1多智能体M1设计收敛

三路只读审计全部完成，未修改训练代码、未执行GPU测试、未启动PPO。精确审批合同：

```text
docs/V10_GO2_WALK_M1_DESIGN_APPROVAL.md
```

推荐首版：

```text
task: Unitree-Go2-Rough-V10-Walk-MikiTeacher-V1
input: 382-D = proprio/history124 + per-foot terrain208 + privileged50
action: 12-D Go2 joint-position target
initialization: random
training: 1024 env, 14000 updates, seeds 42/43/44, serial
scope: Walk Teacher only
```

关键纠正：Miki原版391-D输入包含13-D CPG状态，并与16-D phase/joint residual、foot
CPG和解析IK成套使用。项目既定12-D joint-target接口不能诚实保留该13-D输入；因此V10
删除Miki CPG，加入Walk These Ways有直接依据的4足sine clock，明确登记为382-D
Miki-derived/WTW-action混合方案，不声称论文复现。

参考仓库已冻结：

```text
walk-these-ways 0e7236bdc81ce855cbe3d70345a7899452bdeb1c, MIT
legged_gym      8fa29acc6fd1910c3d9659eef6310bdd301cde0a, BSD-3-Clause
Lee supplement 277d19b007fd3109956d1ea0cc9e0cd50d3ecb5b, MIT + robot terms
```

当前Reference Gate仍要求用户明确批准审批合同列出的15项实质差异。批准前：

```text
M1_DESIGN_READY_FOR_USER_APPROVAL
M2_NOT_STARTED
M3_FORMAL_TRAINING_NOT_STARTED
GPU_IDLE
```
