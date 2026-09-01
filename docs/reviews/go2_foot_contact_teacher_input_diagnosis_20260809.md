# Go2 Teacher 下一输入变量诊断（2026-08-09）

## 结论

```text
SELECTED_HYPOTHESIS=foot_contact(4)
SOURCE_ACTOR_DIM=234
CANDIDATE_ACTOR_DIM=238
RETAIN_REJECTED_BASE_LIN_VEL=false
ADD_TERRAIN_OR_PLACEMENT_INPUT=false
FORMAL_TRAINING_STARTED=false
DEFAULT_MODEL_REPLACED=false
```

本轮若只允许给 V7 actor 增加一种输入，优先验证按项目真实足序排列的瞬时
`foot_contact(4)`。它是当前证据最支持的**单变量假设**，不是已经证明有效的修复。
在启动 PPO 前，必须先做 evaluation-only 的增量可观测性诊断；诊断不支持时结论为
`INCONCLUSIVE / DO_NOT_TRAIN`。

候选必须从锁定的 V7 234-D actor 直接构造：

```text
control:   V7 actor = 234-D
candidate: V7 actor + foot_contact(4) = 238-D
critic:    261-D，保持不变
action:    12-D，保持不变
actor new slice: [234:238]
contact runtime order: FL, FR, RL, RR
normalizer source: source critic [245:249]
```

旧记录中的 241-D 不是干净实验：`241 = 234 + base_lin_vel(3) + foot_contact(4)`。
其中 `base_lin_vel(3)` 已在 V8 中被拒绝；保留它会令本轮相对 V7 同时改变两个输入，无法
归因。因此不得从 V8 237-D checkpoint 继续追加 contact，也不得继承 V8 optimizer。

## 证据链

1. V7 actor 已有 `height_scan(187)`，但 `foot_contact(4)` 只在 261-D critic 中。
2. 屏蔽 height scan 后平地也全部失败，说明现有扫描是全局必要输入；它不能证明高坡失败
   来自局部地形信息不足。
3. 64-repeat friction matched triplet 的正式结论为 `CONTACT_CAUSAL`：
   `vx=.3` completion 为 `39/42/45`、forward gain 提升 `30.6%`；`vx=.5` 为
   `11/10/49`、gain 提升 `74.1%`，failure-risk ratio 为 `.864/.278`，且注册的
   `1.2x` safety guardrail 通过。
4. 固定 `+0.05 rad` foot-placement 反事实在两个速度格都明显有害。它不能否定所有未来的
   地形输入，但足以反对当前直接采用笼统的“落脚/局部地形信息”。
5. local-tangent contact stabilizer、stance-slip reward 400-update arm 均未产生安全 survivor。
   这说明接触是机制问题，但简单控制偏置或 reward 惩罚不是已验证解法。
6. V8 `base_lin_vel(3)` 新列权重确实增长，却没有 causal survivor；因此“新增 privileged
   输入”本身不能作为训练理由。

主要证据：

```text
docs/PROJECT_JOURNAL.md
docs/reviews/go2_privileged_linvel_teacher_acceptance.json
docs/reviews/go2_privileged_teacher_high_slope_selection.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/complex_terrain_causal_diagnostic_v2_summary.md
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/friction_contact_causal_strict_clean_seed42_64repeats_probe12_v13_summary.md
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/foot_placement_counterfactual_strict_clean_seed42_32repeats_v1.json
```

## 训练前必做诊断

`foot_contact` 当前只是 `ContactSensor.data.found > 0` 的瞬时二值。现有因果证据证明接触/
牵引能力是限制因素，但没有证明 contact bit 含有 V7 的 phase、joint state、last action 和
height scan 无法推断的增量信息。

必须在同一 pre-action 时间戳记录 V7 234-D actor observation、FL/FR/RL/RR contact4，以及
唯一且精确定义的 terrain-relative foot-clearance4 对照。按 matched seed/route/slot 分组切分，
禁止逐 step 随机切分导致同一 rollout 泄漏。比较下列输入对未来 10--50 steps 的 slip onset、
unexpected touchdown/liftoff、failure 和 progress 的 out-of-sample 预测增益：

```text
V7 234-D only
V7 234-D + contact4
V7 234-D + clearance4（诊断对照，不代表允许一起训练）
```

只有 contact4 在 clean/randomized、`vx=.3/.5` 上取得方向一致、bootstrap CI 排除无增益的
改善，并且时序、足序、coverage、finite 和 provenance 全部有效，才允许预登记 238-D arm。
critic block ablation/permutation 只能作为辅助证据，不能冒充 actor 因果证据。

## 若诊断通过：冻结的单变量训练合同

- source 仅为 V7 `model_13600.pt`，SHA256
  `73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`。
- matched `control_234` / `candidate_238` 使用同一 environment state、common step、seed 42、
  2048 env、fresh optimizer 和 400 PPO updates。
- candidate 第一层新四列精确置零；旧列、bias、critic 全部严格复制；contact normalizer 从
  V7 critic `[245:249]` 复制。不得错误复制 critic `[234:238]`。
- 运行时 contact sensor 的自然槽序必须实测为 `FL/FR/RL/RR`；actor 直接复用 critic term，
  不在本次单变量实验中新增重排。Evaluator 若要按 site canonical `FR/FL/RR/RL` 报告，必须
  单独记录 permutation `[1,0,3,2]`，不能把报告顺序误写成 observation 顺序。
- source normalizer 的 contact slice 已检查：count `423247872`，mean 约
  `[.6338,.6278,.6133,.6176]`，std 约 `[.4818,.4834,.4870,.4860]`。
- reward、terrain、sampler、curriculum、command、gait、termination、network、PPO、actuator
  与 action interface 全冻结。本轮是输入因果实验，不混入 bounded-action、reward 或 gait 改动。
- 学习前两臂 action parity `<=1e-6`；先做 32-env optimizer smoke 和 2048-env no-learning
  preflight，再保留 update `100/200/300/400`。
- 必须记录新列 gradient/weight norm 和 counterfactual contact-flip action sensitivity；权重非零
  只证明输入被使用，不等于行为通过。
- 同 update 配对筛选；不跨 update 拼接，不只挑 final，不用加权总分掩盖硬门槛。
- 任一 NaN/Inf、OOM、identity、state restoration、coverage、action/joint-target fault 或 safety
  gate 失败即 fail closed。没有 causal survivor 时不进入 42-invocation full acceptance。

正式长训仍未开始，V7 保持默认。Student、奔跑步态、动作接口升级和硬件部署都不在本轮
单变量实验范围内。
