# V7 高坡 matched route 离线归因契约

## 范围与训练 gate

本阶段只解析已经保存的 JSON，目标是区分：

1. `straight`、`arc`、`s_curve` 在相同 high/extreme pyramid slope 上都因前向响应不足而失败；
2. `straight` 明显通过，但 `arc` 与 `s_curve` 在同一 matched matrix 上稳定失败；
3. 三种路线均通过，或证据不满足预先声明的归因阈值。

离线脚本不导入 MuJoCo/Mjlab，不创建环境、不加载 policy、不修改训练配置，也不启动
PPO。本轮训练 gate 始终为 `NO-GO`。即使归因明确，输出也只给出下一轮允许考虑的
唯一变量，不授权实施训练。

实现位于：

```text
scripts/diagnose_go2_high_slope_attribution.py
```

## Matched high-slope 输入契约

脚本接受 evaluator 的 profile 包装结构：

```text
root
  checkpoint / task_id / seed
  profiles[profile]
    matched_invariants
    route_results
      straight
      arc
      s_curve
```

`route_results` 必须严格保持 `straight -> arc -> s_curve` 顺序。每种路线的场景数、
`matched_slot` 顺序以及下列字段必须一致：

```text
slope_direction
level
difficulty_label
difficulty
effective_terrain_parameters
radius
speed
turn_sign
repeat
route_length
```

共同身份 `checkpoint/task_id/seed/profile`、`steps` 和每路线环境数必须与
`matched_invariants` 一致。每个 route kind 必须由 fresh same-seed environment 产生；
这一事实由 evaluator 写入 invariants，离线工具不从结果相似性反推随机化是否 matched。

每个 scenario 必须提供：

- completion、progress、steps、reset、catastrophic termination 和 first failure reason；
- commanded/actual `[vx, vy, wz]` mean 与 evaluator 的 `response_gain`；
- cross-track 与 heading 的 RMS/P95/max/final、final position error；
- controller saturation fraction；
- action acceleration 与 slip 的 mean/P95/max；
- base、upper-leg、calf 的 non-terminating contact count/rate；
- termination counts；
- corridor/scan footprint inside-patch 标志和 boundary margin；
- terrain assignment 与 route placement 最大误差。

输出中的 axis response gain 由 mean actual / mean commanded 再独立计算。mean command
为零的 axis 不进行除法，写为：

```json
{"value": null, "reason": "mean_command_is_zero"}
```

输入 metric 为 `null` 时必须存在同名 `*_reason`。任意 NaN/Inf、缺字段、matched slot
漂移、场景顺序漂移或 unsafe geometry 都会在归因前失败。输出使用
`json.dumps(..., allow_nan=False)`，因此同样禁止 NaN/Inf。

## Geometry 与 evaluator eligibility

只有以下条件全部成立，scenario 才能进入 policy attribution：

- corridor 和 yaw-aligned height-scan footprint 均在 patch 内；
- corridor/scan boundary margin 非负；
- terrain assignment position error max 不超过 `1e-4 m`；
- route placement position error max 不超过 `1e-4 m`；
- matched identity、slot 和 route length 未漂移。

固定从 18×18 m patch 中心沿 local `+x` 出发时，`radius=4.0 m` straight 的共同路长为
`2*pi*4/3 = 8.3776 m`。加入约 `0.8 m` 的 scan forward half extent 后，scan 终点约为
`9 + 8.3776 + 0.8 = 18.1776 m`，超出 patch 约 `0.1776 m`。因此这一组合必须在
evaluator preflight 被拒绝；不能把其失败当作 V7 locomotion failure，也不能通过只缩小
straight 路长来制造不 matched 的对照。

若仍要求共同覆盖 `radius=4.0 m`，应由 Integration Agent 重新选择一个对三类路线均合法
的共同 route placement 或更大的 evaluation patch，并重新执行全部 matched geometry
验收。归因脚本不会替 evaluator 修正几何。

## 预先声明的归因阈值

默认阈值全部写入输出 JSON，也可通过 CLI 显式覆盖：

| 参数 | 默认值 | 用途 |
| --- | ---: | --- |
| pass completion rate | 0.80 | “明显通过” |
| failure completion rate | 0.50 | “稳定失败” |
| forward under-response gain | 0.80 | 明显 `vx` 欠跟踪 |
| similar forward gain absolute spread | 0.15 | 三路线 `vx` gain 相似 |
| controller saturation fraction | 0.10 | 排除 controller-limit 主导 |
| near-end progress ratio | 0.90 | 识别 slow step-limit 假失败 |
| nominal/retry horizon | 2400 / 3000 | 只允许一次预声明 retry |

这些阈值必须在正式 JSON 解析前固定；不得观察结果后反复调整阈值直到获得想要的归因。

## 高坡归因规则

决策顺序如下：

1. 若存在 `speed <= 0.3 m/s`、`first_failure_reason=step_limit`、
   `progress_ratio >= 0.90`、零 reset、无 catastrophic termination 的 2400-step case，
   先输出 `horizon_retry_required`。每个失败 slot 只允许一次 3000-step retry，其他配置、
   seed、matched slot 与 acceptance 口径全部不变；不得继续增加 horizon 直到通过。
2. 三类路线 completion rate 均至少 0.80：
   `all_routes_passed_no_training`。之前的失败更可能来自未匹配 terrain、geometry、horizon
   或场景定义，不授权训练。
3. 三类路线 completion rate 均不超过 0.50，三者 mean `vx` gain 均不超过 0.80，gain
   最大绝对差不超过 0.15，且 saturation max 不超过 0.10：
   `sustained_high_slope_locomotion_limitation`。下一轮唯一候选变量是提高
   high/extreme sustained slope hard-case sampling。
4. straight completion rate 至少 0.80，而 arc 与 S 均不超过 0.50，且 saturation max
   不超过 0.10：
   `high_slope_forward_yaw_curvature_coupling_limitation`。下一轮唯一候选变量是提高
   high-slope parameterized forward+yaw/curvature command sampling。
5. 其他组合：`inconclusive_no_training`。包括只有一个 curve kind 失败、controller
   saturation 过高、completion 落在灰区、或三路线欠响应程度明显不同。

无论落入哪条规则，`training_authorized` 都为 false。

## Level-9 stairs multi-seed 契约

楼梯使用现有 continuous straight evaluator，固定：

```text
checkpoint: 与 high-slope matched 相同的 V7 checkpoint
task: Unitree-Go2-Rough-V7
profile: randomized
terrain_suite: continuous
mode: line_follow
transition_case: stairs_up, stairs_down
level: 9
seed: 42, 43, 44
target_speed: 0.5 m/s
steps: 2400
route: approach-flat -> stairs -> exit-flat
cross/yaw offset: 0
```

三个 seed 必须完整提供上下楼梯共六个 scenario，不得挑选通过 seed。checkpoint 和 task
必须跨 seed 一致。

按方向独立判断：

- 同一方向至少 2/3 seeds 为 `illegal_calf_contact`：
  `stable_calf_termination_risk`；
- 同一方向仅 1/3 seed 为 calf termination：
  `low_confidence_or_incidental_calf_risk`；
- 无稳定 calf 模式但失败原因跨 seed/方向异质：
  `heterogeneous_failures_require_more_diagnosis`；
- 三个 seed 均通过：`passed_all_seeds`。

stairs 风险必须与高坡训练变量分开记录。即使楼梯风险稳定，也不得在同一个 probe 中同时
改变 slope sampling 和 stairs sampling。

## 离线用法

只分析 matched high-slope：

```bash
conda run -n unitree_rl_mjlab python \
  scripts/diagnose_go2_high_slope_attribution.py \
  --matched-json <high-slope-matched.json> \
  --output <high-slope-attribution.json>
```

同时分析 stairs seed 42/43/44：

```bash
conda run -n unitree_rl_mjlab python \
  scripts/diagnose_go2_high_slope_attribution.py \
  --matched-json <high-slope-matched.json> \
  --stairs-json <stairs-seed42.json> \
  --stairs-json <stairs-seed43.json> \
  --stairs-json <stairs-seed44.json> \
  --output <high-slope-and-stairs-attribution.json>
```

正式报告必须同时保留 evaluator gate、model gate 和 attribution confidence；本工具只负责
schema/geometry 合法后的离线 attribution，不替代 Acceptance Agent 的独立 GPU JSON 验收。
