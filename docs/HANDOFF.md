# 新聊天窗口交接摘要

## 2026-07-22 V7 triggered actuator-headroom 因果诊断（最新）

已完成 evaluation-only 的事件触发式 1.25x actuator-headroom counterfactual。没有训练、
没有修改 reward/task/robot asset；默认模型仍为 V7 `model_13600.pt`（SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`）。

新增且仅修改本轮专用文件：

```text
scripts/audit_go2_actuator_headroom_triggered.py
tests/test_go2_actuator_headroom_triggered.py
```

正式 artifact 和摘要：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/actuator_headroom_triggered_clean_seed42_64worlds_1200steps_v4.json
SHA256 3f1ad2a52b57690a5af5379a6208bc3f1610e05197b1c86343e0f8e2f8fd16f2
evaluator SHA256 156da8db2f78ec21fd8e38648547ed76990c6a7e0dbc04b8217ed62bdea2bd41
actuator_headroom_triggered_clean_seed42_64worlds_1200steps_v4_summary.md
```

实现按 control 的同 joint 三连续有效饱和 row 检测，在 `detect+1` 前把完整 control
integration/manager/sensor/observation 状态复制到 probe，再只把 probe 的 hip/thigh/calf
range 从 `23.5/45` 改为 `29.375/56.25 Nm`。旧 probe 前史显式丢弃并记录；即使旧 probe
已失败也不会阻止 control trigger。所有 applied branch 的 state/obs/action identity 为 0，
runtime/control range drift 为 0，flat sentinel 16/16 未触发且 lifecycle 一致。

正式 clean 结果：10 个 slope-up trigger，完整 post-50/post-100 只有 5/4 对，低于各 8
的 hard coverage gate；可评估窗口 saturation 下降 `93.3%/90.5%`，lifecycle 为 2 win、
1 loss、1 harm，applied cohort completion delta 为 `.3:0`、`.5:+2`，但 post-300 gait
覆盖不足且 risk gate 未通过。正式 verdict 为 **`INCONCLUSIVE`**；不能升级为
`TRIGGERED_HEADROOM_CAUSAL`、`SATURATION_DOWNSTREAM` 或 `HEADROOM_INSUFFICIENT`。

两条未触发 slope probe 在相同 1.0x 下仍发生 lifecycle 分叉；多次 clean rollout 的
trigger timing/coverage 也有变化。这是 MuJoCo-Warp 高坡接触非确定性边界。不要把未触发
probe failure 算作 intervention harm。下一步唯一建议是 evaluation-only 的 matched
1.0x sham-branch sentinel，用与 probe 相同的 full-copy 时点量化 branch 后自然分叉；
在此之前不要训练或修改 reward。

## 2026-07-22 V7 actuator/force Gate 1 审计（最新）

已完成 evaluation-only actuator/force 审计。Gate 1 为
`SATURATION_CONFIRMED`，因此没有实现 foot-placement reward、没有启动 PPO、没有
新 run/checkpoint/TensorBoard。默认模型继续是 V7 `model_13600.pt`，SHA256：

```text
73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

新增：

```text
scripts/audit_go2_high_slope_actuators.py
tests/test_go2_high_slope_actuator_audit.py
```

正式 artifact：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_actuator_audit_clean_seed42_48slots_1200steps_v1.json
SHA256 e73a40f996839099b1f788bdf06c24a978549340f51ecb1aae6f4429c24bc537
evaluator source SHA256 07a957ffe8dc01178932c5993006201b315b44e5ca9b287e793d2755842dcc76
```

矩阵与 gait v2 相同：flat / slope-up high / slope-down extreme，`vx=.3/.5`，
8 repeats，seed42，100 warmup + 1200 steps；stable tail=300，failure windows=50/100。
48 rows 全部有 matched/placement/lifecycle 合同，24 个失败 row 全有完整 50/100 步
pre-reset 窗口。assignment/placement max 为 `5.36e-7/2.38e-7`。

flat 两档均 `8/8` 完成，joint P95 effort utilization 最大约 `.268`，无 near-limit
样本。slope-up `.3` 为 `6/8` 完成、`.5` 为 `2/8` 完成；8 个上坡失败 attempt
全部在失败前窗口触发持续饱和判据。最强上坡案例为 `RL_thigh`：最后 50 步
force saturation fraction `.54`，最长连续 `27` 步，actuator force 固定在
`23.5 Nm` 而未裁剪 PD demand 继续上升。extreme-down 两速度仍为 `0/8`，其中
高速 8 个失败均有 calf saturation，最长连续约 7–8 步。正式 artifact 共记录
91 条 joint/window saturation evidence，flat 为 0、slope-up 27、slope-down 64。

ContactSensor force sign 已沿 MuJoCo-Warp 实现核验：foot 为 primary、terrain 为
secondary 时，netforce 是作用在 terrain 上的力，因此正常承载对 terrain-up normal
投影为负；正式窗口负向比例范围 `.984–1.0`。termination/contact cost 使用 force
模长，sign 不会造成误终止。asset 中的 hip/thigh `23.5 Nm`、calf `45 Nm` limit
均按配置生效，未发现 evaluator/asset mapping 错误。

结论更新：A 仍是上坡短步的运动学症状，C 仍是 slip/切向载荷伴随项；D 已从风险
信号升级为失败前明确 actuator force boundary，成为当前 Gate blocker；B 不支持，
E 的 evaluator/placement/reset/force-sign mapping 通过。不要进入 reward probe。
下一步唯一建议是 evaluation-only 的 matched actuator-headroom counterfactual，只改变
effort limit 一个变量来判定 saturation 对失败的因果贡献；在此之前不要训练或改 reward。

## 2026-07-22 V7 持续高坡步态诊断（最新）

已完成 evaluation-only 诊断，未训练、未修改训练配置。默认模型继续是 V7：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

新增脚本和测试：

```text
scripts/diagnose_go2_high_slope_gait.py
tests/test_go2_high_slope_gait.py
```

正式 clean matched JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/high_slope_gait_diagnostics_clean_seed42_48slots_1200steps_v2.json
```

矩阵为 flat、high slope up（gradient +0.32）、extreme slope down（gradient -0.40），
`vx=0.3/0.5`、`vy=0`、`yaw=0`、seed42、8 repeats、100 warmup、1200 control steps。
三种 condition 共用 16 个 matched slots；ContactSensor XML 顺序 `FL/FR/RL/RR`
已重排到输出足序 `FR/FL/RR/RL`。JSON finite/schema/lifecycle、placement、source SHA
和 checkpoint SHA 均通过；CPU unittest 41 项通过；无残留 GPU 进程。

on-incline 结果显示 flat gain `.778/.836`、完整 swing displacement `.131/.233 m`；
上坡 gain `.207/.200`、on-incline step `.051/.072 m`、clearance `.060/.071 m`、
slip `.115/.162 m/s`、切向力 `20.5/24.6 N`、pitch P95 `.328/.508 rad`。下坡
两速度均 8/8 失败，主要为 upper-leg/base contact；其后足坡面暴露不足，不能解释为
稳定下坡 gait。

本阶段 scoped 归因：主类别 A（上坡短步/推进不足）；强伴随 C（坡面 slip/切向载荷）；
D 仅为风险信号（pitch/action 增大，尚未证明 actuator saturation）；B 不支持
（clearance 未下降）；E（评估器/placement/reset）通过，但下坡 contact failure 需
按短存活窗口解读。下一步若获准，先做 force-sign/actuator-limit 最小核验，再只设计
一个 high-slope forward-dominant foot-placement/step-length reward probe；不要提高
high-slope sampling，不要同时改其他变量。

## 2026-07-21 High-slope 10% sampling probe 训练结果（最新）

本轮从 V7 `model_13600.pt` warm start，正式完成
`Unitree-Go2-Rough-V7-HighSlopeProbe`：2048 env、400 iterations、seed42，唯一变量是
`H = slope_up levels 8/9 + slope_down level 9` 的 reset exposure 从 checkpoint
`3.125%` 提高到 target `10%`。Run：

```text
logs/rsl_rl/go2_velocity/2026-07-21_16-21-46_go2_rough_v7_high_slope_sampling_probe_2048env_400iter
```

训练耗时 `18m37s`，无 NaN/OOM/stall。Final sampler 为
`1996/19966=9.996995%`，所有 checkpoint 的 optimizer、terrain state、sampler
RNG/quota/counter/histogram 完整。训练管线和单变量合同 PASS。

阶段筛选按预声明 lexicographic 规则选出 `model_13900.pt`，不是 final：

```text
checkpoint  total/min-route/weighted-vx/terms  straight/arc/S
13700       4/0/.508/18                        0/1/3
13800       9/2/.453/12                        4/2/3
13900       9/3/.548/13                        3/3/3   <- candidate
13999       8/2/.459/13                        3/2/3
```

完整 high-slope completion：

```text
                 candidate 13900     final 13999       required
clean            5/16,3/16,5/16     2/16,4/16,4/16   >=12/16 each
randomized       3/16,4/16,4/16     4/16,3/16,4/16   >=10/16 each
```

Candidate gain 为 clean `.528/.634/.693`、randomized `.461/.515/.522`，均未达到
`.80`；相对 V7 completion 也未达到每路线 `+0.20`。Candidate 多个 matched slot 的
slip/action P95 超过 V7 `1.2x`，final clean weighted slip 达到 V7 的
`1.24x/1.28x/1.35x`。High-slope Model Gate 对 candidate/final 均 FAIL。

通用回归方面，candidate 的 patch clean/randomized 都是 `48/48`，continuous 都是
`12/12`，stairs up/down seeds42/43/44 均 `3/3`，没有通用遗忘。Final 虽也保持 patch
和 continuous，却把 stairs 降到 up `2/3`、down `1/3`，新增一次 base 与两次 calf
failure。Aggregate fixed-command tracking 看似只小幅下降，但 retained-scene 审查发现
randomized `forward_0.3` stairs gain 从 `0.681 -> 0.526`，candidate continuous calf
contact rate 约为 V7 的 3 倍，因此安全与保留场景 gate 也 FAIL。

最终结论：**REJECT `model_13900.pt` 和 `model_13999.pt`，不追加训练。** 默认部署模型
继续是：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

这次失败说明单纯把高坡 exposure 从约 3% 提到 10% 不足以解决持续高坡欠速。下一步先
诊断高坡 straight forward gain 为什么仍低于约 0.56；若确认是落脚/步长不足，再设计
command/terrain-conditioned foot-placement 或 step-length 单变量 probe。不要继续提高
hard-case ratio，不同时改 reward、termination、gait 或网络。

## 2026-07-21 Final high-slope diagnosis 与训练就绪（历史）

当前 integration 分支为 `exp/final-slope-diagnosis-integration`，probe implementation
HEAD 为 `19bf43b`。默认部署模型仍是 V7：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

Controller-headroom A/B 已完成。只把 closed-loop lateral/yaw limits 从 scale 1.0
放宽到 1.5 后，clean straight/arc/S 仍为 `0/4,0/4,0/4`，randomized 仍为
`0/4,0/4,0/4`；scale 1.5 的 randomized mean vx gain 为 `0.488/0.515/0.508`，
maximum saturation 均 `<0.01`。完成率没有恢复，contact/fall/low-progress failure 仍在，
最终归因为 `sustained_slope_locomotion_limited`，controller-vs-policy confidence HIGH。
Evaluator/Artifact Gate PASS，Model Gate FAIL。

正式 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/final_high_slope_headroom_clean_seed42_r2p5_v0p5_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/final_high_slope_headroom_randomized_seed42_r2p5_v0p5_2400steps.json
```

唯一允许的训练 probe 已实现并验收：task
`Unitree-Go2-Rough-V7-HighSlopeProbe`，只把
`H = slope_up levels 8/9 + slope_down level 9` 的 reset exposure 从 nominal `3.0%`
（checkpoint `64/2048=3.125%`）提高到 target `10%`。Sampler 保留原
`terrain_levels_vel`，在 curriculum 后、base reset 前只转换最少数量的 slot；H/non-H
条件分布、V7 reward、command、terrain geometry、termination、gait、randomization、
height scan、observation 和 network 均冻结。Sampler RNG/quota/count/histogram 已进入
checkpoint state；V7 warm start 会在 terrain restore 后清除 preload reset 的幽灵统计。

最终 Acceptance：targeted `70/70`、full `321` PASS（1 个既有无关 skip），compile、
CLI、registry/config/runner 和 diff-check PASS。真实 2048-env GPU strict warm-start：
restore 后 H=`64/2048`、sampler counters=0；首次真实 full reset 为
`204/2048=9.96094%`，只改变理论最少的 `179` slot，origin error=0，root-relative
error max=`3.81e-6`，observation finite，sampler state round-trip PASS。无残留进程，
GPU 空闲。

Training Gate 为 **TRAINING-READY**，但本轮没有调用 `learn`、没有新 run/checkpoint。
下一轮直接从 V7 `model_13600.pt` warm start：2048 env、400 iterations、seed42、唯一
变量 `target_hard_case_ratio=0.10`。固定命令和训练后完整验收口径见
`docs/PROJECT_JOURNAL.md` 最新章节及
`docs/reviews/final_slope_training_decision.md`。训练后必须重跑 clean/randomized
high-slope matched、stairs seeds 42/43/44 和 V7 regression；slip/action acceleration
不得超过 V7 `1.2x`。任一 gate 失败即拒绝新模型并继续保留 V7。

## 2026-07-21 High-slope matched attribution（历史）

当前 integration 分支 `exp/high-slope-attribution-integration`，HEAD `8aba90d`。
已实现并验收 strict high-slope matched straight/arc/S evaluator 和离线归因；
PRE-GPU 全量 276 tests PASS（1 skipped），V7 注册、配置加载、CLI、compileall、
diff-check、worktree/process/GPU clean 均通过。GPU 后发现 completed 场景曾输出
`first_failure_reason="none"`，已修复为 JSON null 并完整重跑，正式使用 `_v2.json`。

V7 `model_13600.pt` 在 r=2.5、high/extreme slope up/down、v=0.3/0.5、seed42、
2400-step clean matched 中：straight `4/16`、arc `3/16`、S `3/16`；forward gain
约 `0.504/0.516/0.526`。randomized 为 `5/16、4/16、4/16`，gain 约
`0.505/0.566/0.557`。失败普遍存在，但 arc/S controller saturation 超过预声明
阈值，clean/randomized 离线归因均为 `inconclusive_no_training`；无 3000-step retry
候选。r=4 straight 因 scan footprint 越出 18×18 patch 约 0.1776 m 被几何拒绝，
未计为策略失败。

交叉分析显示 clean 有 `12/16` matched slot 三路线共同失败，randomized 有 `11/16`
共同失败；绝大多数失败 slot 没有 saturation，少数高 saturation slot 甚至完成。
因此定性证据更支持 sustained high-slope/contact-stability 边界，而非普遍曲率耦合；
但正式 analyzer 按预声明 max-saturation gate 仍必须保持 LOW/INCONCLUSIVE。

Randomized level-9 stairs seed42/43/44：上楼梯仅 seed42 calf failure，下楼梯仅
seed43 calf failure，都是 1/3；结论为异质、低置信风险，不是稳定同方向楼梯缺陷。

本阶段固定 **NO-GO，未训练、无新 checkpoint**。默认部署模型仍是：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

下一步优先做 controller/command-tape 对照，拆开 closed-loop saturation 与 policy
高坡执行能力。建议只针对 slope-up high 与 slope-down extreme 做 controller-headroom
A/B，冻结 policy、terrain、route、seed 和 horizon，只统一放宽 lateral/yaw controller
limit 并记录分轴 saturation；未形成唯一归因前不得启动 hard-case sampling PPO。

## 2026-07-16 High terrain boundary 与完整 rollout metrics（最新）

当前 integration 分支为 `exp/terrain-boundary-integration`，基线 `5d233f4`。本轮
补齐 terrain route 的 action acceleration / slip mean、P95、max，以及非终止
base/upper-leg/calf contact count/rate；所有统计以原 attempt 的 active control-step
为分母，reset/completion/failure 后冻结。Straight/curve 也新增 cross-track/heading
P95 和逐 scenario progress ratio；修复 straight `--steps` 未同步延长 episode timeout
的问题。

Evaluator Gate 最终 PASS：239 tests PASS、1 个既有 intentional skip；6 份正式 GPU
JSON 的 finite/schema/sample freeze/P95/contact/placement/corridor/scan/coverage 全部通过。
Low/medium clean arc 为 `64/64`。High/extreme clean arc 和 S 均为 `37/64`；失败集中在
长距离 pyramid slope，random rough 为 `16/16`。High-slope randomized arc 为 `7/32`。
失败场景 controller saturation 仅约 `1.1--2.8%`，commanded vx 约 `0.39 m/s`、actual
仅 `0.16--0.17 m/s`，且 scan margin `>1.27 m`，因此是可信 locomotion 能力边界，
不是 controller 或 patch 越界。

Continuous straight levels 7/9 clean 为 `12/12`；randomized 为 `10/12`，level-9
stairs up/down 各因 calf contact reset 一次。旧 `...12env_1800steps.json` 的单项
time-out 来自 evaluator timeout 未延长，不能用于模型结论；正式 clean 文件是
`...12env_2400steps.json`。

本轮固定 **NO-GO，未训练、无新 checkpoint**。V7 `model_13600.pt` 仍为默认模型，
但不宣称 high/extreme sustained slope 已通过。下一步先在同一 18 m high-slope patch
做 matched straight/arc/S，区分长坡暴露与曲线耦合；再对 randomized level-9 stairs
做多 seed 复现，之后才定义单变量 hard-case sampling probe。

## 2026-07-16 Matched randomization 与 terrain curve smoke（最新）

当前 integration 分支为 `exp/terrain-curve-matched-integration`。严格 matched flat
straight/arc/S 使用统一路长 `2*pi*r/3`、相同 seed/profile/horizon，并实际执行 10-step
settle 和输出 command energy。Clean 三类均 `18/18`；full-randomized 均 `17/18`，
唯一未完成是共同的 slow `r=4,v=0.3,right` step-limit、零 reset。Randomized action
acceleration straight/arc/S 为 `0.23202/0.23257/0.23292`，S 仅高 `0.39%/0.15%`，
明确不是 S 特异问题。

因素归因：S clean `0.07524`，dynamics-only `0.07705`，push-only `0.07563`，
observation-only `0.22738`；actor corruption only `0.22810`，encoder bias only
`0.07622`。动作粗糙度主要来自 actor observation corruption，不是曲线、push 或
dynamics randomization。

新增 18 x 18 m evaluation-only terrain curve evaluator。Low/medium slope up/down、
random rough、discrete obstacle：clean arc `64/64`，clean S 补 horizon 后 `64/64`；
randomized arc `64/64`，randomized S 的 slow r4/v0.3 子矩阵补 horizon 后全部通过。
所有 rollout 零 reset/termination，terrain assignment error `5.96e-8`、route placement
error `0`。旧 8 x 4 m transition patch 不容纳曲线，明确拒绝；未覆盖 continuous
transition 或 stairs curve。

Terrain 结果目前只算可信 smoke：没有 slip/action P95/max 和非终止 base/upper/calf
contact rate，不能声称完整 formal complex-terrain gate。训练决策仍为 **NO-GO**，
本轮未训练、没有新 checkpoint；默认模型继续是 V7 `model_13600.pt`。下一步先补齐
terrain matched metric aggregation，再扩 high difficulty/transition，不改 reward/sampler。
最终 Acceptance Agent 在 integration `f7e7ceb` 上给出 PASS：200 tests PASS、1 个按设计
skip，关键 GPU JSON、CLI、V7 注册、worktree 和残留进程检查全部通过。

## 2026-07-16 S 弯瞬态与 randomized flat 结果（最新）

当前 integration 分支为 `exp/s-curve-transient-integration`。已新增逐控制步/逐 segment
的 response gain、IAE、rise/overshoot/settling、yaw sign-switch latency、command
slew/saturation、slip 和 action acceleration；S command-tape 严格按 step index 换向，
reset 后 attempt 冻结。near-zero command 的 gain 现在输出 `null`，不会把接近零的
`vy` 噪声误报为大 gain。

V7 clean S command-tape 在理想固定时长内为 `0/18`，ID 的两段 vx gain 为
`0.7902/0.8209`，wz gain 为 `0.9408/0.9357`，yaw 换向延迟 `0.0457 s`，零 reset/contact。
它说明普通 forward under-gain，不是 S 弯换向失效。clean closed-loop 为 `18/18`，
mean lateral RMS `0.00659 m`、最大 lateral error `0.03174 m`、mean heading RMS
`1.42 deg`、最大 heading error `4.66 deg`，零 reset/termination、saturation `0`。

randomized flat closed-loop 2000-step 矩阵为 `17/18`、mean progress `0.99939`，唯一
`r=4.0,v=0.3,right` 在 `98.9%` 因 step limit 结束；延长至 2400 steps 后 `1/1`
完成，确认不是 policy failure。原矩阵 lateral RMS mean `0.02653 m`、lateral max
`0.12909 m`、heading RMS mean `1.89 deg`、heading max `6.48 deg`，零 reset/contact。
randomized tape 的 ID yaw 换向为 `0.0486 s`，仍无带宽短板。randomized action
acceleration `0.23258` 约为 clean `0.07524` 的 `3.1x`，但缺少同 profile straight/arc
matched reference，后续必须先补参考，不能据此训练。

Training Decision Agent 结论为 **NO-GO**：没有授权 15% curve sampler、transition
sequence sampler 或 PPO；本轮未训练、没有新 checkpoint。当前 evaluator 只覆盖
flat curves，明确不覆盖 rough curves 或 terrain transitions；不得宣称已验证坡地、
rough、障碍或楼梯曲线。下一步优先实现 scan/corridor/relocation 兼容的复杂地形曲线
evaluator，或先补 randomized straight/arc matched action-acceleration reference。

正式 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_command_tape_clean_seed42_18env_1600steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_clean_seed42_18env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_randomized_flat_seed42_18env_2000steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_closed_loop_r4v03_right_randomized_seed42_1env_2400steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_s_curve_command_tape_randomized_flat_seed42_18env_1600steps.json
```

默认模型继续是 V7：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

## 2026-07-16 当前状态（优先于下方历史记录）

当前目标已从横移专项调整为统一 Go2 policy 的路径执行闭环：

```text
global/parameterized path
-> local body-frame vx/vy/yaw controller
-> one V7 locomotion policy
-> measured path/terrain completion
```

最新训练前审计分支为 `exp/curve-pretrain-integration`。本轮没有实现 curve sampler、
没有修改生产 curriculum/reward/terrain/termination/gait/network，也没有启动训练。
三个独立 Agent 已完成 curriculum、command response 和 acceptance 审计。

结论为 **NO-GO：当前不授权 15% curve sampler 或 PPO**。有效 scheduled tape 的
14 个 V7 general-yaw ID 场景为：`vx gain=0.8108`、`wz gain=0.9525`、平均 progress
`0.8558`；clean flat pure forward `0.6 m/s` gain 为 `0.8540`，同速 ID coupled
gain 为 `0.8490`，仅低约 `0.6%`。补足 horizon 后 closed-loop 为 `18/18`、零 reset。
因此缺口是已有通用 forward under-gain，没有证据表明 forward+yaw 存在额外耦合退化。

同时确认 `terrain_levels_vel()` 只使用 episode 末净位移和最后一个平移命令：完美
完整圆会降级，纯 yaw 失败没有降级信号，换向抵消及零命令 settle 会改变难度判定。
若未来确有 coupled 短板，matched control/probe 必须都以 2048 env 从
`model_13600.pt` 恢复相同 level/type，移除 `terrain_levels` term 后全程冻结；
首轮不要同时上线 command-aware curriculum。checkpoint 已确认保存完整的 2048 个
levels/types，mean level `5.2754`、level `0..9`、type `0..19`。

新增离线诊断：

```text
scripts/diagnose_go2_command_response.py
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/curve_command_diagnostics_offline_seed42.json
```

下一步先完成 clean S 弯、randomized/rough 曲线，或补充带逐步时序的严格 matched
pure-forward/pure-yaw/coupled 短诊断。历史 JSON 没有 rise time/overshoot 序列，
不得从 aggregate mean 反推。默认模型继续是 V7 `model_13600.pt`。

当前 curved-route integration 分支为 `exp/curved-route-integration`，曲线路径
代码/测试基线为 `e95a4bc`。新增固定半径圆弧/S 弯纯 tensor geometry、`command_tape` 和
`closed_loop` evaluator，以及独立 acceptance tests。Test Agent 在最终 HEAD
给出 PASS：34/34 tests、py_compile、CLI、V7 registration/import、非法 steps、
diff-check 和两项 GPU smoke 均通过；placement error 为 0、无 reset、JSON finite。

V7 clean arc baseline：真正按理想时间调度的 command-tape 为 `0/18` completion，
progress ratio `0.812..0.922`；closed-loop 在 1200 steps 为 `16/18`，两项
`r=4.0,v=0.3` 增加到 1600 steps 后 `2/2` 完成，故足够时长下闭环圆弧通过。
无证据授权修改 policy：本阶段未启动训练、没有新 checkpoint。下一步先跑 S 弯，
再做 randomized/rough 曲线路径；楼梯曲线必须先验证 corridor geometry。

有效 JSON：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_command_tape_clean_seed42_18env_1200steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_closed_loop_clean_seed42_18env_1200steps.json
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/route_baseline_curved_arc_closed_loop_r4v03_clean_seed42_2env_1600steps.json
```

旧的 `route_baseline_curved_arc_command_tape_clean_seed42_72.json` 使用了随实际
progress 无限延长命令的错误 tape 语义，结论无效，不得引用。

默认部署模型仍为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
```

不要采用 lateral pose probe 的 `model_13900.pt` 或 `model_14099.pt`。本阶段没有
训练，也没有新 checkpoint。

当前 integration 分支为 `exp/complex-path-integration`。已新增参数化直线路径
evaluator `scripts/evaluate_go2_routes.py`、纯 tensor route helper、两组独立测试和
training design review。关键 integration commits：

```text
c14c18e  training design review
7ef2fb5  acceptance tests
cb9a9d5  route evaluator
0038222  place route after wrapper reset
351c6ea  preserve rollout step index
```

Test Agent 在最终 integration HEAD `351c6ea` 上确认 PASS：route tests、编译、
V7 registration/import、CLI 和 diff-check 已通过；Integration Agent 的 GPU smoke
也通过，最终无残留训练/评估进程。
V7 baseline 结果：

```text
flat open-loop clean:                  16/16 complete
flat line-follow clean:               144/144 complete
flat line-follow randomized:          144/144 complete
7-patch line-follow clean:            112/112 complete
7-patch line-follow randomized:       112/112 complete
7-patch offsets clean, levels 3/7:    503/504 complete
唯一失败: pyramid_stairs level 7, yaw +0.2 rad, illegal calf contact
```

所有正式 JSON 位于 V7 run 目录，前缀为 `route_baseline_`；详细命令、路径和指标见
`docs/PROJECT_JOURNAL.md` 的 `2026-07-15` 章节。

现已增加 evaluation-only continuous suite：同一 8 m x 4 m patch 内包含入口平地、
完整 stairs/slope 和出口平地，四类为 stairs/slope up/down。它不注册训练 task，
不修改 V7 配置或 checkpoint。最终 scan-safe contract 是 start `1.0 m`、feature
`2.0..4.4 m`、end `7.0 m`、route `6.0 m`；terrain scan `+-0.8 m` 不跨 patch。

最终 V7 continuous baseline：clean `64/64`、randomized `64/64`、带
`+-0.2 m` cross-track 和 `+-0.2 rad` yaw 初始偏差的 clean `144/144`，全部零 reset、
零 termination，terrain placement error 为 0。Test Agent 在最终 HEAD `3b2b471`
给出 PASS（41/41 tests）。直线完整楼梯/坡地过渡 gate 已通过，没有证据支持此时
修改 reward 或启动 PPO；V7 `model_13600.pt` 继续作为默认模型。

Coverage 应准确表述为 `continuous_intra_patch_transitions=true`、
`continuous_inter_patch_transitions=false`：已经证明单 patch 内 approach->feature->exit，
没有证明跨生成 patch 的无缝世界。下一步可进入固定半径圆弧、S 弯、forward+yaw、
stop-and-go 和急转恢复的路径 controller baseline；仍先评估，不先训练。

Viser 网页回放已支持固定命令和指定连续地形。推荐上楼梯命令：

```bash
python scripts/play.py Unitree-Go2-Rough-V7 \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt \
  --terrain-demo stairs_up \
  --terrain-level 5 \
  --fixed-vx 0.4 \
  --fixed-vy 0.0 \
  --fixed-yaw-rate 0.0 \
  --viewer viser
```

`terrain-demo` 还支持 `stairs_down`、`stairs_up_down`、`slope_up`、`slope_down`。Demo 会自动使用单
环境、固定入口/yaw、clean profile、固定 body command，并关闭 terrain 随机重选。

连续上楼再下楼使用 `--terrain-demo stairs_up_down`。该 demo 是 12 m x 4 m 单
patch：8 级上楼梯、顶部平台、8 级下楼梯和出口平地。

当前项目：`/home/jensen/projects/unitree_rl_mjlab`

目标：沿 Unitree 官方 `unitree_rl_mjlab` 路线优化 Go2 rough-terrain locomotion。当前默认模型是 V7 `model_13600.pt`；lateral-conditioned hip pose tolerance 单变量探针已完成但未通过足端摆幅/稳定性 gate，下一步应研究 command-conditioned foot-placement/step-length reward。

## 已完成

- 保留旧项目：`/home/jensen/projects/quadruped_mujoco_rl`
- 使用官方仓库：`/home/jensen/projects/unitree_rl_mjlab`
- 当前官方 HEAD：`1425b15 Fix the warnings during rough-terrain training.`
- Conda 环境：`unitree_rl_mjlab`
- 关键依赖修正完成：
  - `mujoco==3.5.0`
  - `mujoco-warp==3.5.0`
  - `mjlab==1.2.0`
  - `warp-lang==1.12.0`
  - `scipy`
- `python scripts/list_envs.py` 已可用。
- `Unitree-Go2-Flat` smoke test、512 env、1024 env、2048 env resume 训练均已完成。
- `Unitree-Go2-Rough` smoke test、3000 iter、resume +1500 iter、resume +4500 iter 均已完成。
- 已实现 Go2 Rough low-speed curriculum v1，并通过 128 env / 2 iter smoke test。
- Go2 Rough low-speed curriculum v1 / 1024 env / 3000 iter 已完成。
- Go2 Rough low-speed curriculum v1 resume2999 / plus 3000 iter 已完成。
- 已实现 Go2 Rough forward curriculum v2，并通过 128 env / 2 iter smoke test。
- Go2 Rough forward curriculum v2 resume5998 / plus 2000 iter 已完成。
- 已实现 Go2 Rough contact-clean curriculum v3，并通过 128 env / 2 iter smoke test。
- Go2 Rough contact-clean curriculum v3 resume7997 / plus 2000 iter 已完成。
- 已新增 `Unitree-Go2-Flat-RoughObs`，用于训练 rough-compatible flat prior，并通过 128 env / 2 iter smoke test。
- `Unitree-Go2-Flat-RoughObs` / 2048 env / 3000 iter 已完成，结果优于普通 Flat baseline。
- `Unitree-Go2-Rough` 从 Flat-RoughObs prior warmstart / 2048 env / plus 3000 iter 已完成。
- 已新增并注册 `Unitree-Go2-Rough-V4`，随机策略、v3 checkpoint warmstart smoke 和 2048 env / plus 2000 iter 正式训练均已完成。
- 已实现并注册 `Unitree-Go2-Rough-V5`：按 base、upper-leg、calf 拆分接触并改为摆动脚 clearance；2048 env / 500 iter 探针已完成。
- 已恢复当前工作区中缺失的 `unitree_go2_flat_rough_obs_env_cfg()`，`scripts/list_envs.py` 重新可用。
- TensorBoard 可用：

```bash
tensorboard --logdir logs/rsl_rl/go2_velocity
```

## 当前最佳 Flat 模型

最新推荐 Flat baseline：

```text
logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt
logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/policy.onnx
```

评价指标：

```text
Train/mean_reward: 54.71
Train/mean_episode_length: 994.19
Episode_Termination/fell_over: 0
Episode_Termination/illegal_contact: 0
Episode_Reward/track_linear_velocity: 0.841
Episode_Reward/track_angular_velocity: 0.937
Metrics/slip_velocity_mean: 0.072
Episode_Metrics/mean_action_acc: 0.667
```

结论：Go2 Flat 已经稳定跑通，2048 env resume 版本优于早期 1024 env baseline，建议作为当前 Flat baseline。

## 当前 Rough 状态

已完成 run：

```text
logs/rsl_rl/go2_velocity/2026-07-06_10-04-23_go2_rough_1024env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-06_13-40-01_go2_rough_resume2800_plus1500iter/model_4299.pt
logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/model_8798.pt
logs/rsl_rl/go2_velocity/2026-07-07_10-11-16_go2_rough_low_speed_curriculum_v1_1024env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/model_5998.pt
logs/rsl_rl/go2_velocity/2026-07-09_10-32-58_go2_rough_forward_curriculum_v2_resume5998_plus2000iter/model_7997.pt
logs/rsl_rl/go2_velocity/2026-07-09_14-36-38_go2_rough_contact_clean_v3_resume7997_plus2000iter/model_9996.pt
logs/rsl_rl/go2_velocity/2026-07-10_11-56-36_go2_rough_from_flat_roughobs_prior_2048env_plus3000iter/model_5998.pt
logs/rsl_rl/go2_velocity/2026-07-13_11-14-09_go2_rough_v4_relative_clearance_contact_2048env_plus2000iter/model_11995.pt
logs/rsl_rl/go2_velocity/2026-07-13_13-17-21_go2_rough_v5_bodypart_contact_probe_2048env_500iter/model_10699.pt
logs/rsl_rl/go2_velocity/2026-07-13_15-06-34_go2_rough_v5_1_orientation_gated_calf_probe_2048env_500iter/model_11198.pt
logs/rsl_rl/go2_velocity/2026-07-13_15-36-37_go2_rough_v5_1_orientation_gated_calf_controlled_2048env_plus500iter/model_11697.pt
logs/rsl_rl/go2_velocity/2026-07-13_16-56-07_go2_rough_v6_curriculum_persistent_hfield_dr_2048env_2000iter/model_13000.pt
```

关键结论：

```text
3000 iter rough tail100 reward: 39.74, terrain level: 1.11, linear tracking: 0.667
resume +1500 tail100 reward: 40.48, terrain level: 1.25, linear tracking: 0.684
resume +4500 final tail100 reward: 38.24, terrain level: 0.15, linear tracking: 0.288
low-speed v1 tail100 reward: 40.71, terrain level: 2.82, linear tracking: 0.740
low-speed v1 resume tail100 reward: 43.76, terrain level: 2.97, linear tracking: 0.765
forward v2 tail100 reward: 35.30, terrain level: 4.76, linear tracking: 0.644
contact-clean v3 tail100 reward: 39.45, terrain level: 4.36, linear tracking: 0.667
prior warmstart 2048 tail100 reward: 39.39, terrain level: 4.37, linear tracking: 0.666
V4 tail100 reward: 40.61, terrain level: 4.15, linear tracking: 0.731
V5 probe tail100 reward: 44.53, terrain level: 3.56, linear tracking: 0.758
V5.1 probe tail100 reward: 48.92, terrain level: 3.77, linear tracking: 0.799
V5.1 controlled +500 tail100 reward: 48.09, terrain level: 3.79, linear tracking: 0.795
V6 model_13000 tail100 reward: 50.88, terrain level: 5.33, linear tracking: 0.837
```

5000 PPO iterations 附近速度课程扩张后，linear tracking 明显崩掉，terrain level 也掉到很低。最后的 `model_8798.pt` 不推荐作为最佳 rough 模型；如果要回放这条线，优先看：

```text
logs/rsl_rl/go2_velocity/2026-07-06_15-19-55_go2_rough_resume4299_plus4500iter/model_4900.pt
```

## 回放命令

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt \
  --num-envs 1
```

如果 viewer 有问题：

```bash
python scripts/play.py Unitree-Go2-Flat \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-06-26_10-56-49_go2_flat_2048env_resume999_plus1000iter/model_1998.pt \
  --num-envs 1 \
  --viewer viser
```

Rough 当前建议回放：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/model_5800.pt \
  --num-envs 1 \
  --viewer viser
```

最终 checkpoint 回放：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-07_13-55-40_go2_rough_low_speed_curriculum_v1_resume2999_plus3000iter/model_5998.pt \
  --num-envs 1 \
  --viewer viser
```

V2 高地形 checkpoint 回放：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-09_10-32-58_go2_rough_forward_curriculum_v2_resume5998_plus2000iter/model_7997.pt \
  --num-envs 1 \
  --viewer viser
```

V3 contact-clean checkpoint 回放：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-09_14-36-38_go2_rough_contact_clean_v3_resume7997_plus2000iter/model_9000.pt \
  --num-envs 1 \
  --viewer viser
```

Prior warmstart 2048 checkpoint 回放：

```bash
python scripts/play.py Unitree-Go2-Rough \
  --checkpoint-file logs/rsl_rl/go2_velocity/2026-07-10_11-56-36_go2_rough_from_flat_roughobs_prior_2048env_plus3000iter/model_5600.pt \
  --num-envs 1 \
  --viewer viser
```

## 已知注意点

- 训练命令使用 `--agent.max-iterations`，不是 `--runner.max-iterations`。
- 训练建议显式加 `--agent.logger tensorboard`，避免默认 W&B 登录问题。
- `.pt` 用于回放/继续训练；`policy.onnx` 是部署推理用 actor，不是完整训练 checkpoint。
- `play.py` 里速度命令随机采样，所以轨迹看起来随机；不是回放固定路线。
- `Unitree-Go2-Flat` 平地任务较轻，1024 env 在 RTX 5060 Laptop 8GB 上可以跑。
- Flat 容量测试显示 8192 env 可跑短测试，但长训建议 4096/6144；Rough 更重，初始仍建议 1024。
- Rough 使用 height scan/raycast 地形高度观测，不是相机视觉。
- Rough 的 `terrain_levels` 是难度等级，不是高度米数。
- 旧 Rough 的 terrain scan 包含 geom group 2，会命中 Go2 visual mesh；V4 已限制为 terrain group 0。为保持历史实验可复现，旧任务没有回改。

## 下一步

1. 不建议继续原始 Rough 配置从 `model_8798.pt` 往后硬训。
2. v1 已改内容：
   - `max_init_terrain_level=2`
   - `rel_heading_envs=0.0`
   - 初始 `lin_vel_x=(0.0, 0.8)`
   - 初始 `lin_vel_y=(-0.2, 0.2)`
   - 初始 `ang_vel_z=(-0.5, 0.5)`
   - 8000 iter 和 12000 iter 后才小幅扩速
3. v1 resume 结果：reward、线速度跟踪、fell_over、illegal_contact、slip 和动作平滑度都比第一段 v1 更好；terrain level 稳在约 3。
4. v2 已改成更强的 forward curriculum：
   - `rel_standing_envs=0.02`
   - `rel_heading_envs=0.0`
   - 初始 `lin_vel_x=(0.2, 0.8)`
   - 初始 `lin_vel_y=(-0.1, 0.1)`
   - 初始 `ang_vel_z=(-0.3, 0.3)`
   - 10000 iter 后才扩到稍宽速度范围
5. v2 结果：terrain level 成功到约 4.76，但 reward、linear tracking、illegal_contact 和 slip 都变差。v2 是高地形探索策略，不是当前最佳展示策略。
6. 展示优先用 v1 resume 的 `model_5800.pt` 或 `model_5998.pt`；研究突破 terrain 时再看 v2 的 `model_7997.pt`。
7. v3 已加入接触/抬脚/稳定性约束：
   - `lin_vel_x=(0.15, 0.8)`
   - `body_orientation_l2` weight `-1.2`
   - `action_rate_l2` weight `-0.07`
   - `foot_clearance` weight `-1.2`
   - `foot_clearance.target_height=0.12`
   - 新增 `nonfoot_contact` reward，weight `-2.0`
8. v3 正式训练已完成：

```text
v2 tail100: reward 35.30, terrain 4.76, linear tracking 0.644, illegal 0.304, slip 0.110, action_acc 1.103
v3 tail100: reward 39.45, terrain 4.36, linear tracking 0.667, illegal 0.204, slip 0.086, action_acc 0.881
```

9. v3 结论：有用但不是肉眼质变。它把 illegal contact、slip 和 action acceleration 明显降下来，但 terrain level 从 v2 的 4.76 降到 4.36，linear tracking 只小幅回升。继续硬训 v3 预计边际收益不高。
10. 已新增 rough-compatible flat prior 任务：
   - 任务名：`Unitree-Go2-Flat-RoughObs`
   - 平地 terrain，但保留 `terrain_scan` / actor `height_scan` / critic `height_scan`
   - actor shape `234`，critic shape `261`，与 `Unitree-Go2-Rough` 一致
   - smoke run：`logs/rsl_rl/go2_velocity/2026-07-09_15-52-41_go2_flat_roughobs_prior_smoke`
11. `Unitree-Go2-Flat-RoughObs` 正式训练已完成：

```text
logs/rsl_rl/go2_velocity/2026-07-09_16-04-12_go2_flat_roughobs_prior_2048env_3000iter/model_2999.pt
logs/rsl_rl/go2_velocity/2026-07-09_16-04-12_go2_flat_roughobs_prior_2048env_3000iter/policy.onnx
```

Tail100:

```text
reward 55.96
episode length 997.37
linear tracking 0.884
angular tracking 0.946
illegal_contact 0.0079
slip 0.057
action_acc 0.563
```

结论：prior shape 兼容 Rough，平地指标也优于普通 Flat 2048 baseline，是当前最合适的 rough warmstart 起点。
12. prior warmstart rough / 2048 env 已完成：

```text
logs/rsl_rl/go2_velocity/2026-07-10_11-56-36_go2_rough_from_flat_roughobs_prior_2048env_plus3000iter/model_5998.pt
tail100: reward 39.39, terrain 4.37, linear 0.666, illegal_contact 0.317, slip 0.0866, action_acc 0.874
model_5600.pt: reward 39.59, terrain 4.47, linear 0.685, illegal_contact 0.270, slip 0.088, action_acc 0.892
```

13. prior warmstart 结论：

```text
terrain level 达到 v3 同档
slip 和 action_acc 接近 v3
illegal_contact 明显高于 v3
不是新的最佳 rough 策略，不建议继续原样硬训
```

14. 已实现独立的 `Unitree-Go2-Rough-V4`：
   - `foot_clearance` 改为相对脚下最近 terrain-scan 点的高度
   - terrain scan 限制为 geom group 0，排除 Go2 visual mesh 污染
   - 非足端接触改为 5 N soft threshold 后连续增加的 force cost，weight `-1.5`
   - illegal contact 改为 35 N 且连续 2 个仿真子步才终止
   - 新增 `Metrics/nonfoot_contact_force_mean`
   - actor/critic shape 仍为 `234/261`
   - 有效 smoke run：`logs/rsl_rl/go2_velocity/2026-07-10_14-48-57_go2_rough_v4_from_v3_9996_smoke`
15. V4 正式训练已完成：

```text
logs/rsl_rl/go2_velocity/2026-07-13_11-14-09_go2_rough_v4_relative_clearance_contact_2048env_plus2000iter/model_11995.pt
tail100: reward 40.61, terrain 4.15, linear 0.731, angular 0.819
tail100: fell_over 0.0079, illegal_contact 0.301, slip 0.100, action_acc 0.971
训练耗时约 1.47 小时
```

16. V4 结论：线速度跟踪明显提升，但 terrain 略降，illegal contact、slip 和动作加速度变差。V4 的 illegal-contact 终止阈值更宽松但发生率仍更高，说明严重持续接触确实增加。V4 不是新的综合 best，不建议原样继续训练。
17. V4 建议回放 `model_10200.pt` 和 `model_11995.pt`；高 terrain 综合策略仍优先 v3 `model_9000.pt` / `model_9996.pt`。
18. V5 已按 body/leg 部位拆分非足端接触统计，并把 clearance 限制到摆动脚；2048 env / 500 iter 探针已完成：

```text
logs/rsl_rl/go2_velocity/2026-07-13_13-17-21_go2_rough_v5_bodypart_contact_probe_2048env_500iter/model_10699.pt
tail100: terrain 3.558, linear 0.758, angular 0.847, slip 0.095, action_acc 0.884
tail100 termination: base 0.0142, upper-leg 0.0158, calf 0.1817
tail100 active force: base 12.0 N, upper-leg 7.6 N, calf 50.0 N
耗时约 22.6 分钟
```

19. 已新增 `scripts/diagnose_calf_contacts.py` 并对 `model_10400.pt` / `model_10699.pt` 做 512 env、固定 0.6 m/s、1500 steps 自动事件诊断。完整结果：`calf_diagnostics_512env_1500steps.json`。
20. 诊断结论：当前无条件 `60 N × 3 substeps` 会终止大量可恢复 calf 擦碰，但完全删除终止也不安全。V5.1 推荐改为 `60 N × 3 substeps + body tilt > 15°`，保留 calf soft penalty；先做短探针复验。
21. V5.1 2048 env / 500 iter 探针已完成，耗时约 22 分 10 秒：

```text
logs/rsl_rl/go2_velocity/2026-07-13_15-06-34_go2_rough_v5_1_orientation_gated_calf_probe_2048env_500iter/model_11198.pt
tail100: reward 48.921, terrain 3.766, linear 0.799, angular 0.888
tail100: slip 0.0845, action_acc 0.774, fell_over 0.0333
tail100 termination: base 0.0229, upper-leg 0.0308, calf 0.0367
```

22. V5.1 在相同 terrain `3.560` 的连续 100 iter 窗口中：linear `0.801`、slip `0.0838`、action_acc `0.765`、fell_over `0.0167`、calf termination `0.0354`。相同难度下 fell-over 与 V5 的 `0.0142` 接近，而 calf termination 比 V5 的 `0.1817` 低约 80%，说明 15° 姿态门控有效。
23. V5.1 受控 +500 已完成，耗时约 22 分 17 秒：

```text
logs/rsl_rl/go2_velocity/2026-07-13_15-36-37_go2_rough_v5_1_orientation_gated_calf_controlled_2048env_plus500iter/model_11697.pt
tail100: reward 48.094, terrain 3.788, linear 0.795, angular 0.887
tail100: slip 0.0858, action_acc 0.817, fell_over 0.0075
tail100 termination: base 0.0146, upper-leg 0.0329, calf 0.0346
```

24. 与 `model_11198.pt` 的等难度 tail100 相比，`model_11697.pt` 的 fell-over 从 `0.0333` 降到 `0.0075`，但 linear/terrain 基本持平、action_acc 从 `0.774` 回退到 `0.817`。优先稳定性用 `model_11697.pt`，优先平滑度用 `model_11198.pt`；不再继续追加 PPO。
25. 已新增 `Unitree-Go2-Rough-V6`，V5.1 不变。runner checkpoint 现在保存每个环境的 terrain level/type；32 env 往返 smoke 将平均 level `4.469` 精确恢复为 `4.469`。旧 checkpoint 仍可加载，但第一次使用配置的起始 level。
26. 已新增 `scripts/evaluate_go2_rough.py`。V5.1 `model_11697.pt` 在 V6 最终地形上的固定基线为：overall linear error `0.146`、slip `0.049`、fell/env `0.0063`、calf term/env `0.0250`；主要弱项是上楼梯 linear error `0.238`、下楼梯 calf term/env `0.125` 和下坡 fell/env `0.0625`。
27. V6 使用轻量 heightfield 地形：15% flat、30% stairs、20% slopes、15% random rough、20% discrete obstacles；增加 payload `-1..+3 kg`、motor capacity `0.9..1.1`，push 改为每 10–15 s 的水平扰动。primitive boxes/stones 在 2048 env 只有 `1760 FPS`，已放弃；heightfield 版本恢复到 `17.5–18k FPS`。
28. V6 正式训练按用户要求在阶段 checkpoint `model_13000.pt` 安全停止，完成约 1303/2000 iter，耗时约 57 分钟：

```text
logs/rsl_rl/go2_velocity/2026-07-13_16-56-07_go2_rough_v6_curriculum_persistent_hfield_dr_2048env_2000iter/model_13000.pt
tail100: reward 50.884, terrain 5.326, linear 0.837, angular 0.911
tail100: slip 0.0775, action_acc 0.741, fell_over 0.0079
tail100 termination: base 0.0058, upper 0.0204, calf 0.0271
```

29. 固定复评显示 V6 阶段模型改善复杂地形：overall linear error `0.146 -> 0.127`、fell/env `0.0063 -> 0.0031`、calf term/env `0.0250 -> 0.0031`；上/下楼梯 linear error 分别从 `0.238/0.193` 降到 `0.142/0.123`。flat、random rough、discrete obstacles 的 linear error 小幅回退。完整 JSON：`fixed_eval_model_13000_320env_1000steps.json`。
30. `model_13000.pt` 保存的 2048-env curriculum 平均 level 为 `5.306`。剩余 697 iter 已完成，续训日志确认恢复到 `5.306`，没有重置 terrain level：

```text
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/model_13696.pt
tail100: reward 49.473, terrain 5.240, linear 0.829, angular 0.900
tail100: slip 0.0806, action_acc 0.788, fell_over 0.0058
tail100 termination: base 0.0075, upper 0.0138, calf 0.0458
```

31. seed 42 的全阶段 checkpoint 固定评估显示 `model_13500.pt` 是最佳综合点；`model_13696.pt` 的 tracking 已明显回退。
32. seed `42/43/44` 复评中，`model_13500.pt` 相对 `model_13000.pt` 的三 seed 平均：

```text
linear error: 0.1270 -> 0.1062
yaw error: 0.0527 -> 0.0474
slip: 0.0469 -> 0.0485
action_acc: 0.1043 -> 0.1015
fell/env: 0.0063 -> 0.0010
upper term/env: 0.0083 -> 0.0010
calf term/env: 0.0063 -> 0.0052
```

33. V6 当前推荐 checkpoint：

```text
logs/rsl_rl/go2_velocity/2026-07-14_09-34-39_go2_rough_v6_resume13000_remaining697iter/model_13500.pt
```

960 env 合计跌倒 `6 -> 1`，linear error 改善约 16.4%；代价是 slip 增加约 3.5%。停止原样追加 V6 PPO。
34. V6 已完成并 multi-seed，下一阶段再考虑 asymmetric PPO / teacher-student：
   - actor 只用可部署观测
   - critic/teacher 训练时可用 privileged terrain/dynamics 信息
35. 如果只是继续优化普通 Flat，可跑：

```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs=1024 \
  --agent.max-iterations=3000 \
  --agent.run-name go2_flat_1024env_3000iter \
  --agent.logger tensorboard
```

36. 如果想控制固定速度，优先研究 `--viewer viser` 的 joystick 面板，或修改 Go2 play 模式下 `twist` command 的采样逻辑。
37. `scripts/evaluate_go2_rough.py` 已支持：
   - 批量 `--command-cases`
   - `clean/dynamics/randomized` profile
   - command × level / command × terrain type 交叉汇总
   - 旧 `--command-x/y/yaw` 接口保持兼容
38. V6 鲁棒性矩阵已完成：2 checkpoints × 2 profiles × 3 seeds，每次 1120 env / 1000 steps，总计 13440 env instances。`model_13500.pt` 三 seed 全命令平均：

```text
profile     linear    yaw     slip   action_acc   fell/base/upper/calf
clean       0.1101  0.0508  0.0367    0.0784       5/0/6/19
randomized  0.1373  0.0758  0.0458    0.2390       4/3/19/32
```

39. randomized 下 `model_13500.pt` 相比 `model_13000.pt`：linear error 低 2.7%，跌倒 `9 -> 4`、upper term `22 -> 19`，但 calf term `27 -> 32`；仍以 `model_13500.pt` 为默认综合模型，`model_13000.pt` 可作为 0.9 m/s 高速复杂地形的保守备选。
40. 当前最明显缺口：
   - lateral `±0.3 m/s` 的误差约 `0.211–0.214 m/s`
   - randomized `0.9 m/s` inverted slope error `0.224`，term 为 `1/0/9/3`
   - 失败集中在 level 9；randomized term 为 `4/2/14/27`
   - 低/中速上楼梯仍有较多 calf term
41. 下一步建立独立 V7，从 `model_13500.pt` warm start，先冻结 reward/termination；显式采样纯横移、纯 yaw、高速前进，并加强高速 × level 7–9 stairs/down-slope 组合。先跑 500 iter，横移误差至少改善 15% 且 overall fell/calf 不恶化后再继续。暂不同时加入 RMA，以便归因。
42. V7 已实现 ModeVelocityCommand，保持 V6 的 reward/termination/terrain/观测不变：
   - general 40%
   - pure lateral 25%, |v_y|=0.1..0.3
   - pure yaw 15%, |w_z|=0.2..0.7
   - high-speed forward 20%, v_x=0.8..1.0
   - level >=7 的 stairs/down-slope focus terrain 将 high-speed 提升到 45%
43. V7 2048 env / 500 iter 已完成：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13999.pt
tail100: reward 50.071, terrain 5.650, linear 0.829, angular 0.901
tail100: slip 0.0795, action_acc 0.766, fell_over 0.0054
tail100 termination: base 0.0071, upper 0.0142, calf 0.0554
~~~

44. V7 阶段点 model_13600.pt 在 randomized 三 seed 中相对 baseline model_13500.pt：

~~~text
linear 0.1410 -> 0.1392
yaw 0.0813 -> 0.0754
slip 0.0465 -> 0.0440
action_acc 0.2424 -> 0.2411
fell/base/upper/calf flags: 4/1/58/53 -> 0/5/19/24
~~~

45. V7 的稳定性目标达到，但横移目标未达到：lateral_left/right 平均误差约
0.212 m/s，没有改善 15%；0.9 m/s + inverted slope error 也没有下降。
V7 推荐保留 model_13600.pt 作为 randomized stability candidate，V6
model_13500.pt 继续作为 fixed-forward benchmark；停止当前 V7 配置继续训练。
下一轮只提高 lateral mode 到 40–50% 并做 ±0.1 -> ±0.3 m/s curriculum，
暂不改 reward/termination 或引入 RMA。
46. V7.1 已注册，模式比例改为 general/lateral/yaw/high-speed =
30/45/10/15%。从 V7 model_13600.pt 的 common step 326664 恢复；前 250 iter
横移范围为 0.10–0.20 m/s，后 250 iter 扩到 0.10–0.30 m/s。
47. V7.1 2048 env / 500 iter 已完成：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter/model_14099.pt
tail100: reward 52.621, terrain 4.798, linear 0.844, angular 0.918
tail100: slip 0.0708, action_acc 0.722
tail100 termination: fell 0.0058, base 0.0046, upper 0.0192, calf 0.0271
~~~

48. randomized seed42 阶段筛选：

~~~text
                           model_13600  model_14099
overall linear                0.1382       0.1408
fell/base/upper/calf          0/4/8/6      3/2/5/9
forward_0.6 error             0.1408       0.1533
lateral_left/right average    0.2121       0.2039
~~~

49. V7.1 横移只改善约 3.9%，未达到 15%；forward_0.6 退化约 8.9%，跌倒和
calf 增加，terrain 降到 4.80。按 gate 停止，不做 seed43/44，不采用
model_14099。默认仍为 V7 model_13600.pt。
50. 下一步先扩展 evaluator 的 x/y 分量误差和 lateral response gain，并诊断
flat/stairs/inverted-slope 上的足端轨迹、接触相位和 12 关节动作。确认固定 trot
是否限制横移后，再决定 command-conditioned gait/foot-placement；诊断前不改
reward、不引入 RMA、不追加 PPO。
51. 修复 scripts/evaluate_go2_rough.py 的固定 terrain assignment：此前只更新
level/type/origin，没有将 robot root 平移到目标 patch。旧 JSON 的 overall 相对
比较仍可参考，但旧 by_level/by_terrain_type 归因无效。新评估器会平移 root、
forward/sense，并输出 terrain_assignment_position_error_max。
52. 修复后重跑 randomized seed42，默认模型结论不变：

~~~text
model       linear   yaw     slip   action_acc   fell/base/upper/calf
13500       0.1398  0.0815  0.0466    0.2425       1/1/12/15
13600       0.1393  0.0755  0.0443    0.2412       0/1/4/6
14099       0.1410  0.0743  0.0451    0.2371       0/1/1/9
~~~

53. 新增 scripts/diagnose_go2_lateral.py。clean 144-env 正确 patch 诊断显示：

~~~text
command          response gain   direction correct   fixed-trot match
forward_0.6          0.818             99.1%              95.3%
lateral_left         0.349             95.7%              94.0%
lateral_right        0.366             94.2%              91.0%
~~~

横移方向正确，但足端横向相位摆幅仅 3–4 cm，前进足端纵向摆幅为 15–16 cm；
横移的 thigh/calf action std 约 0.225/0.75–0.79，明显小于前进的
0.652/1.195，并有更多四足同时接触。策略是在侧挪，不是充分侧步。
54. 根因是 pose 与 tracking reward trade-off，而不是单纯样本不足。近似分解：

~~~text
command          pose   tracking   sum
forward_0.6      0.861    0.938   1.799
lateral_left     0.949    0.854   1.802
lateral_right    0.947    0.860   1.807
~~~

欠跟踪横移并保持默认姿态能得到与正常前进几乎相同的总 reward。固定 diagonal
trot match 仍很高，说明相位本身没有明显错误，但 foot_gait 不奖励方向或步长，
会容许低幅度 shuffle。
55. 下一步从 V7 model_13600.pt 做单变量 probe：只对 lateral-dominant command
将 hip pose tolerance 从 0.15 插值放宽到约 0.30 rad；恢复 V7 的 25% lateral
分布，其他 reward、gait、termination、terrain 和 randomization 全冻结。若仍
不能增加足端横摆，再考虑 command-conditioned foot placement；暂不改 trot
phase，不引入 RMA。
56. 已按多 Agent + 独立 worktree 流程完成实现和验收。dirty 工作区先固化为
baseline commit `e8a7eee`，integration 分支为 `exp/lateral-pose-integration`；
Reward/Analysis/Test Agent 分别提交并报告 commit。最终训练前 HEAD `5071764`
通过 16 项 unittest、编译、diff、任务注册、配置单变量审查、32-env 随机 smoke
和 128-env V7 model_13600 strict warm-start smoke。
57. 新任务 `Unitree-Go2-Rough-V7-LateralPose` 直接继承 V7，保留普通 terrain
的 general/lateral/yaw/high-speed=`40/25/15/20%`、lateral `0.1..0.3 m/s` 和
全部 terrain/randomization/termination/gait。唯一行为变化是 hip pose std：
`alpha=clamp((|vy|-max(|vx|,|wz|))/0.30,0,1)`，有效 std 从当前速度 regime
基线连续插值到 `0.30 rad`；standing 仍为 `0.05`，forward/yaw 不变。
58. 2048 env / 500 iter / seed 42 正式 probe 已完成，warm start 为 V7
`model_13600.pt`：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_15-58-33_go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter/model_14099.pt
tail100: reward 50.912, terrain 5.479, linear 0.835, angular 0.909
tail100: slip 0.07735, action_acc 0.74996
tail100 termination: fell/base/upper/calf 0.00667/0.00833/0.01792/0.03417
~~~

59. 修正 evaluator 的 randomized seed42 阶段筛选实际为 2240 env（7 commands x
4 levels x 20 columns x 4 repeats）。横移最佳阶段是 `model_13900.pt`：平均
lateral gain `0.317 -> 0.377`，平均 lateral error `0.213 -> 0.198`；
forward_0.6 gain `0.807 -> 0.805`，overall linear error `0.1390 -> 0.1352`。
代价是 slip `0.0434 -> 0.0453`，fell/base/upper/calf flags
`1/1/15/9 -> 4/2/22/16`。JSON：

~~~text
logs/rsl_rl/go2_velocity/2026-07-14_15-58-33_go2_rough_v7_lateral_pose_tolerance_probe_2048env_500iter/robustness_stage_randomized_seed42_2240env_1000steps.json
~~~

60. clean lateral diagnostic 对 `model_13900.pt` 的结论：forward gain 保持
`0.827`；左右 lateral gain `0.349/0.366 -> 0.383/0.399`，但平均仅 `0.391`，
未达到 `0.40` gate。横向相位足端摆幅仅 `3.14/3.34 cm -> 3.44/3.85 cm`，
平均 `3.64 cm`，未达到 `5 cm` gate。测试 Agent 最终判定 FAIL，拒绝
`model_13900.pt` 和 final `model_14099.pt` 作为部署模型；默认继续使用 V7
`model_13600.pt`。下一步考虑 command-conditioned foot-placement/step-length
reward，仍不先改 trot phase、不增加 lateral 采样、不引入 RMA。
61. 完成 V7 evaluation-only matched actuator-headroom counterfactual。新增
`scripts/audit_go2_actuator_headroom_counterfactual.py` 和对应 unittest；没有修改
既有正式 gait/actuator audit 脚本。最终 artifact 为
`actuator_headroom_counterfactual_clean_seed42_96worlds_1200steps_v2.json`，SHA256
`bbe25e739367d05a254be077274cdbbb00e2b459637066d78e951618328c1e4e`，evaluator
SHA256 `acf3a710ce0be858d0e0fbddd238717ba58c7e277ad9e3901c6636bcf229a116`。
62. Runtime identity PASS：control hip/thigh/calf 为 `23.5/23.5/45 Nm`，headroom
为 `29.375/29.375/56.25 Nm`；三地形 range write error 和 rollout drift 均为 0，
初始 state/observation/first action pair error 为 0，placement 小于 `1e-6 m`。
terminal reset-hook row 因 MuJoCo-Warp forward 相位差只从 demand-dependent 指标中
排除，仍保留在 force/contact/failure 终态统计中。
63. 正式结论为 **INCONCLUSIVE**。14 个可对齐饱和 cohort 的 saturated joint-steps
从 `112 -> 2`（下降 `98.2%`），但 slope-up completion 在 `.3/.5 m/s` 均为
`5/8 -> 5/8`、`2/8 -> 2/8`；10 个 pair 违反风险 guardrail，3 个 control survivor
在 headroom arm 失败，另有 5 个 probe 早于 control 终止而缺 control-aligned 100-step
窗口。若忽略 hard coverage gate，方向性证据更接近 `SATURATION_DOWNSTREAM`，不是
`ACTUATOR_CAUSAL` 或 `HEADROOM_INSUFFICIENT`，但不得升级为正式分类。
64. 本阶段没有训练、没有修改 reward/task/robot asset、没有产生候选部署模型。
默认模型仍为 V7 `model_13600.pt`。下一步若继续诊断，唯一变量建议是 1.25x
headroom 的 activation timing：保持 1.00x，直到 matched control 在线首次出现持续
saturation，再开启 1.25x，并按事件后固定窗口比较；其余配置继续冻结。

## 2026-07-23 V7 全量复杂地形因果诊断（最新）

本轮完成 evaluation-only 因子诊断并复核 actuator-headroom triplet；没有训练、没有修改
reward/task/robot asset，默认模型仍为 V7 `model_13600.pt`。

新增工具与测试：

```text
scripts/diagnose_go2_complex_terrain_causes.py
tests/test_go2_complex_terrain_causes.py
scripts/audit_go2_actuator_headroom_triplet.py
tests/test_go2_actuator_headroom_triplet.py
```

正式产物：

```text
complex_terrain_causal_diagnostic_v2.json
  SHA256 9f09d35229c6ab235df3445aa5133d4451984d78128e2cbe3f552a2e2f118520
actuator_headroom_triplet_clean_seed42_48worlds_1200steps_v1.json
  SHA256 740c6613473ba28e2b81fc20c764e7ae33a7cbf9bb8de3ff0cf23dc1a33f5511
```

因子矩阵固定 seed=42、8 repeats、100 warmup、1200 sample steps、matched slots，
比较 friction `0.3/0.6/1.2` 与 height-scan masked；placement、lifecycle、strict
recursive finite 和已有 baseline provenance 均通过。相对同一 refreshed nominal `0.6`
的 `1.2` friction 在 high slope、`vx=0.5` 上 completion `+7/8`、gain `+249%`、
local-tangent step `+227%`、stance slip `-47%`；extreme down、`vx=0.5` 为
`+8/8`、gain `+23.7%`、step `+54.3%`。clearance 没有系统性下降。height-scan
mask 使 flat 也全部失败，证明该通道全局重要，但不能单独证明 slope-specific 感知缺陷。

综合分类：**C 为最强主因（摩擦/接触/支撑稳定性不足）**；A 是短步/推进不足的症状，
B 没有得到支持，D 仍是重要放大器但不是已证实根因，E 保留为模型与非确定性限制。
当前综合 JSON 的 `training_ready=false`。

同时完成 control/sham/probe actuator triplet：source/sham/probe saturation 为
`19/10/10`，表面 probe reduction `47.4%`，但同样的 source-sham 自然差异也是
`47.4%`；完整 post-100 coverage 仅 `2` 对，低于 hard gate `8`，lifecycle
probe-vs-sham 为 `1 win/1 loss`，正式 verdict **INCONCLUSIVE**。因此不能把 actuator
saturation 判为主要因果原因，也不能据此进入训练。

静态与回归验证：CLI help、py_compile、定向测试通过；全量 unittest 为 `364 PASS、1
skip`；GPU smoke 和正式产物已完成，结束时 GPU/训练/评估进程为空。下一步唯一建议仍是
先扩大 matched 1.0x sham/trigger cohort 以量化 MuJoCo-Warp 自然分叉；若后续训练获准，
只设计 terrain-relative stance/contact-stability（或 local-tangent slip）单变量，
不提高 hard-case sampling，不同时改其它训练因素。

## 2026-07-23 V7 causal-coverage 扩容复核（最新）

本阶段仍为 evaluation-only，没有训练、没有修改 reward/task/robot asset，V7
`model_13600.pt` 仍是默认模型。新增专用工具：

```text
scripts/audit_go2_actuator_headroom_causal_coverage.py
tests/test_go2_actuator_headroom_causal_coverage.py
```

该工具保留旧 triplet 的 estimand（source 同 joint 连续 3 个 `.98` 饱和 row，probe
在 detect+1 切换 1.25x），只增加统一 1600-step horizon、post-300 记录、固定 checkpoint
SHA、matched identity、逐 cell coverage、source-sham noise、whole-slot bootstrap 和
外部 manifest；没有修改旧 formal evaluator。

三档独立 invocation 结果如下，不能跨档合并：

```text
repeats   slope-up vx=.3 post100   slope-up vx=.5 post100
16        5                         3
32        1                         5
64        6                         17
gate      8 required                8 required
```

64-repeat 档 runtime identity、placement、matched slot/repeat、strict finite 和
provenance 均通过，但关键 `.3` cell 仍为 `6/8` coverage；`.5` cell 为 `17/8`。
`.3` post-300 为 3 个，`.5` post-300 为 8 个。GPU/MJWarp world-count 变化明显影响
自然分叉，因此不同 repeats 档不作 pooled 统计。

64-repeat artifact：

```text
actuator_headroom_causal_coverage_clean_seed42_64repeats_1600steps_v1.json
SHA256 070567402548cf1200a6dd199be7f85d6daac0071165ce9961a1df7e5f4896ee
```

正式判断仍为 **INCONCLUSIVE**，`TRAINING_READY=false`。原因不是“方向看起来合理”，
而是 `.3` cell 未达到 8 个完整 post-100 triplet，且 legacy triplet 没有同时记录
probe/sham 的 slip、pitch、contact、gain、step 和逐事件时间戳；因此不能确认 saturation
先于失稳，也不能用它单独完成 CONTACT_CAUSAL。已有 complex v2 的 C 类摩擦证据仍保留
为方向性主因，但严格训练前还需要同步 friction source/sham/probe 或接受覆盖未达标的
明确结论。

本阶段验证：新 targeted tests 通过，全量 unittest 为 `369 PASS、1 skip`；CLI help、
py_compile、diff-check、strict JSON/recursive finite、manifest SHA 和 GPU 空闲检查通过。
本轮没有启动训练，也没有生成候选模型。下一步唯一诊断建议是新增从 rollout 起点同步
分叉的 friction `0.6/0.6/1.2` source/sham/probe，以严格验证 C 类；在该 gate 通过前不
进入训练。

## 2026-07-23 17:51：V7 多因素因果测试阶段性停机记录

用户要求下班前阶段性停止。已中断尚未完成的 action-recovery 正式 rollout，并停止一个
残留的 friction `0.8` evaluator 子进程。结束检查为 GPU `0% / 0 MiB`，无 train、
evaluate、audit、play 或 TensorBoard 进程。本阶段没有启动训练，默认模型仍为 V7
`model_13600.pt`，checkpoint SHA256 仍为
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`。

本阶段新增/更新 evaluation-only 工具：

```text
scripts/audit_go2_friction_contact_causal.py
scripts/audit_go2_foot_placement_counterfactual.py
scripts/audit_go2_action_recovery_counterfactual.py
tests/test_go2_friction_contact_causal.py
tests/test_go2_foot_placement_counterfactual.py
tests/test_go2_action_recovery_counterfactual.py
```

定向 CPU 合约最后为 `13 PASS`；三个 evaluator 均通过 py_compile、CLI help 和
diff-check。Friction evaluator 修正了 stance/contact 混入 clearance 的问题，strict gate
改用 swing-only terrain-relative clearance，并补充 warmup failure、body contact 和
failure-risk 记录。旧 v10 曾输出 provisional `CONTACT_CAUSAL=true`，但最终审计发现低速
completion/failure risk 反向恶化，因此补充安全 gate 后作废，不得引用为正式结论。

正式 artifacts：

```text
friction_contact_causal_strict_clean_seed42_32repeats_1200steps_v11.json
  full SHA256 daf16a19d8ba3e3b05568a01eaecb9f4b899400f3dcc9fba856f8ad7885c2951
  verdict INCONCLUSIVE
friction_contact_causal_strict_clean_seed42_32repeats_probe09_v1.json
  full SHA256 e6bbf80eb96e20c54889565c6b1377643edb2f37c16b176ba46ad5204e7f008a
  verdict INCONCLUSIVE
friction_contact_causal_strict_clean_seed42_32repeats_probe09_v2.json
  full SHA256 759c97783cbe268c7aa07ca14e7ed58b64af5193540460e72e135530609986c7
  verdict INCONCLUSIVE
friction_contact_causal_strict_clean_seed42_32repeats_probe08_v1.json
  full SHA256 c5382b27267be7154b583ff9c64b3e9f06e22d51b557f8ac46ab4e7cda1b4b90
  verdict INCONCLUSIVE
foot_placement_counterfactual_strict_clean_seed42_32repeats_v1.json
  full SHA256 688f5b11a49c7f65f621ff400008c1167114e9922ad75079456d896ae6f4a6f4
  verdict INCONCLUSIVE
action_recovery_counterfactual_strict_smoke_seed42_v1.json
  full SHA256 834b1ba79915fb937effde1b5f622f2ebdd4301b94c09368ca4eebdd9d798290
  smoke only; formal rollout interrupted, no formal artifact
```

主要结果：friction `1.2/0.9/0.8` 都改善 `vx=.5`，但在 `vx=.3` 重复出现
completion/failure-risk 恶化；因此 contact/traction 是强速度相关因素，但尚未覆盖两个
速度格，不能判 `CONTACT_CAUSAL`。Foot placement `+0.05 rad` 在两个速度格均强烈有害：
低速 completion delta `-16`、failure risk `3.286x`，高速 upper/calf contact 为
`4.88x/5.52x`，该方向已拒绝。Action recovery blend `0.5` 仅完成 smoke；正式 32-repeat
在用户要求停机时中断，未生成可用结果。

当前正式状态仍为：

```text
verdict = INCONCLUSIVE
training_ready = false
```

下次恢复点：先从现有 recovery smoke/source SHA 复核工作区和 GPU，再决定是否重新运行
`audit_go2_action_recovery_counterfactual.py --repeats 32`。不要复用本次被中断的 rollout，
不要把不同 invocation 合并凑样本；若 recovery 失败，再转 bounded height-scan/OOD
sentinel。训练仍被禁止，直到真实获得 `CONTACT_CAUSAL` 且 `training_ready=true`。

## 2026-07-24 10:27：V7 高坡因果覆盖正式通过

从 2026-07-23 停机点继续，固定 V7 `model_13600.pt`（SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`），全程
evaluation-only。开始状态为 branch `exp/high-slope-probe-integration`、HEAD
`0a204b645a2325cb06264725c58cc5745da64a43`，GPU/相关进程为空；保留所有既有 dirty
worktree 修改，没有 reset、checkout、clean、切分支或 commit。

三个只读子智能体复审结论：执行器 headroom 已显著降级为单独主因，但因 `.3` coverage
和同步 onset 不足不能写“已排除”；contact/gait 复审确认 friction `1.2` 的 32-repeat 主
效果很强但旧 artifact 的 action/pitch/upper-leg guardrail 未通过；lifecycle/statistics
复审指出原 action-recovery 只是两抽头 FIR、缺真实 runtime identity、actuator mediation、
flat sentinel 和正确 onset。子智能体未编辑文件、未运行 GPU。

本次更新/新增 evaluation-only 文件：

```text
scripts/audit_go2_action_recovery_counterfactual.py
tests/test_go2_action_recovery_counterfactual.py
scripts/audit_go2_contact_stabilization_causal.py
tests/test_go2_contact_stabilization_causal.py
```

Action-recovery evaluator 补齐 FIR identity、source/sham no-op、PD demand/force/utilization/
saturation、action onset、flat sentinel、top-level noise/bootstrap/schema。正式 32-repeat
artifact：

```text
action_recovery_counterfactual_strict_clean_seed42_32repeats_1200steps_v1.json
full SHA256 742e8487833ffd528a430aa6b3c9235658c662c798d802c5a2bab35b324f2351
```

结果为 `INCONCLUSIVE/training_ready=false`，且 probe 明显有害：`.3/.5` completion delta
分别 `-10/-5`，gain 约 `-12.1%/-13.5%`，pitch、腿部接触与失败风险恶化。简单 FIR
action smoothing 因而被拒绝。

新增 loaded-stance 局部切向抗滑 evaluator，固定 `20 N/(m/s)` 且裁剪到
`0.20*mu*Fn`。runtime force/torque/no-op/cap identity 与 flat sentinel 均通过。正式
32-repeat artifact：

```text
contact_stabilization_causal_strict_clean_seed42_32repeats_1200steps_v1.json
full SHA256 43a6db374320f01a39186a6b33e0fb7d19aa082b3e1e7171314e79ef6929dc2c
```

结果仍为 `INCONCLUSIVE`：低速仅小幅改善；高速虽然 slip/cone/gain 方向改善，但
completion delta `-6`、failure-risk ratio `1.24`，故该简单抗滑 wrench 被拒绝。

随后对不变的注册 friction `0.6/0.6/1.2` 做一次独立 64-repeat confirmatory
invocation；不与旧 32-repeat 合并、不改阈值、不换剂量。正式 artifact：

```text
friction_contact_causal_strict_clean_seed42_64repeats_probe12_v13.json
full SHA256 a48bb9de4a9f7e168d0e7f9b40913cca10debd4554854a391b39b3c45accc30b
canonical artifact field 96141319b56b4385ae273c1b56692e5fe01ab933b3994fb240506223a4a68eb6
evaluator SHA256 e7172b83a50c1b15b3a7b851548185f02e7ffcfd47a74a1e0ca2dc3c845b8a80
```

两个高坡 cell 各有 64 matched triplets，runtime friction identity error 为 0，placement、
lifecycle、strict finite/schema/provenance 均通过。`.3` completion source/sham/probe 为
`39/42/45`，gain `+30.6%`；`.5` 为 `11/10/49`，gain `+74.1%`。slip/cone/gain 的
方向一致率与 whole-slot bootstrap CI、effect-minus-sham-noise CI 全部通过。所有
1.2x guardrail 通过：action ratio `.1.148/.1.149`、pitch `.1.159/.1.066`、clearance
`.941/.839`、failure risk `.864/.278`，未新增系统性 body-contact 风险。

正式验收：

```text
verdict = CONTACT_CAUSAL
training_ready = true
primary_cause = foot-contact sliding-friction/traction limitation under the evaluator's MuJoCo contact model
```

下一训练阶段唯一变量：`terrain-relative local-tangent stance-slip/contact-stability shaping`。
不要同时修改 terrain、termination、command、gait、observation、network 或 PPO。该结论只
表示可以开始设计单变量训练，并不表示本阶段已训练或生成新模型。

测量限制：runtime 只读回 foot geom friction coefficient，没有读取最终 effective contact-
pair friction；时序 gate 使用注册的 friction-cone onset；结果属于 MuJoCo 仿真因果证据，
不直接证明真实 Go2 的摩擦、热、延迟、电池或 torque-speed envelope。

验证：19 项相关定向 unittest、py_compile/CLI、diff-check、recursive finite、sidecar SHA
均通过。结束时 GPU `0% / 0 MiB`，无 train/evaluate/audit/play/TensorBoard 进程。本阶段
没有启动训练，V7 `model_13600.pt` 仍为默认模型。

## 2026-07-24：V7 local-tangent stance-slip 训练前准备完成

已完成 `CONTACT_CAUSAL` 之后的第一轮单变量训练设计与实现，但未调用 PPO `learn`、未生成
新 checkpoint。新任务为 `Unitree-Go2-Rough-V7-StanceSlip`，直接继承 V7，仅新增
`terrain_tangent_stance_slip` reward；terrain、high-slope sampling、termination、command、
gait、observation、network、PPO 和既有 reward 全部冻结。

数学定义：每只脚使用 0.25 m 内最近有效 terrain ray 的局部法向；将足端世界速度投影到
切平面，只对 contact、ray valid 且局部法向载荷 `>=15 N` 的 loaded stance 计费。slip
deadband=`0.03 m/s`、scale=`0.10 m/s`、单脚 cost clip=`4`，按法向载荷归一化。唯一
训练变量为新 reward weight=`-0.05`，第一轮不做多权重 sweep。

新增：

```text
scripts/preflight_go2_stance_slip_training.py
tests/test_go2_stance_slip_reward.py
docs/reviews/stance_slip_training_design.md
```

新 reward 合约 `9 PASS`，相关定向矩阵 `25 PASS`，全量 unittest `397 PASS、1 skip`。
32-env actor-only calibration 与真实 2048-env full-resume no-learning
GPU preflight 均通过，后者严格恢复 iteration `13600` 和 terrain mean level `5.275`，动作、
总 reward、新 reward 均 finite，`learn_called=false`。正式 preflight：

```text
stance_slip_training_preflight_seed42_2048env_8steps_fullresume_v3.json
SHA256 348377defa91a940d4683d9b2188642dc423e757b6c3fae36d12f0b495e9bc9d
```

训练矩阵和完整 high-slope/flat/rough/stairs/line/arc/S gate 已预登记在 review 文档。固定
训练为 V7 `model_13600.pt` full resume、2048 env、400 iterations、seed42，仅运行一个
weight=`-0.05` candidate arm。当前状态为 **TRAINING-READY，但训练尚未启动**；V7 继续
作为默认模型。

## 2026-07-27 stance-slip 正式训练后状态

训练已完整结束，但多目标验收结论为 **REJECT / NO_SAFE_SURVIVOR**。正式 run：

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter
```

唯一变量严格是 `terrain_tangent_stance_slip` weight `-0.05`。2048 env、400 iterations、
seed42、V7 model_13600 full resume 正常完成；TensorBoard 62 tags / 24800 values 全部
finite，无 OOM、NaN、resume 或 simulator 错误。stage hashes：

```text
model_13700.pt 02f9e821739babb844598b735da5aaac4d42d8a88f203431d28689d06519f2fc
model_13800.pt 5c0b909232e2df6e5b0616731acecdee567e00c7bda4842ccbe99ae650ab04bd
model_13900.pt 4ab8740c7170b25923d4130b850fc77407f365923ad1634bb96a92ebf2eb8dea
model_13999.pt db46dcc1272cb0a722b695568c8cdf4d086af1075cd4d0b53da7a75a643563e3
```

四个 stage 均跑完 matched clean/randomized high-slope line/arc/S 16-slot screen。最佳单路线
completion 仍只有 clean `5/16`、randomized `6/16`，不满足 `12/16` 与 `10/16`；最低
forward gain 均低于 `0.40`，不满足 `0.80`。先执行 1.2x guardrail，四 stage 分别有
`9/13/7/25` 个违规组，因此无 survivor，不得从中选择 candidate。

```text
acceptance/stance_slip_checkpoint_selection.json
SHA256 52e42466b10be55e55a74df1c0d368902eba118b6fd59b2f41eb08ed8f9a88bd
selection_status = NO_SAFE_SURVIVOR
selected_checkpoint = null
```

因 checkpoint selection 已失败，retained flat/rough/obstacle、continuous terrain/stairs、
stairs seeds42/43/44 与 line/arc/S 完整套件没有进入，不得标为 PASS。默认模型继续是：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

不要采用本 run 的任何 stage，不要追加第二训练变量或补训。详细 gate/provenance 见
`docs/reviews/stance_slip_training_acceptance_20260727.md`。结束复核 GPU 空闲，无相关进程。

## 2026-07-27 stance-slip 失败机制诊断后状态

本阶段严格为 diagnosis/evaluation-only，没有启动 PPO 或修改训练配置。四个固定模型已完成
clean slope-up high/extreme、`vx=.3/.5`、4 repeats、100+2400 steps 的逐足 gait 与
pre-reset actuator 定向诊断。正式 artifact：

```text
logs/rsl_rl/go2_velocity/2026-07-27_09-58-02_go2_rough_v7_local_tangent_stance_slip_2048env_400iter/diagnostics/stance_slip_failure_mechanism_clean_seed42_r4_2400steps_v1.json
SHA256 27c9042870c065731aebf9038ea93ed55970bff97e93dbc1ef9bc4d03601655d
```

四 checkpoint 完整路径/SHA 均内嵌，artifact recursive finite，并绑定每段执行 JSON、
evaluator/dependency、branch/HEAD/dirty fingerprint。首次 monolithic GPU invocation 因
MuJoCo-Warp 重复环境构建导致 CUDA graph/allocation 错误且未写 partial JSON；随后完全相同
的配置按 checkpoint 分进程串行执行并严格 CPU 合并，四段均正常结束。

正式机制边界：

```text
REWARD_AVOIDANCE = INCONCLUSIVE
  13700 reduced-speed/short-step signal = SUGGESTIVE
  unloading avoidance = NOT SUPPORTED
OBJECTIVE_CONFLICT = INCONCLUSIVE
PHYSICAL_AUTHORITY_LIMIT = SUPPORTED for V7 under evaluator MuJoCo contact model
  candidate-specific persistence = INCONCLUSIVE
```

关键原因：13700 的某些共同 full-horizon uphill cells 同时降低 slip、gain 和步长，但
loaded fraction 总体上升且模式未跨后续 checkpoints 重复，也没有可靠的 gait/progress
onset 顺序；后续 stages 多为 slip/action/pitch 一起恶化，不能证明稳定 objective conflict。
V7 friction v13 仍通过 contact causal gates；candidate 没有同场景 friction triplet。

下一步是 **DO_NOT_TRAIN**。不要 weight sweep、补训 13999、增加第二 reward，或把 friction
`1.2` 写进训练/部署。只有需要 candidate-specific 物理归因时，才考虑对四个固定 checkpoint
执行注册 `0.6/0.6/1.2` source/sham/probe 的 evaluation-only 对照。默认模型不变：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

完整解释见 `docs/reviews/stance_slip_failure_mechanism_diagnosis.md`。

## 2026-07-27 proprioceptive student 当前交接状态

当前可启动新的 proprioceptive sim2real 训练 arm，但本轮尚未启动。冻结任务：

```text
Unitree-Go2-Rough-Sim2Real-Proprio-V1-Distill
Unitree-Go2-Rough-Sim2Real-Proprio-V1
```

Actor/critic/teacher/action shape 为 `425/261/234/12`。Actor 只使用 SDK 可构造的 10 帧
本体感知历史、当前 command/phase 和 previous action；没有 height scan/contact/base linear
velocity。History 为 term-major、oldest-to-newest，canonical schema SHA256：

```text
c0b80faccc897cb77290ed55832cefd6ed97432831898129598df996486f87ae
```

先做 V7 deterministic teacher-rollout BC 300 iter（仅 terrain levels 0-6），再 actor-only
handoff 到 fresh critic/optimizer 的纯 PPO 4000 iter。唯一允许的正式命令：

```bash
conda activate unitree_rl_mjlab
cd /home/jensen/projects/unitree_rl_mjlab
python scripts/train_go2_proprioceptive.py
```

Orchestrator 会校验 V7 teacher SHA、拒绝同名重复 run，并精确解析新 distillation run 的
`model_299.pt`；provenance/schema 不一致时不会进入 PPO。不要手工拼接 load-run，不要使用
被拒绝 stance-slip checkpoints。

2048-env no-learning preflight artifact SHA256 为
`1fe5b626e68816f6408b97efc9739cbd187de0dff0004ce03b543cbc2503c489`；定向测试
15 PASS，全量 418 PASS/1 skip，ONNX parity、C++ history、官方 SDK2 的
`unitree_mujoco`/`go2_ctrl` build 和 headless DDS bridge smoke 均 PASS。

```text
PROPRIOCEPTIVE_SIM2REAL_TRAINING_READY=true
HARDWARE_READY=false
```

G0-G7 无硬件部分均 PASS。真实 Go2 型号/固件、SDK 实测时延抖动、编码器/IMU 校准、
热/电池/力矩和急停仍为 `HARDWARE_PENDING`，不阻止仿真训练但阻止实机部署。默认模型仍为
V7 `model_13600.pt`，SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`。

先阅读：

```text
docs/reviews/go2_proprioceptive_student_readiness.md
docs/reviews/go2_proprioceptive_student_training_contract.md
docs/reviews/go2_proprioceptive_student_readiness.json
```

## 2026-07-27 proprioceptive formal launch revalidation

旧 schema/readiness 已被以下当前证据取代：

```text
student schema SHA256 379d982c61c839286fe7a566fee40160f599831752041250dbd804709a6e4b10
2048-env preflight SHA256 fff95311e3e0335ab244594c98891e7cc1d0a1569ebb3758857554bc251560a0
source manifest precomputed SHA256 18ed1acd1de0b6932c6e04ece4244e37a7117179c68fa8ccd1fa30281ddca24a
tests 17 targeted + 9 provenance/retry + 429 full PASS, 2 full skips
```

Orchestrator 现在使用全周期 `fcntl` lock、内嵌 untracked source 的 canonical manifest、
阶段边界 source revalidation，以及 checkpoint/ONNX manifest SHA。C++ 部署接口增加逐关节
processed-target 限位、first-valid-action latch、动作读写同步、atomic policy stop、quaternion
normalization 和严格 ONNX `[1,425] -> [1,12]` shape/count。未训练 candidate ONNX 的接口
preflight 已 PASS；实际训练 checkpoint 的 ONNX/controller closed loop 仍是训练后 gate，不能
据此声称 hardware ready。

首次 formal launch 因 RSL-RL distillation logger 在零 learning 前读取未初始化 student
distribution 而失败；失败目录和 traceback 保留并以固定 SHA marker 注册。兼容修复不改变
实验合同，32-env one-iteration discard smoke PASS。下一次唯一命令会忽略且只忽略这个
`TECHNICAL_FAILURE_PRE_LEARNING` run，其他重复 run 继续 fail closed。

## 2026-07-27 active proprioceptive formal run

正式技术重试已经启动，禁止重复运行。Stage 1 已完整结束：

```text
distillation run
logs/rsl_rl/go2_velocity/2026-07-27_16-27-06_go2_sim2real_proprio_v1_v7_teacher_distill_2048env_300iter

handoff checkpoint
model_299.pt
SHA256 f63b6ec244833572e2017243b8b032b9f23040a20b92030d987502b06f839565
```

teacher bitwise unchanged，student finite，stage/teacher/schema/source-manifest provenance 与
run 内 manifest 副本均通过。Orchestrator 已自动启动 Stage 2：

```text
logs/rsl_rl/go2_velocity/2026-07-27_16-40-03_go2_sim2real_proprio_v1_ppo_2048env_4000iter
```

这是 2048-env、seed42、fresh critic/optimizer/iteration-0 的唯一 pure PPO。首个正式
`model_250.pt` 已通过结构/finite/provenance/ONNX shape 审计，SHA256
`1dd699c2ba6c7e394cb1251fce4a6d6c55db6a4f3bf58d25cbb3a934bce3db8c`。训练仍 active；
不得并行运行 GPU evaluator，不得修改 formal source manifest 内文件，不得按 reward 或 final
checkpoint 提前选模型。

部署接口新增同步 measured-position hold 和 joint-map fail-closed 检查，CPU/C++ tests 与
go2_ctrl rebuild PASS，controller SHA256
`e61ea9c68361ddb52be7adbf1bf9f997c5a717ad138abc8b05928de7264ed2f5`。训练后仍必须补
policy inference stale watchdog、selected PyTorch/ONNX/Python/C++ parity、trained-policy
DDS/MuJoCo closed loop 和 Passive fault injection。当前仍为 `HARDWARE_PENDING`。

## 2026-07-29 proprioceptive training final handoff

正式训练已完整结束。Stage 1 的 `model_299.pt` SHA256 为
`f63b6ec244833572e2017243b8b032b9f23040a20b92030d987502b06f839565`。
原 PPO 进程在 iteration 1642 后被外部中断；使用只改变持久化 cursor
`1500 -> 1501` 的精确 resume anchor 后，续跑完整覆盖 `1501..3999`。最终 checkpoint：

```text
logs/rsl_rl/go2_velocity/2026-07-29_10-10-14_go2_sim2real_proprio_v1_ppo_2048env_4000iter_exact_resume_1501_3999/model_3999.pt
SHA256 d48d08188c0823e42610a9ffd5de4cead2093af2cc9171517bb7099c40bb4760
exact-resume start 2026-07-29 10:10:14 CST
final checkpoint written 2026-07-29 11:58:53 CST
```

正式 17-checkpoint lineage：

```text
logs/rsl_rl/go2_velocity/proprioceptive_acceptance_20260729/checkpoint_lineage.json
SHA256 166727662f222c6036b66859e343cb957a2068352df209d009b925e960f38ffa
```

所有 checkpoint 的 tensor、provenance/lifecycle、425-D schema、静态 ONNX 和
PyTorch/ONNX parity 都通过。但 acceptance screener 曾把非零 action fault count 与
`action_limits_valid=true` 同时写入，属于 fail-open。现已修复为 count 驱动布尔值，并在
bundle/selector 增加一致性 fail-closed 检查。

使用完全相同的 16 个固定 screening 输入，在新目录非覆盖重跑：17/17 checkpoint 均有
action-limit fault，单 checkpoint `4..14/16`，合计 `127/272`。因此所有候选均违反预登记
screening 硬门槛，正式选择结果：

```text
logs/rsl_rl/go2_velocity/proprioceptive_acceptance_20260729/selection_screening_hard_gate.json
SHA256 b61bda30158c64ba2a88a4409e7cc2e715b78b5be7e6e4c008f4244697588359
selection_status NO_SAFE_SURVIVOR
selected_checkpoint null
```

252 项 GPU rollout 计划在 V7 完成 10 份 raw JSON 后停止。原因不是 OOM、NaN 或模型加载
失败，而是硬门槛优先：所有 student 已在 checkpoint screening 淘汰，后续 242 项不可能
改变 survivor 集合。已完成 raw 与日志全部保留；不得把未跑的 completion、slip、contact、
stand/basic tracking 或部署闭环标成 PASS。没有 survivor，所以没有复制 ONNX、没有生成
部署候选包，也没有执行 survivor-only controller/DDS/MuJoCo gate。

最终唯一项目结论：

```text
TRAINING_REJECTED
```

训练过程本身完整、finite；被拒绝的是 17 个训练产物的安全验收。默认模型继续为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

不要从该 PPO run 推广任何 checkpoint，不追加第二变量或补训。`HARDWARE_READY=false`，
`HARDWARE_PENDING`。最终 GPU 为 `0 MiB / 0%`，无 train/evaluate/TensorBoard/
unitree_mujoco 进程。

未来若开启新的训练 arm，在进入 GPU rollout 前还必须修复未使用的后级 selector：精确校验
scene/profile/route inventory，冻结 stand/basic tracking 数值门槛，并明确 body contact 只作
1.2x guardrail 还是也进入词典序。当前 `0` 个 screening survivor，后级 selector 未执行，
这些缺口不改变本轮 `TRAINING_REJECTED`。

最终验证为 targeted `14/14`、full unittest `444 OK, 1 skipped`、py_compile、selection
recursive-finite/dependency SHA、`git diff --check` 全部通过。分支仍为
`exp/high-slope-probe-integration`，HEAD
`0a204b645a2325cb06264725c58cc5745da64a43`；所有既有 dirty changes 保留，未 commit。

## 2026-07-29 model_3999 真实 rollout 动作安全诊断

已完成 evaluation-only matched 检测，回答 synthetic screening 是否误拒绝模型。结论是：

```text
ACTUAL_ROLLOUT_ACTION_FAULT = CONFIRMED
SYNTHETIC_ONLY_EXPLANATION = REJECTED
```

候选在 flat clean/randomized 共 108 个 line/arc/S 场景中 action fault 为零；问题集中于高难
楼梯和高坡。continuous stairs 为 clean `6`、randomized `23` fault control steps；high
slope 为 clean `933`、randomized `973`，最大动作分别 `8.5521/10.7027`。同时存在
`abs(action)>4` 和动作小于 4 但 processed joint target 越界两种 fault。

候选相对 V7 明显改善高坡 completion 和 action safety，但仍违反绝对零 fault gate，因此
`TRAINING_REJECTED` 不变、`model_3999.pt` 不推广、默认 V7 不替换。V7 也存在真实复杂地形
action fault，只是仿真默认而非 hardware-ready 安全基线。下一训练 arm 应预登记并只改变
一个训练/ONNX/C++ 一致的 bounded asymmetric per-joint action mapping；禁止只在部署端静默
clip。完整报告：

```text
docs/reviews/go2_action_rollout_safety_diagnostic_20260729.md
```

## 2026-07-30 safe-action V2 training start

Safe-action V2 implementation and formal preflight are complete. V2 uses a
squashed Gaussian with the canonical per-joint asymmetric mapping, while the V7
teacher remains unchanged and is mapped once during distillation. Focused tests
are `65/65 PASS`; 2048-env no-learning preflight, 32-env distillation/PPO
optimizer smoke, checkpoint/ONNX export and TensorBoard action-chain telemetry
all pass with zero action or joint-target fault rows.

Frozen source manifest:

```text
logs/rsl_rl/go2_velocity/provenance/go2_safe_action_v2_source_manifest_40cfd6951f4c187f050d02f24f03188af0ff3744a68fe727cefe0b6874fe43e4.json
SHA256 40cfd6951f4c187f050d02f24f03188af0ff3744a68fe727cefe0b6874fe43e4
```

Formal orchestrator command is `python scripts/train_go2_proprioceptive_safe_action.py`.
It exclusively runs Stage 1 at 2048 env/300 iterations and then Stage 2 fresh
PPO at 2048 env/4000 iterations, both seed 42. Do not start a duplicate process,
modify a manifest-bound source, or run a GPU evaluator concurrently. V7
`model_13600.pt` remains the default until complete post-training acceptance.

Stage 1 completed successfully with `model_299.pt` SHA256
`92d5d07266cdcec395eeab1634cbeb374e363f04aa0c6fe45d12f84ec4df9021`.
The first Stage 2 process was externally interrupted after finite iteration 386;
its last checkpoint is `model_250.pt` SHA256
`70a0467d3bca26fa5b2f9be8c079561d3956148ff24fe7cd8a3f196f033e7c4a`.
Exact recovery is active in
`2026-07-30_11-21-28_go2_sim2real_proprio_v2_safe_action_ppo_2048env_3749iter_exact_resume_251_3999`,
using an anchor identical except for cursor `250 -> 251` (SHA256
`cec64e1f4af2c4b7414ba0c408bfd8237881dc4764087b2563cff3a6416bff31`).
The recovery log starts at iteration 251. Do not start another training process.
## Active replacement safe-action V2 training (2026-07-30)

Reject the earlier V2 `model_250.pt` split line. Synchronous CUDA diagnosis
proved PPO numerical divergence: applied-space rollout telemetry stayed finite
and inside all bounds, but the surrogate loss spiked to `27992.5636` and the
actor latent mean became NaN in a minibatch update. The bounded policy now uses
`mean=5*tanh(raw_mean/5)` before sampling, with the same final asymmetric
applied-action interface and no other training-variable change.

Replacement fixed identities:

```text
schema SHA256 2c6da479c6c833f127c672b774e877e0fc6b14967e39b06ff2333dd241b87c3f
source manifest SHA256 aa2c147e24fa1d692f0c2775233660580a8eb7997b73013b99e93030d6c1279e
preflight SHA256 2f06204beef9f69c8d3df17d7a583cd681a2f6f300dcbe25de4b69804b2b258f
```

The formal orchestrator PID is `65535`; its current distillation child is
`65622`, run
`2026-07-30_12-18-36_go2_sim2real_proprio_v2_safe_action_meanbound5_v7_teacher_distill_2048env_300iter`.
Do not launch another training or GPU evaluator while it is active. The
orchestrator must validate the 300-iteration output and then start fresh PPO.

## Safe-action V2 replacement: post-training acceptance paused (2026-07-30)

Replacement training is complete; the old PID note above is historical. The
formal PPO run is:

```text
logs/rsl_rl/go2_velocity/2026-07-30_12-32-20_go2_sim2real_proprio_v2_safe_action_meanbound5_ppo_2048env_4000iter
```

`model_3999.pt` SHA256 is
`1d4e8502ed13b40197bc5e6c00a95aa8a7cf8f955d05d280eea11ea6324cd28b`.
The complete 17-checkpoint schedule, all finite TensorBoard scalars, 4000/4000
finite action telemetry, and zero action/target fault rows are verified.

CPU/ONNX screening retained 15 checkpoints and rejected iterations 1000 and
2250 only for exceeding the frozen `1e-5` parity threshold. The formal lineage
and screening decision are:

```text
docs/reviews/go2_v2_meanbound5_formal_checkpoint_lineage.json
SHA256 cb1300431299041bcfcbc6aed23be596f337c5a7573740448afccc0aa1ac8421
docs/reviews/go2_v2_meanbound5_screening_decision.json
SHA256 409e33636768ed51377f5d47ea5d774c78f347fd54a7ea7e83ed9e39bd618679
```

The serial acceptance matrix is intentionally paused. There are 57 complete
artifacts under `docs/reviews/go2_v2_meanbound5_formal_acceptance_raw`; the next
unfinished item is `model_750/flat_matched_routes.json`, which is absent (no
partial JSON). GPU and evaluator processes are idle. Resume without repeating
completed work using:

```bash
/home/jensen/anaconda3/envs/unitree_rl_mjlab/bin/python \
  scripts/evaluate_go2_proprioceptive_acceptance.py \
  --checkpoint-manifest docs/reviews/go2_v2_meanbound5_formal_checkpoint_lineage.json \
  --output-dir docs/reviews/go2_v2_meanbound5_formal_acceptance_raw \
  --plan-file docs/reviews/go2_v2_meanbound5_formal_acceptance_resume1_plan.json \
  --screening-dir docs/reviews/go2_v2_meanbound5_checkpoint_screening \
  --bundle-dir docs/reviews/go2_v2_meanbound5_formal_acceptance_bundles \
  --selection-file docs/reviews/go2_v2_meanbound5_formal_selection.json \
  --execute --resume-existing
```

Do not delete the 57 raw artifacts, reuse the original plan filename, or start
training. V7 remains the default until selection, complete deployment parity,
and bundle checks finish.

## Safe-action V2 完整验收结论（2026-08-03）

正式串行矩阵已完成 `252/252` 个 GPU invocation，所有 raw JSON 均可解析且
finite。GPU 阶段没有 NaN/Inf、OOM、模拟器异常或重复正式任务。随后 CPU 汇总发现
`model_250.pt`、`model_500.pt` 在大量场景中启动即灾难性失败，未产生任何 15 N
loaded-stance 足样本；其 terrain-tangent slip 应为“不可用”，不能伪装为 0。汇总器已
按 fail-closed 修复为记录 availability、禁止 partial mean，并通过
`unified_metrics_valid=false` 硬拒绝这两个 checkpoint。没有修改任何 raw rollout，也
没有重跑 GPU。

最终证据：

```text
252/252 raw JSON
17 acceptance bundles
docs/reviews/go2_v2_meanbound5_formal_acceptance_resume4_plan.json
SHA256 6f3556dfc464704a44654e21e28187167d746ce840dcd239b566aab5ea07e7e2
docs/reviews/go2_v2_meanbound5_formal_acceptance_resume4_plan.inventory.json
SHA256 e01a2c5e4ac4f78449aa1164ba9de4a62242442a20e7556149ef2a6b60c68d3e
docs/reviews/go2_v2_meanbound5_formal_selection.json
SHA256 4861770ccdcf0ff52eff7a79b58bb015dcd681b9ff009025efb14734a47174e1
selection_status NO_SAFE_SURVIVOR
survivor_count 0
selected_checkpoint null
```

17/17 checkpoint 均至少违反一个 completion gate，并且均违反 matched V7 的
1.2x safety guardrail；17/17 failure risk 失败。另有 calf contact `16/17`、pitch
`15/17`、mechanical power `14/17`、energy/slip/upper-leg contact `13/17` 等
回归。10/17 未通过 mean forward-gain；`model_1000.pt`、`model_2250.pt` 仍因
ONNX parity > `1e-5` 被拒；`model_250.pt`、`model_500.pt` 还因 loaded-stance
slip 不可用被硬拒。bounded action 与 joint-target finite/fault-free 门槛本身通过。

因此最终结论是：

```text
TRAINING_REJECTED
DEPLOYMENT_BUNDLE_READY=false
HARDWARE_READY=false
HARDWARE_PENDING
```

没有 survivor，故不得挑选“最少违规”的 checkpoint，也不执行 survivor-only
Python/ONNX/C++ parity 或部署包推广。默认模型继续为：

```text
logs/rsl_rl/go2_velocity/2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter/model_13600.pt
SHA256 73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff
```

验证：定向 acceptance 单测 `19/19 PASS`；acceptance/screening/safe-action/
provenance/distribution 联合 unittest、compileall、`git diff --check` 全部退出 0。
最终 GPU 为 `0% / 12 MiB`，无 train/evaluate/TensorBoard/play/audit/
unitree_mujoco 进程。保留全部既有 dirty changes，未 reset、checkout、clean、切分支、
commit 或替换默认模型。
## 2026-08-04 V8 teacher probe handoff

The 234-D control and 237-D `base_lin_vel` candidate both finished 400 updates.
All 9 high-slope screening evaluations finished; the selector result is
`NO_CAUSAL_SURVIVOR`, so the privileged teacher probe is `REJECTED`. Full
42-artifact acceptance was deliberately not launched. The default remains the
V7 `model_13600.pt` checkpoint with SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`.

The new columns learned nonzero weights (first-layer norm 1.3138 -> 3.5861
over updates 100 -> 400), so the failure is not a transfer or unused-feature
issue. Review `docs/reviews/go2_privileged_teacher_high_slope_selection.json`
and `docs/reviews/go2_privileged_linvel_teacher_acceptance.json` before
preregistering any independent 241-D `foot_contact(4)` experiment. Student
distillation remains blocked.

## 2026-08-09 foot-contact teacher 输入决策

本轮三个子智能体只读复核后，下一 actor 输入假设选为 `foot_contact(4)`，但尚未启动评估或
训练。必须纠正旧 241-D 计划：干净的 V7 单变量 candidate 是 `234+4=238-D`；241-D 会叠加
已经失败的 `base_lin_vel(3)`，不得使用 V8 237-D checkpoint 或 optimizer 初始化。新增
actor slice 为 `[234:238]`，normalizer source 是 V7 critic `[245:249]`。运行时传感器自然
槽序经 artifact 复核为 `FL/FR/RL/RR`；`FR/FL/RR/RL` 是 evaluator 重排后的 site canonical
顺序，对应 permutation `[1,0,3,2]`。为避免新增第二种干预，本探针直接复用 critic term 并
冻结自然槽序，actor `[234:238]` 必须等于 critic `[245:249]`。

contact/traction 是既有严格因果证据支持的失败机制；这不等于瞬时二值 contact 输入已证明
具有 actor 增量价值。正式 PPO 前先执行 evaluation-only 可观测性诊断；诊断不支持、足序/
时序/coverage/provenance 不明时必须 `INCONCLUSIVE_DO_NOT_TRAIN`。诊断通过后也只允许准备
matched 234/238 schema、32-env optimizer smoke 和 2048-env no-learning preflight；正式
400-update 训练必须单独确认。

执行指令与依据：

```text
docs/reviews/go2_foot_contact_teacher_probe_multi_agent_instruction.md
SHA256 0451e1625a7970bb5987f7578252fe37220dfb1dbd9e244835164433240cbc61
docs/reviews/go2_foot_contact_teacher_input_diagnosis_20260809.md
SHA256 222f001729e8c1db891b50cf715184a38092d2af88883503a57324ae019519b8
```

当前 `FORMAL_TRAINING_STARTED=false`、V7 仍是默认模型、student/running/hardware 均未推进。
`2026-08-09 19:38 CST` GPU 为 `0% / 0 MiB`，没有项目训练或评估任务。

## 2026-08-09 foot-contact observability handoff

正式 evaluation-only 诊断已完成：18/18 chunk、288 trajectories、319162 observed raw
rows，训练状态未改变且没有调用 `learn()`。最终 reducer 返回：

```text
INCONCLUSIVE_DO_NOT_TRAIN
analysis_status coverage_inconclusive
technical_failures []
```

直接阻塞项是原始瞬时二值 `found>0` contact 的 chatter：四个 clean/randomized ×
`vx=.3/.5` 分层最大值为 `0.1997/0.2752/0.2023/0.3175`，均超过预注册 `0.10`。
同时 H10 catastrophic-failure positive anchors 为 `54/101/52/92`，低于 `200`；部分
H25 低速层也不足。由于 coverage fail-closed，正式 analyzer 没有执行模型增益比较，不能
把本结论解释成 contact 已无效，也不能放宽门槛后补做选择。

独立审计确认实现计数无误，但 H10 的统一 `positive_anchors >= 200` 门槛设计为不可达：
每层 `72` 条 trajectory、stride `5`、H10 每个终止至多贡献 `2` 个 anchor，理论最大值仅
`144`。未来版本必须在采集前按 terminal events/clusters 做可达性与功效审计；不得用这个
发现修改当前正式结果。raw-contact chatter 独立失败，因此仍然明确禁止开始训练。

禁止直接启动 238-D Teacher PPO。下一步若继续 contact 路线，必须先把“输入定义”作为新的
单变量假设预注册，例如固定的 2-tick debounced contact 或带迟滞的 loaded-contact；不得把
它与原始 contact arm 混为同一实验，也不得用本次 raw 事后调整阈值后宣称通过。另一个选择是
补充独立 failure-event coverage，但它不能解决已经确认的 chatter gate。

证据：

```text
docs/reviews/go2_foot_contact_observability_raw_20260809/manifest.json
SHA256 bbcdb4eb754e54d26186762e1a0353e542ac09890d98df5ce61a425349327c6b
docs/reviews/go2_foot_contact_observability_summary_20260809.json
SHA256 8232f7944eaf1a7d9721d6154e364699483ba72956d26400a73c5f0910db724b
docs/reviews/go2_foot_contact_observability_summary_20260809.md
SHA256 eacbaffdcd0b9d5d467bd9177740942c2bcc1f70bd48fbd99d21fac8fbf22425
```

当前仍使用 V7 `model_13600.pt`（SHA256
`73f68beb29ed23f561fd3364e476e32167269c8a9f88078a7344db4d504f2dff`）；没有 238-D task、
新 checkpoint、student 蒸馏或 running teacher。诊断单测 `20/20 PASS`、py_compile 与
`git diff --check` 均通过。

## 2026-08-10 mandatory Reference Gate and rejected causal-contact probe

The user rejected the causal two-tick confirmed-contact RL Teacher hypothesis
because no directly matching paper or auditable GitHub implementation was
provided before the formal diagnostic was launched. Related evidence is not
equivalent to a direct reference.

The `..._v2_restart1` collection was stopped after 6/24 complete chunks; the
seventh chunk was interrupted, no manifest exists, and no formal analyzer or
PPO training was run. The directory is permanently excluded and contains
`ABORTED_USER_REJECTED_UNREFERENCED_HYPOTHESIS.md`.

Effective immediately, every innovative observation/preprocessing rule,
reward, architecture, estimator, loss, curriculum, Teacher/Student interface,
training mechanism, or evaluation mechanism must pass `REFERENCE_GATE` before
any implementation, smoke, GPU diagnostic, or training:

1. Cite at least one directly relevant primary paper or public auditable GitHub
   implementation.
2. Map the exact referenced component to the proposed implementation and state
   every material difference. For GitHub, record URL, commit/tag, and license.
3. Obtain explicit user confirmation of the reference and deviations.
4. Project logs, adjacent literature, or agent reasoning alone are insufficient.
5. If no direct reference exists, return `NO_DIRECT_REFERENCE_DO_NOT_IMPLEMENT`.

The rejected V2 artifacts remain audit-only. Do not resume, analyze, promote,
or use them to authorize a 238-D Teacher. V7 remains the default; no Student or
running Teacher work was started.

## 2026-08-10 contact-force Teacher partial screening handoff

The user approved a new reference-backed experiment using privileged four-foot
3-D contact forces from Lee et al. (Science Robotics 2020) and a
function-preserving V7 input expansion. Both formal arms completed 400 PPO
updates from the locked V7 checkpoint:

```text
control_234:  2026-08-10_01-05-52_go2_contact_force_teacher_v1_control_234_2048env_400iter
candidate_246: 2026-08-10_01-24-25_go2_contact_force_teacher_v1_candidate_246_2048env_400iter
```

The candidate appends only Actor `[234:246] = foot_contact_forces(12)` and maps
its normalizer from Critic `[249:261]`; all other contracts are frozen. Paired
2048-env preflight passed with exact initial action parity (`max_abs=0.0`), and
all eight checkpoints passed CPU/TensorBoard screening. Candidate new-column
weight norm grows `1.1241 -> 2.3210`, so the input is being used.

Only update 100 high-slope evaluation has completed. It is rejected:

```text
                         completion  forward_gain  progress  slip
locked V7                 0.218750     0.429108     0.509031  0.058547
control_234@100           0.208333     0.349154     0.503169  0.056164
candidate_246@100         0.156250     0.325402     0.489578  0.054139
```

The candidate reduces slip slightly but loses completion, forward gain and
progress versus both V7 and control. This rejects update 100 only; updates
200/300/400 are unevaluated. The user requested a pause after the first pair.
No evaluator is running and GPU is idle. Resume tomorrow with:

```bash
/home/jensen/anaconda3/envs/unitree_rl_mjlab/bin/python \
  scripts/run_go2_contact_force_teacher_high_slope.py \
  --output-dir docs/reviews/go2_contact_force_teacher_high_slope \
  --resume-existing
```

This skips the two completed JSONs and starts at `control_234_model_200`. Do
not select or replace V7 until all remaining stages and hard-gate selection are
complete. Compact status is in
`docs/reviews/go2_contact_force_teacher_update100_screening_summary.json`.
