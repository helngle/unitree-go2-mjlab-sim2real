# V7 曲线命令响应诊断

本阶段只解析已有有效 JSON，不修改 command sampler、训练配置、reward、terrain、
termination、gait 或网络，也没有启动训练。默认 checkpoint 为 V7
`model_13600.pt`。

## 结论

**NO-GO：当前证据不授权增加 curve sampling 或启动 curve probe。**

有效 command-tape 显示的主要问题是通用 forward under-gain，而不是额外的
forward+yaw 耦合退化：14 个 V7 general-yaw 分布内场景的平均 `vx` response gain
为 `0.8108`，`wz` response gain 为 `0.9525`。同一 V7 既有 clean gait 诊断中的
纯 forward `0.6 m/s` aggregate gain 为 `0.8184`；只取 flat 为 `0.8540`，而 tape 中
同速且 ID 的 coupled gain 为 `0.8490`，差约 `-0.6%`。这不支持“yaw 耦合导致
forward tracking 明显额外退化”。

此外，补足 horizon 后 closed-loop 圆弧为 `18/18` 完成、零 reset。现有证据更符合：
policy 的 yaw 执行正常，forward 本来就欠跟踪，closed-loop 通过延长执行时间补偿。

## 数据契约

只使用以下有效结果：

```text
V7/route_baseline_curved_arc_command_tape_clean_seed42_18env_1200steps.json
V7/route_baseline_curved_arc_closed_loop_clean_seed42_18env_1200steps.json
V7/route_baseline_curved_arc_closed_loop_r4v03_clean_seed42_2env_1600steps.json
V7.1-run/lateral_gait_diagnostics_clean_seed42_144env_900steps.json
V7.1-run/robustness_key_models_relocated_randomized_seed42_1120env_1000steps.json
```

其中 `V7` 指
`2026-07-14_11-29-13_go2_rough_v7_explicit_modes_focus_probe_2048env_500iter`，
`V7.1-run` 指
`2026-07-14_13-45-50_go2_rough_v7_1_lateral45_staged_probe_2048env_500iter`；后两个
JSON 中只选择 checkpoint 名为 `model_13600.pt` 的 result。

旧文件 `route_baseline_curved_arc_command_tape_clean_seed42_72.json` 没有
`motion_steps`、`settle_steps` 和固定 tape lifecycle 字段，且会按实际 progress 延长
命令，继续判定为无效。离线诊断脚本会拒绝该 schema。

response 定义为：

```text
forward gain = mean actual body vx / mean commanded body vx
yaw gain     = abs(mean actual wz / mean commanded wz)
ID           = abs(required wz) <= 0.3 rad/s
OOD          = abs(required wz) > 0.3 rad/s
```

## Command-tape 结果

| subset | n | vx gain | wz gain | progress | cross-axis | slip | action acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ID | 14 | 0.8108 | 0.9525 | 0.8558 | 0.0387 | 0.0274 | 0.0731 |
| OOD | 4 | 0.8290 | 0.9450 | 0.8884 | 0.0497 | 0.0358 | 0.0886 |
| all | 18 | 0.8148 | 0.9508 | 0.8630 | 0.0411 | 0.0293 | 0.0766 |

三组均为零 reset、零 fell/base/upper-leg/calf contact。OOD 是
`r=1.5 m, v={0.5,0.6} m/s` 的左右转，共 4 个；它们没有比 ID 表现更差，但不能用
来声明 V7 的 general-mode 分布内能力。

按速度汇总：

| speed | vx gain | wz gain | progress |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.7728 | 0.9567 | 0.8213 |
| 0.5 | 0.8261 | 0.9502 | 0.8740 |
| 0.6 | 0.8456 | 0.9456 | 0.8937 |

progress 比 gain 略高是合理的：几何投影进度还受实际 yaw、横向运动和 settle 后 pose
影响，不能把 progress ratio 直接当作 body-frame `vx` gain。

## Pure-axis 对照的边界

可用 pure-forward 证据：

```text
clean gait diagnostic, forward_0.6:
  all selected terrain/levels gain = 0.8184
  flat-only gain                   = 0.8540

randomized relocated robustness:
  forward_0.3 gain = 0.7039
  forward_0.6 gain = 0.8112
  forward_0.9 gain = 0.8333
```

可用 pure-yaw JSON 只保存 yaw absolute error：`yaw_left=0.0836 rad/s`、
`yaw_right=0.0847 rad/s`，command 为 `+-0.5 rad/s`。它没有保存 actual signed yaw
mean，因此无法从 mean absolute error 唯一恢复 response gain。诊断工具将 yaw gain
输出为 `null`，不会用 `1-error/command` 猜测。

严格 matched 的 pure-forward / pure-yaw / coupled 矩阵仍缺失：历史文件没有在完全
相同的 flat patch、speed/yaw、profile、warmup 和 sample horizon 上覆盖所有
`v={0.3,0.5,0.6}` 与对应 yaw。因此当前判断是强证据下的 NO-GO，不是声称已完成
严格因果实验。正式训练前若仍怀疑耦合短板，应先补一个短诊断矩阵，而不是先训练。

## Command 带宽审查

V7 训练 sampler 每 `3--8 s` 重采样一次 piecewise-constant command；actor observation
每个 `0.02 s` control step 直接读取 command manager 的三维 body command。曲线 evaluator
将自动重采样设为 `1e9 s`，然后每个 control step 直接写 `vel_command_b`：

- command-tape 在每个 arc segment 内仍是常量，只在 tape 结束时归零；它主要测试稳态
  twist 执行，不是高频 controller。
- closed-loop 每 `0.02 s` 可更新一次 command。其圆弧已经通过，说明当前平滑反馈在
  clean flat 上可执行，但不能外推到快速反向 S 弯或高频噪声命令。
- policy 没有显式 command rate limiter，训练也没有专门覆盖高频 command change。
  下一步 S 弯验收应记录 command delta/saturation，并单独测试 step、ramp 与左右 yaw
  切换；不要在同一 probe 中同时修改 sampler 比例和 resampling cadence。

历史 JSON 只有 attempt aggregate，没有逐 control-step command/response 序列。因此
rise time、overshoot、settling time 和 slew-rate 分布当前都不可获得，不能从 mean gain
反推。若要补齐，应让 evaluator 在线累计或可选保存短时序，再运行小规模诊断；本阶段
没有为此启动 GPU rollout。

## 离线复现

新增 `scripts/diagnose_go2_command_response.py`，它只使用 Python 标准库。典型命令：

```bash
python scripts/diagnose_go2_command_response.py \
  --arc-tape-json <valid-tape.json> \
  --closed-loop-json <closed-loop-1200.json> \
  --closed-loop-json <r4v03-retry-1600.json> \
  --reference-json <clean-gait-diagnostic.json> \
  --reference-json <randomized-robustness.json> \
  --output /tmp/go2_curve_command_diagnostics.json
```

脚本独立验证 ID/OOD 标签，汇总 `vx/wz gain`、progress、cross-axis、slip、action
acceleration、reset/contact，并让后提交的 closed-loop retry 覆盖同一 attempt 的旧结果。

## 决策

当前训练 gate 为 **NO-GO**：

1. ID arc yaw gain `0.9525`，没有 yaw tracking 崩溃。
2. coupled `v=0.6` ID forward gain `0.8490` 与 pure flat forward `0.8540` 基本一致。
3. closed-loop 补足 horizon 后 `18/18` 完成。
4. rise/overshoot 和严格 matched pure-axis 对照尚未采集。

因此不实现 curve sampler、不启动 PPO。先完成 S 弯/randomized baseline，或用短时序
matched command-response 诊断证明 coupled response 确有额外退化；只有后者成立时，
才重新评估 15% correlated curve-command 单变量 probe。
