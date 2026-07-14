# 新聊天窗口交接摘要

当前项目：`/home/jensen/projects/unitree_rl_mjlab`

目标：沿 Unitree 官方 `unitree_rl_mjlab` 路线，在 MuJoCo/MJLab 里训练和回放 Go2。现在 `Unitree-Go2-Flat` 已经训练通并有稳定 baseline；`Unitree-Go2-Rough` 已做多段训练，但暴露出 curriculum 问题，下一步应先调整课程/预训练路线，而不是继续原配置硬训。

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
