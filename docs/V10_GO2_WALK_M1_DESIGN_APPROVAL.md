# V10-Walk Teacher V1：M1 设计与用户审批合同

日期：`2026-08-10`

状态：`M1_DESIGN_READY_FOR_USER_APPROVAL`

实现状态：`NOT_IMPLEMENTED`

训练状态：`NOT_STARTED`

对应总方案：[`V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md`](V10_GO2_COMPLEX_LOCOMOTION_PROJECT_PLAN.md)

对应实时日志：[`V10_GO2_PROJECT_JOURNAL.md`](V10_GO2_PROJECT_JOURNAL.md)

## 1. 本轮范围

本轮只建立并训练一个新的复杂地形行走 Teacher：

```text
task: Unitree-Go2-Rough-V10-Walk-MikiTeacher-V1
initialization: random actor + random critic + fresh optimizer
source checkpoint: none
action: 12-D normalized joint-position target
control frequency: 50 Hz
sim timestep: 0.005 s
decimation: 4
Student: not trained in this round
Running: not trained in this round
Sim2Real: not executed in this round
```

目标是先得到一个从随机初始化训练、能稳定处理复杂地形的行走基础Teacher。V7只作为同场景评估基线，不参与初始化。

## 2. 多智能体结论

三个并行只读子任务分别审计了参考文献、当前MJLab实现和训练资源。交叉结论：

1. Miki 2022直接支持复杂地形Teacher的本体历史、四足多尺度地形输入、privileged encoder和per-foot terrain encoder；
2. Miki原版391-D输入与16-D CPG+IK动作成套，不能保留391-D CPG输入却改成12-D直出；
3. Walk These Ways直接支持50 Hz、12-D关节位置目标、0.25 rad action scale、四足clock和当前PPO参数；
4. 推荐首版是明确标注偏差的382-D混合合同，不声称复现任何一篇论文；
5. 8 GB显存下模型参数不是主要风险，MuJoCo-Warp terrain/contact/raycast workspace才是主要不确定项；
6. 复杂地形奔跑、统一walk/run Student和Sim2Real不在本轮范围。

## 3. 直接参考登记

访问日期均为`2026-08-10`。

### 3.1 Miki复杂地形Teacher

- 论文：https://arxiv.org/abs/2201.08117
- 完整公开训练仓库：未发现；以原论文和supplementary作为直接依据；
- 直接组件：50 Hz proprio history、每足52个多尺度高度样本、50-D privileged state、per-foot terrain encoder、privileged encoder；
- 不直接授权：Go2资产、MJLab/MuJoCo、12-D无CPG动作、本文足序/frame/miss处理和V10验收门。

### 3.2 Walk These Ways动作、clock和PPO

- 论文：https://arxiv.org/abs/2212.03238
- GitHub：https://github.com/Improbable-AI/walk-these-ways
- commit：`0e7236bdc81ce855cbe3d70345a7899452bdeb1c`
- tag：无；
- license：MIT，依赖和第三方资产分别遵循各自许可证；
- 直接组件：50 Hz、12-D action、default joint position加scaled offset、主scale 0.25、四足clock和PPO基线；
- 不直接授权：Go1 hip×0.5、actuator-net、6-step lag、±5 m/s或complex-terrain running。

### 3.3 Rudin/legged_gym地形课程

- 论文：https://arxiv.org/abs/2109.11978
- GitHub：https://github.com/leggedrobotics/legged_gym
- commit：`8fa29acc6fd1910c3d9659eef6310bdd301cde0a`
- tag：无；
- license：BSD-3-Clause，assets/dependencies另有许可证；
- 直接组件：大量并行PPO和按每回合行走距离升降terrain level的game-inspired curriculum；
- 本项目现有`terrain_levels_vel`与公开实现的move-up/move-down结构一致。

### 3.4 Lee辅助参考

- 论文：https://arxiv.org/abs/2010.11251
- GitHub：https://github.com/leggedrobotics/learning_quadrupedal_locomotion_over_challenging_terrain_supplementary
- commit：`277d19b007fd3109956d1ea0cc9e0cd50d3ecb5b`
- tag：无；
- license：MIT；`rsc/robot/`另受ANYbotics许可证约束；
- 直接组件：privileged Teacher、terrain/contact information和adaptive terrain difficulty；
- 边界：仓库不含完整训练流水线，CPG/IK动作也不是V10的12-D直出。

## 4. 推荐382-D Teacher输入

```text
proprio/history       124
per-foot terrain      208
privileged             50
total                 382
```

全部输入来自同一pre-action控制时刻的当前状态或严格过去历史，不使用未来帧。

### 4.1 Proprio/history：124-D

| 顺序 | term | 维数 | frame/时序 |
|---:|---|---:|---|
| 1 | velocity command | 3 | body-yaw frame，当前`vx/vy/yaw` |
| 2 | projected gravity | 3 | base local，当前 |
| 3 | base linear velocity | 3 | IMU site local，当前仿真真值 |
| 4 | base angular velocity | 3 | IMU site local，当前 |
| 5 | current joint position relative | 12 | rad，当前 |
| 6 | joint-position history | 36 | `t-3,t-2,t-1`，oldest→newest |
| 7 | current joint velocity | 12 | rad/s，当前 |
| 8 | joint-velocity history | 24 | `t-2,t-1`，oldest→newest |
| 9 | physical joint-target history | 24 | `t-2,t-1`，最终target rad |
| 10 | four-foot trot clock | 4 | FL/FR/RL/RR，当前sin clock |

```text
3+3+3+3+12+36+12+24+24+4 = 124
```

历史reset使用ObservationManager现有首帧backfill语义。physical target不能用normalized `last_action`冒充，必须由default position、scale和真实已执行action构造。

四足clock参考Walk These Ways，但V10-Walk固定为trot：

```text
foot order: FL, FR, RL, RR
phase offsets: 0.0, 0.5, 0.5, 0.0
period: 0.6 s
frequency: 1.6666667 Hz
```

Miki的13-D CPG information完整删除，不放置常量或伪造CPG状态。

### 4.2 Per-foot terrain：208-D

每足52个向下ray，参考Miki五个同心环：

| ring | radius (m) | points |
|---:|---:|---:|
| 1 | 0.08 | 6 |
| 2 | 0.16 | 8 |
| 3 | 0.26 | 10 |
| 4 | 0.36 | 12 |
| 5 | 0.48 | 16 |

```text
52 per foot * 4 feet = 208
```

V10工程偏差：

- foot-major顺序FL、FR、RL、RR；
- ring从小到大，每环从局部+x开始逆时针；
- ray保持world-horizontal分布并向world-z负方向发射；
- feature为`terrain_hit_z - foot_z`，单位m；
- miss使用`-max_distance`并单独计数；
- 不复用V7的base-centered 17×11 yaw scan。

论文没有公开离散环起始角、足序、miss策略和精确frame，以上均属于需批准偏差。

### 4.3 Privileged：50-D

| term | 维数 | V10语义 |
|---|---:|---|
| foot contact state | 4 | FL/FR/RL/RR，当前binary |
| foot contact force | 12 | 每足world xyz raw net force，不做`sign*log1p` |
| contact terrain normal | 12 | 每足最强接触world xyz，无接触置0 |
| foot friction | 4 | 每足foot geom friction参数 |
| thigh/shank contact | 8 | 每足thigh 1 bit + calf/shank OR 1 bit |
| external base wrench | 6 | world force3+torque3；无wrench事件时为0 |
| foot air time | 4 | 当前秒数 |

V1不新增无直接范围依据的external-wrench事件，因此该6-D可能恒零，不允许临时自造扰动范围。

base linear velocity对Teacher可用，但当前Go2部署schema没有等价实机源。它不会直接进入未来Student，Student需通过有参考的历史estimator估计或由真实状态估计器提供。

## 5. Actor与Critic网络

### 5.1 Actor

```text
shared per-foot terrain: 52 -> 80 -> 60 -> 24
four foot latents: 4*24 = 96
privileged: 50 -> 64 -> 32 -> 24
fusion input: 124+96+24 = 244
fusion: 244 -> 256 -> 160 -> 128 -> action mean12
activation: LeakyReLU
distribution: Gaussian, learned std, init std=1.0
```

论文写每足分别进入encoder但未明确是否共享权重。V10选择shared foot encoder以保持腿间对称并减少参数，登记为`inferred_shared_weights`。

### 5.2 Critic

Critic使用同一382-D raw information和相同分支形状，但拥有完全独立参数，输出1-D value。Actor和critic不共享encoder、normalizer或parameter object。

Miki未公开critic结构；该选择是V10偏差。独立actor/critic由当前RSL-RL和Walk These Ways实现支持。

### 5.3 资源

shared encoder下，Actor+同构Critic参数少于当前V7双MLP。1024 env、24 steps、Actor/Critic各存382-D raw observation的FP32 rollout约`71.6 MiB`。主要显存风险来自raycast、MuJoCo-Warp和backward，必须实测。

## 6. 12-D动作合同

```text
order:
FL_hip, FL_thigh, FL_calf,
FR_hip, FR_thigh, FR_calf,
RL_hip, RL_thigh, RL_calf,
RR_hip, RR_thigh, RR_calf

q_target = q_default + 0.25 * normalized_action
unit: rad
control period: 0.02 s
```

Go2当前PD：

```text
hip:   Kp=20, Kd=1, effort=23.5 Nm
thigh: Kp=20, Kd=1, effort=23.5 Nm
calf:  Kp=40, Kd=2, effort=45 Nm
```

不复制Go1 hip×0.5，不增加bounded distribution，不提高stock effort limit。

## 7. Reward与termination

首版避免同时发明route reward、安全约束PPO或复杂聚合，复用已在Go2 V7中运行过的线性reward骨架：

| reward | weight |
|---|---:|
| track linear velocity | `+1.0` |
| track angular velocity | `+1.0` |
| variable posture | `+1.0` |
| body orientation L2 | `-1.2` |
| body angular velocity | `-0.05` |
| angular momentum | `-0.025` |
| joint acceleration L2 | `-2.5e-7` |
| joint position limits | `-10.0` |
| action rate L2 | `-0.07` |
| foot gait/contact timing | `+0.5` |
| terrain-relative foot clearance | `-1.2`，target `0.12 m` |
| foot slip | `-0.25` |
| soft landing | `-1e-3` |
| stand still | `-1.0` |
| base contact | `-3.0` |
| upper-leg contact | `-2.0` |
| calf contact | `-0.75` |
| non-timeout termination | `-200.0` |

这不是Miki reward复现，而是项目已有Go2 reward基线。参考论文为velocity tracking、contact timing、clearance、collision、joint smoothness和slip组件提供依据；全部数值偏差在此显式登记。

暂不加入route-progress、constrained PPO、completion bonus、stance-slip新reward、torque-speed reward、WTW指数聚合、Miki未知`c0`的reward curriculum或中途权重变化。

termination复用Go2 V7已登记的base、upper-leg、orientation和calf定义，不复制Miki未公开阈值。

## 8. Terrain、command与randomization

### 8.1 Terrain

复用V6 terrain primitive和比例：

```text
flat 15%; stairs up 15%; stairs down 15%;
slope up 10%; slope down 10%; random rough 15%; obstacles 20%
max_init_terrain_level = 2
```

只使用legged_gym/Rudin式distance curriculum：超过terrain长度一半level+1；不足`0.5*commanded planar speed*episode length`时level-1；到最高level后回到随机较低level。不实现自定义W0-W3配额sampler。

### 8.2 Walk command

```text
vx: [0.0, 1.2] m/s
vy: [-0.3, 0.3] m/s
yaw rate: [-0.7, 0.7] rad/s
standing environments: 5%
resampling: [3, 8] s
```

不使用V7 focused high-speed mode，不加入1.5 m/s以上奔跑命令。

### 8.3 Domain randomization

```text
foot friction: [0.3, 1.6]
encoder bias: [-0.015, 0.015] rad
base CoM offset: each axis [-0.05, 0.05] m
base payload: add [-1.0, 3.0] kg
motor strength: scale [0.9, 1.1], all 12 joints
push interval: [10, 15] s
push root velocity x/y: [-0.5, 0.5] m/s
```

必须用runtime coverage测试证明motor-strength覆盖12个关节。本轮不新增延迟、actuator-net或external-wrench事件。

## 9. PPO与训练预算

PPO直接复用WTW和当前项目一致配置：

```text
num_steps_per_env=24; value_loss_coef=1.0; clipped value loss=true;
clip=0.2; entropy=0.01; epochs=5; mini_batches=4;
learning_rate=1e-3 adaptive; gamma=0.99; lambda=0.95;
desired_kl=0.01; max_grad_norm=1.0
```

正式训练：

```text
num_envs: 1024
training seeds: 42, 43, 44
updates per seed: 14000
transitions per seed: 344,064,000
save interval: 250 completed updates
GPU concurrency: exactly 1
resume at seed start: false
source checkpoint: none
```

预算：每seed约9-12 GPU小时；训练约27-36 GPU小时；含筛选验收约40-60 GPU小时；磁盘预留至少100 GiB。这是估算，不是性能承诺。

## 10. 实现与M2预检

批准后只新增隔离文件：

```text
src/tasks/velocity/v10_walk_miki_schema.py
src/tasks/velocity/mdp/v10_walk_miki_sensors.py
src/tasks/velocity/mdp/v10_walk_miki_observations.py
src/tasks/velocity/rl/v10_walk_miki_teacher_model.py
src/tasks/velocity/rl/v10_walk_miki_runner.py
src/tasks/velocity/config/go2/v10_walk_env_cfg.py
src/tasks/velocity/config/go2/v10_walk_rl_cfg.py
src/tasks/velocity/evaluation/v10_walk_acceptance.py
scripts/preflight_go2_v10_walk.py
scripts/train_go2_v10_walk.py
scripts/evaluate_go2_v10_walk_acceptance.py
scripts/select_go2_v10_walk_checkpoints.py
tests/test_go2_v10_walk_*.py
```

仅允许对`config/go2/__init__.py`做最小task注册追加。

预检：

1. CPU schema/history/ring/observation/model/action/curriculum/provenance tests；
2. 32-env exactly 1 update optimizer smoke；
3. 1024-env 8 steps no-learning，NVML peak `<=5.5 GiB`；
4. 1024-env exactly 3 updates disposable capacity smoke：NVML peak `<=6.75 GiB`、余量`>=1.25 GiB`、peak增长`<=128 MiB`；
5. smoke checkpoint禁止resume/selection；
6. GPU退出后回到0 MiB；
7. 全部通过才生成正式命令。

env数变更必须重新审批，不能在OOM后临时缩放继续。

## 11. Screening与验收

复用现有同场景评估器和V10总方案能力/安全门，不发明新训练reward或selector：

```text
clean per-route completion >= 12/16
randomized per-route completion >= 10/16
mean forward gain >= 0.80
retained completion >= matched V7 - 0.05
high-slope completion >= matched V7
safety metric <= matched V7 * 1.2
V7 zero fault remains exact zero
```

覆盖flat、ordinary rough、continuous、obstacles、stairs、line、arc、S-curve和high-slope clean/randomized，以及slip、pitch、action acceleration、effort/saturation、body contact和failure。

若没有checkpoint通过：

```text
NO_V10_WALK_TEACHER_SURVIVOR
```

不得追加输入、临时改reward或从失败checkpoint补训。

## 12. 本文件不授权的内容

```text
Walk Teacher继续训练成复杂地形Run Teacher
retained replay自定义hard curriculum
1.5-2.5 m/s复杂地形奔跑课程
Miki terrain belief与RMA dynamics latent组合Student
统一walk/run final Student
Student蒸馏后的RL fine-tuning
新route-progress/constrained-PPO/acceptance算法
```

这些阶段需要以后另找直接参考；找不到时保持`NO_DIRECT_REFERENCE_DO_NOT_IMPLEMENT`。

## 13. 请求批准的15项差异

批准本文件即代表接受：

1. 随机初始化，不使用V7 warm start；
2. Miki 391-D/16-D CPG改为382-D/12-D Go2 joint-target；
3. 删除CPG13，改用WTW式4足sine clock和固定0.6 s trot；
4. body orientation使用projected gravity；
5. 使用本文history lag、reset backfill、足序、ray顺序、frame、height符号和miss语义；
6. foot encoder使用推断的共享权重；
7. privileged force/normal/friction/contact采用本文MuJoCo语义；
8. external wrench6保留但本轮可能恒零；
9. Actor和critic同构但参数完全独立；
10. 复用Go2 V7 reward骨架，不复制Miki/WTW完整reward；
11. 只使用Rudin/legged_gym distance terrain curriculum；
12. 使用固定walk command范围和本文domain randomization；
13. 1024 env、14000 updates、三个seed和约40-60 GPU小时总预算；
14. M2加入1024-env planned backward capacity smoke；
15. 本轮只训练V10-Walk Teacher，不实施Run、Student或Sim2Real。

批准前状态：

```text
M1_DESIGN_READY_FOR_USER_APPROVAL
M2_NOT_STARTED
M3_FORMAL_TRAINING_NOT_STARTED
GPU_IDLE
```
