# LIBERO 256MiB arena：四套任务保存动作重放

**2026-10-16：四套任务共978个保存动作已全部重放，记录中的机器人状态、指定图像和终止标志与原轨迹完全一致，MuJoCo警告为0。** 这验证了256MiB arena方案在这四条已知轨迹上的一致性；没有调用策略模型，不增加机器人基准回合数，也不能据此宣布VLA-JEPA闭环完成。

arena容量从原空间任务的1.235GiB改为256MiB，但本次测量**没有显示对应的物理RSS下降**。空间任务reset后，新方案RSS约2.307GiB，原诊断约2.304GiB；预留容量与实际驻留内存必须分开解释。

## 为什么检查arena

此前[分页VLA-JEPA闭环](VLAJEPA_PAGED_RESULTS.md)已加载模型权重，却在首个环境初始化/reset阶段触发系统可用内存下限。为分离模型与环境成本，先做不加载策略、只初始化空间任务的诊断，再检验是否可以减小模拟器的arena分配容量。

[首次初始化诊断](../artifacts/frontier/libero_init_profile_20261016.json)因Windows原子日志替换遭拒而中止，状态为`error`、异常为`PermissionError`；它不是环境完整初始化成功或内存保护退出。保留该记录后进行[独立重试](../artifacts/frontier/libero_init_profile_20261016_retry.json)，成功完成一次官方reset，没有模型调用或策略动作。重试观察保存为[NPZ证据](../artifacts/frontier/libero_init_profile_20261016_retry.npz)。

原诊断记录`data_narena=1,326,448,640`字节（1.235GiB），而已观察到的arena最高使用量为9,453,556字节。MuJoCo 3.3.2中，`size.memory`指定arena的动态数组分配容量，`njmax`是影响容量推导的旧属性；`nconmax`仍限制生成的接触数量。新方案显式指定`memory=268435456`，去掉与之冲突的`njmax`，保留`nconmax=5000`。见[官方3.3.2 XML说明](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#size)。

arena不足可能让接触被丢弃、约束求解器停用或触发错误，所以只确认“能运行”不够。本次同时检查逐步数值、图像和MuJoCo警告，并记录`maxuse_arena`；这些量的含义见[官方3.3.2内存分配说明](https://mujoco.readthedocs.io/en/3.3.2/modeling.html#memory-allocation)。

## 固定范围与实际结果

[执行前计划](../artifacts/frontier/libero_arena_replay_20261016.plan.json)保存了代码与原始轨迹身份，SHA-256为`23392c8714cd5dbf2d266d96060326e9757d040b50bb0455515ebb74ab710cf5`。输入来自已封存SmolVLA前瞻确认中，四套任务各task1、init41、seed1000、full10的完整动作序列。保持360×360相机、hard reset、官方10步settle及原动作处理；以零容差比较float32机器人状态、uint8图像和终止标志。

| Suite | 保存动作数 | 最大机器人状态差 | 初始双相机/最终主相机 | 逐步终止标志 | MuJoCo警告 | 最大arena使用量 |
|---|---:|---:|---|---|---:|---:|
| libero_spatial | 93 | 0 | 逐像素相同 | 相同 | 0 | 9,453,556字节 |
| libero_object | 280 | 0 | 逐像素相同 | 相同 | 0 | 4,828,844字节 |
| libero_goal | 85 | 0 | 逐像素相同 | 相同 | 0 | 5,030,332字节 |
| libero_10 | 520 | 0 | 逐像素相同 | 相同 | 0 | 8,646,500字节 |

[总报告](../artifacts/frontier/libero_arena_replay_20261016.json)为`passed_exact_replay`、计划4、完成4、失败列表为空，`new_policy_inference_calls=0`。总共978个动作，加上每条轨迹的初始状态，共982条状态/容量检查位置；比较的图像是初始双相机和最终主相机，不是全程每帧像素。四条轨迹的最高arena使用量均低于256MiB，但这不是所有任务或未来动作的容量上界。

每个suite保存`attempt.json`、`arena_audit.json`、`capacity.jsonl`、`executed_steps.jsonl`、`trajectory.npz`、三张图像及逐文件哈希的`evidence.json`。审计确认只改编译输入的`size`标签内存相关属性、标签外字节不变、接触上限保留；原源码和资产文件未被就地改写。可逐项查看[spatial证据](../artifacts/frontier/libero_arena_replay_20261016/libero_spatial/evidence.json)、[object证据](../artifacts/frontier/libero_arena_replay_20261016/libero_object/evidence.json)、[goal证据](../artifacts/frontier/libero_arena_replay_20261016/libero_goal/evidence.json)和[long-horizon证据](../artifacts/frontier/libero_arena_replay_20261016/libero_10/evidence.json)。实现入口为[重放探针](../scripts/probe_libero_arena.py)与[arena预算适配器](../scripts/libero_arena_budget.py)。

## 容量改变与实际内存测量

| 指标 | 原初始化诊断 | 256MiB重放诊断 |
|---|---|---|
| 空间任务arena容量 | 1,326,448,640字节，1.235GiB | 268,435,456字节，0.250GiB |
| 空间任务reset后RSS | 官方reset返回时2.303GiB；保存证据/GC后约**2.304GiB** | 第0步容量记录2,477,010,944字节，**2.307GiB** |
| 全诊断每秒采样最高RSS | 单次空间任务初始化：2.407GiB | 串行四suite重放：3,280,461,824字节，**3.055GiB** |
| 模型加载与推理 | 无 | 无 |

空间任务的[容量逐步记录](../artifacts/frontier/libero_arena_replay_20261016/libero_spatial/capacity.jsonl)和原初始化重试是在不同时间运行，后者还包含诊断日志与观察保存。它们不是严格配对的内存消融；四suite峰值与单suite峰值的测量范围也不同。即使arena容量减少约0.985GiB，当前reset后RSS仍接近原值，不能写成“已节省约1GiB物理内存”。全进程还包括模型之外的运行库、网格纹理、渲染与分配器等开销；每秒采样不能保证捕获所有瞬时峰值。

## 对后续闭环意味着什么

初始化诊断的两份原始脚本快照也保留：[首次脚本](../artifacts/frontier/libero_init_profile_20261016.source.py)、[文件占用重试脚本](../artifacts/frontier/libero_init_profile_20261016_retry.source.py)。它们用于审计当时执行内容；主复现入口仍是上面的正式重放脚本。发布清单分别记录原始诊断字节与隐私清理后的导出字节，不把经过清理的副本冒充原始文件。

本次完成的是四条已有轨迹的环境一致性验证，且0个新模型调用、0个新增评测回合。后续真实策略执行见[compact结果](VLAJEPA_COMPACT_RESULTS.md)、[独立进程结果](VLAJEPA_ISOLATED_RESULTS.md)及[单长任务资源实验](VLAJEPA_LONG_HORIZON_RESULTS.md)。这些回合有各自的新模型调用与封存记录；本页重放不增加它们的分母，也不证明arena带来物理RSS节省。

四条已知轨迹逐值一致并不证明所有状态、接触构型或未来策略动作下物理行为均相同；后续运行仍需保留警告、容量和系统内存保护。这个诊断的可展示贡献是可复核的改动范围、明确的等价性检查及未观察到RSS改善的真实结果，不能替代完整LIBERO基准、泛化或稳定内存收益的证据。
