# VLA-JEPA 分页存储：单观测一致性与闭环内存中止

**2026-10-16：本页原分页阶段通过一次真实模型前向，但该阶段闭环未完成。** 单个保存观察上的模型动作和环境动作均与已有动态INT8输出逐值相等。随后独立闭环成功加载权重，但第一个环境初始化/reset阶段系统可用内存降到0.447GiB，保护退出。没有执行预测动作，没有封存回合，也没有可报告的任务成功率。

后续[compact](VLAJEPA_COMPACT_RESULTS.md)和[独立进程](VLAJEPA_ISOLATED_RESULTS.md)各完成三个任务；长任务在披露的[资源干预](VLAJEPA_RESOURCE_RELEASE_RESULTS.md)下完成[中断恢复](VLAJEPA_RECOVERY_RESULTS.md)，执行至520步时限但官方任务失败。恢复不增加独立评测样本，也不补足本页原分页阶段。

这是CPU低内存部署的可行性研究。已有SmolVLA的28回合缓存实验和16回合前瞻确认已经完成；本页只记录VLA-JEPA分页变体，不把它的探针或失败启动加入那些分母，也不将其称为完整LIBERO基准或SOTA复现。

## 变更与证据范围

原动态INT8变体将共享BF16 token embedding完整放入owned内存。新变体保留`151936 × 2048`完整词表与BF16数值，用`mmap.ACCESS_COPY`私有文件映射承载，lookup输出仍转FP32；300个Qwen Linear继续动态INT8量化，其他推理部分及完整四步动作头仍为FP32。共享Parameter引用、源文件身份、有限值与偏移均在真实加载时检查。

622,329,856字节（0.580GiB）是embedding payload及理论可避免的owned分配，不是保证的RSS节省。映射页可进入进程工作集和系统缓存；模型参数量没有因为分页而减少。实现与独立预算见[执行前协议](VLAJEPA_PAGED_PROTOCOL.md)，原5GiB owned阶段见[历史结果](VLAJEPA_CLOSED_LOOP_RESULTS.md)。分页阶段采用独立3.5GiB启动预算，并保留自身RSS上限5.5GiB、系统可用内存下限0.5GiB和每秒采样；3.5GiB是探索预算，不是官方最低硬件要求。

## 单个保存观察的真实前向

原始证据为[分页探针JSON](../artifacts/frontier/vlajepa_paged_probe_20261016.json)和[保存动作](../artifacts/frontier/vlajepa_paged_probe_20261016.actions.npz)。对照为此前[owned INT8探针](../artifacts/frontier/vlajepa_int8_probe_20261015.json)，不同时加载两个模型。

| 检查或测量 | 实际结果 |
|---|---|
| 真实前向 | 1次，`inference_executed=true`；输出`(1,7,7)` |
| 量化模块与动作头 | 300个量化Linear实际执行；原四步flow，action-head NFE为4 |
| 共享embedding | BF16、完整词表、共享Parameter身份检查通过；运行时映射数1 |
| 模型动作一致性 | `exact_equal=true`，不同元素0，最大绝对差0 |
| 环境动作一致性 | `exact_equal=true`，不同元素0，最大绝对差0 |
| 加载和量化 | 64.543秒 |
| 单次前向 | 3.912秒 |
| 每秒采样最高RSS | 3,144,630,272字节，**2.929GiB** |
| 每秒采样最低系统可用内存 | 2,508,206,080字节，2.336GiB |

原owned探针的采样最高RSS为3.521GiB、单次前向为6.954秒。两次运行发生在共享电脑的不同时间，没有交错重复、等同主机负载或配对计时，不能把差额解释为已验证的内存因果收益或加速比。每秒采样也可能漏掉瞬时峰值。这里能确定的是新存储路径实际执行成功，并在这一个已知观察上输出完全相同的动作；它不证明所有输入等价、原生BF16等价或机器人任务质量不变。

## 分页闭环的实际中止

独立阶段目录为`outputs/frontier/vlajepa_paged_closed_loop`，协议指纹为`c35bc7f4f94964ecd19313cbe0fb923ad4daf924692153b050189288e719fe3d`。四个计划回合依次为spatial/object/goal/libero_10的task1、init41、seed1000，动作头四步、每次执行七个动作再重新观察；完整定义见[冻结协议](../outputs/frontier/vlajepa_paged_closed_loop/protocol.json)。

| 阶段 | 原始证据与测量 |
|---|---|
| 入口与加载 | [run_0001](../outputs/frontier/vlajepa_paged_closed_loop/runs/run_0001.json)：入口可用5.610GiB；物化前4.997GiB；`weights_loaded=true`；加载和量化78.882秒，加载后RSS2.904GiB |
| 首回合开始准备 | [attempt记录](../outputs/frontier/vlajepa_paged_closed_loop/episodes/vlajepa_paged__libero_spatial__1__41__clean__native4_chunk7/attempt.json)建立；首个环境初始化/reset阶段尚未产生模型规划或动作记录 |
| 内存保护 | [独立停止记录](../outputs/frontier/vlajepa_paged_closed_loop/runs/run_0001.memory_stop.json)：系统可用480,092,160字节（**0.447GiB**），低于0.5GiB；当时自身RSS5,105,790,976字节（4.755GiB），未达到5.5GiB自身上限；只退出本实验自身进程 |
| 完成状态 | [部分报告](../outputs/frontier/vlajepa_paged_closed_loop/report.json)：计划4，完成0，官方成功0，完整运行任务失败0；spatial为`unsealed`，其余3项为`missing` |
| 动作证据 | 首回合目录只有`attempt.json`，**0个预测动作执行**，无轨迹、结果或封存凭据；环境准备不能当作完成回合 |

`run_0001.json`保留了中止前的`running_episodes`快照；独立`memory_stop`记录才给出中止状态，不能据前者推断进程还在运行。停止记录本身没有`weights_loaded`字段，部分报告对该条目显示的默认`false`不能推翻主运行记录中的`true`。报告中缺失的采样峰值也不是0；当前可核实的是停止瞬间的RSS和系统可用内存。

本次触发的是**整个系统可用内存不足**，不是模型任务失败，也不是策略给出了错误动作。单次探针通过并不保证模型和渲染模拟器同时驻留时内存足够；共享主机的其他分配同样会影响余量，当前证据不能把全部变化归因于某一个组件。一次未封存尝试加三项未开始，不能写成0/4成功率、四次任务失败，或删除未封存记录后重算分母。原记录完整保留，不在同一冻结目录自动覆盖重试。

## 后续独立实验及结论边界

后续[256MiB arena重放](LIBERO_ARENA_REPLAY_RESULTS.md)完成了978个保存动作的一致性验证，但没有观察到对应的物理RSS下降。[compact闭环](VLAJEPA_COMPACT_RESULTS.md)与[独立进程闭环](VLAJEPA_ISOLATED_RESULTS.md)各完成三个真实任务，第四项均因5.5GiB进程RSS保护中止；[单长程任务6GiB实验](VLAJEPA_LONG_HORIZON_RESULTS.md)另行记录。新的预算、模型调用和任务证据各自冻结，不能补足本页未封存回合；重放本身也不增加模型评测分母。

当前可用于研究或求职展示的结论是：完成了有来源校验的分页加载实现、真实单观测动作一致性检查，并通过实际中止定位了“单模型前向可运行、与环境共同驻留仍受资源限制”的工程边界。尚不能据此宣称VLA-JEPA完整基准复现、稳定加速、量化质量无损、显著性能改进或真机实时能力。原owned阶段四次入口阻断、新分页阶段一次运行中止及后续紧凑内存区实验分别保存，不能互相替代未完成回合。
