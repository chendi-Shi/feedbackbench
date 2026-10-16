# VLA-JEPA 本机四任务闭环：分页模型与256MiB arena

**2026-10-16实际终态：前三个任务已真实闭环成功并封存；第四个环境初始化时触发进程RSS保护，四回合阶段未完成。** [部分报告](../outputs/frontier/vlajepa_compact_closed_loop/report.json)记录计划4、完成3、官方成功3、完整运行任务失败0、未封存1。前三回合共327个动作、48次基于当前观察的模型规划；第四项没有执行预测动作。不能把这写成4/4完成，也不能删去未封存项后报告整组100%成功率。

这是VLA：策略接收当前双相机、任务文字及机器人自身状态，输出机械臂/夹爪动作，执行后再观察和预测。此前模型单观测探针、环境保存动作重放与本次真实模型闭环是三种不同证据；本次已封存回合包含新的模型调用与实际执行动作。

## 实验身份与前置验证

模型是固定官方VLA-JEPA-LIBERO检查点的CPU数值变体：300个Qwen Linear动态INT8量化，完整BF16 token embedding采用私有文件映射，其他推理部分和动作头为FP32。保留原四步flow、每块七个动作，执行完队列后读取最新观察；没有训练模型或更新权重。环境使用256MiB arena，保留原接触上限、相机、物理和控制配置。

[执行前书面协议](VLAJEPA_COMPACT_PROTOCOL.md)与[冻结文件](../outputs/frontier/vlajepa_compact_closed_loop/protocol.json)绑定20个源文件和实际模型、运行库、资产及初态身份。协议指纹为`4cf9f71bb5ededd906d48f16995792a631eb6e01381e25d75d5eafc275f5064e`。范围固定为spatial/object/goal/libero_10各task1、init41、seed1000，控制步上限280/280/300/520，CPU两个计算线程、一个inter-op线程；在同一台16GB、无NVIDIA GPU电脑上运行，零云费用。

两个执行前门槛均有独立证据：

- [分页模型单观测验证](VLAJEPA_PAGED_RESULTS.md)：同一保存观察的模型动作与环境动作均与原owned INT8输出逐值相等，最大差0；这只支持该观察，不证明原生BF16等价。
- [256MiB arena重放](LIBERO_ARENA_REPLAY_RESULTS.md)：四套任务978个保存动作全部执行，记录的机器人状态、初始双相机/最终主相机像素及终止标志逐值一致，MuJoCo警告0；没有模型推理，不增加基准回合数。

## 当前封存结果

实验完成和任务成功分别计数。一个回合完整到达成功条件或规定时限、证据封存并通过核验，才计为完成；达到时限但未成功属于任务失败。入口阻断、运行错误或未封存不能当作任务失败。本次三个完成回合均达到官方成功条件，第四项仅有尝试记录；阶段整体保持`complete=false`。

| Suite | 当前证据状态 | 官方成功 | 执行动作/上限 | 规划次数 | 动作头NFE | 推理总秒数 | 回合墙钟秒数 |
|---|---|---|---:|---:|---:|---:|---:|
| libero_spatial | 完成并封存 | 是 | 109/280 | 16 | 64 | 71.322 | 107.516 |
| libero_object | 完成并封存 | 是 | 128/280 | 19 | 76 | 90.538 | 120.912 |
| libero_goal | 完成并封存 | 是 | 90/300 | 13 | 52 | 54.749 | 78.534 |
| libero_10 | 内存保护中止，未封存 | 未评定 | 0/520 | 0 | 0 | — | — |

空间任务的原始指令是“pick up the black bowl next to the ramekin and place it on the plate”，即把小烤盅旁边的黑碗拿起并放到盘子上。[结果JSON](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_spatial__1__41__clean__native4_chunk7/result.json)记录`success=true`、`terminated=true`、`truncated=false`，在109步达到官方成功条件；[录像](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_spatial__1__41__clean__native4_chunk7/rollout.gif)用于检查过程，成功判定取自环境记录。[封存凭据](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_spatial__1__41__clean__native4_chunk7/receipt.json)绑定轨迹、逐步动作、调用、图像、录像和容量审计。

对象任务的原始指令是“pick up the cream cheese and place it in the basket”，即拿起奶油奶酪并放入篮子；[结果JSON](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_object__1__41__clean__native4_chunk7/result.json)与[录像](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_object__1__41__clean__native4_chunk7/rollout.gif)对应128步成功。目标任务的原始指令是“put the bowl on the stove”，即把碗放到炉灶上；[结果JSON](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_goal__1__41__clean__native4_chunk7/result.json)与[录像](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_goal__1__41__clean__native4_chunk7/rollout.gif)对应90步成功。两项同样为`terminated=true`、`truncated=false`，各自保留封存凭据。

前三回合48次规划均实际执行四次动作头求值，总NFE为192。逐调用`forward_seconds`的中位数/95%分位分别为spatial **4.159/5.458秒**、object **4.827/5.685秒**、goal **3.972/4.953秒**。95%分位描述实际调用的耗时分布，不是置信区间。三个完成回合总推理时间216.608秒，墙钟分别计入reset、控制、物理和相机渲染，不包括模型加载与GIF编码。仿真在推理时暂停，20Hz是模拟控制频率，不是CPU能够20Hz推理的证据。

## 加载、内存与容量审计

[本次主运行记录](../outputs/frontier/vlajepa_compact_closed_loop/runs/run_0001.json)显示权重已加载，导入28.055秒，加载和量化85.411秒，加载后RSS为3,116,953,600字节（2.903GiB）。这只是模型加载后的快照，不是模型与环境共同运行的峰值。

在11:30:37（北京时间）初始化第四个环境时，[独立停止记录](../outputs/frontier/vlajepa_compact_closed_loop/runs/run_0001.memory_stop.json)显示自身RSS为**5,962,772,480字节（5.553GiB）**，超过5.5GiB上限；系统可用内存为**761,851,904字节（0.710GiB）**，仍高于0.5GiB下限。因此本次是进程RSS越界，与此前分页阶段的系统可用内存不足是不同触发条件。该记录是停止瞬间测量，不是完整运行最大RSS或最低系统可用内存的汇总。退出码72，保护只退出自身进程。

主运行JSON停留在中止前的`running_episodes`，应结合独立停止记录阅读，不能解释为进程仍运行。停止记录不含`weights_loaded`字段，部分报告对它展示的默认`false`也不代表模型未加载。第四回合目录只有[attempt.json](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_10__1__41__clean__native4_chunk7/attempt.json)，没有模型调用、动作轨迹或封存凭据；初始化中止不评为任务失败。

三个完成回合的arena记录分别为110、129、91条，均包含reset与每个控制步，容量固定268,435,456字节；最高用量分别为9,453,556、4,828,844、5,030,332字节，全部八项警告计数为0，原生warning列表也为空。空间回合的[逐步记录](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_spatial__1__41__clean__native4_chunk7/arena_steps.jsonl)和[初始化审计](../outputs/frontier/vlajepa_compact_closed_loop/episodes/vlajepa_compact__libero_spatial__1__41__clean__native4_chunk7/arena_initialization.json)展示证据结构；这不等于所有未来接触状态均已验证。

资源规则为导入前和权重物化前至少3.5GiB可用内存、自身RSS上限5.5GiB、运行时系统可用内存下限0.5GiB，每秒采样，保护仅退出自身进程。3.5GiB是独立探索预算，不是已证明的硬件最低配置。采样可能漏掉瞬时峰值，主机负载也会改变内存余量。

此前arena单独诊断中，空间reset后RSS为新方案约2.307GiB、原容量约2.304GiB，未显示物理RSS改善。因此即使本次回合能够运行，也不能把可行性改善归因于arena节省了约1GiB物理内存。分页探针的2.929GiB与原探针的3.521GiB同样属于不同时间的描述性测量，不是匹配消融或稳定加速证据。

## 工程限制与后续实验

本版本的JSON短暂访问拒绝重试覆盖runner和回合结果写入，但没有覆盖来自另一模块的`seal_episode`写入器，不能声称所有封存写入都已有重试。上游LIBERO的reset循环含`finally: continue`，可能吞掉原生warning回调抛出的异常，因此“所有warning都能立即终止reset”也不是已保证的性质。三个已封存回合实际warning均为空，未观察到这两条异常路径；已发生的本次中止原因是明确记录的RSS保护。冻结代码与已有证据保留，这些限制需在独立后续版本中处理。

后续[每任务独立进程实验](VLAJEPA_ISOLATED_RESULTS.md)已重新执行：前三任务再次成功，第四项首次推理时仍触发5.5GiB进程RSS保护。前三任务的模型输出、RNG、动作、状态和图像经[独立只读审计](../artifacts/vlajepa-process-isolation-consistency-20261016.json)完全一致；进程隔离本身未解决长任务的预算限制。进一步的[单长任务6GiB实验](VLAJEPA_LONG_HORIZON_RESULTS.md)独立记录，不能与本页前三成功拼成一次4/4，也不能覆盖本次未封存记录。

## 结论边界

目前能够支持的结论是：这一固定VLA-JEPA变体已在这台电脑上完成三个真实“观察—预测—执行—再观察”的机器人任务，但单进程串行四任务设计没有全部完成。第四项未封存，所以当前阶段不满足完整严格核验，部分报告不能充当完成凭据。每次规划的输入指纹、噪声、RNG、300个量化模块覆盖、NFE及前后处理动作都保留，不能用保存动作重放替代这些调用。

四个初态此前已用于SmolVLA研究，是已知任务的接入诊断，不是完整LIBERO benchmark或新的held-out质量集。检查点、数值路径和动作间隔不同，不能与SmolVLA结果直接解释为方法改进；没有原生BF16闭环对照，不能估计量化质量损失；共享主机少量计时不能证明稳定吞吐。一个或四个任务都不足以支持SOTA、泛化、统计显著性或真机实时能力声明。

原owned阶段的四次入口阻断、分页阶段的1个未封存/3个缺失，以及本compact阶段独立保留。新阶段的完成结果不能补足旧目录，也不与MINERVA232、SmolVLA缓存28、前瞻确认16或环境重放合并分母。
