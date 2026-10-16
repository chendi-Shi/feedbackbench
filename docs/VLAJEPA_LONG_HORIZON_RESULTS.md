# VLA-JEPA 长程任务：独立6GiB预算下的单回合验证

**2026-10-16实际终态：本阶段计划1回合，保存了398步动作和57次完整模型调用，随后触发系统可用内存保护，回合未完成、未封存。** [部分报告](../outputs/frontier/vlajepa_long_horizon/report.json)为计划1、完成0、成功0、完整运行任务失败0、未封存1。目标为`libero_10` task1、init41、seed1000、最多520步；尚未跑满时限，不能评为任务失败，也不能报0/1任务成功率。

此前[compact四任务阶段](VLAJEPA_COMPACT_RESULTS.md)完成前三项，第四项环境初始化时因自身RSS超过5.5GiB中止；随后[每回合独立进程阶段](VLAJEPA_ISOLATED_RESULTS.md)重新完成前三项，但第四项首次模型前向尚未返回时再次超过5.5GiB，未执行预测动作。因此进程隔离本身没有完成原四任务设计。本阶段针对尚缺完整结果的同一个长任务，另行声明更大的进程资源预算，原始中止记录保持不变。

## 独立预算与保持不变的内容

[执行前协议](VLAJEPA_LONG_HORIZON_PROTOCOL.md)规定：启动前和权重加载前系统可用内存至少**5.75GiB**，自身RSS上限**6.0GiB**，运行时系统可用内存下限仍为**0.5GiB**，每秒采样。启动阈值和允许上限是根据前次结果提出的探索预算，不是已证明最低硬件要求或保证可完成的配置。保护仅退出本实验进程，不调整其他项目，也不使用云算力。

模型、动作与环境语义沿用既有compact方案：固定官方checkpoint和处理器，300个Qwen Linear动态INT8量化、完整BF16 token embedding私有文件映射、其余推理部分与动作头FP32，原四步flow、七步动作块；256MiB arena、接触上限5000，完整360×360双相机、8维机器人自身状态和真实任务文字，官方224×224预处理、20Hz相对OSC、hard reset与10步settle。没有训练、改变随机种子或初态、降图像质量、提前截断时限或添加控制兜底。

入口为[单任务运行器](../scripts/run_vlajepa_long_horizon.py)，使用新的独立Python进程，输出目录为`outputs/frontier/vlajepa_long_horizon`。[实际协议](../outputs/frontier/vlajepa_long_horizon/protocol.json)已冻结22个执行源文件及模型、官方源码、处理器、资产、初态身份，指纹为`1ee163c469021461d0d0da6b1b02368b9cf6e734ab6c81939daf99dab2cf7100`；执行前复核[单观测动作一致性](VLAJEPA_PAGED_RESULTS.md)和[978动作环境重放](LIBERO_ARENA_REPLAY_RESULTS.md)。[入口测试](../artifacts/vlajepa-long-horizon-tests-20261016.xml)为30 passed、0失败/错误/跳过、0.578秒；这些测试不是模型回合，也不表示完整闭环已经通过。

## 单回合实际结果

| 项目 | 本阶段记录 |
|---|---|
| 计划范围 | 1回合，`libero_10` task1 / init41 / seed1000 |
| 实验完成与封存 | 完成0、未封存1；保护中止，退出码72 |
| 官方任务成功 | 末条保存记录为未成功；回合最终质量未评定，不计任务失败 |
| 执行动作/上限 | 已保存398步 / 520；末条`terminated=false`、`truncated=false` |
| 完整规划次数与动作头NFE | 57次完整返回，NFE228；57组调用JSON及NPZ |
| 推理时间与回合墙钟 | 已完成调用总前向283.880秒；完整回合墙钟无结果，不填0 |
| 导入、加载和量化成本 | 导入35.947秒；加载和量化71.236秒 |
| 每秒采样最高RSS/最低系统可用内存 | 无完整汇总；停止瞬间RSS5.260GiB、系统可用0.426GiB |
| 录像、receipt及严格核验 | 无最终录像/结果/receipt；不满足完整阶段严格核验 |

此任务指令为“put both the cream cheese box and the butter in the basket”，即把奶油奶酪盒和黄油都放进篮子。只有达到官方成功条件，或执行完整520步时限后保存证据并通过严格核验，才计为该回合执行完成；到达时限未成功属于任务失败，资源保护、异常或未封存分别报告。[逐步执行记录](../outputs/frontier/vlajepa_long_horizon/episodes/vlajepa_long_horizon__libero_10__1__41__clean__native4_chunk7/executed_steps.jsonl)最后一条是第398步，不能推断未执行的后122步会成功或失败。

57次已保存调用均为`complete`，全部具有对应NPZ并记录四次动作头求值；最后一次为[call_0056](../outputs/frontier/vlajepa_long_horizon/episodes/vlajepa_long_horizon__libero_10__1__41__clean__native4_chunk7/call_0056.json)，已经返回，不是另一个未完成前向。已完成调用前向p50为4.969秒、p95为6.426秒，仅描述该中止轨迹的调用分布。[arena逐步记录](../outputs/frontier/vlajepa_long_horizon/episodes/vlajepa_long_horizon__libero_10__1__41__clean__native4_chunk7/arena_steps.jsonl)有reset加398步，共399条，最高arena用量8,646,500字节，八项warning计数均为0。已有动作/调用证据完整保留，但没有整回合结果和receipt，不能替代封存回合。

[只读中止审计](../artifacts/vlajepa-long-horizon-audit-20261016.json)通过，入口为[审计脚本](../scripts/audit_vlajepa_interrupted.py)，没有导入Torch、加载模型或创建receipt。它核对22个冻结源、57次完整调用的300模块覆盖与NFE、57个NPZ哈希、保存RNG/noise哈希、398步动作与预测块前缀、每次重规划的最新状态/相机哈希，以及399条arena记录和零warning。审计状态是`verified_saved_nonterminal_prefix`，同时明确`completed=0`、`strict_validation_passed=false`。RNG未重新执行、后续相机未重渲染；保护触发时若有尚未落盘的进行中动作，本审计无法推断，因此398是可核实的保存前缀，不是对未保存执行的猜测。

## 本次中止的触发条件

[运行记录](../outputs/frontier/vlajepa_long_horizon/runs/run_0001.json)确认模型已加载。12:19:40（北京时间）的[独立停止记录](../outputs/frontier/vlajepa_long_horizon/runs/run_0001.memory_stop.json)记录自身RSS为**5,648,232,448字节（5.260GiB）**，低于6.0GiB上限；系统可用内存为**456,900,608字节（0.426GiB）**，低于0.5GiB下限。因此本次触发的是整个系统可用内存保护，与前两个阶段因自身RSS越界中止不同，不能将原因混写。

`run_0001.json`仍是中止前`running_episodes`快照，应与独立停止记录一起读，不表示进程仍在执行。停止瞬间的两项内存值也不是整轮最大RSS与最低余量汇总。共享主机的全局可用内存会波动；本次记录证明保护按实际条件中止，不能据此确定其他某个项目是原因，也不能保证仅扩大启动余量就一定完成。原目录不覆盖重跑；后续任何独立重跑都需公开保留本次中止及其执行范围。

## 成本、实现与解释边界

本次需要重新加载模型。导入、构造、加载/量化、模型推理、reset和环境执行应分开记录；回合墙钟包含reset、控制、物理与渲染，排除模型加载和GIF编码。单次任务与共享主机上的调用分布不构成稳定吞吐、配对加速或常驻部署成本的证明。仿真在推理时暂停，20Hz模拟控制频率不等于20Hz模型推理。

一秒采样可能漏掉瞬时峰值。若只留下停止瞬间的内存值，不能当作完整运行峰值/最低余量。arena容量降低的既有诊断未显示对应物理RSS改善，不能从本阶段结果反推约1GiB物理内存收益。没有原生BF16闭环对照，不能估计量化质量损失。

JSON与receipt写入的有界PermissionError重试只重试文件操作，不重试模型调用或动作。底层reset可能吞掉持续warning异常的限制仍存在；正常返回后要求原生warning和八项计数均为0，持续无进展必须保留中止记录，不能跳过警告封存。原已冻结源码、阈值与不完整回合不就地修改。

## 与四任务覆盖的关系

本阶段已中止并保持未封存。此前isolated阶段的三个已封存任务仍各自属于其5.5GiB RSS预算，本阶段属于6.0GiB预算；允许按任务索引展示不同协议下的实际覆盖，并同时保留此前两次四回合设计都未完成的事实。不能将三项旧结果与本页拼接成同一协议4/4、完整官方LIBERO成绩或无条件100%成功率。

本阶段是看到资源中止后的针对性可行性检查，不是新的held-out质量评测。一个已知任务/初态上的成功或失败都不足以支持SOTA、显著改进、泛化、真机能力或长期稳定运行。结果以该独立目录的实际记录为准，不能仅凭入口已实现、测试通过或报告生成声称闭环完成。

中止后另做了[600秒只读内存观察](../artifacts/frontier/vlajepa_memory_wait_20261016.json)，41个样本中的可用内存为5.925–6.483GiB，未达到6.75GiB连续60秒的等待条件，因此没有启动从头重跑。观察未调用模型，不增加实验尝试或评测分母；[观察脚本原件](../artifacts/frontier/vlajepa_memory_wait_20261016.source.py)一并保留。

后继[恢复协议](VLAJEPA_RECOVERY_PROTOCOL.md)已在新目录完成：第6次启动精确重放前392步、实际重算边界模型调用并核对6步重叠，随后新增执行122步至520步完整时限；官方任务未成功，恢复结果是**完整执行的任务失败**。19次实际forward包含1次边界重算和18次新规划，共76NFE，独立receipt与原CLI严格验证通过，见[恢复结果页](VLAJEPA_RECOVERY_RESULTS.md)及[完成审计](../artifacts/vlajepa-recovery-completion-audit-20261016.json)。

恢复启动前及导入后有用户授权的资源准备和working-set回收，包含额外controller对实际VLA子进程的一次`K32EmptyWorkingSet`，详见[资源干预记录](VLAJEPA_RESOURCE_RELEASE_RESULTS.md)。原冻结算法与阈值未改，但不能称资源流程完全无干预，也不能与本次中止轨迹直接比较RSS或速度。恢复仍属于原轨迹连续性验证，不是新的独立评测样本；本页原398步尝试及所有源文件保持未封存原状，不被后继结果改写为未中断完成。
