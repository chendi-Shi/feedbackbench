# VLA-JEPA 资源释放后的恢复结果

2026-10-16，额外资源管理辅助下的第六次启动尝试完成了同一中断轨迹的恢复：到达 **520步时限，官方任务成功为false**。恢复执行与证据核验完成，机器人任务失败；两者不能混写。这是原398步中断尝试的连续性验证，**新增独立评测样本为0**，不能与其他三个任务拼成同协议4/4或合并成功率。[最终报告](../outputs/frontier/vlajepa_recovery/report.json)、[第六次运行记录](../outputs/frontier/vlajepa_recovery/runs/run_0006.json)、[恢复结果](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/result.json)。

本页记录资源操作及其证据边界；[恢复结果页](VLAJEPA_RECOVERY_RESULTS.md)说明轨迹验证，[原长任务结果](VLAJEPA_LONG_HORIZON_RESULTS.md)保留首次中断。原长任务目录仍未封存，没有被改写为一次未中断完成的评测。

## 实际恢复了什么

| 项目 | 本次实际记录 |
| --- | --- |
| 任务与初态 | `libero_10` task 1、init 41、seed 1000；原520步上限 |
| 保存动作重放 | 第1–392步；核对保存的8维机器人观测、双相机像素哈希及终态标志 |
| 边界重算与重叠 | 恢复原`call_0056`的forward前CPU RNG并真实重算；前6个动作对应第393–398步，逐步核对 |
| 新执行 | 第399–520步，共122步；第399步消费边界动作块剩余的第7个动作 |
| 实际模型执行 | 19次forward＝1次边界重算＋18次后续预测；每次4次动作头求值，共76 NFE |
| 最终状态 | `complete=true`，`success=false`，`terminated=false`，`truncated=true` |
| 独立样本 | 0个新增独立benchmark回合；不是重新运行全部历史57次forward |

[边界证据](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/boundary.json)记录模型动作、环境动作、输入机器人状态、噪声和forward前RNG五组数组精确相同，以及模块调用次数和4次NFE一致。[恢复证明](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/recovery.json)区分重放、重叠核对和新执行；[录像](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/rollout.gif)标注这些阶段。原记录没有完整物理内部状态，不能把观测一致扩大为完整MuJoCo内部状态等价。

本页编写时另作只读核验：原25份冻结源码哈希一致；两个controller的计划、源码快照、原协议和最终run哈希绑定一致；恢复[receipt](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/receipt.json)精确覆盖54个文件且全部哈希匹配；执行日志520行，初态及前398步核对日志399行，19份调用记录均完成、各覆盖300个量化模块。原严格`validate`命令退出码为0；哈希完整性检查本身不代替数值验证。

## 额外资源操作与失败尝试

用户授权释放资源后，先发生了以下操作；它们也属于本次结果的外部运行条件，不能只披露最后一次模型进程trim。

| 前置操作（UTC） | 记录与边界 |
| --- | --- |
| 05:24 停止闲置Ollama应用及服务 | [记录](../artifacts/frontier/vlajepa_resource_release_20261016_ollama.json)列明两个PID及创建时间，操作前已加载模型数为0；此次由PowerShell执行，没有单独源码快照 |
| 05:26 回收三个ChatGPT renderer的工作集 | [记录](../artifacts/frontier/vlajepa_resource_release_20261016_renderers.json)、[helper源码副本](../artifacts/frontier/vlajepa_resource_release_20261016_renderers.source.py)；未终止这些进程，private usage没有下降 |
| 05:29 回收OneDrive工作集 | [记录](../artifacts/frontier/vlajepa_resource_release_20261016_onedrive.json)、[helper源码副本](../artifacts/frontier/vlajepa_resource_release_20261016_onedrive.source.py)；进程仍运行，未修改同步配置 |

以上早期记录与源码副本保留操作过程；它们不具备后续controller那样完整的事前计划—源码—最终run哈希链。特别是Ollama停止是真实服务操作，不能将后续准备记录中的`terminated_processes=0`概括成整个会话从未停止服务。

前四次预检阻断记录均保留在原目录。第五次尝试采用事先保存源码与计划的外部controller，但入口内存不足，运行退出码3、`weights_loaded=false`，controller的`interventions`为空，**未执行模型进程trim**。证据：[计划](../artifacts/frontier/vlajepa_recovery_import_trim_20261016.plan.json)、[源码快照](../artifacts/frontier/vlajepa_recovery_import_trim_20261016.source.py)、[最终审计](../artifacts/frontier/vlajepa_recovery_import_trim_20261016.json)、[run0005](../outputs/frontier/vlajepa_recovery/runs/run_0005.json)。五次预检阻断不是五个机器人评测回合。

第六次尝试前，单独的资源准备程序对三个已识别的ChatGPT renderer及OneDrive各执行一次工作集回收；此准备记录中停止进程数为0。随后启动原恢复入口，在导入结束的stdout标记处，仅对本次新建的实际模型解释器调用一次`K32EmptyWorkingSet`。证据：[准备记录](../artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.json)、[准备源码](../artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.source.py)、[controller计划](../artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06.plan.json)、[controller源码](../artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06.source.py)、[完成审计](../artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06.json)。

controller审计记录唯一目标PID为5948，创建时间为`1792129054.371122`；选择依据包括本次launcher后代、精确脚本参数、进程创建时间及打开句柄后的创建时间校验。API返回成功，调用前、调用时与调用后观察到的阶段均为`building_meta`。stdout不是同步屏障，因此这些阶段是实际记录，不能表述为一般性的无竞态保证。controller最后记录运行退出码0、`final_run_status=complete`，并绑定最终run0006哈希。

这些操作使本次执行有了**额外外部资源管理条件**。原25源指纹`04ca0526e94e67f01c6acb9538ed045a07678a50e8511938b58f95b3139bc86f`所绑定的模型、任务、数值实现及5.75GiB双预检门槛、6GiB自身RSS上限、0.5GiB系统可用内存下限均未改变；外部controller、准备程序及前置服务操作不属于旧25源指纹，另由上述源码、计划与审计绑定。不得将本结果表述为未受额外资源干预的原资源流程复现。

## 观测内存与耗时的解释

| 模型进程单次trim前后快照 | 数值 |
| --- | --- |
| RSS | 468,865,024 → 25,403,392字节，下降443,461,632字节 |
| 系统可用物理内存 | 6,297,686,016 → 6,358,298,624字节，增加60,612,608字节 |
| 进程private usage | 518,049,792 → 522,711,040字节，增加4,661,248字节 |

RSS下降不等于等量物理内存或提交额度释放。Windows工作集回收改变驻留页，页面可重新进入工作集；脏页写回和其他进程活动也影响系统快照，不能把差值作为受控因果结果。[微软工作集说明](https://learn.microsoft.com/en-us/windows/win32/memory/working-set)、[EmptyWorkingSet文档](https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-emptyworkingset)。本次没有修改页文件、冻结预算、模型精度或控制参数，也没有持续循环trim。

| 本次运行成本 | 记录值 |
| --- | --- |
| 导入 / 加载量化 | 18.722秒 / 63.480秒 |
| 19次模型forward合计 | 79.807秒 |
| 恢复回合墙钟 | 175.340秒；含reset、重放、重算与继续执行，不含模型加载和GIF编码 |
| reset及392步重放 | 75.773秒 |
| 边界重算及6步重叠 | 6.393秒 |
| 122步新继续执行 | 93.174秒 |
| 每秒采样的RSS峰值 | 5,700,780,032字节，约5.309GiB |
| 每秒采样的最低系统可用内存 | 1,011,326,976字节，约0.942GiB |

这些数字描述这次共享电脑上的执行，不用于与之前未trim或不同时间的运行比较吞吐、内存收益或最小算力要求。每秒采样也不能证明瞬时峰值未出现于采样间隔。仿真在模型推理期间暂停，20Hz是模拟控制频率，不能据此声称实时控制。

完整时限内未达到官方成功谓词是本次任务失败的依据。本页不把最后一帧或资源操作解释为失败原因；没有对应消融或重复实验来分离策略、环境和运行条件的影响。
