# VLA-JEPA 长任务中断后恢复：完整执行，任务失败

**2026-10-16最新终态：第6次启动完成恢复，执行到520步时限并封存，官方任务未成功。** [当前报告](../outputs/frontier/vlajepa_recovery/report.json)为计划1次恢复、完成1、官方成功0、完整任务失败1，`complete=true`；[结果](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/result.json)记录`success=false`、`terminated=false`、`truncated=true`。原CLI严格`validate`退出0。[完成审计](../artifacts/vlajepa-recovery-completion-audit-20261016.json)与[带阶段标记的录像](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/rollout.gif)提供对应证据。

这是原398步中断轨迹的恢复验证，**没有新增独立评测样本**。原`outputs/frontier/vlajepa_long_horizon`仍未封存；前三个独立进程任务与本次恢复可以按任务展示不同协议下的证据覆盖，不能合成同协议4/4或合并成功率。最新结果区分了工程执行完成与机器人任务成功：恢复完成，长任务失败。

## 实际执行与连续性检查

任务是`libero_10` task 1、init 41、seed 1000，指令为“put both the cream cheese box and the butter in the basket”。[恢复协议](VLAJEPA_RECOVERY_PROTOCOL.md)与[实际冻结文件](../outputs/frontier/vlajepa_recovery/protocol.json)指纹为`04ca0526e94e67f01c6acb9538ed045a07678a50e8511938b58f95b3139bc86f`，绑定25个源文件和原中止目录的130个文件。冻结的策略、处理器、控制算法和数值设置未修改；本次资源准备有额外干预，见下节，不能称资源执行流程完全未变。

| 执行部分 | 实际结果 | 证据与计数边界 |
| --- | --- | --- |
| 初态与保存动作重放 | 初态及第1–392步精确核对通过 | 使用原动作，不重新调用模型；核对8维机器人观测、双相机像素哈希和终止/成功标志 |
| 边界模型重算 | 恢复原`call_0056`的forward前CPU RNG，实际重算1次完整forward；5类数组逐值一致 | 输入状态、RNG、噪声、模型动作、环境动作全部精确一致，300个量化模块实际执行、4NFE；不推测不存在的after RNG |
| 重叠边界 | 第393–398步共6步与原保存前缀精确相同 | 使用重算动作块前6个动作；边界真实forward后的RNG自然延续到后续调用 |
| 新执行 | 第399–520步共122步 | 第399步执行重算块的第7个动作，之后按最新观察完成18次新规划 |
| 模型实际调用 | 1次边界重算＋18次新规划＝19次，76NFE | 原57次历史调用另行保留，不能记成本次新调用；392步纯重放不计模型计算 |
| 终态与封存 | 520/520步，任务失败；恢复receipt及严格验证通过 | 未提前达到官方成功谓词；原398步目录仍未封存 |

[逐步恢复检查](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/recovery_checks.jsonl)包含初态和398步前缀比对；[边界证据](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/boundary.json)为`exact_recomputed_boundary`。严格验证还核对边界实际after RNG与下一次调用before RNG一致、后续调用使用最新机器人状态和双相机、动作队列对应真实执行日志、初末图像、轨迹与逐文件receipt。

原数据没有保存完整qpos/qvel等物理内部状态，所以精确比较支持的是已保存机器人观测、图像和标志一致，不能写成完整物理内部状态等价。录像将前392步标为`SAVED ACTION REPLAY`、393–398步标为`RECOMPUTED OVERLAP`、399步起标为`NEW CONTINUATION`；它不是一次全新且未中断的520步模型运行。

[arena记录](../outputs/frontier/vlajepa_recovery/episodes/vlajepa_recovery__libero_10__1__41__clean__native4_chunk7/arena_steps.jsonl)覆盖reset及520步，共521条，容量268,435,456字节，最高使用8,646,500字节，八项warning计数全部为0。已知上游reset可能吞掉持续warning异常的限制仍保留；本次实测零警告不等于所有异常路径均已验证。

## 资源干预与计时解释

冻结预算仍为模型导入前、加载前可用内存至少5.75GiB，自身RSS上限6GiB，运行期系统可用内存下限0.5GiB，每秒采样。[run_0006](../outputs/frontier/vlajepa_recovery/runs/run_0006.json)记录入口可用6.468GiB、物化前可用5.916GiB，均通过原门槛。

本次获得用户授权后，曾停止没有驻留模型的Ollama服务，并对若干桌面进程做working-set回收；第6次启动另外通过controller，在模型导入后、`building_meta`阶段，对识别出的实际VLA子进程调用**一次**`K32EmptyWorkingSet`。[桌面进程资源准备](../artifacts/frontier/vlajepa_recovery_resource_prep_20261016_retry06.json)及[子进程干预原始记录](../artifacts/frontier/vlajepa_recovery_import_trim_20261016_retry06.json)保留PID、创建时间、API结果和前后资源快照，详见[资源释放记录](VLAJEPA_RESOURCE_RELEASE_RESULTS.md)。额外controller属于原25个冻结源之外的资源干预，必须随结果披露。

该操作回收常驻工作集，没有证明私有内存被释放或算法内存需求降低，页面可再次进入工作集。本次不能作为“无外部资源干预”的运行，也不能与此前未干预的RSS或延迟作受控性能比较。算法源文件与保护阈值保持原样，不消除资源流程偏离这一事实。

| 成本或资源 | 第6次启动的实际值 | 统计范围 |
| --- | ---: | --- |
| 模型导入 | 18.722秒 | 运行记录的imports阶段 |
| 加载和量化 | 63.480秒 | 本进程重新加载固定检查点 |
| reset及392步保存动作重放 | 75.773秒 | 含初态创建、渲染与前缀验证 |
| 边界重算及6步重叠核对 | 6.393秒 | 包含真实边界forward和环境执行 |
| 第399–520步新执行 | 93.174秒 | 包含后续真实规划、环境执行与记录 |
| 恢复回合墙钟 | 175.340秒 | 上述三段之和；排除模型加载、GIF编码及外部资源准备 |
| 19次完整forward总时间 | 79.807秒 | 已包含在回合墙钟，不能再次相加；不是全任务从头推理时间 |
| 每秒采样最高RSS | 5,700,780,032字节（5.309GiB） | 受working-set回收影响，采样可能漏掉瞬时峰值 |
| 每秒采样最低系统可用内存 | 1,011,326,976字节（0.942GiB） | 此次共享主机运行的观测值，不是稳定最低配置证明 |

仿真在推理期间暂停，20Hz模拟控制不等于20Hz模型推理。重放省去此前56次模型调用，不能用本次175.340秒与原中断运行耗时相除宣称加速；导入、加载、录像编码、报告及资源准备也不能隐藏为零成本。

## 前五次启动阻断与原中断记录

五次阻断均保留，权重均未加载、没有重放或新执行动作，不是五个任务失败。第3、4次已完成模型依赖导入，在权重物化前的第二道门槛阻断；其余三次在入口门槛阻断。第6次首次产生恢复回合，启动次数不增加独立任务样本。

| 启动记录 | 入口可用GiB | 加载前可用GiB | 当次终态 |
| --- | ---: | ---: | --- |
| [run_0001](../outputs/frontier/vlajepa_recovery/runs/run_0001.json) | 5.724 | 未进入 | `blocked_preflight`，入口未达5.75 |
| [run_0002](../outputs/frontier/vlajepa_recovery/runs/run_0002.json) | 0.348 | 未进入 | `blocked_preflight`，入口未达5.75 |
| [run_0003](../outputs/frontier/vlajepa_recovery/runs/run_0003.json) | 6.272 | 5.666 | `blocked_preflight`，导入后未达5.75 |
| [run_0004](../outputs/frontier/vlajepa_recovery/runs/run_0004.json) | 6.241 | 5.746 | `blocked_preflight`，导入后未达5.75 |
| [run_0005](../outputs/frontier/vlajepa_recovery/runs/run_0005.json) | 5.576 | 未进入 | `blocked_preflight`，入口未达5.75 |
| [run_0006](../outputs/frontier/vlajepa_recovery/runs/run_0006.json) | 6.468 | 5.916 | 恢复完整封存，520步任务失败 |

[首次阻断审计](../artifacts/vlajepa-recovery-validation-20261016.json)和[第二次阻断审计](../artifacts/vlajepa-recovery-validation-20261016-retry02.json)绑定的是当时报告快照，均真实记录`strict_validation_passed=false`。它们是历史记录，不能覆盖为成功，也不与最新严格验证相矛盾。第二次启动前的[600秒内存观察](../artifacts/frontier/vlajepa_recovery_memory_wait_20261016_02.json)有41次采样，可用1.463–3.035GiB，没有达到5.75GiB条件；等待不算模型运行。后续资源授权和操作按独立记录披露，不能把早期“未停止服务”的事实扩展到第6次启动。

[原单长任务](VLAJEPA_LONG_HORIZON_RESULTS.md)在保存398步和57次完整forward后，由系统可用内存456,900,608字节（0.426GiB）触发保护；当时RSS5,648,232,448字节（5.260GiB）未超过6GiB。其[只读前缀审计](../artifacts/vlajepa-long-horizon-audit-20261016.json)与全部原文件继续保留，原目录仍没有最终receipt。原尝试是资源中断，恢复后的完整时限结果才是任务失败，两者分别报告。

## 实现验证与结论范围

[恢复定向测试](../artifacts/vlajepa-recovery-tests-20261016.xml)为75 passed、2.798秒；[发布校验器测试](../artifacts/publication-verifier-tests-20261016.xml)为39 passed、1.066秒；[同两组联合重跑](../artifacts/vlajepa-recovery-publication-tests-20261016.xml)为114 passed、3.730秒，均0失败/错误/跳过。联合重跑包含前两组，不重复相加。轻量替身测试仅验证实现契约；真实恢复依据本次实际forward、仿真前缀精确核对、后续执行、receipt与严格验证。

本次支持在披露的资源干预条件下恢复这一已知初态的保存轨迹，并完整观察到官方任务失败。它不支持完整LIBERO benchmark、量化无损、原生BF16等价、未见任务泛化、稳定最低内存、真实机器人实时控制或SOTA结论。不同协议的三个成功任务与这次恢复失败可逐项展示，不能变成新的一组四任务独立评测。
