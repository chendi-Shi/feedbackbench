# VLA-JEPA 每回合独立进程：四任务本机闭环结果

**2026-10-16实际终态：独立四回合中，前三个重新执行并封存成功，第四个在首次模型前向中触发RSS保护，整体未完成。** [部分报告](../outputs/frontier/vlajepa_isolated_closed_loop/report.json)为计划4、完成3、官方成功3、完整运行任务失败0、未封存1，`complete=false`。前三项全部来自本阶段的新进程与新调用，没有导入compact阶段的旧结果。进程隔离本身尚不足以在当前5.5GiB RSS限制下完成四任务设计。

该实验检验在同一台16GB、无NVIDIA GPU的Windows电脑上，固定预训练VLA数值变体能否通过逐任务独立进程完成真实闭环。相机、任务指令和机器人自身状态送入策略；预测动作顺序执行后，再从当前观察规划。全程使用本机CPU，零云费用；它是VLA机器人控制实验，不只是VLM图文问答或保存动作播放。

## 独立设计与冻结身份

上一[compact阶段](VLAJEPA_COMPACT_RESULTS.md)完成并封存三个任务，但切换第四个环境时，自身RSS达5.553GiB，超过5.5GiB保护上限；第四项没有执行预测动作。新阶段每个回合启动新的Python进程，重新校验与加载模型，回合结束后退出，再串行启动下一进程。它检验进程隔离的执行可行性，不预设上一阶段存在已证实的内存泄漏。

[书面协议](VLAJEPA_ISOLATED_PROTOCOL.md)与[实际冻结文件](../outputs/frontier/vlajepa_isolated_closed_loop/protocol.json)绑定**22个执行相关源文件**，协议指纹为`cf5de5849b5ae7a8cf299dff07532a2462720ed2df03e50cb2090746be530c51`。四个新case均为`vlajepa_isolated`，顺序spatial/object/goal/libero_10，各task1、init41、seed1000，时限280/280/300/520。前三项虽与旧实验任务相同，也必须在新协议下重新执行并保存所有实际调用。

模型、处理器与控制路径沿用既定变体：300个Qwen Linear动态INT8量化、完整BF16 token embedding私有文件映射、其余推理部分及动作头FP32，原四步flow、七步动作块；无训练、额外预热或控制兜底。环境为256MiB MuJoCo arena、`nconmax=5000`，保留360×360双相机、官方224×224模型预处理、20Hz相对OSC、hard reset和10步settle。每进程使用两个CPU计算线程和一个inter-op线程，不并发加载多个模型。

执行前重新检查[分页单观测逐值一致性](VLAJEPA_PAGED_RESULTS.md)和[978动作环境重放](LIBERO_ARENA_REPLAY_RESULTS.md)的身份与实际数组/图像。这两项是前置验证，不计入四个模型评测回合。[隔离入口与契约测试](../artifacts/vlajepa-isolated-tests-20261016.xml)为**22 passed、0失败/错误/跳过，0.773秒**；测试用例不是机器人任务，也不与其他历史测试批次相加。

## 四回合结果

以下仅用本阶段实际证据统计。三个完整回合共327个动作、48次完整返回的模型规划、192次动作头求值；第四项另有一次已启动但未返回的前向，没有执行预测动作。完整到达规定时限而未成功才属于任务失败；第四项是资源中止，不是任务失败，也不能从分母中删去后称整组100%成功。

| Suite | 本阶段完成状态 | 官方成功 | 动作/上限 | 规划次数 | 动作头NFE | 推理总秒数 | 回合墙钟秒数 |
|---|---|---|---:|---:|---:|---:|---:|
| libero_spatial | 完成并封存 | 是 | 109/280 | 16 | 64 | 71.614 | 109.600 |
| libero_object | 完成并封存 | 是 | 128/280 | 19 | 76 | 79.288 | 106.920 |
| libero_goal | 完成并封存 | 是 | 90/300 | 13 | 52 | 53.162 | 79.410 |
| libero_10 | RSS保护中止，未封存 | 未评定 | 0/520 | 1次启动、0次返回 | 未完成测量 | 未完成测量 | 未完成测量 |

三个成功均为`terminated=true`、`truncated=false`，模型调用包含真实当前观察、RNG/噪声、300个量化模块覆盖和四次动作头求值；逐步执行、机器人状态、图像、录像及arena警告随回合封存。每回合的指令与证据如下：

| Suite | 真实任务指令 | 本阶段结果与录像 |
|---|---|---|
| spatial | pick up the black bowl next to the ramekin and place it on the plate | [结果](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_spatial__1__41__clean__native4_chunk7/result.json)、[录像](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_spatial__1__41__clean__native4_chunk7/rollout.gif) |
| object | pick up the cream cheese and place it in the basket | [结果](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_object__1__41__clean__native4_chunk7/result.json)、[录像](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_object__1__41__clean__native4_chunk7/rollout.gif) |
| goal | put the bowl on the stove | [结果](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_goal__1__41__clean__native4_chunk7/result.json)、[录像](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_goal__1__41__clean__native4_chunk7/rollout.gif) |
| libero_10 | put both the cream cheese box and the butter in the basket | [首次前向启动记录](../outputs/frontier/vlajepa_isolated_closed_loop/episodes/vlajepa_isolated__libero_10__1__41__clean__native4_chunk7/call_0000.json)；无完成结果或动作录像 |

录像用于检查行为，任务成功取自LIBERO官方成功标志，不能仅凭末帧推断。三个封存回合的逐调用前向p50/p95依次为spatial **4.183/5.707秒**、object **4.069/4.678秒**、goal **3.924/4.824秒**；总前向时间204.064秒。分位数是本次调用的描述分布，不是置信区间。第四次未返回前向的耗时与NFE没有完整测量，不填为0。

## 逐进程成本与资源

每个worker都重新导入、构造、加载和量化同一检查点，其加载开销需要逐项报告，不能隐藏或与常驻单模型的回合墙钟混为一谈。回合`wall_seconds`包括reset、控制、物理和相机渲染，排除模型加载与GIF编码；完整操作耗时还可能包含来源检查、报告、进程启动和人工串行间隔。四个worker的阶段耗时之和不等于用户等待的精确端到端时间。

| Suite | 导入秒数 | 加载和量化秒数 | 每秒采样最高RSS/GiB | 每秒采样最低系统可用内存/GiB | Worker与运行证据 |
|---|---:|---:|---:|---:|---|
| libero_spatial | 29.701 | 86.336 | 4.993 | 0.986 | [worker1](../outputs/frontier/vlajepa_isolated_closed_loop/workers/worker_0001.json)、[run1](../outputs/frontier/vlajepa_isolated_closed_loop/runs/run_0001.json) |
| libero_object | 22.014 | 70.973 | 4.984 | 1.045 | [worker2](../outputs/frontier/vlajepa_isolated_closed_loop/workers/worker_0002.json)、[run2](../outputs/frontier/vlajepa_isolated_closed_loop/runs/run_0002.json) |
| libero_goal | 21.028 | 72.182 | 4.683 | 1.208 | [worker3](../outputs/frontier/vlajepa_isolated_closed_loop/workers/worker_0003.json)、[run3](../outputs/frontier/vlajepa_isolated_closed_loop/runs/run_0003.json) |
| libero_10 | 19.689 | 69.191 | 未汇总；停止瞬间5.503 | 未汇总；停止瞬间0.828 | [worker4](../outputs/frontier/vlajepa_isolated_closed_loop/workers/worker_0004.json)、[run4](../outputs/frontier/vlajepa_isolated_closed_loop/runs/run_0004.json)、[停止记录](../outputs/frontier/vlajepa_isolated_closed_loop/runs/run_0004.memory_stop.json) |

四个新进程累计加载和量化298.682秒，包括随后中止的第四项，不能只计三个成功回合的加载。前三个底层run状态为`stopped_after_requested_limit`：这是每worker按`limit=1`完成一项后正常退出，worker账本均为`complete`并绑定新封存结果，不是运行失败。

第四项已完成reset并保存初始双相机、arena审计及首次前向启动记录，但没有`call_0000.npz`、执行动作记录或receipt。11:54:57（北京时间）保护记录的自身RSS为**5,908,762,624字节（5.503GiB）**，比5.5GiB上限高3,182,592字节；系统可用内存为**889,200,640字节（0.828GiB）**，仍高于0.5GiB下限。实际触发条件是进程RSS，不是系统内存下限。退出码72，独立停止记录与worker PID/创建时间一致；worker4的`started`和run4的`running_episodes`是未更新的中间快照，不表示进程仍在运行。

这个中止发生在全新第四进程，说明仅隔离前序环境还不够；不能把本轮称为已修复或完整闭环。瞬间只超过少量字节不构成修改既定上限后继续沿用本协议的理由。原始未封存证据保留，后续若另设6.0GiB RSS、5.75GiB启动预算和0.5GiB系统余量的长任务诊断，必须使用独立协议与目录；不能把它的结果回填此处或与本页三个成功拼成4/4。

资源阈值保持不变：导入前及权重物化前系统可用内存至少3.5GiB，自身RSS上限5.5GiB、运行时系统可用内存下限0.5GiB，每秒采样，只退出自己的实验进程。3.5GiB是探索预算，不是硬件最低配置证明；每秒采样会漏掉部分瞬时峰值。每个worker的身份、前后完成数和起止时间须与实际运行对应，不能在常驻进程内重复调用入口冒充隔离。

逐调用p50/p95及合并调用分布只能描述本次调用，不能称为稳定部署P95或配对加速；不同任务的动作数、成功结束时刻及系统负载均影响计时。仿真在模型推理时暂停，20Hz模拟控制不代表CPU20Hz实时推理。arena从1.235GiB改为256MiB后，已有reset诊断RSS约2.304GiB与2.307GiB接近，因此没有已证实的约1GiB物理RSS收益；本轮若完成也不能倒推该因果结论。

## 核验、限制与阶段账目

阶段完成须同时满足：自身四个回合全封存；来源、逐文件哈希、逐步动作、调用及容量证据一致；四个独立worker账本完整；不带部分完成选项的严格`validate`通过。当前第四项未封存，整体不满足完成条件，报告生成成功不能代替这一条件。已完整封存回合可只读核验后跳过，未封存或错误回合不自动覆盖重跑。

三个封存回合的arena记录分别有110、129、91条，覆盖reset及每步动作；其最高arena用量分别为9,453,556、4,828,844、5,030,332字节，所有八项警告计数为0。容量审计和模型证据共同支持这些实际回合，不能外推第四项尚未完成的控制过程。

本版本补足receipt写入函数的有界PermissionError重试，仅重试文件写入，绝不重试动作或模型推理。仍保留上游reset循环可能吞掉warning回调异常的限制；正常返回后要求原生warning列表与全部八项计数均为0，不能把未验证的异常路径说成已保证立即终止。持续初始化无进展需保留中止记录，不能忽略警告后封存。

这是四个已知任务和初态的本机可行性评测，不是完整LIBERO benchmark、SOTA、未见任务泛化或真机结果。没有原生BF16闭环对照，不能声称INT8质量无损。即使全部完成，也只支持这一轮配置和共享主机条件下的实际运行，不证明长期稳定或普遍硬件兼容。

原owned阶段四次入口阻断、paged阶段一次初始化中止、compact阶段三成功加第四项初始化中止全部保留，分别见[owned结果](VLAJEPA_CLOSED_LOOP_RESULTS.md)、[paged结果](VLAJEPA_PAGED_RESULTS.md)与[compact结果](VLAJEPA_COMPACT_RESULTS.md)。这些证据不转记为本阶段结果；本阶段也不与MINERVA232、SmolVLA缓存28、SmolVLA确认16或978动作环境重放合并成功率分母。
