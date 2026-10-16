# VLA-JEPA INT8 四任务闭环：准备状态与结果口径

**2026-10-16：本原始 owned embedding 阶段的四回合尚未实际执行。** 首轮入口在2026-10-15因3.658GiB可用内存不足5GiB而退出；之后一次重试为4.198GiB，前台等待后启动的一次在导入后仅剩4.545GiB。最新第四次入口有5.136GiB，导入后仅剩4.586GiB，再次退出。四次正式记录均为 `blocked_preflight`、`weights_loaded=false`。四个回合是计划范围，不能把入口阻断写成0/4任务成功或四次任务失败。

后续[分页 embedding 实验](VLAJEPA_PAGED_RESULTS.md)完成了单观测动作一致性检查；[compact闭环](VLAJEPA_COMPACT_RESULTS.md)与[独立进程闭环](VLAJEPA_ISOLATED_RESULTS.md)各完成三个真实任务，第四项分别因进程RSS保护中止。另有[单长程任务的6GiB预算实验](VLAJEPA_LONG_HORIZON_RESULTS.md)。这些阶段各自保留协议、结果和失败，不改变本页5GiB门槛，也不补足本阶段的四个缺失回合。

此前[一次真实观测的量化前向](../artifacts/frontier/vlajepa_int8_probe_20261015.json)已经通过，但当时没有推进环境或执行预测动作。本阶段要把同一数值变体接入LIBERO，反复“获取当前观察 → 预测七个动作 → 顺序执行 → 再观察”，检验完整任务过程。它研究的是VLA机器人策略，VLM是其中的图文理解主干，见[项目说明](VLA_PROJECT_OVERVIEW.md)。

## 当前已有证据

| 项目 | 已完成事实或当前状态 | 证据入口 |
|---|---|---|
| 执行协议 | 四任务、初态、时限、动作接口、内存保护及来源已在执行前封存 | [书面协议](VLAJEPA_CLOSED_LOOP_PROTOCOL.md)、[实际冻结文件](../outputs/frontier/vlajepa_closed_loop/protocol.json) |
| 运行器与只读核验 | 已实现逐回合执行、动作证据保存、封存校验及报告入口 | [runner](../scripts/run_vlajepa_closed_loop.py) |
| 当前接口与单元测试 | **47 passed，0 failure，0 error，0 skipped**；44项单元测试加3项真实微型PyTorch动态INT8接口测试 | [当前XML](../artifacts/vlajepa-closed-loop-interface-tests.xml)、[单元测试源码](../tests/test_vlajepa_closed_loop.py)、[真实微型接口测试](../tests/test_vlajepa_closed_loop_adapter.py) |
| 前台收尾入口 | **59 passed**；15项收尾流程模拟测试加44项既有单元测试，无新模型回合；各批次不加总 | [本轮XML](../artifacts/vlajepa-stage-completion-tests-20261016.xml)、[完成入口](../scripts/complete_vlajepa_stage.py)、[流程测试](../tests/test_complete_vlajepa_stage.py) |
| Windows文件占用修复 | **62 passed**；18项流程模拟测试加44项既有单元测试，包含短暂共享锁重试、永久失败及独立终态回归 | [修复后XML](../artifacts/vlajepa-stage-completion-recovery-tests-20261016.xml)；首次59项记录保留，不相加 |
| 首次定向测试 | **44 passed，0 failure，0 error，0 skipped**，保留首次批次 | [首次XML](../artifacts/vlajepa-closed-loop-tests.xml)；不与当前47项或历史566项相加 |
| 代码检查 | 本轮runner及测试的Ruff检查通过 | 与上行实现和测试对应，不作为模型质量证据 |
| 环境接口smoke | 无模型权重的单次no-op接口检查通过 | 仅检查环境接口；不计作任务样本，也不表示模型已加载 |
| 首轮正式启动 | `blocked_preflight`，可用内存3.658GiB低于固定5GiB；`weights_loaded=false` | [运行原始记录](../outputs/frontier/vlajepa_closed_loop/runs/run_0001.json)；没有新任务结果 |
| 第二次正式启动 | `blocked_preflight`，可用内存4.198GiB低于固定5GiB；`weights_loaded=false` | [第二次运行记录](../outputs/frontier/vlajepa_closed_loop/runs/run_0002.json)；未创建episode |
| 第三次正式启动 | 通过入口检查，导入运行库后4.545GiB不足5GiB，`blocked_preflight`；未加载权重 | [第三次运行记录](../outputs/frontier/vlajepa_closed_loop/runs/run_0003.json)；无机器人动作或任务结果 |
| 第四次正式启动 | 入口5.136GiB；导入后4.586GiB不足5GiB，`blocked_preflight`；未加载权重 | [第四次运行记录](../outputs/frontier/vlajepa_closed_loop/runs/run_0004.json)；无机器人动作或任务结果 |
| 部分报告 | 计划4、已完成0、成功0、任务失败0；四个计划回合均为missing | [部分报告](../outputs/frontier/vlajepa_closed_loop/report.md)、[状态JSON](../outputs/frontier/vlajepa_closed_loop/report.json)；不是0/4成功率 |
| 工程保存核验 | 实现验证通过；四份协议的冻结源码符合，两个既有ZIP哈希未变；实际任务执行仍为blocked | [独立验收记录](../artifacts/vlajepa-closed-loop-verification.json) |

三项真实微型接口测试实际执行PyTorch动态INT8运算，比较独立、未包裹的调用与闭环适配器的随机数和动作；还覆盖连续两次规划、使用新观察、reset后的复现，以及异常时hooks和patch的清理。这提高了接口可信度，但没有加载完整VLA-JEPA检查点或执行四个LIBERO任务，不能当成闭环成绩。

实际协议指纹为 `7a762b0d556f1b7e935e3237349891181ad1fd3de2463f0aac820fa72d2ee0b8`，由runner在执行前封存源码、模型、处理器、仿真资产及初态身份。书面协议和运行器不能为适配后续观察结果而修改；本结果页只解释实际状态。

首轮运行记录的可用内存为 **3,927,801,856字节（3.658GiB）**，固定要求为 **5,368,709,120字节（5GiB）**。入口检查后立即退出，没有加载权重，也没有创建episode目录。这条记录证明资源检查按规则阻止了本次尝试；它不评价机器人策略的任务质量，也不表示四个任务已经执行。

## 四回合的固定范围

| 条件 | 本阶段定义 |
|---|---|
| 模型 | 固定官方VLA-JEPA-LIBERO检查点的独立动态INT8变体，不训练权重 |
| 任务 | `libero_spatial`、`libero_object`、`libero_goal`、`libero_10`，各task1 |
| 初态与种子 | 每任务init41、seed1000；共**计划4回合** |
| 时限 | 依次280、280、300、520控制步；仅官方成功可提前结束 |
| 动作生成 | 保留原四步flow，预测七个动作后顺序执行，队列耗尽后获取最新观察；无时间集成或缓存 |
| 观察与控制 | 当前双相机、8维机器人本体状态和真实任务文字；相对OSC控制，仿真在推理时暂停 |
| 运行预算 | CPU两个计算线程、一个inter-op线程；一次只加载此模型和一个环境 |
| 内存保护 | 导入前、权重物化前至少5GiB可用内存；自身RSS上限5.5GiB，系统可用内存下限0.5GiB，每秒采样 |
| 预热 | 不另做预热；首调用计入回合，加载成本单列 |

这是一项 **diagnostic（小规模接入诊断）**，不是新的held-out质量集：init41此前已用于SmolVLA研究。不同检查点、数值实现、动作执行间隔和运行时使它不能直接成为SmolVLA的匹配对照。四个任务各一个初态也不足以支持完整LIBERO成绩、显著性、非劣、泛化或真机能力声明。

数值方案为动态INT8 Linear、BF16 embedding存储且输出转FP32，其余推理部分及动作头FP32。动态量化内核也会量化激活，不能简称W8A32。原生BF16此前未通过入口内存门槛，本阶段没有原精度闭环对照；即使成功，也不能归因于“量化无损”。详见[量化实现与已有前向证据](CPU_VLAJEPA_INT8.md)。

本阶段与MINERVA **232回合**、ActionCache–SmolVLA **28回合**、SmolVLA前瞻确认 **16回合**分别统计。环境接口smoke、模型前向探针和保存动作重放均不增加闭环评测分母。

## 运行与核验入口

在项目已有固定依赖、上游源码、检查点和仿真资产准备完成后，从项目根目录执行。下面使用新的复现目录，保留本次原始证据：

若继续项目内已有的四回合冻结阶段，可直接执行 `python scripts/run.py scripts/complete_vlajepa_stage.py --wait-seconds 600`。它最多在前台等待600秒达到固定5GiB门槛，然后串行调用原runner、报告和不带 `--partial` 的严格核验。只有四回合全部封存、报告和核验都确认完成，才返回0；部分/错误回合、锁冲突、等待超时或子命令失败都返回非零。默认不等待，不自动后台恢复，不清除锁或覆盖错误证据。每次操作日志写入新的 `.runtime/closed_loop_completion/` 子目录。

```powershell
# 固定协议与来源，不启动模型回合。
python scripts/run.py scripts/run_vlajepa_closed_loop.py --output outputs/reproduction-vlajepa-closed-loop --freeze-only

# 按冻结顺序运行；不足固定内存门槛时记录阻断并退出。
python scripts/run.py scripts/run_vlajepa_closed_loop.py --output outputs/reproduction-vlajepa-closed-loop

# 完整只读核验：不启动模型或环境；未完成阶段不能当作完整通过。
python scripts/run.py scripts/run_vlajepa_closed_loop.py --output outputs/reproduction-vlajepa-closed-loop --validate-only

# 从已有证据生成报告；缺失、错误和未完成状态仍须显式保留。
python scripts/run.py scripts/run_vlajepa_closed_loop.py --output outputs/reproduction-vlajepa-closed-loop --report
```

内存不足时等待系统自然恢复，不停止其它项目来满足门槛。内存保护只能退出自身进程；每秒采样不能保证捕捉所有瞬时峰值。完成并封存的回合可以核验后跳过；部分、错误或运行中保护退出的回合保留，不自动覆盖重跑。只有尚未开始的回合能在同一冻结协议下继续。

本次前台等待采样范围约3.096–5.228GiB；约570秒时启动第三次正式入口，最终在导入后资源检查处退出。收尾过程中还遇到Windows文件占用导致日志原子替换失败：保留了旧快照和待写终态，确认进程退出后恢复错误终态；原始协议和实验记录未改。当前入口会对短暂共享访问拒绝做有界重试，永久失败则写独立错误终态并返回非零。[本轮审计](../artifacts/closure-audit-20261016.json)区分模型资源阻断、操作日志错误及修复后的模拟回归测试。严格四回合核验仍不通过，不能声明阶段完成。

报告文件是该阶段目录内的 `report.json`、`report.md`、`episodes.csv`。当前已生成的是记录四次入口阻断和四个缺失回合的**部分报告**，不能作为完整闭环验证。第三、四次入口保留了导入阶段的内存采样；由于尚无模型调用或机器人动作，没有任务推理耗时及闭环运行内存测量，不能把缺失值当作零。逐回合动作、轨迹、图像及封存凭据只在实际回合运行后按过程保存。

## 最终结果怎样解释

结果必须同时列出**计划4回合、已完成、官方成功、官方失败、运行错误、缺失与未封存**。完成一次运行不等于完成任务；成功率分母不能通过删除错误或缺失回合来缩小。完整时限内未满足官方成功条件才属于任务失败，模型加载前的内存阻断不属于任务失败。

每个实际回合还应报告控制步数、规划次数、动作头求值次数、逐调用耗时、加载成本和内存采样。七动作队列用完后必须使用最新观察，不能把保存的单次动作块反复播放当成模型闭环。每次规划的输入指纹、噪声与随机状态、前后处理动作块，以及实际量化模块覆盖都应能回查。

若某个任务成功，它只支持该固定配置和初态上的一次成功；若失败，应保留完整过程并区分任务行为与运行错误。任何单次或少量耗时都不等于稳定吞吐；同步仿真的20Hz控制频率也不等于CPU能够20Hz实时推理。原始图像、动作与官方成功标志用于解释结果，不能据末帧臆测完整动作原因。
