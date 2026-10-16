# 前沿模型与本机复现边界

**本机闭环进展（2026-10-16）：** VLA-JEPA的CPU数值变体现已完成三个任务的真实闭环，重复运行的动作与轨迹一致；第四个长程任务的[独立6GiB尝试](VLAJEPA_LONG_HORIZON_RESULTS.md)中止后，在披露的[资源干预](VLAJEPA_RESOURCE_RELEASE_RESULTS.md)下完成[精确恢复](VLAJEPA_RECOVERY_RESULTS.md)，执行至520步时限但任务失败；原中止记录保留，恢复不增加独立评测样本。见[独立进程结果与限制](VLAJEPA_ISOLATED_RESULTS.md)。这只更新本机实测，不改变下文明确标注日期的上游发布状态，也不构成完整官方基准或原生精度复现。

核对日期：**2026-10-14**。本项目当前约束是 **仅使用现有 Windows / Intel i5-1130G7 / 约 16 GB 系统内存 / 无 NVIDIA CUDA GPU 的电脑，零云费用**。下面的远程服务和 GPU 要求用于说明开放状态与可行性边界，不构成本项目的执行依赖。

**本机进展补记（2026-10-15）：** 新初态 full10 / noise2 前瞻闭环确认已完成 16/16 回合；独立动态 INT8 VLA-JEPA 变体已在完整检查点上通过一次真实观测前向，300 个量化 Linear 均实际执行，前向约 6.95 秒，1 秒采样的峰值进程 RSS 约 3.52 GiB。原生 BF16 同日仍因可用内存低于原 6 GiB 门槛退出，未执行前向。详见[后续研究结果](CONTINUATION_RESULTS.md)与[量化实现及实际证据](CPU_VLAJEPA_INT8.md)。这一进展不改变下文截至 10 月 14 日的官方发布核对，也不建立量化质量、原精度等价或 VLA-JEPA 闭环成功。

本页区分三件事：官方最新研究发布、官方可下载模型、已经在本机完成的实测。下列模型是当前官方开放生态的代表路线，不是效果排名；官网演示、公开权重和本项目复现成功也不能互相替代。发布日期按官方正文记录，当前仓库状态按核对日读取。

## Physical Intelligence：最新研究与公开权重不同步

| 层次 | 核实到的事实 | 对本项目的含义 |
|---|---|---|
| 最新官网公告 | **2026-10-13** 发布 Pi API beta，提供托管微调及推理，分批申请接入。公告没有给最新后端一个具体的 π 型号 | 不能自行称为 π0.8，也不能把 API 可申请等同于权重可下载；不纳入零云费用本机执行路线 |
| 最新具名研究文章 | **π0.7，2026-04-16**，研究可引导性和涌现能力 | 是跟踪前沿的研究材料；本次核对没有找到 openpi 中对应的官方权重发布 |
| 官方 openpi 公开模型 | 当前 README 列出 **π0、π0-FAST、π0.5**；π0.5 的仓库发布记录为 **2025-09**。包括 `pi05_base`、`pi05_droid`、`pi05_libero` 检查点 | π0.5 是官方公开可复现的版本，不应写成 PI 最新具名研究模型；官方 LIBERO 检查点已包含基准适配 |

来源：[Pi API 公告](https://www.pi.website/blog/api-beta)、[API 申请页](https://www.pi.website/waitlist)、[π0.7 官方文章](https://www.pi.website/blog/pi07)、[openpi 官方仓库](https://github.com/Physical-Intelligence/openpi)、[官方 LIBERO 示例](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)。

openpi 的官方硬件要求是 NVIDIA GPU：推理 **>8 GB 显存**、LoRA 微调 **>22.5 GB 显存**、全量微调 **>70 GB 显存**；支持声明限于已测试的 Ubuntu 22.04。README 还明确列出 PyTorch 实现目前不支持 LoRA，不能把另一后端的微调能力直接套到 PyTorch。系统内存容量不能代替这些显存及 CUDA 要求。官方远程推理接口存在，但本项目不据此假定已有免费服务或外部 GPU。[官方要求与实现限制](https://github.com/Physical-Intelligence/openpi#requirements)

## NVIDIA：公开 VLA 与世界动作模型

| 路线 | 官方发布与开放状态 | 官方硬件路径及本机限制 |
|---|---|---|
| **GR00T N1.7** | 当前官方仓库标为 N1.7 GA，3B 参数；2026-03-16 公告为早期访问，2026-07-07 官方技术文章介绍公开代码与权重。提供基础模型及 LIBERO、DROID、SimplerEnv 检查点 | 推理要求 **1 张 16 GB+ 显存 GPU**；微调建议 **40 GB+ 显存**。官方是 Linux / CUDA 路径，依赖的 `nvidia/Cosmos-Reason2-2B` 还需 Hugging Face 访问授权；没有据此验证本机 CPU 运行 |
| **DreamZero** | 论文提交于 **2026-02-17**。官方项目公开训练代码、14B DreamZero-DROID、DreamZero-AgiBot 权重，联合预测未来视频与动作 | 官方分布式推理要求**至少 2 张 GPU、CUDA 12.9+**，测试于 GB200 / H100。官方另有需申请的托管接口，但不属于本项目本机路线 |
| **Cosmos 3** | 官方仓库记录 **2026-05** 发布；模型族包含 Edge 4B、Nano 16B、Super 64B，统一处理世界理解、生成及动作等模态。**Cosmos3-Edge-Policy-DROID 4B** 模型卡明确发布于 **2026-07-20** | Edge 的“端侧”仍指有 NVIDIA GPU 的设备。策略模型卡列出 Linux、NVIDIA GPU 及 BF16 测试路径；不能因为参数较少就宣称本机已能复现 |

GR00T 来源：[官方仓库](https://github.com/NVIDIA/Isaac-GR00T)、[2026-03-16 公告](https://nvidianews.nvidia.com/news/nvidia-and-global-robotics-leaders-take-physical-ai-to-the-real-world)、[2026-07-07 技术文章](https://developer.nvidia.com/blog/develop-humanoid-robot-policies-end-to-end-with-nvidia-isaac-gr00t/)、[3B 模型卡](https://huggingface.co/nvidia/GR00T-N1.7-3B)、[LIBERO 检查点](https://huggingface.co/nvidia/GR00T-N1.7-LIBERO)、[硬件指南](https://github.com/NVIDIA/Isaac-GR00T/blob/main/getting_started/hardware_recommendation.md)。

DreamZero 来源：[论文](https://arxiv.org/abs/2602.15922)、[NVIDIA GEAR 项目官方仓库](https://github.com/dreamzero0/dreamzero)、[DROID 模型卡](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)。这里引用作者项目 `dreamzero0/dreamzero`，不将第三方同名仓库或复现当成 NVIDIA 官方实现。

Cosmos 来源：[官方仓库](https://github.com/NVIDIA/cosmos)、[技术报告](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)、[Edge-Policy-DROID 模型卡](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID)。该策略是基于 Edge 的后训练示例检查点，不能把所有 Cosmos 基础模型直接视为已适配任意机器人的控制策略。

Cosmos Edge 模型卡确有 **16 GB Jetson Thor T2000** 的测量，但该设备带 NVIDIA GPU，部分测量使用交换空间，不代表本机 16 GB CPU 笔记本可运行。模型卡中的多组延迟采用不同动作块、图像处理和采样设置，也不能合并成一个通用实时性能结论。[测量配置及限制](https://huggingface.co/nvidia/Cosmos3-Edge-Policy-DROID#pbr-performance-benchmark-reporting)

**GR00T N2 的开放范围需单独辨认。** 2026-03-16 官方公告将其作为基于 DreamZero 研究的下一代世界动作模型预告，计划年底提供。核对时 NVIDIA 账号已有一个特定的 Agile One S 行走部署资产，但模型卡明确其只包含一个部署模型，不能据此宣称通用 N2 训练、通用权重和完整基准已全部开放。[官方预告](https://nvidianews.nvidia.com/news/nvidia-expands-open-model-families-to-power-the-next-wave-of-agentic-physical-and-healthcare-ai)、[特定行走部署模型卡](https://huggingface.co/nvidia/agile_one_s_walk_n2_1680)

## 许可按具体代码与权重分别记录

| 产物 | 核实范围 |
|---|---|
| GR00T N1.7 | 官方仓库代码为 Apache-2.0；`nvidia/GR00T-N1.7-3B` 与 LIBERO 模型卡明确权重为 NVIDIA Open Model License。不能仅据代码许可证把权重称作 Apache-2.0 |
| DreamZero | 官方代码仓库为 Apache-2.0；**`GEAR-Dreams/DreamZero-DROID` 模型卡**标注 CC-BY-NC-4.0。该结论限定于此检查点，不自动覆盖 AgiBot 或其它衍生权重，也不能据代码许可证宣称全部权重可商用 |
| Cosmos 3 | 官方仓库说明代码与模型采用 OpenMDW-1.1；`nvidia/Cosmos3-Edge-Policy-DROID` 模型卡也明确该许可 |

许可依据分别为上文对应官方仓库和具体模型卡。实际采用时应记录选定 revision 及其许可证文件；本页不将某个模型卡的许可泛化到整个机构或同名第三方模型。

## 本项目已有证据与新的本机选型

**截至2026-10-14的选型定位：主线是2026 ActionCache方法向SmolVLA的独立本机迁移；VLA-JEPA是另行隔离的原生BF16可行性探针。** 前者研究缓存检索、动作续算、成功率与计算成本的关系，后者核实一个新模型是否能在现有 CPU / 内存上完成真实推理。两者均不依赖云 GPU 或付费 API，也不预设正结果。

| 本机路线 | 已有证据与边界 |
|---|---|
| MINERVA 基础设施基线 | 40 覆盖 + 32 开发 + 128 主比较 + 32 延迟，共 **232 个完整回合**。旧计划的 SmolVLA 12 回合和 flow-step 4 回合暂停且未尝试；不能称旧 248 回合全部完成。见 [基线完整结果](BASELINE_RESULTS.md) |
| ActionCache → SmolVLA | **28/28回合已完成**：4开发+24测试；主比较full10 **7/8**、cache2 **6/8**，没有支持保持成功率并稳定加速。真实检查点原10步及8+2与上游逐位一致。采用独立 **offline frozen cache**，见[研究结果](RESULTS.md)、[冻结协议](CACHE_PROTOCOL.md)、[CPU profile](FRONTIER_CPU_PROFILE.md)；不把SmolVLA本身称为2026 SOTA |
| VLA-JEPA 原生BF16探针（2026-10-14） | 固定官方源码与完整 **6,163,212,550字节** checkpoint已下载并校验；真实前向入口执行后因 **5.768GiB可用内存低于固定6GiB门槛**而`blocked_preflight`退出，未构建模型、加载权重或执行前向。见[实际探针](../artifacts/frontier/vlajepa_real_probe.json)。这是当次可用RAM阻碍，不证明CPU移植本质不可能；历史meta检查仍只是结构证据 |

[ActionCache 作者论文](https://arxiv.org/html/2607.06370v2) 的 v2 日期为 **2026-08-03**。本项目借鉴其动作状态检索与细化机制，独立加入开发集成功回合准入和测试不可更新的约束；没有采用作者代码，不能宣称复现论文在 π0.5 / GR00T 上的数字。原 MINERVA 调度研究的负结果作为基线保留，不混入新研究的模型、任务或回合分母。

**VLA-JEPA 与 V-JEPA Policy 是不同论文、不同实现和不同开放状态。** [VLA-JEPA](https://arxiv.org/abs/2602.10098) 首次提交于 **2026-02-10**；Hugging Face 官方 [LeRobot 接入说明](https://huggingface.co/docs/lerobot/main/en/vla_jepa) 提供其移植及转换检查点，结合 Qwen3-VL、flow action head 和训练期 JEPA 世界模型。本项目选用的 `lerobot/VLA-JEPA-LIBERO` 是官方 LeRobot 端口的检查点，不把它改称为原作者未改动实现。

[V-JEPA Policy](https://arxiv.org/abs/2609.37250) 则首次提交于 **2026-09-29**。截至核对日，[作者仓库的 checkpoint release plan](https://github.com/breez3young/VJEPA-Policy#checkpoint-release-plan) 仍将 LIBERO、RoboCasa-GR1 策略及 DROID predictor 列为待发布；已公开的是代码与训练配方。本项目没有将 V-JEPA 2/2.1 的视觉编码器权重冒充完整策略权重，也不把 VLA-JEPA 的下载与实测归给 V-JEPA Policy。

VLA-JEPA 探针固定 LeRobot commit `ae2b57f12223c8c5184fcd814eb63cd9ce7de203`、策略 revision `735d9f692981e286ade093b5046627eda876e5d0`、Qwen 元数据 revision `89644892e4d85e24eaac8bacfd4f463576704203`。代码与具体权重许可分别核实，见 [第三方前沿材料记录](THIRD_PARTY.md#前沿研究方法与隔离探针)。探针下载完整策略 checkpoint（包含训练模块）及 Qwen 配置/tokenizer，只实例化官方推理所需模块，不另外下载 Qwen 基础权重；运行要求独立重模型时段、加载前可用内存检查和进程内存监控。本段原生BF16探针没有实际前向或任务结果，不能据此承诺其延迟或成功率；后继CPU数值变体的实测见页首链接。

在目前核实到的官方支持范围内，**没有证据支持上述 π0.5、GR00T N1.7、DreamZero 或 Cosmos 3 已在这台电脑完成复现**。这不等于证明任何 CPU 移植都不可能；它意味着不能把移植设想当成结果，也不能把收费算力、待审批 API 或其它硬件作为本次零云费用目标的隐含前提。
