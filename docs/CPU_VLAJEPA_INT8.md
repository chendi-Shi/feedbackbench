# VLA-JEPA 单机动态 INT8 推理方案

**2026-10-15：一次真实观测上的量化动作推理与后处理已通过。** 在 16 GB、无 NVIDIA GPU 的 Windows 电脑上，推理所需权重已真实加载，300/300 个 Qwen 动态 INT8 Linear 均实际执行；单次前向 **6.954 秒**，1 秒采样的进程 RSS 最大值约 **3.521 GiB**。这证明本次配置能产生通过输出检查的动作块，不代表闭环任务成功、质量保持、原精度等价或稳定部署性能。

代码：[量化 probe](../scripts/probe_vlajepa_int8.py) · [对应测试](../tests/test_vlajepa_int8_probe.py) · [原 BF16 probe](../scripts/probe_vlajepa.py)。原 BF16 方案同日[再次尝试的记录](../artifacts/frontier/vlajepa_bf16_probe_20261015.json)为 `blocked_preflight`：可用内存约 **5.887 GiB**，低于原有 **6 GiB** 门槛，未构造模型、加载权重或执行前向。2026-10-14 的[首次阻断记录](../artifacts/frontier/vlajepa_real_probe.json)也保留，彼时可用内存约 5.768 GiB。这些是各次运行时的可用资源限制，不证明所有 CPU 都无法运行原精度模型。

## 固定来源与改变范围

本方案使用已下载、完整校验的官方 [VLA-JEPA-LIBERO checkpoint](https://huggingface.co/lerobot/VLA-JEPA-LIBERO/tree/735d9f692981e286ade093b5046627eda876e5d0) 与 [LeRobot 固定源码](https://github.com/huggingface/lerobot/tree/ae2b57f12223c8c5184fcd814eb63cd9ce7de203/src/lerobot/policies/vla_jepa)，不另取 Qwen 基础模型权重、不改主环境、不重新训练。来源文件及 SHA 见 [下载 manifest](../models/vlajepa/manifest.json)。权重文件为 6,163,212,550 字节，SHA-256 为 `b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c`。

这是在官方架构和权重上实现的**数值变体**。它与原 BF16 probe、SmolVLA 闭环试验分别记录，不共享成功率分母，不声称官方提供或验证了这条量化路径。

与前沿研究的关联是：围绕公开预训练 VLA 的视觉语言主干、flow 动作头和推理资源开销，验证一种受限硬件上的部署方案；不表示所选模型是当前效果最强的模型。选用的是 Hugging Face LeRobot 的 VLA-JEPA 端口，不是另一个项目 V-JEPA Policy，二者区别见[前沿来源核对](FRONTIER_LANDSCAPE.md)。主线的 SmolVLA 缓存／少步采样研究用真实 LIBERO 闭环回答质量与成本问题；本支线目前验证加载、一次前向和输出契约，不能用工程接入替代闭环结论。

| 部分 | 本方案处理 | 对前向的含义 |
|---|---|---|
| Qwen 300 个原生 `nn.Linear` | 逐层转换为 PyTorch 动态 INT8 Linear | 权重 INT8；内核还会动态量化激活，外部输入／输出 FP32 |
| 文本 token embedding | 权重保持 BF16；lookup 后用 hook 转 FP32 | 减少存储；文本后续计算接收 FP32 |
| 共享 `lm_head` | 先逐字节验证 checkpoint 两份值相同，再恢复官方共享 | 与 embedding 共用存储；官方取隐藏状态的推理路径不调用该词表输出层 |
| Qwen 其余推理张量 | FP32，保留原模块 | 包括视觉 Conv3d、位置 embedding、归一化参数 |
| Qwen autocast | 仅本变体在调用范围内关闭 CPU BF16 autocast | 避免视觉卷积把动态 INT8 Linear 的输入转成不支持的 BF16 |
| Action head | 全部 FP32，保留官方四步 flow 与后处理 | 未量化 action head，未减少四步预算 |
| 视频编码器／预测器 | 仅不实例化官方预测动作路径未调用的训练模块 | 仍校验完整下载；省略前缀显式列出，不能省略推理所需参数 |

这里不能简称为严格的 **W8A32**：尽管算子边界是 FP32，PyTorch 动态 INT8 内核内部会量化激活。本机小测试实际确认直接给动态 Linear 传 BF16 会报错，因此精度转换是显式设计的一部分。

Qwen 的根模块先注册视觉模型，视觉模型的首个参数是保留 FP32 的 patch Conv3d。脚本会检查实际 `model.dtype`、`visual.dtype` 均为 FP32，BF16 embedding 与 head 真正共享；前向 hooks 还要求全部 300 个动态 Linear 实际收到 CPU FP32 输入。Action head 的全部参数必须为 FP32。

## 流式加载与内存预算

模型先以 meta 参数构造，对所需推理 state 与 checkpoint 逐键、逐形状核对，并单独记录跳过的训练模块。加载器按 safetensors header 的精确偏移，用 `readinto(bytearray)` 读取一个张量，检查有限值后转换或打包，避免长期持有整个文件的 mmap。动态量化按层执行，不先生成完整 FP32 Qwen；最大 Linear 是 `4096 × 4096`，其单份 FP32 权重为 **64 MiB**。原始 BF16 缓冲、INT8 打包及临时工作区会另外占内存，64 MiB 不是转换峰值。

下表是静态参数载荷核算，不是实测 RSS：

| 存储部分 | 元素数 | 预计字节数 |
|---|---:|---:|
| 300 个 Qwen Linear 的 INT8 权重 | 1,811,939,328 | 1,811,939,328 |
| 104 个 Linear 的 FP32 bias | 245,760 | 983,040 |
| 唯一共享 BF16 token embedding | 311,164,928 | 622,329,856 |
| Qwen 其他唯一 FP32 参数 | 4,182,016 | 16,728,064 |
| 完整 FP32 action head | 155,182,343 | 620,729,372 |
| **合计** | — | **3,072,709,660（约 2.862 GiB）** |

若 token embedding 也采用 FP32，以上合计为 3,695,039,516 字节（约 3.441 GiB）。原 BF16 主干／FP32 action head 方案在恢复共享后，理论唯一权重为 4,875,793,436 字节（约 4.541 GiB）。这些数值均不含打包对齐与元数据、框架、激活、文件读取缓冲、转换临时空间或共享电脑上的其他程序，不能推断一定能运行。

新变体在导入前及权重物化前均要求至少 **4.25 GiB 系统可用内存**；运行时每秒监测一次，若自身 RSS 超过 **5 GiB** 或系统可用内存低于 **0.5 GiB**，只终止自己的进程并保存内存停止记录。采样监测不能保证捕获所有瞬时峰值。原 BF16 probe 的 **6 GiB** 门槛和实现保持不变；新的预算只对应上表明确不同的量化载荷。

## 实际结果与证据

[真实 probe 主记录](../artifacts/frontier/vlajepa_int8_probe_20261015.json)的状态为 **`passed_quantized_forward`**，`forward_started=true`、`inference_executed=true`、`forward_calls=1`。运行使用 Python 3.12.14、PyTorch `2.7.1+cpu`、`x86` 量化后端；文本与视觉 attention 均为 `sdpa`，Torch 两个计算线程、一个 inter-op 线程。输入 batch=1，动作头执行官方四步 flow；没有 warmup 或重复前向。

输入是[此前从真实 LIBERO 仿真保存的双相机与机器人状态观测](../artifacts/policy_smoke_observation.npz)，SHA-256 为 `114121b20b8103cfcdcf4906ac1938e6fc13ab1da9f45801bef586557687a0aa`，指令为 **“pick up the black bowl between the plate and the ramekin and place it on the plate”**。相机朝向已由官方 LIBERO 适配流程处理，此次不再次翻转；不是随机张量或合成目标。probe 读取这个固定观测，没有推进环境或执行预测动作。

[加载审计](../artifacts/frontier/vlajepa_int8_probe_20261015.loading.json)确认所有必需推理键均已消费、源张量有限、embedding/head 真实共享，300 个 Linear 完成动态 INT8 转换。实际前向的 300/300 个量化模块均收到 CPU FP32 输入。模型输出及官方后处理后的动作均通过形状与有限值检查，保存为[动作 NPZ](../artifacts/frontier/vlajepa_int8_probe_20261015.actions.npz)中的 `model_actions`、`environment_actions`：均为 **float32 `(1, 7, 7)`**。这只检查“一批、七步、七维”的输出契约，不能把有限值、形状正确或动作距离当作成功率或质量。

| 本次测量 | 结果 | 范围 |
|---|---:|---|
| 运行时导入 | 26.936 秒 | 不属于动作前向 |
| meta 架构构造 | 6.342 秒 | 未物化完整权重 |
| 逐张量加载与量化 | 101.351 秒 | 一次加载，含转换与检查 |
| 观测预处理 | 0.017 秒 | `predict_action_chunk` 外的策略预处理 |
| 单次 `predict_action_chunk` | **6.954 秒** | 含 Qwen 输入准备、Qwen 与动作头；不含加载、后续动作后处理或仿真 |
| 加载完成后 RSS | 3,696,209,920 字节，约 **3.442 GiB** | 单次快照 |
| 运行期采样 RSS 最大值 | 3,781,103,616 字节，约 **3.521 GiB** | 1 秒采样，不能视为精确瞬时峰值 |
| 运行期采样最低系统可用内存 | 2,679,341,056 字节，约 **2.495 GiB** | 共享电脑上的本次观测值 |

前向内部三段各调用一次：Qwen 输入准备 **0.132 秒**、Qwen 前向 **5.651 秒**、flow action head **1.149 秒**。它们是总前向计时的组成部分，不能再与 6.954 秒相加；本次脚本未单列动作后处理耗时。完整进程还包含 manifest 校验、审计写盘等成本，不能用表中计时简单相加替代进程 wall。

来源可追溯到[本次脚本快照](../artifacts/frontier/vlajepa_int8_probe_20261015.source.py)、[复用基础脚本快照](../artifacts/frontier/vlajepa_int8_probe_20261015.base_source.py)、[输入 manifest 快照](../artifacts/frontier/vlajepa_int8_probe_20261015.inputs.json)与[schema](../artifacts/frontier/vlajepa_int8_probe_20261015.schema.json)，其 SHA 均由主记录绑定。主记录 SHA-256 为 `7d9565ab05110a5cca4ba1230608dbf4fae135548cd4eb041420e8796e8aa647`；动作 NPZ SHA-256 为 `d6556d8a6fdf1c26245a25e8f7404bb121743f171f5d7b0eddbbf0f981595e15`。

另有[本轮代码测试](../artifacts/continuation-unit-tests.xml) **566 passed、1 skipped**，其中启用并通过两项真实 CPU 小测试：动态 INT8 Linear 与 BF16 embedding 输出转 FP32 的组合，以及独立 BF16 张量读取／解码。这些是代码／小算子测试，不计为机器人回合；上面的真实模型前向也独立于测试数量。

证据分别记录：

- 调用前落盘 `forward_started=true`；异常中止时保留尝试状态。
- 函数返回后立即落盘 `inference_executed=true`、`forward_calls=1`，即使后续验证失败也不抹去已执行事实。
- 只有所有加载、精度、形状、有限值与后处理检查通过，并保存动作 NPZ 后，才写 `passed_quantized_forward`。
- 每次保留两份 probe 源码快照与主记录；后续仅在实际完成对应阶段后产生 manifest 快照、schema、加载审计、观测与动作 SHA、分项耗时和内存采样。入口即被阻断的尝试不会具有完整加载／动作证据；失败及内存停止记录不覆盖。

## 求职讲解中的准确定位

当前可讲的工程贡献是：从固定 checkpoint 核算参数存储，保留共享权重，按层完成动态量化，显式处理 BF16／FP32 边界，并将尝试、前向返回和有效输出分开记录。已有证据支持“在 16 GB CPU 电脑上完成 VLA-JEPA 数值变体的一次真实观测动作推理，前向 6.954 秒、采样 RSS 最大值约 3.521 GiB”；应同时说明 batch=1、未做闭环、没有原精度前向对照，不能扩展为稳定部署或任务性能结论。模型架构、权重、原动作处理器及任务基准均归属上游；个人表述应符合自己实际理解和参与的范围。

面试时可以连接三个具体问题：为何完整 FP32 加载后再量化会增加峰值内存；为何外部 FP32 不等于动态 INT8 内核内部保持 FP32 激活；为何一次有限值动作输出仍不能证明机器人能完成任务。这些问题对应系统与验证能力。研究岗位的质量证据仍应回到[主项目的协议、闭环对照和负结果](PORTFOLIO_GUIDE.md)，不把本支线的小算子测试作为新的算法效果结论。

## 复现与后续质量验证

在已准备好本项目固定环境、源码 overlay、模型和观测的项目根目录执行；环境与固定下载入口见 [RUNTIME](RUNTIME.md#isolated-frontier-probe)。与本项目其他模型／仿真实验串行运行，不干预其他项目进程；等待可用内存达到门槛。使用新的输出名，脚本会拒绝覆盖任何已有结果或配套文件。

```powershell
# 可选：只验证真实小算子；不加载完整模型。
$env:FEEDBACKBENCH_INT8_KERNEL_TEST = "1"
python scripts/run.py -m pytest tests/test_vlajepa_int8_probe.py -q
Remove-Item Env:FEEDBACKBENCH_INT8_KERNEL_TEST

# 一次完整量化前向；该命令仍受固定内存门槛约束。
python scripts/run.py scripts/probe_vlajepa_int8.py --threads 2 --output artifacts/frontier/vlajepa_int8_probe_repeat_01.json
```

下一项质量实验应另行预声明：固定任务、初态、动作约定与完整 horizon，保存闭环成功／失败和原始轨迹；只有当相同权重的原精度基线也能运行时，才做匹配输入／噪声及闭环配对比较。目前原 BF16 前向仍被预检阻断，**没有数值精度或速度对照**，不能给出量化损失、等价性或加速比。脚本设置了随机种子，但没有保存动作头初始噪声，不能仅据相同 seed 宣称严格同噪声 parity。单次 probe 也不足以估计稳定延迟、精确峰值、实时控制能力或整套 LIBERO 表现。
