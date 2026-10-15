# 上游来源、归属与固定版本

FeedbackBench 使用公开机器人策略、模型处理器与仿真基准。本项目没有训练 MINERVA、SmolVLA 或 SmolVLM，也不把 LIBERO 任务、控制器、成功条件及公开论文成绩视为自己的成果。此表说明实际使用的来源；上游许可证与声明仍适用于相应材料。

## 策略源码与检查点

| 组件 | 本项目使用的版本与材料 | 上游声明与证据 |
|---|---|---|
| MINERVA / LeRobot 源码分支 | commit `64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e`；使用其 Tinyflow、SmolVLA、处理器及 ACT 时间集成实现 | [固定源码](https://github.com/k1000dai/MINERVA/tree/64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e)，[Apache-2.0 LICENSE](https://github.com/k1000dai/MINERVA/blob/64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e/LICENSE)，[NOTICE](https://github.com/k1000dai/MINERVA/blob/64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e/NOTICE)，[第三方声明](https://github.com/k1000dai/MINERVA/blob/64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e/THIRD_PARTY_NOTICES.md) |
| MINERVA 公开权重 | `k1000dai/MINERVA`，revision `1b4fb1743f00a7d8eb87c7059c446447907d12bf`，子目录 `t05_l1_0.54M` | [固定检查点](https://huggingface.co/k1000dai/MINERVA/tree/1b4fb1743f00a7d8eb87c7059c446447907d12bf/t05_l1_0.54M)；下载的根模型卡声明 Apache-2.0 |
| SmolVLA LIBERO 公开权重 | `HuggingFaceVLA/smolvla_libero`，revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102` | [固定检查点与模型卡](https://huggingface.co/HuggingFaceVLA/smolvla_libero/tree/6721902bc4d61e50a3bfdb11dfb4cb626f05d102)；模型卡声明 Apache-2.0 |
| SmolVLM 配置与 tokenizer | `HuggingFaceTB/SmolVLM2-500M-Instruct`，revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`；仅下载配置、词表和处理器资料 | [固定模型资源](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)；该 revision 的官方 Hub 元数据声明 Apache-2.0 |

MINERVA 是 LeRobot 的非官方分支。其许可证、NOTICE 和第三方声明保留在本机 `vendor/MINERVA/`；其中对其他内含代码的归属同样保留。列出这些声明不表示 FeedbackBench 使用了其中每个机器人驱动或模型实现。

MINERVA 的任务文本通过检查点内的 40 项映射变成任务 ID；它没有语言编码器。SmolVLA 则使用语言、图像和状态条件。本项目使用已经包含完整 VLM 参数的 SmolVLA 检查点，因此不另行下载一个 SmolVLM 基础权重文件。所选 SmolVLA 模型卡未完整列出训练数据细节；本项目没有据此声称审计了完整训练集、多个训练种子或预训练数据去重。

权重身份以如下字节哈希为准：

| 本机文件 | 字节数 | SHA-256 |
|---|---:|---|
| `models/minerva/model.safetensors` | 2,199,896 | `f173def4c0b11030161486401dce52e63471d4180c248779678eaa6952c32da5` |
| `models/smolvla/model.safetensors` | 1,218,047,032 | `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f` |

每个模型目录的 `download_manifest.json` 还保存配置、归一化器、tokenizer 等逐文件哈希和来源 URL。模型采用 safetensors；处理器使用上游配置与统计量。SmolVLA 的 CPU 构建使用 Transformers 官方 `no_init_weights` 上下文略过随后会被检查点覆盖的随机初始化，仍执行完整严格加载及非持久缓冲区验证；权重保持冻结，没有量化或训练更新。

## 仿真与资产

| 组件 | 固定来源 | 归属与范围 |
|---|---|---|
| LIBERO / `hf-libero` | 安装 `hf-libero==0.1.4`，使用其 BDDL、官方初态与任务成功判据 | [LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)，[MIT LICENSE](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/LICENSE)；安装 wheel 的许可证同为 Lifelong Robot Learning 的 MIT 文本 |
| LIBERO 场景资产 | `lerobot/libero-assets`，revision `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` | [固定资产快照](https://huggingface.co/datasets/lerobot/libero-assets/tree/0b3ea86be5fe169d0fd036ae63d1070ec09e90f6)；本机 manifest 记录 586 个文件、422,320,936 字节及逐文件校验 |
| robosuite | `1.4.0` | [固定 LICENSE](https://github.com/ARISE-Initiative/robosuite/blob/v1.4.0/LICENSE)：MIT，并包含有关 MuJoCo 派生部分的 Apache-2.0 声明 |
| MuJoCo | `3.3.2` | [固定 LICENSE](https://github.com/google-deepmind/mujoco/blob/3.3.2/LICENSE)：Apache-2.0 |

资产和代码的来源分别记录。经官方 Hub API 与本机快照核对，上述 `libero-assets` revision 没有统一 license 字段，也没有 README、LICENSE 或 NOTICE 文件；因此本页没有把 LIBERO 代码的 MIT 许可证自动套用到每个网格、纹理和物体资产。仓库默认不打包这些文件，下载脚本按固定上游来源恢复资产；如需重新分发完整资产包，应先补齐相应资产来源的声明。

Windows 兼容层只对固定 robosuite 源码作进程内适配：DLL 定位、GLFW 后端选择和日志路径。原始文件不变，原始与载入源码哈希及替换内容写入运行审计。它没有修改物理动力学、控制器、任务资产或成功条件。完整平台说明见 [RUNTIME](RUNTIME.md)。

## 方法与研究归属

动作分块、固定反馈周期、事件触发调度和时间集成均已有先例。这里的图像/夹爪变化门控是候选启发式，是否有收益由冻结协议下的真实结果决定，不声称这些思想由本项目首次提出。时间集成保持上游 ACT 的旧预测优先权重方向，来源可追溯到 [上游 ACT 实现](https://github.com/k1000dai/MINERVA/blob/64c3cc8be364eb62df4f9fdc4bdccb1be9d1118e/src/lerobot/policies/act/modeling_act.py)。与 AAC、RTC 和 LeRobot 异步推理的区别见 [RELATED_WORK](RELATED_WORK.md)。

本项目承担的是评测系统、因果调度接口、跨方法初态配对、验证、资源测量、消融与可追溯报告。源码提交、权重身份和成功率分母应随结果一起展示；上游论文分数不填入本机结果表。

## 依赖记录与可移植范围

`requirements-lock.txt` 是实际 Windows / Python 3.12 / CPU 环境的版本快照，不是多平台统一验证声明，也不是带所有 wheel 哈希的通用安装锁。它使用相对的 `-e ./vendor/MINERVA`，上游 commit 由 bootstrap 单独核对；本项目 editable 安装在 bootstrap 末尾完成。安装先使用 PyTorch 官方 CPU wheel 源，再按快照安装其余依赖。

Windows 运行环境位于 `%LOCALAPPDATA%/FeedbackBench/`，项目 `.venv` 是本机目录连接。复制仓库到另一台电脑后应运行 bootstrap 重建环境，而不是复制旧机器的连接、路径或已安装包。`scripts/run.py` 保留短路径，以适应 LIBERO 的长任务文件名；锁定版本不等于已经验证 Linux、macOS、GPU 或真机。

`repair_top_level_metadata.py` 只为六个已审计发行包补充缺失的 `top_level.txt`。它使用当前解释器的推断规则并验证模块/命名空间与 RECORD 证据，保留已有声明和原 wheel RECORD，记录新增元数据的内容与哈希；默认无 `--apply` 时仅预览并写审计。它是安装元数据修复，不是模型参数或第三方源代码修改。

其余运行依赖保留各发行包的许可证和 wheel metadata；当前版本见锁定快照，安装资料位于各包的 `.dist-info`。Windows 的 EGL 与 OpenCV 声明替代使 `pip check` 存在四项已解释差异，不能声称依赖检查完全无异常。模型、资产和运行时默认由 `.gitignore` 排除，项目代码的许可不能自动替代这些第三方材料的许可。

## 前沿研究方法与隔离探针

核对日期：**2026-10-14**。以下新增路线与既有 MINERVA 源树、232 回合基线结果分别保存。来源身份、许可和运行状态是不同事项：可下载、模型卡许可声明或 meta 架构检查均不能证明本机真实推理成功。

| 材料 | 固定来源 | 许可与采用范围 |
|---|---|---|
| ActionCache 研究方法 | [作者论文 arXiv:2607.06370v2，2026-08-03](https://arxiv.org/html/2607.06370v2) | 作为动作缓存/检索/细化的研究依据；本项目没有引入作者实现代码或新模型权重。`action_cache.py` / `cache_sampler.py` / `cache_policy.py` 为独立迁移，保持原 SmolVLA/LeRobot 归属；不把论文公开许可当作另一代码库的使用许可 |
| VLA-JEPA 的官方 LeRobot 端口 | `huggingface/lerobot` commit `ae2b57f12223c8c5184fcd814eb63cd9ce7de203`；[固定源码](https://github.com/huggingface/lerobot/tree/ae2b57f12223c8c5184fcd814eb63cd9ce7de203/src/lerobot/policies/vla_jepa) | [固定 LICENSE](https://github.com/huggingface/lerobot/blob/ae2b57f12223c8c5184fcd814eb63cd9ce7de203/LICENSE) 的主许可为 Apache-2.0，并保留其内含组件的额外归属声明；不能删掉附带的 MIT 等第三方文本。本机独立保存在 `frontier/vendor/lerobot/` |
| VLA-JEPA LIBERO 策略权重、处理器及统计量 | `lerobot/VLA-JEPA-LIBERO` revision `735d9f692981e286ade093b5046627eda876e5d0`；[固定模型卡](https://huggingface.co/lerobot/VLA-JEPA-LIBERO/blob/735d9f692981e286ade093b5046627eda876e5d0/README.md) | 下载的该 revision 原始模型卡在 YAML 与正文均声明 **Apache-2.0**；本结论只覆盖这个具体仓库快照，不自动替代原作者其它权重的许可。官方 [LeRobot 说明](https://huggingface.co/docs/lerobot/main/en/vla_jepa) 将其描述为由 `ginwind/VLA-JEPA` 转换的端口；本项目未自行训练 |
| Qwen3-VL 元数据与 tokenizer | `Qwen/Qwen3-VL-2B-Instruct` revision `89644892e4d85e24eaac8bacfd4f463576704203`；[固定模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md) | 该模型卡声明 **Apache-2.0**。本项目仅取配置、tokenizer、处理器及模型卡；不单独下载 Qwen 基础 safetensors，Qwen 推理参数来自上述机器人策略 checkpoint |

上述两个模型卡结论分别核对了本机保存的原始 README；LeRobot 代码的 Apache-2.0 不是推导权重许可的依据。探针文件和依赖记录见 [probe_vlajepa.py](../scripts/probe_vlajepa.py)：源码逐 Git blob 核验，模型/元数据逐文件 SHA-256 核验；完整策略文件已下载并通过校验，为 **6,163,212,550 字节**、SHA-256 `b2a163c16889f89fb1d5e570d95f5c62313c84d0ecebdd384cd87c35e9a8540c`。完整下载身份见 [manifest.json](../models/vlajepa/manifest.json)；下载通过本身不证明真实推理通过。

独立 overlay 只包含固定的纯 Python wheel：`diffusers==0.39.0`（SHA-256 `912aca51b5787365110806e984d5555735bf8a461073bb8459029d0bca7870ef`）及 `importlib_metadata==8.7.1`（`5a1f80bf1daa489495071efbb095d75a634cf28a8bc299581244063b53176151`），完整 wheel 与归属信息保存在 `.runtime/frontier_deps/`，不覆盖既有 venv 或冻结源码。源码、wheel、元数据及完整 checkpoint 的最终下载身份由探针 manifest 绑定；`manifest.pending.json` 保留下载进行时的历史记录及早期结构探针绑定。

VLA-JEPA 的 [2026-02-10 作者论文](https://arxiv.org/abs/2602.10098) 与 [2026-09-29 V-JEPA Policy 作者论文](https://arxiv.org/abs/2609.37250) 分别归属各自作者。后者的 [作者仓库](https://github.com/breez3young/VJEPA-Policy#checkpoint-release-plan) 在核对时仍待发布完整策略权重；其 MIT 代码许可不能视为未发布权重的许可，也不能将 Meta 的 V-JEPA 编码器等同于完整控制策略。本项目本次未采用该仓库代码或策略权重。

ActionCache 迁移的设计与证据范围见 [CACHE_PROTOCOL](CACHE_PROTOCOL.md) 和 [CPU profile](FRONTIER_CPU_PROFILE.md)；公开模型路线比较见 [FRONTIER_LANDSCAPE](FRONTIER_LANDSCAPE.md)。本项目贡献是独立实现、验证、冻结评测及可追溯结论，不将上游预训练、论文成绩或仅计划执行的前向测试计作自己的已完成结果。
