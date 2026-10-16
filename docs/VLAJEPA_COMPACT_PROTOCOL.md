# VLA-JEPA：分页权重与受审计仿真内存的四任务闭环协议

本独立协议在闭环首次执行前冻结，承接分页权重单观测验证和环境容量重放验证。研究问题是：保持已声明的 VLA-JEPA 动态 INT8 数值路径、相机、物理与控制配置，减少不必要的仿真 arena 预留后，能否在这台无独立 GPU 的 16 GB 电脑上完成四个 LIBERO 回合？原 owned 阶段和分页阶段的资源阻断、未封存尝试均保留。新结果不补足旧阶段，也不合并统计。

入口为 `scripts/run_vlajepa_compact.py`，输出为 `outputs/frontier/vlajepa_compact_closed_loop`，case model 为 `vlajepa_compact`。模型仍是 [分页协议](VLAJEPA_PAGED_PROTOCOL.md) 中的 `cpu_qwen_dynamic_int8_paged_bf16_embedding_fp32_action`：300 个实际执行的动态 INT8 Linear，完整 BF16 token embedding 私有文件映射，FP32 动作头，官方四步 flow 和七步动作块。处理器、检查点、随机数顺序、模型先加载的顺序均沿用原闭环实现。

## 唯一环境改变及前置证据

MuJoCo 3.3.2 旧 XML 的 `njmax=5000` 导致 arena 预留约 1.235 GiB。通过进程内 `MjSim.from_xml_string` 的临时封装，将 `<size>` 中 `memory/njmax/nstack` 容量属性统一替换为 `memory="268435456"`（256 MiB）。必须保留 `nconmax=5000` 和其他 size 属性；size 起始标签以外的 XML 字节保持不变。不得更改渲染分辨率、阴影、MSAA、几何、纹理、求解器或 hard reset。场景文件不落盘修改。

此改变不能仅凭“没有抛异常”认定物理等价。MuJoCo 容量不足可能跳过接触或约束求解，因此必须捕获原生 warning，并在每次 reset 和控制步检查完整 8 项 warning 计数、实际 arena、最高用量与接触数。任何 warning、超容量或 XML 语义不符均失败；保留证据，禁止继续动作或封存为成功。上下文退出时恢复原编译方法和 warning callback。

执行前环境重放已测得：首个 spatial reset 后 RSS 2,477,010,944 字节，与原容量诊断的 2,473,123,840 字节几乎相同。容量缩小主要减少预留地址空间，未证明降低物理驻留量。因此本阶段不预设内存收益；它检验已审计容量下完整模型和环境共同执行的可行性，结果仍受共享主机余量影响。

闭环前必须完成两个独立门槛：

1. 已保存观察上分页模型与原 owned INT8 模型的前后处理动作逐值一致，零差异；核对实际 300 模块、4 次动作头调用、全部输入与来源身份。
2. 独立 `probe_libero_arena.py` 用已封存 SmolVLA full10 的四任务、task 1、init 41、seed 1000 的全部动作重放，逐步 robot state、初始双相机、最终主相机及终止 flags 逐值一致。必须重放完整原轨迹，所有警告为零。计划、源文件、参考 receipt、全部参考与重放文件 SHA 被验证；运行入口还重新比较保留的数组和像素。该重放不加载策略、不产生新任务分母，仅支持这四条已知轨迹的容量改变验证，不证明所有物理状态等价。

## 四回合设计与预算

固定 `libero_spatial / libero_object / libero_goal / libero_10` 各 task 1、init state 41、seed 1000，时限为 280/280/300/520 控制步。最新双相机、8 维机器人自身状态和任务文本输入策略；没有物体状态或成功标签。360×360 观察、官方 224×224 模型预处理、20 Hz 相对 OSC、hard reset、10 步 settle、每 7 步重规划均不变。推理时仿真暂停，不宣称实时控制。两个 CPU 计算线程、一个 inter-op 线程，一次一个模型和环境，不使用云或调整其他项目进程。

沿用分页阶段探索性预算：导入前和权重加载前可用内存至少 3.5 GiB，自身 RSS 上限 5.5 GiB，运行时系统可用内存下限 0.5 GiB，每秒采样。3.5 GiB 不是已证实硬件最低配置，采样不能保证捕捉瞬时峰值。触发保护退出自身进程并保留原始中止记录；未执行、运行错误或未封存不计为任务失败。只有完整到达官方时限且未成功，才计为任务失败。

## 冻结与验证

冻结全部原闭环和分页源、当前入口、arena helper、重放脚本及本协议，共 20 个源文件；同时绑定官方模型与环境 manifest、初态、仿真资产、依赖、通过的两个门槛身份。仍沿用原每次规划动作、噪声、RNG、输入指纹、实际模块覆盖和 NFE 记录，以及逐步执行、轨迹、图像、视频和逐文件 SHA receipt。

新增 `arena_initialization.json` 保存每次编译输入/输出 XML hash、容量属性和原生 warning，`arena_steps.jsonl` 保存 reset 及每个控制步的容量/最高用量/接触/约束与警告。严格验证需要步数加一条容量记录，并确认每次编译只改变声明容量。临时 Windows 文件占用只对 JSON 写入做有界 PermissionError 重试，绝不重试动作或推理。

已完整封存的回合只读验证后可跳过；缺失回合可继续；部分、异常和未封存回合不覆盖重跑。任何后续改变预算或实现必须另立显式协议并保留本次结果。完成须以四回合封存及严格 `validate` 通过确认；报告分别列完成数、成功数、任务失败和运行中止。四个已知初态不是完整 LIBERO benchmark，不支持 SOTA、泛化、原生 BF16 质量、真机或统计显著性结论。

容量语义依据 [MuJoCo 3.3.2 size XML 文档](https://mujoco.readthedocs.io/en/3.3.2/XMLreference.html#size) 与 [内存分配说明](https://mujoco.readthedocs.io/en/3.3.2/modeling.html#memory-allocation)；实际实现以冻结版本和运行证据为准。
