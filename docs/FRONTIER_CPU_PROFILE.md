# SmolVLA 真实 CPU 分项测试

**2026-10-14 已完成：一个真实 LIBERO 观测、完整预训练 SmolVLA、2 次 warmup、3 轮交错顺序计时。** 本页用于判断本机成本构成与研究可行性，不是闭环成功率结果，也不是实际缓存加速实验。

证据为 [profile.json](../outputs/frontier/profile.json)、[profile.npz](../outputs/frontier/profile.npz)、[运行脚本](../scripts/profile_cache_sampler.py) 和 [原始真实观测](../artifacts/policy_smoke_observation.npz)。JSON 记录源码、上游源码、模型 manifest 和观测 SHA，完整运行版本、噪声、内存快照及每次调用分项；NPZ 保存噪声、中间 latent、归一化输出和后处理后的动作。

## 配置与测量范围

Windows / Intel i5-1130G7 / 约 16 GB 系统内存 / 无 CUDA；Python 3.12.14、PyTorch 2.7.1+cpu、NumPy 2.2.6、Transformers 5.5.4。该进程 Torch intra-op 为 2、inter-op 为 4，`OMP_NUM_THREADS=MKL_NUM_THREADS=2`。后续闭环 runner 设置 inter-op=1，因此不能把这里的毫秒数直接作为闭环调用预测值。

模型共 **604,934,176 参数**，CPU float32，固定 revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，权重 SHA-256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。实际严格加载一次；适配器构造耗时 **16.310 秒**。观测包含两张 `(3,360,360)` RGB 图像、8 维机器人状态和指令 “pick up the black bowl between the plate and the ramekin and place it on the plate”。

所有计时重复使用相同观测与 seed `37` 的相同 `(1,50,32)` 噪声。每个测量条件**独立重新执行观测预处理、当前 VLM 前缀计算、动作采样/续算和动作后处理**，没有共享已算好的上下文来降低完整调用成本。三种方法各占据测试顺序的第一、第二、第三位置一次。计时重复不是独立任务样本。

full10 从噪声执行 10 步，noise2 从同一噪声执行覆盖完整区间的 2 步；oracle2 使用 warmup 中该观测、该噪声自己的第 8 步 latent，再在新算出的相同观测上下文上续算原网格最后 2 步。**oracle2 是理想续算上界，未检索缓存，未计入生成所给 latent 的成本。** 它不能代表开发缓存遇到新初态时的实际命中或动作质量。

## 实测结果

单位均为秒，3 次测量的中位数；完整调用范围另列 min–max。

| 条件 | NFE | 完整调用中位数 | 完整调用范围 | 观测准备中位数 | 采样中位数 |
|---|---:|---:|---:|---:|---:|
| full10 | 10 | 10.202 | 7.806–13.904 | 6.945 | 3.253 |
| noise2 | 2 | 5.589 | 5.373–9.467 | 5.163 | 0.521 |
| oracle2：自身 latent 理想续算 | 2 | 8.571 | 5.720–9.878 | 7.878 | 0.688 |

预处理/后处理中位数分别为：full10 **0.00625 / 0.00020**，noise2 **0.00597 / 0.00016**，oracle2 **0.00404 / 0.00023**。准备包含输入检查、视觉编码与前缀 embedding、VLM、键特征导出；采样外层计时包含 sampler 调用与验证。各分项的中位数不可简单相加当作完整调用中位数，JSON 内嵌套计时也不能重复求和。

两次 warmup 完整耗时分别为 **8.012、7.382 秒**，不进入上表。正式计时的 full10 不导出中间 latent，以免额外收集工作影响其成本；闭环研究的采样包装仍可能导出 latent，因此两者具体测量路径也应区分。

正式测量前后内存快照：进程 RSS **3.104–3.105 GB**，系统可用内存 **3.645–3.760 GB**，此处 GB 按十进制字节换算；不是峰值内存，也没有连续采样。实际展平特征维度为 **169,920**，对应 128 图像 token、48 语言槽、1 个状态 token，隐藏维度 960。

## 数值验证与结论边界

三轮 full10 和 oracle2 的全部 `(1,50,7)` 归一化输出、后处理动作均与 warmup 参考**逐位一致**。这验证自身状态在原 8+2 网格上的确定性续算；独立上游 parity 测试另见 [真实 checkpoint 测试](../tests/test_cache_sampler_integration.py)，该测试直接对照原始 `predict_action_chunk`，不是只将包装器与自身比较。

noise2 相对 full10 的最大归一化动作绝对差为 **0.947438**，限定于这一个观测、一个噪声的 50 动作输出。该值表示模型输出差异，不是成功率、真实动作误差或质量评分；与闭环协议只保存首 10 个已执行动作的一致性指标也不同。

当前观测准备在各条件中仍占主要成本，而且同一计算条件随轮次明显波动。上表不能证明 noise2 固有地比 oracle2 更快，不能给出可靠真实缓存加速倍数，更不能证明质量保持。所有条件都未执行检索或计入投影/搜索、缓存建造、仿真、磁盘 I/O；固定噪声预创建、内存快照和事后数组验证也在调用计时外。实际缓存是否有效，应由 [28 回合冻结协议](CACHE_PROTOCOL.md) 的成功率、命中率、真实 NFE 和完整调用成本共同判断。

本研究关联 2026 年 [ActionCache 作者论文](https://arxiv.org/html/2607.06370v2) 的动作状态复用问题；本机对象是既有 SmolVLA，不宣称其为最新 SOTA，也不声称复现论文公布的加速倍数。公开模型与硬件范围见 [前沿路线核对](FRONTIER_LANDSCAPE.md)。

## 证据完整性与重跑

已用 `allow_pickle=False` 读取 NPZ，核对全部数组的形状、dtype、有限值、6 组 full10/oracle2 对参考输出的逐位一致，以及全部绑定源码 SHA。

| 产物 | SHA-256 |
|---|---|
| profile.json | `5cb56a3e854820f0ce8c5b0c261a6e0041495e69dd63ba99c596ee38e1b72fea` |
| profile.npz | `b42bd0d8b91208b5ded67d40796ae01f9ed07103d9f27200056b7f633eec43b4` |
| profile_cache_sampler.py | `70420e53190b93638b1d9044c5e166ea978bff4d87f07ac11089a1cc24805609` |

在已安装本项目运行环境与固定模型的项目根目录中串行执行，选择新的输出名，避免覆盖已有证据：

```powershell
python scripts/run.py scripts/profile_cache_sampler.py --output outputs/frontier/profile_repeat.json
```

脚本会拒绝已有 JSON、NPZ 或 progress 文件。此次没有启动仿真、长回合或额外模型进程；已在完成后释放模型内存。
