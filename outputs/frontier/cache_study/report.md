# SmolVLA frozen offline ActionCache transfer pilot v1 — 完整试验记录

Can successful development latents reduce flow NFE on held-out initial states without losing task success?

已核验 28/28 个计划回合；缺失 0、未封存 0、错误 0。开发集与测试集分开；主比较与消融的重叠回合不重复相加。

所有完整回合均核对协议与源码指纹、receipt 文件哈希、模型权重来源、调用日志及轨迹。配对核对初始图像/机器人状态和首决策噪声，并检查共有决策序号的噪声一致性。

## 方法与范围

- `full10`：当前观测条件下从纯噪声完整采样10步。
- `noise2`：当前观测条件下从纯噪声采样2步。
- `cache2`：命中开发集 t=.2 的32维中间潜变量后沿原网格细化2步；未命中完整10步。
- `cache0`：命中开发集 t=0 的7维最终归一化动作后直接反归一化；未命中完整10步。

缓存阈值 **0.95**，预先固定、未经调优。只接纳成功开发回合，测试期冻结；这与 [ActionCache 论文](https://arxiv.org/abs/2607.06370v2) 的在线成功回合 LRU 缓存不同。

## 成功与采样预算

成功数以已完成回合计，旁列计划分母；部分数据不能代表整个计划成功率。

| 范围 / 方法 | 完成 / 计划 | 任务数 | 成功 / 已完成 | NFE / 规划 | hit / query | 完整 / 细化 / 回退 / 直接缓存 |
|---|---:|---:|---:|---:|---:|---:|
| development / full10 | 4/4 | 4/4 | 3/4 | 10.000 | 0/0 | 82 / 0 / 0 / 0 |
| primary / full10 | 8/8 | 4/4 | 7/8 | 10.000 | 0/0 | 145 / 0 / 0 / 0 |
| primary / cache2 | 8/8 | 4/4 | 6/8 | 8.871 | 23/163 | 140 / 23 / 140 / 0 |
| ablation_init11 / full10 | 4/4 | 4/4 | 4/4 | 10.000 | 0/0 | 57 / 0 / 0 / 0 |
| ablation_init11 / noise2 | 4/4 | 4/4 | 4/4 | 2.000 | 0/0 | 0 / 0 / 0 / 0 |
| ablation_init11 / cache2 | 4/4 | 4/4 | 3/4 | 8.162 | 17/74 | 57 / 17 / 57 / 0 |
| ablation_init11 / cache0 | 4/4 | 4/4 | 3/4 | 7.838 | 16/74 | 58 / 0 / 58 / 16 |

回退次数是完整10步调用的子集，不能与完整次数相加。noise2 计数另存 report.json。

![Primary success and NFE](figures/cache_primary_success_nfe.png)

![Common-init ablations](figures/cache_ablation_success_nfe.png)

## 配对比较

以下均为 candidate − baseline。成功差先在每个任务内平均，再等权平均任务；按任务簇重抽样 2000 次，seed=20261014。仅4个计划任务簇，区间不稳定；零宽区间仅说明观察到的任务效应无变异，不能证明等价、非劣或真实不确定性为零。

| 范围 / 对照 | 配对完成 / 计划 | 任务簇 | 成功差 pp [95% CI] | 每调用均值差秒 [95% CI] | 双成功对数 / wall差秒 |
|---|---:|---:|---:|---:|---:|
| primary / cache2 − full10 | 8/8 | 4 | -12.50 [-37.50, +0.00] | -1.378 [-4.162, +0.338] | 6 / -24.182 |
| ablation_init11 / cache2 − full10 | 4/4 | 4 | -25.00 [-75.00, +0.00] | -2.927 [-5.543, -0.688] | 3 / -59.032 |
| ablation_init11 / noise2 − full10 | 4/4 | 4 | 0.00 [+0.00, +0.00] | -3.410 [-5.770, +0.120] | 4 / -50.160 |
| ablation_init11 / cache0 − full10 | 4/4 | 4 | -25.00 [-75.00, +0.00] | -2.375 [-4.752, +0.160] | 3 / -51.680 |
| ablation_init11 / cache2 − noise2 | 4/4 | 4 | -25.00 [-75.00, +0.00] | 0.483 [-3.450, +4.415] | 3 / -16.226 |
| ablation_init11 / cache2 − cache0 | 4/4 | 4 | 0.00 [+0.00, +0.00] | -0.552 [-1.357, +0.621] | 3 / -7.352 |

配对延迟比较的是各回合的平均调用耗时；首决策之后轨迹/观测可能不同，不能解释为相同输入的纯算法计时。wall差只列共同成功回合，仍是受结果选择影响的参考量。所有回合原始wall在 episodes.csv；失败更早结束不是加速。

## 分项时延

每格为 p50 / p95 秒；调用池分位数为描述性统计，较长回合贡献更多调用。VLM 总项包含 vision-prefix 和 transformer，不能将三列相加。end-to-end 包含最近一次观测预处理；amortized 另计上次规划以来所有非规划 tick 的预处理，详见 JSON/CSV。

| 范围 / 方法 | calls | VLM总项 | vision-prefix | transformer | action head | lookup | end-to-end |
|---|---:|---:|---:|---:|---:|---:|---:|
| development / full10 | 82 | 7.735 / 12.767 | 6.238 / 10.474 | 1.426 / 2.360 | 3.915 / 6.373 | 0.000 / 0.000 | 11.602 / 19.174 |
| primary / full10 | 145 | 9.630 / 13.275 | 8.011 / 10.972 | 1.757 / 2.412 | 4.362 / 6.596 | 0.000 / 0.000 | 13.721 / 20.118 |
| primary / cache2 | 163 | 10.351 / 13.085 | 8.523 / 10.741 | 1.878 / 2.525 | 4.640 / 6.318 | 0.015 / 0.025 | 14.716 / 19.262 |
| ablation_init11 / full10 | 57 | 9.303 / 13.721 | 7.726 / 11.223 | 1.764 / 2.517 | 4.797 / 6.821 | 0.000 / 0.000 | 13.721 / 20.432 |
| ablation_init11 / noise2 | 57 | 10.833 / 13.264 | 8.677 / 10.865 | 1.847 / 2.513 | 0.996 / 1.354 | 0.000 / 0.000 | 11.865 / 14.520 |
| ablation_init11 / cache2 | 74 | 8.479 / 12.514 | 6.957 / 10.440 | 1.600 / 2.388 | 3.394 / 5.804 | 0.013 / 0.024 | 11.736 / 18.460 |
| ablation_init11 / cache0 | 74 | 8.934 / 13.429 | 7.366 / 11.074 | 1.670 / 2.425 | 4.019 / 6.304 | 0.012 / 0.024 | 13.133 / 19.433 |

![Timing components](figures/cache_latency_components.png)

## 行为差与限制

pairs.csv 列出双方都有完整前10步时的7维逐通道 RMSE/MAE，以及混合坐标的距离。任何一方少于10步时，距离记为缺失，保留可用步数和原因，不纳入均值；该回合的成功结果仍计入主分析。这是已执行动作的行为距离，不是质量或成功的替代指标；后续观测可能已分叉。

| 范围 / 对照 | first10有效 / 已完成配对 | 前10步混合坐标 RMSE 的回合均值 |
|---|---:|---:|
| primary / cache2 − full10 | 8/8 | 0.006516 |
| ablation_init11 / cache2 − full10 | 4/4 | 0.005802 |
| ablation_init11 / noise2 − full10 | 4/4 | 0.035454 |
| ablation_init11 / cache0 − full10 | 4/4 | 0.007353 |
| ablation_init11 / cache2 − noise2 | 4/4 | 0.037119 |
| ablation_init11 / cache2 − cache0 | 4/4 | 0.001662 |

这是少量已知预训练任务上的初始状态泛化试验。CPU 同步仿真在推理时暂停；不支持真实实时部署、SOTA 或新任务泛化声明。

## 可复核证据

[协议](protocol.json) · [完整机器可读报告](report.json) · [方法汇总](groups.csv) · [所有回合](episodes.csv) · [逐次调用](calls.csv) · [配对结果与动作差](pairs.csv)

Protocol SHA fingerprint: `aa7226e19feef89f972207d22fc0e087b8d9e2bbd44c3761103f73e9ae821bc5`。模型 revision、权重 SHA、参数量与运行时版本取自原始记录和协议，见 report.json。

- `smolvla`：604934176 参数，cpu / float32；revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，权重 SHA256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。
- `smolvla`：604934176 参数，cpu / float32；revision `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`，权重 SHA256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`。
