# 40 任务覆盖诊断：已完成结果

`10_coverage` 已完成 **40 / 40 个预声明回合，37 次成功、3 次失败，0 个错误、0 个缺失**。这里评估的是冻结的 MINERVA 公开检查点及上游逐步执行方式；尚不包含自适应调度或方法收益结论。

证据入口：[完整阶段报告](../outputs/campaign/10_coverage/report.html)、[统计 JSON](../outputs/campaign/10_coverage/summary.json)、[冻结协议](../outputs/campaign/10_coverage/protocol.json)。下表从 40 份原始 `result.json` 逐项汇总，与阶段统计一致。

| 任务集 | 任务数 / 回合数 | 成功 / 回合 | 观察成功率 | 累计控制步 / 推理次数 |
|---|---:|---:|---:|---:|
| LIBERO-Spatial | 10 / 10 | 10 / 10 | 100% | 999 / 999 |
| LIBERO-Object | 10 / 10 | 10 / 10 | 100% | 1,294 / 1,294 |
| LIBERO-Goal | 10 / 10 | 9 / 10 | 90% | 1,214 / 1,214 |
| LIBERO-10 | 10 / 10 | 8 / 10 | 80% | 3,056 / 3,056 |
| 合计 | 40 / 40 | 37 / 40 | 92.5% | 6,563 / 6,563 |

每个任务恰好一次，因此任务宏平均与回合成功率均为 **92.5%**。所有回合使用初态编号 `0`、种子 `1000`、`clean` 观测，以及上游 `select_action` 的每步推理和系数 `0.01` 的时间集成。成功取官方任务谓词；失败按各任务集的完整时长截断，没有用动作距离、奖励或视觉印象替代成功判定。任务编号指本次固定 benchmark 排序，应连同任务文本核对。

## 三个失败回合

| 任务集 / ID | 官方任务文本 | 结局及证据 |
|---|---|---|
| `libero_goal` / `3` | open the top drawer and put the bowl inside | 300 / 300 步，超时未成功；[原始 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/result.json)、[最终图像](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/final.png) |
| `libero_10` / `3` | put the black bowl in the bottom drawer of the cabinet and close it | 520 / 520 步，超时未成功；[原始 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/result.json)、[最终图像](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/final.png) |
| `libero_10` / `6` | put the white mug on the plate and put the chocolate pudding to the right of the plate | 520 / 520 步，超时未成功；[原始 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/result.json)、[最终图像](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/final.png) |

三次失败都保留在分母中。端点图像和机器人轨迹的事后诊断见 [FAILURE_ANALYSIS](FAILURE_ANALYSIS.md) 的 FA-001 至 FA-003；这些分析不参与阈值选择或正式实验配置变更。

## 这些结果支持什么

结果证明本机已完成四个任务集的真实闭环覆盖，并在该固定检查点、该初态和执行方式下观察到上述成功率。它不能估计每个任务跨初态的稳定性，也不代表公开 benchmark 的完整复现分数。MINERVA 的 40 项任务映射来自检查点，这里评测的是已知任务；每任务一条回合不能支持未见语言、未见任务或真机泛化结论。

本阶段只有一种执行方式，不能据此说自适应调度减少了计算或提高了成功率。后续开发集选参、保留测试、消融与延迟对照使用各自冻结的协议与独立报告。本阶段也不与集成 smoke 相加：其中一条初态与 smoke 重复，不能计作额外独立证据。

每回合平均执行 **164.075 步**、调用模型 **164.075 次**，即每百步 **100 次推理**。总回合墙钟为 **3,414.21 秒（56.90 分钟）**，不含模型加载与 GIF 编码；覆盖运行期间后台模型服务负载发生变化，因此这些墙钟仅是本机运行记录，不能据此计算受控吞吐或方法加速比。

结果绑定的协议指纹为 `271455baeac1396f967cf52d0cf209e28fe6a396f01dc05ed6c993dec43c4939`。阶段 [summary.json](../outputs/campaign/10_coverage/summary.json) 内保留协议及所有原始结果文件的 SHA-256，便于核对分母与数据来源。
