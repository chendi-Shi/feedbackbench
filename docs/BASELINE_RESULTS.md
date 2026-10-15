# MINERVA 基线：232 个已完成回合

截至 **2026-10-14**，旧 campaign 的四个 MINERVA 阶段均已完整完成：**40 + 32 + 128 + 32 = 232 回合，阶段内 0 错误、0 缺失**。原定 248 回合中的 SmolVLA 调度 12 回合与 flow-step 4 回合暂停且未尝试。新 ActionCache–SmolVLA 28 回合有独立协议；任何新结果均不能补进旧 248 的分母，也不把 MINERVA、SmolVLA 或 VLA-JEPA 混成同一模型的成功率。

本页保留基线的负结果：**开发集选中的 adaptive 没有在 clean 主比较中保持 fixed8 的观察成功率，调用数反而增加；delay4 下也未建立优于 fixed8 的证据。** 这是固定模型、小任务集和有限公开初态上的结果，不是对所有自适应反馈方法或 VLA 模型的普遍结论。

## 完成范围与证据入口

| 阶段 | 实际完成 / 计划 | 模型与样本结构 | 结果入口 |
|---|---:|---|---|
| 覆盖诊断 | 40 / 40 | MINERVA，4 suites × 10 任务 × 1 初态，原上游调用路径 | [报告](../outputs/campaign/10_coverage/report.html)、[JSON](../outputs/campaign/10_coverage/summary.json)、[人工解读](COVERAGE_RESULTS.md) |
| 开发选择 | 32 / 32 | MINERVA，4 个 task0 × 2 初态 × 4 方法 | [报告](../outputs/campaign/20_development/report.html)、[JSON](../outputs/campaign/20_development/summary.json)、[选择记录](../outputs/campaign/development_selection.json) |
| clean 主比较与消融 | 128 / 128 | MINERVA，8 任务 × 2 初态 × 8 方法 | [报告](../outputs/campaign/30_methods/report.html)、[JSON](../outputs/campaign/30_methods/summary.json)、[完整方法解读](METHOD_RESULTS.md) |
| delay4 对照 | 32 / 32 | MINERVA，相同 8 任务 × 2 初态 × 2 方法 | [阶段报告](../outputs/campaign/40_delay/report.html)、[JSON](../outputs/campaign/40_delay/summary.json)、[配对 clean–delay 报告](../outputs/campaign/delay_comparison.html) |
| 原 SmolVLA 调度 | **0 / 12** | 计划 4 任务 × 1 初态 × 3 方法 | [保留计划](../outputs/campaign/frozen_plans/50_smolvla.json)；未尝试 |
| 原 SmolVLA flow-step | **0 / 4** | 计划 4 任务 × 1 初态 × 1 新条件 | [保留计划](../outputs/campaign/frozen_plans/60_flow_steps.json)；未尝试 |

阶段承担不同用途，且任务反复出现；232 既不是 232 个独立任务，也不应汇总成一个代表模型能力的总成功率。[总报告](../outputs/campaign/report.html) 和 [转向记录](../outputs/campaign/TRANSITION.md) 保留完整来龙去脉。

## 覆盖与开发选择

覆盖诊断为 **37/40**：spatial 10/10、object 10/10、goal 9/10、libero_10 8/10。每任务仅一个公开初态，证明的是这 40 个已知任务的运行与有限覆盖，不是完整公开 benchmark 的复现成功率；失败个例和原图/轨迹链接见 [覆盖解读](COVERAGE_RESULTS.md)。

开发阶段每方法 8 回合：fixed8 和 adaptive_low 均 **8/8**，adaptive_medium **7/8**，adaptive_high **6/8**。仅 low 达到参照成功率，因此按预声明规则被选中；其平均调用 **24.625** 次，高于 fixed8 的 **18.500** 次（+33.1%），不是改进证据。视觉/夹爪阈值 **0.015 / 0.001** 在后续测试前冻结；选择记录注明 `test_results_read=false`。详见 [开发结果](DEVELOPMENT_RESULTS.md)。

## clean 主比较：未支持预期改进

各方法为相同 **8 任务 × 初态 11、29 = 16 回合**；主比较方向为 adaptive 减 fixed8。

| 指标 | fixed8 | adaptive |
|---|---:|---:|
| 成功 / 回合 | 16/16 | 15/16 |
| 任务宏成功率 | 100% | 93.75% |
| 平均调用 / 回合 | 18.9375 | 36.5625 |
| 调用 / 百控制步 | 12.8227 | 22.6393 |

成功率差 **−6.25 个百分点**，任务簇 95% bootstrap 区间 **[−18.75, 0.00]**；16 对中 adaptive 为 **0 胜、1 负、15 平**。平均调用增加 **93.07%**，百步调用也更高，不能仅归因于失败回合运行更久。区间包含零，不据此宣称普遍劣效，但这组数据没有支持“保持成功并减少推理”的研究假设。

其余七条预声明对 fixed8 比较和三组件直接消融完整保留在报告中。fixed15 在这 16 回合也为 16/16 且平均调用 10.1875，但不因此事后更换主方案、宣称最优或证明等价。固定重规划频率还与活动时间集成历史数量耦合，不能解释成纯反馈频率的独立因果效应；详见 [方法结果](METHOD_RESULTS.md)。

## delay4：观察延迟明显损害成功率

`delay4` 将**两路相机和本体状态整个观测包**延迟 4 个仿真控制步，队列起始重复初始观测。20 Hz 下对应规定的 0.2 秒观测滞后；仿真仍在推理期间暂停，因此不是 CPU 推理阻塞、网络传输或异步真机实时延迟的复现。

| delay4 指标 | fixed8 | adaptive |
|---|---:|---:|
| 成功 / 回合 | 7/16 | 8/16 |
| 任务宏成功率 | 43.75% | 50.00% |
| 平均调用 / 回合 | 35.5625 | 69.5625 |
| 调用 / 百控制步 | 12.5857 | 24.2484 |

延迟条件内 adaptive 减 fixed8 为 **+6.25 个百分点**，任务簇 95% 区间 **[−18.75, +37.50]**，配对 **2 胜、1 负、13 平**；平均调用增加 **95.61%**。区间很宽且跨零，不能据一例净胜宣称 adaptive 更稳健。

相同方法、相同任务与初态的 delay4 减 clean 比较如下；这些配对复用已完成主阶段，不额外增加回合数。

| 方法 | clean → delay4 成功 | 成功率差 / 百分点 | 任务簇 95% 区间 / 百分点 | 平均调用变化 |
|---|---|---:|---|---:|
| fixed8 | 16/16 → 7/16 | −56.25 | [−75.00, −37.50] | +16.625 |
| adaptive | 15/16 → 8/16 | −43.75 | [−50.00, −31.25] | +33.000 |

两者均受到延迟损害。不同下降幅度不能单独作为已验证的方法×延迟交互效应，亦不能掩盖 adaptive 调用成本增加。配对数据见 [delay_comparison.json](../outputs/campaign/delay_comparison.json)。所有区间使用 **2,000 次任务簇 percentile bootstrap，seed 20261014**，重采样单位为 8 个任务，每任务保留两个初态；它没有覆盖重新训练随机性或更广泛任务分布。共享 CPU 上的墙钟与模型时间只作描述，不能用计时波动替代调用预算结论。

## 原始证据保全与暂停状态

本页编写时逐文件复核 [resume_audit.json](../outputs/campaign/resume_audit.json)：**1,174 个原始 episode 文件（139,325,502 字节）和 9 个冻结源码 SHA-256 全部匹配**。原始 `result.json`、轨迹、图像与视频没有因新研究而删改或重试。展示报告可从这些原始输入重新生成；报告文件本身的重建不应混同于回合证据变更。

原进程在完整阶段边界退出，见 [stop_receipt.json](../outputs/checkpoints/before_smol/stop_receipt.json)。保留的 `campaign_status.json` 是退出前快照，其中 `running` 不代表仍有进程或已经执行 SmolVLA。审计与当前目录都未发现 `50_smolvla` / `60_flow_steps` 回合尝试；本结论以保留证据和 runner 不删除进度的约束为基础，不将单次目录检查描述为证明历史上任何外部删除都不可能。

关键结果文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| 主比较 summary.json | `934092104b2dbf442399e63e533fadfce548e61f762d8a9a0c329e2364123577` |
| 延迟 summary.json | `85bd4d710b1c91f2887499551df25d490904ae4f8d8dfba4cdb210c0cf4473e9` |
| delay_comparison.json | `343e4c505d07d24d019149af01323a605de4687c6b291d3c1bbb2c0cd7c4c43e` |
| resume_audit.json | `d4fbc140427dac96deee46e2c822984117c379a9310e9f4254d699c6d29fc0bc` |

新研究见 [ActionCache 冻结协议](CACHE_PROTOCOL.md)；真实 SmolVLA 单观测计时见 [CPU profile](FRONTIER_CPU_PROFILE.md)。它们是新的证据链，不改变本页的 MINERVA 基线结论和旧计划暂停状态。
