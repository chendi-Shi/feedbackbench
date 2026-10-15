# 失败分析

本文是观察原始结果后撰写的诊断记录，不是预注册的因果检验。案例因出现失败而被选中，仅用于解释和展示；**不用于选择任务、初态、模型、调度阈值、消融配置、实验规模或停止规则**。原始失败保留在正式统计中；诊断重放不新增评估回合，也不能替换原始结果。单例不能代表失败发生率或不同方法的优劣。

## FA-001：抽屉伸出后，完整任务仍未完成，末段末端平移接近停滞

案例 ID：`minerva__libero_goal__3__0__clean__upstream`。

| 项目 | 原始记录 |
|---|---|
| 阶段 | `10_coverage`，每任务一次的覆盖诊断 |
| 任务 | `libero_goal` / task `3`：open the top drawer and put the bowl inside |
| 初态 / 种子 | `0` / `1000` |
| 模型 / 执行方式 | MINERVA，官方 `select_action`，每步推理及原有时间集成 |
| 条件 | `clean`；无自定义调度器 |
| 结局 | `success=false`；300 / 300 步；`terminated=false`，`truncated=true` |
| 推理次数 | 300，保存的重规划步恰为 `0…299` |

原始证据：[结果 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/result.json)、[轨迹 NPZ](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/trajectory.npz)、[初始前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/initial.png)、[最终前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/final.png)、[初始腕部相机](../outputs/campaign/10_coverage/episodes/minerva__libero_goal__3__0__clean__upstream/initial_wrist.png)。

### 直接观察

初始前相机中，碗在柜子左侧的桌面上，上层抽屉未呈现最终帧中的伸出状态。最终前相机中，上层抽屉明显向左伸出；碗仍可见于伸出抽屉的左侧，机器人夹爪也在这一区域。端点画面支持“抽屉发生了位移，但没有看到碗完成入屉”的描述，官方成功谓词确认完整任务失败。两张图不能确定过程中是否成功抓起过碗、何时脱手，或具体发生了哪一次接触。

轨迹保存了 300 个七维动作、300 个动作执行前的八维本体状态，以及单独的最终状态；不含碗位姿、抽屉关节状态、接触力或中间图像。动作与状态均为有限值，最终状态与 JSON 完全一致。按保存状态计算，最后 50 个动作对应的末端位置变化很小：

| 动作索引（从 0 开始，含两端） | 末端净位移 | 累计末端路径长度 | 平移命令范数中位数 |
|---|---:|---:|---:|
| 200–249 | 86.35 mm | 108.12 mm | 0.290 |
| 250–299 | 3.21 mm | 5.25 mm | 0.189 |

这里用 `states[t, :3]` 和下一时刻的位置构造位移，最后一段包含 `final_state[:3]`。路径长度为逐步欧氏距离之和；平移命令是 `actions[:, :3]` 的无量纲控制输入，不能当作米、速度或实际运动量。最后 50 步的 x/y/z 位置极差分别只有 1.16 / 3.17 / 1.69 mm，尽管平移命令仍非零。这支持“末段平移接近停滞”，不单独证明碰撞或控制器异常，也不表示姿态完全不再变化。

夹爪命令在步骤 120–195 持续为正，196–299 持续为负；两指关节坐标之差在步骤 175 约为 2.08 mm，在步骤 299 约为 70.86 mm。这个量描述夹爪状态变化，**不能作为成功抓持碗的证据**。

### 尚待验证的解释

接触约束或放置位置不合适，可能解释末段持续发出运动命令而实际位移很小。抓取未建立、搬运时失去碗，也可能导致终态未完成入屉。现有端点图和机器人状态无法区分这些机制，不能据此给出确定的“碰撞”“掉落”或“感知错误”标签。

本例采用每步重新推理的上游执行方式，没有启用自适应调度或人工观测延迟，因此不能据此认定某个调度规则造成或解决了失败。该仿真按控制步推进，回合墙钟时间也不能单独证明计算延迟导致失败。

### 后续诊断与证据绑定

已使用 [replay_episode.py](../scripts/replay_episode.py) 在独立目录按原始 300 个动作完成物理重放：[诊断录像](../outputs/replays/fa-001/replay.gif)、[重放审计](../outputs/replays/fa-001/replay_audit.json)。全部保存状态的最大绝对差为 **0.0**，初始两路相机、最终图像、步数、终止标志与成功标志均一致；原始证据哈希未改变。重放未调用策略，且不计入评估回合或方法选择。

抽查执行 60、120、180、240 步后的中间帧，可见抽屉已伸出，夹爪先后在碗附近下降、升起并再次下降；在第 180 和 240 步的可见画面中，碗仍位于伸出抽屉的左侧外部区域。中间图像补充了操作顺序，但遮挡和缺少物体位姿、接触记录仍限制了机制判断，不能把这一观察升级为确定的“抓取失败”或“搬运掉落”结论。

绑定本条目所读取的文件版本：

- `result.json` SHA-256：`b5451f4a604959f487e57abcad73e0569d7960dc368acbd4a30dd97ea1b65d3a`
- `trajectory.npz` SHA-256：`ce7226d5bf1385f493d73739c7e0da234f94db780538b938c44d12bab9623d15`
- 原始协议指纹：`271455baeac1396f967cf52d0cf209e28fe6a396f01dc05ed6c993dec43c4939`

## FA-002：入屉并关屉任务超时，端点遮挡限制了失败环节判断

案例 ID：`minerva__libero_10__3__0__clean__upstream`。

| 项目 | 原始记录 |
|---|---|
| 阶段 | `10_coverage`，每任务一次的覆盖诊断 |
| 任务 | `libero_10` / task `3`：put the black bowl in the bottom drawer of the cabinet and close it |
| 初态 / 种子 | `0` / `1000` |
| 模型 / 执行方式 | MINERVA，官方 `select_action`，每步推理及原有时间集成 |
| 条件 | `clean`；无自定义调度器 |
| 结局 | `success=false`；520 / 520 步；`terminated=false`，`truncated=true` |
| 推理次数 | 520，保存的重规划步恰为 `0…519` |

原始证据：[结果 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/result.json)、[轨迹 NPZ](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/trajectory.npz)、[初始前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/initial.png)、[最终前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/final.png)、[初始腕部相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__3__0__clean__upstream/initial_wrist.png)。

### 直接观察

初始画面中，下层抽屉已经打开，碗位于其右侧桌面。最终画面中，碗不再出现在原桌面位置；机械臂移到柜子前方，遮挡了部分抽屉和内部区域。仅凭最终帧，不能确认碗是否完整位于下抽屉内，也不能量化抽屉是否达到官方“关闭”判定阈值。原始结果只能确定复合任务未成功，**不能据此将失败确定归为“没放进去”或“没关上”中的某一项**。

轨迹包含 520 个七维动作和动作执行前的八维本体状态；动作、状态均为有限值，保存的最终状态与 JSON 完全一致。使用 FA-001 的相同计算方式，最后 50 个动作（470–519）对应的末端净平移为 **2.06 mm**、累计路径为 **2.15 mm**，x/y/z 位置极差为 0.35 / 1.05 / 1.75 mm；该窗口的平移命令范数中位数仍为 **0.792**。从动作 300 到 519，220 步的累计末端路径只有 **14.64 mm**。这说明后段末端平移接近停滞，不等于整个机械臂姿态或物体状态完全静止。

命令仍非零而末端平移很小，提供了进一步诊断的具体位置；由于未保存接触、物体位姿和抽屉关节状态，当前不对卡阻原因或具体失败环节作因果归因。该回合也没有启用自适应调度或人工延迟，不能作为这些方法的效果证据。

### 分析边界与证据绑定

本条目仅使用已完成回合的端点图像、JSON 和 NumPy 轨迹，没有运行模型、仿真或重放。案例属于事后诊断，继续遵守本文开头的选择隔离规则；不会据此修改任务、方法或阈值。若随后开展独立诊断重放，应先验证其与原始回合一致，再讨论中间操作过程，并明确排除在评估计数之外。

- `result.json` SHA-256：`5013d0f5bc7f2b881a9ecb41d9f68da203f605b61f99450e90eb12d55b04f37a`
- `trajectory.npz` SHA-256：`2fa913db88b68227d4362feaceec241b4e362c6945d6979aca79cf7d702b8e8b`
- 原始协议指纹：`271455baeac1396f967cf52d0cf209e28fe6a396f01dc05ed6c993dec43c4939`

## FA-003：杯与布丁的复合放置任务超时，末段仍有明显平移

案例 ID：`minerva__libero_10__6__0__clean__upstream`。

| 项目 | 原始记录 |
|---|---|
| 阶段 | `10_coverage`，每任务一次的覆盖诊断 |
| 任务 | `libero_10` / task `6`：put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 初态 / 种子 | `0` / `1000` |
| 模型 / 执行方式 | MINERVA，官方 `select_action`，每步推理及原有时间集成 |
| 条件 | `clean`；无自定义调度器 |
| 结局 | `success=false`；520 / 520 步；`terminated=false`，`truncated=true` |
| 推理次数 | 520，保存的重规划步恰为 `0…519` |

原始证据：[结果 JSON](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/result.json)、[轨迹 NPZ](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/trajectory.npz)、[初始前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/initial.png)、[最终前相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/final.png)、[初始腕部相机](../outputs/campaign/10_coverage/episodes/minerva__libero_10__6__0__clean__upstream/initial_wrist.png)。

### 直接观察

初始前相机中，白杯位于盘子的右后方，布丁盒位于盘子的左后方。最终前相机中，白杯与盘面区域重叠，布丁盒可见于盘子在图像中的左侧；机械臂仍处于桌面上方。这里的“左/右”仅描述图像位置，不等同于 BDDL 任务空间的相对关系判据。端点图像不能单独验证白杯的接触/支撑关系或布丁盒是否进入目标区域，也不能断定究竟哪一个子条件失败。可确定的是官方复合成功谓词在回合结束时仍为假。

轨迹保存了 520 个七维动作、520 个动作执行前的八维本体状态，以及最终状态；动作和状态均为有限值，最终状态与 JSON 完全一致。沿用 FA-001 的位置差分计算，取 `states[t, :3]` 到下一时刻的末端位移，最后一步使用 `final_state[:3]`：

| 动作索引（从 0 开始，含两端） | 末端净位移 | 累计末端路径长度 | 平移命令范数中位数 |
|---|---:|---:|---:|
| 300–349 | 15.56 mm | 25.80 mm | 0.071 |
| 420–469 | 107.78 mm | 163.67 mm | 0.208 |
| 470–519 | 102.90 mm | 105.67 mm | 0.228 |

末 50 步仍累计移动约 10.57 cm，因此本例**不支持“末段末端平移接近停滞”**的描述。平移命令范数仍是无量纲控制输入，不能直接解释为实际速度。夹爪命令在动作 277–519 持续为负；其间的机器人运动及夹爪坐标不能证明对布丁盒或白杯建立了成功抓持。

### 分析边界与证据绑定

当前证据提示，应在后续独立诊断中区分“最终几何/支撑关系未满足”与“操作仍在进行但已到时限”等解释；没有中间图像、物体位姿或接触记录，现阶段无法在这些解释之间作选择，也不能依据剩余运动推断延长时限就会成功。该回合没有自适应调度或人工延迟，不能作为这些方法的效果证据。

本条目只读取了原始 JSON、NPZ 与初始/最终图像，并进行轻量 NumPy 计算；未运行模型、仿真或重放。案例不改变正式实验任务、阈值或时长，失败保留在覆盖结果的分母中。

- `result.json` SHA-256：`104324ff6834ae0c4231e28e2cda4d21b62c0a18c890b0b89195b3acb3c2c33d`
- `trajectory.npz` SHA-256：`0a6689cf8f5b09e6f2be16f6063e6c2f0ba2c6cc0dd16ac72a9224ede2c5995f`
- 原始协议指纹：`271455baeac1396f967cf52d0cf209e28fe6a396f01dc05ed6c993dec43c4939`

## FA-004：缓存调用中 NFE 减少，但配对回合未完成任务

这是新 SmolVLA 缓存研究中首个完成的四方法配对案例：`libero_spatial` / task `1` / init `11`，任务为“pick up the black bowl next to the ramekin and place it on the plate”。案例在观察结果后选择用于解释预算口径；它不代表其余任务的总体效果，不改变后续回合或缓存阈值。

| 方法 | 成功 | 控制步 / 调用 | 总 NFE / 每调用平均 NFE | 缓存命中 | 原始证据 |
|---|---|---|---|---|---|
| full10 | 是 | 98 / 10 | 100 / 10.000 | 不适用 | [结果](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__full10/result.json)、[录像](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__full10/rollout.gif) |
| cache2 | 否 | 280 / 28 | 232 / 8.286 | 6 / 28 | [结果](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__cache2/result.json)、[录像](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__cache2/rollout.gif) |
| noise2 | 是 | 100 / 10 | 20 / 2.000 | 不适用 | [结果](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__noise2/result.json)、[录像](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__noise2/rollout.gif) |
| cache0 | 否 | 280 / 28 | 230 / 8.214 | 5 / 28 | [结果](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__cache0/result.json)、[录像](../outputs/frontier/cache_study/episodes/smolvla__libero_spatial__1__11__clean__cache0/rollout.gif) |

所有方法均每 10 步重规划、关闭时间集成，使用相同初态和按决策序号配对的噪声。两个缓存条件均正常跑满 280 步，无运行异常；失败保留在测试分母中。**较低的每调用 NFE 没有转化为较低的整回合计算总量**：cache2 的总 NFE 是该成功 full10 回合的 2.32 倍，因为失败回合持续调用到完整时限。这不能称为保持任务质量的加速。

cache2 的六次命中分别使用索引 0–5，cache0 的五次命中分别使用索引 0–4，均来自本任务成功开发回合。当前单例不支持“反复检索同一条目造成卡住”的解释，也不能仅凭同任务来源认定检索动作适合另一初态。其余调用均回退完整 10 步；回退机制存在，不等于一定能恢复已经偏离的闭环行为。

最终图像中，full10 的碗位于盘子区域，官方成功谓词为真；两个缓存条件的盘子仍为空，夹爪位于画面左缘附近。物体存在遮挡，不能据此确定抓取是否建立、是否掉落或哪一次接触失败。cache2/cache0 最后 50 步的末端累计平移分别约 **339.92 mm / 303.37 mm**，计算方法与 FA-001 相同，因此本例也不支持“末端完全停滞”的描述。

这组结果只提示需要区分检索键相似、动作可用性、局部计算节省和完整任务结果。具体失败机制仍需额外受控实验；不能把单例推广为 ActionCache 原论文或所有动作缓存方法无效。完整结果以全部预声明配对的[正式报告](../outputs/frontier/cache_study/report.md)为准。

新协议指纹：`aa7226e19feef89f972207d22fc0e087b8d9e2bbd44c3761103f73e9ae821bc5`。cache2 / cache0 的原始结果 SHA-256 分别为 `ca453232438936b956d816e97dbc772af29e02b49273789b7cd53e5c860bba1d`、`aa10a6b22047b1eb493c2d0a012b3fdabac1e52448b095740d8e0216e3417e94`；每回合 receipt 还绑定轨迹、图像、录像与逐调用日志。
