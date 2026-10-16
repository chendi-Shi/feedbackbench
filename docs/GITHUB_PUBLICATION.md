# GitHub 发布快照

仓库：<https://github.com/chendi-Shi/feedbackbench>。首次发布设为私有；GitHub Actions 只支持手动触发，推送不会自动启动云端计算。

Git 快照按发布清单保存源码、说明、测试、冻结协议与选定实验原始证据。旧 MINERVA 232 回合、SmolVLA 缓存研究 28 回合、前瞻确认 16 回合分别保存。VLA-JEPA 的原始、分页、compact、独立进程和单长任务阶段各自保存预算及结果，不合并成功率。测试用例和保存动作重放不是机器人评测回合。

历史提交 `a1111bd6c8c95e85a9b4c1b175bd0875caec1370` 保存了前台收尾入口、文件占用恢复测试、资源阻断和[当时的闭环审计](../artifacts/closure-audit-20261016.json)。提交 `24dbd140b742fff493cd320c6099221bbb04dc61` 保存三个真实闭环任务、长任务中止与首次恢复阻断；`7ae576aacb17af2cf712ce5be19f9edc19032706` 保存第二次恢复预检及600秒等待。本次增量保存后续资源准备、其余预检阻断与第六次启动的520步完整恢复，官方任务判定失败。实际版本由本文件所属Git提交确定，清单中的`previous_commit`指向更新前的基线；旧提交及冻结内容保留。

- [分页实验](VLAJEPA_PAGED_RESULTS.md)：真实单观测动作与原INT8逐值相同；首次闭环的系统可用内存保护中止也保留。
- [arena重放](LIBERO_ARENA_REPLAY_RESULTS.md)：978个保存动作的状态、指定图像和终止标志一致；无新模型推理，未证实物理RSS收益。
- [compact](VLAJEPA_COMPACT_RESULTS.md)及[独立进程](VLAJEPA_ISOLATED_RESULTS.md)：各计划4、完成成功3、内存中止未封存1；48次成对调用及327步轨迹有[精确一致性审计](../artifacts/vlajepa-process-isolation-consistency-20261016.json)。
- [单长任务资源实验](VLAJEPA_LONG_HORIZON_RESULTS.md)：6GiB自身RSS预算下保存398步与57次完整推理，系统可用内存保护中止，未封存；[恢复验证](VLAJEPA_RECOVERY_RESULTS.md)另行记录，不能回填旧阶段或形成同协议4/4成绩。
- [恢复执行](VLAJEPA_RECOVERY_RESULTS.md)：25个冻结源、130个原证据文件绑定；前五次启动均在预检阻断，第六次完成520步，`success=false`、`truncated=true`。392步保存动作重放＋6步边界重叠核对＋122步新执行；实际19次模型前向、76 NFE。原严格核验退出0，[完成审计](../artifacts/vlajepa-recovery-completion-audit-20261016.json)与完整录像同时保存。新增独立评测样本为0，原398步中止目录保持不变。
- [额外资源管理](VLAJEPA_RESOURCE_RELEASE_RESULTS.md)：停止闲置Ollama、回收已识别桌面进程工作集，并在第六次启动中对本次模型解释器执行一次工作集回收。准备、controller计划/源码/实际记录分别保存；这些外部操作不属于原25源指纹。模型和内存门槛未改，但不能据此声称原资源流程无干预复现、稳定内存收益或性能加速。
- 分页、arena/compact、进程隔离及单长任务的历史测试分别记录。恢复75项、发布校验器39项及两组联合重跑114项均通过；联合记录包含前两组，不重复相加，也不把测试批次计作模型或任务样本。本次执行实际恢复、严格验证及发布完整性检查，没有把历史测试写成本轮重跑。

发布内容以 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json) 为准。该清单记录每个发布文件的 SHA-256、字节数和原始来源哈希；自身不计入其文件清单。诊断副本中的个人路径与主机名经过脱敏，原始证据中的哈希仍指向原始字节，清单同时提供导出字节身份。正式协议、轨迹、动作、结果与冻结源码保持原样。

可用 `python scripts/verify_publication.py` 在下载的仓库中只读核验发布快照。它先核对每个导出文件的实际字节与哈希，再检查冻结源、回合receipt和独立进程账本；运行诊断的兼容性补丁文本若含本机路径，只允许清单明确标注的白名单字段脱敏。worker和历史审计保留原件哈希，校验器会把这类检查标为`source_binding_via_redaction_manifest`，与实际原始字节一致的检查区分。清单中的原件哈希是发布方声明的来源身份，不能用经过脱敏的字节重新推导或证明它。该发布完整性核验不启动模型，不替代原始数值轨迹审计，也不会把未封存实验改为完整。

历史 continuation ZIP 的清单保留在 [publication/continuation-bundle-manifest.json](../publication/continuation-bundle-manifest.json)，只描述原 ZIP，不能用来校验更新后的整个 Git 仓库。原 ZIP SHA-256 为 `7d440c6cdf18b8ba546d30f177aa71f2bc40f0f3c641e6ec40fdc80bfc927858`。历史检查点报告中的相对链接按其[原始上下文规则](../outputs/checkpoints/before_smol/README.md)解析。

模型权重、虚拟环境、完整第三方源码树、完整仿真资产、私有运行目录和大型 ZIP 不进入 Git。arena 重放源身份审计单独保留固定 LeRobot 的未修改 `envs/libero.py` 与完整 LICENSE；其余第三方依赖仍由固定版本入口恢复。仓库中的小型模型/资产 manifest 记录下载身份，上游归属见 [THIRD_PARTY](THIRD_PARTY.md)。本项目尚未指定自身的开源许可证，私有发布不改变上游许可。

本机发布使用 `deliverables/github/feedbackbench` 中的独立 Git 工作区，避免混入父目录的其他项目。研究运行目录的 `.gitignore` 继续排除自动生成的大目录；发布时只将已经核验、列入清单的文件精确加入 Git。不要在研究目录对整个输出目录使用强制添加，也不要向父目录的 NLP 仓库推送。

`.gitattributes` 禁用自动换行转换，使 Git 保存和检出冻结源文件与证据的原始字节。复现实验请使用新的输出路径，保留仓库中的历史证据。
