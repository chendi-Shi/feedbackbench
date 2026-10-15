# GitHub 发布快照

仓库：<https://github.com/chendi-Shi/feedbackbench>。首次发布设为私有；GitHub Actions 只支持手动触发，推送不会自动启动云端计算。

本次 Git 快照包含当前源码、说明、测试、冻结协议与选定实验原始证据。旧 MINERVA 232 回合、SmolVLA 缓存研究 28 回合、前瞻确认 16 回合分别保存。VLA-JEPA 动态 INT8 单次真实前向已通过；其新闭环计划 4 回合、实际执行 0 回合，状态为 `blocked_preflight`。测试用例不是机器人评测回合。

发布内容以 [PUBLICATION_MANIFEST.json](../PUBLICATION_MANIFEST.json) 为准。该清单记录每个发布文件的 SHA-256、字节数和原始来源哈希；自身不计入其文件清单。诊断副本中的个人路径与主机名经过脱敏，原始证据中的哈希仍指向原始字节，清单同时提供导出字节身份。正式协议、轨迹、动作、结果与冻结源码保持原样。

历史 continuation ZIP 的清单保留在 [publication/continuation-bundle-manifest.json](../publication/continuation-bundle-manifest.json)，只描述原 ZIP，不能用来校验更新后的整个 Git 仓库。原 ZIP SHA-256 为 `7d440c6cdf18b8ba546d30f177aa71f2bc40f0f3c641e6ec40fdc80bfc927858`。本次发布在其已核验内容上加入最新闭环执行器、协议、测试和内存阻断记录。历史检查点报告中的相对链接按其[原始上下文规则](../outputs/checkpoints/before_smol/README.md)解析。

模型权重、虚拟环境、第三方源码、完整仿真资产、私有运行目录和大型 ZIP 不进入 Git。仓库中保留的小型模型/资产 manifest 记录下载身份；按 README 中的固定版本入口恢复运行环境。上游归属见 [THIRD_PARTY](THIRD_PARTY.md)。本项目尚未指定自身的开源许可证，私有发布不改变上游许可。

本机发布使用 `deliverables/github/feedbackbench` 中的独立 Git 工作区，避免混入父目录的其他项目。研究运行目录的 `.gitignore` 继续排除自动生成的大目录；发布时只将已经核验、列入清单的文件精确加入 Git。不要在研究目录对整个输出目录使用强制添加，也不要向父目录的 NLP 仓库推送。

`.gitattributes` 禁用自动换行转换，使 Git 保存和检出冻结源文件与证据的原始字节。复现实验请使用新的输出路径，保留仓库中的历史证据。
