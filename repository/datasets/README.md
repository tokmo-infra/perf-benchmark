# 本地语料资源

此目录用于本地维护外部语料，源码交付保留使用说明。数据集正文由运行环境提供，普通测试使用仓库内的小型夹具。

数据集维护者记录来源、版本或 revision、文件校验值及字段结构，并为对照测试冻结同一批输入。JSONL 和 Parquet 的加载路径通过 `loader.settings.data_files` 指定；tokenizer 通过 `normalization.settings.tokenizer_path` 指向与被测模型匹配的本地目录。

加载和离线生成按当前 [使用指南](../../docs/user-guide.md)配置实际数据路径与支持的格式。文件管理边界见 [本地资源说明](../README.md)和 [开发指南](../../docs/development-guide.md)。
