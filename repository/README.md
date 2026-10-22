# 本地外部资源

`repository/` 用于准备实际服务测试的本地资源。源码交付保留使用说明，模型 tokenizer 和外部语料由运行环境提供，并通过 YAML 配置实际路径。

按实际使用的模型准备所需目录，再将 Run 的 `normalization.settings.tokenizer_path` 或离线命令的模型路径指向该本地目录。普通测试自行生成小型 tokenizer fixture；真实数据与服务验收另有资源前置条件。

正式测试应记录资源来源、revision 与文件哈希，保证冻结输入可复核；按实际模型准备所需资源即可。

外部语料说明见 [datasets](datasets/README.md)。Docker 可在构建时包含已准备的本地资源，或在运行时挂载；部署见 [使用指南](../docs/user-guide.md)，源码交付边界见 [开发指南](../docs/development-guide.md)。
