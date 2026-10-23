# perf-benchmark

LLM 性能测试工具，支持五种 HTTP 协议、单轮与多轮、闭环与开环、并发扫描与 SLO 搜索。
八阶段 Pipeline 将数据准备、请求执行、指标计算、SQLite 持久化和报告导出分离。
实际服务 usage 生成 maxtpm、缓存命中和收入；缺失值保留未知。
maxTPM 从首遍完成 40% 轮次时开始；实际请求并发连续 60 秒低于预设并发时，以这段下降的起点为终点，按区间内完整分钟统计。

- [使用与部署指南](docs/user-guide.md)：安装、YAML、并发/用户池、数据、TPM/收入、CSV/Excel、Docker 与观测。
- [静态 Prefill & Decode 压测指南](docs/static-benchmark-guide.md)：分词表随机抽取、输入长度与并发矩阵、复杂系统编程长文本、思考档位与 Excel 导出。
- [开发指南](docs/development-guide.md)：模块职责、插件与 Hook、计时及存储契约、测试和代码提交范围。
- 代表性示例：[静态 Prefill 与 Decode 压测](examples/static-benchmark.yaml)使用本地 tokenizer 生成输入；[静态固定并发](examples/source-only-openai.yaml)、[动态并发扫描与 SLO](examples/concurrency-sweep.yaml)、[多轮对话](examples/multi-turn.yaml)共用小样本，只用于检查请求流程。

```bash
uv sync --locked --extra observation
uv run perf-benchmark --help
```

源码交付包含 Python/Rust 实现、前端源码、构建配置、核心测试、使用文档与代表性 YAML 示例。数据集、模型及 tokenizer 由使用者按实际服务准备，结果和构建产物保存在运行环境。运行需要 Python 3.11+、uv，以及 `perf_benchmark/rust-toolchain.toml` 指定的 Rust 工具链；首次安装会编译原生扩展。离线机器使用预备的依赖包、Cargo 依赖、工具链、本地数据和 tokenizer。

源码包保留核心 Python 回归测试及必要夹具，覆盖数据准备、分词、多轮/SWE、请求执行、SLO、指标计费、持久化、静态矩阵与导出；另保留 Rust 和前端构建验证所需的测试源码。完整开发仓库保存大型性能实验、跨工具对比及真实平台专项测试；`.gitattributes` 的 `export-ignore` 定义 `git archive` 的交付范围。安装和运行步骤见下文。

先准备与被测模型相同的 tokenizer 目录，复制 [静态并发配置](examples/source-only-openai.yaml)，修改 `tokenizer_path`、`base_url`、`model` 和结果路径，然后执行：

```bash
uv run perf-benchmark run examples/source-only-openai.yaml
```

逐记录数据准备默认使用 4 个进程构造请求；需要分词时，批量分词另使用 4 个原生 CPU 线程。分别通过 `PERF_BENCHMARK_PREPARATION_WORKERS` 和 `RAYON_NUM_THREADS` 调整，详见使用指南的“准备与分词并行度”。完整 SWE 回放直接发送录制上下文，长度处理交给服务端。

Chat、Responses及静态Decode测试默认在正式测量前按 `max → xhigh → high` 选择服务接受的思考档位，并固定正式请求参数。显式思考配置优先；`null` 使用服务端设置。选择结果保存在运行证据和生成参数中，探测请求独立于正式性能统计，配置方式见使用指南的“默认思考档位”。

配置里的ShareGPT小样本只用于检查请求流程，正式性能测试替换成自己的数据集。相对路径按YAML所在目录解析；`examples/source-only-openai.yaml` 的 `debug/results/source-only-001.db` 实际落在 `examples/debug/results/source-only-001.db`。

## 共享数据集与样本数量

数据集统一放在项目的 `datasets/` 目录，测试配置与结果放在 `result/<测试名称>/`。多个测试通过 `data_files` 引用同一份共享数据集。以下示例以 `datasets/swe/data.jsonl` 中的原生 SWE 完整会话为例，每行一个会话；逐轮输入覆盖 200,000 参考 token 两侧，数据集附带的说明和 manifest 保存会话数量、实际分布与来源。数据集文件由使用者准备，源码仓库保存使用说明和小型示例。

选择样本数量直接使用 `loader.settings.split`。将测试配置保存为 `result/<测试名称>/benchmark.yaml`，其中的数据读取片段如下，选择原始顺序中的前 384 个完整会话：

```yaml
pipeline:
  loader:
    plugin: hf-datasets
    settings:
      dataset: json
      data_files: ../../datasets/swe/data.jsonl
      split: "train[:384]"
  normalization:
    plugin: open-swe-traces
    settings: {}
  selector:
    plugin: all
    settings: {}
```

`split: train` 选择共享文件内的全部会话，完整会话数量见数据集的 manifest；`split: "train[:N]"` 选择前 N 个会话。各测试直接读取这份共享文件，样本数量通过 `split` 配置。`data_files` 相对于 YAML 所在目录解析，示例中的路径指向项目内的共享数据集。每个选中会话保留完整多轮历史，实际请求轮数由这些会话包含的轮次决定。并发和活跃用户数由 `scheduler.settings` 单独配置。同一批样本的选择范围、源文件身份及运行配置随测试结果记录，供模型与并发配置之间复核。

| 配置目标 | 配置位置与文档 |
|---|---|
| 写一份完整YAML | [使用指南](docs/user-guide.md)第2节；高级模式、鉴权与计费见对应章节 |
| 换JSONL／Parquet／多轮数据 | `loader.settings.data_files`、`normalization.plugin`；使用指南第5节给出字段和格式 |
| 指定服务、模型、生成参数 | `executor.settings.base_url/model/extra_body`；生成默认最多4,096 token，保留自然 EOS，使用指南第3节 |
| 改并发、用户池、速率或时长 | `scheduler.settings`；使用指南第4节包含固定并发、扫描和SLO搜索 |
| 决定SQLite放在哪里 | `saver.settings.path`；使用指南第10节 |
| 导出HTML／CSV／Excel | 使用指南第7节；`report/export`读取已有 SQLite 或封存结果生成报告 |

例如读取已有结果：

```bash
uv run perf-benchmark report examples/debug/results/source-only-001.db --output examples/debug/results/source-only-001.html
uv run perf-benchmark export examples/debug/results/source-only-001.db --output examples/debug/results/source-only-001-tables --formats csv xlsx
```

每次运行默认保存 SQLite 和 `<数据库名>-tables/results.xlsx`；`exporter: {}` 启用终端摘要和 Excel 导出。Excel保留 `summary`、`minutes`、`requests`，采集服务容量时另附 `kv_capacity`：按独立池记录GPU/CPU可缓存token数、时间、来源及未知原因。完整原始指标保存在 SQLite / 同身份 Parquet。固定单位并入列标题，例如 `ttft_mean (s)`，长表按行保留单位。显式指定 `--formats csv xlsx` 会额外生成对应CSV。`summary` 前几列放模型、平均输入/输出长度、并发、活跃用户数与生成参数，后面是全局mean/P50/P90/P99，末尾是精简的maxtpm/收入。静态prefill/decode每点默认至少三轮请求配额，并显示有效样本量与P99样本提示。先看Run状态与缺失原因，再比较吞吐和时延。每次测试使用新的数据库文件名。

浏览器结果工作台需要先构建前端资源，构建产物保留本地：

```bash
cd frontend
npm ci --omit=peer
npm run build
cd ..
uv run perf-benchmark results serve --root /absolute/results --host 127.0.0.1 --port 8088
```

打开 `http://127.0.0.1:8088/#/runs` 查看结果。运行中的事实写入 SQLite；结束后由 Saver 封存为 Parquet。
汇总长期保留，SQLite 与完整明细从 Run 结束起保留七天；`report/export` 通过统一读取器访问可用结果。
每次 Run 的 SQLite 路径由 saver.path 指定，默认 Excel 位于其同级的 `<数据库名>-tables/results.xlsx`；`exporter.output` 可显式指定导出目录。相对路径按 YAML 目录解析。
