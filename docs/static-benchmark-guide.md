# 静态 Prefill / Decode 测试

`perf-benchmark static` 按输入长度和并发矩阵生成单轮任务。每个测试点运行一次原生八阶段 Pipeline，使用与多轮测试相同的 Rust HTTP 计时、StandardMetrics、SQLite 和 Parquet 存储。静态入口只负责生成负载、依次运行和整理 Excel。

## 使用

复制 [示例](../examples/static-benchmark.yaml)，填写必需的单行 `case` 测试说明、服务地址、模型名、配套的**本地 fast tokenizer** 目录和输出路径：

```bash
uv run perf-benchmark static examples/static-benchmark.yaml --dry-run
uv run perf-benchmark static examples/static-benchmark.yaml
uv run perf-benchmark static examples/static-benchmark.yaml --only-prefill
uv run perf-benchmark static examples/static-benchmark.yaml --only-decode --output results/decode.xlsx
```

独立脚本 `python benchmarks/static_benchmark.py config.yaml` 调用同一入口。`--base-url`、`--model` 可覆盖配置；`--dry-run` 执行配置校验与矩阵预览；正式运行再加载分词器并发送请求。真实运行的两种模式均需要本地 tokenizer；相对路径按 YAML 所在目录解析。

## 负载

示例在 `service.extra_body.chat_template_kwargs.enable_thinking` 中显式开启思考模式，并在 `decode.reasoning_effort: auto` 中启用档位协商。Prefill 与 Decode 共用模板开关，Decode 的准备探测携带相同模板参数。各阶段继续使用各自的输出 token 目标。服务采用其他思考开关时，按服务协议替换 `service.extra_body` 中的对应字段。

- Prefill：从非特殊 token 随机生成文本，并用本地 **chat template（包含 generation prompt）** 校准目标长度。目标长度校验通过后进入测量，校验失败时停止准备并记录原因。默认每个测试点、每个请求采用不同的确定性序列，减少重复前缀；共享模板仍可能命中缓存，因此实际缓存命中以服务 usage 为准。引擎支持 `cache_salt` 时可显式开启 `service.cache_salt_enabled`。
- Decode：默认轮换八个复杂编码问题，也可通过 `decode.prompts` 指定列表。`reasoning_effort: auto` 为默认设置，在正式测试前依次探测 `max`、`xhigh`、`high`，选择服务接受的最高档位并固定整个矩阵的请求值。服务明确拒绝当前思考档位时继续下一档；鉴权、限流、连接和其他请求错误立即报告。三档均被拒绝时停止准备。显式字符串固定发送该值，`null` 使用服务端设置。输出遵循 `max_tokens` 和 `ignore_eos`，正式请求错误沿用原生失败事实。
- 思考能力探测使用独立短请求，服务用量由部署侧计量；性能统计及计费采用正式测试请求。`workload.json` 保存选择结果和探测状态，Excel 配置记录正式请求采用的档位。Prefill 保持单 token 测量设置。
- Prefill和Decode默认均使用 `temperature: 0.1`；阶段内显式温度优先，包含0。Prefill 与 Decode 的默认输出上限分别为 1 token 和 2560 token，采样温度与生成上限分别配置。
- 每点默认 `rounds: 3`，至少三轮请求配额，连续闭环执行。总请求数为 `rounds × (requests_per_concurrency 或并发数)`。每轮配额必须大于等于所选并发。各轮共用一个原生 Run/Trial，全部请求合并计算均值和分位数；轮次连续执行并沿用同一服务缓存和生命周期。需要独立实验时另开输出目录重复运行。
- 并发 1 默认采集 3 条请求，Excel 列出 TTFT 和速度各自的有效样本数。`requests_per_concurrency` 可增加每点样本量；100 条样本对应约 1% 的经验分位分辨率，尾部分位数的稳定性结合样本量与重复测试结果评估。输入长度与并发矩阵按模型上下文和 KV 容量配置。

## 指标口径

| 指标 | 原生口径 |
|---|---|
| TTFT | 请求开始到首个有效输出事件；role-only、空 delta、usage 与结束事件按协议控制信息处理 |
| 单请求 Prefill 速度 | 实际输入 tokens / TTFT；描述包含排队、传输与首输出生成的客户端速度 |
| 单请求 Decode 速度 | 原生 decode_throughput：实际输出 tokens 与流式输出窗口；沿用原生 eligibility 和缺失原因 |
| 批次吞吐 | 原生 workload_prompt / workload_completion，**包含 TTFT 和排队** |
| Token 数 | 使用服务实际 usage；观测缺失时记录空白及 unavailable，SSE chunk 按传输事件单独记录 |
| maxTPM | 沿用原生完整分钟窗口规则；短批次无完整窗口时 unavailable |

时延/速度展示 mean、P50、P90、P99。空白表示指标处于 unknown 或 unavailable 状态，零值表示实际观测到的数值 0。具体公式、样本 eligibility、可用性和来源以原生事实及“指标可用性”工作表为准。

## 产物与复核

默认只导出一份合并 Excel：`Prefill性能测试`、`Decode性能测试`、`测试概览与配置`、`指标可用性`、`KV容量`。性能表首列记录case与测试点；首行/首列冻结，首列宽度按完整文本调整。固定单位直接写入列标题；长表的指标单位随行保留。每点完成后追加新增结果，失败/取消及原生 Run ID 一并保留。请求失败或取消时 CLI 返回非零退出码。

`output.excel_path` 支持已有工作簿。同名子表按完整列名对齐后追加，原有列顺序和历史行保留，新指标列追加到末尾，缺失指标留空。单位参与列名匹配，首列标题须一致，其余标题须为唯一的非空文本。缺少的子表自动创建，其他子表保留。将它与多轮 YAML 中 `pipeline.exporter.plugins` 列表内 tabular 的 `settings.excel_path` 设置为同一个路径，即可在同一 Excel 中保存静态 Prefill、Decode 和多轮结果。重复运行按 case 区分；每次静态运行只追加各测试点一次，配置记录一次。两个入口共用文件锁及原子发布流程。

KV容量在每点请求开始前采集，记录GPU和CPU内存独立池能缓存的token总量。多轮及其他 `run` 测试使用同一采集入口，存入原生Run证据及Excel的 `kv_capacity` 表。SGLang读取每DP的server-info容量和HiCache总量指标；同一 DP 内的 TP 分片按一个逻辑缓存池计入容量。采集超时、鉴权限制或字段缺失会明确标未知，测试继续。引擎内部 block 预算单独标记为 `gpu_block_slots`；混合缓存的完整模型 token 容量结合缓存组布局换算，布局缺失时保留未知。

部署端可用 `service.capacity_snapshot`（通用测试为 `application.service_capacity.snapshot`）传入实测容量JSON，例如：

```json
{"model":"served-model","observed_at":"2026-10-16T12:00:00+08:00","pools":[{"tier":"gpu","pool":"dp:0","capacity_tokens":2000000,"source":"engine startup GPU KV cache size"},{"tier":"host_cpu","pool":"shared-lmcache","capacity_tokens":4000000,"source":"registered complete KV chunk geometry and allocated CPU pool"}]}
```

vLLM/LMCache 的完整 token 容量使用实际注册的缓存组、精度、chunk 布局和已分配内存换算；容量证据完整时填写数值，其余情况保留未知。共享CPU池只列一次。快照保留其原采集时间，使用者需确保来自本次部署；运行中改变缓存预算应重新采集。

`<Excel文件名>-facts/run-<唯一标识>/<测试点>/` 保存 `run.db`、其原生 Parquet 归档和精简 `workload.json`（种子、分词器指纹、请求哈希）。每次运行使用独立目录。临时输入在该点执行后清理，各点结果追加至同一 Excel。原生配置、计量事实和可用性记录保存在结果目录中，响应正文随临时请求资源释放。`output.json_path` 默认为 null，需要时可额外导出汇总 JSON。

已有 Excel 按表头契约追加；可选的 JSON 输出每次使用新路径。HTML/CSV 按需从对应原生数据库生成，例如：

```bash
uv run perf-benchmark export results/performance-facts/run-<唯一标识>/prefill-128-c1/run.db --output results/review --formats csv
```

完整明细沿用原生七天保留策略，汇总长期保留。多个 Run 同时运行时，Parquet 封存会按原生规则延后到活动写入结束。鉴权通过服务配置传给原生执行器；报告保存脱敏配置与计量证据。
