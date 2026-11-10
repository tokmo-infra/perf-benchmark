# vLLM 逐请求生成计时

工具解析器可能缓冲整个工具调用后才发出一个 SSE 输出事件。此时客户端
首末输出区间不足，`tpot` 必须保持不可用，不能填零或只统计可观测请求后
宣称覆盖全部请求。服务端生成计时用于独立观察这个盲区，不修改客户端指标。

## 启用与传输

支持该扩展的 vLLM 启动添加 `--enable-per-request-metrics`，不同时使用
`--disable-log-stats`。流式请求使用 `stream_options.include_usage=true`
（本工具 OpenAI Chat 执行器默认请求 usage）。先验证后端及中间 router
保留最终 usage chunk 的顶层 `metrics`；不支持该字段的服务正常运行，
相关指标保持不可用。执行器也支持非流式响应的同名字段。

执行器只接受 `generation_time_ms`、`mean_itl_ms`、`queue_time_ms` 和
`time_to_first_token_ms` 中有限、非负的 JSON 数字，保存为逐请求
`measurements.request.server_metrics`，随后冻结并持久化；布尔值、字符串、
负数和空值不当作计时。metrics-only 帧不增加输出事件或改变客户端时钟。

## 独立指标与统计分母

当 `generation_time_ms > 0` 且 usage 中 `completion_tokens > 1` 时，报告：

| 指标 | 公式 | 单位 |
|---|---|---|
| `server_generation_time` | `generation_time_ms` | ms |
| `server_tpot` | `generation_time_ms / (completion_tokens - 1)` | ms |
| `server_decode_throughput` | `(completion_tokens - 1) * 1000 / generation_time_ms` | tok/s |

窗口沿用原始成功请求 cohort：普通窗口按请求起点，maxTPM 按其既有
完成时刻及首遍会话规则。均值为请求等权算术平均；`n` 是可用样本数，
`eligible_n` 是该 cohort 全部成功请求数，缺测不会缩小分母。maxTPM 仍须
有完整分钟。任何全请求 SLO 判断必须额外检查 `n == eligible_n > 0`。

上述扩展约定生成时长为引擎首个到末个输出 token 的区间，不含排队和
prefill；启用前须核实被测服务实现。投机解码一次可能输出多个 token，
因此引擎批次时钟也不是每个 token 的独立时间戳。服务端
`time_to_first_token_ms` 从 scheduled 起算，不能替代含客户端等待和网络的
TTFT；服务端指标不能悄悄替代可见 SSE 的 TPOT，也不能回补旧结果。

客户端 TTFT、TPOT、原始 decode 分布、usage 及 TPM/收入计费规则不变。
缺失或不可用的服务端时长不从客户端时钟推算，不使用 `mean_itl_ms`
作为缺测 fallback，也不裁剪、填零或选择性加权样本。
