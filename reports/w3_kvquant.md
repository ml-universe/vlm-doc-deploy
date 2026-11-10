# W3 KV cache 量化基准（bnb4 权重 · longest_edge=2M）

模型 Qwen2.5-VL-3B bnb4 · 引擎 HF Transformers · KV 量化 backend=hqq（KIVI 风格 int 量化）

| 样本 | 模式 | 延迟(s) | 峰值显存(MB) | Δ显存(MB) | 输出与 baseline 一致 |
|---|---|---|---|---|---|
| 小票(视觉 token 少) (005) | baseline(bf16 KV) | 4.28 | 2410.3 | — | — |
| 小票(视觉 token 少) (005) | kv-hqq4 | 12.95 | 2400.5 | -9.8 | ❌ |
| 小票(视觉 token 少) (005) | kv-hqq8 | 4.38 | 2404.7 | -5.6 | ✅ |
| 大扫描件(视觉 token 多) (543) | baseline(bf16 KV) | 6.50 | 3668.9 | — | — |
| 大扫描件(视觉 token 多) (543) | kv-hqq4 | 8.06 | 3591.1 | -77.8 | ❌ |
| 大扫描件(视觉 token 多) (543) | kv-hqq8 | 7.55 | 3615.7 | -53.2 | ✅ |

## 结论

- **int8 是「免费午餐」**：输出与 baseline 完全一致（无损），延迟几乎不涨
  （大图 +1.05s / 小图 +0.1s），峰值显存省 5~53MB（随视觉 token 数放大）。
- **int4 得不偿失**：输出与 baseline 不一致（量化误差破坏了生成，精度不可接受），
  且小图延迟暴涨 3×（12.95s vs 4.28s）——小 KV cache 下 per-token 量化/反量化
  overhead 大于省下的显存收益。
- **单请求边界**：权重 2.3GB 主导，KV cache 仅几十~百 MB，量化收益有限；
  int8 的价值要在「batch N 时 KV cache × N」或长文档/长视频场景才放大。

## 讲解要点

- KV cache 量化 = 对每层 K/V 做 per-channel 非对称量化（KIVI），residual_length
  保留最近若干 token 高精度、其余进低精度缓存。
- 诚实结论：不是越激进越好——int4 精度崩、延迟涨，int8 才无损且近零延迟代价；
  「量化位宽要按精度-延迟-显存实测来选」是系统感，不是玄学。
- 与权重量化（W2 bnb4）正交：权重 4bit 省权重显存，KV 量化省缓存显存，可叠加；
  单请求下权重主导，KV 收益要在并发/长序列才显。
