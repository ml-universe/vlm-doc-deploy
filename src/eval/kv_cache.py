"""KV cache 显存估算：定量解释「VLM 的 KV cache 为何比纯文本更吃显存」。

================================================================================
【这个文件做什么】
    纯计算（不加载模型）地估算推理时 KV cache 占用的显存，是 W3「引擎对比」
    的定量工具。核心结论：KV cache 随「序列长度（视觉 token + 文本 token）×
    层数」线性增长，而 VLM 的视觉 token 数量级远大于纯文本，所以 VLM 的
    KV cache 是第一显存瓶颈。

【KV cache 大小公式】
    KV_cache_bytes = 2(K+V) × layers × heads × head_dim × seq_len × dtype_bytes
    - 2：K 和 V 各一份；
    - layers × heads × head_dim：模型的 KV 维度（由 config 决定）；
    - seq_len：prompt（含视觉 token）+ 已生成的 token 数；
    - dtype_bytes：bf16=2、fp16=2、fp8=1、4bit=0.5（若有 KV 量化）。

【为什么这是「系统感」最强的点】
    文本模型一个 prompt 几百 token；VLM 一张图几百到上万视觉 token（SROIE 大扫描件
    ~1.1 万）。同样 3B 模型，VLM 单次推理的 KV cache 可能是纯文本的几十倍。
    这就是「边缘部署 VLM 比 LLM 更吃显存」的数学根源，不是玄学，是可算的。

【视觉前缀缓存（W3 优化点，本模块配套讲）】
    若多个请求/多轮对话共享同一张图，视觉编码器的输出（视觉 token 的 K/V）可
    只算一次、缓存复用，跨请求省去视觉部分的重复计算与 KV 存储。这是
    「视觉 token 跨请求共享复用」的降延迟手段。

用法::

    from src.eval.kv_cache import estimate_kv_cache
    est = estimate_kv_cache(layers=36, heads=2, head_dim=128, seq_len=11150, dtype="bf16")
"""
from __future__ import annotations

# dtype 对应的字节数
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "int4": 0.5,
}


def estimate_kv_cache(
    layers: int,
    heads: int,
    head_dim: int,
    seq_len: int,
    dtype: str = "bf16",
) -> dict:
    """估算单序列 KV cache 显存（MB）。

    参数来自模型 config：layers=num_hidden_layers, heads=num_key_value_heads
    （GQA 下是 KV heads，非 query heads）, head_dim=head_dim。
    """
    b = DTYPE_BYTES.get(dtype, 2.0)
    # 2 = K + V
    bytes_total = 2 * layers * heads * head_dim * seq_len * b
    return {
        "seq_len": seq_len,
        "dtype": dtype,
        "bytes": bytes_total,
        "mb": round(bytes_total / 1024**2, 2),
        "gb": round(bytes_total / 1024**3, 3),
    }


def compare_vlm_vs_text(
    layers: int,
    heads: int,
    head_dim: int,
    text_seq_len: int = 200,
    visual_tokens: list[int] | None = None,
    dtype: str = "bf16",
) -> dict:
    """对比「纯文本」与「VLM（文本 + N 个视觉 token）」的 KV cache 大小。

    visual_tokens 给几个典型值（如 [500, 2600, 11000]），对应小票/中图/大扫描件。
    """
    visual_tokens = visual_tokens or [500, 2645, 11000]
    text = estimate_kv_cache(layers, heads, head_dim, text_seq_len, dtype)
    vlm = {
        vt: estimate_kv_cache(layers, heads, head_dim, text_seq_len + vt, dtype)
        for vt in visual_tokens
    }
    return {"text_only": text, "vlm": vlm}
