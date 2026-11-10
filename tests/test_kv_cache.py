"""kv_cache 模块单元测试：KV cache 显存估算公式。"""
from __future__ import annotations

from src.eval.kv_cache import compare_vlm_vs_text, estimate_kv_cache


def test_estimate_formula():
    # 2(K+V) * layers * heads * head_dim * seq_len * bytes
    # layers=2, heads=2, head_dim=4, seq_len=10, bf16=2bytes
    # = 2 * 2 * 2 * 4 * 10 * 2 = 640 bytes
    r = estimate_kv_cache(layers=2, heads=2, head_dim=4, seq_len=10, dtype="bf16")
    assert r["bytes"] == 640
    assert r["mb"] == round(640 / 1024**2, 2)


def test_fp8_is_half_of_bf16():
    r_bf16 = estimate_kv_cache(2, 2, 4, 10, "bf16")
    r_fp8 = estimate_kv_cache(2, 2, 4, 10, "fp8")
    assert r_fp8["bytes"] == r_bf16["bytes"] / 2


def test_kv_cache_scales_linearly_with_seq_len():
    r1 = estimate_kv_cache(2, 2, 4, 10, "bf16")
    r2 = estimate_kv_cache(2, 2, 4, 20, "bf16")
    assert r2["bytes"] == r1["bytes"] * 2


def test_vlm_much_bigger_than_text():
    cmp = compare_vlm_vs_text(layers=2, heads=2, head_dim=4, text_seq_len=100, visual_tokens=[1000])
    assert cmp["vlm"][1000]["bytes"] > cmp["text_only"]["bytes"]
    # 1100/100 = 11 倍序列长度，KV cache 也 11 倍
    assert cmp["vlm"][1000]["seq_len"] == 1100
