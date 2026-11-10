"""W3：定量分析「视觉 token 数 → KV cache 显存」的关系（用真实模型 config）。

================================================================================
【这个文件做什么】
    读 Qwen2.5-VL-3B 的 config（层数/KV heads/head_dim），套用 KV cache 估算
    公式，输出「纯文本 vs 不同视觉 token 数」的 KV cache 显存对比表，并落到
    reports/visual_tokens_kv.md。这是 W3「VLM KV cache 吃显存」的定量证据，
    与 W2 实测的「大扫描件 21GB 峰值」互相印证。

【为什么这是「引擎对比」的起点】
    W3 要讲「KV cache / 视觉前缀缓存优化」，前提是先定量说明「KV cache 到底
    吃多少、视觉 token 为什么是主因」。这个脚本给的就是这个定量基线；后续
    引擎优化（vLLM PagedAttention / 前缀缓存）都建立在这个数上。

【关键公式】
    KV_cache = 2(K+V) × layers × kv_heads × head_dim × seq_len × 2bytes(bf16)
    seq_len = 文本 token + 视觉 token。视觉 token 由图片分辨率/切图策略决定
    （W2 已实测：大扫描件 11147 token，longest_edge=2M 后 2645 token）。

用法::

    py -3.11 -m scripts.analyze_visual_tokens
"""
from __future__ import annotations

import json
from pathlib import Path

from src.eval.kv_cache import compare_vlm_vs_text
from src.utils.logger import get_logger
from src.utils.model_download import resolve_model

log = get_logger(__name__)


def _load_config(path: str) -> dict:
    return json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))


def main() -> None:
    path = resolve_model("Qwen/Qwen2.5-VL-3B-Instruct", "models")
    cfg = _load_config(path)
    layers = cfg["num_hidden_layers"]
    kv_heads = cfg.get("num_key_value_heads", cfg.get("num_attention_heads"))
    head_dim = cfg.get("head_dim") or (cfg["hidden_size"] // cfg["num_attention_heads"])

    log.info("模型 config: layers=%d, kv_heads=%d, head_dim=%d", layers, kv_heads, head_dim)

    # 视觉 token 三个典型档：小票 / 中图(2M约束) / 大扫描件(无约束)
    cmp = compare_vlm_vs_text(layers, kv_heads, head_dim, text_seq_len=200)

    lines = [
        "# 视觉 token → KV cache 显存分析",
        "",
        f"模型 Qwen2.5-VL-3B · layers={layers} · kv_heads={kv_heads} · head_dim={head_dim} · dtype=bf16",
        "",
        "| 场景 | 视觉 token | 总序列长度 | KV cache |",
        "|---|---|---|---|",
        f"| 纯文本（无图） | 0 | 200 | {cmp['text_only']['mb']:.1f} MB |",
    ]
    for vt, r in cmp["vlm"].items():
        lines.append(f"| VLM + {vt} 视觉 token | {vt} | {r['seq_len']} | {r['mb']:.1f} MB |")
    lines += [
        "",
        "## 结论",
        "",
        f"- 纯文本 200 token 的 KV cache 仅 **{cmp['text_only']['mb']:.1f} MB**；",
    ]
    vlm_vals = sorted(cmp["vlm"].values(), key=lambda r: r["seq_len"])
    lines.append(
        f"- 大扫描件 ~11000 视觉 token 时 KV cache 达 **{vlm_vals[-1]['mb']:.1f} MB**"
        f"（是纯文本的 **{vlm_vals[-1]['mb'] / cmp['text_only']['mb']:.0f} 倍**）。"
    )
    lines.append(
        "- 这解释了 W2 实测：大扫描件推理峰值显存 21GB、延迟 60s+ —— KV cache 随"
        "视觉 token 线性爆炸，是 VLM 边缘部署的第一瓶颈。"
    )
    lines.append(
        "- 优化方向（W3）：① 视觉前缀缓存（跨请求共享视觉 K/V）；② 分辨率约束"
        "（longest_edge）；③ KV 量化（fp8/int4）；④ PagedAttention（碎片化）。"
    )

    out = Path("reports") / "visual_tokens_kv.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("分析已保存: %s", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
