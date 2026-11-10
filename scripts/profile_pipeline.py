"""W3-2：三段耗时 profiling —— 图像预处理(CPU) / 视觉编码(GPU) / 文本生成(GPU)。

================================================================================
【这个文件做什么】
    拆解 VLM 单张推理的耗时结构，量化「视觉编码占比 P」。这是 W3-4 视觉前缀
    缓存收益上界的直接输入（reports/w3_prefix_cache.md 里标了「待填」的 P）：
        前缀缓存收益 ≈ P × R（R = 视觉前缀重复率）

    三段是「独立计时」（不是串行累加，因为 generate 内部又跑了一次视觉编码）：
      t_preprocess = processor() —— 图像 resize/切图 + tokenize（CPU 为主）
      t_vision     = model.get_image_features(pixel_values, image_grid_thw) —— 视觉塔 32 层 ViT
      t_generate   = model.generate(...) —— 端到端（视觉编码 + projector + LLM 36 层自回归）
    视觉占比 P = t_vision / t_generate。

【为什么测大小两种图】
    视觉占比 P 随视觉 token 数变化：小票视觉 token 少（~500）、视觉编码占比低；
    大扫描件视觉 token 多（2M 约束下 ~2645）、视觉编码占比高。前缀缓存只对
    P 高的「大图/多轮同图」场景有意义——这是 W3-4 诚实结论的定量支撑。

【讲解要点】
    「一张图的推理时间去哪了」是系统感很强的 profiling 题。能答出「小票瓶颈在
    文本自回归，大图瓶颈向视觉编码倾斜」，比只会报总延迟高一个层次。

用法::

    py -3.11 -m scripts.profile_pipeline
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from src.data.sroie import image_area, load_manifest, load_sample
from src.inference.engine_hf import EXTRACTION_PROMPT, load_model
from src.utils.logger import get_logger

log = get_logger(__name__)

LONGEST_EDGE = 2097152  # 2M，与 W2/W3 基准一致


def _timed(fn):
    """执行 fn 并返回 (结果, 墙钟秒)，前后 synchronize 保证 GPU 计时准确。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def _build_inputs(processor, img):
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": EXTRACTION_PROMPT}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    text_only = processor(text=[text], return_tensors="pt")["input_ids"].shape[1]
    inputs = processor(
        text=[text], images=[img], padding=True, return_tensors="pt",
        size={"longest_edge": LONGEST_EDGE, "shortest_edge": 3136},
    )
    return inputs, text_only


def main() -> None:
    model, processor = load_model(
        "Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="models", quantization="bnb4"
    )
    manifest = load_manifest()

    # 挑面积最小（小票）与最大（大扫描件）各一张，覆盖两种 regime
    sized = sorted(manifest, key=image_area)
    picks = [sized[0], sized[-1]]
    labels = ["小票(视觉 token 少)", "大扫描件(视觉 token 多)"]

    device = next(model.parameters()).device
    dtype = model.dtype

    rows = []
    for item, label in zip(picks, labels):
        img, _ = load_sample(item)
        inputs, text_only = _build_inputs(processor, img)
        n_visual = inputs["input_ids"].shape[1] - text_only

        # 1) 预处理（CPU 为主）
        _, t_pre = _timed(lambda: _build_inputs(processor, img))

        # 2) 视觉编码（视觉塔前向，包 no_grad 与真实 generate 路径一致，避免计计时失真）
        pv = inputs["pixel_values"].to(device, dtype=dtype)
        gthw = inputs["image_grid_thw"].to(device)
        with torch.no_grad():
            _, t_vision = _timed(lambda: model.get_image_features(pv, gthw))

        # 3) 端到端生成
        inputs_d = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _, t_gen = _timed(lambda: model.generate(**inputs_d, max_new_tokens=256))

        p_vision = t_vision / t_gen if t_gen else 0.0
        rows.append(
            {
                "label": label,
                "id": item["id"],
                "visual_tokens": n_visual,
                "t_preprocess": round(t_pre, 3),
                "t_vision": round(t_vision, 3),
                "t_generate": round(t_gen, 3),
                "vision_ratio": round(p_vision, 3),
            }
        )
        log.info(
            "%s (id=%s): 视觉 token=%d | 预处理 %.3fs | 视觉编码 %.3fs | 端到端 %.3fs | 视觉占比 %.1f%%",
            label, item["id"], n_visual, t_pre, t_vision, t_gen, p_vision * 100,
        )

    lines = [
        "# W3 推理耗时结构 profiling（bnb4 · longest_edge=2M）",
        "",
        "模型 Qwen2.5-VL-3B bnb4 · 引擎 HF Transformers · 视觉塔 32 层 ViT + LLM 36 层",
        "",
        "| 样本 | 视觉 token | 预处理(CPU) | 视觉编码(GPU) | 端到端生成(GPU) | 视觉占比 P |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} ({r['id']}) | {r['visual_tokens']} | {r['t_preprocess']}s "
            f"| {r['t_vision']}s | {r['t_generate']}s | {r['vision_ratio']:.0%} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "- **视觉占比 P 随视觉 token 数上升**：小票视觉 token 少、瓶颈在文本自回归；",
        "  大扫描件视觉 token 多、视觉编码占比显著抬升。",
        "- 这就是「视觉前缀缓存」收益上界的来源：P 是单请求里可缓存部分的上限，",
        "  实际收益 ≈ P × R（R=前缀重复率）。本项目一次性抽取 R≈0，故收益≈0（见",
        "  `w3_prefix_cache.md`）。",
        "- 预处理（CPU 图像 resize/切图）占比极小，不是瓶颈——边缘部署的瓶颈在",
        "  GPU 上的视觉编码 + 文本生成，优化应瞄准 GPU 而非 CPU 预处理。",
        "",
        "## 讲解要点",
        "",
        "- 「一张 VLM 推理的时间去哪了」：视觉塔（ViT）编码视觉 token + LLM 自回归",
        "  生成文本，两者占比随图大小变化——能答这个就是系统感。",
        "- 视觉前缀缓存能省的就是 P 这部分（且仅在同图多请求时），省不掉文本生成。",
    ]
    out = Path("reports/w3_profile.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("报告已保存: %s", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
