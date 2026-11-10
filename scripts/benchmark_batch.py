"""W3-1：静态批处理基准 —— 测 batch∈{1,2,4} 的吞吐/延迟/显存三轴权衡。

================================================================================
【这个文件做什么】
    加载 bnb4 量化模型，对同一批 SROIE 收据分别用 batch=1/2/4 做静态批处理推理，
    记录每种 batch 的总吞吐（样本/秒）、每样本延迟、峰值显存，输出对比报告到
    reports/w3_batching.md。回答部署的核心问题：「8GB 卡上，批处理能提多少吞吐、
    付出多少延迟和显存？」

【为什么做这个】
    W3「引擎/推理层优化」的落地点之一。vLLM 的「连续批处理」在原生 Windows +
    Blackwell 装不了（已实证），但 HF 原生的「静态批处理」能做、能测、能讲清
    吞吐-延迟-显存的三角关系。这个基准是「部署优化」的量化证据，也是讲
    「静态批处理 vs 连续批处理」区别的实验基础。

【为什么正确性要先验】
    批处理改了 generate 的输入形态（padding 到最大视觉 token），必须先确认
    batch 推理的输出与单图推理一致（不因 padding 串味），再谈性能数字，否则
    基准是错的。

【时间估算】
    bnb4 单张 ~6s，30 张 × 3 个 batch ≈ 6-8 分钟，后台可跑。

用法::

    py -3.11 -m scripts.benchmark_batch --num_samples 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from src.data.sroie import FIELDS, load_manifest, load_sample
from src.inference.engine_hf import _repair_json, generate_batch, generate_text, load_model
from src.utils.logger import get_logger

log = get_logger(__name__)

LONGEST_EDGE = 2097152  # 2M，与 W2 基准一致，控大图视觉 token


def _batch_timed(model, processor, images, batch_size: int) -> tuple[float, float]:
    """把 images 按 batch_size 分组推理，返回 (总延迟秒, 峰值显存MB)。"""
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for i in range(0, len(images), batch_size):
        chunk = images[i : i + batch_size]
        generate_batch(model, processor, chunk, longest_edge=LONGEST_EDGE)
    torch.cuda.synchronize()
    latency = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    return latency, peak_mb


def main() -> None:
    ap = argparse.ArgumentParser(description="静态批处理基准")
    ap.add_argument("--num_samples", type=int, default=30)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    model, processor = load_model(
        "Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="models", quantization="bnb4"
    )
    manifest = load_manifest()
    items = manifest[: args.num_samples]
    images = [load_sample(it)[0] for it in items]
    log.info("加载 %d 张图，准备批处理基准", len(images))

    # 正确性自检：batch∈{1,2,4} 的批处理输出，逐条与单图推理比对（字段级）。
    # 验证 padding 不串味——批处理改输入形态，必须先确认输出与单图一致再谈性能。
    def _parse(text: str) -> dict:
        try:
            parsed = _repair_json(text)
        except ValueError:
            parsed = {}
        return {f: parsed.get(f, "") for f in FIELDS}

    # 自检用前 max_batch 张图（单图推理慢，不必全量）；样本不足时按实际数量比对，防止越界。
    max_batch = max(args.batches)
    check_n = min(len(images), max_batch)
    singles = [
        _parse(generate_text(model, processor, img, longest_edge=LONGEST_EDGE))
        for img in images[:check_n]
    ]
    consistent = True
    for bs in args.batches:
        # 样本不足该 batch 时跳过自检（拿不出 bs 张图无从比对）
        if bs > len(images):
            log.warning("样本不足 batch=%d，跳过该档自检", bs)
            continue
        batched = generate_batch(model, processor, images[:bs], longest_edge=LONGEST_EDGE)
        batch_parsed = [_parse(t) for t in batched]
        ok = all(batch_parsed[i] == singles[i] for i in range(bs))
        consistent = consistent and ok
        log.info("正确性自检：batch=%d 输出与单图 %s", bs, "一致 ✅" if ok else "不一致 ❌")

    rows = []
    for bs in args.batches:
        total_lat, peak_mb = _batch_timed(model, processor, images, bs)
        rows.append(
            {
                "batch": bs,
                "throughput_img_s": round(len(images) / total_lat, 3),
                "latency_per_img_s": round(total_lat / len(images), 3),
                "peak_mb": round(peak_mb, 1),
            }
        )
        log.info(
            "batch=%d: 吞吐 %.3f 样本/s | 每样本延迟 %.3fs | 峰值显存 %.1fMB",
            bs,
            rows[-1]["throughput_img_s"],
            rows[-1]["latency_per_img_s"],
            rows[-1]["peak_mb"],
        )

    # 相对 batch=1 的提升（基准恒取 batch=1，不依赖用户在 --batches 里的顺序）
    base = next((r["throughput_img_s"] for r in rows if r["batch"] == 1), rows[0]["throughput_img_s"])
    batch_list = ", ".join(str(b) for b in args.batches)
    lines = [
        "# W3 静态批处理基准（bnb4 · longest_edge=2M）",
        "",
        f"样本 {len(images)} 张 SROIE 收据 · 模型 Qwen2.5-VL-3B bnb4 · 引擎 HF Transformers",
        "",
        f"正确性自检：batch∈{{{batch_list}}} 输出与单图 {'一致' if consistent else '不一致'}",
        "",
        "| batch | 吞吐(样本/s) | 相对提升 | 每样本延迟(s) | 峰值显存(MB) |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        speedup = r["throughput_img_s"] / base if base else 0.0
        lines.append(
            f"| {r['batch']} | {r['throughput_img_s']:.3f} | ×{speedup:.2f} "
            f"| {r['latency_per_img_s']:.3f} | {r['peak_mb']:.0f} |"
        )
    # 数据驱动的结论段：按实测数据判断「吞吐/延迟是否单调」，不硬编码结论。
    thr = [r["throughput_img_s"] for r in rows]
    lat = [r["latency_per_img_s"] for r in rows]
    peak = [r["peak_mb"] for r in rows]
    thr_up = all(thr[i] >= thr[i - 1] for i in range(1, len(thr)))
    lat_down = all(lat[i] <= lat[i - 1] for i in range(1, len(lat)))

    if thr_up:
        thr_line = f"- 吞吐随 batch 单调上升（{thr[0]:.3f} → {thr[-1]:.3f}），GPU 矩阵乘利用率提升；"
    else:
        best_i = thr.index(max(thr))
        thr_line = f"- 吞吐在 batch={rows[best_i]['batch']} 达峰 {thr[best_i]:.3f} 后回落，非单调（后续 batch 显存逼近上限 + padding 浪费）；"
    if lat_down:
        lat_line = (
            f"- **反直觉**：每样本延迟随 batch 反而下降（{lat[0]:.3f} → {lat[-1]:.3f}s），"
            "因 GPU 未饱和、batch 摊销 kernel 启动等固定开销；经典「延迟随 batch 上升」只在 GPU 饱和后才成立；"
        )
    else:
        lat_line = "- 每样本延迟随 batch 上升（GPU 趋近饱和，等待整批完成的代价显现）。"

    lines += [
        "",
        "## 结论",
        "",
        thr_line,
        lat_line,
        f"- 峰值显存随 batch 上升（{peak[0]:.0f} → {peak[-1]:.0f}MB），8GB 卡批处理上限由「峰值显存不越界」决定。",
        "",
        "## 讲解要点",
        "",
        "- 静态批处理 =「凑齐一批再一起 forward」，请求之间互相等（P99 延迟差）；",
        "- 连续批处理（vLLM）=「请求级调度、谁好了谁补位」，吞吐更高、延迟更稳；",
        "- 本项目 HF 原生只能静态批处理，连续批处理是 vLLM 的卖点——讲清这个区别，",
        "  就讲清了「为什么生产要用 vLLM 而不是裸 HF」。",
    ]
    out = Path("reports/w3_batching.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("报告已保存: %s", out)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
