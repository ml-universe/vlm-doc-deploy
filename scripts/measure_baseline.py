"""测 bf16 原生推理的显存/延迟基线，作为 W2 量化实验的 Pareto 锚点。

================================================================================
【这个文件做什么】
    加载 Qwen2.5-VL-3B（bf16），报告：
      1. 模型加载后 GPU 显存占用 + 总 footprint（含 device_map offload 到 CPU 的部分）；
      2. 对 N 张 SROIE 真实收据各推理一次，测延迟、峰值显存、吞吐；
    结果落到 reports/baseline_bf16.json，供 W2 各量化方法与它对比。

【为什么基线重要】
    「4bit 掉 0.5% 精度、显存省一半」这类结论，必须有个「不量化的 bf16」做参照。
    这个脚本产出的就是那个参照点。

用法::

    py -3.11 -m scripts.measure_baseline --num_samples 5
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.data.sroie import load_manifest, load_sample
from src.eval.memory import gpu_snapshot, model_footprint_mb, timed_inference
from src.inference.engine_hf import generate_text, load_model
from src.utils.logger import get_logger

log = get_logger(__name__)


def measure_baseline(
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir: str = "models",
    num_samples: int = 5,
    out: str = "reports/baseline_bf16.json",
    quantization: str = "none",
) -> dict:
    model, processor = load_model(model_name, cache_dir, quantization=quantization)

    loaded = gpu_snapshot()
    footprint = model_footprint_mb(model)
    log.info(
        "模型加载后: GPU allocated=%.1fMB, reserved=%.1fMB, 总 footprint=%.1fMB（含 CPU offload）",
        loaded["gpu_allocated_mb"],
        loaded["gpu_reserved_mb"],
        footprint,
    )

    manifest = load_manifest()
    latencies: list[float] = []
    peaks: list[float] = []
    details: list[dict] = []
    for item in manifest[:num_samples]:
        img, _ = load_sample(item)
        timed = timed_inference(lambda: generate_text(model, processor, img))
        latencies.append(timed["latency_s"])
        peaks.append(timed["gpu_peak_mb"])
        details.append({"id": item["id"], **timed})
        log.info(
            "样本 %s: 延迟 %.2fs, 峰值显存 %.1fMB, 吞吐 %.1f tok/s",
            item["id"],
            timed["latency_s"],
            timed["gpu_peak_mb"],
            timed["throughput_tok_s"],
        )

    result = {
        "model": model_name,
        "quantization": quantization if quantization != "none" else "bf16",
        "engine": "hf-transformers",
        "loaded_gpu_allocated_mb": loaded["gpu_allocated_mb"],
        "loaded_gpu_reserved_mb": loaded["gpu_reserved_mb"],
        "model_footprint_mb": footprint,
        "num_samples": num_samples,
        "latency_s_mean": round(statistics.mean(latencies), 3),
        "latency_s_median": round(statistics.median(latencies), 3),
        "gpu_peak_mb_mean": round(statistics.mean(peaks), 1),
        "gpu_peak_mb_max": round(max(peaks), 1),
        "throughput_tok_s_mean": round(
            statistics.mean(d["throughput_tok_s"] for d in details), 2
        ),
        "details": details,
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("基线已保存: %s", out_path)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="测推理显存/延迟基线（可指定量化）")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache_dir", default="models")
    ap.add_argument("--num_samples", type=int, default=5)
    ap.add_argument("--out", default=None, help="输出路径，默认按量化方式派生")
    ap.add_argument("--quantization", default="none", choices=["none", "bnb4"])
    args = ap.parse_args()
    quant = args.quantization if args.quantization != "none" else "bf16"
    out = args.out or f"reports/baseline_{quant}.json"
    measure_baseline(args.model, args.cache_dir, args.num_samples, out, args.quantization)


if __name__ == "__main__":
    main()
