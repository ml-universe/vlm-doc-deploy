"""量化方法完整评估：精度（字段 F1 + bootstrap CI）+ 速度 + 显存，三轴合一。

================================================================================
【这个文件做什么】
    W2 量化对比的核心评估入口。对指定量化方式（none=bf16 / bnb4 / …）：
      1. 加载模型；
      2. 对 SROIE 前 N 张收据逐一推理，记录每个样本的预测字段 + 延迟 + 峰值显存，
         并实时落盘到 reports/preds_<quant>.json（支持断点续跑）；
      3. 算字段级 F1（exact + normalized 两档）+ bootstrap 95% CI；
      4. 汇总落到 reports/eval_<quant>.json。

【为什么落盘 per-sample + 断点续跑】
    全量 624 张 bf16 要 ~3 小时，若中途 OOM/中断，已推理的样本不能白跑。落盘
    每个样本结果后重跑可跳过已完成 id（SROIE id 唯一）。这是第一个项目
    「断点续训」工程经验的迁移。

【为什么精度用「量化前后对比」而非绝对分】
    项目定位是部署/推理优化。核心结论是「4bit 相比 bf16 精度损失多少、显存/速度
    换来多少」，故固定同一批样本、同一字段、同一 prompt，只变量化方式。

用法::

    py -3.11 -m scripts.evaluate_quant --quantization bnb4 --num_samples 624
    py -3.11 -m scripts.evaluate_quant --quantization none --num_samples 624 --longest_edge 2097152
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from src.data.sroie import FIELDS, load_manifest, load_sample
from src.eval.memory import gpu_snapshot, model_footprint_mb, timed_inference
from src.eval.metrics import bootstrap_f1, field_f1
from src.inference.engine_hf import generate_text, load_model, _repair_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def _preds_path(quant: str, reports_dir: str) -> Path:
    return Path(reports_dir) / f"preds_{quant}.json"


def _load_done(preds_path: Path) -> tuple[set[str], list[dict]]:
    """读已完成的 per-sample 结果，返回 (已完成 id 集合, 结果列表)。"""
    if not preds_path.exists():
        return set(), []
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    done = {r["id"] for r in data}
    return done, data


def evaluate(
    quantization: str = "none",
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    cache_dir: str = "models",
    num_samples: int = 624,
    out: str | None = None,
    longest_edge: int | None = None,
    reports_dir: str = "reports",
) -> dict:
    quant = quantization if quantization != "none" else "bf16"
    preds_path = _preds_path(quant, reports_dir)
    done, per_sample = _load_done(preds_path)

    # 只有「需要推理的样本」才加载模型；全部已完成则直接出报告
    manifest = load_manifest()
    todo = [item for item in manifest[:num_samples] if item["id"] not in done]

    loaded = {"gpu_allocated_mb": 0.0}
    footprint = 0.0
    # 先读旧结果兜底：当「全部样本已完成、跳过加载模型」时，不能把模型级显存
    # 口径覆盖成 0（否则断点续跑会污染 eval 里的 footprint/allocated 字段）。
    old_out = Path(out or f"reports/eval_{quant}.json")
    if old_out.exists():
        try:
            _old = json.loads(old_out.read_text(encoding="utf-8"))
            if _old.get("loaded_gpu_allocated_mb"):
                loaded["gpu_allocated_mb"] = _old["loaded_gpu_allocated_mb"]
            if _old.get("model_footprint_mb"):
                footprint = _old["model_footprint_mb"]
        except Exception:  # noqa: BLE001
            pass
    if todo:
        model, processor = load_model(model_name, cache_dir, quantization=quantization)
        loaded = gpu_snapshot()
        footprint = model_footprint_mb(model)

        for i, item in enumerate(todo, 1):
            img, gold = load_sample(item)
            timed = timed_inference(
                lambda: generate_text(model, processor, img, longest_edge=longest_edge)
            )
            try:
                parsed = _repair_json(timed["output_text"])
            except ValueError:
                parsed = {}  # 输出不可解析时，字段全空（算漏抽）
            pred = {f: parsed.get(f, "") for f in FIELDS}
            per_sample.append(
                {
                    "id": item["id"],
                    "gold": gold,
                    "pred": pred,
                    "latency_s": timed["latency_s"],
                    "gpu_peak_mb": timed["gpu_peak_mb"],
                }
            )
            # 每个样本落盘（断点续跑安全），写原子式（临时 + 替换）
            tmp = preds_path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(per_sample, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(preds_path)
            if i % 10 == 0:
                log.info("进度 %d/%d", i, len(todo))
    else:
        log.info("全部 %d 样本已推理，跳过加载模型", len(per_sample))

    # 汇总指标：只取「本次请求范围内的样本」——断点续跑时旧 preds_*.json 可能含
    # 超过 num_samples 的样本（例如先跑过 624 再重跑 --num_samples 50），若不按
    # manifest[:num_samples] 过滤，指标会被旧数据污染、与 num_samples 口径矛盾。
    target_ids = {item["id"] for item in manifest[:num_samples]}
    used = [r for r in per_sample if r["id"] in target_ids]
    preds = [r["pred"] for r in used]
    golds = [r["gold"] for r in used]
    latencies = [r["latency_s"] for r in used]
    peaks = [r["gpu_peak_mb"] for r in used]

    if not used:
        log.warning("无可用样本（num_samples=%d 且无已完成结果），跳过汇总", num_samples)
        return {}

    exact = field_f1(preds, golds, FIELDS, mode="exact")
    norm = field_f1(preds, golds, FIELDS, mode="normalized")
    exact_ci = bootstrap_f1(preds, golds, FIELDS, mode="exact", n_boot=1000)
    norm_ci = bootstrap_f1(preds, golds, FIELDS, mode="normalized", n_boot=1000)

    result = {
        "model": model_name,
        "quantization": quant,
        "engine": "hf-transformers",
        "num_samples": len(used),
        "longest_edge": longest_edge,
        "f1_exact": exact,
        "f1_normalized": norm,
        "f1_exact_ci": exact_ci,
        "f1_normalized_ci": norm_ci,
        "loaded_gpu_allocated_mb": loaded["gpu_allocated_mb"],
        "model_footprint_mb": footprint,
        "latency_s_mean": round(statistics.mean(latencies), 3),
        "latency_s_median": round(statistics.median(latencies), 3),
        "gpu_peak_mb_max": round(max(peaks), 1),
    }
    out = out or f"reports/eval_{quant}.json"
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("评估完成: %s", out_path)
    log.info(
        "F1 exact=%.4f [%.4f,%.4f] / norm=%.4f [%.4f,%.4f] | 延迟均值 %.2fs | 峰值显存 %.1fMB",
        exact["f1"], exact_ci["ci_low"], exact_ci["ci_high"],
        norm["f1"], norm_ci["ci_low"], norm_ci["ci_high"],
        result["latency_s_mean"],
        result["gpu_peak_mb_max"],
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="量化方法完整评估")
    ap.add_argument("--quantization", default="none", choices=["none", "bnb4"])
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache_dir", default="models")
    ap.add_argument("--num_samples", type=int, default=624)
    ap.add_argument("--longest_edge", type=int, default=None, help="约束输入最长边控显存（如 2097152=2M）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    evaluate(args.quantization, args.model, args.cache_dir, args.num_samples, args.out, args.longest_edge)


if __name__ == "__main__":
    main()
