"""W1 冒烟测试：原生推理跑通 + SROIE 真实样本抽取验证。

================================================================================
【这个文件做什么】
    W1 的最终验收脚本，一次性验证两件事：
      1. 合成票据图 → Qwen2.5-VL 原生推理 → 输出 company/date/address/total；
      2. SROIE 真实收据（前 N 张）→ 原生推理 → 对比金标，粗看是否端到端可用。
    全部结果落到 reports/w1_smoke.json（机器可读）+ 控制台打印。

【为什么先做「冒烟」而不是直接上指标】
    量化/引擎/并发那些优化（W2-W4）都要以「FP16 原生推理能跑、能出对」为前提。
    冒烟测试先证明这条链路通，再谈优化才有意义。它不需要严格 F1（那是 W2 的
    eval 模块的事），只需要「喂图能吐出合理字段」。

【讲解要点】
    「先冒烟、后基准」是工程节奏——避免在坏掉的基线上做一堆优化实验。

用法::

    py -3.11 scripts/run_w1_smoke.py                 # 合成图 + SROIE 前 5 张
    py -3.11 scripts/run_w1_smoke.py --num_sroie 20 --model Qwen/Qwen2.5-VL-3B-Instruct
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.sroie import load_manifest, load_sample
from src.inference.engine_hf import extract_fields, load_model
from src.utils.logger import get_logger
from src.utils.sample_ticket import render_ticket

log = get_logger(__name__)


def run_smoke(
    model_name: str,
    cache_dir: str = "models",
    num_sroie: int = 5,
    out: str = "reports/w1_smoke.json",
) -> dict:
    """跑 W1 冒烟，返回结果 dict（合成图 + SROIE 样本各若干）。"""
    model, processor = load_model(model_name, cache_dir)

    results: dict = {"model": model_name, "synthetic": None, "sroie": []}

    # 1) 合成票据图
    log.info("[1/2] 合成票据图 → 原生推理")
    syn_img = render_ticket()
    syn_pred = extract_fields(model, processor, syn_img)
    results["synthetic"] = {
        "gold": {
            "company": "STARBUCKS COFFEE",
            "date": "2026-08-25",
            "address": "88 West Lake Ave, Hangzhou",
            "total": "38.50",
        },
        "pred": syn_pred,
    }
    log.info("合成图预测: %s", json.dumps(syn_pred, ensure_ascii=False))

    # 2) SROIE 真实样本
    manifest = load_manifest()
    log.info("[2/2] SROIE 真实样本（前 %d 张）→ 原生推理", num_sroie)
    for item in manifest[:num_sroie]:
        img, gold = load_sample(item)
        pred = extract_fields(model, processor, img)
        results["sroie"].append({"id": item["id"], "gold": gold, "pred": pred})
        log.info("SROIE %s 预测: %s", item["id"], json.dumps(pred, ensure_ascii=False))

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("结果已保存: %s", out_path)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="W1 冒烟测试")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache_dir", default="models")
    ap.add_argument("--num_sroie", type=int, default=5)
    ap.add_argument("--out", default="reports/w1_smoke.json")
    args = ap.parse_args()
    run_smoke(args.model, args.cache_dir, args.num_sroie, args.out)


if __name__ == "__main__":
    main()
