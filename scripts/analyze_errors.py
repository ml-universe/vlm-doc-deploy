"""读 per-sample 推理结果，输出错误分析报告（Markdown）。

================================================================================
【这个文件做什么】
    读 reports/preds_<quant>.json（evaluate_quant.py 的落盘），对 pred vs gold
    做字段级错误归类，输出 reports/error_analysis_<quant>.md，含：
      1. 总体错误分布（5 类）；
      2. 分字段错误分布（哪个字段最容易错）；
      3. 每类错误的典型错例（最多 5 条）。

【为什么读 preds_*.json 而不是重跑推理】
    推理很慢（全量 624 张 bf16 要 3h），错误分析是纯 CPU 的字符串比对，
    秒级完成，直接复用已落盘的每样本结果即可。

用法::

    py -3.11 -m scripts.analyze_errors --quant bnb4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.sroie import FIELDS
from src.eval.error_analysis import (
    HALLUCINATION,
    MISSING,
    NOISE,
    WRONG,
    analyze_errors,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def _fmt_examples(examples: dict) -> str:
    lines = []
    for cat, label in [(WRONG, "语义错抽"), (HALLUCINATION, "误抽"), (MISSING, "漏抽"), (NOISE, "噪声")]:
        lines.append(f"### {label}")
        if not examples.get(cat):
            lines.append("（无）")
            continue
        for e in examples[cat]:
            lines.append(
                f"- `{e['field']}`: pred=`{e['pred'][:40]}` ← gold=`{e['gold'][:40]}`"
            )
        lines.append("")
    return "\n".join(lines)


def build_report(quant: str, preds_path: Path) -> str:
    data = json.loads(preds_path.read_text(encoding="utf-8"))
    preds = [r["pred"] for r in data]
    golds = [r["gold"] for r in data]
    result = analyze_errors(preds, golds, FIELDS)

    lines = [
        f"# 错误分析报告 —— {quant}",
        "",
        f"样本数：{len(data)} · 字段数：{result['total_fields']}",
        "",
        "## 总体分布",
        "",
        "| 类别 | 数量 | 占比 |",
        "|---|---|---|",
    ]
    for cat in [WRONG, HALLUCINATION, MISSING, NOISE, "correct"]:
        cnt = result["overall_counts"].get(cat, 0)
        ratio = result["overall_ratio"].get(cat, 0.0)
        lines.append(f"| {cat} | {cnt} | {ratio:.2%} |")

    lines += [
        "",
        "## 分字段分布",
        "",
        "| 字段 | 完全正确 | 噪声 | 漏抽 | 误抽 | 语义错抽 |",
        "|---|---|---|---|---|---|",
    ]
    for f in FIELDS:
        c = result["per_field"][f]
        lines.append(
            f"| {f} | {c.get('correct', 0)} | {c.get(NOISE, 0)} | {c.get(MISSING, 0)} "
            f"| {c.get(HALLUCINATION, 0)} | {c.get(WRONG, 0)} |"
        )

    lines += ["", "## 典型错例", "", _fmt_examples(result["examples"])]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="错误分析报告")
    ap.add_argument("--quant", default="bnb4")
    ap.add_argument("--preds", default=None, help="preds json 路径，默认 reports/preds_<quant>.json")
    args = ap.parse_args()

    preds_path = Path(args.preds) if args.preds else Path(f"reports/preds_{args.quant}.json")
    if not preds_path.exists():
        log.error("未找到 %s，先运行 evaluate_quant.py 生成", preds_path)
        raise SystemExit(1)

    report = build_report(args.quant, preds_path)
    out = Path("reports") / f"error_analysis_{args.quant}.md"
    out.write_text(report + "\n", encoding="utf-8")
    log.info("错误分析已保存: %s", out)
    print(report)


if __name__ == "__main__":
    main()
