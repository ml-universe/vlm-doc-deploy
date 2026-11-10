"""字段级错误分析：把抽取错误归类为「漏抽/误抽/字符噪声/语义错抽」。

================================================================================
【这个文件做什么】
    量化评估的「诚实性」补足——F1 是一个数字，但不告诉你「错在哪」。这个模块
    把每个字段的 pred vs gold 对比归类到 5 类，输出典型错例，让报告能讲清
    「误差来源」而非只报一个分。延续第一个项目 error_analysis 的方法论。

【错误分类（字段级，按严重度从轻到重）】
    1. correct   ：exact 相等（完全对）
    2. noise     ：exact 不等但 normalize 后相等 —— 货币前缀（RM/$）、标点、空白、
                   大小写等「语义对、字符差」的 OCR 噪声，人看是「抽对了」；
    3. missing   ：gold 非空但 pred 空 —— 漏抽（recall 损失）；
    4. hallucination：gold 空但 pred 非空 —— 误抽/幻觉（precision 损失）；
    5. wrong     ：两者都非空但 normalize 也不等 —— 真·语义错误。

【为什么这对项目重要】
    讲「量化损失 1.7pp（全量 624 张）」时，评审会问「那错的都是什么？」。能答出
    「大部分是收据 OCR 的货币前缀/标点噪声（exact 0.83 → normalize 0.93 的差距），
    真正的语义错误只占 ~5-7%、幻觉仅 0.08%」，比只会报数字高一个层次——体现你真正
    看过数据、做过分析。

用法::

    from src.eval.error_analysis import analyze_errors
    report = analyze_errors(preds, golds, fields)
"""
from __future__ import annotations

from collections import Counter

from src.eval.metrics import normalize

# 分类标签
CORRECT = "correct"
NOISE = "noise"
MISSING = "missing"
HALLUCINATION = "hallucination"
WRONG = "wrong"


def classify_field(pred: str, gold: str) -> str:
    """单字段分类。pred/gold 为原始字符串（未 trim）。"""
    p = (pred or "").strip()
    g = (gold or "").strip()
    if p == g:
        return CORRECT
    if g == "" and p != "":
        return HALLUCINATION
    if p == "" and g != "":
        return MISSING
    if normalize(p) == normalize(g):
        return NOISE
    return WRONG


def analyze_errors(
    preds: list[dict[str, str]],
    golds: list[dict[str, str]],
    fields: list[str],
) -> dict:
    """返回错误分析报告：总体分布 + 分字段分布 + 典型错例（每类最多 5 条）。

    preds/golds 同长，每个元素是字段 dict。
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds({len(preds)}) 与 golds({len(golds)}) 长度不一致")
    overall = Counter()
    per_field = {f: Counter() for f in fields}
    examples: dict[str, list[dict]] = {k: [] for k in [WRONG, HALLUCINATION, MISSING, NOISE]}

    for idx, (pred, gold) in enumerate(zip(preds, golds)):
        for f in fields:
            p = pred.get(f) or ""
            g = gold.get(f) or ""
            cat = classify_field(p, g)
            overall[cat] += 1
            per_field[f][cat] += 1
            # 记录错例（前 5 条）
            if cat in examples and len(examples[cat]) < 5:
                examples[cat].append({"sample_idx": idx, "field": f, "pred": p, "gold": g})

    total = sum(overall.values())
    return {
        "overall_counts": dict(overall),
        "overall_ratio": {k: round(v / total, 4) for k, v in overall.items()} if total else {},
        "total_fields": total,
        "per_field": {f: dict(c) for f, c in per_field.items()},
        "examples": examples,
    }
