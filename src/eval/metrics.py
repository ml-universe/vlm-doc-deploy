"""字段级抽取评估：exact F1 + 归一化 F1（量化精度对比的统一口径）。

================================================================================
【这个文件做什么】
    度量「预测字段 dict vs 金标字段 dict」的匹配程度，是量化实验（W2）的
    「精度」维度。与 memory.py（显存/速度）合起来构成 Pareto 三轴。

【为什么有两档 F1】
    1. exact F1：归一化后完全相等才算对，最严格。反映「能用」的比例。
    2. normalized F1：小写 + 去空白 + 去常见符号差异后再比。因为收据 OCR
       的预测常有 `RM 33.90` vs `33.90`（货币前缀）、`27,` vs `27`（标点）
       这类「语义对、字符差」的噪声，exact 会误杀。normalized 更接近人
       判断「抽对了没有」。

【字段级 F1 定义】
    把 4 个字段当 4 个槽位。对每档匹配：
      - TP = 预测非空 且 匹配金标
      - FP = 预测非空 且 不匹配（或金标为空但预测非空）
      - FN = 金标非空 且 预测为空（或预测与金标不匹配的「漏」）
    用微平均（micro）：P=TP/(TP+FP), R=TP/(TP+FN), F1=2PR/(P+R)。
    对所有样本累加 TP/FP/FN 后统一算，避免个别空样本除零。

用法::

    from src.eval.metrics import field_f1
    score = field_f1(preds=[{...}], golds=[{...}], fields=["company",...])
"""
from __future__ import annotations

import random
import re


def normalize(text: str) -> str:
    """归一化：小写、去所有空白、去货币前缀与常见标点差异。

    保留「字母数字」按 Unicode 语义（str.isalnum），中文/其它非 ASCII 字母不会被
    误删——这是跨域（XFUND 中文表单）评估正确性的关键：旧版用 [^a-z0-9] 会把中文
    全删成空串，导致中文值被误判相等。
    """
    t = text.strip().lower()
    # 去货币前缀（RM/$/€ 等，可能带空格）
    t = re.sub(r"^(rm|myr|usd|\$|€|£)\s*", "", t)
    # 去所有空白与标点，保留字母数字（含中文等 Unicode 字母数字）
    t = "".join(ch for ch in t if ch.isalnum())
    return t


def _field_match(pred: str, gold: str, mode: str) -> bool:
    if mode == "exact":
        return pred.strip() == gold.strip()
    return normalize(pred) == normalize(gold)


def field_f1(
    preds: list[dict[str, str]],
    golds: list[dict[str, str]],
    fields: list[str],
    mode: str = "exact",
) -> dict:
    """字段级 F1。preds/golds 同长，每个元素是字段 dict。

    mode: "exact"（严格）或 "normalized"（归一化软匹配）。
    返回 {precision, recall, f1, tp, fp, fn, field_acc}。
    """
    if len(preds) != len(golds):
        raise ValueError(f"preds({len(preds)}) 与 golds({len(golds)}) 长度不一致")
    tp = fp = fn = 0
    correct = total = 0
    for pred, gold in zip(preds, golds):
        for f in fields:
            p = (pred.get(f) or "").strip()
            g = (gold.get(f) or "").strip()
            total += 1
            if g == "" and p == "":
                correct += 1  # 双方都空：不算 TP 也不算错误，但计入 field_acc（稳定口径）
                continue
            if _field_match(p, g, mode):
                tp += 1
                correct += 1
            else:
                if g == "":
                    fp += 1  # 金标空但预测非空：误抽
                elif p == "":
                    fn += 1  # 金标非空但预测空：漏抽
                else:
                    fp += 1
                    fn += 1  # 预测错：既是误又是漏（槽位级别）

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "field_acc": round(correct / total, 4) if total else 0.0,
    }


def bootstrap_f1(
    preds: list[dict[str, str]],
    golds: list[dict[str, str]],
    fields: list[str],
    mode: str = "exact",
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """字段级 F1 的 bootstrap 95% 置信区间。

    对 N 个样本做有放回重采样，每次重算 micro 字段级 F1，取 2.5%/97.5% 分位数
    作为 95% CI。沿用第一个项目「bootstrap CI」的实验方法论——量化对比里
    「bnb4 掉 1.7pp」（全量 624 张）这个结论必须带 CI，否则无法判断差异是否显著。

    返回 {f1, ci_low, ci_high, n, n_boot}。
    """
    rng = random.Random(seed)
    n = len(preds)
    if n == 0:
        return {"f1": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0, "n_boot": n_boot}

    f1s: list[float] = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        bp = [preds[i] for i in idx]
        bg = [golds[i] for i in idx]
        f1s.append(field_f1(bp, bg, fields, mode)["f1"])
    f1s.sort()
    lo = f1s[int(0.025 * n_boot)]
    hi = f1s[int(0.975 * n_boot)]
    return {
        "f1": field_f1(preds, golds, fields, mode)["f1"],
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "n": n,
        "n_boot": n_boot,
    }
