"""error_analysis 模块单元测试：字段错误分类。"""
from __future__ import annotations

from src.eval.error_analysis import (
    HALLUCINATION,
    MISSING,
    NOISE,
    WRONG,
    analyze_errors,
    classify_field,
)

FIELDS = ["company", "date", "address", "total"]


def test_classify_correct():
    assert classify_field("ABC", "ABC") == "correct"


def test_classify_noise_currency():
    assert classify_field("RM 33.90", "33.90") == NOISE
    assert classify_field("$ 38.50", "38.50") == NOISE


def test_classify_noise_punct():
    assert classify_field("27, Jalan", "27 Jalan") == NOISE


def test_classify_missing():
    assert classify_field("", "BOOK TA") == MISSING


def test_classify_hallucination():
    assert classify_field("WRONG", "") == HALLUCINATION


def test_classify_wrong():
    assert classify_field("ABC", "XYZ") == WRONG


def test_analyze_errors_distribution():
    preds = [
        {"company": "A", "date": "d", "address": "x", "total": "RM 1.0"},  # total 噪声
        {"company": "", "date": "d", "address": "x", "total": "1"},  # company 漏抽
    ]
    golds = [
        {"company": "A", "date": "d", "address": "x", "total": "1.0"},
        {"company": "BOOK", "date": "d", "address": "x", "total": "1"},
    ]
    r = analyze_errors(preds, golds, FIELDS)
    assert r["overall_counts"][NOISE] == 1
    assert r["overall_counts"][MISSING] == 1
    assert r["total_fields"] == 8


def test_analyze_errors_examples_capped_at_five():
    # 构造 6 个 wrong 错例，验证每类 examples 最多保留 5 条
    preds = [{"company": f"WRONG{i}", "date": "", "address": "", "total": ""} for i in range(6)]
    golds = [{"company": f"RIGHT{i}", "date": "", "address": "", "total": ""} for i in range(6)]
    r = analyze_errors(preds, golds, FIELDS)
    assert len(r["examples"][WRONG]) == 5
    assert r["overall_counts"][WRONG] == 6


def test_analyze_errors_length_mismatch_raises():
    import pytest
    with pytest.raises(ValueError):
        analyze_errors([{"company": "A"}], [], FIELDS)
