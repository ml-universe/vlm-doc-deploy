"""metrics 模块单元测试：normalize / 字段 F1 / bootstrap CI。"""
from __future__ import annotations

from src.eval.metrics import bootstrap_f1, field_f1, normalize

FIELDS = ["company", "date", "address", "total"]


def test_normalize_strips_currency_and_punct():
    assert normalize("RM 33.90") == "3390"
    assert normalize("$ 38.50") == "3850"
    assert normalize("27, Jalan DEDAP") == "27jalandedap"
    assert normalize("ABC") == normalize("abc")


def test_normalize_preserves_chinese():
    # 跨域（中文表单）关键：中文字母数字不能被删成空串
    assert normalize("广东省教育厅") == "广东省教育厅"
    assert normalize("夏艳辰") == "夏艳辰"
    # 不同中文值不能因归一化误判相等
    assert normalize("广东省教育厅") != normalize("完全不同的值")


def test_normalize_chinese_punct():
    # 中文标点（全角逗号/冒号）应被去掉，但汉字保留
    assert normalize("电话：") == "电话"
    assert normalize("（2019）年") == "2019年"


def test_field_f1_perfect():
    preds = [{"company": "A", "date": "d", "address": "x", "total": "1.0"}]
    golds = [{"company": "A", "date": "d", "address": "x", "total": "1.0"}]
    s = field_f1(preds, golds, FIELDS)
    assert s["f1"] == 1.0
    assert s["precision"] == 1.0
    assert s["recall"] == 1.0


def test_field_f1_skips_both_empty():
    # 双方都空的字段不算错（SROIE 常见某字段缺失）
    preds = [{"company": "A", "date": "", "address": "", "total": ""}]
    golds = [{"company": "A", "date": "", "address": "", "total": ""}]
    s = field_f1(preds, golds, FIELDS)
    assert s["f1"] == 1.0
    assert s["tp"] == 1


def test_field_f1_wrong_is_fp_and_fn():
    preds = [{"company": "WRONG", "date": "", "address": "", "total": ""}]
    golds = [{"company": "RIGHT", "date": "", "address": "", "total": ""}]
    s = field_f1(preds, golds, FIELDS)
    assert s["tp"] == 0
    assert s["fp"] == 1
    assert s["fn"] == 1
    assert s["f1"] == 0.0


def test_bootstrap_ci_perfect_is_narrow():
    preds = [{"company": "A", "date": "d", "address": "x", "total": "1"} for _ in range(10)]
    golds = [{"company": "A", "date": "d", "address": "x", "total": "1"} for _ in range(10)]
    r = bootstrap_f1(preds, golds, FIELDS, n_boot=200)
    assert r["f1"] == 1.0
    assert r["ci_low"] == 1.0
    assert r["ci_high"] == 1.0
    assert r["n"] == 10


def test_field_f1_hallucination_is_fp_only():
    # gold 空但 pred 非空 → 误抽（只计 fp，不计 fn）
    preds = [{"company": "X", "date": "", "address": "", "total": ""}]
    golds = [{"company": "", "date": "", "address": "", "total": ""}]
    s = field_f1(preds, golds, FIELDS)
    assert s["tp"] == 0
    assert s["fp"] == 1
    assert s["fn"] == 0
    assert s["f1"] == 0.0


def test_field_f1_missing_is_fn_only():
    # gold 非空但 pred 空 → 漏抽（只计 fn，不计 fp）
    preds = [{"company": "", "date": "", "address": "", "total": ""}]
    golds = [{"company": "BOOK", "date": "", "address": "", "total": ""}]
    s = field_f1(preds, golds, FIELDS)
    assert s["tp"] == 0
    assert s["fp"] == 0
    assert s["fn"] == 1
    assert s["f1"] == 0.0


def test_field_f1_length_mismatch_raises():
    import pytest
    preds = [{"company": "A"}]
    golds = []
    with pytest.raises(ValueError):
        field_f1(preds, golds, FIELDS)


def test_bootstrap_f1_empty_input():
    r = bootstrap_f1([], [], FIELDS, n_boot=50)
    assert r["n"] == 0
    assert r["f1"] == 0.0
