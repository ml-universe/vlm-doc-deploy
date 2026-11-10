"""engine_hf 的纯 CPU 单元测试：JSON 修复 + 样例票据渲染（不依赖 GPU / 模型）。"""
from __future__ import annotations

from src.inference.engine_hf import FIELDS, _repair_json
from src.utils.sample_ticket import render_ticket


def test_repair_json_plain():
    assert _repair_json('{"company": "A", "date": "2026-08-25", "address": "", "total": "38.50"}') == {
        "company": "A",
        "date": "2026-08-25",
        "address": "",
        "total": "38.50",
    }


def test_repair_json_strips_code_fence():
    text = '```json\n{"company": "B", "date": "", "address": "", "total": "1.00"}\n```'
    assert _repair_json(text)["company"] == "B"


def test_repair_json_fixes_trailing_comma():
    text = '{"company": "C", "total": "9.90",}'
    assert _repair_json(text)["total"] == "9.90"


def test_repair_json_wraps_loose_text():
    text = 'Here is the result: {"company": "D", "date": "", "address": "", "total": "0"} thanks'
    assert _repair_json(text)["company"] == "D"


def test_repair_json_single_quote_keys_and_values():
    # 单引号兜底：键与值都用单引号
    got = _repair_json("{'company': 'A', 'total': '1.0'}")
    assert got == {"company": "A", "total": "1.0"}


def test_repair_json_preserves_apostrophe_in_value():
    # 值内含撇号（MCDONALD'S）不应被单引号兜底破坏
    got = _repair_json("{'company': \"MCDONALD'S\", 'total': '3.50'}")
    assert got == {"company": "MCDONALD'S", "total": "3.50"}


def test_repair_json_missing_braces_raises():
    import pytest
    with pytest.raises(ValueError):
        _repair_json("no json here")


def test_repair_json_truncated_object():
    # max_new_tokens 截断导致的未闭合 JSON：补齐 }
    got = _repair_json('{"company": "A", "total": "1.0"')
    assert got == {"company": "A", "total": "1.0"}


def test_repair_json_truncated_nested():
    got = _repair_json('{"a": {"b": "value", "c": "x"')
    assert got == {"a": {"b": "value", "c": "x"}}


def test_fields_schema():
    assert FIELDS == ["company", "date", "address", "total"]


def test_render_ticket_returns_rgb_image():
    img = render_ticket()
    assert img.mode == "RGB"
    assert img.size == (640, 800)
