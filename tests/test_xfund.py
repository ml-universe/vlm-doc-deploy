"""XFUND 加载器的纯 CPU 单元测试：linking 关系的 question/answer 配对逻辑。"""
from __future__ import annotations

from src.data.xfund import _norm, _pair_qa


def _word(wid, text, label, linking=()):
    return {"id": wid, "text": text, "label": label, "box": [0, 0, 10, 10], "linking": list(linking)}


def test_pair_qa_direct_link():
    # question 1 直接 link 到 answer 2
    doc = [
        _word(1, "姓名:", "question", [(1, 2)]),
        _word(2, "夏艳辰", "answer", [(1, 2)]),
    ]
    fields = _pair_qa(doc)
    assert fields == {"姓名": "夏艳辰"}


def test_pair_qa_multi_answer():
    # 一个 question link 多个 answer（跨行长答案）
    doc = [
        _word(1, "隐私声明:", "question", [(1, 2), (1, 3)]),
        _word(2, "内容A", "answer", [(1, 2)]),
        _word(3, "内容B", "answer", [(1, 3)]),
    ]
    fields = _pair_qa(doc)
    assert fields["隐私声明"] == "内容A内容B"


def test_pair_qa_parent_field_skipped():
    # 「电话」只 link 到 question（父字段），其子字段「手机」link 到 answer（叶子）
    doc = [
        _word(1, "电话:", "question", [(1, 2)]),       # 父，link 到 question 2
        _word(2, "手机:", "question", [(1, 2), (2, 3)]),  # 子，link 到父 + answer 3
        _word(3, "1584", "answer", [(2, 3)]),
    ]
    fields = _pair_qa(doc)
    # 父字段「电话」被跳过，只保留叶子「手机」
    assert fields == {"手机": "1584"}


def test_pair_qa_ignores_header_and_other():
    doc = [
        _word(1, "标题", "header"),
        _word(2, "正文", "other"),
        _word(3, "电话:", "question", [(3, 4)]),
        _word(4, "123", "answer", [(3, 4)]),
    ]
    fields = _pair_qa(doc)
    assert fields == {"电话": "123"}


def test_norm_strips_colon_and_space():
    assert _norm("电话:") == "电话"
    assert _norm(" 手机 ：") == "手机"
