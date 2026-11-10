"""SROIE 加载器的纯 CPU 单元测试（在 50 样本子集上，不依赖 GPU / 模型）。"""
from __future__ import annotations

import json

from src.data.sroie import FIELDS, load_manifest, load_sample


def test_fields_schema():
    assert FIELDS == ["company", "date", "address", "total"]


def test_load_manifest_has_samples():
    manifest = load_manifest()
    assert len(manifest) >= 50  # 至少是 subset 下载的 50 张
    first = manifest[0]
    assert set(first) == {"id", "image", "label"}
    assert first["id"] == "000"


def test_load_sample_returns_image_and_label():
    manifest = load_manifest()
    img, label = load_sample(manifest[0])
    assert img.mode == "RGB"
    assert set(label) == set(FIELDS)
    # 000.json 金标已知
    assert label["company"].startswith("BOOK TA")
    assert label["total"] == "9.00"


def test_missing_field_filled_empty(tmp_path):
    # 构造一个缺失字段的标注，验证 load_sample 补空串
    sroie_dir = tmp_path / "sroie"
    (sroie_dir / "img").mkdir(parents=True)
    (sroie_dir / "key").mkdir(parents=True)
    (sroie_dir / "key" / "999.json").write_text(
        json.dumps({"company": "X", "date": "2026-01-01"}), encoding="utf-8"
    )
    (sroie_dir / "manifest.json").write_text(
        json.dumps([{"id": "999", "image": "sroie/img/999.jpg", "label": "sroie/key/999.json"}]),
        encoding="utf-8",
    )
    from PIL import Image

    Image.new("RGB", (10, 10)).save(sroie_dir / "img" / "999.jpg")

    manifest = load_manifest(tmp_path)
    _, label = load_sample(manifest[0], tmp_path)
    assert label["address"] == ""
    assert label["total"] == ""
