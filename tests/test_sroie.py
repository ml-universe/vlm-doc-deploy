"""SROIE 加载器的纯 CPU 单元测试（在 50 样本子集上，不依赖 GPU / 模型）。"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.data.sroie import DEFAULT_RAW_DIR, FIELDS, load_manifest, load_sample

# 仓库不含 data/（.gitignore 排除），纯克隆跑测试时这两个用例会因缺 manifest 而失败。
# 没有数据就显式 skip，而不是红一片——需要真跑时先下载：
#   py -3.11 -m src.data.download_sroie
SROIE_MANIFEST = Path(DEFAULT_RAW_DIR) / "sroie" / "manifest.json"
needs_sroie_data = pytest.mark.skipif(
    not SROIE_MANIFEST.exists(),
    reason=f"缺少 {SROIE_MANIFEST}；先运行 py -3.11 -m src.data.download_sroie",
)


def test_fields_schema():
    assert FIELDS == ["company", "date", "address", "total"]


@needs_sroie_data
def test_load_manifest_has_samples():
    manifest = load_manifest()
    assert len(manifest) >= 50  # 至少是 subset 下载的 50 张
    first = manifest[0]
    assert set(first) == {"id", "image", "label"}
    assert first["id"] == "000"


@needs_sroie_data
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
