"""SROIE 数据集加载：读 manifest + 图片 + 4 字段标注，统一输出结构化样本。

================================================================================
【这个文件做什么】
    给「数据 → 推理 → 评估」提供统一入口：读 data/raw/sroie/manifest.json，
    按样本 id 返回 (PIL.Image, 字段 dict)。后续 W1 推理冒烟、W2+ 量化评估、
    W4 API demo 都从这里拿数据，避免各处重复解析路径/标签。

【字段 schema】
    SROIE 任务 2 的 4 个关键字段：company（公司名）、date（日期）、
    address（地址）、total（总价）。标注是 JSON，字段值可能缺失（无该字段）。

【关键实现点】
    - manifest 里路径是 Windows 反斜杠（\），Path 在 Windows 下能正确解析；本项目仅 Windows 环境运行；
    - 缺失字段补空串，保证返回结构稳定，评估时与预测对齐。

用法::

    from src.data.sroie import load_manifest, load_sample
    manifest = load_manifest()
    img, label = load_sample(manifest[0])
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

FIELDS = ["company", "date", "address", "total"]

DEFAULT_RAW_DIR = "data/raw"


def load_manifest(raw_dir: str | Path = DEFAULT_RAW_DIR) -> list[dict]:
    """读取 data/raw/sroie/manifest.json，返回样本清单（含 image/label 路径）。"""
    manifest_path = Path(raw_dir) / "sroie" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"未找到 {manifest_path}，请先运行 py -3.11 src/data/download_sroie.py"
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve(raw_dir: Path, rel: str) -> Path:
    # manifest 里的路径可能是反斜杠（Windows），Path 能兼容处理
    return raw_dir / Path(rel)


def load_sample(
    item: dict, raw_dir: str | Path = DEFAULT_RAW_DIR
) -> tuple[Image.Image, dict[str, str]]:
    """加载单样本：返回 (图片, 字段 dict)。缺失字段补空串。"""
    raw_dir = Path(raw_dir)
    img = Image.open(_resolve(raw_dir, item["image"])).convert("RGB")
    label = json.loads(_resolve(raw_dir, item["label"]).read_text(encoding="utf-8"))
    return img, {f: label.get(f, "") for f in FIELDS}


def image_area(item: dict, raw_dir: str | Path = DEFAULT_RAW_DIR) -> int:
    """从图片头读尺寸（不加载像素），返回面积（宽×高）。用于挑最小/最大图。"""
    with Image.open(_resolve(Path(raw_dir), item["image"])) as im:
        w, h = im.size
    return w * h
