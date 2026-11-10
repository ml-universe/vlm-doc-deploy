"""生成一张合成收据图片，用于 W1 冒烟测试（无需等待 SROIE 数据）。

================================================================================
【这个文件做什么】
    在没有真实 SROIE 数据前，用 PIL 画一张含「公司名/日期/地址/总价」的收据图，
    让原生推理「喂一张图 → 输出字段 JSON」这条链路先端到端跑通。

【为什么用合成图而不是网上随便找】
    1. 字段完全已知（company/date/address/total），可精确核对模型抽取对不对；
    2. 不依赖网络、不依赖 HF，秒级生成、可复现；
    3. 后续 SROIE 真实数据到位后，推理入口不变，只换输入图片。

用法::

    py -3.11 src/utils/sample_ticket.py --out data/raw/sample_ticket.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_font(size: int):
    """尝试加载一个中文字体；失败则回退默认字体（英文也能看）。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
        "C:/Windows/Fonts/simhei.ttf",    # 黑体
        "C:/Windows/Fonts/simsun.ttc",    # 宋体
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                continue
    return ImageFont.load_default()


def render_ticket(
    company: str = "STARBUCKS COFFEE",
    date: str = "2026-08-25",
    address: str = "88 West Lake Ave, Hangzhou",
    total: str = "38.50",
) -> Image.Image:
    """渲染一张 640x800 的收据图，返回 PIL.Image。"""
    W, H = 640, 800
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    f_title = _load_font(34)
    f_body = _load_font(24)
    f_big = _load_font(40)

    # 顶部公司名 + 分隔线
    draw.text((W // 2, 80), company, fill="black", anchor="mm", font=f_title)
    draw.line((60, 140, W - 60, 140), fill="black", width=2)

    # 字段行（标签 + 值），刻意留出「字段可抽取」的结构
    rows = [
        ("DATE", date),
        ("ADDRESS", address),
        ("", ""),
        ("TOTAL", total),
    ]
    y = 200
    for label, value in rows:
        if label:
            draw.text((80, y), f"{label}:", fill="black", font=f_body)
            draw.text((240, y), value, fill="black", font=f_body)
        y += 70

    # 底部 TOTAL 用大字体强调
    draw.text((W // 2, 640), "TOTAL", fill="black", anchor="mm", font=f_title)
    draw.text((W // 2, 700), f"$ {total}", fill="black", anchor="mm", font=f_big)

    return img


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成收据图")
    ap.add_argument("--out", default="data/raw/sample_ticket.png")
    ap.add_argument("--company", default="STARBUCKS COFFEE")
    ap.add_argument("--date", default="2026-08-25")
    ap.add_argument("--address", default="88 West Lake Ave, Hangzhou")
    ap.add_argument("--total", default="38.50")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    img = render_ticket(args.company, args.date, args.address, args.total)
    img.save(out)
    print(f"样例收据已生成: {out}")


if __name__ == "__main__":
    main()
