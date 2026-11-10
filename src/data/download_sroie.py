"""下载 SROIE（ICDAR2019）数据集到本地，绕开 HuggingFace。

================================================================================
【这个文件做什么】
    从 GitHub 镜像（zzzDavid/ICDAR-2019-SROIE）下载 SROIE 收据数据集，
    只提取本项目需要的两个子目录：
      - data/img/*.jpg   —— 624 张扫描收据图
      - data/key/*.json  —— 对应 4 字段标注（company/date/address/total）
    并生成 manifest.json（样本清单），供后续推理/评估统一读取。

【为什么用 GitHub 镜像而不是 HF datasets】
    SROIE 原始在 HuggingFace datasets（rth/sroie-2019-v2），但 HF 被墙。
    GitHub 镜像 zzzDavid/ICDAR-2019-SROIE 国内可达（api.github.com /
    codeload.github.com 可用，raw.githubusercontent.com 被墙），且已修正原版
    标注错误。这是「工程要适配真实网络环境」的又一体现。

【为什么用 codeload zip 而不是 git clone / 逐个文件】
    - git clone 走 git:// 协议被墙（invalid index-pack）；
    - raw.githubusercontent.com 被墙，逐个 curl 624 文件太慢；
    - codeload.github.com 的 zip 端点可达，一次下完整个仓库压缩包，
      再本地解压只取 img + key，最快最稳。

【下载后的目录布局】
    data/raw/sroie/
      ├── img/           # 624 张收据图（X.jpg）
      ├── key/           # 624 个字段标注（X.json）
      └── manifest.json  # [{"id": "000", "image": ..., "label": ...}, ...]

用法::

    py -3.11 src/data/download_sroie.py            # 下载完整 624 张
    py -3.11 src/data/download_sroie.py --subset 50   # 只保留前 50 张做冒烟
"""
from __future__ import annotations

import argparse
import io
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

REPO = "zzzDavid/ICDAR-2019-SROIE"
ZIP_URL = f"https://codeload.github.com/{REPO}/zip/refs/heads/master"


def _download_zip(url: str, timeout: int = 600) -> bytes:
    """下载 zip 到内存（~25MB），带 UA 头规避部分 CDN 拦截。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _extract_to(zf: zipfile.ZipFile, name: str, dest: Path, retries: int = 3) -> None:
    """从 zip 直接读字节写到 dest，规避 Windows 下 extract+move 的文件锁问题。

    直接 zf.open(name) 读、显式 open(dest,"wb") 写并立刻关闭，不再经临时目录
    移动。写文件偶发被系统实时防护短暂锁定（WinError 32），加少量重试。
    """
    for attempt in range(retries):
        try:
            with zf.open(name) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(0.5)


def download_sroie(raw_dir: str | Path = "data/raw", subset: int | None = None) -> Path:
    """下载并解压 SROIE 的 img + key 到 raw_dir/sroie/，返回 sroie 目录。"""
    raw_dir = Path(raw_dir)
    sroie_dir = raw_dir / "sroie"
    img_dir = sroie_dir / "img"
    key_dir = sroie_dir / "key"
    img_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)

    log.info("下载 SROIE 仓库压缩包: %s", ZIP_URL)
    data = _download_zip(ZIP_URL)

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        # 仓库顶层文件夹名形如 ICDAR-2019-SROIE-master/
        prefix = ""
        for n in zf.namelist():
            if n.endswith("/data/img/000.jpg"):
                prefix = n[: -len("data/img/000.jpg")]
                break
        log.info("zip 内前缀: %r", prefix)

        img_names = sorted(n for n in zf.namelist() if n.startswith(prefix + "data/img/") and n.endswith(".jpg"))
        key_names = sorted(n for n in zf.namelist() if n.startswith(prefix + "data/key/") and n.endswith(".json"))
        # 基准固定 624 张（原始 626 张剔除 2 个计时异常样本 017/018，与全项目口径一致）
        img_names = [n for n in img_names if Path(n).stem not in {"017", "018"}]
        key_names = [n for n in key_names if Path(n).stem not in {"017", "018"}]

        if subset is not None:
            img_names = img_names[:subset]
            key_names = key_names[:subset]
        log.info("提取 %d 张图片 / %d 个标注", len(img_names), len(key_names))

        # 收据编号即文件名（000.jpg -> 000），直接解到最终目录，不经临时目录移动
        for n in img_names:
            _extract_to(zf, n, img_dir / Path(n).name)
        for n in key_names:
            _extract_to(zf, n, key_dir / Path(n).name)

    # 生成 manifest.json：只依据本次实际解压的 img_names（而非 glob 目录），
    # 避免 --subset 重跑时被目录里残留的历史图片污染（subset 语义失效）。
    manifest: list[dict] = []
    for n in sorted(img_names):
        name = Path(n).name
        sid = Path(name).stem
        key_path = key_dir / f"{sid}.json"
        manifest.append(
            {
                "id": sid,
                "image": str((img_dir / name).relative_to(raw_dir)),
                "label": str(key_path.relative_to(raw_dir)),
            }
        )
    (sroie_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log.info("SROIE 下载完成: %s（%d 样本）", sroie_dir, len(manifest))
    return sroie_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="下载 SROIE 数据集")
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--subset", type=int, default=None, help="只下载前 N 张（冒烟用）")
    args = ap.parse_args()
    download_sroie(args.raw_dir, args.subset)


if __name__ == "__main__":
    main()
