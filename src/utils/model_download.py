"""模型下载：优先本地路径，否则从 ModelScope 下载（规避 HuggingFace 被墙）。

================================================================================
【这个文件做什么】
    统一解决「模型从哪来」的问题。加载模型前先调 resolve_model，得到「一个本地
    可用的模型路径」。

【为什么需要它（国内环境的现实约束）】
    HuggingFace 在国内被墙（HTTP 000），直接 from_pretrained("Qwen/...") 会卡死。
    ModelScope（阿里魔搭）国内可达。所以：
      - 若 name 是已存在的本地目录 → 直接用（离线、最快）；
      - 否则 → 用 ModelScope 下载到 cache_dir，返回本地缓存路径；
      - 兜底 → 返回原 name 交给 transformers（万一 HF 通了）。

【讲解要点】
    体现「工程要适配真实环境」——知道 HF 在国内不可用，主动走 ModelScope，
    是工程细节里能体现「踩过坑、有经验」的点。

用法::

    py -3.11 src/utils/model_download.py --model Qwen/Qwen2.5-VL-3B-Instruct --cache_dir models
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.env import _project_root
from src.utils.logger import get_logger

log = get_logger(__name__)


def _local_cache_snapshot(name: str, cache_dir: str | None) -> Path | None:
    """在 ModelScope 本地缓存布局中查找已下载快照（离线可用）。

    ModelScope 的缓存布局为::

        <cache_dir>/models/<org>--<model>/snapshots/<revision>/

    同一模型可能有多个 revision 目录，取修改时间最新的一个。
    找到则直接返回，避免离线时 snapshot_download 发网络请求失败。
    """
    if not cache_dir or "/" not in name:
        return None
    org, _, model = name.partition("/")
    base = Path(cache_dir)
    if not base.is_absolute():
        root = _project_root()
        if root is not None:
            base = root / base
    snaps = base / "models" / f"{org}--{model}" / "snapshots"
    if not snaps.is_dir():
        return None
    revs = [p for p in snaps.iterdir() if p.is_dir()]
    if not revs:
        return None
    return max(revs, key=lambda p: p.stat().st_mtime)


def resolve_model(name: str, cache_dir: str | None = None) -> str:
    """返回可被 transformers 加载的本地路径。

    - 若 ``name`` 是已存在的目录（绝对/相对当前 cwd）→ 直接返回
    - 否则尝试相对项目根目录解析（便于跨 cwd 运行）
    - 再查 ModelScope 本地缓存布局（离线时直接用，不联网）
    - 再尝试用 ModelScope 下载 → 返回本地缓存路径
    - 兜底返回原 ``name``（交给 transformers 尝试 HF）
    """
    if Path(name).exists():
        return str(Path(name).resolve())
    root = _project_root()
    if root and (root / name).exists():
        return str((root / name).resolve())
    cached = _local_cache_snapshot(name, cache_dir)
    if cached is not None:
        return str(cached.resolve())
    try:
        from modelscope import snapshot_download

        log.info("从 ModelScope 下载模型: %s", name)
        return snapshot_download(name, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("ModelScope 下载失败(%s)，回退到原 id 加载", exc)
        return name


def main() -> None:
    ap = argparse.ArgumentParser(description="从 ModelScope 下载模型")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--cache_dir", default="models")
    args = ap.parse_args()
    path = resolve_model(args.model, args.cache_dir)
    print(f"模型路径: {path}")


if __name__ == "__main__":
    main()
