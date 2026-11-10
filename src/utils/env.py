"""轻量 .env 加载（不引入 python-dotenv 依赖）。

本项目以推理为主，暂不需要外部 API key；若后续用 LLM 辅助标注/评估，
key 走 .env 并 gitignore，此处复用第一个项目的实现。
"""
from __future__ import annotations

import os
from pathlib import Path


def _project_root() -> Path | None:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "src").is_dir():
            return parent
    return None


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """加载 .env 到 os.environ，返回被设置的键值。已存在的环境变量优先。"""
    env_path = Path(path) if path else (_project_root() / ".env" if _project_root() else None)
    loaded: dict[str, str] = {}
    if env_path is None or not env_path.exists():
        return loaded

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded[key] = value
    return loaded


def get_env(key: str, default: str = "") -> str:
    """读取环境变量（自动先加载 .env）。"""
    load_dotenv()
    return os.getenv(key, default)
