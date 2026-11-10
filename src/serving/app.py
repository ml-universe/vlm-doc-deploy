"""FastAPI 服务骨架：加载量化模型，暴露 /extract 接口做票据字段抽取。

================================================================================
【这个文件做什么】
    W4 部署的起点：把推理引擎（src.inference.engine_hf）包成一个 HTTP API。
    启动时加载模型（默认 bnb4 量化），客户端上传票据图，返回字段 JSON。

【为什么用 FastAPI】
    - 原生 async + 自动 OpenAPI 文档（/docs），Pydantic 校验请求/响应；
    - 与后续 W4 压测（wrk/locust）直接对接，量化模型 + API 端到端。

【关键设计】
    1. 模型全局单例：启动时加载一次，所有请求复用，避免每请求重载模型；
    2. 懒加载 /startup 事件，用 lifespan 管理生命周期（FastAPI 推荐）；
    3. 响应用 Pydantic model 定义 schema（company/date/address/total），
       与 SROIE 字段对齐，便于前端/压测工具解析。

【讲解要点】
    「模型服务化」是 MLSys 的最后一公里：把离线推理包成 API，才能讲并发、
    吞吐、SLO。这个文件是 W4 压测的载体。

用法::

    py -3.11 -m src.serving.app          # 默认 bnb4 量化，端口 8000
    # 或 uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
    # 测试：curl -X POST -F "file=@data/raw/sroie/img/000.jpg" http://127.0.0.1:8000/extract
"""
from __future__ import annotations

import io
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from src.inference.engine_hf import extract_fields, load_model
from src.utils.logger import get_logger

log = get_logger(__name__)


class ExtractResponse(BaseModel):
    """票据字段抽取响应，与 SROIE 4 字段对齐。"""

    company: str = ""
    date: str = ""
    address: str = ""
    total: str = ""


# 模型全局单例（启动时加载，请求间复用）
_model = None
_processor = None
_LONGEST_EDGE = 2097152  # 2M，控大图视觉 token，与基准一致

# 单模型单卡：推理必须串行。并发直接 forward 会显存叠加（8GB 卡 2 并发即 OOM），
# 用锁串行化，并发请求在锁上排队而非阻塞事件循环（sync endpoint 被 FastAPI 丢进 threadpool）。
_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动加载模型；关闭时仅记日志（显存随进程退出由系统回收）。"""
    global _model, _processor
    log.info("加载模型（bnb4 量化）…")
    _model, _processor = load_model(
        "Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="models", quantization="bnb4"
    )
    log.info("模型加载完成，服务就绪")
    yield
    log.info("服务关闭")


app = FastAPI(title="vlm-doc-deploy", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """健康检查，压测前确认服务存活。"""
    return {"status": "ok", "model_loaded": _model is not None}


@app.post("/extract", response_model=ExtractResponse)
def extract(file: UploadFile = File(...)) -> ExtractResponse:
    """上传票据图，返回抽取出的 4 字段 JSON。

    同步 endpoint（不是 async）：推理是阻塞的 GPU 计算，若用 async def 会在事件循环上
    阻塞，导致推理期间连 /health 都无响应。sync def 会被 FastAPI 自动丢进 threadpool，
    事件循环不被占住；推理本身用 _inference_lock 串行（单模型单卡，并发 forward 会
    显存叠加 OOM）。
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    data = file.file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"无法解析图片: {exc}") from exc

    with _inference_lock:
        try:
            fields = extract_fields(_model, _processor, img, longest_edge=_LONGEST_EDGE)
        except Exception as exc:  # noqa: BLE001
            log.error("抽取失败: %s", exc)
            raise HTTPException(status_code=500, detail="推理失败") from exc

    return ExtractResponse(**fields)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
