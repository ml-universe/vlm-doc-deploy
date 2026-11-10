"""serving 层单元测试：不加载真模型，验证路由/schema（纯 CPU）。"""
from __future__ import annotations

import io

from fastapi.testclient import TestClient
from PIL import Image

from src.serving.app import ExtractResponse, app

# 注意：lifespan 里会尝试加载模型，但 TestClient 默认不触发 lifespan（除非 with 块）。
# 这里不进入 lifespan，_model 保持 None，只测路由存在与 schema 校验。


def test_extract_response_schema():
    r = ExtractResponse(company="A", date="d", address="x", total="1")
    assert r.company == "A"
    assert r.total == "1"
    # 默认值空串
    r2 = ExtractResponse()
    assert r2.company == ""
    assert r2.date == ""


def test_routes_registered():
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/health" in paths
    assert "/extract" in paths
    assert "/docs" in paths


def test_health_without_model():
    # 未加载模型时 health 应返回 model_loaded=False
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_extract_without_model_returns_503():
    client = TestClient(app)
    img = Image.new("RGB", (10, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post("/extract", files={"file": ("t.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 503
    assert "模型未加载" in resp.json()["detail"]
