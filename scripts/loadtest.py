"""W4：FastAPI 并发压测 —— locust 脚本，压 /extract 出并发-QPS-P95 曲线。

================================================================================
【这个文件做什么】
    对 src/serving/app.py（bnb4 + /extract）做并发负载测试，回答部署核心问题：
    「8GB 卡单模型在多少并发下 QPS 到顶、P95 延迟涨到什么程度」。因为推理是
    GPU 串行执行（FastAPI 同步 endpoint + 单模型单卡，推理用锁串行化），并发请求
    会排队——压测要量化这个「排队导致的延迟上升 + 吞吐饱和」的曲线。

【为什么用 locust 而不是 wrk】
    wrk 在 Windows 上需要编译；locust 纯 Python，pip 装完即用，且自带 Web UI
    （--web-host）与 headless（--headless）两种模式，报告 CSV 便于画曲线。

【关键设计】
    - 每个 locust user 上传一张真实 SROIE 票据图到 /extract，校验返回 JSON；
    - 分多档并发（spawn rate / user 数）跑，记录 QPS + P50/P95/P99 延迟；
    - 用固定小图（避免大扫描件 60s 拉高方差），或标注清楚图类型。

【用法（headless）】
    先启动服务：py -3.11 -m src.serving.app
    再压测：       locust -f scripts/loadtest.py --headless -u 4 -r 1 -t 60s \
                       --host http://127.0.0.1:8000 --csv=reports/w4_loadtest \
                       --html=reports/w4_loadtest.html

    分档建议手动跑多组 -u 1/2/4/8，记录各自 QPS/P95，再手工汇总成曲线。
"""
from __future__ import annotations

from pathlib import Path

from locust import HttpUser, between, task

# 固定用一张 SROIE 真实票据图压测（小图，视觉 token 少、延迟稳定）
_DEFAULT_IMAGE = "data/raw/sroie/img/000.jpg"


class ExtractUser(HttpUser):
    """模拟客户端：上传票据图 → 等 /extract 返回字段 JSON。"""

    wait_time = between(0.1, 0.5)  # 请求间隔，模拟真实用户节奏

    @task
    def extract(self) -> None:
        img_path = Path(_DEFAULT_IMAGE)
        if not img_path.exists():
            # 兜底：找不到固定图就换 manifest 里第一张
            from src.data.sroie import load_manifest, load_sample

            img = load_sample(load_manifest()[0])[0]
            import io

            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            data = buf.getvalue()
        else:
            data = img_path.read_bytes()

        with self.client.post(
            "/extract",
            files={"file": ("ticket.jpg", data, "image/jpeg")},
            timeout=120,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:200]}")
                return
            body = resp.json()
            # 校验 schema 完整（4 字段都在）
            for k in ("company", "date", "address", "total"):
                if k not in body:
                    resp.failure(f"响应缺字段 {k}")
                    return
            resp.success()
