# W4 FastAPI 部署压测 —— 并发-QPS-P95 曲线 + 端到端 demo

> 日期：2026-08-26 · 模型 Qwen2.5-VL-3B bnb4 · 引擎 HF Transformers · 服务 FastAPI + uvicorn

---

## 一、服务形态

`src/serving/app.py`：FastAPI + bnb4 模型全局单例（启动加载一次）+ `POST /extract`（上传票据图 → 4 字段 JSON）+ `GET /health`。

**关键事实**：`/extract` 内的推理是**同步执行**（HF `generate()` 阻塞），且模型是**单实例**——
这是理解下面「并发不涨吞吐、只涨延迟」的根源。

---

## 二、端到端 demo（真实票据 → JSON）

| 图片 | company | date | total | 与金标 |
|---|---|---|---|---|
| `img/000.jpg` | BOOK TA.K(TAMAN DAYA) SDN BHD | 25/12/2018 | 9.00 | ✅（company 仅 OCR 空格差异 "TA .K" vs "TA.K"） |
| `img/001.jpg` | INDAH GIFT & HOME DECO | 19/10/2018 | 60.30 | ✅ |

```bash
curl -X POST -F "file=@data/raw/sroie/img/000.jpg" http://127.0.0.1:8000/extract
# → {"company":"BOOK TA.K(TAMAN DAYA) SDN BHD","date":"25/12/2018",
#    "address":"NO.53 55,57 & 59, JALAN SAGU 18, TAMAN DAYA, 81100 JOHOR BAHRU, JOHOR.",
#    "total":"9.00"}
```

---

## 三、并发压测结果（locust · 固定小图 img/000.jpg）

| 并发 u | QPS(样本/s) | 平均延迟(s) | P95(s) | P99(s) |
|---|---|---|---|---|
| 1 | 0.18 | 5.1 | 5.9 | 5.9 |
| 2 | 0.22 | 8.2 | 9.1 | 9.1 |
| 4 | 0.22 | 15.7 | 18.0 | 18.0 |

> 单图延迟 ~5s（小票），与 W2 基准一致。压测 0 失败（failures 全 0）。

---

## 四、结论：8GB 单模型并发上限

- **QPS 几乎不涨**（0.18 → 0.22），**P95 随并发线性涨**（5.9 → 9.1 → 18s）——这是
  「单模型 + 同步推理」的典型排队曲线：并发请求不能并行执行，只能在 GPU 前排队，
  所以吞吐不提升、延迟反而被排队拉高。
- **吞吐上限由「单张推理延迟」决定**（≈1/5s = 0.2 样本/s），并发改变不了它；
  要提升吞吐只有两条路：① **静态批处理**（W3-1：batch2 吞吐 ×1.28）② **连续批处理**
  （vLLM PagedAttention，本机装不了，见 W2 工具链约束）。
- **生产启示**：8GB 边缘卡单模型服务的真实约束是「吞吐饱和于单卡算力」，
  并发压测的价值不是「测出更高 QPS」，而是**量化「并发只恶化延迟、不换吞吐」**——
  这决定了「要不要上批处理 / 要不要上多卡」。

---

## 五、讲解要点

- **为什么并发不涨吞吐**：HF 同步 `generate()` + 单模型实例 → 请求串行执行，FastAPI
  的 async 在这里**挡不住**（GPU 推理阻塞事件循环），并发只能排队。
- **P95 是部署 SLO 的关键**：压测要看 P95/P99 而非均值——并发 4 时 P95 已 18s，
  若设「P95 ≤ 10s」的 SLO，并发上限就是 2，这是容量规划的硬结论。
- **批处理 vs 并发**：W3-1 的静态批处理（同请求内部凑批）提吞吐 ×1.28，而这里的
  HTTP 并发（跨请求）只涨延迟——两者区别是「GPU 一次 forward 算几张图」vs
  「几个请求同时等着用 GPU」。
- 压测工具选型：locust 纯 Python、Windows 免编译（wrk 需编译），`--csv` 落盘可复现。

---

## 附：复现命令

```bash
python -m src.serving.app                                   # 启动服务（8000）
PYTHONUTF8=1 locust -f scripts/loadtest.py --headless \
  -u 1 -r 1 -t 30s --host http://127.0.0.1:8000 --csv=reports/w4_loadtest_u1
# 分档跑 -u 1/2/4，汇总成上面的并发曲线
```

> 注：locust 读 `pyproject.toml` 时在 Windows 下会按 GBK 解码中文 `description` 报错，
> 需加 `PYTHONUTF8=1` 强制 UTF-8（本项目环境坑，已记）。
