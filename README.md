# vlm-doc-deploy — 多模态文档理解模型边缘部署

> 用 **Qwen2.5-VL** 做文档/票据/表单的**结构化抽取**，并把它**部署优化到 8GB 卡**上跑得快、准、能上线。

这是姜涛的**第二个科研实践项目**，与第一个项目 SciTriple 拼成「非结构化 → 结构化信息抽取」主线：SciTriple 做文本微调（训练/研究），本项目做图片抽取 + 部署优化（系统/工程，MLSys 方向）。

## 一句话定位

图片（票据/表单/收据） → 字段 JSON（company / date / total …），聚焦**推理部署优化**而非训练。

## 技术选型

| 环节 | 主力（已落地） | 对照（展望/工具链约束记录） |
|---|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct（ModelScope 下载） | Qwen2.5-VL-7B（展望，未跑） |
| 量化 | **bitsandbytes NF4 4bit**（唯一跑通） | vLLM AWQ / autoawq / FP8 / GGUF（Windows+Blackwell 装不了） |
| 推理引擎 | **HF Transformers**（静态批处理 + KV 量化） | vLLM（PagedAttention + 连续批处理，装不了） |
| 部署 | FastAPI + 压测（locust） | — |
| 数据集 | SROIE（ICDAR2019，GitHub 镜像）；XFUND 中文（跨域泛化） | FUNSD / CORD（Google Drive / 私有存储，国内不可达） |

## 环境

- **Python 3.11**（**本机必须用 `py -3.11`**；默认 `python` 是 3.12，缺少本项目依赖）。
  依赖已装在系统 3.11：torch 2.11.0+cu128 / transformers 5.x / bitsandbytes / fastapi，
  本项目也已 `py -3.11 -m pip install -e .` 装好
- 不需要虚拟环境；项目里不保留 `.venv/`，所有命令直接走系统 3.11
- RTX 5060 Laptop **8GB**（Blackwell sm_120），torch 2.11 + CUDA 12.8
- HuggingFace 被墙 → 模型/数据走 **ModelScope** / GitHub 镜像
- transformers **v5.x**（`torch_dtype` 改名 `dtype`、`eval_strategy` 等）

一步自检（应输出 `Python 3.11.x` 和 48 passed）：

```bash
py -3.11 -V
py -3.11 -m pytest tests/ -q
```

## 快速开始

```bash
py -3.11 -m pip install -e .                      # 可编辑安装，使 from src.xxx import ... 全局可用

py -3.11 src/utils/model_download.py --model Qwen/Qwen2.5-VL-3B-Instruct   # 下载模型到 models/（约 7GB）
py -3.11 -m src.data.download_sroie                                       # 下载 SROIE 624 张到 data/raw/sroie/
py -3.11 -m src.inference.engine_hf --image data/raw/sroie/img/000.jpg     # 原生推理冒烟
```

> 仓库里**不含** `data/` 与 `models/`（体积原因，已被 `.gitignore` 排除）。上面的下载命令跑完后
> `pytest` 才会全绿（缺数据时依赖 SROIE 的 2 个用例会显式 skip，不会报错）。

## 目录结构

```
vlm-doc-deploy/
├── data/               # raw: SROIE 图片+标注；processed: 统一格式化
├── models/             # ModelScope 下载的权重（gitignore）
├── src/
│   ├── data/           # 数据加载/预处理（SROIE）
│   ├── inference/      # HF Transformers 推理引擎（generate_text / generate_batch）
│   ├── quantization/   # 量化（bitsandbytes NF4 加载路径，见 inference/engine_hf.py）
│   ├── serving/        # FastAPI 服务
│   ├── eval/           # 字段级 F1、错误分析、吞吐/延迟/显存度量、KV cache 估算
│   └── utils/          # 日志 / 模型下载 / 环境
├── scripts/            # 实验/压测/绘图脚本
├── reports/            # 基准报告、实验汇总
├── tests/              # pytest（不依赖 GPU）
└── docs/               # 技术手册 / 部署与演示指南 / 汇报大纲
```

## 量化结论（全量 624 张正式基准）

> 模型 Qwen2.5-VL-3B · SROIE 624 张（原始 626 张，剔除 2 个计时异常样本，剔除不影响精度结论）· HF Transformers · 统一 `longest_edge=2M` 约束大图

| 方法 | F1 exact | F1 norm | 延迟均值 | 延迟中位 | 峰值显存 | 权重显存 |
|---|---|---|---|---|---|---|
| bf16 原生 | 0.8451 | 0.9501 | 19.6s | 17.1s | 7.8GB | 6.1GB |
| **bnb4 (NF4)** | 0.8304 | 0.9330 | **7.1s** | **6.0s** | **3.7GB** | **2.3GB** |

**bnb4 相对 bf16：精度 −1.7pp（norm F1），速度中位 ×2.85 / 均值 ×2.8，峰值显存 −52%。**
（30 样本初判 −5.5pp 是小样本噪声，全量 624 张已修正为 −1.7pp，近乎无损。）

三个关键发现（详见 `reports/w2_quant_stage1.md` / `w2_quant_stage2.md`）：
1. **4bit 不只省显存还提速** —— bf16 权重 6.1GB 逼近 8GB 卡上限被迫 offload 到 CPU（大图延迟 60s+、中位 17.1s），bnb4 全进 GPU 无 offload（均值 7.1s、中位 6.0s），尾部延迟被压平。
2. **视觉 token 是 VLM 显存第一瓶颈** —— 大扫描件切 ~1.1 万视觉 token 时**激活显存**（视觉塔中间激活 + attention 矩阵；KV cache 仅 ~394MB）爆炸到 21GB（bnb4 未约束实测）。`longest_edge=2M` 约束是「8GB 卡跑大图的硬前提」：无约束大图单次分配需 15.97GB 直接 OOM；约束后全量 624 张视觉 token 总量削减 51%、显存降 83%，norm F1 仍有 0.9330。
3. **量化工具链约束** —— vLLM（无 Windows wheel + sm_120 不兼容）、autoawq（不认 Qwen2.5-VL）均装不了，bitsandbytes NF4 是当前环境唯一跑通方案。

> 精度定位（为什么 0.83/0.93 是合理且优秀的）见 `reports/accuracy_positioning.md`；
> 视觉 token 全量统计见 `reports/visual_tokens_kv.md` / `reports/visual_token_stats.json`。

## 引擎与部署优化结论

| 维度 | 结果 |
|---|---|
| 静态批处理 | batch1/2/4 吞吐 0.178/0.228/0.247（×1.39）、延迟 5.63→4.04s 反降、峰值显存 3670/4968/7591MB（代价只在显存） |
| 耗时结构 | 视觉占比 小票 ≈10% / 大扫描件 ≈16%（`reports/w3_profile.md`） |
| KV cache 量化 | int8 无损省显存；int4 精度崩 + 小图延迟 ×3（`reports/w3_kvquant.md`） |
| 视觉前缀缓存 | 一次性抽取场景收益≈0，不硬造假实现（`reports/w3_prefix_cache.md`） |
| FastAPI 压测 | 并发 1/2/4 → QPS 0.18/0.22/0.22、P95 5.9/9.1/18s（并发只涨延迟不涨吞吐，`reports/w4_serving.md`） |

## 常用命令

```bash
py -3.11 -m pip install -e .                                       # 可编辑安装

# 下载
py -3.11 src/utils/model_download.py --model Qwen/Qwen2.5-VL-3B-Instruct
py -3.11 -m src.data.download_sroie                        # SROIE 624 张
# XFUND 中文表单（跨域泛化，GitHub Release 下载，见 scripts/eval_xfund.py 头部说明）

# 推理
py -3.11 -m src.inference.engine_hf --image data/raw/sample_ticket.png
py -3.11 -m scripts.run_w1_smoke --num_sroie 5             # 冒烟测试

# 量化评估（断点续跑，每样本落盘）
py -3.11 -m scripts.run_full_benchmark                     # 全量 bnb4 + bf16
py -3.11 -m scripts.plot_quant_pareto                      # Pareto 图 + 汇总表
py -3.11 -m scripts.analyze_errors --quant bnb4            # 错误分析报告
py -3.11 -m scripts.analyze_visual_tokens                  # KV cache 定量分析
py -3.11 -m scripts.eval_xfund --num_samples 50            # 跨域泛化（XFUND 中文表单）

# 服务
py -3.11 -m src.serving.app                                # FastAPI（bnb4 + /extract）

# 测试
py -3.11 -m pytest
```

## 环境约束

- RTX 5060 Laptop **8GB**（Blackwell sm_120），torch 2.11 + CUDA 12.8
- HuggingFace 被墙 → 模型走 **ModelScope**、数据走 **GitHub codeload**（`raw.githubusercontent.com` 也被墙）
- transformers **v5.x**（`torch_dtype`→`dtype`）；`AutoProcessor` 需 **torchvision**
- **量化工具链**：vLLM/autoawq 在 Windows+Blackwell 不可用 → 用 bitsandbytes NF4（详见 §13 交接文档）
