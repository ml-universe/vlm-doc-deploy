# W2 量化实验 —— 第二阶段（全量基准 + 工具链约束结论）

> 日期：2026-08-25 · 模型 Qwen2.5-VL-3B · 引擎 HF Transformers · 数据 SROIE 全 624 张
> 本文件是「结论 + 方法论」部分；精确数字见 `quant_summary.md`（跑完自动生成）。

---

## 一、量化工具链在「Windows + RTX 5060 (Blackwell sm_120)」上的实测约束

这是本阶段最重要的**工程结论**，比单个量化数字更有项目价值。逐一实测三条路：

| 工具 | 启动规划定位 | 实测结论 | 根因 |
|---|---|---|---|
| vLLM AWQ-4bit | 主力 | ❌ **无法安装** | 无 Windows 原生 wheel（仅 Linux 源码，`pip --only-binary win_amd64` 报 no matching distribution）；需 WSL2 未安装；RTX 50 系列 sm_120 已知「no kernel image」不兼容 |
| autoawq | 对照 | ❌ **不支持 Qwen2.5-VL** | 其 `qwen2vl.py` 只映射 `Qwen2VLDecoderLayer`，而 Qwen2.5-VL 是 `Qwen2_5_VLDecoderLayer`，模块名对不上，量化会错层 |
| FP8 / torchao | 对照 | ⚠️ 未投入（风险同 VLM 量化不成熟） | torch 原生支持 `float8_e4m3fn`，但 torchao 对 VLM 支持同样不成熟 |

**讲解口径**：项目启动规划（该规划文档不在本仓库内）§9.1 预警「VLM 量化工具链不成熟」被实证——不仅冷门工具不成熟，连最稳的 vLLM 在「消费级新卡（sm_120）+ 原生 Windows」组合下都物理装不了。这是边缘部署的真实约束：**模型新、卡新、系统小众，三者叠加就是工具链真空区**。踩过这个坑，才知道「生产环境的可用性」比「论文里的 SOTA 方法」更重要。

**当前环境唯一跑通的量化方案：bitsandbytes NF4 4bit**（权重 ~2.3GB，全进 GPU，无 offload）。

---

## 二、全量 624 张正式基准（bootstrap 95% CI）

> 数字由 `scripts/run_full_benchmark.py` 产出，`longest_edge=2M` 统一约束大图。

| 方法 | F1 exact | F1 norm | 延迟均值 | 峰值显存 | 权重显存 |
|---|---|---|---|---|---|
| bf16 原生 | 0.8451 | 0.9501 | 19.6s | 7799MB | 6132MB |
| bnb4 (NF4) | 0.8304 | 0.9330 | 7.1s | 3710MB | 2302MB |

（以上为全量 624 张正式基准，`longest_edge=2M`；F1 为 micro 平均，bootstrap 95% CI 见 `quant_summary.md`。）

---

## 三、方法论（延续第一个项目的严谨实验风格）

1. **固定变量**：同一批 624 样本、同一字段 schema、同一 prompt、同一 `longest_edge=2M`，只变量化方式，对照干净。
2. **双档 F1**：exact（严格）+ normalized（小写/去空白/去货币前缀），后者更贴近「人判断抽对没有」，避免 `RM 33.90` vs `33.90` 这类 OCR 噪声误杀。
3. **bootstrap 95% CI**：对 N 样本有放回重采样 1000 次，取 2.5%/97.5% 分位数。量化差异必须带 CI 才能谈「是否显著」。
4. **断点续跑**：每样本落盘 `preds_<quant>.json`，中途 OOM 重跑跳过已完成 id。

---

## 四、讲解要点（本阶段新增）

1. **量化工具链的版本/硬件适配**：Qwen2.5-VL 是 2025 新模型，工具链没跟上 → 知道「选型要盯模型-工具-硬件三角匹配」。
2. **AWQ vs bitsandbytes 的本质区别**（即使 AWQ 没跑通也能讲原理）：AWQ 是 activation-aware（按激活分布长尾选通道缩放），bitsandbytes NF4 是统一的分块非线性量化（不需要校准集）——这是「为什么 bnb4 在工具链不成熟时反而是最优解」的原因。
3. **视觉 token 控制**（延续第一阶段）：`longest_edge` 是边缘部署的显存旋钮，大扫描件从 11147 token 降到 2645，激活显存 21GB→3.7GB（KV cache 本身仅 394MB）。
