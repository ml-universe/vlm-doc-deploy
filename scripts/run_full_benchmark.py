"""串行跑全量 624 张的量化基准（bnb4 先，bf16 后），产出正式报告数字。

================================================================================
【这个文件做什么】
    一次性串行跑完 W2 的正式基准：
      1. bnb4（NF4 4bit）+ longest_edge=2M，全 624 张；
      2. bf16（原生对照）+ longest_edge=2M，全 624 张；
    每个方法内部自带 per-sample 落盘 + 断点续跑（见 evaluate_quant.py），
    中途被杀重跑本脚本即可从断点继续。

【为什么串行不并行】
    8GB 卡装不下两个模型同时推理（bnb4 占 ~2.3GB、bf16 占 ~6.1GB），且都吃
    GPU 算力，并行反而互相拖慢。串行还能让显存峰值更可控，降低 OOM 风险。

【时间估算】
    bnb4 ~5.9s/张 × 624 ≈ 1 小时；bf16 ~19s/张 × 624 ≈ 3.3 小时。
    总计 ~4.5 小时，后台运行 + 心跳监控（沿用第一个项目「不能全信通知」教训）。

用法::

    py -3.11 -m scripts.run_full_benchmark
"""
from __future__ import annotations

from scripts.evaluate_quant import evaluate
from src.utils.logger import get_logger

log = get_logger(__name__)

LONGEST_EDGE = 2097152  # 2M，约束大图视觉 token，控显存


def main() -> None:
    log.info("=== [1/2] bnb4 全 624 张 ===")
    evaluate(
        quantization="bnb4",
        num_samples=624,
        longest_edge=LONGEST_EDGE,
        out="reports/eval_bnb4_2M.json",
    )

    log.info("=== [2/2] bf16 全 624 张 ===")
    evaluate(
        quantization="none",
        num_samples=624,
        longest_edge=LONGEST_EDGE,
        out="reports/eval_bf16_2M.json",
    )

    log.info("全部完成，可运行 py -3.11 -m scripts.plot_quant_pareto 汇总")


if __name__ == "__main__":
    main()
