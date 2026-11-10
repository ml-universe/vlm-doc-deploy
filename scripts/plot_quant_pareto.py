"""汇总各量化方法的 eval_*.json，画「精度-速度-显存」Pareto 图 + 汇总表。

================================================================================
【这个文件做什么】
    W2 量化实验的收尾：读 reports/ 下所有 eval_*.json（每种量化一条三轴记录），
    汇总成 Markdown 表 + matplotlib 图，输出到 reports/quant_summary.md/.png。

【为什么画 Pareto 而非单维度】
    量化是「精度换速度/显存」的权衡，单看任何一个维度都会误导。Pareto 前沿
    （不被任何其他点在所有维度同时超过的点）才是「最优解集」，讲解「我做了
    Pareto 分析」比「我对比了几个数字」高一个层次。

【图怎么画（三轴 → 二维投影）】
    主图：x=显存(MB)，y=吞吐(tok/s)，气泡大小/颜色=精度损失（相对 bf16）。
    一眼看清「越靠左上越好」。辅助表列出精确数字。

用法::

    py -3.11 -m scripts.plot_quant_pareto
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)


def load_all_eval(reports_dir: str = "reports") -> list[dict]:
    """读 reports/eval_*.json 全部量化结果。"""
    rows = []
    for p in sorted(Path(reports_dir).glob("eval_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        rows.append(d)
    return rows


def _label(r: dict) -> str:
    """量化标签，带分辨率约束后缀区分同名方法。"""
    q = r["quantization"]
    le = r.get("longest_edge")
    if le is not None:
        return f"{q}-{le // 1024 // 1024}M"
    return f"{q}-full"


def summarize(rows: list[dict]) -> str:
    """生成 Markdown 汇总表。"""
    lines = ["# 量化实验汇总", ""]
    lines.append("| 方法 | 分辨率约束 | F1 exact | F1 norm | 延迟均值(s) | 延迟中位(s) | 峰值显存(MB) | 权重显存(MB) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: x.get("quantization", "")):
        le = r.get("longest_edge")
        le_str = f"{le // 1024 // 1024}M" if le else "无"
        lines.append(
            f"| {_label(r)} | {le_str} | {r['f1_exact']['f1']:.4f} | {r['f1_normalized']['f1']:.4f} "
            f"| {r['latency_s_mean']:.2f} | {r['latency_s_median']:.2f} "
            f"| {r['gpu_peak_mb_max']:.0f} | {r['loaded_gpu_allocated_mb']:.0f} |"
        )
    return "\n".join(lines)


def plot(rows: list[dict], out_png: str) -> None:
    """画 显存 vs 延迟 气泡图（气泡大小=精度损失）。中文标注（系统有 SimHei/微软雅黑）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 配置中文字体（Windows 自带黑体/微软雅黑，避免中文显示为方框）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

    bf16 = next((r for r in rows if r["quantization"] == "bf16"), None)
    bf16_f1 = bf16["f1_normalized"]["f1"] if bf16 else None

    fig, ax = plt.subplots(figsize=(9, 6))
    for r in rows:
        x = r["gpu_peak_mb_max"]
        y = r["latency_s_median"]  # 用中位延迟更抗离群
        loss = 0.0
        if bf16_f1 is not None:
            loss = max(0.0, bf16_f1 - r["f1_normalized"]["f1"]) * 100
        ax.scatter(x, y, s=400 + loss * 300, alpha=0.75, label=_label(r))
        ax.annotate(_label(r), (x, y), textcoords="offset points", xytext=(8, 4))

    ax.set_xlabel("峰值显存 (MB)")
    ax.set_ylabel("延迟中位数 (s)")
    ax.set_title("量化 Pareto 前沿：延迟 vs 显存（气泡大小 = 精度损失）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info("Pareto 图已保存: %s", out_png)


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总量化结果 + 画 Pareto 图")
    ap.add_argument("--reports_dir", default="reports")
    args = ap.parse_args()

    rows = load_all_eval(args.reports_dir)
    if not rows:
        log.warning("未找到 eval_*.json，先跑 evaluate_quant.py")
        return

    summary = summarize(rows)
    out_md = Path(args.reports_dir) / "quant_summary.md"
    out_md.write_text(summary + "\n", encoding="utf-8")
    log.info("汇总已保存: %s", out_md)
    print(summary)

    try:
        plot(rows, str(Path(args.reports_dir) / "quant_pareto.png"))
    except Exception as e:  # noqa: BLE001
        log.warning("画图失败（可能缺 matplotlib）: %s", e)


if __name__ == "__main__":
    main()
