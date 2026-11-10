"""W2 收尾串联：等 bf16 全量跑完 → 自动跑 Pareto 汇总 + 错误分析 + 填报告数字。

================================================================================
【这个文件做什么】
    把「跑完基准」之后的一串手动步骤自动串起来，做成一个幂等的收尾入口：
      1. 等待 bf16 全量 624 张跑完（轮询 reports/preds_bf16.json 条数）；
      2. 跑 plot_quant_pareto —— 读 eval_*.json 生成 quant_summary.md + Pareto 图；
      3. 跑 analyze_errors —— 对 bnb4 和 bf16 各出一份字段级错误分析 md；
      4. 把全量数字填进 reports/w2_quant_stage2.md 的表（替换 _待填_ 占位）。

【为什么做成「等待 + 串联」的单一入口】
    后台一次 `py -3.11 -m scripts.finish_w2` 就能把整条尾巴跑完，不用等人在场
    手动接下一棒。轮询 preds_bf16.json 是因为 evaluate_quant.py 只在全部样本
    跑完后才写 eval_bf16_2M.json，而 preds 是逐样本落盘的，能实时反映进度。

【幂等性】
    - plot_quant_pareto / analyze_errors 都是读产物重算，可重复覆盖；
    - fill_stage2 只在检测到 "_待填_" 占位时才替换，已填过则跳过。

用法::

    py -3.11 -m scripts.finish_w2   # 建议 nohup / 后台运行
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from src.utils.logger import get_logger

log = get_logger(__name__)

TARGET_N = 624
STAGE2 = Path("reports/w2_quant_stage2.md")


def _bf16_count() -> int:
    p = Path("reports/preds_bf16.json")
    if not p.exists():
        return 0
    try:
        return len(json.loads(p.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return 0


def wait_bf16(timeout_s: int = 10 * 3600) -> bool:
    """轮询 preds_bf16.json 直到 624 条；超时返回 False。"""
    waited = 0
    while _bf16_count() < TARGET_N:
        if waited >= timeout_s:
            log.error("等待 bf16 超时（%ds），当前 %d/%d", waited, _bf16_count(), TARGET_N)
            return False
        log.info("bf16 未完成（%d/%d），120s 后再查", _bf16_count(), TARGET_N)
        time.sleep(120)
        waited += 120
    return True


def run_module(mod: str, *args: str) -> None:
    cmd = [sys.executable, "-m", mod, *args]
    log.info(">>> %s", " ".join(cmd))
    r = subprocess.run(cmd, check=False)
    log.info(">>> %s 退出码 %d", mod, r.returncode)


def _row(label: str, q: dict) -> str:
    return (
        f"| {label} | {q['f1_exact']['f1']:.4f} | {q['f1_normalized']['f1']:.4f} "
        f"| {q['latency_s_mean']:.1f}s | {q['gpu_peak_mb_max']:.0f}MB "
        f"| {q['loaded_gpu_allocated_mb']:.0f}MB |"
    )


def fill_stage2() -> None:
    """把 eval_bf16_2M / eval_bnb4_2M 的全量数字填进 stage2 报告的表。"""
    bf16 = json.loads(Path("reports/eval_bf16_2M.json").read_text(encoding="utf-8"))
    bnb4 = json.loads(Path("reports/eval_bnb4_2M.json").read_text(encoding="utf-8"))

    content = STAGE2.read_text(encoding="utf-8")
    if "_待填_" not in content:
        log.info("stage2 报告已填写过，跳过")
        return

    content = content.replace(
        "| bf16 原生 | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |",
        _row("bf16 原生", bf16),
    )
    content = content.replace(
        "| bnb4 (NF4) | _待填_ | _待填_ | _待填_ | _待填_ | _待填_ |",
        _row("bnb4 (NF4)", bnb4),
    )
    # 把「30 样本初步结果」的过渡性说明替换成正式说明
    note_old = (
        "（30 样本初步结果：bf16 norm F1=0.9417 / 延迟 19.3s / 峰值 7.5GB；"
        "bnb4 norm F1=0.8870 / 延迟 5.9s / 峰值 3.7GB。全量后数字可能有 ±1-2pp 波动，"
        "以 `quant_summary.md` 为准。）"
    )
    note_new = (
        "（以上为全量 624 张正式基准，`longest_edge=2M`；F1 为 micro 平均，"
        "bootstrap 95% CI 见 `quant_summary.md`。）"
    )
    content = content.replace(note_old, note_new)
    STAGE2.write_text(content, encoding="utf-8")
    log.info("stage2 报告已填入全量数字: %s", STAGE2)


def main() -> None:
    if not wait_bf16():
        raise SystemExit(1)
    log.info("bf16 全量完成，开始收尾串联")

    run_module("scripts.plot_quant_pareto")
    run_module("scripts.analyze_errors", "--quant", "bnb4")
    run_module("scripts.analyze_errors", "--quant", "bf16")
    fill_stage2()

    log.info("W2 收尾全部完成")


if __name__ == "__main__":
    main()
