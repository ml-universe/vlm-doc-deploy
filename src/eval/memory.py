"""显存/延迟度量工具：量化实验的「显存」维度统一口径。

================================================================================
【这个文件做什么】
    提供一套统一的显存与延迟度量，量化实验（W2）用它对比不同精度（bf16/AWQ/
    FP8/bnb4bit/GGUF）在 8GB 卡上的真实占用。所有后端都用同一把尺子量，
    Pareto 图才有可比性。

【为什么显存测量有坑】
    1. torch 的 allocated / reserved 是「进程内分配」，不含 CUDA context 和
       offload 到 CPU 的权重；device_map="auto" 时 3B bf16 会有部分层在 CPU，
       必须同时看「GPU 占用」和「总 footprint」。
    2. 峰值显存靠 reset_peak_memory_stats() + max_memory_allocated() 捕捉，
       否则只能看到稳态占用，漏掉 forward 时的激活/视觉 token 峰值。
    3. VLM 的视觉 token 会让显存随输入图波动，量化对比要固定输入尺寸。

【度量口径】
    - gpu_allocated_mb      ：当前 GPU 进程内已分配（PyTorch）
    - gpu_peak_mb           ：推理阶段 GPU 峰值分配
    - model_footprint_mb    ：模型总内存（含 CPU offload），get_memory_footprint
    - latency_s             ：单次推理墙钟时间
    - throughput_tok_s      ：输出 token 数 / 延迟

用法::

    from src.eval.memory import gpu_snapshot, timed_inference
    snap = gpu_snapshot()
    out = timed_inference(fn)
"""
from __future__ import annotations

import time
from typing import Callable


def gpu_snapshot() -> dict:
    """返回当前 CUDA 进程内显存快照（MB）。无 GPU 时返回全 0。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_allocated_mb": 0.0,
                "gpu_reserved_mb": 0.0,
                "gpu_peak_mb": 0.0,
            }
        return {
            "gpu_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "gpu_peak_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
    except Exception:  # noqa: BLE001
        return {
            "gpu_allocated_mb": 0.0,
            "gpu_reserved_mb": 0.0,
            "gpu_peak_mb": 0.0,
        }


def model_footprint_mb(model) -> float:
    """模型总内存（MB），含 device_map offload 到 CPU 的部分。"""
    try:
        return round(model.get_memory_footprint() / 1024**2, 1)
    except Exception:  # noqa: BLE001
        return 0.0


def timed_inference(fn: Callable[[], str]) -> dict:
    """执行一次推理并计时，返回 (输出文本, 延迟, token 数, 吞吐, 峰值显存)。

    调用方负责在 fn 内做一次完整 forward/generate；本函数在 fn 前后重置峰值
    统计，保证 gpu_peak_mb 反映的是「这次推理」的峰值而非历史累计。
    """
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    output_text = fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    latency = time.perf_counter() - t0

    n_tokens = len(output_text.split())  # 粗略按空格分词估算单词数，非精确 token 数（近似吞吐口径）
    n_chars = len(output_text)
    return {
        "output_text": output_text,
        "latency_s": round(latency, 3),
        "output_chars": n_chars,
        "output_tokens_approx": n_tokens,
        "throughput_tok_s": round(n_tokens / latency, 2) if latency > 0 else 0.0,
        **gpu_snapshot(),
    }
