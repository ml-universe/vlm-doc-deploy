"""HF Transformers 原生推理后端：加载 Qwen2.5-VL，喂图 → 输出结构化字段 JSON。

================================================================================
【这个文件做什么】
    W1 的「跑通」目标：用 HuggingFace Transformers 加载 Qwen2.5-VL-3B，把一张
    票据图喂进去，让它按 prompt 抽取 company/date/address/total 四个字段，
    输出结构化 JSON。这是后续所有量化/引擎优化的「精度基线（FP16 原生）」对照。

【为什么先做 HF 原生，而不是直接上 vLLM】
    1. 先拿到「不量化、不优化」的原生精度基线，后续 AWQ/FP8/GGUF 都跟它比；
    2. HF 加载路径最稳、依赖最少，先把「模型能跑、能出对」验证掉；
    3. vLLM 的 VLM 支持（尤其视觉 token）还需单独踩坑，放 W3 引擎对比再做。

【关键实现点】
    - 模型走 src.utils.model_download.resolve_model（ModelScope 优先，HF 被墙）；
    - transformers v5.x：torch_dtype 已改名 dtype；
    - Qwen2.5-VL 用 chat template + 图像 token，processor 直接吃 PIL 图（无需
      qwen_vl_utils，省一个依赖）；
    - 输出解析：让模型输出 JSON，再做容错解析（剥离代码块围栏、尾逗号修复）。

【讲解要点】
    这个文件是「精度基线」的锚点：讲量化损失时必须先有 FP16 原生做参照，
    否则「4bit 掉 0.5%」无从谈起。

用法::

    py -3.11 -m src.inference.engine_hf --image data/raw/sample_ticket.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from PIL import Image

from src.utils.logger import get_logger
from src.utils.model_download import resolve_model

log = get_logger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# SROIE 的 4 个字段，作为抽取 schema（后续会扩展到 CORD 更多字段）
FIELDS = ["company", "date", "address", "total"]

EXTRACTION_PROMPT = (
    "Read this receipt image and extract the following fields as JSON.\n"
    'Return ONLY a JSON object with keys: "company", "date", "address", "total".\n'
    'If a field is missing, set it to "".\n'
    'Example: {"company": "STARBUCKS", "date": "2026-08-25", "address": "...", "total": "38.50"}'
)


def _repair_json(text: str) -> dict:
    """容错解析模型输出的 JSON（剥离代码块、修复尾逗号/单引号/截断等常见脏输出）。"""
    text = text.strip()
    # 去掉 ```json ... ``` 围栏
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # 截取第一个 { 到最后一个 }（无 } 时取末尾，交给后续兜底补齐）
    start = text.find("{")
    if start == -1:
        raise ValueError(f"输出中找不到 JSON 对象: {text[:200]!r}")
    end = text.rfind("}")
    text = text[start : end + 1] if end != -1 else text[start:]
    # 修复尾逗号："," 后面紧跟 "}" 或 "]"
    text = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 兜底 1：把「单引号包裹的 key/值」改成双引号（只处理紧邻冒号/逗号/括号的引号，
        # 避免误伤值内部的撇号如 MCDONALD'S）。分两步：先换键，再换紧邻结构边界的值引号。
        text2 = re.sub(r"([{,]\s*)'([^']*?)'(\s*:)", r'\1"\2"\3', text)
        text2 = re.sub(r"(:\s*)'([^']*?)'(\s*[,}])", r'\1"\2"\3', text2)
        try:
            return json.loads(text2)
        except json.JSONDecodeError:
            # 兜底 2：截断 JSON 补齐——模型输出被 max_new_tokens 截断时，括号/引号不闭合。
            # 按括号栈补齐缺失的 } ] 和结尾引号，再重试。可能丢最后一个字段，但能救回前面的字段。
            text3 = _balance_truncated_json(text2)
            return json.loads(text3)


def _balance_truncated_json(text: str) -> str:
    """补齐被截断的 JSON：按栈补缺失的右括号/引号。仅作「救回已生成字段」的兜底。"""
    # 统计未闭合的引号（奇数个引号 → 最后一个字符串被截断，补一个引号）
    in_str = False
    stack: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
            elif ch == "]":
                if stack and stack[-1] == "[":
                    stack.pop()
        i += 1
    out = text
    if in_str:
        out += '"'
    # 逆序补右括号（{ 补 }，[ 补 ]）
    for opener in reversed(stack):
        out += "]" if opener == "[" else "}"
    return out


def load_model(
    model_name: str = DEFAULT_MODEL,
    cache_dir: str | None = "models",
    dtype: str = "auto",
    quantization: str = "none",
):
    """加载 Qwen2.5-VL 模型与 processor，返回 (model, processor)。

    - dtype="auto" 时用 bf16（RTX 5060 支持 bf16，显存省一半）；
    - quantization 可选 "none" / "bnb4"（bitsandbytes NF4 4bit）。bnb4 时
      权重 ~2.3GB，可全部进 GPU、消除 device_map offload，是 W2 量化对比之一。
    """
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    path = resolve_model(model_name, cache_dir)
    log.info("加载模型: %s（量化=%s）", path, quantization)

    if quantization == "bnb4":
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        # dtype 字符串映射到 torch dtype（argparse 的 choices 用友好名，此处归一化）
        dtype = {
            "auto": torch.bfloat16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }.get(dtype, torch.bfloat16)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            path,
            dtype=dtype,
            device_map="auto",
        )
    processor = AutoProcessor.from_pretrained(path)
    # decoder-only 模型批量推理必须左 padding：新 token 才能追加在真实序列末尾，
    # 否则 right-padding 会污染 position ids，batch>1 的输出与单图不一致。
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def generate_text(
    model,
    processor,
    image: Image.Image | str | Path,
    max_new_tokens: int = 256,
    longest_edge: int | None = None,
    prompt: str | None = None,
) -> str:
    """喂一张图，返回模型生成的原生文本（未解析）。

    从 extract_fields 里抽出，供度量脚本复用：度量需要拿输出文本算 token 数、
    并精确计时，故单独成函数。

    longest_edge：约束输入最长边（像素），Qwen2.5-VL 会按 size 的 longest_edge
    动态切图。默认 None 走模型配置的 12845056（12.8M），但大图（A4 扫描件
    4961×7016，或 2481×3508）会切出 ~1.1 万视觉 token，KV cache 在 8GB 卡上爆炸。
    边缘部署设小（如 2097152≈2M）可把视觉 token 降到 ~2600（−76%），
    这是「精度 vs 显存」的显式旋钮。

    prompt：自定义抽取指令，默认用 EXTRACTION_PROMPT（SROIE 4 字段）。跨域
    评估（如 XFUND 表单）传入自定义 prompt，无需复制本函数的 image→generate 逻辑。
    """
    import torch

    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt or EXTRACTION_PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    if longest_edge is not None:
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt",
            size={"longest_edge": longest_edge, "shortest_edge": 3136},
        )
    else:
        inputs = processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def generate_batch(
    model,
    processor,
    images: list[Image.Image | str | Path],
    max_new_tokens: int = 256,
    longest_edge: int | None = None,
) -> list[str]:
    """喂多张图，返回每张图的原生输出文本列表（静态批处理）。

    ================================================================================
    【这是什么】
        W3「引擎/推理层优化」的第一项：静态批处理（static batching）。与
        generate_text 的区别是，一次 forward 同时处理 batch_size 张图，让 GPU 的
        矩阵乘更「满」——提高吞吐（样本/秒），代价是每样本延迟上升（要等整批
        完成）+ 峰值显存随 batch 线性上升。这是「吞吐 vs 延迟 vs 显存」三轴权衡
        的载体，与 W2 的量化 Pareto 同构，讲解「批处理是部署的吞吐旋钮」。

    【实现要点】
        - Qwen2.5-VL 的 processor 接受 list 输入 + padding=True，batch 内不同图
          的视觉 token 数不同会 pad 到最大长度（pad 位置靠 attention_mask 屏蔽）；
        - 截取「新生成 token」用 input_ids 的 shape[1]（含 padding 的完整输入
          长度）——因为 generate() 无论 left/right padding，都把新 token 追加在
          序列「末尾」，故切片起点恒为完整输入长度。不能用「非 pad token 计数」
          （那是真实 prompt 长度，right-padding 下会过早切片、把尾部 pad 位也
          带进输出）。

    【与连续批处理（continuous batching）的区别】
        静态批处理是把一批请求「凑齐再一起 forward」，请求之间互相等；连续批
        处理（vLLM PagedAttention）是请求级调度、谁好了谁补位。本项目在 HF 原生
        下只能做静态批处理，连续批处理是 vLLM 的卖点——这是「为什么 vLLM 吞吐
        更高」的根源，我在 W3 文档里讲原理。
    """
    import torch

    imgs = [
        img if isinstance(img, Image.Image) else Image.open(img).convert("RGB")
        for img in images
    ]
    texts = []
    for _ in imgs:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": EXTRACTION_PROMPT},
                ],
            }
        ]
        texts.append(
            processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    kwargs = {"text": texts, "images": imgs, "padding": True, "return_tensors": "pt"}
    if longest_edge is not None:
        kwargs["size"] = {"longest_edge": longest_edge, "shortest_edge": 3136}
    inputs = processor(**kwargs)

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # 新生成 token 的切片起点 = 完整输入长度（含 padding）。generate() 把新 token
    # 追加在序列末尾，无论 left/right padding 起点都是 input_ids.shape[1]。
    input_len = inputs["input_ids"].shape[1]
    out_texts = []
    for row in generated_ids:
        new_ids = row[input_len:]
        out_texts.append(
            processor.batch_decode(
                [new_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        )
    return out_texts


def extract_fields(
    model,
    processor,
    image: Image.Image | str | Path,
    max_new_tokens: int = 256,
    longest_edge: int | None = None,
) -> dict:
    """喂一张图，返回抽取出的字段 dict（company/date/address/total）。"""
    output_text = generate_text(model, processor, image, max_new_tokens, longest_edge)
    parsed = _repair_json(output_text)
    # 补齐缺失字段为空串，保证返回结构稳定
    return {f: parsed.get(f, "") for f in FIELDS}


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen2.5-VL 原生推理冒烟测试")
    ap.add_argument("--image", default="data/raw/sample_ticket.png")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--cache_dir", default="models")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "float16"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    if not Path(args.image).exists():
        log.error("图片不存在: %s（可先运行 py -3.11 src/utils/sample_ticket.py 生成）", args.image)
        raise SystemExit(1)

    model, processor = load_model(args.model, args.cache_dir, args.dtype)
    result = extract_fields(model, processor, args.image, args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
