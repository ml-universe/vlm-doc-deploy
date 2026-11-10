"""XFUND 中文表单数据集加载器：把词级 question/answer 标注配对成「字段 → 值」。

================================================================================
【这个文件做什么】
    XFUND（doc-analysis/XFUND）是多语言表单理解基准，中文子集 zh.val.json 的标注是
    **词级 BIO**：每个词带 label ∈ {question, answer, header, other}。要把它们配对成
    「字段 → 值」金标，才能和模型的 JSON 抽取对比。

【配对规则（XFUND 官方 linking 关系）】
    XFUND 标注里每个词带 linking 字段，是 [a_id, b_id] 对，语义「a 与 b 关联」：
    - question ↔ answer：这是「字段 → 值」的**叶子对**（如「姓名」→「夏艳辰」）；
    - question ↔ question：字段嵌套/父子关系（如「电话」下挂「手机」「家中」），
      父字段无直接 answer，跳过，只取「直接 link 到 answer」的叶子 question。
    因此对每个 question 词，收集 linking 里「另一个是 answer」的 pair，把 answer
    文本按 id 顺序拼成值；无直接 answer 的 question（父字段）跳过。

【为什么这是「跨域泛化」的关键】
    SROIE 是固定 4 字段（company/date/address/total）的收据；XFUND 是**动态字段**的
    中文表单（电话/手机/邮箱/地址/…）。若模型在「只见过 SROIE」的零样本下，仍能对
    XFUND 的任意字段抽对，就证明了「不是背 4 个字段，而是真懂『字段→值』抽取」。

【评估口径】
    配对后得到 {字段名: 值}。因字段是动态的（每张不同），评估用 scripts/eval_xfund.py
    里的自定义 `_field_f1_dynamic`（按字段名匹配，复用 metrics.normalize 做值归一化），
    不直接用固定 FIELDS 的 field_f1。

用法::

    from src.data.xfund import load_xfund_zh
    samples = load_xfund_zh("data/raw/xfund_zh/zh.val.json", "data/raw/xfund_zh")
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

# 字段名做归一化（去冒号、空白），作为配对 key；值保留原始文本（评估时再归一化比对）
def _norm(text: str) -> str:
    return "".join(ch for ch in text if ch not in "：:　 \t").strip()


def _pair_qa(document: list[dict]) -> dict[str, str]:
    """把词级标注配对成 {字段名: 值}，用 XFUND 官方的 linking 关系（非空间位置）。

    linking 是 [a_id, b_id] 对，语义是「a 与 b 关联」：
    - question ↔ answer：这是「字段 → 值」的**叶子对**（如「姓名」→「夏艳辰」）。
    - question ↔ question：字段嵌套/父子关系（如「电话」下挂「手机」「家中」），
      父字段无直接 answer，跳过；只取「直接 link 到 answer」的叶子 question。

    因此规则：对每个 question 词，收集 linking 里「另一个是 answer」的那些 pair，
    把 answer 文本拼成值；无直接 answer 的 question（父字段）跳过。
    """
    id2word = {w["id"]: w for w in document}
    qtext = {w["id"]: w["text"] for w in document if w["label"] == "question"}
    is_answer = {w["id"]: (w["label"] == "answer") for w in document}

    # question id -> 它直接关联的 answer id 列表
    q_to_as: dict[int, list[int]] = {}
    for w in document:
        for a, b in w.get("linking", []):
            if a in qtext and is_answer.get(b):
                q_to_as.setdefault(a, []).append(b)
            elif b in qtext and is_answer.get(a):
                q_to_as.setdefault(b, []).append(a)

    fields: dict[str, str] = {}
    for q in sorted(qtext.keys()):
        if q not in q_to_as:
            continue  # 父字段（只 link 到 question），跳过
        key = _norm(qtext[q])
        val = "".join(id2word[a]["text"] for a in sorted(set(q_to_as[q])))
        fields[key] = fields.get(key, "") + val
    return fields


def load_xfund_zh(
    json_path: str | Path = "data/raw/xfund_zh/zh.val.json",
    img_dir: str | Path = "data/raw/xfund_zh",
) -> list[dict]:
    """加载 XFUND 中文子集，返回 [{id, image_path, fields}]，fields 为 {字段: 值}。

    json_path 是 zh.val.json（标注），img_dir 是解压后的图片目录。
    """
    json_path = Path(json_path)
    img_dir = Path(img_dir)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    samples: list[dict] = []
    for doc in data["documents"]:
        img_fname = doc["img"]["fname"]
        img_path = img_dir / img_fname
        if not img_path.exists():
            continue  # 图片未下载则跳过
        samples.append(
            {
                "id": doc["id"],
                "image_path": str(img_path),
                "fields": _pair_qa(doc["document"]),
            }
        )
    return samples


def load_image(sample: dict) -> Image.Image:
    """加载样本图片为 RGB PIL.Image。"""
    return Image.open(sample["image_path"]).convert("RGB")


if __name__ == "__main__":
    samples = load_xfund_zh()
    print(f"加载 {len(samples)} 个样本")
    if samples:
        s = samples[0]
        print(f"样本 {s['id']} 字段数 {len(s['fields'])}:")
        for k, v in list(s["fields"].items())[:10]:
            print(f"  {k} = {v}")
