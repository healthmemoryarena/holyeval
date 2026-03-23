"""
AgentClinic → HolyEval 数据转换器

支持两种 AgentClinic 数据格式:
  - medqa format: OSCE_Examination 嵌套结构，含 Correct_Diagnosis
  - nejm format: question + answers 多选题结构，含 correct: true 的答案

转换映射 (medqa):
  OSCE_Examination.Patient_Actor.*             →  strict_inputs[0]（病例描述）
  OSCE_Examination.Physical_Examination_Findings  →  strict_inputs[0]（体格检查部分）
  OSCE_Examination.Test_Results               →  strict_inputs[0]（辅助检查部分）
  OSCE_Examination.Correct_Diagnosis          →  eval.standard_answer

转换映射 (nejm):
  question                                    →  strict_inputs[0]
  answers[correct=True].text                  →  eval.standard_answer

BenchItem 不含 target 字段（由运行时 --target-type/--target-model 决定）。

用法:
  python -m generator.agentclinic.converter medqa input.jsonl output.jsonl
  python -m generator.agentclinic.converter nejm input.jsonl output.jsonl

数据集下载:
  git clone https://github.com/SamuelSchmidgall/AgentClinic
  文件: agentclinic_medqa.jsonl, agentclinic_nejm.jsonl
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================
# 工具函数
# ============================================================


def _dict_to_text(data: Any, depth: int = 0) -> str:
    """递归将嵌套 dict/list 转换为可读文本"""
    if data is None:
        return ""
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, (int, float, bool)):
        return str(data)
    if isinstance(data, list):
        items = [_dict_to_text(item, depth) for item in data if item is not None]
        return "; ".join(i for i in items if i)
    if isinstance(data, dict):
        parts = []
        for key, val in data.items():
            val_text = _dict_to_text(val, depth + 1)
            if val_text:
                clean_key = key.replace("_", " ")
                indent = "  " * depth
                parts.append(f"{indent}{clean_key}: {val_text}")
        return "\n".join(parts)
    return str(data)


# ============================================================
# MedQA 格式（OSCE_Examination 嵌套结构）
# ============================================================


def _build_medqa_case_text(osce: dict) -> str:
    """将 OSCE_Examination 结构展开为自然语言病例描述"""
    parts: list[str] = []
    patient = osce.get("Patient_Actor", {})

    # 人口统计
    if demo := patient.get("Demographics"):
        parts.append(f"患者: {demo}")

    # 病史（History 是 str，Symptoms 单独字段）
    if hist := patient.get("History"):
        parts.append(f"病史: {hist}")

    symptoms = patient.get("Symptoms", {})
    if isinstance(symptoms, dict):
        if primary := symptoms.get("Primary_Symptom"):
            parts.append(f"主诉: {primary}")
        if secondary := symptoms.get("Secondary_Symptoms"):
            parts.append(f"伴随症状: {_dict_to_text(secondary)}")

    if pmh := patient.get("Past_Medical_History"):
        parts.append(f"既往史: {pmh}")
    if social := patient.get("Social_History"):
        parts.append(f"社会史: {social}")
    if ros := patient.get("Review_of_Systems"):
        parts.append(f"系统回顾: {ros}")

    # 体格检查
    pe = osce.get("Physical_Examination_Findings", {})
    if pe:
        pe_text = _dict_to_text(pe)
        if pe_text:
            parts.append(f"体格检查:\n{pe_text}")

    # 辅助检查
    tr = osce.get("Test_Results", {})
    if tr:
        tr_text = _dict_to_text(tr)
        if tr_text:
            parts.append(f"辅助检查:\n{tr_text}")

    parts.append("\n请根据以上病例信息，给出最可能的诊断（中文或英文均可）。")
    return "\n".join(parts)


def _convert_medqa(entry: dict, index: int) -> Optional[dict]:
    """将单条 AgentClinic MedQA 数据转换为 HolyEval BenchItem dict"""
    osce = entry.get("OSCE_Examination", entry)

    correct_diagnosis = osce.get("Correct_Diagnosis", "").strip()
    if not correct_diagnosis:
        logger.warning("跳过无 Correct_Diagnosis 的条目 #%d", index)
        return None

    case_text = _build_medqa_case_text(osce)
    if not case_text:
        logger.warning("跳过无有效内容的条目 #%d", index)
        return None

    # 提取专科标签（从 Neurological_Examination 等字段推断，默认"内科"）
    specialty_tag = _infer_specialty_medqa(osce)

    return {
        "id": f"ac_medqa_{index:04d}",
        "title": f"AgentClinic 诊断 — {specialty_tag}",
        "description": f"临床诊断评测，正确答案: {correct_diagnosis}",
        "user": {
            "type": "manual",
            "strict_inputs": [case_text],
        },
        "eval": {
            "evaluator": "preset_answer",
            "standard_answer": correct_diagnosis,
            "match_mode": "keyword",
        },
        "tags": ["source:agentclinic", "format:medqa", f"specialty:{specialty_tag}"],
    }


def _infer_specialty_medqa(osce: dict) -> str:
    """从 OSCE 结构推断专科（简单关键词匹配）"""
    pe = osce.get("Physical_Examination_Findings", {})
    text = json.dumps(osce, ensure_ascii=False).lower()

    if any(kw in text for kw in ["neurolog", "cranial", "reflex", "motor strength", "gait"]):
        return "神经内科"
    if any(kw in text for kw in ["cardiac", "heart", "ecg", "ekg", "coronary"]):
        return "心内科"
    if any(kw in text for kw in ["pulmon", "respiratory", "lung", "chest"]):
        return "呼吸科"
    if any(kw in text for kw in ["gastro", "abdomen", "liver", "bowel"]):
        return "消化科"
    if any(kw in text for kw in ["derma", "skin", "rash", "lesion"]):
        return "皮肤科"
    if any(kw in text for kw in ["psych", "anxiety", "depression", "mental"]):
        return "精神科"
    if "pe" in pe:  # noqa: F821
        pass
    return "内科"


# ============================================================
# NEJM 格式（多选题结构）
# ============================================================


def _convert_nejm(entry: dict, index: int) -> Optional[dict]:
    """将单条 AgentClinic NEJM 数据转换为 HolyEval BenchItem dict"""
    question = entry.get("question", "").strip()
    answers = entry.get("answers", [])

    if not question:
        logger.warning("跳过无 question 的条目 #%d", index)
        return None

    # 找到正确答案
    correct_answer = next((a["text"] for a in answers if isinstance(a, dict) and a.get("correct")), None)
    if not correct_answer:
        logger.warning("跳过无正确答案的条目 #%d", index)
        return None

    # 在问题后附加选项列表（让模型有选项可选）
    if answers:
        options_text = "\n选项:\n" + "\n".join(
            f"  {chr(65 + i)}. {a['text']}" for i, a in enumerate(answers) if isinstance(a, dict)
        )
        full_question = question + options_text + "\n\n请选择最可能的诊断。"
    else:
        full_question = question

    return {
        "id": f"ac_nejm_{index:04d}",
        "title": "AgentClinic 诊断 — NEJM 病例",
        "description": f"NEJM 临床诊断多选题，正确答案: {correct_answer}",
        "user": {
            "type": "manual",
            "strict_inputs": [full_question],
        },
        "eval": {
            "evaluator": "preset_answer",
            "standard_answer": correct_answer,
            "match_mode": "keyword",
        },
        "tags": ["source:agentclinic", "format:nejm", "specialty:皮肤科"],
    }


# ============================================================
# 批量转换
# ============================================================


def convert(
    fmt: str,
    input_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
) -> int:
    """将 AgentClinic JSONL 转换为 HolyEval BenchItem JSONL

    Args:
        fmt:         数据格式，"medqa" 或 "nejm"
        input_path:  AgentClinic 源 JSONL 路径
        output_path: 输出 HolyEval BenchItem JSONL 路径
        limit:       最大转换条数（None 表示全部）

    Returns:
        成功转换的用例数
    """
    if fmt not in ("medqa", "nejm"):
        raise ValueError(f"不支持的格式: {fmt!r}，请使用 'medqa' 或 'nejm'")

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converter_fn = _convert_medqa if fmt == "medqa" else _convert_nejm

    converted = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            if limit is not None and converted >= limit:
                break
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("第 %d 行 JSON 解析失败: %s", i + 1, e)
                skipped += 1
                continue

            bench_item = converter_fn(entry, i)
            if bench_item is None:
                skipped += 1
                continue

            fout.write(json.dumps(bench_item, ensure_ascii=False) + "\n")
            converted += 1

    logger.info("转换完成: %d 条成功, %d 条跳过, 输出: %s", converted, skipped, output_path)
    return converted


# ============================================================
# CLI 入口
# ============================================================


def main() -> None:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="将 AgentClinic JSONL 转换为 HolyEval BenchItem JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python -m generator.agentclinic.converter medqa agentclinic_medqa.jsonl benchmark/data/agentclinic/medqa.jsonl\n"
            "  python -m generator.agentclinic.converter nejm agentclinic_nejm.jsonl benchmark/data/agentclinic/nejm.jsonl\n"
        ),
    )
    parser.add_argument("format", choices=["medqa", "nejm"], help="数据格式")
    parser.add_argument("input", help="AgentClinic 源 JSONL 文件路径")
    parser.add_argument("output", help="输出 HolyEval BenchItem JSONL 文件路径")
    parser.add_argument("--limit", type=int, default=None, help="最大转换条数")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细日志")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    count = convert(args.format, args.input, args.output, limit=args.limit)
    print(f"转换完成: {count} 条 BenchItem → {args.output}")


if __name__ == "__main__":
    main()
