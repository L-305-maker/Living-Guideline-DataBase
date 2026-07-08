from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.sqlite_store import DEFAULT_DB_PATH, retrieve_chunks_sqlite
from src.utils.io import DATA_DIR


QUESTIONS: list[dict[str, str]] = [
    {"clinical_department": "肾内科", "question": "成人慢性肾病 CKD 合并蛋白尿时 SGLT2i 推荐用于哪些 eGFR 或 ACR 阈值？"},
    {"clinical_department": "肾内科", "question": "糖尿病合并 CKD 患者血压控制和 ACEI 或 ARB 用药有什么推荐？"},
    {"clinical_department": "呼吸内科", "question": "哮喘急性发作时氧疗目标血氧饱和度和吸入支气管扩张剂如何推荐？"},
    {"clinical_department": "呼吸内科", "question": "COPD 急性加重时短效支气管扩张剂、抗生素和全身糖皮质激素如何使用？"},
    {"clinical_department": "感染内科", "question": "成人社区获得性肺炎门诊经验性抗菌治疗首选哪些方案？"},
    {"clinical_department": "感染内科", "question": "HIV 检测服务在结核病和性传播感染服务中是否应常规提供？"},
    {"clinical_department": "心血管内科", "question": "成人高血压一线降压药物 ACEI ARB 钙通道阻滞剂和血压目标如何推荐？"},
    {"clinical_department": "心血管内科", "question": "房颤患者卒中预防抗凝治疗 DOAC 和华法林如何选择？"},
    {"clinical_department": "神经内科", "question": "偏头痛急性期曲坦类和 CGRP 相关药物以及预防性治疗如何推荐？"},
    {"clinical_department": "神经内科", "question": "急性缺血性卒中溶栓、抗血小板和机械取栓的推荐是什么？"},
    {"clinical_department": "内分泌科", "question": "2 型糖尿病患者二甲双胍、SGLT2 抑制剂、GLP-1 受体激动剂和 HbA1c 目标如何推荐？"},
    {"clinical_department": "内分泌科", "question": "骨质疏松患者维生素 D、钙剂、双膦酸盐和骨折预防如何管理？"},
    {"clinical_department": "消化内科", "question": "幽门螺杆菌根除治疗中铋剂四联疗法、PPI 和抗生素如何推荐？"},
    {"clinical_department": "消化内科", "question": "溃疡性结肠炎或炎症性肠病诱导缓解和维持治疗如何使用激素和生物制剂？"},
    {"clinical_department": "肿瘤科", "question": "乳腺癌术后辅助内分泌治疗、HER2 阳性曲妥珠单抗治疗有什么推荐？"},
    {"clinical_department": "肿瘤科", "question": "非小细胞肺癌 EGFR 突变、PD-1 免疫治疗和一线治疗如何选择？"},
    {"clinical_department": "小儿内科", "question": "儿童哮喘长期控制中吸入糖皮质激素和 SABA 缓解治疗如何推荐？"},
    {"clinical_department": "小儿内科", "question": "儿童发热、急性中耳炎或肺炎抗生素治疗何时推荐？"},
    {"clinical_department": "精神心理科", "question": "成人抑郁症一线治疗中 SSRI、心理治疗和 CBT 如何推荐？"},
    {"clinical_department": "精神心理科", "question": "精神分裂症抗精神病药、氯氮平和长效针剂适用情形是什么？"},
    {"clinical_department": "骨科", "question": "膝骨关节炎运动治疗、NSAID 和关节腔糖皮质激素注射如何推荐？"},
    {"clinical_department": "骨科", "question": "髋部骨折术后康复、抗凝预防和骨质疏松治疗有什么推荐？"},
    {"clinical_department": "血液内科", "question": "静脉血栓栓塞 VTE 抗凝治疗 DOAC 和疗程如何推荐？"},
    {"clinical_department": "血液内科", "question": "缺铁性贫血铁剂治疗和输血阈值有什么指南建议？"},
    {"clinical_department": "免疫科/风湿免疫科", "question": "类风湿关节炎甲氨蝶呤、生物制剂和 treat-to-target 策略如何推荐？"},
    {"clinical_department": "免疫科/风湿免疫科", "question": "系统性红斑狼疮或狼疮肾炎羟氯喹、糖皮质激素和免疫抑制剂如何使用？"},
    {"clinical_department": "眼科", "question": "糖尿病视网膜病变筛查频率和抗 VEGF 治疗适应证是什么？"},
    {"clinical_department": "眼科", "question": "青光眼降眼压治疗中前列腺素类似物和目标眼压如何推荐？"},
    {"clinical_department": "产科", "question": "妊娠期高血压和子痫前期预防中阿司匹林、硫酸镁如何推荐？"},
    {"clinical_department": "产科", "question": "妊娠期糖尿病筛查、饮食管理、胰岛素或二甲双胍治疗如何推荐？"},
    {"clinical_department": "皮肤性病科", "question": "银屑病治疗中外用糖皮质激素、光疗和生物制剂如何选择？"},
    {"clinical_department": "皮肤性病科", "question": "特应性皮炎保湿剂、外用激素和 dupilumab 治疗有什么推荐？"},
    {"clinical_department": "老年病科", "question": "老年人跌倒预防中运动训练、维生素 D 和药物审查如何推荐？"},
    {"clinical_department": "老年病科", "question": "痴呆患者胆碱酯酶抑制剂、照护者干预和行为症状管理如何推荐？"},
    {"clinical_department": "泌尿外科", "question": "前列腺癌 PSA 筛查、活检和主动监测适用于哪些患者？"},
    {"clinical_department": "泌尿外科", "question": "导尿管相关尿路感染或复发性尿路感染抗生素治疗如何推荐？"},
    {"clinical_department": "营养科", "question": "成人肥胖管理中饮食、运动、行为干预和减重手术如何推荐？"},
    {"clinical_department": "营养科", "question": "危重症患者肠内营养启动时机、蛋白和能量目标如何推荐？"},
    {"clinical_department": "口腔科", "question": "牙科操作前感染性心内膜炎抗生素预防适用于哪些患者？"},
    {"clinical_department": "口腔科", "question": "牙周炎治疗中洁治、根面平整和氯己定辅助治疗如何推荐？"},
    {"clinical_department": "妇科", "question": "宫颈癌筛查中 HPV 检测、细胞学和阴道镜转诊标准是什么？"},
    {"clinical_department": "妇科", "question": "子宫内膜异位症疼痛治疗中 NSAID、激素治疗和手术如何推荐？"},
    {"clinical_department": "过敏/变态反应科", "question": "过敏性鼻炎鼻用糖皮质激素、抗组胺药和免疫治疗如何推荐？"},
    {"clinical_department": "过敏/变态反应科", "question": "过敏反应或过敏性休克肾上腺素自动注射器和观察时间如何推荐？"},
    {"clinical_department": "康复医学科", "question": "卒中康复早期活动、作业治疗和言语吞咽训练如何推荐？"},
    {"clinical_department": "康复医学科", "question": "非特异性下腰痛康复中运动治疗、教育和避免卧床如何推荐？"},
    {"clinical_department": "新生儿科", "question": "新生儿黄疸光疗、胆红素监测和换血治疗阈值如何推荐？"},
    {"clinical_department": "新生儿科", "question": "早产儿呼吸窘迫综合征 CPAP、表面活性物质和氧饱和度目标如何推荐？"},
    {"clinical_department": "耳鼻喉科", "question": "急性鼻窦炎抗生素、鼻用糖皮质激素和观察等待如何推荐？"},
    {"clinical_department": "耳鼻喉科", "question": "儿童中耳炎鼓膜置管、听力随访和抗生素治疗如何推荐？"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 50 retrieve goldest questions with top-k chunks.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "goldest" / "retrieve_goldest.json")
    parser.add_argument("--jsonl-output", type=Path, default=DATA_DIR / "goldest" / "retrieve_goldest.jsonl")
    parser.add_argument("--report-output", type=Path, default=DATA_DIR / "goldest" / "retrieve_goldest_report.json")
    parser.add_argument("--topk", type=int, default=10)
    return parser.parse_args()


def chunk_payload(rank: int, chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": chunk["chunk_id"],
        "doc_id": chunk["doc_id"],
        "chunk_type": chunk.get("chunk_type", ""),
        "score": chunk.get("score"),
        "title": chunk.get("title", ""),
        "publication_date": chunk.get("publication_date", "unknown"),
        "source_institution": chunk.get("source_institution", ""),
        "clinical_department": chunk.get("clinical_department", ""),
        "section_path": chunk.get("section_path") or [],
        "retrieval_key": chunk.get("retrieval_key", ""),
        "token_count": chunk.get("token_count", 0),
        "source_file": chunk.get("source_file", ""),
        "markdown_clean_path": chunk.get("markdown_clean_path", ""),
        "text": chunk.get("content", ""),
        "retrieval_text": chunk.get("retrieval_text", ""),
    }


def build_rows(db_path: Path, topk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(QUESTIONS, start=1):
        retrieval_query = normalize_retrieval_query(seed["question"])
        results = retrieve_chunks_sqlite(
            retrieval_query,
            db_path=db_path,
            clinical_department=seed["clinical_department"],
            topk=topk,
            exclude_reference_sections=True,
        )
        rows.append(
            {
                "query_id": f"retrieve_goldest_{index:03d}",
                "task": "retrieve",
                "question": seed["question"],
                "query": seed["question"],
                "retrieval_query": retrieval_query,
                "clinical_department": seed["clinical_department"],
                "topk": topk,
                "retrieval_backend": "sqlite_fts_rrf",
                "top_chunks": [chunk_payload(rank, chunk) for rank, chunk in enumerate(results, start=1)],
            }
        )
    return rows


def normalize_retrieval_query(query: str) -> str:
    query = query.replace("-", " ").replace("/", " ")
    return re.sub(r"\s+", " ", query).strip()


def validate(rows: list[dict[str, Any]], topk: int) -> dict[str, Any]:
    errors: list[str] = []
    if len(rows) != 50:
        errors.append(f"expected 50 questions, got {len(rows)}")
    ids = [row["query_id"] for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate query_id values")
    department_counts = Counter(row["clinical_department"] for row in rows)
    for row in rows:
        chunks = row["top_chunks"]
        if len(chunks) != topk:
            errors.append(f"{row['query_id']} expected {topk} chunks, got {len(chunks)}")
        chunk_ids = [chunk["chunk_id"] for chunk in chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            errors.append(f"{row['query_id']} has duplicate chunk ids")
        mismatched = [chunk["chunk_id"] for chunk in chunks if chunk["clinical_department"] != row["clinical_department"]]
        if mismatched:
            errors.append(f"{row['query_id']} has chunks outside department filter: {mismatched[:3]}")
    return {
        "ok": not errors,
        "errors": errors,
        "question_count": len(rows),
        "topk": topk,
        "department_counts": dict(sorted(department_counts.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    rows = build_rows(args.db_path, args.topk)
    validation = validate(rows, args.topk)
    report = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "db_path": str(args.db_path),
        "output": str(args.output),
        "jsonl_output": str(args.jsonl_output),
        "validation": validation,
    }
    if not validation["ok"]:
        raise SystemExit(json.dumps(report, ensure_ascii=False, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    write_jsonl(args.jsonl_output, rows)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
