# 谨慎地把当前仍标为未分类（clinical_department == "未分类"）的文档重新归类。
#
# 设计动机：clinical_department 通常由 doc 级规则引擎给出，但部分文档因标题歧义
# 或 front-matter 缺失被默认到"未分类"；本模块基于"标题高精度信号"做二次归类。
# 仅以标题为信号、避免引入摘要正文扩大科室范围（摘要常常跨学科），保证保守。
"""Conservatively reclassify documents whose department is still unknown."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.clinical_department import UNKNOWN_DEPARTMENT
from src.utils.front_matter import dump_front_matter, parse_front_matter
from src.utils.io import DATA_DIR, read_jsonl


# 科室 → 正则；flags=IGNORECASE。中英文双语词表，便于中文指南 + 英文文献兼容。
# 顺序：高频科室在前；正则宁可漏掉也不要误伤（保守策略的代价是覆盖率）。
RULES: tuple[tuple[str, str], ...] = (
    ("肿瘤科", r"癌|肿瘤|胶质瘤|转移瘤|间质瘤|carcinoma|cancer|malignan|neoplasm|tumo(?:u)?r|metastasi"),
    ("呼吸内科", r"呼吸|肺炎|肺功能|慢性阻塞性肺|哮喘|支气管|pulmonary|pneumonia|asthma|\bcopd\b|chronic obstructive|lung disease"),
    ("心血管内科", r"心血管|冠心|冠状动脉|心肌|心律|心房颤动|房颤|心力衰竭|高血压|cardiovascular|cardiac|coronary|myocard|heart failure|arrhythmia|atrial fibrillation|hypertension|hipertensión|\bqt\b"),
    ("神经内科", r"神经|脑血管|脑卒中|卒中|癫痫|帕金森|痴呆|neurolog|stroke|epilep|seizure|parkinson|dementia|multiple sclerosis|migraine|cerebrovascular"),
    ("内分泌科", r"糖尿病|甲状腺|垂体|催乳素|性腺功能|生长激素|骨质疏松|diabet|thyroid|pituitar|prolactin|hypogonad|growth hormone|\bgh deficiency\b|osteoporosis|metabolic syndrome"),
    ("血液内科", r"白血病|淋巴瘤|贫血|血友病|输血|血小板|凝血|leuk[ae]mia|lymphoma|an[ae]mia|h[ae]mophilia|transfusion|platelet|coagulation|myeloma"),
    ("肾内科", r"肾病|肾炎|肾衰|肾功能|透析|renal|kidney|nephro|dialysis"),
    ("免疫科/风湿免疫科", r"风湿|类风湿|红斑狼疮|痛风|rheumat|lupus|vasculitis|sjogren|gout|autoimmune"),
    ("感染内科", r"感染|结核|肝炎|艾滋|狂犬|抗菌药|infecti|tuberculosis|hepatitis|\bhiv\b|rabies|antimicrobial|antibiotic|sepsis"),
    ("过敏/变态反应科", r"过敏|变态反应|荨麻疹|allerg|anaphyl|urticaria|hypersensitiv"),
    ("消化内科", r"消化|胃|肠|食管|肝硬化|脂肪肝|胰腺炎|gastro|digestive|bowel|crohn|colitis|cirrhosis|liver disease|pancreati|hepatology"),
    ("普通外科", r"普通外科|腹股沟疝|切口疝|阑尾炎|general surgery|inguinal hernia|incisional hernia|appendicitis"),
    ("小儿外科", r"小儿外科|先天性巨结肠|肛门直肠畸形|pediatric surgery|paediatric surgery|hirschsprung|anorectal malformation"),
    ("胸外科", r"胸外科|胸腔镜|肺叶切除|thoracic surgery|thoracoscop|lobectomy"),
    ("心血管外科", r"心脏外科|心血管外科|冠状动脉旁路|cardiac surgery|cardiovascular surgery|coronary bypass|valve surgery"),
    ("神经外科", r"神经外科|开颅|颅内动脉瘤|脑肿瘤|脑转移|neurosurg|craniotom|intracranial aneurysm|brain tumo(?:u)?r|brain metast"),
    ("肝胆外科", r"肝胆外科|肝切除|胰十二指肠切除|胆囊切除|hepatectomy|liver resection|pancreaticoduodenectomy|cholecystectomy"),
    ("血管外科", r"血管外科|外周动脉|主动脉瘤|静脉血栓|vascular surgery|peripheral arter|aortic aneurysm|venous thrombo"),
    ("泌尿外科", r"泌尿|膀胱|前列腺|尿路|肾结石|urolog|bladder|prostate|urinary tract|renal stone"),
    ("骨科", r"骨科|骨折|脊柱|关节置换|胫骨|股骨|骨缺损|orthop|fracture|spine|spinal|arthroplasty|tibia|femur|bone defect"),
    ("妇产科", r"妇产|妊娠|孕妇|分娩|产前|产后|胎儿|obstetric|pregnan|prenatal|postpartum|fetal"),
    ("妇科", r"妇科|子宫|卵巢|宫颈|阴道|gynec|uter|ovarian|cervical|vaginal"),
    ("生殖医学科", r"不孕|辅助生殖|生育力|生殖医学|infertility|assisted reproduction|fertility|reproductive medicine"),
    ("新生儿科", r"新生儿|早产儿|neonat|preterm infant"),
    ("小儿内科", r"儿科|儿童|小儿|pediatric|paediatric|children|childhood"),
    ("精神心理科", r"精神|抑郁|焦虑|自杀|成瘾|孤独症|psychiatr|depressi|anxiety|suicid|addiction|autism"),
    ("皮肤性病科", r"皮肤|银屑病|湿疹|皮炎|白癜风|痤疮|dermat|psoriasis|eczema|vitiligo|acne"),
    ("耳鼻喉科", r"耳鼻喉|鼻窦|听力|耳聋|中耳|咽喉|otolaryng|sinusitis|hearing loss|otitis|laryng"),
    ("眼科", r"眼科|视网膜|青光眼|白内障|角膜|黄斑|ophthalm|retina|glaucoma|cataract|cornea|macular"),
    ("口腔科", r"口腔|牙周|牙髓|龋|正畸|根管|dental|dentistry|periodont|endodont|orthodont|caries"),
    ("康复医学科", r"康复|功能训练|rehabilitation|physical therapy"),
    ("营养科", r"营养|膳食|肠内营养|肠外营养|nutrition|dietary|enteral|parenteral nutrition"),
    ("器官移植", r"器官移植|心移植|肺移植|肝移植|organ transplant|heart transplant|lung transplant|liver transplant"),
    ("烧伤科", r"烧伤|烫伤|冻伤|burn injury|thermal injury|frostbite"),
    ("整形美容科", r"整形外科|美容外科|皮瓣|瘢痕|plastic surgery|reconstructive surgery|skin flap|scar management"),
    ("重症医学科", r"重症|危重|体外膜氧合|critical care|intensive care|\bicu\b|\becmo\b"),
    ("急诊医学科", r"急诊|心肺复苏|灾难医学|emergency medicine|resuscitation|major trauma|disaster medicine"),
    ("麻醉科", r"麻醉|镇静|围术期|an[ae]sthes|sedation|perioperative"),
    ("医学影像科", r"影像诊断|放射诊断|磁共振|radiology|diagnostic imaging|magnetic resonance"),
    ("病理科", r"病理诊断|组织病理|pathology guideline|histopathology"),
    ("检验科", r"临床检验|实验室医学|laboratory medicine|clinical laboratory"),
)

# 预编译所有正则：避免每次 classify 都重编译，单条记录成本可忽略但大批量很可观。
COMPILED_RULES = tuple((department, re.compile(pattern, re.IGNORECASE)) for department, pattern in RULES)

# 排除掉明显不是指南标题的"杂项"：图表、FAQ、勘误等，避免误命中规则。
ARTIFACT_TITLE = re.compile(
    r"^(?:figures?|tables?(?:\s+\d.*)?|frequently asked questions|faq|coming soon|loading|"
    r"position statement|resumen|whereas.*|measure scoring sheet|pdsa worksheet.*|"
    r"prior authorization checklist|insurance verification|erratum.*)$",
    re.IGNORECASE,
)


def identify_departments(title: str) -> tuple[list[str], str]:
    """仅依据标题高精度信号归类，避免摘要中跨学科描述扩大标签。

    返回 (按规则命中的去重科室列表, 命中原因)。
    原因：
    - "title_rule"：至少命中一条规则
    - "artifact_or_generic_title"：标题属于排除词（图表/FAQ 等）
    - "no_precise_signal"：标题正常但无规则命中
    """
    normalized_title = unicodedata.normalize("NFKC", title or "").strip()
    if not normalized_title or ARTIFACT_TITLE.fullmatch(normalized_title):
        return [], "artifact_or_generic_title"
    text = normalized_title
    # dict.fromkeys 保序去重：保留 RULES 中定义的优先级（高频科室在前）。
    labels = [department for department, pattern in COMPILED_RULES if pattern.search(text)]
    return list(dict.fromkeys(labels)), "title_rule" if labels else "no_precise_signal"


def rewrite_jsonl(path: Path, updates: dict[str, list[str]]) -> int:
    """原地重写 JSONL，按 doc_id 更新 clinical_departments 字段。

    返回实际改写的行数；其它行原样写回。
    失败时 unlink 临时文件保留原文件。
    """
    if not path.exists():
        return 0
    tmp = path.with_suffix(path.suffix + ".tmp")
    changed = 0
    try:
        with path.open("r", encoding="utf-8-sig") as source, tmp.open("w", encoding="utf-8", newline="\n") as target:
            for line_no, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                labels = updates.get(str(record.get("doc_id") or ""))
                if labels:
                    record["clinical_departments"] = labels
                    record["clinical_department"] = labels[0]
                    # 多个科室时记为 compositive（composite scope）；单科为 single。
                    record["department_scope"] = "compositive" if len(labels) > 1 else "single"
                    changed += 1
                target.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return changed


def rewrite_markdown(paths: set[Path], updates: dict[str, list[str]]) -> int:
    """同步修改 Markdown front-matter 中的科室字段。

    仅处理文档级 front-matter，不动正文；保证与 JSONL manifest 保持一致。
    """
    changed = 0
    for path in sorted(paths):
        if not path.exists():
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8", errors="replace"))
        labels = updates.get(str(metadata.get("id") or ""))
        if not labels:
            continue
        metadata["clinical_departments"] = labels
        metadata["clinical_department"] = labels[0]
        metadata["department_scope"] = "compositive" if len(labels) > 1 else "single"
        path.write_text(dump_front_matter(metadata, body), encoding="utf-8", newline="\n")
        changed += 1
    return changed


def reclassify_unknown(data_dir: str | Path = DATA_DIR, apply: bool = False) -> dict[str, Any]:
    """识别当前仍为"未分类"的文档，按标题规则给出建议科室。

    关键安全：
    - 默认 dry-run（apply=False），仅生成 reports/unknown_department_reclassification.jsonl；
    - apply=True 时才同步改 documents.jsonl / documents_raw.jsonl 与对应 Markdown；
    - 输出统计：unknown_before / identified / unresolved / multi_department / 分布等。
    """
    data_path = Path(data_dir)
    documents_path = data_path / "documents.jsonl"
    unknown: list[dict[str, Any]] = []
    markdown_paths: set[Path] = set()
    for record in read_jsonl(documents_path):
        labels = record.get("clinical_departments") or [record.get("clinical_department")]
        if labels != [UNKNOWN_DEPARTMENT]:
            continue
        identified, reason = identify_departments(str(record.get("title") or ""))
        unknown.append({
            "doc_id": str(record["doc_id"]),
            "title": str(record.get("title") or ""),
            "source_institution": str(record.get("source_institution") or ""),
            "document_kind": str(record.get("document_kind") or ""),
            "status": "identified" if identified else "unresolved",
            "clinical_departments": identified or [UNKNOWN_DEPARTMENT],
            "clinical_department": identified[0] if identified else UNKNOWN_DEPARTMENT,
            "department_scope": "compositive" if len(identified) > 1 else "single",
            "method": reason,
        })
        if identified:
            # 仅收集命中的文档的 markdown 路径，未命中的不写。
            for key in ("markdown_clean_path", "markdown_raw_path"):
                if record.get(key):
                    markdown_paths.add(Path(str(record[key])))

    updates = {row["doc_id"]: row["clinical_departments"] for row in unknown if row["status"] == "identified"}
    report_dir = data_path / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "unknown_department_reclassification.jsonl"
    with report_path.open("w", encoding="utf-8", newline="\n") as target:
        for row in unknown:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    changed = {"documents": 0, "documents_raw": 0, "markdown": 0}
    if apply and updates:
        changed["documents"] = rewrite_jsonl(documents_path, updates)
        changed["documents_raw"] = rewrite_jsonl(data_path / "documents_raw.jsonl", updates)
        changed["markdown"] = rewrite_markdown(markdown_paths, updates)

    distribution = Counter(label for labels in updates.values() for label in labels)
    return {
        "unknown_before": len(unknown),
        "identified": len(updates),
        "unresolved": len(unknown) - len(updates),
        "multi_department": sum(len(labels) > 1 for labels in updates.values()),
        "department_distribution": dict(distribution.most_common()),
        "applied": apply,
        "changed": changed,
        "report": str(report_path),
    }


def main() -> None:
    """CLI 入口：默认 dry-run；--apply 才改写。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(reclassify_unknown(args.data_dir, args.apply), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()