"""基于关键词规则的临床科室分类器。

模块职责：
- DEPARTMENT_RULES：约 40 条科室的关键词词典（中英文双语 + 短语形式）；
- score_clinical_departments：标题加权 + 正文加权的打分函数；
- classify_clinical_departments：多标签分类（返回 departments list + 主科室 + scope）；
- classify_chunk_departments：在文档级科室集合内做 chunk 级再分类；
- allowed_departments：暴露给上层的科室枚举。

设计原则：
- 标题命中权重（title_bonus=4）远高于正文（text_weight=1），防止跨学科正文污染；
- 关键词分中英文两套匹配路径：CJK 走子串包含，英文走单词边界正则；
- 多标签阈值：相对阈值 35%（默认）+ 最小分数 2，避免单次偶然命中拉高科室数量；
- 容错：未命中 → 返回 [UNKNOWN_DEPARTMENT]，下游可识别并兜底。"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable


UNKNOWN_DEPARTMENT = "未分类"


@dataclass(frozen=True)
class DepartmentRule:
    department: str
    keywords: tuple[str, ...]
    title_bonus: int = 4
    text_weight: int = 1


DEPARTMENT_RULES: tuple[DepartmentRule, ...] = (
    DepartmentRule(
        "呼吸内科",
        (
            "respiratory",
            "pulmonary",
            "lung disease",
            "pneumonia",
            "asthma",
            "copd",
            "chronic obstructive pulmonary",
            "sleep apnea",
            "obstructive sleep apnea",
            "oxygen therapy",
            "ventilation",
            "tuberculosis respiratory",
            "bronchitis",
            "bronchiolitis",
            "呼吸",
            "肺炎",
            "哮喘",
            "慢阻肺",
            "睡眠呼吸暂停",
            "支气管炎",
        ),
    ),
    DepartmentRule(
        "消化内科",
        (
            "gastro",
            "digestive",
            "gastritis",
            "hepatitis",
            "cirrhosis",
            "liver disease",
            "inflammatory bowel",
            "crohn",
            "ulcerative colitis",
            "colitis",
            "diarrhea",
            "constipation",
            "pancreatitis",
            "gallstone",
            "gastrointestinal",
            "消化",
            "胃炎",
            "肝炎",
            "肝硬化",
            "肠炎",
            "炎症性肠病",
            "胰腺炎",
        ),
    ),
    DepartmentRule(
        "心血管内科",
        (
            "cardiovascular",
            "cardiac",
            "heart failure",
            "hypertension",
            "coronary",
            "arrhythmia",
            "atrial fibrillation",
            "myocardial infarction",
            "angina",
            "dyslipidemia",
            "cholesterol",
            "lipid",
            "心血管",
            "高血压",
            "冠心病",
            "心律失常",
            "心力衰竭",
            "心肌梗死",
        ),
    ),
    DepartmentRule(
        "神经内科",
        (
            "neurology",
            "neurologic",
            "stroke",
            "epilepsy",
            "seizure",
            "parkinson",
            "dementia",
            "multiple sclerosis",
            "migraine",
            "headache",
            "neuropathy",
            "脑卒中",
            "癫痫",
            "帕金森",
            "痴呆",
            "偏头痛",
            "神经病变",
        ),
    ),
    DepartmentRule(
        "内分泌科",
        (
            "diabetes",
            "diabetic",
            "insulin",
            "thyroid",
            "hyperthyroid",
            "hypothyroid",
            "osteoporosis",
            "endocrine",
            "obesity",
            "metabolic",
            "adrenal",
            "pituitary",
            "糖尿病",
            "甲亢",
            "甲减",
            "骨质疏松",
            "内分泌",
            "肥胖",
        ),
    ),
    DepartmentRule(
        "血液内科",
        (
            "hematology",
            "haematology",
            "leukemia",
            "leukaemia",
            "lymphoma",
            "myeloma",
            "anemia",
            "anaemia",
            "thrombocytopenia",
            "hemophilia",
            "haemophilia",
            "sickle cell",
            "白血病",
            "贫血",
            "血小板减少",
            "淋巴瘤",
            "骨髓瘤",
            "血友病",
        ),
    ),
    DepartmentRule(
        "肾内科",
        (
            "kidney disease",
            "renal",
            "nephrology",
            "nephritis",
            "nephrotic",
            "dialysis",
            "hemodialysis",
            "haemodialysis",
            "uremia",
            "chronic kidney",
            "acute kidney",
            "肾炎",
            "肾病",
            "尿毒症",
            "透析",
            "慢性肾脏病",
        ),
    ),
    DepartmentRule(
        "免疫科/风湿免疫科",
        (
            "rheumatology",
            "rheumatoid",
            "lupus",
            "systemic lupus",
            "arthritis",
            "spondyloarthritis",
            "vasculitis",
            "autoimmune",
            "sjogren",
            "gout",
            "类风湿",
            "红斑狼疮",
            "风湿",
            "免疫",
            "痛风",
        ),
    ),
    DepartmentRule(
        "感染内科",
        (
            "infectious",
            "infection",
            "sepsis",
            "antimicrobial",
            "antibiotic",
            "viral",
            "bacterial",
            "fungal",
            "hiv",
            "aids",
            "covid",
            "influenza",
            "tuberculosis",
            "hepatitis b",
            "hepatitis c",
            "感染",
            "乙肝",
            "结核",
            "败血症",
            "抗菌药",
            "新冠",
            "流感",
        ),
    ),
    DepartmentRule(
        "过敏/变态反应科",
        (
            "allergy",
            "allergic",
            "anaphylaxis",
            "atopic",
            "rhinitis",
            "urticaria",
            "hypersensitivity",
            "过敏",
            "变态反应",
            "过敏性鼻炎",
            "荨麻疹",
            "过敏反应",
        ),
    ),
    DepartmentRule(
        "老年病科",
        (
            "geriatric",
            "older people",
            "older adults",
            "elderly",
            "frailty",
            "老年",
            "衰弱",
        ),
    ),
    DepartmentRule(
        "普通外科",
        (
            "general surgery",
            "appendicitis",
            "hernia",
            "breast disease",
            "breast surgery",
            "thyroid surgery",
            "surgical site",
            "wound management",
            "阑尾炎",
            "疝气",
            "乳腺",
            "普外",
        ),
    ),
    DepartmentRule(
        "骨科",
        (
            "orthopedic",
            "orthopaedic",
            "fracture",
            "spine",
            "spinal",
            "joint replacement",
            "osteoarthritis",
            "low back pain",
            "back pain",
            "scoliosis",
            "骨折",
            "关节炎",
            "脊柱",
            "骨科",
            "腰痛",
        ),
    ),
    DepartmentRule(
        "泌尿外科",
        (
            "urology",
            "urologic",
            "urinary",
            "prostate",
            "renal stone",
            "kidney stone",
            "bladder",
            "incontinence",
            "benign prostatic",
            "肾结石",
            "前列腺",
            "膀胱",
            "泌尿",
        ),
    ),
    DepartmentRule(
        "胸外科",
        (
            "thoracic surgery",
            "lung cancer surgery",
            "esophageal cancer",
            "oesophageal cancer",
            "mediastinal",
            "pneumothorax",
            "肺癌",
            "食管癌",
            "胸外",
            "气胸",
        ),
    ),
    DepartmentRule(
        "心血管外科",
        (
            "cardiac surgery",
            "cardiothoracic surgery",
            "coronary artery bypass",
            "valve surgery",
            "aortic aneurysm",
            "vascular surgery",
            "心脏搭桥",
            "瓣膜",
            "主动脉瘤",
            "血管外科",
        ),
    ),
    DepartmentRule(
        "神经外科",
        (
            "neurosurgery",
            "brain tumor",
            "brain tumour",
            "intracranial",
            "subarachnoid",
            "brain hemorrhage",
            "brain haemorrhage",
            "脑肿瘤",
            "脑出血",
            "神经外科",
        ),
    ),
    DepartmentRule(
        "肝胆外科",
        (
            "hepatobiliary",
            "liver cancer surgery",
            "cholecystectomy",
            "bile duct",
            "gallbladder",
            "cholangiocarcinoma",
            "肝癌",
            "胆结石",
            "胆囊",
            "胆管",
            "肝胆",
        ),
    ),
    DepartmentRule(
        "肛肠外科",
        (
            "colorectal surgery",
            "hemorrhoid",
            "haemorrhoid",
            "anal fistula",
            "perianal",
            "rectal prolapse",
            "痔疮",
            "肛瘘",
            "肛肠",
        ),
    ),
    DepartmentRule(
        "器官移植",
        (
            "transplant",
            "transplantation",
            "organ donor",
            "immunosuppression after transplant",
            "器官移植",
            "移植",
        ),
    ),
    DepartmentRule(
        "烧伤科",
        (
            "burn",
            "burns",
            "scald",
            "烧伤",
            "烫伤",
        ),
    ),
    DepartmentRule(
        "妇科",
        (
            "gynecology",
            "gynaecology",
            "menstrual",
            "endometriosis",
            "menopause",
            "ovarian",
            "cervical",
            "uterine",
            "vaginal",
            "月经",
            "子宫内膜异位",
            "卵巢",
            "宫颈",
            "妇科",
        ),
    ),
    DepartmentRule(
        "产科",
        (
            "pregnancy",
            "pregnant",
            "antenatal",
            "prenatal",
            "postpartum",
            "postnatal",
            "labour",
            "labor and delivery",
            "obstetric",
            "obstetrics",
            "分娩",
            "产后",
            "孕检",
            "妊娠",
            "产科",
        ),
    ),
    DepartmentRule(
        "计划生育科",
        (
            "contraception",
            "contraceptive",
            "abortion",
            "family planning",
            "termination of pregnancy",
            "避孕",
            "流产",
            "计划生育",
        ),
    ),
    DepartmentRule(
        "生殖医学科",
        (
            "fertility",
            "infertility",
            "reproductive",
            "ivf",
            "assisted reproduction",
            "不孕",
            "生殖",
            "辅助生殖",
        ),
    ),
    DepartmentRule(
        "新生儿科",
        (
            "neonatal",
            "newborn",
            "preterm infant",
            "premature infant",
            "新生儿",
            "早产儿",
        ),
    ),
    DepartmentRule(
        "小儿内科",
        (
            "pediatric asthma",
            "paediatric asthma",
            "children with asthma",
            "childhood asthma",
            "pediatric pneumonia",
            "paediatric pneumonia",
            "pediatric sleep apnea",
            "paediatric sleep apnea",
            "儿童哮喘",
            "儿童肺炎",
            "小儿呼吸",
        ),
    ),
    DepartmentRule(
        "小儿内科",
        (
            "pediatric gastro",
            "paediatric gastro",
            "children with inflammatory bowel",
            "pediatric constipation",
            "paediatric constipation",
            "小儿消化",
            "儿童腹泻",
            "儿童便秘",
        ),
    ),
    DepartmentRule(
        "小儿外科",
        (
            "pediatric surgery",
            "paediatric surgery",
            "children undergoing surgery",
            "小儿外科",
        ),
    ),
    DepartmentRule(
        "小儿内科",
        (
            "pediatric",
            "paediatric",
            "child",
            "children",
            "infant",
            "toddler",
            "adolescent",
            "儿童",
            "小儿",
            "婴幼儿",
            "青少年",
        ),
        title_bonus=2,
    ),
    DepartmentRule(
        "耳鼻喉科",
        (
            "ear nose throat",
            "ent",
            "otolaryngology",
            "otitis",
            "sinusitis",
            "hearing loss",
            "tonsil",
            "larynx",
            "nasal",
            "rhinosinusitis",
            "耳鼻喉",
            "中耳炎",
            "鼻窦炎",
            "听力",
            "扁桃体",
        ),
    ),
    DepartmentRule(
        "眼科",
        (
            "ophthalmology",
            "eye",
            "glaucoma",
            "cataract",
            "retinopathy",
            "macular",
            "visual",
            "conjunctivitis",
            "眼科",
            "青光眼",
            "白内障",
            "视网膜",
            "黄斑",
        ),
    ),
    DepartmentRule(
        "口腔科",
        (
            "dental",
            "dentistry",
            "oral health",
            "periodontal",
            "caries",
            "tooth",
            "teeth",
            "口腔",
            "牙",
            "龋齿",
            "牙周",
        ),
    ),
    DepartmentRule(
        "皮肤性病科",
        (
            "dermatology",
            "skin",
            "eczema",
            "psoriasis",
            "acne",
            "sexually transmitted",
            "syphilis",
            "gonorrhea",
            "chlamydia",
            "皮肤",
            "湿疹",
            "银屑病",
            "痤疮",
            "性病",
            "梅毒",
            "淋病",
        ),
    ),
    DepartmentRule(
        "肿瘤科",
        (
            "cancer",
            "carcinoma",
            "tumor",
            "tumour",
            "oncology",
            "chemotherapy",
            "radiotherapy",
            "neoplasm",
            "malignancy",
            "malignant",
            "sarcoma",
            "肿瘤",
            "癌",
            "化疗",
            "放疗",
            "恶性",
        ),
    ),
    DepartmentRule(
        "精神心理科",
        (
            "mental health",
            "psychiatry",
            "psychological",
            "depression",
            "anxiety",
            "schizophrenia",
            "bipolar",
            "autism",
            "adhd",
            "substance use",
            "addiction",
            "insomnia",
            "sleep disorder",
            "精神",
            "心理",
            "抑郁",
            "焦虑",
            "精神分裂",
            "自闭症",
            "失眠",
        ),
    ),
    DepartmentRule(
        "整形美容科",
        (
            "plastic surgery",
            "cosmetic",
            "aesthetic",
            "reconstructive surgery",
            "整形",
            "美容",
        ),
    ),
    DepartmentRule(
        "康复医学科",
        (
            "rehabilitation",
            "physiotherapy",
            "physical therapy",
            "occupational therapy",
            "speech therapy",
            "康复",
            "物理治疗",
        ),
    ),
    DepartmentRule(
        "营养科",
        (
            "nutrition",
            "diet",
            "dietary",
            "malnutrition",
            "enteral feeding",
            "parenteral nutrition",
            "营养",
            "膳食",
            "饮食",
        ),
    ),
    DepartmentRule('妇产科', ('obstetrics and gynecology', 'obstetrics & gynecology', '妇产科', '妇产')),
    DepartmentRule('急诊医学科', ('emergency medicine', 'resuscitation', 'major trauma', '急诊', '复苏', '严重创伤', '多发伤')),
    DepartmentRule('重症医学科', ('critical care', 'intensive care', 'icu', 'ecmo', '重症', '危重', '体外膜氧合')),
    DepartmentRule('麻醉科', ('anesthesia', 'anaesthesia', 'sedation', 'perioperative', '麻醉', '镇静', '围术期')),
    DepartmentRule('医学影像科', ('radiology', 'medical imaging', 'magnetic resonance', 'pet-ct', '影像诊断', '磁共振', '放射诊断')),
    DepartmentRule('检验科', ('laboratory medicine', 'clinical laboratory', 'diagnostic assay', '临床检验', '检测试剂', '方法学比对')),
    DepartmentRule('病理科', ('pathology', 'histopathology', 'pathologic diagnosis', '病理诊断', '组织病理')),
    DepartmentRule('血管外科', ('vascular surgery', 'peripheral arterial', 'venous disease', '血管外科', '外周动脉')),

)


def helper_contains_keyword(text: str, keyword: str) -> bool:
    key = keyword.lower()
    if re.search(r"[\u4e00-\u9fff]", key):
        return key in text
    return re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", text) is not None


def helper_normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").lower()


def score_clinical_departments(title: str = "", abstract: str = "", content: str = "") -> Counter[str]:
    """Return deterministic keyword scores for every matched department."""
    title_text = helper_normalize_text(title)
    body_text = helper_normalize_text(f"{abstract or ''}\n{(content or '')[:12000]}")
    scores: Counter[str] = Counter()
    for rule in DEPARTMENT_RULES:
        for keyword in rule.keywords:
            if helper_contains_keyword(title_text, keyword):
                scores[rule.department] += rule.title_bonus
            if helper_contains_keyword(body_text, keyword):
                scores[rule.department] += rule.text_weight
    return scores


def classify_clinical_departments(
    title: str = "", abstract: str = "", content: str = "",
    allowed: Iterable[str] | None = None, relative_threshold: float = 0.35, minimum_score: int = 2,
) -> dict[str, object]:
    """Return multi-label departments and single/compositive scope."""
    scores = score_clinical_departments(title, abstract, content)
    if allowed is not None:
        allowed_set = set(allowed)
        scores = Counter({key: value for key, value in scores.items() if key in allowed_set})
    if not scores:
        return {"clinical_departments": [UNKNOWN_DEPARTMENT], "clinical_department": UNKNOWN_DEPARTMENT, "department_scope": "single", "department_scores": {}}
    ordered = scores.most_common()
    threshold = max(minimum_score, ordered[0][1] * relative_threshold)
    labels = [department for department, score in ordered if score >= threshold] or [ordered[0][0]]
    return {
        "clinical_departments": labels,
        "clinical_department": labels[0],
        "department_scope": "compositive" if len(labels) > 1 else "single",
        "department_scores": {department: scores[department] for department in labels},
    }


def classify_chunk_departments(
    section_path: Iterable[str], content: str, parent_departments: Iterable[str], chunk_type: str = "",
) -> dict[str, object]:
    """Classify a chunk only among its parent's labels using chunk-local text."""
    parent = [item for item in parent_departments if item and item != UNKNOWN_DEPARTMENT]
    result = classify_clinical_departments(
        title=" > ".join(section_path), abstract=chunk_type, content=content,
        allowed=parent, relative_threshold=0.5, minimum_score=1,
    )
    labels = [item for item in result["clinical_departments"] if item in parent]
    if not labels:
        labels = [parent[0] if parent else UNKNOWN_DEPARTMENT]
    result["clinical_departments"] = labels
    result["clinical_department"] = labels[0]
    result["department_scope"] = "compositive" if len(labels) > 1 else "single"
    return result


def classify_clinical_department(title: str = "", abstract: str = "", content: str = "") -> str:
    """Backward-compatible single-label classifier."""
    return str(classify_clinical_departments(title, abstract, content)["clinical_department"])


def allowed_departments() -> Iterable[str]:
    return list(dict.fromkeys(rule.department for rule in DEPARTMENT_RULES)) + [UNKNOWN_DEPARTMENT]
