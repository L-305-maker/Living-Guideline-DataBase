import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ROOT = Path("data/raw_pdf").resolve()
CONSENSUS = ROOT / "consensus"
DOCUMENTS = Path("data/evidence/documents_raw.jsonl")
OUTPUT = Path("data/reports/guide_department_classification_20260713")
UNKNOWN = {"", "未识别", "未分类"}

SOURCE_DEPARTMENT = {
    "aao_hns_pdf": "耳鼻喉科",
    "aaos_pdf": "骨科",
    "boa_pdf": "骨科",
    "posna_pdf": "骨科",
    "aapd_pdf": "口腔科",
    "bsp_pdf": "口腔科",
    "bspd_pdf": "口腔科",
    "sdcep_pdf": "口腔科",
    "asps_pdf": "整形美容科",
    "bapras_pdf": "整形美容科",
    "isaps_pdf": "整形美容科",
    "ameriburn_pdf": "烧伤科",
    "isbi_pdf": "烧伤科",
    "eshre_pdf": "生殖医学科",
    "asrm_pdf": "生殖医学科",
    "naspghan_pdf": "小儿内科",
    "espghan_pdf": "小儿内科",
    "apsa_pdf": "小儿外科",
    "ada_pdf": "内分泌科",
    "gina_pdf": "呼吸内科",
    "gold_pdf": "呼吸内科",
    "acog_pdf": "妇产科",
    "rcog_pdf": "妇产科",
    "acpgbi_pdf": "肛肠外科",
    "escp_pdf": "肛肠外科",
    "rcpch_pdf": "小儿内科",
    "rch_pdf": "小儿内科",
    "eacts_pdf": "心血管外科",
    "aats_pdf": "心血管外科",
    "sts_pdf": "心血管外科",
    "idsa_pdf": "感染内科",
    "aha_pdf": "心血管内科",
    "wounds_pdf": "烧伤科",
    "eaes_pdf": "普通外科",
    "eras_pdf": "普通外科",
    "ernica_pdf": "小儿外科",
    "wses_pdf": "普通外科",
    "east_pdf": "急诊医学科",
    "esvs_pdf": "血管外科",
    "aaaai_pdf": "过敏/变态反应科",
    "aan_pdf": "神经内科",
    "aasld_pdf": "消化内科",
    "ats_pdf": "呼吸内科",
    "btf_pdf": "神经外科",
    "bts_pdf": "器官移植",
    "cns_pdf": "神经外科",
    "esicm_guidelines": "重症医学科",
    "iwgdf_pdf": "内分泌科",
    "jacc_pdf": "心血管内科",
    "kdigo_pdf": "肾内科",
    "sages_pdf": "普通外科",
    "sccm_guidelines": "重症医学科",
    "svs_pdf": "血管外科",
    "wjes_pdf": "普通外科",
    "wses_pdf": "普通外科",
}

KEYWORD_DEPARTMENT = [
    ("病理科", r"病理(?:诊断|分型|检查|报告|学)|组织病理"),
    ("医学影像科", r"磁共振|MRI|CT扫描|PET.?CT|影像(?:检查|诊断|技术)|超声(?:检查|诊断|造影)|放射(?:诊断|检查|技术)"),
    ("检验科", r"实验室|临床检验|检测试剂|基因检测|分子检测|CLSI|方法学比对|生物标志物"),
    ("口腔科", r"口腔|牙(?:科|周|髓|龈|槽|冠|根|体)|龋|正畸|颌面|根管|种植体"),
    ("眼科", r"眼科|视网膜|青光眼|白内障|角膜|葡萄膜|黄斑|屈光|弱视|眼底|视神经"),
    ("耳鼻喉科", r"耳鼻|鼻窦|鼻炎|咽喉|喉癌|声带|听力|耳聋|中耳|眩晕|前庭"),
    ("血液内科", r"白血病|淋巴瘤|血友病|贫血|血小板|凝血|输血|骨髓瘤|骨髓增生|造血"),
    ("肿瘤科", r"癌|恶性肿瘤|肿瘤诊疗|肿瘤治疗|化疗|放疗|抗肿瘤"),
    ("感染内科", r"感染|结核|肝炎|艾滋|HIV|病毒|细菌|真菌|寄生虫|抗菌药|脓毒|败血症|疫苗接种"),
    ("妇产科", r"妊娠|孕前|孕期|孕妇|产前|产后|分娩|胎儿|围产|引产|前置胎盘|母乳喂养|哺乳期"),
    ("妇科", r"妇科|子宫|卵巢|宫颈|阴道|盆底|绝经|月经"),
    ("生殖医学科", r"不孕|辅助生殖|胚胎移植|生育力|生殖医学|促排卵|精子|卵泡"),
    ("新生儿科", r"新生儿|早产儿|胎龄"),
    ("免疫科/风湿免疫科", r"风湿|类风湿|红斑狼疮|狼疮性|结缔组织病|强直性脊柱炎|干燥综合征|痛风|血管炎"),
    ("肾内科", r"肾病|肾炎|肾衰|肾功能|透析|肾小球|肾移植"),
    ("内分泌科", r"糖尿病|血糖|甲状腺|肥胖|代谢性疾病|激素|骨质疏松|肾上腺|垂体|生长激素|高胆固醇"),
    ("心血管内科", r"冠脉|冠心|心肌|心律|房颤|心力衰竭|高血压|血脂|心电图|超声心动|心脏病|肺栓塞|血栓性疾病|血管超声"),
    ("呼吸内科", r"呼吸|肺功能|肺病|哮喘|气道|支气管|胸膜|戒烟|呼吸机|通气"),
    ("肝胆外科", r"(?:肝|胆|胰).*(?:手术|切除|外科)|肝移植|胆道外科"),
    ("消化内科", r"消化|胃|肠|食管|便秘|腹泻|内镜|胰腺|肝硬化|脂肪肝"),
    ("神经内科", r"神经|脑血管|卒中|癫痫|痴呆|认知|帕金森|头痛|肌无力|脊髓|意识丧失|平山病|脱髓鞘"),
    ("精神心理科", r"精神|抑郁|焦虑|失眠|睡眠障碍|自杀|心理|成瘾|孤独症|多动障碍"),
    ("泌尿外科", r"泌尿|膀胱|前列腺|尿路|尿失禁|神经源性膀胱|阴茎|睾丸"),
    ("皮肤性病科", r"皮肤|银屑病|湿疹|皮炎|白癜风|脱发|痤疮|性传播"),
    ("骨科", r"骨折|骨盆|关节|脊柱|腰椎|颈椎|肌腱|韧带|软组织损伤|运动损伤|骨科|肢体创伤|脊髓损伤"),
    ("急诊医学科", r"急诊|严重创伤|多发伤|休克|复苏|创伤出血|灾难医学"),
    ("重症医学科", r"重症|危重|ICU|体外膜氧合|ECMO"),
    ("麻醉科", r"麻醉|镇静|围术期管理|术前评估|术后镇痛"),
    ("营养科", r"营养|膳食|维生素|微量元素|肠内营养|肠外营养"),
    ("康复医学科", r"康复|功能训练|运动干预"),
    ("器官移植", r"器官移植|心移植|肺移植|肝移植"),
    ("整形美容科", r"整形|美容|皮瓣|瘢痕|脂肪移植"),
    ("烧伤科", r"烧伤|烫伤|冻伤|创面|伤口"),
    ("过敏/变态反应科", r"过敏|变态反应|变应原|荨麻疹"),
    ("老年病科", r"老年|衰弱"),
    ("小儿内科", r"儿童|儿科|小儿|青少年"),
]


def keyword_department(text):
    for department, pattern in KEYWORD_DEPARTMENT:
        if re.search(pattern, text, re.IGNORECASE):
            return department
    return ""


def norm(path):
    return str(Path(path).resolve()).lower()


def load_metadata():
    records = {}
    training_text = []
    training_label = []
    with DOCUMENTS.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            source_file = row.get("source_file")
            if not source_file:
                continue
            title = (row.get("title") or "").strip()
            abstract = (row.get("abstract") or "").strip()
            department = (row.get("clinical_department") or "").strip()
            text = f"{title} {title} {abstract[:2500]}".strip()
            records[norm(source_file)] = {"title": title, "text": text, "department": department}
            if department not in UNKNOWN and text:
                training_text.append(text)
                training_label.append(department)
    return records, training_text, training_label



def read_pdf(path):
    try:
        with fitz.open(path) as document:
            return " ".join(page.get_text() for page in document[:2])[:5000]
    except Exception:
        return ""


def predict(model, text):
    probabilities = model.predict_proba([text])[0]
    order = probabilities.argsort()[::-1]
    classes = model.named_steps["classifier"].classes_
    first, second = order[:2]
    return classes[first], float(probabilities[first]), classes[second], float(probabilities[second])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--include-consensus", action="store_true")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    metadata, texts, labels = load_metadata()
    label_counts = Counter(labels)
    stable = [(text, label) for text, label in zip(texts, labels) if label_counts[label] >= 5]
    texts = [text for text, _ in stable]
    labels = [label for _, label in stable]
    train_x, test_x, train_y, test_y = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    model = Pipeline(
        [
            (
                "vectorizer",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    min_df=2,
                    max_features=40_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(train_x, train_y)
    test_prediction = model.predict(test_x)
    accuracy = accuracy_score(test_y, test_prediction)
    macro_f1 = f1_score(test_y, test_prediction, average="macro")

    cwd = Path.cwd().resolve()
    paths = [
        path.resolve()
        for path in ROOT.rglob("*.pdf")
        if args.include_consensus or CONSENSUS not in path.resolve().parents
    ]
    guides = pd.DataFrame(
        {
            "file": [str(path.relative_to(cwd)) for path in paths],
            "source": [path.relative_to(ROOT).parts[0] for path in paths],
            "absolute": [norm(path) for path in paths],
        }
    )
    guides["department"] = guides["absolute"].map(
        lambda path: metadata.get(path, {}).get("department") or "未识别"
    )
    keyword_correct = Counter()
    keyword_total = Counter()
    source_correct = Counter()
    source_total = Counter()
    for row in guides.itertuples(index=False):
        if row.department in UNKNOWN:
            continue
        info = metadata.get(row.absolute, {})
        title = info.get("title") or Path(row.file).stem
        keyword = keyword_department(title)
        if keyword:
            keyword_total[keyword] += 1
            keyword_correct[keyword] += keyword == row.department
        if row.source in SOURCE_DEPARTMENT:
            source_total[row.source] += 1
            source_correct[row.source] += SOURCE_DEPARTMENT[row.source] == row.department
    keyword_confidence = {
        department: keyword_correct[department] / count for department, count in keyword_total.items()
    }
    source_confidence = {source: source_correct[source] / count for source, count in source_total.items()}

    results = []
    extracted = 0
    for row in guides.itertuples(index=False):
        original = row.department
        info = metadata.get(row.absolute, {})
        title = info.get("title") or Path(row.file).stem
        keyword = keyword_department(title)
        if original not in UNKNOWN:
            department = original
            method = "existing_metadata"
            confidence = 1.0
            second_department = ""
            second_confidence = 0.0
        elif row.source in SOURCE_DEPARTMENT:
            department = SOURCE_DEPARTMENT[row.source]
            method = "source_rule"
            confidence = source_confidence.get(row.source, 0.90)
            second_department = ""
            second_confidence = 0.0
        elif keyword:
            department = keyword
            method = "keyword_rule"
            confidence = keyword_confidence.get(keyword, 0.85)
            second_department = ""
            second_confidence = 0.0
        else:
            text = info.get("text") or f"{title} {title} {Path(row.file).stem}"
            department, confidence, second_department, second_confidence = predict(model, text)
            method = "model_metadata" if info.get("text") else "model_filename"
            if confidence < 0.45:
                pdf_text = read_pdf(Path(row.absolute))
                if pdf_text:
                    extracted += 1
                    department, confidence, second_department, second_confidence = predict(
                        model, f"{title} {title} {pdf_text}"
                    )
                    method = "model_pdf_text"

        margin = confidence - second_confidence
        review = confidence < 0.45 or (bool(second_department) and margin < 0.10)
        results.append(
            {
                "file": row.file,
                "source": row.source,
                "department": department,
                "original_department": original,
                "method": method,
                "confidence": round(confidence, 4),
                "second_department": second_department,
                "second_confidence": round(second_confidence, 4),
                "review_required": review,
                "evidence_title": title,
            }
        )

    output = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_dir / "guide_department_classification.csv", index=False, encoding="utf-8-sig")
    counts = output.groupby("department").size().reset_index(name="count").sort_values("count", ascending=False)
    counts.to_csv(output_dir / "department_counts.csv", index=False, encoding="utf-8-sig")
    cross = output.groupby(["source", "department"]).size().reset_index(name="count")
    cross.to_csv(output_dir / "source_department_counts.csv", index=False, encoding="utf-8-sig")
    review = output[output["review_required"]].sort_values("confidence")
    review.to_csv(output_dir / "review_queue.csv", index=False, encoding="utf-8-sig")

    target = output[output["original_department"].isin(UNKNOWN)]
    labeled = output[~output["original_department"].isin(UNKNOWN)].copy()
    labeled["keyword_guess"] = labeled["evidence_title"].map(keyword_department)
    keyword_eval = labeled[labeled["keyword_guess"] != ""]
    source_eval = labeled[labeled["source"].isin(SOURCE_DEPARTMENT)].copy()
    source_eval["source_guess"] = source_eval["source"].map(SOURCE_DEPARTMENT)
    keyword_accuracy = (keyword_eval["keyword_guess"] == keyword_eval["original_department"]).mean()
    source_accuracy = (source_eval["source_guess"] == source_eval["original_department"]).mean()
    summary = {
        "guide_count": int(len(output)),
        "target_count": int(len(target)),
        "unclassified_after": int(output["department"].isin(UNKNOWN).sum()),
        "department_count": int(output["department"].nunique()),
        "holdout_accuracy": round(float(accuracy), 4),
        "holdout_macro_f1": round(float(macro_f1), 4),
        "keyword_rule_eval_count": int(len(keyword_eval)),
        "keyword_rule_eval_accuracy": round(float(keyword_accuracy), 4),
        "source_rule_eval_count": int(len(source_eval)),
        "source_rule_eval_accuracy": round(float(source_accuracy), 4),
        "pdf_text_extracted": extracted,
        "review_required": int(output["review_required"].sum()),
        "method_counts": {key: int(value) for key, value in output["method"].value_counts().items()},
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
