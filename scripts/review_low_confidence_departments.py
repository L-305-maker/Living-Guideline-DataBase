import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import fitz
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

import classify_guide_departments as base


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/reports/guide_department_classification_20260713"
OUTPUT = ROOT / "data/reports/guide_department_secondary_review_20260714"


def train_models():
    _, texts, labels = base.load_metadata()
    counts = Counter(labels)
    stable = [(text, label) for text, label in zip(texts, labels) if counts[label] >= 5]
    train_x = [text for text, _ in stable]
    train_y = [label for _, label in stable]

    def model(vectorizer):
        pipeline = Pipeline(
            [
                ("vectorizer", vectorizer),
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
        return pipeline.fit(train_x, train_y)

    return (
        model(
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                min_df=2,
                max_features=40_000,
                sublinear_tf=True,
            )
        ),
        model(
            TfidfVectorizer(
                ngram_range=(1, 2),
                min_df=2,
                max_features=60_000,
                sublinear_tf=True,
            )
        ),
    )


def read_text(path):
    try:
        if path.suffix.lower() == ".md":
            return path.read_text(encoding="utf-8", errors="ignore")[:10_000]
        with fitz.open(path) as document:
            return " ".join(page.get_text() for page in document[:5])[:10_000]
    except Exception:
        return ""


def prediction(model, text):
    probabilities = model.predict_proba([text])[0]
    index = probabilities.argmax()
    return model.classes_[index], float(probabilities[index])


def review(row, char_model, word_model):
    path = ROOT / row.file
    title = str(row.evidence_title or path.stem)
    body = read_text(path)
    text = f"{title} {title} {body}"
    char_department, char_confidence = prediction(char_model, text)
    word_department, word_confidence = prediction(word_model, text)

    signals = {}
    source_department = base.SOURCE_DEPARTMENT.get(row.source, "")
    keyword_department = base.keyword_department(title)
    if source_department:
        signals["source"] = source_department
    if keyword_department:
        signals["keyword"] = keyword_department
    if char_confidence >= 0.25:
        signals["char_model"] = char_department
    if word_confidence >= 0.25:
        signals["word_model"] = word_department

    support = defaultdict(list)
    for signal, department in signals.items():
        support[department].append(signal)
    ranked = sorted(support.items(), key=lambda item: (-len(item[1]), item[0]))
    candidate, agreeing = ranked[0] if ranked else (row.department, [])
    confirms_initial = (
        candidate == row.department
        and "keyword" in agreeing
        and ("char_model" in agreeing or "word_model" in agreeing)
    )
    confirmed = (
        len(agreeing) >= 3
        or (len(agreeing) >= 2 and "source" in agreeing)
        or confirms_initial
    )
    return {
        "secondary_department": candidate,
        "secondary_status": "auto_confirmed" if confirmed else "manual_review",
        "secondary_support_count": len(agreeing),
        "secondary_signals": json.dumps(signals, ensure_ascii=False, sort_keys=True),
        "char_department": char_department,
        "char_confidence": round(char_confidence, 4),
        "word_department": word_department,
        "word_confidence": round(word_confidence, 4),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report_dir = args.report.resolve()
    output_dir = args.output.resolve()
    classification = pd.read_csv(
        report_dir / "guide_department_classification.csv", encoding="utf-8-sig"
    )
    queue = pd.read_csv(report_dir / "review_queue.csv", encoding="utf-8-sig")
    char_model, word_model = train_models()
    decisions = pd.DataFrame(
        [review(row, char_model, word_model) for row in queue.itertuples(index=False)]
    )
    reviewed = pd.concat([queue.reset_index(drop=True), decisions], axis=1)
    reviewed["department_changed"] = (
        (reviewed.secondary_status == "auto_confirmed")
        & (reviewed.secondary_department != reviewed.department)
    )

    decision_by_file = reviewed.set_index("file")
    revised = classification.copy()
    revised["initial_department"] = revised.department
    revised["secondary_status"] = "not_reviewed"
    for index, row in revised[revised.file.isin(decision_by_file.index)].iterrows():
        decision = decision_by_file.loc[row.file]
        revised.at[index, "secondary_status"] = decision.secondary_status
        if decision.secondary_status == "auto_confirmed":
            revised.at[index, "department"] = decision.secondary_department

    output_dir.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output_dir / "secondary_review.csv", index=False, encoding="utf-8-sig")
    reviewed[reviewed.secondary_status == "manual_review"].to_csv(
        output_dir / "manual_review_queue.csv", index=False, encoding="utf-8-sig"
    )
    revised.to_csv(
        output_dir / "guide_department_classification_reviewed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    counts = (
        revised.groupby("department").size().reset_index(name="count").sort_values("count", ascending=False)
    )
    counts.to_csv(output_dir / "department_counts_reviewed.csv", index=False, encoding="utf-8-sig")

    auto = reviewed.secondary_status == "auto_confirmed"
    summary = {
        "classification_count": int(len(revised)),
        "input_low_confidence": int(len(reviewed)),
        "auto_confirmed": int(auto.sum()),
        "auto_confirmed_changed": int(reviewed.department_changed.sum()),
        "manual_review_remaining": int((~auto).sum()),
        "classification_count_unchanged": len(revised) == len(classification),
        "only_low_confidence_rows_reviewed": set(reviewed.file) == set(queue.file),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
