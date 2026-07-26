"""Build high-recall Recommendation candidates from Section records."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from src.guideline_chunking.block_classifier import classify_block_type
from src.guideline_information.extraction.context_builder import ContextBuilder
from src.guideline_information.ids import (
    ID_ALGORITHM_VERSION,
    make_candidate_id,
    make_source_block_key,
    make_source_revision_id,
)
from src.guideline_information.models import RecommendationCandidate


Classifier = Callable[[str, list[str]], str]


class CandidateBuilder:
    def __init__(
        self,
        *,
        classifier: Classifier = classify_block_type,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.classifier = classifier
        self.context_builder = context_builder or ContextBuilder()

    def build(self, sections: Iterable[dict[str, Any]], pipeline_run_id: str) -> list[RecommendationCandidate]:
        section_list = list(sections)
        candidates: list[RecommendationCandidate] = []
        for index, record in enumerate(section_list):
            text = str(record.get("content") or "").strip()
            if not text:
                continue
            section_path = [str(item) for item in record.get("section_path") or []]
            block_type = self.classifier(text, section_path)
            if block_type != "recommendation_candidate":
                continue
            source_block_key = make_source_block_key(
                str(record.get("doc_id") or ""),
                section_path,
                int(record.get("char_start") or 0),
                int(record.get("char_end") or 0),
            )
            source_revision_id = make_source_revision_id(text)
            context = self.context_builder.build(section_list, index)
            candidate_id = make_candidate_id(source_block_key, source_revision_id, text)
            candidates.append(
                RecommendationCandidate(
                    candidate_id=candidate_id,
                    pipeline_run_id=pipeline_run_id,
                    doc_id=str(record.get("doc_id") or ""),
                    source_block_key=source_block_key,
                    source_revision_id=source_revision_id,
                    source_id_algorithm=ID_ALGORITHM_VERSION,
                    section_path=section_path,
                    candidate_text=text,
                    context_before=context.before,
                    context_after=context.after,
                    candidate_signals=[block_type],
                    candidate_profile={"block_type": block_type},
                    candidate_score=1.0,
                )
            )
        return candidates
