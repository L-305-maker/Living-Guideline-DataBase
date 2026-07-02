"""Pipeline entry points for guideline evidence ingestion."""

from src.pipeline.orchestration import run_pipeline

__all__ = ["run_pipeline"]
