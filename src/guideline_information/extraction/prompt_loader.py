"""Load versioned prompt resources."""

from __future__ import annotations

from pathlib import Path

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
PROMPT_FILES = {
    "extraction_v1": "extraction_v1.txt",
    "verification_v1": "verification_v1.txt",
    "recommendation_extraction_v1": "extraction_v1.txt",
    "recommendation_verification_v1": "verification_v1.txt",
}


def load_prompt(prompt_version: str) -> str:
    filename = PROMPT_FILES.get(prompt_version)
    if not filename:
        raise ValueError(f"Unknown prompt_version: {prompt_version}")
    return (PROMPT_DIR / filename).read_text(encoding="utf-8")
