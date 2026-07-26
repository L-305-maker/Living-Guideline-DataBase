"""Append-only JSONL repository helpers for guideline information runs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, TypeVar

from pydantic import BaseModel

from src.utils.io import append_jsonl, read_jsonl, write_jsonl

T = TypeVar("T", bound=BaseModel)


def write_models(path: str | Path, records: Iterable[BaseModel]) -> None:
    write_jsonl(path, [record.model_dump(mode="json") for record in records])


def append_model(path: str | Path, record: BaseModel) -> None:
    append_jsonl(path, record.model_dump(mode="json"))


def read_models(path: str | Path, model: type[T]) -> list[T]:
    return [model.model_validate(record) for record in read_jsonl(path)]
