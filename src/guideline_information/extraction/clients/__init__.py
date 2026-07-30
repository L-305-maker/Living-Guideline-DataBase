"""Structured model clients."""

from src.guideline_information.extraction.clients.base import ModelResponse, StructuredModelClient
from src.guideline_information.extraction.clients.fake import FakeStructuredModelClient

__all__ = ["FakeStructuredModelClient", "ModelResponse", "StructuredModelClient"]
