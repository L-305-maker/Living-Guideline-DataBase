from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Optional


MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEFAULT_INPUT = "outputs/data.jsonl"
DEFAULT_OUTPUT = "outputs/recommendations.jsonl"
DEFAULT_MIN_SCORE = 0.75

MAX_RECOMMENDATION_CHARS = 1200
MIN_RECOMMENDATION_CHARS = 25


