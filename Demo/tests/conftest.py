"""Shared pytest fixtures.

The spaCy model is loaded once per session: loading it takes a second and every
analysis test needs the same pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from character_network.extraction import load_pipeline  # noqa: E402


@pytest.fixture(scope="session")
def nlp():
    return load_pipeline()
