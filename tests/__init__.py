"""
tests/__init__.py
==================
Stub dependensi yang belum terinstall sebelum main.py diimpor.

Package yang di-stub (MISSING di environment ini):
  - faiss          (faiss-cpu tidak tersedia)
  - deep_translator

Semua package lain (jwt, bcrypt, dotenv, numpy, pandas, pyarrow,
joblib, sklearn, sentence_transformers) sudah terinstall dan dipakai asli.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock


def _mock_module(name: str, **attrs) -> types.ModuleType:
    """Buat module palsu dengan atribut yang diberikan."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


# ---------------------------------------------------------------------------
# Stub: faiss  (faiss-cpu tidak kompatibel dengan Python 3.14 di Windows)
# ---------------------------------------------------------------------------
if "faiss" not in sys.modules:
    _mock_index = MagicMock()
    _mock_index.search.return_value = (
        MagicMock(),  # distances
        MagicMock(),  # indices
    )
    _faiss = _mock_module(
        "faiss",
        read_index=MagicMock(return_value=_mock_index),
        IndexFlatIP=MagicMock,
        IndexFlatL2=MagicMock,
        write_index=MagicMock(),
    )
    sys.modules["faiss"] = _faiss

# ---------------------------------------------------------------------------
# Stub: deep_translator  (tidak terinstall)
# ---------------------------------------------------------------------------
if "deep_translator" not in sys.modules:
    _translator_instance = MagicMock()
    _translator_instance.translate.side_effect = lambda text: text  # passthrough
    _GoogleTranslator = MagicMock(return_value=_translator_instance)
    _dt = _mock_module(
        "deep_translator",
        GoogleTranslator=_GoogleTranslator,
    )
    sys.modules["deep_translator"] = _dt
