"""Loads data/cedict_chars.json: a compact, fully offline single-character
Chinese -> English dictionary derived from CC-CEDICT (see
data/CEDICT_LICENSE.md for license/attribution), covering ~10800 hanzi -
far beyond the 214 radicals + ~76 common characters curated by hand.

Used two ways in ocr/recognize.py:
1. As an offline fallback meaning shown for any character outside the
   curated DB, with no network access required at all.
2. As better source text for the online Chinese -> Russian translation
   step: translating the English gloss (e.g. "to run") gives noticeably
   more reliable results than translating the bare, context-less hanzi
   (which providers often mistranslate or misread as a surname/proper
   noun) - see ocr/recognize.py:translate_ru.
"""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

with open(DATA_DIR / "cedict_chars.json", encoding="utf-8") as f:
    _CEDICT = json.load(f)


def lookup(char):
    """Return {"pinyin": ..., "en": short gloss, "en_full": longer gloss}
    for a single hanzi, or None if it's not in the dictionary."""
    if not char:
        return None
    return _CEDICT.get(char)
