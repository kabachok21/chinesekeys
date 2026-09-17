"""Loads data/radicals.json (+ data/common_chars.json) and indexes them by
character (incl. variant forms for radicals)."""
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load(filename):
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        return json.load(f)


RADICALS = _load("radicals.json")
for _r in RADICALS:
    _r["kind"] = "radical"

COMMON_CHARS = _load("common_chars.json")
for _c in COMMON_CHARS:
    _c["kind"] = "word"

_INDEX = {}
for _r in RADICALS:
    _INDEX[_r["char"]] = _r
    for _v in _r.get("variants", []):
        _INDEX.setdefault(_v, _r)
for _c in COMMON_CHARS:
    _INDEX.setdefault(_c["char"], _c)


def find_radical(char):
    """Look up a single hanzi: one of the 214 radicals, or (if not a
    radical) one of the extra common-character entries. Returns the DB
    entry (with a "kind" of "radical" or "word") or None."""
    if not char:
        return None
    return _INDEX.get(char)


def get_by_id(radical_id):
    for r in RADICALS:
        if r["id"] == radical_id:
            return r
    return None


def all_radicals():
    return RADICALS


def all_common_chars():
    return COMMON_CHARS


def _matches(entry, q, fields):
    haystack = [entry.get(f) for f in fields]
    haystack += entry.get("variants", [])
    return any(q in str(h).lower() for h in haystack if h)


def search(query):
    """Search over the 214 radicals (char/variants/pinyin incl. tone marks/ru/en)."""
    if not query:
        return RADICALS
    q = query.strip().lower()
    return [r for r in RADICALS if _matches(r, q, ("char", "pinyin", "pinyin_marks", "ru", "en"))]


def search_common(query):
    """Search over the extra common-character list (char/pinyin incl. tone marks/ru)."""
    if not query:
        return COMMON_CHARS
    q = query.strip().lower()
    return [c for c in COMMON_CHARS if _matches(c, q, ("char", "pinyin", "pinyin_marks", "ru"))]
