"""One-off build script: parse the raw CC-CEDICT text file and produce a
compact data/cedict_chars.json with one entry per single simplified hanzi
(pinyin + a short English gloss), for offline use as a translation fallback.

Not part of the running app — run manually whenever CC-CEDICT is refreshed:
    python build_cedict.py <path-to-cedict_ts.u8-or-.txt> <output.json>
"""
import json
import re
import sys

SURNAME_RE = re.compile(r"^surname\b", re.IGNORECASE)
VARIANT_RE = re.compile(r"^(variant of|old variant of|archaic variant of)\b", re.IGNORECASE)
NOISE_DEF_RE = re.compile(r"^(CL:|Taiwan pr\.|also pr\.|see also|see \[)", re.IGNORECASE)
# CC-CEDICT cross-references look like "traditional|simplified[pin1 yin1]" or
# "simplified[pin1 yin1]" embedded inside a definition (e.g. "(used mostly in
# 會水|会水[hui4 shui3])") - collapse each to just the simplified characters.
CROSSREF_RE = re.compile(r"(?:\S+\|)?(\S+)\[[^\]]*\]")
# Leading usage-register tags, e.g. "(bound form) to walk" or
# "(literary) (of hearing) acute" - fine in a full dictionary entry, but
# just noise on a short beginner-facing gloss / as MT input.
REGISTER_PREFIX_RE = re.compile(r"^\([^)]*\)\s*")


def clean_text(s):
    s = CROSSREF_RE.sub(r"\1", s)
    while True:
        stripped = REGISTER_PREFIX_RE.sub("", s)
        if stripped == s or not stripped:
            break
        s = stripped
    return s


def parse_line(line):
    # TRADITIONAL SIMPLIFIED [pin1 yin1] /def1/def2/.../
    m = re.match(r"^(\S+)\s+(\S+)\s+\[([^\]]*)\]\s+/(.*)/$", line.rstrip("\n"))
    if not m:
        return None
    trad, simp, pinyin, defs_raw = m.groups()
    defs = [d.strip() for d in defs_raw.split("/") if d.strip()]
    return trad, simp, pinyin, defs


def is_proper_noun_reading(pinyin):
    first_syllable = pinyin.split(" ", 1)[0]
    return first_syllable[:1].isupper()


def clean_defs(defs):
    """Drop surname/cross-reference/"variant of X" noise and strip usage
    register tags; keep everything else in file order (CC-CEDICT already
    lists the common sense first)."""
    out = []
    for d in defs:
        if SURNAME_RE.match(d) or NOISE_DEF_RE.match(d) or VARIANT_RE.match(d):
            continue
        cleaned = clean_text(d)
        if cleaned:
            out.append(cleaned)
    return out


def pick_entry(entries):
    """entries: list of (pinyin, raw_defs) for one simplified character -
    one per (traditional source, pinyin) combination in CC-CEDICT.

    Multiple traditional characters can collapse to the same simplified
    glyph (e.g. 游/遊 both -> 游), each contributing its own entry; when two
    such entries share the same pinyin reading they're really the same
    word, so their definitions are merged (in file order, deduped) rather
    than picked between - otherwise whichever traditional source happens
    to have more definitions listed wins arbitrarily. That arbitrary pick
    produced wrong "primary" glosses in testing, e.g. simplified 游 (from
    游+遊) picking "to walk" (遊's sense) over "to swim" (游's own, far more
    common, sense) just because 遊's entry listed more definitions.

    Distinct pinyin readings (genuine heteronyms, e.g. 长 cháng/zhǎng) are
    kept separate; among those we pick whichever has the most (non-noise,
    post-merge) definitions as a proxy for "the sense that's actually
    useful", since some minor readings only have one obscure gloss."""
    common = [e for e in entries if not is_proper_noun_reading(e[0])]
    pool = common or entries

    groups = {}
    order = []
    for pinyin, raw_defs in pool:
        if pinyin not in groups:
            groups[pinyin] = []
            order.append(pinyin)
        for d in clean_defs(raw_defs):
            if d not in groups[pinyin]:
                groups[pinyin].append(d)

    best_pinyin = max(order, key=lambda p: len(groups[p]))
    defs = groups[best_pinyin]
    if not defs:
        # every definition on the best-reading entry was filtered as noise
        # (rare) - fall back to its raw definitions, cross-refs cleaned only
        fallback_raw = next(raw for p, raw in pool if p == best_pinyin)
        defs = [clean_text(d) for d in fallback_raw]
    return best_pinyin, defs


def build(src_path, out_path):
    by_char = {}
    with open(src_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parsed = parse_line(line)
            if not parsed:
                continue
            _trad, simp, pinyin, defs = parsed
            if len(simp) != 1:
                continue
            by_char.setdefault(simp, []).append((pinyin, defs))

    out = {}
    for char, entries in by_char.items():
        pinyin, defs = pick_entry(entries)
        if not defs:
            continue
        short = defs[0]
        full = "; ".join(defs[:3])
        out[char] = {"pinyin": pinyin, "en": short, "en_full": full}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, sort_keys=True)

    print(f"{len(out)} characters written to {out_path}")


if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
