"""Image preprocessing + Tesseract OCR + lookup pipeline.

Given a photo of a single Chinese radical/character, this module:
1. preprocesses the image (crop to ink, upscale, binarize) so a photo of a
   key on paper/screen looks more like the clean glyphs Tesseract expects,
2. runs Tesseract (chi_sim) in a few single-character page-segmentation
   modes and collects candidate hanzi,
3. resolves each candidate against the 214 Kangxi radicals; candidates that
   are real hanzi but not one of the 214 radicals still get a pinyin
   reading and (best-effort, online) a Russian translation.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from PIL import Image

from . import cedict, radicals_db

HANZI_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")
_HAS_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)

# Guards against a slow/hung network call to the translation providers
# blocking a request indefinitely (the app is served synchronously).
_TRANSLATE_TIMEOUT_SECONDS = 6
_translate_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translate")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_TESSDATA = os.path.join(PROJECT_ROOT, "tessdata")


def _configure_tesseract():
    env_cmd = os.environ.get("TESSERACT_CMD")
    if env_cmd:
        pytesseract.pytesseract.tesseract_cmd = env_cmd
    else:
        for candidate in (
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ):
            if os.path.exists(candidate):
                pytesseract.pytesseract.tesseract_cmd = candidate
                break


def _tessdata_dir_config():
    """chi_sim.traineddata ships in the project's own tessdata/ folder (no
    admin rights needed to write into Program Files), so point Tesseract at
    it explicitly unless the user configured TESSDATA_PREFIX themselves."""
    if os.environ.get("TESSDATA_PREFIX"):
        return ""
    if os.path.exists(os.path.join(PROJECT_TESSDATA, "chi_sim.traineddata")):
        # pytesseract splits the config string on whitespace and passes each
        # token straight to subprocess (no shell), so the path must NOT be
        # quoted here even though it contains no spaces.
        return f"--tessdata-dir {PROJECT_TESSDATA}"
    return ""


_configure_tesseract()


class TesseractUnavailable(RuntimeError):
    pass


def preprocess(image_bytes):
    """Bytes -> PIL.Image cropped to the ink bounding box, upscaled and
    re-binarized onto a padded white square canvas."""
    if not image_bytes:
        raise ValueError("Файл пустой или повреждён")
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except cv2.error:
        img = None
    if img is None:
        raise ValueError("Не удалось прочитать изображение — попробуй другой файл")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = cv2.findNonZero(thresh)
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = int(max(w, h) * 0.15) + 1
        x0, y0 = max(x - pad, 0), max(y - pad, 0)
        x1, y1 = min(x + w + pad, gray.shape[1]), min(y + h + pad, gray.shape[0])
        gray = gray[y0:y1, x0:x1]

    h, w = gray.shape[:2]
    scale = max(1, 420 // max(h, w, 1))
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = bw.shape[:2]
    side = int(max(h, w) * 1.3) + 1
    canvas = np.full((side, side), 255, dtype=np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = bw

    return Image.fromarray(canvas)


def ocr_candidates(pil_img):
    """Run Tesseract in a few single-glyph page segmentation modes, return
    an ordered list of unique hanzi candidates (best guess first).

    Each page-segmentation mode can disagree on what the glyph is; instead
    of trusting whichever mode happens to run first, every candidate is
    ranked by the highest OCR confidence it received across all modes."""
    best_conf = {}
    order = []
    tessdata_dir = _tessdata_dir_config()
    for psm in (10, 7, 6):
        config = f"--psm {psm} {tessdata_dir}".strip()
        try:
            data = pytesseract.image_to_data(
                pil_img, lang="chi_sim", config=config, output_type=pytesseract.Output.DICT
            )
        except pytesseract.TesseractNotFoundError as exc:
            raise TesseractUnavailable(str(exc)) from exc
        except Exception:
            continue
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                conf = -1.0
            for ch in text:
                if not HANZI_RE.match(ch):
                    continue
                if ch not in best_conf:
                    order.append(ch)
                    best_conf[ch] = conf
                elif conf > best_conf[ch]:
                    best_conf[ch] = conf
    order.sort(key=lambda ch: best_conf[ch], reverse=True)
    return order


def get_pinyin(char):
    try:
        from pypinyin import pinyin, Style
        result = pinyin(char, style=Style.TONE)
        return " ".join(p[0] for p in result) or None
    except Exception:
        return None


def _clean_translation(text, source_text=None):
    """Reject garbage a translation provider can return for a single,
    context-less phrase: an empty/whitespace-only answer, a lone
    punctuation mark (e.g. "." was seen for 学), or the input echoed back
    untranslated (MyMemory does this for some English phrases, e.g.
    "valley" -> "valley"). Also normalizes casing and strips a trailing
    "." some providers add to single-word answers ("Забыться."), to match
    the plain lowercase style used everywhere else in the app."""
    if not text:
        return None
    text = text.strip()
    if not text or not _HAS_LETTER_RE.search(text):
        return None
    if source_text and text.lower() == source_text.strip().lower():
        return None
    if text.endswith(".") and not (source_text and source_text.rstrip().endswith(".")):
        text = text[:-1].rstrip()
    if text[0].isupper() and not text.isupper():
        text = text[0].lower() + text[1:]
    return text


def _translate_text(source_text, providers):
    for provider in providers:
        future = _translate_executor.submit(provider)
        try:
            text = future.result(timeout=_TRANSLATE_TIMEOUT_SECONDS)
        except Exception:
            continue
        text = _clean_translation(text, source_text)
        if text:
            return text
    return None


def translate_ru(char, cedict_entry=None):
    """Best-effort online translation for a hanzi outside the radical DB.
    Requires internet access; fails silently (returns None) if every
    provider fails or times out.

    When the character is in the local CC-CEDICT dictionary (see
    ocr/cedict.py), its English gloss is translated instead of the bare
    hanzi: EN->RU is a far better-supported language pair for these
    providers than ZH->RU, and an unambiguous phrase like "to run" avoids
    the wrong-sense/surname-reading mistranslations a single, context-less
    character invites (e.g. 跑 alone was seen coming back as "Запуск" -
    "[program/process] launch" - instead of "бежать"). Falls back to
    translating the hanzi directly if there's no dictionary entry, or if
    that route fails."""
    from deep_translator import GoogleTranslator, MyMemoryTranslator

    if cedict_entry is None:
        cedict_entry = cedict.lookup(char)

    if cedict_entry:
        gloss = cedict_entry["en"]
        text = _translate_text(gloss, [
            lambda: MyMemoryTranslator(source="en-GB", target="ru-RU").translate(gloss),
            lambda: GoogleTranslator(source="en", target="ru").translate(gloss),
        ])
        if text:
            return text

    # MyMemory is tried before Google here too: in testing it was far less
    # prone to rate-limiting than deep-translator's unofficial Google
    # endpoint, which returned HTTP 429 (TooManyRequests) consistently.
    return _translate_text(char, [
        lambda: MyMemoryTranslator(source="zh-CN", target="ru-RU").translate(char),
        lambda: GoogleTranslator(source="zh-CN", target="ru").translate(char),
    ])


def resolve_char(char, use_online_translate=False):
    """Build a display entry for one hanzi: radical DB entry if it's one of
    the 214 keys, otherwise a best-effort pinyin/translation fallback.

    The English gloss (entry["en"]) comes from the local CC-CEDICT-derived
    dictionary and needs no network access, so it's always populated when
    available - the Russian translation (entry["ru"]) still requires the
    online_translate opt-in since it calls out to a third-party service."""
    radical = radicals_db.find_radical(char)
    if radical:
        return {"char": char, "source": "radical", "radical": radical}

    cedict_entry = cedict.lookup(char)
    entry = {
        "char": char,
        "source": "fallback",
        "pinyin": get_pinyin(char),
        "en": cedict_entry["en"] if cedict_entry else None,
        "ru": None,
    }
    if use_online_translate:
        entry["ru"] = translate_ru(char, cedict_entry=cedict_entry)
    return entry


def recognize(image_bytes, use_online_translate=False):
    """Full pipeline: preprocess -> OCR -> resolve candidates.

    Returns {"candidates": [...resolved entries...], "raw": [hanzi,...]}
    with the best guess first. Raises TesseractUnavailable if the Tesseract
    binary isn't installed/configured.
    """
    pil_img = preprocess(image_bytes)
    raw = ocr_candidates(pil_img)
    resolved = [resolve_char(ch, use_online_translate=use_online_translate) for ch in raw]
    return {"raw": raw, "candidates": resolved}
