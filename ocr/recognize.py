"""Image preprocessing + OCR + lookup pipeline.

Given a photo of a single Chinese radical/character, this module:
1. preprocesses the image (crop to ink, upscale, binarize) so a photo of a
   key on paper/screen looks more like the clean glyphs the OCR steps
   expect,
2. classifies it with two independent OCR sources and merges the results
   (best guess first) - see ocr_candidates(): ocr/hanzi_ocr.py's ONNX
   single-character model (primary; markedly more reliable on this app's
   actual input than line-oriented OCR) and Tesseract (chi_sim, secondary,
   contributes extra alternative guesses if it disagrees),
3. resolves each candidate against the 214 Kangxi radicals; candidates that
   are real hanzi but not one of the 214 radicals still get a pinyin
   reading, an offline English gloss (ocr/cedict.py) and (best-effort,
   online, opt-in) a Russian translation.
"""
import os
import re
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pytesseract
from PIL import Image

from . import cedict, hanzi_ocr, radicals_db

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


def _flatten_illumination(gray):
    """Divide out large-scale lighting variation (a shadow, backlight, or an
    uneven-light gradient across the frame) before thresholding.

    The Otsu threshold below is a single global cutoff; on a photo with an
    uneven-light gradient, the darker side of the *background* can fall
    below that cutoff and get classified as ink, turning a whole part of the
    frame solid black and corrupting the crop/recognition (verified: a
    top-to-bottom gray gradient background made a clearly-legible 水 get
    misread as an unrelated character). Estimating the local background via
    a large morphological closing (removes the comparatively thin, dark
    strokes, keeps the slow-varying background) and dividing it out flattens
    that gradient while leaving the strokes dark relative to their now
    roughly-uniform surroundings.

    The closing runs on a downscaled copy so its cost doesn't scale with the
    megapixel count of a real phone photo - the background varies slowly by
    construction, so a small kernel on the downscaled image is already
    reliably larger than the (proportionally shrunk) stroke width.
    """
    h, w = gray.shape[:2]
    scale = min(1.0, 500 / max(h, w, 1))
    small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
    kernel = np.ones((25, 25), np.uint8)
    background = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    if scale < 1.0:
        background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    background = np.maximum(background, 1)
    return cv2.divide(gray, background, scale=255)


def _looks_like_multiple_chars(thresh):
    """Heuristic: ink spread much wider than the overall ink is tall usually
    means the photo has more than one character (e.g. a whole word), not one
    glyph with naturally wide/disconnected strokes.

    This only ever adds an on-page hint (see recognize() /
    "multiple_chars_suspected") - it never blocks recognition - precisely
    because it's a heuristic that can't be made airtight: tested against
    every official Kangxi radical with disconnected strokes (小, 门, 言, 心,
    州, 灬 - the "four dots of fire" radical is the widest/flattest of the
    214 and the one most easily confused with a short row of characters),
    the width/height ratio for a real multi-word photo (~3.1 for 3
    characters, ~7.4 for 7) only reliably separates from the single widest
    legitimate radical (~3.1 for 灬) above roughly 4x - so the threshold
    below is deliberately conservative (catches an obvious multi-character
    photo, stays silent on anything a single radical could produce) rather
    than tuned to catch every 2-3 character mistake, since a false positive
    here would incorrectly cast doubt on a correct single-radical result."""
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 0]
    if len(boxes) < 2:
        return False
    max_area = max(bw * bh for (_, _, bw, bh) in boxes)
    significant = [b for b in boxes if b[2] * b[3] >= 0.08 * max_area]
    if len(significant) < 2:
        return False
    left = min(b[0] for b in significant)
    right = max(b[0] + b[2] for b in significant)
    top = min(b[1] for b in significant)
    bottom = max(b[1] + b[3] for b in significant)
    total_h = bottom - top
    if total_h <= 0:
        return False
    return (right - left) / total_h > 4.0


def preprocess(image_bytes):
    """Bytes -> (PIL.Image cropped to the ink bounding box, upscaled and
    re-binarized onto a padded white square canvas; multiple_chars_suspected
    flag - see _looks_like_multiple_chars)."""
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
    gray = _flatten_illumination(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    multiple_chars_suspected = _looks_like_multiple_chars(thresh)

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

    return Image.fromarray(canvas), multiple_chars_suspected


def tesseract_candidates(pil_img):
    """Run Tesseract in a few single-glyph page segmentation modes, return
    an ordered list of unique hanzi candidates (best guess first).

    Each page-segmentation mode can disagree on what the glyph is; instead
    of trusting whichever mode happens to run first, every candidate is
    ranked by the highest OCR confidence it received across all modes.

    Returns [] if the Tesseract binary isn't installed/configured -
    Tesseract is a secondary source of alternative guesses (see
    ocr_candidates below), not required for recognition to work at all."""
    best_conf = {}
    order = []
    tessdata_dir = _tessdata_dir_config()
    for psm in (10, 7, 6):
        config = f"--psm {psm} {tessdata_dir}".strip()
        try:
            data = pytesseract.image_to_data(
                pil_img, lang="chi_sim", config=config, output_type=pytesseract.Output.DICT
            )
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


# Extra angles tried in ocr_candidates() below, on top of the as-uploaded
# orientation, to cover common real-photo mistakes: a hand-held tilt (a
# photo rotated 25 degrees was verified to already fool the model, so
# +/-20 degrees is tried explicitly rather than relying on the model's own
# tolerance) and holding the source upside down (a full 180-degree flip).
_ROTATION_RETRY_ANGLES = (-20, 20, 180)


def ocr_candidates(pil_img):
    """Merge both OCR sources into one ranked list, best guess first.

    ocr/hanzi_ocr.py's single-character ONNX model is primary - testing
    showed it's dramatically more reliable than Tesseract for this app's
    actual input (one pre-cropped glyph, not a text line). Tesseract's
    guesses (tesseract_candidates) are appended as extra alternatives for
    the "не то распозналось?" UI, for whatever it disagrees on.

    The primary model is also run on a few rotated copies of the same glyph
    (_ROTATION_RETRY_ANGLES) since it was found to misread a moderately
    tilted or upside-down photo with high confidence and no other signal
    that something was off; merging in whatever those extra passes find
    only ever adds candidates, so a straight photo's result is unaffected."""
    primary = list(hanzi_ocr.recognize_candidates(pil_img))  # [(char, confidence), ...]
    for angle in _ROTATION_RETRY_ANGLES:
        rotated = pil_img.rotate(angle, fillcolor=255, expand=True)
        primary.extend(hanzi_ocr.recognize_candidates(rotated))
    primary_sorted = (ch for ch, _ in sorted(primary, key=lambda item: item[1], reverse=True))

    order = list(dict.fromkeys(primary_sorted))
    for ch in tesseract_candidates(pil_img):
        if ch not in order:
            order.append(ch)
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

    Returns {"candidates": [...resolved entries...], "raw": [hanzi,...],
    "multiple_chars_suspected": bool} with the best guess first. Never
    raises for a missing/broken OCR backend - if every source comes up
    empty, "raw"/"candidates" are just empty lists (the UI already has a
    dedicated "not recognized" state).
    """
    pil_img, multiple_chars_suspected = preprocess(image_bytes)
    raw = ocr_candidates(pil_img)
    resolved = [resolve_char(ch, use_online_translate=use_online_translate) for ch in raw]
    return {
        "raw": raw,
        "candidates": resolved,
        "multiple_chars_suspected": multiple_chars_suspected,
    }
