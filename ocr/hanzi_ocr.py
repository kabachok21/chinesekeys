"""Single-character recognizer built on RapidOCR (ONNX PP-OCRv6 models via
onnxruntime - see data/RAPIDOCR_LICENSE.md), used instead of relying only
on Tesseract for the "what character is this?" step.

Tesseract's chi_sim model (ocr/recognize.py:ocr_candidates) is built for
lines of printed text, not one isolated, possibly hand-photographed glyph,
and testing showed it missing even common everyday characters entirely
(e.g. 跑 never appeared among its candidates - it guessed unrelated 人/包
instead). This recognition-only ONNX model, run directly on the same
pre-cropped glyph image ocr/recognize.py already prepares, was right
15/16 on a test set that included several 20+-stroke rare characters, so
it's used as the primary/highest-ranked candidate source - see
ocr/recognize.py:ocr_candidates for how its output is merged with
Tesseract's (kept as a secondary source of alternative guesses).

Model files are bundled in data/rapidocr_models/ (instead of the network
download RapidOCR does by default on first use) so recognition works
fully offline and a fresh container/instance doesn't depend on reaching
an external CDN before it can serve a request.
"""
import os
import re

import numpy as np

HANZI_RE = re.compile(r"[一-鿿㐀-䶿]")

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "rapidocr_models"
)

_engine = None
_load_attempted = False


def _get_engine():
    """Lazily construct the RapidOCR engine on first use (not at import
    time, so a broken/missing onnxruntime install degrades to
    Tesseract-only instead of breaking app startup) and cache it."""
    global _engine, _load_attempted
    if _load_attempted:
        return _engine
    _load_attempted = True
    try:
        from rapidocr import RapidOCR

        _engine = RapidOCR(
            params={
                "Rec.model_path": os.path.join(_MODELS_DIR, "PP-OCRv6_rec_small.onnx"),
                "Det.model_path": os.path.join(_MODELS_DIR, "PP-OCRv6_det_small.onnx"),
                "Cls.model_path": os.path.join(_MODELS_DIR, "ch_ppocr_mobile_v2.0_cls_mobile.onnx"),
                "Global.log_level": "warning",
                # One thread per session: by default onnxruntime spawns a thread per
                # HOST core, but a small container (Render free tier is ~0.1 CPU)
                # only gets a sliver of one, and the busy-waiting threads fight
                # for it - a single-glyph inference then takes seconds.
                "EngineConfig.onnxruntime.intra_op_num_threads": 1,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        )
    except Exception:
        _engine = None
    return _engine


def recognize_candidates(pil_img):
    """Classify one pre-cropped glyph image (same input as
    ocr/recognize.py's Tesseract path). Returns an ordered list of
    (char, confidence) tuples, best guess first.

    Returns [] if the model is unavailable (import/init failure) rather
    than raising - Tesseract stays a working fallback OCR source either
    way, so a broken RapidOCR install shouldn't take down recognition."""
    engine = _get_engine()
    if engine is None:
        return []

    arr = np.array(pil_img.convert("RGB"))
    try:
        result = engine(arr, use_det=False, use_cls=False, use_rec=True)
    except Exception:
        return []

    if not result or not result.txts:
        return []

    candidates = []
    seen = set()
    for txt, score in zip(result.txts, result.scores):
        for ch in txt:
            if ch in seen or not HANZI_RE.match(ch):
                continue
            seen.add(ch)
            candidates.append((ch, float(score)))
    return candidates
