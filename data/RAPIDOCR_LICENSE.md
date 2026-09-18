# RapidOCR / PaddleOCR model attribution

The `.onnx` files in this folder (`PP-OCRv6_rec_small.onnx`,
`PP-OCRv6_det_small.onnx`, `ch_ppocr_mobile_v2.0_cls_mobile.onnx`) are
bundled from **[RapidOCR](https://github.com/RapidAI/RapidOCR)**, an ONNX
conversion of **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**
models.

- RapidOCR source and model conversion: Copyright RapidAI Authors,
  licensed under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
- Upstream model weights: Copyright Baidu / PaddleOCR authors, also
  licensed under Apache License 2.0.

Bundled here (instead of letting RapidOCR download them at first use) so
the app works fully offline and a fresh deploy doesn't depend on an
external CDN being reachable at container start - see
`ocr/hanzi_ocr.py`. Only the `Rec` (recognition) model is actually used by
this app (single pre-cropped glyph -> character, no text detection); `Det`
and `Cls` ship alongside for completeness/future use.
