# CC-CEDICT attribution

`cedict_chars.json` in this folder is derived from **CC-CEDICT**
(https://cc-cedict.org/, published by [MDBG](https://www.mdbg.net/)),
a community-maintained Chinese–English dictionary.

- Source: https://www.mdbg.net/chinese/dictionary?page=cc-cedict
- License: [Creative Commons Attribution-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/)
- Built with [`scripts/build_cedict.py`](../scripts/build_cedict.py): keeps
  only single-character entries, picks/merges the non-proper-noun reading(s),
  and trims each definition list to a short gloss for on-screen display and
  as translation-provider input (see `ocr/cedict.py`).

As a derivative of a CC BY-SA 4.0 work, `cedict_chars.json` itself is
licensed under CC BY-SA 4.0 as well.
