// Show a loading state on submit: OCR + preprocessing + (optionally) a
// network call to a translation provider can take a few seconds, and with
// no feedback users tend to click "Распознать" again, firing a duplicate
// upload.
(function () {
  const form = document.querySelector(".upload-form");
  const btn = document.getElementById("submitBtn");
  if (!form || !btn) return;
  form.addEventListener("submit", () => {
    btn.disabled = true;
    btn.textContent = "Распознаём…";
    btn.classList.add("is-loading");
  });
})();

// File preview + clipboard paste (Ctrl+V) on the upload page
(function () {
  const input = document.getElementById("photoInput");
  const preview = document.getElementById("preview");
  const dropText = document.getElementById("fileDropText");
  if (!input) return;

  function showFile(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.hidden = false;
    dropText.textContent = file.name || "Изображение из буфера обмена";
  }

  input.addEventListener("change", () => {
    showFile(input.files && input.files[0]);
  });

  document.addEventListener("paste", (event) => {
    const items = (event.clipboardData || window.clipboardData || {}).items;
    if (!items) return;
    for (const item of items) {
      if (item.kind === "file" && item.type.startsWith("image/")) {
        const file = item.getAsFile();
        if (!file) continue;
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
        showFile(file);
        event.preventDefault();
        break;
      }
    }
  });
})();

// Stroke-order animation via Hanzi Writer (loaded from CDN in base.html).
// Fails silently (leaves the box empty) if offline or the char has no data.
(function () {
  if (typeof HanziWriter === "undefined") return;
  document.querySelectorAll(".stroke-order[data-char]").forEach((el) => {
    const char = el.getAttribute("data-char");
    if (!char) return;
    try {
      const writer = HanziWriter.create(el, char, {
        width: 130,
        height: 130,
        padding: 8,
        showOutline: true,
        strokeAnimationSpeed: 1,
        delayBetweenStrokes: 300,
      });
      el.addEventListener("click", () => writer.animateCharacter());
      writer.animateCharacter();
    } catch (e) {
      // no stroke data available for this character; leave the box empty
    }
  });
})();

// Stroke count for characters outside the 214-key database: read straight
// from the same Hanzi Writer data used for the animation above, so it
// works for any character that has stroke data (thousands, not just 214).
(function () {
  if (typeof HanziWriter === "undefined" || !HanziWriter.loadCharacterData) return;
  document.querySelectorAll(".js-stroke-count[data-char]").forEach((el) => {
    const char = el.getAttribute("data-char");
    if (!char) return;
    HanziWriter.loadCharacterData(char)
      .then((data) => {
        if (!data || !data.strokes) return;
        el.querySelector(".sc-value").textContent = data.strokes.length;
        el.hidden = false;
      })
      .catch(() => {
        // no stroke data for this character; leave the badge hidden
      });
  });
})();
