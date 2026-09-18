// Upload page: pick a photo (gallery/files, camera, or Ctrl+V), optionally
// crop it - with auto-detected character boxes to tap when the photo has
// several - then submit the cropped, downscaled JPEG instead of the raw
// multi-megapixel original (much faster upload and recognition).
(function () {
  const form = document.getElementById("uploadForm");
  const input = document.getElementById("photoInput");
  if (!form || !input) return;
  const camera = document.getElementById("cameraInput");
  const btn = document.getElementById("submitBtn");
  const slowHint = document.getElementById("slowHint");
  const dropText = document.getElementById("fileDropText");
  const cropper = document.getElementById("cropper");
  const stage = document.getElementById("cropStage");
  const img = document.getElementById("cropImg");
  const box = document.getElementById("cropBox");
  const MAX_SIDE = 1600;
  const MIN_BOX = 24; // px, in displayed-image coordinates

  let work = null; // canvas holding the downscaled, orientation-corrected photo
  let rect = null; // crop rect in displayed-image px: {x, y, w, h}
  let suggestions = [];
  let ready = false;
  let loadToken = 0;

  // OCR + preprocessing + (optionally) a translation call can take a few
  // seconds; without feedback users click again and fire a duplicate upload.
  function setLoading() {
    btn.disabled = true;
    btn.textContent = "Распознаём…";
    setTimeout(() => { slowHint.hidden = false; }, 6000);
    btn.classList.add("is-loading");
  }

  async function decode(file) {
    if (window.createImageBitmap) {
      try { return await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (e) { /* fall through */ }
    }
    return new Promise((resolve, reject) => {
      const el = new Image();
      el.onload = () => resolve(el);
      el.onerror = reject;
      el.src = URL.createObjectURL(file);
    });
  }

  function shown() { return { w: img.clientWidth, h: img.clientHeight }; }

  function drawBox() {
    box.style.left = rect.x + "px";
    box.style.top = rect.y + "px";
    box.style.width = rect.w + "px";
    box.style.height = rect.h + "px";
  }

  function resetRect() {
    const s = shown();
    rect = { x: 0, y: 0, w: s.w, h: s.h };
    drawBox();
  }

  function clearSuggestions() {
    suggestions.forEach((el) => el.remove());
    suggestions = [];
  }

  function showSuggestions(boxes) {
    clearSuggestions();
    const s = shown();
    boxes.forEach((b) => {
      const el = document.createElement("button");
      el.type = "button";
      el.className = "crop-suggest";
      el.setAttribute("aria-label", "Выбрать этот символ");
      el.style.left = b.x * s.w + "px";
      el.style.top = b.y * s.h + "px";
      el.style.width = b.w * s.w + "px";
      el.style.height = b.h * s.h + "px";
      el.addEventListener("click", () => {
        rect = { x: b.x * s.w, y: b.y * s.h, w: b.w * s.w, h: b.h * s.h };
        drawBox();
      });
      stage.appendChild(el);
      suggestions.push(el);
    });
  }

  async function fetchSuggestions(blob, token) {
    clearSuggestions();
    try {
      const fd = new FormData();
      fd.append("photo", blob, "photo.jpg");
      const resp = await fetch("/segment", { method: "POST", body: fd });
      const data = await resp.json();
      if (token === loadToken && data.boxes && data.boxes.length > 1) showSuggestions(data.boxes);
    } catch (e) { /* suggestions are optional */ }
  }

  async function loadFile(file) {
    if (!file) return;
    const token = ++loadToken;
    ready = false;
    dropText.textContent = file.name || "Изображение из буфера обмена";
    try {
      const bmp = await decode(file);
      const bw = bmp.width || bmp.naturalWidth;
      const bh = bmp.height || bmp.naturalHeight;
      const f = Math.min(1, MAX_SIDE / Math.max(bw, bh));
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(bw * f);
      canvas.height = Math.round(bh * f);
      canvas.getContext("2d").drawImage(bmp, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.9));
      if (token !== loadToken) return;
      work = canvas;
      img.onload = () => {
        if (token !== loadToken) return;
        resetRect();
        ready = true;
        fetchSuggestions(blob, token);
      };
      img.src = URL.createObjectURL(blob);
      cropper.hidden = false;
    } catch (e) {
      // can't decode in-browser (e.g. HEIC on desktop): submit the original as-is
      work = null;
      cropper.hidden = true;
    }
  }

  function bindPicker(el) {
    el.addEventListener("change", () => {
      const file = el.files && el.files[0];
      if (el !== input && file) {
        const dt = new DataTransfer();
        dt.items.add(file);
        input.files = dt.files;
      }
      loadFile(file);
    });
  }
  bindPicker(input);
  if (camera) bindPicker(camera);

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
        loadFile(file);
        event.preventDefault();
        break;
      }
    }
  });

  document.getElementById("cropReset").addEventListener("click", () => { if (ready) resetRect(); });

  // Drag the box to move it, drag a corner to resize it, drag on the photo
  // outside the box to draw a new one. Pointer events cover mouse and touch.
  let drag = null;
  function pt(ev) {
    const r = stage.getBoundingClientRect();
    return { x: ev.clientX - r.left, y: ev.clientY - r.top };
  }
  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  stage.addEventListener("pointerdown", (ev) => {
    if (!ready || ev.target.classList.contains("crop-suggest")) return;
    const p = pt(ev);
    const handle = ev.target.dataset && ev.target.dataset.h;
    if (handle) {
      drag = { mode: "resize", handle, start: p, orig: { ...rect } };
    } else if (ev.target === box) {
      drag = { mode: "move", start: p, orig: { ...rect } };
    } else {
      drag = { mode: "draw", start: p };
    }
    stage.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });

  stage.addEventListener("pointermove", (ev) => {
    if (!drag) return;
    const s = shown();
    const p = pt(ev);
    if (drag.mode === "move") {
      rect.x = clamp(drag.orig.x + p.x - drag.start.x, 0, s.w - rect.w);
      rect.y = clamp(drag.orig.y + p.y - drag.start.y, 0, s.h - rect.h);
    } else {
      let x0, y0, x1, y1;
      if (drag.mode === "draw") {
        x0 = drag.start.x; y0 = drag.start.y; x1 = p.x; y1 = p.y;
      } else {
        const o = drag.orig;
        const h = drag.handle;
        x0 = h.includes("w") ? p.x : o.x;
        x1 = h.includes("e") ? p.x : o.x + o.w;
        y0 = h.includes("n") ? p.y : o.y;
        y1 = h.includes("s") ? p.y : o.y + o.h;
      }
      const lx = clamp(Math.min(x0, x1), 0, s.w), hx = clamp(Math.max(x0, x1), 0, s.w);
      const ly = clamp(Math.min(y0, y1), 0, s.h), hy = clamp(Math.max(y0, y1), 0, s.h);
      rect = { x: lx, y: ly, w: Math.max(hx - lx, 1), h: Math.max(hy - ly, 1) };
    }
    drawBox();
  });

  function endDrag() {
    if (!drag) return;
    if (rect.w < MIN_BOX || rect.h < MIN_BOX) resetRect();
    drag = null;
  }
  stage.addEventListener("pointerup", endDrag);
  stage.addEventListener("pointercancel", endDrag);

  // Submit the cropped region; fall back to the untouched original file
  // whenever the in-browser pipeline isn't available.
  let submitting = false;
  form.addEventListener("submit", async (ev) => {
    if (submitting) { ev.preventDefault(); return; }
    if (!work || !ready) { setLoading(); return; }
    ev.preventDefault();
    submitting = true;
    setLoading();
    try {
      const s = shown();
      const kx = work.width / s.w, ky = work.height / s.h;
      const sx = Math.round(rect.x * kx), sy = Math.round(rect.y * ky);
      const sw = Math.max(1, Math.round(rect.w * kx)), sh = Math.max(1, Math.round(rect.h * ky));
      const out = document.createElement("canvas");
      out.width = sw;
      out.height = sh;
      const ctx = out.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, sw, sh);
      ctx.drawImage(work, sx, sy, sw, sh, 0, 0, sw, sh);
      const blob = await new Promise((res) => out.toBlob(res, "image/jpeg", 0.92));
      const dt = new DataTransfer();
      dt.items.add(new File([blob], "photo.jpg", { type: "image/jpeg" }));
      input.files = dt.files;
    } catch (e) { /* submit whatever the input holds */ }
    form.submit();
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
