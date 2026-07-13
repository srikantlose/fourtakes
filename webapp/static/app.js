(() => {
  const STYLES = ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"];

  const dropzone = document.getElementById("dropzone");
  const dropzoneEmpty = document.getElementById("dropzoneEmpty");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const mockCheckbox = document.getElementById("mockCheckbox");
  const generateBtn = document.getElementById("generateBtn");
  const stepper = document.getElementById("stepper");
  const errorBanner = document.getElementById("errorBanner");
  const runSummary = document.getElementById("runSummary");
  const baseCaption = document.getElementById("baseCaption");
  const tabBar = document.querySelector(".tab-bar");
  const tabPanels = document.querySelectorAll(".tab-panel");

  let selectedFile = null;
  let eventSource = null;
  let stylesDone = 0;

  function setStep(name, state) {
    const el = stepper.querySelector(`[data-step="${name}"]`);
    if (!el) return;
    el.classList.remove("active", "done");
    if (state) el.classList.add(state);
  }

  function stepLabel(name, text) {
    const el = stepper.querySelector(`[data-step="${name}"] .step-label`);
    if (el) el.textContent = text;
  }

  function resetUI() {
    setStep("frames", null);
    setStep("base", null);
    setStep("styles", null);
    stepLabel("frames", "Extracting frames");
    stepLabel("base", "Base caption");
    stepLabel("styles", "Four takes");
    errorBanner.hidden = true;
    runSummary.hidden = true;
    stylesDone = 0;

    baseCaption.textContent = "Waiting for a clip…";
    baseCaption.classList.add("placeholder");
    baseCaption.classList.remove("shimmer");

    tabPanels.forEach((panel) => {
      const p = panel.querySelector(".caption-text");
      const btn = panel.querySelector(".copy-btn");
      const oldNote = panel.querySelector(".fallback-note");
      if (oldNote) oldNote.remove();
      p.textContent = "No caption yet";
      p.classList.add("placeholder");
      p.classList.remove("shimmer");
      btn.hidden = true;
    });

    tabBar.querySelectorAll(".tab-btn").forEach((btn) => btn.classList.remove("ready"));
  }

  function handleFile(file) {
    if (!file.type.startsWith("video/")) {
      showError("That doesn't look like a video file.");
      return;
    }
    selectedFile = file;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    dropzoneEmpty.hidden = true;
    generateBtn.disabled = false;
    resetUI();
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    if (fileInput.files.length) handleFile(fileInput.files[0]);
  });

  ["dragenter", "dragover"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove("drag-over");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.hidden = false;
  }

  function setTabText(style, text, isFallback) {
    const panel = document.querySelector(`.tab-panel[data-style="${style}"]`);
    const p = panel.querySelector(".caption-text");
    const btn = panel.querySelector(".copy-btn");
    p.textContent = text;
    p.classList.remove("placeholder", "shimmer");
    btn.hidden = false;
    btn.onclick = () => {
      navigator.clipboard.writeText(text);
      btn.textContent = "Copied";
      setTimeout(() => (btn.textContent = "Copy"), 1200);
    };
    if (isFallback) {
      const note = document.createElement("span");
      note.className = "fallback-note";
      note.textContent = "This style call failed — showing the base caption instead.";
      panel.appendChild(note);
    }
    const tabBtn = tabBar.querySelector(`[data-style="${style}"]`);
    if (tabBtn) tabBtn.classList.add("ready");
  }

  tabBar.addEventListener("click", (e) => {
    const btn = e.target.closest(".tab-btn");
    if (!btn) return;
    tabBar.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    tabPanels.forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    document.querySelector(`.tab-panel[data-style="${btn.dataset.style}"]`).classList.add("active");
  });

  function onEvent(evt) {
    switch (evt.type) {
      case "frames_extracting":
        setStep("frames", "active");
        break;
      case "frames_done":
        setStep("frames", "done");
        stepLabel("frames", `Extracted ${evt.extracted} frames, sent ${evt.sent}`);
        break;
      case "base_captioning":
        setStep("base", "active");
        break;
      case "base_done":
        setStep("base", "done");
        setStep("styles", "active");
        baseCaption.textContent = evt.text;
        baseCaption.classList.remove("placeholder", "shimmer");
        STYLES.forEach((s) => {
          const p = document.querySelector(`.tab-panel[data-style="${s}"] .caption-text`);
          p.textContent = "";
          p.classList.add("shimmer");
        });
        break;
      case "style_done":
        setTabText(evt.style, evt.text, evt.fallback);
        stylesDone += 1;
        stepLabel("styles", `Four takes (${stylesDone}/4)`);
        if (stylesDone >= STYLES.length) setStep("styles", "done");
        break;
      case "done":
        runSummary.hidden = false;
        runSummary.textContent = `Done in ${evt.seconds}s — ${evt.calls} API call(s)${
          evt.mock ? " (mock mode)" : ""
        }`;
        generateBtn.disabled = false;
        if (eventSource) eventSource.close();
        break;
      case "error":
        showError(evt.message || "Something went wrong.");
        generateBtn.disabled = false;
        if (eventSource) eventSource.close();
        break;
    }
  }

  generateBtn.addEventListener("click", async () => {
    if (!selectedFile) return;
    resetUI();
    generateBtn.disabled = true;
    if (eventSource) eventSource.close();

    const form = new FormData();
    form.append("video", selectedFile);
    form.append("mock", mockCheckbox.checked ? "true" : "false");

    let jobId;
    try {
      const res = await fetch("/api/caption", { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `Upload failed (HTTP ${res.status})`);
      }
      const body = await res.json();
      jobId = body.job_id;
    } catch (err) {
      showError(err.message);
      generateBtn.disabled = false;
      return;
    }

    eventSource = new EventSource(`/api/events/${jobId}`);
    eventSource.onmessage = (e) => onEvent(JSON.parse(e.data));
    eventSource.onerror = () => {
      eventSource.close();
      generateBtn.disabled = false;
    };
  });

  resetUI();
})();
