// Thin client: all state lives on the server; we send commands and render status.
const $ = (id) => document.getElementById(id);
// Method parameters shown in the sidebar (ids match the server's TUNABLE keys)
const PARAMS = [
  "bocpd_gamma", "memory_size", "sum_frames", "model_binomial", "use_bayesian_predictive",
  "features", "num_scales", "num_orientations", "pca_components", "max_filter_size",
  "denoise", "denoise_window_size", "denoise_noise_gain", "linearize_response", "flux_est_quantile",
  "white_balance", "tonemap", "entropy_event_threshold", "backend",
];
const TEXT_PARAMS = new Set(["features", "tonemap", "backend"]);
const STREAMS = ["flux", "sample_weight", "gradient", "events"];
// Sample weights at which to label the window size: omega = 2 / (r + 2) for a run of r exposures
const WEIGHT_TICKS = [1, 0.75, 0.5, 0.25, 0.1];
let ws;
let state = null;
let seeking = false;

function send(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

// Method configs in conf/method/, shown under the paper's names
const VARIANT_LABELS = {
  default: "Robust (binomial + Log-Gabor)",
  fast: "Fast (binomial + gradients)",
  binomial: "Binomial (temporal only)",
};
const VARIANT_ORDER = ["default", "fast", "binomial"];

function options(select, names, labels = {}) {
  if (select.options.length !== names.length) {
    select.innerHTML = names.map((n) => `<option value="${n}">${labels[n] ?? n}</option>`).join("");
  }
}

function setParams(params) {
  for (const k of PARAMS) {
    if ($(k).type === "checkbox") $(k).checked = Boolean(params[k]);
    else $(k).value = params[k] ?? (k === "pca_components" ? 0 : "");
  }
}

function renderLegends(s) {
  const stops = s.legends.sample_weight;
  $("weight-bar").style.background = `linear-gradient(to top, ${stops.join(", ")})`;
  $("weight-ticks").innerHTML = WEIGHT_TICKS.map((w) => {
    const binary = Math.round((2 / w - 1) * s.params.sum_frames);
    return `<span style="top:${(1 - w) * 100}%">ω=${w} · ${binary.toLocaleString()} frames</span>`;
  }).join("");
  const ev = s.legends.events;
  $("event-key").innerHTML = [
    [ev[ev.length - 1], "Surprise (ΔH > τ)"],
    [ev[Math.floor(ev.length / 2)], "No event"],
    [ev[0], "Resolution (ΔH < −τ)"],
  ].map(([c, t]) => `<li><i style="background:${c}"></i>${t}</li>`).join("");
}

function render(s) {
  const first = state === null;
  const paramsChanged = !state || JSON.stringify(state.params) !== JSON.stringify(s.params);
  state = s;
  options($("scene"), s.scenes);
  const variants = [...s.variants].sort(
    (a, b) => (VARIANT_ORDER.indexOf(a) + 1 || 99) - (VARIANT_ORDER.indexOf(b) + 1 || 99)
  );
  if (!variants.includes(s.variant)) variants.push(s.variant);
  options($("variant"), variants, { ...VARIANT_LABELS, custom: "Custom" });
  if (document.activeElement !== $("scene")) $("scene").value = s.scene;
  if (document.activeElement !== $("variant")) $("variant").value = s.variant;
  if (first || paramsChanged) {
    setParams(s.params);
    renderLegends(s);
  }

  $("play").textContent = s.playing ? "❚❚ Pause" : "▶ Play";
  $("play").setAttribute("aria-label", s.playing ? "Pause" : "Play");
  const last = Math.max(0, s.num_frames - 1);
  $("seek").max = last;
  if (!seeking && s.index !== null) $("seek").value = s.index;
  $("position").textContent = `${s.index ?? "–"} / ${last}`;
  $("progress-bar").style.width = `${s.index === null ? 0 : ((s.index + 1) / s.num_frames) * 100}%`;
  $("scene-info").textContent =
    `${s.shape[0]}×${s.shape[1]} · ${s.num_frames} frames of ${s.binary_frames_per_frame} binary frames ` +
    `(${s.exposures_per_frame} × ${s.params.sum_frames})`;
  $("info").textContent = `${s.compute_ms} ms/frame · ${s.params.backend} backend`;

  $("busy").hidden = !s.busy;
  $("busy").textContent = s.busy ? `⏳ ${s.busy}…` : "";
  $("error").hidden = !s.error;
  $("error").textContent = s.error;
  if (s.frames) for (const k of STREAMS) $(k).src = `data:image/jpeg;base64,${s.frames[k]}`;
}

function connect() {
  ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "error") {
      $("error").hidden = false;
      $("error").textContent = msg.error;
    } else {
      render(msg);
    }
  };
  ws.onclose = () => setTimeout(connect, 1000);
}

function readParams() {
  const params = {};
  for (const k of PARAMS) {
    const el = $(k);
    params[k] = el.type === "checkbox" ? el.checked : TEXT_PARAMS.has(k) ? el.value : Number(el.value);
  }
  return params;
}

$("play").onclick = () => send({ type: "play", playing: !(state && state.playing) });
$("scene").onchange = () => send({ type: "load", scene: $("scene").value });
// Switching method loads that variant's defaults; Apply then tweaks them
$("variant").onchange = () => send({ type: "params", variant: $("variant").value, params: {} });
$("seek").oninput = () => { seeking = true; };
$("seek").onchange = () => {
  seeking = false;
  send({ type: "seek", index: Number($("seek").value) });
};
$("apply").onclick = () => send({ type: "params", params: readParams() });
// Toggles and menus apply immediately (typed numbers wait for Apply)
for (const k of PARAMS) {
  const el = $(k);
  if (el.type === "checkbox" || el.tagName === "SELECT") el.onchange = () => $("apply").click();
}
document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  if (state && (e.code === "ArrowRight" || e.code === "ArrowLeft")) {
    const index = (state.index ?? 0) + (e.code === "ArrowRight" ? 1 : -1);
    send({ type: "seek", index });
  }
});
connect();
