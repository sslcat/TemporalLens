"use strict";
const $ = (id) => document.getElementById(id);
const video = $("video");
const tones = {
  "Throw left": ["←", "#89bdfa"],
  "Throw right": ["→", "#87dfbd"],
  "Zoom in": ["+", "#d0aff5"],
  "Zoom out": ["−", "#ffbf86"],
  Left: ["←", "#89bdfa"],
  Right: ["→", "#87dfbd"],
  Up: ["↑", "#d0aff5"],
  Down: ["↓", "#ffbf86"],
};
let serverStatus;
let activeURL,
  duration = 0,
  playhead;
function element(tag, text, cls) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function showStatus() {
  if (!serverStatus) return;
  $("status").textContent =
    $("mode").value === "gesture" && !serverStatus.gesture_ready
      ? "Gesture setup required"
      : "● " + serverStatus.device.toUpperCase() + " ONLINE";
}
function message(text, error = false) {
  $("events").replaceChildren(
    element("div", text, error ? "message error" : "message"),
  );
}
function seek(time) {
  video.currentTime = time;
}
function draw(result) {
  duration = result.duration;
  $("events").replaceChildren();
  $("count").textContent = String(result.events.length).padStart(2, "0");
  $("details").textContent = duration.toFixed(1) + " s";
  for (let i = 0; i < 5; i++)
    $("ruler").append(element("span", ((duration * i) / 4).toFixed(1) + " s"));
  for (const ev of result.events) {
    const [arrow, color] = tones[ev.label];
    const range = ev.start.toFixed(1) + " – " + ev.end.toFixed(1) + " s";
    const row = element("button", undefined, "row");
    row.style.setProperty("--tone", color);
    const name = element("div");
    name.append(
      element(
        "b",
        $("mode").value === "gesture"
          ? ev.label
          : "Move " + ev.label.toLowerCase(),
      ),
      element("small", range),
    );
    row.append(
      element("span", arrow, "arrow"),
      name,
      element(
        "em",
        (ev.uncertain ? "Uncertain · " : "Score ") + ev.score.toFixed(2),
      ),
    );
    row.onclick = () => seek(ev.start);
    $("events").append(row);
    const block = element("button", arrow + " " + ev.label, "block");
    block.style.setProperty("--tone", color);
    block.style.left = (100 * ev.start) / duration + "%";
    block.style.width = (100 * (ev.end - ev.start)) / duration + "%";
    block.setAttribute(
      "aria-label",
      "Seek to " + ev.label + " at " + ev.start + " seconds",
    );
    block.onclick = row.onclick;
    $("track").append(block);
  }
  if (!result.supported)
    message(
      "Unsupported input. The cyan marker could not be tracked throughout this clip.",
    );
  else if (!result.events.length)
    message(
      $("mode").value === "gesture"
        ? "No gesture met the detection threshold."
        : "No sustained directional motion detected.",
    );
  playhead = element("div", undefined, "playhead");
  $("track").append(playhead);
  updateTime();
}
function updateTime() {
  $("time").textContent = video.currentTime.toFixed(1) + " s";
  if (playhead && duration)
    playhead.style.left =
      Math.min(100, (100 * video.currentTime) / duration) + "%";
}
video.addEventListener("timeupdate", updateTime);
async function load(file) {
  $("mode").disabled = $("demo").disabled = $("upload").disabled = true;
  duration = 0;
  playhead = null;
  for (const id of ["ruler", "track", "details"]) $(id).replaceChildren();
  $("count").textContent = "—";
  $("time").textContent = "0.0 s";
  message("Reading frames and estimating event boundaries…");
  video.pause();
  video.removeAttribute("src");
  video.load();
  if (activeURL) URL.revokeObjectURL(activeURL);
  $("filename").textContent = file ? file.name : "marker-study.mp4";
  $("source").textContent = file ? "UPLOAD" : "EXAMPLE";
  try {
    if (file && file.size > 24 * 1024 * 1024)
      throw Error("Choose a file smaller than 24 MiB.");
    if (!file) {
      const demo = await fetch("/assets/example.mp4");
      if (!demo.ok) throw Error("The example could not be loaded.");
      file = await demo.blob();
    }
    activeURL = URL.createObjectURL(file);
    video.src = activeURL;
    const response = await fetch("/api/analyze?mode=" + $("mode").value, {
      method: "POST",
      body: file,
      headers: { "Content-Type": "application/octet-stream" },
    });
    const result = await response.json();
    if (!response.ok) throw Error(result.detail || "Analysis failed.");
    draw(result);
  } catch (error) {
    message(error.message, true);
  } finally {
    $("mode").disabled = $("demo").disabled = $("upload").disabled = false;
    $("file").value = "";
  }
}
$("mode").onchange = () => {
  const gesture = $("mode").value === "gesture";
  showStatus();
  $("demo").hidden = gesture;
  $("result-link").href = gesture
    ? "/gesture-results.png"
    : "/assets/results.png";
  $("model-name").textContent = gesture
    ? "DINOv2-S/14 + temporal CNN"
    : "Temporal CNN";
  $("subtitle").textContent = gesture
    ? "Experimental detector for four touchless hand gestures."
    : "Inspect directional motion in short, single-marker clips.";
  $("scope").textContent = gesture
    ? "Throw left, throw right, zoom in and zoom out only. The pilot missed most held-out gestures and produced false detections. Other actions are unsupported."
    : "One bright cyan marker on a dark background. Human actions and other subjects are unsupported.";
  if (!gesture) {
    load(null);
    return;
  }
  video.pause();
  video.removeAttribute("src");
  video.load();
  if (activeURL) URL.revokeObjectURL(activeURL);
  activeURL = null;
  playhead = null;
  duration = 0;
  for (const id of ["track", "ruler", "details", "source"])
    $(id).replaceChildren();
  $("filename").textContent = "Choose a gesture clip";
  $("count").textContent = "—";
  $("time").textContent = "0.0 s";
  message("Upload a 5–15 second video to run the local gesture model.");
};
$("upload").onclick = () => $("file").click();
$("file").onchange = () => {
  if ($("file").files[0]) load($("file").files[0]);
};
$("demo").onclick = () => load(null);
fetch("/api/status")
  .then((r) => r.json())
  .then((s) => {
    serverStatus = s;
    showStatus();
  })
  .catch(() => {
    $("status").textContent = "Offline";
  });
load(null);
