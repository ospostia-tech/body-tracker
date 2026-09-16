import { FilesetResolver, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d", { alpha: true });
const permission = $("permission");
const cameraScreen = $("cameraScreen");
const startBtn = $("startBtn");
const stopBtn = $("stopBtn");
const retryBtn = $("retryBtn");
const cameraSwitch = $("cameraSwitch");
const fullscreenBtn = $("fullscreenBtn");
const poseToggle = $("poseToggle");
const mirrorToggle = $("mirrorToggle");
const effectsToggle = $("effectsToggle");
const modelStatus = $("modelStatus");
const trackingStatus = $("trackingStatus");
const fpsValue = $("fpsValue");
const peopleValue = $("peopleValue");
const confidenceValue = $("confidenceValue");
const deviceLabel = $("deviceLabel");
const modeLabel = $("modeLabel");
const errorPanel = $("errorPanel");
const errorTitle = $("errorTitle");
const errorText = $("errorText");

let poseLandmarker = null;
let stream = null;
let facingMode = "user";
let mirror = true;
let poseMode = true;
let hudMode = true;
let running = false;
let lastVideoTime = -1;
let lastFrameTime = performance.now();
let frameCounter = 0;
let smoothedBoxes = [];

const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7], [0,8],[8,9],[9,10],
  [0,11],[11,12],[12,13],[13,14], [14,15], [12,16],[16,17],[17,18],[18,19],
  [11,23],[12,24],[23,24], [23,25],[25,27],[27,29],[29,31], [24,26],[26,28],[28,30]
];

function setCanvasSize() {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function videoToScreen(normX, normY) {
  const rect = canvas.getBoundingClientRect();
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const scale = Math.max(rect.width / vw, rect.height / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  const ox = (rect.width - dw) / 2;
  const oy = (rect.height - dh) / 2;
  let x = ox + normX * dw;
  const y = oy + normY * dh;
  if (mirror) x = rect.width - x;
  return { x, y };
}

function visibleLandmarks(landmarks) {
  return landmarks.filter((p) => (p.visibility ?? 1) > 0.25 && Number.isFinite(p.x) && Number.isFinite(p.y));
}

function calculateBox(landmarks) {
  const pts = visibleLandmarks(landmarks);
  if (!pts.length) return null;
  let minX = 1, minY = 1, maxX = 0, maxY = 0, confidence = 0;
  for (const p of pts) {
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
    confidence += (p.visibility ?? 1);
  }
  confidence /= pts.length;
  const padX = Math.min((maxX - minX) * 0.10, 0.10);
  const padY = Math.min((maxY - minY) * 0.10, 0.10);
  const tl = videoToScreen(Math.max(0, minX - padX), Math.max(0, minY - padY));
  const br = videoToScreen(Math.min(1, maxX + padX), Math.min(1, maxY + padY));
  return {
    x: Math.min(tl.x, br.x),
    y: Math.min(tl.y, br.y),
    w: Math.abs(br.x - tl.x),
    h: Math.abs(br.y - tl.y),
    confidence
  };
}

function smoothBox(prev, next, alpha = 0.28) {
  if (!prev) return next;
  return {
    x: prev.x + (next.x - prev.x) * alpha,
    y: prev.y + (next.y - prev.y) * alpha,
    w: prev.w + (next.w - prev.w) * alpha,
    h: prev.h + (next.h - prev.h) * alpha,
    confidence: prev.confidence + (next.confidence - prev.confidence) * alpha
  };
}

function cornerBox(box) {
  const c = Math.max(16, Math.min(26, box.w * 0.12));
  const x = box.x, y = box.y, w = box.w, h = box.h;
  ctx.save();
  ctx.strokeStyle = "#6cf7c5";
  ctx.shadowColor = "#6cf7c5";
  ctx.shadowBlur = 12;
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  ctx.moveTo(x, y+c); ctx.lineTo(x,y); ctx.lineTo(x+c,y);
  ctx.moveTo(x+w-c,y); ctx.lineTo(x+w,y); ctx.lineTo(x+w,y+c);
  ctx.moveTo(x,y+h-c); ctx.lineTo(x,y+h); ctx.lineTo(x+c,y+h);
  ctx.moveTo(x+w-c,y+h); ctx.lineTo(x+w,y+h); ctx.lineTo(x+w,y+h-c);
  ctx.stroke();
  ctx.restore();
}

function labelBox(box, index) {
  const label = `PERSON #${index + 1}`;
  const conf = `${Math.round(box.confidence * 100)}% CONFIDENCE`;
  const x = Math.max(6, Math.min(box.x, canvas.clientWidth - 150));
  const y = Math.max(10, box.y - 28);
  ctx.save();
  ctx.font = "800 10px Inter, system-ui, sans-serif";
  const width = Math.max(ctx.measureText(label).width, ctx.measureText(conf).width) + 18;
  ctx.fillStyle = "rgba(4,9,13,.78)";
  ctx.strokeStyle = "rgba(108,247,197,.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.roundRect(x, y, width, 24, 7);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.fillText(label, x+9, y+10);
  ctx.fillStyle = "#6cf7c5";
  ctx.font = "700 8px Inter, system-ui, sans-serif";
  ctx.fillText(conf, x+9, y+19);
  ctx.restore();
}

function drawPose(landmarks) {
  const pts = landmarks.map((p) => videoToScreen(p.x, p.y));
  ctx.save();
  ctx.strokeStyle = "rgba(121,168,255,.88)";
  ctx.fillStyle = "rgba(108,247,197,.95)";
  ctx.shadowColor = "rgba(121,168,255,.65)";
  ctx.shadowBlur = 5;
  ctx.lineWidth = 2;
  for (const [a,b] of CONNECTIONS) {
    if (!pts[a] || !pts[b]) continue;
    const va = landmarks[a].visibility ?? 1;
    const vb = landmarks[b].visibility ?? 1;
    if (va < 0.22 || vb < 0.22) continue;
    ctx.beginPath(); ctx.moveTo(pts[a].x,pts[a].y); ctx.lineTo(pts[b].x,pts[b].y); ctx.stroke();
  }
  ctx.shadowBlur = 10;
  for (let i=0;i<pts.length;i++) {
    if ((landmarks[i].visibility ?? 1) < 0.32) continue;
    ctx.beginPath(); ctx.arc(pts[i].x,pts[i].y,i===0 ? 5 : 3,0,Math.PI*2); ctx.fill();
  }
  ctx.restore();
}

function drawHeadMarker(landmarks) {
  const head = landmarks[0];
  if (!head || (head.visibility ?? 0) < 0.3) return;
  const p = videoToScreen(head.x, head.y);
  const t = performance.now() / 500;
  const r = 10 + Math.sin(t)*2;
  ctx.save();
  ctx.strokeStyle = "rgba(108,247,197,.85)";
  ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.arc(p.x,p.y,2.5,0,Math.PI*2); ctx.fillStyle="#6cf7c5"; ctx.fill();
  ctx.restore();
}

function renderResults(results) {
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0,0,rect.width,rect.height);
  const poses = results?.landmarks || [];
  peopleValue.textContent = String(poses.length);
  if (!hudMode) return;
  if (!poses.length) {
    trackingStatus.textContent = "WAITING FOR PERSON...";
    confidenceValue.textContent = "--%";
    smoothedBoxes = [];
    return;
  }
  trackingStatus.textContent = poses.length > 1 ? "PEOPLE DETECTED" : "PERSON DETECTED";
  const nextBoxes = poses.map(calculateBox);
  const boxes = nextBoxes.map((b,i) => {
    smoothedBoxes[i] = smoothBox(smoothedBoxes[i], b);
    return smoothedBoxes[i];
  }).filter(Boolean);
  const avgConf = boxes.length ? boxes.reduce((s,b)=>s+b.confidence,0)/boxes.length : 0;
  confidenceValue.textContent = `${Math.round(avgConf*100)}%`;
  boxes.forEach((box,i) => {
    cornerBox(box);
    labelBox(box,i);
    if (poseMode) drawPose(poses[i]);
    drawHeadMarker(poses[i]);
  });
}

async function createPoseLandmarker() {
  modelStatus.innerHTML = '<span class="dot"></span> LOADING POSE MODEL';
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  try {
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
      runningMode: "VIDEO",
      numPoses: 2,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputSegmentationMasks: false
    });
  } catch (gpuError) {
    console.warn("GPU delegate failed on this device/browser. Falling back to CPU.", gpuError);
    modelStatus.innerHTML = '<span class="dot"></span> LOADING CPU MODEL';
    return await PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
      runningMode: "VIDEO",
      numPoses: 2,
      minPoseDetectionConfidence: 0.45,
      minPosePresenceConfidence: 0.45,
      minTrackingConfidence: 0.45,
      outputSegmentationMasks: false
    });
  }
}

async function openCamera(preferredFacingMode) {
  const base = {
    audio: false,
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      frameRate: { ideal: 30, max: 60 }
    }
  };

  // Try the exact Samsung-compatible camera constraint first.
  try {
    return await navigator.mediaDevices.getUserMedia({
      ...base,
      video: { ...base.video, facingMode: { exact: preferredFacingMode } }
    });
  } catch (firstError) {
    console.warn("Exact camera selection failed; retrying with ideal facingMode.", firstError);
    return await navigator.mediaDevices.getUserMedia({
      ...base,
      video: { ...base.video, facingMode: { ideal: preferredFacingMode } }
    });
  }
}

async function startCamera() {
  hideError();
  startBtn.disabled = true;
  cameraSwitch.disabled = true;
  try {
    if (!window.isSecureContext && location.hostname !== "localhost") {
      throw new Error("SECURE_CONTEXT_REQUIRED");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support camera access. Use current Chrome or Samsung Internet.");
    }

    running = false;
    stopTracksOnly();
    await new Promise((resolve) => setTimeout(resolve, 120));

    stream = await openCamera(facingMode);
    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    mirror = facingMode === "user";
    applyMirror();

    await new Promise((resolve, reject) => {
      if (video.readyState >= 1) return resolve();
      video.addEventListener("loadedmetadata", resolve, { once: true });
      video.addEventListener("error", reject, { once: true });
    });

    await video.play();
    permission.classList.add("hidden");
    cameraScreen.classList.remove("hidden");
    deviceLabel.textContent = /Mobi|Android|SamsungBrowser/i.test(navigator.userAgent) ? "ANDROID" : "DESKTOP";
    setCanvasSize();

    if (!poseLandmarker) poseLandmarker = await createPoseLandmarker();
    modelStatus.innerHTML = '<span class="dot"></span> AI READY';
    trackingStatus.textContent = "SEARCHING...";
    running = true;
    lastVideoTime = -1;
    lastFrameTime = performance.now();
    frameCounter = 0;
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    showError("Unable to start camera", friendlyCameraError(err));
    startBtn.disabled = false;
  } finally {
    cameraSwitch.disabled = false;
  }
}

function stopTracksOnly() {
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
    stream = null;
  }
}

function stopCamera() {
  running = false;
  stopTracksOnly();
  video.pause();
  video.srcObject = null;
  smoothedBoxes = [];
  ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
  cameraScreen.classList.add("hidden");
  permission.classList.remove("hidden");
  startBtn.disabled = false;
  modelStatus.innerHTML = '<span class="dot"></span> READY';
  trackingStatus.textContent = "WAITING FOR CAMERA";
  peopleValue.textContent = "0";
  confidenceValue.textContent = "--%";
}

async function switchCamera() {
  if (!running) return;
  facingMode = facingMode === "user" ? "environment" : "user";
  const wasRunning = running;
  running = false;
  stopTracksOnly();
  if (wasRunning) await startCamera();
}

function applyMirror() {
  video.style.transform = mirror ? "scaleX(-1)" : "none";
  canvas.style.transform = "none";
  mirrorToggle.classList.toggle("active", mirror);
}

function loop() {
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || !poseLandmarker) return;
  if (video.currentTime === lastVideoTime) return;

  lastVideoTime = video.currentTime;
  const now = performance.now();

  try {
    const result = poseLandmarker.detectForVideo(video, now);
    renderResults(result);
  } catch (err) {
    console.warn("Pose inference skipped:", err);
    return;
  }

  frameCounter++;
  if (now - lastFrameTime >= 750) {
    const measuredFps = Math.round(frameCounter * 1000 / (now - lastFrameTime));
    fpsValue.textContent = String(Math.min(measuredFps, 60));
    frameCounter = 0;
    lastFrameTime = now;
  }
}

function friendlyCameraError(err) {
  const name = err?.name || "";
  if (err?.message === "SECURE_CONTEXT_REQUIRED" || (location.protocol !== "https:" && location.hostname !== "localhost")) {
    return "On Android, camera access requires a secure HTTPS website. Open this app from its https:// address, such as GitHub Pages. Do not open index.html from Downloads.";
  }
  if (name === "NotAllowedError" || name === "SecurityError") {
    return "Camera permission was denied. In Samsung/Chrome, open Site settings for this website and set Camera to Allow, then reload the page.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") return "No camera was found on this device.";
  if (name === "NotReadableError" || name === "TrackStartError") return "The camera is busy or unavailable. Close other camera apps and try again.";
  if (name === "OverconstrainedError") return "This camera does not support the requested mode. Try switching cameras or reload the page.";
  if (name === "AbortError") return "The camera start was interrupted. Try START CAMERA again.";
  return err?.message || "The camera could not be started.";
}

function showError(title, text) {
  errorTitle.textContent = title;
  errorText.textContent = text;
  errorPanel.classList.remove("hidden");
}
function hideError() { errorPanel.classList.add("hidden"); }

startBtn.addEventListener("click", startCamera);
retryBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
cameraSwitch.addEventListener("click", switchCamera);
poseToggle.addEventListener("click", () => {
  poseMode = !poseMode;
  poseToggle.classList.toggle("active", poseMode);
  modeLabel.textContent = poseMode ? "POSE" : "BOX";
});
mirrorToggle.addEventListener("click", () => { mirror = !mirror; applyMirror(); });
effectsToggle.addEventListener("click", () => {
  hudMode = !hudMode;
  effectsToggle.classList.toggle("active", hudMode);
  if (!hudMode) ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
});
fullscreenBtn.addEventListener("click", async () => {
  try {
    if (!document.fullscreenElement) await document.documentElement.requestFullscreen();
    else await document.exitFullscreen();
  } catch {}
});
window.addEventListener("resize", setCanvasSize);
window.addEventListener("orientationchange", () => setTimeout(setCanvasSize, 300));
window.addEventListener("pageshow", () => setCanvasSize());

document.addEventListener("visibilitychange", () => {
  if (document.hidden && stream) stream.getVideoTracks().forEach((t) => t.enabled = false);
  if (!document.hidden && stream) stream.getVideoTracks().forEach((t) => t.enabled = true);
});

// Helpful on mobile browsers: release the camera when the page is closed.
window.addEventListener("pagehide", () => stopTracksOnly());

permission.classList.remove("hidden");
setCanvasSize();
