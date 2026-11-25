/* Panomotion graphical player - dependency free.
 * Renders the equirectangular view and a software perspective sphere view,
 * sharing the same spherical directions / trajectories / selection.
 */
"use strict";

const D2R = Math.PI / 180, R2D = 180 / Math.PI;
const TAU = Math.PI * 2;

const $ = (id) => document.getElementById(id);
const fjson = (u) => fetch(u).then((r) => {
  if (!r.ok) throw new Error(u + " -> " + r.status);
  return r.json();
});

// ---------------------------------------------------------------- math ----
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]];
}
function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
function normz(a) { const n = norm(a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; }
function lonLatToDir(lon, lat) {
  const c = Math.cos(lat);
  return [c * Math.cos(lon), Math.sin(lat), c * Math.sin(lon)];
}
function dirToLonLat(d) {
  return [Math.atan2(d[2], d[0]), Math.asin(Math.max(-1, Math.min(1, d[1])))];
}
function pixelToDir(u, v, W, H) {
  return lonLatToDir((u + 0.5) / W * TAU - Math.PI,
                     Math.PI / 2 - (v + 0.5) / H * Math.PI);
}
function dirToPixel(d, W, H) {
  const [lon, lat] = dirToLonLat(d);
  return [(lon + Math.PI) / TAU * W - 0.5,
          (Math.PI / 2 - lat) / Math.PI * H - 0.5];
}
function rotX(v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
}
function rotY(v, a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
}
function rodrigues(omega, d) {
  const th = norm(omega);
  if (th < 1e-9) return normz([d[0] + omega[0], d[1] + omega[1], d[2] + omega[2]]);
  const k = [omega[0] / th, omega[1] / th, omega[2] / th];
  const kd = dot(k, d);
  const kxd = cross(k, d);
  const c = Math.cos(th), s = Math.sin(th);
  return normz([
    d[0] * c + kxd[0] * s + k[0] * kd * (1 - c),
    d[1] * c + kxd[1] * s + k[1] * kd * (1 - c),
    d[2] * c + kxd[2] * s + k[2] * kd * (1 - c)]);
}
function omegaBetween(d0, d1) {
  const th = Math.acos(Math.max(-1, Math.min(1, dot(d0, d1))));
  if (th < 1e-9) return [0, 0, 0];
  const ax = normz(cross(d0, d1));
  return [ax[0] * th, ax[1] * th, ax[2] * th];
}
function slerp(d0, d1, t) {
  return rodrigues(omegaBetween(d0, d1).map((w) => w * t), d0);
}
function angDist(a, b) {
  return Math.acos(Math.max(-1, Math.min(1, dot(a, b)))) * R2D;
}

// --------------------------------------------------------------- state ----
const state = {
  runs: [], runName: "demo", meta: null, report: null, tracks: [],
  hotspots: null, hotspotDoc: null, userHotspots: [],
  pairIndex: 0, pairCache: {}, pairData: null, loadSeq: 0,
  layer: "frames",
  show: { est: true, gt: true, tracks: true, seam: true, polar: true, rejected: false, hotspots: true },
  selection: null,
  playing: false, timer: null, intervalMs: 700,
  yaw: Math.PI / 2, pitch: 0.12, fov: 68,
  sphereDirty: true,
  lastPickDir: null,
  followHotspot: false,
};

const eqCanvas = $("equirectCanvas"), eqCtx = eqCanvas.getContext("2d");
const spCanvas = $("sphereCanvas"), spCtx = spCanvas.getContext("2d");
const imgCache = new Map();
const off = document.createElement("canvas");
const offCtx = off.getContext("2d", { willReadFrequently: true });
let sourceData = null;

function artifactUrl(rel) {
  return `/artifacts/${encodeURIComponent(state.runName)}/${rel}`;
}
function loadImage(url) {
  if (imgCache.has(url)) return imgCache.get(url);
  const p = new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = reject;
    im.src = url;
  });
  imgCache.set(url, p);
  return p;
}

// ------------------------------------------------------------ data IO ----
async function loadRuns() {
  const d = await fjson("/api/runs");
  state.runs = d.runs;
  const sel = $("runSelect");
  sel.innerHTML = "";
  if (!d.runs.length) {
    sel.innerHTML = '<option>（暂无数据，请构建 Demo）</option>';
    return;
  }
  for (const r of d.runs) {
    const o = document.createElement("option");
    o.value = r.name; o.textContent = r.name;
    sel.appendChild(o);
  }
  const want = d.runs.find((r) => r.name === "demo") || d.runs[0];
  sel.value = want.name;
  state.runName = want.name;
}

async function selectRun(name) {
  state.runName = name;
  state.loadSeq++;
  state.pairCache = {};
  state.pairIndex = 0;
  state.selection = null;
  const [meta, report, tracks, hotspotDoc] = await Promise.all([
    fjson(`/api/meta?name=${name}`),
    fjson(`/api/report?name=${name}`),
    fjson(`/api/tracks?name=${name}`),
    fjson(`/api/hotspots?name=${name}`).catch(() => null),
  ]);
  state.meta = meta; state.report = report; state.tracks = tracks;
  state.hotspotDoc = hotspotDoc;
  state.hotspots = hotspotDoc ? hotspotDoc.hotspots : [];
  state.userHotspots = [];
  $("pairSlider").max = Math.max(0, report.pairs.length - 1);
  $("pairSlider").value = 0;
  renderSummary();
  renderVideos();
  renderAnomalies();
  renderTrackList();
  renderHotspotPanel();
  await loadPair(0);
}

async function loadPair(i) {
  const seq = ++state.loadSeq;
  state.pairIndex = i;
  $("pairSlider").value = i;
  if (!state.pairCache[i]) {
    state.pairCache[i] = await fjson(`/api/pair?name=${state.runName}&i=${i}`);
  }
  if (seq !== state.loadSeq) return; // a newer selection/pair superseded this
  state.pairData = state.pairCache[i];
  $("pairLabel").textContent = `帧对 ${i} → ${i + 1}`;
  renderPairStats();
  renderSeam();
  renderInspector();
  renderHotspotPanel();
  drawEquirect();
  refreshSourceAndSphere();
}

// ------------------------------------------------------------- helpers ----
function nodeSrcDir(k, pd) {
  const cols = pd.cols, W = pd.width, H = pd.height;
  const i = k % cols, j = (k / cols) | 0;
  return pixelToDir((i + 0.5) * W / cols, (j + 0.5) * H / pd.rows, W, H);
}
function nodeTargetDir(k, pd) {
  return lonLatToDir(pd.target_lon[k] * D2R, pd.target_lat[k] * D2R);
}
function nodeGtDir(k, pd) {
  if (!pd.gt_lon) return null;
  return lonLatToDir(pd.gt_lon[k] * D2R, pd.gt_lat[k] * D2R);
}
function classBadge(c) {
  if (c === 1) return '<span class="badge global">相机旋转一致</span>';
  if (c === 2) return '<span class="badge local">局部物体运动</span>';
  return '<span class="badge rejected">未支持 / 已填充</span>';
}

// --------------------------------------------------------- equirect draw --
function currentFrameUrl() {
  const i = state.pairIndex;
  if (state.layer === "reproj") return artifactUrl(`reproj/reproj_${String(i).padStart(3, "0")}.jpg`);
  if (state.layer === "error") return artifactUrl(`error/error_${String(i).padStart(3, "0")}.jpg`);
  return artifactUrl(`frames/frame_${String(i).padStart(3, "0")}.jpg`);
}

async function drawEquirect() {
  if (!state.pairData) return;
  const url = currentFrameUrl();
  const im = await loadImage(url);
  eqCtx.clearRect(0, 0, eqCanvas.width, eqCanvas.height);
  eqCtx.drawImage(im, 0, 0, eqCanvas.width, eqCanvas.height);
  drawEquirectOverlays();
}

function drawEquirectOverlays() {
  const pd = state.pairData;
  if (!pd) return;
  const W = eqCanvas.width, H = eqCanvas.height;
  const ctx = eqCtx;

  if (state.show.polar) {
    const y75 = (Math.PI / 2 - 75 * D2R) / Math.PI * H;
    ctx.fillStyle = "rgba(150,120,255,0.10)";
    ctx.fillRect(0, 0, W, y75);
    ctx.fillRect(0, H - y75, W, y75);
    ctx.strokeStyle = "rgba(200,180,255,0.55)";
    ctx.setLineDash([7, 6]); ctx.lineWidth = 1.2;
    for (const y of [y75, H - y75]) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "rgba(220,210,255,0.8)"; ctx.font = "12px sans-serif";
    ctx.fillText("北纬 75° 极区：局部坐标基切换", 8, y75 - 5);
    ctx.fillText("南纬 75°", 8, H - y75 + 14);
  }

  if (state.show.seam) {
    ctx.strokeStyle = "#ff3cdc"; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(2, 0); ctx.lineTo(2, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W - 2, 0); ctx.lineTo(W - 2, H); ctx.stroke();
    ctx.setLineDash([8, 6]); ctx.lineWidth = 1.4;
    ctx.strokeStyle = "rgba(255,120,230,0.9)";
    ctx.beginPath(); ctx.moveTo(8, 0); ctx.lineTo(8, H); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(W - 8, 0); ctx.lineTo(W - 8, H); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff9be9"; ctx.font = "bold 12px sans-serif";
    ctx.fillText("经度接缝 −180°（与 +180° 邻接）", 14, 16);
    const t = "+180°"; const tw = ctx.measureText(t).width;
    ctx.fillText(t, W - tw - 12, 16);
  }

  // node dots
  const n = pd.cls.length;
  for (let k = 0; k < n; k++) {
    const c = pd.cls[k];
    if (c === 0 && !state.show.rejected) continue;
    const d = nodeSrcDir(k, pd);
    const [x, y] = dirToPixel(d, W, H);
    ctx.fillStyle = c === 1 ? "rgba(90,200,255,.9)"
      : c === 2 ? "rgba(255,176,40,.95)" : "rgba(120,132,160,.5)";
    ctx.beginPath(); ctx.arc(x + 0.5, y + 0.5, c ? 2.4 : 1.6, 0, TAU); ctx.fill();
  }

  // arrows (strided)
  const cols = pd.cols;
  for (let k = 0; k < n; k++) {
    const i = k % cols, j = (k / cols) | 0;
    if (i % 2 || j % 2) continue;
    const c = pd.cls[k];
    if (state.show.est && c > 0) {
      drawEqArrow(nodeSrcDir(k, pd), nodeTargetDir(k, pd),
        c === 2 ? "#ffb028" : "#7fd8ff", 1.6);
    }
    if (state.show.gt && pd.gt_lon) {
      drawEqArrow(nodeSrcDir(k, pd), nodeGtDir(k, pd), "rgba(125,255,176,.85)", 1.1);
    }
  }

  if (state.show.tracks) drawAllTracksEquirect();
  drawHotspotsEquirect();

  if (state.selection && state.selection.kind === "node") {
    const k = state.selection.k;
    drawEqArrow(nodeSrcDir(k, pd), nodeTargetDir(k, pd), "#ffffff", 3);
    const [x, y] = dirToPixel(nodeSrcDir(k, pd), W, H);
    eqCtx.strokeStyle = "#fff"; eqCtx.lineWidth = 2;
    eqCtx.beginPath(); eqCtx.arc(x, y, 7, 0, TAU); ctx.stroke();
  }
  if (state.selection && state.selection.kind === "track") {
    drawTrackEquirect(state.selection.id, "#ffe14d", 3.2);
  }
}

function drawEqArrow(d0, d1, color, width) {
  const W = eqCanvas.width, H = eqCanvas.height, ctx = eqCtx;
  const pts = [];
  for (let s = 0; s <= 8; s++) {
    const [x, y] = dirToPixel(slerp(d0, d1, s / 8), W, H);
    pts.push([x, y]);
  }
  ctx.strokeStyle = color; ctx.lineWidth = width;
  let run = [pts[0]];
  const flush = () => {
    if (run.length >= 2) { ctx.beginPath(); ctx.moveTo(run[0][0], run[0][1]);
      for (const p of run) ctx.lineTo(p[0], p[1]); ctx.stroke(); }
  };
  for (let i = 1; i < pts.length; i++) {
    if (Math.abs(pts[i][0] - pts[i - 1][0]) > W / 2) { flush(); run = [pts[i]]; }
    else run.push(pts[i]);
  }
  flush();
  const end = pts[pts.length - 1], prev = pts[pts.length - 2];
  const a = Math.atan2(end[1] - prev[1], end[0] - prev[0]);
  ctx.beginPath(); ctx.moveTo(end[0], end[1]);
  ctx.lineTo(end[0] - 7 * Math.cos(a - 0.42), end[1] - 7 * Math.sin(a - 0.42));
  ctx.moveTo(end[0], end[1]);
  ctx.lineTo(end[0] - 7 * Math.cos(a + 0.42), end[1] - 7 * Math.sin(a + 0.42));
  ctx.stroke();
}

function trackDirs(t) {
  return t.est_lon.map((lo, i) => lonLatToDir(lo * D2R, t.est_lat[i] * D2R));
}
function trackReliable(t) {
  if (t.median_error_deg != null) return t.median_error_deg < 1.0;
  const acc = t.accepted.filter(Boolean).length / Math.max(1, t.accepted.length);
  return acc >= 0.6;
}

// ------------------------------------------------------------ hotspots ----
function allHotspotObjects() {
  // Pipeline hotspots (with per-frame chains) plus editor-added ones.
  const objs = [];
  for (const h of state.hotspots || []) {
    const frame = h.frames && h.frames[String(state.pairIndex)];
    const dir = frame ? frame.dir : h.anchor_dir;
    objs.push({ id: h.hotspot_id, name: h.name, dir, radiusDeg: h.radius_deg,
                source: "pipeline" });
  }
  state.userHotspots.forEach((h, idx) => {
    objs.push({ id: 1000 + idx, name: h.name, dir: h.dir,
                radiusDeg: h.radiusDeg, source: "editor" });
  });
  return objs;
}

function hotspotOutlineDirs(h) {
  const c = ST.normalize(h.dir);
  const up = [0, 1, 0];
  let b1 = ST.normalize(ST.cross(up, c));
  if (ST.norm(ST.cross(up, c)) < 1e-9) b1 = ST.normalize(ST.cross([1, 0, 0], c));
  const b2 = ST.cross(c, b1);
  const r = h.radiusDeg * D2R;
  const out = [];
  for (let k = 0; k < 64; k++) {
    const a = k / 64 * TAU;
    const tx = Math.cos(a) * r, ty = Math.sin(a) * r;
    const theta = Math.hypot(tx, ty);
    const move = [b1[0] * tx + b2[0] * ty, b1[1] * tx + b2[1] * ty,
                  b1[2] * tx + b2[2] * ty];
    if (theta < 1e-12) { out.push(c); continue; }
    const u = [move[0] / theta, move[1] / theta, move[2] / theta];
    const s = Math.sin(theta), co = Math.cos(theta);
    out.push(ST.normalize([
      c[0] * co + u[0] * s, c[1] * co + u[1] * s, c[2] * co + u[2] * s]));
  }
  return out;
}

function drawHotspotsEquirect() {
  if (!state.show.hotspots) return;
  const W = eqCanvas.width, H = eqCanvas.height, ctx = eqCtx;
  for (const h of allHotspotObjects()) {
    const dirs = hotspotOutlineDirs(h);
    const pts = dirs.map((d) => {
      const [x, y] = dirToPixel(d, W, H);
      return { x, y };
    });
    const color = h.source === "editor" ? "#7dff8a" : "#ffd166";
    const frags = ST.splitProjected(pts, (a, b) =>
      Math.abs(b.x - a.x) > W / 2, h.id);
    ctx.strokeStyle = color; ctx.lineWidth = 2.2;
    for (const f of frags) {
      ctx.beginPath();
      ctx.moveTo(f.points[0].x, f.points[0].y);
      f.points.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    }
    const [cx, cy] = dirToPixel(ST.normalize(h.dir), W, H);
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(cx, cy, 3.4, 0, TAU); ctx.fill();
  }
}

function drawHotspotsSphere() {
  if (!state.show.hotspots) return;
  const ctx = spCtx;
  for (const h of allHotspotObjects()) {
    const color = h.source === "editor" ? "rgba(125,255,138,.95)" : "rgba(255,209,102,.95)";
    const dirs = hotspotOutlineDirs(h);
    const pts = dirs.map((d) => projectDir(d));
    const frags = ST.splitProjected(pts, () => false, h.id);
    ctx.strokeStyle = color; ctx.lineWidth = 2.2;
    for (const f of frags) {
      ctx.beginPath();
      ctx.moveTo(f.points[0].x, f.points[0].y);
      f.points.forEach((p) => ctx.lineTo(p.x, p.y));
      ctx.stroke();
    }
    const pc = projectDir(ST.normalize(h.dir));
    if (pc) { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(pc.x, pc.y, 3.6, 0, TAU); ctx.fill(); }
  }
}

function renderHotspotPanel() {
  const list = $("hotspotList");
  const objs = allHotspotObjects();
  list.innerHTML = objs.length
    ? objs.map((h) => `<div class="hs-row" data-id="${h.id}">
        <span class="hs-dot ${h.source}"></span>${h.name}
        <span class="muted">r=${h.radiusDeg.toFixed(1)}° · id=${h.id}</span></div>`).join("")
    : '<span class="muted">暂无热点。</span>';
  const ev = $("hotspotEvidence");
  const doc = state.hotspotDoc;
  if (doc && doc.projection_checks) {
    ev.innerHTML = doc.projection_checks.map((c) => `
      <div class="hs-check ${c.same_spherical_object ? "ok" : "bad"}">
        ${c.name}：等距柱状 ${c.equirect_fragment_count} 段 / 透视 ${c.perspective_fragment_count} 段，
        同一球面对象 = ${c.same_spherical_object ? "是" : "否"}，
        有效区 ${(c.valid_region_coverage * 100).toFixed(1)}%
      </div>`).join("")
      + (doc.gt_angular_error || []).map((g) => `
      <div class="hs-check ok">${g.name}：GT 球面角误差中位 ${g.median_error_deg}°，
        最大 ${g.max_error_deg}°，有效比例 ${g.valid_fraction}</div>`).join("");
  }
}

function drawAllTracksEquirect() {
  for (const t of state.tracks) {
    if (state.selection?.kind === "track" && state.selection.id === t.id) continue;
    const rel = trackReliable(t);
    eqCtx.globalAlpha = rel ? 1 : 0.22;
    drawTrackEquirect(t.id, t.stress === "object" ? "rgba(255,176,40,.95)"
      : "rgba(90,220,200,.6)", rel ? 1.6 : 1.1, false);
    eqCtx.globalAlpha = 1;
  }
}
function drawTrackEquirect(id, color, width, markers = true) {
  const t = state.tracks.find((x) => x.id === id);
  if (!t) return;
  const W = eqCanvas.width, H = eqCanvas.height, ctx = eqCtx;
  const ds = trackDirs(t);
  let run = [];
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath();
  const path = [];
  ds.forEach((d, idx) => {
    const [x, y] = dirToPixel(d, W, H);
    path.push([x, y]);
  });
  run = [path[0]];
  const runs = [run];
  for (let i = 1; i < path.length; i++) {
    if (Math.abs(path[i][0] - path[i - 1][0]) > W / 2) {
      run = [path[i]]; runs.push(run);
    } else run.push(path[i]);
  }
  for (const r of runs) {
    ctx.beginPath(); ctx.moveTo(r[0][0], r[0][1]);
    for (const p of r) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
  }
  if (markers) {
    path.forEach((p, i) => {
      ctx.fillStyle = i === state.pairIndex ? "#fff" : color;
      ctx.beginPath(); ctx.arc(p[0], p[1], i === state.pairIndex ? 5 : 3, 0, TAU);
      ctx.fill();
    });
  }
}

// ----------------------------------------------------------- sphere 3D ----
function viewParams() {
  const W = spCanvas.width, H = spCanvas.height;
  const focal = (H / 2) / Math.tan(state.fov * D2R / 2);
  return { W, H, focal, cx: W / 2, cy: H / 2 };
}
function worldToView(d) {
  // inverse of world = Ry(yaw) Rx(pitch) v
  let v = rotY(d, -state.yaw);
  v = rotX(v, -state.pitch);
  return v;
}
function projectDir(d) {
  const { W, H, focal, cx, cy } = viewParams();
  const v = worldToView(d);
  if (v[2] <= 0.06) return null;
  return { x: cx + focal * v[0] / v[2], y: cy - focal * v[1] / v[2] };
}

async function refreshSourceAndSphere() {
  const url = currentFrameUrl();
  const im = await loadImage(url);
  if (sourceData?.key !== url) {
    off.width = im.naturalWidth; off.height = im.naturalHeight;
    offCtx.drawImage(im, 0, 0);
    sourceData = { key: url, data: offCtx.getImageData(0, 0, off.width, off.height).data,
                   W: off.width, H: off.height };
  }
  state.sphereDirty = true;
  drawEquirect();
}

function renderSphere() {
  state.sphereDirty = false;
  const ctx = spCtx, { W, H, focal, cx, cy } = viewParams();
  const img = ctx.createImageData(W, H);
  const buf = img.data;
  if (sourceData) {
    const sd = sourceData.data, SW = sourceData.W, SH = sourceData.H;
    for (let py = 0; py < H; py++) {
      const sy = (cy - (py + 0.5)) / focal;
      for (let px = 0; px < W; px++) {
        const sx = (px + 0.5 - cx) / focal;
        const l = Math.hypot(sx, sy, 1);
        let v = [sx / l, sy / l, 1 / l];
        let w = rotX(v, state.pitch);
        w = rotY(w, state.yaw);
        const [lon, lat] = dirToLonLat(w);
        let u = (lon + Math.PI) / TAU * SW - 0.5;
        let vv = (Math.PI / 2 - lat) / Math.PI * SH - 0.5;
        const x0 = Math.floor(u), y0 = Math.floor(vv);
        const fx = u - x0, fy = vv - y0;
        const xa = ((x0 % SW) + SW) % SW, xb = (xa + 1) % SW;
        const ya = Math.max(0, Math.min(SH - 1, y0));
        const yb = Math.max(0, Math.min(SH - 1, y0 + 1));
        const i00 = (ya * SW + xa) * 4, i01 = (ya * SW + xb) * 4;
        const i10 = (yb * SW + xa) * 4, i11 = (yb * SW + xb) * 4;
        const o = (py * W + px) * 4;
        for (let c = 0; c < 3; c++) {
          const top = sd[i00 + c] * (1 - fx) + sd[i01 + c] * fx;
          const bot = sd[i10 + c] * (1 - fx) + sd[i11 + c] * fx;
          buf[o + c] = top * (1 - fy) + bot * fy;
        }
        buf[o + 3] = 255;
      }
    }
  } else {
    for (let i = 3; i < buf.length; i += 4) buf[i] = 255;
  }
  ctx.putImageData(img, 0, 0);
  drawSphereOverlays();
}

function drawProjectedPolyline(dirs, color, width) {
  const ctx = spCtx;
  let run = [];
  const stroke = () => {
    if (run.length >= 2) {
      ctx.beginPath(); ctx.moveTo(run[0].x, run[0].y);
      for (const p of run) ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
  };
  ctx.strokeStyle = color; ctx.lineWidth = width;
  for (const d of dirs) {
    const p = projectDir(d);
    if (!p) { stroke(); run = []; continue; }
    run.push(p);
  }
  stroke();
}

function drawSphereOverlays() {
  const pd = state.pairData;
  const ctx = spCtx;
  if (state.show.seam) {
    const dirs = [];
    for (let a = -90; a <= 90; a += 4) dirs.push(lonLatToDir(Math.PI, a * D2R));
    drawProjectedPolyline(dirs, "#ff3cdc", 2.6);
  }
  if (state.show.polar) {
    for (const lat of [75, -75]) {
      const ring = [];
      for (let lo = -180; lo <= 180; lo += 6)
        ring.push(lonLatToDir(lo * D2R, lat * D2R));
      drawProjectedPolyline(ring, "rgba(200,180,255,.8)", 1.3);
    }
  }
  if (!pd) return;
  const n = pd.cls.length, cols = pd.cols;
  for (let k = 0; k < n; k++) {
    const c = pd.cls[k];
    if (c === 0 && !state.show.rejected) continue;
    const p = projectDir(nodeSrcDir(k, pd));
    if (!p) continue;
    ctx.fillStyle = c === 1 ? "rgba(120,216,255,.95)"
      : c === 2 ? "rgba(255,176,40,1)" : "rgba(130,142,170,.6)";
    ctx.beginPath(); ctx.arc(p.x, p.y, c ? 2.6 : 1.7, 0, TAU); ctx.fill();
  }
  for (let k = 0; k < n; k++) {
    const i = k % cols, j = (k / cols) | 0;
    if (i % 2 || j % 2) continue;
    const c = pd.cls[k];
    if (state.show.est && c > 0) drawSphereArrow(
      nodeSrcDir(k, pd), nodeTargetDir(k, pd), c === 2 ? "#ffb028" : "#9fe4ff");
    if (state.show.gt && pd.gt_lon) drawSphereArrow(
      nodeSrcDir(k, pd), nodeGtDir(k, pd), "rgba(125,255,176,.9)");
  }
  if (state.show.tracks) {
    for (const t of state.tracks) {
      if (state.selection?.kind === "track" && state.selection.id === t.id) continue;
      const rel = trackReliable(t);
      const col = !rel ? "rgba(150,160,185,.30)"
        : t.stress === "object" ? "rgba(255,176,40,.85)"
        : "rgba(90,220,200,.6)";
      drawProjectedPolyline(trackDirs(t), col, rel ? 1.6 : 1.0);
    }
  }
  if (state.selection?.kind === "track") {
    const t = state.tracks.find((x) => x.id === state.selection.id);
    if (t) {
      drawProjectedPolyline(trackDirs(t), "#ffe14d", 3.4);
      trackDirs(t).forEach((d, i) => {
        const p = projectDir(d); if (!p) return;
        ctx.fillStyle = i === state.pairIndex ? "#fff" : "#ffe14d";
        ctx.beginPath(); ctx.arc(p.x, p.y, i === state.pairIndex ? 5.5 : 3.4, 0, TAU);
        ctx.fill();
      });
    }
  }
  if (state.selection?.kind === "node") {
    const k = state.selection.k;
    drawSphereArrow(nodeSrcDir(k, pd), nodeTargetDir(k, pd), "#ffffff", 3);
    const p = projectDir(nodeSrcDir(k, pd));
    if (p) { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, TAU); ctx.stroke(); }
  }
  drawHotspotsSphere();
}

function drawSphereArrow(d0, d1, color, width = 1.8) {
  const dirs = [];
  for (let s = 0; s <= 8; s++) dirs.push(slerp(d0, d1, s / 8));
  drawProjectedPolyline(dirs, color, width);
  const p1 = projectDir(d1), p0 = projectDir(slerp(d0, d1, 0.82));
  if (p1 && p0) {
    const a = Math.atan2(p1.y - p0.y, p1.x - p0.x);
    spCtx.strokeStyle = color; spCtx.lineWidth = width;
    spCtx.beginPath(); spCtx.moveTo(p1.x, p1.y);
    spCtx.lineTo(p1.x - 8 * Math.cos(a - .42), p1.y - 8 * Math.sin(a - .42));
    spCtx.moveTo(p1.x, p1.y);
    spCtx.lineTo(p1.x - 8 * Math.cos(a + .42), p1.y - 8 * Math.sin(a + .42));
    spCtx.stroke();
  }
}

// ------------------------------------------------------------- panels ----
function fmt(v, d = 2, suffix = "") {
  return v === null || v === undefined ? "—" : `${Number(v).toFixed(d)}${suffix}`;
}
function metricCard(k, v, cls = "", small = "") {
  return `<div class="metric"><div class="k">${k}</div>
    <div class="v ${cls}">${v}${small ? `<small> ${small}</small>` : ""}</div></div>`;
}

function renderSummary() {
  const s = state.report.summary;
  const err = s.mean_angle_error_median_deg;
  const errCls = err == null ? "" : err < 0.3 ? "good" : err < 0.8 ? "warn" : "bad";
  const psnr = s.mean_reprojection_psnr;
  const psnrCls = psnr > 30 ? "good" : psnr > 24 ? "warn" : "bad";
  $("summaryGrid").innerHTML = [
    metricCard("可信方向对应占比", fmt(100 * s.mean_credible_fraction, 0), "", "%"),
    metricCard("角度误差中位（纯斑块）", fmt(err, 3), errCls, "°"),
    metricCard("最差帧对 p95 误差", fmt(s.worst_pair_p95_error_deg, 2), "", "°"),
    metricCard("重投影 PSNR", fmt(psnr, 1), psnrCls, "dB"),
    metricCard("GT 网格重投影 PSNR", fmt(s.mean_gt_reproj_psnr, 1), "", "dB"),
    metricCard("计算异常条目", s.total_anomalies,
      s.total_anomalies ? "warn" : "good", "条"),
  ].join("");

  const b = s.planar_baseline_mean;
  if (b) {
    $("baselineBox").innerHTML = `
      <div class="muted" style="margin:6px 0">朴素平面 UV 块匹配基线（全部匹配，未做球面几何）：</div>
      <table><tr><th></th><th>全图</th><th>中低纬</th><th>极区</th><th>接缝带</th></tr>
      <tr><td>平面法角度误差中位 °</td><td>${fmt(b.all_median, 2)}</td>
      <td>${fmt(b.mid_median, 2)}</td><td class="seam-bad">${fmt(b.polar_median, 2)}</td>
      <td class="seam-bad">${fmt(b.seam_median, 2)}</td></tr></table>
      <div class="muted" style="margin-top:5px">平面像素步长在极区不对应相同角距离，
      且无法表达跨极点的经度翻转；球面切平面法的当前帧对对比见下。</div>`;
  } else {
    $("baselineBox").innerHTML = "";
  }
}

function renderPairStats() {
  const pd = state.pairData;
  if (!pd) return;
  const rep = state.report.pairs[state.pairIndex];
  const es = pd.error_stats || {};
  const sphPolar = es.polar?.median;
  const sphSeam = es.seam?.median;
  const pl = rep.planar;
  $("pairStats").innerHTML = `
    <div class="kv">
      <span class="k">主旋转角/帧</span><span>${fmt(pd.global_rotation_deg, 3)}°</span>
      <span class="k">支持占比</span><span>${fmt(100 * pd.support_fraction, 1)}%</span>
      <span class="k">重投影 PSNR</span><span>${fmt(pd.reprojection.psnr, 2)} dB
        （MAE ${fmt(pd.reprojection.mae, 4)}）</span>
      ${pd.gt_reproj_psnr ? `<span class="k">GT 重投影 PSNR</span><span>${fmt(pd.gt_reproj_psnr, 2)} dB</span>` : ""}
      ${es.pure_credible ? `<span class="k">角度误差中位/p95</span>
        <span class="${es.pure_credible.median < 0.3 ? "seam-ok" : "seam-bad"}">
        ${fmt(es.pure_credible.median, 3)}° / ${fmt(es.pure_credible.p95, 3)}°</span>` : ""}
    </div>
    ${pl ? `<table style="margin-top:9px">
      <tr><th>当前帧对误差中位 °</th><th>球面切平面</th><th>朴素平面</th></tr>
      <tr><td>极区</td><td class="seam-ok">${fmt(sphPolar, 3)}</td>
        <td class="seam-bad">${fmt(pl.polar_median, 2)}</td></tr>
      <tr><td>接缝带</td><td class="seam-ok">${fmt(sphSeam, 3)}</td>
        <td class="seam-bad">${fmt(pl.seam_median, 2)}</td></tr>
    </table>` : ""}`;
}

function renderSeam() {
  const pd = state.pairData;
  if (!pd) return;
  const a = pd.seam_agg || {};
  const rows = pd.seam || [];
  const head = `<div class="kv" style="margin-bottom:7px">
    <span class="k">双侧可信接缝行</span><span>${a.rows_both_credible ?? 0} / ${pd.rows}</span>
    <span class="k">ω 差中位</span><span>${fmt(a.median_omega_diff_deg, 4)}°</span>
    <span class="k">最大相互预测偏差</span>
    <span class="${(a.max_cross_prediction_deg || 0) > 1.5 ? "seam-bad" : "seam-ok"}">
      ${fmt(a.max_cross_prediction_deg, 4)}°</span></div>`;
  const trs = rows.map((r) => `<tr class="${r.both_credible ? "" : "muted"}">
      <td>${r.lat_deg}°</td><td>${r.src_gap_deg}°</td>
      <td>${r.naive_lon_gap_deg}°</td>
      <td class="${r.cross_prediction_deg > 1.5 ? "seam-bad" : "seam-ok"}">
        ${r.cross_prediction_deg}°</td>
      <td>${r.both_credible ? "✓" : "—"}</td></tr>`).join("");
  $("seamBox").innerHTML = head + `<div class="seam-wrap"><table class="seam-table">
    <tr><th>纬度</th><th>真实弧间隔</th><th>朴素经度差</th><th>相互预测偏差</th><th>双侧可信</th></tr>
    ${trs}</table></div>`;
}

function renderAnomalies() {
  const list = state.report.anomalies || [];
  const box = $("anomalyBox");
  if (!list.length) {
    box.innerHTML = '<div class="anom none">未发现超过阈值的球面运动异常。</div>';
    return;
  }
  box.innerHTML = list.map((a, idx) => `
    <div class="anom ${a.level}" data-pair="${a.pair}">
      <div>${a.level === "warning" ? "⚠" : "ℹ"} ${a.message}</div>
      <div class="meta">帧对 ${a.pair} · ${a.type}</div>
    </div>`).join("");
  box.querySelectorAll(".anom[data-pair]").forEach((el) => {
    el.onclick = () => { pause(); loadPair(Number(el.dataset.pair)); };
  });
}

function renderTrackList() {
  const box = $("trackList");
  box.innerHTML = state.tracks.map((t) => `
    <div class="track-item ${trackReliable(t) ? "" : "unreliable"}" data-id="${t.id}">
      <span>${t.name} <span class="tag">[${t.stress}]</span>
        ${t.crosses_seam ? '<span class="badge seam">跨接缝</span>' : ""}
        ${t.crosses_pole ? '<span class="badge pole">跨极点</span>' : ""}</span>
      <span class="err">${t.median_error_deg != null ? t.median_error_deg + "°" : ""}</span>
    </div>`).join("");
  box.querySelectorAll(".track-item").forEach((el) => {
    el.onclick = () => {
      const id = Number(el.dataset.id);
      state.selection = { kind: "track", id };
      markActiveTrack();
      renderInspector(); drawEquirect(); state.sphereDirty = true;
    };
  });
}
function markActiveTrack() {
  document.querySelectorAll(".track-item").forEach((el) =>
    el.classList.toggle("active",
      state.selection?.kind === "track" && Number(el.dataset.id) === state.selection.id));
}

function renderInspector() {
  const box = $("inspector");
  const pd = state.pairData;
  if (state.selection?.kind === "track") return renderTrackInspector(state.selection.id);
  if (!pd || state.selection?.kind !== "node") {
    box.innerHTML = '<p class="muted">在球面或展开图上点击采样点或轨迹。</p>';
    return;
  }
  const k = state.selection.k;
  const d0 = nodeSrcDir(k, pd), d1 = nodeTargetDir(k, pd);
  const [lo0, la0] = dirToLonLat(d0).map((x) => x * R2D);
  const [lo1, la1] = dirToLonLat(d1).map((x) => x * R2D);
  const arc = angDist(d0, d1);
  const om = omegaBetween(d0, d1).map((x) => x * R2D);
  const err = pd.error_deg ? pd.error_deg[k] : null;
  const seamRow = pd.seam.find((r) => r.row === ((k / pd.cols) | 0));
  const onSeamCol = (k % pd.cols === 0 || k % pd.cols === pd.cols - 1);
  box.innerHTML = `
    <h3>采样点 #${k} ${classBadge(pd.cls[k])}</h3>
    <div class="kv">
      <span class="k">源方向 (lon,lat)</span><span>${lo0.toFixed(2)}°, ${la0.toFixed(2)}°</span>
      <span class="k">目标方向 (lon,lat)</span><span>${lo1.toFixed(2)}°, ${la1.toFixed(2)}°</span>
      <span class="k">球面位移（弧长）</span><span><b>${arc.toFixed(3)}°</b></span>
      <span class="k">世界系旋转向量</span>
      <span>(${om.map((x) => x.toFixed(3)).join(", ")})°</span>
      <span class="k">NCC / 纹理</span><span>${pd.ncc[k]} / ${pd.texture[k]}</span>
      ${err != null ? `<span class="k">对 GT 角度误差</span>
        <span class="${err < 0.5 ? "seam-ok" : "seam-bad"}">${err.toFixed(3)}°</span>` : ""}
      ${pd.pure ? `<span class="k">度量斑块纯度</span><span>${pd.pure[k] ? "单一图层" : "跨物体边界"}</span>` : ""}
      ${onSeamCol && seamRow ? `<span class="k">接缝行相互预测偏差</span>
        <span class="${seamRow.cross_prediction_deg > 1.5 ? "seam-bad" : "seam-ok"}">
        ${seamRow.cross_prediction_deg}°（真实弧间隔 ${seamRow.src_gap_deg}°，
        朴素经度差 ${seamRow.naive_lon_gap_deg}°）</span>` : ""}
    </div>`;
}

function renderTrackInspector(id) {
  const t = state.tracks.find((x) => x.id === id);
  markActiveTrack();
  const cur = state.pairIndex;
  const rows = t.est_lon.map((_, i) => {
    if (i === 0) return null;
    const seg = i - 1;
    return `<tr style="${seg === cur ? "background:#16314a" : ""}">
      <td>${seg}→${i}</td><td>${t.arc_deg[seg]}°</td>
      <td>${t.ncc[seg]}</td>
      <td>${t.segment_error_deg ? t.segment_error_deg[seg] + "°" : "—"}</td>
      <td>${t.accepted[seg] ? "✓" : t.inferred[seg] ? "推断" : "弱"}</td></tr>`;
  }).filter(Boolean).join("");
  $("inspector").innerHTML = `
    <h3>${t.name} <span class="tag">[${t.stress}]</span></h3>
    <div>
      ${t.crosses_seam ? '<span class="badge seam">跨越经度接缝（邻接）</span>' : ""}
      ${t.crosses_pole ? '<span class="badge pole">跨越极点（切换坐标基）</span>' : ""}
      ${t.median_error_deg != null ? `<span class="badge ok">误差中位 ${t.median_error_deg}°</span>` : ""}
    </div>
    <div class="kv" style="margin-top:8px">
      <span class="k">最大朴素经度跳变</span><span class="seam-bad">${t.max_lon_jump_deg ?? "—"}°</span>
      <span class="k">该处真实弧长步长</span><span class="seam-ok">${t.arc_at_jump_deg ?? "—"}°</span>
      <span class="k">解缠绕经度步长</span><span>${t.unwrapped_lon_step_deg ?? "—"}°</span>
      ${t.max_error_deg != null ? `<span class="k">最大角度误差</span><span>${t.max_error_deg}°</span>` : ""}
    </div>
    <div class="mini-seg"><table>
      <tr><th>帧段</th><th>弧长</th><th>NCC</th><th>GT误差</th><th>状态</th></tr>
      ${rows}</table></div>`;
}

function renderVideos() {
  const name = state.runName;
  const v = (file, cap) => `<div class="video-cell">
    <span class="cap">${cap}</span>
    <video controls preload="metadata" src="/artifacts/${name}/videos/${file}"></video>
    <a href="/artifacts/${name}/videos/${file}" target="_blank">在新标签打开 / 下载 ${file}</a>
  </div>`;
  let html = v("original.mp4", "原始合成全景序列")
    + v("estimated_spherical_motion.mp4", "估计球面方向运动")
    + v("reprojection_check.mp4", "重投影四联检查（A / 重投影 / B / 残差）");
  if (state.report.has_ground_truth) {
    html = v("ground_truth_motion.mp4", "真实球面运动 GT") + html;
  }
  html += `<div class="video-cell"><span class="cap">重投影检查 GIF 预览</span>
    <img src="/artifacts/${name}/videos/reprojection_check.gif" alt="reprojection gif"/>
    <a href="/artifacts/${name}/videos/reprojection_check.gif" target="_blank">打开 GIF</a></div>`;
  $("videoPanel").innerHTML = html;
}

// ----------------------------------------------------------- picking ------
function pickDirection(clientX, clientY, canvas) {
  const r = canvas.getBoundingClientRect();
  const x = (clientX - r.left) * canvas.width / r.width;
  const y = (clientY - r.top) * canvas.height / r.height;
  return { x, y };
}
function pickEquirect(ev) {
  const p = pickDirection(ev.clientX, ev.clientY, eqCanvas);
  const d = pixelToDir(p.x, p.y, eqCanvas.width, eqCanvas.height);
  return resolvePick(d);
}
function pickSphere(ev) {
  const p = pickDirection(ev.clientX, ev.clientY, spCanvas);
  const vp = viewParams();
  const sx = (p.x + 0.5 - vp.cx) / vp.focal;
  const sy = (vp.cy - (p.y + 0.5)) / vp.focal;
  const l = Math.hypot(sx, sy, 1);
  let v = [sx / l, sy / l, 1 / l];
  v = rotX(v, state.pitch); v = rotY(v, state.yaw);
  return resolvePick(normz(v));
}
function resolvePick(d) {
  state.lastPickDir = ST.normalize(d);
  const pd = state.pairData;
  // Grid spacing is ~360/cols degrees; accept up to the cell half-diagonal.
  let bestK = -1, bestAng = 4.2;
  if (pd) {
    for (let k = 0; k < pd.cls.length; k++) {
      const a = angDist(d, nodeSrcDir(k, pd));
      if (a < bestAng) { bestAng = a; bestK = k; }
    }
  }
  let bestT = -1, bestTA = 4.0, bestTAng = Infinity;
  for (const t of state.tracks) {
    for (let i = 0; i < t.est_lon.length; i++) {
      const td = lonLatToDir(t.est_lon[i] * D2R, t.est_lat[i] * D2R);
      const a = angDist(d, td);
      if (a < bestTA) { bestTA = a; bestT = t.id; bestTAng = a; }
    }
  }
  if (bestT >= 0 && (bestK < 0 || bestTAng <= bestAng + 0.8)) {
    state.selection = { kind: "track", id: bestT };
  } else if (bestK >= 0) {
    state.selection = { kind: "node", k: bestK };
  } else if (bestT >= 0) {
    state.selection = { kind: "track", id: bestT };
  }
  markActiveTrack();
  renderInspector();
  drawEquirect();
  state.sphereDirty = true;
}

// ------------------------------------------------------------ events ------
function bindEvents() {
  $("runSelect").onchange = (e) => selectRun(e.target.value).catch(console.error);
  document.querySelectorAll("#layerSwitch .seg").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll("#layerSwitch .seg").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      state.layer = b.dataset.layer;
      drawEquirect(); refreshSourceAndSphere();
    };
  });
  for (const key of ["Est", "Gt", "Tracks", "Seam", "Polar", "Rejected", "Hotspots"]) {
    const map = { Est: "est", Gt: "gt", Tracks: "tracks", Seam: "seam",
                  Polar: "polar", Rejected: "rejected", Hotspots: "hotspots" };
    $("tg" + key).onchange = (e) => {
      state.show[map[key]] = e.target.checked;
      drawEquirect(); state.sphereDirty = true;
    };
  }
  $("pairSlider").oninput = (e) => loadPair(Number(e.target.value));
  $("playBtn").onclick = togglePlay;
  $("speedSel").onchange = (e) => {
    state.intervalMs = Number(e.target.value);
    if (state.playing) { clearInterval(state.timer); startTimer(); }
  };
  $("rebuildBtn").onclick = startBuild;

  $("btnAddHotspot").onclick = addHotspotAtPick;
  $("btnClearHotspots").onclick = () => {
    state.userHotspots = [];
    renderHotspotPanel(); drawEquirect(); state.sphereDirty = true;
  };
  $("btnFollowHotspot").onclick = followFirstHotspot;
  $("btnStepMotion").onclick = stepSphericalMotion;
  $("btnStabilize").onclick = stabilizeHotspots;

  // equirect click
  eqCanvas.addEventListener("click", pickEquirect);

  // sphere drag rotate + click pick
  let down = null;
  spCanvas.addEventListener("pointerdown", (e) => {
    spCanvas.setPointerCapture(e.pointerId);
    down = { x: e.clientX, y: e.clientY, moved: false };
    spCanvas.classList.add("dragging");
  });
  spCanvas.addEventListener("pointermove", (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (Math.abs(dx) + Math.abs(dy) > 4) down.moved = true;
    if (down.moved) {
      state.yaw -= dx * 0.0045;
      state.pitch = Math.max(-1.45, Math.min(1.45, state.pitch - dy * 0.0045));
      down.x = e.clientX; down.y = e.clientY;
      state.sphereDirty = true;
    }
  });
  const endDrag = (e) => {
    spCanvas.classList.remove("dragging");
    if (down && !down.moved) pickSphere(e);
    down = null;
  };
  spCanvas.addEventListener("pointerup", endDrag);
  spCanvas.addEventListener("pointercancel", () => { down = null; });
  spCanvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    state.fov = Math.max(28, Math.min(100, state.fov * Math.exp(e.deltaY * 0.0011)));
    state.sphereDirty = true;
  }, { passive: false });
}

function pause() {
  state.playing = false;
  $("playBtn").textContent = "▶ 播放";
  if (state.timer) clearInterval(state.timer);
}
function startTimer() {
  state.timer = setInterval(() => {
    const max = state.report.pairs.length - 1;
    let next = state.pairIndex + 1;
    if (next > max) next = 0;
    loadPair(next);
  }, state.intervalMs);
}
function togglePlay() {
  if (state.playing) { pause(); return; }
  state.playing = true;
  $("playBtn").textContent = "⏸ 暂停";
  startTimer();
}

// ----------------------------------------------------- motion controls ----
function setMotionReadout(html) { $("motionReadout").innerHTML = html; }

function addHotspotAtPick() {
  if (!state.lastPickDir) {
    setMotionReadout("请先在球面或展开图上点击一个方向，再添加热点。");
    return;
  }
  const idx = state.userHotspots.length;
  state.userHotspots.push({
    name: `editor-hotspot-${idx + 1}`,
    dir: state.lastPickDir.slice(),
    radiusDeg: 5.0,
  });
  renderHotspotPanel();
  drawEquirect();
  state.sphereDirty = true;
  const [lon, lat] = dirToLonLat(state.lastPickDir).map((x) => x * R2D);
  setMotionReadout(`已添加热点，锚点方向 lon=${lon.toFixed(2)}° lat=${lat.toFixed(2)}°（球面方向，非像素）。`);
}

function yawPitchLookingAt(d) {
  // Mirror backend PerspectiveProjection.look_at for the player's camera:
  // rotate the view forward onto the target unit direction.
  const [x, y, z] = d;
  const pitch = Math.asin(Math.max(-1, Math.min(1, y)));
  const yaw = Math.atan2(x, z);
  return { yaw, pitch };
}

function followFirstHotspot() {
  const objs = allHotspotObjects();
  if (!objs.length) { setMotionReadout("当前没有可跟随的热点。"); return; }
  const h = objs[0];
  const { yaw, pitch } = yawPitchLookingAt(ST.normalize(h.dir));
  state.yaw = yaw; state.pitch = pitch;
  state.followHotspot = true;
  state.sphereDirty = true;
  const [lon, lat] = dirToLonLat(h.dir).map((x) => x * R2D);
  setMotionReadout(`视角已转动到热点「${h.name}」(lon=${lon.toFixed(2)}°, lat=${lat.toFixed(2)}°)。`);
}

async function callMotionApi(dir) {
  const i = state.pairIndex;
  const q = dir.map((x) => encodeURIComponent(x.toFixed(6))).join(",");
  return fjson(`/api/motion?name=${encodeURIComponent(state.runName)}&i=${i}&dir=${q}`);
}

async function stepSphericalMotion() {
  if (!state.lastPickDir) {
    setMotionReadout("请先点击一个方向，再推进该方向的球面运动。");
    return;
  }
  try {
    const res = await callMotionApi(state.lastPickDir);
    const moved = res.moved_dir[0];
    const arc = res.arc_deg[0];
    const [lo0, la0] = dirToLonLat(state.lastPickDir).map((x) => x * R2D);
    const [lo1, la1] = dirToLonLat(moved).map((x) => x * R2D);
    setMotionReadout(
      `帧对 ${res.pair} 真实 ω 场推进：(${lo0.toFixed(2)}°,${la0.toFixed(2)}°) → ` +
      `(${lo1.toFixed(2)}°,${la1.toFixed(2)}°)，球面位移 <b>${arc.toFixed(3)}°</b>，` +
      `主旋转向量 (${res.global_omega_deg.map((x) => x.toFixed(3)).join(", ")})°`);
    state.lastPickDir = ST.normalize(moved);
    state.sphereDirty = true;
  } catch (e) {
    setMotionReadout("运动接口调用失败：" + e.message);
  }
}

async function stabilizeHotspots() {
  const objs = allHotspotObjects().filter((h) => h.source === "pipeline");
  if (!objs.length) { setMotionReadout("没有带运动场链的管线热点可稳定。"); return; }
  // Ask the backend to advance the first pipeline hotspot through the real
  // field, proving frame-to-frame stability rather than showing fixed data.
  const h = objs[0];
  try {
    const i = state.pairIndex;
    const res = await fjson(
      `/api/motion?name=${encodeURIComponent(state.runName)}&i=${i}&hotspot=${h.id}`);
    const before = ST.normalize(h.dir);
    const after = ST.normalize(res.moved_dir[0]);
    const seam = Math.abs(ST.wrapLongitude(dirToLonLat(after)[0]
      - dirToLonLat(before)[0])) < Math.abs(
      dirToLonLat(after)[0] - dirToLonLat(before)[0]);
    setMotionReadout(
      `热点「${h.name}」由 ω 场推进，球面位移 ${res.arc_deg[0].toFixed(3)}°；` +
      `锚点始终是同一球面对象 id=${h.id}` +
      (seam ? "（含经度环绕，方向向量传递无接缝跳变）" : "") + "。");
    if (state.followHotspot) {
      const { yaw, pitch } = yawPitchLookingAt(after);
      state.yaw = yaw; state.pitch = pitch;
    }
    state.sphereDirty = true;
  } catch (e) {
    setMotionReadout("稳定接口调用失败：" + e.message);
  }
}

// ------------------------------------------------------------- build ------
async function startBuild() {
  const btn = $("rebuildBtn");
  btn.disabled = true;
  await fetch("/api/build", { method: "POST", body: "{}" });
  $("jobBar").classList.remove("hidden");
  pollJob();
}
let jobPoll = null;
async function pollJob() {
  if (jobPoll) clearTimeout(jobPoll);
  const tick = async () => {
    const j = await fjson("/api/job");
    $("jobMsg").textContent = j.message || j.stage;
    $("jobPct").textContent = Math.round((j.frac || 0) * 100) + "%";
    $("jobFill").style.width = ((j.frac || 0) * 100) + "%";
    if (j.running) { jobPoll = setTimeout(tick, 900); return; }
    $("rebuildBtn").disabled = false;
    if (j.error) { $("jobMsg").textContent = "构建失败：" + j.error; return; }
    $("jobMsg").textContent = "完成，正在载入…";
    try {
      await loadRuns();
      await selectRun("demo");
      setTimeout(() => $("jobBar").classList.add("hidden"), 1200);
    } catch (e) { console.error(e); }
  };
  tick();
}

// -------------------------------------------------------------- init ------
function loop() {
  if (state.sphereDirty && state.meta) renderSphere();
  requestAnimationFrame(loop);
}

window.addEventListener("DOMContentLoaded", async () => {
  bindEvents();
  requestAnimationFrame(loop);
  try {
    await loadRuns();
    if (state.runName) await selectRun(state.runName);
  } catch (e) {
    console.error(e);
    $("summaryGrid").innerHTML =
      '<div class="metric"><div class="k">尚未找到分析结果</div>'
      + '<div class="v warn" style="font-size:13px">点击右上角“重新合成并分析 Demo”，'
      + '或运行 python -m panomotion synth</div></div>';
    $("rebuildBtn").disabled = false;
  }
  // Resume job UI if a build is already running (server restart scenario).
  try {
    const j = await fjson("/api/job");
    if (j.running) { $("jobBar").classList.remove("hidden"); pollJob(); }
  } catch (e) { /* ignore */ }
});
