/* Shared spherical transform module.
 *
 * Both the perspective sphere view and the equirect outlet go through these
 * pure functions, mirroring panomotion/projection.py on the backend.  The
 * module holds no solver state: it only converts between unit directions,
 * lon/lat, pixels and view coordinates, and applies world-frame rotation
 * vectors.  Motion data always arrives as space direction vectors / omegas;
 * per-viewport tangent frames are never used to transport motion.
 */
"use strict";

const ST = (() => {
  const TAU = Math.PI * 2;
  const D2R = Math.PI / 180, R2D = 180 / Math.PI;

  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function cross(a, b) {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]];
  }
  function norm(a) { return Math.hypot(a[0], a[1], a[2]); }
  function normalize(a) {
    const n = norm(a) || 1;
    return [a[0] / n, a[1] / n, a[2] / n];
  }

  // --- direction / lonlat / pixel (same convention as sphere.py) ---------
  function lonLatToDir(lon, lat) {
    const c = Math.cos(lat);
    return [c * Math.cos(lon), Math.sin(lat), c * Math.sin(lon)];
  }
  function dirToLonLat(d) {
    return [Math.atan2(d[2], d[0]),
            Math.asin(Math.max(-1, Math.min(1, d[1])))];
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
  function wrapLongitude(lon) {
    let x = (lon + Math.PI) % TAU;
    if (x < 0) x += TAU;
    return x - Math.PI;
  }
  function geodesicAngle(a, b) {
    return Math.acos(Math.max(-1, Math.min(1, dot(a, b))));
  }

  // --- world-frame rotation vectors (motion transport) -------------------
  function rodrigues(omega, d) {
    const th = norm(omega);
    if (th < 1e-12) return normalize([d[0] + omega[0], d[1] + omega[1],
                                      d[2] + omega[2]]);
    const k = [omega[0] / th, omega[1] / th, omega[2] / th];
    const kd = dot(k, d), kxd = cross(k, d);
    const c = Math.cos(th), s = Math.sin(th);
    return normalize([
      d[0] * c + kxd[0] * s + k[0] * kd * (1 - c),
      d[1] * c + kxd[1] * s + k[1] * kd * (1 - c),
      d[2] * c + kxd[2] * s + k[2] * kd * (1 - c)]);
  }
  function omegaBetween(d0, d1) {
    const th = geodesicAngle(d0, d1);
    if (th < 1e-12) return [0, 0, 0];
    const ax = normalize(cross(d0, d1));
    return [ax[0] * th, ax[1] * th, ax[2] * th];
  }
  function slerp(d0, d1, t) {
    return rodrigues(omegaBetween(d0, d1).map((w) => w * t), d0);
  }

  // --- perspective camera frame (mirrors PerspectiveProjection) ----------
  function rotX(v, a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
  }
  function rotY(v, a) {
    const c = Math.cos(a), s = Math.sin(a);
    return [c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]];
  }
  function lookAtMatrix(centerDir, worldUp) {
    const f = normalize(centerDir);
    const up = worldUp || [0, 1, 0];
    let r = normalize(cross(up, f));
    if (norm(cross(up, f)) < 1e-9) r = normalize(cross([1, 0, 0], f));
    const u = cross(f, r);
    // Column-major 3x3: columns are right, up, forward (view -> world).
    return [r[0], u[0], f[0],
            r[1], u[1], f[1],
            r[2], u[2], f[2]];
  }
  function matVec(M, v) {
    return [M[0] * v[0] + M[3] * v[1] + M[6] * v[2],
            M[1] * v[0] + M[4] * v[1] + M[7] * v[2],
            M[2] * v[0] + M[5] * v[1] + M[8] * v[2]];
  }
  function matTransposeVec(M, v) {
    return [M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
            M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
            M[6] * v[0] + M[7] * v[1] + M[8] * v[2]];
  }

  // Yaw/pitch camera used by the interactive sphere view (player convention:
  // world = Rx(pitch) Ry(yaw) view).
  function yawPitchView(d, yaw, pitch) {
    return rotX(rotY(d, -yaw), -pitch);
  }
  function yawPitchWorld(v, yaw, pitch) {
    return rotY(rotX(v, pitch), yaw);
  }

  // --- polyline splitting with a preserved object id ---------------------
  // Returns 2-D fragment arrays; every fragment belongs to objectId.
  function splitProjected(points, isBreak, objectId) {
    const fragments = [];
    let run = [];
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p === null || (run.length && isBreak(run[run.length - 1], p))) {
        if (run.length >= 2) fragments.push({ objectId, points: run });
        run = p === null ? [] : [p];
      } else {
        run.push(p);
      }
    }
    if (run.length >= 2) fragments.push({ objectId, points: run });
    return fragments;
  }
  function equirectBreak(W) {
    return (a, b) => Math.abs(b.x - a.x) > W / 2;
  }

  return {
    D2R, R2D, TAU, dot, cross, norm, normalize,
    lonLatToDir, dirToLonLat, pixelToDir, dirToPixel, wrapLongitude,
    geodesicAngle, rodrigues, omegaBetween, slerp,
    rotX, rotY, lookAtMatrix, matVec, matTransposeVec,
    yawPitchView, yawPitchWorld, splitProjected, equirectBreak,
  };
})();
