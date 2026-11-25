/* VIEW, part 1: the seven pictures. Each is generated from rules in the style of one instrument and baked off screen;
   none is data. Per frame the page only places the baked pictures and draws what moves. */
const M = window.HFD, $ = id => document.getElementById(id);
const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
const cv = $('view'), ctx = cv.getContext('2d');
const clamp = M.clamp, lerp = (a, b, t) => a + (b - a) * t, TAU = Math.PI * 2;
const SMALL = Math.min(window.innerWidth || 1280, (window.screen && screen.width) || 1280) < 700;
const BK = SMALL ? 700 : 1100, BKW = SMALL ? 900 : 1600;   /* side of each baked picture, in pixels; the one a waypoint rests on is finer */
const ASP = SMALL ? 1.42 : 1;                  /* the stage is taller than wide on a phone */
const U = [1e-3, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-10];   /* metres per world unit at each level: mm, µm, µm, µm, nm, nm, Å */
const NL = M.LEVELS.length;
const mk = (w, h) => { const c = document.createElement('canvas'); c.width = Math.max(1, Math.round(w)); c.height = Math.max(1, Math.round(h)); return c; };
const hash2 = (ix, iy, s) => { let n = Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(s, 1274126177); n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967296; };
const vnoise = (x, y, s) => { const ix = Math.floor(x), iy = Math.floor(y), fx = x - ix, fy = y - iy, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), a = hash2(ix, iy, s), b = hash2(ix + 1, iy, s), c = hash2(ix, iy + 1, s), d = hash2(ix + 1, iy + 1, s); return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v; };
const rgba = (c, a) => 'rgba(' + c[0] + ',' + c[1] + ',' + c[2] + ',' + a + ')';
const COL = { blue: [95, 176, 255], gold: [231, 199, 90], red: [255, 106, 74], green: [127, 214, 164], violet: [183, 156, 255], pale: [176, 208, 222] };
const angDiff = (to, a) => Math.atan2(Math.sin(to - a), Math.cos(to - a));
/* a smooth curve through points (Catmull and Rom), as a list of points */
const spline = (P, closed, n = 6) => { const out = [], N = P.length, at = i => closed ? P[(i + N) % N] : P[clamp(i, 0, N - 1)];
  for (let i = 0; i < (closed ? N : N - 1); i++) { const a = at(i - 1), b = at(i), c = at(i + 1), d = at(i + 2); for (let k = 0; k < n; k++) { const t = k / n, t2 = t * t, t3 = t2 * t;
      out.push([0.5 * (2 * b[0] + (c[0] - a[0]) * t + (2 * a[0] - 5 * b[0] + 4 * c[0] - d[0]) * t2 + (3 * b[0] - a[0] - 3 * c[0] + d[0]) * t3), 0.5 * (2 * b[1] + (c[1] - a[1]) * t + (2 * a[1] - 5 * b[1] + 4 * c[1] - d[1]) * t2 + (3 * b[1] - a[1] - 3 * c[1] + d[1]) * t3)]); } }
  if (!closed) out.push(P[N - 1]); return out; };
const trace = (g, pts, closed) => { g.beginPath(); g.moveTo(pts[0][0], pts[0][1]); for (let i = 1; i < pts.length; i++) g.lineTo(pts[i][0], pts[i][1]); if (closed) g.closePath(); };
/* blurring. A true Gaussian where the browser has one; otherwise by shrinking and enlarging. */
const HASF = (() => { try { const g = mk(2, 2).getContext('2d'); return typeof g.filter === 'string'; } catch (e) { return false; } })();
function soft(src, rad) { if (rad < 0.35) return src; const w = src.width, h = src.height, t = mk(w, h), g = t.getContext('2d');
  if (HASF) { g.filter = 'blur(' + rad.toFixed(2) + 'px)'; g.drawImage(src, 0, 0); g.filter = 'none'; return t; }
  const k = Math.max(1.5, rad * 1.6), s = mk(w / k, h / k), q = s.getContext('2d'); q.imageSmoothingQuality = 'high'; q.drawImage(src, 0, 0, s.width, s.height); const s2 = mk(s.width / 2, s.height / 2); s2.getContext('2d').drawImage(s, 0, 0, s2.width, s2.height); g.imageSmoothingQuality = 'high'; g.globalAlpha = 0.5; g.drawImage(s, 0, 0, w, h); g.drawImage(s2, 0, 0, w, h); g.globalAlpha = 1; g.globalCompositeOperation = 'destination-over'; g.drawImage(s, 0, 0, w, h); return t; }
/* box blur of a float image, in place; three passes come close to a Gaussian */
function boxBlur(a, w, h, r, passes = 2) { const t = new Float32Array(Math.max(w, h)); r = Math.max(1, Math.round(r));
  for (let p = 0; p < passes; p++) { for (let j = 0; j < h; j++) { const o = j * w; let s = 0; for (let i = -r; i <= r; i++) s += a[o + clamp(i, 0, w - 1)]; for (let i = 0; i < w; i++) { t[i] = s / (2 * r + 1); s += a[o + Math.min(w - 1, i + r + 1)] - a[o + Math.max(0, i - r)]; } for (let i = 0; i < w; i++) a[o + i] = t[i]; }
    for (let i = 0; i < w; i++) { let s = 0; for (let j = -r; j <= r; j++) s += a[clamp(j, 0, h - 1) * w + i]; for (let j = 0; j < h; j++) { t[j] = s / (2 * r + 1); s += a[Math.min(h - 1, j + r + 1) * w + i] - a[Math.max(0, j - r) * w + i]; } for (let j = 0; j < h; j++) a[j * w + i] = t[j]; } } return a; }

/* ════ Level 1: a T1-weighted MRI slice, in millimetres. The plane is coronal (ear to ear), through the thalamus, the
   pons and the front of the cerebellum. The outline, the ventricles, the deep nuclei and the named sulci are placed by
   hand from standard anatomy; smaller folds are grown by rule. The cortex is every point of the cerebrum within one
   cortical thickness of its surface. The picture is made at a quarter of a millimetre and then sampled into voxels. ════ */
const MRI = (() => {
  const G = SMALL ? 660 : 880, MMW = 220, q = MMW / G, YO = 12, SC = 0.94, N = G * G, r = M.rng(31), rs = () => r() * 2 - 1, AX = 10.5, TH = M.SET.cortexMm;
  const mirror = H => H.concat(H.slice().reverse().filter(p => p[0] !== 0).map(p => [-p[0], p[1]]));
  const layer = draw => { const c = mk(G, G), g = c.getContext('2d'); g.setTransform(SC / q, 0, 0, SC / q, G / 2, G / 2 + YO / q); g.lineCap = 'round'; g.lineJoin = 'round'; g.fillStyle = g.strokeStyle = '#fff'; draw(g); const d = g.getImageData(0, 0, G, G).data, o = new Uint8Array(N); for (let k = 0; k < N; k++) o[k] = d[4 * k + 3] > 127 ? 1 : 0; return o; };
  const blob = (g, P, lw) => { trace(g, spline(P, true), true); g.fill(); if (lw) { g.lineWidth = lw; g.stroke(); } };
  const both = (g, P, lw) => { blob(g, P, lw); blob(g, P.map(p => [-p[0], p[1]]), lw); };
  /* a sulcus: a wandering cut that is wide at its mouth and narrow at its floor, and may fork there */
  const sulc = (g, x, y, tx, ty, w0, w1, wig, fork, back = 2.6) => { const a0 = Math.atan2(ty - y, tx - x); let a = a0 + rs() * 0.12, L = Math.hypot(tx - x, ty - y) + back, b = 0; x -= Math.cos(a) * back; y -= Math.sin(a) * back;
    for (let l = 0; l < L; l += 0.7) { const m = l - back, w = m < 3.5 && back ? lerp(w0 * 2.4, w0, clamp(m / 3, 0, 1)) : lerp(w0, w1, l / L); b = b * 0.82 + rs() * wig; a += b + 0.07 * angDiff(a0, a); const nx = x + Math.cos(a) * 0.7, ny = y + Math.sin(a) * 0.7; g.lineWidth = w; g.beginPath(); g.moveTo(x, y); g.lineTo(nx, ny); g.stroke(); x = nx; y = ny; }
    if (fork) for (const s of [-1, 1]) if (r() < fork) { const fa = a + s * (0.8 + 0.5 * r()), fl = L * (0.22 + 0.2 * r()); sulc(g, x, y, x + Math.cos(fa) * fl, y + Math.sin(fa) * fl, w1, w1 * 0.9, wig, 0, 0); } };
  const CER = mirror([[0, -83.5], [10, -86], [24, -85.5], [40, -80], [54, -68], [63.5, -50], [68, -28], [67.5, -8], [66.5, 7], [65, 19], [61, 28], [53, 34.5], [40, 31], [27, 24.5], [17.5, 19], [13, 12], [9, 5], [0, 6]]);
  const STEM = mirror([[0, 1], [9, 2], [14.5, 8], [13, 14], [16.5, 20], [18.5, 30], [16, 40], [10.5, 46.5], [8.5, 58], [7, 70], [6, 90], [6, 118], [0, 118]]);
  const CBL = [[15, 25.5], [30, 31.5], [45, 38.5], [56, 45.5], [58.5, 55], [53, 65], [41, 71], [27, 71.5], [17, 66], [13, 56], [13.5, 44], [14, 33]];
  /* named sulci of one side: mouth, floor, width at the mouth and at the floor, wander, chance of a fork */
  const NAMED = [[24.5, -86, 20, -62, 1.0, 0.45, 0.07, 0.8], [43, -79, 33, -58, 1.0, 0.45, 0.08, 0.7], [58.5, -61, 42, -49, 1.0, 0.45, 0.08, 0.7], [66.5, -38, 49, -33, 0.95, 0.45, 0.08, 0.6], [67.5, -17, 53, -19, 0.9, 0.45, 0.08, 0.5],
    [66, 20, 47, 15, 1.0, 0.45, 0.07, 0.7], [57, 32, 48, 22, 0.9, 0.45, 0.08, 0.4], [41, 31.5, 38, 18, 0.9, 0.45, 0.07, 0.5], [26, 24, 30, 14, 0.85, 0.5, 0.07, 0.3]];
  const brain = layer(g => { blob(g, CER); g.globalCompositeOperation = 'destination-out';
    sulc(g, 0, -88, 0, -39.6, 1.25, 1.0, 0.006, 0, 0);                                               /* the fissure between the hemispheres */
    for (const s of [-1, 1]) {
      sulc(g, s * 71, 3 + rs(), s * 43, -6, 2.5, 1.2, 0.03, 0, 0); sulc(g, s * 43.4, -5, s * 40.5, -25, 1.2, 0.6, 0.04, 0, 0); sulc(g, s * 43.4, -6, s * 44, 9, 1.2, 0.6, 0.04, 0, 0); sulc(g, s * 56, -1.5, s * 55, -11, 0.8, 0.45, 0.06, 0, 0); sulc(g, s * 55, 0.5, s * 56, 9, 0.8, 0.45, 0.06, 0, 0);     /* Sylvian fissure and the sulcus round the insula */
      for (const S of NAMED) sulc(g, s * (S[0] + rs()), S[1] + rs(), s * (S[2] + 2 * rs()), S[3] + 2 * rs(), S[4], S[5], S[6], S[7]);
      sulc(g, s * 0.6, -39.2, s * 8.5, -40.2, 0.9, 0.6, 0.02, 0, 0);                                    /* callosal sulcus */
      sulc(g, s * 0.4, -51 + 2 * rs(), s * 14, -53, 0.9, 0.45, 0.06, 0.6, 0); sulc(g, s * 0.4, -63 + 2 * rs(), s * 10, -62, 0.8, 0.45, 0.07, 0.4, 0); sulc(g, s * 0.4, -74 + 2 * rs(), s * 6.5, -72, 0.8, 0.45, 0.07, 0, 0);      /* cingulate and medial sulci */
      sulc(g, s * 7.5, 15.5, s * 17, 7.5, 1.2, 0.9, 0.02, 0, 0); sulc(g, s * 17, 7.5, s * 27, 8.5, 0.8, 0.5, 0.03, 0, 0);       /* the cleft between the thalamus and the hippocampus */
      /* smaller folds between the named ones */
      const out = spline(CER.slice(0, 16), false, 10); let run = 4 + 5 * r();
      for (let i = 1; i < out.length - 1; i++) { const p = out[i], d = Math.hypot(p[0] - out[i - 1][0], p[1] - out[i - 1][1]); run -= d; if (run > 0) continue; const x = s * p[0], y = p[1];
        if (Math.abs(x - AX) < 9 && y < -70) { run = 2; continue; } let near = Math.abs(p[1] - 3) < 7 && p[0] > 55; for (const S of NAMED) if (Math.hypot(p[0] - S[0], p[1] - S[1]) < 7.5) near = true; if (near) { run = 2; continue; }
        const nx = -(out[i + 1][1] - out[i - 1][1]), ny = out[i + 1][0] - out[i - 1][0], nl = Math.hypot(nx, ny), len = 5 + 12 * r() * r(), ja = rs() * 0.5, ca = Math.cos(ja), sa = Math.sin(ja), ux = (nx * ca - ny * sa) / nl, uy = (nx * sa + ny * ca) / nl;
        sulc(g, x, y, x + s * ux * len, y + uy * len, 0.8 + 0.3 * r(), 0.42, 0.1, 0.35); run = 8 + 7 * r(); } } });
  const stem = layer(g => { blob(g, STEM); both(g, [[10, 2], [17, -2], [21, 6], [15, 11]]); });
  const cbl = layer(g => both(g, CBL));
  const wmForce = layer(g => { g.lineWidth = 5.2; trace(g, spline([[-27, -42.5], [-14, -38.6], [0, -36.8], [14, -38.6], [27, -42.5]], false, 8)); g.stroke(); blob(g, [[-3, -27.5], [3, -27.5], [2.4, -24.5], [-2.4, -24.5]]); both(g, [[4, -3], [14, -5], [19, 4], [14, 13], [6, 12]]); });
  const vent = layer(g => { both(g, [[1.1, -34], [8, -36.4], [18, -40.4], [23, -40], [18.5, -32.4], [9.5, -27.8], [1.3, -26.6]]); g.beginPath(); g.ellipse(0, -12.5, 1.25, 9.8, 0, 0, TAU); g.fill(); for (const s of [-1, 1]) { g.beginPath(); g.ellipse(s * 29.5, 9.6, 3.6, 1.15, s * 0.25, 0, TAU); g.fill(); } });
  const thal = layer(g => both(g, [[2.2, -23], [10, -25.5], [17, -20.5], [18.5, -11], [13.5, -4], [4.5, -3], [2.4, -12]]));
  const caud = layer(g => { for (const s of [-1, 1]) { g.beginPath(); g.ellipse(s * 23.6, -33.6, 3.2, 4.4, s * 0.5, 0, TAU); g.fill(); } });
  const put = layer(g => both(g, [[27, -22], [31.5, -13], [32.5, -2], [29.5, 5.5], [25.5, 1.5], [24, -10]]));
  const hip = layer(g => both(g, [[15.5, 13.5], [21, 10.6], [27.5, 11.4], [29, 15.6], [23, 18.6], [16.5, 17.6]]));
  const cranial = layer(g => { blob(g, CER, 5.4); blob(g, STEM.filter(p => p[1] < 70), 7); both(g, CBL, 5); trace(g, [[-22, 0], [22, 0], [22, 70], [-22, 70]], true); g.fill(); });
  const head = layer(g => { blob(g, mirror([[0, 40], [60, 40], [80, 46], [80.5, 62], [73, 84], [65, 104], [63, 130], [0, 130]])); g.lineWidth = 3.4; for (const s of [-1, 1]) { trace(g, spline([[s * 78, 26], [s * 87, 30], [s * 90.5, 42], [s * 87, 54], [s * 80, 60]], false, 8)); g.stroke(); } });
  /* distances, in pixels, to the nearest point outside (or inside) a mask */
  const dt = (mask, want) => { const d = new Float32Array(N); for (let k = 0; k < N; k++) d[k] = mask[k] === want ? 1e4 : 0;
    const pass = (i0, i1, di, j0, j1, dj) => { for (let j = j0; j !== j1; j += dj) for (let i = i0; i !== i1; i += di) { const k = j * G + i; let v = d[k]; if (!v) continue; v = Math.min(v, d[k - di] + 1, d[k - dj * G] + 1, d[k - dj * G - di] + 1.414, d[k - dj * G + di] + 1.414); d[k] = v; } };
    pass(1, G - 1, 1, 1, G - 1, 1); pass(G - 2, 0, -1, G - 2, 0, -1); return d; };
  const tissue = new Uint8Array(N); for (let k = 0; k < N; k++) tissue[k] = brain[k] || stem[k] || cbl[k] || wmForce[k] ? 1 : 0;
  const dIn = dt(tissue, 1), dOut = dt(cranial, 0), dCb = dt(cbl, 1), softM = (m, rad) => boxBlur(Float32Array.from(m), G, G, rad, 2), thalS = softM(thal, 5), caudS = softM(caud, 3), putS = softM(put, 4), hipS = softM(hip, 3), stemS = softM(stem, 3);
  /* the top and the bottom of the cerebellum above each column, so that its leaves can be laid between them */
  const cbT = new Float32Array(G), cbB = new Float32Array(G); for (let i = 0; i < G; i++) { let a = -1, b = -1; for (let j = 0; j < G; j++) if (cbl[j * G + i]) { if (a < 0) a = j; b = j; } cbT[i] = a; cbB[i] = b; }
  for (const A of [cbT, cbB]) { const c = A.slice(); for (let i = 0; i < G; i++) { let s = 0, m = 0; for (let k = -9; k <= 9; k++) { const v = c[clamp(i + k, 0, G - 1)]; if (v >= 0) { s += v; m++; } } A[i] = m ? s / m : -1; } }
  /* the skin: the skull's shell and scalp over the vault, and a simple neck and ears below */
  const skinMask = new Uint8Array(N); for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const k = j * G + i, x = ((i + 0.5) * q - MMW / 2) / SC, y = ((j + 0.5) * q - MMW / 2 - YO) / SC, lat = M.smooth((Math.abs(x) - 46) / 14) * M.smooth((y + 34) / 22); skinMask[k] = dOut[k] * q < 12.6 + 5.5 * lat - 1.2 * M.smooth((y - 30) / 30) || head[k] ? 1 : 0; }
  const dSkin = dt(skinMask, 1);
  const Ic = new Float32Array(N), lab = new Uint8Array(N);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const k = j * G + i, x = ((i + 0.5) * q - MMW / 2) / SC, y = ((j + 0.5) * q - MMW / 2 - YO) / SC, ax = Math.abs(x), n1 = vnoise(x * 0.11, y * 0.11, 5), n2 = vnoise(x * 0.42, y * 0.42, 6), n3 = vnoise(x * 1.3, y * 1.3, 7); let v;
    if (!skinMask[k]) v = 1.5;
    else if (!cranial[k]) { const d = dOut[k] * q, lat = M.smooth((ax - 46) / 14) * M.smooth((y + 34) / 22), t = 7.2 - 3.9 * lat - 1.6 * M.smooth((y - 44) / 14), ds = dSkin[k] * q;
      if (ds < 1.0) v = 118;
      else if (d < t) v = d < 1.25 || d > t - 1.25 || t < 4.4 ? 9 + 5 * n3 : 150 + 75 * vnoise(x * 0.5, y * 0.5, 8) - 45 * n3;                 /* the skull: two dark tables with bright marrow between */
      else if (y < 36) v = d < t + 0.7 ? 70 : lat > 0.02 && d < t + 0.7 + 8.5 * lat * M.smooth((48 - y) / 30) ? 86 + 14 * n2 - 26 * (vnoise(x * 0.2, y * 1.1, 9) > 0.72 ? 1 : 0) : 226 - 20 * n2;        /* temporal muscle, then scalp fat */
      else { const mast = Math.hypot((ax - 58) / 10.5, (y - 55) / 12), cond = Math.hypot((ax - 17) / 9.5, (y - 88) / 8.5), fatty = Math.abs(vnoise(x * 0.07, y * 0.035, 10) - 0.5) < 0.045 || vnoise(x * 0.09 + 3, y * 0.05, 11) > 0.69;
        if (mast < 1) v = vnoise(x * 0.75, y * 0.75, 12) > 0.42 ? 3 : 46;                                                                    /* air cells of the mastoid */
        else if (d < 4.6 && y < 86) v = 10 + 26 * n2;
        else if (cond < 1) v = cond > 0.82 ? 14 : 150 + 50 * n2;
        else if (ax < 9.5 && y > 60) v = 20 + 6 * n2;
        else if (ds < 5.5 + 3 * n1) v = 228 - 22 * n2;
        else v = fatty ? 214 - 24 * n3 : 92 + 20 * n1 - 12 * n3; }
      if (stem[k] && y > 60) v = 172 + 5 * n2; }
    else if (!tissue[k]) { v = 17 + 7 * n2;                                                                  /* fluid, with the folds of dura and the vessels that lie in it */
      const top = -89.6; if (ax < 0.45 && y < -39) v = 58; if (y < top + 7.5 && ax < 5 * (1 - (y - top) / 7.5)) v = 66 + 30 * n2;
      const yt = 22.6 + (ax - 13) * 0.447; if (ax > 12 && ax < 60 && Math.abs(y - yt) < 0.55) v = 60;
      const ci = Math.floor(x / 3.2), cj = Math.floor(y / 3.2); if (hash2(ci, cj, 13) < 0.16 && Math.hypot(x - (ci + 0.3 + 0.4 * hash2(ci, cj, 14)) * 3.2, y - (cj + 0.3 + 0.4 * hash2(ci, cj, 15)) * 3.2) < 0.55) v = 185;
      for (const p of [[50, -0.5], [46.5, -4], [44.5, 3.5], [44, -14]]) if (Math.hypot(ax - p[0], y - p[1]) < 0.95) v = 205;
      if (Math.hypot(ax - 8.5, y + 29.8) < 1.5) v = 104; }
    else if (cbl[k] && !stem[k]) { lab[k] = 2; const dc = dCb[k] * q, yt = cbT[i], yb = cbB[i], half = Math.max(1, (yb - yt) / 2 * q), u = clamp((ax - 13) / 45, 0, 1), f = ((j - (yt + yb) / 2) * q) / half + 0.2 * (vnoise(x * 0.12, y * 0.12, 16) - 0.5) + 0.07 * (vnoise(x * 0.4, y * 0.4, 17) - 0.5);
      /* leaves stacked between the upper and lower surfaces, opening on to both; thin leaves merge towards the side */
      const NL0 = 34, gg = f * (1 + 0.42 * u), p = (gg + 1.42) / 2.84 * NL0, leaf = 2.84 / NL0 * half / (1 + 0.42 * u), m = Math.round(p), lv = m % 4 === 0 ? 0 : m % 2 === 0 ? 1 : 2, fis = Math.abs(p - m) * leaf, pm = ((p % 4) + 4) % 4;
      const there = leaf * (lv === 2 ? 1 : lv === 1 ? 2 : 4) > 1.25 && dc < [21, 12.5, 6.5][lv] + 5 * (n1 - 0.5), lam1 = Math.abs(pm - 2) * leaf, lam2 = Math.min(Math.abs(pm - 1), Math.abs(pm - 3)) * leaf, mcp = ax < 25 && Math.abs(y - 31 - (ax - 14) * 1.25) < 6.5 - 0.2 * (ax - 14);
      if (dc > 13.5 + 4 * (n1 - 0.5) || mcp) v = Math.abs(Math.hypot((ax - 22) / 6, (y - 49) / 3.6) - 1) < 0.13 + 0.09 * Math.sin(Math.atan2(y - 49, ax - 22) * 11) ? 146 : 182 - 6 * n2;                    /* the white core, entering from the pons, with the dentate nucleus in it */
      else if (there && fis < 0.17 + 0.16 * M.smooth((3 - dc) / 3)) v = 26;
      else if (lam1 < 1.5 * M.smooth((dc - 3.5) / 9) || (leaf > 0.95 && lam2 < 0.6 * M.smooth((dc - 2.5) / 8))) v = 178 - 8 * n2;
      else v = 106 + 9 * n2; }
    else if (stem[k] && !(brain[k] && y < 3)) { lab[k] = 3; v = y < 15 ? 154 + 10 * n2 : y < 45 ? 142 - 12 * Math.sin(y * 1.7 + 6 * n1 + 0.2 * ax) * M.smooth((16 - ax) / 6) + 12 * (n1 - 0.5) + 8 * n2 + 16 * M.smooth(1.3 - Math.hypot((ax - 5.5) / 3.6, (y - 30) / 10)) - (ax < 0.6 ? 8 : 0) - 14 * M.smooth(1.4 - Math.hypot(ax / 7, (y - 20) / 5)) : 156 + 8 * (n1 - 0.5) + 5 * n2 - 14 * M.smooth(1.3 - Math.hypot(ax / 3, (y - 54) / 7));
      if (y < 15) { const u = (ax - 8) * 0.88 + (y - 9.4) * 0.47, w = -(ax - 8) * 0.47 + (y - 9.4) * 0.88; if (Math.hypot(u / 5.2, w / 1.7) < 1) v = 134; if (Math.hypot(ax - 4.4, y - 5.4) < 2.7) v = 146; }
      if (ax > 12.5 && y > 21 && y < 45) v = lerp(v, 182, M.smooth((ax - 12.5) / 3)); }
    else { lab[k] = 1; const d = dIn[k] * q, deep = d > TH * (0.96 + 0.16 * (n1 - 0.5));
      if (vent[k] && d > 1.5) v = 18 + 6 * n2;
      else if (wmForce[k]) v = 190 - 5 * n2;
      else { v = deep ? 184 + 9 * (n1 - 0.5) + 5 * (n2 - 0.5) : 114 + 9 * (n2 - 0.5) + 5 * (n1 - 0.5); if (!deep && d > TH * 0.86) v = lerp(v, 184, 0.35);
        if (deep) { v = lerp(v, 163 + 14 * (n1 - 0.5) - 10 * M.smooth((6 - ax) / 4) + 6 * (n2 - 0.5), M.smooth(thalS[k] * 1.1)); v = lerp(v, 134 + 6 * n2, M.smooth(caudS[k] * 1.2)); v = lerp(v, 146 + 10 * (n2 - 0.5) + 10 * (vnoise(x * 0.25, y * 0.25, 26) - 0.5) - 8 * M.smooth((ax - 30) / 2), M.smooth(putS[k] * 1.1) * (0.75 + 0.25 * n1)); v = lerp(v, 124 + 6 * n2, M.smooth(hipS[k] * 1.2));
          if (ax > 33.7 && ax < 34.5 && y > -22 && y < 3) v -= 22; if (ax > 19 && ax < 23.5 && Math.abs(y + 8) < 14) v += 4; } } }
    Ic[k] = v; }
  /* a smooth, slightly lopsided warp, so that the two sides are not mirror images; then the blur of the scanner */
  const I0 = new Float32Array(N), labW = new Uint8Array(N);
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) { const x = i * q, y = j * q, fx = clamp(i + (5.2 * (vnoise(x / 34, y / 34, 22) - 0.5) + 1.4 * (vnoise(x / 13, y / 13, 24) - 0.5)) / q, 0, G - 1.001), fy = clamp(j + (5.2 * (vnoise(x / 34, y / 34, 23) - 0.5) + 1.4 * (vnoise(x / 13, y / 13, 25) - 0.5)) / q, 0, G - 1.001), a = Math.floor(fx), b = Math.floor(fy), u = fx - a, w = fy - b, k = b * G + a;
      I0[j * G + i] = Ic[k] * (1 - u) * (1 - w) + Ic[k + 1] * u * (1 - w) + Ic[k + G] * (1 - u) * w + Ic[k + G + 1] * u * w; labW[j * G + i] = lab[Math.round(fy) * G + Math.round(fx)]; }
  boxBlur(I0, G, G, 1, 2);
  let ay = -95; const ai = Math.round((AX + MMW / 2) / q); while (ay < 0 && labW[Math.round((ay + MMW / 2) / q) * G + ai] !== 1) ay += q;
  const S = { G, MMW, anchor: [AX, ay + 0.65], vox: 1, canvas: null, fine: null };
  /* average the fine picture into voxels of side v mm; thermal noise grows as the voxel shrinks (less signal in each one).
     The noise is Rician: the scanner reports the size of a complex number. A viewer then interpolates between voxels. */
  S.voxelise = v => { const n = Math.round(MMW / v), bs = G / n, cc = mk(n, n), gg = cc.getContext('2d'), im = gg.createImageData(n, n), rr = M.rng(77), sd = Math.min(15, 3.4 / v), V = new Float32Array(n * n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { let s = 0, m = 0; const j0 = Math.floor(j * bs), j1 = Math.max(j0 + 1, Math.floor((j + 1) * bs)), i0 = Math.floor(i * bs), i1 = Math.max(i0 + 1, Math.floor((i + 1) * bs));
        for (let b = j0; b < j1; b++) for (let a = i0; a < i1; a++) { s += I0[b * G + a]; m++; }
        const x = (i + 0.5) * v - MMW / 2, y = (j + 0.5) * v - MMW / 2, bias = 0.93 + 0.1 * Math.exp(-(x * x + (y - 6) * (y - 6)) / 9000) + 0.05 * Math.sin(x / 55 + 0.6) * Math.cos(y / 70), sig = s / m * bias, n1 = M.gauss(rr) * sd, n2 = M.gauss(rr) * sd, val = clamp(Math.sqrt((sig + n1) * (sig + n1) + n2 * n2), 0, 255), o = 4 * (j * n + i);
        V[j * n + i] = val; im.data[o] = val * 0.98; im.data[o + 1] = val; im.data[o + 2] = val * 1.02; im.data[o + 3] = 255; }
    gg.putImageData(im, 0, 0); S.canvas = cc; S.vox = v; S.n = n;
    /* the interpolated picture a viewer shows: cubic, to three or four times the voxel grid */
    const f = clamp(Math.round(G / n), 1, 4); if (f === 1) { S.fine = cc; return; }
    const m = n * f, T = new Float32Array(m * n), fc = mk(m, m), fg = fc.getContext('2d'), fim = fg.createImageData(m, m), cub = (a, b, c, d, t) => b + 0.5 * t * (c - a + t * (2 * a - 5 * b + 4 * c - d + t * (3 * (b - c) + d - a)));
    for (let j = 0; j < n; j++) for (let i = 0; i < m; i++) { const u = (i + 0.5) / f - 0.5, a = Math.floor(u), o = j * n; T[j * m + i] = cub(V[o + clamp(a - 1, 0, n - 1)], V[o + clamp(a, 0, n - 1)], V[o + clamp(a + 1, 0, n - 1)], V[o + clamp(a + 2, 0, n - 1)], u - a); }
    for (let j = 0; j < m; j++) { const u = (j + 0.5) / f - 0.5, a = Math.floor(u), t = u - a, r0 = clamp(a - 1, 0, n - 1) * m, r1 = clamp(a, 0, n - 1) * m, r2 = clamp(a + 1, 0, n - 1) * m, r3 = clamp(a + 2, 0, n - 1) * m; for (let i = 0; i < m; i++) { const val = clamp(cub(T[r0 + i], T[r1 + i], T[r2 + i], T[r3 + i], t), 0, 255), o = 4 * (j * m + i); fim.data[o] = val * 0.98; fim.data[o + 1] = val; fim.data[o + 2] = val * 1.02; fim.data[o + 3] = 255; } }
    fg.putImageData(fim, 0, 0); S.fine = fc; };
  S.voxelise(1); return S;
})();

/* ════ Levels 2 and 3: fluorescence micrographs, in micrometres. Origin: the spot the descent continues into.
   Cells are grown by rule (a wandering, tapering, forking walk for every dendrite and axon), sorted by depth, and each
   depth is blurred by its distance from the plane of focus, as a microscope would blur it. Bright things bloom. ════ */
const PIA = -650, CTX = M.SET.cortexMm * 1000, LAY = [0, 0.09, 0.18, 0.40, 0.52, 0.70, 1.0], LNAME = ['I', 'II', 'III', 'IV', 'V', 'VI'];
const layerOf = y => { const d = (y - PIA) / CTX; if (d < 0) return -1; for (let i = 0; i < 6; i++) if (d < LAY[i + 1]) return i; return 6; };
const TINT = [[176, 214, 226], [226, 172, 152], [140, 180, 228], [152, 216, 192]];
const HERO_DY = 1.45 - (M.NEURON.y0[M.HOME_SITE] + M.NEURON.y1[M.HOME_SITE]) / 2;      /* slide the cell so its oblique dendrite passes exactly where the later pictures put it */
let labelFrac = 0.02;
const tintI = (c, I) => { const k = Math.min(I, 1), w = I > 0.8 ? (I - 0.8) * 210 : 0; return 'rgb(' + Math.min(255, (c[0] * k + w) | 0) + ',' + Math.min(255, (c[1] * k + w) | 0) + ',' + Math.min(255, (c[2] * k + w) | 0) + ')'; };
/* cell bodies: one grid cell of 26 µm holds at most one */
const CS = 26, DENS = [0.06, 0.92, 0.7, 0.98, 0.5, 0.72];
function cellAt(i, j) { const Y = (j + 0.5 + 0.7 * (hash2(i, j, 42) - 0.5)) * CS, L = layerOf(Y); if (L < 0 || L > 5 || hash2(i, j, 41) > DENS[L]) return null;
  const X = (i + 0.5 + 0.7 * (hash2(i, j, 43) - 0.5)) * CS, R = (L === 4 ? 7.5 : L === 3 ? 4.2 : 5.5) * (0.8 + 0.5 * hash2(i, j, 44)); if (Math.hypot(X + 150, Y - 228) < 40) return null;
  return { X, Y, R, L, i, j, lab: hash2(i, j, 45), z: hash2(i, j, 59), t: (hash2(i, j, 46) * 4) | 0, br: 0.5 + 0.5 * hash2(i, j, 47) }; }
/* the branches of one labelled cell; lod 0 is a sketch for the widest picture, 2 is complete */
function growCell(c, lod) { const R = M.rng(1 + ((Math.imul(c.i, 73856093) ^ Math.imul(c.j, 19349663)) >>> 0)), rs = () => R() * 2 - 1, rnd = (a, b) => a + (b - a) * R(), paths = [], X = c.X, Y = c.Y, r = c.R, L = c.L, up = -Math.PI / 2;
  const limb = (x, y, a, len, w0, w1, o) => { const st = o.step || 7, n = Math.max(2, Math.round(len / st)), p = [x, y]; let bend = 0;
    for (let i = 1; i <= n; i++) { bend = bend * 0.72 + rs() * o.wig; a += bend + (o.to !== undefined ? angDiff(o.to, a) * (o.pull || 0.05) : 0); if (o.roof && y < PIA + 30) a += angDiff(Math.cos(a) < 0 ? Math.PI : 0, a) * 0.35;
      x += Math.cos(a) * st; y += Math.sin(a) * st; if (o.roof) y = Math.max(y, PIA + 7); p.push(x, y); if (o.at && o.at.length && i * st >= o.at[0]) { o.at.shift(); o.sprout(x, y, a, lerp(w0, w1, i / n)); } }
    paths.push({ p, w0, w1, k: o.k || 0 }); return [x, y, a]; };
  const tree = (x, y, a, len, w, order, o) => { const e = limb(x, y, a, len, w, w * (order > 0 ? 0.8 : 0.45), o); if (order > 0) { const sp = o.spread * rnd(0.7, 1.25), tilt = rs() * 0.25; for (const s of [-1, 1]) tree(e[0], e[1], e[2] + s * sp / 2 * rnd(0.5, 1.5) + tilt, len * o.decay * rnd(0.6, 1.4), w * rnd(0.64, 0.8), order - 1, o); } };
  const pyr = L !== 3 && R() < 0.82;
  if (pyr) { const big = L === 4, deep = L === 5, toPia = L <= 2 || (big && R() < 0.85) || (deep && R() < 0.25), topY = toPia ? PIA + rnd(45, 120) : Y - rnd(260, 620), trunk = Math.max(30, Y - r * 1.2 - topY), w0 = big ? 3.6 : 2.4, at = [];
    if (lod > 0) { for (let i = 0, n = big ? 8 : 5; i < n; i++) at.push(trunk * Math.pow(rnd(0.04, 0.8), 1.3)); at.sort((a, b) => a - b); }
    const e = limb(X, Y - r * 1.2, up + rs() * 0.04, trunk, w0, w0 * 0.5, { wig: 0.014, to: up, pull: 0.07, k: 1, step: 9, at, sprout: (px, py, pa, pw) => tree(px, py, pa + (R() < 0.5 ? -1 : 1) * rnd(0.85, 1.3), rnd(50, 115), Math.max(0.7, pw * 0.4), lod > 1 && R() < 0.45 ? 1 : 0, { wig: 0.05, spread: 0.9, decay: 0.9 }) });
    tree(e[0], e[1], e[2], rnd(40, 75), w0 * 0.5, lod > 0 ? (big ? 3 : 2) : 1, { wig: 0.085, spread: big ? 1.3 : 1.1, decay: 1.3, roof: toPia, to: up, pull: 0.012 });
    const nb = big ? 7 : 4 + Math.floor(R() * 3); for (let i = 0; i < nb; i++) { const a = lerp(-0.25, Math.PI + 0.25, (i + 0.5) / nb + rs() * 0.06); tree(X + Math.cos(a) * r * 0.8, Y + r * 0.42 + Math.sin(a) * r * 0.32, a, rnd(22, 38) * (big ? 1.4 : 1), r * 0.26, lod > 0 ? 2 : 1, { wig: 0.06, spread: 0.85, decay: 1.5 }); }
    if (lod > 0) limb(X, Y + r, Math.PI / 2 + rs() * 0.1, Math.max(60, Math.min(PIA + CTX + 160 - Y, rnd(500, 1500))), 0.7, 0.5, { wig: 0.03, to: Math.PI / 2, pull: 0.05, k: 2, step: 12, at: [rnd(60, 140), rnd(180, 330)], sprout: (px, py) => limb(px, py, (R() < 0.5 ? 0 : Math.PI) + rs() * 0.4, rnd(120, 320), 0.45, 0.4, { wig: 0.06, k: 2, step: 10 }) }); }
  else { const n = 5 + Math.floor(R() * 3); for (let i = 0; i < n; i++) { const a = i * TAU / n + rs() * 0.4; tree(X + Math.cos(a) * r * 0.8, Y + Math.sin(a) * r * 0.8, a, rnd(26, 46), r * 0.3, lod > 0 ? 2 : 1, { wig: 0.075, spread: 0.9, decay: 1.3 }); }
    if (lod > 0) for (let i = 0; i < 3; i++) tree(X, Y + r * 0.8, Math.PI / 2 + rs() * 2.6, rnd(40, 70), 0.5, lod > 1 ? 3 : 2, { wig: 0.16, spread: 1.15, decay: 0.92, k: 2, step: 6 }); }
  return { paths, pyr }; }
function somaPath(g, x, y, r, pyr) { g.beginPath(); if (!pyr) { g.ellipse(x, y, r, r * 0.92, 0, 0, TAU); return; }
  g.moveTo(x, y - r * 1.6); g.bezierCurveTo(x + r * 0.36, y - r * 0.8, x + r * 1.05, y + r * 0.05, x + r * 0.98, y + r * 0.6); g.bezierCurveTo(x + r * 0.7, y + r * 0.96, x - r * 0.7, y + r * 0.96, x - r * 0.98, y + r * 0.6); g.bezierCurveTo(x - r * 1.05, y + r * 0.05, x - r * 0.36, y - r * 0.8, x, y - r * 1.6); }
function drawSoma(g, x, y, r, pyr, col, s, I) { if (r * s < 2.2) { const R = Math.max(r * 1.3, 1.5 / s), gr = g.createRadialGradient(x, y, 0, x, y, R); gr.addColorStop(0, tintI(col, 1.05 * I)); gr.addColorStop(0.5, tintI(col, 0.7 * I)); gr.addColorStop(1, 'rgba(0,0,0,0)'); g.fillStyle = gr; g.fillRect(x - R, y - R, 2 * R, 2 * R); return; }
  const ny = y + (pyr ? r * 0.1 : 0), gr = g.createRadialGradient(x - r * 0.08, ny, r * 0.08, x, ny, r * 1.3); gr.addColorStop(0, tintI(col, 0.26 * I)); gr.addColorStop(0.3, tintI(col, 0.32 * I)); gr.addColorStop(0.47, tintI(col, 0.98 * I)); gr.addColorStop(0.7, tintI(col, 1.12 * I)); gr.addColorStop(1, tintI(col, 0.92 * I)); somaPath(g, x, y, r, pyr); g.fillStyle = gr; g.fill(); }
function drawCell(g, c, m, s, I) { const col = TINT[c.t], psf = 1.0 / s; g.lineCap = 'round'; g.lineJoin = 'round'; let h = (Math.imul(c.i, 7919) + c.j) | 0; const rn = () => { h = (Math.imul(h, 1103515245) + 12345) & 0x7fffffff; return h / 0x7fffffff; };
  for (const P of m.paths) { const p = P.p, n = p.length / 2, ax = P.k === 2, ch = Math.max(2, Math.ceil(n / 3));
    for (let i = 0; i < n - 1; i += ch) { const j = Math.min(n - 1, i + ch), w = lerp(P.w0, P.w1, (i + j) / 2 / (n - 1)), J = Math.max(ax ? 0.2 : 0.3, Math.min(1.05, (ax ? 0.32 : 0.46) + 0.5 * Math.sqrt(w / 2.5)) * Math.pow(Math.min(1, w / psf), 0.36)) * I * (0.86 + 0.28 * rn());
      g.strokeStyle = tintI(col, J); g.lineWidth = Math.max(w, psf); g.beginPath(); g.moveTo(p[2 * i], p[2 * i + 1]); for (let q = i + 1; q <= j; q++) g.lineTo(p[2 * q], p[2 * q + 1]); g.stroke(); }
    if (ax && s > 0.7) { g.fillStyle = tintI(col, 0.8 * I); g.beginPath(); for (let i = 2; i < p.length; i += 2) if (rn() < 0.5) { const q = Math.max(0.55 + 0.4 * rn(), 0.9 / s); g.moveTo(p[i] + q, p[i + 1]); g.arc(p[i], p[i + 1], q, 0, TAU); } g.fill(); } }
  drawSoma(g, c.X, c.Y, c.R, m.pyr, col, s, I); }

/* the filled cell of the third picture: its branches are the model's compartments, drawn as a microscope would show them */
const HERO = (() => { const T = M.NEURON, br = [];
  for (let b = 0; b < T.first.length; b++) { const f = T.first[b], P = [[T.x0[f], T.y0[f]]], D = [T.d[f]]; for (let i = f; i <= T.last[b]; i++) { P.push([T.x1[i], T.y1[i]]); D.push(T.d[i]); }
    const S = spline(P, false, 4), pts = [], n = S.length; for (let m = 0; m < n; m++) { const a = S[Math.max(0, m - 1)], c = S[Math.min(n - 1, m + 1)], L = Math.hypot(c[0] - a[0], c[1] - a[1]) || 1, e = Math.min(1, m / 6, (n - 1 - m) / 3), keep = b === M.OBL ? M.smooth((Math.abs(S[m][0]) - 4) / 14) : 1, off = 1.5 * (vnoise(m * 0.3, b * 7.3, 301) - 0.5) * e * keep, di = m / 4, d0 = D[Math.min(D.length - 1, Math.floor(di))], d1 = D[Math.min(D.length - 1, Math.floor(di) + 1)];
      pts.push([S[m][0] - (c[1] - a[1]) / L * off, S[m][1] + (c[0] - a[0]) / L * off, lerp(d0, d1, di % 1) * (0.9 + 0.3 * vnoise(m * 0.45, b * 3.1, 303))]); }
    const hz = hash2(b, 3, 302); br.push({ pts, zc: b === 0 || b === M.OBL || b === M.OBL + 1 ? 0 : hz < 0.4 ? 0 : hz < 0.78 ? 1 : 2, b }); }
  /* the axon is not part of the model: it leaves the base of the cell body, gives off collaterals and carries boutons */
  const R = M.rng(404), rs = () => R() * 2 - 1, ax = [], sx = T.x0[0], sy = (T.y0[0] + T.y1[0]) / 2;
  const limb = (x, y, a, len, to, zc, gen) => { const p = [[x, y]]; let bend = 0, k = 0, next = gen ? 8 + 14 * R() : 5; for (let l = 0; l < len; l += 5) { bend = bend * 0.7 + rs() * (gen ? 0.09 : 0.05); a += bend + angDiff(to, a) * (gen ? 0.02 : 0.04); x += Math.cos(a) * 5; y += Math.sin(a) * 5; p.push([x, y]);
      if (++k >= next && gen < 3) { next = k + (gen ? 12 + 22 * R() : 7 + 9 * R()); const sd = R() < 0.5 ? 1 : -1, ba = gen ? a + sd * (0.7 + 0.8 * R()) : (sd > 0 ? 0 : Math.PI) + rs() * 0.7 - (R() < 0.3 ? sd * 1.1 : 0); limb(x, y, ba, (gen ? 60 : 150) + 190 * R(), ba + rs() * 0.5 - (gen === 0 && R() < 0.35 ? 0.9 * sd : 0), R() < 0.55 ? 0 : 1, gen + 1); } } ax.push({ p, zc, gen }); };
  limb(sx, sy + 9, Math.PI / 2 + 0.06, 640, Math.PI / 2, 0, 0);
  return { br, ax, sx, sy }; })();
function drawHero(g, s, zc, I) { const T = M.NEURON, col = [196, 232, 246], psf = 1.0 / s; g.save(); g.translate(0, HERO_DY); g.lineCap = 'round'; g.lineJoin = 'round'; g.globalCompositeOperation = 'lighter';
  for (const B of HERO.br) { if (B.zc !== zc) continue; const p = B.pts;
    if (s > 0.5) { g.strokeStyle = tintI(col, (s <= 1.6 ? 0.13 : 0.07) * I); g.lineWidth = 2.6 + 1 / s; g.beginPath(); g.moveTo(p[B.b === 0 ? 30 : 0][0], p[B.b === 0 ? 30 : 0][1]); for (let m = B.b === 0 ? 31 : 1; m < p.length; m++) g.lineTo(p[m][0], p[m][1]); g.stroke(); }      /* spines too small to see singly: a soft fringe */
    for (let m = 0; m < p.length - 1; m += 2) { const e = Math.min(p.length - 1, m + 2), w = p[m][2], J = Math.min(0.98, 0.34 + 0.5 * Math.sqrt(w / 2.6)) * Math.pow(Math.min(1, w / psf), 0.4) * I * (0.78 + 0.3 * vnoise(m * 0.21, B.b * 5.3, 305)); g.strokeStyle = tintI(col, J); g.lineWidth = Math.max(w, psf); g.beginPath(); g.moveTo(p[m][0], p[m][1]); g.lineTo(p[m + 1][0], p[m + 1][1]); g.lineTo(p[e][0], p[e][1]); g.stroke(); }
    if (s > 1.6) { g.strokeStyle = tintI(col, 0.42 * I); g.fillStyle = tintI(col, 0.7 * I); g.lineWidth = Math.max(0.17, 0.75 / s); g.beginPath(); const heads = [];
      for (let m = (B.b === 0 ? 30 : 1); m < p.length; m++) { const a = p[m - 1], c = p[m], dx = c[0] - a[0], dy = c[1] - a[1], L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L, off = c[2] * 0.45;
        for (let k = 0, d = hash2(m, B.b, 311) * 0.7; d < L; k++, d += 0.28 + 0.75 * hash2(m * 16 + k, B.b, 312)) { const bx = a[0] + ux * d, by = a[1] + uy * d; if (B.b === M.OBL && Math.abs(bx) < 0.9) continue; const sd = hash2(m * 16 + k, B.b, 313) < 0.5 ? 1 : -1, hh = hash2(m * 16 + k, B.b, 314), len = 0.35 + 1.25 * hh * hh, lean = (hash2(m * 16 + k, B.b, 315) - 0.5) * 0.9, nx = -uy * sd + ux * lean, ny = ux * sd + uy * lean;
          g.moveTo(bx + nx * off, by + ny * off); g.lineTo(bx + nx * (off + len), by + ny * (off + len)); heads.push(bx + nx * (off + len), by + ny * (off + len), Math.max(0.55 / s, len < 0.5 ? 0.16 : 0.22 + 0.2 * hash2(m * 16 + k, B.b, 316))); } }
      g.stroke(); g.beginPath(); for (let k = 0; k < heads.length; k += 3) { g.moveTo(heads[k] + heads[k + 2], heads[k + 1]); g.arc(heads[k], heads[k + 1], heads[k + 2], 0, TAU); } g.fill(); } }
  for (const A of HERO.ax) { if (A.zc !== zc) continue; const p = A.p; g.strokeStyle = tintI(col, (A.gen ? 0.36 : 0.46) * Math.pow(Math.min(1, 0.6 / psf), 0.3) * I); g.lineWidth = Math.max(A.gen ? 0.4 : 0.7, psf * (A.gen ? 0.7 : 0.9)); g.beginPath(); g.moveTo(p[0][0], p[0][1]); for (let m = 1; m < p.length; m++) g.lineTo(p[m][0], p[m][1]); g.stroke();
    if (s > 0.6) { g.fillStyle = tintI(col, 0.66 * I); g.beginPath(); for (let m = 3; m < p.length; m++) if (hash2(m, p.length, 317) < 0.62) { const q = Math.max(0.5 + 0.45 * hash2(m, p.length, 318), 0.85 / s), u = hash2(m, p.length, 319), x = lerp(p[m - 1][0], p[m][0], u), y = lerp(p[m - 1][1], p[m][1], u); g.moveTo(x + q, y); g.arc(x, y, q, 0, TAU); } g.fill(); } }
  if (zc === 0) { const sx = HERO.sx, sy = HERO.sy; g.strokeStyle = tintI(col, 0.9 * I); g.lineWidth = Math.max(1.5, psf); g.beginPath(); g.moveTo(sx, sy + 8); g.lineTo(sx + 0.3, sy + 16); g.stroke(); g.globalCompositeOperation = 'source-over'; drawSoma(g, sx, sy + 1, 10.4, true, col, s, I); g.globalCompositeOperation = 'lighter';
    if (s > 1.6) { g.fillStyle = tintI(col, 0.9 * I); g.beginPath(); g.rect(-0.06, 0.5 - HERO_DY, 0.12, 0.47); g.ellipse(0, 0.24 - HERO_DY, 0.27, 0.25, 0, 0, TAU); g.fill(); } }
  g.restore(); }
/* vessels carry no label: penetrating arterioles and veins from the surface, their branches, and a mesh of capillaries */
function vesselsIn(x0, y0, x1, y1) { const out = [], wm = PIA + CTX;
  const walk = (x, y, a, len, w, to, wig, pen, seed) => { const p = [x, y]; let bend = 0; for (let l = 0, k = 0; l < len; l += 14, k++) { bend = bend * 0.7 + (hash2(k, seed, 331) - 0.5) * 2 * wig; a += bend + angDiff(to, a) * 0.06; x += Math.cos(a) * 14; y += Math.sin(a) * 14; p.push(x, y);
      if (pen && k % 9 === 5) { const ba = (hash2(k, seed, 332) < 0.5 ? 0.3 : Math.PI - 0.3) + (hash2(k, seed, 333) - 0.5); walk(x, y, ba, 140 + 260 * hash2(k, seed, 334), 6 + 3 * hash2(k, seed, 335), ba + (hash2(k, seed, 336) - 0.5), 0.12, false, seed * 31 + k); } } out.push({ p, w, pen }); };
  for (let i = Math.floor((x0 - 500) / 640); i <= (x1 + 500) / 640; i++) { const X = (i + 0.2 + 0.6 * hash2(i, 0, 91)) * 640; if (Math.abs(X + 60) < 260) continue; walk(X, PIA - 30, Math.PI / 2, CTX * (0.45 + 0.6 * hash2(i, 0, 92)) + 30, 13 + 15 * hash2(i, 0, 93), Math.PI / 2, 0.035, true, i * 7 + 1); }
  for (let j = Math.floor(Math.max(y0 - 300, PIA + 40) / 150); j <= Math.min(y1 + 300, wm + 300) / 150; j++) for (let i = Math.floor((x0 - 300) / 150); i <= (x1 + 300) / 150; i++) { if (hash2(i, j, 337) > 0.42) continue; const X = (i + hash2(i, j, 338)) * 150, Y = (j + hash2(i, j, 339)) * 150; if (Math.hypot(X + 60, Y - 60) < 130) continue; { const ba = hash2(i, j, 340) * TAU; walk(X, Y, ba, 110 + 190 * hash2(i, j, 341), 5.5 + 2.5 * hash2(i, j, 342), ba + 1.2 * (hash2(i, j, 343) - 0.5), 0.14, false, i * 977 + j); } }
  return out; }
const strokeP = (g, p) => { g.beginPath(); g.moveTo(p[0], p[1]); for (let i = 2; i < p.length; i += 2) g.lineTo(p[i], p[i + 1]); g.stroke(); };
function drawVessels(g, Bk, k) { const s = Bk.s; g.lineCap = 'round'; g.lineJoin = 'round';
  for (const V of vesselsIn(Bk.x0, Bk.y0, Bk.x0 + Bk.w, Bk.y0 + Bk.w)) { const a = (V.pen ? 1 : 0.55) * k, w = Math.max(V.w, 1.3 / s); g.strokeStyle = 'rgba(0,1,2,' + 0.2 * a + ')'; g.lineWidth = w * 2; strokeP(g, V.p); g.strokeStyle = 'rgba(0,1,2,' + 0.36 * a + ')'; g.lineWidth = w * 1.4; strokeP(g, V.p); g.strokeStyle = 'rgba(0,1,2,' + 0.66 * a + ')'; g.lineWidth = w * 0.86; strokeP(g, V.p);
    if (V.pen && V.w * s > 3) { g.strokeStyle = 'rgba(170,205,214,0.1)'; g.lineWidth = Math.max(0.6, 0.9 / s); strokeP(g, V.p.map((v, i) => i % 2 ? v : v - V.w * 0.47)); strokeP(g, V.p.map((v, i) => i % 2 ? v : v + V.w * 0.47)); } } }
const HOLE = (() => { const c = mk(32, 32), g = c.getContext('2d'), gr = g.createRadialGradient(16, 16, 0, 16, 16, 16); gr.addColorStop(0, 'rgba(0,1,2,0.86)'); gr.addColorStop(0.55, 'rgba(0,1,2,0.7)'); gr.addColorStop(1, 'rgba(0,1,2,0)'); g.fillStyle = gr; g.fillRect(0, 0, 32, 32); return c; })();
/* autofluorescence of the neuropil: brighter where cells are dense, dimmer with depth, streaked in the white matter */
const LB = [0.86, 1.04, 0.92, 1.12, 0.8, 0.96];
const hazeLum = d => { if (d < 0) return 0; let L = 5; for (let i = 0; i < 6; i++) if (d < LAY[i + 1]) { L = i; break; } const m = (LAY[L] + LAY[L + 1]) / 2, o = d < m ? L - 1 : L + 1, e = d < m ? LAY[L] : LAY[L + 1], t = clamp(0.5 - Math.abs(d - e) / 0.07, 0, 0.5); return lerp(LB[L], o < 0 ? 0.5 : o > 5 ? 1.0 : LB[o], t); };
function* hazeOf(Bk, gain) { const n = Math.round(Bk.px / 3), c = mk(n, n), g = c.getContext('2d'), im = g.createImageData(n, n), q = Bk.w / n, f = Math.max(1, q / 6);
  for (let j = 0; j < n; j++) { const Y = Bk.y0 + (j + 0.5) * q; for (let i = 0; i < n; i++) { const X = Bk.x0 + (i + 0.5) * q, d = (Y - PIA) / CTX, o = 4 * (j * n + i); let v;
      if (d <= 1) v = hazeLum(d) * (0.5 + 0.55 * vnoise(X / 300, Y / 300, 41) + 0.34 * vnoise(X / 70, Y / 70, 42) + 0.2 * vnoise(X / (14 * f), Y / (14 * f), 40) + 0.16 * hash2(i, j, 43)) * (1 - 0.3 * d);
      else { const t = clamp((d - 1) / 0.03, 0, 1); v = lerp(0.9, 0.8 + 0.9 * vnoise(X / 800, (Y + 40 * Math.sin(X / 900)) / 26, 44) + 0.4 * vnoise(X / 190, Y / 11, 45) + 0.12 * hash2(i, j, 43), t) * (0.72 - 0.2 * clamp((d - 1) / 1.2, 0, 1)); }
      v *= gain; im.data[o] = 13 * v; im.data[o + 1] = 25 * v; im.data[o + 2] = 30 * v; im.data[o + 3] = 255; } if ((j & 63) === 63) yield; }
  g.putImageData(im, 0, 0); return c; }
let TMPC = null; const tmpFor = px => { if (!TMPC || TMPC.width !== px) TMPC = mk(px, px); const g = TMPC.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1; g.clearRect(0, 0, px, px); return TMPC; };
const worldOn = (c, Bk) => { const g = c.getContext('2d'); g.setTransform(Bk.s, 0, 0, Bk.s, -Bk.x0 * Bk.s, -Bk.y0 * Bk.s); return g; };
const stamp = (g, c, blur, a) => { g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'lighter'; g.globalAlpha = a; g.drawImage(soft(c, blur), 0, 0); g.restore(); };
/* the unlabelled tissue and the surface, shared by both fluorescence pictures */
function* tissue(Bk, gain, holeA) { const g = world(Bk), s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, x1 = x0 + Bk.w, y1 = y0 + Bk.w, wm = PIA + CTX;
  g.fillStyle = '#010203'; g.fillRect(x0, y0, Bk.w, Bk.w); const hz = yield* hazeOf(Bk, gain); g.imageSmoothingEnabled = true; g.imageSmoothingQuality = 'high'; g.drawImage(hz, x0, y0, Bk.w, Bk.w);
  let cnt = 0; for (let j = Math.floor(Math.max(y0, PIA) / CS) - 1; j <= Math.min(y1, wm) / CS + 1; j++) { for (let i = Math.floor(x0 / CS) - 1; i <= x1 / CS + 1; i++) { const c = cellAt(i, j); if (!c || c.lab < labelFrac) continue; const r = Math.max(c.R * 1.35, 0.9 / s), ry = r * (c.L !== 3 ? 1.2 : 1); g.globalAlpha = holeA * (0.55 + 0.45 * hash2(i, j, 48)) * Math.min(1, c.R * 1.35 / r); g.drawImage(HOLE, c.X - r, c.Y - ry, 2 * r, 2 * ry); } if (++cnt % 40 === 0) yield; }
  g.globalAlpha = 1; }
function surface(g, Bk) { const s = Bk.s, x0 = Bk.x0 - 50, x1 = Bk.x0 + Bk.w + 50, pia = x => PIA + 6 * Math.sin(x / 380) + 3 * Math.sin(x / 97 + 1); if (Bk.y0 > PIA + 40) return;
  g.beginPath(); g.moveTo(x0, Bk.y0 - 10); for (let x = x0; x <= x1; x += 20) g.lineTo(x, pia(x)); g.lineTo(x1, Bk.y0 - 10); g.closePath(); g.fillStyle = '#010203'; g.fill();
  const line = (dy, col, w) => { g.beginPath(); for (let x = x0; x <= x1; x += 16) g.lineTo(x, pia(x) + dy(x)); g.strokeStyle = col; g.lineWidth = w; g.stroke(); };
  line(() => 0, 'rgba(150,200,214,0.1)', Math.max(22, 5 / s)); line(() => 0, 'rgba(160,208,220,0.2)', Math.max(9, 2.4 / s)); line(() => 0, 'rgba(206,232,238,0.6)', Math.max(3, 1.1 / s));
  line(x => -70 - 22 * Math.sin(x / 470) - 8 * Math.sin(x / 103), 'rgba(160,196,206,0.15)', Math.max(5, 1 / s)); line(x => -34 - 12 * Math.sin(x / 330 + 1), 'rgba(160,196,206,0.07)', Math.max(12, 2 / s)); line(x => -190 + 18 * Math.sin(x / 610 + 1), 'rgba(140,176,190,0.13)', Math.max(4, 1 / s));
  for (let i = Math.floor(x0 / 420); i <= x1 / 420; i++) { const x = (i + 0.15 + 0.7 * hash2(i, 1, 95)) * 420, w = 22 + 70 * hash2(i, 2, 95) * hash2(i, 3, 95), h = w * (0.55 + 0.25 * hash2(i, 4, 95)); g.beginPath(); g.ellipse(x, pia(x) - h * 0.85, w, h, 0, 0, TAU); g.fillStyle = '#000'; g.fill(); g.strokeStyle = 'rgba(180,212,222,0.24)'; g.lineWidth = Math.max(2.2, 1 / s); g.stroke(); } }
function* paintCortex(Bk) { const s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, x1 = x0 + Bk.w, y1 = y0 + Bk.w, wm = PIA + CTX, g = world(Bk), lod = s < 0.2 ? 0 : s < 0.5 ? 1 : 2;
  yield* tissue(Bk, 0.82, 0.8);
  /* labelled cells, far to near; each depth has its own blur */
  const cells = [[], [], []], mg = 700; for (let j = Math.floor((Math.max(y0, PIA) - mg) / CS); j <= (Math.min(y1, wm) + mg) / CS; j++) for (let i = Math.floor((x0 - 300) / CS); i <= (x1 + 300) / CS; i++) { const c = cellAt(i, j); if (c && c.lab < labelFrac) { const dz = Math.abs(c.z - 0.5); cells[dz < 0.13 ? 0 : dz < 0.31 ? 1 : 2].push(c); } }
  const thin = labelFrac > 0.05 ? 1 : 2;
  for (const k of [2, 1, 0]) { const t = tmpFor(Bk.px), tg = worldOn(t, Bk); tg.globalCompositeOperation = 'lighter'; let n = 0;
    if (k === 2) { tg.lineCap = 'round'; for (let j = Math.floor(wm / 16); j <= y1 / 16; j++) for (let i = Math.floor(x0 / 420) - 3; i <= x1 / 420; i++) { if (hash2(i, j, 31) > 0.6) continue; const X = (i + hash2(i, j, 32)) * 420, Y = j * 16 + 16 * hash2(i, j, 33), L = 300 + 900 * hash2(i, j, 34); if (Y < wm + 4) continue; tg.strokeStyle = rgba([130, 172, 190], 0.1 + 0.2 * hash2(i, j, 38)); tg.lineWidth = Math.max(1.6, 0.9 / s); tg.beginPath(); tg.moveTo(X, Y); tg.bezierCurveTo(X + L / 3, Y + 18 * (hash2(i, j, 35) - 0.5), X + 2 * L / 3, Y + 18 * (hash2(i, j, 36) - 0.5), X + L, Y + 26 * (hash2(i, j, 37) - 0.5)); tg.stroke(); } }
    for (const c of cells[k]) { drawCell(tg, c, growCell(c, Math.min(lod, thin)), s, c.br * (k === 0 ? 1 : k === 1 ? 0.8 : 0.6)); if (++n % 24 === 0) yield; }
    if (k === 0) drawHero(tg, s, 0, 1), drawHero(tg, s, 1, 1), drawHero(tg, s, 2, 1);
    const b = [0.45, Math.max(1.0, 4 * s), Math.max(2.4, 11 * s)][k]; stamp(g, t, b, [1, 0.9, 0.78][k]); if (k === 0 && s < 0.6) stamp(g, t, 0.7, 0.55); stamp(g, t, b * 2 + 4, 0.4); if (k === 0) stamp(g, t, 14, 0.24); yield; }
  drawVessels(g, Bk, 1); surface(g, Bk); }
function* paintNeuron(Bk) { const s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, x1 = x0 + Bk.w, y1 = y0 + Bk.w, g = world(Bk);
  yield* tissue(Bk, 0.62, 0.62);
  { const c = mk(Bk.px, Bk.px); c.getContext('2d').drawImage(Bk.c, 0, 0); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(soft(c, Math.max(1.2, 2.4 * s)), 0, 0); g.restore(); }      /* the tissue round the cell lies off the plane of focus */
  drawVessels(g, Bk, 0.6); yield;
  /* neighbours carry the same label but lie in front of the focal plane or behind it */
  const near = [[], []]; for (let j = Math.floor((y0 - 700) / CS); j <= (y1 + 700) / CS; j++) for (let i = Math.floor((x0 - 300) / CS); i <= (x1 + 300) / CS; i++) { const c = cellAt(i, j); if (c && c.lab < 0.02) near[Math.abs(c.z - 0.5) < 0.22 ? 0 : 1].push(c); }
  for (const k of [1, 0]) { const t = tmpFor(Bk.px), tg = worldOn(t, Bk); tg.globalCompositeOperation = 'lighter'; let n = 0; for (const c of near[k]) { drawCell(tg, c, growCell(c, 2), s, c.br); if (++n % 12 === 0) yield; } stamp(g, t, Math.max(1.6, (k ? 13 : 5) * s), k ? 0.3 : 0.42); yield; }
  for (const k of [2, 1, 0]) { const t = tmpFor(Bk.px), tg = worldOn(t, Bk); drawHero(tg, s, k, 1); const b = [Math.max(0.5, 0.2 * s), Math.max(1, 2.2 * s), Math.max(2.2, 5.5 * s)][k]; stamp(g, t, b, [0.92, 0.8, 0.6][k]); stamp(g, t, b * 2 + 3.5, 0.26); if (k === 0) { stamp(g, t, Math.max(2, 1.4 * s), 0.16); stamp(g, t, 11, 0.16); stamp(g, t, 30, 0.1); } yield; }
  surface(g, Bk); }

/* ════ Levels 4 and 5: electron micrographs of thin sections, as one function of position (µm) so that the two agree.
   Every point of the neuropil belongs to some profile. Three populations are laid over one another: large profiles
   (dendrites, myelinated axons, glia), middling ones (boutons full of vesicles, spines, fine glial sheets) and a
   ground of fine axons cut across or along. Each has its own membrane and its own interior. The dendrite, its spine
   and the terminal that meets it are fixed shapes shared by every later picture. ════ */
const sdEll = (x, y, cx, cy, rx, ry) => (Math.hypot((x - cx) / rx, (y - cy) / ry) - 1) * Math.min(rx, ry);
const sdCap = (x, y, ax, ay, bx, by, r) => { const dx = bx - ax, dy = by - ay, t = clamp(((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy), 0, 1); return Math.hypot(x - ax - dx * t, y - ay - dy * t) - r; };
let emId = 0, memD = 40;
const idHash = (id, k) => hash2(id, id >> 11, k);
/* a membrane at depth e inside a profile. Seen closely it is two dark leaflets; from further off, one dark line. */
const mem = (v, e, pw, k) => { if (e > 0.014 * k + pw) return v;
  if (pw < 0.0009) { const a = Math.min(Math.abs(e - 0.0016), Math.abs(e - 0.0058)), t = clamp((0.0019 - a) / 0.001, 0, 1); return lerp(e < 0.0058 ? lerp(v, 80, 0.4) : v, memD, t); }
  const w = Math.max(pw < 0.006 ? 0.0066 : 0.0085 * k, pw * 0.6); return lerp(v, memD, clamp((w - e) / pw + 0.5, 0, 1)); };
/* synaptic vesicles: rings about 40 nm across, of slightly different sizes, bunched where `keep` is high */
const vesicles = (x, y, pw, base, keep) => { const c = 0.043, j = Math.floor(y / c), i = Math.floor(x / c); let g = base;
  for (let b = j - 1; b <= j + 1; b++) for (let a = i - 1; a <= i + 1; a++) { const cx = (a + 0.5 + 0.56 * (hash2(a, b, 62) - 0.5)) * c, cy = (b + 0.5 + 0.5 * (hash2(a, b, 63) - 0.5)) * c; if (hash2(a, b, 61) > keep(cx, cy)) continue;
      const r = 0.016 + 0.0075 * hash2(a, b, 65), d = Math.hypot(x - cx, y - cy); if (d > r + pw * 0.5) continue; const h = hash2(a, b, 64);
      if (pw > 0.012) g = Math.min(g, base - 30 - 14 * h); else { const t = clamp(Math.min(r + pw * 0.5 - d, d - (r - 0.0052)) / Math.max(pw, 0.0012), 0, 1); if (d > r - 0.0052) g = Math.min(g, lerp(base, base - 100 + 30 * h, t)); else if (g === base) g = base + 18 - 30 * h * h; } }
  return g; };
const cloud = (cx, cy) => clamp(1.45 - Math.hypot(cx * 0.8, cy + 0.04) / 0.31, 0, 0.97) * (0.5 + 0.9 * vnoise(cx / 0.12, cy / 0.12, 66));
/* a mitochondrion in its own coordinates: two membranes, a dense matrix and cristae that run part of the way across */
const mito = (u, v, rx, ry, pw, sd) => { const e = Math.hypot(u / rx, v / ry); if (e > 1) return -1; const d = (1 - e) * ry;
  if (pw > 0.016) return d < pw ? 56 : 92 + 16 * Math.sin(u / 0.03 + sd);
  if (d < 0.0065) return 50; if (d < 0.0125) return 128; if (d < 0.019) return 56;
  const c = Math.sin(u / 0.0125 + 3.2 * vnoise(v / 0.05, u / 0.3, 67 + sd) + sd), on = vnoise(u / 0.05 + sd, v / 0.045, 68) > 0.3; return !on ? 100 : Math.abs(c) < 0.26 ? 142 : Math.abs(c) < 0.6 ? 60 : 100; };
/* microtubules cut across: small dark rings; granules of glycogen: small dark dots in bunches */
const dots = (x, y, pw, c, r, ring, p, s) => { const i = Math.floor(x / c), j = Math.floor(y / c); let m = 0;
  for (let b = j - 1; b <= j + 1; b++) for (let a = i - 1; a <= i + 1; a++) { if (hash2(a, b, s) > p) continue; const d = Math.hypot(x - (a + 0.5 + 0.8 * (hash2(a, b, s + 1) - 0.5)) * c, y - (b + 0.5 + 0.8 * (hash2(a, b, s + 2) - 0.5)) * c); if (d < r + pw * 0.5 && (!ring || pw > 0.006 || d > r - 0.0055)) m = 1; }
  return m; };
/* one blob-shaped profile of a population: signed distance and what it is */
const BL = [{ c: 2.2, p: 0.6, r0: 0.4, r1: 0.62, s: 100 }, { c: 0.8, p: 0.66, r0: 0.15, r1: 0.2, s: 120 }];
const hit = { sd: 0, sd2: 0, id: 0, kind: 0, cx: 0, cy: 0, a: 0, b: 0, co: 1, si: 0, psd: 0 };
const clearOf = (cx, cy, R) => Math.abs(cy - 1.45) > 0.74 + R && Math.hypot(cx, cy + 0.05) > 0.62 + R && sdCap(cx, cy, 0.32, -0.46, 2.9, -1.75, 0.12) > R && Math.hypot(cx - 0.66, cy - 0.3) > 0.34 + R;
function blobs(x, y, L) { const B = BL[L], c = B.c, i0 = Math.floor(x / c), j0 = Math.floor(y / c), s = B.s; hit.sd = 1e9; hit.sd2 = 1e9; hit.psd = 0;
  for (let j = j0 - 1; j <= j0 + 1; j++) for (let i = i0 - 1; i <= i0 + 1; i++) { if (hash2(i, j, s) > B.p) continue; const cx = (i + 0.2 + 0.6 * hash2(i, j, s + 1)) * c, cy = (j + 0.2 + 0.6 * hash2(i, j, s + 2)) * c, h = hash2(i, j, s + 3), R = B.r0 + B.r1 * h * h, dx = x - cx, dy = y - cy; if (dx * dx + dy * dy > R * R * 5.5) continue; if (!clearOf(cx, cy, R * 1.25)) continue;
      const asp = 1 + (L ? 2.2 : 1.1) * Math.pow(hash2(i, j, s + 4), 2.5), a = R * Math.sqrt(asp), b = R / Math.sqrt(asp), th = hash2(i, j, s + 5) * Math.PI, co = Math.cos(th), si = Math.sin(th), u = dx * co + dy * si, v = -dx * si + dy * co, rho = Math.hypot(u / a, v / b) || 1e-6, ph = 37 * hash2(i, j, s + 6);
      const sd = (rho - 1 - 0.34 * (vnoise(1.5 * u / (a * rho) + ph, 1.5 * v / (b * rho) + ph, s + 7) - 0.5)) * b * (0.6 + 0.4 * Math.min(rho, 1.5));
      if (sd < hit.sd) { hit.sd2 = hit.sd; hit.sd = sd; hit.id = (L + 1) * 20000000 + (i & 4095) * 4096 + (j & 4095); hit.kind = hash2(i, j, s + 8); hit.cx = cx; hit.cy = cy; hit.a = a; hit.b = b; hit.co = co; hit.si = si; if (L === 1 && hit.kind < 0.36 && sd > 0 && sd < 0.06) { const pa = hash2(i, j, s + 9) * TAU, al = Math.cos(Math.atan2(v / b, u / a) - pa); hit.psd = al > 0.72 ? (al - 0.72) / 0.28 : 0; } else hit.psd = 0; }
      else if (sd < hit.sd2) hit.sd2 = sd; }
  return hit.sd < 0; }
const EMS = [0.16, 0.25, 0.17];
const emRg = (x, y) => vnoise(x / 2.3 + 40, y / 2.3 + 40, 20) + 0.25 * (vnoise(x / 0.7, y / 0.7, 21) - 0.5), emScale = (x, y) => { const rg = emRg(x, y); return rg < 0.42 ? 0 : rg < 0.66 ? 1 : 2; };
const leader = (i, j, sc) => { let a = i, b = j; for (let h = 0; h < 2; h++) { if (hash2(a, b, 11 + sc) < 0.5) break; const hz = vnoise(a / 7, b / 7, 13) < 0.5, sg = hash2(a, b, 12) < 0.5 ? 1 : -1; if (hz) a += sg; else b += sg; } return 1 + (((a & 8191) * 8192 + (b & 8191)) * 4 + sc); };
function emPix(X, Y, pw) {
  const g1 = vnoise(X / 0.0075, Y / 0.0075, 5) - 0.5, g2 = vnoise(X / 0.045, Y / 0.045, 6) - 0.5, fl = pw < 0.006 ? vnoise(X / 0.019, Y / 0.019, 28) - 0.5 : 0, grain = (pw < 0.006 ? 30 : 15) * g1 + 11 * g2 + 22 * fl, mk1 = 0.8 + 0.6 * vnoise(X / 0.31, Y / 0.31, 9); memD = 38 + (pw < 0.006 ? 70 * M.smooth((vnoise(X / 0.06, Y / 0.06, 93) - 0.6) / 0.25) : 0);
  /* the dendrite, its spine, the terminal and a sheet of astrocyte beside the cleft */
  if (Y > -1.9 && Y < 2.25) {
    const sB = Math.max(Math.min(sdEll(X, Y, 0.02, -0.27, 0.42, 0.31), sdCap(X, Y, 0.32, -0.46, 2.9, -1.75, 0.085)), Y + 0.002);
    const sS = Math.min(Math.max(sdEll(X, Y, 0, 0.22, 0.27, 0.25), 0.02 - Y), sdCap(X, Y, 0, 0.44, 0.02, 0.97, 0.065));
    const wave = 0.11 * Math.sin(X * 1.1) + 0.04 * Math.sin(X * 3.1), sD = Math.min(sS, Math.abs(Y - 1.45 - wave) - 0.5 - 0.07 * Math.sin(X * 2.3 + 1));
    if (sB < 0) { emId = 900001; let v = 152 + grain; const dc = Math.hypot(X + 0.24, Y + 0.36);
      const m = mito((X - 0.16) * 0.955 - (Y + 0.39) * 0.296, (X - 0.16) * 0.296 + (Y + 0.39) * 0.955, 0.15, 0.08, pw, 1);
      if (m >= 0) v = m + grain * 0.5;
      else if (dc < 0.046) v = dc < 0.028 ? 62 + grain : dc > 0.04 ? 70 : 168;                                                    /* one dense-core vesicle */
      else if (Y > -0.05) v = 152 + grain - 44 * M.smooth((Y + 0.04) / 0.032) * M.smooth((0.185 - Math.abs(X)) / 0.03) * (0.25 + 0.75 * vnoise(X / 0.026, Y / 0.02, 91));      /* dense material of the active zone */
      else if (Y < -0.052 && sB < -0.028 && X < 0.34) v = vesicles(X, Y, pw, 152, cloud) + grain;
      else if (X > 0.4) v = 158 - (dots(X, Y, pw, 0.06, 0.011, true, 0.5, 130) ? 50 : 0) + grain;
      return mem(v, -sB, pw, 1); }
    if (sD < 0) { emId = 900002; let v;
      if (sS < 0) { v = 174 - 22 * vnoise(X / 0.028, Y / 0.028, 8) - 12 * vnoise(X / 0.09, Y / 0.09, 10) + grain;                    /* the spine: a fluffy web of actin */
        if (Math.abs(X) < 0.2 && Y < 0.12) { const th = (0.03 + 0.034 * vnoise(X / 0.06, 3.3, 9)) * M.smooth((0.2 - Math.abs(X)) / 0.045), u = clamp((Y - 0.026) / Math.max(th, 1e-4) + 0.7 * (vnoise(X / 0.014, Y / 0.014, 92) - 0.5), 0, 1.6), t = M.smooth(u * 0.7); v = lerp(46 + 1.5 * grain, v, t * t); }       /* the postsynaptic density */
        const ay = Y - 0.36 - 0.012 * Math.sin(X * 60) - 0.1 * X, li = Math.floor(ay / 0.03), lw = 0.03 + 0.035 * hash2(li, 0, 145), lx = 0.03 * (hash2(li, 1, 145) - 0.5); if (li >= 0 && li < 5 && Math.abs(X - lx) < lw) { const l = Math.abs((ay / 0.03) % 1 - 0.5), en = clamp((lw - Math.abs(X - lx)) / 0.008, 0, 1); v = lerp(v, l < 0.12 ? 176 : l < 0.27 ? 64 : 118 + grain, en); }        /* the spine apparatus: stacked sacs */
        return mem(v, -sD, pw, 1); }
      const yy = Y - 1.45 - wave, mx = Math.round((X + 1.3) / 7.3) * 7.3 - 1.3, m = mito(X - mx, (Y - 1.5 - 0.11 * Math.sin(mx * 1.1)) - 0.03 * (X - mx), 0.5, 0.12, pw, 2);
      if (m >= 0) v = m + grain * 0.5;
      else { const li = Math.floor(yy / 0.052), lc = (li + 0.5 + 0.5 * (hash2(li, 0, 141) - 0.5)) * 0.052 + 0.006 * Math.sin(X * 9 + li), on = vnoise(X / 0.5 + li * 3.7, li, 142) > 0.3; v = 172 + grain - (on && Math.abs(yy - lc) < Math.max(0.0055, pw * 0.5) ? 44 : 0) - 14 * vnoise(X / 0.03, Y / 0.03, 8);
        const ex = X - (Math.round((X - 2.1) / 5.9) * 5.9 + 2.1), sr = Math.abs(Math.hypot(ex / 2.6, (yy + 0.2) / 1) * 0.16 - 0.11); if (sr < 0.006 && Math.abs(ex) < 0.3) v = 70; }
      return mem(v, -sD, pw, 1); }
    if (Y > 0 && Y < 0.02 && Math.abs(X) < 0.19) { emId = 0; return 136 - 22 * Math.exp(-Math.pow((Y - 0.0105) / 0.003, 2)) * (0.5 + vnoise(X / 0.012, 1, 88)) - 26 * (vnoise(X / 0.009, Y / 0.02, 89) > 0.62 ? 1 : 0) + grain; }
    const sG = Math.min(sdCap(X, Y, 0.31, 0.045, 0.6, 0.27, 0.036 + 0.012 * Math.sin(X * 31)), sdEll(X, Y, 0.72, 0.36, 0.2 + 0.03 * Math.sin(Y * 23), 0.13));
    if (sG < 0 && X > 0.2) { emId = 900003; return mem(198 + grain * 0.7 - (vnoise(X / 0.09, Y / 0.09, 143) > 0.55 && dots(X, Y, pw, 0.03, 0.0075, false, 0.7, 144) ? 80 : 0), -sG, pw, 1); }
  }
  let x = X + 0.26 * (vnoise(X * 1.1, Y * 1.1, 1) - 0.5) + 0.1 * (vnoise(X * 4, Y * 4, 3) - 0.5) + 0.042 * (vnoise(X / 0.14, Y / 0.14, 26) - 0.5), y = Y + 0.26 * (vnoise(X * 1.1, Y * 1.1, 2) - 0.5) + 0.1 * (vnoise(X * 4, Y * 4, 4) - 0.5) + 0.042 * (vnoise(X / 0.14, Y / 0.14, 27) - 0.5); const mtD = pw > 0.008 ? 0.4 : 1;
  /* cell bodies, on a coarse grid: a nucleus with clumped chromatin and a nucleolus, cytoplasm with stacks of reticulum */
  { const c = 24, i = Math.floor(x / c), j = Math.floor(y / c); if (hash2(i, j, 71) < 0.3) { const cx = (i + 0.33 + 0.34 * hash2(i, j, 72)) * c, cy = (j + 0.33 + 0.34 * hash2(i, j, 73)) * c, R = 5 + 2.4 * hash2(i, j, 74); if (Math.hypot(cx, cy) > R + 9) { const d = Math.hypot(x - cx, y - cy) - R * (1 + 0.1 * (vnoise(x * 0.4, y * 0.4, 75) - 0.5)); if (d < 0) { emId = 800000 + i * 131 + j;
          const dn = Math.hypot(x - cx - 0.5, y - cy + 0.3) - R * (0.64 + 0.05 * vnoise(x * 0.9, y * 0.9, 80)); let v;
          if (dn < 0) { const nu = Math.hypot(x - cx - 1.1, y - cy - 0.4); v = nu < 0.8 ? 74 + 30 * vnoise(X / 0.12, Y / 0.12, 81) : 190 - (vnoise(X * 1.7, Y * 1.7, 76) > 0.64 || dn > -0.55 * vnoise(X * 2.5, Y * 2.5, 77) ? 62 : 0) - 16 * vnoise(X / 0.09, Y / 0.09, 82); v = mem(v + grain, -dn, pw, 1.6); }
          else { const er = Math.sin((x * 0.7 + y * 0.7) / 0.11 + 6 * vnoise(X * 1.2, Y * 1.2, 83)); v = 138 - (vnoise(X * 2.2, Y * 2.2, 78) > 0.6 && er > 0.2 ? 46 : 0) - (vnoise(X * 5.5, Y * 5.5, 79) > 0.72 ? 52 : 0) + (vnoise(X * 9, Y * 3, 84) > 0.74 ? 34 : 0) + 1.4 * grain; }
          return mem(v, -d, pw, 1.2); } } } }
  /* a capillary: an empty lumen, a thin wall of endothelium and its basement membrane */
  if (Math.abs(x + 21) < 3.7 && Math.abs(y + 13.5) < 3.7) { const dx = x + 21, dy = y + 13.5, d = Math.hypot(dx, dy) - 2.7 - 0.18 * Math.sin(Math.atan2(dy, dx) * 3 + 1); if (d < 0.62) { emId = 700001; const nb = Math.hypot(dx - 2.1, dy + 1.3) - 1.05; if (d < 0) return nb < 0 ? (nb > -0.06 ? 60 : 150 - (vnoise(X * 3, Y * 3, 85) > 0.55 ? 60 : 0) + grain) : 232 + grain * 0.3; return d < 0.05 ? 58 : d < 0.34 ? 132 + grain - (vnoise(X / 0.06, Y / 0.06, 86) > 0.78 ? 60 : 0) : d < 0.4 ? 86 : d < 0.56 ? 150 + grain : 60; } }
  /* large profiles */
  if (blobs(x, y, 0)) { const e = Math.min(-hit.sd, (hit.sd2 - hit.sd) * 0.5), k = hit.kind, id = hit.id, u = (x - hit.cx) * hit.co + (y - hit.cy) * hit.si, w = -(x - hit.cx) * hit.si + (y - hit.cy) * hit.co; emId = id; let v;
    if (k < 0.16) { const th = Math.min(hit.b * 0.32, 0.07 + 0.09 * idHash(id, 91)); if (e < th) return pw < 0.006 ? 32 + 12 * Math.sin(e / 0.0118 * TAU) : 30 + 10 * g2; if (e < th + 0.014) return 178; v = 180 + grain - (dots(X, Y, pw, 0.045, 0.0055, false, 0.75, 150) ? 40 : 0); const m = idHash(id, 92) < 0.5 ? mito(u - hit.a * 0.2, w, hit.b * 0.26, hit.b * 0.2, pw, 3) : -1; return m >= 0 ? m : mem(v, e - th - 0.014, pw, 1); }
    if (k < 0.34) { v = 200 + grain * 0.7 - 14 * vnoise(X / 0.2, Y / 0.2, 151) - (vnoise(X / 0.11, Y / 0.11, 143) > 0.6 && dots(X, Y, pw, 0.034, 0.0095, false, 0.7, 144) ? 96 : 0) - (Math.abs(Math.sin((u * 0.8 + w * 0.6) / 0.016)) < 0.2 && vnoise(X / 0.3, Y / 0.3, 152) > 0.62 ? 36 : 0); return mem(v, e, pw, mk1); }
    const mi = idHash(id, 93), m = mi < 0.7 ? mito(u - hit.a * (mi - 0.35), w - hit.b * 0.25 * (idHash(id, 94) - 0.5) * 2, Math.min(hit.a * 0.42, 0.34 + 0.2 * mi), Math.min(hit.b * 0.36, 0.1 + 0.08 * idHash(id, 95)), pw, 4 + (id & 7)) : -1;
    if (m >= 0 && e > 0.035) return m + grain * 0.5;
    v = 170 + 14 * (idHash(id, 97) - 0.5) + grain - 12 * vnoise(X / 0.03, Y / 0.03, 8) - (dots(X, Y, pw, 0.066, 0.011, true, 0.45, 130) ? 50 * mtD : 0) - (vnoise(X / 0.16, Y / 0.16, 153) > 0.74 && Math.abs(vnoise(X / 0.05, Y / 0.05, 154) - 0.5) < 0.04 ? 60 : 0);
    return mem(v, e, pw, mk1); }
  const s0 = hit.sd;
  /* middling profiles */
  if (blobs(x, y, 1)) { const e = Math.min(-hit.sd, (hit.sd2 - hit.sd) * 0.5), k = hit.kind, id = hit.id, u = (x - hit.cx) * hit.co + (y - hit.cy) * hit.si, w = -(x - hit.cx) * hit.si + (y - hit.cy) * hit.co; emId = id; let v;
    if (k < 0.36) { const pa = idHash(id, 98), m = hit.b > 0.15 && pa < 0.6 ? mito(u + hit.a * 0.3, w, hit.a * 0.34, hit.b * 0.42, pw, 5 + (id & 7)) : -1; if (m >= 0 && e > 0.03) return m + grain * 0.5;
      const bx = hit.cx, by = hit.cy; v = (e > 0.03 ? vesicles(X, Y, pw, 150, (cx, cy) => 0.2 + 0.85 * vnoise(cx / 0.16 + bx, cy / 0.16 + by, 66)) : 150) + grain; return mem(v, e, pw, mk1); }
    if (k < 0.72) { v = 176 + 12 * (idHash(id, 97) - 0.5) + grain - 20 * vnoise(X / 0.028, Y / 0.028, 8) - (hit.b > 0.2 && dots(X, Y, pw, 0.066, 0.011, true, 0.4, 130) ? 50 * mtD : 0); return mem(v, e, pw, mk1); }
    if (k < 0.87) { v = 202 + grain * 0.7 - (vnoise(X / 0.11, Y / 0.11, 143) > 0.62 && dots(X, Y, pw, 0.034, 0.0095, false, 0.7, 144) ? 96 : 0); return mem(v, e, pw, mk1); }
    v = 164 + grain - (dots(X, Y, pw, 0.045, 0.0055, false, 0.7, 150) ? 40 : 0); return mem(v, e, pw, mk1); }
  const psd = Math.max(hit.psd * clamp(1.25 - hit.sd / 0.045, 0, 1), 0), near = Math.min(hit.sd, s0);
  /* the ground of fine axons: every point belongs to the nearest site; neighbouring sites that share a leader are one process */
  const rg = emRg(x, y), sc = rg < 0.42 ? 0 : rg < 0.66 ? 1 : 2, E = EMS[sc], dl = 1.6 * Math.max(0.0085, pw * 0.6);
  if ((Math.abs(rg - 0.42) < 0.9 * dl || Math.abs(rg - 0.66) < 0.9 * dl) && (emScale(x + dl, y) !== sc || emScale(x - dl, y) !== sc || emScale(x, y + dl) !== sc || emScale(x, y - dl) !== sc)) { emId = 0; return 46; }
  let xv = x, yv = y; if (sc === 2) { const th = 0.6 + 2.2 * vnoise(x / 9, y / 9, 22), co = Math.cos(th), si = Math.sin(th); xv = (x * co + y * si) / 3.4; yv = -x * si + y * co; }
  const ci = Math.floor(xv / E), cj = Math.floor(yv / E), sx = (a, b) => (a + 0.1 + 0.8 * hash2(a, b, 14 + sc)) * E, sy = (a, b) => (b + 0.1 + 0.8 * hash2(a, b, 17 + sc)) * E; let d1 = 1e9, d2 = 1e9, d3 = 1e9, a1 = 0, b1 = 0, a2 = 0, b2 = 0, a3 = 0, b3 = 0;
  for (let b = cj - 1; b <= cj + 1; b++) for (let a = ci - 1; a <= ci + 1; a++) { const d = Math.hypot(xv - sx(a, b), yv - sy(a, b)); if (d < d1) { d3 = d2; a3 = a2; b3 = b2; d2 = d1; a2 = a1; b2 = b1; d1 = d; a1 = a; b1 = b; } else if (d < d2) { d3 = d2; a3 = a2; b3 = b2; d2 = d; a2 = a; b2 = b; } else if (d < d3) { d3 = d; a3 = a; b3 = b; } }
  const id1 = leader(a1, b1, sc), s1x = sx(a1, b1), s1y = sy(a1, b1), bis = (a, b, d) => (d * d - d1 * d1) / (2 * Math.hypot(sx(a, b) - s1x, sy(a, b) - s1y) + 1e-9); let edge = 9;
  if (leader(a2, b2, sc) !== id1) edge = bis(a2, b2, d2); if (leader(a3, b3, sc) !== id1) edge = Math.min(edge, bis(a3, b3, d3)); if (edge === 9) edge = d3 * 0.5; if (sc === 2) edge *= 1.4;
  emId = id1; const  fineM = pw < 0.006, gap = fineM ? 0.005 : 0, kind = idHash(id1, 91);
  let v = 154 + 62 * (idHash(id1, 97) - 0.5) + grain - 20 * vnoise(X / 0.022, Y / 0.022, 98);
  if (sc === 2) v -= Math.abs(Math.sin(yv / 0.021 + 4 * vnoise(xv * 2, yv * 9, 7))) < 0.16 ? 30 : 0; else if (dots(X, Y, pw, 0.055, 0.0105, true, 0.44, 130)) v -= 52 * mtD;
  if (fineM) { v -= 26 * (vnoise(X / 0.007, Y / 0.007, 155) > 0.66 + 0.2 * idHash(id1, 101) ? 1 : 0); const mh = idHash(id1, 102); if (mh < 0.16 && edge > 0.022) { const m = mito(xv - s1x, yv - s1y, 0.05 + 0.03 * mh / 0.16, 0.042, pw, 9); if (m >= 0) v = m + grain * 0.5; } }
  if (kind < 0.24 && sc !== 2 && edge > 0.026) v = vesicles(X, Y, pw, v, (cx, cy) => 0.15 + 0.8 * vnoise(cx / 0.12, cy / 0.12, 66));
  else if (kind > 0.9) v += 26;
  if (psd > 0) v = lerp(v, 52, psd * (0.55 + 0.45 * vnoise(X / 0.02, Y / 0.02, 87)));
  if (near < gap) return fineM ? 150 + grain * 0.6 : 46; if (edge < gap) return 152 + grain * 0.6 - 30 * (vnoise(X / 0.008, Y / 0.008, 90) > 0.7 ? 1 : 0);
  return mem(v, Math.min(edge, near) - gap, pw, mk1); }
const segColour = id => { if (id === 0) return null; if (id === 900002) return [84, 158, 255]; if (id === 900001) return [246, 204, 78]; if (id === 900003) return [150, 232, 150]; const h = idHash(id, 99) * 6, sct = 0.3 + 0.36 * idHash(id, 100), v = 246, c = v * sct, xm = c * (1 - Math.abs(h % 2 - 1)), m = v - c; const p = h < 1 ? [c, xm, 0] : h < 2 ? [xm, c, 0] : h < 3 ? [0, c, xm] : h < 4 ? [0, xm, c] : h < 5 ? [xm, 0, c] : [c, 0, xm]; return [p[0] + m, p[1] + m, p[2] + m]; };
/* a generator, so a bake can be spread over several frames. unit: world units per µm (1 for level 4, 1000 for level 5).
   What the microscope and the knife add is put in here: uneven staining, chatter, a score from the knife, specks of stain, shot noise. */
function* paintEM(Bk, unit, colour) { const n = Bk.px, g = Bk.c.getContext('2d'), im = g.createImageData(n, n), d = im.data, pw = Bk.w / n / unit; let cim = null, cd = null;
  if (colour) { cim = colour.getContext('2d').createImageData(n, n); cd = cim.data; }
  const nz = Math.max(pw, 0.0026), t0 = performance.now(); let t1 = t0;
  for (let j = 0; j < n; j++) { const y = (Bk.y0 + (j + 0.5) * Bk.w / n) / unit; for (let i = 0; i < n; i++) { const x = (Bk.x0 + (i + 0.5) * Bk.w / n) / unit; let v = emPix(x, y, pw);
      v = (v - 128) * (0.9 + 0.2 * vnoise(x / 7 + 3, y / 7, 160)) + 128 + 16 * (vnoise(x / 11, y / 11, 161) - 0.5) + (pw < 0.03 ? 3.2 * Math.sin((x * 0.966 + y * 0.259) / 0.12 * TAU) * (0.4 + vnoise(x / 3, y / 3, 162)) : 0);
      const sc = Math.abs(x * 0.906 - y * 0.423 - 5.3 - 0.02 * vnoise(x * 0.966 + y * 0.259, 0, 163)); if (sc < 0.02 + pw) v += 12 * (1 - sc / (0.02 + pw));
      if (hash2(Math.floor(x / 0.09), Math.floor(y / 0.09), 164) > 0.992 && Math.hypot(x / 0.09 % 1 - 0.5 + (x < 0), y / 0.09 % 1 - 0.5 + (y < 0)) < 0.18) v = 22;
      v = clamp(128 + (v - 128) * 1.12 + 26 * (vnoise(x / nz, y / nz, 165) - 0.5) + 20 * (hash2(i, j, 7) - 0.5), 0, 255); const o = 4 * (j * n + i); d[o] = v; d[o + 1] = v; d[o + 2] = v * 0.985; d[o + 3] = 255;
      if (cd) { const c = segColour(emId), k = 0.3 + 0.76 * v / 170; if (c) { cd[o] = Math.min(255, c[0] * k); cd[o + 1] = Math.min(255, c[1] * k); cd[o + 2] = Math.min(255, c[2] * k); } else { cd[o] = cd[o + 1] = cd[o + 2] = v; } cd[o + 3] = 255; } }
    if ((j & 3) === 3 && performance.now() - t1 > 7) { yield; t1 = performance.now(); } }
  g.putImageData(im, 0, 0); if (colour) colour.getContext('2d').putImageData(cim, 0, 0); }

/* ════ Level 6: membrane, vesicles and proteins in the style of a slice through a cryo-electron tomogram, in nanometres.
   Dense matter is dark. Contrast is low and the grain is coarse, because the specimen can bear only a small dose.
   On top of the tomogram, as in a published figure, a model of the channel is drawn as ribbons. ════ */
const MEM_Y = -1.1;                             /* mid-plane of the terminal's membrane; the origin is the channel's selectivity filter */
const POST_Y = MEM_Y + 20;
const memY = x => MEM_Y + 0.9 * Math.sin(x / 37 + 1) * M.smooth(Math.abs(x) / 16) + 0.5 * Math.sin(x / 11) * M.smooth((Math.abs(x) - 12) / 16);
const postY = x => POST_Y + 1.1 * Math.sin(x / 29 + 2) * M.smooth(Math.abs(x) / 14) + 0.5 * Math.sin(x / 9.3 + 1);
const VES = [[-33, MEM_Y - 23.2, 1], [52, MEM_Y - 23.6, 1], [-9, MEM_Y - 68, 0], [-92, MEM_Y - 62, 0], [101, MEM_Y - 71, 0], [40, MEM_Y - 104, 0], [-55, MEM_Y - 112, 0], [118, MEM_Y - 24.5, 1], [-121, MEM_Y - 27, 0]];
/* the helices of the two subunits shown, in ångströms; the picture of atoms uses the same ones */
const HELIX = []; for (const s of [-1, 1]) HELIX.push([s * 5.2, -33, s * 19, 9, 1 + s, s], [s * 16.5, 12.5, s * 8.2, -7.6, 2 - s, s], [s * 25, -30, s * 30.5, 15, 3 + s, s], [s * 8.6, 9.5, s * 15, 15.5, 0.5, s]);
function* paintMachine(Bk) { const g = world(Bk), s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, x1 = x0 + Bk.w, y1 = y0 + Bk.w, n = Bk.px;
  g.fillStyle = 'rgb(150,150,150)'; g.fillRect(x0, y0, Bk.w, Bk.w); g.lineCap = 'round'; g.lineJoin = 'round';
  const blob = (X, Y, R, a) => { const gr = g.createRadialGradient(X, Y, 0, X, Y, R); gr.addColorStop(0, 'rgba(20,20,20,' + a + ')'); gr.addColorStop(0.5, 'rgba(20,20,20,' + a * 0.6 + ')'); gr.addColorStop(1, 'rgba(20,20,20,0)'); g.fillStyle = gr; g.fillRect(X - R, Y - R, 2 * R, 2 * R); };
  /* a protein: a few lobes packed together, each a few nanometres across */
  const protein = (X, Y, size, a, sd) => { const k = 2 + ((hash2(sd, 1, 171) * 5) | 0), th = hash2(sd, 2, 171) * TAU, el = 0.4 + 1.2 * hash2(sd, 3, 171); for (let q = 0; q < k; q++) { const t = (q - (k - 1) / 2) * size * 0.5 * el, o = (hash2(sd, q, 172) - 0.5) * size * 0.7; blob(X + Math.cos(th) * t - Math.sin(th) * o, Y + Math.sin(th) * t + Math.cos(th) * o, size * (0.42 + 0.3 * hash2(sd, q, 173)), a); } };
  const inVes = (X, Y, m) => { for (const V of VES) if (Math.hypot(X - V[0], Y - V[1]) < 20.4 + m) return true; return false; };
  /* soluble proteins of the terminal, material of the cleft, the dense web under the far membrane, and filaments */
  const c = 5.4; let cnt = 0; for (let j = Math.floor(y0 / c) - 1; j <= y1 / c + 1; j++) { for (let i = Math.floor(x0 / c) - 1; i <= x1 / c + 1; i++) { const X = (i + hash2(i, j, 121)) * c, Y = (j + hash2(i, j, 122)) * c, h = hash2(i, j, 123), my = memY(X), py = postY(X);
      const cleft = Y > my + 3.4 && Y < py - 3.4, psd = Y > py + 3.5 && Y < py + 36 + 8 * vnoise(X / 20, 0, 124), deep = Y >= py + 36; if (Math.abs(Y - my) < 4.6 || Math.abs(Y - py) < 4.6 || Math.hypot(X, Y - MEM_Y + 8) < 7.5 || (Y < my && inVes(X, Y, 3))) continue;
      if (h > (cleft ? 0.36 : psd ? 0.97 : deep ? 0.4 : 0.42 + 0.5 * vnoise(X / 26, Y / 26, 125))) continue; protein(X, Y, (psd ? 3.3 : cleft ? 1.8 : 2.3) + 2.8 * hash2(i, j, 126) * hash2(i, j, 127), psd ? 0.5 : cleft ? 0.4 : 0.46, i * 131 + j); } if (++cnt % 30 === 0) yield; }
  g.strokeStyle = 'rgba(24,24,24,0.3)';
  for (let i = Math.floor((x0 - 60) / 23); i <= (x1 + 60) / 23; i++) for (let k = 0; k < 3; k++) { const X = (i + hash2(i, k, 128)) * 23, Y = postY(X) + 6 + 70 * hash2(i, k, 129), a = (hash2(i, k, 130) - 0.5) * 2.4, L = 30 + 50 * hash2(i, k, 131); g.lineWidth = 2.6 + 4 * (k === 0); g.globalAlpha = k === 0 ? 0.5 : 1; g.beginPath(); g.moveTo(X, Y); g.quadraticCurveTo(X + Math.cos(a) * L * 0.5 + 6 * (hash2(i, k, 132) - 0.5), Y + Math.sin(a) * L * 0.5, X + Math.cos(a) * L, Y + Math.sin(a) * L); g.stroke(); }
  g.globalAlpha = 1; g.lineWidth = 1.5; g.strokeStyle = 'rgba(24,24,24,0.36)'; g.beginPath(); for (let i = Math.floor(x0 / 5.5); i <= x1 / 5.5; i++) { const X = (i + 0.6 * hash2(i, 0, 133)) * 5.5; if (hash2(i, 0, 134) < 0.55 && Math.abs(X) > 6) { g.moveTo(X, memY(X) + 2.8); g.bezierCurveTo(X + 2.4 * (hash2(i, 1, 135) - 0.5), MEM_Y + 8, X + 3.4 * (hash2(i, 2, 135) - 0.5), MEM_Y + 13, X + 2 * (hash2(i, 0, 135) - 0.5), postY(X) - 2.8); } } g.stroke();
  /* a bilayer: two thin dark leaflets of head groups 3.8 nm apart, with a paler core of tails */
  const leaflets = path => { g.strokeStyle = 'rgba(176,176,176,0.5)'; g.lineWidth = 2.6; path(0); g.stroke(); for (const o of [-1.9, 1.9]) { g.strokeStyle = 'rgba(22,22,22,0.62)'; g.lineWidth = 1.25; path(o); g.stroke(); } };
  const st = Math.max(0.5, 1.5 / s); for (const f of [memY, postY]) leaflets(o => { g.beginPath(); for (let X = x0 - 2; X <= x1 + 2; X += st) g.lineTo(X, f(X) + o); });
  for (const [X, Y, docked] of VES) { if (X + 40 < x0 || X - 40 > x1 || Y + 40 < y0 || Y - 40 > y1) continue; const sd = Math.round(X * 7 + Y); blob(X, Y, 19, 0.1);
    leaflets(o => { g.beginPath(); for (let a = 0; a <= 64; a++) { const t = a / 64 * TAU, R = 18.4 + o + 0.5 * Math.sin(3 * t + sd) + 0.3 * Math.sin(7 * t + sd); g.lineTo(X + R * Math.cos(t), Y + R * 0.985 * Math.sin(t)); } g.closePath(); });
    for (let k = 0; k < 22; k++) { const a = k / 22 * TAU + 0.3 * hash2(k, sd, 141), R = 21.6 + 2.6 * hash2(k, sd, 142); if (docked && Math.sin(a) > 0.84) continue; if (hash2(k, sd, 144) < 0.8) protein(X + R * Math.cos(a), Y + R * Math.sin(a), 1.6 + 1.6 * hash2(k, sd, 143), 0.42, sd + k); }
    const a = -2.2 + (sd % 3); g.strokeStyle = 'rgba(22,22,22,0.5)'; g.lineWidth = 2.6; g.beginPath(); g.moveTo(X + 20 * Math.cos(a), Y + 20 * Math.sin(a)); g.lineTo(X + 27 * Math.cos(a), Y + 27 * Math.sin(a)); g.stroke(); blob(X + 31.5 * Math.cos(a), Y + 31.5 * Math.sin(a), 6.2, 0.6); blob(X + 20.6 * Math.cos(a), Y + 20.6 * Math.sin(a), 3.6, 0.4);       /* the proton pump: one or two per vesicle (Takamori et al. 2006) */
    if (docked) { g.strokeStyle = 'rgba(22,22,22,0.5)'; g.lineWidth = 1.3; g.beginPath(); for (const o of [-6, -2, 3, 7]) { g.moveTo(X + o * 1.5, Y + 18.6); g.quadraticCurveTo(X + o * 1.3 + 0.8, (Y + 19 + memY(X)) / 2, X + o, memY(X + o) - 2.4); } g.stroke(); } }
  /* receptors of the far membrane: two tiers of lobes reaching into the cleft, a waist in the membrane, a tail below */
  for (let i = Math.floor((x0 - 20) / 17); i <= (x1 + 20) / 17; i++) { if (hash2(i, 0, 161) < 0.25) continue; const X = (i + 0.2 + 0.6 * hash2(i, 0, 162)) * 17, py = postY(X); for (const o of [-1.9, 1.9]) { blob(X + o * 1.25, py - 10.4, 2.6, 0.5); blob(X + o, py - 6.2, 2.6, 0.56); } blob(X, py, 3.4, 0.5); blob(X + 0.6, py + 4.6, 3.4, 0.34); }
  /* the channel as the tomogram would show it: a dense plug in the membrane with a knob below */
  for (const H of HELIX) { g.strokeStyle = 'rgba(22,22,22,0.36)'; g.lineWidth = 1.5; g.beginPath(); g.moveTo(H[0] / 10, H[1] / 10); g.lineTo(H[2] / 10, H[3] / 10); g.stroke(); } blob(0, MEM_Y - 8.4, 4.4, 0.44); blob(-2.2, MEM_Y - 6.2, 2.4, 0.3); blob(2.2, MEM_Y - 6.2, 2.4, 0.3);
  yield;
  /* the instrument: limited resolution, low contrast, and noise at every scale down to the pixel of the detector */
  { const c0 = mk(n, n); c0.getContext('2d').drawImage(Bk.c, 0, 0); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(soft(c0, clamp(0.2 * s, 0.6, 7)), 0, 0); g.restore(); yield; }
  const im = g.getImageData(0, 0, n, n), d = im.data, q = Bk.w / n; let t1 = performance.now();
  for (let j = 0; j < n; j++) { const y = y0 + (j + 0.5) * q; for (let i = 0; i < n; i++) { const x = x0 + (i + 0.5) * q, o = 4 * (j * n + i);
      const v = clamp(150 + (d[o] - 150) * 0.85 + 30 * (vnoise(x / 0.19, y / 0.19, 186) - 0.5) + 42 * (vnoise(x / 0.46, y / 0.46, 181) - 0.5) + 24 * (vnoise(x / 1.5 + 9, y / 1.5, 182) - 0.5) + 13 * (vnoise(x / 6, y / 6, 183) - 0.5) + 14 * (vnoise(x / 70, y / 70, 184) - 0.5) + 12 * (hash2(i, j, 185) - 0.5), 0, 255); d[o] = v * 0.985; d[o + 1] = v; d[o + 2] = v * 1.01; }
    if ((j & 7) === 7 && performance.now() - t1 > 7) { yield; t1 = performance.now(); } }
  g.putImageData(im, 0, 0);
  if (s > 9) ribbons(g, s); }
/* the fitted model, as in a figure where an atomic model is docked into the tomogram: helices as lit cylinders, loops as cords */
function ribbons(g, s) { const CH = { '-1': [92, 178, 232], '1': [240, 178, 92] }; g.lineJoin = 'round'; g.lineCap = 'round';
  const tube = (P, col, w) => { for (const [k, lw, o] of [[0.34, w + 1.6 / s, 0], [0.82, w, 0], [1.08, w * 0.62, -w * 0.1], [1.5, w * 0.2, -w * 0.22]]) { g.strokeStyle = rgba(col.map(v => Math.min(255, v * k) | 0), 1); g.lineWidth = lw; g.beginPath(); g.moveTo(P[0][0] + o, P[0][1] + o); for (let i = 1; i < P.length; i++) g.lineTo(P[i][0] + o, P[i][1] + o); g.stroke(); } };
  for (const sg of [-1, 1]) { const col = CH[sg]; tube(spline([[sg * 1.9, 0.9], [sg * 2.6, 0.1], [sg * 2.9, -1.6], [sg * 2.5, -3.0]], false, 8), col.map(v => v * 0.6 | 0), 0.13); }
  for (const hi of [2, 0, 3, 1]) for (const H of HELIX.filter((h, k) => k % 4 === hi)) { const ax = H[0] / 10, ay = H[1] / 10, bx = H[2] / 10, by = H[3] / 10, L = Math.hypot(bx - ax, by - ay), ux = (bx - ax) / L, uy = (by - ay) / L, nx = -uy, ny = ux, col = CH[H[5]], w = hi === 3 ? 0.36 : 0.5;
    tube([[ax, ay], [bx, by]], col, w); g.save(); g.lineCap = 'butt'; g.strokeStyle = rgba(col.map(v => v * 0.6 | 0), 0.34); g.lineWidth = 0.06; g.beginPath(); for (let t = 0.2; t < L - 0.1; t += 0.27) { g.moveTo(ax + ux * t + nx * w * 0.47, ay + uy * t + ny * w * 0.47); g.lineTo(ax + ux * (t + 0.2) - nx * w * 0.47, ay + uy * (t + 0.2) - ny * w * 0.47); } g.stroke(); g.restore(); }
  for (const sg of [-1, 1]) { const col = CH[sg].map(v => Math.min(255, v * 1.1) | 0);
    tube(spline([[sg * 1.5, 1.55], [sg * 2.0, 2.15], [sg * 2.6, 1.9], [sg * 3.05, 1.5]], false, 8), col, 0.17); tube(spline([[sg * 0.86, 0.95], [sg * 0.66, 1.5], [sg * 1.1, 1.9], [sg * 1.5, 1.55]], false, 8), col, 0.17);
    tube(spline([[sg * 0.82, -0.76], [sg * 0.46, -0.62], [sg * 0.42, 0], [sg * 0.44, 0.62], [sg * 0.86, 0.95]], false, 8), col, 0.17); } }

/* ════ Level 7: a potassium channel's pore as a cryo-EM figure: a cage of density with atoms fitted into it, in ångströms.
   The atoms have a depth as well as a place on the page, and are drawn from the back forwards: far ones dimmer and a
   little out of focus. The cage is the surface of constant density, drawn as its outlines in three sets of planes. ════ */
const PORE = (() => { const atoms = M.FILTER.atoms.map((a, i) => Object.assign({ core: true, z: 0.45 * ((i % 3) - 1) }, a)), bonds = M.FILTER.bonds.map(b => b.slice());
  const helix = (xa, ya, xb, yb, sd) => { const L = Math.hypot(xb - xa, yb - ya), n = Math.round(L / 1.5), ux = (xb - xa) / L, uy = (yb - ya) / L, nx = -uy, ny = ux; let prev = -1;
    for (let k = 0; k <= n; k++) { const t = k * 1.5, ph = k * 1.745 + sd, w = Math.sin(ph) * 2.1, zc = Math.cos(ph) * 2.1, X = xa + ux * t + nx * w, Y = ya + uy * t + ny * w, el = k % 3 === 0 ? 'N' : 'C', i = atoms.push({ el, x: X, y: Y, z: zc }) - 1; if (prev >= 0) bonds.push([prev, i]); prev = i;
      if (k % 3 === 2) { const o = atoms.push({ el: 'O', x: X + nx * Math.sign(w || 1) * 1.25, y: Y + ny * Math.sign(w || 1) * 1.25, z: zc * 1.2 }) - 1; bonds.push([i, o]); }
      if (k % 3 === 1 && hash2(k, Math.round(xa * 7), 171) < 0.75) { const sgn = Math.sign(w || 1); let p = i; const len = 1 + ((hash2(k, Math.round(xa), 172) * 3) | 0); for (let q = 1; q <= len; q++) { const a = atoms.push({ el: q === len && hash2(k, q, 173) < 0.3 ? (hash2(k, q, 174) < 0.5 ? 'O' : 'N') : 'C', x: X + nx * sgn * 1.5 * q + ux * 0.5 * (q % 2), y: Y + ny * sgn * 1.5 * q + uy * 0.5 * (q % 2), z: zc * (1 + 0.35 * q) + 0.5 * (q % 2) }) - 1; bonds.push([p, a]); p = a; } } } };
  for (const H of HELIX) helix(H[0], H[1], H[2], H[3], H[4]);
  return { atoms, bonds }; })();
const ELC = { C: [226, 208, 140], N: [112, 154, 255], O: [255, 100, 84], K: [190, 150, 255] }, ELW = { C: 6, N: 7, O: 8, K: 19 };
const proj = (x, y, z) => { const p = 1 + z / 110; return [(x + 0.2 * z) * p, (y - 0.14 * z) * p, p]; };
function* paintAtoms(Bk) { const g = world(Bk), s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, W = Bk.w, cxw = x0 + W / 2, cyw = y0 + W / 2;
  { const gr = g.createRadialGradient(cxw, cyw, 0, cxw, cyw, W * 0.75); gr.addColorStop(0, '#0b1420'); gr.addColorStop(0.6, '#060a11'); gr.addColorStop(1, '#020304'); g.fillStyle = gr; g.fillRect(x0, y0, W, W); }
  /* density on a grid: a Gaussian for each atom, heavier atoms denser */
  const h = W > 50 ? 0.5 : 0.3, m = 5, gx0 = x0 - m, gy0 = y0 - m, gz0 = -7, nx = Math.ceil((W + 2 * m) / h) + 1, ny = nx, nz = Math.ceil(14 / h) + 1, F = new Float32Array(nx * ny * nz), sg = 0.62, R = Math.ceil(2.4 * sg / h), at = (i, j, k) => F[(k * ny + j) * nx + i];
  for (const a of PORE.atoms) { const ci = Math.round((a.x - gx0) / h), cj = Math.round((a.y - gy0) / h), ck = Math.round((a.z - gz0) / h); if (ci < -R || cj < -R || ci >= nx + R || cj >= ny + R) continue;
    for (let k = Math.max(0, ck - R); k <= Math.min(nz - 1, ck + R); k++) for (let j = Math.max(0, cj - R); j <= Math.min(ny - 1, cj + R); j++) for (let i = Math.max(0, ci - R); i <= Math.min(nx - 1, ci + R); i++) { const dx = gx0 + i * h - a.x, dy = gy0 + j * h - a.y, dz = gz0 + k * h - a.z; F[(k * ny + j) * nx + i] += ELW[a.el] / 6 * Math.exp(-(dx * dx + dy * dy + dz * dz) / (2 * sg * sg)); } }
  yield;
  /* outlines of the surface in planes one ångström apart, facing each of the three axes; sorted into far and near */
  const iso = 0.5, every = Math.max(1, Math.round((W > 50 ? 1.5 : 1) / h)), far = [[], [], []], nearS = [[], [], []];
  const seg = (ax, bx, ay, by, az, bz) => { const zm = (az + bz) / 2, P = proj(ax, ay, az), Q = proj(bx, by, bz), b = zm < -2.6 ? 0 : zm < -0.3 ? 1 : 2; (zm < 0.25 ? far : nearS)[zm < 0.25 ? b : zm < 2.4 ? 0 : 1].push(P[0], P[1], Q[0], Q[1]); };
  const march = (n1, n2, val, emit) => { for (let b = 0; b < n2 - 1; b++) for (let a = 0; a < n1 - 1; a++) { const A = val(a, b) - iso, B = val(a + 1, b) - iso, C = val(a + 1, b + 1) - iso, D = val(a, b + 1) - iso, mm = (A > 0) | (B > 0) << 1 | (C > 0) << 2 | (D > 0) << 3; if (mm === 0 || mm === 15) continue;
      const T = [a + A / (A - B), b], Rr = [a + 1, b + B / (B - C)], Bt = [a + D / (D - C), b + 1], Lf = [a, b + A / (A - D)];
      switch (mm) { case 1: case 14: emit(Lf, T); break; case 2: case 13: emit(T, Rr); break; case 4: case 11: emit(Rr, Bt); break; case 8: case 7: emit(Bt, Lf); break; case 3: case 12: emit(Lf, Rr); break; case 6: case 9: emit(T, Bt); break; case 5: emit(Lf, T); emit(Rr, Bt); break; case 10: emit(T, Rr); emit(Bt, Lf); break; } } };
  for (let k = 0; k < nz; k += every) { const z = gz0 + k * h; march(nx, ny, (i, j) => at(i, j, k), (p, q) => seg(gx0 + p[0] * h, gx0 + q[0] * h, gy0 + p[1] * h, gy0 + q[1] * h, z, z)); } yield;
  for (let j = 0; j < ny; j += every) { const y = gy0 + j * h; march(nx, nz, (i, k) => at(i, j, k), (p, q) => seg(gx0 + p[0] * h, gx0 + q[0] * h, y, y, gz0 + p[1] * h, gz0 + q[1] * h)); } yield;
  for (let i = 0; i < nx; i += every) { const x = gx0 + i * h; march(ny, nz, (j, k) => at(i, j, k), (p, q) => seg(x, x, gy0 + p[0] * h, gy0 + q[0] * h, gz0 + p[1] * h, gz0 + q[1] * h)); } yield;
  const mesh = (gg, S, col, a, lw) => { gg.strokeStyle = rgba(col, a); gg.lineWidth = lw; gg.beginPath(); for (let i = 0; i < S.length; i += 4) { gg.moveTo(S[i], S[i + 1]); gg.lineTo(S[i + 2], S[i + 3]); } gg.stroke(); };
  /* the fitted model: sticks and balls, lit from the upper left, sorted by depth */
  const items = []; for (const a of PORE.atoms) items.push({ z: a.z, a });
  for (const [i, j] of PORE.bonds) { const a = PORE.atoms[i], b = PORE.atoms[j], mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, mz = (a.z + b.z) / 2; items.push({ z: (a.z + mz) / 2 - 0.01, a, b: { x: mx, y: my, z: mz } }, { z: (b.z + mz) / 2 - 0.01, a: b, b: { x: mx, y: my, z: mz } }); }
  items.sort((p, q) => p.z - q.z);
  const fog = (c, z, k) => { const f = clamp((2.5 - z) / 11, 0, 0.72); return [lerp(c[0] * k, 14, f) | 0, lerp(c[1] * k, 22, f) | 0, lerp(c[2] * k, 34, f) | 0]; };
  const drawItem = (gg, it) => { const a = it.a, P = proj(a.x, a.y, a.z), c = ELC[a.el];
    if (it.b) { const Q = proj(it.b.x, it.b.y, it.b.z), w = Math.max(0.3 * P[2], 2 / s), dx = Q[0] - P[0], dy = Q[1] - P[1], L = Math.hypot(dx, dy) || 1, ox = -dy / L * w / 2, oy = dx / L * w / 2, gr = gg.createLinearGradient(P[0] - ox, P[1] - oy, P[0] + ox, P[1] + oy);
      gr.addColorStop(0, rgba(fog(c, a.z, ox + oy < 0 ? 0.95 : 0.3), 1)); gr.addColorStop(0.35, rgba(fog(c, a.z, 1.12), 1)); gr.addColorStop(0.62, rgba(fog(c, a.z, 0.8), 1)); gr.addColorStop(1, rgba(fog(c, a.z, ox + oy < 0 ? 0.3 : 0.95), 1)); gg.strokeStyle = gr; gg.lineWidth = w; gg.lineCap = 'butt'; gg.beginPath(); gg.moveTo(P[0], P[1]); gg.lineTo(Q[0] + dx / L * 0.02, Q[1] + dy / L * 0.02); gg.stroke(); return; }
    const r = Math.max((a.el === 'C' ? 0.36 : a.core && a.el === 'O' ? 0.46 : 0.4) * P[2], 2.2 / s), gr = gg.createRadialGradient(P[0] - r * 0.36, P[1] - r * 0.4, r * 0.04, P[0] - r * 0.1, P[1] - r * 0.1, r * 1.1);
    gr.addColorStop(0, rgba(fog([255, 255, 255], a.z, 1), 1)); gr.addColorStop(0.16, rgba(fog(c.map(v => Math.min(255, v * 1.15 + 20)), a.z, 1), 1)); gr.addColorStop(0.55, rgba(fog(c, a.z, 0.86), 1)); gr.addColorStop(1, rgba(fog(c, a.z, 0.3), 1)); gg.fillStyle = gr; gg.beginPath(); gg.arc(P[0], P[1], r, 0, TAU); gg.fill(); };
  const lw = Math.max(0.03, 0.8 / s), MC = [92, 158, 224];
  /* far half, slightly out of focus */
  { const t = tmpFor(Bk.px), tg = worldOn(t, Bk); tg.lineCap = 'round'; mesh(tg, far[0], MC, 0.3, lw); for (const it of items) if (it.z < -2.2) drawItem(tg, it); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.drawImage(soft(t, clamp(0.085 * s, 0.6, 5)), 0, 0); g.restore(); yield; }
  g.lineCap = 'round'; mesh(g, far[1], MC, 0.42, lw); for (const it of items) if (it.z >= -2.2 && it.z < 0.25) drawItem(g, it); mesh(g, far[2], MC, 0.5, lw); yield;
  for (const it of items) if (it.z >= 0.25) drawItem(g, it);
  g.lineCap = 'round'; mesh(g, nearS[0], [120, 186, 246], 0.62, lw); mesh(g, nearS[1], [150, 206, 255], 0.5, lw * 1.25);
  /* a soft glow where the cage is dense, as in a rendered figure */
  { const c = mk(Bk.px, Bk.px); c.getContext('2d').drawImage(Bk.c, 0, 0); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'lighter'; g.globalAlpha = 0.22; g.drawImage(soft(c, 7), 0, 0); g.restore(); } }

/* ════ Baking ════ */
/* several pictures of each level, each finer and narrower than the last, so that the view stays sharp as it closes in.
   The one that fills the stage at the waypoint has the most pixels. A small quick one is made first, as a preview. */
const tierW = i => { const L = M.LEVELS[i], wide = L.top / U[i] * ASP * 1.03, wp0 = L.fov / U[i] * ASP * 1.3, fine = L.bot / U[i] * ASP * 1.6, out = [], em = i === 3 || i === 4, top = em ? Math.round(BKW * 0.875) : BKW;
  let w = wp0; if (wide > wp0 * 1.25) out.push([wide, BK]); else w = wide; out.push([w, top]);
  if (w / fine > 1.45) { const n = Math.ceil(Math.log(w / fine) / Math.log(2.9)); for (let k = 1; k <= n; k++) out.push([w * Math.pow(fine / w, k / n), BK]); } return out; };
const BAKES = [];                /* BAKES[i] = { tiers: [{ c, col, w, x0, y0, s, px }], done } */
const newBake = (w, px) => ({ c: mk(px, px), col: null, w, x0: -w / 2, y0: -w / 2, s: px / w, px });
const world = Bk => { const g = Bk.c.getContext('2d'); g.setTransform(Bk.s, 0, 0, Bk.s, -Bk.x0 * Bk.s, -Bk.y0 * Bk.s); return g; };
const GRAINT = (() => { const c = mk(256, 256), g = c.getContext('2d'), im = g.createImageData(256, 256), q = M.rng(6); for (let i = 0; i < 65536; i++) { const v = clamp(128 + (q() + q() + q() - 1.5) * 150, 0, 255); im.data[4 * i] = im.data[4 * i + 1] = im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; } g.putImageData(im, 0, 0); return c; })();
function feather(c) { const g = c.getContext('2d'), n = c.width, m = 0.07; g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'destination-in';
  for (const hz of [true, false]) { const gr = hz ? g.createLinearGradient(0, 0, n, 0) : g.createLinearGradient(0, 0, 0, n); gr.addColorStop(0, 'rgba(0,0,0,0)'); gr.addColorStop(m, 'rgba(0,0,0,1)'); gr.addColorStop(1 - m, 'rgba(0,0,0,1)'); gr.addColorStop(1, 'rgba(0,0,0,0)'); g.fillStyle = gr; g.fillRect(0, 0, n, n); }
  g.globalCompositeOperation = 'source-over'; }
/* noise that belongs to the detector: baked in at the picture's own pixel size, and stronger where the picture is brighter */
function shotNoise(Bk, amount) { const g = Bk.c.getContext('2d'); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'overlay'; g.globalAlpha = amount; g.fillStyle = g.createPattern(GRAINT, 'repeat'); g.fillRect(0, 0, Bk.px, Bk.px); g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; }
const PAINT = [null, function* (Bk) { yield* paintCortex(Bk); shotNoise(Bk, 0.34); }, function* (Bk) { yield* paintNeuron(Bk); shotNoise(Bk, 0.3); }, Bk => paintEM(Bk, 1, Bk.col), Bk => paintEM(Bk, 1000, null), paintMachine, paintAtoms];
function* bakeLevel(i) { const fresh = !BAKES[i] || !BAKES[i].tiers.length, out = { tiers: [], done: false }, tw = tierW(i), widest = Math.max.apply(null, tw.map(t => t[0]));
  const one = function* (w, px) { const Bk = newBake(w, px); if (i === 3) Bk.col = mk(px, px); yield* PAINT[i](Bk); if (w < widest) { feather(Bk.c); if (Bk.col) feather(Bk.col); } return Bk; };
  if (fresh) { BAKES[i] = out; const pv = yield* one(widest, Math.round(BK / 3.2)); pv.preview = true; out.tiers.push(pv); yield; }
  for (const [w, px] of tw.slice().sort((a, b) => (b[1] - a[1]) || (b[0] - a[0]))) { const Bk = yield* one(w, px); if (w === widest) out.tiers = out.tiers.filter(t => !t.preview); out.tiers.push(Bk); out.tiers.sort((a, b) => b.w - a.w); yield; }
  out.done = true; BAKES[i] = out; }
const JOBS = []; let job = null, bakesDone = false;
const wanted = new Set();
function request(i, urgent) { if (i < 1 || i >= NL || (BAKES[i] && BAKES[i].done) || (job && job.i === i)) return; const k = JOBS.indexOf(i); if (k >= 0) { if (urgent && k > 0) { JOBS.splice(k, 1); JOBS.unshift(i); } return; } if (wanted.has(i)) return; wanted.add(i); if (urgent) JOBS.unshift(i); else JOBS.push(i); }
function pump(budget) { const t0 = performance.now(); let did = false; while (performance.now() - t0 < budget) { if (!job) { if (!JOBS.length) { bakesDone = true; break; } const i = JOBS.shift(); job = { i, it: bakeLevel(i) }; bakesDone = false; } did = true; if (job.it.next().done) { wanted.delete(job.i); job = null; } } return did; }
function rebake(i) { if (job && job.i === i) job = null; wanted.delete(i); const k = JOBS.indexOf(i); if (k >= 0) JOBS.splice(k, 1); wanted.add(i); JOBS.unshift(i); bakesDone = false; }
