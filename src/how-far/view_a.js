/* VIEW, part 2: every scale. The camera, the baking of pictures on approach, and the twenty-one pictures that the brain's
   seven did not have. Every picture is generated here from rules; none is data and none is a photograph. */
const st = { z: EV.LV(EV.HOME).z, fly: null, running: !reduce, level: -99, recon: 0.62, site: M.HOME_SITE, nsyn: 8, nmda: true, p: 0.3, nrel: 5, V: -20, nch: 1, auto5: false, touched: false, vox: 1, bare: false,
  zq: 1, h0: 0, Ealpha: 5, Q: 1, cand: 0, mass: -7.66, flash: null, evSeed: 1, born: Date.now() };
let W = 1280, H = 960, DPR = 1, dirty = true, rigDirty = true;
let raf = 0, visible = true, lastReadout = 0, lastSay = 0, lastDraw = 0, lastMaths = 0, mathsDue = false;
function resize() { const r = cv.getBoundingClientRect(); if (!r.width) return; DPR = Math.min(window.devicePixelRatio || 1, r.width < 700 ? 1.5 : 2); const w = Math.round(r.width * DPR), h = Math.round(r.height * DPR); if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; } W = w; H = h; dirty = true; }
const K = EV.K, IDS = EV.LEVELS.map(L => L.id), unit = i => EV.LV(i).unit, smooth = EV.smooth;
const fovNow = () => EV.fovOf(st.z);
const gaugeW = () => (SMALL ? 62 : 104) * DPR;
const gaugeY = z => lerp(40 * DPR, H - 26 * DPR, z);
const CXs = () => (W - gaugeW()) / 2;
const NOW_MS = Date.now();                       /* the date used for the planets and the Voyagers: the day the page is opened */
const EARTH_AT = EV.planetAt(EV.PLANETS[2], NOW_MS);

/* ── more helpers ── */
const hash3 = (ix, iy, iz, s) => hash2(ix + Math.imul(iz, 1619), iy + Math.imul(iz, 31337), s + iz);
const vnoise3 = (x, y, z, s) => { const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z), fx = x - ix, fy = y - iy, fz = z - iz, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), w = fz * fz * (3 - 2 * fz);
  const a = lerp(lerp(hash3(ix, iy, iz, s), hash3(ix + 1, iy, iz, s), u), lerp(hash3(ix, iy + 1, iz, s), hash3(ix + 1, iy + 1, iz, s), u), v), b = lerp(lerp(hash3(ix, iy, iz + 1, s), hash3(ix + 1, iy, iz + 1, s), u), lerp(hash3(ix, iy + 1, iz + 1, s), hash3(ix + 1, iy + 1, iz + 1, s), u), v); return lerp(a, b, w); };
const fbm = (x, y, s, o = 4) => { let v = 0, a = 0.5, f = 1, n = 0; for (let k = 0; k < o; k++) { v += a * vnoise(x * f, y * f, s + k); n += a; a *= 0.5; f *= 2.03; } return v / n; };
const fbm3 = (x, y, z, s, o = 4) => { let v = 0, a = 0.5, f = 1, n = 0; for (let k = 0; k < o; k++) { v += a * vnoise3(x * f, y * f, z * f, s + k); n += a; a *= 0.5; f *= 2.03; } return v / n; };
const ss = (a, b, x) => smooth((x - a) / (b - a));
const SPRITES = {};
const sprite = c => { const k = c.join(','); if (SPRITES[k]) return SPRITES[k]; const s = mk(64, 64), g = s.getContext('2d'), gr = g.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(255,255,255,1)'); gr.addColorStop(0.12, rgba(c, 0.95)); gr.addColorStop(0.4, rgba(c, 0.28)); gr.addColorStop(1, rgba(c, 0)); g.fillStyle = gr; g.fillRect(0, 0, 64, 64); return (SPRITES[k] = s); };
const dot = (g, x, y, r, c, a = 1) => { g.globalAlpha = a; g.drawImage(sprite(c), x - r, y - r, 2 * r, 2 * r); };
const fillAll = (Bk, col) => { const g = Bk.c.getContext('2d'); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'source-over'; g.globalAlpha = 1; g.fillStyle = col; g.fillRect(0, 0, Bk.px, Bk.px); g.restore(); };
const RGBS = {}, rgbs = c => RGBS[c] || (RGBS[c] = 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')');
/* a hard point of light, h world units to a side from its centre */
const pt = (g, x, y, h, c, a = 1) => { g.globalAlpha = a; g.fillStyle = rgbs(c); g.fillRect(x - h, y - h, 2 * h, 2 * h); };
/* run fn(x, y, out) for every pixel of a bake, a strip at a time */
function* pixels(Bk, fn, strip = 48) { const n = Bk.px, g = Bk.c.getContext('2d'), im = g.createImageData(n, n), d = im.data, q = Bk.w / n, out = [0, 0, 0, 255];
  for (let j = 0; j < n; j++) { const y = Bk.y0 + (j + 0.5) * q; for (let i = 0; i < n; i++) { out[3] = 255; fn(Bk.x0 + (i + 0.5) * q, y, out, q); const o = 4 * (j * n + i); d[o] = out[0]; d[o + 1] = out[1]; d[o + 2] = out[2]; d[o + 3] = out[3]; } if (j % strip === strip - 1) yield; }
  g.putImageData(im, 0, 0); }
const bloom = (Bk, rad, a) => { const g = Bk.c.getContext('2d'), b = soft(Bk.c, rad); g.save(); g.setTransform(1, 0, 0, 1, 0, 0); g.globalCompositeOperation = 'lighter'; g.globalAlpha = a; g.drawImage(b, 0, 0); g.restore(); };
/* stars at infinity: they do not move as the frame changes, so one tile serves every level between the Earth and the nearest stars */
const STARF = (() => { const c = mk(512, 512), g = c.getContext('2d'), r = M.rng(404); g.fillStyle = '#020305'; g.fillRect(0, 0, 512, 512);
  for (let i = 0; i < 1100; i++) { const x = r() * 512, y = r() * 512, m = Math.pow(r(), 7), t = r(), col = t < 0.2 ? [255, 214, 170] : t < 0.75 ? [236, 240, 255] : [190, 210, 255]; g.globalAlpha = 0.16 + 0.8 * m; g.fillStyle = rgba(col, 1); const s = m > 0.55 ? 1.5 : 1; g.fillRect(x, y, s, s); if (m > 0.72) dot(g, x, y, 2.5 + 3.5 * m, col, 0.55); }
  g.globalAlpha = 1; return c; })();
let STARP = null;
function starfield(a) { if (!STARP) STARP = ctx.createPattern(STARF, 'repeat'); ctx.save(); ctx.globalAlpha = a; ctx.scale(DPR, DPR); ctx.fillStyle = STARP; ctx.fillRect(0, 0, W / DPR, H / DPR); ctx.restore(); }

/* ════ Baking: on approach only ════ */
const BAKES = {};                /* BAKES[id] = { tiers: [{ c, col, w, x0, y0, s, px }], done } */
const newBake = (w, px, cx = 0, cy = 0) => ({ c: mk(px, px), col: null, w, x0: cx - w / 2, y0: cy - w / 2, s: px / w, px });
const VC = {};                   /* per level: c0 (centre at the waypoint), an (centre when closing in on the next level), paint, live, maxTiers */
const MRI_AN = [10.5, -82.6];
const lg = Math.log10, CUP = 600;
const cup = fov => M.smooth((lg(fov) - lg(1.7e-3)) / (lg(3.2e-3) - lg(1.7e-3)));
const ZERO = [0, 0];
const centre = (i, fov) => { if (i === 1) return [0, CUP * cup(fov)];
  if (i === 0) { const s = M.smooth((lg(M.W0) - lg(fov)) / (lg(M.W0) - lg(0.05))), an = MRI ? MRI.anchor : MRI_AN; return [lerp(0, an[0], s), lerp(4, an[1] + CUP * 1e-3 * cup(fov), s)]; }
  const C = VC[i]; if (!C) return ZERO; const c0 = C.c0 || ZERO; if (!C.an) return c0;
  const L = EV.LV(i), an = typeof C.an === 'function' ? C.an() : C.an, a = lg(L.fov * 0.8), b = lg(Math.min(L.fov * 0.25, L.bot * 2.5)), d = Math.hypot(an[0] - c0[0], an[1] - c0[1]);
  let s = smooth((a - lg(fov)) / (a - b)); if (d > 0) s = clamp(Math.max(s, 1 - 0.36 * (fov / L.unit) / d), 0, 1); return [lerp(c0[0], an[0], s), lerp(c0[1], an[1], s)]; };
const scaleOf = (i, fov) => W / (fov / unit(i));
const toScreen = (i, fov, x, y) => { const c = centre(i, fov), s = scaleOf(i, fov); return [CXs() + (x - c[0]) * s, H / 2 + (y - c[1]) * s]; };
function place(canvas, bx0, by0, bw, i, fov) { const c = centre(i, fov), s = scaleOf(i, fov), k = canvas.width / bw;
  const cx = CXs(), vx0 = Math.max(bx0, c[0] - cx / s), vy0 = Math.max(by0, c[1] - H / 2 / s), vx1 = Math.min(bx0 + bw, c[0] + (W - cx) / s), vy1 = Math.min(by0 + bw, c[1] + H / 2 / s); if (vx1 <= vx0 || vy1 <= vy0) return;
  ctx.drawImage(canvas, (vx0 - bx0) * k, (vy0 - by0) * k, (vx1 - vx0) * k, (vy1 - vy0) * k, cx + (vx0 - c[0]) * s, H / 2 + (vy0 - c[1]) * s, (vx1 - vx0) * s, (vy1 - vy0) * s); }
/* several pictures of each level, each finer and narrower than the last. A small quick one is made first, as a preview. */
const tierW = i => { const L = EV.LV(i), u = L.unit, C = VC[i] || {}, wide = L.top / u * ASP * 1.03, wp0 = L.fov / u * ASP * 1.3, fine = L.bot / u * ASP * 1.6, out = [], em = i === 3 || i === 4, top = C.top ? Math.min(BKW, C.top) : em ? Math.round(BKW * 0.875) : BKW;
  let w = wp0; if (wide > wp0 * 1.25) out.push([wide, BK]); else w = wide; out.push([w, top]);
  if (w / fine > 1.45 && C.maxTiers !== 0) { const n = Math.min(Math.ceil(Math.log(w / fine) / Math.log(2.9)), C.maxTiers || 9); for (let k = 1; k <= n; k++) out.push([w * Math.pow(fine / w, k / n), BK]); } return out; };
const painterOf = i => i >= 1 && i <= 6 ? PAINT_B[i] : VC[i] && VC[i].paint;
function* bakeLevel(i) {
  if (i === 0) { if (!MRI) { yield; MRI = makeMRI(); if (st.vox !== 1) MRI.voxelise(st.vox); } BAKES[0] = { tiers: [], done: true }; return; }
  const P = painterOf(i); if (!P) { BAKES[i] = { tiers: [], done: true }; return; }
  const fresh = !BAKES[i] || !BAKES[i].tiers.length, out = { tiers: [], done: false }, tw = tierW(i), widest = Math.max.apply(null, tw.map(t => t[0])), L = EV.LV(i);
  const one = function* (w, px, pv) { const c = centre(i, w * L.unit / (ASP * 1.3)), Bk = newBake(w, px, i >= 1 && i <= 6 ? 0 : c[0], i >= 1 && i <= 6 ? 0 : c[1]); Bk.preview = pv; Bk.id = i; if (i === 3) Bk.col = mk(px, px); yield* P(Bk); if (w < widest) { feather(Bk.c); if (Bk.col) feather(Bk.col); } return Bk; };
  if (fresh) { BAKES[i] = out; const wp = tw.find(t => t[1] > BK) || tw[0], pv = yield* one(Math.min(widest, wp[0] * 2.2), Math.round(BK / 3.2), true); out.tiers.push(pv); yield; }
  for (const [w, px] of tw.slice().sort((a, b) => (b[1] - a[1]) || (b[0] - a[0]))) { const Bk = yield* one(w, px, false); if (w === widest) out.tiers = out.tiers.filter(t => !t.preview); out.tiers.push(Bk); out.tiers.sort((a, b) => b.w - a.w); yield; }
  out.tiers = out.tiers.filter(t => !t.preview); out.done = true; BAKES[i] = out; }
const JOBS = []; let job = null, bakesDone = false;
function request(i, urgent) { if (i < EV.IDMIN || i > EV.IDMAX || (BAKES[i] && BAKES[i].done) || (job && job.i === i)) return; if (i !== 0 && !painterOf(i)) return; const k = JOBS.indexOf(i); if (k >= 0) { if (urgent && k > 0) { JOBS.splice(k, 1); JOBS.unshift(i); } return; } if (urgent) JOBS.unshift(i); else JOBS.push(i); }
function pump(budget) { const t0 = performance.now(); let did = false; while (performance.now() - t0 < budget) { if (!job) { if (!JOBS.length) { bakesDone = true; break; } const i = JOBS.shift(); job = { i, it: bakeLevel(i) }; bakesDone = false; } did = true; if (job.it.next().done) job = null; } return did; }
function rebake(i) { if (job && job.i === i) job = null; const k = JOBS.indexOf(i); if (k >= 0) JOBS.splice(k, 1); if (BAKES[i]) BAKES[i].done = false; JOBS.unshift(i); bakesDone = false; }
/* let go of pictures more than two waypoints from the frame, and of work queued for them */
function tidy(cur) { for (const k in BAKES) if (Math.abs(+k - cur) > 2) { if (+k === 0) continue; delete BAKES[k]; } for (let j = JOBS.length - 1; j >= 0; j--) if (Math.abs(JOBS[j] - cur) > 2) JOBS.splice(j, 1);
  if (job && Math.abs(job.i - cur) > 2) { if (BAKES[job.i] && !BAKES[job.i].done) delete BAKES[job.i]; job = null; } }
function drawLevel(i, fov, alpha) { ctx.globalAlpha = alpha;
  if (i === 0) { if (!MRI) { request(0, true); ctx.globalAlpha = 1; return; } const px = MRI.vox * scaleOf(0, fov), hard = M.smooth((px - 5) / 7); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; if (hard < 1) place(MRI.fine, -110, -110, 220, 0, fov); if (hard > 0) { ctx.globalAlpha = alpha * hard; ctx.imageSmoothingEnabled = false; place(MRI.canvas, -110, -110, 220, 0, fov); ctx.imageSmoothingEnabled = true; } ctx.globalAlpha = 1; return; }
  if (VC[i] && VC[i].back) VC[i].back(fov, alpha);
  const B = BAKES[i]; if (!B || !B.tiers.length) { request(i, true); ctx.globalAlpha = 1; return; }
  ctx.globalAlpha = alpha; ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high'; if (VC[i] && VC[i].add) ctx.globalCompositeOperation = 'lighter';
  for (const T of B.tiers) place(T.c, T.x0, T.y0, T.w, i, fov);
  if (i === 3 && st.recon > 0) { ctx.globalAlpha = alpha * st.recon * 0.74; for (const T of B.tiers) if (T.col) place(T.col, T.x0, T.y0, T.w, i, fov); }
  ctx.globalAlpha = 1; ctx.globalCompositeOperation = 'source-over'; }

/* ════ A person: in the style of a whole-body MRI scan, cut from side to side. Metres; the origin is the middle of the head. ════ */
const BODY = [[0.05, 0.10], [0.062, 0.16], [0.13, 0.195], [0.19, 0.225], [0.178, 0.33], [0.166, 0.45], [0.152, 0.55], [0.165, 0.66], [0.186, 0.76], [0.19, 0.88], [0.176, 1.0], [0.15, 1.13], [0.142, 1.22], [0.15, 1.32], [0.116, 1.46], [0.098, 1.535], [0.128, 1.588], [0.06, 1.6], [0.048, 1.53], [0.044, 1.4], [0.04, 1.24], [0.046, 1.13], [0.026, 1.0], [0.012, 0.87], [0, 0.845]];
function* paintPerson(Bk) { const s = Bk.s, n = Bk.px, T = tmpFor(n), g = worldOn(T, Bk), mm = 0.001; g.lineCap = 'round'; g.lineJoin = 'round';
  const grey = v => 'rgb(' + v + ',' + v + ',' + v + ')', out = BODY.concat(BODY.slice(0, -1).reverse().map(p => [-p[0], p[1]]));
  const sil = () => { trace(g, spline(out, true, 6), true); }, arms = (w, col) => { g.strokeStyle = col; for (const sx of [-1, 1]) { g.lineWidth = w; g.beginPath(); g.moveTo(sx * 0.2, 0.27); g.lineTo(sx * 0.236, 0.55); g.stroke(); g.lineWidth = w * 0.82; g.beginPath(); g.moveTo(sx * 0.236, 0.55); g.lineTo(sx * 0.258, 0.81); g.stroke(); g.lineWidth = w * 0.62; g.beginPath(); g.moveTo(sx * 0.258, 0.83); g.lineTo(sx * 0.264, 0.92); g.stroke(); } };
  const head = (dx, dy, col) => { g.fillStyle = col; g.beginPath(); g.ellipse(0, 0.004, 0.0805 - dx, 0.108 - dy, 0, 0, TAU); g.fill(); };
  /* fat under the skin, then muscle inside it */
  g.fillStyle = grey(214); sil(); g.fill(); arms(0.094, grey(214)); head(0, 0, grey(214));
  g.save(); sil(); g.clip(); g.fillStyle = grey(96); g.fillRect(-1, 0.1, 2, 1.6); g.strokeStyle = grey(214); g.lineWidth = 0.03; sil(); g.stroke(); g.restore(); arms(0.064, grey(96));
  /* bones: a dark shell and bright marrow */
  const bone = (x0, y0, x1, y1, w) => { for (const sx of [-1, 1]) { g.strokeStyle = grey(26); g.lineWidth = w; g.beginPath(); g.moveTo(sx * x0, y0); g.lineTo(sx * x1, y1); g.stroke(); g.strokeStyle = grey(178); g.lineWidth = w * 0.55; g.beginPath(); g.moveTo(sx * x0, y0); g.lineTo(sx * x1, y1); g.stroke(); } };
  bone(0.118, 0.8, 0.092, 1.12, 0.03); bone(0.09, 1.17, 0.08, 1.5, 0.026); bone(0.118, 1.2, 0.108, 1.48, 0.012); bone(0.2, 0.28, 0.234, 0.54, 0.022); bone(0.24, 0.57, 0.258, 0.8, 0.016); bone(0.06, 0.2, 0.17, 0.225, 0.012);
  for (const sx of [-1, 1]) { g.fillStyle = grey(30); g.beginPath(); g.ellipse(sx * 0.105, 0.72, 0.06, 0.046, sx * 0.5, 0, TAU); g.fill(); g.fillStyle = grey(168); g.beginPath(); g.ellipse(sx * 0.105, 0.72, 0.05, 0.036, sx * 0.5, 0, TAU); g.fill(); g.fillStyle = grey(176); g.beginPath(); g.arc(sx * 0.112, 0.795, 0.024, 0, TAU); g.fill(); }
  for (let k = 0; k < 24; k++) { const y = 0.12 + k * 0.0262, w = 0.026 + 0.012 * k / 24; g.fillStyle = grey(30); g.fillRect(-w / 2 - 0.003, y, w + 0.006, 0.0262); g.fillStyle = grey(162); g.fillRect(-w / 2, y + 0.004, w, 0.0182); }
  /* the chest and the belly */
  for (const sx of [-1, 1]) { g.fillStyle = grey(18); trace(g, spline([[sx * 0.03, 0.27], [sx * 0.075, 0.232], [sx * 0.13, 0.27], [sx * 0.15, 0.36], [sx * 0.15, 0.45], [sx * 0.1, 0.475], [sx * 0.035, 0.44], [sx * 0.026, 0.35]], true, 6), true); g.fill();
    g.strokeStyle = grey(70); g.lineWidth = 2.5 * mm; for (let k = 0; k < 7; k++) { g.beginPath(); g.moveTo(sx * 0.03, 0.34 + 0.01 * k); g.quadraticCurveTo(sx * 0.08, 0.3 + 0.022 * k, sx * (0.1 + 0.006 * k), 0.26 + 0.03 * k); g.stroke(); } }
  g.fillStyle = grey(118); g.beginPath(); g.ellipse(0.022, 0.4, 0.052, 0.062, -0.5, 0, TAU); g.fill(); g.fillStyle = grey(52); g.beginPath(); g.ellipse(0.03, 0.405, 0.026, 0.034, -0.5, 0, TAU); g.fill();
  g.fillStyle = grey(128); trace(g, spline([[-0.14, 0.47], [-0.06, 0.455], [0.03, 0.475], [0.02, 0.52], [-0.07, 0.565], [-0.142, 0.55]], true, 6), true); g.fill();
  g.fillStyle = grey(38); g.beginPath(); g.ellipse(0.085, 0.5, 0.036, 0.026, 0.3, 0, TAU); g.fill(); g.fillStyle = grey(112); g.beginPath(); g.ellipse(0.105, 0.535, 0.022, 0.034, 0.2, 0, TAU); g.fill();
  for (const sx of [-1, 1]) { g.fillStyle = grey(140); g.beginPath(); g.ellipse(sx * 0.075, 0.6, 0.022, 0.04, sx * 0.25, 0, TAU); g.fill(); g.fillStyle = grey(200); g.beginPath(); g.ellipse(sx * 0.068, 0.6, 0.007, 0.016, sx * 0.25, 0, TAU); g.fill(); }
  const rb = M.rng(9); for (let k = 0; k < 46; k++) { const x = (rb() - 0.5) * 0.24, y = 0.6 + rb() * 0.16; if (Math.abs(x) < 0.025) continue; g.fillStyle = grey(40 + 150 * rb() | 0); g.beginPath(); g.ellipse(x, y, 0.012 + 0.012 * rb(), 0.009 + 0.008 * rb(), rb() * 3, 0, TAU); g.fill(); }
  g.fillStyle = grey(54); g.beginPath(); g.ellipse(0, 0.815, 0.034, 0.024, 0, 0, TAU); g.fill();
  /* the head: scalp, skull, brain; it must agree with the MRI slice that follows */
  head(0.0065, 0.0065, grey(22)); head(0.013, 0.013, grey(116)); g.fillStyle = grey(176); for (const sx of [-1, 1]) { g.beginPath(); g.ellipse(sx * 0.03, -0.022, 0.022, 0.034, sx * 0.3, 0, TAU); g.fill(); } g.fillStyle = grey(24); g.beginPath(); g.ellipse(0, -0.026, 0.01, 0.012, 0, 0, TAU); g.fill();
  g.fillStyle = grey(98); g.fillRect(-0.05, 0.068, 0.1, 0.05); g.fillStyle = grey(150); g.fillRect(-0.012, 0.02, 0.024, 0.1);
  yield;
  const G = Bk.c.getContext('2d'); fillAll(Bk, '#040506'); G.drawImage(soft(T, Math.max(0.6, 0.0035 * s)), 0, 0);
  /* the coil is less sensitive towards the ends of the table */
  const gr = G.createLinearGradient(0, (-0.2 - Bk.y0) * s, 0, (1.7 - Bk.y0) * s); gr.addColorStop(0, 'rgba(0,0,0,0.22)'); gr.addColorStop(0.2, 'rgba(0,0,0,0)'); gr.addColorStop(0.7, 'rgba(0,0,0,0.05)'); gr.addColorStop(1, 'rgba(0,0,0,0.42)'); G.fillStyle = gr; G.fillRect(0, 0, n, n);
  shotNoise(Bk, 0.3); }
VC[-1] = { c0: [0, 0.745], an: [0, 0], paint: paintPerson, maxTiers: 2 };

/* ════ A city at night from orbit. Kilometres; the origin is a street corner. Not a real city. ════ */
const CITY = { river: x => 2.3 * Math.sin(x / 5.6 + 0.6) + 1.1 * Math.sin(x / 2.2 + 2) + 1.6, dens: (x, y) => { const r = Math.hypot(x, y * 1.15); return r > 60 ? 0 : Math.exp(-r / 6.2) * (0.35 + 1.1 * fbm(x / 5 + 9, y / 5 + 3, 71, 2)) + 0.5 * Math.exp(-r / 1.6); },
  towns: (() => { const r = M.rng(55), t = []; for (let k = 0; k < 46; k++) { const a = r() * TAU, d = 26 + 300 * Math.pow(r(), 0.8); t.push([Math.cos(a) * d, Math.sin(a) * d, 1.2 + 5 * r() * r()]); } return t; })() };
function* paintCity(Bk) { const g = world(Bk), s = Bk.s, x0 = Bk.x0, y0 = Bk.y0, w = Bk.w, r = M.rng(1000 + Math.round(Math.log(w) * 9)); fillAll(Bk, '#04050a'); g.globalCompositeOperation = 'lighter';
  const SOD = [255, 176, 92], LED = [226, 234, 255], WARM = [255, 214, 150], pr = Math.max(0.011, 1.25 / s), near = w < 3;
  const hp = Math.max(0.62 / s, 0.012), big = hp * s > 2.4, lamp = (x, y, b, c) => { if (Math.abs(y - CITY.river(x)) < 0.19 + 0.06 * Math.sin(x)) return; if (big) dot(g, x, y, hp * 2.6, c, 0.4 + 0.6 * b); else pt(g, x, y, hp * (0.8 + 0.5 * b), c, 0.35 + 0.65 * b); if (b > 0.5 && !big) dot(g, x, y, Math.max(3.2 / s, 0.03), c, 0.3 * b); };
  /* streets: every district has its own grid, turned its own way */
  const R = w / 2, cx = x0 + R, cy = y0 + R, cap = Bk.preview ? 2.5e5 : 1.5e6, tries = Math.min(cap, 1100 * w * w), rep = 1100 * w * w / tries;
  for (let k = 0; k < tries; k++) { if (k % 80000 === 79999) yield; const x = cx + (r() * 2 - 1) * R, y = cy + (r() * 2 - 1) * R; if (r() * 1.25 > CITY.dens(x, y)) continue;
    /* the street grid turns and bends smoothly from one district to the next; a lamp is kept only where it falls on a street */
    const th = (vnoise(x / 8 + 3, y / 8, 81) - 0.5) * 2.2, co = Math.cos(th), si = Math.sin(th), u = x * co + y * si + 2.4 * (vnoise(x / 5, y / 5 + 7, 85) - 0.5), v = -x * si + y * co + 2.4 * (vnoise(x / 5 + 9, y / 5, 86) - 0.5), cell = 0.16, fu = Math.abs(u / cell - Math.round(u / cell)), fv = Math.abs(v / cell - Math.round(v / cell));
    if (Math.min(fu, fv * 0.8) > 0.07) continue; const b = r(), dn = vnoise(x / 2.6, y / 2.6, 83) + 0.2 * (r() - 0.5); lamp(x, y, Math.min(1, b * b * Math.sqrt(rep)), dn < 0.34 ? LED : dn < 0.55 ? WARM : SOD); }
  /* main roads out of the centre, two rings and an outer motorway */
  const road = (f, n, c, b) => { for (let k = 0; k < n; k++) { const p = f(k / n); if (p[0] < x0 - 1 || p[0] > x0 + w + 1 || p[1] < y0 - 1 || p[1] > y0 + w + 1) continue; lamp(p[0], p[1], b * (0.6 + 0.4 * hash2(k, n, 7)), c); } };
  for (let a = 0; a < 11; a++) { const th = a / 11 * TAU + 0.3 * hash2(a, 1, 91), bend = (hash2(a, 2, 92) - 0.5) * 0.04; road(t => { const d = 0.5 + t * 150, an = th + bend * d * 0.2; return [Math.cos(an) * d, Math.sin(an) * d]; }, near ? 60000 : 3000, a % 3 ? SOD : LED, 0.8); }
  for (const [rad, c] of [[3.4, WARM], [8.2, SOD], [19, LED]]) road(t => { const an = t * TAU, d = rad * (1 + 0.16 * (fbm(Math.cos(an) * 1.7 + rad, Math.sin(an) * 1.7, 93, 3) - 0.5) * 4); return [Math.cos(an) * d, Math.sin(an) * d * 0.9]; }, Math.round(rad * 110), c, 0.7);
  for (let b = -5; b <= 5; b++) { const bx = b * 2.3 + 0.4; road(t => [bx, CITY.river(bx) + (t - 0.5) * 0.6], 14, LED, 1); }
  /* other towns, seen when the frame is wide */
  if (w > 60) for (const t of CITY.towns) { const nn = Math.round(260 * t[2]); for (let k = 0; k < nn; k++) { const a = r() * TAU, d = -Math.log(1 - r()) * t[2] * 0.8; pt(g, t[0] + Math.cos(a) * d, t[1] + Math.sin(a) * d, hp, r() < 0.3 ? LED : SOD, 0.3 + 0.6 * r()); } }
  if (near) { /* close to: pools of light on the road */ g.globalAlpha = 0.5; for (let k = -40; k <= 40; k++) for (const ax of [0, 1]) { const u = k * 0.034, x = ax ? 0 : u, y = ax ? u : 0; if (x < x0 || x > x0 + w || y < y0 || y > y0 + w) continue; dot(g, x, y, 0.026, WARM, 0.55); dot(g, x, y, 0.004, [255, 244, 220], 1); } }
  g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; yield; bloom(Bk, Math.max(2, Bk.px / 260), 0.55); yield; bloom(Bk, Bk.px / 60, 0.22); shotNoise(Bk, 0.16); }
VC[-2] = { paint: paintCity, maxTiers: 3 };

/* ════ The Earth as a camera at a million miles sees it. Thousands of kilometres. Coastlines are rough outlines drawn from memory. ════ */
const EARTH = (() => { const R = 6.371, lon0 = 12 * Math.PI / 180, lat0 = 24 * Math.PI / 180;
  const LAND = [[[-17, 15], [-17, 21], [-13, 27], [-10, 30], [-6, 35.5], [-1, 35.5], [10, 37], [11, 33.5], [20, 32.5], [25, 31.5], [32, 31], [34, 28], [37, 22], [39, 15], [43, 11.5], [51, 12], [48, 5], [42, -1], [39, -6], [40, -15], [35, -22], [32, -28], [27, -34], [20, -34.8], [18, -33], [15, -27], [12, -17], [13.5, -11], [12, -5], [9, -1], [9.5, 4], [5, 5.5], [-4, 5], [-8, 4.5], [-13, 8.5], [-16.5, 12]],
    [[-9, 37], [-9.5, 43], [-1.5, 43.5], [-4.5, 48.5], [2, 51], [8, 54], [8.5, 57], [10.5, 57.5], [9, 54.5], [14, 54], [21, 55], [24, 59.5], [29, 60], [22, 60.5], [21, 64], [25, 65.7], [22, 66], [17, 61], [18, 59], [12.5, 56], [11, 59], [5, 59], [5, 62], [14, 68], [25, 71], [32, 70], [41, 66.5], [44, 68], [60, 69], [70, 73], [90, 76], [110, 77], [140, 72], [170, 69], [180, 66], [165, 60], [142, 54], [140, 46], [128, 39], [122, 31], [121, 25], [110, 21], [108, 11], [104, 9], [100, 13], [98, 8], [103, 1.5], [98, 16], [94, 17], [90, 22], [80, 15], [77, 8], [73, 19], [67, 24], [61, 25], [57, 26.5], [50, 30], [48, 30], [51, 24.5], [56, 25], [58, 20.5], [52, 16.5], [43, 13], [39, 21], [35, 28], [36, 36.5], [27, 36.5], [26, 40], [23, 38], [19, 42], [13, 45.5], [16, 41], [18, 40], [15.5, 38], [12, 44], [8, 44], [3, 43], [0, 39], [-5, 36]],
    [[-5, 50], [1.5, 51.5], [0, 53.5], [-2, 56], [-3, 58.5], [-6, 57], [-3, 54], [-5, 52]], [[-10, 52], [-6, 52], [-6, 55], [-8, 55]], [[44, -25], [47, -25], [50, -15.5], [49, -12], [44, -17]], [[-24, 63.5], [-14, 64.5], [-15, 66.5], [-23, 66]],
    [[-80, -3], [-77, 8], [-72, 12], [-62, 10.5], [-52, 5], [-50, 0], [-35, -5.5], [-39, -13], [-41, -22], [-48, -27], [-54, -34], [-58, -38.5], [-65, -41], [-69, -52], [-74, -50], [-73, -37], [-70, -18], [-76, -14], [-81, -5]],
    [[-55, 60], [-44, 60], [-20, 70], [-20, 80], [-60, 82], [-70, 77], [-52, 68]], [[-180, -70], [-60, -64], [-58, -72], [0, -70], [60, -67], [120, -66], [180, -70], [180, -90], [-180, -90]]];
  let mask = null, mw = 720, mh = 360;
  const build = () => { const c = mk(mw, mh), g = c.getContext('2d'); g.fillStyle = '#000'; g.fillRect(0, 0, mw, mh); g.fillStyle = '#fff'; for (const P of LAND) { g.beginPath(); P.forEach((p, k) => { const x = (p[0] + 180) * 2, y = (90 - p[1]) * 2; if (k) g.lineTo(x, y); else g.moveTo(x, y); }); g.closePath(); g.fill(); }
    const d = soft(c, 1.3).getContext('2d').getImageData(0, 0, mw, mh).data; mask = new Float32Array(mw * mh); for (let k = 0; k < mw * mh; k++) mask[k] = d[4 * k] / 255; };
  const land = (lon, lat) => { const fx = clamp((lon + 180) * 2 - 0.5, 0, mw - 1.001), fy = clamp((90 - lat) * 2 - 0.5, 0, mh - 1.001), i = fx | 0, j = fy | 0, u = fx - i, v = fy - j, k = j * mw + i; return mask[k] * (1 - u) * (1 - v) + mask[k + 1] * u * (1 - v) + mask[k + mw] * (1 - u) * v + mask[k + mw + 1] * u * v; };
  const cl = Math.cos(lat0), sl = Math.sin(lat0), sun = (() => { const v = [0.5, 0.42, 0.76], n = Math.hypot(v[0], v[1], v[2]); return v.map(a => a / n); })();
  /* a point of the disc → [r, g, b]; x to the right, y down, in units of the radius */
  const shade = (X, Y, out) => { const r2 = X * X + Y * Y; if (r2 > 1.07) { out[0] = 2; out[1] = 3; out[2] = 5; return; }
    if (r2 >= 1) { const t = Math.exp(-(Math.sqrt(r2) - 1) * 70), li = clamp(0.25 + 0.9 * (X * sun[0] - Y * sun[1]), 0, 1); out[0] = 2 + 60 * t * li; out[1] = 3 + 120 * t * li; out[2] = 5 + 220 * t * li; return; }
    const z = Math.sqrt(1 - r2), y1 = -Y * cl + z * sl, z1 = Y * sl + z * cl; let lat = Math.asin(clamp(y1, -1, 1)) * 180 / Math.PI, lon = (Math.atan2(X, z1) + lon0) * 180 / Math.PI;
    const px = Math.cos(lat * Math.PI / 180) * Math.sin(lon * Math.PI / 180), py = y1, pz = Math.cos(lat * Math.PI / 180) * Math.cos(lon * Math.PI / 180); const wob = (fbm3(px * 9, py * 9, pz * 9, 31, 4) - 0.5) * 9;
    const lonR = lon, latR = lat, lm = land(((lon + wob * 0.6 + 540) % 360) - 180, clamp(lat + wob * 0.5, -89, 89)), isLand = lm > 0.5; lat = latR + wob * 1.4; lon = lonR + wob * 1.8; const al = Math.abs(lat), n1 = fbm3(px * 5, py * 5, pz * 5, 37, 4), n2 = fbm3(px * 22, py * 22, pz * 22, 41, 3);
    let cr, cg, cb;
    if (isLand) { const dry = Math.max(ss(11, 17, lat) * ss(34, 28, lat) * ss(-19, -12, lon) * ss(64, 52, lon), ss(-31, -25, lat) * ss(-16, -21, lat) * ss(10, 15, lon) * ss(27, 21, lon), ss(34, 39, lat) * ss(50, 44, lat) * ss(52, 60, lon)), wet = ss(9, 4, al) * ss(6, 10, lon) * ss(32, 26, lon) + ss(8, 3, al) * ss(-48, -56, lon);
      const sav = [122, 116, 70], des = [206, 176, 124], rain = [34, 72, 40], temp = [70, 96, 58], tun = [118, 112, 96]; let c = al > 58 ? tun : al > 34 ? temp : sav; const m = (a, b, t) => [lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)];
      c = m(c, al > 34 ? temp : sav, 0); c = m(c, rain, clamp(wet * (0.6 + 0.8 * n1), 0, 1)); c = m(c, des, clamp(dry * (0.75 + 0.6 * n1), 0, 1)); const k = 0.78 + 0.5 * n2; cr = c[0] * k; cg = c[1] * k; cb = c[2] * k;
      if (lat < -64 || (lat > 61 && lon < -24 && lon > -75) || al > 78) { cr = 232; cg = 238; cb = 244; } }
    else { const sh = ss(0.12, 0.5, lm); cr = lerp(7, 26, sh); cg = lerp(26, 84, sh); cb = lerp(62, 118, sh); if (al > 72 + 6 * n1) { cr = 214; cg = 226; cb = 236; } }
    /* cloud: belts in the middle latitudes and near the equator, clear over the deserts */
    const w1 = fbm3(px * 2.3 + 3, py * 2.3, pz * 2.3, 51, 3) - 0.5, w2 = fbm3(px * 2.3, py * 2.3 + 7, pz * 2.3, 53, 3) - 0.5, wx = px * 4.6 + 3.4 * w1, wy = py * 8.5 + 3.0 * w2, wz = pz * 4.6 + 3.4 * w1 * w2 * 4, c0 = 0.62 * fbm3(wx, wy, wz, 57, 5) + 0.38 * fbm3(wx * 3.1, wy * 3.1, wz * 3.1, 59, 4);
    const band = 0.3 + 0.36 * Math.exp(-Math.pow((al - 52) / 15, 2)) + 0.26 * Math.exp(-Math.pow((latR - 6) / 7, 2)) - 0.2 * Math.exp(-Math.pow((al - 24) / 9, 2)), cloud = ss(0.585, 0.76, c0 + band * 0.36) * (0.75 + 0.25 * n2), wht = 232 + 20 * n2; cr = lerp(cr, wht, cloud * 0.95); cg = lerp(cg, wht + 2, cloud * 0.95); cb = lerp(cb, wht + 4, cloud * 0.95);
    /* sunlight, a glint on the sea, and air at the limb */
    const nd = X * sun[0] - Y * sun[1] + z * sun[2], li = clamp(nd * 1.25 + 0.12, 0, 1), day = Math.pow(li, 0.75) * 1.06 + 0.012, hx = sun[0], hy = sun[1], hz = sun[2] + 1, hn = Math.hypot(hx, hy, hz), sp = isLand ? 0 : Math.pow(Math.max(0, (X * hx - Y * hy + z * hz) / hn), 60) * (1 - cloud) * 150;
    const rim = Math.pow(1 - z, 2.6) * li; out[0] = cr * day + sp + 70 * rim; out[1] = cg * day + sp + 130 * rim; out[2] = cb * day + sp * 0.9 + 235 * rim; };
  const lonC = -0.2 * Math.PI / 180, latC = 51.5 * Math.PI / 180, dl = lonC - lon0, cx = Math.cos(latC) * Math.sin(dl), cy3 = Math.sin(latC), cz = Math.cos(latC) * Math.cos(dl);
  return { R, shade, build, ready: () => !!mask, city: [cx * R, -(cy3 * cl - cz * sl) * R], sun }; })();
function* paintEarth(Bk) { if (!EARTH.ready()) EARTH.build(); const R = EARTH.R, o = [0, 0, 0]; yield* pixels(Bk, (x, y, out, q) => { EARTH.shade(x / R, y / R, out); }, Bk.preview ? 400 : 40); }
VC[-3] = { an: () => EARTH.city, paint: paintEarth, maxTiers: 3, back: (f, a) => starfield(a * 0.8) };
let EARTH_SPR = null, MOON_SPR = null;
const smallWorlds = () => { if (EARTH_SPR) return; if (!EARTH.ready()) EARTH.build(); const n = 120, mkS = fn => { const c = mk(n, n), g = c.getContext('2d'), im = g.createImageData(n, n), o = [0, 0, 0, 255]; for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const X = (i + 0.5 - n / 2) / (n / 2 / 1.07), Y = (j + 0.5 - n / 2) / (n / 2 / 1.07); o[3] = 255; fn(X, Y, o); const k = 4 * (j * n + i); im.data[k] = o[0]; im.data[k + 1] = o[1]; im.data[k + 2] = o[2]; im.data[k + 3] = o[3]; } g.putImageData(im, 0, 0); return c; };
  EARTH_SPR = mkS((X, Y, o) => { EARTH.shade(X, Y, o); if (X * X + Y * Y > 1.06) o[3] = 0; });
  MOON_SPR = mkS((X, Y, o) => { const r2 = X * X + Y * Y; if (r2 > 1) { o[3] = 0; return; } const z = Math.sqrt(1 - r2), s = EARTH.sun, li = clamp((X * s[0] - Y * s[1] + z * s[2]) * 1.3 + 0.05, 0, 1), mare = ss(0.52, 0.6, fbm(X * 1.6 + 3, Y * 1.6, 61, 4)), v = (168 - 62 * mare + 30 * (fbm(X * 9, Y * 9, 63, 3) - 0.5)) * Math.pow(li, 0.8); o[0] = v; o[1] = v * 0.98; o[2] = v * 0.95; }); };
VC[-4] = { c0: [192.2, 0], an: [0, 0], back: (f, a) => starfield(a) };

/* ════ The Solar System, from above its plane. Astronomical units. Only the belts are baked; the planets are drawn where they are today. ════ */
function* paintSolar(Bk) { const g = world(Bk), r = M.rng(12), pr = Math.max(0.012, 0.8 / Bk.s); g.clearRect(Bk.x0, Bk.y0, Bk.w, Bk.w); g.globalCompositeOperation = 'lighter';
  const nA = Math.min(5200, Math.round(20 * Bk.s * Bk.s * 0.16)); for (let k = 0; k < nA; k++) { const a = r() * TAU, d = 2.1 + 1.2 * (r() + r()) / 2; pt(g, Math.cos(a) * d, Math.sin(a) * d, 0.55 / Bk.s, [206, 196, 176], 0.25 + 0.35 * r()); }
  const nK = Math.min(26000, Math.round(3000 * Bk.s * Bk.s * 0.03)); for (let k = 0; k < nK; k++) { const a = r() * TAU, d = 38 + 11 * (r() + r() + r()) / 3 + (r() < 0.1 ? 18 * r() : 0); pt(g, Math.cos(a) * d, Math.sin(a) * d, 0.55 / Bk.s, [176, 198, 226], 0.3 + 0.4 * r()); }
  const gr = g.createRadialGradient(0, 0, 0, 0, 0, 6); gr.addColorStop(0, 'rgba(255,240,205,0.3)'); gr.addColorStop(0.12, 'rgba(255,214,150,0.08)'); gr.addColorStop(1, 'rgba(255,200,120,0)'); g.globalAlpha = 1; g.fillStyle = gr; g.fillRect(-6, -6, 12, 12); g.globalCompositeOperation = 'source-over'; }
VC[-5] = { an: () => EARTH_AT, paint: paintSolar, maxTiers: 0, add: true, back: (f, a) => starfield(a) };

/* ════ The heliosphere: Parker's (1961) model of a steady wind from a point meeting a uniform flow. Astronomical units.
   The interstellar gas arrives from the right. Stream function, with θ measured from the downstream (left) axis:
   ψ = ½ (r/a)² sin²θ − cos θ; the surface ψ = 1 is the heliopause, a from the centre at its nose and 2a wide far downstream. ════ */
const HELIO = { a: 119, ts: th => 86 * (1.28 - 0.28 * Math.cos(Math.PI - th)), hp: th => 119 * Math.sqrt(2 / Math.max(1e-6, 1 - Math.cos(th))), v1: 32 * Math.PI / 180, v2: -47 * Math.PI / 180 };
function* paintHelio(Bk) { const a = HELIO.a; yield* pixels(Bk, (x, y, o, q) => { const r = Math.hypot(x, y) + 1e-6, th = Math.atan2(Math.abs(y), -x), psi = 0.5 * (r / a) * (r / a) * Math.sin(th) * Math.sin(th) - Math.cos(th), inside = psi < 1, rts = HELIO.ts(th);
    const n = fbm(x / 34 + 7, y / 34, 131, 4), n2 = fbm(x / 9, y / 9, 137, 3); let cr, cg, cb;
    if (inside && r < rts) { const k = 1 / (1 + Math.pow(r / 16, 2)); cr = 10 + 235 * k; cg = 14 + 190 * k; cb = 44 + 110 * k + 26 * (r / rts); }
    else if (inside) { const d = clamp((r - rts) / (HELIO.hp(th) - rts + 1e-6), 0, 1), hot = 0.62 + 0.5 * n2 - 0.3 * d; cr = 226 * hot + 20; cg = 96 * hot + 30 * (1 - d); cb = 44 + 70 * d * hot; }
    else { const wall = Math.exp(-Math.max(0, psi - 1) * 2.2) * ss(0.2, 1.4, -Math.cos(th) + 1), flow = 0.5 + 0.5 * n; cr = 6 + 24 * wall + 6 * flow; cg = 20 + 66 * wall + 20 * flow; cb = 40 + 96 * wall + 34 * flow; }
    /* streamlines: curves of constant ψ, spaced evenly in sqrt(|ψ ∓ 1|) so that they look evenly spread */
    const sv = Math.sqrt(Math.abs(psi - 1)) * (inside ? 5.2 : 3.6), f = Math.abs(sv - Math.round(sv)), ray = Math.abs((Math.atan2(y, x) / TAU * 44) % 1), line = inside && r < rts ? 0.07 * Math.exp(-Math.pow((Math.min(ray, 1 - ray)) * r / 44 * TAU / (0.5 + q), 2)) * ss(6, 30, r) : Math.round(sv) === 0 ? 0 : Math.exp(-Math.pow(f / (0.06 + 0.5 * q / (r * 0.02 + 6)), 2)) * (inside ? 0.18 : 0.24) * ss(0, 14, Math.abs(y));
    const edge = Math.exp(-Math.pow((r - HELIO.hp(th)) / (1.6 + q + 0.02 * r), 2)) * 0.8 * ss(0.05, 0.5, 1 - Math.cos(th)) + (inside ? Math.exp(-Math.pow((r - rts) / (1.4 + q), 2)) * 0.5 : 0);
    o[0] = cr + 255 * (line + edge * 0.9); o[1] = cg + 255 * (line + edge * 0.86); o[2] = cb + 255 * (line * 0.9 + edge * 0.7); }, Bk.preview ? 400 : 48); shotNoise(Bk, 0.08); }
VC[-6] = { c0: [28, 0], an: [0, 0], paint: paintHelio, maxTiers: 1 };

/* ════ The Oort cloud: never seen. Dots placed by rule in thousands of astronomical units: a flattened inner cloud and a round outer one. ════ */
function* paintOort(Bk) { const g = world(Bk), r = M.rng(77), pr = Math.max(0.05, 0.75 / Bk.s), gz = () => M.gauss(r); g.clearRect(Bk.x0, Bk.y0, Bk.w, Bk.w); g.globalCompositeOperation = 'lighter';
  const N = Bk.preview ? 9000 : 52000; for (let k = 0; k < N; k++) { const inner = k % 5 < 2, d = inner ? 2 + 18 * Math.pow(r(), 1.7) : 20 * Math.pow(5, r()), x = gz(), y = gz() * (inner ? 0.42 : 1), z = gz(), n = d / Math.hypot(x, y, z);
    pt(g, x * n, y * n, 0.5 / Bk.s, inner ? [196, 214, 236] : [150, 182, 226], (inner ? 0.13 : 0.5) * (0.3 + 0.7 * r())); }
  const gr = g.createRadialGradient(0, 0, 0, 0, 0, 60); gr.addColorStop(0, 'rgba(150,180,230,0.16)'); gr.addColorStop(1, 'rgba(120,150,220,0)'); g.globalAlpha = 1; g.fillStyle = gr; g.fillRect(-60, -60, 120, 120); g.globalCompositeOperation = 'source-over'; }
VC[-7] = { paint: paintOort, maxTiers: 1, add: true, back: (f, a) => starfield(a) };
VC[-8] = { back: (f, a) => starfield(a) };

/* ════ The Milky Way from above: a drawing. Nobody has seen it from outside. Thousands of light-years; the origin is the centre. ════ */
const GAL = { pitch: 1 / Math.tan(12.5 * Math.PI / 180), bar: 28 * Math.PI / 180, sunY: EV.SGRA.Dly / 1000 };
function galaxyPix(x, y, o, q, seed = 0) { const r = Math.hypot(x, y) + 1e-6, th = Math.atan2(y, x), cb = Math.cos(GAL.bar), sb = Math.sin(GAL.bar), xb = x * cb + y * sb, yb = -x * sb + y * cb;
  const n1 = fbm(x / 5 + 11 + seed, y / 5 + 3, 151, 3), n2 = fbm(x / 1.6 + seed, y / 1.6, 157, 3), disc = Math.exp(-r / 13) * ss(62, 40, r), bulge = 1.0 * Math.exp(-Math.pow(r / 3.4, 1.15)), bar = 0.62 * Math.exp(-Math.pow(xb / 9.5, 2) - Math.pow(yb / 2.5, 2));
  let arm = 0, dust = 0, knots = 0; const ph0 = Math.log(r / 10.5) * GAL.pitch, on = ss(7, 13, r) * ss(58, 36, r);
  for (let k = 0; k < 4; k++) { const st0 = k % 2 ? 0.55 : 1, d0 = Math.atan2(Math.sin(th - ph0 - GAL.bar - k * Math.PI / 2 - (k % 2 ? 0.25 : 0)), Math.cos(th - ph0 - GAL.bar - k * Math.PI / 2 - (k % 2 ? 0.25 : 0))) * r, wd = 3.0 + 0.07 * r + 3.2 * (n1 - 0.5);
    arm += st0 * Math.exp(-d0 * d0 / (wd * wd)); dust += st0 * Math.exp(-Math.pow((d0 + 2.4 + 2.2 * (n2 - 0.5)) / 1.1, 2)); knots += st0 * Math.exp(-Math.pow((d0 - 0.3) / 1.1, 2)); }
  /* the Orion spur, where the Sun is */
  { const d0 = Math.atan2(Math.sin(th - ph0 * 0.86 - 1.96), Math.cos(th - ph0 * 0.86 - 1.96)) * r; arm += 0.4 * Math.exp(-d0 * d0 / 2.2) * ss(18, 24, r) * ss(36, 28, r); }
  arm = Math.min(arm, 1.3) * on * (0.22 + 1.5 * ss(0.3, 0.72, n2) * (0.5 + n1)); dust *= on; const ci = Math.floor(x / 1.1 + seed), cj = Math.floor(y / 1.1), hk = hash2(ci, cj, 163), kn = hk > 0.9 ? Math.exp(-(Math.pow(x / 1.1 + seed - ci - 0.5, 2) + Math.pow(y / 1.1 - cj - 0.5, 2)) * 14) * knots * on : 0;
  const young = disc * 4.2 * arm + disc * 0.5 * (0.5 + n2), old = disc * 1.0 * (0.7 + 0.6 * n1) + bulge + bar, dk = 1 - 0.5 * clamp(dust * ss(0.35, 0.6, n2) * 1.6, 0, 1) * ss(0.02, 0.12, disc + bulge), grain = 0.86 + 0.28 * hash2((x / q) | 0, (y / q) | 0, 167);
  let R = (old * 250 + young * 150 + kn * 300) * dk, G = (old * 205 + young * 180 + kn * 120) * dk, B = (old * 150 + young * 250 + kn * 170) * dk; const tone = v => 255 * (1 - Math.exp(-v * grain / 78));
  o[0] = tone(R) + 2; o[1] = tone(G) + 2; o[2] = tone(B) + 4; }
function* paintGalaxy(Bk) { yield* pixels(Bk, (x, y, o, q) => galaxyPix(x, y, o, q), Bk.preview ? 400 : 32);
  if (Bk.s > 12 && !Bk.preview) { /* close in, the smooth light breaks up into stars */ const g = world(Bk), r = M.rng(Math.round(Bk.w * 7)), N = 26000, a = ss(12, 120, Bk.s); g.globalCompositeOperation = 'lighter'; for (let k = 0; k < N; k++) { const t = r(), c = t < 0.25 ? [255, 206, 160] : t < 0.8 ? [240, 240, 255] : [180, 205, 255], m = Math.pow(r(), 5); dot(g, Bk.x0 + r() * Bk.w, Bk.y0 + r() * Bk.w, (0.7 + 3.4 * m) / Bk.s, c, a * (0.25 + 0.75 * m)); } g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; }
  shotNoise(Bk, 0.12); }
VC[-9] = { an: [0, GAL.sunY], paint: paintGalaxy, maxTiers: 3, top: 1300 };
let GAL_SPR = null;
const galSprite = () => { if (GAL_SPR) return GAL_SPR; const n = 220, c = mk(n, n), g = c.getContext('2d'), im = g.createImageData(n, n), o = [0, 0, 0, 255], q = 130 / n; for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const x = (i + 0.5) * q - 65, y = (j + 0.5) * q - 65; galaxyPix(x, y, o, q); const k = 4 * (j * n + i), m = Math.max(o[0], o[1], o[2]); im.data[k] = o[0]; im.data[k + 1] = o[1]; im.data[k + 2] = o[2]; im.data[k + 3] = clamp((m - 6) * 9, 0, 255); } g.putImageData(im, 0, 0); return (GAL_SPR = c); };

/* ════ The Local Group. Millions of light-years; the origin is the Milky Way. Distances of the three large galaxies are true;
   directions and the dwarf galaxies are schematic. ════ */
const LOCAL = { m31: [-1.72, -1.83], m33: [-1.2, -2.45], lmc: [0.118, 0.112], smc: [0.152, 0.13] };
function* paintLocal(Bk) { const g = world(Bk), s = Bk.s, r = M.rng(303), S = galSprite(); fillAll(Bk, '#020306'); g.globalCompositeOperation = 'lighter';
  const gal = (x, y, R, incl, pa, a = 1) => { dot(g, x, y, Math.max(R * 3.2, 5 / s), [176, 190, 255], 0.3 * a); g.save(); g.translate(x, y); g.rotate(pa); g.scale(1, Math.max(0.08, Math.cos(incl))); g.globalAlpha = a; g.drawImage(S, -R * 1.3, -R * 1.3, R * 2.6, R * 2.6); g.restore(); if (R * s < 3) dot(g, x, y, Math.max(R * 1.6, 2.2 / s), [226, 226, 255], a); };
  for (let k = 0; k < 2600; k++) pt(g, Bk.x0 + r() * Bk.w, Bk.y0 + r() * Bk.w, (0.5 + 0.5 * Math.pow(r(), 4)) / s, r() < 0.4 ? [255, 214, 176] : [206, 216, 255], 0.12 + 0.4 * Math.pow(r(), 3));
  const rf = M.rng(304); for (let k = 0; k < 26; k++) { const a = rf() * TAU, d = 7 + 34 * rf(), R = 0.03 + 0.07 * rf(); gal(Math.cos(a) * d, Math.sin(a) * d, R, rf() * 1.4, rf() * 3, 0.9); }
  for (let k = 0; k < 54; k++) { const host = k % 2 ? LOCAL.m31 : [0, 0], a = rf() * TAU, d = 0.12 + 0.9 * rf() * rf(); dot(g, host[0] + Math.cos(a) * d, host[1] + Math.sin(a) * d * 0.8, Math.max(0.012 + 0.012 * rf(), 1.4 / s), [214, 214, 236], 0.3 + 0.3 * rf()); }
  for (const p of [[1.1, -1.2], [-2.3, 0.6], [0.4, 2.6], [2.0, 1.6], [-0.6, -3.4], [-3.4, -2.6]]) dot(g, p[0], p[1], Math.max(0.03, 1.8 / s), [196, 208, 255], 0.6);
  dot(g, LOCAL.lmc[0], LOCAL.lmc[1], Math.max(0.03, 2 / s), [214, 222, 255], 0.8); dot(g, LOCAL.smc[0], LOCAL.smc[1], Math.max(0.02, 1.6 / s), [214, 222, 255], 0.7);
  gal(LOCAL.m31[0], LOCAL.m31[1], 0.11, 77 * Math.PI / 180, -0.9, 1); gal(LOCAL.m33[0], LOCAL.m33[1], 0.031, 0.95, 0.4, 0.95); gal(0, 0, 0.05, 0, 0, 1);
  g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; shotNoise(Bk, 0.14); }
VC[-10] = { c0: [-0.86, -1.1], an: [0, 0], paint: paintLocal, maxTiers: 3 };

/* ════ Laniakea, after the figure of Tully et al. (2014): the paths that galaxies follow once the expansion is taken away.
   Millions of light-years. A toy: flow down the gravitational potential of a few softened masses, placed roughly where the named
   clusters lie. The paths, the colours and the edge are computed from that toy, not from the Cosmicflows data. ════ */
const LAN = (() => { const A = [['Great Attractor', -163, -23, 3.4, 30, 1], ['Centaurus', -128, 46, 1.0, 34, 1], ['Virgo', -11, 52, 0.75, 30, 1], ['Hydra', -90, 74, 0.6, 30, 1], ['Pavo–Indus', -128, -100, 0.7, 36, 1],
    ['Perseus–Pisces', 196, -78, 3.0, 34, 0], ['Coma', 10, 300, 3.2, 34, 0], ['Shapley', -470, 250, 9, 44, 0], ['Hercules', -300, -330, 3.6, 40, 0], ['', 330, 250, 3, 40, 0], ['', 120, -360, 3, 40, 0]];
  const grad = (x, y) => { let gx = 0, gy = 0, rho = 0; for (const a of A) { const dx = x - a[1], dy = y - a[2], d2 = dx * dx + dy * dy + a[4] * a[4], d3 = d2 * Math.sqrt(d2); gx -= a[3] * dx / d3; gy -= a[3] * dy / d3; rho += a[3] * a[4] * 30 / d2; } return [gx, gy, rho]; };
  const run = (x, y, n = 500, h = 3) => { const p = [x, y]; for (let k = 0; k < n; k++) { const v = grad(x, y), m = Math.hypot(v[0], v[1]); if (m < 1e-9) break; x += v[0] / m * h; y += v[1] / m * h; p.push(x, y); let hit = false; for (const a of A) if (Math.hypot(x - a[1], y - a[2]) < 5) hit = true; if (hit) break; } return p; };
  const home = (x, y) => { const p = run(x, y, 420, 5), ex = p[p.length - 2], ey = p[p.length - 1]; let b = 0, bd = 1e9; A.forEach((a, k) => { const d = Math.hypot(ex - a[1], ey - a[2]); if (d < bd) { bd = d; b = k; } }); return A[b][5]; };
  let field = null; const N = 110, X0 = -520, Y0 = -420, SP = 8;
  const ensure = () => { if (field) return; field = new Float32Array(N * N); for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) field[j * N + i] = home(X0 + i * SP, Y0 + j * SP); boxBlur(field, N, N, 1, 2); };
  return { A, grad, run, ensure, inside: (x, y) => { const fx = clamp((x - X0) / SP, 0, N - 1.001), fy = clamp((y - Y0) / SP, 0, N - 1.001), i = fx | 0, j = fy | 0, u = fx - i, v = fy - j, k = j * N + i; return field[k] * (1 - u) * (1 - v) + field[k + 1] * u * (1 - v) + field[k + N] * (1 - u) * v + field[k + N + 1] * u * v; }, N, X0, Y0, SP, get field() { return field; } }; })();
function* paintLaniakea(Bk) { LAN.ensure(); yield; const g = world(Bk), s = Bk.s, r = M.rng(808);
  yield* pixels(Bk, (x, y, o) => { const rho = LAN.grad(x, y)[2], t = clamp(Math.log(1 + rho * 3.2) / 3.4, 0, 1), inn = ss(0.42, 0.58, LAN.inside(x, y)), n = fbm(x / 30, y / 30, 181, 3);
    const ramp = [[4, 8, 26], [10, 36, 92], [18, 96, 130], [60, 150, 96], [214, 196, 70], [236, 110, 50]], f = t * 4.999, k = f | 0, u = f - k, a = ramp[k], b = ramp[k + 1], dim = (0.42 + 0.58 * inn) * (0.82 + 0.36 * n); o[0] = lerp(a[0], b[0], u) * dim; o[1] = lerp(a[1], b[1], u) * dim; o[2] = lerp(a[2], b[2], u) * (0.55 + 0.45 * inn) * (0.82 + 0.36 * n); }, Bk.preview ? 400 : 48);
  g.lineCap = 'round'; g.lineJoin = 'round'; g.globalCompositeOperation = 'lighter'; const NL2 = Bk.preview ? 260 : 900;
  for (let k = 0; k < NL2; k++) { const x = -520 + r() * 900, y = -420 + r() * 800, inn = LAN.inside(x, y) > 0.5, p = LAN.run(x, y, 400, 3.2); if (p.length < 12) continue; g.strokeStyle = inn ? 'rgba(255,255,255,0.3)' : 'rgba(120,160,255,0.16)'; g.lineWidth = 1.1 / s; strokeP(g, p); }
  for (let k = 0; k < 9000; k++) { const x = Bk.x0 + r() * Bk.w, y = Bk.y0 + r() * Bk.w, rho = LAN.grad(x, y)[2]; if (r() > rho * 0.9) continue; pt(g, x, y, 0.6 / s, [255, 250, 236], 0.4 + 0.5 * r()); }
  /* the edge of the basin: where the paths part company */
  g.globalCompositeOperation = 'source-over'; g.strokeStyle = 'rgba(255,150,40,0.95)'; g.lineWidth = 2.6 / s; const F = LAN.field, N = LAN.N, sp = LAN.SP, at = (i, j) => F[j * N + i] - 0.5, ip = (a, b) => a / (a - b); g.beginPath();
  for (let j = 0; j < N - 1; j++) for (let i = 0; i < N - 1; i++) { const a = at(i, j), b = at(i + 1, j), c = at(i + 1, j + 1), d = at(i, j + 1), pts = []; if ((a > 0) !== (b > 0)) pts.push([i + ip(a, b), j]); if ((b > 0) !== (c > 0)) pts.push([i + 1, j + ip(b, c)]); if ((d > 0) !== (c > 0)) pts.push([i + ip(d, c), j + 1]); if ((a > 0) !== (d > 0)) pts.push([i, j + ip(a, d)]);
    for (let k = 0; k + 1 < pts.length; k += 2) { g.moveTo(LAN.X0 + pts[k][0] * sp, LAN.Y0 + pts[k][1] * sp); g.lineTo(LAN.X0 + pts[k + 1][0] * sp, LAN.Y0 + pts[k + 1][1] * sp); } } g.stroke(); shotNoise(Bk, 0.1); }
VC[-11] = { c0: [-96, 6], an: [0, 0], paint: paintLaniakea, maxTiers: 2 };

/* ════ The cosmic web as a redshift survey draws it: one dot for each galaxy, in two fans either side of the Milky Way's disc.
   Billions of light-years (comoving). The dots follow a ridged noise, not a survey. ════ */
const WEB = { dens: (x, y) => { const a = 1 - Math.abs(2 * vnoise(x * 3.3 + 5, y * 3.3 + 9, 201) - 1), b = 1 - Math.abs(2 * vnoise(x * 2.05 + y * 2.0 + 2, y * 2.05 - x * 2.0, 211) - 1), m = Math.max(a, b * 0.97); return m < 0.6 ? 0.012 : Math.pow(m, 10) * (0.45 + 1.1 * vnoise(x * 8, y * 8, 221)) + 0.012; }, fan: 0.74,
  zMax: 3.5, rMax: EV.COSMO.chi(3.5) / K.Gly, tracer: r => r < 4.6 ? 0 : r < 10.2 ? 1 : r < 14.9 ? 2 : 3, COL: [[255, 238, 190], [255, 150, 92], [120, 196, 255], [150, 236, 170]] };
function* paintWeb(Bk) { const g = world(Bk), s = Bk.s, r = M.rng(900 + Math.round(Math.log(Bk.w) * 11)), want = Bk.preview ? 26000 : SMALL ? 90000 : 170000; fillAll(Bk, '#020308'); g.globalCompositeOperation = 'lighter';
  const pr = 0.55 / s; let made = 0;
  for (let k = 0; k < want * 60 && made < want; k++) { const x = Bk.x0 + r() * Bk.w, y = Bk.y0 + r() * Bk.w, d = Math.hypot(x, y); if (d > WEB.rMax || Math.abs(Math.atan2(x, Math.abs(y))) > WEB.fan) continue;
    const t = WEB.tracer(d), sel = t === 3 ? 0.16 : t === 2 ? 0.5 * ss(15.4, 13, d) + 0.3 : t === 1 ? 0.85 : 1; if (r() > sel) continue; if (r() > WEB.dens(x, y) * 1.5) continue; pt(g, x, y, pr, WEB.COL[t], 0.45 + 0.5 * r()); made++; if (made % 30000 === 0) yield; }
  g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; }
VC[-12] = { paint: paintWeb, maxTiers: 2 };

/* ════ A deep field in the style of the James Webb Space Telescope's camera: every smudge a galaxy, a few stars with six long
   spikes and two short. Arbitrary units; this is a view of the sky, not a slice of space. ════ */
const DEEP = (() => { const r = M.rng(1444), gs = []; for (let k = 0; k < 3400; k++) { const t = r(), size = 0.006 * Math.exp(2.4 * r() * r() * (t < 0.08 ? 1.5 : 1)), kind = t < 0.1 ? 0 : t < 0.42 ? 1 : t < 0.8 ? 2 : 3;
    gs.push({ x: (r() * 2 - 1) * 6.4, y: (r() * 2 - 1) * 6.4, size: kind === 3 ? 0.006 + 0.006 * r() : size * (kind === 0 ? 2.4 : 1), q: 0.25 + 0.75 * r(), pa: r() * Math.PI, kind, b: 0.35 + 0.65 * r(), h: r() }); }
  const stars = []; for (let k = 0; k < 9; k++) stars.push({ x: (r() * 2 - 1) * 5.5, y: (r() * 2 - 1) * 5.5, m: 0.25 + 0.75 * r() * r() }); stars[0] = { x: -1.9, y: -1.25, m: 1 };
  return { gs, stars, mark: { x: 0.92, y: 0.46 } }; })();
function* paintDeep(Bk) { const g = world(Bk), s = Bk.s; fillAll(Bk, '#02030a'); g.globalCompositeOperation = 'lighter';
  const COLK = [[196, 214, 255], [255, 232, 196], [255, 176, 110], [255, 92, 60]];
  for (const G of DEEP.gs) { if (G.x < Bk.x0 - 0.3 || G.x > Bk.x0 + Bk.w + 0.3 || G.y < Bk.y0 - 0.3 || G.y > Bk.y0 + Bk.w + 0.3) continue; const c = COLK[G.kind].map((v, k) => clamp(v + (G.h - 0.5) * (k === 2 ? 70 : 30), 0, 255) | 0), R = Math.max(G.size, 0.9 / s);
    g.save(); g.translate(G.x, G.y); g.rotate(G.pa); g.scale(1, G.q); dot(g, 0, 0, R * 2.4, c, G.b); if (G.kind === 0 && R * s > 3) { for (const sg of [-1, 1]) for (let k = 1; k < 7; k++) { const a = k * 0.62, d = R * 0.34 * k; dot(g, sg * Math.cos(a) * d, sg * Math.sin(a) * d, R * (1.1 - 0.1 * k), [176, 200, 255], G.b * 0.34); } dot(g, 0, 0, R * 0.9, [255, 236, 200], G.b); } g.restore(); }
  dot(g, DEEP.mark.x, DEEP.mark.y, Math.max(0.016, 1.6 / s), [255, 70, 50], 0.9);
  for (const S of DEEP.stars) { const L = 0.75 * S.m; dot(g, S.x, S.y, 0.3 * S.m, [220, 232, 255], 0.9); dot(g, S.x, S.y, 0.07 * S.m, [255, 255, 255], 1);
    for (let k = 0; k < 8; k++) { const six = k < 6, a = six ? Math.PI / 2 + k * Math.PI / 3 : (k - 6) * Math.PI, len = six ? L : L * 0.36, gr = g.createLinearGradient(S.x, S.y, S.x + Math.cos(a) * len, S.y + Math.sin(a) * len); gr.addColorStop(0, 'rgba(226,236,255,' + (six ? 0.85 : 0.5) + ')'); gr.addColorStop(0.3, 'rgba(206,222,255,0.3)'); gr.addColorStop(1, 'rgba(160,190,255,0)'); g.globalAlpha = 1; g.strokeStyle = gr; g.lineWidth = Math.max(1.1 / s, 0.012 * S.m); g.beginPath(); g.moveTo(S.x, S.y); g.lineTo(S.x + Math.cos(a) * len, S.y + Math.sin(a) * len); g.stroke(); } }
  g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; yield; bloom(Bk, Bk.px / 300, 0.35); shotNoise(Bk, 0.2); }
VC[-13] = { paint: paintDeep, maxTiers: 1 };

/* ════ The microwave sky in the style of the Planck maps: the whole sky unrolled into an ellipse (Mollweide's projection).
   The pattern is a Gaussian random sky drawn from the model's spectrum by summing spherical harmonics. The map is 2 units wide. ════ */
const CMBMAP = { sky: null, Lmax: 0 };
const PLANCKCOL = [[0, 0, 150], [0, 60, 255], [0, 170, 255], [150, 226, 255], [255, 240, 218], [255, 190, 40], [255, 90, 0], [150, 10, 0]];
function* paintCMB(Bk) { const n = Bk.px, Lmax = Bk.preview ? 72 : SMALL ? 230 : 360, nth = Bk.preview ? 96 : SMALL ? 300 : 460, nph = nth * 2;
  if (!CMBMAP.sky || CMBMAP.Lmax < Lmax) { CMBMAP.sky = EV.cmbSky(Lmax, 2018); CMBMAP.Lmax = Lmax; rigDirty = true; } const sky = CMBMAP.sky.Lmax === Lmax ? CMBMAP.sky : EV.cmbSky(Lmax, 2018); yield;
  const T = new Float32Array(nth * nph), row = new Float64Array(nph); for (let j = 0; j < nth; j++) { sky.ring((j + 0.5) / nth * Math.PI, nph, row); T.set(row, j * nph); if (j % 6 === 5) yield; }
  const S2 = Math.SQRT2; yield* pixels(Bk, (x, y, o) => { const X = x * 2 * S2, Y = y * 2 * S2, e = X * X / 8 + Y * Y / 2; if (e > 1) { o[0] = 3; o[1] = 4; o[2] = 7; return; } const th = Math.asin(clamp(Y / S2, -1, 1)), lat = Math.asin(clamp((2 * th + Math.sin(2 * th)) / Math.PI, -1, 1)), lon = Math.PI * X / (2 * S2 * Math.max(1e-6, Math.cos(th)));
    const fj = clamp((Math.PI / 2 + lat) / Math.PI * nth - 0.5, 0, nth - 1.001), fi = ((lon + Math.PI) / TAU * nph - 0.5 + nph) % nph, j = fj | 0, i = fi | 0, i2 = (i + 1) % nph, u = fi - i, v = fj - j, t = (T[j * nph + i] * (1 - u) + T[j * nph + i2] * u) * (1 - v) + (T[(j + 1) * nph + i] * (1 - u) + T[(j + 1) * nph + i2] * u) * v;
    const f = clamp((t + 330) / 660, 0, 0.9999) * 7, k = f | 0, w = f - k, a = PLANCKCOL[k], b = PLANCKCOL[k + 1], edge = ss(1, 0.985, e); o[0] = lerp(a[0], b[0], w) * edge; o[1] = lerp(a[1], b[1], w) * edge; o[2] = lerp(a[2], b[2], w) * edge; }, Bk.preview ? 400 : 64); }
VC[-14] = { paint: paintCMB, maxTiers: 0 };

/* ════ Everything that can be seen, as a slice with us at the middle. Billions of light-years, comoving. The texture is not to scale. ════ */
const HOR = { R: EV.COSMO.horizon / K.Gly, cmb: EV.LAST.chi / K.Gly, dark: EV.COSMO.chi(20) / K.Gly };
function* paintHorizon(Bk) { yield* pixels(Bk, (x, y, o, q) => { const r = Math.hypot(x, y), web = Math.pow(Math.max(1 - Math.abs(2 * vnoise(x * 0.95 + 40, y * 0.95 + 40, 231) - 1), 0.97 * (1 - Math.abs(2 * vnoise(x * 0.6 + y * 0.55 + 4, y * 0.6 - x * 0.55, 232) - 1))), 7) * (0.5 + 1.1 * fbm(x * 2.1, y * 2.1, 233, 2)), sm = fbm(x * 0.2, y * 0.2, 237, 3);
    if (r > HOR.R) { const v = 8 + 70 * web, hatch = ((x + y) / (q * 11) % 1 + 1) % 1 < 0.18 ? 1 : 0; o[0] = v * 0.5 + 52 * hatch; o[1] = v * 0.45 + 26 * hatch; o[2] = v * 0.62 + 14 * hatch; return; }
    const young = ss(0.2, 0.74, r / HOR.R), clump = lerp(1, 0.12, young), dawn = ss(HOR.dark - 3.5, HOR.dark + 2, r), v = (8 + 250 * web * clump + 26 * young) * (1 - 0.82 * dawn) + 10 * sm; let cr = v * lerp(0.92, 1.0, young), cg = v * lerp(0.95, 0.74, young), cb = v * lerp(1.05, 0.6, young);
    const glow = ss(HOR.cmb - 3.2, HOR.cmb, r); cr += 250 * glow * glow; cg += 150 * glow * glow * glow; cb += 60 * Math.pow(glow, 5);
    if (r > HOR.cmb) { const t = (r - HOR.cmb) / (HOR.R - HOR.cmb), sp = fbm(Math.atan2(y, x) * 70, r * 9, 239, 3); cr = 255; cg = lerp(206 + 40 * sp, 250, t); cb = lerp(120 + 60 * sp, 236, t); }
    const edge = Math.exp(-Math.pow((r - HOR.R) / (q * 1.3), 2)); o[0] = cr + 255 * edge; o[1] = cg + 255 * edge; o[2] = cb + 255 * edge; }, Bk.preview ? 400 : 48); shotNoise(Bk, 0.07); }
VC[-15] = { paint: paintHorizon, maxTiers: 0 };

/* ════ Inside one atom: the electrons of a potassium ion as a cloud of chance. Picometres; the nucleus is at the origin.
   Brightness follows the electron density summed along the line of sight, from Slater-type shells; the speckle is a dot-density
   rendering of the same numbers. Nothing can photograph this. ════ */
const ATOM = (() => { const sh = EV.KION.shells.map(s => ({ n: s.n, N: s.N, b: s.scale * 1e12 })), fact = n => { let f = 1; for (let k = 2; k <= n; k++) f *= k; return f; };
  const rho = r => { let v = 0; for (const s of sh) v += s.N * Math.pow(r, 2 * s.n - 2) * Math.exp(-r / s.b) / (4 * Math.PI * fact(2 * s.n) * Math.pow(s.b, 2 * s.n + 1)); return v; };
  const NB = 260, lb0 = Math.log(1e-3), lb1 = Math.log(700), tab = new Float64Array(NB);
  for (let k = 0; k < NB; k++) { const b = Math.exp(lb0 + (lb1 - lb0) * k / (NB - 1)); let s = 0, z0 = 0; for (let m = 0; m < 400; m++) { const z1 = (Math.exp(m / 400 * Math.log(1 + 900 / 0.02)) - 1) * 0.02, zm = (z0 + z1) / 2; s += rho(Math.hypot(b, zm)) * (z1 - z0); z0 = z1; } tab[k] = 2 * s; }
  const column = b => { const f = clamp((Math.log(Math.max(b, 1e-3)) - lb0) / (lb1 - lb0), 0, 1) * (NB - 1), k = Math.min(NB - 2, f | 0); return lerp(tab[k], tab[k + 1], f - k); };
  return { rho, column, c0: column(1e-3) }; })();
function* paintAtom(Bk) { const R = EV.KION.R * 1e12, q0 = Bk.w / Bk.px, cmax = Math.log(ATOM.column(0.02)), cmin = Math.log(ATOM.column(150)), rr = M.rng(71 + Math.round(Math.log(Bk.w) * 13));
  yield* pixels(Bk, (x, y, o, q) => { const b = Math.hypot(x, y), col = Math.log(ATOM.column(b)), u = clamp((col - cmin) / (cmax - cmin), 0, 1), lum = Math.pow(u, 1.6), spk = rr() < 0.08 + 1.5 * u ? 0.7 + 0.6 * rr() : 0.14;
    const v = Math.min(1, lum * 1.5 * (0.25 + 0.75 * spk)), nuc = Math.exp(-Math.pow(b / Math.max(R, q * 0.9), 2)); o[0] = 6 + 60 * v * v + 255 * nuc; o[1] = 9 + 186 * v + 230 * nuc; o[2] = 16 + 250 * Math.pow(v, 0.8) + 200 * nuc; }, Bk.preview ? 400 : 64); }
VC[7] = { paint: paintAtom, maxTiers: 5 };

/* ════ The nucleus of potassium-39: nineteen protons and twenty neutrons. Femtometres. A drawing of a model: nucleons are not
   little balls, and nothing has imaged a nucleus. The soft edge follows the density that electron scattering measures. ════ */
const NUC = (() => { const r = M.rng(39), R = EV.KION.R * 1e15, pts = [{ x: 0.9, y: -0.6, z: 2.3, p: 1 }]; let sep = 1.72, tries = 0;
  while (pts.length < 39) { const x = (r() * 2 - 1) * R, y = (r() * 2 - 1) * R, z = (r() * 2 - 1) * R; if (++tries > 30000) { sep *= 0.97; tries = 0; } if (Math.hypot(x, y, z) > R - 0.55) continue; let ok = true; for (const p of pts) if (Math.hypot(x - p.x, y - p.y, z - p.z) < sep) { ok = false; break; } if (ok) pts.push({ x, y, z, p: 0 }); }
  const order = pts.slice(1).sort((a, b) => hash2(a.x * 97 | 0, a.y * 89 | 0, 3) - hash2(b.x * 97 | 0, b.y * 89 | 0, 3)); order.forEach((p, k) => { p.p = k < 18 ? 1 : 0; }); pts.sort((a, b) => a.z - b.z); return { pts, R, an: [0.9, -0.6] }; })();
function* paintNucleus(Bk) { const g = world(Bk), s = Bk.s; fillAll(Bk, '#03040a'); const R = NUC.R, halo = g.createRadialGradient(0, 0, 0, 0, 0, R * 1.9); halo.addColorStop(0, 'rgba(120,140,255,0.34)'); halo.addColorStop(0.5, 'rgba(110,120,240,0.2)'); halo.addColorStop(0.72, 'rgba(90,90,200,0.07)'); halo.addColorStop(1, 'rgba(60,60,160,0)'); g.fillStyle = halo; g.fillRect(-R * 2, -R * 2, R * 4, R * 4);
  for (const p of NUC.pts) { const rad = 0.98 * (1 + p.z / 60), dep = 0.5 + 0.5 * (p.z + R) / (2 * R), c = p.p ? [240, 112, 84] : [112, 150, 214], gr = g.createRadialGradient(p.x - rad * 0.3, p.y - rad * 0.34, rad * 0.05, p.x, p.y, rad * 1.18);
    gr.addColorStop(0, rgba(c.map(v => Math.min(255, v * 1.25 + 40) | 0), 0.96 * dep + 0.04)); gr.addColorStop(0.45, rgba(c.map(v => v * (0.4 + 0.6 * dep) | 0), 0.95)); gr.addColorStop(0.82, rgba(c.map(v => v * 0.3 * dep | 0), 0.8)); gr.addColorStop(1, rgba(c.map(v => v * 0.2 | 0), 0)); g.fillStyle = gr; g.beginPath(); g.arc(p.x, p.y, rad * 1.18, 0, TAU); g.fill(); }
  yield; const b = soft(Bk.c, Math.max(0.8, 0.07 * s)); const G = Bk.c.getContext('2d'); G.setTransform(1, 0, 0, 1, 0, 0); G.globalAlpha = 0.6; G.drawImage(b, 0, 0); G.globalAlpha = 1; shotNoise(Bk, 0.16); }
VC[8] = { an: NUC.an, paint: paintNucleus, maxTiers: 2 };
VC[9] = { an: [-268, 172] };

/* ════ A collision as a detector draws it: the view along the beam. Tracks curl in the magnet; towers show energy caught further out.
   Generated: a made-up event with two jets and a muon, not data. The picture is of a machine about 15 m across; what it measures
   is 10²⁰ times smaller. ════ */
function makeEvent(seed) { const r = M.rng(4000 + seed), tracks = [], towers = [], phiJ = r() * TAU, jets = [phiJ, phiJ + Math.PI + (r() - 0.5) * 0.5];
  jets.forEach((ph, j) => { const n = 9 + (r() * 9 | 0); for (let k = 0; k < n; k++) tracks.push({ phi: ph + M.gauss(r) * 0.11, pt: 4 + 90 * Math.pow(r(), 2.2), q: r() < 0.5 ? -1 : 1, jet: 1 }); for (let k = 0; k < 7; k++) towers.push({ phi: ph + M.gauss(r) * 0.09, e: 0.25 + 0.75 * r() * (k < 3 ? 1 : 0.4), had: k % 2, jet: j }); });
  const soft2 = 34 + (r() * 30 | 0); for (let k = 0; k < soft2; k++) tracks.push({ phi: r() * TAU, pt: 0.45 + 2.6 * r() * r(), q: r() < 0.5 ? -1 : 1, jet: 0 }); for (let k = 0; k < 22; k++) towers.push({ phi: r() * TAU, e: 0.04 + 0.12 * r(), had: k % 2, jet: -1 });
  const mu = { phi: phiJ + 1.2 + r() * 0.9, pt: 38 + 40 * r(), q: r() < 0.5 ? -1 : 1 }; return { seed, tracks, towers, jets, mu, pt: 900 + Math.round(2600 * r()) }; }
let EVENT = makeEvent(1);
function* paintEvent(Bk) { const g = world(Bk), s = Bk.s, U0 = EV.LV(10).fov / EV.LV(10).unit * 0.46 * Math.min(1, 1 / (ASP * 0.88)); fillAll(Bk, '#020304'); g.lineCap = 'round';
  const RT = 0.3 * U0, RE = [0.36 * U0, 0.44 * U0], RH = [0.46 * U0, 0.62 * U0], RM = [0.72, 0.8, 0.89, 0.98].map(v => v * U0), lw = v => Math.max(v / s, v * U0 / 900);
  g.strokeStyle = 'rgba(120,150,170,0.34)'; g.lineWidth = lw(1); for (const rr of [0.04, 0.07, 0.11, 0.16, 0.21, 0.26, 0.3]) { g.beginPath(); g.arc(0, 0, rr * U0, 0, TAU); g.stroke(); }
  g.fillStyle = 'rgba(60,150,90,0.16)'; g.beginPath(); g.arc(0, 0, RE[1], 0, TAU); g.arc(0, 0, RE[0], 0, TAU, true); g.fill(); g.fillStyle = 'rgba(60,110,190,0.15)'; g.beginPath(); g.arc(0, 0, RH[1], 0, TAU); g.arc(0, 0, RH[0], 0, TAU, true); g.fill();
  g.strokeStyle = 'rgba(160,170,180,0.4)'; g.lineWidth = lw(2.4); g.beginPath(); g.arc(0, 0, 0.66 * U0, 0, TAU); g.stroke();
  for (const rm of RM) for (let k = 0; k < 12; k++) { const a = k * TAU / 12 + 0.26; g.save(); g.rotate(a); g.fillStyle = 'rgba(150,60,50,0.3)'; g.strokeStyle = 'rgba(210,110,90,0.5)'; g.lineWidth = lw(0.9); const hw = rm * Math.tan(TAU / 24) * 0.9; g.fillRect(rm, -hw, 0.035 * U0, 2 * hw); g.strokeRect(rm, -hw, 0.035 * U0, 2 * hw); g.restore(); }
  g.globalCompositeOperation = 'lighter';
  for (const t of EVENT.towers) { const r0 = t.had ? RH[0] : RE[0], len = t.e * (t.had ? 0.3 : 0.2) * U0, wd = 0.022; g.save(); g.rotate(t.phi); g.fillStyle = t.had ? 'rgba(90,150,255,0.9)' : 'rgba(110,236,130,0.9)'; g.fillRect(r0, -r0 * wd, len, 2 * r0 * wd); g.restore(); }
  /* a track of transverse momentum pT curls with radius ∝ pT; here 1 GeV ↔ 0.09 of the tracker */
  for (const t of EVENT.tracks) { const rc = t.pt * 0.09 * RT * 2.2, cxp = -t.q * Math.sin(t.phi) * rc, cyp = t.q * Math.cos(t.phi) * rc, a0 = Math.atan2(-cyp, -cxp), reach = 2 * rc < RT ? Math.PI * 1.9 : 2 * Math.asin(Math.min(1, RT / (2 * rc)));
    g.strokeStyle = t.jet ? 'rgba(255,220,120,0.85)' : 'rgba(150,226,170,0.5)'; g.lineWidth = lw(t.jet ? 1.3 : 0.9); g.beginPath(); g.arc(cxp, cyp, rc, a0, a0 + t.q * reach, t.q < 0); g.stroke(); }
  { const t = EVENT.mu, x1 = Math.cos(t.phi) * 1.04 * U0, y1 = Math.sin(t.phi) * 1.04 * U0; g.strokeStyle = 'rgba(255,90,70,0.95)'; g.lineWidth = lw(1.8); g.beginPath(); g.moveTo(0, 0); g.quadraticCurveTo(Math.cos(t.phi + t.q * 0.05) * 0.5 * U0, Math.sin(t.phi + t.q * 0.05) * 0.5 * U0, x1, y1); g.stroke(); for (const rm of RM) dot(g, Math.cos(t.phi) * (rm + 0.017 * U0), Math.sin(t.phi) * (rm + 0.017 * U0), 0.02 * U0, [255, 120, 90], 1); }
  dot(g, 0, 0, 0.03 * U0, [255, 255, 255], 1); g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; yield; bloom(Bk, Bk.px / 240, 0.4); }
VC[10] = { paint: paintEvent, maxTiers: 0 };

/* ════ The Planck length, imagined three ways. In Planck lengths. None of these pictures rests on any measurement. ════ */
const worley = (x, y, s) => { const ix = Math.floor(x), iy = Math.floor(y); let f1 = 9, f2 = 9; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const dx = ix + i + hash2(ix + i, iy + j, s) - x, dy = iy + j + hash2(ix + i, iy + j, s + 1) - y, d = dx * dx + dy * dy; if (d < f1) { f2 = f1; f1 = d; } else if (d < f2) f2 = d; } return [Math.sqrt(f1), Math.sqrt(f2)]; };
const NET = (() => { const r = M.rng(1616), nodes = [], edges = [], ext = 220; for (let j = -ext; j <= ext; j += 1) for (let i = -ext; i <= ext; i += 1) { if (Math.abs(i) > 14 || Math.abs(j) > 14) { if ((i % 4) || (j % 4)) continue; } nodes.push([i * 1.9 + (r() - 0.5) * 1.3, j * 1.9 + (r() - 0.5) * 1.3]); }
  /* link each node to its nearest few neighbours (a brute search over a bucket grid) */
  const B = new Map(), key = (a, b) => a * 4096 + b, cs = 8; nodes.forEach((p, k) => { const kk = key(Math.floor(p[0] / cs), Math.floor(p[1] / cs)); if (!B.has(kk)) B.set(kk, []); B.get(kk).push(k); });
  nodes.forEach((p, k) => { const cand = []; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) { const L = B.get(key(Math.floor(p[0] / cs) + a, Math.floor(p[1] / cs) + b)); if (L) for (const m of L) if (m !== k) cand.push([Math.hypot(nodes[m][0] - p[0], nodes[m][1] - p[1]), m]); } cand.sort((u, v) => u[0] - v[0]); for (const c of cand.slice(0, 3)) if (c[1] > k || !cand.slice(0, 3).some(() => false)) { const a = Math.min(k, c[1]), b = Math.max(k, c[1]); edges.push([a, b]); } });
  const seen = new Set(), E2 = []; for (const e of edges) { const kk = e[0] * 1e6 + e[1]; if (seen.has(kk)) continue; seen.add(kk); E2.push([e[0], e[1], 1 + (hash2(e[0], e[1], 5) * 3 | 0)]); } return { nodes, edges: E2 }; })();
function* paintPlanck(Bk) { const g = world(Bk), s = Bk.s, mode = st.cand;
  if (mode === 0) { yield* pixels(Bk, (x, y, o, q) => { const wx = x + 0.5 * (fbm(x * 0.6, y * 0.6, 251, 3) - 0.5), wy = y + 0.5 * (fbm(x * 0.6 + 9, y * 0.6, 253, 3) - 0.5), a = worley(wx * 0.62, wy * 0.62, 261), b = worley(wx * 1.9 + 4, wy * 1.9, 263), big = fbm(x / 26, y / 26, 267, 3);
      const fine = ss(0.55, 2.2, 1 / (q * 1.9)), e1 = Math.exp(-Math.pow((a[1] - a[0]) / (0.07 + q * 0.5), 2)), e2 = Math.exp(-Math.pow((b[1] - b[0]) / (0.11 + q * 1.6), 2)) * fine, bub = Math.pow(1 - clamp(a[0] / 0.85, 0, 1), 1.6), sheen = 0.5 + 0.5 * Math.sin(a[0] * 9 + b[0] * 5 + big * 6);
      const lum = e1 * 0.95 + e2 * 0.42 + bub * 0.2 * sheen; o[0] = 10 + 26 * bub + 255 * lum * (0.86 + 0.14 * sheen); o[1] = 8 + 22 * bub * sheen + 214 * lum; o[2] = 30 + 70 * bub + 150 * lum + 60 * e2; }, Bk.preview ? 400 : 48); shotNoise(Bk, 0.1); return; }
  fillAll(Bk, '#05060f'); if (mode === 2) { g.lineCap = 'round'; g.globalCompositeOperation = 'lighter'; const x1 = Bk.x0 + Bk.w, y1 = Bk.y0 + Bk.w, N = NET.nodes, vis = p => p[0] > Bk.x0 - 4 && p[0] < x1 + 4 && p[1] > Bk.y0 - 4 && p[1] < y1 + 4;
    for (const e of NET.edges) { const a = N[e[0]], b = N[e[1]]; if (!vis(a) && !vis(b)) continue; const c = e[2] === 1 ? 'rgba(90,214,190,' : e[2] === 2 ? 'rgba(130,190,255,' : 'rgba(236,206,120,'; g.strokeStyle = c + '0.22)'; g.lineWidth = Math.max(5 / s, 0.1 * e[2] + 0.16); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); g.strokeStyle = c + '0.9)'; g.lineWidth = Math.max(1.2 / s, 0.035 * e[2]); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke(); }
    for (const p of N) if (vis(p)) dot(g, p[0], p[1], Math.max(3 / s, 0.2), [240, 246, 255], 1); g.globalAlpha = 1; g.globalCompositeOperation = 'source-over'; } }
VC[12] = { paint: paintPlanck, maxTiers: 2 };
