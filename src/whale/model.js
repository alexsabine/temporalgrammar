/* The Humpback: the model. No drawing in this file; it runs in node for the tests.
   Whale biology, schooling and ocean acoustics are from the literature (see the notes on the page).
   The only thing taken from CRR is a timing form: a non-negative count since the last event,
   a fixed threshold, and a restart. It times the breaths, the start of each song unit and the breach. */
(function (root) {
'use strict';
const DT = 1 / 60, TAU = Math.PI * 2;
const clamp = (x, a, b) => x < a ? a : x > b ? b : x;
const lerp = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { x = clamp((x - a) / (b - a), 0, 1); return x * x * (3 - 2 * x); };
const angd = (a, b) => { let d = (a - b) % TAU; if (d > Math.PI) d -= TAU; if (d < -Math.PI) d += TAU; return d; };
function rng32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

/* ---------- sound in the sea ---------- */
/* Mackenzie (1981), nine-term equation. T in deg C, S in parts per thousand, D depth in metres. Speed in m/s. */
function soundSpeed(T, S, D) {
  return 1448.96 + 4.591 * T - 5.304e-2 * T * T + 2.374e-4 * T * T * T + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D * D
    - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D * D * D;
}
/* A simple temperature profile: a mixed layer, then an exponential thermocline down to cold deep water. A setting, not data. */
const SEA = { Tdeep: 2, mixed: 50, H: 400, bottom: 4500 };
function tempAt(z, Ts) { return SEA.Tdeep + (Ts - SEA.Tdeep) * Math.exp(-Math.max(0, z - SEA.mixed) / SEA.H); }
function speedAt(z, Ts, S) { return soundSpeed(tempAt(z, Ts), S, z); }
function channel(Ts, S) {
  let depth = 0, speed = Infinity;
  for (let z = 0; z <= SEA.bottom; z += 5) { const c = speedAt(z, Ts, S); if (c < speed - 1e-9) { speed = c; depth = z; } }
  return { depth, speed, surface: speedAt(0, Ts, S) };
}
/* Rays by Snell's law in a layered sea: a ray turns towards slower water. Angles in degrees from level, positive downwards. */
function rays(Ts, S, src, angles, range, ds) {
  ds = ds || 50; range = range || 150000; const out = [];
  for (const a0 of angles) {
    let r = 0, z = src, th = a0 * Math.PI / 180, top = 0, bot = 0, time = 0; const pts = [0, z], tt = [0];
    for (let n = 0; r < range && n < 20000; n++) {
      const c = speedAt(z, Ts, S), g = (speedAt(z + 1, Ts, S) - speedAt(z - 1, Ts, S)) / 2;
      th -= Math.cos(th) / c * g * ds; time += ds / c;
      r += Math.cos(th) * ds; z += Math.sin(th) * ds;
      if (z < 0) { z = -z; th = -th; top++; }
      if (z > SEA.bottom) { bot++; pts.push(r, SEA.bottom); tt.push(time); break; }
      if (n % 4 === 3) { pts.push(r, z); tt.push(time); }
    }
    out.push({ angle: a0, pts, tt, top, bot, time });
  }
  return out;
}

/* ---------- the song ---------- */
/* Units: single sounds. contour = [seconds, Hz] points. These were composed for this page after published
   descriptions of unit types; they are not transcribed from a recording. */
const UNITS = [
  { id: 'lowmoan', name: 'low moan', dur: 3.4, loud: 0.95, contour: [[0, 112], [0.8, 128], [2.4, 122], [3.4, 104]], timbre: 'rich', vib: 0.010, form: [280, 760] },
  { id: 'moan', name: 'moan', dur: 2.5, loud: 0.9, contour: [[0, 150], [0.9, 186], [1.9, 178], [2.5, 158]], timbre: 'rich', vib: 0.012, form: [340, 900] },
  { id: 'upsweep', name: 'upsweep', dur: 1.4, loud: 0.8, contour: [[0, 210], [0.5, 290], [1.1, 560], [1.4, 660]], timbre: 'reed', vib: 0.006, form: [520, 1300] },
  { id: 'downsweep', name: 'downsweep', dur: 1.6, loud: 0.8, contour: [[0, 920], [0.4, 760], [1.1, 380], [1.6, 250]], timbre: 'reed', vib: 0.006, form: [600, 1500] },
  { id: 'cry', name: 'cry', dur: 1.8, loud: 0.85, contour: [[0, 420], [0.6, 980], [1.0, 1150], [1.8, 690]], timbre: 'reed', vib: 0.014, form: [900, 2100] },
  { id: 'chirp', name: 'chirp', dur: 0.6, loud: 0.6, contour: [[0, 1750], [0.25, 2500], [0.6, 3250]], timbre: 'pure', vib: 0, form: [2400, 3400] },
  { id: 'growl', name: 'growl', dur: 2.2, loud: 0.85, contour: [[0, 118], [1.2, 106], [2.2, 100]], timbre: 'rich', vib: 0.004, form: [240, 620], pulse: 27 },
  { id: 'whoop', name: 'whoop', dur: 1.0, loud: 0.75, contour: [[0, 300], [0.45, 880], [1.0, 520]], timbre: 'reed', vib: 0.008, form: [700, 1700] },
  { id: 'squeak', name: 'squeak', dur: 0.8, loud: 0.55, contour: [[0, 2400], [0.35, 3600], [0.8, 2800]], timbre: 'pure', vib: 0.01, form: [2800, 3800] }
];
const UNIT = {}; UNITS.forEach((u, i) => { u.index = i; UNIT[u.id] = u; });
/* Phrases of units; a theme is one phrase repeated; the themes come in a fixed order (Payne & McVay 1971). */
const THEMES = [
  { id: 'A', name: 'moans', phrase: ['lowmoan', 'moan', 'upsweep'], reps: 4, gap: 0.7, phraseGap: 0.9 },
  { id: 'B', name: 'sweeps and chirps', phrase: ['downsweep', 'chirp', 'chirp', 'chirp'], reps: 4, gap: 0.5, phraseGap: 0.9 },
  { id: 'C', name: 'growl and cries', phrase: ['growl', 'cry', 'cry'], reps: 3, gap: 0.7, phraseGap: 1.0 },
  { id: 'D', name: 'whoops', phrase: ['moan', 'whoop', 'whoop', 'squeak'], reps: 4, gap: 0.6, phraseGap: 1.1 }
];
const ORDER = ['A', 'B', 'C', 'D'];
/* The timing rule for the song. While a unit sounds nothing is counted. When it ends a count starts at zero and rises
   with time; when it reaches the rest that follows that unit, the next unit starts and the count is cleared. */
function newSinger() { return { th: 0, rep: 0, k: -1, left: 0, C: 0, thr: 0, songs: 0, n: 0 }; }
function singerStep(s) {
  if (s.left > 0) { s.left--; return null; }
  if (s.k >= 0) { s.C += DT; if (s.C < s.thr - 1e-9) return null; }
  s.C = 0; s.k++;
  let th = THEMES[s.th];
  if (s.k >= th.phrase.length) { s.k = 0; s.rep++; if (s.rep >= th.reps) { s.rep = 0; s.th++; if (s.th >= THEMES.length) { s.th = 0; s.songs++; } th = THEMES[s.th]; } }
  const u = UNIT[th.phrase[s.k]];
  s.left = Math.round(u.dur * 60) - 1;
  s.thr = th.gap + (s.k === th.phrase.length - 1 ? th.phraseGap : 0);
  s.n++;
  return { unit: u.id, index: u.index, theme: th.id, themeIndex: s.th, phrase: s.rep + 1, of: th.reps, k: s.k + 1, len: th.phrase.length, song: s.songs };
}
function buildSequence() {
  const s = newSinger(), seq = []; let f = 0;
  for (; f < 60 * 600; f++) { const e = singerStep(s); if (e) { if (e.song > 0) break; e.frame = f; e.t = f / 60; e.dur = UNIT[e.unit].dur; e.loud = UNIT[e.unit].loud; seq.push(e); } }
  return { seq, frames: f };
}
const SEQ = buildSequence();
const SONG = UNITS.slice();           /* the list of units ... */
SONG.units = UNITS; SONG.themes = THEMES; SONG.order = ORDER; SONG.sequence = SEQ.seq; SONG.frames = SEQ.frames; SONG.duration = SEQ.frames / 60;
SONG.pitchAt = function (u, t) { const c = (typeof u === 'string' ? UNIT[u] : u).contour; if (t <= c[0][0]) return c[0][1];
  for (let i = 1; i < c.length; i++) if (t <= c[i][0]) { const a = c[i - 1], b = c[i], k = (t - a[0]) / (b[0] - a[0]); return a[1] * Math.pow(b[1] / a[1], k); } return c[c.length - 1][1]; };
SONG.at = function (frame) { const f = ((frame % SEQ.frames) + SEQ.frames) % SEQ.frames; let lo = 0, hi = SEQ.seq.length - 1; while (lo < hi) { const m = (lo + hi + 1) >> 1; if (SEQ.seq[m].frame <= f) lo = m; else hi = m - 1; }
  const e = SEQ.seq[lo]; return { entry: e, sounding: f - e.frame < Math.round(e.dur * 60), into: (f - e.frame) / 60 }; };

/* ---------- the sea with whales and fish in it ---------- */
const MAXSPRAY = 4200, MAXBUB = 2600, MAXFOAM = 8;
function makeSim(opts) {
  opts = opts || {};
  const seed = opts.seed === undefined ? 7 : opts.seed;
  const R = rng32(seed * 7919 + 13), RP = rng32(seed * 104729 + 71);
  const NF = opts.nFish === undefined ? 420 : opts.nFish;
  const script = (opts.events || []).slice().sort((a, b) => a.at - b.at);
  const auto = opts.auto === undefined ? true : !!opts.auto;
  const S = { frame: 0, t: 0, whales: [], log: [], ev: [], foam: [], net: null, dives: [], eaten: 0, NF, DT,
    anchor: { x: 0, y: 0, z: 0, yaw: 0 }, school: { x: 0, y: -8, z: 0, pol: 0 }, home: { x: 0, y: -8, z: 0 } };

  function whale(role, L, x, y, z, yaw, Tdive, Cs, series) {
    return { role, L, x, y, z, yaw, pitch: 0, roll: 0, v: 1.6, vc: 1.8, yr: 0, ph: R() * TAU, amp: 0.5, bend: 0, arch: 0, gape: 0, gulp: 0,
      C: 0, Tdive, Cs, series, nb: 0, tb: 0, lastSeries: -1, state: 'dive', st: 0, B: 0, rb: 0, want: null, blowT: 0,
      vx: 0, vy: 0, vz: 0, wet: 0, splashed: false, pec: 0, singing: false, bn: null, peak: 0 };
  }
  const mum = whale('mother', 14, 30, -14, 6, 1.7, 75, 0.13, 3); mum.C = 0.55;
  const calf = whale('calf', 6, 27, -11, 2, 1.7, 34, 0.24, 2); calf.C = 0.3;
  const male = whale('singer', 13, -30, -24, -16, 0.6, 150, 0.07, 3); male.C = 0.2; male.v = 0.1;
  male.station = { x: -30, y: -24, z: -16 };
  S.whales.push(mum, calf, male);
  S.singer = newSinger(); S.unit = null; S.unitFrame = -1;

  /* fish */
  const F = { x: new Float32Array(NF), y: new Float32Array(NF), z: new Float32Array(NF), vx: new Float32Array(NF), vy: new Float32Array(NF), vz: new Float32Array(NF), gone: new Uint8Array(NF), fear: new Float32Array(NF), nb: new Int16Array(NF * 10), nc: new Uint8Array(NF), fw: new Int8Array(NF) };
  for (let i = 0; i < NF; i++) { const a = R() * TAU, r = 4.5 * Math.cbrt(R()), u = R() * 2 - 1, q = Math.sqrt(1 - u * u);
    F.x[i] = 12 + r * q * Math.cos(a); F.y[i] = -8 + r * u * 0.6; F.z[i] = r * q * Math.sin(a); F.vx[i] = 0.2 * (R() - 0.5); F.vy[i] = 0; F.vz[i] = 1.1 + 0.2 * (R() - 0.5); }
  S.fish = F;
  const HB = 2048, head = new Int32Array(HB), next = new Int32Array(Math.max(1, NF)), CELL = 1.3;
  const hsh = (ix, iy, iz) => ((ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791)) & (HB - 1);

  /* particles */
  const P = { n: MAXSPRAY, x: new Float32Array(MAXSPRAY), y: new Float32Array(MAXSPRAY), z: new Float32Array(MAXSPRAY), vx: new Float32Array(MAXSPRAY), vy: new Float32Array(MAXSPRAY), vz: new Float32Array(MAXSPRAY),
    life: new Float32Array(MAXSPRAY), max: new Float32Array(MAXSPRAY), size: new Float32Array(MAXSPRAY), kind: new Uint8Array(MAXSPRAY), at: 0 };
  const Bb = { n: MAXBUB, x: new Float32Array(MAXBUB), y: new Float32Array(MAXBUB), z: new Float32Array(MAXBUB), vy: new Float32Array(MAXBUB), ph: new Float32Array(MAXBUB), life: new Float32Array(MAXBUB), size: new Float32Array(MAXBUB), net: new Uint8Array(MAXBUB), at: 0 };
  S.spray = P; S.bubbles = Bb;
  function spray(kind, x, y, z, vx, vy, vz, life, size) { const i = P.at; P.at = (P.at + 1) % MAXSPRAY; P.kind[i] = kind; P.x[i] = x; P.y[i] = y; P.z[i] = z; P.vx[i] = vx; P.vy[i] = vy; P.vz[i] = vz; P.life[i] = life; P.max[i] = life; P.size[i] = size; }
  function bubble(x, y, z, vy, size, net) { const i = Bb.at; Bb.at = (Bb.at + 1) % MAXBUB; Bb.x[i] = x; Bb.y[i] = y; Bb.z[i] = z; Bb.vy[i] = vy; Bb.ph[i] = RP() * TAU; Bb.life[i] = 40; Bb.size[i] = size; Bb.net[i] = net ? 1 : 0; }
  function foam(x, z, r, grow, life, ring, s) { if (S.foam.length >= MAXFOAM) S.foam.shift(); const f = { x, z, r, grow, age: 0, life, ring: ring | 0, s }; S.foam.push(f); return f; }
  function log(type, w, extra) { const e = Object.assign({ frame: S.frame, type, whale: w ? w.role : null }, extra || {}); S.log.push(e); S.ev.push(e); }

  const fwd = w => { const cp = Math.cos(w.pitch); return [cp * Math.cos(w.yaw), Math.sin(w.pitch), cp * Math.sin(w.yaw)]; };
  /* a point given in the whale's own frame (lengths in body lengths; x forward, y up), roll ignored */
  function at(w, lx, ly) { const c = Math.cos(w.pitch), s = Math.sin(w.pitch), X = (lx * c - ly * s) * w.L, Y = (lx * s + ly * c) * w.L; return [w.x + X * Math.cos(w.yaw), w.y + Y, w.z + X * Math.sin(w.yaw)]; }
  const blowhole = w => at(w, 0.23, 0.062), snout = w => at(w, 0.4, 0);
  S.blowhole = blowhole; S.snout = snout; S.at = at;
  const free = w => w.state === 'dive' || w.state === 'shallow' || w.state === 'sound' || w.state === 'rise' || w.state === 'breathe';
  const busy = w => !free(w);

  function trigger(what) {
    if (what === 'breach') { if (free(mum)) { mum.state = 'br_dive'; mum.st = 0; mum.want = null; log('breach_start', mum); return true; } mum.want = 'breach'; return false; }
    if (what === 'blow') { if (mum.state === 'dive') { mum.C = Math.max(mum.C, 1); return true; } return false; }
    if (what === 'bubblenet') {
      if (S.net && S.net.phase !== 'done') return false;
      if (busy(mum) || busy(male)) { mum.want = 'bubblenet'; return false; }
      mum.want = null;
      const cx = S.school.x, cz = S.school.z, a0 = Math.atan2(mum.z - cz, mum.x - cx);
      S.net = { cx, cz, R: 9, phase: 'go', t: 0, ring: 0, ringR: 0, ringN: 0, foam: null, f0: S.frame, taken: 0 };
      [mum, male].forEach((w, i) => { w.state = 'bn_go'; w.st = 0; w.bn = { a: a0 + i * Math.PI, dir: 1 }; });
      log('net_start', mum, { x: cx, z: cz }); return true;
    }
    return false;
  }
  S.trigger = trigger;

  /* where each whale would like to go when nothing special is happening */
  function roam(w, o) {
    if (w === mum) {
      const Rr = 28 + 14 * Math.sin(S.t / 23), a = Math.atan2(w.z, w.x) + 0.55, tx = Rr * Math.cos(a), tz = Rr * Math.sin(a);
      o.yawT = Math.atan2(tz - w.z, tx - w.x); o.yT = -(15 + 6 * Math.sin(S.t / 37)); o.vT = 1.8;
    } else if (w === calf) {
      if (busy(mum)) { o.yawT = w.yaw + 0.25; o.yT = -6; o.vT = 0.9; }
      else {
        const c = Math.cos(mum.yaw), s = Math.sin(mum.yaw), ox = -1.0, oz = 4.2;
        const sx = mum.x + ox * c - oz * s, sz = mum.z + ox * s + oz * c, sy = Math.min(-2.2, mum.y + 2.4);
        const ax = sx + c * 7, az = sz + s * 7, along = (sx - w.x) * c + (sz - w.z) * s;
        o.yawT = Math.atan2(az - w.z, ax - w.x); o.yT = sy; o.vT = clamp(mum.v + along * 0.45, 0.5, 4); o.turn = 0.5;
      }
    } else {
      const st = w.station, dx = st.x - w.x, dz = st.z - w.z, d = Math.hypot(dx, dz);
      if (d > 5 || w.y > st.y + 6) { o.yawT = Math.atan2(dz, dx); o.yT = st.y; o.vT = clamp(d * 0.3, 0.7, 2.4); }
      else { o.yawT = w.yaw + 0.02; o.vT = 0.06; o.pitchT = -0.85; o.hang = true; }
    }
  }

  function emitBlow(w) {
    const b = blowhole(w), rx = -Math.sin(w.yaw), rz = Math.cos(w.yaw), k = Math.sqrt(w.L / 14);
    for (let n = 0; n < 7; n++) {
      const side = n & 1 ? 1 : -1, a = 0.26 + 0.2 * (RP() - 0.5), sp = (6.5 + 4.5 * RP()) * k, j = 0.9;
      const vx = rx * Math.sin(a) * side * sp + (RP() - 0.5) * j - Math.cos(w.yaw) * 0.6, vz = rz * Math.sin(a) * side * sp + (RP() - 0.5) * j - Math.sin(w.yaw) * 0.6;
      spray(1, b[0] + rx * side * 0.12, Math.max(b[1], 0.05), b[2] + rz * side * 0.12, vx, Math.cos(a) * sp, vz, 2.6 + 2.2 * RP(), (0.5 + 0.7 * RP()) * k);
    }
    for (let n = 0; n < 5; n++) { const side = n & 1 ? 1 : -1, a = 0.3 + 0.5 * (RP() - 0.5), sp = (3 + 6.5 * RP()) * k;
      spray(3, b[0], Math.max(b[1], 0.05), b[2], rx * Math.sin(a) * side * sp + (RP() - 0.5) * 1.6, Math.cos(a) * sp, rz * Math.sin(a) * side * sp + (RP() - 0.5) * 1.6, 4.5 + 4 * RP(), (0.16 + 0.22 * RP()) * k); }
  }
  function splash(x, z, n, power, plume, mistK, crown) { mistK = mistK === undefined ? 0.5 : mistK;
    /* a crown: spokes of thrown water, each a stack of heavy drops leaving at the same angle */
    for (let sp = 0; sp < (crown || 0); sp++) { const a = (sp + RP() * 0.6) / crown * TAU, c = Math.cos(a), sn = Math.sin(a), r0 = (1.6 + 1.2 * RP()) * power, top = (6.5 + 4.5 * RP()) * power, out = (1.6 + 2.6 * RP()) * power;
      for (let q = 0; q < 7; q++) { const f = 0.35 + 0.65 * (q + RP()) / 7; spray(2, x + c * r0, 0.1, z + sn * r0, c * out * (0.6 + 0.6 * f), top * f, sn * out * (0.6 + 0.6 * f), 6, (0.55 + 0.5 * RP()) * (1.15 - 0.5 * f)); } }
    for (let i = 0; i < n; i++) { const a = RP() * TAU, r = RP() * 2.5 * power, out = (2 + 7 * RP()) * power, up = (3 + 8 * RP() * RP()) * power;
      spray(0, x + Math.cos(a) * r, 0.1, z + Math.sin(a) * r, Math.cos(a) * out, up, Math.sin(a) * out, 6, 0.16 + 0.34 * RP()); }
    for (let i = 0; i < n * mistK; i++) { const a = RP() * TAU, r = RP() * 3.5 * power; spray(1, x + Math.cos(a) * r, 0.3, z + Math.sin(a) * r, Math.cos(a) * 3 * RP(), (1 + 7 * RP()) * power, Math.sin(a) * 3 * RP(), 2.5 + 2.5 * RP(), 0.8 + RP()); }
    for (let i = 0; i < plume; i++) { const a = RP() * TAU, r = Math.sqrt(RP()) * 4 * power; bubble(x + Math.cos(a) * r, -0.3 - RP() * RP() * 9 * power, z + Math.sin(a) * r, 0.5 + 1.1 * RP(), 0.05 + 0.1 * RP(), false); }
  }

  /* white water where a body breaks the surface */
  function waterline(w, n, power) {
    const sp = Math.sin(w.pitch); if (Math.abs(sp) < 0.08) return; const lx = clamp(-w.y / (w.L * sp), -0.55, 0.4), p = at(w, lx, 0), rad = 0.09 * w.L;
    for (let i = 0; i < n; i++) { const a = RP() * TAU, c = Math.cos(a), s = Math.sin(a), out = (1.5 + 4 * RP()) * power;
      const mist = i % 5 === 0; spray(mist ? 1 : 0, p[0] + c * rad, 0.15, p[2] + s * rad, c * out, (2 + 6 * RP()) * power, s * out, mist ? 2 + 2 * RP() : 5, mist ? 0.5 + 0.6 * RP() : 0.14 + 0.3 * RP()); }
  }
  function stepWhale(w) {
    const o = { yawT: w.yaw, yT: null, pitchT: null, vT: w.vc, turn: 0.22, acc: 0.5, pr: 0.5, rollT: null, hang: false, free: true };
    w.st += DT; w.tb += DT;
    const prev = w.state;
    if (w.state !== 'air') w.C += DT * (0.5 + 0.5 * (w.v / 1.8) * (w.v / 1.8)) / w.Tdive;
    const ySurf = -0.035 * w.L;

    if (w === mum) {
      /* the breach count: it rises by an amount redrawn each second, and she breaches when it reaches one */
      if (auto) { if (S.frame % 60 === 0) w.rb = (0.3 + 1.4 * R()) / 260; if (!S.net || S.net.phase === 'done') w.B += DT * w.rb; if (w.B >= 1 && w.state === 'dive') { w.B = 0; w.state = 'br_dive'; w.st = 0; log('breach_start', w); } }
      if (w.want && free(w)) trigger(w.want);
    }

    switch (w.state) {
      case 'dive': roam(w, o); if (w.C >= 1) { w.state = 'rise'; w.st = 0; } break;
      case 'rise': {
        roam(w, o); o.hang = false; o.yT = ySurf; o.pitchT = null; o.gain = 0.16; o.vT = 2;
        const b = blowhole(w);
        if (b[1] > 0.1 * w.L / 14 && w.st > 0.3) {
          if (w.nb === 0) { if (w === mum && w.lastSeries >= 0) S.dives.push(S.t - w.lastSeries); w.lastSeries = S.t; }
          w.nb++; w.C = 0; w.tb = 0; w.blowT = 0.9; w.peak = 0; w.state = 'breathe'; w.st = 0; log('blow', w, { x: b[0], y: b[1], z: b[2], n: w.nb });
          foam(b[0], b[2], 0.8 * w.L / 14, 0.5, 7, 0, 0.5);
        }
        break; }
      case 'breathe': roam(w, o); o.hang = false; o.yT = ySurf; o.pitchT = null; o.vT = 1.4; if (w.st > 1.9) { w.state = w.nb < w.series ? 'shallow' : 'sound'; w.st = 0; } break;
      case 'shallow': roam(w, o); o.hang = false; o.pitchT = null; o.yT = -0.26 * w.L; o.vT = 1.5; if (w.C >= w.Cs) { w.state = 'rise'; w.st = 0; } break;
      case 'sound': roam(w, o); o.hang = false; o.yT = null; o.pitchT = -0.6; o.vT = 2.1; o.pr = 0.4; if (w.y < -0.8 * w.L || w.st > 9) { w.state = 'dive'; w.st = 0; w.nb = 0; } break;

      case 'br_dive': roam(w, o); o.yT = -19; o.vT = 3; o.gain = 0.1; o.pmax = 0.6; if (w.y < -16.5 || w.st > 12) { w.state = 'br_run'; w.st = 0; } break;
      case 'br_run': o.pitchT = 1.12; o.pr = 1.0; o.vT = 8.6; o.acc = 0.95; o.rollT = 0;
        if (w.y > -1.2 && w.st > 0.5) { const f = fwd(w); w.vx = f[0] * w.v; w.vy = f[1] * w.v; w.vz = f[2] * w.v; w.state = 'air'; w.st = 0; w.splashed = false; w.exitV = w.v; log('exit', w, { v: w.v, x: w.x, z: w.z });
          const s = snout(w); splash(s[0], s[2], 160, 0.7, 0, 0.2, 18); foam(s[0], s[2], 2, 1.6, 12, 0, 0.9); }
        break;
      case 'air': {
        o.free = false; w.vy -= 9.81 * DT; w.x += w.vx * DT; w.y += w.vy * DT; w.z += w.vz * DT;
        w.pitch -= 0.72 * DT; w.roll = Math.min(1.35, w.roll + 1.0 * DT); if (!w.splashed || w.y > -1) waterline(w, 7, 1); w.amp = lerp(w.amp, 0.15, 0.05); w.arch = lerp(w.arch, 0.25, 0.04);
        const s = snout(w); if (s[1] > w.peak) w.peak = s[1];
        if (w.vy < 0 && w.y < 0.4 && !w.splashed) { w.splashed = true; log('splash', w, { x: w.x, z: w.z }); const m = at(w, 0.1, 0); splash(m[0], m[2], 900, 1.25, 420, 0.10, 46); foam(m[0], m[2], 3, 2.6, 16, 0, 1); foam(m[0], m[2], 3.5, 5.5, 13, 2, 0.9); }
        if (w.vy < 0 && w.y < -1.6) { w.state = 'br_in'; w.st = 0; w.v = Math.hypot(w.vx, w.vz) * 0.6 + 0.5; }
        break; }
      case 'br_in': o.pitchT = clamp((-5 - w.y) * 0.08, -0.3, 0.3); o.pr = 0.5; o.vT = 1.3; o.acc = 1.4; o.rollT = 0;
        if (w.st > 5.5) { w.state = 'rise'; w.st = 0; w.C = Math.max(w.C, 1); } break;

      case 'bn_go': {
        const N = S.net, tx = N.cx + N.R * Math.cos(w.bn.a), tz = N.cz + N.R * Math.sin(w.bn.a), d = Math.hypot(tx - w.x, tz - w.z);
        o.yawT = Math.atan2(tz - w.z, tx - w.x); o.yT = -14; o.vT = clamp(d * 0.45, 1.2, 3.4); o.turn = 0.45; w.bn.d = d;
        break; }
      case 'bn_spin': {
        const N = S.net, u = clamp(w.st / 20, 0, 1), a = w.bn.a + 0.36 * w.st, rad = lerp(N.R, N.R - 2, u);
        const tx = N.cx + rad * Math.cos(a), tz = N.cz + rad * Math.sin(a), ty = lerp(-14, -7.5, u), k = 0.02 + 0.13 * sstep(0, 3, w.st);
        w.x += (tx - w.x) * k; w.y += (ty - w.y) * k; w.z += (tz - w.z) * k;
        w.yaw += clamp(angd(a + Math.PI / 2, w.yaw) * 3, -0.9, 0.9) * DT; w.pitch = lerp(w.pitch, 0.1, 0.05); w.roll = lerp(w.roll, -0.4, 0.03);
        w.v = lerp(w.v, 0.36 * rad, 0.05); w.bend = lerp(w.bend, 0.5, 0.04); o.free = false;
        if (w.st > 0.8) { const b = blowhole(w); for (let n = 0; n < 2; n++) bubble(b[0] + (RP() - 0.5) * 0.8, b[1] + (RP() - 0.5) * 0.5, b[2] + (RP() - 0.5) * 0.8, 0.75 + 0.6 * RP(), 0.06 + 0.12 * RP(), true); }
        break; }
      case 'bn_under': {
        const N = S.net, T = 4.5, u = clamp(w.st / T, 0, 1), e = sstep(0, 1, u), q = w.bn;
        const tx = N.cx + 2.2 * Math.cos(q.a2), tz = N.cz + 2.2 * Math.sin(q.a2), ty = -15.5;
        w.x = lerp(q.x0, tx, e); w.z = lerp(q.z0, tz, e); w.y = lerp(q.y0, ty, e) - 1.5 * Math.sin(Math.PI * u);
        w.yaw += clamp(angd(Math.atan2(N.cz - w.z, N.cx - w.x), w.yaw) * 2, -1.2, 1.2) * DT;
        w.pitch = lerp(q.p0, 1.36, sstep(0.35, 1, u)) - 0.5 * Math.sin(Math.PI * Math.min(1, u / 0.6)) * (u < 0.6 ? 1 : 0);
        w.roll = lerp(w.roll, 0, 0.05); w.v = 2.6; w.bend = lerp(w.bend, 0, 0.05); o.free = false;
        break; }
      case 'bn_lunge': {
        o.free = false; w.v = lerp(w.v, 5.4, 0.06); w.pitch = lerp(w.pitch, 1.36, 0.1); const f = fwd(w); w.x += f[0] * w.v * DT; w.y += f[1] * w.v * DT; w.z += f[2] * w.v * DT;
        w.gape = sstep(-10, -4.5, w.y); w.gulp = Math.max(w.gulp, w.gape * sstep(-5, -1, w.y)); w.amp = lerp(w.amp, 1.2, 0.05);
        const s = snout(w);
        if (s[1] > 0 && !w.splashed) { w.splashed = true; log('lunge', w, { x: s[0], z: s[2] }); splash(s[0], s[2], 220, 0.6, 120, 0.12); foam(s[0], s[2], 1.4, 0.9, 12, 0, 0.75); }
        if (s[1] > w.peak) w.peak = s[1]; if (s[1] > 0) waterline(w, 4, 0.6);
        if (w.y > -0.17 * w.L) { w.state = 'bn_sink'; w.st = 0; }
        break; }
      case 'bn_sink': o.pitchT = 0.2; o.pr = 0.35; o.vT = 0.7; o.acc = 1.6; w.gape = Math.max(0, w.gape - DT / 1.6); w.gulp = Math.max(0, w.gulp - DT / 14);
        if (w.st < 1.2) w.gulp = Math.max(w.gulp, 0.9);
        if (w.st > 6) { w.state = 'rise'; w.st = 0; w.C = Math.max(w.C, 1); w.nb = 0; } break;
    }
    if (w.state !== 'bn_sink' && w.state !== 'bn_lunge') { w.gape = Math.max(0, w.gape - DT); w.gulp = Math.max(0, w.gulp - DT / 14); }

    if (o.free) {
      if (o.pitchT === null) o.pitchT = o.yT === null ? 0 : clamp((o.yT - w.y) * (o.gain || 0.09), -(o.pmax || 0.42), 0.45);
      const dy = angd(o.yawT, w.yaw), yr = clamp(dy * 1.2, -o.turn, o.turn) * Math.min(1, 0.25 + w.v);
      w.yaw += yr * DT; w.yr = lerp(w.yr, yr, 0.05);
      w.pitch += clamp((o.pitchT - w.pitch) * 1.5, -o.pr, o.pr) * DT;
      const a = (o.vT - w.v) * o.acc; w.v += a * DT;
      const f = fwd(w); w.x += f[0] * w.v * DT; w.y += f[1] * w.v * DT; w.z += f[2] * w.v * DT;
      if (o.hang) { w.y += (w.station.y - w.y) * 0.01; w.x += (w.station.x - w.x) * 0.003; w.z += (w.station.z - w.z) * 0.003; }
      w.roll = lerp(w.roll, o.rollT === null ? -w.yr * 1.5 : o.rollT, 0.03);
      w.amp = lerp(w.amp, clamp(0.25 + Math.max(0, a) * 0.9 + w.v * 0.2, 0.12, 1.3), 0.04);
      w.bend = lerp(w.bend, yr * 1.6, 0.05); w.arch = lerp(w.arch, clamp((o.pitchT - w.pitch) * 0.5, -0.3, 0.3), 0.04);
      if (w.y > ySurf + 0.3 && w.state !== 'br_run') { w.y = lerp(w.y, ySurf + 0.3, 0.1); }
    }
    w.ph += TAU * (0.09 + 0.085 * w.v * Math.sqrt(14 / w.L)) * DT;
    /* flippers: spread when slow or turning, flung wide in the air and in a lunge */
    const pt = w.state === 'air' || w.state === 'br_run' ? 1 : (w.state === 'bn_lunge' || w.state === 'bn_sink') ? 0.8 : o.hang ? 0.55 : clamp(0.15 + Math.abs(w.yr) * 1.6 + (1.6 - w.v) * 0.12, 0.05, 0.7);
    w.pec = lerp(w.pec, pt, 0.03);
    if (w.blowT > 0) { w.blowT -= DT; emitBlow(w); }
    w.singing = w === male && (w.state === 'dive' || w.state === 'rise' || w.state === 'breathe' || w.state === 'shallow' || w.state === 'sound');
  }

  function stepNet() {
    const N = S.net; if (!N || N.phase === 'done') return; N.t += DT;
    if (N.phase === 'go') {
      if ((mum.bn.d < 5.5 && male.bn.d < 5.5 && N.t > 1) || N.t > 26) { N.phase = 'spin'; N.t = 0;
        [mum, male].forEach(w => { w.state = 'bn_spin'; w.st = 0; w.bn.a = Math.atan2(w.z - N.cz, w.x - N.cx); }); log('net_spin', mum); }
    } else if (N.phase === 'spin') {
      if (N.t > 20) { N.phase = 'under'; N.t = 0; [mum, male].forEach(w => { w.state = 'bn_under'; w.st = 0; Object.assign(w.bn, { x0: w.x, y0: w.y, z0: w.z, p0: w.pitch, a2: Math.atan2(w.z - N.cz, w.x - N.cx) }); }); }
    } else if (N.phase === 'under') {
      if (N.t > 4.5) { N.phase = 'lunge'; N.t = 0; [mum, male].forEach(w => { w.state = 'bn_lunge'; w.st = 0; w.splashed = false; w.peak = 0; }); log('net_lunge', mum); }
    } else if (N.phase === 'lunge') {
      if (mum.state !== 'bn_lunge' && mum.state !== 'bn_sink' && male.state !== 'bn_lunge' && male.state !== 'bn_sink') { N.phase = 'done'; log('net_done', mum, { taken: N.taken }); }
    }
    if (N.ringN > 20 && !N.foam) N.foam = foam(N.cx, N.cz, N.ringR / N.ringN, 0, 60, 1, 0);
    if (N.foam) { N.foam.r = N.ringR / N.ringN; N.foam.s = clamp(N.ringN / 900, 0, 1); N.foam.x = N.cx; N.foam.z = N.cz; if (N.phase !== 'done') N.foam.age = Math.min(N.foam.age, 30); }
  }

  function stepFish() {
    const N = S.net, netOn = N && (N.phase === 'spin' || N.phase === 'under' || N.phase === 'lunge');
    const H = S.home;
    if (N && N.phase !== 'done') { H.x = lerp(H.x, N.cx, 0.02); H.z = lerp(H.z, N.cz, 0.02); H.y = lerp(H.y, netOn ? -3.2 : -7, 0.01); }
    else { H.x = lerp(H.x, 12 * Math.cos(S.t / 50), 0.02); H.z = lerp(H.z, 12 * Math.sin(S.t / 50), 0.02); H.y = lerp(H.y, -8, 0.01); }
    head.fill(-1);
    for (let i = 0; i < NF; i++) { if (F.gone[i]) continue; const h = hsh(Math.floor(F.x[i] / CELL), Math.floor(F.y[i] / CELL), Math.floor(F.z[i] / CELL)); next[i] = head[h]; head[h] = i; }
    /* whale points the fish watch: head and mid-body */
    const WP = []; for (const w of S.whales) { const a = at(w, 0.3, 0), b = at(w, -0.05, 0), c = at(w, -0.4, 0), g = w.state === 'bn_lunge' ? 0.22 : 1; WP.push(a[0], a[1], a[2], 0.6 * w.L, g, b[0], b[1], b[2], 0.5 * w.L, g, c[0], c[1], c[2], 0.3 * w.L, g); }
    S.wp = WP; let cx = 0, cy = 0, cz = 0, hx = 0, hy = 0, hz = 0, cnt = 0;
    const lungeMouths = []; if (N && N.phase === 'lunge') for (const w of [mum, male]) if (w.gape > 0.4) { const m = at(w, 0.36, -0.04); lungeMouths.push(m[0], m[1], m[2], 0.16 * w.L); }
    for (let i = 0; i < NF; i++) {
      if (F.gone[i]) continue;
      const x = F.x[i], y = F.y[i], z = F.z[i]; let vx = F.vx[i], vy = F.vy[i], vz = F.vz[i];
      let n = 0, ax = 0, ay = 0, az = 0, mx = 0, my = 0, mz = 0, sx = 0, sy = 0, sz = 0, wx = 0, wy = 0, wz = 0;
      const ix = Math.floor(x / CELL), iy = Math.floor(y / CELL), iz = Math.floor(z / CELL);
      scan: for (let k = 0; k < 27; k++) {
        const ox = OFF[k * 3], oy = OFF[k * 3 + 1], oz = OFF[k * 3 + 2];
        for (let j = head[hsh(ix + ox, iy + oy, iz + oz)]; j >= 0; j = next[j]) {
          if (j === i) continue; const dx = F.x[j] - x, dy = F.y[j] - y, dz = F.z[j] - z, d2 = dx * dx + dy * dy + dz * dz;
          if (d2 > 2.6 || d2 < 1e-8) continue;
          F.nb[i * 10 + n] = j; n++; mx += F.vx[j]; my += F.vy[j]; mz += F.vz[j]; sx += dx; sy += dy; sz += dz;
          if (d2 < 0.3) { const q = (0.3 - d2) / (0.3 * Math.sqrt(d2)); wx -= dx * q; wy -= dy * q; wz -= dz * q; }
          if (n >= 10) break scan;
        }
      }
      if (n) { ax += (mx / n - vx) * 1.5 + sx / n * 0.8 + wx * 5; ay += (my / n - vy) * 1.5 + sy / n * 0.8 + wy * 5; az += (mz / n - vz) * 1.5 + sz / n * 0.8 + wz * 5; }
      let dx = H.x - x, dy = H.y - y, dz = H.z - z, d = Math.hypot(dx, dy, dz);
      if (d > 5) { const q = Math.min(3, (d - 5) * 0.3) / d; ax += dx * q; ay += dy * q; az += dz * q; }
      if (y > -1.0) ay -= (y + 1.0) * 8; if (y < -40) ay += (-40 - y) * 2;
      let fear = 0; F.nc[i] = n; F.fw[i] = -1;
      for (let k = 0; k < WP.length; k += 5) { dx = x - WP[k]; dy = y - WP[k + 1]; dz = z - WP[k + 2]; d = Math.hypot(dx, dy, dz); const rad = WP[k + 3];
        if (d < rad) { const q = (1 - d / rad); if (q > fear) F.fw[i] = k / 5; fear = Math.max(fear, q); const g = q * q * 34 * WP[k + 4] / (d + 0.01); ax += dx * g; ay += dy * g * 0.6; az += dz * g; } }
      if (netOn) { dx = x - N.cx; dz = z - N.cz; d = Math.hypot(dx, dz); const lim = N.R - 2.6;
        if (d > lim && d < N.R + 3) { const q = (d - lim) * 4 / (d + 0.01); ax -= dx * q; az -= dz * q; fear = Math.max(fear, 0.4); }
        if (N.phase !== 'spin') { ay += 1.5; } }
      for (let k = 0; k < lungeMouths.length; k += 4) { dx = x - lungeMouths[k]; dy = y - lungeMouths[k + 1]; dz = z - lungeMouths[k + 2]; if (dx * dx + dy * dy + dz * dz < lungeMouths[k + 3] * lungeMouths[k + 3]) { F.gone[i] = 1; S.eaten++; N.taken++; } }
      F.fear[i] = lerp(F.fear[i], fear, 0.1);
      vx += ax * DT; vy += ay * DT; vz += az * DT; vy *= 0.985;
      const sp = Math.hypot(vx, vy, vz) + 1e-6, want = 1.1 + 2.6 * F.fear[i], ns = clamp(sp + (want - sp) * 2 * DT, 0.7, 4.2), q = ns / sp;
      vx *= q; vy *= q; vz *= q; F.vx[i] = vx; F.vy[i] = vy; F.vz[i] = vz;
      F.x[i] = x + vx * DT; F.y[i] = Math.min(-0.25, y + vy * DT); F.z[i] = z + vz * DT;
      cx += x; cy += y; cz += z; hx += vx / ns; hy += vy / ns; hz += vz / ns; cnt++;
    }
    if (cnt) { const k = S.frame === 0 ? 1 : 0.05; S.school.x = lerp(S.school.x, cx / cnt, k); S.school.y = lerp(S.school.y, cy / cnt, k); S.school.z = lerp(S.school.z, cz / cnt, k); S.school.pol = Math.hypot(hx, hy, hz) / cnt; }
  }

  function stepParticles() {
    for (let i = 0; i < MAXSPRAY; i++) { if (P.life[i] <= 0) continue; P.life[i] -= DT;
      const kd = P.kind[i]; if (kd === 0 || kd === 2) { P.vy[i] -= 9.81 * DT; const k = 1 - (kd === 2 ? 0.7 : 0.35) * DT; P.vx[i] *= k; P.vy[i] *= k; P.vz[i] *= k; }
      else if (kd === 1) { P.vy[i] -= 1.1 * DT; const k = 1 - 2.3 * DT; P.vx[i] = P.vx[i] * k + 0.9 * DT; P.vy[i] *= k; P.vz[i] *= k; P.size[i] += 0.5 * DT; }
      else { P.vy[i] -= 0.25 * DT; const k = 1 - 1.6 * DT; P.vx[i] = P.vx[i] * k + 1.5 * DT; P.vy[i] *= k; P.vz[i] *= k; P.size[i] += 0.22 * DT; }
      P.x[i] += P.vx[i] * DT; P.y[i] += P.vy[i] * DT; P.z[i] += P.vz[i] * DT; if (P.y[i] < 0) P.life[i] = 0; }
    const N = S.net;
    for (let i = 0; i < MAXBUB; i++) { if (Bb.life[i] <= 0) continue; Bb.life[i] -= DT; Bb.ph[i] += 4 * DT;
      Bb.y[i] += Bb.vy[i] * DT; Bb.x[i] += Math.sin(Bb.ph[i]) * 0.12 * DT; Bb.z[i] += Math.cos(Bb.ph[i] * 0.83) * 0.12 * DT;
      if (Bb.y[i] > -0.06) { Bb.life[i] = 0; if (Bb.net[i] && N) { N.ringN++; N.ringR += Math.hypot(Bb.x[i] - N.cx, Bb.z[i] - N.cz); } } }
    for (let i = S.foam.length - 1; i >= 0; i--) { const f = S.foam[i]; f.age += DT; f.r += f.grow * DT * Math.exp(-f.age * 0.25); if (f.age > f.life) S.foam.splice(i, 1); }
  }

  function step() {
    S.ev.length = 0;
    while (script.length && script[0].at <= S.frame) { const e = script.shift(); if (e.event) { const ok = trigger(e.event); if (!ok && e.event === 'blow') mum.want = null; } }
    for (const w of S.whales) stepWhale(w);
    stepNet();
    if (male.singing) { const e = singerStep(S.singer); if (e) { S.unit = e; S.unitFrame = S.frame; log('unit', male, e); } }
    if (NF) stepFish();
    stepParticles();
    const k = 0.035; S.anchor.x = lerp(S.anchor.x, mum.x, k); S.anchor.y = lerp(S.anchor.y, mum.y, k); S.anchor.z = lerp(S.anchor.z, mum.z, k); S.anchor.yaw += angd(mum.yaw, S.anchor.yaw) * 0.02;
    S.frame++; S.t = S.frame * DT;
  }
  S.step = step;
  S.anchor.x = mum.x; S.anchor.y = mum.y; S.anchor.z = mum.z; S.anchor.yaw = mum.yaw; S.home.x = 12; S.school.x = 12;
  S.mother = mum; S.calf = calf; S.male = male;
  S.sounding = () => S.unit && male.singing && S.frame - S.unitFrame < Math.round(UNIT[S.unit.unit].dur * 60);
  return S;
}
const OFF = []; { const o = []; for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) for (let c = -1; c <= 1; c++) o.push([a, b, c]); o.sort((p, q) => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]) - (q[0] * q[0] + q[1] * q[1] + q[2] * q[2])); for (const p of o) OFF.push(p[0], p[1], p[2]); }

const WORDS = { dive: 'under water', rise: 'coming up', breathe: 'breathing', shallow: 'between breaths', sound: 'going down, flukes up', br_dive: 'diving to breach', br_run: 'swimming up to breach', air: 'in the air', br_in: 'back in the water',
  bn_go: 'going under the fish', bn_spin: 'blowing the net', bn_under: 'turning under the net', bn_lunge: 'lunging', bn_sink: 'straining the mouthful' };

const API = { DT, soundSpeed, SEA, tempAt, speedAt, channel, rays, UNITS, UNIT, THEMES, ORDER, SONG, newSinger, singerStep, makeSim, WORDS, rng32, MAXSPRAY, MAXBUB, MAXFOAM, clamp, lerp, sstep, angd };
if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.WHALE_MODEL = API;
})(typeof window !== 'undefined' ? window : globalThis);
