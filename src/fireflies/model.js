/* The firefly model. No drawing in here. Runs in a browser or in node.
   Units are metres and seconds.

   What comes from the literature:
   - Photinus carolinus males flash in bursts of several flashes about half a second apart, then go dark for several
     seconds; in a swarm the bursts come together and the dark gaps become regular (Copeland and Moiseff 1995;
     Sarfati, Hayes and Peleg 2021)
   - a firefly kept alone flashes in bursts at irregular, widely spread intervals; in a group the interval shortens and
     narrows, and is close to the shortest a firefly ever waits on its own (Sarfati et al. 2023). The model used here
     follows the shape of theirs: a recovery time after each burst with a sharp lower edge, then a chance each moment
     of starting a burst unprompted, and joining on the next beat on seeing another's flash once recovered
   - a firefly sees another's flash less often the further away it is, because leaves are in the way
   - within a burst, a flash seen nudges the next flash of one's own toward it (pulse coupling: Buck 1988;
     Mirollo and Strogatz 1990)
   - artificial light suppresses flashing (Owens and Lewis 2018)
   What is this page's own choice:
   - every number below. The recovery time, the chance of an unprompted start, the sight distance and the size of the
     nudge are settings, chosen so that a lone firefly's intervals spread widely. They are not fitted to data.
   - the count is of dark time: a flash seen while recovering starts it again (hold). Without this the swarm never went
     dark between bursts (see BETS.md, v1 and the explorations after v2).
   The timing grammar is a count, a threshold and a restart: each firefly counts dark time since its last burst; once
   past its threshold it is ready; the burst is the cut; the count restarts at zero and a new threshold is drawn. */
(function (root) {
'use strict';
const DEFAULTS = {
  W: 40, D: 40, z0: 8, yLo: .3, yHi: 4,           /* the clearing: x from -W/2 to W/2, z from z0 to z0+D (distance from the path), height */
  Tc: 9.0, TcSd: .3, TcMin: 6,                     /* recovery: dark time needed after a burst, drawn afresh each time (v1, v2: sd 0.9) */
  tauS: 26,                                        /* mean wait for an unprompted start, once recovered */
  kLo: 4, kHi: 8,                                  /* flashes in a burst */
  Tf: .52, TfSd: .03, flashJit: .02,               /* time between flashes in a burst, and its jitter */
  react: .52, reactSd: .02,                        /* from seeing a flash to one's own first: the next beat (v1, v2: 0.22 s) */
  lambda: 6,                                       /* sight: probability exp(-d/lambda) */
  see: true, nudge: .45, hold: true, occl: 4,       /* hold: a flash seen while recovering restarts the count; occl: seconds a line of sight lasts */
  speed: .45, flyShare: .7,                        /* drifting flight */
  torch: 0, torchX: 0, torchY: .02, torchHalf: .2, torchHalfV: .17, torchH: 1.2,   /* 0 off, 1 white, 2 red; beam centre (bearing, elevation, radians), half-widths, height held */
};
function rng(seed) { let s = (seed || 7) % 2147483647; if (s <= 0) s += 2147483646; return () => (s = (s * 16807) % 2147483647) / 2147483647; }
function gauss(r) { let u = 0, v = 0; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307 * v); }

const CHARGE = 0, READY = 1, BURST = 2;

function make(n, opt) {
  const P = Object.assign({}, DEFAULTS, opt || {}), r = rng(P.seed || 7), N = Math.max(n, P.nMax || n);
  const f = () => new Float32Array(N);
  const S = { n, nMax: N, P, r, t: 0, flashes: [], log: [], logOn: P.log !== false,
    x: f(), y: f(), z: f(), vx: f(), vy: f(), vz: f(), hx: f(), hy: f(), hz: f(),
    st: new Uint8Array(N), c: f(), thr: f(), left: new Uint8Array(N), next: new Float64Array(N), Tf: f(),
    seen: new Float64Array(N).fill(-1e9), glow: f(), lastFlash: new Float64Array(N), lastBurst: new Float64Array(N), lit: new Uint8Array(N), id: new Uint32Array(N),
    nFlash: 0, nBurst: 0, nSpont: 0, due: [], qd: new Uint32Array(N), qs: 1, t1: 0 };
  for (let i = 0; i < N; i++) place(S, i, true);
  return S;
}
function place(S, i, fresh) {
  const P = S.P, r = S.r;
  S.x[i] = (r() - .5) * P.W; S.z[i] = P.z0 + r() * P.D; S.y[i] = P.yLo + Math.pow(r(), 1.6) * (P.yHi - P.yLo);
  S.hx[i] = gauss(r); S.hy[i] = gauss(r) * .3; S.hz[i] = gauss(r);
  S.vx[i] = S.vy[i] = S.vz[i] = 0;
  S.Tf[i] = P.Tf + gauss(r) * P.TfSd;
  S.st[i] = CHARGE; S.thr[i] = drawThr(S); S.c[i] = fresh ? r() * (S.thr[i] + P.tauS * .6) : 0;
  S.lastFlash[i] = S.lastBurst[i] = -1e9; S.glow[i] = 0; S.id[i] = (S.id[i] || i) + (fresh ? 0 : S.nMax);
}
function drawThr(S) { const P = S.P; return Math.max(P.TcMin, P.Tc + gauss(S.r) * P.TcSd); }

function setCount(S, n) { n = Math.max(1, Math.min(S.nMax, Math.round(n))); for (let i = S.n; i < n; i++) place(S, i, true); S.n = n; }

/* is firefly i lit by the torch? The torch is held on the path at the origin; its beam is an ellipse of directions. */
function inBeam(S, i) { const P = S.P; if (!P.torch) return false; const a = (Math.atan2(S.x[i], S.z[i]) - P.torchX) / P.torchHalf, e = (Math.atan2(S.y[i] - P.torchH, Math.hypot(S.x[i], S.z[i])) - P.torchY) / P.torchHalfV; return a * a + e * e < 1; }

function startBurst(S, i, t0) {
  const P = S.P; S.st[i] = BURST; S.left[i] = P.kLo + Math.floor(S.r() * (P.kHi - P.kLo + 1)); S.next[i] = t0; S.nBurst++;
}

/* whether a leaf is in the way between i and j: the same for a few seconds at a time, not drawn afresh each flash */
function sight(S, i, j, t) {
  let a = S.id[i], b = S.id[j]; if (a > b) { const q = a; a = b; b = q; }
  let h = Math.imul(a, 73856093) ^ Math.imul(b, 19349663); const ph = ((h >>> 0) % 1000) / 1000;
  h ^= Math.imul(Math.floor(t / S.P.occl + ph) | 0, 83492791); h = Math.imul(h ^ (h >>> 15), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
/* one flash, by firefly i at time t: everyone who sees it may respond */
function flash(S, i, t) {
  const P = S.P; S.nFlash++; S.lastFlash[i] = t; S.glow[i] = 1;
  if (S.logOn) S.flashes.push(t, i); if (S.onFlash) S.onFlash(i, t);
  if (!P.see) return;
  const xi = S.x[i], yi = S.y[i], zi = S.z[i], n = S.n, inv = 1 / P.lambda, r = S.r;
  for (let j = 0; j < n; j++) {
    if (j === i || S.lit[j]) continue; const s = S.st[j];
    const dx = S.x[j] - xi, dy = S.y[j] - yi, dz = S.z[j] - zi, d = Math.sqrt(dx * dx + dy * dy + dz * dz);
    if (sight(S, i, j, t) > Math.exp(-d * inv)) continue;
    S.seen[j] = t;
    if (s === CHARGE) { if (P.hold) S.c[j] = 0; continue; }   /* still recovering: the count is of dark time, so it starts again */
    if (s === READY) startBurst(S, j, t + Math.max(.05, P.react + gauss(r) * P.reactSd));   /* join, on the next beat */
    else if (P.nudge > 0) {                          /* in a burst: pull my next flash toward this one's rhythm */
      const want = t + S.Tf[j], gap = want - S.next[j];
      if (Math.abs(gap) < S.Tf[j] * .5 && S.next[j] > t + .04) S.next[j] += gap * P.nudge; else continue;
    } else continue;
    if (S.next[j] < S.t1 && S.qd[j] !== S.qs) { S.qd[j] = S.qs; S.due.push(j); }
  }
}

/* one step for every firefly */
function step(S, dt) {
  const P = S.P, n = S.n, r = S.r, t0 = S.t, t1 = S.t + dt, pS = dt / P.tauS;
  /* flashes due in this step are taken in time order, so that one flash can bring on another within the step */
  const due = S.due; due.length = 0; S.t1 = t1; S.qs++;
  for (let i = 0; i < n; i++) {
    S.lit[i] = P.torch === 1 && inBeam(S, i) ? 1 : 0;
    const s = S.st[i];
    if (s === BURST) { if (S.lit[i]) { endBurst(S, i); continue; } if (S.next[i] < t1) { S.qd[i] = S.qs; due.push(i); } continue; }
    S.c[i] += dt;                                    /* the count: time since the last burst */
    if (s === CHARGE && S.c[i] >= S.thr[i]) S.st[i] = READY;
    if (S.st[i] === READY && !S.lit[i] && r() < pS) { startBurst(S, i, t0 + r() * dt); S.nSpont++; if (S.next[i] < t1) { S.qd[i] = S.qs; due.push(i); } }
  }
  /* a small queue: process due flashes in time order, adding any that become due */
  let guard = 0;
  while (due.length && guard++ < 20000) {
    let k = 0; for (let q = 1; q < due.length; q++) if (S.next[due[q]] < S.next[due[k]]) k = q;
    const i = due[k]; due[k] = due[due.length - 1]; due.pop();
    S.qd[i] = 0; if (S.st[i] !== BURST || S.next[i] >= t1) continue;
    const tf = S.next[i]; flash(S, i, tf);
    if (--S.left[i] <= 0) endBurst(S, i);
    else { S.next[i] = tf + S.Tf[i] + gauss(r) * P.flashJit; if (S.next[i] < t1) { S.qd[i] = S.qs; due.push(i); } }
  }
  /* flight: a slow drift that wanders, kept inside the clearing */
  for (let i = 0; i < n; i++) {
    S.hx[i] += gauss(r) * .9 * Math.sqrt(dt) - S.hx[i] * .25 * dt; S.hy[i] += gauss(r) * .5 * Math.sqrt(dt) - S.hy[i] * .5 * dt; S.hz[i] += gauss(r) * .9 * Math.sqrt(dt) - S.hz[i] * .25 * dt;
    const fly = S.st[i] === BURST ? 1 : P.flyShare, h = Math.hypot(S.hx[i], S.hy[i], S.hz[i]) || 1, sp = P.speed * fly;
    let tx = S.hx[i] / h * sp, ty = S.hy[i] / h * sp * .6, tz = S.hz[i] / h * sp;
    const hx = P.W / 2; if (S.x[i] > hx) tx -= (S.x[i] - hx); if (S.x[i] < -hx) tx -= (S.x[i] + hx);
    if (S.z[i] < P.z0) tz += P.z0 - S.z[i]; if (S.z[i] > P.z0 + P.D) tz -= S.z[i] - P.z0 - P.D;
    if (S.y[i] < P.yLo) ty += (P.yLo - S.y[i]) * 2; if (S.y[i] > P.yHi) ty -= (S.y[i] - P.yHi) * 2;
    const a = Math.min(1, dt * 1.5); S.vx[i] += (tx - S.vx[i]) * a; S.vy[i] += (ty - S.vy[i]) * a; S.vz[i] += (tz - S.vz[i]) * a;
    S.x[i] += S.vx[i] * dt; S.y[i] += S.vy[i] * dt; S.z[i] += S.vz[i] * dt;
  }
  S.t = t1;
  if (S.logOn && S.flashes.length > 400000) S.flashes.splice(0, 200000);
}
function endBurst(S, i) { S.st[i] = CHARGE; S.c[i] = 0; S.thr[i] = drawThr(S); S.lastBurst[i] = S.t; }

/* ─── measurements, all from the record of flashes ─── */
/* collective bursts: runs of flashes with no gap longer than gap seconds */
function collective(fl, from, to, gap) {
  gap = gap || 1.5; const out = []; let cur = null;
  for (let q = 0; q < fl.length; q += 2) { const t = fl[q], i = fl[q + 1]; if (t < from || t > to) continue;
    if (!cur || t - cur.end > gap) { cur = { start: t, end: t, n: 0, who: new Set() }; out.push(cur); }
    cur.end = t; cur.n++; cur.who.add(i); }
  return out;
}
/* each firefly's own burst starts: a flash more than 1.5 s after its previous flash */
function ownStarts(fl, from, to) {
  const last = new Map(), starts = new Map();
  for (let q = 0; q < fl.length; q += 2) { const t = fl[q], i = fl[q + 1]; const l = last.get(i); last.set(i, t);
    if (t < from || t > to) continue; if (l === undefined || t - l > 1.5) { if (!starts.has(i)) starts.set(i, []); starts.get(i).push(t); } }
  return starts;
}
function stats(a) { const n = a.length; if (!n) return { n: 0, mean: NaN, sd: NaN, cv: NaN }; let m = 0; for (const v of a) m += v; m /= n; let s = 0; for (const v of a) s += (v - m) * (v - m); const sd = Math.sqrt(s / Math.max(1, n - 1)); return { n, mean: m, sd, cv: sd / m }; }
function intervals(ts) { const o = []; for (let k = 1; k < ts.length; k++) o.push(ts[k] - ts[k - 1]); return o; }
/* peaks: flashes grouped with gaps under 0.2 s; the spread of each peak */
function peaks(fl, from, to, gap) {
  gap = gap || .2; const out = []; let cur = null;
  for (let q = 0; q < fl.length; q += 2) { const t = fl[q]; if (t < from || t > to) continue;
    if (!cur || t - cur.last > gap) { cur = { ts: [], last: t }; out.push(cur); } cur.ts.push(t); cur.last = t; }
  return out.map(p => ({ n: p.ts.length, t: p.ts[0], sd: stats(p.ts).sd }));
}

function run(n, sec, opt, dt) { const S = make(n, opt); dt = dt || 1 / 120; const k = Math.round(sec / dt); for (let s = 0; s < k; s++) step(S, dt); return S; }

const API = { DEFAULTS, make, step, setCount, inBeam, collective, ownStarts, stats, intervals, peaks, run, CHARGE, READY, BURST };
if (typeof module !== 'undefined' && module.exports) module.exports = API; else root.FIREFLY = API;
})(typeof window !== 'undefined' ? window : globalThis);
