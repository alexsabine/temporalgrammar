/* Holding weight through time: the model, with no drawing. Runs in node (see test_model.js) and on the page as window.EQModel.
   Everything here is standard: the nascent delta and comb, their running integral, the bell and minus its logarithm,
   the scalar Kalman filter for a random walk seen through noise (Kalman 1960; Muth 1960), and two estimates that each
   move toward the other. Noise variance R = 1 unless stated; drift variance Q = r R. */
(function (root) {
  'use strict';
  const PHI = (1 + Math.sqrt(5)) / 2;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  /* ---- seeded numbers: a hash, so that draw k of stream s does not depend on what was asked for before ---- */
  function hash(a, b, c) { let n = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263) ^ Math.imul(c | 0, 1440662683);
    n = Math.imul(n ^ (n >>> 13), 1274126177); n = Math.imul(n ^ (n >>> 15), 2246822519); return ((n ^ (n >>> 16)) >>> 0) / 4294967296; }
  function gauss(seed, k, stream) { const u = Math.max(1e-12, hash(seed, k, stream * 2 + 11)), v = hash(seed, k, stream * 2 + 12);
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(6.283185307179586 * v); }

  /* ---- 1 and 4. the bell (a nascent delta as its width goes to nothing) and the bowl ---- */
  const bell = (x, s) => Math.exp(-x * x / (2 * s * s)) / (s * Math.sqrt(2 * Math.PI));   /* area 1 at every width */
  const bowl = (x, s) => x * x / (2 * s * s);                                             /* minus log of the bell, measured from its lowest point */

  /* ---- 2 and 3. the comb and its running total ---- */
  const TEETH = [1, 2, 3, 4, 5, 6, 7, 8];
  const count = (x, teeth) => { let n = 0; const t = teeth || TEETH; for (let i = 0; i < t.length; i++) if (t[i] <= x) n++; return n; };
  const comb = (x, s, teeth) => { let v = 0; const t = teeth || TEETH; for (let i = 0; i < t.length; i++) v += bell(x - t[i], s); return v; };

  /* ---- 5. the Kalman filter for a drifting level ---- */
  const steadyK = r => (-r + Math.sqrt(r * r + 4 * r)) / 2;                                /* solves K^2 + rK - r = 0 */
  function kalmanStep(m, P, y, Q, R) { const Pm = P + Q, K = Pm / (Pm + R); return { Pm, K, m: m + K * (y - m), P: (1 - K) * Pm }; }
  const impliedR = K => K * K / (1 - K);                                                   /* the drift-to-noise ratio for which K is the right gain */
  const believed = (K, R) => ({ Pm: K * R / (1 - K), P: K * R });                          /* and the uncertainty that goes with it, before and after */
  const fixedGainMSE = (K, Q, R) => ((1 - K) * (1 - K) * Q + K * K * R) / (1 - (1 - K) * (1 - K));

  /* a dial: a number, {from,to,start,end[,ease:'linear']} in frames, {steps:[[frame,value],...]}, or a function of the frame */
  function param(spec, frame, dflt) {
    if (spec === undefined || spec === null) return dflt;
    if (typeof spec === 'number') return spec;
    if (typeof spec === 'function') return spec(frame);
    if (spec.steps) { let v = dflt; for (let i = 0; i < spec.steps.length; i++) { if (spec.steps[i][0] <= frame) v = spec.steps[i][1]; else break; } return v; }
    if (spec.from !== undefined) { const a = spec.start || 0, b = spec.end === undefined ? a + 1 : spec.end; let t = clamp((frame - a) / Math.max(1e-9, b - a), 0, 1);
      if (spec.ease !== 'linear') t = t * t * (3 - 2 * t); return spec.from + (spec.to - spec.from) * t; }
    return dflt;
  }

  /* One run of "carried and met", n meetings (and one more looked ahead, for drawing).
     o.K: gain as a function of the meeting number, or null to let the filter find its own (the Kalman recursion).
     o.r: drift-to-noise ratio as a function of the meeting number (the world's drift, and what the filter is told).
     L: when the world leaves +-L the origin is moved by L (centre[k]); nothing but the label of the place changes. */
  function runE(o) {
    const seed = o.seed === undefined ? 7 : o.seed, n = o.n, R = o.R || 1, N = n + 2, L = o.L || 8;
    const Kf = o.K || null, rf = o.r || (() => 1);
    const x = new Float64Array(N), y = new Float64Array(N), m = new Float64Array(N), Pm = new Float64Array(N), P = new Float64Array(N), K = new Float64Array(N),
      mse = new Float64Array(N), caught = new Float64Array(N), centre = new Float64Array(N), ahead = new Float64Array(N), pathM = new Float64Array(N), pathW = new Float64Array(N);
    let r0 = rf(0), se = 0, hit = 0, c = 0, sa = 0;
    const kz = Kf ? Kf(0) : NaN;
    if (kz === kz) { const k0 = clamp(kz, 1e-6, 1 - 1e-6); P[0] = k0 * R; K[0] = k0; } else { K[0] = steadyK(r0); P[0] = K[0] * R; }
    Pm[0] = P[0];
    for (let k = 1; k < N; k++) {
      const Q = rf(k) * R;
      x[k] = x[k - 1] + Math.sqrt(Q) * gauss(seed, k, 0);
      y[k] = x[k] + Math.sqrt(R) * gauss(seed, k, 1);
      const kk = Kf ? Kf(k) : NaN;                       /* a gain set by hand, or not-a-number to let the filter work out its own */
      if (kk === kk) { K[k] = clamp(kk, 1e-6, 1 - 1e-6); Pm[k] = K[k] * R / (1 - K[k]); P[k] = K[k] * R; }
      else { Pm[k] = P[k - 1] + Q; K[k] = Pm[k] / (Pm[k] + R); P[k] = (1 - K[k]) * Pm[k]; }
      if (Math.abs(x[k] - m[k - 1]) <= 2 * Math.sqrt(Pm[k])) hit++;
      sa += (x[k] - m[k - 1]) * (x[k] - m[k - 1]); ahead[k] = sa / k;      /* the surprise: squared distance from where it was to where the world turned out to be */
      m[k] = m[k - 1] + K[k] * (y[k] - m[k - 1]);
      pathM[k] = pathM[k - 1] + Math.abs(m[k] - m[k - 1]); pathW[k] = pathW[k - 1] + Math.abs(x[k] - x[k - 1]);
      se += (m[k] - x[k]) * (m[k] - x[k]); mse[k] = se / k; caught[k] = hit / k;
      if (x[k] - c > L) c += L; else if (x[k] - c < -L) c -= L; centre[k] = c;
    }
    return { n, x, y, m, Pm, P, K, mse, caught, centre, ahead, pathM, pathW, R };
  }

  /* Two who meet. Each drifts on its own (sd q a meeting), then each moves its gain of the way toward where the other was.
     a, b: after the exchange; aPre, bPre: just before it. */
  function runF(o) {
    const seed = o.seed === undefined ? 11 : o.seed, n = o.n, N = n + 2, q = o.q === undefined ? 0.5 : o.q;
    const Kt = o.Kt, Kl = o.Kl, a = new Float64Array(N), b = new Float64Array(N), aPre = new Float64Array(N), bPre = new Float64Array(N), kt = new Float64Array(N), kl = new Float64Array(N);
    a[0] = aPre[0] = o.a0 === undefined ? -6 : o.a0; b[0] = bPre[0] = o.b0 === undefined ? 6 : o.b0; kt[0] = Kt(0); kl[0] = Kl(0);
    let swaps = 0, sumAbs = 0, sumSqPre = 0;
    let firstShare = NaN;
    const swapsAt = new Float64Array(N), meanAbs = new Float64Array(N), rmsPre = new Float64Array(N);
    for (let k = 1; k < N; k++) {
      aPre[k] = a[k - 1] + q * gauss(seed, k, 2); bPre[k] = b[k - 1] + q * gauss(seed, k, 3);
      kt[k] = Kt(k); kl[k] = Kl(k);
      a[k] = aPre[k] + kt[k] * (bPre[k] - aPre[k]); b[k] = bPre[k] + kl[k] * (aPre[k] - bPre[k]);
      if (k === 1) firstShare = Math.abs(a[1] - aPre[1]) / (Math.abs(a[1] - aPre[1]) + Math.abs(b[1] - bPre[1]));
      if ((b[k] - a[k]) * (bPre[k] - aPre[k]) < 0 && Math.abs(b[k] - a[k]) > 1e-9) swaps++;
      sumAbs += Math.abs(b[k] - a[k]); sumSqPre += (bPre[k] - aPre[k]) * (bPre[k] - aPre[k]);
      swapsAt[k] = swaps; meanAbs[k] = sumAbs / k; rmsPre[k] = Math.sqrt(sumSqPre / k);
    }
    return { n, a, b, aPre, bPre, kt, kl, swapsAt, meanAbs, rmsPre, q, firstShare };
  }

  const PRESETS = {
    E: { heavy: { K: 0.08, r: 1, name: 'Too heavy on the past', say: 'Too heavy on the past. The cone of what could happen next is too narrow, so the world keeps turning up outside it, and the marble walks far less than the world does.' }, light: { K: 0.97, r: 1, name: 'Too light on the past', say: 'Too light on the past. The cone is too wide: anything might be out there. The marble chases every glimpse and walks further than the world does.' }, balance: { K: null, r: 1, name: 'Balance', say: 'Balance. The cone is the size of the world\u2019s real surprises, and the marble walks as far as the world does. The extra distance is not waste when it matches the world\u2019s own wandering.' } },
    F: { hold: { Kt: 0.08, Kl: 0.08, name: 'Both hold hard' }, letgo: { Kt: 0.9, Kl: 0.9, name: 'Both let go' }, mirror: { Kt: 0.02, Kl: 0.98, name: 'One\u2019s picture becomes the other' }, balance: { Kt: 0.382, Kl: 0.618, name: 'Balance' } }
  };
  const api = { PHI, hash, gauss, bell, bowl, TEETH, count, comb, steadyK, kalmanStep, impliedR, believed, fixedGainMSE, param, runE, runF, PRESETS, clamp };
  if (typeof module === 'object' && module.exports) module.exports = api; else root.EQModel = api;
})(typeof window !== 'undefined' ? window : globalThis);
