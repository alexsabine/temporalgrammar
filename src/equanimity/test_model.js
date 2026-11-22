/* Model tests for equanimity.html. The bets are in BETS.md and were written first. Run: node test_model.js */
const M = require('./model.js');
let pass = 0, fail = 0;
function bet(name, ok, detail) { (ok ? pass++ : fail++); console.log((ok ? 'PASS  ' : 'FAIL  ') + name + (detail ? '   ' + detail : '')); }
const f6 = v => v.toFixed(6);

/* B1 */
{ let P = 1, Pm = 0, K = 0; for (let i = 0; i < 200; i++) { const s = M.kalmanStep(0, P, 0, 1, 1); Pm = s.Pm; K = s.K; P = s.P; }
  bet('B1 Kalman numbers at r = 1', f6(K) === '0.618034' && f6(1 - K) === '0.381966' && f6(Pm) === '1.618034' && f6(P) === '0.618034', `K ${f6(K)} kept ${f6(1 - K)} before ${f6(Pm)} after ${f6(P)}`);
  const gained = 1 / P - 1 / Pm, lost = 1 / P - 1 / (P + 1);
  bet('B1 one unit of information in, one out', Math.abs(gained - 1) < 1e-9 && Math.abs(lost - 1) < 1e-9, `gained ${gained.toFixed(10)} lost ${lost.toFixed(10)}`); }
/* B2 */
{ let ok = true, worst = 0; for (const r of [0.01, 0.1, 0.236, 1, 4, 100]) { const K = M.steadyK(r), res = Math.abs(K * K + r * K - r); let P = 1, k = 0; for (let i = 0; i < 400; i++) { const s = M.kalmanStep(0, P, 0, r, 1); P = s.P; k = s.K; }
    worst = Math.max(worst, res, Math.abs(k - K)); if (res > 1e-12 || Math.abs(k - K) > 1e-9) ok = false; }
  bet('B2 steady gain solves K^2 + rK - r = 0 and the recursion reaches it', ok, 'worst error ' + worst.toExponential(2));
  const k3 = M.steadyK(1 / Math.pow(M.PHI, 3)); bet('B2 K = 0.382 is best only at r = 1/phi^3', f6(k3) === '0.381966', `r = ${(1 / Math.pow(M.PHI, 3)).toFixed(6)} gives K ${f6(k3)}`); }
/* B3, B4 */
{ const n = 20000, seeds = [7, 1795, 3, 41, 99, 2024], Ks = { heavy: 0.08, light: 0.97, balance: 1 / M.PHI }, sum = { heavy: 0, light: 0, balance: 0 }; let order = true; const rows = [];
  for (const sd of seeds) { const v = {}; for (const k in Ks) { v[k] = M.runE({ seed: sd, n, K: () => Ks[k] }).mse[n]; sum[k] += v[k]; } if (!(v.heavy > v.light && v.light > v.balance)) order = false; rows.push(`seed ${sd}: ${v.heavy.toFixed(3)} / ${v.light.toFixed(3)} / ${v.balance.toFixed(3)}`); }
  bet('B3 heavy > light > balance for every seed', order, rows.join('; '));
  for (const k in Ks) { const mean = sum[k] / seeds.length, th = M.fixedGainMSE(Ks[k], 1, 1), tol = k === 'heavy' ? 0.15 : 0.05; bet(`B3 ${k} MSE near the textbook value`, Math.abs(mean / th - 1) < tol, `measured ${mean.toFixed(4)} textbook ${th.toFixed(4)} (${((mean / th - 1) * 100).toFixed(1)}%)`); }
  let best = 0, bm = 1e9; for (let K = 0.02; K < 0.981; K += 0.02) { const v = M.runE({ seed: 7, n, K: () => K }).mse[n]; if (v < bm) { bm = v; best = K; } }
  bet('B3 lowest measured MSE is near K = 0.618', best >= 0.56 && best <= 0.68, `lowest at K = ${best.toFixed(2)} (MSE ${bm.toFixed(4)})`);
  const auto = M.runE({ seed: 7, n }); bet('B3 the filter left to itself settles at K = 0.618034', f6(auto.K[n]) === '0.618034', `K ${f6(auto.K[n])} MSE ${auto.mse[n].toFixed(4)}`);
  const c = {}; for (const k in Ks) c[k] = M.runE({ seed: 7, n, K: () => Ks[k] }).caught[n];
  bet('B4 share of meetings with the world inside the fan', c.balance > 0.93 && c.balance < 0.97 && c.heavy < 0.35 && c.light > 0.99, `balance ${c.balance.toFixed(3)} heavy ${c.heavy.toFixed(3)} light ${c.light.toFixed(3)}`); }
/* B5 */
{ const one = (Kt, Kl) => { const r = M.runF({ n: 1, q: 0, a0: 0, b0: 1, Kt: () => Kt, Kl: () => Kl }); return r.b[1] - r.a[1]; };
  const want = [[.382, .618, 0], [.5, .5, 0], [.618, .618, -.236], [.382, .382, .236], [.2, .8, 0]]; let ok = true; const rows = [];
  for (const [a, b, g] of want) { const v = one(a, b); rows.push(`(${a}, ${b}) ${v >= 0 ? '+' : ''}${v.toFixed(3)}`); if (Math.abs(v - g) > 5e-4 || Math.abs(v - (1 - a - b)) > 1e-12) ok = false; }
  bet('B5 gap after one exchange = 1 - Kt - Kl, as in compare.py', ok, rows.join('  '));
  const lg = M.runF({ n: 10, q: 0, Kt: () => .9, Kl: () => .9 }); bet('B5 both let go: swap sides at each of ten exchanges', lg.swapsAt[10] === 10, 'swaps ' + lg.swapsAt[10]);
  const hh = M.runF({ n: 10, q: 0, Kt: () => .08, Kl: () => .08 }), ratio = (hh.b[10] - hh.a[10]) / 12; bet('B5 both hold hard: gap after ten exchanges is 0.84^10 of the start', Math.abs(ratio - Math.pow(.84, 10)) < 1e-9, `ratio ${ratio.toFixed(6)} (so they do approach, 16% each time)`);
  const n = 5000, res = {}; let never = true; for (const k in M.PRESETS.F) { const p = M.PRESETS.F[k], r = M.runF({ n, Kt: () => p.Kt, Kl: () => p.Kl }); res[k] = r.rmsPre[n]; for (let i = 1; i <= n; i++) if (r.bPre[i] - r.aPre[i] === 0) never = false; }
  bet('B5 with drift the gap before an exchange is never zero', never);
  bet('B5 rms gap before an exchange at balance near 0.707', Math.abs(res.balance / Math.SQRT1_2 - 1) < 0.1, `balance ${res.balance.toFixed(3)}`);
  bet('B5 hold and let-go keep a larger gap than balance', res.hold > res.balance && res.letgo > res.balance, `hold ${res.hold.toFixed(3)} letgo ${res.letgo.toFixed(3)} mirror ${res.mirror.toFixed(3)} balance ${res.balance.toFixed(3)}`); }
/* B6 */
{ const N = 9001, x0 = 0, x1 = 9, dx = (x1 - x0) / (N - 1);   /* v1 used 8,001 points, which puts no grid point on a tooth: see the log in BETS.md */ let run = 0, ok = true, runI = 0, worst = 0, okI = true;
  for (let i = 0; i < N; i++) { const x = x0 + i * dx; const tooth = M.TEETH.some(t => Math.abs(t - x) < dx / 2) ? 1 : 0; run += tooth; if (run !== M.count(x + dx / 2)) ok = false;
    runI += M.comb(x, 0.01) * dx; const near = M.TEETH.some(t => Math.abs(t - x) < 0.08); if (!near) { const e = Math.abs(runI - M.count(x)); worst = Math.max(worst, e); if (e > 1e-6) okI = false; } }
  bet('B6 running sum of the comb equals the staircase', ok);
  bet('B6 running integral of a comb of narrow bells equals the staircase away from the teeth', okI, 'worst error ' + worst.toExponential(2)); }
/* B7, B8 */
{ const W = [0.05, 0.3, 0.5, 1, 2, 6, 20]; let ok = true, worst = 0, areaOk = true, wa = 0;
  for (const s of W) { for (let i = -300; i <= 300; i++) { const x = i / 10 * s, a = -Math.log(M.bell(x, s) / M.bell(0, s)), b = M.bowl(x, s), e = Math.abs(a - b); const rel = e / Math.max(1e-300, Math.abs(b)); if (e > 1e-12 && rel > 1e-9) ok = false; worst = Math.max(worst, Math.min(e, rel)); }
    let A = 0; const n = 400000, lim = 12 * s, dx = 2 * lim / n; for (let i = 0; i <= n; i++) A += M.bell(-lim + i * dx, s) * (i === 0 || i === n ? .5 : 1) * dx; wa = Math.max(wa, Math.abs(A - 1)); if (Math.abs(A - 1) > 1e-6) areaOk = false; }
  bet('B7 bowl = minus log of the bell (from its lowest point) at every width', ok, 'worst error ' + worst.toExponential(2));
  bet('B7 the bowl becomes a well, then level ground', M.bowl(0.1, 1e-4) > 1e5 && M.bowl(0.1, 100) < 1e-6, `at x = 0.1: width 0.0001 gives ${M.bowl(0.1, 1e-4).toExponential(2)}, width 100 gives ${M.bowl(0.1, 100).toExponential(2)}`);
  bet('B8 area under the bell is 1 at every width', areaOk, 'worst error ' + wa.toExponential(2));
  let I = 0; const s = 0.001, n = 200000, lim = 12 * s, dx = 2 * lim / n; for (let i = 0; i <= n; i++) { const x = -lim + i * dx; I += M.bell(x, s) * Math.cos(x) * dx; }
  bet('B8 a narrow bell picks out one value', Math.abs(I - 1) < 1e-6, 'integral ' + I.toFixed(9)); }
/* B9 */
{ const a = M.runE({ seed: 7, n: 500 }), b = M.runE({ seed: 7, n: 500 }); let same = true; for (const k of ['x', 'y', 'm', 'Pm', 'P', 'K', 'mse']) for (let i = 0; i < 502; i++) if (a[k][i] !== b[k][i]) same = false;
  const c = M.runF({ n: 500, Kt: () => .382, Kl: () => .618 }), d = M.runF({ n: 500, Kt: () => .382, Kl: () => .618 }); for (const k of ['a', 'b', 'aPre', 'bPre']) for (let i = 0; i < 502; i++) if (c[k][i] !== d[k][i]) same = false;
  bet('B9 same seed, same run', same);
  const sp = { from: 2, to: 4, start: 100, end: 200 }; bet('B9 a dial given as a ramp', M.param(sp, 0) === 2 && M.param(sp, 300) === 4 && Math.abs(M.param(sp, 150) - 3) < 1e-12 && M.param(0.5, 9) === 0.5 && M.param(undefined, 9, 7) === 7); }
/* B10 */
{ let ok = true; for (const K of [0.08, 0.382, 0.618, 0.97]) if (Math.abs(M.steadyK(M.impliedR(K)) - K) > 1e-12) ok = false; bet('B10 a fixed gain implies a belief about drift', ok, `K 0.08 believes r = ${M.impliedR(.08).toFixed(5)}; K 0.97 believes r = ${M.impliedR(.97).toFixed(2)}`); }
/* B12, B13, B14: added after the coordinator's surplus.py */
{ const n = 20000, got = {}; for (const [nm, K] of [['heavy', .08], ['balance', 1 / M.PHI], ['light', .97]]) { const r = M.runE({ seed: 7, n, K: () => K }); got[nm] = { cone: M.believed(K, 1).Pm, err: r.ahead[n], path: r.pathM[n] / r.pathW[n] }; }
  const g = got, s = k => `${k}: cone ${g[k].cone.toFixed(3)} surprise ${g[k].err.toFixed(3)} path ${g[k].path.toFixed(3)}`;
  bet('B12 cone values', g.heavy.cone.toFixed(3) === '0.087' && g.balance.cone.toFixed(3) === '1.618' && g.light.cone.toFixed(3) === '32.333');
  bet('B12 measured surprise near 6.52 (heavy) and 1.94 (light)', Math.abs(g.heavy.err / 6.52 - 1) < .1 && Math.abs(g.light.err / 1.94 - 1) < .05, s('heavy') + '; ' + s('light'));
  bet('B12 at balance the cone is the size of the surprise', Math.abs(g.balance.cone / g.balance.err - 1) < .05, s('balance') + ` ratio ${(g.balance.cone / g.balance.err).toFixed(3)}`);
  bet('B13 path walked / world path: 0.22, 1.00, 1.66', Math.abs(g.heavy.path - .22) < .03 && Math.abs(g.balance.path - 1) < .03 && Math.abs(g.light.path - 1.66) < .03, `${g.heavy.path.toFixed(3)} / ${g.balance.path.toFixed(3)} / ${g.light.path.toFixed(3)}`);
  const m0 = M.runF({ n: 1, q: 0, a0: 0, b0: 1, Kt: () => .02, Kl: () => .98 }), m1 = M.runF({ n: 3, Kt: () => .02, Kl: () => .98 });
  bet('B14 they meet where the first one stood (0.02 of the way)', Math.abs(m0.a[1] - .02) < 1e-12 && Math.abs(m0.b[1] - .02) < 1e-12 && Math.abs(m1.firstShare - .02) < 1e-9, `first at ${m0.a[1].toFixed(3)}, second at ${m0.b[1].toFixed(3)}; share moved by the first ${m1.firstShare.toFixed(4)}`); }
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
