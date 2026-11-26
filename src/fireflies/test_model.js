/* Runs the bets in BETS.md against model.js. node test_model.js > test_output_vN.txt */
const M = require('./model.js');
const SEEDS = [11, 22, 33, 44, 55], WARM = 120, RUN = 600;
const f3 = x => (isFinite(x) ? x.toFixed(3) : String(x)), f2 = x => (isFinite(x) ? x.toFixed(2) : String(x));
const out = (...a) => console.log(...a);

/* B1 */
out('B1  a lone firefly. Pass: CV of intervals between its own burst starts > 0.4');
const S1 = M.run(1, 4000, { seed: 3 });
const lone = M.stats(M.intervals(M.ownStarts(S1.flashes, 0, 4000).get(0) || []));
out(`  bursts ${lone.n + 1}, mean interval ${f2(lone.mean)} s, sd ${f2(lone.sd)}, CV ${f3(lone.cv)}, shortest ${f2(Math.min(...M.intervals(M.ownStarts(S1.flashes, 0, 4000).get(0))))} s  ${lone.cv > .4 ? 'B1 PASS' : 'B1 FAIL'}`);

/* B2, B3 (300 side), B4 (nudge on), B7 */
out('\nB2  together they keep time. Pass (4 of 5): mean collective interval <= 0.6 x lone mean, CV < lone CV / 3');
let p2 = 0, cvs = [], share300 = [], spreadsOn = [];
for (const seed of SEEDS) {
  const S = M.run(300, WARM + RUN, { seed });
  const cb = M.collective(S.flashes, WARM, WARM + RUN);
  const iv = M.stats(M.intervals(cb.map(b => b.start)));
  const sh = M.stats(cb.map(b => b.who.size / S.n));
  const pk = M.peaks(S.flashes, WARM, WARM + RUN).filter(p => p.n >= 20).map(p => p.sd).sort((a, b) => a - b);
  const med = pk.length ? pk[pk.length >> 1] : NaN; spreadsOn.push(med);
  const ok = iv.mean <= .6 * lone.mean && iv.cv < lone.cv / 3; if (ok) p2++; cvs.push(iv.cv); share300.push(sh.mean);
  out(`  seed ${seed}: collective bursts ${cb.length}, mean interval ${f2(iv.mean)} s, CV ${f3(iv.cv)}, share flashing ${f3(sh.mean)}, flashes ${S.nFlash}, unprompted starts ${S.nSpont}, median peak spread ${f3(med)} s  ${ok ? 'pass' : 'FAIL'}`);
}
out(`  B2 ${p2 >= 4 ? 'PASS' : 'FAIL'} (${p2}/5)`);

out('\nB3  it takes a crowd. Pass: mean share in a collective burst < 0.5 at 15 and > 0.8 at 300');
let share15 = [];
for (const seed of SEEDS) {
  const S = M.run(15, WARM + RUN, { seed });
  const cb = M.collective(S.flashes, WARM, WARM + RUN); const sh = M.stats(cb.map(b => b.who.size / S.n)); share15.push(sh.mean);
  out(`  seed ${seed}: 15 fireflies, collective bursts ${cb.length}, share ${f3(sh.mean)}`);
}
const m15 = M.stats(share15).mean, m300 = M.stats(share300).mean;
out(`  mean share at 15: ${f3(m15)}, at 300: ${f3(m300)}  B3 ${m15 < .5 && m300 > .8 ? 'PASS' : 'FAIL'}`);

out('\nB4  flashes inside a burst line up. Pass: median peak spread < 0.08 s with nudging, > 0.12 s without');
const spreadsOff = [];
for (const seed of SEEDS) {
  const S = M.run(300, WARM + RUN, { seed, nudge: 0 });
  const pk = M.peaks(S.flashes, WARM, WARM + RUN).filter(p => p.n >= 20).map(p => p.sd).sort((a, b) => a - b);
  spreadsOff.push(pk.length ? pk[pk.length >> 1] : NaN);
  out(`  seed ${seed}: nudging off, peaks of 20+ flashes ${pk.length}, median spread ${f3(spreadsOff[spreadsOff.length - 1])} s`);
}
const on = M.stats(spreadsOn).mean, off = M.stats(spreadsOff).mean;
out(`  mean of medians: on ${f3(on)} s, off ${f3(off)} s  B4 ${on < .08 && off > .12 ? 'PASS' : 'FAIL'}`);

out('\nB5  hidden from one another. Pass: pooled CV of each firefly\'s own burst intervals > 0.4');
{ const S = M.run(300, WARM + RUN, { seed: 11, see: false }); const st = M.ownStarts(S.flashes, WARM, WARM + RUN); let all = [];
  for (const ts of st.values()) all = all.concat(M.intervals(ts)); const s = M.stats(all);
  const cb = M.collective(S.flashes, WARM, WARM + RUN), sh = M.stats(cb.map(b => b.who.size / S.n));
  out(`  intervals ${s.n}, mean ${f2(s.mean)} s, CV ${f3(s.cv)}; swarm share per collective run ${f3(sh.mean)}  B5 ${s.cv > .4 ? 'PASS' : 'FAIL'}`); }

out('\nB6  a white torch silences what it lights (code check). Pass: flashes per firefly-minute in the beam fall >= 80%');
{ const S = M.run(300, WARM, { seed: 22 }); const dt = 1 / 120, beam = i => M.inBeam(Object.assign({}, S, { P: Object.assign({}, S.P, { torch: 1, torchX: 0, torchHalf: .35, torchHalfV: .3 }) }), i);
  /* v2: a flash is counted as in the beam by where the firefly is when it flashes (v1 used its position at the end of the window) */
  let inFl = 0, occ = 0, steps = 0; S.onFlash = i => { if (beam(i)) inFl++; };
  const count = (from, to) => { const fl = inFl; inFl = 0; const r = { fl, inb: occ / Math.max(1, steps) }; occ = 0; steps = 0; return r; };
  const stepCount = () => { M.step(S, dt); steps++; for (let i = 0; i < S.n; i++) if (beam(i)) occ++; };
  /* before: run 120 s with no torch */
  inFl = 0; for (let s = 0; s < 120 / dt; s++) stepCount(); const b = count(WARM, WARM + 120);
  S.P.torch = 1; S.P.torchX = 0; S.P.torchHalf = .35; S.P.torchHalfV = .3;
  for (let s = 0; s < 120 / dt; s++) stepCount(); const a = count(WARM + 120, WARM + 240);
  const rb = b.fl / Math.max(1, b.inb) / 2, ra = a.fl / Math.max(1, a.inb) / 2, fall = 1 - ra / rb;
  out(`  in the beam: ${f2(b.inb)} fireflies on average before, ${f2(a.inb)} during; flashes per firefly-minute ${f2(rb)} before, ${f2(ra)} during; fall ${(fall * 100).toFixed(1)}%  B6 ${fall >= .8 ? 'PASS' : 'FAIL'}`); }

out('\nB7  reported, not judged. Collective interval CV against the CRR candidate value 1/(2 pi) = 0.159');
out(`  CVs: ${cvs.map(f3).join(', ')}; mean ${f3(M.stats(cvs).mean)}`);
