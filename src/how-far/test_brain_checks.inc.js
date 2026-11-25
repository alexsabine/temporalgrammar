// The brain model's own checks, sections 3 to 8 of the earlier seven-level page's test, unchanged. M is the brain model (EV.HFD).
// ── 3. Counting and the size of the data ──
{
  const S = M.SRC, D = M.DATA, indep = 1.4e15 / 1.05 * (1508 / 1.04 * 1000);
  ok('3a whole-brain data size equals an independent calculation from the paper’s figures (bet: 1e-12 relative)', rel(D.brainBytes, indep) < 1e-12, M.fmtBytes(D.brainBytes));
  ok('3b it comes to about two zettabytes (bet: between 1.5 and 2.5 × 10^21 bytes)', D.brainBytes > 1.5e21 && D.brainBytes < 2.5e21);
  const vox = 1.05 / (4e-6 * 4e-6 * 33e-6);
  ok('3c the H01 volume at 4 × 4 × 33 nm is about two thousand million million voxels: a "petavoxel" sample (bet: equals 1.05 mm³ / voxel volume to 1e-9, and lies between 1e15 and 3e15)', rel(D.voxels, vox) < 1e-9 && vox > 1e15 && vox < 3e15, vox.toExponential(3));
  ok('3d at one byte a voxel that is within 15% of the 1.8 petabytes of raw data the paper reports (bet: < 15%)', rel(vox, S.h01.rawBytes) < 0.15, (100 * rel(vox, S.h01.rawBytes)).toFixed(1) + '%');
  ok('3e the cerebellum holds about four neurons in five (bet: share between 0.78 and 0.82)', D.cerebellumShare > 0.78 && D.cerebellumShare < 0.82, D.cerebellumShare.toFixed(3));
  ok('3f neurons and glia do not exceed the cells counted in H01, and glia outnumber neurons about 2 to 1 (bet: ratio between 1.9 and 2.1)', S.h01.neurons + S.h01.glia <= S.h01.cells && S.h01.glia / S.h01.neurons > 1.9 && S.h01.glia / S.h01.neurons < 2.1, (S.h01.glia / S.h01.neurons).toFixed(2));
  const c = M.cubeCounts(1e-3);
  ok('3g a cube 1 mm wide holds the H01 densities (bet: neurons = 16,087/1.05 and synapses = 150e6/1.05 to 1e-9)', rel(c.neurons, 16087 / 1.05) < 1e-9 && rel(c.synapses, 150e6 / 1.05) < 1e-9, Math.round(c.neurons) + ' neurons, ' + c.synapses.toExponential(3) + ' synapses');
  ok('3h counts scale with the cube of the frame (bet: a frame ten times narrower holds 1,000 times fewer, to 1e-9)', rel(M.cubeCounts(1e-4).synapses * 1000, c.synapses) < 1e-9);
}

// ── 4. The cable ──
{
  const lam = M.lambdaUm(1), tau = M.tauMs();
  ok('4a λ for a 1 µm branch equals sqrt((d/4)(Rm/Ri)) worked independently (bet: 577.35 µm to 0.01)', Math.abs(lam - Math.sqrt((1e-4 / 4) * (20000 / 150)) * 1e4) < 0.01, lam.toFixed(2) + ' µm');
  ok('4b τ = Rm Cm = 20 ms (bet: exact to 1e-9)', Math.abs(tau - 20) < 1e-9);
  // a uniform cable sealed at both ends, against the analytic steady state
  const T = M.buildTree([{ from: -1, x: 0, y: 0, len: 1000, d: 1, ang: 0 }], { soma: false }), xi = 70, v = M.steady(T, { [xi]: 100 }); let worst = 0;
  for (const k of [0, 10, 30, 50, 70, 85, 99]) worst = Math.max(worst, rel(v[k], M.sealedCable(100, (k + 0.5) * 10, (xi + 0.5) * 10, 1000, 1)));
  ok('4c compartmental steady state of a sealed 1 mm cable matches cosh(X<) cosh(L − X>) / sinh(L) at seven places (bet: < 0.5%)', worst < 0.005, (100 * worst).toFixed(4) + '%');
  // a long cable decays as e^(−x/λ)
  const TL = M.buildTree([{ from: -1, x: 0, y: 0, len: 6000, d: 1, ang: 0 }], { soma: false }), vl = M.steady(TL, { 0: 100 }), fit = -1000 / Math.log(vl[150] / vl[50]);
  ok('4d on a long cable the voltage falls by e for every λ: fitted λ against the formula (bet: within 1%)', rel(fit, lam) < 0.01, fit.toFixed(1) + ' µm against ' + lam.toFixed(1));
  // an isopotential compartment discharges with time constant τ
  const T1 = M.buildTree([], { soma: { x: 0, y: 0, d: 20 } }), D1 = M.Dend(T1); D1.v[0] = 10; for (let k = 0; k < 2000; k++) D1.step(0.01);
  ok('4e a single compartment left alone decays to 1/e in one τ (bet: v(20 ms)/v(0) within 1% of e⁻¹)', rel(D1.v[0] / 10, Math.exp(-1)) < 0.01, (D1.v[0] / 10).toFixed(5));
  // Rall's 3/2 power law: a branching tree that is exactly one cylinder
  const dc = 2 / Math.pow(2, 2 / 3), l2 = 0.5 * M.lambdaUm(dc), TB = M.buildTree([{ from: -1, x: 0, y: 0, len: 300, d: 2, ang: 0 }, { from: 0, len: l2, d: dc, ang: 0.5 }, { from: 0, len: l2, d: dc, ang: -0.5 }], { soma: false, dx: 5 });
  const Ltot = 300 / M.lambdaUm(2) + 0.5, Rin = M.steady(TB, { 0: 100 })[0] / 100 * 1000, Ran = M.rInfMOhm(2) / Math.tanh(Ltot);
  ok('4f Rall’s equivalent cylinder: a tree obeying the 3/2 power rule has the input resistance R∞ coth(L) of one cylinder (bet: < 1%)', rel(Rin, Ran) < 0.01, Rin.toFixed(2) + ' MΩ against ' + Ran.toFixed(2));
  // the tree solver against dense elimination, and reciprocity
  const N = M.NEURON, n = N.n, A = Array.from({ length: n }, () => new Float64Array(n + 1)), site = M.HOME_SITE;
  for (let i = 0; i < n; i++) { A[i][i] = N.gm[i] + N.gsum[i]; const p = N.parent[i]; if (p >= 0) { A[i][p] -= N.ga[i]; A[p][i] -= N.ga[i]; } } A[site][n] = 100;
  for (let c = 0; c < n; c++) { let p = c; for (let i = c; i < n; i++) if (Math.abs(A[i][c]) > Math.abs(A[p][c])) p = i; const t = A[c]; A[c] = A[p]; A[p] = t; for (let i = c + 1; i < n; i++) { const f = A[i][c] / A[c][c]; if (f) for (let j = c; j <= n; j++) A[i][j] -= f * A[c][j]; } }
  const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = A[i][n]; for (let j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; }
  const vs = M.steady(N, { [site]: 100 }); let wd = 0; for (let i = 0; i < n; i++) wd = Math.max(wd, Math.abs(vs[i] - x[i]));
  ok('4g the tree solver agrees with dense Gaussian elimination on the ' + n + '-compartment neuron (bet: max difference < 1e-9 mV)', wd < 1e-9, wd.toExponential(2));
  const back = M.steady(N, { 0: 100 });
  ok('4h reciprocity: current at the branch seen at the cell body equals current at the cell body seen at the branch (bet: 1e-9 relative)', rel(vs[0], back[site]) < 1e-9, vs[0].toFixed(4) + ' mV');
  let monoA = true, pa = 2; for (const k of M.PATH.slice(1)) { const s = M.steady(N, { [k]: 100 }), a = s[0] / s[k]; if (!(a < pa + 1e-12) || a >= 1) monoA = false; pa = a; }
  ok('4i along the path the visitor can choose, less of the voltage reaches the cell body the further out the input (bet: the ratio falls at every step and is below 1)', monoA, 'at the far end ' + pa.toFixed(3));
  ok('4j the page’s claim that a real tree attenuates more than e^(−x/λ) holds at the default site (bet: computed ratio < e^(−x/λ))', vs[0] / vs[site] < Math.exp(-N.dist[site] / M.lambdaUm(N.d[site])), (vs[0] / vs[site]).toFixed(3) + ' against ' + Math.exp(-N.dist[site] / M.lambdaUm(N.d[site])).toFixed(3));
  // the dendritic threshold
  const on = k => M.volley(N, site, k, { nmda: true }).soma, off = k => M.volley(N, site, k, { nmda: false }).soma, o4 = on(4), o8 = on(8), o16 = on(16), o32 = on(32);
  ok('4k with NMDA receptors, 8 synapses give the cell body more than twice what 4 give (bet: ratio > 2; set after exploration)', o8 / o4 > 2, (o8 / o4).toFixed(2));
  ok('4l and the branch saturates: 32 give less than twice what 16 give (bet: ratio < 2; set after exploration)', o32 / o16 < 2, (o32 / o16).toFixed(2));
  ok('4m without NMDA receptors doubling the synapses never doubles the response (bet: ratio < 2 for 2→4, 4→8, 8→16, 16→32)', [2, 4, 8, 16].every(k => off(2 * k) / off(k) < 2), [2, 4, 8, 16].map(k => (off(2 * k) / off(k)).toFixed(2)).join(', '));
  ok('4n the magnesium block lifts with depolarisation (bet: B(−70) < 0.1, B(0) > 0.7, and B rises throughout)', M.mgBlock(-70) < 0.1 && M.mgBlock(0) > 0.7 && Array.from({ length: 100 }, (_, k) => M.mgBlock(-90 + k) < M.mgBlock(-89 + k)).every(Boolean), M.mgBlock(-70).toFixed(3) + ', ' + M.mgBlock(0).toFixed(3));
  note('4o volleys at the default site', '4 → ' + o4.toFixed(2) + ' mV, 8 → ' + o8.toFixed(2) + ', 16 → ' + o16.toFixed(2) + ', 32 → ' + o32.toFixed(2) + ' at the cell body');
}

// ── 5. Hodgkin and Huxley ──
{
  const r0 = M.hhRun(0), r2 = M.hhRun(2), r10 = M.hhRun(10);
  ok('5a with no current the membrane stays at rest (bet: within 0.5 mV of −65 after 100 ms, no impulses)', Math.abs(r0.end + 65) < 0.5 && r0.spikes === 0, r0.end.toFixed(3));
  ok('5b 2 µA/cm² is below threshold and 10 µA/cm² fires repeatedly (bet: 0 impulses, and at least 5; set after exploration)', r2.spikes === 0 && r10.spikes >= 5, r2.spikes + ' and ' + r10.spikes);
  ok('5c an impulse overshoots zero and peaks below the sodium potential (bet: peak between +20 and +50 mV)', r10.peak > 20 && r10.peak < M.HH.ENa, r10.peak.toFixed(1) + ' mV');
  let worst = 0; for (const V of [-80, -65, -40, -20, 0, 30]) { let n = M.HH.nInf(-65); const dt = 0.001; for (let t = 0; t < 5; t += dt) n += dt * (M.HH.an(V) * (1 - n) - M.HH.bn(V) * n); const an = M.HH.nInf(V) - (M.HH.nInf(V) - M.HH.nInf(-65)) * Math.exp(-5 / M.HH.tauN(V)); worst = Math.max(worst, Math.abs(n - an)); }
  ok('5d after a voltage step the gate n relaxes exponentially to n∞ with time constant 1/(α+β) (bet: < 1e-3 at 5 ms, six voltages)', worst < 1e-3, worst.toExponential(2));
  ok('5e the rate functions are finite at their removable singularities (bet: αn(−55) = 0.1 and αm(−40) = 1 to 1e-6)', Math.abs(M.HH.an(-55) - 0.1) < 1e-6 && Math.abs(M.HH.am(-40) - 1) < 1e-6);
}

// ── 6. Nernst, single channels, ions ──
{
  const [K, Na, Cl, Ca] = M.MAMMAL.map(m => M.nernst(m[1], m[2], m[3]));
  ok('6a Nernst potentials for the round mammalian concentrations match the usual textbook figures (bet: K −89 ± 1.5, Na +61 ± 1.5, Cl −64 ± 1.5, Ca +132 ± 2 mV)', Math.abs(K + 89) < 1.5 && Math.abs(Na - 61) < 1.5 && Math.abs(Cl + 64) < 1.5 && Math.abs(Ca - 132) < 2, [K, Na, Cl, Ca].map(v => v.toFixed(1)).join(', '));
  ok('6b a tenfold concentration ratio is worth 61.5 mV at 37 °C and 58.2 mV at 20 °C (bet: within 0.1 mV)', Math.abs(M.nernst(1, 10, 1) - 61.54) < 0.1 && Math.abs(M.nernst(1, 10, 1, 293.15) - 58.17) < 0.1, M.nernst(1, 10, 1).toFixed(2) + ', ' + M.nernst(1, 10, 1, 293.15).toFixed(2));
  ok('6c reversing the ratio or the charge reverses the sign (bet: exact to 1e-12)', Math.abs(M.nernst(1, 5, 140) + M.nernst(1, 140, 5)) < 1e-12 && Math.abs(M.nernst(1, 5, 140) + M.nernst(-1, 5, 140)) < 1e-12);
  ok('6d i = γ(V − E): 20 pS and 57 mV of driving force give 1.14 pA (bet: 1e-12)', Math.abs(M.unitCurrent(-20) - 1.14) < 1e-12 && M.unitCurrent(M.HH.EK) === 0);
  ok('6e one picoamp is 6,241.5 elementary charges a millisecond (bet: equals 1e-12 / e / 1000 to 1e-9, and 6,241.5 ± 0.1)', rel(M.ionsPerMs(1), 1e-12 / 1.602176634e-19 / 1000) < 1e-9 && Math.abs(M.ionsPerMs(1) - 6241.5) < 0.1, M.ionsPerMs(1).toFixed(2));
  // a population of channels against the binomial prediction
  const N = 200, V = -20, C = M.Channels(N, V, 11), S = M.channelStats(N, V); let s1 = 0, s2 = 0, m = 0; for (let k = 0; k < 8000; k++) C.step(0.05); for (let k = 0; k < 400000; k++) { const o = C.step(0.05); if (k % 4 === 0) { s1 += o; s2 += o * o; m++; } }
  const mean = s1 / m, variance = s2 / m - mean * mean;
  ok('6f 200 simulated channels for 20 s: mean number open against N p (bet: within 3%)', rel(mean, N * S.p) < 0.03, mean.toFixed(2) + ' against ' + (N * S.p).toFixed(2));
  ok('6g and its variance against N p (1 − p) (bet: within 12%)', rel(variance, N * S.p * (1 - S.p)) < 0.12, variance.toFixed(2) + ' against ' + (N * S.p * (1 - S.p)).toFixed(2));
  ok('6h in current: mean = N p i and σ² = N p (1 − p) i² as channelStats reports them (bet: 1e-9 relative)', rel(S.mean, N * S.p * S.i) < 1e-9 && rel(S.sd * S.sd, N * S.p * (1 - S.p) * S.i * S.i) < 1e-9);
  const one = M.Channels(1, V, 12); let dwell = 0, nd = 0, cur = 0; for (let k = 0; k < 1200000; k++) { const o = one.step(0.05); if (o) cur++; else if (cur) { dwell += cur * 0.05; nd++; cur = 0; } }
  ok('6i one channel: mean time spent open against 1/(4β), the rate at which any of four gates shuts (bet: within 8%)', rel(dwell / nd, S.openMs) < 0.08, (dwell / nd).toFixed(3) + ' ms against ' + S.openMs.toFixed(3) + ' over ' + nd + ' openings');
  ok('6j the flicker as a fraction of the mean falls as 1/√N (bet: cv(100)/cv(10,000) = 10 to 1e-9)', rel(M.channelStats(100, V).cv / M.channelStats(10000, V).cv, 10) < 1e-9);
  const cvSim = Nn => { const c = M.Channels(Nn, V, 13 + Nn); let a = 0, b = 0, q = 0; for (let k = 0; k < 4000; k++) c.step(0.05); for (let k = 0; k < 120000; k++) { const o = c.step(0.05); if (k % 4 === 0) { a += o; b += o * o; q++; } } const mu = a / q; return Math.sqrt(b / q - mu * mu) / mu; };
  const c10 = cvSim(10), c1000 = cvSim(1000);
  ok('6k simulated: 1,000 channels flicker about a tenth as much, in proportion, as 10 (bet: ratio between 7.5 and 12.5)', c10 / c1000 > 7.5 && c10 / c1000 < 12.5, (c10 / c1000).toFixed(2));
  const C2 = M.Channels(4000, -65, 14); C2.setV(0); let o = 0; for (let k = 0; k < 40; k++) o = C2.step(0.05);
  const n2 = M.HH.nInf(0) - (M.HH.nInf(0) - M.HH.nInf(-65)) * Math.exp(-2 / M.HH.tauN(0));
  ok('6l after a step from −65 to 0 mV the fraction open follows n(t)⁴, the Hodgkin–Huxley potassium current (bet: within 0.03 at 2 ms, 4,000 channels)', Math.abs(o / 4000 - Math.pow(n2, 4)) < 0.03, (o / 4000).toFixed(3) + ' against ' + Math.pow(n2, 4).toFixed(3));
}

// ── 7. Quantal release and diffusion ──
{
  const n = 5, p = 0.3, cv = 0.25, q = M.quantal(n, p, 1, cv), rr = M.rng(99), trials = 200000; let s1 = 0, s2 = 0, f = 0; const cnt = new Array(n + 1).fill(0);
  for (let k = 0; k < trials; k++) { const o = M.release(rr, n, p, 1, cv); s1 += o.amp; s2 += o.amp * o.amp; if (o.k === 0) f++; cnt[o.k]++; }
  const mean = s1 / trials, variance = s2 / trials - mean * mean;
  ok('7a 200,000 simulated impulses: mean response against n p q (bet: within 1%)', rel(mean, n * p) < 0.01 && rel(q.mean, 1.5) < 1e-12, mean.toFixed(4) + ' against ' + q.mean);
  ok('7b variance against n p q² (1 − p + cv²) (bet: within 3%)', rel(variance, q.variance) < 0.03 && rel(q.variance, n * p * (1 - p + cv * cv)) < 1e-12, variance.toFixed(4) + ' against ' + q.variance.toFixed(4));
  ok('7c failures against (1 − p)ⁿ (bet: within 0.005)', Math.abs(f / trials - Math.pow(0.7, 5)) < 0.005 && rel(q.fail, Math.pow(0.7, 5)) < 1e-12, (f / trials).toFixed(4) + ' against ' + q.fail.toFixed(4));
  ok('7d the number released follows the binomial distribution (bet: every frequency within 0.005 of the prediction; the prediction sums to 1 to 1e-12)', cnt.every((c, k) => Math.abs(c / trials - q.pmf[k]) < 0.005) && Math.abs(q.pmf.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  ok('7e with cv = 0 the variance is the pure binomial n p (1 − p) q² (bet: 1e-12)', rel(M.quantal(8, 0.4, 2, 0).variance, 8 * 0.4 * 0.6 * 4) < 1e-12);
  const t = M.diffusionUs(20), D = 0.33, walkers = 40000; let sq = 0; const steps = 50, sd = Math.sqrt(2 * D * (t / 1000) / steps); for (let w = 0; w < walkers; w++) { let x = 0; for (let k = 0; k < steps; k++) x += sd * M.gauss(rr); sq += x * x; }
  ok('7f t = x²/2D for a 20 nm cleft and D = 0.33 µm²/ms is 0.61 µs (bet: 0.606 ± 0.001)', Math.abs(t - 0.606) < 0.001, t.toFixed(4) + ' µs');
  ok('7g 40,000 random walkers released for that long have wandered 20 nm, root mean square (bet: within 2%)', rel(Math.sqrt(sq / walkers) * 1000, 20) < 0.02, (Math.sqrt(sq / walkers) * 1000).toFixed(2) + ' nm');
}

// ── 8. Instruments and the pore ──
{
  ok('8a a 300 kV electron has a wavelength of 1.97 pm, and a 100 kV electron 3.70 pm (bet: 1.969 ± 0.002 and 3.701 ± 0.005 pm)', Math.abs(M.electronLambda(3e5) * 1e12 - 1.969) < 0.002 && Math.abs(M.electronLambda(1e5) * 1e12 - 3.701) < 0.005, (M.electronLambda(3e5) * 1e12).toFixed(4) + ', ' + (M.electronLambda(1e5) * 1e12).toFixed(4));
  ok('8b without the relativistic term the 300 kV figure would be 2.24 pm: the correction matters (bet: non-relativistic h/√(2meV) is 12–15% longer)', (() => { const nr = M.PHYS.h / Math.sqrt(2 * M.PHYS.me * M.PHYS.e * 3e5) / M.electronLambda(3e5); return nr > 1.12 && nr < 1.15; })());
  ok('8c Abbe: 920 nm light through NA 1.0 resolves 460 nm; halving the wavelength halves the limit (bet: exact)', M.abbe(920, 1) === 460 && M.abbe(460, 1) === 230);
  ok('8d every instrument in the table resolves less than it takes in, and only MRI and the two-photon microscope are marked alive (bet: exactly those two)', M.INSTRUMENTS.every(I => I.res < I.fov) && M.INSTRUMENTS.filter(I => I.alive).map(I => I.name).join('|') === 'MRI, living person|Two-photon microscope, living animal');
  ok('8e the finest detail in the table is the 1.22 Å of Nakane et al. and nothing on the page claims finer (bet: minimum resolution = 1.22e-10 m)', Math.abs(Math.min(...M.INSTRUMENTS.map(I => I.res)) - 1.22e-10) < 1e-20);
  const F = M.FILTER;
  ok('8f the sketched filter is 12 Å long with four sites, as Doyle et al. describe (bet: length = 12, sites = 4, equally spaced)', F.length === M.SRC.kcsa.filterA && F.sites.length === 4 && F.sites.every((y, k) => k === 0 || Math.abs(y - F.sites[k - 1] - F.rung) < 1e-12));
  ok('8g ions two sites apart in the sketch are within 25% of the 7.5 Å of the crystal structure (bet: < 25%)', rel(2 * F.rung, M.SRC.kcsa.ionGapA) < 0.25, (2 * F.rung) + ' Å against ' + M.SRC.kcsa.ionGapA);
  ok('8h every bond in the sketch joins two atoms between 1.0 and 1.8 Å apart (bet: all bonds)', F.bonds.every(([a, b]) => { const d = Math.hypot(F.atoms[a].x - F.atoms[b].x, F.atoms[a].y - F.atoms[b].y); return d > 1.0 && d < 1.8; }), F.bonds.map(([a, b]) => Math.hypot(F.atoms[a].x - F.atoms[b].x, F.atoms[a].y - F.atoms[b].y)).reduce((m, d) => [Math.min(m[0], d), Math.max(m[1], d)], [9, 0]).map(d => d.toFixed(2)).join('–') + ' Å');
}

