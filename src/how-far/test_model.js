// Model checks for how-far-does-it-go.html. The bets are in BETS.md and were written before the model was first run.
// Each line repeats its bet. Run: node src/how-far/test_model.js > src/how-far/test_output.txt
const fs = require('fs'), path = require('path');
const file = path.join(__dirname, '..', '..', 'how-far-does-it-go.html'), html = fs.readFileSync(file, 'utf8');
const code = html.match(/<script id="model">([\s\S]*?)<\/script>/)[1];
const mod = { exports: {} }; new Function('module', code)(mod); const E = mod.exports, K = E.K;
let fail = 0, n = 0; const ok = (name, cond, detail = '') => { n++; console.log((cond ? 'PASS ' : 'FAIL ') + name + (detail !== '' ? '  ' + detail : '')); if (!cond) fail++; };
const note = (name, detail) => console.log('NOTE ' + name + '  ' + detail), rel = (a, b) => Math.abs(a - b) / Math.abs(b), between = (x, a, b) => x >= a && x <= b, e = (x, d = 4) => x.toExponential(d);
console.log('Model checks, how-far-does-it-go.html, run ' + new Date().toISOString().slice(0, 10));

console.log('\n── Physics ──');
{ const P = E.planck; ok('P1 Planck length, mass, time, temperature from ħ, G, c agree with CODATA 2022 (bet: each within 2e-5)', rel(P.l, K.lPTab) < 2e-5 && rel(P.m, K.mPTab) < 2e-5 && rel(P.t, K.tPTab) < 2e-5 && rel(P.T, K.TPTab) < 2e-5, [e(P.l, 6), e(P.m, 6), e(P.t, 6), e(P.T, 6)].join('  ') + '  worst ' + e(Math.max(rel(P.l, K.lPTab), rel(P.m, K.mPTab), rel(P.t, K.tPTab), rel(P.T, K.TPTab)), 1)); }
{ const c = E.crossing; ok('P2a Compton = Schwarzschild, found by bisection, is at sqrt(hc/2G) (bet: within 1e-9)', rel(c.m, c.analytic) < 1e-9, e(c.m, 9) + ' kg');
  ok('P2b that is sqrt(π) Planck masses (bet: within 1e-9)', rel(c.overPlanck, Math.sqrt(Math.PI)) < 1e-9, c.overPlanck.toFixed(10));
  ok('P2c and the common length is 2 sqrt(π) Planck lengths (bet: within 1e-9)', rel(c.lengthOverPlanck, 2 * Math.sqrt(Math.PI)) < 1e-9 && rel(E.schwarzschild(c.m), E.compton(c.m)) < 1e-9, c.lengthOverPlanck.toFixed(10)); }
ok('P3 Bohr radius from ε0, ħ, m_e, e (bet: within 1e-8 of 5.29177210544e-11 m)', rel(E.bohr(), K.a0Tab) < 1e-8, e(E.bohr(), 11) + '  rel ' + e(rel(E.bohr(), K.a0Tab), 1));
ok('P4a R(8A) = 2 R(A) (bet: within 1e-12)', rel(E.nuclearRadius(216), 2 * E.nuclearRadius(27)) < 1e-12);
ok('P4b R(³⁹K) with r0 = 1.2 fm is 4.0 to 4.2 fm; R(¹⁹⁷Au) is 6.9 to 7.1 fm', between(E.nuclearRadius(39) * 1e15, 4.0, 4.2) && between(E.nuclearRadius(197) * 1e15, 6.9, 7.1), (E.nuclearRadius(39) * 1e15).toFixed(3) + ' fm, ' + (E.nuclearRadius(197) * 1e15).toFixed(3) + ' fm');
ok('P5a a 5 MeV alpha stops 44 to 47 fm from the centre of a gold nucleus', between(E.closestApproach(79, 5) * 1e15, 44, 47), (E.closestApproach(79, 5) * 1e15).toFixed(2) + ' fm');
ok('P5b Rutherford cross-section at 60° is 16 times that at 180° (bet: within 1e-9)', rel(E.rutherford(79, 5, Math.PI / 3) / E.rutherford(79, 5, Math.PI), 16) < 1e-9);
ok('P6a hc/E at 13.6 TeV is 9.0 to 9.2e-20 m', between(E.LHC.lambda, 9.0e-20, 9.2e-20), e(E.LHC.lambda));
ok('P6b ħc/Λ at 35.8 TeV is 5.4 to 5.6e-21 m', between(E.LIMITS.seenFloor, 5.4e-21, 5.6e-21), e(E.LIMITS.seenFloor));
ok('P7a bending radius of a 7 TeV proton at 8.33 T (bet: within 0.5% of 2,803.95 m)', rel(E.bendRadius(7 * K.TeV), 2803.95) < 0.005, E.bendRadius(7 * K.TeV).toFixed(1) + ' m');
{ const r = E.ringFor(E.planck.E); ok('P7b a ring to reach the Planck energy is 1,000 to 5,000 light-years around', between(r.Cly, 1000, 5000), Math.round(r.Cly) + ' ly, ' + e(r.CoverLHC, 2) + ' LHCs'); }
{ const C = E.COSMO, P = E.PLANCK18, zs = P.zStar;
  ok('P8a age of the universe (bet: 13.797 ± 0.03 billion years)', Math.abs(C.age / K.Gyr - 13.797) < 0.03, (C.age / K.Gyr).toFixed(4));
  ok('P8b particle horizon is 45.5 to 47.0 billion light-years', between(C.horizon / K.Gly, 45.5, 47.0), (C.horizon / K.Gly).toFixed(3) + ' Gly = ' + (C.horizon / K.Mpc / 1000).toFixed(3) + ' Gpc = ' + e(C.horizon) + ' m');
  ok('P8c matter–radiation equality (bet: z = 3402 ± 50)', Math.abs(C.zEq - 3402) < 50, C.zEq.toFixed(1));
  ok('P8d comoving distance to z = 1089.92 is 13.7 to 14.0 Gpc', between(C.chi(zs) / K.Mpc / 1000, 13.7, 14.0), (C.chi(zs) / K.Mpc / 1000).toFixed(4));
  ok('P8e sound horizon within 1% of 144.43 Mpc and 100θ within 1% of 1.0411', rel(E.LAST.rs / K.Mpc, 144.43) < 0.01 && rel(E.LAST.theta * 100, 1.0411) < 0.01, (E.LAST.rs / K.Mpc).toFixed(2) + ' Mpc, 100θ = ' + (E.LAST.theta * 100).toFixed(5));
  let worst = 0; for (const z of [0.5, 3, 14.44, zs]) worst = Math.max(worst, rel(C.lookbackOfChi(C.chi(z)), C.lookback(z)), rel(C.zOfChi(C.chi(z)), z)); ok('P8f lookback time by two routes, and redshift recovered from distance, at z = 0.5, 3, 14.44, 1089.92 (bet: within 1e-4)', worst < 1e-4, 'worst ' + e(worst, 1));
  ok('P8g with H0 = 73.04 the age is 12.4 to 13.0 billion years', between(E.COSMO_SH0ES.age / K.Gyr, 12.4, 13.0), (E.COSMO_SH0ES.age / K.Gyr).toFixed(3) + ', horizon ' + (E.COSMO_SH0ES.horizon / K.Gly).toFixed(2) + ' Gly');
  /* an independent check of the integrator: for matter alone the age is 2/(3 H0) exactly */
  /* P8h as first written used matter alone, where the integrand jumps at a = 0 and Simpson's rule loses 8e-5; that was a fault in the check, kept in
     test_output_v1.txt. Replaced by matter plus radiation with no dark energy, which has closed forms: t0 = 2[(Ωm − 2Ωr) + 2Ωr^1.5] / (3 H0 Ωm²), χ = 2c(1 − sqrt(Ωr)) / (H0 Ωm). */
  const t = E.Cosmo({ H0: 70 }), Or = t.Or, mr = E.Cosmo({ H0: 70, Om: 1 - Or }), tA = 2 * ((mr.P.Om - 2 * Or) + 2 * Math.pow(Or, 1.5)) / (3 * mr.H0 * mr.P.Om * mr.P.Om), xA = 2 * K.c * (1 - Math.sqrt(Or)) / (mr.H0 * mr.P.Om);
  ok('P8h (added; revised after the first run) matter plus radiation, no dark energy: age and horizon against their closed forms (bet: within 1e-6)', Math.abs(mr.OL) < 1e-12 && rel(mr.age, tA) < 1e-6 && rel(mr.horizon, xA) < 1e-6, e(rel(mr.age, tA), 1) + ', ' + e(rel(mr.horizon, xA), 1)); }
ok('P9a hydrogen is half ionised (Saha) between z = 1300 and 1450, at 3,500 to 4,000 K', between(E.RECOMB.zHalf, 1300, 1450) && between(E.RECOMB.THalf, 3500, 4000), 'z = ' + E.RECOMB.zHalf.toFixed(1) + ', T = ' + E.RECOMB.THalf.toFixed(0) + ' K');
ok('P9b hydrogen binding energy from the constants (bet: 13.606 ± 0.001 eV)', Math.abs(E.RYD_EV - 13.606) < 0.001, E.RYD_EV.toFixed(5));
{ const lt = E.lightTime, v1 = E.voyager('v1');
  ok('P10a light to the Moon 1.28 ± 0.01 s; to the Sun 499.0 ± 0.1 s; to Neptune 4.17 ± 0.01 h', Math.abs(lt(K.moonA) - 1.28) < 0.01 && Math.abs(lt(K.au) - 499.0) < 0.1 && Math.abs(lt(E.PLANETS[7].a * K.au) / 3600 - 4.17) < 0.01, lt(K.moonA).toFixed(4) + ' s, ' + lt(K.au).toFixed(3) + ' s, ' + (lt(E.PLANETS[7].a * K.au) / 3600).toFixed(4) + ' h');
  ok('P10b sunlight reaches Voyager 1 in 23.8 to 24.0 h on 25 Nov 2026', between(v1.lightHours, 23.8, 24.0), v1.lightHours.toFixed(3) + ' h at ' + v1.au.toFixed(3) + ' AU, ' + v1.kms.toFixed(2) + ' km/s');
  ok('P10c light from Proxima Centauri (bet: 4.2465 ± 0.0005 yr)', Math.abs(E.STARS[0].ly - 4.2465) < 0.0005, E.STARS[0].ly.toFixed(5)); }
ok('P11 shadow of a still black hole of 4.30e6 Suns at 8,277 pc within 2.3 µas of the 51.8 µas ring', Math.abs(E.SGRA.shadowMuas - 51.8) <= 2.3, E.SGRA.shadowMuas.toFixed(2) + ' µas');
ok('P12 MoM-z14 (z = 14.44): the universe was 270 to 300 million years old', between(E.JWST[0].ageMyr, 270, 300), E.JWST[0].ageMyr.toFixed(1) + ' Myr; lookback ' + E.JWST[0].lookbackGyr.toFixed(3) + ' Gyr; now ' + E.JWST[0].chiGly.toFixed(2) + ' Gly away');
ok('P13 inflation: H < 5e13 GeV ± 10%, so c/H > 3 to 5e-30 m', rel(E.INFL.HGeV, 5e13) < 0.1 && between(E.INFL.hubbleLength, 3e-30, 5e-30), e(E.INFL.HGeV, 3) + ' GeV, ' + e(E.INFL.hubbleLength, 3) + ' m');
{ const sky = E.cmbSky(200, 2018); let worst = 0, where = 0; for (let l0 = 20; l0 + 20 <= 200; l0 += 20) { let a = 0, b = 0; for (let l = l0; l < l0 + 20; l++) { a += sky.measured(l); b += E.cmbCl(l); } if (rel(a, b) > worst) { worst = rel(a, b); where = l0; } }
  ok('P14a spectrum of one sky, in bands of 20 from ℓ = 20 to 200 (bet: within 20% of the input in every band)', worst < 0.2, 'worst band starts at ℓ = ' + where + ': ' + (100 * worst).toFixed(1) + '%');
  /* the map itself: variance over a grid of rings, weighted by sin θ, against Σ(2ℓ+1)C_ℓ/4π */
  const nth = 120, nph = 240, row = new Float64Array(nph); let s = 0, w = 0; for (let j = 0; j < nth; j++) { const th = (j + 0.5) / nth * Math.PI; sky.ring(th, nph, row); for (let i = 0; i < nph; i++) { s += row[i] * row[i] * Math.sin(th); w += Math.sin(th); } }
  ok('P14b variance of the synthesised map against Σ(2ℓ+1)C_ℓ/4π (bet: within 15%)', rel(s / w, sky.variance()) < 0.15, 'map ' + (s / w).toFixed(0) + ' µK², sum ' + sky.variance().toFixed(0) + ' µK², rms ' + Math.sqrt(s / w).toFixed(1) + ' µK');
  ok('P14c (added) the stand-in spectrum peaks between ℓ = 200 and 240', (() => { let b = 2; for (let l = 2; l < 1000; l++) if (E.cmbDl(l) > E.cmbDl(b)) b = l; return between(b, 200, 240) ? b : 0; })() > 0); }
ok('P15a baryons in the observable universe: 1e79 to 1e81', between(E.COSMO.baryons, 1e79, 1e81), e(E.COSMO.baryons, 2));
ok('P15b photon density (bet: 410.7 ± 0.5 per cm³)', Math.abs(E.COSMO.photonDensity / 1e6 - 410.7) < 0.5, (E.COSMO.photonDensity / 1e6).toFixed(2));

console.log('\n── The zoom ──');
{ let worst = 0, mono = true, prev = Infinity; for (let k = 0; k <= 10000; k++) { const z = k / 10000, w = E.fovOf(z); worst = Math.max(worst, Math.abs(E.zoomOf(w) - z)); if (!(w < prev)) mono = false; prev = w; }
  ok('Z1 zoomOf(fovOf(z)) = z for 10,001 values (bet: max error < 1e-12), and the frame narrows monotonically', worst < 1e-12 && mono, e(worst, 2));
  ok('Z2 range 61 to 63 powers of ten; horizon diameter ÷ Planck length 61.5 to 62.0', between(E.DECADES, 61, 63) && between(E.SPAN.decades, 61.5, 62.0), E.DECADES.toFixed(3) + ', ' + E.SPAN.decades.toFixed(3) + ' (' + e(E.SPAN.ratio, 3) + ')'); }
{ const L = E.LEVELS; let inside = true, tile = true, at = true; L.forEach((l, k) => { if (!(l.top > l.fov && l.fov > l.bot)) inside = false; if (k && l.top !== L[k - 1].bot) tile = false; if (E.levelAt(l.fov) !== l.id) at = false; const m = E.levelMix(l.fov); if (m.b !== null) at = false; });
  ok('Z3 28 waypoints: each inside its band, bands tile the range, levelAt(waypoint) is that waypoint with no cross-fade in progress', L.length === 28 && inside && tile && at && L[0].top === E.W0 && L[27].bot === E.W1, L.length + ' waypoints'); }
{ let bad = 0, worst = ''; for (let k = 0; k < 2000; k++) { const m = Math.exp(E.LN1 + (E.LN0 - E.LN1) * k / 1999), t = E.fmtLen(m), v = parseFloat(t.replace(/,/g, '')), sb = E.scaleBar(m), f = sb.m / m; /* BET CHANGED after the second run (see RESULTS.md): kilometres are allowed to run to 1,000,000, because that is how such distances are written (the Moon is 384,400 km away). Every other unit keeps the limit of 10,000. */
    const lim = / km$/.test(t) && !/million/.test(t) ? 1000000 : 10000; if (!(v >= 0.1 && v < lim) || !(f <= 0.25 && f >= 0.04)) { bad++; worst = e(m, 2) + ' → ' + t + ' / bar ' + sb.label + ' (' + f.toFixed(3) + ')'; } }
  ok('Z4 2,000 log-spaced lengths: mantissa from 0.1 to under 10,000 (kilometres: under 1,000,000; bet changed, see RESULTS.md); scale bar 0.04 to 0.25 of the frame', bad === 0, bad ? bad + ' bad, e.g. ' + worst : ''); }
{ const B = E.HFD.LEVELS; let same = true; for (let i = 0; i < 7; i++) { const l = E.LV(i); if (l.fov !== B[i].fov || (i < 6 && l.bot !== B[i].bot) || (i > 0 && l.top !== B[i - 1].bot)) same = false; }
  ok('Z5 the brain\'s seven levels keep their frame widths and the boundaries between them', same); }

console.log('\n── CRR (definitions from crr-explained.html) ──');
{ const R = E.CRR;
  ok('C1 Z2 diameter: Fisher–Rao length from p = 0 to 1 by numerical integration is π (bet: within 1e-4), so Ω = 1/π; an inner stretch, 0.1 to 0.7, agrees with the closed form', rel(R.A.numeric, Math.PI) < 1e-4 && rel(R.A.omega, 1 / Math.PI) < 1e-4 && rel(R.A.partial[0], R.A.partial[1]) < 1e-6, R.A.numeric.toFixed(7) + '  rel ' + e(rel(R.A.numeric, Math.PI), 2));
  ok('C2a recombination: C*Ω = 1/2 at the redshift where hydrogen is half ionised (bet: within 1e-6)', rel(R.B.zHalfTurn, R.B.zSahaHalf) < 1e-6, R.B.zHalfTurn.toFixed(3));
  ok('C2b the occasion does NOT reach the cut: with 2.4e-4 left ionised C*Ω stops at 0.985 to 0.995 (bet: not reached)', !R.B.reached && between(R.B.final, 0.985, 0.995), 'C*Ω = ' + R.B.final.toFixed(5) + '; by Saha at z* it is ' + R.B.atZStar.toFixed(4) + '; 0.1 at z = ' + R.B.z10.toFixed(0) + ', 0.9 at z = ' + R.B.z90.toFixed(0));
  note('C2c', 'verdict fixed in advance: INCONCLUSIVE. The path is the Saha equation relabelled; nothing is predicted.');
  ok('C3 waypoint spacing: the spread of the 27 steps in ln(width) is NOT the candidate 0.159 (bet: it lies between 0.4 and 1.2)', between(R.C.cv, 0.4, 1.2), 'spread ' + R.C.cv.toFixed(3) + ' (mean step ' + R.C.mean.toFixed(2) + ', s.d. ' + R.C.sd.toFixed(2) + '); candidate ' + R.C.target.toFixed(3) + ' → CRR check FAILS, as bet');
  ok('C4a Ω_m (Planck) is within 1σ of 1/π; DESI DR2 is more than 2σ away', R.D.sigmaPlanck < 1 && R.D.sigmaDesi > 2, R.D.sigmaPlanck.toFixed(2) + 'σ, ' + R.D.sigmaDesi.toFixed(2) + 'σ');
  ok('C4b Ω_m(z) = 1/π only at |z| < 0.05; at z = 1 it is far from it', Math.abs(R.D.zEqual) < 0.05 && Math.abs(R.D.OmAt1 - 1 / Math.PI) > 0.3, 'z = ' + R.D.zEqual.toFixed(4) + '; Ω_m(1) = ' + R.D.OmAt1.toFixed(3));
  note('C4c', 'verdict fixed in advance: a coincidence of the present epoch, INCONCLUSIVE; not a claim the programme makes.');
  ok('C5 the crossing is sqrt(π) m_P with h and 1/sqrt(2) m_P with ħ: the π is a convention (bet: both within 1e-9)', rel(R.D.crossOverPlanck, Math.sqrt(Math.PI)) < 1e-9 && rel(R.D.reducedOverPlanck, Math.SQRT1_2) < 1e-9, R.D.crossOverPlanck.toFixed(6) + ', ' + R.D.reducedOverPlanck.toFixed(6) + ' → not a CRR signature');
  ok('C6a dark energy with DESI DR2 w0, wa: density peaks where w = −1 (bet: redshifts agree within 1e-6)', Math.abs(R.E.zCross - R.E.zPeak) < 1e-6, R.E.zCross.toFixed(6) + ' and ' + R.E.zPeak.toFixed(6));
  ok('C6b the same holds for an unrelated model, w = −1.3 + 0.9a² (bet: within 1e-3), so it is an identity', Math.abs(R.E.otherCross - R.E.otherPeak) < 1e-3, R.E.otherCross.toFixed(5) + ' and ' + R.E.otherPeak.toFixed(5));
  ok('C6c direction: w > −1 today and w < −1 in the past (the site\'s F2 test is not triggered)', R.E.wToday > -1 && R.E.wPast < -1, 'w(now) = ' + R.E.wToday + ', w(a = 0.3) = ' + R.E.wPast.toFixed(3));
  ok('C7 crossovers of cosmic history: three events, two intervals (bet: too few to test)', R.F.n === 2, 'ln a = ' + R.F.events.map(v => v.toFixed(2)).join(', ')); }

console.log('\n── The brain model, unchanged (its own checks, sections 3 to 8) ──');
{ const M = E.HFD, r = M.rng(2027), inc = fs.readFileSync(path.join(__dirname, 'test_brain_checks.inc.js'), 'utf8'); new Function('M', 'ok', 'note', 'rel', 'r', inc)(M, ok, note, rel, r); }

console.log('\n── Words on the page against the model ──');
{ const has = s => html.includes(s);
  ok('W1 headline figures in the prose equal the model\'s: 14.44, 13.797, 0.84075, 13.6 TeV, 51.8, 121.6 AU, 4.2465, 46 billion', has('14.44') && E.JWST[0].z === 14.44 && has('13.797') && E.PLANCK18.ageGyr === 13.797 && has('0.84075') && K.rpCharge === 0.84075e-15 && has('13.6 TeV') && E.LHC.EcmTeV === 13.6 && has('51.8') && E.SGRA.ringMuas === 51.8 && has('121.6 AU') && E.VOY.heliopause.v1 === 121.6 && has('4.2465') && has('46 billion') && Math.round(E.COSMO.horizon / K.Gly) === 46);
  ok('W2 "5.5 × 10⁻²¹ m" for the first line, "1.616 × 10⁻³⁵ m" for the Planck length and "9 × 10⁻²⁰ m" for the LHC are what the model computes', has('5.5 × 10⁻²¹ m') && (E.LIMITS.seenFloor * 1e21).toFixed(1) === '5.5' && has('1.616 × 10⁻³⁵ m') && (E.planck.l * 1e35).toFixed(3) === '1.616' && has('9 × 10⁻²⁰ m') && Math.round(E.LHC.lambda * 1e20) === 9);
  ok('W3 "35 powers of ten above" and "27 below" for a person are right to the nearest whole number', Math.round(Math.log10(E.HUMAN / E.planck.l)) === 35 && Math.round(Math.log10(2 * E.COSMO.horizon / E.HUMAN)) === 27, Math.log10(E.HUMAN / E.planck.l).toFixed(2) + ', ' + Math.log10(2 * E.COSMO.horizon / E.HUMAN).toFixed(2));
  ok('W4 "about 15 powers of ten" of desert below the line', Math.round(E.SEEN.below) === 15, E.SEEN.below.toFixed(2));
  ok('W5 the words of the page do not mention machine assistance and do not call CRR established', !/\b(language model|artificial intelligence|chatbot|AI)\b/.test(html.replace(/<script[\s\S]*?<\/script>/g, '')) && has('It is not established physics') && has('none of the scientists cited on this page proposed it or endorses it'));
  ok('W6 every imagined waypoint is marked imagined in the model, and the two frames beyond the lines are labelled', E.LV(11).kind === 'imagined' && E.LV(12).kind === 'imagined' && E.zone(E.LV(12).fov) === 'below' && E.zone(E.LV(-15).fov) === 'beyond' && E.zone(E.LV(-14).fov) === 'seen'); }
console.log('\n' + (n - fail) + ' of ' + n + ' checks passed' + (fail ? ', ' + fail + ' FAILED' : ''));
process.exit(fail ? 1 : 0);
