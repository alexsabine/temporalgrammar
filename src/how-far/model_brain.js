/* MODEL for "How far down does it go?". Pure calculation: no drawing, no DOM. Runs under Node.
   Units, unless a name says otherwise: lengths in metres for the zoom; micrometres, megohms, nanosiemens, picofarads,
   milliseconds, millivolts and picoamps for the electrical models (nS·mV = pA; pF·mV/ms = pA).
   Every constant below is either a measured figure with its source, or is marked "setting". */
const HFD = (() => {
  'use strict';
  const rng = seed => { let a = seed >>> 0; return () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; };
  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
  const smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };

  /* ── physical constants (SI 2019 exact values; electron mass CODATA 2018) ── */
  const PHYS = { R: 8.314462618, F: 96485.33212, e: 1.602176634e-19, h: 6.62607015e-34, me: 9.1093837015e-31, c: 299792458 };

  /* ── measured figures, each from the paper named ── */
  const SRC = {
    brain: { neurons: 86.1e9, neuronsSd: 8.1e9, nonNeurons: 84.6e9, cortexShare: 0.19,       /* Azevedo et al. 2009, abstract */
             cerebellumNeurons: 69e9, cortexNeurons: 16e9, massG: 1508 },                     /* Herculano-Houzel 2009, table 2 (figures of Azevedo et al.) */
    h01: { cells: 57180, neurons: 16087, glia: 32315, synapses: 150e6, volMm3: 1.05, bytes: 1.4e15, rawBytes: 1.8e15,
           voxNm: [4, 4, 33], sections: 5019, thickMm: 0.170, vesselMm: 230 },                /* Shapson-Coe et al. 2024 */
    microns: { cells: 200000, synapses: 524e6, imaged: 75000, dimsMm: [1.3, 0.87, 0.82] },    /* MICrONS Consortium 2025 */
    fly: { neurons: 139255, synapses: 5e7, types: 8453 },                                     /* Dorkenwald et al. 2024; Schlegel et al. 2024 */
    bouton: { proteins: 300000, exoCopies: 20000, endoCopies: 150 },                          /* Wilhelm et al. 2014, abstract */
    cryo: { apoA: 1.22, gabaA: 1.7, yipA: 1.25, particlesApo: 363126, particlesGaba: 371693, kV: 300 },   /* Nakane et al. 2020; Yip et al. 2020 */
    kcsa: { xrayA: 3.2, xray2A: 2.0, filterA: 12, ionGapA: 7.5 },                             /* Doyle et al. 1998; Zhou et al. 2001 */
    mri: { vivoM: 1e-3, exVivoM: 100e-6, exVivoHours: 100 }                                   /* 1 mm: routine structural scans (Fischl & Dale 2000 used such scans); Edlow et al. 2019 */
  };
  /* settings: plausible round values chosen by this page, not measurements */
  const SET = {
    densityGml: 1.04,              /* density of brain tissue, to turn 1,508 g into a volume */
    cortexMm: 2.5,                 /* thickness of the cortex in the drawing */
    Rm: 20000, Ri: 150, Cm: 1,     /* Ω·cm², Ω·cm, µF/cm²: passive membrane, typical textbook values (Koch 1999) */
    Vrest: -70, Esyn: 0,           /* mV */
    gAMPA: 0.5, gNMDA: 1.2,        /* nS at the peak, per synapse */
    tAMPA: [0.2, 2], tNMDA: [2, 60],   /* rise and decay, ms */
    Mg: 1,                         /* mM, for the Jahr & Stevens (1990) block */
    cleftNm: 20, vesicleNm: 40,    /* widths used in the drawing; of the order reported by Savtchenko & Rusakov 2007 and Takamori et al. 2006 */
    Dglu: 0.33,                    /* µm²/ms: glutamate in the cleft, about a third of free solution (Nielsen et al. 2004) */
    gamma: 20,                     /* pS: conductance of one open channel */
    twoPhoton: { lambdaNm: 920, NA: 1.0 }, confocal: { lambdaNm: 520, NA: 1.4 }
  };

  /* ════ 1. The zoom: one number z in [0,1] ↔ the width of the frame in metres, on a logarithmic scale ════ */
  const W0 = 0.26, W1 = 1.6e-9;
  const fovOf = z => W0 * Math.pow(W1 / W0, clamp(z, 0, 1));
  const zoomOf = w => clamp(Math.log(w / W0) / Math.log(W1 / W0), 0, 1);
  const DECADES = Math.log10(W0 / W1);
  const UNITS = [[1e-2, 'cm'], [1e-3, 'mm'], [1e-6, 'µm'], [1e-9, 'nm'], [1e-10, 'Å']];
  const unitOf = m => { for (const u of UNITS) if (m >= u[0] * 0.9999) return u; return UNITS[4]; };
  const sig = (x, n = 3) => { if (x === 0) return '0'; const d = Math.max(0, n - 1 - Math.floor(Math.log10(Math.abs(x)) + 1e-9)); return (+x.toFixed(Math.min(d, 6))).toString(); };
  const fmtLen = (m, n = 3) => { const u = unitOf(m); return sig(m / u[0], n) + ' ' + u[1]; };
  /* scale bar: the longest of 1, 2, 5 × 10^k that is no more than `frac` of the frame */
  const scaleBar = (fov, frac = 0.22) => { const t = fov * frac, k = Math.floor(Math.log10(t) + 1e-9), b = Math.pow(10, k), r = t / b, m = r >= 5 ? 5 : r >= 2 ? 2 : 1; const len = m * b; return { m: len, label: fmtLen(len, 1) }; };

  const LEVELS = [
    { id: 'brain',   name: 'The whole brain',            fov: 0.25,    top: 0.26,   bot: 9e-3,   tool: 'Magnetic resonance imaging (MRI)',          res: 1e-3 },
    { id: 'cortex',  name: 'A patch of cortex',          fov: 4.2e-3,  top: 9e-3,   bot: 1.4e-3,  tool: 'Fluorescence (two-photon) microscopy',      res: null },
    { id: 'neuron',  name: 'One neuron',                 fov: 1.05e-3, top: 1.4e-3,  bot: 6e-5,   tool: 'Fluorescence microscopy of one filled cell', res: null },
    { id: 'wiring',  name: 'Every cell and every wire',  fov: 1.6e-5,  top: 6e-5,   bot: 3.2e-6, tool: 'Serial-section electron microscopy',        res: 4e-9 },
    { id: 'synapse', name: 'One synapse',                fov: 1.5e-6,  top: 3.2e-6, bot: 1.6e-7, tool: 'Electron microscopy of thin sections',      res: 4e-9 },
    { id: 'machine', name: 'The molecular machines',     fov: 5.2e-8,  top: 1.6e-7, bot: 7e-9,   tool: 'Cryo-electron tomography, protein counting and patch clamp', res: null },
    { id: 'atoms',   name: 'Atoms',                      fov: 2.4e-9,  top: 7e-9,   bot: W1,     tool: 'Single-particle cryo-electron microscopy',  res: 1.22e-10 }
  ];
  const TEXTBOOK_LINE = LEVELS[1].bot;      /* the frame width at which the textbook's account stops: below the layers of one column */
  const FADE = 0.2;                         /* width of each cross-fade, in powers of ten */
  /* which pictures are on screen at a frame width, and how strongly: {a, b, t}; t is the weight of b (the finer one) */
  const levelMix = fov => { const lf = Math.log10(fov);
    for (let i = 0; i < LEVELS.length - 1; i++) { const B = Math.log10(LEVELS[i].bot); if (lf > B + FADE / 2) return { a: i, b: -1, t: 0 }; if (lf >= B - FADE / 2) return { a: i, b: i + 1, t: smooth((B + FADE / 2 - lf) / FADE) }; }
    return { a: LEVELS.length - 1, b: -1, t: 0 }; };
  const levelAt = fov => { const m = levelMix(fov); return m.b >= 0 && m.t > 0.5 ? m.b : m.a; };
  /* the finest detail resolved anywhere on the descent, against the widest frame */
  const SPAN = { ratio: W0 / (SRC.cryo.apoA * 1e-10), decades: Math.log10(W0 / (SRC.cryo.apoA * 1e-10)) };

  /* ════ 2. Counting ════ */
  const perMm3 = { neurons: SRC.h01.neurons / SRC.h01.volMm3, cells: SRC.h01.cells / SRC.h01.volMm3, synapses: SRC.h01.synapses / SRC.h01.volMm3 };
  /* a cube of human temporal cortex whose face is the frame (densities of H01) */
  const cubeCounts = fov => { const v = Math.pow(fov * 1e3, 3); return { mm3: v, neurons: perMm3.neurons * v, cells: perMm3.cells * v, synapses: perMm3.synapses * v }; };
  const brainVolMm3 = SRC.brain.massG / SET.densityGml * 1000;
  const DATA = (() => { const h = SRC.h01, voxMm3 = h.voxNm[0] * h.voxNm[1] * h.voxNm[2] * 1e-18, voxels = h.volMm3 / voxMm3;
    return { voxels, bytesPerMm3: h.bytes / h.volMm3, brainVolMm3, brainBytes: h.bytes / h.volMm3 * brainVolMm3, brainVoxels: brainVolMm3 / voxMm3,
             cerebellumShare: SRC.brain.cerebellumNeurons / SRC.brain.neurons, synPerNeuron: h.synapses / h.neurons }; })();
  const fmtBytes = b => { const U = ['bytes', 'kB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']; let i = 0; while (b >= 1000 && i < U.length - 1) { b /= 1000; i++; } return sig(b, 2) + ' ' + U[i]; };
  const fmtCount = x => x >= 1e9 ? sig(x / 1e9, 3) + ' billion' : x >= 1e6 ? sig(x / 1e6, 3) + ' million' : x >= 1e4 ? Math.round(x).toLocaleString('en-GB') : x >= 100 ? String(Math.round(x)) : sig(x, 2);

  /* ════ 3. Instruments and their limits ════ */
  /* Abbe (1873): two points closer than d = λ / (2 NA) cannot be told apart with light of wavelength λ */
  const abbe = (lambda, NA) => lambda / (2 * NA);
  /* de Broglie wavelength of an electron accelerated through V volts, with the relativistic correction:
     λ = h / sqrt(2 m e V (1 + eV / (2 m c²))) */
  const electronLambda = V => PHYS.h / Math.sqrt(2 * PHYS.me * PHYS.e * V * (1 + PHYS.e * V / (2 * PHYS.me * PHYS.c * PHYS.c)));
  const INSTRUMENTS = [
    { name: 'MRI, living person',            res: SRC.mri.vivoM,   fov: 0.26,    alive: true,  src: 'routine 1 mm scans' },
    { name: 'MRI, donated brain, 100 h scan', res: SRC.mri.exVivoM, fov: 0.26,    alive: false, src: 'Edlow et al. 2019' },
    { name: 'Two-photon microscope, living animal', res: abbe(SET.twoPhoton.lambdaNm * 1e-9, SET.twoPhoton.NA), fov: SRC.microns.dimsMm[0] * 1e-3, alive: true, src: 'Denk et al. 1990; MICrONS 2025' },
    { name: 'Serial-section electron microscopy', res: SRC.h01.voxNm[0] * 1e-9, fov: 3e-3, alive: false, src: 'Shapson-Coe et al. 2024' },
    { name: 'X-ray crystallography',         res: SRC.kcsa.xray2A * 1e-10, fov: 1e-8, alive: false, src: 'Zhou et al. 2001' },
    { name: 'Single-particle cryo-EM',       res: SRC.cryo.apoA * 1e-10, fov: 1.2e-8, alive: false, src: 'Nakane et al. 2020' }
  ];

  /* ════ 4. A dendritic tree as an electrical cable (Rall 1959; Koch 1999) ════
     Passive cable:  λ² ∂²V/∂x² = τ ∂V/∂t + V,   λ = sqrt((d/4)(Rm/Ri)),   τ = Rm Cm.
     The tree is cut into compartments 10 µm long. Compartment i has a membrane conductance g_i = π d L / Rm, a capacitance
     c_i = π d L Cm, and is joined to its parent by an axial conductance made of the two half-compartments in series.
     One time step of the backward Euler method solves (C/dt + G) v' = C/dt v + I, which for a tree is a single upward
     and a single downward sweep. v is measured from rest. */
  const lambdaUm = (dUm, Rm = SET.Rm, Ri = SET.Ri) => Math.sqrt((dUm * 1e-4 / 4) * (Rm / Ri)) * 1e4;
  const tauMs = (Rm = SET.Rm, Cm = SET.Cm) => Rm * Cm * 1e-3;                                  /* Ω·cm² × µF/cm² = µs·10³ */
  const raMOhm = (L, d, Ri = SET.Ri) => Ri * L * 4e-2 / (Math.PI * d * d);                     /* axial resistance of a length L, MΩ */
  const rInfMOhm = (d, Rm = SET.Rm, Ri = SET.Ri) => raMOhm(lambdaUm(d, Rm, Ri), d, Ri);        /* input resistance of a semi-infinite cable = r_a λ */
  /* steady state of a cable of length ell sealed at both ends, current I (pA) injected at xi, voltage (mV) read at x:
     V(x) = I R∞ cosh(X<) cosh(L − X>) / sinh(L), with X = x/λ and L = ell/λ */
  const sealedCable = (I, x, xi, ell, d, Rm = SET.Rm, Ri = SET.Ri) => { const lam = lambdaUm(d, Rm, Ri), L = ell / lam, a = Math.min(x, xi) / lam, b = Math.max(x, xi) / lam; return I * 1e-3 * rInfMOhm(d, Rm, Ri) * Math.cosh(a) * Math.cosh(L - b) / Math.sinh(L); };

  /* branches: [{from: branch index or -1 for the soma, at: 0..1 along the parent, len, d, x, y, ang, bend}] */
  function buildTree(branches, opt = {}) {
    const dx = opt.dx || 10, Rm = opt.Rm || SET.Rm, Ri = opt.Ri || SET.Ri, Cm = opt.Cm || SET.Cm, r = rng(opt.seed || 7);
    const T = { parent: [], L: [], d: [], x0: [], y0: [], x1: [], y1: [], dist: [], branch: [], first: [], last: [], Rm, Ri, Cm };
    const add = (p, L, d, x0, y0, x1, y1, b) => { const i = T.parent.length; T.parent.push(p); T.L.push(L); T.d.push(d); T.x0.push(x0); T.y0.push(y0); T.x1.push(x1); T.y1.push(y1); T.branch.push(b);
      T.dist.push(p < 0 ? 0 : T.dist[p] + T.L[p] / 2 + L / 2); return i; };
    if (opt.soma !== false) { const s = opt.soma || { x: 0, y: 0, d: 20 }; add(-1, s.d, s.d, s.x, s.y - s.d / 2, s.x, s.y + s.d / 2, -1); }
    branches.forEach((B, bi) => { let p, x, y;
      if (B.from < 0 || B.from === undefined) { p = T.parent.length ? 0 : -1; x = B.x; y = B.y; }
      else { const f = T.first[B.from], l = T.last[B.from]; p = f + Math.round((l - f) * (B.at === undefined ? 1 : B.at)); x = (T.x0[p] + T.x1[p]) / 2; y = (T.y0[p] + T.y1[p]) / 2; if (B.at === undefined || B.at === 1) { x = T.x1[p]; y = T.y1[p]; } }
      const n = Math.max(1, Math.round(B.len / dx)), L = B.len / n; let a = B.ang || 0; T.first[bi] = T.parent.length;
      for (let k = 0; k < n; k++) { if (k * L >= (B.straight || 0)) a += (B.bend || 0) + (B.wig || 0) * (r() * 2 - 1); const x1 = x + Math.cos(a) * L, y1 = y + Math.sin(a) * L; p = add(p, L, B.d * (B.taper ? 1 - (1 - B.taper) * k / n : 1), x, y, x1, y1, bi); x = x1; y = y1; }
      T.last[bi] = T.parent.length - 1; });
    const n = T.parent.length; T.n = n; T.gm = new Float64Array(n); T.cap = new Float64Array(n); T.ga = new Float64Array(n); T.gsum = new Float64Array(n);
    for (let i = 0; i < n; i++) { const A = Math.PI * T.d[i] * T.L[i]; T.gm[i] = 10 / Rm * A; T.cap[i] = 1e-2 * Cm * A; }
    for (let i = 0; i < n; i++) { const p = T.parent[i]; if (p < 0) continue; T.ga[i] = 1000 / (raMOhm(T.L[i] / 2, T.d[i], Ri) + raMOhm(T.L[p] / 2, T.d[p], Ri)); T.gsum[i] += T.ga[i]; T.gsum[p] += T.ga[i]; }
    T.length = T.L.reduce((s, v, i) => s + (T.branch[i] >= 0 ? v : 0), 0); T.area = T.L.reduce((s, v, i) => s + Math.PI * T.d[i] * v, 0);
    return T;
  }
  /* solve A v = rhs for a tree: A has `diag` on the diagonal and −ga[i] between i and parent[i]; children come after parents */
  function hines(T, diag, rhs) { const n = T.n, p = T.parent, ga = T.ga;
    for (let i = n - 1; i > 0; i--) { const f = ga[i] / diag[i]; diag[p[i]] -= f * ga[i]; rhs[p[i]] += f * rhs[i]; }
    rhs[0] /= diag[0]; for (let i = 1; i < n; i++) rhs[i] = (rhs[i] + ga[i] * rhs[p[i]]) / diag[i]; return rhs; }
  /* steady-state voltages (mV from rest) for currents inj = {compartment: pA} */
  function steady(T, inj) { const d = new Float64Array(T.n), r = new Float64Array(T.n); for (let i = 0; i < T.n; i++) d[i] = T.gm[i] + T.gsum[i]; for (const k in inj) r[k] = inj[k]; return hines(T, d, r); }
  const dexp = (s, tr, td) => { if (s <= 0) return 0; const tp = tr * td / (td - tr) * Math.log(td / tr), pk = Math.exp(-tp / td) - Math.exp(-tp / tr); return (Math.exp(-s / td) - Math.exp(-s / tr)) / pk; };
  /* Jahr & Stevens (1990): the fraction of NMDA-receptor conductance not blocked by magnesium at membrane potential V (mV) */
  const mgBlock = (V, Mg = SET.Mg) => 1 / (1 + Mg / 3.57 * Math.exp(-0.062 * V));
  /* a live dendrite: fire(site, n) opens n synapses at a compartment; step(dt) advances every compartment */
  function Dend(T, opt = {}) {
    const P = Object.assign({ nmda: true, gAMPA: SET.gAMPA, gNMDA: SET.gNMDA, Vrest: SET.Vrest, Esyn: SET.Esyn }, opt);
    const v = new Float64Array(T.n), diag = new Float64Array(T.n), rhs = new Float64Array(T.n), S = { T, P, v, t: 0, syn: [], inj: {} };
    S.fire = (site, n) => { S.syn.push({ i: site, n, t0: S.t }); };
    S.reset = () => { v.fill(0); S.syn.length = 0; S.t = 0; };
    S.step = dt => { for (let i = 0; i < T.n; i++) { const c = T.cap[i] / dt; diag[i] = c + T.gm[i] + T.gsum[i]; rhs[i] = c * v[i]; }
      for (const k in S.inj) rhs[k] += S.inj[k];
      for (let j = S.syn.length - 1; j >= 0; j--) { const y = S.syn[j], s = S.t + dt - y.t0; if (s > 400) { S.syn.splice(j, 1); continue; }
        const g = y.n * (P.gAMPA * dexp(s, SET.tAMPA[0], SET.tAMPA[1]) + (P.nmda ? P.gNMDA * dexp(s, SET.tNMDA[0], SET.tNMDA[1]) * mgBlock(P.Vrest + v[y.i]) : 0));
        diag[y.i] += g; rhs[y.i] += g * (P.Esyn - P.Vrest); }
      hines(T, diag, rhs); v.set(rhs); S.t += dt; };
    return S;
  }
  /* one volley of n synapses at a site: the peaks at the site and at the soma (mV above rest) */
  function volley(T, site, n, opt = {}) { const D = Dend(T, opt), dt = opt.dt || 0.1; D.fire(site, n); let ps = 0, pd = 0; for (let t = 0; t < (opt.T || 120); t += dt) { D.step(dt); if (D.v[0] > ps) ps = D.v[0]; if (D.v[site] > pd) pd = D.v[site]; } return { soma: ps, site: pd }; }
  const ioCurve = (T, site, nMax, opt) => { const out = []; for (let n = 0; n <= nMax; n++) out.push(n ? volley(T, site, n, opt) : { soma: 0, site: 0 }); return out; };
  /* the layer III pyramidal cell of the picture. Origin: a point just above an oblique dendrite; y grows downwards; micrometres. */
  const OBL = 10;                               /* index of the oblique dendrite that the later pictures follow */
  const NEURON_SPEC = (() => { const D = Math.PI / 180, B = [];
    B.push({ from: -1, x: -150, y: 218, len: 430, d: 3, taper: 0.7, ang: -90 * D, wig: 0.014 });                          /* 0 apical trunk */
    [[-124, -0.012], [-88, 0.006], [-54, 0.016]].forEach(([a, bend], k) => { const i = B.length; B.push({ from: 0, len: 70 + 14 * k, d: 1.6, ang: a * D, bend, wig: 0.06 });   /* 1–9 the tuft */
      B.push({ from: i, len: 120 - 10 * k, d: 1.0, taper: 0.6, ang: (a - 24) * D, bend: -0.01, wig: 0.07 }); B.push({ from: i, len: 105 + 12 * k, d: 1.0, taper: 0.6, ang: (a + 22) * D, bend: 0.012, wig: 0.07 }); });
    B.push({ from: 0, at: 0.5035, len: 290, d: 1.0, ang: 0, straight: 165, bend: -0.028, wig: 0.04 });                      /* 10 the oblique that passes under the origin */
    B.push({ from: 10, at: 0.72, len: 90, d: 0.8, ang: 38 * D, bend: 0.01, wig: 0.07 });
    B.push({ from: 0, at: 0.3, len: 170, d: 1.0, ang: -158 * D, bend: 0.022, wig: 0.06 }); B.push({ from: 12, at: 0.45, len: 90, d: 0.8, ang: 160 * D, wig: 0.07 });
    B.push({ from: 0, at: 0.68, len: 150, d: 0.9, ang: -28 * D, bend: -0.022, wig: 0.06 }); B.push({ from: 14, at: 0.5, len: 80, d: 0.8, ang: 4 * D, wig: 0.07 });
    B.push({ from: 0, at: 0.86, len: 120, d: 0.9, ang: -152 * D, bend: 0.016, wig: 0.06 });
    B.push({ from: 0, at: 0.16, len: 140, d: 1.0, ang: -14 * D, bend: 0.012, wig: 0.06 });
    [160, 122, 84, 48, 14].forEach((a, k) => { const i = B.length; B.push({ from: -1, x: -150 + 9 * Math.cos(a * D), y: 228 + 9 * Math.sin(a * D), len: 40, d: 1.8, ang: a * D, wig: 0.06 });
      const j = B.length; B.push({ from: i, len: 85 + 9 * k, d: 1.0, taper: 0.7, ang: (a + 26) * D, wig: 0.07 }); B.push({ from: i, len: 100 - 6 * k, d: 1.0, taper: 0.7, ang: (a - 24) * D, wig: 0.07 });
      B.push({ from: j + (k & 1), at: 0.45, len: 70, d: 0.8, taper: 0.7, ang: (a + (k & 1 ? -52 : 54)) * D, wig: 0.08 }); });
    return B; })();
  const NEURON = buildTree(NEURON_SPEC, { soma: { x: -150, y: 228, d: 20 }, seed: 11 });
  /* the path a visitor can walk: soma → trunk → oblique; compartments in order of distance */
  const PATH = (() => { const T = NEURON, out = [0]; let join = T.parent[T.first[OBL]]; for (let i = T.first[0]; i <= join; i++) out.push(i); for (let i = T.first[OBL]; i <= T.last[OBL]; i++) out.push(i); return out; })();
  /* the compartment of the oblique nearest the origin: the spine of the later pictures sits on it */
  const HOME_SITE = (() => { const T = NEURON; let b = T.first[OBL], bd = 1e9; for (let i = T.first[OBL]; i <= T.last[OBL]; i++) { const d = Math.hypot((T.x0[i] + T.x1[i]) / 2, (T.y0[i] + T.y1[i]) / 2); if (d < bd) { bd = d; b = i; } } return b; })();

  /* ════ 5. Hodgkin & Huxley (1952), in the modern convention (rest near −65 mV), squid axon at 6.3 °C ════
     C dV/dt = I − gNa m³h (V − ENa) − gK n⁴ (V − EK) − gL (V − EL);  dx/dt = αx(V)(1 − x) − βx(V) x  for x = m, h, n. */
  const lin = (x, y) => Math.abs(x / y) < 1e-6 ? y * (1 - x / y / 2) : x / (1 - Math.exp(-x / y));    /* x / (1 − e^(−x/y)), safe at x = 0 */
  const HH = { C: 1, gNa: 120, gK: 36, gL: 0.3, ENa: 50, EK: -77, EL: -54.387,
    an: V => 0.01 * lin(V + 55, 10), bn: V => 0.125 * Math.exp(-(V + 65) / 80),
    am: V => 0.1 * lin(V + 40, 10), bm: V => 4 * Math.exp(-(V + 65) / 18),
    ah: V => 0.07 * Math.exp(-(V + 65) / 20), bh: V => 1 / (1 + Math.exp(-(V + 35) / 10)) };
  HH.nInf = V => HH.an(V) / (HH.an(V) + HH.bn(V)); HH.tauN = V => 1 / (HH.an(V) + HH.bn(V));
  HH.mInf = V => HH.am(V) / (HH.am(V) + HH.bm(V)); HH.hInf = V => HH.ah(V) / (HH.ah(V) + HH.bh(V));
  /* a patch of membrane with current I (µA/cm²) switched on at t = 5 ms; returns the trace and the spikes */
  function hhRun(I, T = 100, dt = 0.01) { let V = -65, m = HH.mInf(V), h = HH.hInf(V), n = HH.nInf(V), spikes = 0, peak = -1e9, up = false; const tr = [], every = Math.round(0.1 / dt);
    for (let k = 0; k * dt < T; k++) { const t = k * dt, Iin = t >= 5 ? I : 0;
      const dV = (Iin - HH.gNa * m * m * m * h * (V - HH.ENa) - HH.gK * n * n * n * n * (V - HH.EK) - HH.gL * (V - HH.EL)) / HH.C;
      m += dt * (HH.am(V) * (1 - m) - HH.bm(V) * m); h += dt * (HH.ah(V) * (1 - h) - HH.bh(V) * h); n += dt * (HH.an(V) * (1 - n) - HH.bn(V) * n); V += dt * dV;
      if (V > 0 && !up) { up = true; spikes++; } if (V < -20) up = false; if (V > peak) peak = V; if (k % every === 0) tr.push(V); }
    return { spikes, peak, end: V, trace: tr }; }

  /* ════ 6. Ions and single channels ════ */
  /* Nernst: the membrane potential at which diffusion and the electric field balance for one ion. mV; T in kelvin. */
  const nernst = (z, out, inn, T = 310.15) => PHYS.R * T / (z * PHYS.F) * Math.log(out / inn) * 1000;
  const MAMMAL = [['K⁺', 1, 5, 140], ['Na⁺', 1, 145, 15], ['Cl⁻', -1, 110, 10], ['Ca²⁺', 2, 2, 1e-4]];   /* mM outside, inside: round textbook concentrations (settings) */
  /* one open channel passes i = γ (V − E): pS × mV = fA, so /1000 for pA */
  const unitCurrent = (V, E = HH.EK, gamma = SET.gamma) => gamma * (V - E) / 1000;
  const ionsPerMs = iPA => Math.abs(iPA) * 1e-12 / PHYS.e / 1000;
  /* N potassium channels, each with four independent gates that open at rate αn(V) and close at βn(V): Hodgkin and Huxley's n⁴,
     read as a Markov chain. A channel conducts when all four gates are open, so p = n∞⁴ at rest;
     the number open has mean N p and variance N p (1 − p) (Sigworth 1980). */
  function Channels(N, V, seed = 1) { const r = rng(seed), k = new Uint8Array(N), S = { N, V, k, t: 0, open: 0 };
    const draw = () => { const ni = HH.nInf(S.V); S.open = 0; for (let c = 0; c < N; c++) { let g = 0; for (let j = 0; j < 4; j++) if (r() < ni) g++; k[c] = g; if (g === 4) S.open++; } };
    S.settle = draw; draw();
    S.setV = V2 => { S.V = V2; };
    S.step = dt => { const a = HH.an(S.V) * dt, b = HH.bn(S.V) * dt; let o = 0; for (let c = 0; c < N; c++) { const g = k[c], u = r(), up = (4 - g) * a; if (u < up) k[c] = g + 1; else if (u < up + g * b) k[c] = g - 1; if (k[c] === 4) o++; } S.open = o; S.t += dt; return o; };
    S.resize = N2 => Channels(N2, S.V, seed + N2); return S; }
  const channelStats = (N, V, gamma = SET.gamma) => { const n = HH.nInf(V), p = n * n * n * n, i = unitCurrent(V, HH.EK, gamma); return { n, p, i, mean: N * p * i, sd: Math.sqrt(N * p * (1 - p)) * Math.abs(i), cv: N * p > 0 ? Math.sqrt((1 - p) / (N * p)) : 0, openMs: 1 / (4 * HH.bn(V)) }; };

  /* ════ 7. Quantal release (del Castillo & Katz 1954) ════
     n vesicles ready, each released with probability p, each adding q (with a spread cv·q) to the response:
     mean = n p q;  variance = n p q² (1 − p + cv²);  P(nothing) = (1 − p)ⁿ. */
  const binom = (n, k) => { let c = 1; for (let i = 0; i < k; i++) c = c * (n - i) / (i + 1); return c; };
  const quantal = (n, p, q = 1, cv = 0) => ({ mean: n * p * q, variance: n * p * q * q * (1 - p + cv * cv), fail: Math.pow(1 - p, n), pmf: Array.from({ length: n + 1 }, (_, k) => binom(n, k) * Math.pow(p, k) * Math.pow(1 - p, n - k)) });
  const gauss = r => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
  const release = (r, n, p, q = 1, cv = 0) => { const which = []; let amp = 0; for (let i = 0; i < n; i++) if (r() < p) { which.push(i); amp += Math.max(0, q * (1 + cv * gauss(r))); } return { k: which.length, which, amp }; };
  /* Einstein's relation for diffusion in one dimension: a molecule wanders x in a time t = x² / 2D. x in nm, D in µm²/ms → microseconds */
  const diffusionUs = (xNm, D = SET.Dglu) => Math.pow(xNm * 1e-3, 2) / (2 * D) * 1000;

  /* ════ 8. The selectivity filter of a potassium channel, as a flat sketch in ångströms ════
     Five rings of oxygen atoms 3 Å apart make four sites in a filter 12 Å long (Doyle et al. 1998); ions sit two at a time
     with a water between them (7.5 Å apart in the crystal; two sites, 6 Å, in this sketch) and move in single file
     (Morais-Cabral et al. 2001). Positions are schematic. */
  const FILTER = (() => { const atoms = [], bonds = [], rung = 3.0, ys = [-6, -3, 0, 3, 6];
    for (const s of [-1, 1]) { let prev = -1; ys.forEach((y, k) => { const o = atoms.push({ el: 'O', x: s * 2.45, y }) - 1, c = atoms.push({ el: 'C', x: s * 3.62, y: y + 0.25 }) - 1, ca = atoms.push({ el: 'C', x: s * 4.75, y: y + 1.2 }) - 1, n = atoms.push({ el: 'N', x: s * 4.3, y: y + 2.3 }) - 1;
        bonds.push([o, c], [c, ca], [ca, n]); if (prev >= 0) bonds.push([prev, c]); prev = n;
        if (k % 2 === 0) { const b = atoms.push({ el: 'C', x: s * 5.95, y: y + 1.35 }) - 1; bonds.push([ca, b]); if (k === 2) { const g = atoms.push({ el: 'C', x: s * 6.8, y: y + 0.2 }) - 1, d = atoms.push({ el: 'O', x: s * 8.1, y: y + 0.45 }) - 1; bonds.push([b, g], [g, d]); } } }); }
    return { atoms, bonds, sites: [-4.5, -1.5, 1.5, 4.5], rung, length: ys[4] - ys[0] }; })();

  return { rng, clamp, smooth, PHYS, SRC, SET, W0, W1, DECADES, SPAN, fovOf, zoomOf, unitOf, sig, fmtLen, scaleBar, LEVELS, TEXTBOOK_LINE, FADE, levelMix, levelAt,
    perMm3, cubeCounts, DATA, fmtBytes, fmtCount, abbe, electronLambda, INSTRUMENTS,
    lambdaUm, tauMs, raMOhm, rInfMOhm, sealedCable, buildTree, hines, steady, dexp, mgBlock, Dend, volley, ioCurve, NEURON_SPEC, NEURON, OBL, PATH, HOME_SITE,
    HH, hhRun, nernst, MAMMAL, unitCurrent, ionsPerMs, Channels, channelStats, binom, quantal, release, gauss, diffusionUs, FILTER };
})();
