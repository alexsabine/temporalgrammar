/* MODEL for "How Far Does It Go?". Pure calculation: no drawing, no DOM. Runs under Node.
   It extends the brain model above (HFD) to every scale from the Planck length to the particle horizon.
   SI units throughout unless a name says otherwise. Every constant is a published value with its source named,
   or is marked "setting". Nothing here is fitted to make a check pass. */
const EV = (() => {
  'use strict';
  const PI = Math.PI, clamp = (x, a, b) => Math.max(a, Math.min(b, x)), smooth = t => { t = clamp(t, 0, 1); return t * t * (3 - 2 * t); };
  const rng = HFD.rng;

  /* ════ 1. Constants ════ */
  const K = {
    /* exact by definition, SI 2019 */
    c: 299792458, h: 6.62607015e-34, e: 1.602176634e-19, kB: 1.380649e-23,
    /* CODATA 2022 recommended values (Mohr, Newell, Taylor & Tiesinga 2025), read from physics.nist.gov on 25 November 2026 */
    G: 6.67430e-11, me: 9.1093837139e-31, mp: 1.67262192595e-27, alpha: 7.2973525643e-3, eps0: 8.8541878188e-12,
    a0Tab: 5.29177210544e-11, rpCharge: 0.84075e-15, rpErr: 0.00064e-15,
    lPTab: 1.616255e-35, mPTab: 2.176434e-8, tPTab: 5.391247e-44, TPTab: 1.416784e32,
    /* IAU: astronomical unit (2012 Resolution B2), Julian year, nominal solar mass parameter and Earth radius (2015 Resolution B3) */
    au: 149597870700, yr: 365.25 * 86400, GMsun: 1.3271244e20, REarthEq: 6378.1e3,
    REarth: 6371.0088e3,           /* mean radius of the GRS80 ellipsoid (IUGG) */
    moonA: 384399e3,               /* mean Earth–Moon distance, semi-major axis (lunar laser ranging; NASA fact sheet) */
    RSun: 6.957e8,                 /* IAU 2015 nominal solar radius */
    Tcmb: 2.7255                   /* K, Fixsen 2009 */
  };
  K.hbar = K.h / (2 * PI);
  K.ly = K.c * K.yr;               /* IAU light-year: 9.4607304725808e15 m */
  K.pc = K.au * 648000 / PI;       /* IAU 2015 Resolution B2 */
  K.Msun = K.GMsun / K.G;
  K.eV = K.e; K.GeV = 1e9 * K.e; K.TeV = 1e12 * K.e;
  K.sigmaSB = 2 * Math.pow(PI, 5) * Math.pow(K.kB, 4) / (15 * Math.pow(K.h, 3) * K.c * K.c);
  K.Mpc = 1e6 * K.pc; K.Gly = 1e9 * K.ly; K.Gyr = 1e9 * K.yr;

  /* ════ 2. Planck units, and the two lengths every mass has ════ */
  const planck = (() => { const l = Math.sqrt(K.hbar * K.G / Math.pow(K.c, 3)), m = Math.sqrt(K.hbar * K.c / K.G), t = Math.sqrt(K.hbar * K.G / Math.pow(K.c, 5));
    return { l, m, t, T: m * K.c * K.c / K.kB, E: m * K.c * K.c, EGeV: m * K.c * K.c / K.GeV }; })();
  /* Compton wavelength: below this size a particle of mass m cannot be pinned down without making more particles */
  const compton = m => K.h / (m * K.c);
  /* Schwarzschild radius: inside this size a mass m is a black hole */
  const schwarzschild = m => 2 * K.G * m / (K.c * K.c);
  /* where the two are equal: h/(mc) = 2Gm/c²  →  m = sqrt(hc/2G) = sqrt(π) m_P. Found numerically here, by bisection on ln m,
     so that the test can compare it with the formula. With the reduced wavelength ħ/(mc) the crossing is at m_P/sqrt(2). */
  const crossing = (() => { let a = Math.log(1e-12), b = Math.log(1e-3); for (let i = 0; i < 200; i++) { const m = (a + b) / 2; if (compton(Math.exp(m)) > schwarzschild(Math.exp(m))) a = m; else b = m; }
    const m = Math.exp((a + b) / 2); return { m, overPlanck: m / planck.m, length: compton(m), lengthOverPlanck: compton(m) / planck.l, analytic: Math.sqrt(K.h * K.c / (2 * K.G)), reduced: planck.m / Math.SQRT2 }; })();

  /* ════ 3. Atoms and nuclei ════ */
  const bohr = () => 4 * PI * K.eps0 * K.hbar * K.hbar / (K.me * K.e * K.e);
  const R0_FM = 1.2;                               /* setting: r0 in R = r0 A^(1/3); fits to electron scattering give 1.07 to 1.25 fm depending on what is called the edge (Hofstadter 1956) */
  const nuclearRadius = (A, r0 = R0_FM) => r0 * 1e-15 * Math.cbrt(A);
  /* Rutherford (1911): an alpha particle (charge 2e) of kinetic energy E aimed straight at a nucleus of charge Ze stops at
     d = 2Ze² / (4π ε0 E); at an angle θ the cross-section is dσ/dΩ = (d/4)² / sin⁴(θ/2) */
  const closestApproach = (Z, EMeV, z = 2) => z * Z * K.e * K.e / (4 * PI * K.eps0 * EMeV * 1e6 * K.e);
  const rutherford = (Z, EMeV, thetaRad, z = 2) => Math.pow(closestApproach(Z, EMeV, z) / 4, 2) / Math.pow(Math.sin(thetaRad / 2), 4);
  /* the potassium ion of the channel picture: 19 protons, 20 neutrons (³⁹K, 93% of natural potassium), 18 electrons.
     Slater's rules (1930) give each shell an effective nuclear charge; the most likely radius of shell n is n² a0 / Zeff. */
  const KION = (() => { const Z = 19, A = 39, a0 = bohr(), sh = [{ n: 1, N: 2, Zeff: Z - 0.30 }, { n: 2, N: 8, Zeff: Z - 0.35 * 7 - 0.85 * 2 }, { n: 3, N: 8, Zeff: Z - 0.35 * 7 - 0.85 * 8 - 2 }];
    sh.forEach(s => { s.rPeak = s.n * s.n * a0 / s.Zeff; s.scale = s.n * a0 / (2 * s.Zeff); });   /* radial density ∝ r^(2n) e^(−r/scale): a gamma distribution of shape 2n+1 */
    return { Z, A, N: A - Z, electrons: 18, shells: sh, R: nuclearRadius(A), ionicRadius: 1.38e-10 /* Shannon 1976, six-fold coordination */ }; })();

  /* ════ 4. What it takes to see small things ════ */
  const lambdaOfE = E => K.h * K.c / E;                       /* wavelength of a probe of energy E (joules), for E ≫ mc² */
  const energyOfLambda = l => K.h * K.c / l;
  const LHC = { EcmTeV: 13.6, BT: 8.33, circumference: 26659, rho: 2803.95 };   /* Run 3 energy; design dipole field, ring and bending radius (LHC Design Report, CERN-2004-003) */
  LHC.lambda = lambdaOfE(LHC.EcmTeV * K.TeV); LHC.reduced = LHC.lambda / (2 * PI); LHC.fill = 2 * PI * LHC.rho / LHC.circumference;
  /* a particle of momentum p and charge e in a field B turns on a circle of radius ρ = p / (eB); for E ≫ mc², p = E/c */
  const bendRadius = (EbeamJ, B = LHC.BT) => EbeamJ / (K.e * K.c * B);
  /* the ring a proton collider would need to reach a collision energy E: two beams of E/2, LHC magnets, LHC share of the ring bent.
     Synchrotron radiation, which grows as E⁴/ρ, is ignored: the true answer is worse. */
  const ringFor = EcmJ => { const rho = bendRadius(EcmJ / 2), C = 2 * PI * rho / LHC.fill; return { rho, C, CoverLHC: C / LHC.circumference, Cly: C / K.ly }; };
  /* upper limits on the size of quarks and electrons. None has ever shown a size. */
  const LIMITS = {
    quarkRadius: 0.43e-18,          /* ZEUS 2016: effective quark radius below 0.43 × 10⁻¹⁶ cm, 95% confidence */
    quarkCI: [12.8, 17.5],          /* TeV. CMS 2018, dijet angles: quark contact interactions excluded up to these scales (destructive, constructive) */
    leptonQuarkCI: 35.8             /* TeV. ATLAS 2020, electron and muon pairs: two-quark two-lepton contact interactions excluded up to this scale */
  };
  LIMITS.lengthOf = TeV => K.hbar * K.c / (TeV * K.TeV);                  /* the length ħc/Λ that belongs to an energy scale Λ */
  LIMITS.seenFloor = LIMITS.lengthOf(LIMITS.leptonQuarkCI);              /* the finest length at which anything has been tested: the first hard line */
  const ACCEL = [                                                          /* collision energy, GeV; fixed-target ones as the energy in the centre of mass */
    { name: 'Rutherford, radium alphas (1911)', short: 'Rutherford 1911', E: 5e-3, kind: 'done' }, { name: 'Hofstadter, electrons (1955)', short: 'Hofstadter 1955', E: 0.19, kind: 'done' },
    { name: 'SLAC, deep inelastic (1968)', short: 'SLAC 1968', E: 6, kind: 'done' }, { name: 'HERA, electron–proton (1992–2007)', short: 'HERA', E: 318, kind: 'done' },
    { name: 'LHC, proton–proton (now)', short: 'LHC', E: 13600, kind: 'done' }, { name: 'FCC-hh (proposed)', short: 'FCC (proposed)', E: 1e5, kind: 'proposed' } ];
  /* inflation, if it happened: the tensor-to-scalar ratio r fixes the expansion rate then, H = π M_Pl sqrt(A_s r / 2) with M_Pl the
     reduced Planck mass. An upper limit on r is an upper limit on H, so a lower limit on the Hubble length c/H at that time. */
  const INFL = (() => { const As = Math.exp(3.044) * 1e-10, rMax = 0.036, MplGeV = planck.EGeV / Math.sqrt(8 * PI), HGeV = PI * MplGeV * Math.sqrt(As * rMax / 2);
    return { As, rMax, ns: 0.9649, HGeV, hubbleLength: K.hbar * K.c / (HGeV * K.GeV), VquarterGeV: Math.pow(3 * HGeV * HGeV * MplGeV * MplGeV, 0.25) }; })();

  /* ════ 5. The expanding universe: the Friedmann equation, integrated ════
     Flat ΛCDM with the parameters of Planck 2018 (Planck Collaboration VI 2020, table 2, TT,TE,EE+lowE+lensing).
     H(a)² = H0² [Ωr a⁻⁴ + Ωm a⁻³ + Ω_DE(a)], a = 1/(1+z).
     Radiation: photons at 2.7255 K plus three kinds of neutrino treated as massless (N_eff = 3.046), a small simplification.
     Every time and distance below is a numerical integral of this one equation (Simpson's rule in s = sqrt(a), which keeps the
     integrand smooth through the radiation era). */
  const PLANCK18 = { H0: 67.36, H0err: 0.54, Om: 0.3153, OmErr: 0.0073, Obh2: 0.02237, Och2: 0.1200, ns: 0.9649, sigma8: 0.8111, Neff: 3.046, Yp: 0.2454,
    ageGyr: 13.797, ageErr: 0.023, zStar: 1089.92, rStarMpc: 144.43, theta100: 1.04110, zEq: 3402, zEqErr: 26, tau: 0.0544 };
  const SH0ES = { H0: 73.04, err: 1.04 };                         /* Riess et al. 2022 */
  const DESI2 = { Om: 0.2975, OmErr: 0.0086, w0: -0.752, w0Err: 0.057, wa: -0.86, waErr: [0.20, 0.23], sigmaRange: [2.8, 4.2], objects: 14e6 };   /* DESI Collaboration 2025 (DR2 Results II); w0, wa from DESI + CMB + DES-SN5YR */
  function Cosmo(opt = {}) {
    const P = Object.assign({ H0: PLANCK18.H0, Om: PLANCK18.Om, Obh2: PLANCK18.Obh2, Tcmb: K.Tcmb, Neff: PLANCK18.Neff, w0: -1, wa: 0 }, opt);
    const H0 = P.H0 * 1000 / K.Mpc, h = P.H0 / 100, rhoCrit = 3 * H0 * H0 / (8 * PI * K.G);
    const rhoGamma = 4 * K.sigmaSB * Math.pow(P.Tcmb, 4) / Math.pow(K.c, 3), Og = rhoGamma / rhoCrit, Or = Og * (1 + 0.22710731766 * P.Neff), Ob = P.Obh2 / (h * h), OL = 1 - P.Om - Or;
    const de = a => P.w0 === -1 && P.wa === 0 ? 1 : Math.pow(a, -3 * (1 + P.w0 + P.wa)) * Math.exp(-3 * P.wa * (1 - a));     /* ρ_DE(a)/ρ_DE(1), for w(a) = w0 + wa(1 − a) */
    const E = a => Math.sqrt(Or / (a * a * a * a) + P.Om / (a * a * a) + OL * de(a));
    const simpson = (f, a, b, n = 2000) => { const hh = (b - a) / n; let s = f(a) + f(b); for (let i = 1; i < n; i++) s += f(a + i * hh) * (i & 1 ? 4 : 2); return s * hh / 3; };
    /* with a = s²:  dt = da/(aH) = 2 s³ ds / (H0 sqrt(Or + Om s² + OL de s⁸));   dχ = c da/(a²H) = 2c s ds / (H0 sqrt(…)) */
    const root = s => { const a = s * s; return Math.sqrt(Or + P.Om * a + OL * de(a) * a * a * a * a); };
    const tOfA = a => simpson(s => 2 * s * s * s / root(s), 0, Math.sqrt(a)) / H0;                     /* seconds since the beginning */
    const chiOfA = a => simpson(s => 2 * s / root(s), Math.sqrt(a), 1) * K.c / H0;                    /* comoving distance light has covered since a, metres */
    const age = tOfA(1), horizon = simpson(s => 2 * s / root(s), 0, 1, 4000) * K.c / H0;
    /* the sound horizon: how far a pressure wave in the photon–baryon fluid had travelled by a. c_s = c / sqrt(3(1 + R)), R = 3ρ_b/4ρ_γ */
    const soundHorizon = a1 => simpson(s => { const a = s * s; return 2 * s / root(s) / Math.sqrt(3 * (1 + 0.75 * Ob / Og * a)); }, 0, Math.sqrt(a1), 4000) * K.c / H0;
    const S = { P, H0, h, rhoCrit, Og, Or, Ob, OL, E, de, age, horizon, tOfA, chiOfA, soundHorizon };
    S.t = z => tOfA(1 / (1 + z)); S.lookback = z => age - S.t(z); S.chi = z => chiOfA(1 / (1 + z)); S.H = z => H0 * E(1 / (1 + z));
    S.zEq = P.Om / Or - 1;                                         /* matter and radiation equal */
    S.zLambda = OL > 0 ? Math.cbrt(OL / P.Om) - 1 : NaN;           /* matter and dark energy equal (constant dark energy) */
    S.zAccel = OL > 0 ? Math.cbrt(2 * OL / P.Om) - 1 : NaN;        /* expansion begins to speed up */
    S.OmAt = z => { const a = 1 / (1 + z), e = E(a); return P.Om / (a * a * a) / (e * e); };
    /* a table, so that the inverse (distance → redshift) is quick: ln(1+z) from 0 to ln(1e7) */
    let TAB = null; const table = () => { if (TAB) return TAB; const n = 600, u = new Float64Array(n + 1), chi = new Float64Array(n + 1), lb = new Float64Array(n + 1), um = Math.log(1e7);
      /* cumulative Simpson in u = ln(1+z): dχ = c dz/H = c e^u du / H;  dt = du / H */
      const fz = uu => { const z = Math.exp(uu) - 1; return 1 / (H0 * E(1 / (1 + z))); };
      for (let i = 0; i <= n; i++) u[i] = um * i / n;
      for (let i = 1; i <= n; i++) { const a = u[i - 1], b = u[i], m = (a + b) / 2, hh = (b - a) / 6; chi[i] = chi[i - 1] + K.c * hh * (Math.exp(a) * fz(a) + 4 * Math.exp(m) * fz(m) + Math.exp(b) * fz(b)); lb[i] = lb[i - 1] + hh * (fz(a) + 4 * fz(m) + fz(b)); }
      TAB = { n, u, chi, lb, um }; return TAB; };
    const interp = (xs, ys, x) => { let lo = 0, hi = xs.length - 1; if (x <= xs[0]) return ys[0]; if (x >= xs[hi]) return ys[hi]; while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xs[m] <= x) lo = m; else hi = m; } return ys[lo] + (ys[hi] - ys[lo]) * (x - xs[lo]) / (xs[hi] - xs[lo]); };
    S.zOfChi = chi => { const T = table(); return chi >= horizon ? Infinity : Math.exp(interp(T.chi, T.u, chi)) - 1; };
    S.lookbackOfChi = chi => { const T = table(); return chi >= horizon ? age : Math.min(age, interp(T.chi, T.lb, chi)); };
    S.chiOfLookback = t => { const T = table(); return t >= age ? horizon : interp(T.lb, T.chi, t); };
    /* counts in the observable universe today */
    S.volume = 4 / 3 * PI * Math.pow(horizon, 3); S.baryons = Ob * rhoCrit * S.volume / K.mp;
    S.photonDensity = 16 * PI * 1.2020569031595942 * Math.pow(K.kB * P.Tcmb / (K.h * K.c), 3);                  /* n_γ = (2ζ(3)/π²)(kT/ħc)³ */
    S.photons = S.photonDensity * S.volume;
    return S; }
  const COSMO = Cosmo(), COSMO_SH0ES = Cosmo({ H0: SH0ES.H0 });
  const LAST = (() => { const z = PLANCK18.zStar, chi = COSMO.chi(z), rs = COSMO.soundHorizon(1 / (1 + z)); return { z, chi, t: COSMO.t(z), rs, theta: rs / chi, T: K.Tcmb * (1 + z) }; })();

  /* recombination: the Saha equation for hydrogen alone,  x²/(1 − x) = (1/n_H) (m_e k T / 2πħ²)^(3/2) e^(−13.6 eV / kT),
     with x the share of hydrogen that is ionised. It gives the right epoch; the real process lags a little behind it (Peebles 1968). */
  const RYD_EV = K.me * Math.pow(K.e, 4) / (8 * K.eps0 * K.eps0 * K.h * K.h) / K.e;      /* 13.606 eV, from the constants above */
  const saha = z => { const T = K.Tcmb * (1 + z), nH = (1 - PLANCK18.Yp) * COSMO.Ob * COSMO.rhoCrit / K.mp * Math.pow(1 + z, 3);
    const S = Math.pow(K.me * K.kB * T / (2 * PI * K.hbar * K.hbar), 1.5) * Math.exp(-RYD_EV * K.e / (K.kB * T)) / nH; return S > 1e8 ? 1 - 1 / S : (-S + Math.sqrt(S * S + 4 * S)) / 2; };
  const solve = (f, a, b, n = 80) => { let fa = f(a); for (let i = 0; i < n; i++) { const m = (a + b) / 2, fm = f(m); if ((fa < 0) === (fm < 0)) { a = m; fa = fm; } else b = m; } return (a + b) / 2; };
  const RECOMB = (() => { const zHalf = solve(z => saha(z) - 0.5, 800, 2500); return { zHalf, THalf: K.Tcmb * (1 + zHalf), tHalf: COSMO.t(zHalf), xAtZStar: saha(PLANCK18.zStar), xResidual: 2.4e-4 /* setting: the share left ionised when recombination freezes out (order of magnitude; Peebles 1968; Seager, Sasselov & Scott 1999) */ }; })();

  /* ════ 6. Nearer things ════ */
  const lightTime = d => d / K.c;
  /* Voyager distances from the Sun: JPL Horizons, heliocentric range and range-rate at 2026-Nov-25 00:00 TDB, read on that day.
     Later dates are a straight-line extrapolation, good to a small fraction of an AU for some years. */
  const VOY = { epoch: Date.UTC(2026, 10, 25), v1: { r: 172.4533400537501, rate: 9.749921107556349e-3, rEarth: 173.2360296971671 }, v2: { r: 144.5635024245678, rate: 8.690217428157740e-3 },
    heliopause: { v1: 121.6, v1Date: '25 August 2012', v2: 119.0, v2Date: '5 November 2018' }, lightDayAU: 86400 * 299792458 / 149597870700 };
  const voyager = (which, ms = VOY.epoch) => { const v = VOY[which], days = (ms - VOY.epoch) / 86400000, r = v.r + v.rate * days; return { au: r, m: r * K.au, lightHours: r * K.au / K.c / 3600, kms: v.rate * K.au / 86400 / 1000, auPerYr: v.rate * 365.25 }; };
  const PLANETS = [['Mercury', 0.38710, 252.25, 149472.67], ['Venus', 0.72333, 181.98, 58517.82], ['Earth', 1.00000, 100.46, 35999.37], ['Mars', 1.52368, 355.43, 19140.30], ['Jupiter', 5.2026, 34.40, 3034.75], ['Saturn', 9.5549, 49.94, 1222.49], ['Uranus', 19.2184, 313.23, 428.48], ['Neptune', 30.1104, 304.88, 218.46]]
    .map(p => ({ name: p[0], a: p[1], L0: p[2], rate: p[3], periodYr: Math.pow(p[1], 1.5) }));       /* semi-major axis (AU), mean longitude at J2000 (°) and its rate (° per century): Standish's approximate elements, JPL. Kepler: T² = a³ */
  const planetAt = (p, ms) => { const T = (ms - Date.UTC(2000, 0, 1, 12)) / 86400000 / 36525, L = (p.L0 + p.rate * T) * PI / 180; return [p.a * Math.cos(L), -p.a * Math.sin(L)]; };   /* circular orbit at the mean longitude; y is down the screen */
  /* the nearest stars: right ascension, declination (degrees), parallax (milliarcseconds), a colour class.
     Gaia EDR3 values from the Gaia Catalogue of Nearby Stars (Gaia Collaboration, Smart et al. 2021), read from VizieR J/A+A/649/A6 on 25 November 2026,
     except the four marked *, which are too bright or too faint for Gaia and are quoted from memory of Hipparcos and later work. */
  const STARS = [['Proxima Centauri', 217.3923, -62.6761, 768.067, 'M'], ['α Centauri A and B*', 219.90, -60.83, 750.81, 'G'], ['Barnard’s Star', 269.4485, 4.7394, 546.976, 'M'], ['Luhman 16*', 162.31, -53.32, 501.6, 'L'],
    ['Wolf 359', 164.1032, 7.0027, 415.179, 'M'], ['Lalande 21185', 165.8310, 35.9487, 392.753, 'M'], ['Sirius', 101.2866, -16.7209, 374.490, 'A'], ['Luyten 726-8', 24.7717, -17.9477, 373.844, 'M'], ['Ross 154', 282.4588, -23.8371, 336.027, 'M'],
    ['Ross 248', 355.4800, 44.1704, 316.481, 'M'], ['ε Eridani', 53.2283, -9.4582, 310.577, 'K'], ['Lacaille 9352', 346.5039, -35.8472, 304.135, 'M'], ['Ross 128', 176.9377, 0.7991, 296.305, 'M'], ['EZ Aquarii*', 339.69, -15.30, 294.3, 'M'],
    ['61 Cygni', 316.7485, 38.7639, 285.995, 'K'], ['Procyon*', 114.83, 5.22, 284.56, 'F'], ['Struve 2398', 280.6831, 59.6384, 283.840, 'M'], ['Groombridge 34', 4.6132, 44.0248, 280.707, 'M'], ['DX Cancri', 127.4501, 26.7733, 279.250, 'M'],
    ['ε Indi', 330.8724, -56.7973, 274.843, 'K'], ['τ Ceti', 26.0091, -15.9337, 273.810, 'G'], ['GJ 1061', 54.0034, -44.5144, 272.162, 'M'], ['YZ Ceti', 18.1333, -16.9962, 269.057, 'M'], ['Luyten’s Star', 111.8546, 5.2094, 264.127, 'M'],
    ['Teegarden’s Star', 43.2696, 16.8644, 260.988, 'M'], ['Kapteyn’s Star', 77.9599, -45.0438, 254.199, 'M'], ['Lacaille 8760', 319.2950, -38.8725, 251.912, 'M']]
    .map(s => { const d = 1000 / s[3] * K.pc, ra = s[1] * PI / 180, de = s[2] * PI / 180; return { name: s[0], plx: s[3], cls: s[4], d, ly: d / K.ly, x: d * Math.cos(de) * Math.cos(ra) / K.ly, y: d * Math.cos(de) * Math.sin(ra) / K.ly, z: d * Math.sin(de) / K.ly }; });
  const starsWithin = ly => STARS.filter(s => s.ly <= ly).length;
  /* the black hole at the centre of the Milky Way. Its shadow, for a black hole that does not spin, is 2 sqrt(27) GM/c² across. */
  const SGRA = (() => { const M = 4.30e6 * K.Msun /* GRAVITY Collaboration 2022 */, D = 8277 * K.pc /* same paper: 8277 ± 9 ± 33 pc; quoted from memory */, rs = 2 * K.G * M / (K.c * K.c), shadow = Math.sqrt(27) * rs / D, muas = 180 / PI * 3600e6;
    return { M, D, Dly: D / K.ly, rs, shadowMuas: shadow * muas, ringMuas: 51.8, ringErr: 2.3 /* Event Horizon Telescope Collaboration 2022 */, lambdaMm: 1.3, baselineM: 1.07e7 /* setting: longest baseline, close to Earth's diameter */, resMuas: 1.3e-3 / 1.07e7 * muas }; })();
  const MILKYWAY = { starsLow: 1e11, starsHigh: 4e11, diameterLy: 1e5, sunR: SGRA.Dly };      /* star count and disc size: round figures in common use (Bland-Hawthorn & Gerhard 2016 give a stellar mass of 5 ± 1 × 10¹⁰ Suns) */
  const LANIAKEA = { diameterMpc: 160, massSun: 1e17 };                                         /* Tully et al. 2014 */
  const JWST = [{ name: 'MoM-z14', z: 14.44, err: 0.02, src: 'Naidu et al. 2025' }, { name: 'JADES-GS-z14-0', z: 14.32, err: [0.20, 0.08], src: 'Carniani et al. 2024' }, { name: 'JADES-GS-z14-1', z: 13.90, err: 0.17, src: 'Carniani et al. 2024' }, { name: 'PAN-z14-1', z: 13.53, err: 0.06, src: 'Donnan et al. 2026' }]
    .map(g => Object.assign(g, { ageMyr: COSMO.t(g.z) / K.yr / 1e6, lookbackGyr: COSMO.lookback(g.z) / K.Gyr, chiGly: COSMO.chi(g.z) / K.Gly }));

  /* ════ 7. The microwave background: a spectrum and a sky drawn from it ════
     D_ℓ = ℓ(ℓ+1)C_ℓ/2π in µK². This is a smooth stand-in shaped like the measured spectrum (a plateau, a first peak at ℓ ≈ 220,
     a second near 540, a third near 810), not the Planck data and not a Boltzmann-code calculation. */
  const cmbDl = l => l < 2 ? 0 : 1000 * Math.pow(l / 10, -0.05) + 4650 * Math.exp(-Math.pow((l - 220) / 78, 2) / 2) + 1450 * Math.exp(-Math.pow((l - 537) / 62, 2) / 2) + 1500 * Math.exp(-Math.pow((l - 812) / 70, 2) / 2) + 500 * Math.exp(-Math.pow((l - 400) / 220, 2) / 2);
  const cmbCl = l => l < 2 ? 0 : cmbDl(l) * 2 * PI / (l * (l + 1));
  /* a Gaussian sky: a_ℓm with variance C_ℓ. Stored for m ≥ 0 as real and imaginary parts; a_ℓ,−m follows from the sky being real. */
  function cmbSky(Lmax, seed = 2018) { const r = rng(seed), g = () => HFD.gauss(r), n = (Lmax + 1) * (Lmax + 2) / 2, re = new Float32Array(n), im = new Float32Array(n), idx = (l, m) => m * (2 * Lmax + 1 - m) / 2 + l;
    for (let l = 2; l <= Lmax; l++) { const s = Math.sqrt(cmbCl(l)); re[idx(l, 0)] = s * g(); for (let m = 1; m <= l; m++) { re[idx(l, m)] = s * g() * Math.SQRT1_2; im[idx(l, m)] = s * g() * Math.SQRT1_2; } }
    const S = { Lmax, re, im, idx };
    /* the spectrum of this one sky: Ĉ_ℓ = Σ_m |a_ℓm|² / (2ℓ+1). It scatters about C_ℓ by sqrt(2/(2ℓ+1)): cosmic variance. */
    S.measured = l => { let s = re[idx(l, 0)] * re[idx(l, 0)]; for (let m = 1; m <= l; m++) s += 2 * (re[idx(l, m)] * re[idx(l, m)] + im[idx(l, m)] * im[idx(l, m)]); return s / (2 * l + 1); };
    /* one ring of the sky at colatitude θ: T(φ_i), i = 0..nphi−1. Normalised associated Legendre functions by upward recurrence in ℓ. */
    const bre = new Float64Array(Lmax + 1), bim = new Float64Array(Lmax + 1); let ct = null, stb = null, nphi0 = 0;
    S.ring = (theta, nphi, out) => { const x = Math.cos(theta), sn = Math.sin(theta); let pmm = Math.sqrt(1 / (4 * PI));
      for (let m = 0; m <= Lmax; m++) { if (m > 0) pmm *= -Math.sqrt((2 * m + 1) / (2 * m)) * sn; let p0 = pmm, p1 = x * Math.sqrt(2 * m + 3) * pmm, sr = 0, si = 0;
        if (Math.abs(pmm) < 1e-30 && m > 20) { bre[m] = 0; bim[m] = 0; for (let k = m + 1; k <= Lmax; k++) { bre[k] = 0; bim[k] = 0; } break; }
        for (let l = m; l <= Lmax; l++) { let p; if (l === m) p = p0; else if (l === m + 1) p = p1; else { const a = Math.sqrt((4 * l * l - 1) / (l * l - m * m)), b = Math.sqrt(((l - 1) * (l - 1) - m * m) / (4 * (l - 1) * (l - 1) - 1)); p = a * (x * p1 - b * p0); p0 = p1; p1 = p; }
          const k = idx(l, m); sr += re[k] * p; si += im[k] * p; }
        bre[m] = sr; bim[m] = si; }
      if (nphi !== nphi0) { ct = new Float64Array(nphi); stb = new Float64Array(nphi); for (let i = 0; i < nphi; i++) { ct[i] = Math.cos(2 * PI * i / nphi); stb[i] = Math.sin(2 * PI * i / nphi); } nphi0 = nphi; }
      for (let i = 0; i < nphi; i++) { let s = bre[0]; for (let m = 1, k = i; m <= Lmax; m++, k += i) { const j = k % nphi; s += 2 * (bre[m] * ct[j] - bim[m] * stb[j]); } out[i] = s; } return out; };
    S.variance = () => { let v = 0; for (let l = 2; l <= Lmax; l++) v += (2 * l + 1) * cmbCl(l) / (4 * PI); return v; };
    return S; }

  /* ════ 8. The zoom: one number z in [0,1] ↔ the width of the frame, over all the scales, in logarithms ════ */
  const W0 = 4.2e27, W1 = 5 * planck.l, LN0 = Math.log(W0), LN1 = Math.log(W1), DECADES = (LN0 - LN1) / Math.LN10;
  const lnFovOf = z => LN0 + clamp(z, 0, 1) * (LN1 - LN0), fovOf = z => Math.exp(lnFovOf(z));
  const zoomOfLn = lnw => clamp((lnw - LN0) / (LN1 - LN0), 0, 1), zoomOf = w => zoomOfLn(Math.log(w));
  const SPAN = { ratio: 2 * COSMO.horizon / planck.l, decades: Math.log10(2 * COSMO.horizon / planck.l) };
  const HUMAN = 1.7;                               /* setting: the height of the person in the picture, metres */
  const SEEN = { floor: LIMITS.seenFloor, ceiling: 2 * COSMO.horizon };     /* the two hard lines, as frame widths */
  SEEN.decades = Math.log10(SEEN.ceiling / SEEN.floor); SEEN.below = Math.log10(SEEN.floor / planck.l);
  /* units for labels, largest first */
  const UNITS = [[K.Gly, 'billion ly'], [1e6 * K.ly, 'million ly'], [1e3 * K.ly, 'thousand ly'], [K.ly, 'ly'], [K.au, 'AU'], [1e9, 'million km'], [1e3, 'km'], [1, 'm'], [1e-2, 'cm'], [1e-3, 'mm'], [1e-6, 'µm'], [1e-9, 'nm'], [1e-10, 'Å'], [1e-12, 'pm'], [1e-15, 'fm'], [1e-18, 'am'], [1e-21, 'zm'], [1e-24, 'ym'], [1e-27, 'rm'], [1e-30, 'qm'], [planck.l, 'Planck lengths']];
  const unitOf = m => { if (m < 1e-31) return UNITS[UNITS.length - 1]; for (const u of UNITS) { const lo = u[1] === 'AU' ? 0.1 * K.au : u[1] === 'ly' ? 0.1 * K.ly : u[1] === 'qm' ? 1e-31 : u[0] * 0.9999; if (m >= lo) return u; } return UNITS[UNITS.length - 1]; };
  const sig = HFD.sig, group = x => Math.round(x).toLocaleString('en-GB');
  const fmtLen = (m, n = 3) => { const u = unitOf(m), v = m / u[0]; return (v >= 1000 ? group(+v.toPrecision(n)) : sig(v, n)) + ' ' + u[1]; };
  /* scale bar: the longest of 1, 2, 5 × 10^k, in the unit it will be labelled with, that is no more than `frac` of the frame */
  const scaleBar = (fov, frac = 0.22) => { const T = fov * frac; let u = UNITS[UNITS.length - 1]; for (const c of UNITS) if (T >= c[0] * (c[1] === 'qm' ? 0.1 : 1)) { u = c; break; }
    const t = T / u[0], k = Math.floor(Math.log10(t) + 1e-9), b = Math.pow(10, k), r = t / b, mm = r >= 5 ? 5 : r >= 2 ? 2 : 1, v = mm * b; return { m: v * u[0], label: (v >= 1000 ? group(v) : sig(v, 1)) + ' ' + u[1] }; };
  const sup = n => String(n).replace(/[-0-9]/g, c => c === '-' ? '⁻' : '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]);
  const sci = (x, d = 2) => { const [m, e] = x.toExponential(d).split('e'); return m + ' × 10' + sup(+e); };
  const fmtTime = s => { const y = s / K.yr; if (s < 1e-9) return sci(s, 1) + ' s'; if (s < 1e-6) return sig(s * 1e9, 3) + ' ns'; if (s < 1e-3) return sig(s * 1e6, 3) + ' µs'; if (s < 1) return sig(s * 1e3, 3) + ' ms'; if (s < 120) return sig(s, 3) + ' s'; if (s < 7200) return sig(s / 60, 3) + ' min'; if (s < 172800) return sig(s / 3600, 3) + ' h'; if (y < 1) return sig(s / 86400, 3) + ' days';
    if (y < 1e4) return group(+y.toPrecision(3)) + ' years'; if (y < 1e6) return sig(y / 1e3, 3) + ' thousand years'; if (y < 1e9) return sig(y / 1e6, 3) + ' million years'; return sig(y / 1e9, 4) + ' billion years'; };
  const fmtBig = x => { if (x < 1e6) return x >= 100 ? group(x) : sig(x, 2); if (x < 1e9) return sig(x / 1e6, 3) + ' million'; if (x < 1e12) return sig(x / 1e9, 3) + ' billion'; if (x < 1e15) return sig(x / 1e12, 3) + ' trillion'; const e = Math.floor(Math.log10(x)); return sig(x / Math.pow(10, e), 2) + ' × 10' + sup(e); };

  /* the waypoints, widest first. id: the brain's seven keep 0 to 6; wider ones count down from −1, finer ones up from 7.
     fov: frame width at the waypoint (m). unit: metres per unit of the picture's own coordinates.
     kind: 'seen' (an instrument has resolved it), 'inferred' (measured, but no picture exists), 'imagined' (no data at all). */
  const B = HFD.LEVELS, GLY = K.Gly, MLY = 1e6 * K.ly;
  const LEVELS = [
    { id: -15, key: 'horizon',  name: 'The edge of what can be seen', short: 'Horizon',      fov: 2.4e27,  unit: GLY,        kind: 'inferred', tool: 'No instrument: a map computed from the Friedmann equation' },
    { id: -14, key: 'cmb',      name: 'The wall of light',            short: 'First light',  fov: 8.5e26,  unit: 3.7e26,     kind: 'seen',     tool: 'Microwave telescopes in space (COBE, WMAP, Planck)' },
    { id: -13, key: 'jwst',     name: 'The first galaxies',           short: 'First galaxies', fov: 5.6e26, unit: 1e26,      kind: 'seen',     tool: 'James Webb Space Telescope, infrared' },
    { id: -12, key: 'web',      name: 'The cosmic web',               short: 'Cosmic web',   fov: 2.7e26,  unit: GLY,        kind: 'seen',     tool: 'Galaxy redshift surveys (SDSS, DESI)' },
    { id: -11, key: 'laniakea', name: 'Laniakea, our supercluster',   short: 'Laniakea',     fov: 6.2e24,  unit: MLY,        kind: 'seen',     tool: 'Galaxy distances and velocities (Cosmicflows)' },
    { id: -10, key: 'local',    name: 'The Local Group',              short: 'Local Group',  fov: 6.6e22,  unit: MLY,        kind: 'seen',     tool: 'Optical and infrared telescopes' },
    { id: -9,  key: 'galaxy',   name: 'The Milky Way',                short: 'Milky Way',    fov: 1.3e21,  unit: 1e3 * K.ly, kind: 'inferred', tool: 'Gaia astrometry, radio and infrared surveys' },
    { id: -8,  key: 'stars',    name: 'The nearest stars',            short: 'Nearest stars', fov: 2.7e17, unit: K.ly,       kind: 'seen',     tool: 'Parallax (the Gaia satellite)' },
    { id: -7,  key: 'oort',     name: 'The Oort cloud',               short: 'Oort cloud',   fov: 3.6e16,  unit: 1e3 * K.au, kind: 'inferred', tool: 'No instrument: inferred from the orbits of comets' },
    { id: -6,  key: 'helio',    name: 'The Sun’s bubble, and the Voyagers', short: 'Voyagers', fov: 7.2e13, unit: K.au,     kind: 'seen',     tool: 'Voyager 1 and 2, in place' },
    { id: -5,  key: 'solar',    name: 'The Solar System',             short: 'Solar System', fov: 1.02e13, unit: K.au,       kind: 'seen',     tool: 'Telescopes, radar and spacecraft' },
    { id: -4,  key: 'moon',     name: 'Earth and Moon',               short: 'Earth–Moon',   fov: 6.4e8,   unit: 1e6,        kind: 'seen',     tool: 'Lunar laser ranging' },
    { id: -3,  key: 'earth',    name: 'The Earth',                    short: 'Earth',        fov: 1.75e7,  unit: 1e6,        kind: 'seen',     tool: 'A camera a million miles away (DSCOVR EPIC)' },
    { id: -2,  key: 'city',     name: 'A city at night',              short: 'City',         fov: 3.0e4,   unit: 1e3,        kind: 'seen',     tool: 'A camera in low orbit' },
    { id: -1,  key: 'person',   name: 'A person',                     short: 'You',          fov: 2.5,     unit: 1,          kind: 'seen',     tool: 'Whole-body magnetic resonance imaging' },
    ...B.map((L, i) => ({ id: i, key: L.id, name: L.name, short: ['Brain', 'Cortex', 'Neuron', 'Wiring', 'Synapse', 'Molecules', 'Atoms'][i], fov: L.fov, unit: [1e-3, 1e-6, 1e-6, 1e-6, 1e-9, 1e-9, 1e-10][i], kind: 'seen', tool: L.tool })),
    { id: 7,   key: 'atom',     name: 'Inside one atom',              short: 'One atom',     fov: 4.6e-10, unit: 1e-12,      kind: 'inferred', tool: 'No microscope: computed from quantum mechanics' },
    { id: 8,   key: 'nucleus',  name: 'The nucleus',                  short: 'Nucleus',      fov: 2.2e-14, unit: 1e-15,      kind: 'inferred', tool: 'Scattering: alpha particles, then electrons' },
    { id: 9,   key: 'proton',   name: 'Inside a proton',              short: 'Proton',       fov: 3.0e-15, unit: 1e-18,      kind: 'inferred', tool: 'Deep inelastic scattering' },
    { id: 10,  key: 'collide',  name: 'The smallest thing measured',  short: 'Collisions',   fov: 2.0e-19, unit: 1e-20,      kind: 'inferred', tool: 'The Large Hadron Collider, 13.6 TeV' },
    { id: 11,  key: 'desert',   name: 'The desert',                   short: 'Desert',       fov: 1.0e-24, unit: 1e-24,      kind: 'imagined', tool: 'Imagined: no instrument reaches here' },
    { id: 12,  key: 'planck',   name: 'The Planck length',            short: 'Planck',       fov: 12 * planck.l, unit: planck.l, kind: 'imagined', tool: 'Imagined: no instrument reaches here' }
  ];
  /* where one picture gives way to the next (frame widths). Between the brain's seven they are the brain model's own. */
  const CUTS = { '-15': 1.25e27, '-14': 7.0e26, '-13': 4.2e26, '-12': 2.2e25, '-11': 6e23, '-10': 9e21, '-9': 2.2e18, '-8': 9.5e16, '-7': 2.2e14, '-6': 2.3e13, '-5': 8e9, '-4': 7.5e7, '-3': 5e5, '-2': 45, '-1': 0.42,
    6: 1.1e-9, 7: 2.4e-13, 8: 7.5e-15, 9: 2.4e-17, 10: LIMITS.seenFloor, 11: 4e-33 };
  LEVELS.forEach((L, k) => { L.k = k; L.top = k === 0 ? W0 : null; });
  LEVELS.forEach((L, k) => { L.bot = k === LEVELS.length - 1 ? W1 : (L.id >= 0 && L.id < 6 ? B[L.id].bot : CUTS[L.id]); if (k > 0) L.top = LEVELS[k - 1].bot; });
  LEVELS.forEach(L => { L.z = zoomOf(L.fov); L.fade = Math.min(0.2, 0.42 * Math.log10(L.fov / L.bot), L.k < LEVELS.length - 1 ? 0.42 * Math.log10(L.bot / LEVELS[L.k + 1].fov) : 1); });
  const NU = 15, LV = id => LEVELS[id + NU], IDMIN = -NU, IDMAX = LEVELS.length - 1 - NU, HOME = -1;
  /* which pictures are on screen at a frame width, and how strongly: {a, b, t} with ids; t is the weight of b (the finer one) */
  const levelMix = fov => { const lf = Math.log10(fov);
    for (let k = 0; k < LEVELS.length - 1; k++) { const L = LEVELS[k], Bc = Math.log10(L.bot), F = L.fade; if (lf > Bc + F / 2) return { a: L.id, b: null, t: 0 }; if (lf >= Bc - F / 2) return { a: L.id, b: L.id + 1, t: smooth((Bc + F / 2 - lf) / F) }; }
    return { a: IDMAX, b: null, t: 0 }; };
  const levelAt = fov => { const m = levelMix(fov); return m.b !== null && m.t > 0.5 ? m.b : m.a; };
  const zone = fov => fov < SEEN.floor ? 'below' : fov > SEEN.ceiling ? 'beyond' : 'seen';

  /* ════ 9. CRR: the checks. The definitions are those of the programme's own statement (crr-explained.html).
     Nothing below is physics that the cited authors proposed or endorse; each check says what it can and cannot show. ════ */
  /* Fisher–Rao distance between two Bernoulli distributions with parameters p and q: 2 |asin sqrt(p) − asin sqrt(q)| */
  const fisherRao = (p, q) => 2 * Math.abs(Math.asin(Math.sqrt(p)) - Math.asin(Math.sqrt(q)));
  /* the same by brute force: arc length ∫ sqrt(g) dp with g = 1/(p(1−p)), by the midpoint rule. The integrand is infinite, but integrable,
     at p = 0 and p = 1; the substitution p = u² below one half, and 1 − p = u² above it, removes the square root from each end.
     (First version: a plain midpoint rule in p stopping 1e-6 short of the ends. It was 1.65 parts in 10⁴ low and failed bet C1.) */
  const frHalf = (lo, hi, n) => { const a = Math.sqrt(lo), b = Math.sqrt(hi), hh = (b - a) / n; let s = 0; for (let i = 0; i < n; i++) { const u = a + (i + 0.5) * hh; s += 2 * hh / Math.sqrt(1 - u * u); } return s; };
  const fisherRaoNumeric = (p, q, n = 20000) => { const a = Math.min(p, q), b = Math.max(p, q); return (a < 0.5 ? frHalf(a, Math.min(b, 0.5), n) : 0) + (b > 0.5 ? frHalf(1 - b, 1 - Math.max(a, 0.5), n) : 0); };
  const OMEGA_Z2 = 1 / PI, CV_CANDIDATE = 1 / (2 * PI);
  const cvOf = xs => { const n = xs.length, m = xs.reduce((s, v) => s + v, 0) / n, v = xs.reduce((s, x) => s + (x - m) * (x - m), 0) / (n - 1); return { n, mean: m, sd: Math.sqrt(v), cv: Math.sqrt(v) / m }; };
  const CRR = (() => { const out = {};
    /* A. the Z2 diameter. The distance from p = 0 to p = 1 is π, so Ω = 1/π for a two-state carrier. */
    { out.A = { numeric: fisherRaoNumeric(0, 1), closed: fisherRao(0, 1), target: PI, omega: 1 / fisherRaoNumeric(0, 1), partial: [fisherRaoNumeric(0.1, 0.7), fisherRao(0.1, 0.7)] }; }
    /* B. recombination read as one Z2 occasion. Carrier: one hydrogen atom, ionised with probability x(z) from the Saha equation.
       C* is the Fisher–Rao distance from the fully ionised start; the cut is at C*Ω = 1. Because x falls monotonically, the path never
       turns back and the surplus S = C − C* is zero. */
    { const cs = z => fisherRao(1, saha(z)) * OMEGA_Z2, zHalfTurn = solve(z => cs(z) - 0.5, 800, 2500), final = fisherRao(1, RECOMB.xResidual) * OMEGA_Z2;
      out.B = { zHalfTurn, zSahaHalf: RECOMB.zHalf, atZStar: cs(PLANCK18.zStar), final, reached: final >= 1, shortBy: 1 - final, z90: solve(z => cs(z) - 0.9, 300, 2500), z10: solve(z => cs(z) - 0.1, 800, 4000) }; }
    /* C. the spacing of the waypoints, as steps in ln(frame width), against the programme's candidate spread 1/2π ≈ 0.159.
       The waypoints were chosen by the page's author, so this tests the author's choices and nothing in nature. */
    { const steps = []; for (let k = 1; k < LEVELS.length; k++) steps.push(Math.log(LEVELS[k - 1].fov / LEVELS[k].fov)); out.C = Object.assign(cvOf(steps), { target: CV_CANDIDATE, steps });
      /* a second list that the page did not choose: the SI prefixes, which are evenly spaced by construction (CV = 0) */
      out.C.controlCv = 0; }
    /* D. numbers on this page that happen to sit near 1/π */
    { const d = PLANCK18.Om - OMEGA_Z2, zEqual = solve(z => COSMO.OmAt(z) - OMEGA_Z2, -0.5, 5);
      out.D = { Om: PLANCK18.Om, target: OMEGA_Z2, diff: d, sigmaPlanck: Math.abs(d) / PLANCK18.OmErr, sigmaDesi: Math.abs(DESI2.Om - OMEGA_Z2) / DESI2.OmErr, zEqual,
        OmAt1: COSMO.OmAt(1), OmAtStar: COSMO.OmAt(PLANCK18.zStar), crossOverPlanck: crossing.overPlanck, sqrtPi: Math.sqrt(PI), reducedOverPlanck: crossing.reduced / planck.m }; }
    /* E. a claim made elsewhere on this site (darkenergy.html; claims/P4): that the dark-energy density peaks at the redshift where its
       equation of state crosses w = −1. For w(a) = w0 + wa(1 − a) with the DESI DR2 values both are found here independently:
       the crossing by solving w(a) = −1, the peak by a search for the maximum of ρ_DE(a). */
    { const w0 = DESI2.w0, wa = DESI2.wa, w = a => w0 + wa * (1 - a), rho = a => Math.pow(a, -3 * (1 + w0 + wa)) * Math.exp(-3 * wa * (1 - a));
      const aCross = 1 - (-1 - w0) / wa; let lo = 0.2, hi = 1; for (let i = 0; i < 200; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (rho(m1) < rho(m2)) lo = m1; else hi = m2; } const aPeak = (lo + hi) / 2;
      /* the same for an arbitrary other model, to see whether the coincidence is special: w(a) = −1.3 + 0.9 a² */
      const w2 = a => -1.3 + 0.9 * a * a, lnRho2 = a => { let s = 0; const n = 4000; for (let i = 0; i < n; i++) { const x = a + (1 - a) * (i + 0.5) / n; s += 3 * (1 + w2(x)) / x * (1 - a) / n; } return s; };
      const aCross2 = Math.sqrt(0.3 / 0.9); let lo2 = 0.2, hi2 = 1; for (let i = 0; i < 80; i++) { const m1 = lo2 + (hi2 - lo2) / 3, m2 = hi2 - (hi2 - lo2) / 3; if (lnRho2(m1) < lnRho2(m2)) lo2 = m1; else hi2 = m2; }
      out.E = { zCross: 1 / aCross - 1, zPeak: 1 / aPeak - 1, wToday: w(1), wPast: w(0.3), peakOverToday: rho(aPeak), otherCross: 1 / aCross2 - 1, otherPeak: 2 / (lo2 + hi2) - 1, sigma: DESI2.sigmaRange }; }
    /* F. the crossovers of cosmic history as events in ln a: equality, recombination, the onset of acceleration. Three events give two
       intervals, too few for any statistic of spread to mean anything. Reported, not tested. */
    { const ev = [COSMO.zEq, RECOMB.zHalf, COSMO.zAccel].map(z => -Math.log(1 + z)), gaps = [ev[1] - ev[0], ev[2] - ev[1]]; out.F = { events: ev, gaps, n: gaps.length, cv: cvOf(gaps).cv }; }
    return out; })();

  /* ════ 10. Time on this page: the calculations of the section "Time on this page: a CRR reading" ════
     Everything here is standard physics or standard geometry of probability, relabelled. The bets (T1 to T12) are in BETS.md
     and were written before this section was run. Angles are Fisher–Rao lengths in radians (σ = 1 rad); a half-turn is π. */
  const TIME = (() => { const T = {}, H0 = COSMO.H0, Om = COSMO.P.Om, Or = COSMO.Or, OL = COSMO.OL;
    /* the Gudermannian and its inverse: sin(gd x) = tanh x */
    const gd = x => 2 * Math.atan(Math.tanh(x / 2)), gdInv = y => Math.log(Math.tan(PI / 4 + y / 2));
    T.gd = gd; T.gdInv = gdInv;

    /* ── A. energy ↔ time. Radiation era: H² = (8πG/3) ρ, ρ = (π²/30) g* (kT)⁴/(ħc)³, a ∝ t^(1/2), so t = 1/(2H):
       t = sqrt(90/(32π³ g*)) ħ E_P/(kT)²  ≈  2.42 g*^(−1/2) (T/MeV)^(−2) s   (Kolb & Turner 1990; Olive & Peacock in the Review of
       Particle Physics 2024, eq. 22.43–22.44, who write 2.4). g*: the Review's table of degrees of freedom, as steps at each mass
       (a crude setting: a species really fades over a range of temperature). Below the electron mass the Review's 7.25 counts
       neutrinos at the photons' temperature; they are cooler by (4/11)^(1/3) after the electrons go, which gives 3.38, used here. */
    const gLate = 2 + 7 / 8 * 2 * PLANCK18.Neff * Math.pow(4 / 11, 4 / 3);
    const GSTEPS = [[0.51099895e-3, gLate, 'photons and neutrinos'], [0.1056584, 10.75, 'and electrons'], [0.13957, 14.25, 'and muons'], [0.155 /* setting: the quark–hadron transition, about 155 MeV */, 17.25, 'and pions'],
      [1.27, 61.75, 'quarks u, d, s and gluons'], [1.77686, 72.25, 'and charm'], [4.18, 75.75, 'and tau'], [80.377, 86.25, 'and bottom'], [125.25, 95.25, 'and W, Z'], [172.69, 96.25, 'and the Higgs'], [Infinity, 106.75, 'and top: all known particles']];   /* upper edge (GeV), g* below it */
    T.gLate = gLate; T.GSTEPS = GSTEPS;
    T.gStar = kTGeV => { for (const g of GSTEPS) if (kTGeV < g[0]) return g[1]; return 106.75; };
    T.tRad = (kTJ, g) => Math.sqrt(90 / (32 * PI * PI * PI * g)) * K.hbar * planck.E / (kTJ * kTJ);
    T.coeff = T.tRad(1e6 * K.eV, 1);                                        /* the "2.42 s" */
    T.kTswitch = 1e4 * K.eV;                                                /* below 10 keV the page's own Friedmann integral is used (it knows about matter and dark energy) */
    /* the time at which kT = E. Convention: a typical particle then carried E; the mean energy is about 3 kT. */
    T.timeOfEnergy = EJ => { const Tk = EJ / K.kB, zz = Tk / K.Tcmb - 1; if (EJ >= T.kTswitch) { const g = T.gStar(EJ / K.GeV); return { t: T.tRad(EJ, g), g, z: zz, how: 'radiation', future: false }; }
      return { t: COSMO.tOfA(1 / (1 + zz)), g: gLate, z: zz, how: 'Friedmann', future: zz < 0 }; };
    T.timeOfLength = l => T.timeOfEnergy(K.h * K.c / l);
    T.lambdaNow = K.h * K.c / (K.kB * K.Tcmb);                              /* the wavelength at which kT₀ = hc/λ today: where the two branches meet */
    T.ENERGIES = [['the Planck energy', planck.E], ['35.8 TeV, the first hard line', 35.8 * K.TeV], ['13.6 TeV, the LHC', 13.6 * K.TeV], ['1 GeV', K.GeV], ['1 MeV', 1e6 * K.eV], ['10 keV', 1e4 * K.eV], ['13.6 eV', RYD_EV * K.eV]]
      .map(e => Object.assign({ name: e[0], E: e[1] }, T.timeOfEnergy(e[1])));
    T.planckRatio = T.tRad(planck.E, 106.75) / planck.t;
    T.crossCheck = (() => { const E = T.kTswitch, a = T.tRad(E, gLate), b = COSMO.tOfA(K.kB * K.Tcmb / E); return { rad: a, friedmann: b, rel: Math.abs(a - b) / b }; })();
    /* every stop, both branches. Outward: when the light now arriving from the frame's edge set out. Inward: when kT = hc/(frame width). */
    T.STOPS = LEVELS.map(L => { const half = L.fov / 2; if (L.id < HOME) { const beyond = half >= COSMO.horizon, lb = beyond ? COSMO.age : COSMO.lookbackOfChi(half), z = beyond ? Infinity : COSMO.zOfChi(half); return { id: L.id, name: L.name, short: L.short, branch: 'out', fov: L.fov, t: Math.max(0, COSMO.age - lb), z, beyond, kind: L.kind }; }
      if (L.id === HOME) return { id: L.id, name: L.name, short: L.short, branch: 'here', fov: L.fov, t: COSMO.age, z: 0, kind: L.kind };
      const r = T.timeOfLength(L.fov); return { id: L.id, name: L.name, short: L.short, branch: 'in', fov: L.fov, t: r.t, z: r.z, E: K.h * K.c / L.fov, g: r.g, future: r.future, how: r.how, kind: L.kind }; });
    T.seenIn = T.timeOfEnergy(LIMITS.leptonQuarkCI * K.TeV).t; T.seenOut = LAST.t;        /* the two hard lines, as times after the beginning */
    T.geoMean = Math.sqrt(planck.l * COSMO.horizon);                        /* B′. the geometric mean of the Planck length and the horizon radius */

    /* ── B. the contents as one path. p = (radiation, matter, dark energy) shares; the point 2 sqrt(p) lies on an octant of a
       sphere of radius 2. Distance: 2 acos(Σ sqrt(p_i q_i)), written with the chord, 4 asin(|sqrt p − sqrt q| / 2), which keeps its
       accuracy for neighbouring points. */
    const fisherRao3 = (p, q) => { let s = 0; for (let i = 0; i < p.length; i++) { const d = Math.sqrt(p[i]) - Math.sqrt(q[i]); s += d * d; } return 4 * Math.asin(Math.min(1, Math.sqrt(s) / 2)); };
    const fisherRao3acos = (p, q) => { let s = 0; for (let i = 0; i < p.length; i++) s += Math.sqrt(p[i] * q[i]); return 2 * Math.acos(Math.min(1, s)); };
    const sharesOf = C => a => { const r = C.Or / (a * a * a * a), m = C.P.Om / (a * a * a), l = C.OL * C.de(a), s = r + m + l; return [r / s, m / s, l / s]; };
    const shares = sharesOf(COSMO), CORNER = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    /* arc length of any path u ↦ shares(e^u) between u0 and u1, as a polyline of chords; the two ends beyond are closed with the
       geodesic to the corner the path starts from and the one it ends at */
    const arc = (sh, u0, u1, n, from, to) => { const U = new Float64Array(n + 1), Cc = new Float64Array(n + 1); let prev = sh(Math.exp(u0)); Cc[0] = from ? fisherRao3(from, prev) : 0; U[0] = u0;
      for (let i = 1; i <= n; i++) { const u = u0 + (u1 - u0) * i / n, p = sh(Math.exp(u)); U[i] = u; Cc[i] = Cc[i - 1] + fisherRao3(prev, p); prev = p; } return { U, C: Cc, total: Cc[n] + (to ? fisherRao3(prev, to) : 0), u0, u1, n }; };
    const PATH = arc(shares, -34, 16, 25000, CORNER[0], CORNER[2]);
    const lerpTab = (xs, ys, x) => { const n = xs.length - 1, f = (x - xs[0]) / (xs[n] - xs[0]) * n, i = Math.max(0, Math.min(n - 1, Math.floor(f))); return ys[i] + (ys[i + 1] - ys[i]) * (f - i); };
    T.fisherRao3 = fisherRao3; T.fisherRao3acos = fisherRao3acos; T.shares = shares; T.sharesOf = sharesOf; T.arc = arc; T.CORNER = CORNER; T.PATH = PATH;
    /* the state of the path when the universe was a times its present size. Lengths in radians and in half-turns (÷π, that is ×Ω). */
    T.pathAt = a => { const u = clamp(Math.log(a), PATH.u0, PATH.u1), p = shares(a), C = u <= PATH.u0 ? fisherRao3(CORNER[0], p) : lerpTab(PATH.U, PATH.C, u), Cs = fisherRao3(CORNER[0], p);
      return { a, z: 1 / a - 1, p, C, Cstar: Cs, S: C - Cs, Ch: C / PI, CsOmega: Cs / PI, Sh: (C - Cs) / PI, walked: C / PATH.total }; };
    T.shortfall = a => 2 * Math.asin(Math.sqrt(shares(a)[0]));              /* π − C*: what is left before the cut, exactly; it never reaches 0 */
    T.speed = a => { const p = shares(a), nb = 4 * p[0] + 3 * p[1]; return Math.sqrt(Math.max(0, p[0] * (4 - nb) * (4 - nb) + p[1] * (3 - nb) * (3 - nb) + p[2] * nb * nb)); };   /* per e-fold: the s.d. of n = 3(1 + w) over the contents */
    T.q = a => { const p = shares(a); return (4 * p[0] + 3 * p[1]) / 2 - 1; };   /* the deceleration parameter: q = n̄/2 − 1 */
    T.whole = { C: PATH.total, Ch: PATH.total / PI, CsOmega: 1, Sh: PATH.total / PI - 1 };
    T.now = T.pathAt(1);
    /* matter's largest share: found numerically (a search in ln a) and by the formula a⁴ = Ω_r/(3Ω_Λ) */
    T.peak = (() => { let lo = Math.log(1e-3), hi = 0; for (let i = 0; i < 200; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (shares(Math.exp(m1))[1] < shares(Math.exp(m2))[1]) lo = m1; else hi = m2; }
      const a = Math.exp((lo + hi) / 2), an = Math.pow(Or / (3 * OL), 0.25); return Object.assign(T.pathAt(a), { share: shares(a)[1], analyticA: an }); })();
    T.EVENTS = [{ key: 'eq', name: 'matter overtakes radiation', z: COSMO.zEq }, { key: 'peak', name: 'matter’s largest share', z: T.peak.z }, { key: 'accel', name: 'slowing turns to speeding', z: COSMO.zAccel }, { key: 'lambda', name: 'dark energy overtakes matter', z: COSMO.zLambda }, { key: 'now', name: 'now', z: 0 }]
      .map(e => Object.assign(e, T.pathAt(1 / (1 + e.z)), { t: COSMO.t(e.z) }));
    /* the same construction for any list of fluids (exponents n = 3(1+w), amplitudes today): the whole path in half-turns, and the
       largest share the middle one reaches. Used to show that the pattern is geometry. */
    T.generic = (ns, amps, u0 = -60, u1 = 60, n = 60000) => { const sh = a => { const v = ns.map((k, i) => amps[i] * Math.pow(a, -k)), s = v[0] + v[1] + v[2]; return v.map(x => x / s); }, A = arc(sh, u0, u1, n, CORNER[0], CORNER[2]); let mid = 0; for (let i = 0; i <= 2000; i++) mid = Math.max(mid, sh(Math.exp(u0 + (u1 - u0) * i / 2000))[1]); return { Ch: A.total / PI, mid }; };
    T.GENERIC = { stiff: T.generic([6, 3, 2], [1e-6, 1, 1e-3]), thirds: T.generic([4, 3, 0], [1, 1, 1]) };

    /* ── C. the last leg alone: matter and dark energy, no radiation. Ω_Λ(a) = Ω_Λ a³/(Ω_m + Ω_Λ a³), φ = 2 asin sqrt(Ω_Λ(a)).
       The textbook solution a(t) = (Ω_m/Ω_Λ)^(1/3) sinh^(2/3)(t/t_Λ) (e.g. Ryden 2016) gives Ω_Λ(a) = tanh²(t/t_Λ), so sin(φ/2) = tanh(t/t_Λ):
       φ/2 = gd(t/t_Λ). */
    const OLn = 1 - Om, tL = 2 / (3 * H0 * Math.sqrt(OLn));
    const LEG = { Om, OL: OLn, tLambda: tL };
    LEG.share = a => OLn * a * a * a / (Om + OLn * a * a * a);
    LEG.phiOfA = a => 2 * Math.asin(Math.sqrt(LEG.share(a)));
    LEG.aOfT = t => Math.cbrt(Om / OLn) * Math.pow(Math.sinh(t / tL), 2 / 3);
    LEG.phiOfT = t => 2 * gd(t / tL);
    LEG.tOfPhi = phi => tL * gdInv(phi / 2);
    LEG.tOfA = a => LEG.tOfPhi(LEG.phiOfA(a));
    LEG.logOdds = a => Math.log(OLn / Om) + 3 * Math.log(a);
    LEG.accel = (() => { const phi = 2 * Math.asin(Math.sqrt(1 / 3)); return { phi, frac: phi / PI, t: LEG.tOfPhi(phi), z: Math.cbrt(2 * OLn / Om) - 1 }; })();
    LEG.equal = { phi: PI / 2, frac: 0.5, t: tL * Math.log(1 + Math.SQRT2), tGd: LEG.tOfPhi(PI / 2), z: Math.cbrt(OLn / Om) - 1 }; LEG.equal.efoldsAgo = Math.log(1 + LEG.equal.z);
    LEG.now = { phi: LEG.phiOfA(1), frac: LEG.phiOfA(1) / PI, t: LEG.tOfA(1) };
    /* with a resolution σ (radians) the cut comes when φ ≥ π − σ:  t = t_Λ gd⁻¹(π/2 − σ/2) = t_Λ ln cot(σ/4) ≈ t_Λ ln(4/σ) */
    LEG.cutTime = s => tL * -Math.log(Math.tan(s / 4)); LEG.cutApprox = s => tL * Math.log(4 / s);
    /* the error of the relation when radiation is put back: the true cosmic time of the page's cosmology goes into the formula */
    LEG.errWithRadiation = a => { const p = shares(a); return (2 / PI) * (gd(COSMO.tOfA(a) / tL) - Math.asin(Math.sqrt(p[2] / (p[1] + p[2])))); };
    T.LEG = LEG;
    /* the resolution at which a Z₂ occasion that stopped with a share x left over counts as complete: 2 asin sqrt(x) */
    T.sigmaNeeded = x => 2 * Math.asin(Math.sqrt(x));
    T.RESIDUALS = [1e-4, RECOMB.xResidual, 5e-4].map(x => ({ x, sigma: T.sigmaNeeded(x), stopsAt: fisherRao(1, x) / PI }));
    /* the three-way path of B at resolution σ: its cut is where 2 asin sqrt(radiation's share) = σ */
    T.threeWayCut = s => { if (s >= PI) return null; const a = Math.exp(solve(u => T.shortfall(Math.exp(u)) - s, -40, 40, 100)); return { a, z: 1 / a - 1, t: COSMO.tOfA(a), past: a <= 1 }; };

    /* ── D. conformal time: the comoving distance light has covered (the horizon) and all it can still cover, c ∫ da/(a²H) from 1 to ∞ */
    T.CONFORMAL = (() => { const n = 4000, hh = 1 / n; let s = 0; for (let i = 0; i <= n; i++) { const v = i * hh, f = v === 0 ? 1 / Math.sqrt(OL) : 1 / COSMO.E(1 / v); s += f * (i === 0 || i === n ? 1 : i & 1 ? 4 : 2); } const future = s * hh / 3 * K.c / H0; return { past: COSMO.horizon, future, frac: COSMO.horizon / (COSMO.horizon + future) }; })();

    /* ── E. the growth of structure. Linear growth of matter inside the horizon, radiation taken as smooth:
       D″ + (2 + d ln H/d ln a) D′ − (3/2) Ω_m(a) D = 0, primes in ln a; d ln H/d ln a = −(4 p_r + 3 p_m)/2.
       Started deep in the radiation era on the growing solution for matter and radiation alone, D = 1 + (3/2) a/a_eq (Mészáros 1974).
       Fourth-order Runge–Kutta. f = D′/D is the growth rate; Linder (2005) gave the rule f ≈ Ω_m(a)^0.55. */
    T.GROWTH = (() => { const u0 = Math.log(1e-7), u1 = 12, n = 28000, hh = (u1 - u0) / n, U = new Float64Array(n + 1), D = new Float64Array(n + 1), F = new Float64Array(n + 1), aeq = Or / Om;
      const acc = (u, d, v) => { const p = shares(Math.exp(u)); return -(2 - (4 * p[0] + 3 * p[1]) / 2) * v + 1.5 * p[1] * d; };
      let d = 1 + 1.5 * Math.exp(u0) / aeq, v = 1.5 * Math.exp(u0) / aeq; U[0] = u0; D[0] = d; F[0] = v / d;
      for (let i = 1; i <= n; i++) { const u = u0 + (i - 1) * hh, k1d = v, k1v = acc(u, d, v), k2d = v + hh / 2 * k1v, k2v = acc(u + hh / 2, d + hh / 2 * k1d, v + hh / 2 * k1v), k3d = v + hh / 2 * k2v, k3v = acc(u + hh / 2, d + hh / 2 * k2d, v + hh / 2 * k2v), k4d = v + hh * k3v, k4v = acc(u + hh, d + hh * k3d, v + hh * k3v);
        d += hh / 6 * (k1d + 2 * k2d + 2 * k3d + k4d); v += hh / 6 * (k1v + 2 * k2v + 2 * k3v + k4v); U[i] = u + hh; D[i] = d; F[i] = v / d; }
      const Dof = a => lerpTab(U, D, clamp(Math.log(a), u0, u1)), fOf = a => lerpTab(U, F, clamp(Math.log(a), u0, u1)), rule = a => Math.pow(shares(a)[1], 0.55);
      /* the same ratio by the rule alone: exp ∫ Ω_m^0.55 d ln a from now on (Simpson) */
      let s = 0; const m = 6000, h2 = 12 / m; for (let i = 0; i <= m; i++) s += rule(Math.exp(i * h2)) * (i === 0 || i === m ? 1 : i & 1 ? 4 : 2);
      let worst = 0, worstZ = 0; for (let z = 0; z <= 5.0001; z += 0.01) { const a = 1 / (1 + z), e = Math.abs(fOf(a) - rule(a)); if (e > worst) { worst = e; worstZ = z; } }
      return { D: Dof, f: fOf, rule, fNow: fOf(1), ruleNow: rule(1), DinfOverNow: D[n] / Dof(1), ruleInfOverNow: Math.exp(s * h2 / 3), D10OverInf: Dof(10) / D[n], worst, worstZ, sinceEq: Dof(1) / Dof(aeq), U, Dtab: D, Ftab: F }; })();

    /* ── F. hydrogen, there and back. x: the share ionised. Down by Saha to the residual; back up by the tanh step in (1+z)^(3/2)
       that Planck 2018 uses for its mid-point redshift z_re = 7.67 ± 0.73 (width 0.5 in redshift). Helium is left out, and the model
       takes hydrogen back to fully ionised. */
    T.REION = { z: 7.67, err: 0.73, dz: 0.5 };
    T.xH = (z, xres = RECOMB.xResidual) => { const xr = Math.max(saha(z), xres), y = Math.pow(1 + z, 1.5), yre = Math.pow(1 + T.REION.z, 1.5), dy = 1.5 * Math.sqrt(1 + T.REION.z) * T.REION.dz; return xr + (1 - xr) * 0.5 * (1 + Math.tanh((yre - y) / dy)); };
    T.roundTrip = (xres = RECOMB.xResidual, n = 6000) => { const l0 = Math.log(1 + 3000), Z = new Float64Array(n + 1), X = new Float64Array(n + 1), Cc = new Float64Array(n + 1), Cs = new Float64Array(n + 1); let prev = T.xH(3000, xres), c = fisherRao(1, prev), maxCs = 0;
      for (let i = 0; i <= n; i++) { const z = Math.exp(l0 * (1 - i / n)) - 1, x = T.xH(z, xres); c += fisherRao(prev, x); prev = x; Z[i] = z; X[i] = x; Cc[i] = c / PI; Cs[i] = fisherRao(1, x) / PI; maxCs = Math.max(maxCs, Cs[i]); }
      const S = Cc[n] - Cs[n]; return { Z, X, C: Cc, Cs, n, Ch: Cc[n], CsEnd: Cs[n], maxCs, Sh: S, salience: Math.exp(S) }; };   /* e^(SΩ) with S in radians and Ω = 1/π is e^(S in half-turns) */
    T.HYDROGEN = T.roundTrip();

    /* ── G. if DESI's hinted w(a) = w0 + wa(1 − a) and Ω_m are taken literally and run into the future, which the formula was not made for */
    T.DESI = (() => { const C = Cosmo({ Om: DESI2.Om, w0: DESI2.w0, wa: DESI2.wa }), sh = sharesOf(C), two = a => { const p = sh(a); return p[2] / (p[1] + p[2]); };
      let lo = 1, hi = 6; for (let i = 0; i < 200; i++) { const m1 = lo + (hi - lo) / 3, m2 = hi - (hi - lo) / 3; if (two(m1) < two(m2)) lo = m1; else hi = m2; } const aPeak = (lo + hi) / 2;
      return { aPeak, share: two(aPeak), share3: sh(aPeak)[2], maxFrac: 2 * Math.asin(Math.sqrt(two(aPeak))) / PI, aOnePercent: solve(a => two(a) - 0.01, aPeak, 40), nowFrac: 2 * Math.asin(Math.sqrt(two(1))) / PI, rateNow: -3 * (DESI2.w0), two }; })();

    /* ── H. four clocks for "how far through are we?" */
    T.CLOCKS = { years: COSMO.age, conformal: T.CONFORMAL.frac, path: T.now.walked, leg: LEG.now.frac };
    return T; })();

  return { K, planck, compton, schwarzschild, crossing, bohr, R0_FM, nuclearRadius, closestApproach, rutherford, KION, lambdaOfE, energyOfLambda, LHC, bendRadius, ringFor, LIMITS, ACCEL, INFL,
    PLANCK18, SH0ES, DESI2, Cosmo, COSMO, COSMO_SH0ES, LAST, RYD_EV, saha, RECOMB, solve, lightTime, VOY, voyager, PLANETS, planetAt, STARS, starsWithin, SGRA, MILKYWAY, LANIAKEA, JWST,
    cmbDl, cmbCl, cmbSky, W0, W1, LN0, LN1, DECADES, lnFovOf, fovOf, zoomOfLn, zoomOf, SPAN, HUMAN, SEEN, UNITS, unitOf, fmtLen, scaleBar, fmtTime, fmtBig, sup, sci, sig, group,
    LEVELS, NU, LV, IDMIN, IDMAX, HOME, levelMix, levelAt, zone, fisherRao, fisherRaoNumeric, OMEGA_Z2, CV_CANDIDATE, cvOf, CRR, TIME, clamp, smooth, rng, HFD };
})();
if (typeof window !== 'undefined') { window.HFD = HFD; window.EV = EV; }
if (typeof module !== 'undefined') module.exports = EV;
