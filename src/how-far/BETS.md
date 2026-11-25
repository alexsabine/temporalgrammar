# Bets for how-far-does-it-go.html, written before the model was first run

Written on 25 November 2026, after model.js was typed and before it was executed for the first time. Each bet has a
threshold. Results go in RESULTS.md and test_output.txt as they come out. A bet changed after a result is seen is listed
in RESULTS.md with the reason, and the earlier output is kept.

Tabulated values are CODATA 2022 (NIST, read 25 Nov 2026) and Planck 2018 table 2 (TT,TE,EE+lowE+lensing).

## Physics

P1 Planck units from ħ, G, c agree with the CODATA table: length, mass, time and temperature each within 2 parts in 10⁵
   (the table's own uncertainty is 1.1 in 10⁵).
P2 Compton = Schwarzschild. The crossing mass found by bisection equals sqrt(hc/2G) within 1 part in 10⁹, and is
   sqrt(π) Planck masses within 1 part in 10⁹. At the crossing the common length is 2 sqrt(π) Planck lengths (same tolerance).
P3 Bohr radius from ε0, ħ, m_e, e agrees with the CODATA value within 1 part in 10⁸.
P4 Nuclear radius law. R(A=8A0) = 2 R(A0) within 1 part in 10¹²; R(³⁹K) with r0 = 1.2 fm lies between 4.0 and 4.2 fm;
   R(¹⁹⁷Au) between 6.9 and 7.1 fm.
P5 Rutherford. A 5 MeV alpha aimed at gold (Z = 79) stops between 44 and 47 fm from the centre. The cross-section at 60°
   is 16 times that at 180° within 1 part in 10⁹ (the ratio of 1/sin⁴(θ/2)).
P6 Accelerator wavelength. hc/E at 13.6 TeV lies between 9.0 and 9.2 × 10⁻²⁰ m. ħc/Λ at Λ = 35.8 TeV lies between 5.4 and
   5.6 × 10⁻²¹ m.
P7 Ring size. The bending radius for a 7 TeV proton at 8.33 T is within 0.5% of the LHC's 2,803.95 m. A collider reaching
   the Planck energy with those magnets would be between 1,000 and 5,000 light-years around.
P8 Friedmann integration.
   a. age 13.797 Gyr ± 0.03;
   b. particle horizon between 45.5 and 47.0 billion light-years;
   c. matter–radiation equality at z = 3402 ± 50;
   d. comoving distance to z = 1089.92 between 13.7 and 14.0 Gpc;
   e. sound horizon at z = 1089.92 within 1% of 144.43 Mpc, and the acoustic angle 100θ within 1% of 1.0411;
   f. the two independent routes to a lookback time (direct integral, and the table used for distance → time) agree
      within 1 part in 10⁴ at z = 0.5, 3, 14.44 and 1089.92;
   g. with H0 = 73.04 and the other parameters held, the age is between 12.4 and 13.0 Gyr.
P9 Saha. Hydrogen is half ionised between z = 1300 and z = 1450, at a temperature between 3,500 and 4,000 K.
   The hydrogen binding energy computed from the constants is 13.606 eV ± 0.001.
P10 Light-travel times. Moon 1.28 s ± 0.01; Sun 499.0 s ± 0.1; Neptune (30.11 AU) 4.17 h ± 0.01; Voyager 1 from the Sun
    on 25 Nov 2026 between 23.8 and 24.0 h; Proxima Centauri 4.2465 yr ± 0.0005.
P11 Sagittarius A*. The shadow of a non-spinning black hole of 4.30 × 10⁶ Suns at 8,277 pc is within one quoted
    uncertainty (2.3 µas) of the measured ring, 51.8 µas. This is a weak bet: a ring is not a shadow.
P12 JWST. MoM-z14 (z = 14.44): age of the universe then between 270 and 300 million years (the paper says 280).
P13 Inflation scale. With r < 0.036 and ln(10¹⁰ A_s) = 3.044, H < 5 × 10¹³ GeV ± 10%, so c/H > 3 to 5 × 10⁻³⁰ m.
P14 CMB sky. For a sky drawn to ℓ = 200: the spectrum of the sky, averaged over ℓ in bands of 20, is within 20% of the
    input in every band above ℓ = 20; the variance of a synthesised ring-average of the map agrees with Σ(2ℓ+1)C_ℓ/4π
    within 15% (one sky, so cosmic variance is real and the tolerance is loose).
P15 Counts. Baryons in the observable universe between 10⁷⁹ and 10⁸¹; photon density 410.7 per cm³ ± 0.5.

## The zoom

Z1 fovOf and zoomOf are inverse over the whole range: for 10,001 evenly spaced z, |zoomOf(fovOf(z)) − z| < 10⁻¹².
Z2 The range is 61 to 63 powers of ten of frame width; horizon diameter ÷ Planck length is 61.5 to 62.0 powers of ten.
Z3 Every waypoint's frame lies strictly inside its own band (top > fov > bot), the bands tile the range with no gap,
   and levelAt(fov of waypoint) returns that waypoint, for all 28.
Z4 Units. fmtLen never returns a mantissa below 0.1 or of 10,000 or more for 2,000 log-spaced lengths, and the scale bar is
   never longer than 0.25 of the frame or shorter than 0.04 of it.
Z5 The seven brain levels keep their original frame widths and boundaries (compared with the brain model's own table).

## CRR (definitions from crr-explained.html; each bet says what it can show)

C1 Z₂ diameter. The Fisher–Rao length from p = 0 to p = 1, integrated numerically, is π within 1 part in 10⁴, so Ω = 1/π.
   Expected to pass: it is a property of the Bernoulli family, known long before CRR.
C2 Recombination as one Z₂ occasion (carrier: one hydrogen atom, ionised or not; path: Saha).
   a. C*Ω = 1/2 falls at the same redshift as x = 1/2, within 1 part in 10⁶ (it must: asin sqrt(1/2) = π/4). Expected to pass, trivially.
   b. Does the occasion reach the cut, C*Ω = 1? My bet: NO. With a residual ionisation of 2.4 × 10⁻⁴ it stops at
      0.985 to 0.995. If so the page says the cut is not reached.
   c. Nothing here is a prediction: the path is the Saha equation relabelled. Verdict fixed in advance: INCONCLUSIVE at best.
C3 Waypoint spacing against the candidate spread 1/2π = 0.159. My bet: FAIL. The coefficient of variation of the 27 steps
   in ln(frame width) will lie between 0.4 and 1.2. The waypoints are my choices, so even a pass would show nothing.
C4 Ω_m against 1/π. Planck's 0.3153 ± 0.0073 is within 1σ of 0.3183 (I expect 0.4σ). DESI DR2's 0.2975 ± 0.0086 is more
   than 2σ away. Ω_m(z) equals 1/π at one epoch only, within |z| < 0.05 of today. Verdict fixed in advance: a coincidence of
   the present epoch, INCONCLUSIVE, and not a claim the programme makes.
C5 Compton–Schwarzschild crossing against π. The crossing is at sqrt(π) Planck masses with h and at 1/sqrt(2) with ħ.
   The π is the 2π between h and ħ. Verdict fixed in advance: not a CRR signature (FAIL as a signature).
C6 The site's own claim (darkenergy.html, claims/P4): the dark-energy density peaks where w crosses −1. With DESI DR2's
   w0, wa: the two redshifts agree within 10⁻⁶. They will also agree, within 10⁻³, for an arbitrary second model
   w(a) = −1.3 + 0.9a², because dρ/d ln a = −3(1 + w)ρ for any fluid. So PASS, and the page must say it is an identity
   of the continuity equation and not evidence for CRR. Direction: w > −1 today and w < −1 in the past (the site's F2
   test is not triggered).
C7 Crossovers of cosmic history as events. Three events, two intervals. Verdict fixed in advance: too few to test.

## The page

G1 Desktop 1280×800 and phone 390×844: no page errors, no console errors.
G2 At 390 px the document is no wider than the window.
G3 The page does not bake every picture at the start: 3 s after load no more than 4 levels hold baked pictures, and
   after flying from the Planck length to the horizon no more than 6 do.
G4 Every one of the 28 waypoints can be reached by its menu entry, and the title shown matches the model's name for it.
G5 The frame width, the lookback time and the Voyager distance on screen equal the model's values at the moment of reading.
G6 With reduced motion requested the page starts paused and still shows a picture.
G7 Keyboard: arrow keys change the zoom; Page Up and Page Down move between waypoints.
G8 First picture on screen within 4 s of load in headless Chromium at 1280×800 (the earlier seven-level page took 33 s to
   finish its bakes).

---

# Second set of bets, 25 November 2026 (later the same day): "Time on this page: a CRR reading"

Written before any of the new model code (section 10 of model.js) was typed or run. Nothing above is deleted or changed.
The page's seven CRR checks (C1 to C7) stay on record above and in RESULTS.md; on the page they shrink to a folded note,
and C1 and C2 are reused below.

## Honest note on what was known beforehand

Most of the numbers quoted in these bets were known before the bets were written. They were worked out, and partly
computed in a scratch script, by the person who set this task, and were handed over with it. So for those numbers a
pass shows only that the page's model reproduces the scratch script by its own route; it is a check of arithmetic, not a
prediction. Each bet says which kind it is:

- **(known)** the number was given beforehand from the scratch script;
- **(hand)** I worked it out by hand, from formulas, while writing the bet;
- **(open)** no value was known when the bet was written.

Definitions are those of README.md (lines 14 to 70) as it stands today: C is accumulated Fisher–Rao arc length since the
last cut, in units of σ; C* is the geodesic distance from where the occasion began; Ω = 1/(the manifold's diameter) = 1/π;
the cut comes the first time C*Ω = 1; the surplus is S = C − C* ≥ 0. Lengths below are in half-turns (units of π) with σ = 1 rad
unless a bet says otherwise.

All of T1 to T11 are exact identities or standard physics relabelled. None can count as evidence for CRR whatever
the result. Verdict fixed in advance for the whole set: a reading, NOT A TEST of CRR.

## Physics and geometry

T1 Energy and time. In the radiation era t = sqrt(90/(32π³ g*)) ħ E_P/(kT)², which is the textbook
   t ≈ 2.42 g*^(−1/2) (T/MeV)^(−2) s. The page takes kT = E for a probe of energy E = hc/λ (a convention: the mean energy
   of a particle is about three times kT, which would move every time by a factor of about nine).
   a. With g* = 1 and kT = 1 MeV the constants give 2.42 s within 0.5%. (known: it is the textbook coefficient)
   b. With a step table for g*: 13.6 TeV → 1.2 to 1.4 × 10⁻¹⁵ s; 35.8 TeV → 1.7 to 1.9 × 10⁻¹⁶ s; 1 GeV → 2.8 to
      3.3 × 10⁻⁷ s; 1 MeV → 0.70 to 0.78 s. (known)
   c. At the Planck energy the formula gives between 0.01 and 0.1 Planck times. It is not valid there. (hand: 0.029)
   d. Where both can be used they agree: at kT = 10 keV the radiation formula with the late value of g* and the page's
      own Friedmann integral give the same time within 1%. (open)
   e. The two branches meet where kT₀ = hc/λ for today's sky, at a wavelength between 5.2 and 5.4 mm, not at 1.7 m.
      (hand: 5.3 mm. The task's wording puts "now" at the reader; if this passes the page must say 5 mm.)
T2 The geometric mean of the Planck length and the horizon radius is between 80 and 88 µm. (known: about 84 µm)
T3 The contents of the universe as one path on an octant of a sphere of radius 2.
   a. The three pure states are π apart, each from each, within 10⁻¹²; for two outcomes the three-outcome distance equals
      the Bernoulli distance already on the page within 10⁻¹²; the chord form 4 asin(|√p − √q|/2) and the form
      2 acos(Σ√(p_i q_i)) agree within 10⁻⁹ at separations above 0.1. (hand)
   b. Whole history, radiation to dark energy: C = 1.949 ± 0.002 half-turns for C* = 1, so S = 0.949 ± 0.002. (known)
   c. Now: C = 1.569 ± 0.002, C*Ω = 0.994 ± 0.001, S = 0.575 ± 0.002; 80.5% ± 0.2 of the whole path is walked. (known)
   d. Matter's share peaks at 99.52% ± 0.02 at z = 11.2 ± 0.1 (known); the numerical peak is at a⁴ = Ω_r/(3Ω_Λ) within
      10⁻⁶ in a (hand).
   e. An independent integration in the test (a polyline of chords on a different grid in a different variable) agrees
      with the model's C(now) and C(whole) within 10⁻⁵ half-turns. (open)
   f. C*Ω is below 1 at every finite a tried, up to a = 10⁶, and rises throughout; 1 − C*Ω falls by a factor of
      100 ± 1 for each tenfold growth at late times. So the cut is reached only as a → ∞. (hand)
   g. The pattern is generic. A made-up succession with no connection to our universe (a stiff fluid, w = 1, then
      matter, then a fluid with w = −1/3, with amplitudes 10⁻⁶, 1 and 10⁻³) gives C between 1.9 and 2.0 half-turns for
      C* = 1. A succession whose middle stage never dominates (the real three fluids with equal shares today) gives
      C between 1.0 and 1.5. (open)
T4 Log-odds. d ln(ρ₁/ρ₂)/d ln a = 3(w₂ − w₁): +1 for matter over radiation and +3 for dark energy over matter, by finite
   difference at a = 10⁻⁵, 10⁻², 1 and 10, within 10⁻⁶. For DESI's w(a) today the rate for dark energy over matter is
   −3 w(1) = 2.256 within 10⁻⁶. (hand. The task wrote 3(w₁ − w₂); the sign is a matter of which is over which.)
T5 On the matter-to-dark-energy leg alone (no radiation), with φ = 2 asin sqrt(Ω_Λ(a)): dφ/d ln a = (3/2) sin φ by central
   difference at nine values of a from 0.05 to 20, within 10⁻⁸. It is fastest at Ω_Λ(a) = ½, z = 0.295 ± 0.001, which is
   0.24 to 0.28 of an e-fold ago. (known)
T6 The Gudermannian relation, t = t_Λ gd⁻¹(φ/2), that is C*Ω = (2/π) gd(t/t_Λ) on that leg.
   a. t_Λ = 2/(3 H₀ sqrt(Ω_Λ)) = 11.70 ± 0.01 billion years. (known)
   b. Against a direct numerical integration of the Friedmann equation without radiation, done in the test and not
      in the model, at a = 0.1, 0.3, 0.5, 1, 2, 5 and 20 and at the equality: relative difference below 10⁻¹². (open)
   c. Slowing turns to speeding at C*Ω = 0.392 ± 0.001, t = 7.70 ± 0.02; equality at exactly ½ (within 10⁻¹²),
      t = t_Λ ln(1 + √2) = 10.31 ± 0.02; now 0.620 ± 0.001. (known)
   d. With radiation put back (the page's full cosmology), using the true cosmic time in the formula, the error in C*Ω is
      below 10⁻³ at every a from 0.01 to 10. The value at a = 1 is to be reported. (open; hand estimate 5 × 10⁻⁴)
T7 Resolution and the cut. With a resolution σ (radians of Fisher angle) the cut on that leg comes when φ ≥ π − σ.
   a. t_cut = t_Λ gd⁻¹(π/2 − σ/2): 43 ± 0.5 billion years for σ = 0.1; 70 ± 0.5 for 0.01; 178 ± 1 for 10⁻⁶. (known)
   b. t_Λ ln(4/σ) is within 0.1% of it for every σ ≤ 0.1. (hand)
   c. Each factor of e in σ adds t_Λ: t_cut(σ/e) − t_cut(σ) = t_Λ within 0.1% for σ ≤ 0.1. (known)
   d. At t_cut the angle found from a(t) ∝ sinh^(2/3)(t/t_Λ) is π − σ within 10⁻⁹. (hand)
   e. Recombination stopped at C*Ω = 0.990 ± 0.001 (bet C2 above). It counts as complete only for σ ≥ 2 asin sqrt(x_res):
      0.0200, 0.0310 and 0.0447 rad (each ± 0.0005) for a residual ionisation of 1 × 10⁻⁴, 2.4 × 10⁻⁴ and 5 × 10⁻⁴. (known for
      the middle one; hand for the others)
   f. A consequence nobody asked for: on the three-way path of T3, which began at pure radiation, every state without
      radiation is "opposite". With σ = 0.031 that occasion has therefore already reached its cut, at a redshift
      between 0.3 and 0.6. (hand: about 0.43. If it passes the page says so.)
T8 Conformal time: 46.14 ± 0.1 billion light-years behind us and 16.69 ± 0.1 ahead, so 73.4% ± 0.3 has passed. (known)
T9 Growth of structure, from the linear growth equation integrated through radiation, matter and dark energy.
   a. f = d ln D/d ln a differs from Ω_m(a)^0.55 by less than 0.01 everywhere from z = 5 to z = 0. (open)
   b. D(∞)/D(now) by the 0.55 rule is 1.44 ± 0.02 (known). By the equation itself it is between 1.33 and 1.45 (open; I
      expect a little under the rule's value, because the true f falls faster than Ω_m^0.55 once matter is rare).
   c. The equation's D(∞)/D(now) agrees with the closed integral D ∝ H ∫ da/(aH)³ for matter and dark energy alone
      within 2 parts in 10³. (open)
   d. f today is between 0.51 and 0.54. Growth freezes: D at a = 10 is more than 99% of D(∞). (hand)
T10 Hydrogen, there and back. Saha down to a residual 2.4 × 10⁻⁴, then reionisation as a tanh step at z = 7.67 of width
    0.5 in redshift: C = 1.98 ± 0.005 half-turns, C* at the end below 0.001, so S = 1.98 ± 0.005 and e^(SΩ) = 7.2 ± 0.1;
    the largest C*Ω on the way is 0.990 ± 0.001. (known) An illustration of the salience proposal, not a test of it.
T11 If DESI's hinted w₀, w_a and Ω_m are taken literally and extended into the future (which the parametrisation was not
    made for): dark energy's share peaks between 82% and 86% when the universe is 1.8 to 1.95 times its present size,
    is below 1% again by 8 times, and on the two-state leg C*Ω never exceeds 0.736 ± 0.01. (hand)

## The page (added to G1 to G9)

G10 Every new control (the time slider, the resolution slider, the buttons of the new section and its fold-outs) can be
    reached with the Tab key, and arrow keys on the two sliders change the numbers shown.
G11 At 390 px every new control is at least 44 px tall and, with every fold-out of the new section open, the document
    is no wider than the window.
G12 No page error and no console error after every new control has been used, at 1280×800 and at 390×844.
G13 With reduced motion requested the new figures are drawn and nothing in the new section moves by itself.
G14 Numbers shown equal the model's: C, C* and S at "now" in the octant read-out, and the cut time at σ = 0.01.
G15 At the first-galaxies stop, with "follow the zoom" on, the marker's redshift equals the redshift of the frame's edge.
W7  Figures written in the prose of the new section (1.949, 0.949, 80.5%, 11.7, 84 µm, 73.4%, 7.2 and the rest) equal
    what the model computes, to the digits written.
T12 (added while the model was being typed, before it was run) The Fisher–Rao speed of the contents' path, per e-fold, is
    the standard deviation over the contents of the exponent n = 3(1 + w) by which each thins out, and its square equals
    −d n̄/d ln a = −3 dw_eff/d ln a = −2 dq/d ln a, with q the deceleration parameter. Checked by finite difference of q from
    the Friedmann equation at a = 10⁻⁴, 10⁻², 0.5, 1 and 3, within 10⁻⁶. (hand)
