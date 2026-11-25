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
