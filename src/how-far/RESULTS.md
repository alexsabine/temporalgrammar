# Results for how-far-does-it-go.html

Bets are in BETS.md, written on 25 November 2026 before model.js was first executed. This file records what came
out, what failed, and every change made after a result was seen. Earlier outputs are kept beside the final ones.

## Honest note on what was known beforehand

Before the test file existed, model.js was run once through a scratch script that printed its main numbers (age,
horizon, the Saha redshift, the CRR table, the level table). BETS.md was written before that run. The test's thresholds
are those of BETS.md, not of the printout, but the printout had been seen by the time test_model.js was typed.
The values of the physical constants and of Planck's parameters were known from the sources before anything was run,
so bets such as "age 13.797 ± 0.03" are checks of the integrator, not predictions.

## Model checks (test_model.js)

| Run | File | Result |
|---|---|---|
| 1 | test_output_v1.txt | 107 of 110; three failed |
| 2 | test_output_v2.txt | 109 of 110; one failed |
| 3 | test_output.txt | 110 of 110 |

### Failures on the first run

1. **C1, the Z₂ diameter by numerical integration.** Got 3.1410734, low by 1.65 parts in 10⁴; the bet was 1 part
   in 10⁴. Cause: a plain midpoint rule in p, stopped 10⁻⁶ short of p = 0 and p = 1, where the integrand is infinite
   but integrable. The integrator was changed (substitution p = u² in each half, which removes the square root). It now
   gives π to 1.3 parts in 10¹⁰. The bet was not changed. The first version is noted in a comment in the model.
2. **Z4, units.** Lengths between 6,324 and 12,650 AU were printed in AU with five digits, and lengths between 0.05
   and 0.1 AU as "0.09 AU". The model was changed: light-years begin at 0.1 ly and AU at 0.1 AU. After that one failure
   remained (run 2): kilometres run to 946,000 km before "million km" takes over. **The bet was changed here**, because
   it was wrong for kilometres: distances such as the Moon's are written 384,400 km, and a five- or six-digit mantissa
   in km is how such distances are written. Z4 now allows kilometres up to 1,000,000 and keeps 10,000 for every other unit.
3. **P8h, a check I added while writing the test (it is not in BETS.md).** It compared the integrator with the
   closed form for a universe of matter alone. There the integrand in the page's variable jumps at a = 0 and Simpson's rule
   loses 8 parts in 10⁵. The fault was in the check, not in the page's use of the integrator, where radiation makes the
   integrand smooth. The check was replaced by matter plus radiation without dark energy, which also has closed forms;
   that agrees to 10⁻¹⁰.

### What the bets found (final run)

- Planck units from ħ, G, c: within 1.6 parts in 10⁷ of the CODATA table.
- Compton = Schwarzschild at 1.7724539 Planck masses (√π), common length 3.5449 Planck lengths (2√π).
- Bohr radius to 1 part in 10¹¹. R(³⁹K) = 4.069 fm, R(¹⁹⁷Au) = 6.982 fm with r₀ = 1.2 fm.
- 5 MeV alpha on gold stops at 45.5 fm.
- hc/E at 13.6 TeV = 9.116 × 10⁻²⁰ m. ħc/Λ at 35.8 TeV = 5.512 × 10⁻²¹ m (the first hard line).
- Bending radius at 7 TeV, 8.33 T: 2,803 m. Ring for the Planck energy: 2,457 light-years (8.7 × 10¹⁴ LHCs).
- Friedmann integration with Planck 2018: age 13.795 Gyr (Planck: 13.797 ± 0.023); horizon 46.14 Gly = 14.15 Gpc
  = 4.365 × 10²⁶ m; equality z = 3418 (Planck: 3402 ± 26; the page treats neutrinos as massless);
  distance to last scattering 13.867 Gpc; sound horizon 144.27 Mpc (Planck: 144.43); 100θ = 1.04034 (Planck: 1.04110).
  With H₀ = 73.04: age 12.72 Gyr, horizon 42.6 Gly.
- Saha: hydrogen half ionised at z = 1370, T = 3,736 K.
- Light: Moon 1.282 s; Sun 499.0 s; Neptune 4.174 h; Voyager 1 from the Sun 23.90 h (25 Nov 2026); Proxima 4.2465 yr.
- Sgr A* shadow 53.3 µas against a measured ring of 51.8 ± 2.3 µas.
- MoM-z14: universe 282 Myr old; lookback 13.51 Gyr; now 33.8 Gly away.
- Inflation: H < 4.7 × 10¹³ GeV, so c/H > 4.2 × 10⁻³⁰ m.
- CMB sky to ℓ = 200: worst band 9% from the input spectrum; map variance within 1% of Σ(2ℓ+1)C_ℓ/4π.
- 28 waypoints over 61.7 powers of ten of frame width; round trip z → width → z exact to 10⁻¹².
- The brain model's own 60 or so checks (sections 3 to 8 of its test) pass unchanged.

### CRR checks

| # | Question | Result | Numbers |
|---|---|---|---|
| C1 | Z₂ diameter is π, so Ω = 1/π | PASS (established mathematics, not CRR's) | 3.1415927 |
| C2 | Recombination as one Z₂ occasion | INCONCLUSIVE, as fixed in advance | C*Ω = ½ at z = 1369.75, exactly where x = ½; the cut is not reached: C*Ω stops at 0.9901 |
| C3 | Waypoint spacing has spread 1/2π = 0.159 | FAIL, as bet | spread 0.913 over 27 steps |
| C4 | Ω_m = 1/π | INCONCLUSIVE, as fixed in advance | Planck 0.41σ away, DESI DR2 2.42σ away; equal only at z = 0.005 |
| C5 | √π at the Compton–Schwarzschild crossing is a CRR signature | FAIL | √π with h, 1/√2 with ħ |
| C6 | The site's claim: dark-energy density peaks where w = −1 | PASS, as an identity | z = 0.405229 both ways for DESI DR2; also equal for an unrelated model |
| C7 | Cosmic crossovers as events | NOT TESTABLE | three events, two gaps |

The page reports these in the same words. Nothing in them is evidence for CRR.

## Page checks (test_page.js)

Final run (test_output_page.txt): 14 of 14. First picture on the stage 0.7 s after opening; three levels held pictures
at 3 s; never more than 5 of 28 at once; longest wait for one stop's sharp pictures 20 s in headless software drawing
(a coarse preview is shown first). The first run (test_output_page_v1_harness_fault.txt) was 12 of 14. G7 failed through
a fault in the test, which pressed Page Up twice and expected to be back where it began. G2b, a check added while writing
the test, failed truly: the two fold-out headings were 32 px tall, under the 44 px the site asks for. The page was changed.

## Changes made after looking at pictures (none touches a bet)

- The frame of the Earth–Moon stop was narrowed from 10⁹ m to 6.4 × 10⁸ m, and that of the Local Group from
  9.5 × 10²² to 6.6 × 10²² m, because the bodies were too small to read. The microwave-sky stop was moved from
  9.8 × 10²⁶ to 8.5 × 10²⁶ m so that it sits inside the "can be seen" line, where it belongs, and the first-galaxies
  stop from 6.4 to 5.6 × 10²⁶ m. C3's spread changed from 0.909 to 0.913 as a result.
- Many changes to the drawing only.

## Not verified, or verified only in part

- GRAVITY (2022): the mass, 4.30 × 10⁶ Suns, is in the abstract. The distance used, 8,277 pc, is from memory.
- LHC ring length, dipole field and bending radius: from memory of the Design Report; the DOI resolves; the three
  figures agree with one another through ρ = E/(ecB) to 0.03%.
- α Centauri, Procyon, EZ Aquarii and Luhman 16: parallaxes from memory (they are not in the Gaia table used).
- Mukhanov & Chibisov (1981) and Oort (1950): cited from memory, no DOI.
- Planetary mean longitudes (Standish's approximate elements): from memory. Positions are good to a few degrees at best.
- Milky Way star count and disc diameter: round figures in common use, not taken from one paper.
- Slater's rules for the potassium ion's shells: from memory.
- The residual ionisation after recombination (2.4 × 10⁻⁴) is an order-of-magnitude setting.
- Coordinates of clusters in the Laniakea picture: rough, from memory of the published figure.
- The brain's sources were verified when the earlier seven-level page was made, not again here.

## Verified on 25 November 2026

CODATA 2022 values (physics.nist.gov); Planck 2018 table 2 (arXiv:1807.06209 text); DESI DR2 II (arXiv:2503.14738,
numbers read from the paper's HTML); Riess et al. 2022; Naidu et al. 2025 (z = 14.44) and Fontana et al. 2026
(arXiv:2610.12121: "galaxies up to z = 14.44, the redshift of the most distant galaxy currently known"); Carniani et
al. 2024; EHT 2022 (51.8 ± 2.3 µas); ZEUS 2016 (0.43 × 10⁻¹⁶ cm); CMS 2018 (12.8, 17.5 TeV); ATLAS 2020 (35.8 TeV);
Voyager ranges from JPL Horizons for 2026-Nov-25 (V1 172.453 AU from the Sun, 173.236 AU and 1.0005 light-days from
the Earth; V2 144.564 AU), and NASA's statement that Voyager 1 reached one light-day on 18 November 2026; Gaia EDR3
parallaxes of 23 nearby stars from the Gaia Catalogue of Nearby Stars (VizieR J/A+A/649/A6). All DOIs in the page's
list were resolved through Crossref or DataCite.
