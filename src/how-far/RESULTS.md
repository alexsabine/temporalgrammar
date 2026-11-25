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

---

# Second set, 25 November 2026 (later the same day): "Time on this page: a CRR reading"

The page's CRR section was replaced by a new one about time. Nothing above is deleted. The seven checks C1 to C7 keep
their record above; on the page they are now a folded note, "Earlier checks, kept on record", one line each with the
same results and the same live numbers, and a link to this file. C1 and C2 are used again in the new section.

The bets for the new section (T1 to T12, G10 to G15, W7) are the second set in BETS.md. They were written before
section 10 of model.js was typed or run. T12 was added while the model was being typed and before it was run.

## Honest note on what was known beforehand

Most of the numbers were known before the bets were written: they came with the task, worked out and partly computed
in a scratch script by the person who set it. Each bet is marked (known), (hand) or (open) in BETS.md. For the (known)
ones a pass shows only that the page's model reproduces the scratch script by its own route. No value was known in
advance for T1d, T3e, T3g, T6b, T6d, T9a, T9c and the equation's value in T9b.

Verdict fixed in advance for the whole set, and unchanged: these are exact identities and standard physics read in
CRR's words. **Nothing in them is evidence for CRR and nothing in them tests it.**

## Runs

| Run | File | Result |
|---|---|---|
| Model, first run of the new bets | test_output_v4_first_run_of_time_bets.txt | 143 of 143 (33 new) |
| Model, final (two checks of the page's words added, W7 and W8) | test_output.txt | 145 of 145 |
| Page, first run with the new section | test_output_page_v2_time_harness_faults.txt | 18 of 20; two failed |
| Page, final | test_output_page.txt | 20 of 20 |

No bet failed in the model. No bet was changed. The two page failures were both faults in the test, not in the page:
G12a expected "Play the history" to begin again from the start, but it goes on from where the marker is; G13 expected a
second press of "Next moment" to go later still, but after the far future the next moment is the beginning. The tests
were corrected and the first output is kept.

One check is weaker than its wording suggests. T7d (the angle at the cut is π − σ) holds to 2 × 10⁻¹³ for σ from 0.5 to
0.001, as bet. For σ near 10⁻⁸ the same comparison loses all its figures, because π − φ is then a difference of nearly
equal numbers in double precision. The page never does that subtraction: it uses 2 asin sqrt(share) for what is left.

## What came out

| Bet | Kind | Result |
|---|---|---|
| T1a coefficient | known | 2.4203 s |
| T1b energies | known | 13.6 TeV → 1.266 × 10⁻¹⁵ s; 35.8 TeV → 1.828 × 10⁻¹⁶ s; 1 GeV → 3.080 × 10⁻⁷ s; 1 MeV → 0.738 s (g* = 106.75, 106.75, 61.75, 10.75) |
| T1c Planck energy | hand | 0.0291 Planck times; the formula has no meaning there |
| T1d two methods at 10 keV | open | agree to 2.7 parts in 10⁵ |
| T1e where "now" is on the way in | hand | 5.279 mm, not 1.7 m |
| T2 geometric mean | known | 84.00 µm |
| T3a corners, reductions | hand | π to 12 places; worst difference of the two forms 4 × 10⁻¹⁵ |
| T3b whole path | known | C = 1.94876 half-turns, S = 0.94876 |
| T3c now | known | C = 1.56913, C*Ω = 0.99389, S = 0.57524, 80.52% walked |
| T3d matter's peak | known | 99.526% at z = 11.217; numerical and formula agree to 10⁻⁸ in a |
| T3e independent integration | open | agrees to 10⁻⁸ half-turns |
| T3f the cut only at infinity | hand | 1 − C*Ω falls exactly 100-fold per tenfold growth |
| T3g generic | open | made-up universe 1.987 half-turns (middle stage peaks at 99.97%); equal thirds 1.335 (peaks at 36%) |
| T4 log-odds | hand | +1 and +3 to 7 × 10⁻¹²; DESI today 2.256 |
| T5 dφ/d ln a = (3/2) sin φ | known | to 4 × 10⁻¹⁰; fastest at z = 0.29497, 0.2585 e-folds ago |
| T6a t_Λ | known | 11.695 billion years |
| T6b Gudermannian against direct integration | open | worst relative difference 7 × 10⁻¹⁵ at eight values of a |
| T6c values | known | 0.39183 at 7.701; ½ at 10.308; now 0.62044 (age without radiation 13.801) |
| T6d with radiation put back | open | worst error in C*Ω 2.2 × 10⁻⁴, at a = 0.5; at a = 1 it is −1.5 × 10⁻⁴ |
| T7a cut times | known | 43.14, 70.07, 177.79 billion years |
| T7b, c approximation, t_Λ per factor e | hand, known | within 6 × 10⁻⁵ and 2 × 10⁻⁴ |
| T7e recombination | known, hand | complete only for σ ≥ 0.0200, 0.0310, 0.0447 at residuals 1 × 10⁻⁴, 2.4 × 10⁻⁴, 5 × 10⁻⁴ |
| T7f the three-way path at σ = 0.031 | hand | its cut is in the past: z = 0.431, 9.11 billion years in |
| T8 conformal time | known | 46.142 behind, 16.692 ahead, 73.44% |
| T9a growth rule | open | worst difference from Ω_m^0.55 since z = 5 is 0.0027, today (f = 0.5274, rule 0.5300) |
| T9b D(∞)/D(now) | known, open | 1.442 by the rule; **1.4085 by the equation** |
| T9c against the closed integral | open | agree to 8 parts in 10⁶ |
| T9d freezing | hand | D(a = 10) is 99.5% of D(∞) |
| T10 hydrogen | known | C = 1.9804, C* at the end 10⁻⁵, S = 1.9804, e^(SΩ) = 7.25, largest C*Ω 0.9901 |
| T11 DESI run forward | hand | share peaks at 83.8% at a = 1.874; below 1% from a = 6.7; C*Ω on the leg never passes 0.736 |
| T12 speed of the path | hand | L² = −2 dq/d ln a to 3 × 10⁻⁷ |

## Where the task's numbers were corrected or sharpened

- **Where "now" is.** The task put "now" on the gauge at the reader, 1.7 m. On the way in, the frame that matches
  today's sky (kT₀ = hc/λ) is 5.28 mm wide. Wider frames, the whole brain among them, match a sky colder than today's
  and so belong to the future; 1.7 m matches the sky 114 billion years after the beginning. The page says 5.3 mm.
- **Matter's largest share** is 99.526%, which the page writes as 99.53% (the task: 99.52%).
- **Growth still to come** is a factor 1.41 by the growth equation; the task's 1.44 is the 0.55 rule's value, which the
  page also reports.
- **The log-odds rule** is written d ln(ρ₁/ρ₂)/d ln a = 3(w₂ − w₁). The task wrote 3(w₁ − w₂), which is the same with the
  two swapped.
- **The fog cleared at 371,000 years** in this page's model (Planck's z* = 1089.92); the page says 370,000 where the
  task said 380,000. **Reionisation**: Planck 2018 gives z = 7.67 ± 0.73 (the task: about 7.7).
- **C*Ω = 0.994 now** was in the task and is right, and the page spells out what it means: on the three-way path the
  opposite of "all radiation" is any state with no radiation, matter included. C*Ω was already 0.962 when matter peaked.

## Things the work turned up that were not asked for

- With σ = 0.031, the resolution at which recombination counts as a cut, the three-way path's own cut is already past
  (T7f). The page shows this in the resolution panel.
- Hydrogen's round trip reads two ways. At fine resolution it is one unfinished occasion that is almost all surplus.
  At σ ≥ 0.031 it is two completed occasions with none. The page says the framework does not yet choose.
- The Fisher–Rao speed of the contents' path per e-fold is the standard deviation of 3(1 + w) over the contents, and
  its square is −2 dq/d ln a (T12). Elementary; not searched for in the literature.

## The Gudermannian relation: what a brief search found

Two web searches on 25 November 2026. The pieces are standard: a(t) ∝ sinh^(2/3)(t/t_Λ) is in textbooks (Ryden 2016)
and in Boylan-Kolchin & Weisz (2021, arXiv:2103.15825); the density written with tanh²(t/t_Λ) is in Chavanis (2015,
arXiv:1504.08355). The Gudermannian appears in cosmology in a different, non-standard model of dark energy (Lazkoz et
al. 2023, arXiv:2311.10526). The statement C*Ω = (2/π) gd(t/t_Λ), or any link between the Gudermannian and a
Fisher–Rao angle of the density shares, was not found. Two searches do not show it is new, and the page says so.

## Settings and simplifications in the new section

- "As hot as" means kT = E = hc/λ. With the mean energy, about 3kT, every time on the inward branch moves about ninefold.
- The length probed is taken to be the width of the frame.
- g* is a step function at each particle's mass, from the table in the Review of Particle Physics; the quark–hadron
  transition is put at 155 MeV; below the electron mass 3.38 is used (neutrinos cooler than photons). Particle masses
  are from memory.
- Treating the three shares as a probability distribution is a modelling choice.
- Growth: linear, matter only, radiation smooth, baryons treated as free to clump throughout.
- Reionisation: Planck's tanh step, width 0.5 in redshift; helium left out; hydrogen taken back to fully ionised.
- DESI's w(a) run to a = 40 is far outside what it was fitted to. It is there to show the direction of the change.

## Citations added, and how each was checked (25 November 2026)

Resolved through Crossref or the arXiv listing, with title and authors matching: Linder 2005; Crooks 2007; Ellis 2006;
Navas et al. 2024; Kolb & Turner (the CRC reissue's DOI); Ryden 2016; Amari & Nagaoka 2000; Aaronson, Carroll &
Ouellette 2014; Chavanis 2015; Lazkoz et al. 2023.

Read in the source itself: the Review of Particle Physics, section 22 (eq. 22.43–22.44, written there with 2.4, and
the table of degrees of freedom); Planck 2018 VI (z_re = 7.67 ± 0.73 in table 2, and the tanh step with Δz = 0.5);
Linder 2005 (γ = 0.55 + 0.05[1 + w(z = 1)] in the text; it is not in the abstract); Primack & Abrams 2006, chapter 6
("Adapting an idea of Sheldon Glashow", with a note citing Glashow & Bova, Interactions, 1988, chapter 14).

**Not verified, or verified only in part:**

- Glashow & Bova (1988): not seen; cited through Primack & Abrams.
- Kolb & Turner (1990): the DOI resolves; the coefficient 2.42 is quoted from memory of the book and agrees with the
  constants (2.4203) and with the Review's 2.4.
- Ryden (2016): the DOI resolves; the sinh^(2/3) solution is quoted from memory of the book, and was checked here against
  direct integration (T6b).
- Aaronson, Carroll & Ouellette: the abstract was read (entropy rises, complexity rises then falls, the coffee
  automaton); the body was not.
- Chavanis (2015) and Lazkoz et al. (2023): known only from search summaries of their formulas; the papers exist with
  those titles and authors. They are cited only for what the summaries state.
- Albert (2000), Penrose (1989), Broad (1923), Whitehead (1929): books, no DOI, cited from memory. The Whitehead
  sentence "the many become one, and are increased by one" is quoted from memory.
- Crooks (2007): title and DOI verified; "bounds dissipation" is from memory of the abstract.
- Amari & Nagaoka: Crossref dates the AMS edition 2007; the page gives 2000, the year of the translation, from memory.

## Changes made after looking at pictures (none touches a bet)

Labels in the octant figure were moved outside it on wide screens and stacked on narrow ones; the enlarged corner was
moved; labels in the loop figure were shortened at 390 px; a time-scale label that ran into the stops was dropped.
Screenshots were taken at 1280×800 and 390×844. The page has one theme.
