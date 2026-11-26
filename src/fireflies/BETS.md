# Bets for the fireflies, written before the model was run

Written before `model.js` had been run once. Each bet has a threshold. The test script is `test_model.js`; its outputs are kept in this folder, failures included. Settings are not to be changed to pass a bet without recording the earlier failure here.

Units are metres and seconds. "Burst" means a run of flashes by one firefly about half a second apart. A "collective burst" is a run of flashes by the whole swarm with no gap longer than 1.5 s. Its start is its first flash.

The default world: 300 fireflies in a clearing 40 m wide and 40 m deep, flying between 0.3 and 4 m up. A firefly sees another's flash with probability exp(−d/6 m).

**B1. A lone firefly has no steady rhythm.**
One firefly, 4,000 s, its own burst starts. Pass if the intervals between them have CV above 0.4.
Background: isolated *Photinus carolinus* in a tent flashed at irregular, widely spread intervals (Sarfati et al. 2023, cited from memory; check).

**B2. Together they keep time, and faster.**
Default world, 600 s after a 120 s warm-up, five seeds. Pass if, in at least 4 of 5 seeds, the mean interval between collective burst starts is at least 40% shorter than the lone firefly's mean (B1) and its CV is under a third of the lone firefly's CV.
This is the main bet: the group's period is not written into any one firefly.

**B3. It takes a crowd.**
Same clearing, 15 fireflies instead of 300, five seeds. Pass if the mean share of the swarm flashing in a collective burst is under 0.5 at 15 and over 0.8 at 300.
Background: synchrony in the wild appeared only above a density (Sarfati, Hayes and Peleg 2021, cited from memory; check).

**B4. Flashes inside a burst line up.**
Default world. Group every flash into peaks (gap > 0.2 s starts a new peak). Pass if the median spread (standard deviation) of flash times within a peak of at least 20 flashes is under 0.08 s with flash-to-flash nudging on, and over 0.12 s with it off (control: the bursts still start together, but each firefly then keeps its own flash clock).

**B5. Hidden from one another, they go back to being alone.**
Default world with seeing switched off. Pass if the CV of each firefly's own burst intervals, pooled, is above 0.4.

**B6. A white torch silences the fireflies it lights.** (A check of the code, not a discovery.)
Default world, a beam covering about a third of the clearing for 120 s after warm-up. Pass if flashes per firefly per minute inside the beam fall by at least 80% against the same region before the torch.

**B7. Reported, not judged.**
The CV of the collective interval in B2 is reported next to the candidate value from the CRR programme, Ω/2 = 1/(2π) ≈ 0.159. No pass mark. My guess before running: it will be smaller, about 0.05 to 0.1, because it is set by how widely the fireflies' recovery times are spread, which is a setting. Nothing will be tuned toward 0.159.

---

# Log

**v1 (`test_output_v1.txt`)** B1 pass (CV 0.68). **B2 FAIL**: the swarm of 300 never went dark for 1.5 s, so the whole run was one collective burst. **B3 FAIL**: share 0.93 at 15. **B4 FAIL** (no peaks to measure). B5 pass. **B6 FAIL**: 68.6% fall.

Faults found, and what was changed:
- B6 was a fault of the test harness, not the model: it decided whether a flash was in the beam by where the firefly was at the end of the 120 s window, and fireflies drift 50 m in that time. v2 decides it at the moment of the flash.
- Sight was redrawn for every flash, so over a burst of six flashes nearly every pair saw each other, however far apart. A leaf in the way should stay in the way. v2: whether i sees j is fixed for 4 s at a time.
- Bursts rolled on: fireflies that recovered while others were still flashing joined in and kept the light going. v2 added a rule, "hold": a flash seen while recovering starts the count again, so the count is of dark time.

**v2 (`test_output_v2.txt`)** B1 pass. **B2 pass 5/5** (13.3–13.9 s, CV 0.07–0.18). **B3 FAIL**: share 0.55 at 15 and only 0.26 at 300. **B4 FAIL**: spread 0.74 s on, 1.07 s off. B5 pass. B6 pass (100%).

Then a set of short explorations (seed 11, 420 s, not kept as files; the numbers below are copied from the console). Two causes were found:
- Joiners flashed 0.22 s after the flash they saw, nearly half a beat out of step, so peaks smeared. Joining on the next beat (0.52 s) brought the spread to about 0.05 s in every variant tried.
- With "hold", recovery times spread with sd 0.9 s meant that fireflies with longer recovery were reset by every burst and shut out: a quarter took part. Without "hold" the swarm never went dark (tried: no hold; a ready firefly answering only a flash that follows darkness; unprompted starts only after 2 s dark; counting recovery from the start of one's own burst; Mirollo–Strogatz style advances of the count by 0.3 or 0.6 s a beat; sight distance 12 and 20 m). Even started all together, bursts lengthened cycle on cycle, because fireflies make 4 to 8 flashes, finish at different moments, recover at different moments and join the next burst late.
- One exploration had a bug of mine: the limiter meant to allow one advance per beat used the time of the last flash seen, which in a dense swarm is never more than a few milliseconds ago, so the advance never happened. Fixed before those runs were read; the "push" results above are from after the fix.
Chosen for v3: "hold" kept; recovery sd 0.3 s instead of 0.9 s (recordings of isolated fireflies, as I remember them, show a sharp lower edge to the intervals, which a wide spread contradicts; cited from memory; check); join on the next beat. The advance-of-count option and the other options were removed from the code.

**v3 (`test_output_v3.txt`)** B1 pass (CV 0.69, mean 43.6 s, shortest 11.8 s). **B2 PASS 5/5** (14.3–14.5 s, CV 0.047–0.068). **B3 FAIL**: share 0.68 at 15 and 0.69 at 300. This model has no density threshold: with sight falling off over 6 m, fifteen fireflies in the clearing still find each other, and at 300 about 30% miss each burst. Not tuned further; it goes on the page under what falls short. **B4 PASS** (0.052 s on, 1.045 s off). B5 pass. B6 pass. **B7**: CV 0.047–0.068, mean 0.056, as guessed (0.05–0.1); well under the candidate 0.159. It is set by the spread of recovery times, which is a setting.

**v4 (`test_output_v4.txt`, and `test_output_v4_B6_elliptic_beam.txt` for B6 alone)** The torch's beam was a vertical slab of bearings, which on the page lit the leaves overhead. It is now an ellipse of directions (bearing and elevation), the same in the model and the picture. B6 then used a beam of 0.35 × 0.3 rad so that it still covers about a third of the clearing as the bet says (v1–v3: a slab 0.2 rad either side). Nothing else changed, and B1–B5 and B7 came out exactly as in v3. B6 pass (100%).

**On the page.** The collective measure (no gap over 1.5 s) finds nothing when the fireflies are hidden from each other, because 300 independent fireflies never leave the clearing dark that long. The page's chart therefore shows each firefly's own wait between bursts, as Sarfati and colleagues measured it, and the readout says "never dark" in that case.
