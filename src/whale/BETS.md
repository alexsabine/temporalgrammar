# Bets for the humpback, written before the model was run

Written before `model.js` had been run once. Each bet has a threshold. The test scripts are `test_model.js` (node, no drawing) and `test_page.js` (Playwright); their outputs are kept in this folder, failures included. Settings are not to be changed in order to pass a bet without recording the earlier failure in the log below.

All times are model seconds (60 frames = 1 s). "Mother" is the adult the readouts follow.

**B1. The sound-speed formula is typed correctly.**
Mackenzie (1981) gives a check value with his nine-term equation: T = 25 °C, S = 35, D = 1000 m gives c = 1550.744 m/s (check value quoted from memory).
Pass if `soundSpeed(25,35,1000)` is within 0.01 m/s of that.

**B2. There is a deep sound channel, it moves, and it traps rays.**
With the page's temperature profile (see NOTES.md; the profile is a simplification of mine):
(a) surface 18 °C, salinity 35: the depth of slowest sound lies between 600 m and 1400 m.
(b) surface 2 °C or colder: the slowest sound is at the surface (0 m), as in polar seas.
(c) warming the surface from 10 °C to 28 °C makes the channel deeper, not shallower, at every 2 °C step.
(d) a ray launched from the channel depth at 6° above or below level travels 150 km without touching surface or bottom; a ray launched at 6° from 20 m depth touches the surface at least once.

**B3. The song is a hierarchy and the timing rule reproduces the published list.**
(a) every unit lasts 0.5 to 4 s and every point of every pitch contour lies between 100 Hz and 4000 Hz.
(b) the unit onsets logged by a running model over one whole song equal the frames in `WHALE.song.sequence` exactly (no bubble net in the run).
(c) themes always come in the same order in three successive songs.

**B4. Breathing.**
Mother, 1200 s, no breach and no bubble net.
Pass if: the blowholes are above the water (y > 0) at every logged blow; every surfacing has at least 2 blows; no stretch between two blows is longer than 150 s.
Reported, not judged: mean and CV of the time between first blows of successive surfacings. The candidate value from the CRR programme for a spread of intervals is 0.159. My guess before running: the measured CV will be far below it, under 0.1, because the rate is nearly steady and nothing is random in it. Nothing will be tuned towards 0.159.

**B5. A breach leaves the water and comes back.**
Breach asked for at frame 600.
Pass if: the tip of her snout reaches at least 4 m above the surface; her speed as she leaves is between 6 and 9 m/s (Segre and colleagues 2020 report breaches near 8 m/s, from memory); the splash is logged within 20 s of the request; she blows within 40 s after the splash.

**B6. The blow is low and bushy.**
Pass if the highest mist particle of a mother's blow reaches between 2 m and 4 m above the blowholes, and the mist is wider than a third of its height at its highest.

**B7. The fish school, and give way to the whale.**
420 fish, 600 s, no events.
(a) after the first 60 s the polarisation (length of the mean unit heading) averages above 0.5.
(b) the median distance to the nearest neighbour, sampled every 10 s, is between 0.2 m and 1.0 m.
(c) fewer than 1% of fish-samples lie within 1.5 m of a whale's centre line.
I am least sure of (c): she swims through the school from time to time and the fish may not clear in time.

**B8. The bubble net.**
Asked for at frame 600.
Pass if: bubbles reach the surface in a ring whose mean radius from the net centre is 6 to 10 m; at the lunge at least 60% of the fish are inside 10 m of the centre and above 8 m depth; both whales' snouts clear the surface; at least 5 fish are taken; the whole thing is over (both whales back to ordinary states) within 90 s.

**B9. Same frame, same picture.**
(a) Two models with the same seed and the same events agree exactly at frame 1800 (every whale number and every fish position).
(b) In the browser, `__clay.shot(300, v)` twice gives identical pixels; and `shot(300, v)` taken after `shot(900, v)` is identical to the first.

**B10. The page.**
Zero page errors and zero console errors at 1280×800 and at 390 px wide. No sideways scroll at 390 px. Every button at least 44 px high. In `__TEST` mode no animation loop runs (two readings of the frame counter 1 s apart are equal) and nothing but the picture is visible.

---

# Log

**v1 (`test_output_v1.txt`), first run of the model.** B1 pass (1550.7440). B2a–d pass (channel at 975 m for an 18 °C surface; at the surface for 2 °C and 0 °C; deeper with every 2 °C of warming; both rays from the channel depth clear surface and bottom for 150 km, the ray from 20 m touched the surface twice). B3a–c pass (9 units, 100 to 3600 Hz, 53 onsets all equal, 127.4 s). B4 pass: 30 blows in 1200 s, three per surfacing, longest gap 97.8 s. Reported: 9 dive cycles, mean 120.4 s, CV 0.053, against the candidate 0.159. My guess (under 0.1) held; the value is far from the candidate and nothing was tuned towards it. B5 pass (snout 5.28 m up, leaving at 8.32 m/s). B6 pass, near the top of the allowed band (3.85 m high, 3.34 m wide). B7a–c pass (polarisation 0.954, nearest neighbour 0.37 m, no fish-sample within 1.5 m of a whale's centre line; my worry about (c) was wrong).
**B8 FAIL: no fish taken (needed 5).** The ring (8.2 m), the penning (100% inside) and the lunges were fine, but the whales took 26 s to reach their places (the time limit, not arrival, started the circling) and every fish fled the rising mouths. Changes made after this failure, recorded here: arrival distance loosened from 3.5 m to 5.5 m; fish flee a lunging whale at 0.22 of the usual strength (they are penned against the surface); mouth catch radius 0.14 to 0.16 of body length. Rerun: pass, 35 fish taken, over in 44.8 s.

**v2 (`test_output_v2.txt`), after the picture was tuned** (spray at the waterline, two bubbles a step instead of three, flippers spread earlier in a breach; none of these touch the bets' thresholds). All sixteen model bets pass. B8: ring 8.58 m from 3619 bubbles, 35 taken. Dive cycles unchanged: mean 120.4 s, CV 0.053.

**Page bets (`test_page_output_v1.txt`, `_v2.txt`).** B9b pass: frame 300 gave identical pixels twice, again after frame 900, after another view, and after filming a breach. B10 pass: no page or console errors at 1280×800, on the iPhone 13 profile, or in film mode; phone scroll width 390 of 390; smallest control 44 px; film mode frame counter did not move in a second and no page furniture was visible. Ignored in the error count: the font and favicon requests that the test itself blocks.
Film scheduling, measured: a breach asked to show at frame 640 left the water at 640 exactly; a blow asked for at 1200 came at 1201.

Not bet on, so not claimed: frame rate on a real phone (tests ran on software WebGL only), how the sound sounds (no test listened to it), and whether the whale reads as a humpback to a specialist.
