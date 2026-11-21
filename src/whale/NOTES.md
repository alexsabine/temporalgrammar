# The Humpback: notes

`whale.html` is assembled by `build.sh` from `head.html`, `model.js` (no drawing; runs in node) and `app.js` (three.js picture, Web Audio, controls). Edit the parts, then run `./build.sh`.

## Science stated on the page, and its standing

- **Humpbacks do not echolocate.** They are baleen whales. Echolocation belongs to toothed whales. Nothing in the piece echolocates; the page says so.
- **Only males sing** (Garland et al. 2011 abstract, checked). Payne & McVay (1971) did not know the singer's sex; the page does not attribute that to them.
- Song hierarchy and "7 to 30 minutes": Payne & McVay 1971 abstract (checked in PubMed).
- Larynx, no air let out: Elemans et al. 2024 (existence and abstract checked; "without letting air out" is my summary of air recycling in baleen whales, **unverified wording**).

## Citations

Checked against PubMed (title, authors, journal, volume, pages): Payne & McVay 1971; Payne & Webb 1971; Garland et al. 2011; Elemans et al. 2024; Fish & Battle 1995; Segre et al. 2020; Szabo et al. 2024; Couzin et al. 2002; Pope & Fry 1997.

**Not checked, from memory** (marked so on the page): Mackenzie 1981; Munk 1974; Ewing & Worzel 1948; Jurasz & Jurasz 1979; Wiley et al. 2011; Sharpe & Dill 1997; Clapham & Mead 1999; Reynolds 1987.

## Numbers from memory, unverified

- Mackenzie's coefficients and his check value 1550.744 m/s at 25 °C, 35, 1000 m. The code reproduces the check value, which supports both, but both came from the same memory.
- Body length 12 to 16 m; flipper "close to a third of body length"; calf 6 m (a calf some months old); singer 13 m.
- Breach speed "about 8 m/s" (attributed to Segre et al. 2020; I did not re-read the paper's figures).
- Blow height about 3 m. Dive lengths in life (3 to 10 minutes or more): not cited on the page, which says only "several times shorter than in life".
- Song heard "for tens of kilometres": unverified, no source given on the page.
- Payne & Webb's argument concerns fin whale 20 Hz calls; "fin and blue whales" on the page is from memory.
- "Herring will not cross a wall of bubbles" overstates Sharpe & Dill a little; they reported strong avoidance in tanks.
- Light: the page cites Pope & Fry for red being absorbed first. The three extinction values in the shader (0.17, 0.042, 0.024 per metre) were set by eye, not taken from their table, and the page says so in Settings.
- The singer's head-down posture at about 24 m is from general descriptions; no source cited.

## Modelling choices that are mine, not data

- Temperature profile: 50 m mixed layer, exponential thermocline (scale 400 m) to 2 °C, constant salinity. Chosen so that an 18 °C surface puts the channel near 1 km.
- The song units, phrases and themes were composed for the page. They are not a transcription.
- Breath count rate, thresholds, breach count, net geometry and timings: see Settings at the foot of the page.
- Bubble rise about 1 m/s (faster than small bubbles) so the net forms in 20 s.
- Fish flee a lunging whale less than a cruising one (changed after bet B8 failed; see BETS.md).
- Singing and bubble-net feeding share one sea and the singer joins the net. The page says this is not how it is.

## Film hook

Set `window.__TEST = true` before load. No loop runs, the page chrome is hidden, the canvas fills the window.

- `__clay.shot(frame, view)` advances the seeded model to `frame` (60 = 1 s) and draws one picture synchronously. Views do not persist between calls. Going backwards, or changing the events, reruns the model from frame 0.
- `view.cam`: `'under' | 'follow' | 'surface' | 'above' | 'school'`. `view.yaw`, `view.pitch`: degrees added to the preset. `view.zoom`: 1 is the preset distance, 2 is twice as close. Also `view.dist` (metres), `view.target` (`'mother' | 'calf' | 'singer' | 'school' | 'top'`), `view.dpr`, `view.seed`.
- `view.event`: `'breach' | 'blow' | 'bubblenet' | null`, with `view.eventAt`. For a breach, `eventAt` is the frame she leaves the water; for a blow, the frame of the blow (the page works out when to start; measured: exact for the breach, one frame late for the blow in the test). For the bubble net, `eventAt` is when it starts; the lunge comes about 41 s later. `view.eventStart: true` makes `eventAt` the raw start for all three. Several events: `view.events = [{event, at}, ...]`.
- An event too early to prepare (a breach before about frame 350, a blow before about 1000) happens as soon as it can; ask `plan`.
- `__clay.plan(view, untilFrame)` returns the frames of every logged event (blow, breach_start, exit, splash, net_start, net_spin, net_lunge, lunge, net_done) for that schedule, without drawing.
- Spontaneous breaches are off in film mode unless `view.auto` is true.
- `WHALE.state()` gives frame, each whale's position, depth, state, breath count, the song position, fish, net, dive statistics, sound speeds, the log and the camera.
- `WHALE.song` is an array of the nine units (`id, name, dur, loud, contour [[s, Hz]...], timbre, vib, form, pulse`). On it: `.themes`, `.order`, `.sequence` (every onset of one song: `frame, t, unit, theme, phrase, of, k, dur, loud`), `.frames`, `.duration`, `.pitchAt(unit, t)`, `.at(frame)`. The singer starts at frame 0 and repeats every `.frames` frames, unless a bubble net interrupts him (he is silent while netting and then carries on from where he stopped, so the table is offset after a net).

## Weaker than intended

- Views from above the water. The sea surface is plain, whales seen through it from a boat look pale and glassy, and the breach splash is a soft white cloud with droplets, not sheets of water.
- The open mouth in a lunge is a hinged lower jaw with a dark fringe for baleen; it reads at a distance, not close up.
- Tubercles are modelled but small on screen; the dorsal fin is a simple blade.
- The calf is a scaled-down adult.
- During the net the whales are moved along a set spiral and turn under it by interpolation, which can look like sliding.
- Sound was never listened to. The synthesis ran without errors in the browser test; that is all that is known.
- Frame rate on real hardware is untested (software WebGL only). Costs: three whales of about 20,000 triangles each, 420 instanced fish (260 on small screens), 36 additive shaft quads, about 5,000 point sprites, one surface mesh; pixel ratio capped at 2 (1.5 small screens) and stepped down if frames are slow.
