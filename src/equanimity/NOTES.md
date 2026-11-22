# Notes for equanimity.html ("Holding Weight Through Time")

Build: `./build.sh` joins head.html + three.js (CDN tag) + model.js + app.js into ../../equanimity.html. Do not edit the built page.
Tests: `node test_model.js` (bets B1-B10, B12-B14), `node test_page.js` (B11; needs `python3 -m http.server 8765` in the repo root). Outputs kept here, failures included.

## Film hook
Set `window.__TEST = {}` (optionally `{budget: pixels, low: true}`) before load. No loop, no page furniture, canvas fills the viewport.
`window.__clay.shot(frame, view)` draws one frame synchronously. 60 frames = 1 s. Nothing is carried from one call to the next: the run is recomputed from frame 0 each time.
`view` keys (every number may also be `{from, to, start, end, ease?:'linear'}` in frames, or `{steps:[[frame,value],...]}`):
- `scene` 'A'..'G'
- `yaw`, `pitch` (radians, added to the scene's default), `zoom` (1 = default; 2 = twice as close)
- `card` 0..1 (0 hides the card and frames the objects alone; 1 takes the whole card in)
- `focus` (distance of sharp focus in bench units; default follows the objects)
- `width` (A, D): 0.3 to 6. A at 0.3 is the bare pin.
- `K` (E): gain set by hand. `r` (E, G): drift to noise; with no `K` the filter finds its own gain.
- `preset` (E): 'heavy' | 'light' | 'balance'. (F): 'hold' | 'letgo' | 'mirror' | 'balance'.
- `Kt`, `Kl` (F): the gold and green gains.
- `seed` (E, F, G).
`window.__clay.meeting` = 48 (frames between meetings), `.loop` = 480 (frames for the marble's pass in B and C: teeth are passed at frames 61, 110, 159, 208, 257, 306, 355, 404).
`window.EQ.state()` returns the numbers on the readouts for the last frame drawn.

## Choices of mine (not mathematics)
- One unit of place is 0.3 bench units. Bell height 1 is 1.5 bench units. Bowl height x^2/2P is scaled by 0.204 and cut off at a rim 0.7 high.
- Bowls are channels (a curve pulled along its depth), not round bowls, so that the curve shows from the front.
- When K is set by hand the bowl and the cone show P = K (after) and K/(1-K) (before), noise = 1: the belief for which that K is the Kalman gain.
- In F the bowls are drawn the same way from each one's gain. There is no noise model in F; this is a drawing convention only.
- In F each drifts with sd 0.5 a meeting, starting at -6 and +6. Pins mark the starts.
- Within a meeting interval the marble slides during the first 8% to 42% (E) or 0 to 35% (F) of the interval; the card uses the same timing.
- E seed 232 was chosen so that the world stays on the rail (within 6.8 of the start) for 90 meetings. Its scores over the first 30 meetings (heavy 3.83, light 0.92, balance 0.74) are near the long-run values 5.55, 0.94, 0.62 for light and balance and lower for heavy. If the world leaves +-8 the origin is moved by 8 (everything slides together).
- The staircase blocks drop into place from a short height when a tooth is passed. That is stagecraft.
- The marble's glass uses three.js transmission on larger screens and a plain glazed look on small ones.

## Honest notes
- "Both hold hard" does not mean never meeting: 16% of the gap closes at each exchange (bet B5). The page says so.
- In the 0.02 / 0.98 preset the first one still drifts on its own, so "where gold began" and where they sit slowly part company.
