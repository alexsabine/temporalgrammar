# Bets for "Holding weight through time" (equanimity.html)

Written before `model.js` had been run once. Each bet has a threshold. The scripts are `test_model.js` (node, no drawing)
and `test_page.js` (Playwright). Outputs are kept in this folder, failures included. Settings are not to be changed to pass
a bet without the earlier failure being recorded in the log at the foot.

Conventions. One "meeting" is one step of the filter. Noise variance R = 1 throughout; drift variance Q = r R.
"Mean squared error" is the squared distance between the estimate just after a meeting and the world at that meeting.

**B1. Kalman at equal drift and noise (r = 1).** Iterate P- = P + Q, K = P-/(P- + R), P = (1-K) P- two hundred times from P = 1.
Pass if, to six decimal places: K = 0.618034, 1-K = 0.381966, P- = 1.618034, P = 0.618034.
And information: 1/P - 1/P- = 1 (gained at the meeting) and the same amount lost by the next meeting, each within 1e-9.
These must agree with `/home/user/care/crr-checks/compare.py`, section 4.

**B2. The steady gain.** `steadyK(r)` satisfies K^2 + rK - r = 0 within 1e-12 for r = 0.01, 0.1, 0.236, 1, 4, 100.
The recursion started from P = 1 reaches `steadyK(r)` within 1e-9 after 400 steps for each of those r.
`steadyK(1/phi^3)` = 0.381966 to six places.

**B3. The three presets differ in score, in the expected order.** World drifts with r = 1. 20,000 meetings, six seeds.
Fixed gain K = 0.08 ("too heavy"), 0.97 ("too light"), 0.618034 ("balance").
Pass if, for every seed: MSE(heavy) > MSE(light) > MSE(balance).
Pass if the mean over seeds is within 15% (heavy) and 5% (light, balance) of the textbook value for a fixed gain,
((1-K)^2 Q + K^2 R) / (1 - (1-K)^2): 5.547, 0.943, 0.618.
Pass if, scanning K from 0.02 to 0.98 in steps of 0.02 on the same noise, the lowest measured MSE is at K between 0.56 and 0.68.
My guess: all pass. Heavy is the least certain because its errors are strongly correlated from meeting to meeting.

**B4. The fan.** The fan is the band of two standard deviations either side of where the mind expects the world next,
using the mind's own reckoning of its uncertainty (for a fixed gain K, the reckoning that makes K the right gain: P- = K R/(1-K)).
Share of meetings in which the world falls inside the fan, 20,000 meetings:
pass if balance is between 0.93 and 0.97, heavy is below 0.35, light is above 0.99.

**B5. Two who meet.** No drift, start at 0 and 1, one exchange. The gap must equal (1 - Kt - Kl) and match compare.py section 5:
(0.382, 0.618) gives 0.000; (0.5, 0.5) gives 0.000; (0.618, 0.618) gives -0.236; (0.382, 0.382) gives +0.236; (0.2, 0.8) gives 0.000 (to 3 places).
"Both let go" (0.9, 0.9), no drift: the gap changes sign at every one of the first ten exchanges.
"Both hold hard" (0.08, 0.08), no drift: after ten exchanges the gap is 0.84^10 = 0.175 of the start, within 1e-9.
Note written before the run: this means "they never meet" is not what the arithmetic says. They close 16% of the gap each time.
The page must say that, and not "never".
With each drifting on its own (sd 0.5 a meeting), 5,000 meetings:
the gap just before an exchange is never exactly zero in any preset;
root mean square gap before an exchange at balance is within 10% of sqrt(2) x 0.5 = 0.707;
the same measure is larger for "both hold hard" and for "both let go" than for balance. Expected from the algebra:
sqrt(2 q^2 / (1 - f^2)) with f = 1 - Kt - Kl, so 1.30 (hold) and 1.18 (let go).

**B6. The count is the comb added up.** Eight teeth. On a grid of 8,001 points the running sum of the discrete comb
equals the staircase `count(x)` at every grid point. And the running integral of a comb of narrow bells (width 0.01)
equals the staircase within 1e-6 at every grid point more than 0.08 from a tooth.

**B7. The bowl is minus the logarithm of the bell.** Widths 0.05, 0.3, 0.5, 1, 2, 6, 20; x on a grid with |x| <= 30 widths
(beyond that the bell underflows in double precision). Pass if -log(bell(x)/bell(0)) = x^2/(2 s^2) within 1e-9 relative (or 1e-12 absolute).
At a fixed place x = 0.1 the bowl height rises without limit as the width falls (greater than 1e5 at s = 1e-4) and falls to
nothing as the width grows (less than 1e-6 at s = 100).

**B8. The bell keeps its weight.** For each width in B7 the area under the bell is 1 within 1e-6.
A narrow bell picks out one value: the integral of bell(x, s) cos(x) is within 1e-6 of cos(0) = 1 at s = 0.001.

**B9. Same seed, same run.** Two runs of the filter and of the pair with the same seed are identical in every stored number.
A dial given as {from, to, start, end} equals `from` before `start`, `to` after `end`, and is halfway at the middle frame.

**B10. A fixed gain implies a belief.** For K = 0.08, 0.382, 0.618, 0.97: `steadyK(K^2/(1-K))` returns K within 1e-12.

**B11. Page (test_page.js).** No page errors or console errors on desktop (1280x800), phone (390 px wide) and film mode.
No sideways scroll at 390 px. Every button, slider and link in the controls is at least 44 px high.
Film hook: the same frame and view give byte-identical PNGs when called twice, and again after other scenes have been drawn in between.
`EQ.state()` in scene E at frame 4800 with the balance preset reports weight given 0.618034 and uncertainty 1.618034 / 0.618034 to six places.

## Log
- v1 (test_output_v1.txt): 25 passed, 1 failed. **B6 (running sum) failed, and the fault was in my test, not the model.** I wrote "a grid of 8,001 points" over 0 to 9, whose spacing is 0.001125, so no grid point falls on a tooth and the sum and the staircase disagree by one grid step around each tooth. The bet as written is recorded as failed. v2 uses 9,001 points (spacing 0.001, a point on every tooth). The model was not changed.
- v1, B5: rms gap for "both hold hard" came out 1.395 where the steady-state algebra gives 1.30. The run starts 12 apart, and that opening is in the average. Not a failure of the bet (which only asked for larger than balance).

## Bets added later (written before these were run; the model code for them did not yet exist)

**B12. The cone it expects and the surprise it gets.** Must agree with `/home/user/care/crr-checks/surplus.py`.
A filter with gain K assumes a variance before each meeting of K/(1-K) (noise = 1). 20,000 meetings, r = 1, seed 7.
Pass if the cone is 0.087 (K = 0.08), 1.618 (K = 0.618034), 32.333 (K = 0.97) to three places;
if the measured mean squared one-step-ahead error is within 10% of 6.52 (K = 0.08) and within 5% of 1.94 (K = 0.97);
and if cone / measured error is within 5% of 1 at K = 0.618034.

**B13. Path walked against the world's path.** Sum of the distances the estimate moved at each meeting, over the sum of the
distances the world moved. Pass if within 0.03 of 0.22 (K = 0.08), 1.00 (K = 0.618034), 1.66 (K = 0.97).

**B14. One's picture becomes the other.** Gains 0.02 and 0.98, no drift, start 0 and 1: after one exchange both are at 0.02 within 1e-12.
With drift (the page's run), the share of the first exchange's movement made by the first one is 0.02 within 1e-9.
- v2 (test_output_v2.txt): 26 passed, 0 failed, with the B6 grid corrected as above.
- v3 (test_output_v3.txt): B12 to B14 added and run: 31 passed, 0 failed. Measured: surprise 6.670 (heavy), 1.570 (balance, cone 1.618, ratio 1.031), 1.893 (light); path ratios 0.225, 1.004, 1.674.
- v4 (test_output_v4.txt): after a change to how a swap of sides is counted (a gap of 1e-17 after an exact meeting had been counted as a swap): 31 passed.
- Page (test_page_output_v1.txt, v2): 12 passed, 0 failed both times. Software rendering only; no real phone or GPU was used.
