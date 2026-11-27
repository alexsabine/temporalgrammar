# Simulations at the bottom of each dive

Each note on the page dives from the photograph, through real photographs, to a live simulation.
One file per simulation here; `../build.sh` sets them into the page.

## The contract

```js
HSM.sim('key', {
  slider: {label:'Light', lo:'dusk', hi:'full sun', value:.45},   // optional, one at most
  toggle: {off:'Dry it out', on:'Let it rain'},                    // optional, one at most; label shows the action a press would take
  setup: function(w, h, R) { return state },                       // w,h in CSS px (portrait, about 3:4, 300 to 700 px wide). R() is a seeded random 0..1
  frame: function(s, g, w, h, t, dt, ui) { }                       // paint the whole frame, background included
});
```

- `g` is a 2D canvas context already scaled for the screen. Draw in CSS px.
- `t` seconds since the simulation appeared, `dt` seconds since the last frame (0 to 0.05). Move things by `dt`.
- `ui.a` the slider (0..1) or undefined. `ui.on` the toggle. `ui.px, ui.py` pointer position or -1. `ui.down` pointer pressed.
- `setup` is called again on resize. Keep everything in the returned state; no globals, no DOM, no timers.
- Helpers in `_lib.js`: `HSM.blob(rgba, r, hard)` cached soft sprite, `HSM.label(g,x,y,text,align)`, `HSM.vignette`, `HSM.clamp`, `HSM.smooth`.

## Rules

- Canvas 2D only. No `ctx.filter`, no `shadowBlur` inside loops (Safari and phones). Pre-render sprites to offscreen canvases in `setup`.
- Under 4 ms a frame at 390x520 in the harness. It has to run on a phone.
- It must read correctly as a still at any moment, and from the first second.
- At most three `HSM.label` captions, short, plain words. The page supplies the title and explanation.
- True to the science. If a simplification is made, it should be one a biologist would accept in a museum.

## Trying one

Open `harness.html?sim=key` in a browser, or `node simshot.js key /some/out/prefix 0,0.5,1` for stills at three slider positions.
`cells.js` is the reference for the standard expected.
