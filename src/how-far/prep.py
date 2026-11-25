#!/usr/bin/env python3
"""One-off: cuts the brain's seven pictures and their read-outs out of the earlier seven-level page's scripts
(paint_brain.js, legacy_view2.js, legacy_view3.js, taken unchanged from it) and writes view_brain_a.js and view_brain_b.js,
with the few edits listed below. Run once; build.sh then only concatenates."""
import re, io
R = lambda p: io.open(p, encoding='utf8').read().split('\n')
def sub(s, a, b, n=1):
    assert s.count(a) >= 1, 'missing: ' + a[:70]
    return s.replace(a, b) if n == 0 else s.replace(a, b, n)

# ---- A: helpers and painters (everything before the baking section) ----
P = R('paint_brain.js')
cut = next(i for i, l in enumerate(P) if 'Baking' in l and l.startswith('/*'))
A = '\n'.join(P[:cut])
A = sub(A, "const M = window.HFD, $ = id => document.getElementById(id);", "const M = HFD, $ = id => document.getElementById(id);")
A = sub(A, "const MRI = (() => {", "let MRI = null; const makeMRI = () => {")
A = sub(A, "  S.voxelise(1); return S;\n})();", "  S.voxelise(1); return S;\n};")
tail = [l for l in P[cut:] if l.startswith(('const world', 'const GRAINT', 'function feather', '  for (const hz', "  g.globalCompositeOperation = 'source-over'; }", 'function shotNoise', '/* noise that belongs'))]
A += '\n' + '\n'.join(tail) + '\n'
A += "const PAINT_B = [null, function* (Bk) { yield* paintCortex(Bk); shotNoise(Bk, 0.34); }, function* (Bk) { yield* paintNeuron(Bk); shotNoise(Bk, 0.3); }, Bk => paintEM(Bk, 1, Bk.col), Bk => paintEM(Bk, 1000, null), paintMachine, paintAtoms];\n"
io.open('view_brain_a.js', 'w', encoding='utf8').write(A)

# ---- B: what moves, the traces, the words (levels 0 to 6) ----
V = R('legacy_view2.js'); L = lambda a, b: '\n'.join(V[a - 1:b])
B = '\n'.join([L(3, 3), L(5, 6), L(27, 46), L(48, 82), L(84, 96), L(127, 160), L(163, 174), L(175, 192), L(202, 217)])
B = sub(B, "function simulate(dt, now) {", "function simulateBrain(dt, now) {")
B = sub(B, "function live(i, fov, a, now) {", "function liveBrain(i, fov, a, now) {")
B = sub(B, "function drawRig() {", "function rigBrain() {")
B = sub(B, "const CAPS = [", "const CAPS_B = [")
B = sub(B, "const ARIA = [", "const ARIA_B = [")
B = sub(B, "function say() {", "function sayBrain() {")
B = sub(B, "  $('say').textContent = t; }", "  return t; }")
B = sub(B, "function facts() {", "function factsBrain() {")
B = sub(B, "  $('facts').innerHTML = rows.map(r => '<dt>' + r[0] + '</dt><dd>' + r[1] + '</dd>').join(''); }", "  return rows; }")
B = sub(B, "function maths() {", "function mathsBrain() {")
B = sub(B, " This is the level at which the textbook makes its claims.", "")
B = sub(B, " It is not the H01 data; those are linked below.", " It is not the H01 data.")
# the old zoom equation is replaced by the new one
B = re.sub(r"\n  setLive\('lvScale'.*?\n", "\n", B, count=1, flags=re.S)
# labels on the stage in the site's reading face
B = sub(B, "'px ui-sans-serif,system-ui,-apple-system,\"Segoe UI\",Roboto,Arial,sans-serif'", "'px \"Source Serif 4\",Georgia,serif'", 0)
# the old instrument chart measured from the old frame; it now reads the new one
B = sub(B, "const xf = X(fovNow());", "const xf = X(clamp(fovNow(), 2e-10, 0.3));")
B = sub(B, "M.fmtLen(fov, 2)", "EV.fmtLen(fov, 2)", 0)
C3 = R('legacy_view3.js')
ctl = '\n'.join(C3[40:59])
ctl = sub(ctl, "if (MRI.vox !== mm) MRI.voxelise(mm);", "st.vox = mm; if (MRI && MRI.vox !== mm) MRI.voxelise(mm);")
ctl = sub(ctl, "clearTimeout(ioTimer); ioTimer = setTimeout(() => { computeIO(); if (st.level === 2) { say(); maths(); } kick(); }, 120);",
          "clearTimeout(ioTimer); if (st.level === 2) ioTimer = setTimeout(() => { computeIO(); if (st.level === 2) { say(); maths(); } kick(); }, 120); else io3 = null;")
B = B.replace('MRI.vox', 'st.vox')
io.open('view_brain_b.js', 'w', encoding='utf8').write(B + '\n')
io.open('view_brain_c.js', 'w', encoding='utf8').write(ctl + '\n')
print('ok', len(A), len(B), len(ctl))
