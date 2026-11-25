#!/bin/sh
# Assembles ../../how-far-does-it-go.html. No other step.
#   head.html                      the page, with four places where the brain's words are let in
#   brain_*.html                   those words, cut from the earlier seven-level page (see prep.py)
#   model_brain.js + model.js      the <script id="model"> block: pure calculation, runs under Node
#   view_brain_a.js .. view_c.js   the drawing and the controls
cd "$(dirname "$0")"
python3 - <<'PY'
import io
R = lambda p: io.open(p, encoding='utf8').read()
h = R('head.html')
for k, f in [('<!--BRAIN_TEXT-->', 'brain_text.html'), ('<!--BRAIN_CONTROLS-->', 'brain_controls.html'), ('<!--BRAIN_MATHS-->', 'brain_maths.html'), ('<!--BRAIN_SOURCES-->', 'brain_sources.html')]:
    assert k in h, k
    h = h.replace(k, R(f).rstrip('\n'))
out = [h, '<script id="model">', R('model_brain.js'), R('model.js'), '</script>', '<script>']
for f in ['view_brain_a.js', 'view_a.js', 'view_brain_b.js', 'view_b.js', 'view_brain_c.js', 'view_c.js']:
    out.append(R(f))
out += ['</script>', '</body>', '</html>', '']
io.open('../../how-far-does-it-go.html', 'w', encoding='utf8').write('\n'.join(out))
PY
