#!/bin/sh
# Assembles ../../half-a-square-metre.html from page.html, with the photograph and its mask set in as data.
#   page.html     the whole page: words, styles, shader and controls
#   photo.webp    the photograph, untouched
#   mask.webp     where the water, the leaves and the living cover are (see make_mask.py)
#   sims/*.js     the simulation at the bottom of each dive (see sims/README.md)
# The photographs passed on the way down are in ../../half-a-square-metre/ and load only when a dive begins.
cd "$(dirname "$0")"
python3 - <<'PY'
import io, base64
R = lambda p: io.open(p, encoding='utf8').read()
B = lambda p: 'data:image/webp;base64,' + base64.b64encode(open(p, 'rb').read()).decode()
h = R('page.html')
for k, f in [('{{PHOTO}}', 'photo.webp'), ('{{MASK}}', 'mask.webp')]:
    assert h.count(k) == 1, k
    h = h.replace(k, B(f))
import glob, os
sims = [R('sims/_lib.js')] + [R(f) for f in sorted(glob.glob('sims/*.js')) if os.path.basename(f) not in ('_lib.js', 'simshot.js')]
assert h.count('<!--SIMS-->') == 1
h = h.replace('<!--SIMS-->', '\n'.join('<script>\n' + x.strip() + '\n</script>' for x in sims))
io.open('../../half-a-square-metre.html', 'w', encoding='utf8').write(h)
PY
