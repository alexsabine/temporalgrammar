/* VIEW, part 4: the controls of the new scales, and the frame loop. */
function kick() { if (!raf && !document.hidden) raf = requestAnimationFrame(frame); }
function setZ(z) { st.z = clamp(z, 0, 1); st.fly = null; dirty = true; rigDirty = true; mathsDue = true; kick(); }
function flyTo(i) { const to = EV.LV(i).z; if (reduce) { st.z = to; st.fly = null; } else { const d = Math.abs(to - st.z) * EV.DECADES; st.fly = { from: st.z, to, t0: performance.now(), dur: clamp(700 + 420 * d, 1000, 9000) }; } dirty = true; rigDirty = true; mathsDue = true; kick(); }
function stepWay(dir) { const cur = st.fly ? st.fly.to : st.z, eps = 0.0006; let t = null; if (dir > 0) { for (const L of EV.LEVELS) if (L.z > cur + eps) { t = L.id; break; } } else { for (let k = EV.LEVELS.length - 1; k >= 0; k--) if (EV.LEVELS[k].z < cur - eps) { t = EV.LEVELS[k].id; break; } } if (t !== null) flyTo(t); }

/* waypoints: a menu, widest first, with the two hard lines marked in it */
(() => { const sel = $('way'); let grp = null; const group = n => { grp = document.createElement('optgroup'); grp.label = n; sel.appendChild(grp); };
  EV.LEVELS.forEach(L => { if (L.id === EV.IDMIN) group('Out to the horizon'); if (L.id === 0) group('Into the brain'); if (L.id === 7) group('Below the atom'); if (L.id === 11) group('Below the line: imagined'); const o = document.createElement('option'); o.value = String(L.id); o.textContent = L.name + ' · ' + EV.fmtLen(L.fov, 1); grp.appendChild(o); });
  sel.value = String(EV.HOME); sel.addEventListener('change', () => flyTo(+sel.value)); })();
$('back').addEventListener('click', () => stepWay(-1)); $('next').addEventListener('click', () => stepWay(1));
$('zin').addEventListener('click', () => setZ((st.fly ? st.fly.to : st.z) + 0.006)); $('zout').addEventListener('click', () => setZ((st.fly ? st.fly.to : st.z) - 0.006));
$('zoom').addEventListener('input', e => setZ(+e.target.value / 2000));
$('run').addEventListener('click', () => { st.running = !st.running; $('run').textContent = st.running ? 'Pause' : 'Play'; $('run').setAttribute('aria-pressed', String(st.running)); lastT = performance.now(); kick(); });
if (reduce) { $('run').textContent = 'Play'; $('run').setAttribute('aria-pressed', 'false'); }
$('dive').addEventListener('click', () => { flyTo(0); cv.focus({ preventScroll: true }); }); $('flyout').addEventListener('click', () => { flyTo(-2); cv.focus({ preventScroll: true }); });

/* pointer: a drag up or down with a mouse zooms; on a touch screen one finger scrolls the page and two fingers zoom; the gauge can be pressed */
const ptr = new Map(); let drag = null, pinch = null;
const local = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; };
const zFromGauge = y => clamp((y - gaugeY(0)) / (gaugeY(1) - gaugeY(0)), 0, 1);
cv.addEventListener('pointerdown', e => { const p = local(e); ptr.set(e.pointerId, p); cv.focus({ preventScroll: true });
  if (ptr.size === 2) { const a = [...ptr.values()]; pinch = { d: Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), z: st.z }; drag = null; return; }
  if (p[0] > W - gaugeW()) { drag = { gauge: true }; setZ(zFromGauge(p[1])); try { cv.setPointerCapture(e.pointerId); } catch (_) {} return; }
  drag = { y: p[1], x: p[0], z: st.z, moved: false, mouse: e.pointerType !== 'touch' }; if (drag.mouse) { try { cv.setPointerCapture(e.pointerId); } catch (_) {} } });
cv.addEventListener('pointermove', e => { if (!ptr.has(e.pointerId)) return; const p = local(e); ptr.set(e.pointerId, p);
  if (pinch && ptr.size === 2) { const a = [...ptr.values()], d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]); if (pinch.d > 10) setZ(pinch.z + Math.log10(d / pinch.d) / EV.DECADES); return; }
  if (!drag) return; if (drag.gauge) { setZ(zFromGauge(p[1])); return; } if (Math.abs(p[1] - drag.y) + Math.abs(p[0] - drag.x) > 6 * DPR) drag.moved = true; if (drag.mouse && drag.moved) setZ(drag.z + (drag.y - p[1]) / H * 0.09); });
const up = e => { const p = ptr.get(e.pointerId); ptr.delete(e.pointerId); if (ptr.size < 2) pinch = null;
  if (drag && !drag.gauge && !drag.moved && p && st.level === 2 && e.type === 'pointerup') { const T = M.NEURON, fov = fovNow(); let b = -1, bd = 22 * DPR; for (let k = 1; k < T.n; k++) { const c = toScreen(2, fov, (T.x0[k] + T.x1[k]) / 2, (T.y0[k] + T.y1[k]) / 2 + HERO_DY), d = Math.hypot(c[0] - p[0], c[1] - p[1]); if (d < bd) { bd = d; b = k; } } if (b > 0) pickSite(b); }
  drag = null; };
cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
cv.addEventListener('wheel', e => { if (document.activeElement !== cv && !e.ctrlKey) return; e.preventDefault(); setZ(st.z + clamp(e.deltaY, -120, 120) * 0.00007); }, { passive: false });
cv.addEventListener('keydown', e => { const k = e.key; let used = true;
  if (k === 'ArrowDown' || k === '+' || k === '=') setZ(st.z + 0.002); else if (k === 'ArrowUp' || k === '-' || k === '_') setZ(st.z - 0.002);
  else if (k === 'PageDown' || k === ']' || k === 'ArrowRight') stepWay(1); else if (k === 'PageUp' || k === '[' || k === 'ArrowLeft') stepWay(-1);
  else if (k === 'Home') flyTo(EV.HOME); else if (k === 'End') flyTo(EV.IDMAX); else used = false; if (used) e.preventDefault(); });

/* the controls of the new levels */
const setInput = (id, v) => { const e = $(id); e.value = v; e.dispatchEvent(new Event('input')); };
$('flash').addEventListener('click', () => { st.flash = { t0: performance.now(), lv: st.level }; dirty = true; kick(); });
const ZQ = v => Math.pow(10, -2 + v / 1000 * (Math.log10(1100) + 2));
bind('zq', v => { st.zq = ZQ(v); $('zqOut').textContent = 'z = ' + EV.sig(st.zq, 3); if (st.level < -11) { say(); mathsDue = true; } });
const setH0 = k => { st.h0 = k; $('h0a').setAttribute('aria-pressed', String(k === 0)); $('h0b').setAttribute('aria-pressed', String(k === 1)); if (st.level < -11) { say(); facts(); } mathsDue = true; kick(); };
$('h0a').addEventListener('click', () => setH0(0)); $('h0b').addEventListener('click', () => setH0(1));
bind('ealpha', v => { st.Ealpha = v; $('ealphaOut').textContent = fmt(v, 1) + ' MeV'; rigDirty = true; dirty = true; if (st.level === 8) say(); mathsDue = true; });
bind('probe', v => { st.Q = v; const E = Math.pow(10, v); $('probeOut').textContent = E < 1 ? fmt(E * 1000, 0) + ' MeV' : E < 1000 ? fmt(E, E < 10 ? 1 : 0) + ' GeV' : fmt(E / 1000, 1) + ' TeV'; dirty = true; if (st.level === 9) say(); });
$('again').addEventListener('click', () => { st.evSeed++; EVENT = makeEvent(st.evSeed); rebake(10); if (st.level === 10) { say(); facts(); } kick(); });
document.querySelectorAll('[data-cand]').forEach(b => b.addEventListener('click', () => { st.cand = +b.dataset.cand; document.querySelectorAll('[data-cand]').forEach(o => o.setAttribute('aria-pressed', String(o === b))); delete BAKES[12]; rebake(12); if (st.level === 12) say(); kick(); }));
bind('mass', v => { st.mass = v; $('massOut').textContent = EV.sci(Math.pow(10, v), 1) + ' kg'; rigDirty = true; mathsDue = true; });
document.querySelectorAll('[data-try]').forEach(b => b.addEventListener('click', () => { const t = b.dataset.try; st.touched = true;
  if (t === 'home') flyTo(EV.HOME); if (t === 'floor') flyTo(10); if (t === 'ceiling') flyTo(-15);
  if (t === 'moon') { flyTo(-4); setTimeout(() => { st.flash = { t0: performance.now(), lv: -4 }; kick(); }, reduce ? 0 : 3200); }
  if (t === 'voyager') flyTo(-6); if (t === 'z14') { flyTo(-13); setInput('zq', Math.round((Math.log10(14.44) + 2) / (Math.log10(1100) + 2) * 1000)); }
  if (t === 'tension') { flyTo(-15); setH0(st.h0 ? 0 : 1); } if (t === 'touch') { flyTo(8); setInput('ealpha', 16); } if (t === 'gentle') { flyTo(8); setInput('ealpha', 5); }
  if (t === 'blur') { flyTo(9); setInput('probe', -1.1); } if (t === 'hard') { flyTo(9); setInput('probe', 2.5); }
  if (t === 'cross') { flyTo(12); setInput('mass', Math.log10(EV.crossing.m)); } if (t === 'proton') { flyTo(12); setInput('mass', -26.78); }
  if (t === 'synapse') { flyTo(4); setInput('nrel', 5); setInput('prel', 0.1); $('many5').click(); } if (t === 'channel') { flyTo(5); setInput('volt', -20); setInput('nch', 7); }
  $('stage').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' }); kick(); }));
const tb = $('tradeBody'); if (tb) tb.innerHTML = M.INSTRUMENTS.map(I => '<tr><th scope="row">' + I.name + '</th><td class="n">' + M.fmtLen(I.res, 3) + '</td><td class="n">' + M.fmtLen(I.fov, 2) + '</td><td class="n">1 : ' + fmtN(I.fov / I.res) + '</td><td>' + (I.alive ? 'alive' : 'not alive') + '</td><td>' + I.src + '</td></tr>').join('');
/* the table of waypoints, from the model */
$('wayBody').innerHTML = EV.LEVELS.map(L => { const half = L.fov / 2, t = L.fov > 0.3 ? (half >= C0.horizon ? 'none yet' : EV.fmtTime(C0.lookbackOfChi(half))) : EV.fmtTime(half / K.c); return '<tr data-kind="' + L.kind + '"><th scope="row">' + L.name + '</th><td class="n">' + EV.fmtLen(L.fov, 2) + '</td><td class="n">10<sup>' + fmt(Math.log10(L.fov), 1) + '</sup> m</td><td class="n">' + t + '</td><td>' + L.tool + '</td><td>' + KINDTXT[L.kind].toLowerCase() + '</td></tr>'; }).join('');

function frame(now) { raf = 0; const dt = clamp((now - lastT) / 1000 || 0, 0, 0.05); lastT = now;
  if (st.fly) { const u = clamp((now - st.fly.t0) / st.fly.dur, 0, 1); st.z = lerp(st.fly.from, st.fly.to, M.smooth(u)); if (u >= 1) st.fly = null; dirty = true; rigDirty = true; mathsDue = true; }
  setLevel(EV.levelAt(fovNow())); const lv = st.level, anim = st.running && visible && !reduce;
  if (st.running && visible && lv >= 0 && lv <= 6) simulateBrain(dt, now);
  const baking = pump(st.fly ? 6 : 14); if (baking) dirty = true;
  if (anim && (lv === 0 || lv === 9 || lv === 11 || lv === 12)) dirty = true; if (st.flash && now - st.flash.t0 < 40000 && FLASH[lv]) dirty = true; if (anim && now - lastDraw > 80) dirty = true;
  if (dirty) { draw(now); lastDraw = now; dirty = false; if (now - lastReadout > 90) { readout(); lastReadout = now; if (lv === 0 || lv === 3 || lv === -2 || lv === 11 || lv === 12) facts(); } }
  if (rigDirty || lv === 0 || lv === 1 || lv === 3) { drawRig(); rigDirty = false; }
  if (now - lastSay > 600 || sayT === 0) { say(); sayT = 1; lastSay = now; if (lv === 4 || lv === 6) maths(); }
  if (mathsDue && now - lastMaths > 350 && !st.fly) { maths(); mathsDue = false; lastMaths = now; }
  if (st.fly || baking || JOBS.length || job || mathsDue || (st.running && visible && !reduce) || (st.flash && now - st.flash.t0 < 40000)) kick(); else readout(); }
new ResizeObserver(() => { resize(); kick(); }).observe(cv);
if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible) { lastT = performance.now(); kick(); } }, { threshold: 0.02 }).observe($('stage'));
document.addEventListener('visibilitychange', () => { if (!document.hidden) { lastT = performance.now(); kick(); } });
if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { dirty = true; rigDirty = true; kick(); });
resize(); setLevel(EV.levelAt(fovNow())); readout(); maths();
window.EVVIEW = { st, flyTo, setZ, stepWay, ready: () => bakesDone && !JOBS.length && !job, BAKES, baked: () => Object.keys(BAKES).filter(k => BAKES[k].tiers.length || +k === 0).map(Number), frame, draw: () => { draw(performance.now()); drawRig(); }, hasMRI: () => !!MRI };
kick();
