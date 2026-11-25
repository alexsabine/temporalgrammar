/* VIEW, part 3: controls and the frame loop. */
let raf = 0, visible = true, lastReadout = 0, lastSay = 0;
function kick() { if (!raf && !document.hidden) raf = requestAnimationFrame(frame); }
function setZ(z) { st.z = clamp(z, 0, 1); st.fly = null; dirty = true; kick(); }
function flyTo(i) { const to = M.zoomOf(M.LEVELS[i].fov); if (reduce) { st.z = to; st.fly = null; } else { const d = Math.abs(to - st.z) * M.DECADES; st.fly = { from: st.z, to, t0: performance.now(), dur: clamp(700 + 900 * d, 900, 5200) }; } request(i, true); dirty = true; kick(); }
const nearest = () => { let b = 0, bd = 9; M.LEVELS.forEach((L, i) => { const d = Math.abs(M.zoomOf(L.fov) - (st.fly ? st.fly.to : st.z)); if (d < bd) { bd = d; b = i; } }); return b; };
function stepWay(dir) { const cur = st.fly ? st.fly.to : st.z, eps = 0.004; let t = -1; if (dir > 0) { for (let i = 0; i < NL; i++) if (M.zoomOf(M.LEVELS[i].fov) > cur + eps) { t = i; break; } } else { for (let i = NL - 1; i >= 0; i--) if (M.zoomOf(M.LEVELS[i].fov) < cur - eps) { t = i; break; } } if (t >= 0) flyTo(t); }

/* waypoints */
(() => { const ol = $('way'), short = ['Brain', 'Cortex', 'Neuron', 'Wiring', 'Synapse', 'Molecules', 'Atoms'];
  M.LEVELS.forEach((L, i) => { const li = document.createElement('li'), b = document.createElement('button'); b.type = 'button'; b.className = 'mk'; b.innerHTML = '<b>' + (i + 1) + '</b><span>' + short[i] + '</span>'; b.setAttribute('aria-label', 'Waypoint ' + (i + 1) + ': ' + L.name + ', frame ' + M.fmtLen(L.fov, 2) + ' wide'); b.addEventListener('click', () => flyTo(i)); li.appendChild(b); ol.appendChild(li);
    if (i === 1) { const t = document.createElement('li'); t.className = 'tb'; t.textContent = 'Textbook stops'; t.setAttribute('aria-label', 'The textbook’s account stops here, between waypoints 2 and 3'); ol.appendChild(t); } }); })();
$('back').addEventListener('click', () => stepWay(-1)); $('next').addEventListener('click', () => stepWay(1));
$('zin').addEventListener('click', () => setZ((st.fly ? st.fly.to : st.z) + 0.025)); $('zout').addEventListener('click', () => setZ((st.fly ? st.fly.to : st.z) - 0.025));
$('zoom').addEventListener('input', e => setZ(+e.target.value / 1000));
$('run').addEventListener('click', () => { st.running = !st.running; $('run').textContent = st.running ? 'Pause' : 'Play'; $('run').setAttribute('aria-pressed', String(st.running)); lastT = performance.now(); kick(); });
if (reduce) { $('run').textContent = 'Play'; $('run').setAttribute('aria-pressed', 'false'); }

/* pointer: a drag up or down with a mouse zooms; on a touch screen one finger scrolls the page and two fingers zoom; the gauge can be pressed */
const ptr = new Map(); let drag = null, pinch = null;
const local = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height]; };
const zFromGauge = y => clamp((y - gaugeY(0)) / (gaugeY(1) - gaugeY(0)), 0, 1);
cv.addEventListener('pointerdown', e => { const p = local(e); ptr.set(e.pointerId, p); cv.focus({ preventScroll: true });
  if (ptr.size === 2) { const a = [...ptr.values()]; pinch = { d: Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]), z: st.z }; drag = null; return; }
  if (p[0] > W - gaugeW()) { drag = { gauge: true }; setZ(zFromGauge(p[1])); try { cv.setPointerCapture(e.pointerId); } catch (_) {} return; }
  drag = { y: p[1], x: p[0], z: st.z, moved: false, mouse: e.pointerType !== 'touch' }; if (drag.mouse) { try { cv.setPointerCapture(e.pointerId); } catch (_) {} } });
cv.addEventListener('pointermove', e => { if (!ptr.has(e.pointerId)) return; const p = local(e); ptr.set(e.pointerId, p);
  if (pinch && ptr.size === 2) { const a = [...ptr.values()], d = Math.hypot(a[0][0] - a[1][0], a[0][1] - a[1][1]); if (pinch.d > 10) setZ(pinch.z + Math.log10(d / pinch.d) / M.DECADES); return; }
  if (!drag) return; if (drag.gauge) { setZ(zFromGauge(p[1])); return; } if (Math.abs(p[1] - drag.y) + Math.abs(p[0] - drag.x) > 6 * DPR) drag.moved = true; if (drag.mouse && drag.moved) setZ(drag.z + (drag.y - p[1]) / H * 0.45); });
const up = e => { const p = ptr.get(e.pointerId); ptr.delete(e.pointerId); if (ptr.size < 2) pinch = null;
  if (drag && !drag.gauge && !drag.moved && p && st.level === 2 && e.type === 'pointerup') { const T = M.NEURON, fov = fovNow(); let b = -1, bd = 22 * DPR; for (let k = 1; k < T.n; k++) { const c = toScreen(2, fov, (T.x0[k] + T.x1[k]) / 2, (T.y0[k] + T.y1[k]) / 2 + HERO_DY), d = Math.hypot(c[0] - p[0], c[1] - p[1]); if (d < bd) { bd = d; b = k; } } if (b > 0) pickSite(b); }
  drag = null; };
cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
cv.addEventListener('wheel', e => { if (document.activeElement !== cv && !e.ctrlKey) return; e.preventDefault(); setZ(st.z + clamp(e.deltaY, -120, 120) * 0.00045); }, { passive: false });
cv.addEventListener('keydown', e => { const k = e.key; let used = true;
  if (k === 'ArrowDown' || k === '+' || k === '=') setZ(st.z + 0.01); else if (k === 'ArrowUp' || k === '-' || k === '_') setZ(st.z - 0.01);
  else if (k === 'PageDown' || k === ']' || k === 'ArrowRight') stepWay(1); else if (k === 'PageUp' || k === '[' || k === 'ArrowLeft') stepWay(-1);
  else if (k === 'Home') flyTo(0); else if (k === 'End') flyTo(NL - 1); else if (k >= '1' && k <= '7') flyTo(+k - 1); else used = false; if (used) e.preventDefault(); });

/* the controls of each level */
const bind = (id, fn) => { const e = $(id); e.addEventListener('input', () => { fn(+e.value); kick(); }); fn(+e.value); };
let labTimer = 0, ioTimer = 0;
bind('vox', v => { const mm = VOX[v]; $('voxOut').textContent = M.fmtLen(mm * 1e-3, 2); if (MRI.vox !== mm) MRI.voxelise(mm); dirty = true; if (st.level === 0) { facts(); say(); } });
bind('lab', v => { $('labOut').textContent = fmt(v, 1) + '%'; if (Math.abs(labelFrac - v / 100) < 1e-9) return; labelFrac = v / 100; clearTimeout(labTimer); labTimer = setTimeout(() => { rebake(1); kick(); }, 140); if (st.level === 1) say(); });
function pickSite(k) { st.site = k; const T = M.NEURON, pi = M.PATH.indexOf(k); if (pi > 0) $('site').value = pi; $('siteOut').textContent = fmt(T.dist[k], 0) + ' µm'; D3.reset(); TR.n = 0; TR.site.fill(0); TR.soma.fill(0); clearTimeout(ioTimer); ioTimer = setTimeout(() => { computeIO(); if (st.level === 2) { say(); maths(); } kick(); }, 120); if (st.level === 2) { facts(); maths(); } dirty = true; kick(); }
$('site').max = M.PATH.length - 1; $('site').value = M.PATH.indexOf(M.HOME_SITE);
bind('site', v => { if (M.PATH[v] !== st.site || $('siteOut').textContent === '') pickSite(M.PATH[v]); });
bind('nsyn', v => { st.nsyn = v; $('nsynOut').textContent = String(v); rigDirty = true; if (st.level === 2) say(); });
$('fire3').addEventListener('click', () => { st.touched = true; fire3(); kick(); });
$('nmda').addEventListener('click', () => { st.nmda = !st.nmda; $('nmda').setAttribute('aria-pressed', String(st.nmda)); rigDirty = true; say(); kick(); });
bind('recon', v => { st.recon = v / 100; $('reconOut').textContent = v + '%'; dirty = true; });
bind('prel', v => { st.p = v; $('prelOut').textContent = fmt(v, 2); resetQ(); if (st.level === 4) { say(); maths(); } });
bind('nrel', v => { st.nrel = v; $('nrelOut').textContent = String(v); resetQ(); if (st.level === 4) { say(); facts(); maths(); } dirty = true; });
$('fire5').addEventListener('click', () => { fire5(performance.now()); maths(); kick(); });
$('auto5').addEventListener('click', () => { st.auto5 = !st.auto5; $('auto5').setAttribute('aria-pressed', String(st.auto5)); kick(); });
$('many5').addEventListener('click', () => { const now = performance.now(); for (let k = 0; k < 500; k++) fire5(now); say(); maths(); kick(); });
$('clear5').addEventListener('click', () => { resetQ(); say(); maths(); kick(); });
bind('volt', v => { st.V = v; $('voltOut').textContent = fmt(v, 0) + ' mV'; CH.setV(v); rigDirty = true; if (st.level >= 5) { say(); facts(); maths(); } });
bind('nch', v => { st.nch = NCH[v]; $('nchOut').textContent = String(st.nch); resetCH(); if (st.level === 5) { say(); facts(); maths(); } });
const setInput = (id, v) => { const e = $(id); e.value = v; e.dispatchEvent(new Event('input')); };
document.querySelectorAll('[data-try]').forEach(b => b.addEventListener('click', () => { const t = b.dataset.try; st.touched = true;
  if (t === 'line') flyTo(1);
  if (t.indexOf('dend') === 0) { flyTo(2); setInput('site', M.PATH.indexOf(M.HOME_SITE)); setInput('nsyn', +t.slice(4)); setTimeout(fire3, reduce ? 0 : 400); }
  if (t === 'fail' || t === 'sure') { flyTo(4); setInput('nrel', 5); setInput('prel', t === 'fail' ? 0.1 : 0.9); $('many5').click(); }
  if (t === 'one' || t === 'thousand') { flyTo(5); setInput('volt', -20); setInput('nch', t === 'one' ? 0 : 7); }
  if (t === 'rev' || t === 'neg' || t === 'pos') { flyTo(6); setInput('volt', t === 'rev' ? -77 : t === 'neg' ? -100 : 0); }
  $('stage').scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'nearest' }); kick(); }));

/* the trade-off table, from the model's list of instruments */
$('tradeBody').innerHTML = M.INSTRUMENTS.map(I => '<tr><th scope="row">' + I.name + '</th><td class="n">' + M.fmtLen(I.res, 3) + '</td><td class="n">' + M.fmtLen(I.fov, 2) + '</td><td class="n">1 : ' + fmtN(I.fov / I.res) + '</td><td>' + (I.alive ? 'alive' : 'not alive: fixed, frozen or crystallised') + '</td><td>' + I.src + '</td></tr>').join('');

function frame(now) { raf = 0; const dt = clamp((now - lastT) / 1000 || 0, 0, 0.05); lastT = now;
  if (st.fly) { const u = clamp((now - st.fly.t0) / st.fly.dur, 0, 1); st.z = lerp(st.fly.from, st.fly.to, M.smooth(u)); if (u >= 1) st.fly = null; dirty = true; }
  setLevel(M.levelAt(fovNow()));
  if (st.running && visible) simulate(dt, now);
  const baking = pump(st.fly ? 5 : 12); if (baking) dirty = true;
  if (st.running && visible && st.level === 0) dirty = true;
  if (dirty) { draw(now); dirty = false; if (now - lastReadout > 90) { readout(); lastReadout = now; if (st.level === 0 || st.level === 3) { facts(); } } }
  if (rigDirty || st.level === 0 || st.level === 1 || st.level === 3) { drawRig(); rigDirty = false; }
  if (now - lastSay > 600 || sayT === 0) { say(); sayT = 1; lastSay = now; if (st.level === 4 || st.level === 6) maths(); }
  if (st.fly || baking || JOBS.length || (st.running && visible)) kick(); else readout(); }
new ResizeObserver(() => { resize(); kick(); }).observe(cv);
if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; if (visible) { lastT = performance.now(); kick(); } }, { threshold: 0.02 }).observe($('stage'));
document.addEventListener('visibilitychange', () => { if (!document.hidden) { lastT = performance.now(); kick(); } });
resize(); setLevel(0); readout(); maths();
for (let i = 1; i < NL; i++) request(i);
window.HFDVIEW = { st, flyTo, setZ, ready: () => bakesDone && !JOBS.length && !job, BAKES, MRI, frame, draw: () => { draw(performance.now()); drawRig(); } };
kick();
