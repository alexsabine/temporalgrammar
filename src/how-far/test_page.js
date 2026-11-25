// Page checks for how-far-does-it-go.html (bets G1 to G8 in BETS.md). Run: node src/how-far/test_page.js > src/how-far/test_output_page.txt
const path = require('path'), { chromium } = require('playwright');
const file = path.join(__dirname, '..', '..', 'how-far-does-it-go.html'), url = 'file://' + file;
let fail = 0, n = 0; const ok = (name, cond, detail = '') => { n++; console.log((cond ? 'PASS ' : 'FAIL ') + name + (detail !== '' ? '  ' + detail : '')); if (!cond) fail++; };
const note = (name, detail) => console.log('NOTE ' + name + '  ' + detail);
(async () => {
  console.log('Page checks, how-far-does-it-go.html, run ' + new Date().toISOString().slice(0, 10) + ' (headless Chromium, software drawing; the web fonts are answered with an empty file)');
  const b = await chromium.launch();
  const open = async (opt) => { const ctx = await b.newContext(opt), p = await ctx.newPage(), errs = []; p.on('pageerror', e => errs.push('pageerror: ' + e.message)); p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
    await p.route(/fonts\.(googleapis|gstatic)\.com/, r => r.fulfill({ status: 200, contentType: 'text/css', body: '' })); const t0 = Date.now(); await p.goto(url); return { p, errs, t0, ctx }; };
  const settle = async (p, max = 90000) => { const t0 = Date.now(); while (Date.now() - t0 < max) { await p.waitForTimeout(250); if (await p.evaluate(() => EVVIEW.ready() && !EVVIEW.st.fly)) return Date.now() - t0; } return -1; };
  const lit = p => p.evaluate(() => { const c = document.getElementById('view'), g = c.getContext('2d'), w = Math.round(c.width * 0.8), d = g.getImageData(0, 60, w, c.height - 120).data; let s = 0, m = 0; for (let k = 0; k < d.length; k += 4 * 97) { s += d[k] + d[k + 1] + d[k + 2]; m++; } return s / m / 3; });

  // ── desktop ──
  { const { p, errs, t0 } = await open({ viewport: { width: 1280, height: 800 } });
    let first = -1; for (let k = 0; k < 80; k++) { await p.waitForTimeout(100); if (await lit(p) > 6) { first = Date.now() - t0; break; } }
    ok('G8 a picture is on the stage within 4 s of opening (bet: under 4,000 ms)', first > 0 && first < 4000, first + ' ms');
    await p.waitForTimeout(Math.max(0, 3000 - (Date.now() - t0))); const early = await p.evaluate(() => EVVIEW.baked());
    ok('G3a three seconds after opening no more than 4 levels hold pictures (bet: ≤ 4 of 28)', early.length <= 4, 'levels ' + JSON.stringify(early));
    const s0 = await settle(p); note('', 'first settle after ' + s0 + ' ms more');
    // every waypoint by its menu entry
    const ids = await p.evaluate(() => EV.LEVELS.map(L => L.id)); let titles = true, worstMs = 0, dark = [], maxHeld = 0; const names = [];
    for (const id of ids) { await p.selectOption('#way', String(id)); const ms = await settle(p); worstMs = Math.max(worstMs, ms); const r = await p.evaluate(i => ({ t: document.getElementById('lvtitle').textContent, want: EV.LV(i).name, lv: EVVIEW.st.level, held: EVVIEW.baked().length, cap: document.getElementById('cap').textContent.length, say: document.getElementById('say').textContent.length, facts: document.getElementById('facts').children.length }), id);
      if (r.t !== r.want || r.lv !== id || r.cap < 20 || r.say < 20 || r.facts < 4 || ms < 0) { titles = false; names.push(id + ':' + JSON.stringify(r)); } maxHeld = Math.max(maxHeld, r.held); const L = await lit(p); if (L < 2.5) dark.push(id + ' (' + L.toFixed(1) + ')'); }
    ok('G4 all 28 waypoints reached from the menu; title, caption, live sentence and facts present for each', titles, names.join(' | ') || 'longest wait for sharp pictures ' + worstMs + ' ms');
    ok('G4b no waypoint leaves the stage black (bet: mean brightness above 2.5 of 255 at every one)', dark.length === 0, dark.join(', '));
    // fly from the Planck length to the horizon
    await p.evaluate(() => EVVIEW.flyTo(-15)); let peak = 0; for (let k = 0; k < 60; k++) { await p.waitForTimeout(200); peak = Math.max(peak, await p.evaluate(() => EVVIEW.baked().length)); if (!(await p.evaluate(() => !!EVVIEW.st.fly))) break; } await settle(p);
    ok('G3b flying from the Planck length to the horizon, never more than 6 levels hold pictures at once (bet: ≤ 6)', Math.max(peak, maxHeld) <= 6, 'peak in flight ' + peak + ', peak when stepping ' + maxHeld);
    // numbers on screen against the model
    await p.selectOption('#way', '-6'); await settle(p);
    const v = await p.evaluate(() => ({ fov: document.getElementById('rFov').textContent, want: EV.fmtLen(EV.fovOf(EVVIEW.st.z), 3), light: document.getElementById('rLight').textContent, lwant: EV.fmtTime(EV.COSMO.lookbackOfChi(EV.fovOf(EVVIEW.st.z) / 2)), facts: document.getElementById('facts').textContent, au: EV.voyager('v1', Date.now()).au }));
    ok('G5 frame width, light-time and Voyager 1 distance on screen equal the model\'s', v.fov === v.want && v.light === v.lwant && v.facts.includes(v.au.toFixed(2) + ' AU'), v.fov + ' | ' + v.light + ' | ' + v.au.toFixed(2) + ' AU');
    // keyboard
    await p.selectOption('#way', '-1'); await settle(p); await p.focus('#view'); const z0 = await p.evaluate(() => EVVIEW.st.z); await p.keyboard.press('ArrowDown'); await p.keyboard.press('ArrowDown'); const z1 = await p.evaluate(() => EVVIEW.st.z); await p.keyboard.press('ArrowUp'); const z2 = await p.evaluate(() => EVVIEW.st.z);
    /* first version pressed Page Up twice and expected to be back at the person; one press is the way back (harness fault, kept in test_output_page_v1_harness_fault.txt) */
    await p.keyboard.press('PageDown'); await settle(p); const l1 = await p.evaluate(() => EVVIEW.st.level); await p.keyboard.press('PageUp'); await settle(p); const l2 = await p.evaluate(() => EVVIEW.st.level);
    ok('G7 arrow keys change the zoom; Page Down and Page Up move between waypoints', z1 > z0 && z2 < z1 && l1 === 0 && l2 === -1, 'z ' + z0.toFixed(4) + ' → ' + z1.toFixed(4) + ' → ' + z2.toFixed(4) + '; levels ' + l1 + ', ' + l2);
    // wheel and the gauge
    const box = await p.locator('#view').boundingBox(); await p.mouse.move(box.x + box.width * 0.4, box.y + box.height * 0.5); const zw0 = await p.evaluate(() => EVVIEW.st.z); await p.mouse.wheel(0, 120); await p.waitForTimeout(100); const zw1 = await p.evaluate(() => EVVIEW.st.z);
    await p.mouse.click(box.x + box.width - 20, box.y + box.height * 0.5); await p.waitForTimeout(150); const zg = await p.evaluate(() => EVVIEW.st.z);
    ok('G7b the mouse wheel zooms when the picture has focus, and a press on the gauge jumps there', zw1 > zw0 && Math.abs(zg - 0.5) < 0.08, 'wheel ' + (zw1 - zw0).toFixed(4) + ', gauge z = ' + zg.toFixed(3));
    // every control and every try-this button
    const tries = await p.evaluate(() => [...document.querySelectorAll('[data-try]')].map(b => b.dataset.try)); for (const t of tries) { await p.click('[data-try="' + t + '"]'); await p.waitForTimeout(300); } await settle(p);
    for (const [lv, sel] of [[-4, '#flash'], [10, '#again'], [12, '[data-cand="1"]'], [12, '[data-cand="2"]'], [12, '[data-cand="0"]'], [-15, '#h0b'], [-15, '#h0a']]) { await p.selectOption('#way', String(lv)); await settle(p); await p.click(sel); await p.waitForTimeout(300); } await settle(p);
    const tb = await p.evaluate(() => ({ rows: document.getElementById('wayBody').children.length, live: [...document.querySelectorAll('.eq .live, td .live')].filter(e => e.textContent.trim().length < 8).map(e => e.id) }));
    ok('G9 (added) every try-this button and every new control was pressed; the ladder table has 28 rows and every live line under an equation is filled', tb.rows === 28 && tb.live.length === 0, tries.length + ' buttons; empty: ' + JSON.stringify(tb.live));
    ok('G1a desktop 1280×800: no page errors and no console errors', errs.length === 0, errs.slice(0, 4).join(' | ')); await p.close(); }

  // ── phone ──
  { const { p, errs } = await open({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }); await settle(p);
    const w = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, small: [...document.querySelectorAll('button, select, input[type=range], summary')].filter(e => e.offsetParent && (e.getBoundingClientRect().height < 43.5)).map(e => (e.id || e.textContent.slice(0, 12)) + ':' + Math.round(e.getBoundingClientRect().height)) }));
    ok('G2 at 390 px the document is no wider than the window', w.sw <= w.iw, w.sw + ' ≤ ' + w.iw);
    ok('G2b (added) every visible button, menu and slider is at least 44 px tall', w.small.length === 0, w.small.join(', '));
    for (const id of [-15, -9, -2, 3, 9, 12]) { await p.selectOption('#way', String(id)); await settle(p); const s = await p.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth); if (!s) errs.push('wider than the window at ' + id); }
    ok('G1b phone 390×844: no page errors, no console errors, and no sideways scroll at six waypoints', errs.length === 0, errs.slice(0, 4).join(' | ')); await p.close(); }

  // ── reduced motion ──
  { const { p, errs } = await open({ viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce' }); await settle(p);
    const r = await p.evaluate(() => ({ running: EVVIEW.st.running, btn: document.getElementById('run').textContent })), L = await lit(p); await p.click('#next'); await settle(p); const lv = await p.evaluate(() => EVVIEW.st.level);
    ok('G6 with reduced motion requested the page starts paused, shows a picture, and "In" still moves one waypoint', r.running === false && r.btn === 'Play' && L > 6 && lv === 0 && errs.length === 0, 'brightness ' + L.toFixed(1) + ', level after In: ' + lv); await p.close(); }
  await b.close(); console.log('\n' + (n - fail) + ' of ' + n + ' checks passed' + (fail ? ', ' + fail + ' FAILED' : '')); process.exit(fail ? 1 : 0);
})();
