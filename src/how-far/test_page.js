// Page checks for how-far-does-it-go.html (bets G1 to G8 and, for the section on time, G10 to G15 in BETS.md). Run: node src/how-far/test_page.js > src/how-far/test_output_page.txt
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
    const tb = await p.evaluate(() => ({ rows: document.getElementById('wayBody').children.length, live: [...document.querySelectorAll('.eq .live, td .live, li .live, .box .live')].filter(e => e.textContent.trim().length < 8).map(e => e.id) }));
    ok('G9 (added) every try-this button and every new control was pressed; the ladder table has 28 rows and every live line under an equation is filled', tb.rows === 28 && tb.live.length === 0, tries.length + ' buttons; empty: ' + JSON.stringify(tb.live));
    // ── the section on time (bets G10, G12, G14, G15 of the second set) ──
    { await p.selectOption('#way', '-13'); await settle(p); await p.waitForTimeout(500);
      const g15 = await p.evaluate(() => ({ follow: EVTIME.tm.follow, z: 1 / EVTIME.tm.a - 1, want: EV.COSMO.zOfChi(EV.fovOf(EVVIEW.st.z) / 2), ring: document.getElementById('tLoopRing').getAttribute('cx') }));
      ok('G15 at the first-galaxies stop, with "follow the zoom" on, the marker\'s redshift equals the redshift of the frame\'s edge', g15.follow && Math.abs(g15.z - g15.want) < 1e-9 * g15.want && +g15.ring > 0, 'z = ' + g15.z.toFixed(4) + ' and ' + g15.want.toFixed(4));
      await p.click('#tNow'); const g14 = await p.evaluate(() => { const T = EV.TIME, f = (x, d) => x.toFixed(d), s = document.getElementById('tSig'); s.value = '-200'; s.dispatchEvent(new Event('input')); const t = id => document.getElementById(id).textContent;
        return { C: t('tC'), Cw: f(T.now.Ch, 3) + ' of ' + f(T.whole.Ch, 3), Cs: t('tCs'), Csw: f(T.now.CsOmega, 3) + ' of 1', S: t('tS'), Sw: f(T.now.Sh, 3), cut: t('tCut'), cutw: f(T.LEG.cutTime(0.01) / EV.K.Gyr, 1) + ' billion years', follow: EVTIME.tm.follow }; });
      ok('G14 numbers shown equal the model\'s: C, C* and S at "now", and the cut time at σ = 0.01', g14.C === g14.Cw && g14.Cs === g14.Csw && g14.S === g14.Sw && g14.cut === g14.cutw && g14.follow === false, [g14.C, g14.Cs, g14.S, g14.cut].join(' | '));
      // keyboard: Tab order through the new controls, arrow keys on the two sliders, Enter on every fold-out
      await p.focus('#tA'); const a0 = await p.evaluate(() => document.getElementById('tAOut').textContent); await p.keyboard.press('ArrowLeft'); await p.keyboard.press('ArrowLeft'); const a1 = await p.evaluate(() => document.getElementById('tAOut').textContent);
      const order = []; for (let k = 0; k < 3; k++) { await p.keyboard.press('Tab'); order.push(await p.evaluate(() => document.activeElement.id)); }
      await p.focus('#tSig'); const c0 = await p.evaluate(() => document.getElementById('tCut').textContent); await p.keyboard.press('ArrowLeft'); const c1 = await p.evaluate(() => document.getElementById('tCut').textContent);
      const nSum = await p.locator('#crr summary').count(); let opened = 0; for (let k = 0; k < nSum; k++) { await p.locator('#crr summary').nth(k).focus(); await p.keyboard.press('Enter'); if (await p.evaluate(i => document.querySelectorAll('#crr details')[i].open, k)) opened++; }
      const tabbable = await p.evaluate(() => [...document.querySelectorAll('#crr button, #crr input, #crr summary')].filter(e => e.tabIndex < 0 || e.disabled).length);
      ok('G10 every new control can be reached with the Tab key; arrow keys on the two sliders change the numbers; Enter opens every fold-out', a1 !== a0 && c1 !== c0 && order.join() === 'tPlay,tNow,tFollow' && opened === nSum && nSum >= 5 && tabbable === 0, 'size: "' + a0 + '" → "' + a1 + '"; cut: ' + c0 + ' → ' + c1 + '; Tab order ' + order.join(', ') + '; ' + opened + ' of ' + nSum + ' fold-outs opened');
      // use everything
      /* first version expected play to begin again from the start; it goes on from where the marker is, unless that is now or the end (harness fault, kept in test_output_page_v2_time_harness_faults.txt) */
      const before = await p.evaluate(() => EVTIME.tm.a); await p.click('#tPlay'); await p.waitForTimeout(700); const mid = await p.evaluate(() => EVTIME.tm.a); await p.click('#tPlay'); await p.click('#tFollow'); await p.click('#tFollow'); await p.click('#tNow');
      for (const v of ['-600', '-150', '-30']) await p.evaluate(x => { const s = document.getElementById('tSig'); s.value = x; s.dispatchEvent(new Event('input')); }, v);
      for (const v of ['-1400', '-700', '500', '0']) await p.evaluate(x => { const s = document.getElementById('tA'); s.value = x; s.dispatchEvent(new Event('input')); }, v);
      const figs = await p.evaluate(() => ['tLoop', 'tOct', 'tRes', 'tHyd'].map(id => document.getElementById(id).querySelectorAll('svg path, svg line').length).concat([document.getElementById('tClocks').children.length, document.getElementById('tStopBody').children.length]));
      ok('G12a desktop: every control of the section on time was used; four figures, four clocks and 28 rows are drawn (errors are counted in G1a)', mid > before && figs.slice(0, 4).every(x => x > 8) && figs[4] === 4 && figs[5] === 28, 'play went from a = ' + before.toFixed(3) + ' to ' + mid.toFixed(3) + ' in 0.7 s' + '; drawn: ' + figs.join(', ')); }
    ok('G1a desktop 1280×800: no page errors and no console errors', errs.length === 0, errs.slice(0, 4).join(' | ')); await p.close(); }

  // ── phone ──
  { const { p, errs } = await open({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }); await settle(p);
    const w = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, small: [...document.querySelectorAll('button, select, input[type=range], summary')].filter(e => e.offsetParent && (e.getBoundingClientRect().height < 43.5)).map(e => (e.id || e.textContent.slice(0, 12)) + ':' + Math.round(e.getBoundingClientRect().height)) }));
    ok('G2 at 390 px the document is no wider than the window', w.sw <= w.iw, w.sw + ' ≤ ' + w.iw);
    ok('G2b (added) every visible button, menu and slider is at least 44 px tall', w.small.length === 0, w.small.join(', '));
    { await p.evaluate(() => document.querySelectorAll('#crr details').forEach(d => { d.open = true; })); await p.locator('#tFollow').scrollIntoViewIfNeeded(); await p.tap('#tFollow'); await p.tap('#tNow'); await p.tap('#tPlay'); await p.waitForTimeout(400); await p.tap('#tPlay');
      for (const v of ['-600', '-30', '-200']) await p.evaluate(x => { const s = document.getElementById('tSig'); s.value = x; s.dispatchEvent(new Event('input')); }, v);
      for (const v of ['-1400', '500', '0']) await p.evaluate(x => { const s = document.getElementById('tA'); s.value = x; s.dispatchEvent(new Event('input')); }, v); await p.waitForTimeout(200);
      const t = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, iw: window.innerWidth, small: [...document.querySelectorAll('#crr button, #crr input[type=range], #crr summary')].filter(e => e.offsetParent && e.getBoundingClientRect().height < 43.5).map(e => (e.id || e.textContent.slice(0, 12)) + ':' + Math.round(e.getBoundingClientRect().height)), n: document.querySelectorAll('#crr button, #crr input[type=range], #crr summary').length, wide: [...document.querySelectorAll('#crr svg')].filter(e => e.getBoundingClientRect().right > window.innerWidth + 0.5).length }));
      ok('G11 at 390 px, with every fold-out of the section on time open: the document is no wider than the window, no figure runs off it, and every new control is at least 44 px tall', t.sw <= t.iw && t.small.length === 0 && t.wide === 0 && t.n >= 10, t.sw + ' ≤ ' + t.iw + '; ' + t.n + ' controls; too small: ' + JSON.stringify(t.small)); }
    for (const id of [-15, -9, -2, 3, 9, 12]) { await p.selectOption('#way', String(id)); await settle(p); const s = await p.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth); if (!s) errs.push('wider than the window at ' + id); }
    ok('G1b phone 390×844: no page errors, no console errors (G12 included: the controls of the section on time were used), and no sideways scroll at six waypoints', errs.length === 0, errs.slice(0, 4).join(' | ')); await p.close(); }

  // ── reduced motion ──
  { const { p, errs } = await open({ viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce' }); await settle(p);
    const r = await p.evaluate(() => ({ running: EVVIEW.st.running, btn: document.getElementById('run').textContent })), L = await lit(p); await p.click('#next'); await settle(p); const lv = await p.evaluate(() => EVVIEW.st.level);
    /* first version expected the second press to go later still; from the far future the next moment is the beginning again (harness fault, same file) */
    { const r0 = await p.evaluate(() => ({ btn: document.getElementById('tPlay').textContent, a: EVTIME.tm.a, figs: ['tLoop', 'tOct', 'tRes', 'tHyd'].every(id => document.getElementById(id).querySelector('svg')) })); await p.click('#tPlay'); const a1 = await p.evaluate(() => EVTIME.tm.a); await p.waitForTimeout(600); const a2 = await p.evaluate(() => EVTIME.tm.a); await p.click('#tPlay'); const a3 = await p.evaluate(() => EVTIME.tm.a);
      ok('G13 with reduced motion requested the new figures are drawn and nothing in the section on time moves by itself: the play button becomes "Next moment" and goes one step for one press', r0.btn === 'Next moment' && r0.figs && a1 !== r0.a && a2 === a1 && a3 !== a1 && errs.length === 0, 'a: ' + [r0.a, a1, a2, a3].map(x => x.toExponential(2)).join(' → ')); }
    ok('G6 with reduced motion requested the page starts paused, shows a picture, and "In" still moves one waypoint', r.running === false && r.btn === 'Play' && L > 6 && lv === 0 && errs.length === 0, 'brightness ' + L.toFixed(1) + ', level after In: ' + lv); await p.close(); }
  await b.close(); console.log('\n' + (n - fail) + ' of ' + n + ' checks passed' + (fail ? ', ' + fail + ' FAILED' : '')); process.exit(fail ? 1 : 0);
})();
