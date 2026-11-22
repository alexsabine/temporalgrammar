/* Page tests for equanimity.html (bet B11 in BETS.md). Needs the repo served: python3 -m http.server 8765 in the repo root.
   Run: node test_page.js [outDir] */
const { chromium, devices } = require('playwright'); const fs = require('fs'), crypto = require('crypto');
const OUT = process.argv[2] || '/tmp/claude-0/-home-user/1edd5cc0-6987-5771-b57b-68d2ef3dbc39/scratchpad/equanimity', URL = 'http://localhost:8765/equanimity.html';
let pass = 0, fail = 0; const bet = (n, ok, d) => { ok ? pass++ : fail++; console.log((ok ? 'PASS  ' : 'FAIL  ') + n + (d ? '   ' + d : '')); };
const ARGS = ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'];
async function open(b, ctxOpt, init) { const ctx = await b.newContext(ctxOpt), p = await ctx.newPage(), errs = [];
  await p.route('**/three.min.js', r => r.fulfill({ body: fs.readFileSync('/home/user/Phoenix-1-/MUSEUM/studio/vendor/three-0.158.0.min.js', 'utf8'), contentType: 'application/javascript' }));
  await p.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
  p.on('pageerror', e => errs.push('pageerror: ' + e.message));
  p.on('console', m => { if (m.type() === 'error' && !/ERR_FAILED|favicon/.test(m.text())) errs.push('console: ' + m.text().slice(0, 300)); });   /* the aborted font requests are ours */
  if (init) await p.addInitScript(init); await p.goto(URL); return { p, errs, ctx }; }
(async () => {
  const b = await chromium.launch({ args: ARGS });
  /* desktop */
  { const { p, errs } = await open(b, { viewport: { width: 1280, height: 800 } }); await p.waitForTimeout(9000);
    for (const id of ['E', 'F', 'C', 'D']) { await p.click(`#scenes button:has-text("${id === 'E' ? 'Carried' : id === 'F' ? 'Two who' : id === 'C' ? 'The count' : 'Bell and'}")`); await p.waitForTimeout(6000); if (id === 'E') { await p.click('#ctlEp button:has-text("Too heavy")'); await p.waitForTimeout(9000); await p.screenshot({ path: OUT + '/page_desktop_E.png', fullPage: false }); } }
    await p.screenshot({ path: OUT + '/page_desktop.png', fullPage: true });
    bet('B11 desktop: no page or console errors', errs.length === 0, errs.join(' | ')); await p.close(); }
  /* phone */
  { const { p, errs } = await open(b, { ...devices['iPhone 13'] }); await p.waitForTimeout(9000);
    await p.click('#scenes button:has-text("Carried")'); await p.waitForTimeout(9000);
    const m = await p.evaluate(() => { const small = []; document.querySelectorAll('.btn, .home, input[type=range]').forEach(e => { const r = e.getBoundingClientRect(); if (r.width && r.height < 43.5) small.push((e.id || e.textContent.trim().slice(0, 20)) + ' ' + r.height.toFixed(1)); });
      return { sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, small }; });
    bet('B11 phone: no sideways scroll at 390 px', m.sw <= m.cw, `scrollWidth ${m.sw} clientWidth ${m.cw}`);
    bet('B11 phone: controls at least 44 px high', m.small.length === 0, m.small.join(', '));
    await p.screenshot({ path: OUT + '/page_phone.png', fullPage: true });
    bet('B11 phone: no page or console errors', errs.length === 0, errs.join(' | ')); await p.close(); }
  /* reduced motion */
  { const { p, errs } = await open(b, { viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce' }); await p.waitForTimeout(5000); await p.click('#scenes button:has-text("Carried")'); await p.waitForTimeout(5000);
    const a = await p.evaluate(() => EQ.state().frame); await p.waitForTimeout(2500); const c = await p.evaluate(() => EQ.state().frame);
    bet('reduced motion: nothing moves until asked', a === c, `frame ${a} then ${c}`); bet('reduced motion: no errors', errs.length === 0, errs.join(' | ')); await p.close(); }
  /* film mode */
  { const { p, errs } = await open(b, { viewport: { width: 1280, height: 720 } }, () => { window.__TEST = { budget: 1280 * 720 }; }); await p.waitForFunction(() => window.__clay, null, { timeout: 60000 });
    const shot = async (f, v) => { await p.evaluate(([f, v]) => window.__clay.shot(f, v), [f, v]); return crypto.createHash('sha1').update(await p.screenshot()).digest('hex'); };
    const v1 = { scene: 'E', preset: 'heavy', yaw: .1 }, v2 = { scene: 'D', width: { from: 3, to: .3, start: 0, end: 120 }, card: .5 };
    const a1 = await shot(333, v1), a2 = await shot(333, v1), d1 = await shot(60, v2), f1 = await shot(500, { scene: 'F', preset: 'letgo' }), c1 = await shot(250, { scene: 'C' }), a3 = await shot(333, v1), d2 = await shot(60, v2);
    bet('B11 film: same frame and view twice running give identical pixels', a1 === a2, a1.slice(0, 10) + ' ' + a2.slice(0, 10));
    bet('B11 film: identical again after other scenes were drawn in between', a1 === a3 && d1 === d2, a3.slice(0, 10) + ' / ' + d1.slice(0, 10) + ' ' + d2.slice(0, 10));
    bet('B11 film: different views give different pixels', new Set([a1, d1, f1, c1]).size === 4);
    await p.evaluate(() => window.__clay.shot(4800, { scene: 'E', preset: 'balance', card: 0 })); const st = await p.evaluate(() => EQ.state());
    bet('B11 film: EQ.state at balance', st.weightGiven.toFixed(6) === '0.618034' && st.uncertaintyBefore.toFixed(6) === '1.618034' && st.uncertaintyAfter.toFixed(6) === '0.618034', JSON.stringify({ given: st.weightGiven, before: st.uncertaintyBefore, after: st.uncertaintyAfter, mse: st.mse, surprise: st.surprise, path: st.pathRatio }));
    const furn = await p.evaluate(() => { const c = document.getElementById('bench').getBoundingClientRect(); return { w: c.width, h: c.height, hdr: getComputedStyle(document.querySelector('header')).display, scroll: document.documentElement.scrollHeight }; });
    bet('B11 film: canvas fills the viewport, no page furniture', furn.w === 1280 && furn.h === 720 && furn.hdr === 'none' && furn.scroll <= 720, JSON.stringify(furn));
    bet('B11 film: no page or console errors', errs.length === 0, errs.join(' | ')); await p.close(); }
  await b.close(); console.log(`\n${pass} passed, ${fail} failed`); process.exit(fail ? 1 : 0);
})();
