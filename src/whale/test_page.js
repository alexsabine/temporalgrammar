/* Page bets B9b and B10, with Playwright. node test_page.js [screenshot dir] */
const { chromium, devices } = require('playwright'); const fs = require('fs'); const { spawn } = require('child_process'); const crypto = require('crypto');
const OUT = process.argv[2] || '/tmp/claude-0/-home-user/1edd5cc0-6987-5771-b57b-68d2ef3dbc39/scratchpad/whale';
const THREE = '/home/user/Phoenix-1-/MUSEUM/studio/vendor/three-0.158.0.min.js';
const res = {}; const verdict = (k, ok, msg) => { res[k] = ok; console.log((ok ? 'PASS ' : 'FAIL ') + k + '  ' + msg); };
(async () => {
  const srv = spawn('python3', ['-m', 'http.server', '8766', '--bind', '127.0.0.1'], { cwd: __dirname + '/../..', stdio: 'ignore' }); await new Promise(r => setTimeout(r, 700));
  const b = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
  const open = async (ctxOpts, test) => { const ctx = await b.newContext(ctxOpts); const p = await ctx.newPage(); const errs = [];
    p.on('pageerror', e => errs.push('pageerror: ' + e.message)); p.on('console', m => { if (m.type() === 'error' && !/ERR_FAILED|favicon/.test(m.text())) errs.push('console: ' + m.text().slice(0, 600)); });
    await p.route('**/three.min.js', r => r.fulfill({ body: fs.readFileSync(THREE, 'utf8'), contentType: 'application/javascript' })); await p.route(/fonts\.(googleapis|gstatic)\.com/, r => r.abort());
    if (test) await p.addInitScript(() => { window.__TEST = true; }); await p.goto('http://127.0.0.1:8766/whale.html', { waitUntil: 'load' }); return { p, errs, ctx }; };
  try {
    /* B9b: same frame, same picture */
    { const { p, errs } = await open({ viewport: { width: 960, height: 540 } }, true); const v = { cam: 'follow', yaw: -40 };
      const grab = async (f, vv) => { await p.evaluate(([f, vv]) => window.__clay.shot(f, vv), [f, vv]); return crypto.createHash('md5').update(await p.screenshot()).digest('hex'); };
      const a = await grab(300, v), a2 = await grab(300, v), c = await grab(900, v), a3 = await grab(300, v), d = await grab(300, { cam: 'under' }), e = await grab(300, v);
      const br = await grab(660, { cam: 'surface', event: 'breach', eventAt: 640 }), back = await grab(300, v);
      verdict('B9b', a === a2 && a === a3 && a === e && a === back && a !== c && a !== d, 'hashes 300:' + a.slice(0, 8) + ' again:' + a2.slice(0, 8) + ' after 900:' + a3.slice(0, 8) + ' after other view:' + e.slice(0, 8) + ' after a breach film:' + back.slice(0, 8) + ' (900 differs: ' + (a !== c) + ')');
      const plan = await p.evaluate(() => window.__clay.plan({ event: 'breach', eventAt: 640 }, 1500).concat(window.__clay.plan({ event: 'blow', eventAt: 1200 }, 1500)).filter(e => e.whale === 'mother').map(e => e.frame + ':' + e.type).join(' ')); console.log('   plan (breach asked to show at 640, blow at 1200): ' + plan);
      /* B11: layers */
      { const base = { cam: 'follow' }, g = (o) => grab(600, Object.assign({}, base, o)); const n0 = await g({}), n1 = await g({ layer: 'none' }), n2 = await g({ layer: 'warmth', layerAmt: 0 }), n3 = await g({});
        const h = {}; for (const L of ['links', 'sound', 'warmth', 'count']) h[L] = await g(Object.assign({ layer: L }, L === 'sound' ? { target: 'singer' } : L === 'links' ? { cam: 'school' } : {}));
        const sBase = await g({ target: 'singer' }), lBase = await g({ cam: 'school' }), half = await g({ layer: 'warmth', layerAmt: 0.5 }), again = await g({ layer: 'warmth' }), n4 = await g({});
        const differ = h.links !== lBase && h.sound !== sBase && h.warmth !== n0 && h.count !== n0;
        verdict('B11', n0 === n1 && n0 === n2 && n0 === n3 && n0 === n4 && differ && half !== n0 && half !== h.warmth && again === h.warmth, 'none ' + n0.slice(0, 8) + ' = layer none ' + n1.slice(0, 8) + ' = amount 0 ' + n2.slice(0, 8) + ' = after the layers ' + n4.slice(0, 8) + '; each layer differs: ' + differ + '; half differs from both: ' + (half !== n0 && half !== h.warmth) + '; warmth twice equal: ' + (again === h.warmth)); }
      /* B13 (page half): the label in the scene uses the panel's numbers */
      { const out = []; let ok = true; for (const T of [18, 26]) { const r = await p.evaluate(T => { WHALE.setSea(T, 35); window.__clay.shot(600, { cam: 'under', target: 'singer', layer: 'sound' }); const c = WHALE.channel(T, 35); return { label: WHALE.state().layer.soundLabel, want: 'slowest at ' + Math.round(c.depth).toLocaleString('en-GB') + ' m: ' + Math.round(c.speed).toLocaleString('en-GB') + ' m/s' }; }, T); out.push(T + ' C: "' + r.label + '"'); if (r.label !== r.want) ok = false; }
        await p.evaluate(() => WHALE.setSea(18, 35)); verdict('B13b', ok, out.join('; ')); }
      const f0 = await p.evaluate(() => WHALE.state().frame); await p.waitForTimeout(1000); const f1 = await p.evaluate(() => WHALE.state().frame);
      const vis = await p.evaluate(() => [...document.querySelectorAll('header,.dock,.read,.notes,.side,.say,.note')].filter(e => e.getClientRects().length).length);
      const song = await p.evaluate(() => ({ arr: Array.isArray(WHALE.song), n: WHALE.song.length, u0: WHALE.song[0], seq: WHALE.song.sequence.length, dur: WHALE.song.duration }));
      console.log('   WHALE.song: array ' + song.arr + ', ' + song.n + ' units, first ' + JSON.stringify(song.u0) + ', sequence ' + song.seq + ' onsets, ' + song.dur.toFixed(1) + ' s');
      res._test = { still: f0 === f1, vis, errs: errs.slice() }; }
    /* B10: the live page */
    const info = {};
    for (const [name, opts] of [['desktop', { viewport: { width: 1280, height: 800 } }], ['phone', devices['iPhone 13']]]) {
      const { p, errs } = await open(opts, false); await p.waitForTimeout(6000);
      await p.screenshot({ path: OUT + '/page_' + name + '.png' }); await p.screenshot({ path: OUT + '/page_' + name + '_full.png', fullPage: true });
      const m = await p.evaluate(() => ({ sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth, minBtn: Math.min(...[...document.querySelectorAll('button,.home,input[type=range]')].filter(e => e.getClientRects().length).map(e => e.getBoundingClientRect().height)), frame: WHALE.state().frame, depth: document.getElementById('rDepth').textContent, song: document.getElementById('rSong').textContent, ocean: document.getElementById('oceanP').textContent }));
      /* press things */
      for (const v of ['links', 'sound', 'warmth', 'count']) { await p.locator('input[name=show][value=' + v + ']').check({ force: true }); await p.waitForTimeout(1500); if (name === 'desktop' || v === 'warmth') await p.locator('.stage').screenshot({ path: OUT + '/live_' + name + '_' + v + '.png' }); }
      m.cap = await p.evaluate(() => document.getElementById('cap').textContent.slice(0, 40)); m.minShow = await p.evaluate(() => Math.min(...[...document.querySelectorAll('.show span')].map(e => e.getBoundingClientRect().height)));
      await p.locator('input[name=show][value=none]').check({ force: true });
      for (const sel of ['[data-cam=follow]', '[data-cam=surface]', '[data-cam=above]', '[data-cam=school]', '#bBlow', '#snd', '#src', '#pause', '#pause']) { await p.click(sel); await p.waitForTimeout(500); }
      await p.evaluate(() => { const t = document.getElementById('temp'); t.value = 4; t.dispatchEvent(new Event('input')); }); await p.waitForTimeout(300);
      m.cold = await p.evaluate(() => document.getElementById('oceanP').textContent); m.audio = await p.evaluate(() => document.getElementById('snd').getAttribute('aria-pressed'));
      await p.locator('.side').first().screenshot({ path: OUT + '/side_' + name + '.png' });
      info[name] = { m, errs }; console.log('   ' + name + ': ' + JSON.stringify(m)); if (errs.length) console.log('   errors: ' + errs.join(' | '));
    }
    const t = res._test; delete res._test;
    verdict('B10', !info.desktop.errs.length && !info.phone.errs.length && !t.errs.length && info.phone.m.sw <= info.phone.m.cw && info.phone.m.minBtn >= 44 && info.desktop.m.minBtn >= 44 && info.phone.m.minShow >= 44 && t.still && t.vis === 0,
      'errors desktop/phone/film ' + info.desktop.errs.length + '/' + info.phone.errs.length + '/' + t.errs.length + '; phone scroll width ' + info.phone.m.sw + ' against ' + info.phone.m.cw + '; smallest control ' + info.phone.m.minBtn + ' px (phone), ' + info.desktop.m.minBtn + ' px (desktop); film mode frame counter still: ' + t.still + '; page furniture visible in film mode: ' + t.vis);
    if (t.errs.length) console.log('   film errors: ' + t.errs.join(' | '));
  } finally { await b.close(); srv.kill(); }
  console.log(JSON.stringify(res));
})();
