/* Runs the bets in BETS.md that need no drawing. node test_model.js > test_output_vN.txt */
const M = require('./model.js');
const out = (...a) => console.log(...a);
const res = {};
function verdict(k, ok, msg) { res[k] = ok; out((ok ? 'PASS ' : 'FAIL ') + k + '  ' + msg); }
// B1
{ const c = M.soundSpeed(25, 35, 1000); verdict('B1', Math.abs(c - 1550.744) < 0.01, 'c(25,35,1000) = ' + c.toFixed(4) + ' m/s, check value 1550.744'); }
// B2
{ const a = M.channel(18, 35); verdict('B2a', a.depth >= 600 && a.depth <= 1400, 'channel at ' + a.depth + ' m, ' + a.speed.toFixed(1) + ' m/s; surface ' + a.surface.toFixed(1));
  const b = M.channel(2, 35), b0 = M.channel(0, 35); verdict('B2b', b.depth === 0 && b0.depth === 0, 'surface 2 C: ' + b.depth + ' m; 0 C: ' + b0.depth + ' m');
  let ok = true, prev = -1, list = []; for (let T = 10; T <= 28; T += 2) { const d = M.channel(T, 35).depth; list.push(T + 'C:' + d); if (d <= prev) ok = false; prev = d; } verdict('B2c', ok, list.join(' '));
  const r = M.rays(18, 35, a.depth, [-6, 6], 150000), r2 = M.rays(18, 35, 20, [6], 150000);
  verdict('B2d', r.every(q => q.top === 0 && q.bot === 0) && r2[0].top + r2[0].bot >= 1 && r2[0].top >= 1, 'axis rays top/bottom hits ' + r.map(q => q.top + '/' + q.bot).join(' ') + '; 20 m ray ' + r2[0].top + '/' + r2[0].bot);
  out('   salinity: channel at S=30 ' + M.channel(18, 30).depth + ' m, S=38 ' + M.channel(18, 38).depth + ' m'); }
// B3
{ let ok = true, lo = 1e9, hi = 0; for (const u of M.UNITS) { if (u.dur < 0.5 || u.dur > 4) ok = false; for (const p of u.contour) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); if (p[1] < 100 || p[1] > 4000) ok = false; } }
  verdict('B3a', ok, M.UNITS.length + ' units, pitch ' + lo + ' to ' + hi + ' Hz, song ' + M.SONG.duration.toFixed(1) + ' s, ' + M.SONG.sequence.length + ' unit onsets');
  const S = M.makeSim({ nFish: 0, auto: false }); const got = []; const n3 = M.SONG.frames * 3 + 10; const themes = [];
  for (let f = 0; f < n3; f++) { S.step(); for (const e of S.ev) if (e.type === 'unit') { got.push(e); if (!themes.length || themes[themes.length - 1] !== e.theme) themes.push(e.theme); } }
  let same = true, firstBad = -1; for (let i = 0; i < M.SONG.sequence.length; i++) { const a = M.SONG.sequence[i], b = got[i]; if (!b || b.frame !== a.frame || b.unit !== a.unit) { same = false; firstBad = i; break; } }
  verdict('B3b', same, same ? 'all ' + M.SONG.sequence.length + ' onsets equal' : 'first difference at onset ' + firstBad + ': ' + JSON.stringify(got[firstBad]) + ' vs ' + JSON.stringify(M.SONG.sequence[firstBad]));
  verdict('B3c', themes.join('').startsWith('ABCDABCDABCD'), 'theme order ' + themes.join('')); }
// B4
{ const S = M.makeSim({ nFish: 0, auto: false }); const blows = []; let okAbove = true, maxGap = 0, last = null, series = [], cur = 0;
  for (let f = 0; f < 1200 * 60; f++) { S.step(); for (const e of S.ev) if (e.type === 'blow' && e.whale === 'mother') { blows.push(e); if (e.y <= 0) okAbove = false; if (last !== null) maxGap = Math.max(maxGap, (e.frame - last) / 60); last = e.frame; if (e.n === 1 && cur) { series.push(cur); cur = 0; } cur = e.n; } }
  const d = S.dives, mean = d.reduce((a, b) => a + b, 0) / d.length, sd = Math.sqrt(d.reduce((a, b) => a + (b - mean) * (b - mean), 0) / d.length);
  verdict('B4', okAbove && series.every(n => n >= 2) && maxGap <= 150 && blows.length > 6, blows.length + ' blows, blowholes above water at all: ' + okAbove + ', blows per surfacing ' + series.join(',') + ', longest gap ' + maxGap.toFixed(1) + ' s');
  out('   reported: ' + d.length + ' dive cycles, mean ' + mean.toFixed(1) + ' s, CV ' + (sd / mean).toFixed(3) + ' (candidate 0.159; nothing tuned). cycles: ' + d.map(x => x.toFixed(0)).join(' '));
  const oth = { calf: 0, singer: 0 }; for (const e of S.log) if (e.type === 'blow' && oth[e.whale] !== undefined) oth[e.whale]++; out('   calf blows ' + oth.calf + ', singer blows ' + oth.singer); }
// B5
{ const S = M.makeSim({ nFish: 0, auto: false, events: [{ event: 'breach', at: 600 }] }); let peak = 0, exit = null, spl = null, blowAfter = null;
  for (let f = 0; f < 600 + 90 * 60; f++) { S.step(); const s = S.snout(S.mother); if (S.mother.state === 'air' || S.mother.state === 'br_run') peak = Math.max(peak, s[1]);
    for (const e of S.ev) { if (e.type === 'exit') exit = e; if (e.type === 'splash') spl = e; if (e.type === 'blow' && e.whale === 'mother' && spl && blowAfter === null) blowAfter = (e.frame - spl.frame) / 60; } }
  verdict('B5', peak >= 4 && exit && exit.v >= 6 && exit.v <= 9 && spl && (spl.frame - 600) / 60 <= 20 && blowAfter !== null && blowAfter <= 40,
    'snout peak ' + peak.toFixed(2) + ' m, exit speed ' + (exit ? exit.v.toFixed(2) : '-') + ' m/s at frame ' + (exit ? exit.frame : '-') + ', splash ' + (spl ? ((spl.frame - 600) / 60).toFixed(1) : '-') + ' s after request, blow ' + (blowAfter === null ? '-' : blowAfter.toFixed(1)) + ' s after splash'); }
// B6
{ const S = M.makeSim({ nFish: 0, auto: false, events: [{ event: 'blow', at: 120 }] }); let b = null, top = 0, wid = 0;
  for (let f = 0; f < 60 * 60 && (!b || f < b.frame + 240); f++) { S.step(); for (const e of S.ev) if (e.type === 'blow' && e.whale === 'mother' && !b) b = e;
    if (b) { const P = S.spray; let mx = 0, lo = 1e9, hi = -1e9; const rx = -Math.sin(S.mother.yaw), rz = Math.cos(S.mother.yaw);
      for (let i = 0; i < P.n; i++) if (P.life[i] > 0 && P.kind[i] === 1) { const h = P.y[i] - b.y; if (h > mx) mx = h; const q = (P.x[i] - b.x) * rx + (P.z[i] - b.z) * rz; if (q < lo) lo = q; if (q > hi) hi = q; }
      if (mx > top) { top = mx; wid = hi - lo; } } }
  verdict('B6', b && top >= 2 && top <= 4 && wid > top / 3, 'blow at frame ' + (b ? b.frame : '-') + ': top ' + top.toFixed(2) + ' m above the blowholes, ' + wid.toFixed(2) + ' m wide'); }
// B7
{ const S = M.makeSim({ auto: false }); const F = S.fish; let pol = 0, np = 0, nn = [], close = 0, tot = 0;
  for (let f = 0; f < 600 * 60; f++) { S.step(); if (f > 3600) { pol += S.school.pol; np++; }
    if (f % 600 === 599) { for (let i = 0; i < S.NF; i += 4) { let b = 1e9; for (let j = 0; j < S.NF; j++) if (j !== i) { const d = Math.hypot(F.x[i] - F.x[j], F.y[i] - F.y[j], F.z[i] - F.z[j]); if (d < b) b = d; } nn.push(b); } }
    if (f % 30 === 0) { for (const w of S.whales) { const pts = []; for (let k = 0; k <= 8; k++) pts.push(S.at(w, 0.4 - k * 0.11, 0)); for (let i = 0; i < S.NF; i++) { tot++; for (const p of pts) if (Math.hypot(F.x[i] - p[0], F.y[i] - p[1], F.z[i] - p[2]) < 1.5) { close++; break; } } } } }
  nn.sort((a, b) => a - b); const med = nn[nn.length >> 1];
  verdict('B7a', pol / np > 0.5, 'mean polarisation ' + (pol / np).toFixed(3)); verdict('B7b', med >= 0.2 && med <= 1.0, 'median nearest neighbour ' + med.toFixed(3) + ' m');
  verdict('B7c', close / tot < 0.01, (100 * close / tot).toFixed(3) + '% of fish-samples within 1.5 m of a whale centre line'); }
// B8
{ const S = M.makeSim({ auto: false, events: [{ event: 'bubblenet', at: 600 }] }); const F = S.fish; let inside = null, peaks = {}, done = null, lungeF = null;
  for (let f = 0; f < 600 + 120 * 60; f++) { S.step(); for (const e of S.ev) { if (e.type === 'net_lunge') { lungeF = e.frame; let n = 0; for (let i = 0; i < S.NF; i++) if (Math.hypot(F.x[i] - S.net.cx, F.z[i] - S.net.cz) < 10 && F.y[i] > -8) n++; inside = n / S.NF; } if (e.type === 'net_done') done = e.frame; }
    if (S.net && S.net.phase === 'lunge') for (const w of [S.mother, S.male]) peaks[w.role] = Math.max(peaks[w.role] || -99, S.snout(w)[1]);
    if (done && !S._bk && S.mother.state !== 'bn_sink' && S.male.state !== 'bn_sink') S._bk = f; }
  const N = S.net, rr = N ? N.ringR / Math.max(1, N.ringN) : 0;
  verdict('B8', N && rr >= 6 && rr <= 10 && inside >= 0.6 && peaks.mother > 0 && peaks.singer > 0 && S.eaten >= 5 && done && (done - 600) / 60 <= 90,
    'ring radius ' + rr.toFixed(2) + ' m from ' + (N ? N.ringN : 0) + ' bubbles; fish inside at lunge ' + (inside === null ? '-' : (100 * inside).toFixed(0) + '%') + '; snout peaks ' + JSON.stringify(peaks) + '; taken ' + S.eaten + '; lunge at frame ' + lungeF + '; over ' + (done ? ((done - 600) / 60).toFixed(1) : '-') + ' s after request');
  out('   log: ' + S.log.filter(e => e.type !== 'unit').map(e => e.frame + ':' + e.type + (e.whale ? '(' + e.whale[0] + ')' : '')).join(' ')); }
// B9a
{ const mk = () => { const S = M.makeSim({ events: [{ event: 'breach', at: 300 }] }); for (let f = 0; f < 1800; f++) S.step(); let h = 0; const add = v => { h = (h * 31 + Math.round(v * 1e6)) | 0; }; for (const w of S.whales) [w.x, w.y, w.z, w.yaw, w.pitch, w.roll, w.v, w.ph, w.C].forEach(add); for (let i = 0; i < S.NF; i++) { add(S.fish.x[i]); add(S.fish.y[i]); add(S.fish.z[i]); } for (let i = 0; i < 200; i++) add(S.spray.x[i]); return h; };
  const a = mk(), b = mk(); verdict('B9a', a === b, 'state hashes ' + a + ' and ' + b); }
// B12
{ const S = M.makeSim({ auto: false }); const F = S.fish; let ok = true, mx = 0, far = 0, flee = true, links = 0;
  for (let f = 0; f < 3000; f++) { S.step(); if (f % 300 === 299) for (let i = 0; i < S.NF; i++) { mx = Math.max(mx, F.nc[i]); for (let k = 0; k < F.nc[i]; k++) { const j = F.nb[i * 10 + k], d = Math.hypot(F.x[i] - F.x[j], F.y[i] - F.y[j], F.z[i] - F.z[j]); links++; far = Math.max(far, d); if (d > 1.9) ok = false; }
    if (F.fw[i] >= 0) { const o = F.fw[i] * 5, d = Math.hypot(F.x[i] - S.wp[o], F.y[i] - S.wp[o + 1], F.z[i] - S.wp[o + 2]); if (d > S.wp[o + 3] + 0.2) flee = false; } } }
  verdict('B12', ok && mx <= 10 && flee, links + ' recorded links sampled, longest ' + far.toFixed(2) + ' m, most per fish ' + mx + ', fleeing fish inside the whale radius: ' + flee); }
// B13 (model half): travel time along the level ray in the channel
{ const ch = M.channel(18, 35), r = M.rays(18, 35, ch.depth, [0], 150000, 60)[0]; verdict('B13a', r.time >= 99 && r.time <= 102, 'level ray from ' + ch.depth + ' m takes ' + r.time.toFixed(1) + ' s over 150 km'); }
// speed
{ const S = M.makeSim({}); const t0 = Date.now(); for (let f = 0; f < 3600; f++) S.step(); out('   3600 steps with 420 fish took ' + (Date.now() - t0) + ' ms in node'); }
out(JSON.stringify(res));
