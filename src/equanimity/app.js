/* Holding weight through time: the bench, the card, the controls and the film hook. The model is in model.js (window.EQModel). */
(function () {
'use strict';
const TEST = window.__TEST || null, M = window.EQModel;
if (TEST) document.body.classList.add('film');
const canvas = document.getElementById('bench'), $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v)), lerp = (a, b, t) => a + (b - a) * t, TAU = Math.PI * 2;
const sstep = (a, b, x) => { const t = clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const LOW = !!(TEST && TEST.low) || (!TEST && Math.min(window.innerWidth, window.innerHeight) < 700);

/* ───────────────────────── settings shared by the bench and the card ───────────────────────── */
const U = 0.3;                 /* bench units for one unit of place */
const TE = 48;                 /* frames between meetings (0.8 s) */
const LOOP = 480;              /* frames for one pass of the marble along the comb */
const SMIN = 0.3, SMAX = 6;    /* the narrowest bell the clay can show, and the widest */
const WB = 1.15, HB = 0.7, HS = 0.204, BASE = 0.1, RHO = 0.155;   /* bowl: half length, rim height, height scale, floor, marble radius */
const BS = 1.5;                /* bench height of a bell of height 1 */
const PINX = i => (i - 3.5) * 0.7, MX0 = -3.0, MX1 = 3.0, F0 = 30, F1 = 450;
const SEED_E = 232, SEED_F = 11;

/* ───────────────────────── what is true at a frame: numbers only ───────────────────────── */
function fnOf(spec, dflt) { return k => M.param(spec, k * TE, dflt); }
function resolve(frame, v) {
  v = v || {}; const sc = String(v.scene || 'A').toUpperCase().charAt(0), P = (s, d) => M.param(s, frame, d);
  const S = { scene: sc, frame, yaw: P(v.yaw, 0), pitch: P(v.pitch, 0), zoom: P(v.zoom, 1), card: clamp(P(v.card, 1), 0, 1), focus: v.focus };
  if (sc === 'A' || sc === 'D') {
    const s = clamp(P(v.width, sc === 'A' ? SMIN : 1), SMIN, SMAX); S.width = s; S.height = M.bell(0, s); S.point = sc === 'A' && s <= SMIN + 1e-9;
    let A = 0; const n = 2000, lim = 12 * s, dx = 2 * lim / n; for (let i = 0; i <= n; i++) A += M.bell(-lim + i * dx, s) * (i === 0 || i === n ? .5 : 1) * dx; S.area = A;
    S.bowlAt1 = M.bowl(1, s);
  } else if (sc === 'B' || sc === 'C') {
    const lf = ((frame % LOOP) + LOOP) % LOOP; S.lf = lf; S.mx = lerp(MX0, MX1, clamp((lf - F0) / (F1 - F0), 0, 1));
    S.tg = (S.mx - PINX(0)) / 0.7 + 1; S.passed = M.count(S.tg); S.count = S.passed;
    S.passFrame = i => F0 + (PINX(i) - MX0) / (MX1 - MX0) * (F1 - F0);
  } else if (sc === 'E' || sc === 'G') {
    const pre = v.preset && M.PRESETS.E[v.preset], t = frame / TE, k = Math.floor(t), tau = t - k;
    let Ks = v.K, rs = v.r; if (pre && Ks === undefined) Ks = pre.K === null ? undefined : pre.K; if (sc === 'G') Ks = undefined;
    const Kf = (Ks === undefined || Ks === null) ? null : fnOf(Ks, NaN), rf = fnOf(rs, 1);
    const run = M.runE({ seed: v.seed === undefined ? SEED_E : v.seed, n: k + 1, K: Kf, r: rf });
    S.run = run; S.k = k; S.tau = tau; S.t = t; S.auto = !Kf || !(Kf(k) === Kf(k)); S.r = rf(k);
    const e = eAt(run, t); S.m = e.m; S.P = e.P; S.world = e.x; S.centre = e.c; S.flash = e.flash;
    S.K = run.K[k]; S.kept = 1 - S.K; S.before = k ? run.Pm[k] : run.Pm[1]; S.after = run.P[k]; S.mse = k ? run.mse[k] : NaN; S.surprise = k ? run.ahead[k] : NaN; S.pathRatio = k && run.pathW[k] > 0 ? run.pathM[k] / run.pathW[k] : NaN; S.pathM = run.pathM[k]; S.pathW = run.pathW[k]; S.caught = k ? run.caught[k] : NaN; S.meetings = k;
    S.steadyK = M.steadyK(S.r);
  } else {
    const pre = (v.preset && M.PRESETS.F[v.preset]) || M.PRESETS.F.balance, t = frame / TE, k = Math.floor(t), tau = t - k;
    const Ktf = fnOf(v.Kt, pre.Kt), Klf = fnOf(v.Kl, pre.Kl);
    const run = M.runF({ seed: v.seed === undefined ? SEED_F : v.seed, n: k + 1, Kt: Ktf, Kl: Klf });
    S.scene = 'F'; S.run = run; S.k = k; S.tau = tau; S.t = t; const f = fAt(run, t); S.a = f.a; S.b = f.b; S.Pa = f.Pa; S.Pb = f.Pb; S.gap = f.b - f.a;
    S.Kt = Ktf(k); S.Kl = Klf(k); S.factor = 1 - S.Kt - S.Kl; S.swaps = run.swapsAt[k]; S.meanAbs = k ? run.meanAbs[k] : NaN; S.firstShare = k ? run.firstShare : NaN; S.rmsPre = k ? run.rmsPre[k] : NaN; S.meetings = k;
  }
  return S;
}
/* the run of scene E read at a continuous time t (in meetings): the marble slides and the bowl narrows early in each interval, then the bowl widens */
function eAt(run, t) {
  t = Math.max(0, t); const k = Math.min(Math.floor(t), run.n), tau = t - k, ez = sstep(.08, .42, tau);
  const m = k ? lerp(run.m[k - 1], run.m[k], ez) : run.m[0];
  const P = k ? (tau < .42 ? lerp(run.Pm[k], run.P[k], ez) : lerp(run.P[k], run.Pm[k + 1], (tau - .42) / .58)) : lerp(run.P[0], run.Pm[1], tau);
  const x = lerp(run.x[k], run.x[k + 1], tau);
  return { k, tau, m, P, x, pw: run.pathW[k] + Math.abs(x - run.x[k]), pm: k ? run.pathM[k - 1] + Math.abs(m - run.m[k - 1]) : 0, c: k ? lerp(run.centre[k - 1], run.centre[k], sstep(0, .5, tau)) : 0, flash: k ? Math.exp(-tau * 7) : 0 };
}
function fAt(run, t) {
  t = Math.max(0, t); const k = Math.min(Math.floor(t), run.n), tau = t - k, ez = sstep(0, .35, tau), d = clamp((tau - .35) / .65, 0, 1);
  const pos = (post, pre) => k ? (tau < .35 ? lerp(pre[k], post[k], ez) : lerp(post[k], pre[k + 1], d)) : lerp(post[0], pre[1], tau);
  const bel = K => { const b = M.believed(clamp(K, .001, .999), 1); return k ? (tau < .35 ? lerp(b.Pm, b.P, ez) : lerp(b.P, b.Pm, d)) : lerp(b.P, b.Pm, tau); };
  return { a: pos(run.a, run.aPre), b: pos(run.b, run.bPre), Pa: bel(run.kt[Math.max(1, k)]), Pb: bel(run.kl[Math.max(1, k)]) };
}

/* ───────────────────────── the card: real graph paper, drawn every frame from the same numbers ───────────────────────── */
const CW = 1280, CH = 800, INK = '#14202a', GOLD = '#a8780f', VIR = '#0d7a70', SOFT = '#4d5a61', SERIF = 'Georgia, "Times New Roman", serif';
const gc = document.createElement('canvas'); gc.width = CW; gc.height = CH; const g = gc.getContext('2d');
gc.setAttribute('role', 'img');
function paper(title, note) {
  g.setTransform(1, 0, 0, 1, 0, 0); g.fillStyle = '#fbf6e6'; g.fillRect(0, 0, CW, CH); g.lineWidth = 1;
  for (let x = 0; x <= CW; x += 20) { g.strokeStyle = x % 100 ? 'rgba(13,122,112,.13)' : 'rgba(13,122,112,.3)'; g.beginPath(); g.moveTo(x + .5, 0); g.lineTo(x + .5, CH); g.stroke(); }
  for (let y = 0; y <= CH; y += 20) { g.strokeStyle = y % 100 ? 'rgba(13,122,112,.13)' : 'rgba(13,122,112,.3)'; g.beginPath(); g.moveTo(0, y + .5); g.lineTo(CW, y + .5); g.stroke(); }
  g.fillStyle = INK; g.textBaseline = 'alphabetic'; g.textAlign = 'left'; g.font = 'italic 62px ' + SERIF; g.fillText(title, 52, 80);
  if (note) { g.textAlign = 'right'; g.font = '37px ' + SERIF; g.fillStyle = INK; g.fillText(note, CW - 46, 78); }
}
function panel(x0, y0, w, h, xa, xb, ya, yb) {
  const X = v => x0 + (v - xa) / (xb - xa) * w, Y = v => y0 + h - (v - ya) / (yb - ya) * h;
  return { X, Y, x0, y0, w, h, xa, xb, ya, yb,
    clip(fn) { g.save(); g.beginPath(); g.rect(x0, y0 - 2, w, h + 4); g.clip(); fn(); g.restore(); },
    axes(xl, yl, xt, yt, o) { o = o || {}; g.strokeStyle = INK; g.fillStyle = INK; g.lineWidth = 3; const yb0 = o.zero !== undefined ? Y(o.zero) : y0 + h;
      g.beginPath(); g.moveTo(x0, y0); g.lineTo(x0, y0 + h); g.moveTo(x0, yb0); g.lineTo(x0 + w, yb0); g.stroke();
      g.font = '32px ' + SERIF; g.textAlign = 'center'; g.textBaseline = 'top';
      (xt || []).forEach(t => { const lab = Array.isArray(t) ? t[1] : String(t), v = Array.isArray(t) ? t[0] : t; if (v < xa - 1e-9 || v > xb + 1e-9) return; g.beginPath(); g.moveTo(X(v), yb0); g.lineTo(X(v), yb0 + 10); g.stroke(); if (!o.noXLab) g.fillText(lab, X(v), y0 + h + 13); });
      g.textAlign = 'right'; g.textBaseline = 'middle';
      (yt || []).forEach(t => { const lab = Array.isArray(t) ? t[1] : String(t), v = Array.isArray(t) ? t[0] : t; g.beginPath(); g.moveTo(x0 - 10, Y(v)); g.lineTo(x0, Y(v)); g.stroke(); g.fillText(lab, x0 - 15, Y(v)); });
      g.font = 'italic 35px ' + SERIF; g.textBaseline = 'alphabetic';
      if (xl) { g.textAlign = 'right'; g.fillText(xl, x0 + w, y0 + h + (o.noXLab ? 38 : 80)); }
      if (yl) { g.textAlign = 'left'; g.fillStyle = o.yc || INK; g.fillText(yl, x0 + 14, y0 + 32 + (o.ylDy || 0)); } },
    line(pts, col, lw, dash) { if (pts.length < 2) return; g.save(); g.beginPath(); g.rect(x0 - 2, y0 - 2, w + 4, h + 4); g.clip(); g.strokeStyle = col; g.lineWidth = lw; g.lineJoin = 'round'; g.lineCap = 'round'; g.setLineDash(dash || []); g.beginPath(); pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke(); g.restore(); },
    area(top, bot, fill) { g.save(); g.beginPath(); g.rect(x0, y0, w, h); g.clip(); g.fillStyle = fill; g.beginPath(); top.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); for (let i = bot.length - 1; i >= 0; i--) g.lineTo(X(bot[i][0]), Y(bot[i][1])); g.closePath(); g.fill(); g.restore(); },
    arrow(x, y1, col, lw) { g.strokeStyle = col; g.fillStyle = col; g.lineWidth = lw; g.lineCap = 'butt'; g.beginPath(); g.moveTo(X(x), Y(0)); g.lineTo(X(x), Y(y1) + 14); g.stroke(); g.beginPath(); g.moveTo(X(x), Y(y1)); g.lineTo(X(x) - 11, Y(y1) + 22); g.lineTo(X(x) + 11, Y(y1) + 22); g.closePath(); g.fill(); },
    dot(x, y, r, col, open) { g.beginPath(); g.arc(X(x), Y(y), r, 0, TAU); if (open) { g.fillStyle = '#fbf6e6'; g.fill(); g.strokeStyle = col; g.lineWidth = 3; g.stroke(); } else { g.fillStyle = col; g.fill(); } } };
}
const label = (t, x, y, col, o) => { o = o || {}; g.fillStyle = col || INK; g.textAlign = o.align || 'left'; g.textBaseline = 'alphabetic'; g.font = (o.it === false ? '' : 'italic ') + (o.size || 35) + 'px ' + SERIF; g.fillText(t, x, y); };
const e2 = a => a[a.length - 1][1];
function bellPts(s, xa, xb) { const p = []; for (let i = 0; i <= 480; i++) { const x = xa + (xb - xa) * i / 480; p.push([x, M.bell(x, s)]); } return p; }
function drawBell(pn, s, point) {
  const p = bellPts(s, pn.xa, pn.xb); pn.area(p, [[pn.xa, 0], [pn.xb, 0]], 'rgba(224,174,53,.34)'); pn.line(p, INK, 5);
  if (point) pn.arrow(0, pn.yb * .97, INK, 5);
}
function drawCard(S) {
  const sc = S.scene;
  if (sc === 'A') { paper('A point', 'width ' + S.width.toFixed(2) + '   height ' + S.height.toFixed(2) + '   weight ' + S.area.toFixed(2));
    const pn = panel(120, 130, 1100, 560, -4, 4, 0, 1.5); drawBell(pn, S.width, S.point); pn.axes('place', 'how much is there', [-4, -3, -2, -1, 0, 1, 2, 3, 4], [0, .5, 1, 1.5]);
    if (S.point) { label('narrower still, and it is a point:', pn.X(0) + 150, pn.Y(1.2), INK); label('all the weight in one place', pn.X(0) + 150, pn.Y(1.2) + 44, INK); }
  } else if (sc === 'B' || sc === 'C') {
    const two = sc === 'C'; paper(two ? 'The count' : 'A comb', two ? 'the comb, added up as you pass:  ' + S.count : 'teeth passed:  ' + S.passed);
    const cb = two ? panel(120, 560, 1100, 150, 0, 9, 0, 1.25) : panel(120, 180, 1100, 500, 0, 9, 0, 1.3), tg = clamp(S.tg, 0, 9);
    if (two) { const st = panel(120, 130, 1100, 370, 0, 9, 0, 8.6); const full = [[0, 0]]; for (let i = 1; i <= 8; i++) { full.push([i, i - 1], [i, i]); } full.push([9, 8]);
      st.line(full, 'rgba(13,122,112,.3)', 3, [4, 9]);
      const done = [[0, 0]]; for (let i = 1; i <= 8 && i <= tg; i++) done.push([i, i - 1], [i, i]); done.push([tg, M.count(tg)]);
      st.area(done, [[0, 0], [tg, 0]], 'rgba(13,122,112,.12)'); st.line(done, VIR, 6);
      g.save(); g.setLineDash([3, 8]); g.strokeStyle = SOFT; g.lineWidth = 2; for (let i = 1; i <= 8 && i <= tg; i++) { g.beginPath(); g.moveTo(st.X(i), st.Y(i - 1)); g.lineTo(cb.X(i), cb.Y(1)); g.stroke(); } g.restore();
      st.axes('', 'the count', [1, 2, 3, 4, 5, 6, 7, 8], [0, 2, 4, 6, 8], { noXLab: true, yc: VIR }); st.dot(tg, M.count(tg), 9, VIR);
      cb.axes('time', 'the comb', [1, 2, 3, 4, 5, 6, 7, 8], [], { ylDy: -34 });
    } else cb.axes('time', 'one tooth at each tick', [1, 2, 3, 4, 5, 6, 7, 8], [0, 1]);
    for (let i = 1; i <= 8; i++) cb.arrow(i, 1, i <= tg ? INK : 'rgba(20,32,42,.28)', 5);
    cb.dot(tg, 0, 10, VIR);
  } else if (sc === 'D') { paper('Bell and bowl', 'width ' + S.width.toFixed(2));
    const top = panel(120, 120, 1100, 250, -4, 4, 0, 1.5), bot = panel(120, 440, 1100, 270, -4, 4, 0, 4.5);
    drawBell(top, S.width, false); top.axes('', 'how much is there', [-4, -3, -2, -1, 0, 1, 2, 3, 4], [0, .5, 1, 1.5], { noXLab: true, yc: GOLD });
    const p = []; for (let i = 0; i <= 480; i++) { const x = -4 + 8 * i / 480; p.push([x, Math.min(4.5, M.bowl(x, S.width))]); }
    bot.area([[-4, 4.5], [4, 4.5]], p, 'rgba(13,122,112,.15)'); bot.line(p, VIR, 5);
    bot.axes('place', '', [-4, -3, -2, -1, 0, 1, 2, 3, 4], [0, 2, 4]);
    label('how hard it is to be elsewhere', bot.x0 + 14, bot.y0 + bot.h - 14, VIR); label('bowl = minus log of bell', bot.x0 + bot.w - 8, bot.y0 + bot.h - 14, SOFT, { align: 'right' });
  } else if (sc === 'E') { const run = S.run, t = S.t, win = 20, t0 = Math.max(0, t - win + 3), cy = Math.round(S.centre);
    paper('Carried and met', 'kept ' + S.kept.toFixed(3) + '   given ' + S.K.toFixed(3));
    const pn = panel(120, 165, 1100, 340, t0, t0 + win, cy - 8.5, cy + 8.5), k1 = Math.max(1, Math.floor(t0)), kn = Math.floor(t);
    /* the cone: two standard deviations either side of where the marble sits, by its own reckoning, drawn in the same time as the bench */
    const up = [], dn = [], est = [], wl = [], pw = [], pe = []; for (let i = 0; i <= 600; i++) { const tt = t0 + (t - t0) * i / 600, e = eAt(run, tt), sd = 2 * Math.sqrt(e.P); up.push([tt, e.m + sd]); dn.push([tt, e.m - sd]); est.push([tt, e.m]); wl.push([tt, e.x]); pw.push([tt, e.pw]); pe.push([tt, e.pm]); }
    pn.area(up, dn, 'rgba(13,122,112,.17)');
    const gl = []; for (let j = k1; j <= kn; j++) gl.push([j, run.y[j]]);
    pn.line(gl, 'rgba(20,32,42,.45)', 2, [3, 7]); pn.line(wl, GOLD, 6); pn.line(est, VIR, 6);
    pn.clip(() => gl.forEach(p => pn.dot(p[0], p[1], 7, INK, true)));
    const tk = []; for (let j = Math.ceil(t0 / 5) * 5; j <= t0 + win; j += 5) tk.push(j); pn.axes('', 'place', tk, [cy - 8, cy, cy + 8], { noXLab: true });
    let lx = 128; const key = (txt, col, kind) => { g.lineWidth = 6; g.strokeStyle = col; if (kind === 'dot') { g.beginPath(); g.arc(lx + 20, 128, 7, 0, TAU); g.fillStyle = '#fbf6e6'; g.fill(); g.lineWidth = 3; g.strokeStyle = INK; g.stroke(); } else if (kind === 'band') { g.fillStyle = 'rgba(13,122,112,.24)'; g.fillRect(lx, 112, 40, 30); } else { g.beginPath(); g.moveTo(lx, 128); g.lineTo(lx + 40, 128); g.stroke(); }
      label(txt, lx + 50, 140, INK, { size: 36 }); lx += 50 + g.measureText(txt).width + 34; };
    key('the world', GOLD); key('glimpses', INK, 'dot'); key('what is carried', VIR); key('the cone it expects', VIR, 'band');
    /* how far each has walked, added up */
    const top = Math.max(12, e2(pw), e2(pe)) * 1.12, pp = panel(120, 560, 1100, 150, t0, t0 + win, 0, top), st = Math.pow(10, Math.floor(Math.log10(top / 2))), stp = top / st > 10 ? st * 5 : top / st > 4 ? st * 2 : st, yt = []; for (let v = 0; v <= top; v += stp) yt.push(v);
    pp.line(pw, GOLD, 5); pp.line(pe, VIR, 5); pp.axes('meetings', 'path walked so far', tk, yt, { ylDy: -6 });
    label('the world', pp.X(t) + 12, pp.Y(e2(pw)) + (e2(pw) >= e2(pe) ? -4 : 30), GOLD, { size: 31 }); label('what is carried', pp.X(t) + 12, pp.Y(e2(pe)) + (e2(pw) >= e2(pe) ? 30 : -4), VIR, { size: 31 });
  } else if (sc === 'F') { const run = S.run, t = S.t, win = 20, t0 = Math.max(0, t - win + 3);
    paper('Two who meet', 'gains ' + S.Kt.toFixed(3) + ' and ' + S.Kl.toFixed(3));
    const top = panel(120, 130, 1100, 330, t0, t0 + win, -8.5, 8.5), bot = panel(120, 520, 1100, 190, t0, t0 + win, -13, 13);
    const pa = [], pb = [], pg = []; for (let i = 0; i <= 700; i++) { const tt = t0 + (t - t0) * i / 700, f = fAt(run, tt); pa.push([tt, f.a]); pb.push([tt, f.b]); pg.push([tt, f.b - f.a]); }
    const tk = []; for (let j = Math.ceil(t0 / 5) * 5; j <= t0 + win; j += 5) tk.push(j);
    top.line([[t0, run.a[0]], [t0 + win, run.a[0]]], GOLD, 2, [10, 10]); top.line([[t0, run.b[0]], [t0 + win, run.b[0]]], VIR, 2, [10, 10]);
    label('where gold began', top.x0 + top.w - 8, top.Y(run.a[0]) - 10, GOLD, { align: 'right', size: 31 }); label('where green began', top.x0 + top.w - 8, top.Y(run.b[0]) + 36, VIR, { align: 'right', size: 31 });
    top.line(pa, GOLD, 6); top.line(pb, VIR, 6); top.axes('', 'where each one is', tk, [-8, -4, 0, 4, 8], { noXLab: true });
    bot.area(pg, [[t0, 0], [t, 0]], 'rgba(20,32,42,.12)'); bot.line(pg, INK, 5); bot.axes('exchanges', 'the gap between them', tk, [-12, 0, 12], { zero: 0, ylDy: -34 });
  } else { const run = S.run, t = S.t, win = 10, t0 = Math.max(0, t - win + 1.5);
    paper('The vessel breathes', 'uncertainty now ' + S.P.toFixed(3));
    const pn = panel(120, 150, 1100, 540, t0, t0 + win, 0, 2.4), p = []; for (let i = 0; i <= 800; i++) { const tt = t0 + (t - t0) * i / 800; p.push([tt, eAt(run, tt).P]); }
    const lo = S.after, hi = S.before; g.save(); g.setLineDash([10, 10]); g.strokeStyle = SOFT; g.lineWidth = 2; [lo, hi].forEach(v => { g.beginPath(); g.moveTo(pn.x0, pn.Y(v)); g.lineTo(pn.x0 + pn.w, pn.Y(v)); g.stroke(); }); g.restore();
    pn.area(p, [[t0, 0], [t, 0]], 'rgba(13,122,112,.13)'); pn.line(p, VIR, 6); pn.dot(t, S.P, 10, VIR);
    const tk = []; for (let j = Math.ceil(t0); j <= t0 + win; j += (win > 12 ? 5 : 1)) tk.push(j); pn.axes('meetings', 'how wide the bowl is (variance)', tk, [0, .5, 1, 1.5, 2], { ylDy: -6 });
    label('before a meeting  ' + hi.toFixed(3), pn.x0 + pn.w - 10, pn.Y(hi) - 12, SOFT, { align: 'right' }); label('after a meeting  ' + lo.toFixed(3), pn.x0 + pn.w - 10, pn.Y(lo) + 36, SOFT, { align: 'right' });
  }
}

/* ───────────────────────── state of the page (also used, fixed, by the film hook) ───────────────────────── */
const SCENES = [['A', 'A point'], ['B', 'A comb'], ['C', 'The count'], ['D', 'Bell and bowl'], ['E', 'Carried and met'], ['F', 'Two who meet'], ['G', 'The vessel breathes']];
const SAY = {
  A: 'A bell that keeps its weight as it narrows. At the end it is a pin: all the weight in one place.',
  B: 'Points at even spacing make a comb. The marble ticks as it passes each one.',
  C: 'Add the teeth up as the marble passes and the blocks rise, one step for each tooth. The staircase is the comb added up.',
  D: 'The same curve, turned over. A narrow bell is a deep well that holds the marble hard. A wide bell is nearly level ground.',
  E: 'The lantern is the world; it drifts. Each bead of light is a noisy glimpse of it. The marble keeps a share of where it was and gives the rest to the glimpse.',
  F: 'Each takes the other as its glimpse. Gains that add to one meet at once; each then drifts, so a small difference stays alive.',
  G: 'At balance the bowl narrows at each meeting and widens again before the next. It never closes to a point and never lies flat.' };
const SAYF = { hold: 'Both hold hard. Each gives 8%, so only 16% of the gap closes at each exchange. They draw together slowly and stay further apart than at balance.',
  letgo: 'Both let go. Each gives 90%, so they overshoot, swap sides and thrash, closing a fifth of the gap each time.',
  mirror: 'One gives 2% and the other 98%. They do meet, at once, but almost exactly where the first one began. The one who does not move decides where the meeting is.',
  balance: 'Gains of 0.382 and 0.618 add to one, so they meet in one exchange, part of the way along. Then each drifts, and a small difference stays alive.' };
let lastS = null;
window.EQ = { model: M, resolve, scenes: SCENES.map(s => s[0]),
  state() { const S = lastS; if (!S) return null; const o = { scene: S.scene, frame: S.frame };
    if (S.scene === 'A' || S.scene === 'D') Object.assign(o, { width: S.width, height: S.height, area: S.area, bowlAtOne: S.bowlAt1 });
    else if (S.scene === 'B' || S.scene === 'C') Object.assign(o, { teethPassed: S.passed, count: S.count });
    else if (S.scene === 'F') Object.assign(o, { Kt: S.Kt, Kl: S.Kl, factor: S.factor, gap: S.gap, a: S.a, b: S.b, exchanges: S.meetings, swaps: S.swaps, meanGapAfter: S.meanAbs, rmsGapBefore: S.rmsPre, firstMoveShareOfFirst: S.firstShare, start: [-6, 6] });
    else Object.assign(o, { weightKept: S.kept, weightGiven: S.K, uncertaintyBefore: S.before, uncertaintyAfter: S.after, uncertaintyNow: S.P, cone: S.before, surprise: S.surprise, pathWalked: S.pathM, worldPath: S.pathW, pathRatio: S.pathRatio, mse: S.mse, fanCaught: S.caught, meetings: S.meetings, r: S.r, auto: S.auto, estimate: S.m, world: S.world });
    return o; } };

/* ───────────────────────── the bench (WebGL) ───────────────────────── */
let GL = null;
try { GL = (window.THREE && canvas.getContext('webgl2')) ? makeBench() : null; } catch (err) { GL = null; console.error(err); }
if (!GL) { const n = $('nogl'); if (n) n.style.display = 'grid'; }

function makeBench() {
const THREE = window.THREE;
function h3(x, y, z) { let n = Math.imul(x | 0, 374761393) ^ Math.imul(y | 0, 668265263) ^ Math.imul(z | 0, 1440662683); n = Math.imul(n ^ (n >>> 13), 1274126177); return ((n ^ (n >>> 16)) >>> 0) / 4294967296; }
function vn3(x, y, z) { const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z); let fx = x - xi, fy = y - yi, fz = z - zi; fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
  const a = h3(xi, yi, zi), b = h3(xi + 1, yi, zi), c = h3(xi, yi + 1, zi), d = h3(xi + 1, yi + 1, zi), e = h3(xi, yi, zi + 1), f = h3(xi + 1, yi, zi + 1), gg = h3(xi, yi + 1, zi + 1), h = h3(xi + 1, yi + 1, zi + 1);
  const ab = a + (b - a) * fx, cd = c + (d - c) * fx, ef = e + (f - e) * fx, gh = gg + (h - gg) * fx, p = ab + (cd - ab) * fy, q = ef + (gh - ef) * fy; return p + (q - p) * fz; }
function fbm3(x, y, z) { return vn3(x, y, z) * .55 + vn3(x * 2.1 + 5, y * 2.1 + 3, z * 2.1 + 1) * .3 + vn3(x * 4.3 + 9, y * 4.3 + 7, z * 4.3 + 2) * .15; }
const rnd = (i, j) => h3(i, j, 77);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: true });
renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap; renderer.toneMapping = THREE.NoToneMapping;
const HDR = renderer.extensions.has('EXT_color_buffer_float') || renderer.extensions.has('EXT_color_buffer_half_float');
/* contact-hardening soft shadows, patched into three's shadow lookup (as on the potter's wheel) */
(function () {
  const pcss = `
  float ignoise(vec2 p){ return fract(52.9829189*fract(dot(p, vec2(0.06711056,0.00583715)))); }
  float pcssShadow(sampler2D sm, vec3 c, float K){
    float rot = ignoise(gl_FragCoord.xy)*6.2831853;
    float search = K*0.05, bsum=0., bn=0.;
    for(int i=0;i<${LOW ? 8 : 14};i++){ float fi=float(i)+.5; float r=sqrt(fi/${LOW ? 8 : 14}.); float a=fi*2.39996323+rot;
      float d = unpackRGBAToDepth(texture2D(sm, c.xy+vec2(cos(a),sin(a))*r*search));
      if(d < c.z-0.0006){ bsum+=d; bn+=1.; } }
    if(bn<.5) return 1.;
    float pen = clamp((c.z-bsum/bn)*K, 0.0006, 0.02);
    float s=0.;
    for(int i=0;i<${LOW ? 12 : 28};i++){ float fi=float(i)+.5; float r=sqrt(fi/${LOW ? 12 : 28}.); float a=fi*2.39996323+rot*1.7;
      s += step(c.z, unpackRGBAToDepth(texture2D(sm, c.xy+vec2(cos(a),sin(a))*r*pen))); }
    return s/${LOW ? 12 : 28}.;
  }`;
  let ch = THREE.ShaderChunk.shadowmap_pars_fragment;
  ch = ch.replace('#ifdef USE_SHADOWMAP', '#ifdef USE_SHADOWMAP\n' + pcss);
  ch = ch.replace('#if defined( SHADOWMAP_TYPE_PCF )', 'return pcssShadow(shadowMap, shadowCoord.xyz, shadowRadius);\n#if defined( SHADOWMAP_TYPE_PCF )');
  THREE.ShaderChunk.shadowmap_pars_fragment = ch;
})();

/* materials: clay and wood are worked out in the shader from the place on the object, so nothing is a picture stretched over a shape */
const GLSL_NOISE = `
float hash13(vec3 p){ p=fract(p*.1031); p+=dot(p,p.zyx+31.32); return fract((p.x+p.y)*p.z); }
float vn(vec3 x){ vec3 i=floor(x), f=fract(x); f=f*f*(3.-2.*f);
  return mix(mix(mix(hash13(i),hash13(i+vec3(1,0,0)),f.x),mix(hash13(i+vec3(0,1,0)),hash13(i+vec3(1,1,0)),f.x),f.y),
             mix(mix(hash13(i+vec3(0,0,1)),hash13(i+vec3(1,0,1)),f.x),mix(hash13(i+vec3(0,1,1)),hash13(i+vec3(1,1,1)),f.x),f.y),f.z); }`;
const CLAY_PARS = GLSL_NOISE + `
uniform mat3 normalMatrix;
uniform float uBump, uSmear, uSpeck, uPlank, uGrain; uniform vec3 uCol2, uOff;
varying vec3 vObjPos; varying vec3 vObjN;
float clayH(vec3 q){
  float h = (vn(q*8.)-.5) + (vn(q*21.+3.)-.5)*.5 + (vn(q*57.+7.)-.5)*.22;
  h += smoothstep(.8,.95, vn(vec3(q.x*3., q.y*40., q.z*3.)+2.))*-.07;
  return h;
}
float woodH(vec3 p){ vec2 P=p.xz*uGrain; return vn(vec3(P.x*6., P.y*120., p.y*90.))*.5 + vn(vec3(P.x*1.5,P.y*40.,p.y*30.))*.5; }
vec3 clayNormal(vec3 nView){
  vec3 p = vObjPos + uOff; float e = .0035; vec3 nO = normalize(vObjN);
  #ifdef MODE_CLAY
  float h0 = clayH(p);
  vec3 g = vec3(clayH(p+vec3(e,0,0))-h0, clayH(p+vec3(0,e,0))-h0, clayH(p+vec3(0,0,e))-h0)/e;
  float k = .011;
  #else
  float h0 = woodH(p);
  vec3 g = vec3(woodH(p+vec3(e,0,0))-h0, woodH(p+vec3(0,e,0))-h0, woodH(p+vec3(0,0,e))-h0)/e;
  float k = .006;
  #endif
  g -= dot(g,nO)*nO;
  vec3 nP = normalize(nO - g*uBump*k);
  vec3 nV = normalize(normalMatrix*nP);
  return dot(nV,nView) < 0. ? -nV : nV;
}
float clayRough(float r){
  #ifdef MODE_CLAY
  float m = vn((vObjPos+uOff)*6.+40.);
  return clamp(r + (m-.5)*.3 - smoothstep(.7,.9,vn((vObjPos+uOff)*3.+9.))*.14, .12, 1.);
  #else
  return clamp(r + (woodH(vObjPos+uOff)-.5)*.25, .2, 1.);
  #endif
}
vec3 clayAlbedo(vec3 base){
  vec3 p = vObjPos + uOff; vec3 c = base;
  #ifdef MODE_CLAY
    float m = vn(p*5.)*.6 + vn(p*17.+4.)*.4;
    c *= .86 + .28*m;
    float s = smoothstep(.6,.8, vn(vec3(p.x*3.,p.y*8.,p.z*3.)+31.));
    c = mix(c, uCol2, s*uSmear);
    float sp = smoothstep(.9,.95, vn(p*150.+17.))*step(.75, vn(p*9.+70.));
    c = mix(c, vec3(.82,.78,.7), sp*.35*uSpeck);
  #endif
  #ifdef MODE_WOOD
    vec2 P = p.xz*uGrain; float pid = floor(P.y/uPlank+.5); float py = fract(P.y/uPlank+.5);
    float gw = vn(vec3(P.x*.5+pid*7., P.y*5., pid + p.y*2.));
    float rings = sin(P.y*46. + p.y*31. + gw*9. + vn(vec3(P.x*.25,P.y*1.5,pid*3.))*14.);
    c = mix(vec3(.25,.14,.07), vec3(.47,.29,.15), .5+.5*rings*.7);
    c *= .8 + .35*vn(vec3(P.x*8., P.y*160., pid + p.y*50.));
    c *= mix(.45,1., smoothstep(0.,.02,py)*smoothstep(1.,.98,py));
    c *= base;
  #endif
  return c;
}`;
let offC = 1;
function mat(o) {
  o = o || {}; const mode = o.mode || 'CLAY';
  const m = new THREE.MeshPhysicalMaterial({ color: o.color !== undefined ? o.color : 0xffffff, roughness: o.rough !== undefined ? o.rough : .56, metalness: 0,
    sheen: mode === 'CLAY' ? .45 : 0, sheenRoughness: .5, sheenColor: new THREE.Color(o.color !== undefined ? o.color : 0xffffff).lerp(new THREE.Color(0xffffff), .5),
    specularIntensity: o.spec !== undefined ? o.spec : .75, envMapIntensity: o.env !== undefined ? o.env : 1, side: o.side || THREE.FrontSide });
  const oc = offC++, u = { uBump: { value: o.bump !== undefined ? o.bump : 1 }, uSmear: { value: o.smear || 0 }, uSpeck: { value: o.speck !== undefined ? o.speck : 1 }, uCol2: { value: new THREE.Color(o.color2 !== undefined ? o.color2 : 0xffffff) },
    uPlank: { value: o.plank || 1.15 }, uGrain: { value: o.grain || 1 }, uOff: { value: o.off === false ? new THREE.Vector3() : new THREE.Vector3(h3(oc, 1, 5) * 40, h3(oc, 2, 5) * 40, h3(oc, 3, 5) * 40) } };
  m.defines = Object.assign({}, m.defines, { ['MODE_' + mode]: '' }); m.userData.u = u;
  m.onBeforeCompile = sh => { Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader.replace('#include <common>', '#include <common>\nvarying vec3 vObjPos; varying vec3 vObjN;').replace('#include <begin_vertex>', '#include <begin_vertex>\nvObjPos=position; vObjN=normal;');
    sh.fragmentShader = sh.fragmentShader.replace('#include <common>', '#include <common>\n' + CLAY_PARS)
      .replace('#include <color_fragment>', '#include <color_fragment>\ndiffuseColor.rgb = clayAlbedo(diffuseColor.rgb);')
      .replace('#include <roughnessmap_fragment>', '#include <roughnessmap_fragment>\nroughnessFactor = clayRough(roughnessFactor);')
      .replace('#include <normal_fragment_maps>', '#include <normal_fragment_maps>\nnormal = clayNormal(normal);'); };
  m.customProgramCacheKey = () => 'eq|' + mode;
  return m;
}
const wood = (tint, o) => mat(Object.assign({ mode: 'WOOD', color: tint === undefined ? 0xffffff : tint, rough: .55, spec: .5, plank: 60, grain: 1.6 }, o || {}));
const GLZ = (c, r) => { const m = mat({ color: c, rough: r || .2, spec: 1, bump: .22, speck: .3, side: THREE.DoubleSide }); m.clearcoat = 1; m.clearcoatRoughness = .1; return m; };
const brass = (r) => new THREE.MeshPhysicalMaterial({ color: 0xd2a650, roughness: r || .28, metalness: 1, envMapIntensity: 1.25 });
const steel = new THREE.MeshPhysicalMaterial({ color: 0xb4b9bf, roughness: .22, metalness: 1, envMapIntensity: 1.3 });

function weld(geo) {
  const p = geo.attributes.position, n = new Float32Array(p.count * 3), map = new Map(), key = new Array(p.count);
  for (let i = 0; i < p.count; i++) { key[i] = Math.round(p.getX(i) * 2e4) + '_' + Math.round(p.getY(i) * 2e4) + '_' + Math.round(p.getZ(i) * 2e4); if (!map.has(key[i])) map.set(key[i], [0, 0, 0]); }
  const idx = geo.index, tri = idx ? idx.count / 3 : p.count / 3, a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let t = 0; t < tri; t++) { const i0 = idx ? idx.getX(t * 3) : t * 3, i1 = idx ? idx.getX(t * 3 + 1) : t * 3 + 1, i2 = idx ? idx.getX(t * 3 + 2) : t * 3 + 2;
    a.fromBufferAttribute(p, i0); b.fromBufferAttribute(p, i1); c.fromBufferAttribute(p, i2); b.sub(a); c.sub(a); b.cross(c);
    for (const i of [i0, i1, i2]) { const m = map.get(key[i]); m[0] += b.x; m[1] += b.y; m[2] += b.z; } }
  for (let i = 0; i < p.count; i++) { const m = map.get(key[i]); const l = Math.hypot(m[0], m[1], m[2]) || 1; n[i * 3] = m[0] / l; n[i * 3 + 1] = m[1] / l; n[i * 3 + 2] = m[2] / l; }
  geo.setAttribute('normal', new THREE.BufferAttribute(n, 3)); return geo;
}
let seedC = 1;
function clayify(geo, amp, freq) { const sd = (seedC++) * 13.7; weld(geo); const p = geo.attributes.position, n = geo.attributes.normal;
  for (let i = 0; i < p.count; i++) { const x = p.getX(i), y = p.getY(i), z = p.getZ(i), d = (fbm3(x * freq + sd, y * freq + sd * .7, z * freq + sd * .3) - .5) * 2 * amp; p.setXYZ(i, x + n.getX(i) * d, y + n.getY(i) * d, z + n.getZ(i) * d); }
  weld(geo); geo.computeBoundingSphere(); return geo; }
function rbox(w, h, d, r, amp, freq, s) { s = s || 10; const gg = new THREE.BoxGeometry(w, h, d, s, s, s), p = gg.attributes.position, hx = w / 2 - r, hy = h / 2 - r, hz = d / 2 - r, v = new THREE.Vector3(), c = new THREE.Vector3();
  for (let i = 0; i < p.count; i++) { v.fromBufferAttribute(p, i); c.set(clamp(v.x, -hx, hx), clamp(v.y, -hy, hy), clamp(v.z, -hz, hz)); v.sub(c); if (v.lengthSq() > 1e-12) { v.normalize().multiplyScalar(r).add(c); p.setXYZ(i, v.x, v.y, v.z); } }
  return clayify(gg, amp !== undefined ? amp : .004, freq || 7); }
function lathe(pts, amp, freq, seg) { return clayify(new THREE.LatheGeometry(pts.map(q => new THREE.Vector2(q[0], q[1])), seg || 44), amp !== undefined ? amp : .01, freq || 7); }
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0b0a09);
const camera = new THREE.PerspectiveCamera(27, 16 / 9, .6, 60);
function mesh(gg, m, x, y, z, parent) { const o = new THREE.Mesh(gg, m); o.position.set(x || 0, y || 0, z || 0); o.castShadow = true; o.receiveShadow = true; (parent || scene).add(o); return o; }

/* post: ambient occlusion, lens blur, film (the potter's wheel chain) */
const postScene = new THREE.Scene(), postCam = new THREE.Camera();
const tri = new THREE.Mesh(new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 3, -1, 0, -1, 3, 0]), 3)));
tri.frustumCulled = false; postScene.add(tri);
const VS = 'varying vec2 vUv; void main(){ vUv=position.xy*.5+.5; gl_Position=vec4(position.xy,0.,1.); }';
const pm = (fs, u) => new THREE.ShaderMaterial({ vertexShader: VS, fragmentShader: fs, uniforms: u, depthTest: false, depthWrite: false });
const DEPTH = `uniform sampler2D tDepth; uniform float uNear,uFar; uniform vec2 uProj;
float linZ(vec2 uv){ float d=texture2D(tDepth,uv).x; return (uNear*uFar)/(uFar-d*(uFar-uNear)); }
vec3 vpos(vec2 uv){ float z=linZ(uv); vec2 n=uv*2.-1.; return vec3(n.x/uProj.x*z, n.y/uProj.y*z, -z); }`;
const UU = { tDepth: { value: null }, uNear: { value: .6 }, uFar: { value: 40 }, uProj: { value: new THREE.Vector2() }, uRes: { value: new THREE.Vector2() } };
const aoMat = pm(DEPTH + `uniform vec2 uRes; uniform float uAOs; varying vec2 vUv;
void main(){ vec3 C=vpos(vUv); vec3 n=normalize(cross(dFdx(C),dFdy(C)));
  float R=.2*uAOs, ssr=R*uProj.y*.5/(-C.z);
  float rot=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))))*6.2831853, sum=0.;
  for(int i=0;i<${LOW ? 12 : 22};i++){ float a=(float(i)+.5)/${LOW ? 12 : 22}.; float ang=a*6.2831853*7.+rot; float r=ssr*a;
    vec3 v=vpos(vUv+vec2(cos(ang)*uRes.y/uRes.x,sin(ang))*r)-C; float vv=dot(v,v), vn=dot(v,n);
    float f=max(1.-vv/(R*R),0.); sum+=f*f*f*max((vn+C.z*.0025)/(vv+.0004),0.); }
  gl_FragColor=vec4(vec3(max(0.,1.-sum*5.*R/${LOW ? 12 : 22}.*1.9)),1.); }`, Object.assign({ uAOs: { value: 1 } }, UU));
const prepMat = pm(DEPTH + `uniform sampler2D tColor,tAO; uniform vec2 uRes; uniform float uFocus,uAperture; varying vec2 vUv;
void main(){ float z=linZ(vUv); float ao=0., w=0.;
  for(int x=-2;x<2;x++) for(int y=-2;y<2;y++){ vec2 o=(vec2(float(x),float(y))+.5)/uRes; float zz=linZ(vUv+o);
    float k=max(0.,1.-abs(zz-z)/(.03*z)); ao+=texture2D(tAO,vUv+o).r*k; w+=k; }
  ao = w>0. ? ao/w : 1.;
  vec3 c=texture2D(tColor,vUv).rgb; c*=mix(vec3(.16,.13,.14), vec3(1.), pow(ao,1.25));
  gl_FragColor=vec4(c, clamp((1./uFocus-1./z)*uAperture,-1.,1.)); }`, Object.assign({ tColor: { value: null }, tAO: { value: null }, uFocus: { value: 5 }, uAperture: { value: 6 } }, UU));
const dofMat = pm(`uniform sampler2D tPrep; uniform vec2 uRes; uniform float uMaxPx,uStep; varying vec2 vUv;
void main(){ vec4 c0=texture2D(tPrep,vUv); float cs=abs(c0.a)*uMaxPx; vec3 acc=c0.rgb; float tot=1.; float r=uStep;
  float ang=fract(52.9829189*fract(dot(gl_FragCoord.xy,vec2(.06711056,.00583715))))*6.2831853;
  for(int i=0;i<${LOW ? 90 : 260};i++){ if(r>uMaxPx) break; ang+=2.39996323;
    vec4 s=texture2D(tPrep, vUv+vec2(cos(ang),sin(ang))*r/uRes); float ss=abs(s.a)*uMaxPx;
    if(s.a>c0.a) ss=clamp(ss,0.,cs*2.);
    float m=smoothstep(r-.5,r+.5,ss); acc+=mix(acc/tot,s.rgb,m); tot+=1.; r+=uStep/r; }
  gl_FragColor=vec4(acc/tot,1.); }`, { tPrep: { value: null }, uRes: UU.uRes, uMaxPx: { value: 20 }, uStep: { value: 1.3 } });
const finMat = pm(`uniform sampler2D tDof; uniform vec2 uRes; uniform vec3 uFilm; uniform float uExpo; varying vec2 vUv;
float h(vec2 p){ vec3 q=fract(vec3(p.xyx)*.1031); q+=dot(q,q.yzx+33.33); return fract((q.x+q.y)*q.z); }
vec3 aces(vec3 x){ return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.); }
void main(){ vec2 uv=vUv+uFilm.xy/uRes; vec2 d=uv-.5; float r2=dot(d,d);
  vec2 ca=d*r2*.008;
  vec3 c=vec3(texture2D(tDof,uv+ca).r, texture2D(tDof,uv).g, texture2D(tDof,uv-ca).b);
  vec3 glow=textureLod(tDof,uv,4.).rgb*.5+textureLod(tDof,uv,6.).rgb*.5;
  c=mix(c,glow,.06)+max(glow-.9,0.)*.12;
  c*=uExpo*(1.-r2*.9);
  c=aces(c*vec3(1.03,1.,.97));
  c=pow(c,vec3(1./2.2));
  c=mix(c, c*c*(3.-2.*c), .2);
  float g=h(gl_FragCoord.xy+uFilm.z*91.7)-.5; float l=dot(c,vec3(.3,.6,.1));
  c+=g*.034*(1.-.6*l);
  gl_FragColor=vec4(c,1.); }`, { tDof: { value: null }, uRes: UU.uRes, uFilm: { value: new THREE.Vector3() }, uExpo: { value: 1 } });

let rt = null, budget = TEST && TEST.budget ? TEST.budget : (LOW ? 9e5 : 2.1e6); const msaa = LOW ? 2 : 4;
let aspect = 16 / 9, sizeKey = '';
function build() {
  const cw = canvas.clientWidth || window.innerWidth, chh = canvas.clientHeight || window.innerHeight, dpr = Math.min(window.devicePixelRatio || 1, LOW ? 1.5 : 2);
  const scale = Math.min(dpr, Math.sqrt(budget / (cw * chh)));
  renderer.setPixelRatio(scale); renderer.setSize(cw, chh, false);
  const w = Math.max(2, Math.floor(cw * scale)), h = Math.max(2, Math.floor(chh * scale));
  if (rt) for (const k in rt) rt[k].dispose();
  const T = HDR ? THREE.HalfFloatType : THREE.UnsignedByteType;
  const dt = new THREE.DepthTexture(w, h); dt.type = THREE.UnsignedIntType; dt.minFilter = dt.magFilter = THREE.NearestFilter;
  rt = { scene: new THREE.WebGLRenderTarget(w, h, { type: T, samples: msaa, depthTexture: dt }), ao: new THREE.WebGLRenderTarget(w, h, { type: THREE.UnsignedByteType }), prep: new THREE.WebGLRenderTarget(w, h, { type: T }),
    dof: new THREE.WebGLRenderTarget(w, h, { type: T, generateMipmaps: true, minFilter: THREE.LinearMipmapLinearFilter }) };
  UU.uRes.value.set(w, h); aspect = cw / chh; camera.aspect = aspect; camera.fov = aspect < 1 ? 44 : (aspect < 1.4 ? 33 : 27); camera.updateProjectionMatrix(); sizeKey = cw + 'x' + chh + '@' + budget;
}

/* studio: a tall window to the left, a warm wall behind, the bounce off the bench */
(function () {
  const es = new THREE.Scene(), bm = (c, i) => { const m = new THREE.MeshBasicMaterial({ color: new THREE.Color(c).multiplyScalar(i), side: THREE.DoubleSide }); m.toneMapped = false; return m; };
  es.add(new THREE.Mesh(new THREE.SphereGeometry(30, 24, 16), bm(0x2a2520, .25)));
  const add = (w, h, c, i, x, y, z) => { const p = new THREE.Mesh(new THREE.PlaneGeometry(w, h), bm(c, i)); p.position.set(x, y, z); p.lookAt(0, 0, 0); es.add(p); };
  add(8, 11, 0xf4f1ea, 11, -11, 8, 2); add(4, 11, 0xf4f1ea, 5, -9, 8, -6); add(9, 8, 0xffe2c0, 1.6, 11, 5, 4); add(18, 5, 0xffd9b0, 2.2, 2, 10, -10); add(30, 30, 0x4a3320, .55, 0, -6, 0);
  const pmg = new THREE.PMREMGenerator(renderer); scene.environment = pmg.fromScene(es, .03).texture; pmg.dispose();
})();
const key = new THREE.DirectionalLight(0xfff0dc, 3.3); key.position.set(-7.5, 9.5, 5.5); key.target.position.set(0, .4, -.4);
key.castShadow = true; key.shadow.mapSize.set(LOW ? 2048 : 4096, LOW ? 2048 : 4096);
{ const sc = key.shadow.camera; sc.left = -7.5; sc.right = 7.5; sc.top = 6; sc.bottom = -5; sc.near = 1; sc.far = 30; sc.updateProjectionMatrix(); key.shadow.bias = -.0008; key.shadow.normalBias = .03; key.shadow.radius = .085 * (sc.far - sc.near) / (sc.right - sc.left); }
scene.add(key, key.target);
const rim = new THREE.DirectionalLight(0xffd2a6, .8); rim.position.set(6, 4, -7); scene.add(rim);

/* the set: bench, wall, a shelf of pots out of focus, a tray of spare marbles, a pencil */
mesh(new THREE.BoxGeometry(44, .5, 26), mat({ mode: 'WOOD', color: 0xb9ab9e, rough: .66, spec: .38, env: .6, off: false, bump: 1.2 }), 0, -.25, -2).castShadow = false;
mesh(new THREE.BoxGeometry(44, 16, .4), mat({ color: 0xbfb3a2, rough: .95, spec: .15, env: .5, bump: .5, speck: .2 }), 0, 6, -7.6).castShadow = false;
[[[.05, 0], [.7, .02], [.95, .6], [.85, 1.3], [.5, 1.7], [.42, 2.0], [.5, 2.1]], [[.05, 0], [.55, .02], [.8, .5], [.7, 1.1], [.45, 1.4], [.5, 1.5]], [[.05, 0], [.6, .02], [.75, 1.0], [.72, 1.9], [.78, 2.0]], [[.05, 0], [.45, .02], [.9, .4], [.6, 1.0], [.35, 1.5], [.4, 1.6]]]
  .forEach((p, i) => { p = new THREE.SplineCurve(p.map(q => new THREE.Vector2(q[0], q[1]))).getPoints(40).map(q => [Math.max(.01, q.x), q.y]); const o = mesh(lathe(p, .008, 3, 88), GLZ([0x6f9a8c, 0x2a1a12, 0xd9d2c2, 0x2f4f86][i], .24), [-7.6, -5.9, 6.0, 7.7][i], 0, [-6.0, -6.6, -6.5, -6.0][i]); o.scale.setScalar(.85); });
const tray = new THREE.Group(); tray.position.set(-4.7, 0, -1.5); tray.rotation.y = .35; scene.add(tray);
mesh(rbox(1.5, .08, 1.0, .03), wood(0xe8d2ac), 0, .04, 0, tray);
[[0, .5, 1.5, .06], [0, -.5, 1.5, .06], [.72, 0, .06, 1.0], [-.72, 0, .06, 1.0]].forEach(q => mesh(rbox(q[2], .2, q[3], .02, .002, 7, 4), wood(0xe8d2ac), q[0], .16, q[1], tray));
function glass(col, low) { return (LOW || low) ? new THREE.MeshPhysicalMaterial({ color: col, roughness: .04, metalness: 0, clearcoat: 1, clearcoatRoughness: .03, envMapIntensity: 1.7, specularIntensity: 1, emissive: col, emissiveIntensity: .08 })
  : new THREE.MeshPhysicalMaterial({ color: 0xffffff, roughness: .02, metalness: 0, transmission: 1, thickness: .3, ior: 1.5, attenuationColor: new THREE.Color(col), attenuationDistance: .16, envMapIntensity: 1.5, specularIntensity: 1, clearcoat: 1, clearcoatRoughness: .02 }); }
const sph = new THREE.SphereGeometry(1, 40, 28);
[[-.4, -.2, 0x0d7a70], [-.12, .18, 0xc58a1a], [.2, -.1, 0x8c2f2a], [.42, .22, 0x2f4f86], [-.5, .26, 0xd9d2c2]].forEach(q => { const b = mesh(sph, glass(q[2], true), q[0], .08 + RHO * .9, q[1], tray); b.scale.setScalar(RHO * .9); });
{ const pencil = new THREE.Group(); const body = mesh(new THREE.CylinderGeometry(.035, .035, 1.5, 6), new THREE.MeshPhysicalMaterial({ color: 0x9c2a22, roughness: .35, clearcoat: .6 }), 0, 0, 0, pencil); body.rotation.z = Math.PI / 2;
  const tip = mesh(new THREE.ConeGeometry(.035, .14, 6), new THREE.MeshPhysicalMaterial({ color: 0xd8b98a, roughness: .7 }), .82, 0, 0, pencil); tip.rotation.z = -Math.PI / 2;
  const lead = mesh(new THREE.ConeGeometry(.012, .05, 6), new THREE.MeshPhysicalMaterial({ color: 0x25272b, roughness: .4 }), .875, 0, 0, pencil); lead.rotation.z = -Math.PI / 2;
  pencil.position.set(3.9, .036, -1.35); pencil.rotation.y = .5; scene.add(pencil); }

/* the card: graph paper in an oak frame on a low plinth at the back of the bench */
const CARD = { w: 5.76, h: 3.6, y: 2.8, z: -2.0, tilt: -.1 };
const cardTex = new THREE.CanvasTexture(gc); cardTex.colorSpace = THREE.SRGBColorSpace; cardTex.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy()); cardTex.generateMipmaps = true; cardTex.minFilter = THREE.LinearMipmapLinearFilter;
const cardG = new THREE.Group(); scene.add(cardG);
{ const plinth = mesh(rbox(6.7, .82, .8, .03, .003, 5, 6), wood(0xd9c39e, { grain: 1 }), 0, .41, CARD.z + .05, cardG);
  const tilt = new THREE.Group(); tilt.position.set(0, .82, CARD.z + .12); tilt.rotation.x = CARD.tilt; cardG.add(tilt);
  const paperM = new THREE.MeshPhysicalMaterial({ map: cardTex, roughness: .93, metalness: 0, specularIntensity: .2, envMapIntensity: .7, emissive: 0xffffff, emissiveMap: cardTex, emissiveIntensity: .16 });
  const pp = new THREE.Mesh(new THREE.PlaneGeometry(CARD.w, CARD.h), paperM); pp.position.set(0, CARD.h / 2 + .17, .03); pp.receiveShadow = true; tilt.add(pp);
  mesh(new THREE.BoxGeometry(CARD.w + .3, CARD.h + .3, .05), wood(0x8a7a66, { grain: 1 }), 0, CARD.h / 2 + .17, -.01, tilt);
  const fm = () => wood(0xf0d9b0, { grain: 2.2 }), t = .15;
  mesh(rbox(CARD.w + 2 * t, t, .12, .015, .001, 7, 4), fm(), 0, .17 - t / 2, .05, tilt); mesh(rbox(CARD.w + 2 * t, t, .12, .015, .001, 7, 4), fm(), 0, CARD.h + .17 + t / 2, .05, tilt);
  const s1 = mesh(rbox(CARD.h + 2 * t, t, .12, .015, .001, 7, 4), fm(), -CARD.w / 2 - t / 2, CARD.h / 2 + .17, .05, tilt); s1.rotation.z = Math.PI / 2;
  const s2 = mesh(rbox(CARD.h + 2 * t, t, .12, .015, .001, 7, 4), fm(), CARD.w / 2 + t / 2, CARD.h / 2 + .17, .05, tilt); s2.rotation.z = Math.PI / 2;
  const strut = mesh(new THREE.BoxGeometry(.12, 2.6, .06), wood(0xd9c39e), 0, 1.25, -.42, tilt); strut.rotation.x = -.3; plinth.name = 'plinth'; }

/* a shape pulled along z from a curve: the bell mound and the bowl. The top edge is eased over so that it catches the light. */
function profile(W, depth, bevel, nx) {
  const NX = nx || 120, rows = 10, n = NX + 1, pos = new Float32Array((rows * n + 8) * 3), nor = new Float32Array((rows * n + 8) * 3), idx = [];
  const xs = new Float32Array(n); for (let i = 0; i < n; i++) { const t = i / NX * 2 - 1; xs[i] = W * Math.sign(t) * Math.pow(Math.abs(t), 2.2); }
  const strip = (ra, rb, flip) => { for (let i = 0; i < NX; i++) { const a = ra * n + i, b = ra * n + i + 1, c = rb * n + i, d = rb * n + i + 1; if (flip) idx.push(a, b, c, b, d, c); else idx.push(a, c, b, b, c, d); } };
  strip(0, 1, false); strip(2, 3, false); strip(4, 5, false); strip(6, 7, true); strip(8, 9, true);
  const e0 = rows * n; idx.push(e0, e0 + 1, e0 + 2, e0 + 1, e0 + 3, e0 + 2, e0 + 4, e0 + 6, e0 + 5, e0 + 5, e0 + 6, e0 + 7);
  const geo = new THREE.BufferGeometry(); geo.setAttribute('position', new THREE.BufferAttribute(pos, 3)); geo.setAttribute('normal', new THREE.BufferAttribute(nor, 3)); geo.setIndex(idx);
  const ys = new Float32Array(n);
  function set(fn, d2) { const hd = (d2 || depth) / 2, bv = Math.min(bevel, hd * .5);
    for (let i = 0; i < n; i++) ys[i] = fn(xs[i]);
    for (let i = 0; i < n; i++) { const i0 = Math.max(0, i - 1), i1 = Math.min(NX, i + 1), sl = (ys[i1] - ys[i0]) / (xs[i1] - xs[i0]), L = Math.hypot(sl, 1), nx_ = -sl / L, ny_ = 1 / L, x = xs[i], y = ys[i], yb = y - Math.min(bv, y * .6);
      const put = (r, X, Y, Z, a, b, c) => { const o = (r * n + i) * 3; pos[o] = X; pos[o + 1] = Y; pos[o + 2] = Z; nor[o] = a; nor[o + 1] = b; nor[o + 2] = c; };
      put(0, x, y, -hd + bv, nx_, ny_, 0); put(1, x, y, hd - bv, nx_, ny_, 0);
      put(2, x, y, hd - bv, nx_ * .7, ny_ * .7, .7); put(3, x, yb, hd, nx_ * .7, ny_ * .7, .7); put(4, x, yb, hd, 0, 0, 1); put(5, x, 0, hd, 0, 0, 1);
      put(6, x, y, -hd + bv, nx_ * .7, ny_ * .7, -.7); put(7, x, yb, -hd, nx_ * .7, ny_ * .7, -.7); put(8, x, yb, -hd, 0, 0, -1); put(9, x, 0, -hd, 0, 0, -1); }
    const e = e0 * 3, yl = ys[0], yr = ys[NX], q = [[-W, 0, -hd, -1], [-W, 0, hd, -1], [-W, yl, -hd, -1], [-W, yl, hd, -1], [W, 0, -hd, 1], [W, 0, hd, 1], [W, yr, -hd, 1], [W, yr, hd, 1]];
    q.forEach((v, j) => { pos[e + j * 3] = v[0]; pos[e + j * 3 + 1] = v[1]; pos[e + j * 3 + 2] = v[2]; nor[e + j * 3] = v[3]; nor[e + j * 3 + 1] = 0; nor[e + j * 3 + 2] = 0; });
    geo.attributes.position.needsUpdate = true; geo.attributes.normal.needsUpdate = true; geo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 1, 0), 3); }
  return { geo, set };
}
const softmin = (a, b) => -Math.log(Math.exp(-14 * a) + Math.exp(-14 * b)) / 14;
const bowlTop = P => { const a = HS / (2 * P * U * U); return x => BASE + Math.max(0, softmin(HB, a * x * x)); };
function restY(P, rho) { const a = HS / (2 * P * U * U); if (rho <= 1 / (2 * a)) return BASE + rho; const xc2 = rho * rho - 1 / (4 * a * a), yc = a * xc2; if (yc <= HB) return BASE + a * rho * rho + 1 / (4 * a); const xr = Math.sqrt(HB / a); return BASE + HB + Math.sqrt(Math.max(0, rho * rho - xr * xr)); }
function makeBowl(col, depth) { const p = profile(WB, depth, .035), m = mesh(p.geo, GLZ(col, .18)); m.userData.set = P => p.set(bowlTop(P)); return m; }
function makeMarble(col, vane) { const gm = new THREE.Group(), b = mesh(sph, glass(col), 0, 0, 0, gm); b.scale.setScalar(RHO);
  const vg = new THREE.PlaneGeometry(RHO * 1.3, RHO * .62, 12, 1), vp = vg.attributes.position; for (let i = 0; i < vp.count; i++) { const x = vp.getX(i), y = vp.getY(i), a = x / (RHO * .65) * 1.3, w = Math.cos(x / (RHO * .65) * 1.45); vp.setXYZ(i, x, y * w * Math.cos(a), y * w * Math.sin(a)); } vg.computeVertexNormals();
  const v = new THREE.Mesh(vg, new THREE.MeshPhysicalMaterial({ color: vane, roughness: .3, side: THREE.DoubleSide, emissive: vane, emissiveIntensity: .25 })); gm.add(v); gm.userData.ball = b; scene.add(gm); return gm; }
const rollTo = (gm, x, y, z) => { gm.position.set(x, y, z); gm.rotation.set(0, 0, -x / RHO); };

/* ── scenes A and D: the pin, the bell, the bowl ── */
const gA = new THREE.Group(), gD = new THREE.Group(), gComb = new THREE.Group(), gC = new THREE.Group(), gE = new THREE.Group(), gF = new THREE.Group(); scene.add(gA, gD, gComb, gC, gE, gF);
const headGeo = new THREE.SphereGeometry(.05, 20, 14);
function makePin(h, col, parent) { const p = new THREE.Group(); const sh = mesh(new THREE.CylinderGeometry(.013, .013, h, 8), steel, 0, h / 2, 0, p);
  const hm = new THREE.MeshPhysicalMaterial({ color: col, roughness: .08, clearcoat: 1, envMapIntensity: 1.6, emissive: 0xffc46a, emissiveIntensity: 0 }); mesh(headGeo, hm, 0, h + .03, 0, p); p.userData.head = hm; p.userData.shaft = sh; (parent || scene).add(p); return p; }
const bellP = profile(WB, .5, .03), bellM = mesh(bellP.geo, mat({ color: 0xb98a62, rough: .62, spec: .6, bump: 1.1, smear: .3, color2: 0x8f6a4c, side: THREE.DoubleSide }));
const bellTop = s => x => .028 + BS * M.bell(x / U, s);
const pinA = makePin(BS * M.bell(0, SMIN) + .06, 0x0d6a62, gA);
const bowlD = makeBowl(0x7fa697, .8), marbleD = makeMarble(0x0d7a70, 0xe9e2c8);
gD.add(bowlD);

/* ── scenes B and C: the comb, the marble, the staircase ── */
{ const rc = document.createElement('canvas'); rc.width = 2048; rc.height = 128; const c = rc.getContext('2d'); c.fillStyle = '#fff'; c.fillRect(0, 0, 2048, 128); c.fillStyle = '#3a2a10'; c.strokeStyle = '#3a2a10';
  const L = 6.3; for (let i = 0; i <= 90; i++) { const x = (PINX(0) + (i - 10) * .07 + L / 2) / L * 2048, big = i % 10 === 0; if (x < 6 || x > 2042) continue; c.fillRect(x - (big ? 2 : 1), 128 - (big ? 60 : i % 5 === 0 ? 42 : 26), big ? 4 : 2, 128); if (big && i >= 10) { c.font = '44px ' + SERIF; c.textAlign = 'center'; c.fillText(String(i / 10), x + 26, 52); } }
  const rt2 = new THREE.CanvasTexture(rc); rt2.colorSpace = THREE.SRGBColorSpace; rt2.anisotropy = 8;
  const bm = brass(.3); const top = new THREE.MeshPhysicalMaterial({ color: 0xd2a650, roughness: .3, metalness: 1, envMapIntensity: 1.25, map: rt2 });
  const rule = new THREE.Mesh(new THREE.BoxGeometry(L, .035, .42), [bm, bm, top, bm, bm, bm]); rule.position.set(0, .0175, -.1); rule.castShadow = true; rule.receiveShadow = true; gComb.add(rule);
  mesh(rbox(6.9, .05, .26, .012, .001, 7, 4), wood(0x9a7350, { grain: 2 }), 0, .025, .42, gComb); }
const pins = []; for (let i = 0; i < 8; i++) { const p = makePin(.95, 0x0d6a62, gComb); p.position.set(PINX(i), .03, -.22); pins.push(p); }
const marbleB = makeMarble(0x0d7a70, 0xe9e2c8);
const blockGeo = rbox(.66, .2, .56, .018, .0015, 9, 5), blocks = [];
for (let i = 0; i < 8; i++) for (let j = 0; j <= i; j++) { const top = j === i, b = mesh(blockGeo, wood(top ? 0xffe9bd : [0xc79a68, 0xb88a5c, 0xd0a674][(i * 3 + j) % 3], { grain: 2.4 }), 0, 0, 0, gC);
  b.userData = { i, j, x: PINX(i) + .35 + (h3(i, j, 3) - .5) * .02, y: .1 + j * .2, z: -.55 + (h3(i, j, 4) - .5) * .03, ry: (h3(i, j, 5) - .5) * .05 }; blocks.push(b); }

/* ── scenes E, F, G: rails, bowls, marbles, the lantern, the fan ── */
function rails(z, parent) { const r = brass(.22); [-.26, .26].forEach(dz => { const m = mesh(new THREE.CylinderGeometry(.028, .028, 7.4, 12), r, 0, .055, z + dz, parent); m.rotation.z = Math.PI / 2; });
  [-3.5, -1.75, 0, 1.75, 3.5].forEach(x => mesh(rbox(.2, .05, .82, .012, .001, 7, 3), wood(0x8a6444, { grain: 2 }), x, .025, z, parent)); }
rails(0, gE); rails(-.52, gF); rails(.52, gF);
const RAIL = .083;
const bowlE = makeBowl(0x7fa697, .8), marbleE = makeMarble(0x0d7a70, 0xe9e2c8); gE.add(bowlE);
const bowlFa = makeBowl(0xcfb27a, .7), bowlFb = makeBowl(0x7fa697, .7), marbleFa = makeMarble(0xc58a1a, 0xfff4d6), marbleFb = makeMarble(0x0d7a70, 0xe9e2c8); gF.add(bowlFa, bowlFb);
{ const pa = makePin(.55, 0xb98619, gF); pa.position.set(-6 * U, 0, -1.02); const pb = makePin(.55, 0x0d6a62, gF); pb.position.set(6 * U, 0, 1.02); }
const ZL = 1.8, ZW = 1.02;
const lantern = new THREE.Group(); gE.add(lantern);
{ const b = brass(.3); mesh(new THREE.CylinderGeometry(.17, .19, .05, 24), b, 0, .025, 0, lantern); mesh(new THREE.CylinderGeometry(.15, .15, .03, 24), b, 0, .4, 0, lantern);
  mesh(new THREE.ConeGeometry(.17, .13, 24), b, 0, .48, 0, lantern); const ring = mesh(new THREE.TorusGeometry(.05, .01, 8, 20), b, 0, .58, 0, lantern);
  for (let i = 0; i < 4; i++) { const a = i * Math.PI / 2 + .78; mesh(new THREE.CylinderGeometry(.01, .01, .36, 6), b, Math.cos(a) * .14, .22, Math.sin(a) * .14, lantern); }
  const gl = new THREE.Mesh(new THREE.CylinderGeometry(.115, .115, .33, 20, 1, true), new THREE.MeshPhysicalMaterial({ color: 0xfff1d0, roughness: .15, transparent: true, opacity: .22, emissive: 0xffb050, emissiveIntensity: .5, side: THREE.DoubleSide, depthWrite: false })); gl.position.y = .22; gl.renderOrder = 3; lantern.add(gl);
  const flame = new THREE.Mesh(new THREE.SphereGeometry(.05, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd9a0 })); flame.scale.set(1, 1.7, 1); flame.position.y = .2; lantern.add(flame); lantern.userData.flame = flame; lantern.userData.glass = gl; }
const lampL = new THREE.PointLight(0xffa64a, 0, 5, 1.8); lampL.position.set(0, .3, ZL); gE.add(lampL);
{ const tr = mesh(new THREE.BoxGeometry(7.4, .012, .1), brass(.35), 0, .006, ZL, gE); tr.castShadow = false;
  const w = mesh(new THREE.CylinderGeometry(.006, .006, 7.2, 6), steel, 0, .2, ZW, gE); w.rotation.z = Math.PI / 2; w.castShadow = false;
  [-3.6, 3.6].forEach(x => { mesh(new THREE.CylinderGeometry(.03, .04, .26, 12), brass(.3), x, .13, ZW, gE); }); }
const beads = []; for (let i = 0; i < 4; i++) { const m = new THREE.Mesh(new THREE.SphereGeometry(.055, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff })); gE.add(m); beads.push(m); }
const beadL = new THREE.PointLight(0xffe2b0, 0, 2.2, 2); gE.add(beadL);
const fanU = { uX: { value: 0 }, uSd: { value: .3 }, uI: { value: 1 }, uZ0: { value: .46 }, uZ1: { value: 2.25 } };
const fan = new THREE.Mesh(new THREE.PlaneGeometry(8, 2.2, 1, 1), new THREE.ShaderMaterial({ uniforms: fanU, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  vertexShader: 'varying vec3 vW; void main(){ vec4 w=modelMatrix*vec4(position,1.); vW=w.xyz; gl_Position=projectionMatrix*viewMatrix*w; }',
  fragmentShader: `uniform float uX,uSd,uI,uZ0,uZ1; varying vec3 vW;
  void main(){ float t=clamp((vW.z-uZ0)/(uZ1-uZ0),0.,1.); float sd=.035+uSd*t; float d=(vW.x-uX)/sd; float body=exp(-.5*d*d);
    float edge=exp(-pow((abs(d)-2.)*sd/.022,2.))*.35;
    float fall=smoothstep(0.,.06,t)*smoothstep(1.,.86,t);
    float a=(body*min(1.,.34/sd)+edge)*fall*uI; gl_FragColor=vec4(vec3(1.,.74,.38)*a,1.); }` }));
fan.rotation.x = -Math.PI / 2; fan.position.set(0, .004, 1.35); fan.renderOrder = 2; gE.add(fan);

/* ───────────────────────── put everything where the numbers say ───────────────────────── */
const BOX = { A: [-1.5, 1.5, 0, 2.2, -.4, .4], B: [-3.55, 3.55, 0, 1.05, -.4, .65], C: [-3.55, 3.55, 0, 1.85, -.95, 1.35], D: [-2.85, 2.85, 0, 2.15, -.5, .5], E: [-3.8, 3.8, 0, .95, -.5, 2.3], F: [-3.8, 3.8, 0, .95, -1, 1], G: [-1.45, 1.45, 0, 1.0, -.5, .6] };
const PIT = { A: .2, B: .34, C: .34, D: .28, E: .42, F: .38, G: .26 }, PITC = { A: .15, B: .2, C: .22, D: .17, E: .26, F: .22, G: .2 }, APER = { A: 6, B: 6, C: 5, D: 6, E: 3.5, F: 5, G: 6 };
const CARDBOX = [-3.2, 3.2, 0, 4.72, -2.5, -1.75];
const corners = (b, dx) => { const o = []; for (let i = 0; i < 8; i++) o.push([(i & 1 ? b[1] : b[0]) + (dx || 0), i & 2 ? b[3] : b[2], i & 4 ? b[5] : b[4]]); return o; };
/* the nearest camera, looking along (yaw, pitch), that keeps every point in shot */
function fit(pts, yaw, pitch, margin, zoom) {
  const cp = Math.cos(pitch), sp = Math.sin(pitch), sy = Math.sin(yaw), cy = Math.cos(yaw), f = [-sy * cp, -sp, -cy * cp], r = [cy, 0, -sy], u = [sy * f[1], -sy * f[0] - cy * f[2], cy * f[1]];
  const tv = Math.tan(camera.fov * Math.PI / 360) * (1 - margin), th = tv * aspect / (1 - margin) * (1 - margin * .6), dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
  const A = pts.map(p => dot(p, r)), B = pts.map(p => dot(p, u)), C = pts.map(p => dot(p, f)); let cc = 1e9, mc = 0;
  for (let i = 0; i < pts.length; i++) { mc += C[i] / pts.length; for (let j = 0; j < pts.length; j++) { cc = Math.min(cc, (C[i] + C[j] - (B[i] - B[j]) / tv) / 2, (C[i] + C[j] - (A[i] - A[j]) / th) / 2); } }
  const mid = (V, t) => { let lo = -1e9, hi = 1e9; for (let i = 0; i < pts.length; i++) { lo = Math.max(lo, V[i] - (C[i] - cc) * t); hi = Math.min(hi, V[i] + (C[i] - cc) * t); } return (lo + hi) / 2; };
  const ca = mid(A, th), cb = mid(B, tv); cc += (mc - cc) * (1 - 1 / zoom);
  return { pos: [ca * r[0] + cb * u[0] + cc * f[0], ca * r[1] + cb * u[1] + cc * f[1], ca * r[2] + cb * u[2] + cc * f[2]], f, depth: mc - cc };
}
let focusDist = 8, aperture = 6;
const V3 = new THREE.Vector3();
function place(S) {
  const sc = S.scene, k = S.card;
  gA.visible = sc === 'A'; gD.visible = sc === 'D'; bellM.visible = sc === 'A' || sc === 'D'; marbleD.visible = sc === 'D'; gComb.visible = sc === 'B' || sc === 'C'; marbleB.visible = gComb.visible; gC.visible = sc === 'C';
  gE.visible = sc === 'E' || sc === 'G'; marbleE.visible = gE.visible; gF.visible = sc === 'F'; marbleFa.visible = marbleFb.visible = sc === 'F'; cardG.visible = k > .001;
  let tx = 0;
  if (sc === 'A') { const s = S.width, th = .04 + .46 * sstep(SMIN, 1.2, s); bellM.visible = !S.point; bellM.position.set(0, 0, 0); bellP.set(bellTop(s), th); }
  else if (sc === 'D') { const s = S.width; bellM.position.set(-1.55, 0, 0); bellP.set(bellTop(s), .5); bowlD.position.set(1.55, 0, 0); bowlD.userData.set(s * s); marbleD.position.set(1.55, restY(s * s, RHO), 0); }
  else if (sc === 'B' || sc === 'C') { const zc = sc === 'C' ? .75 : 0; gComb.position.z = zc; rollTo(marbleB, S.mx, .05 + RHO - .004, zc + .42);
    pins.forEach((p, i) => { const age = S.lf - S.passFrame(i), on = age >= 0 ? Math.exp(-age / 9) : 0; p.userData.head.emissiveIntensity = on * 7; p.rotation.z = on * .09 * Math.sin(age * 1.1); p.userData.head.color.setHex(age >= 0 ? 0xb98619 : 0x0d6a62); });
    if (sc === 'C') blocks.forEach(b => { const u = b.userData, age = S.lf - S.passFrame(u.i) - u.j * 1.6, e = clamp(age / 11, 0, 1), ez = 1 - (1 - e) * (1 - e) * (1 - e); b.visible = age > 0; b.position.set(u.x, u.y + (1 - ez) * .55, u.z); b.rotation.y = u.ry; b.scale.setScalar(.72 + .28 * ez); }); }
  else if (sc === 'E' || sc === 'G') { const o = S.centre, bx = (S.m - o) * U, lx = (S.world - o) * U, run = S.run;
    bowlE.position.set(bx, RAIL, 0); bowlE.userData.set(S.P); rollTo(marbleE, bx, RAIL + restY(S.P, RHO), 0);
    const fl = .82 + .18 * Math.sin(S.frame * .9) * Math.sin(S.frame * .37 + 1.3) + S.flash * .9;
    lantern.position.set(lx, .012, ZL); lampL.position.x = lx; lampL.intensity = 2.2 * fl; lantern.userData.flame.material.color.setRGB(3.2 * fl, 2.1 * fl, 1.0 * fl); lantern.userData.glass.emissiveIntensity = .5 * fl;
    let bl = 0, blx = 0; beads.forEach((b, i) => { const j = S.k - i; if (j < 1) { b.visible = false; return; } const age = S.t - j, I = Math.exp(-age * 1.5) * (age < 2.6 ? 1 : 0), pop = sstep(0, .08, age); b.visible = I > .02; b.position.set((run.y[j] - o) * U, .2, ZW); b.scale.setScalar((.55 + .6 * I) * pop); b.material.color.setRGB(1 + 9 * I, .8 + 6 * I, .5 + 2.6 * I); if (i === 0) { bl = I; blx = b.position.x; } });
    beadL.position.set(blx, .26, ZW); beadL.intensity = bl * 1.6;
    fanU.uX.value = bx; fanU.uSd.value = Math.sqrt(S.P) * U; fanU.uI.value = sc === 'G' ? .5 : .9;
    if (sc === 'G') { const t = S.t, kk = S.k, a = eAt(run, Math.max(0, t - .5)).m, b = eAt(run, t + .5 > run.n + 1 ? t : Math.min(t + .5, run.n + .99)).m; tx = ((a + b) / 2 - o) * U; }
  } else { bowlFa.position.set(S.a * U, RAIL, -.52); bowlFa.userData.set(S.Pa); rollTo(marbleFa, S.a * U, RAIL + restY(S.Pa, RHO), -.52);
    bowlFb.position.set(S.b * U, RAIL, .52); bowlFb.userData.set(S.Pb); rollTo(marbleFb, S.b * U, RAIL + restY(S.Pb, RHO), .52); }
  /* camera: between a close framing of the objects (card 0) and one that takes the card in (card 1) */
  const kk = sstep(0, 1, k), pitch = lerp(PIT[sc], PITC[sc], kk) + S.pitch, yaw = S.yaw, ob = corners(BOX[sc], tx);
  const c0 = fit(ob, yaw, pitch, .07, S.zoom), c1 = kk > 0 ? fit(ob.concat(corners(CARDBOX, 0)), yaw, pitch, .035, S.zoom) : c0;
  camera.position.set(lerp(c0.pos[0], c1.pos[0], kk), lerp(c0.pos[1], c1.pos[1], kk), lerp(c0.pos[2], c1.pos[2], kk));
  camera.lookAt(camera.position.x + c0.f[0], camera.position.y + c0.f[1], camera.position.z + c0.f[2]);
  const b0 = BOX[sc], fo = V3.set((b0[0] + b0[1]) / 2 + tx, (b0[2] + b0[3]) / 3, (b0[4] + b0[5]) / 2).distanceTo(camera.position), fc = V3.set(0, CARD.y, CARD.z).distanceTo(camera.position);
  focusDist = S.focus !== undefined ? M.param(S.focus, S.frame, fo) : lerp(fo, (fo + fc) / 2, kk); aperture = lerp(APER[sc], 1.6, kk);
}
let frameNo = 0;
function render() {
  renderer.setRenderTarget(rt.scene); renderer.render(scene, camera);
  const PM = camera.projectionMatrix.elements;
  for (const m of [aoMat, prepMat]) { m.uniforms.tDepth.value = rt.scene.depthTexture; m.uniforms.uNear.value = camera.near; m.uniforms.uFar.value = camera.far; m.uniforms.uProj.value.set(PM[0], PM[5]); }
  tri.material = aoMat; renderer.setRenderTarget(rt.ao); renderer.render(postScene, postCam);
  prepMat.uniforms.tColor.value = rt.scene.texture; prepMat.uniforms.tAO.value = rt.ao.texture; prepMat.uniforms.uFocus.value = focusDist; prepMat.uniforms.uAperture.value = aperture;
  tri.material = prepMat; renderer.setRenderTarget(rt.prep); renderer.render(postScene, postCam);
  dofMat.uniforms.tPrep.value = rt.prep.texture; dofMat.uniforms.uMaxPx.value = UU.uRes.value.y * .021; dofMat.uniforms.uStep.value = Math.max(1.1, UU.uRes.value.y / 820);
  tri.material = dofMat; renderer.setRenderTarget(rt.dof); renderer.render(postScene, postCam);
  finMat.uniforms.tDof.value = rt.dof.texture; finMat.uniforms.uFilm.value.set((rnd(frameNo, 50) - .5) * .6, (rnd(frameNo, 51) - .5) * .6, rnd(frameNo, 52) * 100); finMat.uniforms.uExpo.value = 1.0;
  tri.material = finMat; renderer.setRenderTarget(null); renderer.render(postScene, postCam);
}
return { draw(S) { const cw = canvas.clientWidth || window.innerWidth, chh = canvas.clientHeight || window.innerHeight; if (!rt || sizeKey !== cw + 'x' + chh + '@' + budget) build(); frameNo = Math.round(S.frame); place(S); if (cardG.visible) cardTex.needsUpdate = true; render(); },
  finish() { renderer.getContext().finish(); }, slower() { if (budget > 4e5) { budget *= .72; return true; } return false; }, renderer };
}

/* one picture: numbers, card, bench */
function picture(frame, view) { const S = resolve(frame, view); lastS = S; drawCard(S); if (GL) GL.draw(S); return S; }

/* ───────────────────────── film hook: no loop, no page furniture ───────────────────────── */
if (TEST) { window.__clay = { shot(frame, view) { picture(frame, view || {}); if (GL) GL.finish(); return true; }, scenes: SCENES.map(s => s[0]), fps: 60, meeting: TE, loop: LOOP }; return; }

/* ───────────────────────── the page ───────────────────────── */
const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const ui = { scene: 'A', frame: 0, playing: !reduce, card: 1, yaw: 0, pitch: 0, wA: SMIN, wD: 1, K: [[0, NaN]], r: [[0, 1]], Kt: [[0, .382]], Kl: [[0, .618]], sound: false };
$('cardslot').appendChild(gc); gc.setAttribute('aria-label', 'The card of graph paper from the back of the bench, shown flat. It carries the curve for the scene chosen, drawn from the same numbers as the objects.');
const sBtns = {}; SCENES.forEach(([id, name]) => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn'; b.innerHTML = '<b>' + id + '</b>' + name; b.setAttribute('aria-pressed', 'false'); b.onclick = () => setScene(id); $('scenes').appendChild(b); sBtns[id] = b; });
const mkPresets = (el, set, fn) => Object.keys(set).forEach(k => { const b = document.createElement('button'); b.type = 'button'; b.className = 'btn'; b.textContent = set[k].name; b.onclick = () => fn(k, set[k]); $(el).appendChild(b); });
const w2s = t => SMIN * Math.pow(SMAX / SMIN, t), s2w = s => Math.log(s / SMIN) / Math.log(SMAX / SMIN);
const last = a => a[a.length - 1][1];
function push(a, v) { if (a[a.length - 1][0] >= ui.frame) a[a.length - 1][1] = v; else a.push([ui.frame, v]); dirty = true; }
function restart() { ui.frame = 0; ['K', 'r', 'Kt', 'Kl'].forEach(k => { ui[k] = [[0, last(ui[k])]]; }); evt = -1; dirty = true; }
function view() { const sc = ui.scene; return { scene: sc, card: ui.card, yaw: ui.yaw, pitch: ui.pitch, width: sc === 'A' ? ui.wA : ui.wD, K: sc === 'E' ? { steps: ui.K } : undefined, r: { steps: sc === 'G' ? [[0, 1]] : ui.r }, Kt: { steps: ui.Kt }, Kl: { steps: ui.Kl } }; }
function setScene(id) { ui.scene = id; for (const k in sBtns) sBtns[k].setAttribute('aria-pressed', String(k === id)); $('say').textContent = SAY[id];
  $('ctlW').hidden = !(id === 'A' || id === 'D'); $('ctlE').hidden = $('ctlEp').hidden = id !== 'E'; $('ctlF').hidden = $('ctlFp').hidden = id !== 'F';
  $('play').hidden = $('again').hidden = (id === 'A' || id === 'D');
  if (id === 'A' || id === 'D') { $('width').value = s2w(id === 'A' ? ui.wA : ui.wD); }
  ui.yaw = ui.pitch = 0; restart(); if (reduce && id !== 'A' && id !== 'D') ui.frame = id === 'B' || id === 'C' ? 300 : TE * 14 + 30; sync(); }
function sync() { const sc = ui.scene; $('widthOut').textContent = (sc === 'A' ? ui.wA : ui.wD).toFixed(2); const K = last(ui.K), auto = !(K === K);
  $('auto').setAttribute('aria-pressed', String(auto)); const r = last(ui.r); $('rOut').textContent = r.toFixed(2); $('r').value = Math.log10(r);
  if (!auto) { $('K').value = K; $('KOut').textContent = K.toFixed(3); }
  $('Kt').value = last(ui.Kt); $('Kl').value = last(ui.Kl); $('KtOut').textContent = last(ui.Kt).toFixed(3); $('KlOut').textContent = last(ui.Kl).toFixed(3);
  $('play').textContent = ui.playing ? 'Pause' : 'Play'; $('play').setAttribute('aria-pressed', String(ui.playing)); $('cardBtn').setAttribute('aria-pressed', String(ui.card > .5)); $('sound').textContent = ui.sound ? 'Sound on' : 'Sound off'; $('sound').setAttribute('aria-pressed', String(ui.sound)); dirty = true; }
$('width').oninput = e => { const s = w2s(+e.target.value); if (ui.scene === 'A') ui.wA = s; else ui.wD = s; sync(); };
$('K').oninput = e => { push(ui.K, +e.target.value); sync(); };
$('auto').onclick = () => { push(ui.K, NaN); sync(); };
$('r').oninput = e => { push(ui.r, Math.pow(10, +e.target.value)); sync(); };
$('Kt').oninput = e => { push(ui.Kt, +e.target.value); sync(); }; $('Kl').oninput = e => { push(ui.Kl, +e.target.value); sync(); };
mkPresets('ctlEp', M.PRESETS.E, (k, p) => { ui.K = [[0, p.K === null ? NaN : p.K]]; ui.r = [[0, p.r]]; restart(); sync(); $('say').textContent = p.say; });
mkPresets('ctlFp', M.PRESETS.F, (k, p) => { ui.Kt = [[0, p.Kt]]; ui.Kl = [[0, p.Kl]]; restart(); sync(); $('say').textContent = SAYF[k]; });
$('play').onclick = () => { ui.playing = !ui.playing; sync(); }; $('again').onclick = () => { restart(); };
$('cardBtn').onclick = () => { ui.card = ui.card > .5 ? 0 : 1; sync(); };
/* sound: a soft tick, only after it has been asked for */
let AC = null; $('sound').onclick = () => { ui.sound = !ui.sound; if (ui.sound && !AC) { try { AC = new (window.AudioContext || window.webkitAudioContext)(); } catch (e) { ui.sound = false; } } if (AC && ui.sound) AC.resume(); sync(); };
function tick(f, vol) { if (!AC || !ui.sound) return; const t = AC.currentTime, o = AC.createOscillator(), gn = AC.createGain(); o.type = 'sine'; o.frequency.setValueAtTime(f, t); o.frequency.exponentialRampToValueAtTime(f * .6, t + .12); gn.gain.setValueAtTime(vol, t); gn.gain.exponentialRampToValueAtTime(.0005, t + .22); o.connect(gn).connect(AC.destination); o.start(t); o.stop(t + .25); }
/* drag to look round */
let down = null; canvas.addEventListener('pointerdown', e => { down = { x: e.clientX, y: e.clientY }; });
window.addEventListener('pointermove', e => { if (!down) return; const dx = e.clientX - down.x, dy = e.clientY - down.y; if (Math.abs(dx) < 3 && Math.abs(dy) < 3) return; ui.yaw = clamp(ui.yaw - dx * .004, -.7, .7); ui.pitch = clamp(ui.pitch + dy * .003, -.12, .4); down = { x: e.clientX, y: e.clientY }; dirty = true; });
window.addEventListener('pointerup', () => { down = null; }); window.addEventListener('pointercancel', () => { down = null; });
canvas.tabIndex = 0; canvas.addEventListener('keydown', e => { const k = e.key; if (k === 'ArrowLeft') ui.yaw = clamp(ui.yaw + .06, -.7, .7); else if (k === 'ArrowRight') ui.yaw = clamp(ui.yaw - .06, -.7, .7); else if (k === 'ArrowUp') ui.pitch = clamp(ui.pitch + .04, -.12, .4); else if (k === 'ArrowDown') ui.pitch = clamp(ui.pitch - .04, -.12, .4); else return; e.preventDefault(); dirty = true; });
/* readouts */
const fmt = (v, d) => v === v && v !== undefined && v !== null ? v.toFixed(d === undefined ? 3 : d) : '–';
function readouts(S) { let rows;
  if (S.scene === 'A' || S.scene === 'D') { rows = [['Width', fmt(S.width, 2)], ['Height at the centre', fmt(S.height, 2)], ['Weight (area under the bell)', fmt(S.area, 3)]]; if (S.scene === 'D') rows.push(['Bowl height one step away', fmt(S.bowlAt1, 2)]); }
  else if (S.scene === 'B') rows = [['Teeth passed', String(S.passed)], ['Teeth in the comb', '8']];
  else if (S.scene === 'C') rows = [['Teeth passed', String(S.passed)], ['The count', String(S.count)]];
  else if (S.scene === 'F') rows = [['Gold gives', fmt(S.Kt)], ['Green gives', fmt(S.Kl)], ['Gap left after an exchange', fmt(S.factor * 100, 1) + '%', 'setting: 1 minus the two gains'], ['Gap now', fmt(S.gap, 2)], ['Exchanges', String(S.meetings)], ['Times they swapped sides', String(S.swaps)], ['Mean gap just after', fmt(S.meanAbs, 2)], ['Typical gap just before', fmt(S.rmsPre, 2)], ['First exchange: gold moved', S.firstShare === S.firstShare ? fmt(S.firstShare * 100, 0) + '%' : '\u2013', 'of the distance closed'], ['First exchange: green moved', S.firstShare === S.firstShare ? fmt((1 - S.firstShare) * 100, 0) + '%' : '\u2013', 'of the distance closed']];
  else rows = [['Weight kept', fmt(S.kept)], ['Weight given', fmt(S.K), S.auto ? 'found by the filter' : 'set by hand'], ['The cone it expects', fmt(S.before), 'uncertainty before a meeting' + (S.auto ? '' : ', by its own reckoning')], ['The surprise it gets', fmt(S.surprise), 'measured: mean squared miss, one step ahead'], ['Uncertainty after', fmt(S.after), S.auto ? '' : 'by its own reckoning'], ['Mean squared error so far', fmt(S.mse), S.meetings + (S.meetings === 1 ? ' meeting' : ' meetings')], ['Path walked / the world\u2019s path', fmt(S.pathRatio, 2)], ['World inside the cone', S.caught === S.caught ? Math.round(S.caught * 100) + '%' : '–', 'of meetings']];
  $('read').innerHTML = rows.map(r => '<div><dt>' + r[0] + '</dt><dd>' + r[1] + (r[2] ? '<small>' + r[2] + '</small>' : '') + '</dd></div>').join(''); }
/* loop */
let dirty = true, evt = -1, visible = true, lastT = 0, slow = 0, lastRead = 0;
if ('IntersectionObserver' in window) new IntersectionObserver(es => { visible = es[0].isIntersecting; }, { threshold: 0 }).observe(document.querySelector('.stage'));
window.addEventListener('resize', () => { dirty = true; });
function loop(t) { requestAnimationFrame(loop); const dt = Math.min(100, t - lastT); lastT = t; if (document.hidden || !visible) return;
  const moving = ui.playing && !(ui.scene === 'A' || ui.scene === 'D'); if (moving) ui.frame += dt * .06;
  if (!moving && !dirty) return; dirty = false;
  const t0 = performance.now(), S = picture(ui.frame, view());
  if (GL && moving) { GL.finish(); const ms = performance.now() - t0; if (ms > 40) { if (++slow > 6 && GL.slower()) slow = 0; } else slow = Math.max(0, slow - 1); }
  if (t - lastRead > 120 || !moving) { readouts(S); lastRead = t; if (ui.scene === 'E' && S.auto) { $('K').value = S.K; $('KOut').textContent = S.K.toFixed(3); } }
  const ev = (S.scene === 'B' || S.scene === 'C') ? S.passed + 100 * Math.floor(ui.frame / LOOP) : (S.meetings || 0);
  if (moving && evt >= 0 && ev > evt) { if (S.scene === 'B' || S.scene === 'C') tick(520 + 60 * S.passed, .16); else tick(S.scene === 'F' ? 330 : 392, .1); } evt = ev; }
setScene('A'); requestAnimationFrame(loop);
})();
