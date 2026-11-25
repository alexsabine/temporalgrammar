
/* VIEW, part 5: the section "Time on this page: a CRR reading". Four figures, drawn as SVG from the model (EV.TIME), and their
   controls. Nothing here moves unless the visitor asks, and every number shown is taken from the model at the moment of drawing. */
var TM = null;
function timeSync() { if (TM) TM.sync(); }
(() => {
  const T = EV.TIME, PI = Math.PI, Gyr = K.Gyr, CO = EV.COSMO;
  const GOLD = '#e7c75a', GREEN = '#7fd6a4', RED = '#ff8a66', RUST = '#e0764a', BLUE = '#8fc4ff', INKC = 'rgba(241,238,233,.92)', DIMC = 'rgba(241,238,233,.62)', FAINT = 'rgba(241,238,233,.16)', BG = '#07090b';
  const tm = { a: 1, follow: true, sigma: 0.01, play: 0, w: {} };
  const widthOf = (id, lo, hi) => clamp(Math.round($(id).clientWidth || 640), lo, hi);
  const svg = (w, h, body) => '<svg viewBox="0 0 ' + w + ' ' + h + '" width="' + w + '" height="' + h + '" aria-hidden="true" focusable="false">' + body + '</svg>';
  const tx = (s, x, y, o = {}) => '<text x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" font-size="' + (o.px || 12) + '" fill="' + (o.fill || INKC) + '" text-anchor="' + (o.anchor || 'start') + '"' + (o.cp ? ' class="cp"' : '') + (o.halo ? ' paint-order="stroke" stroke="' + BG + '" stroke-width="4" stroke-linejoin="round"' : '') + (o.it ? ' font-style="italic"' : '') + '>' + s + '</text>';
  const line = (x1, y1, x2, y2, col, w = 1, dash) => '<line x1="' + x1.toFixed(1) + '" y1="' + y1.toFixed(1) + '" x2="' + x2.toFixed(1) + '" y2="' + y2.toFixed(1) + '" stroke="' + col + '" stroke-width="' + w + '"' + (dash ? ' stroke-dasharray="' + dash + '"' : '') + '/>';
  const dOf = pts => pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('');
  const pathEl = (pts, col, w = 1, o = {}) => '<path' + (o.id ? ' id="' + o.id + '"' : '') + ' d="' + (typeof pts === 'string' ? pts : dOf(pts)) + '" fill="' + (o.fill || 'none') + '" stroke="' + col + '" stroke-width="' + w + '" stroke-linejoin="round" stroke-linecap="round"' + (o.dash ? ' stroke-dasharray="' + o.dash + '"' : '') + (o.op ? ' opacity="' + o.op + '"' : '') + '/>';
  const sup10 = (x, d = 1) => { const [m, e] = x.toExponential(d).split('e'); return m + ' × 10' + EV.sup(+e); };
  /* a time in words; short form for narrow figures */
  const tWords = (s, short) => { if (!short) return EV.fmtTime(s); const y = s / K.yr; if (y >= 1e9) return EV.sig(y / 1e9, 3) + ' bn yr'; if (y >= 1e6) return EV.sig(y / 1e6, 3) + ' m yr'; if (y >= 1e4) return EV.group(+y.toPrecision(2)) + ' yr'; if (y >= 1) return EV.sig(y, 2) + ' yr'; return EV.fmtTime(s); };
  const pct = x => x >= 0.9995 ? fmt(x * 100, 2) + '%' : x >= 0.001 ? fmt(x * 100, x < 0.1 ? 2 : 1) + '%' : x < 1e-12 ? '0%' : x >= 1e-6 ? (x * 100).toFixed(x >= 1e-5 ? 3 : 4) + '%' : sup10(x * 100, 0) + '%';
  const sigTxt = s => s >= 0.001 ? fmt(s, s < 0.1 ? 3 : 2) : sup10(s, 1);
  /* spread labels down a column so that none overlaps: each keeps its own y if it can */
  const spread = (L, gap, lo, hi) => { L.sort((a, b) => a.y - b.y); let prev = lo - gap; for (const l of L) { l.ly = Math.max(l.y, prev + gap); prev = l.ly; } let next = hi + gap; for (let i = L.length - 1; i >= 0; i--) { L[i].ly = Math.min(L[i].ly, next - gap); next = L[i].ly; } return L; };

  /* ── 1. the loop: time after the beginning for every stop on both branches ── */
  let loopPos = {};
  function drawLoop() { const W = widthOf('tLoop', 320, 760), nar = W < 560, H = nar ? 640 : 660, top = 104, bot = H - 66, xl = nar ? 24 : 58, xr = W - xl, lt1 = Math.log10(CO.age), lt0 = -45.2, px = nar ? 10.5 : 12.5;
    const y = t => t <= 0 ? bot : top + (lt1 - Math.log10(t)) / (lt1 - lt0) * (bot - top); let s = ''; loopPos = {};
    /* guide lines and the scale of time, in the middle */
    const ticks = [[1e-40, '10⁻⁴⁰ s'], [1e-30, '10⁻³⁰ s'], [1e-20, '10⁻²⁰ s'], [1e-10, '10⁻¹⁰ s'], [1, '1 second'], [K.yr, '1 year'], [1e6 * K.yr, '1 million years']];
    /* the two branches: whole above the hard line, broken below it */
    const yIn = y(T.seenIn), yOut = y(T.seenOut);
    s += pathEl('M' + xl + ' ' + top + 'C' + xl + ' ' + (top - 62) + ' ' + xr + ' ' + (top - 62) + ' ' + xr + ' ' + top, GOLD, 2);
    s += line(xl, top, xl, yIn, GOLD, 2) + line(xl, yIn, xl, bot, RUST, 2, '2 6') + line(xr, top, xr, yOut, GOLD, 2) + line(xr, yOut, xr, bot, RUST, 2, '2 6');
    s += pathEl('M' + xl + ' ' + bot + 'C' + xl + ' ' + (bot + 50) + ' ' + xr + ' ' + (bot + 50) + ' ' + xr + ' ' + bot, RUST, 2, { dash: '2 6' });
    s += line(xl - 8, yIn, xl + 8, yIn, RUST, 2) + line(xr - 8, yOut, xr + 8, yOut, RUST, 2);
    /* the ends */
    const apex = top - 46.5; s += '<circle cx="' + W / 2 + '" cy="' + apex + '" r="5" fill="' + GOLD + '"/>' + tx('Now · you', W / 2, apex - (nar ? 16 : 30), { px: nar ? 14 : 15, cp: 1, fill: GOLD, anchor: 'middle' }) + (nar ? '' : tx(fmt(CO.age / Gyr, 1) + ' billion years after the beginning', W / 2, apex - 13, { px, fill: DIMC, anchor: 'middle' }));
    s += tx('The beginning', W / 2, bot + 30, { px: nar ? 14 : 15, cp: 1, fill: RUST, anchor: 'middle' }) + tx('both ends of the zoom', W / 2, bot + 47, { px, fill: DIMC, anchor: 'middle' });
    s += tx('Diving in', xl - 10, top - 70, { px: 13, cp: 1, fill: DIMC }) + tx('Flying out', xr + 10, top - 70, { px: 13, cp: 1, fill: DIMC, anchor: 'end' });
    /* stops and their labels */
    const LL = [{ y: top, s: EV.fmtLen(T.lambdaNow, 2) + ' · today’s sky', col: GOLD }, { y: yIn, s: (nar ? 'untested before ' : 'nothing tested earlier than ') + sup10(T.seenIn) + ' s', col: RUST }], RR = [{ y: yOut, s: (nar ? 'unseen before ' : 'nothing seen earlier than ') + tWords(T.seenOut, nar), col: RUST }]; let near = null, dots = '';
    for (const S of T.STOPS) { if (S.branch === 'here') { loopPos[S.id] = [W / 2, apex]; continue; }
      if (S.branch === 'in') { if (S.future) { const u = 0.26, c = 1 - u, bx = c * c * c * xl + 3 * c * c * u * xl + 3 * c * u * u * xr + u * u * u * xr, by = c * c * c * top + 3 * c * c * u * (top - 62) + 3 * c * u * u * (top - 62) + u * u * u * top; loopPos[S.id] = [bx, by]; dots += '<circle cx="' + bx.toFixed(1) + '" cy="' + by.toFixed(1) + '" r="3.2" fill="' + BG + '" stroke="' + GOLD + '" stroke-width="1.4"/>'; continue; }
        const yy = y(S.t); loopPos[S.id] = [xl, yy]; dots += '<circle cx="' + xl + '" cy="' + yy.toFixed(1) + '" r="3.4" fill="' + (S.kind === 'imagined' ? RUST : GOLD) + '"/>'; LL.push({ y: yy, s: S.short + ' · ' + tWords(S.t, nar), col: S.kind === 'imagined' ? RUST : INKC }); continue; }
      const yy = S.beyond ? bot : y(S.t); loopPos[S.id] = [xr, yy]; dots += '<circle cx="' + xr + '" cy="' + yy.toFixed(1) + '" r="3.4" fill="' + (S.beyond ? RUST : GOLD) + '"/>';
      if (S.beyond) RR.push({ y: yy, s: S.short + ' · no light yet', col: RUST }); else if (CO.age - S.t < 0.05 * CO.age) { if (!near || S.t < near.t) near = S; } else RR.push({ y: yy, s: S.short + ' · ' + tWords(S.t, nar), col: INKC }); }
    if (near) RR.push({ y: top, s: nar ? 'City to ' + near.short + ' · now' : 'City to ' + near.short + ' · within the last ' + tWords(CO.age - near.t, nar), col: INKC });
    for (const l of spread(LL, px + 4.5, top + 4, bot - 4)) s += line(xl + 5, l.y, xl + 13, l.ly, 'rgba(241,238,233,.3)') + tx(l.s, xl + 16, l.ly + 4, { px, fill: l.col });
    let lowR = top; for (const l of spread(RR, px + 4.5, top + 4, bot - 4)) { if (l.y < bot - 30) lowR = Math.max(lowR, l.ly); }
    for (const k of ticks) { const yy = y(k[0]); if (nar || yy < lowR + 16) s += line(xl + 6, yy, xr - 6, yy, 'rgba(241,238,233,.07)'); else s += line(xl + 6, yy, W / 2 - 56, yy, 'rgba(241,238,233,.08)') + line(W / 2 + 56, yy, xr - 6, yy, 'rgba(241,238,233,.08)') + tx(k[1], W / 2, yy + 4, { px: 11.5, fill: 'rgba(241,238,233,.42)', anchor: 'middle' }); }
    for (const l of RR) s += line(xr - 5, l.y, xr - 13, l.ly, 'rgba(241,238,233,.3)') + tx(l.s, xr - 16, l.ly + 4, { px, fill: l.col, anchor: 'end' });
    s += dots + '<circle id="tLoopRing" r="8.5" fill="none" stroke="' + INKC + '" stroke-width="1.6" cx="-20" cy="-20"/>';
    $('tLoop').innerHTML = svg(W, H, s); ringLoop(); }
  function ringLoop() { const r = $('tLoopRing'), p = loopPos[st.level]; if (r && p) { r.setAttribute('cx', p[0].toFixed(1)); r.setAttribute('cy', p[1].toFixed(1)); } }

  /* ── 2. the octant ── */
  let OC = null;
  const unit = p => [Math.sqrt(p[0]), Math.sqrt(p[1]), Math.sqrt(p[2])];
  function drawOct() { const W = widthOf('tOct', 320, 760), wide = W >= 640, nar = !wide, mxL = wide ? 40 : 14, mxR = wide ? 190 : 14, topPad = 44, botPad = 56, sc = Math.min((W - mxL - mxR) / 1.4142, 330), H = Math.round(topPad + botPad + 1.3938 * sc), cx = mxL + (W - mxL - mxR) / 2, cy = topPad + 0.8165 * sc, px = W < 560 ? 11.5 : 12.5;
    const pr = v => [cx + sc * (v[2] - v[0]) * Math.SQRT1_2, cy - sc * (2 * v[1] - v[0] - v[2]) / Math.sqrt(6)];
    const edge = (i, j, n = 48) => { const o = []; for (let k = 0; k <= n; k++) { const t = k / n * PI / 2, v = [0, 0, 0]; v[i] = Math.cos(t); v[j] = Math.sin(t); o.push(pr(v)); } return o; };
    const eRM = edge(0, 1), eML = edge(1, 2), eLR = edge(2, 0), R = pr([1, 0, 0]), Mm = pr([0, 1, 0]), Lm = pr([0, 0, 1]);
    let s = '<defs><radialGradient id="tOctG" cx="50%" cy="46%" r="62%"><stop offset="0" stop-color="rgba(231,199,90,.13)"/><stop offset="1" stop-color="rgba(231,199,90,.02)"/></radialGradient><clipPath id="tOctClip"><circle cx="0" cy="0" r="58"/></clipPath></defs>';
    s += pathEl(dOf(eRM.concat(eML.slice(1), eLR.slice(1))) + 'Z', 'none', 0, { fill: 'url(#tOctG)' });
    /* lines of equal share: a quarter, a half, three-quarters of each ingredient */
    for (let i = 0; i < 3; i++) for (const c of [0.25, 0.5, 0.75]) { const o = [], j = (i + 1) % 3, k = (i + 2) % 3; for (let n = 0; n <= 40; n++) { const t = n / 40 * PI / 2, v = [0, 0, 0]; v[i] = Math.sqrt(c); v[j] = Math.sqrt(1 - c) * Math.cos(t); v[k] = Math.sqrt(1 - c) * Math.sin(t); o.push(pr(v)); } s += pathEl(o, 'rgba(231,199,90,' + (c === 0.5 ? 0.24 : 0.12) + ')', 1); }
    s += pathEl(eRM, 'rgba(241,238,233,.34)', 1) + pathEl(eML, 'rgba(241,238,233,.34)', 1) + pathEl(eLR, GREEN, 1.2, { dash: '5 5', op: 0.55 });
    /* the path, sampled in ln a */
    const us = [], pts = []; for (let u = -19; u <= 9.0001; u += 0.035) { us.push(u); pts.push(pr(unit(T.shares(Math.exp(u))))); }
    s += pathEl([R].concat(pts, [Lm]), GOLD, 1.6, { dash: '1 5', op: 0.75 }) + pathEl('M0 0', GREEN, 1.8, { id: 'tOctChord' }) + pathEl('M0 0', GOLD, 3.2, { id: 'tOctWalk' });
    /* corners */
    const by = cy + 0.5774 * sc + 24;
    s += tx('Radiation', R[0] - 4, by, { px: 14, cp: 1, fill: GOLD }) + tx('the beginning', R[0] - 4, by + 16, { px, fill: DIMC, it: 1 });
    s += tx('Dark energy', Lm[0] + 4, by, { px: 14, cp: 1, fill: GOLD, anchor: 'end' }) + tx('the far future', Lm[0] + 4, by + 16, { px, fill: DIMC, anchor: 'end', it: 1 });
    s += tx('Matter', Mm[0], Mm[1] - 16, { px: 14, cp: 1, fill: GOLD, anchor: 'middle' });
    s += tx(W < 560 ? 'the direct way: one half-turn' : 'the direct way, never taken: one half-turn', cx, cy + 0.5774 * sc - 12, { px, fill: GREEN, anchor: 'middle', it: 1 });
    for (const c of [R, Mm, Lm]) s += '<circle cx="' + c[0].toFixed(1) + '" cy="' + c[1].toFixed(1) + '" r="3.5" fill="' + BG + '" stroke="' + GOLD + '" stroke-width="1.5"/>';
    /* the labelled moments */
    const EVN = { eq: nar ? 'matter = radiation' : 'matter overtakes radiation', peak: 'matter’s largest share, ' + fmt(T.peak.share * 100, 2) + '%', accel: 'slowing turns to speeding', lambda: 'dark energy = matter', now: 'now' };
    for (const e of T.EVENTS) { const q = pr(unit(e.p)), zt = e.key === 'now' ? '' : ' · z = ' + (e.z > 100 ? EV.group(+e.z.toPrecision(2)) : fmt(e.z, e.z > 5 ? 1 : 2));
      s += '<circle cx="' + q[0].toFixed(1) + '" cy="' + q[1].toFixed(1) + '" r="3" fill="' + BG + '" stroke="' + INKC + '" stroke-width="1.3"/>';
      const zl = zt.slice(3);
      if (e.key === 'eq') s += wide ? tx(EVN.eq, q[0] + 12, q[1] + 2, { px, halo: 1 }) + tx(zl, q[0] + 12, q[1] + 2 + px + 4, { px, halo: 1, fill: DIMC }) : line(q[0] + 3, q[1] + 4, q[0] + 9, q[1] + 20, 'rgba(241,238,233,.3)') + tx(EVN.eq, q[0] + 8, q[1] + 34, { px, halo: 1 }) + tx(zl, q[0] + 8, q[1] + 34 + px + 4, { px, halo: 1, fill: DIMC });
      else if (e.key === 'peak') { const o = wide ? 44 : 27; s += line(q[0], q[1] + 5, q[0], q[1] + o - 13, 'rgba(241,238,233,.3)') + tx(EVN.peak, Mm[0], q[1] + o, { px, anchor: 'middle', halo: 1 }) + tx('z = ' + fmt(e.z, 1) + ', then it turned', Mm[0], q[1] + o + px + 3, { px, anchor: 'middle', halo: 1, fill: DIMC }); }
      else if (wide) s += tx(EVN[e.key], q[0] + 16, q[1] + 2, { px, fill: e.key === 'now' ? GOLD : INKC }) + (zl ? tx(zl, q[0] + 16, q[1] + 2 + px + 4, { px, fill: DIMC }) : '');
      else s += tx(EVN[e.key], q[0] - 12, q[1] + (zl ? -2 : 4), { px, anchor: 'end', halo: 1, fill: e.key === 'now' ? GOLD : INKC }) + (zl ? tx(zl, q[0] - 12, q[1] - 2 + px + 4, { px, anchor: 'end', halo: 1, fill: DIMC }) : ''); }
    /* the turn at the matter corner, enlarged (only where there is room for it) */
    if (wide) { const ix = mxL + 28, iy = topPad + 30, k = 3.2, g = pts.filter(p => Math.hypot(p[0] - Mm[0], p[1] - Mm[1]) < 40), pk = pr(unit(T.peak.p));
      s += line(ix + 52, iy - 14, Mm[0] - 8, Mm[1] + 2, 'rgba(241,238,233,.22)', 1, '2 4') + '<circle cx="' + ix + '" cy="' + iy + '" r="58" fill="' + BG + '" stroke="rgba(241,238,233,.34)"/>';
      s += '<g transform="translate(' + ix + ' ' + iy + ')" clip-path="url(#tOctClip)"><g transform="scale(' + k + ') translate(' + (-Mm[0]).toFixed(2) + ' ' + (-(Mm[1] + 11)).toFixed(2) + ')">' + pathEl(eRM, 'rgba(241,238,233,.34)', 1 / k) + pathEl(eML, 'rgba(241,238,233,.34)', 1 / k) + pathEl(g, GOLD, 2.2 / k) + '<circle cx="' + Mm[0].toFixed(2) + '" cy="' + Mm[1].toFixed(2) + '" r="' + (3.5 / k).toFixed(2) + '" fill="' + BG + '" stroke="' + GOLD + '" stroke-width="' + (1.5 / k).toFixed(2) + '"/><circle cx="' + pk[0].toFixed(2) + '" cy="' + pk[1].toFixed(2) + '" r="' + (3 / k).toFixed(2) + '" fill="' + BG + '" stroke="' + INKC + '" stroke-width="' + (1.3 / k).toFixed(2) + '"/></g></g>';
      s += tx('the turn, enlarged', ix, iy + 76, { px: 12, fill: DIMC, anchor: 'middle', it: 1 }); }
    s += '<g id="tOctMark"><circle r="11" fill="none" stroke="' + GOLD + '" stroke-opacity=".5" stroke-width="1.2"/><circle r="5.5" fill="' + GOLD + '" stroke="' + BG + '" stroke-width="1.5"/></g>';
    $('tOct').innerHTML = svg(W, H, s); OC = { pr, us, pts, R, Lm }; updOct(); }
  function updOct() { if (!OC) return; const a = tm.a, u = Math.log(a), P = T.pathAt(a), v = unit(P.p), q = OC.pr(v);
    const walked = [OC.R]; for (let i = 0; i < OC.us.length && OC.us[i] < u; i++) walked.push(OC.pts[i]); walked.push(q);
    $('tOctWalk').setAttribute('d', dOf(walked));
    const w = P.Cstar / 2, ch = []; if (w > 1e-6) for (let i = 0; i <= 36; i++) { const t = i / 36, A = Math.sin((1 - t) * w) / Math.sin(w), B = Math.sin(t * w) / Math.sin(w); ch.push(OC.pr([A + B * v[0], B * v[1], B * v[2]])); }
    $('tOctChord').setAttribute('d', ch.length ? dOf(ch) : 'M0 0'); $('tOctMark').setAttribute('transform', 'translate(' + q[0].toFixed(1) + ' ' + q[1].toFixed(1) + ')');
    /* the panel */
    const z = 1 / a - 1, near = Math.abs(u) < 0.004, G = T.GROWTH;
    $('tAOut').textContent = near ? 'as now' : a < 1 ? '1/' + (1 / a >= 100 ? EV.group(+(1 / a).toPrecision(3)) : EV.sig(1 / a, 3)) + ' of today' : EV.sig(a, 3) + ' times today';
    $('tAge').textContent = (near ? 'Now, ' : (a < 1 ? 'Redshift z = ' + (z >= 100 ? EV.group(+z.toPrecision(3)) : EV.sig(z, 3)) + ', ' : 'In the future, ')) + EV.fmtTime(CO.tOfA(a)) + ' after the beginning.';
    $('tSh').textContent = 'Radiation ' + pct(P.p[0]) + ' · matter ' + pct(P.p[1]) + ' · dark energy ' + pct(P.p[2]);
    $('tShares').innerHTML = '<i style="width:' + (P.p[0] * 100).toFixed(2) + '%;background:var(--leaf)"></i><i style="width:' + (P.p[1] * 100).toFixed(2) + '%;background:var(--ink)"></i><i style="width:' + (P.p[2] * 100).toFixed(2) + '%;background:var(--viridian)"></i>';
    $('tC').textContent = fmt(P.Ch, 3) + ' of ' + fmt(T.whole.Ch, 3); $('tCs').textContent = fmt(P.CsOmega, 3) + ' of 1'; $('tS').textContent = fmt(P.Sh, 3); $('tWalk').textContent = fmt(P.walked * 100, 1) + '% walked';
    $('tGrow').textContent = 'f = ' + fmt(G.f(a), 3); $('tLump').textContent = (G.D(a) / G.D(1) >= 0.01 ? fmt(G.D(a) / G.D(1) * 100, 1) + '%' : sup10(G.D(a) / G.D(1) * 100, 1) + '%') + ' of today’s';
    $('tBars').innerHTML = [[P.Ch, 'var(--gold)'], [P.CsOmega, 'var(--viridian)'], [P.Sh, 'var(--rust)']].map(b => '<div class="bar"><i style="width:' + (clamp(b[0] / 2, 0, 1) * 100).toFixed(2) + '%;background:' + b[1] + '"></i></div>').join('');
    $('tOct').setAttribute('aria-label', 'One-eighth of a sphere with corners radiation, matter and dark energy. The marked moment: radiation ' + pct(P.p[0]) + ', matter ' + pct(P.p[1]) + ', dark energy ' + pct(P.p[2]) + '. Path walked ' + fmt(P.Ch, 3) + ' half-turns, distance from the start ' + fmt(P.CsOmega, 3) + ', surplus ' + fmt(P.Sh, 3) + '.'); }
  const UMIN = -14, UMAX = 5;
  function setA(a, fromSlider) { tm.a = clamp(a, Math.exp(UMIN), Math.exp(UMAX)); if (!fromSlider) $('tA').value = String(Math.round(100 * Math.log(tm.a))); updOct(); }
  function setFollow(on) { tm.follow = on; $('tFollow').setAttribute('aria-pressed', String(on)); if (on) sync(); }
  function stopPlay() { if (tm.play) cancelAnimationFrame(tm.play); tm.play = 0; $('tPlay').textContent = reduce ? 'Next moment' : 'Play the history'; }
  /* the redshift at the edge of the frame above, for the stops that have one */
  function frameA() { const fov = fovNow(); if (st.level >= EV.HOME) return 1; const half = fov / 2; if (half >= CO.horizon) return Math.exp(UMIN); const z = CO.zOfChi(half); return z < 0.001 ? 1 : 1 / (1 + z); }
  function sync() { ringLoop(); if (tm.follow && !tm.play) { const a = frameA(); if (Math.abs(Math.log(a / tm.a)) > 1e-9) setA(a); } }

  /* ── 3a. four clocks ── */
  function drawClocks() { const C = T.CONFORMAL, L = T.LEG, row = (name, sub, track, val) => '<div class="clock" role="listitem"><div><b>' + name + '</b><small>' + sub + '</small></div><div>' + track + '<div class="val">' + val + '</div></div></div>';
    $('tClocks').innerHTML = row('Years', 'ordinary cosmic time', '<div class="track endR"><i style="width:22%"></i><u style="left:22%"></u></div>', fmt(CO.age / Gyr, 2) + ' billion years gone, and no end ahead: no fraction can be given.') +
      row('Light’s reach', 'conformal time: the distance light can cover', '<div class="track"><i style="width:' + (C.frac * 100).toFixed(2) + '%"></i><u style="left:' + (C.frac * 100).toFixed(2) + '%"></u></div>', fmt(C.past / K.Gly, 2) + ' billion light-years covered, ' + fmt(C.future / K.Gly, 2) + ' still to come: ' + fmt(C.frac * 100, 1) + '% gone. The future that light can ever reach is finite.') +
      row('Factors of e in size', 'e-folds of expansion', '<div class="track endLR"><i style="width:50%"></i><u style="left:50%"></u></div>', 'Endless both ways: there was no first one and there will be no last.') +
      row('Half-turns of the path', 'the contents of the universe, as above', '<div class="track"><i style="width:' + (T.now.walked * 100).toFixed(2) + '%"></i><u style="left:' + (T.now.walked * 100).toFixed(2) + '%"></u></div>', fmt(T.now.Ch, 3) + ' of ' + fmt(T.whole.Ch, 3) + ' half-turns: ' + fmt(T.now.walked * 100, 1) + '% walked. On the last leg alone, ' + fmt(L.now.frac * 100, 1) + '% of the way from matter to dark energy.'); }

  /* ── 3b. the resolution slider ── */
  function drawRes() { const W = widthOf('tRes', 320, 760), nar = W < 560, H = nar ? 300 : 330, l = 44, r = 16, t = 34, b = 44, pw = W - l - r, ph = H - t - b, L = T.LEG, TMAX = 200, sg = tm.sigma, px = nar ? 11.5 : 12.5;
    const X = g => l + g / TMAX * pw, Y = v => t + (1 - v) * ph; let s = '';
    for (const g of [0, 50, 100, 150, 200]) s += line(X(g), t, X(g), t + ph, 'rgba(241,238,233,.07)') + tx(g === 200 ? (nar ? '200 bn yr' : '200 billion years') : String(g), X(g), t + ph + 18, { px, fill: DIMC, anchor: g === 200 ? 'end' : g === 0 ? 'start' : 'middle' });
    for (const v of [[0, '0'], [0.5, '½'], [1, '1']]) s += line(l, Y(v[0]), l + pw, Y(v[0]), v[0] === 1 ? 'rgba(241,238,233,.5)' : 'rgba(241,238,233,.07)', 1, v[0] === 1 ? '4 4' : null) + tx(v[1], l - 8, Y(v[0]) + 4, { px, fill: DIMC, anchor: 'end' });
    s += tx('time after the beginning', l + pw / 2, H - 6, { px, fill: DIMC, anchor: 'middle', it: 1 }) + tx(nar ? 'C*Ω: all dark energy at 1' : 'C*Ω, the share of the half-turn covered · all dark energy at 1', l, 18, { px, fill: DIMC, it: 1 });
    /* what this observer cannot tell from the end */
    const yb = Y(1 - sg / PI); s += '<rect x="' + l + '" y="' + t + '" width="' + pw + '" height="' + Math.max(0, yb - t).toFixed(1) + '" fill="' + RUST + '" opacity=".26"/>';
    const pts = []; for (let i = 0; i <= 240; i++) { const g = i / 240 * TMAX; pts.push([X(g), Y(L.phiOfT(g * Gyr) / PI)]); } s += pathEl(pts, GOLD, 2.4);
    const tc = L.cutTime(sg) / Gyr; if (tc <= TMAX) s += line(X(tc), t - 4, X(tc), t + ph, GREEN, 1.6) + tx('the cut · ' + fmt(tc, tc < 100 ? 1 : 0) + (nar ? ' bn yr' : ' billion years'), X(tc) + (tc > 120 ? -8 : 8), t + ph * 0.42, { px, fill: GREEN, anchor: tc > 120 ? 'end' : 'start', halo: 1 });
    else s += tx('the cut is off the chart, at ' + fmt(tc, 0) + ' billion years →', l + pw - 6, t + ph * 0.42, { px, fill: GREEN, anchor: 'end', halo: 1 });
    const mk = [[L.accel.t / Gyr, L.accel.frac, nar ? 'speeding' : 'slowing turns to speeding'], [L.equal.t / Gyr, 0.5, 'equal'], [L.now.t / Gyr, L.now.frac, 'now']];
    mk.forEach((m, i) => { s += '<circle cx="' + X(m[0]).toFixed(1) + '" cy="' + Y(m[1]).toFixed(1) + '" r="3.2" fill="' + BG + '" stroke="' + INKC + '" stroke-width="1.3"/>' + tx(m[2], X(m[0]) + 10, Y(m[1]) + 12 - i * 2, { px, fill: i === 2 ? GOLD : INKC, halo: 1 }); });
    $('tRes').innerHTML = svg(W, H, s); }
  function updRes() { const sg = tm.sigma, L = T.LEG, tc = L.cutTime(sg);
    $('tSigOut').textContent = 'σ = ' + sigTxt(sg) + ' rad'; $('tSig').setAttribute('aria-valuetext', 'sigma ' + sigTxt(sg) + ' radians, 1 part in ' + EV.group(+(PI / sg).toPrecision(3)) + ' of a half-turn');
    $('tCut').textContent = fmt(tc / Gyr, 1) + ' billion years'; $('tCutFrom').textContent = fmt((tc - L.now.t) / Gyr, 1) + ' billion years'; $('tCutApprox').textContent = fmt(L.cutApprox(sg) / Gyr, 1) + ' billion years';
    $('tRec').innerHTML = T.RESIDUALS.map(R => '<dt>' + sup10(R.x, 1) + ' left ionised: needs σ ≥ ' + fmt(R.sigma, 3) + '</dt><dd>' + (sg >= R.sigma ? 'yes' : 'no') + '</dd>').join('');
    const c = T.threeWayCut(sg); $('tThree').textContent = 'The whole three-way path began at pure radiation, so its opposite is any state with no radiation. For this observer that cut ' + (c.past ? 'has already come: at z = ' + (c.z > 100 ? EV.group(+c.z.toPrecision(2)) : fmt(c.z, 2)) + ', ' + EV.fmtTime(c.t) + ' after the beginning.' : 'comes when the universe is ' + EV.sig(c.a, 3) + ' times its present size, ' + EV.fmtTime(c.t) + ' after the beginning.');
    drawRes(); }
  function setSigma(sg, fromSlider) { tm.sigma = clamp(sg, 1e-6, 0.5); if (!fromSlider) $('tSig').value = String(Math.round(100 * Math.log10(tm.sigma))); updRes(); }

  /* ── 4. hydrogen, there and back ── */
  function drawHyd() { const W = widthOf('tHyd', 320, 1180), nar = W < 560, H = nar ? 380 : 400, l = 44, r = nar ? 16 : 150, pw = W - l - r, t1 = 30, h1 = nar ? 96 : 104, t2 = t1 + h1 + 44, h2 = H - t2 - 44, Hy = T.HYDROGEN, lz = Math.log10(3001), px = nar ? 11.5 : 12.5;
    const X = z => l + (1 - Math.log10(1 + z) / lz) * pw, Y1 = v => t1 + (1 - v) * h1, Y2 = v => t2 + (1 - v / 2) * h2; let s = '';
    const marks = [[EV.RECOMB.zHalf, 'the fog clears'], [T.REION.z, nar ? 'first stars' : 'the first stars shine']];
    for (const m of marks) s += line(X(m[0]), t1 - 6, X(m[0]), t2 + h2, 'rgba(241,238,233,.2)', 1, '2 4') + tx(m[1], X(m[0]), t1 - 12, { px, fill: DIMC, anchor: 'middle', it: 1 });
    for (const z of nar ? [1000, 100, 10, 0] : [1000, 100, 10, 1, 0]) s += line(X(z), t2 + h2, X(z), t2 + h2 + 5, DIMC) + tx(z === 0 ? 'now' : 'z = ' + EV.group(z), X(z), t2 + h2 + 20, { px, fill: DIMC, anchor: z === 0 ? 'end' : 'middle' });
    s += tx('redshift; time runs to the right', l + pw / 2, H - 6, { px, fill: DIMC, anchor: 'middle', it: 1 });
    for (const v of [0, 1]) s += line(l, Y1(v), l + pw, Y1(v), 'rgba(241,238,233,.09)') + tx(v ? '1' : '0', l - 8, Y1(v) + 4, { px, fill: DIMC, anchor: 'end' });
    for (const v of [0, 1, 2]) s += line(l, Y2(v), l + pw, Y2(v), v === 1 ? 'rgba(241,238,233,.45)' : 'rgba(241,238,233,.09)', 1, v === 1 ? '4 4' : null) + tx(String(v), l - 8, Y2(v) + 4, { px, fill: DIMC, anchor: 'end' });
    const cx = [], cc = [], cs = [], cS = []; for (let i = 0; i <= Hy.n; i += 4) { const xx = X(Hy.Z[i]); cx.push([xx, Y1(Hy.X[i])]); cc.push([xx, Y2(Hy.C[i])]); cs.push([xx, Y2(Hy.Cs[i])]); cS.push([xx, Y2(Hy.C[i] - Hy.Cs[i])]); }
    s += pathEl(cx, BLUE, 2.2) + pathEl(cS, RED, 2, { dash: '6 4' }) + pathEl(cc, GOLD, 2.4) + pathEl(cs, GREEN, 2.2);
    s += tx('share of hydrogen ionised', X(90), Y1(0.5) + 4, { px, fill: BLUE, anchor: 'middle', halo: 1 }) + tx(nar ? 'the cut' : 'the cut, one half-turn', l + 6, Y2(1) - 6, { px, fill: DIMC, it: 1, halo: 1 });
    if (nar) s += tx('C ' + fmt(Hy.Ch, 2), l + pw - 4, Y2(Hy.Ch) - 8, { px, fill: GOLD, anchor: 'end', halo: 1 }) + tx('S ' + fmt(Hy.Sh, 2), l + pw - 4, Y2(Hy.Ch) + 16, { px, fill: RED, anchor: 'end', halo: 1 }) + tx('C* back to 0', l + pw - 4, Y2(0) - 16, { px, fill: GREEN, anchor: 'end', halo: 1 }) + tx('C* ' + fmt(Hy.maxCs, 3), X(60), Y2(1) + 16, { px, fill: GREEN, anchor: 'middle', halo: 1 });
    else s += tx('path walked, C = ' + fmt(Hy.Ch, 2), l + pw + 10, Y2(Hy.Ch) - 2, { px, fill: GOLD }) + tx('surplus, S = ' + fmt(Hy.Sh, 2), l + pw + 10, Y2(Hy.Ch) + 15, { px, fill: RED }) + tx('distance from the', l + pw + 10, Y2(0) - 14, { px, fill: GREEN }) + tx('start, C*: back to 0', l + pw + 10, Y2(0) + 2, { px, fill: GREEN }) + tx('C* stops at ' + fmt(Hy.maxCs, 3), X(60), Y2(1) + 17, { px, fill: GREEN, anchor: 'middle', halo: 1 }) + tx('ionised again', l + pw + 10, Y1(1) + 4, { px, fill: BLUE });
    $('tHyd').innerHTML = svg(W, H, s); }

  /* ── the words that are numbers ── */
  function fill() { const L = T.LEG, G = T.GROWTH, Hy = T.HYDROGEN, D = T.DESI;
    $('tEnergy').innerHTML = T.ENERGIES.slice(0, 5).map(e => '<dt>' + e.name.charAt(0).toUpperCase() + e.name.slice(1) + '</dt><dd>' + (e.E === EV.planck.E ? fmt(T.planckRatio, 3) + ' Planck times (no meaning)' : EV.fmtTime(e.t)) + '</dd>').join('') + '<dt>Earliest light seen</dt><dd>' + EV.fmtTime(T.seenOut) + '</dd><dt>Now meets now at</dt><dd>' + EV.fmtLen(T.lambdaNow, 3) + '</dd><dt>√(Planck length × horizon)</dt><dd>' + EV.fmtLen(T.geoMean, 3) + '</dd>';
    $('tStopBody').innerHTML = T.STOPS.map(S => { const how = S.branch === 'here' ? 'here and now' : S.branch === 'out' ? (S.beyond ? 'beyond the horizon: no light has arrived' : 'when the light now arriving from the frame’s edge set out' + (S.z > 0.001 ? ' (z = ' + (S.z > 100 ? EV.group(+S.z.toPrecision(3)) : EV.sig(S.z, 3)) + ')' : '')) : S.future ? 'still to come: the sky is not yet this cold (it will be when the universe is ' + EV.sig(1 / (1 + S.z), 3) + ' times its present size)' : S.how === 'radiation' ? 'kT = hc/width = ' + e10(S.E / K.GeV, 2) + ' GeV; radiation formula with g* = ' + S.g : 'kT = hc/width: the sky at ' + EV.sig(K.Tcmb * (1 + S.z), 3) + ' K; the Friedmann integral';
      return '<tr data-kind="' + S.kind + '"><th scope="row">' + S.name + '</th><td class="n">' + EV.fmtLen(S.fov, 2) + '</td><td>' + (S.branch === 'out' ? 'flying out' : S.branch === 'in' ? 'diving in' : 'here') + '</td><td class="n">' + (S.beyond ? 'the beginning' : S.t < 1e-9 ? e10(S.t, 1) + ' s' : EV.fmtTime(S.t)) + '</td><td>' + how + '</td></tr>'; }).join('');
    setLive('tvE', 'coefficient from the constants: ' + fmt(T.coeff, 4) + ' s &nbsp; ' + T.ENERGIES.slice(1, 6).map(e => e.name.replace(/,.*/, '') + ': g* = ' + fmt(e.g, 2) + ' → ' + (e.t < 1e-3 ? e10(e.t, 2) + ' s' : EV.fmtTime(e.t))).join(' &nbsp; ') + ' &nbsp; at 10 keV the two methods differ by ' + e10(T.crossCheck.rel, 1) + ' &nbsp; Planck energy: ' + fmt(T.planckRatio, 4) + ' t<sub>P</sub>');
    setLive('tvP', 'whole path: C = ' + fmt(T.whole.Ch, 4) + ' half-turns, C*Ω → 1, S = ' + fmt(T.whole.Sh, 4) + ' &nbsp; now: C = ' + fmt(T.now.Ch, 4) + ', C*Ω = ' + fmt(T.now.CsOmega, 4) + ', S = ' + fmt(T.now.Sh, 4) + ', ' + fmt(T.now.walked * 100, 2) + '% walked &nbsp; matter’s largest share ' + fmt(T.peak.share * 100, 3) + '% at z = ' + fmt(T.peak.z, 2) + ' (a⁴ = Ω<sub>r</sub>/3Ω<sub>Λ</sub>) &nbsp; made-up universe: ' + fmt(T.GENERIC.stiff.Ch, 3) + ' &nbsp; middle stage never dominant: ' + fmt(T.GENERIC.thirds.Ch, 3));
    setLive('tvL', 'now: L = ' + fmt(T.speed(1), 4) + ' radians per e-fold, q = ' + fmt(T.q(1), 4) + ' &nbsp; at matter’s peak: L = ' + fmt(T.speed(T.peak.a), 4) + ', q = ' + fmt(T.q(T.peak.a), 4) + ' &nbsp; at matter = radiation: L = ' + fmt(T.speed(1 / (1 + CO.zEq)), 4));
    setLive('tvG', 'today f = ' + fmt(G.fNow, 4) + ', rule ' + fmt(G.ruleNow, 4) + ' &nbsp; largest difference since z = 5: ' + fmt(G.worst, 4) + ' &nbsp; growth since matter = radiation: × ' + EV.group(G.sinceEq) + ' &nbsp; D(∞)/D(now) = ' + fmt(G.DinfOverNow, 4) + ' by the equation, ' + fmt(G.ruleInfOverNow, 4) + ' by the rule &nbsp; D(a = 10)/D(∞) = ' + fmt(G.D10OverInf, 4));
    setLive('tvO', 'matter over radiation: 3 × (1/3 − 0) = +1 per e-fold &nbsp; dark energy over matter: 3 × (0 − (−1)) = +3 &nbsp; with DESI’s hinted w today: ' + fmt(D.rateNow, 3));
    setLive('tvPhi', 'fastest at Ω<sub>Λ</sub>(a) = ½: z = ' + fmt(L.equal.z, 4) + ', ' + fmt(L.equal.efoldsAgo, 3) + ' e-folds ago &nbsp; now φ/π = ' + fmt(L.now.frac, 4) + ', dφ/d ln a = ' + fmt(1.5 * Math.sin(L.now.phi), 4));
    setLive('tvGd', 't<sub>Λ</sub> = ' + fmt(L.tLambda / Gyr, 3) + ' billion years &nbsp; slowing turns to speeding: C*Ω = ' + fmt(L.accel.frac, 4) + ' at ' + fmt(L.accel.t / Gyr, 2) + ' &nbsp; equality: ½ at t<sub>Λ</sub> ln(1 + √2) = ' + fmt(L.equal.t / Gyr, 2) + ' &nbsp; now: ' + fmt(L.now.frac, 4) + ' at ' + fmt(L.now.t / Gyr, 2) + ' (with radiation the age is ' + fmt(CO.age / Gyr, 2) + ', and the formula is off by ' + e10(Math.abs(L.errWithRadiation(1)), 1) + ')');
    setLive('tvCut', [0.1, 0.01, 1e-6].map(s => 'σ = ' + (s < 1e-3 ? '10<sup>−6</sup>' : s) + ': ' + fmt(L.cutTime(s) / Gyr, 1) + ' (approx. ' + fmt(L.cutApprox(s) / Gyr, 1) + ')').join(' &nbsp; ') + ' billion years &nbsp; recombination complete only for σ ≥ ' + T.RESIDUALS.map(R => fmt(R.sigma, 4)).join(', ') + ' at residuals ' + T.RESIDUALS.map(R => R.x).join(', '));
    setLive('tvDesi', 'DESI DR2, w₀ = ' + EV.DESI2.w0 + ', w<sub>a</sub> = ' + EV.DESI2.wa + ', Ω<sub>m</sub> = ' + EV.DESI2.Om + ': dark energy’s share of matter plus dark energy peaks at ' + fmt(D.share * 100, 1) + '% at a = ' + fmt(D.aPeak, 2) + ', falls below 1% by a = ' + fmt(D.aOnePercent, 1) + '; C*Ω on the last leg never passes ' + fmt(D.maxFrac, 3) + ' (today ' + fmt(D.nowFrac, 3) + ').'); }

  /* ── controls ── */
  $('tA').addEventListener('input', e => { stopPlay(); if (tm.follow) setFollow(false); setA(Math.exp(+e.target.value / 100), true); });
  $('tNow').addEventListener('click', () => { stopPlay(); if (tm.follow) setFollow(false); setA(1); });
  $('tFollow').addEventListener('click', () => { stopPlay(); setFollow(!tm.follow); });
  const MOMENTS = [UMIN].concat(T.EVENTS.map(e => Math.log(e.a)), [UMAX]);
  $('tPlay').addEventListener('click', () => { if (tm.follow) setFollow(false);
    if (reduce) { const u = Math.log(tm.a), nx = MOMENTS.find(m => m > u + 0.006); setA(Math.exp(nx === undefined ? UMIN : nx)); return; }
    if (tm.play) { stopPlay(); return; } const u0 = Math.log(tm.a) >= UMAX - 0.01 || Math.abs(Math.log(tm.a)) < 0.004 ? UMIN : Math.log(tm.a), t0 = performance.now(), dur = 16000 * (UMAX - u0) / (UMAX - UMIN); $('tPlay').textContent = 'Stop';
    const step = now => { const k = clamp((now - t0) / dur, 0, 1); setA(Math.exp(u0 + (UMAX - u0) * k)); if (k < 1) tm.play = requestAnimationFrame(step); else stopPlay(); }; tm.play = requestAnimationFrame(step); });
  $('tSig').addEventListener('input', e => setSigma(Math.pow(10, +e.target.value / 100), true));
  if (reduce) $('tPlay').textContent = 'Next moment';

  function drawAll() { drawLoop(); drawOct(); drawRes(); drawHyd(); }
  fill(); drawClocks(); drawAll(); updRes();
  let lastW = 0; new ResizeObserver(() => { const w = $('tOct').clientWidth; if (Math.abs(w - lastW) > 1) { lastW = w; drawAll(); } }).observe($('tOct'));
  TM = { tm, sync, setA, setSigma, setFollow, frameA, drawAll }; window.EVTIME = TM; sync();
})();
