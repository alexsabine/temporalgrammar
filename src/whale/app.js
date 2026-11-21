/* The Humpback: drawing, sound and controls. The model is in model.js (window.WHALE_MODEL). */
(function () {
'use strict';
const M = window.WHALE_MODEL, TEST = window.__TEST || null;
const $ = id => document.getElementById(id);
const cv = $('sea');
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
const small = Math.min(innerWidth, innerHeight) < 560;
const { clamp, lerp, sstep, angd } = M, TAU = Math.PI * 2;
if (TEST) document.body.classList.add('film');

let simKey = 'live';
let sim = M.makeSim({ seed: 7, nFish: small ? 260 : 420, auto: !TEST });
const sea = { T: 18, S: 35, fromChannel: false };
const DEF = { cam: 'under', yaw: 0, pitch: 0, zoom: 1 };
let view = Object.assign({}, DEF);

/* ---------- what tests and films can read ---------- */
function state() {
  const w = k => { const q = sim.whales[k]; return { role: q.role, length: q.L, x: q.x, y: q.y, z: q.z, depth: Math.max(0, -q.y), yaw: q.yaw, pitch: q.pitch, roll: q.roll, speed: q.v, state: q.state, count: q.C, sinceBreath: q.tb, blows: q.nb, gape: q.gape, flukePhase: q.ph, blowhole: sim.blowhole(q), snout: sim.snout(q) }; };
  const m = sim.mother, ch = M.channel(sea.T, sea.S), d = sim.dives, mean = d.length ? d.reduce((a, b) => a + b, 0) / d.length : null;
  return { frame: sim.frame, t: sim.t, mother: w(0), calf: w(1), singer: w(2),
    song: { singing: sim.male.singing, sounding: !!sim.sounding(), unit: sim.unit ? sim.unit.unit : null, theme: sim.unit ? sim.unit.theme : null, phrase: sim.unit ? sim.unit.phrase : null, of: sim.unit ? sim.unit.of : null, k: sim.unit ? sim.unit.k : null, unitFrame: sim.unitFrame, songs: sim.singer.songs },
    fish: { n: sim.NF, taken: sim.eaten, polarisation: sim.school.pol, x: sim.school.x, y: sim.school.y, z: sim.school.z },
    net: sim.net ? { phase: sim.net.phase, x: sim.net.cx, z: sim.net.cz, ringRadius: sim.net.ringN ? sim.net.ringR / sim.net.ringN : 0, bubblesUp: sim.net.ringN } : null,
    dives: { n: d.length, mean, cv: d.length > 1 ? Math.sqrt(d.reduce((a, b) => a + (b - mean) * (b - mean), 0) / d.length) / mean : null },
    sea: { surfaceTemp: sea.T, salinity: sea.S, speedAtMother: M.speedAt(Math.max(0, -m.y), sea.T, sea.S), channelDepth: ch.depth, channelSpeed: ch.speed, surfaceSpeed: ch.surface },
    log: sim.log.filter(e => e.type !== 'unit').slice(-60), camera: camInfo };
}
let camInfo = null;
window.WHALE = { song: M.SONG, state, soundSpeed: M.soundSpeed, speedAt: M.speedAt, tempAt: M.tempAt, channel: M.channel, rays: M.rays, model: M, sim: () => sim,
  trigger: what => sim.trigger(what), setSea: (T, S) => { sea.T = T; sea.S = S; drawOcean(); } };

/* ---------- side panels (plain canvas) ---------- */
const INK = '#14202a', SOFT = '#4d5a61', HAIR = '#d9cfae', GOLD = '#b98619', VIR = '#0d7a70';
function fit2d(c) { const r = c.getBoundingClientRect(), d = Math.min(2, devicePixelRatio || 1), w = Math.max(10, Math.round(r.width * d)), h = Math.max(10, Math.round(r.height * d)); if (c.width !== w || c.height !== h) { c.width = w; c.height = h; } const g = c.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); return { g, w: r.width, h: r.height }; }
const fmt = n => Math.round(n).toLocaleString('en-GB');
function drawOcean() {
  const c = $('ocean'); if (!c || TEST) return; const { g, w, h } = fit2d(c);
  g.clearRect(0, 0, w, h);
  const ch = M.channel(sea.T, sea.S), Z = M.SEA.bottom, top = 16, bot = h - 20, x0 = 30, x1 = x0 + Math.max(60, w * 0.24), x2 = x1 + 14, x3 = w - 4;
  const yz = z => top + (bot - top) * z / Z;
  let lo = 1e9, hi = 0; const cs = []; for (let z = 0; z <= Z; z += 25) { const v = M.speedAt(z, sea.T, sea.S); cs.push(v); lo = Math.min(lo, v); hi = Math.max(hi, v); }
  lo = Math.floor(lo / 10) * 10 - 5; hi = Math.ceil(hi / 10) * 10 + 5; const xc = v => x0 + (x1 - x0) * (v - lo) / (hi - lo);
  g.font = '12px "IM Fell English SC", Georgia, serif'; g.fillStyle = SOFT; g.textBaseline = 'middle';
  /* water tint behind the rays */
  const grd = g.createLinearGradient(0, top, 0, bot); grd.addColorStop(0, '#d8ebe9'); grd.addColorStop(0.25, '#bcd6da'); grd.addColorStop(1, '#8fb0bd'); g.fillStyle = grd; g.fillRect(x2, top, x3 - x2, bot - top);
  g.fillStyle = '#eef3ee'; g.fillRect(x0, top, x1 - x0, bot - top);
  g.strokeStyle = HAIR; g.lineWidth = 1; g.strokeRect(x0 + 0.5, top + 0.5, x1 - x0 - 1, bot - top - 1); g.strokeRect(x2 + 0.5, top + 0.5, x3 - x2 - 1, bot - top - 1);
  g.fillStyle = SOFT; g.textAlign = 'right';
  for (const z of [0, 1000, 2000, 3000, 4000]) { g.fillText(z ? z / 1000 + ' km' : '0', x0 - 4, yz(z)); g.strokeStyle = 'rgba(20,32,42,.12)'; g.beginPath(); g.moveTo(x0, yz(z) + 0.5); g.lineTo(x3, yz(z) + 0.5); g.stroke(); }
  g.textAlign = 'center'; g.textBaseline = 'alphabetic'; g.fillText('speed', (x0 + x1) / 2, top - 4); g.fillText('150 km of sea', (x2 + x3) / 2, top - 4);
  g.textBaseline = 'top'; g.textAlign = 'left'; g.fillText(Math.round(lo + 5), x0, bot + 4); g.textAlign = 'right'; g.fillText(Math.round(hi - 5) + ' m/s', x1, bot + 4);
  /* the channel */
  const deep = ch.depth > 30;
  g.strokeStyle = GOLD; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(x0, yz(ch.depth) + 0.5); g.lineTo(x3, yz(ch.depth) + 0.5); g.stroke(); g.setLineDash([]);
  /* the speed curve */
  g.strokeStyle = INK; g.lineWidth = 1.6; g.beginPath(); cs.forEach((v, i) => { const X = xc(v), Y = yz(i * 25); i ? g.lineTo(X, Y) : g.moveTo(X, Y); }); g.stroke();
  g.fillStyle = GOLD; g.beginPath(); g.arc(xc(ch.speed), yz(ch.depth), 3.2, 0, TAU); g.fill();
  /* rays */
  const src = sea.fromChannel ? ch.depth : Math.max(5, -sim.male.y), R = 150000;
  const rr = M.rays(sea.T, sea.S, src, [-12, -9, -6, -3, -1, 1, 3, 6, 9, 12], R, 60);
  g.save(); g.beginPath(); g.rect(x2, top, x3 - x2, bot - top); g.clip(); g.lineWidth = 1;
  for (const r of rr) { g.strokeStyle = r.bot ? 'rgba(20,32,42,.28)' : r.top ? 'rgba(13,122,112,.55)' : 'rgba(10,86,81,.95)'; g.beginPath(); for (let i = 0; i < r.pts.length; i += 2) { const X = x2 + (x3 - x2) * r.pts[i] / R, Y = yz(r.pts[i + 1]); i ? g.lineTo(X, Y) : g.moveTo(X, Y); } g.stroke(); }
  g.restore(); g.fillStyle = INK; g.beginPath(); g.arc(x2 + 2, yz(src), 3, 0, TAU); g.fill();
  const trapped = rr.filter(r => !r.top && !r.bot).length, lost = rr.filter(r => r.bot).length;
  $('oceanP').textContent = (deep ? 'Sound is slowest at ' + fmt(ch.depth) + ' m: ' + fmt(ch.speed) + ' m/s, against ' + fmt(ch.surface) + ' at the surface. ' : 'The slowest water is at the surface, ' + fmt(ch.speed) + ' m/s, so the channel is there. ')
    + (!deep && sea.fromChannel ? 'Paths from there bend back up and skip along under the surface.' : 'Of ten paths from ' + (sea.fromChannel ? 'that depth' : 'the singer, ' + fmt(src) + ' m down') + ', ' + trapped + (trapped === 1 ? ' stays' : ' stay') + ' clear of surface and bottom and ' + lost + (lost === 1 ? ' ends' : ' end') + ' in the sea bed.');
}
function drawStrip() {
  const c = $('strip'); if (!c || TEST) return; const { g, w, h } = fit2d(c); g.clearRect(0, 0, w, h);
  const u = sim.unit, top = 6, bot = h - 16, x0 = 34, x1 = w - 6, fy = f => bot - (bot - top) * Math.log(f / 80) / Math.log(5000 / 80);
  g.font = '12px "IM Fell English SC", Georgia, serif'; g.textBaseline = 'middle'; g.textAlign = 'right'; g.fillStyle = SOFT;
  for (const f of [100, 1000, 4000]) { g.fillText(f < 1000 ? f + '' : f / 1000 + 'k', x0 - 5, fy(f)); g.strokeStyle = 'rgba(20,32,42,.12)'; g.beginPath(); g.moveTo(x0, fy(f) + 0.5); g.lineTo(x1, fy(f) + 0.5); g.stroke(); }
  g.textAlign = 'left'; g.textBaseline = 'top'; g.fillText('Hz', 4, bot + 2);
  if (!u) { $('songP').textContent = 'The singer has not begun.'; return; }
  const seq = M.SONG.sequence; let i0 = seq.findIndex(e => e.themeIndex === u.themeIndex && e.phrase === u.phrase && e.k === 1), i1 = i0; while (i1 + 1 < seq.length && seq[i1 + 1].themeIndex === u.themeIndex && seq[i1 + 1].phrase === u.phrase) i1++;
  const t0 = seq[i0].t, tEnd = (i1 + 1 < seq.length ? seq[i1 + 1].t : M.SONG.duration), span = tEnd - t0, cur = seq[i0 + u.k - 1], now = cur.t - t0 + (sim.frame - sim.unitFrame) / 60, tx = t => x0 + (x1 - x0) * t / span;
  for (let i = i0; i <= i1; i++) { const e = seq[i], un = M.UNIT[e.unit], on = e === cur && sim.sounding();
    g.strokeStyle = on ? VIR : i < i0 + u.k - 1 || e === cur ? INK : 'rgba(20,32,42,.4)'; g.lineWidth = on ? 3 : 1.8; g.beginPath();
    for (let k = 0; k <= 24; k++) { const t = un.dur * k / 24, X = tx(e.t - t0 + t), Y = fy(M.SONG.pitchAt(un, t)); k ? g.lineTo(X, Y) : g.moveTo(X, Y); } g.stroke(); }
  if (sim.male.singing) { g.strokeStyle = GOLD; g.lineWidth = 1; g.beginPath(); g.moveTo(tx(Math.min(now, span)) + 0.5, top); g.lineTo(tx(Math.min(now, span)) + 0.5, bot); g.stroke(); }
  const th = M.THEMES[u.themeIndex];
  $('songP').textContent = sim.male.singing ? 'Theme ' + u.theme + ' (' + th.name + '), phrase ' + u.phrase + ' of ' + u.of + ', unit ' + u.k + ' of ' + u.len + ': ' + M.UNIT[u.unit].name + '. Four themes, always in the same order.' : 'He has stopped singing to feed.';
}

/* ---------- three.js ---------- */
const T3 = window.THREE; let renderer = null, gl = null;
function noGL() { const n = $('nogl'); if (n) n.style.display = 'grid'; }
if (T3) { try { renderer = new T3.WebGLRenderer({ canvas: cv, antialias: !small, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: !!TEST }); gl = renderer.getContext(); } catch (e) { renderer = null; } }
if (!renderer) { noGL(); }

const SUN = [0.45, 0.82, 0.35]; { const l = Math.hypot(...SUN); SUN[0] /= l; SUN[1] /= l; SUN[2] /= l; }
const SUNR = [SUN[0] / 1.333, 0, SUN[2] / 1.333]; SUNR[1] = Math.sqrt(1 - SUNR[0] * SUNR[0] - SUNR[2] * SUNR[2]);

const COMMON = `
precision highp float;
uniform vec3 uCam; uniform vec3 uSun; uniform vec3 uSunR; uniform float uTime; uniform float uUnder;
const vec3 EXT = vec3(0.17, 0.042, 0.024);
const vec3 KD = vec3(0.26, 0.050, 0.026);
const vec3 BODY = vec3(0.006, 0.060, 0.135);
float hash(vec2 p){ p = fract(p*vec2(123.34, 456.21)); p += dot(p, p+45.32); return fract(p.x*p.y); }
float vnoise(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.0-2.0*f); return mix(mix(hash(i), hash(i+vec2(1,0)), f.x), mix(hash(i+vec2(0,1)), hash(i+vec2(1,1)), f.x), f.y); }
float fbm(vec2 p){ float a = 0.5, s = 0.0; for(int i=0;i<4;i++){ s += a*vnoise(p); p = p*2.03+vec2(17.1, 9.2); a *= 0.5; } return s; }
vec3 skyCol(vec3 d){ float y = max(d.y, 0.0); vec3 c = mix(vec3(0.62,0.76,0.90), vec3(0.10,0.30,0.72), pow(y, 0.55));
  float s = max(dot(d, uSun), 0.0); c += vec3(1.0,0.94,0.82)*(pow(s, 1200.0)*40.0 + pow(s, 90.0)*0.9 + pow(s, 8.0)*0.16); return c; }
vec3 waterCol(vec3 d, float camDepth){
  float up = d.y; vec3 deep = vec3(0.002,0.022,0.070), mid = vec3(0.006,0.105,0.230), top = vec3(0.050,0.340,0.470);
  vec3 c = mix(mid, top, smoothstep(0.0, 0.95, up)); c = mix(c, deep, smoothstep(0.05, -0.85, up));
  float g = max(dot(d, uSunR), 0.0); c += vec3(0.05,0.16,0.15)*pow(g, 5.0) + vec3(0.10,0.20,0.16)*pow(g, 40.0);
  return c*exp(-camDepth*vec3(0.050,0.020,0.012)); }
vec3 applyWater(vec3 col, vec3 wp){
  vec3 v = wp - uCam; float dist = length(v); vec3 d = v/max(dist, 1e-4); float dw = dist; vec3 sc;
  if(uUnder < 0.5){ if(wp.y >= 0.0) return col; float sn2 = (1.0 - d.y*d.y)/1.777; dw = -wp.y/sqrt(max(1.0 - sn2, 0.05)); sc = BODY; }
  else { if(wp.y > 0.0) dw = dist*(-uCam.y)/max(wp.y - uCam.y, 1e-3); sc = waterCol(d, max(0.0, -uCam.y)); }
  vec3 T = exp(-EXT*dw); return col*T + sc*(1.0 - T); }
float caustic(vec2 uv, float t){
  vec2 p = mod(uv*6.2831853, 6.2831853) - 250.0; vec2 i = p; float c = 1.0; const float inten = 0.005;
  for(int n=0;n<4;n++){ float tt = t*(1.0 - 3.5/float(n+1)); i = p + vec2(cos(tt - i.x) + sin(tt + i.y), sin(tt - i.y) + cos(tt + i.x));
    c += 1.0/length(vec2(p.x/(sin(i.x+tt)/inten), p.y/(cos(i.y+tt)/inten))); }
  c /= 4.0; c = 1.17 - pow(c, 1.4); return pow(abs(c), 5.0); }
float caust(vec3 wp){ vec2 q = wp.xz - wp.y*uSunR.xz/uSunR.y; float a = caustic(q*0.16, uTime*0.5), b = caustic(q*0.11 + 3.7, uTime*0.38 + 2.0); return clamp(sqrt(a*b)*1.5 + (a + b)*0.22, 0.0, 2.2); }
vec3 tone(vec3 x){ x *= 1.25; x = (x*(2.51*x + 0.03))/(x*(2.43*x + 0.59) + 0.14); return pow(clamp(x, 0.0, 1.0), vec3(0.4545)); }
`;
const GLOB = { uCam: { value: null }, uSun: { value: null }, uSunR: { value: null }, uTime: { value: 0 }, uUnder: { value: 1 } };

/* ----- the whale's shape, in body lengths: snout at x = +0.4, flukes at x = -0.6, back up ----- */
const PROF = [[0, .004, .003, .005], [.02, .045, .020, .036], [.06, .074, .032, .062], [.12, .094, .046, .090], [.2, .108, .064, .108], [.3, .116, .090, .118], [.4, .116, .106, .118], [.5, .106, .108, .110], [.6, .086, .100, .094], [.7, .060, .086, .074], [.8, .036, .062, .054], [.9, .019, .040, .036], [.96, .012, .026, .024], [1.0, .003, .006, .006]];
function prof(s) {
  s = clamp(s, 0, 1); let i = 0; while (i < PROF.length - 2 && s > PROF[i + 1][0]) i++;
  const a = PROF[i], b = PROF[i + 1], p0 = PROF[Math.max(0, i - 1)], p3 = PROF[Math.min(PROF.length - 1, i + 2)], h = b[0] - a[0], t = (s - a[0]) / h, o = [];
  for (let k = 1; k <= 3; k++) { const m0 = i > 0 ? (b[k] - p0[k]) / (b[0] - p0[0]) * h : (b[k] - a[k]), m1 = i + 2 < PROF.length ? (p3[k] - a[k]) / (p3[0] - a[0]) * h : (b[k] - a[k]);
    const t2 = t * t, t3 = t2 * t; o.push(Math.max(0.002, (2 * t3 - 3 * t2 + 1) * a[k] + (t3 - 2 * t2 + t) * m0 + (-2 * t3 + 3 * t2) * b[k] + (t3 - t2) * m1)); }
  return { w: o[0], t: o[1], b: o[2], n: s < 0.25 ? 2.5 : s < 0.45 ? lerp(2.5, 2.1, (s - 0.25) / 0.2) : s < 0.7 ? 2.1 : lerp(2.1, 1.65, Math.min(1, (s - 0.7) / 0.2)) };
}
const THJ = 1.3, HINGE = 0.27;
function skin(s, th, lower) {
  const p = prof(s), sn = Math.sin(th), cs = Math.cos(th), e = 2 / p.n;
  let z = p.w * Math.sign(sn) * Math.pow(Math.abs(sn), e), y = (cs > 0 ? p.t : p.b) * Math.sign(cs) * Math.pow(Math.abs(cs), e);
  if (cs > 0) { y += 0.013 * Math.exp(-Math.pow((s - 0.655) / 0.05, 2)) * Math.pow(cs, 3); y += 0.007 * Math.exp(-Math.pow((s - 0.168) / 0.03, 2)) * Math.exp(-Math.pow(th > Math.PI ? th - TAU : th, 2) / 0.1); }
  if (lower && s < HINGE) { const f = sstep(HINGE, HINGE - 0.08, s), a = Math.abs((th > Math.PI ? TAU - th : th) - THJ); const lip = Math.exp(-a * a / 0.05) * f; z *= 1 + 0.07 * lip; y += 0.004 * lip; }
  return [0.4 - s, y, z];
}
function buildWhale() {
  const R = M.rng32(4242), pos = [], aS = [], aJaw = [], aPart = [], aInfo = [], idx = [];
  const NS = small ? 120 : 150, NU = small ? 22 : 26, NLo = small ? 34 : 40;
  const sAt = i => { const u = i / NS; return u < 0.5 ? 0.3 * u / 0.5 : 0.3 + 0.7 * (u - 0.5) / 0.5; };
  /* tubercles: knobs on the head and along the lower jaw */
  const tub = [];
  for (let i = 0; i < 7; i++) tub.push([0.016 + i * 0.021, 0, .0058, .0095]);
  for (const sg of [-1, 1]) { for (let i = 0; i < 9; i++) tub.push([0.02 + i * 0.022 + 0.004 * R(), sg * (0.72 + 0.1 * R()), .0052, .0088]); for (let i = 0; i < 9; i++) tub.push([0.012 + i * 0.024 + 0.004 * R(), sg > 0 ? THJ + 0.2 + 0.08 * R() : TAU - THJ - 0.2 - 0.08 * R(), .0052, .0088]); }
  for (let i = 0; i < 10; i++) tub.push([0.03 + 0.16 * R(), (R() * 2 - 1) * 1.05, .0045, .008]);
  for (let i = 0; i < 6; i++) tub.push([0.018 + 0.04 * R(), Math.PI + (R() - 0.5) * 0.9, .005, .009]);
  const tp = tub.map(t => { const th = (t[1] + TAU) % TAU, lower = th > THJ && th < TAU - THJ; return skin(t[0], th, lower); });
  const seamU = [[], []], seamL = [[], []];
  function grid(lower, NA, th0, th1) {
    const base = pos.length / 3;
    for (let i = 0; i <= NS; i++) { const s = sAt(i);
      for (let j = 0; j <= NA; j++) {
        const th = ((lerp(th0, th1, j / NA)) + TAU) % TAU, p = skin(s, th, lower), fold = th > Math.PI ? TAU - th : th;
        let bump = 0;
        if (s < 0.27) { let d = 0; for (let k = 0; k < tub.length; k++) { const q = tp[k], dx = p[0] - q[0], dy = p[1] - q[1], dz = p[2] - q[2], g = Math.exp(-(dx * dx + dy * dy + dz * dz) / (tub[k][3] * tub[k][3])); d += tub[k][2] * g; bump = Math.max(bump, g); }
          const l = Math.hypot(p[1], p[2]) + 1e-6, nx = s < 0.03 ? 0.6 : 0.1; p[0] += d * nx; p[1] += d * p[1] / l; p[2] += d * p[2] / l; }
        pos.push(p[0], p[1], p[2]); aS.push(s); aPart.push(lower ? 1 : 0);
        aJaw.push(lower && s < HINGE ? sstep(HINGE, 0.06, s) : 0);
        aInfo.push(fold, bump, lower ? sstep(THJ + 0.1, THJ + 0.75, fold) * sstep(0.02, 0.12, s) * (1 - sstep(0.42, 0.6, s)) : 0);
        if (j === 0) (lower ? seamL : seamU)[0].push(base + i * (NA + 1)); if (j === NA) (lower ? seamL : seamU)[1].push(base + i * (NA + 1) + NA);
      } }
    for (let i = 0; i < NS; i++) for (let j = 0; j < NA; j++) { const a = base + i * (NA + 1) + j, b = a + NA + 1; idx.push(a, b, a + 1, a + 1, b, b + 1); }
  }
  grid(false, NU, -THJ, THJ); grid(true, NLo, THJ, TAU - THJ);
  /* flukes */
  const XH = 0.4 - 0.955, NE = small ? 32 : 44, NC = 8;
  function flukePt(eta, c, sgn) { const ae = Math.abs(eta), le = XH + 0.036 - 0.088 * Math.pow(ae, 1.25), chord = 0.108 * Math.pow(Math.max(0, 1 - Math.pow(ae, 2.2)), 0.75) + 0.004;
    const te = le - chord + 0.032 * Math.exp(-Math.pow(eta / 0.07, 2)) + 0.0035 * Math.sin(ae * 46) * sstep(0.1, 0.3, ae);
    const th = 0.017 * Math.pow(1 - ae, 0.7) * Math.sqrt(c) * (1 - c) * 2.6; return [le + (te - le) * c, sgn * th + 0.002, eta * 0.168]; }
  for (const sgn of [1, -1]) { const base = pos.length / 3;
    for (let i = 0; i <= NC; i++) for (let j = 0; j <= NE; j++) { const p = flukePt(j / NE * 2 - 1, i / NC, sgn); pos.push(p[0], p[1], p[2]); aS.push(0.955); aJaw.push(0); aPart.push(2); aInfo.push(sgn > 0 ? 0 : Math.PI, i / NC, 0); }
    for (let i = 0; i < NC; i++) for (let j = 0; j < NE; j++) { const a = base + i * (NE + 1) + j, b = a + NE + 1; if (sgn > 0) idx.push(a, b, a + 1, a + 1, b, b + 1); else idx.push(a, a + 1, b, a + 1, b + 1, b); } }
  /* the small fin on its hump */
  for (const sgn of [1, -1]) { const base = pos.length / 3, NUf = 10, NVf = 5;
    for (let i = 0; i <= NUf; i++) for (let j = 0; j <= NVf; j++) { const u = i / NUf, v = j / NVf, s0 = lerp(0.622, 0.70, u), hh = 0.019 * Math.pow(Math.sin(Math.PI * Math.pow(u, 0.62)), 0.85);
      const s = clamp(s0 + 0.022 * v * v, 0, 1), y = skin(s0, 0, false)[1] - 0.004 + v * hh; pos.push(0.4 - s, y, sgn * 0.0065 * (1 - v * v) * Math.sin(Math.PI * u)); aS.push(s); aJaw.push(0); aPart.push(3); aInfo.push(0.3, 0, 0); }
    for (let i = 0; i < NUf; i++) for (let j = 0; j < NVf; j++) { const a = base + i * (NVf + 1) + j, b = a + NVf + 1; if (sgn > 0) idx.push(a, a + 1, b, a + 1, b + 1, b); else idx.push(a, b, a + 1, a + 1, b, b + 1); } }
  /* baleen: a dark fringe under each side of the upper jaw, hidden until the mouth opens */
  for (const sgn of [1, -1]) { const base = pos.length / 3, NB = 30;
    for (let i = 0; i <= NB; i++) { const s = lerp(0.03, 0.255, i / NB), p = skin(s, sgn > 0 ? THJ : TAU - THJ, false), f = Math.sin(Math.PI * i / NB);
      pos.push(p[0], p[1] - 0.002, p[2] * 0.97, p[0], p[1] - 0.012 - 0.05 * f, p[2] * 0.86); for (let k = 0; k < 2; k++) { aS.push(s); aJaw.push(0); aPart.push(5); aInfo.push(1.6, 0, 0); } }
    for (let i = 0; i < NB; i++) { const a = base + i * 2; idx.push(a, a + 1, a + 2, a + 2, a + 1, a + 3); } }
  const g = new T3.BufferGeometry();
  g.setAttribute('position', new T3.Float32BufferAttribute(pos, 3)); g.setAttribute('aS', new T3.Float32BufferAttribute(aS, 1)); g.setAttribute('aJaw', new T3.Float32BufferAttribute(aJaw, 1));
  g.setAttribute('aPart', new T3.Float32BufferAttribute(aPart, 1)); g.setAttribute('aInfo', new T3.Float32BufferAttribute(aInfo, 3)); g.setIndex(idx); g.computeVertexNormals();
  /* close the seam behind the jaw hinge so the flank is smooth */
  const nrm = g.attributes.normal;
  for (let side = 0; side < 2; side++) { const U = seamU[side === 0 ? 1 : 0], L = seamL[side]; for (let i = 0; i < U.length; i++) { if (aS[U[i]] < HINGE - 0.02) continue; const a = U[i], b = L[i];
    let x = nrm.getX(a) + nrm.getX(b), y = nrm.getY(a) + nrm.getY(b), z = nrm.getZ(a) + nrm.getZ(b); const l = Math.hypot(x, y, z) || 1; x /= l; y /= l; z /= l; nrm.setXYZ(a, x, y, z); nrm.setXYZ(b, x, y, z); } }
  g.boundingSphere = new T3.Sphere(new T3.Vector3(-0.1, 0, 0), 1.2);
  const eye = skin(0.243, THJ - 0.2, false), bh = skin(0.168, 0, false);
  return { geo: g, eye, bh };
}
function buildFlipper() {
  const pos = [], aS = [], aJaw = [], aPart = [], aInfo = [], idx = [], NUp = small ? 36 : 48, NC = 8, LP = 0.32;
  for (const sgn of [1, -1]) { const base = pos.length / 3;
    for (let i = 0; i <= NUp; i++) for (let j = 0; j <= NC; j++) { const u = i / NUp, c = j / NC;
      const chord = 0.068 * Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(0.12 + 0.88 * u, 0.8))), 0.6) + 0.002, xm = -0.035 * u * u;
      const knob = Math.pow(Math.max(0, Math.sin(Math.PI * u * 10.5)), 1.2) * (0.35 + 0.65 * u), le = xm + 0.45 * chord + 0.0075 * knob, te = xm - 0.55 * chord + 0.003 * Math.sin(u * 31) * u;
      const th = 0.0115 * (1 - 0.62 * u) * Math.sqrt(c) * (1 - c) * 2.6 * (1 + 0.5 * knob * (1 - c) * (1 - c));
      pos.push(le + (te - le) * c, sgn * th, u * LP); aS.push(0.3); aJaw.push(0); aPart.push(4); aInfo.push(sgn > 0 ? 0 : Math.PI, knob * (1 - c) * (1 - c), 0); }
    for (let i = 0; i < NUp; i++) for (let j = 0; j < NC; j++) { const a = base + i * (NC + 1) + j, b = a + NC + 1; if (sgn > 0) idx.push(a, b, a + 1, a + 1, b, b + 1); else idx.push(a, a + 1, b, a + 1, b + 1, b); } }
  const g = new T3.BufferGeometry();
  g.setAttribute('position', new T3.Float32BufferAttribute(pos, 3)); g.setAttribute('aS', new T3.Float32BufferAttribute(aS, 1)); g.setAttribute('aJaw', new T3.Float32BufferAttribute(aJaw, 1));
  g.setAttribute('aPart', new T3.Float32BufferAttribute(aPart, 1)); g.setAttribute('aInfo', new T3.Float32BufferAttribute(aInfo, 3)); g.setIndex(idx); g.computeVertexNormals();
  g.boundingSphere = new T3.Sphere(new T3.Vector3(0, 0, 0.15), 0.4);
  return g;
}
const WHALE_VS = COMMON + `
attribute float aS; attribute float aJaw; attribute float aPart; attribute vec3 aInfo;
uniform float uPh, uAmp, uBend, uArch, uGape, uGulp, uFlex;
varying vec3 vW; varying vec3 vN; varying vec3 vRest; varying vec3 vAx; varying float vS; varying float vPart; varying vec3 vInfo;
float hy(float s){ float e = 0.005 + 0.105*pow(smoothstep(0.3, 1.0, s), 1.5); float d = s - 0.4; return uFlex*(uAmp*e*sin(uPh - 4.2*s) + uArch*d*d); }
void main(){
  vec3 p = position; vec3 n = normal; float s = aS; vRest = position; vS = s; vPart = aPart; vInfo = aInfo;
  if(aJaw > 0.0){ float a = -uGape*0.85*aJaw; float c = cos(a), sn = sin(a); vec2 h = vec2(0.4 - 0.27, 0.02); vec2 q = p.xy - h; p.xy = h + vec2(c*q.x - sn*q.y, sn*q.x + c*q.y); n.xy = vec2(c*n.x - sn*n.y, sn*n.x + c*n.y); }
  if(aInfo.z > 0.0 && aPart < 1.5){ vec2 r = normalize(p.yz + vec2(0.03, 0.0)); p.yz += r*aInfo.z*uGulp*0.075; }
  float xs = 0.4 - s; float dx = p.x - xs; float e = 0.01;
  float y0 = hy(s); float sl = -(hy(s + e) - hy(s - e))/(2.0*e);
  float ang = atan(sl); if(aPart > 1.5 && aPart < 2.5) ang += uFlex*0.42*uAmp*cos(uPh - 4.2*0.97);
  float c = cos(ang), sn = sin(ang);
  vec2 q = vec2(dx, p.y); q = vec2(c*q.x - sn*q.y, sn*q.x + c*q.y); p.x = xs + q.x; p.y = y0 + q.y; n.xy = vec2(c*n.x - sn*n.y, sn*n.x + c*n.y);
  float d = s - 0.4; float zb = uFlex*uBend*d*d*0.9; float a2 = atan(-uFlex*uBend*2.0*d*0.9); c = cos(a2); sn = sin(a2);
  vec2 q2 = vec2(p.x - xs, p.z); q2 = vec2(c*q2.x - sn*q2.y, sn*q2.x + c*q2.y); p.x = xs + q2.x; p.z = zb + q2.y; n.xz = vec2(c*n.x - sn*n.z, sn*n.x + c*n.z);
  vec4 w = modelMatrix*vec4(p, 1.0); vW = w.xyz; vN = normalize(mat3(modelMatrix)*n); vAx = normalize(mat3(modelMatrix)*vec3(1.0, 0.0, 0.0));
  gl_Position = projectionMatrix*viewMatrix*w;
}`;
const WHALE_FS = COMMON + `
uniform vec3 uEye; uniform vec3 uBh; uniform float uGulp; uniform float uSeed;
varying vec3 vW; varying vec3 vN; varying vec3 vRest; varying vec3 vAx; varying float vS; varying float vPart; varying vec3 vInfo;
void main(){
  vec3 N = normalize(vN); bool back = !gl_FrontFacing; if(back) N = -N;
  vec3 V = normalize(uCam - vW);
  float th = vInfo.x; float sd = vRest.z < 0.0 ? 1.0 : 0.0;
  float m1 = fbm(vec2(vS*26.0 + uSeed, th*5.0 + sd*9.0)), m2 = fbm(vec2(vS*90.0, th*22.0 + sd*5.0 + uSeed));
  vec3 dark = vec3(0.050, 0.056, 0.064) + 0.035*m1; vec3 pale = vec3(0.52, 0.54, 0.52)*(0.82 + 0.3*m2);
  vec3 alb; float gloss = 0.5;
  if(vPart < 1.5){
    float edge = 1.92 + 0.5*(fbm(vec2(vS*6.0 + uSeed*3.0, sd*7.0)) - 0.5) + 0.45*smoothstep(0.55, 0.9, vS) - 0.25*smoothstep(0.12, 0.0, vS);
    float belly = smoothstep(edge - 0.06, edge + 0.10, th + 0.12*(m2 - 0.5));
    belly *= 1.0 - 0.75*smoothstep(0.78, 0.95, vS);
    alb = mix(dark, pale, belly);
    /* throat pleats */
    float pf = smoothstep(1.45, 1.75, th)*smoothstep(0.015, 0.08, vS)*(1.0 - smoothstep(0.50, 0.60, vS));
    float wv = th*44.0; float gr = pow(0.5 + 0.5*sin(wv), 5.0);
    alb *= 1.0 - 0.62*gr*pf; alb = mix(alb, vec3(0.42, 0.25, 0.24), uGulp*pf*smoothstep(0.2, 0.8, 0.5 + 0.5*sin(wv))*0.6);
    vec3 Tn = normalize(cross(N, vAx)); N = normalize(N + Tn*cos(wv)*0.42*pf*(vRest.z < 0.0 ? -1.0 : 1.0));
    /* knobs, barnacle specks and scars */
    alb = mix(alb, vec3(0.20, 0.21, 0.21), vInfo.y*0.55);
    float sp = smoothstep(0.72, 0.80, vnoise(vRest.xz*vec2(260.0, 240.0) + vRest.y*130.0))*smoothstep(0.3, 0.0, vS)*(0.3 + 0.7*vInfo.y);
    alb = mix(alb, vec3(0.55, 0.55, 0.50), sp*0.7);
    float scar = 0.5*smoothstep(0.992, 1.0, sin(vS*140.0 + th*9.0 + m1*8.0))*smoothstep(0.3, 0.5, m1)*(1.0 - belly);
    alb = mix(alb, vec3(0.16, 0.17, 0.18), scar*0.6);
    /* mouth line, eye, blowholes */
    alb *= 1.0 - 0.55*exp(-pow((th - 1.3)/0.035, 2.0))*(1.0 - smoothstep(0.24, 0.275, vS));
    vec3 ep = vec3(vRest.xy, abs(vRest.z)); float ed = length(ep - uEye);
    alb = mix(alb, vec3(0.004), smoothstep(0.0075, 0.0055, ed)); alb = mix(alb, vec3(0.09, 0.09, 0.09), smoothstep(0.013, 0.0085, ed)*smoothstep(0.0055, 0.0075, ed)*0.6);
    if(ed < 0.0065) gloss = 3.0;
    vec2 bq = vec2((vRest.x - uBh.x)/0.014, (abs(vRest.z) - 0.0055)/0.0032); alb *= 1.0 - 0.9*smoothstep(1.0, 0.6, length(bq))*step(0.0, vRest.y);
    if(back){ alb = vec3(0.30, 0.15, 0.15)*(0.55 + 0.45*m2); gloss = 0.2; }
  } else if(vPart < 2.5){
    float eta = abs(vRest.z)/0.168; float chord = vInfo.y;
    if(th > 1.5){ float w = smoothstep(0.16, 0.5, fbm(vec2(vRest.z*30.0 + uSeed*5.0, vRest.x*36.0)) + 0.5*(1.0 - eta) - 0.75*smoothstep(0.55, 0.95, chord) - 0.4*smoothstep(0.75, 1.0, eta) - 0.35*exp(-pow(vRest.z/0.02, 2.0)));
      alb = mix(dark*1.2, pale, w); } else alb = dark;
    alb = mix(alb, vec3(0.25), 0.25*smoothstep(0.93, 1.0, chord)*m2);
  } else if(vPart < 3.5){ alb = dark;
  } else if(vPart < 4.5){
    float u = vRest.z/0.32;
    float w = th > 1.5 ? 1.0 : smoothstep(0.38, 0.62, fbm(vec2(vRest.z*26.0 + uSeed, vRest.x*30.0)) + 0.55*u - 0.12);
    w = max(w, vInfo.y*0.9); w *= smoothstep(0.0, 0.08, u + 0.03*m2);
    alb = mix(dark*1.3, pale, w);
    alb = mix(alb, vec3(0.48, 0.48, 0.44), smoothstep(0.70, 0.80, vnoise(vRest.xz*vec2(300.0, 260.0)))*vInfo.y*0.7);
  } else { alb = vec3(0.060, 0.050, 0.040)*(0.5 + 0.5*sin(vS*1500.0)); gloss = 0.1; }

  float depth = max(0.0, -vW.y);
  vec3 att = exp(-KD*depth);
  float cz = caust(vW); float cf = exp(-depth/30.0)*smoothstep(-0.25, 0.55, N.y);
  float ndl = max(dot(N, uSunR), 0.0);
  vec3 direct = vec3(3.4, 3.3, 3.0)*att*ndl*(0.34 + 3.6*cz*cf);
  vec3 amb = mix(vec3(0.004, 0.030, 0.060), vec3(0.11, 0.36, 0.50), smoothstep(-1.0, 1.0, N.y))*exp(-vec3(0.11, 0.028, 0.016)*depth) + vec3(0.0, 0.012, 0.024);
  amb += vec3(0.01, 0.05, 0.07)*exp(-0.02*depth)*max(0.0, -N.y)*0.6;
  vec3 colU = alb*(direct + amb); if(uUnder < 0.5) colU *= 0.5;
  float fr = pow(1.0 - max(dot(N, V), 0.0), 4.0);
  colU += fr*0.10*gloss*waterCol(reflect(-V, N), depth);
  vec3 Hh = normalize(uSunR + V); colU += att*gloss*0.25*pow(max(dot(N, Hh), 0.0), 60.0)*(0.3 + cz*cf*2.0);
  /* out of the water: sun, sky and a wet shine */
  vec3 albA = alb*(0.45 + 1.5*alb); vec3 colA = albA*(vec3(1.55, 1.48, 1.36)*max(dot(N, uSun), 0.0) + mix(vec3(0.07, 0.10, 0.13), vec3(0.22, 0.30, 0.40), N.y*0.5 + 0.5));
  vec3 Ha = normalize(uSun + V); colA += gloss*(1.6*pow(max(dot(N, Ha), 0.0), 260.0) + 0.16*pow(max(dot(N, Ha), 0.0), 30.0))*vec3(1.0, 0.96, 0.9);
  vec3 rl = reflect(-V, N); rl.y = abs(rl.y); colA += gloss*(0.03 + 0.5*fr)*skyCol(rl)*0.5;
  vec3 col = mix(colU, colA, smoothstep(-0.05, 0.12, vW.y));
  col = applyWater(col, vW);
  gl_FragColor = vec4(tone(col), 1.0);
}`;

const SURF_VS = COMMON + `
varying vec3 vW; varying float vR;
uniform vec2 uCentre;
float wv(vec2 p, vec2 d, float lam, float a, float ph){ float k = 6.2831853/lam; return a*sin(dot(p, d)*k - sqrt(9.81*k)*uTime + ph); }
void main(){
  vec3 p = position; p.xz += uCentre; float r = length(position.xz); vR = r; float sp = r*0.07 + 0.3;
  float h = wv(p.xz, vec2(0.80, 0.60), 23.0, 0.17, 0.0)*clamp(2.0 - sp/5.0, 0.0, 1.0) + wv(p.xz, vec2(0.35, 0.94), 13.0, 0.10, 1.3)*clamp(2.0 - sp/3.0, 0.0, 1.0)
    + wv(p.xz, vec2(0.96, -0.28), 7.1, 0.06, 4.1)*clamp(2.0 - sp/1.6, 0.0, 1.0) + wv(p.xz, vec2(-0.5, 0.87), 4.3, 0.035, 2.2)*clamp(2.0 - sp/1.0, 0.0, 1.0);
  p.y = h; vW = p; gl_Position = projectionMatrix*viewMatrix*vec4(p, 1.0);
}`;
const SURF_FS = COMMON + `
varying vec3 vW; varying float vR;
uniform vec4 uFoam[8]; uniform vec4 uFoamK[8];
vec2 dw(vec2 p, vec2 d, float lam, float a, float ph){ float k = 6.2831853/lam; return d*(a*k*cos(dot(p, d)*k - sqrt(9.81*k)*uTime + ph)); }
void main(){
  vec2 p = vW.xz; float dist = length(vW - uCam);
  float graze = 1.0 - abs(normalize(vW - uCam).y); float fine = uUnder > 0.5 ? 1.0/(1.0 + dist*0.02) : 1.0/(1.0 + dist*dist*0.006*graze*graze); float mid = uUnder > 0.5 ? 1.0 : 1.0/(1.0 + dist*dist*0.00008*graze*graze);
  vec2 g = dw(p, vec2(0.80, 0.60), 23.0, 0.17, 0.0) + dw(p, vec2(0.35, 0.94), 13.0, 0.10, 1.3) + (dw(p, vec2(0.96, -0.28), 7.1, 0.06, 4.1) + dw(p, vec2(-0.5, 0.87), 4.3, 0.035, 2.2))*mid;
  g += (dw(p, vec2(0.2, -0.98), 2.3, 0.022, 0.7) + dw(p, vec2(-0.86, -0.5), 1.37, 0.013, 5.2) + dw(p, vec2(0.71, 0.71), 0.83, 0.008, 3.3) + dw(p, vec2(-0.3, 0.95), 0.51, 0.005, 1.1))*fine;
  g += (vec2(vnoise(p*1.9 + uTime*0.35), vnoise(p*1.9 + 31.0 - uTime*0.3)) - 0.5)*(uUnder > 0.5 ? 0.22 : 0.10)*fine;
  vec3 n = normalize(vec3(-g.x, 1.0, -g.y));
  vec3 V = normalize(vW - uCam);
  /* foam */
  float foam = 0.0, ringGlow = 0.0;
  for(int i=0;i<8;i++){ vec4 f = uFoam[i]; if(f.w <= 0.0) continue; float d = length(p - f.xy); float m;
    if(uFoamK[i].x > 0.5){ m = exp(-pow((d - f.z)/1.25, 2.0)); ringGlow += m*f.w*uFoamK[i].y; m *= 0.6*(0.45 + 0.9*vnoise(p*0.9 + 7.0)); } else m = 1.0 - smoothstep(f.z*0.55, f.z, d);
    foam += m*f.w*uFoamK[i].y; }
  float pat = smoothstep(0.30, 0.62, fbm(p*1.6 + vec2(uTime*0.03, 0.0)) + 0.25*foam);
  foam = clamp(foam*pat*1.3, 0.0, 1.0);
  vec3 col; float alpha = 1.0;
  if(uUnder > 0.5){
    vec3 nd = -n; float cosi = clamp(dot(-V, nd), 0.0, 1.0); float k = 1.0 - 1.777*(1.0 - cosi*cosi);
    vec3 rf = refract(V, nd, 1.333); if(k <= 0.0) rf = vec3(0.0, 1.0, 0.0);
    vec3 through = skyCol(normalize(rf))*vec3(0.80, 0.95, 1.0)*0.95;
    vec3 rl = reflect(V, nd); rl.y = -abs(rl.y); vec3 mirror = waterCol(rl, 0.0)*1.15;
    float win = smoothstep(0.0, 0.10, k); float fres = 0.02 + 0.98*pow(1.0 - cosi, 5.0);
    col = mix(mirror, through, win*(1.0 - fres));
    col = mix(col, vec3(0.55, 0.80, 0.82), foam*0.75) + ringGlow*vec3(0.10, 0.22, 0.20);
    col = applyWater(col, vW);
    gl_FragColor = vec4(tone(col), 1.0);
  } else {
    float c = max(dot(n, -V), 0.0); float F = 0.02 + 0.98*pow(1.0 - c, 5.0);
    vec3 rl = reflect(V, n); rl.y = abs(rl.y); vec3 sky = skyCol(rl);
    float hz = smoothstep(150.0, 1500.0, dist);
    F = mix(F, 1.0, hz); sky = mix(sky, vec3(0.62, 0.76, 0.90), hz);
    col = sky*F; alpha = F;
    /* light scattered back up out of the water */
    col += (1.0 - F)*vec3(0.004, 0.036, 0.070)*(0.6 + 0.8*max(dot(n, uSun), 0.0)); alpha += (1.0 - F)*mix(0.30, 0.62, graze*graze);
    col += ringGlow*vec3(0.05, 0.20, 0.18)*(1.0 - F);
    vec3 fc = mix(vec3(0.86, 0.92, 0.94), vec3(0.50, 0.86, 0.80), clamp(ringGlow, 0.0, 1.0)*0.55)*(0.5 + 0.55*max(dot(n, uSun), 0.0));
    foam *= 0.92; col = mix(col, fc, foam); alpha = mix(alpha, 1.0, foam);
    gl_FragColor = vec4(tone(col/max(alpha, 1e-3))*alpha, alpha);
  }
}`;
const DOME_VS = COMMON + `varying vec3 vD; void main(){ vD = position; vec4 w = vec4(position + cameraPosition, 1.0); gl_Position = projectionMatrix*viewMatrix*w; gl_Position.z = gl_Position.w*0.99999; }`;
const DOME_FS = COMMON + `varying vec3 vD;
void main(){ vec3 d = normalize(vD); vec3 col;
  if(uUnder > 0.5) col = waterCol(d, max(0.0, -uCam.y));
  else { if(d.y >= 0.0){ col = skyCol(d); float cl = smoothstep(0.52, 0.80, fbm(d.xz/(d.y + 0.12)*1.1 + 3.0))*smoothstep(0.0, 0.10, d.y)*(1.0 - smoothstep(0.30, 0.65, d.y)); col = mix(col, vec3(0.92, 0.94, 0.96), cl*0.55); } else col = mix(vec3(0.62, 0.76, 0.90), BODY, smoothstep(0.0, -0.012, d.y)); }
  gl_FragColor = vec4(tone(col), 1.0); }`;
const SHAFT_VS = COMMON + `
attribute vec3 aBase; attribute vec2 aUV; uniform float uWrap;
varying float vU; varying float vDepth; varying float vDist; varying float vSeed;
void main(){ vec2 c = aBase.xy + floor((uCam.xz - aBase.xy)/uWrap + 0.5)*uWrap; vec3 axis = -uSunR; vec3 p0 = vec3(c.x, 0.0, c.y) + axis*52.0*aUV.y;
  vec3 side = normalize(cross(axis, uCam - p0)); vec3 p = p0 + side*aUV.x*(4.5 + 4.0*aBase.z);
  vU = aUV.x; vDepth = -p.y; vDist = length(p - uCam); vSeed = aBase.z; gl_Position = projectionMatrix*viewMatrix*vec4(p, 1.0); }`;
const SHAFT_FS = COMMON + `
varying float vU; varying float vDepth; varying float vDist; varying float vSeed;
void main(){ float s = vU*3.0 + vSeed*17.0; float a = 0.5 + 0.5*sin(s*2.1 + uTime*0.5 + sin(s*0.7 - uTime*0.3)*2.0); a *= 0.5 + 0.5*sin(s*5.3 - uTime*0.8 + vSeed*40.0);
  a = pow(a, 1.6); float e = 1.0 - vU*vU; a *= e*e*exp(-vDepth/17.0)*smoothstep(0.0, 2.0, vDepth)*exp(-vDist*0.028)*smoothstep(3.0, 12.0, vDist);
  gl_FragColor = vec4(vec3(0.30, 0.62, 0.66)*a*0.62*exp(-max(0.0, -uCam.y)*0.02), 1.0); }`;
const MOTE_VS = COMMON + `
attribute float aSeed; uniform float uBox; uniform float uPx; varying float vA;
void main(){ vec3 q = position + vec3(sin(uTime*0.11 + aSeed*40.0), sin(uTime*0.07 + aSeed*70.0), cos(uTime*0.09 + aSeed*90.0))*0.012;
  vec3 p = uCam + (fract(q - uCam/uBox) - 0.5)*uBox; vec4 mv = viewMatrix*vec4(p, 1.0); float d = -mv.z;
  float sz = (0.012 + 0.03*aSeed)*uPx/max(d, 0.3); gl_PointSize = clamp(sz, 1.0, 14.0);
  vA = smoothstep(0.4, 1.5, d)*(1.0 - smoothstep(uBox*0.3, uBox*0.5, d))*step(p.y, -0.3)*exp(-max(0.0, -p.y)*0.03)*(0.35 + 0.65*fract(aSeed*17.0))*min(1.0, sz);
  gl_Position = projectionMatrix*mv; }`;
const MOTE_FS = COMMON + `varying float vA; void main(){ vec2 c = gl_PointCoord - 0.5; float a = smoothstep(0.5, 0.1, length(c))*vA; gl_FragColor = vec4(vec3(0.55, 0.85, 0.88)*a*0.55, 1.0); }`;
const FISH_VS = COMMON + `
attribute float aPh; varying vec3 vW; varying vec3 vN; varying vec3 vL;
void main(){ vec3 p = position; float t = 0.5 - p.x; p.z += sin(uTime*(9.0 + 5.0*fract(aPh*3.1)) + aPh*6.2831853 - p.x*7.0)*0.07*t*t;
  vec4 w = modelMatrix*instanceMatrix*vec4(p, 1.0); vW = w.xyz; vN = normalize(mat3(modelMatrix)*mat3(instanceMatrix)*normal); vL = position; gl_Position = projectionMatrix*viewMatrix*w; }`;
const FISH_FS = COMMON + `
varying vec3 vW; varying vec3 vN; varying vec3 vL;
void main(){ vec3 N = normalize(vN); if(!gl_FrontFacing) N = -N; vec3 V = normalize(uCam - vW); float depth = max(0.0, -vW.y); vec3 att = exp(-KD*depth);
  float up = vL.y/0.1; vec3 backc = vec3(0.012, 0.045, 0.060); float silver = smoothstep(0.55, 0.15, up);
  vec3 rl = reflect(-V, N); vec3 env = waterCol(rl, depth)*2.2;
  float ndl = max(dot(N, uSunR), 0.0); vec3 Hh = normalize(uSunR + V);
  vec3 lit = vec3(1.8)*att*(ndl*0.6 + 3.0*pow(max(dot(N, Hh), 0.0), 18.0)) + vec3(0.05, 0.20, 0.28)*exp(-0.02*depth);
  vec3 col = mix(backc*lit*1.5, vec3(0.55, 0.60, 0.62)*lit*0.55 + env*0.75, silver);
  col *= 1.0 - 0.6*smoothstep(0.012, 0.006, length(vL.xy - vec2(0.36, 0.02)));
  col = applyWater(col, vW); gl_FragColor = vec4(tone(col), 1.0); }`;
const SPRAY_VS = COMMON + `attribute float aSz; attribute float aA; uniform float uPx; varying float vA; varying vec3 vW;
void main(){ vec4 mv = viewMatrix*vec4(position, 1.0); gl_PointSize = clamp(aSz*uPx/max(-mv.z, 0.5), 1.0, 220.0); vA = aA; vW = position; gl_Position = projectionMatrix*mv; if(aA <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }`;
const SPRAY_FS = COMMON + `varying float vA; varying vec3 vW; void main(){ vec2 c = gl_PointCoord - 0.5; float r = length(c); float a = smoothstep(0.5, 0.05, r)*vA;
  vec3 col = vec3(0.92, 0.95, 0.97)*(0.75 + 0.5*smoothstep(0.3, -0.3, c.y + c.x*0.4)); gl_FragColor = vec4(tone(col)*a, a); }`;
const BUB_VS = COMMON + `attribute float aSz; attribute float aA; uniform float uPx; varying float vA; varying vec3 vW;
void main(){ vec4 mv = viewMatrix*vec4(position, 1.0); gl_PointSize = clamp(aSz*uPx/max(-mv.z, 0.5), 1.5, 60.0); vA = aA; vW = position; gl_Position = projectionMatrix*mv; if(aA <= 0.0) gl_Position = vec4(2.0, 2.0, 2.0, 1.0); }`;
const BUB_FS = COMMON + `varying float vA; varying vec3 vW; void main(){ vec2 c = gl_PointCoord - 0.5; float r = length(c)*2.0; float rim = smoothstep(0.55, 0.9, r)*smoothstep(1.0, 0.9, r); float hl = smoothstep(0.35, 0.0, length(c - vec2(-0.14, -0.16))*2.0);
  float a = clamp(rim*0.55 + hl + 0.30*smoothstep(1.0, 0.7, r), 0.0, 1.0)*vA; float depth = max(0.0, -vW.y);
  vec3 col = vec3(0.50, 0.85, 0.90)*exp(-KD*depth*0.6)*(0.7 + hl); col = applyWater(col, vW); gl_FragColor = vec4(tone(col)*a, a); }`;

let scene, camera, dome, surf, shafts, motes, fishMesh, sprayPts, bubPts, whales3 = [], W = 0, H = 0, dpr = 1;
const V3 = (a, b, c) => new T3.Vector3(a, b, c);
function mat(vs, fs, extra, opt) { return new T3.ShaderMaterial(Object.assign({ vertexShader: vs, fragmentShader: fs, uniforms: Object.assign({}, GLOB, extra || {}) }, opt || {})); }

function build() {
  GLOB.uCam.value = V3(0, -10, 0); GLOB.uSun.value = V3(SUN[0], SUN[1], SUN[2]); GLOB.uSunR.value = V3(SUNR[0], SUNR[1], SUNR[2]);
  scene = new T3.Scene(); camera = new T3.PerspectiveCamera(46, 1.6, 0.25, 6000);
  renderer.setClearColor(0x06223a, 1); renderer.sortObjects = true;
  dome = new T3.Mesh(new T3.SphereGeometry(2000, 32, 20), mat(DOME_VS, DOME_FS, null, { side: T3.BackSide, depthWrite: false, depthTest: false })); dome.renderOrder = -10; dome.frustumCulled = false; scene.add(dome);
  /* surface: rings that widen with distance, carried under the camera */
  { const pos = [0, 0, 0], idx = [], NR = 66, NA = small ? 72 : 108; for (let k = 0; k < NR; k++) { const r = 0.5 * Math.pow(1.142, k); for (let j = 0; j < NA; j++) { const a = j / NA * TAU; pos.push(r * Math.cos(a), 0, r * Math.sin(a)); } }
    for (let j = 0; j < NA; j++) idx.push(0, 1 + (j + 1) % NA, 1 + j);
    for (let k = 0; k < NR - 1; k++) for (let j = 0; j < NA; j++) { const a = 1 + k * NA + j, b = 1 + k * NA + (j + 1) % NA, c = a + NA, d = b + NA; idx.push(a, b, c, b, d, c); }
    const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.Float32BufferAttribute(pos, 3)); g.setIndex(idx);
    const foamU = [], foamK = []; for (let i = 0; i < 8; i++) { foamU.push(new T3.Vector4()); foamK.push(new T3.Vector4()); }
    surf = new T3.Mesh(g, mat(SURF_VS, SURF_FS, { uCentre: { value: new T3.Vector2() }, uFoam: { value: foamU }, uFoamK: { value: foamK } }, { side: T3.DoubleSide, transparent: true, premultipliedAlpha: true, depthWrite: true })); surf.frustumCulled = false; scene.add(surf); }
  /* shafts of sunlight */
  { const base = [], uv = [], idx = [], N = 6, CELLW = 15, R = M.rng32(99); let n = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) { const x = (i + 0.2 + 0.6 * R()) * CELLW, z = (j + 0.2 + 0.6 * R()) * CELLW, sd = R();
      for (const q of [[-1, 0], [1, 0], [-1, 1], [1, 1]]) { base.push(x, z, sd); uv.push(q[0], q[1]); } idx.push(n, n + 1, n + 2, n + 1, n + 3, n + 2); n += 4; }
    const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.Float32BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('aBase', new T3.Float32BufferAttribute(base, 3)); g.setAttribute('aUV', new T3.Float32BufferAttribute(uv, 2)); g.setIndex(idx);
    shafts = new T3.Mesh(g, mat(SHAFT_VS, SHAFT_FS, { uWrap: { value: N * CELLW } }, { side: T3.DoubleSide, transparent: true, blending: T3.AdditiveBlending, depthWrite: false })); shafts.frustumCulled = false; shafts.renderOrder = 3; scene.add(shafts); }
  /* motes */
  { const N = small ? 700 : 1400, R = M.rng32(5), pos = new Float32Array(N * 3), sd = new Float32Array(N); for (let i = 0; i < N; i++) { pos[i * 3] = R(); pos[i * 3 + 1] = R(); pos[i * 3 + 2] = R(); sd[i] = R(); }
    const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.BufferAttribute(pos, 3)); g.setAttribute('aSeed', new T3.BufferAttribute(sd, 1));
    motes = new T3.Points(g, mat(MOTE_VS, MOTE_FS, { uBox: { value: 34 }, uPx: { value: 800 } }, { transparent: true, blending: T3.AdditiveBlending, depthWrite: false })); motes.frustumCulled = false; motes.renderOrder = 4; scene.add(motes); }
  /* whales */
  const wg = buildWhale(), fg = buildFlipper();
  sim.whales.forEach((w, i) => {
    const U = () => ({ uPh: { value: 0 }, uAmp: { value: 0 }, uBend: { value: 0 }, uArch: { value: 0 }, uGape: { value: 0 }, uGulp: { value: 0 }, uFlex: { value: 1 }, uEye: { value: V3(wg.eye[0], wg.eye[1], Math.abs(wg.eye[2])) }, uBh: { value: V3(wg.bh[0], wg.bh[1], 0) }, uSeed: { value: 1.7 + i * 3.1 } });
    const m = mat(WHALE_VS, WHALE_FS, U(), { side: T3.DoubleSide }), body = new T3.Mesh(wg.geo, m); body.frustumCulled = false; body.rotation.order = 'YZX';
    const fu = U(); fu.uFlex.value = 0; const fm = mat(WHALE_VS, WHALE_FS, fu, { side: T3.DoubleSide }), fins = [];
    for (const sg of [1, -1]) { const f = new T3.Mesh(fg, fm); f.frustumCulled = false; f.rotation.order = 'YXZ'; f.position.set(0.4 - 0.305, -0.05, 0.104 * sg); f.scale.z = sg; body.add(f); fins.push(f); }
    scene.add(body); whales3.push({ body, m, fm, fins });
  });
  /* fish */
  { const rings = [[0.5, 0, 0], [0.42, 0.045, 0.022], [0.28, 0.085, 0.036], [0.08, 0.10, 0.042], [-0.14, 0.088, 0.036], [-0.32, 0.05, 0.02], [-0.42, 0.022, 0.008]], NA = 6, pos = [], idx = [];
    rings.forEach(r => { for (let j = 0; j < NA; j++) { const a = j / NA * TAU; pos.push(r[0], r[1] * Math.cos(a), r[2] * Math.sin(a)); } });
    for (let i = 0; i < rings.length - 1; i++) for (let j = 0; j < NA; j++) { const a = i * NA + j, b = i * NA + (j + 1) % NA, c = a + NA, d = b + NA; idx.push(a, c, b, b, c, d); }
    const t0 = pos.length / 3; pos.push(-0.42, 0, 0, -0.56, 0.10, 0, -0.50, 0, 0, -0.56, -0.10, 0); idx.push(t0, t0 + 1, t0 + 2, t0, t0 + 2, t0 + 3);
    const d0 = pos.length / 3; pos.push(0.05, 0.095, 0, -0.02, 0.15, 0, -0.10, 0.09, 0); idx.push(d0, d0 + 1, d0 + 2);
    const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.Float32BufferAttribute(pos, 3)); g.setIndex(idx); g.computeVertexNormals();
    const ph = new Float32Array(sim.NF), R = M.rng32(31); for (let i = 0; i < sim.NF; i++) ph[i] = R(); g.setAttribute('aPh', new T3.InstancedBufferAttribute(ph, 1));
    fishMesh = new T3.InstancedMesh(g, mat(FISH_VS, FISH_FS, null, { side: T3.DoubleSide }), sim.NF); fishMesh.frustumCulled = false; fishMesh.userData.size = Float32Array.from({ length: sim.NF }, () => 0.27 + 0.1 * R()); scene.add(fishMesh); }
  /* spray and bubbles */
  const pts = (n, vs, fs, order) => { const g = new T3.BufferGeometry(); g.setAttribute('position', new T3.BufferAttribute(new Float32Array(n * 3), 3)); g.setAttribute('aSz', new T3.BufferAttribute(new Float32Array(n), 1)); g.setAttribute('aA', new T3.BufferAttribute(new Float32Array(n), 1));
    const p = new T3.Points(g, mat(vs, fs, { uPx: { value: 800 } }, { transparent: true, premultipliedAlpha: true, depthWrite: false })); p.frustumCulled = false; p.renderOrder = order; scene.add(p); return p; };
  sprayPts = pts(M.MAXSPRAY, SPRAY_VS, SPRAY_FS, 7); bubPts = pts(M.MAXBUB, BUB_VS, BUB_FS, 4);
}

function resize(force) {
  const r = cv.getBoundingClientRect(), d = TEST ? (view.dpr || 1) : Math.min(small ? 1.5 : 2, devicePixelRatio || 1) * quality, w = Math.max(2, Math.round(r.width)), h = Math.max(2, Math.round(r.height));
  if (!force && w === W && h === H && d === dpr) return; W = w; H = h; dpr = d; renderer.setPixelRatio(d); renderer.setSize(w, h, false); camera.aspect = w / h;
}
let quality = 1;

/* ---------- camera ---------- */
const CAMS = {
  under: { el: -0.16, az: 0.9, dist: 27, rel: false, tgt: 'mother', side: -1 },
  follow: { el: 0.10, az: 2.45, dist: 20, rel: true, tgt: 'mother', side: -1 },
  surface: { el: 0.085, az: 0.6, dist: 29, rel: false, tgt: 'top', side: 1 },
  above: { el: 1.02, az: 0.3, dist: 62, rel: false, tgt: 'top', side: 1 },
  school: { el: -0.10, az: 2.2, dist: 9, rel: false, tgt: 'school', side: -1 }
};
function netBlend() { const N = sim.net; if (!N) return 0; let k = sstep(0, 240, sim.frame - N.f0); if (N.phase === 'done') { let f = sim.frame; for (let i = sim.log.length - 1; i >= 0; i--) if (sim.log[i].type === 'net_done') { f = sim.log[i].frame; break; } k *= 1 - sstep(240, 600, sim.frame - f); } return k; }
function placeCamera(v) {
  const c = CAMS[v.cam] || CAMS.under, A = sim.anchor, m = sim.mother, who = v.target || c.tgt; let tx, ty, tz;
  if (who === 'school') { tx = sim.school.x; ty = sim.school.y; tz = sim.school.z; }
  else if (who === 'singer') { tx = sim.male.x; ty = sim.male.y; tz = sim.male.z; }
  else if (who === 'calf') { tx = sim.calf.x; ty = sim.calf.y; tz = sim.calf.z; }
  else if (who === 'top') { tx = A.x; tz = A.z; ty = 1.0 + Math.max(0, A.y + 1.5) * 0.7; }
  else { tx = A.x; ty = A.y; tz = A.z; }
  if (c.side > 0 && sim.net) { const k = netBlend(); tx = lerp(tx, sim.net.cx, k); tz = lerp(tz, sim.net.cz, k); }
  if (c.side > 0 && who !== 'top') ty = 1.0;
  const yaw = c.az + (v.yaw || 0) * Math.PI / 180 + (c.rel ? A.yaw : 0), el = clamp(c.el + (v.pitch || 0) * Math.PI / 180, c.side < 0 ? -1.3 : 0.02, c.side < 0 ? 1.0 : 1.5);
  const asp = W / Math.max(1, H), dist = (v.dist || (v.cam === 'above' ? lerp(c.dist, 42, netBlend()) : c.dist)) / (v.zoom || 1) * (asp < 0.95 ? 1.18 : 1);
  let px = tx + dist * Math.cos(el) * Math.cos(yaw), py = ty + dist * Math.sin(el), pz = tz + dist * Math.cos(el) * Math.sin(yaw);
  if (c.side < 0) py = Math.min(py, -0.9); else py = Math.max(py, 0.9);
  return { px, py, pz, tx, ty, tz, fov: asp < 0.95 ? 58 : 46, side: c.side };
}
let camS = null;
function setCamera(v, smooth) {
  const c = placeCamera(v);
  if (smooth && camS && camS.side === c.side) { const k = 0.07; for (const q of ['px', 'py', 'pz', 'tx', 'ty', 'tz']) camS[q] = lerp(camS[q], c[q], k); camS.fov = c.fov; if (c.side < 0) camS.py = Math.min(camS.py, -0.9); else camS.py = Math.max(camS.py, 0.9); }
  else camS = c;
  camera.position.set(camS.px, camS.py, camS.pz); camera.fov = camS.fov; camera.up.set(0, 1, 0); camera.lookAt(camS.tx, camS.ty, camS.tz); camera.updateProjectionMatrix(); camera.updateMatrixWorld();
  camInfo = { cam: v.cam, x: camS.px, y: camS.py, z: camS.pz, tx: camS.tx, ty: camS.ty, tz: camS.tz, fov: camS.fov };
}

/* ---------- one picture ---------- */
const m4 = T3 ? new T3.Matrix4() : null;
function render(smooth) {
  resize(false);
  setCamera(view, smooth);
  const under = camera.position.y < 0;
  GLOB.uCam.value.copy(camera.position); GLOB.uTime.value = sim.frame / 60; GLOB.uUnder.value = under ? 1 : 0;
  const px = H * dpr / (2 * Math.tan(camera.fov * Math.PI / 360));
  shafts.visible = under; motes.visible = under; surf.renderOrder = under ? 1 : 5; surf.material.depthWrite = true;
  motes.material.uniforms.uPx.value = px; sprayPts.material.uniforms.uPx.value = px; bubPts.material.uniforms.uPx.value = px; sprayPts.visible = !under;
  surf.material.uniforms.uCentre.value.set(camera.position.x, camera.position.z);
  const fu = surf.material.uniforms.uFoam.value, fk = surf.material.uniforms.uFoamK.value;
  for (let i = 0; i < 8; i++) { const f = sim.foam[i]; if (f) { fu[i].set(f.x, f.z, f.r, f.s); fk[i].set(f.ring, sstep(0, 0.6, f.age) * (1 - sstep(f.life * 0.55, f.life, f.age)), 0, 0); } else { fu[i].set(0, 0, 0, 0); fk[i].set(0, 0, 0, 0); } }
  sim.whales.forEach((w, i) => { const o = whales3[i], u = o.m.uniforms;
    o.body.position.set(w.x, w.y, w.z); o.body.rotation.set(w.roll, -w.yaw, w.pitch); o.body.scale.setScalar(w.L);
    u.uPh.value = w.ph; u.uAmp.value = w.amp; u.uBend.value = w.bend; u.uArch.value = w.arch; u.uGape.value = w.gape; u.uGulp.value = w.gulp; o.fm.uniforms.uGulp.value = 0;
    const sc = 0.05 * Math.sin(w.ph * 0.5), sweep = lerp(0.72, 0.08, w.pec), droop = lerp(0.46, 0.10, w.pec);
    o.fins[0].rotation.set(droop + sc, -sweep, -0.25 + 0.2 * w.pec + w.yr * 0.8); o.fins[1].rotation.set(-droop + sc, sweep, -0.25 + 0.2 * w.pec - w.yr * 0.8); });
  /* fish */
  { const F = sim.fish, a = fishMesh.instanceMatrix.array, sz = fishMesh.userData.size;
    for (let i = 0; i < sim.NF; i++) { const o = i * 16; if (F.gone[i]) { for (let k = 0; k < 16; k++) a[o + k] = 0; a[o + 15] = 1; a[o + 13] = -9999; continue; }
      let fx = F.vx[i], fy = F.vy[i], fz = F.vz[i]; const l = Math.hypot(fx, fy, fz) || 1; fx /= l; fy /= l; fz /= l;
      let rx = -fz, rz = fx; const rl = Math.hypot(rx, rz) || 1; rx /= rl; rz /= rl; const ux = -rz * fy, uy = rz * fx - rx * fz, uz = rx * fy, s = sz[i];
      a[o] = fx * s; a[o + 1] = fy * s; a[o + 2] = fz * s; a[o + 3] = 0; a[o + 4] = ux * s; a[o + 5] = uy * s; a[o + 6] = uz * s; a[o + 7] = 0; a[o + 8] = rx * s; a[o + 9] = 0; a[o + 10] = rz * s; a[o + 11] = 0; a[o + 12] = F.x[i]; a[o + 13] = F.y[i]; a[o + 14] = F.z[i]; a[o + 15] = 1; }
    fishMesh.instanceMatrix.needsUpdate = true; }
  /* spray and bubbles */
  { const P = sim.spray, g = sprayPts.geometry, p = g.attributes.position.array, s = g.attributes.aSz.array, al = g.attributes.aA.array;
    for (let i = 0; i < P.n; i++) { if (P.life[i] <= 0) { al[i] = 0; continue; } p[i * 3] = P.x[i]; p[i * 3 + 1] = P.y[i]; p[i * 3 + 2] = P.z[i]; const k = P.life[i] / P.max[i];
      if (P.kind[i] === 1) { s[i] = P.size[i] * 2.0; al[i] = 0.09 * Math.min(1, k * 2.2) * Math.min(1, (1 - k) * 12 + 0.25); } else { s[i] = P.size[i]; al[i] = 0.8 * Math.min(1, k * 6); } }
    g.attributes.position.needsUpdate = true; g.attributes.aSz.needsUpdate = true; g.attributes.aA.needsUpdate = true; }
  { const B = sim.bubbles, g = bubPts.geometry, p = g.attributes.position.array, s = g.attributes.aSz.array, al = g.attributes.aA.array;
    for (let i = 0; i < B.n; i++) { if (B.life[i] <= 0) { al[i] = 0; continue; } p[i * 3] = B.x[i]; p[i * 3 + 1] = B.y[i]; p[i * 3 + 2] = B.z[i]; s[i] = B.size[i] * (under ? 1.6 : 2.6); al[i] = under ? 0.6 : 0.22; }
    g.attributes.position.needsUpdate = true; g.attributes.aSz.needsUpdate = true; g.attributes.aA.needsUpdate = true; }
  renderer.render(scene, camera);
}

/* ---------- sound, made here, only after the button ---------- */
const AU = { ctx: null, on: false };
function audioStart() {
  if (AU.ctx) return; const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return; const ctx = new AC(); AU.ctx = ctx;
  const R = M.rng32(11);
  AU.master = ctx.createGain(); AU.master.gain.value = 0; const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -16; comp.ratio.value = 4; AU.master.connect(comp); comp.connect(ctx.destination);
  /* the sea's echo: a long, dark, decaying wash */
  const len = Math.floor(ctx.sampleRate * 5), ir = ctx.createBuffer(2, len, ctx.sampleRate);
  for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); let lp = 0; for (let i = 0; i < len; i++) { const t = i / ctx.sampleRate, k = 0.25 + 0.6 * Math.exp(-t * 1.2); lp += ((R() * 2 - 1) - lp) * k; d[i] = lp * Math.exp(-t * 1.05) * (t < 0.03 ? t / 0.03 : 1) * (1 + 0.7 * Math.exp(-Math.pow((t - 0.31 - c * 0.04) / 0.02, 2)) + 0.5 * Math.exp(-Math.pow((t - 0.74 + c * 0.05) / 0.03, 2))); } }
  AU.rev = ctx.createConvolver(); AU.rev.buffer = ir; AU.revG = ctx.createGain(); AU.revG.gain.value = 0.55; AU.rev.connect(AU.revG); AU.revG.connect(AU.master);
  AU.song = ctx.createGain(); AU.songLP = ctx.createBiquadFilter(); AU.songLP.type = 'lowpass'; AU.songLP.frequency.value = 9000; AU.song.connect(AU.songLP); AU.songLP.connect(AU.master); AU.songLP.connect(AU.rev);
  AU.air = ctx.createGain(); AU.air.connect(AU.master);
  /* noise beds */
  const nb = ctx.createBuffer(1, ctx.sampleRate * 4, ctx.sampleRate), nd = nb.getChannelData(0); let b = 0; for (let i = 0; i < nd.length; i++) { b = (b + 0.02 * (R() * 2 - 1)) / 1.02; nd[i] = b * 3.5; }
  const wb = ctx.createBuffer(1, ctx.sampleRate * 2, ctx.sampleRate), wd = wb.getChannelData(0); for (let i = 0; i < wd.length; i++) wd[i] = R() * 2 - 1; AU.white = wb;
  const bed = ctx.createBufferSource(); bed.buffer = nb; bed.loop = true; const bf = ctx.createBiquadFilter(); bf.type = 'lowpass'; bf.frequency.value = 260; AU.bedG = ctx.createGain(); AU.bedG.gain.value = 0.5; bed.connect(bf); bf.connect(AU.bedG); AU.bedG.connect(AU.master); bed.start();
  const wind = ctx.createBufferSource(); wind.buffer = wb; wind.loop = true; const wf = ctx.createBiquadFilter(); wf.type = 'bandpass'; wf.frequency.value = 700; wf.Q.value = 0.4; AU.windG = ctx.createGain(); AU.windG.gain.value = 0; wind.connect(wf); wf.connect(AU.windG); AU.windG.connect(AU.master); wind.start();
  /* timbres */
  const wave = (n, f) => { const re = new Float32Array(n + 1), im = new Float32Array(n + 1); for (let k = 1; k <= n; k++) im[k] = f(k); return ctx.createPeriodicWave(re, im); };
  AU.waves = { rich: wave(40, k => 1 / Math.pow(k, 0.9) * (k % 2 ? 1 : 0.75)), reed: wave(24, k => 1 / Math.pow(k, 1.25)), pure: wave(6, k => 1 / Math.pow(k, 2.6)) };
}
function playUnit(u, when) {
  const ctx = AU.ctx; if (!ctx || !AU.on) return; const t0 = Math.max(ctx.currentTime + 0.02, when || 0), d = u.dur, c = u.contour;
  const out = ctx.createGain(); out.gain.setValueAtTime(0.0001, t0); const A = Math.min(0.28, d * 0.22), Rl = Math.min(0.5, d * 0.35), pk = 0.5 * u.loud;
  out.gain.exponentialRampToValueAtTime(pk * 0.8, t0 + A); out.gain.linearRampToValueAtTime(pk, t0 + d * 0.55); out.gain.setValueAtTime(pk, t0 + d - Rl); out.gain.exponentialRampToValueAtTime(0.0001, t0 + d + 0.12);
  const mix = ctx.createGain(); mix.gain.value = 1;
  const voice = (ratio, gain, det) => { const o = ctx.createOscillator(); o.setPeriodicWave(AU.waves[u.timbre] || AU.waves.reed); o.frequency.setValueAtTime(c[0][1] * ratio, t0); for (let i = 1; i < c.length; i++) o.frequency.exponentialRampToValueAtTime(c[i][1] * ratio, t0 + c[i][0]); o.detune.value = det;
    const g = ctx.createGain(); g.gain.value = gain; o.connect(g); g.connect(mix); o.start(t0); o.stop(t0 + d + 0.2);
    if (u.vib) { const l = ctx.createOscillator(), lg = ctx.createGain(); l.frequency.setValueAtTime(3.5, t0); l.frequency.linearRampToValueAtTime(6.5, t0 + d); lg.gain.setValueAtTime(0, t0); lg.gain.linearRampToValueAtTime(c[0][1] * ratio * u.vib, t0 + d * 0.5); l.connect(lg); lg.connect(o.frequency); l.start(t0); l.stop(t0 + d + 0.2); }
    return o; };
  voice(1, 0.8, 0); voice(1, 0.3, 9); if (c[0][1] < 400) voice(0.5, 0.22, -4);
  /* breath in the voice */
  { const n = ctx.createBufferSource(); n.buffer = AU.white; n.loop = true; const f = ctx.createBiquadFilter(); f.type = 'bandpass'; f.Q.value = 3; f.frequency.setValueAtTime(c[0][1] * 2, t0); for (let i = 1; i < c.length; i++) f.frequency.exponentialRampToValueAtTime(c[i][1] * 2, t0 + c[i][0]); const g = ctx.createGain(); g.gain.value = 0.10; n.connect(f); f.connect(g); g.connect(mix); n.start(t0); n.stop(t0 + d + 0.2); }
  /* a pulsed growl */
  let src = mix; if (u.pulse) { const am = ctx.createGain(); am.gain.value = 0.55; const l = ctx.createOscillator(); l.type = 'sawtooth'; l.frequency.value = u.pulse; const lg = ctx.createGain(); lg.gain.value = 0.45; l.connect(lg); lg.connect(am.gain); mix.connect(am); l.start(t0); l.stop(t0 + d + 0.2); src = am; }
  /* two resonances and a dark direct path, like a large body */
  const f1 = ctx.createBiquadFilter(); f1.type = 'bandpass'; f1.frequency.value = u.form[0]; f1.Q.value = 2.2; const f2 = ctx.createBiquadFilter(); f2.type = 'bandpass'; f2.frequency.value = u.form[1]; f2.Q.value = 3.5;
  const lp = ctx.createBiquadFilter(); lp.type = 'lowpass'; lp.frequency.value = Math.max(700, c[0][1] * 3); const g1 = ctx.createGain(); g1.gain.value = 1.1; const g2 = ctx.createGain(); g2.gain.value = 0.6; const g3 = ctx.createGain(); g3.gain.value = 0.55;
  src.connect(f1); f1.connect(g1); g1.connect(out); src.connect(f2); f2.connect(g2); g2.connect(out); src.connect(lp); lp.connect(g3); g3.connect(out);
  out.connect(AU.song);
}
function noiseBurst(kind) {
  const ctx = AU.ctx; if (!ctx || !AU.on) return; const t0 = ctx.currentTime + 0.02, n = ctx.createBufferSource(); n.buffer = AU.white; n.loop = true; const f = ctx.createBiquadFilter(), g = ctx.createGain(); n.connect(f); f.connect(g); g.connect(AU.air); g.gain.setValueAtTime(0.0001, t0);
  if (kind === 'blow') { f.type = 'bandpass'; f.Q.value = 0.7; f.frequency.setValueAtTime(1500, t0); f.frequency.exponentialRampToValueAtTime(600, t0 + 0.9); g.gain.exponentialRampToValueAtTime(0.5, t0 + 0.06); g.gain.exponentialRampToValueAtTime(0.05, t0 + 0.95); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 1.1);
    g.gain.setValueAtTime(0.0001, t0 + 1.25); g.gain.exponentialRampToValueAtTime(0.12, t0 + 1.5); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 2.1); n.start(t0); n.stop(t0 + 2.2); }
  else { const big = kind === 'splash' ? 1 : 0.6; f.type = 'lowpass'; f.frequency.setValueAtTime(5000, t0); f.frequency.exponentialRampToValueAtTime(350, t0 + 2.6 * big); g.gain.exponentialRampToValueAtTime(0.9 * big, t0 + 0.04); g.gain.exponentialRampToValueAtTime(0.0001, t0 + 3.2 * big); n.start(t0); n.stop(t0 + 3.4);
    const o = ctx.createOscillator(), og = ctx.createGain(); o.frequency.setValueAtTime(70, t0); o.frequency.exponentialRampToValueAtTime(32, t0 + 0.6); og.gain.setValueAtTime(0.8 * big, t0); og.gain.exponentialRampToValueAtTime(0.0001, t0 + 0.9); o.connect(og); og.connect(AU.master); o.start(t0); o.stop(t0 + 1); }
}
function audioFrame() {
  const ctx = AU.ctx; if (!ctx) return; const t = ctx.currentTime, under = camera && camera.position.y < 0, tc = 0.25;
  AU.master.gain.setTargetAtTime(AU.on && running ? 0.9 : 0, t, 0.15);
  let near = 1; if (camera) { const d = Math.hypot(camera.position.x - sim.male.x, camera.position.y - sim.male.y, camera.position.z - sim.male.z); near = 1 / (1 + d / 90); }
  AU.song.gain.setTargetAtTime(under ? 0.55 + 0.45 * near : 0.10, t, tc); AU.songLP.frequency.setTargetAtTime(under ? 9000 : 500, t, tc);
  AU.bedG.gain.setTargetAtTime(under ? 0.5 : 0.08, t, tc); AU.windG.gain.setTargetAtTime(under ? 0 : 0.045, t, tc); AU.air.gain.setTargetAtTime(under ? 0.18 : 0.8, t, tc);
}

/* ---------- live page ---------- */
let running = false, userPaused = false, onscreen = true, last = 0, acc = 0, slow = 0, lastRead = 0, sayT = 0;
const say = (s, hold) => { const e = $('say'); if (e) { e.textContent = s; sayT = sim.t + (hold || 9); } };
function onEvents() {
  for (const e of sim.ev) {
    if (e.type === 'unit') { playUnit(M.UNIT[e.unit]); continue; }
    if (e.type === 'blow') { if (e.whale === 'mother') { noiseBurst('blow'); if (e.n === 1) say('She comes up to breathe. The blow is warm breath and spray from two blowholes, about three metres high.'); } else if (e.whale === 'calf' && sim.t > sayT) say('The calf comes up more often than its mother.', 6); }
    else if (e.type === 'breach_start') say('She dives, then turns upward and swims hard.');
    else if (e.type === 'exit') { noiseBurst('lunge'); say('A breach. Most of her body leaves the water.'); }
    else if (e.type === 'splash') noiseBurst('splash');
    else if (e.type === 'net_start') say('Two whales go under the herring.', 14);
    else if (e.type === 'net_spin') say('They circle, letting air out of their blowholes. The rising bubbles pen the fish.', 22);
    else if (e.type === 'net_lunge') say('They come up through the middle with mouths open.');
    else if (e.type === 'lunge') noiseBurst('lunge');
    else if (e.type === 'net_done') say('The throat closes and the water is pressed out through the baleen. ' + e.taken + ' fish taken.', 12);
  }
  if (sim.mother.state === 'sound' && sim.mother.st < 0.05) say('Flukes up. She is going down for a longer dive.', 8);
}
function readouts() {
  const m = sim.mother, d = sim.dives, u = sim.unit;
  $('rDepth').innerHTML = (m.y > 0 ? 0 : -m.y).toFixed(1) + ' m';
  $('rBreath').innerHTML = Math.round(m.tb) + ' s<small>count ' + Math.min(1, m.C).toFixed(2) + ' of 1</small>';
  $('rState').innerHTML = (M.WORDS[m.state] || m.state) + (m.nb ? '<small>breath ' + m.nb + ' of ' + m.series + '</small>' : '');
  if (d.length) { const mean = d.reduce((a, b) => a + b, 0) / d.length, cv = d.length > 1 ? Math.sqrt(d.reduce((a, b) => a + (b - mean) * (b - mean), 0) / d.length) / mean : null; $('rDives').innerHTML = d.length + '<small>mean ' + Math.round(mean) + ' s from first breath to first breath' + (cv === null ? '' : ', spread (CV) ' + cv.toFixed(2)) + '</small>'; }
  else $('rDives').innerHTML = '0<small>counted from her second surfacing</small>';
  $('rSong').innerHTML = !sim.male.singing ? 'silent' : u ? 'theme ' + u.theme + '<small>phrase ' + u.phrase + ' of ' + u.of + ', ' + (sim.sounding() ? M.UNIT[u.unit].name : 'a rest') + '</small>' : '&ndash;';
  $('rSpeed').innerHTML = fmt(M.speedAt(Math.max(0, -m.y), sea.T, sea.S)) + '<small>m/s, from the two sliders</small>';
  $('bNet').disabled = !!(sim.net && sim.net.phase !== 'done'); $('bBreach').disabled = !(m.state === 'dive' || m.state === 'shallow' || m.state === 'sound' || m.state === 'rise' || m.state === 'breathe'); $('bBlow').disabled = m.state !== 'dive';
}
function frame(now) {
  if (!running) return; requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now; acc += dt; let n = 0;
  while (acc >= M.DT && n < 4) { sim.step(); onEvents(); acc -= M.DT; n++; } if (acc > M.DT) acc = 0;
  if (renderer) render(true);
  audioFrame();
  if (now - lastRead > 200) { lastRead = now; readouts(); drawStrip(); if (sea.fromChannel === false && sim.frame % 600 < 15) drawOcean(); }
  /* drop the pixel ratio a step if frames stay slow */
  if (dt > 0.034) slow++; else slow = Math.max(0, slow - 2); if (slow > 90 && quality > 0.55) { quality -= 0.15; slow = 0; resize(true); }
}
function setRun() { const want = !userPaused && onscreen && !document.hidden; if (want && !running) { running = true; last = performance.now(); acc = 0; requestAnimationFrame(frame); } else if (!want && running) { running = false; audioFrame(); } }

function settingsText() {
  const m = sim.mother, c = sim.calf, s = sim.male;
  $('settings').textContent = 'Settings. Mother ' + m.L + ' m, calf ' + c.L + ' m, singer ' + s.L + ' m; flippers 0.32 of body length; ' + sim.NF + ' herring of about 0.3 m. A breath is due when the count reaches 1; the count rises by (0.5 + 0.5 × (speed ÷ 1.8 m/s)²) ÷ ' + m.Tdive + ' each second for the mother, ÷ ' + c.Tdive + ' for the calf, ÷ ' + s.Tdive + ' for the singer. '
    + m.series + ' breaths at each surfacing. Breach count: rises each second by a number drawn evenly between 0.3 and 1.7, divided by 260. Breach run to 8.6 m/s at 64° from level. Net 9 m across its radius, blown from 14 m up to 7.5 m over 20 s. Bubbles rise at about 1 m/s, a few times faster than small bubbles do. '
    + 'Song: ' + M.UNITS.length + ' units, ' + M.THEMES.length + ' themes, ' + M.SONG.duration.toFixed(0) + ' s. Sea: mixed layer 50 m, then temperature falls towards 2 °C with a scale of 400 m; bottom 4,500 m. Light lost per metre of water: red 0.17, green 0.042, blue 0.024, set by eye near clear-ocean values.';
}

function initLive() {
  document.querySelectorAll('[data-cam]').forEach(b => b.addEventListener('click', () => { view = Object.assign({}, DEF, { cam: b.dataset.cam }); camS = null; document.querySelectorAll('[data-cam]').forEach(q => q.setAttribute('aria-pressed', q === b ? 'true' : 'false'));
    if (b.dataset.cam === 'school') say('Each herring steers by the few fish nearest to it. No fish leads.', 8); if (!running && renderer) render(false); }));
  $('bBlow').addEventListener('click', () => { sim.trigger('blow'); say('She turns for the surface.', 6); });
  $('bBreach').addEventListener('click', () => { sim.trigger('breach'); });
  $('bNet').addEventListener('click', () => { if (!sim.trigger('bubblenet')) say('They will start the net when both are free.', 6); });
  $('snd').addEventListener('click', () => { audioStart(); if (!AU.ctx) return; AU.on = !AU.on; if (AU.on && AU.ctx.state === 'suspended') AU.ctx.resume(); $('snd').setAttribute('aria-pressed', AU.on ? 'true' : 'false'); audioFrame();
    if (AU.on) say('The song comes from the male hanging head down. Under water it carries; in air it is almost lost.', 8); });
  $('pause').addEventListener('click', () => { userPaused = !userPaused; $('pause').setAttribute('aria-pressed', userPaused ? 'true' : 'false'); $('pause').textContent = userPaused ? 'Play' : 'Pause'; setRun(); });
  const tO = $('tempOut'), sO = $('saltOut');
  $('temp').addEventListener('input', e => { sea.T = +e.target.value; tO.textContent = sea.T + ' °C'; drawOcean(); readouts(); });
  $('salt').addEventListener('input', e => { sea.S = +e.target.value; sO.textContent = sea.S.toFixed(1) + ' ‰'; drawOcean(); readouts(); });
  $('src').addEventListener('click', () => { sea.fromChannel = !sea.fromChannel; $('src').setAttribute('aria-pressed', sea.fromChannel ? 'true' : 'false'); $('src').textContent = sea.fromChannel ? 'Sound from the singer' : 'Sound from the channel'; drawOcean(); });
  /* look round: drag sideways and up, pinch or wheel with ctrl to zoom */
  let drag = null; const ptrs = new Map(); let pinch = 0;
  cv.addEventListener('pointerdown', e => { ptrs.set(e.pointerId, [e.clientX, e.clientY]); if (ptrs.size === 1) drag = { x: e.clientX, y: e.clientY, yaw: view.yaw, pitch: view.pitch, moved: false }; else if (ptrs.size === 2) { const p = [...ptrs.values()]; pinch = Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) / view.zoom; drag = null; } });
  cv.addEventListener('pointermove', e => { if (!ptrs.has(e.pointerId)) return; ptrs.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptrs.size === 2 && pinch) { const p = [...ptrs.values()]; view.zoom = clamp(Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) / pinch, 0.5, 3); e.preventDefault(); }
    else if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (!drag.moved && Math.abs(dx) < 6) return; if (!drag.moved) { drag.moved = true; try { cv.setPointerCapture(e.pointerId); } catch (_) {} }
      view.yaw = drag.yaw + dx * 0.3; view.pitch = clamp(drag.pitch + dy * 0.2, -60, 60); }
    if (!running && renderer) render(false); });
  const up = e => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; if (!ptrs.size) drag = null; };
  cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up);
  cv.addEventListener('wheel', e => { if (!e.ctrlKey && !e.metaKey) return; e.preventDefault(); view.zoom = clamp(view.zoom * Math.exp(-e.deltaY * 0.002), 0.5, 3); if (!running && renderer) render(false); }, { passive: false });
  cv.tabIndex = 0; cv.addEventListener('keydown', e => { const k = e.key; if (k === 'ArrowLeft') view.yaw -= 6; else if (k === 'ArrowRight') view.yaw += 6; else if (k === 'ArrowUp') view.pitch = clamp(view.pitch + 4, -60, 60); else if (k === 'ArrowDown') view.pitch = clamp(view.pitch - 4, -60, 60); else if (k === '+' || k === '=') view.zoom = clamp(view.zoom * 1.1, 0.5, 3); else if (k === '-') view.zoom = clamp(view.zoom / 1.1, 0.5, 3); else return; e.preventDefault(); if (!running && renderer) render(false); });
  addEventListener('resize', () => { if (renderer) { resize(true); if (!running) render(false); } drawOcean(); drawStrip(); });
  document.addEventListener('visibilitychange', setRun);
  if ('IntersectionObserver' in window) new IntersectionObserver(es => { onscreen = es[0].isIntersecting; setRun(); }).observe(cv);
  if (renderer) cv.addEventListener('webglcontextlost', e => { e.preventDefault(); userPaused = true; setRun(); noGL(); });
  settingsText();
  /* start a little way in, so the whales are already going about their day */
  for (let i = 0; i < 240; i++) sim.step();
  say('She is ' + Math.round(-sim.mother.y) + ' metres down, with her calf at her side. Further off a male hangs head down and sings.', 12);
  readouts(); drawOcean(); drawStrip();
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => { drawOcean(); drawStrip(); });
  if (renderer) { resize(true); render(false); }
  if (still) { userPaused = true; $('pause').setAttribute('aria-pressed', 'true'); $('pause').textContent = 'Play'; }
  setRun();
}

/* ---------- film hook: no loop, no overlays, one deterministic picture per call ---------- */
function events(v) { const e = []; if (v.event) e.push({ event: v.event, at: Math.max(0, Math.round(v.eventAt || 0)) }); if (Array.isArray(v.events)) for (const q of v.events) if (q && q.event) e.push({ event: q.event, at: Math.max(0, Math.round(q.at === undefined ? (q.eventAt || 0) : q.at)) }); return e; }
/* For films, eventAt names the moment that shows: the blow itself, or the instant she leaves the water.
   The model is run ahead without fish (they do not move the whales) to find when to start. */
const leadCache = {};
function resolve(ev, v) {
  if (v.eventStart) return ev;
  return ev.map(e => { const want = e.event === 'blow' ? 'blow' : e.event === 'breach' ? 'exit' : null; if (!want) return e; const key = e.event + e.at + '|' + (v.seed === undefined ? 7 : v.seed); if (leadCache[key] !== undefined) return { event: e.event, at: leadCache[key] };
    const when = T => { const S = M.makeSim({ seed: v.seed === undefined ? 7 : v.seed, nFish: 0, auto: false, events: [{ event: e.event, at: T }] }); for (let f = 0; f < T + 3600; f++) { S.step(); if (f >= T) for (const q of S.ev) if (q.type === want && q.whale === 'mother') return q.frame; } return -1; };
    let T = Math.max(0, e.at - (want === 'blow' ? 1000 : 350)), best = T, err = 1e9;
    for (let i = 0; i < 8; i++) { const B = when(T); if (B < 0) break; const d = e.at - B; if (Math.abs(d) < err) { err = Math.abs(d); best = T; } if (d === 0) break; const T2 = Math.max(0, T + d); if (T2 === T) break; T = T2; }
    leadCache[key] = best; return { event: e.event, at: best }; });
}
function simFor(v, frame) {
  const ev = resolve(events(v), v), key = JSON.stringify([ev, v.seed === undefined ? 7 : v.seed, v.fish === undefined ? 420 : v.fish, !!v.auto]);
  if (key !== simKey || frame < sim.frame) { sim = M.makeSim({ seed: v.seed === undefined ? 7 : v.seed, nFish: v.fish === undefined ? 420 : v.fish, auto: !!v.auto, events: ev }); simKey = key; }
  while (sim.frame < frame) sim.step();
}
if (renderer) { try { build(); } catch (e) { console.error(e); renderer = null; noGL(); } }
if (TEST) {
  let fishN = sim.NF;
  window.__clay = {
    shot(frame, v) { view = Object.assign({}, DEF, v || {}); if (view.fish === undefined) view.fish = fishN; if (view.fish !== fishN) { view.fish = fishN; } simFor(view, Math.max(0, Math.round(frame))); if (!renderer) return false; camS = null; render(false); gl.finish(); return true; },
    /* when things happen for a given schedule, without drawing: frames of each logged event up to `until` */
    plan(v, until) { const ev = resolve(events(v || {}), v || {}), S = M.makeSim({ seed: (v && v.seed) === undefined ? 7 : v.seed, nFish: fishN, auto: !!(v && v.auto), events: ev }), out = []; const end = until || 60 * 120; for (let f = 0; f < end; f++) { S.step(); for (const e of S.ev) if (e.type !== 'unit') out.push({ frame: e.frame, type: e.type, whale: e.whale, n: e.n }); } return out; },
    state, song: M.SONG
  };
} else initLive();
})();
