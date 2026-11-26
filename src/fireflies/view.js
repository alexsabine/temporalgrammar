/* The wood, the fireflies, the torch, the rhythm, the sound and the controls. The fireflies themselves are in the model above. */
(function () {
'use strict';
const M = window.FIREFLY, $ = id => document.getElementById(id);
const canvas = $('wood');
const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, depth: true, powerPreference: 'high-performance' });
if (!gl) { canvas.style.display = 'none'; $('fail').style.display = 'block'; }
const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
const small = Math.min(innerWidth, innerHeight) < 620 || (matchMedia('(pointer: coarse)').matches && innerWidth < 900);
const NMAX = 600, N0 = 300;
const S = M.make(N0, { seed: 7 + (Date.now() % 97), nMax: NMAX });
const FAR = 160, CAMH = 1.55;

/* ─── what is measured, as the run goes: collective bursts, who joins them, and flashes per tenth of a second ─── */
const BIN = .1, NB = 600, bins = new Uint16Array(NB);
let ownIv = [], prevF = new Float64Array(NMAX).fill(-1e9), ownStart = new Float64Array(NMAX).fill(-1e9), binT = Math.floor(S.t / BIN), lastAny = -1e9, cur = null, starts = [], shares = [], ivAll = [], stamp = new Uint32Array(NMAX), stampN = 1, beat = 0, beatT = -1e9, flashesThisFrame = 0;
S.logOn = false;
S.onFlash = (i, t) => {
  const b = Math.floor(t / BIN); while (binT < b) { binT++; bins[binT % NB] = 0; } bins[b % NB]++;
  flashesThisFrame++;
  /* each firefly's own bursts: a flash more than 1.5 s after its last one starts a new burst */
  if (t - prevF[i] > 1.5) { if (ownStart[i] > -1e8) { ownIv.push(t - ownStart[i]); if (ownIv.length > 2000) ownIv.splice(0, 500); } ownStart[i] = t; } prevF[i] = t;
  if (t - lastAny > 1.5) { if (cur) closeBurst(); cur = { start: t, who: 0 }; stampN++; beat = 0; beatT = t; starts.push(t); if (starts.length > 1) { ivAll.push(t - starts[starts.length - 2]); if (ivAll.length > 400) ivAll.shift(); } if (starts.length > 60) starts.shift(); }
  lastAny = t; if (!cur) return;                    /* measures were reset mid-burst: wait for the next start */
  if (t - beatT > .26) { beat++; beatT = t; }
  if (stamp[i] !== stampN) { stamp[i] = stampN; cur.who++; }
};
function closeBurst() { shares.push(cur.who / S.n); if (shares.length > 30) shares.shift(); cur = null; }
function resetMeasures() { starts = []; shares = []; ivAll = []; cur = null; ownIv = []; ownStart.fill(-1e9); }

/* a lone firefly, run once with no pictures, for comparison in the chart */
const lone = (() => { const L = M.run(1, 3000, { seed: 5 }, 1 / 60); const st = M.ownStarts(L.flashes, 0, 3000).get(0) || []; return M.intervals(st); })();

/* ─── shaders ─── */
function prog(vs, fs) { const p = gl.createProgram(); for (const [t, src] of [[gl.VERTEX_SHADER, vs], [gl.FRAGMENT_SHADER, fs]]) { const s = gl.createShader(t); gl.shaderSource(s, '#version 300 es\nprecision highp float;\n' + src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) console.error(gl.getShaderInfoLog(s)); gl.attachShader(p, s); }
  gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) console.error(gl.getProgramInfoLog(p)); const u = {}; const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS); for (let i = 0; i < n; i++) { const a = gl.getActiveUniform(p, i); u[a.name] = gl.getUniformLocation(p, a.name); } return { p, u }; }

const NOISE = `
float h1(vec2 p){p=fract(p*vec2(123.34,456.21));p+=dot(p,p+45.32);return fract(p.x*p.y);}
float n2(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h1(i),h1(i+vec2(1,0)),f.x),mix(h1(i+vec2(0,1)),h1(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){float a=.5,s=0.;for(int i=0;i<5;i++){s+=a*n2(p);p=p*2.07+vec2(17.3,9.1);a*=.5;}return s;}
float fbm3(vec2 p){float a=.5,s=0.;for(int i=0;i<3;i++){s+=a*n2(p);p=p*2.11+vec2(5.3,11.7);a*=.5;}return s;}`;

const BG_V = `layout(location=0) in vec2 aP;out vec2 vUv;void main(){vUv=aP*.5+.5;gl_Position=vec4(aP,0.,1.);}`;
/* The wood is drawn as a few upright layers at set distances, each cut out by rule: trunks, a canopy with gaps, an
   understory of rhododendron, ferns close to the path. For each pixel the nearest layer that covers it decides the
   colour and the depth, so fireflies behind a trunk are hidden by it. Leaves near a flash are lit by it: the light
   buffer holds, for each part of the screen, how much yellow-green light is nearby and at roughly what distance. */
const BG_F = `in vec2 vUv;out vec4 o;
uniform vec3 uFwd,uRight,uUp;uniform vec2 uTan;uniform float uT,uPix,uTorch,uTorchX,uTorchY,uTorchHalf,uTorchHalfV,uCamH,uFar;uniform sampler2D uGlow;uniform vec2 uGlowScale;
${NOISE}
const vec3 LC=vec3(.62,1.,.20);
/* coverage of one layer at distance D; X is across, Y is height, both in metres */
float trunk(float X,float Y,float cx,float r,float lean,float seed,float aa,out float sh){
  float xx=X-cx-Y*lean;r*=1.+.6*exp(-Y*1.8);xx+=(n2(vec2(Y*1.1,seed))-.5)*r*.3;sh=clamp(xx/r,-1.,1.);return smoothstep(r+aa,r-aa,abs(xx));}
vec4 layer(vec3 d,float D,int kind,float aa){
  float s=D/d.z;float X=d.x*s,Y=uCamH+d.y*s;float pix=aa*s;
  float cover=0.,sh=0.,leafy=0.,sh2;
  if(kind==0){ /* beside the path: two trunks at the edges, leaves hanging into the top corners, ferns at the foot */
    float t1=trunk(X,Y,-2.75,.34,.02,1.,pix,sh2);if(t1>cover){cover=t1;sh=sh2;}
    float t2=trunk(X,Y,2.55,.42,-.035,2.,pix,sh2);if(t2>cover){cover=t2;sh=sh2;}
    float hang=fbm(vec2(X,Y)*1.1+3.)+.32*n2(vec2(X,Y)*9.)+(Y-3.2)*.17+(abs(X)-1.5)*.13;float lv=smoothstep(.83,.8,1.-hang+.5);
    if(lv>cover){cover=lv;leafy=1.;sh=0.;}
    float fv=fbm(vec2(X*2.4,Y*3.2)+vec2(sin(uT*.6+X)*.02,0.))+.3*n2(vec2(X,Y)*14.)+.4*smoothstep(.9,2.4,abs(X))-.2;
    float fern=smoothstep(.5+.55*Y-.015,.5+.55*Y+.015,fv);if(Y<.9&&fern>cover){cover=fern;leafy=1.;sh=0.;}
  } else if(kind==1){ /* the clearing: a few tall trees standing in it, each with a crown against the sky, and low rhododendron */
    float cw=14.,ci=floor(X/cw);
    for(int k=-1;k<=1;k++){float cc=ci+float(k);float hs=h1(vec2(cc,D));if(hs>.42)continue;
      float cx=(cc+.25+.5*h1(vec2(cc,D+3.)))*cw,r=mix(.22,.42,h1(vec2(cc,D+7.))),lean=(h1(vec2(cc,D+9.))-.5)*.04;
      float top=mix(9.,15.,h1(vec2(cc,D+5.)));
      float tt=trunk(X,Y,cx,r,lean,cc,pix,sh2)*step(Y,top);if(tt>cover){cover=tt;sh=sh2;leafy=0.;}
      vec2 q=vec2((X-cx-top*lean)/mix(3.,5.,hs*2.),(Y-top)/3.6);float cr=length(q)+(fbm(vec2(X,Y)*.45+cc)-.5)*1.1+(n2(vec2(X,Y)*3.5)-.5)*.35;
      float crown=smoothstep(1.+pix*.3,1.-pix*.3,cr);if(crown>cover){cover=crown;leafy=1.;sh=0.;}}
    float bush=smoothstep(.5,.47,fbm(vec2(X*.45,Y*.8)+D)+Y*.33-.18-.25*smoothstep(.3,.7,n2(vec2(X*.07,D))));if(bush>cover){cover=bush;leafy=1.;sh=0.;}
  } else { /* the wall of the wood across the clearing */
    float top=9.+6.*fbm3(vec2(X*.035,D))+2.2*fbm(vec2(X*.25,D));
    float cw=3.6,ci=floor(X/cw);
    for(int k=-1;k<=1;k++){float cc=ci+float(k);if(h1(vec2(cc,D))>.75)continue;float cx=(cc+.2+.6*h1(vec2(cc,D+3.)))*cw;
      float tt=trunk(X,Y,cx,mix(.2,.45,h1(vec2(cc,D+7.))),0.,cc,pix,sh2)*step(Y,top-3.);if(tt>cover){cover=tt;sh=sh2;}}float cn=smoothstep(.55,.5,fbm(vec2(X,Y)*.35+D)-.25*smoothstep(top-6.,top,Y)+.25);
    float can=step(Y,top+1.5*fbm(vec2(X*.8,D)))*smoothstep(top-8.,top-4.,Y)*cn;if(can>cover){cover=can;leafy=1.;sh=0.;}
    float under=smoothstep(.55,.5,fbm(vec2(X*.3,Y*.5)+D*.3)+(Y-3.)*.12);if(under>cover){cover=under;leafy=1.;sh=0.;}
    if(Y<top-4.&&cover<.5&&D>60.){cover=1.;leafy=1.;}
  }
  return vec4(cover,leafy,sh,Y);}
vec3 sky(vec3 d){float e=max(d.y,0.);
  vec3 c=mix(vec3(.10,.135,.18),vec3(.018,.03,.07),pow(e,.45));
  float az=atan(d.x,d.z);c+=vec3(.05,.07,.05)*exp(-e*9.)*smoothstep(.2,-.6,az);
  vec2 sp=d.xz/(d.y+.03)*260.;vec2 cell=floor(sp);float st=h1(cell);float tw=.65+.35*sin(uT*2.3+st*50.);
  vec2 f=fract(sp)-.5;c+=vec3(.8,.85,1.)*smoothstep(.9965,1.,st)*tw*smoothstep(.06,.25,d.y)*smoothstep(.25,.0,length(f))*1.3;
  return c;}
void main(){
  vec2 ndc=vUv*2.-1.;vec3 d=normalize(uFwd+ndc.x*uTan.x*uRight+ndc.y*uTan.y*uUp);
  float aa=uPix*1.2;
  /* the light from the fireflies near this part of the picture, and at what distance */
  vec4 g=texture(uGlow,vUv);float L=g.r*uGlowScale.x;float zL=g.r>1e-5?g.g/g.r*uGlowScale.y:50.;
  float bearing=atan(d.x,d.z);vec2 bq=vec2((bearing-uTorchX)/uTorchHalf,(atan(d.y,length(d.xz))-uTorchY)/uTorchHalfV);float beam=uTorch>.5?smoothstep(1.,.55,length(bq)):0.;
  vec3 tcol=uTorch>1.5?vec3(.85,.05,.02):vec3(1.,.93,.8);
  float Ds[6];Ds[0]=4.2;Ds[1]=17.;Ds[2]=29.;Ds[3]=42.;Ds[4]=56.;Ds[5]=78.;
  vec3 col=vec3(0.);float depth=uFar;bool hit=false;
  /* ground, if the ray reaches it before the layers */
  float tg=d.y<-.001?uCamH/(-d.y):1e9;float zg=tg*d.z;
  for(int k=0;k<6;k++){float D=Ds[k];if(zg<D)break;
    int kind=k==0?0:(k<4?1:2);vec4 ly=layer(d,D,kind,aa);
    if(ly.x>.5){
      float Y=ly.w;
      float bark=.75+.5*n2(vec2(ly.z*6.,Y*.7+D));vec3 alb=ly.y>.5?mix(vec3(.10,.15,.06),vec3(.17,.22,.09),n2(vec2(Y*3.,D))):mix(vec3(.13,.11,.09),vec3(.2,.17,.14),.5+.5*ly.z)*bark;
      float amb=.05*smoothstep(-1.,12.,Y)+.016;
      vec3 c=alb*amb*vec3(.55,.7,1.)*(1.-.4*abs(ly.z));if(ly.y<.5)c+=vec3(.05,.07,.1)*.25*smoothstep(.6,1.,abs(ly.z))*smoothstep(0.,8.,Y);
      float near=max(exp(-abs(D-zL)/4.),D<6.?.06:0.)*(1.-smoothstep(1.,7.,abs(Y-1.8)-1.));
      c+=alb*LC*L*near*2.4;
      c+=alb*tcol*beam*min(2.,70./(D*D+8.));
      float haze=1.-exp(-D/45.);c=mix(c,vec3(.04,.055,.08),haze*.8);
      col=c;depth=D;hit=true;break;}
  }
  if(!hit){
    if(tg<1e8){vec2 w=d.xz*tg;
      /* the meadow: grass in tufts, darker hollows, catching the sky from above */
      float lit=fbm(w*.9);float tuft=n2(w*vec2(7.,2.3))*.6+n2(w*vec2(23.,6.))*.4;
      vec3 alb=mix(vec3(.08,.10,.045),vec3(.19,.23,.10),lit)*(.55+.8*tuft);
      vec3 c=alb*.085*vec3(.55,.7,1.)*(.6+.4*smoothstep(.3,.7,lit))*(1.+.8*exp(-zg/5.));
      c+=alb*LC*L*exp(-abs(zg-zL)/5.)*2.6;
      c+=alb*tcol*beam*min(2.,70./(zg*zg+8.));
      c=mix(c,vec3(.04,.055,.08),(1.-exp(-zg/45.))*.8);col=c;depth=zg;
    } else {col=sky(d);depth=uFar;}
  }
  /* low mist in the hollow, faintly lit by the flashes and the torch */
  float mist=fbm(vec2(bearing*5.+uT*.008,d.y*7.))*exp(-abs(d.y-.01)*22.)*smoothstep(8.,40.,depth);
  col+=vec3(.05,.07,.095)*mist*.9+LC*L*.035+tcol*beam*(.09*mist+.012);
  vec2 v=vUv-.5;col*=1.-.35*dot(v,v);
  col+=(h1(gl_FragCoord.xy+fract(uT)*61.)-.5)/255.*1.4;
  o=vec4(pow(col,vec3(1./1.12)),1.);
  gl_FragDepth=depth/uFar;}`;

/* a firefly is a point. Its own light is a hard bright core and a soft halo; the light buffer gets a much wider, dim
   version so that it falls on the leaves around it. */
const FF_V = `layout(location=0) in vec4 aP;
uniform vec3 uCam,uFwd,uRight,uUp;uniform vec2 uTan;uniform float uFocal,uFar,uMode,uDpr;out float vI;out float vCore;out float vZ;
void main(){vec3 q=aP.xyz-uCam;float z=dot(q,uFwd);vI=aP.w;vZ=z;
  float x=dot(q,uRight)/uTan.x,y=dot(q,uUp)/uTan.y;gl_Position=vec4(x,y,(z/uFar*2.-1.)*z,z);
  if(aP.w<.003||z<.5){gl_PointSize=0.;gl_Position=vec4(2.,2.,2.,1.);return;}
  float px=uFocal/z;
  if(uMode<.5){float halo=max(18.*uDpr,1.1*px);float core=max(2.3*uDpr,.03*px);gl_PointSize=min(halo,128.);vCore=core/gl_PointSize;}
  else{gl_PointSize=clamp(2.2*px,4.,96.);vCore=0.;}}`;
const FF_F = `in float vI;in float vCore;in float vZ;out vec4 o;uniform float uMode,uZs;
void main(){vec2 p=gl_PointCoord*2.-1.;float r2=dot(p,p);if(r2>1.)discard;
  if(uMode<.5){float core=exp(-r2/(vCore*vCore)*1.4);float halo=exp(-r2*4.)*.16+exp(-r2*26.)*.5+exp(-r2*120.)*.6;
    vec3 c=vec3(.62,1.,.2)*halo*vI*1.3+vec3(1.,1.,.72)*core*vI*1.8;o=vec4(c,1.);}
  else{float w=exp(-r2*3.)*vI;o=vec4(w,w*vZ/uZs,0.,1.);}}`;

let bgP, ffP;
if (gl) { bgP = prog(BG_V, BG_F); ffP = prog(FF_V, FF_F); }

/* ─── buffers ─── */
let quad, ffVAO, ffBuf, ffData = new Float32Array(NMAX * 4);
const glow = { fb: null, tex: null, w: 0, h: 0, float: false };
if (gl) {
  quad = gl.createVertexArray(); gl.bindVertexArray(quad); { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0); }
  ffVAO = gl.createVertexArray(); gl.bindVertexArray(ffVAO); ffBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ffBuf); gl.bufferData(gl.ARRAY_BUFFER, ffData.byteLength, gl.DYNAMIC_DRAW); gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
  glow.float = !!gl.getExtension('EXT_color_buffer_float') || !!gl.getExtension('EXT_color_buffer_half_float');
  glow.fb = gl.createFramebuffer(); glow.tex = gl.createTexture();
}
const GLOW_K = glow.float ? 1 : 1 / 6, ZS = glow.float ? 1 : 60;   /* without float targets, store small values and scale them back */

/* ─── the camera: someone standing on the path ─── */
const cam = { pos: [0, CAMH, 0], yaw: 0, tyaw: 0, pitch: .02, tanX: .62, tanY: .35, fwd: [0, 0, 1], right: [1, 0, 0], up: [0, 1, 0], focal: 900 };
let dpr = 1, quality = 1;
function size() { const w = canvas.clientWidth, h = canvas.clientHeight; if (!w || !h) return; dpr = Math.min(devicePixelRatio || 1, small ? 1.5 : 2) * quality;
  const W = Math.round(w * dpr), H = Math.round(h * dpr); if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
  const asp = W / H; cam.tanY = asp >= 1 ? .36 : .44; cam.tanX = cam.tanY * asp; cam.focal = H / 2 / cam.tanY;
  const gw = Math.max(64, W >> 2), gh = Math.max(36, H >> 2);
  if (gl && (glow.w !== gw || glow.h !== gh)) { glow.w = gw; glow.h = gh; gl.bindTexture(gl.TEXTURE_2D, glow.tex);
    if (glow.float) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, gw, gh, 0, gl.RGBA, gl.HALF_FLOAT, null); else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gw, gh, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, glow.fb); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, glow.tex, 0); gl.bindFramebuffer(gl.FRAMEBUFFER, null); } }
function aim() { const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  cam.fwd = [sy * cp, sp, cy * cp]; cam.right = [cy, 0, -sy]; cam.up = [-sy * sp, cp, -cy * sp]; }

/* how bright a firefly is, a moment after it flashed: a quick rise and a slower fade, about a fifth of a second in all */
function lightOf(i, t) { const a = t - S.lastFlash[i]; if (a < 0 || a > .5) return 0; return a < .045 ? a / .045 : Math.exp(-(a - .045) / .085); }

function fill() { const n = S.n, t = S.t; let m = 0;
  for (let i = 0; i < n; i++) { const I = lightOf(i, t); if (I < .003) continue; ffData[m * 4] = S.x[i]; ffData[m * 4 + 1] = S.y[i]; ffData[m * 4 + 2] = S.z[i]; ffData[m * 4 + 3] = I; m++; }
  gl.bindBuffer(gl.ARRAY_BUFFER, ffBuf); gl.bufferSubData(gl.ARRAY_BUFFER, 0, ffData, 0, m * 4); return m; }
function ffUniforms(mode, focal) { const u = ffP.u; gl.useProgram(ffP.p); gl.uniform3fv(u.uCam, cam.pos); gl.uniform3fv(u.uFwd, cam.fwd); gl.uniform3fv(u.uRight, cam.right); gl.uniform3fv(u.uUp, cam.up);
  gl.uniform2f(u.uTan, cam.tanX, cam.tanY); gl.uniform1f(u.uFocal, focal); gl.uniform1f(u.uFar, FAR); gl.uniform1f(u.uMode, mode); gl.uniform1f(u.uDpr, dpr); gl.uniform1f(u.uZs, ZS); }

let tDraw = 0;
function draw() { if (!gl) return; size(); cam.yaw += (cam.tyaw - cam.yaw) * .05; aim(); const W = canvas.width, H = canvas.height, m = fill();
  /* 1. the light that falls on the leaves */
  gl.bindFramebuffer(gl.FRAMEBUFFER, glow.fb); gl.viewport(0, 0, glow.w, glow.h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  ffUniforms(1, cam.focal * glow.h / H); gl.uniform1f(ffP.u.uZs, ZS); gl.bindVertexArray(ffVAO);
  if (!glow.float) { /* scale into range for an 8-bit target */ gl.blendColor(GLOW_K, GLOW_K, GLOW_K, 1); gl.blendFunc(gl.CONSTANT_COLOR, gl.ONE); }
  if (m) gl.drawArrays(gl.POINTS, 0, m);
  /* 2. the wood, with its depth */
  gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); gl.disable(gl.BLEND);
  gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.ALWAYS); gl.depthMask(true); gl.clear(gl.DEPTH_BUFFER_BIT);
  const u = bgP.u; gl.useProgram(bgP.p); gl.uniform3fv(u.uFwd, cam.fwd); gl.uniform3fv(u.uRight, cam.right); gl.uniform3fv(u.uUp, cam.up); gl.uniform2f(u.uTan, cam.tanX, cam.tanY);
  gl.uniform1f(u.uT, tDraw); gl.uniform1f(u.uPix, 2 * cam.tanY / H); gl.uniform1f(u.uTorch, S.P.torch); gl.uniform1f(u.uTorchX, S.P.torchX - cam.yaw); gl.uniform1f(u.uTorchHalf, S.P.torchHalf); gl.uniform1f(u.uTorchY, S.P.torchY); gl.uniform1f(u.uTorchHalfV, S.P.torchHalfV);
  gl.uniform1f(u.uCamH, CAMH); gl.uniform1f(u.uFar, FAR); gl.uniform2f(u.uGlowScale, 1 / GLOW_K * .5, ZS);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, glow.tex); gl.uniform1i(u.uGlow, 0);
  gl.bindVertexArray(quad); gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  /* 3. the fireflies, hidden where the wood is nearer */
  gl.depthFunc(gl.LEQUAL); gl.depthMask(false); gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE);
  ffUniforms(0, cam.focal); gl.bindVertexArray(ffVAO); if (m) gl.drawArrays(gl.POINTS, 0, m);
  gl.bindVertexArray(null); gl.depthMask(true); gl.disable(gl.BLEND); }

/* ─── the rhythm: flashes in each tenth of a second over the last minute, and how long the dark gaps last ─── */
const rc = $('rhythm'), rx = rc.getContext('2d');
const css = k => getComputedStyle(document.documentElement).getPropertyValue(k).trim();
const INK = css('--ink') || '#14202a', SOFT = css('--soft') || '#4d5a61', HAIR = css('--hair') || '#d9cfae', GOLD = css('--gold') || '#b98619', VIR = css('--viridian') || '#0d7a70';
function drawRhythm() {
  const w = rc.clientWidth, h = rc.clientHeight, r = Math.min(devicePixelRatio || 1, 2); if (!w) return;
  if (rc.width !== Math.round(w * r)) { rc.width = Math.round(w * r); rc.height = Math.round(h * r); }
  rx.setTransform(r, 0, 0, r, 0, 0); rx.clearRect(0, 0, w, h);
  const narrow = w < 560, gap = 26, lw = narrow ? w : Math.round(w * .62), hw = narrow ? w : w - lw - gap, top = 22, lh = narrow ? (h - 2 * top - 64) / 2 : h - top - 26;
  rx.font = '13px "IM Fell English SC", Georgia, serif'; rx.fillStyle = SOFT; rx.textBaseline = 'alphabetic';
  /* left: the last 60 s */
  rx.fillText('flashes, each tenth of a second, the last minute', 0, 13);
  const n = S.n, tNow = S.t, b1 = Math.floor(tNow / BIN), y0 = top + lh; let peak = 0;
  for (let k = 0; k < NB; k++) { const b = b1 - k; if (b > binT || b < 0) continue; peak = Math.max(peak, bins[b % NB]); }
  const sc = lh / Math.max(8, Math.min(n, peak * 1.1));
  rx.strokeStyle = HAIR; rx.lineWidth = 1; rx.beginPath(); rx.moveTo(0, y0 + .5); rx.lineTo(lw, y0 + .5); rx.stroke();
  rx.fillStyle = GOLD;
  for (let k = 0; k < NB; k++) { const b = b1 - k; if (b > binT || b < 0) continue; const v = bins[b % NB]; if (!v) continue; const x = lw - (k + 1) * lw / NB; rx.fillRect(x, y0 - v * sc, Math.max(1, lw / NB), v * sc); }
  rx.fillStyle = SOFT; rx.textAlign = 'right'; rx.fillText('now', lw, y0 + 16); rx.textAlign = 'left'; rx.fillText('a minute ago', 0, y0 + 16);
  /* right: how long between bursts. The swarm (viridian) against one firefly alone (outline). */
  const ox = narrow ? 0 : lw + gap, oy = narrow ? y0 + 46 : top, hh = narrow ? lh : lh, maxS = 60, nb = 30, bw = hw / nb;
  rx.fillStyle = SOFT; rx.fillText('seconds each firefly waits between its bursts', ox, oy - 9);
  const hist = (arr) => { const hh2 = new Float32Array(nb); for (const v of arr) { const k = Math.floor(v / maxS * nb); if (k >= 0 && k < nb) hh2[k]++; } let s = 0; for (const v of hh2) s += v; if (s) for (let k = 0; k < nb; k++) hh2[k] /= s; return hh2; };
  const A = hist(ownIv), B = hist(lone); let mx = .05; for (let k = 0; k < nb; k++) mx = Math.max(mx, A[k], B[k]);
  const by = oy + hh; rx.strokeStyle = HAIR; rx.beginPath(); rx.moveTo(ox, by + .5); rx.lineTo(ox + hw, by + .5); rx.stroke();
  rx.fillStyle = VIR; for (let k = 0; k < nb; k++) if (A[k]) rx.fillRect(ox + k * bw + .5, by - A[k] / mx * hh, bw - 1, A[k] / mx * hh);
  rx.strokeStyle = INK; rx.lineWidth = 1; rx.beginPath(); let started = false;
  for (let k = 0; k < nb; k++) { const y = by - B[k] / mx * hh; if (!started) { rx.moveTo(ox + k * bw, by); started = true; } rx.lineTo(ox + k * bw, y); rx.lineTo(ox + (k + 1) * bw, y); } rx.lineTo(ox + hw, by); rx.stroke();
  rx.fillStyle = SOFT; for (const s of [0, 20, 40, 60]) { rx.textAlign = s === 0 ? 'left' : s === 60 ? 'right' : 'center'; rx.fillText(String(s), ox + s / maxS * hw, by + 16); } rx.textAlign = 'left';
}

/* ─── readouts: all measured from this run ─── */
function sample() {
  let now = 0; const t = S.t; for (let i = 0; i < S.n; i++) if (t - S.lastFlash[i] < .25) now++;
  $('r1').textContent = S.n.toLocaleString('en-GB');
  $('r2').textContent = now.toLocaleString('en-GB');
  const iv = ivAll.slice(-10), st = M.stats(iv);
  $('r3').textContent = iv.length >= 3 ? `${st.mean.toFixed(1)} s, give or take ${st.sd.toFixed(1)}` : (S.t - lastAny < 1.5 && ownIv.length > 30 ? 'never dark' : 'not yet measured');
  const sh = shares.slice(-6); $('r4').textContent = sh.length ? Math.round(100 * sh.reduce((a, b) => a + b, 0) / sh.length) + '%' : '–';
  $('r5').textContent = (S.nSpont).toLocaleString('en-GB');
}

/* ─── sound: a stream, a tree cricket, and the flashes made audible as soft notes (fireflies are silent) ─── */
let ac = null, soundOn = false, master = null, noteGain = null, lastNote = 0, beatSeen = 0;
const SCALE = [0, 2, 4, 7, 9, 12, 14, 16];
function startSound() {
  if (ac) { ac.resume(); return; }
  const AC = window.AudioContext || window.webkitAudioContext; if (!AC) return; ac = new AC();
  master = ac.createGain(); master.gain.value = 0; master.connect(ac.destination);
  const verb = ac.createConvolver(); { const len = ac.sampleRate * 2.6, b = ac.createBuffer(2, len, ac.sampleRate); for (let c = 0; c < 2; c++) { const d = b.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 3); } verb.buffer = b; }
  const wet = ac.createGain(); wet.gain.value = .5; verb.connect(wet); wet.connect(master);
  noteGain = ac.createGain(); noteGain.gain.value = .9; noteGain.connect(master); noteGain.connect(verb);
  /* the Little River over stones: brown noise, two bands, slowly breathing */
  const len = ac.sampleRate * 4, nb = ac.createBuffer(2, len, ac.sampleRate);
  for (let c = 0; c < 2; c++) { const d = nb.getChannelData(c); let last = 0; for (let i = 0; i < len; i++) { last = (last + .02 * (Math.random() * 2 - 1)) / 1.02; d[i] = last * 3.2; } }
  for (const [f, q, g] of [[380, .7, .22], [1900, 1.2, .045]]) { const s = ac.createBufferSource(); s.buffer = nb; s.loop = true; s.playbackRate.value = .9 + Math.random() * .2;
    const bp = ac.createBiquadFilter(); bp.type = 'bandpass'; bp.frequency.value = f; bp.Q.value = q; const gg = ac.createGain(); gg.gain.value = g;
    const lfo = ac.createOscillator(), lg = ac.createGain(); lfo.frequency.value = .07 + Math.random() * .05; lg.gain.value = g * .35; lfo.connect(lg); lg.connect(gg.gain); lfo.start();
    s.connect(bp); bp.connect(gg); gg.connect(master); s.start(); }
  /* a tree cricket in the distance: at 20 °C, Dolbear's rule gives about two chirps a second */
  const cr = ac.createOscillator(), cg = ac.createGain(), am = ac.createOscillator(), amg = ac.createGain(); cr.frequency.value = 2950; cg.gain.value = 0;
  am.type = 'square'; am.frequency.value = 1.87; amg.gain.value = .006; am.connect(amg); amg.connect(cg.gain); cr.connect(cg); const cp = ac.createStereoPanner ? ac.createStereoPanner() : null;
  if (cp) { cp.pan.value = .55; cg.connect(cp); cp.connect(master); cp.connect(verb); } else cg.connect(master); cr.start(); am.start();
  master.gain.setTargetAtTime(.8, ac.currentTime, .6);
}
function note(k, amt) { if (!ac || !soundOn) return; const t = ac.currentTime; if (t - lastNote < .07) return; lastNote = t;
  const f = 880 * Math.pow(2, SCALE[Math.min(SCALE.length - 1, k)] / 12), o = ac.createOscillator(), o2 = ac.createOscillator(), g = ac.createGain();
  o.type = 'sine'; o.frequency.value = f; o2.type = 'sine'; o2.frequency.value = f * 2.76; const g2 = ac.createGain(); g2.gain.value = .18;
  const a = Math.min(.09, .012 + .08 * Math.sqrt(amt)); g.gain.setValueAtTime(0, t); g.gain.linearRampToValueAtTime(a, t + .012); g.gain.exponentialRampToValueAtTime(.0005, t + 1.4);
  o.connect(g); o2.connect(g2); g2.connect(g); g.connect(noteGain); o.start(t); o2.start(t); o.stop(t + 1.5); o2.stop(t + 1.5); }

/* ─── controls ─── */
let paused = false, visible = true, last = 0, frames = 0, slow = 0;
$('snd').onclick = () => { soundOn = !soundOn; $('snd').setAttribute('aria-pressed', String(soundOn)); if (soundOn) startSound(); else if (ac) master.gain.setTargetAtTime(0, ac.currentTime, .2); if (soundOn && master) master.gain.setTargetAtTime(.8, ac.currentTime, .4); };
$('n').oninput = e => { M.setCount(S, +e.target.value); $('nv').textContent = e.target.value; resetMeasures(); };
const TORCH = ['Torch off', 'White torch', 'Red torch'];
$('torch').onclick = () => { S.P.torch = (S.P.torch + 1) % 3; $('torch').textContent = TORCH[S.P.torch]; $('torch').setAttribute('aria-pressed', S.P.torch ? 'true' : 'false'); };
$('hide').onclick = () => { S.P.see = !S.P.see; $('hide').setAttribute('aria-pressed', S.P.see ? 'false' : 'true'); resetMeasures(); };
function setPaused(p) { paused = p; $('pause').setAttribute('aria-pressed', p ? 'true' : 'false'); $('pause').textContent = p ? 'Paused' : 'Pause'; last = 0; }
$('pause').onclick = () => setPaused(!paused);
/* the torch follows the pointer across the picture; without a torch the view leans a little toward it */
function aimAt(e) { const r = canvas.getBoundingClientRect(), fx = ((e.clientX - r.left) / r.width - .5) * 2; const fy = -((e.clientY - r.top) / r.height - .5) * 2; S.P.torchX = Math.atan(fx * cam.tanX) + cam.yaw; S.P.torchY = Math.max(-.25, Math.min(.35, Math.atan(fy * cam.tanY) + cam.pitch)); cam.tyaw = S.P.torch ? 0 : fx * .05; }
canvas.addEventListener('pointermove', aimAt); canvas.addEventListener('pointerdown', aimAt);
canvas.addEventListener('pointerleave', () => { cam.tyaw = 0; });
addEventListener('resize', () => { size(); if (paused) { draw(); drawRhythm(); } });
if ('IntersectionObserver' in window) new IntersectionObserver(e => { visible = e[0].isIntersecting; }, { threshold: .02 }).observe(canvas);
document.addEventListener('visibilitychange', () => { if (ac && soundOn) { if (document.hidden) ac.suspend(); else ac.resume(); } last = 0; });

function advance(sec) { const dt = 1 / 120, k = Math.round(sec / dt); for (let s = 0; s < k; s++) M.step(S, dt); }
let sampleAt = 0;
function frame(ts) {
  requestAnimationFrame(frame);
  if (document.hidden || !visible) { last = 0; return; }
  const dt = last ? Math.min(.05, (ts - last) / 1000) : 1 / 60; last = ts;
  if (!paused) {
    flashesThisFrame = 0; const b0 = beat, s0 = stampN; advance(dt); tDraw += dt;
    if (flashesThisFrame && (beat !== beatSeen || stampN !== s0)) { beatSeen = beat; note(Math.max(0, beat - 1), flashesThisFrame / Math.max(1, S.n * .4)); }
    frames++; if (dt > .03) slow++; if (frames % 120 === 0) { if (slow > 70 && quality > .55) { quality *= .8; size(); } slow = 0; }
  }
  draw();
  if (ts > sampleAt) { sampleAt = ts + 400; sample(); drawRhythm(); }
}

/* start a little way in, so that the swarm has found its rhythm */
advance(20); resetMeasures(); advance(still ? 50 : 40); setPaused(still); sample(); drawRhythm();
window.FIREFLIES = { S, cam, advance, draw, setPaused, state: () => ({ n: S.n, t: S.t, paused, frames, quality, sound: soundOn, ac: ac ? ac.state : 'none', iv: ivAll.slice(-10), own: M.stats(ownIv), shares: shares.slice(-6), glowFloat: glow.float }) };
requestAnimationFrame(frame);
})();
