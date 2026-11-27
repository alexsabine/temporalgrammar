/* Rain eating rock. A magnified slice through the surface of a schist: quartz (glassy grey), feldspar
   (cream-pink) and mica flakes lying along the tilted layering. Rain carries dissolved CO2, a weak acid.
   It attacks the feldspar from the surface down along the grain boundaries: the feldspar pits and turns to
   cloudy clay, the quartz is left standing. The ions set loose (K violet, Ca pale lime, Mg teal) leave in
   the water film, each travelling with the carbon that freed it, now bicarbonate. A time-lapse; at the end
   it fades back to fresh rock and begins again. */
HSM.sim('weather',{
 slider:{label:'CO₂ in the rain',lo:'little',hi:'a lot',value:.4},
 setup:function(w,h,R){
  var k=w/390,D=2,th=-.3,ct=Math.cos(th),sn=Math.sin(th),st=1.8,c=28*k,y0=h*.3,fy=h*.2,i,j,n,q,g;
  function hash(a,b){var v=Math.sin(a*127.1+b*311.7)*43758.5453;return v-Math.floor(v)}
  function surf(x){return y0+(x-w/2)*.035+k*(3*Math.sin(x*.021/k+1.3)+2*Math.sin(x*.057/k+4.1)+1.1*Math.sin(x*.13/k+.7))}
  function cv(W,H){var e=document.createElement('canvas');e.width=Math.ceil(W);e.height=Math.ceil(H);return e}
  function clip(P,nx,ny,cc){var O=[],m=P.length;for(var a=0;a<m;a++){var A=P[a],B=P[(a+1)%m],da=nx*A[0]+ny*A[1]-cc,db=nx*B[0]+ny*B[1]-cc;
    if(da<=0)O.push(A);if((da<0&&db>0)||(da>0&&db<0)){var f=da/(da-db);O.push([A[0]+(B[0]-A[0])*f,A[1]+(B[1]-A[1])*f])}}return O}
  function poly(g,P){g.beginPath();g.moveTo(P[0][0],P[0][1]);for(var n=1;n<P.length;n++)g.lineTo(P[n][0],P[n][1]);g.closePath()}
  function surfPath(g){g.beginPath();g.moveTo(-6,h+6);for(var x=-6;x<=w+6;x+=3)g.lineTo(x,surf(x));g.lineTo(w+6,h+6);g.closePath()}
  function soft(g,x,y,r,col){var q=g.createRadialGradient(x,y,0,x,y,r);q.addColorStop(0,col);q.addColorStop(1,col.replace(/[\d.]+\)$/,'0)'));g.fillStyle=q;g.fillRect(x-r,y-r,r*2,r*2)}

  /* grains: a Voronoi mosaic computed in a space stretched along the layering, so the grains lie with it */
  var seeds=[],grains=[];
  for(j=-40;j<40;j++)for(i=-30;i<30;i++){var U=(i+.5+(j&1)*.5+(R()-.5)*.75)*c,V=(j+.5+(R()-.5)*.75)*c,u=U*st,x=w/2+u*ct-V*sn,y=h/2+u*sn+V*ct;
   if(x>-130*k&&x<w+130*k&&y>y0-140*k&&y<h+130*k)seeds.push({U:U,V:V,x:x,y:y})}
  seeds.forEach(function(s){
   if(s.x<-45*k||s.x>w+45*k||s.y<y0-45*k||s.y>h+45*k)return;
   var P=[[s.U-2.2*c,s.V-2.2*c],[s.U+2.2*c,s.V-2.2*c],[s.U+2.2*c,s.V+2.2*c],[s.U-2.2*c,s.V+2.2*c]];
   for(var n=0;n<seeds.length&&P.length>2;n++){var o=seeds[n];if(o===s)continue;var dx=o.U-s.U,dy=o.V-s.V;if(dx*dx+dy*dy>11*c*c)continue;
    P=clip(P,dx,dy,(o.U*o.U+o.V*o.V-s.U*s.U-s.V*s.V)/2)}
   if(P.length<3)return;
   var Q=P.map(function(p){var u=p[0]*st,x=w/2+u*ct-p[1]*sn,y=h/2+u*sn+p[1]*ct,a=Math.round(x/2),b=Math.round(y/2);
    return[x+(hash(a,b)-.5)*c*.24,y+(hash(b+9,a-4)-.5)*c*.24]}),O=[];
   for(n=0;n<Q.length;n++){var A=Q[n],B=Q[(n+1)%Q.length],A2=A,B2=B;O.push(A);
    if(A[0]>B[0]||(A[0]==B[0]&&A[1]>B[1])){A2=B;B2=A}
    var ex=B2[0]-A2[0],ey=B2[1]-A2[1],len=Math.sqrt(ex*ex+ey*ey);if(len<7*k)continue;
    var mx=(A[0]+B[0])/2,my=(A[1]+B[1])/2,dd=(hash(Math.round(mx),Math.round(my))-.5)*len*.2;O.push([mx-ey/len*dd,my+ex/len*dd])}
   var G={P:O,cx:0,cy:0,x0:1e9,y0:1e9,x1:-1e9,y1:-1e9};
   O.forEach(function(p){G.cx+=p[0]/O.length;G.cy+=p[1]/O.length;G.x0=Math.min(G.x0,p[0]);G.x1=Math.max(G.x1,p[0]);G.y0=Math.min(G.y0,p[1]);G.y1=Math.max(G.y1,p[1])});
   if(G.x1<-4||G.x0>w+4||G.y1<y0-12*k)return;
   G.F=R()<(G.y0<surf(G.cx)+4*k?.66:.3+.36*(.5+.5*Math.sin(s.V/(c*1.7)+.6)));G.t=R();G.ga=R()*6.283;G.cl=R()*3.14;grains.push(G)});

  /* fresh rock, with the sky above it */
  var rock=cv(w*D,h*D);g=rock.getContext('2d');g.scale(D,D);
  q=g.createLinearGradient(0,0,0,y0);q.addColorStop(0,'#0a141b');q.addColorStop(.6,'#1b3440');q.addColorStop(1,'#2d4f5c');g.fillStyle=q;g.fillRect(0,0,w,h);
  soft(g,w*.3,y0*.5,w*.7,'rgba(140,180,190,.16)');soft(g,w*.85,y0*.2,w*.4,'rgba(90,120,150,.14)');
  g.save();surfPath(g);g.clip();g.fillStyle='#191614';g.fillRect(0,0,w,h);g.lineJoin='round';
  grains.forEach(function(G){var P=G.P,L,len=Math.max(G.x1-G.x0,G.y1-G.y0)*.55,ca=Math.cos(G.ga),sa=Math.sin(G.ga),n,x,y;
   q=g.createLinearGradient(G.cx-ca*len,G.cy-sa*len,G.cx+ca*len,G.cy+sa*len);
   if(G.F){if(G.t<.6){q.addColorStop(0,'rgb(244,202,180)');q.addColorStop(1,'rgb(196,138,120)')}else{q.addColorStop(0,'rgb(246,214,196)');q.addColorStop(1,'rgb(214,160,140)')}}
   else{L=G.t<.25?128:172+56*G.t;q.addColorStop(0,'rgb('+(L+8|0)+','+(L+15|0)+','+(L+22|0)+')');q.addColorStop(1,'rgb('+(L-82|0)+','+(L-74|0)+','+(L-62|0)+')')}
   poly(g,P);g.fillStyle=q;g.fill();g.save();g.clip();
   if(G.F){/* cleavage: fine parallel lines, a couple of broader twin bands */
    var a=G.cl,cx=Math.cos(a),sx=Math.sin(a);g.lineWidth=.6*k;
    for(n=-18;n<=18;n++){x=G.cx-sx*n*3.1*k;y=G.cy+cx*n*3.1*k;g.strokeStyle=n&1?'rgba(255,244,230,.2)':'rgba(120,66,54,.17)';g.beginPath();g.moveTo(x-cx*80*k,y-sx*80*k);g.lineTo(x+cx*80*k,y+sx*80*k);g.stroke()}
    for(n=0;n<2;n++){var o=(R()-.5)*30*k;g.lineWidth=(4+R()*5)*k;g.strokeStyle='rgba(255,236,214,.13)';g.beginPath();g.moveTo(G.cx-sx*o-cx*80*k,G.cy+cx*o-sx*80*k);g.lineTo(G.cx-sx*o+cx*80*k,G.cy+cx*o+sx*80*k);g.stroke()}
    soft(g,G.cx-ca*len*.5,G.cy-sa*len*.5,len*.9,'rgba(255,246,236,.22)')}
   else{/* glass: internal reflections, hairline fractures, trails of tiny inclusions */
    for(n=0;n<3;n++){g.fillStyle=n<2?'rgba(255,255,255,'+(.07+R()*.09)+')':'rgba(10,20,34,.16)';g.beginPath();
     g.moveTo(G.cx+(R()-.5)*len*2.4,G.cy+(R()-.5)*len*2.4);g.lineTo(G.cx+(R()-.5)*len*2.4,G.cy+(R()-.5)*len*2.4);g.lineTo(G.cx+(R()-.5)*len*2.4,G.cy+(R()-.5)*len*2.4);g.fill()}
    soft(g,G.cx-ca*len*.55,G.cy-sa*len*.55,len*.8,'rgba(255,255,255,.34)');soft(g,G.cx+ca*len*.6,G.cy+sa*len*.6,len*.9,'rgba(8,14,26,.3)');
    for(n=0;n<2;n++){var x1=G.cx+(R()-.5)*len*2,y1=G.cy+(R()-.5)*len*2,x2=G.cx+(R()-.5)*len*2,y2=G.cy+(R()-.5)*len*2,xm=G.cx+(R()-.5)*len,ym=G.cy+(R()-.5)*len;
     g.lineWidth=.7*k;g.strokeStyle='rgba(255,255,255,.3)';g.beginPath();g.moveTo(x1,y1);g.quadraticCurveTo(xm,ym,x2,y2);g.stroke();
     g.strokeStyle='rgba(20,30,44,.2)';g.beginPath();g.moveTo(x1+k,y1+k);g.quadraticCurveTo(xm+k,ym+k,x2+k,y2+k);g.stroke()}
    x=G.cx+(R()-.5)*len;y=G.cy+(R()-.5)*len;a=R()*6.283;g.fillStyle='rgba(30,40,54,.3)';
    for(n=0;n<9;n++){g.beginPath();g.arc(x+Math.cos(a)*n*2.4*k+(R()-.5)*k,y+Math.sin(a)*n*2.4*k+(R()-.5)*k,(.3+R()*.5)*k,0,6.283);g.fill()}}
   g.restore()});
  grains.forEach(function(G){poly(g,G.P);g.lineWidth=2.4*k;g.strokeStyle='rgba(255,255,255,.13)';g.stroke()});
  grains.forEach(function(G){poly(g,G.P);g.lineWidth=.8*k;g.strokeStyle='rgba(22,18,16,.8)';g.stroke()});
  g.restore();

  /* over everything in the rock: mica flakes along the layering, a fine speckle, depth and a wet lit rim */
  var ov=cv(w*D,h*D);g=ov.getContext('2d');g.scale(D,D);surfPath(g);g.clip();
  function lath(U,V,L,T,musc){var u=U*st,x=w/2+u*ct-V*sn,y=h/2+u*sn+V*ct;if(x<-L||x>w+L||y<y0-L||y>h+L)return;
   g.save();g.translate(x,y);g.rotate(th+(R()-.5)*.16);
   g.beginPath();g.moveTo(-L/2,-T*.15);g.lineTo(-L/2+3*k,-T/2);g.lineTo(L/2-2*k,-T/2);g.lineTo(L/2,T*.2);g.lineTo(L/2-3.5*k,T/2);g.lineTo(-L/2+2*k,T/2);g.closePath();
   var q=g.createLinearGradient(0,-T/2,0,T/2);
   if(musc){q.addColorStop(0,'#3a4146');q.addColorStop(.3,'#8d989c');q.addColorStop(.45,'#cfd8d8');q.addColorStop(.65,'#6f7a80');q.addColorStop(1,'#2e3539')}
   else{q.addColorStop(0,'#0d0907');q.addColorStop(.3,'#4f341d');q.addColorStop(.46,'#9a7140');q.addColorStop(.62,'#33210f');q.addColorStop(1,'#0b0706')}
   g.fillStyle=q;g.fill();g.lineWidth=.7*k;g.strokeStyle='rgba(0,0,0,.65)';g.stroke();
   g.strokeStyle='rgba(0,0,0,.4)';g.lineWidth=.5*k;g.beginPath();g.moveTo(-L*.42,T*.18);g.lineTo(L*.4,T*.2);g.moveTo(-L*.3,-T*.28);g.lineTo(L*.44,-T*.26);g.stroke();
   g.strokeStyle=musc?'rgba(255,255,255,.8)':'rgba(255,226,170,.7)';g.lineWidth=.8*k;g.beginPath();g.moveTo(-L*(.1+R()*.3),-T*.06);g.lineTo(L*(.05+R()*.3),-T*.05);g.stroke();g.restore()}
  for(j=-9;j<10;j++){var Vb=j*c*2.55+(R()-.5)*c*.5;for(var Uu=-520*k;Uu<520*k;){var L=(34+R()*46)*k;if(R()<.82)lath(Uu,Vb+(R()-.5)*c*.42,L,(5+R()*6)*k,R()<.14);if(R()<.45)lath(Uu+L*.2,Vb+(R()-.5)*c*.6,L*.7,(3.5+R()*4)*k,R()<.14);Uu+=L*(.34+R()*.3)/st*1.6}}
  for(i=0;i<34;i++)lath((R()-.5)*600*k,(R()-.5)*900*k,(14+R()*20)*k,(2.5+R()*2.5)*k,R()<.3);
  for(i=0;i<2200;i++){g.fillStyle=R()<.5?'rgba(255,255,255,'+(.03+R()*.07)+')':'rgba(0,0,0,'+(.04+R()*.09)+')';g.fillRect(R()*w,y0-10+R()*(h-y0+10),(.6+R()*.9)*k,(.6+R()*.9)*k)}
  q=g.createLinearGradient(0,y0,0,h);q.addColorStop(0,'rgba(4,8,12,0)');q.addColorStop(.45,'rgba(4,8,12,.14)');q.addColorStop(1,'rgba(4,8,12,.6)');g.fillStyle=q;g.fillRect(0,0,w,h);
  g.beginPath();for(x=-6;x<=w+6;x+=3)g.lineTo(x,surf(x)+5*k);g.lineWidth=12*k;g.strokeStyle='rgba(150,205,225,.1)';g.stroke();
  g.beginPath();for(x=-6;x<=w+6;x+=3)g.lineTo(x,surf(x));g.lineWidth=1.6*k;g.strokeStyle='rgba(226,244,252,.5)';g.stroke();

  /* each feldspar grain near the surface: an etched stage, a clay stage and, where it meets the water, a pit */
  var fel=[],sf=[],depthMax=(h-y0)*.6;
  grains.forEach(function(G){if(!G.F)return;var sy=surf(G.cx),del=.06+Math.max(0,G.cy-sy)/depthMax+(R()-.5)*.1;if(del>1.2)return;
   var pad=5*k,X=G.x0-pad,Y=G.y0-pad,W=G.x1-G.x0+2*pad,H=G.y1-G.y0+2*pad,P=G.P,n,m=P.length;
   function sprite(fn){var e=cv(W*D,H*D),q=e.getContext('2d');q.scale(D,D);q.translate(-X,-Y);surfPath(q);q.clip();q.lineJoin='round';fn(q);return e}
   function rim(){var a=Math.floor(R()*m),A=P[a],B=P[(a+1)%m],f=R(),x=A[0]+(B[0]-A[0])*f,y=A[1]+(B[1]-A[1])*f,t=.06+R()*.3;return[x+(G.cx-x)*t,y+(G.cy-y)*t]}
   var o={x:X,y:Y,w:W,h:H,del:Math.max(.06,del)};
   o.A=sprite(function(q){poly(q,P);q.lineWidth=5.5*k;q.strokeStyle='rgba(84,60,38,.5)';q.stroke();q.lineWidth=2.2*k;q.strokeStyle='rgba(22,16,12,.8)';q.stroke();
    q.save();poly(q,P);q.clip();for(n=0;n<16;n++){var p=rim();soft(q,p[0],p[1],(4+R()*6)*k,'rgba(240,235,216,.6)')}
    for(n=0;n<22;n++){p=rim();q.fillStyle='rgba(36,24,16,.6)';q.beginPath();q.arc(p[0],p[1],(.5+R()*1)*k,0,6.283);q.fill()}q.restore()});
   o.B=sprite(function(q){q.save();poly(q,P);q.clip();q.fillStyle='rgba(226,226,214,.95)';q.fillRect(X,Y,W,H);
    for(n=0;n<34;n++){var x=X+R()*W,y=Y+R()*H;soft(q,x,y,(3+R()*8)*k,n%3==0?'rgba(252,252,248,.6)':n%3==1?'rgba(170,160,128,.3)':'rgba(120,124,116,.3)')}
    q.lineWidth=.7*k;for(n=0;n<90;n++){x=X+R()*W;y=Y+R()*H;var a=R()*6.283,l=(1+R()*2.2)*k;q.strokeStyle=n&1?'rgba(104,88,66,.38)':'rgba(255,255,250,.4)';q.beginPath();q.moveTo(x,y);q.lineTo(x+Math.cos(a)*l,y+Math.sin(a)*l);q.stroke()}
    q.strokeStyle='rgba(46,34,24,.6)';q.lineWidth=.8*k;for(n=0;n<3;n++){var p=rim(),p2=rim();q.beginPath();q.moveTo(p[0],p[1]);for(var e=1;e<=5;e++)q.lineTo(p[0]+(p2[0]-p[0])*e/5+(R()-.5)*4*k,p[1]+(p2[1]-p[1])*e/5+(R()-.5)*4*k);q.stroke()}
    q.restore();poly(q,P);q.lineWidth=3.2*k;q.strokeStyle='rgba(30,22,16,.85)';q.stroke();q.lineWidth=6*k;q.strokeStyle='rgba(60,44,30,.3)';q.stroke()});
   if(G.y0<sy-1*k&&G.x1-G.x0>12*k){/* exposed at the surface */
    var xa=Math.max(G.x0,-4),xb=Math.min(G.x1,w+4),pd=Math.min((G.y1-sy)*.6,15*k),ph=R()*6;
    function py(x){return surf(x)+pd*Math.pow(Math.max(0,Math.sin(3.1416*(x-G.x0)/(G.x1-G.x0))),.55)*(.8+.2*Math.sin(x*.45/k+ph))}
    o.C=sprite(function(q){q.save();poly(q,P);q.clip();q.beginPath();q.moveTo(G.x0,Y);for(var x=G.x0;x<=G.x1;x+=1.5)q.lineTo(x,py(x));q.lineTo(G.x1,Y);q.closePath();
     var z=q.createLinearGradient(0,sy-4*k,0,sy+pd);z.addColorStop(0,'rgb(44,92,110)');z.addColorStop(1,'rgb(15,36,48)');q.fillStyle=z;q.fill();
     q.beginPath();for(x=G.x0;x<=G.x1;x+=1.5)q.lineTo(x,py(x)+.8*k);q.lineWidth=2.2*k;q.strokeStyle='rgba(20,14,10,.55)';q.stroke();
     q.beginPath();for(x=G.x0;x<=G.x1;x+=1.5)q.lineTo(x,py(x)-.6*k);q.lineWidth=.8*k;q.strokeStyle='rgba(170,220,235,.4)';q.stroke();q.restore()});
    if(xb-xa>10*k)sf.push({xa:xa+3*k,xb:xb-3*k})}
   G.del=o.del;fel.push(o)});
  fel.sort(function(a,b){return b.del-a.del});

  /* motes: carbon (warm, as in the leaf) and the three ions */
  function mote(col,ring){var r=12,e=cv(r*2*D,r*2*D),q=e.getContext('2d');q.scale(D,D);q.drawImage(HSM.blob(col,32,.3),0,0,r*2,r*2);q.lineWidth=1.6;q.strokeStyle=ring;q.beginPath();q.arc(r,r,r*.6,0,6.283);q.stroke();return e}
  var ms=[mote('rgba(255,186,104,1)','rgba(120,60,6,.7)'),mote('rgba(200,160,255,1)','rgba(70,36,130,.7)'),mote('rgba(226,255,170,1)','rgba(70,100,20,.7)'),mote('rgba(110,236,214,1)','rgba(10,90,84,.7)')];
  /* which grains the captions point at */
  function pick(F,tx,ty){var b=null,bd=1e9;grains.forEach(function(G){if(G.F!=F||G.x1-G.x0<26*k||(F&&G.del==null))return;var d=Math.abs(G.cx-tx)+Math.abs(G.cy-ty)*1.4;if(d<bd){bd=d;b=G}});return b||{cx:tx,cy:ty}}
  var s={k:k,R:R,rock:rock,ov:ov,fel:fel,sf:sf,ms:ms,y0:y0,fy:fy,surf:surf,G:.22,motes:[],drops:[],spl:[],rain:[],next:.2,
   lf:pick(true,w*.3,y0+26*k),lq:pick(false,w*.66,y0+(h-y0)*.5)};
  for(i=0;i<26;i++)s.rain.push({x:R()*w*1.1,y:R()*fy,l:(10+R()*22)*k,v:(260+R()*240)*k});
  for(i=0;i<9;i++)s.motes.push({x:R()*w,y:fy+14*k,st:1,vx:34*k,lf:.25+R()*.5,r:R(),ph:R()*6.28,ion:1+(i%3),fl:0});
  return s},

 frame:function(s,g,w,h,t,dt,ui){
  var a=ui.a==null?.4:ui.a,sm=HSM.smooth,C=HSM.clamp,k=s.k,R=s.R,i,m,o,x,y,n;
  s.G+=dt*(.011+.05*Math.pow(a,1.25));if(s.G>1.14)s.G=.03;
  var G=s.G,F=1-sm(1,1.13,G),spl=s.spl;
  function ft(x){var y=s.fy+(x-w/2)*.035+k*(1.5*Math.sin(x*.05/k+t*1.3)+1*Math.sin(x*.11/k-t*.9));
   for(var n=0;n<spl.length;n++){var ag=t-spl[n].t,d=Math.abs(x-spl[n].x)/k;if(d<90)y+=k*4.5*Math.exp(-ag*2.6)*Math.cos(d*.24-ag*9)*Math.exp(-d/(14+ag*40))}return y}
  g.drawImage(s.rock,0,0,w,h);
  /* rain in the air */
  g.beginPath();for(i=0;i<s.rain.length;i++){m=s.rain[i];m.y+=m.v*dt;if(m.y-m.l>s.fy){m.y=-R()*40;m.x=R()*w*1.1}g.moveTo(m.x,m.y);g.lineTo(m.x+m.l*.1,m.y-m.l)}
  g.lineWidth=1*k;g.strokeStyle='rgba(190,220,240,.2)';g.stroke();
  /* the weathering front: nearest the water first */
  for(i=0;i<s.fel.length;i++){o=s.fel[i];var wv=C((G*1.3-o.del)/.45,0,1);if(wv<=0)continue;
   g.globalAlpha=sm(0,.5,wv)*F;g.drawImage(o.A,o.x,o.y,o.w,o.h);
   if(wv>.3){g.globalAlpha=sm(.3,1,wv)*F;g.drawImage(o.B,o.x,o.y,o.w,o.h)}
   if(o.C&&wv>.6){g.globalAlpha=sm(.6,1,wv)*F;g.drawImage(o.C,o.x,o.y,o.w,o.h)}}
  g.globalAlpha=1;g.drawImage(s.ov,0,0,w,h);
  /* the film of water */
  while(spl.length&&t-spl[0].t>1.7)spl.shift();
  g.beginPath();g.moveTo(-4,ft(-4));for(x=2;x<=w+6;x+=6)g.lineTo(x,ft(x));for(x=w+6;x>=-6;x-=6)g.lineTo(x,s.surf(x)+1);g.closePath();
  var q=g.createLinearGradient(0,s.fy-6*k,0,s.y0+10*k);q.addColorStop(0,'rgba(120,196,214,.5)');q.addColorStop(.25,'rgba(58,124,146,.62)');q.addColorStop(1,'rgba(22,62,80,.8)');g.fillStyle=q;g.fill();
  g.globalCompositeOperation='lighter';
  for(i=0;i<5;i++){x=((t*(16+i*5)*k+i*97*k)%(w+160*k))-80*k;g.globalAlpha=.1;g.drawImage(HSM.blob('rgba(150,220,235,1)',32),x-50*k,s.fy+(x-w/2)*.035-2*k+i*3*k,100*k,30*k)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  g.beginPath();g.moveTo(-4,ft(-4));for(x=2;x<=w+6;x+=6)g.lineTo(x,ft(x));g.lineWidth=4.5*k;g.strokeStyle='rgba(150,215,235,.22)';g.stroke();g.lineWidth=1.3*k;g.strokeStyle='rgba(232,248,255,.85)';g.stroke();
  /* drops arrive, each with its share of CO2 */
  s.next-=dt;if(s.next<0){s.next=.28+R()*.34;n=Math.floor(.35+a*2.65+R());s.drops.push({x:w*.04+R()*w*.92,y:-20,n:n})}
  for(i=s.drops.length-1;i>=0;i--){m=s.drops[i];m.y+=520*k*dt;var top=ft(m.x);
   if(m.y>=top){spl.push({x:m.x,t:t});if(spl.length>6)spl.shift();
    for(n=0;n<m.n&&s.motes.length<84;n++){var tg=s.sf[Math.floor(R()*s.sf.length)],t2=s.sf[Math.floor(R()*s.sf.length)];if(t2&&Math.abs((t2.xa+t2.xb)/2-m.x-20*k)<Math.abs((tg.xa+tg.xb)/2-m.x-20*k))tg=t2;
     var tx=tg?tg.xa+R()*(tg.xb-tg.xa):m.x;s.motes.push({x:m.x+(R()-.5)*8*k,y:top+(3+R()*5)*k,st:0,tx:tx,r:R(),ph:R()*6.28,fl:0,lf:.2+R()*.55,vx:0})}
    s.drops.splice(i,1);continue}
   g.lineCap='round';g.lineWidth=2*k;g.strokeStyle='rgba(215,238,250,.3)';g.beginPath();g.moveTo(m.x,m.y);g.lineTo(m.x+2.2*k,m.y-22*k);g.stroke();
   g.drawImage(HSM.blob('rgba(215,240,255,1)',16,.5),m.x-4.5*k,m.y-6*k,9*k,11*k);
   if(m.n)g.drawImage(s.ms[0],m.x-3*k,m.y-3.6*k,6*k,6*k);g.lineCap='butt'}
  for(i=0;i<spl.length;i++){var ag=t-spl[i].t;if(ag<.5){g.globalAlpha=(1-ag*2)*.7;g.lineWidth=1*k;g.strokeStyle='rgba(235,250,255,1)';g.beginPath();g.ellipse(spl[i].x,ft(spl[i].x),(4+ag*46)*k,(1.2+ag*5)*k,.035,3.1416,6.2832);g.stroke()}}
  g.globalAlpha=1;
  /* carbon sinks to the feldspar, frees an ion, and the pair leaves with the flow */
  g.beginPath();
  for(i=s.motes.length-1;i>=0;i--){m=s.motes[i];
   if(m.st==0){var ty=s.surf(m.tx)+1.5*k,dx=m.tx-m.x,dy=ty-m.y,d=Math.sqrt(dx*dx+dy*dy)+.001,sp=(24+12*m.r)*k;
    m.x+=(dx/d*sp+Math.sin(t*3+m.ph)*9*k)*dt;m.y+=(dy/d*sp+Math.cos(t*2.3+m.ph)*5*k)*dt;
    if(d<3.5*k){m.st=1;m.fl=1;m.hx=m.x;m.hy=m.y;var z=R();m.ion=z<.42?1:z<.84?2:3}}
   else{m.vx+=((30+16*m.r)*k-m.vx)*dt*1.1;m.x+=m.vx*dt;var top2=ft(m.x),ly=top2+m.lf*(s.surf(m.x)-top2)+Math.sin(t*1.7+m.ph)*2.5*k;m.y+=(ly-m.y)*Math.min(1,dt*1.6);
    if(m.x>w+14*k){s.motes.splice(i,1);continue}
    m.ix=m.x+Math.cos(m.ph+t*1.5)*6*k;m.iy=m.y+Math.sin(m.ph+t*1.5)*4.6*k;g.moveTo(m.x,m.y);g.lineTo(m.ix,m.iy)}}
  g.lineWidth=1*k;g.strokeStyle='rgba(255,255,255,.35)';g.stroke();
  var r=4.6*k;
  for(i=0;i<s.motes.length;i++){m=s.motes[i];g.drawImage(s.ms[0],m.x-r,m.y-r,r*2,r*2);
   if(m.st==1){g.drawImage(s.ms[m.ion],m.ix-r*.9,m.iy-r*.9,r*1.8,r*1.8);
    if(m.fl>0){g.globalCompositeOperation='lighter';g.globalAlpha=m.fl;g.drawImage(HSM.blob('rgba(255,236,180,1)',32),m.hx-13*k,m.hy-13*k,26*k,26*k);g.globalAlpha=1;g.globalCompositeOperation='source-over';m.fl-=dt*2}}}
  HSM.vignette(g,w,h,.42);
  /* captions */
  function lead(x1,y1,x2,y2){g.lineWidth=2.6;g.strokeStyle='rgba(6,10,8,.5)';g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();g.lineWidth=1;g.strokeStyle='rgba(255,255,255,.8)';g.stroke();
   g.fillStyle='rgba(255,255,255,.95)';g.beginPath();g.arc(x2,y2,2,0,6.283);g.fill()}
  var f=s.lf,z2=s.lq,y1=s.y0+(h-s.y0)*.36,y2=s.y0+(h-s.y0)*.74;
  lead(Math.min(f.cx,60*k),y1-9,f.cx,f.cy);HSM.label(g,12,y1,'feldspar turning to clay');
  lead(Math.max(z2.cx,w-70*k),y2-9,z2.cx,z2.cy);HSM.label(g,w-12,y2,'quartz: untouched','right');
  y=s.fy-26*k;lead(w-70*k,y+9,w-52*k,ft(w-52*k)+13*k);HSM.label(g,w-12,y,'minerals washed away with the carbon','right');
 }});
