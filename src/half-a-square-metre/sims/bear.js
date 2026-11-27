/* A water bear (tardigrade) in the film of water on a moss leaf, seen from above in bright field.
   It walks on eight telescopic legs, feet planted while the body lumbers past them. When the film
   dries it draws in head and legs and shrinks to a tun; when water returns it swells and walks on.
   A bdelloid rotifer feeds nearby and survives drying the same way. */
(function(){
var C=HSM.clamp,sm=HSM.smooth,TAU=6.2832,PI=Math.PI;
function mix(a,b,k){return a+(b-a)*k}
function col(a,b,k,al){return 'rgba('+Math.round(mix(a[0],b[0],k))+','+Math.round(mix(a[1],b[1],k))+','+Math.round(mix(a[2],b[2],k))+','+al+')'}
/* a smooth closed outline around a spine: centres, normals, half-widths */
function tube(g,o,n,keep){var X=o.X,Y=o.Y,i,j,m=n*2;
 for(i=0;i<n;i++){X[i]=o.bx[i]+o.nx[i]*o.hw[i];Y[i]=o.by[i]+o.ny[i]*o.hw[i];j=m-1-i;X[j]=o.bx[i]-o.nx[i]*o.hw[i];Y[j]=o.by[i]-o.ny[i]*o.hw[i]}
 if(!keep)g.beginPath();g.moveTo((X[m-1]+X[0])/2,(Y[m-1]+Y[0])/2);
 for(i=0;i<m;i++){j=(i+1)%m;g.quadraticCurveTo(X[i],Y[i],(X[i]+X[j])/2,(Y[i]+Y[j])/2)}
 g.closePath()}
function cap(x,e){return x>=e?1:Math.sqrt(Math.max(0,1-(1-x/e)*(1-x/e)))}
/* a point inside the animal: u along the body (0 mouth, 1 tail), v across it (-1..1) */
function P(s,u,v){var f=Math.acos(1-2*C(u,0,1))/PI*(s.M-1),i=Math.min(s.M-2,f|0),q=f-i,
  nx=mix(s.nx[i],s.nx[i+1],q),ny=mix(s.ny[i],s.ny[i+1],q),hw=mix(s.hw[i],s.hw[i+1],q);
 s.qx=mix(s.bx[i],s.bx[i+1],q)+nx*hw*v;s.qy=mix(s.by[i],s.by[i+1],q)+ny*hw*v;s.qnx=nx;s.qny=ny;s.qh=hw}
function leaf(g,x,y,a,len,wid,R,al){g.save();g.translate(x,y);g.rotate(a);
 g.beginPath();g.moveTo(0,-wid*.5);g.bezierCurveTo(len*.4,-wid*.78,len*.8,-wid*.3,len,0);g.bezierCurveTo(len*.8,wid*.3,len*.4,wid*.78,0,wid*.5);g.closePath();
 var q=g.createLinearGradient(0,-wid/2,0,wid/2);q.addColorStop(0,'rgba(104,140,44,'+al+')');q.addColorStop(.5,'rgba(150,180,62,'+al*.8+')');q.addColorStop(1,'rgba(96,132,44,'+al+')');
 g.fillStyle=q;g.fill();g.save();g.clip();
 var cl=wid*.17,cw=wid*.075,row=0;
 for(var yy=-wid*.7;yy<wid*.7;yy+=cw,row++)for(var xx=(row&1)*cl*.5;xx<len;xx+=cl){g.fillStyle='rgba(212,230,120,'+(al*(.2+R()*.34))+')';g.beginPath();g.ellipse(xx+cl*.5,yy,cl*.42,cw*.34,0,0,TAU);g.fill()}
 g.fillStyle='rgba(70,100,30,'+al*.5+')';g.fillRect(0,-wid*.035,len*.8,wid*.07);
 g.restore();g.strokeStyle='rgba(56,84,26,'+al*.9+')';g.lineWidth=wid*.035;g.stroke();g.restore()}
function grain(g,x,y,r,R,fill,rim){var n=9,i,a,rr,px=[],py=[];for(i=0;i<n;i++){a=i/n*TAU;rr=r*(.6+R()*.6);px.push(x+Math.cos(a)*rr);py.push(y+Math.sin(a)*rr)}
 g.beginPath();g.moveTo((px[n-1]+px[0])/2,(py[n-1]+py[0])/2);for(i=0;i<n;i++){var j=(i+1)%n;g.quadraticCurveTo(px[i],py[i],(px[i]+px[j])/2,(py[i]+py[j])/2)}
 g.fillStyle=fill;g.fill();if(rim){g.strokeStyle=rim;g.lineWidth=Math.max(.6,r*.09);g.stroke()}}
function cv(w,h){var c=document.createElement('canvas');c.width=Math.max(2,Math.round(w));c.height=Math.max(2,Math.round(h));return c}

HSM.sim('bear',{
 toggle:{off:'Dry it out',on:'Let it rain'},
 setup:function(w,h,R){
  var i,x,y,r,g,q;
  /* ---- the field: three focal planes, painted once ---- */
  var bg=cv(w,h);g=bg.getContext('2d');
  q=g.createRadialGradient(w*.46,h*.44,0,w*.5,h*.5,Math.max(w,h)*.8);q.addColorStop(0,'#f6f4e6');q.addColorStop(.6,'#e6e8d6');q.addColorStop(1,'#c3cbb6');g.fillStyle=q;g.fillRect(0,0,w,h);
  var f=cv(w*.11,h*.11),fg=f.getContext('2d');fg.scale(.11,.11);            /* far below focus */
  leaf(fg,-w*.25,h*.2,-.5,w*1.25,w*.42,R,.5);leaf(fg,w*1.2,h*.72,2.75,w*1.1,w*.4,R,.42);leaf(fg,w*.1,h*1.12,-1.0,w*.8,w*.3,R,.34);
  for(i=0;i<9;i++)grain(fg,R()*w,R()*h,w*(.04+R()*.07),R,'rgba(120,96,50,'+(.12+R()*.14)+')');
  g.imageSmoothingQuality='high';g.drawImage(f,0,0,w,h);
  var m=cv(w*.34,h*.34),mg=m.getContext('2d');mg.scale(.34,.34);            /* just out of focus */
  leaf(mg,w*.02,h*1.02,-.82,w*.95,w*.36,R,.62);leaf(mg,w*.52,-h*.06,.36,w*.9,w*.3,R,.5);
  for(i=0;i<14;i++){x=R()*w;y=R()*h;grain(mg,x,y,w*(.012+R()*.035),R,'rgba('+(110+R()*50|0)+','+(86+R()*40|0)+',46,'+(.22+R()*.25)+')','rgba(70,50,20,.25)')}
  g.drawImage(m,0,0,w,h);
  for(i=0;i<70;i++){x=R()*w;y=R()*h;r=.5+R()*1.6;g.fillStyle='rgba(60,64,34,'+(.12+R()*.22)+')';g.beginPath();g.ellipse(x,y,r*(1+R()*2),r,R()*PI,0,TAU);g.fill()}   /* in focus: bacteria, fines */
  for(i=0;i<5;i++){x=R()*w;y=R()*h;r=w*(.012+R()*.016);grain(g,x,y,r,R,'rgba(236,232,214,.5)','rgba(80,74,50,.55)');g.fillStyle='rgba(255,255,255,.6)';g.beginPath();g.arc(x-r*.25,y-r*.25,r*.3,0,TAU);g.fill()}
  (function(){var dx=w*.2,dy=h*.86,a=.5,l=w*.075;g.save();g.translate(dx,dy);g.rotate(a);g.beginPath();g.ellipse(0,0,l,l*.24,0,0,TAU);       /* a diatom */
   var d=g.createLinearGradient(0,-l*.24,0,l*.24);d.addColorStop(0,'rgba(206,170,70,.6)');d.addColorStop(.5,'rgba(240,226,150,.45)');d.addColorStop(1,'rgba(190,150,60,.6)');g.fillStyle=d;g.fill();g.strokeStyle='rgba(90,70,24,.7)';g.lineWidth=.9;g.stroke();
   g.beginPath();g.moveTo(-l*.85,0);g.lineTo(l*.85,0);for(var e=-8;e<=8;e++){g.moveTo(e*l*.1,-l*.19*Math.sqrt(1-e*e/81));g.lineTo(e*l*.1,l*.19*Math.sqrt(1-e*e/81))}g.lineWidth=.5;g.strokeStyle='rgba(90,70,24,.4)';g.stroke();g.restore()})();
  /* what the field looks like once the water has gone: tinted, crusted, gritty */
  var dt_=cv(w,h);g=dt_.getContext('2d');g.fillStyle='rgba(150,118,66,.25)';g.fillRect(0,0,w,h);
  for(i=0;i<420;i++){g.fillStyle='rgba(74,52,22,'+(.08+R()*.24)+')';r=.4+R()*1.5;g.fillRect(R()*w,R()*h,r,r)}
  g.lineWidth=.7;for(i=0;i<24;i++){x=R()*w;y=R()*h;r=1+R()*2.6;g.save();g.translate(x,y);g.rotate(R()*PI);g.strokeStyle='rgba(255,252,236,'+(.3+R()*.4)+')';g.strokeRect(-r,-r,r*2,r*2);g.fillStyle='rgba(255,252,236,.12)';g.fillRect(-r,-r,r*2,r*2);g.restore()}
  /* a storage cell: a pale droplet with a refracting rim */
  var sc=cv(16,16);g=sc.getContext('2d');q=g.createRadialGradient(7,7,0,8,8,8);q.addColorStop(0,'rgba(255,255,244,.75)');q.addColorStop(.6,'rgba(226,220,176,.4)');q.addColorStop(.86,'rgba(96,90,50,.55)');q.addColorStop(1,'rgba(96,90,50,0)');g.fillStyle=q;g.fillRect(0,0,16,16);
  /* ---- the animal ---- */
  var L=w*.47,N=13,M=26,s={bg:bg,dryTex:dt_,sc:sc,R:R,L:L,W:L*.185,N:N,M:M,d:L/(N-1),V:L*.12,stride:L*.185*.8,LL:L*.185*.9,
   px:new Float32Array(N),py:new Float32Array(N),bx:new Float32Array(M),by:new Float32Array(M),nx:new Float32Array(M),ny:new Float32Array(M),tx:new Float32Array(M),ty:new Float32Array(M),
   hw:new Float32Array(M),us:new Float32Array(M),pw:new Float32Array(M),pt:new Float32Array(M),X:new Float32Array(M*2),Y:new Float32Array(M*2),
   hd:2.62,om:0,turnT:.1,v:0,ph:0,mode:1,tm:7,tt:2,sw:0,wl:0,k:0,ls:1,go:1,wake:0,wasDry:false,rip:9,re:1,rt:5,legs:[],cells:[],gran:[],motes:[],
   rot:{bx:new Float32Array(10),by:new Float32Array(10),nx:new Float32Array(10),ny:new Float32Array(10),hw:new Float32Array(10),X:new Float32Array(20),Y:new Float32Array(20)}};
  for(i=0;i<N;i++){s.px[i]=w*.4-Math.cos(s.hd)*s.d*i;s.py[i]=h*.56-Math.sin(s.hd)*s.d*i}
  for(i=0;i<M;i++){var u=.5-.5*Math.cos(PI*i/(M-1));s.us[i]=u;
   s.pw[i]=cap(u,.13)*cap(1-u,.2)*(.76+.24*sm(.1,.3,u))*(1-.14*sm(.75,1,u))*(1-.08*Math.exp(-Math.pow((u-.165)/.03,2)))*(1+.08*Math.cos(TAU*(u-.27)/.2)*sm(.15,.24,u)*(1-sm(.82,.92,u)));
   s.pt[i]=Math.pow(cap(u,.34)*cap(1-u,.34),.85)*(1+.045*Math.cos(TAU*u*8))}
  var lu=[.27,.47,.67,.935];
  for(i=0;i<4;i++)for(var sd=-1;sd<2;sd+=2)s.legs.push({u:lu[i],sd:sd,v0:i<3?.6:.42,hind:i===3,off:(3-i)*.25+(sd>0?.5:0),fx:null,fy:0,sx:0,sy:0,sw:-1,c:0});
  for(i=0;i<26;i++)s.cells.push({u:.2+R()*.72,v:(R()<.5?-1:1)*(.3+R()*.5),ph:R()*TAU,r:.1+R()*.08});
  for(i=0;i<34;i++)s.gran.push({u:.28+R()*.52,v:(R()-.5)*.8,r:.03+R()*.05});
  for(i=0;i<54;i++)s.motes.push({x:R()*w,y:R()*h,vx:0,vy:0,r:.6+R()*1.5,a:.14+R()*.3});
  return s},

 frame:function(s,g,w,h,t,dt,ui){
  var dry=!!ui.on,i,j,q,u,x,y,r,W=s.W,L=s.L,N=s.N,M=s.M,R=s.R;
  /* ---- water, and the animal's answer to it ---- */
  if(dry!==s.wasDry){if(!dry)s.rip=0;s.wasDry=dry}
  s.rip+=dt;
  s.wl=C(s.wl+(dry?dt/4.2:-dt/1.7),0,1);
  if(dry){if(s.wl>.36)s.k=Math.min(1,s.k+dt/3.3)}else if(s.wl<.4)s.k=Math.max(0,s.k-dt/3.8);
  var k=sm(0,1,s.k);
  if(s.k>0)s.wake=1.5;else s.wake-=dt;
  s.go+=((dry?1-sm(.03,.34,s.wl):(s.wake<=0?1:0))-s.go)*Math.min(1,dt*2.2);
  /* ---- wandering: walk, pause and look about, walk again ---- */
  s.tm-=dt*s.go;if(s.tm<=0){s.mode=s.mode?0:1;s.tm=s.mode?5+R()*6:1.5+R()*2.2;s.turnT=(R()-.5)*.44}
  s.tt-=dt;if(s.tt<=0){s.tt=1.4+R()*2.6;s.turnT=(R()-.5)*.44}
  var hx=s.px[0],hy=s.py[0],gx=null,gy;
  if(ui.down&&ui.px>=0){gx=ui.px;gy=ui.py}else if(hx<w*.34||hx>w*.66||hy<h*.36||hy>h*.7){gx=w*.5;gy=h*.54}
  if(gx!=null){var da=Math.atan2(gy-hy,gx-hx)-s.hd;da=Math.atan2(Math.sin(da),Math.cos(da));s.turnT=C(da*1.2,-.3,.3)}
  s.om+=(s.turnT-s.om)*Math.min(1,dt*1.8);
  s.v+=((s.mode?s.V:0)*s.go-s.v)*Math.min(1,dt*2.4);
  var ve=s.v*(.72+.28*Math.sin(TAU*s.ph*2)),vr=s.v/s.V;
  s.hd+=s.om*dt*vr;s.ph+=ve*dt/s.stride;
  s.sw+=((s.mode?.22:1)*s.go-s.sw)*Math.min(1,dt*1.6);
  var ls=1-.56*k,dls=ls-s.ls;s.ls=ls;
  s.px[0]+=Math.cos(s.hd)*ve*dt;s.py[0]+=Math.sin(s.hd)*ve*dt;
  var dd=s.d*ls,pa=s.hd,an;for(i=1;i<N;i++){an=Math.atan2(s.py[i-1]-s.py[i],s.px[i-1]-s.px[i])-pa;an=pa+C(Math.atan2(Math.sin(an),Math.cos(an)),-.17,.17);s.px[i]=s.px[i-1]-Math.cos(an)*dd;s.py[i]=s.py[i-1]-Math.sin(an)*dd;pa=an}  /* a body can only bend so far */
  if(dls){x=s.px[0]-s.px[N-1];y=s.py[0]-s.py[N-1];q=dls*L*.5/(Math.sqrt(x*x+y*y)||1);for(i=0;i<N;i++){s.px[i]+=x*q;s.py[i]+=y*q}}
  /* ---- the body's midline, flexing ---- */
  for(i=0;i<M;i++){var f=s.us[i]*(N-1),a=Math.min(N-2,f|0),c=f-a,p0=a?a-1:0,p3=a+2<N?a+2:N-1,c2=c*c,c3=c2*c;
   s.bx[i]=.5*(2*s.px[a]+(s.px[a+1]-s.px[p0])*c+(2*s.px[p0]-5*s.px[a]+4*s.px[a+1]-s.px[p3])*c2+(3*s.px[a]-s.px[p0]-3*s.px[a+1]+s.px[p3])*c3);
   s.by[i]=.5*(2*s.py[a]+(s.py[a+1]-s.py[p0])*c+(2*s.py[p0]-5*s.py[a]+4*s.py[a+1]-s.py[p3])*c2+(3*s.py[a]-s.py[p0]-3*s.py[a+1]+s.py[p3])*c3)}
  for(var pass=0;pass<2;pass++){
   for(i=0;i<M;i++){a=i?i-1:0;c=i<M-1?i+1:i;x=s.bx[c]-s.bx[a];y=s.by[c]-s.by[a];q=Math.sqrt(x*x+y*y)||1;s.tx[i]=x/q;s.ty[i]=y/q;s.nx[i]=-y/q;s.ny[i]=x/q}
   if(!pass)for(i=0;i<M;i++){u=s.us[i];var lat=(Math.sin(TAU*s.ph-u*4.2)*W*.085*vr*sm(0,.3,u)+(Math.sin(t*1.25)+.6*Math.sin(t*2.9+1))*W*.42*s.sw*Math.pow(1-u,3))*(1-k);s.bx[i]+=s.nx[i]*lat;s.by[i]+=s.ny[i]*lat}}
  var breathe=1+.014*Math.sin(t*2.1)*(1-k);
  for(i=0;i<M;i++)s.hw[i]=W*breathe*mix(s.pw[i],s.pt[i],k)*(1-.1*k);
  var mid=M>>1,cx=s.bx[mid],cy=s.by[mid];

  /* ---- paint: field, water's edge, things adrift ---- */
  g.drawImage(s.bg,0,0,w,h);
  var diag=Math.sqrt(w*w+h*h),wr=diag*.74*Math.pow(1-s.wl,1.25);
  if(s.wl>.001){
   g.save();g.beginPath();g.rect(0,0,w,h);
   if(wr>3){for(i=0;i<=90;i++){a=i/90*TAU;r=wr*(1+.1*Math.sin(3*a+1.3)+.06*Math.sin(5*a+t*.25)+.035*Math.sin(9*a-t*.4));x=cx+Math.cos(a)*r;y=cy+Math.sin(a)*r*.92;i?g.lineTo(x,y):g.moveTo(x,y)}g.closePath()}
   g.clip('evenodd');g.globalAlpha=sm(0,.12,s.wl);g.drawImage(s.dryTex,0,0,w,h);g.restore();
   if(wr>3){g.beginPath();for(i=0;i<=90;i++){a=i/90*TAU;r=wr*(1+.1*Math.sin(3*a+1.3)+.06*Math.sin(5*a+t*.25)+.035*Math.sin(9*a-t*.4));x=cx+Math.cos(a)*r;y=cy+Math.sin(a)*r*.92;i?g.lineTo(x,y):g.moveTo(x,y)}g.closePath();
    q=sm(0,.1,s.wl)*sm(3,30,wr);g.lineJoin='round';g.lineWidth=11;g.strokeStyle='rgba(60,54,30,'+.1*q+')';g.stroke();g.lineWidth=5;g.strokeStyle='rgba(50,46,26,'+.3*q+')';g.stroke();g.lineWidth=2.2;g.strokeStyle='rgba(255,255,250,'+.8*q+')';g.stroke()}}
  if(s.rip<1.8){g.lineWidth=1.6;for(j=0;j<3;j++){q=s.rip-j*.22;if(q>0){g.strokeStyle='rgba(255,255,255,'+(.5*(1-q/1.6)>0?.5*(1-q/1.6):0)+')';g.beginPath();g.ellipse(cx,cy,q*w*.6+4,q*w*.56+4,0,0,TAU);g.stroke()}}}
  for(i=0;i<s.motes.length;i++){var m=s.motes[i];x=m.x-cx;y=m.y-cy;var wet=x*x+y*y<wr*wr*.8;
   if(wet){m.vx+=(R()-.5)*60*dt;m.vy+=(R()-.5)*60*dt;m.vx*=.96;m.vy*=.96;m.x+=m.vx*dt+2*dt;m.y+=m.vy*dt+1.2*dt;if(m.x>w+4)m.x=-4;if(m.y>h+4)m.y=-4;if(m.x<-4)m.x=w+4;if(m.y<-4)m.y=h+4}
   g.fillStyle='rgba(52,58,28,'+m.a*(wet?1:.6)+')';g.fillRect(m.x,m.y,m.r*1.6,m.r)}

  /* ---- a rotifer, anchored by its foot, wheels turning ---- */
  var o=s.rot,rl=w*.26,rw=w*.036;
  s.rt-=dt;if(s.rt<0)s.rt=6+R()*7;
  var reT=s.wl>.3?.3:(s.rt<.7?.45:1);s.re+=(reT-s.re)*Math.min(1,dt*(reT<s.re?7:1.1));
  var e=s.re,rk=sm(.5,1,s.wl),ra=2.42+.1*Math.sin(t*.7)*e,rpf=[.14,.2,.3,.62,.96,1,.78,.52,.6,.72];
  var ax=w*.93,ay=h*.1,rdx=Math.cos(ra),rdy=Math.sin(ra);
  for(i=0;i<10;i++){u=i/9;var bend=Math.sin(u*2.4+t*.9)*rw*.5*e*u;o.bx[i]=ax+rdx*rl*e*u-rdy*bend;o.by[i]=ay+rdy*rl*e*u+rdx*bend;o.nx[i]=-rdy;o.ny[i]=rdx;o.hw[i]=rw*rpf[i]*(1+.75*(1-e))*(i===9?mix(.5,1,sm(.6,.95,e)):1)}
  g.globalAlpha=.1;g.drawImage(HSM.blob('rgba(50,60,30,1)',32),o.bx[5]-rw*2+3,o.by[5]-rw*2+5,rw*4,rw*4);g.globalAlpha=1;
  tube(g,o,10);g.lineWidth=3.5;g.strokeStyle='rgba(255,255,250,.3)';g.stroke();g.fillStyle=col([238,232,208],[186,150,92],rk,mix(.55,.88,rk));g.fill();
  g.save();g.clip();g.lineWidth=rw*.7;g.strokeStyle=col([110,104,60],[90,60,26],rk,.16);g.stroke();
  r=rw*.74*(1+.5*(1-e));g.globalAlpha=.62;g.drawImage(HSM.blob('rgba(150,128,44,1)',24,.45),o.bx[4]-r,o.by[4]-r,r*2,r*2);g.drawImage(HSM.blob('rgba(104,124,40,1)',24,.45),o.bx[5]-r*.8,o.by[5]-r*.8,r*1.6,r*1.6);g.globalAlpha=1;
  g.strokeStyle='rgba(80,70,40,.22)';g.lineWidth=.7;g.beginPath();for(i=1;i<9;i++){g.moveTo(o.bx[i]+o.nx[i]*o.hw[i],o.by[i]+o.ny[i]*o.hw[i]);g.lineTo(o.bx[i]-o.nx[i]*o.hw[i],o.by[i]-o.ny[i]*o.hw[i])}g.stroke();
  var chew=Math.sin(t*14)*.5+.5;g.strokeStyle='rgba(60,46,20,'+.75*(1-rk)+')';g.lineWidth=1.2;g.beginPath();g.arc(o.bx[7]+o.nx[7]*rw*.14*(1+chew),o.by[7]+o.ny[7]*rw*.14*(1+chew),rw*.2,ra-1.2,ra+1.2);g.arc(o.bx[7]-o.nx[7]*rw*.14*(1+chew),o.by[7]-o.ny[7]*rw*.14*(1+chew),rw*.2,ra+PI-1.2,ra+PI+1.2);g.stroke();
  g.fillStyle='rgba(170,40,30,'+.8*(1-rk*.7)+')';g.beginPath();g.arc(o.bx[8]+o.nx[8]*rw*.22,o.by[8]+o.ny[8]*rw*.22,1.1,0,TAU);g.arc(o.bx[8]-o.nx[8]*rw*.22,o.by[8]-o.ny[8]*rw*.22,1.1,0,TAU);g.fill();
  g.restore();tube(g,o,10);g.lineWidth=1;g.strokeStyle=col([70,64,40],[70,46,20],rk,.7);g.stroke();
  var cor=sm(.78,.97,e)*(1-sm(0,.25,s.wl));
  if(cor>.02){var hxr=o.bx[9]+rdx*rw*.5,hyr=o.by[9]+rdy*rw*.5,cr=rw*.56;g.globalAlpha=cor;
   for(j=-1;j<2;j+=2){x=hxr-rdy*cr*1.02*j;y=hyr+rdx*cr*1.02*j;g.fillStyle='rgba(244,240,220,.6)';g.strokeStyle='rgba(70,64,40,.6)';g.lineWidth=.9;g.beginPath();g.arc(x,y,cr,0,TAU);g.fill();g.stroke();
    g.beginPath();for(i=0;i<12;i++){a=i/12*TAU+t*7*j;var cl2=cr*(1.28+.14*Math.sin(i*2.1+t*20));g.moveTo(x+Math.cos(a)*cr,y+Math.sin(a)*cr);g.lineTo(x+Math.cos(a+.5*j)*cl2,y+Math.sin(a+.5*j)*cl2)}g.strokeStyle='rgba(60,56,36,.6)';g.lineWidth=.8;g.stroke()}
   g.fillStyle='rgba(56,60,30,.55)';for(i=0;i<8;i++){q=(t*.45+i/8)%1;j=i&1?1:-1;a=ra+j*(2.3-q*2.6);r=cr*(4.2-3.2*q);g.globalAlpha=cor*sm(0,.25,q)*(1-sm(.85,1,q));g.fillRect(hxr-rdy*cr*j+Math.cos(a)*r,hyr+rdx*cr*j+Math.sin(a)*r,1.6,1.6)}
   g.globalAlpha=1}

  /* ---- the water bear ---- */
  var shadow=HSM.blob('rgba(54,62,34,1)',32);g.globalAlpha=.085;for(i=2;i<M-1;i+=3){r=s.hw[i]*1.75+W*.3;g.drawImage(shadow,s.bx[i]-r+W*.2,s.by[i]-r+W*.3,r*2,r*2)}g.globalAlpha=1;
  /* legs: each foot stays planted while the body passes, then swings forward */
  var LL=s.LL,lf=1-.97*k,lg,legs=s.legs,tips=s.tipbuf||(s.tipbuf=new Float32Array(48));
  g.save();g.beginPath();g.rect(0,0,w,h);tube(g,s,M,1);g.clip('evenodd');
  g.beginPath();
  for(i=0;i<8;i++){lg=legs[i];P(s,lg.u,lg.sd*lg.v0);var Bx=s.qx,By=s.qy,Nx=s.qnx*lg.sd,Ny=s.qny*lg.sd,Fx=-s.qny,Fy=s.qnx; /* F points to the head */
   var rx,ry;if(lg.hind){rx=Bx+(-Fx*.9+Nx*.5)*LL*.78;ry=By+(-Fy*.9+Ny*.5)*LL*.78}else{rx=Bx+(Nx*.88+Fx*.3)*LL*.78;ry=By+(Ny*.88+Fy*.3)*LL*.78}
   if(lg.fx==null){lg.fx=rx+Fx*s.stride*(.42-((lg.off+.5)%1)*.9);lg.fy=ry+Fy*s.stride*(.42-((lg.off+.5)%1)*.9);lg.c=Math.floor(s.ph-lg.off)}
   x=lg.fx-Bx;y=lg.fy-By;var dist=Math.sqrt(x*x+y*y),cyc=Math.floor(s.ph-lg.off);
   var ex0=lg.fx-rx,ey0=lg.fy-ry;
   if(lg.sw<0&&(cyc!==lg.c||dist>LL*1.35||ex0*ex0+ey0*ey0>s.stride*s.stride*.6)){lg.c=cyc;lg.sw=0;lg.sx=lg.fx;lg.sy=lg.fy}
   if(lg.sw>=0){lg.sw+=dt/.42;q=sm(0,1,lg.sw);lg.fx=mix(lg.sx,rx+Fx*s.stride*.42*vr,q);lg.fy=mix(lg.sy,ry+Fy*s.stride*.42*vr,q);if(lg.sw>=1)lg.sw=-1;x=lg.fx-Bx;y=lg.fy-By;dist=Math.sqrt(x*x+y*y)}
   var len=C(dist,LL*.62,LL)*lf*(lg.sw>=0?1-.16*Math.sin(PI*lg.sw):1);dist=dist||1;var dx=x/dist,dy=y/dist;
   if(k>0){dx=mix(dx,lg.hind?-Fx*.8+Nx*.6:Nx,k);dy=mix(dy,lg.hind?-Fy*.8+Ny*.6:Ny,k);q=Math.sqrt(dx*dx+dy*dy)||1;dx/=q;dy/=q}
   var Tx=Bx+dx*len,Ty=By+dy*len,bw=W*.4*(1-.5*k),tw=W*.23*(1-.5*k),ox=-dy,oy=dx;
   g.moveTo(Bx-ox*bw,By-oy*bw);g.quadraticCurveTo((Bx+Tx)/2-ox*(bw+tw)*.56,(By+Ty)/2-oy*(bw+tw)*.56,Tx-ox*tw,Ty-oy*tw);
   g.bezierCurveTo(Tx-ox*tw+dx*tw*1.25,Ty-oy*tw+dy*tw*1.25,Tx+ox*tw+dx*tw*1.25,Ty+oy*tw+dy*tw*1.25,Tx+ox*tw,Ty+oy*tw);
   g.quadraticCurveTo((Bx+Tx)/2+ox*(bw+tw)*.56,(By+Ty)/2+oy*(bw+tw)*.56,Bx+ox*bw,By+oy*bw);g.closePath();
   j=i*6;tips[j]=Tx+dx*tw*.8;tips[j+1]=Ty+dy*tw*.8;tips[j+2]=dx;tips[j+3]=dy;tips[j+4]=tw;tips[j+5]=len}
  g.lineJoin='round';g.lineWidth=3.5;g.strokeStyle='rgba(255,255,250,.3)';g.stroke();
  g.fillStyle=col([232,227,198],[168,130,72],k,mix(.62,.92,k));g.fill();g.lineWidth=W*.2;g.save();g.clip();g.strokeStyle=col([110,104,60],[90,60,26],k,.2);g.stroke();g.restore();
  g.lineWidth=1.1;g.strokeStyle=col([70,64,40],[70,46,20],k,.75);g.stroke();
  g.beginPath();   /* claws: two double hooks on each foot; a fold across each leg */
  for(i=0;i<8;i++){j=i*6;x=tips[j];y=tips[j+1];dx=tips[j+2];dy=tips[j+3];tw=tips[j+4];var cl=W*.25;
   for(var n=0;n<4;n++){a=[-.95,-.35,.35,.95][n];var sx=x-dy*a*tw*.6,sy=y+dx*a*tw*.6,ca=Math.cos(a*.7),sa=Math.sin(a*.7),ex=dx*ca-dy*sa,ey=dy*ca+dx*sa,hk=(n&1?-1:1)*.7;
    g.moveTo(sx,sy);g.quadraticCurveTo(sx+ex*cl*.8,sy+ey*cl*.8,sx+ex*cl*.85-ey*cl*hk*.5,sy+ey*cl*.85+ex*cl*hk*.5)}
   var mx=x-dx*tips[j+5]*.5,my=y-dy*tips[j+5]*.5;g.moveTo(mx-dy*tw*1.2,my+dx*tw*1.2);g.quadraticCurveTo(mx+dx*tw*.3,my+dy*tw*.3,mx+dy*tw*1.2,my-dx*tw*1.2)}
  g.lineCap='round';g.lineWidth=1.05;g.strokeStyle='rgba(54,44,22,'+.8*(1-k*.9)+')';g.stroke();g.restore();
  /* body */
  tube(g,s,M);g.lineWidth=5;g.strokeStyle='rgba(255,255,250,.42)';g.stroke();
  g.fillStyle=col([238,233,206],[176,138,78],k,mix(.6,.94,k));g.fill();
  g.save();g.clip();
  g.strokeStyle=col([112,108,62],[92,60,24],k,.085);g.lineWidth=W*1.1;g.stroke();g.lineWidth=W*.62;g.stroke();g.lineWidth=W*.26;g.strokeStyle=col([112,108,62],[92,60,24],k,.16);g.stroke();
  /* the gut, coloured by its last meal of moss cells, squeezing slowly */
  var gA=HSM.blob('rgba(96,124,38,1)',24,.42),gB=HSM.blob('rgba(134,110,42,1)',24,.42);
  for(i=0;i<13;i++){P(s,.27+i*.045,.16*Math.sin(i*2.3));r=s.qh*(.6+.1*Math.sin(t*1.3-i*.9)*(1-k));g.globalAlpha=.56-.2*k;g.drawImage(i%3?gA:gB,s.qx-r,s.qy-r,r*2,r*2)}
  g.globalAlpha=1;g.fillStyle='rgba(44,64,16,'+(.5-.25*k)+')';g.beginPath();for(i=0;i<s.gran.length;i++){q=s.gran[i];P(s,q.u,q.v*.6);g.moveTo(s.qx+W*q.r,s.qy);g.arc(s.qx,s.qy,W*q.r,0,TAU)}g.fill();
  /* storage cells, loose in the body fluid, sloshing with each step */
  for(i=0;i<s.cells.length;i++){q=s.cells[i];P(s,q.u+(.014*Math.sin(t*.9+q.ph)+.012*Math.sin(TAU*s.ph+q.ph))*(1-k),q.v+.06*Math.sin(t*.7+q.ph*2)*(1-k));r=W*q.r;g.globalAlpha=.7-.4*k;g.drawImage(s.sc,s.qx-r,s.qy-r,r*2,r*2)}
  g.globalAlpha=1;
  /* body rings; in the tun they deepen into folds */
  g.beginPath();var seg=[.19,.37,.57,.77,.89];for(i=0;i<5;i++){P(s,seg[i],-1);x=s.qx;y=s.qy;g.moveTo(x,y);P(s,seg[i]+.035,0);var mx2=s.qx,my2=s.qy;P(s,seg[i],1);g.quadraticCurveTo(mx2,my2,s.qx,s.qy)}
  g.lineWidth=1;g.strokeStyle=col([90,84,50],[80,50,20],k,.2+.2*k);g.stroke();
  if(k>.02){g.beginPath();for(i=0;i<11;i++){u=.08+i*.084;P(s,u,-1);g.moveTo(s.qx,s.qy);P(s,u+.03*Math.sin(i*1.7),0);x=s.qx;y=s.qy;P(s,u,1);g.quadraticCurveTo(x,y,s.qx,s.qy)}
   g.lineWidth=1.3;g.strokeStyle='rgba(74,44,14,'+.42*k+')';g.stroke();g.lineWidth=.7;g.translate(1.2,1.4);g.strokeStyle='rgba(255,236,190,'+.3*k+')';g.stroke();g.translate(-1.2,-1.4)}
  /* mouth tube, stylets, the pumping bulb of the pharynx, eyespots */
  var ha=1-.9*k;
  P(s,.012,0);x=s.qx;y=s.qy;P(s,.135,0);var ux=s.qx,uy=s.qy;
  g.lineCap='butt';g.strokeStyle='rgba(66,58,36,'+.7*ha+')';g.lineWidth=W*.085;g.beginPath();g.moveTo(x,y);g.lineTo(ux,uy);g.stroke();g.strokeStyle='rgba(250,246,226,'+.75*ha+')';g.lineWidth=W*.035;g.stroke();
  g.strokeStyle='rgba(66,58,36,'+.5*ha+')';g.lineWidth=.8;g.beginPath();for(j=-1;j<2;j+=2){P(s,.04,j*.1);g.moveTo(s.qx,s.qy);P(s,.09,j*.2);x=s.qx;y=s.qy;P(s,.14,j*.34);g.quadraticCurveTo(x,y,s.qx,s.qy)}g.stroke();
  P(s,.175,0);r=W*.27*(1+.07*Math.sin(t*5.5)*s.sw*(1-k));g.fillStyle='rgba(222,214,176,'+.7*ha+')';g.strokeStyle='rgba(84,76,46,'+.7*ha+')';g.lineWidth=1.1;g.beginPath();g.arc(s.qx,s.qy,r,0,TAU);g.fill();g.stroke();
  g.fillStyle='rgba(255,255,244,'+.5*ha+')';g.beginPath();g.arc(s.qx-r*.3,s.qy-r*.3,r*.34,0,TAU);g.fill();
  g.strokeStyle='rgba(70,60,36,'+.6*ha+')';g.lineWidth=1.2;g.beginPath();for(j=-1;j<2;j++){g.moveTo(s.qx+s.qnx*j*r*.36-s.qny*r*.3,s.qy+s.qny*j*r*.36+s.qnx*r*.3);g.lineTo(s.qx+s.qnx*j*r*.36+s.qny*r*.3,s.qy+s.qny*j*r*.36-s.qnx*r*.3)}g.stroke();
  g.fillStyle='rgba(18,14,10,'+(.9-.6*k)+')';g.beginPath();for(j=-1;j<2;j+=2){P(s,.1,j*.56);g.moveTo(s.qx+W*.075,s.qy);g.arc(s.qx,s.qy,W*.075,0,TAU)}g.fill();
  /* gloss: light from the upper left along the rounded back */
  g.lineCap='round';g.lineJoin='round';g.beginPath();for(i=2;i<M-2;i++){x=s.bx[i]-s.hw[i]*.34;y=s.by[i]-s.hw[i]*.42;i===2?g.moveTo(x,y):g.lineTo(x,y)}
  g.strokeStyle='rgba(255,255,250,'+(.13-.05*k)+')';g.lineWidth=W*.62;g.stroke();g.lineWidth=W*.26;g.strokeStyle='rgba(255,255,250,'+(.2-.1*k)+')';g.stroke();
  g.restore();
  tube(g,s,M);g.lineWidth=1.35;g.strokeStyle=col([66,60,38],[66,42,18],k,.82);g.stroke();
  g.lineCap='butt';

  /* ---- field stop, scale, captions ---- */
  HSM.vignette(g,w,h,.34);
  var bar=L*.3;g.fillStyle='rgba(6,10,8,.6)';g.fillRect(w-bar-15,h-19,bar+2,4);g.fillStyle='rgba(255,255,255,.95)';g.fillRect(w-bar-14,h-18,bar,2);
  HSM.label(g,w-14,h-29,'0.1 mm','right');
  HSM.label(g,12,18,s.k>.55?'a tun: it can wait like this for years':'about a third of a millimetre long');
 }});
})();
