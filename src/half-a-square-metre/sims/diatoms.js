/* The living skin on wet rock, seen from above through a film of water.
   Diatoms (single cells in ribbed glass shells, golden-brown chloroplasts inside) glide in
   straight, jerky runs and reverse. Chains of blue-green bacteria sway and creep. Everything
   sits in a faint translucent slime over dark mineral grains. In light the film makes oxygen
   faster than the water can carry it off: bubbles form, grow, let go and rise toward us. */
HSM.sim('diatoms',{
 slider:{label:'Light',lo:'shade',hi:'sun',value:.6},
 setup:function(w,h,R){
  var U=w/390,i,k,q;
  function cv(cw,ch){var c=document.createElement('canvas');c.width=Math.max(2,Math.round(cw));c.height=Math.max(2,Math.round(ch));return c}
  /* ---- the rock: dark, grainy, a few bright crystals ---- */
  var gd=cv(w,h),g=gd.getContext('2d');q=g.createLinearGradient(0,0,w,h);q.addColorStop(0,'#2b2a26');q.addColorStop(.5,'#1e1d1c');q.addColorStop(1,'#151719');g.fillStyle=q;g.fillRect(0,0,w,h);
  for(i=0;i<520;i++){var x=R()*w,y=R()*h,sz=(2+R()*R()*26)*U,tn=R();g.globalAlpha=tn<.45?.25+R()*.35:tn<.9?.05+R()*.1:.12+R()*.2;
   g.save();g.translate(x,y);g.rotate(R()*3.14);g.drawImage(HSM.blob(tn<.45?'rgba(6,7,9,1)':tn<.75?'rgba(150,132,104,1)':tn<.9?'rgba(96,116,120,1)':'rgba(236,226,200,1)',24,tn<.9?.55:.35),-sz,-sz*.65,sz*2,sz*1.3);g.restore()}
  for(i=0;i<26;i++){x=R()*w;y=R()*h;sz=(1.5+R()*3.5)*U;g.globalAlpha=.12+R()*.2;g.fillStyle=R()<.5?'#e8dcc0':'#b7c8cc';g.save();g.translate(x,y);g.rotate(R()*3.14);
   g.beginPath();g.moveTo(-sz,0);g.lineTo(-sz*.2,-sz*.55);g.lineTo(sz,-sz*.15);g.lineTo(sz*.4,sz*.6);g.closePath();g.fill();g.restore()}
  g.globalAlpha=1;
  /* the film itself, a little out of focus below the gliders: golden and green patches of small cells, and stay-at-home diatoms */
  for(i=0;i<34;i++){x=R()*w;y=R()*h;sz=(26+R()*60)*U;g.globalAlpha=.07+R()*.1;g.drawImage(HSM.blob(R()<.55?'rgba(196,150,50,1)':'rgba(96,170,90,1)',32,.25),x-sz,y-sz*.75,sz*2,sz*1.5)}
  for(i=0;i<70;i++){var px=R()*w,py=R()*h,gold=R()<.5,nn=8+Math.floor(R()*26),sp0=(8+R()*22)*U;
   for(k=0;k<nn;k++){var aa=R()*6.283,rr0=Math.sqrt(R())*sp0,cr=(1.1+R()*1.7)*U;g.globalAlpha=.2+R()*.32;
    g.drawImage(HSM.blob(gold?'rgba(214,160,54,1)':'rgba(120,196,96,1)',8,.5),px+Math.cos(aa)*rr0-cr,py+Math.sin(aa)*rr0-cr,cr*2,cr*2)}}
  for(i=0;i<38;i++){x=R()*w;y=R()*h;var dl=(6+R()*8)*U,dw=dl*(.17+R()*.1);g.save();g.translate(x,y);g.rotate(R()*3.14);g.globalAlpha=.22+R()*.22;
   g.beginPath();g.moveTo(-dl,0);g.quadraticCurveTo(0,-dw*2,dl,0);g.quadraticCurveTo(0,dw*2,-dl,0);g.fillStyle='rgba(176,124,40,.7)';g.fill();g.lineWidth=1.3*U;g.strokeStyle='rgba(224,240,240,.95)';g.stroke();g.beginPath();g.moveTo(-dl*.8,0);g.lineTo(dl*.8,0);g.lineWidth=.6*U;g.stroke();g.restore()}
  g.globalAlpha=1;
  /* ---- the slime: clouds, strands, and the countless ordinary bacteria living in it ---- */
  function slime(seed,fine){var m=30*U,c=cv((w+m*2)*.5,(h+m*2)*.5),g=c.getContext('2d'),RR=HSM.rnd(seed);g.scale(.5,.5);
   for(var i=0;i<46;i++){var x=RR()*(w+m*2),y=RR()*(h+m*2),r=(40+RR()*90)*U;g.globalAlpha=.035+RR()*.06;
    g.drawImage(HSM.blob(RR()<.6?'rgba(170,200,150,1)':'rgba(200,190,140,1)',32,.3),x-r,y-r*.7,r*2,r*1.4)}
   g.lineCap='round';
   for(i=0;i<34;i++){x=RR()*(w+m*2);y=RR()*(h+m*2);var a=RR()*6.283,cc=0;g.beginPath();g.moveTo(x,y);
    for(var s=0;s<40;s++){cc+=(RR()-.5)*.14;cc*=.92;a+=cc;x+=Math.cos(a)*7*U;y+=Math.sin(a)*7*U;g.lineTo(x,y)}
    g.globalAlpha=.04+RR()*.06;g.lineWidth=(1.5+RR()*5)*U;g.strokeStyle='rgb(206,226,200)';g.stroke()}
   if(fine)for(i=0;i<620;i++){x=RR()*(w+m*2);y=RR()*(h+m*2);a=RR()*6.283;var l=(1.5+RR()*3)*U;g.globalAlpha=.1+RR()*.24;g.lineWidth=1.5*U;g.strokeStyle=RR()<.7?'rgb(214,222,196)':'rgb(150,214,190)';
    g.beginPath();g.moveTo(x,y);g.lineTo(x+Math.cos(a)*l,y+Math.sin(a)*l);g.stroke()}
   return c}
  /* ---- light rippling down through the water ---- */
  var ca=cv(w*.4,h*.4),cg=ca.getContext('2d'),RC=HSM.rnd(17);cg.scale(.25,.25);cg.lineCap='round';
  for(i=0;i<26;i++){x=RC()*w*1.6;y=RC()*h*1.6;var a=RC()*6.283,cc=0;cg.beginPath();cg.moveTo(x,y);
   for(k=0;k<30;k++){cc+=(RC()-.5)*.3;cc*=.9;a+=cc;x+=Math.cos(a)*16*U;y+=Math.sin(a)*16*U;cg.lineTo(x,y)}
   cg.globalAlpha=.1+RC()*.14;cg.lineWidth=(5+RC()*12)*U;cg.strokeStyle='rgb(200,240,235)';cg.stroke()}
  /* ---- diatom shells ---- */
  function diatom(len,wid,ea,eb,seed){var Q=2,pad=4,c=cv((len+pad*2)*Q,(wid+pad*2)*Q),g=c.getContext('2d'),RR=HSM.rnd(seed),hl=len/2,hw=wid/2,n;g.scale(Q,Q);g.translate(hl+pad,hw+pad);
   function half(u){return hw*Math.pow(Math.max(0,1-Math.pow(Math.abs(u),ea)),eb)}
   function shape(sc){g.beginPath();for(n=0;n<=60;n++){var u=-1+n/30;if(n<=30){}; }
    for(n=0;n<=48;n++){var u1=-1+n/24;n?g.lineTo(u1*hl*(sc>.95?1:.97),-half(u1)*sc):g.moveTo(u1*hl*(sc>.95?1:.97),-half(u1)*sc)}
    for(n=48;n>=0;n--){var u2=-1+n/24;g.lineTo(u2*hl*(sc>.95?1:.97),half(u2)*sc)}g.closePath()}
   shape(1);g.fillStyle='rgba(176,206,210,.2)';g.fill();
   g.save();shape(1);g.clip();
   /* two chloroplast plates, one under each long side, parted round the nucleus in the middle */
   for(var sd=-1;sd<=1;sd+=2)for(var hf=-1;hf<=1;hf+=2){var q=g.createLinearGradient(0,sd*hw*.1,0,sd*hw);q.addColorStop(0,'rgba(236,178,52,.95)');q.addColorStop(.6,'rgba(196,128,26,.95)');q.addColorStop(1,'rgba(120,72,14,.95)');
    g.fillStyle=q;g.beginPath();var x0=hf*hl*.07,x1=hf*hl*(.74+RR()*.1);g.moveTo(x0,sd*hw*.16);
    g.bezierCurveTo(x0+(x1-x0)*.3,sd*hw*(.02+RR()*.12),x0+(x1-x0)*.75,sd*hw*.1,x1,sd*half(x1/hl)*.5);
    g.lineTo(x1,sd*hw*1.2);g.lineTo(x0,sd*hw*1.2);g.closePath();g.fill()}
   for(n=0;n<len*1.4;n++){var gx=(RR()*2-1)*hl*.8,gy=(RR()*2-1)*hw;if(Math.abs(gy)<hw*.2)continue;g.fillStyle=RR()<.5?'rgba(90,50,8,.22)':'rgba(255,214,110,.2)';g.beginPath();g.arc(gx,gy,.5+RR()*.8,0,6.283);g.fill()}
   /* oil droplets: the cell's food store */
   for(n=0;n<4;n++){var ox=(n<2?-1:1)*hl*(.25+RR()*.3),oy=(RR()-.5)*hw*.5,or=hw*(.16+RR()*.12);q=g.createRadialGradient(ox-or*.3,oy-or*.3,0,ox,oy,or);
    q.addColorStop(0,'rgba(255,252,220,.95)');q.addColorStop(.7,'rgba(255,226,140,.55)');q.addColorStop(1,'rgba(120,80,20,.5)');g.fillStyle=q;g.beginPath();g.arc(ox,oy,or,0,6.283);g.fill()}
   /* fine ribbing of the glass */
   g.lineWidth=.45;g.strokeStyle='rgba(232,246,250,.42)';g.beginPath();
   for(var sx=-hl;sx<hl;sx+=1.5){var tl=sx/hl*hw*.25;g.moveTo(sx,-hw*.16);g.lineTo(sx+tl,-hw);g.moveTo(sx,hw*.16);g.lineTo(sx+tl,hw)}g.stroke();
   g.lineWidth=.45;g.strokeStyle='rgba(10,20,26,.28)';g.beginPath();for(sx=-hl+.75;sx<hl;sx+=1.5){tl=sx/hl*hw*.25;g.moveTo(sx,-hw*.16);g.lineTo(sx+tl,-hw);g.moveTo(sx,hw*.16);g.lineTo(sx+tl,hw)}g.stroke();
   g.restore();
   /* the slit (raphe) the cell glides on, with its central knot */
   g.lineCap='round';g.lineWidth=.8;g.strokeStyle='rgba(240,250,255,.7)';g.beginPath();g.moveTo(-hl*.9,0);g.lineTo(-hl*.06,0);g.moveTo(hl*.06,0);g.lineTo(hl*.9,0);g.stroke();
   g.fillStyle='rgba(240,250,255,.6)';g.beginPath();g.arc(0,0,hw*.13,0,6.283);g.fill();
   /* glass walls: seen edge-on at the rim they are bright */
   shape(1);g.lineWidth=2.2;g.strokeStyle='rgba(8,16,20,.55)';g.stroke();g.lineWidth=1.15;g.strokeStyle='rgba(226,244,248,.92)';g.stroke();
   shape(.8);g.lineWidth=.5;g.strokeStyle='rgba(226,244,248,.3)';g.stroke();
   g.save();shape(1);g.clip();q=g.createLinearGradient(0,-hw,0,hw);q.addColorStop(0,'rgba(255,255,255,.34)');q.addColorStop(.3,'rgba(255,255,255,0)');q.addColorStop(.8,'rgba(0,0,0,0)');q.addColorStop(1,'rgba(0,10,20,.3)');g.fillStyle=q;g.fillRect(-hl,-hw,len,wid);g.restore();
   return{c:c,len:len,wid:wid,pad:pad}}
  var types=[diatom(104*U,21*U,3.2,.62,3),diatom(78*U,19*U,2,.85,5),diatom(54*U,14*U,2,.8,8),diatom(92*U,9.5*U,2,1.1,9),diatom(42*U,19*U,2,.55,12)];
  var ds=[];
  function addD(ty,hx,hy,rg){var a=R()*6.283;ds.push({ty:ty,x:hx+(R()-.5)*rg,y:hy+(R()-.5)*rg,hx:hx,hy:hy,rg:rg,a:a,dir:R()<.5?1:-1,v:(13+R()*9)*U*(ty===4?.25:1),run:1.5+R()*6,pause:0,f:1.1+R()*1.4,ph:R()*6.283,cv:(R()-.5)*.05,tr:[],td:0,sp:0})}
  addD(0,w*.36,h*.7,90*U);
  var lay=[1,2,3,1,2,4,0,3,2,1,4,2];for(i=0;i<lay.length;i++)addD(lay[i],w*(.08+R()*.84),h*(.08+R()*.84),(120+R()*120)*U);
  /* ---- blue-green bacteria: bead chains, with the odd bigger pale cell that fixes nitrogen ---- */
  function bead(c0,c1,c2,rim){var S=16,c=cv(S*2,S*2),g=c.getContext('2d'),q=g.createRadialGradient(S*.75,S*.65,S*.05,S,S,S*.95);
   q.addColorStop(0,c0);q.addColorStop(.6,c1);q.addColorStop(.92,c2);q.addColorStop(1,'rgba(4,30,34,0)');g.fillStyle=q;g.beginPath();g.arc(S,S,S,0,6.283);g.fill();
   if(rim){g.lineWidth=1.6;g.strokeStyle=rim;g.beginPath();g.arc(S,S,S*.78,0,6.283);g.stroke()}return c}
  var bd=bead('#b6fbe0','#2fb39c','#126a74'),bd2=bead('#9af0e6','#249fa6','#0f5c78'),het=bead('#f4f6c8','#b9c98a','#6f8a5c','rgba(250,252,220,.7)');
  var fl=[],fdef=[[.22,.2,36],[.8,.5,30],[.55,.88,26],[.12,.6,22],[.7,.1,20],[.9,.82,24]];
  for(i=0;i<fdef.length;i++){var N=fdef[i][2],d=5.1*U,pts=[],hx=fdef[i][0]*w,hy=fdef[i][1]*h,a2=R()*6.283,cc2=0,hp=3+Math.floor(R()*9);x=hx;y=hy;
   for(k=0;k<N;k++){pts.push(x,y);cc2+=(R()-.5)*.12;cc2*=.9;a2+=cc2;x-=Math.cos(a2)*d;y-=Math.sin(a2)*d}
   fl.push({pts:pts,N:N,d:d,a:Math.atan2(pts[1]-pts[3],pts[0]-pts[2]),c:0,ph:R()*6.283,hp:hp,v:(2.2+R()*2)*U,hx:hx,hy:hy,alt:i%2})}
  /* ---- oxygen bubble: a clear ball with a dark refracting edge, a window of light, a glow inside the far wall ---- */
  var S=64,bb=cv(S*2,S*2);g=bb.getContext('2d');q=g.createRadialGradient(S,S,0,S,S,S);
  q.addColorStop(0,'rgba(220,244,250,.1)');q.addColorStop(.6,'rgba(200,236,246,.14)');q.addColorStop(.82,'rgba(60,100,114,.4)');q.addColorStop(.92,'rgba(10,26,34,.66)');q.addColorStop(.955,'rgba(226,248,255,.95)');q.addColorStop(1,'rgba(214,244,252,0)');
  g.fillStyle=q;g.fillRect(0,0,S*2,S*2);
  g.save();g.beginPath();g.arc(S,S,S*.9,0,6.283);g.clip();
  q=g.createRadialGradient(S*1.3,S*1.36,S*.1,S*1.1,S*1.14,S*.8);q.addColorStop(0,'rgba(190,246,255,.0)');q.addColorStop(.72,'rgba(170,240,255,.0)');q.addColorStop(.9,'rgba(180,244,255,.6)');q.addColorStop(1,'rgba(180,244,255,0)');g.fillStyle=q;g.fillRect(0,0,S*2,S*2);
  q=g.createRadialGradient(S*.62,S*.56,0,S*.62,S*.56,S*.36);q.addColorStop(0,'rgba(255,255,255,.95)');q.addColorStop(.3,'rgba(255,255,255,.5)');q.addColorStop(1,'rgba(255,255,255,0)');g.fillStyle=q;g.fillRect(0,0,S*2,S*2);
  g.restore();
  var bub=[];
  for(i=0;i<34;i++){var b={x:i?w*(.06+R()*.88):w*.74,y:i?h*(.06+R()*.88):h*.2,thr:i?(i<4?.1:.12+R()*.83):0,rm:(i?(i>25?3+R()*3:9+R()*12):15)*U,rate:.6+R()*.8,r:-1,z:0,wait:0,ph:R()*6.283,seed:R()};if(i){var ex=b.x-w*.74,ey=b.y-h*.2;if(ex*ex+ey*ey<3600*U*U)b.y+=70*U}bub.push(b)}
  return{U:U,gd:gd,sl:slime(21,true),sl2:slime(44,false),ca:ca,types:types,ds:ds,fl:fl,bd:bd,bd2:bd2,het:het,bb:bb,bub:bub,R:R}},

 frame:function(s,g,w,h,t,dt,ui){
  var L=ui.a==null?.6:ui.a,C=HSM.clamp,U=s.U,i,k,m=30*U,live=.3+1.1*L;
  g.drawImage(s.gd,0,0,w,h);
  /* slime under everything, breathing a little with the water */
  var sx=Math.sin(t*.23)*5*U,sy=Math.cos(t*.19)*4*U;
  g.drawImage(s.sl,-m+sx,-m+sy,w+m*2,h+m*2);
  /* the trails of slime the diatoms leave behind them */
  g.lineCap='round';g.lineJoin='round';g.strokeStyle='rgba(206,230,214,.085)';
  for(i=0;i<s.ds.length;i++){var d=s.ds[i],tr=d.tr;if(tr.length<4)continue;g.lineWidth=s.types[d.ty].wid*.42;g.beginPath();g.moveTo(d.x,d.y);for(k=tr.length-2;k>=0;k-=2)g.lineTo(tr[k],tr[k+1]);g.stroke()}
  /* blue-green bacteria: the head creeps, the chain follows, the whole thread sways */
  for(i=0;i<s.fl.length;i++){var f=s.fl[i],P=f.pts,N=f.N;
   f.c+=(s.R()-.5)*.5*dt;f.c*=1-dt*.5;f.a+=f.c*dt*3;
   var bx=f.hx-P[0],by=f.hy-P[1],bl=Math.sqrt(bx*bx+by*by);if(bl>70*U){var want=Math.atan2(by,bx);f.a+=Math.sin(want-f.a)*dt*.5}
   var v=f.v*live*dt;P[0]+=Math.cos(f.a)*v;P[1]+=Math.sin(f.a)*v;
   for(k=1;k<N;k++){var dx=P[k*2]-P[k*2-2],dy=P[k*2+1]-P[k*2-1],dl=Math.sqrt(dx*dx+dy*dy)||1,e=f.d/dl;P[k*2]=P[k*2-2]+dx*e;P[k*2+1]=P[k*2-1]+dy*e}
   /* sheath */
   var amp=1.9*U,r=3.3*U,X=f.X||(f.X=new Float32Array(N*2));
   for(k=0;k<N;k++){var k0=Math.max(0,k-1),k1=Math.min(N-1,k+1),nx=-(P[k1*2+1]-P[k0*2+1]),ny=P[k1*2]-P[k0*2],nl=Math.sqrt(nx*nx+ny*ny)||1,o=Math.sin(t*(.7+.5*L)-k*.33+f.ph)*amp*Math.min(1,k/5+.3);
    X[k*2]=P[k*2]+nx/nl*o;X[k*2+1]=P[k*2+1]+ny/nl*o}
   g.beginPath();g.moveTo(X[0],X[1]);for(k=1;k<N;k++)g.lineTo(X[k*2],X[k*2+1]);g.lineWidth=r*3.2;g.strokeStyle='rgba(120,220,210,.09)';g.stroke();g.lineWidth=r*2.2;g.strokeStyle='rgba(4,20,24,.3)';g.stroke();
   var sp=f.alt?s.bd2:s.bd;
   for(k=N-1;k>=0;k--){var ht=(k%13===f.hp),rr=ht?r*1.38:r*(k===0?.9:1);g.drawImage(ht?s.het:sp,X[k*2]-rr,X[k*2+1]-rr,rr*2,rr*2)}}
  /* diatoms */
  var sh=HSM.blob('rgba(0,4,8,1)',24,.35);
  for(i=0;i<s.ds.length;i++){d=s.ds[i];var T=s.types[d.ty];
   if(d.pause>0){d.pause-=dt;d.sp+=(0-d.sp)*Math.min(1,dt*8)}
   else{d.run-=dt*live;var jerk=Math.max(0,.55+.75*Math.sin(t*d.f+d.ph)+.5*Math.sin(t*d.f*2.9+d.ph*2))*(Math.sin(t*d.f*.37+d.ph*3)>-.75?1:0);
    d.sp+=(d.dir*d.v*live*jerk-d.sp)*Math.min(1,dt*9);
    var ox=d.x-d.hx,oy=d.y-d.hy,out=ox*ox+oy*oy>d.rg*d.rg||d.x<16*U||d.x>w-16*U||d.y<16*U||d.y>h-16*U,hd=d.dir*(Math.cos(d.a)*ox+Math.sin(d.a)*oy);
    if(d.x<16*U||d.x>w-16*U||d.y<16*U||d.y>h-16*U)hd=d.dir*(Math.cos(d.a)*(d.x-w/2)+Math.sin(d.a)*(d.y-h/2));
    if(d.run<0||(out&&hd>0)){d.dir=-d.dir;d.run=2+s.R()*7;d.pause=.35+s.R()*.5;d.a+=(s.R()-.5)*.5;d.cv=(s.R()-.5)*.05}}
   var st=d.sp*dt;d.x+=Math.cos(d.a)*st;d.y+=Math.sin(d.a)*st;d.a+=d.cv*st*.1;
   d.td+=Math.abs(st);if(d.td>5*U){d.td=0;d.tr.push(d.x,d.y);if(d.tr.length>44)d.tr.splice(0,2)}
   var hl=T.len/2+T.pad,hw=T.wid/2+T.pad;
   g.save();g.translate(d.x,d.y);g.rotate(d.a);
   g.globalAlpha=.5;g.drawImage(sh,-hl*1.04+2*U,-hw*1.25+3.5*U,hl*2.08,hw*2.5);
   g.globalAlpha=1;g.drawImage(T.c,-hl,-hw,hl*2,hw*2);
   g.globalCompositeOperation='lighter';g.globalAlpha=.07+.13*L;g.drawImage(HSM.blob('rgba(255,190,70,1)',24,.2),-hl*.9,-hw*1.3,hl*1.8,hw*2.6);g.globalCompositeOperation='source-over';
   g.restore()}
  g.globalAlpha=.55;g.drawImage(s.sl2,-m-sx*1.4,-m-sy*1.4,w+m*2,h+m*2);g.globalAlpha=1;
  /* oxygen: bubbles swell where the film is working, let go, and rise toward the surface (toward us) */
  var S2=HSM.blob('rgba(0,6,10,1)',24,.3),lab=null;
  for(i=0;i<s.bub.length;i++){var b=s.bub[i],act=L>=b.thr;
   if(b.r<0)b.r=act?b.rm*(.3+.68*b.seed)*C((L-b.thr)*3+.45,0,1):0;
   if(b.z>0){b.z+=dt/1.7;if(b.z>=1){b.z=0;b.r=0;b.wait=.5+s.R()*3}}
   else if(b.wait>0)b.wait-=dt;
   else if(act){b.r+=(.25+2.4*(L-b.thr*.5))*b.rate*U*dt/(1+b.r/(7*U));if(b.r<1.2*U)b.r=1.2*U;if(b.r>=b.rm)b.z=.001}
   else if(b.r>0)b.r=Math.max(0,b.r-.5*U*dt);
   if(b.r<=0)continue;
   var z=b.z,e2=z*z,wob=z>0?0:Math.sin(t*2.1+b.ph)*.35*U*HSM.smooth(.7,1,b.r/b.rm),br=b.r*(1+.55*e2),x=b.x+wob-e2*16*U+Math.sin(z*5+b.ph)*3*U*z,y=b.y-e2*26*U,al=1-HSM.smooth(.45,1,z);
   g.globalAlpha=.5*(1-z)*(.6+.4*L);var sr=b.r*(1.25+z*1.2);g.drawImage(S2,b.x+b.r*.28+z*8*U-sr,b.y+b.r*.36+z*12*U-sr,sr*2,sr*2);
   g.globalAlpha=al;g.drawImage(s.bb,x-br,y-br,br*2,br*2);
   if(!lab&&i===0)lab=[x,y,br,al]}
  g.globalAlpha=1;
  /* light through moving water */
  g.globalCompositeOperation='lighter';g.globalAlpha=.14+.6*L;
  var c1=Math.sin(t*.31)*18*U,c2=Math.cos(t*.27)*16*U;g.drawImage(s.ca,-40*U+c1,-40*U+c2,w*1.6,h*1.6);
  g.globalAlpha=.06+.36*L;g.drawImage(s.ca,-w*.5-c2*1.2,-h*.45+c1*.9,w*1.6,h*1.6);
  g.globalAlpha=1;if(L>.6){g.fillStyle='rgba(255,240,190,'+((L-.6)*.2)+')';g.fillRect(0,0,w,h)}
  g.globalCompositeOperation='source-over';
  g.fillStyle='rgba(2,8,14,'+(.32*(1-HSM.smooth(0,.7,L)))+')';g.fillRect(0,0,w,h);
  HSM.vignette(g,w,h,.45+.15*(1-L));
  d=s.ds[0];HSM.label(g,C(d.x,96,w-96),C(d.y+(18+Math.abs(Math.sin(d.a))*46)*U,14,h-12),'diatom: a cell in a glass box','center');
  b=s.bub[0];HSM.label(g,b.x+b.rm+8*U,b.y,'oxygen','left');
  f=s.fl[0];k=Math.floor(f.N*.45);HSM.label(g,C(f.pts[k*2]+10*U,8,w-126),C(f.pts[k*2+1]-13*U,14,h-12),'blue-green bacteria','left');
 }});
