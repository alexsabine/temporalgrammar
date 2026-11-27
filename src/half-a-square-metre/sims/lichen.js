/* Inside a lichen. A thin slice through a crust lichen on rock, light from above.
   Pale fungal threads (hyphae) branch, fuse and clasp round green algal cells, each with one big
   chloroplast. The algae make sugar (golden motes); it passes into the fungus where a thread
   clasps the cell and is carried away along the threads. Dry: everything shrinks, greys and stops,
   but nothing dies. Wet: within moments it swells, greens and the sugar flows again.
   ui.on true = dried out (the button then offers 'Wet it'). */
HSM.sim('lichen',{
 toggle:{off:'Dry it out',on:'Wet it'},
 setup:function(w,h,R){
  var U=w/390,rockY=h*.905,W=7.4*U,ST=4*U,i,j,n;
  function cv(cw,ch){var c=document.createElement('canvas');c.width=Math.max(2,Math.round(cw));c.height=Math.max(2,Math.round(ch));return c}
  /* ---- algal cells, scattered through the upper-middle of the slice ---- */
  var algae=[],tries=0;
  while(algae.length<13&&tries++<6000){var r=(20+R()*9)*U,x=w*(-.02+R()*1.04),y=h*(.27+R()*.47),ok=true;
   for(i=0;i<algae.length;i++){var dx=algae[i].x-x,dy=algae[i].y-y;if(dx*dx+dy*dy<Math.pow(algae[i].r+r+20*U,2)){ok=false;break}}
   if(ok)algae.push({x:x,y:y,r:r,rot:R()*6.283,v:algae.length%3,ph:R()*6.283,emit:R()*2,pulse:0})}
  /* ---- fungal threads: polylines with a fixed step, so sugar can ride along them ---- */
  var paths=[];
  function push(a){for(var q=0;q<algae.length;q++){var A=algae[q],dx=a.x-A.x,dy=a.y-A.y,d=Math.sqrt(dx*dx+dy*dy),m=A.r+W*.75;
    if(d<m){a.x=A.x+dx/d*m;a.y=A.y+dy/d*m;var tg=Math.atan2(dx,-dy);if(Math.cos(tg-a.ang)<0)tg+=Math.PI;a.ang=tg}}}
  function wander(x,y,ang,max,bias,bk,fuse,avoid,top){var pts=[x,y],a={x:x,y:y,ang:ang},c=0,link=null;
   for(var s=0;s<max;s++){c+=(R()-.5)*.11;c*=.9;a.ang+=c+Math.sin(bias-a.ang)*bk;a.x+=Math.cos(a.ang)*ST;a.y+=Math.sin(a.ang)*ST;if(avoid)push(a);
    if(fuse&&s>9){var hit=false;for(var p=0;p<paths.length&&!hit;p++){var P=paths[p].pts;for(var q=4;q<P.length-4;q+=2){var dx=P[q]-a.x,dy=P[q+1]-a.y;if(dx*dx+dy*dy<ST*ST*1.2){pts.push(P[q],P[q+1]);link={p:p,i:q/2};hit=true;break}}}
     if(hit)break}
    pts.push(a.x,a.y);
    if(a.x<-w*.12||a.x>w*1.12||a.y<(top||-h*.05)||a.y>rockY+14*U)break}
   return{pts:pts,link:link}}
  /* one thread for each alga: it arrives, wraps part-way round the cell, and leaves downward */
  for(i=0;i<algae.length;i++){var A=algae[i],a0=R()*6.283,dir=R()<.5?1:-1,span=2.3+R()*1.5,ar=A.r+W*.42,na=Math.ceil(span*ar/ST),arc=[];
   for(n=0;n<=na;n++){var aa=a0+dir*span*n/na;arc.push(A.x+Math.cos(aa)*ar,A.y+Math.sin(aa)*ar)}
   var inn=wander(arc[0],arc[1],a0-dir*Math.PI/2+(R()-.5)*.6,70,-Math.PI/2+(R()-.5)*.9,.06,false,true,h*.15).pts,pre=[];
   for(n=inn.length-2;n>=2;n-=2)pre.push(inn[n],inn[n+1]);
   var ae=a0+dir*span,out=wander(arc[arc.length-2],arc[arc.length-1],ae+dir*Math.PI/2,120,Math.PI/2+(R()-.5)*1.3,.035,true,true,h*.15);
   var pts=pre.concat(arc,out.pts.slice(2));
   paths.push({pts:pts,link:out.link,br:[],clasp:pre.length/2+Math.floor(na*.5),cl:0});A.path=paths.length-1}
  /* side branches that leave one thread and fuse with another */
  for(i=0;i<11;i++){var hp=Math.floor(R()*paths.length),H=paths[hp],hn=H.pts.length/2;if(hn<30)continue;var hi=8+Math.floor(R()*(hn-16));
   var ta=Math.atan2(H.pts[hi*2+3]-H.pts[hi*2+1],H.pts[hi*2+2]-H.pts[hi*2])+(R()<.5?1:-1)*(.7+R()*.6);
   var b=wander(H.pts[hi*2],H.pts[hi*2+1],ta,90,ta+(R()-.5),.02,true,true,h*.15);if(b.pts.length<16)continue;
   paths.push({pts:b.pts,link:b.link,br:[],cl:1});H.br.push({i:hi,p:paths.length-1})}
  /* the dense skin of threads at the top, and loose ones above the rock */
  var nmain=paths.length;
  for(i=0;i<5;i++){var lf2=R()<.5;paths.push({pts:wander(lf2?-w*.1:w*1.1,h*(.76+R()*.12),lf2?.1:Math.PI-.1,130,lf2?0:Math.PI,.03,false,true).pts,link:null,br:[],cl:1})}
  /* build the strokes once */
  var hy=new Path2D(),cx=new Path2D(),sep=new Path2D(),sep2=new Path2D();
  for(i=0;i<paths.length;i++){var P=paths[i].pts,T=paths[i].cl?cx:hy,SP=paths[i].cl?sep2:sep;T.moveTo(P[0],P[1]);for(n=2;n<P.length;n+=2)T.lineTo(P[n],P[n+1]);
   var gap=6+Math.floor(R()*4),off=Math.floor(R()*gap);
   for(n=off+2;n<P.length/2-2;n+=gap){var tx=P[n*2+2]-P[n*2-2],ty=P[n*2+3]-P[n*2-1],tl=Math.sqrt(tx*tx+ty*ty)||1,hw=W*(paths[i].cl?.34:.46);tx/=tl;ty/=tl;
    SP.moveTo(P[n*2]-ty*hw,P[n*2+1]+tx*hw);SP.lineTo(P[n*2]+ty*hw,P[n*2+1]-tx*hw)}}
  /* ---- out-of-focus depth: threads and cells further into the slice, drawn small and blown up ---- */
  function back(wet){var q=.25,c=cv(w*1.2*q,h*q),g=c.getContext('2d');g.scale(q,q);g.translate(w*.1,0);g.lineCap='round';g.lineJoin='round';
   var RR=HSM.rnd(wet?91:91);
   for(var k=0;k<13;k++){var bx=RR()*w,by=h*(.2+RR()*.55),br=(14+RR()*12)*U*(wet?1:.85);g.globalAlpha=wet?.34:.26;
    g.drawImage(HSM.blob(wet?'rgba(110,200,60,1)':'rgba(128,134,96,1)',32,.45),bx-br,by-br,br*2,br*2)}
   for(k=0;k<26;k++){var x=RR()*w,y=RR()*rockY,a=RR()*6.283,c2=0;g.beginPath();g.moveTo(x,y);
    for(var s=0;s<70;s++){c2+=(RR()-.5)*.1;c2*=.9;a+=c2;x+=Math.cos(a)*5*U;y+=Math.sin(a)*5*U;g.lineTo(x,y)}
    g.globalAlpha=.07+RR()*.1;g.lineWidth=(6+RR()*5)*U;g.strokeStyle=wet?'rgb(206,222,190)':'rgb(170,166,150)';g.stroke()}
   return c}
  /* ---- the skin (cortex): fungal cells packed tight, the lichen's weatherproof roof ---- */
  function cortex(wet){var cw=w*1.24,chh=h*.2,q=2,c=cv(cw*q,chh*q),g=c.getContext('2d'),RR=HSM.rnd(33),rows=8,rh2=chh*.105;g.scale(q,q);
   for(var rw=rows-1;rw>=0;rw--){var cy=chh*.1+rw*rh2,x=-RR()*10*U,dp=rw/(rows-1);
    while(x<cw){var rx=(6.5+RR()*5)*U,ry=rh2*(.56+RR()*.2),yy=cy+(RR()-.5)*rh2*.5+Math.sin(x*.02)*3*U;x+=rx;
     if(RR()>.25+dp*dp*.75&&rw>4){x+=rx;continue}
     g.save();g.translate(x,yy);g.rotate((RR()-.5)*.7);
     var gr=g.createRadialGradient(-rx*.2,-ry*.4,0,0,0,rx);
     if(wet){gr.addColorStop(0,'rgba(250,248,214,'+(.82-dp*.3)+')');gr.addColorStop(.7,'rgba('+Math.round(216-dp*30)+','+Math.round(222-dp*14)+','+Math.round(168-dp*10)+','+(.66-dp*.25)+')');gr.addColorStop(1,'rgba(150,170,120,'+(.6-dp*.25)+')')}
     else{gr.addColorStop(0,'rgba(206,198,172,'+(.8-dp*.3)+')');gr.addColorStop(.7,'rgba(168,160,138,'+(.66-dp*.25)+')');gr.addColorStop(1,'rgba(118,112,98,'+(.6-dp*.25)+')')}
     g.beginPath();g.ellipse(0,0,rx,ry,0,0,6.283);g.fillStyle=gr;g.fill();g.lineWidth=1.1*U;g.strokeStyle='rgba(30,36,24,.55)';g.stroke();
     g.beginPath();g.ellipse(0,0,rx*.55,ry*.5,0,0,6.283);g.fillStyle=wet?'rgba(255,255,236,.16)':'rgba(255,250,230,.08)';g.fill();
     g.restore();x+=rx*.86}}
   /* a thin weathered crust on the very top */
   var tg=g.createLinearGradient(0,chh*.03,0,chh*.2);tg.addColorStop(0,wet?'rgba(120,128,70,.55)':'rgba(120,112,92,.6)');tg.addColorStop(1,'rgba(120,128,70,0)');
   g.globalCompositeOperation='source-atop';g.fillStyle=tg;g.fillRect(0,0,cw,chh);return c}
  /* ---- the alga: wall, one big lobed chloroplast, a pale pyrenoid at its heart ---- */
  function alga(seed,wet){var S=56,c=cv(S*2.2,S*2.2),g=c.getContext('2d'),RR=HSM.rnd(seed),q,a,k,o=S*1.1;
   var l1=6+Math.floor(RR()*3),p1=RR()*6.283,p2=RR()*6.283,p3=RR()*6.283;
   function outline(rad,amp,lob){g.beginPath();for(k=0;k<=72;k++){a=k/72*6.283;var rr=rad*(1+amp*Math.sin(lob*a+p1)+amp*.6*Math.sin((lob*2+1)*a+p2));
     k?g.lineTo(o+Math.cos(a)*rr,o+Math.sin(a)*rr):g.moveTo(o+Math.cos(a)*rr,o+Math.sin(a)*rr)}g.closePath()}
   /* wall and cytoplasm */
   outline(S*.97,wet?.006:.035,5);q=g.createRadialGradient(o,o,S*.3,o,o,S);
   if(wet){q.addColorStop(0,'rgba(196,240,120,.92)');q.addColorStop(.8,'rgba(150,214,84,.88)');q.addColorStop(1,'rgba(92,150,58,.9)')}
   else{q.addColorStop(0,'rgba(150,154,112,.92)');q.addColorStop(.8,'rgba(124,128,96,.9)');q.addColorStop(1,'rgba(78,80,62,.92)')}
   g.fillStyle=q;g.fill();g.lineWidth=S*.05;g.strokeStyle=wet?'rgba(232,255,196,.85)':'rgba(186,184,160,.7)';g.stroke();
   g.lineWidth=S*.018;g.strokeStyle='rgba(20,40,16,.6)';outline(S*1.0,wet?.006:.035,5);g.stroke();
   /* chloroplast */
   g.save();outline(S*(wet?.74:.6),wet?.085:.11,l1);g.clip();
   q=g.createRadialGradient(o-S*.1,o-S*.38,S*.05,o,o,S*.82);
   if(wet){q.addColorStop(0,'#b6f03a');q.addColorStop(.45,'#58b81c');q.addColorStop(.85,'#257212');q.addColorStop(1,'#184f0e')}
   else{q.addColorStop(0,'#8e9a5c');q.addColorStop(.5,'#66713f');q.addColorStop(1,'#3c4428')}
   g.fillStyle=q;g.fillRect(0,0,S*2.2,S*2.2);
   for(k=0;k<70;k++){a=RR()*6.283;var rr=Math.sqrt(RR())*S*.7;g.fillStyle=wet?'rgba(16,74,8,'+(.1+RR()*.16)+')':'rgba(40,44,26,'+(.1+RR()*.14)+')';
    g.beginPath();g.arc(o+Math.cos(a)*rr,o+Math.sin(a)*rr,S*(.03+RR()*.05),0,6.283);g.fill()}
   /* pyrenoid, ringed with starch */
   q=g.createRadialGradient(o+S*.04,o+S*.03,0,o+S*.04,o+S*.03,S*.27);
   q.addColorStop(0,wet?'rgba(236,255,170,.95)':'rgba(176,180,134,.8)');q.addColorStop(.6,wet?'rgba(190,236,110,.6)':'rgba(150,154,110,.5)');q.addColorStop(1,'rgba(160,220,90,0)');
   g.fillStyle=q;g.fillRect(0,0,S*2.2,S*2.2);
   for(k=0;k<11;k++){a=k/11*6.283+p3;g.fillStyle=wet?'rgba(246,255,214,.6)':'rgba(190,190,160,.4)';g.beginPath();g.arc(o+S*.04+Math.cos(a)*S*.2,o+S*.03+Math.sin(a)*S*.2,S*.028,0,6.283);g.fill()}
   g.restore();
   g.lineWidth=S*.02;g.strokeStyle=wet?'rgba(14,60,8,.5)':'rgba(40,44,28,.5)';outline(S*(wet?.74:.6),wet?.085:.11,l1);g.stroke();
   return c}
  var spW=[],spD=[];for(i=0;i<3;i++){spW.push(alga(11+i*7,true));spD.push(alga(11+i*7,false))}
  /* soft shading that stays put while the cell sprite is turned: lit from above, dark below */
  var sh=cv(128,128),sg=sh.getContext('2d'),q2=sg.createRadialGradient(64,64,30,64,78,62);q2.addColorStop(0,'rgba(0,20,0,0)');q2.addColorStop(1,'rgba(0,18,4,.5)');
  sg.beginPath();sg.arc(64,64,62,0,6.283);sg.fillStyle=q2;sg.fill();
  q2=sg.createRadialGradient(50,34,0,50,34,34);q2.addColorStop(0,'rgba(255,255,236,.6)');q2.addColorStop(1,'rgba(255,255,236,0)');sg.fillStyle=q2;sg.fillRect(0,0,128,128);
  /* ---- the rock underneath ---- */
  var rk=cv(w,h-rockY+22*U),rg=rk.getContext('2d'),RK=HSM.rnd(5),rh=rk.height;
  rg.beginPath();rg.moveTo(0,rh);for(x=0;x<=w;x+=6){rg.lineTo(x,14*U+Math.sin(x*.021+1)*5*U+Math.sin(x*.07)*2.5*U+(RK()-.5)*3*U)}rg.lineTo(w,rh);rg.closePath();rg.save();rg.clip();
  var q3=rg.createLinearGradient(0,0,0,rh);q3.addColorStop(0,'#5d5a55');q3.addColorStop(.35,'#35332f');q3.addColorStop(1,'#151413');rg.fillStyle=q3;rg.fillRect(0,0,w,rh);
  for(i=0;i<260;i++){var gx=RK()*w,gy=RK()*rh,gs=(1.5+RK()*RK()*11)*U,tn=RK();rg.globalAlpha=tn<.5?.2+RK()*.3:tn<.92?.08+RK()*.14:.3+RK()*.35;
   rg.drawImage(HSM.blob(tn<.5?'rgba(10,10,12,1)':tn<.92?'rgba(176,168,150,1)':'rgba(244,232,204,1)',16,tn<.92?.5:.3),gx-gs,gy-gs*.7,gs*2,gs*1.4)}
  rg.globalAlpha=1;var q4=rg.createLinearGradient(0,8*U,0,26*U);q4.addColorStop(0,'rgba(214,206,184,.5)');q4.addColorStop(1,'rgba(214,206,184,0)');rg.fillStyle=q4;rg.fillRect(0,0,w,rh);
  rg.restore();
  /* labels: an alga well inside the frame, and a stretch of thread low down */
  var la=0,best=1e9;for(i=0;i<algae.length;i++){var d=Math.abs(algae[i].x-w*.3)+Math.abs(algae[i].y-h*.3)*1.2;if(d<best){best=d;la=i}}
  return{U:U,W:W,ST:ST,rockY:rockY,algae:algae,paths:paths,nmain:nmain,hy:hy,cx:cx,sep:sep,sep2:sep2,cxW:cortex(true),cxD:cortex(false),bkW:back(true),bkD:back(false),spW:spW,spD:spD,sh:sh,rk:rk,
   motes:[],k:null,front:0,la:la,R:R}},

 frame:function(s,g,w,h,t,dt,ui){
  var C=HSM.clamp,U=s.U,W=s.W,i,m,A,dry=!!ui.on;
  if(s.k==null){s.k=dry?0:1;s.front=1;
   /* start with sugar already on its way, so the first still tells the story */
   if(!dry)for(i=0;i<70;i++){A=s.algae[i%s.algae.length];var P0=s.paths[A.path];s.motes.push({st:1,p:A.path,i:P0.clasp+s.R()*(P0.pts.length/2-P0.clasp-2),v:(15+s.R()*12)*U,life:s.R()*.7,sz:.8+s.R()*.5,off:(s.R()-.5)})}}
  /* wetting is quick (seconds), drying slower */
  var was=s.k;s.k=C(s.k+(dry?-dt/3.2:dt/2.1),0,1);if(!dry&&was<.3&&s.front>=1)s.front=0;if(s.front<1)s.front=Math.min(1,s.front+dt/2.4);
  var k=s.k*s.k*(3-2*s.k),kc=HSM.smooth(.15,.95,s.k);
  function mix(a,b){return Math.round(a+(b-a)*kc)}
  /* ground: deep inside the slice, lit from above */
  var bg=g.createLinearGradient(0,0,0,h);
  bg.addColorStop(0,'rgb('+mix(96,74)+','+mix(90,104)+','+mix(72,68)+')');bg.addColorStop(.3,'rgb('+mix(48,26)+','+mix(46,52)+','+mix(38,36)+')');
  bg.addColorStop(1,'rgb('+mix(16,7)+','+mix(15,17)+','+mix(13,14)+')');g.fillStyle=bg;g.fillRect(0,0,w,h);
  g.drawImage(s.rk,0,s.rockY-14*U,w,s.rk.height);
  /* the whole body shrinks toward the rock as it dries */
  var sc=.865+.135*k;g.save();g.translate(w/2,s.rockY);g.scale(sc,sc);g.translate(-w/2,-s.rockY);
  g.globalAlpha=1-kc;g.drawImage(s.bkD,-w*.1,0,w*1.2,h);g.globalAlpha=kc;g.drawImage(s.bkW,-w*.1,0,w*1.2,h);g.globalAlpha=1;
  /* algae */
  var rs=.9+.1*k;
  for(i=0;i<s.algae.length;i++){A=s.algae[i];var r=A.r*rs*(1+.012*k*Math.sin(t*1.3+A.ph)),gl=r*2.5;
   g.globalCompositeOperation='lighter';g.globalAlpha=(.2+.1*Math.sin(t*.8+A.ph)+A.pulse*.25)*kc;g.drawImage(HSM.blob('rgba(120,255,70,1)',48,.2),A.x-gl,A.y-gl,gl*2,gl*2);g.globalCompositeOperation='source-over';
   g.save();g.translate(A.x,A.y);g.rotate(A.rot);var d=r*1.1;
   if(kc<1){g.globalAlpha=1;g.drawImage(s.spD[A.v],-d,-d,d*2,d*2)}
   if(kc>0){g.globalAlpha=kc;g.drawImage(s.spW[A.v],-d,-d,d*2,d*2)}
   g.restore();g.globalAlpha=.35+.65*kc;g.drawImage(s.sh,A.x-r,A.y-r,r*2,r*2);
   if(A.pulse>0)A.pulse=Math.max(0,A.pulse-dt*1.6)}
  g.globalAlpha=1;
  /* fungal threads: a dark edge, a pale translucent body, a bright core where light runs down the tube */
  g.lineCap='round';g.lineJoin='round';var hw=W*(.68+.32*k),h2=hw*.7;
  g.strokeStyle='rgba('+mix(34,16)+','+mix(32,34)+','+mix(26,20)+',.6)';g.lineWidth=h2+2.2*U;g.stroke(s.cx);g.lineWidth=hw+2.4*U;g.stroke(s.hy);
  g.strokeStyle='rgba('+mix(160,208)+','+mix(154,224)+','+mix(134,190)+','+(.46+.1*kc)+')';g.lineWidth=h2;g.stroke(s.cx);
  g.strokeStyle='rgba('+mix(170,222)+','+mix(162,232)+','+mix(138,196)+','+(.5+.1*kc)+')';g.lineWidth=hw;g.stroke(s.hy);
  g.lineWidth=1.1*U;g.strokeStyle='rgba('+mix(60,36)+','+mix(58,66)+','+mix(48,40)+',.42)';g.stroke(s.sep);g.stroke(s.sep2);
  g.save();g.translate(-hw*.12,-hw*.2);g.strokeStyle='rgba(255,255,238,'+(.14+.3*kc)+')';g.lineWidth=h2*.3;g.stroke(s.cx);g.lineWidth=hw*.3;g.stroke(s.hy);g.restore();
  g.globalAlpha=1-kc;g.drawImage(s.cxD,-w*.12,h*.15-h*.2*.86,w*1.24,h*.2);g.globalAlpha=kc;g.drawImage(s.cxW,-w*.12,h*.15-h*.2*.86,w*1.24,h*.2);g.globalAlpha=1;
  /* sugar: made in the alga, handed over at the clasp, carried off along the thread */
  var flow=kc;
  if(flow>.5)for(i=0;i<s.algae.length;i++){A=s.algae[i];A.emit-=dt*flow;if(A.emit<0&&s.motes.length<110){A.emit=.9+s.R()*1.5;A.pulse=1;var a=s.R()*6.283,rr=s.R()*A.r*.5;
    s.motes.push({st:0,a:i,u:0,x0:A.x+Math.cos(a)*rr,y0:A.y+Math.sin(a)*rr,p:A.path,i:s.paths[A.path].clasp,v:(15+s.R()*12)*U,life:0,sz:.8+s.R()*.5,off:(s.R()-.5)})}}
  g.globalCompositeOperation='lighter';var gb=HSM.blob('rgba(255,160,30,1)',16,.15),gc=HSM.blob('rgba(255,226,130,1)',8,.5);
  for(i=s.motes.length-1;i>=0;i--){m=s.motes[i];var P=s.paths[m.p],pts=P.pts,x,y,al=1;
   if(m.st===0){m.u+=dt*flow*.8;var e=m.u*m.u*(3-2*m.u),ci=P.clasp*2;x=m.x0+(pts[ci]-m.x0)*e;y=m.y0+(pts[ci+1]-m.y0)*e;al=C(m.u*4,0,1);if(m.u>=1){m.st=1}}
   else{var prev=Math.floor(m.i);m.i+=m.v*flow*dt/s.ST;m.life+=dt*flow/22;var n=pts.length/2,ii=Math.floor(m.i);
    if(ii>prev)for(var b=0;b<P.br.length;b++)if(P.br[b].i===ii&&s.R()<.4){m.p=P.br[b].p;m.i=0;ii=0;P=s.paths[m.p];pts=P.pts;n=pts.length/2;break}
    if(ii>=n-1){if(P.link){m.p=P.link.p;m.i=P.link.i;ii=P.link.i;pts=s.paths[m.p].pts;n=pts.length/2}else{s.motes.splice(i,1);continue}}
    if(m.life>=1||ii>=n-1){s.motes.splice(i,1);continue}
    var f=m.i-ii,j=ii*2,tx=pts[j+2]-pts[j],ty=pts[j+3]-pts[j+1];x=pts[j]+tx*f-ty/s.ST*m.off*hw*.22;y=pts[j+1]+ty*f+tx/s.ST*m.off*hw*.22;
    al=Math.min(1,(1-m.life)*3)*Math.min(1,(n-1-m.i)/5+(P.link?1:0))}
   al*=kc;if(al<=.01)continue;var gr=6.5*U*m.sz,cr=2.1*U*m.sz;
   g.globalAlpha=al*.8;g.drawImage(gb,x-gr,y-gr,gr*2,gr*2);g.globalAlpha=al;g.drawImage(gc,x-cr,y-cr,cr*2,cr*2)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  g.restore();
  /* light from above, and the sheet of water soaking down when it is wetted */
  g.globalCompositeOperation='lighter';
  var lg=g.createLinearGradient(0,0,0,h*.62);lg.addColorStop(0,'rgba(255,244,196,'+(.26+.03*Math.sin(t*.7))+')');lg.addColorStop(.35,'rgba(255,240,180,.07)');lg.addColorStop(1,'rgba(255,240,180,0)');g.fillStyle=lg;g.fillRect(0,0,w,h*.62);
  if(s.front<1){var fy=(-.15+s.front*1.3)*h,fa=Math.sin(s.front*Math.PI)*.34,wg=g.createLinearGradient(0,fy-h*.3,0,fy+h*.05);
   wg.addColorStop(0,'rgba(120,200,230,0)');wg.addColorStop(.85,'rgba(150,220,240,'+fa+')');wg.addColorStop(1,'rgba(150,220,240,0)');g.fillStyle=wg;g.fillRect(0,fy-h*.3,w,h*.35)}
  g.globalCompositeOperation='source-over';
  HSM.vignette(g,w,h,.5);
  /* labels ride with the shrinking body */
  A=s.algae[s.la];var lx=w/2+(A.x-w/2)*sc,ly=s.rockY+(A.y-A.r*rs-11*U-s.rockY)*sc;
  HSM.label(g,C(lx,62,w-62),ly,'alga: makes sugar','center');
  HSM.label(g,w-12,s.rockY-20*U,'fungus: shelter and water','right');
  if(kc<.5){g.globalAlpha=1-kc*2;HSM.label(g,w/2,20,'dry and dormant, not dead','center');g.globalAlpha=1}
 }});
