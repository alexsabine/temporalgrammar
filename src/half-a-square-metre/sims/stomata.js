/* The underside of a saxifrage leaf. Jigsaw skin cells, and scattered among them the stomata:
   each a pair of guard cells around a pore. Full of water they swell, bow apart and open the pore;
   short of water they go slack and it closes. CO2 drifts in through open pores. Water vapour leaves
   by the same doors, many times more of it: a leaf loses hundreds of water molecules for each CO2 it gains. */
HSM.sim('stomata',{
 slider:{label:'Water',lo:'thirsty',hi:'plenty',value:.75},
 setup:function(w,h,R){
  /* skin: a warped Voronoi pattern, its walls drawn once into a sprite */
  var seeds=[],n=Math.round(w*h/5200),i,j;
  for(i=0;i<n;i++)seeds.push([R()*w,R()*h,R()]);
  var st=[],tries=0;
  while(st.length<Math.max(5,Math.round(w*h/30000))&&tries<500){tries++;var x=w*.12+R()*w*.76,y=h*.1+R()*h*.8,ok=true;
   for(j=0;j<st.length;j++)if(Math.hypot(st[j].x-x,st[j].y-y)<w*.3)ok=false;
   if(ok)st.push({x:x,y:y,a:(R()-.5)*1.1,lag:.6+R()*.8,o:.5,sz:.9+R()*.25})}
  var K=2,cw=Math.round(w*K),chh=Math.round(h*K),c=document.createElement('canvas');c.width=cw;c.height=chh;
  var g=c.getContext('2d'),img=g.createImageData(cw,chh),d=img.data;
  /* cell interiors and walls from distance to nearest and second-nearest seed */
  for(var py=0;py<chh;py++)for(var px=0;px<cw;px++){
   var X=px/K,Y=py/K,wx=X+Math.sin(Y*.11+Math.sin(X*.05)*2)*5.5+Math.sin(Y*.31+X*.07)*2.2,wy=Y+Math.cos(X*.12+Math.sin(Y*.04)*2)*5.5+Math.cos(X*.29-Y*.05)*2.2;
   var d1=1e9,d2=1e9,k1=0;
   for(i=0;i<seeds.length;i++){var dx=seeds[i][0]-wx,dy=seeds[i][1]-wy,dd=dx*dx+dy*dy;if(dd<d1){d2=d1;d1=dd;k1=i}else if(dd<d2)d2=dd}
   var e=Math.sqrt(d2)-Math.sqrt(d1),tone=seeds[k1][2],o=(py*cw+px)*4;
   var wall=Math.max(0,1-e/2.2),rim=Math.max(0,1-Math.abs(e-3.2)/2.2)*(1-wall);
   var cr=196+tone*14,cg=222+tone*10,cb=176+tone*12,lift=Math.min(1,e/26)*14;
   d[o]=cr+lift-wall*110+rim*30;d[o+1]=cg+lift-wall*90+rim*26;d[o+2]=cb+lift-wall*110+rim*24;d[o+3]=255}
  g.putImageData(img,0,0);
  /* guard cell sprite (one pair, pore drawn live) */
  var gs=Math.round(w*.11),G=document.createElement('canvas');G.width=gs*4;G.height=gs*3;var q=G.getContext('2d'),cx=gs*2,cy=gs*1.5;
  for(var side=-1;side<=1;side+=2){q.save();q.translate(cx,cy+side*gs*.36);
   var gr=q.createRadialGradient(-gs*.2,-side*gs*.1,gs*.1,0,0,gs*1.1);gr.addColorStop(0,'#cfe9a0');gr.addColorStop(.6,'#8fc25a');gr.addColorStop(1,'#5e9638');
   q.beginPath();q.ellipse(0,0,gs*1.05,gs*.52,0,0,6.283);q.fillStyle=gr;q.fill();q.lineWidth=gs*.06;q.strokeStyle='rgba(60,95,40,.8)';q.stroke();
   for(i=0;i<14;i++){var a=R()*6.283,r=Math.sqrt(R());q.fillStyle='rgba('+(40+R()*30|0)+','+(110+R()*40|0)+',30,.85)';q.beginPath();q.ellipse(Math.cos(a)*r*gs*.78,Math.sin(a)*r*gs*.32,gs*.09,gs*.065,a,0,6.283);q.fill()}
   q.fillStyle='rgba(255,255,230,.35)';q.beginPath();q.ellipse(-gs*.3,-side*gs*.15,gs*.4,gs*.12,0,0,6.283);q.fill();q.restore()}
  var mol=[];for(i=0;i<160;i++)mol.push({live:0});
  return{skin:c,st:st,G:G,gs:gs,mol:mol,R:R,acc:0}},
 frame:function(s,g,w,h,t,dt,ui){
  var W=ui.a==null?.75:ui.a,sm=HSM.smooth,i,m,k;dt=Math.max(0,dt);
  var open=sm(.15,.85,W);
  g.drawImage(s.skin,0,0,w,h);
  /* gentle depth of field breathing, as when the stage drifts */
  var fq=g.createRadialGradient(w*.5,h*.5,Math.min(w,h)*.2,w*.5,h*.5,Math.max(w,h)*.7);fq.addColorStop(0,'rgba(255,255,240,.08)');fq.addColorStop(1,'rgba(40,60,30,.25)');g.fillStyle=fq;g.fillRect(0,0,w,h);
  var gs=s.gs;
  for(i=0;i<s.st.length;i++){var S=s.st[i];S.o+=(open-S.o)*Math.min(1,dt*.9/S.lag);
   g.save();g.translate(S.x,S.y);g.rotate(S.a);g.scale(S.sz,S.sz);
   /* the pair: open guard cells bow apart */
   var sep=gs*.12*S.o;
   g.save();g.translate(0,-sep);g.drawImage(s.G,0,0,gs*4,gs*1.5,-gs*2,-gs*1.5,gs*4,gs*1.5);g.restore();
   g.save();g.translate(0,sep);g.drawImage(s.G,0,gs*1.5,gs*4,gs*1.5,-gs*2,0,gs*4,gs*1.5);g.restore();
   /* the pore */
   var pr=gs*(.05+.26*S.o);g.beginPath();g.ellipse(0,0,gs*.7,pr,0,0,6.283);
   var pg=g.createLinearGradient(0,-pr,0,pr);pg.addColorStop(0,'#1d2a1a');pg.addColorStop(.5,'#3a4d2c');pg.addColorStop(1,'#1d2a1a');g.fillStyle=pg;g.fill();
   g.lineWidth=1.2;g.strokeStyle='rgba(230,250,200,.55)';g.stroke();
   g.restore();
   S.gx=S.x;S.gy=S.y}
  /* molecules: water leaves often, CO2 enters rarely */
  s.acc+=dt*open*40;
  while(s.acc>1){s.acc-=1;var St=s.st[Math.floor(s.R()*s.st.length)],isC=s.R()<.08;
   for(k=0;k<s.mol.length;k++)if(!s.mol[k].live){m=s.mol[k];break}
   if(!m)break;
   if(isC){var a=s.R()*6.283,r=gs*(2.6+s.R()*1.4);m.c=1;m.tx=St.x;m.ty=St.y;m.x=St.x+Math.cos(a)*r;m.y=St.y+Math.sin(a)*r;m.v=.25+s.R()*.15}
   else{var a2=s.R()*6.283;m.c=0;m.x=St.x+(s.R()-.5)*gs*.8;m.y=St.y;m.vx=Math.cos(a2)*(30+s.R()*40);m.vy=Math.sin(a2)*(30+s.R()*40)}
   m.live=1;m.age=0;m=null}
  for(k=0;k<s.mol.length;k++){m=s.mol[k];if(!m.live)continue;m.age+=dt;
   if(m.c){m.x+=(m.tx-m.x)*Math.min(1,dt*m.v*3);m.y+=(m.ty-m.y)*Math.min(1,dt*m.v*3);var dd=Math.hypot(m.tx-m.x,m.ty-m.y);var al=Math.min(1,dd/(gs*.5));if(dd<3||m.age>6)m.live=0;
     g.globalAlpha=.95*al;g.drawImage(HSM.blob('rgba(255,170,70,1)',14,.4),m.x-6,m.y-6,12,12);g.globalAlpha=al;g.lineWidth=1.2;g.strokeStyle='rgba(110,50,0,.75)';g.beginPath();g.arc(m.x,m.y,3.6,0,6.283);g.stroke()}
   else{m.x+=m.vx*dt;m.y+=m.vy*dt;m.vx*=.997;m.vy*=.997;var al2=Math.max(0,1-m.age/3.2);if(al2<=0)m.live=0;
     g.globalAlpha=al2;g.drawImage(HSM.blob('rgba(120,200,255,1)',14,.45),m.x-6,m.y-6,12,12);g.lineWidth=1;g.strokeStyle='rgba(20,70,120,'+(.7*al2)+')';g.beginPath();g.arc(m.x,m.y,3.2,0,6.283);g.stroke()}}
  g.globalAlpha=1;
  HSM.vignette(g,w,h,.4);
  var S0=s.st[0];
  if(S0){var lx=Math.min(w-12,S0.x+gs*1.4),ly=Math.max(16,S0.y-gs*1.3);g.strokeStyle='rgba(255,255,255,.7)';g.lineWidth=1;g.beginPath();g.moveTo(S0.x+gs*.6,S0.y-gs*.4);g.lineTo(lx-4,ly+4);g.stroke();HSM.label(g,lx,ly,'guard cells',lx>w*.7?'right':'left')}
  HSM.label(g,12,18,open<.15?'Thirsty: the pores are shut':'CO₂ in (orange), water out (blue)');
  HSM.label(g,12,h-16,'every breath in costs hundreds of breaths of water out');
 }});
