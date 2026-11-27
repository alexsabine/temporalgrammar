/* Inside a moss leaf. One layer of cells, light coming through. The chloroplasts are not fixed:
   in gentle light they spread across the face of the cell to catch it, in harsh light they
   retreat to the side walls and turn edge-on. CO2 drifts in and is taken up; oxygen leaves. */
HSM.sim('cells',{
 slider:{label:'Light',lo:'dusk',hi:'full sun',value:.45},
 setup:function(w,h,R){
  var cw=w/3.4,ch=cw*.62,cells=[],plasts=[],gas=[];
  function vtx(i,j){var x=i*cw*.5,y=j*ch*.5,k=Math.sin(i*127.1+j*311.7)*43758.5453;k-=Math.floor(k);var k2=Math.sin(i*269.5+j*183.3)*43758.5453;k2-=Math.floor(k2);return[x+(k-.5)*cw*.16,y+(k2-.5)*ch*.2]}
  for(var r=-1;r<h/ch+1;r++)for(var c=-1;c<w/cw+1;c++){
   var ox=c*2+(r&1),oy=r*2,P=[vtx(ox,oy+.0-.0),vtx(ox+1,oy-0.66),vtx(ox+2,oy),vtx(ox+2,oy+1.34),vtx(ox+1,oy+2),vtx(ox,oy+1.34)];
   P=[[ox*cw*.5,oy*ch*.5+ch*.17],[ (ox+1)*cw*.5,oy*ch*.5-ch*.17],[(ox+2)*cw*.5,oy*ch*.5+ch*.17],[(ox+2)*cw*.5,(oy+2)*ch*.5-ch*.17],[(ox+1)*cw*.5,(oy+2)*ch*.5+ch*.17],[ox*cw*.5,(oy+2)*ch*.5-ch*.17]]
     .map(function(p,n){var ii=Math.round(p[0]/(cw*.5)),jj=Math.round(p[1]/(ch*.17));var k=Math.sin(ii*127.1+jj*311.7)*43758.5453;k-=Math.floor(k);var k2=Math.sin(ii*269.5+jj*183.3)*43758.5453;k2-=Math.floor(k2);return[p[0]+(k-.5)*cw*.14,p[1]+(k2-.5)*ch*.16]});
   var cx=0,cy=0;P.forEach(function(p){cx+=p[0]/6;cy+=p[1]/6});
   var cell={P:P,cx:cx,cy:cy,ph:R()*6.28};cells.push(cell);
   var n=9+Math.floor(R()*6);
   for(var k=0;k<n;k++){var a=R()*6.283,rr=Math.sqrt(R())*.95,e=Math.floor(R()*6),f=.15+R()*.7,A=P[e],B=P[(e+1)%6];
    var wx=A[0]+(B[0]-A[0])*f,wy=A[1]+(B[1]-A[1])*f;wx+=(cx-wx)*.13;wy+=(cy-wy)*.13;
    plasts.push({fx:cx+Math.cos(a)*rr*cw*.42,fy:cy+Math.sin(a)*rr*ch*.42,wx:wx,wy:wy,wa:Math.atan2(B[1]-A[1],B[0]-A[0]),k:R(),ph:R()*6.28,sz:.85+R()*.3,lag:.4+R()*.6,rot:R()*6.28,busy:0})}}
  /* chloroplast sprite: a green lens with darker grana and a soft highlight */
  var S=Math.round(cw*.155),sp=document.createElement('canvas');sp.width=sp.height=S*2;var g=sp.getContext('2d'),q=g.createRadialGradient(S*.8,S*.75,S*.1,S,S,S*.98);
  q.addColorStop(0,'#a8e04a');q.addColorStop(.55,'#4f9e1f');q.addColorStop(.9,'#2c6d14');q.addColorStop(1,'rgba(30,80,15,0)');g.fillStyle=q;g.beginPath();g.arc(S,S,S,0,6.283);g.fill();
  for(var i=0;i<26;i++){var a2=R()*6.283,r2=Math.sqrt(R())*S*.72;g.fillStyle='rgba(20,70,10,'+(.18+R()*.2)+')';g.beginPath();g.arc(S+Math.cos(a2)*r2,S+Math.sin(a2)*r2,S*(.06+R()*.07),0,6.283);g.fill()}
  q=g.createRadialGradient(S*.7,S*.6,0,S*.7,S*.6,S*.5);q.addColorStop(0,'rgba(240,255,190,.55)');q.addColorStop(1,'rgba(240,255,190,0)');g.fillStyle=q;g.fillRect(0,0,S*2,S*2);
  for(i=0;i<46;i++)gas.push({x:R()*w,y:R()*h,vx:(R()-.5)*14,vy:(R()-.5)*14,o2:i%2,life:R()});
  return{cells:cells,plasts:plasts,gas:gas,sp:sp,S:S,cw:cw,ch:ch,k:0,R:R,sugar:[]}},
 frame:function(s,g,w,h,t,dt,ui){
  var L=ui.a==null?.45:ui.a,C=HSM.clamp,sm=HSM.smooth,i,p;
  var avoid=sm(.62,.92,L);s.k+=(avoid-s.k)*Math.min(1,dt*1.2);
  var rate=sm(.03,.5,L)*(1-.35*sm(.8,1,L)); /* photosynthesis: saturates, then sags a little in harsh light */
  /* light through the leaf */
  var b=.14+.86*sm(0,.75,L),bg=g.createLinearGradient(0,0,w,h);
  bg.addColorStop(0,'rgb('+Math.round(214*b+6)+','+Math.round(236*b+10)+','+Math.round(150*b+8)+')');
  bg.addColorStop(1,'rgb('+Math.round(160*b+4)+','+Math.round(205*b+8)+','+Math.round(96*b+6)+')');g.fillStyle=bg;g.fillRect(0,0,w,h);
  /* cell bodies: a faint lens of cytoplasm in each */
  for(i=0;i<s.cells.length;i++){var c=s.cells[i],P=c.P,q=g.createRadialGradient(c.cx,c.cy,0,c.cx,c.cy,s.cw*.55);
   q.addColorStop(0,'rgba(255,255,235,'+(.22*b)+')');q.addColorStop(1,'rgba(90,140,40,'+(.10+.1*b)+')');
   g.beginPath();g.moveTo(P[0][0],P[0][1]);for(var n=1;n<6;n++)g.lineTo(P[n][0],P[n][1]);g.closePath();g.fillStyle=q;g.fill()}
  /* walls: a dark seam with a bright refracting edge either side */
  g.lineJoin='round';
  for(var pass=0;pass<3;pass++){g.beginPath();for(i=0;i<s.cells.length;i++){P=s.cells[i].P;g.moveTo(P[0][0],P[0][1]);for(n=1;n<6;n++)g.lineTo(P[n][0],P[n][1]);g.closePath()}
   g.lineWidth=[s.cw*.085,s.cw*.05,s.cw*.014][pass];g.strokeStyle=['rgba(70,110,40,'+(.35+.2*b)+')','rgba(235,250,200,'+(.25+.6*b)+')','rgba(60,90,40,.55)'][pass];g.stroke()}
  /* chloroplasts */
  var S=s.S;
  for(i=0;i<s.plasts.length;i++){p=s.plasts[i];var k=sm(0,1,C(s.k*1.5-p.lag*.5,0,1)),jx=Math.sin(t*.5+p.ph)*s.cw*.012,jy=Math.cos(t*.42+p.ph*1.7)*s.cw*.012;
   var x=p.fx+(p.wx-p.fx)*k+jx*(1-k),y=p.fy+(p.wy-p.fy)*k+jy*(1-k);
   if(x<-S||y<-S||x>w+S||y>h+S)continue;p.x=x;p.y=y;
   g.save();g.translate(x,y);g.rotate(p.rot+(p.wa-p.rot)*k);g.scale(p.sz,p.sz*(1-.55*k));
   g.globalAlpha=.55+.45*b;g.drawImage(s.sp,-S,-S);
   if(p.busy>0){g.globalCompositeOperation='lighter';g.globalAlpha=p.busy*.8;g.drawImage(HSM.blob('rgba(255,255,170,1)',S),-S,-S);g.globalCompositeOperation='source-over';p.busy-=dt*2.2}
   g.restore()}
  g.globalAlpha=1;
  /* gases: CO2 (warm) wanders until a working chloroplast takes it; O2 (cool) is released and drifts off */
  for(i=0;i<s.gas.length;i++){var m=s.gas[i];m.vx+=(s.R()-.5)*40*dt;m.vy+=(s.R()-.5)*40*dt;m.vx*=.985;m.vy*=.985;m.x+=m.vx*dt;m.y+=m.vy*dt;
   if(m.x<-10)m.x=w+10;if(m.x>w+10)m.x=-10;if(m.y<-10)m.y=h+10;if(m.y>h+10)m.y=-10;
   if(!m.o2&&s.R()<rate*dt*1.6){var tg=s.plasts[Math.floor(s.R()*s.plasts.length)];if(tg.x!=null&&Math.abs(tg.x-m.x)<s.cw*.6&&Math.abs(tg.y-m.y)<s.cw*.6){tg.busy=1;m.o2=1;m.x=tg.x;m.y=tg.y;m.vx=(s.R()-.5)*50;m.vy=-20-s.R()*30;m.life=0}}
   if(m.o2){m.life+=dt*.12;if(m.life>1){m.o2=0;m.life=0;m.x=s.R()*w;m.y=s.R()<.5?-8:h+8}}
   var r=m.o2?5:4.2;g.globalAlpha=m.o2?.9*(1-m.life*.6):.85;
   g.drawImage(HSM.blob(m.o2?'rgba(150,225,255,1)':'rgba(255,196,120,1)',16,.3),m.x-r,m.y-r,r*2,r*2);
   g.globalAlpha=1;g.lineWidth=1;g.strokeStyle=m.o2?'rgba(20,60,90,.55)':'rgba(110,60,10,.6)';g.beginPath();g.arc(m.x,m.y,r*.62,0,6.283);g.stroke()}
  g.globalAlpha=1;
  /* glare in harsh light, dimness at dusk */
  if(L>.75){g.globalCompositeOperation='lighter';g.fillStyle='rgba(255,255,230,'+((L-.75)*.5)+')';g.fillRect(0,0,w,h);g.globalCompositeOperation='source-over'}
  HSM.vignette(g,w,h,.35+.3*(1-b));
  HSM.label(g,12,18,L<.25?'Too dim: little work is done':avoid>.5?'Harsh light: chloroplasts hide at the walls':'Chloroplasts spread out to catch the light');
 }});
