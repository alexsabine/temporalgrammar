/* A stream breathing out. A cut through a few millimetres of flowing water: rock and its living film
   below, air above. The water arrives loaded with CO2 from the soil upstream and loses it through the
   surface; oxygen goes the other way. Gas only crosses in the thin top layer, so what matters is how
   fast turbulence carries deep water up to the surface: faster, rougher flow exchanges gas much faster. */
HSM.sim('gas',{
 slider:{label:'Flow',lo:'slow',hi:'fast',value:.45},
 setup:function(w,h,R){
  var D=2,y0=h*.33,yb=h*.8,i,x,y,g,q,n;
  var bp=[R()*6.28,R()*6.28,R()*6.28];
  function bedY(x){return yb+Math.sin(x*.019+bp[0])*h*.02+Math.sin(x*.047+bp[1])*h*.01+Math.sin(x*.12+bp[2])*h*.004}
  /* air: a shaded bank far out of focus, sun from the upper left */
  var ah=Math.ceil(y0+40),air=document.createElement('canvas');air.width=Math.round(w*D);air.height=ah*D;g=air.getContext('2d');g.scale(D,D);
  q=g.createLinearGradient(0,0,0,ah);q.addColorStop(0,'#3a5f45');q.addColorStop(.6,'#22443a');q.addColorStop(1,'#173630');g.fillStyle=q;g.fillRect(0,0,w,ah);
  g.globalCompositeOperation='lighter';
  var bk=['rgba(150,200,110,1)','rgba(225,232,150,1)','rgba(90,165,115,1)','rgba(200,232,222,1)'];
  for(i=0;i<46;i++){var br=8+R()*R()*44;g.globalAlpha=.03+R()*.11;g.drawImage(HSM.blob(bk[i%4],32,.72),R()*w-br,R()*ah*.95-br,br*2,br*2)}
  g.globalAlpha=.3;g.drawImage(HSM.blob('rgba(255,238,180,1)',64),-w*.45,-w*.55,w*1.1,w*1.1);
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  /* the bed: a cut face of rock, with a green-brown film of algae, diatoms and bacteria on top */
  var top=Math.floor(yb-h*.07),bh=Math.ceil(h-top),bed=document.createElement('canvas');bed.width=Math.round(w*D);bed.height=bh*D;g=bed.getContext('2d');g.scale(D,D);g.translate(0,-top);
  function rock(){g.beginPath();g.moveTo(-6,bedY(-6));for(x=-2;x<=w+6;x+=4)g.lineTo(x,bedY(x));g.lineTo(w+6,h+6);g.lineTo(-6,h+6);g.closePath()}
  g.save();rock();g.clip();
  q=g.createLinearGradient(0,yb-10,0,h);q.addColorStop(0,'#6b6657');q.addColorStop(.35,'#4a463d');q.addColorStop(1,'#1f1d1b');g.fillStyle=q;g.fillRect(0,top,w,bh);
  for(i=0;i<16;i++){var rr=20+R()*50;g.globalAlpha=.1+R()*.14;g.drawImage(HSM.blob(i%2?'rgba(170,160,135,1)':'rgba(15,14,12,1)',32),R()*w-rr,yb+R()*(h-yb)-rr,rr*2,rr*1.4)}
  g.globalAlpha=1;
  for(i=0;i<7;i++){y=yb+6+i*(h-yb)/6+R()*6;g.beginPath();g.moveTo(-5,y);for(x=0;x<=w+20;x+=20)g.lineTo(x,y+x*.06+Math.sin(x*.03+i)*3+R()*1.5);g.lineWidth=.6+R()*1.4;g.strokeStyle=i%2?'rgba(205,195,170,.13)':'rgba(10,10,10,.22)';g.stroke()}
  var gr=['rgba(215,205,180,','rgba(12,12,14,','rgba(150,95,55,','rgba(235,235,225,'];
  for(i=0;i<1100;i++){x=R()*w;y=yb-12+R()*(h-yb+14);g.fillStyle=gr[Math.floor(R()*R()*4)]+(.1+R()*.3)+')';g.beginPath();g.ellipse(x,y,.4+R()*R()*2.6,.3+R()*1.3,R()*3.14,0,6.283);g.fill()}
  for(i=0;i<3;i++){x=R()*w;y=bedY(x)+4;g.beginPath();g.moveTo(x,y);for(n=0;n<7;n++){x+=(R()-.35)*14;y+=5+R()*9;g.lineTo(x,y)}g.lineWidth=.8;g.strokeStyle='rgba(8,8,8,.55)';g.stroke()}
  g.beginPath();for(x=-6;x<=w+6;x+=4)g.lineTo(x,bedY(x)+3);g.lineWidth=9;g.strokeStyle='rgba(215,205,160,.2)';g.stroke();
  q=g.createLinearGradient(0,h-40,0,h);q.addColorStop(0,'rgba(0,0,0,0)');q.addColorStop(1,'rgba(0,0,0,.5)');g.fillStyle=q;g.fillRect(0,h-40,w,40);
  g.restore();
  var fc=['rgba(84,112,38,1)','rgba(58,84,30,1)','rgba(124,108,46,1)','rgba(150,165,62,1)','rgba(96,72,34,1)'];
  for(x=-4;x<w+4;x+=.8){var th=2.2+2*Math.sin(x*.09+bp[1])+1.5*Math.sin(x*.31);rr=1.6+R()*2.6+th*.4;y=bedY(x)+1.5-R()*(2+th);g.globalAlpha=.35+R()*.4;g.drawImage(HSM.blob(fc[Math.floor(R()*5)],8,.4),x-rr,y-rr,rr*2,rr*2)}
  g.globalAlpha=1;
  for(i=0;i<w*.9;i++){x=R()*w;y=bedY(x)-1-R()*4;g.fillStyle=R()<.5?'rgba(214,196,92,.75)':'rgba(190,225,110,.7)';g.fillRect(x,y,.9,.9)}
  /* a shaft of light: bright under the surface, fading with depth, soft at the sides */
  var ray=document.createElement('canvas');ray.width=16;ray.height=64;g=ray.getContext('2d');
  q=g.createLinearGradient(0,0,0,64);q.addColorStop(0,'rgba(215,255,225,.95)');q.addColorStop(.5,'rgba(190,245,220,.5)');q.addColorStop(1,'rgba(180,240,215,.3)');g.fillStyle=q;g.fillRect(0,0,16,64);
  g.globalCompositeOperation='destination-in';q=g.createLinearGradient(0,0,16,0);q.addColorStop(0,'rgba(0,0,0,0)');q.addColorStop(.5,'rgba(0,0,0,1)');q.addColorStop(1,'rgba(0,0,0,0)');g.fillStyle=q;g.fillRect(0,0,16,64);
  /* things in the water */
  var N=Math.ceil(w/6)+3,co=[],ox=[],sp=[],ed=[],tu=[];
  function dep(x,f){return y0+(bedY(x)-y0)*f}
  for(i=0;i<66;i++){x=R()*w;co.push(i<7?{x:x,y:y0-10-R()*y0*.7,vx:8,vy:-9,wx:0,wy:0,st:1,life:.1+R()*.6,cr:0}:{x:x,y:dep(x,.06+.9*Math.sqrt(R())),vx:0,vy:0,wx:0,wy:0,st:0,life:0,cr:0})}
  for(i=0;i<38;i++){x=R()*w;ox.push(i<10?{x:x,y:dep(x,.05+.6*R()*R()),vx:0,vy:0,wx:0,wy:0,st:1,cr:0}:{x:x,y:R()*(y0-12),vx:0,vy:0,wx:0,wy:0,st:0,cr:0})}
  for(i=0;i<90;i++){x=R()*w;sp.push({x:x,y:dep(x,.04+.94*R()),vx:0,vy:0,z:.4+R()*.6})}
  for(i=0;i<9;i++)ed.push({x:R()*(w+120)-60,d:.2+R()*.62,r:(15+R()*20)*h/520,s:i%4==3?-1:1,y:0,r2:0});
  for(i=0;i<34;i++){x=R()*w;tu.push({x:x,l:6+R()*13,ph:R()*6.28,f:1.5+R()*2,c:i%3})}
  return{R:R,bedY:bedY,y0:y0,yb:yb,air:air,ah:ah,bed:bed,top:top,bh:bh,ray:ray,N:N,ys:new Float32Array(N),fo:new Float32Array(N),co:co,ox:ox,sp:sp,ed:ed,tu:tu,ph:R()*50,f:-1}},
 frame:function(s,g,w,h,t,dt,ui){
  var F=ui.a==null?.45:ui.a,C=HSM.clamp,R=s.R,i,j,m,x,y,e,N=s.N,ys=s.ys,fo=s.fo,y0=s.y0,bedY=s.bedY;
  if(s.f<0)s.f=F;s.f+=(F-s.f)*Math.min(1,dt*3);F=s.f;
  /* the surface: long swells always, short ripples as the flow quickens. Where it bulges it works as a lens. */
  s.ph+=dt*(.45+1.5*F);
  var A=1.1+6.2*F,a4=.26*F,p=s.ph,nk=.000784+.00205+.0041+a4*.0529;
  for(i=0;i<N;i++){x=(i-1)*6;var s1=Math.sin(x*.028-p*.5),s2=Math.sin(x*.061-p*.9+1.3),s3=Math.sin(x*.117+p*1.3+2.1),s4=Math.sin(x*.23-p*3.1+.7);
   ys[i]=y0+A*(s1+.55*s2+.3*s3+a4*s4);var f=-(.000784*s1+.00205*s2+.0041*s3+a4*.0529*s4)/nk;f=C(.5+.95*f,0,1);fo[i]=f*f}
  function surf(x){var k=x/6+1;if(k<0)k=0;if(k>N-1.001)k=N-1.001;var a=k|0;return ys[a]+(ys[a+1]-ys[a])*(k-a)}
  /* the flow: fast at the top, held back by the bed, with eddies shed from the bed tumbling through it */
  var U=13+150*F,G=3+62*Math.pow(F,1.15),sig=3+34*F,VX=0,VY=0,E=s.ed;
  for(j=0;j<E.length;j++){e=E[j];e.x+=U*(.1+.9*Math.pow(1-e.d,.45))*.7*dt;if(e.x>w+70){e.x=-70;e.d=.2+R()*.62}var sy=surf(e.x);e.y=sy+(bedY(e.x)-sy)*e.d;e.r2=e.r*e.r}
  function vel(x,y){var sy=surf(x),by=bedY(x),d=C((by-y)/(by-sy),0,1),u=U*(.1+.9*Math.pow(d,.45)),v=0,j,e,dx,dy,rr,q;
   for(j=0;j<E.length;j++){e=E[j];dx=x-e.x;dy=y-e.y;rr=dx*dx+dy*dy;if(rr>e.r2*8)continue;q=G*e.s*Math.exp(.5-rr/(2*e.r2))/e.r;u-=dy*q;v+=dx*q}
   VX=u;VY=v}
  var ou=Math.sqrt(12*4*dt);
  function swim(m){vel(m.x,m.y);m.wx+=-m.wx*2*dt+(R()-.5)*ou*sig*.5;m.wy+=-m.wy*2*dt+(R()-.5)*ou*sig;m.vx=VX+m.wx;m.vy=VY+m.wy;m.x+=m.vx*dt;m.y+=m.vy*dt;
   var sy=surf(m.x),by=bedY(m.x);if(m.y<sy+3.5){m.y=sy+3.5;if(m.wy<0)m.wy*=-.4}if(m.y>by-7){m.y=by-7;if(m.wy>0)m.wy*=-.4}}
  var kx=.09+2.3*Math.pow(F,1.3),wind=5+9*F;
  /* air, then the far edge of the water's skin, then the water itself */
  g.drawImage(s.air,0,0,w,s.ah);
  g.beginPath();g.moveTo(-6,ys[0]);for(i=1;i<N;i++)g.lineTo((i-1)*6,ys[i]);for(i=N-1;i>=0;i--)g.lineTo((i-1)*6,y0+(ys[Math.min(N-1,i+5)]-y0)*.6-8);g.closePath();
  var q=g.createLinearGradient(0,y0-14,0,y0+8);q.addColorStop(0,'rgba(196,232,214,.5)');q.addColorStop(1,'rgba(110,190,176,.75)');g.fillStyle=q;g.fill();
  g.beginPath();g.moveTo(-6,ys[0]);for(i=1;i<N;i++)g.lineTo((i-1)*6,ys[i]);g.lineTo(w+6,h);g.lineTo(-6,h);g.closePath();
  q=g.createLinearGradient(0,y0-8,0,s.yb+16);q.addColorStop(0,'rgb(126,204,186)');q.addColorStop(.1,'rgb(66,156,150)');q.addColorStop(.55,'rgb(27,100,106)');q.addColorStop(1,'rgb(12,56,64)');g.fillStyle=q;g.fill();
  /* sunlight, focused and scattered by the surface, slanting down to the bed */
  var sl=.34,con=.45+.55*F;
  g.globalCompositeOperation='lighter';g.save();g.transform(1,0,0,1,0,0);
  for(i=0;i<N;i++){x=(i-1)*6;var I=(1-con)*.3+con*fo[i],dp=bedY(x+sl*(s.yb-y0))-ys[i]+3;g.globalAlpha=.05+.3*I;
   g.setTransform?0:0;g.save();g.transform(1,0,sl,1,x-sl*ys[i],0);g.drawImage(s.ray,-8,ys[i],16,dp);g.restore()}
  g.restore();g.globalAlpha=1;g.globalCompositeOperation='source-over';
  /* fine drifting specks show the current */
  g.lineCap='round';g.lineWidth=1.15;g.strokeStyle='rgba(222,246,236,.34)';g.beginPath();
  for(i=0;i<s.sp.length;i++){m=s.sp[i];vel(m.x,m.y);m.x+=VX*m.z*dt;m.y+=VY*dt;if(m.x>w+8){m.x=-8;m.y=surf(0)+(bedY(0)-surf(0))*(.04+.94*R())}
   sy=surf(m.x);var by=bedY(m.x);if(m.y<sy+2)m.y=sy+2;if(m.y>by-3)m.y=by-3;
   g.moveTo(m.x,m.y);g.lineTo(m.x-VX*m.z*.055-.6,m.y-VY*.055)}
  g.stroke();
  /* the bed, the light playing on it, and the film's filaments leaning with the current */
  g.drawImage(s.bed,0,s.top,w,s.bh);
  g.globalCompositeOperation='lighter';var cb=HSM.blob('rgba(255,242,180,1)',16);
  for(i=0;i<N;i++){x=(i-1)*6;I=(1-con)*.3+con*fo[i];var xb=x+sl*(s.yb-ys[i]);g.globalAlpha=.07+.5*I;g.drawImage(cb,xb-15,bedY(xb)-6,30,13)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  for(j=0;j<3;j++){g.beginPath();for(i=j;i<s.tu.length;i+=3){m=s.tu[i];y=bedY(m.x);var sw=Math.sin(t*m.f*(.5+F)+m.ph)*m.l*.12*(.3+F),tx=m.x+m.l*(.2+.75*F)+sw,ty=y-m.l*(1-.6*F)+sw*.4;
    g.moveTo(m.x,y+1);g.quadraticCurveTo(m.x+m.l*.1,y-m.l*(.6-.25*F),tx,ty)}
   g.lineWidth=1.3;g.strokeStyle=['rgba(132,168,58,.85)','rgba(88,124,44,.85)','rgba(160,150,70,.8)'][j];g.stroke()}
  /* the gases. CO2 leaves from the top few moments of water; oxygen enters the same way */
  var co=s.co,ox=s.ox;
  for(i=0;i<co.length;i++){m=co[i];if(m.cr>0)m.cr-=dt*1.4;
   if(m.st==0){swim(m);if(m.x>w+10||m.x<-14){m.x=-10;m.y=surf(0)+(bedY(0)-surf(0))*(.06+.9*Math.sqrt(R()))}
    else if(m.y<surf(m.x)+10&&R()<kx*dt){m.st=1;m.cr=1;m.life=0;m.vy=-34;m.vx=m.vx*.25}}
   else{m.life+=dt/4.6;m.vy+=(-10-m.vy)*dt*1.6+(R()-.5)*ou*5;m.vx+=(wind-m.vx)*dt+(R()-.5)*ou*6;m.x+=m.vx*dt;m.y+=m.vy*dt;
    if(m.life>1||m.y<-12){m.st=0;m.x=-10;m.wx=m.wy=0;m.y=surf(0)+(bedY(0)-surf(0))*(.06+.9*Math.sqrt(R()))}}}
  for(i=0;i<ox.length;i++){m=ox[i];if(m.cr>0)m.cr-=dt*1.4;
   if(m.st==1){swim(m);if(m.x>w+10){m.st=0;m.x=R()*w;m.y=-10;m.vx=wind;m.vy=10}}
   else{m.vx+=(wind-m.vx)*dt+(R()-.5)*ou*9;m.vy+=(1.5-m.vy)*dt*.8+(R()-.5)*ou*9;m.x+=m.vx*dt;m.y+=m.vy*dt;
    if(m.x>w+10)m.x=-10;if(m.x<-12)m.x=w+8;if(m.y<-12){m.y=-12;m.vy=Math.abs(m.vy)}
    sy=surf(m.x);if(m.y>sy-4){if(m.y>sy-12&&R()<kx*.9*dt*8||0){}m.y=sy-4;m.vy=-Math.abs(m.vy)*.5}
    if(m.y>sy-11&&R()<kx*.9*dt){m.st=1;m.cr=1;m.y=sy+4;m.wy=24;m.wx=0}}}
  var set=[co,ox],col=['rgba(255,196,120,1)','rgba(150,225,255,1)'],ring=['rgba(110,60,10,.6)','rgba(20,60,90,.55)'],trail=['rgba(255,200,130,.4)','rgba(160,228,255,.4)'],k;
  for(k=0;k<2;k++){var L=set[k],b=HSM.blob(col[k],16,.3),r=k?5:4.5;
   g.lineWidth=2.2;g.strokeStyle=trail[k];g.beginPath();for(i=0;i<L.length;i++){m=L[i];g.moveTo(m.x,m.y);g.lineTo(m.x-m.vx*.075,m.y-m.vy*.075)}g.stroke();
   for(i=0;i<L.length;i++){m=L[i];var al=1;if(k==0&&m.st==1)al=1-HSM.smooth(.55,1,m.life);
    if(m.cr>0){g.globalCompositeOperation='lighter';g.globalAlpha=m.cr*.55*al;g.drawImage(b,m.x-r*3,m.y-r*3,r*6,r*6);g.globalCompositeOperation='source-over'}
    g.globalAlpha=.92*al;g.drawImage(b,m.x-r,m.y-r,r*2,r*2);m.al=al}
   g.globalAlpha=1;g.lineWidth=1;g.strokeStyle=ring[k];g.beginPath();for(i=0;i<L.length;i++){m=L[i];if(m.al<.5)continue;g.moveTo(m.x+r*.62,m.y);g.arc(m.x,m.y,r*.62,0,6.283)}g.stroke()}
  /* the skin of the water, catching the sun */
  g.lineJoin='round';g.beginPath();g.moveTo(-6,ys[0]);for(i=1;i<N;i++)g.lineTo((i-1)*6,ys[i]);
  g.lineWidth=9;g.strokeStyle='rgba(190,250,232,.16)';g.stroke();g.lineWidth=1.8;g.strokeStyle='rgba(240,255,248,.82)';g.stroke();
  g.globalCompositeOperation='lighter';var gb=HSM.blob('rgba(255,252,225,1)',16);
  for(i=1;i<N;i+=2){I=fo[i];if(I<.3)continue;var gs=5+13*I;g.globalAlpha=(I-.3)*.9;g.drawImage(gb,(i-1)*6-gs*1.6,ys[i]-gs*.5-1,gs*3.2,gs)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  HSM.vignette(g,w,h,.38);
  g.drawImage(HSM.blob(col[0],16,.3),10,13,10,10);HSM.label(g,25,18,'CO₂ leaving the water');
  g.drawImage(HSM.blob(col[1],16,.3),w-20,13,10,10);HSM.label(g,w-25,18,'oxygen going in','right');
  HSM.label(g,12,h-14,'rock, with its living film');
 }});
