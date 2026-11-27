/* A root finds the crack. Cut-away of a crack between two rock faces, part filled with dust and humus, each
   grain wearing a film of water. The view travels down with the root tip, so the growth never has to cut:
   the cap pushes grains aside, the cells behind it stretch (cross walls drawn in the root's own material
   frame: short at the tip, long and still once grown), and only where stretching has stopped do root hairs
   sprout and thread between the grains. Water moves from the films into the hairs and up the core.
   The toggle is a dry spell: films shrink to thin skins, the hairs lose their turgor, growth stalls. */
HSM.sim('root',{
 toggle:{off:'Dry spell',on:'Rain'},
 setup:function(w,h,R){
  var k=w/390,D=2,Hp=Math.round(h*1.4),W=6.28318/Hp,Rm=14.5*k,tipY=h*.64,sm=HSM.smooth,i,n,j,q,g,x,y;
  function cc(Y){return w*.5+k*(15*Math.sin(Y*W+.4)+7*Math.sin(2*Y*W+2))}
  function rx(Y){return cc(Y)+k*4*Math.sin(3*Y*W+1)}
  function xL(Y){return cc(Y)-k*(66+9*Math.sin(2*Y*W+1.2)+5*Math.sin(3*Y*W+4)+3*Math.sin(7*Y*W)+2*Math.sin(13*Y*W+2)+1.2*Math.sin(29*Y*W))}
  function xR(Y){return cc(Y)+k*(66+8*Math.sin(2*Y*W+3.2)+5*Math.sin(4*Y*W+.5)+3*Math.sin(6*Y*W+1)+2*Math.sin(11*Y*W+5)+1.2*Math.sin(31*Y*W))}
  function prof(d){return d<=0?0:Rm*(.84+.16*sm(25*k,120*k,d))*Math.min(1,Math.sqrt(d/(24*k)))}
  function cv(W_,H_){var e=document.createElement('canvas');e.width=Math.ceil(W_);e.height=Math.ceil(H_);return e}
  function wd(a){return a>Hp/2?a-Hp:a<-Hp/2?a+Hp:a}

  /* rock faces: foliated, speckled, a few mica glints. Everything repeats over Hp so the descent is endless */
  var rock=cv(w*D,Hp*D);g=rock.getContext('2d');g.scale(D,D);
  q=g.createLinearGradient(0,0,w,0);q.addColorStop(0,'#26292b');q.addColorStop(.28,'#4b4c4a');q.addColorStop(.72,'#434445');q.addColorStop(1,'#1f2224');g.fillStyle=q;g.fillRect(0,0,w,Hp);
  for(i=0;i<680;i++){x=R()*w;y=R()*Hp;var L=44+R()*130,a=(8+R()*26)*k,b=(1.8+R()*5)*k,rot=-.55+(R()-.5)*.2,wm=R()<.3;
   g.fillStyle='rgba('+(L+(wm?20:0)|0)+','+(L+(wm?7:2)|0)+','+(L+(wm?-10:7)|0)+','+(.05+R()*.13)+')';
   for(n=-1;n<2;n++){g.beginPath();g.ellipse(x,y+n*Hp,a,b,rot,0,6.283);g.fill()}}
  for(i=0;i<170;i++){x=R()*w;y=R()*Hp;a=(4+R()*12)*k;rot=-.55+(R()-.5)*.25;g.lineWidth=(.6+R()*.9)*k;g.strokeStyle=R()<.6?'rgba(240,224,186,'+(.12+R()*.4)+')':'rgba(12,10,8,'+(.3+R()*.4)+')';
   for(n=-1;n<2;n++){g.beginPath();g.moveTo(x-Math.cos(rot)*a,y+n*Hp-Math.sin(rot)*a);g.lineTo(x+Math.cos(rot)*a,y+n*Hp+Math.sin(rot)*a);g.stroke()}}
  for(i=0;i<5200;i++){g.fillStyle=R()<.5?'rgba(255,255,255,'+(.03+R()*.08)+')':'rgba(0,0,0,'+(.05+R()*.12)+')';g.fillRect(R()*w,R()*Hp,(.6+R())*k,(.6+R())*k)}

  /* dust and humus. Grains that lie in the root's path are kept apart: the tip will shoulder them aside */
  var grains=[],cg=[],bs=24*k,bins=[],nb=Math.ceil(Hp/bs);for(i=0;i<nb;i++)bins.push([]);
  var MIN=[['#c4c0b4','#57544e'],['#c9ac80','#5b4830'],['#d4dcde','#66727a'],['#97938c','#383634'],['#b98f66','#4c3220']],HUM=['#604632','#4c3826','#6e4e2e','#3e2c1e'];
  function mk(x,Y,r){var hum=R()<(r<4*k?.5:.66),m=hum?8+Math.floor(R()*3):4+Math.floor(R()*3),V=[],a0=R()*6.283,c=hum?[HUM[Math.floor(R()*4)],'#150d08']:MIN[Math.floor(R()*5)];
   for(var v=0;v<m;v++)V.push([a0+(v+(R()-.5)*.5)/m*6.283,hum?.74+R()*.3:.66+R()*.4]);return{x:x,Y:Y,r:r,hum:hum,V:V,c:c}}
  for(i=0;i<5200;i++){var Y=R()*Hp,r=(2.3+Math.pow(R(),2.2)*6.4)*k,l=xL(Y)+r*.7,rr=xR(Y)-r*.7;x=l+R()*(rr-l);var ok=1,b0=Math.floor(Y/bs);
   for(n=-1;n<2&&ok;n++){var B=bins[(b0+n+nb)%nb];for(j=0;j<B.length;j++){var o=B[j],dx=o.x-x,dy=wd(o.Y-Y),dm=o.r+r+1.1*k;if(dx*dx+dy*dy<dm*dm){ok=0;break}}}
   if(!ok)continue;
   if(Math.abs(x-rx(Y))<Rm+r+1.5*k){if(r<6.5*k&&cg.length<150){var G=mk(x,Y,r);cg.push(G);bins[b0].push(G);G.ch=1}continue}
   G=mk(x,Y,r);grains.push(G);bins[b0].push(G)}
  function grain(g,G,x,y,wet){var V=G.V,r=G.r,q,v;g.beginPath();for(v=0;v<V.length;v++)g.lineTo(x+Math.cos(V[v][0])*r*V[v][1],y+Math.sin(V[v][0])*r*V[v][1]);g.closePath();
   if(G.hum)q=g.createRadialGradient(x-r*.35,y-r*.4,r*.05,x,y,r*1.15);else q=g.createLinearGradient(x-r*.7,y-r*.8,x+r*.6,y+r*.8);
   q.addColorStop(0,G.c[0]);q.addColorStop(1,G.c[1]);g.fillStyle=q;g.fill();g.lineWidth=.6*k;g.strokeStyle=G.hum?'rgba(0,0,0,.55)':'rgba(255,255,255,.2)';g.stroke();
   if(wet){g.fillStyle='rgba(225,242,255,.6)';g.beginPath();g.arc(x-r*.42,y-r*.5,Math.min(1.1*k,r*.2),0,6.283);g.fill()}}
  function wall(g){g.beginPath();g.moveTo(xL(-30),-30);for(var Y=-27;Y<=Hp+30;Y+=3)g.lineTo(xL(Y),Y);for(Y=Hp+30;Y>=-30;Y-=3)g.lineTo(xR(Y),Y);g.closePath()}
  /* the whole static scene twice: with thin skins of water (dry) and with fat films and menisci (wet) */
  function build(wet){var e=cv(w*D,Hp*D),g=e.getContext('2d'),q,n,i,G,rr;g.scale(D,D);g.drawImage(rock,0,0,w,Hp);
   wall(g);g.lineWidth=9*k;g.strokeStyle='rgba(235,225,205,.05)';g.stroke();
   g.save();wall(g);g.clip();q=g.createLinearGradient(w*.5-80*k,0,w*.5+80*k,0);q.addColorStop(0,'#060504');q.addColorStop(.5,'#1a120c');q.addColorStop(1,'#060504');g.fillStyle=q;g.fillRect(0,0,w,Hp);
   wall(g);g.lineWidth=18*k;g.strokeStyle='rgba(0,0,0,.4)';g.stroke();
   wall(g);g.lineWidth=(wet?6:2.4)*k;g.strokeStyle=wet?'rgba(70,160,230,.4)':'rgba(60,140,200,.2)';g.stroke();
   var bl=HSM.blob(wet?'rgba(84,156,212,.44)':'rgba(64,140,200,.28)',32,.8);
   for(i=0;i<grains.length;i++){G=grains[i];rr=G.r+(wet?4:1.5)*k;for(n=-1;n<2;n++)g.drawImage(bl,G.x-rr,G.Y+n*Hp-rr,rr*2,rr*2)}
   if(wet){g.globalCompositeOperation='lighter';bl=HSM.blob('rgba(30,90,150,.2)',32,.3);for(i=0;i<grains.length;i+=3){G=grains[i];rr=G.r+9*k;for(n=-1;n<2;n++)g.drawImage(bl,G.x-rr,G.Y+n*Hp-rr,rr*2,rr*2)}g.globalCompositeOperation='source-over'}
   for(i=0;i<grains.length;i++){G=grains[i];for(n=-1;n<2;n++)if(G.Y+n*Hp>-12*k&&G.Y+n*Hp<Hp+12*k)grain(g,G,G.x,G.Y+n*Hp,wet)}
   g.restore();wall(g);g.lineWidth=1.2*k;g.strokeStyle='rgba(222,214,196,.42)';g.stroke();return e}
  var A=build(0),Bw=build(1);
  cg.forEach(function(G){var S=G.r*1.2+2,e=cv(S*2*D,S*2*D),q=e.getContext('2d');q.scale(D,D);grain(q,G,S,S,0);G.sp=e;G.S=S});

  /* root hairs: each threads outward from the root, steering round the grains it meets */
  var Nh=Math.round(Hp/(4.3*k)),dh=Hp/Nh,hairs=[];
  for(j=0;j<Nh;j++){var side=j&1?1:-1,L2=(15+R()*18)*k,st=L2/8,ang=(R()-.5)*1.1,px=[],py=[];Y=j*dh;x=rx(Y)+side*Rm*.9;y=Y;px.push(x);py.push(0);
   for(n=0;n<8;n++){var ddx=side*Math.cos(ang),ddy=Math.sin(ang),b1=Math.floor((((y%Hp)+Hp)%Hp)/bs);
    for(var bb=-1;bb<2;bb++){B=bins[(b1+bb+nb)%nb];for(i=0;i<B.length;i++){o=B[i];if(o.ch)continue;var ex=x-o.x,ey=wd((((y%Hp)+Hp)%Hp)-o.Y),d=Math.sqrt(ex*ex+ey*ey)+.01,lim=o.r+2.2*k+st;if(d<lim){var f=(lim-d)/lim*1.8;ddx+=ex/d*f;ddy+=ey/d*f}}}
    if(ddx*side<.3)ddx=.3*side;d=Math.sqrt(ddx*ddx+ddy*ddy);ddx/=d;ddy/=d;x+=ddx*st;y+=ddy*st;x=HSM.clamp(x,xL(y)+2*k,xR(y)-2*k);px.push(x);py.push(y-Y);
    ang=Math.atan2(ddy,ddx*side)*.8+(R()-.5)*.32}
   hairs.push({px:px,py:py,side:side})}

  /* the root itself, drawn straight; the frame bends it along its path */
  var SW=2*(Rm+10*k),SH=tipY+9*k,sp=cv(SW*D,SH*D),cx=SW/2;g=sp.getContext('2d');g.scale(D,D);g.lineCap='round';g.lineJoin='round';
  g.beginPath();g.moveTo(cx,-4);g.lineTo(cx,tipY-8*k);g.lineWidth=Rm*3;g.strokeStyle='rgba(255,246,206,.07)';g.stroke();g.lineWidth=Rm*2.5;g.stroke();
  function body(){g.beginPath();g.moveTo(cx-prof(tipY+4),-4);for(var y=-2;y<tipY;y+=1.5)g.lineTo(cx-prof(tipY-y),y);g.lineTo(cx,tipY);for(y=tipY-.5;y>=-4;y-=1.5)g.lineTo(cx+prof(tipY-y),y);g.closePath()}
  body();q=g.createLinearGradient(cx-Rm,0,cx+Rm,0);q.addColorStop(0,'rgba(150,138,100,.96)');q.addColorStop(.16,'rgba(222,214,180,.98)');q.addColorStop(.4,'#fffcec');q.addColorStop(.62,'#f4eed4');q.addColorStop(.86,'rgba(206,196,160,.98)');q.addColorStop(1,'rgba(132,120,88,.96)');g.fillStyle=q;g.fill();
  g.save();body();g.clip();
  q=g.createLinearGradient(0,tipY,0,0);function zs(d,c){q.addColorStop(Math.min(1,d*k/tipY),c)}
  zs(0,'rgba(226,190,96,.66)');zs(30,'rgba(234,208,128,.42)');zs(52,'rgba(255,255,255,.1)');zs(76,'rgba(255,255,255,.16)');zs(112,'rgba(255,255,255,0)');zs(175,'rgba(176,160,112,.26)');zs(999,'rgba(170,152,104,.3)');g.fillStyle=q;g.fillRect(0,0,SW,SH);
  q=g.createLinearGradient(0,tipY-38*k,0,tipY-96*k);q.addColorStop(0,'rgba(150,132,70,0)');q.addColorStop(1,'rgba(150,132,70,.5)');
  g.fillStyle=q;g.fillRect(cx-Rm*.3,0,Rm*.6,tipY-38*k);g.strokeStyle=q;g.lineWidth=1.3*k;g.lineCap='butt';
  for(i=-1;i<2;i++){g.beginPath();g.moveTo(cx+i*Rm*.2,0);g.lineTo(cx+i*Rm*.2,tipY-38*k);g.stroke()}
  g.lineWidth=.6*k;g.strokeStyle='rgba(130,112,70,.34)';for(i=0;i<4;i++){var fr=[-.6,-.2,.2,.6][i];g.beginPath();for(y=-2;y<tipY-5*k;y+=2)g.lineTo(cx+fr*prof(tipY-y)*.96,y);g.stroke()}
  for(i=0;i<90;i++){var d=(5+R()*34)*k;g.fillStyle='rgba(150,112,40,'+(.22+R()*.3)+')';g.beginPath();g.arc(cx+(R()-.5)*1.7*prof(d),tipY-d,(.5+R()*.5)*k,0,6.283);g.fill()}
  g.lineCap='round';g.lineWidth=1.5*k;g.strokeStyle='rgba(255,255,255,.4)';g.beginPath();for(y=-2;y<tipY-13*k;y+=2)g.lineTo(cx-.52*prof(tipY-y),y);g.stroke();
  g.restore();body();g.lineWidth=.9*k;g.strokeStyle='rgba(104,90,56,.65)';g.stroke();
  /* root cap: a thimble of tougher, expendable cells, shedding at its edge */
  function capw(d){return Rm*.88*Math.sqrt(Math.max(0,(d+5*k)/(25*k)))}
  g.beginPath();for(d=20*k;d>=-5*k;d-=1)g.lineTo(cx-capw(d),tipY-d);for(d=-5*k;d<=20*k;d+=1)g.lineTo(cx+capw(d),tipY-d);g.quadraticCurveTo(cx,tipY-5*k,cx-capw(20*k),tipY-20*k);g.closePath();
  q=g.createRadialGradient(cx-2*k,tipY-9*k,1,cx,tipY-6*k,22*k);q.addColorStop(0,'rgba(240,198,118,.84)');q.addColorStop(1,'rgba(134,90,44,.92)');g.fillStyle=q;g.fill();g.lineWidth=.8*k;g.strokeStyle='rgba(84,54,24,.75)';g.stroke();
  g.save();g.clip();for(i=0;i<36;i++){g.fillStyle='rgba(100,64,26,'+(.25+R()*.3)+')';g.beginPath();g.arc(cx+(R()-.5)*Rm*1.8,tipY+5*k-R()*25*k,(.6+R()*.6)*k,0,6.283);g.fill()}
  g.lineWidth=1.4*k;g.strokeStyle='rgba(255,236,190,.4)';g.beginPath();for(d=16*k;d>=-1*k;d-=1)g.lineTo(cx-capw(d)*.6,tipY-d);g.stroke();g.restore();
  for(i=0;i<7;i++){d=(2+R()*19)*k;var sd=i&1?1:-1;g.save();g.translate(cx+sd*(capw(d)+(1.6+R()*3)*k),tipY-d);g.rotate(sd*.5+(R()-.5));g.beginPath();g.ellipse(0,0,2.1*k,1.1*k,0,0,6.283);g.fillStyle='rgba(214,170,104,.7)';g.fill();g.lineWidth=.5*k;g.strokeStyle='rgba(84,54,24,.6)';g.stroke();g.restore()}

  var e=cv(24*D,24*D);q=e.getContext('2d');q.scale(D,D);q.drawImage(HSM.blob('rgba(120,206,255,1)',32,.3),0,0,24,24);q.lineWidth=1.6;q.strokeStyle='rgba(14,60,120,.7)';q.beginPath();q.arc(12,12,7.2,0,6.283);q.stroke();
  var s={k:k,R:R,A:A,B:Bw,Hp:Hp,cg:cg,hairs:hairs,Nh:Nh,dh:dh,sp:sp,SW:SW,SH:SH,Rm:Rm,tipY:tipY,rx:rx,xL:xL,xR:xR,prof:prof,ms:e,scroll:0,wet:null,motes:[],perc:[]};
  for(i=0;i<36;i++)s.motes.push({st:-1,r:R()});
  for(i=0;i<20;i++)s.perc.push({y:R()*h,sd:R()<.5?-1:1,o:(3+R()*20)*k,v:(34+R()*40)*k,ph:R()*6.28});
  return s},

 frame:function(s,g,w,h,t,dt,ui){
  var sm=HSM.smooth,k=s.k,R=s.R,tgt=ui.on?0:1,i,j,n,m,x,y,H;
  if(s.wet==null)s.wet=tgt;s.wet+=(tgt-s.wet)*Math.min(1,dt*.6);
  var wet=s.wet,limp=1-sm(.1,.7,wet),Hp=s.Hp,tipY=s.tipY,Rm=s.Rm,Nh=s.Nh,dh=s.dh,d0=74*k;
  s.scroll+=dt*8.5*k*sm(.25,.9,wet);var sc=s.scroll,off=sc%Hp;
  /* soil and rock, dry state with the wet one laid over it */
  g.drawImage(s.A,0,-off,w,Hp);if(Hp-off<h)g.drawImage(s.A,0,Hp-off,w,Hp);
  if(wet>.01){g.globalAlpha=wet;g.drawImage(s.B,0,-off,w,Hp);if(Hp-off<h)g.drawImage(s.B,0,Hp-off,w,Hp);g.globalAlpha=1}
  /* rain trickling down the faces of the crack */
  for(i=0;i<s.perc.length;i++){m=s.perc[i];m.y+=m.v*dt*(.2+.8*wet);if(m.y>h+10){m.y=-10-R()*60;m.o=(3+R()*20)*k}
   var Ys=m.y+sc;x=(m.sd<0?s.xL(Ys)+m.o:s.xR(Ys)-m.o)+Math.sin(m.y*.06/k+m.ph)*3*k;g.globalAlpha=.75*sm(.45,1,wet);
   if(g.globalAlpha>.01)g.drawImage(HSM.blob('rgba(150,214,255,1)',16,.4),x-2.6*k,m.y-6*k,5.2*k,12*k)}
  g.globalAlpha=1;
  /* grains in the root's path, pushed aside as the tip arrives */
  for(i=0;i<s.cg.length;i++){var G=s.cg[i];y=G.Y-off;if(y<-20)y+=Hp;if(y>h+20)continue;var sd=tipY-y,need=sd>-5*k?Math.max(s.prof(sd),sd<20*k?Rm*.88*Math.sqrt((sd+5*k)/(25*k)):0)+G.r*.9:0,cxr=s.rx(y+sc),o=G.x-cxr;
   x=Math.abs(o)<need?cxr+(o<0?-need:need):G.x;
   var rr=G.r+(1.5+2.5*wet)*k;g.globalAlpha=.6+.4*wet;g.drawImage(HSM.blob('rgba(84,156,212,.44)',32,.8),x-rr,y-rr,rr*2,rr*2);g.globalAlpha=1;
   g.drawImage(G.sp,x-G.S,y-G.S,G.S*2,G.S*2)}
  /* root hairs: grown where they stand, so they pass upward with the soil */
  function hp(j,kf,out){var H=s.hairs[((j%Nh)+Nh)%Nh],k0=Math.min(7,Math.floor(kf)),fr=Math.min(1,kf-k0),ax=H.px[0],fk=kf/8;
   var hx=H.px[k0]+(H.px[k0+1]-H.px[k0])*fr,hy=H.py[k0]+(H.py[k0+1]-H.py[k0])*fr;
   out[0]=ax+(hx-ax)*(1-.28*limp*fk)+limp*Math.sin(fk*4+j)*2.2*k*fk;out[1]=j*dh-sc+hy*(1-.35*limp)+limp*fk*fk*12*k}
  var P=[0,0],j0=Math.ceil((sc-34*k)/dh),j1=Math.floor((sc+tipY-d0)/dh);
  g.lineCap='round';g.lineJoin='round';g.beginPath();
  for(j=j0;j<=j1;j++){var f=sm(0,90*k,sc+tipY-j*dh-d0)*8;if(f<.15)continue;hp(j,0,P);g.moveTo(P[0],P[1]);var qx=P[0],qy=P[1];for(n=1;n<=f;n++){hp(j,n,P);g.quadraticCurveTo(qx,qy,(qx+P[0])/2,(qy+P[1])/2);qx=P[0];qy=P[1]}hp(j,f,P);g.quadraticCurveTo(qx,qy,P[0],P[1])}
  g.lineWidth=3.6*k;g.strokeStyle='rgba(190,226,255,'+(.05+.09*wet)+')';g.stroke();
  g.lineWidth=1*k;g.strokeStyle='rgba('+Math.round(232+14*wet)+','+Math.round(218+22*wet)+','+Math.round(184+36*wet)+','+(.55+.27*wet)+')';g.stroke();
  /* the root, bent along its path in strips */
  var SW=s.SW;for(y=0;y<s.SH;y+=4)g.drawImage(s.sp,0,y*2,SW*2,8,s.rx(y+sc)-SW/2,y,SW,4);
  /* cross walls in the root's own frame: crowded where cells divide, drawn apart where they stretch, then still */
  var kk=30*k,Dl=15*k;g.beginPath();
  for(var fi=0;fi<5;fi++){var a0=-1+fi*.4,ph=((fi*.37)%1)*Dl;
   for(var sv=-1.2*kk+((sc+ph)%Dl);;sv+=Dl){var d=kk*Math.log(1+Math.exp(sv/kk));if(d>tipY+4)break;y=tipY-d;var Rr=s.prof(d)*.96;x=s.rx(y+sc);g.moveTo(x+a0*Rr,y);g.lineTo(x+(a0+.4)*Rr,y)}}
  g.lineWidth=.7*k;g.strokeStyle='rgba(126,106,64,.42)';g.stroke();
  /* water: waits in a film at a hair's end, is drawn along the hair, then rises in the core */
  var act=Math.round(s.motes.length*(.3+.7*wet)),ja=Math.ceil((sc+6*k)/dh),jb=Math.floor((sc+tipY-d0-45*k)/dh),r=4*k;
  for(i=0;i<s.motes.length;i++){m=s.motes[i];
   if(m.st<0){m.j=ja+Math.floor(R()*(jb-ja+1));m.st=0;m.w=R()*2.6;m.u=0;if(t<.2&&i%3==0){m.st=2;m.Y=sc+R()*tipY*.7;m.xo=0}}
   if(m.st<2&&m.j*dh-sc<-8*k){m.st=-1;continue}
   if(m.st==0){m.w-=dt*wet;hp(m.j,8,P);x=P[0]+Math.cos(t*1.3+i)*2.4*k+s.hairs[((m.j%Nh)+Nh)%Nh].side*3*k;y=P[1]+Math.sin(t*1.7+i*2)*2.4*k;if(m.w<0&&i<act)m.st=1}
   else if(m.st==1){m.u+=dt*1.15*(.15+.85*wet);if(m.u>=1){m.st=2;m.Y=m.j*dh;m.xo=s.hairs[((m.j%Nh)+Nh)%Nh].px[0]-s.rx(m.Y)}hp(m.j,(1-Math.min(1,m.u))*8,P);x=P[0];y=P[1]}
   if(m.st==2){m.Y-=dt*(46+18*m.r)*k*(.12+.88*wet);m.xo*=1-Math.min(1,dt*3.5);x=s.rx(m.Y)+m.xo+(m.r-.5)*Rm*.4;y=m.Y-sc;if(y<-8){m.st=-1;continue}}
   if(i>=act&&m.st==0)continue;
   g.globalAlpha=(m.st==0?.55+.3*Math.sin(t*2+i):.95)*(.35+.65*wet);g.drawImage(s.ms,x-r,y-r,r*2,r*2)}
  g.globalAlpha=1;
  /* light from the surface above, darkness below */
  var q=g.createLinearGradient(0,0,0,h*.34);q.addColorStop(0,'rgba(255,236,190,.2)');q.addColorStop(1,'rgba(255,236,190,0)');g.globalCompositeOperation='lighter';g.fillStyle=q;g.fillRect(0,0,w,h*.34);g.globalCompositeOperation='source-over';
  q=g.createLinearGradient(0,h*.72,0,h);q.addColorStop(0,'rgba(0,0,0,0)');q.addColorStop(1,'rgba(0,0,0,.5)');g.fillStyle=q;g.fillRect(0,h*.72,w,h*.28);
  HSM.vignette(g,w,h,.4);
  function lead(x1,y1,x2,y2){g.lineWidth=2.6;g.strokeStyle='rgba(6,10,8,.5)';g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.stroke();g.lineWidth=1;g.strokeStyle='rgba(255,255,255,.8)';g.stroke();
   g.fillStyle='rgba(255,255,255,.95)';g.beginPath();g.arc(x2,y2,2,0,6.283);g.fill()}
  var tx=s.rx(tipY+sc);y=tipY+26*k;lead(w-88*k,y-8,tx+5*k,tipY-3*k);HSM.label(g,w-12,y,'root cap','right');
  y=h*.3;lead(96*k,y+2,tx-Rm-13*k,y+34*k);HSM.label(g,10,y-8,'root hairs drink');HSM.label(g,10,y+8,'from films of water');
  HSM.label(g,10,18,limp>.5?'dry spell: growth stalls':'rain: the root pushes on');
 }});
