/* A leaf being taken apart. A fallen leaf in the stream under the microscope (dark field): brown cells
   between a net of tougher veins. Aquatic fungi (hyphomycetes) thread through it; their enzymes dissolve
   the soft cells and leave the lignin-rich veins, a leaf skeleton. The fungi shed four-armed spores
   (tetraradiate conidia), a shape that snags on the next leaf downstream. Some of the leaf's carbon
   leaves as CO2. Weeks are squeezed into half a minute, then a fresh leaf arrives. */
HSM.sim('rot',{
 setup:function(w,h,R){
  var D=w>480?1.5:2,W=Math.round(w*D),H=Math.round(h*D),u=w/390,i,j,n,g,q,x,y;
  function cv(){var c=document.createElement('canvas');c.width=W;c.height=H;c.getContext('2d').scale(D,D);return c}
  /* cells: a jittered honeycomb */
  var cw=w/17,ch=cw*.866,cells=[];
  function jit(px,py){var ii=Math.round(px/(cw*.5)),jj=Math.round(py/(ch*.17)),k=Math.sin(ii*127.1+jj*311.7)*43758.5453,k2=Math.sin(ii*269.5+jj*183.3)*43758.5453;k-=Math.floor(k);k2-=Math.floor(k2);return[px+(k-.5)*cw*.3,py+(k2-.5)*ch*.3]}
  for(var r=-1;r<h/ch+1;r++)for(var c=-1;c<w/cw+1;c++){var ox=c*2+(r&1),oy=r*2,
   P=[[ox*cw*.5,oy*ch*.5+ch*.17],[(ox+1)*cw*.5,oy*ch*.5-ch*.17],[(ox+2)*cw*.5,oy*ch*.5+ch*.17],[(ox+2)*cw*.5,(oy+2)*ch*.5-ch*.17],[(ox+1)*cw*.5,(oy+2)*ch*.5+ch*.17],[ox*cw*.5,(oy+2)*ch*.5-ch*.17]].map(function(p){return jit(p[0],p[1])});
   var cx=0,cy=0;P.forEach(function(p){cx+=p[0]/6;cy+=p[1]/6});cells.push({P:P,cx:cx,cy:cy,t0:1e9,dur:5,k:0,acc:0})}
  function poly(g,P){g.moveTo(P[0][0],P[0][1]);for(var n=1;n<6;n++)g.lineTo(P[n][0],P[n][1]);g.closePath()}
  /* veins: a midrib, side veins curving off it, and a fine net with blind-ending veinlets */
  var V=[];
  function mid(s){return[-12+(w+24)*s,h*.8-h*.6*s+Math.sin(s*3.1)*h*.035]}
  var p=[];for(i=0;i<=24;i++)p.push(mid(i/24));V.push({p:p.slice(0,10),w:8.5*u},{p:p.slice(9,18),w:7*u},{p:p.slice(17),w:5.6*u});
  for(i=0;i<7;i++){var s0=.06+i*.14,m0=mid(s0),m1=mid(s0+.02),sg=i%2?1:-1,a=Math.atan2(m1[1]-m0[1],m1[0]-m0[0])+sg*(.85+R()*.25),pa=[m0.slice()],pb;x=m0[0];y=m0[1];
   for(n=0;n<44;n++){a-=sg*(.026+R()*.012);a+=(R()-.5)*.06;x+=Math.cos(a)*13*u;y+=Math.sin(a)*13*u;pa.push([x,y]);if(x<-20||y<-20||x>w+20||y>h+20)break}
   n=Math.min(pa.length,13);pb=pa.slice(n-1);V.push({p:pa.slice(0,n),w:4.2*u});if(pb.length>1)V.push({p:pb,w:2.9*u})}
  var sp=41*u,GX=Math.ceil(w/sp)+2,GY=Math.ceil(h/sp)+2,Gp=[];
  for(j=0;j<GY;j++)for(i=0;i<GX;i++)Gp.push([(i-.5)*sp+(R()-.5)*sp*.7,(j-.5)*sp+(R()-.5)*sp*.7]);
  function curve(A,B,bend,wd){var mx=(A[0]+B[0])/2+(R()-.5)*bend,my=(A[1]+B[1])/2+(R()-.5)*bend,p=[],k,t;for(k=0;k<=6;k++){t=k/6;p.push([(1-t)*(1-t)*A[0]+2*t*(1-t)*mx+t*t*B[0],(1-t)*(1-t)*A[1]+2*t*(1-t)*my+t*t*B[1]])}V.push({p:p,w:wd})}
  for(j=0;j<GY;j++)for(i=0;i<GX;i++){var A=Gp[j*GX+i];
   if(i<GX-1&&R()<.9)curve(A,Gp[j*GX+i+1],sp*.35,(1.3+R()*1.1)*u);
   if(j<GY-1&&R()<.9)curve(A,Gp[(j+1)*GX+i],sp*.35,(1.3+R()*1.1)*u);
   if(R()<.75){a=R()*6.283;var l=sp*(.28+R()*.3);curve(A,[A[0]+Math.cos(a)*l,A[1]+Math.sin(a)*l],sp*.2,1.05*u)}}
  function veins(g,glow){var k,v,n,pass;g.lineCap='round';g.lineJoin='round';
   for(pass=glow?0:1;pass<4;pass++)for(k=0;k<V.length;k++){v=V[k];g.beginPath();var o=pass==3?-.22*v.w:0;g.moveTo(v.p[0][0]+o,v.p[0][1]+o);for(n=1;n<v.p.length;n++)g.lineTo(v.p[n][0]+o,v.p[n][1]+o);
    g.lineWidth=[v.w+9,v.w+2.4,v.w,Math.max(.5,v.w*.3)][pass];g.strokeStyle=['rgba(255,196,96,.05)','rgba(30,16,6,.82)',glow?'#c79a56':'#b58545','rgba(255,232,170,.55)'][pass];g.stroke()}}
  /* what is left at the end: dark water with the lace of veins, and the faintest ghost of the cell walls */
  var bg=cv();g=bg.getContext('2d');
  q=g.createRadialGradient(w*.3,h*.25,0,w*.4,h*.4,h*.9);q.addColorStop(0,'#16434a');q.addColorStop(.55,'#0a2530');q.addColorStop(1,'#04121a');g.fillStyle=q;g.fillRect(0,0,w,h);
  g.globalCompositeOperation='lighter';for(i=0;i<26;i++){var br=(6+R()*R()*40)*u;g.globalAlpha=.03+R()*.07;g.drawImage(HSM.blob(i%2?'rgba(120,200,190,1)':'rgba(200,220,150,1)',32,.7),R()*w-br,R()*h-br,br*2,br*2)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  g.beginPath();for(i=0;i<cells.length;i++)poly(g,cells[i].P);g.lineWidth=.7;g.strokeStyle='rgba(215,180,120,.085)';g.stroke();
  veins(g,1);
  /* the whole leaf: translucent browns, blotched, each cell lit from within, darker walls */
  var leaf=cv();g=leaf.getContext('2d');g.fillStyle='#3a2110';g.fillRect(0,0,w,h);
  function nz(x,y){return .5+.28*Math.sin(x*.021/u+1.3)*Math.sin(y*.017/u+.5)+.22*Math.sin(x*.05/u+y*.043/u+2)}
  for(i=0;i<cells.length;i++){c=cells[i];var tn=HSM.clamp(nz(c.cx,c.cy)+(R()-.5)*.42,0,1),cr=70+tn*108,cg=38+tn*74,cb=17+tn*34;
   q=g.createRadialGradient(c.cx-cw*.12,c.cy-cw*.14,0,c.cx,c.cy,cw*.62);q.addColorStop(0,'rgb('+Math.round(cr*1.32)+','+Math.round(cg*1.3)+','+Math.round(cb*1.25)+')');q.addColorStop(.6,'rgb('+Math.round(cr)+','+Math.round(cg)+','+Math.round(cb)+')');q.addColorStop(1,'rgb('+Math.round(cr*.62)+','+Math.round(cg*.6)+','+Math.round(cb*.6)+')');
   g.beginPath();poly(g,c.P);g.fillStyle=q;g.fill();
   n=2+Math.floor(R()*5);for(j=0;j<n;j++){a=R()*6.283;var rr=Math.sqrt(R())*cw*.34;g.fillStyle=R()<.75?'rgba(38,18,6,'+(.25+R()*.3)+')':'rgba(240,190,100,'+(.15+R()*.2)+')';g.beginPath();g.arc(c.cx+Math.cos(a)*rr,c.cy+Math.sin(a)*rr,(.6+R()*1.5)*u,0,6.283);g.fill()}}
  g.lineJoin='round';g.beginPath();for(i=0;i<cells.length;i++)poly(g,cells[i].P);g.lineWidth=2.6*u;g.strokeStyle='rgba(36,19,8,.8)';g.stroke();g.lineWidth=.8*u;g.strokeStyle='rgba(214,160,92,.4)';g.stroke();
  for(i=0;i<14;i++){br=(18+R()*46)*u;g.globalAlpha=.12+R()*.2;g.drawImage(HSM.blob(i%3?'rgba(26,12,4,1)':'rgba(240,186,90,1)',32),R()*w-br,R()*h-br,br*2,br*2)}
  g.globalAlpha=1;veins(g,0);
  g.globalCompositeOperation='lighter';g.globalAlpha=.2;g.drawImage(HSM.blob('rgba(255,214,150,1)',64),-w*.35,-w*.4,w*1.3,w*1.3);g.globalAlpha=1;g.globalCompositeOperation='source-over';
  var rem=cv(),hy=cv(),rg=rem.getContext('2d'),hg=hy.getContext('2d');hg.lineCap='round';rg.lineJoin='round';
  var s={R:R,u:u,D:D,bg:bg,leaf:leaf,rem:rem,hy:hy,cells:cells,cw:cw,nodes:[],tips:[],spores:[],gas:[],ck:0,fo:0,nc:0,fin:0,sa:0,ga:0,act:0,
   dust:[],lab:0};
  for(i=0;i<7;i++)s.dust.push({x:R()*w,y:R()*h,r:(10+R()*26)*u,v:(14+R()*22)*u,a:.04+R()*.05});
  var rad2=cw*cw*1.5;
  function node(tp){var k,c,dx,dy,o=1.1*u;
      hg.strokeStyle='rgba(255,250,222,.1)';hg.lineWidth=tp.w+4.5;hg.beginPath();hg.moveTo(tp.lx,tp.ly);hg.lineTo(tp.x,tp.y);hg.stroke();
   hg.strokeStyle='rgba(240,238,216,.92)';hg.lineWidth=tp.w;hg.beginPath();hg.moveTo(tp.lx,tp.ly);hg.lineTo(tp.x,tp.y);hg.stroke();
   hg.globalCompositeOperation='destination-over';hg.strokeStyle='rgba(18,9,3,.42)';hg.lineWidth=tp.w+1.8;hg.beginPath();hg.moveTo(tp.lx+o*.7,tp.ly+o);hg.lineTo(tp.x+o*.7,tp.y+o);hg.stroke();hg.globalCompositeOperation='source-over';
   tp.lx=tp.x;tp.ly=tp.y;s.nodes.push({x:tp.x,y:tp.y});
   for(k=0;k<cells.length;k++){c=cells[k];if(c.t0<1e8)continue;dx=c.cx-tp.x;dy=c.cy-tp.y;if(dx*dx+dy*dy<rad2){c.t0=s.ck+3+R()*6;c.dur=5.5+R()*4}}}
  function tip(x,y,a,wd){return{x:x,y:y,lx:x,ly:y,a:a,c:(R()-.5)*.3,acc:0,w:wd,v:.8+R()*.4}}
  s.step=function(dt){var k,tp,c,d,T=s.tips,nd;s.ck+=dt;
   if(s.ck<25)for(k=T.length-1;k>=0;k--){tp=T[k];tp.a+=(tp.c+(R()-.5)*3.6)*dt;if(R()<dt*.5)tp.c=(R()-.5)*.36;d=13*u*tp.v*dt;tp.x+=Math.cos(tp.a)*d;tp.y+=Math.sin(tp.a)*d;tp.acc+=d;
    if(tp.acc>=5*u){tp.acc=0;node(tp);if(T.length<24&&R()<.085)T.push(tip(tp.x,tp.y,tp.a+(R()<.5?1:-1)*(.55+R()*.7),Math.max(1*u,tp.w*.88)))}
    if(tp.x<-14||tp.y<-14||tp.x>w+14||tp.y>h+14){if(s.ck>19){T.splice(k,1);continue}nd=s.nodes[Math.floor(R()*s.nodes.length)];T[k]=tip(nd.x,nd.y,R()*6.283,1.15*u)}}
   if(!s.fin&&s.ck>23){s.fin=1;for(k=0;k<cells.length;k++)if(cells[k].t0>1e8){cells[k].t0=23+R()*4;cells[k].dur=4+R()*3.5}}
   /* digestion: each cell reached by a thread fades out in steps, walls last */
   rg.globalCompositeOperation='destination-out';s.act=0;
   for(k=0;k<cells.length;k++){c=cells[k];if(c.k>=17||c.t0>s.ck)continue;s.act++;c.acc+=dt;d=c.dur/17;
    while(c.acc>=d&&c.k<17){c.acc-=d;c.k++;rg.globalAlpha=c.k==17?1:.2;rg.beginPath();poly(rg,c.P);rg.fill();if(c.k==17){rg.lineWidth=1.6;rg.stroke();s.nc++}}}
   rg.globalAlpha=1;
   /* spores bud from the threads while there is still leaf to feed on */
   var left=1-s.nc/cells.length;
   if(s.ck>2&&s.nodes.length){s.sa+=dt*(.15+1.9*left);if(s.sa>1&&s.spores.length<18){s.sa=0;nd=s.nodes[Math.floor(R()*s.nodes.length)];if(nd.x<w*.9&&nd.x>4&&nd.y>24&&nd.y<h-10)
     s.spores.push({x:nd.x,y:nd.y,nx:nd.x,ny:nd.y,g:0,a:R()*6.28,b:R()*6.28,c:R()*6.28,wa:(R()<.5?1:-1)*(.35+R()*.8),wb:(R()<.5?1:-1)*(.3+R()*.7),wc:(R()-.5)*.8,sc:(.8+R()*.45)*u,vx:0,vy:0,tv:(24+R()*20)*u,ty:(R()-.55)*12*u,ph:R()*6.28,L:[.85+R()*.3,.85+R()*.3,.85+R()*.3,1+R()*.35]})}
    s.ga+=dt*Math.min(3.6,.4+s.act*.05);if(s.ga>1){s.ga=0;for(k=0;k<8;k++){c=cells[Math.floor(R()*cells.length)];if(c.k<17&&c.t0<s.ck&&c.cx>0&&c.cx<w&&c.cy>0&&c.cy<h){s.gas.push({x:c.cx,y:c.cy,vx:(8+R()*10)*u,vy:(R()-.6)*8*u,life:0});break}}}}};
  s.reset=function(){var k,c,j,a0;rg.save();rg.setTransform(1,0,0,1,0,0);rg.globalCompositeOperation='copy';rg.globalAlpha=1;rg.drawImage(leaf,0,0);rg.restore();
   hg.save();hg.setTransform(1,0,0,1,0,0);hg.clearRect(0,0,W,H);hg.restore();
   for(k=0;k<cells.length;k++){c=cells[k];c.t0=1e9;c.k=0;c.acc=0}
   s.nodes=[];s.tips=[];s.ck=0;s.nc=0;s.fin=0;
   var S=[[.24,.27],[.72,.5],[.33,.8]];for(k=0;k<3;k++){x=w*(S[k][0]+(R()-.5)*.14);y=h*(S[k][1]+(R()-.5)*.12);a0=R()*6.28;for(j=0;j<4;j++)s.tips.push(tip(x,y,a0+j*1.571+(R()-.5)*.6,1.6*u))}
   for(k=0;k<70;k++)s.step(.05)};
  s.reset();
  return s},
 frame:function(s,g,w,h,t,dt,ui){
  if(!(dt>0))dt=0;
  var sm=HSM.smooth,u=s.u,R=s.R,i,k,p,m;
  if(dt>0)s.step(dt);
  if(s.ck>=41.5){s.reset();s.fo=1}
  if(s.fo>0)s.fo-=dt/1.6;
  var ov=s.ck>39?sm(39,41.5,s.ck):s.fo>0?sm(0,1,s.fo):0,frac=s.nc/s.cells.length;
  g.drawImage(s.bg,0,0,w,h);
  /* soft light wandering over the water */
  g.globalCompositeOperation='lighter';g.globalAlpha=.09;var lb=HSM.blob('rgba(150,230,215,1)',64);
  g.drawImage(lb,w*(.3+.25*Math.sin(t*.21))-w*.5,h*(.35+.2*Math.cos(t*.17))-w*.5,w,w);g.drawImage(lb,w*(.7+.2*Math.cos(t*.14))-w*.4,h*(.7+.15*Math.sin(t*.19))-w*.4,w*.8,w*.8);
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  g.drawImage(s.rem,0,0,w,h);
  g.globalAlpha=1-.3*sm(.8,1,frac);g.drawImage(s.hy,0,0,w,h);g.globalAlpha=1;
  /* growing tips glow */
  if(s.ck<25&&ov<1){g.globalCompositeOperation='lighter';g.globalAlpha=.55*(1-ov);var tb=HSM.blob('rgba(255,250,215,1)',16);for(i=0;i<s.tips.length;i++){p=s.tips[i];g.drawImage(tb,p.x-5*u,p.y-5*u,10*u,10*u)}g.globalAlpha=1;g.globalCompositeOperation='source-over'}
  if(ov>0){g.globalAlpha=ov;g.drawImage(s.leaf,0,0,w,h);g.globalAlpha=1}
  /* CO2: last year's carbon going back */
  var gb=HSM.blob('rgba(255,196,120,1)',16,.3),r=4.7;
  for(i=s.gas.length-1;i>=0;i--){m=s.gas[i];m.life+=dt/5;m.vx+=(R()-.5)*30*dt;m.vy+=(R()-.5)*30*dt;m.x+=m.vx*dt;m.y+=m.vy*dt;if(m.life>1||m.x>w+10){s.gas.splice(i,1);continue}
   var al=sm(0,.12,m.life)*(1-sm(.6,1,m.life));g.globalAlpha=.9*al;g.drawImage(gb,m.x-r,m.y-r,r*2,r*2);g.lineWidth=1;g.strokeStyle='rgba(110,60,10,.6)';g.beginPath();g.arc(m.x,m.y,r*.62,0,6.283);g.stroke()}
  g.globalAlpha=1;
  /* spores: four arms from one point, tumbling as the current takes them */
  var B=[[.577,.577,.577],[.577,-.577,-.577],[-.577,.577,-.577],[-.577,-.577,.577]],glow=HSM.blob('rgba(190,240,230,1)',32);
  g.lineCap='round';
  for(i=s.spores.length-1;i>=0;i--){p=s.spores[i];
   if(p.g<1){p.g+=dt/2.2;var gg=sm(0,1,p.g);p.x=p.nx+7*u*gg;p.y=p.ny-9*u*gg;p.a+=p.wa*dt*.15}
   else{p.vx+=(p.tv-p.vx)*dt*1.2;p.vy+=(p.ty+Math.sin(t*1.3+p.ph)*7*u-p.vy)*dt*1.2;p.x+=p.vx*dt;p.y+=p.vy*dt;p.a+=p.wa*dt;p.b+=p.wb*dt;p.c+=p.wc*dt;gg=1;if(p.x>w+40*u){s.spores.splice(i,1);continue}}
   var ca=Math.cos(p.a),sa=Math.sin(p.a),cb=Math.cos(p.b),sb=Math.sin(p.b),cc=Math.cos(p.c),sc=Math.sin(p.c),Lm=22*p.sc*(.25+.75*gg);
   if(p.g<1){g.lineWidth=1.3*u;g.strokeStyle='rgba(232,240,222,.85)';g.beginPath();g.moveTo(p.nx,p.ny);g.lineTo(p.x,p.y);g.stroke()}
   g.globalCompositeOperation='lighter';g.globalAlpha=.2*gg;g.drawImage(glow,p.x-Lm*1.5,p.y-Lm*1.5,Lm*3,Lm*3);g.globalCompositeOperation='source-over';g.globalAlpha=1;
   for(var pass=0;pass<2;pass++)for(k=0;k<4;k++){var v=B[k],y1=v[1]*ca-v[2]*sa,z1=v[1]*sa+v[2]*ca,x2=v[0]*cb+z1*sb,z2=-v[0]*sb+z1*cb,ax=(x2*cc-y1*sc)*Lm*p.L[k],ay=(x2*sc+y1*cc)*Lm*p.L[k],
     qx=p.x+ax*.5-ay*.14,qy=p.y+ay*.5+ax*.14,mx=p.x*.25+qx*.5+(p.x+ax)*.25,my=p.y*.25+qy*.5+(p.y+ay)*.25,wd=p.sc*(.5+.5*gg),br=.62+.3*z2;
    if(pass==0){g.strokeStyle='rgba(6,22,26,.28)';g.lineWidth=5.2*wd;g.beginPath();g.moveTo(p.x,p.y);g.quadraticCurveTo(qx,qy,p.x+ax,p.y+ay);g.stroke()}
    else{g.strokeStyle='rgba(214,244,236,'+br+')';g.lineWidth=1.9*wd;g.beginPath();g.moveTo(p.x,p.y);g.quadraticCurveTo(qx,qy,p.x+ax,p.y+ay);g.stroke();
     g.lineWidth=3.3*wd;g.beginPath();g.moveTo(p.x,p.y);g.lineTo(mx,my);g.stroke();
     g.strokeStyle='rgba(255,255,255,'+(br*.7)+')';g.lineWidth=.9*wd;g.beginPath();g.moveTo(p.x-.5,p.y-.5);g.lineTo(mx-.5,my-.5);g.stroke()}}}
  /* out-of-focus debris passing above, close to the lens */
  g.globalCompositeOperation='lighter';var db=HSM.blob('rgba(170,220,210,1)',32,.6);
  for(i=0;i<s.dust.length;i++){m=s.dust[i];m.x+=m.v*dt;if(m.x>w+m.r){m.x=-m.r;m.y=R()*h}g.globalAlpha=m.a;g.drawImage(db,m.x-m.r,m.y-m.r,m.r*2,m.r*2)}
  g.globalAlpha=1;g.globalCompositeOperation='source-over';
  HSM.vignette(g,w,h,.5);
  g.globalAlpha=1-ov;HSM.label(g,12,18,'fungal threads');
  s.lab+=((s.spores.length>1?1:0)-s.lab)*Math.min(1,dt*2);g.globalAlpha=s.lab;HSM.label(g,w-12,18,'four-armed spores','right');
  g.globalAlpha=sm(.5,.8,frac)*(1-ov);if(g.globalAlpha>.01)HSM.label(g,12,h-14,'only the veins remain');
  g.globalAlpha=1;
 }});
