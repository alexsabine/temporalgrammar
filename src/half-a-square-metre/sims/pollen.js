/* The tip of a saxifrage flower's stigma under the microscope. Pollen grains land on the glistening
   finger-like cells, take up water and swell, and each grows a tube down through the style towards
   an ovule. When a tube arrives there is fertilisation, and the ovule begins to become a seed.
   Tap to drop a grain. */
HSM.sim('pollen',{
 setup:function(w,h,R){
  var i,top=h*.30,bot=h*.84;
  /* papillae: rounded fingers across the stigma surface, drawn once */
  var P=document.createElement('canvas');P.width=w*2;P.height=h*2;var g=P.getContext('2d');g.scale(2,2);
  var bg=g.createLinearGradient(0,0,0,h);bg.addColorStop(0,'#2c2a1a');bg.addColorStop(.25,'#3d3a22');bg.addColorStop(.32,'#d9d6b2');bg.addColorStop(1,'#c4c39a');
  g.fillStyle='#2b2919';g.fillRect(0,0,w,h);
  /* style tissue: long translucent cells */
  for(var x=-10;x<w+20;x+=17+R()*6){for(var y=top-10-R()*40;y<bot+30;){var cl=34+R()*30,cw2=15+R()*5,xx=x+(R()-.5)*3;
   var c=g.createLinearGradient(xx,0,xx+cw2,0);c.addColorStop(0,'rgba(205,200,160,.96)');c.addColorStop(.45,'rgba(240,236,204,.96)');c.addColorStop(1,'rgba(196,190,150,.96)');
   g.fillStyle=c;g.beginPath();if(g.roundRect)g.roundRect(xx+1,y+1,cw2,cl,7);else g.rect(xx+1,y+1,cw2,cl);g.fill();g.strokeStyle='rgba(130,125,85,.5)';g.lineWidth=1;g.stroke();
   g.fillStyle='rgba(150,140,95,.35)';g.beginPath();g.ellipse(xx+1+cw2/2,y+1+cl*(.3+R()*.4),cw2*.22,cw2*.18,0,0,6.283);g.fill();y+=cl+1}}
  /* lower chamber: dark cavity behind the ovules */
  var cv=g.createLinearGradient(0,bot-10,0,h);cv.addColorStop(0,'rgba(60,50,30,0)');cv.addColorStop(.3,'rgba(48,40,26,.92)');cv.addColorStop(1,'rgba(30,25,16,1)');g.fillStyle=cv;g.fillRect(0,bot-10,w,h-bot+10);
  /* papillae along the top, two rows for depth */
  var paps=[];
  for(var row=0;row<2;row++){var pw=w/(9+row*2);for(x=-pw*.3+row*pw*.5;x<w+pw;x+=pw*(.85+R()*.2)){
   var ph=h*(.16+R()*.08)*(row?0.8:1),bx=x,by=top+6-row*8;paps.push({x:bx,y:by-ph*.92,r:pw*.42,row:row});
   var pg=g.createLinearGradient(bx-pw*.45,0,bx+pw*.45,0);
   pg.addColorStop(0,row?'#a9a47a':'#bdb98c');pg.addColorStop(.35,row?'#e6e2bd':'#f6f3d4');pg.addColorStop(1,row?'#8f8a62':'#a39f74');
   g.fillStyle=pg;g.beginPath();g.moveTo(bx-pw*.42,by+10);g.lineTo(bx-pw*.42,by-ph+pw*.42);g.arc(bx,by-ph+pw*.42,pw*.42,Math.PI,0);g.lineTo(bx+pw*.42,by+10);g.closePath();g.fill();
   g.strokeStyle='rgba(100,96,60,.55)';g.lineWidth=1;g.stroke();
   /* wet gleam */
   g.fillStyle='rgba(255,255,255,.55)';g.beginPath();g.ellipse(bx-pw*.13,by-ph+pw*.4,pw*.08,pw*.2,-.2,0,6.283);g.fill()}}
  /* pollen sprite: a sculpted sphere with spines and a gold oily coat */
  var r0=Math.max(9,w*.03),S=Math.ceil(r0*1.6),sp=document.createElement('canvas');sp.width=sp.height=S*2*2;var q=sp.getContext('2d');q.scale(2,2);
  for(i=0;i<22;i++){var a=i/22*6.283;q.strokeStyle='#b88a14';q.lineWidth=r0*.18;q.lineCap='round';q.beginPath();q.moveTo(S+Math.cos(a)*r0*.9,S+Math.sin(a)*r0*.9);q.lineTo(S+Math.cos(a)*r0*1.32,S+Math.sin(a)*r0*1.32);q.stroke()}
  var pg2=q.createRadialGradient(S-r0*.35,S-r0*.35,r0*.1,S,S,r0);pg2.addColorStop(0,'#fff6b0');pg2.addColorStop(.5,'#f0c53a');pg2.addColorStop(1,'#a67710');
  q.fillStyle=pg2;q.beginPath();q.arc(S,S,r0,0,6.283);q.fill();
  for(i=0;i<16;i++){var a3=R()*6.283,rr=Math.sqrt(R())*r0*.75;q.fillStyle='rgba(140,95,10,.35)';q.beginPath();q.arc(S+Math.cos(a3)*rr,S+Math.sin(a3)*rr,r0*.12,0,6.283);q.fill()}
  /* ovules */
  var ov=[],no=Math.max(4,Math.round(w/70));for(i=0;i<no;i++)ov.push({x:w*(i+.5)/no+(R()-.5)*w*.04,y:bot+(h-bot)*.42+(R()-.5)*8,r:w/no*.3,seed:0,flash:0,taken:0});
  return{P:P,paps:paps,sp:sp,S:S,r0:r0,ov:ov,gr:[],top:top,bot:bot,R:R,next:1.2,wasDown:false,clock:0}},
 frame:function(s,g,w,h,t,dt,ui){
  dt=Math.max(0,Math.min(.05,dt));s.clock+=dt;var R=s.R,i,j,G;
  g.drawImage(s.P,0,0,w,h);
  /* new grains: now and then, and wherever the visitor taps */
  var inner=s.paps.filter(function(q){return q.x>w*.08&&q.x<w*.92});
  function drop(x){var p=inner[Math.floor(R()*inner.length)];if(x!=null){var best=1e9;inner.forEach(function(q){var d=Math.abs(q.x-x);if(d<best){best=d;p=q}})}
   var free=s.ov.filter(function(o){return !o.taken});if(!free.length||s.gr.length>9)return;
   free.sort(function(a,b){return Math.abs(a.x-p.x)-Math.abs(b.x-p.x)});var o=free[Math.min(free.length-1,Math.floor(R()*2))];o.taken=1;
   var pts=[[p.x,p.y+s.r0*.6]],n=14;for(var k=1;k<=n;k++){var f=k/n;pts.push([p.x+(o.x-p.x)*f*f+(R()-.5)*w*.05*Math.sin(f*Math.PI),p.y+(o.y-o.r-p.y)*f])}
   s.gr.push({x:p.x,y:-20,ty:p.y,vy:0,state:0,sw:0,len:0,pts:pts,o:o,age:0,fade:1,spin:R()*6.28})}
  s.next-=dt;if(s.next<=0){drop();s.next=2.2+R()*2.5}
  if(ui.down&&!s.wasDown&&ui.py<s.bot)drop(ui.px);s.wasDown=ui.down;
  /* tubes first, under the grains */
  for(i=0;i<s.gr.length;i++){G=s.gr[i];if(G.len<=0)continue;
   var tot=0,seg=[];for(j=1;j<G.pts.length;j++){var L=Math.hypot(G.pts[j][0]-G.pts[j-1][0],G.pts[j][1]-G.pts[j-1][1]);seg.push(L);tot+=L}
   G.tot=tot;var rem=G.len,x=G.pts[0][0],y=G.pts[0][1];
   g.lineCap='round';g.lineJoin='round';
   for(var pass=0;pass<2;pass++){g.beginPath();g.moveTo(G.pts[0][0],G.pts[0][1]);rem=G.len;
    for(j=1;j<G.pts.length&&rem>0;j++){var f=Math.min(1,rem/seg[j-1]);x=G.pts[j-1][0]+(G.pts[j][0]-G.pts[j-1][0])*f;y=G.pts[j-1][1]+(G.pts[j][1]-G.pts[j-1][1])*f;g.lineTo(x,y);rem-=seg[j-1]}
    g.globalAlpha=G.fade;g.lineWidth=pass?2:6;g.strokeStyle=pass?'rgba(255,250,210,.9)':'rgba(190,150,40,.75)';g.stroke()}
   if(G.state===2){var pulse=.6+.4*Math.sin(s.clock*6+i);g.globalCompositeOperation='lighter';g.globalAlpha=G.fade*pulse;g.drawImage(HSM.blob('rgba(255,230,120,1)',16,.2),x-14,y-14,28,28);g.globalCompositeOperation='source-over'}
   G.tx=x;G.ty2=y;g.globalAlpha=1}
  /* ovules */
  for(i=0;i<s.ov.length;i++){var o=s.ov[i];if(o.seed>0&&o.seed<1)o.seed=Math.min(1,o.seed+dt*.12);o.flash=Math.max(0,o.flash-dt*.8);
   var rr=o.r*(1+.35*o.seed),og=g.createRadialGradient(o.x-rr*.3,o.y-rr*.3,rr*.1,o.x,o.y,rr);
   var c1=o.seed?'rgb('+(240-o.seed*90|0)+','+(232-o.seed*120|0)+','+(200-o.seed*150|0)+')':'#f2ecd0';
   og.addColorStop(0,'#fffbe8');og.addColorStop(.5,c1);og.addColorStop(1,o.seed?'rgb('+(150-o.seed*60|0)+','+(120-o.seed*60|0)+',60)':'#b9b08a');
   g.fillStyle=og;g.beginPath();g.ellipse(o.x,o.y,rr*.82,rr,0,0,6.283);g.fill();g.strokeStyle='rgba(80,70,40,.6)';g.lineWidth=1;g.stroke();
   g.fillStyle='rgba(120,100,60,.45)';g.beginPath();g.arc(o.x,o.y+rr*.15,rr*.32,0,6.283);g.fill();
   if(o.flash>0){g.globalCompositeOperation='lighter';g.globalAlpha=o.flash;g.drawImage(HSM.blob('rgba(255,245,190,1)',32,.25),o.x-rr*2.4,o.y-rr*2.4,rr*4.8,rr*4.8);g.globalCompositeOperation='source-over';g.globalAlpha=1}}
  /* grains */
  for(i=s.gr.length-1;i>=0;i--){G=s.gr[i];G.age+=dt;
   if(G.state===0){G.vy+=140*dt;G.y+=G.vy*dt;G.spin+=dt*2;if(G.y>=G.ty){G.y=G.ty;G.state=1;G.sw=0}}
   else if(G.state===1){G.sw+=dt*.45;if(G.sw>=1){G.state=2}}
   else if(G.state===2){G.len+=dt*(w*.12);if(G.tot&&G.len>=G.tot){G.len=G.tot;G.state=3;G.o.flash=1;G.o.seed=.01}}
   else if(G.state===3){G.fade-=dt*.08;if(G.fade<=0){s.gr.splice(i,1);continue}}
   var sc=1+.22*HSM.smooth(0,1,G.sw||0);g.save();g.globalAlpha=Math.max(0,G.fade);g.translate(G.x,G.y);g.rotate(G.spin);g.scale(sc,sc);g.drawImage(s.sp,-s.S,-s.S,s.S*2,s.S*2);g.restore()}
  g.globalAlpha=1;
  /* when every ovule has become a seed, wait, then let the flower start again */
  if(s.ov.every(function(o){return o.seed>=1})){s.reset=(s.reset||0)+dt;if(s.reset>4){s.ov.forEach(function(o){o.seed=0;o.taken=0});s.reset=0}}
  HSM.vignette(g,w,h,.4);
  HSM.label(g,12,18,'pollen grains land and swell; tap to add one');
  var Gl=null;for(i=0;i<s.gr.length;i++)if(s.gr[i].state===2&&s.gr[i].len>w*.25){Gl=s.gr[i];break}
  if(Gl)HSM.label(g,Math.min(w-12,Gl.tx+12),Gl.ty2-8,'pollen tube',Gl.tx>w*.6?'right':'left');
  HSM.label(g,12,h-16,'ovules, becoming seeds');
 }});
