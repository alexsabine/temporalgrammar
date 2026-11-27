/* The spikemoss's giant cousins. A 40 second time-lapse, side view, cut through the ground like a diorama.
   Scale trees (Lepidodendron) stand in a coal swamp 300 million years ago: pole trunks with diamond leaf
   cushions, crowns that fork in two again and again, grass-like leaves at the tips. Trees fall, the wood
   does not rot in the sour water and piles up as peat; mud and sand bury it; the weight presses it into a
   thin black seam of coal. Today the seam is dug and burnt and its carbon goes back to the air.
   Everything is a pure function of loop time, so any moment is a correct still. */
HSM.sim('coal',{
 setup:function(w,h,R){
  var D=2,C=HSM.clamp,sm=HSM.smooth,PI=Math.PI,i,k,q,g;
  function cv(a,b){var c=document.createElement('canvas');c.width=Math.max(2,Math.round(a*D));c.height=Math.max(2,Math.round(b*D));c.g=c.getContext('2d');c.g.scale(D,D);c.cw=a;c.ch=b;return c}
  var yW=.68*h,yB=.91*h,ST=.085*h,PH=.15*h,SH=.53*h,CT=.03*h,LOOP=40;

  /* ---------- a scale tree ---------- */
  function tree(g,X,Y,H,tw,dp,dn){
   var th=H*(.5+R()*.06),tp=tw*.6,B=[],LV=[[],[],[]],CN=[],ll=H*.052;
   function leaf(x,y,a,l){var c=R()<.15?2:(R()<(x<X?.62:.3)?1:0);LV[c].push(x,y,x+Math.cos(a)*l*.55,y+Math.sin(a)*l*.55-l*.1,x+Math.cos(a)*l,y+Math.sin(a)*l+l*.42)}
   function br(x,y,a,len,wd,d){
    var bd=(R()-.5)*.3,x2=x+Math.cos(a)*len,y2=y+Math.sin(a)*len;B.push([x,y,x+Math.cos(a+bd)*len*.5,y+Math.sin(a+bd)*len*.5,x2,y2,wd]);
    if(d<=2){var n=Math.round(len/3.2*dn);for(var j=0;j<n;j++){var f=(j+R())/n;leaf(x+(x2-x)*f,y+(y2-y)*f,a+(R()<.5?-1:1)*(.8+R()*.7),ll*(.45+R()*.4))}}
    if(d>0){var sp=.3+R()*.2,a1=a-sp,a2=a+sp;a1+=(-PI/2-a1)*.12;a2+=(-PI/2-a2)*.12;br(x2,y2,a1,len*(.7+R()*.1),wd*.7,d-1);br(x2,y2,a2,len*(.7+R()*.1),wd*.7,d-1)}
    else{n=Math.round(15*dn);for(j=0;j<n;j++)leaf(x2,y2,a+(R()-.5)*3.3,ll*(.6+R()*.6));if(R()<.3)CN.push([x2,y2])}}
   br(X,Y-th+2,-PI/2-.36-R()*.08,H*.15,tp*.78,dp-1);br(X,Y-th+2,-PI/2+.36+R()*.08,H*.15,tp*.78,dp-1);
   g.lineCap='round';
   for(var j=0;j<B.length;j++){var b=B[j];g.strokeStyle='#1b1f15';g.lineWidth=b[6];g.beginPath();g.moveTo(b[0],b[1]);g.quadraticCurveTo(b[2],b[3],b[4],b[5]);g.stroke();
    if(b[6]>1.6){g.strokeStyle='rgba(126,134,82,.5)';g.lineWidth=b[6]*.3;g.beginPath();g.moveTo(b[0]-b[6]*.25,b[1]);g.quadraticCurveTo(b[2]-b[6]*.25,b[3],b[4]-b[6]*.25,b[5]);g.stroke()}}
   var cols=['rgba(34,62,26,.9)','rgba(104,146,50,.88)','rgba(206,226,124,.85)'];
   for(var c=0;c<3;c++){var L=LV[c];g.strokeStyle=cols[c];g.lineWidth=Math.max(.6,H*.0034);g.beginPath();for(j=0;j<L.length;j+=6){g.moveTo(L[j],L[j+1]);g.quadraticCurveTo(L[j+2],L[j+3],L[j+4],L[j+5])}g.stroke()}
   for(j=0;j<CN.length;j++){g.fillStyle='#5a4a26';g.beginPath();g.ellipse(CN[j][0],CN[j][1]+H*.016,H*.006,H*.018,0,0,6.283);g.fill()}
   /* trunk: a lit cylinder, then the diamond leaf cushions */
   g.save();g.beginPath();g.moveTo(X-tw*1.05,Y);g.quadraticCurveTo(X-tw*.52,Y-H*.035,X-tw*.5,Y-H*.11);g.lineTo(X-tp*.5,Y-th);g.lineTo(X+tp*.5,Y-th);g.lineTo(X+tw*.5,Y-H*.11);g.quadraticCurveTo(X+tw*.52,Y-H*.035,X+tw*1.05,Y);g.closePath();
   q=g.createLinearGradient(X-tw*.6,0,X+tw*.6,0);q.addColorStop(0,'#aab07a');q.addColorStop(.14,'#6c7248');q.addColorStop(.5,'#2f3323');q.addColorStop(1,'#11130e');g.fillStyle=q;g.fill();g.clip();
   var sp2=tw*.5;g.lineWidth=Math.max(.5,tw*.05);
   for(var p=0;p<2;p++){g.strokeStyle=p?'rgba(210,214,150,.2)':'rgba(6,8,4,.6)';g.beginPath();for(var d=-tw*4;d<H;d+=sp2){var o=p*tw*.07;g.moveTo(X-tw*1.1,Y-d+o);g.lineTo(X+tw*1.1,Y-d-tw*3.3+o);g.moveTo(X-tw*1.1,Y-d-tw*3.3+o);g.lineTo(X+tw*1.1,Y-d+o)}g.stroke()}
   q=g.createLinearGradient(0,Y-H*.3,0,Y);q.addColorStop(0,'rgba(10,22,10,0)');q.addColorStop(1,'rgba(8,20,10,.62)');g.fillStyle=q;g.fillRect(X-tw*1.2,Y-H*.3,tw*2.4,H*.3);
   g.restore()}
  /* a young one: an unbranched pole, bristling with leaves */
  function pole(g,X,Y,H,tw){var L=[[],[]];for(var j=0;j<H*1.1;j++){var y=Y-H*.08-R()*H*.92,f=(Y-y)/H,l=H*(.07+.09*f),a=(R()<.5?PI:0)+(R()-.5)*.5-(R()<.5?-1:1)*0,sx=Math.cos(a)<0?-1:1;L[R()<.4?1:0].push(X,y,X+sx*l*.6,y-l*.5,X+sx*l,y-l*.1+R()*l*.5)}
   for(j=0;j<26;j++){a=-PI/2+(R()-.5)*2.2;l=H*.16;L[j&1].push(X,Y-H,X+Math.cos(a)*l*.6,Y-H+Math.sin(a)*l*.7,X+Math.cos(a)*l,Y-H+Math.sin(a)*l+l*.2)}
   g.lineCap='round';for(var c=0;c<2;c++){g.strokeStyle=c?'rgba(150,186,78,.85)':'rgba(40,74,30,.9)';g.lineWidth=.9;g.beginPath();for(j=0;j<L[c].length;j+=6){g.moveTo(L[c][j],L[c][j+1]);g.quadraticCurveTo(L[c][j+2],L[c][j+3],L[c][j+4],L[c][j+5])}g.stroke()}
   q=g.createLinearGradient(X-tw*.5,0,X+tw*.5,0);q.addColorStop(0,'#8f9662');q.addColorStop(.4,'#3a3f2a');q.addColorStop(1,'#14170f');g.fillStyle=q;g.beginPath();g.moveTo(X-tw*.6,Y);g.lineTo(X-tw*.3,Y-H);g.lineTo(X+tw*.3,Y-H);g.lineTo(X+tw*.6,Y);g.fill()}
  /* giant horsetail (Calamites) for the understorey */
  function cal(g,x,y,H,col){g.strokeStyle=col;g.lineCap='round';g.lineWidth=Math.max(1,H*.035);g.beginPath();g.moveTo(x,y);g.lineTo(x+H*.05,y-H);g.stroke();g.lineWidth=Math.max(.6,H*.012);g.beginPath();
   for(var j=1;j<10;j++){var yy=y-H*j/10,xx=x+H*.05*j/10,l=H*.22*(1.1-j/11);for(var m=-3;m<=3;m++){if(!m)continue;g.moveTo(xx,yy);g.quadraticCurveTo(xx+m*l*.2,yy-l*.05,xx+m*l*.33,yy-l*(.75-.12*Math.abs(m)))}}g.stroke()}
  function haze(c,col,a1,a2){var g=c.g;g.globalCompositeOperation='source-atop';var q=g.createLinearGradient(0,0,0,c.ch);q.addColorStop(0,col+a1+')');q.addColorStop(1,col+a2+')');g.fillStyle=q;g.fillRect(0,0,c.cw,c.ch);g.globalCompositeOperation='source-over'}

  /* ---------- forest sky, layers, light ---------- */
  var skyF=cv(w,h);g=skyF.g;q=g.createLinearGradient(0,0,0,yW);q.addColorStop(0,'#3f6450');q.addColorStop(.45,'#8fb086');q.addColorStop(.85,'#d4dcaa');q.addColorStop(1,'#e2e4b4');g.fillStyle=q;g.fillRect(0,0,w,h);
  q=g.createRadialGradient(w*.36,h*.13,0,w*.36,h*.13,h*.62);q.addColorStop(0,'rgba(255,250,212,.95)');q.addColorStop(.25,'rgba(250,244,190,.5)');q.addColorStop(1,'rgba(240,240,180,0)');g.fillStyle=q;g.fillRect(0,0,w,h);
  var far=cv(w,yW),mid=cv(w,yW),yF=yW-ST,yM=yW-ST*.6;
  for(i=0;i<15;i++)tree(far.g,(i+.2+R()*.6)/15*w,yF+R()*3,h*(.25+R()*.12),5+R()*2.5,4,.6);
  for(i=0;i<14;i++)cal(far.g,R()*w,yF+2,h*(.05+R()*.05),'#2c4426');
  haze(far,'rgba(206,220,170,',.62,.82);
  var mx=[.02,.17,.36,.52,.7,.9];
  for(i=0;i<6;i++)tree(mid.g,(mx[i]+R()*.05)*w,yM+R()*3,h*(.37+R()*.1),9+R()*3,5,.8);
  for(i=0;i<12;i++)cal(mid.g,R()*w,yM+3,h*(.06+R()*.07),'#1f3a1f');
  haze(mid,'rgba(150,180,128,',.3,.6);
  var sh=[],sx=w*.36,sy=-h*.3;for(i=0;i<2;i++){c=document.createElement('canvas');c.width=Math.round(w/6);c.height=Math.round(yW/6);c.g=c.getContext('2d');c.g.scale(1/6,1/6);sh.push(c)}
  for(i=0;i<8;i++){g=sh[i&1].g;var a0=-.5+i*.15+R()*.05,dw=.02+R()*.03,Lr=yW-sy;q=g.createLinearGradient(sx,sy,sx+Math.tan(a0)*Lr,yW);q.addColorStop(0,'rgba(255,250,205,.0)');q.addColorStop(.3,'rgba(255,250,205,.3)');q.addColorStop(.7,'rgba(255,250,205,.16)');q.addColorStop(1,'rgba(255,250,205,.03)');
   g.fillStyle=q;g.beginPath();g.moveTo(sx-3,sy);g.lineTo(sx+3,sy);g.lineTo(sx+Math.tan(a0+dw)*Lr,yW);g.lineTo(sx+Math.tan(a0-dw)*Lr,yW);g.fill()}
  /* water seen from above the bank: sky and far trees upside down in it */
  var refl=cv(w,ST);g=refl.g;q=g.createLinearGradient(0,0,0,ST);q.addColorStop(0,'#bcc894');q.addColorStop(.3,'#6f8f6a');q.addColorStop(1,'#1a3024');g.fillStyle=q;g.fillRect(0,0,w,ST);
  g.save();g.globalAlpha=.8;g.scale(1,-.36);g.translate(0,-yF);g.drawImage(far,0,0,w,yW);g.restore();
  g.save();g.globalAlpha=.85;g.translate(0,ST*.4);g.scale(1,-.36);g.translate(0,-yM);g.drawImage(mid,0,0,w,yW);g.restore();
  q=g.createLinearGradient(0,0,0,ST);q.addColorStop(0,'rgba(14,30,20,.0)');q.addColorStop(.5,'rgba(12,28,18,.5)');q.addColorStop(1,'rgba(8,22,14,.92)');g.fillStyle=q;g.fillRect(0,0,w,ST);
  /* near trees, each its own sprite so it can fall */
  var near=[],nd=[[.07,.56,19,-1,0],[.27,.6,21,4.2,1],[.47,.5,16,11.2,-1],[.64,.57,20,-1,0],[.83,.53,18,7.6,-1],[.99,.6,21,-1,0]];
  for(i=0;i<nd.length;i++){var H=nd[i][1]*h,c=cv(H*.9,H*1.03+6);tree(c.g,c.cw/2,c.ch-3,H,nd[i][2],5,1);near.push({x:nd[i][0]*w,y:yW-ST*(.3-.04*(i%3)),H:H,c:c,f:nd[i][3],dir:nd[i][4]})}
  var young=[];for(i=0;i<3;i++){H=h*(.24+R()*.08);c=cv(H*.5,H*1.2+4);pole(c.g,c.cw/2,c.ch-2,H,6);young.push({x:[.18,.4,.75][i]*w,y:yW-ST*.22,c:c,t0:1+i*2.2})}
  var fore=cv(w,ST*2.4);for(i=0;i<9;i++)cal(fore.g,(i<5?R()*.2:.8+R()*.2)*w,fore.ch,h*(.05+R()*.07),'#12241a');

  /* ---------- the ground, as cut faces ---------- */
  var peat=cv(w,PH);g=peat.g;q=g.createLinearGradient(0,0,0,PH);q.addColorStop(0,'#4a3018');q.addColorStop(.5,'#33200f');q.addColorStop(1,'#1c1109');g.fillStyle=q;g.fillRect(0,0,w,PH);
  g.lineCap='round';for(i=0;i<900;i++){var x=R()*w,y=R()*PH,l=6+R()*26,f=y/PH;g.strokeStyle=['rgba(150,104,52,','rgba(20,12,6,','rgba(196,150,84,','rgba(86,56,26,'][i&3]+(.18+R()*.3*(1.2-f))+')';g.lineWidth=.6+R()*1.6;g.beginPath();g.moveTo(x,y);g.quadraticCurveTo(x+l*.5,y+(R()-.5)*5,x+l,y+(R()-.5)*3);g.stroke()}
  for(i=0;i<40;i++){x=R()*w;y=R()*PH;g.fillStyle='rgba(18,10,5,.6)';g.beginPath();g.ellipse(x,y,5+R()*14,1.5+R()*2.5,(R()-.5)*.2,0,6.283);g.fill()}
  var log=cv(w*.6,15);g=log.g;g.save();g.beginPath();g.moveTo(4,3);g.lineTo(log.cw-8,5);g.quadraticCurveTo(log.cw,7.5,log.cw-8,10);g.lineTo(4,13);g.quadraticCurveTo(0,8,4,3);g.closePath();q=g.createLinearGradient(0,2,0,14);q.addColorStop(0,'#7a6a45');q.addColorStop(.35,'#3b3222');q.addColorStop(1,'#0e0c08');g.fillStyle=q;g.fill();g.clip();
  g.strokeStyle='rgba(5,5,3,.6)';g.lineWidth=.8;g.beginPath();for(x=-20;x<log.cw+20;x+=5){g.moveTo(x,2);g.lineTo(x+16,14);g.moveTo(x+16,2);g.lineTo(x,14)}g.stroke();g.restore();
  var strata=cv(w,SH);g=strata.g;var bands=[['#5d5348',.07],['#3d3e42',.08],['#a48a5e',.13],['#55585a',.06],['#7b6f5c',.09],['#b59a6c',.12],['#44464b',.07],['#8a5e46',.08],['#a9946f',.12],['#6a6257',.08],['#978a6a',.22]],yy=SH,tp=[];
  for(i=0;i<bands.length;i++){yy-=bands[i][1]*SH;tp.push([yy,R()*6,R()*6,1.5+R()*3])}
  function top(i,x){var b=tp[i];return b[0]+Math.sin(x*.02+b[1])*b[3]+Math.sin(x*.05+b[2])*b[3]*.4+Math.sin(x*.009+1)*5}
  for(i=bands.length-1;i>=0;i--){g.fillStyle=bands[i][0];g.beginPath();g.moveTo(0,SH);for(x=0;x<=w+8;x+=6)g.lineTo(x,top(i,x));g.lineTo(w+8,SH);g.fill()}
  for(i=0;i<70;i++){g.globalAlpha=.05+R()*.05;g.drawImage(HSM.blob(i&1?'rgba(255,236,200,1)':'rgba(0,0,10,1)',32,.1),R()*w-60,R()*SH-15,60+R()*140,14+R()*30)}g.globalAlpha=1;
  for(i=0;i<bands.length;i++){var bh=bands[i][1]*SH,sand=i==2||i==5||i==8||i==10;g.lineWidth=.6;
   for(k=0;k<(sand?9:13);k++){var off=R()*bh,lx=R()*w*.8-20,ln=30+R()*w*.6;g.strokeStyle=(R()<.5?'rgba(255,240,210,':'rgba(12,10,12,')+(.08+R()*.14)+')';g.beginPath();for(x=lx;x<lx+ln;x+=6){y=top(i,x)+off+(sand?(x-lx)*-.07:0);if(y>top(i,x)+bh)break;x==lx?g.moveTo(x,y):g.lineTo(x,y)}g.stroke()}
   if(sand)for(k=0;k<26;k++){g.fillStyle='rgba(236,220,180,'+(.12+R()*.2)+')';x=R()*w;g.beginPath();g.ellipse(x,top(i,x)+R()*bh,1+R()*2.2,.7+R(),0,0,6.283);g.fill()}
   g.strokeStyle='rgba(0,0,0,.3)';g.lineWidth=1.2;g.beginPath();for(x=0;x<=w+8;x+=6)g.lineTo(x,top(i,x)-.6);g.stroke();g.strokeStyle='rgba(255,240,214,.2)';g.lineWidth=.8;g.beginPath();for(x=0;x<=w+8;x+=6)g.lineTo(x,top(i,x)+.7);g.stroke()}
  for(i=0;i<4200;i++){g.fillStyle=(i&1?'rgba(255,240,210,':'rgba(10,8,6,')+(.05+R()*.14)+')';g.fillRect(R()*w,R()*SH,.6+R()*1.3,.5+R())}
  g.strokeStyle='rgba(10,8,8,.22)';g.lineWidth=.7;g.beginPath();for(i=0;i<9;i++){x=R()*w;y=R()*SH;g.moveTo(x,y);g.lineTo(x+(R()-.5)*6,y+10+R()*20);g.lineTo(x+(R()-.5)*8,y+26+R()*26)}g.stroke();
  q=g.createLinearGradient(0,0,0,SH);q.addColorStop(0,'rgba(0,0,0,0)');q.addColorStop(1,'rgba(8,8,14,.42)');g.fillStyle=q;g.fillRect(0,0,w,SH);
  var clay=cv(w,h-yB+1);g=clay.g;q=g.createLinearGradient(0,0,0,clay.ch);q.addColorStop(0,'#57534a');q.addColorStop(1,'#25231f');g.fillStyle=q;g.fillRect(0,0,w,clay.ch);
  for(i=0;i<500;i++){g.fillStyle=(i&1?'rgba(220,210,190,':'rgba(0,0,0,')+(.05+R()*.1)+')';g.fillRect(R()*w,R()*clay.ch,1+R(),.7+R())}
  g.lineCap='round';for(i=0;i<3;i++){x=(.05+i*.36+R()*.1)*w;l=w*(.16+R()*.08);var pts=[[x,-2,x+l*.45,6,x+l*.6,clay.ch*.42,3.4],[x+l*.6,clay.ch*.42,x+l*.8,clay.ch*.5,x+l*1.05,clay.ch*.5,2.2],[x+l*.6,clay.ch*.42,x+l*.7,clay.ch*.7,x+l*.95,clay.ch*.86,2]];
   for(k=0;k<3;k++){var p=pts[k];g.strokeStyle='rgba(120,112,96,.28)';g.lineWidth=.6;g.beginPath();for(var m=0;m<22;m++){f=R();var ux=(1-f)*(1-f)*p[0]+2*f*(1-f)*p[2]+f*f*p[4],uy=(1-f)*(1-f)*p[1]+2*f*(1-f)*p[3]+f*f*p[5],an=R()*6.283,rl=4+R()*10;g.moveTo(ux,uy);g.lineTo(ux+Math.cos(an)*rl,uy+Math.sin(an)*rl*.7)}g.stroke();
    g.strokeStyle='#191611';g.lineWidth=p[6];g.beginPath();g.moveTo(p[0],p[1]);g.quadraticCurveTo(p[2],p[3],p[4],p[5]);g.stroke();g.strokeStyle='rgba(150,140,118,.35)';g.lineWidth=.7;g.beginPath();g.moveTo(p[0],p[1]-p[6]*.3);g.quadraticCurveTo(p[2],p[3]-p[6]*.3,p[4],p[5]-p[6]*.3);g.stroke()}}
  /* a drowned trunk foot with its forking roots (Stigmaria), seen through the water */
  var stump=cv(120,26);g=stump.g;g.strokeStyle='#0a120c';g.lineCap='round';g.fillStyle='#0a120c';g.beginPath();g.moveTo(52,0);g.lineTo(68,0);g.quadraticCurveTo(69,10,80,19);g.lineTo(40,19);g.quadraticCurveTo(51,10,52,0);g.fill();
  for(k=-1;k<=1;k+=2){g.lineWidth=5;g.beginPath();g.moveTo(60+k*12,15);g.quadraticCurveTo(60+k*26,19,60+k*38,20);g.stroke();g.lineWidth=2.6;g.beginPath();g.moveTo(60+k*38,20);g.quadraticCurveTo(60+k*48,19,60+k*57,16);g.moveTo(60+k*38,20);g.quadraticCurveTo(60+k*47,23,60+k*55,24);g.stroke()}
  var coal=cv(w,20);g=coal.g;q=g.createLinearGradient(0,0,0,20);q.addColorStop(0,'#1a1b20');q.addColorStop(.2,'#07070a');q.addColorStop(1,'#020203');g.fillStyle=q;g.fillRect(0,0,w,20);
  for(i=0;i<9;i++){y=2+R()*16;g.strokeStyle='rgba(150,165,200,'+(.1+R()*.3)+')';g.lineWidth=.5+R()*.8;g.beginPath();x=R()*w*.4;g.moveTo(x,y);g.lineTo(x+w*(.2+R()*.6),y+(R()-.5)*1.5);g.stroke()}
  g.strokeStyle='rgba(120,130,160,.16)';g.lineWidth=.6;g.beginPath();for(i=0;i<80;i++){x=R()*w;y=R()*14;g.moveTo(x,y);g.lineTo(x+(R()-.5)*2,y+3+R()*6)}g.stroke();

  /* ---------- today ---------- */
  var skyT=cv(w,h*.7);g=skyT.g;q=g.createLinearGradient(0,0,0,h*.42);q.addColorStop(0,'#5f86ab');q.addColorStop(.55,'#b5cbd4');q.addColorStop(1,'#f3e3c0');g.fillStyle=q;g.fillRect(0,0,w,h*.7);
  q=g.createRadialGradient(w*.2,h*.33,0,w*.2,h*.33,h*.4);q.addColorStop(0,'rgba(255,244,214,.8)');q.addColorStop(1,'rgba(255,240,210,0)');g.fillStyle=q;g.fillRect(0,0,w,h*.7);
  for(i=0;i<7;i++){g.globalAlpha=.3+R()*.3;g.drawImage(HSM.blob('rgba(255,255,250,1)',48,.1),R()*w-60,h*(.04+R()*.2),120+R()*120,16+R()*16)}g.globalAlpha=1;
  var hills=cv(w,h*.12);g=hills.g;
  for(k=0;k<2;k++){g.fillStyle=k?'#5d8266':'#9db7b4';g.beginPath();g.moveTo(0,hills.ch);p1=R()*6;for(x=0;x<=w;x+=4)g.lineTo(x,hills.ch*(k?.72:.3)+Math.sin(x*.012+p1)*hills.ch*.16+Math.sin(x*.05+p1*2)*hills.ch*.05-(k&&R()<.25?R()*4:0));g.lineTo(w,hills.ch);g.fill()}
  q=g.createLinearGradient(0,0,0,hills.ch);q.addColorStop(0,'rgba(240,230,200,0)');q.addColorStop(1,'rgba(240,230,200,.4)');g.globalCompositeOperation='source-atop';g.fillStyle=q;g.fillRect(0,0,w,hills.ch);g.globalCompositeOperation='source-over';
  var grass=cv(w,16);g=grass.g;g.fillStyle='#3a2a1a';g.fillRect(0,9,w,7);g.fillStyle='#2a1d12';g.fillRect(0,13,w,3);g.lineCap='round';
  for(i=0;i<700;i++){x=R()*w;l=3+R()*6;g.strokeStyle=['#3f6b2d','#5d8a36','#8fb04e','#2c5226'][i&3];g.lineWidth=.9;g.beginPath();g.moveTo(x,10.5);g.lineTo(x+(R()-.5)*4,10.5-l);g.stroke()}
  var works=cv(64,70);g=works.g;g.fillStyle='#283036';g.fillRect(8,52,40,18);g.fillRect(2,60,12,10);g.beginPath();g.moveTo(50,70);g.lineTo(52,6);g.lineTo(57,6);g.lineTo(59,70);g.fill();
  g.fillStyle='rgba(255,236,200,.55)';g.fillRect(52,6,1.2,64);g.fillRect(8,52,1.2,18);g.fillStyle='#ffd27a';for(i=0;i<5;i++)g.fillRect(13+i*7,57,3,2.4);g.fillStyle='#b94a3a';g.fillRect(52,14,5.4,3);
  var frameS=cv(30,34);g=frameS.g;g.strokeStyle='#232a2f';g.lineWidth=2;g.lineCap='round';g.beginPath();g.moveTo(11,34);g.lineTo(15,7);g.lineTo(19,34);g.moveTo(15,7);g.lineTo(29,34);g.moveTo(12.5,22);g.lineTo(22,22);g.stroke();g.lineWidth=1.4;g.beginPath();g.arc(15,6,4.2,0,6.283);g.stroke();g.strokeStyle='rgba(255,236,200,.5)';g.lineWidth=.7;g.beginPath();g.moveTo(10.4,34);g.lineTo(14.4,7);g.stroke();

  var mist=HSM.blob('rgba(228,238,198,1)',64,.08),smoke=[HSM.blob('rgba(52,46,44,1)',48,.12),HSM.blob('rgba(126,112,100,1)',48,.12)],glow=HSM.blob('rgba(255,180,80,1)',32,.1),white=HSM.blob('rgba(255,255,240,1)',32,.1);
  var M=[];for(i=0;i<10;i++)M.push({x:R()*w,y:i<4?yF-h*(.02+R()*.2):i<7?yM-h*(.01+R()*.14):yW-ST*.5-R()*h*.05,sx:w*(.5+R()*.5),sy:h*(.07+R()*.08),v:(4+R()*7)*(i&1?1:-1),a:i<4?.5:i<7?.36:.22,ph:R()*6});
  var fl=[];for(i=0;i<34;i++)fl.push({x:R(),ph:R(),v:.03+R()*.05,l:1.5+R()*4,c:i%3});
  var shaftX=.3*w,stX=.66*w;

  function thA(u){return (.025+.125*sm(0,15,u))*h}
  function paint(g,u,t,raw){
   var comp=sm(16,26,u),pk=1+(CT/PH-1)*comp,tA=thA(u),th=tA*pk,yP=yB-th,sedT=.518*h*sm(15,25,u),ySed=yP-sedT;
   var surf=u<15?yW:ySed-.08*h*(1-sm(17,25,u)),oy=surf-yW,fF=1-sm(14,18.5,u),tL=sm(23,26,u),ca=sm(.3,.85,comp),j,a,x,y,n,e;
   /* ----- above ground ----- */
   if(fF<1){g.drawImage(skyT,0,0,w,skyT.ch);
    var aw=sm(14,17,u)*(1-sm(22.5,25.5,u));if(aw>0){g.globalAlpha=aw;var q=g.createLinearGradient(0,surf-ST,0,surf);q.addColorStop(0,'#dfe0c8');q.addColorStop(.3,'#a9bcc0');q.addColorStop(1,'#5f7a80');g.fillStyle=q;g.fillRect(0,surf-ST,w,ST);
     for(j=0;j<9;j++){x=(j*.137+t*.012*(1+j%3))%1*w;g.globalAlpha=aw*(.3+.3*Math.sin(t*1.3+j*2));g.drawImage(white,x-25,surf-ST+4+j*ST*.1,50,1.6)}g.globalAlpha=1}
    a=.4*sm(15,19,u)+.6*tL;g.globalAlpha=a;g.drawImage(hills,0,surf-(aw>0?ST*aw:0)-hills.ch+1,w,hills.ch);g.globalAlpha=1}
   if(fF>0){g.save();g.translate(0,oy);g.globalAlpha=fF;g.drawImage(skyF,0,0,w,h);g.drawImage(far,0,0,w,yW);
    g.globalCompositeOperation='lighter';g.globalAlpha=fF*(.62+.3*Math.sin(t*.37));g.drawImage(sh[0],0,0,w,yW);g.globalAlpha=fF*(.62+.3*Math.sin(t*.29+2));g.drawImage(sh[1],0,0,w,yW);g.globalCompositeOperation='source-over';
    function mst(a,b){for(var j=a;j<b;j++){var m=M[j],x=((m.x+m.v*t)%(w+m.sx)+w+m.sx)%(w+m.sx)-m.sx;g.globalAlpha=fF*m.a*(.75+.25*Math.sin(t*.2+m.ph));g.drawImage(mist,x,m.y-m.sy*.5,m.sx,m.sy)}}
    mst(0,4);g.globalAlpha=fF;g.drawImage(mid,0,0,w,yW);
    g.globalCompositeOperation='lighter';g.globalAlpha=fF*.22;g.drawImage(sh[0],0,0,w,yW);g.globalCompositeOperation='source-over';
    /* a griffinfly crosses */
    var du=(u-2.5)/8;if(du>0&&du<1){x=-20+du*(w+40);y=h*.34+Math.sin(du*9)*h*.03;g.globalAlpha=fF;g.strokeStyle='#1b2418';g.lineWidth=2;g.beginPath();g.moveTo(x-14,y+1.5);g.lineTo(x+7,y);g.stroke();g.fillStyle='rgba(250,255,225,.55)';a=Math.sin(t*38)*.35;
     for(j=0;j<4;j++){g.save();g.translate(x+2-(j>>1)*3.5,y);g.rotate((j&1?1:-1)*(1.25+a*(j>1?-1:1)));g.beginPath();g.ellipse(10,0,10,2.4,0,0,6.283);g.fill();g.restore()}}
    mst(4,7);g.globalAlpha=fF;g.drawImage(refl,0,yF,w,ST);
    g.save();g.beginPath();g.rect(0,yF,w,ST);g.clip();
    for(j=0;j<near.length;j++){n=near[j];if(n.f>0&&u>n.f)continue;g.save();g.translate(n.x,n.y);g.scale(1,-.36);g.globalAlpha=fF*.5;g.drawImage(n.c,-n.c.cw/2,-n.c.ch+3,n.c.cw,n.c.ch);g.restore()}
    g.restore();
    for(j=0;j<12;j++){x=((j*.618+t*.008*(1+j%3))%1)*w;g.globalAlpha=fF*(.35+.35*Math.sin(t*1.1+j*1.9));g.drawImage(white,x-22,yF+3+(j*7.3)%(ST-6),44,1.5)}
    for(j=0;j<young.length;j++){n=young[j];var gr=.2+.8*sm(n.t0,15,u),gw=.25+.75*gr;g.globalAlpha=fF;g.drawImage(n.c,n.x-n.c.cw*.5*gw,n.y-n.c.ch*gr+2,n.c.cw*gw,n.c.ch*gr)}
    for(j=0;j<near.length;j++){n=near[j];e=n.f>0?C((u-n.f)/2.4,0,1):0;var sk=n.f>0?sm(2.3,3.5,u-n.f):0;if(sk>=1)continue;
     g.save();g.translate(n.x,n.y+sk*12);if(e>0)g.rotate(n.dir*1.5708*Math.pow(e,2.6));g.globalAlpha=fF*(1-sk);g.drawImage(n.c,-n.c.cw/2,-n.c.ch+3,n.c.cw,n.c.ch);g.restore();
     if(e>=1){a=(u-n.f-2.4)/1.6;if(a<1)for(k=0;k<5;k++){g.globalAlpha=fF*.6*(1-a);var px=n.x+n.dir*n.H*(.15+k*.2);g.drawImage(mist,px-30-a*30,n.y-14-a*26,60+a*60,26+a*30)}}}
    g.globalAlpha=fF;g.drawImage(fore,0,yW-fore.ch+2,w,fore.ch);g.restore();g.globalAlpha=1}
   /* ----- below ground ----- */
   var mud=sm(14.5,17.5,u),wd=ySed-surf;
   if(wd>.5){q=g.createLinearGradient(0,surf,0,surf+Math.max(wd,30));q.addColorStop(0,'#3d5a40');q.addColorStop(1,'#101c14');g.fillStyle=q;g.fillRect(0,surf,w,wd+1);
    if(mud>0){g.globalAlpha=mud*.8;q=g.createLinearGradient(0,surf,0,surf+Math.max(wd,30));q.addColorStop(0,'#8d8a68');q.addColorStop(1,'#5d4c30');g.fillStyle=q;g.fillRect(0,surf,w,wd+1);g.globalAlpha=1}
    for(j=0;j<fl.length;j++){n=fl[j];y=surf+((n.ph+t*n.v*(1+mud*2))%1)*wd;x=(n.x*w+Math.sin(t*.5+j)*6+w)%w;g.fillStyle=(n.c==0?'rgba(214,190,120,':n.c==1?'rgba(120,150,70,':'rgba(60,40,20,')+(.35+.4*mud)+')';g.fillRect(x,y,n.l,1.3)}
    if(fF>0){g.globalCompositeOperation='lighter';for(j=0;j<6;j++){g.save();g.translate(w*(.1+j*.17)+Math.sin(t*.3+j)*8,surf);g.rotate(.32);g.globalAlpha=fF*(.07+.05*Math.sin(t*.7+j*2.1));g.drawImage(white,-9,-wd*.3,18,wd*1.7);g.restore()}g.globalCompositeOperation='source-over';for(j=0;j<near.length;j++){n=near[j];g.globalAlpha=fF*.42;g.fillStyle='#0a120c';g.fillRect(n.x-6.2,surf,12.4,Math.max(0,wd-13));g.drawImage(stump,n.x-46,surf+wd-14,92,20)}g.globalAlpha=1}
    g.fillStyle='rgba(236,244,206,.7)';g.fillRect(0,surf-.6,w,1.4);g.fillStyle='rgba(236,244,206,.12)';g.fillRect(0,surf+.8,w,4)}
   if(sedT>.5){g.drawImage(strata,0,(SH-sedT)*D,w*D,sedT*D,0,ySed,w,sedT+.5);a=.5*(1-sm(21,27,u));if(a>0){g.fillStyle='rgba(84,58,26,'+a+')';g.fillRect(0,ySed,w,sedT)}
    if(wd>.5){g.fillStyle='rgba(196,170,110,.35)';g.fillRect(0,ySed-1.5,w,3)}}
   g.drawImage(peat,0,(PH-tA)*D,w*D,tA*D,0,yP,w,th+.5);
   g.fillStyle='rgba(206,160,90,'+(.35*(1-ca))+')';g.fillRect(0,yP-.5,w,1.2);
   for(j=0;j<near.length;j++){n=near[j];if(n.f<0||u<n.f+2.4)continue;e=sm(0,1,C((u-n.f-2.4)/2.6,0,1));var d0=thA(n.f+5)-5,yl=u<15?yW+8+(yB-d0-yW-8)*e:yB-d0*pk;
    g.save();g.translate(n.x+n.dir*log.cw*.5,yl);g.rotate(n.dir*(1-e)*.12+(j-2)*.012);g.scale(n.dir,Math.max(.3,pk));g.globalAlpha=sm(0,.3,e)*(1-ca);g.drawImage(log,-log.cw/2,-7.5,log.cw,15);g.restore()}
   g.globalAlpha=1;
   if(ca>0){g.globalAlpha=ca;g.drawImage(coal,0,yP,w,th+.5);g.globalAlpha=1}
   g.drawImage(clay,0,yB,w,clay.ch);g.fillStyle='rgba(0,0,0,.35)';g.fillRect(0,yB,w,1.5);
   var dg=sm(29,35.5,u);
   if(ca>.5){x=((t*.09)%1.6-.3)*w;g.globalCompositeOperation='lighter';g.globalAlpha=.34*sm(.5,1,ca);g.drawImage(white,x-70,yP-1,140,th*.7);g.globalAlpha=.16*ca;g.drawImage(white,w-x-90,yP+th*.3,180,th*.6);g.globalCompositeOperation='source-over';g.globalAlpha=1}
   /* ----- today: the seam is dug and burnt ----- */
   if(tL>0){g.globalAlpha=tL;g.drawImage(grass,0,surf-10,w,16);g.globalAlpha=1}
   var mn=sm(25.5,27,u);
   if(mn>0){var sp=sm(26.5,29,u),sb=surf+4+(yP+th*.5-surf-4)*sp;g.globalAlpha=mn;g.drawImage(works,stX,surf-69,64,70);g.drawImage(frameS,shaftX-15,surf-33,30,34);g.globalAlpha=1;
    if(sp>0){g.fillStyle='#0a0908';g.fillRect(shaftX-2.6,surf+2,5.2,sb-surf-2);g.fillStyle='rgba(255,230,180,.22)';g.fillRect(shaftX-2.6,surf+2,.8,sb-surf-2)}
    if(dg>0){var xl=shaftX-dg*w*.2,xr=shaftX+dg*w*.56,fk=.7+.3*Math.sin(t*9);g.fillStyle='#030202';g.fillRect(xl,yP+1.5,xr-xl,th-2.5);
     g.globalCompositeOperation='lighter';g.globalAlpha=.75*fk*(1-sm(35.5,37,u));g.drawImage(glow,xl-14,yP+th*.5-14,28,28);g.drawImage(glow,xr-14,yP+th*.5-14,28,28);g.globalAlpha=.5;g.drawImage(glow,shaftX-8,yP+th*.5-8,16,16);
     g.globalCompositeOperation='source-over';g.globalAlpha=1;g.fillStyle='#0b0a09';
     for(j=0;j<7;j++){e=(t*.16+j/7)%1;var up=yP+th*.5-surf+4,run=stX+10-shaftX,tot=up+run,dd=e*tot;if(u-e/.16<29.6)continue;if(dd<up)g.fillRect(shaftX-1.3,yP+th*.5-dd,2.6,2.6);else g.fillRect(shaftX+dd-up,surf-5.5,3,2.6)}}
    /* the plume: each puff is drawn only if it left the chimney while the fire was lit */
    var cx=stX+54.5,cy=surf-64;
    for(j=0;j<30;j++){a=(t*.17+j/30)%1;var born=u-a/.17;if(born<29.2||born>36.4)continue;x=cx+a*a*w*.3+Math.sin(j*7.1+a*5)*a*16;y=cy-a*(cy+50)*(1-.25*a);var r=5+a*66;
     g.globalAlpha=sm(0,.06,a)*Math.pow(1-a,1.1)*.7;g.drawImage(smoke[j&1],x-r,y-r,r*2,r*2)}
    if(u>29.2&&u<36.4){g.globalCompositeOperation='lighter';g.globalAlpha=.6;g.drawImage(glow,cx-7,cy-6,14,14);g.globalCompositeOperation='source-over'}
    g.globalAlpha=1}
   if(raw)return;
   /* ----- back to the forest, softly ----- */
   var fo=Math.max(sm(36.8,39.5,u),1-sm(0,1.6,u));if(fo>0){g.globalAlpha=fo;g.drawImage(s.snap,0,0,w,h);g.globalAlpha=1}
   HSM.vignette(g,w,h,.42);
   HSM.label(g,12,18,fo>.5||u<15?'300 million years ago':u<26?Math.max(1,Math.round(300-299*Math.pow((u-15)/11,.7)))+' million years ago':'today');
   a=(1-sm(19,21.5,u))*(1-fo);if(a>0){g.globalAlpha=a;HSM.label(g,w-12,Math.min(yB-9,yP+th*.5),'peat','right');g.globalAlpha=1}
   a=sm(22.5,25,u)*(1-fo);if(a>0){g.globalAlpha=a;HSM.label(g,w-12,yP+th*.5+.5,'coal','right');g.globalAlpha=1}
   a=sm(30.5,32,u)*(1-fo);if(a>0){g.globalAlpha=a;HSM.label(g,stX+42,surf-h*.165,'its carbon, back in the air','right');g.globalAlpha=1}
  }
  var s={paint:paint,snap:null};var sn=cv(w,h);paint(sn.g,.02,0,1);s.snap=sn;return s},
 frame:function(s,g,w,h,t,dt,ui){s.paint(g,(t+2)%40,t,0)}
});
