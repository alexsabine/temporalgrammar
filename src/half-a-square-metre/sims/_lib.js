/* Shared by every simulation at the bottom of a dive. Canvas 2D only. */
window.HSM=window.HSM||{sims:{}};
HSM.sim=function(key,def){HSM.sims[key]=def};
HSM.rnd=function(seed){var s=seed||1;return function(){s=(s*16807)%2147483647;return s/2147483647}};
/* a soft round sprite, cached: HSM.blob('rgba(120,255,200,1)',32) -> canvas 64x64, drawn with drawImage */
HSM._b={};
HSM.blob=function(col,r,hard){var k=col+'|'+r+'|'+(hard||0);if(HSM._b[k])return HSM._b[k];
  var c=document.createElement('canvas');c.width=c.height=r*2;var g=c.getContext('2d'),q=g.createRadialGradient(r,r,0,r,r,r);
  q.addColorStop(0,col);q.addColorStop(hard||0.25,col);q.addColorStop(1,col.replace(/[\d.]+\)$/,'0)'));g.fillStyle=q;g.fillRect(0,0,r*2,r*2);return HSM._b[k]=c};
/* a small leader label, the same in every simulation. Use sparingly: three at most. */
HSM.label=function(g,x,y,text,align){g.save();g.font='400 12.5px "Source Serif 4",Georgia,serif';g.textAlign=align||'left';g.textBaseline='middle';
  g.lineWidth=3;g.strokeStyle='rgba(6,10,8,.75)';g.strokeText(text,x,y);g.fillStyle='rgba(255,255,255,.92)';g.fillText(text,x,y);g.restore()};
HSM.vignette=function(g,w,h,a){var q=g.createRadialGradient(w/2,h/2,Math.min(w,h)*.35,w/2,h/2,Math.max(w,h)*.75);q.addColorStop(0,'rgba(0,0,0,0)');q.addColorStop(1,'rgba(0,0,0,'+(a==null?.55:a)+')');g.fillStyle=q;g.fillRect(0,0,w,h)};
HSM.clamp=function(x,a,b){return x<a?a:x>b?b:x};
HSM.smooth=function(a,b,x){x=HSM.clamp((x-a)/(b-a),0,1);return x*x*(3-2*x)};
