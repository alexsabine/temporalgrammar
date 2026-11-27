// Screenshot a simulation in the harness.  node simshot.js <sim> <outPrefix> [slider values, comma separated] [on: 0|1] [seconds to wait, default 6]
const {chromium}=require('playwright');const path=require('path');
(async()=>{const [sim,out,as='',on='0',secs='6']=process.argv.slice(2);
 const b=await chromium.launch();const p=await b.newPage({viewport:{width:420,height:620},deviceScaleFactor:2});
 const errs=[];p.on('pageerror',e=>errs.push(e.message));p.on('console',m=>{if(m.type()==='error')errs.push(m.text())});
 for(const a of (as?as.split(','):[''])){
  await p.goto('file://'+path.join(__dirname,'harness.html')+'?sim='+sim+'&w=390&h=520'+(a!==''?'&a='+a:'')+'&on='+on);
  await p.waitForTimeout(+secs*1000);
  await p.locator('#c').screenshot({path:out+(a!==''?'-a'+a:'')+'.png'});
  console.log(sim,'a='+a,'ms/frame',await p.evaluate(()=>window.__ms&&window.__ms.toFixed(2)));}
 console.log('errors:',errs);await b.close()})();
