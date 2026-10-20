// Real headless-Chrome verification, using its built-in DevTools protocol.
// No npm install, network fetch or changes to the user's Chrome profile.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {spawn} from 'node:child_process';
import assert from 'node:assert/strict';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const out=path.join(root,'outputs/vcg_t_almeida_v2_experiment');
const profile=fs.mkdtempSync(path.join(os.tmpdir(),'vcg09-chrome-'));
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',[
 '--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check',
 '--allow-file-access-from-files','--remote-debugging-port=0','--window-size=1440,1100',
 '--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let stderr='',socket;
try{
 const endpoint=await new Promise((resolve,reject)=>{
  const timer=setTimeout(()=>reject(Error('Chrome startup timeout: '+stderr)),20000);
  chrome.stderr.on('data',chunk=>{stderr+=chunk.toString();const m=stderr.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m){clearTimeout(timer);resolve(m[1])}});
  chrome.on('exit',code=>reject(Error('Chrome exit '+code+': '+stderr)));
 });
 socket=new WebSocket(endpoint);
 await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});
 let sequence=0;const pending=new Map(),exceptions=[];
 socket.onmessage=event=>{const message=JSON.parse(event.data);if(message.id){const wait=pending.get(message.id);if(wait){clearTimeout(wait.timer);pending.delete(message.id);message.error?wait.reject(Error(JSON.stringify(message.error))):wait.resolve(message.result)}}else if(message.method==='Runtime.exceptionThrown'){exceptions.push(message.params.exceptionDetails.text+': '+JSON.stringify(message.params.exceptionDetails.exception||{}))}};
 function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const id=++sequence,timer=setTimeout(()=>{pending.delete(id);reject(Error('CDP timeout '+method))},60000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params,...(sessionId?{sessionId}:{})}))})}
 const target=await call('Target.createTarget',{url:'about:blank'});
 const attach=await call('Target.attachToTarget',{targetId:target.targetId,flatten:true}),session=attach.sessionId;
 await call('Runtime.enable',{},session);await call('Page.enable',{},session);
 async function evaluate(expression){const r=await call('Runtime.evaluate',{expression,awaitPromise:true,returnByValue:true},session);if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value}
 const url=pathToFileURL(path.join(out,'gallery/index.html')).href;
 await call('Page.navigate',{url},session);
 for(let i=0;i<100;i++){
  if(await evaluate('Boolean(window.__almeidaQA?.current)'))break;
  await new Promise(resolve=>setTimeout(resolve,100));
 }
 assert.equal(await evaluate('window.__almeidaQA.records.length'),448);
 assert.equal(await evaluate('window.__almeidaQA.current.row.record_id'),'patient104/s0306lre');
 assert.equal(await evaluate('document.querySelectorAll("canvas").length'),3);
 const screenshotDir=path.join(out,'gallery/qa');fs.mkdirSync(screenshotDir,{recursive:true});
 async function screenshot(name){const result=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false},session);fs.writeFileSync(path.join(screenshotDir,name),Buffer.from(result.data,'base64'))}
 await screenshot('initial.png');
 let tabsChecked=0;
 for(const record of ['patient104/s0306lre','patient051/s0179lre','patient034/s0109lre','patient033/s0157lre']){
  await evaluate(`window.__almeidaQA.selectRecord(${JSON.stringify(record)})`);
  for(const tab of ['signals','wavelets','single','tls','loops','history']){
   await evaluate(`window.__almeidaQA.setTab(${JSON.stringify(tab)}); new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))`);
   assert.equal(await evaluate('getComputedStyle(document.getElementById("error")).display'), 'none');
   tabsChecked++;
  }
 }
 await evaluate('window.__almeidaQA.selectRecord("patient104/s0306lre")');
 await evaluate('window.__almeidaQA.setTab("tls");new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
 for(const boundary of ['t_on','t_off']){
  await evaluate(`(()=>{const s=document.getElementById('${boundary}Select');s.value=s.options[s.options.length-1].value;s.dispatchEvent(new Event('change'))})()`);
 }
 await evaluate('document.getElementById("fitt_on").closest(".grid").scrollIntoView({block:"start"})');
 await screenshot('tls.png');
 await evaluate('window.__almeidaQA.setTab("loops");new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
 await evaluate('document.getElementById("content").scrollIntoView({block:"start"})');
 await screenshot('loops.png');
 const payloadStats={records:0,medians:0,loops:0};
 for(let start=0;start<448;start+=40){
  const stats=await evaluate(`(async()=>{let n=0,medians=0,loops=0;for(const r of window.__almeidaQA.records.slice(${start},${start+40})){const p=await window.__almeidaQA.loadPayload(r.record_id);if(p.row.record_id!==r.record_id)throw Error('ID mismatch');n++;if(p.xyz){medians++;if(p.time_ms.length!==p.xyz.length)throw Error('Time/XYZ mismatch')}else if(p.arrays.t_loop_v2)throw Error('Fake loop');if(p.arrays.t_loop_v2){loops++;if(p.arrays.t_loop_v2.length!==p.row.t_off_v2_sample-p.row.t_on_v2_sample+1)throw Error('Inclusive slicing mismatch')}}return {n,medians,loops}})()`);
  payloadStats.records+=stats.n;payloadStats.medians+=stats.medians;payloadStats.loops+=stats.loops;
  console.log('Browser payloads',payloadStats.records+'/448');
 }
 await evaluate(`(()=>{document.getElementById('label').value='HC';document.getElementById('label').dispatchEvent(new Event('change'))})()`);
 assert.equal(await evaluate('document.querySelectorAll(".record").length'),80);
 await evaluate(`(()=>{document.getElementById('label').value='';document.getElementById('label').dispatchEvent(new Event('change'));document.getElementById('overview').click()})()`);
 assert.equal(await evaluate('document.querySelectorAll(".contact img").length'),8);
 await evaluate(`document.getElementById('pageNext').click()`);
 assert.equal(await evaluate('document.querySelectorAll(".contact img").length'),8);
 for(const [id,value] of [['split','valid'],['status','unresolved'],['qc','auto_candidate']]){
  await evaluate(`(()=>{for(const key of ['search','label','split','status','qc'])document.getElementById(key).value='';const s=document.getElementById('${id}');s.value='${value}';s.dispatchEvent(new Event('change'))})()`);
  const expected=await evaluate(`window.__almeidaQA.records.filter(r=>r.${id==='qc'?'v1_qc_status':id}==='${value}').length`);
  assert.equal(await evaluate('document.querySelectorAll(".record").length'),expected);
 }
 await evaluate(`(()=>{for(const key of ['label','split','status','qc'])document.getElementById(key).value='';const s=document.getElementById('search');s.value='record-that-does-not-exist';s.dispatchEvent(new Event('input'))})()`);
 assert.equal(await evaluate('document.querySelectorAll(".record").length'),0);
 assert.equal(exceptions.length,0,exceptions.join('\n'));
 const report={browser:'Real headless Google Chrome',url,records:payloadStats.records,
  medians:payloadStats.medians,t_loops:payloadStats.loops,representative_tab_views:tabsChecked,
  iteration_selectors_checked:true,label_filter_checked:true,split_status_qc_filters_checked:true,
  empty_search_checked:true,contact_pages_checked:2,
  javascript_exceptions:exceptions,screenshots:['initial.png','tls.png','loops.png'],
  visual_inspection_is_separate:true,not_all_record_images_manually_reviewed:true};
 fs.writeFileSync(path.join(out,'gallery/browser_validation.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report,null,2));
} finally {
 if(socket)socket.close();
 chrome.kill('SIGTERM');
}
