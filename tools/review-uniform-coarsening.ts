/** Build a self-contained property/field review from stage-scaling captures.
 * node --import tsx tools/review-uniform-coarsening.ts baseline.json experiment.json ... --out=artifacts/uniform-coarsening/review.html
 * GPU captures must run serially before opening a live simulation browser.
 */
import assert from "node:assert/strict";
import {readFileSync,writeFileSync,mkdirSync} from "node:fs";
import {dirname,resolve} from "node:path";
const paths=process.argv.slice(2).filter(a=>!a.startsWith("--"));
assert.ok(paths.length>=2,"Provide a baseline and at least one experimental capture");
const out=resolve(process.argv.find(a=>a.startsWith("--out="))?.slice(6)??"artifacts/uniform-coarsening/review.html");
const captures=paths.map(path=>JSON.parse(readFileSync(path,"utf8")));
const base=captures[0];
for(const capture of captures){
 assert.equal(capture.sceneId,base.sceneId);assert.equal(capture.dt_s,base.dt_s);
 assert.deepEqual(capture.lattice,base.lattice);
 assert.ok(capture.failure||capture.sourceFingerprintAfter,"Completed capture needs an ending source fingerprint");
 if(capture.sourceFingerprintAfter)assert.equal(capture.sourceFingerprint,capture.sourceFingerprintAfter,"Capture changed production sources while running");
 assert.equal(capture.sourceFingerprint,base.sourceFingerprint,"Compare identical production source versions");
 assert.ok(capture.qualitySnapshots?.length,"Capture requires --quality-every");
}
const mean=(v:number[])=>v.reduce((a,b)=>a+b,0)/v.length;
const commonLast=Math.min(...captures.map(c=>c.rows.at(-1).frame));
const data=captures.map(c=>({
 name:c.surfaceTolerance>0?`${c.coarseExtension?"Coarse velocity extension":"Smooth surface at 4h"} · ${c.surfaceTolerance}h tolerance`:"Fine surface · reference",
 scene:c.sceneId,lattice:c.lattice,dt:c.dt_s,failure:c.failure??null,fullPressureEnvelope:!!c.fullPressureEnvelope,pressureReserve:c.pressureReserve??null,
 gpu:mean(c.rows.filter((r:any)=>r.frame>8&&r.frame<=commonLast).map((r:any)=>r.trace.total_ms)),
 meanFine:mean(c.rows.filter((r:any)=>r.frame>8&&r.frame<=commonLast).map((r:any)=>r.work.uniformMixedFineTiles)),
 frames:c.qualitySnapshots.filter((s:any)=>s.frame<=commonLast).map((s:any)=>({...s,projection:s.projection.map((v:number)=>+v.toFixed(5)),phiSlice:s.phiSlice.map((v:number)=>+v.toFixed(6))})),
}));
const json=JSON.stringify(data).replaceAll("<","\\u003c");
mkdirSync(dirname(out),{recursive:true});
writeFileSync(out,`<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Uniform · coarse work exploration</title><style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#111920;color:#e3edf2}body{margin:0 auto;padding:28px;max-width:1450px}h1{font-size:27px;margin:0 0 10px}p{color:#b0c4d0;max-width:1000px;line-height:1.5}button,select{background:#263744;color:#e3edf2;border:1px solid #486273;border-radius:5px;padding:8px}button{cursor:pointer}.controls{display:flex;gap:18px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#111920;padding:16px 0;z-index:2}input{width:300px}.cards{display:grid;gap:18px;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr))}.card{border:1px solid #354b59;border-radius:8px;overflow:hidden;background:#192630}h2{font-size:16px;padding:0 16px;min-height:40px}.metric{font-size:13px;font-variant-numeric:tabular-nums;padding:0 16px 16px;line-height:1.8}canvas{width:100%;display:block;image-rendering:pixelated}strong{color:#a4e5e6}.legend{font-size:13px}table{border-collapse:collapse;width:100%;margin-top:25px;font-size:13px}td,th{text-align:left;border-bottom:1px solid #354b59;padding:9px}small{color:#b0c4d0}</style>
<h1>How much work can stay at 4h?</h1>
<p>Same long-dam scene, timestep and numerical tolerances. Each card names its surface/velocity policy and pressure schedule. These are field views, not the production water rendering. Use the time slider to inspect front travel, wall impact and the returning sheet.</p>
<div class="controls"><button id="play">Play</button><input id="time" aria-label="Simulation time" type="range" min="0" value="0"><strong id="stamp"></strong><select id="view" aria-label="Field view"><option value="volume">Liquid amount · depth average</option><option value="phi">Surface · middle cross-section</option><option value="difference">Liquid difference from reference</option></select></div>
<p class="legend" id="legend"></p><div class="cards" id="cards"></div><table><thead><tr><th>Policy</th><th>GPU ms/step</th><th>Mean h tiles</th><th>Final mass drift</th><th>Largest sampled pre-impact front difference</th></tr></thead><tbody id="summary"></tbody></table>
<p>Review prompts: Is the travelling front recognizably the same? Does the impact climb and return with comparable weight? Are sheets disappearing, becoming blocky, or spreading too diffusely? Tiny ripples may differ. A fast run with visibly wrong transport is not an improvement.</p>
<small>GPU timings cover common steps 9–${commonLast}, excluding readbacks and rendering. Property differences are measurements, not automatic acceptance. Failed captures stop at their last saved sample. Scheduling costs are included; compare policy and cost together.</small>
<script>const data=${json};
const slider=document.querySelector('#time'),view=document.querySelector('#view'),cards=document.querySelector('#cards'),stamp=document.querySelector('#stamp');
const frames=data[0].frames.filter(s=>data.every(d=>d.frames.some(t=>t.frame===s.frame)));slider.max=frames.length-1;
const fmt=(x,d=2)=>x===null?'—':Number(x).toFixed(d);const canvases=[];
data.forEach(d=>{const card=document.createElement('section');card.className='card';const title=document.createElement('h2');title.textContent=d.name;card.append(title);{const note=document.createElement('p');note.style.cssText='padding:0 16px;font-size:12px;min-height:54px;margin:0';note.textContent=d.failure||(d.fullPressureEnvelope?'Full pressure allowance; scheduling cost included.':d.pressureReserve!==null?'Lagged plan + '+d.pressureReserve+' reserve slot(s); unchanged acceptance tolerance.':'Normal pressure schedule; unchanged acceptance tolerance.');card.append(note);}const canvas=document.createElement('canvas');canvas.width=d.lattice.nx;canvas.height=d.lattice.ny;card.append(canvas);const metric=document.createElement('div');metric.className='metric';card.append(metric);cards.append(card);canvases.push({canvas,metric});});
function render(){const frame=frames[+slider.value].frame;stamp.textContent=fmt(frame*data[0].dt)+' s';const ref=data[0].frames.find(s=>s.frame===frame);document.querySelector('#legend').textContent=view.value==='difference'?'Orange: more liquid than reference. Blue: less liquid. Fixed scale ±0.25 depth-mean fill.':view.value==='phi'?'Cyan: negative phi (liquid). Gray: air. Fine white line: zero-contour crossing.':'Dark: empty. Brighter cyan: more liquid, fixed scale 0–1. Orange: depth-average overfill.';
data.forEach((d,k)=>{const s=d.frames.find(s=>s.frame===frame),{canvas,metric}=canvases[k],ctx=canvas.getContext('2d'),im=ctx.createImageData(canvas.width,canvas.height),n=canvas.width;
for(let y=0;y<canvas.height;y++)for(let x=0;x<n;x++){const i=x+n*y,j=4*(x+n*(canvas.height-1-y));let c;if(view.value==='difference'){const delta=s.projection[i]-ref.projection[i],t=Math.min(1,Math.abs(delta)/.25);c=delta>=0?[25+225*t,35+95*t,43-10*t]:[25+20*t,35+110*t,43+210*t];}else if(view.value==='phi'){const p=s.phiSlice[i],edge=(x+1<n&&p*s.phiSlice[i+1]<0)||(y+1<canvas.height&&p*s.phiSlice[i+n]<0);c=edge?[230,245,245]:p<=0?[50,163,181]:[24,34,43];}else{const v=s.projection[i],t=Math.min(1,Math.max(0,v));c=v>1.001?[246,150,66]:[18+43*t,28+166*t,37+172*t];}im.data.set([...c.map(Math.round),255],j);}ctx.putImageData(im,0,0);
const drift=100*(s.mass/d.frames[0].mass-1),dx=s.massFront_cells.p99-ref.massFront_cells.p99;
metric.innerHTML='Mass drift <strong>'+fmt(drift,3)+'%</strong> · excess '+fmt(100*s.excess/s.mass)+'% · h tiles '+s.fineTiles+'<br>99% mass front '+fmt(s.massFront_cells.p99)+'h · difference '+fmt(dx)+'h<br>Center of mass (x, y) '+s.centroid_cells.slice(0,2).map(v=>fmt(v)+'h').join(', ')+'<br>Liquid in phi-air '+fmt(100*s.massInPhiAir/s.mass)+'% · max fill '+fmt(s.maxVolume);
});}slider.oninput=render;view.onchange=render;let timer;document.querySelector('#play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='Play';}else{e.target.textContent='Pause';timer=setInterval(()=>{slider.value=(+slider.value+1)%frames.length;render();},650);}};
data.forEach(d=>{const last=d.frames.at(-1);let front=0;for(const s of d.frames){const r=data[0].frames.find(r=>r.frame===s.frame);if(r&&r.massFront_cells.p99<data[0].lattice.nx-4)front=Math.max(front,Math.abs(s.massFront_cells.p99-r.massFront_cells.p99));}const tr=document.createElement('tr');for(const text of [d.name,fmt(d.gpu),fmt(d.meanFine,0),fmt(100*(last.mass/d.frames[0].mass-1),3)+'%',fmt(front)+'h']){const td=document.createElement('td');td.textContent=text;tr.append(td);}document.querySelector('#summary').append(tr);});render();
</script></html>`);
console.log(out);
