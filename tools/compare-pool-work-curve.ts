/** Summarize same-pool speed curves; pass timed and diagnostic JSON paths.
 * node --import tsx tools/compare-pool-work-curve.ts artifacts/.../prefix-*.json
 * Known .models/.regions/.bin metadata sidecars are ignored; every window2–30.
 */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const captures=process.argv.slice(2).filter(path=>!/[.](?:models|regions|bin)[.]json$/.test(path)).map(path=>({path,...JSON.parse(readFileSync(path,"utf8"))}));assert.ok(captures.length,"expected curve capture JSON paths");
const mean=(values:number[])=>values.reduce((sum,x)=>sum+x,0)/Math.max(1,values.length);
const grid=(run:any)=>{const {impact,geometrySnapshots,...rest}=run.configuration;return {...rest,impactXZ_m:impact.positionXZ_m};};
for(const run of captures){assert.equal(run.scenario,"pool-impact");assert.equal(run.steps,30);assert.equal(run.from,2);assert.equal(run.noBody,false);assert.equal(run.split,false);assert.equal(run.forceFull,false);assert.notEqual(run.configuration.geometrySnapshots,false);assert.deepEqual(grid(run),grid(captures[0]),"pool/timestep/paging/impact position differ");if(!run.workCounters)assert.equal(run.profiled,true);}
const hashes=[...new Set(captures.map(run=>run.sourceHash))];
const groups=hashes.map(hash=>{
 const runs=captures.filter(run=>run.sourceHash===hash),speeds=[...new Set<number>(runs.map(run=>run.configuration.impact.speed_m_s))].sort((a,b)=>a-b);
 const points=speeds.map(speed=>{
  const timed=runs.filter(run=>run.configuration.impact.speed_m_s===speed&&!run.workCounters),counts=runs.filter(run=>run.configuration.impact.speed_m_s===speed&&run.workCounters);
  assert.equal(timed.length,1,`need exactly one timed run per source/speed: ${hash} / ${speed}`);assert.ok(counts.length<=1,"duplicate diagnostic capture");const time=timed[0],diagnostic=counts[0];
  const rows=time.rows.filter((row:any)=>row.frame>=2),walls=rows.map((row:any)=>row.wall_ms).sort((a:number,b:number)=>a-b),diagnosticRows=diagnostic?.rows.filter((row:any)=>row.frame>=2);
  const bins=diagnosticRows?.[0]?.work?.regionsByHorizontalDistance.map((sample:any,index:number)=>{
   const items=diagnosticRows.map((row:any)=>row.work.regionsByHorizontalDistance[index]);
   const stores=Object.fromEntries(Object.keys(sample.stores).map(label=>[label,mean(items.map((item:any)=>item.stores[label]??0))]));
   const cheap=(stores.affine??0)+(stores.middle??0),full=(stores.fullLinear??0)+(stores.fullCubic??0);
   return {range_m:sample.range_m,storedRegions:sample.storedRegions,regionsWithStores:mean(items.map((item:any)=>item.regionsWithStores)),normalDirtyRegions:mean(items.map((item:any)=>item.normalDirtyRegions??0)),geometryDirtyRegions:mean(items.map((item:any)=>item.geometryDirty??0)),storesPerFrame:stores,cheapTransportShare:cheap/Math.max(1,cheap+full)};
  });
  const stores=bins?.reduce((sum:Record<string,number>,bin:any)=>{for(const [label,count]of Object.entries(bin.storesPerFrame))sum[label]=(sum[label]??0)+(count as number);return sum;},{});
  const passMeans=time.summary.passMeans_ms as Record<string,number>,sumStages=(pattern:RegExp)=>Object.entries(passMeans).filter(([label])=>pattern.test(label)).reduce((sum,[,ms])=>sum+ms,0);
  const surface_ms=sumStages(/^Uniform mixed surface advect/),geometry_ms=sumStages(/^Uniform mixed geometric fill$/),control_ms=sumStages(/^Uniform 16h /),forcePass_ms=passMeans["Uniform mixed body forces"]??0;
  return {speed_m_s:speed,maxSpeed_m_s:time.summary.maxSpeed_m_s,finalBodyPosition_m:time.summary.finalBodyPosition_m,costGroups_ms:{surface:surface_ms,geometry:geometry_ms,control:control_ms,forcePass:forcePass_ms,surfaceGeometryControl:surface_ms+geometry_ms+control_ms,includingWholeForcePass:surface_ms+geometry_ms+control_ms+forcePass_ms},timedPath:time.path,diagnosticPath:diagnostic?.path,window:[2,30],mean_ms:mean(walls),median_ms:walls[Math.floor(walls.length/2)],maximum_ms:Math.max(...walls),passMeans_ms:time.summary.passMeans_ms,correctness:time.summary.correctness,storesPerFrame:stores,distanceTable:bins};
 });
 return {sourceHash:hash,role:runs[0].benchmarkRole??"unspecified",points};
});
const candidate=groups.find(group=>group.role==="candidate"),baseline=groups.find(group=>group.role==="baseline");
const comparisons=candidate&&baseline?candidate.points.map(point=>{const reference=baseline.points.find(row=>row.speed_m_s===point.speed_m_s);return reference?{speed_m_s:point.speed_m_s,mean_ms:{baseline:reference.mean_ms,candidate:point.mean_ms,ratio:point.mean_ms/reference.mean_ms},median_ms:{baseline:reference.median_ms,candidate:point.median_ms,ratio:point.median_ms/reference.median_ms},costGroups_ms:{baseline:reference.costGroups_ms,candidate:point.costGroups_ms}}:undefined;}).filter(Boolean):undefined;
console.log(JSON.stringify({comparisons,scope:"Same authored pool, body geometry, timestep and32h paging; speed varies.29 measured frames(2–30), including all outliers. Diagnostics use separate runs with atomics; final stores exclude discarded deferred attempts. Dirty masks describe final encodes, not all passes. Shape/body changes can vary actual needed work even at the same speed.",configuration:grid(captures[0]),groups},null,2));
