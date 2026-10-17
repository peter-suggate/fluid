/** Compare activity on the same authored pool, with diagnostic atomics separated.
 * node --import tsx tools/compare-pool-activity.ts settled-time.json pool-impact-time.json settled-counts.json pool-impact-counts.json
 */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const paths=process.argv.slice(2);assert.equal(paths.length,4,"expected settled-time pool-impact-time settled-counts pool-impact-counts JSON files");
const [settled,impact,settledCounts,impactCounts]=paths.map(path=>JSON.parse(readFileSync(path,"utf8")));
for(const run of [impact,settledCounts,impactCounts])assert.equal(run.sourceHash,settled.sourceHash,"capture sources differ");
for(const run of [settled,settledCounts])assert.equal(run.scenario,"settled");
for(const run of [impact,impactCounts])assert.equal(run.scenario,"pool-impact");
for(const run of [settled,impact]){assert.equal(Boolean(run.workCounters),false);assert.equal(run.profiled,true);}
for(const run of [settledCounts,impactCounts])assert.equal(run.workCounters,true);
const gridConfiguration=(run:any)=>{const {impact:_impact,...grid}=run.configuration;return grid;};
for(const run of [impact,settledCounts,impactCounts])if(settled.configuration&&run.configuration)assert.deepEqual(gridConfiguration(run),gridConfiguration(settled),"timestep/lattice/container differ");
if(impact.configuration?.impact&&impactCounts.configuration?.impact)assert.deepEqual(impact.configuration.impact,impactCounts.configuration.impact,"impact diagnostic and timed settings differ");
const mean=(values:number[])=>values.reduce((sum,x)=>sum+x,0)/Math.max(1,values.length);
const wall=(run:any)=>{
 const selected=run.rows.filter((row:any)=>row.frame>=run.from);
 return {window:[run.from,run.steps],mean_ms:run.summary.mean_ms,median_ms:run.summary.median_ms,
  excludingFirstFrameMean_ms:mean(selected.filter((row:any)=>row.frame>1).map((row:any)=>row.wall_ms))};
};
function distanceMeans(run:any){
 const rows=run.rows.filter((row:any)=>row.frame>=run.from);
 assert.ok(rows.length&&rows.every((row:any)=>row.work?.regionsByHorizontalDistance),"missing region final-store diagnostics");
 return rows[0].work.regionsByHorizontalDistance.map((sample:any,index:number)=>{
  const samples=rows.map((row:any)=>row.work.regionsByHorizontalDistance[index]);
  const average=(key:string)=>mean(samples.map((item:any)=>item[key]??0));
  const stores=Object.fromEntries(Object.keys(sample.stores).map(label=>[label,mean(samples.map((item:any)=>item.stores[label]??0))]));
  const transport=(stores.affine??0)+(stores.middle??0)+(stores.fullLinear??0)+(stores.fullCubic??0);
  return {range_m:sample.range_m,storedRegions:average("storedRegions"),regionsWithStores:average("regionsWithStores"),geometryDirtyRegions:average("geometryDirty"),normalDirtyRegions:sample.normalDirtyRegions===undefined?undefined:average("normalDirtyRegions"),storesPerFrame:stores,
   transportStoresPerFrame:transport,cheapTransportShare:((stores.affine??0)+(stores.middle??0))/Math.max(1,transport)};
 });
}
const calmBins=distanceMeans(settledCounts),splashBins=distanceMeans(impactCounts);assert.equal(calmBins.length,splashBins.length);
const stageLabels=new Set<string>([...Object.keys(settled.summary.passMeans_ms),...Object.keys(impact.summary.passMeans_ms)].filter(label=>/surface advect|geometric fill|body forces|capillar|normal|curvature|work control|surface model|work evidence/i.test(label)));
console.log(JSON.stringify({sourceHash:settled.sourceHash,configuration:settled.configuration,
 scope:"Same authored pool and paging; rendering excluded. Timed runs disable diagnostic atomics. Distance is horizontal from the configured impact center (default0,0). Region counts and final-store counts are per-frame means over each diagnostic window. Geometry/normal flags describe their final encode, not sums over all geometry passes.",
 wall:{settled:wall(settled),poolImpact:wall(impact)},
 stageMeans_ms:Object.fromEntries([...stageLabels].map(label=>[label,{settled:settled.summary.passMeans_ms[label]??0,poolImpact:impact.summary.passMeans_ms[label]??0}])),
 diagnostics:{settledWindow:[settledCounts.from,settledCounts.steps],poolImpactWindow:[impactCounts.from,impactCounts.steps]},
 distanceTable:calmBins.map((calm:any,index:number)=>{const splash=splashBins[index]!;assert.deepEqual(calm.range_m,splash.range_m);return {range_m:calm.range_m,settled:calm,poolImpact:splash,fullStoreIncreasePerFrame:(splash.storesPerFrame.fullLinear+splash.storesPerFrame.fullCubic)-(calm.storesPerFrame.fullLinear+calm.storesPerFrame.fullCubic)};})},null,2));
