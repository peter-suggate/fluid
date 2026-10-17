/** Offline map data from an existing WORK_COUNTERS capture; no GPU work.
 * node --import tsx tools/summarize-pool-work-regions.ts counts.json > region-map.json
 */
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
const path=process.argv[2];assert.ok(path,"expected WORK_COUNTERS capture JSON path");
const capture=JSON.parse(readFileSync(path,"utf8")),sidecar=JSON.parse(readFileSync(path+".regions.json","utf8"));
assert.equal(capture.workCounters,true);assert.equal(sidecar.sourceHash,capture.sourceHash);assert.equal(sidecar.frame,capture.steps);
const labels=sidecar.labels as string[];assert.deepEqual(labels,["identity","affine","middle","fullLinear","fullCubic"]);
function fractions(stores:Record<string,number>){
 const total=Object.values(stores).reduce((sum,value)=>sum+value,0),transport=total-(stores.identity??0),cheap=(stores.affine??0)+(stores.middle??0),full=(stores.fullLinear??0)+(stores.fullCubic??0);
 return {stores,totalStores:total,transportStores:transport,tierFractions:Object.fromEntries(labels.map(label=>[label,(stores[label]??0)/Math.max(1,total)])),transportTierFractions:Object.fromEntries(labels.filter(label=>label!=="identity").map(label=>[label,(stores[label]??0)/Math.max(1,transport)])),cheapTransportShare:cheap/Math.max(1,transport),fullTransportShare:full/Math.max(1,transport)};
}
const extent=capture.configuration.cellSize_m.map((h:number)=>h*capture.configuration.workRegionEdge_h);
const records=sidecar.records.map((record:any)=>{
 assert.equal(record.stores.length,labels.length);const stores=Object.fromEntries(labels.map((label,index)=>[label,record.stores[index]]));
 return {region:record.region,center_m:record.center_m,bounds_m:{minimum:record.center_m.map((value:number,axis:number)=>value-extent[axis]/2),maximum:record.center_m.map((value:number,axis:number)=>value+extent[axis]/2)},horizontalDistance_m:record.horizontalDistance_m,geometryDirty:record.geometryDirty,normalDirty:record.normalDirty,...fractions(stores)};
});
const measured=capture.rows.filter((row:any)=>row.frame>=capture.from&&row.frame<=capture.steps),last=measured.at(-1),finalBins=last.work.regionsByHorizontalDistance;
const finalDistanceBins=finalBins.map((bin:any)=>({...bin,...fractions(bin.stores)}));
const meanDistanceBins=finalBins.map((bin:any,index:number)=>{
 const items=measured.map((row:any)=>row.work.regionsByHorizontalDistance[index]);
 const average=(key:string)=>items.reduce((sum:number,item:any)=>sum+(item[key]??0),0)/items.length;
 const stores=Object.fromEntries(labels.map(label=>[label,items.reduce((sum:number,item:any)=>sum+(item.stores[label]??0),0)/items.length]));
 return {range_m:bin.range_m,storedRegions:average("storedRegions"),regionsWithStores:average("regionsWithStores"),geometryDirty:average("geometryDirty"),normalDirtyRegions:average("normalDirtyRegions"),...fractions(stores)};
});
for(const label of labels){const regionSum=records.reduce((sum:number,record:any)=>sum+record.stores[label],0),binSum=finalBins.reduce((sum:number,bin:any)=>sum+bin.stores[label],0);assert.equal(regionSum,binSum,`region and distance totals differ for ${label}`);}
console.log(JSON.stringify({sourceHash:capture.sourceHash,probeSourceHash:capture.probeSourceHash,scenario:capture.scenario,configuration:capture.configuration,finalBodyPosition_m:capture.summary.finalBodyPosition_m,waterline_m:capture.configuration.waterline_m??capture.summary.correctness?.waterline,frame:capture.steps,measuredWindow:[capture.from,capture.steps],regionExtent_m:extent,labels,
 scope:"Current 16h regions and completed final texture stores, not attempts or unique world vertices. Fractions include identity; transport fractions exclude identity. Dirty flags describe final geometry/capillary encodes, not all passes. Per-region map is the final frame; mean distance bins cover the stated window. No instrumentation is added to timed runs.",
 finalDistanceBins,meanDistanceBins,regions:records},null,2));
