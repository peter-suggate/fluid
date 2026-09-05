import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("source topology lease brackets capture only and releases on success or failure", async () => {
 for(const failure of [false,true]) {
  const writes: {offset:number;words:number[]}[]=[];
  const resident=Object.assign(Object.create(WebGPUSparseCM12Resident.prototype),{
   destroyed:false,sourceTopologyLeaseHeld:false,activity:{},
   device:{queue:{writeBuffer(_buffer:unknown,offset:number,data:Uint32Array){
    writes.push({offset,words:[...data]});
   }}},
   async captureGenerationTransferSourceWhileLeased(){
    assert.equal(writes.length,1);
    assert.deepEqual(writes[0],{offset:25*4,words:[1,0]});
    await Promise.resolve();
    if(failure) throw new Error("capture failed");
    return {captured:true};
   },
  }) as WebGPUSparseCM12Resident;
  if(failure) await assert.rejects(resident.captureGenerationTransferSource(),/capture failed/);
  else assert.deepEqual(await resident.captureGenerationTransferSource(),{captured:true});
  assert.deepEqual(writes,[{offset:100,words:[1,0]},{offset:100,words:[0]}]);
  // Release is complete before a caller could begin expensive preparation.
  const release=resident.acquireGenerationSourceTopologyLease(); release(); release();
  assert.equal(writes.length,4,"an already released lease must not issue another write");
 }
});

test("source lease gates optional scheduling while urgent physical work revokes it", () => {
 const wgsl=readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",import.meta.url),"utf8");
 const allocator=wgsl.slice(wgsl.indexOf("fn allocateSparseWorldFrontier("),
  wgsl.indexOf("fn clearSparseWorldFrontierResolutionCache("));
 assert.match(allocator,/revokeCM12SourceTopologyLease\(\);\s*let leaf=cm12WorldAllocateExact/);
 assert.match(wgsl,/fn stageDemandedFrontierPage\(brick:u32\)\{\s*revokeCM12SourceTopologyLease\(\);/);
 assert.match(wgsl,/if\(requested>current&&brickCandidatePlanningEnabled\(brick\)\)\{revokeCM12SourceTopologyLease\(\);\}/);
 const schedule=wgsl.slice(wgsl.indexOf("fn scheduleTopologyPreparation("),
  wgsl.indexOf("fn candidateTopologyPageBase("));
 assert.match(schedule,/workgroupUniformLoad\(&sourceTopologyLeaseForSchedule\)/);
 assert.match(schedule,/setTopologyPreparationScheduled\(activityRecord\(brick\),false\)/);
 assert.match(schedule,/atomicStore\(&activity\[16\],0u\)/);
 const resident=readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",import.meta.url),"utf8");
 assert.match(resident,/observed\[0\] !== 1 \|\| observed\[1\] !== 0 \|\| observed\[2\] !== activity.acceptedTopologyGeneration/);
 assert.match(resident,/latest.acceptedTopologyGeneration !== source.activity.acceptedTopologyGeneration/,
  "coherent capture does not authorize publishing after later topology changes");
});
