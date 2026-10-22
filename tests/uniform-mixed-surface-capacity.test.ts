import assert from "node:assert/strict";
import test from "node:test";
import {uniformMixedSurfaceCapacity,UNIFORM_SURFACE_STAGING_TILES} from "../lib/methods/uniform/uniform-mixed-surface-capacity";

test("surface staging stays bounded independently of admitted fine resolution",()=>{
  for(const tiles of [0,1,512,7225,32768,262144,1048576]){
    const {traceEntries,cubicEntries}=uniformMixedSurfaceCapacity(tiles);
    assert.equal(traceEntries,64*Math.min(tiles,UNIFORM_SURFACE_STAGING_TILES));
    assert.equal(cubicEntries,125*Math.min(tiles,UNIFORM_SURFACE_STAGING_TILES));
    assert.ok(cubicEntries>=traceEntries,"opposite-end cubic lists cannot meet for admitted traces");
  }
  const small=uniformMixedSurfaceCapacity(512);
  assert.deepEqual(small,{traceEntries:32768,cubicEntries:64000});
  assert.deepEqual(uniformMixedSurfaceCapacity(262144),uniformMixedSurfaceCapacity(1048576));
  for(const invalid of [-1,NaN,Infinity,.5,Number.MAX_SAFE_INTEGER+1])assert.throws(()=>uniformMixedSurfaceCapacity(invalid));
});

test("two cubic partitions plus deferred overflow preserve every scheduled vertex",()=>{
  // Model every partition of an admitted batch, including all-free and
  // all-solid cubics. Trace overflow belongs to the general vertex list,
  // never either bounded cubic partition.
  const {traceEntries,cubicEntries}=uniformMixedSurfaceCapacity(1);
  for(let total=0;total<=traceEntries+16;total++){
    const admitted=Math.min(total,traceEntries);
    for(let free=0;free<=admitted;free++){
      const solid=admitted-free;
      const freeEnd=free-1,solidBegin=cubicEntries-solid;
      assert.ok(freeEnd<solidBegin);
      assert.ok(solidBegin>=0&&solidBegin<=cubicEntries);
      assert.equal(free+solid+Math.max(0,total-traceEntries),total);
    }
  }
});
