import {test} from "node:test";
import assert from "node:assert/strict";
import {uniformDetailCensus} from "../tools/uniform-detail-census";

const run=(f:(x:number,y:number,z:number)=>number)=>uniformDetailCensus([4,4,4],new Uint32Array([0x80000000]),new Float32Array(64).fill(.5),Float32Array.from({length:125},(_,i)=>f(i%5,Math.floor(i/5)%5,Math.floor(i/25))),1);
test("affine surface is reproduced by both candidate widths",()=>{
 const r=run((x,y,z)=>x+2*y+3*z-7.5);
 assert.equal(r.crossingTiles,1);
 for(const c of r.candidates){assert.equal(c.crossingError_h.p99,0);assert.equal(c.thresholds[0]!.eligibleCrossingTiles,1);}
 assert.equal(r.candidates[0]!.thresholds[0]!.hypotheticalOwnerReduction,7/8);
});
test("sub-coarse-grid liquid sheet is protected by its sampled sign changes",()=>{
 const r=run(x=>Math.abs(x-1)-.2);
 for(const c of r.candidates){assert.equal(c.crossingSignSafe,0);assert.equal(c.thresholds.at(-1)!.eligibleCrossingTiles,0);}
});
test("2h can resolve a feature which 4h erases",()=>{
 const r=run(x=>Math.abs(x-2)-.2);
 assert.equal(r.candidates[0]!.crossingSignSafe,1);
 assert.equal(r.candidates[1]!.crossingSignSafe,0);
});
