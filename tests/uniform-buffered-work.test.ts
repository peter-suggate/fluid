import assert from "node:assert/strict";
import test from "node:test";
import {uniformBufferedWork} from "../lib/methods/uniform/uniform-buffered-work";

test("direct work budget grows with headroom and shrinks gradually to one",()=>{
 let budget=1;
 budget=uniformBufferedWork(budget,100,4096);assert.equal(budget,125);
 budget=uniformBufferedWork(budget,0,4096);assert.equal(budget,63);
 for(let i=0;i<8;i++)budget=uniformBufferedWork(budget,0,4096);
 assert.equal(budget,1);
 assert.equal(uniformBufferedWork(budget,4096,4096),4096);
});

test("a stale buffered grid covers every current slot exactly once at every tier",()=>{
 // Empty -> full and full -> empty transitions, including lists larger
 // than both the budget and the hardware saturation cap.
 for(const prior of [0,1,17,4096])for(const live of [0,1,7,65,5003]){
  const budget=uniformBufferedWork(1,prior,8192);
  for(const [slotsPerGroup,cap] of [[1,2048],[1,4096],[4,2048],[8,1024],[64,256]]){
   const groups=Math.min(Math.ceil(budget/slotsPerGroup!),cap!);
   const visits=new Uint8Array(live);
   for(let g=0;g<groups;g++)for(let base=g*slotsPerGroup!;base<live;base+=groups*slotsPerGroup!)
    for(let lane=0;lane<slotsPerGroup!&&base+lane<live;lane++)visits[base+lane]++;
   assert.ok(visits.every(n=>n===1),`${prior} -> ${live}, tier ${slotsPerGroup}`);
  }
 }
});
