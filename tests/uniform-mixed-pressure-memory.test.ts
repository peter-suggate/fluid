import assert from "node:assert/strict";
import test from "node:test";
import {UniformScratchLayout} from "../lib/methods/uniform/uniform-scratch-arena";
import {createUniformMixedLayout} from "../lib/methods/uniform/uniform-mixed-layout";
import {planUniformMixedPressureMemory} from "../lib/methods/uniform/uniform-mixed-pressure-memory";

test("unified all-fine pressure fits borrowed fields while preserving the native 4h continuation",()=>{
 for(const d of [[32,32,32],[64,64,64],[128,128,128],[256,64,128]] as const){
  const n=d.reduce<number>((a,b)=>a*b,1),arena=new UniformScratchLayout(d,n*40),prefix=arena.offset("Uniform CM11a L2 pressure A")!*4;
  const layout=createUniformMixedLayout({dimensions:d,cellSize_m:[1,1,1],origin_m:{x:0,y:0,z:0}},[]);
  const plan=planUniformMixedPressureMemory(layout,prefix,arena.conditioningBytes);
  assert.ok(plan.bytes<=prefix);assert.equal(plan.levels[0]!.phi.size,n*4);
  for(let i=0;i<3;i++){
   const l=plan.levels[i]!,views=[l.pressure,...l.rhs,...l.minimum,l.slopes,l.frozen,l.residual,...(i?[l.phi]:[]),plan.backup];
   for(let a=0;a<views.length;a++){assert.equal(views[a]!.offset%256,0);for(let b=0;b<a;b++){const x=views[a]!,y=views[b]!;assert.ok(x.offset+x.size<=y.offset||y.offset+y.size<=x.offset,"simultaneously live views overlap");}}
  }
 }
});
