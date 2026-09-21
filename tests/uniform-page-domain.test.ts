import test from "node:test";
import assert from "node:assert/strict";
import {initialUniformPageDomain} from "../lib/methods/uniform/uniform-page-domain";

test("page-owned cells and vertices have one owner under reordered and partial pages",()=>{
 for(const dims of [[32,32,32],[64,64,64],[80,48,96]] as const){
  const d=initialUniformPageDomain(dims,32,true);
  const cells=new Uint8Array(dims[0]*dims[1]*dims[2]);
  const vertices=new Uint8Array((dims[0]+1)*(dims[1]+1)*(dims[2]+1));
  for(let page=0;page<d.count;page++){
   const slot=d.words[16+16*d.capacity+page]!;
   const origin=[0,1,2].map(a=>d.words[16+16*slot+a]!*32);
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++){
    const q=[origin[0]!+x,origin[1]!+y,origin[2]!+z];
    if(q.every((v,a)=>v<dims[a]!)&&x<32&&y<32&&z<32)cells[q[0]!+dims[0]*(q[1]!+dims[1]*q[2]!)]!++;
    if(q.some((v,a)=>v>dims[a]!))continue;
    if([x,y,z].some((v,a)=>v>=32&&q[a]!==dims[a]))continue;
    vertices[q[0]!+(dims[0]+1)*(q[1]!+(dims[1]+1)*q[2]!)]!++;
   }
  }
  assert.ok(cells.every(n=>n===1),`cell ownership ${dims}`);
  assert.ok(vertices.every(n=>n===1),`vertex ownership ${dims}`);
 }
});

test("mini64 uses the same page traversal as large scenes",()=>{
 const d=initialUniformPageDomain([64,64,64]);
 assert.equal(d.count,8);
 assert.deepEqual(Array.from(d.words.slice(0,3)),[64,8,8]);
 assert.deepEqual(Array.from(d.words.slice(4,7)),[72,9,9]);
});


test("geometric pages use the evidence budget independently of legacy saved controls",async()=>{
 const {uniformGeometricSolverOptions}=await import("../lib/methods/uniform/uniform-geometric-options");
 const {resolveUniformGeometricValues}=await import("../lib/methods/uniform/uniform-geometric-parameters");
 const saved={pressureCycleBudget:"fixed",pressureBudgetHeadroom:4};
 const values=resolveUniformGeometricValues(saved);
 assert.equal(values.pressureCycleBudget,undefined);
 assert.equal(values.pressureBudgetHeadroom,undefined);
 assert.equal(uniformGeometricSolverOptions(saved).pressureCycleBudget,"lagged");
 assert.equal(uniformGeometricSolverOptions(saved).pressureBudgetHeadroom,0);
});

test("every geometric scene uses pages regardless of saved dense storage",async()=>{
 const {uniformGeometricSolverOptions}=await import("../lib/methods/uniform/uniform-geometric-options");
 const {UNIFORM_GEOMETRIC_PARAMS,resolveUniformGeometricValues}=await import("../lib/methods/uniform/uniform-geometric-parameters");
 for(const volumeStorage of ["auto","dense","pages16","pages32"]){
  const options=uniformGeometricSolverOptions({volumeStorage,activeRegion:"on",pressureWindow:"window"});
  assert.equal(options.pageDomain,true);assert.equal(options.volumePages,32);
  assert.equal(options.activeRegion,false);assert.equal(options.pressureWindow,false);
  assert.equal(resolveUniformGeometricValues({volumeStorage}).volumeStorage,undefined);
 }
 assert.equal(UNIFORM_GEOMETRIC_PARAMS.some(p=>p.key==="volumeStorage"),false);
});


test("page-size selection controls both domain and transport and requires rebuild",async()=>{
 const {uniformGeometricSolverOptions}=await import("../lib/methods/uniform/uniform-geometric-options");
 const {UNIFORM_GEOMETRIC_PARAMS,resolveUniformGeometricValues}=await import("../lib/methods/uniform/uniform-geometric-parameters");
 const param=UNIFORM_GEOMETRIC_PARAMS.find(p=>p.key==="pageSize");
 assert.ok(param);assert.equal(param.update,"solver");
 const {uniformVolumeMethod}=await import("../lib/methods/uniform/uniform-volume-method");
 const {methodConfigurationImpact}=await import("../lib/core/method-lifecycle");
 assert.equal(methodConfigurationImpact(uniformVolumeMethod,"balanced",{pageSize:"32"},{pageSize:"16"}),"rebuild");
 for(const pageSize of ["16","32"]){
  assert.equal(resolveUniformGeometricValues({pageSize}).pageSize,pageSize);
  assert.equal(uniformGeometricSolverOptions({pageSize}).volumePages,Number(pageSize));
 }
 assert.equal(resolveUniformGeometricValues({pageSize:"8"}).pageSize,"32");
 assert.equal(uniformGeometricSolverOptions().volumePages,32);
});
