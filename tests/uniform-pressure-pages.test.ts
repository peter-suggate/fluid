import assert from "node:assert/strict";
import test from "node:test";
import {uniformPressurePageExtent,uniformPressurePageWorkgroups,uniformPressurePagedShader} from "../lib/methods/uniform/uniform-pressure-pages";
import {uniformPressureMultigridWGSL} from "../lib/methods/uniform/webgpu-uniform-pressure-multigrid.wgsl";

test("pressure page coordinates cover each logical cell once across partial pages",()=>{
 for(const d of [[66,66,66],[146,98,98],[11,5,5]] as const){
  const extent=uniformPressurePageExtent(d),groups=uniformPressurePageWorkgroups(d);
  const visited=new Set<number>();let count=0;
  const grid=d.map(n=>Math.ceil(n/16));
  for(let page=0;page<groups[0]/4;page++)for(let z=0;z<16;z++)for(let y=0;y<16;y++)for(let x=0;x<16;x++){
   const q=[16*(page%grid[0]!)+x,16*(Math.floor(page/grid[0]!)%grid[1]!)+y,16*Math.floor(page/(grid[0]!*grid[1]!))+z];
   if(q.some((v,a)=>v>=d[a]!))continue;
   const physical=[x+16*(page%(extent[0]/16)),y+16*(Math.floor(page/(extent[0]/16))%(extent[1]/16)),z+16*Math.floor(page/((extent[0]/16)*(extent[1]/16)))];
   assert.ok(physical.every((v,a)=>v<extent[a]!));
   visited.add(physical[0]!+extent[0]*(physical[1]!+extent[1]*physical[2]!));count++;
  }
  assert.equal(count,d[0]*d[1]*d[2]);assert.equal(visited.size,count);
 }
});
test("CM11a page conversion covers nested loads and all hierarchy fields",()=>{
 const shader=uniformPressurePagedShader(uniformPressureMultigridWGSL);
 const operators=shader.slice(0,shader.indexOf("fn mgPageAddress"));
 assert.doesNotMatch(operators,/texture(?:Load|Store)\(mg/);
 assert.match(operators,/mgPressureOutStore\(id,mgResidualInLoad\(id\)\)/);
 assert.match(operators,/mgRhsInLoad\(id\)/);
 assert.match(operators,/mgMinimumInLoad\(id\)/);
 assert.match(operators,/let slot=gid.x\/16u/);
});

test("logical-dispatch oracle retains atlas loads without tile launch decoding",()=>{
 const shader=uniformPressurePagedShader(uniformPressureMultigridWGSL,true);
 assert.match(shader,/fn mgActiveId\(gid:vec3u\)->vec3i\{return vec3i\(gid\);\}/);
 assert.doesNotMatch(shader,/let slot=gid.x\/16u/);
 assert.match(shader,/mgPressureInLoad/);
 assert.match(shader,/fn mgPageAddress/);
});
