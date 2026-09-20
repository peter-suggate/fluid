/** Figure 3: track visible airborne drops separately from conservative mass.
 * node --import tsx tools/wasm/uniform-figure3-regression.ts
 * The native runner executes the same Rust world and resolved values as Wasm.
 */
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { sceneDocument } from "../../lib/core/scene-definition";
import { getSceneDefinition } from "../../lib/core/scenes";
import { uniformLabSeed, UNIFORM_LAB_VALUES } from "../../lib/physics-wasm/uniform-controller";
const out="docs/research/uniform-geometric-figure3-2026-09-20";
mkdirSync(out,{recursive:true});
const seed=uniformLabSeed(sceneDocument(getSceneDefinition("cm12-figure-3")));
const [nx,ny]=seed.dimensions;
const save=(name:string,value:unknown)=>writeFileSync(`${out}/${name}.json.gz`,gzipSync(JSON.stringify(value)));
assert.equal(spawnSync("cargo",["build","--release","--manifest-path","rust/Cargo.toml","-p","fluid-core","--example","uniform_geometric_scene"],{stdio:"inherit"}).status,0);
function measure(volume:number[],phi:number[]){
 let airborneVolume=0,airborneArea=0;
 const ballAreas=[0,0,0,0];
 for(let y=25;y<ny;y++)for(let x=0;x<nx;x++){
  const i=x+nx*y,j=x+(nx+1)*y;airborneVolume+=volume[i]!;
  const c=[phi[j]!,phi[j+1]!,phi[j+nx+1]!,phi[j+nx+2]!];
  let area=0;
  if(c.every(v=>v<0))area=1;
  else if(c.some(v=>v<0))for(let sy=0;sy<8;sy++)for(let sx=0;sx<8;sx++){
   const u=(sx+.5)/8,v=(sy+.5)/8;
   if((c[0]!*(1-u)+c[1]!*u)*(1-v)+(c[2]!*(1-u)+c[3]!*u)*v<0)area+=1/64;
  }
  airborneArea+=area;ballAreas[x<40?0:x<74?1:x<103?2:3]!+=area;
 }
 return {mass:volume.reduce((s,v)=>s+v,0),airborneVolume,airborneArea,ballAreas};
}
const initial=measure(seed.volume,seed.phi);
const summaries:unknown[]=[];
const baselineBin=process.argv.find(a=>a.startsWith("--baseline-bin="))?.split("=").slice(1).join("=");
for(const [name,balance,pool] of [["fixed",false,true],["fixed-balanced",true,true],["no-pool",false,false],...(baselineBin?[["prior",false,true] as const]:[])] as const){
 const input={...seed,options:UNIFORM_LAB_VALUES,dt:1/30,frames:pool?40:24,
  volume:seed.volume.map((v,i)=>pool||Math.floor(i/nx)>25?v:0),
  phi:seed.phi.map((v,i)=>pool||Math.floor(i/(nx+1))>25?v:(26-Math.floor(i/(nx+1)))*seed.cellSize[1]),
  energyExperiment:{mode:balance?"balance-surface-deficit":"off",auditReceipt:false},
  energySnapshotFrames:[12,20,24,30,36,40],energySnapshotStages:["projected"]};
 assert.equal(input.options.extensionFrontSweeps,2,"regression must retain the requested two-sweep budget");
 const child=spawnSync(name==="prior"?baselineBin!:"rust/target/release/examples/uniform_geometric_scene",[],{input:JSON.stringify(input),encoding:"utf8",maxBuffer:128*1024*1024});
 assert.equal(child.status,0,child.stderr);const result=JSON.parse(child.stdout);
 save(`${name}-input`,input);save(name,result);
 const frames=result.energySnapshots.map((s:{frame:number;volume:number[];phi:number[]})=>({frame:s.frame,...measure(s.volume,s.phi)}));
 const before=frames.find((s:{frame:number})=>s.frame===24)!;
 const end=measure(result.volume,result.phi);
 const inputMass=input.volume.reduce((s,v)=>s+v,0);
 assert.ok(Math.abs(end.mass-inputMass)/inputMass<1e-5,"conservative mass must remain conserved");
 if(name!=="prior"){
  assert.ok(before.airborneArea/initial.airborneArea>.9,`${name}: airborne area ${before.airborneArea/initial.airborneArea}`);
  for(let b=0;b<4;b++)assert.ok(before.ballAreas[b]/initial.ballAreas[b]!>.5,`${name}: ball ${b} lost most of its visible area`);
  if(pool){
   const impact=frames.find((s:{frame:number})=>s.frame===30)!;
   assert.ok(impact.airborneVolume/initial.airborneVolume<.02,`${name}: mass stranded above the pool at impact`);
  }
 }else{
  assert.ok(before.airborneArea/initial.airborneArea<.8,"prior configuration must reproduce the regression");
  assert.ok(end.airborneVolume/initial.airborneVolume>.5,"prior leaves invisible mass aloft");
 }
 const summary={name,sweeps:input.options.extensionFrontSweeps,balance,elapsedMs:result.elapsedMs,frames};summaries.push(summary);console.log(JSON.stringify(summary));
}
writeFileSync(`${out}/summary.json`,JSON.stringify({initial,summaries},null,2)+"\n");
