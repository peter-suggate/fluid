/** Same manufactured input and extension; only the air correction differs. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { airFixture, runAirFixture } from "../tests/support/air-extension-fixture";
const output=process.argv.find(v=>v.startsWith("--out="))?.slice(6)??"artifacts/level-set-volume/air-frozen-3d.json";
await acquireWebGPUExclusiveLock("dawn-probe","frozen air correction surface flux");let device:GPUDevice|undefined;
try{
  const dawn=await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);Object.assign(globalThis,dawn.globals);
  const gpu=dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND??"metal"}`]);const adapter=await gpu.requestAdapter();assert.ok(adapter);device=await adapter.requestDevice();assert.ok(device);
  const results=[];
  for(const widths of [[1,1,1,1],[2,1,1],[1,1,2]]){
    const fixture=airFixture(widths);const center=[1.7,1.8,2.1],radius=1.25;
    fixture.cells.forEach(c=>c.phi=Math.hypot(...c.center.map((v,a)=>v-center[a]!))-radius);
    fixture.faces.forEach(f=>{
      const [x,y]=f.center.map(v=>v*Math.PI/4);
      f.velocity=f.terms.length===1?0:[Math.sin(x!)*Math.cos(y!),-Math.cos(x!)*Math.sin(y!),0][f.axis]!;
    });
    const count=512;const normals=Array.from({length:count},(_,i)=>{
      const z=1-2*(i+.5)/count,r=Math.sqrt(1-z*z),angle=i*Math.PI*(3-Math.sqrt(5));return [r*Math.cos(angle),r*Math.sin(angle),z];});
    const samples=normals.map(n=>n.map((v,a)=>center[a]!+radius*v));
    const arms=[];
    for(const project of [false,true]){
      const result=await runAirFixture(device,fixture,{samples,project});
      const flux=normals.reduce((s,n,i)=>s+n.reduce((v,a,k)=>v+a*result.samples[3*i+k]!,0),0)*4*Math.PI*radius*radius/count;
      arms.push({project,outwardFlux:flux,correctionCandidate:result.receipt});
    }
    results.push({widths,center,radius,samples:count,arms});
  }
  mkdirSync(dirname(output),{recursive:true});writeFileSync(output,JSON.stringify({kind:"frozen manufactured 3D sphere, identical face extension and sampler",results},null,2)+"\n");console.log(JSON.stringify(results));
}finally{device?.destroy();await releaseWebGPUExclusiveLock();}
