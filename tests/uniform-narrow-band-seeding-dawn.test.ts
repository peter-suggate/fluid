import assert from "node:assert/strict";
import test from "node:test";
import {sceneDocument} from "../lib/core/scene-definition";
import {getSceneDefinition} from "../lib/core/scenes";
import {uniformNarrowBandMethod} from "../lib/methods/uniform/uniform-narrow-band-method";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";
import type {WebGPUUniformReferenceSolver} from "../lib/methods/uniform/webgpu-uniform-reference";
import {withUniformDevice} from "./helpers/uniform-geometric";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("listed seed candidates preserve cell seeding and capacity receipts",{timeout:180_000},async()=>{
 await withUniformDevice("NB listed seeding",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];scene.fluid.inflow=undefined;
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{adaptiveSurface:"on",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   const access=stage as unknown as {parity:number;bins:GPUBuffer;next:GPUBuffer;params:GPUBuffer;activityWord:number;dispatch(e:GPUCommandEncoder,entry:string,group?:string):void};
   access.parity=0;
   const cells=32**3,tiles=cells/64,vertices=33**3,depth=2*cells+2*tiles,band=depth+vertices+cells;
   const words=new Uint32Array(access.bins.size/4),floats=new Float32Array(words.buffer);
   for(let i=0;i<cells;i++)words[i]=[0,3,7,8][i%4]!;
   for(let z=0;z<=32;z++)for(let y=0;y<=32;y++)for(let x=0;x<=32;x++)floats[depth+x+33*(y+33*z)]=y-16;
   words[band+1]=tiles;
   for(let t=0;t<tiles;t++){
    words[band+4+2*tiles+t]=1;
    words[band+4+4*tiles+t]=t;
    floats[access.activityWord+3*tiles+t]=2;
   }
   // Seed-all makes the serial bootstrap and listed replenishment evaluate
   // the same sites, including the projected outer shell.
   device.queue.writeBuffer(access.params,0,new Float32Array([1/32,1/32,1/32,0,0.95,0,0,3]));
   const read=async(buffer:GPUBuffer,offset:number,size:number)=>{
    const target=device.createBuffer({size,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{const e=device.createCommandEncoder();e.copyBufferToBuffer(buffer,offset,target,0,size);device.queue.submit([e.finish()]);await target.mapAsync(GPUMapMode.READ);return target.getMappedRange().slice(0);}
    finally{target.destroy();}
   };
   const run=async(listed:boolean,first=0)=>{
    device.queue.writeBuffer(access.bins,0,words);device.queue.writeBuffer(stage.state,0,new Uint32Array([0,first,0,0]));
    const e=device.createCommandEncoder();
    for(const entry of listed?["seedCells","seedSources","seed"]:["seedInitial"])access.dispatch(e,entry,"update");
    device.queue.submit([e.finish()]);
    const receipt=new Uint32Array(await read(stage.state,0,16));
    const kept=Math.min(stage.capacity,receipt[1]!)-first;
    const particles=kept?new Float32Array(await read(stage.particles[0],48*first,48*kept)):new Float32Array();
    return {receipt,particles};
   };
   const reference=await run(false),actual=await run(true);
   assert.ok(reference.receipt[1]!>1000,"exercise substantial seeding, shell projection, and partially populated cells");
   assert.deepEqual(actual.receipt,reference.receipt);
   const records=(values:Float32Array)=>Array.from({length:values.length/12},(_,i)=>Array.from(values.subarray(12*i,12*i+12))).sort((a,b)=>{
    for(let i=0;i<12;i++){const d=a[i]!-b[i]!;if(d!==0)return d;}return 0;
   });
   const expected=records(reference.particles),observed=records(actual.particles);
   for(let i=0;i<expected.length;i++)for(let j=0;j<12;j++)assert.ok(Math.abs(expected[i]![j]!-observed[i]![j]!)<2e-6,`candidate ${i}, component ${j}`);
   const clipped=await run(true,stage.capacity-7);
   assert.equal(clipped.receipt[1],stage.capacity-7+reference.receipt[1]!);
   assert.equal(clipped.receipt[2],reference.receipt[1]!-7);
   assert.equal(clipped.particles.length,7*12);
   assert.ok(clipped.particles.every(Number.isFinite));
  }finally{solver.destroy();}
 },["subgroups"]);
});
