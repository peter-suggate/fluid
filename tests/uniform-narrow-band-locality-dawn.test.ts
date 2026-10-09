import assert from "node:assert/strict";
import test from "node:test";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformNarrowBandMethod } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { withUniformDevice } from "./helpers/uniform-geometric";
import { readMixedBuffer } from "./helpers/uniform-mixed-native-fields";

(process.env.WEBGPU_NODE_MODULE?test:test.skip)("NB stable compaction preserves survivors and tile marking matches individual particles",{timeout:180_000},async()=>{
 await withUniformDevice("NB particle locality",async device=>{
  const scene=structuredClone(sceneDocument(getSceneDefinition("minimal-power-dam-break-32")));
  Object.assign(scene.container,{width_m:1,height_m:1,depth_m:1,fillFraction:0.5});
  scene.voxelDomain.finestCellSize_m=1/32;scene.rigidBodies=[];scene.solidVoxels=[];
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,"balanced",{adaptiveSurface:"on",detailPolicy:"full"},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  try{
   const stage=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   const access=stage as unknown as {parity:number;bins:GPUBuffer;next:GPUBuffer;params:GPUBuffer;activityWord:number;encodeParticleWork(e:GPUCommandEncoder):void;dispatch(e:GPUCommandEncoder,entry:string,group?:string):void};
   access.parity=0;
   const cells=32**3,tiles=cells/64,surface=2*cells+tiles;
   const theta=access.activityWord+5*tiles,ownHeat=theta+tiles;
   const words=async(b:GPUBuffer)=>new Uint32Array((await readMixedBuffer(device,b)).buffer);
   const bits=(v:number)=>new Uint32Array(new Float32Array([v]).buffer)[0]!;
   const below=(v:number)=>new Float32Array(new Uint32Array([bits(v)-1]).buffer)[0]!;
   device.queue.writeBuffer(access.params,0,new Float32Array([1/32,1/32,1/32,1/30,0.95,0,0,1]));
   // Cross lane, workgroup, and old 65,536-lane loop boundaries; include empty
   // input and an incomplete final block. Every record has a unique identity.
   for(const n of [0,63,64,65,131137]){
    const input=new Float32Array(n*12),counts=new Uint32Array(cells),starts=new Uint32Array(cells);
    for(let i=0;i<n;i++){
     const order=Math.floor(i/64),tile=order>>>6,local=order&63;
     const c=[4*(tile%8)+local%4,4*(Math.floor(tile/8)%8)+Math.floor(local/4)%4,4*Math.floor(tile/64)+Math.floor(local/16)];
     const heat=i%13===0?0:(i%4+1)/2,depth=i%5===0?-5:i%3===0?-0.5:-2;
     input.set([i%17===0?below(c[0]!+1):c[0]!+0.25,c[1]!+0.75,c[2]!+0.25,heat,i,2*i,-i,depth,i+1,i+2,i+3,0],12*i);
     counts[order]++;
    }
    let offset=0;for(let c=0;c<cells;c++){starts[c]=offset;offset+=counts[c]!;}
    const reset=device.createCommandEncoder();reset.clearBuffer(access.bins);reset.clearBuffer(access.next);device.queue.submit([reset.finish()]);
    device.queue.writeBuffer(access.bins,0,counts);device.queue.writeBuffer(access.next,0,starts);
    if(n)device.queue.writeBuffer(stage.particles[0],0,input);
    device.queue.writeBuffer(stage.state,0,new Uint32Array([n,0,0,0]));
    const e=device.createCommandEncoder();access.encodeParticleWork(e);
    for(const entry of ["resampleCount","resamplePrefix","resample"])access.dispatch(e,entry);
    device.queue.submit([e.finish()]);
    const expected:number[]=[];const retainedCounts=counts.slice();
    for(let i=0;i<n;i++){
     const cell=Math.floor(i/64),depth=input[12*i+7]!,heat=input[12*i+3]!;
     if(depth<-4||heat<=0){retainedCounts[cell]--;continue;}
     if(depth>=-1||i-starts[cell]!<16)expected.push(i);
    }
    const receipt=await words(stage.state),actual=await readMixedBuffer(device,stage.particles[1]);
    assert.equal(receipt[1],expected.length,`count at ${n}`);
    for(let j=0;j<expected.length;j++)assert.deepEqual(actual.subarray(12*j,12*j+12),input.subarray(12*expected[j]!,12*expected[j]!+12),`stable survivor ${j} at ${n}`);
    assert.deepEqual((await words(access.bins)).subarray(0,cells),retainedCounts,"retirement preserves seeding counts");
    assert.deepEqual((await words(access.next)).subarray(0,cells),starts,"compaction preserves cell-start metadata");

    // Independently scatter each original particle's support on the CPU. In
    // particular, near-integer float32 positions exercise halo roundoff.
    const compact=new Float32Array(n*4),expectedSurface=new Uint32Array(tiles),expectedTheta=new Float32Array(tiles),expectedOwn=new Float32Array(tiles);
    for(let i=0;i<n;i++){
     const q=Array.from(input.subarray(12*i,12*i+3)),heat=input[12*i+3]!;
     compact.set([...q,heat],4*i);
     const home=q.map(x=>Math.floor(x/4)),homeIndex=home[0]!+8*(home[1]!+8*home[2]!);
     expectedOwn[homeIndex]=Math.max(expectedOwn[homeIndex]!,heat);
     const lo=q.map(x=>Math.max(0,Math.floor(Math.fround(x-2)/4))),hi=q.map(x=>Math.min(7,Math.floor(Math.fround(x+2)/4)));
     for(let z=lo[2]!;z<=hi[2]!;z++)for(let y=lo[1]!;y<=hi[1]!;y++)for(let x=lo[0]!;x<=hi[0]!;x++){
      const t=x+8*(y+8*z);expectedSurface[t]=1;expectedTheta[t]=Math.max(expectedTheta[t]!,heat);
     }
    }
    const clear=device.createCommandEncoder();clear.clearBuffer(access.bins);device.queue.submit([clear.finish()]);
    device.queue.writeBuffer(access.bins,0,counts);device.queue.writeBuffer(access.next,0,starts);
    if(n)device.queue.writeBuffer(access.next,4*cells,compact);
    const marks=device.createCommandEncoder();access.dispatch(marks,"markParticleTiles");device.queue.submit([marks.finish()]);
    const marked=await words(access.bins);
    assert.deepEqual(marked.subarray(surface,surface+tiles),expectedSurface,`support at ${n}`);
    assert.deepEqual(marked.subarray(theta,theta+tiles),new Uint32Array(expectedTheta.buffer),`halo heat at ${n}`);
    assert.deepEqual(marked.subarray(ownHeat,ownHeat+tiles),new Uint32Array(expectedOwn.buffer),`occupied heat at ${n}`);

    // The surface scatter must equal the original radius-two vertex gather,
    // including crowded cells, empty input, domain faces and near-integer q.
    const splat=device.createCommandEncoder();access.dispatch(splat,"surfaceCells");access.dispatch(splat,"surfaceSplat");device.queue.submit([splat.finish()]);
    const distances=await words(access.bins),expectedDistance=new Float32Array(33**3).fill(4);
    for(let i=0;i<n;i++){
     const q=Array.from(compact.subarray(4*i,4*i+3)),c=q.map(Math.floor);
     for(let z=Math.max(0,c[2]!-1);z<=Math.min(32,c[2]!+2);z++)for(let y=Math.max(0,c[1]!-1);y<=Math.min(32,c[1]!+2);y++)for(let x=Math.max(0,c[0]!-1);x<=Math.min(32,c[0]!+2);x++){
      const dx=Math.fround(x-q[0]!),dy=Math.fround(y-q[1]!),dz=Math.fround(z-q[2]!);
      const d=dx*dx+dy*dy+dz*dz,index=x+33*(y+33*z);expectedDistance[index]=Math.min(expectedDistance[index]!,d);
     }
    }
    for(let i=0;i<expectedDistance.length;i++){
     const key=distances[2*cells+2*tiles+i]!,value=key===0?4:new Float32Array(new Uint32Array([~key>>>0]).buffer)[0]!;
     assert.ok(Math.abs(value-expectedDistance[i]!)<1e-6,`surface distance ${i} at ${n}: ${value} vs ${expectedDistance[i]}`);
    }
   }
  }finally{solver.destroy();}
 },["subgroups"]);
});
