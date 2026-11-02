import {test} from "node:test";
import assert from "node:assert/strict";
import {createNbflipAnalyticScene, NBFLIP_ANALYTIC_DT_S} from "../lib/core/nbflip-analytic-scenes";
import {advanceUniform, createUniformSolver, withUniformDevice} from "./helpers/uniform-geometric";
import type {UniformNarrowBandFlip} from "../lib/methods/uniform/uniform-narrow-band-flip";

// A block translating in zero gravity at (0.75, 0.75, 0) m/s keeps that
// velocity: FLIP's delta is zero for a uniform field. The initial band's
// seeds (seedCell's emit) once skipped the nbSeedFirst mark, so flipDelta
// "swept" read their before.xyz (their velocity) as a delta and G2P doubled
// it: every sample at 1.46 m/s after one step, the block 15.7 h off in 20.
test("NB-FLIP seeds keep a translating block's velocity (swept delta)",{skip:!process.env.WEBGPU_NODE_MODULE},async()=>{
 await withUniformDevice("NB-FLIP Galilean seeds",async device=>{
  const solver=await createUniformSolver(device,createNbflipAnalyticScene("translation"),{particleTransfer:"flip",flipDelta:"swept"});
  try{
   const flip=(solver as unknown as {mixedFrame:{narrowBandFlip:UniformNarrowBandFlip}}).mixedFrame.narrowBandFlip;
   for(let frame=1;frame<=4;frame++)await advanceUniform(solver,frame*NBFLIP_ANALYTIC_DT_S);
   const stride=flip.particleStride,bytes=flip.activeParticles.size;
   const read=device.createBuffer({size:bytes,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
   const encoder=device.createCommandEncoder();encoder.copyBufferToBuffer(flip.activeParticles,0,read,0,bytes);device.queue.submit([encoder.finish()]);
   await read.mapAsync(GPUMapMode.READ);const a=new Float32Array(read.getMappedRange().slice(0));read.unmap();read.destroy();
   let n=0;const sum=[0,0,0];let worst=0;
   for(let o=0;o+stride<=a.length;o+=stride){
    // Live: heat (position.w) > 0 at a real position; ballistic samples carry no grid velocity.
    if(!(a[o]!>0&&a[o+3]!>0))continue;
    n++;for(let c=0;c<3;c++)sum[c]+=a[o+4+c]!;
    worst=Math.max(worst,Math.hypot(a[o+4]!-0.75,a[o+5]!-0.75,a[o+6]!));
   }
   assert.ok(n>1000,`the translating block seeds a band (${n} samples)`);
   const mean=sum.map(s=>s/n);
   for(const [c,expected] of [0.75,0.75,0].entries())assert.ok(Math.abs(mean[c]!-expected)<=0.01,`mean sample velocity ${mean.map(v=>v.toFixed(4))} against (0.75, 0.75, 0)`);
   assert.ok(worst<=0.25,`no sample departs the uniform field by more than 0.25 m/s (worst ${worst.toFixed(3)})`);
  }finally{solver.destroy();}
 },["subgroups"]);
});
