/** Reproduce source starvation in Figure 7 at the UI's 1/60 s timestep.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-narrow-band-inflow.ts current --verify
 * --dt=0.041666666666666664 also checks the authored 24 Hz timestep.
 * Writes a centre-plane phi slice each second to /tmp/fluid-inflow-NAME.json.
 */
import {writeFileSync} from 'node:fs';
import assert from 'node:assert/strict';
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {inflowOutletCenter} from '../lib/core/inflow-boundary';
import {uniformNarrowBandMethod} from '../lib/methods/uniform/uniform-narrow-band-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {advanceUniform,readUniformFields,withUniformDevice} from '../tests/helpers/uniform-geometric';
import {readMixedBuffer} from '../tests/helpers/uniform-mixed-native-fields';
const name=process.argv[2]??'current';
assert.match(name,/^[a-z0-9-]+$/);
const rows:unknown[]=[];
const requestedDt=Number(process.argv.find(a=>a.startsWith('--dt='))?.slice(5)??1/60);
assert.ok(requestedDt>0&&requestedDt<=1/24);
const steps=Math.round(5/requestedDt),every=Math.round(1/requestedDt);
await withUniformDevice('NB inflow probe',async device=>{
 const scene=structuredClone(sceneDocument(getSceneDefinition('nbflip-figure-7-pour'))),dt=requestedDt,h=scene.voxelDomain.finestCellSize_m;
 scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
 const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,'balanced',{...uniformNarrowBandMethod.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 const inflow=scene.fluid.inflow!,outlet=inflowOutletCenter(inflow),origin=[outlet.x/h+64,outlet.y/h,outlet.z/h+64];
 const u=[inflow.velocity_m_s.x,inflow.velocity_m_s.y,0].map(v=>v/h),speed=Math.hypot(...u),dir=u.map(v=>v/speed);
 let core:number[]|undefined;
 try{
  for(let frame=1;frame<=steps;frame++){
   await advanceUniform(solver,frame*dt);
   if(frame%every!==0&&frame!==steps)continue;
   const fields=await readUniformFields(device,solver);
   const band=(solver as unknown as {mixedFrame:{narrowBandFlip:{activeParticles:GPUBuffer}}}).mixedFrame.narrowBandFlip;
   const particles=await readMixedBuffer(device,band.activeParticles),info=solver.narrowBandFlipInfo!;
   const bins=Array.from({length:8},()=>({n:0,u:0,v:0,min:Infinity,max:-Infinity}));
   for(let i=0;i<info.particles;i++){
    const d=[0,1,2].map(a=>particles[12*i+a]!-origin[a]!);const axial=d.reduce((n,v,a)=>n+v*dir[a]!,0);
    const radial=Math.hypot(...d.map((v,a)=>v-axial*dir[a]!));
    if(axial<0||axial>=32||radial>15)continue;
    const b=bins[Math.floor(axial/4)]!;b.n++;b.u+=particles[12*i+4]!/h;b.v+=particles[12*i+5]!/h;b.min=Math.min(b.min,particles[12*i+4]!/h);b.max=Math.max(b.max,particles[12*i+4]!/h);
   }
   const slices:number[]=[];for(let y=100;y<=240;y++)for(let x=0;x<=128;x++)slices.push(fields.vertex(x,y,64)/h);
   core??=slices.flatMap((value,i)=>value < -2&&100+Math.floor(i/129)>=170&&100+Math.floor(i/129)<=220?[i]:[]);
   assert.ok(core.length>1000,"sample the resolved core of the free stream");
   const coreLiquidFraction=core.filter(i=>slices[i]!<0).length/core.length;
   const row={coreLiquidFraction,frame,time:frame*dt,info,stats:await solver.readStats(),bins:bins.map(b=>({...b,u:b.u/b.n,v:b.v/b.n})),origin,u,slices};rows.push(row);
   console.log(JSON.stringify({...row,slices:undefined,stats:{volumeDrift:row.stats.volumeDrift}}));
   if(process.argv.includes('--verify'))assert.ok(coreLiquidFraction>0.99,`stream core stays liquid at ${frame*dt}s: ${coreLiquidFraction}`);
  }
 }finally{solver.destroy();}
});
writeFileSync(`/tmp/fluid-inflow-${name}.json`,JSON.stringify(rows));
