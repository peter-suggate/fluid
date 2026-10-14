/** Behaviour A/B of the pressure band's rows in a buffer of their own (the
 * QA arm) against the stage scratch (the product). Two repeats per arm on the
 * 64^3 mini dam, one second at 1/60 s: NB-FLIP runs vary run to run, so the
 * arms are compared by range. Any WebGPU validation error fails the probe.
 * WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js node --import tsx tools/probe-band-rows-shared.ts
 */
import {sceneDocument} from '../lib/core/scene-definition';
import {getSceneDefinition} from '../lib/core/scenes';
import {setUniformMixedBandRowsRetainedForQA} from '../lib/methods/uniform/uniform-mixed-frame';
import {uniformNarrowBandMethod} from '../lib/methods/uniform/uniform-narrow-band-method';
import type {WebGPUUniformReferenceSolver} from '../lib/methods/uniform/webgpu-uniform-reference';
import {advanceUniform,withUniformDevice} from '../tests/helpers/uniform-geometric';
const sceneId=process.argv.find(a=>a.startsWith('--scene='))?.slice(8)??'minimal-power-dam-break-64';
const dt=1/60,steps=60;
await withUniformDevice('Band rows storage probe',async device=>{
 for(const arm of ['own','shared','own','shared'] as const){
  setUniformMixedBandRowsRetainedForQA(arm==='own');
  const scene=structuredClone(sceneDocument(getSceneDefinition(sceneId)));
  scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
  const started=performance.now();
  const solver=await uniformNarrowBandMethod.createSolverAsync!(device,scene,'balanced',{...uniformNarrowBandMethod.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
  const created=performance.now()-started;
  try{
   for(let frame=1;frame<=steps;frame++){
    await advanceUniform(solver,frame*dt);
    if(frame%20!==0)continue;
    const stats=await solver.readStats(),info=solver.narrowBandFlipInfo!,execution=solver.info;
    console.log(JSON.stringify({arm,frame,create_ms:Math.round(created),allocatedMiB:+(execution.allocatedBytes/1048576).toFixed(1),particles:info.particles,
     volumeDrift:stats.volumeDrift,fineTiles:execution.uniformMixedFineTiles,residual:execution.pressureResidual}));
   }
  }finally{solver.destroy();}
 }
});
setUniformMixedBandRowsRetainedForQA(false);
