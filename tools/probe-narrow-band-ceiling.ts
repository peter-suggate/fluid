/** Per-stage volume by height, including wet ceiling corners.
 * node --import tsx tools/probe-narrow-band-ceiling.ts [scene] [steps] [output]
 * Uses scene defaults with a 1/60 s step. Diagnostics never modify fluid state.
 */
import assert from 'node:assert/strict';
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { sceneDocument } from '../lib/core/scene-definition';
import { findSceneDefinition } from '../lib/core/scenes';
import { uniformNarrowBandMethod as method } from '../lib/methods/uniform/uniform-narrow-band-method';
import type { WebGPUUniformReferenceSolver } from '../lib/methods/uniform/webgpu-uniform-reference';
import { narrowBandCeilingProbe } from './narrow-band-ceiling-probe';
const sceneId=process.argv[2]??'minimal-power-dam-break-64',steps=Number(process.argv[3]??300),output=process.argv[4]??'/tmp/fluid-ceiling.json';
assert.ok(Number.isInteger(steps)&&steps>0&&steps<=2400);
const definition=findSceneDefinition(sceneId);assert.ok(definition,`Unknown scene: ${sceneId}`);
const scene=structuredClone(sceneDocument(definition)),dt=1/60;
scene.numerics={...scene.numerics,fixedDt_s:dt,maxDt_s:dt};
const dawn=await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const adapter=await createProcessRetainedDawnGPU(dawn,['backend=metal']).requestAdapter();assert.ok(adapter);
const device=managedGPUDevice(await adapter.requestDevice({requiredLimits:requiredFluidDeviceLimits(adapter.limits)}),{requireWorkerRealm:false});
const errors:string[]=[],rows:unknown[]=[];
device.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
let solver:WebGPUUniformReferenceSolver|undefined,probe:Awaited<ReturnType<typeof narrowBandCeilingProbe>>|undefined;
try{
 solver=await method.createSolverAsync!(device,scene,'balanced',{...method.appDefaults,timeStep:'scene'},undefined,()=>{}) as WebGPUUniformReferenceSolver;
 probe=await narrowBandCeilingProbe(device,solver);
 for(let step=1;step<=steps;step++){
  solver.advanceTo(step*dt,[]);await solver.awaitFrameCompletion();
  assert.equal(solver.info.simulationPipelineError,undefined);
  const stages=await probe.read();const row={step,time:step*dt,stages};rows.push(row);
  if(step%30===0)console.log(JSON.stringify({step,time:row.time,stages:Object.fromEntries(Object.entries(stages).map(([name,s])=>[name,{volume:s.volume,wetCeilingCorners:s.wetCeilingCorners}]))}));
 }
 assert.deepEqual(errors,[]);
}finally{
 writeFileSync(output,JSON.stringify({sceneId,dt,rows,errors},null,2)+'\n');
 probe?.destroy();solver?.destroy();await device.queue.onSubmittedWorkDone();device.destroy();
}
