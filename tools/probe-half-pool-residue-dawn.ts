import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { resolveMethodValues } from '../lib/core/method-contract';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { adaptiveMassMethod } from '../lib/methods/adaptive-mass/method';
import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
await acquireWebGPUExclusiveLock('dawn-test', 'half-pool-residue');
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis, dawn.globals);
const gpu: GPU = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]);
try {
 const adapter = await gpu.requestAdapter(); assert.ok(adapter);
 device = await adapter.requestDevice({requiredLimits: requiredFluidDeviceLimits(adapter.limits)});
 const errors: string[] = [];
 device.addEventListener('uncapturederror', event => {event.preventDefault();errors.push(event.error.message);});
 const scene = sceneDocument(getSceneDefinition('coarse-first-pool-impact-half'));
 const values = resolveMethodValues(adaptiveMassMethod, 'balanced', {});
 solver = await adaptiveMassMethod.createSolverAsync!(device, scene, 'balanced', values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
 await solver.waitForSimulationReady();
 let initialMass=0;
 const steps = Number(process.env.RESIDUE_STEPS ?? 120);
 for(let step=0; step<=steps; step++) {
  if(step) { while(!solver.advanceTo(step / 30, [])) await new Promise(setImmediate);
   await solver.waitForTopologyReady(); await solver.assertSimulationHealthy(); }
  if(step%15===0) {
   const f = await solver.readDiagnosticFields(true);
   const nx=solver.info.nx, ny=solver.info.ny;
   let mass=0, upperMass=0, moment=0, speed=0, maximum=0, cells=0;
   for(let i=0;i<f.density.length;i++) {const rho=f.density[i];mass+=rho;
    const y=Math.floor(i/nx)%ny;
    if(y>=32 && rho>0) {upperMass+=rho;moment+=rho*(y+.5);speed+=rho*f.velocity[4*i+1];maximum=Math.max(maximum,rho);cells++;}}
   console.log(JSON.stringify({step,mass,upperMass,centerY:moment/upperMass,vy:speed/upperMass,maximum,cells}));
   if(step===0) initialMass=mass;
   if(process.env.RESIDUE_VERIFY==='1') {
    assert.ok(Math.abs(mass/initialMass-1)<.001, 'falling residue must retain total mass');
    if(step===30) assert.ok(upperMass<1e-6, 'the initial drop must clear the upper air after one second');
   }
  }
 }
 assert.deepEqual(errors, []);
} finally {solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock(); assert.ok(gpu);}
