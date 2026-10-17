/** Real Metal/WebGPU + Box3D coupled smoke; no mocked physics or readback.
 * FLUID_COLD_COMPILE_NONCE: a fresh identifier to bypass native pipeline caches.
 * FLUID_INTERACTIVE_REPORT: output JSON path (defaults to /tmp/fluid-interactive-gpu.json).
 * FLUID_COMPARE_REPORT: assert exact body/packet equality with a saved report.
 */
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { managedGPUDevice, gpuCompilationManagerFor } from '../lib/core/gpu-compilation-manager';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { resolveMethodValues } from '../lib/core/method-contract';
import { uniformNarrowBandMethod as method } from '../lib/methods/uniform/uniform-narrow-band-method';
import { createEmptyScene } from '../lib/core/empty-scene';
import { interactionCatalog } from '../lib/interaction/installed';
import { interactionScene } from '../lib/interaction/adapters/scene';
import { SceneInteractionHost } from '../lib/interaction/runtime/host';
import type { FluidExchange } from '../lib/interaction/contracts';
const scene = createEmptyScene({ extents_m:{x:2,y:2,z:2},finestCellSize_m:.125 });
scene.scenery=undefined;scene.systems={fluid:true};scene.container.fillFraction=.5;scene.fluid.initialCondition='tank-fill';scene.fluid.surfaceTension_N_m=0;scene.numerics.fixedDt_s=1/60;scene.numerics.maxDt_s=1/60;
scene.interaction={schemaVersion:1,entities:[interactionCatalog.createEntity({id:'float',kind:'fluid.crate',transform:{position_m:{x:0,y:.9,z:0},orientation:{x:0,y:0,z:0,w:1}}})],connections:[]};
const dawn = await import(pathToFileURL(resolve('node_modules/webgpu/index.js')).href);Object.assign(globalThis,dawn.globals);
const gpu=createProcessRetainedDawnGPU(dawn,['backend=metal']),adapter=await gpu.requestAdapter();assert.ok(adapter);
const raw=await adapter.requestDevice({requiredFeatures:adapter.features.has('subgroups')?['subgroups']:[],requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
const errors:string[]=[];raw.addEventListener('uncapturederror',e=>{e.preventDefault();errors.push(e.error.message);console.error(e.error.message);});
// A unique entry-point name bypasses the native shader cache without changing
// any simulation work. Keep this opt-in and confined to the measurement tool.
const coldNonce=process.env.FLUID_COLD_COMPILE_NONCE;
const compilation:{module:string;entryPoint:string;constants:Record<string,number>;ms:number}[]=[];
if(coldNonce){
 assert.match(coldNonce,/^[a-zA-Z][a-zA-Z0-9_]*$/);
 const modules=new WeakMap<GPUShaderModule,{label:string;entries:Map<string,string>}>();
 const createModule=raw.createShaderModule.bind(raw),createPipeline=raw.createComputePipelineAsync.bind(raw);
 raw.createShaderModule=descriptor=>{
  const entries=new Map([...descriptor.code.matchAll(/@compute\b[^{};]*?\bfn\s+(\w+)/g)].map(match=>[match[1],`${match[1]}_${coldNonce}`]));
  const code=descriptor.code.replace(/\b\w+\b/g,name=>entries.get(name)??name);
  const module=createModule({...descriptor,code});modules.set(module,{label:descriptor.label??'',entries});return module;
 };
 raw.createComputePipelineAsync=async descriptor=>{
  const metadata=modules.get(descriptor.compute.module),entryPoint=descriptor.compute.entryPoint!;
  assert.ok(metadata);
  assert.ok(metadata.entries.has(entryPoint),`cold compile must rename ${entryPoint}`);
  const started=performance.now();
  const pipeline=await createPipeline({...descriptor,compute:{...descriptor.compute,entryPoint:metadata.entries.get(entryPoint)}});
  compilation.push({module:metadata.label,entryPoint,constants:descriptor.compute.constants??{},ms:performance.now()-started});return pipeline;
 };
}
const device=managedGPUDevice(raw,{requireWorkerRealm:false});
const values=resolveMethodValues(method,'balanced',{...method.appDefaults,timeStep:'scene',rigidCoupling:'on'});
const start=performance.now();
console.log('Preparing coupled fluid pipelines');
const manager=gpuCompilationManagerFor(device);
const progress=setInterval(()=>{const p=manager.snapshot();console.log(JSON.stringify({phase:'compile',queued:p.queued,active:p.active,completed:p.cacheStats.pipelines,current:p.progress?.label}));},30000);
let solver;
try { solver=await method.createSolverAsync!(device,interactionScene(scene),'balanced',values,undefined,p=>{if(p.phase==='warmup')console.log(p.label)}); }
finally { clearInterval(progress); }
const construction_ms=performance.now()-start;
assert.ok(solver.advanceExternalRigidStep);const rows:{tick:number;ms:number;packets:readonly FluidExchange[]}[]=[];
const host=new SceneInteractionHost(scene,{fluid:{step:async request=>{
 const t=performance.now(),packets=await solver.advanceExternalRigidStep!(request);
 rows.push({tick:request.tick,ms:performance.now()-t,packets});return packets;
}}});
try{await host.ready;console.log('Stepping native/GPU world');for(let i=0;i<30;i++){await host.advance(1/60);if(i%10===9)console.log(`Completed ${i+1} coupled ticks`);}
 assert.equal(host.getView().snapshot!.tick,30);assert.ok(Math.abs(host.getView().snapshot!.time_s-.5)<1e-8);assert.equal(errors.length,0,errors.join('\n'));
 const body=host.getView().snapshot!.bodies[0];assert.ok(body.transform.position_m.y>.7&&body.transform.position_m.y<1.3,'floating crate remains near the waterline');assert.ok(body.submergedFraction!>0);
 assert.ok(rows.reduce((sum,row)=>sum+row.packets[0].linearImpulse_N_s.y,0)>1,'water supplies upward pressure impulse');
 const mass=host.currentBodies()[0].mass_kg;
 const momentumError_N_s=Object.fromEntries((['x','y','z'] as const).map(axis=>{
  const impulse=rows.reduce((sum,row)=>sum+row.packets[0].linearImpulse_N_s[axis],0)+mass*scene.fluid.gravity_m_s2[axis]*.5;
  const error=mass*body.linearVelocity_m_s[axis]-impulse;
  assert.ok(Math.abs(error)<.001,`${axis} momentum matches gravity plus one application of the fluid impulses`);
  return [axis,error];
 }));
 let authoring: unknown;
 if(process.env.FLUID_CHECK_AUTHORING==='1') {
  let committed=scene;
  const before=host.getView(),crate=before.document.entities[0];
  const rebuilt=await host.dispatch({id:'gpu-rebuild',generation:before.generation,documentRevision:before.documentRevision,
    command:{kind:'configure',entityId:crate.id,value:{version:crate.version,config:{...crate.config,density:700}}}});
  assert.ok(rebuilt.accepted,JSON.stringify(rebuilt));assert.equal(rebuilt.impact,'rebuild');
  committed={...scene,interaction:host.getView().document};
  solver.applySceneUniforms!(interactionScene(committed));
  await host.advance(1/60);
  assert.equal(host.getView().generation,before.generation);
  assert.equal(host.getView().snapshot!.bodies[0].bodyGeneration,before.snapshot!.bodies[0].bodyGeneration+1);
  // Clear an existing floor corner, then restore it. Both hosts preflight the
  // same document, and neither the fluid nor Box3D timeline is reset.
  const hole={...committed,solidVoxels:[...committed.solidVoxels,{operation:'clear' as const,minimum:[0,-1,0] as [number,number,number],maximumExclusive:[2,0,2] as [number,number,number]}]};
  for(const next of [hole,committed]) {
    host.validateSolidEdit(next);solver.validateLiveSolidEdit!(interactionScene(next));
    solver.applySceneUniforms!(interactionScene(next));await host.replaceScene(next);await host.advance(1/60);
  }
  assert.equal(host.getView().snapshot!.tick,33);assert.equal(host.getView().readiness.state,'ready');
  assert.equal(errors.length,0,errors.join('\n'));
  authoring={bodyRebuild:rebuilt,solidEdits:2,snapshot:host.getView().snapshot,rows:rows.slice(30)};
 }
 const info=adapter.info;
 const baselineRows=rows.slice(0,30);
 const costs=baselineRows.map(row=>row.ms).sort((a,b)=>a-b);
 const timing={construction_ms,steps_ms:costs.reduce((sum,ms)=>sum+ms,0),median_ms:(costs[14]+costs[15])/2,p95_ms:costs[28]};
 const result={authoring,backend:'Metal/Dawn',adapter:{vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description},coldNonce,compilation,ticks:30,dt_s:1/60,elapsed_ms:performance.now()-start,timing,momentumError_N_s,body,fluid:solver.info,rows:baselineRows,errors};
 if(process.env.FLUID_COMPARE_REPORT){
  const before=JSON.parse(readFileSync(process.env.FLUID_COMPARE_REPORT,'utf8')) as {body:unknown;rows:{packets:unknown}[]};
  assert.deepEqual(body,before.body,'final body state must match the reference exactly');
  assert.deepEqual(baselineRows.map(row=>row.packets),before.rows.slice(0,30).map(row=>row.packets),'every fluid exchange packet must match the reference exactly');
 }
 const report=process.env.FLUID_INTERACTIVE_REPORT??'/tmp/fluid-interactive-gpu.json';writeFileSync(report,JSON.stringify(result,null,2));console.log(JSON.stringify({ticks:30,body,timing,momentumError_N_s,report}));
}finally{await host.dispose();solver.destroy();raw.destroy();}
