/**
 * Isolated ocean coarse-first publication attribution and payload receipts.
 * Run with WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js and
 * FLUID_GPU_ISOLATE_PASS_LABELS=1 to split publication from surface proofs.
 * OCEAN_OUT saves the JSON; OCEAN_CAPTURE_PREFIX also saves terminal buffers.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { GPUPassTimestampRecorder } from '../lib/core/performance-trace';
import { resolveMethodValues } from '../lib/core/method-contract';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { adaptiveMassMethod } from '../lib/methods/adaptive-volume/method';
import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver';

const steps=Number(process.env.OCEAN_STEPS ?? 10);
assert.ok(Number.isSafeInteger(steps)&&steps>0);
assert.ok(process.env.WEBGPU_NODE_MODULE, "Set WEBGPU_NODE_MODULE to native Dawn");
await acquireWebGPUExclusiveLock('dawn-probe', 'ocean coarse-first presentation');
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({requiredFeatures:['timestamp-query'], requiredLimits:requiredFluidDeviceLimits(adapter.limits)});
  const errors: string[] = [];
  device!.addEventListener('uncapturederror', e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  const sourceHash=createHash('sha256').update(await readFile(new URL('../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts',import.meta.url))).digest('hex');
  const scene = sceneDocument(getSceneDefinition('ocean-seiche'));
  const values = resolveMethodValues(adaptiveMassMethod, 'balanced', {selectorMode:'coarse-first',timeStep:'paper'});
  solver = await adaptiveMassMethod.createSolverAsync!(device!, scene, 'balanced', values, undefined, phase => console.error(phase)) as WebGPUAdaptiveMassSolver;
  console.error("solver created");
  await solver.waitForSimulationReady();
  console.error("simulation ready");
  const createEncoder = device!.createCommandEncoder.bind(device);
  let recorder: GPUPassTimestampRecorder | undefined;
  device!.createCommandEncoder = descriptor => {
    const encoder = createEncoder(descriptor);
    if (!recorder) return encoder;
    const wrapped = recorder.instrument(encoder);
    // Only instrument publication to avoid perturbing the pressure schedule.
    return new Proxy(encoder, { get(target, property) {
      if(property === 'beginComputePass') return (desc?: GPUComputePassDescriptor) =>
        desc?.label?.includes('presentation') ? wrapped.beginComputePass(desc) : target.beginComputePass(desc);
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    }});
  };
  const samples = [];
  const bufferBytes = async (source: GPUBuffer, size: number) => {
    const readback=device!.createBuffer({size, usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    const encoder=createEncoder(); encoder.copyBufferToBuffer(source,0,readback,0,size);
    device!.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
    const bytes=new Uint8Array(readback.getMappedRange()).slice();
    readback.unmap(); readback.destroy(); return bytes;
  };
  for(let step=1;step<=steps;step++) {
    recorder = new GPUPassTimestampRecorder(device!, 256);
    const started = performance.now();
    while(!solver.advanceTo(step/30, [])) {
      assert.deepEqual(errors, []);
      assert.ok(performance.now()-started<60_000, `step ${step} stalled: ${JSON.stringify(solver.info)}`);
      await new Promise(setImmediate);
    }
    await solver.waitForTopologyReady();
    const encoder = createEncoder(); recorder.resolve(encoder); device!.queue.submit([encoder.finish()]);
    await device!.queue.onSubmittedWorkDone();
    const timing = await recorder.read(); recorder.destroy(); recorder=undefined;
    const wall_ms=performance.now()-started;
    const publication=solver.globalFineLevelSetSource;
    const payload=await bufferBytes(publication.samples,publication.plan.payloadCapacityBytes);
    const metadata=await bufferBytes(publication.metadata,publication.metadata.size);
    const payloadHash=createHash('sha256').update(payload).digest('hex');
    if(process.env.OCEAN_CAPTURE_PREFIX && step===steps) {
      await writeFile(`${process.env.OCEAN_CAPTURE_PREFIX}.samples.bin`,payload);
      await writeFile(`${process.env.OCEAN_CAPTURE_PREFIX}.metadata.bin`,metadata);
    }
    const policy=await solver.readGPUActivityPolicy();
    const census:Record<string,number>={};
    for(const brick of policy.bricks) if(brick.active) {
      const key=`span${brick.spanBricks}/B${brick.acceptedResolution}`;
      census[key]=(census[key]??0)+1;
    }
    const sample = {step, wall_ms, timing, payloadHash, census}; samples.push(sample);
    console.log(JSON.stringify(sample));
    assert.deepEqual(errors, []);
  }
  const fields=await solver.readDiagnosticFields(true);
  const densityHash=createHash('sha256').update(new Uint8Array(fields.density.buffer)).digest('hex');
  await writeFile(process.env.OCEAN_OUT ?? '/tmp/ocean-coarse-presentation.json', JSON.stringify({sourceHash,samples,densityHash,errors},null,2));
} finally { solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock(); }
