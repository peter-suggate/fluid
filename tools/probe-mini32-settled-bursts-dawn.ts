/** Reproduce late mini32 density bursts with the UI scene and min/max1 region.
 * BURST_AUDIT_STEPS=300,301 records unmodified production stage boundaries.
 * BURST_OVERRIDES supplies runtime control arms; no solver source is patched.
 */
import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { sceneDocument } from '../lib/core/scene-definition';
import { getSceneDefinition } from '../lib/core/scenes';
import { resolveMethodValues } from '../lib/core/method-contract';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { adaptiveMassMethod } from '../lib/methods/adaptive-volume/method';
import type { WebGPUAdaptiveMassSolver } from '../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from '../lib/harness/webgpu-smoke-isolation';
import { axisArtifactStageAudit } from './axis-artifact-stage-audit';
const steps = Number(process.env.BURST_STEPS ?? 330);
const from = Number(process.env.BURST_FROM ?? 240);
const output = process.env.BURST_OUTPUT ?? 'artifacts/mini32-settled-bursts/base';
const audits = new Set((process.env.BURST_AUDIT_STEPS ?? '').split(',').map(Number));
await acquireWebGPUExclusiveLock('dawn-probe', 'mini32-settled-bursts');
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ['backend=metal']);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', e => { e.preventDefault(); errors.push(e.error.message); });
  // Diagnostic only: this closed-tank ablation exempts zero-aperture rows.
  // It does not implement ceiling separation or repair persistent density drift.
  if(process.env.BURST_WALL_GUARD === '1' || process.env.BURST_LEGACY_WALL_GUARD === '1') {
    const createShaderModule = device.createShaderModule.bind(device);
    device.createShaderModule = descriptor => {
      const code = descriptor.code.replace(/fn pressureCellSubmerged\(id:u32\)->bool\{[\s\S]*?\n\}/, body => {
        const start = body.indexOf('    if(count<2u)');
        const end = body.indexOf('    for(var term=', start);
        assert.ok(start >= 0 && end > start);
        const replacement = process.env.BURST_LEGACY_WALL_GUARD === '1'
          ? 'if(count<2u){return false;}' :
          `if(count<2u){
            let voxelBase=p.solidOffsets.y+((3u*p.counts.y+3u)&0xfffffffcu);
            let voxelOpen=select(1.0,state[voxelBase+row],(p.solidOffsets.z&4u)!=0u);
            let closed=(p.solidOffsets.z&1u)!=0u&&state[p.solidOffsets.y+3u*row]*voxelOpen<=1e-8;
            if(closed){continue;}return false;}`;
        return body.slice(0, start) + '    ' + replacement + '\n' + body.slice(end);
      });
      return createShaderModule({...descriptor,code});
    };
  }
  const scene = sceneDocument(getSceneDefinition('minimal-power-dam-break-32'));
  scene.fluid.refinementRegions = [{ id: 'whole-tank-minmax1', rule: 'minimum-cell-size',
    minimumCellSize_cells: 1, maximumCellSize_cells: 1,
    min_m: {x:-scene.container.width_m/2,y:0,z:-scene.container.depth_m/2},
    max_m: {x:scene.container.width_m/2,y:scene.container.height_m,z:scene.container.depth_m/2}}];
  const values = resolveMethodValues(adaptiveMassMethod, 'balanced', {
    timeStep:'paper', selectorMode:'surface', ...JSON.parse(process.env.BURST_OVERRIDES ?? '{}') });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, 'balanced', values, undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  await mkdir(output, {recursive:true});
  await writeFile(`${output}/configuration.json`, JSON.stringify({scene,values,steps,from,wallGuard:process.env.BURST_WALL_GUARD === '1',legacyWallGuard:process.env.BURST_LEGACY_WALL_GUARD === '1',switchStep:process.env.BURST_SWITCH_STEP,switchOverrides:process.env.BURST_SWITCH_OVERRIDES},null,2));
  const trace = [];
  let runtimeValues = values;
  let previous: Float32Array | undefined;
  for(let step=0;step<=steps;step++) {
    if(step === Number(process.env.BURST_SWITCH_STEP)) {
      runtimeValues = {...values, ...JSON.parse(process.env.BURST_SWITCH_OVERRIDES ?? '{}')};
      solver.applyRuntimeValues(runtimeValues);
    }
    if(step) {
      const finish = audits.has(step) ? await axisArtifactStageAudit(device,solver,output,step,runtimeValues.gammaDiffusion !== 'off') : undefined;
      while(!solver.advanceTo(step/30,[])) await new Promise(setImmediate);
      await finish?.();
    }
    if(step < from && step%30 !== 0) continue;
    const fields = await solver.readDiagnosticFields(true);
    const rho=fields.density;
    let mass=0,max=-Infinity,min=Infinity,maxAt=0,delta=0,deltaAt=0,excess=0;
    for(let i=0;i<rho.length;i++) {
      assert.ok(Number.isFinite(rho[i])); mass+=rho[i];
      if(fields.solidOpenFraction[i]<.99) continue;
      if(rho[i]>max){max=rho[i];maxAt=i;} min=Math.min(min,rho[i]);
      excess+=Math.max(0,rho[i]-1);
      if(previous && rho[i]-previous[i]>delta){delta=rho[i]-previous[i];deltaAt=i;}
    }
    const xyz=(i:number)=>[i%32,Math.floor(i/32)%32,Math.floor(i/1024)];
    const stats=await solver.readStats();
    const row={step,time:solver.info.simulatedTime_s,mass,min,max,maxAt:xyz(maxAt),delta,deltaAt:xyz(deltaAt),excess,
      pressureResidual:stats.pressureRelativeResidual,pressureIterations:stats.pressureIterationsExecuted};
    trace.push(row);
    if(step>=from) for(const name of ['density','gamma','velocity','pressure','divergence'] as const)
      await writeFile(`${output}/${step}-${name}.bin`,new Uint8Array(fields[name].buffer));
    if(step===0) await writeFile(`${output}/solid.bin`,new Uint8Array(fields.solidOpenFraction.buffer));
    if(step%30===0 || delta>.1) console.log(JSON.stringify(row));
    previous=rho;
  }
  const activity=await solver.readGPUActivityPolicy();
  assert.ok(activity.bricks.every(b=>!b.active || (b.acceptedResolution===8 && b.spanBricks===1)));
  assert.deepEqual(errors,[]);
  await writeFile(`${output}/trace.json`,JSON.stringify(trace,null,2));
} finally { solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock(); }
