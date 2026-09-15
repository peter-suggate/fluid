/** Native Figure 7 free-fall diagnostics: accepted rungs, volume and shape. */
import assert from "node:assert/strict";
import { writeFileSync, mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sparseCM12DawnDefaultOptions } from "../lib/harness/sparse-cm12-dawn-defaults";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { readPublishedCM12Field } from "./sparse-cm12-published-field";
const arg = (key: string, fallback: string) => process.argv.find(v => v.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const sceneId = arg("scene", "cm12-figure-7");
const steps = Number(arg("steps", "30"));
const verifyHealth = process.argv.includes("--verify-health");
const verifyCoarseFloor = process.argv.includes("--verify-coarse-floor");
const output = arg("output", "artifacts/level-set-volume/figure7-deformation.json");
assert.ok(Number.isInteger(steps) && steps >= 0);
await acquireWebGPUExclusiveLock("dawn-probe", "figure7 free fall");
let device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
const report: Record<string, unknown> = { sceneId, steps, dt: 1 / 30, checkpoints: [] };
const errors: string[] = [];
try {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal", "enable-dawn-features=disable_blob_cache"]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); });
  const scene = getScenePreset(sceneId).create();
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(device, scene, "balanced",
    undefined, { ...sparseCM12DawnDefaultOptions(),
      surfaceSharpeningEnabled: arg("sharpening", "on") !== "off" }, () => {});
  await solver.waitForSimulationReady();
  report.dimensions = [solver.info.nx, solver.info.ny, solver.info.nz];
  const frameTimes: number[] = [];
  let initialMass = 0, initialCentreY = 0;
  for (let step = 0; step <= steps; step++) {
    if (step > 0) {
      const started = performance.now();
      while (!solver.advanceTo(step / 30, [])) await new Promise<void>(setImmediate);
      await solver.awaitFrameCompletion(); await device.queue.onSubmittedWorkDone();
      frameTimes.push(performance.now() - started);
    }
    if (!verifyCoarseFloor && !verifyHealth && !process.argv.includes("--every-frame") && ![0, 1, 2, 3, 4, 6, 10, 15, 20, 25, 30, 35, 40, 45, 60, steps].includes(step)) continue;
    const diagnostics = await Promise.all([
      solver.readDiagnosticFields(true), readPublishedCM12Field(device, solver),
      solver.readAcceptedGeometricVolumeQA(), solver.readGeometricVolumeTransportReceiptQA(),
    ]);
    const [fields, phi, volume] = diagnostics;
    const transport: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGeometricVolumeTransportReceiptQA"]>> = diagnostics[3];
    let phiNegativeFineCells = 0;
    let airSideVolume = 0, deepAirVolume = 0, fractionalFineCells = 0, densitySum = 0;
    for (let i = 0; i < fields.density.length; i++) {
      const density = fields.density[i]!; densitySum += density;
      if (phi.values[i]! < 0) phiNegativeFineCells++;
      if (density > 1e-5 && density < 1 - 1e-5) fractionalFineCells++;
      if (phi.values[i]! > 0) airSideVolume += density;
      if (phi.values[i]! > scene.voxelDomain.finestCellSize_m) deepAirVolume += density;
    }
    const activity: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>> = await solver.readGPUActivityPolicy();
    const phiQA = process.argv.includes("--phi-debug") && step >= 18
      ? await solver.readAdaptiveLevelSetQA(true) : undefined;
    if (phiQA?.vertices) phiQA.vertices = phiQA.vertices.filter(v => v.positionFine[0] >= 32 && v.positionFine[0] <= 42 && v.positionFine[1] >= 32 && v.positionFine[1] <= 48 && v.positionFine[2] >= 52 && v.positionFine[2] <= 60);
    const velocityY={minimum:Infinity,maximum:-Infinity,weighted:0,weight:0};
    for(let i=0;i<fields.density.length;i++)if(fields.density[i]!>0.5){
      const v=fields.velocity[4*i+1]!;velocityY.minimum=Math.min(velocityY.minimum,v);velocityY.maximum=Math.max(velocityY.maximum,v);
      velocityY.weighted+=v*fields.density[i]!;velocityY.weight+=fields.density[i]!;
    }
    velocityY.weighted/=velocityY.weight||1;
    const nx=solver.info.nx, ny=solver.info.ny, nz=solver.info.nz;
    let lowestLiquidY = Infinity;
    let mass=0, excess=0;const centroid=[0,0,0],second=[0,0,0], excessCentroid=[0,0,0];
    for(let i=0;i<fields.density.length;i++){
      const rho=fields.density[i]!;const q=[i%nx+.5,Math.floor(i/nx)%ny+.5,Math.floor(i/(nx*ny))+.5];
      if (rho > 0.5) lowestLiquidY = Math.min(lowestLiquidY, q[1]! - 0.5);
      mass+=rho;const e=Math.max(0,rho-1);excess+=e;
      for(let a=0;a<3;a++){centroid[a]!+=rho*q[a]!;second[a]!+=rho*q[a]!**2;excessCentroid[a]!+=e*q[a]!;}
    }
    for(let a=0;a<3;a++){centroid[a]!/=mass;second[a]=second[a]!/mass-centroid[a]!**2;excessCentroid[a]!/=excess||1;}
    const interior = { samples: 0, missingPhi: 0, airPhi: 0, lowDensity: 0, minimumDensity: Infinity, badPages: {} as Record<string, number> };
    for (let i = 0; i < fields.density.length; i++) {
      const q = [i % nx + .5, Math.floor(i / nx) % ny + .5, Math.floor(i / (nx * ny)) + .5];
      if (q.reduce((sum, value, axis) => sum + (value - centroid[axis]!) ** 2, 0) >= 12 ** 2) continue;
      interior.samples++;
      const missing = !Number.isFinite(phi.values[i]);
      const air = phi.values[i]! >= 0;
      const low = fields.density[i]! < .9;
      interior.missingPhi += Number(missing); interior.airPhi += Number(air); interior.lowDensity += Number(low);
      interior.minimumDensity = Math.min(interior.minimumDensity, fields.density[i]!);
      if (missing || air || low) { const key = q.map(v => Math.floor(v / 8)).join(","); interior.badPages[key] = (interior.badPages[key] ?? 0) + 1; }
    }
    const pageVolumes: Array<{ leaf: number; q: readonly number[]; mass: number; maximum: number; nonzero: number }> = [];
    if (process.argv.includes("--growth")) for (const b of activity.bricks) {
      if (!b.active) continue;
      let pageMass = 0, maximum = 0, nonzero = 0;
      for (let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++) {
        const q = [8*b.coordinate[0]+x,8*b.coordinate[1]+y,8*b.coordinate[2]+z];
        if(q.some((v,axis)=>v<0||v>=[nx,ny,nz][axis]!))continue;
        const rho=fields.density[q[0]!+nx*(q[1]!+ny*q[2]!)]!;
        pageMass+=rho;maximum=Math.max(maximum,rho);nonzero+=Number(rho!==0);
      }
      pageVolumes.push({leaf:b.leafId,q:b.coordinate,mass:pageMass,maximum,nonzero});
    }
    if (step === 0) { initialMass = mass; initialCentreY = centroid[1]!; }
    (report.checkpoints as unknown[]).push({ step, pageVolumes, growth: process.argv.includes("--growth") ? await solver.readWorldGrowthReceiptQA() : undefined, interior, lowestLiquidY, phiQA, velocityY, mass, centroid, variance:second,excess,excessCentroid,activity,volume, effects: activity.commitFailed ? await solver.readCandidateEffectsTransactionQA() : undefined, coupling: transport.coupling,
      transportFault: transport.fault, outflowFineCells3: transport.outflowFineCells3, phiNegativeFineCells, airSideVolume, deepAirVolume, fractionalFineCells, densitySum });
    const accountedMass = mass + transport.coupling.cumulativeResidueDeletedVolumeFine3;
    if (verifyHealth) {
      assert.equal(activity.commitFailed, false, `frame ${step}: rejected topology`);
      assert.equal(activity.faultFlags, 0, `frame ${step}: activity fault`);
      assert.equal(transport.fault, 0, `frame ${step}: transport fault`);
      assert.equal(transport.coupling.edgeOverflowCount, 0, `frame ${step}: coupling overflow`);
      assert.ok(Number.isFinite(mass) && fields.density.every(value => Number.isFinite(value) && value >= 0),
        `frame ${step}: invalid accepted density`);
    }
    if (verifyCoarseFloor) {
      assert.equal(sceneId, "cm12-figure-7");
      assert.equal(activity.commitFailed, false, `frame ${step}: rejected topology`);
      assert.equal(interior.missingPhi + interior.airPhi + interior.lowDensity, 0, `frame ${step}: interior holes ${JSON.stringify(interior)}`);
      assert.equal(activity.faultFlags, 0, `frame ${step}: activity fault`);
      assert.ok(Math.abs(accountedMass / initialMass - 1) < 1e-5, `frame ${step}: unexplained mass drift after recorded residue deletion`);
      assert.ok(activity.bricks.some(b => b.active && b.acceptedResolution === 2
        && fields.density[(8*b.coordinate[0]+4) + nx*((8*b.coordinate[1]+4) + ny*(8*b.coordinate[2]+4))]! > 0.9
        && b.coordinate.every((v, axis) => Math.abs(8 * v + 4 - centroid[axis]!) < 7)),
        `frame ${step}: no four-spacing liquid cell near centre`);
      if (step === steps) {
        assert.ok(step >= 25 && lowestLiquidY <= 1, `test stopped before floor contact: frame ${step}, bottom ${lowestLiquidY}`);
        report.dynamicTransport = await solver.readDynamicTransportPacketsQA();
      }
    }
    if ((process.argv.includes("--verify-fall") && step <= 20) || (verifyCoarseFloor && step <= 24)) {
      assert.equal(sceneId, "cm12-figure-7");
      assert.ok(Math.abs(accountedMass / initialMass - 1) < 1e-5, `frame ${step}: unexplained mass drift after recorded residue deletion`);
      // Semi-implicit Euler gravity: displacement is g*dt²*n*(n+1)/2.
      const expectedY = initialCentreY + scene.fluid.gravity_m_s2.y
        * (1 / 30) ** 2 * step * (step + 1) / 2 / scene.voxelDomain.finestCellSize_m;
      assert.ok(Math.abs(centroid[1]! - expectedY) < 0.05,
        `frame ${step}: centre ${centroid[1]}, expected ${expectedY}`);
      assert.ok(Math.max(...second) / Math.min(...second) < 1.05,
        `frame ${step}: distorted free-fall shape ${second}`);
      if (step <= 6) {
        assert.ok(excess < 1e-3, `frame ${step}: early excess ${excess}`);
        assert.ok(Math.abs(velocityY.maximum - velocityY.minimum) < 1e-4,
          `frame ${step}: nonuniform falling velocity`);
        assert.ok(activity.bricks.some(b => b.active && b.acceptedResolution === 2
          && b.coordinate.every((v, axis) => Math.abs(8 * v + 4 - centroid[axis]!) < 7)),
          `frame ${step}: no four-spacing core cell`);
      }
    }
  }
  report.frameTimesMs = frameTimes;
  assert.deepEqual(errors, [], "GPU validation errors");
  report.completed = true;
} catch (e) {
  report.completed = false; report.error = String(e); process.exitCode = 1;
  if (solver && process.argv.includes("--failure-debug")) {
    report.failureActivity = await solver.readGPUActivityPolicy();
    report.failureTransport = await solver.readGeometricVolumeTransportReceiptQA();
    report.failureGrowth = await solver.readWorldGrowthReceiptQA();
    const owner = /owner=(\d+)/.exec(String(e));
    if (owner && String(e).includes("GEOMETRIC_VOLUME_TRANSPORT")) {
      report.failureCell = await solver.readAcceptedGeometricCellRowsQA(Number(owner[1]));
    }
  }
} finally {
  report.validationErrors = errors;
  solver?.destroy(); device?.destroy();
  await releaseWebGPUExclusiveLock();
  mkdirSync(dirname(output), { recursive: true }); writeFileSync(output, JSON.stringify(report, null, 2) + "\n");
  console.log(JSON.stringify({ output, completed: report.completed, error: report.error }));
}
