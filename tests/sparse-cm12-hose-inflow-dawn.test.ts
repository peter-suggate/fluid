import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { inflowOutletCenter } from "../lib/core/inflow-boundary";
import { createPaperScenario } from "../lib/core/paper-scenarios";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { readFineUpperSurfaceField } from
  "../lib/harness/webgpu-smoke-readbacks";
import {
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG as FPP_FLAG,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER as FPP_HEADER,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const dawnModule = process.env.WEBGPU_NODE_MODULE;
const dawnTest = dawnModule ? test : test.skip;

async function readGPUWords(
  device: GPUDevice,
  binding: GPUBufferBinding,
  wordCount: number,
): Promise<Uint32Array> {
  const readback = device.createBuffer({
    label: "Sparse CM12 hose control readback",
    size: 4 * wordCount,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(binding.buffer, binding.offset ?? 0, readback, 0,
      4 * wordCount);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function readCompactHoseStats(
  device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver,
  tank: ReturnType<typeof createPaperScenario>["container"],
  spacing: readonly [number, number, number],
  outlet: ReturnType<typeof inflowOutletCenter>,
  radius_m: number,
  poolSurface_m: number,
): Promise<{ airborneSamples: number; axialFront_m: number }> {
  const source = solver.globalFineLevelSetSource;
  const capacity = source.plan.maximumResidentBricks;
  const [worklist, metadata, samples] = await Promise.all([
    readGPUWords(device, { buffer: source.worklist }, 7 + capacity),
    readGPUWords(device, { buffer: source.metadata }, 4 * capacity),
    readGPUWords(device, { buffer: source.samples }, source.plan.payloadCapacityBytes / 4),
  ]);
  let airborneSamples = 0;
  let axialFront_m = 0;
  const [brickNx, brickNy] = source.plan.brickDimensions;
  const brickResolution = source.plan.brickResolution;
  for (let work = 0; work < Math.min(worklist[1]!, capacity); work++) {
    const page = worklist[7 + work]!;
    const key = metadata[4 * page + 1]!;
    const bz = Math.floor(key / (brickNx * brickNy));
    const by = Math.floor((key - bz * brickNx * brickNy) / brickNx);
    const bx = key - brickNx * (by + brickNy * bz);
    for (let local = 0; local < source.plan.samplesPerBrick; local++) {
      const sample = samples[page * source.plan.samplesPerBrick + local]!;
      if (((sample >>> 16) & 16) === 0) continue;
      const lz = Math.floor(local / (brickResolution * brickResolution));
      const ly = Math.floor((local - lz * brickResolution * brickResolution)
        / brickResolution);
      const lx = local - brickResolution * (ly + brickResolution * lz);
      const point = [-0.5 * tank.width_m + (bx * brickResolution + lx + 0.5)
        * spacing[0], (by * brickResolution + ly + 0.5) * spacing[1],
        -0.5 * tank.depth_m + (bz * brickResolution + lz + 0.5) * spacing[2]];
      const axial = point[0] - outlet.x;
      if (point[1] <= poolSurface_m + spacing[1]
        || Math.abs(point[2] - outlet.z) >= 2 * radius_m) continue;
      axialFront_m = Math.max(axialFront_m, axial);
      if (axial > 0.2) airborneSamples += 1;
    }
  }
  return { airborneSamples, axialFront_m };
}

dawnTest("Sparse CM12 hose-tank launches a continuous airborne jet",
  { timeout: 240_000 }, async () => {
    await acquireWebGPUExclusiveLock("dawn-test",
      "tests/sparse-cm12-hose-inflow-dawn.test.ts");
    let device: GPUDevice | undefined;
    try {
      const dawn = await import(pathToFileURL(dawnModule!).href) as {
        create(options: string[]): GPU;
        globals: Record<string, unknown>;
      };
      Object.assign(globalThis, dawn.globals);
      const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
      const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
      assert.ok(adapter, "Dawn must expose a WebGPU adapter");
      device = await adapter.requestDevice({
        requiredLimits: requiredFluidDeviceLimits(adapter.limits),
      });
      const uncaptured: string[] = [];
      device.addEventListener("uncapturederror", (event) => {
        event.preventDefault();
        uncaptured.push(event.error.message);
      });

      const scene = createPaperScenario("hose-tank");
      const inflow = scene.fluid.inflow!;
      device.pushErrorScope("validation");
      const solver = await WebGPUAdaptiveMassSolver.createAsync(
        device, scene, "balanced", undefined,
        {
          resolutionMode: "adaptive",
          brickFineResolution: 8,
          timeStep: "paper",
          pressureIterations: 8,
        },
        () => {},
      );
      try {
        await solver.waitForSimulationReady();
        const before = await solver.readDiagnosticFields();
        assert.equal(solver.advanceTo(CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        assert.equal(solver.advanceTo(2 * CM12_PAPER_DT_S, []), true);
        await device.queue.onSubmittedWorkDone();
        const after = await solver.readDiagnosticFields();
        const transportPacketQA = await solver.readTransportPacketIndirectQA();
        const vexQA = await solver.readVelocityExtensionQA();
        const vexVelocity = new Float32Array(vexQA.velocityBits.buffer,
          vexQA.velocityBits.byteOffset, vexQA.velocityBits.length);
        let vexMaximum = 0;
        for (let at = 0; at < vexVelocity.length; at += 4) {
          vexMaximum = Math.max(vexMaximum, Math.hypot(vexVelocity[at]!,
            vexVelocity[at + 1]!, vexVelocity[at + 2]!));
        }
        const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
        const tank = scene.container;
        const spacing = [tank.width_m / nx, tank.height_m / ny,
          tank.depth_m / nz] as const;
        const outlet = inflowOutletCenter(inflow);
        const speed = Math.hypot(inflow.velocity_m_s.x, inflow.velocity_m_s.y,
          inflow.velocity_m_s.z);
        assert.ok(transportPacketQA[0]! > 0,
          "the hose source must schedule conservative transport packets");
        assert.ok(vexMaximum * spacing[0] > 0.8 * speed,
          `the effective transport plane must retain hose speed; measured ${
            vexMaximum * spacing[0]} m/s`);
        const direction = [inflow.velocity_m_s.x / speed,
          inflow.velocity_m_s.y / speed, inflow.velocity_m_s.z / speed] as const;
        let addedMass = 0;
        let forwardMomentum = 0;
        let momentumWeight = 0;
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
          for (let x = 0; x < nx; x += 1) {
            const point = [-0.5 * tank.width_m + (x + 0.5) * spacing[0],
              (y + 0.5) * spacing[1],
              -0.5 * tank.depth_m + (z + 0.5) * spacing[2]] as const;
            const relative = [point[0] - outlet.x, point[1] - outlet.y,
              point[2] - outlet.z] as const;
            const axial = relative[0] * direction[0] + relative[1] * direction[1]
              + relative[2] * direction[2];
            const radial = Math.hypot(relative[0] - axial * direction[0],
              relative[1] - axial * direction[1], relative[2] - axial * direction[2]);
            if (axial < -spacing[0] || axial > speed * CM12_PAPER_DT_S + spacing[0]
              || radial > inflow.radius_m + Math.max(...spacing)) continue;
            const cell = x + nx * (y + ny * z);
            const added = Math.max(0, after.density[cell]! - before.density[cell]!);
            addedMass += added;
            if (after.density[cell]! <= 0.05) continue;
            const vx = after.velocity[4 * cell]!;
            const vy = after.velocity[4 * cell + 1]!;
            const vz = after.velocity[4 * cell + 2]!;
            forwardMomentum += after.density[cell]!
              * (vx * direction[0] + vy * direction[1] + vz * direction[2]);
            momentumWeight += after.density[cell]!;
          }
        }
        const expectedAddedMass = Math.PI * inflow.radius_m ** 2 * speed
          * CM12_PAPER_DT_S / (spacing[0] * spacing[1] * spacing[2]);
        assert.ok(addedMass > 0.8 * expectedAddedMass,
          `the first hose step must deliver its authored flux; measured ${addedMass}`
          + ` of ${expectedAddedMass} finest-cell volumes`);
        const meanForwardSpeed = forwardMomentum / Math.max(momentumWeight, 1e-12);
        assert.ok(meanForwardSpeed > 0.5,
          `the injected plug must retain forward momentum; measured ${meanForwardSpeed} m/s`);

        for (let step = 3; step <= 12; step += 1) {
          assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();
        const curved = await solver.readDiagnosticFields();
        const poolSurface_m = tank.fillFraction * tank.height_m;
        let axialFront = 0;
        let airborneMassBeyondSource = 0;
        let maximumDensity = 0;
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
          for (let x = 0; x < nx; x += 1) {
            const point = [-0.5 * tank.width_m + (x + 0.5) * spacing[0],
              (y + 0.5) * spacing[1],
              -0.5 * tank.depth_m + (z + 0.5) * spacing[2]] as const;
            const relative = [point[0] - outlet.x, point[1] - outlet.y,
              point[2] - outlet.z] as const;
            const axial = relative[0] * direction[0] + relative[1] * direction[1]
              + relative[2] * direction[2];
            const density = curved.density[x + nx * (y + ny * z)]!;
            maximumDensity = Math.max(maximumDensity, density);
            if (density > 0.05 && point[1] > poolSurface_m + spacing[1]
              && Math.abs(relative[2]) < 2 * inflow.radius_m) {
              axialFront = Math.max(axialFront, axial);
              if (axial > 0.2) airborneMassBeyondSource += density;
            }
          }
        }
        assert.ok(axialFront > 0.6 && airborneMassBeyondSource > 500,
          `resident hose must reach the pool with substantial airborne volume; front ${
            axialFront} m, mass ${airborneMassBeyondSource}`);
        assert.ok(maximumDensity <= 3.01,
          `the nozzle reservoir must not accumulate unbounded compressed rho; max ${maximumDensity}`);
        const curvedPresentation = await readCompactHoseStats(device, solver, tank,
          spacing, outlet, inflow.radius_m, poolSurface_m);
        assert.ok(curvedPresentation.axialFront_m > 0.6
          && curvedPresentation.airborneSamples > 500,
        `compact hose publication must reach the pool; ${JSON.stringify(curvedPresentation)}`);
        const binCount = 6;
        const reach_m = 0.62;
        const weights = new Float64Array(binCount);
        const heights = new Float64Array(binCount);
        for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
          for (let x = 0; x < nx; x += 1) {
            const point = [-0.5 * tank.width_m + (x + 0.5) * spacing[0],
              (y + 0.5) * spacing[1],
              -0.5 * tank.depth_m + (z + 0.5) * spacing[2]] as const;
            if (point[1] <= poolSurface_m + 0.5 * spacing[1]) continue;
            const relative = [point[0] - outlet.x, point[1] - outlet.y,
              point[2] - outlet.z] as const;
            const axial = relative[0] * direction[0] + relative[1] * direction[1]
              + relative[2] * direction[2];
            if (axial < 0 || axial >= reach_m) continue;
            const sideDistance = Math.abs(relative[2]);
            if (sideDistance > 2 * inflow.radius_m) continue;
            const cell = x + nx * (y + ny * z);
            const density = curved.density[cell]!;
            if (density <= 0.05) continue;
            const bin = Math.min(binCount - 1, Math.floor(axial / reach_m * binCount));
            weights[bin] += density;
            heights[bin] += density * point[1];
          }
        }
        const resolvedHeights = Array.from(heights, (sum, index) =>
          weights[index]! > 0.5 ? sum / weights[index]! : Number.NaN)
          .filter(Number.isFinite);
        assert.ok(resolvedHeights.length >= 3,
          `the hose arc must resolve across at least three axial bins; measured ${resolvedHeights}`);
        assert.ok(resolvedHeights.at(-1)! < outlet.y - 0.04
          && resolvedHeights.at(-1)! < Math.max(...resolvedHeights) - 0.02,
          `gravity must bend the jet toward the pool; measured heights ${resolvedHeights}`);

        const longRunSteps = Math.ceil(6.25 / CM12_PAPER_DT_S);
        for (let step = 13; step <= longRunSteps; step += 1) {
          assert.equal(solver.advanceTo(step * CM12_PAPER_DT_S, []), true);
        }
        await device.queue.onSubmittedWorkDone();
        const activity = await solver.readGPUActivityPolicy();
        assert.equal(activity.faultFlags, 0,
          `the continuous source must not fault resident topology; 0x${activity.faultFlags.toString(16)}`);
        assert.equal(activity.acceptedSteps, longRunSteps,
          "every long-run hose frame must commit");
        const longRun = await solver.readDiagnosticFields();
        const stateMass = longRun.density.reduce((sum, value) => sum + value, 0);
        const initialMass = before.density.reduce((sum, value) => sum + value, 0);
        const integratedSourceMass = Math.PI * inflow.radius_m ** 2 * speed
          * (longRunSteps * CM12_PAPER_DT_S)
          / (spacing[0] * spacing[1] * spacing[2]);
        assert.ok(stateMass > initialMass + 0.4 * integratedSourceMass,
          `six seconds of hose inflow must retain a full airborne tube; ${
            initialMass} -> ${stateMass}, source ${integratedSourceMass}`);
        const fineSource = solver.globalFineLevelSetSource;
        assert.ok(fineSource?.presentationControl,
          "Sparse CM12 must expose its compact presentation receipt");
        const fpp = await readGPUWords(device, fineSource.presentationControl, 32);
        assert.equal(fpp[FPP_HEADER.faultCode], 0,
          `compact water publication faulted with code ${fpp[FPP_HEADER.faultCode]}`);
        assert.ok((fpp[FPP_HEADER.flags]! & FPP_FLAG.executionComplete) !== 0,
          `compact water publication did not complete; flags 0x${fpp[FPP_HEADER.flags]!.toString(16)}`);
        assert.ok(fpp[FPP_HEADER.publishedPageCount]! > 0,
          "compact water publication must contain visible pages");
        assert.equal(fpp[FPP_HEADER.generationReceipt], longRunSteps,
          "the renderer publication must receipt the latest accepted frame");
        const longPresentation = await readCompactHoseStats(device, solver, tank,
          spacing, outlet, inflow.radius_m, poolSurface_m);
        assert.ok(longPresentation.axialFront_m > 0.6
          && longPresentation.airborneSamples > 500,
        `the published hose must remain continuous after six seconds; ${
          JSON.stringify(longPresentation)}`);
        const surface = await readFineUpperSurfaceField(device, solver, [nx, ny, nz]);
        assert.ok(surface, "Sparse CM12 must expose its compact fine surface");
        const visibleColumns = surface.reduce((count, height) =>
          count + (Number.isFinite(height) ? 1 : 0), 0);
        assert.ok(visibleColumns > 0.25 * nx * nz,
          `the tank pool must remain renderer-visible; only ${visibleColumns}/${nx * nz} columns published`);
      } finally {
        solver.destroy();
      }
      const validation = await device.popErrorScope();
      assert.equal(validation?.message, undefined);
      assert.deepEqual(uncaptured, []);
    } finally {
      device?.destroy();
      releaseWebGPUExclusiveLock();
    }
  });
