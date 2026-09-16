/** Focused half-pool publication A/B; derived from probe-pool-impact-ab-dawn.ts.
 * AXIS_DISABLE_HEIGHT changes only shader modules created by this diagnostic.
 * Saves three density/publication checkpoints to limit disk usage. */
import { axisArtifactStageAudit } from "./axis-artifact-stage-audit";
import { createPoolEnergyBudget } from "./pool-impact-energy-budget";
import { createPoolScalarStageAudit } from "./pool-scalar-stage-audit";
import { createPoolFrameFailureAudit } from "./pool-frame-failure-audit";
import { unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readFile, writeFile as originalWriteFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { adaptiveMassMethod } from "../lib/methods/adaptive-volume/method";
import type { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-volume/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

async function read(device: GPUDevice, source: GPUBuffer, bytes = source.size, offset = 0) {
  const target = device.createBuffer({ size: bytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, offset, target, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await target.mapAsync(GPUMapMode.READ);
    return new Uint8Array(target.getMappedRange()).slice();
  } finally {
    if (target.mapState === "mapped") target.unmap();
    target.destroy();
  }
}

async function writeFile(path: string, data: Parameters<typeof originalWriteFile>[1]) {
  if (/configuration\.json$|trace\.json$|\/0-solidOpenFraction\.bin$|\/\d+-density\.bin$|presentation-\d+\/(phi\.bin|source\.json)$/.test(path)) await originalWriteFile(path, data);
}

// Run one arm per process, under the same GPU lease as the regression suite.
// Compare the accepted volume averages, independently of the surface renderer.
const modulePath = process.env.WEBGPU_NODE_MODULE;
assert.ok(modulePath, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
const maxCell = Number(process.env.POOL_MAX_CELL ?? 1);
const steps = Number(process.env.POOL_STEPS ?? 100);
const freezeStep = Number(process.env.POOL_FREEZE_STEP ?? 0);
const impactShiftXCells = Number(process.env.AXIS_IMPACT_SHIFT_X_CELLS ?? 0);
assert.ok(Number.isFinite(impactShiftXCells));
const paddedTank = process.env.AXIS_PADDED_TANK === "1";
const tankShiftCells = Number(process.env.AXIS_TANK_SHIFT_CELLS ?? 0);
assert.ok(Number.isInteger(tankShiftCells) && tankShiftCells >= 0 && tankShiftCells <= 4);
const legacyFace = process.env.POOL_LEGACY_FACE === "1";
const legacyWarmup = process.env.POOL_LEGACY_WARMUP === "1";
const energySteps = new Set((process.env.POOL_ENERGY_STEPS ?? "").split(",").filter(Boolean).map(Number));
assert.ok(process.env.POOL_SCALAR_AUDIT !== "1" || energySteps.size === 0,
  "scalar and energy captures each own the diagnostic stage observer; run separate arms");
assert.ok(Number.isSafeInteger(freezeStep) && freezeStep >= 0 && freezeStep <= steps);
const dt = Number(process.env.POOL_DT ?? 1 / 30);
const output = process.env.POOL_OUTPUT ?? "artifacts/axis-artifacts-2026-09-10/base";
assert.ok([0, 1, 2, 4, 8].includes(maxCell));
assert.ok(Number.isSafeInteger(steps) && steps > 0);
assert.ok(Number.isFinite(dt) && dt > 0);
await acquireWebGPUExclusiveLock("dawn-probe", "pool-impact-ab");
const live = new Set<GPU>();
let gpu: GPU | undefined, device: GPUDevice | undefined, solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(modulePath).href);
  Object.assign(globalThis, dawn.globals);
  gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  live.add(gpu!);
  const adapter = await gpu!.requestAdapter(); assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  if (process.env.AXIS_DISABLE_HEIGHT === "1") {
    const createShaderModule = device.createShaderModule.bind(device);
    device.createShaderModule = (descriptor: GPUShaderModuleDescriptor) => {
      const code = descriptor.code.replace(
        /fn presentationHeightPolicyEnabled\(brick:u32\)->bool\{[\s\S]*?\n\}/g,
        "fn presentationHeightPolicyEnabled(brick:u32)->bool{_=brick;return false;}"
      );
      return createShaderModule({ ...descriptor, code });
    };
  }
  const errors: string[] = [];
  device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  const scene = sceneDocument(getSceneDefinition("coarse-first-pool-impact-half"));
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  assert.equal(scene.fluid.initialLiquidVolumes?.length, 1);
  const impact = scene.fluid.initialLiquidVolumes![0]!;
  assert.equal(impact.shape, "sphere");
  impact.center_m.x += impactShiftXCells * scene.voxelDomain.finestCellSize_m;
  if (paddedTank) {
    // Keep the 64x48x64 physical tank, but place its solid shell and liquid
    // inside an 80x48x80 lattice. Translate tank and drop together, leaving
    // the brick lattice and numerical resolution unchanged.
    const offset = 8 + tankShiftCells;
    const shift = tankShiftCells * scene.voxelDomain.finestCellSize_m;
    scene.container.width_m = scene.container.depth_m = 4;
    scene.container.fillFraction = 0;
    scene.solidVoxels = scene.solidVoxels!.map(patch => ({ ...patch,
      minimum: [patch.minimum[0] + offset, patch.minimum[1], patch.minimum[2] + offset],
      maximumExclusive: [patch.maximumExclusive[0] + offset, patch.maximumExclusive[1], patch.maximumExclusive[2] + offset],
    }));
    impact.center_m.x += shift;
    impact.center_m.z += shift;
    scene.fluid.initialLiquidVolumes!.unshift({ shape: "box",
      min_m: { x: -1.6 + shift, y: 0, z: -1.6 + shift },
      max_m: { x: 1.6 + shift, y: 0.8, z: 1.6 + shift },
    });
  }
  if (maxCell) scene.fluid.refinementRegions = [{
    id: "ab-whole-domain", rule: "minimum-cell-size", minimumCellSize_cells: 1,
    maximumCellSize_cells: maxCell,
    min_m: { x: -scene.container.width_m / 2, y: 0, z: -scene.container.depth_m / 2 },
    max_m: { x: scene.container.width_m / 2, y: scene.container.height_m,
      z: scene.container.depth_m / 2 },
  }];
  const values = resolveMethodValues(adaptiveMassMethod, "balanced", {
    selectorMode: "surface", brickFineResolution: "8",
    timeStep: "scene", ...JSON.parse(process.env.POOL_OVERRIDES ?? "{}"),
  });
  solver = await adaptiveMassMethod.createSolverAsync!(device, scene, "balanced", values,
    undefined, () => {}) as WebGPUAdaptiveMassSolver;
  await solver.waitForSimulationReady();
  // Historical face-filter ablations may request their original warm-up.
  // Current adaptive/frozen comparisons use the current solver from reset.
  if (freezeStep && legacyWarmup) solver.setLegacyFaceTransportForQA(true);
  const [nx, ny, nz] = [solver.info.nx, solver.info.ny, solver.info.nz];
  await mkdir(output, { recursive: true });
  const residentWGSLHash = createHash("sha256").update(await readFile(new URL(
    "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident.wgsl.ts", import.meta.url))).digest("hex");
  await writeFile(`${output}/configuration.json`, JSON.stringify({ scene, values, grid: [nx, ny, nz],
    steps, dt, maxCell, freezeStep, legacyFace, legacyWarmup, residentWGSLHash, impactShiftXCells, paddedTank, tankShiftCells,
    axisDisableHeight: process.env.AXIS_DISABLE_HEIGHT === "1" }, null, 2));
  const trace = [];
  const failureAudit = process.env.POOL_AUDIT_FAILURES === "1"
    ? createPoolFrameFailureAudit(solver, output) : undefined;
  let frozenTopology: unknown;
  let energyBudget: Awaited<ReturnType<typeof createPoolEnergyBudget>> | undefined;
  const topology = (a: Awaited<ReturnType<WebGPUAdaptiveMassSolver["readGPUActivityPolicy"]>>) => ({
    generation: a.acceptedTopologyGeneration,
    bricks: a.bricks.map(b => [b.leafId, b.coordinate, b.spanBricks, b.active, b.acceptedResolution]),
  });
  for (let step = 0; step <= steps; step++) {
    if (step === Number(process.env.AXIS_TOGGLE_HEIGHT_STEP)) {
      solver.applyRuntimeValues({ ...values, presentationColumnHeight: "off" });
    }
    if (step) {
      const axisAudit = (process.env.AXIS_AUDIT_STEPS ?? "").split(",").includes(String(step))
        ? await axisArtifactStageAudit(device, solver, output, step, values.gammaDiffusion !== "off") : undefined;
      const scalarAudit = process.env.POOL_SCALAR_AUDIT === "1"
        ? await createPoolScalarStageAudit(device, solver, scene.voxelDomain.finestCellSize_m,
          values.gammaDiffusion !== "off") : undefined;
      if (energySteps.has(step)) {
        assert.ok(frozenTopology, "energy budget requires the frozen checkpoint");
        energyBudget ??= await createPoolEnergyBudget(device, solver, scene.voxelDomain.finestCellSize_m);
        energyBudget.arm();
      }
      while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
      await solver.waitForTopologyReady();
      await axisAudit?.();
      if (scalarAudit) await writeFile(`${output}/${step}-scalar-budget.json`,
        JSON.stringify(await scalarAudit(), null, 2));
      assert.equal(solver.info.encodedSteps, step, "deferred preparation must not drop a step");
      if (energySteps.has(step)) await writeFile(`${output}/${step}-energy.json`,
        JSON.stringify(await energyBudget!.read(), null, 2));
    }
    if (freezeStep && step === freezeStep) {
      frozenTopology = topology(await solver.readGPUActivityPolicy());
      solver.setTopologyFrozen(true);
      solver.setLegacyFaceTransportForQA(legacyFace);
      await writeFile(`${output}/frozen-topology.json`, JSON.stringify(frozenTopology));
    }
    await failureAudit?.capture(step, step * dt);
    if (step % 5 !== 0 && step !== steps && step !== freezeStep) continue;
    const fields = await solver.readDiagnosticFields(true);
    for (const name of ["density", "solidOpenFraction", "velocity", "pressure", "divergence", "gamma", "pressureRhs", "pressureDiagonal"] as const) {
      const data = fields[name];
      assert.ok(data.every(Number.isFinite), `${name} finite at ${step}`);
      await writeFile(`${output}/${step}-${name}.bin`, new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    }
    if ([60, 90, steps].includes(step)) {
      await mkdir(`${output}/presentation-${step}`, { recursive: true });
      const source: WebGPUAdaptiveMassSolver["globalFineLevelSetSource"] = solver.globalFineLevelSetSource;
      const blobs: Uint8Array[] = await Promise.all([
        read(device, source.worklist), read(device, source.metadata),
        read(device, source.samples, source.plan.payloadCapacityBytes),
      ]);
      const [worklist, metadata, samples] = blobs.map(b => new Uint32Array(b.buffer));
      const names = ["worklist", "metadata", "samples"];
      for (let i = 0; i < 3; i++)
        await writeFile(`${output}/presentation-${step}/${names[i]}.bin`, blobs[i]!);
      await writeFile(`${output}/presentation-${step}/source.json`, JSON.stringify(source.plan));
      const [nx, ny, nz] = source.plan.sampleDimensions;
      const heightBytes = 4 * (9 * nx * nz + 16);
      if (source.samples.size >= 2 * source.plan.payloadCapacityBytes + heightBytes) {
        await writeFile(`${output}/presentation-${step}/height-field.bin`, await read(device, source.samples,
          heightBytes, 2 * source.plan.payloadCapacityBytes));
      }
      assert.equal(source.plan.brickResolution, 8);
      // This dense convenience view covers ordinary pages around the pool's
      // free surface. Leave macro interiors as NaN; their complete encoded data
      // remains in metadata.bin / samples.bin for other consumers.
      const phi = new Float32Array(nx * ny * nz).fill(NaN), width = new Uint8Array(nx * ny * nz);
      for (let at = 0; at < worklist![1]!; at++) {
        const page = worklist![7 + at]!, key = metadata![4 * page + 1]!;
        assert.equal(metadata![4 * page], page);
        assert.equal(metadata![4 * page + 2], worklist![0]);
        const descriptor = metadata![4 * page + 3]!;
        if ((descriptor & 0x80000000) !== 0 && ((descriptor >>> 24) & 31) !== 0) continue;
        const bx = (key & 2047) - 1024, by = ((key >>> 11) & 1023) - 512, bz = ((key >>> 21) & 2047) - 1024;
        for (let q = 0; q < 512; q++) {
          const x = bx * 8 + q % 8, y = by * 8 + Math.floor(q / 8) % 8, z = bz * 8 + Math.floor(q / 64);
          if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz)
            continue;
          const packed = samples![page * 512 + q]!;
          if (!(packed & 0x10000))
            continue;
          phi[x + nx * (y + ny * z)] = unpackFineLevelSetPackedPhi(packed);
          width[x + nx * (y + ny * z)] = 1 << ((packed >>> 24) & 15);
        }
      }
      await writeFile(`${output}/presentation-${step}/phi.bin`, new Uint8Array(phi.buffer));
      await writeFile(`${output}/presentation-${step}/width.bin`, width);
    }

    const activity = await solver.readGPUActivityPolicy();
    if (frozenTopology) assert.deepEqual(topology(activity), frozenTopology, "frozen topology is immutable");
    if (maxCell === 1) assert.ok(activity.bricks.every(b => !b.active || b.acceptedResolution === 8),
      "the enforcement region must keep every active cell at max1");
    await writeFile(`${output}/${step}-activity.json`, JSON.stringify(activity));
    const stats = await solver.readStats();
    await writeFile(`${output}/${step}-stats.json`, JSON.stringify(stats));
    const heights = new Float32Array(nx * nz);
    let mass = 0, momentY = 0, momentR2 = 0, kinetic = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      const at = x + nx * (y + ny * z), rho = fields.density[at]!;
      mass += rho; heights[x + nx * z] += rho;
      momentY += rho * (y + 0.5); momentR2 += rho * ((x + 0.5 - nx / 2) ** 2 + (z + 0.5 - nz / 2) ** 2);
      kinetic += 0.5 * rho * (fields.velocity[4 * at]! ** 2 + fields.velocity[4 * at + 1]! ** 2 + fields.velocity[4 * at + 2]! ** 2);
    }
    await writeFile(`${output}/${step}-height.bin`, new Uint8Array(heights.buffer));
    const row = { step, time: step * dt, mass, centerOfMassY: momentY / mass,
      rmsRadius: Math.sqrt(momentR2 / mass), kinetic,
      cells: activity.bricks.filter(b => b.active).reduce((n, b) => n + b.acceptedResolution ** 3, 0),
      histogram: Object.fromEntries([1, 2, 4, 8].map(r => [r, activity.bricks.filter(b => b.active && b.acceptedResolution === r).length])),
      pressureIterations: stats.pressureIterationsExecuted, pressureResidual: stats.pressureRelativeResidual };
    trace.push(row); console.log(JSON.stringify(row));
    assert.deepEqual(errors, []);
  }
  await writeFile(`${output}/trace.json`, JSON.stringify(trace, null, 2));
  await failureAudit?.finish();
} finally {
  solver?.destroy(); device?.destroy(); await releaseWebGPUExclusiveLock();
  if (gpu) live.delete(gpu);
}
