/** Verify the registered forest waterfall with the production NB-FLIP solver.
 *
 * node --import tsx tools/check-forest-waterfall-live.ts --steps=360
 *   --out=artifacts/forest-waterfall/live-physics.json
 *
 * This is a solver readback, not a Blender render or a replacement simulation.
 * No resolution, timestep, capacity, source, or method settings are overridden.
 * Static visual assets need not be loaded for the solver's own SolidWorld.
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { inflowOutletCenter } from "../lib/core/inflow-boundary";
import { managedGPUDevice, gpuCompilationManagerFor } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import type { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { sceneFluidFrame } from "../lib/core/scene-lattice";
import { createNbflipVolumeProbe } from "./probe-nbflip-volume";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const argument of process.argv.slice(2)) assert.match(argument, /^--(?:steps|every|out)=.+$/);
const steps = Number(option("steps") ?? 360), every = Number(option("every") ?? 60);
assert.ok(Number.isInteger(steps) && steps > 0);
assert.ok(Number.isInteger(every) && every > 0);
const output = option("out") ?? "artifacts/forest-waterfall/live-physics.json";
mkdirSync(dirname(output), { recursive: true });
const definition = getSceneDefinition("forest-waterfall");
const scene = structuredClone(sceneDocument(definition));
const physicsDigest = createHash("sha256").update(JSON.stringify({
  container: scene.container, fluid: scene.fluid, terrain: scene.terrain,
  solidVoxels: scene.solidVoxels, solidBoundary: scene.solidBoundary,
  numerics: scene.numerics, voxelDomain: scene.voxelDomain,
  rigidBodies: scene.rigidBodies, randomSeed: scene.randomSeed,
})).digest("hex");
assert.equal(scene.sceneId, "forest-waterfall");
assert.equal(definition.methodProfile?.methodId, method.id);
assert.equal(method.id, "uniform-narrow-band-flip");
const quality = definition.methodProfile?.quality ?? "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile?.overrides });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s;
assert.ok(steps * dt <= scene.duration_s, "Run stays inside the authored scene duration");
const inflow = scene.fluid.inflow;
assert.ok(inflow && inflow.enabled !== false, "The registered scene has a live fluid source");
const outlet = inflowOutletCenter(inflow);
const poolLevel = Math.max(...(scene.fluid.initialLiquidVolumes ?? []).map(volume =>
  volume.shape === "box" ? volume.max_m.y : -Infinity));
assert.ok(Number.isFinite(poolLevel), "The registered scene initializes a real pool");

paceDawnEventPolling();
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter);
const features = ["float32-filterable", "subgroups"].filter(f => adapter.features.has(f as GPUFeatureName)) as GPUFeatureName[];
const device = managedGPUDevice(await adapter.requestDevice({
  requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits),
}), { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
let solver: WebGPUUniformReferenceSolver | undefined;
const rows: Record<string, number | null>[] = [];
let failure: string | undefined;
let metadata: Record<string, unknown> = {};
let volumeProbe: Awaited<ReturnType<typeof createNbflipVolumeProbe>> | undefined;

async function readBytes(buffer: GPUBuffer, size: number): Promise<ArrayBuffer> {
  if (size === 0) return new ArrayBuffer(0);
  const read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, read, 0, size);
    device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ);
    return read.getMappedRange().slice(0);
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

try {
  const began = performance.now();
  const progress = setInterval(() => {
    const { state, queued, active, cached } = gpuCompilationManagerFor(device).snapshot();
    console.log(JSON.stringify({ phase: "initializing", elapsed_s: (performance.now() - began) / 1000,
      compilation: { state, queued, active, cached } }));
  }, 15000);
  try {
    solver = await method.createSolverAsync!(device, scene, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver;
  } finally { clearInterval(progress); }
  const band = (solver as unknown as { mixedFrame: { narrowBandFlip: UniformNarrowBandFlip } }).mixedFrame.narrowBandFlip;
  const capacity = band.capacity;
  metadata = {
    scene: definition.id, physicsDigest, method: method.id, quality, values, dt, steps,
    requestedTime_s: steps * dt, grid: [solver.info.nx, solver.info.ny, solver.info.nz],
    fluidCell_m: scene.voxelDomain.finestCellSize_m, poolLevel_m: poolLevel,
    outlet_m: outlet, sourceRadius_m: inflow.radius_m, sourceVelocity_m_s: inflow.velocity_m_s,
    capacity, allocatedBytes: solver.info.allocatedBytes, features,
    validation: "Production solver and registered scene; GPU particle readback. Does not validate the rendered appearance.",
  };
  console.log(JSON.stringify({ phase: "constructed", ...metadata }));
  volumeProbe = await createNbflipVolumeProbe(device, solver);
  const fluidFrame = sceneFluidFrame(scene);
  const cellVolume = fluidFrame.cellSize_m.reduce((a, b) => a * b, 1);
  let initialPoolVolume = 0;
  async function sample(frame: number, wall_ms = 0) {
    const state = new Uint32Array(await readBytes(band.state, 128));
    assert.equal(state[3], 0, "No fatal particle capacity overflow");
    assert.ok(state[0]! <= capacity);
    const source = band.particleSource;
    assert.ok(source.frame);
    const coordinates = new Float32Array(await readBytes(source.frame.buffer, 48));
    const particles = new Float32Array(await readBytes(band.activeParticles, state[0]! * band.particleStride * 4));
    let live = 0, pool = 0, stream = 0, descending = 0, impactRegion = 0;
    let minStreamHeight = Infinity, maxSpeed = 0, minStreamVy = Infinity;
    for (let i = 0; i < state[0]!; i++) {
      const at = i * band.particleStride;
      if (particles[at + 3]! <= 0) continue; // Slab holes are not live particles.
      const x = coordinates[8]! + particles[at]! * coordinates[4]!;
      const y = coordinates[9]! + particles[at + 1]! * coordinates[5]!;
      const z = coordinates[10]! + particles[at + 2]! * coordinates[6]!;
      const vx = particles[at + 4]!, vy = particles[at + 5]!, vz = particles[at + 6]!;
      assert.ok([x, y, z, vx, vy, vz].every(Number.isFinite), `Finite live particle ${i}`);
      live++;
      maxSpeed = Math.max(maxSpeed, Math.hypot(vx, vy, vz));
      if (y <= poolLevel + .5 && Math.abs(x) < 11 && Math.abs(z - 2) < 10.5) pool++;
      if (Math.abs(x - outlet.x) < 6 && Math.abs(z - outlet.z) < 7 && y > poolLevel + .5) {
        stream++;
        minStreamHeight = Math.min(minStreamHeight, y);
        minStreamVy = Math.min(minStreamVy, vy);
        if (vy < -.25) descending++;
        if (y < poolLevel + 2.5 && vy < -.25) impactRegion++;
      }
    }
    // Settled regions retain Eulerian water after retiring all particles.
    // The production phi-volume probe reads live GPU ownership and solid
    // fractions, including these frozen regions.
    const volume = await volumeProbe!.read();
    const { nx, ny } = solver!.info;
    let poolVolume = 0, totalVolume = 0;
    for (let i = 0; i < volume.length; i++) {
      const amount = volume[i]! * cellVolume;
      assert.ok(Number.isFinite(amount) && amount >= 0);
      totalVolume += amount;
      const y = fluidFrame.container_m[1] + (fluidFrame.boxLow[1] + Math.floor(i / nx) % ny) * fluidFrame.cellSize_m[1];
      if (y < poolLevel + .5) poolVolume += amount;
    }
    if (frame === 0) initialPoolVolume = poolVolume;
    assert.ok(poolVolume > 100, "The live Eulerian field contains the authored pond");
    assert.ok(poolVolume > .8 * initialPoolVolume, "The settled pond retains its water when particles retire");
    const row = {
      frame, time_s: frame * dt, wall_ms, live, pool, stream, descending, impactRegion,
      poolVolume_m3: poolVolume, totalVolume_m3: totalVolume,
      minStreamHeight_m: Number.isFinite(minStreamHeight) ? minStreamHeight : null,
      minStreamVy_m_s: Number.isFinite(minStreamVy) ? minStreamVy : null,
      maxSpeed_m_s: maxSpeed, extent: state[0]!, reportedLive: state[1]!,
      moveSpill: state[16]!, seedSpill: state[21]!,
    };
    rows.push(row);
    console.log(JSON.stringify(row));
    return row;
  }
  await sample(0);
  for (let frame = 1; frame <= steps; frame++) {
    const began = performance.now();
    assert.ok(solver.advanceTo(frame * dt, []), `Frame ${frame} was admitted`);
    await solver.awaitFrameCompletion();
    assert.equal(solver.info.simulationPipelineError, undefined);
    assert.equal(band.capacity, capacity, "The authored run does not change particle capacity");
    assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - frame * dt) < 1e-8);
    if (frame === 1 || frame % every === 0 || frame === steps) await sample(frame, performance.now() - began);
  }
  const last = rows.at(-1)!;
  if (steps * dt >= inflow.start_s + .75) {
    assert.ok(last.stream! > 0, "The production source emits a live waterfall");
    assert.ok(last.descending! > 0, "Emitted water has downward velocity");
    assert.ok(last.minStreamHeight_m! < outlet.y - .25, "Water leaves the source and descends through the scene");
  }
  if (steps * dt >= inflow.start_s + 2.5) {
    assert.ok(rows.some(row => row.minStreamHeight_m !== null && row.minStreamHeight_m! < (outlet.y + poolLevel) / 2),
      "A sustained run advances the waterfall into the lower half of the drop");
  }
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ phase: "complete", completedTime_s: solver.info.completedTime_s, errors }));
} catch (error) {
  failure = error instanceof Error ? error.stack ?? error.message : String(error);
  throw error;
} finally {
  writeFileSync(output, JSON.stringify({ ...metadata, completedTime_s: solver?.info.completedTime_s,
    rows, errors, failure, passed: !failure && errors.length === 0 }, null, 2) + "\n");
  volumeProbe?.destroy();
  solver?.destroy();
  await device.queue.onSubmittedWorkDone();
  device.destroy();
}
