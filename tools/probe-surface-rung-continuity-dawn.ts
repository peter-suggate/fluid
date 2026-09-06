import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { adaptiveMassSolverOptions } from "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";

// Retain the native Dawn instance until all asynchronous readbacks finish.
const live = new Set<GPU>();
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

const dawnModule = process.env.WEBGPU_NODE_MODULE;
assert.ok(dawnModule, "Set WEBGPU_NODE_MODULE to the native Dawn module path");
await acquireWebGPUExclusiveLock("dawn-probe", "surface-rung-continuity");
const errors: string[] = [];
let gpu: GPU | undefined;
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const { create, globals } = await import(pathToFileURL(dawnModule).href);
  Object.assign(globalThis, globals);
  gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]) as GPU;
  live.add(gpu);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter);
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  device.addEventListener("uncapturederror", event => errors.push(event.error.message));
  const scene = sceneDocument(getSceneDefinition("water-box-tank-fill"));
  scene.rigidBodies = [];
  scene.container = { ...scene.container, width_m: .8, height_m: .8, depth_m: .8, fillFraction: .6 };
  scene.voxelDomain.finestCellSize_m = .05;
  scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
  const dt = 1e-6;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
  const admitted = process.env.SURFACE_TRANSITION_MODE === "admit";
  const region = (width: number, minimum = width) => [{ id: "rung-control", rule: "minimum-cell-size" as const,
    minimumCellSize_cells: minimum, maximumCellSize_cells: width,
    min_m: { x: -.4, y: 0, z: -.4 }, max_m: { x: .4, y: .8, z: .4 } }];
  scene.fluid.refinementRegions = region(2);
  const defaults = adaptiveMassSolverOptions({ selectorMode: "coarse-first", timeStep: "scene" });
  solver = await WebGPUAdaptiveMassSolver.createAsync(device, scene, "balanced", undefined, {
    ...defaults, initialResolutionForQA: 4, maximumMacroSpanBricks: 1,
    topologyPageBudget: 0, gammaDiffusionEnabled: false, surfaceSharpeningEnabled: false,
    activityPolicy: { ...defaults.activityPolicy!, topologyCadenceSteps: 1, demoteEpochs: 1, prepareBricksPerFrame: 256,
      surfaceDisplacementToleranceCells: Number(process.env.SURFACE_TOLERANCE ?? defaults.activityPolicy!.surfaceDisplacementToleranceCells) },
  }, () => {});
  await solver.waitForSimulationReady();
  const source = solver.fieldSnapshotSourceForQA;
  const words = source.templateWords, floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const profile = process.env.SURFACE_PROFILE ?? "bowl";
  const height = (x: number, z: number) => profile === "flat" ? 9.6
    : 9.1 + .006 * ((x - 8) ** 2 + (z - 8) ** 2);
  const density = new Float32Array(source.cellCapacity);
  // Exact finest-column volume, restricted to each native template cell. The
  // D4 bowl avoids introducing a field incompatible with authored symmetry.
  for (let id = 0; id < words[2]!; id++) {
    const at = words[6]! + 8 * id;
    const center = [floats[at]!, floats[at + 1]!, floats[at + 2]!];
    const width = [floats[at + 4]!, floats[at + 5]!, floats[at + 6]!];
    let sum = 0;
    for (let z = 0; z < width[2]!; z++) for (let x = 0; x < width[0]!; x++) {
      const h = height(center[0]! - width[0]! / 2 + x + .5,
        center[2]! - width[2]! / 2 + z + .5);
      sum += Math.min(width[1]!, Math.max(0, h - (center[1]! - width[1]! / 2)));
    }
    density[id] = sum / (width[0]! * width[1]! * width[2]!);
  }
  for (const offset of [source.layout.densityA, source.layout.densityB])
    device.queue.writeBuffer(source.state, 4 * offset, density);
  for (let step = 1; step <= (admitted ? 10 : 5); step++) {
    if (step >= 2) {
      scene.fluid.refinementRegions = admitted && step >= 3 ? region(2, 1) : region(step % 2 === 0 ? 1 : 2);
      solver.applySceneUniforms(structuredClone(scene));
    }
    while (!solver.advanceTo(step * dt, [])) await new Promise(setImmediate);
    await solver.waitForTopologyReady();
    assert.equal(solver.info.encodedSteps, step);
    const output = `${process.env.SURFACE_OUTPUT ?? "artifacts/surface-rung-continuity"}/step-${step}`;
    await mkdir(output, { recursive: true });
    const stats = await solver.readStats();
    assert.ok(Math.abs((stats.simulatedTime_s ?? NaN) - step * dt) < 1e-9);
    const fields = await solver.readDiagnosticFields(true);
    for (const name of ["density", "solidOpenFraction", "velocity", "pressure"] as const)
      await writeFile(`${output}/${name}.bin`, new Uint8Array(fields[name].buffer));
    const source: WebGPUAdaptiveMassSolver["globalFineLevelSetSource"] = solver.globalFineLevelSetSource;
    const blobs: Uint8Array[] = await Promise.all([
      read(device, source.worklist), read(device, source.metadata),
      read(device, source.samples, source.plan.payloadCapacityBytes),
    ]);
    const [worklist, metadata, samples] = blobs.map(b => new Uint32Array(b.buffer));
    const names = ["worklist", "metadata", "samples"];
    for (let i = 0; i < 3; i++)
      await writeFile(`${output}/${names[i]}.bin`, blobs[i]!);
    await writeFile(`${output}/source.json`, JSON.stringify(source.plan));
    const [nx, ny, nz] = source.plan.sampleDimensions;
    const heightBytes = 4 * (9 * nx * nz + 16);
    if (source.samples.size >= 2 * source.plan.payloadCapacityBytes + heightBytes) {
      await writeFile(`${output}/height-field.bin`, await read(device, source.samples,
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
    await writeFile(`${output}/phi.bin`, new Uint8Array(phi.buffer));
    await writeFile(`${output}/width.bin`, width);
    await writeFile(`${output}/activity.json`, JSON.stringify(await solver.readGPUActivityPolicy()));
    await writeFile(`${output}/stats.json`, JSON.stringify(await solver.readStats()));
    console.log(JSON.stringify({ step, time: step * dt, output, errors }));
  }
  assert.deepEqual(errors, []);
} finally {
  solver?.destroy();
  device?.destroy();
  await releaseWebGPUExclusiveLock();
  if (gpu) live.delete(gpu);
}
