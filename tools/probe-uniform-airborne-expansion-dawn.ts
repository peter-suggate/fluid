/** Matched Uniform Geometric symmetric-expansion A/B, with stage and D4 diagnostics. */
import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformGeometricSolverOptions } from "../lib/methods/uniform/uniform-geometric-options";
import { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

process.env.FLUID_UNIFORM_SYMMETRY_STAGE_AUDIT = "1";
const frames = Number(process.env.FLUID_AIRBORNE_AB_FRAMES ?? 45);
const modes = process.env.FLUID_AIRBORNE_AB_MODE === "off" ? ["off"] as const
  : ["off", "on"] as const;
const densePressureAudit = process.env.FLUID_AIRBORNE_SYM_DENSE_PRESSURE === "1";
const valueOverrides = JSON.parse(process.env.FLUID_AIRBORNE_SYM_VALUES ?? "{}") as Record<string, string | number>;
if (densePressureAudit) process.env.FLUID_UNIFORM_PRESSURE_SETUP_AUDIT = "1";
const checkpoints = new Set(Array.from({ length: frames + 1 }, (_, frame) => frame));
const dt = 1 / 30;
type Snapshot = Record<string, Float32Array>;
async function read(device: GPUDevice, texture: GPUTexture, components: number,
  dims: readonly [number, number, number]): Promise<Float32Array> {
  const [nx, ny, nz] = dims;
  const row = Math.ceil(nx * components * 4 / 256) * 256;
  const buffer = device.createBuffer({ size: row * ny * nz,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: ny },
      { width: nx, height: ny, depthOrArrayLayers: nz });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const raw = new Float32Array(buffer.getMappedRange());
    const values = new Float32Array(nx * ny * nz * components);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) {
      const source = (z * ny + y) * row / 4;
      values.set(raw.subarray(source, source + nx * components),
        (z * ny + y) * nx * components);
    }
    return values;
  } finally { buffer.unmap(); buffer.destroy(); }
}
function difference(a: Float32Array, b: Float32Array, components: number) {
  let sum = 0, max = 0, rms = 0, count = 0;
  for (let i = 0; i < a.length; i++) {
    if (components === 4 && i % 4 === 3) continue;
    const d = Math.abs(a[i]! - b[i]!);
    sum += d; rms += d * d; max = Math.max(max, d); count++;
  }
  return { mean: sum / count, rms: Math.sqrt(rms / count), max };
}
function range(field: Float32Array) {
  let min = Infinity, max = -Infinity, nonzero = 0;
  for (const value of field) { min = Math.min(min, value); max = Math.max(max, value); if (value !== 0) nonzero++; }
  return { min, max, nonzero };
}
function largestDifferences(a: Float32Array, b: Float32Array,
  dims: readonly [number, number, number], components: number) {
  const [nx, ny] = dims;
  const ranked: { delta: number; on: number; off: number; xyz: number[]; component: number }[] = [];
  for (let i = 0; i < a.length; i++) {
    if (components === 4 && i % 4 === 3) continue;
    const delta = Math.abs(a[i]! - b[i]!);
    if (delta === 0 || (ranked.length === 5 && delta <= ranked[4]!.delta)) continue;
    const cell = Math.floor(i / components);
    ranked.push({ delta, on: a[i]!, off: b[i]!,
      xyz: [cell % nx, Math.floor(cell / nx) % ny, Math.floor(cell / (nx * ny))],
      component: i % components });
    ranked.sort((left, right) => right.delta - left.delta);
    if (ranked.length > 5) ranked.pop();
  }
  return ranked;
}
function metrics(snapshot: Snapshot, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const volume = snapshot.volume!, velocity = snapshot.velocity!;
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  let mass = 0, wet = 0, kinetic = 0, velocityHighFrequency = 0;
  let d4Mean = 0, d4Max = 0, d4Count = 0;
  const height = new Int16Array(nx * nz).fill(-1);
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    const i = index(x, y, z), v = volume[i]!;
    mass += v;
    if (v >= 0.5) { wet++; height[x + nx * z] = y; }
    const vx = velocity[4 * i]!, vy = velocity[4 * i + 1]!, vz = velocity[4 * i + 2]!;
    kinetic += v * (vx * vx + vy * vy + vz * vz) / 2;
    if (x > 0 && x + 1 < nx && z > 0 && z + 1 < nz && v > 0.1) {
      const lapX = vx - (velocity[4 * index(x - 1, y, z)]! + velocity[4 * index(x + 1, y, z)]!) / 2;
      const lapZ = vz - (velocity[4 * index(x, y, z - 1) + 2]! + velocity[4 * index(x, y, z + 1) + 2]!) / 2;
      velocityHighFrequency += v * (lapX * lapX + lapZ * lapZ);
    }
    for (const j of [index(nx - 1 - x, y, z), index(x, y, nz - 1 - z), index(z, y, x)]) {
      const d = Math.abs(v - volume[j]!);
      d4Mean += d; d4Max = Math.max(d4Max, d); d4Count++;
    }
  }
  let heightD4 = 0, heightRoughness = 0, heightCount = 0;
  for (let z = 1; z + 1 < nz; z++) for (let x = 1; x + 1 < nx; x++) {
    const h = height[x + nx * z]!;
    heightD4 += Math.abs(h - height[(nx - 1 - x) + nx * z]!);
    heightD4 += Math.abs(h - height[x + nx * (nz - 1 - z)]!);
    heightD4 += Math.abs(h - height[z + nx * x]!);
    const neighbors = [height[x - 1 + nx * z]!, height[x + 1 + nx * z]!,
      height[x + nx * (z - 1)]!, height[x + nx * (z + 1)]!];
    heightRoughness += Math.abs(h - neighbors.reduce((a, b) => a + b, 0) / 4);
    heightCount++;
  }
  return { mass, wet, kinetic, velocityHighFrequency, d4Mean: d4Mean / d4Count,
    d4Max, heightD4: heightD4 / (3 * heightCount), heightRoughness: heightRoughness / heightCount };
}
function largestSymmetryErrors(volume: Float32Array, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number) => volume[x + nx * (y + ny * z)]!;
  const ranked: { delta: number; xyz: number[]; mirror: number[]; value: number; mirrored: number }[] = [];
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    for (const mirror of [[nx - 1 - x, y, z], [x, y, nz - 1 - z], [z, y, x]]) {
      const value = at(x, y, z), mirrored = at(mirror[0]!, mirror[1]!, mirror[2]!);
      const delta = Math.abs(value - mirrored);
      if (delta === 0 || (ranked.length === 5 && delta <= ranked[4]!.delta)) continue;
      ranked.push({ delta, xyz: [x, y, z], mirror, value, mirrored });
      ranked.sort((left, right) => right.delta - left.delta);
      if (ranked.length > 5) ranked.pop();
    }
  }
  return ranked;
}
function scalarD4(field: Float32Array, dims: readonly [number, number, number], interiorOnly = false) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number) => field[x + nx * (y + ny * z)]!;
  let maximum = 0, sum = 0, count = 0, signMismatch = 0;
  let worst: { xyz: number[]; mirror: number[]; value: number; mirrored: number; delta: number } | undefined;
  for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
    if (interiorOnly && (x === 0 || y === 0 || z === 0 || x + 1 === nx || y + 1 === ny || z + 1 === nz)) continue;
    for (const mirror of [[nx - 1 - x, y, z], [x, y, nz - 1 - z], [z, y, x]]) {
      const value = at(x, y, z), mirrored = at(mirror[0]!, mirror[1]!, mirror[2]!);
      const d = Math.abs(value - mirrored);
      if ((value < 0) !== (mirrored < 0)) signMismatch++;
      if (d > maximum) worst = { xyz: [x, y, z], mirror, value, mirrored, delta: d };
      maximum = Math.max(maximum, d); sum += d; count++;
    }
  }
  return { max: maximum, mean: sum / count, signMismatch, worst };
}
function physicalFaceD4(field: Float32Array, dims: readonly [number, number, number]) {
  const [nx, ny, nz] = dims;
  const at = (x: number, y: number, z: number, axis: number) =>
    field[4 * (x + nx * (y + ny * z)) + axis]!;
  let maximum = 0, sum = 0, count = 0;
  const worst: { delta: number; face: number[]; reflected: number[]; value: number; mirrorValue: number }[] = [];
  const add = (x: number, y: number, z: number, axis: number,
    rx: number, ry: number, rz: number, raxis: number, sign: number) => {
    const value = at(x, y, z, axis), mirrorValue = sign * at(rx, ry, rz, raxis);
    const delta = Math.abs(value - mirrorValue);
    maximum = Math.max(maximum, delta); sum += delta; count++;
    if (delta > 0 && (worst.length < 3 || delta > worst[2]!.delta)) {
      worst.push({ delta, face: [x, y, z, axis], reflected: [rx, ry, rz, raxis], value, mirrorValue });
      worst.sort((a, b) => b.delta - a.delta);
      if (worst.length > 3) worst.pop();
    }
  };
  // Compare stored interior positive faces only. Negative domain faces are in a separate buffer.
  for (let z = 1; z + 1 < nz; z++) for (let y = 1; y + 1 < ny; y++)
    for (let x = 1; x + 1 < nx; x++) {
      add(x, y, z, 0, nx - 2 - x, y, z, 0, -1);
      add(x, y, z, 2, nx - 1 - x, y, z, 2, 1);
      add(x, y, z, 1, nx - 1 - x, y, z, 1, 1);
      add(x, y, z, 0, x, y, nz - 1 - z, 0, 1);
      add(x, y, z, 2, x, y, nz - 2 - z, 2, -1);
      add(x, y, z, 1, x, y, nz - 1 - z, 1, 1);
      add(x, y, z, 0, z, y, x, 2, 1);
      add(x, y, z, 2, z, y, x, 0, 1);
      add(x, y, z, 1, z, y, x, 1, 1);
    }
  return { max: maximum, mean: sum / count, worst };
}

await acquireWebGPUExclusiveLock("dawn-probe", "Uniform Geometric airborne symmetric expansion A/B");
let device: GPUDevice | undefined;
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  device = managedGPUDevice(await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) }),
    { requireWorkerRealm: false });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });
  const arms = new Map<string, Map<number, Snapshot>>();
  let dims: [number, number, number] = [0, 0, 0];
  for (const mode of modes) {
    const scene = sceneDocument(getSceneDefinition("symmetric-expansion"));
    scene.numerics.fixedDt_s = scene.numerics.maxDt_s = dt;
    const solver = await WebGPUUniformReferenceSolver.createAsync(device, scene, "balanced", undefined,
      { ...uniformGeometricSolverOptions({ ...valueOverrides, airborneMomentum: mode }, scene),
        ...(process.env.FLUID_AIRBORNE_SYM_LINEAR_EXTENSION === "1" ? { sourceAwareExtension: false } : {}),
        ...(densePressureAudit ? { scratchStorageForQA: "separate" as const } : {}) }, () => {});
    const samples = new Map<number, Snapshot>(); arms.set(mode, samples);
    try {
      dims = [solver.info.nx, solver.info.ny, solver.info.nz];
      assert.equal(dims[0], dims[2]);
      const capture = async (frame: number) => {
        await device!.queue.onSubmittedWorkDone();
        const audit = solver.symmetryStageAuditTextures!;
        const entries = [
          ["volume", solver.volumeTexture, 1], ["velocity", solver.velocityTexture, 4],

          ["authority", audit.extrapolationDensityAuthority, 1],
          ["preVelocity", audit.preExtrapolationVelocity, 4],
          ["prediction", audit.velocityPrediction, 4],
          ["advection", audit.velocityAdvection, 4],
          ["projection", audit.pressureProjection, 4],
        ] as const;
        const sample: Snapshot = {};
        for (const [name, texture, components] of entries) {
          sample[name] = await read(device!, texture, components, dims);
        }
        const hd: [number, number, number] = [dims[0]+2,dims[1]+2,dims[2]+2];
        const halo = await read(device!, solver.extrapolatedVelocityTexture, 4, hd);
        sample.extrapolated = new Float32Array(dims[0]*dims[1]*dims[2]*4);
        for (let z=0;z<dims[2];z++) for(let y=0;y<dims[1];y++) for(let x=0;x<dims[0];x++) {
          const src = 4*((x+1)+hd[0]*((y+1)+hd[1]*(z+1)));
          sample.extrapolated.set(halo.subarray(src,src+4),4*(x+dims[0]*(y+dims[1]*z)));
        }
        sample.phi = await read(device!, solver.vertexPhiTexture!, 1,
          [dims[0] + 1, dims[1] + 1, dims[2] + 1]);
        sample.phiAdvected = await read(device!, solver.advectedVertexPhiTexture!, 1,
          [dims[0] + 1, dims[1] + 1, dims[2] + 1]);
        const pressureFields = solver.physicsFieldsForQA;
        sample.pressure = await read(device!, pressureFields.pressure, 1,
          pressureFields.latticeDimensions);
        if (densePressureAudit) {
          const multigrid = (solver as unknown as { pressureMultigrid: {
            levels: readonly { rhs: readonly [GPUTexture, GPUTexture];
              phi: readonly [GPUTexture, GPUTexture]; volume: readonly [GPUTexture, GPUTexture];
              coefficients: GPUTexture }[];
            setupRhsSnapshot?: readonly [GPUTexture, GPUTexture] } }).pressureMultigrid;
          const finest = multigrid.levels[0]!;
          for (const [name, texture, components] of [
            ["rhsA", finest.rhs[0], 1], ["rhsB", finest.rhs[1], 1],
            ["pressurePhiA", finest.phi[0], 1], ["pressurePhiB", finest.phi[1], 1],
            ["pressureVolume", finest.volume[0], 4],
            ["pressureCoefficients", finest.coefficients, 4],
            ...(multigrid.setupRhsSnapshot ? [
              ["setupRhsA", multigrid.setupRhsSnapshot[0], 1] as const,
              ["setupRhsB", multigrid.setupRhsSnapshot[1], 1] as const,
            ] : []),
          ] as const) sample[name] = await read(device!, texture, components,
            pressureFields.latticeDimensions);
        }
        if (process.env.FLUID_AIRBORNE_SYM_DUMP === "1" && [2,8,9,10,24,30].includes(frame)) {
          const extension = (solver as unknown as {velocityExtrapolator: {
            resolvedValues: GPUTexture; hierarchyLevels: {dims: [number,number,number]; down:GPUTexture; up:GPUTexture}[]
          }}).velocityExtrapolator;
          const resolved = await read(device!, extension.resolvedValues, 4, dims);
          await writeFile(`/tmp/uniform-extension-resolved-${mode}-${frame}.json`, JSON.stringify({dims, values:Array.from(resolved), authority:Array.from(sample.authority!), preVelocity:Array.from(sample.preVelocity!), phi:Array.from(sample.phi!), phiAdvected:Array.from(sample.phiAdvected!)}));
          const hierarchy = [];
          for (const level of extension.hierarchyLevels) {
            const down = await read(device!,level.down,4,level.dims), up = await read(device!,level.up,4,level.dims);
            hierarchy.push({dims:level.dims, down:physicalFaceD4(down,level.dims),up:physicalFaceD4(up,level.dims)});
          }
          console.error(JSON.stringify({frame, resolved:physicalFaceD4(resolved,dims), hierarchy}));
        }
        samples.set(frame, sample);
        const measured=metrics(sample,dims);
        if(process.env.FLUID_AIRBORNE_ASSERT_SYMMETRY === "1" && mode === "off" && frame<=30){
          assert.ok(measured.d4Mean<1e-4,`frame ${frame}: mean volume symmetry error ${measured.d4Mean}`);
          assert.equal(measured.heightD4,0,`frame ${frame}: liquid column-height symmetry`);
        }
        console.log(JSON.stringify({ mode, frame, time: frame * dt, ...measured,
          symmetry: { authority: scalarD4(sample.authority!,dims), phi: scalarD4(sample.phi!, [dims[0] + 1, dims[1] + 1, dims[2] + 1]),
            phiAdvected: scalarD4(sample.phiAdvected!, [dims[0] + 1, dims[1] + 1, dims[2] + 1]),
            pressure: scalarD4(sample.pressure!, solver.physicsFieldsForQA.latticeDimensions, true),
            ...(densePressureAudit ? Object.fromEntries(["rhsA", "rhsB", "setupRhsA", "setupRhsB", "pressurePhiA", "pressurePhiB"]
              .map(name => [name, scalarD4(sample[name]!, solver.physicsFieldsForQA.latticeDimensions)])) : {}),
            preVelocity: physicalFaceD4(sample.preVelocity!, dims),
            extrapolated: physicalFaceD4(sample.extrapolated!, dims),
            prediction: physicalFaceD4(sample.prediction!, dims),
            advection: physicalFaceD4(sample.advection!, dims),
            projection: physicalFaceD4(sample.projection!, dims) },
          ...(densePressureAudit ? { pressureRanges: Object.fromEntries(
            ["pressure", "setupRhsA", "setupRhsB", "rhsA", "rhsB"]
              .map(name => [name, range(sample[name]!)])) } : {}),
          ...([2, 5, 12, 20, 24].includes(frame) ? { largestSymmetryErrors: largestSymmetryErrors(sample.volume!, dims) } : {}) }));
      };
      if (checkpoints.has(0)) await capture(0);
      for (let frame = 1; frame <= frames; frame++) {
        assert.ok(solver.advanceTo(frame * dt, []));
        if (checkpoints.has(frame)) await capture(frame);
      }
    } finally { solver.destroy(); }
  }
  for (const frame of modes.length === 2 ? checkpoints : []) {
    const off = arms.get("off")!.get(frame), on = arms.get("on")!.get(frame);
    if (!off || !on) continue;
    const vector = (name: string) => ["velocity", "extrapolated", "preVelocity", "prediction", "advection", "projection"].includes(name);
    const stages = Object.fromEntries(Object.keys(off).map(name =>
      [name, difference(on[name]!, off[name]!, vector(name) ? 4 : 1)]));
    const firstChanged = Object.keys(stages).filter(name => stages[name]!.max > 0);
    const context = frame === 24 || frame === 25 ? (() => {
      const [nx, ny] = dims;
      const cells = [[16, 1, 15], [16, 2, 15], [16, 1, 16], [16, 2, 16],
        [15, 2, 15], [15, 2, 16]];
      const phiCenter = (sample: Snapshot, x: number, y: number, z: number) => {
        let sum = 0;
        for (let dz = 0; dz <= 1; dz++) for (let dy = 0; dy <= 1; dy++)
          for (let dx = 0; dx <= 1; dx++) sum += sample.phi![x + dx + (nx + 1) * (y + dy + (ny + 1) * (z + dz))]!;
        return sum / 8;
      };
      return cells.map(([x, y, z]) => ({ xyz: [x, y, z],
        off: { volume: off.volume![x + nx * (y + ny * z)]!, phi: phiCenter(off, x, y, z) },
        on: { volume: on.volume![x + nx * (y + ny * z)]!, phi: phiCenter(on, x, y, z) } }));
    })() : undefined;
    console.log(JSON.stringify({ comparison: true, frame, time: frame * dt, stages, context,
      ...(firstChanged.length > 0 && frame <= 30 ? {
        largest: Object.fromEntries(firstChanged.map(name => [name,
          largestDifferences(on[name]!, off[name]!, dims, vector(name) ? 4 : 1)])),
      } : {}) }));
  }
  assert.deepEqual(errors, []);
} finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
