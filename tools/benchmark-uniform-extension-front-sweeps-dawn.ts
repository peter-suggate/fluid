/**
 * Uniform Geometric Sec. 3.3 velocity extension: how many FIM front sweeps the
 * two-cell accurate band actually needs, what a smaller sweep budget changes in
 * one step from an identical input state, and what each budget costs.
 *
 * Measurement only. Nothing under lib/ is modified; the tool reaches the
 * extrapolator, its convergence diagnostics and the solver's persistent fields
 * through a structural cast, exactly as the 4h tile-work benchmark does.
 *
 * Three phases, per scene:
 *
 *  1. exactness  One N=16 reference trajectory. After every step the four
 *                convergence words (activeA, activeB, latestParity,
 *                executedPasses) are read back outside timing. `executedPasses`
 *                counts the update sweeps that ran with a non-empty active
 *                list; every later sweep dispatches zero workgroups and writes
 *                nothing, so a budget equal to the per-frame maximum is
 *                bit-identical to 16 on that same input.
 *
 *  2. sensitivity At selected frames the state the step starts from is copied
 *                into a holder solver, and one single step is replayed from
 *                that identical state at each sweep budget by a third solver.
 *                Full trajectories are chaotic on this solver, so nothing here
 *                compares trajectories: every number is one step from one
 *                shared input. The N=16 replays double as the fidelity assert
 *                (the packed shell must be bit-identical to the reference step)
 *                and as the noise floor (conservative transport's CAS float
 *                sums make V itself not bit-reproducible).
 *
 *  3. cost       Six 30-frame arms in one process, ordered 16,1,8,2,4,16 so the
 *                two N=16 arms bracket any drift. GPU timestamps bracket the
 *                whole extension stage; the frame is a queue-fenced wall.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/benchmark-uniform-extension-front-sweeps-dawn.ts \
 *     [--scene=<id>] [--only=exactness|sensitivity|cost] [--out=<path.json>]
 */
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";
import { resolve } from "node:path";
import { mkdirSync, writeFileSync } from "node:fs";
import { createProcessRetainedDawnGPU } from "../lib/harness/node-dawn-provider";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock, readWebGPUExclusiveLockHolder,
} from "../lib/harness/webgpu-smoke-isolation";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { uniformVolumeMethod } from "../lib/methods/uniform/uniform-volume-method";
import { resolveMethodValues } from "../lib/core/method-contract";
import { CM12_LIQUID_ISOVALUE } from "../lib/core/cm12-numerics";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";

const argument = (name: string, fallback: string) =>
  process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;

const SCENES = argument("scene", "minimal-power-dam-break-64,large-power-dam-break").split(",");
const ONLY = argument("only", "all");
const OUT = argument("out", "docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.json");
const REFERENCE_FRAMES = Number(argument("reference-frames", "60"));
const SNAPSHOT_FRAMES = argument("snapshots", "5,15,30,45,60").split(",").map(Number);
/** Two N=16 replays: the first is the fidelity assert, the second the floor.
 * N=3 is not in the asked-for set; it is one extra one-step replay that
 * separates "two accurate rings" from "converged", which is the difference
 * between recommending 4 and recommending 3. */
const REPLAY_BUDGETS = [16, 16, 8, 4, 3, 2, 1];
const COST_BUDGETS = [16, 1, 8, 2, 4, 16];
const COST_FRAMES = Number(argument("cost-frames", "30"));
const COST_WARMUP = 3;

const delay = (ms: number) => new Promise(done => setTimeout(done, ms));
const median = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  return (s[Math.floor((s.length - 1) / 2)]! + s[Math.floor(s.length / 2)]!) / 2;
};
const quantile = (xs: number[], q: number) => {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))))]!;
};

interface ExtrapolatorAccess {
  setFrontPasses(passes: number): void;
  readonly frontPasses: number;
  readonly activeFrontPassCeiling: number;
  readonly convergenceDiagnostics: GPUBuffer;
}

interface SolverAccess {
  velocityA: GPUTexture; velocityB: GPUTexture; velocityC: GPUTexture; velocityD: GPUTexture;
  pressureA: GPUTexture; pressureB: GPUTexture;
  volumeA: GPUTexture; volumeB: GPUTexture;
  surfaceA: GPUTexture; surfaceB: GPUTexture;
  gammaA: GPUTexture; gammaB: GPUTexture;
  heightA: GPUTexture; heightB: GPUTexture; terrainTexture: GPUTexture;
  transportA: GPUTexture; transportB: GPUTexture;
  vertexPhiTexture?: GPUTexture; vertexPhiScratch?: GPUTexture;
  boundaryVelocityA: GPUBuffer; boundaryVelocityB: GPUBuffer;
  boundaryVelocityC: GPUBuffer; boundaryVelocityD: GPUBuffer;
  reductions: GPUBuffer; conditioningScratch: GPUBuffer;
  activeRegion: GPUBuffer; activeScratch: GPUBuffer; rigidExchange: GPUBuffer;
  pressureMultigrid: { pressureTexture: GPUTexture };
  velocityExtrapolator: ExtrapolatorAccess;
  lastTime: number;
  referenceVolumeCells: number;
  encodeVelocityExtrapolation(e: GPUCommandEncoder, predicted: boolean, seam?: unknown): void;
}

const STATE_TEXTURES = [
  "velocityA", "velocityB", "velocityC", "velocityD", "pressureA", "pressureB",
  "volumeA", "volumeB", "surfaceA", "surfaceB", "gammaA", "gammaB",
  "heightA", "heightB", "terrainTexture", "transportA", "transportB",
  "vertexPhiTexture", "vertexPhiScratch",
] as const;
const STATE_BUFFERS = [
  "boundaryVelocityA", "boundaryVelocityB", "boundaryVelocityC", "boundaryVelocityD",
  "reductions", "conditioningScratch", "activeRegion", "activeScratch", "rigidExchange",
] as const;

const components = (format: GPUTextureFormat) =>
  format === "rgba32float" ? 4 : format === "rg32float" ? 2 : 1;

async function readTexture(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const width = texture.width * components(texture.format);
  const row = Math.ceil(width * 4 / 256) * 256;
  const layers = texture.height * texture.depthOrArrayLayers;
  const buffer = device.createBuffer({
    size: row * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: texture.height },
      [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const source = new Float32Array(buffer.getMappedRange());
    const out = new Float32Array(width * layers);
    for (let i = 0; i < layers; i += 1) out.set(source.subarray(i * row / 4, i * row / 4 + width), i * width);
    return out;
  } finally { buffer.unmap(); buffer.destroy(); }
}

async function readWords(device: GPUDevice, buffer: GPUBuffer, bytes: number): Promise<Uint32Array> {
  const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    return new Uint32Array(staging.getMappedRange().slice(0));
  } finally { staging.unmap(); staging.destroy(); }
}

function copySolverState(device: GPUDevice, from: SolverAccess, to: SolverAccess): void {
  const encoder = device.createCommandEncoder({ label: "extension sweeps state copy" });
  for (const field of STATE_TEXTURES) {
    const source = from[field], target = to[field];
    if (!source || !target) continue;
    assert.equal(source.width, target.width);
    encoder.copyTextureToTexture({ texture: source }, { texture: target },
      [source.width, source.height, source.depthOrArrayLayers]);
  }
  const pressure = from.pressureMultigrid.pressureTexture, pressureTo = to.pressureMultigrid.pressureTexture;
  encoder.copyTextureToTexture({ texture: pressure }, { texture: pressureTo },
    [pressure.width, pressure.height, pressure.depthOrArrayLayers]);
  for (const field of STATE_BUFFERS) {
    const source = from[field], target = to[field];
    assert.equal(source.size, target.size);
    encoder.copyBufferToBuffer(source, 0, target, 0, source.size);
  }
  device.queue.submit([encoder.finish()]);
  to.lastTime = from.lastTime;
  to.referenceVolumeCells = from.referenceVolumeCells;
}

interface Lattice { nx: number; ny: number; nz: number; hx: number; hy: number; hz: number; h: number }

/** Signed distance at a MAC face centre, trilinear on the vertex phi lattice. */
function facePhi(phi: Float32Array, l: Lattice, x: number, y: number, z: number, axis: number): number {
  const sx = x + 0.5 + (axis === 0 ? 0.5 : 0);
  const sy = y + 0.5 + (axis === 1 ? 0.5 : 0);
  const sz = z + 0.5 + (axis === 2 ? 0.5 : 0);
  const stride = l.nx + 1, plane = stride * (l.ny + 1);
  const at = (i: number, j: number, k: number) => phi[
    Math.min(l.nx, Math.max(0, i)) + stride * Math.min(l.ny, Math.max(0, j)) + plane * Math.min(l.nz, Math.max(0, k))
  ]!;
  const x0 = Math.floor(sx), y0 = Math.floor(sy), z0 = Math.floor(sz);
  const fx = sx - x0, fy = sy - y0, fz = sz - z0;
  let value = 0;
  for (let k = 0; k <= 1; k += 1) for (let j = 0; j <= 1; j += 1) for (let i = 0; i <= 1; i += 1) {
    const w = (i ? fx : 1 - fx) * (j ? fy : 1 - fy) * (k ? fz : 1 - fz);
    if (w > 0) value += w * at(x0 + i, y0 + j, z0 + k);
  }
  return value;
}

const BUCKETS = [[0, 1], [1, 2], [2, 3], [3, 5], [5, 10], [10, Infinity]] as const;
const bucketLabels = BUCKETS.map(([a, b]) => b === Infinity ? `>${a}` : `${a}-${b}`);
const bucketOf = (cells: number) => {
  for (let i = 0; i < BUCKETS.length; i += 1) if (cells < BUCKETS[i]![1]) return i;
  return BUCKETS.length - 1;
};

/** Differences are accumulated already normalised, so the tail counts are
 * comparable across frames and scenes. */
const THRESHOLDS = [1e-3, 1e-2, 1e-1, 1] as const;
interface Accumulator { max: number; sumSquares: number; count: number; exceed: number[] }
const accumulator = (): Accumulator => ({ max: 0, sumSquares: 0, count: 0, exceed: THRESHOLDS.map(() => 0) });
const add = (a: Accumulator, value: number) => {
  a.max = Math.max(a.max, value); a.sumSquares += value * value; a.count += 1;
  for (let i = 0; i < THRESHOLDS.length; i += 1) if (value > THRESHOLDS[i]!) a.exceed[i]! += 1;
};
const summary = (a: Accumulator) => ({
  count: a.count,
  max: a.max,
  rms: a.count > 0 ? Math.sqrt(a.sumSquares / a.count) : 0,
  exceed: Object.fromEntries(THRESHOLDS.map((t, i) => [`>${t}`, a.exceed[i]!])),
});

interface Snapshot {
  frame: number;
  requestedTime_s: number;
  phiIn: Float32Array;
  volumeIn: Float32Array;
  velocityIn: Float32Array;
  shellReference: Float32Array;
  phiReference: Float32Array;
  volumeReference: Float32Array;
  maxSourceFaceSpeed: number;
  maxShellSpeed: number;
  maxLiquidCellSpeed: number;
  liquidCells: number;
  faceBuckets: number[];
  referenceDigests: { shell: string; phi: string; volume: string };
}

/** Bucket index per (cell, component), precomputed from the input vertex phi. */
function faceBucketMap(phi: Float32Array, l: Lattice): Int8Array {
  const map = new Int8Array(l.nx * l.ny * l.nz * 3);
  for (let z = 0; z < l.nz; z += 1) for (let y = 0; y < l.ny; y += 1) for (let x = 0; x < l.nx; x += 1) {
    for (let axis = 0; axis < 3; axis += 1) {
      const distance = Math.max(0, facePhi(phi, l, x, y, z, axis)) / l.h;
      map[(x + l.nx * (y + l.ny * z)) * 3 + axis] = bucketOf(distance);
    }
  }
  return map;
}

function shellDifference(a: Float32Array, b: Float32Array, l: Lattice, buckets: Int8Array, scale: number) {
  const overall = accumulator();
  const perBucket = BUCKETS.map(() => accumulator());
  const stride = l.nx + 2, plane = stride * (l.ny + 2);
  let openFaces = 0, changedFaces = 0;
  for (let z = 0; z < l.nz; z += 1) for (let y = 0; y < l.ny; y += 1) for (let x = 0; x < l.nx; x += 1) {
    const base = ((x + 1) + stride * (y + 1) + plane * (z + 1)) * 4;
    const openMask = Math.round(b[base + 3]!) >>> 3;
    for (let axis = 0; axis < 3; axis += 1) {
      if ((openMask & (1 << axis)) === 0) continue;
      openFaces += 1;
      const error = Math.abs(a[base + axis]! - b[base + axis]!) / scale;
      if (error > 0) changedFaces += 1;
      add(overall, error);
      add(perBucket[buckets[(x + l.nx * (y + l.ny * z)) * 3 + axis]!]!, error);
    }
  }
  return {
    openFaces, changedFaces, changedFraction: openFaces > 0 ? changedFaces / openFaces : 0,
    overall: summary(overall),
    byDistance: Object.fromEntries(bucketLabels.map((label, i) => [label, summary(perBucket[i]!)])),
  };
}

function phiDifference(a: Float32Array, b: Float32Array, l: Lattice) {
  const band = accumulator(), all = accumulator();
  const limit = 2 * l.h;
  let changedBand = 0;
  for (let i = 0; i < b.length; i += 1) {
    const error = Math.abs(a[i]! - b[i]!) / l.h;
    add(all, error);
    if (Math.abs(b[i]!) < limit) { add(band, error); if (error > 0) changedBand += 1; }
  }
  return { band: summary(band), changedBandVertices: changedBand, everywhere: summary(all) };
}

/** FNV-1a over the raw bytes: equal digests mean a bit-identical field. */
function digest(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let hash = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i += 1) {
    hash ^= bytes[i]!;
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash.toString(16).padStart(8, "0");
}

function volumeDifference(a: Float32Array, b: Float32Array) {
  let maximum = 0, sumA = 0, sumB = 0, changed = 0, nonFinite = 0;
  let maxArm = -Infinity, maxReference = -Infinity, overfullArm = 0, overfullReference = 0;
  // This configuration (liquid capacity balancing off by default) already
  // leaves a population of grossly overfull cells in the N=16 reference
  // itself, and they dominate a plain max. Report the max over the ordinary
  // cells beside it.
  let maximumOrdinary = 0;
  for (let i = 0; i < b.length; i += 1) {
    if (!Number.isFinite(a[i]!)) nonFinite += 1;
    const error = Math.abs(a[i]! - b[i]!);
    if (error > 0) changed += 1;
    if (a[i]! <= 1.5 && b[i]! <= 1.5) maximumOrdinary = Math.max(maximumOrdinary, error);
    maximum = Math.max(maximum, error);
    maxArm = Math.max(maxArm, a[i]!); maxReference = Math.max(maxReference, b[i]!);
    if (a[i]! > 1.5) overfullArm += 1;
    if (b[i]! > 1.5) overfullReference += 1;
    sumA += a[i]!; sumB += b[i]!;
  }
  return { maxCell: maximum, maxCellExcludingOverfull: maximumOrdinary,
    changedCells: changed, nonFiniteCells: nonFinite,
    maxCellValueArm: maxArm, maxCellValueReference: maxReference,
    overfullCellsArm: overfullArm, overfullCellsReference: overfullReference,
    totalArm: sumA, totalReference: sumB, totalDelta: sumA - sumB };
}

async function acquireWithWait(): Promise<void> {
  for (let attempt = 0; ; attempt += 1) {
    const holder = await readWebGPUExclusiveLockHolder();
    if (holder) {
      if (!holder.alive) throw new Error(`GPU lock held by a dead owner (${holder.description}); clear it by hand`);
      if (attempt % 4 === 0) console.error(`waiting for GPU lock: ${holder.description}`);
      await delay(15_000);
      continue;
    }
    try { await acquireWebGPUExclusiveLock("dawn-probe", "uniform extension front-sweep budget"); return; }
    catch { await delay(5_000); }
  }
}

await acquireWithWait();
let device: GPUDevice | undefined;
const report: Record<string, unknown> = {
  tool: "tools/benchmark-uniform-extension-front-sweeps-dawn.ts",
  generatedAt: new Date().toISOString(),
  method: "uniform-volume", quality: "balanced", values: "resolveMethodValues(uniformVolumeMethod,'balanced',{})",
  referenceFrames: REFERENCE_FRAMES, snapshotFrames: SNAPSHOT_FRAMES,
  replayBudgets: REPLAY_BUDGETS, costBudgets: COST_BUDGETS, costFrames: COST_FRAMES,
  costWarmupFramesDropped: COST_WARMUP,
  scenes: {} as Record<string, unknown>,
};

try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, ["backend=metal"]);
  const adapter = await gpu.requestAdapter();
  assert.ok(adapter); assert.ok(adapter.features.has("timestamp-query"));
  device = managedGPUDevice(await adapter.requestDevice({
    requiredFeatures: ["timestamp-query"], requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  }), { requireWorkerRealm: false });
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    const typed = event as GPUUncapturedErrorEvent & { preventDefault(): void };
    typed.preventDefault(); errors.push(typed.error.message); console.error(typed.error.message);
  });

  const build = async (sceneId: string) => {
    const scene = structuredClone(sceneDocument(getSceneDefinition(sceneId)));
    const values = resolveMethodValues(uniformVolumeMethod, "balanced", {});
    const solver = await uniformVolumeMethod.createSolverAsync!(
      device!, scene, "balanced", values, undefined, () => {}) as WebGPUUniformReferenceSolver;
    return solver;
  };

  const scenesOut = report.scenes as Record<string, Record<string, unknown>>;
  for (const sceneId of SCENES) {
    const sceneOut: Record<string, unknown> = {};
    scenesOut[sceneId] = sceneOut;

    if (ONLY === "all" || ONLY === "exactness" || ONLY === "sensitivity") {
      const reference = await build(sceneId);
      const holder = await build(sceneId);
      const replay = await build(sceneId);
      const R = reference as unknown as SolverAccess;
      const H = holder as unknown as SolverAccess;
      const S = replay as unknown as SolverAccess;
      const { nx, ny, nz } = reference.info;
      const container = reference.scene.container;
      const lattice: Lattice = {
        nx, ny, nz,
        hx: container.width_m / nx, hy: container.height_m / ny, hz: container.depth_m / nz,
        h: Math.max(container.width_m / nx, container.height_m / ny, container.depth_m / nz),
      };
      sceneOut.lattice = { nx, ny, nz, cellSize_m: [lattice.hx, lattice.hy, lattice.hz],
        defaultFrontSweeps: R.velocityExtrapolator.frontPasses,
        frontPassCeiling: R.velocityExtrapolator.activeFrontPassCeiling };
      assert.equal(R.velocityExtrapolator.frontPasses, 16, "the default sweep budget must be 16");

      const convergence: { frame: number; activeA: number; activeB: number; parity: number; executed: number }[] = [];
      const snapshots: Snapshot[] = [];
      const replayRows: Record<string, unknown>[] = [];

      try {
        for (let frame = 1; frame <= REFERENCE_FRAMES; frame += 1) {
          const requested = frame / 30;
          const wantSnapshot = SNAPSHOT_FRAMES.includes(frame) && (ONLY !== "exactness");
          if (wantSnapshot) { copySolverState(device!, R, H); await device!.queue.onSubmittedWorkDone(); }
          assert.ok(reference.advanceTo(requested), `reference frame ${frame} did not advance`);
          await device!.queue.onSubmittedWorkDone();
          const words = await readWords(device!, R.velocityExtrapolator.convergenceDiagnostics, 16);
          convergence.push({ frame, activeA: words[0]!, activeB: words[1]!, parity: words[2]!, executed: words[3]! });
          if (!wantSnapshot) continue;

          const phiIn = await readTexture(device!, H.vertexPhiTexture!);
          const volumeIn = await readTexture(device!, H.volumeA);
          const velocityIn = await readTexture(device!, H.velocityA);
          const shellReference = await readTexture(device!, R.transportA);
          const phiReference = await readTexture(device!, R.vertexPhiTexture!);
          const volumeReference = await readTexture(device!, R.volumeA);

          let maxSourceFaceSpeed = 0, maxLiquidCellSpeed = 0, liquidCells = 0;
          for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
            const cell = x + nx * (y + ny * z);
            const liquid = volumeIn[cell]! > CM12_LIQUID_ISOVALUE;
            if (liquid) {
              liquidCells += 1;
              maxLiquidCellSpeed = Math.max(maxLiquidCellSpeed, Math.hypot(
                velocityIn[cell * 4]!, velocityIn[cell * 4 + 1]!, velocityIn[cell * 4 + 2]!));
            }
            for (let axis = 0; axis < 3; axis += 1) {
              const nx1 = x + (axis === 0 ? 1 : 0), ny1 = y + (axis === 1 ? 1 : 0), nz1 = z + (axis === 2 ? 1 : 0);
              const neighbour = nx1 < nx && ny1 < ny && nz1 < nz
                ? volumeIn[nx1 + nx * (ny1 + ny * nz1)]! > CM12_LIQUID_ISOVALUE : false;
              if (liquid || neighbour) {
                maxSourceFaceSpeed = Math.max(maxSourceFaceSpeed, Math.abs(velocityIn[cell * 4 + axis]!));
              }
            }
          }
          // Every extended value is a positive-weight average of seed values,
          // so the largest component in the packed shell IS the largest liquid
          // face speed the extension carries that frame. The V>0.5 census
          // above is only a diagnostic: the kernel's own source test reads the
          // rho'=rho/V authority field, not V.
          let maxShellSpeed = 0;
          for (let i = 0; i < shellReference.length; i += 4) {
            for (let axis = 0; axis < 3; axis += 1) {
              maxShellSpeed = Math.max(maxShellSpeed, Math.abs(shellReference[i + axis]!));
            }
          }
          const buckets = faceBucketMap(phiIn, lattice);
          const faceBuckets = BUCKETS.map(() => 0);
          for (const b of buckets) faceBuckets[b]! += 1;
          const snapshot: Snapshot = { frame, requestedTime_s: requested, phiIn, volumeIn, velocityIn,
            shellReference, phiReference, volumeReference, maxSourceFaceSpeed, maxShellSpeed, maxLiquidCellSpeed,
            liquidCells, faceBuckets,
            referenceDigests: { shell: digest(shellReference), phi: digest(phiReference),
              volume: digest(volumeReference) } };
          snapshots.push(snapshot);

          const savedLastTime = H.lastTime;
          for (let index = 0; index < REPLAY_BUDGETS.length; index += 1) {
            const budget = REPLAY_BUDGETS[index]!;
            copySolverState(device!, H, S);
            await device!.queue.onSubmittedWorkDone();
            S.velocityExtrapolator.setFrontPasses(budget);
            assert.equal(S.velocityExtrapolator.frontPasses, budget);
            assert.equal(S.lastTime, savedLastTime);
            assert.ok(replay.advanceTo(requested), `replay N=${budget} frame ${frame} did not advance`);
            await device!.queue.onSubmittedWorkDone();
            const armWords = await readWords(device!, S.velocityExtrapolator.convergenceDiagnostics, 16);
            const shell = await readTexture(device!, S.transportA);
            const phiOut = await readTexture(device!, S.vertexPhiTexture!);
            const volumeOut = await readTexture(device!, S.volumeA);
            let identicalShell = shell.length === shellReference.length;
            if (identicalShell) for (let i = 0; i < shell.length; i += 1) {
              if (shell[i] !== shellReference[i]) { identicalShell = false; break; }
            }
            let identicalPhi = phiOut.length === phiReference.length;
            if (identicalPhi) for (let i = 0; i < phiOut.length; i += 1) {
              if (phiOut[i] !== phiReference[i]) { identicalPhi = false; break; }
            }
            const scale = Math.max(1e-12, snapshot.maxShellSpeed);
            replayRows.push({
              frame, budget, replayIndex: index,
              executedPasses: armWords[3]!, terminalActive: armWords[0]! + armWords[1]!,
              normalisedBy: snapshot.maxShellSpeed,
              maxSourceFaceSpeed: snapshot.maxSourceFaceSpeed,
              maxLiquidCellSpeed: snapshot.maxLiquidCellSpeed, liquidCells: snapshot.liquidCells,
              shellBitIdentical: identicalShell, phiBitIdentical: identicalPhi,
              shellDigest: digest(shell), phiDigest: digest(phiOut), volumeDigest: digest(volumeOut),
              shell: shellDifference(shell, shellReference, lattice, buckets, scale),
              phi: phiDifference(phiOut, phiReference, lattice),
              volume: volumeDifference(volumeOut, volumeReference),
            });
            console.error(`${sceneId} frame ${frame} N=${budget}: shellIdentical=${identicalShell} phiIdentical=${identicalPhi}`);
          }
          // Free the largest arrays; the per-snapshot rows keep the metrics.
          snapshot.phiIn = new Float32Array(0); snapshot.volumeIn = new Float32Array(0);
          snapshot.velocityIn = new Float32Array(0); snapshot.shellReference = new Float32Array(0);
          snapshot.phiReference = new Float32Array(0); snapshot.volumeReference = new Float32Array(0);
        }
        const executed = convergence.map(c => c.executed);
        const histogram: Record<string, number> = {};
        for (const value of executed) histogram[String(value)] = (histogram[String(value)] ?? 0) + 1;
        sceneOut.exactness = {
          frames: convergence.length,
          executedPassHistogram: histogram,
          maxExecutedPasses: Math.max(...executed),
          minExecutedPasses: Math.min(...executed),
          medianExecutedPasses: median(executed),
          framesWithNonzeroTerminalActive: convergence.filter(c => c.activeA + c.activeB > 0).length,
          maxTerminalActive: Math.max(...convergence.map(c => c.activeA + c.activeB)),
          smallestBitIdenticalBudget: Math.max(...executed),
          perFrame: convergence,
        };
        sceneOut.sensitivity = {
          buckets: bucketLabels,
          bucketBasis: "max(0, phi) at the MAC face centre from the input vertex phi, in cells",
          normalisation: "max |u_component| over the N=16 packed transport shell, which equals the largest seed (liquid face) speed that frame",
          snapshots: snapshots.map(s => ({ frame: s.frame, liquidCells: s.liquidCells,
            maxShellSpeed: s.maxShellSpeed,
            maxSourceFaceSpeed: s.maxSourceFaceSpeed, maxLiquidCellSpeed: s.maxLiquidCellSpeed,
            referenceDigests: s.referenceDigests,
            openFacesByDistance: Object.fromEntries(bucketLabels.map((l, i) => [l, s.faceBuckets[i]!])) })),
          rows: replayRows,
        };
      } finally {
        reference.destroy(); holder.destroy(); replay.destroy();
      }
      writeFileSync(OUT, JSON.stringify(report, null, 2));
    }

    if (ONLY === "all" || ONLY === "cost") {
      const marker = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      const markerPipeline = await device.createComputePipelineAsync({ layout: "auto", compute: {
        module: device.createShaderModule({ code:
          "@group(0) @binding(0) var<storage,read_write> count:atomic<u32>; @compute @workgroup_size(1) fn main(){atomicAdd(&count,1u);}" }),
        entryPoint: "main" } });
      const markerGroup = device.createBindGroup({ layout: markerPipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: marker } }] });
      const arms: Record<string, unknown>[] = [];
      try {
        for (let armIndex = 0; armIndex < COST_BUDGETS.length; armIndex += 1) {
          const budget = COST_BUDGETS[armIndex]!;
          const solver = await build(sceneId);
          const access = solver as unknown as SolverAccess;
          access.velocityExtrapolator.setFrontPasses(budget);
          const query = device.createQuerySet({ type: "timestamp", count: 2 });
          const resolved = device.createBuffer({ size: 256, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
          const staging = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          const stamp = (encoder: GPUCommandEncoder, index: number) => {
            encoder.copyBufferToBuffer(access.conditioningScratch, 0, marker, 0, 4);
            const pass = encoder.beginComputePass({ timestampWrites: { querySet: query, beginningOfPassWriteIndex: index } });
            pass.setPipeline(markerPipeline); pass.setBindGroup(0, markerGroup); pass.dispatchWorkgroups(1); pass.end();
          };
          const original = access.encodeVelocityExtrapolation.bind(access);
          access.encodeVelocityExtrapolation = (encoder, predicted, seam) => {
            stamp(encoder, 0); original(encoder, predicted, seam); stamp(encoder, 1);
          };
          const samples: { frame: number; wall_ms: number; extension_ms: number }[] = [];
          try {
            for (let frame = 1; frame <= COST_FRAMES; frame += 1) {
              const start = performance.now();
              assert.ok(solver.advanceTo(frame / 30));
              await device.queue.onSubmittedWorkDone();
              const wall_ms = performance.now() - start;
              const encoder = device.createCommandEncoder();
              encoder.resolveQuerySet(query, 0, 2, resolved, 0);
              encoder.copyBufferToBuffer(resolved, 0, staging, 0, 16);
              device.queue.submit([encoder.finish()]);
              await staging.mapAsync(GPUMapMode.READ);
              const times = new BigUint64Array(staging.getMappedRange());
              assert.ok(times[0]! > 0n && times[1]! >= times[0]!, `invalid timestamps ${times[0]} ${times[1]}`);
              const extension_ms = Number(times[1]! - times[0]!) / 1e6;
              staging.unmap();
              samples.push({ frame, wall_ms, extension_ms });
            }
            const measured = samples.slice(COST_WARMUP);
            const walls = measured.map(s => s.wall_ms), extensions = measured.map(s => s.extension_ms);
            const stats = (xs: number[]) => ({ median: median(xs), p25: quantile(xs, 0.25), p75: quantile(xs, 0.75),
              min: Math.min(...xs), max: Math.max(...xs), mean: xs.reduce((a, b) => a + b, 0) / xs.length });
            const row = { armIndex, budget, frames: COST_FRAMES, measuredFrames: measured.length,
              wall_ms: stats(walls), extension_ms: stats(extensions), samples };
            arms.push(row);
            console.error(`${sceneId} cost arm ${armIndex} N=${budget}: wall ${median(walls).toFixed(2)} ms, extension ${median(extensions).toFixed(3)} ms`);
          } finally {
            solver.destroy(); query.destroy(); resolved.destroy(); staging.destroy();
          }
        }
      } finally { marker.destroy(); }
      sceneOut.cost = { order: COST_BUDGETS, arms };
      writeFileSync(OUT, JSON.stringify(report, null, 2));
    }
    assert.deepEqual(errors, []);
  }
  report.uncapturedErrors = errors;
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
  mkdirSync("docs/benchmarks", { recursive: true });
  writeFileSync(OUT, JSON.stringify(report, null, 2));
  console.error(`wrote ${OUT}`);
}
