#!/usr/bin/env node
/** Paired FPP1 vs immutable HEAD publisher byte oracle; no runtime fallback. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { createMinimalPowerDamBreak64Scene } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";
import { SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER as H,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_FLAG as F,
  SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS } from
  "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";

const argument = (name: string, fallback: number): number => {
  const prefix = `--${name}=`;
  const raw = process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
  const value = Number(raw ?? fallback);
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${name} must be positive`);
  return value;
};
const steps = argument("steps", 5);
const brickFineResolution = argument("brick-fine", 16) as 4 | 8 | 16;
const requirePhysicalExact = !process.argv.includes("--allow-physical-drift");
if (![4, 8, 16].includes(brickFineResolution)) {
  throw new RangeError("brick-fine must be 4, 8, or 16");
}
const out = resolve(process.argv.find((value) => value.startsWith("--out="))?.slice(6)
  ?? "artifacts/sparse-cm12-fpp1-byte-exact-dam64.json");
const modulePath = process.env.WEBGPU_NODE_MODULE ?? `${process.cwd()}/node_modules/webgpu/index.js`;
const backend = process.env.FLUID_WEBGPU_BACKEND ?? "metal";
const activityPolicy = Object.freeze({
  activitySignals: true, finestTravelCells: 1, fourTravelCells: 0.5,
  twoTravelCells: 0.25, thinFeatureCells: 2, thinFeatureDensity: 0,
  residencyDensity: 0.005, residencyMassFineCells: 1,
  surfaceDensityMinimum: 0.05, surfaceDensityMaximum: 0.95,
  detailTolerance: 0.08, frontLookaheadSteps: 4, topologyCadenceSteps: 1,
  prepareBricksPerFrame: 64, promoteEpochs: 2, demoteEpochs: 1,
  promoteScore: 160 / 255, demoteScore: 96 / 255, emergencyScore: 224 / 255,
});

const sha = (bytes: Uint8Array): string => createHash("sha256").update(bytes).digest("hex");
const bytes = (view: ArrayBufferView): Uint8Array =>
  new Uint8Array(view.buffer, view.byteOffset, view.byteLength);

async function readBuffer(device: GPUDevice, source: GPUBuffer,
  sourceOffset: number, size: number): Promise<Uint8Array> {
  const readback = device.createBuffer({ size,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, sourceOffset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint8Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

interface Capture {
  readonly presentation: Uint8Array;
  readonly metadata: Uint8Array;
  readonly physical: Readonly<Record<string, string>>;
  readonly fppHeader?: Uint32Array;
}

async function capture(device: GPUDevice, oracle: boolean): Promise<Capture> {
  const scene = createMinimalPowerDamBreak64Scene();
  const options = {
    resolutionMode: "adaptive", brickFineResolution,
    presentationPageResolution: brickFineResolution, surfaceFineRings: 8,
    activityPolicy, timeStep: "paper",
  } as const;
  const create = oracle
    ? WebGPUAdaptiveMassSolver.createPresentationPublisherOracleForQA
    : WebGPUAdaptiveMassSolver.createAsync;
  const solver = await create.call(WebGPUAdaptiveMassSolver,
    device, scene, "balanced", undefined, options, () => {});
  try {
    for (let step = 1; step <= steps; step += 1) {
      while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) {
        await new Promise(setImmediate);
      }
    }
    await device.queue.onSubmittedWorkDone();
    const source = solver.globalFineLevelSetSource;
    const presentation = await readBuffer(device, source.samples, 0,
      source.plan.payloadCapacityBytes);
    const metadata = await readBuffer(device, source.metadata, 0, source.metadata.size);
    const fields = await solver.readDiagnosticFields();
    const physical = Object.fromEntries(Object.entries(fields).map(([name, view]) =>
      [name, sha(bytes(view as ArrayBufferView))]));
    if (oracle) return { presentation, metadata, physical };
    const sparse = solver.sparseAdaptiveGridSource;
    const framePlan = sparse.framePlan;
    assert(framePlan, "production solver did not expose FPL1");
    const fppBase = framePlan.layout.totalWords;
    const headerBytes = await readBuffer(device, sparse.activity.buffer,
      (sparse.activity.offset ?? 0) + 4 * fppBase,
      4 * SPARSE_CM12_FRAME_PLAN_PRESENTATION_HEADER_WORDS);
    return { presentation, metadata, physical,
      fppHeader: new Uint32Array(headerBytes.buffer) };
  } finally {
    solver.destroy();
  }
}

function firstMismatch(left: Uint8Array, right: Uint8Array): number {
  const count = Math.min(left.length, right.length);
  for (let index = 0; index < count; index += 1) if (left[index] !== right[index]) return index;
  return left.length === right.length ? -1 : count;
}

await acquireWebGPUExclusiveLock("presentation-byte-exact", "FPP1/HEAD paired dam smoke");
let device: GPUDevice | undefined;
try {
  const dawn = await import(modulePath) as { create: (flags: string[]) => GPU,
    globals: Record<string, unknown> };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${backend}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert(adapter, "no Dawn adapter");
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();validationErrors.push(event.error.message);
  });
  const production = await capture(device, false);
  const oracle = await capture(device, true);
  const mismatch = firstMismatch(production.presentation, oracle.presentation);
  assert.equal(mismatch, -1, `presentation byte mismatch at ${mismatch} (word ${mismatch >>> 2})`);
  assert.equal(firstMismatch(production.metadata, oracle.metadata), -1,
    "presentation metadata differs from immutable QA publisher");
  const physicalExact = JSON.stringify(production.physical) === JSON.stringify(oracle.physical);
  if (requirePhysicalExact) {
    assert.deepEqual(production.physical, oracle.physical, "paired physical fields changed");
  }
  assert.deepEqual(validationErrors, [], "Dawn validation errors");
  const header = production.fppHeader!;
  assert.equal(header[H.faultCode], 0, "FPP1 global fault");
  assert.equal(header[H.coverageFaultCount], 0, "FPP1 coverage fault");
  assert.equal(header[H.omittedPageCount], 0, "FPP1 omitted a scheduled page");
  assert.equal(header[H.dirtyPageCount], header[H.executedPageCount],
    "FPP1 scheduled/executed page mismatch");
  assert.equal(header[H.executedPageCount], header[H.publishedPageCount],
    "FPP1 candidate/commit page mismatch");
  assert.notEqual(header[H.flags] & F.executionComplete, 0,
    "FPP1 did not publish an execution-complete receipt");
  const metadata = new Uint32Array(production.metadata.buffer,
    production.metadata.byteOffset, production.metadata.byteLength / 4);
  let metadataGenerationInvalid = 0;
  for (let page = 0; page < metadata.length / 4; page += 1) {
    if (metadata[4 * page] !== page || metadata[4 * page + 2] !== 1) {
      metadataGenerationInvalid += 1;
    }
  }
  assert.equal(metadataGenerationInvalid, 0, "FPP1 left invalid renderer page generations");
  const receipt = { kind: "sparse-cm12-fpp1-byte-exact",
    steps, brickFineResolution, presentationPageResolution: brickFineResolution,
    physicalExact,
    presentationBytes: production.presentation.byteLength,
    presentationSha256: sha(production.presentation), physical: production.physical,
    fpp: { dirtyPages: header[H.dirtyPageCount], omittedPages: header[H.omittedPageCount],
      executedPages: header[H.executedPageCount], publishedPages: header[H.publishedPageCount],
      executedTiles: header[H.executedTileCount], acceptedGeneration: header[H.acceptedGeneration],
      generationReceipt: header[H.generationReceipt], flags: header[H.flags],
      faultCode: header[H.faultCode], firstFaultBrick: header[H.firstFaultBrick],
      firstFaultTile: header[H.firstFaultTile], firstFaultCause: header[H.firstFaultCause],
      metadataGenerationInvalid } };
  await mkdir(dirname(out), { recursive: true });
  await writeFile(out, `${JSON.stringify(receipt, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify({ ...receipt, out }, null, 2)}\n`);
} finally {
  device?.destroy();
  await releaseWebGPUExclusiveLock();
}
