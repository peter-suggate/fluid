#!/usr/bin/env node
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { webgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";

type Variant = "restart" | "continuation";

const INVALID = 0xffff_ffff;
const SVO_STATUS_MISS_FOR_HOST = 0;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const width = positiveInteger(process.env.FLUID_SVO_CONTINUATION_WIDTH ?? "128", "width");
const height = positiveInteger(process.env.FLUID_SVO_CONTINUATION_HEIGHT ?? "128", "height");
const depth = positiveInteger(process.env.FLUID_SVO_CONTINUATION_DEPTH ?? "5", "depth");
const warmups = positiveInteger(process.env.FLUID_SVO_CONTINUATION_WARMUPS ?? "4", "warmups");
const cycles = positiveInteger(process.env.FLUID_SVO_CONTINUATION_CYCLES ?? "12", "cycles");
const dispatches = positiveInteger(process.env.FLUID_SVO_CONTINUATION_DISPATCHES ?? "1", "dispatches");
assert.ok(depth >= 3 && depth <= 6, "depth must be between 3 and 6");

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new RangeError(`${label} must be a positive integer`);
  return parsed;
}

function denseFixture(maximumDepth: number): {
  nodes: Uint32Array<ArrayBuffer>;
  leaves: Uint32Array<ArrayBuffer>;
} {
  const levelCounts = Array.from({ length: maximumDepth + 1 }, (_, level) => 8 ** level);
  const levelOffsets: number[] = [];
  let nodeCount = 0;
  for (const count of levelCounts) { levelOffsets.push(nodeCount); nodeCount += count; }
  const leafCount = levelCounts[maximumDepth];
  const nodes = new Uint32Array(nodeCount * 8), leaves = new Uint32Array(leafCount * 4);
  for (let level = 0; level <= maximumDepth; level += 1) {
    const count = levelCounts[level], offset = levelOffsets[level];
    for (let local = 0; local < count; local += 1) {
      const nodeIndex = offset + local, base = nodeIndex * 8;
      nodes[base] = local >>> 0;
      nodes[base + 1] = Math.floor(local / 0x1_0000_0000) >>> 0;
      nodes[base + 2] = level;
      if (level < maximumDepth) {
        nodes[base + 3] = 0xff;
        nodes[base + 4] = levelOffsets[level + 1] + local * 8;
        nodes[base + 5] = 8;
        nodes[base + 6] = INVALID;
      } else {
        nodes[base + 4] = INVALID;
        nodes[base + 6] = local;
        leaves.set([nodeIndex, local * 64, nodes[base], nodes[base + 1]], local * 4);
      }
    }
  }
  return { nodes, leaves };
}

function variantBody(variant: Variant): string {
  const begin = variant === "continuation"
    ? "var continuation: SvoTraversalContinuation; svoTraversalContinuationBegin(ray, mapping, &continuation);"
    : "";
  const next = variant === "continuation"
    ? "svoTraversalContinuationNext(narrowed, mapping, mapping.maximumDepth, &continuation)"
    : "svoTraverseWithDepthLimit(narrowed, mapping, mapping.maximumDepth)";
  return `${begin}
  var cursor = ray.tMin;
  var status = SVO_STATUS_MISS;
  var visitSum = 0u;
  var leafCount = 0u;
  var sequenceHash = 2166136261u;
  for (var attempt = 0u; attempt < 48u; attempt += 1u) {
    let narrowed = SvoRay(ray.origin, cursor, ray.direction, ray.tMax);
    let hit = ${next};
    status = hit.status;
    visitSum += hit.visits;
    if (hit.status != SVO_STATUS_HIT) { break; }
    leafCount += 1u;
    sequenceHash = (sequenceHash ^ hit.nodeIndex) * 16777619u;
    sequenceHash = (sequenceHash ^ hit.leafIndex) * 16777619u;
    sequenceHash = (sequenceHash ^ bitcast<u32>(hit.tEnter)) * 16777619u;
    sequenceHash = (sequenceHash ^ bitcast<u32>(hit.tExit)) * 16777619u;
    cursor = hit.tExit + 0.0017420508;
  }
  return BenchmarkResult(status, visitSum, leafCount, sequenceHash);`;
}

function shader(variant: Variant, nodeCount: number, leafCount: number): string {
  const extent = 2 ** depth * 4;
  return `${webgpuSvoTraversalWGSL}
struct BenchmarkResult { status: u32, visits: u32, leaves: u32, sequenceHash: u32 }
@group(0) @binding(3) var<storage, read_write> results: array<BenchmarkResult>;
fn enumerateLeaves(ray: SvoRay, mapping: SvoMapping) -> BenchmarkResult {
  ${variantBody(variant)}
}
@compute @workgroup_size(128)
fn benchmarkMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${width * height}u) { return; }
  let pixel = vec2u(gid.x % ${width}u, gid.x / ${width}u);
  let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(${width}.0, ${height}.0);
  let extent = ${extent}.0;
  let ray = SvoRay(vec3f(-extent * 0.5, (0.25 + uv.x * 0.5) * extent, (0.25 + uv.y * 0.5) * extent), 0.0,
    normalize(vec3f(1.0, (uv.x - 0.5) * 0.002 + 0.0003, (uv.y - 0.5) * 0.0015 + 0.0002)), extent * 3.0);
  let mapping = SvoMapping(vec3f(0.0), 4u, vec3f(1.0), ${depth}u, ${nodeCount}u, ${leafCount}u, 256u, 0u);
  results[gid.x] = enumerateLeaves(ray, mapping);
}`;
}

function dataBuffer(device: GPUDevice, label: string, data: Uint32Array<ArrayBuffer>): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data);
  return buffer;
}

function median(values: readonly number[]): number {
  const ordered = [...values].sort((a, b) => a - b), middle = Math.floor(ordered.length / 2);
  return ordered.length % 2 === 0 ? (ordered[middle - 1] + ordered[middle]) * 0.5 : ordered[middle];
}

const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU adapter unavailable");
assert.ok(adapter.features.has("timestamp-query"), "timestamp-query is required");
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const fixture = denseFixture(depth);
const control = dataBuffer(device, "continuation benchmark control", new Uint32Array(32));
const nodes = dataBuffer(device, "continuation benchmark nodes", fixture.nodes);
const leaves = dataBuffer(device, "continuation benchmark leaves", fixture.leaves);
const resultBytes = width * height * 16;
const output = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const readback = device.createBuffer({ size: resultBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const targets = new Map<Variant, { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup }>();

for (const variant of ["restart", "continuation"] as const) {
  const shaderModule = device.createShaderModule({ label: `${variant} leaf enumeration`, code: shader(variant, fixture.nodes.length / 8, fixture.leaves.length / 4) });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error").map(({ lineNum, linePos, message }) => ({ lineNum, linePos, message })), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "benchmarkMain" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: control } }, { binding: 1, resource: { buffer: nodes } },
    { binding: 2, resource: { buffer: leaves } }, { binding: 3, resource: { buffer: output } },
  ] });
  targets.set(variant, { pipeline, bindGroup });
}

async function outputFor(variant: Variant): Promise<Uint32Array> {
  const target = targets.get(variant)!;
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width * height / 128)); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, resultBytes); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copied = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  return copied;
}

const restartOutput = await outputFor("restart"), continuationOutput = await outputFor("continuation");
for (let index = 0; index < restartOutput.length; index += 4) {
  assert.equal(restartOutput[index], SVO_STATUS_MISS_FOR_HOST, `restart invocation ${index / 4} did not enumerate to miss`);
  assert.ok(restartOutput[index + 2] >= 8, `invocation ${index / 4} crossed only ${restartOutput[index + 2]} leaves`);
}
assert.deepEqual(
  Array.from(continuationOutput, (_, index) => index % 4 === 1 ? 0 : continuationOutput[index]),
  Array.from(restartOutput, (_, index) => index % 4 === 1 ? 0 : restartOutput[index]),
  "continuation changed status, leaf count, nearest-hit ordering, or intervals",
);
let restartVisits = 0, continuationVisits = 0, leafTotal = 0;
for (let index = 0; index < restartOutput.length; index += 4) {
  restartVisits += restartOutput[index + 1]; continuationVisits += continuationOutput[index + 1]; leafTotal += restartOutput[index + 2];
}
assert.ok(continuationVisits < restartVisits, "continuation did not reduce topology visits");

const variants = ["restart", "continuation"] as const;
for (let warmup = 0; warmup < warmups; warmup += 1) {
  for (const variant of variants) await outputFor(variant);
}
const timestampCount = cycles * variants.length * 2;
const querySet = device.createQuerySet({ type: "timestamp", count: timestampCount });
const resolve = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const queryReadback = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
const labels: Variant[] = [];
let query = 0;
const encoder = device.createCommandEncoder();
for (let cycle = 0; cycle < cycles; cycle += 1) {
  const order = cycle % 2 === 0 ? variants : (["continuation", "restart"] as const);
  for (const variant of order) {
    const target = targets.get(variant)!;
    const pass = encoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: query, endOfPassWriteIndex: query + 1 } });
    pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
    for (let dispatch = 0; dispatch < dispatches; dispatch += 1) pass.dispatchWorkgroups(Math.ceil(width * height / 128));
    pass.end(); labels.push(variant); query += 2;
  }
}
encoder.resolveQuerySet(querySet, 0, timestampCount, resolve, 0);
encoder.copyBufferToBuffer(resolve, 0, queryReadback, 0, timestampCount * 8);
device.queue.submit([encoder.finish()]); await queryReadback.mapAsync(GPUMapMode.READ);
const timestamps = new BigUint64Array(queryReadback.getMappedRange().slice(0)); queryReadback.unmap();
const samples: Record<Variant, number[]> = { restart: [], continuation: [] };
for (let index = 0; index < labels.length; index += 1) {
  samples[labels[index]].push(Number(timestamps[index * 2 + 1] - timestamps[index * 2]) / 1e6 / dispatches);
}
const restartMedian = median(samples.restart), continuationMedian = median(samples.continuation);
await device.queue.onSubmittedWorkDone();
assert.deepEqual(validationErrors, []);

const invocationCount = width * height;
console.log(JSON.stringify({
  backend: process.env.FLUID_WEBGPU_BACKEND ?? "metal", dimensions: [width, height], depth, cycles,
  parity: { exactSequenceOutputs: true, invocations: invocationCount, averageLeaves: leafTotal / invocationCount },
  nodeVisits: {
    restartPerRay: restartVisits / invocationCount,
    continuationPerRay: continuationVisits / invocationCount,
    reductionPercent: (1 - continuationVisits / restartVisits) * 100,
  },
  gpuMilliseconds: {
    restartMedian, continuationMedian,
    improvementPercent: (1 - continuationMedian / restartMedian) * 100,
    restartSamples: samples.restart, continuationSamples: samples.continuation,
  },
}, null, 2));

control.destroy(); nodes.destroy(); leaves.destroy(); output.destroy(); readback.destroy();
resolve.destroy(); queryReadback.destroy(); querySet.destroy(); device.destroy();
