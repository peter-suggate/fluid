#!/usr/bin/env node
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { webgpuSvoTraversalWGSL } from "../lib/webgpu-svo-traversal";

type Variant = "baseline" | "optimized";
type Comparison = "expansion" | "morton-decode";
type FixtureKind = "dense" | "deep-sparse";

interface PackedFixture {
  nodes: Uint32Array;
  leaves: Uint32Array;
  depth: number;
  brickSize: 4;
}

const INVALID = 0xffff_ffff;
const modulePath = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
const width = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_WIDTH ?? "512", "width");
const height = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_HEIGHT ?? "512", "height");
const depth = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_DEPTH ?? "5", "depth");
const warmups = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_WARMUPS ?? "4", "warmups");
const cycles = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_CYCLES ?? "12", "cycles");
const dispatchesPerSample = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_DISPATCHES ?? "1", "dispatches per sample");
const traversalsPerInvocation = positiveInteger(process.env.FLUID_SVO_TRAVERSAL_RAYS_PER_INVOCATION ?? "4", "rays per invocation");
const comparison = benchmarkComparison(process.env.FLUID_SVO_TRAVERSAL_COMPARISON ?? "expansion");
const fixtureKind = benchmarkFixture(process.env.FLUID_SVO_TRAVERSAL_FIXTURE ?? "dense");
const amplifiedMortonDecodesPerTraversal = comparison === "morton-decode" ? 64 : 0;
const workgroupSizes = (process.env.FLUID_SVO_TRAVERSAL_WORKGROUPS ?? "32,64,128,256")
  .split(",").map(Number);
assert.ok(fixtureKind === "deep-sparse" || (depth >= 3 && depth <= 6), "dense depth must be between 3 and 6");
assert.ok(workgroupSizes.length > 0 && workgroupSizes.every((value) => [32, 64, 128, 256].includes(value)),
  "workgroup sizes must be selected from 32,64,128,256");

function positiveInteger(value: string, label: string): number {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new RangeError(`${label} must be a positive integer`);
  return parsed;
}

function benchmarkComparison(value: string): Comparison {
  if (value === "expansion" || value === "morton-decode") return value;
  if (value === "parent-bounds" || value === "carried-bounds") {
    throw new RangeError(`comparison "${value}" was retired: its baseline reconstructed pre-carried-bounds traversal `
      + "text that no longer exists in the library; use \"expansion\" (candidate-array vs in-stack insertion) instead");
  }
  throw new RangeError("comparison must be expansion or morton-decode");
}

function benchmarkFixture(value: string): FixtureKind {
  if (value === "dense" || value === "deep-sparse") return value;
  throw new RangeError("fixture must be dense or deep-sparse");
}

function fullDenseFixture(maximumDepth: number): PackedFixture {
  const levelCounts = Array.from({ length: maximumDepth + 1 }, (_, level) => 8 ** level);
  const levelOffsets: number[] = [];
  let nodeCount = 0;
  for (const count of levelCounts) { levelOffsets.push(nodeCount); nodeCount += count; }
  const leafCount = levelCounts[maximumDepth];
  const nodes = new Uint32Array(nodeCount * 8);
  const leaves = new Uint32Array(leafCount * 4);
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
        const leafBase = local * 4;
        leaves[leafBase] = nodeIndex;
        leaves[leafBase + 1] = local * 64;
        leaves[leafBase + 2] = nodes[base];
        leaves[leafBase + 3] = nodes[base + 1];
      }
    }
  }
  return { nodes, leaves, depth: maximumDepth, brickSize: 4 };
}

function deepSparseFixture(maximumDepth = 21): PackedFixture {
  const nodeCount = maximumDepth + 1;
  const nodes = new Uint32Array(nodeCount * 8);
  const leaves = new Uint32Array(4);
  for (let level = 0; level <= maximumDepth; level += 1) {
    const base = level * 8;
    nodes[base + 2] = level;
    if (level < maximumDepth) {
      nodes[base + 3] = 1;
      nodes[base + 4] = level + 1;
      nodes[base + 5] = 1;
      nodes[base + 6] = INVALID;
    } else {
      nodes[base + 4] = INVALID;
      nodes[base + 6] = 0;
      leaves[0] = level;
      leaves[1] = 0;
    }
  }
  return { nodes, leaves, depth: maximumDepth, brickSize: 4 };
}

/**
 * Replace one top-level WGSL function (located by its `fn name(` header and
 * brace matching) so baseline construction never touches the rest of the
 * library source — in particular svoTraversalContinuationNext, which shares
 * textual shapes with svoTraverseWithDepthLimit.
 */
function replaceWgslFunction(source: string, name: string, replacement: string): string {
  const header = `fn ${name}(`;
  const start = source.indexOf(header);
  assert.ok(start >= 0, `WGSL function not found: ${name}`);
  assert.equal(source.indexOf(header, start + 1), -1, `WGSL function ${name} is not unique`);
  const braceStart = source.indexOf("{", start);
  assert.ok(braceStart >= 0, `WGSL function ${name} has no body`);
  let depth = 0;
  let end = -1;
  for (let index = braceStart; index < source.length; index += 1) {
    const character = source[index];
    if (character === "{") depth += 1;
    else if (character === "}") {
      depth -= 1;
      if (depth === 0) { end = index + 1; break; }
    }
  }
  assert.ok(end > 0, `WGSL function ${name} has unbalanced braces`);
  return source.slice(0, start) + replacement + source.slice(end);
}

/**
 * Verbatim snapshot of svoTraverseWithDepthLimit as it stood on main (commit
 * 2162041) before the in-stack streaming expansion landed: each internal node
 * insertion-sorts up to eight SvoCandidate entries into a function-local
 * scratch array and copies the deferred ones onto the stack afterwards. The
 * "expansion" comparison splices this over the current function so baseline
 * and optimized variants compile against the same library helpers.
 */
const CANDIDATE_ARRAY_EXPANSION_BASELINE = `fn svoTraverseWithDepthLimit(ray: SvoRay, mapping: SvoMapping, maximumTraversalDepth: u32) -> SvoTraversalHit {
  if (svoControl[12] != 0u) { return svoMiss(SVO_STATUS_SOURCE_OVERFLOW, 0u); }
  if (mapping.nodeCount == 0u) { return svoMiss(SVO_STATUS_MISS, 0u); }
  let root = svoNodes[0];
  if (root.address.x != 0u || root.address.y != 0u || root.address.z != 0u) {
    return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, 0u);
  }
  let inverseDirection = 1.0 / ray.direction;
  let rootBounds = svoRootBounds(mapping);
  let rootInterval = svoRayAabbWithInverse(ray, inverseDirection, rootBounds);
  if (rootInterval.x == 0.0) { return svoMiss(SVO_STATUS_MISS, 0u); }
  var stack: array<SvoStackEntry, 32>;
  var stackSize = 0u;
  var current = SvoStackEntry(0u, rootInterval.y, rootInterval.z);
  var currentBounds = rootBounds;
  var currentBoundsValid = true;
  var visits = 0u;
  let visitLimit = min(max(mapping.maxVisits, 1u), SVO_MAX_VISITS);
  var traversalGuard = 0u;
  while (traversalGuard <= SVO_MAX_VISITS) {
    traversalGuard += 1u;
    if (visits >= visitLimit) { return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits); }
    if (current.nodeIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    let node = svoNodes[current.nodeIndex];
    visits += 1u;
    if (node.address.z > mapping.maximumDepth) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
    if (node.address.z > maximumTraversalDepth) { return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits); }
    if (node.links.z != SVO_INVALID) {
      if (node.links.z >= mapping.leafCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let leaf = svoLeaves[node.links.z];
      if (leaf.topology.x != current.nodeIndex) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      return SvoTraversalHit(SVO_STATUS_HIT, visits, current.nodeIndex, node.links.z,
        leaf.topology.y, node.address.z, current.tEnter, current.tExit);
    }
    let mask = node.address.w & 0xffu;
    if (mask == 0u) {
      if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
      stackSize -= 1u;
      current = stack[stackSize];
      currentBoundsValid = false;
      continue;
    }
    if (node.links.x == SVO_INVALID || countOneBits(mask) != node.links.y) {
      return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits);
    }
    var parentBounds = currentBounds;
    if (!currentBoundsValid) { parentBounds = svoNodeBounds(node, mapping); }
    var candidates: array<SvoCandidate, 8>;
    var candidateCount = 0u;
    for (var octant = 0u; octant < 8u; octant += 1u) {
      if ((mask & (1u << octant)) == 0u) { continue; }
      let childIndex = node.links.x + svoPopcountBefore(mask, octant);
      if (childIndex >= mapping.nodeCount) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let child = svoNodes[childIndex];
      if (child.address.z != node.address.z + 1u) { return svoMiss(SVO_STATUS_INVALID_TOPOLOGY, visits); }
      let childBounds = svoChildBounds(parentBounds, octant);
      let interval = svoRayAabbWithInverse(ray, inverseDirection, childBounds);
      if (interval.x == 0.0) { continue; }
      var insertion = candidateCount;
      loop {
        if (insertion == 0u) { break; }
        let previous = candidates[insertion - 1u];
        let ordered = previous.tEnter < interval.y
          || (previous.tEnter == interval.y && (previous.tExit < interval.z
          || (previous.tExit == interval.z && previous.octant <= octant)));
        if (ordered) { break; }
        candidates[insertion] = previous;
        insertion -= 1u;
      }
      candidates[insertion] = SvoCandidate(childIndex, octant, interval.y, interval.z);
      candidateCount += 1u;
    }
    if (candidateCount == 0u) {
      if (stackSize == 0u) { return svoMiss(SVO_STATUS_MISS, visits); }
      stackSize -= 1u;
      current = stack[stackSize];
      currentBoundsValid = false;
      continue;
    }
    if (stackSize + candidateCount - 1u > SVO_STACK_CAPACITY) {
      return svoMiss(SVO_STATUS_STACK_OVERFLOW, visits);
    }
    var remaining = candidateCount;
    loop {
      if (remaining <= 1u) { break; }
      remaining -= 1u;
      let deferred = candidates[remaining];
      stack[stackSize] = SvoStackEntry(deferred.nodeIndex, deferred.tEnter, deferred.tExit);
      stackSize += 1u;
    }
    let nearest = candidates[0];
    current = SvoStackEntry(nearest.nodeIndex, nearest.tEnter, nearest.tExit);
    currentBounds = svoChildBounds(parentBounds, nearest.octant);
    currentBoundsValid = true;
  }
  return svoMiss(SVO_STATUS_WORK_EXHAUSTED, visits);
}`;

function expansionBaselineTraversalWGSL(): string {
  assert.ok(!webgpuSvoTraversalWGSL.includes("var candidates: array<SvoCandidate, 8>;"),
    "library still uses the candidate-array expansion; the expansion baseline would measure nothing");
  assert.ok(webgpuSvoTraversalWGSL.includes("struct SvoCandidate"),
    "baseline snapshot needs the SvoCandidate struct from the library");
  return replaceWgslFunction(webgpuSvoTraversalWGSL, "svoTraverseWithDepthLimit", CANDIDATE_ARRAY_EXPANSION_BASELINE);
}

function mortonDecodeBaselineTraversalWGSL(): string {
  const optimized = `fn svoCompactMortonBits(value: vec3u) -> vec3u {
  var compact = value & vec3u(0x49249249u);
  compact = (compact ^ (compact >> vec3u(2u))) & vec3u(0xc30c30c3u);
  compact = (compact ^ (compact >> vec3u(4u))) & vec3u(0x0f00f00fu);
  compact = (compact ^ (compact >> vec3u(8u))) & vec3u(0xff0000ffu);
  return (compact ^ (compact >> vec3u(16u))) & vec3u(0x0000ffffu);
}

fn svoDecodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  let levelMask = (1u << level) - 1u;
  let lowBits = svoCompactMortonBits(vec3u(low, low >> 1u, low >> 2u));
  let highBits = svoCompactMortonBits(vec3u(high >> 1u, high >> 2u, high));
  return (lowBits | (highBits << vec3u(11u, 11u, 10u))) & vec3u(levelMask);
}`;
  const baseline = `fn svoKeyBit(low: u32, high: u32, bit: u32) -> u32 {
  if (bit < 32u) { return (low >> bit) & 1u; }
  return (high >> (bit - 32u)) & 1u;
}

fn svoDecodeMorton(low: u32, high: u32, level: u32) -> vec3u {
  var result = vec3u(0u);
  for (var bit = 0u; bit < level; bit += 1u) {
    let scale = 1u << bit;
    result.x += svoKeyBit(low, high, 3u * bit) * scale;
    result.y += svoKeyBit(low, high, 3u * bit + 1u) * scale;
    result.z += svoKeyBit(low, high, 3u * bit + 2u) * scale;
  }
  return result;
}`;
  assert.ok(webgpuSvoTraversalWGSL.includes(optimized), "optimized Morton decode signature changed");
  return webgpuSvoTraversalWGSL.replace(optimized, baseline);
}

function baselineTraversalWGSL(): string {
  if (comparison === "morton-decode") return mortonDecodeBaselineTraversalWGSL();
  return expansionBaselineTraversalWGSL();
}

function shader(variant: Variant, workgroupSize: number, fixture: PackedFixture): string {
  const traversal = variant === "optimized" ? webgpuSvoTraversalWGSL : baselineTraversalWGSL();
  const extent = 2 ** fixture.depth * fixture.brickSize;
  return `${traversal}
struct BenchmarkResult { status: u32, visits: u32, nodeIndex: u32, leafIndex: u32 }
@group(0) @binding(3) var<storage, read_write> results: array<BenchmarkResult>;
@compute @workgroup_size(${workgroupSize})
fn benchmarkMain(@builtin(global_invocation_id) invocation: vec3u) {
  let index = invocation.x;
  if (index >= ${width * height}u) { return; }
  let pixel = vec2u(index % ${width}u, index / ${width}u);
  let uv = (vec2f(pixel) + vec2f(0.5)) / vec2f(${width}.0, ${height}.0);
  let mapping = SvoMapping(vec3f(0.0), ${fixture.brickSize}u, vec3f(1.0), ${fixture.depth}u,
    ${fixture.nodes.length / 8}u, ${fixture.leaves.length / 4}u, 256u, 0u);
  var statusMask = 0u;
  var visitSum = 0u;
  var nodeHash = 0u;
  var leafHash = 0u;
  for (var sample = 0u; sample < ${traversalsPerInvocation}u; sample += 1u) {
    ${comparison === "morton-decode" ? `for (var decodeSample = 0u; decodeSample < ${amplifiedMortonDecodesPerTraversal}u; decodeSample += 1u) {
      let keyLow = index * 747796405u + decodeSample * 2891336453u + sample * 277803737u;
      let keyHigh = (index ^ (decodeSample * 1597334677u)) & 0x7fffffffu;
      let decoded = svoDecodeMorton(keyLow, keyHigh, 21u);
      nodeHash = ((nodeHash * 16777619u) ^ decoded.x) + decoded.y * 31u + decoded.z * 131u;
    }` : ""}
    let jitter = vec2f(f32(sample) * 0.000013, f32(sample) * -0.000017);
    ${fixtureKind === "deep-sparse" ? `let ray = SvoRay(
      vec3f(uv * ${fixture.brickSize}.0, -${extent * 0.5}),
      0.0,
      vec3f(0.0, 0.0, 1.0),
      ${extent * 2}.0
    );` : `
    let ray = SvoRay(
      vec3f((uv.x + jitter.x) * ${extent}.0, (uv.y + jitter.y) * ${extent}.0, -${extent * 0.5}),
      0.0,
      normalize(vec3f((uv.x - 0.5) * 0.12 + 0.013, (uv.y - 0.5) * 0.10 + 0.017, 1.0)),
      ${extent * 3}.0
    );`}
    let hit = svoTraverse(ray, mapping);
    statusMask |= 1u << hit.status;
    visitSum += hit.visits;
    nodeHash = (nodeHash * 16777619u) ^ hit.nodeIndex;
    leafHash = (leafHash * 16777619u) ^ hit.leafIndex;
  }
  results[index] = BenchmarkResult(statusMask, visitSum, nodeHash, leafHash);
}`;
}

function bufferWithData(device: GPUDevice, label: string, data: Uint32Array): GPUBuffer {
  const buffer = device.createBuffer({ label, size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(buffer, 0, data.slice().buffer as ArrayBuffer);
  return buffer;
}

function median(values: readonly number[]): number {
  const sorted = [...values].sort((left, right) => left - right), middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) * 0.5 : sorted[middle];
}

function percentile(values: readonly number[], fraction: number): number {
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * fraction))];
}

const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
assert.ok(adapter, "WebGPU did not expose an adapter");
assert.ok(adapter.features.has("timestamp-query"), "SVO traversal benchmark requires timestamp-query support");
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

const fixture = fixtureKind === "deep-sparse" ? deepSparseFixture() : fullDenseFixture(depth);
if (fixtureKind === "dense") assert.ok(fixture.leaves.length / 4 > 64, "dense fixture must contain more than 64 leaves");
const control = bufferWithData(device, "SVO traversal benchmark control", new Uint32Array(32));
const nodes = bufferWithData(device, "SVO traversal benchmark nodes", fixture.nodes);
const leaves = bufferWithData(device, "SVO traversal benchmark leaves", fixture.leaves);
const resultBytes = width * height * 16;
const output = device.createBuffer({ label: "SVO traversal benchmark output", size: resultBytes,
  usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
const readback = device.createBuffer({ label: "SVO traversal benchmark readback", size: resultBytes,
  usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

const pipelines = new Map<string, { pipeline: GPUComputePipeline; bindGroup: GPUBindGroup; workgroupSize: number }>();
for (const workgroupSize of workgroupSizes) {
  for (const variant of ["baseline", "optimized"] as const) {
    const shaderModule = device.createShaderModule({ label: `${variant} SVO traversal ${workgroupSize}`, code: shader(variant, workgroupSize, fixture) });
    const compilation = await shaderModule.getCompilationInfo();
    const compilationErrors = compilation.messages.filter(({ type }) => type === "error");
    assert.deepEqual(compilationErrors.map(({ lineNum, linePos, message }) => ({ lineNum, linePos, message })), [],
      `${variant}/${workgroupSize} shader did not compile`);
    const pipeline = device.createComputePipeline({ label: `${variant} SVO traversal ${workgroupSize}`,
      layout: "auto", compute: { module: shaderModule, entryPoint: "benchmarkMain" } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: control } }, { binding: 1, resource: { buffer: nodes } },
      { binding: 2, resource: { buffer: leaves } }, { binding: 3, resource: { buffer: output } },
    ] });
    pipelines.set(`${variant}/${workgroupSize}`, { pipeline, bindGroup, workgroupSize });
  }
}

async function outputFor(variant: Variant, workgroupSize: number): Promise<Uint32Array> {
  const target = pipelines.get(`${variant}/${workgroupSize}`)!;
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width * height / workgroupSize)); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, resultBytes);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const copied = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  return copied;
}

const reference = await outputFor("baseline", workgroupSizes[0]);
let hitInvocationCount = 0, visitSum = 0;
for (let index = 0; index < reference.length; index += 4) {
  if (reference[index] === (1 << 1)) hitInvocationCount += 1;
  visitSum += reference[index + 1];
}
for (const workgroupSize of workgroupSizes) {
  assert.deepEqual(await outputFor("optimized", workgroupSize), reference,
    `optimized workgroup ${workgroupSize} changed traversal output`);
}

const timestampCount = workgroupSizes.length * 2 * cycles * 2;
const querySet = device.createQuerySet({ type: "timestamp", count: timestampCount });
const queryResolve = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
const queryReadback = device.createBuffer({ size: timestampCount * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

for (const target of pipelines.values()) {
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(); pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
  pass.dispatchWorkgroups(Math.ceil(width * height / target.workgroupSize)); pass.end();
  device.queue.submit([encoder.finish()]);
}
for (let index = 1; index < warmups; index += 1) {
  for (const workgroupSize of workgroupSizes) for (const variant of ["baseline", "optimized"] as const) {
    const target = pipelines.get(`${variant}/${workgroupSize}`)!;
    const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
    pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
    for (let dispatch = 0; dispatch < dispatchesPerSample; dispatch += 1) {
      pass.dispatchWorkgroups(Math.ceil(width * height / workgroupSize));
    }
    pass.end(); device.queue.submit([encoder.finish()]);
  }
}
await device.queue.onSubmittedWorkDone();

const samples = new Map<string, number[]>();
let queryIndex = 0;
const encoder = device.createCommandEncoder();
for (let cycle = 0; cycle < cycles; cycle += 1) {
  const sizes = cycle % 2 === 0 ? workgroupSizes : [...workgroupSizes].reverse();
  for (const workgroupSize of sizes) {
    const variants: readonly Variant[] = cycle % 2 === 0 ? ["baseline", "optimized"] : ["optimized", "baseline"];
    for (const variant of variants) {
      const target = pipelines.get(`${variant}/${workgroupSize}`)!;
      const pass = encoder.beginComputePass({ timestampWrites: { querySet,
        beginningOfPassWriteIndex: queryIndex, endOfPassWriteIndex: queryIndex + 1 } });
      pass.setPipeline(target.pipeline); pass.setBindGroup(0, target.bindGroup);
      for (let dispatch = 0; dispatch < dispatchesPerSample; dispatch += 1) {
        pass.dispatchWorkgroups(Math.ceil(width * height / workgroupSize));
      }
      pass.end(); queryIndex += 2;
    }
  }
}
encoder.resolveQuerySet(querySet, 0, timestampCount, queryResolve, 0);
encoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, timestampCount * 8);
device.queue.submit([encoder.finish()]);
await queryReadback.mapAsync(GPUMapMode.READ);
const timestamps = new BigUint64Array(queryReadback.getMappedRange());
queryIndex = 0;
for (let cycle = 0; cycle < cycles; cycle += 1) {
  const sizes = cycle % 2 === 0 ? workgroupSizes : [...workgroupSizes].reverse();
  for (const workgroupSize of sizes) {
    const variants: readonly Variant[] = cycle % 2 === 0 ? ["baseline", "optimized"] : ["optimized", "baseline"];
    for (const variant of variants) {
      const key = `${variant}/${workgroupSize}`;
      const values = samples.get(key) ?? [];
      values.push(Number(timestamps[queryIndex + 1] - timestamps[queryIndex]) / 1e6);
      samples.set(key, values); queryIndex += 2;
    }
  }
}
queryReadback.unmap();

const rows = workgroupSizes.map((workgroupSize) => {
  const baseline = samples.get(`baseline/${workgroupSize}`)!, optimized = samples.get(`optimized/${workgroupSize}`)!;
  const baselineMedian = median(baseline), optimizedMedian = median(optimized);
  const pairedSpeedups = baseline.map((value, index) => value / optimized[index]);
  return {
    workgroupSize,
    baselineMedian_ms: Number((baselineMedian / dispatchesPerSample).toFixed(3)),
    optimizedMedian_ms: Number((optimizedMedian / dispatchesPerSample).toFixed(3)),
    optimizedP95_ms: Number((percentile(optimized, 0.95) / dispatchesPerSample).toFixed(3)),
    speedup: Number((baselineMedian / optimizedMedian).toFixed(3)),
    pairedMedianSpeedup: Number(median(pairedSpeedups).toFixed(3)),
    improvementPercent: Number(((1 - optimizedMedian / baselineMedian) * 100).toFixed(1)),
    millionRaysPerSecond: Number((width * height * traversalsPerInvocation * dispatchesPerSample / optimizedMedian / 1000).toFixed(2)),
  };
});
console.table(rows);
console.log(JSON.stringify({
  phase: "svo-traversal-gpu-benchmark",
  comparison,
  fixtureKind,
  adapter: adapter.info,
  fixture: { depth: fixture.depth, nodes: fixture.nodes.length / 8, leaves: fixture.leaves.length / 4,
    baselineStackEntryBytes: 12,
    optimizedStackEntryBytes: 12,
    baselineExpansionScratchWords: comparison === "expansion" ? 32 : 0,
    optimizedExpansionScratchWords: comparison === "expansion" ? 7 : 0,
    expansionScratchNote: comparison === "expansion"
      ? "baseline: array<SvoCandidate, 8> = 32 words; optimized: nearest SvoCandidate (4) + deferred SvoStackEntry (3) in registers"
      : "expansion scratch identical in both variants for this comparison",
    stackCapacity: 32,
    invocationCount: width * height, rayCount: width * height * traversalsPerInvocation,
    allHitInvocationCount: hitInvocationCount, averageNodeVisits: visitSum / (width * height * traversalsPerInvocation) },
  measurement: { warmups, cycles, traversalsPerInvocation, dispatchesPerSample,
    amplifiedMortonDecodesPerTraversal,
    timestampUnit: "GPU timestamp-query nanoseconds converted to milliseconds",
    equivalence: "status mask, visit sum, and nearest node/leaf hashes are bit-identical for every invocation" },
  results: rows,
  validationErrors,
}));
assert.deepEqual(validationErrors, [], `WebGPU validation errors: ${validationErrors.join("; ")}`);

for (const resource of [control, nodes, leaves, output, readback, queryResolve, queryReadback]) resource.destroy();
querySet.destroy(); device.destroy();
