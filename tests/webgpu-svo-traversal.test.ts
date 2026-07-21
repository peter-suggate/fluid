import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { planAdaptiveSparseBrickOctree } from "../lib/adaptive-sparse-brick-plan";
import { mortonEncode3D, packSparseBrickPlan, planSparseBrickOctree } from "../lib/sparse-brick-octree";
import {
  compareSvoIntersectionArithmetic,
  createWebgpuSvoTraversalWGSL,
  intersectSvoRayAabb,
  SVO_WGSL_STATUS,
  svoBrickVoxelIndex,
  traversePackedSvo,
  webgpuSvoTraversalWGSL,
  type SvoPackedTopologyView,
  type SvoWorldMapping,
  type SvoTraversalWorkDiagnostics,
} from "../lib/webgpu-svo-traversal";

const mapping: SvoWorldMapping = {
  origin: [10, 20, 30],
  cellSize: [0.5, 1, 2],
  brickSize: 4,
  maximumDepth: 2,
};

function packedView(plan: ReturnType<typeof planSparseBrickOctree>): SvoPackedTopologyView {
  const packed = packSparseBrickPlan(plan);
  return {
    nodes: packed.nodes,
    leaves: packed.leaves,
    publishedNodeCount: plan.nodes.length,
    publishedLeafCount: plan.leaves.length,
  };
}

test("ray/AABB intersection handles hits, misses, inside starts, and parallel boundary rays", () => {
  const bounds = { minimum: [1, 2, 3] as const, maximum: [5, 6, 7] as const };
  assert.deepEqual(intersectSvoRayAabb({ origin: [0, 4, 5], direction: [1, 0, 0] }, bounds), { tEnter: 1, tExit: 5 });
  assert.deepEqual(intersectSvoRayAabb({ origin: [2, 4, 5], direction: [1, 0, 0] }, bounds), { tEnter: 0, tExit: 3 });
  assert.deepEqual(intersectSvoRayAabb({ origin: [0, 2, 5], direction: [1, 0, 0] }, bounds), { tEnter: 1, tExit: 5 });
  assert.equal(intersectSvoRayAabb({ origin: [0, 1.99, 5], direction: [1, 0, 0] }, bounds), null);
  assert.equal(intersectSvoRayAabb({ origin: [0, 4, 8], direction: [1, 0, 0] }, bounds), null);
});

test("direct traversal returns fine leaves and anisotropic world bounds", () => {
  const plan = planSparseBrickOctree([{ x: 1, y: 2, z: 3 }], { brickSize: 4, maximumDepth: 2 });
  const result = traversePackedSvo(
    { origin: [0, 30, 58], direction: [1, 0, 0] },
    packedView(plan),
    mapping,
  );
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  assert.equal(result.hit.level, 2);
  assert.deepEqual(result.hit.coordinate, [1, 2, 3]);
  assert.deepEqual(result.hit.bounds, { minimum: [12, 28, 54], maximum: [14, 32, 62] });
  assert.equal(result.hit.tEnter, 12);
  assert.equal(result.hit.tExit, 14);
});

test("direct traversal returns adaptively coarse environment leaves", () => {
  const plan = planAdaptiveSparseBrickOctree({
    brickSize: 4,
    solverBricks: [{ x: 0, y: 0, z: 0 }],
    proxyBricks: [{ x: 2, y: 0, z: 0 }],
    maximumDepth: 2,
    maximumEnvironmentCoarseningPower: 1,
  });
  const packed = packSparseBrickPlan(plan);
  const result = traversePackedSvo(
    { origin: [13, 22, 32], direction: [1, 0, 0] },
    { nodes: packed.nodes, leaves: packed.leaves },
    mapping,
  );
  assert.equal(result.status, "hit");
  if (result.status !== "hit") return;
  assert.equal(result.hit.level, 1);
  assert.deepEqual(result.hit.coordinate, [1, 0, 0]);
  assert.deepEqual(result.hit.bounds, { minimum: [14, 20, 30], maximum: [18, 28, 46] });
});

test("children are visited geometrically near-to-far from either ray direction", () => {
  const plan = planSparseBrickOctree([
    { x: 0, y: 0, z: 0 },
    { x: 1, y: 0, z: 0 },
  ], { brickSize: 4, maximumDepth: 1 });
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const topology = packedView(plan);
  const forward = traversePackedSvo({ origin: [-1, 2, 2], direction: [1, 0, 0] }, topology, localMapping);
  const backward = traversePackedSvo({ origin: [9, 2, 2], direction: [-1, 0, 0] }, topology, localMapping);
  assert.equal(forward.status, "hit");
  assert.equal(backward.status, "hit");
  if (forward.status === "hit" && backward.status === "hit") {
    assert.deepEqual(forward.hit.coordinate, [0, 0, 0]);
    assert.deepEqual(backward.hit.coordinate, [1, 0, 0]);
  }
});

test("shared-face boundary traversal has stable ascending-octant tie breaking", () => {
  const plan = planSparseBrickOctree([
    { x: 0, y: 0, z: 0 },
    { x: 0, y: 1, z: 0 },
  ], { brickSize: 4, maximumDepth: 1 });
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const result = traversePackedSvo({ origin: [-1, 4, 2], direction: [1, 0, 0] }, packedView(plan), localMapping);
  assert.equal(result.status, "hit");
  if (result.status === "hit") assert.deepEqual(result.hit.coordinate, [0, 0, 0]);
});

test("miss, visit exhaustion, stack overflow, and source overflow are distinct", () => {
  const coordinates = Array.from({ length: 8 }, (_, octant) => ({
    x: octant & 1,
    y: (octant >> 1) & 1,
    z: (octant >> 2) & 1,
  }));
  const plan = planSparseBrickOctree(coordinates, { brickSize: 4, maximumDepth: 1 });
  const topology = packedView(plan);
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  assert.equal(traversePackedSvo({ origin: [-1, 20, 2], direction: [1, 0, 0] }, topology, localMapping).status, "miss");
  assert.equal(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, topology, localMapping, { maxNodeVisits: 1 },
  ).status, "work-exhausted");
  assert.equal(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, topology, localMapping, { stackCapacity: 1 },
  ).status, "stack-overflow");
  assert.deepEqual(traversePackedSvo(
    { origin: [-1, 4, 4], direction: [1, 0, 0] }, { ...topology, overflowFlags: 5 }, localMapping,
  ), { status: "source-overflow", visits: 0, overflowFlags: 5 });
});

test("malformed packed topology fails closed", () => {
  const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }], { brickSize: 4, maximumDepth: 0 });
  const topology = packedView(plan);
  topology.leaves[0] = 99;
  const result = traversePackedSvo({ origin: [-1, 1, 1], direction: [1, 0, 0] }, topology, {
    origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 0,
  });
  assert.equal(result.status, "invalid-topology");
});

test("WGSL helper consumes packed topology directly with bounded traversal", () => {
  assert.match(webgpuSvoTraversalWGSL, /var<storage, read> svoNodes: array<SvoNode>/);
  assert.match(webgpuSvoTraversalWGSL, /var<storage, read> svoLeaves: array<SvoLeaf>/);
  assert.match(webgpuSvoTraversalWGSL, /SVO_STACK_CAPACITY: u32 = 32u/);
  assert.match(webgpuSvoTraversalWGSL, /SVO_MAX_VISITS: u32 = 256u/);
  assert.match(webgpuSvoTraversalWGSL, /svoNodeBounds/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoCompactMortonBits\(value: vec3u\) -> vec3u/);
  assert.match(webgpuSvoTraversalWGSL, /0x49249249u/);
  assert.match(webgpuSvoTraversalWGSL, /highBits << vec3u\(11u, 11u, 10u\)/);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /fn svoDecodeMorton[^}]*for\s*\(/);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /fn svoKeyBit/);
  assert.match(webgpuSvoTraversalWGSL, /let inverseDirection = 1\.0 \/ ray\.direction/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoChildBounds/);
  assert.match(webgpuSvoTraversalWGSL, /let childBounds = svoChildBounds\(parentBounds, octant\)/);
  assert.match(webgpuSvoTraversalWGSL, /struct SvoStackEntry \{ nodeIndex: u32, tEnter: f32, tExit: f32 \}/);
  assert.match(webgpuSvoTraversalWGSL, /struct SvoCandidate \{ nodeIndex: u32, octant: u32, tEnter: f32, tExit: f32 \}/);
  assert.match(webgpuSvoTraversalWGSL, /struct SvoTraversalContinuation \{/);
  assert.match(webgpuSvoTraversalWGSL, /stack: array<SvoStackEntry, 32>/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoTraversalContinuationBegin\(/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoTraversalContinuationNext\(/);
  assert.match(webgpuSvoTraversalWGSL, /max\(current\.tEnter, ray\.tMin\), min\(current\.tExit, ray\.tMax\)/);
  assert.match(webgpuSvoTraversalWGSL, /svoTraversalContinuationAdvance\(continuation\);\n      return hit/);
  assert.match(webgpuSvoTraversalWGSL, /currentBounds = svoChildBounds\(parentBounds, nearest\.octant\)/);
  assert.match(webgpuSvoTraversalWGSL, /currentBoundsValid = false/);
  assert.match(webgpuSvoTraversalWGSL, /stackSize \+ candidateCount - 1u/);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /svoRayAabb\(ray, svoNodeBounds\(child, mapping\)\)/);
  assert.match(webgpuSvoTraversalWGSL, /countOneBits\(mask/);
  assert.match(webgpuSvoTraversalWGSL, /fn svoBrickVoxelIndex/);
  assert.doesNotMatch(webgpuSvoTraversalWGSL, /DebugRecord|voxelRecords|brickRecords/);
  assert.match(createWebgpuSvoTraversalWGSL({ group: 2, control: 4, nodes: 5, leaves: 6 }), /@group\(2\) @binding\(5\) var<storage, read> svoNodes/);
  assert.equal(svoBrickVoxelIndex(512, [2, 3, 1], 8), 602);
});

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;
test("constant-time WGSL Morton decode preserves all 21 coordinate bits", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU Morton checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

  const cases: Array<readonly [number, number, number]> = [];
  for (let level = 0; level <= 21; level += 1) {
    const mask = 2 ** level - 1;
    cases.push([0xffff_ffff, 0x7fff_ffff, level]);
    for (let sample = 0; sample < 4; sample += 1) {
      const x = ((level * 0x45d9f3b) ^ (sample * 0x119de1f3)) & mask;
      const y = ((level * 0x27d4eb2d) ^ (sample * 0x3449f5)) & mask;
      const z = ((level * 0x165667b1) ^ (sample * 0x9e3779b)) & mask;
      const morton = mortonEncode3D(x, y, z);
      cases.push([Number(morton & 0xffff_ffffn), Number(morton >> 32n), level]);
    }
  }
  const sourceWords = new Uint32Array(cases.length * 4);
  cases.forEach(([low, high, level], index) => sourceWords.set([low, high, level, 0], index * 4));
  const byteLength = sourceWords.byteLength;
  const source = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const output = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: byteLength, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.queue.writeBuffer(source, 0, sourceWords);
  const shader = `${webgpuSvoTraversalWGSL}
@group(0) @binding(3) var<storage, read> mortonCases: array<vec4u>;
@group(0) @binding(4) var<storage, read_write> mortonResults: array<vec4u>;
@compute @workgroup_size(64)
fn decodeMortonCases(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= ${cases.length}u) { return; }
  let source = mortonCases[gid.x];
  mortonResults[gid.x] = vec4u(svoDecodeMorton(source.x, source.y, source.z), 0u);
}`;
  const shaderModule = device.createShaderModule({ code: shader });
  const compilation = await shaderModule.getCompilationInfo();
  assert.deepEqual(compilation.messages.filter(({ type }) => type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "decodeMortonCases" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 3, resource: { buffer: source } }, { binding: 4, resource: { buffer: output } },
  ] });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(Math.ceil(cases.length / 64)); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, byteLength); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const actual = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  cases.forEach(([low, high, level], index) => {
    const morton = BigInt(low) | (BigInt(high) << 32n), expected = [0, 0, 0];
    for (let bit = 0; bit < level; bit += 1) for (let axis = 0; axis < 3; axis += 1) {
      expected[axis] += Number((morton >> BigInt(3 * bit + axis)) & 1n) * 2 ** bit;
    }
    assert.deepEqual([...actual.slice(index * 4, index * 4 + 3)], expected, `Morton case ${index} at level ${level}`);
  });
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, []);
  source.destroy(); output.destroy(); readback.destroy(); device.destroy();
});

test("dense structural fixture quantifies eliminated child-bound decode and division work", () => {
  const coordinates = Array.from({ length: 8 }, (_, octant) => ({
    x: octant & 1,
    y: (octant >> 1) & 1,
    z: (octant >> 2) & 1,
  }));
  const topology = packedView(planSparseBrickOctree(coordinates, { brickSize: 4, maximumDepth: 1 }));
  const localMapping: SvoWorldMapping = { origin: [0, 0, 0], cellSize: [1, 1, 1], brickSize: 4, maximumDepth: 1 };
  const diagnostics: SvoTraversalWorkDiagnostics = { rootAabbTests: 0, internalNodesExpanded: 0, childAabbTests: 0 };
  const result = traversePackedSvo(
    { origin: [-1, -1, -1], direction: [1, 1.125, 1.25] },
    topology,
    localMapping,
    { diagnostics },
  );
  assert.equal(result.status, "hit");
  assert.deepEqual(diagnostics, { rootAabbTests: 1, internalNodesExpanded: 1, childAabbTests: 8 });
  assert.deepEqual(compareSvoIntersectionArithmetic(diagnostics), {
    previous: { mortonBoundsDecodes: 9, rayDirectionDivisions: 27 },
    optimized: { mortonBoundsDecodes: 2, rayDirectionDivisions: 3 },
  });
});

test("WGSL continuation yields the restart sequence with fewer visits and unchanged limits", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU continuation checks",
}, async () => {
  const plan = planSparseBrickOctree([{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }], {
    brickSize: 4, maximumDepth: 1,
  });
  const packed = packSparseBrickPlan(plan);
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));
  const makeBuffer = (data: Uint32Array<ArrayBuffer>, usage: GPUBufferUsageFlags) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage });
    device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  };
  const control = makeBuffer(new Uint32Array(32), GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const nodes = makeBuffer(packed.nodes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const leaves = makeBuffer(packed.leaves, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const output = device.createBuffer({ size: 64, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const shader = `${webgpuSvoTraversalWGSL}
@group(0) @binding(3) var<storage, read_write> continuationResults: array<vec4u>;
fn continuationHash(previous: u32, hit: SvoTraversalHit) -> u32 {
  var result = (previous ^ hit.nodeIndex) * 16777619u;
  result = (result ^ hit.leafIndex) * 16777619u;
  result = (result ^ bitcast<u32>(hit.tEnter)) * 16777619u;
  return (result ^ bitcast<u32>(hit.tExit)) * 16777619u;
}
@compute @workgroup_size(1)
fn continuationOracle() {
  let ray = SvoRay(vec3f(-1.0, 2.0, 2.0), 0.0, vec3f(1.0, 0.0, 0.0), 20.0);
  let mapping = SvoMapping(vec3f(0.0), 4u, vec3f(1.0), 1u, ${plan.nodes.length}u, ${plan.leaves.length}u, 256u, 0u);
  var restartCursor = 0.0; var restartVisits = 0u; var restartLeaves = 0u; var restartHash = 2166136261u; var restartStatus = 0u;
  for (var attempt = 0u; attempt < 8u; attempt += 1u) {
    let hit = svoTraverseWithDepthLimit(SvoRay(ray.origin, restartCursor, ray.direction, ray.tMax), mapping, 1u);
    restartStatus = hit.status; restartVisits += hit.visits;
    if (hit.status != SVO_STATUS_HIT) { break; }
    restartLeaves += 1u; restartHash = continuationHash(restartHash, hit); restartCursor = hit.tExit + 0.0017420508;
  }
  continuationResults[0] = vec4u(restartStatus, restartVisits, restartLeaves, restartHash);
  var continuation: SvoTraversalContinuation; svoTraversalContinuationBegin(ray, mapping, &continuation);
  var cursor = 0.0; var visits = 0u; var leafCount = 0u; var hash = 2166136261u; var status = 0u;
  for (var attempt = 0u; attempt < 8u; attempt += 1u) {
    let hit = svoTraversalContinuationNext(SvoRay(ray.origin, cursor, ray.direction, ray.tMax), mapping, 1u, &continuation);
    status = hit.status; visits += hit.visits;
    if (hit.status != SVO_STATUS_HIT) { break; }
    leafCount += 1u; hash = continuationHash(hash, hit); cursor = hit.tExit + 0.0017420508;
  }
  continuationResults[1] = vec4u(status, visits, leafCount, hash);
  var depthContinuation: SvoTraversalContinuation; svoTraversalContinuationBegin(ray, mapping, &depthContinuation);
  let depthHit = svoTraversalContinuationNext(ray, mapping, 0u, &depthContinuation);
  continuationResults[2] = vec4u(depthHit.status, depthHit.visits, depthHit.level, depthHit.nodeIndex);
  var budgetMapping = mapping; budgetMapping.maxVisits = 1u;
  var budgetContinuation: SvoTraversalContinuation; svoTraversalContinuationBegin(ray, budgetMapping, &budgetContinuation);
  let budgetHit = svoTraversalContinuationNext(ray, budgetMapping, 1u, &budgetContinuation);
  continuationResults[3] = vec4u(budgetHit.status, budgetHit.visits, budgetHit.level, budgetHit.nodeIndex);
}`;
  const shaderModule = device.createShaderModule({ code: shader }), info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "continuationOracle" } });
  const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
    { binding: 0, resource: { buffer: control } }, { binding: 1, resource: { buffer: nodes } },
    { binding: 2, resource: { buffer: leaves } }, { binding: 3, resource: { buffer: output } },
  ] });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
  pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, 64); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const actual = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap();
  assert.deepEqual([...actual.slice(0, 4)], [0, 4, 2, actual[3]]);
  assert.deepEqual([...actual.slice(4, 8)], [0, 3, 2, actual[3]], "continuation must preserve the exact leaf/interval hash");
  assert.deepEqual([...actual.slice(8, 10)], [SVO_WGSL_STATUS.workExhausted, 2]);
  assert.deepEqual([...actual.slice(12, 14)], [SVO_WGSL_STATUS.workExhausted, 1]);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(validationErrors, []);
  control.destroy(); nodes.destroy(); leaves.destroy(); output.destroy(); readback.destroy(); device.destroy();
});
