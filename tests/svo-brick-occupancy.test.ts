import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import type { SparseBrickOctreeGPU } from "../lib/sparse-brick-octree";
import {
  SVO_BRICK_OCCUPANCY,
  SVO_BRICK_LIFECYCLE,
  buildSvoBrickOccupancy,
  decodeSvoBrickOccupancy,
  decodeSvoBrickLifecycle,
  encodeSvoBrickOccupancy,
  encodeSvoBrickLifecycle,
  isSvoBrickLifecycleCurrent,
  replaceSvoBrickLifecycle,
  replaceSvoBrickOccupancy,
  svoBrickOccupancyRetainedCellCount,
  svoBrickOccupancyWGSL,
} from "../lib/svo-brick-occupancy";
import {
  WebGpuSvoBrickOccupancyBuilder,
  SVO_BRICK_OCCUPANCY_WORKLIST_LAYOUT,
  webgpuSvoBrickOccupancyBuildWGSL,
} from "../lib/webgpu-svo-brick-occupancy";
import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";

const cell = (x: number, y: number, z: number) => x + 8 * (y + 8 * z);

test("occupancy and lifecycle independently round-trip one terminal flags word", () => {
  const summary = {
    ready: true,
    occupied: true,
    macroMask: 0b1010_0101,
    minInclusive: [1, 2, 3] as const,
    maxInclusive: [7, 6, 5] as const,
  };
  const lifecycle = { active: true, dirty: true, queued: true, relocating: false };
  const packed = replaceSvoBrickLifecycle(encodeSvoBrickOccupancy(summary), lifecycle);
  assert.deepEqual(decodeSvoBrickOccupancy(packed), summary);
  assert.deepEqual(decodeSvoBrickLifecycle(packed), lifecycle);
  assert.equal(packed & SVO_BRICK_LIFECYCLE.mask, encodeSvoBrickLifecycle(lifecycle));
  assert.equal(isSvoBrickLifecycleCurrent(lifecycle), false);
  assert.equal(isSvoBrickLifecycleCurrent({ ...lifecycle, dirty: false, queued: false }), true);
  const replacedOccupancy = replaceSvoBrickOccupancy(packed, {
    ready: true, occupied: false, macroMask: 0, minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0],
  });
  assert.deepEqual(decodeSvoBrickLifecycle(replacedOccupancy), lifecycle,
    "occupancy rebuilds cannot implicitly finalize a dirty leaf");
  assert.equal(replaceSvoBrickLifecycle(replacedOccupancy, {
    active: false, dirty: false, queued: false, relocating: true,
  }) & SVO_BRICK_OCCUPANCY.metadataMask, replacedOccupancy & SVO_BRICK_OCCUPANCY.metadataMask);
  assert.deepEqual(decodeSvoBrickOccupancy(0), {
    ready: false, occupied: false, macroMask: 0, minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0],
  });
  assert.equal(SVO_BRICK_OCCUPANCY.incrementalStorageBytesPerBrick, 0);
  assert.equal(SVO_BRICK_OCCUPANCY.metadataBytesPerBrick, 4);
});

test("builder conservatively summarizes exact material occupancy and ignores owner bits", () => {
  const owners = new Uint32Array(2 * 512);
  owners[cell(1, 2, 3)] = 0x1234_0007;
  owners[cell(7, 6, 5)] = 0xffff_0009;
  owners[cell(0, 0, 0)] = 0x4321_0000;
  const built = buildSvoBrickOccupancy(owners);
  assert.equal(built.occupiedCellCount, 2);
  assert.deepEqual(built.minInclusive, [1, 2, 3]);
  assert.deepEqual(built.maxInclusive, [7, 6, 5]);
  assert.equal(built.macroMask, (1 << 0) | (1 << 7));
  assert.deepEqual(decodeSvoBrickOccupancy(built.packed), {
    ready: true, occupied: true, macroMask: 0b1000_0001,
    minInclusive: [1, 2, 3], maxInclusive: [7, 6, 5],
  });

  const second = buildSvoBrickOccupancy(owners, 512);
  assert.equal(second.packed, SVO_BRICK_OCCUPANCY.readyBit);
  assert.deepEqual(second, {
    packed: SVO_BRICK_OCCUPANCY.readyBit, ready: true, occupied: false, macroMask: 0,
    minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0], occupiedCellCount: 0,
  });
});

test("CPU occupancy rebuild preserves live lifecycle flags", () => {
  const lifecycle = encodeSvoBrickLifecycle({ active: true, dirty: true, queued: true, relocating: true });
  const built = buildSvoBrickOccupancy(new Uint32Array(512), 0, lifecycle | 0x05555555);
  assert.equal((built.packed & SVO_BRICK_LIFECYCLE.mask) >>> 0, lifecycle);
  assert.equal(built.packed & SVO_BRICK_OCCUPANCY.readyBit, SVO_BRICK_OCCUPANCY.readyBit);
});

test("macrocell retention quantifies conservative skipped payload work", () => {
  const oneCorner = new Uint32Array(512);
  oneCorner[cell(0, 0, 0)] = 1;
  const sparse = buildSvoBrickOccupancy(oneCorner);
  assert.equal(svoBrickOccupancyRetainedCellCount(sparse), 64);
  assert.equal(512 - svoBrickOccupancyRetainedCellCount(sparse), 448,
    "a one-macrocell brick can reject 87.5% of fine cells before DDA");
  assert.equal(svoBrickOccupancyRetainedCellCount({ ready: false, macroMask: 0 }), 512,
    "absent metadata must retain exact baseline behavior");
});

test("invalid summaries and incomplete payloads fail closed", () => {
  assert.throws(() => encodeSvoBrickOccupancy({
    ready: false, occupied: true, macroMask: 1, minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0],
  }), /Unavailable/);
  assert.throws(() => encodeSvoBrickOccupancy({
    ready: true, occupied: true, macroMask: 0, minInclusive: [0, 0, 0], maxInclusive: [0, 0, 0],
  }), /at least one/);
  assert.throws(() => encodeSvoBrickOccupancy({
    ready: true, occupied: true, macroMask: 1, minInclusive: [4, 0, 0], maxInclusive: [3, 7, 7],
  }), /ordered/);
  assert.throws(() => buildSvoBrickOccupancy(new Uint32Array(511)), /complete/);
});

test("WGSL build and traversal helpers preserve fallback and exact-hit semantics", () => {
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /let lifecycle=topologyRead\(flagsIndex\)&LIFECYCLE_MASK/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /topologyWrite\(flagsIndex,packed\|lifecycle\)/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /fn buildWorklistBrickOccupancy/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /rebuildLeaf\(leafWorklist\[entry\]\)/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /FLUID_RESIDENCY_ENTRIES\+workIndex\*FLUID_RESIDENCY_STRIDE\+1u/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /let fluidMaterial=payload\[payloadIndex\]&0xffffu/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /let sceneMaterial=sceneMaterialOwners\[voxelOffset\+localIndex\]&0xffffu/);
  assert.match(svoBrickOccupancyWGSL, /if\(summary\.ready==0u\)\{return true;\}/);
  assert.match(svoBrickOccupancyWGSL, /fn svoBrickLifecycleCurrent/);
  assert.match(svoBrickOccupancyWGSL, /summary\.maxInclusive\+vec3u\(1u\)/);
});

test("dry-scene brick modes are compile-time A/B variants with an exact off baseline", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1, "hybrid", "off"), svoDrySceneShader);
  const bounds = createSvoDrySceneFragmentWGSL(1, "hybrid", "bounds");
  const macro = createSvoDrySceneFragmentWGSL(1, "hybrid", "macro");
  const macroHdda = createSvoDrySceneFragmentWGSL(1, "hybrid", "macro-hdda");
  assert.match(bounds, /svoBrickOccupiedBounds/);
  assert.doesNotMatch(bounds, /let macroSkip=dryBrickMacroSkip/);
  assert.match(macro, /let macroSkip=dryBrickMacroSkip/);
  assert.match(macroHdda, /fn traceLeafPayloadMacroHdda/);
  assert.match(macroHdda, /fn traceLeafPayloadVisibilityMacroHdda/);
  assert.match(macroHdda, /for\(var macroIteration=0u;macroIteration<8u;macroIteration\+=1u\)/);
  assert.match(macroHdda, /traceLeafPayloadFineInterval/);
  assert.doesNotMatch(macroHdda, /let macroSkip=dryBrickMacroSkip/,
    "macro HDDA must not branch or reinitialize inside every fine-cell step");
  assert.throws(() => createSvoDrySceneFragmentWGSL(1, "hybrid", "bad" as never), /brick occupancy mode/);
});

test("GPU builder uses a compact indirect leaf worklist and reserves full scans for initialization", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1 } });
  const dispatches: Array<{ kind: "direct" | "indirect"; values: unknown[] }> = [];
  const resources: unknown[] = [];
  const pipeline = { getBindGroupLayout: () => ({}) };
  const device = {
    createShaderModule: () => ({}),
    createComputePipeline: () => pipeline,
    createBindGroup: (descriptor: { entries: Array<{ resource: unknown }> }) => {
      resources.push(...descriptor.entries.map((entry) => entry.resource));
      return descriptor;
    },
  } as unknown as GPUDevice;
  const pass = {
    setPipeline: () => undefined,
    setBindGroup: () => undefined,
    dispatchWorkgroups: (...dimensions: number[]) => dispatches.push({ kind: "direct", values: dimensions }),
    dispatchWorkgroupsIndirect: (...values: unknown[]) => dispatches.push({ kind: "indirect", values }),
    end: () => undefined,
  };
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const buffer = {} as GPUBuffer;
  try {
    const builder = new WebGpuSvoBrickOccupancyBuilder(device);
    assert.equal(builder.incrementalAllocatedBytes, 0);
    const tree = {
      brickSize: 8, leafCapacity: 129, structure: buffer, control: buffer, topology: buffer, payload: buffer,
      voxelCapacity: 129 * 512, sceneMaterialOwners: buffer, sceneMaterialOwnerOffsetBytes: 0,
    } as SparseBrickOctreeGPU;
    assert.equal(builder.encodeAllLeavesForInitialization(encoder, tree), "encoded");
    assert.equal(builder.encodeWorklist(encoder, tree, { buffer, kind: "dirty" }), "encoded");
    assert.equal(builder.encodeFluidResidency(encoder, tree, buffer), "encoded");
    assert.deepEqual(dispatches, [
      { kind: "direct", values: [3] },
      { kind: "indirect", values: [buffer, SVO_BRICK_OCCUPANCY_WORKLIST_LAYOUT.dispatchIndirectOffsetBytes] },
      { kind: "direct", values: [3] },
    ]);
    assert.equal(resources.length, 11, "each pass binds the structural arena once; recurring work also binds its worklist");
    assert.equal(builder.encodeAllLeavesForInitialization(encoder, {
      brickSize: 4, leafCapacity: 129, structure: buffer, control: buffer, topology: buffer, payload: buffer,
    } as SparseBrickOctreeGPU), "unsupported-brick-size");
    assert.equal(dispatches.length, 3, "unsupported bricks preserve the current generation without dispatch");
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});

const webgpuModulePath = process.env.WEBGPU_NODE_MODULE;
test("brick occupancy build and traversal WGSL helpers compile on the target backend", {
  skip: !webgpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webgpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const build = device.createShaderModule({ code: webgpuSvoBrickOccupancyBuildWGSL });
    const helpers = device.createShaderModule({ code: `${svoBrickOccupancyWGSL}\n@compute @workgroup_size(1) fn validate(){let s=svoBrickOccupancyDecode(0u);let occupied=svoBrickMacroOccupied(s,vec3u(0u));let bounds=svoBrickOccupiedBounds(s,vec3f(0.0),vec3f(1.0));}` });
    const boundsVariant = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(1, "hybrid", "bounds") });
    const macroVariant = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(1, "hybrid", "macro") });
    const macroHddaVariant = device.createShaderModule({ code: createSvoDrySceneFragmentWGSL(1, "hybrid", "macro-hdda") });
    const messages = [
      ...(await build.getCompilationInfo()).messages,
      ...(await helpers.getCompilationInfo()).messages,
      ...(await boundsVariant.getCompilationInfo()).messages,
      ...(await macroVariant.getCompilationInfo()).messages,
      ...(await macroHddaVariant.getCompilationInfo()).messages,
    ].filter(({ type }) => type === "error");
    assert.deepEqual(messages, []);
  } finally {
    device.destroy();
  }
});
