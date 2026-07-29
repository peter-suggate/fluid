import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import type { SparseBrickOctreeGPU } from "../lib/sparse-brick-octree";
import {
  SVO_BRICK_OCCUPANCY,
  buildSvoBrickOccupancy,
  decodeSvoBrickOccupancy,
  encodeSvoBrickOccupancy,
  svoBrickOccupancyRetainedCellCount,
  svoBrickOccupancyWGSL,
} from "../lib/svo-brick-occupancy";
import {
  WebGpuSvoBrickOccupancyBuilder,
  webgpuSvoBrickOccupancyBuildWGSL,
} from "../lib/webgpu-svo-brick-occupancy";
import { createSvoDrySceneFragmentWGSL, svoDrySceneShader } from "../lib/webgpu-svo-dry-scene";
import { OctreeSparseBrickWorld } from "../lib/webgpu-octree-sparse-bricks";

const cell = (x: number, y: number, z: number) => x + 8 * (y + 8 * z);

test("8^3 occupancy metadata round-trips every bound and macrocell bit", () => {
  const summary = {
    ready: true,
    occupied: true,
    macroMask: 0b1010_0101,
    minInclusive: [1, 2, 3] as const,
    maxInclusive: [7, 6, 5] as const,
  };
  const packed = encodeSvoBrickOccupancy(summary);
  assert.deepEqual(decodeSvoBrickOccupancy(packed), summary);
  assert.equal(packed >>> 28, 0, "high four node-flag bits remain reserved");
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
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /if\(control\[11\]!=8u\)\{topology\[nodeIndex\*8u\+7u\]=0u;return;\}/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /let material=payload\[payloadIndex\]&0xffffu/);
  assert.match(webgpuSvoBrickOccupancyBuildWGSL, /topology\[nodeIndex\*8u\+7u\]=packed/);
  assert.match(svoBrickOccupancyWGSL, /if\(summary\.ready==0u\)\{return true;\}/);
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

test("GPU builder reuses topology flags and allocates no persistent buffer", () => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1 } });
  const dispatches: number[][] = [];
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
    dispatchWorkgroups: (...dimensions: number[]) => dispatches.push(dimensions),
    end: () => undefined,
  };
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const buffer = {} as GPUBuffer;
  try {
    const builder = new WebGpuSvoBrickOccupancyBuilder(device);
    assert.equal(builder.incrementalAllocatedBytes, 0);
    assert.equal(builder.encode(encoder, {
      brickSize: 8, leafCapacity: 129, control: buffer, topology: buffer, payload: buffer,
    } as SparseBrickOctreeGPU), "encoded");
    assert.deepEqual(dispatches, [[3]]);
    assert.equal(resources.length, 3);
    assert.equal(builder.encode(encoder, {
      brickSize: 4, leafCapacity: 129, control: buffer, topology: buffer, payload: buffer,
    } as SparseBrickOctreeGPU), "unsupported-brick-size");
    assert.deepEqual(dispatches, [[3]], "unsupported bricks preserve the baseline without dispatch");
  } finally {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage);
    else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  }
});

test("world publication rebuilds occupancy after every authoritative material-owner write", () => {
  const initial = OctreeSparseBrickWorld.prototype.encodeStaticPublication.toString();
  assert.ok(initial.indexOf("proxyVoxelizer.encode") < initial.indexOf("brickOccupancyBuilder.encode"));
  assert.ok(initial.indexOf("brickOccupancyBuilder.encode") < initial.indexOf("structuralStaticPipeline"));
  const dynamic = OctreeSparseBrickWorld.prototype.encode.toString();
  assert.ok(dynamic.indexOf("tree.encodeFromDenseFields") < dynamic.indexOf("brickOccupancyBuilder.encode"));
  assert.ok(dynamic.indexOf("brickOccupancyBuilder.encode") < dynamic.indexOf("structuralFramePipeline"));
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
