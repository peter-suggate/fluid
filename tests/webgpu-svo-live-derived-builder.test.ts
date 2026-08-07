import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { SVO_NODE_MIP_LAYOUT, planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { createWebGpuSvoNodeMipDirectPageTable } from "../lib/webgpu-svo-node-mip-pyramid";
import {
  liveSvoBasePageDimensions,
  liveSvoDenseFinestPages,
  sparseSceneOctreeMaximumDepth,
} from "../lib/webgpu-octree-sparse-bricks";
import {
  LIVE_SVO_DERIVED_WORKLIST,
  WebGpuLiveSvoDerivedBuilder,
  WebGpuLiveSvoDerivedWorklistPlanner,
  liveSvoDerivedBuildWGSL,
  liveSvoDerivedCopyWGSL,
  liveSvoDerivedEmptyInitializationWGSL,
  liveSvoLeafBasePages,
  liveSvoLeafPage,
  liveSvoRadianceFeedbackWGSL,
  liveSvoDerivedWorklistWGSL,
} from "../lib/webgpu-svo-live-derived-builder";

function installGpuConstants() {
  const previousTexture = Object.getOwnPropertyDescriptor(globalThis, "GPUTextureUsage");
  const previousBuffer = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  Object.defineProperty(globalThis, "GPUTextureUsage", { configurable: true, value: { TEXTURE_BINDING: 1, STORAGE_BINDING: 2, COPY_DST: 4 } });
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, INDIRECT: 8 } });
  return () => {
    if (previousTexture) Object.defineProperty(globalThis, "GPUTextureUsage", previousTexture); else delete (globalThis as { GPUTextureUsage?: unknown }).GPUTextureUsage;
    if (previousBuffer) Object.defineProperty(globalThis, "GPUBufferUsage", previousBuffer); else delete (globalThis as { GPUBufferUsage?: unknown }).GPUBufferUsage;
  };
}

class TextureMock {
  destroyed = false;
  constructor(readonly descriptor: GPUTextureDescriptor) {}
  createView() { return { texture: this } as unknown as GPUTextureView; }
  destroy() { this.destroyed = true; }
}
class BufferMock {
  destroyed = false;
  constructor(readonly descriptor: GPUBufferDescriptor = { size: 4096, usage: 0 }) {}
  destroy() { this.destroyed = true; }
}

function fixture() {
  const textures: TextureMock[] = [], buffers: BufferMock[] = [], bindGroups: GPUBindGroupDescriptor[] = [], writes: unknown[][] = [];
  const pipelines: GPUComputePipelineDescriptor[] = [];
  const createPipeline = (descriptor: GPUComputePipelineDescriptor) => {
    pipelines.push(descriptor); return { getBindGroupLayout: () => ({ entry: descriptor.compute.entryPoint }) };
  };
  const device = {
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) },
    createTexture: (descriptor: GPUTextureDescriptor) => { const value = new TextureMock(descriptor); textures.push(value); return value; },
    createBuffer: (descriptor: GPUBufferDescriptor) => { const value = new BufferMock(descriptor); buffers.push(value); return value; },
    createShaderModule: () => ({}),
    createSampler: () => ({}),
    createComputePipeline: createPipeline,
    createComputePipelineAsync: async (descriptor: GPUComputePipelineDescriptor) => createPipeline(descriptor),
    createBindGroup: (descriptor: GPUBindGroupDescriptor) => { bindGroups.push(descriptor); return descriptor; },
  } as unknown as GPUDevice;
  const targetTexture = () => new TextureMock({ size: [40, 10, 10], dimension: "3d", format: "rgba8unorm", usage: 0 }) as unknown as GPUTexture;
  const validityTexture = new TextureMock({ size: [4, 1], format: "r32uint", usage: 0 });
  const treeBuffer = new BufferMock() as unknown as GPUBuffer;
  const worklists = [new BufferMock(), new BufferMock()].map((buffer) => ({ buffer: buffer as unknown as GPUBuffer, capacity: 2 }));
  const nodeMips = {
    texture: targetTexture(), pageValidity: { texture: validityTexture as unknown as GPUTexture, view: validityTexture.createView(), format: "r32uint" as const, capacity: 2 },
    atlasPages: [4, 1, 1] as const, pageCapacity: 4,
    directPageTableTexture: targetTexture(), directPageTableDimensions: [4, 1, 1] as const,
    directPageTableLevelZOffsets: new Uint32Array(12),
  };
  const radianceTextures = [targetTexture(), targetTexture(), targetTexture(), targetTexture()] as [GPUTexture, GPUTexture, GPUTexture, GPUTexture];
  const radianceValidity = new TextureMock({ size: [4, 1], format: "r32uint", usage: 0 });
  const radiance = { textures: radianceTextures, pageValidity: { texture: radianceValidity as unknown as GPUTexture,
    view: radianceValidity.createView(), format: "r32uint" as const, capacity: 4 }, atlasPages: [4, 1, 1] as const, pageCapacity: 4 };
  const tree = { control: treeBuffer, topology: treeBuffer, payload: treeBuffer,
    leafCapacity: 2, leafPayloadMode: "dense" as const,
    velocityOffsetBytes: 256, materialOwnerOffsetBytes: 512, sceneGeometryOffsetBytes: 768,
    bandedLaneWordOffsets: [0, 0, 0, 0, 0] as const,
    scenePayloadLanes: { mode: "dense" as const, materialOwnerWords: 256,
      occupancyWords: 0, recordMaskWords: 0, headerWords: 0, blobWords: 0, recordWords: 0 },
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  return { device, textures, buffers, bindGroups, pipelines, writes, nodeMips, radiance, tree, worklists, treeBuffer };
}

test("GPU builder ABI is bounded, level-ordered, and contains no CPU/readback construction path", () => {
  assert.equal(LIVE_SVO_DERIVED_WORKLIST.recordWords, 12);
  assert.match(liveSvoDerivedBuildWGSL, /let count=min\(worklist\[0\],params\.limits\.x\)/);
  assert.match(liveSvoDerivedBuildWGSL, /textureStore\(opacityScratch/);
  assert.match(liveSvoDerivedBuildWGSL, /worklist\[2\]==0u/);
  assert.match(liveSvoDerivedBuildWGSL, /childRadiance/);
  assert.match(liveSvoDerivedBuildWGSL, /dynamicIdentity&0xffffu\)==1u[^]*sceneIdentity&0xffffu\)==1u[^]*let solid=1\.-\(1\.-dynamicSolid\)\*\(1\.-sceneSolid\)/,
    "container glass must remain out of the cone-opacity hierarchy so the tank casts no projected cutout");
  assert.match(liveSvoDerivedBuildWGSL, /params\.laneOffsets\.z\+voxel\*4u/);
  assert.match(liveSvoDerivedBuildWGSL, /referencedChildrenReady/);
  assert.match(liveSvoDerivedBuildWGSL, /textureLoad\(opacityValidity[^]*==0u/);
  assert.match(liveSvoDerivedBuildWGSL, /textureLoad\(radianceValidity[^]*==0u/);
  assert.match(liveSvoDerivedCopyWGSL, /textureStore\(opacityValidity[^]*vec4u\(0u\)/);
  assert.match(liveSvoDerivedCopyWGSL, /scratchValidity\[recordIndex\]!=0u/);
  assert.match(liveSvoDerivedEmptyInitializationWGSL, /slot>=params\.limits\.w/);
  assert.match(liveSvoDerivedEmptyInitializationWGSL, /generationState\[params\.limits\.z\]/);
  assert.match(liveSvoDerivedEmptyInitializationWGSL, /textureStore\(opacityValidity/);
  assert.match(liveSvoDerivedEmptyInitializationWGSL, /textureStore\(radianceValidity/);
  assert.match(liveSvoDerivedWorklistWGSL,
    /let levelStart=params\.zOffsets\[level\];var levelEnd=dims\.z;if\(level\+1u<params\.domain\.y\)\{levelEnd=params\.zOffsets\[level\+1u\];\}if\(p\.z>=levelEnd-levelStart\)\{return INVALID;\}/,
    "a padded leaf must not address the next virtual level's direct-table Z slab");
  assert.doesNotMatch(liveSvoDerivedBuildWGSL + liveSvoDerivedCopyWGSL, /mapAsync|getMappedRange|array<ScenePrimitive>|for\(var primitive/);
});

test("GPU planner unions multiple dirty streams, deduplicates by shared generation, and can initialize from all live leaves", async () => {
  assert.match(liveSvoDerivedWorklistWGSL, /atomicExchange\(&claims\[slot\],generation\)/);
  assert.match(liveSvoDerivedWorklistWGSL, /let allLive=params\.limits\.w!=0u/);
  assert.match(liveSvoDerivedWorklistWGSL, /generationState\[params\.source\.y\]/);
  const restore = installGpuConstants();
  try {
    const mock = fixture();
    const dirtyA = new BufferMock(), dirtyB = new BufferMock(), generation = new BufferMock();
    const planner = new WebGpuLiveSvoDerivedWorklistPlanner(mock.device, {
      tree: mock.tree, nodeMips: mock.nodeMips,
      dirtyLeafSources: [
        { buffer: dirtyA as unknown as GPUBuffer, countOffsetBytes: 0, recordOffsetBytes: 32, capacity: 65, recordStrideWords: 2 },
        { buffer: dirtyB as unknown as GPUBuffer, countOffsetBytes: 4, recordOffsetBytes: 64, capacity: 17, recordStrideWords: 1 },
      ],
      generationSource: { buffer: generation as unknown as GPUBuffer, offsetBytes: 8 },
      levelCount: 2, finestLevel: 4, pageCapacityPerLevel: 2,
    });
    await planner.initializePipelines();
    assert.equal(planner.worklists.length, 2);
    assert.deepEqual(planner.worklists.map(({ indirectOffsetBytes }) => indirectOffsetBytes), [16, 272]);
    const clears: Array<[number | undefined, number | undefined]> = [], dispatches: number[] = [];
    const encoder = {
      clearBuffer: (_buffer: GPUBuffer, offset?: number, size?: number) => clears.push([offset, size]),
      beginComputePass: () => ({ setPipeline: () => undefined, setBindGroup: () => undefined,
        dispatchWorkgroups: (count: number) => dispatches.push(count), end: () => undefined }),
    } as unknown as GPUCommandEncoder;
    const allocations = [mock.buffers.length, mock.bindGroups.length];
    planner.encode(encoder);
    assert.deepEqual(dispatches, [2, 1, 2], "two bounded dirty sources are populated before one two-level finalize");
    dispatches.length = 0; planner.encodeInitial(encoder);
    assert.deepEqual(dispatches, [1, 2, 1, 2], "initial publication adds the fixed-capacity live-leaf source to the same union");
    dispatches.length = 0; planner.encodeRadianceFeedback(encoder, 2, 4);
    assert.deepEqual(dispatches, [1, 2], "feedback compacts one partitioned all-leaf source then finalizes both levels");
    assert.deepEqual(clears.slice(1, 3), [[0, 32], [256, 32]], "slot claims clear before the two worklist headers");
    assert.deepEqual([mock.buffers.length, mock.bindGroups.length], allocations, "planner encoding allocates no resources");
    planner.destroy();
  } finally { restore(); }
});

test("4- and 8-cell leaves map generically into fixed 8-cell pages and arbitrary level counts", () => {
  assert.deepEqual(liveSvoLeafBasePages({ coordinate: [6, 4, 2], leafLevel: 5, finestLevel: 5, brickSize: 4 }), [[3, 2, 1]]);
  assert.deepEqual(liveSvoLeafBasePages({ coordinate: [7, 5, 3], leafLevel: 5, finestLevel: 5, brickSize: 4 }), [[3, 2, 1]],
    "eight finest 4-cell leaves can contribute to one base page");
  assert.deepEqual(liveSvoLeafBasePages({ coordinate: [3, 2, 1], leafLevel: 5, finestLevel: 5, brickSize: 8 }), [[3, 2, 1]]);
  assert.deepEqual(liveSvoLeafBasePages({ coordinate: [1, 0, 1], leafLevel: 4, finestLevel: 5, brickSize: 8 }), [
    [2, 0, 2], [3, 0, 2], [2, 1, 2], [3, 1, 2],
    [2, 0, 3], [3, 0, 3], [2, 1, 3], [3, 1, 3],
  ], "the legacy extent expansion covers every base page a coarse leaf spans");

  // What the pyramid is actually seeded from: one page per leaf, at the level
  // whose texels are its voxels. The eight base pages above hold the same 8^3
  // voxels between them; page [1,0,1] at level 1 holds them once.
  assert.deepEqual(liveSvoLeafPage({ coordinate: [3, 2, 1], leafLevel: 5, finestLevel: 5, brickSize: 8 }),
    { level: 0, coordinate: [3, 2, 1] }, "a finest 8-cell leaf is one base page");
  assert.deepEqual(liveSvoLeafPage({ coordinate: [1, 0, 1], leafLevel: 4, finestLevel: 5, brickSize: 8 }),
    { level: 1, coordinate: [1, 0, 1] }, "a coarse leaf is one page at its own level, not 8^p base pages");
  assert.deepEqual(liveSvoLeafPage({ coordinate: [7, 5, 3], leafLevel: 5, finestLevel: 5, brickSize: 4 }),
    { level: 0, coordinate: [3, 2, 1] }, "a brick smaller than a page still lands in the page containing it");

  for (const [brickSize, brickDimensions, expectedPages] of [
    [4, [5, 3, 7], [3, 2, 4]],
    [8, [5, 3, 7], [5, 3, 7]],
  ] as const) {
    const baseDimensions = liveSvoBasePageDimensions(brickDimensions, brickSize);
    assert.deepEqual(baseDimensions, expectedPages);
    const levelCount = sparseSceneOctreeMaximumDepth(baseDimensions, []) + 1;
    const plan = planSvoNodeMipPyramid({
      generation: 9, occupiedPages: liveSvoDenseFinestPages(baseDimensions), levelCount,
    });
    assert.equal(plan.complete, true);
    assert.equal(plan.pages.filter(({ key }) => key.level === 0).length,
      expectedPages.reduce((product, value) => product * value, 1));
    assert.equal(Math.max(...plan.pages.map(({ key }) => key.level)) + 1, levelCount);
  }
  assert.match(liveSvoDerivedWorklistWGSL, /cellMinimum=brickOrigin\*control\[11\]/);
  // The GPU mirror of `liveSvoLeafPage`: one seed level per leaf, walked up.
  assert.match(liveSvoDerivedWorklistWGSL, /seedLevel=firstTrailingBit\(cells\)-3u/);
  assert.match(liveSvoDerivedWorklistWGSL, /page=cellMinimum\/\(8u<<seedLevel\)/);
  assert.match(liveSvoDerivedWorklistWGSL, /for\(var level=seedLevel;level<params\.domain\.y/);
  assert.match(liveSvoDerivedWorklistWGSL, /fn deepestLeaf\(globalCell:vec3u\)/);
  assert.match(liveSvoDerivedBuildWGSL, /fn leafLocal\(globalCell:vec3u,leaf:u32\)/);
});

test("GPU builder preallocates scratch resources and encodes invalidate then finest-to-coarsest indirect work", async () => {
  const restore = installGpuConstants();
  try {
    const mock = fixture();
    const builder = new WebGpuLiveSvoDerivedBuilder(mock.device, {
      tree: mock.tree, nodeMips: mock.nodeMips, radiance: mock.radiance,
      materialEmission: mock.treeBuffer, worklists: mock.worklists,
      generationSource: { buffer: mock.treeBuffer, offsetBytes: 8 }, plannedPageCount: 4,
      finestLevel: 4,
    });
    await builder.initializePipelines();
    assert.deepEqual(mock.textures.map((texture) => texture.descriptor.format), ["rgba8unorm", "rgba16float"]);
    assert.deepEqual(builder.scratchAtlasPages, [2, 1, 1], "scratch follows the largest dirty-level budget, not target atlas capacity");
    const physicalSize = SVO_NODE_MIP_LAYOUT.physicalSize;
    assert.deepEqual(mock.textures.map((texture) => texture.descriptor.size),
      [[2 * physicalSize, physicalSize, physicalSize], [2 * physicalSize, physicalSize, 4 * physicalSize]]);
    assert.equal(mock.buffers.length, 2, "only fixed scratch-validity and parameter buffers are allocated by the builder");
    assert.deepEqual([...new Uint32Array(mock.writes[0][2] as ArrayBuffer).slice(12, 16)], [64, 128, 192, 256],
      "fluid/dynamic and scene geometry/material lanes remain disjoint inputs");
    const labels: string[] = [], direct: number[] = [], indirect: Array<[unknown, number]> = [];
    const encoder = { beginComputePass: (descriptor: GPUComputePassDescriptor) => {
      labels.push(descriptor.label ?? ""); return { setPipeline: () => undefined, setBindGroup: () => undefined,
        dispatchWorkgroups: (x: number) => direct.push(x),
        dispatchWorkgroupsIndirect: (buffer: unknown, offset: number) => indirect.push([buffer, offset]), end: () => undefined };
    } } as unknown as GPUCommandEncoder;
    const allocations = [mock.textures.length, mock.buffers.length, mock.bindGroups.length];
    builder.encode(encoder); builder.encode(encoder);
    assert.deepEqual([mock.textures.length, mock.buffers.length, mock.bindGroups.length], allocations, "frame encoding allocates no GPU resources");
    assert.deepEqual(labels.slice(0, 6), [
      "Invalidate live SVO derived pages", "Invalidate live SVO derived pages",
      "Build live SVO derived pages", "Publish live SVO opacity pages",
      "Publish live SVO radiance lobe 0", "Publish live SVO radiance lobe 1",
    ]);
    assert.deepEqual(direct.slice(0, 2), [1, 1]);
    assert.deepEqual(indirect.slice(0, 4).map((entry) => entry[1]), [16, 16, 16, 16]);
    labels.length = 0; direct.length = 0; indirect.length = 0;
    builder.encode(encoder, true);
    assert.equal(labels[0], "Certify live SVO address pages as empty");
    assert.equal(direct[0], 1, "all four planned pages are certified before occupied work begins");
    assert.equal(labels[1], "Invalidate live SVO derived pages",
      "occupied pages overwrite certified-empty pages through ordinary fail-closed rebuild ordering");
    builder.destroy(); assert.ok(mock.textures.every((texture) => texture.destroyed)); assert.ok(mock.buffers.every((buffer) => buffer.destroyed));
  } finally { restore(); }
});

test("GPU builder encodes opt-in temporal radiance feedback without touching opacity", async () => {
  const restore = installGpuConstants();
  try {
    const mock = fixture();
    const builder = new WebGpuLiveSvoDerivedBuilder(mock.device, {
      tree: mock.tree, nodeMips: mock.nodeMips, radiance: mock.radiance,
      materialEmission: mock.treeBuffer, materialPbr: mock.treeBuffer, environmentLighting: mock.treeBuffer,
      lights: mock.treeBuffer, lightCount: 1,
      radianceFeedback: true,
      worklists: mock.worklists, generationSource: { buffer: mock.treeBuffer, offsetBytes: 8 },
      plannedPageCount: 4, finestLevel: 4,
    });
    await builder.initializePipelines();
    assert.equal(builder.radianceFeedbackEnabled, true);
    assert.equal(mock.buffers.length, 3, "feedback adds only its fixed parameter buffer");
    const labels: string[] = [], indirect: number[] = [];
    const encoder = { beginComputePass: (descriptor: GPUComputePassDescriptor) => {
      labels.push(descriptor.label ?? ""); return { setPipeline: () => undefined, setBindGroup: () => undefined,
        dispatchWorkgroupsIndirect: (_buffer: unknown, offset: number) => indirect.push(offset), end: () => undefined };
    } } as unknown as GPUCommandEncoder;
    builder.encodeRadianceFeedback(encoder);
    assert.equal(labels.filter((label) => label === "Feed back live SVO diffuse radiance").length, 2);
    assert.equal(labels.some((label) => label.includes("opacity")), false);
    assert.equal(indirect.length, 10, "each level runs one feedback build and four lobe publications");
    builder.destroy();
  } finally { restore(); }
});

const webGpuModulePath = process.env.WEBGPU_NODE_MODULE;
test("GPU valid-empty initialization certifies every planned garden page", {
  skip: !webGpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webGpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const plannedPageCount = 4_455;
  const generation = 37;
  const bytesPerRow = Math.ceil(plannedPageCount * 4 / 256) * 256;
  const validity = [0, 1].map((index) => device.createTexture({
    label: `test live validity ${index}`, size: [plannedPageCount, 1], format: "r32uint",
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
  }));
  const generationState = device.createBuffer({ size: 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const params = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const readbacks = [0, 1].map(() => device.createBuffer({
    size: bytesPerRow, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  }));
  try {
    device.queue.writeBuffer(generationState, 0, new Uint32Array([generation]));
    const parameterWords = new Uint32Array(16); parameterWords[10] = 0; parameterWords[11] = plannedPageCount;
    device.queue.writeBuffer(params, 0, parameterWords);
    const shaderModule = device.createShaderModule({ code: liveSvoDerivedEmptyInitializationWGSL });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    for (const code of [liveSvoDerivedWorklistWGSL, liveSvoDerivedBuildWGSL, liveSvoDerivedCopyWGSL, liveSvoRadianceFeedbackWGSL]) {
      const validationModule = device.createShaderModule({ code });
      const validationInfo = await validationModule.getCompilationInfo();
      assert.deepEqual(validationInfo.messages.filter(({ type }) => type === "error"), []);
    }
    const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "initializeValidEmptyPages" } });
    const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: validity[0].createView() }, { binding: 1, resource: validity[1].createView() },
      { binding: 2, resource: { buffer: generationState } }, { binding: 3, resource: { buffer: params } },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(plannedPageCount / 256)); pass.end();
    validity.forEach((texture, index) => encoder.copyTextureToBuffer(
      { texture }, { buffer: readbacks[index], bytesPerRow }, [plannedPageCount, 1],
    ));
    device.queue.submit([encoder.finish()]);
    await Promise.all(readbacks.map((buffer) => buffer.mapAsync(GPUMapMode.READ)));
    for (const buffer of readbacks) {
      const words = new Uint32Array(buffer.getMappedRange(), 0, plannedPageCount);
      assert.equal(words.every((value) => value === generation), true,
        "opacity and radiance validity cover every address-resident garden page");
      buffer.unmap();
    }
  } finally {
    validity.forEach((texture) => texture.destroy()); generationState.destroy(); params.destroy();
    readbacks.forEach((buffer) => buffer.destroy()); device.destroy();
  }
});

test("GPU initial live-derived publication covers a compact non-power-of-two hierarchy", {
  skip: !webGpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webGpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const generation = 41, finestLevel = 2, brickSize = 8;
  const plan = planSvoNodeMipPyramid({
    generation: 1,
    occupiedPages: liveSvoDenseFinestPages([3, 2, 3]),
    levelCount: 3,
  });
  assert.deepEqual([0, 1, 2].map((level) => plan.pages.filter((page) => page.key.level === level).length), [18, 4, 1]);
  const direct = createWebGpuSvoNodeMipDirectPageTable(plan, device.limits.maxTextureDimension3D);
  assert.equal(direct.ready, true);

  const gpuBuffer = (size: number, usage: GPUBufferUsageFlags, words?: Uint32Array<ArrayBuffer>) => {
    const buffer = device.createBuffer({ size: Math.max(16, size), usage });
    if (words) device.queue.writeBuffer(buffer, 0, words);
    return buffer;
  };
  const nodeCount = 1 + 8 + 64, leafCount = 64, leafRecordWordOffset = nodeCount * 8;
  const controlWords = new Uint32Array(64);
  controlWords[0] = nodeCount; controlWords[1] = leafCount; controlWords[11] = brickSize; controlWords[16] = leafRecordWordOffset;
  const topologyWords = new Uint32Array(leafRecordWordOffset + leafCount * 4);
  topologyWords[2] = 0; topologyWords[3] = 0xff; topologyWords[4] = 1; topologyWords[6] = 0xffff_ffff;
  for (let parent = 0; parent < 8; parent += 1) {
    const node = 1 + parent, nodeBase = node * 8;
    topologyWords[nodeBase] = parent; topologyWords[nodeBase + 2] = 1;
    topologyWords[nodeBase + 3] = 0xff; topologyWords[nodeBase + 4] = 9 + parent * 8;
    topologyWords[nodeBase + 6] = 0xffff_ffff;
    for (let child = 0; child < 8; child += 1) {
      const leaf = parent * 8 + child, leafNode = 9 + leaf, leafNodeBase = leafNode * 8;
      topologyWords[leafNodeBase] = leaf; topologyWords[leafNodeBase + 2] = finestLevel;
      topologyWords[leafNodeBase + 4] = 0xffff_ffff; topologyWords[leafNodeBase + 6] = leaf;
      topologyWords[leafNodeBase + 7] = 0x1000_0000;
      const leafBase = leafRecordWordOffset + leaf * 4;
      topologyWords[leafBase] = leafNode; topologyWords[leafBase + 1] = leaf * brickSize ** 3;
      topologyWords[leafBase + 2] = leaf;
    }
  }
  const voxelCount = leafCount * brickSize ** 3, laneWords = voxelCount * 4;
  const fluidOffset = laneWords, ownerOffset = fluidOffset + laneWords;
  const sceneOffset = ownerOffset + voxelCount, sceneOwnerOffset = sceneOffset + laneWords;
  const payloadWords = new Uint32Array(sceneOwnerOffset + voxelCount);
  const control = gpuBuffer(controlWords.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, controlWords);
  const topology = gpuBuffer(topologyWords.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, topologyWords);
  const payload = gpuBuffer(payloadWords.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, payloadWords);
  const generationState = gpuBuffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array([generation]));
  const dirty = gpuBuffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(4));
  const emission = gpuBuffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(4));
  const pbr = gpuBuffer(96, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(24));
  const environment = gpuBuffer(96, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(24));
  const feedbackLights = gpuBuffer(112, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, new Uint32Array(28));
  const directTable = device.createTexture({ size: direct.dimensions, dimension: "3d", format: "r32uint",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  device.queue.writeTexture({ texture: directTable }, direct.words,
    { bytesPerRow: direct.dimensions[0] * 4, rowsPerImage: direct.dimensions[1] }, direct.dimensions);
  const atlasSize = plan.atlas.texels as [number, number, number];
  const opacity = device.createTexture({ size: atlasSize, dimension: "3d", format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING });
  const radianceTextures = [0, 1, 2, 3].map(() => device.createTexture({ size: atlasSize, dimension: "3d", format: "rgba16float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING }));
  const validity = [0, 1].map(() => device.createTexture({ size: [plan.pages.length, 1], format: "r32uint",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC }));
  const tree = {
    control, topology, topologyOffsetBytes: 0, payload, leafCapacity: leafCount,
    velocityOffsetBytes: fluidOffset * 4, materialOwnerOffsetBytes: ownerOffset * 4,
    sceneGeometryOffsetBytes: sceneOffset * 4, leafPayloadMode: "dense" as const,
    bandedLaneWordOffsets: [0, 0, 0, 0, 0] as const,
    scenePayloadLanes: { mode: "dense" as const, materialOwnerWords: sceneOwnerOffset,
      occupancyWords: 0, recordMaskWords: 0, headerWords: 0, blobWords: 0, recordWords: 0 },
  } as unknown as import("../lib/sparse-brick-octree").SparseBrickOctreeGPU;
  const nodeTarget = {
    texture: opacity, pageValidity: { texture: validity[0], view: validity[0].createView(), format: "r32uint" as const, capacity: plan.pages.length },
    atlasPages: plan.atlas.pages, pageCapacity: plan.pages.length,
    directPageTableTexture: directTable, directPageTableDimensions: direct.dimensions,
    directPageTableLevelZOffsets: direct.levelZOffsets,
  };
  const radianceTarget = {
    textures: radianceTextures as [GPUTexture, GPUTexture, GPUTexture, GPUTexture],
    pageValidity: { texture: validity[1], view: validity[1].createView(), format: "r32uint" as const, capacity: plan.pages.length },
    atlasPages: plan.atlas.pages, pageCapacity: plan.pages.length,
  };
  const generationSource = { buffer: generationState, offsetBytes: 0 };
  const planner = new WebGpuLiveSvoDerivedWorklistPlanner(device, {
    tree, nodeMips: nodeTarget, dirtyLeafSources: [{ buffer: dirty, countOffsetBytes: 0, recordOffsetBytes: 4, capacity: 1, recordStrideWords: 1 }],
    generationSource, levelCount: 3, finestLevel, pageCapacityPerLevel: plan.pages.length,
  });
  const builder = new WebGpuLiveSvoDerivedBuilder(device, {
    tree, nodeMips: nodeTarget, radiance: radianceTarget, materialEmission: emission,
    materialPbr: pbr, environmentLighting: environment, lights: feedbackLights, lightCount: 0,
    worklists: planner.worklists, generationSource, plannedPageCount: plan.pages.length, finestLevel,
  });
  await Promise.all([planner.initializePipelines(), builder.initializePipelines()]);
  const expectedLevels = gpuBuffer(plan.pages.length * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    new Uint32Array(plan.pages.map(({ key }) => key.level)));
  const ownershipResults = gpuBuffer(16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
    new Uint32Array(4));
  const ownershipModule = device.createShaderModule({ code: /* wgsl */ `
@group(0) @binding(0) var<storage,read> worklist:array<u32>;
@group(0) @binding(1) var<storage,read> expectedLevels:array<u32>;
@group(0) @binding(2) var<storage,read_write> results:array<atomic<u32>>;
@compute @workgroup_size(64) fn validate(@builtin(global_invocation_id) gid:vec3u){
  let count=worklist[0];let level=worklist[2];
  if(gid.x==0u){atomicStore(&results[1u+level],count);}
  if(gid.x>=count){return;}let slot=worklist[8u+gid.x*12u];
  if(slot>=arrayLength(&expectedLevels)||expectedLevels[slot]!=level){atomicAdd(&results[0],1u);}
}` });
  const ownershipPipeline = device.createComputePipeline({ layout: "auto", compute: { module: ownershipModule, entryPoint: "validate" } });
  const ownershipBindGroups = planner.worklists.map((worklist) => device.createBindGroup({
    layout: ownershipPipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: worklist.buffer, offset: worklist.bindingOffsetBytes, size: worklist.bindingSizeBytes } },
      { binding: 1, resource: { buffer: expectedLevels } }, { binding: 2, resource: { buffer: ownershipResults } },
    ],
  }));
  const bytesPerRow = 256;
  const readbacks = validity.map(() => device.createBuffer({ size: bytesPerRow, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
  const ownershipReadback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); planner.encodeInitial(encoder);
    ownershipBindGroups.forEach((bindGroup) => {
      const pass = encoder.beginComputePass(); pass.setPipeline(ownershipPipeline); pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(plan.pages.length / 64)); pass.end();
    });
    builder.encode(encoder, true);
    planner.encodeRadianceFeedback(encoder, 0, 4); builder.encodeRadianceFeedback(encoder);
    validity.forEach((texture, index) => encoder.copyTextureToBuffer({ texture }, { buffer: readbacks[index], bytesPerRow }, [plan.pages.length, 1]));
    encoder.copyBufferToBuffer(ownershipResults, 0, ownershipReadback, 0, 16);
    device.queue.submit([encoder.finish()]);
    await Promise.all([...readbacks, ownershipReadback].map((buffer) => buffer.mapAsync(GPUMapMode.READ)));
    assert.deepEqual([...new Uint32Array(ownershipReadback.getMappedRange())], [0, 18, 4, 1],
      "every planned slot must be emitted by its own virtual mip level exactly once");
    ownershipReadback.unmap();
    for (const buffer of readbacks) {
      const words = new Uint32Array(buffer.getMappedRange(), 0, plan.pages.length);
      assert.deepEqual([...words], Array(plan.pages.length).fill(generation));
      buffer.unmap();
    }
  } finally {
    builder.destroy(); planner.destroy(); readbacks.forEach((buffer) => buffer.destroy()); ownershipReadback.destroy(); validity.forEach((texture) => texture.destroy());
    radianceTextures.forEach((texture) => texture.destroy()); opacity.destroy(); directTable.destroy();
    [control, topology, payload, generationState, dirty, emission, pbr, environment, feedbackLights, expectedLevels, ownershipResults].forEach((buffer) => buffer.destroy()); device.destroy();
  }
});

test("GPU base-page construction samples canonical 4- and 8-cell topologies", {
  skip: !webGpuModulePath && "set WEBGPU_NODE_MODULE for GPU validation",
}, async () => {
  const { create, globals } = await import(pathToFileURL(webGpuModulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const shaderModule = device.createShaderModule({ code: liveSvoDerivedBuildWGSL });
  const info = await shaderModule.getCompilationInfo();
  assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "buildPages" } });

  const run = async (brickSize: 4 | 8) => {
    const finestLevel = brickSize === 4 ? 1 : 0;
    const leafCount = brickSize === 4 ? 8 : 1;
    const nodeCount = brickSize === 4 ? 9 : 1;
    const leafRecordOffset = nodeCount * 8;
    const voxelsPerLeaf = brickSize ** 3;
    const voxelCount = leafCount * voxelsPerLeaf;
    const controlWords = new Uint32Array(32);
    controlWords[0] = nodeCount; controlWords[1] = leafCount; controlWords[11] = brickSize; controlWords[16] = leafRecordOffset;
    const topologyWords = new Uint32Array(leafRecordOffset + leafCount * 4);
    topologyWords[2] = 0; topologyWords[4] = brickSize === 4 ? 1 : 0xffff_ffff;
    topologyWords[3] = brickSize === 4 ? 0xff : 0;
    topologyWords[6] = brickSize === 8 ? 0 : 0xffff_ffff;
    topologyWords[7] = brickSize === 8 ? 0x1000_0000 : 0;
    for (let leaf = 0; leaf < leafCount; leaf += 1) {
      const node = brickSize === 4 ? leaf + 1 : 0;
      if (brickSize === 4) {
        const nodeBase = node * 8;
        topologyWords[nodeBase] = leaf; topologyWords[nodeBase + 2] = 1;
        topologyWords[nodeBase + 4] = 0xffff_ffff; topologyWords[nodeBase + 6] = leaf;
        topologyWords[nodeBase + 7] = 0x1000_0000;
      }
      const leafBase = leafRecordOffset + leaf * 4;
      topologyWords[leafBase] = node; topologyWords[leafBase + 1] = leaf * voxelsPerLeaf;
      topologyWords[leafBase + 2] = brickSize === 4 ? leaf : 0;
    }
    const laneWords = voxelCount * 4;
    const fluidOffset = laneWords, ownerOffset = fluidOffset + laneWords;
    const sceneOffset = ownerOffset + voxelCount, sceneOwnerOffset = sceneOffset + laneWords;
    const payloadWords = new Uint32Array(sceneOwnerOffset + voxelCount);
    const payloadFloats = new Float32Array(payloadWords.buffer);
    for (let leaf = 0; leaf < leafCount; leaf += 1) for (let z = 0; z < brickSize; z += 1) {
      for (let y = 0; y < brickSize; y += 1) for (let x = 0; x < brickSize; x += 1) {
        const voxel = leaf * voxelsPerLeaf + x + y * brickSize + z * brickSize * brickSize;
        const occupied = brickSize === 4 ? leaf === 7 : x >= 4;
        payloadFloats[sceneOffset + voxel * 4 + 1] = occupied ? -1 : 1;
        payloadFloats[sceneOffset + voxel * 4 + 2] = occupied ? 1 : 0;
      }
    }
    const worklistWords = new Uint32Array(64);
    worklistWords.set([1, 11, 0, LIVE_SVO_DERIVED_WORKLIST.headerWords], 0);
    worklistWords.set([0, 0, 0, 0], LIVE_SVO_DERIVED_WORKLIST.headerWords);
    for (let octant = 0; octant < 8; octant += 1) {
      worklistWords[LIVE_SVO_DERIVED_WORKLIST.headerWords + LIVE_SVO_DERIVED_WORKLIST.sourceLeafWord + octant]
        = brickSize === 4 ? octant : 0;
    }
    const parameterWords = new Uint32Array(16);
    parameterWords.set([1, 1, 1, 0], 0); parameterWords.set([1, 1, 1, finestLevel], 4);
    parameterWords.set([1, 10, 0, 1], 8);
    parameterWords.set([fluidOffset, ownerOffset, sceneOffset, sceneOwnerOffset], 12);
    const gpuBuffer = (input: Uint32Array, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
      const data = new Uint32Array(input);
      const buffer = device.createBuffer({ size: Math.max(16, data.byteLength), usage });
      device.queue.writeBuffer(buffer, 0, data); return buffer;
    };
    const buffers = [gpuBuffer(controlWords), gpuBuffer(topologyWords), gpuBuffer(payloadWords), gpuBuffer(worklistWords),
      gpuBuffer(new Uint32Array(4)), gpuBuffer(parameterWords, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
      gpuBuffer(new Uint32Array(1))];
    const opacitySource = device.createTexture({ size: [10, 10, 10], dimension: "3d", format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING });
    const opacityScratch = device.createTexture({ size: [10, 10, 10], dimension: "3d", format: "rgba8unorm",
      usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
    const radianceSources = [0, 1, 2, 3].map(() => device.createTexture({ size: [10, 10, 10], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.TEXTURE_BINDING }));
    const radianceScratch = device.createTexture({ size: [10, 10, 40], dimension: "3d", format: "rgba16float", usage: GPUTextureUsage.STORAGE_BINDING });
    const validity = [0, 1].map(() => device.createTexture({ size: [1, 1], format: "r32uint", usage: GPUTextureUsage.TEXTURE_BINDING }));
    const bytesPerRow = 256, rowsPerImage = 10;
    const readback = device.createBuffer({ size: bytesPerRow * rowsPerImage * 10, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const bindGroup = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: buffers[0] } }, { binding: 1, resource: { buffer: buffers[1] } },
        { binding: 2, resource: { buffer: buffers[2] } }, { binding: 3, resource: { buffer: buffers[3] } },
        { binding: 4, resource: { buffer: buffers[4] } }, { binding: 5, resource: opacitySource.createView({ dimension: "3d" }) },
        { binding: 6, resource: opacityScratch.createView({ dimension: "3d" }) },
        ...radianceSources.map((texture, index) => ({ binding: 7 + index, resource: texture.createView({ dimension: "3d" }) })),
        { binding: 11, resource: radianceScratch.createView({ dimension: "3d" }) },
        { binding: 12, resource: { buffer: buffers[5] } }, { binding: 13, resource: validity[0].createView() },
        { binding: 14, resource: validity[1].createView() }, { binding: 15, resource: { buffer: buffers[6] } },
      ] });
      const encoder = device.createCommandEncoder(); const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroups(4); pass.end();
      encoder.copyTextureToBuffer({ texture: opacityScratch }, { buffer: readback, bytesPerRow, rowsPerImage }, [10, 10, 10]);
      device.queue.submit([encoder.finish()]); await readback.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(readback.getMappedRange());
      const solid = (x: number, y: number, z: number) => bytes[(z + 1) * rowsPerImage * bytesPerRow + (y + 1) * bytesPerRow + (x + 1) * 4];
      for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) for (let x = 0; x < 8; x += 1) {
        const expected = brickSize === 4 ? x >= 4 && y >= 4 && z >= 4 : x >= 4;
        assert.equal(solid(x, y, z), expected ? 255 : 0, `${brickSize}-cell topology at ${x},${y},${z}`);
      }
      readback.unmap();
    } finally {
      buffers.forEach((buffer) => buffer.destroy()); opacitySource.destroy(); opacityScratch.destroy();
      radianceSources.forEach((texture) => texture.destroy()); radianceScratch.destroy(); validity.forEach((texture) => texture.destroy()); readback.destroy();
    }
  };
  try { await run(4); await run(8); } finally { device.destroy(); }
});
