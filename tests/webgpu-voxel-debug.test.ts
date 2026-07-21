import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE,
  SPARSE_VOXEL_DEBUG_RECORD_STRIDE,
  SURFACE_VOXEL_MAXIMUM_INSTANCES_PER_RECORD,
  SparseVoxelDebugRenderer,
  surfaceVoxelInstanceCapacity,
  voxelDebugComputeShader,
  voxelDebugPlan,
  voxelDebugRenderShader
} from "../lib/webgpu-voxel-debug";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("public sparse voxel render source is structural and exposes only inspection modes", () => {
  const external = {} as GPUBuffer;
  const binding = { buffer: external };
  const source = {
    voxelRecords: binding, voxelCount: binding, brickRecords: binding, brickCount: binding, materials: binding,
    voxelCapacity: 64, brickCapacity: 8, materialCount: 3, revision: 7
  } satisfies SparseVoxelRenderSource;
  const modes = ["raw-voxels", "surface-voxels", "brick-grid"] as const;
  assert.deepEqual(modes.map((mode) => voxelDebugPlan(mode, source).recordKind), ["voxels", "surface-voxels", "bricks"]);
});

test("voxel inspection plans adaptive, finest-surface, and brick-grid views from one source", () => {
  const source = { voxelCapacity: 129, brickCapacity: 65 };
  assert.deepEqual(voxelDebugPlan("raw-voxels", source), {
    enabled: true, recordKind: "voxels", capacity: 129, computeWorkgroups: 3, verticesPerInstance: 36, topology: "triangle-list",
    overlayCapacity: 65, overlayWorkgroups: 2
  });
  assert.deepEqual(voxelDebugPlan("surface-voxels", source), {
    enabled: true, recordKind: "surface-voxels", capacity: 129, computeWorkgroups: 3, verticesPerInstance: 36, topology: "triangle-list",
    overlayCapacity: 65, overlayWorkgroups: 2
  });
  assert.deepEqual(voxelDebugPlan("brick-grid", source), {
    enabled: true, recordKind: "bricks", capacity: 65, computeWorkgroups: 2, verticesPerInstance: 24, topology: "line-list",
    overlayCapacity: 65, overlayWorkgroups: 2
  });
  assert.equal(voxelDebugPlan("brick-grid", { voxelCapacity: 1, brickCapacity: 0 }).enabled, false);
  assert.equal(voxelDebugPlan("raw-voxels", { voxelCapacity: 1, brickCapacity: 0 }).overlayWorkgroups, 0,
    "sources without brick records draw no residency outlines");
  assert.equal(SURFACE_VOXEL_MAXIMUM_INSTANCES_PER_RECORD, 8);
  assert.equal(surfaceVoxelInstanceCapacity(source.voxelCapacity), 129 * 8,
    "every source record reserves its complete 2x2x2 finest-shell expansion");
});

test("fluid brick residency outlines compact separately and color by state", () => {
  // Residency bits (CORE 2 | HALO 4 | ACTIVATED 8) only ever accompany fluid
  // solver leaves, so the overlay filter excludes environment records.
  assert.match(voxelDebugComputeShader, /const FLUID_RESIDENCY: u32 = 14u;/);
  assert.match(voxelDebugComputeShader, /fn compactFluidBricks/);
  assert.match(voxelDebugComputeShader, /materialAndFlags\.y & FLUID_RESIDENCY\) == 0u/);
  assert.match(voxelDebugComputeShader, /atomicAdd\(&overlayDrawArguments\.instanceCount, 1u\)/);
  assert.match(voxelDebugComputeShader, /compactSettings\.overlayCapacity/);
  // Both prepare entry points reset the overlay's 24-vertex line draw.
  assert.equal(voxelDebugComputeShader.split("overlayDrawArguments.vertexCount = 24u").length, 3);
  assert.match(voxelDebugRenderShader, /fn fluidResidencyColor/);
  assert.match(voxelDebugRenderShader, /fn overlayVertex/);
  assert.match(voxelDebugRenderShader, /fn overlayFragment/);
  // Outlines sit exactly on voxel faces; a camera-ward bias wins depth ties.
  assert.match(voxelDebugRenderShader, /output\.position\.z -= 0\.0015 \* output\.position\.w;/);
  // Grid mode routes resident fluid bricks through the shared palette while
  // the environment lattice keeps its alternating level tint.
  assert.match(voxelDebugRenderShader, /input\.flags & 14u/);
  assert.match(voxelDebugRenderShader, /input\.level & 1u/);
});

test("voxel debug ABI and shaders retain GPU material color and indirect instance production", () => {
  assert.equal(SPARSE_VOXEL_DEBUG_RECORD_STRIDE, 48);
  assert.equal(SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE, 32);
  assert.match(voxelDebugComputeShader, /atomicAdd\(&drawArguments\.instanceCount/);
  assert.match(voxelDebugComputeShader, /drawArguments\.vertexCount = 36u/);
  assert.match(voxelDebugComputeShader, /drawArguments\.vertexCount = 24u/);
  assert.match(voxelDebugComputeShader, /materialAndFlags\.y & ACTIVE/);
  assert.match(voxelDebugComputeShader, /compactSettings\.capacity/);
  assert.match(voxelDebugComputeShader, /fn compactSurfaceVoxels/);
  assert.match(voxelDebugComputeShader, /record\.extent\.xyz \/ compactSettings\.finestCell\.xyz/,
    "surface inspection derives finest-cell subdivisions from structural world spacing");
  assert.match(voxelDebugComputeShader, /fn claimSurfaceInstance[\s\S]*atomicCompareExchangeWeak/,
    "expanded surface voxels cannot publish an indirect count beyond their bounded instance arena");
  assert.match(voxelDebugComputeShader, /for \(var z = 1u; z \+ 1u < subdivisions\.z/,
    "coarse cells emit only their unique boundary shell rather than walking their interior volume");
  assert.match(voxelDebugRenderShader, /let material = materials\[/);
  assert.match(voxelDebugRenderShader, /material\.baseColor\.a <= 0\.001\) \{ discard; \}/);
  assert.match(voxelDebugRenderShader, /let separatedCorner = mix\(vec3f\(0\.035\), vec3f\(0\.965\), corner\);/,
    "raw cubes leave a visible gap instead of hiding the internal grid behind a continuous exterior shell");
  assert.match(voxelDebugRenderShader, /shadeUnifiedSurface\(closure, lighting\)/);
  const rawFragment = voxelDebugRenderShader.slice(
    voxelDebugRenderShader.indexOf("fn rawFragment"),
    voxelDebugRenderShader.indexOf("fn glassPaneFragment"),
  );
  assert.match(rawFragment, /let lambert = 0\.28 \+ 0\.72 \* max\(dot\(normal, normalize\(view\.lightDirection\.xyz\)\), 0\.0\);/,
    "raw occupancy keeps a bounded material-colored visibility floor instead of collapsing to NaN/black");
  assert.match(rawFragment, /let pixelWidth = max\(fwidth\(facePosition\), vec2f\(1e-4\)\);/,
    "raw occupancy keeps individual cell seams visible across projected cell sizes");
  assert.match(rawFragment, /linearColor \*= mix\(1\.0, 0\.24, edge\);/,
    "contiguous occupied cells remain structurally legible instead of reading as one smooth slab");
  assert.match(rawFragment, /input\.materialId == 3u && edge < 0\.35/);
  assert.match(rawFragment, /input\.voxelSeed \* 747796405u/,
    "fluid faces use a distinct per-cell screen-door pattern so deeper voxels remain visible without alpha sorting");
  assert.doesNotMatch(rawFragment, /shadeMaterial\(/, "raw occupancy never enters the degenerate PBR half-vector path");
  assert.match(voxelDebugRenderShader, /input\.level & 1u/);
  assert.match(voxelDebugRenderShader, /array<vec3f, 24>/);
  assert.doesNotMatch(voxelDebugComputeShader + voxelDebugRenderShader, /textureLoad|mapAsync|readBuffer/);
});

test("voxel debug rendering uses indirect draws and destroys only owned buffers once", async (t) => {
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousShaderStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, INDIRECT: 8 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  t.after(() => {
    for (const [name, descriptor] of [["GPUBufferUsage", previousBufferUsage], ["GPUShaderStage", previousShaderStage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });

  const destroyed: string[] = [];
  const writes: unknown[] = [];
  const pipelineDescriptors: GPURenderPipelineDescriptor[] = [];
  const renderDescriptors: GPURenderPassDescriptor[] = [];
  let indirectDraws = 0;
  const paneDraws: unknown[][] = [];
  const pipeline = {} as GPUComputePipeline & GPURenderPipeline;
  const device = {
    createShaderModule: () => ({}),
    createBindGroupLayout: () => ({}),
    createPipelineLayout: () => ({}),
    createComputePipelineAsync: async () => pipeline,
    createRenderPipelineAsync: async (descriptor: GPURenderPipelineDescriptor) => { pipelineDescriptors.push(descriptor); return pipeline; },
    createBuffer: ({ label }: GPUBufferDescriptor) => ({ destroy: () => destroyed.push(label ?? "unlabelled") }),
    createBindGroup: () => ({}),
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) }
  } as unknown as GPUDevice;
  const computePass = { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
  const renderPass = { setBindGroup() {}, setPipeline() {}, draw: (...args: unknown[]) => { paneDraws.push(args); }, drawIndirect: () => { indirectDraws += 1; }, end() {} };
  const encoder = {
    beginComputePass: () => computePass,
    beginRenderPass: (descriptor: GPURenderPassDescriptor) => { renderDescriptors.push(descriptor); return renderPass; }
  } as unknown as GPUCommandEncoder;
  let externalDestroyCount = 0;
  const external = { destroy: () => { externalDestroyCount += 1; } } as unknown as GPUBuffer;
  const binding = { buffer: external };
  const renderer = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
  await renderer.initialize();
  const rawPipeline = pipelineDescriptors.find(({ label }) => label === "Raw sparse voxel cubes");
  assert.equal(rawPipeline?.primitive?.cullMode, "none",
    "raw cubes remain two-sided because backend framebuffer orientation reverses their standalone clip-space winding");
  renderer.setSource({
    voxelRecords: binding, voxelCount: binding, brickRecords: binding, brickCount: binding, materials: binding,
    voxelCapacity: 80, brickCapacity: 20, materialCount: 2, revision: 1,
    structural: { domain: {
      worldOrigin_m: [0, 0, 0], cellSize_m: [0.25, 0.25, 0.25],
      dimensionsCells: [8, 8, 8], brickSize: 8, maximumDepth: 2,
    } } as unknown as SparseVoxelRenderSource["structural"],
  });
  const common = {
    colorTarget: {} as GPUTextureView, depthTarget: {} as GPUTextureView,
    viewProjection: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
    cameraPosition: [0, 0, 4] as const,
    containerBounds: { min: [-1, 0, -1] as const, max: [1, 2, 1] as const },
    containerClosedTop: false
  };
  assert.equal(renderer.encode(encoder, { ...common, mode: "raw-voxels", depthLoadOp: "clear", colorLoadOp: "clear" }), true);
  assert.equal(renderer.encode(encoder, { ...common, mode: "surface-voxels", depthLoadOp: "clear", colorLoadOp: "clear" }), true);
  assert.equal(renderer.encode(encoder, { ...common, mode: "brick-grid" }), true);
  assert.equal(indirectDraws, 6, "each mode draws its primary records plus the fluid residency outline overlay");
  assert.deepEqual(paneDraws, [
    [6, 1, 0, 3], [6, 1, 0, 0], [6, 1, 0, 1], [6, 1, 0, 2], [6, 1, 0, 4],
    [6, 1, 0, 3], [6, 1, 0, 0], [6, 1, 0, 1], [6, 1, 0, 2], [6, 1, 0, 4]
  ], "both voxel modes draw open tank panes back-to-front after opaque voxels");
  assert.equal(renderDescriptors[0].depthStencilAttachment?.depthClearValue, 1);
  const firstColorAttachment = Array.from(renderDescriptors[0].colorAttachments)[0];
  assert.equal(firstColorAttachment?.loadOp, "clear");
  assert.deepEqual(firstColorAttachment?.clearValue, { r: 0.008, g: 0.012, b: 0.018, a: 1 });
  assert.equal(writes.length, 6, "each voxel view uploads only view and declared capacity");

  renderer.setSource(undefined);
  assert.ok(destroyed.includes("Sparse voxel debug instances (80)"), "surface mode retires the raw-sized voxel arena");
  assert.ok(destroyed.includes("Sparse voxel debug instances (640)"), "detaching inspection releases the complete surface-expansion arena");
  assert.ok(destroyed.includes("Sparse voxel debug overlay instances (20)"), "detaching inspection releases the overlay arena");
  renderer.destroy();
  renderer.destroy();
  assert.equal(externalDestroyCount, 0, "source buffers remain owned by the sparse representation");
  assert.deepEqual(destroyed.sort(), [
    "Sparse voxel debug compaction settings",
    "Sparse voxel debug indirect draw",
    "Sparse voxel debug instances (640)",
    "Sparse voxel debug instances (80)",
    "Sparse voxel debug overlay indirect draw",
    "Sparse voxel debug overlay instances (20)",
    "Sparse voxel debug view"
  ]);
});

test("page-native raw inspection can omit filled tank panes", async (t) => {
  const previousBufferUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousShaderStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, INDIRECT: 8 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 } });
  t.after(() => {
    for (const [name, descriptor] of [["GPUBufferUsage", previousBufferUsage], ["GPUShaderStage", previousShaderStage]] as const) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else Reflect.deleteProperty(globalThis, name);
    }
  });
  const pipeline = {} as GPUComputePipeline & GPURenderPipeline;
  let directDraws = 0;
  const device = {
    createShaderModule: () => ({}), createBindGroupLayout: () => ({}), createPipelineLayout: () => ({}),
    createComputePipelineAsync: async () => pipeline, createRenderPipelineAsync: async () => pipeline,
    createBuffer: () => ({ destroy() {} }), createBindGroup: () => ({}), queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  const computePass = { setBindGroup() {}, setPipeline() {}, dispatchWorkgroups() {}, end() {} };
  const renderPass = { setBindGroup() {}, setPipeline() {}, draw: () => { directDraws += 1; }, drawIndirect() {}, end() {} };
  const encoder = { beginComputePass: () => computePass, beginRenderPass: () => renderPass } as unknown as GPUCommandEncoder;
  const external = {} as GPUBuffer;
  const binding = { buffer: external };
  const renderer = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
  await renderer.initialize();
  renderer.setSource({
    voxelRecords: binding, voxelCount: binding, brickRecords: binding, brickCount: binding, materials: binding,
    voxelCapacity: 1, brickCapacity: 1, materialCount: 4, revision: 1, drawContainerGlass: false,
  });
  assert.equal(renderer.encode(encoder, {
    mode: "raw-voxels", colorTarget: {} as GPUTextureView, depthTarget: {} as GPUTextureView,
    viewProjection: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]), cameraPosition: [0, 0, 4],
    containerBounds: { min: [-1, 0, -1], max: [1, 2, 1] }, containerClosedTop: false,
  }), true);
  assert.equal(directDraws, 0, "compact fluid inspection must not cover voxels with filled glass panes");
  renderer.destroy();
});

async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  const device = await adapter.requestDevice({
    requiredFeatures: optionalFluidDeviceFeatures(adapter.features),
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  return { device, validationErrors };
}

function packDebugRecord(origin: readonly number[], extent: readonly number[], material: number, flags: number, level: number) {
  const record = new ArrayBuffer(SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
  new Float32Array(record, 0, 8).set([...origin, 0, ...extent, 0]);
  new Uint32Array(record, 32, 4).set([material, flags, level, 0xffff]);
  return new Uint8Array(record);
}

test("a known active raw voxel produces a material-colored filled region", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU render checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const storage = (data: Uint8Array | Uint32Array) => {
      const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return buffer;
    };
    const voxelRecords = storage(packDebugRecord([-0.45, -0.45, 0.1], [0.9, 0.9, 0.4], 2, 1, 0));
    const voxelCount = storage(new Uint32Array([1]));
    const brickRecords = storage(new Uint8Array(SPARSE_VOXEL_DEBUG_RECORD_STRIDE));
    const brickCount = storage(new Uint32Array([0]));
    const packedMaterials = new ArrayBuffer(3 * SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE);
    // Material 1 remains transparent so the deliberately off-screen tank
    // cannot contribute pixels. Material 2 is the known occupied voxel.
    new Float32Array(packedMaterials, 2 * SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE, 8).set([
      0.08, 0.82, 0.24, 1,
      0.01, 0.08, 0.02, 0.5,
    ]);
    const materials = storage(new Uint8Array(packedMaterials));
    const renderer = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
    await renderer.initialize();
    renderer.setSource({
      voxelRecords: { buffer: voxelRecords }, voxelCount: { buffer: voxelCount },
      brickRecords: { buffer: brickRecords }, brickCount: { buffer: brickCount },
      materials: { buffer: materials }, voxelCapacity: 1, brickCapacity: 1, materialCount: 3, revision: 1
    });
    const size = 64;
    const color = device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [size, size], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const readback = device.createBuffer({ size: 256 * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    assert.equal(renderer.encode(encoder, {
      mode: "raw-voxels", colorTarget: color.createView(), depthTarget: depth.createView(),
      viewProjection: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]),
      cameraPosition: [0, 0, 4],
      containerBounds: { min: [10, 10, 10], max: [11, 11, 11] }, containerClosedTop: false,
      colorLoadOp: "clear", depthLoadOp: "clear"
    }), true);
    encoder.copyTextureToBuffer({ texture: color }, { buffer: readback, bytesPerRow: 256, rowsPerImage: size }, [size, size, 1]);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const pixels = new Uint8Array(readback.getMappedRange());
    let greenInteriorPixels = 0;
    for (let y = 20; y < 44; y += 1) for (let x = 20; x < 44; x += 1) {
      const base = y * 256 + x * 4;
      const [r, g, b] = pixels.subarray(base, base + 3);
      if (g > 35 && g > r * 2 && g > b * 2) greenInteriorPixels += 1;
    }
    assert.ok(greenInteriorPixels > 400,
      `expected a filled material-colored cube interior, received ${greenInteriorPixels} green pixels`);
    readback.unmap();
    assert.deepEqual(validationErrors, []);
    renderer.destroy();
    for (const resource of [voxelRecords, voxelCount, brickRecords, brickCount, materials, readback]) resource.destroy();
    color.destroy(); depth.destroy();
  } finally {
    device.destroy();
  }
});

test("resident fluid bricks render outlines while environment bricks stay outline-free in raw mode", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU render checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  try {
    const storage = (data: Uint8Array | Uint32Array) => {
      const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(buffer, 0, data.buffer as ArrayBuffer, data.byteOffset, data.byteLength);
      return buffer;
    };
    // Three same-scale brick records: a CORE fluid brick on the left, a
    // freshly ACTIVATED halo brick on the right, and an environment brick
    // above the pair that must not receive an outline.
    const records = new Uint8Array(3 * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    records.set(packDebugRecord([-0.8, -0.25, 0], [0.5, 0.5, 0.5], 2, 1 | 2, 3), 0);
    records.set(packDebugRecord([0.3, -0.25, 0], [0.5, 0.5, 0.5], 2, 1 | 4 | 8, 3), SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    records.set(packDebugRecord([-0.25, 0.35, 0], [0.5, 0.5, 0.5], 33, 1, 3), 2 * SPARSE_VOXEL_DEBUG_RECORD_STRIDE);
    const brickRecords = storage(records);
    const brickCount = storage(new Uint32Array([3]));
    const voxelRecords = storage(new Uint8Array(SPARSE_VOXEL_DEBUG_RECORD_STRIDE));
    const voxelCount = storage(new Uint32Array([0]));
    // Zero-alpha materials keep raw cubes and glass panes invisible so the
    // readback isolates the residency outline overlay.
    const materials = storage(new Uint8Array(2 * SPARSE_VOXEL_DEBUG_MATERIAL_STRIDE));
    const renderer = new SparseVoxelDebugRenderer(device, { colorFormat: "rgba8unorm" });
    await renderer.initialize();
    renderer.setSource({
      voxelRecords: { buffer: voxelRecords }, voxelCount: { buffer: voxelCount },
      brickRecords: { buffer: brickRecords }, brickCount: { buffer: brickCount },
      materials: { buffer: materials }, voxelCapacity: 1, brickCapacity: 3, materialCount: 2, revision: 1
    });
    const size = 64;
    const color = device.createTexture({ size: [size, size], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const depth = device.createTexture({ size: [size, size], format: "depth24plus", usage: GPUTextureUsage.RENDER_ATTACHMENT });
    const readback = device.createBuffer({ size: 256 * size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const renderAndRead = async (mode: "raw-voxels" | "brick-grid") => {
      const encoder = device.createCommandEncoder();
      assert.equal(renderer.encode(encoder, {
        mode, colorTarget: color.createView(), depthTarget: depth.createView(),
        viewProjection: new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.1, 0, 0, 0, 0.5, 1]),
        cameraPosition: [0, 0, 4],
        containerBounds: { min: [-30, -30, -30], max: [-29, -29, -29] }, containerClosedTop: false,
        colorLoadOp: "clear", depthLoadOp: "clear"
      }), true);
      encoder.copyTextureToBuffer({ texture: color }, { buffer: readback, bytesPerRow: 256, rowsPerImage: size }, [size, size, 1]);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const pixels = new Uint8Array(readback.getMappedRange().slice(0));
      readback.unmap();
      return (region: { x0: number; x1: number; y0: number; y1: number }, classify: (r: number, g: number, b: number) => boolean) => {
        let hits = 0;
        for (let y = 0; y < size; y += 1) for (let x = 0; x < size; x += 1) {
          const ndcX = (x + 0.5) / size * 2 - 1, ndcY = 1 - (y + 0.5) / size * 2;
          if (ndcX < region.x0 || ndcX > region.x1 || ndcY < region.y0 || ndcY > region.y1) continue;
          const base = y * 256 + x * 4;
          if (classify(pixels[base], pixels[base + 1], pixels[base + 2])) hits += 1;
        }
        return hits;
      };
    };
    const isCoreBlue = (r: number, g: number, b: number) => b > 180 && r < 80 && b > g;
    const isActivatedGreen = (r: number, g: number, b: number) => g > 180 && g > b && g > r;
    const isAnyLine = (r: number, g: number, b: number) => r > 60 || g > 60 || b > 60;
    const raw = await renderAndRead("raw-voxels");
    const coreRegion = { x0: -0.85, x1: -0.25, y0: -0.3, y1: 0.3 };
    const activatedRegion = { x0: 0.25, x1: 0.85, y0: -0.3, y1: 0.3 };
    const environmentRegion = { x0: -0.3, x1: 0.3, y0: 0.32, y1: 0.88 };
    assert.ok(raw(coreRegion, isCoreBlue) > 8, "the CORE brick draws a bright blue outline in raw mode");
    assert.ok(raw(activatedRegion, isActivatedGreen) > 8, "the freshly ACTIVATED brick flashes green in raw mode");
    assert.equal(raw(environmentRegion, isAnyLine), 0, "environment bricks draw no raw-mode outline");
    const grid = await renderAndRead("brick-grid");
    assert.ok(grid(coreRegion, isCoreBlue) > 8, "grid mode keeps core-blue for CORE bricks");
    assert.ok(grid(environmentRegion, isAnyLine) > 8, "grid mode still draws the environment lattice");
    assert.deepEqual(validationErrors, []);
    renderer.destroy();
    for (const resource of [brickRecords, brickCount, voxelRecords, voxelCount, materials, readback]) resource.destroy();
    color.destroy(); depth.destroy();
  } finally {
    device.destroy();
  }
});
