import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  GPUOctreeOwnerPageArena,
  OCTREE_OWNER_PAGE_LOOKUP_STATUS,
  OCTREE_OWNER_PAGE_WORD_VALID,
  SVO_OWNER_PAGE_CONTROL_WORDS,
  SVO_OWNER_PAGE_STATUS,
  WebGPUSvoOwnerPageAllocator,
  WebGPUOctreeSimulationOwnerPages,
  OctreeOwnerPageLifecycleMirror,
  canonicalMissingAirOwner,
  decodeOctreeOwnerPageWord,
  lookupOctreeOwnerPage,
  octreeOwnerPageLookupWgsl,
  packOctreeOwnerPageWord,
  planOctreeOwnerPages,
  svoRendererOwnerPageAllocatorShader,
  unpackOctreeOwnerPageWord,
  type OctreeOwnerLeafSize,
} from "../lib/webgpu-octree-owner-pages";
import { svoRenderResidencyConsumerLayout, type WebGPUSvoRenderResidencyConsumer } from "../lib/webgpu-svo-render-residency-consumer";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const modulePath = process.env.WEBGPU_NODE_MODULE;
const projectionSource = readFileSync(new URL("../lib/webgpu-octree.ts", import.meta.url), "utf8");

test("octree owner page planning halves the full-capacity owner payload and obeys byte ceilings", () => {
  const ocean = planOctreeOwnerPages([288, 96, 64]);
  assert.deepEqual(ocean.brickDimensions, [36, 12, 8]);
  assert.equal(ocean.logicalBrickCount, 3456);
  assert.equal(ocean.capacity, 3456);
  assert.equal(ocean.bytesPerPage, 2052);
  assert.equal(ocean.pageHashCapacity, 4_608);
  assert.equal(ocean.allocatedBytes, 7_128_640);
  assert.equal(ocean.denseOwnerBytes, 14_155_776);
  assert.ok(ocean.allocatedBytes < ocean.denseOwnerBytes * 0.51);

  const quarter = planOctreeOwnerPages([64, 64, 64], { maximumResidentFraction: 0.25 });
  assert.equal(quarter.logicalBrickCount, 512);
  assert.equal(quarter.capacity, 128);
  const twoPageBytes = (16 + 3 * 2 + 2 * (1 + 512)) * 4;
  const bounded = planOctreeOwnerPages([64, 64, 64], { maximumArenaBytes: twoPageBytes });
  assert.equal(bounded.capacity, 2);
  assert.equal(bounded.degraded, true);
  assert.equal(bounded.allocatedBytes, twoPageBytes);
  assert.throws(() => planOctreeOwnerPages([64, 64, 64], { maximumArenaBytes: 100 }), /cannot hold one physical page/);
});

test("compact simulation owner capacity follows pressure and surface bounds instead of box volume", () => {
  const ocean = planOctreeOwnerPages([320, 96, 80], {
    adaptiveBounds: { pressureRowCapacity: 384_768, surfacePageCapacity: 123_126 },
  });
  assert.equal(ocean.logicalBrickCount, 4_800);
  assert.equal(ocean.adaptiveCapacity, 4_014);
  assert.equal(ocean.capacity, 4_014);
  assert.equal(ocean.pageHashCapacity, 5_352);
  assert.equal(ocean.allocatedBytes, 8_279_608);

  const target = planOctreeOwnerPages([640, 192, 160], {
    adaptiveBounds: { pressureRowCapacity: 1_540_864, surfacePageCapacity: 493_077 },
  });
  const oldNinetyPercent = planOctreeOwnerPages([640, 192, 160], { maximumResidentFraction: 0.90 });
  assert.equal(target.adaptiveCapacity, 16_073);
  assert.equal(target.pageHashCapacity, 21_431);
  assert.equal(target.allocatedBytes, 33_153_308);
  assert.equal(oldNinetyPercent.allocatedBytes - target.allocatedBytes, 38_132_516);
  assert.ok(target.capacity < target.logicalBrickCount / 2);
  assert.throws(() => planOctreeOwnerPages([64, 64, 64], {
    adaptiveBounds: { pressureRowCapacity: 0, surfacePageCapacity: 1 },
  }), /pressure-row capacity/);
});

test("projection derives compact owner capacity and keeps overflow on the canonical fallback path", () => {
  assert.match(projectionSource, /adaptiveBounds:[\s\S]*pressureRowCapacity: this\.pressureCapacity\.rowCapacity[\s\S]*surfacePageCapacity: Math\.max/);
  assert.match(projectionSource,
    /surfacePageCapacity: Math\.max\(1,[\s\S]*adaptiveSurfacePageFraction[\s\S]*maximumResidentFraction: adaptiveSurfacePageFraction/,
    "owner pages and surface pages must derive capacity from the same bounded fraction");
  assert.match(projectionSource, /ownerPagesEnabled\(\) && atomicLoad\(&owners\[2\]\) != 0u/,
    "owner-page exhaustion must fail the topology pass closed");
  assert.match(projectionSource, /return Owner\(packOrigin\(origin\), size\)/,
    "missing physical pages retain the deterministic coarse-owner lookup");
});

test("packed owner words round-trip every leaf size and max-32 owners across brick seams", () => {
  const sizes: OctreeOwnerLeafSize[] = [1, 2, 4, 8, 16, 32];
  for (const size of sizes) {
    const origin = [32, 32, 32] as const;
    const cell = [32 + size - 1, 32 + Math.floor(size / 2), 32] as const;
    const packed = packOctreeOwnerPageWord(cell, origin, size);
    assert.ok((packed & OCTREE_OWNER_PAGE_WORD_VALID) !== 0);
    assert.deepEqual(unpackOctreeOwnerPageWord(packed, cell), { origin: [...origin], size, missing: false });
  }
  const origin = [32, 64, 96] as const;
  for (const cell of [[32, 64, 96], [39, 71, 103], [40, 72, 104], [55, 87, 119], [63, 95, 127]] as const) {
    const packed = packOctreeOwnerPageWord(cell, origin, 32);
    assert.deepEqual(unpackOctreeOwnerPageWord(packed, cell), { origin: [...origin], size: 32, missing: false });
  }
  assert.throws(() => packOctreeOwnerPageWord([64, 0, 0], [0, 0, 0], 32), /inside its leaf/);
  assert.throws(() => unpackOctreeOwnerPageWord(0, [0, 0, 0]), /missing/);
});

test("missing pages decode to a deterministic domain-fitting coarse air owner", () => {
  assert.deepEqual(canonicalMissingAirOwner([1, 1, 1], [19, 10, 9], 32), {
    origin: [0, 0, 0], size: 8, missing: true,
  });
  assert.deepEqual(canonicalMissingAirOwner([18, 9, 8], [19, 10, 9], 32), {
    origin: [18, 9, 8], size: 1, missing: true,
  });
  assert.deepEqual(decodeOctreeOwnerPageWord(0, [1, 1, 1], [19, 10, 9], 32), {
    origin: [0, 0, 0], size: 8, missing: true,
  });
  const packed = packOctreeOwnerPageWord([9, 9, 9], [8, 8, 8], 4);
  assert.equal(decodeOctreeOwnerPageWord(packed, [9, 9, 9], [19, 10, 9], 32).missing, false);
});

function ownerLookupFixture() {
  const plan = planOctreeOwnerPages([64, 32, 32], { maximumPages: 6 });
  // Deliberately expose only four payload pages. An encoded page within the
  // declared capacity can therefore still be rejected by the arena-length
  // bound, matching a truncated/degraded GPU allocation.
  const words = plan.ownerPagesOffsetWords + 4 * plan.pageVoxels;
  const arena = new Uint32Array(words);
  const logical = (x: number) => Math.floor(x / 8);
  const insert = (logicalBrick: number, encodedPage: number) => {
    let slot = (Math.imul(logicalBrick, 0x9e37_79b1) >>> 0) % plan.pageHashCapacity;
    while (arena[plan.pageTableOffsetWords + slot] !== 0) slot = (slot + 1) % plan.pageHashCapacity;
    arena[plan.pageTableOffsetWords + slot] = logicalBrick + 1;
    arena[plan.pageTableValueOffsetWords + slot] = encodedPage;
  };
  const payload = (slot: number, cell: readonly [number, number, number]) =>
    plan.ownerPagesOffsetWords + slot * plan.pageVoxels
      + (cell[0] & 7) + (cell[1] & 7) * 8 + (cell[2] & 7) * 64;
  const left = [7, 7, 7] as const;
  const right = [8, 7, 7] as const;
  insert(logical(left[0]), 1);
  insert(logical(right[0]), 2);
  arena[payload(0, left)] = packOctreeOwnerPageWord(left, [0, 0, 0], 32);
  arena[payload(1, right)] = packOctreeOwnerPageWord(right, [0, 0, 0], 32);
  insert(logical(24), 0xffff_ffff);
  insert(logical(32), plan.capacity + 1);
  insert(logical(40), 3);
  arena[payload(2, [40, 7, 7])] = (OCTREE_OWNER_PAGE_WORD_VALID | (6 << 18)) >>> 0;
  insert(logical(48), 4); // zero payload is ordinary missing air
  insert(logical(56), 6); // declared slot exists, backing words do not
  const cells = [left, right, [16, 7, 7], [24, 7, 7], [32, 7, 7], [40, 7, 7],
    [48, 7, 7], [56, 7, 7], [-1, 7, 7], [64, 7, 7]] as const;
  return { plan, arena, cells };
}

test("bounded owner-page lookup crosses seams and fails closed for missing, malformed, truncated, and invalid inputs", () => {
  const { plan, arena, cells } = ownerLookupFixture();
  const results = cells.map((cell) => lookupOctreeOwnerPage(arena, plan, cell, 32));
  assert.deepEqual(results[0], { origin: [0, 0, 0], size: 32, missing: false, status: 0 });
  assert.deepEqual(results[1], results[0], "brick-relative words on either side of a seam resolve the same leaf32 owner");
  assert.equal(results[2].status, OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing);
  assert.equal(results[6].status, OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing, "a zero-initialized resident word is canonical air");
  for (const index of [3, 4, 5, 7, 8, 9]) {
    assert.equal(results[index].status,
      OCTREE_OWNER_PAGE_LOOKUP_STATUS.missing | OCTREE_OWNER_PAGE_LOOKUP_STATUS.invalid);
    assert.equal(results[index].missing, true);
  }
  assert.deepEqual(results[8].origin, [0, 0, 0], "negative coordinates clamp before canonical-air tiling");
  assert.deepEqual(results[9].origin, [32, 0, 0], "upper out-of-domain coordinates clamp deterministically");
});

test("owner-page WGSL lookup is a binding-neutral, bounded read-only helper", () => {
  assert.match(octreeOwnerPageLookupWgsl, /arrayLength\(&ownerPageArena\)/);
  assert.match(octreeOwnerPageLookupWgsl, /logical \* 0x9e3779b1u/);
  assert.match(octreeOwnerPageLookupWgsl, /probe < hashCapacity/);
  assert.match(octreeOwnerPageLookupWgsl, /encodedPage > capacity/);
  assert.match(octreeOwnerPageLookupWgsl, /pageVoxels != OWNER_PAGE_VOXELS/);
  assert.match(octreeOwnerPageLookupWgsl, /exponent > 5u/);
  assert.match(octreeOwnerPageLookupWgsl, /return ownerPageCanonicalAir\(cell, 0u\)/);
  assert.doesNotMatch(octreeOwnerPageLookupWgsl, /@group|@binding|read_write|atomic/,
    "the consumer chooses bindings and the helper cannot mutate owner residency");
});

test("CPU owner lifecycle allocates before retirement, then reuses slots and reports overflow", () => {
  const lifecycle = new OctreeOwnerPageLifecycleMirror(4, 2);
  let stats = lifecycle.update([0, 1], []);
  assert.deepEqual({ resident: stats.resident, free: stats.free, activated: stats.activated, overflow: stats.overflow }, {
    resident: 2, free: 0, activated: 2, overflow: 0,
  });
  assert.equal(lifecycle.slot(0), 0);
  assert.equal(lifecycle.slot(1), 1);

  stats = lifecycle.update([2], [0]);
  assert.equal(stats.overflow, 1, "retiring storage is deliberately unavailable to activation in the same frame");
  assert.equal(stats.retired, 1);
  assert.equal(lifecycle.slot(0), undefined);
  assert.equal(lifecycle.slot(2), undefined);

  stats = lifecycle.update([2], []);
  assert.equal(stats.overflow, 0);
  assert.equal(lifecycle.slot(2), 0, "the next publication can reuse the retired physical slot");
  assert.equal(stats.peakResident, 2);
  assert.equal(stats.generation, 3);
  assert.throws(() => lifecycle.update([3], [3]), /active and retired/);
});

test("renderer owner publication fence rejects zero, stale, and unchanged generations without lifecycle work", () => {
  const lifecycle = new OctreeOwnerPageLifecycleMirror(4, 2);
  assert.equal(lifecycle.publish(0, [0], []).status, SVO_OWNER_PAGE_STATUS.unpublished);
  const first = lifecycle.publish(5, [0, 1], []);
  assert.equal(first.status, SVO_OWNER_PAGE_STATUS.ready);
  assert.equal(first.stats.generation, 5);
  assert.equal(lifecycle.slot(0), 0);
  assert.equal(lifecycle.slot(1), 1);
  assert.equal(lifecycle.publish(5, [2], [0]).status, SVO_OWNER_PAGE_STATUS.unchanged);
  assert.equal(lifecycle.publish(4, [2], [0]).status, SVO_OWNER_PAGE_STATUS.stale);
  assert.equal(lifecycle.slot(0), 0, "rejected publications cannot mutate residency");
  const next = lifecycle.publish(6, [2], [0]);
  assert.ok((next.status & SVO_OWNER_PAGE_STATUS.overflow) !== 0);
  assert.equal(next.stats.retired, 1, "retirement still completes after bounded activation overflow");
  assert.equal(lifecycle.slot(2), undefined, "activation cannot consume a slot retired by the same publication");
});

test("renderer owner allocator shader consumes fenced compact lists in deterministic order without readback or fine-payload claims", () => {
  assert.match(svoRendererOwnerPageAllocatorShader, /sourceControl\[2u\]/);
  assert.match(svoRendererOwnerPageAllocatorShader, /sourceControl\[7u\]/);
  assert.match(svoRendererOwnerPageAllocatorShader, /sourceControl\[10u\]/);
  assert.match(svoRendererOwnerPageAllocatorShader, /for \(var item = 0u; item < activeCount; item \+= 1u\)[\s\S]*for \(var item = 0u; item < retiredCount/);
  assert.match(svoRendererOwnerPageAllocatorShader, /atomicStore\(&arena\[7\], sourceGeneration\)/);
  assert.match(svoRendererOwnerPageAllocatorShader, /for \(var local = 0u; local < PAGE_VOXELS/);
  assert.doesNotMatch(svoRendererOwnerPageAllocatorShader, /mapAsync|copyBufferToBuffer|denseOwners|finePhi/);
  assert.equal(SVO_OWNER_PAGE_CONTROL_WORDS.acceptedGeneration, 7);
  assert.equal(SVO_OWNER_PAGE_CONTROL_WORDS.status, 10);
});

test("renderer owner allocator owns its arena and encodes one GPU-only fenced publication", (t) => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { COMPUTE: 1 } });
  t.after(() => {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage); else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage); else Reflect.deleteProperty(globalThis, "GPUShaderStage");
  });
  const sourceLayout = svoRenderResidencyConsumerLayout(4);
  const source = {
    layout: sourceLayout,
    control: { size: sourceLayout.controlByteLength },
    entries: { size: sourceLayout.entryByteLength },
  } as unknown as WebGPUSvoRenderResidencyConsumer;
  const created: Array<{ label?: string; size: number; destroyed: boolean }> = [];
  const writes: unknown[][] = [];
  const bound: GPUBindGroupEntry[][] = [];
  const device = {
    createBuffer: (descriptor: GPUBufferDescriptor) => {
      const buffer = { ...descriptor, destroyed: false, destroy() { buffer.destroyed = true; } };
      created.push(buffer); return buffer;
    },
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) },
    createBindGroupLayout: () => ({}),
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => ({ code, getCompilationInfo: async () => ({ messages: [] }) }),
    createPipelineLayout: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({ entryPoint: compute.entryPoint }),
    createBindGroup: ({ entries }: GPUBindGroupDescriptor) => { bound.push([...entries]); return {}; },
  } as unknown as GPUDevice;
  const allocator = new WebGPUSvoOwnerPageAllocator(device, [32, 8, 8], source, { maximumPages: 2 });
  assert.equal(writes.length, 2, "only immutable arena initialization and parameters are uploaded");
  assert.equal((bound[0][1].resource as GPUBufferBinding).buffer, source.control);
  assert.equal((bound[0][2].resource as GPUBufferBinding).buffer, source.entries);
  assert.equal(allocator.telemetryBinding().size, 64);
  assert.equal(allocator.storageBinding().offset, 0, "storage consumers bind the aligned arena and use plan word offsets");
  assert.equal(allocator.storageBinding().size, allocator.plan.allocatedBytes);
  const calls: unknown[][] = [];
  const encoder = {
    beginComputePass: ({ label }: GPUComputePassDescriptor) => ({
      setPipeline: () => {}, setBindGroup: () => {},
      dispatchWorkgroups: (...args: unknown[]) => calls.push([label, ...args]),
      end: () => calls.push([label, "end"]),
    }),
  } as unknown as GPUCommandEncoder;
  allocator.encode(encoder);
  assert.deepEqual(calls, [["Apply SVO renderer owner-page residency", 1], ["Apply SVO renderer owner-page residency", "end"]]);
  assert.equal("readStats" in allocator, false, "production allocator exposes GPU bindings, not a CPU map path");
  allocator.destroy(); allocator.destroy();
  assert.equal(created.filter((buffer) => buffer.destroyed).length, 3);
});

async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
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

test("WGSL owner-page lookup matches the CPU oracle without upload-queue staging", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU owner-page checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  const fixture = ownerLookupFixture();
  const resultWords = fixture.cells.length * 8;
  const arena = device.createBuffer({
    label: "Owner lookup bounded arena",
    size: fixture.arena.byteLength,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Uint32Array(arena.getMappedRange()).set(fixture.arena);
  arena.unmap();
  const params = device.createBuffer({
    label: "Owner lookup parameters",
    size: 48,
    usage: GPUBufferUsage.UNIFORM,
    mappedAtCreation: true,
  });
  new Uint32Array(params.getMappedRange()).set([
    ...fixture.plan.dimensions, 32,
    ...fixture.plan.brickDimensions, fixture.plan.logicalBrickCount,
    fixture.plan.pageTableOffsetWords, fixture.plan.ownerPagesOffsetWords,
    fixture.plan.capacity, fixture.plan.pageVoxels,
  ]);
  params.unmap();
  const inputs = device.createBuffer({
    label: "Owner lookup cells",
    size: fixture.cells.length * 16,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  const inputWords = new Int32Array(inputs.getMappedRange());
  fixture.cells.forEach((cell, index) => inputWords.set([...cell, 0], index * 4));
  inputs.unmap();
  const output = device.createBuffer({
    label: "Owner lookup results",
    size: resultWords * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readback = device.createBuffer({
    label: "Owner lookup result readback",
    size: resultWords * 4,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const shader = `${octreeOwnerPageLookupWgsl}
@group(0) @binding(0) var<storage, read> ownerPageArena: array<u32>;
@group(0) @binding(1) var<uniform> ownerPageLookupParams: OctreeOwnerPageLookupParams;
@group(0) @binding(2) var<storage, read> lookupCells: array<vec4i>;
@group(0) @binding(3) var<storage, read_write> lookupResults: array<vec4u>;
@compute @workgroup_size(1)
fn testLookup(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= arrayLength(&lookupCells)) { return; }
  let owner = octreeOwnerPageLookup(lookupCells[gid.x].xyz);
  lookupResults[gid.x * 2u] = vec4u(owner.origin, owner.size);
  lookupResults[gid.x * 2u + 1u] = vec4u(owner.status, 0u, 0u, 0u);
}`;
  const shaderModule = device.createShaderModule({ label: "Owner page lookup parity", code: shader });
  try {
    const compilation = await shaderModule.getCompilationInfo();
    assert.deepEqual(compilation.messages.filter(({ type }) => type === "error"), []);
    const layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    ] });
    const pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
      compute: { module: shaderModule, entryPoint: "testLookup" },
    });
    const bindGroup = device.createBindGroup({ layout, entries: [
      { binding: 0, resource: { buffer: arena } },
      { binding: 1, resource: { buffer: params } },
      { binding: 2, resource: { buffer: inputs } },
      { binding: 3, resource: { buffer: output } },
    ] });
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline); pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(fixture.cells.length); pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, resultWords * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const actual = new Uint32Array(readback.getMappedRange());
    const expected = fixture.cells.flatMap((cell) => {
      const owner = lookupOctreeOwnerPage(fixture.arena, fixture.plan, cell, 32);
      return [...owner.origin, owner.size, owner.status, 0, 0, 0];
    });
    assert.deepEqual(Array.from(actual), expected);
    readback.unmap();
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(validationErrors, []);
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    arena.destroy(); params.destroy(); inputs.destroy(); output.destroy(); readback.destroy(); device.destroy();
  }
});

test("GPU owner arena mirrors allocate-retire-overflow-reuse lifecycle", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU owner-page checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  const arena = new GPUOctreeOwnerPageArena(device, [32, 8, 8], { maximumPages: 2 });
  const publish = async (active: readonly number[], retired: readonly number[]) => {
    const encoder = device.createCommandEncoder();
    arena.encodeLifecycle(encoder, active, retired);
    device.queue.submit([encoder.finish()]);
    return arena.readState();
  };
  try {
    let state = await publish([0, 1], []);
    assert.deepEqual(Array.from(state.pageTable), [1, 2, 0, 0]);
    assert.deepEqual({ resident: state.stats.resident, free: state.stats.free, overflow: state.stats.overflow }, { resident: 2, free: 0, overflow: 0 });

    state = await publish([2], [0]);
    assert.deepEqual(Array.from(state.pageTable), [0, 2, 0, 0]);
    assert.equal(state.stats.overflow, 1);
    assert.equal(state.stats.retired, 1);

    state = await publish([2], []);
    assert.deepEqual(Array.from(state.pageTable), [0, 2, 1, 0]);
    assert.equal(state.stats.overflow, 0);
    assert.equal(state.stats.generation, 3);
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(validationErrors, []);
  } finally {
    arena.destroy();
    device.destroy();
  }
});

test("simulation owner pages consume GPU brick residency and reuse retired slots in-frame", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU owner-page checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  const worklist = device.createBuffer({ size: 32 * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT });
  const pages = new WebGPUOctreeSimulationOwnerPages(device, [32, 8, 8], worklist, { maximumPages: 2 });
  const readback = device.createBuffer({ size: 20 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const publish = async (active: readonly number[], retired: readonly number[]) => {
    const words = new Uint32Array(32); words[0] = active.length; words[4] = retired.length;
    words.set([active.length * 2, 1, 1], 1);
    words.set([retired.length * 2, 1, 1], 5);
    active.forEach((logical, slot) => { words[16 + slot * 2] = logical; });
    retired.forEach((logical, slot) => { words[16 + 8 + slot * 2] = logical; });
    device.queue.writeBuffer(worklist, 0, words);
    const encoder = device.createCommandEncoder(); pages.encode(encoder);
    encoder.copyBufferToBuffer(pages.arena, 0, readback, 0, 20 * 4); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const result = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap(); return result;
  };
  try {
    let state = await publish([0, 1], []);
    assert.deepEqual([...state.slice(16, 18)].sort(), [1, 2]);
    const retained = state[17];
    state = await publish([2], [0]);
    assert.equal(state[16], 0); assert.equal(state[17], retained); assert.ok(state[18] > 0 && state[18] !== retained); assert.equal(state[19], 0);
    assert.equal(state[2], 0); assert.equal(state[1], 2);
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(validationErrors, []);
  } finally { pages.destroy(); worklist.destroy(); readback.destroy(); device.destroy(); }
});

test("GPU renderer owner allocator fences generations and reuses slots deterministically without allocator readback", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU owner-page checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  const layout = svoRenderResidencyConsumerLayout(4);
  const control = device.createBuffer({ label: "Owner allocator test source control", size: layout.controlByteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const entries = device.createBuffer({ label: "Owner allocator test source entries", size: layout.entryByteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const source = { layout, control, entries } as unknown as WebGPUSvoRenderResidencyConsumer;
  const allocator = new WebGPUSvoOwnerPageAllocator(device, [32, 8, 8], source, { maximumPages: 2 });
  const readbackWords = 16 + allocator.plan.logicalBrickCount;
  const readback = device.createBuffer({ label: "Test-owned owner allocator readback", size: readbackWords * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const publish = async (generation: number, active: readonly number[], retired: readonly number[]) => {
    const controls = new Uint32Array(layout.controlByteLength / 4);
    controls[0] = 1; controls[1] = generation; controls[2] = generation;
    controls[7] = active.length; controls[10] = retired.length;
    const lists = new Uint32Array(layout.entryByteLength / 4);
    active.forEach((logical, index) => { lists[layout.entryOffsetsBytes.active / 4 + index * 2] = logical; });
    retired.forEach((logical, index) => { lists[layout.entryOffsetsBytes.retired / 4 + index * 2] = logical; });
    device.queue.writeBuffer(control, 0, controls);
    device.queue.writeBuffer(entries, 0, lists);
    const encoder = device.createCommandEncoder();
    allocator.encode(encoder);
    encoder.copyBufferToBuffer(allocator.arena, 0, readback, 0, readbackWords * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const state = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    return state;
  };
  try {
    let state = await publish(5, [0, 1], []);
    assert.deepEqual(Array.from(state.slice(16)), [1, 2, 0, 0]);
    state = await publish(5, [2], []);
    assert.equal(state[10], SVO_OWNER_PAGE_STATUS.unchanged);
    assert.deepEqual(Array.from(state.slice(16)), [1, 2, 0, 0]);
    state = await publish(6, [2], [0]);
    assert.ok((state[10] & SVO_OWNER_PAGE_STATUS.overflow) !== 0);
    assert.deepEqual(Array.from(state.slice(16)), [0, 2, 0, 0]);
    state = await publish(7, [2], []);
    assert.deepEqual(Array.from(state.slice(16)), [0, 2, 1, 0]);
    assert.equal(state[7], 7);
    await device.queue.onSubmittedWorkDone();
    assert.deepEqual(validationErrors, []);
  } finally {
    allocator.destroy(); readback.destroy(); control.destroy(); entries.destroy(); device.destroy();
  }
});
