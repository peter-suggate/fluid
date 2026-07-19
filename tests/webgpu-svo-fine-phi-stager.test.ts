import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { planOctreeOwnerPages, SVO_OWNER_PAGE_STATUS, type WebGPUSvoOwnerPageAllocator } from "../lib/webgpu-octree-owner-pages";
import {
  SVO_FINE_PHI_CONTROL_WORDS,
  SVO_FINE_PHI_STATUS,
  SvoFinePhiPublicationMirror,
  WebGPUSvoFinePhiStager,
  gradientFinePhiReference,
  planSvoFinePhiStaging,
  sampleFinePhiReference,
  svoFinePhiSamplingWGSL,
  svoFinePhiStagingShader,
} from "../lib/webgpu-svo-fine-phi-stager";
import { svoRenderResidencyConsumerLayout, type WebGPUSvoRenderResidencyConsumer } from "../lib/webgpu-svo-render-residency-consumer";
import type { SparseSurfaceBandGPUSource } from "../lib/webgpu-sparse-surface-band";
import { optionalFluidDeviceFeatures, requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";

const fakeBuffer = (size: number, label = "buffer") => ({ size, label, destroy() {} } as unknown as GPUBuffer);
const fakeSource = (overrides: Partial<SparseSurfaceBandGPUSource> = {}): SparseSurfaceBandGPUSource => ({
  mode: "authoritative",
  pageTable: { buffer: fakeBuffer(64) }, states: { buffer: fakeBuffer(64) }, activePages: { buffer: fakeBuffer(80) },
  phi: { buffer: fakeBuffer(16 * 512 * 4) }, velocity: { buffer: fakeBuffer(4) }, params: { buffer: fakeBuffer(112) }, control: { buffer: fakeBuffer(64) },
  coarseLevelSet: {} as GPUTexture, coarseVelocity: {} as GPUTexture,
  fineDimensions: [32, 16, 16], brickDimensions: [4, 2, 2], brickSize: 8, refinementFactor: 2,
  pageCapacity: 16, revision: 3, ...overrides,
});

test("fine-phi staging ABI allocates aligned apron tiles and degrades under a byte ceiling", () => {
  const owner = planOctreeOwnerPages([16, 8, 8]);
  const source = fakeSource();
  const full = planSvoFinePhiStaging(owner, source, [0.2, 0.4, 0.6]);
  assert.equal(full.tileEdge, 18);
  assert.equal(full.tileVoxels, 18 ** 3);
  assert.equal(full.pageGenerationOffsetWords, 64);
  assert.equal(full.payloadOffsetWords, 128);
  assert.equal(full.capacity, 2);
  assert.deepEqual(full.fineCellSize_m, [0.1, 0.2, 0.3]);
  const oneTileBytes = (full.payloadOffsetWords + full.tileVoxels) * 4;
  const bounded = planSvoFinePhiStaging(owner, source, [0.2, 0.4, 0.6], { maximumArenaBytes: oneTileBytes });
  assert.equal(bounded.capacity, 1);
  assert.equal(bounded.degraded, true);
  assert.throws(() => planSvoFinePhiStaging(owner, source, [0.2, 0.4, 0.6], { maximumArenaBytes: 128 }), /cannot hold one apron tile/);
  assert.throws(() => planSvoFinePhiStaging(owner, fakeSource({ fineDimensions: [31, 16, 16] }), [0.2, 0.4, 0.6]), /not refinement-aligned/);
});

test("fine-phi staging embeds a solver-local source at a nonzero structural origin", () => {
  const owner = planOctreeOwnerPages([48, 32, 24], { maximumPages: 3 });
  const plan = planSvoFinePhiStaging(owner, fakeSource(), [0.2, 0.4, 0.6], {
    sourceOriginBricks: [2, 1, 1],
  });
  assert.deepEqual(plan.ownerDimensions, [48, 32, 24]);
  assert.deepEqual(plan.fineDimensions, [96, 64, 48]);
  assert.deepEqual(plan.sourceFineDimensions, [32, 16, 16]);
  assert.deepEqual(plan.sourceOriginFine, [32, 16, 16]);
  assert.throws(() => planSvoFinePhiStaging(owner, fakeSource(), [0.2, 0.4, 0.6], {
    sourceOriginBricks: [5, 1, 1],
  }), /falls outside/);
});

test("fine-phi publication oracle rejects stale and mismatched generations before publishing capability", () => {
  const fence = new SvoFinePhiPublicationMirror();
  assert.equal(fence.publish({ structuralGeneration: 7, fineGeneration: 3, sourceFineGeneration: 2, ownerReady: true }), SVO_FINE_PHI_STATUS.sourceRejected);
  assert.equal(fence.acceptedFineGeneration, 0);
  assert.equal(fence.publish({ structuralGeneration: 7, fineGeneration: 3, sourceFineGeneration: 3, ownerReady: true }), SVO_FINE_PHI_STATUS.ready);
  assert.equal(fence.publish({ structuralGeneration: 7, fineGeneration: 3, sourceFineGeneration: 3, ownerReady: true }), SVO_FINE_PHI_STATUS.unchanged);
  assert.equal(fence.publish({ structuralGeneration: 6, fineGeneration: 4, sourceFineGeneration: 4, ownerReady: true }), SVO_FINE_PHI_STATUS.stale);
  assert.equal(fence.publish({ structuralGeneration: 8, fineGeneration: 4, sourceFineGeneration: 4, ownerReady: false }), SVO_FINE_PHI_STATUS.sourceRejected);
  assert.deepEqual([fence.acceptedStructuralGeneration, fence.acceptedFineGeneration], [7, 3]);
});

test("apron-compatible trilinear values and anisotropic gradients remain continuous across an owner-brick seam", () => {
  const dimensions = [32, 8, 8] as const;
  const values = new Float32Array(dimensions[0] * dimensions[1] * dimensions[2]);
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) for (let x = 0; x < dimensions[0]; x += 1) {
    values[x + dimensions[0] * (y + dimensions[1] * z)] = 2 * x + 3 * y - 4 * z;
  }
  const left = sampleFinePhiReference(values, dimensions, [15.999, 3.25, 2.5]);
  const right = sampleFinePhiReference(values, dimensions, [16.001, 3.25, 2.5]);
  assert.ok(Math.abs((right - left) - 0.004) < 1e-5);
  const gradient = gradientFinePhiReference(values, dimensions, [16, 3.25, 2.5], [0.5, 0.25, 2]);
  assert.deepEqual(gradient, [4, 12, -2]);
});

test("staging and sampling WGSL fence both generations, copy aprons, scrub retirement, and fail to coarse", () => {
  assert.match(svoFinePhiStagingShader, /fineGeneration!=params\.source\.y/);
  assert.match(svoFinePhiStagingShader, /ownerGeneration!=structural/);
  assert.match(svoFinePhiStagingShader, /tileCoordinate\(local\)/);
  assert.match(svoFinePhiStagingShader, /bitcast<u32>\(AIR_PHI\)/);
  assert.match(svoFinePhiStagingShader, /atomicStore\(&fineArena\[19\].*atomicStore\(&fineArena\[20\]/s);
  const sampling = svoFinePhiSamplingWGSL(0, 0, 1, 2);
  assert.match(sampling, /acceptedStructuralGeneration|svoFineArena\[19u\]!=expectedStructural/);
  assert.match(sampling, /svoFineArena\[svoFineParams\.offsets\.y\+logical\]!=fineGeneration/);
  assert.match(sampling, /svoFinePhi\(position-vec3f\(1,0,0\)/);
  assert.doesNotMatch(svoFinePhiStagingShader + sampling, /mapAsync|directWater|textureStore/);
});

test("fine-phi stager binds all producer resources read-only and encodes GPU-only prepare/stage/finalize", (t) => {
  const previousUsage = Object.getOwnPropertyDescriptor(globalThis, "GPUBufferUsage");
  const previousStage = Object.getOwnPropertyDescriptor(globalThis, "GPUShaderStage");
  Object.defineProperty(globalThis, "GPUBufferUsage", { configurable: true, value: { STORAGE: 1, COPY_SRC: 2, COPY_DST: 4, UNIFORM: 8, INDIRECT: 16 } });
  Object.defineProperty(globalThis, "GPUShaderStage", { configurable: true, value: { COMPUTE: 1 } });
  t.after(() => {
    if (previousUsage) Object.defineProperty(globalThis, "GPUBufferUsage", previousUsage); else Reflect.deleteProperty(globalThis, "GPUBufferUsage");
    if (previousStage) Object.defineProperty(globalThis, "GPUShaderStage", previousStage); else Reflect.deleteProperty(globalThis, "GPUShaderStage");
  });
  const ownerPlan = planOctreeOwnerPages([16, 8, 8]);
  const ownerArena = fakeBuffer(ownerPlan.allocatedBytes, "owner arena"), retiredSlots = fakeBuffer(8, "retired slots");
  const owner = { plan: ownerPlan, arena: ownerArena, retiredSlots, storageBinding: () => ({ buffer: ownerArena, offset: 0, size: ownerPlan.allocatedBytes }) } as unknown as WebGPUSvoOwnerPageAllocator;
  const residencyLayout = svoRenderResidencyConsumerLayout(2);
  const residency = { layout: residencyLayout, control: fakeBuffer(256, "residency control"), entries: fakeBuffer(1024, "residency entries") } as unknown as WebGPUSvoRenderResidencyConsumer;
  const source = fakeSource();
  const created: Array<{ label?: string; size: number; destroyed: boolean }> = [], writes: unknown[][] = [], layouts: GPUBindGroupLayoutEntry[][] = [];
  const device = {
    limits: { maxStorageBufferBindingSize: 1 << 28, maxBufferSize: 1 << 28 },
    createBuffer: (descriptor: GPUBufferDescriptor) => { const buffer = { ...descriptor, destroyed: false, destroy() { buffer.destroyed = true; } }; created.push(buffer); return buffer; },
    queue: { writeBuffer: (...args: unknown[]) => writes.push(args) },
    createBindGroupLayout: ({ entries }: GPUBindGroupLayoutDescriptor) => { layouts.push([...entries]); return {}; },
    createShaderModule: ({ code }: GPUShaderModuleDescriptor) => ({ code, getCompilationInfo: async () => ({ messages: [] }) }),
    createPipelineLayout: () => ({}),
    createComputePipeline: ({ compute }: GPUComputePipelineDescriptor) => ({ entryPoint: compute.entryPoint }),
    createBindGroup: ({ entries }: GPUBindGroupDescriptor) => ({ entries }),
  } as unknown as GPUDevice;
  const stager = new WebGPUSvoFinePhiStager(device, owner, residency, source, [0.2, 0.2, 0.2]);
  assert.deepEqual(layouts[0].slice(0, 7).map((entry) => entry.buffer?.type), Array(7).fill("read-only-storage"));
  assert.equal(layouts[0][7].buffer?.type, "storage");
  const calls: unknown[][] = [];
  const encoder = {
    copyBufferToBuffer: (...args: unknown[]) => calls.push(["copy", ...args]),
    beginComputePass: ({ label }: GPUComputePassDescriptor) => ({
      setPipeline: (pipeline: unknown) => calls.push([label, "pipeline", pipeline]), setBindGroup: () => {},
      dispatchWorkgroups: (...args: unknown[]) => calls.push([label, "direct", ...args]),
      dispatchWorkgroupsIndirect: (...args: unknown[]) => calls.push([label, "indirect", ...args]), end: () => calls.push([label, "end"]),
    }),
  } as unknown as GPUCommandEncoder;
  stager.encode(encoder, 3);
  assert.deepEqual(calls.filter((call) => call[0] === "copy").map((call) => call[2]), [112, 160]);
  assert.equal(calls.filter((call) => call[1] === "direct").length, 2);
  assert.deepEqual(calls.filter((call) => call[1] === "indirect").map((call) => call[3]), [12, 0]);
  assert.equal("readStats" in stager, false);
  assert.deepEqual(stager.capability().coarseFallbackRequired, true);
  assert.deepEqual(stager.capability().directWaterOwnership, false);
  assert.equal(stager.capability().statusWord, SVO_FINE_PHI_CONTROL_WORDS.status);
  stager.destroy(); stager.destroy();
  assert.equal(created.filter((buffer) => buffer.destroyed).length, 3);
  assert.ok(writes.length >= 3);
  assert.throws(() => new WebGPUSvoFinePhiStager(device, owner, residency, fakeSource({ mode: "mirror" }), [0.2, 0.2, 0.2]), /authoritative/);
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
async function createDevice() {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: optionalFluidDeviceFeatures(adapter.features), requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = []; device.addEventListener("uncapturederror", (event: unknown) => validationErrors.push((event as { error: { message: string } }).error.message));
  return { device, validationErrors };
}

test("real GPU fine staging publishes seams, rejects stale source generations, and scrubs retired tiles", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU fine-phi checks" }, async () => {
  const { device, validationErrors } = await createDevice();
  const ownerPlan = planOctreeOwnerPages([16, 8, 8]);
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;
  const ownerArena = device.createBuffer({ size: ownerPlan.allocatedBytes, usage: storage });
  const retiredSlots = device.createBuffer({ size: 8, usage: storage });
  const owner = { plan: ownerPlan, arena: ownerArena, retiredSlots, storageBinding: () => ({ buffer: ownerArena, offset: 0, size: ownerPlan.allocatedBytes }) } as unknown as WebGPUSvoOwnerPageAllocator;
  const residencyLayout = svoRenderResidencyConsumerLayout(2);
  const residencyControl = device.createBuffer({ size: residencyLayout.controlByteLength, usage: storage | GPUBufferUsage.INDIRECT });
  const residencyEntries = device.createBuffer({ size: residencyLayout.entryByteLength, usage: storage });
  const residency = { layout: residencyLayout, control: residencyControl, entries: residencyEntries } as unknown as WebGPUSvoRenderResidencyConsumer;
  const surfaceControl = device.createBuffer({ size: 64, usage: storage });
  const surfacePageTable = device.createBuffer({ size: 16 * 4, usage: storage });
  const surfacePhi = device.createBuffer({ size: 16 * 512 * 4, usage: storage });
  const surfaceParams = device.createBuffer({ size: 112, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const source = fakeSource({
    control: { buffer: surfaceControl }, pageTable: { buffer: surfacePageTable }, phi: { buffer: surfacePhi }, params: { buffer: surfaceParams }, revision: 3,
  });
  const ownerWords = new Uint32Array(ownerPlan.allocatedWords); ownerWords[7] = 7; ownerWords[10] = SVO_OWNER_PAGE_STATUS.ready;
  ownerWords[ownerPlan.pageTableOffsetWords] = 1; ownerWords[ownerPlan.pageTableOffsetWords + 1] = 2; device.queue.writeBuffer(ownerArena, 0, ownerWords);
  const surfaceControlWords = new Uint32Array(16); surfaceControlWords[1] = 3; device.queue.writeBuffer(surfaceControl, 0, surfaceControlWords);
  device.queue.writeBuffer(surfacePageTable, 0, Uint32Array.from({ length: 16 }, (_, index) => index));
  const phi = new Float32Array(16 * 512);
  for (let logical = 0; logical < 16; logical += 1) {
    const bx = logical % 4, by = Math.floor(logical / 4) % 2, bz = Math.floor(logical / 8);
    for (let z = 0; z < 8; z += 1) for (let y = 0; y < 8; y += 1) for (let x = 0; x < 8; x += 1) phi[logical * 512 + x + 8 * (y + 8 * z)] = bx * 8 + x + 2 * (by * 8 + y) - (bz * 8 + z);
  }
  device.queue.writeBuffer(surfacePhi, 0, phi);
  const surfaceParamWords = new Uint32Array(28); surfaceParamWords.set([16, 8, 8, 2], 0); surfaceParamWords.set([32, 16, 16, 8], 4); surfaceParamWords.set([4, 2, 2, 16], 8); device.queue.writeBuffer(surfaceParams, 0, surfaceParamWords);
  const residencyWords = new Uint32Array(64); residencyWords[0] = 1; residencyWords[2] = 7; residencyWords[7] = 2; residencyWords[28] = 1; residencyWords[29] = 1; residencyWords[30] = 1; device.queue.writeBuffer(residencyControl, 0, residencyWords);
  const entryWords = new Uint32Array(residencyLayout.entryByteLength / 4); entryWords[0] = 0; entryWords[2] = 1; device.queue.writeBuffer(residencyEntries, 0, entryWords);
  const stager = new WebGPUSvoFinePhiStager(device, owner, residency, source, [1, 1, 1]);
  const readback = device.createBuffer({ size: stager.plan.allocatedBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const submit = async (generation: number) => {
    const encoder = device.createCommandEncoder(); stager.encode(encoder, generation); encoder.copyBufferToBuffer(stager.arena, 0, readback, 0, stager.plan.allocatedBytes); device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ); const result = new Uint32Array(readback.getMappedRange().slice(0)); readback.unmap(); return result;
  };
  try {
    let result = await submit(3);
    assert.ok((result[SVO_FINE_PHI_CONTROL_WORDS.status] & SVO_FINE_PHI_STATUS.ready) !== 0);
    assert.deepEqual([result[SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration], result[SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration]], [7, 3]);
    assert.deepEqual(Array.from(result.slice(stager.plan.pageGenerationOffsetWords, stager.plan.pageGenerationOffsetWords + 2)), [3, 3]);
    const stagedValue = (position: readonly [number, number, number]) => {
      const ownerBrick = Math.min(1, Math.floor(position[0] / 16)), local = [position[0] - ownerBrick * 16 + 1, position[1] + 1, position[2] + 1];
      const a = local.map(Math.floor), b = a.map((value) => value + 1), t = local.map((value, axis) => value - a[axis]);
      const payload = new Float32Array(result.buffer);
      const at = (x: number, y: number, z: number) => payload[stager.plan.payloadOffsetWords + ownerBrick * stager.plan.tileVoxels + x + 18 * (y + 18 * z)];
      const mix = (x: number, y: number, amount: number) => x + (y - x) * amount;
      const z0 = mix(mix(at(a[0], a[1], a[2]), at(b[0], a[1], a[2]), t[0]), mix(at(a[0], b[1], a[2]), at(b[0], b[1], a[2]), t[0]), t[1]);
      const z1 = mix(mix(at(a[0], a[1], b[2]), at(b[0], a[1], b[2]), t[0]), mix(at(a[0], b[1], b[2]), at(b[0], b[1], b[2]), t[0]), t[1]);
      return mix(z0, z1, t[2]);
    };
    assert.ok(Math.abs((stagedValue([16.001, 3.25, 2.5]) - stagedValue([15.999, 3.25, 2.5])) - 0.002) < 1e-4,
      "adjacent owner tiles share the same source-defined seam through their aprons");
    surfaceControlWords[1] = 3; device.queue.writeBuffer(surfaceControl, 0, surfaceControlWords); ownerWords[7] = 8; ownerWords[ownerPlan.pageTableOffsetWords] = 0; device.queue.writeBuffer(ownerArena, 0, ownerWords);
    residencyWords[2] = 8; residencyWords[7] = 1; residencyWords[10] = 1; residencyWords[28] = 1; residencyWords[40] = 1; residencyWords[41] = 1; residencyWords[42] = 1; device.queue.writeBuffer(residencyControl, 0, residencyWords);
    entryWords[0] = 1; entryWords[residencyLayout.entryOffsetsBytes.retired / 4] = 0; device.queue.writeBuffer(residencyEntries, 0, entryWords); device.queue.writeBuffer(retiredSlots, 0, new Uint32Array([0]));
    result = await submit(4);
    assert.equal(result[SVO_FINE_PHI_CONTROL_WORDS.status], SVO_FINE_PHI_STATUS.sourceRejected);
    assert.deepEqual([result[SVO_FINE_PHI_CONTROL_WORDS.acceptedStructuralGeneration], result[SVO_FINE_PHI_CONTROL_WORDS.acceptedFineGeneration]], [7, 3], "stale source cannot advance capability");
    surfaceControlWords[1] = 4; device.queue.writeBuffer(surfaceControl, 0, surfaceControlWords); result = await submit(4);
    assert.equal(result[stager.plan.pageGenerationOffsetWords], 0); assert.equal(result[SVO_FINE_PHI_CONTROL_WORDS.retiredClearedPageCount], 1);
    const payload = new Float32Array(result.buffer, stager.plan.payloadOffsetWords * 4, stager.plan.tileVoxels);
    assert.equal(payload.every((value) => value === 1_000_000), true, "retired physical tile is deterministically scrubbed");
    await device.queue.onSubmittedWorkDone(); assert.deepEqual(validationErrors, []);
  } finally {
    stager.destroy(); readback.destroy(); ownerArena.destroy(); retiredSlots.destroy(); residencyControl.destroy(); residencyEntries.destroy(); surfaceControl.destroy(); surfacePageTable.destroy(); surfacePhi.destroy(); surfaceParams.destroy(); device.destroy();
  }
});
