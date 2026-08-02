import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_CONSUMER_MAX_FACE_CANDIDATES,
  createGlobalFineLevelSetConsumerSource,
  globalFineCoarseGenerationPairIsValid,
  octreeConsumerSamplingWGSL,
  planOctreeConsumerTraffic,
  sampleOctreeFaceVelocity,
  validateGlobalFineLevelSetConsumerSource,
  type GlobalFineLevelSetConsumerSource,
  type OctreeConsumerFaceSample,
} from "../lib/octree-consumer-sampling";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";

function faces(value: readonly [number, number, number]): OctreeConsumerFaceSample[] {
  return [0, 1, 2].flatMap((axis) => [0, 1].map((plane) => ({
    origin: [axis === 0 ? plane : 0, axis === 1 ? plane : 0, axis === 2 ? plane : 0] as const,
    axis: axis as 0 | 1 | 2,
    span: 1,
    normalVelocity: value[axis],
  })));
}

test("adaptive face sampler exactly preserves a constant vector field", () => {
  sampleOctreeFaceVelocity([0.37, 0.61, 0.24], faces([2, -3, 4]))
    .forEach((value, axis) => assert.ok(Math.abs(value - [2, -3, 4][axis]) < 1e-12));
});

test("adaptive face sampler is resolution aware across a 2:1 neighbourhood", () => {
  const candidates = faces([1, 2, 3]);
  candidates.push({ origin: [0, 0, 0], axis: 0, span: 2, normalVelocity: 1 });
  const sampled = sampleOctreeFaceVelocity([0.8, 0.7, 0.6], candidates);
  sampled.forEach((value, axis) => assert.ok(Math.abs(value - axis - 1) < 1e-12));
});

test("adaptive face sampler enforces the bounded 2:1 incidence contract", () => {
  const candidates = Array.from({ length: OCTREE_CONSUMER_MAX_FACE_CANDIDATES + 1 }, () => faces([1, 1, 1])[0]);
  assert.throws(() => sampleOctreeFaceVelocity([0, 0, 0], candidates), /48-face/);
});

test("global fine consumer ABI indexes canonical factor-4 and factor-8 lattices", () => {
  const buffer = {} as GPUBuffer;
  const source = (factor: 4 | 8): GlobalFineLevelSetConsumerSource => ({
    kind: "global-fine-levelset-sampling", metadata: { buffer }, worklist: { buffer },
    flags: { buffer }, phi: { buffer }, coarsePhiDirectory: { buffer }, coarsePhiRowCapacity: 128,
    topologyControl: { buffer },
    sampleDimensions: [60 * factor, 45 * factor, 40 * factor],
    brickDimensions: [Math.ceil(60 * factor / 4), Math.ceil(45 * factor / 4), Math.ceil(40 * factor / 4)],
    brickResolution: 4, samplesPerBrick: 64,
    pageCapacity: 100, fineFactor: factor, fineCellWidth: 0.05 / factor,
    domainOrigin: [0, 0, 0], generation: 7,
  });
  assert.doesNotThrow(() => validateGlobalFineLevelSetConsumerSource(source(4)));
  assert.doesNotThrow(() => validateGlobalFineLevelSetConsumerSource(source(8)));
});

test("global fine source adapter aliases the single SPGrid and compact coarse directory", () => {
  const buffer = {} as GPUBuffer;
  const coarse = {} as GPUBuffer;
  const topology = {} as GPUBuffer;
  const plan = planFineLevelSetBricks({ domainOrigin: [0, 0, 0], finestCellDimensions: [4, 3, 2],
    finestCellWidth: 1, fineFactor: 8, brickResolution: 4, maximumResidentBricks: 8 });
  const consumer = createGlobalFineLevelSetConsumerSource({ plan, generation: 3, generationSlot: 1,
    params: buffer, metadata: buffer, worklist: buffer, flags: buffer, phi: buffer,
    workA: buffer, workB: buffer, rollbackPhi: buffer,
    coarsePhiDirectory: coarse, coarsePhiRowCapacity: 16, topologyControl: topology });
  assert.equal(consumer.phi.buffer, buffer);
  assert.equal(consumer.worklist.buffer, buffer);
  assert.equal(consumer.coarsePhiDirectory?.buffer, coarse);
  assert.equal(consumer.topologyControl?.buffer, topology);
  assert.equal(consumer.fineFactor, 8);
  assert.deepEqual(consumer.sampleDimensions, [32, 24, 16]);
  assert.deepEqual(consumer.brickDimensions, [8, 6, 4]);
});

test("global fine consumer ABI rejects ambiguous indexing and incomplete coarse fallback", () => {
  const buffer = {} as GPUBuffer;
  const valid: GlobalFineLevelSetConsumerSource = {
    kind: "global-fine-levelset-sampling", metadata: { buffer }, worklist: { buffer },
    flags: { buffer }, phi: { buffer }, coarsePhiDirectory: { buffer }, coarsePhiRowCapacity: 8,
    topologyControl: { buffer },
    sampleDimensions: [16, 12, 8], brickDimensions: [4, 3, 2], brickResolution: 4,
    samplesPerBrick: 64, pageCapacity: 8,
    fineFactor: 4, fineCellWidth: 0.25, domainOrigin: [0, 0, 0], generation: 1,
  };
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, samplesPerBrick: 63 }), /stride/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, brickDimensions: [4, 2, 2] }), /complete logical/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, coarsePhiRowCapacity: undefined }), /provided together/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, topologyControl: undefined }),
    /requires current-slot topology provenance/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid,
    sampleDimensions: [65_536, 12, 8], brickDimensions: [16_384, 3, 2] }), /16-bit/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid,
    domainOrigin: [1, 0, 0] }), /zero domain origin/);
});

test("global fine render epochs require one clean current publication", () => {
  const clean = new Uint32Array([0, 1, 1, 1, 1, 0, 1, 0]);
  const rollback = new Uint32Array([16, 1, 1, 1, 1, 1, 1, 2]);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 7, clean), true);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 6, rollback), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 5, rollback), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 7, rollback), false,
    "a rejected fine rebuild cannot authorize a stale Section-5 correction");
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 6, clean), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 7, undefined), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 5,
    new Uint32Array([0x20, 1, 1, 1, 1, 1, 1, 2])), false,
  "unknown rollback flags remain fail-closed");
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 5,
    new Uint32Array([16, 1, 1, 1, 1, 1, 1, 0])), false,
  "a rollback without a rejection reason remains fail-closed");
});

test("traffic plan reports eliminated dense consumer allocation without hiding gather cost", () => {
  const plan = planOctreeConsumerTraffic({
    finestCellCount: 1_000_000, velocityQueries: 100, phiQueries: 100,
    averageFaceCandidatesPerVelocityQuery: 12, legacyPublicationBytes: 2_000_000,
  });
  assert.equal(plan.densePersistentBytes, 22_000_000);
  assert.equal(plan.adaptivePersistentBytes, 0);
  assert.equal(plan.persistentBytesAvoided, 22_000_000);
  assert.ok(plan.adaptiveFieldReadBytesUpperBound > plan.denseFieldReadBytes,
    "uncached scattered face gathers must not be presented as a bandwidth win");
});

test("WGSL library is binding-neutral and exposes the shared velocity sampler", () => {
  assert.doesNotMatch(octreeConsumerSamplingWGSL, /@group|@binding/);
  assert.match(octreeConsumerSamplingWGSL, /fn octreeConsumerVelocity/);
  assert.doesNotMatch(octreeConsumerSamplingWGSL, /octreeConsumerPhi|pageResolution|PageLoad|FineSeedLeaf/);
  assert.match(octreeConsumerSamplingWGSL, /array<OctreeConsumerFaceSample,48>/);
});

test("Dawn executes the shared adaptive velocity sampler", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU consumer checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=metal"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const shader = `${octreeConsumerSamplingWGSL}
@group(0) @binding(0) var<storage,read_write> result:array<vec4f,1>;
@compute @workgroup_size(1) fn main(){
  var candidates:array<OctreeConsumerFaceSample,48>;
  candidates[0]=OctreeConsumerFaceSample(0u,0u,0u,4u,1.0,0u);
  candidates[1]=OctreeConsumerFaceSample(0u,0u,0u,5u,2.0,0u);
  candidates[2]=OctreeConsumerFaceSample(0u,0u,0u,6u,3.0,0u);
  let velocity=octreeConsumerVelocity(vec3f(0.5),candidates,3u,vec3f(0.0));
  result[0]=vec4f(velocity,1.0);
}`;
  const shaderModule = device.createShaderModule({ code: shader });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });
  const output = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const readback = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, 16); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  const values = [...new Float32Array(readback.getMappedRange().slice(0))];
  assert.deepEqual(values, [1, 2, 3, 1]);
  readback.unmap(); output.destroy(); readback.destroy(); device.destroy();
});
