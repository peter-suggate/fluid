import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_CONSUMER_MAX_FACE_CANDIDATES,
  createGlobalFineLevelSetConsumerSource,
  createUnifiedOctreeConsumerAdapters,
  createUnifiedOctreeConsumerSource,
  globalFineCoarseGenerationPairIsValid,
  octreeConsumerSamplingWGSL,
  planOctreeConsumerTraffic,
  sampleOctreeFaceVelocity,
  sampleOctreeSurfacePhi,
  validateGlobalFineLevelSetConsumerSource,
  type GlobalFineLevelSetConsumerSource,
  type OctreeConsumerFaceSample,
  type UnifiedOctreeConsumerSource,
} from "../lib/octree-consumer-sampling";
import { planFineLevelSetBricks } from "../lib/octree-fine-levelset-bricks";
import { planOctreeSurfacePages } from "../lib/webgpu-octree-surface-pages";

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

test("surface sampling uses the leaf plane when no narrow-band page is resident", () => {
  assert.equal(sampleOctreeSurfacePhi([3, 2, 2], {
    origin: [0, 0, 0], size: 4, phiGradient: [5, 2, 0, 0],
  }), 7);
});

test("surface sampling trilinearly reconstructs a linear 4-cubed page", () => {
  const page = new Float32Array(64);
  for (let z = 0; z < 4; z += 1) for (let y = 0; y < 4; y += 1) for (let x = 0; x < 4; x += 1) {
    page[x + 4 * (y + 4 * z)] = (x + 0.5) + 2 * (y + 0.5) - (z + 0.5);
  }
  assert.ok(Math.abs(sampleOctreeSurfacePhi([1.2, 2.1, 1.7], {
    origin: [0, 0, 0], size: 4, phiGradient: [0, 0, 0, 0], phiPage: page,
  }) - (1.2 + 2 * 2.1 - 1.7)) < 1e-6);
});

test("surface sampling trilinearly reconstructs the default linear 2-cubed page", () => {
  const page = new Float32Array(8);
  for (let z = 0; z < 2; z += 1) for (let y = 0; y < 2; y += 1) for (let x = 0; x < 2; x += 1) {
    page[x + 2 * (y + 2 * z)] = (2 * x + 1) + 2 * (2 * y + 1) - (2 * z + 1);
  }
  assert.ok(Math.abs(sampleOctreeSurfacePhi([1.2, 2.1, 1.7], {
    origin: [0, 0, 0], size: 4, phiGradient: [0, 0, 0, 0], phiPage: page,
  }) - (1.2 + 2 * 2.1 - 1.7)) < 1e-6);
});

test("surface sampling rejects non-canonical page payloads", () => {
  assert.throws(() => sampleOctreeSurfacePhi([1, 1, 1], {
    origin: [0, 0, 0], size: 4, phiGradient: [0, 0, 0, 0], phiPage: new Float32Array(27),
  }), /8 or 64 samples/);
});

test("consumer adapters alias compact buffers and permit only transient diagnostics", () => {
  const source = { kind: "unified-octree-sampling", faceCapacity: 10, leafCapacity: 4, pageCapacity: 2, generation: 7 } as UnifiedOctreeConsumerSource;
  const adapters = createUnifiedOctreeConsumerAdapters(source);
  assert.equal(adapters.renderer.source, source);
  assert.equal(adapters.renderer.materialization, "none");
  assert.equal(adapters.particles.materialization, "none");
  assert.equal(adapters.diagnostics.materialization, "transient-output-only");
});

test("source adapter rejects topology generations with different leaf capacities", () => {
  const buffer = {} as GPUBuffer;
  const face = {
    plan: { rowCapacity: 4, faceCapacity: 10 }, control: buffer, faces: buffer, incidence: buffer,
  } as Parameters<typeof createUnifiedOctreeConsumerSource>[0];
  const validPlan = planOctreeSurfacePages(4, [4, 4, 4], { maximumPages: 2 });
  const surface = {
    plan: { ...validPlan, leafCapacity: 5 }, arena: { buffer }, leaves: { buffer }, params: { buffer },
    activePages: { indirectBuffer: buffer, indirectOffsetBytes: 0 },
    phiAOffsetBytes: validPlan.phiAOffsetWords * 4, pageTableOffsetBytes: validPlan.pageTableOffsetWords * 4,
  } as Parameters<typeof createUnifiedOctreeConsumerSource>[1];
  assert.throws(() => createUnifiedOctreeConsumerSource(face, surface), /capacities must match/);
  surface.plan = validPlan;
  const source = createUnifiedOctreeConsumerSource(face, surface, 3);
  assert.equal(source.faces.buffer, buffer);
  assert.equal(source.surfaceArena.buffer, buffer);
  assert.equal(source.generation, 3);
});

test("global fine consumer ABI indexes canonical factor-4 and factor-8 lattices", () => {
  const buffer = {} as GPUBuffer;
  const source = (factor: 4 | 8): GlobalFineLevelSetConsumerSource => ({
    kind: "global-fine-levelset-sampling", hash: { buffer }, metadata: { buffer }, worklist: { buffer },
    flags: { buffer }, phi: { buffer }, coarsePhiDirectory: { buffer }, coarsePhiHashCapacity: 128,
    topologyControl: { buffer },
    sampleDimensions: [60 * factor, 45 * factor, 40 * factor],
    brickDimensions: [Math.ceil(60 * factor / 4), Math.ceil(45 * factor / 4), Math.ceil(40 * factor / 4)],
    brickResolution: 4, samplesPerBrick: 64, hashCapacity: 256, maximumHashProbes: 32,
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
    params: buffer, hash: buffer, metadata: buffer, worklist: buffer, flags: buffer, phi: buffer,
    workA: buffer, workB: buffer, rollbackPhi: buffer,
    coarsePhiDirectory: coarse, coarsePhiHashCapacity: 16, topologyControl: topology });
  assert.equal(consumer.phi.buffer, buffer);
  assert.equal(consumer.hash.buffer, buffer);
  assert.equal(consumer.coarsePhiDirectory?.buffer, coarse);
  assert.equal(consumer.topologyControl?.buffer, topology);
  assert.equal(consumer.fineFactor, 8);
  assert.deepEqual(consumer.sampleDimensions, [32, 24, 16]);
  assert.deepEqual(consumer.brickDimensions, [8, 6, 4]);
});

test("global fine consumer ABI rejects ambiguous indexing and incomplete coarse fallback", () => {
  const buffer = {} as GPUBuffer;
  const valid: GlobalFineLevelSetConsumerSource = {
    kind: "global-fine-levelset-sampling", hash: { buffer }, metadata: { buffer }, worklist: { buffer },
    flags: { buffer }, phi: { buffer }, coarsePhiDirectory: { buffer }, coarsePhiHashCapacity: 8,
    topologyControl: { buffer },
    sampleDimensions: [16, 12, 8], brickDimensions: [4, 3, 2], brickResolution: 4,
    samplesPerBrick: 64, hashCapacity: 16, maximumHashProbes: 8, pageCapacity: 8,
    fineFactor: 4, fineCellWidth: 0.25, domainOrigin: [0, 0, 0], generation: 1,
  };
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, samplesPerBrick: 63 }), /stride/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, brickDimensions: [4, 2, 2] }), /complete logical/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, hashCapacity: 15 }), /power of two/);
  assert.throws(() => validateGlobalFineLevelSetConsumerSource({ ...valid, coarsePhiHashCapacity: undefined }), /provided together/);
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
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 6, rollback), false,
    "a retagged rollback field is not a new render publication");
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 7, rollback), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 6, clean), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 5, rollback), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(7, 7, undefined), false);
  assert.equal(globalFineCoarseGenerationPairIsValid(0, 0x3fff_ffff, rollback), false,
    "a wrapped prior rollback epoch remains rejected");
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

test("WGSL library is binding-neutral and exposes velocity and phi entry functions", () => {
  assert.doesNotMatch(octreeConsumerSamplingWGSL, /@group|@binding/);
  assert.match(octreeConsumerSamplingWGSL, /fn octreeConsumerVelocity/);
  assert.match(octreeConsumerSamplingWGSL, /fn octreeConsumerPhi/);
  assert.match(octreeConsumerSamplingWGSL, /pageResolution:u32/);
  assert.match(octreeConsumerSamplingWGSL, /octreeConsumerPageIndex\(q:vec3u,resolution:u32\)/);
  assert.match(octreeConsumerSamplingWGSL, /octreeConsumerPageLoad\(pageBase/);
  assert.doesNotMatch(octreeConsumerSamplingWGSL, /page:array<f32,64>/);
  assert.doesNotMatch(octreeConsumerSamplingWGSL, /octreeConsumerPageIndex4/);
  assert.match(octreeConsumerSamplingWGSL, /array<OctreeConsumerFaceSample,48>/);
});

test("Dawn executes the shared adaptive velocity and phi sampler", {
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
@group(0) @binding(0) var<storage,read_write> result:array<vec4f,2>;
@group(0) @binding(1) var<storage,read_write> page:array<f32>;
fn octreeConsumerPageLoad(base:u32,index:u32)->f32{return page[base+index];}
@compute @workgroup_size(1) fn main(){
  var candidates:array<OctreeConsumerFaceSample,48>;
  candidates[0]=OctreeConsumerFaceSample(0u,0u,0u,4u,1.0,0u);
  candidates[1]=OctreeConsumerFaceSample(0u,0u,0u,5u,2.0,0u);
  candidates[2]=OctreeConsumerFaceSample(0u,0u,0u,6u,3.0,0u);
  let leaf=OctreeConsumerSurfaceLeaf(0u,0u,0u,4u,vec4f(5.0,2.0,0.0,0.0));
  let velocity=octreeConsumerVelocity(vec3f(0.5),candidates,3u,vec3f(0.0));
  result[0]=vec4f(velocity,octreeConsumerPhi(vec3f(3.0,2.0,2.0),leaf,0u,2u,false));
  for(var z=0u;z<2u;z+=1u){for(var y=0u;y<2u;y+=1u){for(var x=0u;x<2u;x+=1u){page[x+2u*(y+2u*z)]=f32(2u*x+1u)+2.0*f32(2u*y+1u)-f32(2u*z+1u);}}}
  result[1]=vec4f(octreeConsumerPhi(vec3f(1.2,2.1,1.7),leaf,0u,2u,true),0.0,0.0,0.0);
}`;
  const shaderModule = device.createShaderModule({ code: shader });
  assert.deepEqual((await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error"), []);
  const pipeline = device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "main" } });
  const output = device.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  const pageBuffer = device.createBuffer({ size: 64 * 4, usage: GPUBufferUsage.STORAGE });
  const readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const group = device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: output } }, { binding: 1, resource: { buffer: pageBuffer } }] });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass(); pass.setPipeline(pipeline); pass.setBindGroup(0, group); pass.dispatchWorkgroups(1); pass.end();
  encoder.copyBufferToBuffer(output, 0, readback, 0, 32); device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  await device.queue.onSubmittedWorkDone();
  assert.deepEqual(errors, []);
  const values = [...new Float32Array(readback.getMappedRange().slice(0))];
  assert.deepEqual(values.slice(0,4), [1, 2, 3, 7]);
  assert.ok(Math.abs(values[4]-3.7)<1e-5);
  readback.unmap(); output.destroy(); pageBuffer.destroy(); readback.destroy(); device.destroy();
});
