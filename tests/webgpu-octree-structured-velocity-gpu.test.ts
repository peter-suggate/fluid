import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { decodeGeneratedOctreePowerCatalog } from "../lib/generated/octree-power-catalog";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { WebGPUOctreePowerTopology } from "../lib/webgpu-octree-power-topology";
import { usePerformanceInstrumentationStore } from "../lib/stores/performance-instrumentation-store";
import {
  OCTREE_STRUCTURED_GPU_ERROR,
  WebGPUDirectStructuredVelocityAuthority,
  directStructuredVelocityPublicationWGSL,
  planStructuredVelocityGPU,
} from "../lib/webgpu-octree-structured-velocity-gpu";

// Numerical harnesses submit their own raw encoders; exercise the production
// shader variant instead of the solver-owned activity binding session.
usePerformanceInstrumentationStore.getState().setMode("timeline");

test("direct structured authority has six fixed families and nine disjoint worksets", () => {
  const plan = planStructuredVelocityGPU(128, 30, 256);
  assert.equal(plan.slotCapacity, 3_840);
  assert.equal(plan.offsets.rowFamilyPrefix + 6 * 128, plan.offsets.rowFamilyHandles);
  assert.equal(plan.offsets.rowFamilyHandles + 6 * 128, plan.offsets.rowFamilySlots);
  assert.equal(plan.authorityWords, plan.offsets.rowFamilySlots + 48 * 128);
  assert.equal(plan.worksetBytes, 9 * plan.worksetStrideWords * 4);
});

test("direct publisher contains no general face/incidence authority or floating scatter", () => {
  assert.match(directStructuredVelocityPublicationWGSL,
    /sourceRowDelta\[p\.deltaControlOffset\]/,
    "structured candidate rows must use the immutable row-delta transaction");
  assert.doesNotMatch(directStructuredVelocityPublicationWGSL, /sourceRowCount/,
    "dirty-tile compaction is not a candidate row-count authority");
  assert.doesNotMatch(directStructuredVelocityPublicationWGSL,
    /PowerFaceRecord|PowerIncidence|incidenceRows|atomicAdd\s*\([^)]*(?:value|velocity|rhs)/i);
  assert.match(directStructuredVelocityPublicationWGSL, /fn findRow\(/);
  assert.match(directStructuredVelocityPublicationWGSL, /fn reciprocal\(/);
  assert.match(directStructuredVelocityPublicationWGSL, /fn prefixStructuredFamilies\(/);
  assert.match(directStructuredVelocityPublicationWGSL, /fn finalizeStructuredPublication\(/);
});

test("classify and scatter run one invocation per (row, catalog slot)", () => {
  // `begin` used to publish a rows*48 record at word 3 that no dispatch read.
  // 48 is the rowFamilySlots stride (6 families x 8 orientations), not the
  // catalog-slot bound, so the record has to be `maxSlots` wide to index a
  // (row, slot) pair exactly once.
  // The record is published two-dimensionally: slots are rows * maximumFace-
  // Incidence, so at the catalog's 30 the block count passes the one-dimensional
  // workgroup limit near 140,000 rows and the indirect-args validator would
  // silently zero it. X is pinned at 65,535 on saturation so the kernels can
  // recover their linear item from a constant stride rather than a uniform.
  assert.match(directStructuredVelocityPublicationWGSL,
    /publishBlockDispatch\(3u,\(rows\*p\.maxSlots\+63u\)\/64u\)/,
    "word 3 is the (row, catalog slot) launch record consumed at byte offset 12");
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn publishBlockDispatch\(at:u32,blocks:u32\)\{let x=max\(1u,min\(65535u,blocks\)\);\s*publicationDispatch\[at\]=x;publicationDispatch\[at\+1u\]=\(blocks\+x-1u\)\/x;publicationDispatch\[at\+2u\]=1u;\}/,
    "a saturating record must pin X and carry the remainder in Y, not overflow one dimension");
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn foldedItem\(g:vec3u\)->u32\{return g\.x\+g\.y\*65535u\*64u;\}/,
    "the item fold must use the same pinned extent the record was published with");
  for (const entry of ["classifyStructuredCatalogSlots", "scatterStructuredFamilySlots"]) {
    assert.match(directStructuredVelocityPublicationWGSL,
      new RegExp(`fn ${entry}\\(@builtin\\(global_invocation_id\\)g:vec3u\\)\\{let rows=min\\(control\\.rowCount,p\\.rowCapacity\\);let slot=foldedItem\\(g\\);let row=slot/p\\.maxSlots;let local=slot%p\\.maxSlots;`),
      `${entry} must decode one (row, slot) pair per invocation from the folded item`);
  }
  const host = WebGPUDirectStructuredVelocityAuthority.prototype as unknown as
    Record<string, () => void>;
  const encode = host.encodeCandidatePasses.toString();
  assert.equal([...encode.matchAll(/dispatchWorkgroupsIndirect\(this\.liveRowDispatch,\s*12\)/g)].length, 2,
    "classify and scatter are the two stages launched off the (row, slot) record");
  // The per-row family fold the classifier used to do in place. It must stay a
  // single owner per row: the counters are a plain sequential accumulation, so
  // making it per-slot would need atomics, and atomics are not an ordering.
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn countStructuredRowFamilies\(@builtin\(global_invocation_id\)g:vec3u\)\{let row=g\.x;/);
  assert.doesNotMatch(directStructuredVelocityPublicationWGSL,
    /fn countStructuredRowFamilies\([\s\S]*?atomicAdd/,
    "family counts stay a deterministic per-row reduction, never atomics-as-ordering");
  // Scatter's handle used to come from a running per-row counter. Per slot it
  // is recomputed as "earlier owner slots of the same family in this row",
  // which is the same integer without any cross-invocation ordering.
  assert.match(directStructuredVelocityPublicationWGSL,
    /var rank=0u;for\(var earlier=0u;earlier<local;earlier\+=1u\)\{let earlierMeta=[\s\S]*?\(earlierMeta&7u\)==family\)\{rank\+=1u;\}\}/,
    "the scattered handle rank must be recomputed, not inherited from a serial counter");
});

test("Section 6.3 publication binds only its current reachable stage ABI", () => {
  // The stages moved from `encodeCandidate` into `encodeCandidatePasses` when
  // `encodeCandidate` became the repeat wrapper the wall-clock cost probe
  // needs. The invariant is unchanged -- read the method that now owns the
  // dispatches, so the assertion still fails if a binding creeps in.
  const prototype = WebGPUDirectStructuredVelocityAuthority.prototype as unknown as
    Record<string, () => void>;
  const encode = prototype.encodeCandidatePasses.toString();
  const stage = encode.slice(encode.indexOf("Publish direct Section 6.3 rows and worksets"),
    encode.indexOf("Finalize direct structured publication"));
  assert.ok(stage.length > 0, "Section 6.3 host stage must remain present");
  assert.match(stage, /setPipeline\(this\.section63Pipeline\)/);
  assert.match(stage, /this\.group\(this\.section63Pipeline/,
    "pipeline and inferred bind-group layout must come from the same entry point");
  assert.deepEqual([...stage.matchAll(/binding:\s*(\d+)/g)].map((match) => Number(match[1])),
    [0, 1, 2, 4, 5, 8, 9, 10, 11]);
  assert.doesNotMatch(stage, /binding:\s*6\b/,
    "publishSection63Rows does not reach the row-delta binding owned by scatter/finalize");
});

test("rejected structured candidates can mutate only inactive-bank and candidate-owned bytes", () => {
  assert.match(directStructuredVelocityPublicationWGSL,
    /control\.activeBank=select\(0u,1u-\(atomicLoad\(&acceptedControl\[4\]\)&1u\),hasAccepted\)/,
    "an existing publication always sends candidate payload writes to the opposite bank");
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn candidateAuthorityBase\(\)->u32\{return control\.activeBank\*p\.authorityWords;\}[\s\S]*fn acceptedAuthorityBase\(\)->u32\{return \(1u-control\.activeBank\)\*p\.authorityWords;\}/);
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn section63BankBase\(\)->u32\{return control\.activeBank[\s\S]*fn rowBankBase\(\)->u32\{return control\.activeBank[\s\S]*fn worksetBankBase\(\)->u32\{return control\.activeBank/);
  assert.match(directStructuredVelocityPublicationWGSL,
    /if\(clean&&cls<4u\)\{let dispatch=6u\+3u\*cls;publicationDispatch\[dispatch\]=groups/,
    "a rejected candidate must preserve the accepted class-dispatch records");
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn acceptStructuredPublication\(\)\{if\(atomicLoad\(&control\.flags\)!=[^{]+\{return;\}[\s\S]*acceptedControl/,
    "candidate validation is the sole gate allowed to mutate accepted control");
});

test("changed topology faces remain pending for old-field transfer while exact carries are marked", () => {
  assert.match(directStructuredVelocityPublicationWGSL,
    /fn carryValue\([^)]*\)->vec2f[\s\S]*return vec2f\(bitcast<f32>\([\s\S]*?valuesOffset[\s\S]*?\),1\.0\);/,
    "an exact old face identity must carry its value and publish the transfer-skip marker");
  assert.match(directStructuredVelocityPublicationWGSL,
    /centroidOffset\+4u\*handle\+3u\]=bitcast<u32>\(carried\.y\)/,
    "the otherwise-unused centroid lane owns the candidate-only carry marker");
  assert.doesNotMatch(directStructuredVelocityPublicationWGSL,
    /fn carryValue\([^)]*\)->f32/,
    "zero must no longer be indistinguishable from an exactly carried physical zero");
});

test("Dawn Metal compiles every direct structured publication stage", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const shaderModule = device.createShaderModule({ code: directStructuredVelocityPublicationWGSL });
  const errors = (await shaderModule.getCompilationInfo()).messages.filter((message) => message.type === "error");
  assert.deepEqual(errors, []);
  device.pushErrorScope("validation");
  for (const entryPoint of ["beginStructuredPublication", "classifyStructuredCatalogSlots", "prefixStructuredFamilies",
    "scatterStructuredFamilySlots", "publishSection63Rows", "finalizeStructuredPublication",
    "reconstructStructuredCellVelocity", "acceptStructuredPublication"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const error = await device.popErrorScope();
  assert.equal(error, null, error?.message);
  device.destroy();
});

test("rejected packed A/B publication preserves every accepted byte", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU lifecycle validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: {
    maxStorageBuffersPerShaderStage: Math.min(10, adapter.limits.maxStorageBuffersPerShaderStage),
  } });
  const catalogBytes = readFileSync(new URL("../lib/generated/octree-power-catalog.bin", import.meta.url));
  const catalog = decodeGeneratedOctreePowerCatalog(catalogBytes.buffer.slice(
    catalogBytes.byteOffset, catalogBytes.byteOffset + catalogBytes.byteLength));
  const topology = new WebGPUOctreePowerTopology(device, 1, catalog);
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const headerWords = new Uint32Array(12); headerWords[3] = 1;
  const headers = device.createBuffer({ size: 48, usage: storage });
  device.queue.writeBuffer(headers, 0, headerWords.buffer);
  const metricWords = new Uint32Array(4), metricFloats = new Float32Array(metricWords.buffer);
  metricWords[0] = 0; metricWords[1] = 0x8000_0000; metricFloats[2] = catalog.entryVolumes[0]!;
  device.queue.writeBuffer(topology.metrics, 0, metricWords.buffer);
  const deltaWords = new Uint32Array(20); deltaWords[0] = 1; deltaWords[7] = 1; deltaWords[16] = 0;
  const delta = device.createBuffer({ size: deltaWords.byteLength, usage: storage });
  device.queue.writeBuffer(delta, 0, deltaWords.buffer);
  const publication = new WebGPUDirectStructuredVelocityAuthority(device, {
    leafHeaders: headers, topology: topology.source,
    rowDelta: { rows: delta, rowCapacity: 1, controlOffsetWords: 0,
      newToOldOffsetWords: 16, oldToNewOffsetWords: 17, dirtyRowsOffsetWords: 18,
      affectedRowsOffsetWords: 19 },
    dimensions: [4, 4, 4], physicalCellSize: 1, closedBoundaryMask: 0,
  });
  const execute = async (epoch: number, failure = 0) => {
    const encoder = device.createCommandEncoder(); const broker = new PassBroker(encoder);
    publication.encode(broker, epoch, failure); device.queue.submit([broker.finish()]);
    await device.queue.onSubmittedWorkDone();
  };
  const read = async (buffer: GPUBuffer, offset: number, size: number) => {
    const staging = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, offset, staging, 0, size);
    device.queue.submit([encoder.finish()]); await device.queue.onSubmittedWorkDone();
    await staging.mapAsync(GPUMapMode.READ); const bytes = new Uint8Array(staging.getMappedRange()).slice();
    staging.unmap(); staging.destroy(); return bytes;
  };
  await execute(1);
  const source = publication.source;
  const acceptedControl = await read(source.control, 0, 128);
  const controlWords = new Uint32Array(acceptedControl.buffer);
  assert.deepEqual([...controlWords.slice(0, 6)], [0, 0xffff_ffff, 1, 1, 0, 6]);
  const bank = controlWords[4]!;
  const snapshots = await Promise.all([
    read(source.authority, bank * source.authorityBankStrideWords * 4, source.authorityBankStrideWords * 4),
    read(source.section63.coefficients, bank * source.section63.coefficientBankStrideWords * 4,
      source.section63.coefficientBankStrideWords * 4),
    read(source.familyWorksets.regularInterior.buffer, bank * source.worksetBankStrideWords * 4,
      source.worksetBankStrideWords * 4),
    read(source.rowVelocities, bank * source.rowBankStrideWords * 16, source.rowBankStrideWords * 16),
    read(source.rowGeometry, bank * source.rowBankStrideWords * 16, source.rowBankStrideWords * 16),
    read(source.section63.classDispatch, source.section63.classDispatchOffsetBytes, 48),
  ]);
  deltaWords[7] = 2; deltaWords[16] = 1; device.queue.writeBuffer(delta, 0, deltaWords.buffer);
  await execute(2, OCTREE_STRUCTURED_GPU_ERROR.geometry);
  assert.deepEqual(await read(source.control, 0, 128), acceptedControl);
  const after = await Promise.all([
    read(source.authority, bank * source.authorityBankStrideWords * 4, source.authorityBankStrideWords * 4),
    read(source.section63.coefficients, bank * source.section63.coefficientBankStrideWords * 4,
      source.section63.coefficientBankStrideWords * 4),
    read(source.familyWorksets.regularInterior.buffer, bank * source.worksetBankStrideWords * 4,
      source.worksetBankStrideWords * 4),
    read(source.rowVelocities, bank * source.rowBankStrideWords * 16, source.rowBankStrideWords * 16),
    read(source.rowGeometry, bank * source.rowBankStrideWords * 16, source.rowBankStrideWords * 16),
    read(source.section63.classDispatch, source.section63.classDispatchOffsetBytes, 48),
  ]);
  after.forEach((bytes, index) => assert.deepEqual(bytes, snapshots[index]));
  publication.destroy(); topology.destroy(); headers.destroy(); delta.destroy(); device.destroy();
});
