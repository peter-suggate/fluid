import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import type { OctreeFaceMirrorSource } from "../lib/webgpu-octree-face-mirror";
import {
  OCTREE_SURFACE_ADAPTER_PUBLICATION,
  WebGPUOctreeSurfaceAdapter,
  octreeSurfaceCandidateShader,
  octreeSurfaceAdapterShader,
  planOctreeSurfaceAdapter,
} from "../lib/webgpu-octree-surface-adapter";
import {
  OCTREE_SURFACE_STATE,
  WebGPUOctreeSurfacePages,
} from "../lib/webgpu-octree-surface-pages";

test("surface adapter allocation follows compact rows, not domain depth", () => {
  const plan = planOctreeSurfaceAdapter(100);
  assert.deepEqual(plan, {
    rowCapacity: 100,
    leafBytes: 6_400,
    candidateBytes: 800,
    allocatedBytes: 13_840,
  });
  assert.equal(planOctreeSurfaceAdapter(100).allocatedBytes, plan.allocatedBytes);
  assert.throws(() => planOctreeSurfaceAdapter(0), /must be positive/);
});

test("adapter shader publishes the SurfaceLeaf and indirect candidate ABIs", () => {
  assert.match(octreeSurfaceAdapterShader, /struct SurfaceLeaf \{ originX:u32,originY:u32,originZ:u32,size:u32,flags:u32/);
  assert.match(octreeSurfaceAdapterShader, /row<rowControl\[0\]/,
    "surface indexing must consume the compact pressure row count directly");
  assert.match(octreeSurfaceAdapterShader, /fn sampleMotionComponent/);
  assert.match(octreeSurfaceAdapterShader, /INCIDENCE_PER_ROW=48u/);
  assert.match(octreeSurfaceCandidateShader, /Candidate\(row,candidateFlags\)/);
  assert.match(octreeSurfaceCandidateShader, /generation!=0u&&error==0u/,
    "publication validity is independent of a possibly-zero candidate count");
  assert.match(octreeSurfaceCandidateShader, /rowControl\[pressureControl\]/,
    "pressure/topology overflow must reject the candidate generation");
  assert.match(octreeSurfaceAdapterShader, /let flags=LIVE\|candidateFlags/,
    "all live leaves must remain hash-addressable even when they have no fine page");
  assert.match(octreeSurfaceAdapterShader,
    /fn airCellKey\(p:vec3u\)->u32\{return p\.x\+dims\(\)\.x\*\(p\.y\+dims\(\)\.y\*p\.z\)\+1u;\}/,
    "previous-generation air aliases retain exact linear identities beyond coordinate 1023");
  assert.doesNotMatch(octreeSurfaceAdapterShader, /p\.x\|\(p\.y<<10u\)/,
    "previous-generation air lookup must not use a 10:10:10 key");
  assert.match(WebGPUOctreeSurfaceAdapter.toString().replace(/\s+/g, ""),
    /mappedAtCreation:true[\s\S]*rowCapacity/,
    "the immutable candidate control template carries dispatch, validity, and capacity words");
  assert.match(WebGPUOctreeSurfaceAdapter.prototype.encode.toString(), /copyBufferToBuffer\(this\.candidateTemplate/,
    "each encoded generation restores count zero and immutable dispatch xyz without a compute prepare pass");
});

test("surface candidates reuse the previous page generation after dense bootstrap", () => {
  assert.match(octreeSurfaceAdapterShader, /fn pagedPhiAvailable/);
  assert.match(octreeSurfaceAdapterShader, /\(r==2u\|\|r==4u\)/,
    "the production 2^3 page layout and the high-detail 4^3 layout must both remain sparse-authoritative");
  assert.match(octreeSurfaceAdapterShader, /pageArena\[7\]>0u/,
    "a completed sparse generation remains authoritative without resident fine pages");
  assert.match(octreeSurfaceAdapterShader, /pageArena\[6\]>0u/,
    "an empty publication must keep using analytic/dense bootstrap phi until at least one page is active");
  assert.match(octreeSurfaceAdapterShader, /fn previousRow/);
  assert.match(octreeSurfaceAdapterShader, /fn previousPhi/);
  assert.match(octreeSurfaceAdapterShader,
    /if\(!pagedPhiAvailable\(\)\)\{if\(params\.selection\.z!=0u\)\{return analyticInitialPhi[\s\S]*return textureLoad\(levelSet,q,0\)\.x;\}/,
    "analytic bootstrap bypasses the sampled fallback while imported-shape compatibility retains it");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

function bytes(data: ArrayBufferView): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(data.byteLength);
  result.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return result;
}

test("Dawn adapts live compact rows into surface pages without dense allocations", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU surface-adapter checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const gpuAdapter = await gpu.requestAdapter(); assert.ok(gpuAdapter);
  const device = await gpuAdapter.requestDevice({requiredLimits:{maxStorageBuffersPerShaderStage:gpuAdapter.limits.maxStorageBuffersPerShaderStage}}); const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const owned: GPUBuffer[] = [];
  const make = (data: ArrayBufferView, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength), usage });
    device.queue.writeBuffer(buffer, 0, bytes(data)); owned.push(buffer); return buffer;
  };
  const headers = new Uint32Array(24);
  headers.set([0, 0, 0, 1], 0);
  headers.set([1, 0, 0, 1], 12);
  const leafHeaders = make(headers);
  const rowCount = make(new Uint32Array([2]));
  const publicationWords = new Uint32Array(16); publicationWords[0] = 2;
  const publicationControl = make(publicationWords);
  const frontierWords = new Uint32Array(4); frontierWords[3] = 1;
  const frontier = make(frontierWords);
  const levelSet = device.createTexture({
    label: "Surface adapter test level set",
    size: [4, 1, 1], dimension: "3d", format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const phiUpload = new Float32Array(64);
  phiUpload.set([-2, -0.2, 0.5, 2]);
  device.queue.writeTexture(
    { texture: levelSet }, bytes(phiUpload), { bytesPerRow: 256, rowsPerImage: 1 }, [4, 1, 1],
  );
  const faceControl = make(new Uint32Array([1, 0, 1, 2, 1, 0]));
  const faceData = new ArrayBuffer(32), faceU32 = new Uint32Array(faceData), faceF32 = new Float32Array(faceData);
  faceU32.set([0, 1, 1, 0, 0, 4]); faceF32[6] = 3; faceF32[7] = 1;
  const faces = make(new Uint8Array(faceData));
  const incidenceWords = new Uint32Array(98);
  incidenceWords[0] = 1; incidenceWords[1] = 1;
  incidenceWords[2] = 0; incidenceWords[50] = 0;
  const incidence = make(incidenceWords);
  const parity = make(new Uint32Array(8));
  const faceSource: OctreeFaceMirrorSource = {
    plan: { rowCapacity: 2, faceCapacity: 1, faceBytes: 32, incidenceBytes: 392, allocatedBytes: 484 },
    control: faceControl, faces, incidence, parity,
  };
  device.pushErrorScope("validation"); device.pushErrorScope("internal");
  const surfaceAdapter = new WebGPUOctreeSurfaceAdapter(device, {
    leafHeaders, rowCount, publicationControl, frontier, levelSet,
    dimensions: [4, 1, 1], cellSize: [1, 1, 1],
  }, faceSource, 2, { finestLeafSize: 1, haloCells: 3, directPageSampling: true });
  const pages = new WebGPUOctreeSurfacePages(device, surfaceAdapter.source, 2, [4, 1, 1], [1, 1, 1], {
    maximumPages: 2, maximumResidentFraction: 1,
  });
  try {
    const info = await device.createShaderModule({ code: octreeSurfaceAdapterShader }).getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error"), []);
    const encoder = device.createCommandEncoder();
    surfaceAdapter.encode(encoder);
    pages.encodeLifecycle(encoder);
    const leafReadback = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const candidateReadback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const arenaReadback = device.createBuffer({ size: pages.plan.arenaBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(surfaceAdapter.leaves, 0, leafReadback, 0, 128);
    encoder.copyBufferToBuffer(surfaceAdapter.countAndDispatch, 0, candidateReadback, 0, 32);
    encoder.copyBufferToBuffer(surfaceAdapter.candidates, 0, candidateReadback, 32, 16);
    encoder.copyBufferToBuffer(pages.arena, 0, arenaReadback, 0, pages.plan.arenaBytes);
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      leafReadback.mapAsync(GPUMapMode.READ), candidateReadback.mapAsync(GPUMapMode.READ), arenaReadback.mapAsync(GPUMapMode.READ),
    ]);
    const leafCopy = leafReadback.getMappedRange().slice(0); leafReadback.unmap(); leafReadback.destroy();
    const candidateCopy = candidateReadback.getMappedRange().slice(0); candidateReadback.unmap(); candidateReadback.destroy();
    const arenaCopy = arenaReadback.getMappedRange().slice(0); arenaReadback.unmap(); arenaReadback.destroy();
    const leafWords = new Uint32Array(leafCopy), leafFloats = new Float32Array(leafCopy);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    assert.equal(internalError, null);
    assert.equal(validationError, null);
    assert.deepEqual(errors, []);
    assert.deepEqual(Array.from(new Uint32Array(candidateCopy, 0, 4)), [2, 1, 1, 1]);
    const control = new Uint32Array(candidateCopy, 0, 8);
    assert.equal(control[OCTREE_SURFACE_ADAPTER_PUBLICATION.generation], 1);
    assert.equal(control[OCTREE_SURFACE_ADAPTER_PUBLICATION.published], 1);
    assert.equal(control[OCTREE_SURFACE_ADAPTER_PUBLICATION.error], 0);
    const published = [
      { row: new Uint32Array(candidateCopy)[8], flags: new Uint32Array(candidateCopy)[9] },
      { row: new Uint32Array(candidateCopy)[10], flags: new Uint32Array(candidateCopy)[11] },
    ].sort((a, b) => a.row - b.row);
    assert.deepEqual(published, [
      { row: 0, flags: OCTREE_SURFACE_STATE.halo },
      { row: 1, flags: OCTREE_SURFACE_STATE.core },
    ]);
    assert.deepEqual([...leafWords.slice(0, 4)], [0, 0, 0, 1]);
    assert.deepEqual([...leafWords.slice(16, 20)], [1, 0, 0, 1]);
    assert.equal(leafFloats[12], 3); assert.equal(leafFloats[13], 0); assert.equal(leafFloats[14], 0);
    assert.equal(leafFloats[28], 3); assert.equal(leafFloats[31], 3);
    const arenaWords = new Uint32Array(arenaCopy);
    assert.equal(arenaWords[3], 0, "surface lifecycle must remain authoritative");
    assert.equal(arenaWords[6], 2, "adapter candidates must activate both pages");
    await device.queue.onSubmittedWorkDone();
  } finally {
    pages.destroy(); surfaceAdapter.destroy(); levelSet.destroy();
    for (const buffer of owned) buffer.destroy(); device.destroy();
  }
});
