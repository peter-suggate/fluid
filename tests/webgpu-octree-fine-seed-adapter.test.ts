import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import type { OctreePowerVelocitySource } from "../lib/webgpu-octree-power-velocity";
import {
  OCTREE_FINE_SEED_ADAPTER_PUBLICATION,
  WebGPUOctreeFineSeedAdapter,
  octreeFineSeedCandidateShader,
  octreeFineSeedAdapterShader,
  planOctreeFineSeedAdapter,
} from "../lib/webgpu-octree-fine-seed-adapter";
import { OCTREE_FINE_SEED_STATE } from "../lib/octree-fine-seed-leaves";
import { PassBroker } from "../lib/webgpu-pass-broker";

test("fine-seed adapter allocation follows compact rows, not domain depth", () => {
  const plan = planOctreeFineSeedAdapter(100);
  assert.deepEqual(plan, {
    rowCapacity: 100,
    leafBytes: 6_400,
    candidateBytes: 800,
    allocatedBytes: 13_828,
  });
  assert.equal(planOctreeFineSeedAdapter(100).allocatedBytes, plan.allocatedBytes);
  assert.throws(() => planOctreeFineSeedAdapter(0), /must be positive/);
});

test("adapter shader publishes the FineSeedLeaf and indirect candidate ABIs", () => {
  assert.match(octreeFineSeedAdapterShader, /struct FineSeedLeaf \{ originX:u32,originY:u32,originZ:u32,size:u32,flags:u32/);
  assert.match(octreeFineSeedAdapterShader, /row<rowControl\[0\]/,
    "surface indexing must consume the compact pressure row count directly");
  assert.match(octreeFineSeedAdapterShader, /fn powerVelocityPublicationValid/);
  assert.match(octreeFineSeedAdapterShader, /let motion=powerVelocities\[row\]\.xyz/,
    "leaf motion must consume native power-cell velocity without Cartesian reconstruction");
  assert.match(octreeFineSeedAdapterShader,
    /powerVelocityControl\[0\]==POWER_VELOCITY_VALID[\s\S]*powerVelocityControl\[5\]==powerVelocityControl\[2\]/,
    "missing, failed, or partial native velocity publications must fail closed");
  assert.doesNotMatch(octreeFineSeedAdapterShader,
    /FaceRecord|faceControl|incidence|sampleMotionComponent|MAX_FACE_CANDIDATES/,
    "the Cartesian face mirror and bounded-incidence reconstruction are deleted");
  assert.match(octreeFineSeedCandidateShader, /Candidate\(row,selectedFlags\)/);
  assert.match(octreeFineSeedCandidateShader, /compaction\[params\.change\.z\]==1u/,
    "surface deltas require the exact upstream row-topology reuse publication");
  assert.match(octreeFineSeedCandidateShader, /compaction\[params\.change\.w\+index\]==frontier\[3\]/,
    "surface deltas consume current-generation interface-membership dirty stamps");
  assert.match(octreeFineSeedCandidateShader, /generation!=0u&&error==0u/,
    "publication validity is independent of a possibly-zero candidate count");
  assert.match(octreeFineSeedCandidateShader, /rowControl\[pressureControl\]/,
    "pressure/topology overflow must reject the candidate generation");
  assert.match(octreeFineSeedAdapterShader, /let flags=LIVE\|candidateFlags/,
    "all live leaves must remain directory-addressable even when they have no fine page");
  assert.match(octreeFineSeedAdapterShader, /struct CoarsePhi \{ phi:f32, minimumPhi:f32, maximumPhi:f32, flags:u32 \}/);
  assert.match(octreeFineSeedAdapterShader, /fn coarsePublicationValid\(\)/,
    "fine-seed selection must consume the compact coarse publication directly");
  assert.match(octreeFineSeedAdapterShader, /coarseControl\[11\]==0x80000000u/,
    "only a committed compact coarse publication may classify recurring leaves");
  assert.doesNotMatch(octreeFineSeedAdapterShader, /airCellKey|surfaceDirectoryRow|pageArena/,
    "the deleted page and air-alias directories must not survive in the adapter");
  assert.match(octreeFineSeedCandidateShader,
    /@compute @workgroup_size\(256\) fn publishFineSeedCandidates[\s\S]*candidateCounts\[lid\]=localCount[\s\S]*for\(var stride=1u;stride<256u;stride<<=1u\)[\s\S]*compactCandidates\[cursor\]=candidate/,
    "one bounded workgroup must count, prefix, and scatter the exact selected-row stream");
  assert.deepEqual([...octreeFineSeedCandidateShader.matchAll(
    /@compute\s+@workgroup_size\([^)]*\)\s*fn\s+([A-Za-z_]\w*)/g,
  )].map((match) => match[1]), ["publishFineSeedCandidates"],
  "the five-stage candidate compaction and snapshot pipeline must stay deleted");
  assert.doesNotMatch(octreeFineSeedCandidateShader, /\branks\b|candidateRecords|rankFineSeed|prefixFineSeed|scatterFineSeed|finalizeFineSeed|snapshotDirty/,
    "the retired full-row candidate and rank arenas have no backing WGSL");
  assert.doesNotMatch(octreeFineSeedCandidateShader, /for\(var row=0u;row<capacity;row\+=1u\)/,
    "candidate publication must not retain a serial capacity loop");
  assert.doesNotMatch(octreeFineSeedCandidateShader,
    /atomic(?:Load|Store|Add|Or|Min|Max|CompareExchange)|atomic<u32>/,
    "fine-seed candidate selection is deterministic and recurring-atomic-free");
  assert.doesNotMatch(WebGPUOctreeFineSeedAdapter.prototype.encode.toString(), /candidateTemplate/,
    "the deleted atomic counter no longer needs a copied reset template");
  const encode = WebGPUOctreeFineSeedAdapter.prototype.encode.toString();
  assert.doesNotMatch(encode, /copyBufferToBuffer/,
    "recurring fine-seed adaptation must not copy the full leaf-capacity arena");
  assert.equal((encode.match(/dispatchWorkgroups/g) ?? []).length, 2,
    "leaf rebuild plus persistent candidate publication are the only recurring adapter dispatches");
  assert.doesNotMatch(encode,
    /selectPipeline|rankCandidatesPipeline|prefixCandidateBlocksPipeline|scatterCandidatesPipeline|finalizeCandidatesPipeline|snapshotDirtyLeavesPipeline/,
    "the old host-staged candidate graph must stay deleted");
  assert.match(octreeFineSeedCandidateShader,
    /candidateControl\[5\]=publicationValid[\s\S]*workgroupUniformLoad\(&publicationValid\)[\s\S]*if\(valid!=0u\)[\s\S]*previousFineSeedLeaves\[row\]=fineSeedLeaves\[row\]/,
    "only the validated compact dirty-row publication may update previous leaves in the same transaction");
  assert.doesNotMatch(octreeFineSeedCandidateShader,
    /for\s*\([^)]*arrayLength\(&fineSeedLeaves\)/,
    "the snapshot must not retain a full-capacity leaf loop");
});

test("fine-seed candidates cut over from analytic t=0 to compact coarse phi", () => {
  assert.match(octreeFineSeedAdapterShader, /let coarse=coarseRowValid\(row\)/);
  assert.match(octreeFineSeedAdapterShader, /if\(!coarse&&params\.selection\.z==0u\)\{return;\}/,
    "a missing recurring coarse publication preserves the prior compact leaf generation");
  assert.match(octreeFineSeedAdapterShader,
    /if\(coarse\)\{let sample=coarsePhi\[row\];centrePhi=sample\.phi;minimumPhi=sample\.minimumPhi;maximumPhi=sample\.maximumPhi;\}else\{let sampleCell=[\s\S]*analyticInitialPhi/,
    "analytic phi is a cold-start seed only; recurring classification comes from compact coarse phi");
  assert.doesNotMatch(octreeFineSeedAdapterShader,
    /pagedPhiAvailable|previousPhi|textureLoad|surfacePagePhi|pageArena/,
    "the adapter has no page, dense-texture, or synthetic page fallback");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;

function bytes(data: ArrayBufferView): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(data.byteLength);
  result.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return result;
}

test("Dawn adapts live compact rows into global-fine seed candidates without dense allocations", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU fine-seed-adapter checks",
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
  const coarseData = new ArrayBuffer(32);
  const coarseWords = new Uint32Array(coarseData);
  const coarseFloats = new Float32Array(coarseData);
  coarseFloats.set([-2, -2, -1], 0); coarseWords[3] = 9;
  coarseFloats.set([0.5, -0.25, 1], 4); coarseWords[7] = 9;
  const coarsePhi = make(new Uint8Array(coarseData));
  const coarseControlWords = new Uint32Array(12);
  coarseControlWords[2] = 2;
  coarseControlWords[10] = 1;
  coarseControlWords[11] = 0x80000000;
  const coarseControl = make(coarseControlWords);
  const powerVelocityControlWords = new Uint32Array(8);
  powerVelocityControlWords.set([0x80000000, 0xffffffff, 2, 1, 2, 2, 0, 1]);
  const powerVelocityControl = make(powerVelocityControlWords);
  const powerVelocities = make(new Float32Array([3, 0, 0, 1, 3, 0, 0, 1]));
  const powerVelocitySource: OctreePowerVelocitySource = {
    plan: { rowCapacity: 2, velocityBytes: 32, statusBytes: 8, allocatedBytes: 104 },
    control: powerVelocityControl, velocities: powerVelocities,
  };
  device.pushErrorScope("validation"); device.pushErrorScope("internal");
  const surfaceAdapter = new WebGPUOctreeFineSeedAdapter(device, {
    leafHeaders, rowCount, publicationControl, frontier,
    dimensions: [4, 1, 1], cellSize: [1, 1, 1],
  }, 2, { finestLeafSize: 1, haloCells: 3 });
  surfaceAdapter.setPowerVelocitySource(powerVelocitySource);
  surfaceAdapter.setCoarsePhiSource({ values: coarsePhi, control: coarseControl });
  try {
    const info = await device.createShaderModule({ code: octreeFineSeedAdapterShader }).getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error"), []);
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    surfaceAdapter.encode(broker);
    broker.fence("fine-seed adapter test readback");
    const leafReadback = device.createBuffer({ size: 128, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const candidateReadback = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    encoder.copyBufferToBuffer(surfaceAdapter.leaves, 0, leafReadback, 0, 128);
    encoder.copyBufferToBuffer(surfaceAdapter.countAndDispatch, 0, candidateReadback, 0, 32);
    encoder.copyBufferToBuffer(surfaceAdapter.candidates, 0, candidateReadback, 32, 16);
    device.queue.submit([encoder.finish()]);
    await Promise.all([
      leafReadback.mapAsync(GPUMapMode.READ), candidateReadback.mapAsync(GPUMapMode.READ),
    ]);
    const leafCopy = leafReadback.getMappedRange().slice(0); leafReadback.unmap(); leafReadback.destroy();
    const candidateCopy = candidateReadback.getMappedRange().slice(0); candidateReadback.unmap(); candidateReadback.destroy();
    const leafWords = new Uint32Array(leafCopy), leafFloats = new Float32Array(leafCopy);
    const internalError = await device.popErrorScope();
    const validationError = await device.popErrorScope();
    assert.equal(internalError, null);
    assert.equal(validationError, null);
    assert.deepEqual(errors, []);
    assert.deepEqual(Array.from(new Uint32Array(candidateCopy, 0, 4)), [2, 1, 1, 1]);
    const control = new Uint32Array(candidateCopy, 0, 8);
    assert.equal(control[OCTREE_FINE_SEED_ADAPTER_PUBLICATION.generation], 1);
    assert.equal(control[OCTREE_FINE_SEED_ADAPTER_PUBLICATION.published], 1);
    assert.equal(control[OCTREE_FINE_SEED_ADAPTER_PUBLICATION.error], 0);
    const published = [
      { row: new Uint32Array(candidateCopy)[8], flags: new Uint32Array(candidateCopy)[9] },
      { row: new Uint32Array(candidateCopy)[10], flags: new Uint32Array(candidateCopy)[11] },
    ].sort((a, b) => a.row - b.row);
    assert.deepEqual(published, [
      { row: 0, flags: OCTREE_FINE_SEED_STATE.halo },
      { row: 1, flags: OCTREE_FINE_SEED_STATE.core },
    ]);
    assert.deepEqual([...leafWords.slice(0, 4)], [0, 0, 0, 1]);
    assert.deepEqual([...leafWords.slice(16, 20)], [1, 0, 0, 1]);
    assert.equal(leafFloats[12], 3); assert.equal(leafFloats[13], 0); assert.equal(leafFloats[14], 0);
    assert.equal(leafFloats[28], 3); assert.equal(leafFloats[31], 3);
    await device.queue.onSubmittedWorkDone();
  } finally {
    surfaceAdapter.destroy();
    for (const buffer of owned) buffer.destroy(); device.destroy();
  }
});
