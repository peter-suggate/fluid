import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import type { DirectStructuredVelocitySource } from "../lib/webgpu-octree-structured-velocity-gpu";
import {
  OCTREE_FINE_SEED_ADAPTER_PUBLICATION,
  OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY,
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
    allocatedBytes: 13_892,
  });
  assert.equal(planOctreeFineSeedAdapter(100).allocatedBytes, plan.allocatedBytes);
  assert.throws(() => planOctreeFineSeedAdapter(0), /must be positive/);
});

test("adapter shader publishes the FineSeedLeaf and indirect candidate ABIs", () => {
  assert.match(octreeFineSeedAdapterShader, /struct FineSeedLeaf \{ originX:u32,originY:u32,originZ:u32,size:u32,flags:u32/);
  assert.match(octreeFineSeedAdapterShader, /row<structuredVelocityControl\[2\]/,
    "surface indexing must consume the epoch-stable accepted structured row count");
  assert.match(octreeFineSeedAdapterShader, /fn structuredVelocityPublicationValid/);
  assert.match(octreeFineSeedAdapterShader, /let motion=structuredRowVelocities\[structuredVelocityRowIndex\(row\)\]\.xyz/,
    "leaf motion must consume the accepted structured row velocity without Cartesian reconstruction");
  assert.match(octreeFineSeedAdapterShader,
    /structuredVelocityControl\[0\]==0u[\s\S]*structuredVelocityControl\[3\]!=0u[\s\S]*structuredVelocityControl\[4\]<=1u/,
    "missing, failed, or partial structured publications must fail closed");
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
  assert.equal((octreeFineSeedCandidateShader.match(
    /candidateControl\[7\]=params\.dimsCapacity\.w/g) ?? []).length, 2,
    "the residency consumer's capacity handshake must not receive the changing live-row count");
  assert.match(octreeFineSeedCandidateShader,
    /structuredControl\[0\]==0u[\s\S]*structuredControl\[3\]!=0u[\s\S]*rawSourceRows==structuredControl\[2\]/,
    "a missing, mismatched, or rejected structured authority must reject the candidate generation");
  assert.match(octreeFineSeedAdapterShader, /let flags=LIVE\|candidateFlags/,
    "all live leaves must remain directory-addressable even when they have no fine page");
  assert.match(octreeFineSeedAdapterShader,
    /let virtualInterface=!openTop&&origin\.y\+header\.size>=dims\(\)\.y&&centrePhi<0\.0/,
    "closed-lid liquid leaves seed the nascent separating interface");
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
  )].map((match) => match[1]), [
    "publishFineSeedAdapterDispatch",
    "publishFineSeedCandidates",
  ], "one exact-work planner plus candidate publication replace the deleted five-stage pipeline");
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
  assert.equal((encode.match(/dispatchWorkgroupsIndirect/g) ?? []).length, 2,
    "the hierarchical fallback retains exact GPU-authored dispatches");
  assert.equal((encode.match(/dispatchWorkgroups\(1\)/g) ?? []).length, 3,
    "the compact path has two singleton kernels and the fallback has one exact-work planner");
  assert.match(encode,
    /rowCapacity\s*<=\s*OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY[\s\S]*dispatchWorkgroups\(1\)[\s\S]*dispatchWorkgroups\(1\)[\s\S]*return/,
    "small compact systems keep both maintenance kernels in one pass");
  assert.doesNotMatch(encode, /updateIndirectBuffer|copyBufferToBuffer/,
    "the planner must author the indirect records directly without staging copies");
  assert.match(encode,
    /fine-seed direct dispatch publication[\s\S]*dispatchWorkgroupsIndirect\(this\.dispatch,\s*0\)[\s\S]*dispatchWorkgroupsIndirect\(this\.dispatch,\s*12\)/,
    "direct STORAGE output crosses one legality boundary before both INDIRECT consumers");
  assert.doesNotMatch(encode, /this\.workgroups|dispatchWorkgroups\([^1]/,
    "no recurring launch dimension is shaped by allocated row capacity");
  assert.match(octreeFineSeedCandidateShader,
    /dispatchMetadata\[0\]=\(rows\+63u\)\/64u[\s\S]*dispatchMetadata\[3\]=select\(0u,1u,rows>0u\)/,
    "empty live-row and candidate work must publish zero x dimensions");
  assert.match(octreeFineSeedCandidateShader,
    /rawSourceRows<=params\.dimsCapacity\.w[\s\S]*sourceRows==structuredControl\[2\][\s\S]*let rows=select\(0u,sourceRows,sourceValid\)/,
    "cold bootstrap must build the complete accepted compact row stream, never a truncated interface prefix");
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

test("cold bootstrap samples the geometric centre of even-sized leaves symmetrically", () => {
  assert.match(octreeFineSeedAdapterShader,
    /let lattice=point-vec3f\(0\.5\);let base=vec3i\(floor\(lattice\)\);let t=fract\(lattice\)/,
    "the dense cell-centred field must be interpolated at non-cell-centred leaf positions");
  assert.match(octreeFineSeedAdapterShader,
    /centrePhi=bootstrapPhi\(centre\)/,
    "the affine value and the affine evaluation origin must name the same geometric point");
  assert.doesNotMatch(octreeFineSeedAdapterShader,
    /sampleCell=vec3i\(clamp\(vec3u\(centre\)/,
    "even leaf centres must not be rounded toward the positive lattice direction");
});

test("fine-seed candidates cut over from analytic t=0 to compact coarse phi", () => {
  assert.match(octreeFineSeedAdapterShader, /let coarse=coarseRowValid\(row\)/);
  assert.match(octreeFineSeedAdapterShader, /if\(!coarse&&params\.selection\.z==0u\)\{continue;\}/,
    "a missing recurring coarse publication preserves the prior compact leaf generation");
  assert.match(octreeFineSeedAdapterShader,
    /if\(coarse\)\{let sample=coarsePhi\[row\];centrePhi=sample\.phi;minimumPhi=sample\.minimumPhi;maximumPhi=sample\.maximumPhi;\}else\{centrePhi=bootstrapPhi\(centre\);gradient=[\s\S]*bootstrapPhi/,
    "bootstrap phi is a cold-start seed only; recurring classification comes from compact coarse phi");
  // The cold-start seed dispatches over both bootstrap authorities: scenes
  // with no closed-form surface read the imported dense level set instead.
  assert.match(octreeFineSeedAdapterShader,
    /fn bootstrapPhi\(point:vec3f\)->f32\{\s*if\(params\.selection\.z==3u\)\{return bootstrapTexturePhi\(point\);\}\s*return analyticInitialPhi\(point\);\}/,
    "mode 3 selects the imported dense level set; every other mode keeps the analytic form");
  assert.doesNotMatch(octreeFineSeedAdapterShader,
    /pagedPhiAvailable|previousPhi|surfacePagePhi|pageArena/,
    "the adapter has no page or synthetic page fallback");
  // The imported dense level set is a cold-start authority, never a recurring
  // fallback: it is reachable only through the mode-gated bootstrapPhi above,
  // so exactly one sample site may exist and it must live in that helper.
  assert.equal(octreeFineSeedAdapterShader.match(/textureLoad/g)?.length, 1,
    "the dense bootstrap level set has exactly one sample site");
  assert.match(octreeFineSeedAdapterShader,
    /fn bootstrapTexturePhi\(point:vec3f\)->f32\{[\s\S]*value\+=weight\*textureLoad\(/,
    "the only textureLoad belongs to the mode-gated cold-start helper");
});

test("small fine-seed maintenance halves the exact compute-pass spine", () => {
  const encode = (rowCapacity: number) => {
    const dispatches: string[] = [];
    const pass = {
      setPipeline() {}, setBindGroup() {}, end() {},
      dispatchWorkgroups() { dispatches.push("direct"); },
      dispatchWorkgroupsIndirect() { dispatches.push("indirect"); },
    } as unknown as GPUComputePassEncoder;
    const encoder = {
      beginComputePass() { return pass; },
    } as unknown as GPUCommandEncoder;
    const broker = new PassBroker(encoder, { isolateLabels: false });
    const adapter = Object.assign(Object.create(WebGPUOctreeFineSeedAdapter.prototype), {
      destroyed: false,
      plan: { rowCapacity },
      buildPipeline: {}, publishCandidatesPipeline: {}, planDispatchPipeline: {},
      bindGroup: {}, selectBindGroup: {}, plannerBindGroup: {}, dispatch: {},
    }) as WebGPUOctreeFineSeedAdapter;
    adapter.encode(broker);
    return { passes: broker.computePassCount, dispatches };
  };

  assert.deepEqual(encode(OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY), {
    passes: 1, dispatches: ["direct", "direct"],
  }, "the compact path fuses the former planner and consumer boundary");
  assert.deepEqual(encode(OCTREE_FINE_SEED_PERSISTENT_ROW_CAPACITY + 1), {
    passes: 2, dispatches: ["direct", "indirect", "indirect"],
  }, "the hierarchical path retains one required storage-to-indirect boundary");
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
  const structuredVelocityControl = make(new Uint32Array([0, 0xffffffff, 2, 1, 0, 2]));
  const structuredRowVelocities = make(new Float32Array([
    3, 0, 0, 1, 3, 0, 0, 1,
    0, 0, 0, 0, 0, 0, 0, 0,
  ]));
  const structuredVelocitySource = {
    plan: { rowCapacity: 2 }, control: structuredVelocityControl,
    rowVelocities: structuredRowVelocities,
  } as unknown as DirectStructuredVelocitySource;
  device.pushErrorScope("validation"); device.pushErrorScope("internal");
  // The seed kernel always binds a bootstrap level set; this fixture drives
  // the coarse-phi authority, so the texture is present but never sampled.
  const bootstrapLevelSet = device.createTexture({
    label: "Fine-seed adapter test bootstrap level set",
    size: [4, 1, 1], dimension: "3d", format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  const surfaceAdapter = new WebGPUOctreeFineSeedAdapter(device, {
    leafHeaders, rowCount, publicationControl, frontier,
    dimensions: [4, 1, 1], cellSize: [1, 1, 1],
  }, 2, { finestLeafSize: 1, haloCells: 3, bootstrapLevelSet });
  surfaceAdapter.setStructuredVelocitySource(structuredVelocitySource);
  surfaceAdapter.setCoarsePhiSource({ values: coarsePhi, control: coarseControl });
  try {
    const info = await device.createShaderModule({ code: octreeFineSeedAdapterShader }).getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error"), []);
    const encoder = device.createCommandEncoder();
    const broker = new PassBroker(encoder);
    surfaceAdapter.encode(broker);
    assert.equal(broker.computePassCount, 1,
      "the real compact Dawn command graph must encode one compute pass");
    assert.equal(broker.boundaryAudit.get("stage indirect args"), undefined,
      "the compact Dawn graph must issue no indirect-argument staging copy");
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
      { row: 0, flags: OCTREE_FINE_SEED_STATE.core },
      { row: 1, flags: OCTREE_FINE_SEED_STATE.core },
    ], "the negative closed-lid leaf is a virtual-interface core seed");
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
