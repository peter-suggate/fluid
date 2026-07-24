import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import {
  OCTREE_FACE_BAND_CONTROL_BYTES,
  OCTREE_FACE_BAND_ERROR,
  OCTREE_FACE_BAND_FACE_BYTES,
  OCTREE_FACE_BAND_ROW_BYTES,
  OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES,
  octreeFaceBandWGSL,
} from "../lib/webgpu-octree-face-closest-point";

const INVALID = 0xffff_ffff;
let sharedDevicePromise: Promise<GPUDevice> | undefined;

async function dawnDevice(): Promise<GPUDevice> {
  sharedDevicePromise ??= (async () => {
    const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
      create(options: string[]): GPU;
      globals: Record<string, unknown>;
    };
    Object.assign(globalThis, dawn.globals);
    const adapter = await dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]).requestAdapter();
    assert.ok(adapter);
    return adapter.requestDevice();
  })();
  return sharedDevicePromise;
}

test.after(async () => {
  if (sharedDevicePromise) (await sharedDevicePromise).destroy();
});

test("Dawn seeds a positive-centre coarse-mixed row before closest-point extension", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const seed = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "seedFaceCentroids" },
  });

  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(28);
  parameterWords.set([2, 2, 2], 0);
  parameterWords[3] = 1; // maximumLeaf
  parameterWords[4] = 2; // rowCapacity
  parameterWords[5] = 1; // faceCapacity
  parameterWords[6] = 1; // rowDirectoryCapacity
  parameterWords[9] = 1; // powerRowCapacity
  parameterWords[14] = 1; // axisStride
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const rowBytes = new ArrayBuffer(OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowBytes);
  const rowFloats = new Float32Array(rowBytes);
  // ROW_PHI | ROW_CORE | ROW_COARSE_MIXED, deliberately without
  // ROW_CENTER_WET. The current coarse interval still contains liquid.
  rowWords.set([0, 0, 0x205, 1], 0);
  rowFloats.set([0.2, -0.01, 0.2, 0], 4);
  const rows = storage(rowBytes);
  // Exact (cell,size) directory entry followed by the one-row count word.
  const rowDirectory = storage(new Uint32Array([0, 0, 1, 0]).buffer);
  const powerRowVelocities = storage(new Float32Array([1.25, -2.5, 0.75, 1]).buffer);

  const faceBytes = new ArrayBuffer(OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(faceBytes);
  const faceFloats = new Float32Array(faceBytes);
  faceWords.set([0, INVALID, 0, 7], 0);
  faceFloats.set([0, 0, 0, 0], 4);
  faceFloats.set([0.5, 0.5, 0.5, 1], 8);
  faceFloats[12] = 0.2;
  faceFloats[13] = 1;
  faceWords[14] = 0x45; // LIVE | PHI_VALID | VELOCITY_TARGET
  const faces = storage(faceBytes, GPUBufferUsage.COPY_SRC);

  const metrics = storage(new Uint32Array([0, 0x8000_0000, 0, 0]).buffer);
  const tetraHeaders = storage(new Uint32Array([0, 0, 1]).buffer);
  const tetrahedra = storage(new Uint32Array(1).buffer);
  const tetraVertices = storage(new Float32Array(4).buffer);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(seed);
  pass.setBindGroup(0, device.createBindGroup({
    layout: seed.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 7, resource: { buffer: rowDirectory } },
      { binding: 10, resource: { buffer: powerRowVelocities } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 27, resource: { buffer: metrics } },
      { binding: 28, resource: { buffer: tetraHeaders } },
      { binding: 29, resource: { buffer: tetrahedra } },
      { binding: 30, resource: { buffer: tetraVertices } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.end();

  const faceReadback = device.createBuffer({
    size: OCTREE_FACE_BAND_FACE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, OCTREE_FACE_BAND_FACE_BYTES);
  device.queue.submit([encoder.finish()]);
  await faceReadback.mapAsync(GPUMapMode.READ);

  const mappedFace = faceReadback.getMappedRange();
  const resultWords = new Uint32Array(mappedFace);
  const resultFloats = new Float32Array(mappedFace);
  assert.equal(resultWords[14], 0x67,
    "the mixed row publishes SEED | FACE_VELOCITY_VALID without a negative centre");
  assert.equal(resultWords[15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(4, 8)), [1.25, -2.5, 0.75, 1]);

  faceReadback.unmap();
  for (const buffer of [
    params, rows, rowDirectory, powerRowVelocities, faces, metrics,
    tetraHeaders, tetrahedra, tetraVertices, faceReadback,
  ]) buffer.destroy();
});

test("Dawn reproduces the step-228 missing-CPT face rejection without the UI", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const reconstruct = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "reconstructBandRowVelocity" },
  });
  const finalize = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "finalizeFaceBandClosestPointDiagnostics" },
  });

  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(24);
  parameterWords.set([16, 16, 16], 0); // dims
  parameterWords[3] = 2; // maximumLeaf
  parameterWords[4] = 1; // rowCapacity
  parameterWords[5] = 1; // faceCapacity
  parameterWords[10] = 229; // fine generation
  parameterWords[13] = 228; // power generation
  parameterWords[14] = 1; // axisStride
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const controlWords = new Uint32Array(OCTREE_FACE_BAND_CONTROL_BYTES / 4);
  for (const word of [1, 7, 19, 20, 21, 22, 23, 28, 29, 30, 31]) controlWords[word] = INVALID;
  controlWords[2] = 1; // rowCount
  controlWords[3] = 1; // faceCount
  controlWords[4] = 1; // incidenceCount
  controlWords[5] = 228; // candidate generation
  const control = storage(controlWords.buffer, GPUBufferUsage.COPY_SRC);

  const rowBytes = new ArrayBuffer(OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowBytes);
  const rowFloats = new Float32Array(rowBytes);
  rowWords.set([1128, 2758, 1, 2], 0); // positive-air ROW_PHI support row
  rowFloats.set([0.19973, 0.19973, 0.19973, 0], 4);
  const rows = storage(rowBytes);

  const faceBytes = new ArrayBuffer(OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(faceBytes);
  const faceFloats = new Float32Array(faceBytes);
  faceWords.set([0, INVALID, 8, 3387], 0);
  faceFloats.set([9.8682, 4.4176, 1.9539, -3206], 4);
  faceFloats.set([10, 7, 5, 1], 8);
  faceFloats[12] = 0.19978;
  faceFloats[13] = 4;
  // LIVE | PHI_VALID | CLOSEST_POINT_VALID | VELOCITY_TARGET, without
  // FACE_VELOCITY_VALID. pad=4 is CPT_MISSING_VERTEX.
  faceWords[14] = 0x55;
  faceWords[15] = 4;
  const faces = storage(faceBytes, GPUBufferUsage.COPY_SRC);

  // One fixed-incidence target face. The production reconstruction must reject
  // the row before attempting its least-squares solve.
  const incidence = storage(new Uint32Array([1, 0]).buffer);
  const rowVelocities = storage(new Float32Array(4).buffer);
  const provisionalVelocities = storage(new Float32Array(4).buffer);
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[9] = 0; // support1End
  transitionWords[11] = 1; // support3NodeEnd
  transitionWords[12] = 1; // endpointEnd
  const transitionControl = storage(transitionWords.buffer);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(reconstruct);
  pass.setBindGroup(0, device.createBindGroup({
    layout: reconstruct.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 5, resource: { buffer: control } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 19, resource: { buffer: rowVelocities } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 44, resource: { buffer: provisionalVelocities } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(finalize);
  pass.setBindGroup(0, device.createBindGroup({
    layout: finalize.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 5, resource: { buffer: control } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 44, resource: { buffer: provisionalVelocities } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.end();

  const controlReadback = device.createBuffer({
    size: OCTREE_FACE_BAND_CONTROL_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const faceReadback = device.createBuffer({
    size: OCTREE_FACE_BAND_FACE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(control, 0, controlReadback, 0, OCTREE_FACE_BAND_CONTROL_BYTES);
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, OCTREE_FACE_BAND_FACE_BYTES);
  device.queue.submit([encoder.finish()]);
  await Promise.all([
    controlReadback.mapAsync(GPUMapMode.READ),
    faceReadback.mapAsync(GPUMapMode.READ),
  ]);

  const result = new Uint32Array(controlReadback.getMappedRange());
  assert.equal(result[0],
    OCTREE_FACE_BAND_ERROR.unresolved | OCTREE_FACE_BAND_ERROR.incompleteVector);
  assert.equal(result[10], 1, "the missing-CPT target remains unresolved");
  assert.equal(result[18], 1, "the target is attributed to liquid interpolation");
  assert.equal(result[22], 3387, "the physical face id is retained");
  assert.equal(result[23], 0, "the one support row is the first incomplete vector");
  assert.equal(result[27], 1, "CPT_MISSING_VERTEX is counted exactly");
  assert.equal(result[31], 0, "the first missing-CPT face slot is retained");
  assert.equal(new Float32Array(faceReadback.getMappedRange())[7], -3206,
    "failure-only velocity.w retains the exact containing-band sentinel");

  controlReadback.unmap();
  faceReadback.unmap();
  for (const buffer of [
    params, control, rows, faces, incidence, rowVelocities, provisionalVelocities,
    transitionControl, controlReadback, faceReadback,
  ]) buffer.destroy();
});

test("Dawn repairs one residual dry CPT face from its fixed incidence star in one pass", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const gatherRepairs = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commitRepairs = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const reconstruct = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "reconstructBandRowVelocity" },
  });
  const finalize = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "finalizeFaceBandClosestPointDiagnostics" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1; // maximumLeaf
  parameterWords[4] = 2; // rowCapacity
  parameterWords[5] = 5; // faceCapacity
  parameterWords[14] = 5; // fixed incidence stride
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const controlWords = new Uint32Array(OCTREE_FACE_BAND_CONTROL_BYTES / 4);
  for (const word of [1, 7, 19, 20, 21, 22, 23, 28, 29, 30, 31]) controlWords[word] = INVALID;
  controlWords[2] = 1;
  controlWords[3] = 5;
  controlWords[4] = 5;
  controlWords[6] = 0x8000_0000;
  const control = storage(controlWords.buffer, GPUBufferUsage.COPY_SRC);

  const rowBytes = new ArrayBuffer(2 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowBytes);
  const rowFloats = new Float32Array(rowBytes);
  for (let row = 0; row < 2; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    rowWords.set([row, INVALID, 1, 1], word); // positive ROW_PHI; no liquid/mixed authority
    rowFloats.set([0.2, 0.2, 0.2, 0], word + 4);
  }
  const rows = storage(rowBytes);

  const facesBytes = new ArrayBuffer(5 * OCTREE_FACE_BAND_FACE_BYTES);
  const facesWords = new Uint32Array(facesBytes);
  const facesFloats = new Float32Array(facesBytes);
  const writeFace = (slot: number, axis: number, globalFace: number,
    velocity: readonly [number, number, number, number],
    centroid: readonly [number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    facesWords.set([0, INVALID, axis, globalFace], word);
    facesFloats.set(velocity, word + 4);
    facesFloats.set([...centroid, 1], word + 8);
    facesFloats[word + 12] = phi;
    facesFloats[word + 13] = 1;
    facesWords[word + 14] = flags;
    facesWords[word + 15] = pad;
  };
  // The residual stores its physical CPT in velocity.xyz and its failed
  // containing row in w, exactly like the step-246 record.
  writeFace(0, 0, 2460, [0.5, 0.5, 0.5, -1], [0.4, 0.5, 0.5], 0.2, 0x55, 4);
  facesWords[1] = 1;
  // Every eligible non-SEED carrier has PRIMARY_EXTENSION. Ranking first
  // minimizes |phi|, then distance squared, then global face identity.
  writeFace(1, 0, 2461, [1.5, 2.5, 3.5, 1], [0.51, 0.5, 0.5], 0.15, 0x165, 0);
  writeFace(2, 1, 30, [4, 5, 6, 1], [1.5, 0.5, 0.5], 0.05, 0x165, 0);
  writeFace(3, 2, 20, [7, 8, 9, 1], [0.5, 1, 0.5], 0.05, 0x165, 0);
  writeFace(4, 2, 10, [10, 11, 12, 1], [0.5, 0, 0.5], 0.05, 0x165, 0);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  const incidence = storage(new Uint32Array([
    5, 1,
    0, 1, 2, 3, 4,
    0, 0, 0, 0, 0,
  ]).buffer);
  // Reuse the production SupportCandidate scratch ABI: two u32 words/face.
  const repairCarrierSlots = storage(new Uint32Array(10).buffer);
  const rowVelocities = storage(new Float32Array(4).buffer);
  const provisionalVelocities = storage(new Float32Array(4).buffer, GPUBufferUsage.COPY_SRC);
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[11] = 1; // support3NodeEnd
  transitionWords[12] = 1; // endpointEnd
  const transitionControl = storage(transitionWords.buffer);

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gatherRepairs);
  pass.setBindGroup(0, device.createBindGroup({
    layout: gatherRepairs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: repairCarrierSlots } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(commitRepairs);
  pass.setBindGroup(0, device.createBindGroup({
    layout: commitRepairs.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: repairCarrierSlots } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(reconstruct);
  pass.setBindGroup(0, device.createBindGroup({
    layout: reconstruct.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 5, resource: { buffer: control } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 19, resource: { buffer: rowVelocities } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 44, resource: { buffer: provisionalVelocities } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(finalize);
  pass.setBindGroup(0, device.createBindGroup({
    layout: finalize.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 5, resource: { buffer: control } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 44, resource: { buffer: provisionalVelocities } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.end();

  const controlReadback = device.createBuffer({
    size: OCTREE_FACE_BAND_CONTROL_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const faceReadback = device.createBuffer({
    size: OCTREE_FACE_BAND_FACE_BYTES,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const provisionalReadback = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(control, 0, controlReadback, 0, OCTREE_FACE_BAND_CONTROL_BYTES);
  encoder.copyBufferToBuffer(faces, 0, faceReadback, 0, OCTREE_FACE_BAND_FACE_BYTES);
  encoder.copyBufferToBuffer(provisionalVelocities, 0, provisionalReadback, 0, 16);
  device.queue.submit([encoder.finish()]);
  await Promise.all([
    controlReadback.mapAsync(GPUMapMode.READ),
    faceReadback.mapAsync(GPUMapMode.READ),
    provisionalReadback.mapAsync(GPUMapMode.READ),
  ]);

  const result = new Uint32Array(controlReadback.getMappedRange());
  assert.equal(result[0], 0, "one-pass repair must leave no unresolved/incomplete flag");
  assert.equal(result[9], 5);
  assert.equal(result[10], 0);
  assert.equal(result[18], 0);
  assert.equal(result[27], 0);
  assert.equal(result[22], INVALID);
  assert.equal(result[23], INVALID);
  const repairedBytes = faceReadback.getMappedRange();
  const repairedWords = new Uint32Array(repairedBytes);
  const repairedFloats = new Float32Array(repairedBytes);
  assert.equal(repairedWords[14], 0x75);
  assert.equal(repairedWords[15], 0);
  assert.deepEqual(Array.from(repairedFloats.slice(4, 8)), [10, 11, 12, 1],
    "|phi|, distance squared, then global face rank the immutable primary carriers");
  assert.deepEqual(Array.from(new Float32Array(provisionalReadback.getMappedRange())),
    [5.75, 5, 10.5, 1], "row reconstruction consumes the repaired face in the next dispatch");

  controlReadback.unmap();
  faceReadback.unmap();
  provisionalReadback.unmap();
  for (const buffer of [
    params, control, rows, faces, incidence, repairCarrierSlots, rowVelocities, provisionalVelocities,
    transitionControl, controlReadback, faceReadback, provisionalReadback,
  ]) buffer.destroy();
});

test("Dawn residual repair preserves monotonicity across fixed fallback waves", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const gather = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commit = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };
  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1;
  parameterWords[4] = 4;
  parameterWords[5] = 4;
  parameterWords[14] = 2;
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const rowsBytes = new ArrayBuffer(4 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowsWords = new Uint32Array(rowsBytes);
  const rowsFloats = new Float32Array(rowsBytes);
  for (let row = 0; row < 4; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    rowsWords.set([row, INVALID, 1, 1], word);
    rowsFloats.set([0.3, 0.2, 0.3, 0], word + 4);
  }
  const rows = storage(rowsBytes);

  const facesBytes = new ArrayBuffer(4 * OCTREE_FACE_BAND_FACE_BYTES);
  const words = new Uint32Array(facesBytes);
  const floats = new Float32Array(facesBytes);
  const face = (slot: number, negative: number, positive: number, globalFace: number,
    velocity: readonly [number, number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    words.set([negative, positive, 0, globalFace], word);
    floats.set(velocity, word + 4);
    floats.set([0.5 + 0.1 * slot, 0.5, 0.5, 1], word + 8);
    floats[word + 12] = phi;
    floats[word + 13] = 1;
    words[word + 14] = flags;
    words[word + 15] = pad;
  };
  face(0, 3, INVALID, 100, [1, 2, 3, 1], 0.05, 0x165, 0);
  face(1, 1, 2, 101, [0.6, 0.5, 0.5, -1], 0.2, 0x55, 4);
  face(2, 0, 1, 102, [0.7, 0.5, 0.5, -2], 0.3, 0x55, 4);
  face(3, 2, 3, 103, [4, 5, 6, 1], 0.1, 0x75, 0);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  // A reaches the immutable primary through the row 2 -> row 3 bridge. B is
  // one dependency wave farther away and may consume A only after A commits;
  // the strictly increasing .05 -> .2 -> .3 phi ordering proves outward
  // closest-face extension.
  const incidence = storage(new Uint32Array([
    1, 2, 2, 2,
    2, 0,
    1, 2,
    1, 3,
    0, 3,
  ]).buffer);
  const scratch = storage(new Uint32Array(8).buffer);
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[11] = 4; // support3NodeEnd: all bridge endpoints are S2/S3 nodes
  const transitionControl = storage(transitionWords.buffer);
  const gatherGroup = device.createBindGroup({
    layout: gather.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const commitGroup = device.createBindGroup({
    layout: commit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  // Production encodes exactly these eight fixed dependency waves.
  for (let iteration = 0; iteration < 8; iteration += 1) {
    pass.setPipeline(gather);
    pass.setBindGroup(0, gatherGroup);
    pass.dispatchWorkgroups(1);
    pass.setPipeline(commit);
    pass.setBindGroup(0, commitGroup);
    pass.dispatchWorkgroups(1);
  }
  pass.end();
  const readback = device.createBuffer({
    size: facesBytes.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, readback, 0, facesBytes.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const resultBytes = readback.getMappedRange();
  const resultWords = new Uint32Array(resultBytes);
  const resultFloats = new Float32Array(resultBytes);
  const a = OCTREE_FACE_BAND_FACE_BYTES / 4;
  const b = 2 * a;
  assert.equal(resultWords[a + 14], 0x75,
    "the first fallback becomes valid but is not marked as a primary extension");
  assert.deepEqual(Array.from(resultFloats.slice(a + 4, a + 8)), [1, 2, 3, 1]);
  assert.equal(resultWords[b + 14], 0x75,
    "the outer residual consumes a strictly inward immutable carrier");
  assert.equal(resultWords[b + 15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(b + 4, b + 8)), [4, 5, 6, 1]);

  readback.unmap();
  for (const buffer of [
    params, rows, faces, incidence, scratch, transitionControl, readback,
  ]) buffer.destroy();
});

test("Dawn repairs a positive CPT_NO_SIMPLEX face between current-dry coarse-mixed endpoints", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const gather = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commit = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };
  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1;
  parameterWords[4] = 3;
  parameterWords[5] = 4;
  parameterWords[14] = 3;
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const rowsBytes = new ArrayBuffer(3 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowsBytes);
  const rowFloats = new Float32Array(rowsBytes);
  for (let row = 0; row < 3; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    rowWords.set([row, INVALID, 0x201, 1], word); // ROW_PHI | ROW_COARSE_MIXED
    rowFloats.set([0.1, 0.05, 0.2, 0], word + 4);
  }
  const rows = storage(rowsBytes);

  const facesBytes = new ArrayBuffer(4 * OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(facesBytes);
  const faceFloats = new Float32Array(facesBytes);
  const writeFace = (slot: number, negative: number, positive: number, globalFace: number,
    velocity: readonly [number, number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    faceWords.set([negative, positive, 0, globalFace], word);
    faceFloats.set(velocity, word + 4);
    faceFloats.set([0.5 + 0.1 * slot, 0.5, 0.5, 1], word + 8);
    faceFloats[word + 12] = phi;
    faceFloats[word + 13] = 1;
    faceWords[word + 14] = flags;
    faceWords[word + 15] = pad;
  };
  // Post-primary-extension state from the residual cases. The local wet
  // interpolant found no simplex. Face 2 deliberately touches a negative
  // incident face: local interpolation gaps next to the interface may use an
  // already-valid positive endpoint carrier even though the separate
  // support-owner outward march must retain its complete-star dry proof.
  writeFace(0, 0, 1, 246, [0.5, 0.5, 0.5, -1], 0.2, 0x55, 3);
  writeFace(1, 0, INVALID, 245, [3, 4, 5, 1], 0.05, 0x165, 0);
  writeFace(2, 0, 2, 247, [0.7, 0.5, 0.5, -3], 0.3, 0x55, 3);
  writeFace(3, 2, INVALID, 248, [0, 0, 0, 0], -0.01, 0x5, 0);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  const incidence = storage(new Uint32Array([
    3, 1, 2,
    0, 1, 2,
    0, 0, 0,
    2, 3, 0,
  ]).buffer);
  const scratch = storage(new Uint32Array(8).buffer);
  // support1End remains zero: this exercises the bounded S2 full-star proof
  // used by the exact residual, rather than trusting marched row extrema.
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[11] = 3; // support3NodeEnd: row 2 is a bounded S2 opposite endpoint
  const transitionControl = storage(transitionWords.buffer);
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gather);
  pass.setBindGroup(0, device.createBindGroup({
    layout: gather.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.setPipeline(commit);
  pass.setBindGroup(0, device.createBindGroup({
    layout: commit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  }));
  pass.dispatchWorkgroups(1);
  pass.end();
  const readback = device.createBuffer({
    size: facesBytes.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, readback, 0, facesBytes.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const resultBytes = readback.getMappedRange();
  const resultWords = new Uint32Array(resultBytes);
  const resultFloats = new Float32Array(resultBytes);
  assert.equal(resultWords[14], 0x75);
  assert.equal(resultWords[15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(4, 8)), [3, 4, 5, 1]);
  const wetEndpointFace = 2 * OCTREE_FACE_BAND_FACE_BYTES / 4;
  assert.equal(resultWords[wetEndpointFace + 14], 0x75,
    "a local no-simplex gap beside the interface consumes the valid positive endpoint carrier");
  assert.equal(resultWords[wetEndpointFace + 15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(wetEndpointFace + 4, wetEndpointFace + 8)),
    [3, 4, 5, 1]);

  readback.unmap();
  for (const buffer of [
    params, rows, faces, incidence, scratch, transitionControl, readback,
  ]) buffer.destroy();
});

test("Dawn repairs positive S2 CPT_SUPPORT_OWNER but rejects CPT_NO_OWNER", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const gather = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commit = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1;
  parameterWords[4] = 3;
  parameterWords[5] = 3;
  parameterWords[14] = 3;
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const rowsBytes = new ArrayBuffer(3 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowsBytes);
  const rowFloats = new Float32Array(rowsBytes);
  for (let row = 0; row < 3; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    rowWords.set([row, INVALID, 0x201, 1], word); // ROW_PHI | ROW_COARSE_MIXED
    rowFloats.set([0.1, 0.05, 0.2, 0], word + 4);
  }
  const rows = storage(rowsBytes);

  const facesBytes = new ArrayBuffer(3 * OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(facesBytes);
  const faceFloats = new Float32Array(facesBytes);
  const writeFace = (slot: number, negative: number, positive: number, globalFace: number,
    velocity: readonly [number, number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    faceWords.set([negative, positive, 0, globalFace], word);
    faceFloats.set(velocity, word + 4);
    faceFloats.set([0.5 + 0.1 * slot, 0.5, 0.5, 1], word + 8);
    faceFloats[word + 12] = phi;
    faceFloats[word + 13] = 1;
    faceWords[word + 14] = flags;
    faceWords[word + 15] = pad;
  };
  // Exact step-232 shape: owner lookup succeeded, but the local support-owner
  // publication did not. Both S2 endpoint stars are currently strictly dry.
  writeFace(0, 0, 1, 232, [0.5, 0.5, 0.5, -1], 0.2, 0x55, 2);
  writeFace(1, 0, INVALID, 231, [3, 4, 5, 1], 0.05, 0x165, 0);
  // The same positive-star conditions must not broaden the repair to a face
  // for which owner lookup itself failed.
  writeFace(2, 0, 2, 233, [0.7, 0.5, 0.5, -7], 0.3, 0x55, 1);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  const incidence = storage(new Uint32Array([
    3, 1, 1,
    0, 1, 2,
    0, 0, 0,
    2, 0, 0,
  ]).buffer);
  const scratch = storage(new Uint32Array(6).buffer);
  // support1End=0 selects the bounded S2 incident-star proof.
  const transitionControl = storage(
    new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4).buffer);

  const gatherGroup = device.createBindGroup({
    layout: gather.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const commitGroup = device.createBindGroup({
    layout: commit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gather);
  pass.setBindGroup(0, gatherGroup);
  pass.dispatchWorkgroups(1);
  pass.setPipeline(commit);
  pass.setBindGroup(0, commitGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  const readback = device.createBuffer({
    size: facesBytes.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, readback, 0, facesBytes.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const resultBytes = readback.getMappedRange();
  const resultWords = new Uint32Array(resultBytes);
  const resultFloats = new Float32Array(resultBytes);
  assert.equal(resultWords[14], 0x75);
  assert.equal(resultWords[15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(4, 8)), [3, 4, 5, 1]);
  const noOwner = 2 * OCTREE_FACE_BAND_FACE_BYTES / 4;
  assert.equal(resultWords[noOwner + 14], 0x55);
  assert.equal(resultWords[noOwner + 15], 1,
    "CPT_NO_OWNER remains outside the bounded residual repair contract");
  assert.equal(resultFloats[noOwner + 7], -7);

  readback.unmap();
  for (const buffer of [
    params, rows, faces, incidence, scratch, transitionControl, readback,
  ]) buffer.destroy();
});

test("Dawn repairs step-252 S2 residual from a bounded two-ring primary", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const gather = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commit = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1;
  parameterWords[4] = 3;
  parameterWords[5] = 4;
  parameterWords[14] = 3;
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);

  const rowsBytes = new ArrayBuffer(3 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowsBytes);
  const rowFloats = new Float32Array(rowsBytes);
  for (let row = 0; row < 3; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    rowWords.set([row, INVALID, 0x201, 1], word); // ROW_PHI | ROW_COARSE_MIXED
    rowFloats.set([0.1, 0.05, 0.3, 0], word + 4);
  }
  const rows = storage(rowsBytes);

  const facesBytes = new ArrayBuffer(4 * OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(facesBytes);
  const faceFloats = new Float32Array(facesBytes);
  const writeFace = (slot: number, negative: number, positive: number, globalFace: number,
    velocity: readonly [number, number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    faceWords.set([negative, positive, 0, globalFace], word);
    faceFloats.set(velocity, word + 4);
    faceFloats.set([0.5 + 0.1 * slot, 0.5, 0.5, 1], word + 8);
    faceFloats[word + 12] = phi;
    faceFloats[word + 13] = 1;
    faceWords[word + 14] = flags;
    faceWords[word + 15] = pad;
  };
  writeFace(0, 0, 1, 252, [0.5, 0.5, 0.5, -252], 0.2, 0x55, 3);
  // This already committed fallback is a bridge to row 2, never a carrier.
  writeFace(1, 0, 2, 251, [1, 1, 1, 1], 0.1, 0x75, 0);
  // The only direct immutable primary fails the strict primary phi ordering.
  writeFace(2, 1, INVALID, 250, [4, 5, 6, 1], 0.25, 0x165, 0);
  // The eligible immutable primary is on the bridge's opposite endpoint star.
  writeFace(3, 2, INVALID, 249, [7, 8, 9, 1], 0.05, 0x165, 0);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  const incidence = storage(new Uint32Array([
    2, 2, 2,
    0, 1, 0,
    0, 2, 0,
    1, 3, 0,
  ]).buffer);
  const scratch = storage(new Uint32Array(8).buffer);
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[11] = 3; // support3NodeEnd: row 2 is a bounded S2 opposite endpoint
  const transitionControl = storage(transitionWords.buffer);

  const gatherGroup = device.createBindGroup({
    layout: gather.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const commitGroup = device.createBindGroup({
    layout: commit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gather);
  pass.setBindGroup(0, gatherGroup);
  pass.dispatchWorkgroups(1);
  pass.setPipeline(commit);
  pass.setBindGroup(0, commitGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  const readback = device.createBuffer({
    size: facesBytes.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, readback, 0, facesBytes.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const resultBytes = readback.getMappedRange();
  const resultWords = new Uint32Array(resultBytes);
  const resultFloats = new Float32Array(resultBytes);
  assert.equal(resultWords[14], 0x75);
  assert.equal(resultWords[15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(4, 8)), [7, 8, 9, 1],
    "one bounded gather reaches the immutable lower-phi two-ring primary");

  readback.unmap();
  for (const buffer of [
    params, rows, faces, incidence, scratch, transitionControl, readback,
  ]) buffer.destroy();
});

test("Dawn repairs step-257 through its terminal anchor and rejects invalid paths", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for the Dawn regression",
}, async () => {
  const device = await dawnDevice();
  const module = device.createShaderModule({ code: octreeFaceBandWGSL });
  const errors = (await module.getCompilationInfo()).messages.filter(message => message.type === "error");
  assert.deepEqual(errors, []);
  const gather = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "gatherDryFaceClosestPointRepairs" },
  });
  const commit = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "commitDryFaceClosestPointRepairs" },
  });
  const storage = (bytes: ArrayBuffer, extraUsage = 0) => {
    const buffer = device.createBuffer({
      size: Math.max(4, bytes.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | extraUsage,
    });
    device.queue.writeBuffer(buffer, 0, bytes);
    return buffer;
  };

  const parameterWords = new Uint32Array(24);
  parameterWords.set([4, 4, 4], 0);
  parameterWords[3] = 1;
  parameterWords[4] = 8;
  parameterWords[5] = 11;
  parameterWords[14] = 4;
  const params = device.createBuffer({
    size: parameterWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(params, 0, parameterWords);
  const fineWords = new Uint32Array(20);
  new Float32Array(fineWords.buffer)[11] = 1; // fineWidth
  fineWords[18] = 1; // fineFactor: coarseWidth = 1
  const fineParams = device.createBuffer({
    size: fineWords.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(fineParams, 0, fineWords);

  const rowsBytes = new ArrayBuffer(8 * OCTREE_FACE_BAND_ROW_BYTES);
  const rowWords = new Uint32Array(rowsBytes);
  const rowFloats = new Float32Array(rowsBytes);
  for (let row = 0; row < 8; row += 1) {
    const word = row * OCTREE_FACE_BAND_ROW_BYTES / 4;
    const flags = row >= 5 ? 0x41 : (row < 2 ? 0x201 : 0x1);
    rowWords.set([row, INVALID, flags, 1], word);
    rowFloats.set([0.1, 0.05, 0.3, 0], word + 4);
  }
  const rows = storage(rowsBytes);

  const facesBytes = new ArrayBuffer(11 * OCTREE_FACE_BAND_FACE_BYTES);
  const faceWords = new Uint32Array(facesBytes);
  const faceFloats = new Float32Array(facesBytes);
  const writeFace = (slot: number, negative: number, positive: number, globalFace: number,
    velocity: readonly [number, number, number, number], phi: number, flags: number, pad: number) => {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    faceWords.set([negative, positive, 0, globalFace], word);
    faceFloats.set(velocity, word + 4);
    faceFloats.set([0.5 + 0.05 * slot, 0.5, 0.5, 1], word + 8);
    faceFloats[word + 12] = phi;
    faceFloats[word + 13] = 1;
    faceWords[word + 14] = flags;
    faceWords[word + 15] = pad;
  };
  // w=-(anchor+1) names terminal support endpoint row 5.
  writeFace(0, 0, 1, 257, [0.5, 0.5, 0.5, -6], 0.2, 0x55, 3);
  // Terminal topology faces are not physical velocity targets.
  writeFace(1, 5, 2, 256, [0, 0, 0, 0], 0.1, 0x15, 0);
  writeFace(2, 2, INVALID, 255, [7, 8, 9, 1], 0.05, 0x165, 0);
  // A nonintegral anchor sentinel is never rounded into a row identity.
  writeFace(3, 0, 1, 258, [0.6, 0.5, 0.5, -6.5], 0.2, 0x55, 3);
  // Row 6 is a valid terminal anchor, but its bridge is not below the target.
  writeFace(4, 0, 1, 259, [0.7, 0.5, 0.5, -7], 0.2, 0x55, 3);
  writeFace(5, 6, 3, 260, [0, 0, 0, 0], 0.25, 0x15, 0);
  writeFace(6, 3, INVALID, 261, [10, 11, 12, 1], 0.05, 0x165, 0);
  // Canonical Section 5 path: positive target -> positive terminal bridge ->
  // immutable interface seed. The closer-phi fallback must never compete.
  writeFace(7, 0, 1, 262, [0.8, 0.5, 0.5, -8], 0.2, 0x55, 3);
  writeFace(8, 7, 4, 263, [0, 0, 0, 0], 0.1, 0x15, 0);
  writeFace(9, 4, INVALID, 264, [20, 21, 22, 1], -0.02, 0x67, 0);
  writeFace(10, 4, INVALID, 265, [30, 31, 32, 1], -0.005, 0x75, 0);
  faceFloats.set([0.5, 0.5, 0.5, 1], 8 * OCTREE_FACE_BAND_FACE_BYTES / 4 + 8);
  faceFloats.set([0.75, 0.5, 0.5, 1], 9 * OCTREE_FACE_BAND_FACE_BYTES / 4 + 8);
  faceFloats.set([0.55, 0.5, 0.5, 1], 10 * OCTREE_FACE_BAND_FACE_BYTES / 4 + 8);
  const faces = storage(facesBytes, GPUBufferUsage.COPY_SRC);
  const incidence = storage(new Uint32Array([
    4, 4, 2, 2, 3, 1, 1, 1,
    0, 3, 4, 7,
    0, 3, 4, 7,
    1, 2, 0, 0,
    5, 6, 0, 0,
    8, 9, 10, 0,
    1, 0, 0, 0,
    5, 0, 0, 0,
    8, 0, 0, 0,
  ]).buffer);
  const scratch = storage(new Uint32Array(22).buffer);
  const transitionWords = new Uint32Array(OCTREE_FACE_BAND_TRANSITION_CONTROL_BYTES / 4);
  transitionWords[9] = 2; // support1End: row 2 is an S2 carrier row
  transitionWords[11] = 5; // support3NodeEnd
  transitionWords[12] = 8; // endpointEnd
  const transitionControl = storage(transitionWords.buffer);

  const gatherGroup = device.createBindGroup({
    layout: gather.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 1, resource: { buffer: fineParams } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const commitGroup = device.createBindGroup({
    layout: commit.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: params } },
      { binding: 6, resource: { buffer: rows } },
      { binding: 12, resource: { buffer: faces } },
      { binding: 14, resource: { buffer: incidence } },
      { binding: 32, resource: { buffer: transitionControl } },
      { binding: 67, resource: { buffer: scratch } },
    ],
  });
  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(gather);
  pass.setBindGroup(0, gatherGroup);
  pass.dispatchWorkgroups(1);
  pass.setPipeline(commit);
  pass.setBindGroup(0, commitGroup);
  pass.dispatchWorkgroups(1);
  pass.end();
  const readback = device.createBuffer({
    size: facesBytes.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  encoder.copyBufferToBuffer(faces, 0, readback, 0, facesBytes.byteLength);
  device.queue.submit([encoder.finish()]);
  await readback.mapAsync(GPUMapMode.READ);
  const resultBytes = readback.getMappedRange();
  const resultWords = new Uint32Array(resultBytes);
  const resultFloats = new Float32Array(resultBytes);
  assert.equal(resultWords[14], 0x75);
  assert.equal(resultWords[15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(4, 8)), [7, 8, 9, 1]);
  for (const [slot, sentinel] of [[3, -6.5], [4, -7]] as const) {
    const word = slot * OCTREE_FACE_BAND_FACE_BYTES / 4;
    assert.equal(resultWords[word + 14], 0x55);
    assert.equal(resultWords[word + 15], 3);
    assert.equal(resultFloats[word + 7], sentinel);
  }
  const seedTarget = 7 * OCTREE_FACE_BAND_FACE_BYTES / 4;
  assert.equal(resultWords[seedTarget + 14], 0x75);
  assert.equal(resultWords[seedTarget + 15], 0);
  assert.deepEqual(Array.from(resultFloats.slice(seedTarget + 4, seedTarget + 8)),
    [20, 21, 22, 1],
    "the immutable interface SEED wins over a closer-phi non-immutable fallback");

  readback.unmap();
  for (const buffer of [
    params, fineParams, rows, faces, incidence, scratch, transitionControl, readback,
  ]) buffer.destroy();
});
