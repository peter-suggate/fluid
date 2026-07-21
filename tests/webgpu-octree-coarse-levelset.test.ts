import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_COARSE_PHI_CONTROL_BYTES,
  OCTREE_FINE_PHI_CONTRIBUTION_BYTES,
  WebGPUOctreeCoarseLevelSet,
  octreeCoarsePhiCorrectionShader,
  packOctreeFinePhiContributions,
  planOctreeCoarsePhi,
  unpackOctreeCoarsePhiGPUControl,
} from "../lib/webgpu-octree-coarse-levelset";
import { OCTREE_SURFACE_STATE } from "../lib/webgpu-octree-surface-pages";
import { correctCoarsePhiFromFine } from "../lib/octree-coarse-levelset";

test("coarse phi GPU allocation scales only with live row capacity", () => {
  const small = planOctreeCoarsePhi(16);
  const large = planOctreeCoarsePhi(32);
  assert.equal(large.recordBytes, small.recordBytes * 2);
  assert.equal(large.scratchBytes, small.scratchBytes * 2);
});

test("fine correction contributions use a stable 16-byte ABI", () => {
  const packed = packOctreeFinePhiContributions([{ phi: -0.25, distanceSquared: 2 }]);
  assert.equal(packed.byteLength, OCTREE_FINE_PHI_CONTRIBUTION_BYTES);
  assert.equal(new Float32Array(packed)[0], -0.25);
  assert.equal(new Uint32Array(packed)[2], 1);
});

test("coarse correction shader is bounded CSR work and preserves an interval", () => {
  assert.match(octreeCoarsePhiCorrectionShader, /end-begin>params\.maximumPerRow/);
  assert.match(octreeCoarsePhiCorrectionShader, /output\.minimumPhi=minimumPhi/);
  assert.match(octreeCoarsePhiCorrectionShader, /output\.maximumPhi=maximumPhi/);
  assert.match(octreeCoarsePhiCorrectionShader, /sample\.distanceSquared<nearestDistance/);
});

test("coarse bootstrap consumes the shared adapter live-row bit", () => {
  assert.equal(OCTREE_SURFACE_STATE.live, 1 << 5);
  assert.match(octreeCoarsePhiCorrectionShader,
    new RegExp(`leaf\\.flags&${OCTREE_SURFACE_STATE.live}u`));
  assert.doesNotMatch(octreeCoarsePhiCorrectionShader, /leaf\.flags&1u/,
    "surface-page residency is not the SurfaceLeaf live-row contract");
});

test("Dawn bootstraps adapter-style live rows and corrects compact coarse phi without a dense field", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for coarse-phi GPU checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice();
  const compilation = await device.createShaderModule({ code: octreeCoarsePhiCorrectionShader }).getCompilationInfo();
  assert.deepEqual(compilation.messages.filter((message) => message.type === "error"), []);
  const coarse = new WebGPUOctreeCoarseLevelSet(device, 2);
  const leafData = new ArrayBuffer(2 * 48), leafWords = new Uint32Array(leafData), leafFloats = new Float32Array(leafData);
  for (let row = 0; row < 2; row += 1) {
    const base = row * 12;
    leafWords[base] = row * 8;
    leafWords[base + 1] = 8;
    leafWords[base + 2] = OCTREE_SURFACE_STATE.live;
    leafFloats[base + 4] = row === 0 ? -2 : 3;
  }
  assert.equal(leafWords[2], OCTREE_SURFACE_STATE.live); assert.equal(leafWords[14], OCTREE_SURFACE_STATE.live);
  const leaves = device.createBuffer({ size: leafData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true });
  new Uint8Array(leaves.getMappedRange()).set(new Uint8Array(leafData)); leaves.unmap();
  const inputReadback = device.createBuffer({ size: leafData.byteLength,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.pushErrorScope("internal"); device.pushErrorScope("validation");
  const inputEncoder = device.createCommandEncoder();
  inputEncoder.copyBufferToBuffer(leaves, 0, inputReadback, 0, leafData.byteLength);
  device.queue.submit([inputEncoder.finish()]); await device.queue.onSubmittedWorkDone();
  const inputError = await device.popErrorScope(); const inputInternal = await device.popErrorScope();
  assert.equal(inputError, null, inputError?.message); assert.equal(inputInternal, null, inputInternal?.message);
  await inputReadback.mapAsync(GPUMapMode.READ);
  const copiedLeaves = new Uint32Array(inputReadback.getMappedRange().slice(0)); inputReadback.unmap();
  assert.equal(copiedLeaves[2], OCTREE_SURFACE_STATE.live); assert.equal(copiedLeaves[14], OCTREE_SURFACE_STATE.live);
  const bootstrapReadback = device.createBuffer({ size: coarse.plan.recordBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.pushErrorScope("internal"); device.pushErrorScope("validation");
  const bootstrapEncoder = device.createCommandEncoder();
  coarse.encodeBootstrapFromSurfaceLeaves(bootstrapEncoder, leaves);
  bootstrapEncoder.copyBufferToBuffer(coarse.records, 0, bootstrapReadback, 0, coarse.plan.recordBytes);
  device.queue.submit([bootstrapEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();
  const bootstrapError = await device.popErrorScope(); const bootstrapInternal = await device.popErrorScope();
  assert.equal(bootstrapError, null, bootstrapError?.message); assert.equal(bootstrapInternal, null, bootstrapInternal?.message);
  await bootstrapReadback.mapAsync(GPUMapMode.READ);
  const bootstrapped = bootstrapReadback.getMappedRange().slice(0); bootstrapReadback.unmap();
  const bootstrapFloats = new Float32Array(bootstrapped), bootstrapWords = new Uint32Array(bootstrapped);
  assert.equal(bootstrapFloats[0], -2); assert.equal(bootstrapFloats[1], -2); assert.equal(bootstrapFloats[2], -2);
  assert.equal(bootstrapFloats[4], 3); assert.equal(bootstrapFloats[5], 3); assert.equal(bootstrapFloats[6], 3);
  assert.equal(bootstrapWords[3] & 9, 9); assert.equal(bootstrapWords[7] & 9, 9);
  coarse.upload(correctCoarsePhiFromFine([
    { row: 0, origin: [0, 0, 0], size: 8, phi: 2 },
    { row: 1, origin: [8, 0, 0], size: 8, phi: -4 },
  ], []).rows);
  const upload = (data: ArrayBufferView) => {
    const buffer = device.createBuffer({ size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const copy = new Uint8Array(data.byteLength);
    copy.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    device.queue.writeBuffer(buffer, 0, copy);
    return buffer;
  };
  const offsets = upload(new Uint32Array([0, 2, 2]));
  const contributions = upload(new Uint8Array(packOctreeFinePhiContributions([
    { phi: -0.1, distanceSquared: 0.25 }, { phi: 0.1, distanceSquared: 0.25 },
  ])));
  const readback = device.createBuffer({ size: OCTREE_COARSE_PHI_CONTROL_BYTES + coarse.plan.recordBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();
  coarse.encodeFineCorrection(encoder, { rowOffsets: offsets, contributions }, {
    rowCount: 2, contributionCount: 2, maximumContributionsPerRow: 4, generation: 5,
  });
  encoder.copyBufferToBuffer(coarse.control, 0, readback, 0, OCTREE_COARSE_PHI_CONTROL_BYTES);
  encoder.copyBufferToBuffer(coarse.records, 0, readback, OCTREE_COARSE_PHI_CONTROL_BYTES, coarse.plan.recordBytes);
  device.queue.submit([encoder.finish()]);
  await device.queue.onSubmittedWorkDone(); await readback.mapAsync(GPUMapMode.READ);
  const result = readback.getMappedRange().slice(0); readback.unmap();
  assert.deepEqual(unpackOctreeCoarsePhiGPUControl(new Uint32Array(result, 0, 8)), {
    flags: 0, firstErrorRow: 0xffff_ffff, rowCount: 2, contributionCount: 2,
    correctedRows: 1, interfaceRows: 1, generation: 5,
  });
  const values = new Float32Array(result, OCTREE_COARSE_PHI_CONTROL_BYTES, 8);
  assert.ok(Math.abs(values[0] + 0.1) < 1e-6);
  assert.ok(Math.abs(values[1] + 0.1) < 1e-6);
  assert.ok(Math.abs(values[2] - 0.1) < 1e-6);
  assert.equal(values[4], -4);
  coarse.destroy(); leaves.destroy(); inputReadback.destroy(); bootstrapReadback.destroy(); offsets.destroy(); contributions.destroy(); readback.destroy(); device.destroy();
});
