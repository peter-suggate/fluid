import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { WebGPUOctreePhiDifferential, pagedPhiDifferentialShader } from "../lib/webgpu-octree-phi-diagnostic";
import { planOctreeSurfacePages, type OctreeSurfacePageSource } from "../lib/webgpu-octree-surface-pages";

const modulePath = process.env.WEBGPU_NODE_MODULE;

test("paged-phi diagnostic resolves full-domain air-alias identities", () => {
  assert.match(pagedPhiDifferentialShader,
    /fn airCellKey\(p:vec3u\)->u32\{return p\.x\+params\.dimsTile\.x\*\(p\.y\+params\.dimsTile\.y\*p\.z\)\+1u;\}/);
  assert.match(pagedPhiDifferentialShader, /let key=airCellKey\(p\)/);
  assert.doesNotMatch(pagedPhiDifferentialShader, /p\.x\|\(p\.y<<10u\)/,
    "diagnostics must query the same exact linear key as surface publication");
});

function bytes(data: ArrayBufferView): Uint8Array<ArrayBuffer> {
  const result = new Uint8Array(data.byteLength);
  result.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
  return result;
}

test("Dawn differential reports exact tile coverage, fallbacks, error, and worst cell", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU paged-phi diagnostic checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]); const adapter = await gpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice(); const errors: string[] = [];
  device.addEventListener("uncapturederror", (event: unknown) => errors.push((event as { error: { message: string } }).error.message));
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const plan = planOctreeSurfacePages(8, [4, 4, 4], { maximumPages: 1 });
  const arenaWords = new Uint32Array(plan.allocatedWords); arenaWords.fill(0);
  arenaWords.fill(0xffff_ffff, plan.pageTableOffsetWords, plan.pageTableOffsetWords + plan.leafCapacity);
  arenaWords[plan.hashOffsetWords] = 1;
  const arena = device.createBuffer({ size: plan.arenaBytes, usage: storage }); device.queue.writeBuffer(arena, 0, arenaWords);
  const leafData = new ArrayBuffer(plan.leafCapacity * 64), leafU32 = new Uint32Array(leafData), leafF32 = new Float32Array(leafData);
  leafU32.set([0, 0, 0, 4, 1 << 5, 0, 0, 0]); leafF32.set([0, 1, 0, 0], 8);
  const leaves = device.createBuffer({ size: leafData.byteLength, usage: storage }); device.queue.writeBuffer(leaves, 0, bytes(new Uint8Array(leafData)));
  const worklistWords = new Uint32Array(32); worklistWords[0] = 1; worklistWords[1] = 1; worklistWords[16] = 0;
  const worklist = device.createBuffer({ size: 128, usage: storage }); device.queue.writeBuffer(worklist, 0, worklistWords);
  const dispatch = device.createBuffer({ size: 32, usage: GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(dispatch, 0, new Uint32Array([1, 1, 1, 0, 0, 1, 1, 0]));
  const dense = device.createTexture({ size: [4, 4, 4], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
  const upload = new Float32Array(64 * 4 * 4); device.queue.writeTexture({ texture: dense }, bytes(upload), { bytesPerRow: 256, rowsPerImage: 4 }, [4, 4, 4]);
  const params = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM });
  const pageDispatch = device.createBuffer({ size: 12, usage: GPUBufferUsage.INDIRECT });
  const source: OctreeSurfacePageSource = {
    plan, arena: { buffer: arena }, leaves: { buffer: leaves }, params: { buffer: params },
    control: { buffer: arena, wordOffset: 0, overflowWord: 3, generationWord: 7 },
    activePages: { buffer: arena, offsetBytes: plan.activeOffsetWords * 4, entriesOffsetBytes: (plan.activeOffsetWords + 4) * 4, indirectBuffer: pageDispatch, indirectOffsetBytes: 0 },
    phiAOffsetBytes: plan.phiAOffsetWords * 4, pageTableOffsetBytes: plan.pageTableOffsetWords * 4,
  };
  const diagnostic = new WebGPUOctreePhiDifferential(device, dense, source, leaves, worklist, [4, 4, 4], 4);
  const encoder = device.createCommandEncoder(); diagnostic.encode(encoder, dispatch); device.queue.submit([encoder.finish()]);
  const result = await diagnostic.read(device);
  assert.equal(result.samples, 64); assert.equal(result.comparedSamples, 64);
  assert.equal(result.missingLeafSamples, 0); assert.equal(result.affineFallbackSamples, 64);
  assert.ok(Math.abs(result.maximumAbsoluteMismatch - 1.5) < 1e-6);
  assert.ok(Math.abs(result.meanAbsoluteMismatch - 1) < 1e-6);
  assert.equal(result.signMismatchSamples, 32); assert.equal(result.nonFiniteSamples, 0);
  assert.ok(result.maximumMismatchCell); assert.ok(result.maximumMismatchCell[0] === 0 || result.maximumMismatchCell[0] === 3);
  assert.equal(result.maximumMismatchDensePhi, 0); assert.ok(Math.abs(result.maximumMismatchPagedPhi!) === 1.5);
  await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []);
  diagnostic.destroy(); dense.destroy(); arena.destroy(); leaves.destroy(); worklist.destroy(); dispatch.destroy(); params.destroy(); pageDispatch.destroy(); device.destroy();
});
