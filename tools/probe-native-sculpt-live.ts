import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { SceneDescription } from "../lib/core/model";
import type { SolidWorldVoxelPatch } from "../lib/core/solid-world";
import { svoRenderScenePublication, type SvoRenderScenePublication } from "../lib/svo/features/scene-publication/svo-render-scene-publication";
import type { WebGPULiveSvoScene } from "../lib/svo/features/scene-publication/webgpu-live-svo-scene";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";

/** Diagnostic readback permission only; production never copies these planes. */
export function nativeSculptProbeDevice(device: GPUDevice): GPUDevice {
  return new Proxy(device, { get(target, key) {
    if (key === "createTexture") return (descriptor: GPUTextureDescriptor) => target.createTexture({ ...descriptor,
      usage: descriptor.usage | (descriptor.label === "Sparse voxel split exact primary geometry"
        || descriptor.label === "Sparse voxel split exact primary identity" ? GPUTextureUsage.COPY_SRC : 0) });
    const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
}

/** Exercise successive document publications on an already running renderer.
 * Fences/readbacks here are measurement only; no edit path waits on them. */
export async function probeNativeSculptLive(options: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; solver: WebGPULiveSvoScene;
  scene: SceneDescription; target: GPUTexture; width: number; height: number; directory: string;
  initialPublication: SvoRenderScenePublication;
  encodeFrame(encoder: GPUCommandEncoder): void; png(rows: Uint32Array): Uint8Array;
}): Promise<void> {
  const { device, renderer, solver, scene, target, width, height, directory, encodeFrame } = options;
  mkdirSync(directory, { recursive: true });
  assert.equal(scene.sceneId, "atlantic-lighthouse", "This probe's world-space brushes target the lighthouse");
  assert.ok(!scene.solidVoxels.some(p => p.affectsScenery), "Start from the intact scene");
  const native = (renderer as unknown as { importedSurfaceMeshes: Map<string, { vertices: GPUBuffer; indices: GPUBuffer }> }).importedSurfaceMeshes;
  const held = new Map([...native].map(([key, mesh]) => [key, [mesh.vertices, mesh.indices]]));
  let publication = options.initialPublication;
  const samples: { stroke: string; stage_ms: number; assemble_ms: number; publish_ms: number; encode_ms: number; completion_ms: number }[] = [];
  const images: { name: string; rows: Uint32Array }[] = [];
  const geometryFrames = new Map<string, Float32Array>();
  const identityFrames = new Map<string, Uint32Array>();
  const nativePixels: number[] = [];
  async function readTexture(texture: GPUTexture, pixelBytes: number) {
    const bytesPerRow = Math.ceil(width * pixelBytes / 256) * 256;
    const read = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer: read, bytesPerRow, rowsPerImage: height }, [width, height]);
    device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
    const source = new Uint8Array(read.getMappedRange()), bytes = new Uint8Array(width * height * pixelBytes);
    for (let y = 0; y < height; y++) bytes.set(source.subarray(y * bytesPerRow, y * bytesPerRow + width * pixelBytes), y * width * pixelBytes);
    read.unmap(); read.destroy(); return bytes;
  }
  async function capture(name: string) {
    images.push({ name, rows: new Uint32Array((await readTexture(target, 8)).buffer) });
    const planes = renderer as unknown as { splitGeometry: GPUTexture; splitOpaqueIdentity: GPUTexture };
    geometryFrames.set(name, new Float32Array((await readTexture(planes.splitGeometry, 16)).buffer));
    const identities = new Uint32Array((await readTexture(planes.splitOpaqueIdentity, 8)).buffer);
    identityFrames.set(name, identities);
    if (name === "intact") {
      for (let pixel = 0; pixel < width * height; pixel++) {
        if ((publication.data.materialRecords[identities[pixel * 2]! * SVO_MATERIAL_RECORD_WORDS + 23]! & SVO_MATERIAL_FLAGS.sampledVolume) !== 0) nativePixels.push(pixel);
      }
      assert.ok(nativePixels.length > width * height * .05, "the probe must see the native asset surfaces");
    }
  }
  await capture("intact");
  const patches: SolidWorldVoxelPatch[] = [];
  const strokes = Number(process.env.FLUID_NATIVE_SCULPT_STROKES ?? 16);
  for (let step = 0; step <= strokes + 1; step++) {
    const undo = step === strokes, redo = step === strokes + 1;
    if (!undo && !redo) {
      const rock = step >= strokes / 2, shift = step % (strokes / 2);
      patches.push({ operation: "clear", kind: "ellipsoid", affectsScenery: true,
        minimum: rock ? [88 + shift, 8, 116] : [117 + shift, 31, 95],
        maximumExclusive: rock ? [100 + shift, 16, 128] : [127 + shift, 41, 105] });
    }
    const next = { ...scene, solidVoxels: undo ? scene.solidVoxels : [...scene.solidVoxels, ...patches] };
    const start = performance.now(); solver.stageSceneUpdate(next); const staged = performance.now();
    publication = svoRenderScenePublication(next, step + 2, publication)!;
    const assembled = performance.now();
    assert.ok(publication); assert.equal(renderer.publishScene({ ...publication.data, renderRevision: step + 2 }), true);
    const published = performance.now();
    assert.equal(renderer.presentationBundleStatus.state, "ready", "an edit must not compile a presentation pipeline");
    for (const [key, mesh] of native) assert.deepEqual([mesh.vertices, mesh.indices], held.get(key), "live cuts retain the exact native GPU buffers");
    const encoder = device.createCommandEncoder(); solver.encodeSceneMaintenance(encoder); encodeFrame(encoder);
    device.queue.submit([encoder.finish()]); const encoded = performance.now();
    await device.queue.onSubmittedWorkDone();
    samples.push({ stroke: undo ? "undo" : redo ? "redo" : `${step + 1}`, stage_ms: staged - start,
      assemble_ms: assembled - staged, publish_ms: published - assembled, encode_ms: encoded - published, completion_ms: performance.now() - encoded });
    if (step === 0 || step === strokes - 1 || undo || redo) await capture(step === 0 ? "first-cut" : undo ? "undo" : redo ? "redo" : "carved");
  }
  // Image encoding happens after GPU work, outside all measured publications.
  for (const image of images) writeFileSync(join(directory, `${image.name}.png`), options.png(image.rows));
  const geometryDifferences = (a: string, b: string) => {
    const before = geometryFrames.get(a)!, after = geometryFrames.get(b)!;
    return nativePixels.filter(pixel => [0, 1, 2, 3].some(channel => Math.abs(before[pixel * 4 + channel]! - after[pixel * 4 + channel]!) > 1e-4)).length;
  };
  const undoChangedNativePixels = geometryDifferences("intact", "undo");
  const redoChangedNativePixels = geometryDifferences("carved", "redo");
  const firstCutChangedNativePixels = geometryDifferences("intact", "first-cut");
  const undoDifferenceExamples = nativePixels.filter(pixel => [0, 1, 2, 3].some(channel =>
    Math.abs(geometryFrames.get("intact")![pixel * 4 + channel]! - geometryFrames.get("undo")![pixel * 4 + channel]!) > 1e-4))
    .slice(0, 10).map(pixel => ({ x: pixel % width, y: Math.floor(pixel / width),
      before: Array.from(geometryFrames.get("intact")!.subarray(pixel * 4, pixel * 4 + 4)),
      after: Array.from(geometryFrames.get("undo")!.subarray(pixel * 4, pixel * 4 + 4)),
      materials: [identityFrames.get("intact")![pixel * 2], identityFrames.get("undo")![pixel * 2]] }));
  const report = { width, height, samples, nativeMeshBuffersUnchanged: true, editPipelineCompilations: 0,
    nativePixels: nativePixels.length, firstCutChangedNativePixels, undoChangedNativePixels, redoChangedNativePixels,
    undoDifferenceExamples,
    note: "Completion includes queued GPU work; run with the shared GPU timing lock for performance evidence." };
  writeFileSync(join(directory, "live-edits.json"), JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
  assert.ok(firstCutChangedNativePixels > 0 && firstCutChangedNativePixels < nativePixels.length * .1,
    "the first brush changes a local surface region without replacing the asset");
  assert.equal(undoChangedNativePixels, 0, "undo restores native geometry on its first frame, including while the voxel mesh rebuilds");
  assert.equal(redoChangedNativePixels, 0, "redo restores carved geometry on its first frame");
}
