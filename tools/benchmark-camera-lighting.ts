/** Camera-motion attribution on the production dry renderer. Keep the same
 * geometry/light budget across backends; disabled effects are attribution
 * controls, not proposed quality settings. Invoke through the dry-frame harness
 * with FLUID_SVO_DRY_FRAME_CAMERA_BUDGET=1. Reproduction and findings live in
 * docs/validation/tokyo-camera-performance.json.
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import { resolveDisplayGrade } from "../lib/core/webgpu-lighting";
import { svoSceneLighting } from "../lib/svo/features/lighting-visibility/svo-dry-scene-lighting";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import type { SvoConeTracingMode } from "../lib/svo/pipeline/svo-render-options";
import type { SvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";
import { packSvoDryViewUniforms } from "./svo-dry-frame-harness";
import { writeFramePng } from "./write-frame-png";

export async function benchmarkCameraLighting(input: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; uniforms: GPUBuffer; target: GPUTexture;
  view: Parameters<typeof packSvoDryViewUniforms>[0]; tuning: SvoRenderTuning;
  mode: SvoConeTracingMode; adapterInfo: unknown; outPath: string; cycles: number; ready(): Promise<void>;
}) {
  const { device, renderer, uniforms, target, view, tuning, mode, cycles } = input;
  assert.ok(GPUPassTimestampRecorder.supported(device));
  assert.ok(Number.isSafeInteger(cycles) && cycles >= 4 && cycles <= 128);
  const arms = [
    { name: "baseline", ao: true, shadows: true, coarse: tuning.rasterCoarseAoStrength },
    { name: "ao-off", ao: false, shadows: true, coarse: tuning.rasterCoarseAoStrength },
    { name: "shadows-off", ao: true, shadows: false, coarse: tuning.rasterCoarseAoStrength },
    { name: "visibility-off", ao: false, shadows: false, coarse: 0 },
    ...(mode === "raster-ao" ? [{ name: "coarse-ao-off", ao: true, shadows: true, coarse: 0 }] : []),
  ];
  type Samples = { gpu: number[]; host: number[]; missingTimestamps: number; compute: Record<string, number[]> };
  const samples = new Map<string, Samples>();
  const quantile = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.ceil(xs.length * q) - 1]!;
  const directory = dirname(input.outPath); mkdirSync(directory, { recursive: true });
  async function frame(index: number, moving: boolean, timed: boolean) {
    const camera = moving ? { ...view.camera,
      azimuth_rad: view.camera.azimuth_rad + .12 * Math.sin(index * .31),
      elevation_rad: view.camera.elevation_rad + .04 * Math.cos(index * .23) } : view.camera;
    const words = packSvoDryViewUniforms({ ...view, camera, cameraMoving: moving });
    const start = performance.now();
    device.queue.writeBuffer(uniforms, 0, words);
    renderer.setRasterVisibilityView(words);
    const encoder = device.createCommandEncoder();
    const recorder = timed ? new GPUPassTimestampRecorder(device, 512, "Camera lighting") : undefined;
    assert.ok(renderer.encode(recorder?.instrument(encoder) ?? encoder, target));
    // A fully reused frame has no timed passes. It still submits its ordinary
    // maintenance clears; report the empty pass span as zero, never as FPS.
    const resolved = recorder?.resolve(encoder);
    device.queue.submit([encoder.finish()]);
    const reading = resolved ? await recorder!.read() : undefined;
    if (!resolved) { recorder?.destroy(); await device.queue.onSubmittedWorkDone(); }
    return { reading, resolved, host: performance.now() - start };
  }
  async function capture(name: string) {
    const pitch = Math.ceil(view.width * 8 / 256) * 256;
    const read = device.createBuffer({ size: pitch * view.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: pitch }, [view.width, view.height]);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(view.width * view.height * 8), mapped = new Uint8Array(read.getMappedRange());
      for (let y = 0; y < view.height; y++) bytes.set(mapped.subarray(y * pitch, y * pitch + view.width * 8), y * view.width * 8);
      assert.ok(bytes.some(value => value !== 0), "GPU returned an all-zero frame; timings from non-executing work are invalid");
      writeFramePng(join(directory, `${mode}-${name}.png`), { width: view.width, height: view.height,
        packedRows: new Uint32Array(bytes.buffer), grade: resolveDisplayGrade(svoSceneLighting(view.scene).grade) });
    } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
  }
  // Forward/reverse blocks, identical camera sequence in every arm. Compilation
  // and warmup are outside samples. Do not run other GPU tests concurrently.
  for (let round = 0; round < 2; round++) for (const arm of round ? [...arms].reverse() : arms) {
    renderer.setRenderTuning({ ...tuning, rasterCoarseAoStrength: arm.coarse });
    renderer.setLightingOptions({ coneTracingMode: mode, coneLightingScale: tuning.coneLightingScale,
      globalIlluminationEnabled: false, shadowsEnabled: arm.shadows, ambientOcclusionEnabled: arm.ao,
      latticeVisibilityEnabled: false, worldGiCacheEnabled: false });
    await input.ready();
    for (const moving of [false, true]) {
      const key = `${arm.name}/${moving ? "orbit" : "stationary"}`;
      const entry = samples.get(key) ?? { gpu: [], host: [], missingTimestamps: 0, compute: {} };
      for (let index = -4; index < cycles; index++) {
        const { reading, resolved, host } = await frame(index, moving, index >= 0);
        if (index < 0) continue;
        if (moving) assert.ok(resolved, "Moving frames must render, not reuse stale imagery");
        if (reading || !resolved) entry.gpu.push(reading?.span_ms ?? 0);
        else entry.missingTimestamps++;
        entry.host.push(host);
        for (const pass of reading?.passes ?? []) if (pass.trusted) (entry.compute[pass.label] ??= []).push(pass.duration_ms);
      }
      samples.set(key, entry);
      if (round === 0) await capture(`${arm.name}-${moving ? "orbit" : "stationary"}`);
      console.log(JSON.stringify({ phase: "camera-lighting", mode, key, round, gpuMedian_ms: quantile(entry.gpu, .5) ?? null,
        hostMedian_ms: quantile(entry.host, .5), missingTimestamps: entry.missingTimestamps }));
    }
  }
  const results = [...samples].map(([name, values]) => ({ name,
    gpuMedian_ms: values.missingTimestamps ? null : quantile(values.gpu, .5),
    gpuP95_ms: values.missingTimestamps ? null : quantile(values.gpu, .95), missingTimestamps: values.missingTimestamps,
    hostMedian_ms: quantile(values.host, .5), samples_ms: values.gpu,
    computeMedian_ms: Object.fromEntries(Object.entries(values.compute).map(([name, xs]) => [name, quantile(xs, .5)])) }));
  writeFileSync(input.outPath, JSON.stringify({ scene: view.scene.sceneId, mode, adapter: input.adapterInfo,
    valid: results.every(result => result.missingTimestamps === 0),
    width: view.width, height: view.height, tuning, camera: view.camera, cycles,
    scope: "Production dry renderer; excludes fluid, water and browser presentation. GI off in both backends. GPU frame spans and trusted compute timings; render-pass windows are not individual costs. Effect-off arms change quality.",
    method: "Two forward/reverse blocks, four warmup frames per arm/motion; same camera sequence; one GPU fence per frame. Zero means no timestamped passes on a reused frame.", results }, null, 2));
  assert.ok(results.every(result => result.missingTimestamps === 0), "Missing GPU timestamps invalidate this study; inspect the report before using any timings");
}
