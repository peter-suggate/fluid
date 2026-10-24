/** Paired production-renderer measurements and full-resolution image comparisons.
 * The reference retains the old per-fragment visibility; both arms keep AO,
 * the same lights, sample budgets, material shading and mesh. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import { resolveDisplayGrade, sceneLinearToDisplay } from "../lib/core/webgpu-lighting";
import { svoSceneLighting } from "../lib/svo/features/lighting-visibility/svo-dry-scene-lighting";
import { buildSvoSceneLights, packSvoLightRecords, type SvoLightRecord } from "../lib/svo/contracts/svo-light-abi";
import type { SparseVoxelDrySceneData } from "../lib/svo/contracts/scene-publication";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { packSvoDryViewUniforms } from "./svo-dry-frame-harness";
import { writeFramePng } from "./write-frame-png";

function half(v: number): number {
  const e = (v >> 10) & 31, m = v & 1023, sign = v & 32768 ? -1 : 1;
  return sign * (e === 0 ? m * 2 ** -24 : e === 31 ? (m ? NaN : Infinity) : (1 + m / 1024) * 2 ** (e - 15));
}
const quantile = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.max(0, Math.ceil(xs.length * q) - 1)]!;

export async function benchmarkRasterVisibility(input: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; uniforms: GPUBuffer;
  view: Parameters<typeof packSvoDryViewUniforms>[0]; publication: SparseVoxelDrySceneData; outPath: string;
}) {
  const { device, renderer, uniforms, view, publication } = input;
  const { width, height } = view, directory = dirname(input.outPath);
  mkdirSync(directory, { recursive: true });
  const target = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC });
  const grade = resolveDisplayGrade(svoSceneLighting(view.scene).grade);
  const cycles = Number(process.env.FLUID_VISIBILITY_CYCLES ?? 16);
  const distanceScale = Number(process.env.FLUID_VISIBILITY_CAMERA_SCALE ?? 1);
  assert.ok(Number.isFinite(distanceScale) && distanceScale > 0);
  const baseCamera = { ...view.camera, distance_m: view.camera.distance_m * distanceScale };
  assert.ok(Number.isInteger(cycles) && cycles >= 4 && cycles <= 256);
  const lights = buildSvoSceneLights(view.scene).records;
  const extra = (kind: SvoLightRecord["kind"], id: number, position: SvoLightRecord["position_m"], color: SvoLightRecord["colorLinear"]): SvoLightRecord => ({
    ...lights[0]!, kind, lightId: id, ownerId: 0xffffffff, sourceKey: `visibility-study/${kind}`,
    position_m: position, range_m: 6, direction: [0, -1, 0], colorLinear: color, intensity: 2,
    radius_m: .12, axisU: [1, 0, 0], axisV: [0, 0, 1], halfWidth_m: .2, halfHeight_m: .2,
    ...(kind === "spot" ? { cone: { cosInner: .8, cosOuter: .5 } } : {}),
  });
  const fixtures = [
    { name: "authored", publication },
    { name: "mixed-lights", publication: { ...publication, renderRevision: publication.renderRevision + 1,
      lightRecords: packSvoLightRecords([...lights, extra("point", 1001, [-1, 2, 1], [1, .4, .2]),
        extra("spot", 1002, [1, 2, 0], [.2, .5, 1]), extra("sphereArea", 1003, [0, 1.8, -1], [.3, 1, .4])]) } },
  ];
  const results: unknown[] = [];
  let previousArm: boolean | undefined;
  async function frame(enabled: boolean, timed: boolean) {
    renderer.setRasterLightVisibilityEnabled(enabled || process.env.FLUID_VISIBILITY_QUEUE_COMPARE === "1");
    renderer.setRasterVisibilityQueueEnabled(enabled || process.env.FLUID_VISIBILITY_QUEUE_COMPARE !== "1");
    if (previousArm !== enabled) { previousArm = enabled; await frame(enabled, false); }
    const encoder = device.createCommandEncoder();
    const recorder = timed ? new GPUPassTimestampRecorder(device, 512, "Paired raster visibility") : undefined;
    assert.ok(renderer.encode(recorder?.instrument(encoder) ?? encoder, target));
    if (recorder) assert.ok(recorder.resolve(encoder));
    device.queue.submit([encoder.finish()]);
    if (!recorder) { await device.queue.onSubmittedWorkDone(); return undefined; }
    const reading = await recorder.read(); assert.ok(reading); return reading;
  }
  async function capture(name: string) {
    const pitch = Math.ceil(width * 8 / 256) * 256;
    const read = device.createBuffer({ size: pitch * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    try {
      const encoder = device.createCommandEncoder();encoder.copyTextureToBuffer({ texture: target }, { buffer: read, bytesPerRow: pitch }, [width, height]);
      device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);
      const bytes = new Uint8Array(width * height * 8), mapped = new Uint8Array(read.getMappedRange());
      for (let y = 0; y < height; y++) bytes.set(mapped.subarray(y * pitch, y * pitch + width * 8), y * width * 8);
      writeFileSync(join(directory, `${name}.rgba16f`), bytes);
      writeFramePng(join(directory, `${name}.png`), { width, height, packedRows: new Uint32Array(bytes.buffer), grade });
      return new Uint16Array(bytes.buffer);
    } finally { if (read.mapState === "mapped") read.unmap();read.destroy(); }
  }
  try {
    renderer.setRasterVisibilityTilingEnabled(process.env.FLUID_VISIBILITY_COMPACT_TILES === "1");
    renderer.ensureSize(width, height);
    renderer.setLightingOptions({ coneTracingMode: "raster-ao", coneLightingScale: .5, shadowsEnabled: true, ambientOcclusionEnabled: true, globalIlluminationEnabled: false });
    for (const fixture of process.env.FLUID_VISIBILITY_AUTHORED_ONLY === "1" ? fixtures.slice(0, 1) : fixtures) {
      renderer.publishScene(fixture.publication);
      for (const motion of ["stationary", "orbit"] as const) {
        const reference: number[] = [], candidate: number[] = [], paired: number[] = [];
        const compute: Record<string, number[]> = {};
        for (let cycle = -4; cycle < cycles; cycle++) {
          const camera = motion === "orbit" ? { ...baseCamera, azimuth_rad: baseCamera.azimuth_rad + .2 * Math.sin(cycle * .21) } : baseCamera;
          device.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ ...view, camera, cameraMoving: motion === "orbit" }));
          const pair: number[] = [];
          for (const enabled of cycle % 2 ? [true, false] : [false, true]) {
            const reading = await frame(enabled, cycle >= 0);if (!reading) continue;
            (enabled ? candidate : reference).push(reading.span_ms);pair[enabled ? 1 : 0] = reading.span_ms;
            if (enabled) for (const p of reading.passes) if (p.trusted) (compute[p.label] ??= []).push(p.duration_ms);
          }
          if (cycle >= 0) paired.push(pair[0]! / pair[1]!);
        }
        await frame(false, false);const a = await capture(`${fixture.name}-${motion}-reference`);
        await frame(true, false);const b = await capture(`${fixture.name}-${motion}-candidate`);
        let sum = 0, sumSquares = 0, maximum = 0, over2 = 0, over8 = 0, receivers = 0, depthMismatches = 0;
        for (let p = 0; p < width * height; p++) {
          if (a[p * 4 + 3] !== b[p * 4 + 3]) depthMismatches++;
          if (half(a[p * 4 + 3]!) <= 0) continue;
          const x = sceneLinearToDisplay([half(a[p * 4]!), half(a[p * 4 + 1]!), half(a[p * 4 + 2]!)], grade);
          const y = sceneLinearToDisplay([half(b[p * 4]!), half(b[p * 4 + 1]!), half(b[p * 4 + 2]!)], grade);
          let error = 0;
          for (let c = 0; c < 3; c++) { const d = Math.abs(x[c]! - y[c]!) * 255;sum += d;sumSquares += d * d;error = Math.max(error, d); }
          maximum = Math.max(maximum, error);if (error > 2) over2++;if (error > 8) over8++;receivers++;
        }
        let reuseVerified = false;
        if (process.env.FLUID_VISIBILITY_REUSE_CHECK === "1") {
          const camera = motion === "orbit" ? { ...baseCamera, azimuth_rad: baseCamera.azimuth_rad + .2 * Math.sin((cycles - 1) * .21) } : baseCamera;
          const words = packSvoDryViewUniforms({ ...view, camera, cameraMoving: motion === "orbit" });
          renderer.setRasterVisibilityView(words);
          await frame(true, false);
          const cached = await frame(true, true);
          assert.ok(cached && !cached.passes.some(p => p.label === "Reduced all-light visibility"), "unchanged receiver inputs reuse visibility");
          const reused = await capture(`${fixture.name}-${motion}-reused`);
          assert.deepEqual(reused, b, "visibility reuse preserves every HDR and depth bit");
          const moved = new Float32Array(words); moved[0] = moved[0]! + .001;
          renderer.setRasterVisibilityView(moved);
          const invalidated = await frame(true, true);
          assert.ok(invalidated?.passes.some(p => p.label === "Reduced all-light visibility"), "changed view invalidates visibility");
          renderer.setRasterVisibilityView(undefined); reuseVerified = true;
        }
        const row = { reuseVerified, fixture: fixture.name, motion, referenceMedian_ms: quantile(reference, .5), candidateMedian_ms: quantile(candidate, .5),
          referenceP95_ms: quantile(reference, .95), candidateP95_ms: quantile(candidate, .95), pairedMedianSpeedup: quantile(paired, .5), reference, candidate,
          computeMedian_ms: Object.fromEntries(Object.entries(compute).map(([k, v]) => [k, quantile(v, .5)])),
          quality: { receivers, depthMismatches, displayMeanAbsolute_255: sum / (receivers * 3), displayRmse_255: Math.sqrt(sumSquares / (receivers * 3)),
            displayMaximum_255: maximum, fractionOver2: over2 / receivers, fractionOver8: over8 / receivers } };
        assert.equal(depthMismatches, 0, "Visibility changes must not move geometry");
        assert.ok(row.quality.displayRmse_255 < 1, "Receiver display RMSE must stay below one 8-bit level");
        assert.ok(row.quality.fractionOver8 < .01, "Large differences must cover less than 1% of receivers");
        results.push(row);console.log(JSON.stringify({ phase: "visibility-pair", ...row }));
        writeFileSync(input.outPath, JSON.stringify({ scene: view.scene.sceneId, width, height, camera: baseCamera, cycles, adapter: device.adapterInfo, reference: process.env.FLUID_VISIBILITY_QUEUE_COMPARE === "1" ? "inline exact edges, stride 2" : "per-fragment visibility", candidate: process.env.FLUID_VISIBILITY_QUEUE_COMPARE === "1" ? "queued exact edges, stride 2" : "precomputed visibility, stride 2", scope: "Production dry render, AO and shadows on; excludes water, physics and browser presentation", results }, null, 2));
      }
    }
  } finally { renderer.setRasterLightVisibilityEnabled(true);renderer.publishScene(publication);target.destroy(); }
}
