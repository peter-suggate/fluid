/** Same-scene diagnostic ablations. Disabled effects are attribution controls,
 * never proposed quality presets. Measures whole-frame spans on Metal; render
 * pass timestamp windows must not be interpreted as individual pass costs. */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import type { SvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";
import { packSvoDryViewUniforms } from "./svo-dry-frame-harness";

export async function benchmarkRasterLightingBudget(input: {
  device: GPUDevice; renderer: SparseVoxelDrySceneRenderer; uniforms: GPUBuffer;
  view: Parameters<typeof packSvoDryViewUniforms>[0]; tuning: SvoRenderTuning;
  outPath: string;
}) {
  const { device, renderer, uniforms, view, tuning } = input;
  assert.ok(GPUPassTimestampRecorder.supported(device), "GPU timestamps required");
  const arms = [
    { name: "baseline" },
    { name: "ao-off", ao: false },
    { name: "shadows-off", shadows: false },
    { name: "visibility-off", ao: false, shadows: false },
    { name: "coarse-ao-off", coarse: 0 },
    { name: "one-area-sample", area: 1 },
    { name: "one-light-budget", lights: 1 },
    { name: "half-output-resolution", scale: 0.5 },
  ] as const;
  type Arm = { name: string; ao?: boolean; shadows?: boolean; coarse?: number; area?: number; lights?: number; scale?: number };
  const samples = new Map<string, number[]>();
  const aoSamples = new Map<string, number[]>();
  const targets = new Map<number, GPUTexture>();
  let coverage: { emptyPixelFraction: number; emptyTileFraction: number; tileSize: number } | undefined;
  const quantile = (values: number[], q: number) => [...values].sort((a, b) => a - b)[Math.ceil(values.length * q) - 1]!;
  try {
    for (const scale of [1, 0.5]) targets.set(scale, device.createTexture({
      label: `Lighting budget target ${scale}`, size: [Math.floor(view.width * scale), Math.floor(view.height * scale)],
      format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    }));
    // Forward/reverse blocks reduce ordering bias; no other GPU benchmark
    // should run concurrently when these numbers are used as a release gate.
    for (let round = 0; round < 4; round++) {
      for (const arm of (round % 2 ? [...arms].reverse() : arms) as readonly Arm[]) {
        const target = targets.get(arm.scale ?? 1)!;
        renderer.setRenderTuning({ ...tuning,
          rasterCoarseAoStrength: arm.coarse ?? tuning.rasterCoarseAoStrength,
          stableAreaLightSamples: arm.area ?? tuning.stableAreaLightSamples,
          movingAreaLightSamples: arm.area ?? tuning.movingAreaLightSamples,
          maximumShadedLights: arm.lights ?? tuning.maximumShadedLights,
        });
        renderer.setLightingOptions({ coneTracingMode: "raster-ao", coneLightingScale: 0.5,
          globalIlluminationEnabled: false, shadowsEnabled: arm.shadows ?? true, ambientOcclusionEnabled: arm.ao ?? true });
        renderer.ensureSize(target.width, target.height);
        device.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ ...view, width: target.width, height: target.height }));
        for (let frame = -2; frame < 4; frame++) {
          const encoder = device.createCommandEncoder();
          const recorder = frame >= 0 ? new GPUPassTimestampRecorder(device, 512, arm.name) : undefined;
          assert.ok(renderer.encode(recorder?.instrument(encoder) ?? encoder, target));
          if (recorder) assert.ok(recorder.resolve(encoder));
          device.queue.submit([encoder.finish()]);
          if (!recorder) { await device.queue.onSubmittedWorkDone(); continue; }
          const reading = await recorder.read();
          assert.ok(reading, `Missing timestamps for ${arm.name}`);
          const values = samples.get(arm.name) ?? []; values.push(reading.span_ms); samples.set(arm.name, values);
          const ao = reading.passes.find(pass => pass.label === "Half-resolution sun visibility and horizon AO");
          if (ao) { const values = aoSamples.get(arm.name) ?? []; values.push(ao.duration_ms); aoSamples.set(arm.name, values); }
        }
        if (!coverage && arm.name === "baseline") {
          // Read once, outside all timed frames. Reversed-Z zero is a miss;
          // this includes the backdrop as geometry, not merely the tank.
          const depth = renderer.gBufferTextures?.hardwareDepth;
          assert.ok(depth);
          const pitch = Math.ceil(depth.width * 4 / 256) * 256;
          const read = device.createBuffer({ size: pitch * depth.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
          try {
            const copy = device.createCommandEncoder();
            copy.copyTextureToBuffer({ texture: depth, aspect: "depth-only" }, { buffer: read, bytesPerRow: pitch }, [depth.width, depth.height]);
            device.queue.submit([copy.finish()]); await read.mapAsync(GPUMapMode.READ);
            const values = new Float32Array(read.getMappedRange());
            const tileSize = 16, columns = Math.ceil(depth.width / tileSize);
            const occupied = new Uint8Array(columns * Math.ceil(depth.height / tileSize));
            let empty = 0;
            for (let y = 0; y < depth.height; y++) for (let x = 0; x < depth.width; x++) {
              if (values[y * pitch / 4 + x] === 0) empty++;
              else occupied[Math.floor(y / tileSize) * columns + Math.floor(x / tileSize)] = 1;
            }
            coverage = { emptyPixelFraction: empty / (depth.width * depth.height),
              emptyTileFraction: occupied.filter(value => value === 0).length / occupied.length, tileSize };
            read.unmap();
          } finally { read.destroy(); }
        }
      }
    }
    const results = arms.map(arm => ({ name: arm.name, samples_ms: samples.get(arm.name)!,
      median_ms: quantile(samples.get(arm.name)!, 0.5), p95_ms: quantile(samples.get(arm.name)!, 0.95),
      visibilityComputeMedian_ms: aoSamples.has(arm.name) ? quantile(aoSamples.get(arm.name)!, 0.5) : null,
    }));
    const report = { scope: "Production dry renderer; excludes simulation, water and browser presentation. Ablations change quality.",
      scene: view.scene.sceneId, resolution: { width: view.width, height: view.height }, tuning, coverage,
      method: "GPU timestamp frame spans; four alternating forward/reverse arm blocks, two warmup and four measured frames per arm/block",
      results };
    mkdirSync(dirname(input.outPath), { recursive: true });
    writeFileSync(input.outPath, JSON.stringify(report, null, 2));
    console.log(JSON.stringify({ phase: "lighting-budget", ...report }));
  } finally { for (const target of targets.values()) target.destroy(); }
}
