import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { join } from "node:path";
import { GPUPassTimestampRecorder } from "../lib/core/performance-trace";
import type { SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";
import { packSvoDryViewUniforms, type SvoDryViewUniformInputs } from "./svo-dry-frame-harness";
import { mainKernelVariants, type MainKernelVariant } from "./svo-main-kernel-variants";
import { writeFramePng } from "./write-frame-png";

const median = (values: number[]) => [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];
const stats = (values: number[]) => ({ median: median(values), p95: [...values].sort((a,b) => a-b)[Math.ceil(values.length*.95)-1], samples: values });
function pairedRatios(samples: { variant: string; gpu_ms: number }[]) {
  const ratios: number[] = [];
  // Pool eight rounds: every pose appears with both ABBA and BAAB ordering.
  // This cancels submission-position warmup effects before resampling rather
  // than treating alternating cold/warm ratios as independent observations.
  for (let i = 0; i < samples.length; i += 32) {
    const round = samples.slice(i, i + 32);
    const total = (baseline: boolean) => round.filter(s => (s.variant === "baseline") === baseline).reduce((sum, s) => sum + s.gpu_ms, 0);
    ratios.push(total(false) / total(true));
  }
  let seed = 123;
  const bootstrap = Array.from({ length: 2000 }, () => {
    let sum = 0;
    for (let i = 0; i < ratios.length; i++) { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; sum += ratios[Math.floor(seed / 0x1_0000_0000 * ratios.length)]; }
    return sum / ratios.length;
  }).sort((a,b) => a-b);
  return { mean: ratios.reduce((sum, ratio) => sum + ratio, 0) / ratios.length,
    confidence95: ratios.length >= 4 ? [bootstrap[50], bootstrap[1950]] : undefined, blocks: ratios.length, ratios };
}
function otherGpuJobs(): string[] {
  return execFileSync("ps", ["-Ao", "pid,comm,args"], { encoding: "utf8" }).split("\n").filter(line => {
    const columns = line.trim().split(/\s+/);
    return Number(columns[0]) !== process.pid && /(?:^|\/)node$/.test(columns[1] ?? "")
      && /[ /]tools\/(?:benchmark|probe|experiment|run-svo|check-svo)[^ ]*\.(?:ts|mjs)(?: |$)/.test(columns.slice(2).join(" "));
  });
}
function half(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1, exp = (bits >> 10) & 31, fraction = bits & 1023;
  return sign * (exp === 0 ? fraction * 2 ** -24 : exp === 31 ? fraction ? NaN : Infinity : (1 + fraction / 1024) * 2 ** (exp - 15));
}
function compare(a: Uint16Array, b: Uint16Array) {
  let changed = 0, changedDepth = 0, max = 0, squared = 0, invalid = 0;
  for (let i = 0; i < a.length; i++) {
    if (a[i] === b[i]) continue;
    if (i % 4 === 3) { changedDepth++; continue; }
    changed++;
    const delta = Math.abs(half(a[i]) - half(b[i]));
    if (!Number.isFinite(delta)) invalid++; else { max = Math.max(max, delta); squared += delta * delta; }
  }
  return { changedColorComponents: changed, changedDepthPixels: changedDepth, maximumAbsoluteColorError: max,
    rmsColorError: Math.sqrt(squared / (a.length / 4 * 3)), invalid,
    bitIdentical: changed === 0 && changedDepth === 0 };
}
interface Inputs {
  renderer: SparseVoxelDrySceneRenderer; uniforms: GPUBuffer; target: GPUTexture;
  view: SvoDryViewUniformInputs; directory: string; adapterInfo: unknown; validationErrors: string[];
}
export function captureMainKernels(device: GPUDevice) {
  const modules = new WeakMap<GPUShaderModule, string>();
  const records: { pipeline: GPUComputePipeline; descriptor: GPUComputePipelineDescriptor; code: string }[] = [];
  const active = new Set<GPUComputePipeline>();
  const buffers = new Map<string, GPUBuffer>();
  let maps = 0;
  const wrapped = new Proxy(device, { get(target, key) {
    if (key === "createBuffer") return (descriptor: GPUBufferDescriptor) => {
      const buffer = target.createBuffer(descriptor);
      if (descriptor.label) buffers.set(descriptor.label, buffer);
      if (descriptor.label === "Voxel mesh status") {
        const map = buffer.mapAsync.bind(buffer);
        buffer.mapAsync = (...args) => { maps++; return map(...args); };
      }
      return buffer;
    };
    if (key === "createShaderModule") return (descriptor: GPUShaderModuleDescriptor) => {
      const module = target.createShaderModule(descriptor); modules.set(module, descriptor.code); return module;
    };
    if (key === "createComputePipelineAsync") return async (descriptor: GPUComputePipelineDescriptor) => {
      const pipeline = await target.createComputePipelineAsync(descriptor);
      records.push({ pipeline, descriptor, code: modules.get(descriptor.compute.module)! }); return pipeline;
    };
    const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
  } });
  return { device: wrapped, async run(input: Inputs) {
    const { renderer, uniforms, target, view, directory, validationErrors } = input;
    await mkdir(directory, { recursive: true });
    assert.equal(view.scene.sceneId, "backyard-pool-party");
    assert.equal(renderer.surfaceMeshStatus?.state, "ready");
    // The benchmark can control telemetry without adding a product option.
    const internals = renderer as unknown as { surfaceMeshReadbackAt_ms: number; rasterAo: { invalidateVisibility(): void } };
    const readback = device.createBuffer({ size: Math.ceil(view.width * 8 / 256) * 256 * view.height,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const receipt = device.createBuffer({ size: 1024, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const workCounts = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const baseline: MainKernelVariant = { name: "baseline", section: 0, hypothesis: "Shipping mesh + raster AO renderer" };
    const requested = process.env.FLUID_SVO_KERNEL_VARIANTS?.split(",");
    const variants = requested ? mainKernelVariants.filter(v => requested.includes(v.name)) : mainKernelVariants;
    if (requested) assert.equal(variants.length, requested.length, "Unknown experiment");
    const cycles = Number(process.env.FLUID_SVO_KERNEL_CYCLES ?? 32);
    const framesPerSample = Number(process.env.FLUID_SVO_KERNEL_BATCH ?? 1);
    const guardContention = process.env.FLUID_SVO_KERNEL_GUARD === "1";
    const attributedSamples = process.env.FLUID_SVO_KERNEL_ATTRIBUTION === "1";
    assert.ok(Number.isInteger(cycles) && (cycles === 4 || cycles >= 8 && cycles % 8 === 0), "Use four correctness rounds or a multiple of eight timing rounds");
    assert.ok(Number.isInteger(framesPerSample) && framesPerSample >= 1 && framesPerSample <= 8);
    let receiptCopies = 0;
    const receiptTimes = new Map<string, number>();
    const frame = async (variant: MainKernelVariant, replacements: Map<GPUComputePipeline, GPUComputePipeline>, pose: number, options: { capture?: boolean; attributed?: boolean; frames?: number } = {}) => {
      const repeats = options.frames ?? 1;
      const camera = { ...view.camera, azimuth_rad: view.camera.azimuth_rad + [0, .08, -.08, .025][pose % 4] };
      device.queue.writeBuffer(uniforms, 0, packSvoDryViewUniforms({ ...view, camera, cameraMoving: true }));
      renderer.setRasterVisibilityTilingEnabled(variant.tiling === true);
      renderer.setRasterVisibilityQueueEnabled(variant.inlineEdges !== true);
      // Each A/B visits exactly the same view. Force fresh lighting for both;
      // this is the production camera-moving workload, not a cached idle frame.
      internals.rasterAo.invalidateVisibility();
      // Ready-state telemetry assumes one encode per submission. Keep it out
      // of batched kernel timing; section 10 measures its normal cadence.
      if (variant.receipts === "off" || repeats > 1) internals.surfaceMeshReadbackAt_ms = performance.now();
      else internals.surfaceMeshReadbackAt_ms = receiptTimes.get(variant.name) ?? 0;
      const copiesBefore = receiptCopies;
      const encoder = device.createCommandEncoder();
      const recorder = new GPUPassTimestampRecorder(device, 256 * repeats, "Pool party main rendering", options.attributed === true);
      const timed = recorder.instrument(encoder);
      const renderPipelines: string[] = [];
      const selected = new Proxy(timed, { get(target, key) {
        if (key === "clearBuffer" && variant.fusedTileClassification) return (...args: Parameters<GPUCommandEncoder["clearBuffer"]>) => {
          if (args[0] !== buffers.get("Occupied lighting tiles")) target.clearBuffer(...args);
        };
        if (key === "copyBufferToBuffer") return (...args: Parameters<GPUCommandEncoder["copyBufferToBuffer"]>) => {
          if (args[2] === buffers.get("Voxel mesh status")) receiptCopies++;
          return target.copyBufferToBuffer(...args);
        };
        if (key === "beginComputePass") return (descriptor?: GPUComputePassDescriptor) => {
          if (variant.omit?.includes(descriptor?.label ?? "")) return new Proxy({}, { get: () => () => {} });
          if (variant.fusedTileClassification && descriptor?.label === "Normalize visible AO footprint") target.clearBuffer(buffers.get("Occupied lighting tiles")!, 0, 4);
          const pass = target.beginComputePass(descriptor); let entry = "";
          return new Proxy(pass, { get(target, key) {
            if (key === "setPipeline") return (pipeline: GPUComputePipeline) => {
              active.add(pipeline); entry = records.find(r => r.pipeline === pipeline)?.descriptor.compute.entryPoint ?? "";
              target.setPipeline(replacements.get(pipeline) ?? pipeline);
            };
            if (key === "dispatchWorkgroups") return (x: number, y = 1, z = 1) => {
              if (entry === "rasterClassifyTiles" && variant.fusedTileClassification) return;
              if (entry === "rasterAoRangeMain" && variant.rangeShape) { x = Math.ceil(Math.ceil(view.width / 2) / variant.rangeShape[0]); y = Math.ceil(Math.ceil(view.height / 2) / variant.rangeShape[1]); }
              target.dispatchWorkgroups(x, y, z);
            };
            const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
          } });
        };
        if (key === "beginRenderPass") return (descriptor: GPURenderPassDescriptor) => {
          const pass = target.beginRenderPass(descriptor);
          return new Proxy(pass, { get(target, key) {
            if (key === "setPipeline") return (pipeline: GPURenderPipeline) => { renderPipelines.push(pipeline.label); target.setPipeline(pipeline); };
            const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
          } });
        };
        const value = Reflect.get(target, key, target); return typeof value === "function" ? value.bind(target) : value;
      } });
      const started = performance.now();
      for (let repeat = 0; repeat < repeats; repeat++) {
        internals.rasterAo.invalidateVisibility();
        assert.ok(renderer.encode(selected, target));
      }
      if (repeats === 1) receiptTimes.set(variant.name, internals.surfaceMeshReadbackAt_ms);
      const encode_ms = (performance.now() - started) / repeats;
      if (variant.receipts === "sync") assert.ok(renderer.copySurfaceMeshDiagnostics(encoder, receipt));
      recorder.resolve(encoder);
      if (options.capture) encoder.copyTextureToBuffer({ texture: target },
        { buffer: readback, bytesPerRow: Math.ceil(view.width * 8 / 256) * 256 }, [view.width, view.height]);
      if (options.capture) {
        encoder.copyBufferToBuffer(buffers.get("Unrepresented visibility receivers")!, 0, workCounts, 0, 16);
        encoder.copyBufferToBuffer(buffers.get("Occupied lighting tiles")!, 0, workCounts, 16, 16);
      }
      device.queue.submit([encoder.finish()]);
      await device.queue.onSubmittedWorkDone();
      if (variant.receipts === "sync") { await receipt.mapAsync(GPUMapMode.READ); receipt.unmap(); }
      const wall_ms = (performance.now() - started) / repeats;
      const timing = await recorder.read(); assert.ok(timing);
      timing.span_ms /= repeats;
      let pixels: Uint16Array | undefined;
      let work: { exactReceivers: number; occupiedTiles: number | null } | undefined;
      if (options.capture) {
        await workCounts.mapAsync(GPUMapMode.READ);
        const counts = new Uint32Array(workCounts.getMappedRange());
        work = { exactReceivers: counts[0], occupiedTiles: variant.tiling ? counts[4] : null };
        workCounts.unmap();
        await readback.mapAsync(GPUMapMode.READ); const rows = new Uint16Array(readback.getMappedRange());
        pixels = new Uint16Array(view.width * view.height * 4);
        for (let y = 0; y < view.height; y++) pixels.set(rows.subarray(y * Math.ceil(view.width * 8 / 256) * 128,
          y * Math.ceil(view.width * 8 / 256) * 128 + view.width * 4), y * view.width * 4);
        readback.unmap();
      }
      assert.deepEqual(validationErrors, [], variant.name);
      return { timing, encode_ms, wall_ms, pixels, renderPipelines, work,
        diagnosticCopies: receiptCopies - copiesBefore + Number(variant.receipts === "sync") };
    };
    const noReplacements = new Map<GPUComputePipeline, GPUComputePipeline>();
    try {
      for (let i = 0; i < 4; i++) await frame(baseline, noReplacements, i);
      const attribution = await frame(baseline, noReplacements, 0, { attributed: true, capture: true });
      assert.ok(attribution.timing.passes.some(p => p.label === "Normalize visible AO footprint"));
      assert.ok(attribution.timing.passes.some(p => p.label === "Voxel surface mesh rasterization"));
      assert.ok(!attribution.timing.passes.some(p => p.label.includes("brick instance cull")));
      const manifest = { scene: view.scene.sceneId, width: view.width, height: view.height, adapter: input.adapterInfo,
        workload: "Main SVO opaque scene and rigid rendering, shipping mesh + raster AO. Four camera poses; lighting recomputed each frame. Fluid simulation and water composition excluded.",
        framesPerSample, attributedSamples, ordering: "Each pose crosses ABBA/BAAB every four rounds; bootstrap eight-round blocks", mesh: renderer.surfaceMeshStatus, work: attribution.work, renderPipelines: attribution.renderPipelines, attribution: attribution.timing, pipelines: records.filter(r => active.has(r.pipeline)).map(r => ({ label: r.descriptor.label, entry: r.descriptor.compute.entryPoint })) };
      await writeFile(join(directory, "baseline.json"), JSON.stringify(manifest, null, 2));
      await writeFile(join(directory, "baseline.rgba16f"), new Uint8Array(attribution.pixels!.buffer));
      writeFramePng(join(directory, "baseline.png"), { width: view.width, height: view.height, packedRows: new Uint32Array(attribution.pixels!.buffer) });
      for (const entry of ["rasterAoRangeMain", "rasterLightEdgesMain", "rasterExactEdgesMain"])
        await writeFile(join(directory, `${entry}.wgsl`), records.find(r => active.has(r.pipeline) && r.descriptor.compute.entryPoint === entry)!.code);
      console.log(JSON.stringify({ phase: "baseline", mesh: renderer.surfaceMeshStatus, passes: attribution.timing.passes.map(p => ({ label: p.label, ms: p.duration_ms, trusted: p.trusted })) }));
      const report: { manifest: typeof manifest; experiments: unknown[] } = { manifest, experiments: [] };
      for (const variant of variants) {
        const replacements = new Map<GPUComputePipeline, GPUComputePipeline>();
        const matching = records.filter(r => active.has(r.pipeline) && variant.entries?.includes(r.descriptor.compute.entryPoint!));
        if (variant.entries) assert.ok(matching.length, `No active pipeline for ${variant.name}`);
        for (const record of matching) {
          const code = variant.transform!(record.code);
          assert.notEqual(code, record.code, `${variant.name} must change the active shader`);
          const module = device.createShaderModule({ code, label: variant.name });
          const info = await module.getCompilationInfo();
          assert.deepEqual(info.messages.filter(m => m.type === "error"), [], variant.name);
          const pipeline = await device.createComputePipelineAsync({ ...record.descriptor,
            label: `${record.descriptor.label} ${variant.name}`, compute: { ...record.descriptor.compute, module } });
          replacements.set(record.pipeline, pipeline);
        }
        const imageComparisons = [];
        for (let pose = 0; pose < 4; pose++) {
          const a = await frame(baseline, noReplacements, pose, { capture: true });
          const b = await frame(variant, replacements, pose, { capture: true });
          imageComparisons.push({ pose, baselineWork: a.work, candidateWork: b.work, ...compare(a.pixels!, b.pixels!), referenceHash: createHash("sha256").update(new Uint8Array(a.pixels!.buffer)).digest("hex"),
            candidateHash: createHash("sha256").update(new Uint8Array(b.pixels!.buffer)).digest("hex") });
        }
        const samples: { variant: string; pose: number; gpu_ms: number; encode_ms: number; wall_ms: number; diagnosticCopies: number; compute_ms: Record<string, number> }[] = [];
        const mapStart = maps, copyStart = receiptCopies;
        let discardedRounds = 0;
        internals.surfaceMeshReadbackAt_ms = 0;
        for (let round = -2; round < cycles; round++) {
          if (guardContention) {
            let reported = false;
            while (otherGpuJobs().length) {
              if (!reported) console.log(JSON.stringify({ phase: "waiting-for-other-gpu-job", variant: variant.name }));
              reported = true;
              await new Promise(resolve => setTimeout(resolve, 1000));
            }
          }
          const order = (round + Math.floor(round / 4)) % 2 === 0 ? [baseline, variant, variant, baseline] : [variant, baseline, baseline, variant];
          const sampleStart = samples.length;
          let overlapped = false;
          for (const selected of order) {
            const pose = (round + 4) % 4;
            const frames = variant.section === 10 ? 1 : framesPerSample;
            const result = await frame(selected, selected === baseline ? noReplacements : replacements, pose, { frames, attributed: attributedSamples });
            const compute_ms: Record<string, number> = {};
            for (const pass of result.timing.passes) if (pass.trusted) compute_ms[pass.label] = (compute_ms[pass.label] ?? 0) + pass.duration_ms / frames;
            if (round >= 0) samples.push({ variant: selected.name, pose, gpu_ms: result.timing.span_ms, encode_ms: result.encode_ms, wall_ms: result.wall_ms, diagnosticCopies: result.diagnosticCopies, compute_ms });
            if (guardContention && otherGpuJobs().length) overlapped = true;
          }
          if (overlapped) { samples.length = sampleStart; discardedRounds++; round--; }
        }
        const a = samples.filter(s => s.variant === "baseline"), b = samples.filter(s => s.variant !== "baseline");
        const reference = await frame(baseline, noReplacements, 0, { attributed: true });
        const candidate = await frame(variant, replacements, 0, { attributed: true });
        const result = { name: variant.name, section: variant.section, hypothesis: variant.hypothesis,
          pipelines: matching.map(r => r.descriptor.label), imageComparisons,
          exactImageParity: imageComparisons.every(c => c.bitIdentical),
          baseline: { gpu_ms: stats(a.map(s => s.gpu_ms)), cpuEncode_ms: stats(a.map(s => s.encode_ms)), wall_ms: stats(a.map(s => s.wall_ms)) },
          candidate: { gpu_ms: stats(b.map(s => s.gpu_ms)), cpuEncode_ms: stats(b.map(s => s.encode_ms)), wall_ms: stats(b.map(s => s.wall_ms)) },
          gpuRatio: median(b.map(s => s.gpu_ms)) / median(a.map(s => s.gpu_ms)),
          pairedGpuRatio: pairedRatios(samples), framesPerSample: variant.section === 10 ? 1 : framesPerSample,
          receiptMaps: maps - mapStart, receiptCopies: receiptCopies - copyStart, discardedRounds, guardContention,
          attribution: { baseline: reference.timing, candidate: candidate.timing }, samples };
        report.experiments.push(result);
        await writeFile(join(directory, "results.json"), JSON.stringify(report, null, 2));
        console.log(JSON.stringify({ phase: "experiment", name: variant.name, section: variant.section, ratio: result.gpuRatio, parity: result.exactImageParity,
          baseline: result.baseline.gpu_ms.median, candidate: result.candidate.gpu_ms.median, imageComparisons }));
      }
      assert.deepEqual(validationErrors, []);
    } finally { readback.destroy(); receipt.destroy(); workCounts.destroy(); }
  } };
}
