import { pathToFileURL } from "node:url";
import { getScenePreset } from "../lib/scenes";
import { createTallCellLayout, type GPUQuality } from "../lib/tall-cell-grid";
import { activeCubeCapacity, extractionPrepareShader, surfaceExtractionDispatchPlan, surfaceExtractionShader, surfaceVertexCapacity, type SurfaceExtractionDispatchPlan } from "../lib/webgpu-water-pipeline";

const modulePath = process.env.WEBGPU_NODE_MODULE;
if (!modulePath) throw new Error("Set WEBGPU_NODE_MODULE to the installed webgpu package index.js");
const { create, globals } = await import(pathToFileURL(modulePath).href) as {
  create(options: string[]): GPU;
  globals: Record<string, unknown>;
};
Object.assign(globalThis, globals);
const backend = process.env.FLUID_WEBGPU_BACKEND ?? "metal";
const gpu = create([`backend=${backend}`]);

const qualityValue = process.env.FLUID_QUALITY ?? "balanced";
if (!(["balanced", "high", "ultra"] as const).includes(qualityValue as GPUQuality)) throw new Error(`Unknown FLUID_QUALITY=${qualityValue}`);
const quality = qualityValue as GPUQuality;
const iterations = Math.max(10, Math.round(Number(process.env.FLUID_BENCH_ITERATIONS ?? 80)));
const warmups = Math.max(4, Math.round(Number(process.env.FLUID_BENCH_WARMUPS ?? 20)));
const requestedCases = (process.env.FLUID_BENCH_SCENES ?? "water-box-tank-fill,water-box-dam-break,deep-water-ab").split(",").map((value) => value.trim()).filter(Boolean);

type Variant = "full-volume" | "restricted-band";
type Sample = { variant: Variant; pair: number; order: number; duration_ms: number };

function paddedFloatData(source: Float32Array, width: number, height: number, depth: number) {
  const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
  const output = new Uint8Array(bytesPerRow * height * depth);
  for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) {
    const row = new Float32Array(output.buffer, bytesPerRow * (y + height * z), width);
    row.set(source.subarray(width * (y + height * z), width * (y + 1 + height * z)));
  }
  return { data: output, bytesPerRow };
}

function quantile(sorted: readonly number[], q: number) {
  if (sorted.length === 0) return 0;
  const position = Math.max(0, Math.min(sorted.length - 1, (sorted.length - 1) * q));
  const lower = Math.floor(position), upper = Math.ceil(position), fraction = position - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}

function distribution(values: readonly number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const mean = sorted.reduce((sum, value) => sum + value, 0) / Math.max(1, sorted.length);
  const variance = sorted.reduce((sum, value) => sum + (value - mean) ** 2, 0) / Math.max(1, sorted.length - 1);
  const standardDeviation = Math.sqrt(variance);
  return {
    samples: sorted.length,
    minimum_ms: sorted[0] ?? 0,
    p05_ms: quantile(sorted, 0.05),
    median_ms: quantile(sorted, 0.5),
    mean_ms: mean,
    p95_ms: quantile(sorted, 0.95),
    maximum_ms: sorted.at(-1) ?? 0,
    standardDeviation_ms: standardDeviation,
    coefficientOfVariation: mean > 0 ? standardDeviation / mean : 0
  };
}

function bootstrapMedianSpeedup(pairs: ReadonlyArray<{ full: number; restricted: number }>, resamples = 2000) {
  let state = 0x9e3779b9;
  const randomIndex = () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return Math.floor(state / 0x1_0000_0000 * pairs.length);
  };
  const ratios: number[] = [];
  for (let sample = 0; sample < resamples; sample += 1) {
    const full: number[] = [], restricted: number[] = [];
    for (let index = 0; index < pairs.length; index += 1) {
      const pair = pairs[randomIndex()]; full.push(pair.full); restricted.push(pair.restricted);
    }
    full.sort((a, b) => a - b); restricted.sort((a, b) => a - b);
    ratios.push(quantile(full, 0.5) / Math.max(1e-12, quantile(restricted, 0.5)));
  }
  ratios.sort((a, b) => a - b);
  return { lower95: quantile(ratios, 0.025), upper95: quantile(ratios, 0.975), resamples };
}

function workgroupInvocations(plan: SurfaceExtractionDispatchPlan) {
  if (plan.full) return plan.full[0] * plan.full[1] * plan.full[2] * 64;
  return (plan.band?.[0] ?? 0) * (plan.band?.[1] ?? 0) * (plan.band?.[2] ?? 0) * 64
    + (plan.tallSides?.[0] ?? 0) * (plan.tallSides?.[1] ?? 0) * 64
    + (plan.walls?.[0] ?? 0) * (plan.walls?.[1] ?? 0) * 64;
}

interface ProductionStages {
  prepare: GPUComputePipeline;
  prepareBindGroup: GPUBindGroup;
  polygonise: GPUComputePipeline;
  dispatchArgs: GPUBuffer;
}

// Production timing covers the complete extraction sequence: classify sweep,
// worklist compaction, and the indirect polygonise dispatch. Count-only runs
// (production omitted) total triangles during classification instead.
function encodeVariant(pass: GPUComputePassEncoder, variant: Variant, plan: SurfaceExtractionDispatchPlan, pipelines: Record<"full" | "band" | "tallSides" | "walls", GPUComputePipeline>, bindGroup: GPUBindGroup, production?: ProductionStages) {
  pass.setBindGroup(0, bindGroup);
  if (variant === "restricted-band") {
    pass.setPipeline(pipelines.band); pass.dispatchWorkgroups(...plan.band!);
    pass.setPipeline(pipelines.tallSides); pass.dispatchWorkgroups(...plan.tallSides!);
    pass.setPipeline(pipelines.walls); pass.dispatchWorkgroups(...plan.walls!);
  } else {
    pass.setPipeline(pipelines.full); pass.dispatchWorkgroups(...plan.full!);
  }
  if (!production) return;
  pass.setPipeline(production.prepare); pass.setBindGroup(0, production.prepareBindGroup); pass.dispatchWorkgroups(1);
  pass.setPipeline(production.polygonise); pass.setBindGroup(0, bindGroup); pass.dispatchWorkgroupsIndirect(production.dispatchArgs, 0);
}

async function readCount(buffer: GPUBuffer) {
  await buffer.mapAsync(GPUMapMode.READ);
  const count = new Uint32Array(buffer.getMappedRange())[0];
  buffer.unmap();
  return count;
}

const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
if (!adapter) throw new Error("Dawn did not expose a WebGPU adapter");
if (!adapter.features.has("timestamp-query")) throw new Error("Accurate water extraction benchmarking requires the timestamp-query feature");
const adapterInfo = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
const device = await adapter.requestDevice({ requiredFeatures: ["timestamp-query"] });
const validationErrors: string[] = [];
device.addEventListener("uncapturederror", (event) => validationErrors.push(event.error.message));

console.log(JSON.stringify({
  benchmark: "water-surface-extraction",
  phase: "environment",
  backend,
  quality,
  warmups,
  iterations,
  adapter: adapterInfo ? { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, device: adapterInfo.device, description: adapterInfo.description } : undefined,
  timestampUnit: "GPU timestamp-query nanoseconds converted to milliseconds"
}));

try {
  for (const sceneId of requestedCases) {
    const preset = getScenePreset(sceneId);
    if (preset.id !== sceneId) throw new Error(`Unknown FLUID_BENCH_SCENES entry ${sceneId}`);
    const scene = preset.create();
    const layout = createTallCellLayout(scene, quality, device.limits.maxTextureDimension3D);
    if (layout.planning.ordinaryGridFallback) throw new Error(`${sceneId} does not produce a restricted tall-cell layout at ${quality}`);
    const { nx, fineNy: ny, nz, packedNy } = layout;
    const maximumNeighborDelta = layout.settings.maximumNeighborDelta;
    const fullPlan = surfaceExtractionDispatchPlan(nx, ny, nz, packedNy, false, maximumNeighborDelta);
    const restrictedPlan = surfaceExtractionDispatchPlan(nx, ny, nz, packedNy, true, maximumNeighborDelta);

    const uniformBuffer = device.createBuffer({ size: 28 * 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const uniforms = new Float32Array(28);
    uniforms.set([scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.height_m * scene.container.fillFraction], 12);
    uniforms.set([nx, ny, nz, 2], 20);
    device.queue.writeBuffer(uniformBuffer, 0, uniforms);

    const volume = device.createTexture({ label: `${sceneId} packed volume`, size: [nx, packedNy, nz], dimension: "3d", format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const packedVolume = paddedFloatData(layout.initialVolume, nx, packedNy, nz);
    device.queue.writeTexture({ texture: volume }, packedVolume.data, { bytesPerRow: packedVolume.bytesPerRow, rowsPerImage: packedNy }, { width: nx, height: packedNy, depthOrArrayLayers: nz });
    const bases = device.createTexture({ label: `${sceneId} column bases`, size: [nx, nz], format: "r32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const packedBases = paddedFloatData(layout.columnBases, nx, nz, 1);
    device.queue.writeTexture({ texture: bases }, packedBases.data, { bytesPerRow: packedBases.bytesPerRow, rowsPerImage: nz }, { width: nx, height: nz });

    const maximumVertices = surfaceVertexCapacity(nx, ny, nz);
    const vertices = device.createBuffer({ size: maximumVertices * 32, usage: GPUBufferUsage.STORAGE });
    const indirect = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    const activeCubes = device.createBuffer({ size: activeCubeCapacity(maximumVertices) * 8, usage: GPUBufferUsage.STORAGE });
    const meta = device.createBuffer({ size: 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const dispatchArgs = device.createBuffer({ size: 12, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT });
    const extractionModule = device.createShaderModule({ label: "Water extraction benchmark", code: surfaceExtractionShader });
    const prepareModule = device.createShaderModule({ label: "Water extraction prepare benchmark", code: extractionPrepareShader });
    for (const shaderModule of [extractionModule, prepareModule]) {
      const compilation = await shaderModule.getCompilationInfo();
      const compilationErrors = compilation.messages.filter((message) => message.type === "error");
      if (compilationErrors.length) throw new Error(compilationErrors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`).join("\n"));
    }
    const bindGroupLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float", viewDimension: "3d" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: "unfilterable-float" } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    const prepareLayout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } }
    ] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });
    const pipelines = {
      full: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractMain" } }),
      band: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractBandMain" } }),
      tallSides: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractTallSidesMain" } }),
      walls: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractWallMain" } })
    };
    const countPipelines = {
      full: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractMain", constants: { countOnly: 1 } } }),
      band: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractBandMain", constants: { countOnly: 1 } } }),
      tallSides: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractTallSidesMain", constants: { countOnly: 1 } } }),
      walls: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "extractWallMain", constants: { countOnly: 1 } } })
    };
    const bindGroup = device.createBindGroup({ layout: bindGroupLayout, entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: volume.createView({ dimension: "3d" }) },
      { binding: 2, resource: bases.createView() },
      { binding: 3, resource: { buffer: vertices } },
      { binding: 4, resource: { buffer: indirect } },
      { binding: 5, resource: { buffer: activeCubes } },
      { binding: 6, resource: { buffer: meta } }
    ] });
    const production: ProductionStages = {
      prepare: device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [prepareLayout] }), compute: { module: prepareModule, entryPoint: "prepareMain" } }),
      prepareBindGroup: device.createBindGroup({ layout: prepareLayout, entries: [
        { binding: 0, resource: { buffer: meta } },
        { binding: 1, resource: { buffer: activeCubes } },
        { binding: 2, resource: { buffer: dispatchArgs } }
      ] }),
      polygonise: device.createComputePipeline({ layout: pipelineLayout, compute: { module: extractionModule, entryPoint: "polygoniseMain" } }),
      dispatchArgs
    };

    const warmupEncoder = device.createCommandEncoder({ label: `${sceneId} extraction warm-up` });
    for (let index = 0; index < warmups; index += 1) for (const variant of (index % 2 === 0 ? ["full-volume", "restricted-band"] : ["restricted-band", "full-volume"]) as Variant[]) {
      warmupEncoder.clearBuffer(indirect);
      warmupEncoder.clearBuffer(meta);
      const pass = warmupEncoder.beginComputePass();
      encodeVariant(pass, variant, variant === "full-volume" ? fullPlan : restrictedPlan, pipelines, bindGroup, production);
      pass.end();
    }
    device.queue.submit([warmupEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    const querySet = device.createQuerySet({ type: "timestamp", count: iterations * 4 });
    const queryResolve = device.createBuffer({ size: iterations * 4 * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const queryReadback = device.createBuffer({ size: iterations * 4 * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const samples: Sample[] = [];
    const timedEncoder = device.createCommandEncoder({ label: `${sceneId} alternating extraction samples` });
    let query = 0;
    const sampleOrder: Array<{ variant: Variant; pair: number; order: number; start: number; end: number }> = [];
    for (let pair = 0; pair < iterations; pair += 1) {
      const variants = (pair % 2 === 0 ? ["full-volume", "restricted-band"] : ["restricted-band", "full-volume"]) as Variant[];
      for (let order = 0; order < variants.length; order += 1) {
        const variant = variants[order], start = query++, end = query++;
        timedEncoder.clearBuffer(indirect);
        timedEncoder.clearBuffer(meta);
        const pass = timedEncoder.beginComputePass({ timestampWrites: { querySet, beginningOfPassWriteIndex: start, endOfPassWriteIndex: end } });
        encodeVariant(pass, variant, variant === "full-volume" ? fullPlan : restrictedPlan, pipelines, bindGroup, production);
        pass.end();
        sampleOrder.push({ variant, pair, order, start, end });
      }
    }
    timedEncoder.resolveQuerySet(querySet, 0, query, queryResolve, 0);
    timedEncoder.copyBufferToBuffer(queryResolve, 0, queryReadback, 0, query * 8);
    device.queue.submit([timedEncoder.finish()]);
    await queryReadback.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(queryReadback.getMappedRange());
    for (const item of sampleOrder) samples.push({ ...item, duration_ms: Number(timestamps[item.end] - timestamps[item.start]) / 1e6 });
    queryReadback.unmap();

    const fullCountReadback = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const restrictedCountReadback = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const validationEncoder = device.createCommandEncoder({ label: `${sceneId} extraction equivalence` });
    validationEncoder.clearBuffer(indirect);
    let pass = validationEncoder.beginComputePass(); encodeVariant(pass, "full-volume", fullPlan, countPipelines, bindGroup); pass.end();
    validationEncoder.copyBufferToBuffer(indirect, 0, fullCountReadback, 0, 4);
    validationEncoder.clearBuffer(indirect);
    pass = validationEncoder.beginComputePass(); encodeVariant(pass, "restricted-band", restrictedPlan, countPipelines, bindGroup); pass.end();
    validationEncoder.copyBufferToBuffer(indirect, 0, restrictedCountReadback, 0, 4);
    device.queue.submit([validationEncoder.finish()]);
    const fullVertexCount = await readCount(fullCountReadback), restrictedVertexCount = await readCount(restrictedCountReadback);

    const fullSamples = samples.filter((sample) => sample.variant === "full-volume").map((sample) => sample.duration_ms);
    const restrictedSamples = samples.filter((sample) => sample.variant === "restricted-band").map((sample) => sample.duration_ms);
    const pairs = Array.from({ length: iterations }, (_, pair) => ({
      full: samples.find((sample) => sample.pair === pair && sample.variant === "full-volume")!.duration_ms,
      restricted: samples.find((sample) => sample.pair === pair && sample.variant === "restricted-band")!.duration_ms
    }));
    const fullDistribution = distribution(fullSamples), restrictedDistribution = distribution(restrictedSamples);
    const perimeterCubes = 2 * (nx + 1) + 2 * (nz - 1);
    const fullLogicalCandidates = (nx + 1) * (ny + 1) * (nz + 1);
    let expandedTallSideCubes = 0;
    for (let z = 1; z < nz; z += 1) for (let x = 1; x < nx; x += 1) {
      const columns = [x - 1 + nx * (z - 1), x + nx * (z - 1), x - 1 + nx * z, x + nx * z];
      const minimumBase = Math.min(...columns.map((column) => layout.columnBases[column]));
      if (minimumBase <= 0) continue;
      const tallValues = columns.map((column) => layout.initialVolume[column % nx + nx * (packedNy * Math.floor(column / nx))]);
      if (Math.min(...tallValues) < 0.5 && Math.max(...tallValues) >= 0.5) expandedTallSideCubes += minimumBase;
    }
    const restrictedLogicalCandidates = (nx - 1) * (restrictedPlan.bandCubeRows ?? 0) * (nz - 1) + perimeterCubes * (ny + 1) + expandedTallSideCubes;
    console.log(JSON.stringify({
      benchmark: "water-surface-extraction",
      phase: "result",
      sceneId,
      quality,
      grid: { nx, ny, nz, packedNy, regularLayers: layout.settings.regularLayers, maximumNeighborDelta },
      dispatch: {
        full: fullPlan,
        restricted: restrictedPlan,
        fullLogicalCandidates,
        restrictedLogicalCandidates,
        expandedTallSideCubes,
        logicalCandidateReduction: 1 - restrictedLogicalCandidates / fullLogicalCandidates,
        fullWorkgroupInvocations: workgroupInvocations(fullPlan),
        restrictedWorkgroupInvocations: workgroupInvocations(restrictedPlan),
        workgroupInvocationReduction: 1 - workgroupInvocations(restrictedPlan) / workgroupInvocations(fullPlan)
      },
      correctness: { fullVertexCount, restrictedVertexCount, countsMatch: fullVertexCount === restrictedVertexCount, capacityVertices: maximumVertices, productionCapacityWouldClip: fullVertexCount > maximumVertices },
      timing: {
        full: fullDistribution,
        restricted: restrictedDistribution,
        medianSpeedup: fullDistribution.median_ms / Math.max(1e-12, restrictedDistribution.median_ms),
        medianSpeedupConfidence95: bootstrapMedianSpeedup(pairs)
      },
      rawSamples: samples,
      validationErrors
    }));
    if (fullVertexCount !== restrictedVertexCount) throw new Error(`${sceneId}: restricted extraction emitted ${restrictedVertexCount} vertices; full extraction emitted ${fullVertexCount}`);
    if (validationErrors.length) throw new Error(`${sceneId}: WebGPU validation errors: ${validationErrors.join("; ")}`);

    for (const resource of [fullCountReadback, restrictedCountReadback, queryReadback, queryResolve, querySet, indirect, vertices, activeCubes, meta, dispatchArgs, bases, volume, uniformBuffer]) resource.destroy();
  }
} finally {
  device.destroy();
}
