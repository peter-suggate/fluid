/** Restricted-map feasibility gate. This is not a production solver selector.
 * CPU compiles geometry, Dawn clips raw overlaps and gathers one final state.
 */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdir, rename, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createProcessRetainedDawnGPU, type NodeDawnProvider } from "../lib/harness/node-dawn-provider";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from "../lib/harness/webgpu-smoke-isolation";
import { fingerprintSparseCM12RepositorySources } from "./sparse-cm12-source-content-fingerprint";
import { compileMap, inverseMapReference, mapPoint, translatedReference, type MapSpec, type Shear, type V3 } from "../lib/core/geometric-remap/geometry";
import { remapOverlapWGSL } from "../lib/core/geometric-remap/overlap.wgsl";

const argument = (name: string, fallback: string) => process.argv.slice(2)
  .find(v => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
if (process.argv.includes("--help")) {
  console.log(`Large-step geometric remap feasibility probe (Dawn, periodic all-fine cells)
  --size=8         Cells on each axis (4 or 8)
  --case=all       Exact case id, or all
  --list           List case ids without acquiring the GPU
  --out=PATH       Atomic receipt; default artifacts/analytic-motion/large-remap.json
CPU compiles piecewise affine maps. Dawn integrates raw mapped tetrahedra and
gathers receivers once. No state clamp, normalization or redistribution.
The corner-only control MUST fail compatibility. A passing suite includes that
detected failure, and does not establish the production velocity bridge (C3).`);
  process.exit(0);
}
const size = Number(argument("size", "8"));
assert.ok(size === 4 || size === 8, "size must be 4 or 8");
const baseShears = (knots: number): Shear[] => [
  { axis: 0, dependent: 1, amplitude: 0.65, knots },
  { axis: 1, dependent: 2, amplitude: 0.55, knots },
  { axis: 2, dependent: 0, amplitude: 0.75, knots },
];
interface Case { id: string; spec: MapSpec; full?: boolean; expectRejected?: boolean; identity?: boolean }
const cases: Case[] = [
  ...[0.5, 2.5, 8.5, 25.5].flatMap(co => [
    { id: `translation-axis-${co}`, spec: { size, shift: [co, 0, 0] as V3, shears: [] } },
    { id: `translation-diagonal-${co}`, spec: { size, shift: [co, 0.375, -0.125] as V3, shears: [] } },
  ]),
  { id: "translation-integer-25", spec: { size, shift: [25, 0, 0], shears: [] } },
  { id: "nonlinear-single-shear", spec: { size, shift: [25.5, 0.375, -0.125], shears: baseShears(size).slice(0, 1) } },
  ...[size, size * 2].map(knots => ({ id: `nonlinear-composition-${knots}`,
    spec: { size, shift: [25.5, 0.375, -0.125] as V3, shears: baseShears(knots) } })),
  { id: "nonlinear-composition-full", full: true,
    spec: { size, shift: [25.5, 0.375, -0.125], shears: baseShears(size) } },
  { id: "nonlinear-inverse-geometry", identity: true,
    spec: { size, shift: [0, 0, 0], shears: [...baseShears(size),
      ...baseShears(size).reverse().map(s => ({ ...s, amplitude: -s.amplitude }))] } },
  // This grid/phase exposed a duplicated coplanar cap that the 8^3 case missed.
  { id: "inverse-coplanar-4-regression", identity: true,
    spec: { size: 4, shift: [0, 0, 0], shears: [...baseShears(4),
      ...baseShears(4).reverse().map(s => ({ ...s, amplitude: -s.amplitude }))] } },
  { id: "nonlinear-corner-only-control", expectRejected: true,
    spec: { size, shift: [25.5, 0.375, -0.125], shears: baseShears(size), cornerOnly: true } },
];
if (process.argv.includes("--list")) { console.log(cases.map(c => c.id).join("\n")); process.exit(0); }
const selection = argument("case", "all");
const selected = cases.filter(c => selection === "all" || c.id === selection);
assert.ok(selected.length, `unknown case: ${selection}`);
const output = resolve(argument("out", "artifacts/analytic-motion/large-remap.json"));
const root = fileURLToPath(new URL("..", import.meta.url));
// Fixed before the first candidate execution, in unit-cell volumes. These are
// probe-specific f32 integration budgets, not relaxed production tolerances.
const criteria = {
  cpuMappedDonorVolume: 1e-10, localVolume: 2e-5, relativeTotalVolume: 2e-6,
  analyticCellVolume: 2e-5, negativeControlMinimumDefect: 1e-3,
  // Midpoint quadrature crosses discontinuous interfaces. These bounds are
  // separate from the exact geometry/f32 budgets; retain both 16^3 and 32^3
  // references so their own discretization error is visible.
  quadratureMaximumCellError: 0.02, quadratureMeanCellError: 0.002,
};
const fingerprint = await fingerprintSparseCM12RepositorySources(root);
const report: Record<string, unknown> & { cases: Record<string, unknown>[] } = {
  probe: "geometric-large-remap-dawn", status: "initializing", passed: false,
  scope: "prescribed piecewise affine maps; CPU construction; GPU overlap integration and gather",
  productionTransportChanged: false, productionVelocityBridge: "not implemented",
  topology: { size, cells: size ** 3, unitCellCapacity: 1, periodic: true, adaptive: false, solids: false },
  criteria, sourceFingerprint: fingerprint,
  commit: execFileSync("git", ["rev-parse", "HEAD"], { encoding: "utf8", cwd: root }).trim(),
  cases: [],
};
async function checkpoint() {
  await mkdir(dirname(output), { recursive: true });
  const temporary = `${output}.tmp-${process.pid}`;
  await writeFile(temporary, `${JSON.stringify(report, null, 2)}\n`);
  await rename(temporary, output);
}
const sum = (xs: ArrayLike<number>) => Array.from(xs).reduce((a, b) => a + b, 0);
const maxError = (xs: ArrayLike<number>, expected: ArrayLike<number> | number) => Array.from(xs)
  .reduce((error, value, i) => Math.max(error, Math.abs(value - (typeof expected === "number" ? expected : expected[i]!))), 0);

// Independent smooth-flow approximation oracle: trigonometric shears evaluated
// directly, with no polygon operations or piecewise-linear interpolation.
function approximationError(spec: MapSpec): number {
  let maximum = 0;
  for (let i = 0; i < 1000; i++) {
    const p: V3 = [((i * 0.61803398875) % 1) * size,
      ((i * 0.41421356237 + 0.13) % 1) * size, ((i * 0.73205080757 + 0.27) % 1) * size];
    const exact = [...p] as V3;
    for (const s of spec.shears) exact[s.axis]! += s.amplitude * Math.sin(2 * Math.PI * exact[s.dependent]! / size);
    const approximate = mapPoint(p, spec);
    maximum = Math.max(maximum, Math.hypot(...exact.map((x, k) => x + spec.shift[k]! - approximate[k]!)));
  }
  return maximum;
}

let device: GPUDevice | undefined, locked = false;
const validationErrors: string[] = [];
await checkpoint();
try {
  await acquireWebGPUExclusiveLock("dawn-probe", "geometric-large-remap"); locked = true;
  const modulePath = process.env.WEBGPU_NODE_MODULE ?? resolve(root, "node_modules/webgpu/index.js");
  const dawn = await import(pathToFileURL(resolve(modulePath)).href) as NodeDawnProvider;
  Object.assign(globalThis, dawn.globals);
  const gpu = createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "Dawn adapter unavailable");
  const timestamps = adapter.features.has("timestamp-query");
  device = await adapter.requestDevice({ requiredFeatures: timestamps ? ["timestamp-query"] : [] });
  device.addEventListener("uncapturederror", e => validationErrors.push(e.error.message));
  report.adapter = { vendor: adapter.info.vendor, architecture: adapter.info.architecture,
    device: adapter.info.device, description: adapter.info.description, timestamps };
  const compileStart = performance.now();
  const module = device.createShaderModule({ code: remapOverlapWGSL });
  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter(m => m.type === "error").map(m => m.message), [], "WGSL compilation failed");
  const layout = device.createBindGroupLayout({ entries: [
    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
  ] });
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
  const intersect = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "intersect" } });
  const gather = await device.createComputePipelineAsync({ layout: pipelineLayout, compute: { module, entryPoint: "gather" } });
  report.pipelineCompilationMs = performance.now() - compileStart;
  report.status = "running";
  for (const c of selected) {
    const wallStart = performance.now();
    console.error(`Compiling and executing ${c.id}`);
    const map = compileMap(c.spec, c.full);
    const overlaps = map.overlaps, count = overlaps.length, cells = c.spec.size ** 3;
    const packed = new Float32Array(16 * count), ranges = new Uint32Array(2 * cells);
    for (let i = 0; i < count; i++) {
      const record = overlaps[i]!;
      record.vertices.forEach((v, k) => packed.set([...v.p, v.liquid], 16 * i + 4 * k));
      if (i === 0 || overlaps[i - 1]!.receiver !== record.receiver) ranges[2 * record.receiver] = i;
      ranges[2 * record.receiver + 1] = i + 1;
    }
    const cpuMapAndPackingMs = performance.now() - wallStart;
    const buffers: GPUBuffer[] = [];
    const buffer = (label: string, bytes: number, usage: GPUBufferUsageFlags) => {
      const b = device!.createBuffer({ label, size: bytes, usage }); buffers.push(b); return b;
    };
    const input = buffer("Mapped tetrahedron/cell candidates", packed.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const raw = buffer("Raw capacity and liquid overlaps", count * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const rangeBuffer = buffer("Receiver segments", ranges.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const received = buffer("Candidate final state", cells * 8, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    const readback = buffer("Geometry receipts", count * 8 + cells * 8, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    const querySet = timestamps ? device.createQuerySet({ type: "timestamp", count: 4 }) : undefined;
    const queryResolve = timestamps ? buffer("Timestamp resolve", 32, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC) : undefined;
    const queryRead = timestamps ? buffer("Timestamp read", 32, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST) : undefined;
    try {
      device.queue.writeBuffer(input, 0, packed);
      device.queue.writeBuffer(rangeBuffer, 0, ranges);
      const group = device.createBindGroup({ layout, entries: [input, raw, rangeBuffer, received]
        .map((b, binding) => ({ binding, resource: { buffer: b } })) });
      const encode = (capture: boolean) => {
        const encoder = device!.createCommandEncoder();
        for (const [i, pipeline] of [intersect, gather].entries()) {
          const pass = encoder.beginComputePass({ timestampWrites: capture && querySet ? {
            querySet, beginningOfPassWriteIndex: 2 * i, endOfPassWriteIndex: 2 * i + 1,
          } : undefined });
          pass.setPipeline(pipeline); pass.setBindGroup(0, group);
          pass.dispatchWorkgroups(Math.ceil((i === 0 ? count : cells) / (i === 0 ? 32 : 64)));
          pass.end();
        }
        if (capture) {
          encoder.copyBufferToBuffer(raw, 0, readback, 0, count * 8);
          encoder.copyBufferToBuffer(received, 0, readback, count * 8, cells * 8);
          if (querySet) {
            encoder.resolveQuerySet(querySet, 0, 4, queryResolve!, 0);
            encoder.copyBufferToBuffer(queryResolve!, 0, queryRead!, 0, 32);
          }
        }
        return encoder.finish();
      };
      // Warmup repeats the immutable-input candidate, not successive timesteps.
      device.queue.submit([encode(false)]); await device.queue.onSubmittedWorkDone();
      const submitStart = performance.now();
      device.queue.submit([encode(true)]);
      await readback.mapAsync(GPUMapMode.READ);
      const submitReadbackMs = performance.now() - submitStart;
      const transportAndWarmupWallMs = performance.now() - wallStart;
      const result = new Float32Array(readback.getMappedRange());
      const donorBulk = new Float64Array(cells), donorLiquid = new Float64Array(cells);
      const receiverBulk = new Float64Array(cells), receiverLiquid = new Float64Array(cells);
      let maximumRawViolation = 0, nonfinite = 0, positiveOverlaps = 0;
      for (let i = 0; i < count; i++) {
        const capacity = result[2 * i]!, liquid = result[2 * i + 1]!;
        if (!Number.isFinite(capacity) || !Number.isFinite(liquid)) nonfinite++;
        maximumRawViolation = Math.max(maximumRawViolation, -capacity, -liquid, liquid - capacity);
        if (capacity > 0) positiveOverlaps++;
        donorBulk[overlaps[i]!.donor]! += capacity;
        donorLiquid[overlaps[i]!.donor]! += liquid;
        receiverBulk[overlaps[i]!.receiver]! += capacity;
        receiverLiquid[overlaps[i]!.receiver]! += liquid;
      }
      const finalBulk = new Float64Array(cells), finalLiquid = new Float64Array(cells);
      for (let i = 0; i < cells; i++) {
        finalBulk[i] = result[count * 2 + i * 2]!;
        finalLiquid[i] = result[count * 2 + i * 2 + 1]!;
      }
      const totalBefore = sum(map.initialLiquid), totalAfter = sum(finalLiquid);
      const expected = !c.spec.shears.length || c.identity
        ? translatedReference({ ...c.spec, shears: [] }, !!c.full) : undefined;
      let shapeOracle: Record<string, unknown> | null = null;
      let shapePassed = true;
      if (c.spec.shears.length && !c.full && !c.identity && !c.expectRejected) {
        const oracleStart = performance.now();
        const coarse = inverseMapReference(c.spec, 16), fine = inverseMapReference(c.spec, 32);
        const meanError = sum(Array.from(finalLiquid, (v, i) => Math.abs(v - fine[i]!))) / cells;
        const maximumError = maxError(finalLiquid, fine);
        shapePassed = maximumError <= criteria.quadratureMaximumCellError
          && meanError <= criteria.quadratureMeanCellError;
        shapeOracle = { kind: "independent inverse-map midpoint quadrature",
          samplesPerCell: [16 ** 3, 32 ** 3], maximumCellError: maximumError, meanCellError: meanError,
          maximumCoarseFineDifference: maxError(coarse, fine),
          meanCoarseFineDifference: sum(coarse.map((v, i) => Math.abs(v - fine[i]!))) / cells,
          cpuMs: performance.now() - oracleStart, passed: shapePassed };
      }
      const metrics = {
        maximumCpuMappedDonorError: maxError(map.donorMappedVolumes, 1),
        maximumDonorCoverageError: maxError(donorBulk, 1),
        maximumDonorLiquidError: maxError(donorLiquid, map.initialLiquid),
        maximumReceiverCapacityError: maxError(finalBulk, 1),
        maximumReceiverCapacityErrorF64Reduction: maxError(receiverBulk, 1),
        absoluteDonorCoverageError: sum(Array.from(donorBulk, v => Math.abs(v - 1))),
        absoluteReceiverCapacityError: sum(Array.from(finalBulk, v => Math.abs(v - 1))),
        maximumGatherError: Math.max(maxError(finalBulk, receiverBulk), maxError(finalLiquid, receiverLiquid)),
        maximumBoundViolation: Array.from(finalLiquid).reduce((e, v) => Math.max(e, -v, v - 1), 0),
        maximumRawViolation, nonfinite, totalBefore, totalAfter,
        relativeLiquidVolumeError: Math.abs(totalAfter - totalBefore) / totalBefore,
        maximumAnalyticCellError: expected ? maxError(finalLiquid, expected) : null,
        maximumSmoothMapPositionError: c.spec.shears.length && !c.identity ? approximationError(c.spec) : null,
      };
      const checks = {
        cpuMapPreservesVolume: metrics.maximumCpuMappedDonorError <= criteria.cpuMappedDonorVolume,
        donorCoverage: metrics.maximumDonorCoverageError <= criteria.localVolume,
        donorLiquid: metrics.maximumDonorLiquidError <= criteria.localVolume,
        receiverCapacity: metrics.maximumReceiverCapacityError <= criteria.localVolume,
        bounds: metrics.maximumBoundViolation <= criteria.localVolume,
        rawBounds: metrics.maximumRawViolation <= criteria.localVolume,
        finite: metrics.nonfinite === 0,
        conservation: metrics.relativeLiquidVolumeError <= criteria.relativeTotalVolume,
        analytic: metrics.maximumAnalyticCellError === null || metrics.maximumAnalyticCellError <= criteria.analyticCellVolume,
        independentShape: shapePassed,
      };
      const compatible = Object.values(checks).every(Boolean);
      const passed = c.expectRejected ? !compatible && checks.finite && checks.rawBounds
        && checks.receiverCapacity && metrics.maximumCpuMappedDonorError >= criteria.negativeControlMinimumDefect
        && metrics.maximumDonorCoverageError >= criteria.negativeControlMinimumDefect : compatible;
      let gpu: { intersectMs: number; gatherMs: number } | null = null;
      if (queryRead) {
        await queryRead.mapAsync(GPUMapMode.READ);
        const ticks = new BigUint64Array(queryRead.getMappedRange());
        gpu = { intersectMs: Number(ticks[1]! - ticks[0]!) / 1e6, gatherMs: Number(ticks[3]! - ticks[2]!) / 1e6 };
      }
      report.cases.push({ id: c.id, spec: c.spec, full: !!c.full,
        expectedRejection: !!c.expectRejected, compatible, passed, checks, metrics, shapeOracle,
        execution: { requestedMapApplications: 1, executedMapApplications: 1, volumeUpdates: 1,
          geometryShears: c.spec.shears.length, warmupExecutions: 1,
          timeInterpretation: "one prescribed map; no production simulation clock",
          inverseInterpretation: c.identity ? "forward and inverse composed on retained geometry before one remap" : null },
        work: { pieces: map.pieceCount, maximumPiecesPerCell: map.maximumPiecesPerCell,
          tetrahedra: map.tetrahedronCount, candidateOverlaps: count, positiveOverlaps,
          gpuBufferBytes: buffers.reduce((s, b) => s + b.size, 0),
          cpuGeometryMs: map.compileMs, cpuMapAndPackingMs, transportAndWarmupWallMs,
          submitAndReadbackMs: submitReadbackMs, gpu,
          timingScope: "warm GPU timestamps; wall includes uploads, warmup and QA readback; totalProbeCaseMs also includes CPU shape oracles",
          totalProbeCaseMs: performance.now() - wallStart },
      });
      console.error(JSON.stringify({ case: c.id, compatible, passed,
        donorError: metrics.maximumDonorCoverageError, capacityError: metrics.maximumReceiverCapacityError,
        gpu, pieces: map.pieceCount, overlaps: count }));
    } finally {
      for (const b of buffers) { if (b.mapState === "mapped") b.unmap(); b.destroy(); }
      querySet?.destroy();
    }
    await checkpoint();
  }
  const coarse = report.cases.find(c => c.id === `nonlinear-composition-${size}`);
  const fine = report.cases.find(c => c.id === `nonlinear-composition-${2 * size}`);
  const error = (c: Record<string, unknown>) => (c.metrics as { maximumSmoothMapPositionError: number }).maximumSmoothMapPositionError;
  const mapConvergence = coarse && fine ? { coarse: error(coarse), fine: error(fine),
    ratio: error(coarse) / error(fine), passed: error(fine) < error(coarse) / 2 } : null;
  report.mapConvergence = mapConvergence;
  const translations = report.cases.filter(c => String(c.id).startsWith("translation-diagonal-"));
  const counts = translations.map(c => (c.work as { candidateOverlaps: number }).candidateOverlaps);
  const translationScaling = { counts, passed: new Set(counts).size <= 1 };
  report.translationScaling = translationScaling;
  report.passed = report.cases.every(c => c.passed) && validationErrors.length === 0
    && (mapConvergence === null || mapConvergence.passed) && translationScaling.passed;
  report.status = report.passed ? "complete" : "failed-criterion";
} catch (e) {
  report.status = "failed-runtime";
  report.failure = e instanceof Error ? e.stack : String(e);
} finally {
  if (device) { await device.queue.onSubmittedWorkDone(); device.destroy(); }
  if (locked) await releaseWebGPUExclusiveLock();
  report.validationErrors = validationErrors;
  report.sourceFingerprintAfter = await fingerprintSparseCM12RepositorySources(root);
  report.sourceUnchanged = (report.sourceFingerprintAfter as { sha256: string }).sha256 === fingerprint.sha256;
  if (!report.sourceUnchanged) report.passed = false;
  await checkpoint();
}
console.log(JSON.stringify({ passed: report.passed, status: report.status, receipt: output,
  failure: report.failure, cases: report.cases.length, mapConvergence: report.mapConvergence }));
if (report.passed !== true) process.exitCode = 1;
