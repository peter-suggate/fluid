/** One bounded full-resolution solver run, no renderer or parameter sweep.
 * CPU hypotheses/tests: tests/reservoir-particle-cost.test.ts.
 * Run under tools/gpu/fluid-gpu-run.sh timing. Writes incremental JSONL so
 * even a construction timeout retains evidence. Split timestamps diagnose
 * kernels; their frame wall time is NOT an uninstrumented timing result. */
import assert from "node:assert/strict";
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import "../lib/methods";
import { getSceneDefinition } from "../lib/core/scenes";
import { createReservoirSeepScene } from "../lib/core/reservoir-seep-scene";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { createUniformSolver, advanceUniform } from "../tests/helpers/uniform-geometric";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { GPUPassProfile } from "./gpu-pass-profile";
import { particleCellCost } from "./reservoir-particle-cost";

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const maxSteps = Number(option("steps") ?? 600);
const wallBudget = Number(option("wall-seconds") ?? 90);
assert.ok(Number.isInteger(maxSteps) && maxSteps > 0 && maxSteps <= 2500);
assert.ok(wallBudget > 0 && wallBudget <= 160);
const out = resolve(option("out") ?? "artifacts/reservoir-performance/probe.jsonl");
mkdirSync(dirname(out), { recursive: true });
const log = (row: unknown) => { const text = JSON.stringify(row); appendFileSync(out, text + "\n"); console.log(text); };
const began = performance.now();
const watchdog = setTimeout(() => { log({ stopped: "180s total wall limit", elapsed_ms: performance.now() - began }); process.exit(2); }, 180_000);
watchdog.unref();
paceDawnEventPolling();
const scene = createReservoirSeepScene();
const values = getSceneDefinition("reservoir-seep").methodProfile!.overrides;
log({ kind: "metadata", date: new Date().toISOString(), scene: "reservoir-seep", cell_m: 0.005, values,
  maxSteps, simulationWallBudget_s: wallBudget, totalWallBudget_s: 180, rendering: false, splitTimestamps: true });
const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal", "disable-dawn-features=timestamp_quantization"]).requestAdapter();
assert.ok(adapter);
const raw = await adapter.requestDevice({ requiredFeatures: ["timestamp-query", ...(["subgroups", "float32-filterable"] as const).filter(f => adapter.features.has(f))], requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
const profile = new GPUPassProfile(raw, 4096, true);
const device = managedGPUDevice(profile.device, { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); log({ gpuError: e.error.message }); });
let solver: Awaited<ReturnType<typeof createUniformSolver>> | undefined;
async function words(buffer: GPUBuffer, offset: number, count: number) {
  if (!count) return new Uint32Array();
  const read = raw.createBuffer({ size: 4 * count, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = raw.createCommandEncoder(); encoder.copyBufferToBuffer(buffer, 4 * offset, read, 0, 4 * count); raw.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); const result = new Uint32Array(read.getMappedRange()).slice(); read.unmap(); return result;
  } finally { read.destroy(); }
}
try {
  solver = await createUniformSolver(device, scene, values);
  log({ kind: "ready", construction_ms: performance.now() - began, adapter: adapter.info });
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  const nb = frame.narrowBandFlip;
  const bins = nb as unknown as { bins: GPUBuffer; next: GPUBuffer; binCells: number; binSlots: number };
  const simulationStart = performance.now();
  const checkpoints = new Set([1, 60, 180, 300, 450, 600, 1200, 1800, 2400, maxSteps]);
  let interval: number[] = [];
  for (let step = 1; step <= maxSteps; step++) {
    const timed = checkpoints.has(step);
    if (timed) profile.start();
    const start = performance.now(); await advanceUniform(solver, step / 60);
    const wall_ms = performance.now() - start;
    const passes = timed ? await profile.finish() : undefined;
    if (!timed) interval.push(wall_ms);
    const budgetReached = performance.now() - simulationStart >= wallBudget * 1000;
    if (timed || budgetReached) {
      const header = await words(nb.state, 0, 48);
      // Sparse bin slots are rewritten for current slabs only. Dense bins
      // clear the previous occupied tiles before every sort.
      const cellCount = bins.binSlots ? 512 * Math.min(bins.binSlots, header[20]!) : bins.binCells;
      const populations = await words(bins.bins, 0, cellCount);
      const tiers = await words(frame.heat.tiers, frame.heat.regions, 4);
      const hotspots: unknown[] = [];
      if (step === maxSteps && process.argv.includes("--hotspots")) {
        const source = nb.particleSource;
        const sourceWords = await words(source.frame!.buffer, (source.frame!.offset ?? 0) / 4, 12);
        const sourceFloats = new Float32Array(sourceWords.buffer);
        const top: number[] = [];
        for (let cell = 0; cell < populations.length; cell++) {
          if (!populations[cell]) continue;
          if (top.length < 8 || populations[cell]! > populations[top.at(-1)!]!) {
            top.push(cell); top.sort((a, b) => populations[b]! - populations[a]!); top.length = Math.min(8, top.length);
          }
        }
        for (const cell of top) {
          const first = (await words(bins.next, cell, 1))[0]!;
          const count = populations[cell]!;
          const data = await words(nb.activeParticles, first * nb.particleStride, count * nb.particleStride);
          const f = new Float32Array(data.buffer), stride = nb.particleStride;
          let shallow = 0, saturated = 0, ballistic = 0, mass = 0, minDepth = Infinity, maxDepth = -Infinity;
          const positions = new Set<string>(), low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
          for (let i = 0; i < count; i++) {
            const at = i * stride, depth = f[at + 7]!, m = (data[at + 11]! & 7) + 1;
            mass += m; saturated += Number(m === 8); shallow += Number(depth >= -1);
            ballistic += Number((data[at + 11]! >>> 31) !== 0);
            minDepth = Math.min(minDepth, depth); maxDepth = Math.max(maxDepth, depth);
            positions.add(`${data[at]},${data[at + 1]},${data[at + 2]}`);
            for (let a = 0; a < 3; a++) {
              const world = sourceFloats[8 + a]! + f[at + a]! * source.positionScale_m[a]!;
              low[a] = Math.min(low[a]!, world); high[a] = Math.max(high[a]!, world);
            }
          }
          hotspots.push({ bin: cell, count, bounds_m: { low, high }, mass, saturated, ballistic,
            particleDepth: { min: minDepth, max: maxDepth, shallow }, uniquePositions: positions.size });
        }
      }
      const sortedWall = [...interval].sort((a, b) => a - b);
      log({ kind: "sample", step, t_s: step / 60, wall_ms, instrumented: timed,
        untimedFrames: interval.length, untimedMedian_ms: sortedWall[Math.floor(sortedWall.length / 2)] ?? null,
        untimedMean_ms: interval.length ? interval.reduce((a, b) => a + b, 0) / interval.length : null,
        particles: { ...solver.narrowBandFlipInfo }, work: { ...solver.workDomainInfo },
        fineCapacity: solver.info.uniformMixedFineCapacity, bandTiles: solver.info.uniformPressureBandTiles,
        pages: { ...solver.pagePoolInfo }, tiers: [...tiers],
        slabs: { extent: header[0], live: header[1], count: header[20], spill: header[21], scratchSpill: header[16], slowWalk: header[23] },
        sortedCellCost: particleCellCost(populations),
        ...(hotspots.length ? { hotspots } : {}),
        passes: passes?.sort((a, b) => b.ms - a.ms), simulationWall_ms: performance.now() - simulationStart });
      interval = [];
    }
    if (budgetReached) { log({ stopped: `${wallBudget}s simulation wall limit`, step }); break; }
  }
  assert.deepEqual(errors, []);
  log({ kind: "finished", elapsed_ms: performance.now() - began });
} catch (error) { log({ error: String(error), elapsed_ms: performance.now() - began }); throw error; }
finally { clearTimeout(watchdog); solver?.destroy(); profile.destroy(); raw.destroy(); }
