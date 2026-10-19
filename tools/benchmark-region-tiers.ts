/** Production solver benchmark. Uninstrumented timings and optional command
 * diagnostics (--profile) are separate runs. --timestamps also attempts native
 * phase timing; unsupported/invalid samples remain explicit in the report.
 * node --import tsx tools/benchmark-region-tiers.ts --scene=splash --length=128
 *   [--steps=120] [--from=40] [--profile] [--values={...}] [--out=PATH.json]
 */
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync, readFileSync, readdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { cpus } from "node:os";
import { createHash } from "node:crypto";
import { createUniformSolver, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";
import { paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { usePerformanceInstrumentationStore } from "../lib/core/stores/performance-instrumentation-store";
import type { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import type { PerformanceTrace } from "../lib/core/performance-trace";
import type { MethodParamValues } from "../lib/core/method-contract";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";
import { profileRegionTierCommands, type RegionTierCommands } from "./region-tier-command-profile";

interface TierFrame {
  step: number; wall_ms: number; encode_ms: number; tiers: number[]; ceiling: number; idle: boolean;
  particles?: number; fine?: number; band?: number; bandResidual?: number;
  trace?: PerformanceTrace; traceError?: string; commands?: RegionTierCommands;
}

const option = (name: string) => process.argv.find(a => a.startsWith(`--${name}=`))?.slice(name.length + 3);
const mode = (option("scene") ?? "cold") as RegionTierScene, length = Number(option("length") ?? 128);
assert.ok(REGION_TIER_SCENES.includes(mode));
const steps = Number(option("steps") ?? 120), from = Number(option("from") ?? 40), profile = process.argv.includes("--profile");
const timestamps = process.argv.includes("--timestamps");
assert.ok(Number.isInteger(steps) && Number.isInteger(from) && from > 0 && from <= steps);
const values: MethodParamValues = JSON.parse(option("values") ?? "{}");
assert.ok(values && typeof values === "object" && !Array.isArray(values));
const depth = Number(option("depth") ?? 32);
const scene = createRegionTierScene(mode, length, depth);
scene.duration_s = Math.max(scene.duration_s, steps * scene.numerics.fixedDt_s);
const out = resolve(option("out") ?? `artifacts/region-tiers/${mode}-${length}${depth === 32 ? "" : `-depth${depth}`}${profile ? "-profile" : ""}.json`);
process.env.WEBGPU_NODE_MODULE ??= resolve("node_modules/webgpu/index.js");
paceDawnEventPolling();
const source = createHash("sha256");
for (const f of readdirSync("lib/methods/uniform").filter(f => f.endsWith(".ts")).sort()) source.update(f).update(readFileSync(`lib/methods/uniform/${f}`));
const rows: TierFrame[] = [];
await withUniformDevice("region tier benchmark", async device => {
  const commands = profile ? profileRegionTierCommands(device) : undefined;
  device.addEventListener("uncapturederror", event => console.error("GPU validation:", event.error.message));
  const start = performance.now(), solver = await createUniformSolver(device, scene, values);
  const construction_ms = performance.now() - start;
  const frame = (solver as unknown as { mixedFrame: UniformMixedFrame }).mixedFrame;
  // Benchmark-only ablation: reproduce the old host launch policy without
  // recompiling shaders or changing the GPU's per-tile membership predicate.
  const legacyBand = process.argv.includes("--legacy-band");
  if (legacyBand) {
    const band = (frame as unknown as { band: object }).band;
    Object.defineProperty(band, "empty", { get() {
      return frame.ownership.capacity.fineTiles === 0 ||
        (Reflect.get(band, "cold") !== undefined && frame.cold.quiet);
    } });
  }
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  console.log(JSON.stringify({ scene: scene.sceneId, construction_ms, profile, values, legacyBand }));
  try {
    for (let step = 1; step <= steps; step++) {
      const sampling = timestamps && step >= from;
      usePerformanceInstrumentationStore.getState().setEnabled(sampling);
      if (sampling) await new Promise(r => setTimeout(r, 360));
      commands?.begin();
      const t = performance.now();
      assert.ok(solver.advanceTo(step * scene.numerics.fixedDt_s, []));
      const encode_ms = performance.now() - t;
      await solver.awaitFrameCompletion();
      const wall_ms = performance.now() - t;
      const commandCounts = commands?.end();
      assert.equal(solver.info.simulationPipelineError, undefined);
      assert.ok(Math.abs((solver.info.completedTime_s ?? 0) - step * scene.numerics.fixedDt_s) < 1e-8);
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(frame.heat.tiers, 4 * frame.heat.regions, read, 0, 32);
      device.queue.submit([encoder.finish()]); await read.mapAsync(GPUMapMode.READ);
      const tiers = [...new Uint32Array(read.getMappedRange()).slice(0, 5)]; read.unmap();
      if (sampling) {
        for (let n = 0; n < 100 && (solver as unknown as { physicsTracePending: boolean }).physicsTracePending; n++) await new Promise(r => setTimeout(r, 5));
      }
      rows.push({ step, wall_ms, encode_ms, tiers, ceiling: frame.cold.ceiling, idle: frame.narrowBandFlip.idle,
        particles: solver.narrowBandFlipInfo?.particles, fine: solver.info.uniformMixedFineTiles,
        band: solver.info.uniformPressureBandTiles, bandResidual: solver.info.uniformPressureBandResidual,
        trace: sampling ? solver.info.physicsTrace : undefined,
        traceError: sampling ? solver.info.physicsTraceUnavailable : undefined, commands: commandCounts });
      if (step % 20 === 0 || step === steps) console.log(JSON.stringify({ ...rows.at(-1), trace: undefined, commands: undefined }));
    }
    const selected = rows.filter(r => r.step >= from), mean = (v: number[]) => v.reduce((s, n) => s + n, 0) / v.length;
    const quantile = (v: number[], q: number) => [...v].sort((a, b) => a - b)[Math.floor((v.length - 1) * q)];
    const phases: Record<string, number[]> = {};
    for (const r of selected) {
      const sum: Record<string, number> = {};
      for (const p of r.trace?.phases ?? []) sum[p.label] = (sum[p.label] ?? 0) + p.duration_ms;
      for (const [label, ms] of Object.entries(sum)) (phases[label] ??= []).push(ms);
    }
    const stats = await solver.readStats();
    const fields: Record<string, string> = {};
    if (process.argv.includes("--fingerprint")) for (const [name, texture] of Object.entries({
      velocity: solver.velocityTexture, volume: solver.volumeTexture, phi: solver.vertexPhiTexture!, pressure: solver.gridPressureTexture,
    })) {
      const data = await readMixedTexture(device, texture);
      fields[name] = createHash("sha256").update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).digest("hex");
    }
    const summary = { mean_ms: mean(selected.map(r => r.wall_ms)), median_ms: quantile(selected.map(r => r.wall_ms), .5),
      p95_ms: quantile(selected.map(r => r.wall_ms), .95), encode_ms: mean(selected.map(r => r.encode_ms)),
      tiers: [0, 1, 2].map(i => mean(selected.map(r => r.tiers[i]!))), idleFrames: selected.filter(r => r.idle).length,
      phases_ms: Object.fromEntries(Object.entries(phases).map(([k, v]) => [k, mean(v)]).sort((a, b) => Number(b[1]) - Number(a[1]))) };
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify({ date: new Date().toISOString(), cpu: cpus()[0]?.model, node: process.version,
      sourceHash: source.digest("hex"), scope: "simulation advance to fenced completion; excludes construction, renderer and diagnostic reads",
      scene, values, profile, timestamps, legacyBand, steps, from, construction_ms, summary, fields,
      physics: { maxSpeed: stats.maxSpeed_m_s, volume: stats.volumeCellSum, representedVolume: stats.representedVolumeCellSum, drift: stats.volumeDrift, residual: stats.uniformPressureAcceptedResidual }, rows }, null, 2) + "\n");
    console.log(JSON.stringify({ out, summary }));
  } catch (error) {
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(`${out}.failed.json`, JSON.stringify({ scene, values, profile, timestamps, steps, from, error: String(error), rows }, null, 2) + "\n");
    throw error;
  } finally { read.destroy(); solver.destroy(); commands?.destroy(); usePerformanceInstrumentationStore.getState().setEnabled(false); }
}, ["timestamp-query"]);
