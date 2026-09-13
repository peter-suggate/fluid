/** Headless whole-frame Wasm scaling benchmark, with exact publication checks. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { cpus, totalmem } from "node:os";
import { findSceneDefinition } from "../../lib/core/scenes";
import { sceneDocument } from "../../lib/core/scene-definition";
import type { FluidWasmModule } from "../../lib/physics-wasm/module";
import { NodeWebWorker } from "./node-web-worker.mjs";

const root = resolve(import.meta.dirname, "../..");
const argument = (name: string, fallback: string) => process.argv
  .find(value => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const scenes = argument("scenes", "water-box-dam-break,coarse-first-pool-impact-quarter,cm12-figure-3")
  .split(",");
const frames = Number(argument("frames", "20"));
const warmup = Number(argument("warmup", "5"));
const dimension = Number(argument("dimension", "2"));
const dt = Number(argument("dt", String(1 / 30)));
const transportExperiment = argument("transport-experiment", "baseline");
const traceSegments = Number(argument("trace-segments", "1"));
const edgeSamples = Number(argument("edge-samples", "1"));
const cellwiseClosure = argument("cellwise-closure", "band-projection");
assert.ok(Number.isSafeInteger(frames) && frames > 0 && Number.isSafeInteger(warmup) && warmup >= 0);
assert.ok(dimension === 2 || dimension === 3, "--dimension must be 2 or 3");
assert.ok(Number.isFinite(dt) && dt > 0, "--dt must be positive and finite");
assert.ok(["baseline", "cellwise-probe", "cellwise-remap"].includes(transportExperiment),
  "--transport-experiment must be baseline, cellwise-probe, or cellwise-remap");
assert.ok(dimension === 2 || transportExperiment === "baseline",
  "cellwise transport experiments are available only in 2D");
assert.ok(Number.isSafeInteger(traceSegments) && traceSegments >= 1 && traceSegments <= 128,
  "--trace-segments must be an integer in 1..=128");
assert.ok(transportExperiment !== "baseline" || traceSegments === 1,
  "--trace-segments applies only to a cellwise transport experiment");
assert.ok([1, 2, 4].includes(edgeSamples), "--edge-samples must be 1, 2, or 4");
assert.ok(transportExperiment !== "baseline" || edgeSamples === 1,
  "--edge-samples applies only to a cellwise transport experiment");
assert.ok(["none", "local", "band-projection"].includes(cellwiseClosure),
  "--cellwise-closure must be none, local, or band-projection");
assert.ok(transportExperiment !== "baseline" || cellwiseClosure === "band-projection",
  "--cellwise-closure applies only to a cellwise transport experiment");
const transportExperimentOption = traceSegments === 1 && edgeSamples === 1 &&
    cellwiseClosure === "band-projection"
  ? transportExperiment
  : { mode: transportExperiment, traceSegments, edgeSamples, closure: cellwiseClosure };
const methodValues = Object.freeze({
  timeStep: "paper",
  pressureIterations: 28,
  pressureRelativeTolerance: 1e-5,
  brickFineResolution: "8",
  maximumMacroSpanBricks: "1",
  selectorMode: "coarse-first",
  surfaceFineRings: 1,
});
const percentile = (values: number[], fraction: number) => [...values].sort((a, b) => a - b)
  [Math.min(values.length - 1, Math.floor(fraction * values.length))];

// Metadata includes runtime-specific allocation receipts. Compare each published
// numerical plane independently, including exact NaN and signed-zero words.
function planeHashes(bytes: Uint8Array) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const hashes: Record<string, string> = {};
  for (let i = 0; i < view.getUint32(16, true); i++) {
    const at = 32 + 16 * i;
    const id = view.getUint32(at, true), kind = view.getUint32(at + 4, true);
    const offset = view.getUint32(at + 8, true), count = view.getUint32(at + 12, true);
    hashes[id] = createHash("sha256").update(bytes.subarray(offset,
      offset + count * (kind === 1 ? 4 : 1))).digest("hex");
  }
  return hashes;
}

/** Allocation size may vary by Wasm memory growth without changing physics. */
function comparableReceipt(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(comparableReceipt);
  if (!value || typeof value !== "object") return value;
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([key]) => key !== "allocatedBytes" && key !== "presentationBytes")
    .map(([key, child]) => [key, comparableReceipt(child)]));
}

async function lane(name: string) {
  const threaded = name.startsWith("threaded-");
  const artifact = threaded ? "threaded" : name;
  assert.ok(["scalar", "simd", "threaded"].includes(artifact));
  const workers = threaded ? Number(name.slice("threaded-".length)) : 0;
  const directory = resolve(root, "public/wasm/fluid-wasm", artifact);
  const binary = await readFile(resolve(directory, "fluid_wasm_bg.wasm"));
  const buildInfo = JSON.parse(await readFile(resolve(directory, "build-info.json"), "utf8")) as {
    sourceSha256: string; wasmSha256: string;
  };
  assert.equal(createHash("sha256").update(binary).digest("hex"), buildInfo.wasmSha256,
    "Wasm artifact does not match its build fingerprint");
  globalThis.Worker = NodeWebWorker as unknown as typeof Worker;
  globalThis.self = { addEventListener() {}, removeEventListener() {} } as unknown as Window & typeof globalThis;
  const wasm = await import(pathToFileURL(resolve(directory, "fluid_wasm.js")).href) as FluidWasmModule;
  await wasm.default({ module_or_path: binary });
  if (threaded) await wasm.initThreadPool!(workers);
  const cases = [];
  for (const id of scenes) {
    const definition = findSceneDefinition(id);
    assert.ok(definition, `unknown production scene ${id}`);
    const start = performance.now();
    const world = wasm.FluidWorld.from_scene(JSON.stringify(sceneDocument(definition)),
      JSON.stringify({ dimension, pressureIterations: 28, pressureRelativeTolerance: 1e-5,
        tracerBudget: 1000, transportExperiment: transportExperimentOption, methodValues,
        production: { dtS: dt, timeStep: "paper" } }));
    const initializationMs = performance.now() - start;
    try {
      const advanceMs: number[] = [], publicationMs: number[] = [];
      let finalPlanes = {};
      let finalReceipt = {};
      for (let frame = 1; frame <= warmup + frames; frame++) {
        const before = performance.now();
        const receipt = world.advance(frame, dt);
        const after = performance.now();
        const bytes = world.snapshot(0xffff_ffff);
        const published = performance.now();
        if (frame > warmup) {
          advanceMs.push(after - before);
          publicationMs.push(published - after);
        }
        if (frame === warmup + frames) {
          finalReceipt = JSON.parse(receipt);
          finalPlanes = planeHashes(bytes);
        }
      }
      cases.push({ id, initializationMs,
        advanceMedianMs: percentile(advanceMs, .5), advanceP95Ms: percentile(advanceMs, .95),
        publicationMedianMs: percentile(publicationMs, .5),
        completeMedianMs: percentile(advanceMs.map((value, i) => value + publicationMs[i]), .5),
        finalReceipt, finalPlanes });
    } finally { world.free(); }
  }
  return { lane: name, workers, dimension, dt, transportExperiment, traceSegments, edgeSamples,
    cellwiseClosure, methodValues,
    sourceSha256: buildInfo.sourceSha256,
    artifactSha256: buildInfo.wasmSha256, cases };
}

const selected = argument("lane", "");
if (selected) {
  process.stdout.write(`${JSON.stringify(await lane(selected))}\n`);
} else {
  const lanes = argument("lanes", "scalar,simd,threaded-1,threaded-2,threaded-4,threaded-8").split(",");
  const results = lanes.map(name => JSON.parse(execFileSync(process.execPath,
    ["--import", "tsx", process.argv[1], `--lane=${name}`, `--frames=${frames}`,
      `--warmup=${warmup}`, `--scenes=${scenes.join(",")}`,
      `--dimension=${dimension}`, `--dt=${dt}`,
      `--transport-experiment=${transportExperiment}`, `--trace-segments=${traceSegments}`,
      `--edge-samples=${edgeSamples}`, `--cellwise-closure=${cellwiseClosure}`],
    { cwd: root, encoding: "utf8", maxBuffer: 16 * 1024 * 1024 })) as Awaited<ReturnType<typeof lane>>);
  for (const result of results.slice(1)) {
    assert.equal(result.sourceSha256, results[0].sourceSha256, "Wasm artifacts came from different source states");
    for (let i = 0; i < scenes.length; i++) {
    assert.deepEqual(result.cases[i].finalPlanes, results[0].cases[i].finalPlanes,
      `${result.lane} ${scenes[i]} differs from ${results[0].lane}`);
    assert.deepEqual(comparableReceipt(result.cases[i].finalReceipt),
      comparableReceipt(results[0].cases[i].finalReceipt),
      `${result.lane} ${scenes[i]} receipt differs from ${results[0].lane}`);
    }
  }
  process.stdout.write(`${JSON.stringify({ timestamp: new Date().toISOString(),
    platform: process.platform, arch: process.arch, cpu: cpus()[0]?.model,
    logicalCpus: cpus().length, memoryBytes: totalmem(), node: process.version,
    dimension, dt, transportExperiment, traceSegments, edgeSamples, cellwiseClosure, methodValues,
    frames, warmup, exactPublicationParity: true, results }, null, 2)}\n`);
}
