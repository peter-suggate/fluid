/** Several arms of one scene in one process (each a source snapshot and
 * method values), frames alternated: the receipts an A/B of a physics or
 * schedule change reads when same-process A/A is not bit-identical (atomic
 * slot races). Per arm and frame: h tiles, band tiles, particles, the band's
 * cycles and residual, resident pages; at --every frames the volume drift and
 * the field hashes (velocity, volume, vertex phi, read logically).
 *   node --import tsx tools/probe-pressure-tail-ab.ts NAME --arms='[{"root":"/abs/a"},{"root":"/abs/b","values":{"pressureTail":"superblocks"}}]'
 *   [--scene=rest|cold|warm|splash|hot|<scene id>] [--length=128] [--depth=32] [--pool-blocks=N]
 *   [--steps=60] [--every=12] [--out=DIR] [--gate="RUNNER MODE"]
 * --gate: construct every arm (mostly CPU: pipeline compiles) first, then
 * hold a slot of the shared GPU runner over the frames and reads only. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { spawn, type ChildProcess } from "node:child_process";
import type { SceneDescription } from "../lib/core/model";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import type { WebGPUUniformReferenceSolver } from "../lib/methods/uniform/webgpu-uniform-reference";
import { createRegionTierScene, REGION_TIER_SCENES, type RegionTierScene } from "./region-tier-scenes";

const option = (key: string) => process.argv.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
const name = process.argv[2]; assert.ok(name); assert.match(name, /^[a-z0-9.-]+$/);
const arms = JSON.parse(option("arms") ?? "[]") as { root: string; values?: Record<string, unknown>; label?: string }[];
assert.ok(arms.length >= 1 && arms.every(a => typeof a.root === "string"));
const steps = Number(option("steps") ?? 60), every = Number(option("every") ?? 12);
const sceneOption = option("scene") ?? "rest", region = (REGION_TIER_SCENES as readonly string[]).includes(sceneOption);
const length = Number(option("length") ?? 128), depth = Number(option("depth") ?? 32);
const poolBlocks = option("pool-blocks") === undefined ? undefined : Number(option("pool-blocks"));
const out = option("out") ?? "artifacts/pressure-tail"; mkdirSync(out, { recursive: true });
paceDawnEventPolling();
async function acquireGate(gate: string): Promise<ChildProcess> {
  const [command, ...args] = gate.split(" ").filter(Boolean);
  const child = spawn(command!, [...args, "cat"], { stdio: ["pipe", "pipe", "inherit"] });
  await new Promise<void>((done, fail) => { child.stdout!.once("data", () => done()); child.once("exit", code => fail(new Error(`gate exited ${code}`))); child.stdin!.write("held\n"); });
  return child;
}
type Helpers = { readMixedTexture: (d: GPUDevice, t: GPUTexture, a?: unknown) => Promise<Float32Array>; readUniformLiveAddress: (d: GPUDevice, s: WebGPUUniformReferenceSolver) => Promise<unknown> };
interface Row { frame: number; fine: number; band: number; particles: number; cycles: number; residual: number; resident: number; volumeDrift?: number; hashes?: Record<string, string>; pastWall?: number }
let lock: ChildProcess | undefined, raw: GPUDevice | undefined;
const solvers: WebGPUUniformReferenceSolver[] = [], helpers: Helpers[] = [], errors: string[] = [];
const rows: Row[][] = arms.map(() => []);
try {
  const dawn = await import(pathToFileURL(resolve("node_modules/webgpu/index.js")).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter(); assert.ok(adapter);
  const features = (["timestamp-query", "subgroups", "float32-filterable"] as GPUFeatureName[]).filter(f => adapter.features.has(f));
  raw = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  raw.addEventListener("uncapturederror", event => { event.preventDefault(); errors.push(event.error.message); });
  let dt = 0, scene: SceneDescription | undefined;
  for (const [index, arm] of arms.entries()) {
    const load = (file: string) => import(pathToFileURL(resolve(arm.root, file)).href);
    const [{ uniformNarrowBandMethod: method }, { resolveMethodValues }, { sceneDocument }, { getSceneDefinition }, { managedGPUDevice }, fields, geometric] = await Promise.all([
      load("lib/methods/uniform/uniform-narrow-band-method.ts"), load("lib/core/method-contract.ts"), load("lib/core/scene-definition.ts"), load("lib/core/scenes.ts"), load("lib/core/gpu-compilation-manager.ts"),
      load("tests/helpers/uniform-mixed-native-fields.ts"), load("tests/helpers/uniform-geometric.ts"),
    ]);
    helpers.push({ readMixedTexture: fields.readMixedTexture, readUniformLiveAddress: geometric.readUniformLiveAddress });
    let values: unknown, quality: string, built: SceneDescription;
    if (region) {
      built = createRegionTierScene(sceneOption as RegionTierScene, length, depth); quality = "balanced";
      values = resolveMethodValues(method, quality, { ...method.appDefaults, timeStep: "scene", ...arm.values });
    } else {
      const definition = getSceneDefinition(sceneOption); built = structuredClone(sceneDocument(definition));
      quality = definition.methodProfile.quality; values = resolveMethodValues(method, quality, { ...method.appDefaults, ...definition.methodProfile.overrides, ...arm.values });
    }
    if (poolBlocks !== undefined) built.fluid = { ...built.fluid, pagePool: { ...built.fluid?.pagePool, blocks: poolBlocks } } as SceneDescription["fluid"];
    const step = method.effectiveStep_s?.(built, values) ?? built.numerics.fixedDt_s;
    built.duration_s = Math.max(built.duration_s, steps * step);
    if (index) { assert.equal(step, dt); assert.deepEqual(built, scene); }
    scene = built; dt = step;
    const start = performance.now();
    solvers.push(await method.createSolverAsync!(managedGPUDevice(raw, { requireWorkerRealm: false }), built, quality, values, undefined, () => {}) as WebGPUUniformReferenceSolver);
    await raw.queue.onSubmittedWorkDone();
    console.log(JSON.stringify({ phase: "constructed", arm: index, root: arm.root, values: arm.values ?? {}, ms: performance.now() - start }));
  }
  const gate = option("gate");
  if (gate) { const t = performance.now(); lock = await acquireGate(gate); console.log(JSON.stringify({ phase: "locked", wait_ms: performance.now() - t })); }
  const seep = scene!.sceneId === "reservoir-seep";
  for (let frame = 1; frame <= steps; frame++) {
    const order = arms.map((_, i) => (i + frame) % arms.length);
    for (const i of order) {
      const solver = solvers[i]!;
      assert.ok(solver.advanceTo(frame * dt, [])); await solver.awaitFrameCompletion();
      assert.equal(solver.info.simulationPipelineError, undefined);
      const row: Row = { frame, fine: Number(solver.info.uniformMixedFineTiles ?? 0), band: Number(solver.info.uniformPressureBandTiles ?? 0), particles: Number(solver.narrowBandFlipInfo?.particles ?? 0),
        cycles: Number(solver.info.uniformPressureBandCycles ?? 0), residual: Number(solver.info.uniformPressureBandResidual ?? NaN), resident: Number(solver.pagePoolInfo?.residentPages ?? 0) };
      if (frame % every === 0 || frame === steps) {
        row.volumeDrift = (await solver.readStats()).volumeDrift;
        const address = await helpers[i]!.readUniformLiveAddress(raw, solver);
        row.hashes = {};
        for (const [key, texture] of Object.entries({ velocity: solver.velocityTexture, volume: solver.volumeTexture, phi: solver.vertexPhiTexture! })) {
          const data = await helpers[i]!.readMixedTexture(raw, texture, address);
          row.hashes[key] = createHash("sha256").update(new Uint8Array(data.buffer, data.byteOffset, data.byteLength)).digest("hex").slice(0, 16);
          // The seep's volume past the wall (x >= 16 floor(nx/32)): its discharge.
          if (seep && key === "volume") { const { nx } = solver.info, high = 16 * Math.floor(nx / 32); let v = 0; for (let k = 0; k < data.length; k++) if (k % nx >= high) v += data[k]!; row.pastWall = v; }
        }
      }
      rows[i]!.push(row);
    }
    if (frame % every === 0 || frame === steps) console.log(JSON.stringify({ frame, arms: rows.map(r => { const x = r.at(-1)!; return { ...x, hashes: x.hashes && Object.values(x.hashes).join(":") }; }) }));
  }
  lock?.stdin?.end(); lock = undefined;
  assert.deepEqual(errors, []);
  const quantile = (v: number[], q: number) => { const s = [...v].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * s.length))]!; };
  const summary = rows.map((r, i) => ({ arm: i, values: arms[i]!.values ?? {}, root: arms[i]!.root,
    residual: { p50: quantile(r.map(x => x.residual), .5), p90: quantile(r.map(x => x.residual), .9), max: Math.max(...r.map(x => x.residual)) },
    cycles: { mean: r.reduce((n, x) => n + x.cycles, 0) / r.length, max: Math.max(...r.map(x => x.cycles)) }, last: r.at(-1) }));
  writeFileSync(`${out}/${name}.json`, JSON.stringify({ name, scene: scene!.sceneId, steps, every, poolBlocks, arms, summary, rows }, null, 2) + "\n");
  console.log(JSON.stringify({ phase: "complete", summary }));
} finally {
  lock?.stdin?.end();
  for (const solver of solvers) solver.destroy(); await raw?.queue.onSubmittedWorkDone(); raw?.destroy();
}
