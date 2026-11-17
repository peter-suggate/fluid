/** Lane P's M5 rigid gates (docs/plans/nbflip-port/m5-bodies-design-2026-11-17.md §8): one engine on one rigid oracle
 * scene, fenced, a JSONL row a step.
 *
 * node --import tsx tools/nbflip-rigid-gates.ts --scene=rigid-hydrostatic|rigid-float|rigid-sink --engine=brick|uniform
 *   [--steps=N] [--out=DIR] [--name=NAME] [--h=CELL_M] [--pools={"f":..,"c":..,"b":..,"k":..,"particles":..}]
 *
 * Both engines run the scene at its methodProfile over the method's app defaults (dt 1/60), as the app does, with the
 * scene's initial roster passed to every advanceTo (the app's; the GPU owns dynamic bodies between syncs).
 * Rows: the body's pose (readRigidBodyPoses), the liquid volume (tools/nbflip-observe.ts's box measure, cells), the
 * fenced advance's wall ms; on the brick engine also the step's couple words (the exchange, tapped before the
 * integrator: the reaction impulse, N s) and the peak |u| over the live F bricks' interior liquid faces (both cells liquid by centre phi; was: a phi < 0 vertex at
 * the face's cell or its low neighbour within the brick). The summary: the volume drift
 * |V(n) - V(0)| / V(0) at the end, the step ms median over steps 5..N (advance + fence only; the reads after the fence
 * are not timed), and for the brick engine the impulse / dt against rho g V over steps 5..N. Run under
 * tools/gpu/fluid-gpu-run.sh timing. */
import assert from "node:assert/strict";
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import { resolveMethodValues } from "../lib/core/method-contract";
import { managedGPUDevice } from "../lib/core/gpu-compilation-manager";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { createProcessRetainedDawnGPU, paceDawnEventPolling } from "../lib/harness/node-dawn-provider";
import { uniformNarrowBandMethod as method } from "../lib/methods/uniform/uniform-narrow-band-method";
import { initializeRigidBodies } from "../lib/core/rigid-body";
import { GPU_RIGID_EXCHANGE_BYTES } from "../lib/core/webgpu-eulerian";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import type { BrickNBFlipSolver } from "../lib/methods/narrow-band/brick-solver";
import { MG_HEADER, mgLayout } from "../lib/methods/narrow-band/pressure-mgpcg";
import { gpuKernelTimer } from "./gpu-kernel-timer";
import { GPUCompilationManager } from "../lib/core/gpu-compilation-manager";
import { BRICK_ENGINE_SCENES } from "../lib/methods/narrow-band/brick-scene";
import { BRICK_CELLS, BRICK_STRIDE, RECORD_WORDS, H, LIST, REC } from "../lib/methods/narrow-band/brick-abi";
import { boxGeometry } from "./nbflip-observables";
import { createNBFlipBoxObserver } from "./nbflip-observe";
import { solidVoxelsOnMovedLattice } from "../lib/core/scene-lattice";

const args = process.argv.slice(2);
const option = (key: string) => args.find(a => a.startsWith(`--${key}=`))?.slice(key.length + 3);
for (const a of args) assert.ok(/^--(?:scene|engine|steps|out|name|h|pools)=.+$/.test(a) || a === "--no-bodies" || a === "--profile", a);
const sceneId = option("scene") ?? "rigid-hydrostatic", engine = option("engine") ?? "brick";
assert.ok(["rigid-hydrostatic", "rigid-float", "rigid-sink"].includes(sceneId), "--scene=rigid-hydrostatic|rigid-float|rigid-sink");
assert.ok(engine === "brick" || engine === "uniform", "--engine=brick|uniform");
const steps = Number(option("steps") ?? { "rigid-hydrostatic": 15, "rigid-float": 60, "rigid-sink": 30 }[sceneId]);
assert.ok(Number.isInteger(steps) && steps > 0);
const out = resolve(option("out") ?? "artifacts/nbflip-port/rigid"), name = option("name") ?? `${sceneId}-${engine}-${steps}`;
mkdirSync(out, { recursive: true });
const jsonl = `${out}/${name}.jsonl`; writeFileSync(jsonl, "");
const log = (row: Record<string, unknown>) => { const t = JSON.stringify(row); appendFileSync(jsonl, t + "\n"); console.log(t.length > 300 ? t.slice(0, 300) + "…" : t); };

const definition = getSceneDefinition(sceneId), scene = structuredClone(sceneDocument(definition)), profile = definition.methodProfile;
// --h: the scene at another cell (both engines alike; the shell follows the lattice), --pools: its brick pools (JSON).
if (option("h")) {
  const before = structuredClone(scene); scene.voxelDomain.finestCellSize_m = Number(option("h"));
  scene.solidVoxels = solidVoxelsOnMovedLattice(before, scene); delete scene.fluid.brickPools;
}
if (option("pools")) scene.fluid.brickPools = JSON.parse(option("pools")!);
// --no-bodies: the same tank with the bodies removed (the bodies-free reference for the peak and the volume).
const noBodies = args.includes("--no-bodies"), body = scene.rigidBodies[0]!;
if (noBodies) scene.rigidBodies = [];
const quality = profile?.methodId === method.id ? profile.quality : "balanced";
const values = resolveMethodValues(method, quality, { ...method.appDefaults, ...(profile?.methodId === method.id ? profile.overrides : {}), engine });
const dt = method.effectiveStep_s?.(scene, values) ?? scene.numerics.fixedDt_s!;
const r = body.dimensions_m.x, g = Math.hypot(scene.fluid.gravity_m_s2.x, scene.fluid.gravity_m_s2.y, scene.fluid.gravity_m_s2.z);
const buoyancy_N = scene.fluid.density_kg_m3 * g * (4 / 3) * Math.PI * r ** 3;
log({ kind: "meta", sceneId, engine, steps, dt, quality, pools: scene.fluid.brickPools, h: scene.voxelDomain.finestCellSize_m, body: { r, y0: body.position_m.y, density: body.density_kg_m3, motion: body.motion }, buoyancy_N });

paceDawnEventPolling();
const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE ?? resolve("node_modules/webgpu/index.js")).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, ["backend=metal"]).requestAdapter();
assert.ok(adapter, "No WebGPU adapter");
const device = managedGPUDevice(await adapter.requestDevice({ requiredFeatures: (["subgroups", "float32-filterable", ...(process.argv.includes("--profile") ? ["timestamp-query"] as const : [])] as const).filter(f => adapter.features.has(f)), requiredLimits: requiredFluidDeviceLimits(adapter.limits) }), { requireWorkerRealm: false });
const errors: string[] = [];
device.addEventListener("uncapturederror", e => { e.preventDefault(); errors.push(e.error.message); console.error(e.error.message); });

const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b); return s.length ? s.length % 2 ? s[s.length >> 1]! : (s[s.length / 2 - 1]! + s[s.length / 2]!) / 2 : null; };
let solver: GPUSolverInstance | undefined, failure: string | undefined;
const rows: { step: number; ms: number; y?: number; volume: number; liquid?: number; impulse?: number[]; peak?: number }[] = [];
let highK = 0, highF = 0, volume0 = NaN, liquid0: number | undefined;
const byKernel = new Map<string, number>(), iterations: number[] = [], PROFILE_STEPS = 3;
// The liquid volume outside the bodies (coordinator 2026-11-17): both engines continue liquid phi through a body (the
// cut-cell pressure's), so the box measure counts a submerged body's interior. The measure here is the same box
// observer's, each owner's liquid scaled by the fraction of its cell centres outside the sphere: the observer's own
// WGSL (its "Box observer" module, either engine's, through the solver's device) with boxAdd rewritten. A new observer
// a step, its sphere the step's pose (cells, the lattice's: origin at (-w/2, 0, -d/2)).
let exclude: { c: [number, number, number]; r: number } | undefined, rewrites = 0;
const BOX_ADD = /fn boxAdd\(q:vec3u,\s*w:u32,\s*liquid:f32,\s*u:vec3f\)\s*\{\s*if\s*\(liquid\s*<=\s*0\.0\)\s*\{\s*return;\s*\}/;
const excluded = (code: string) => {
  if (!exclude) return code;
  // Absent (another observer module), or already rewritten (the solver's device is this proxy too).
  if (!code.includes("fn boxAdd(") || code.includes("fn boxAdd(q:vec3u,w:u32,whole:f32")) return code;
  if (!BOX_ADD.test(code)) { const i = code.indexOf("fn boxAdd("); throw new Error(`the box observer's boxAdd (nbflip-observables.ts) is not as this rewrite expects: ${i < 0 ? "absent" : JSON.stringify(code.slice(i, i + 120))}`); }
  const { c, r } = exclude, f = (v: number) => v.toFixed(6);
  rewrites++;
  return code.replace(BOX_ADD, `fn boxAdd(q:vec3u,w:u32,whole:f32,u:vec3f){
 var outside=0u;let n=w*w*w;
 for(var k=0u;k<n;k++){let p=vec3f(q)+vec3f(f32(k%w),f32((k/w)%w),f32(k/(w*w)))+vec3f(0.5);if(distance(p,vec3f(${f(c[0])},${f(c[1])},${f(c[2])}))>${f(r)}){outside+=1u;}}
 let liquid=whole*f32(outside)/f32(n);
 if(liquid<=0.0){return;}`);
};
// Both engines' observers compile through the compilation manager (managedGPUDevice; the old engine's on its own
// device, whose detail shaders key on that device's identity, so no device proxy will do, and the manager caches a
// module by its code): GPUCompilationManager's createShaderModule is patched for the observer's creation only.
// A tool's measure, asserted below (the rewrite happened); the proper form is an observer option (a liquid mask: the
// cells a body holds), a candidate for when nbflip-observables is next touched.
const withExcluded = async <T,>(make: () => Promise<T>): Promise<T> => {
  const proto = GPUCompilationManager.prototype as { createShaderModule(m: GPUShaderModuleDescriptor): GPUShaderModule }, original = proto.createShaderModule;
  proto.createShaderModule = function (this: GPUCompilationManager, m: GPUShaderModuleDescriptor) {
    const code = m.label?.startsWith("Box observer") ? excluded(m.code) : m.code, module = original.call(this, { ...m, code });
    if (code !== m.code) void module.getCompilationInfo().then(i => { for (const e of i.messages) if (e.type === "error") console.error(`body-excluded measure: ${e.message} at ${e.lineNum}:${e.linePos}`); });
    return module;
  };
  try { return await make(); } finally { proto.createShaderModule = original; }
};
try {
  // --profile: per-dispatch GPU time (gpuKernelTimer) over steps 5..N, by pipeline label; the split serialises
  // dispatches, so the sums run above the clean step (attribution only, not a timing claim).
  const timer = args.includes("--profile") ? gpuKernelTimer(device) : undefined;
  // The rigid scenes are not in BRICK_ENGINE_SCENES until their pools and the padded walls land: the tool admits the
  // scene it gates for the solver's creation only (as tests/brick-seam-rest.test.ts does).
  const enabled = BRICK_ENGINE_SCENES as string[], had = enabled.includes(sceneId);
  if (!had) enabled.push(sceneId);
  solver = await method.createSolverAsync!(timer?.device ?? device, scene, quality, values, undefined, () => {}).finally(() => { if (!had) enabled.splice(enabled.indexOf(sceneId), 1); });
  await solver.waitForSimulationReady?.();
  const brick = engine === "brick" ? solver as unknown as BrickNBFlipSolver : undefined;
  assert.ok(!brick || noBodies || brick.rigidBodies, "the brick engine hosts the scene's bodies");
  const tap = brick ? device.createBuffer({ label: "rigid gate tap", size: GPU_RIGID_EXCHANGE_BYTES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC }) : undefined;
  const tapRead = brick ? device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }) : undefined;
  if (brick && tap && !noBodies) brick.rigidBodies!.tap = tap;
  const observer = await createNBFlipBoxObserver(device, solver, scene, boxGeometry(scene, "paddleboats"));
  const H_CELL = scene.voxelDomain.finestCellSize_m, origin = [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m];
  const liquidOutside = async (poses?: readonly { position_m: { x: number; y: number; z: number } }[]) => {
    const b = scene.rigidBodies[0], at = poses?.[0]?.position_m ?? b?.position_m;
    if (noBodies || !b || !at || b.shape !== "sphere") return undefined;
    exclude = { c: [(at.x - origin[0]!) / H_CELL, (at.y - origin[1]!) / H_CELL, (at.z - origin[2]!) / H_CELL], r: b.dimensions_m.x / H_CELL };
    const before = rewrites;
    const o = await withExcluded(() => createNBFlipBoxObserver(device, solver!, scene, boxGeometry(scene, "paddleboats")));
    assert.ok(rewrites > before, "the body-excluded measure rewrote its observer's module");
    try { return (await o.observe()).total; } finally { o.destroy(); exclude = undefined; }
  };
  const roster = initializeRigidBodies(scene.rigidBodies);
  const peakU = async () => {
    const e = brick!.brickEngine, L = e.layout;
    const head = await e.read("set", 0, 64), n = Math.min(head[H.LIST_COUNT + LIST.F_ALL]!, L.set.listCap[LIST.F_ALL]!);
    highK = Math.max(highK, head[H.HIGH_K]!); highF = Math.max(highF, head[H.HIGH_F]!);
    const slots = await e.read("set", L.set.lists[LIST.F_ALL]!, Math.max(n, 1));
    const base = L.base("vel", "F", "U0"), words = await e.read("vel", base, 3 * BRICK_STRIDE * L.cap.f), u = new Float32Array(words.buffer, words.byteOffset, words.length);
    // Over the liquid: a face with a liquid vertex (phi < 0) at its cell's low corner or at its low neighbour's in the
    // brick (air faces carry the step's unprojected gravity, g dt).
    const pw = await e.read("phi", L.base("phi", "F", "PHI"), BRICK_STRIDE * L.cap.f), fw = new Float32Array(pw.buffer, pw.byteOffset, pw.length);
    // Over the liquid's interior faces: both cells of the face liquid by their centre phi (the mean of the cell's 8
    // vertices, so cells x, y, z < 7 of the brick), the low cell in the brick. A free-surface face (an air side) and the
    // air faces carry the step's unprojected gravity, g dt, and are not rest velocities.
    // ... and both cells OPEN (FLAGS bit 0): a cell inside a body (bsOpen) or past the walls is no liquid cell.
    const fwl = await e.read("phi", L.base("phi", "F", "FLAGS"), BRICK_STRIDE * L.cap.f);
    const open = (s: number, x: number, y: number, z: number) => (fwl[BRICK_STRIDE * s + x + 8 * y + 64 * z]! & 1) !== 0;
    const centre = (s: number, x: number, y: number, z: number) => {
      let m = 0; for (let k = 0; k < 8; k++) m += fw[BRICK_STRIDE * s + (x + (k & 1)) + 8 * (y + ((k >> 1) & 1)) + 64 * (z + (k >> 2))]!; return m / 8;
    };
    let peak = 0, where: number[] = [];
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {
      const at = BRICK_STRIDE * (slots[i]! + a * L.cap.f);
      for (let l = 0; l < BRICK_CELLS; l++) {
        const v = u[at + l]!; assert.ok(Number.isFinite(v), `u not finite at slot ${slots[i]} axis ${a}`);
        const c = [l & 7, (l >> 3) & 7, l >> 6]; if (c.some(q => q > 6) || c[a]! < 1) continue;
        const lo = [...c]; lo[a]! -= 1;
        if (!(centre(slots[i]!, c[0]!, c[1]!, c[2]!) < 0 && centre(slots[i]!, lo[0]!, lo[1]!, lo[2]!) < 0)) continue;
        if (!(open(slots[i]!, c[0]!, c[1]!, c[2]!) && open(slots[i]!, lo[0]!, lo[1]!, lo[2]!))) continue;
        if (Math.abs(v) > peak) { peak = Math.abs(v); where = [slots[i]!, a, ...c]; }
      }
    }
    return peak;
  };
  volume0 = (await observer.observe()).total; liquid0 = await liquidOutside();
  log({ kind: "step", step: 0, volume: volume0, ...(liquid0 !== undefined ? { liquid: liquid0 } : {}), ...(brick ? { peak: await peakU() } : {}) });
  for (let step = 1; step <= steps; step++) {
    // The profiled window: steps 5..7 (each wrapped encoder holds a Metal sample buffer until collect, and Metal has few).
    if (timer) timer.enabled = step >= 5 && step < 5 + PROFILE_STEPS;
    const t0 = performance.now();
    assert.ok(solver.advanceTo(step * dt, roster), `advance to step ${step} admitted: ${JSON.stringify((solver as unknown as { info: { simulationFailure?: unknown } }).info.simulationFailure)}`);
    await solver.awaitFrameCompletion?.();
    const ms = performance.now() - t0;
    await solver.assertSimulationHealthy?.();
    const poses = await solver.readRigidBodyPoses?.();
    const volume = (await observer.observe()).total, liquid = await liquidOutside(poses);
    let impulse: number[] | undefined, peak: number | undefined;
    if (brick && !noBodies) {
      const enc = device.createCommandEncoder(); enc.copyBufferToBuffer(tap!, 0, tapRead!, 0, 48); device.queue.submit([enc.finish()]);
      await tapRead!.mapAsync(GPUMapMode.READ); const w = new Int32Array(tapRead!.getMappedRange().slice(0)); tapRead!.unmap();
      impulse = [w[0]! * 1e-6, w[1]! * 1e-6, w[2]! * 1e-6];
    }
    if (brick) peak = await peakU();
    if (brick?.brickEngine.receipt && step >= 5) iterations.push(brick.brickEngine.receipt.header[MG_HEADER.ITERATIONS]!);
    // One collect two steps after the window: the engine submits a step's first part at the next advance, and collect
    // destroys the resolve buffers it takes.
    if (timer && step === 5 + PROFILE_STEPS + 1) { await device.queue.onSubmittedWorkDone(); const c = await timer.collect(); for (const [k, v] of c.byLabel) { const name = k.split(" / ").pop()!; byKernel.set(name, (byKernel.get(name) ?? 0) + v); } }
    const row = { step, ms, y: poses?.[0]?.position_m.y, volume, ...(liquid !== undefined ? { liquid } : {}), ...(impulse ? { impulse } : {}), ...(peak !== undefined ? { peak } : {}) };
    rows.push(row); log({ kind: "step", ...row });
  }
} catch (error) { failure = error instanceof Error ? error.message : String(error); console.error(failure); }
await device.queue.onSubmittedWorkDone().catch(() => {});
const timed = rows.filter(r => r.step >= 5).map(r => r.ms), last = rows.at(-1);
const first = rows[0];
const summary: Record<string, unknown> = { kind: "summary", failure, errors: errors.slice(0, 4), steps: rows.length,
  step_ms: { median: median(timed), n: timed.length, window: "steps 5..N, fenced advance only" } };
if (last && first) summary.y = { first: first.y, last: last.y };
if (rows.length) {
  const volumes = rows.map(r => r.volume);
  const liquids = rows.map(r => r.liquid).filter((v): v is number => v !== undefined);
  if (liquid0 !== undefined && liquids.length) summary.liquid = { what: "box measure outside the body (cell centres)", initial: liquid0, last: liquids.at(-1), drift: Math.abs(liquids.at(-1)! - liquid0) / liquid0, peakDrift: Math.max(...liquids.map(v => Math.abs(v - liquid0) / liquid0)) };
  summary.volume = { initial: volume0, last: volumes.at(-1), drift: Math.abs(volumes.at(-1)! - volume0) / volume0, peakDrift: Math.max(...volumes.map(v => Math.abs(v - volume0) / volume0)) };
}
if (engine === "brick" && rows.length) {
  const ratios = noBodies ? [NaN] : rows.filter(r => r.step >= 5).map(r => r.impulse![1]! / dt / buoyancy_N);
  summary.impulse = { buoyancy_N, ratioMin: Math.min(...ratios), ratioMax: Math.max(...ratios) };
  summary.peak = Math.max(...rows.map(r => r.peak ?? 0));
  summary.high = { k: highK, f: highF };
  summary.iterations = { mean: iterations.reduce((a, b) => a + b, 0) / Math.max(iterations.length, 1), n: iterations.length };
  if (byKernel.size) { const n = rows.filter(r => r.step >= 5).length; summary.profile = Object.fromEntries([...byKernel].sort((p, q) => q[1] - p[1]).map(([k, v]) => [k, +(v / PROFILE_STEPS).toFixed(4)])); }
}
log(summary);
solver?.destroy();
device.destroy();
process.exit(failure || errors.length ? 1 : 0);
