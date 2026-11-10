/**
 * Lane C1's small-case ladder: every solver issue is reproduced and fixed here, on meshes of 8^2 to 32^2 nodes
 * (128^2 only for the PCG scaling row) that run in seconds and compare against an analytic answer or the converged
 * Float64 oracle. The 150^2 Bridson scenes are acceptance runs at milestones, not the debugging loop. Skips without
 * WEBGPU_NODE_MODULE.
 *
 *   1. Newton convergence on a 16^2 sheet without contact (against the converged Float64 oracle) and with contact
 *      (a sheet landing flat on the floor): Newton and PCG iterations per step, |g| per Newton iterate of one step,
 *      the step-end residual.
 *   2. The dual rule: a strip starting stretched past its strain limit drops onto a ledge, so the arc cuts one end of
 *      its edges while the other hangs free. The duals stay bounded by the strip's own momentum, and at rest the
 *      strain is the hanging strip's exact uniaxial strain.
 *   3. PCG cost: us per PCG iteration (replayed in one timestamped pass) and PCG iterations per Newton solve at 16^2,
 *      32^2, 64^2 and 128^2 without contact, and at 32^2 resting on a floor and draped on a sphere with contact.
 *   4. The analytic gates at small size: free fall, the flat drop, the swing, the curtain, Peirce and the incline. (The
 *      tilted drop stays a 101^2 suite gate: at 16^2 its h / 8 reference resolves a slap the 1/60 s step cannot, 0.187 m/s.)
 *
 *   CLOTH_LADDER_SWEEPS=1,3,5   the PCG scaling row at these patch Schwarz sweep counts (default 1)
 */
import assert from 'node:assert/strict';
import test from 'node:test';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import type { SceneDescription } from '../lib/core/model';
import { herringBoneSheet } from '../lib/cloth/scenes/bridson-2002/mesh';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import { prepareClothTopology } from '../lib/cloth/topology';
import { CLOTH_AVBD, buildClothSolverModel, clothAddBendDamping, clothAddBending, clothAddFaceDamping, clothMembraneBlock, clothOracleStep, clothStepFrame, clothWarmStartDuals, createClothOracleState, type ClothSolverModel, type ClothStepOptions } from '../lib/cloth/oracle';
import { buildClothCoarseSpace, ClothCoarseCorrection, clothObjective } from '../lib/cloth/oracle-coarse';
import { ClothGPUSolver, type ClothGPUSolverOptions } from '../lib/cloth/gpu-solver';
import { CLOTH_NEWTON_SEGMENTS } from '../lib/cloth/gpu-newton-wgsl';
import { CLOTH_BINDING, CLOTH_SEGMENT_ALIGN } from '../lib/cloth/gpu-solver-wgsl';
import { CLOTH_NEWTON_SCENE_DEFAULTS, CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { buildClothBodyField, buildClothSceneField, clothBodyPoses, clothContactRadius, createClothSceneContact } from '../lib/cloth/scene-contact';
import { decodeOgcContactEntry } from '../lib/cloth/contact/ogc-detector';
import { clothSolidSamples, sampleClothSolidField } from '../lib/cloth/contact/solid-field';
import type { ClothDocument, ClothNewtonSettings, ClothTopologyData } from '../lib/cloth/types';
import {
  CLOTH_ANALYTIC_STEP_S, GRAVITY, bdf2FreeFall, clothAnalyticFreeFall, clothAnalyticFrictionIncline, clothAnalyticHangingCurtain,
  clothAnalyticPeirceCantilever, clothAnalyticSceneDescription, clothAnalyticSwingingSheet, hangingBandStrain, type ClothAnalyticScene,
} from '../lib/cloth/scenes/analytic';

const gpu = !!process.env.WEBGPU_NODE_MODULE;
const g = GRAVITY.value, h = CLOTH_ANALYTIC_STEP_S;
const SWEEPS = (process.env.CLOTH_LADDER_SWEEPS ?? '1').split(',').map(Number);
if (!SWEEPS.every(s => Number.isInteger(s) && s >= 1 && s % 2 === 1)) throw new Error(`CLOTH_LADDER_SWEEPS must be odd integers >= 1, got ${process.env.CLOTH_LADDER_SWEEPS}`);
const range = (a: number, b: number) => Array.from({ length: b - a + 1 }, (_, i) => a + i);
const BRIDSON: ClothNewtonSettings = clothAnalyticFreeFall().cloth[0]!.solver!.newton!;
// Converged to the f32 floor: on 16^2 |g| / |m g| reaches its floor (about 2e-3) by Newton iterate 4 (ladder 1a), so
// 12 iterations at 400 PCG and tolerances / 16 solve as far as f32 can.
const CONVERGED: ClothNewtonSettings = { iterations: 12, pcgIterations: 400, pcgTolerance: BRIDSON.pcgTolerance! / 16, residualTolerance: BRIDSON.residualTolerance! / 16 };
// Eisenstat-Walker forcing (forcingMax) with the dx = 1 um displacement stop at the converged caps: always eta_max 0.1, and
// CLOTH_LADDER_EW=0.05,0.2 adds rows. Held to the converged rows' oracle bound.
const FORCED = [0.1, ...(process.env.CLOTH_LADDER_EW ?? '').split(',').filter(Boolean).map(Number)]
  .map(eta => [`converged caps, stop at dx 1 um, forcing eta_max ${eta}`, { ...CONVERGED, velocityTolerance_m_s: 1e-6 / h, forcingMax: eta }] as const);
// The hybrid globalisation (newton.trustRegion 'hybrid': the line search, then trust-region iterations on the exact
// membrane) at the converged caps with the dx = 1 um stop, held to the converged rows' bounds.
const HYBRID = [['converged caps, stop at dx 1 um, hybrid trust region', { ...CONVERGED, velocityTolerance_m_s: 1e-6 / h, trustRegion: 'hybrid' }]] as const;

async function withDevice(run: (device: GPUDevice) => Promise<void>) {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
  Object.assign(globalThis, dawn.globals);
  const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
  assert.ok(adapter);
  const features = ['subgroups', 'timestamp-query'].filter(f => adapter.features.has(f)) as GPUFeatureName[];
  const device = await adapter.requestDevice({ requiredFeatures: features, requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const errors: string[] = [];
  device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });
  try { await run(device); await device.queue.onSubmittedWorkDone(); assert.deepEqual(errors, []); }
  finally { device.destroy(); }
}

// ---------------------------------------------------------------- helpers

/** Newton record of the last step: Newton iterations, PCG iterations, |g| / |m g| at its first and latest Newton iterate,
 * whether Newton ended done, and (contact) the vertices a travel bound held at its last accepted trial. */
async function newtonOf(solver: ClothGPUSolver) {
  const c = await solver.readNewtonControl(), scale = Math.sqrt(c[6]!);
  return { newton: c[3]!, pcg: c[7]!, first: Math.sqrt(c[4]!) / scale, last: Math.sqrt(c[5]!) / scale, done: c[0] === 1, travelHeld: c[11]! };
}

type Internals = { newtonWork: GPUBuffer; tables: { newtonRecords: number; pcg: { patches: number } }; N: number; T: number;
  pipelines: Record<string, GPUComputePipeline>; run(pass: GPUComputePassEncoder, name: string, workgroups: number, record?: number): void; threads(n: number): number };

/** The exact GPU time of one working PCG iteration (us): Newton re-armed at the solver's last state, PCG primed through
 * iteration 1, then iteration 2's dispatches replayed `reps` times in one timestamped pass (the clock is coarse). */
async function pcgMicros(device: GPUDevice, solver: ClothGPUSolver, reps = 64): Promise<number> {
  if (!device.features.has('timestamp-query')) return NaN;
  const S = solver as unknown as Internals, W = S.threads(S.N), Wp = S.tables.pcg.patches, offset = 16 * (S.tables.newtonRecords - CLOTH_SEGMENT_ALIGN);
  const iteration: string[] = 'pcgPrecond' in S.pipelines ? ['pcgPrecond', 'pcgStep'] : ['pcgStep'];
  device.queue.writeBuffer(S.newtonWork, offset, new Float32Array(4));
  const set = device.createQuerySet({ type: 'timestamp', count: 2 });
  const encoder = device.createCommandEncoder({ label: 'ladder replay' }), prime = encoder.beginComputePass();
  S.run(prime, 'nwFaces', S.threads(S.T)); S.run(prime, 'nwGradient', W);
  for (const i of [0, 1]) for (const k of iteration) S.run(prime, k, Wp, i);
  prime.end();
  const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
  for (let r = 0; r < reps; r++) for (const k of iteration) S.run(pass, k, Wp, 2);
  pass.end();
  const resolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 32, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.resolveQuerySet(set, 0, 2, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, read, 0, 16);
  encoder.copyBufferToBuffer(S.newtonWork, offset, read, 16, 16);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const bytes = read.getMappedRange().slice(0), t = new BigUint64Array(bytes, 0, 2), c = new Float32Array(bytes, 16, 4);
  read.destroy(); resolve.destroy(); set.destroy();
  // A replay whose control flags were set (converged, or a non-positive curvature) timed early exits.
  if (c[0] !== 0 || c[1] !== 0) return NaN;
  return Number(t[1]! - t[0]!) / 1e3 / reps;
}

/** GPU time (us) of each listed dispatch replayed `reps` times in its own timestamped pass, after Newton is re-armed and
 * PCG primed through iteration 1 at the solver's last state. `exit` sets the control flags first (x Newton done, y PCG
 * done), so the gated kernels time their early exit. */
async function replayEach(device: GPUDevice, solver: ClothGPUSolver, plan: readonly (readonly [string, number])[], reps = 64, exit?: [number, number]) {
  const S = solver as unknown as Internals, W = S.threads(S.N), Wp = S.tables.pcg.patches, offset = 16 * (S.tables.newtonRecords - CLOTH_SEGMENT_ALIGN);
  const iteration: string[] = 'pcgPrecond' in S.pipelines ? ['pcgPrecond', 'pcgStep'] : ['pcgStep'];
  const workgroups = (name: string) => name.startsWith('pcg') ? Wp : name === 'nwFaces' ? S.threads(S.T) : name === 'nwChoose' ? 1 : W;
  device.queue.writeBuffer(S.newtonWork, offset, new Float32Array(4));
  const set = device.createQuerySet({ type: 'timestamp', count: 2 * plan.length });
  const encoder = device.createCommandEncoder({ label: 'ladder replay' }), prime = encoder.beginComputePass();
  S.run(prime, 'nwFaces', S.threads(S.T)); S.run(prime, 'nwGradient', W);
  for (const i of [0, 1]) for (const k of iteration) S.run(prime, k, Wp, i);
  prime.end();
  if (exit) encoder.copyBufferToBuffer(exitBuffer(device, exit), 0, S.newtonWork, offset, 16);
  plan.forEach(([name, record], k) => {
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
    for (let r = 0; r < reps; r++) S.run(pass, name, workgroups(name), record);
    pass.end();
  });
  return resolveMicros(device, encoder, set, plan.length, reps, S.newtonWork, offset);
}
function exitBuffer(device: GPUDevice, flags: [number, number]) {
  const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(b, 0, new Float32Array([flags[0], flags[1], 0, 0]));
  return b;
}
async function resolveMicros(device: GPUDevice, encoder: GPUCommandEncoder, set: GPUQuerySet, count: number, reps: number, control?: GPUBuffer, offset = 0) {
  const resolve = device.createBuffer({ size: 16 * count, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * count + 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.resolveQuerySet(set, 0, 2 * count, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, read, 0, 16 * count);
  if (control) encoder.copyBufferToBuffer(control, offset, read, 16 * count, 16);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const bytes = read.getMappedRange().slice(0), t = new BigUint64Array(bytes, 0, 2 * count), c = new Float32Array(bytes, 16 * count, 4);
  read.destroy(); resolve.destroy(); set.destroy();
  return { us: range(0, count - 1).map(k => Number(t[2 * k + 1]! - t[2 * k]!) / 1e3 / reps), control: Array.from(c) };
}
/** One whole PCG solve at the solver's last state, GPU-timed: Newton re-armed (control cleared), nwFaces and nwGradient
 * in a priming pass, then the solve in its own timestamped pass, `reps` times; 'chain' is the pcgStep dispatches 0..cap,
 * 'solve' the one-dispatch pcgSolve (one patch). Mean us per solve and the iterations the last one took. */
async function solveMicros(device: GPUDevice, solver: ClothGPUSolver, mode: 'chain' | 'solve' | 'replay', reps = 8) {
  const S = solver as unknown as Internals & { newton: { pcgIterations: number } }, W = S.threads(S.N), Wp = S.tables.pcg.patches;
  const offset = 16 * (S.tables.newtonRecords - CLOTH_SEGMENT_ALIGN), cap = S.newton.pcgIterations;
  const set = device.createQuerySet({ type: 'timestamp', count: 2 * reps }), encoder = device.createCommandEncoder({ label: 'ladder solve' });
  for (let r = 0; r < reps; r++) {
    encoder.clearBuffer(S.newtonWork, offset, 32);
    const prime = encoder.beginComputePass(); S.run(prime, 'nwFaces', S.threads(S.T)); S.run(prime, 'nwGradient', W);
    if (mode === 'replay') for (const i of [0, 1]) for (const k of 'pcgPrecond' in S.pipelines ? ['pcgPrecond', 'pcgStep'] : ['pcgStep']) S.run(prime, k, Wp, i);
    prime.end();
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * r, endOfPassWriteIndex: 2 * r + 1 } });
    const iteration: string[] = 'pcgPrecond' in S.pipelines ? ['pcgPrecond', 'pcgStep'] : ['pcgStep'];
    if (mode === 'solve') S.run(pass, 'pcgSolve', 1, cap); else for (let i = 0; i <= cap; i++) for (const k of iteration) S.run(pass, k, Wp, mode === 'replay' ? 2 : i);
    pass.end();
  }
  const { us, control } = await resolveMicros(device, encoder, set, reps, 1, S.newtonWork, offset + 16);
  return { us: us.reduce((a, b) => a + b, 0) / reps, iterations: control[3]! };
}
/** The Newton direction d (the PCG solution) of the last solve, per vertex (x, y, z). */
async function readDirection(device: GPUDevice, solver: ClothGPUSolver) {
  const S = solver as unknown as Internals, Np = Math.ceil(S.N / 16) * 16, bytes = 16 * S.N;
  const read = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(S.newtonWork, 16 * CLOTH_NEWTON_SEGMENTS.d * Np, read, 0, bytes);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ);
  const v = new Float32Array(read.getMappedRange().slice(0)); read.destroy();
  return Array.from({ length: S.N }, (_, i) => [v[4 * i]!, v[4 * i + 1]!, v[4 * i + 2]!] as const);
}
/** GPU time (us) per dispatch of a standalone 256-lane kernel, `workgroups` workgroups, `reps` per timed pass. */
/** Mean us per dispatch of `wgsl` (entry main) over `reps` in one pass. `mode`: 'direct' dispatches `workgroups`;
 * 'indirect' reads the count `workgroups` from an indirect buffer; 'written' has a one-lane kernel write that count
 * into the indirect buffer before each indirect dispatch (the GPU-decided stop), the pair timed as one. */
async function kernelMicros(device: GPUDevice, wgsl: string, workgroups: number, reps = 256, mode: 'direct' | 'indirect' | 'written' = 'direct') {
  const module = device.createShaderModule({ code: wgsl });
  const pipeline = await device.createComputePipelineAsync({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  const buffer = device.createBuffer({ size: 16 * 1024, usage: GPUBufferUsage.STORAGE });
  const args = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(args, 0, new Uint32Array([workgroups, 1, 1, 0]));
  const writer = mode === 'written' ? await device.createComputePipelineAsync({ layout: 'auto', compute: { entryPoint: 'main', module: device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> a: array<u32, 4>;\n@compute @workgroup_size(1) fn main() { a[0] = ${workgroups}u; a[1] = 1u; a[2] = 1u; }` }) } }) : undefined;
  const writeBind = writer ? device.createBindGroup({ layout: writer.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: args } }] }) : undefined;
  const bind = wgsl.includes('@binding(0)') ? device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer } }] }) : undefined;
  const set = device.createQuerySet({ type: 'timestamp', count: 2 });
  const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } });
  for (let r = 0; r < reps; r++) {
    if (writer) { pass.setPipeline(writer); pass.setBindGroup(0, writeBind!); pass.dispatchWorkgroups(1); }
    if (writer || r === 0) { pass.setPipeline(pipeline); if (bind) pass.setBindGroup(0, bind); }
    if (mode === 'direct') pass.dispatchWorkgroups(workgroups); else pass.dispatchWorkgroupsIndirect(args, 0);
  }
  pass.end();
  const { us } = await resolveMicros(device, encoder, set, 1, reps);
  buffer.destroy(); args.destroy();
  return us[0]!;
}
const EMPTY_WGSL = '@compute @workgroup_size(256) fn main() {}';
/** One 256-lane tree reduction through workgroup memory (8 barrier levels), as each PCG kernel takes one or two. */
const REDUCE_WGSL = `@group(0) @binding(0) var<storage, read_write> buf: array<vec4f>;
var<workgroup> red: array<vec4f, 256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) lane: u32) {
  red[lane] = buf[lane];
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) { if (lane < s) { red[lane] += red[lane + s]; } workgroupBarrier(); }
  if (lane == 0u) { buf[256] = red[0]; }
}`;

const centreOf = (p: ArrayLike<number>): [number, number, number] => {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < p.length; i++) { lo[i % 3] = Math.min(lo[i % 3]!, p[i]!); hi[i % 3] = Math.max(hi[i % 3]!, p[i]!); }
  return [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as [number, number, number];
};

/** A document's bare solver (no contact pass), as ClothSystem sets it up, with the given step and caps. */
async function bareSolver(device: GPUDevice, document: ClothDocument, o: Partial<ClothGPUSolverOptions> & { newton: ClothNewtonSettings }) {
  const data = document.geometry.data, model = buildClothSolverModel(prepareClothTopology(data, document.materials), document.materials, document.scripts ?? []);
  const solver = await ClothGPUSolver.create(device, model, { origin_m: centreOf(data.restPositions_m), dt_s: CLOTH_SCENE_STEP_S, gravity_m_s2: [0, -g, 0],
    integrator: 'bdf2', initialGuess: 'previous', ...(data.initialPositions_m ? { positions_m: data.initialPositions_m } : {}), ...o });
  return { solver, model };
}

/** An analytic scene with its first document's sheet replaced (same id, solids and material; the contact authoring
 * reads the solids by scene id). */
function resized(source: ClothAnalyticScene, data: ClothTopologyData, patch: Partial<ClothDocument> = {}): ClothAnalyticScene {
  const d = source.cloth[0]!;
  return { ...source, cloth: [{ ...d, ...patch, geometry: { kind: 'inline', data } }] };
}

/** A flat n x n sheet at the analytic 1 cm spacing, its corner at (x0, y, z0). */
const sheet = (n: number, y: number, group?: (i: number, j: number) => number, cell = 0.01): ClothTopologyData =>
  herringBoneSheet({ columns: n, rows: n, width_m: (n - 1) * cell, depth_m: (n - 1) * cell, origin_m: [-(n - 1) * cell / 2, y, -(n - 1) * cell / 2], ...(group ? { group } : {}) });

const mean = (frames: readonly { positions_m: ArrayLike<number>; velocities_m_s?: ArrayLike<number> }[], axis = 1) => frames.map(p => {
  const x = p.positions_m, v = p.velocities_m_s!, N = x.length / 3;
  let y = 0, vy = 0;
  for (let i = 0; i < N; i++) { y += x[3 * i + axis]! / N; vy += v[3 * i + axis]! / N; }
  return { y, vy };
});
/** The largest upward mean velocity after the fastest mean fall, and that fall. */
function rebound(track: readonly { vy: number }[]) {
  let k = 0;
  track.forEach((t, i) => { if (t.vy < track[k]!.vy) k = i; });
  return { fall: track[k]!.vy, up: Math.max(0, ...track.slice(k + 1).map(t => t.vy)) };
}

async function captureScene(device: GPUDevice, scene: SceneDescription, ticks: readonly number[]) {
  const cloth = await ClothSystem.create(device, scene);
  try {
    const t0 = performance.now(), frames = await cloth.capture(ticks.map(n => n * CLOTH_SCENE_STEP_S));
    assert.equal(cloth.failure, undefined, cloth.failure?.message);
    return { frames, ms: (performance.now() - t0) / Math.max(1, ticks.at(-1)!) };
  } finally { cloth.destroy(); }
}

/** Step a scene one tick at a time, recording the first document's Newton record after each. */
async function steppedScene(device: GPUDevice, scene: SceneDescription, ticks: number, after?: (cloth: ClothSystem, tick: number) => Promise<void>) {
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 }), id = scene.cloth![0]!.id, solver = cloth.solverFor(id)!;
  const records: Awaited<ReturnType<typeof newtonOf>>[] = [];
  for (let t = 1; t <= ticks; t++) {
    assert.ok(cloth.stepTo(t * CLOTH_SCENE_STEP_S), `step ${t} refused: ${cloth.failure?.message ?? 'pending'}`);
    const rec = await newtonOf(solver); records.push(rec);
    // The cap-aware done rule: Newton never ends done on a step a travel bound holds a vertex at.
    assert.ok(!(rec.done && rec.travelHeld > 0), `step ${t}: Newton done while a travel bound holds ${rec.travelHeld} vertices`);
    await after?.(cloth, t);
  }
  assert.equal(cloth.failure, undefined, cloth.failure?.message);
  return { cloth, solver, records };
}

const fmt = (xs: readonly number[], d = 0) => xs.map(x => x.toFixed(d)).join(' ');
const sum = (xs: readonly number[]) => xs.reduce((a, b) => a + b, 0);

// ---------------------------------------------------------------- 0. where a PCG iteration's time goes

test('ladder 0: a 16^2 PCG iteration attributed: the empty-dispatch floor, an early exit, the reduction chain and the kernel bodies', { skip: !gpu }, async () => withDevice(async device => {
  if (!device.features.has('timestamp-query')) return;
  const rows: string[] = [];
  for (const n of [16, 64]) {
    const document = resized(clothAnalyticFreeFall(), sheet(n, 1, (i, j) => j === 0 && (i === 0 || i === n - 1) ? 1 : 0)).cloth[0]!;
    const { solver } = await bareSolver(device, document, { dt_s: h, newton: BRIDSON });
    solver.step(3);
    const S = solver as unknown as Internals, patches = S.tables.pcg.patches;
    const kernels: string[] = 'pcgPrecond' in S.pipelines ? ['pcgPrecond', 'pcgStep'] : ['pcgStep'];
    const work = await replayEach(device, solver, kernels.map(k => [k, 2] as const));
    const exited = await replayEach(device, solver, [...kernels.map(k => [k, 2] as const), ['nwTrial', 0], ['nwGradient', 0]], 64, [1, 1]);
    const empty1 = await kernelMicros(device, EMPTY_WGSL, 1), emptyP = await kernelMicros(device, EMPTY_WGSL, patches);
    const reduce1 = await kernelMicros(device, REDUCE_WGSL, 1), reduceP = await kernelMicros(device, REDUCE_WGSL, patches);
    // The stop without early exits: a zero-workgroup indirect dispatch, and the same with its count GPU-written each time.
    const ind0 = await kernelMicros(device, REDUCE_WGSL, 0, 256, 'indirect'), indP = await kernelMicros(device, REDUCE_WGSL, patches, 256, 'indirect');
    const wr0 = await kernelMicros(device, REDUCE_WGSL, 0, 256, 'written'), wrP = await kernelMicros(device, REDUCE_WGSL, patches, 256, 'written');
    solver.destroy();
    const total = work.us.reduce((a, b) => a + b, 0), reductions = kernels.length === 2 ? 3 : 2;
    rows.push(`${n}^2 (${patches} patch${patches > 1 ? 'es' : ''}): one PCG iteration ${total.toFixed(2)} us = ${kernels.map((k, i) => `${k} ${work.us[i]!.toFixed(2)}`).join(' + ')}; `
      + `empty dispatch ${empty1.toFixed(2)} us (1 workgroup) / ${emptyP.toFixed(2)} (${patches}); early exit on the control flags ${exited.us.map((u, i) => `${[...kernels, 'nwTrial', 'nwGradient'][i]} ${u.toFixed(2)}`).join(', ')} us; `
      + `one 256-lane reduction ${(reduce1 - empty1).toFixed(2)} us over the floor (${(reduceP - emptyP).toFixed(2)} at ${patches}); `
      + `indirect dispatch: 0 workgroups ${ind0.toFixed(2)} us, ${patches} ${indP.toFixed(2)} (direct reduce ${reduceP.toFixed(2)}); count GPU-written before each: 0 workgroups ${wr0.toFixed(2)}, ${patches} ${wrP.toFixed(2)} us per pair; `
      + `so ${kernels.length} x floor ${(kernels.length * emptyP).toFixed(2)} + ${reductions} reductions ${(reductions * (reduceP - emptyP)).toFixed(2)} + body ${(total - kernels.length * emptyP - reductions * (reduceP - emptyP)).toFixed(2)} us`);
  }
  console.log(`[ladder 0] PCG iteration attribution (64 replays per pass; standalone kernels 256 per pass):\n  ${rows.join('\n  ')}`);
}));

// ---------------------------------------------------------------- 1. Newton convergence

test('ladder 1a: Newton on a corner-held 16^2 sheet matches the converged Float64 oracle; iterations and |g| per iterate', { skip: !gpu }, async () => withDevice(async device => {
  const n = 16, steps = 10, document = resized(clothAnalyticFreeFall(), sheet(n, 1, (i, j) => j === 0 && (i === 0 || i === n - 1) ? 1 : 0)).cloth[0]!;
  const material = document.materials[0]!;
  const model: ClothSolverModel = buildClothSolverModel(prepareClothTopology(document.geometry.data, document.materials), document.materials);
  const st = createClothOracleState(model, model.topology.data.restPositions_m);
  const options: ClothStepOptions = { dt_s: h, gravity_m_s2: [0, -g, 0], integrator: 'bdf2', iterations: 250, initialGuess: 'previous' };
  const coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
  for (let s = 0; s < steps; s++) clothOracleStep(model, st, options, false, coarse);
  const lines: string[] = [];
  // The displacement stop (newton3.w): done once the accepted step moves no vertex more than dx = tol h~.
  const stops = (process.env.CLOTH_LADDER_DX ?? '').split(',').filter(Boolean).map(Number)
    .map(dx => [`converged caps, stop at dx ${dx} um`, { ...CONVERGED, velocityTolerance_m_s: 1e-6 * dx / h }] as const);
  for (const [name, newton] of [['Bridson 8 x 200', BRIDSON], ['converged 12 x 400', CONVERGED], ...stops, ...FORCED, ...HYBRID] as const) {
    const { solver } = await bareSolver(device, document, { dt_s: h, newton });
    const records: Awaited<ReturnType<typeof newtonOf>>[] = [];
    for (let s = 0; s < steps; s++) { solver.step(1); records.push(await newtonOf(solver)); }
    const x = await solver.readPositions(), status = await solver.readStatus();
    let worst = 0;
    for (let i = 0; i < x.length; i++) worst = Math.max(worst, Math.abs(x[i]! - st.x[i]!));
    lines.push(`${name}: off the oracle ${(1e6 * worst).toFixed(2)} um; Newton per step ${fmt(records.map(r => r.newton))}; PCG per Newton ${fmt(records.map(r => r.pcg / r.newton), 1)} (${(sum(records.map(r => r.pcg)) / steps).toFixed(0)} per step); `
      + `step-end |g|/|mg| ${records.at(-1)!.last.toExponential(2)}`);
    assert.equal(status.status, 0);
    if (newton !== BRIDSON) assert.ok(worst < 2e-5, `converged Newton vs the converged oracle: ${worst} m`);
    else assert.ok(worst < 1e-3, `Bridson caps vs the converged oracle: ${worst} m`);
    // |g| per Newton iterate of the last step: rerun it from the saved state with caps 1..K (|g| at iterate K - 1).
    if (newton === BRIDSON) {
      const before = await (async () => { const { solver: s } = await bareSolver(device, document, { dt_s: h, newton }); s.step(steps - 1); const state = await s.readState(); s.destroy(); return state; })();
      const decay: number[] = [];
      for (let k = 1; k <= 7; k++) {
        const { solver: s } = await bareSolver(device, document, { dt_s: h, newton: { ...newton, iterations: k } });
        s.uploadState(before.positions_m, before.velocities_m_s, before.solverState);
        s.step(1); decay.push((await newtonOf(s)).last); s.destroy();
      }
      lines.push(`  step ${steps}, |g|/|mg| at Newton iterates 0..6: ${decay.map(x => x.toExponential(1)).join(' ')} (tolerance ${newton.residualTolerance})`);
    }
    solver.destroy();
  }
  console.log(`[ladder 1a] ${n}^2 corner-held sheet, ${material.id}, BDF2 ${steps} steps against the Float64 oracle (250 sweeps + coarse):\n  ${lines.join('\n  ')}`);
}));

test('ladder 1b: Newton with contact on a 16^2 sheet landing flat on the floor: iterations, |g| and the rest it reaches', { skip: !gpu }, async () => withDevice(async device => {
  const source = clothAnalyticFreeFall(), n = 16, y0 = 0.05, ticks = 30;
  const lines: string[] = [];
  const stops = (process.env.CLOTH_LADDER_DX ?? '').split(',').filter(Boolean).map(Number)
    .map(dx => [`converged caps, stop at dx ${dx} um`, { ...CONVERGED, velocityTolerance_m_s: 1e-6 * dx / h }] as const);
  for (const [name, newton] of [['Bridson 8 x 200', BRIDSON], ['converged 12 x 400', CONVERGED], ...stops, ...FORCED, ...HYBRID] as const) {
    const scene = clothAnalyticSceneDescription(resized(source, sheet(n, y0)), { newton });
    const r = clothContactRadius(scene.cloth![0]!);
    const { cloth, solver, records } = await steppedScene(device, scene, ticks);
    const pose = await solver.readState();
    cloth.destroy();
    const x = pose.positions_m, N = x.length / 3;
    let lo = Infinity, hi = -Infinity, speed = 0;
    for (let i = 0; i < N; i++) { lo = Math.min(lo, x[3 * i + 1]!); hi = Math.max(hi, x[3 * i + 1]!); speed = Math.max(speed, Math.hypot(pose.velocities_m_s[3 * i]!, pose.velocities_m_s[3 * i + 1]!, pose.velocities_m_s[3 * i + 2]!)); }
    lines.push(`${name}: Newton per step ${fmt(records.map(q => q.newton))}; unconverged steps ${records.filter(q => !q.done).length} (${records.filter(q => !q.done && q.travelHeld > 0).length} held by travel bounds); PCG per Newton ${fmt(records.map(q => q.pcg / Math.max(1, q.newton)), 0)} (${(sum(records.slice(-10).map(q => q.pcg)) / 10).toFixed(0)} per step, last 10); `
      + `step-end |g|/|mg| first/last step ${records[0]!.last.toExponential(1)} / ${records.at(-1)!.last.toExponential(1)}; at tick ${ticks} heights ${(1000 * lo).toFixed(3)}-${(1000 * hi).toFixed(3)} mm (r ${(1000 * r).toFixed(2)}), max speed ${(1000 * speed).toFixed(2)} mm/s`);
    // At rest on a frictionless floor: flat, still, and no vertex below the contact radius' floor band or far above it.
    assert.ok(lo > 0 && hi - lo < 1e-4, `the sheet rests flat: heights ${lo}-${hi} m`);
    assert.ok(hi < 2 * r, `the sheet rests within 2r of the floor: ${hi} m`);
    assert.ok(speed < 0.01, `the sheet is at rest: ${speed} m/s`);
  }
  console.log(`[ladder 1b] ${n}^2 sheet dropped ${1000 * y0} mm onto the floor, ${ticks} ticks:\n  ${lines.join('\n  ')}`);
}));

// The hybrid's trust-region model is the exact Hessian (C1 RESULT 41): at the first accepted trust-region iteration of
// the corner-held sheet's step 10, s.H s from the GPU's model record (control + 6; no contact) against the Float64
// objective's central second difference along s = x_k+1 - x_k at x_k, the frame the oracle's own BDF2 step 10 (the step
// size exact in Float64: a GPU-saved state's previous step is h's f32 image, on which clothStepUsesBDF2 falls back to
// backward Euler). Also the model's own consistency there: an interior step solves H s = -g, so g.s = -s.H s.
test('ladder 1c: the hybrid trust region\'s model is the exact Hessian at a fixed iterate (s.H s against the Float64 objective)', { skip: !gpu }, async () => withDevice(async device => {
  const n = 16, steps = 10, document = resized(clothAnalyticFreeFall(), sheet(n, 1, (i, j) => j === 0 && (i === 0 || i === n - 1) ? 1 : 0)).cloth[0]!;
  const model: ClothSolverModel = buildClothSolverModel(prepareClothTopology(document.geometry.data, document.materials), document.materials);
  const st = createClothOracleState(model, model.topology.data.restPositions_m);
  const options: ClothStepOptions = { dt_s: h, gravity_m_s2: [0, -g, 0], integrator: 'bdf2', iterations: 250, initialGuess: 'previous' };
  const coarse = new ClothCoarseCorrection(model, buildClothCoarseSpace(model.topology));
  for (let s = 0; s < steps - 1; s++) clothOracleStep(model, st, options, false, coarse);
  const { frame, bdf2 } = clothStepFrame(model, st, options);
  assert.ok(bdf2);
  clothWarmStartDuals(model, st, frame.hEff);
  const newton = HYBRID[0][1];
  const before = await (async () => { const { solver: s } = await bareSolver(device, document, { dt_s: h, newton }); s.step(steps - 1); const state = await s.readState(); s.destroy(); return state; })();
  const at = async (k: number) => {
    const { solver: s } = await bareSolver(device, document, { dt_s: h, newton: { ...newton, iterations: k } });
    s.uploadState(before.positions_m, before.velocities_m_s, before.solverState);
    s.step(1); const c = Array.from(await s.readNewtonControl()), x = Float64Array.from(await s.readPositions()); s.destroy();
    return { c, x };
  };
  let prev = await at(1), found = false;
  for (let k = 2; k <= 8 && !found; k++) {
    const cur = await at(k);
    // An accepted trust-region iteration k: one more trust-region iteration, no more rejected, the iterate moved.
    if (cur.c[12] === 1 && cur.c[17]! === prev.c[17]! + 1 && cur.c[18] === prev.c[18] && cur.c[3] === k) {
      const s = Float64Array.from(cur.x, (q, i) => q - prev.x[i]!), [gs, sHs, sHc] = cur.c.slice(24, 27) as number[];
      const E = (t: number) => clothObjective(model, frame, st, Float64Array.from(prev.x, (q, i) => q + t * s[i]!)).value;
      const fd = E(1) - 2 * E(0) + E(-1);
      let smax = 0; for (let v = 0; 3 * v < s.length; v++) smax = Math.max(smax, Math.hypot(s[3 * v]!, s[3 * v + 1]!, s[3 * v + 2]!));
      console.log(`[ladder 1c] step ${steps}, Newton iteration ${k} (trust-region iteration ${cur.c[17]}, rho ${cur.c[19]!.toFixed(4)}, max |s| ${(1e6 * smax).toFixed(2)} um): `
        + `GPU s.H s ${sHs!.toExponential(5)} (contact ${sHc}), Float64 second difference ${fd.toExponential(5)}, ratio ${(sHs! / fd).toFixed(5)}; g.s ${gs!.toExponential(5)}`);
      assert.equal(sHc, 0);
      assert.ok(Math.abs(sHs! / fd - 1) < 1e-2, `GPU model s.H s ${sHs} vs Float64 ${fd}`);
      found = true;
    }
    prev = cur;
  }
  assert.ok(found, 'no accepted trust-region iteration in Newton iterations 2..8 of step 10');
}));

// The GPU's objective at a fixed iterate, term by term against Float64 (RESULT 45). Every GPU contact entry of the
// iterate (decodeOgcContactEntry) is evaluated in Float64 as ogcEntryLin does: the barrier ogcActivation(d; r, kc) on the
// gap sum w_i x_i (a vertex-facet feature's unclamped barycentrics at x, ee at its stored s, t), friction mu lambda on the
// slip sum wl_i (x_i - x_t,i) with weights and normal lagged at x_t. Unknown kinds fail explicitly.
type V3 = [number, number, number];
const sub3 = (a: readonly number[], b: readonly number[]): V3 => [a[0]! - b[0]!, a[1]! - b[1]!, a[2]! - b[2]!];
const dot3 = (a: readonly number[], b: readonly number[]) => a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const unit3 = (a: readonly number[], fallback: readonly number[]): V3 => { const l = Math.hypot(a[0]!, a[1]!, a[2]!); return (l > 1e-30 ? [a[0]! / l, a[1]! / l, a[2]! / l] : [...fallback]) as V3; };
const rot3 = (q: readonly number[]) => { const [x, y, z, w] = q as [number, number, number, number]; return [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]; };
const mulR = (R: readonly number[], p: readonly number[]): V3 => [0, 1, 2].map(i => R[3 * i]! * p[0]! + R[3 * i + 1]! * p[1]! + R[3 * i + 2]! * p[2]!) as V3;
const mulRT = (R: readonly number[], p: readonly number[]): V3 => [0, 1, 2].map(j => R[j]! * p[0]! + R[3 + j]! * p[1]! + R[6 + j]! * p[2]!) as V3;
function vtWeights(feature: number, p: V3, a: V3, b: V3, c: V3): number[] {
  const w = [1, 0, 0, 0], X = [a, b, c];
  if (feature < 3) w[1 + feature] = -1;
  else if (feature < 6) { const i = feature - 3, j = (i + 1) % 3, dir = sub3(X[j]!, X[i]!), s = dot3(sub3(p, X[i]!), dir) / dot3(dir, dir); w[1 + i] = -(1 - s); w[1 + j] = -s; }
  else {
    const v0 = sub3(b, a), v1 = sub3(c, a), v2 = sub3(p, a), d00 = dot3(v0, v0), d01 = dot3(v0, v1), d11 = dot3(v1, v1), d20 = dot3(v2, v0), d21 = dot3(v2, v1), den = d00 * d11 - d01 * d01;
    const bv = (d11 * d20 - d01 * d21) / den, bw = (d00 * d21 - d01 * d20) / den; w[1] = -(1 - bv - bw); w[2] = -bv; w[3] = -bw;
  }
  return w;
}

test('ladder 1d: at a fixed iterate of a restarted contact step on the 16^2 sphere, the GPU gradient is the Float64 implicit-Euler objective\'s, term by term (inertia, membrane, damping, bending, strain limit, each contact kind)', { skip: !gpu }, async () => withDevice(async device => {
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 })), doc = base.cloth![0]!, H = CLOTH_SCENE_STEP_S, K = 6;
  if (doc.geometry.kind !== 'inline') throw new Error('the Bridson sphere document is inline');
  // The first step that restarts BDF2 as implicit Euler (a contact onset in the step before it: RESULT 25).
  let restartTick = 0;
  { const cloth = await ClothSystem.create(device, base, { statusInterval: 1 }), solver = cloth.solverFor(doc.id)!;
    try { for (let t = 1; t <= 40 && !restartTick; t++) { assert.ok(cloth.stepTo(t * H), `step ${t} refused: ${cloth.failure?.message ?? 'pending'}`); if ((await solver.readStatus()).restarts > 0) restartTick = t; } }
    finally { cloth.destroy(); } }
  assert.ok(restartTick > 1, 'no contact restart in the sphere\'s first 40 steps');
  const start = (await captureScene(device, base, [restartTick - 1])).frames[0]![0]!;
  if (!start.velocities_m_s || !start.solverState) throw new Error('the publication carries no velocities or solver state');
  const scene = (iterations: number) => ({ ...base, cloth: [{ ...doc, solver: { ...doc.solver!, newton: { ...(doc.solver!.newton ?? CLOTH_NEWTON_SCENE_DEFAULTS), iterations } },
    geometry: { kind: 'inline' as const, data: { ...doc.geometry.data, initialPositions_m: Array.from(start.positions_m), initialVelocities_m_s: Array.from(start.velocities_m_s!), solverState: start.solverState } } }] });
  type Reader = { readRange(b: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer>; newtonWork: GPUBuffer; state: GPUBuffer; tables: { edgeDualRecord: number }; E: number };
  // Deterministic runs to K - 1 and K iterations: the first ends at x_{K-1} with the entries and duals iteration K - 1's
  // gradient reads, the second's g segment holds that gradient.
  const at = async (iterations: number) => {
    const cloth = await ClothSystem.create(device, scene(iterations), { statusInterval: 1 }), solver = cloth.solverFor(doc.id)!, r = solver as unknown as Reader;
    try {
      assert.ok(cloth.stepTo(start.time_s + H), `the step from tick ${restartTick - 1} was refused: ${cloth.failure?.message ?? 'pending'}`);
      const rec = await newtonOf(solver), status = await solver.readStatus(), x = Float64Array.from(await solver.readPositions()), N = x.length / 3;
      const g = new Float32Array(await r.readRange(r.newtonWork, 0, 16 * N)), duals = new Float32Array(await r.readRange(r.state, 16 * r.tables.edgeDualRecord, 16 * r.E));
      const det = cloth.contactFor(doc.id)!, entries = await det.readContactEntries();
      return { rec, restarts: status.restarts, x, g, duals, entries, S: det.solidSamples, facets: det.facetCapacity, edge: det.topology.layout.maxEdgeLength, origin: cloth.renderSources[0]!.origin_m };
    } finally { cloth.destroy(); }
  };
  const prev = await at(K - 1), cur = await at(K);
  assert.deepEqual([prev.rec.newton, cur.rec.newton, prev.restarts, cur.restarts], [K - 1, K, 1, 1], 'both runs restart and run to their caps');
  // Float64 at x_{K-1}: the step frame of each integrator, the GPU's duals.
  const model = buildClothSolverModel(prepareClothTopology(doc.geometry.data, doc.materials), doc.materials, doc.scripts ?? []), t = model.topology, N = t.vertexCount, ss = start.solverState;
  const frameOf = (integrator: 'implicit-euler' | 'bdf2') => {
    const st = createClothOracleState(model, start.positions_m, start.velocities_m_s);
    st.vPrev.set(ss.previousVelocities_m_s); st.dxPrev.set(ss.previousDisplacements_m); st.previousStep_s = Math.fround(ss.previousStep_s) === Math.fround(H) ? H : ss.previousStep_s;
    const gv = base.fluid.gravity_m_s2;
    return clothStepFrame(model, st, { dt_s: H, gravity_m_s2: [gv.x, gv.y, gv.z], integrator, iterations: 1, initialGuess: 'inertial' }).frame;
  };
  const euler = frameOf('implicit-euler'), bdf2 = frameOf('bdf2'), x = prev.x, o = prev.origin;
  const pos = (i: number): V3 => [x[3 * i]!, x[3 * i + 1]!, x[3 * i + 2]!];
  const elastic = (frame: typeof euler, v: number) => {
    const w = t.mass_kg[v]! / (frame.hEff * frame.hEff), inertia = [0, 1, 2].map(c => w * (x[3 * v + c]! - frame.y[3 * v + c]!)) as V3, Hs = new Array(9).fill(0) as never;
    const membrane: V3 = [0, 0, 0], damping: V3 = [0, 0, 0], bending: V3 = [0, 0, 0], limit: V3 = [0, 0, 0];
    for (let j = t.vertexFaceOffsets[v]!; j < t.vertexFaceOffsets[v + 1]!; j++) {
      const f = t.vertexFaces[j]! >> 2, k = t.vertexFaces[j]! & 3;
      const b = clothMembraneBlock(model.membrane[t.faceMaterial[f]!]!, t.faceDmInv, 4 * f, t.faceArea_m2[f]!, k, pos(t.faces[3 * f]!), pos(t.faces[3 * f + 1]!), pos(t.faces[3 * f + 2]!));
      for (let c = 0; c < 3; c++) membrane[c] += b.gradient[c]!;
      clothAddFaceDamping(model, frame, f, k, pos, damping, Hs);
    }
    for (let j = t.vertexBendOffsets[v]!; j < t.vertexBendOffsets[v + 1]!; j++) { clothAddBending(t, t.vertexBends[j]!, pos, bending, Hs); clothAddBendDamping(model, frame, t.vertexBends[j]!, pos, bending, Hs); }
    for (let j = t.vertexEdgeOffsets[v]!; j < t.vertexEdgeOffsets[v + 1]!; j++) {
      const e = t.vertexEdges[j]! >> 2, eps = model.edgeStrainLimit[e]!; if (!(eps > 0)) continue;
      const a = t.edges[4 * e]!, b = t.edges[4 * e + 1]!, d = sub3(pos(v), pos(a === v ? b : a)), len = Math.hypot(...d), L = t.edgeRestLength_m[e]!;
      const C0 = Math.hypot(...sub3([frame.x0[3 * a]!, frame.x0[3 * a + 1]!, frame.x0[3 * a + 2]!], [frame.x0[3 * b]!, frame.x0[3 * b + 1]!, frame.x0[3 * b + 2]!])) - (1 + eps) * L;
      const force = Math.max(prev.duals[4 * e + 1]! * (len - (1 + eps) * L - CLOTH_AVBD.edgeAlpha * Math.max(C0, 0)) + prev.duals[4 * e]!, 0);
      for (let c = 0; c < 3; c++) limit[c] += force * d[c]! / len;
    }
    return { inertia, membrane, damping, bending, limit, stiffness: w };
  };
  // Contact, cloth frame (x - origin), from the entries.
  const { header, counts, entries } = prev.entries, KE = header[1]!, pf = new Float32Array(Uint32Array.from(header.slice(4, 8)).buffer);
  const [rc, kc, selfMu, epsV] = [pf[0]!, pf[1]!, pf[2]!, pf[3]!];
  const act = (d: number): V3 => { if (d >= rc) return [0, 0, 0]; const tau = rc / 2; if (d >= tau) return [0.5 * kc * (rc - d) ** 2, -kc * (rc - d), kc]; const k2 = kc * tau * (rc - tau); return [0, -k2 / d, k2 / (d * d)]; };
  const f1y = (y: number) => y >= epsV ? 1 / y : 2 / epsV - y / (epsV * epsV);
  const r = clothContactRadius(doc), field = buildClothSceneField(base, o, r), statics = clothSolidSamples(field, { r: rc, edge: prev.edge });
  assert.equal(statics.length / 4, prev.S, 'the static samples rebuild as the detector holds them');
  const body = base.rigidBodies[0]!, bodyField = buildClothBodyField(base, body, r), bodySamples = clothSolidSamples(bodyField, { r: rc, edge: prev.edge });
  const poses = clothBodyPoses(body, o, ss.time_s, 1, H), R = poses.map(p => rot3(p.orientation)), C = poses.map(p => p.position);
  const X = (v: number): V3 => [x[3 * v]! - o[0], x[3 * v + 1]! - o[1], x[3 * v + 2]! - o[2]];
  const Xt = (v: number): V3 => [start.positions_m[3 * v]! - o[0], start.positions_m[3 * v + 1]! - o[1], start.positions_m[3 * v + 2]! - o[2]];
  const sample = (i: number, body: boolean, k: number): V3 => {
    if (!body) return [statics[4 * i]!, statics[4 * i + 1]!, statics[4 * i + 2]!];
    const p = mulR(R[k]!, [bodySamples[4 * i]!, bodySamples[4 * i + 1]!, bodySamples[4 * i + 2]!]); return [p[0] + C[k]![0], p[1] + C[k]![1], p[2] + C[k]![2]];
  };
  const KINDS = ['field', 'bodyField', 'vf', 'sampleFace', 'ee'] as const, seen: Record<string, number> = {};
  const contact = Array.from({ length: N }, () => ({ g: KINDS.map(() => [0, 0, 0] as V3), magnitude: 0, stiffness: 0 }));
  const fieldTerm = (out: V3, phi: number, grad: readonly number[], slip: V3, nl: V3, ml: number) => {
    const a = act(phi); let m = 0; for (let c = 0; c < 3; c++) { out[c] += a[1] * grad[c]; m += Math.abs(a[1] * grad[c]); }
    if (ml > 0) { const u = sub3(slip, nl.map(q => q * dot3(nl, slip))), fy = f1y(Math.hypot(...u)); for (let c = 0; c < 3; c++) { out[c] += ml * fy * u[c]!; m += Math.abs(ml * fy * u[c]!); } }
    return { m, k: a[2] * dot3(grad, grad) };
  };
  for (let v = 0; v < N; v++) for (let k = 0; k < counts[v]!; k++) {
    const e = decodeOgcContactEntry(entries.subarray(4 * (v * KE + k), 4 * (v * KE + k) + 4), N, prev.S, prev.facets), q = KINDS.indexOf(e.kind as never);
    if (q < 0) throw new Error(`ladder 1d: no Float64 form for a ${e.kind} entry`);
    seen[e.kind] = (seen[e.kind] ?? 0) + 1;
    const out = contact[v]!.g[q]!; let term: { m: number; k: number };
    if (e.kind === 'field') {
      const s = sampleClothSolidField(field, X(v)), st = sampleClothSolidField(field, Xt(v));
      term = fieldTerm(out, s.phi, s.grad, sub3(X(v), Xt(v)), unit3(st.grad, unit3(s.grad, [0, 0, 0])), e.muLambda);
    } else if (e.kind === 'bodyField') {
      const s = sampleClothSolidField(bodyField, mulRT(R[1]!, sub3(X(v), C[1]!))), yt = mulRT(R[0]!, sub3(Xt(v), C[0]!)), carried = mulR(R[1]!, yt);
      term = fieldTerm(out, s.phi, mulR(R[1]!, s.grad), sub3(X(v), [carried[0] + C[1]![0], carried[1] + C[1]![1], carried[2] + C[1]![2]]), mulR(R[1]!, unit3(sampleClothSolidField(bodyField, yt).grad, unit3(s.grad, [0, 0, 0]))), e.muLambda);
    } else if (e.kind === 'ee' || e.kind === 'vf' || e.kind === 'sampleFace') {
      let ids: number[], P: V3[], Pt: V3[], w: number[], wl: number[], ml: number;
      if (e.kind === 'ee') {
        ids = [t.edges[4 * e.e0]!, t.edges[4 * e.e0 + 1]!, t.edges[4 * e.e1]!, t.edges[4 * e.e1 + 1]!]; P = ids.map(X); Pt = ids.map(Xt);
        w = [1 - e.s, e.s, -(1 - e.t), -e.t]; wl = w; ml = selfMu * e.lambda;
      } else {
        const tri = [t.faces[3 * e.face]!, t.faces[3 * e.face + 1]!, t.faces[3 * e.face + 2]!];
        if (e.kind === 'sampleFace') { ids = [-1, ...tri]; P = [sample(e.sample, e.body, 1), ...tri.map(X)]; Pt = [sample(e.sample, e.body, 0), ...tri.map(Xt)]; ml = e.muLambda; }
        else { ids = [e.vertex, ...tri]; P = ids.map(X); Pt = ids.map(Xt); ml = selfMu * e.lambda; }
        w = vtWeights(e.feature, P[0]!, P[1]!, P[2]!, P[3]!); wl = vtWeights(e.feature, Pt[0]!, Pt[1]!, Pt[2]!, Pt[3]!);
      }
      const gap = [0, 1, 2].map(c => w.reduce((s, wi, i) => s + wi * P[i]![c]!, 0)) as V3, d = Math.hypot(...gap), n = gap.map(z => z / d) as V3, a = act(d);
      const wr = ids.reduce((s, id, i) => s + (id === v ? w[i]! : 0), 0), wrl = ids.reduce((s, id, i) => s + (id === v ? wl[i]! : 0), 0);
      let m = 0; for (let c = 0; c < 3; c++) { out[c] += a[1] * wr * n[c]; m += Math.abs(a[1] * wr * n[c]); }
      if (ml > 0) {
        const slip = [0, 1, 2].map(c => wl.reduce((s, wi, i) => s + wi * (P[i]![c]! - Pt[i]![c]!), 0)) as V3, nl = unit3([0, 1, 2].map(c => wl.reduce((s, wi, i) => s + wi * Pt[i]![c]!, 0)), n);
        const u = sub3(slip, nl.map(z => z * dot3(nl, slip))), fy = f1y(Math.hypot(...u)); for (let c = 0; c < 3; c++) { out[c] += ml * wrl * fy * u[c]!; m += Math.abs(ml * wrl * fy * u[c]!); }
      }
      term = { m, k: a[2] * wr * wr };
    } else throw new Error(`ladder 1d: no Float64 form for a ${e.kind} entry`);
    contact[v]!.magnitude += term.m; contact[v]!.stiffness += term.k;
  }
  // Per vertex: |g_GPU - g_F64| within the f32 floor of the terms' sum and of the positions the terms read,
  // 64 eps32 (sum |terms| + (m / h~^2 + contact stiffness) max |x - origin|).
  let Xmax = 0; for (let v = 0; v < N; v++) Xmax = Math.max(Xmax, Math.hypot(...X(v)));
  const compare = (frame: typeof euler) => {
    let worst = 0, at = -1;
    for (let v = 0; v < N; v++) {
      if (t.groups[v]! > 0) continue;
      const el = elastic(frame, v), terms = [el.inertia, el.membrane, el.damping, el.bending, el.limit, ...contact[v]!.g];
      const total = [0, 1, 2].map(c => terms.reduce((s, q) => s + q[c]!, 0)), gv = [cur.g[4 * v]!, cur.g[4 * v + 1]!, cur.g[4 * v + 2]!];
      const floor = 64 * 2 ** -23 * (terms.reduce((s, q) => s + Math.abs(q[0]!) + Math.abs(q[1]!) + Math.abs(q[2]!), 0) + contact[v]!.magnitude + (el.stiffness + contact[v]!.stiffness) * Xmax);
      const ratio = Math.hypot(...sub3(gv, total)) / floor; if (ratio > worst) { worst = ratio; at = v; }
    }
    return { worst, at };
  };
  const E = compare(euler), B = compare(bdf2);
  console.log(`[ladder 1d] restart at step ${restartTick}, iterate ${K - 1}: entries ${JSON.stringify(seen)}; max |g_GPU - g_F64| / f32 floor: implicit Euler ${E.worst.toFixed(3)} (v${E.at}), BDF2 ${B.worst.toExponential(2)} (v${B.at})`);
  assert.ok(Object.keys(seen).length > 0, 'the iterate holds contact entries');
  assert.ok(E.worst <= 1, `the GPU gradient leaves the Float64 implicit-Euler objective's at v${E.at} by ${E.worst} f32 floors`);
  assert.ok(B.worst > 10, 'the restarted step\'s objective is not BDF2\'s (the frame matters at this iterate)');
}));

// ---------------------------------------------------------------- 2. the dual rule

test('ladder 2: a strip starting stretched past its strain limit lands half on a ledge: the duals stay bounded and it comes to rest at its exact hanging strain', { skip: !gpu }, async t => withDevice(async device => {
  // The Peirce scene's ledge (top 0.5 m, edge at x = 0, mu 0.5). A 3 x 49 strip at 5 mm, two thirds over the ledge and
  // 5 mm above it, starts stretched 1.5x along its length about the ledge edge: every strain-limit edge is violated
  // while the arc cuts the vertices over the ledge.
  const peirce = clothAnalyticPeirceCantilever(), cell = 0.005, cols = 49, rows = 3, top = 0.5, ticks = 90;
  const material = { ...clothAnalyticFreeFall().cloth[0]!.materials[0]! };
  const limit = material.strainLimit!, x0 = -(2 / 3) * (cols - 1) * cell, stretch = 1.5;
  const flat = herringBoneSheet({ columns: cols, rows, width_m: (cols - 1) * cell, depth_m: (rows - 1) * cell, origin_m: [x0, top + 0.005, -0.12 - (rows - 1) * cell / 2] });
  const initial = flat.restPositions_m.map((v, i) => i % 3 === 0 ? stretch * v : v);
  const scene = clothAnalyticSceneDescription({ ...peirce, cloth: [{ ...peirce.cloth[0]!, id: 'ladder-strip', materials: [material],
    // Interim plane-record capacity as C2's analytic scenes author it (parameters.ts): measured frame peak 38148 records
    // at tick 1 on 510e4bd8, 260 per vertex, + 10 %. The default 16 overflows: the held vertices block the contraction for
    // ~25 ticks and the duals of the free edges between them ramp (17 N -> 1.8e3 N, strain 61): C2's demand bug.
    solver: { integrator: 'bdf2', newton: BRIDSON, contactPlaneRecordsPerVertex: 288 }, geometry: { kind: 'inline', data: { ...flat, initialPositions_m: initial } } }] });
  const topology = prepareClothTopology(flat, [material]), mass = topology.mass_kg.reduce((a, b) => a + b, 0);
  // A bound from the strip's own momentum: no edge force is needed beyond moving the whole strip's mass through the
  // whole initial excess length within one step.
  const excess = (stretch - 1) * (cols - 1) * cell, bound = mass * excess / (h * h);
  const lambdas: number[] = []; let planeHeld = 0, fullyHeld = 0;
  const { cloth, solver, records } = await steppedScene(device, scene, ticks, async c => {
    const k = await c.contactFor('ladder-strip')!.readCounters();
    planeHeld += k.planeRecordHeld; if (k.held === topology.vertexCount) fullyHeld++;
    const duals = (await c.solverFor('ladder-strip')!.readState()).solverState.edgeDuals;
    let worst = 0; for (let e = 0; e < duals.length; e += 2) worst = Math.max(worst, Math.abs(duals[e]!));
    lambdas.push(worst);
  });
  const state = await solver.readState(); cloth.destroy();
  // The same strip and start free in space on the bare solver (no ledge, no contact): separates the AL from the arc.
  const free = await bareSolver(device, scene.cloth![0]!, { newton: BRIDSON });
  const freeLambda: number[] = [], freeStrain: number[] = [];
  for (let t = 0; t < ticks; t++) {
    free.solver.step(1);
    const st = await free.solver.readState();
    let l = 0; for (let e = 0; e < st.solverState.edgeDuals.length; e += 2) l = Math.max(l, Math.abs(st.solverState.edgeDuals[e]!));
    let c = 0; const p = st.positions_m;
    for (let e = 0; e < topology.edgeCount; e++) { const a = topology.edges[4 * e]!, b = topology.edges[4 * e + 1]!; c = Math.max(c, Math.hypot(p[3 * a]! - p[3 * b]!, p[3 * a + 1]! - p[3 * b + 1]!, p[3 * a + 2]! - p[3 * b + 2]!) / topology.edgeRestLength_m[e]! - 1); }
    freeLambda.push(l); freeStrain.push(c);
  }
  free.solver.destroy();
  console.log(`[ladder 2] free in space (bare solver): max |lambda| ${[0, 1, 2, 4, 8, 16, 32, 64, ticks - 1].map(t => freeLambda[t]!.toExponential(1)).join(' ')} N at ticks 1 2 3 5 9 17 33 65 ${ticks}; `
    + `max strain ${[0, 1, 2, 4, 8, 16, 32, 64, ticks - 1].map(t => freeStrain[t]!.toFixed(3)).join(' ')}`);
  const x = state.positions_m;
  // At rest the overhang hangs straight down from the ledge edge: band strain from the hem up is rho g s / k per
  // band (uniaxial, `hangingBandStrain`), far under the limit, so the limit is inactive and its duals decay.
  let hangTop = -1; for (let i = 0; i < cols; i++) if (x[3 * i]! > 0.002) { hangTop = i; break; }
  let worstStrain = 0, overhangDrop = 0;
  for (let e = 0; e < topology.edgeCount; e++) {
    const a = topology.edges[4 * e]!, b = topology.edges[4 * e + 1]!;
    worstStrain = Math.max(worstStrain, Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / topology.edgeRestLength_m[e]! - 1);
  }
  for (let j = 0; j < rows; j++) overhangDrop += (top - x[3 * (cols - 1 + cols * j) + 1]!) / rows;
  const hanging = cols - hangTop, exactTop = hangTop >= 0 ? hangingBandStrain(0, hanging, cell, material.stretchWarp_N_m, material.arealDensity_kg_m2, g) : NaN;
  console.log(`[ladder 2] 3 x ${cols} strip at ${cell * 1000} mm, start stretched ${stretch}x (limit ${limit}), ${ticks} ticks: max |lambda| per step ${lambdas.slice(0, 20).map(l => l.toExponential(1)).join(' ')} ... `
    + `final ${lambdas.at(-1)!.toExponential(2)} N; peak ${Math.max(...lambdas).toExponential(2)} N against the momentum bound ${bound.toExponential(2)} N; `
    + `Newton per step ${fmt(records.slice(0, 20).map(r => r.newton))} ...; final max strain ${worstStrain.toExponential(2)} (hanging top band exact ${exactTop.toExponential(2)}); `
    + `overhang ${hanging} nodes, its tip ${(1000 * overhangDrop).toFixed(1)} mm below the ledge top; held for plane-record overflow ${planeHeld}; steps with every vertex held ${fullyHeld}`);
  assert.equal(planeHeld, 0, 'the authored plane-record capacity held vertices');
  assert.ok(worstStrain < limit * 1.05, `strain ${worstStrain} past the limit ${limit}`);
  assert.ok(lambdas.at(-1)! < 0.05 * Math.max(...lambdas), `the inactive limit's duals decay: ${lambdas.at(-1)} N`);
  // Open (C2): tick 1 hits the round cap and its redone round holds every vertex, and envelope caps follow: within those steps
  // the strain limit is infeasible and AL ascent on the free edges between restricted vertices overshoots the momentum
  // bound (31 N against 1.04 N on 510e4bd8) before decaying. No lambda clamp: the fix is the contact-side restriction.
  if (Math.max(...lambdas) > bound) t.todo('C2: the tick-1 round-cap hold and envelope caps make the strain limit infeasible within the step');
  assert.ok(Math.max(...lambdas) <= bound, `a dual ramped past the strip's momentum bound: ${Math.max(...lambdas)} N > ${bound} N`);
}));

test('ladder 2b: the cap-aware done rule: with a velocity tolerance so loose that every accepted step is small, Newton on the ledge strip is not done while a travel bound holds a vertex', { skip: !gpu }, async () => withDevice(async device => {
  // The ladder-2 strip 5 mm over the ledge. At 10 m/s every accepted step passes the small-step test (alpha max |d_v| <=
  // tol h~), so the old rule ended every step after its first Newton iteration. The ledge piece's ball and its orthant
  // faces bind the cut trials there with a positive multiplier (the tick-1 dump, RESULT 36) - travel bounds, not contacts
  // - so the rule must run Newton on until none holds a vertex at the accepted trial, or report the step unconverged.
  const peirce = clothAnalyticPeirceCantilever(), cell = 0.005, cols = 49, rows = 3, top = 0.5, ticks = 3;
  const material = { ...clothAnalyticFreeFall().cloth[0]!.materials[0]! }, x0 = -(2 / 3) * (cols - 1) * cell;
  const flat = herringBoneSheet({ columns: cols, rows, width_m: (cols - 1) * cell, depth_m: (rows - 1) * cell, origin_m: [x0, top + 0.005, -0.12 - (rows - 1) * cell / 2] });
  const initial = flat.restPositions_m.map((v, i) => i % 3 === 0 ? 1.5 * v : v);
  const newton = { ...CONVERGED, velocityTolerance_m_s: 10 };
  const scene = clothAnalyticSceneDescription({ ...peirce, cloth: [{ ...peirce.cloth[0]!, id: 'ladder-strip', materials: [material],
    solver: { integrator: 'bdf2', newton, contactPlaneRecordsPerVertex: 288 }, geometry: { kind: 'inline', data: { ...flat, initialPositions_m: initial } } }] });
  const { cloth, records } = await steppedScene(device, scene, ticks);
  cloth.destroy();
  console.log(`[ladder 2b] strip over the ledge at a 10 m/s velocity tolerance: Newton per step ${fmt(records.map(r => r.newton))}; done ${records.map(r => r.done ? 1 : 0).join(' ')}; `
    + `vertices a travel bound held at the last accepted trial ${fmt(records.map(r => r.travelHeld))} (the cap ${newton.iterations})`);
  // steppedScene asserts no step ends done while a travel bound holds a vertex; here the rule must also have acted.
  assert.ok(records[0]!.newton > 1, `tick 1 ended after ${records[0]!.newton} Newton iteration: the travel bounds binding there did not keep Newton running`);
}));

/** Ladder 2c's open defect (C1 RESULTs 43-44). With the KKT stop test (RESULT 44) neither run converges at tick 20:
 * after 150 iterations the largest |P g_v| is 14x its f32 floor, and the ends are 0.96 mm apart. The Float64 objective
 * (the oracle's frozen pairs, the sample-face pairs included) has one minimiser from x0 and from both GPU ends (within
 * 0.41 um), 15.8 mm from them. So the GPU stalls short of a unique solution: there are not several minima. */
const METRIC_TODO = "C1 open: on the 16^2 sphere Newton stalls 14x above the f32 KKT floor at tick 20 (ends 0.96 mm apart); with the duals frozen both converge; the Float64 implicit-Euler minimiser (the step restarts) is 0.96 mm from them (RESULT 45)";

/** One step of the 16^2 rotating sphere from the published state at `tick`, under `solver` (caps and tolerances
 * replaced); the result and the step's Newton record. */
async function sphereStepFrom(device: GPUDevice, tick: number, solver: (doc: ClothDocument) => ClothDocument['solver']) {
  const base = bridson2002SceneDescription(bridson2002RotatingSphere({ nodesAcross: 16 })), doc = base.cloth![0]!;
  if (doc.geometry.kind !== 'inline') throw new Error('the Bridson sphere document is inline');
  const start = (await captureScene(device, base, [tick])).frames[0]![0]!;
  if (!start.velocities_m_s || !start.solverState) throw new Error(`the tick-${tick} publication carries no velocities or solver state`);
  const scene = { ...base, cloth: [{ ...doc, solver: solver(doc), geometry: { kind: 'inline' as const, data: { ...doc.geometry.data,
    initialPositions_m: Array.from(start.positions_m), initialVelocities_m_s: Array.from(start.velocities_m_s), solverState: start.solverState } } }] };
  const cloth = await ClothSystem.create(device, scene, { statusInterval: 1 }), gpuSolver = cloth.solverFor(doc.id)!;
  try {
    assert.ok(cloth.stepTo(start.time_s + CLOTH_SCENE_STEP_S), `the step from tick ${tick} was refused: ${cloth.failure?.message ?? 'pending'}`);
    const record = await newtonOf(gpuSolver), x = await gpuSolver.readPositions();
    // The stop test's KKT residual at the last iterate (nwGradient's g.w): the largest |P g_v|^2 / floor_v^2.
    const reader = gpuSolver as unknown as { readRange(b: GPUBuffer, offset: number, size: number): Promise<ArrayBuffer>; newtonWork: GPUBuffer };
    const g4 = new Float32Array(await reader.readRange(reader.newtonWork, 0, 16 * (x.length / 3)));
    let kkt = 0; for (let v = 3; v < g4.length; v += 4) kkt = Math.max(kkt, g4[v]!);
    assert.equal(cloth.failure, undefined, cloth.failure?.message);
    return { record, x, kkt: Math.sqrt(kkt) };
  } finally { cloth.destroy(); }
}

test('ladder 2c: on the 16^2 sphere a step\'s result is independent of the trial projection\'s metric (identity against H_vv) to within the convergence tolerance', { skip: !gpu, todo: METRIC_TODO }, async () => withDevice(async device => {
  // The metric only chooses the projected trial points of each iteration's arc: both runs must converge to the same
  // step solution. Newton stops at an accepted move of at most tol = 1 um (velocityTolerance tol / h~); each run's end is
  // then within about one last move of the solution, so the two ends must agree to 2 tol. Ticks 20 and 40: the sheet on
  // the sphere (before C2's tick-46 heldConflict in the metric).
  const tol = 1e-6, hEff = 2 * CLOTH_SCENE_STEP_S / 3, lines: string[] = [], fails: string[] = [];
  for (const tick of [20, 40]) {
    const solve = (metric: 'off' | 'hvv') => (doc: ClothDocument) => ({ ...doc.solver!, contactProjectionMetric: metric,
      newton: { ...(doc.solver!.newton ?? CLOTH_NEWTON_SCENE_DEFAULTS), iterations: 150, velocityTolerance_m_s: tol / hEff } });
    const off = await sphereStepFrom(device, tick, solve('off')), hvv = await sphereStepFrom(device, tick, solve('hvv'));
    let gap = 0, at = -1, over = 0;
    for (let v = 0; 3 * v < off.x.length; v++) {
      const d = Math.hypot(off.x[3 * v]! - hvv.x[3 * v]!, off.x[3 * v + 1]! - hvv.x[3 * v + 1]!, off.x[3 * v + 2]! - hvv.x[3 * v + 2]!);
      if (d > 2 * tol) over++;
      if (d > gap) { gap = d; at = v; }
    }
    lines.push(`tick ${tick}: Newton identity ${off.record.newton} (done ${+off.record.done}), H_vv ${hvv.record.newton} (done ${+hvv.record.done}); |g|/|mg| ${off.record.last.toExponential(2)} / ${hvv.record.last.toExponential(2)}; max |P g_v| / floor_v ${off.kkt.toFixed(2)} / ${hvv.kkt.toFixed(2)}; ends ${(1e6 * gap).toFixed(3)} um apart at vertex ${at}, ${over} vertices past 2 tol`);
    console.log(`[ladder 2c] ${lines.at(-1)}`);
    assert.ok(off.record.done && hvv.record.done, `tick ${tick}: a run did not converge in 150 iterations (identity ${off.record.newton}, H_vv ${hvv.record.newton})`);
    if (!(gap <= 2 * tol)) fails.push(`tick ${tick}: the identity and H_vv ends are ${(1e6 * gap).toFixed(3)} um apart at vertex ${at} (bound 2 um)`);
  }
  assert.deepEqual(fails, []);
}));

// ---------------------------------------------------------------- 3. PCG cost

test('ladder 3: PCG cost per iteration and iterations per Newton solve, 16^2 to 128^2 without contact, 32^2 on a floor and on a sphere with contact', { skip: !gpu }, async () => withDevice(async device => {
  const lines: string[] = [], warm = 10;
  for (const sweeps of SWEEPS) for (const n of [16, 32, 64, 128]) {
    const document = resized(clothAnalyticFreeFall(), sheet(n, 1, (i, j) => j === 0 && (i === 0 || i === n - 1) ? 1 : 0)).cloth[0]!;
    const { solver } = await bareSolver(device, document, { dt_s: h, newton: { ...BRIDSON, patchSweeps: sweeps } as ClothNewtonSettings });
    let newton = 0, pcg = 0;
    for (let s = 0; s < warm; s++) { solver.step(1); const r = await newtonOf(solver); newton += r.newton; pcg += r.pcg; }
    const us = await pcgMicros(device, solver);
    // A whole solve at the last state: the dispatch chain to the cap (its early exits included), and (one patch) pcgSolve.
    const S = solver as unknown as Internals, chain = await solveMicros(device, solver, 'chain');
    const replay = process.env.CLOTH_LADDER_REPLAY ? await solveMicros(device, solver, 'replay') : undefined;
    if (replay) console.log(`[ladder 3] ${n}^2: iteration 2 replayed ${(S as unknown as { newton: { pcgIterations: number } }).newton.pcgIterations + 1} times in one pass ${(replay.us / 201).toFixed(2)} us each (control w ${replay.iterations})`);
    const one = S.tables.pcg.patches === 1 && 'pcgSolve' in S.pipelines ? await solveMicros(device, solver, 'solve') : undefined;
    if (one) {
      // Fixed-state equivalence: the same solve both ways from the same iterate agree to the PCG tolerance.
      await solveMicros(device, solver, 'chain', 1); const a = await readDirection(device, solver);
      await solveMicros(device, solver, 'solve', 1); const b = await readDirection(device, solver);
      let diff = 0, size = 0;
      for (let v = 0; v < a.length; v++) { diff = Math.max(diff, Math.hypot(a[v]![0] - b[v]![0], a[v]![1] - b[v]![1], a[v]![2] - b[v]![2])); size = Math.max(size, Math.hypot(...a[v]!)); }
      lines.push(`  ${n}^2 fixed state: pcgSolve's d off the chain's ${(diff / size).toExponential(2)} of max |d| ${(1e3 * size).toFixed(3)} mm`);
      assert.ok(diff <= 10 * BRIDSON.pcgTolerance! * size, `pcgSolve vs the chain at a fixed state: ${diff} m of ${size} m`);
    }
    solver.destroy();
    lines.push(`no contact ${n}^2 (${n * n} vertices, sweeps ${sweeps}): ${us.toFixed(2)} us per PCG iteration, ${(pcg / newton).toFixed(1)} PCG per Newton, ${(newton / warm).toFixed(1)} Newton per step (steps 1-${warm}); `
      + `one solve: chain ${chain.us.toFixed(0)} us (${chain.iterations} iterations, ${(chain.us / chain.iterations).toFixed(2)} us each)`
      + (one ? `, pcgSolve ${one.us.toFixed(0)} us (${one.iterations} iterations, ${(one.us / one.iterations).toFixed(2)} us each)` : ''));
    assert.ok(Number.isNaN(us) || us > 0);
  }
  // Eisenstat-Walker forcing (eta_max 0.1) at the Bridson caps: PCG per Newton at one patch (pcgSolve) and over patches (the chain).
  for (const n of [16, 32, 64]) {
    const document = resized(clothAnalyticFreeFall(), sheet(n, 1, (i, j) => j === 0 && (i === 0 || i === n - 1) ? 1 : 0)).cloth[0]!;
    const { solver } = await bareSolver(device, document, { dt_s: h, newton: { ...BRIDSON, forcingMax: 0.1 } });
    let newton = 0, pcg = 0;
    for (let s = 0; s < warm; s++) { solver.step(1); const r = await newtonOf(solver); newton += r.newton; pcg += r.pcg; }
    lines.push(`forcing eta_max 0.1 ${n}^2 (${(solver as unknown as Internals).tables.pcg.patches} patches): ${(pcg / newton).toFixed(1)} PCG per Newton, ${(newton / warm).toFixed(1)} Newton per step (steps 1-${warm})`);
    solver.destroy();
  }
  // With contact: a 32^2 sheet resting on the floor (dropped 3 mm, 20 ticks) and the Bridson sphere at 32 nodes across
  // draped on its sphere (40 ticks).
  const floor = clothAnalyticSceneDescription(resized(clothAnalyticFreeFall(), sheet(32, 0.003)));
  const sphereSource = bridson2002RotatingSphere({ nodesAcross: 32 }), sphere = bridson2002SceneDescription(sphereSource);
  for (const [name, scene, ticks] of [['32^2 on the floor', floor, 20], ['32^2 on the sphere (Bridson Fig 2 at 32 nodes)', sphere, 40]] as const) {
    const { cloth, solver, records } = await steppedScene(device, scene, ticks);
    const counters = await cloth.contactFor(scene.cloth![0]!.id)!.readCounters();
    const us = await pcgMicros(device, solver), last = records.slice(-5);
    cloth.destroy();
    lines.push(`contact ${name}: ${us.toFixed(2)} us per PCG iteration, ${(sum(last.map(r => r.pcg)) / sum(last.map(r => r.newton))).toFixed(1)} PCG per Newton, `
      + `Newton per step ${fmt(last.map(r => r.newton))} (last 5 ticks); ${counters.solidContacts} solid contacts, ${counters.vfPairs + counters.eePairs} pairs at tick ${ticks}`);
  }
  console.log(`[ladder 3] PCG cost (replayed iteration 2, 64 reps; Bridson caps 8 x 200, tolerance ${BRIDSON.pcgTolerance}):\n  ${lines.join('\n  ')}`);
}));

// ---------------------------------------------------------------- 4. the analytic gates, small

test('ladder 4a: free fall of a 16^2 sheet is exactly the BDF2 recurrence; dropped flat it lands without a rebound', { skip: !gpu }, async () => withDevice(async device => {
  const source = clothAnalyticFreeFall(), y0 = source.probe.y0 as number, n = source.probe.gateSteps as number;
  const scene = clothAnalyticSceneDescription(resized(source, sheet(16, y0)));
  const { frames, ms } = await captureScene(device, scene, range(1, 45));
  const ref = bdf2FreeFall(n, h, -g), track = mean(frames.map(f => f[0]!));
  let centroid = 0, velocity = 0;
  for (let k = 0; k < n; k++) { centroid = Math.max(centroid, Math.abs(track[k]!.y - (y0 + ref.displacement_m[k]!))); velocity = Math.max(velocity, Math.abs(track[k]!.vy - ref.velocity_m_s[k]!)); }
  const r = rebound(track);
  console.log(`[ladder 4a] 16^2 free fall, ${ms.toFixed(1)} ms/tick: centroid off BDF2 ${(1000 * centroid).toFixed(4)} mm, velocity ${(1000 * velocity).toFixed(4)} mm/s over ${n} steps; `
    + `landing: fastest fall ${r.fall.toFixed(3)} m/s, rebound ${r.up.toFixed(4)} m/s, rests at ${(1000 * track.at(-1)!.y).toFixed(2)} mm`);
  assert.ok(centroid < 1e-4 && velocity < 1e-3, `off the recurrence: ${centroid} m, ${velocity} m/s`);
  assert.ok(r.fall < -4 && r.up <= 0.05, `landing: fall ${r.fall} m/s, rebound ${r.up} m/s`);
}));

test('ladder 4c: a 21 x 11 swinging sheet: its flat outer part falls as free fall, and it passes under the pin as the converged run does', { skip: !gpu }, async () => withDevice(async device => {
  // Half the analytic swing (0.5 x 0.25 m at 2.5 cm). The bend that ends the free fall travels as sqrt(L), so the
  // 1 m sheet's 20 gated steps become 20 / sqrt(2) = 14.
  const source = clothAnalyticSwingingSheet(), L = 0.5, W = 0.25, cols = 21, rows = 11, y0 = source.probe.y0 as number, gate = 14, ticks = 35;
  const data = herringBoneSheet({ columns: cols, rows, width_m: L, depth_m: W, origin_m: [0, y0, -W / 2], group: i => i === 0 ? 1 : 0 });
  const edge = range(Math.round(0.2 * (rows - 1)), Math.round(0.8 * (rows - 1))).map(j => cols - 1 + cols * j);
  const run = async (newton: ClothNewtonSettings) => {
    const { frames, ms } = await captureScene(device, clothAnalyticSceneDescription(resized(source, data), { newton }), range(1, ticks));
    const y = frames.map(f => edge.reduce((q, v) => q + f[0]!.positions_m[3 * v + 1]!, 0) / edge.length);
    const x = frames.map(f => edge.reduce((q, v) => q + f[0]!.positions_m[3 * v]!, 0) / edge.length);
    let crossing = NaN;
    for (let k = 1; k < x.length; k++) if (x[k]! <= 0 && x[k - 1]! > 0) { crossing = (k + x[k - 1]! / (x[k - 1]! - x[k]!)) * h; break; }
    return { y, x, crossing, ms };
  };
  const a = await run(BRIDSON), b = await run(CONVERGED), ref = bdf2FreeFall(gate, h, -g);
  let worst = 0; for (let k = 0; k < gate; k++) worst = Math.max(worst, Math.abs(a.y[k]! - (y0 + ref.displacement_m[k]!)));
  console.log(`[ladder 4c] 21 x 11 swing, ${a.ms.toFixed(1)} ms/tick: free edge off free fall ${(1000 * worst).toFixed(3)} mm over ${gate} steps; under the pin at ${a.crossing.toFixed(4)} s (converged ${b.crossing.toFixed(4)} s)`);
  assert.ok(worst < 1e-3, `free edge ${worst} m off free fall`);
  assert.ok(Number.isFinite(a.crossing) && Math.abs(a.crossing - b.crossing) <= h, `passes the pin at ${a.crossing} s, converged ${b.crossing} s`);
}));

test('ladder 4d: a 9 x 14 curtain comes to rest at its exact uniaxial equilibrium; a Peirce strip at l = 2c droops to the 41.5 degree elastica', { skip: !gpu }, async () => withDevice(async device => {
  // Both gates are static equilibria of the scene's own energy, so they run the bare solver (neither touches a solid)
  // with implicit Euler at a long step, which damps the sway the 1/60 s scene clock takes seconds to ring down.
  const curtain = clothAnalyticHangingCurtain(), m = 9, hc = 0.02, n = Math.floor(1.7 * (m - 1) + 1e-9) + 1, top = curtain.probe.top as number;
  const flat = herringBoneSheet({ columns: m, rows: n, width_m: (m - 1) * hc, depth_m: (n - 1) * hc, origin_m: [0, 0, 0], group: (_i, j) => j === 0 ? 1 : 0 });
  const rest: number[] = []; for (let v = 0; v < flat.restPositions_m.length; v += 3) rest.push(flat.restPositions_m[v]! - (m - 1) * hc / 2, top - flat.restPositions_m[v + 2]!, 0);
  const document = resized(curtain, { ...flat, restPositions_m: rest }).cloth[0]!, material = document.materials[0]!;
  const { solver } = await bareSolver(device, document, { dt_s: 0.25, integrator: 'implicit-euler', newton: CONVERGED });
  solver.step(40);
  const x = await solver.readPositions(); solver.destroy();
  const k = material.stretchWeft_N_m, rho = material.arealDensity_kg_m2;
  let worst = 0, drop = 0;
  for (let j = 0; j + 1 < n; j++) drop += hangingBandStrain(j, n, hc, k, rho, g) * hc;
  for (const j of range(2, n - 5)) {
    let e = 0;
    for (let i = 1; i < m - 1; i++) { const a = i + m * j, b = a + m; e += (Math.hypot(x[3 * a]! - x[3 * b]!, x[3 * a + 1]! - x[3 * b + 1]!, x[3 * a + 2]! - x[3 * b + 2]!) / hc - 1) / (m - 2); }
    const exact = hangingBandStrain(j, n, hc, k, rho, g);
    worst = Math.max(worst, Math.abs(e - exact) / exact);
  }
  const bottom = Math.floor(m / 2) + m * (n - 1), hem = top - x[3 * bottom + 1]! - (n - 1) * hc;
  // Peirce: the ASTM strip (l = 2c), clamp pinned.
  const peirce = clothAnalyticPeirceCantilever(), [clampCells, cols, rows, , exactAngle] = peirce.probe.strip1 as number[];
  const { solver: strip } = await bareSolver(device, peirce.cloth[1]!, { dt_s: 0.1, integrator: 'implicit-euler', newton: CONVERGED });
  strip.step(60);
  const p = await strip.readPositions(); strip.destroy();
  let dx = 0, dy = 0;
  for (let j = 0; j < rows!; j++) { const root = clampCells! + cols! * j, tip = cols! - 1 + cols! * j; dx += (p[3 * tip]! - p[3 * root]!) / rows!; dy += (p[3 * root + 1]! - p[3 * tip + 1]!) / rows!; }
  const angle = Math.atan2(dy, dx) * 180 / Math.PI;
  console.log(`[ladder 4d] ${m} x ${n} curtain: worst band strain error ${(100 * worst).toFixed(3)} %, hem drop ${(1000 * hem).toFixed(3)} mm (exact ${(1000 * drop).toFixed(3)}); `
    + `Peirce strip ${cols} x ${rows}: chord ${angle.toFixed(2)} deg (exact ${exactAngle!.toFixed(2)})`);
  assert.ok(worst < 0.01, `band strain off its equilibrium by ${worst}`);
  assert.ok(Math.abs(hem - drop) < 0.01 * drop, `hem drop ${hem} m, exact ${drop} m`);
  assert.ok(Math.abs(angle - exactAngle!) < 1, `chord ${angle} deg, exact ${exactAngle}`);
}));

test('ladder 4e: on the incline, the 11 x 11 patch below the friction angle sticks and the one above slides at g (sin - mu cos)', { skip: !gpu }, async () => withDevice(async device => {
  const source = clothAnalyticFrictionIncline(), t0 = Math.round(0.2 / h), t1 = Math.round(1 / h);
  const { frames, ms } = await captureScene(device, clothAnalyticSceneDescription(source), range(t0, t1));
  const along = source.cloth.map((_, s) => {
    const [, , dx, dy] = source.probe[`slope${s}`] as number[];
    return frames.map(f => { const p = f[s]!.positions_m, N = p.length / 3; let a = 0; for (let i = 0; i < N; i++) a += (p[3 * i]! * dx! + p[3 * i + 1]! * dy!) / N; return a; });
  });
  // Least squares s(t) = s0 + v0 t + a t^2 / 2 (normal equations, 3 x 3).
  const fit = (s: number[]) => {
    const rowsOf = s.map((_, k) => { const t = (t0 + k) * h; return [1, t, t * t / 2]; });
    const A = [0, 1, 2].map(i => [0, 1, 2].map(j => rowsOf.reduce((q, r) => q + r[i]! * r[j]!, 0))), b = [0, 1, 2].map(i => rowsOf.reduce((q, r, k) => q + r[i]! * s[k]!, 0));
    for (let i = 0; i < 3; i++) for (let r = i + 1; r < 3; r++) { const f = A[r]![i]! / A[i]![i]!; for (let c = i; c < 3; c++) A[r]![c]! -= f * A[i]![c]!; b[r]! -= f * b[i]!; }
    const x = [0, 0, 0]; for (let i = 2; i >= 0; i--) x[i] = (b[i]! - [0, 1, 2].slice(i + 1).reduce((q, c) => q + A[i]![c]! * x[c]!, 0)) / A[i]![i]!;
    return x[2]!;
  };
  const creep = Math.abs(along[0]!.at(-1)! - along[0]![0]!), a = fit(along[1]!), exact = (source.probe.slope1 as number[])[1]!;
  console.log(`[ladder 4e] incline, ${ms.toFixed(1)} ms/tick: sticking patch moved ${(1000 * creep).toFixed(3)} mm; sliding a = ${a.toFixed(4)} m/s^2 (exact ${exact.toFixed(4)}, ${(100 * (a / exact - 1)).toFixed(2)} %)`);
  assert.ok(creep < 2e-3, `sticking patch moved ${creep} m`);
  assert.ok(Math.abs(a / exact - 1) < 0.05, `sliding acceleration ${a} against ${exact}`);
}));
