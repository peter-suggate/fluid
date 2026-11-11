/**
 * Cloth benchmark: the Bridson 2002 rotating sphere (Fig 2) through the scene's ClothSystem, solver and contact
 * together, from fixed saved states. Each start state (default ticks 2, 5, 35, 120) is restored as a history endpoint
 * (the exact save/load path), then stepped:
 *   timed    `--steps` steps, one per submission, fenced; the GPU span of each step's pass (timestamps);
 *   profile  `--profile` steps from a fresh restore with every dispatch in its own timestamped pass, attributed by
 *            pipeline label ("Cloth <entry>" is the solver, "ogc <kernel>" the detector). Splitting adds a pass
 *            boundary per dispatch, so the profile's sum exceeds the timed span: it attributes, the timed run measures.
 * Per step it reports the Newton iterations used, the PCG iterations that did work (pcgStep dispatches that ran, less
 * the one start dispatch of each Newton iteration), dispatches queued and those that did work, and the detector's frame counters.
 *
 * States are built once by running the scene from tick 0 and cached as JSON in
 * ~/code/me/fluid-scratch/cloth-bench/ (sphere-<nodes>-tick<k>.json; `--rebuild` replaces them).
 *
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js tools/gpu/fluid-gpu-run.sh timing \
 *     node --import tsx tools/benchmark-cloth-sphere.ts [--nodes 150] [--ticks 2,5,35,120] [--steps 10] [--profile 1] [--json out.json]
 *       [--locality off|topology+geometry]   the detector's locality switch (OgcContactConfig.locality)
 *       [--lin off|factorized|blocks] [--lin-records n]  newton.contactLinearization on the restored document (default:
 *         the document's own, off) and its pool (vec4 records per vertex); the run then also reports the compiled
 *         records' demand and replays the two build kernels (as a pair: the build reads the count's headers)
 *       [--newton-iterations n]   the restored document's Newton cap (fixed-state comparisons: 1 or 2 iterations from one state)
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createProcessRetainedDawnGPU } from '../lib/harness/node-dawn-provider';
import { requiredFluidDeviceLimits } from '../lib/core/webgpu-device-limits';
import { cloneScene, type SceneDescription } from '../lib/core/model';
import { publishSceneClothSurfaces, snapshotSceneClothPose } from '../lib/cloth/scene-geometry';
import { CLOTH_SCENE_STEP_S, ClothSystem } from '../lib/cloth/system';
import { bridson2002RotatingSphere } from '../lib/cloth/scenes/bridson-2002';
import { bridson2002SceneDescription } from '../lib/cloth/scenes/bridson-2002/scene-description';
import type { ClothDocument } from '../lib/cloth/types';
import type { ClothGPUSolver } from '../lib/cloth/gpu-solver';
import { CLOTH_SEGMENT_ALIGN } from '../lib/cloth/gpu-solver-wgsl';

const args = process.argv.slice(2);
const opt = (name: string, fallback: string) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1]! : fallback; };
const NODES = Number(opt('nodes', '150')), TICKS = opt('ticks', '2,5,35,120').split(',').map(Number);
const STEPS = Number(opt('steps', '10')), PROFILE_STEPS = Number(opt('profile', '1')), JSON_OUT = opt('json', '');
const REBUILD = args.includes('--rebuild');
/** The detector's locality switch for the timed, profiled and replayed systems (states are always built with it off). */
const LOCALITY = opt('locality', 'off');
const SYSTEM = { statusInterval: 1, ...(LOCALITY !== 'off' ? { contactLocality: LOCALITY } : {}) };
const CACHE = opt('cache', join(homedir(), 'code/me/fluid-scratch/cloth-bench'));
/** A dispatch at or under this GPU time is an early exit (measured: ~1-2 us for an exiting per-vertex dispatch). */
const REPLAY = Number(opt('replay', '64'));
/** newton.contactLinearization for the run ('' keeps the document's), and its pool. */
const LIN = opt('lin', ''), LIN_RECORDS = opt('lin-records', ''), NEWTON_ITERATIONS = opt('newton-iterations', '');

const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href);
Object.assign(globalThis, dawn.globals);
const adapter = await createProcessRetainedDawnGPU(dawn, [`backend=${process.env.FLUID_WEBGPU_BACKEND ?? 'metal'}`]).requestAdapter();
if (!adapter) throw new Error('no WebGPU adapter');
if (!adapter.features.has('timestamp-query')) throw new Error('the benchmark needs timestamp-query');
const device: GPUDevice = await adapter.requestDevice({
  requiredFeatures: ['timestamp-query', ...(adapter.features.has('subgroups') ? ['subgroups' as GPUFeatureName] : [])],
  requiredLimits: requiredFluidDeviceLimits(adapter.limits),
});
const errors: string[] = [];
device.addEventListener('uncapturederror', (e: Event) => { e.preventDefault(); const m = (e as GPUUncapturedErrorEvent).error.message; if (!errors.length) console.error(m); errors.push(m); });

// ------------------------------------------------------------------ timestamps

const QUERIES = 4096;
interface Stamp { label: string; set: number; slot: number; workgroups?: number; indirect?: number; control?: number }
/** Wraps device.createCommandEncoder: in 'span' mode each "Cloth step" pass is timestamped; in 'split' mode every
 * dispatch of every compute pass runs in its own timestamped pass (bindings re-set), indirect counts logged. */
class Timestamps {
  mode: 'off' | 'span' | 'split' = 'off';
  records: Stamp[] = [];
  private sets: GPUQuerySet[] = [];
  private next = 0;
  private indirectLog: GPUBuffer;
  private indirectCount = 0;
  private controlLog: GPUBuffer;
  private controlCount = 0;
  /** The solver's Newton control record (done, PCG done, step, iteration), logged before every split dispatch. */
  control?: { buffer: GPUBuffer; offset: number };
  constructor(private readonly device: GPUDevice) {
    this.indirectLog = device.createBuffer({ label: 'bench indirect log', size: 12 * 65536, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    this.controlLog = device.createBuffer({ label: 'bench control log', size: 16 * 65536, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const create = device.createCommandEncoder.bind(device);
    (device as { createCommandEncoder: GPUDevice['createCommandEncoder'] }).createCommandEncoder = (d?: GPUCommandEncoderDescriptor) => {
      const e = create(d);
      return this.mode === 'off' ? e : this.wrap(e);
    };
  }
  private pair(): { set: number; slot: number; writes: GPUComputePassTimestampWrites } {
    if (this.next >= this.sets.length * QUERIES) this.sets.push(this.device.createQuerySet({ type: 'timestamp', count: QUERIES }));
    const set = Math.floor(this.next / QUERIES), slot = this.next % QUERIES;
    this.next += 2;
    return { set, slot, writes: { querySet: this.sets[set]!, beginningOfPassWriteIndex: slot, endOfPassWriteIndex: slot + 1 } };
  }
  private wrap(encoder: GPUCommandEncoder): GPUCommandEncoder {
    const self = this;
    return new Proxy(encoder, {
      get(target, prop) {
        if (prop === 'beginComputePass') return (d?: GPUComputePassDescriptor) => self.mode === 'span'
          ? self.span(target, d) : self.split(target, d);
        const v = Reflect.get(target, prop);
        return typeof v === 'function' ? v.bind(target) : v;
      },
    });
  }
  private span(encoder: GPUCommandEncoder, d?: GPUComputePassDescriptor): GPUComputePassEncoder {
    if (d?.label !== 'Cloth step') return encoder.beginComputePass(d);
    const p = this.pair();
    this.records.push({ label: 'step span', set: p.set, slot: p.slot });
    return encoder.beginComputePass({ ...d, timestampWrites: p.writes });
  }
  private split(encoder: GPUCommandEncoder, d?: GPUComputePassDescriptor): GPUComputePassEncoder {
    let pipeline: GPUComputePipeline | undefined;
    const groups = new Map<number, [GPUBindGroup, number[]]>();
    const run = (dispatch: (pass: GPUComputePassEncoder) => void, extra: Partial<Stamp>) => {
      let control: number | undefined;
      if (this.control && this.controlCount < 65536) {
        control = this.controlCount++;
        encoder.copyBufferToBuffer(this.control.buffer, this.control.offset, this.controlLog, 16 * control, 16);
      }
      const p = this.pair();
      this.records.push({ label: pipeline?.label ?? '?', set: p.set, slot: p.slot, control, ...extra });
      const pass = encoder.beginComputePass({ label: d?.label, timestampWrites: p.writes });
      pass.setPipeline(pipeline!);
      for (const [i, [g, offsets]] of groups) pass.setBindGroup(i, g, offsets);
      dispatch(pass);
      pass.end();
    };
    return {
      label: d?.label ?? '',
      setPipeline: (p: GPUComputePipeline) => { pipeline = p; },
      setBindGroup: (i: number, g: GPUBindGroup | null, offsets?: Iterable<number>, ...rest: unknown[]) => {
        if (rest.length) throw new Error('bench: the (data, start, length) form of dynamic offsets is not split');
        if (g) groups.set(i, [g, offsets ? Array.from(offsets) : []]); else groups.delete(i);
      },
      dispatchWorkgroups: (x: number, y = 1, z = 1) => run(pass => pass.dispatchWorkgroups(x, y, z), { workgroups: x * y * z }),
      dispatchWorkgroupsIndirect: (buffer: GPUBuffer, offset: number) => {
        if (this.indirectCount >= 65536) throw new Error('bench: indirect log full');
        const k = this.indirectCount++;
        encoder.copyBufferToBuffer(buffer, offset, this.indirectLog, 12 * k, 12);
        run(pass => pass.dispatchWorkgroupsIndirect(buffer, offset), { indirect: k });
      },
      pushDebugGroup: () => undefined, popDebugGroup: () => undefined, insertDebugMarker: () => undefined,
      end: () => undefined,
    } as unknown as GPUComputePassEncoder;
  }
  /** Resolve and read every timestamp and indirect count recorded since the last reset; returns ns per record. */
  async collect(): Promise<{ record: Stamp; ns: number; workgroups: number; control?: Float32Array }[]> {
    const encoder = this.device.createCommandEncoder({ label: 'bench resolve' });
    const reads: GPUBuffer[] = [];
    this.sets.forEach((set, s) => {
      const used = Math.min(QUERIES, this.next - s * QUERIES);
      if (used <= 0) return;
      const resolve = this.device.createBuffer({ size: 8 * QUERIES, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
      const read = this.device.createBuffer({ size: 8 * QUERIES, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      encoder.resolveQuerySet(set, 0, used, resolve, 0);
      encoder.copyBufferToBuffer(resolve, 0, read, 0, 8 * used);
      reads[s] = read;
    });
    const log = this.device.createBuffer({ size: Math.max(16, 12 * this.indirectCount + 4), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (this.indirectCount) encoder.copyBufferToBuffer(this.indirectLog, 0, log, 0, 12 * this.indirectCount);
    const ctl = this.device.createBuffer({ size: Math.max(16, 16 * this.controlCount), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    if (this.controlCount) encoder.copyBufferToBuffer(this.controlLog, 0, ctl, 0, 16 * this.controlCount);
    this.device.queue.submit([encoder.finish()]);
    const stamps = await Promise.all(reads.map(async r => { await r.mapAsync(GPUMapMode.READ); const a = new BigUint64Array(r.getMappedRange().slice(0)); r.destroy(); return a; }));
    await log.mapAsync(GPUMapMode.READ);
    const counts = new Uint32Array(log.getMappedRange().slice(0)); log.destroy();
    await ctl.mapAsync(GPUMapMode.READ);
    const controls = new Float32Array(ctl.getMappedRange().slice(0)); ctl.destroy();
    const out = this.records.map(record => {
      const t = stamps[record.set]!, ns = Number(t[record.slot + 1]! - t[record.slot]!);
      const workgroups = record.indirect !== undefined ? counts[3 * record.indirect]! * counts[3 * record.indirect + 1]! * counts[3 * record.indirect + 2]! : record.workgroups ?? 0;
      return { record, ns, workgroups, ...(record.control !== undefined ? { control: controls.slice(4 * record.control, 4 * record.control + 4) } : {}) };
    });
    this.records = []; this.next = 0; this.indirectCount = 0; this.controlCount = 0;
    return out;
  }
}
const stamps = new Timestamps(device);

// ------------------------------------------------------------------ states

const source = bridson2002RotatingSphere({ nodesAcross: NODES });
const baseScene = bridson2002SceneDescription(source);
const documentId = baseScene.cloth![0]!.id;
const statePath = (tick: number) => join(CACHE, `sphere-${NODES}-tick${tick}.json`);
const plain = (_k: string, v: unknown) => ArrayBuffer.isView(v) && !(v instanceof DataView) ? Array.from(v as unknown as ArrayLike<number>) : v;

async function buildStates(): Promise<void> {
  const missing = TICKS.filter(t => REBUILD || !existsSync(statePath(t))).sort((a, b) => a - b);
  if (!missing.length) return;
  mkdirSync(CACHE, { recursive: true });
  const scene = cloneScene(baseScene), cloth = await ClothSystem.create(device, scene, { statusInterval: 1 });
  const t0 = performance.now();
  for (const tick of missing) {
    await cloth.capture([tick * CLOTH_SCENE_STEP_S]);
    publishSceneClothSurfaces(scene, cloth.surfaces);
    const document = snapshotSceneClothPose(scene).cloth!.find(d => d.id === documentId)!;
    writeFileSync(statePath(tick), JSON.stringify({ nodes: NODES, tick, document }, plain));
    console.log(`state tick ${tick} saved (${((performance.now() - t0) / 1000).toFixed(1)} s)`);
  }
  cloth.destroy();
}

function restored(tick: number): SceneDescription {
  const saved = JSON.parse(readFileSync(statePath(tick), 'utf8')) as { nodes: number; tick: number; document: ClothDocument };
  if (saved.nodes !== NODES || saved.tick !== tick) throw new Error(`${statePath(tick)} is not the ${NODES}-node state at tick ${tick}`);
  const scene = cloneScene(baseScene);
  const d = saved.document, newton = d.solver?.newton ?? baseScene.cloth![0]!.solver!.newton!;
  scene.cloth = [LIN || LIN_RECORDS || NEWTON_ITERATIONS ? { ...d, solver: { ...d.solver, newton: { ...newton, ...(LIN ? { contactLinearization: LIN as 'off' } : {}), ...(LIN_RECORDS ? { contactLinearizationRecords: Number(LIN_RECORDS) } : {}),
    ...(NEWTON_ITERATIONS ? { iterations: Number(NEWTON_ITERATIONS) } : {}) } } } : d];
  return scene;
}

// ------------------------------------------------------------------ runs

const groupOf = (label: string) => {
  if (label.startsWith('ogc ')) return `detector ${label.slice(4)}`;
  if (label.startsWith('Cloth ')) return `solver ${label.slice(6)}`;
  return label;
};

async function stepOnce(cloth: ClothSystem): Promise<number> {
  const t = performance.now();
  if (!cloth.stepTo(cloth.time_s + CLOTH_SCENE_STEP_S)) throw new Error(`step refused at ${cloth.time_s} s: ${cloth.failure?.message ?? 'pending'}`);
  await device.queue.onSubmittedWorkDone();
  return performance.now() - t;
}

interface StepReport { ms: number; spanMs: number; newton: number; pcg: number; /** |g|^2 at the step's first and last Newton iteration, the last accepted step length. */ g2: [number, number]; alpha: number; lin?: Awaited<ReturnType<ClothGPUSolver['readContactLinearization']>>; counters: Awaited<ReturnType<NonNullable<ReturnType<ClothSystem['contactFor']>>['readCounters']>> }
interface ProfileReport {
  steps: number; dispatches: number; worked: number; newtonWorked: number; pcgWorked: number; sumMs: number;
  groups: { group: string; dispatches: number; worked: number; ms: number }[];
}
/** Solver internals the profile reads (harness only). */
type SolverInternals = { newtonWork: GPUBuffer; tables: { newtonRecords: number; pcg: { patches: number; contactLin?: unknown } }; N: number; T: number; E: number;
  run(pass: GPUComputePassEncoder, name: string, workgroups: number, record?: number): void; threads(n: number): number };
const internals = (solver: ClothGPUSolver) => solver as unknown as SolverInternals;
const controlOf = (solver: ClothGPUSolver) => ({ buffer: internals(solver).newtonWork, offset: 16 * (internals(solver).tables.newtonRecords - CLOTH_SEGMENT_ALIGN) });
/** Solver kernels that exit on Newton done (control.x); pcgStep also on PCG done (control.y). */
const NEWTON_GATED = new Set(['nwFaces', 'nwGradient', 'nwLinCount', 'nwLinBuild', 'pcgStep', 'nwProject', 'nwTrial', 'nwChoose', 'nwApply', 'nwEdgeDuals']);
const PCG_GATED = new Set(['pcgStep']);

async function timed(tick: number): Promise<StepReport[]> {
  const cloth = await ClothSystem.create(device, restored(tick), SYSTEM);
  const solver = cloth.solverFor(documentId)!, contact = cloth.contactFor(documentId)!;
  const out: StepReport[] = [];
  for (let s = 0; s < STEPS; s++) {
    stamps.mode = 'span';
    const ms = await stepOnce(cloth);
    stamps.mode = 'off';
    const spans = await stamps.collect();
    const control = await solver.readNewtonControl(), counters = await contact.readCounters();
    if (cloth.failure) throw new Error(`tick ${tick}+${s + 1}: ${cloth.failure.message}`);
    out.push({ ms, spanMs: spans.reduce((a, r) => a + r.ns, 0) / 1e6, newton: control[3]!, pcg: control[7]!, g2: [control[4]!, control[5]!], alpha: control[2]!, lin: await solver.readContactLinearization(), counters });
  }
  cloth.destroy();
  return out;
}

async function profile(tick: number): Promise<ProfileReport> {
  const cloth = await ClothSystem.create(device, restored(tick), SYSTEM);
  stamps.control = controlOf(cloth.solverFor(documentId)!);
  stamps.mode = 'split';
  for (let s = 0; s < PROFILE_STEPS; s++) await stepOnce(cloth);
  stamps.mode = 'off'; stamps.control = undefined;
  const rows = await stamps.collect();
  if (cloth.failure) throw new Error(`profile tick ${tick}: ${cloth.failure.message}`);
  cloth.destroy();
  const groups = new Map<string, { dispatches: number; worked: number; ns: number }>();
  let worked = 0, pcgWorked = 0, newtonWorked = 0, ns = 0;
  rows.forEach((r, k) => {
    const label = r.record.label, kernel = label.startsWith('Cloth ') ? label.slice(6) : '';
    // Work: a detector dispatch with workgroups; a gated solver kernel whose control (before it, and for the PCG pair
    // after it too: a convergence test that sets PCG done is not an iteration) lets it run; any other solver kernel.
    const before = r.control!, after = rows[k + 1]?.control ?? before;
    let did = r.workgroups > 0;
    if (kernel && NEWTON_GATED.has(kernel)) did = before[0] === 0;
    if (kernel && PCG_GATED.has(kernel)) did = before[0] === 0 && before[1] === 0 && after[0] === 0 && after[1] === 0;
    const g = groupOf(label), e = groups.get(g) ?? { dispatches: 0, worked: 0, ns: 0 };
    e.dispatches++; e.ns += r.ns; if (did) { e.worked++; worked++; }
    if (did && kernel === 'pcgStep') pcgWorked++;
    if (did && kernel === 'nwGradient') newtonWorked++;
    ns += r.ns; groups.set(g, e);
  });
  return {
    steps: PROFILE_STEPS, dispatches: rows.length, worked, newtonWorked, pcgWorked: pcgWorked - newtonWorked, sumMs: ns / 1e6,
    groups: [...groups].map(([group, e]) => ({ group, dispatches: e.dispatches, worked: e.worked, ms: e.ns / 1e6 })).sort((a, b) => b.ms - a.ms),
  };
}

/** Exact cost of one working dispatch of each hot solver kernel: after a step, Newton is re-armed and each kernel is
 * dispatched REPLAY times in one timestamped pass (the timestamp clock is 65.5 us coarse; this resolves it). */
async function replay(tick: number): Promise<{ kernel: string; us: number }[]> {
  const cloth = await ClothSystem.create(device, restored(tick), SYSTEM);
  await stepOnce(cloth);
  const solver = cloth.solverFor(documentId)!, S = internals(solver), W = S.threads(S.N), ctl = controlOf(solver), Wp = S.tables.pcg.patches;
  const linKernels: [string, number, number][] = S.tables.pcg.contactLin ? [['nwLinCount', Wp, 0], ['nwLinBuild', Wp, 0]] : [];
  // Timed as a pair: nwLinBuild turns the headers nwLinCount wrote into pool addresses, so it cannot be repeated alone.
  const linPair: [string, number, number][] = linKernels.length ? [['nwLinCount+nwLinBuild', Wp, 0]] : [];
  const plan: [string, number, number][] = [['nwFaces', S.threads(S.T), 0], ['nwGradient', W, 0], ...linPair, ['pcgStep', Wp, 0], ['pcgStep', Wp, 1],
    ['pcgStep', Wp, 2], ['nwProject', W, 0], ['nwTrial', W, 0], ['nwChoose', 1, 0], ['nwApply', W, 0]];
  const set = device.createQuerySet({ type: 'timestamp', count: 2 * plan.length });
  const encoder = device.createCommandEncoder({ label: 'bench replay' });
  device.queue.writeBuffer(ctl.buffer, ctl.offset, new Float32Array(4));
  // Prime the PCG state (dispatches 0 and 1) so the timed dispatch 2 is a full iteration, then time each kernel.
  const prime = encoder.beginComputePass();
  for (const [name, wg, record] of [['nwFaces', S.threads(S.T), 0], ['nwGradient', W, 0], ...linKernels, ['pcgStep', Wp, 0], ['pcgStep', Wp, 1]] as [string, number, number][]) S.run(prime, name, wg, record);
  prime.end();
  plan.forEach(([name, wg, record], k) => {
    const pass = encoder.beginComputePass({ timestampWrites: { querySet: set, beginningOfPassWriteIndex: 2 * k, endOfPassWriteIndex: 2 * k + 1 } });
    for (let r = 0; r < REPLAY; r++) for (const kernel of name.split('+')) S.run(pass, kernel, wg, record);
    pass.end();
  });
  const resolve = device.createBuffer({ size: 16 * plan.length, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
  const read = device.createBuffer({ size: 16 * plan.length, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const ctlRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder.resolveQuerySet(set, 0, 2 * plan.length, resolve, 0);
  encoder.copyBufferToBuffer(resolve, 0, read, 0, 16 * plan.length);
  encoder.copyBufferToBuffer(ctl.buffer, ctl.offset, ctlRead, 0, 16);
  device.queue.submit([encoder.finish()]);
  await read.mapAsync(GPUMapMode.READ); await ctlRead.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(read.getMappedRange().slice(0)), c = new Float32Array(ctlRead.getMappedRange().slice(0));
  read.destroy(); ctlRead.destroy(); resolve.destroy(); set.destroy();
  // A replay whose control flags were set (converged, or a non-positive curvature) timed early exits: say so.
  if (c[0] !== 0 || c[1] !== 0) console.log(`  (replay control after: ${Array.from(c).join(', ')} - some replays exited early)`);
  cloth.destroy();
  return plan.map(([name, , record], k) => ({ kernel: `${name}${PCG_GATED.has(name) ? ` i=${record}` : ''}`, us: Number(t[2 * k + 1]! - t[2 * k]!) / 1e3 / REPLAY }));
}

const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? (s[(s.length - 1) >> 1]! + s[s.length >> 1]!) / 2 : NaN; };

await buildStates();
const vertices = baseScene.cloth![0]!.geometry.data.restPositions_m.length / 3;
console.log(`rotating sphere ${NODES}^2 (${vertices} vertices), Newton ${JSON.stringify(baseScene.cloth![0]!.solver?.newton)}, locality ${LOCALITY}${LIN ? `, contactLinearization ${LIN}` : ''}`);
const report: Record<string, unknown>[] = [];
for (const tick of TICKS) {
  const steps = await timed(tick), prof = PROFILE_STEPS > 0 ? await profile(tick) : undefined;
  const costs = REPLAY > 0 ? await replay(tick) : [];
  const c = steps.map(s => s.counters);
  console.log(`\n== from tick ${tick}: ${STEPS} steps`);
  console.log(`  fenced ms/step median ${median(steps.map(s => s.ms)).toFixed(2)} (min ${Math.min(...steps.map(s => s.ms)).toFixed(2)}, max ${Math.max(...steps.map(s => s.ms)).toFixed(2)}); `
    + `GPU span median ${median(steps.map(s => s.spanMs)).toFixed(2)} ms`);
  console.log(`  Newton iterations used per step: ${steps.map(s => s.newton).join(' ')}; PCG iterations per step: ${steps.map(s => s.pcg).join(' ')}`);
  console.log(`  |g|^2 at the first / last Newton iteration per step: ${steps.map(s => `${s.g2[0].toExponential(4)}/${s.g2[1].toExponential(4)}`).join(' ')}; last step length ${steps.map(s => s.alpha).join(' ')}`);
  const lin = steps[steps.length - 1]!.lin;
  if (lin) console.log(`  contact linearization '${lin.mode}' (last Newton iteration of the last step): ${lin.rows} rows, ${lin.items} items, ${lin.usedRecords} records used (${lin.usedBytes} B) of ${lin.poolRecords} (${lin.poolBytes} B), demand ${lin.demandRecords}; rows left matrix-free ${lin.fallbackRows}`);
  console.log(`  detector per step: candidates ${c.map(x => x.candidates).join(' ')}; pairs (vf+ee) ${c.map(x => x.vfPairs + x.eePairs).join(' ')}; `
    + `rounds (detections+redetections) ${c.map(x => x.detections + x.redetections).join(' ')}; solid contacts ${c.map(x => x.solidContacts).join(' ')}; held ${c.map(x => x.held).join(' ')}; held for plane-record overflow (sum/max) ${c.map(x => `${x.planeRecordHeld}/${x.planeRecordHeldMax}`).join(' ')}; plane-record peak ${c.map(x => x.framePeakPlaneRecords).join(' ')}; status ${c.map(x => x.status).join(' ')}`);
  if (prof) {
    console.log(`  profile (${prof.steps} step, split passes): ${prof.dispatches} dispatches queued, ${prof.worked} did work; `
      + `Newton iterations with work ${prof.newtonWorked}, PCG iterations with work ${prof.pcgWorked}; split sum ${prof.sumMs.toFixed(2)} ms (65.5 us clock)`);
    for (const g of prof.groups.filter(g => g.ms >= 0.05 || g.worked > 0).slice(0, 32))
      console.log(`    ${g.group.padEnd(30)} ${g.ms.toFixed(3).padStart(9)} ms  ${String(g.dispatches).padStart(5)} queued ${String(g.worked).padStart(5)} worked`);
  }
  if (costs.length) console.log(`  replay (${REPLAY} dispatches per kernel, one pass): ${costs.map(c => `${c.kernel} ${c.us.toFixed(1)} us`).join('; ')}`);
  report.push({ tick, steps, profile: prof, replay: costs });
}
await device.queue.onSubmittedWorkDone();
if (JSON_OUT) writeFileSync(JSON_OUT, JSON.stringify({ nodes: NODES, vertices, locality: LOCALITY, report }, null, 1));
if (errors.length) { console.error(`${errors.length} WebGPU errors`); process.exitCode = 1; }
device.destroy();
