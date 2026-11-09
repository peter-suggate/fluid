/**
 * The scene's cloth system: every `SceneDescription.cloth` document runs in
 * its own Tier A `ClothGPUSolver`, one implicit step of `CLOTH_SCENE_STEP_S`
 * per scene-clock tick. There are no substeps; K is the only budget.
 *
 * In a dry scene the cloth owns the clock: each advance the renderer owes
 * steps every document once and moves `info.submittedTime_s` one tick. Beside
 * a fluid solver the cloth follows the fluid's submitted time, stepping
 * whole ticks to catch up (at most `CLOTH_CATCH_UP_STEPS` per advance).
 *
 * The renderer draws the solver's buffers in place. The pose is read back
 * only when something asks for it (`readPose`: a press that picks, an edit,
 * a capture), never on the step cadence; every few steps only the 4-byte
 * status words (the solver's and the contact detector's) are copied back,
 * so a non-finite step or a contact failure halts the scene. A failing status
 * word, a pending solid (`ClothDocument.pendingSolids`), a solid world contact
 * cannot follow or a document the system cannot run halts the scene with an
 * explicit `SimulationFailure`.
 *
 * Every document runs with lane C2's OGC contact (`scene-contact.ts`): self
 * contact and the scene's solids, as one static field per solid world and
 * cloth frame, built at setup and on solid edits, never in a step.
 *
 * An edited document (a cut, an undo) is rebuilt in place of the live one,
 * from the pose and solver state the edit carries; the clock waits for it.
 */
import type { GPUSolverInstance } from '../core/method-contract';
import type { SceneDescription } from '../core/model';
import type { SimulationFailure } from '../core/simulation-failure';
import { clothDocumentsKey } from './content-stamp';
import { ClothGPUSolver } from './gpu-solver';
import { CLOTH_BINDING, CLOTH_STATUS, CLOTH_STATUS_WORDS } from './gpu-solver-wgsl';
import { OgcContactDetector } from './contact/ogc-detector';
import type { ClothSolidField } from './contact/solid-field';
import { buildClothCoarseSpace } from './coarse';
import { buildClothSolverModel } from './oracle';
import type { ClothRenderSource } from './render';
import { CLOTH_CONTACT_FAILING, CLOTH_CONTACT_STATUS_OFFSET, buildClothSceneField, clothContactFieldKey, clothContactRadius, createClothSceneContact } from './scene-contact';
import { sceneClothSurface } from './scene-geometry';
import { clothVertexNeighbours, colourClothVertices, prepareClothTopology } from './topology';
import type { ClothDocument, ClothSolverStateData, ClothSurfacePublication, ClothVector } from './types';

/** One implicit cloth step per 1/60 s frame. */
export const CLOTH_SCENE_STEP_S = 1 / 60;
/** Most steps a lagging document takes in one advance beside a fluid solver. */
export const CLOTH_CATCH_UP_STEPS = 4;
/** Steps between status-word reads (the non-finite watchdog; never the pose). */
export const CLOTH_STATUS_INTERVAL_STEPS = 4;
const DEFAULT_ITERATIONS = 64;
/** K with the two-level correction (lane C1 sweep, 2026-11-09: 150^2 sheet, strain and energy). */
const DEFAULT_COARSE_ITERATIONS = 8;

let systemGenerations = 0, renderRevisions = 0;
const align = (n: number, a: number) => Math.ceil(n / a) * a;

/** Asynchronous copy of the status words and, on request, positions and velocities; one map in flight. */
class PoseMirror {
  private readonly staging: GPUBuffer;
  private readonly statusStaging: GPUBuffer;
  private readonly span: number;
  busy = false;
  statusBusy = false;
  constructor(device: GPUDevice, private readonly solver: ClothGPUSolver, label: string, private readonly contact?: OgcContactDetector) {
    this.span = align(16 * Math.max(solver.N, 1), 256);
    this.staging = device.createBuffer({ label: `${label} pose mirror`, size: 2 * this.span + 4 * CLOTH_STATUS_WORDS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this.statusStaging = device.createBuffer({ label: `${label} status mirror`, size: 4 * CLOTH_STATUS_WORDS + 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  }
  encode(encoder: GPUCommandEncoder): boolean {
    if (this.busy) return false;
    const s = this.solver, v = s.velocities, bytes = 16 * s.N;
    encoder.copyBufferToBuffer(s.positions, 0, this.staging, 0, bytes);
    encoder.copyBufferToBuffer(v.buffer, v.offset ?? 0, this.staging, this.span, bytes);
    encoder.copyBufferToBuffer(s.status, 0, this.staging, 2 * this.span, 4 * CLOTH_STATUS_WORDS);
    this.busy = true;
    return true;
  }
  encodeStatus(encoder: GPUCommandEncoder): boolean {
    if (this.statusBusy) return false;
    encoder.copyBufferToBuffer(this.solver.status, 0, this.statusStaging, 0, 4 * CLOTH_STATUS_WORDS);
    if (this.contact) encoder.copyBufferToBuffer(this.contact.counters, CLOTH_CONTACT_STATUS_OFFSET, this.statusStaging, 4 * CLOTH_STATUS_WORDS, 4);
    this.statusBusy = true;
    return true;
  }
  /** After the encoder holding `encode` is submitted. */
  async read(): Promise<{ positions: Float32Array; velocities: Float32Array; status: number }> {
    try {
      await this.staging.mapAsync(GPUMapMode.READ);
      const all = this.staging.getMappedRange(), n = 4 * this.solver.N;
      const out = { positions: new Float32Array(all.slice(0, 4 * n)), velocities: new Float32Array(all.slice(this.span, this.span + 4 * n)),
        status: new Uint32Array(all.slice(2 * this.span, 2 * this.span + 4))[0]! };
      this.staging.unmap();
      return out;
    } finally { this.busy = false; }
  }
  /** After the encoder holding `encodeStatus` is submitted: the solver's status word and the detector's (0 without contact). */
  async readStatus(): Promise<{ solver: number; contact: number }> {
    try {
      await this.statusStaging.mapAsync(GPUMapMode.READ);
      const words = new Uint32Array(this.statusStaging.getMappedRange().slice(0));
      this.statusStaging.unmap();
      return { solver: words[0]!, contact: this.contact ? words[CLOTH_STATUS_WORDS]! : 0 };
    } finally { this.statusBusy = false; }
  }
  destroy(): void { this.staging.destroy(); this.statusStaging.destroy(); }
}

interface Entry {
  readonly document: ClothDocument;
  readonly key: string;
  readonly solver: ClothGPUSolver;
  readonly mirror: PoseMirror;
  /** The contact detector and the static field it holds (`fieldKey` names the solid world and frame it was built for). */
  readonly contact: OgcContactDetector;
  field: ClothSolidField;
  fieldKey: string;
  readonly origin: ClothVector;
  readonly revision: number;
  readonly topologyRevision: number;
  steps: number;
  sinceStatus: number;
  publication: ClothSurfacePublication;
  inflight?: Promise<void>;
  /** Zero BDF2 history and duals, shared by every mirrored publication (see `restartState`). */
  zeros?: Omit<ClothSolverStateData, 'time_s'>;
}

export interface ClothSystemOptions {
  readonly label?: string;
  /** Steps between status-word reads; default `CLOTH_STATUS_INTERVAL_STEPS`. */
  readonly statusInterval?: number;
}

export class ClothSystem {
  readonly generation = ++systemGenerations;
  failure?: SimulationFailure;
  /** Reported (not failing) status bits, by name, with the scene time a status read first showed each: the
   * detector's roundCap and envelopeCap (it took its authored cap) and the solver's coarseIndefinite (a step ran
   * without the two-level correction), so the scene's diagnostics show them. */
  readonly notes = new Map<string, number>();
  private readonly entries = new Map<string, Entry>();
  private readonly building = new Map<string, { key: string; ready: Promise<void> }>();
  private readonly statusInterval: number;
  private poseRead?: Promise<void>;
  /** A solid edit's field rebuild; the clock waits for it. */
  private solidUpdate?: Promise<void>;
  private solidVoxels: SceneDescription['solidVoxels'];
  private destroyed = false;

  private constructor(private readonly device: GPUDevice, private scene: SceneDescription, private readonly options: ClothSystemOptions) {
    this.statusInterval = options.statusInterval ?? CLOTH_STATUS_INTERVAL_STEPS;
    this.solidVoxels = scene.solidVoxels;
  }

  static async create(device: GPUDevice, scene: SceneDescription, options: ClothSystemOptions = {}): Promise<ClothSystem> {
    const system = new ClothSystem(device, scene, options);
    const entries = await Promise.all((scene.cloth ?? []).map(d => system.build(d, sceneClothSurface(scene, d.id)?.topologyRevision ?? 0)));
    for (const e of entries) system.entries.set(e.document.id, e);
    return system;
  }

  /** Scene time every document has reached. */
  get time_s(): number {
    let t = Infinity;
    for (const e of this.entries.values()) t = Math.min(t, this.entryTime(e));
    return Number.isFinite(t) ? t : 0;
  }
  /** A document is being rebuilt after an edit; the clock waits for it. */
  get pending(): boolean { return this.building.size > 0 || !!this.solidUpdate; }

  /** The poses `readPose` last accepted, for picking, cuts and history; a document being rebuilt is left out. */
  get surfaces(): ClothSurfacePublication[] {
    return [...this.entries.values()].filter(e => !this.building.has(e.document.id)).map(e => e.publication);
  }

  get renderSources(): ClothRenderSource[] {
    return [...this.entries.values()].map(e => ({ document: e.document, positions: e.solver.positionsBinding, normals: e.solver.normals,
      triangles: e.solver.triangles, triangleCount: e.solver.T, origin_m: e.origin, revision: e.revision, tick: this.tick(e) }));
  }

  /** The live solver of a document (harnesses and diagnostics). */
  solverFor(documentId: string): ClothGPUSolver | undefined { return this.entries.get(documentId)?.solver; }
  /** The live contact detector of a document (harnesses and diagnostics). */
  contactFor(documentId: string): OgcContactDetector | undefined { return this.entries.get(documentId)?.contact; }

  private entryTime(e: Entry): number { return e.solver.startTime_s + e.steps * CLOTH_SCENE_STEP_S; }
  private tick(e: Entry): number { return Math.round(this.entryTime(e) / CLOTH_SCENE_STEP_S); }

  /** The field for a frame: a live entry's when it matches (a cut keeps its frame), else built now (seconds of CPU for a large scene). */
  private fieldFor(origin: ClothVector, r: number): { field: ClothSolidField; fieldKey: string } {
    const fieldKey = clothContactFieldKey(this.scene, origin, r);
    for (const e of this.entries.values()) if (e.fieldKey === fieldKey) return { field: e.field, fieldKey };
    return { field: buildClothSceneField(this.scene, origin, r), fieldKey };
  }

  private async build(document: ClothDocument, topologyRevision: number, frame?: ClothVector): Promise<Entry> {
    const settings = document.solver ?? {};
    if (settings.tier === 'B') throw new Error(`Cloth document '${document.id}' asks for the Tier B patch solve, which is not installed yet (lane C1, M3)`);
    if (document.geometry.kind !== 'inline') throw new Error(`Cloth document '${document.id}' has an unsupported geometry kind`);
    const data = document.geometry.data, topology = prepareClothTopology(data, document.materials);
    const model = buildClothSolverModel(topology, document.materials, colourClothVertices(clothVertexNeighbours(topology)), document.scripts ?? []);
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    data.restPositions_m.forEach((v, i) => { lo[i % 3] = Math.min(lo[i % 3]!, v); hi[i % 3] = Math.max(hi[i % 3]!, v); });
    // A rebuilt document keeps its frame, so its contact field is reused.
    const origin = frame ?? [0, 1, 2].map(a => (lo[a]! + hi[a]!) / 2) as unknown as ClothVector;
    const g = this.scene.fluid.gravity_m_s2, gravity: [number, number, number] = [g.x, g.y, g.z], label = `${this.options.label ?? 'Cloth'} ${document.id}`;
    const { field, fieldKey } = this.fieldFor(origin, clothContactRadius(document));
    // The two-level correction's coarse space follows the mesh topology, so it is rebuilt with it (edit time only).
    const coarse = settings.coarse === false ? undefined : buildClothCoarseSpace(topology);
    const contact = await createClothSceneContact(this.device, model, document, origin, field, CLOTH_SCENE_STEP_S, gravity);
    let solver: ClothGPUSolver;
    try {
      solver = await ClothGPUSolver.create(this.device, model, {
        origin_m: origin, dt_s: CLOTH_SCENE_STEP_S, gravity_m_s2: gravity,
        integrator: settings.integrator ?? 'bdf2', iterations: settings.iterations ?? (coarse ? DEFAULT_COARSE_ITERATIONS : DEFAULT_ITERATIONS), chebyshevRho: settings.chebyshevRho ?? 0,
        // The extrapolating guesses overshoot an unconverged solve; the coarse step carries the bulk motion instead.
        ...(coarse ? { coarse, initialGuess: 'previous' as const } : {}),
        positions_m: data.initialPositions_m, velocities_m_s: data.initialVelocities_m_s, solverState: data.solverState, label,
        contact: contact.solverContact(CLOTH_BINDING.contact),
        // The coarse move is truncated like a colour sweep's, before the sweeps start from it.
        hooks: { ...contact.stepHooks, afterCoarse: (pass: GPUComputePassEncoder) => contact.encodeTruncate(pass) },
      });
      contact.bind({ positions: solver.positions, velocities: solver.velocities, stepStart: solver.stepStart });
      contact.setSolidField(field);
    } catch (error) { contact.destroy(); throw error; }
    const positions = Float64Array.from(data.initialPositions_m ?? data.restPositions_m);
    const entry: Entry = { document, key: clothDocumentsKey([document]), solver, mirror: new PoseMirror(this.device, solver, label, contact), contact, field, fieldKey, origin,
      revision: ++renderRevisions, topologyRevision, steps: 0, sinceStatus: 0,
      publication: { documentId: document.id, topology: data, positions_m: positions,
        ...(data.initialVelocities_m_s ? { velocities_m_s: Float64Array.from(data.initialVelocities_m_s) } : {}),
        ...(data.solverState ? { solverState: data.solverState } : {}),
        generation: this.generation, tick: 0, topologyRevision, time_s: solver.startTime_s } };
    entry.publication = { ...entry.publication, tick: this.tick(entry) };
    return entry;
  }

  /**
   * Step every document to `target_s` in whole ticks (at most `maxSteps`
   * each), in one submission. Returns whether any step was encoded. A pending
   * solid the next step would pass halts the system before it is encoded.
   */
  stepTo(target_s: number, maxSteps = 1): boolean {
    if (this.failure || this.destroyed || this.pending) return false;
    const encoder = this.device.createCommandEncoder({ label: 'Cloth system step' }), watched: Entry[] = [];
    let stepped = false;
    for (const e of this.entries.values()) {
      let owed = Math.min(maxSteps, Math.floor((target_s - this.entryTime(e)) / CLOTH_SCENE_STEP_S + 0.5));
      if (owed <= 0) continue;
      const next = this.entryTime(e) + owed * CLOTH_SCENE_STEP_S;
      const blocker = (e.document.pendingSolids ?? []).find(p => next > p.start_s + 1e-9);
      if (blocker) {
        owed = Math.floor((blocker.start_s - this.entryTime(e)) / CLOTH_SCENE_STEP_S + 1e-9);
        if (owed <= 0) { this.fail('cloth-pending-solid', `moving cloth solid '${blocker.id}' requires C2 M6`, 'ClothSystem.stepTo', e, [blocker.start_s]); return false; }
      }
      // One pass per frame: the detector's frame statistics restart, then the steps (its hooks run inside each).
      const pass = encoder.beginComputePass({ label: 'Cloth step' });
      e.contact.encodeFrameBegin(pass);
      for (let s = 0; s < owed; s++) e.solver.encodeStep(pass);
      pass.end();
      e.steps += owed; e.sinceStatus += owed; stepped = true;
      if (e.sinceStatus >= this.statusInterval && e.mirror.encodeStatus(encoder)) { e.sinceStatus = 0; watched.push(e); }
    }
    if (!stepped) return false;
    this.device.queue.submit([encoder.finish()]);
    for (const e of watched) void this.watch(e);
    return true;
  }

  private async watch(e: Entry): Promise<void> {
    let status;
    try { status = await e.mirror.readStatus(); } catch { return; } // device lost or destroyed
    if (this.destroyed || this.entries.get(e.document.id) !== e) return;
    if (status.solver & CLOTH_STATUS.nonFinite) this.fail('cloth-non-finite', ClothGPUSolver.describeStatus(status.solver & CLOTH_STATUS.nonFinite).join('; '), 'finishStep', e, [status.solver]);
    const notes = [...OgcContactDetector.describeStatus(status.contact & ~CLOTH_CONTACT_FAILING), ...(status.solver & CLOTH_STATUS.coarseIndefinite ? ['coarseIndefinite'] : [])];
    for (const name of notes) if (!this.notes.has(name)) this.notes.set(name, this.entryTime(e));
    if (status.contact & CLOTH_CONTACT_FAILING)
      this.fail('cloth-contact', `contact failed: ${OgcContactDetector.describeStatus(status.contact & CLOTH_CONTACT_FAILING).join('; ')}`, 'OgcContactDetector', e, [status.contact]);
  }

  /**
   * Read every document's pose back now and publish it: what a press that
   * picks, an edit or a capture asks for. Concurrent requests share one read.
   */
  readPose(): Promise<void> {
    if (this.poseRead) return this.poseRead;
    const read = (async () => {
      await Promise.all([...this.entries.values()].map(e => e.inflight));
      if (this.destroyed) return;
      const encoder = this.device.createCommandEncoder({ label: 'Cloth pose read' }), entries = [...this.entries.values()].filter(e => e.mirror.encode(encoder));
      this.device.queue.submit([encoder.finish()]);
      await Promise.all(entries.map(e => (e.inflight = this.receive(e, this.entryTime(e)))));
    })().finally(() => { if (this.poseRead === read) this.poseRead = undefined; });
    return (this.poseRead = read);
  }

  /**
   * The solver state a mirrored pose carries: its clock, with the BDF2
   * history and AVBD duals of a fresh start. A cut or undo from it restarts
   * the integrator with one implicit Euler step at the right scene time, so
   * attachment scripts and pending-solid gates stay on the scene clock. (The
   * exact history needs a full state read; lane C1 M4.)
   */
  private restartState(e: Entry, time_s: number): ClothSolverStateData {
    if (!e.zeros) {
      const n = e.solver.N, pins = e.solver.model.topology.groups.reduce((c: number, g: number) => c + (g > 0 ? 1 : 0), 0);
      const zeros = (k: number) => new Array<number>(k).fill(0);
      e.zeros = { previousVelocities_m_s: zeros(3 * n), previousDisplacements_m: zeros(3 * n), previousStep_s: 0, pinDuals: zeros(4 * pins), edgeDuals: zeros(2 * e.solver.E) };
    }
    return { ...e.zeros, time_s };
  }

  private async receive(e: Entry, time_s: number): Promise<void> {
    const tick = Math.round(time_s / CLOTH_SCENE_STEP_S);
    let read;
    try { read = await e.mirror.read(); } catch { return; } // device lost or destroyed: nothing to publish
    if (this.destroyed || this.entries.get(e.document.id) !== e) return;
    if (read.status & CLOTH_STATUS.nonFinite) { this.fail('cloth-non-finite', ClothGPUSolver.describeStatus(read.status).join('; '), 'finishStep', e, [read.status]); return; }
    if (tick < e.publication.tick) return;
    const n = e.solver.N, positions = new Float64Array(3 * n), velocities = new Float64Array(3 * n);
    for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
      positions[3 * i + c] = read.positions[4 * i + c]! + e.origin[c]!; velocities[3 * i + c] = read.velocities[4 * i + c]!;
    }
    e.publication = { documentId: e.document.id, topology: e.document.geometry.data, positions_m: positions, velocities_m_s: velocities,
      solverState: this.restartState(e, time_s), generation: this.generation, tick, topologyRevision: e.topologyRevision, time_s };
  }

  private fail(code: string, message: string, kernel: string, e: Entry, operands: number[]): void {
    this.failure ??= { method: 'cloth', code, message: `${e.document.id}: ${message}`, kernel, frame: this.tick(e), generation: this.generation,
      ownerId: 0, operands, rawWords: operands, scene: this.scene.sceneId, time_s: this.entryTime(e) };
  }

  /**
   * Take a new revision of the scene. Documents whose content changed (a cut,
   * an undo) are rebuilt from the pose the edit carries; the live entry keeps
   * drawing until its replacement is ready. Roster changes rebuild the system.
   */
  adopt(scene: SceneDescription): void {
    this.scene = scene;
    if (scene.solidVoxels !== this.solidVoxels) { this.solidVoxels = scene.solidVoxels; this.followSolids(); }
    for (const document of scene.cloth ?? []) {
      const live = this.entries.get(document.id), key = clothDocumentsKey([document]);
      if (this.building.get(document.id)?.key === key || (!this.building.has(document.id) && live?.key === key)) continue;
      const topologyRevision = Math.max(sceneClothSurface(scene, document.id)?.topologyRevision ?? 0, live ? live.topologyRevision + 1 : 0);
      const ready = this.build(document, topologyRevision, live?.origin).then(entry => {
        if (this.destroyed || this.building.get(document.id)?.key !== key) { this.retire(entry); return; }
        const old = this.entries.get(document.id);
        this.entries.set(document.id, entry); this.building.delete(document.id);
        if (old) this.retire(old);
      }, error => {
        if (this.building.get(document.id)?.key !== key) return;
        this.building.delete(document.id);
        const message = error instanceof Error ? error.message : String(error);
        this.failure ??= { method: 'cloth', code: 'cloth-rebuild', message: `${document.id}: ${message}`, kernel: 'ClothSystem.adopt', frame: 0,
          generation: this.generation, ownerId: 0, operands: [], rawWords: [], scene: scene.sceneId };
      });
      this.building.set(document.id, { key, ready });
    }
  }

  /**
   * A solid edit: rebuild each live document's field from the edited solid
   * world (after this call returns, off the step path) and upload it. A solid
   * contact cannot follow yet halts the scene; the clock waits meanwhile.
   */
  private followSolids(): void {
    const update: Promise<void> = Promise.resolve().then(() => {
      for (const e of this.entries.values()) {
        if (this.destroyed || this.failure) return;
        const key = clothContactFieldKey(this.scene, e.origin, clothContactRadius(e.document));
        if (key === e.fieldKey) continue;
        try {
          const field = buildClothSceneField(this.scene, e.origin, clothContactRadius(e.document));
          e.contact.setSolidField(field);
          e.field = field; e.fieldKey = key;
        } catch (error) { this.fail('cloth-solid-edit', error instanceof Error ? error.message : String(error), 'ClothSystem.adopt', e, []); return; }
      }
    }).finally(() => { if (this.solidUpdate === update) this.solidUpdate = undefined; });
    this.solidUpdate = update;
  }

  /** Resolves when every rebuild started by `adopt` has landed. */
  async settled(): Promise<void> {
    while (this.building.size || this.solidUpdate) await Promise.all([...this.building.values()].map(b => b.ready).concat(this.solidUpdate ?? []));
  }

  private retire(e: Entry): void {
    // The renderer rebinds before its next draw; work already submitted finishes first.
    void this.device.queue.onSubmittedWorkDone().then(() => { e.solver.destroy(); e.mirror.destroy(); e.contact.destroy(); }, () => undefined);
  }

  /**
   * Lane S's capture hook: run the cloth on its own clock, exactly as the
   * scene does (one step per tick), and return the accepted pose of every
   * document at each requested scene time (rounded to the tick). Times must
   * not decrease. Rejects with the failure when the system halts first.
   */
  async capture(times_s: readonly number[]): Promise<ClothSurfacePublication[][]> {
    const frames: ClothSurfacePublication[][] = [];
    let last = -Infinity;
    for (const t of times_s) {
      if (!(t >= last)) throw new Error('Cloth capture times must not decrease');
      last = t;
      while (this.time_s + CLOTH_SCENE_STEP_S / 2 <= t) {
        if (!this.stepTo(this.time_s + CLOTH_SCENE_STEP_S)) break;
        await this.device.queue.onSubmittedWorkDone();
      }
      await this.readPose();
      if (this.failure) throw new Error(`Cloth capture halted at ${this.failure.time_s ?? 0} s: ${this.failure.message}`);
      frames.push(this.surfaces);
    }
    return frames;
  }

  destroy(): void {
    this.destroyed = true;
    for (const e of this.entries.values()) { e.solver.destroy(); e.mirror.destroy(); e.contact.destroy(); }
    this.entries.clear(); this.building.clear();
  }
}

const OVERRIDDEN = new Set<PropertyKey>(['advanceTo', 'clothSurfaces', 'clothRenderSources', 'readClothPose', 'destroy', 'stageSceneUpdate', 'applySceneUniforms', 'reseed']);

/**
 * The scene solver with the cloth system composed onto it. Everything the
 * base instance has (and only that) is forwarded, so the renderer's checks
 * for optional members still see the base solver; the cloth adds its clock,
 * its surfaces and render sources, and takes scene revisions alongside it.
 */
export function withClothSystem(base: GPUSolverInstance, cloth: ClothSystem, fluid: boolean): GPUSolverInstance {
  const latch = () => {
    if (cloth.failure && !base.info.simulationFailure) base.info.simulationFailure = cloth.failure;
    base.info.clothNotes = [...cloth.notes].map(([name, t]) => `${name} since ${t.toFixed(3)} s`);
  };
  const overrides: Partial<GPUSolverInstance> & Record<string, unknown> = {
    advanceTo(time_s, bodies) {
      if (fluid) {
        const advanced = base.advanceTo(time_s, bodies);
        cloth.stepTo(base.info.submittedTime_s ?? 0, CLOTH_CATCH_UP_STEPS);
        latch();
        return advanced;
      }
      base.advanceTo(time_s, bodies);
      const now = base.info.submittedTime_s ?? 0;
      if (time_s + 1e-9 < now + CLOTH_SCENE_STEP_S / 2) return false;
      const stepped = cloth.stepTo(now + CLOTH_SCENE_STEP_S);
      latch();
      if (!stepped) return false;
      base.info.submittedTime_s = base.info.simulatedTime_s = cloth.time_s;
      return true;
    },
    get clothSurfaces() { latch(); return cloth.surfaces; },
    get clothRenderSources() { return cloth.renderSources; },
    readClothPose: () => cloth.readPose().then(latch),
    destroy() { cloth.destroy(); base.destroy(); },
  };
  if (base.stageSceneUpdate) overrides.stageSceneUpdate = (scene: SceneDescription) => { cloth.adopt(scene); base.stageSceneUpdate!(scene); };
  if (base.applySceneUniforms) overrides.applySceneUniforms = (scene: SceneDescription) => { cloth.adopt(scene); base.applySceneUniforms!(scene); };
  if (base.reseed) overrides.reseed = async (scene: SceneDescription) => {
    // Re-seeding restarts the clock at t = 0; the cloth system cannot rewind in place.
    if (scene.cloth?.length) return false;
    return base.reseed!(scene);
  };
  return new Proxy(base, {
    get(target, property) {
      if (OVERRIDDEN.has(property)) return Object.getOwnPropertyDescriptor(overrides, property)?.get?.call(overrides) ?? overrides[property as string];
      const value = Reflect.get(target, property, target);
      return typeof value === 'function' ? value.bind(target) : value;
    },
    has(target, property) { return OVERRIDDEN.has(property) ? property in overrides : Reflect.has(target, property); },
    set(target, property, value) { return Reflect.set(target, property, value, target); },
  });
}

/** Build the cloth system for a scene and compose it onto the scene's solver. */
export async function composeSceneCloth(device: GPUDevice, scene: SceneDescription, base: GPUSolverInstance, fluid: boolean): Promise<GPUSolverInstance> {
  let cloth: ClothSystem;
  try { cloth = await ClothSystem.create(device, scene); }
  catch (error) { base.destroy(); throw error; }
  return withClothSystem(base, cloth, fluid);
}
