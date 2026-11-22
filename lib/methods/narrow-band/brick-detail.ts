/**
 * The brick engine's detail engine: the scene run again at h / 2 or h / 4 by a nested brick engine, but only on a
 * mask of the parent's 4h tiles chosen every step on the GPU (brick-detail-alloc.wgsl.ts: the parent's surface
 * deformation heat under a fixed budget, plus the authored detail regions' tiles), driven by the parent at the mask's
 * edge (brick-detail.wgsl.ts's relaxation shell). A scene keeps its resolution and gains particle-scale fidelity
 * where its surface deforms fastest (a splash's crown, a sheet breaking on an obstacle) for a cost the budget bounds.
 *
 * - brickDetailSpecs: the scene's detail engine (brick-detail-regions.ts brickDetailPlan), its fine scene
 *   (brick-detail-scene.ts: the container at h / factor, empty), its setup and pools (the budget's, never the
 *   container's) and its coupling layout (BrickLayoutOptions.detail);
 * - BrickDetailAlloc: the allocator's six dispatches on the parent's layout, after every parent step (and one before
 *   the fine boot), into the fine engine's `src` (the slot map, lists and the parent's nodes) and `dir` (the mask).
 * The order a step (BrickNBFlipSolver.advanceTo): the parent's step, the allocator, the fine step. The fine engine
 * relaxes toward the parent's state at the step's end (t + dt) on its way from t: the shell is a boundary condition
 * taken at the new time, as an inflow's prescribed velocity is. One way: the parent runs as if the detail engine were
 * not there, and the renderer draws the fine surface on the on tiles in place of the parent's.
 */
import type { SceneDescription } from "../../core/model";
import { fluidLatticeFarPhi } from "../../core/initial-liquid-lattice";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";
import { brickDetailPlan, type DetailPlan } from "./brick-detail-regions";
import { brickDetailFrame, brickDetailSceneGaps, brickDetailSubScene } from "./brick-detail-scene";
import { type BrickDetailLayout } from "./brick-detail.wgsl";
import { DETAIL_ST, brickDetailAllocDispatches, brickDetailAllocWGSL, brickDetailAllocWords } from "./brick-detail-alloc.wgsl";
import { BrickEngine, type Vec3 } from "./brick-engine";
import type { BrickLayout } from "./brick-abi";
import { BrickPresentation } from "./brick-present";
import { BrickDetailSurface } from "./brick-detail-surface";
import type { PressureSolverFactory } from "./brick-pressure-slot";
import type { BrickStagesFactory } from "./brick-stages";
import { brickStaticCutBricks } from "./brick-static.wgsl";
import { solidWorldForFluidLattice } from "../../core/solid-world";
import { brickSceneSetup, type BrickSceneSetup } from "./brick-scene";

export interface BrickDetailSpec {
  readonly plan: DetailPlan;
  /** The fine engine's setup (h / factor, the container's own scene, the budget's pools). */
  readonly setup: BrickSceneSetup;
  readonly layout: BrickDetailLayout;
  /** The container's low corner in the parent's world (m): both lattices' origin. */
  readonly low_m: readonly [number, number, number];
  readonly farPhi_m: number;
}

/** The ring's slots a budget: an active tile's inactive neighbours, a sheet's two sides and its rim. */
export const DETAIL_RING = 2;

/** The fine engine's pools from the budget: its storage follows the on tiles, never the container. */
export function brickDetailPools(layout: Pick<BrickDetailLayout, "factor" | "budget">, k: number) {
  const perPage = (32 / (4 * layout.factor)) ** 3;
  // F: the bound on what the mask keeps (nbDtlBrickKeep), a lone tile's bricks and the layer past its + sides (its
  // upper faces and vertices), ((factor / 2) + 1)³ a tile, and a quarter more for the tiles the mask turns in a step
  // (a leaving slot is free only after the step's claims).
  const f = Math.max(128, Math.ceil(1.25 * layout.budget * (layout.factor / 2 + 1) ** 3));
  // C: a page each of the on tiles' 3³ blocks it meets (a block straddles up to eight pages) and of their ring.
  const c = Math.min(f, Math.max(128, Math.ceil(2 * layout.budget * Math.min(1, 64 / perPage))));
  return { f, c, b: 16, k: Math.min(k, f), particles: Math.max(4096, 400 * f), t: Math.min(512 * c, Math.max(4 * f, 8192)) };
}

/**
 * The scene's detail engine (one or none); an explicit refusal (throws) when the scene asks for detail the engine
 * cannot run (brickDetailSceneGaps, the fine setup's gaps).
 */
export function brickDetailSpecs(scene: SceneDescription, _setup: BrickSceneSetup, values: Readonly<Record<string, unknown>>): BrickDetailSpec[] {
  const cells = sceneLatticeDimensions(scene) as unknown as Vec3;
  const plan = brickDetailPlan(scene, cells);
  if (!plan) return [];
  const gaps = brickDetailSceneGaps(scene);
  if (gaps.length) throw new Error(`NB-FLIP brick engine cannot run scene "${scene.sceneId ?? "(unnamed)"}"'s detail: ${gaps.join("; ")}`);
  const sub = brickDetailSubScene(scene, plan.factor);
  const fine = brickSceneSetup(sub, values);
  if (!fine.setup || fine.gaps.length) throw new Error(`NB-FLIP detail engine: ${fine.gaps.join("; ")}`);
  const budget = plan.budget + plan.forced.length;
  if (budget < 1) throw new Error("NB-FLIP detail engine: a budget of no tiles");
  const layout: BrickDetailLayout = { factor: plan.factor, tiles: plan.tiles, budget, slots: budget * (1 + DETAIL_RING) };
  const setup: BrickSceneSetup = { ...fine.setup, capacities: brickDetailPools(layout, fine.setup.capacities.k) };
  return [{ plan, setup, layout, low_m: brickDetailFrame(scene).low_m, farPhi_m: fluidLatticeFarPhi(sub) }];
}

/** The allocator: the mask, the slots and the parent's nodes, a pass after every parent step. */
export class BrickDetailAlloc {
  private group?: GPUBindGroup;
  private constructor(private readonly device: GPUDevice, private readonly parent: BrickEngine, private readonly pipelines: readonly [GPUComputePipeline, [number, number, number]][], private readonly bindLayout: GPUBindGroupLayout, readonly state: GPUBuffer, private readonly forced: Uint32Array, private readonly forcedAt: number) {}

  static async create(device: GPUDevice, parent: BrickEngine, spec: Pick<BrickDetailSpec, "plan" | "layout">): Promise<BrickDetailAlloc> {
    const forced = spec.plan.forced, o = { dt_s: parent.options.dt_s, h_m: parent.options.h_m, forced: forced.length };
    const code = brickDetailAllocWGSL(parent.layout, spec.layout, o), label = "nb-brick.detailAlloc";
    const module = device.createShaderModule({ label, code });
    const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
    if (errors.length) throw new Error(`NB-FLIP detail allocator does not compile: ${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("; ")}`);
    const C = GPUShaderStage.COMPUTE;
    const layout = device.createBindGroupLayout({ label, entries: [
      ...[0, 1, 2, 3].map(binding => ({ binding, visibility: C, buffer: { type: "read-only-storage" as const } })),
      ...[4, 5, 6].map(binding => ({ binding, visibility: C, buffer: { type: "storage" as const } }))] });
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] });
    const dispatches = brickDetailAllocDispatches(parent.layout, spec.layout, forced.length);
    const pipelines = await Promise.all(Object.entries(dispatches).map(async ([entryPoint, groups]) =>
      [await device.createComputePipelineAsync({ label: `${label}.${entryPoint}`, layout: pipelineLayout, compute: { module, entryPoint } }), groups] as [GPUComputePipeline, [number, number, number]]));
    const words = brickDetailAllocWords(spec.layout, parent.layout, forced.length);
    const state = device.createBuffer({ label: `${label}.state`, size: 4 * words.words, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    return new BrickDetailAlloc(device, parent, pipelines, layout, state, forced, words.forced);
  }

  /** Bind the fine engine's `src` and `dir` (once, before the first encode), and write the authored tiles. */
  bind(src: GPUBuffer, dir: GPUBuffer, fine: BrickLayout) {
    const p = this.parent;
    this.group = this.device.createBindGroup({ label: "nb-brick.detailAlloc", layout: this.bindLayout, entries: [
      { binding: 0, resource: { buffer: p.buffer("set") } }, { binding: 1, resource: { buffer: p.buffer("dir") } },
      { binding: 2, resource: { buffer: p.buffer("phi") } }, { binding: 3, resource: { buffer: p.buffer("vel") } },
      { binding: 4, resource: { buffer: src } }, { binding: 5, resource: { buffer: dir } }, { binding: 6, resource: { buffer: this.state } }] });
    this.device.queue.writeBuffer(this.state, 4 * DETAIL_ST.DTL, new Uint32Array([fine.dir.dtl]));
    if (this.forced.length) this.device.queue.writeBuffer(this.state, 4 * this.forcedAt, new Uint32Array(this.forced));
  }

  encode(encoder: GPUCommandEncoder) {
    if (!this.group) throw new Error("NB-FLIP detail allocator: encoded before bind");
    const pass = encoder.beginComputePass({ label: "nb-brick.detailAlloc" });
    pass.setBindGroup(0, this.group);
    for (const [pipeline, groups] of this.pipelines) { pass.setPipeline(pipeline); pass.dispatchWorkgroups(...groups); }
    pass.end();
  }

  /** Encode and submit alone. */
  submit() {
    const enc = this.device.createCommandEncoder({ label: "nb-brick.detailAlloc" });
    this.encode(enc); this.device.queue.submit([enc.finish()]);
  }

  destroy() { this.state.destroy(); }
}

/** What a detail engine shares with its parent's build. */
export interface BrickDetailBuild {
  readonly stages: BrickStagesFactory;
  readonly solver: PressureSolverFactory;
  readonly values: Readonly<Record<string, unknown>>;
}

/** The detail engine: the fine brick engine, its allocator on the parent, and its presentation. */
export class BrickDetail {
  private constructor(readonly spec: BrickDetailSpec, readonly engine: BrickEngine, readonly alloc: BrickDetailAlloc, readonly presentation: BrickPresentation, readonly surface: BrickDetailSurface) {}

  static async create(device: GPUDevice, parent: BrickEngine, spec: BrickDetailSpec, build: BrickDetailBuild): Promise<BrickDetail> {
    const s = spec.setup;
    if (Math.abs(s.dt_s - parent.options.dt_s) > 1e-12) throw new Error(`NB-FLIP detail engine: its step ${s.dt_s} s is not the parent's ${parent.options.dt_s} s`);
    const alloc = await BrickDetailAlloc.create(device, parent, spec);
    const engine = await BrickEngine.create(device, {
      lattice: s.lattice, ...(s.box ? { box: s.box } : {}), ...(s.openTop ? { openTop: true } : {}), ...(s.openSides ? { openSides: true } : {}),
      farPhi: s.farPhi, h_m: s.h_m, dt_s: s.dt_s, gravity_mps2: s.gravity_mps2, capacities: s.capacities, liquid: s.liquid, initial: s.initial, initialVelocity: s.initialVelocity,
      // No volume budget of its own: the fine engine's liquid is the parent's, held to it by the relaxation.
      solver: build.solver, stages: build.stages, values: build.values, label: s.sceneId,
      fluid: { density_kg_m3: s.density_kg_m3, viscosity_Pa_s: s.viscosity_Pa_s, surfaceTension_N_m: s.surfaceTension_N_m },
      particles: { bandDepth: s.particleBandDepth },
      ...(s.staticMask ? { staticMask: s.staticMask } : {}), ...(s.staticsDeferred ? { staticsDeferred: true } : {}),
      detail: spec.layout,
      primeDetail: (buffer, layout) => { alloc.bind(buffer("src"), buffer("dir"), layout); alloc.submit(); },
    });
    const presentation = await BrickPresentation.create(engine, spec.farPhi_m, s.presentCells);
    const cells = (s.presentCells ?? s.lattice) as unknown as [number, number, number], d = spec.layout;
    const surface = await BrickDetailSurface.create(device, presentation, {
      cells, bricks: s.lattice.map(n => n >> 3) as unknown as [number, number, number], h_m: s.h_m, origin_m: spec.low_m, far_m: spec.farPhi_m,
      mask: { tiles: d.tiles, unit: 4 * d.factor, word: engine.layout.dir.dtl, budget: d.budget },
    }, engine.buffer("dir"));
    return new BrickDetail(spec, engine, alloc, presentation, surface);
  }

  /** The step after the parent's: the mask and the parent's new state, then the fine step (one queue, in order). */
  advance() { this.alloc.submit(); this.engine.advance(1); }

  /** The live voxel edit's preflight on this engine (its own mask, the scene's patches refined): throws, changing nothing. */
  validateSolids(scene: SceneDescription) {
    const mask = this.spec.setup.staticMask;
    if (!mask) throw new Error("the detail engine was built with no static mask");
    const trial = mask.clone();
    try { trial.update(solidWorldForFluidLattice(brickDetailSubScene(scene, this.spec.plan.factor))); }
    catch (error) { throw new Error(`This voxel edit needs more of the detail engine's solid mask than it was built with (${error instanceof Error ? error.message : String(error)})`); }
    const cut = brickStaticCutBricks(trial), k = this.engine.layout.cap.k;
    if (cut > k) throw new Error(`This voxel edit cuts ${cut} of the detail engine's bricks, over its ${k} K slots`);
  }

  /** Fold the live voxel edit (validateSolids first) into this engine's mask and restamp what moved. */
  foldSolids(scene: SceneDescription) {
    const mask = this.spec.setup.staticMask!;
    const lo: [number, number, number] = [Infinity, Infinity, Infinity], hi: [number, number, number] = [-Infinity, -Infinity, -Infinity];
    const ranges = mask.update(solidWorldForFluidLattice(brickDetailSubScene(scene, this.spec.plan.factor)), (x, y, z) => {
      lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); lo[2] = Math.min(lo[2], z);
      hi[0] = Math.max(hi[0], x + 1); hi[1] = Math.max(hi[1], y + 1); hi[2] = Math.max(hi[2], z + 1);
    });
    if (ranges && hi[0] > lo[0]) this.engine.restampStatics(mask.words, ranges, lo, hi);
  }

  /** The presentation's copy, then its mesh. */
  present(encoder: GPUCommandEncoder) { this.presentation.encode(encoder); this.surface.encode(encoder); }

  destroy() { this.surface.destroy(); this.presentation.destroy(); this.engine.destroy(); this.alloc.destroy(); }
}
