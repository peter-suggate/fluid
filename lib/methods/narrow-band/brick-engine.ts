/**
 * The brick engine's host (lane E, M1). This file covers:
 * - the pools and buffers, allocated from the scene's capacities;
 * - the boot;
 * - the recorded schedule (record(): every ring slot's dispatch list, with
 *   bind groups and cfg uniforms built once);
 * - the param ring (one writeBuffer a step);
 * - the two-part submit with the next step encoded ahead;
 * - lagged receipts, which raise a latched FAIL without a fence.
 *
 * Memory and work follow the fluid:
 * - Every buffer is sized from the capacities (scene.fluid.brickPools), apart
 *   from the directory's page window (one word a 32^3 page of the lattice).
 * - Every dispatch is either indirect from a list or sized from a capacity.
 * - Nothing is sized from the lattice.
 */
import { brickLowerOn, brickModuleCode } from "./brick-lower";
import { brickDropDue, brickSourceRing, type BrickSources } from "./brick-source";
import { brickSourceWords } from "./brick-source.wgsl";
import { brickSolidStampKernels, type BrickStaticMaskBinding } from "./brick-solid-stamp.wgsl";
import { ARGS, BODY_FLAG, BOX_TOKEN, BrickLayout, boxPadded, FAIL, H, LIST, NONE, RING, RING_FLAG, RING_SLOTS, RING_WORDS, bricksPerAxis, brickKey, KEY_PAGE, PHI_AIR, REACH_MARGIN, phiFar, type BrickCapacities } from "./brick-abi";
import { bindGroupLayoutEntries, type BrickDispatch, type BrickKernel, type BrickPlacement, ENGINE_BUFFERS } from "./brick-kernel";
import { type PressureSolver, type PressureSolverFactory, validatePressureSolver } from "./brick-pressure-slot";
import { brickSetKernels, churnGroups } from "./brick-set.wgsl";
import { pressureOperatorKernels } from "./pressure-operator.wgsl";
import { brickCensusBuffers, brickCensusKernels, brickCensusSchedule, brickTrigCensusSchedule, censusListArgs } from "./brick-census.wgsl";
import { brickResidencyKernels } from "./brick-residency.wgsl";
import { brickTransferSchedule } from "./brick-transfer.wgsl";
import { brickRegionHeatSchedule } from "./brick-region-heat.wgsl";
import { mgLayout, pressureKickScaleGroups, pressureKickScaleKernel } from "./pressure-mgpcg";
import { type BrickStages, type BrickStagesFactory, validateBrickStages } from "./brick-stages";
import { levelSetReleaseSchedule } from "./levelset-release.wgsl";
import { brickBodyKernels } from "./bodies-stage.wgsl";
import type { BrickBodiesBinding, BrickBodiesHeader } from "./bodies-binding.wgsl";
import { GPU_RIGID_BODY_CAPACITY } from "../../core/rigid-capacity";

export type Vec3 = readonly [number, number, number];
/** A box of initial liquid, in cells (vertex coordinates of the lattice). */
export interface BrickLiquidBox { readonly lo: Vec3; readonly hi: Vec3 }
/**
 * The initial liquid as a page table of vertex φ (brick-initial.ts brickInitialPhi: the old engine's own t = 0 φ,
 * for any liquid description): `pages` the page keys p.x + P.x (p.y + P.y p.z) (P = lattice / 32 + 1 an axis),
 * ascending, and `phi` 32^3 vertex values (in h, x fastest) a page in that order. A vertex of a page the table
 * lacks is far air (phiFar), as the old engine's unstored pages are.
 */
export interface BrickInitialPhi { readonly pages: readonly number[]; readonly phi: Float32Array }
/** The page grid of BrickInitialPhi: pages an axis, the high vertex plane included. */
export const initialPhiPages = (lattice: Vec3): Vec3 => [Math.floor(lattice[0] / 32) + 1, Math.floor(lattice[1] / 32) + 1, Math.floor(lattice[2] / 32) + 1];

/** The owner readers' walls at the storage edge (brickOwnerWGSL without its box): refused on a padded lattice. */
const STORAGE_EDGE_WALLS = [/fn \w*OnWall\(c:vec3i,a:u32\)->bool\{return c\[a\]<=0\|\|c\[a\]>=NB_LATTICE\[a\];\}/, /fn \w*FaceInLattice\(c:vec3i,a:u32\)->bool\{var hi=NB_LATTICE;/];

/** The kick's re-solves from its own pressure: the old arena's UNIFORM_PRESSURE_KICK_RESOLVES (uniform-pressure-arena.ts
 * 138; each re-solve cuts a resting pool's seam error about 3.3x). */
export const BRICK_KICK_RESOLVES = 23;

export interface BrickEngineOptions {
  readonly lattice: Vec3;
  /** The container's cells (the box rule's walls) when the lattice is padded past them to whole pages
   * (BrickLayoutOptions.box). Absent: the lattice. */
  readonly box?: Vec3;
  /** NB_PHI_FAR in h when it is not the lattice's largest extent (BrickLayoutOptions.farPhi). */
  readonly farPhi?: number;
  readonly h_m: number;
  readonly dt_s: number;
  readonly gravity_mps2: Vec3;
  readonly capacities: BrickCapacities;
  readonly apic?: boolean;
  /** The initial liquid in closed form (boxes; their union) ... */
  readonly liquid: readonly BrickLiquidBox[];
  /** ... or, when given, as a page table (`liquid` then empty): any liquid description (brick-initial.ts). */
  readonly initial?: BrickInitialPhi;
  /** The liquid's initial velocity (m/s) on the faces of `tiles` (4h tile keys t.x + T.x (t.y + T.y t.z), T = lattice / 4,
   * ascending: the tiles holding liquid or touching one; brick-initial.ts brickInitialVelocity). Absent: at rest. */
  readonly initialVelocity?: { readonly velocity: Vec3; readonly tiles: readonly number[] };
  /** The scene's scheduled drops in source form (brick-source.ts, lane F's stages read them); the boot's set holds the
   * first step's drop box. Absent: no sources. */
  readonly sources?: BrickSources;
  /** The static voxel mask (brick-static.wgsl.ts brickStaticMask: its words and walls), bound as `solid` and stamped
   * by solidStamp (M3). Absent: the box rule alone, no binding, no stamp. */
  readonly staticMask?: BrickStaticMaskBinding;
  /** The diagnostic lowering of dynamic vector-component writes (brick-lower.ts), off by default: true lowers and
   * checks every module before its shader is created. Absent: NB_BRICK_LOWER=1 in the environment turns it on. */
  readonly lowerDynamicWrites?: boolean;
  /** Lane P's rigid bodies (M5, A2-A6): setMark's body criterion, bodyStamp at the stamp stage, the ring's body header,
   * and the per-step host sync. Absent: no body kernel or binding (the dam's schedule and text). */
  readonly bodies?: BrickBodiesBinding;
  /** The pressure solver at the slot (brick-abi.md §6.1). */
  readonly solver: PressureSolverFactory;
  /** F's stages (brick-abi.md §6.2; brick-stages.ts). */
  readonly stages: BrickStagesFactory;
  /** The method's resolved values and the fluid's constants, passed to the stage factory. */
  readonly values?: Readonly<Record<string, unknown>>;
  readonly fluid?: { readonly density_kg_m3: number; readonly viscosity_Pa_s: number; readonly surfaceTension_N_m: number };
  /** The scene's particle settings for the stage factory (brick-stages.ts BrickStageContext.particles). */
  readonly particles?: { readonly bandDepth: number };
  readonly label?: string;
}

export interface BrickStageSchedule { readonly part1: readonly BrickPlacement[]; readonly part2: readonly BrickPlacement[]; readonly boot: readonly BrickPlacement[] }

const RECEIPTS = 4;
const FAIL_NAMES = Object.entries(FAIL);
export const describeFail = (bits: number, info: number) =>
  `Brick engine failure ${FAIL_NAMES.filter(([, b]) => (bits & b) !== 0).map(([n]) => n).join("|") || `0x${bits.toString(16)}`} (info 0x${(info >>> 0).toString(16)})`;

/** The signed distance (h) of a box at vertex p, as bootPhi computes it. */
export function boxDistance(p: Vec3, b: BrickLiquidBox): number {
  const q = [0, 1, 2].map(a => Math.abs(p[a]! - 0.5 * (b.lo[a]! + b.hi[a]!)) - 0.5 * (b.hi[a]! - b.lo[a]!));
  return Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(q[0]!, q[1]!, q[2]!), 0);
}

/**
 * The reach rule (brick-abi.md §5.4) on the CPU, for the boot and the oracle.
 * `liquid(b)` gives a brick's liquid-vertex box in local coordinates (or
 * undefined), `bound(b)` its velocity bound (m/s), and `particles(b)` whether
 * its bin is non-empty. Returns the keys of every brick to keep or enter.
 */
export function reachSet(lattice: Vec3, bricks: Iterable<Vec3>, liquid: (b: Vec3) => { lo: Vec3; hi: Vec3 } | undefined,
  bound: (b: Vec3) => number, particles: (b: Vec3) => boolean, dt: number, h: number, gravity: Vec3): Set<number> {
  const out = new Set<number>(), nb = bricksPerAxis(lattice), g = Math.hypot(...gravity);
  for (const b of bricks) {
    const parts = particles(b), box = liquid(b);
    if (!parts && !box) continue;
    const lo = parts ? [0, 0, 0] : box!.lo, hi = parts ? [7, 7, 7] : box!.hi;
    const r = Math.fround(Math.fround((bound(b) + g * dt) * dt) / h) + REACH_MARGIN;
    const b0 = [0, 1, 2].map(a => Math.max(Math.floor((8 * b[a]! + lo[a]! - r) / 8), 0));
    const b1 = [0, 1, 2].map(a => Math.min(Math.floor((8 * b[a]! + hi[a]! + r) / 8), nb[a]! - 1));
    for (let z = b0[2]!; z <= b1[2]!; z++) for (let y = b0[1]!; y <= b1[1]!; y++) for (let x = b0[0]!; x <= b1[0]!; x++) out.add(brickKey(x, y, z));
  }
  return out;
}

/** The boot's vertex φ (in h) as bootPhi reads it: the table's (far air off it) or the boxes' union. */
export function bootPhiAt(o: Pick<BrickEngineOptions, "lattice" | "farPhi" | "liquid" | "initial">): (v: Vec3) => number {
  const far = o.farPhi ?? phiFar(o.lattice);
  if (o.initial) {
    const P = initialPhiPages(o.lattice), index = new Map(o.initial.pages.map((k, i) => [k, i])), phi = o.initial.phi;
    return v => {
      if (v.some((x, a) => x < 0 || x > o.lattice[a]!)) return far;
      const i = index.get((v[0] >> 5) + P[0] * ((v[1] >> 5) + P[1] * (v[2] >> 5)));
      return i === undefined ? far : phi[32768 * i + (v[0] & 31) + 32 * ((v[1] & 31) + 32 * (v[2] & 31))]!;
    };
  }
  const f32 = (b: BrickLiquidBox): BrickLiquidBox => ({ lo: o.lattice.map((_, a) => Math.fround(b.lo[a]!)) as unknown as Vec3, hi: o.lattice.map((_, a) => Math.fround(b.hi[a]!)) as unknown as Vec3 });
  const boxes = o.liquid.map(f32);
  return v => Math.min(far, ...boxes.map(box => boxDistance(v, box)));
}

/** The initial bricks: the reach set of the initial liquid's vertices (boxes or table), at rest. */
export function bootBricks(o: Pick<BrickEngineOptions, "lattice" | "farPhi" | "liquid" | "initial" | "dt_s" | "h_m" | "gravity_mps2" | "sources">): number[] {
  if (o.initial && o.liquid.length) throw new Error("Brick engine: the initial liquid is boxes or a table, not both");
  const nb = bricksPerAxis(o.lattice), f32 = (b: BrickLiquidBox): BrickLiquidBox => ({ lo: o.lattice.map((_, a) => Math.fround(b.lo[a]!)) as unknown as Vec3, hi: o.lattice.map((_, a) => Math.fround(b.hi[a]!)) as unknown as Vec3 });
  const boxes = o.liquid.map(f32), candidates = new Map<number, Vec3>(), at = bootPhiAt(o);
  if (o.initial) {
    // Every brick of the table's pages (a page holds 4^3 bricks); off the table no vertex is liquid.
    const P = initialPhiPages(o.lattice);
    for (const k of o.initial.pages) {
      const p = [k % P[0], Math.floor(k / P[0]) % P[1], Math.floor(k / (P[0] * P[1]))];
      for (let j = 0; j < 64; j++) {
        const b = [4 * p[0]! + (j & 3), 4 * p[1]! + ((j >> 2) & 3), 4 * p[2]! + (j >> 4)];
        if (b.every((v, a) => v < nb[a]!)) candidates.set(brickKey(b[0]!, b[1]!, b[2]!), b as unknown as Vec3);
      }
    }
  }
  for (const box of boxes) {
    const lo = [0, 1, 2].map(a => Math.max(Math.floor(box.lo[a]! / 8) - 1, 0)), hi = [0, 1, 2].map(a => Math.min(Math.floor(box.hi[a]! / 8) + 1, nb[a]! - 1));
    for (let z = lo[2]!; z <= hi[2]!; z++) for (let y = lo[1]!; y <= hi[1]!; y++) for (let x = lo[0]!; x <= hi[0]!; x++) candidates.set(brickKey(x, y, z), [x, y, z]);
  }
  const liquid = (b: Vec3) => {
    let lo = [8, 8, 8], hi = [-1, -1, -1];
    for (let l = 0; l < 512; l++) {
      const c = [l & 7, (l >> 3) & 7, l >> 6], v = [0, 1, 2].map(a => 8 * b[a]! + c[a]!) as unknown as Vec3;
      if (v.some((x, a) => x > o.lattice[a]!)) continue;
      const d = Math.min(PHI_AIR, at(v));
      if (Math.fround(d) < 0) { lo = lo.map((m, a) => Math.min(m, c[a]!)); hi = hi.map((m, a) => Math.max(m, c[a]!)); }
    }
    return hi[0]! < 0 ? undefined : { lo: lo as unknown as Vec3, hi: hi as unknown as Vec3 };
  };
  const set = new Set(reachSet(o.lattice, candidates.values(), liquid, () => 0, () => false, o.dt_s, o.h_m, o.gravity_mps2));
  // The first step's drop (brick-source.ts): its box's bricks are resident before that step's census, as the step
  // before a later drop's head claims them (resSource; the old kick's commit stored the first drop's pages).
  const first = o.sources ? brickDropDue(o.sources, 0, o.dt_s) : -1;
  if (first >= 0) {
    const [lo, hi] = o.sources!.drops[first]!.box;
    for (let z = lo[2] >> 3; z <= (hi[2] - 1) >> 3; z++) for (let y = lo[1] >> 3; y <= (hi[1] - 1) >> 3; y++) for (let x = lo[0] >> 3; x <= (hi[0] - 1) >> 3; x++) set.add(brickKey(x, y, z));
  }
  return [...set].sort((a, b) => a - b);
}

/**
 * Tiered: the page-only candidates (KEY_PAGE) the boot claims besides its bricks' pages: the pages that hold those
 * pages' high corner planes. The old engine's page block stores its 33^3 vertices (uniform-page-initialise.ts: a vertex
 * field's 33^3 texels, high planes included), so a boot page's high-plane vertices hold the analytic φ from t = 0
 * (initialLiquidVertexPhi). Here C vertex T is stored in page T >> 3, so the plane x = 32(P+1) is page P + x's: without
 * it the vertex reads NB_PHI_FAR until that page enters, and enters as AIR, where the old engine kept the distance.
 */
export function bootCornerPages(lattice: Vec3, bricks: readonly number[]): number[] {
  const page = (k: number) => [k & 1023, (k >>> 10) & 1023, (k >>> 20) & 1023].map(v => v >> 2);
  const own = new Set(bricks.map(k => brickKey(...(page(k) as [number, number, number]))));
  const extra = new Set<number>();
  for (const pk of own) for (let j = 1; j < 8; j++) {
    const P = [pk & 1023, (pk >>> 10) & 1023, (pk >>> 20) & 1023].map((v, a) => v + ((j >> a) & 1));
    if (P.some((v, a) => 32 * v >= lattice[a]!) || own.has(brickKey(P[0]!, P[1]!, P[2]!))) continue;
    extra.add((brickKey(4 * P[0]!, 4 * P[1]!, 4 * P[2]!) | KEY_PAGE) >>> 0);
  }
  return [...extra].sort((a, b) => a - b);
}

/** A saved engine state (BrickEngine.save): every buffer's words and the step clock. */
export interface BrickEngineState { readonly steps: number; readonly time: number; readonly buffers: ReadonlyMap<string, Uint32Array<ArrayBuffer>> }

interface Recorded { readonly pipeline: GPUComputePipeline; readonly group: GPUBindGroup; readonly dispatch: BrickDispatch }

/**
 * Every upload of the engine's words, as two writes: the 16-byte multiple, then the tail. Dawn (dawn.node, Metal)
 * stages a large write (over its 4 MB ring) in a dedicated buffer whose reservation is the size rounded up, checks it
 * against the write's size, and aborts the process on a mismatch (BufferBase::UploadData). pmgF's 34607172 bytes did
 * in restore. So did boot tables (2026-11-17): a ball over the letters' pool (23986916 bytes), cm12-figure-9 (7471340)
 * and coarse-first-pool-impact (11927924). A lone write of 4 MB + 4 bytes aborts, 1 MB + 4 and 4 MB do not.
 */
export function uploadWords(queue: GPUQueue, buffer: GPUBuffer, offsetBytes: number, data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>) {
  const body = data.length & ~3;
  if (body > 0) queue.writeBuffer(buffer, offsetBytes, data, 0, body);
  if (body < data.length) queue.writeBuffer(buffer, offsetBytes + 4 * body, data, body, data.length - body);
}

export class BrickEngine {
  readonly layout: BrickLayout;
  readonly solver: PressureSolver;
  readonly stages: BrickStages;
  readonly schedule: BrickStageSchedule;
  /** Pipeline compile times (ms), by kernel label, from this engine's creation. */
  readonly compileMs: Readonly<Record<string, number>>;
  private readonly buffers = new Map<string, GPUBuffer>();
  /** Buffers the engine binds but does not own (lane P's `bodies` and `bodyCuts`, the rigid system's): never saved,
   * restored or destroyed here. */
  private readonly external = new Map<string, GPUBuffer>();
  /** The ring's body header of the step being written (A3), from BrickBodiesBinding.sync. */
  private bodyHeader?: BrickBodiesHeader;
  private readonly cfgs = new Map<string, GPUBuffer>();
  private readonly recorded: { part1: Recorded[]; part2: Recorded[] }[] = [];
  private readonly receipts: { buffer: GPUBuffer; busy: boolean }[] = [];
  private ahead?: { step: number; part1: GPUCommandBuffer; part2: GPUCommandBuffer; receipt: number };
  private steps = 0;
  private time = 0;
  /** The newest receipt (the set buffer's control header) and the step it closed. */
  receipt?: { step: number; header: Uint32Array };
  failure?: string;

  private constructor(readonly device: GPUDevice, readonly options: BrickEngineOptions, layout: BrickLayout, solver: PressureSolver, stages: BrickStages,
    schedule: BrickStageSchedule, private readonly pipelines: Map<BrickKernel, GPUComputePipeline>, compileMs: Record<string, number>) {
    this.layout = layout; this.solver = solver; this.stages = stages; this.schedule = schedule; this.compileMs = compileMs;
  }

  /** Every placement of one step and the boot, in order (record()'s input). */
  static scheduleFor(layout: BrickLayout, solver: PressureSolver, stages: BrickStages, boxes: number, mask?: Pick<BrickStaticMaskBinding, "walls">, bodies?: Pick<BrickBodiesBinding, "frame_m" | "h_m" | "openTop">, sources = false): BrickStageSchedule {
    const tiered = stages.tiered === true;
    // P's absent-page guard (FAIL.TIER) stays on in both engines: all-h, an absent page is air only because residency
    // keeps it 8 cells from liquid, so a liquid row with a corner there is a residency defect, never air to read.
    const frame = bodies ? { frame_m: bodies.frame_m, h_m: bodies.h_m } : undefined;
    const set = brickSetKernels(layout, { tiered, activity: stages.activity, ...(frame ? { bodies: frame } : {}) }), pres = pressureOperatorKernels(layout, { faces: "pmgF" in solver.buffers }), f = layout.cap.f, c = layout.cap.c;
    const list = (k: number, g: "G1" | "G8" | "G64"): BrickDispatch => ({ indirect: ARGS.PER_LIST * k + ARGS[g] });
    const direct = (n: number): BrickDispatch => ({ direct: [n, 1, 1] });
    // cfg.z = 1 at boot: the boot's bricks enter h-owned (bootPhi writes F); tiered steps enter 4h (§2.5).
    const update = (z: number): BrickPlacement[] => [
      { kernel: set.setClaimPages, dispatch: direct(churnGroups(f, 64)) },
      { kernel: set.setClaimBricks, dispatch: direct(churnGroups(f, 64)), z },
    ];
    const tail: BrickPlacement[] = [
      { kernel: set.setRecords, dispatch: direct(churnGroups(f, 1)) },
      { kernel: set.setInit, dispatch: direct(churnGroups(f, 1)) },
    ];
    // setLists' cfg.z (brick-abi.md §5.5, ACT): 1 at the head (F_ACT adds the entering bricks, then NOW moves to
    // PREV), 2 at the boot (every live brick is NOW), 0 in part 2.
    const lists = (z: number): BrickPlacement[] => [
      { kernel: set.setLists, dispatch: direct(Math.ceil(Math.max(f, c) / 64)), z },
      { kernel: set.setArgs, dispatch: direct(1) },
    ];
    const solve: BrickPlacement[] = [
      ...(solver.before ?? []),
      { kernel: pres.operator, dispatch: list(LIST.F_ALL, "G1") },
      ...solver.setup,
      ...Array.from({ length: solver.slots }, (_, k) => solver.slot(k)).flat(),
      ...solver.finish,
      // project on F_ALL (lane P's contract, 2026-11-15: WARM and the WALLS bits on every F brick; P reads no F_LIQ).
      { kernel: pres.project, dispatch: list(LIST.F_ALL, "G1") },
    ];
    // The release port's producers (levelset-release.wgsl.ts), where a stage set advects φ: the tile reach words at
    // the old frame plan's point (part 1, after the head, before move), the wall reach before the surface advect.
    const release = stages.levelSet.length ? levelSetReleaseSchedule(layout) : { part1: [], part2: [] };
    // Part 2: the class words and the residency check (setMark), C_LIQ (setMarkPages), the lists rebuilt from them
    // (setArgs with cfg.z = 1 keeps the head's churn counts), then F's level set, velocity, the solve and F's tail.
    const part2: BrickPlacement[] = [
      { kernel: set.setMark, dispatch: list(LIST.F_ALL, "G1") },
      ...(tiered ? [{ kernel: set.setMarkPages, dispatch: list(LIST.C_ALL, "G1") }] : []),
      { kernel: set.setRecords, dispatch: direct(churnGroups(f, 1)) },
      { kernel: set.setLists, dispatch: direct(Math.ceil(Math.max(f, c) / 64)) },
      { kernel: set.setArgs, dispatch: direct(1), z: 1 },
      ...release.part2,
      ...stages.levelSet,
      ...stages.velocity,
      ...solve,
      ...stages.tail,
      { kernel: set.stepEnd, dispatch: direct(1), z: 0 },
    ];
    // Part 1's head (brick-abi.md §2.5, §5.4), on the completed step's lists and state, in both engines:
    // - NB-FLIP's frame-head census (all-h too: F's TRIG, and WANT seeds residency);
    // - the residency (brick-residency.wgsl.ts: seep S2j's local per-axis reach of each source brick, ACTIVE tiles or
    //   particle support): resMark's keeps and claims (pass 0 pages and the bricks of resident pages, setClaimPages,
    //   pass 1 the bricks of the pages it entered), setClaimBricks, then setLeave and setLeavePages retire what no
    //   source reached (no hysteresis);
    // - records, init and lists for the new set;
    // - tiered: censusWant checks every WANT ring (a refused pool latches FAIL.TIER) and the transfer commits;
    // - the churn lists clear for part 2 (stepEnd with cfg.z = 1: no step).
    // The boot runs the census before F's boot stages (the seeding reads TRIG) and commits without the remap:
    // bootPhi writes one analytic φ to F and C, and U = 0. Its set is the CPU reach set (bootBricks).
    const g1 = (k: number) => ({ indirect: ARGS.PER_LIST * k + ARGS.G1 });
    // A bring-up stage set reads no TRIG: on a lattice of partial pages (P's synthetic blocks) it runs without the
    // census, which needs whole pages (every scene's fluid lattice is, sceneFluidBox) and refuses others. With no
    // census there are no residency records, so such a set stays the boot's (stated: no scene runs one).
    const censused = layout.lattice.every(n => n % 32 === 0) || tiered || !stages.bringUp;
    const census = (keep: boolean) => censused ? brickCensusSchedule(layout, g1, LIST.C_ALL, LIST.F_ALL, tiered, keep) : [];
    // The real D (brick-region-heat.wgsl.ts), tiered only: the heat and spread after censusBegin (the head's deep runs
    // in censusCoarse), the WARM promotion in setTransferList (cfg.w = 1), the certificate after the transfer. Its
    // pressureDelta is lane P's PDELTA where the solver publishes it (pmgC), else +Inf.
    const pdelta = "pmgC" in solver.buffers ? mgLayout(layout).PDELTA : undefined;
    const heat = tiered && censused ? brickRegionHeatSchedule(layout, { pdelta }) : undefined;
    // All-h (no heat, D = 0): the certificate alone, for the census's need rule (tuDeformation reads its packed copy,
    // CENSUS.DEF): certFine on every h tile, certError with cfg.w = 1 (the copy, no region reductions or deep pass),
    // where the tiered head runs the transfer and its certificate.
    const certificate: BrickPlacement[] = !tiered && censused ? (() => { const k = brickRegionHeatSchedule(layout).kernels; return [
      { kernel: k.certFine, dispatch: { indirect: ARGS.PER_LIST * LIST.F_ALL + ARGS.X8 } },
      { kernel: k.certError, dispatch: { indirect: ARGS.PER_LIST * LIST.C_ALL + ARGS.X8 }, w: 1 },
    ]; })() : [];
    const headCensus = (): BrickPlacement[] => { const c = censused ? brickCensusSchedule(layout, g1, LIST.C_ALL, LIST.F_ALL, tiered, true, heat !== undefined) : []; return heat ? [c[0]!, ...heat.head, ...c.slice(1)] : c; };
    const transfer = brickTransferSchedule(layout, g1, LIST.C_ALL).map(p => (heat && p.kernel.label === "setTransferList" ? { ...p, w: 1 } : p));
    const want: BrickPlacement[] = tiered ? [{ kernel: brickCensusKernels(layout).censusWant, dispatch: censusListArgs(LIST.C_ALL, "X8") }] : [];
    const churnClear: BrickPlacement = { kernel: set.stepEnd, dispatch: direct(1), z: 1 };
    const bootPhi: BrickPlacement = { kernel: set.bootPhi, dispatch: direct(churnGroups(f, 1)), z: Math.max(boxes, 0), w: boxes < 0 ? 1 : 0 };
    const res = brickResidencyKernels(layout, { tiered });
    const residency: BrickPlacement[] = [
      // Tiered, pass 0 also runs the page sources: a workgroup a page of C_ALL after F_ALL's bricks (FC_G1).
      { kernel: res.resMark, dispatch: tiered ? { indirect: ARGS.FC_G1 } : g1(LIST.F_ALL), z: 0 },
      ...(sources ? [{ kernel: res.resSource, dispatch: direct(8), z: 0 }] : []),
      { kernel: set.setClaimPages, dispatch: direct(churnGroups(f, 64)) },
      { kernel: res.resMark, dispatch: g1(LIST.F_ALL), z: 1 },
      ...(sources ? [{ kernel: res.resSource, dispatch: direct(8), z: 1 }] : []),
      { kernel: set.setClaimBricks, dispatch: direct(churnGroups(f, 64)), z: 0 },
      { kernel: set.setLeave, dispatch: list(LIST.F_ALL, "G64") },
      { kernel: set.setLeavePages, dispatch: list(LIST.C_ALL, "G64") },
    ];
    // The stamp stage (A6): E's solidStamp (M3), then the stage set's (P's bodyStamp, M5), then the K slots go back
    // (A5: after every pop of the step). With no K pool and no stamp it is empty: the dam's schedule is unchanged.
    // P's bodyStamp (A2, A6): the entering bricks a body reaches (cfg.z = 1, after solidStamp), then F_BODY (cfg.z = 0).
    const body = bodies ? brickBodyKernels(layout, frame!, { openTop: mask ? mask.walls.policy === 2 : bodies.openTop, ...(mask ? { mask } : {}) }).bodyStamp : undefined;
    if (bodies && layout.cap.k === 0) throw new Error("Brick engine: rigid bodies need K slots (scene.fluid.brickPools.k)");
    const bodyStamp: BrickPlacement[] = body ? [{ kernel: body, dispatch: list(LIST.F_ENTER, "G1"), z: 1 }, { kernel: body, dispatch: list(LIST.F_BODY, "G1"), z: 0 }] : [];
    const stamp: BrickPlacement[] = [...(mask ? [{ kernel: brickSolidStampKernels(layout, mask).solidStamp, dispatch: list(LIST.F_ENTER, "G1") }] : []), ...bodyStamp, ...(stages.stamp ?? []), ...(layout.cap.k > 0 ? [
      { kernel: set.setKRelease, dispatch: list(LIST.F_ALL, "G64"), z: 0 },
      { kernel: set.setKRelease, dispatch: list(LIST.F_LEAVE, "G64"), z: 1 },
    ] : [])];
    const head = censused ? [...headCensus(), ...residency, ...tail, ...lists(1), ...want, ...(tiered ? [...transfer, ...(heat?.cert ?? [])] : certificate), ...stamp, churnClear] : [];
    const commit = transfer.filter(p => p.kernel.label === "setCommitMask");
    const bootHead = tiered ? [...census(false), ...update(0), ...tail, bootPhi, ...lists(2), ...want, ...commit] : census(false);
    // The first frame's projected half kick (BrickStages.kick, the old host's kickMixedFrame): F's force, the solve,
    // F's after; with F's seed, a TRIG-only census of the kicked state (brickTrigCensusSchedule), then the seed.
    const trigCensus = censused ? brickTrigCensusSchedule(layout, g1, LIST.C_ALL, LIST.F_ALL, tiered) : [];
    // The kick has no last frame, so its seam heads (rowsF's poHeadS: the start's tangential slope) start from the
    // cold prolongation; like the old arena's encodeSolve(kick), it re-solves BRICK_KICK_RESOLVES times from its own
    // pressure (each solve leaves the rows WARM, so the next starts from X), before F's after reads the projection.
    const kickSolve: BrickPlacement[] = Array.from({ length: 1 + BRICK_KICK_RESOLVES }, () => solve).flat();
    // X is in its solve's dt units (U = US - grad X / h; the old arena's pressures are dt-free): the kick solves at
    // dt/2, so its X doubles (pmgKickScale) before step 1's rows start from it.
    const kickScale: BrickPlacement[] = [{ kernel: pressureKickScaleKernel(layout), dispatch: { direct: pressureKickScaleGroups(layout) } }];
    const kick: BrickPlacement[] = stages.kick ? [...stages.kick.force, ...kickSolve, ...kickScale, ...stages.kick.after, ...(stages.kick.seed ? [...trigCensus, ...stages.kick.seed] : [])] : [];
    // The boot's class pass, as part 2's (setMark after F's boot stages, before the kick raises BOUND_F; C_LIQ;
    // setRecords clears the rebuilt counts), then the lists from its classes (cfg.z = 2 ORs NOW onto every live brick
    // and keeps the ACTIVE tiles): step 1's head residency reads them (brick-residency.wgsl.ts), and the kick's lists
    // (F_LIQ) are the boot state's.
    const bootMark: BrickPlacement[] = [...part2.slice(0, part2.findIndex(p => p.kernel === set.setLists)), ...lists(2)];
    const boot: BrickPlacement[] = [...update(1), ...tail, bootPhi, ...lists(2), ...bootHead, ...stamp, ...stages.boot, ...bootMark, ...kick, { kernel: set.stepEnd, dispatch: direct(1), z: 1 }];
    // Part 1: the head (P's bodyStamp, E's solidStamp: M3 and M5; the census, residency and remap), then the release
    // port's reach words (after every head claim, before move), F's move and rebin.
    return { part1: [...head, ...release.part1, ...stages.move, ...stages.rebin], part2, boot };
  }

  static async create(device: GPUDevice, options: BrickEngineOptions): Promise<BrickEngine> {
    const layout = new BrickLayout({ lattice: options.lattice, farPhi: options.farPhi, ...(options.box ? { box: options.box } : {}), ...options.capacities, apic: options.apic });
    const solver = options.solver(layout);
    validatePressureSolver(solver);
    const stages = options.stages({ layout, dt_s: options.dt_s, h_m: options.h_m, gravity_mps2: options.gravity_mps2, values: options.values ?? {}, fluid: options.fluid, particles: options.particles, sources: options.sources });
    validateBrickStages(stages);
    const initial = bootBricks(options);
    if (initial.length > layout.cap.f) throw new Error(`Brick engine: the initial liquid needs ${initial.length} F bricks, over scene.fluid.brickPools.f = ${layout.cap.f}`);
    // boxes < 0 names the page table (bootPhi cfg.w = 1).
    const schedule = BrickEngine.scheduleFor(layout, solver, stages, options.initial ? -1 : options.liquid.length, options.staticMask, options.bodies, options.sources !== undefined);
    // One pipeline a kernel, compiled concurrently; each module once.
    const kernels = [...new Set([...schedule.boot, ...schedule.part1, ...schedule.part2].map(p => p.kernel))];
    // The box hook (M3 padding): every wall token resolved by its builder, and, where the lattice is padded past the
    // box, no owner reader still walling at the storage edge (a wall at the box exactly once).
    const unresolved = kernels.filter(k => k.code.includes(BOX_TOKEN) || k.code.includes("nbInWallBox(")).map(k => k.label);
    if (unresolved.length) throw new Error(`Brick engine: ${unresolved.join(", ")} leave the wall token unresolved (resolveBox)`);
    if (boxPadded(layout)) {
      const edge = kernels.filter(k => STORAGE_EDGE_WALLS.some(re => re.test(k.code))).map(k => k.label);
      if (edge.length) throw new Error(`Brick engine: the lattice is padded past the box ${layout.box.join("x")}, but ${edge.join(", ")} wall at the storage edge (brickOwnerWGSL's box)`);
    }
    // The diagnostic lowering switch (brick-lower.ts): on, every module is lowered and checked here, before any shader
    // is created; off (the default), each is created from its text exactly as written.
    const lower = brickLowerOn(options.lowerDynamicWrites), lowered = new Map(kernels.map(k => [k.code, brickModuleCode(k.label, k.code, lower)]));
    const modules = new Map<string, GPUShaderModule>(), pipelines = new Map<BrickKernel, GPUComputePipeline>(), compileMs: Record<string, number> = {};
    await Promise.all(kernels.map(async k => {
      const t0 = performance.now();
      let module = modules.get(k.code);
      if (!module) {
        const code = lowered.get(k.code)!;
        module = device.createShaderModule({ label: `nb-brick.${k.label}`, code }); modules.set(k.code, module);
        const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
        if (errors.length) {
          const lines = code.split("\n");
          throw new Error(`Brick engine: ${k.label} does not compile:\n${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}\n  ${lines[m.lineNum - 1]?.slice(Math.max(0, m.linePos - 80), m.linePos + 80) ?? ""}`).join("\n")}`);
        }
      }
      const layout_ = device.createPipelineLayout({ bindGroupLayouts: [device.createBindGroupLayout({ entries: bindGroupLayoutEntries(k.bindings) })] });
      pipelines.set(k, await device.createComputePipelineAsync({ label: `nb-brick.${k.label}`, layout: layout_, compute: { module, entryPoint: k.entry } }));
      compileMs[k.label] = performance.now() - t0;
    }));
    const engine = new BrickEngine(device, options, layout, solver, stages, schedule, pipelines, compileMs);
    engine.allocate(stages.tiered === true ? [...initial, ...bootCornerPages(options.lattice, initial)] : initial);
    engine.record();
    engine.boot();
    engine.encodeAhead();
    return engine;
  }

  /** Run placements once, outside the step (tests and tools on a fenced, frozen state); compiles kernels the schedule lacks. */
  async run(placements: readonly BrickPlacement[]) {
    for (const k of new Set(placements.map(p => p.kernel))) {
      if (this.pipelines.has(k)) continue;
      const module = this.device.createShaderModule({ label: `nb-brick.${k.label}`, code: brickModuleCode(k.label, k.code, brickLowerOn(this.options.lowerDynamicWrites)) });
      const errors = (await module.getCompilationInfo()).messages.filter(m => m.type === "error");
      if (errors.length) throw new Error(`Brick engine: ${k.label} does not compile:\n${errors.map(m => `${m.lineNum}:${m.linePos} ${m.message}`).join("\n")}`);
      const layout = this.device.createPipelineLayout({ bindGroupLayouts: [this.device.createBindGroupLayout({ entries: bindGroupLayoutEntries(k.bindings) })] });
      this.pipelines.set(k, await this.device.createComputePipelineAsync({ label: `nb-brick.${k.label}`, layout, compute: { module, entryPoint: k.entry } }));
    }
    await this.fence();
    this.device.queue.submit([this.encode(placements.map(p => this.bind(p, 0)), "nb-brick.run")]);
    await this.fence();
  }

  /** Bytes of every buffer the engine allocated. */
  get bytes(): number { let n = 0; for (const b of this.buffers.values()) n += b.size; return n; }
  get stepsSubmitted(): number { return this.steps; }

  private allocate(initial: readonly number[]) {
    const L = this.layout, d = this.device, U = GPUBufferUsage;
    const make = (name: string, words: number, usage = 0) => {
      const b = d.createBuffer({ label: `nb-brick.${name}`, size: 4 * Math.max(words, 4), usage: U.STORAGE | U.COPY_SRC | U.COPY_DST | usage });
      this.buffers.set(name, b); return b;
    };
    const set = make("set", L.set.words), dir = make("dir", L.dir.words);
    make("args", ARGS.WORDS, U.INDIRECT);
    for (const pool of ["vel", "phi", "pres", "cut"] as const) make(pool, L.poolWords[pool]);
    make("parts", L.partsWords);
    make("xfer", L.xferWords);
    make("census", brickCensusBuffers(L).census);
    const bodies = this.options.bodies;
    if (bodies) { this.external.set("bodies", bodies.bodies); this.external.set("bodyCuts", bodies.cuts); }
    const mask = this.options.staticMask;
    if (mask) uploadWords(d.queue, make("solid", mask.words.length), 0, mask.words as Uint32Array<ArrayBuffer>);
    // The sources' static record (brick-source.ts): every drop's extrusion edges, once; a stub with no sources.
    const edges = this.options.sources?.edges;
    const src = make("src", brickSourceWords(edges ? edges.length / 4 : 0));
    if (edges?.length) uploadWords(d.queue, src, 0, edges);
    for (const [name, words] of [...Object.entries(this.solver.buffers), ...Object.entries(this.stages.buffers)]) {
      if ((ENGINE_BUFFERS as readonly string[]).includes(name) || name === "boot" || name === "bodies" || name === "bodyCuts" || this.buffers.has(name)) throw new Error(`Brick engine: lane buffer ${name} shadows another buffer`);
      make(name, words);
    }
    const t = this.options.initial;
    if (t && t.phi.length !== 32768 * t.pages.length) throw new Error(`Brick engine: the initial φ table holds ${t.phi.length} values for ${t.pages.length} pages`);
    // The liquid (boxes or the table), then the initial velocity block, then its start (NONE: at rest).
    const v = this.options.initialVelocity, head = t ? 1 + t.pages.length + t.phi.length : Math.max(6 * this.options.liquid.length, 4);
    const boxes = new Float32Array(head + (v ? 4 + v.tiles.length : 0) + 1), words = new Uint32Array(boxes.buffer);
    if (t) { words.set([t.pages.length, ...t.pages]); boxes.set(t.phi, 1 + t.pages.length); }
    else this.options.liquid.forEach((b, i) => boxes.set([...b.lo, ...b.hi], 6 * i));
    if (v) { boxes.set(v.velocity, head); words[head + 3] = v.tiles.length; words.set(v.tiles, head + 4); }
    words[boxes.length - 1] = v ? head : NONE;
    uploadWords(d.queue, make("boot", boxes.length), 0, boxes);
    // The set buffer: free stacks full (FREE[i] = CAP - 1 - i), the initial candidates.
    const s = new Uint32Array(L.set.words), cap = L.cap;
    s[H.TOP_F] = cap.f; s[H.TOP_C] = cap.c; s[H.TOP_B] = cap.b; s[H.TOP_K] = cap.k;
    const fill = (at: number, n: number) => { for (let i = 0; i < n; i++) s[at + i] = n - 1 - i; };
    fill(L.set.freeF, cap.f); fill(L.set.freeC, cap.c); fill(L.set.freeB, cap.b); fill(L.set.freeK, cap.k);
    s.set(initial, L.set.cand); s[H.N_CAND] = initial.length;
    uploadWords(d.queue, set, 0, s);
    // The directory: every page, child and record word NONE, the tile masks clear.
    const w = new Uint32Array(L.dir.words).fill(NONE); w.fill(0, L.dir.cmask);
    uploadWords(d.queue, dir, 0, w);
    for (let i = 0; i < RECEIPTS; i++) this.receipts.push({ buffer: d.createBuffer({ label: `nb-brick.receipt${i}`, size: 4 * 256, usage: U.MAP_READ | U.COPY_DST }), busy: false });
  }

  private cfg(x: number, y: number, z: number, w: number): GPUBuffer {
    const key = `${x},${y},${z},${w}`;
    let b = this.cfgs.get(key);
    if (!b) {
      b = this.device.createBuffer({ label: `nb-brick.cfg.${key}`, size: 16, usage: GPUBufferUsage.UNIFORM, mappedAtCreation: true });
      new Uint32Array(b.getMappedRange()).set([x, y, z, w]); b.unmap(); this.cfgs.set(key, b);
    }
    return b;
  }

  private bind(p: BrickPlacement, slot: number): Recorded {
    const pipeline = this.pipelines.get(p.kernel)!;
    const entries = p.kernel.bindings.map((b, i): GPUBindGroupEntry => {
      if (b.name === "cfg") return { binding: i, resource: { buffer: this.cfg(slot, slot & 1, p.z ?? 0, p.w ?? 0) } };
      const buffer = this.buffers.get(b.name) ?? this.external.get(b.name);
      if (!buffer) throw new Error(`Brick engine: ${p.kernel.label} binds unknown buffer ${b.name}`);
      return { binding: i, resource: { buffer } };
    });
    return { pipeline, group: this.device.createBindGroup({ label: `nb-brick.${p.kernel.label}.${slot}`, layout: pipeline.getBindGroupLayout(0), entries }), dispatch: p.dispatch };
  }

  /** record(): each ring slot's two parts as pipelines, bind groups and dispatches, built once. */
  private record() {
    for (let slot = 0; slot < RING_SLOTS; slot++)
      this.recorded.push({ part1: this.schedule.part1.map(p => this.bind(p, slot)), part2: this.schedule.part2.map(p => this.bind(p, slot)) });
  }

  private encode(list: readonly Recorded[], label: string, receipt?: number): GPUCommandBuffer {
    const enc = this.device.createCommandEncoder({ label });
    if (list.length) {
      const pass = enc.beginComputePass({ label });
      const args = this.buffers.get("args")!;
      for (const r of list) {
        pass.setPipeline(r.pipeline); pass.setBindGroup(0, r.group);
        if ("direct" in r.dispatch) pass.dispatchWorkgroups(...r.dispatch.direct); else pass.dispatchWorkgroupsIndirect(args, 4 * r.dispatch.indirect);
      }
      pass.end();
    }
    if (receipt !== undefined) enc.copyBufferToBuffer(this.buffers.get("set")!, 0, this.receipts[receipt]!.buffer, 0, 4 * 256);
    return enc.finish();
  }

  private ringWords(step: number): ArrayBuffer {
    const o = this.options, buf = new ArrayBuffer(4 * RING_WORDS), u = new Uint32Array(buf), f = new Float32Array(buf);
    u[RING.STEP] = step; f[RING.DT] = o.dt_s; f[RING.TIME] = this.time; u[RING.FLAGS] = 0;
    f.set(o.gravity_mps2, RING.GRAVITY); f[RING.H] = o.h_m;
    // A3: the body header; FLAGS.BODIES (E's gate) while any body is present.
    const b = this.bodyHeader;
    if (b) {
      if (!(Number.isInteger(b.count) && b.count >= 0 && b.count <= GPU_RIGID_BODY_CAPACITY)) throw new Error(`Brick engine: ${b.count} rigid bodies (at most ${GPU_RIGID_BODY_CAPACITY}, the rigid capacity)`);
      if ((b.flags & ~(BODY_FLAG.COUPLING | BODY_FLAG.CUTS | BODY_FLAG.INVERTED)) !== 0) throw new Error(`Brick engine: unknown body flags ${b.flags}`);
      if (b.count > 0) u[RING.FLAGS] |= RING_FLAG.BODIES;
      u[RING.BODY_COUNT] = b.count; u[RING.BODY_FLAGS] = b.flags; f[RING.BODY_HORIZON] = o.dt_s; f[RING.BODY_CELL] = o.h_m ** 3; u[RING.BODY_GEN] = b.generation >>> 0;
    }
    // F's source words (brick-source.ts): this step's drop, its added cells, and the drop boxes the head claims.
    if (o.sources) brickSourceRing(o.sources, this.time, u, f);
    return buf;
  }

  private boot() {
    const rec = this.schedule.boot.map(p => this.bind(p, 0));
    this.bodyHeader = this.options.bodies?.sync(0);
    this.device.queue.writeBuffer(this.buffers.get("set")!, 4 * this.layout.set.ring, this.ringWords(0));
    this.device.queue.submit([this.encode(rec, "nb-brick.boot")]);
    // The page table is boot-only (its size follows the initial liquid's pages): released once the boot is queued
    // (destroy waits for the queued work), a 4-word stub in its place for save and restore.
    if (this.options.initial || this.options.initialVelocity) {
      const table = this.buffers.get("boot")!;
      this.buffers.set("boot", this.device.createBuffer({ label: "nb-brick.boot", size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST }));
      table.destroy();
    }
  }

  private encodeAhead() {
    const step = this.steps, rec = this.recorded[step % RING_SLOTS]!;
    const free = this.receipts.findIndex((r, i) => !r.busy && i === step % RECEIPTS);
    const receipt = free < 0 ? undefined : free;
    if (receipt !== undefined) this.receipts[receipt]!.busy = true;
    this.ahead = { step, part1: this.encode(rec.part1, `nb-brick.step${step}.part1`), part2: this.encode(rec.part2, `nb-brick.step${step}.part2`, receipt), receipt: receipt ?? -1 };
  }

  /** Submit `steps` steps: per step one ring write, part 1, part 2, then the next step encoded ahead. */
  advance(steps = 1) {
    for (let i = 0; i < steps; i++) {
      if (this.failure) throw new Error(this.failure);
      const a = this.ahead!, slot = a.step % RING_SLOTS;
      // A4: the host's body sync, queued after the last step's part 2 and before this step's part 1.
      if (this.options.bodies) this.bodyHeader = this.options.bodies.sync(a.step);
      this.device.queue.writeBuffer(this.buffers.get("set")!, 4 * (this.layout.set.ring + RING_WORDS * slot), this.ringWords(a.step));
      this.device.queue.submit([a.part1]);
      this.device.queue.submit([a.part2]);
      if (a.receipt >= 0) this.collect(a.receipt, a.step);
      this.steps++; this.time += this.options.dt_s;
      this.encodeAhead();
    }
  }

  private collect(index: number, step: number) {
    const r = this.receipts[index]!;
    r.buffer.mapAsync(GPUMapMode.READ).then(() => {
      const header = new Uint32Array(r.buffer.getMappedRange().slice(0));
      r.buffer.unmap(); r.busy = false;
      if (!this.receipt || this.receipt.step < step) this.receipt = { step, header };
      if (header[H.FAIL] !== 0 && !this.failure) this.failure = describeFail(header[H.FAIL]!, header[H.FAIL_INFO]!);
    }, () => { r.busy = false; });
  }

  /** Wait for every submitted step (tests and harnesses; never on the step cadence). */
  async fence() { await this.device.queue.onSubmittedWorkDone(); }

  /** Read words of an engine buffer (tests and harnesses). */
  async read(name: string, offsetWords = 0, words?: number): Promise<Uint32Array> {
    const src = this.buffers.get(name);
    if (!src) throw new Error(`Brick engine: no buffer ${name}`);
    const n = words ?? src.size / 4 - offsetWords;
    const staging = this.device.createBuffer({ size: 4 * Math.max(n, 1), usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const enc = this.device.createCommandEncoder(); enc.copyBufferToBuffer(src, 4 * offsetWords, staging, 0, 4 * n);
    this.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0)); staging.unmap(); staging.destroy();
    return out;
  }

  /** An engine buffer, for adapters that bind it outside the recorded step (presentation, observers). */
  buffer(name: string): GPUBuffer {
    const b = this.buffers.get(name);
    if (!b) throw new Error(`Brick engine: no buffer ${name}`);
    return b;
  }

  /** Write words into an engine buffer (tests: perturb a field between steps). */
  write(name: string, offsetWords: number, data: Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>) {
    uploadWords(this.device.queue, this.buffers.get(name)!, 4 * offsetWords, data);
  }

  /** The engine's whole state after a fence: every buffer (fields, set, directory, census and its region heat, the
   * solver's and the stages' buffers) and the step clock. Never on the step cadence (save points, undo, tests). */
  async save(): Promise<BrickEngineState> {
    await this.fence();
    const buffers = new Map<string, Uint32Array<ArrayBuffer>>();
    for (const name of this.buffers.keys()) buffers.set(name, new Uint32Array(await this.read(name)));
    return { steps: this.steps, time: this.time, buffers };
  }

  /** Put a saved state back: every buffer, the clock, and the step encoded ahead for the restored step. A state from
   * another layout fails explicitly (a missing, extra or resized buffer). */
  async restore(state: BrickEngineState) {
    await this.fence();
    if (state.buffers.size !== this.buffers.size) throw new Error(`Brick engine: restore has ${state.buffers.size} buffers, the engine ${this.buffers.size}`);
    for (const [name, b] of this.buffers) {
      const words = state.buffers.get(name);
      if (!words) throw new Error(`Brick engine: restore lacks buffer ${name}`);
      if (4 * words.length !== b.size) throw new Error(`Brick engine: restore buffer ${name} has ${words.length} words, the engine ${b.size / 4}`);
    }
    // A buffer's bytes through uploadWords (Dawn's large-write abort).
    for (const [name, b] of this.buffers) uploadWords(this.device.queue, b, 0, state.buffers.get(name)!);
    await this.fence();
    for (const r of this.receipts) if (r.busy && r.buffer.mapState === "unmapped") r.busy = false;
    this.steps = state.steps; this.time = state.time;
    this.receipt = undefined; this.failure = undefined;
    this.encodeAhead();
  }

  /** Throws the latched failure, if any, after fencing (tests). */
  async assertHealthy() {
    await this.fence();
    const h = await this.read("set", 0, 4);
    if (h[H.FAIL] !== 0) throw new Error(describeFail(h[H.FAIL]!, h[H.FAIL_INFO]!));
  }

  destroy() {
    for (const b of this.buffers.values()) b.destroy();
    for (const b of this.cfgs.values()) b.destroy();
    for (const r of this.receipts) r.buffer.destroy();
  }
}
