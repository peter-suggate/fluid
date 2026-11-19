/**
 * LANE F OWNS THIS FILE. The brick engine's liquid sources (M4): scheduled drops (spheres, disks, letters), host side.
 *
 * The old engine (webgpu-uniform-reference.ts) adds a drop on the one step whose interval (start, end] holds its time
 * (scheduledDrop; more than one due in a step throws), in source form (uniformScheduledDrop: a ball, a disk, or an
 * extrusion behind its bounding ball), and joins its analytic volume to the budget on that step (beginStep's
 * updateBudget). Its pages are asked for on the step before and on its own (requestDropPages,
 * uniformPagePoolDropAhead: the bounding box grown by UNIFORM_PAGE_POOL_PIN_MARGIN = 4 cells), so the census of its
 * own step finds its tiles resident.
 *
 * Here everything is in cells (the lattice's coordinates: cell c spans c .. c + 1, a vertex is an integer point; the
 * cells are cubic, so a distance in h is the old engine's in metres over h):
 * - every drop's extrusion edges (a.xy, b.xy) go once, at build, into F's `src` buffer (BrickStages.init);
 * - the ring's SOURCE words (brick-abi.md §5.2, F's 16-63) hold this step's drop and two cell boxes: this step's drop
 *   and the next step's, each grown by the pin margin, which the head's residency claims (brick-source.wgsl.ts);
 * - the ring's VOLUME word 8 holds the cells this step adds (its drop's analytic volume over h^3, and the nozzle's
 *   flow over the step), which uAdvect adds to the budget after the step's gauge, as beginStep does.
 *
 * The inflow (M4, the old engine's nozzle: inflow-boundary.ts and uniform-source.wgsl.ts, verbatim in cells):
 * - the nozzle's outlet (inflowOutletCenter), radius, velocity (m/s), aperture scale and face (createInflowGridBoundary
 *   and inflowFaceIndex: the face plane clamped into the lattice) go in the ring every step, with the step's strength
 *   (averageInflowStrength over the step, which the tap's `enabled` zeroes), its sweep |u| strength dt / h, whether the
 *   nozzle ran before the step (writeParams' physical.w) and whether the stream needs h (inflowNeedsFine);
 * - its flow (pi r^2 |u| strength dt / h^3) joins the budget each step (beginStep's updateBudget);
 * - its plug box (uniformPageSources: the outlet to its reach over the longest step and the plug overlap, grown by the
 *   radius) is claimed every step, as the old pool pins its pages, and is in the boot's set.
 */
import type { FluidInflow, SceneDescription } from "../../core/model";
import { averageInflowStrength, createInflowGridBoundary, inflowOutletCenter } from "../../core/inflow-boundary";
import { inflowStrength as inflowStrengthAt } from "../../core/initial-fluid";
import { LIQUID_EXTRUSION_MAX_EDGES, liquidExtrusionBounds, liquidExtrusionEdges, liquidExtrusionVolume_m3 } from "../../core/liquid-extrusion";
import type { Vec3 } from "./brick-engine";

/** The ring words of the sources (brick-abi.md §5.2: F's VOLUME 8-15 and SOURCES 16-63). */
export const SRC_RING = {
  /** f32: the cells this step adds (its drop's volume / h^3). */
  ADDED: 8,
  /** u32: this step's drop: 0 none, 1 ball, 2 disk (a cylinder along z), 3 extrusion. */
  KIND: 16,
  /** f32 x3: the drop's centre (cells); an extrusion's bounding ball sits on its mid-plane. */
  CENTRE: 17,
  /** f32: the ball's (or bounding ball's, or disk's) radius (cells). */
  RADIUS: 20,
  /** f32: a disk's half height (cells). */
  HALF_HEIGHT: 21,
  /** u32: an extrusion's first edge in `src` and its edge count. */
  EDGE_START: 22, EDGE_COUNT: 23,
  /** f32: an extrusion's outward offset, half depth and edge radius (cells). */
  OFFSET: 24, HALF_DEPTH: 25, EDGE_RADIUS: 26,
  /** i32 x6 each: a cell box lo (inclusive) xyz, hi (exclusive) xyz: this step's drop's and the next step's. */
  BOX: 27, AHEAD: 33,
  /** u32: bit 0 BOX holds a drop, bit 1 AHEAD does, bit 2 INFLOW_BOX holds the nozzle's plug box. */
  BOXES: 39,
  /** u32: the nozzle this step: bit 0 on (strength > 0), bit 1 it ran before the step (the plug behind the outlet is
   * last step's: physical.w), bit 2 the stream needs h (inflowNeedsFine). */
  INFLOW: 40,
  /** f32: the step's mean strength (averageInflowStrength, 0..1). */
  STRENGTH: 41,
  /** f32 x3: the outlet's centre (cells, inflowOutletCenter). */
  OUTLET: 42,
  /** f32: the nozzle's radius (cells). */
  IN_RADIUS: 45,
  /** f32 x3: the nozzle's velocity at full strength (m/s). */
  VELOCITY: 46,
  /** f32: the aperture scale (createInflowGridBoundary: the disk's projected area over the face's raw aperture). */
  SCALE: 49,
  /** f32: the plug's sweep this step, |u| strength dt / h (cells; the old speed * dimsDt.w over h). */
  SWEEP: 50,
  /** u32: the dominant axis (inflowAxis). i32: the nozzle face (inflowFaceIndex: the face plane is face + 1). */
  AXIS: 51, FACE: 52,
  /** i32 x6: the plug box, lo (inclusive) xyz, hi (exclusive) xyz (cells). */
  INFLOW_BOX: 53,
} as const;
/** The cells the plug reaches past its step's sweep (UNIFORM_SOURCE_PLUG_OVERLAP_CELLS). */
export const SRC_PLUG_OVERLAP = 2;
/** The cells a drop's box is grown by each way (UNIFORM_PAGE_POOL_PIN_MARGIN). */
export const SRC_PIN_MARGIN = 4;

/** A drop in source form, in cells. */
export interface BrickDrop {
  readonly time_s: number;
  readonly kind: 1 | 2 | 3;
  readonly centre: Vec3;
  readonly radius: number;
  readonly halfHeight: number;
  readonly edgeStart: number;
  readonly edgeCount: number;
  readonly offset: number;
  readonly halfDepth: number;
  readonly edgeRadius: number;
  /** The volume the source asks for (cells): what joins the budget. */
  readonly volume: number;
  /** Its bounding box grown by SRC_PIN_MARGIN, held to the lattice: lo inclusive, hi exclusive (cells). */
  readonly box: readonly [Vec3, Vec3];
}

/** The nozzle in source form, in cells (createInflowGridBoundary's, in the lattice). */
export interface BrickInflow {
  readonly inflow: FluidInflow;
  /** The cell (m). */
  readonly h_m: number;
  readonly outlet: Vec3;
  readonly radius: number;
  readonly velocity_mps: Vec3;
  readonly scale: number;
  readonly axis: 0 | 1 | 2;
  readonly face: number;
  /** pi r^2 |u| over h^3 (cells a second at full strength): flowRate_m3_s / h^3. */
  readonly flow: number;
  /** |u| / h (cells a second at full strength). */
  readonly speed: number;
  /** The plug box (uniformPageSources), held to the lattice: lo inclusive, hi exclusive (cells). */
  readonly box: readonly [Vec3, Vec3];
}

export interface BrickSources {
  readonly drops: readonly BrickDrop[];
  /** The nozzle; absent: none. */
  readonly inflow?: BrickInflow;
  /** Every extrusion's edges, a.xy b.xy in cells, in drop order (F's `src` buffer). */
  readonly edges: Float32Array<ArrayBuffer>;
  /** The scene's step (s): the longest step the solver takes, for the drop ahead. */
  readonly dt_s: number;
}

/** The scene's scheduled drops in source form (uniformScheduledDrop), in the lattice's cells. `box` is the container's
 * cells, the lattice's or the box a padded lattice closes (BrickEngineOptions.box): a drop box is held to it. */
export function brickSources(scene: SceneDescription, o: { lattice: Vec3; box?: Vec3; h_m: number; origin_m: Vec3; dt_s: number }): BrickSources | undefined {
  const schedule = scene.fluid.scheduledDrops ?? [];
  if (!schedule.length && !scene.fluid.inflow) return undefined;
  const h = o.h_m, cell = (p: { x: number; y: number; z: number }): Vec3 => [(p.x - o.origin_m[0]) / h, (p.y - o.origin_m[1]) / h, (p.z - o.origin_m[2]) / h];
  const edges: number[] = [], drops: BrickDrop[] = [];
  for (const { time_s, volume } of schedule) {
    if (!(time_s > 0)) throw new Error(`Brick engine: a scheduled drop at ${time_s} s (liquid at t = 0 is an initial liquid volume)`);
    let d: Omit<BrickDrop, "box">;
    if (volume.shape === "sphere") {
      d = { time_s, kind: 1, centre: cell(volume.center_m), radius: volume.radius_m / h, halfHeight: 0, edgeStart: 0, edgeCount: 0, offset: 0, halfDepth: 0, edgeRadius: 0, volume: (4 / 3) * Math.PI * volume.radius_m ** 3 / h ** 3 };
    } else if (volume.shape === "extrusion") {
      const outline = liquidExtrusionEdges(volume), count = outline.length / 4;
      if (count > LIQUID_EXTRUSION_MAX_EDGES) throw new Error(`Brick engine: an extrusion has ${count} outline edges, over the ${LIQUID_EXTRUSION_MAX_EDGES} a drop holds`);
      const { min_m, max_m } = liquidExtrusionBounds(volume), centre_m = { x: 0.5 * (min_m.x + max_m.x), y: 0.5 * (min_m.y + max_m.y), z: volume.centerZ_m };
      let reach = 0;
      for (let i = 0; i < outline.length; i += 4) reach = Math.max(reach, Math.hypot(outline[i]! - centre_m.x, outline[i + 1]! - centre_m.y));
      const start = edges.length / 4;
      for (let i = 0; i < outline.length; i += 2) edges.push((outline[i]! - o.origin_m[0]) / h, (outline[i + 1]! - o.origin_m[1]) / h);
      d = {
        time_s, kind: 3, centre: cell(centre_m), radius: Math.hypot(reach + volume.offset_m, volume.halfDepth_m) / h, halfHeight: 0,
        edgeStart: start, edgeCount: count, offset: volume.offset_m / h, halfDepth: volume.halfDepth_m / h, edgeRadius: volume.edgeRadius_m / h,
        // uniformScheduledDrop's measure: the outline sampled at a quarter cell.
        volume: liquidExtrusionVolume_m3(volume, 0.25 * h) / h ** 3,
      };
    } else {
      throw new Error(`Brick engine: a scheduled drop of shape "${(volume as { shape: string }).shape}" is not supported`);
    }
    drops.push({ ...d, box: brickDropBox(d, o.box ?? o.lattice) });
  }
  const inflow = scene.fluid.inflow ? brickInflow(scene, scene.fluid.inflow, o) : undefined;
  return { drops, ...(inflow ? { inflow } : {}), edges: Float32Array.from(edges), dt_s: o.dt_s };
}

/** requestDropPages: r = max(radius, halfHeight) about the centre, grown by the pin margin, held to `box`. */
function brickDropBox(d: Pick<BrickDrop, "centre" | "radius" | "halfHeight">, box: Vec3): [Vec3, Vec3] {
  const r = Math.max(d.radius, d.halfHeight);
  const lo = d.centre.map((c, a) => Math.min(Math.max(Math.floor(c - r) - SRC_PIN_MARGIN, 0), box[a]!)) as unknown as Vec3;
  const hi = d.centre.map((c, a) => Math.min(Math.max(Math.ceil(c + r) + SRC_PIN_MARGIN, 0), box[a]!)) as unknown as Vec3;
  return [lo, hi];
}

/** No scheduled drop and no nozzle: the sources a live drop joins (brickLiveDrop). */
export const brickNoSources = (dt_s: number): BrickSources => ({ drops: [], edges: new Float32Array(0), dt_s });

/**
 * A ball dropped into the running scene (InjectedLiquidBall: the old engine's injectLiquidBall, uniformInjectedDrop),
 * in source form, due on the step ending at `time_s`: a ball, or a disk (a cylinder along z, the 2D case's liquid)
 * when it has a half height. Its volume joins the budget on that step, as a scheduled drop's.
 */
export function brickLiveDrop(ball: { readonly centre_m: { readonly x: number; readonly y: number; readonly z: number }; readonly radius_m: number; readonly halfHeight_m?: number }, time_s: number,
  o: { lattice: Vec3; box?: Vec3; h_m: number; origin_m: Vec3 }): BrickDrop {
  const h = o.h_m, r = ball.radius_m / h, c = ball.centre_m;
  if (!(r > 0) || !Number.isFinite(r)) throw new Error(`Brick engine: a live drop's radius ${ball.radius_m} m`);
  const centre: Vec3 = [(c.x - o.origin_m[0]) / h, (c.y - o.origin_m[1]) / h, (c.z - o.origin_m[2]) / h];
  if (centre.some(v => !Number.isFinite(v))) throw new Error(`Brick engine: a live drop's centre (${c.x}, ${c.y}, ${c.z}) m`);
  const disk = ball.halfHeight_m !== undefined, halfHeight = disk ? ball.halfHeight_m! / h : 0;
  if (disk && !(halfHeight > 0)) throw new Error(`Brick engine: a live disk's half height ${ball.halfHeight_m} m`);
  const d = { time_s, kind: (disk ? 2 : 1) as 1 | 2, centre, radius: r, halfHeight, edgeStart: 0, edgeCount: 0, offset: 0, halfDepth: 0, edgeRadius: 0,
    volume: disk ? Math.PI * r * r * 2 * halfHeight : (4 / 3) * Math.PI * r ** 3 };
  return { ...d, box: brickDropBox(d, o.box ?? o.lattice) };
}

/** The nozzle in source form (createInflowGridBoundary over the container's cells; its face held to the lattice as
 * inflowFaceIndex holds it: clamp(round(outlet) - 1, 0, n - 2) on its axis). */
function brickInflow(scene: SceneDescription, inflow: FluidInflow, o: { lattice: Vec3; box?: Vec3; h_m: number; origin_m: Vec3; dt_s: number }): BrickInflow {
  const h = o.h_m, box = o.box ?? o.lattice;
  const grid = createInflowGridBoundary(inflow, scene.container, box);
  const out = inflowOutletCenter(inflow), outlet: Vec3 = [(out.x - o.origin_m[0]) / h, (out.y - o.origin_m[1]) / h, (out.z - o.origin_m[2]) / h];
  const v: Vec3 = [inflow.velocity_m_s.x, inflow.velocity_m_s.y, inflow.velocity_m_s.z], speed = Math.hypot(...v);
  const axis = grid.axis, face = Math.min(Math.max(Math.round(outlet[axis]) - 1, 0), o.lattice[axis] - 2);
  // uniformPageSources: the outlet to its reach over the scene's longest step (the fixed step here) and the overlap,
  // grown by the radius; floor and ceil, held to the lattice.
  const d = speed > 0 ? v.map(x => x / speed) : [1, 0, 0], reach = speed * o.dt_s / h + SRC_PLUG_OVERLAP, r = inflow.radius_m / h;
  const end = outlet.map((x, a) => x + reach * d[a]!);
  const lo = outlet.map((x, a) => Math.min(Math.max(Math.floor(Math.min(x, end[a]!) - r), 0), o.lattice[a]!)) as unknown as Vec3;
  const hi = outlet.map((x, a) => Math.min(Math.max(Math.ceil(Math.max(x, end[a]!) + r), 0), o.lattice[a]!)) as unknown as Vec3;
  return { inflow, h_m: h, outlet, radius: r, velocity_mps: v, scale: grid.apertureScale, axis, face, flow: grid.flowRate_m3_s / h ** 3, speed: speed / h, box: [lo, hi] };
}

/** The nozzle's words for the step (start, start + dt]: [flags, strength, sweep] and the cells it adds. */
export function brickInflowStep(n: BrickInflow, start_s: number, dt_s: number): { flags: number; strength: number; sweep: number; added: number } {
  const i = n.inflow, strength = averageInflowStrength(i, start_s, start_s + dt_s);
  // physical.w: the nozzle ran before this step (not first at its start).
  const continues = i.enabled !== false && start_s > i.start_s + 0.5 * dt_s && inflowStrengthAt(start_s, i.start_s, i.end_s, i.ramp_s) > 0;
  // inflowNeedsFine: a thin plug (radius within a 4h cell) or a short sweep (within the coarse cell's axial span).
  const v = n.velocity_mps, speed_mps = Math.hypot(...v) * strength, sweep = n.speed * strength * dt_s;
  const dir = v.map(x => Math.abs(x) / Math.max(Math.hypot(...v), 1e-30));
  const fine = speed_mps > 1e-6 && (n.radius <= 4 || sweep + SRC_PLUG_OVERLAP <= 4 * (dir[0]! + dir[1]! + dir[2]!));
  return { flags: (strength > 0 ? 1 : 0) | (continues ? 2 : 0) | (fine ? 4 : 0), strength, sweep, added: strength > 0 ? n.flow * strength * dt_s : 0 };
}

/** The boot's source boxes: the first step's drop and the nozzle's plug box (claimed every step, the old pinned pages). */
export function brickBootSourceBoxes(s: BrickSources, dt_s: number): (readonly [Vec3, Vec3])[] {
  const first = brickDropDue(s, 0, dt_s);
  return [...(first >= 0 ? [s.drops[first]!.box] : []), ...(s.inflow ? [s.inflow.box] : [])];
}

/** The drop the step (start, end] adds (scheduledDrop: the same slack both ends tiles the clock), or -1. */
export function brickDropDue(s: BrickSources, start_s: number, end_s: number): number {
  const slack = 1e-6 * (end_s - start_s);
  let due = 0, first = -1;
  s.drops.forEach((d, i) => { if (d.time_s > start_s + slack && d.time_s <= end_s + slack && !due++) first = i; });
  if (due > 1) throw new Error(`Brick engine: ${due} scheduled drops fall in the step ending at ${end_s} s (the first at ${s.drops[first]!.time_s} s); a step adds one drop, so space them at least a step apart`);
  return first;
}

/** Fill the ring's source words for the step starting at `start_s` (F's words only; the rest are the engine's). */
export function brickSourceRing(s: BrickSources, start_s: number, u: Uint32Array, f: Float32Array) {
  const i32 = new Int32Array(u.buffer, u.byteOffset, u.length);
  const dt = s.dt_s, now = brickDropDue(s, start_s, start_s + dt), ahead = brickDropDue(s, start_s + dt, start_s + 2 * dt);
  for (let k = SRC_RING.KIND; k <= SRC_RING.INFLOW_BOX + 5; k++) u[k] = 0;
  f[SRC_RING.ADDED] = 0;
  let boxes = 0;
  if (now >= 0) {
    const d = s.drops[now]!;
    u[SRC_RING.KIND] = d.kind; f.set(d.centre, SRC_RING.CENTRE); f[SRC_RING.RADIUS] = d.radius; f[SRC_RING.HALF_HEIGHT] = d.halfHeight;
    u[SRC_RING.EDGE_START] = d.edgeStart; u[SRC_RING.EDGE_COUNT] = d.edgeCount;
    f[SRC_RING.OFFSET] = d.offset; f[SRC_RING.HALF_DEPTH] = d.halfDepth; f[SRC_RING.EDGE_RADIUS] = d.edgeRadius;
    f[SRC_RING.ADDED] = d.volume;
    i32.set([...d.box[0], ...d.box[1]], SRC_RING.BOX); boxes |= 1;
  }
  if (ahead >= 0) { const d = s.drops[ahead]!; i32.set([...d.box[0], ...d.box[1]], SRC_RING.AHEAD); boxes |= 2; }
  if (s.inflow) {
    const n = s.inflow, w = brickInflowStep(n, start_s, dt);
    u[SRC_RING.INFLOW] = w.flags; f[SRC_RING.STRENGTH] = w.strength; f.set(n.outlet, SRC_RING.OUTLET); f[SRC_RING.IN_RADIUS] = n.radius;
    f.set(n.velocity_mps, SRC_RING.VELOCITY); f[SRC_RING.SCALE] = n.scale; f[SRC_RING.SWEEP] = w.sweep;
    u[SRC_RING.AXIS] = n.axis; i32[SRC_RING.FACE] = n.face;
    f[SRC_RING.ADDED] += w.added;
    i32.set([...n.box[0], ...n.box[1]], SRC_RING.INFLOW_BOX); boxes |= 4;
  }
  u[SRC_RING.BOXES] = boxes;
}

/** The F bricks (keys) of a cell box: every brick it touches (the boot's set takes the first step's drop). */
export function brickBoxBricks(box: readonly [Vec3, Vec3]): [Vec3, Vec3] {
  return [box[0].map(v => Math.floor(v / 8)) as unknown as Vec3, box[1].map(v => Math.ceil(v / 8)) as unknown as Vec3];
}
