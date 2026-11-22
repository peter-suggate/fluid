/**
 * Where the brick engine nests its detail engines (brick-detail.ts), in the parent's cells.
 *
 * Two sources:
 * - authored: the scene's detail regions (refinement regions whose minimum cell is 1/2 or 1/4 of a finest cell,
 *   refinement-regions.ts), snapped outward to whole 4h tiles and clipped to the container's cells;
 * - automatic (fluid.automaticDetail): the placement heuristic below.
 * Boxes that overlap or touch are merged into one box (their bounding box, the finer factor): one engine a box, so no
 * two engines ever claim the same cell, and the renderer never draws two surfaces over one place.
 *
 * The heuristic. High-energy splashes come from fast liquid meeting thin obstacles (the NB-FLIP teaser's rods): a
 * sheet wraps the obstacle, thins and breaks up, and that is where h is visibly too coarse. So a box is placed around
 * every thin upstanding static obstacle (a rod, post or pillar: horizontally at most DETAIL_OBSTACLE_MAX_WIDTH cells
 * on both axes, at least DETAIL_OBSTACLE_MIN_HEIGHT tall) whose foot lies below the liquid's top, in a scene with
 * energy to splash: liquid at t = 0 more than DETAIL_ENERGY_DROP cells above its rest level, or a scheduled drop or
 * an inflow. A still tank, however many posts it holds, gets none. The box is the obstacle's tiles dilated by
 * DETAIL_OBSTACLE_MARGIN cells horizontally, from the floor (or the margin below its foot) up to the margin above the
 * higher of its top and the liquid's top: the impact's splash rises along the obstacle.
 */
import type { SceneDescription } from "../../core/model";
import { refinementRegionCellBounds, refinementRegionDetailFactor, refinementRegionLattice, sceneDetailRegions } from "../../core/refinement-regions";
import type { Vec3 } from "./brick-engine";

/** A thin obstacle is at most this wide on both horizontal axes (cells). */
export const DETAIL_OBSTACLE_MAX_WIDTH = 32;
/** ... and at least this tall (cells). */
export const DETAIL_OBSTACLE_MIN_HEIGHT = 8;
/** The air (and liquid) kept about an obstacle (cells; a multiple of the 4h tile). */
export const DETAIL_OBSTACLE_MARGIN = 12;
/** Liquid this far above its rest level at t = 0 has energy to splash (cells). */
export const DETAIL_ENERGY_DROP = 8;
/** A detail box's smallest edge (parent cells). */
export const DETAIL_MIN_EDGE = 16;

export interface DetailBox {
  /** Parent cells, lo inclusive, hi exclusive; whole 4h tiles, inside the container's cells. */
  readonly lo: Vec3;
  readonly hi: Vec3;
  /** The fine engine's refinement: h / factor. */
  readonly factor: 2 | 4;
  /** What placed it: region ids, and "auto:<n>" for the heuristic's obstacles. */
  readonly origins: readonly string[];
}

/** The liquid's vertical extent at t = 0 (brick-scene.ts brickInitialExtent's), and whether sources add energy. */
export interface DetailLiquidReach {
  /** The highest liquid row + 1 (cells). */
  readonly top: number;
  /** The rest level: the liquid's volume over the container's floor (cells). */
  readonly rest: number;
  readonly sources: boolean;
}

type Box = { lo: number[]; hi: number[] };

const tileFloor = (v: number) => 4 * Math.floor(v / 4 + 1e-9);
const tileCeil = (v: number) => 4 * Math.ceil(v / 4 - 1e-9);

/** Every detail box of the scene (authored and automatic, merged), or an explicit refusal. */
export function brickDetailBoxes(scene: SceneDescription, cells: Vec3, reach: DetailLiquidReach): DetailBox[] {
  const boxes: (Box & { factor: 2 | 4; origins: string[] })[] = [];
  const lattice = refinementRegionLattice(scene);
  for (const region of sceneDetailRegions(scene)) {
    const factor = refinementRegionDetailFactor(region)!;
    const b = refinementRegionCellBounds(region, lattice);
    boxes.push({ lo: b.min.map(tileFloor), hi: b.max.map((v, a) => Math.min(cells[a]!, tileCeil(v))), factor, origins: [region.id] });
  }
  const automatic = scene.fluid.automaticDetail;
  if (automatic) automaticDetailBoxes(scene, cells, reach).forEach((b, i) => boxes.push({ ...b, factor: automatic.factor, origins: [`auto:${i}`] }));
  for (const b of boxes) for (let a = 0; a < 3; a++) {
    // A box thinner than a brick on an axis grows about its centre (whole tiles, inside the container).
    if (b.hi[a]! - b.lo[a]! < DETAIL_MIN_EDGE) {
      const lo = Math.max(0, Math.min(cells[a]! - DETAIL_MIN_EDGE, tileFloor(0.5 * (b.lo[a]! + b.hi[a]!) - DETAIL_MIN_EDGE / 2)));
      b.lo[a] = lo; b.hi[a] = Math.min(cells[a]!, lo + DETAIL_MIN_EDGE);
    }
  }
  // Merge overlapping or touching boxes, and keep each coupled high face's storage (below), to a fixed point.
  for (let changed = true; changed;) {
    changed = false;
    for (let merged = true; merged;) {
      merged = false;
      for (let i = 0; i < boxes.length && !merged; i++) for (let j = i + 1; j < boxes.length && !merged; j++) {
        const p = boxes[i]!, q = boxes[j]!;
        if ([0, 1, 2].every(a => p.lo[a]! <= q.hi[a]! && q.lo[a]! <= p.hi[a]!)) {
          boxes[i] = { lo: p.lo.map((v, a) => Math.min(v, q.lo[a]!)), hi: p.hi.map((v, a) => Math.max(v, q.hi[a]!)), factor: Math.max(p.factor, q.factor) as 2 | 4, origins: [...p.origins, ...q.origins] };
          boxes.splice(j, 1); merged = true;
        }
      }
    }
    // A coupled high face (one inside the container) is the lower face of the first fine brick past the box, stored
    // there only when the fine lattice (whole 32-cell pages) reaches past the box: a fine extent of whole pages grows
    // by a 4h tile (or, at the container's high wall, moves its low face down one), so its pages pad past it.
    for (const b of boxes) for (let a = 0; a < 3; a++) {
      if (b.hi[a]! >= cells[a]! || (b.factor * (b.hi[a]! - b.lo[a]!)) % 32 !== 0) continue;
      if (b.hi[a]! + 4 <= cells[a]!) b.hi[a] = b.hi[a]! + 4; else if (b.lo[a]! >= 4) b.lo[a] = b.lo[a]! - 4; else continue;
      changed = true;
    }
  }
  return boxes.filter(b => b.hi.every((v, a) => v > b.lo[a]!)).map(b => ({ lo: b.lo as unknown as Vec3, hi: b.hi as unknown as Vec3, factor: b.factor, origins: b.origins }));
}

/**
 * The heuristic's boxes (parent cells, whole tiles; unmerged). The obstacles are the scene's fill patches, rasterised
 * to 4h tiles by their bounds (a round kind's box over-covers by under a tile, which the margin swallows) and joined
 * into 6-connected components. A patch wider than a page on a horizontal axis is a wall or a floor, never a thin
 * obstacle, and is skipped.
 */
export function automaticDetailBoxes(scene: SceneDescription, cells: Vec3, reach: DetailLiquidReach): Box[] {
  if (!(reach.sources || reach.top - reach.rest > DETAIL_ENERGY_DROP)) return [];
  const nt = cells.map(n => Math.ceil(n / 4)), key = (x: number, y: number, z: number) => x + nt[0]! * (y + nt[1]! * z);
  const tiles = new Set<number>();
  for (const patch of scene.solidVoxels ?? []) {
    if (patch.operation !== "fill") continue;
    const lo = patch.minimum, hi = patch.maximumExclusive;
    if (hi[0] - lo[0] > DETAIL_OBSTACLE_MAX_WIDTH || hi[2] - lo[2] > DETAIL_OBSTACLE_MAX_WIDTH) continue;
    const t0 = [0, 1, 2].map(a => Math.max(0, Math.floor(lo[a]! / 4))), t1 = [0, 1, 2].map(a => Math.min(nt[a]!, Math.ceil(hi[a]! / 4)));
    for (let z = t0[2]!; z < t1[2]!; z++) for (let y = t0[1]!; y < t1[1]!; y++) for (let x = t0[0]!; x < t1[0]!; x++) tiles.add(key(x, y, z));
  }
  const out: Box[] = [], seen = new Set<number>();
  for (const start of tiles) {
    if (seen.has(start)) continue;
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], queue = [start];
    seen.add(start);
    while (queue.length) {
      const k = queue.pop()!, x = k % nt[0]!, y = Math.floor(k / nt[0]!) % nt[1]!, z = Math.floor(k / (nt[0]! * nt[1]!));
      const c = [x, y, z];
      for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a]!, c[a]!); hi[a] = Math.max(hi[a]!, c[a]! + 1); }
      for (const [dx, dy, dz] of [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]] as const) {
        const nx = x + dx, ny = y + dy, nz = z + dz;
        if (nx < 0 || ny < 0 || nz < 0 || nx >= nt[0]! || ny >= nt[1]! || nz >= nt[2]!) continue;
        const n = key(nx, ny, nz);
        if (tiles.has(n) && !seen.has(n)) { seen.add(n); queue.push(n); }
      }
    }
    const clo = lo.map(v => 4 * v), chi = hi.map((v, a) => Math.min(cells[a]!, 4 * v));
    const wide = Math.max(chi[0]! - clo[0]!, chi[2]! - clo[2]!), tall = chi[1]! - clo[1]!;
    if (wide > DETAIL_OBSTACLE_MAX_WIDTH || tall < DETAIL_OBSTACLE_MIN_HEIGHT || clo[1]! >= reach.top) continue;
    const m = DETAIL_OBSTACLE_MARGIN, footLo = clo[1]! <= m ? 0 : clo[1]! - m;
    out.push({
      lo: [Math.max(0, clo[0]! - m), footLo, Math.max(0, clo[2]! - m)],
      hi: [Math.min(cells[0]!, chi[0]! + m), Math.min(cells[1]!, tileCeil(Math.max(chi[1]!, reach.top) + m)), Math.min(cells[2]!, chi[2]! + m)],
    });
  }
  return out;
}
