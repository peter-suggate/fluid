/**
 * Where the brick engine runs its detail engine (brick-detail.ts): one fine engine over the container at h / factor,
 * active only on a mask of the parent's 4h tiles (brick-detail.wgsl.ts). Two sources of tiles:
 * - authored: the scene's detail regions (refinement regions whose minimum cell is 1/2 or 1/4 of a finest cell,
 *   refinement-regions.ts), their 4h tiles always on (outward to whole tiles, clipped to the container);
 * - automatic (fluid.automaticDetail): the hottest tiles of the parent's surface every step, under a budget
 *   (brick-detail-alloc.wgsl.ts: the deformation heat).
 * One engine, so one factor: regions and the automatic detail that ask for different factors are refused.
 */
import type { SceneDescription } from "../../core/model";
import { refinementRegionCellBounds, refinementRegionDetailFactor, refinementRegionLattice, sceneDetailRegions } from "../../core/refinement-regions";
import type { Vec3 } from "./brick-engine";

/**
 * The automatic detail's default budget (mask tiles: parent 4h tiles; a fine brick each at ×2, eight at ×4). 128 holds
 * a crown's rim (cm12-figure-6) and the teaser's rod splashes at about a tenth of the parent's step on the teaser; the
 * fine engine's floor (~5 ms, launch-latency-bound dispatches) dominates below it (2026-11-23).
 */
export const DETAIL_DEFAULT_BUDGET = 128;

export interface DetailPlan {
  /** The fine engine's refinement: h / factor. */
  readonly factor: 2 | 4;
  /** The heat's tiles a step (0: authored regions alone). */
  readonly budget: number;
  /** The authored regions' tiles (tile indices, x fastest over the container's 4h tiles), always on. */
  readonly forced: Uint32Array;
  /** The container's 4h tiles an axis. */
  readonly tiles: Vec3;
  /** What asked for it: region ids, and "auto". */
  readonly origins: readonly string[];
}

/** The scene's detail plan, undefined without one; an explicit refusal (throws) when it cannot be one engine. */
export function brickDetailPlan(scene: SceneDescription, cells: Vec3): DetailPlan | undefined {
  const regions = sceneDetailRegions(scene), automatic = scene.fluid.automaticDetail;
  if (!regions.length && !automatic) return undefined;
  const tiles = cells.map(n => Math.ceil(n / 4)) as unknown as Vec3;
  const factors = new Set<number>(regions.map(r => refinementRegionDetailFactor(r)!));
  if (automatic) factors.add(automatic.factor);
  if (factors.size > 1) throw new Error(`NB-FLIP detail: one detail engine a scene, so one factor; the scene asks for ${[...factors].sort().map(f => `h/${f}`).join(" and ")}`);
  const factor = [...factors][0] as 2 | 4;
  const lattice = refinementRegionLattice(scene), forced = new Set<number>();
  for (const region of regions) {
    const b = refinementRegionCellBounds(region, lattice);
    const lo = b.min.map(v => Math.max(0, Math.floor(v / 4 + 1e-9))), hi = b.max.map((v, a) => Math.min(tiles[a]!, Math.ceil(v / 4 - 1e-9)));
    for (let z = lo[2]!; z < hi[2]!; z++) for (let y = lo[1]!; y < hi[1]!; y++) for (let x = lo[0]!; x < hi[0]!; x++) forced.add(x + tiles[0] * (y + tiles[1] * z));
  }
  const budget = automatic ? automatic.budget ?? DETAIL_DEFAULT_BUDGET : 0;
  return { factor, budget, forced: Uint32Array.from([...forced].sort((p, q) => p - q)), tiles, origins: [...regions.map(r => r.id), ...(automatic ? ["auto"] : [])] };
}
