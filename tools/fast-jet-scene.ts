/** A reduced reservoir seep for fast-front reach (tiered-adaptivity plan §3.3,
 * Phase 4): a head of 0.64 to 0.8 m over a square throat in a 16-cell wall,
 * so the jet leaves at 3.5 to 4 m/s, on a lattice that fits at 5, 2.5 and
 * 1.25 mm. The jet travels 12, 23 and 47 h a 1/60 s step there.
 *
 * What differs from `reservoir-seep`:
 *  - the throat is metric (throat_m, default 7.5 mm: 2, 3 and 6 cells), not
 *    one cell at every h. A one-cell jet is below what phi resolves: at 5 mm
 *    reservoir-seep's throat delivers about 3 mL to the catch chamber and
 *    loses 1.2% of the liquid in its first second, then stops. A resolved
 *    jet makes discharge and trajectory comparable across h;
 *  - the reservoir is a narrow slot (width and depth below), so its pages
 *    are few at 1.25 mm. It drains: the head falls by about a third over
 *    20 s, the same at every h. Report discharge against the measured head;
 *  - the throat sits low (throatHeight_m), so the jet's arc and its box are
 *    short: it lands about sqrt(2 y / g) · 3.5 m/s ≈ 0.45 m past the wall.
 * The geometry is metric and the same at every cell size, except the wall,
 * which stays 16 cells as in `reservoir-seep` (its faces on 16-cell
 * boundaries). */
import { cloneScene, defaultScene, type SceneDescription } from "../lib/core/model";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice-dimensions";

export interface FastJetOptions {
  readonly cellSize_m: number;
  /** Initial head of the free surface over the throat (reservoir-seep: 0.64 m). */
  readonly head_m?: number;
  /** Throat side (square), rounded to whole cells. */
  readonly throat_m?: number;
  readonly throatHeight_m?: number;
  readonly reservoirWidth_m?: number;
  readonly catchWidth_m?: number;
  readonly depth_m?: number;
  /** Air over the reservoir's surface. */
  readonly headroom_m?: number;
  /** Page-pool blocks (SceneDescription.fluid.pagePool.blocks); default from the footprint below. */
  readonly blocks?: number;
}

export interface FastJetGeometry {
  readonly scene: SceneDescription;
  readonly cells: readonly [number, number, number];
  /** Wall cells [low, high) in x; the throat cell (y, z). */
  readonly wall: readonly [number, number];
  readonly throat: readonly [number, number];
  /** Throat side in cells, and the reservoir's free-surface area (m²). */
  readonly throatCells: number;
  readonly reservoirArea_m2: number;
  /** Ideal exit speed sqrt(2 g head) and its travel in h a 1/60 s step. */
  readonly exitSpeed_m_s: number;
  readonly travelPerStep_h: number;
}

export function createFastJetScene(o: FastJetOptions): FastJetGeometry {
  const h = o.cellSize_m, head = o.head_m ?? 0.8, throatY = o.throatHeight_m ?? 0.08;
  const reservoir = o.reservoirWidth_m ?? 0.16, catchWidth = o.catchWidth_m ?? 0.56;
  const n = Math.max(1, Math.round((o.throat_m ?? 0.0075) / h));
  const depth = o.depth_m ?? 0.04, headroom = o.headroom_m ?? 0.08;
  // Wall faces on 16-cell boundaries (reservoir-seep's rule): the reservoir
  // rounds to 16 cells, the wall is 16 cells, the catch chamber rounds to 32.
  const low = 16 * Math.max(1, Math.round(reservoir / (16 * h))), high = low + 16;
  const nx = high + 32 * Math.max(1, Math.round(catchWidth / (32 * h)));
  const ny = 32 * Math.ceil((throatY + head + headroom) / (32 * h));
  const nz = Math.max(8, 2 * Math.round(depth / (2 * h)));
  const scene = cloneScene(defaultScene);
  scene.sceneId = "fast-jet";
  scene.duration_s = 20;
  scene.container = {
    width_m: nx * h, height_m: ny * h, depth_m: nz * h,
    fillFraction: 0, top: "open", fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: h, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: h };
  scene.rigidBodies = [];
  delete scene.terrain;
  const cells = sceneLatticeDimensions(scene);
  if (cells[0] !== nx || cells[1] !== ny || cells[2] !== nz) throw new Error(`Fast jet lattice ${cells} is not ${[nx, ny, nz]}`);
  const mouth = Math.ceil(n * (Math.SQRT2 - 1) / 2) + 1;
  const holeY = Math.round(throatY / h), holeZ = Math.floor((nz - n) / 2), topY = holeY + Math.round(head / h);
  const point = (x: number, y: number, z: number) => ({
    x: x * h - scene.container.width_m / 2, y: y * h, z: z * h - scene.container.depth_m / 2,
  });
  scene.solidVoxels = [
    { operation: "fill", minimum: [low, 0, 0], maximumExclusive: [high, ny, nz], materialId: 2 },
    { operation: "clear", minimum: [low, holeY, holeZ], maximumExclusive: [high, holeY + n, holeZ + n] },
    // Rounded mouths: a cylinder that holds the throat's corners with a
    // cell to spare.
    ...([low, high - 1] as const).map(x => ({
      operation: "clear" as const, kind: "cylinder", axis: 0 as const,
      minimum: [x, holeY - mouth, holeZ - mouth] as const,
      maximumExclusive: [x + 1, holeY + n + mouth, holeZ + n + mouth] as const,
    })),
  ];
  // Pages: the reservoir and a two-page collar, the jet's arc over the
  // catch chamber at the throat's height, and the film over its floor, all
  // 25% over.
  const p = (cells: number) => Math.ceil(cells / 32);
  const reservoirPages = p(low + 64) * p(topY + 64) * p(nz);
  const catchPages = p(nx - high) * (p(holeY + 64) + 1) * p(nz);
  scene.fluid = {
    density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0.001002,
    surfaceTension_N_m: 0.072, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: point(0, 0, 0), max_m: point(low, topY, nz) },
      // The wetted throat, as reservoir-seep starts it.
      { shape: "box", min_m: point(low - 0.25, holeY - 0.25, holeZ - 0.25),
        max_m: point(high, holeY + n + 0.25, holeZ + n + 0.25) }],
    pagePool: { blocks: o.blocks ?? Math.ceil(1.25 * (reservoirPages + catchPages)) + 2 },
    refinementRegions: [{ id: "seep-throat", rule: "minimum-cell-size",
      min_m: point(low - 4, holeY - 4, holeZ - 4), max_m: point(high + 4, holeY + n + 7, holeZ + n + 7),
      minimumCellSize_cells: 1, maximumCellSize_cells: 1 }],
  };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.surfaceStyle = "smooth";
  const g = 9.80665, exitSpeed_m_s = Math.sqrt(2 * g * head);
  return { scene, cells, wall: [low, high], throat: [holeY, holeZ], throatCells: n, reservoirArea_m2: low * h * nz * h, exitSpeed_m_s, travelPerStep_h: exitSpeed_m_s / 60 / h };
}

/** reservoir-seep's NB-FLIP method profile (lib/core/scenes.ts). */
export const FAST_JET_METHOD_OVERRIDES = {
  detailPolicy: "dynamic", detailNearFocus: "off", detailBudgetPercent: 5,
  regionTiering: "error", warmResolution: "h",
} as const;
