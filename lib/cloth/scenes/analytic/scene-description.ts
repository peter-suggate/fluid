/**
 * An analytic cloth scene as a product scene document: a dry scene whose cloth
 * runs on the scene clock (one implicit 1/60 s step per frame, never
 * substepped), standing on the container's floor. Its solids are ordinary
 * scene voxel solids that render, hit-test and edit through the shared
 * solid-world path: an axis-aligned box on lattice planes is an exact fill,
 * anything else one procedural bake, the same rule as the Bridson scenes and
 * the rule cloth contact reads back (`contact-solids.ts`).
 */
import { createEmptyScene } from "../../../core/empty-scene";
import type { CameraState, SceneDescription, Vec3 } from "../../../core/model";
import { bakeProceduralSolid } from "../../../core/procedural-solid";
import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "../../../core/solid-world";
import { VOXEL_MATERIAL_IDS } from "../../../core/voxel-scene";
import type { ClothAnalyticScene, ClothAnalyticSolid, ClothAnalyticSolidShape } from "./scenes";

const BRICK = 8;
const ALIGNED_TOLERANCE_CELLS = 1e-6;

function unknownShape(shape: never): never {
  throw new RangeError(`Unknown analytic cloth solid shape ${String((shape as { kind?: unknown })?.kind ?? shape)}`);
}

/** Signed distance (m), negative inside: exact for the box, the largest plane distance for the wedge (exact inside, 1-Lipschitz, exact zero set). */
export function clothAnalyticSolidDistance(shape: ClothAnalyticSolidShape, p: Vec3): number {
  switch (shape.kind) {
    case "box": {
      const c = [(shape.lo_m.x + shape.hi_m.x) / 2, (shape.lo_m.y + shape.hi_m.y) / 2, (shape.lo_m.z + shape.hi_m.z) / 2];
      const e = [(shape.hi_m.x - shape.lo_m.x) / 2, (shape.hi_m.y - shape.lo_m.y) / 2, (shape.hi_m.z - shape.lo_m.z) / 2];
      const q = [Math.abs(p.x - c[0]!) - e[0]!, Math.abs(p.y - c[1]!) - e[1]!, Math.abs(p.z - c[2]!) - e[2]!];
      return Math.hypot(...q.map(v => Math.max(v, 0))) + Math.min(Math.max(...q), 0);
    }
    case "wedge": {
      const t = shape.angle_deg * Math.PI / 180, slope = Math.sin(t) * (p.x - shape.x0_m) + Math.cos(t) * (p.y - shape.top_m);
      return Math.max(-p.y, shape.x0_m - p.x, p.x - shape.x1_m, Math.abs(p.z - shape.zc_m) - shape.halfWidth_m, slope);
    }
    default: return unknownShape(shape);
  }
}

/** Scene-space bounds of a solid. */
export function clothAnalyticSolidBounds(shape: ClothAnalyticSolidShape): { lo: [number, number, number]; hi: [number, number, number] } {
  switch (shape.kind) {
    case "box": return { lo: [shape.lo_m.x, shape.lo_m.y, shape.lo_m.z], hi: [shape.hi_m.x, shape.hi_m.y, shape.hi_m.z] };
    case "wedge": return { lo: [shape.x0_m, 0, shape.zc_m - shape.halfWidth_m], hi: [shape.x1_m, shape.top_m, shape.zc_m + shape.halfWidth_m] };
    default: return unknownShape(shape);
  }
}

/** World position of lattice node (0, 0, 0) for a scene's container. */
export function clothAnalyticLatticeOrigin(source: ClothAnalyticScene): [number, number, number] {
  return [-source.extents_m.x / 2, 0, -source.extents_m.z / 2];
}

/** A box solid whose faces all lie on lattice planes is baked as an exact fill. */
export function clothAnalyticSolidIsFill(solid: ClothAnalyticSolid, source: ClothAnalyticScene): boolean {
  if (solid.shape.kind !== "box") return false;
  const o = clothAnalyticLatticeOrigin(source), h = source.cell_m, b = clothAnalyticSolidBounds(solid.shape);
  return [...b.lo, ...b.hi].every((v, i) => { const c = (v - o[i % 3]!) / h; return Math.abs(c - Math.round(c)) < ALIGNED_TOLERANCE_CELLS; });
}

function latticeDimensions(source: ClothAnalyticScene): [number, number, number] {
  const e = [source.extents_m.x, source.extents_m.y, source.extents_m.z];
  const d = e.map(v => Math.round(v / source.cell_m));
  if (!d.every((v, a) => v % BRICK === 0 && Math.abs(v * source.cell_m - e[a]!) < 1e-9)) throw new RangeError(`Analytic cloth scene ${source.id}: extents must be whole 8-cell bricks`);
  return d as [number, number, number];
}

function bakeSolids(source: ClothAnalyticScene): Pick<SceneDescription, "solidVoxels" | "solidBoundary"> {
  const h = source.cell_m, o = clothAnalyticLatticeOrigin(source), dims = latticeDimensions(source);
  const fills: SolidWorldVoxelPatch[] = [], shaped: ClothAnalyticSolid[] = [];
  for (const solid of source.solids) {
    if (!clothAnalyticSolidIsFill(solid, source)) { shaped.push(solid); continue; }
    const b = clothAnalyticSolidBounds(solid.shape), cell = (v: number, a: number) => Math.max(0, Math.min(dims[a]!, Math.round((v - o[a]!) / h)));
    fills.push({ operation: "fill", minimum: [cell(b.lo[0], 0), cell(b.lo[1], 1), cell(b.lo[2], 2)],
      maximumExclusive: [cell(b.hi[0], 0), cell(b.hi[1], 1), cell(b.hi[2], 2)], materialId: VOXEL_MATERIAL_IDS.white });
  }
  if (!shaped.length) return { solidVoxels: fills };
  // One brick-aligned lattice around the shaped solids with a two-cell margin, clipped to the container.
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const s of shaped) { const b = clothAnalyticSolidBounds(s.shape); for (let a = 0; a < 3; a++) { lo[a] = Math.min(lo[a]!, b.lo[a]!); hi[a] = Math.max(hi[a]!, b.hi[a]!); } }
  const start = [0, 1, 2].map(a => Math.max(0, BRICK * Math.floor(((lo[a]! - 2 * h) - o[a]!) / (BRICK * h) + 1e-9)));
  const end = [0, 1, 2].map(a => Math.min(dims[a]!, BRICK * Math.ceil(((hi[a]! + 2 * h) - o[a]!) / (BRICK * h) - 1e-9)));
  const size = [0, 1, 2].map(a => end[a]! - start[a]!) as [number, number, number];
  const baked = bakeProceduralSolid(size, [h, h, h], q => {
    const p = { x: o[0] + (start[0]! + q[0]) * h, y: o[1] + (start[1]! + q[1]) * h, z: o[2] + (start[2]! + q[2]) * h };
    let d = Infinity;
    for (const s of shaped) d = Math.min(d, clothAnalyticSolidDistance(s.shape, p));
    return d / h;
  }, VOXEL_MATERIAL_IDS.white);
  const shift = (v: readonly number[]) => v.map((x, a) => x + start[a]!) as [number, number, number];
  const patches = baked.solidVoxels.map(p => ({ ...p, minimum: shift(p.minimum), maximumExclusive: shift(p.maximumExclusive) }));
  // The procedural patches lead: the boundary covers the prefix solidVoxels.slice(0, patchCount).
  return {
    solidVoxels: [...patches, ...fills],
    solidBoundary: { ...baked.solidBoundary, dimensions: [...dims], patchesKey: solidVoxelPatchesKey(patches),
      bricks: baked.solidBoundary.bricks.map(b => ({ ...b, origin: shift(b.origin) })) },
  };
}

export interface ClothAnalyticSceneOptions {
  /** `bake: false` skips the voxel bake (library card previews). */
  readonly bake?: boolean;
  /** Solver iterations for every document (the tests' K sweep); default the scene's own. */
  readonly iterations?: number;
}

/** The document the product opens. */
export function clothAnalyticSceneDescription(source: ClothAnalyticScene, options: ClothAnalyticSceneOptions = {}): SceneDescription {
  const dims = latticeDimensions(source), h = source.cell_m;
  const scene = createEmptyScene({ sceneId: source.id, finestCellSize_m: h, extents_m: { x: dims[0] * h, y: dims[1] * h, z: dims[2] * h } });
  scene.duration_s = source.duration_s;
  scene.numerics = { ...scene.numerics, fixedDt_s: source.step_s, maxDt_s: source.step_s };
  if (options.bake !== false) {
    const solids = bakeSolids(source);
    scene.solidVoxels = [...solids.solidVoxels, ...scene.solidVoxels];
    if (solids.solidBoundary) scene.solidBoundary = solids.solidBoundary;
  }
  scene.cloth = source.cloth.map(d => options.iterations === undefined ? d : { ...d, solver: { ...d.solver, iterations: options.iterations } });
  return scene;
}

export function clothAnalyticCamera(source: ClothAnalyticScene): Partial<CameraState> {
  return source.camera;
}
