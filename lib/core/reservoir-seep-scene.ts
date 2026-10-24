import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { sceneLatticeDimensions } from "./scene-lattice-dimensions";

export const RESERVOIR_SEEP_ID = "reservoir-seep";
export const RESERVOIR_SEEP_CELL_M = 0.005;
export const RESERVOIR_SEEP_CAMERA: Partial<CameraState> = {
  distance_m: 3.8, target_m: { x: 0, y: 0.55, z: 0 },
  azimuth_rad: 0.22, elevation_rad: 0.28,
};

/** A large quiet reservoir drives a tiny jet solely through hydrostatic head.
 * Keep the throat at h when rebuilding the lattice: this is a resolution
 * stress scene, not a geometrically scaled nozzle benchmark. `widthScale`
 * lengthens both chambers about the middle wall and `depthScale` the span,
 * at the same h and head (the scale ladder's box variants). */
export function createReservoirSeepScene(cellSize_m = RESERVOIR_SEEP_CELL_M,
  { widthScale = 1, depthScale = 1 }: { readonly widthScale?: number; readonly depthScale?: number } = {}): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = RESERVOIR_SEEP_ID;
  scene.duration_s = 20;
  scene.container = {
    width_m: 2.56 * widthScale, height_m: 1.28, depth_m: 0.32 * depthScale,
    fillFraction: 0, top: "open", fluidWallMode: "free-slip",
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: cellSize_m };
  scene.rigidBodies = [];
  delete scene.terrain;
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  const hx = scene.container.width_m / nx, hy = scene.container.height_m / ny,
    hz = scene.container.depth_m / nz;
  // The wall's faces sit on 16-cell boundaries so the coarsest tier can meet
  // them directly: one 16-cell span just left of the middle.
  const high = 16 * Math.floor(nx / 32), low = high - 16;
  const holeY = Math.floor(ny / 4), holeZ = Math.floor(nz / 2);
  const point = (x: number, y: number, z: number) => ({
    x: x * hx - scene.container.width_m / 2, y: y * hy,
    z: z * hz - scene.container.depth_m / 2,
  });
  scene.solidVoxels = [
    { operation: "fill", minimum: [low, 0, 0], maximumExclusive: [high, ny, nz], materialId: 2 },
    // A one-cell throat is actually open in the binary collision mask.
    { operation: "clear", minimum: [low, holeY, holeZ], maximumExclusive: [high, holeY + 1, holeZ + 1] },
    // Round both mouths without widening the h throat. A one-cell cylinder
    // alone flattens to a box in the shared voxel/analytic shape contract.
    ...([low, high - 1] as const).map(x => ({
      operation: "clear" as const, kind: "cylinder", axis: 0 as const,
      minimum: [x, holeY - 1, holeZ - 1] as const,
      maximumExclusive: [x + 1, holeY + 2, holeZ + 2] as const,
    })),
  ];
  // Bound the pool by the reservoir plus a two-page support collar and at
  // least 25% growth space. Small rebuilds also need two overhead page layers
  // for the open-top census. This is page residency, not a claim that every solver
  // scratch allocation already follows the h/4h/16h ownership tiers.
  const initialPages = Math.min(Math.ceil(nx / 32), Math.ceil((low + 64) / 32))
    * Math.ceil(ny / 32) * Math.ceil(nz / 32);
  const growthPages = Math.max(Math.ceil(initialPages / 4), 2 * Math.ceil(nx / 32) * Math.ceil(nz / 32));
  scene.fluid = {
    density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0.001002,
    surfaceTension_N_m: 0.072, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: point(0, 0, 0),
      max_m: point(low, Math.floor(3 * ny / 4), nz) },
    // A submerged passage starts wetted, with its meniscus at the outlet.
    // Overlap the reservoir and continue the liquid a quarter cell into the
    // solid walls. Collision clips that continuation; the throat's boundary
    // vertices must not all start at phi=0 (an unresolved thin free sheet).
    { shape: "box", min_m: point(low - 0.25, holeY - 0.25, holeZ - 0.25),
      max_m: point(high, holeY + 1.25, holeZ + 1.25) }],
    pagePool: { blocks: initialPages + growthPages + 2 },
    // Pin only the aperture's immediate neighbourhood. Dynamic detail can
    // follow the escaping jet; quiet water is free to coarsen everywhere else.
    refinementRegions: [{ id: "seep-throat", rule: "minimum-cell-size",
      min_m: point(low - 4, holeY - 4, holeZ - 4),
      max_m: point(high + 4, holeY + 8, holeZ + 8),
      minimumCellSize_cells: 1, maximumCellSize_cells: 1 }],
  };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.surfaceStyle = "smooth";
  return scene;
}
