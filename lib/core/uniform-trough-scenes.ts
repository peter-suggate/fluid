import { cloneScene, defaultScene, type SceneDescription } from "./model";
import { sceneLatticeDimensions } from "./scene-lattice-dimensions";
import { bathInteriorContains, bathSolidContains, bathVoxelBoxes, BATH_FLOOR_HEIGHT_M } from "./voxel-bath";

export type UniformTroughMode = "dam-break" | "settled-tank" | "hose-fill";

/** Matched 3D vessels for transport, hydrostatics, and source growth studies. */
export function createUniformTroughScene(
  mode: UniformTroughMode,
  cellSize_m = mode === "settled-tank" ? 0.0125 : 0.025,
): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = `uniform-trough-${mode}`;
  scene.duration_s = 20;
  scene.rigidBodies = [];
  scene.container = {
    ...scene.container,
    shape: "box",
    vessel: "none",
    width_m: 3.2,
    height_m: 1.2,
    depth_m: 1.2,
    top: "open",
    depthBoundary: "closed",
    fluidWallMode: "free-slip",
    // Liquid is seeded only inside the curved cavity, never outside the bath.
    fillFraction: 0,
  };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  const dimensions = sceneLatticeDimensions(scene);
  const [nx, ny, nz] = dimensions;
  const cell = [3.2 / nx, 1.2 / ny, 1.2 / nz] as const;
  const point = (x: number, y: number, z: number) =>
    [(x + 0.5) * cell[0] - 1.6, (y + 0.5) * cell[1], (z + 0.5) * cell[2] - 0.6] as const;
  scene.solidVoxels = bathVoxelBoxes(dimensions, (x, y, z) => bathSolidContains(...point(x, y, z)));
  scene.fluid = {
    density_kg_m3: 998.2,
    dynamicViscosity_Pa_s: 0,
    surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: mode === "dam-break" ? "dam-break" : "tank-fill",
  };
  if (mode !== "hose-fill") {
    const interior = (x: number, y: number, z: number) => bathInteriorContains(...point(x, y, z));
    const dam = (x: number, y: number, z: number) => {
      const p = point(x, y, z);
      return interior(x, y, z) && p[0] < -0.65 && p[1] < BATH_FLOOR_HEIGHT_M + 0.8;
    };
    // Find the flat waterline holding exactly the dam's rasterized volume.
    let damCells = 0;
    const layerCells = new Array<number>(ny).fill(0);
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      if (interior(x, y, z)) layerCells[y]++;
      if (dam(x, y, z)) damCells++;
    }
    let remaining = damCells;
    let waterline = BATH_FLOOR_HEIGHT_M;
    for (let y = 0; y < ny; y++) {
      const count = layerCells[y]!;
      if (count === 0) continue;
      if (remaining <= count) { waterline = (y + remaining / count) * cell[1]; break; }
      remaining -= count;
    }
    const boxes = bathVoxelBoxes(dimensions, mode === "dam-break" ? dam
      : (x, y, z) => interior(x, y, z) && y * cell[1] < waterline);
    // Overlap internal box faces so their SDF union has no zero-valued seams.
    // Solid walls clip the quarter-cell overlap along the cavity boundary,
    // while the actual free faces stay exact.
    scene.fluid.initialLiquidVolumes = boxes.map(box => ({
      shape: "box",
      min_m: { x: (box.minimum[0] - 0.25) * cell[0] - 1.6,
        y: Math.max(BATH_FLOOR_HEIGHT_M, (box.minimum[1] - 0.25) * cell[1]),
        z: (box.minimum[2] - 0.25) * cell[2] - 0.6 },
      max_m: { x: Math.min(mode === "dam-break" ? -0.65 : Infinity, (box.maximumExclusive[0] + 0.25) * cell[0] - 1.6),
        y: Math.min(mode === "settled-tank" ? waterline : BATH_FLOOR_HEIGHT_M + 0.8,
          (box.maximumExclusive[1] + 0.25) * cell[1]),
        z: (box.maximumExclusive[2] + 0.25) * cell[2] - 0.6 },
    }));
  } else {
    scene.fluid.inflow = {
      center_m: { x: -1.1, y: 0.9 + BATH_FLOOR_HEIGHT_M, z: 0 },
      radius_m: 0.1,
      length_m: 0.15,
      velocity_m_s: { x: 1.5, y: 0, z: 0 },
      start_s: 0,
      end_s: 12,
      ramp_s: 0.5,
    };
  }
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  return scene;
}
