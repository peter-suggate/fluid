import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { bakeProceduralSolid } from "./procedural-solid";
import { quaternionInverseRotate, quaternionRotate } from "./rigid-body";
import { solidVoxelPatchesKey } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { backyardPoolPartyScenery } from "./backyard-pool-party-scenery";
import { interactionCatalog } from "../interaction/installed";

export const BACKYARD_POOL_PARTY_ID = "backyard-pool-party";
export const POOL_PARTY_CELL_M = 0.05;
export const POOL_PARTY_INTERACTIVE_CELL_M = .125;
export const POOL_PARTY_GRID = [320, 160, 240] as const;
export const POOL_PARTY_WATERLINE_M = 1.8;
export const POOL_PARTY_DECK_M = 2;
export const POOL_PARTY_BOARD_ANGLE = -Math.PI / 40; // 4.5° downhill toward +X.
export const POOL_PARTY_BOARD = {
  position: { x: -4.1, y: 5.15, z: 0 },
  dimensions: { x: 4, y: 0.16, z: 0.8 },
  orientation: { w: Math.cos(POOL_PARTY_BOARD_ANGLE / 2), x: 0, y: 0, z: Math.sin(POOL_PARTY_BOARD_ANGLE / 2) },
};

/** Fractional SolidWorld voxels retain the plane's signed distance and slope.
 * A small local bake avoids traversing the whole backyard's empty air. */
export function poolPartyBoardGeometry(h = POOL_PARTY_CELL_M) {
  const board = POOL_PARTY_BOARD;
  const dimensions = [16, 8, 12].map(length => 8 * Math.ceil(length / (8 * h) - 1e-8)) as [number, number, number];
  // Align the little bake to complete bricks of the scene lattice.
  const offset = [1.6, 4, 4.8].map(m => 8 * Math.floor(m / (8 * h) + 1e-8));
  const origin = { x: -8 + offset[0]! * h, y: offset[1]! * h, z: -6 + offset[2]! * h };
  const size = [6.4, 6.4, 7.2].map((high, axis) =>
    8 * Math.ceil((high / h - offset[axis]!) / 8 - 1e-8)) as [number, number, number];
  const baked = bakeProceduralSolid(size, [h, h, h], ([x, y, z]) => {
    const p = quaternionInverseRotate(board.orientation, {
      x: origin.x + x * h - board.position.x,
      y: origin.y + y * h - board.position.y,
      z: origin.z + z * h - board.position.z,
    });
    const q = [Math.abs(p.x) - board.dimensions.x / 2,
      Math.abs(p.y) - board.dimensions.y / 2, Math.abs(p.z) - board.dimensions.z / 2];
    return (Math.hypot(...q.map(v => Math.max(0, v))) + Math.min(0, Math.max(...q))) / h;
  }, VOXEL_MATERIAL_IDS.white);
  const solidVoxels = baked.solidVoxels.map(patch => ({ ...patch,
    minimum: patch.minimum.map((v, axis) => v + offset[axis]!) as [number, number, number],
    maximumExclusive: patch.maximumExclusive.map((v, axis) => v + offset[axis]!) as [number, number, number],
  }));
  return { solidVoxels, solidBoundary: { ...baked.solidBoundary,
    dimensions, patchesKey: solidVoxelPatchesKey(solidVoxels),
    bricks: baked.solidBoundary.bricks.map(brick => ({ ...brick,
      origin: brick.origin.map((v, axis) => v + offset[axis]!) as [number, number, number],
    })),
  } };
}

/** A real-scale 8 × 4 m basin with a controllable poolside robot. */
export function createBackyardPoolPartyScene(cellSize_m = POOL_PARTY_CELL_M): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = BACKYARD_POOL_PARTY_ID;
  scene.randomSeed = 20261017;
  scene.duration_s = 16;
  scene.container = { width_m: 16, height_m: 8, depth_m: 12, fillFraction: 0,
    top: "open", fluidWallMode: "no-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: cellSize_m, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: cellSize_m };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = cellSize_m === POOL_PARTY_INTERACTIVE_CELL_M ? 1 / 24 : 1 / 60;
  scene.fluid = { density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0,
    surfaceTension_N_m: 0, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -4, y: 0.2, z: -2 },
      max_m: { x: 4, y: POOL_PARTY_WATERLINE_M, z: 2 } }],
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.47, .095, .046], roughness: .085, windRippleSlope: .003, causticStrength: .25 },
  };
  // Basin bounds in metres from the domain's lower X/Z corner. The four
  // deck slabs surround an empty cavity, never a decorative floor over water.
  scene.solidVoxels = [
    [[0, -Math.max(.1, cellSize_m), 0], [16, 0, 12]],
    [[4, 0, 4], [12, .2, 8]],
    [[0, 0, 0], [4, 2, 12]], [[12, 0, 0], [16, 2, 12]],
    [[4, 0, 0], [12, 2, 4]], [[4, 0, 8], [12, 2, 12]],
  ].map(([minimum, maximumExclusive]) => ({ operation: "fill" as const,
    minimum: minimum!.map(v => Math.round(v / cellSize_m)) as [number, number, number],
    maximumExclusive: maximumExclusive!.map(v => Math.round(v / cellSize_m)) as [number, number, number],
    materialId: VOXEL_MATERIAL_IDS.white }));
  const geometry = poolPartyBoardGeometry(cellSize_m);
  scene.solidVoxels = [...geometry.solidVoxels, ...scene.solidVoxels];
  // The native board, masonry and tile supply the visible boundary. The slab
  // voxels lie below the board field and remain the solver's exact authority.
  scene.solidBoundary = { ...geometry.solidBoundary, patchCount: scene.solidVoxels.length,
    patchesKey: solidVoxelPatchesKey(scene.solidVoxels) };
  const board = POOL_PARTY_BOARD, radius = 0.24;
  // Keep the release over the continuous downhill face of the sampled board.
  const releaseX = cellSize_m > .15 ? 1.85 : cellSize_m > .1 ? .6 : -1.45;
  const release = quaternionRotate(board.orientation, { x: releaseX, y: board.dimensions.y / 2 + radius + 0.002, z: 0 });
  const atRest = { linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    friction: 0.5, restitution: 0.08 };
  scene.rigidBodies = [
    { id: "pool-party/ball", name: "Pool ball", shape: "sphere", motion: "dynamic",
      dimensions_m: { x: radius, y: radius, z: radius }, density_kg_m3: 650,
      position_m: { x: board.position.x + release.x, y: board.position.y + release.y, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 }, ...structuredClone(atRest) },
  ];
  scene.scenery = backyardPoolPartyScenery(cellSize_m);
  // These match the eleven stainless treads authored in Blender. The narrow
  // ladder uses a climb assist; the board and treads remain physical solids.
  for (let i = 1; i <= 11; i++) {
    const t = i / 11, x = -4.1 - 2.95 + t * 1.05, y = 2.1 + t * 3.25;
    scene.solidVoxels.push({ operation: 'fill',
      minimum: [Math.floor((x - .095 + 8) / cellSize_m), Math.floor((y - .0225) / cellSize_m), Math.floor(5.6 / cellSize_m + 1e-8)],
      maximumExclusive: [Math.ceil((x + .095 + 8) / cellSize_m), Math.ceil((y + .04) / cellSize_m), Math.ceil(6.4 / cellSize_m - 1e-8)],
      materialId: VOXEL_MATERIAL_IDS.white });
  }
  scene.interaction = { schemaVersion: 1, connections: [],
    entities: [interactionCatalog.createEntity({ kind: 'fluid.robot', id: 'pool-robot', name: 'Bub',
      transform: { position_m: { x: -7.35, y: 2.52, z: 1.1 },
        orientation: { w: Math.SQRT1_2, x: 0, y: Math.SQRT1_2, z: 0 } } })],
    ladders: [{ id: 'diving-ladder', bottom_m: { x: -7.05, y: 2, z: 0 },
      top_m: { x: -5.85, y: 5.4, z: 0 }, width_m: .8 }],
  };
  scene.surfaceStyle = "smooth";
  // Afternoon sun crosses the pool toward the pavilion. The low sky carries
  // soft neutral fill into the eaves and the shaded side of the figures.
  scene.lighting = {
    rasterAoContrast: 1.7,
    directional: { direction: [.604, .745, .298], colorLinear: [1, .88, .71], intensity: 4.65 },
    environment: { diffuseScale: .52, specularScale: 1.0, skyCapture: "kloofendal", cloudCoverage: .45, aerialPerspective_m: 230,
      lowerRadianceLinear: [.21, .24, .20], upperRadianceLinear: [.095, .34, .78],
      accentRadianceLinear: [.49, .57, .64] },
    grade: { toneCurve: "agx", exposure: 1.12, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const BACKYARD_POOL_DRIFT_ID = "backyard-pool-drift";

/** The same pool barely disturbed: the ball floats at its rest depth and
 * drifts slowly across the water, and one small drip lands at the far end.
 * Most of the pool stays close to settled while a gentle wake and weak
 * ripples spread, so work should follow them rather than the whole pool. */
export function createBackyardPoolDriftScene(): SceneDescription {
  const scene = createBackyardPoolPartyScene();
  scene.sceneId = BACKYARD_POOL_DRIFT_ID;
  // Keep the quiet fluid benchmark independent of character input/authority.
  scene.interaction = undefined;
  const ball = scene.rigidBodies[0]!, radius = ball.dimensions_m.x;
  // A 650 kg/m3 sphere floats with 65% of its volume submerged: a cap
  // 1.2 radii deep, so its centre sits 0.2 radii below the waterline.
  Object.assign(ball, { position_m: { x: -2.6, y: POOL_PARTY_WATERLINE_M - 0.2 * radius, z: -0.6 },
    linearVelocity_m_s: { x: 0.3, y: 0, z: 0.05 } });
  scene.fluid.scheduledDrops = [{ time_s: 2, volume: { shape: "sphere",
    center_m: { x: 2.8, y: POOL_PARTY_WATERLINE_M + 0.3, z: 1 }, radius_m: 0.04 } }];
  return scene;
}

export const backyardPoolPartyCamera: Partial<CameraState> = {
  azimuth_rad: .90, elevation_rad: .139, distance_m: 11.4, tanHalfFov: .46,
  target_m: { x: -.35, y: 2.5, z: -.85 },
};
