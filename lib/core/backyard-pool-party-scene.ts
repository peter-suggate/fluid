import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { bakeProceduralSolid } from "./procedural-solid";
import { quaternionInverseRotate, quaternionRotate } from "./rigid-body";
import { solidVoxelPatchesKey } from "./solid-world";
import { VOXEL_MATERIAL_IDS } from "./voxel-scene";
import { backyardPoolPartyScenery } from "./backyard-pool-party-scenery";

export const BACKYARD_POOL_PARTY_ID = "backyard-pool-party";
export const POOL_PARTY_CELL_M = 0.05;
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
export function poolPartyBoardGeometry() {
  const h = POOL_PARTY_CELL_M, board = POOL_PARTY_BOARD;
  const origin = { x: -6.4, y: 4.0, z: -1.2 };
  const baked = bakeProceduralSolid([96, 48, 48], [h, h, h], ([x, y, z]) => {
    const p = quaternionInverseRotate(board.orientation, {
      x: origin.x + x * h - board.position.x,
      y: origin.y + y * h - board.position.y,
      z: origin.z + z * h - board.position.z,
    });
    const q = [Math.abs(p.x) - board.dimensions.x / 2,
      Math.abs(p.y) - board.dimensions.y / 2, Math.abs(p.z) - board.dimensions.z / 2];
    return (Math.hypot(...q.map(v => Math.max(0, v))) + Math.min(0, Math.max(...q))) / h;
  }, VOXEL_MATERIAL_IDS.white);
  const offset = [32, 80, 96];
  const solidVoxels = baked.solidVoxels.map(patch => ({ ...patch,
    minimum: patch.minimum.map((v, axis) => v + offset[axis]!) as [number, number, number],
    maximumExclusive: patch.maximumExclusive.map((v, axis) => v + offset[axis]!) as [number, number, number],
  }));
  return { solidVoxels, solidBoundary: { ...baked.solidBoundary,
    dimensions: POOL_PARTY_GRID, patchesKey: solidVoxelPatchesKey(solidVoxels),
    bricks: baked.solidBoundary.bricks.map(brick => ({ ...brick,
      origin: brick.origin.map((v, axis) => v + offset[axis]!) as [number, number, number],
    })),
  } };
}

/** A real-scale 8 × 4 m basin. Only the ball is a rigid body;
 * the watertight basin is SolidWorld geometry, and the party is scenery. */
export function createBackyardPoolPartyScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = BACKYARD_POOL_PARTY_ID;
  scene.randomSeed = 20261017;
  scene.duration_s = 16;
  scene.container = { width_m: 16, height_m: 8, depth_m: 12, fillFraction: 0,
    top: "open", fluidWallMode: "no-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: POOL_PARTY_CELL_M, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: POOL_PARTY_CELL_M };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.fluid = { density_kg_m3: 998.2, dynamicViscosity_Pa_s: 0.001002,
    surfaceTension_N_m: 0.072, gravity_m_s2: { x: 0, y: -9.80665, z: 0 },
    initialCondition: "tank-fill", initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -4, y: 0.2, z: -2 },
      max_m: { x: 4, y: POOL_PARTY_WATERLINE_M, z: 2 } }],
    optics: { indexOfRefraction: 1.333, absorption_mInv: [0.16, 0.045, 0.025], roughness: 0.04 },
  };
  // Basin bounds in metres from the domain's lower X/Z corner. The four
  // deck slabs surround an empty cavity, never a decorative floor over water.
  scene.solidVoxels = [
    [[0, -.1, 0], [16, 0, 12]],
    [[4, 0, 4], [12, .2, 8]],
    [[0, 0, 0], [4, 2, 12]], [[12, 0, 0], [16, 2, 12]],
    [[4, 0, 0], [12, 2, 4]], [[4, 0, 8], [12, 2, 12]],
  ].map(([minimum, maximumExclusive]) => ({ operation: "fill" as const,
    minimum: minimum!.map(v => Math.round(v / POOL_PARTY_CELL_M)) as [number, number, number],
    maximumExclusive: maximumExclusive!.map(v => Math.round(v / POOL_PARTY_CELL_M)) as [number, number, number],
    materialId: VOXEL_MATERIAL_IDS.white }));
  const geometry = poolPartyBoardGeometry();
  scene.solidVoxels = [...geometry.solidVoxels, ...scene.solidVoxels];
  scene.solidBoundary = geometry.solidBoundary;
  const board = POOL_PARTY_BOARD, radius = 0.24;
  const release = quaternionRotate(board.orientation, { x: -1.45, y: board.dimensions.y / 2 + radius + 0.002, z: 0 });
  const atRest = { linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
    friction: 0.5, restitution: 0.08 };
  scene.rigidBodies = [
    { id: "pool-party/ball", name: "Pool ball", shape: "sphere", motion: "dynamic",
      dimensions_m: { x: radius, y: radius, z: radius }, density_kg_m3: 650,
      position_m: { x: board.position.x + release.x, y: board.position.y + release.y, z: 0 },
      orientation: { w: 1, x: 0, y: 0, z: 0 }, ...structuredClone(atRest) },
  ];
  scene.scenery = backyardPoolPartyScenery();
  scene.surfaceStyle = "smooth";
  // The asset reference uses AgX under a 38-degree afternoon sun. Strong
  // direct daylight, cool sky and warm terrace bounce retain pale textiles
  // and warm teak without baking illumination into their PBR base colours.
  scene.lighting = {
    directional: { direction: [-0.452, 0.616, -0.646], colorLinear: [1, 0.96, 0.90], intensity: 10.0 },
    environment: { diffuseScale: 1.0, specularScale: 1.0,
      lowerRadianceLinear: [0.80, 0.72, 0.60], upperRadianceLinear: [0.82, 0.95, 1.12],
      accentRadianceLinear: [0.8, 0.85, 0.9] },
    grade: { toneCurve: "agx", exposure: 2.0, whiteBalance: [1, 1, 1] },
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
  azimuth_rad: -0.65, elevation_rad: 0.55, distance_m: 25, tanHalfFov: 0.48,
  target_m: { x: -0.8, y: 2.4, z: 0 },
};
