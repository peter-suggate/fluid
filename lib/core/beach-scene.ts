import { cloneScene, defaultScene, type CameraState, type RigidPoseTrack, type SceneDescription } from "./model";
import { meshAssetHeader } from "./imported-assets/asset-library";
import type { SceneryNode } from "./scenery-graph";
import type { TerrainDescription } from "./terrain";
import authoredTerrain from "./imported-assets/beach-terrain.json";
import authoredSeaweed from "./imported-assets/beach-seaweed.generated.json";

export const BEACH_SCENE_ID = "beach";
export const BEACH_CELL_M = .125;
export const BEACH_WATERLINE_M = 2.4;
export const BEACH_DURATION_S = 16;
export const BEACH_WAVE_PERIOD_S = 3.2;
export const BEACH_PISTON_AMPLITUDE_M = .55;
export const BEACH_WAVE_RAMP_S = .8;
const PISTON_X = -14.5;
const orientation = { w: 1, x: 0, y: 0, z: 0 };

/** A ramped piston launches a finite wave train into initially still seawater.
 * Only the solid boundary is prescribed; NB-FLIP owns all water motion.
 * A short smooth startup makes surf visible during the opening seconds. */
export function beachPistonPosition(time_s: number): number {
  const time = Math.max(0, Math.min(BEACH_DURATION_S, time_s));
  const ramp = Math.min(1, time / BEACH_WAVE_RAMP_S, (BEACH_DURATION_S - time) / BEACH_WAVE_PERIOD_S);
  const envelope = .5 - .5 * Math.cos(Math.PI * ramp);
  return PISTON_X + BEACH_PISTON_AMPLITUDE_M * envelope * Math.sin(2 * Math.PI * time / BEACH_WAVE_PERIOD_S);
}

export function beachPistonTrack(): RigidPoseTrack {
  return { keyframes: Array.from({ length: BEACH_DURATION_S * 60 + 1 }, (_, i) => ({
    time_s: i / 60, position_m: { x: beachPistonPosition(i / 60), y: 2, z: 0 }, orientation: { ...orientation },
  })) };
}

export function beachTerrain(): TerrainDescription {
  return { baseHeight_m: .35, features: [], grid: { ...structuredClone(authoredTerrain), kind: "grid" } };
}

function beachScenery(): SceneryNode[] {
  return [
    { asset: "beach-sand", x: 0, y: 0, z: 0, yaw: 0, scale: 1 },
    { asset: "beach-dunes", x: 0, y: 0, z: 0, yaw: 0, scale: 1 },
    { asset: "beach-rocks", x: 0, y: 0, z: 0, yaw: 0, scale: 1 },
    ...([[11.3,-5.8,0,1],[12.9,1.8,2.2,1],[12.1,7.8,-.8,1]] as const).map(([x,z,yaw,scale]) => {
      const grid = authoredTerrain;
      const ix = Math.round((x! + 16) / grid.spacing_m), iz = Math.round((z! + 12) / grid.spacing_m);
      return { asset: "beach-palm", x, z, yaw, scale, y: grid.heights_m[iz * grid.size.nx + ix]! };
    }),
  ].map(({ asset, x, y, z, yaw, scale }, index) => {
    const h = meshAssetHeader(asset);
    const offset = h.origin.map((v, i) => v + h.dimensions[i]! * h.cellSize / 2);
    return { kind: "field-program", id: `beach/${asset}/${index}`, tags: ["blender-beach", "static-environment"],
      place: { units: "metres", scale, orientation: { w: Math.cos(yaw / 2), x: 0, y: Math.sin(yaw / 2), z: 0 },
        position: { x: x + scale * (Math.cos(yaw) * offset[0]! + Math.sin(yaw) * offset[2]!),
          y: y + scale * offset[1]!, z: z + scale * (-Math.sin(yaw) * offset[0]! + Math.cos(yaw) * offset[2]!) } },
      program: { ops: [], result: 0, sampledVolume: { asset } },
      material: { colorLinear: [.4, .3, .18], surface: "stone" },
    } satisfies SceneryNode;
  });
}

export function createBeachScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = BEACH_SCENE_ID;
  scene.randomSeed = 20261101;
  scene.duration_s = BEACH_DURATION_S;
  scene.systems = { fluid: true };
  scene.container = { width_m: 32, height_m: 16, depth_m: 24, fillFraction: 0,
    top: "open", fluidWallMode: "free-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: BEACH_CELL_M, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: BEACH_CELL_M };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.fluid = {
    density_kg_m3: 1025, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 }, initialCondition: "tank-fill",
    initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [{ shape: "box", min_m: { x: -14, y: .35, z: -12 },
      max_m: { x: 16, y: BEACH_WATERLINE_M, z: 12 } }],
    // 8 × 4 × 6 domain pages, plus the pool's two sentinels. The tall
    // coastal scene must also admit its terrain/moving-solid seed pages.
    pagePool: { blocks: 194 },
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.22, .065, .035], roughness: .055 },
  };
  scene.terrain = beachTerrain();
  scene.solidVoxels = [];
  delete scene.solidBoundary;
  scene.rigidBodies = [{
    id: "beach-wave-piston", name: "Offshore wave generator", shape: "box", motion: "static",
    dimensions_m: { x: 1, y: 4, z: 24 }, position_m: { x: PISTON_X, y: 2, z: 0 },
    orientation: { ...orientation }, linearVelocity_m_s: { x: 0, y: 0, z: 0 },
    angularVelocity_rad_s: { x: 0, y: 0, z: 0 }, density_kg_m3: 7800,
    restitution: 0, friction: 0, renderHidden: true, poseTrack: beachPistonTrack(),
  }];
  // Resolve the wet proxy with at least three fluid cells. The old thin,
  // very light boxes received unstable pressure impulses at startup.
  const weedThickness = authoredSeaweed.buoyancyProxy.dimensions_m.y, weedDensity = authoredSeaweed.buoyancyProxy.density_kg_m3;
  const weedHeight = BEACH_WATERLINE_M + weedThickness * (.5 - weedDensity / scene.fluid.density_kg_m3);
  for (const [index, [x,z]] of [[.5,-3],[-1.8,1.2],[.4,3.5],[-3.5,-4.5]].entries()) {
    scene.rigidBodies.push({
      id: `beach-seaweed-${index}`, name: "Floating seaweed", shape: "box", appearance: "seaweed", motion: "dynamic",
      dimensions_m: { ...authoredSeaweed.buoyancyProxy.dimensions_m }, density_kg_m3: weedDensity,
      position_m: { x: x!, y: weedHeight, z: z! },
      orientation: { w: Math.cos(index * .65), x: 0, y: Math.sin(index * .65), z: 0 },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
      restitution: .05, friction: .6,
    });
  }
  scene.scenery = { palettes: {}, terrainSurface: "scenery", nodes: [
    { kind: "terrain-shell", id: "beach/collision-terrain", materialModel: "forest-terrain", waterline_m: BEACH_WATERLINE_M },
    ...beachScenery(),
  ] };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-.62, .41, -.67], colorLinear: [1, .86, .65], intensity: 4.2 },
    environment: { diffuseScale: .65, specularScale: 1, cloudCoverage: .55,
      lowerRadianceLinear: [.04, .13, .30], upperRadianceLinear: [.025, .16, .54],
      accentRadianceLinear: [.20, .30, .42] },
    grade: { toneCurve: "agx", exposure: 1, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const beachCamera: Partial<CameraState> = {
  azimuth_rad: -1.2341, elevation_rad: .07066, distance_m: 21.2426, tanHalfFov: .42,
  target_m: { x: 10, y: 4, z: 0 },
};
