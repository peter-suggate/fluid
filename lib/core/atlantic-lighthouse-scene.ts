import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import type { TerrainDescription } from "./terrain";
import { meshAssetHeader } from "./imported-assets/asset-library";
import type { SceneryNode } from "./scenery-graph";
import type { SolidWorldVoxelPatch } from "./solid-world";
import type { ProceduralSolidBoundary } from "./procedural-solid";
import towerCollider from "./imported-assets/metadata/atlantic-lighthouse-collider.json";
import rockLayout from "./imported-assets/atlantic-lighthouse-rock-layout.json";

export const ATLANTIC_LIGHTHOUSE_ID = "atlantic-lighthouse";
export const ATLANTIC_LIGHTHOUSE_CELL_M = .25;
export const ATLANTIC_LIGHTHOUSE_GRID = [192, 96, 192] as const;
export const ATLANTIC_LIGHTHOUSE_WATERLINE_M = 1.6;

/** Shared with tools/blender/atlantic_lighthouse.py. Application Y is up.
 * The visible rock is the collision heightfield, including its broken shelves. */
export function atlanticLighthouseTerrainHeight(x: number, z: number): number {
  let height = .15;
  for (const rock of rockLayout) {
    const dx = x - rock.x, dz = z - rock.z, c = Math.cos(rock.yaw), s = Math.sin(rock.yaw);
    const u = (c * dx + s * dz) / rock.rx, v = (-s * dx + c * dz) / rock.rz;
    const q = Math.max(Math.abs(u), Math.abs(v), Math.abs(u + v) * .65, Math.abs(u - v) * .65);
    const top = rock.height + dx * rock.tiltX + dz * rock.tiltZ;
    height = Math.max(height, top - Math.max(0, q - rock.core) * rock.height / (1 - rock.core));
  }
  height += .07 * Math.sin(x * 3.1 + z * 1.7) * Math.sin(z * 2.3 - x * .4) * Math.min(1, Math.max(0, height - .15));
  return Math.hypot(x - 7, z) < 1.55 ? 5.8 : height;
}

export function atlanticLighthouseTerrain(): TerrainDescription {
  const spacing_m = .25, nx = 161, nz = 129;
  return { baseHeight_m: .15, features: [], grid: {
    kind: "grid", origin_m: { x: -20, z: -16 }, spacing_m, size: { nx, nz },
    heights_m: Array.from({ length: nx * nz }, (_, i) =>
      atlanticLighthouseTerrainHeight(-20 + i % nx * spacing_m, -16 + Math.floor(i / nx) * spacing_m)),
  } };
}

/** FVX assets are centred on their sampled bounds; source GLB origins are
 * the tower footing and the island's centre. Preserve that offset explicitly. */
export function atlanticLighthouseImportedNodes(): SceneryNode[] {
  return [
    { id: "atlantic-lighthouse-rock", base: { x: 7, y: 0, z: 0 }, color: [.055, .065, .075] as const },
    { id: "atlantic-lighthouse-tower", base: { x: 7, y: 5.8, z: 0 }, color: [.31, .30, .25] as const },
  ].map(({ id, base, color }) => {
    const header = meshAssetHeader(id);
    const offset = header.origin.map((value, axis) => value + header.dimensions[axis]! * header.cellSize / 2);
    return { kind: "field-program", id: `atlantic/${id}`, tags: ["blender-atlantic", "static-environment"],
      place: { units: "metres", position: { x: base.x + offset[0]!, y: base.y + offset[1]!, z: base.z + offset[2]! } },
      program: { ops: [], result: 0, sampledVolume: { asset: id } },
      material: { colorLinear: color, surface: "stone" },
    } satisfies SceneryNode;
  });
}

/** A rounded incoming bore accelerates over the submerged rock shelf. Every
 * water surface and splash is produced by the registered narrow-band FLIP
 * method; the Blender stage imports that simulation's exported OBJ. */
export function createAtlanticLighthouseScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = ATLANTIC_LIGHTHOUSE_ID;
  scene.randomSeed = 20261028;
  scene.duration_s = 2.5;
  scene.systems = { fluid: true };
  scene.container = { width_m: 48, height_m: 24, depth_m: 48, fillFraction: 0,
    top: "open", fluidWallMode: "free-slip", vessel: "none" };
  scene.voxelDomain = { finestCellSize_m: ATLANTIC_LIGHTHOUSE_CELL_M, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: ATLANTIC_LIGHTHOUSE_CELL_M };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.fluid = {
    density_kg_m3: 1025, dynamicViscosity_Pa_s: 0, surfaceTension_N_m: 0,
    gravity_m_s2: { x: 0, y: -9.80665, z: 0 }, initialCondition: "tank-fill",
    initialVelocity_m_s: { x: 0, y: 0, z: 0 },
    initialLiquidVolumes: [
      { shape: "box", min_m: { x: -24, y: .15, z: -24 }, max_m: { x: 24, y: 1.6, z: 24 } },
      { shape: "cylinder", center_m: { x: -7, y: 1, z: 0 }, radius_m: 5.5, halfHeight_m: 9 },
    ],
    // The current solver requires bounded terrain domains. Remote boundaries
    // stand well beyond the island for this short coastal impact shot.
    pagePool: { blocks: 146 },
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.26, .075, .052], roughness: .06 },
  };
  scene.terrain = atlanticLighthouseTerrain();
  scene.solidVoxels = structuredClone(towerCollider.solidVoxels) as unknown as SolidWorldVoxelPatch[];
  scene.solidBoundary = structuredClone(towerCollider.solidBoundary) as unknown as ProceduralSolidBoundary;
  scene.rigidBodies = [];
  scene.scenery = { palettes: {}, terrainSurface: "scenery", solidBoundarySurface: "scenery", nodes: [
    { kind: "terrain-shell", id: "atlantic/seabed", materialModel: "forest-terrain", waterline_m: 1.6 },
    { kind: "box", id: "atlantic/seabed-rock", halfSize: { x: 24, y: .15, z: 24 },
      place: { units: "metres", position: { x: 0, y: 0, z: 0 } },
      material: { colorLinear: [.045, .052, .06], surface: "stone" } },
    ...atlanticLighthouseImportedNodes(),
    { kind: "cylinder", id: "atlantic/lantern-light", radius: .21, halfHeight: .32,
      place: { units: "metres", position: { x: 7, y: 15.87, z: 0 } }, material: { colorLinear: [1, .4, .09], emission: 8 } },
  ] };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-.55, .63, .55], colorLinear: [1, .86, .68], intensity: 4 },
    environment: { diffuseScale: .7, specularScale: 1, cloudCoverage: .85, lowerRadianceLinear: [.035, .055, .075], upperRadianceLinear: [.13, .18, .24] },
    grade: { toneCurve: "agx", exposure: .8, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const atlanticLighthouseCamera: Partial<CameraState> = {
  azimuth_rad: -.65, elevation_rad: -.09, distance_m: 32, tanHalfFov: .34,
  target_m: { x: 2, y: 8.4, z: 0 },
};
