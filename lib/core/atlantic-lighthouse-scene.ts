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

/** Shared with tools/blender/lighthouse_terrain.py. Application Y is up.
 * Dry geometry samples this sculpt at 6.25 cm; fluid collision uses 25 cm. */
export function atlanticLighthouseTerrainHeight(x: number, z: number): number {
  let height = .15;
  let relief = 0;
  for (const rock of rockLayout) {
    const dx = x - rock.x, dz = z - rock.z, c = Math.cos(rock.yaw), s = Math.sin(rock.yaw);
    const u = (c * dx + s * dz) / rock.rx, v = (-s * dx + c * dz) / rock.rz;
    const q = Math.max(Math.abs(u), Math.abs(v), Math.abs(u + v) * .65, Math.abs(u - v) * .65);
    if (rock.relief !== undefined) {
      relief = Math.max(relief, Math.max(0, Math.min(1, (1 - q) / (1 - rock.core)))
        * Math.max(0, rock.relief + dx * rock.tiltX + dz * rock.tiltZ));
      continue;
    }
    const top = rock.height + dx * rock.tiltX + dz * rock.tiltZ;
    const fall = Math.max(0, Math.min(1, (q - rock.core) / (1 - rock.core)));
    // Full tidal-boulder shoulders, with the larger bedding slabs kept planar.
    const block = rock.height < 3.5
      ? .15 + (top - .15) * Math.pow(Math.max(0, 1 - fall * fall), 1.2)
      : top - Math.max(0, q - rock.core) * rock.height / (1 - rock.core);
    height = Math.max(height, block);
  }
  height += .07 * Math.sin(x * 3.1 + z * 1.7) * Math.sin(z * 2.3 - x * .4) * Math.min(1, Math.max(0, height - .15));
  const dry = Math.min(1, Math.max(0, (height - 1.8) / 1.4));
  const seam = Math.abs(Math.sin(x * 2.7 + z * .83 + .75 * Math.sin(z * 2.1) + .38 * Math.sin(x * 1.3)));
  const cross = Math.abs(Math.sin(z * 3.4 - x * .61 + .6 * Math.sin(x * 2.4)));
  let detail = .18 * Math.sin(x * 2.3 + z * .8 + 1.7 * Math.sin(z * 1.5)) * Math.sin(z * 2.1 - x * .4 + .8 * Math.sin(x * .7));
  detail += .025 * Math.sin(x * 11 + z * 7) * Math.sin(z * 9 - x * 3);
  detail -= .16 * Math.max(0, 1 - seam / .32) + .11 * Math.max(0, 1 - cross / .26);
  height += dry * (detail + relief);
  const r = Math.hypot(x - 7, z);
  let footing = Math.max(0, Math.min(1, (2.2 - r) / .45));
  footing = footing * footing * (3 - 2 * footing);
  return height * (1 - footing) + 5.8 * footing;
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
    { id: "atlantic-lighthouse-tower", base: { x: 7, y: 5.8, z: 0 }, color: [.7, .69, .63] as const },
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
      place: { units: "metres", position: { x: 7, y: 16.3, z: 0 } }, material: { colorLinear: [1, .4, .09], emission: 8 } },
  ] };
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [.7, .25, .67], colorLinear: [1, .43, .13], intensity: 6 },
    environment: { diffuseScale: .6, specularScale: 1, cloudCoverage: .42,
      lowerRadianceLinear: [.11, .15, .21], upperRadianceLinear: [.28, .40, .58],
      accentRadianceLinear: [.8, .38, .12] },
    grade: { toneCurve: "agx", exposure: .8, whiteBalance: [1, 1, 1] },
  };
  return scene;
}

export const atlanticLighthouseCamera: Partial<CameraState> = {
  azimuth_rad: -.65, elevation_rad: -.09, distance_m: 36, tanHalfFov: .34,
  target_m: { x: 8, y: 8.4, z: 0 },
};
