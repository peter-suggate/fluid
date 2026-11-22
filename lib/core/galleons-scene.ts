import { meshAssetHeader } from "./imported-assets/asset-library";
import collision from "./imported-assets/galleon-collision.json";
import { galleonAsset, galleonOcean as ocean, galleons } from "./galleon-layout";
import { cloneScene, defaultScene, type CameraState, type SceneDescription } from "./model";
import { solidVoxelPatchesKey, type SolidWorldVoxelPatch } from "./solid-world";

export const galleonsCamera: Partial<CameraState> = {
  azimuth_rad: .66, elevation_rad: -.095, distance_m: 80, tanHalfFov: .41, target_m: { x: 0, y: 23, z: 0 },
};

export function createGalleonsScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.sceneId = "galleon-broadside";
  scene.duration_s = 30;
  scene.systems = { fluid: true };
  scene.container = { width_m: ocean.width_m, height_m: ocean.height_m, depth_m: ocean.depth_m,
    fillFraction: ocean.waterline_m / ocean.height_m, top: "open", vessel: "none", fluidWallMode: "free-slip" };
  scene.voxelDomain = { finestCellSize_m: ocean.cell_m, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: ocean.cell_m };
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 60;
  scene.fluid = { initialCondition: "tank-fill", density_kg_m3: 1025, dynamicViscosity_Pa_s: 0,
    surfaceTension_N_m: 0, initialVelocity_m_s: { x: 0, y: 0, z: 0 }, gravity_m_s2: { x: 0, y: -9.81, z: 0 },
    initialHeightField: { kind: "cosine", baseHeight_m: ocean.waterline_m, amplitude_m: 1.35, wavelength_m: 32, originX_m: 5 },
    optics: { indexOfRefraction: 1.333, absorption_mInv: [.48, .16, .095], roughness: .055 } };
  delete scene.terrain;
  scene.solidVoxels = structuredClone(collision) as unknown as SolidWorldVoxelPatch[];
  scene.solidBoundary = { version: 1, dimensions: [ocean.width_m / ocean.cell_m, ocean.height_m / ocean.cell_m, ocean.depth_m / ocean.cell_m],
    cellSize_m: [ocean.cell_m, ocean.cell_m, ocean.cell_m], patchesKey: solidVoxelPatchesKey(scene.solidVoxels),
    patchCount: scene.solidVoxels.length, collision: "voxels", bricks: [] };
  const asset = meshAssetHeader(galleonAsset);
  const center = asset.origin.map((v, axis) => v + asset.dimensions[axis]! * asset.cellSize / 2);
  scene.scenery = {
    palettes: {},
    solidBoundarySurface: "scenery",
    nodes: [
      { kind: "terrain-shell", id: "seabed", materialModel: "forest-terrain", waterline_m: ocean.waterline_m },
      ...galleons.map(ship => ({
        id: ship.id,
        kind: "field-program" as const,
        place: {
          units: "metres" as const,
          position: {
            x: ship.x + ship.heading * center[0]!,
            y: ocean.waterline_m + center[1]!,
            z: ship.z + ship.heading * center[2]!,
          },
          orientation: ship.heading === 1 ? { w: 1, x: 0, y: 0, z: 0 } : { w: 0, x: 0, y: 1, z: 0 },
        },
        program: { ops: [], result: 0, sampledVolume: { asset: galleonAsset } },
        material: { surface: "wood" as const, colorLinear: [.16, .082, .035] as const },
      })),
    ],
  };
  scene.rigidBodies = galleons.flatMap(ship => [0, 1, 2].map(gun => {
    const inward = -Math.sign(ship.x);
    return { id: `${ship.id}-shot-${gun}`, name: `${ship.id === "jackdaw" ? "Jackdaw" : "Kestrel"} · shot ${gun + 1}`,
      shape: "sphere" as const, motion: "static" as const, kinematic: true, dimensions_m: { x: .28, y: .28, z: .28 },
      density_kg_m3: 7200, restitution: .04, friction: .5, orientation: { w: 1, x: 0, y: 0, z: 0 },
      position_m: { x: ship.x + inward * 5.65, y: ocean.waterline_m + 3.92, z: ship.z + ship.heading * (7.2 - 4.8 * gun) },
      linearVelocity_m_s: { x: 0, y: 0, z: 0 }, angularVelocity_rad_s: { x: 0, y: 0, z: 0 },
      launcher: { start_s: .18 + .36 * gun + (ship.heading === 1 ? .76 : 0), period_s: 4.5,
        velocity_m_s: { x: inward * 24, y: 1.35 + gun * .55, z: -ship.z * .55 }, cutRadius_m: 1.45 },
    };
  }));
  scene.surfaceStyle = "smooth";
  scene.lighting = {
    directional: { direction: [-.68, .42, .60], colorLinear: [1, .83, .62], intensity: 3.6 },
    environment: { skyCapture: "kloofendal", diffuseScale: .86, specularScale: 1, cloudCoverage: .54,
      lowerRadianceLinear: [.10, .15, .18], upperRadianceLinear: [.32, .47, .63], accentRadianceLinear: [.76, .56, .35] },
    grade: { toneCurve: "agx", exposure: 1, whiteBalance: [1, 1, 1] },
  };
  return scene;
}
