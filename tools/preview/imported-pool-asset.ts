/** Production SVO preview of a Blender GLB after sparse mesh voxelization.
 * FLUID_POOL_ASSET selects a manifest ID; default is teak-lounger.
 * Uses the live scenery field, UV-sampled PBR palette and source normals. */
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import type { CameraState } from "../../lib/core/model";
import { decodeMeshVolume } from "../../lib/core/imported-assets/mesh-volume";
import { MESH_ASSETS, registerMeshAsset } from "../../lib/core/imported-assets/asset-library";
import { getScenePreset } from "../../lib/core/scenes";

const id = process.env.FLUID_POOL_ASSET ?? "teak-lounger";
if (!Object.hasOwn(MESH_ASSETS, id)) throw new Error("Unknown pool asset");
const volume = decodeMeshVolume(gunzipSync(readFileSync(new URL(`../../public/assets/pool-party/${id}.fvx.gz`, import.meta.url))));
registerMeshAsset(id, volume);
const h = volume.header.cellSize, [nx, ny, nz] = volume.header.dimensions;
export function createScene() {
  const scene = getScenePreset("backyard-pool-party").create();
  scene.sceneId = `imported-${id}`;
  scene.systems = { ...scene.systems, fluid: false };
  scene.container = { width_m: nx * h, height_m: ny * h, depth_m: nz * h, fillFraction: 0,
    top: "open", vessel: "none", fluidWallMode: "no-slip" };
  scene.voxelDomain = { finestCellSize_m: h, detailCellSize_m: h, brickSize_cells: 8 };
  scene.nominalResolution = { length_m: h };
  scene.terrain = undefined;
  scene.backdrop = undefined;
  scene.scenery = { palettes: {}, nodes: [{ kind: "room-shell", id: "empty-shell", faces: [], materialModel: "room",
    floor: { colorLinear: [.7, .7, .7] }, wall: { colorLinear: [.7, .7, .7] }, ceiling: { colorLinear: [.7, .7, .7] } },
    { kind: "field-program", id, program: { ops: [], result: 0, sampledVolume: { asset: id } },
      place: { units: "metres", position: { x: 0, y: ny * h / 2, z: 0 } },
      material: { colorLinear: [.7, .7, .7] } }] };
  scene.rigidBodies = [];
  scene.fluid = { ...scene.fluid, initialLiquidVolumes: [], inflow: undefined };
  scene.solidVoxels = [];
  scene.solidBoundary = undefined;
  scene.surfaceStyle = "smooth";
  return scene;
}
export const camera: Partial<CameraState> = { azimuth_rad: -.7, elevation_rad: .48,
  distance_m: Math.max(nx, ny, nz) * h * 1.5, tanHalfFov: .4,
  target_m: { x: 0, y: ny * h * .48, z: 0 } };
