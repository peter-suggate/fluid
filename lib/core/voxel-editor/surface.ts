import { vesselCutawayPatches } from "../vessel-cutaway";
import { scenerySurfaceAt } from "../editor-scenery";
import type { EditorRay } from "../editor-entity";
import type { SceneDescription, Vec3 } from "../model";
import { buildSvoSolidWorldPlanarBoundaryCatalog } from "../../svo/features/scene-publication/svo-planar-boundary";
import { voxelReaderForScene } from "./voxel-reader";

const visibleScenes = new WeakMap<SceneDescription, { key: readonly unknown[]; visible: SceneDescription }>();

/** Picking uses the same visible voxel residual as rendering. Collision-only
 * shells and source bakes must neither intercept nor hide a visible surface. */
export function visibleVoxelScene(scene: SceneDescription): SceneDescription {
  const key = [scene.solidVoxels, scene.solidVoxels.length, scene.scenery, scene.terrain, scene.solidBoundary,
    scene.container.width_m, scene.container.height_m, scene.container.depth_m, scene.container.shape,
    scene.container.top, scene.container.walls, scene.voxelDomain.finestCellSize_m,
    scene.cutaway?.eye_m.x, scene.cutaway?.eye_m.y, scene.cutaway?.eye_m.z];
  const known = visibleScenes.get(scene);
  if (known && key.every((value, index) => value === known.key[index])) return known.visible;
  const excluded = buildSvoSolidWorldPlanarBoundaryCatalog(scene, scene.solidVoxels, 0,
    { promoteEditablePatches: false }).residualExcludedPatchIndices;
  const drawn = excluded.size ? scene.solidVoxels.filter((_, index) => !excluded.has(index)) : scene.solidVoxels;
  const cutaway = vesselCutawayPatches(scene, drawn);
  const visible = excluded.size || cutaway.length || scene.scenery?.terrainSurface === "scenery"
    ? { ...scene, solidVoxels: cutaway.length ? [...drawn, ...cutaway] : drawn,
      ...(scene.scenery?.terrainSurface === "scenery" ? { terrain: undefined } : {}) } : scene;
  visibleScenes.set(scene, { key, visible });
  return visible;
}

export interface VisibleToolSurface {
  readonly point: Vec3;
  readonly normal: Vec3;
  readonly distance_m: number;
  readonly scenery?: string;
}

/** Nearest visible opaque surface, never a ray cast through scenery to the
 * physics behind it. All surface-following tools use this same query. */
export function visibleToolSurface(scene: SceneDescription, ray: EditorRay, press = false): VisibleToolSurface | undefined {
  const voxel = voxelReaderForScene(visibleVoxelScene(scene)).pick(ray, { rejectBudgetExhaustion: press });
  const scenery = scenerySurfaceAt(scene, ray, voxel?.distance_m, undefined, true);
  if (scenery) return { point: scenery.position_m, normal: scenery.normal, distance_m: scenery.t_m, scenery: scenery.nodeId };
  if (!voxel) return undefined;
  const normal = { x: 0, y: 0, z: 0 };
  normal[(["x", "y", "z"] as const)[voxel.faceAxis]] = voxel.faceSign;
  return { point: voxel.point_m, normal, distance_m: voxel.distance_m };
}
