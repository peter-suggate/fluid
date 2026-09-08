import type { SceneDescription } from "../../../core/model";
import { solidWorldForScene, type SolidWorld } from "../../../core/solid-world";
import { materialIdForRigidShape, VOXEL_MATERIAL_IDS } from "../../../core/voxel-scene";
import type { SparseVoxelDrySceneData } from "../../contracts/scene-publication";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS } from "../../contracts/svo-material-abi";
import { SVO_PRIMITIVE_RECORD_STRIDE_BYTES } from "../../contracts/svo-primitive-abi";
import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_STRIDE_BYTES } from "../../contracts/svo-light-abi";

const solidMaterials = new WeakMap<SolidWorld, ReadonlySet<number>>();

/** Inspect the canonical published solid snapshot, including live material edits.
 * Cache by snapshot identity, never by scene name or light/material revision. */
export function publishOpaqueSurfaceCapability(
  scene: SceneDescription, materials: Uint32Array, primitives: Uint32Array,
): boolean {
  const world = solidWorldForScene(scene);
  let ids = solidMaterials.get(world);
  if (!ids) {
    const used = new Set<number>();
    for (const page of world.pages) for (const id of page.materialId) if (id) used.add(id);
    for (const patch of [...world.patches, ...(world.regions ?? [])]) {
      if (patch.operation === "fill") used.add(patch.materialId ?? VOXEL_MATERIAL_IDS.containerGlass);
    }
    ids = used;
    solidMaterials.set(world, ids);
  }
  const opaque = (id: number): boolean => {
    const offset = id * SVO_MATERIAL_RECORD_WORDS;
    return id > 0 && offset + SVO_MATERIAL_RECORD_WORDS <= materials.length
      && (materials[offset + 23]! & (SVO_MATERIAL_FLAGS.opaque | SVO_MATERIAL_FLAGS.dielectric)) === SVO_MATERIAL_FLAGS.opaque;
  };
  if (![...ids, VOXEL_MATERIAL_IDS.terrain].every(opaque)) return false;
  for (const body of scene.rigidBodies) if (!opaque(materialIdForRigidShape(body.shape))) return false;
  const stride = SVO_PRIMITIVE_RECORD_STRIDE_BYTES / 4;
  if (primitives.length % stride !== 0) return false;
  for (let offset = 0; offset < primitives.length; offset += stride) {
    if (!opaque(primitives[offset + 7]! & 0xffff)) return false;
  }
  return true;
}

/** Evaluated at draw time: a previously compiled specialized pipeline is never
 * evidence that a newly published scene or currently available hierarchy fits. */
export function canUseOpaqueDirectionalCones(
  scene: SparseVoxelDrySceneData | undefined,
  options: { coneMode: string; hierarchyReady: boolean; globalIllumination: boolean; reconstruction: string },
): boolean {
  return scene?.opaqueSurfaceOnly === true
    && !scene.glassRecords?.byteLength && !scene.thickGlassRecords?.byteLength
    && scene.lightRecords?.byteLength === SVO_LIGHT_RECORD_STRIDE_BYTES
    && scene.lightRecords[24] === SVO_LIGHT_KINDS.directional
    && options.coneMode === "cones" && options.hierarchyReady
    && !options.globalIllumination && options.reconstruction === "full-res-relight";
}
