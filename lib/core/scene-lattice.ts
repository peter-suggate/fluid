import type { SceneDescription } from "./model";
import { boxSolidVoxelShell, sphericalSolidVoxelShell,
  type SolidWorldVoxelPatch } from "./solid-world";
export {
  DEFAULT_MAXIMUM_LATTICE_DIMENSION,
  MINIMUM_LATTICE_DIMENSION,
  latticeAxisDimension,
  sceneCellSizes_m,
  sceneLatticeCellCount,
  sceneLatticeDimensions,
} from "./scene-lattice-dimensions";
import { sceneLatticeDimensions } from "./scene-lattice-dimensions";

/**
 * The one place the container's metre extents become a cell count.
 *
 * `createTallCellLayout`, the editor's paint lattice, the configuration
 * readout, and the solver rebuild key all have to agree about this rounding:
 * the moment two of them disagree, an edit either rebuilds a solver that did
 * not need rebuilding or — far worse — keeps one whose arenas no longer match
 * the scene. Scaling the world makes that agreement load-bearing, because the
 * whole point of a world scale is that extent and cell size move together and
 * the lattice does not move at all.
 */

/** Author a one-voxel shell; consumers see only the resulting SolidWorld. */
export function solidVoxelShellForScene(
  scene: SceneDescription,
): readonly SolidWorldVoxelPatch[] {
  const dimensions = sceneLatticeDimensions(scene);
  if (scene.container.shape === "sphere") return sphericalSolidVoxelShell(dimensions);
  return boxSolidVoxelShell(dimensions, {
    top: scene.container.top,
  });
}

/** Remove exactly the shell authored for this lattice, preserving every
 * independent fill/clear patch regardless of its material. */
export function solidVoxelEditsForScene(
  scene: SceneDescription,
): SolidWorldVoxelPatch[] {
  const key = (patch: SolidWorldVoxelPatch): string => [patch.operation,
    ...patch.minimum, ...patch.maximumExclusive, patch.materialId ?? 0].join(":");
  const shellCounts = new Map<string, number>();
  for (const patch of solidVoxelShellForScene(scene)) {
    const signature = key(patch);
    shellCounts.set(signature, (shellCounts.get(signature) ?? 0) + 1);
  }
  return scene.solidVoxels.filter((patch) => {
    const signature = key(patch);
    const remaining = shellCounts.get(signature) ?? 0;
    if (remaining === 0) return true;
    shellCounts.set(signature, remaining - 1);
    return false;
  });
}
