import type { SceneDescription } from "./model";
import type { PlanarBoundaryPatch } from "./planar-boundary";
import { solidVoxelShellForScene } from "./scene-lattice";
import { sceneHasTerrain } from "./terrain";
import {
  planarBoundaryForSolidWorldVoxelPatch,
  type SolidWorldVoxelPatch,
} from "./solid-world";

export type PlanarFluidBoundaryFace =
  | "xLow" | "xHigh" | "yLow" | "yHigh" | "zLow" | "zHigh";

/** Stable six-bit ABI shared by the CPU compiler and topology shaders. */
export const PLANAR_FLUID_BOUNDARY_FACE_BIT: Readonly<
Record<PlanarFluidBoundaryFace, number>
> = Object.freeze({
  xLow: 1 << 0,
  xHigh: 1 << 1,
  yLow: 1 << 2,
  yHigh: 1 << 3,
  zLow: 1 << 4,
  zHigh: 1 << 5,
});

export interface E0PlanarFluidBoundary {
  readonly face: PlanarFluidBoundaryFace;
  readonly patch: PlanarBoundaryPatch;
  readonly sourcePatchIndex: number;
}

export interface E0PlanarFluidBoundaryCompilation {
  readonly admitted: readonly E0PlanarFluidBoundary[];
  readonly rejectedFaces: readonly PlanarFluidBoundaryFace[];
}

export interface E0PlanarFluidBoundaryRuntimeConfiguration {
  readonly planarFluidBoundaryFaceMask: number;
  readonly genericSolidBoundaries: boolean;
}

export function e0PlanarFluidBoundaryFaceMask(
  compilation: E0PlanarFluidBoundaryCompilation,
): number {
  return compilation.admitted.reduce((mask, boundary) =>
    mask | PLANAR_FLUID_BOUNDARY_FACE_BIT[boundary.face], 0);
}

/**
 * Solver routing for a scene's authored solid patches. Exact vessel planes
 * keep the implicit MAC boundary; every residual patch stays on SolidWorld's
 * generic cut-cell path.
 */
export function e0PlanarFluidBoundaryRuntimeConfiguration(
  scene: SceneDescription,
): E0PlanarFluidBoundaryRuntimeConfiguration {
  const compilation = compileE0PlanarFluidBoundaries(scene);
  const admittedSources = new Set(compilation.admitted.map(
    (boundary) => boundary.sourcePatchIndex,
  ));
  return {
    planarFluidBoundaryFaceMask: e0PlanarFluidBoundaryFaceMask(compilation),
    genericSolidBoundaries: sceneHasTerrain(scene)
      || scene.solidVoxels.some((_, index) => !admittedSources.has(index)),
  };
}

const BOX_SHELL_FACES: readonly PlanarFluidBoundaryFace[] = [
  "yLow", "xLow", "xHigh", "zLow", "zHigh", "yHigh",
];

const sameCoordinate = (left: readonly number[], right: readonly number[]) =>
  left.every((value, axis) => value === right[axis]);

const sameFillBox = (left: SolidWorldVoxelPatch, right: SolidWorldVoxelPatch) =>
  left.operation === "fill"
    && sameCoordinate(left.minimum, right.minimum)
    && sameCoordinate(left.maximumExclusive, right.maximumExclusive);

const boxesOverlapVolume = (left: SolidWorldVoxelPatch, right: SolidWorldVoxelPatch) =>
  left.maximumExclusive[0] > right.minimum[0]
    && left.minimum[0] < right.maximumExclusive[0]
    && left.maximumExclusive[1] > right.minimum[1]
    && left.minimum[1] < right.maximumExclusive[1]
    && left.maximumExclusive[2] > right.minimum[2]
    && left.minimum[2] < right.maximumExclusive[2];

/**
 * Compile the box vessel's axis-aligned shell into exact E0 closed-face records.
 *
 * Admission is intentionally strict: the canonical shell fill must be present
 * verbatim and no subtractive edit may overlap it. Holes and edited walls stay
 * with SolidWorld until the later cut-cell topology phases.
 */
export function compileE0PlanarFluidBoundaries(
  scene: SceneDescription,
): E0PlanarFluidBoundaryCompilation {
  if (scene.container.shape === "sphere") {
    return { admitted: [], rejectedFaces: BOX_SHELL_FACES.slice(0, 5) };
  }
  const expected = solidVoxelShellForScene(scene);
  const admitted: E0PlanarFluidBoundary[] = [];
  const rejectedFaces: PlanarFluidBoundaryFace[] = [];
  expected.forEach((shellPatch, shellIndex) => {
    const face = BOX_SHELL_FACES[shellIndex]!;
    const sourcePatchIndex = scene.solidVoxels.findIndex((candidate) =>
      sameFillBox(candidate, shellPatch));
    const cut = scene.solidVoxels.some((candidate) =>
      candidate.operation === "clear" && boxesOverlapVolume(candidate, shellPatch));
    const patch = sourcePatchIndex >= 0 && !cut
      ? planarBoundaryForSolidWorldVoxelPatch(
        scene, scene.solidVoxels[sourcePatchIndex]!, 0xffff,
      )
      : null;
    if (!patch) {
      rejectedFaces.push(face);
      return;
    }
    admitted.push({ face, patch, sourcePatchIndex });
  });
  return { admitted, rejectedFaces };
}
