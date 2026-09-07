import type { SolidWorld } from "../../core/solid-world";
import { assertRetainedSceneIsotropicLattice, compileRetainedSceneFineMeans,
  type RetainedSceneDensity } from "./sparse-cm12-retained-scene-density";
import { compileRetainedOpenSceneFineMeans, compileRetainedSceneSolidFractions,
  type RetainedOpenSceneFineMeans } from "./sparse-cm12-retained-open-density";
import { compileRetainedSceneSubcellMoments,
  type RetainedSceneSubcellMoments } from "./sparse-cm12-retained-subcell-moments";

/** Immutable numerical seed basis, independent of native ownership or the
 * current affine density coefficients. Arrays are read-only by convention.
 * Structured-clone this snapshot into preparation; never transfer buffers
 * away from an advancing owner. No scene callbacks enter a preparation. */
export interface RetainedScenePreparationCache {
  readonly fieldSignature: string;
  readonly dimensions: readonly [number, number, number];
  readonly cellSize: number;
  readonly unrestrictedMeans: Float32Array;
  readonly openMeans: RetainedOpenSceneFineMeans;
  readonly subcellMoments?: RetainedSceneSubcellMoments;
}

function sameDimensions(a: readonly number[], b: readonly number[]): boolean {
  return a.length === b.length && a.every((n, axis) => n === b[axis]);
}

export function validateRetainedScenePreparationCache(cache: RetainedScenePreparationCache,
  field: RetainedSceneDensity, dimensions: readonly [number, number, number], cellSize: number): void {
  assertRetainedSceneIsotropicLattice(field, dimensions, cellSize);
  const count = dimensions[0] * dimensions[1] * dimensions[2], h = Math.fround(cellSize);
  const open = cache.openMeans, subcells = cache.subcellMoments;
  if (cache.fieldSignature !== JSON.stringify(field) || cache.cellSize !== h
    || !sameDimensions(cache.dimensions, dimensions) || cache.unrestrictedMeans.length !== count
    || open.fieldGeneration !== field.generation || open.cellSize !== h || !sameDimensions(open.dimensions, dimensions)
    || open.effectiveMeans.length !== count || open.openFractions.length !== count || open.solidFractions.length !== count
    || (subcells && (subcells.fieldSignature !== cache.fieldSignature || subcells.cellSize !== h
      || !sameDimensions(subcells.dimensions, dimensions) || subcells.seedAmounts.length !== 8 * count
      || subcells.openVolumes.length !== 8 * count || subcells.solidFractions.length !== count
      || subcells.estimatedCellAbsoluteErrors.length !== count))) {
    throw new Error("Retained preparation cache does not match its numeric field and physical lattice");
  }
}

/** Reuse quadrature from a frozen source snapshot. A static edit compares its
 * canonical q8 geometry before reusing open moments; changed geometry can
 * reuse unrestricted moments and the unaffected rigid subcell supports. */
export function compileRetainedScenePreparationCache(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number, world: SolidWorld,
  options: { previous?: RetainedScenePreparationCache; rigid?: boolean } = {}): RetainedScenePreparationCache {
  assertRetainedSceneIsotropicLattice(field, dimensions, cellSize);
  const previous = options.previous, h = Math.fround(cellSize);
  if (previous) validateRetainedScenePreparationCache(previous, field, dimensions, h);
  const unrestrictedMeans = previous?.unrestrictedMeans ?? compileRetainedSceneFineMeans(field, dimensions, h);
  let openMeans = previous?.openMeans;
  if (openMeans) {
    const solid = compileRetainedSceneSolidFractions(dimensions, world);
    if (!solid.every((q8, i) => q8 === openMeans!.solidFractions[i])) openMeans = undefined;
  }
  openMeans ??= compileRetainedOpenSceneFineMeans(field, dimensions, h, world, { seedMeans: unrestrictedMeans });
  let subcellMoments = previous?.subcellMoments;
  if (options.rigid && (!subcellMoments
    || !openMeans.solidFractions.every((q8, i) => q8 === subcellMoments!.solidFractions[i]))) {
    subcellMoments = compileRetainedSceneSubcellMoments(field, dimensions, h, world,
      { previous: subcellMoments, solidFractions: openMeans.solidFractions });
  }
  return Object.freeze({ fieldSignature: JSON.stringify(field),
    dimensions: Object.freeze([...dimensions]) as unknown as RetainedScenePreparationCache["dimensions"],
    cellSize: h, unrestrictedMeans, openMeans, subcellMoments });
}
