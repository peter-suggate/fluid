import type { RetainedSolidEditDelta } from "./sparse-cm12-retained-solid-edit";
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
  /** Immutable sparse overrides. Dense snapshots are materialized only by generation preparation. */
  readonly solidEditPages?: ReadonlyMap<number, ReadonlyMap<number, RetainedSolidEditCell>>;
}

interface RetainedSolidEditCell {
  readonly solid: number; readonly mean: number; readonly open: number;
  readonly amounts?: Float32Array; readonly volumes?: Float32Array; readonly error?: number;
  readonly openErrorBound: number; readonly maximumOpenMeanError: number; readonly maximumSubcellMeanError: number;
}

/** Copy only touched 512-cell overlay pages. Earlier preparation snapshots
 * retain their maps and arrays; repeated edits replace entries rather than
 * retaining an ever-growing transaction history. */
export function withRetainedSolidEdit(cache: RetainedScenePreparationCache,
  delta: RetainedSolidEditDelta): RetainedScenePreparationCache {
  if (delta.fieldGeneration !== cache.openMeans.fieldGeneration) throw new Error("Retained solid edit field generation mismatch");
  if (!delta.indices.length) return cache;
  const pages = new Map(cache.solidEditPages), touched = new Map<number, Map<number, RetainedSolidEditCell>>();
  for (let j = 0; j < delta.indices.length; j++) {
    const index = delta.indices[j]!, pageIndex = Math.floor(index / 512);
    let page = touched.get(pageIndex);
    if (!page) { page = new Map(pages.get(pageIndex)); touched.set(pageIndex, page); pages.set(pageIndex, page); }
    page.set(index, { solid: delta.solidFractions[j]!, mean: delta.effectiveMeans[j]!, open: delta.openFractions[j]!,
      // The edit receipt bounds all boxes; assigning that bound to each changed
      // cell is conservative when later edits replace only part of this page.
      openErrorBound: delta.receipt.estimatedAbsoluteError,
      maximumOpenMeanError: delta.receipt.maximumEstimatedOpenMeanError,
      maximumSubcellMeanError: delta.receipt.maximumEstimatedSubcellMeanError,
      amounts: delta.subcells?.seedAmounts.slice(j * 8, j * 8 + 8),
      volumes: delta.subcells?.openVolumes.slice(j * 8, j * 8 + 8), error: delta.subcells?.estimatedCellAbsoluteErrors[j] });
  }
  return Object.freeze({ ...cache, solidEditPages: pages });
}

function materializeSolidEdits(cache: RetainedScenePreparationCache): RetainedScenePreparationCache {
  if (!cache.solidEditPages?.size) return cache;
  const base = cache.openMeans, sub = cache.subcellMoments;
  const means = base.effectiveMeans.slice(), open = base.openFractions.slice(), solid = base.solidFractions.slice();
  const amounts = sub?.seedAmounts.slice(), volumes = sub?.openVolumes.slice(), errors = sub?.estimatedCellAbsoluteErrors.slice();
  let closed = base.receipt.closedCells, fractional = base.receipt.fractionalCells;
  let openError = base.receipt.estimatedAbsoluteError, maxOpenError = base.receipt.maximumEstimatedOpenMeanError;
  let maxSubcellError = sub?.receipt.maximumEstimatedSubcellMeanError ?? 0;
  for (const page of cache.solidEditPages.values()) for (const [index, cell] of page) {
    closed += Number(cell.solid === 255) - Number(solid[index] === 255);
    fractional += Number(cell.solid > 0 && cell.solid < 255) - Number(solid[index]! > 0 && solid[index]! < 255);
    openError += cell.openErrorBound;
    maxOpenError = Math.max(maxOpenError, cell.maximumOpenMeanError);
    maxSubcellError = Math.max(maxSubcellError, cell.maximumSubcellMeanError);
    means[index] = cell.mean; open[index] = cell.open; solid[index] = cell.solid;
    if (sub) {
      if (!cell.amounts || !cell.volumes) throw new Error("Retained solid edit lacks rigid subcell moments");
      amounts!.set(cell.amounts, index * 8); volumes!.set(cell.volumes, index * 8); errors![index] = cell.error!;
    }
  }
  return Object.freeze({ ...cache, solidEditPages: undefined,
    openMeans: Object.freeze({ ...base, effectiveMeans: means, openFractions: open, solidFractions: solid,
      receipt: Object.freeze({ ...base.receipt, closedCells: closed, fractionalCells: fractional,
        clippedIntegrals: fractional, estimatedAbsoluteError: openError, maximumEstimatedOpenMeanError: maxOpenError }) }),
    subcellMoments: sub ? Object.freeze({ ...sub, solidFractions: solid, seedAmounts: amounts!, openVolumes: volumes!,
      estimatedCellAbsoluteErrors: errors!, changedCellRanges: [],
      receipt: Object.freeze({ ...sub.receipt, estimatedAbsoluteError: errors!.reduce((sum, v) => sum + v, 0),
        maximumEstimatedSubcellMeanError: maxSubcellError }) }) : undefined });
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
  const previous = options.previous ? materializeSolidEdits(options.previous) : undefined, h = Math.fround(cellSize);
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
