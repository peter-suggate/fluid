import type { SparseBrickResolution } from "../../sparse-brick-atlas";

/** Live GPU-authored resolution policy. Accepted topology publication is a
 * separate transaction, so these controls tune candidate requests/history. */
export interface SparseCM12ActivityPolicy {
  readonly activitySignals: boolean;
  readonly coarseFirst: boolean;
  /** Specific kinetic energy (m²/s²) requesting the finest rung. */
  readonly energyThreshold: number;
  /**
   * Maximum normal variation per cell, approximately |curvature| h, applied
   * once when the brick atlas is first built. Per-frame planning no longer
   * reads curvature — feature-preserving coarsening sizes material by
   * deformation and representability — so this only seeds the starting rung of
   * interface bricks and selects the coarse-first bulk cover.
   */
  readonly curvatureTolerance: number;
  readonly surfaceQuietEpochs: number;
  /** Maximum rho=.5 edge-crossing displacement accepted by each one-rung
   * presentation proof, expressed in finest-cell widths. */
  readonly surfaceDisplacementToleranceCells: number;
  /** Enables publication and consumption of surface representability receipts. */
  readonly surfaceCoarseningEnabled: boolean;
  /** QA-only fixed surface rung. Omitted in production and normal UI flows. */
  readonly forcedSurfaceResolutionForQA?: SparseBrickResolution;
  /** Retain accepted bricks and their cell widths; new support may still grow. */
  readonly freezeTopology?: boolean;
  /** Causal control for comparing the former collocated face remap. */
  readonly legacyFaceTransportForQA?: boolean;
  readonly finestTravelCells: number;
  readonly fourTravelCells: number;
  readonly twoTravelCells: number;
  readonly thinFeatureCells: number;
  readonly thinFeatureDensity: number;
  readonly residencyDensity: number;
  readonly residencyMassFineCells: number;
  readonly surfaceDensityMinimum: number;
  readonly detailTolerance: number;
  readonly frontLookaheadSteps: number;
  readonly topologyCadenceSteps: number;
  readonly prepareBricksPerFrame: number;
  readonly promoteEpochs: number;
  readonly demoteEpochs: number;
  readonly promoteScore: number;
  readonly demoteScore: number;
  readonly emergencyScore: number;
}

export const SPARSE_CM12_ACTIVITY_POLICY = Object.freeze({
  activitySignals: true,
  coarseFirst: true,
  energyThreshold: 8,
  curvatureTolerance: 0.25,
  surfaceQuietEpochs: 2,
  surfaceDisplacementToleranceCells: 1,
  surfaceCoarseningEnabled: true,
  finestTravelCells: 1,
  fourTravelCells: 0.5,
  twoTravelCells: 0.25,
  thinFeatureCells: 2,
  thinFeatureDensity: 0,
  residencyDensity: 0.005,
  residencyMassFineCells: 1,
  surfaceDensityMinimum: 0.05,
  detailTolerance: 0.08,
  frontLookaheadSteps: 4,
  topologyCadenceSteps: 1,
  prepareBricksPerFrame: 64,
  promoteEpochs: 2,
  demoteEpochs: 1,
  promoteScore: 160 / 255,
  demoteScore: 96 / 255,
  emergencyScore: 224 / 255,
} satisfies SparseCM12ActivityPolicy);

const finiteClamp = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value)) : fallback;

const integerClamp = (value: unknown, fallback: number, minimum: number, maximum: number) =>
  Math.round(finiteClamp(value, fallback, minimum, maximum));

export function sparseCM12ActivityPolicy(
  values: Partial<Record<keyof SparseCM12ActivityPolicy, unknown>>,
): SparseCM12ActivityPolicy {
  const defaults = SPARSE_CM12_ACTIVITY_POLICY;
  const finestTravelCells = finiteClamp(
    values.finestTravelCells, defaults.finestTravelCells, 0.05, 8,
  );
  const fourTravelCells = Math.min(finestTravelCells, finiteClamp(
    values.fourTravelCells, defaults.fourTravelCells, 0, 8,
  ));
  const twoTravelCells = Math.min(fourTravelCells, finiteClamp(
    values.twoTravelCells, defaults.twoTravelCells, 0, 8,
  ));
  const promoteScore = finiteClamp(values.promoteScore, defaults.promoteScore, 0, 1);
  const forcedSurfaceResolutionForQA = values.forcedSurfaceResolutionForQA === 1
    || values.forcedSurfaceResolutionForQA === 2
    || values.forcedSurfaceResolutionForQA === 4
    || values.forcedSurfaceResolutionForQA === 8
    || values.forcedSurfaceResolutionForQA === 16
    ? values.forcedSurfaceResolutionForQA : undefined;
  return {
    activitySignals: values.activitySignals !== false,
    coarseFirst: values.activitySignals !== false && values.coarseFirst !== false,
    energyThreshold: finiteClamp(values.energyThreshold, defaults.energyThreshold, 0.01, 100),
    curvatureTolerance: finiteClamp(values.curvatureTolerance, defaults.curvatureTolerance, 0.02, 2),
    surfaceQuietEpochs: integerClamp(values.surfaceQuietEpochs, defaults.surfaceQuietEpochs, 1, 32),
    surfaceDisplacementToleranceCells: finiteClamp(
      values.surfaceDisplacementToleranceCells,
      defaults.surfaceDisplacementToleranceCells, 0, 8,
    ),
    surfaceCoarseningEnabled: values.surfaceCoarseningEnabled !== false,
    ...(values.freezeTopology === true ? { freezeTopology: true } : {}),
    ...(values.legacyFaceTransportForQA === true ? { legacyFaceTransportForQA: true } : {}),
    ...(forcedSurfaceResolutionForQA === undefined
      ? {} : { forcedSurfaceResolutionForQA }),
    finestTravelCells,
    fourTravelCells,
    twoTravelCells,
    thinFeatureCells: finiteClamp(
      values.thinFeatureCells, defaults.thinFeatureCells, 0.25, 8,
    ),
    thinFeatureDensity: finiteClamp(
      values.thinFeatureDensity, defaults.thinFeatureDensity, 0, 0.5,
    ),
    residencyDensity: finiteClamp(
      values.residencyDensity, defaults.residencyDensity, 0.000_01, 0.5,
    ),
    residencyMassFineCells: finiteClamp(
      values.residencyMassFineCells, defaults.residencyMassFineCells, 0, 8,
    ),
    surfaceDensityMinimum: finiteClamp(
      values.surfaceDensityMinimum, defaults.surfaceDensityMinimum, 0, 0.49,
    ),
    detailTolerance: finiteClamp(
      values.detailTolerance, defaults.detailTolerance, 0.005, 0.5,
    ),
    frontLookaheadSteps: integerClamp(
      values.frontLookaheadSteps, defaults.frontLookaheadSteps, 1, 32,
    ),
    topologyCadenceSteps: integerClamp(
      values.topologyCadenceSteps, defaults.topologyCadenceSteps, 1, 32,
    ),
    prepareBricksPerFrame: integerClamp(
      values.prepareBricksPerFrame, defaults.prepareBricksPerFrame, 1, 256,
    ),
    promoteEpochs: integerClamp(values.promoteEpochs, defaults.promoteEpochs, 1, 16),
    demoteEpochs: integerClamp(values.demoteEpochs, defaults.demoteEpochs, 1, 32),
    promoteScore,
    demoteScore: Math.min(promoteScore, finiteClamp(
      values.demoteScore, defaults.demoteScore, 0, 1,
    )),
    emergencyScore: Math.max(promoteScore, finiteClamp(
      values.emergencyScore, defaults.emergencyScore, 0, 1,
    )),
  };
}

