import type { MethodParamValues } from "../core/method-contract";
import { sceneDocument, type SceneDefinition } from "../core/scene-definition";
import {
  CEILING_DROP_METHOD_PROFILE,
  COARSE_ONLY_POWER_DAM_METHOD_PROFILE,
  DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY,
  DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
  LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
  LARGE_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
  OCEAN_SEICHE_METHOD_PROFILE,
  POWER_DROPLET_EDGE_CELLS,
  POWER_DROPLET_FINE_BRICK_CAPACITY,
  POWER_DROPLET_PRESSURE_ROW_CAPACITY,
  POWER_FILL_EDGE_CELLS,
  POWER_FILL_FINE_BRICK_CAPACITY,
  POWER_FILL_LIQUID_CELLS,
  POWER_FILL_PRESSURE_ROW_CAPACITY,
  POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE,
  RIGID_COUPLING_ORACLE_METHOD_PROFILE,
  SYMMETRIC_EXPANSION_METHOD_PROFILE,
  findSceneDefinition,
  powerFillReservoirCells,
} from "../core/scenes";
import {
  defineSceneWebGPUSmokeSuite,
  type SceneWebGPUAcceptanceRule,
  type SceneWebGPUCollectionPolicy,
  type SceneWebGPUDiagnosticHook,
  type SceneWebGPUDiagnosticPack,
  type SceneWebGPUSmokeLane,
  type SceneWebGPUSmokeMethod,
  type SceneWebGPUSmokeScene,
  type SceneWebGPUSmokeSuite,
  type WebGPUSmokeMethodId,
} from "./scene-webgpu-smoke";

export const sceneWebGPUSmokeIds = [
  "dam-break-ui",
  "settled-tank",
  "settled-tank-ui",
  "dam-break-boxes",
  "cm12-figure-2",
  "cm12-figure-3",
  "cm12-figure-8",
  "cm12-figure-12",
  "mass-conserving-figure-9-dam-break",
  "hose-tank",
  "sphere-jet",
  "deep-water",
  "garden-pond",
  "garden-hose",
  "garden-dam-break",
  "brick-quad-dam-break",
  "twin-dam-collision",
  "hydrostatic-power-two-level",
  "hydrostatic-power-large-offset",
  "minimal-power-dam-break",
  "high-resolution-dam-break",
  "symmetric-expansion",
  "minimal-power-dam-break-32",
  "minimal-power-dam-break-64",
  "large-power-dam-break",
  "large-power-hydrostatic",
  "deep-power-hydrostatic",
  "power-droplet-64",
  "power-droplet-128",
  "power-droplet-240",
  "power-droplet-256",
  "power-fill-256-100",
  "power-fill-256-800",
  "power-fill-256-6400",
  "rigid-hydrostatic",
  "rigid-float",
  "rigid-sink",
  "ceiling-slab-drop",
  "corner-brick-drop",
  "midair-brick-drop",
  "midair-corner-drop",
  "power-hybrid-deep-ocean",
  "ocean-seiche",
] as const;

export type SceneWebGPUSmokeId = typeof sceneWebGPUSmokeIds[number];

const allMethodIds: readonly WebGPUSmokeMethodId[] = ["losasso"];

function methods(
  ids: readonly WebGPUSmokeMethodId[] = allMethodIds,
  overrides: Partial<Record<WebGPUSmokeMethodId, Readonly<MethodParamValues>>> = {},
  quality: SceneWebGPUSmokeMethod["quality"] = "balanced",
): SceneWebGPUSmokeMethod[] {
  return ids.map((id) => ({
    id,
    quality,
    // Generic authored-scene lanes exercise the product default (Losasso).
    // Frozen Power reference suites opt in through their named profiles below.
    overrides: { ...(overrides[id] ?? {}) },
  }));
}

function usesPowerReference(methods: readonly SceneWebGPUSmokeMethod[]): boolean {
  return methods.some(({ id }) => id === "power-liquids");
}

const coreDiagnostics: readonly SceneWebGPUDiagnosticPack[] = [
  { id: "core-webgpu-health", parameters: {
    maximumValidationErrors: 0,
    maximumNonFiniteValues: 0,
    requireFiniteMaximumSpeed: true,
    requireCommonGridAcrossMethods: true,
    structuredMovingSceneMinimumComponentCfl: 0,
    exactStepTimeTolerance_s: 1e-9,
    pressureWarmupStepsWithoutResidual: 2,
  } },
  { id: "volume-and-topology", parameters: {
    minimumStoredDensity: -0.01,
    maximumStoredDensity: 1.5,
    minimumMeaningfulSpatialFieldSum: 1,
    minimumMeaningfulSpatialFieldDryCells: 1,
    requireCurrentCompactOctreePublication: true,
    maximumTallNeighborDeltaViolations: 0,
    quadtreeMaximumFluidScale: 100,
    quadtreeMaximumPressureRelativeResidual: 1e-4,
    quadtreeMaximumPressureRmsResidual: 1e-5,
    quadtreePressureAcceptance: "relative-or-rms",
    sparseMinimumVoxelCount: 1,
    sparseMinimumBrickCount: 1,
    sparseMinimumActiveVoxelCount: 1,
    sparseMinimumActiveBrickCount: 1,
    sparseMinimumFluidVoxelCount: 1,
    sparseMinimumEnvironmentVoxelCount: 1,
    sparseSphereMaterialMinimumVoxelCount: 1,
    sparseSphereMaterialMaximumVoxelCountExclusive: 10_000,
    sparseMaximumNonFiniteRecordCount: 0,
    sparseMaximumInvalidMaterialCount: 0,
    requirePositiveRawVoxelRenderWallTime: true,
    requirePositiveBrickGridRenderWallTime: true,
    sparseMinimumHybridBodyCount: 1,
    requirePositiveHybridFrameWallTime: true,
    sparseMinimumHybridFrontInterfacePixels: 1,
    sparseFluidColorTolerance: 1e-6,
    requireResidentBrickCountBelowCapacity: true,
    liquidCellThreshold: 0.5,
    wetFlipCensusMinimumJumpCells: 2,
    wetFlipCensusCadenceDt_s: 0.004,
    wetFlipCensusCellSize_m: 0.05,
    wetFlipCensusImpliedSpeed_m_s: 12.5,
  } },
  { id: "octree-authority", methods: ["losasso", "power-liquids"], parameters: {
    expectedGridKind: "octree",
    maximumNeighborRatio: 2,
    maximumTopologyReadbackBytes: 0,
    requirePowerDiagramReady: true,
    requirePowerDiagramAuthoritative: true,
    maximumDescriptorErrors: 0,
    maximumTopologyErrors: 0,
    pressureSolverNameIncludes: "exact-reduction executor",
    maximumPressureRelativeResidual: 1e-4,
    maximumPressureRelativeResidualSquared: 1e-8,
    maximumPressureDiagnosticFlags: 0,
    requirePressureConverged: true,
    minimumPressureRows: 1,
    requireFiniteNonnegativePressureResiduals: true,
    damBreakMaximumProjectionEnergyRatioWithoutAdaptiveTransport: 1.1,
    damBreakMaximumComponentCfl: 3,
    damBreakMaximumExactVolumeDrift: 0.01,
    damBreakMaximumFinalVolumeDrift: 0.01,
    damBreakMinimumDominantComponentFraction: 0.98,
    damBreakImpactWindow_s: [0.9, 1.3],
    damBreakImpactTarget_s: 1.1,
    damBreakMaximumInterfaceFaceGrowthRatio: 6,
    damBreakMaximumEnclosedAirCells: 8,
    rasterCheckpointMinimumFrontPixels: 1,
    rasterCheckpointMinimumBackPixels: 1,
    requireGlobalFineFactorMatchesMethodOverride: true,
  } },
];

const coreAcceptance: readonly SceneWebGPUAcceptanceRule[] = [
  { id: "validation-clean", metric: "methods.*.validationErrorCount", operator: "equal", expected: 0 },
  { id: "finite-state", metric: "methods.*.info.nonFiniteCount", operator: "equal", expected: 0 },
  { id: "finite-speed", metric: "methods.*.info.maxSpeed_m_s", operator: "finite" },
  { id: "volume-lower-bound", metric: "methods.*.matchedSummary.minimum", operator: "at-least", expected: -0.01 },
  { id: "volume-upper-bound", metric: "methods.*.matchedSummary.maximum", operator: "at-most", expected: 1.5 },
  { id: "final-volume-lower-bound", metric: "methods.*.finalSummary.minimum", operator: "at-least", expected: -0.01,
    when: [{ metric: "methods.*.finalSummary", operator: "present" }] },
  { id: "final-volume-upper-bound", metric: "methods.*.finalSummary.maximum", operator: "at-most", expected: 1.5,
    when: [{ metric: "methods.*.finalSummary", operator: "present" }] },
];

const defaultCollection: SceneWebGPUCollectionPolicy = {
  evidenceCollectors: [],
  fieldStats: "final",
  stabilityEnvelope: false,
  spatialField: false,
  sparsePublication: false,
  raster: "none",
  globalFineGeneration: false,
  powerGenerationAudit: false,
  boundaryThetaHistogram: false,
  structuredValidation: false,
  performanceProfile: false,
  gpuCommandAudit: false,
};

interface LaneOptions {
  id?: string;
  description?: string;
  methods?: SceneWebGPUSmokeMethod[];
  target_s: number;
  exactSteps?: number;
  maxDt_s?: number;
  dtPattern_s?: readonly number[];
  oracleSteps: number;
  collect?: Partial<SceneWebGPUCollectionPolicy>;
  diagnostics?: readonly SceneWebGPUDiagnosticPack[];
  acceptance?: readonly SceneWebGPUAcceptanceRule[];
  hooks?: readonly SceneWebGPUDiagnosticHook[];
  timeout_ms?: number;
  /** False for source-driven inflow scenes whose represented volume grows. */
  maximumRepresentedVolumeDrift?: number | false;
  /** Paper surface density may temporarily exceed one while volume recovers. */
  maximumStoredDensity?: number;
}

function backendAcceptance(configuredMethods: readonly SceneWebGPUSmokeMethod[]): SceneWebGPUAcceptanceRule[] {
  const ids = new Set(configuredMethods.map(({ id }) => id));
  const rules: SceneWebGPUAcceptanceRule[] = [];
  if (ids.has("losasso") || ids.has("power-liquids")) {
    // These rules are scoped by the branch rather than by the metric: every
    // catalog lane runs exactly one method, so `methods.*.` resolves to the
    // adaptive evidence the branch already tested for. A lane that ever pairs
    // an adaptive method with a second one has to pin these rules with
    // `methods: [...]`, or `octree-grid-kind` would demand an octree grid of
    // the other one. The rule ids stay `octree-*`: what they assert is the
    // shared engine's grid, which both adaptive methods publish.
    rules.push(
    { id: "octree-grid-kind", metric: "methods.*.info.gridKind", operator: "equal", expected: "octree" },
    { id: "octree-neighbor-ratio", metric: "methods.*.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
    { id: "octree-topology-readback", metric: "methods.*.info.quadtreeTopologyReadbackBytes", operator: "equal", expected: 0 },
    );
    if (usesPowerReference(configuredMethods)) rules.push(
      { id: "octree-power-ready", metric: "methods.*.info.powerDiagramReady", operator: "equal", expected: true },
      { id: "octree-power-authoritative", metric: "methods.*.info.powerDiagramAuthoritative", operator: "equal", expected: true },
      { id: "octree-power-descriptor-errors", metric: "methods.*.octreePowerTopologyDiagnostics.descriptor.errorCount", operator: "equal", expected: 0 },
      { id: "octree-power-topology-errors", metric: "methods.*.octreePowerTopologyDiagnostics.topology.invalidCount", operator: "equal", expected: 0 },
    );
  }
  return rules;
}

function lane(options: LaneOptions): SceneWebGPUSmokeLane {
  const configuredMethods = options.methods ?? methods();
  const powerReference = usesPowerReference(configuredMethods);
  const maximumStoredDensity = options.maximumStoredDensity ?? 1.5;
  const configuredDiagnostics = [
    ...coreDiagnostics.filter((diagnostic) => diagnostic.id !== "octree-authority"
      || powerReference).map((diagnostic) => diagnostic.id === "volume-and-topology"
      ? { ...diagnostic, parameters: { ...diagnostic.parameters, maximumStoredDensity } }
      : diagnostic),
    ...(options.diagnostics ?? []),
  ];
  const maximumRepresentedVolumeDrift = options.maximumRepresentedVolumeDrift === undefined
    ? 0.01 : options.maximumRepresentedVolumeDrift;
  return {
    id: options.id ?? "default",
    description: options.description ?? "Authored scene smoke validation",
    methods: configuredMethods,
    stop: {
      simulatedTime_s: options.target_s,
      ...(options.exactSteps === undefined ? {} : { exactSteps: options.exactSteps }),
      ...(options.maxDt_s === undefined ? {} : { maxDt_s: options.maxDt_s }),
      ...(options.dtPattern_s === undefined ? {} : { dtPattern_s: options.dtPattern_s }),
    },
    oracle: { matchedSteps: options.oracleSteps },
    collect: { ...defaultCollection, ...options.collect },
    diagnostics: configuredDiagnostics,
    acceptance: [
      ...coreAcceptance.map((rule) => rule.id === "volume-upper-bound"
        || rule.id === "final-volume-upper-bound"
        ? { ...rule, expected: maximumStoredDensity }
        : rule),
      ...backendAcceptance(configuredMethods),
      ...(maximumRepresentedVolumeDrift === false ? [] : [{
        id: "represented-volume-drift",
        metric: "methods.*.info.representedVolumeDrift.abs",
        operator: "at-most" as const,
        expected: maximumRepresentedVolumeDrift,
      }]),
      ...(options.acceptance ?? []),
    ],
    hooks: [...(options.hooks ?? [])],
    ...(options.timeout_ms === undefined ? {} : { timeout_ms: options.timeout_ms }),
  };
}

/**
 * Frozen Power-2017 comparison profile: the `power-liquids` method's own tuple.
 * New and backend-agnostic smoke lanes must run the Losasso product default
 * instead of spreading this profile.
 */
const frozenPowerReferenceOverrides = {
  maximumLeafSize: "32",
  interfaceRefinementBandCells: 3,
  globalFineLevelSetFactor: "4",
} as const;
const symmetricExpansionOverrides = SYMMETRIC_EXPANSION_METHOD_PROFILE.overrides;
// The fixed-order compensated-f32 solve is deterministic, but symmetric rows
// traverse distinct published CSR orders and can differ by a handful of ulps.
// Conservative surface-mass transport also consumes that numerically symmetric
// projected field before reconstructing nodal phi, so scalar, ghost, and
// diagonal values are numerical rather than byte-exact. These bounds remain
// below 0.1% of their twenty-step physical ranges; topology stays exact.
const symmetricExpansionFieldLimits = Object.freeze({
  maximumVolumeAbsoluteError: 1e-3,
  maximumVelocityAbsoluteError_m_s: 1e-4,
  maximumPressureAbsoluteError: 0.25,
  maximumRhsAbsoluteError: 0.015625,
  maximumDiagonalAbsoluteError: 1e-3,
});
const largePowerDamOverrides = {
  ...frozenPowerReferenceOverrides,
  maximumLeafSize: "32",
  interfaceRefinementBandCells: 1,
  pressureRowCapacity: 8_192,
  globalFineLevelSetMaximumBricks: LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
} as const;
const largePowerHydrostaticOverrides = {
  ...frozenPowerReferenceOverrides,
  maximumLeafSize: "32",
  interfaceRefinementBandCells: 1,
  pressureRowCapacity: LARGE_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
} as const;
/** The deep still lane differs from the 20x still lane only in its two
 * footprint-derived reserves; every discretization knob is shared, so a wall
 * or row count measured here is comparable to the shallow lane's. */
const deepPowerHydrostaticOverrides = {
  ...largePowerHydrostaticOverrides,
  pressureRowCapacity: DEEP_POWER_HYDROSTATIC_PRESSURE_ROW_CAPACITY,
  globalFineLevelSetMaximumBricks: DEEP_POWER_HYDROSTATIC_FINE_BRICK_CAPACITY,
} as const;
/**
 * The droplet family's reserves, shared by every N in the sweep.
 *
 * These are the same two constants for a 64-cubed container and a 256-cubed
 * one, on purpose: if a lane had to grow a capacity to run a larger empty box,
 * the sweep would be measuring the capacity and not the domain. The
 * `globalFineLevelSetMaximumBricks` half cannot be expressed any other way —
 * it has no environment override — so this object is the only place a droplet
 * benchmark and a droplet smoke run agree on the scene.
 */
const powerDropletOverrides = {
  ...largePowerHydrostaticOverrides,
  pressureRowCapacity: POWER_DROPLET_PRESSURE_ROW_CAPACITY,
  globalFineLevelSetMaximumBricks: POWER_DROPLET_FINE_BRICK_CAPACITY,
} as const;
/**
 * The fill family's reserves, shared by every member of the sweep.
 *
 * Same two knobs as `powerDropletOverrides` and the same reason for existing —
 * a capacity that moved with the member would make a capacity-shaped pass read
 * as live-shaped — except that here the *live* term is what varies, so the
 * constancy is doing strictly more work. Both values are 16x the droplet
 * family's, which is what turns `power-fill-256-100` against
 * `power-droplet-256` into a clean capacity A/B at identical live occupancy.
 */
const powerFillOverrides = {
  ...largePowerHydrostaticOverrides,
  pressureRowCapacity: POWER_FILL_PRESSURE_ROW_CAPACITY,
  globalFineLevelSetMaximumBricks: POWER_FILL_FINE_BRICK_CAPACITY,
} as const;

/**
 * Packs scoped to the whole adaptive family, not to one method.
 *
 * They were authored against the single `octree` method and therefore ran for
 * both coarse backends; the D4 factor-8 cutover command still runs Losasso on
 * a lane the frozen reference authored. Naming both ids keeps that evidence
 * collected instead of silently selecting no method.
 */
const powerDiagnostics: readonly SceneWebGPUDiagnosticPack[] = [
  { id: "structured-power", methods: ["losasso", "power-liquids"], parameters: {
    requireDirectStructuredVelocityAuthority: true,
    maximumNeighborRatio: 2,
    requireEveryStepGenerationAuditWhenCollected: true,
  } },
  { id: "exhaustive-power-generation", methods: ["losasso", "power-liquids"], parameters: {
    minimumStepsWithoutExactStop: 50,
    requireGenerationAuditEveryStep: true,
    requireStabilitySampleEveryStep: true,
    maximumInvalidVolumeSamples: 0,
    maximumNonFiniteVelocitySamples: 0,
    maximumExactVolumeDrift: 0.01,
    maximumPressureRelativeResidual: 1e-4,
  } },
  { id: "global-fine-publication", methods: ["losasso", "power-liquids"], parameters: {
    requirePublishedInitialGeneration: true,
    requirePublishedNonemptyFinalGeneration: true,
    minimumPublishedGeneration: 1,
    minimumFinalActivePages: 1,
    requireFiniteSignedInterface: true,
    minimumValidSamples: 1,
    minimumNegativeValidSamples: 1,
    minimumPositiveValidSamples: 1,
    checkpointCoarseState: 0x8000_0000,
    requireMatchingCoarseGeneration: true,
    requireNoTopologyRollback: true,
    topologyFinalizeReason: 0,
    requireFactorMatchesMethodOverride: true,
  } },
];

const powerAcceptance: readonly SceneWebGPUAcceptanceRule[] = [
  { id: "structured-authority", metric: "methods.*.info.powerDiagramAuthoritative", operator: "equal", expected: true },
  { id: "balanced-power-grid", metric: "methods.*.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
  { id: "power-generation-every-step", metric: "methods.*.powerGenerationAuditedSteps", operator: "equal", expected: { selector: "methods.*.steps" } },
  { id: "power-volume-sample-every-step", metric: "methods.*.stabilityEnvelope.sampledSteps", operator: "equal", expected: { selector: "methods.*.steps" } },
  { id: "power-volume-samples-valid", metric: "methods.*.stabilityEnvelope.invalidVolumeSampleCount", operator: "equal", expected: 0 },
  { id: "power-velocities-finite", metric: "methods.*.stabilityEnvelope.nonFiniteVelocityCount", operator: "equal", expected: 0 },
];

function definition(id: string): SceneDefinition {
  const found = findSceneDefinition(id);
  if (!found) throw new Error(`The WebGPU smoke catalog names an unknown scene ${id}`);
  return found;
}

/**
 * A suite names the authored scene it runs; it never builds one.
 *
 * Eleven local factories used to rebuild a catalog scene here, under the
 * catalog's own name, from `defaultScene` — which meant assigning `environment`
 * as a bare string, so a lane ran a document with no scenery graph at all and
 * `hose-tank` on the GPU lane was not the `hose-tank` in the product. A lane
 * that genuinely needs a different document names a variant of the scene it
 * differs from, and the difference lives beside that scene.
 */
function suite<const Id extends SceneWebGPUSmokeId>(
  sceneId: Id,
  description: string,
  scene: SceneWebGPUSmokeScene,
  lanes: Record<string, SceneWebGPUSmokeLane>,
  defaultLane = "default",
): SceneWebGPUSmokeSuite<Id> {
  const resolved = definition(scene.definitionId);
  return defineSceneWebGPUSmokeSuite({
    sceneId, description, scene,
    createScene: () => sceneDocument(resolved, scene.variantId),
    defaultLane, lanes,
  });
}

const inflowDiagnostics: readonly SceneWebGPUDiagnosticPack[] = [{ id: "inflow-activity", parameters: {
  minimumSourceAdjustedRepresentedVolumeRatio: 0.99,
  motionEvaluationStart_s: 0.3,
  minimumEstablishedMaximumSpeed_m_s: 0.01,
} }];
const inflowAcceptance: readonly SceneWebGPUAcceptanceRule[] = [
  { id: "inflow-retains-source-volume", metric: "methods.*.info.sourceAdjustedRepresentedVolumeRatio", operator: "at-least", expected: 0.99 },
  { id: "established-inflow-moves", metric: "methods.*.info.maxSpeed_m_s", operator: "at-least", expected: 0.01,
    when: [{ metric: "methods.*.info.simulatedTime_s", operator: "at-least", expected: 0.3 }] },
];

const equilibriumDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "equilibrium",
  parameters: {
    maximumExactVolumeDrift: 0.01,
    maximumComponents: 1,
  },
};

const crossMethodFieldParityDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "cross-method-field-parity",
  parameters: {
    minimumCheckpointWetIntersectionOverUnion: 0.35,
    minimumFinalWetIntersectionOverUnion: 0.4,
    maximumMixedCellFractionMultiplier: 2,
    maximumMixedCellFractionOffset: 0.05,
    scalarFieldLiquidThreshold: 0.5,
  },
};

const settlingDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "settling",
  parameters: {
    expectAsymptoticRest: true,
    maximumFinalSampledExactVolumeDrift: 0.01,
    maximumNormalizedNetProjectionEnergyDelta: 0.01,
    maximumNormalizedLateMechanicalEnergySlopePerSecond: 1e-3,
    maximumLateToMiddleKineticEnvelopeRatio: 1,
    damBreakMaximumDriftSignChanges: 3,
    damBreakMaximumLatePeakToPeakDrift: 0.005,
    energyMiddleWindowFraction: [0.2, 0.4],
    energyLateWindowStartFraction: 0.8,
    energyRegressionWindowStartFraction: 0.5,
  },
};

const inviscidStabilityDiagnostic: SceneWebGPUDiagnosticPack = {
  ...settlingDiagnostic,
  parameters: { ...settlingDiagnostic.parameters, expectAsymptoticRest: false },
};

const standardWaterRasterParameters = {
  minimumFrontInterfacePixels: 1,
  minimumBackInterfacePixels: 1,
  maximumBackOnlyPixels: 0,
  preImpactHoleLimitPixels: 0,
  terraceEdgeFractionMaximum: 0,
  boundsToleranceRatio: 1e-4,
  expectedSurfaceGeometrySource: "global-fine-coarse",
  requireGlobalFineCrossingPublished: true,
  requirePresentationFallbackInactive: true,
  requireNonzeroAuthorityLatch: true,
  requireNonemptyGeometry: true,
  requireAllocatorCountsMatch: true,
  requireGeometryWithinCapacity: true,
  requireCleanFineCoarseTransition: true,
  expectedRetainedGeometrySource: "retained-previous",
  requireRetainedRasterIdentity: true,
  preImpactMarginFinestCells: 1,
  minimumValidSamples: 1,
  minimumNegativeValidSamples: 1,
  minimumPositiveValidSamples: 1,
} as const;

const minimalDamMotionParameters = {
  minimumPeakSpeed_m_s: 0.1,
  minimumLateralSpread_m: 0.05,
  maximumMechanicalEnergyRetention: 1.5,
  maximumRitterCelerityRatio: 1.35,
  energyEvaluationAfter_s: 1,
  minimumRitterColumnHeightFraction: 0.92,
  ritterCelerityMultiplier: 2,
} as const;

function exhaustivePowerDiagnostics(maximumExactVolumeDrift: number): readonly SceneWebGPUDiagnosticPack[] {
  return powerDiagnostics.map((diagnostic) => diagnostic.id === "exhaustive-power-generation"
    ? { ...diagnostic, parameters: { ...diagnostic.parameters, maximumExactVolumeDrift } }
    : diagnostic);
}

const minimalDamRasterParameters = {
  ...standardWaterRasterParameters,
  initialDamCornerCaps: true,
  maximumBackOnlyPixels: 2,
  preImpactHoleLimitPixels: 2,
  terraceEdgeFractionMaximum: 0.12,
  minimumReservoirWallCornerCapPixels: 8,
  minimumExposedDamCornerCapPixelsPerFace: 4,
  requireZeroInitialNarrowVerticalSlits: true,
  separatingCeilingContact: true,
  separationEvaluationAfter_s: 2,
  ceilingWetEvaluationStart_s: 1.5,
  ceilingContactEvaluationStart_s: 1.4,
  ceilingWetLayers: 2,
  seamRingCells: 2,
  seamClearanceCells: 2,
  liquidThreshold: 0.5,
  ceilingWetReportStart_s: 1.5,
  seamWetReportStart_s: 1,
  ceilingWetCellLimits: [{ before_s: 1.6, maximum: 9 }, { before_s: 1.7, maximum: 5 },
    { before_s: 2, maximum: 1 }, { maximum: 3 }],
  ceilingContactPixelLimits: [{ before_s: 1.5, maximum: 30 }, { before_s: 1.6, maximum: 18 },
    { maximum: 0 }],
} as const;

/** Read back from the document rather than restated here: the box-stack lane
 * wants eight of the figure's own steps, whatever the paper scenario authors. */
const paperFigureStep_s = sceneDocument(definition("dam-break-boxes")).numerics.maxDt_s;

const suiteList = [
  suite("dam-break-ui", "Exact browser water-box dam-break preset", { definitionId: "water-box-dam-break" }, {
    default: lane({ target_s: 0.2, oracleSteps: 2,
      collect: { fieldStats: "final" },
      maximumRepresentedVolumeDrift: 0.02,
      diagnostics: [] }),
    "octree-runtime": lane({ id: "octree-runtime", target_s: 2, exactSteps: 250, maxDt_s: 0.008, oracleSteps: 250,
      methods: methods(["losasso"]), collect: { stabilityEnvelope: true, checkpointEvery_s: 0.1, fieldStats: "checkpoints" } }),
    "one-step": lane({ id: "one-step", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["losasso"], { losasso: { globalFineLevelSetFactor: "4" } }),
      collect: { globalFineGeneration: true, fieldStats: "none" } }),
    "adaptive-ui-one-step": lane({ id: "adaptive-ui-one-step", target_s: 0.004,
      exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["losasso"], { losasso: {
        losassoVelocityExtension: "causal-front",
        maximumLeafSize: "16", interfaceRefinementBandCells: 4,
        globalFineLevelSetFactor: "1",
      } }),
      collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints",
        checkpointEvery_s: 0.004, raster: "checkpoints" },
      diagnostics: [], timeout_ms: 240_000 }),
    "adaptive-ui-two-step": lane({ id: "adaptive-ui-two-step", target_s: 0.008,
      exactSteps: 2, maxDt_s: 0.004, oracleSteps: 2,
      methods: methods(["losasso"], { losasso: {
        losassoVelocityExtension: "causal-front",
        maximumLeafSize: "16", interfaceRefinementBandCells: 4,
        globalFineLevelSetFactor: "1",
      } }),
      collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints",
        checkpointEvery_s: 0.004, raster: "checkpoints" },
      diagnostics: [], timeout_ms: 240_000 }),
    "two-step": lane({ id: "two-step", target_s: 0.016, exactSteps: 2, maxDt_s: 0.008, oracleSteps: 2,
      methods: methods(["losasso"]), collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints", checkpointEvery_s: 0.008,
        raster: "checkpoints", globalFineGeneration: true },
      hooks: [{ id: "water-raster-integrity", methods: ["losasso"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }], timeout_ms: 240_000 }),
    runtime: lane({ id: "runtime", target_s: 1.52, exactSteps: 190, maxDt_s: 0.008, oracleSteps: 190,
      methods: methods(["losasso"]), collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints", checkpointEvery_s: 0.1,
        raster: "checkpoints", globalFineGeneration: true },
      hooks: [{ id: "water-raster-integrity", methods: ["losasso"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }], timeout_ms: 240_000 }),
    performance: lane({ id: "performance", description: "Release-like profiler lane", target_s: 0.496, exactSteps: 62, maxDt_s: 0.008, oracleSteps: 62,
      methods: methods(["losasso"]), collect: { fieldStats: "none", performanceProfile: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    throughput: lane({ id: "throughput", description: "Warm command-throughput lane", target_s: 0.496, exactSteps: 62, maxDt_s: 0.008, oracleSteps: 62,
      methods: methods(["losasso"]), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
  }),

  suite("settled-tank", "Hydrostatic preservation in a closed level pool", { definitionId: "water-box-tank-fill", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 0.1, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
    acceptance: lane({ id: "acceptance", target_s: 0.0667, oracleSteps: 2,
      methods: methods(["losasso"]), collect: { stabilityEnvelope: true, fieldStats: "final", boundaryThetaHistogram: true },
      diagnostics: [equilibriumDiagnostic], timeout_ms: 240_000 }),
  }),
  suite("settled-tank-ui", "Exact browser settled-tank preset with rigid bodies", { definitionId: "water-box-tank-fill" }, {
    default: lane({ target_s: 0.2, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
    acceptance: lane({ id: "acceptance", target_s: 0.0667, oracleSteps: 2, methods: methods(["losasso"]),
      collect: { stabilityEnvelope: true, fieldStats: "final" }, diagnostics: [equilibriumDiagnostic], timeout_ms: 240_000 }),
  }),
  suite("dam-break-boxes", "Three-dimensional dam break with immersed boxes", { definitionId: "dam-break-boxes" }, {
    default: lane({ target_s: Math.max(paperFigureStep_s * 8, 0.05), oracleSteps: 2 }),
  }),
  suite("cm12-figure-2", "CM12 Figure 2 freely falling 2D liquid ball", { definitionId: "cm12-figure-2" }, {
    "free-fall": lane({ id: "free-fall", description: "Twenty paper steps reach the 0.67 s pre-impact plate",
      target_s: 20 / 30, exactSteps: 20, maxDt_s: 1 / 30, oracleSteps: 20,
      methods: methods(["uniform"], { uniform: { timeStep: "paper", densityPostProcessing: "off" } }),
      timeout_ms: 600_000,
      maximumRepresentedVolumeDrift: 0.03,
      collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
      diagnostics: [],
      acceptance: [
        { id: "cm12-figure2-grid", metric: "methods.uniform.grid", operator: "equal", expected: [128, 128, 8] },
        { id: "cm12-figure2-mass", metric: "methods.uniform.info.rawVolumeDrift.abs", operator: "at-most", expected: 0.001 },
        { id: "cm12-figure2-pressure", metric: "methods.uniform.info.uniformCM11aConverged", operator: "equal", expected: true },
        { id: "cm12-figure2-fim", metric: "methods.uniform.info.uniformFIMConverged", operator: "equal", expected: true },
        { id: "cm12-figure2-falls", metric: "methods.uniform.finalSummary.centroidCells.y", operator: "at-most", expected: 50 },
        { id: "cm12-figure2-no-overshoot", metric: "methods.uniform.finalSummary.centroidCells.y", operator: "at-least", expected: 44 },
        { id: "cm12-figure2-leading-edge", metric: "methods.uniform.finalSummary.wetBounds.minimum.y", operator: "at-least", expected: 34 },
        { id: "cm12-figure2-trailing-edge", metric: "methods.uniform.finalSummary.wetBounds.maximum.y", operator: "at-most", expected: 65 },
        { id: "cm12-figure2-coherent", metric: "methods.uniform.finalSummary.componentCount", operator: "equal", expected: 1 },
        { id: "cm12-figure2-retains-volume", metric: "methods.uniform.finalSummary.wetCells", operator: "at-least", expected: 4_800 },
        { id: "cm12-figure2-no-wall-tail", metric: "methods.uniform.finalSummary.interfaceFaceCount", operator: "at-most", expected: 1_200 },
      ],
    }),
  }, "free-fall"),
  suite("cm12-figure-3", "CM12 Figure 3 freely falling 2D liquid disks", { definitionId: "cm12-figure-3" }, {
    "free-fall": lane({ id: "free-fall", description: "Twenty paper steps expose pre-impact time-step deformation",
      target_s: 20 / 30, exactSteps: 20, maxDt_s: 1 / 30, oracleSteps: 20,
      methods: methods(["uniform"], { uniform: { timeStep: "paper", densityPostProcessing: "off" } }),
      timeout_ms: 600_000,
      maximumRepresentedVolumeDrift: 0.04,
      collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true }, diagnostics: [],
      acceptance: [
        { id: "cm12-figure3-grid", metric: "methods.uniform.grid", operator: "equal", expected: [128, 128, 8] },
        { id: "cm12-figure3-mass", metric: "methods.uniform.info.rawVolumeDrift.abs", operator: "at-most", expected: 0.001 },
        { id: "cm12-figure3-pressure", metric: "methods.uniform.info.uniformCM11aConverged", operator: "equal", expected: true },
        { id: "cm12-figure3-fim", metric: "methods.uniform.info.uniformFIMConverged", operator: "equal", expected: true },
        { id: "cm12-figure3-falls", metric: "methods.uniform.finalSummary.wetBounds.maximum.y", operator: "at-most", expected: 80 },
        { id: "cm12-figure3-no-overshoot", metric: "methods.uniform.finalSummary.wetBounds.maximum.y", operator: "at-least", expected: 74 },
        { id: "cm12-figure3-components", metric: "methods.uniform.finalSummary.componentCount", operator: "equal", expected: 5 },
        { id: "cm12-figure3-retains-volume", metric: "methods.uniform.finalSummary.wetCells", operator: "at-least", expected: 18_000 },
        { id: "cm12-figure3-coherent-interfaces", metric: "methods.uniform.finalSummary.interfaceFaceCount", operator: "at-most", expected: 3_000 },
      ],
    }),
  }, "free-fall"),
  suite("cm12-figure-8", "CM12 Figure 8 dam break in a spherical container", { definitionId: "cm12-figure-8" }, {
    motion: lane({ id: "motion", description: "Fifteen paper steps establish conserved downslope slosh",
      target_s: 0.5, exactSteps: 15, maxDt_s: 1 / 30, oracleSteps: 15,
      methods: methods(["uniform"], { uniform: { timeStep: "paper", densityPostProcessing: "off" } }),
      timeout_ms: 600_000,
      // Sec. 3.7 explicitly permits rho' > 1 during impact and removes it
      // gradually with the bounded artificial-divergence term.
      maximumStoredDensity: 3,
      collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
      diagnostics: [],
      acceptance: [
        { id: "cm12-figure8-grid", metric: "methods.uniform.grid", operator: "equal", expected: [128, 128, 128] },
        { id: "cm12-figure8-mass", metric: "methods.uniform.info.rawVolumeDrift.abs", operator: "at-most", expected: 0.001 },
        { id: "cm12-figure8-finite", metric: "methods.uniform.info.nonFiniteCount", operator: "equal", expected: 0 },
        { id: "cm12-figure8-pressure", metric: "methods.uniform.info.uniformCM11aConverged", operator: "equal", expected: true },
        { id: "cm12-figure8-fim", metric: "methods.uniform.info.uniformFIMConverged", operator: "equal", expected: true },
        { id: "cm12-figure8-paper-cfl", metric: "methods.uniform.stabilityEnvelope.peakLiquidSpeed_m_s", operator: "at-most", expected: 22 },
        { id: "cm12-figure8-falls", metric: "methods.uniform.finalSummary.centroidCells.y", operator: "at-most", expected: 63 },
        { id: "cm12-figure8-spreads", metric: "methods.uniform.finalSummary.centroidCells.x", operator: "at-least", expected: 33 },
      ],
    }),
  }, "motion"),
  suite("cm12-figure-12", "CM12 Figure 12 ball drop in a spherical container", { definitionId: "cm12-figure-12" }, {
    motion: lane({ id: "motion", description: "Fifteen paper steps establish a conserved fall toward the curved floor",
      target_s: 0.5, exactSteps: 15, maxDt_s: 1 / 30, oracleSteps: 15,
      methods: methods(["uniform"], { uniform: { timeStep: "paper", densityPostProcessing: "off" } }),
      timeout_ms: 600_000,
      collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
      diagnostics: [],
      acceptance: [
        { id: "cm12-figure12-grid", metric: "methods.uniform.grid", operator: "equal", expected: [128, 128, 128] },
        { id: "cm12-figure12-mass", metric: "methods.uniform.info.rawVolumeDrift.abs", operator: "at-most", expected: 0.001 },
        { id: "cm12-figure12-finite", metric: "methods.uniform.info.nonFiniteCount", operator: "equal", expected: 0 },
        { id: "cm12-figure12-pressure", metric: "methods.uniform.info.uniformCM11aConverged", operator: "equal", expected: true },
        { id: "cm12-figure12-fim", metric: "methods.uniform.info.uniformFIMConverged", operator: "equal", expected: true },
        { id: "cm12-figure12-falls", metric: "methods.uniform.finalSummary.centroidCells.y", operator: "at-most", expected: 88 },
        { id: "cm12-figure12-coherent", metric: "methods.uniform.stabilityEnvelope.minimumDominantComponentFraction", operator: "at-least", expected: 0.98 },
      ],
    }),
  }, "motion"),
  suite("mass-conserving-figure-9-dam-break",
    "CM12 Figure 9 initial dam phase at the paper's published lattice and timestep",
    { definitionId: "mass-conserving-figure-9-dam-break" }, {
      "uniform-one-step": lane({ id: "uniform-one-step",
        description: "One exact 1/30 s paper step on the published 128x128x64 lattice",
        target_s: 1 / 30, exactSteps: 1, maxDt_s: 1 / 30, oracleSteps: 1,
        methods: methods(["uniform"], { uniform: {
          timeStep: "paper", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
        diagnostics: [],
        acceptance: [
          { id: "cm12-figure9-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [128, 128, 64] },
          { id: "cm12-figure9-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "cm12-figure9-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
          { id: "cm12-figure9-fim-converged",
            metric: "methods.uniform.info.uniformFIMConverged",
            operator: "equal", expected: true },
          { id: "cm12-figure9-raw-mass-lower",
            metric: "methods.uniform.info.rawVolumeDrift",
            operator: "at-least", expected: -0.001 },
          { id: "cm12-figure9-raw-mass-upper",
            metric: "methods.uniform.info.rawVolumeDrift",
            operator: "at-most", expected: 0.001 },
        ],
      }),
      "uniform-4ms-regression": lane({ id: "uniform-4ms-regression",
        description: "Exact Figure 9 low-timestep regression through the former 0.344 s collapse",
        target_s: 0.4, exactSteps: 100, maxDt_s: 0.004, oracleSteps: 100,
        methods: methods(["uniform"], { uniform: {
          timeStep: "scene", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
        diagnostics: [],
        acceptance: [
          { id: "cm12-figure9-4ms-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [128, 128, 64] },
          { id: "cm12-figure9-4ms-final-density",
            metric: "methods.uniform.finalSummary.maximum",
            operator: "at-most", expected: 1.1 },
          { id: "cm12-figure9-4ms-mass",
            metric: "methods.uniform.info.rawVolumeDrift.abs",
            operator: "at-most", expected: 0.001 },
          { id: "cm12-figure9-4ms-dominant-component",
            metric: "methods.uniform.stabilityEnvelope.minimumDominantComponentFraction",
            operator: "at-least", expected: 0.99 },
          { id: "cm12-figure9-4ms-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "cm12-figure9-4ms-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
          { id: "cm12-figure9-4ms-fim-converged",
            metric: "methods.uniform.info.uniformFIMConverged",
            operator: "equal", expected: true },
        ],
      }),
    }, "uniform-one-step"),
  suite("hose-tank", "Fixed cylindrical inflow into a shallow receiving pool", { definitionId: "hose-tank" }, {
    default: lane({ target_s: 0.5, oracleSteps: 2, maximumRepresentedVolumeDrift: false,
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance }),
    drift: lane({ id: "drift", target_s: 0.4611111111111111, exactSteps: 166, maxDt_s: 0.002777777777777778, oracleSteps: 166,
      methods: methods(["losasso"]), maximumRepresentedVolumeDrift: false,
      collect: { fieldStats: "final", evidenceCollectors: [{ id: "hose-jet-drift", phase: "terminal", methods: ["losasso"],
        requires: ["compact velocity"], provides: ["structured velocity", "inflow geometry"] }] },
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance,
      hooks: [{ id: "hose-jet-drift", methods: ["losasso"], requires: ["structured velocity", "inflow geometry"], parameters: {
        compareAgainstCoarseCellWidth: true, minimumSampledVelocityCells: 1, minimumOutletAxialSpeed_m_s: 0.05,
        maximumOutletSideSpeed_m_s: 0.1, maximumSideCentroidOffsetCoarseCells: 0.5,
        minimumSampledAirborneBins: 3, minimumAirborneAxialRetentionRatio: 0.9,
        maximumAdjacentAxialSpeedDropRatio: 0.12, maximumAdjacentMomentumFluxDropRatio: 0.25,
        minimumBallisticGravityVelocityRatio: 0.7, maximumBallisticGravityVelocityRatio: 3.1,
        maximumBallisticCenterlineRelativeError: 1.1,
      } }], timeout_ms: 240_000 }),
  }),
  suite("sphere-jet", "Directed inlet jet past a fixed immersed sphere", { definitionId: "sphere-jet" }, {
    default: lane({ target_s: 0.5, oracleSteps: 2, maximumRepresentedVolumeDrift: false,
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance }),
  }),
  suite("deep-water", "Extreme vertical aspect ratio", { definitionId: "deep-water-ab", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 0.1, oracleSteps: 1, diagnostics: [equilibriumDiagnostic] }),
  }),
  suite("garden-pond", "Hydrostatic rest in an organic terrain pool", { definitionId: "garden-pond", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 0.1, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
  }),
  suite("garden-hose", "Authored terrain pond with continuous hose inflow", { definitionId: "garden-hose" }, {
    default: lane({ target_s: 0.5, oracleSteps: 2, methods: methods(["losasso"]),
      maximumRepresentedVolumeDrift: false, collect: { fieldStats: "final" },
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance, timeout_ms: 240_000 }),
  }),
  suite("garden-dam-break", "Single fluid brick released down terrain into the pond", { definitionId: "garden-dam-break", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 0.2, oracleSteps: 2 }),
    migration: lane({ id: "migration", target_s: 1, oracleSteps: 2, methods: methods(["losasso"]),
      collect: { fieldStats: "none", sparsePublication: true },
      hooks: [{ id: "garden-source-brick-migration", methods: ["losasso"], requires: ["initial fluid brick stats", "final sparse publication"],
        parameters: { initialCoreBricks: 1, evaluateAfter_s: 1, minimumFinalCoreBricks: 2,
          sourceFluidVoxelsAtEnd: 0, sourceCoreResidencyAtEnd: false,
          requireResidentCountBelowCapacity: true } }] }),
  }),
  suite("brick-quad-dam-break", "Four-brick tank whose release crosses every brick boundary", { definitionId: "brick-quad-dam-break", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 1.5, oracleSteps: 2,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.25, spatialField: true, sparsePublication: true },
      hooks: [{ id: "brick-quad-coverage", requires: ["checkpoint fields", "initial fluid brick stats", "final sparse publication"],
        parameters: { expectedBrickGrid: [2, 1, 2], minimumFirstCheckpointColumns: 2,
          minimumInitialResidentBricks: 1, minimumFinalResidentBricks: 2, minimumFinalCoreBricks: 2,
          liquidThreshold: 0.5, requireResidentCountBelowCapacity: true } }] }),
    "uniform-detail-first-step": lane({ id: "uniform-detail-first-step",
      description: "Two DETAIL x2 edits followed by one 1/30 s Uniform step",
      target_s: 1 / 30, exactSteps: 1, maxDt_s: 1 / 30, oracleSteps: 1,
      methods: methods(["uniform"]),
      collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
      diagnostics: [],
      acceptance: [
        { id: "detail-grid", metric: "methods.uniform.grid", operator: "equal", expected: [64, 32, 64] },
        { id: "detail-pressure-converged", metric: "methods.uniform.info.uniformCM11aConverged", operator: "equal", expected: true },
        { id: "detail-pressure-cap", metric: "methods.uniform.info.uniformCM11aCapFailure", operator: "equal", expected: false },
        { id: "detail-first-step-speed", metric: "methods.uniform.info.maxSpeed_m_s", operator: "at-most", expected: 2 },
        { id: "detail-first-step-volume", metric: "methods.uniform.info.volumeDrift.abs", operator: "at-most", expected: 1e-6 },
      ],
    }),
  }),
  suite("twin-dam-collision", "Opposed seeded reservoirs collapsing into an oblique collision", { definitionId: "twin-dam-collision", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 1, oracleSteps: 2, collect: { stabilityEnvelope: true, fieldStats: "checkpoints", checkpointEvery_s: 0.1 } }),
  }),
  suite("hydrostatic-power-two-level", "16-cubed settled leaf-32 power grid", { definitionId: "hydrostatic-power-two-level" }, {
    default: lane({ target_s: 0.2, exactSteps: 50, maxDt_s: 0.004, oracleSteps: 50,
      methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.2, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: exhaustivePowerDiagnostics(1e-4),
      acceptance: [...powerAcceptance,
        { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [16, 16, 16] },
        { id: "hydrostatic-power-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 1e-4 }] }),
    "uniform-16ms": lane({ id: "uniform-16ms",
      description: "Dense hydrostatic-rest regression at the interactive 16 ms scene step",
      target_s: 0.32, exactSteps: 20, maxDt_s: 0.016, oracleSteps: 20,
      // The lane's whole point is the 16 ms interactive step; the solver's
      // paper-dt default (1/30 s) can never fit inside a 16 ms advance and
      // rejects forever, so pin the scene time step explicitly.
      methods: methods(["uniform"], { uniform: { timeStep: "scene" } }), timeout_ms: 240_000,
      collect: { fieldStats: "final", checkpointEvery_s: 0.016, spatialField: true },
      diagnostics: [],
      acceptance: [
        { id: "uniform-hydrostatic-grid", metric: "methods.uniform.grid", operator: "equal", expected: [16, 16, 16] },
        { id: "uniform-hydrostatic-speed", metric: "methods.uniform.info.maxSpeed_m_s", operator: "at-most", expected: 1e-4 },
        { id: "uniform-hydrostatic-volume-drift", metric: "methods.uniform.info.representedVolumeDrift.abs", operator: "at-most", expected: 1e-6 },
      ],
    }),
  }),
  suite("hydrostatic-power-large-offset", "32x24x16 cell-cut leaf-32 power grid", { definitionId: "hydrostatic-power-large-offset" }, {
    default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["power-liquids"], { "power-liquids": { ...frozenPowerReferenceOverrides, interfaceRefinementBandCells: 4 } }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: true } },
      diagnostics: [powerDiagnostics[0], powerDiagnostics[2],
        { id: "authoritative-water-raster", methods: ["power-liquids"], parameters: standardWaterRasterParameters }],
      acceptance: [{ id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 24, 16] }],
      hooks: [{ id: "water-raster-integrity", methods: ["power-liquids"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }] }),
  }),
  suite("minimal-power-dam-break", "Minimal dynamic leaf-32 analytic dam", { definitionId: "minimal-power-dam-break" }, {
    default: lane({ target_s: 2, exactSteps: 500, maxDt_s: 0.004, oracleSteps: 500,
      methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }), timeout_ms: 240_000,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.1, energyEverySteps: 50, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: [...exhaustivePowerDiagnostics(0.01),
        { id: "authoritative-water-raster", methods: ["power-liquids"], parameters: minimalDamRasterParameters }],
      acceptance: [...powerAcceptance,
        { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [16, 16, 16] },
        { id: "minimal-power-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 0.01 },
        { id: "minimal-power-connectivity", metric: "methods.*.stabilityEnvelope.minimumDominantComponentFraction", operator: "at-least", expected: 0.985 },
        { id: "minimal-power-variational-residual", metric: "methods.*.stabilityEnvelope.maximumProjectedVariationalResidual", operator: "at-most", expected: 3.5e-6 },
        { id: "fine-transport-unavailable", metric: "methods.*.finalGlobalFineGeneration.transportVelocityUnavailableFraction", operator: "at-most", expected: 0.08 },
        { id: "fine-transport-outside-band", metric: "methods.*.finalGlobalFineGeneration.transportDepartureOutsideBandFraction", operator: "at-most", expected: 0.005 },
        { id: "fine-transport-committed", metric: "methods.*.finalGlobalFineGeneration.transportCommitted", operator: "equal", expected: true },
        { id: "fine-transport-processed", metric: "methods.*.finalGlobalFineGeneration.transportProcessed", operator: "at-least", expected: 1 },
        { id: "fine-transport-unavailable-accounted", metric: "methods.*.finalGlobalFineGeneration.transportNonfiniteVelocity", operator: "equal",
          expected: { selector: "methods.*.finalGlobalFineGeneration.transportVelocityUnavailable" } },
        { id: "fine-topology-not-rolled-back", metric: "methods.*.finalGlobalFineGeneration.topologyRolledBack", operator: "equal", expected: false }],
      hooks: [{ id: "minimal-dam-motion", methods: ["power-liquids"], requires: ["stability envelope", "checkpoint fields", "mechanical energy"],
          parameters: minimalDamMotionParameters },
        { id: "water-raster-integrity", methods: ["power-liquids"], requires: ["global fine generation", "front/back raster"],
          parameters: minimalDamRasterParameters }] }),
    "two-step": lane({ id: "two-step", target_s: 0.008, exactSteps: 2, maxDt_s: 0.004, oracleSteps: 2,
      methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }), timeout_ms: 240_000,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: [...exhaustivePowerDiagnostics(0.01),
        { id: "authoritative-water-raster", methods: ["power-liquids"], parameters: minimalDamRasterParameters }], acceptance: powerAcceptance,
      hooks: [{ id: "water-raster-integrity", methods: ["power-liquids"], requires: ["global fine generation", "front/back raster"],
        parameters: minimalDamRasterParameters }] }),
    performance: lane({ id: "performance", target_s: 0.248, exactSteps: 62, maxDt_s: 0.004, oracleSteps: 62,
      methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    throughput: lane({ id: "throughput", target_s: 0.248, exactSteps: 62, maxDt_s: 0.004, oracleSteps: 62,
      methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    motion: lane({ id: "motion", target_s: 0.5, oracleSteps: 1, methods: methods(["power-liquids"], { "power-liquids": frozenPowerReferenceOverrides }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.016, spatialField: true, raster: "checkpoints", globalFineGeneration: true },
      hooks: [{ id: "minimal-dam-motion", methods: ["power-liquids"], requires: ["stability envelope", "initial/final raster"],
        parameters: minimalDamMotionParameters }], timeout_ms: 240_000 }),
    "fine-factor-4": lane({ id: "fine-factor-4", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["power-liquids"], { "power-liquids": { ...frozenPowerReferenceOverrides, globalFineLevelSetFactor: "4" } }),
      collect: { fieldStats: "final", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: powerDiagnostics, timeout_ms: 240_000 }),
    "fine-factor-8": lane({ id: "fine-factor-8", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["power-liquids"], { "power-liquids": { ...frozenPowerReferenceOverrides, globalFineLevelSetFactor: "8" } }),
      collect: { fieldStats: "final", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: powerDiagnostics, timeout_ms: 240_000 }),
  }),
  suite("high-resolution-dam-break", "Interactive 128-cubed coarse-band analytic dam",
    { definitionId: "high-resolution-dam-break" }, {
      performance: lane({ id: "performance", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true },
        diagnostics: [{ id: "performance" }], timeout_ms: 240_000,
        acceptance: [{ id: "expected-grid", metric: "methods.*.grid",
          operator: "equal", expected: [128, 128, 128] }] }),
    }, "performance"),
  suite("symmetric-expansion", "Central 2x1x2-brick liquid body with horizontal D4 symmetry",
    { definitionId: "symmetric-expansion" }, {
      default: lane({ target_s: 1, exactSteps: 250, maxDt_s: 0.004, oracleSteps: 250,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }), timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          stabilityEnvelope: true, energyEverySteps: 10,
          structuredValidation: true, raster: "initial-final", globalFineGeneration: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "radial front circularity", "four-wall contact"] }],
        },
        diagnostics: [inviscidStabilityDiagnostic],
        acceptance: [
          { id: "balanced-octree-grid", metric: "methods.*.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
          { id: "paper-pressure-residual", metric: "methods.*.stabilityEnvelope.maximumPressureRelativeResidual", operator: "at-most", expected: 1e-8 },
          { id: "post-projection-divergence", metric: "methods.*.stabilityEnvelope.maximumProjectedVariationalResidual", operator: "at-most", expected: 1e-5 },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "radial front circularity", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: true,
            minimumCheckpointCount: 250,
            maximumWallContactStepSpread: 0,
            circularityEvaluationStart_s: 0.168,
            circularityEvaluationEnd_s: 0.2,
            frontAdvanceEvaluationEnd_s: 0.2,
            // Aug-8 control: 3.630 cells. The adaptive lane must improve on,
            // not merely reproduce, that already-dissipative result.
            minimumMeanFrontAdvance_cells: 3.7,
            wallClimbEvaluationStart_s: 0.24,
            // Aug-8 control reached eight cells on every wall.
            minimumPeakWallClimb_cells: 9,
            maximumAxisDiagonalFrontDifference_cells: 1,
            maximumRadialRmsDeviation_cells: 0.5,
            maximumRadialDeviation_cells: 1,
            minimumCircularityAngularSamples: 64,
          } }],
      }),
      "comparison-uniform": lane({ id: "comparison-uniform",
        description: "Dense arm of the matched-step uniform versus Losasso benchmark",
        target_s: 1, exactSteps: 250, maxDt_s: 0.004, oracleSteps: 250,
        methods: methods(["uniform"]),
        timeout_ms: 600_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.04, spatialField: true,
          energyEverySteps: 10,
        },
        diagnostics: [],
        acceptance: [
          { id: "uniform-grid", metric: "methods.uniform.grid", operator: "equal",
            expected: [32, 16, 32] },
        ],
      }),
      "comparison-losasso": lane({ id: "comparison-losasso",
        description: "Adaptive arm of the matched-step uniform versus Losasso benchmark",
        target_s: 1, exactSteps: 250, maxDt_s: 0.004, oracleSteps: 250,
        methods: methods(["losasso"], { losasso: {
          ...symmetricExpansionOverrides,
          globalFineLevelSetFactor: "1",
        } }),
        timeout_ms: 600_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.04, spatialField: true,
          energyEverySteps: 10,
        },
        diagnostics: [],
        acceptance: [
          { id: "losasso-grid", metric: "methods.*.grid", operator: "equal",
            expected: [32, 16, 32] },
          { id: "losasso-backend", metric: "methods.*.info.coarseDynamicsBackend",
            operator: "equal", expected: "losasso" },
        ],
      }),
      "coarse-only": lane({ id: "coarse-only", target_s: 1, exactSteps: 250,
        maxDt_s: 0.004, oracleSteps: 250,
        methods: methods(["losasso"], { losasso: {
          ...symmetricExpansionOverrides, globalFineLevelSetFactor: "1",
        } }), timeout_ms: 600_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          stabilityEnvelope: true, structuredValidation: true, raster: "initial-final",
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "radial front circularity", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
          { id: "paper-pressure-residual", metric: "methods.*.stabilityEnvelope.maximumPressureRelativeResidual", operator: "at-most", expected: 1e-8 },
          { id: "post-projection-divergence", metric: "methods.*.stabilityEnvelope.maximumProjectedVariationalResidual", operator: "at-most", expected: 1e-5 },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "radial front circularity", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: true,
            minimumCheckpointCount: 250,
            maximumWallContactStepSpread: 0,
            circularityEvaluationStart_s: 0.168,
            circularityEvaluationEnd_s: 0.2,
            frontAdvanceEvaluationEnd_s: 0.2,
            minimumMeanFrontAdvance_cells: 3.7,
            wallClimbEvaluationStart_s: 0.24,
            minimumPeakWallClimb_cells: 9,
            maximumAxisDiagonalFrontDifference_cells: 1,
            maximumRadialRmsDeviation_cells: 0.5,
            maximumRadialDeviation_cells: 1,
            minimumCircularityAngularSamples: 64,
          } }],
      }),
      "raster-construction": lane({ id: "raster-construction", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: {
          ...symmetricExpansionOverrides, globalFineLevelSetFactor: "4",
        } }), timeout_ms: 240_000,
        collect: { fieldStats: "final", raster: "initial-final", globalFineGeneration: true },
        diagnostics: [],
      }),
      "one-step": lane({ id: "one-step", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requirePressureStageAudit: true,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 1,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "two-step": lane({ id: "two-step", target_s: 0.008, exactSteps: 2,
        maxDt_s: 0.004, oracleSteps: 2,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 2,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "three-step": lane({ id: "three-step", target_s: 0.012, exactSteps: 3,
        maxDt_s: 0.004, oracleSteps: 3,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 3,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "twenty-step": lane({ id: "twenty-step", target_s: 0.08, exactSteps: 20,
        maxDt_s: 0.004, oracleSteps: 20,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 20,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "fine-factor-4": lane({ id: "fine-factor-4", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: {
          ...symmetricExpansionOverrides,
          globalFineLevelSetFactor: "4",
        } }), timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true, raster: "initial-final", globalFineGeneration: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "losasso-coarse-backend", metric: "methods.*.info.coarseDynamicsBackend", operator: "equal", expected: "losasso" },
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["losasso"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 1,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "power2017-factor-4": lane({ id: "power2017-factor-4", target_s: 0.012,
        exactSteps: 3, maxDt_s: 0.004, oracleSteps: 3,
        methods: methods(["power-liquids"], {
          "power-liquids": POWER2017_FACTOR4_BENCHMARK_METHOD_PROFILE.overrides,
        }), timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true, raster: "initial-final", globalFineGeneration: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["power-liquids"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "power2017-coarse-backend", metric: "methods.*.info.coarseDynamicsBackend",
            operator: "equal", expected: "power2017" },
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal",
            expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["power-liquids"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "four-wall contact"],
          parameters: {
            ...symmetricExpansionFieldLimits,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 3,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      /**
       * Wall-clock lane for the same fine-factor-4 configuration the symmetry
       * oracle gates on. It carries no evidence collectors or symmetry hooks so
       * the profiled frame is the shipping command graph rather than the
       * validation one; correctness stays with `fine-factor-4`, which is the
       * lane that must still pass after any change scored here.
       */
      performance: lane({ id: "performance", target_s: 0.248, exactSteps: 62,
        maxDt_s: 0.004, oracleSteps: 62,
        methods: methods(["losasso"], { losasso: symmetricExpansionOverrides }),
        collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true },
        diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    }),
  suite("minimal-power-dam-break-32", "32-cubed coarse-only analytic mini dam",
    { definitionId: "minimal-power-dam-break-32" }, {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 32, 32] }] }),
      "two-step": lane({ id: "two-step", target_s: 0.008, exactSteps: 2,
        maxDt_s: 0.004, oracleSteps: 2,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004,
          spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          raster: "checkpoints", globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 32, 32] }] }),
      "early-motion": lane({ id: "early-motion", target_s: 0.044, exactSteps: 11,
        maxDt_s: 0.004, oracleSteps: 11,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004,
          spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          raster: "checkpoints", globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 32, 32] }],
        hooks: [{ id: "minimal-dam-motion", methods: ["losasso"],
          requires: ["stability envelope", "checkpoint fields", "mechanical energy", "front/back raster"],
          parameters: {
            minimumPeakSpeed_m_s: 0.4,
            minimumFinalSpeed_m_s: 0.4,
            minimumLiquidVolumeRetentionRatio: 0.85,
            minimumLateralSpread_m: 0,
            maximumStepPotentialEnergyIncreaseFraction: 0.005,
            maximumStepKineticEnergyDropFraction: 0.6,
            maximumStepLiquidCellGrowthRatio: 1.5,
            energyEvaluationAfter_s: 0,
          } }] }),
      runtime: lane({ id: "runtime", target_s: 0.276, exactSteps: 69,
        maxDt_s: 0.004, oracleSteps: 69,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004,
          spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          raster: "checkpoints", globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 32, 32] }],
        hooks: [{ id: "minimal-dam-motion", methods: ["losasso"],
          requires: ["stability envelope", "checkpoint fields", "mechanical energy", "front/back raster"],
          parameters: {
            minimumPeakSpeed_m_s: 0.4,
            minimumFinalSpeed_m_s: 0.4,
            minimumLiquidVolumeRetentionRatio: 0.85,
            minimumLateralSpread_m: 0.05,
            maximumStepPotentialEnergyIncreaseFraction: 0.005,
            maximumStepKineticEnergyDropFraction: 0.6,
            maximumStepLiquidCellGrowthRatio: 1.5,
            energyEvaluationAfter_s: 0,
          } }] }),
      "surface-regression": lane({ id: "surface-regression", target_s: 0.5, exactSteps: 125,
        maxDt_s: 0.004, oracleSteps: 125,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          raster: "checkpoints", checkpointEvery_s: 0.5, globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [32, 32, 32] },
          { id: "coarse-surface-interface-area",
            metric: "methods.*.finalGlobalFineGeneration.volumeInterfaceArea",
            operator: "at-most", expected: 1.1 },
          { id: "coarse-surface-active-cubes",
            metric: "methods.*.finalGlobalFineRaster.activeCubeCount",
            operator: "at-most", expected: 5_200 },
          { id: "coarse-surface-resident-pages",
            metric: "methods.*.finalGlobalFineGeneration.activePages",
            operator: "at-most", expected: 480 },
          { id: "coarse-surface-no-rejected-generations",
            metric: "methods.*.finalGlobalFineGeneration.topologyRejectionCount",
            operator: "equal", expected: 0 },
          { id: "coarse-surface-transport-committed",
            metric: "methods.*.finalGlobalFineGeneration.transportCommitted",
            operator: "equal", expected: true },
          { id: "coarse-surface-topology-not-rolled-back",
            metric: "methods.*.finalGlobalFineGeneration.topologyRolledBack",
            operator: "equal", expected: false }] }),
      "uniform-ceiling": lane({ id: "uniform-ceiling",
        description: "Dense CM12 dam break through the first closed-ceiling impact window",
        target_s: 0.5, exactSteps: 125, maxDt_s: 0.004, oracleSteps: 125,
        methods: methods(["uniform"], { uniform: {
          timeStep: "scene", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.02,
          spatialField: true, stabilityEnvelope: true, energyEverySteps: 5 },
        diagnostics: [],
        acceptance: [
          { id: "uniform-ceiling-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [32, 32, 32] },
          { id: "uniform-ceiling-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "uniform-ceiling-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
          { id: "uniform-ceiling-fim-converged",
            metric: "methods.uniform.info.uniformFIMConverged",
            operator: "equal", expected: true },
          { id: "uniform-ceiling-raw-mass-lower",
            metric: "methods.uniform.info.rawVolumeDrift",
            operator: "at-least", expected: -0.005 },
          { id: "uniform-ceiling-raw-mass-upper",
            metric: "methods.uniform.info.rawVolumeDrift",
            operator: "at-most", expected: 0.005 },
        ],
      }),
      "uniform-small-step-soak": lane({ id: "uniform-small-step-soak",
        description: "Three-second CM12 density/pressure stability soak at a 4 ms step",
        target_s: 3, exactSteps: 750, maxDt_s: 0.004, oracleSteps: 750,
        methods: methods(["uniform"], { uniform: {
          timeStep: "scene", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.02,
          spatialField: true, stabilityEnvelope: true, energyEverySteps: 5 },
        diagnostics: [],
        acceptance: [
          { id: "uniform-small-step-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [32, 32, 32] },
          { id: "uniform-small-step-final-density",
            metric: "methods.uniform.finalSummary.maximum",
            operator: "at-most", expected: 1.1 },
          { id: "uniform-small-step-peak-speed",
            metric: "methods.uniform.stabilityEnvelope.peakLiquidSpeed_m_s",
            operator: "at-most", expected: 12 },
          { id: "uniform-small-step-mechanical-energy",
            metric: "methods.uniform.energyTraceSummary.maximumMechanicalEnergyRatio",
            operator: "at-most", expected: 1.02 },
          { id: "uniform-small-step-mass",
            metric: "methods.uniform.info.rawVolumeDrift.abs",
            operator: "at-most", expected: 0.001 },
          { id: "uniform-small-step-dominant-component",
            metric: "methods.uniform.stabilityEnvelope.minimumDominantComponentFraction",
            operator: "at-least", expected: 0.99 },
          { id: "uniform-small-step-component-count",
            metric: "methods.uniform.stabilityEnvelope.maximumComponentCount",
            operator: "at-most", expected: 20 },
          { id: "uniform-small-step-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "uniform-small-step-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
        ],
      }),
    }),
  suite("minimal-power-dam-break-64", "64-cubed coarse-only analytic mini dam",
    { definitionId: "minimal-power-dam-break-64" }, {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["losasso"], { losasso: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true },
        acceptance: [
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 64, 64] }] }),
      "uniform-one-step": lane({ id: "uniform-one-step",
        description: "Dense CM12 64-cubed cold-start and first-step stability regression",
        target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["uniform"], { uniform: {
          timeStep: "scene", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true },
        diagnostics: [],
        acceptance: [
          { id: "uniform-64-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [64, 64, 64] },
          { id: "uniform-64-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "uniform-64-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
          { id: "uniform-64-fim-converged",
            metric: "methods.uniform.info.uniformFIMConverged",
            operator: "equal", expected: true },
        ],
      }),
      "uniform-paper-slosh": lane({ id: "uniform-paper-slosh",
        description: "Dense CM12 64-cubed paper-step wall/ceiling slosh without late boiling",
        target_s: 3, exactSteps: 90, maxDt_s: 1 / 30, oracleSteps: 90,
        methods: methods(["uniform"], { uniform: {
          timeStep: "paper", densityPostProcessing: "off",
        } }),
        timeout_ms: 600_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.1,
          spatialField: true, stabilityEnvelope: true, energyEverySteps: 3 },
        diagnostics: [],
        acceptance: [
          { id: "uniform-paper-slosh-grid", metric: "methods.uniform.grid",
            operator: "equal", expected: [64, 64, 64] },
          { id: "uniform-paper-slosh-pressure-converged",
            metric: "methods.uniform.info.uniformCM11aConverged",
            operator: "equal", expected: true },
          { id: "uniform-paper-slosh-pressure-cap",
            metric: "methods.uniform.info.uniformCM11aCapFailure",
            operator: "equal", expected: false },
          { id: "uniform-paper-slosh-fim-converged",
            metric: "methods.uniform.info.uniformFIMConverged",
            operator: "equal", expected: true },
          { id: "uniform-paper-slosh-raw-mass",
            metric: "methods.uniform.info.rawVolumeDrift.abs",
            operator: "at-most", expected: 0.001 },
          // The run-max component count peaks during the wall-impact splash,
          // where CM12 sheds real droplets by design; it barely separates a
          // healthy splash (44) from boiling (47). Keep it as a sanity
          // ceiling and let the three gates below carry the boiling
          // signature: sustained excess-density jets (peak speed), surface
          // shredding (dominant fraction), and failure to re-coalesce after
          // the impact window (final fragment count).
          { id: "uniform-paper-slosh-fragment-count",
            metric: "methods.uniform.stabilityEnvelope.maximumComponentCount",
            operator: "at-most", expected: 48 },
          { id: "uniform-paper-slosh-final-fragments",
            metric: "methods.uniform.finalSummary.componentCount",
            operator: "at-most", expected: 2 },
          { id: "uniform-paper-slosh-dominant-component",
            metric: "methods.uniform.stabilityEnvelope.minimumDominantComponentFraction",
            operator: "at-least", expected: 0.998 },
          { id: "uniform-paper-slosh-peak-speed",
            metric: "methods.uniform.stabilityEnvelope.peakLiquidSpeed_m_s",
            operator: "at-most", expected: 25 },
          { id: "uniform-paper-slosh-projection-does-not-amplify-divergence",
            metric: "methods.uniform.energyTraceSummary.projectionAmplifiedRmsDivergenceSamples",
            operator: "equal", expected: 0 },
        ],
      }),
    }),
  suite("large-power-dam-break", "20x-volume authored dam cold-start and one-step regression",
    { definitionId: "large-power-dam-break" }, {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["power-liquids"], { "power-liquids": largePowerDamOverrides }), timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 20, 64] }] }),
      "runtime-150": lane({ id: "runtime-150", target_s: 0.6, exactSteps: 150,
        maxDt_s: 0.004, oracleSteps: 150,
        methods: methods(["power-liquids"], { "power-liquids": largePowerDamOverrides }), timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.1, spatialField: true,
          stabilityEnvelope: true, structuredValidation: true, globalFineGeneration: true,
          powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: exhaustivePowerDiagnostics(0.01),
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 20, 64] },
          { id: "large-power-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 0.01 }] }),
    }),
  suite("large-power-hydrostatic", "20x-volume scene with a quarter-volume 1,024-cell sparse pool",
    { definitionId: "large-power-hydrostatic" }, {
      default: lane({ target_s: 0.96, exactSteps: 240, maxDt_s: 0.004, oracleSteps: 240, methods: methods(["power-liquids"], { "power-liquids": largePowerHydrostaticOverrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.12, spatialField: true,
          stabilityEnvelope: true, structuredValidation: true, globalFineGeneration: true,
          powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: exhaustivePowerDiagnostics(1e-4),
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 20, 64] },
          { id: "large-hydrostatic-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 1e-4 }] }),
    }),
  // The Bet-4.2 lane. `deep-hydrostatic-interior-coarsening` is the whole
  // reason this scene exists: 163,840 liquid cells that publish at most 61,440
  // rows is 0.375 rows/cell, against `large-power-hydrostatic`'s 1.004. If the
  // interior does not coarsen this rule fails by name instead of the run
  // silently overflowing a 65,536-row arena, and the shallow lane cannot state
  // the claim at all because every one of its cells is interface.
  suite("deep-power-hydrostatic", "Deep 20x still tank: the interior-coarsening measurement lane",
    { definitionId: "deep-power-hydrostatic" }, {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
        methods: methods(["power-liquids"], { "power-liquids": deepPowerHydrostaticOverrides }), timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 48, 64] },
          { id: "deep-hydrostatic-interior-coarsening", metric: "methods.*.info.pressureRequiredRows",
            operator: "at-most", expected: 61_440 },
          { id: "deep-hydrostatic-row-arena-fits", metric: "methods.*.info.pressureCapacityOverflow",
            operator: "equal", expected: false }] }),
      "runtime-240": lane({ id: "runtime-240", target_s: 0.96, exactSteps: 240,
        maxDt_s: 0.004, oracleSteps: 240,
        // `MAXIMUM_WEBGPU_SMOKE_TIMEOUT_MS` is the ceiling the isolated runner
        // enforces, so 240 deep steps must fit 240 s of wall clock: about
        // 1 s/advance. Shorten a first capture with `--steps=N` on the
        // benchmark lane rather than raising a safety envelope.
        methods: methods(["power-liquids"], { "power-liquids": deepPowerHydrostaticOverrides }), timeout_ms: 240_000,
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.12, spatialField: true,
          stabilityEnvelope: true, structuredValidation: true, globalFineGeneration: true,
          powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: exhaustivePowerDiagnostics(1e-4),
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: [64, 48, 64] },
          { id: "deep-hydrostatic-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 1e-4 },
          { id: "deep-hydrostatic-interior-coarsening", metric: "methods.*.info.pressureRequiredRows",
            operator: "at-most", expected: 61_440 },
          { id: "deep-hydrostatic-row-arena-fits", metric: "methods.*.info.pressureCapacityOverflow",
            operator: "equal", expected: false }] }),
    }),
  // The droplet-in-a-vast-domain sweep. One hundred liquid cells at every N, so
  // the pinned t=0 counters below are the family's invariant rather than
  // per-scene baselines: every number here is the *same* number at 64 cubed and
  // at 256 cubed, and a lane that needs a larger one is publishing
  // domain-shaped work — the exact defect the sweep hunts. Measured at 64
  // cubed: 100 air-support rows, 624 resident fine bricks (117 of them
  // interface), against a 262,144-brick logical lattice that is 16.8M at 256
  // cubed. The pins sit at 1,024 and 2,048 — an order of magnitude above the
  // fluid and four below anything domain-proportional, so they discriminate
  // exactly the thing worth discriminating and leave room for the slump
  // transient.
  ...POWER_DROPLET_EDGE_CELLS.map((edgeCells) => {
    const grid = [edgeCells, edgeCells, edgeCells];
    const pinned: readonly SceneWebGPUAcceptanceRule[] = [
      { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: grid },
      { id: "power-droplet-pinned-air-support-rows",
        metric: "methods.*.info.structuredAirSupportRows",
        operator: "at-most", expected: 1_024 },
      { id: "power-droplet-pinned-fine-residency",
        metric: "methods.*.info.globalFineActiveBricks",
        operator: "at-most", expected: 2_048 },
      // `pressureRequiredRows` is the solve's overflow lower bound and reads 0
      // whenever the arena holds, so it gates the failure mode rather than the
      // count. Both halves are kept: one says the reserve was not exceeded,
      // the other that no domain-shaped row demand was even attempted.
      { id: "power-droplet-pinned-rows", metric: "methods.*.info.pressureRequiredRows",
        operator: "at-most", expected: 1_024 },
      { id: "power-droplet-row-arena-fits", metric: "methods.*.info.pressureCapacityOverflow",
        operator: "equal", expected: false },
    ];
    // A gate that is itself O(domain) cannot gate a program about O(domain)
    // work, and two collectors here are exactly that. The spatial field and
    // its field-stats loop walk every cell (13.8M at 240 cubed against the
    // 196k this family was modelled on), and the global-fine sample census
    // behind `global-fine-publication` walks the *fine* lattice, which at
    // factor 4 is `(N*4)^3` — 16.7M samples at 64 cubed but 885M at 240. The
    // 240-cubed gate blew the 240 s isolated-runner ceiling on collection
    // alone while its twenty measured advances take 4.6 s.
    //
    // So above 2M cells the lane collects only what it gates on. What the
    // large members therefore do NOT check, stated rather than quietly
    // dropped: represented field statistics and the fine-publication sample
    // census. What they still check is every pinned counter below, structured
    // authority, the per-step generation audit and volume drift — and the
    // small members run the full collection on the identical solver
    // configuration, which is what makes the omission affordable.
    const denseCollection = edgeCells ** 3 <= 2_097_152;
    return suite(`power-droplet-${edgeCells}`,
      `One hundred liquid cells in a ${edgeCells}-cubed container: the domain-tax instrument`,
      { definitionId: `power-droplet-${edgeCells}` }, {
        default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
          methods: methods(["power-liquids"], { "power-liquids": powerDropletOverrides }), timeout_ms: 240_000,
          collect: { fieldStats: denseCollection ? "final" : "none", spatialField: denseCollection,
            stabilityEnvelope: true, structuredValidation: true,
            globalFineGeneration: denseCollection, powerGenerationAudit: { everySteps: 1, log: false } },
          diagnostics: denseCollection ? [powerDiagnostics[0], powerDiagnostics[2]] : [powerDiagnostics[0]],
          acceptance: [...powerAcceptance, ...pinned] }),
        // The measurement lane. Cold start pays two unavoidable O(domain) CPU
        // loops before the first advance — the footprint budget's triple loop
        // and the tall-cell column walk — which are seconds at 240 cubed, so
        // the 240 s isolated-runner ceiling is a real constraint here and not a
        // formality. Sample `--lane=droplet-<N> --steps=20` on the benchmark
        // first; the smoke lane is affordable only once the slope has fallen
        // far enough for 240 advances to fit.
        "runtime-240": lane({ id: "runtime-240", target_s: 0.96, exactSteps: 240,
          maxDt_s: 0.004, oracleSteps: 240,
          methods: methods(["power-liquids"], { "power-liquids": powerDropletOverrides }), timeout_ms: 240_000,
          collect: { fieldStats: denseCollection ? "checkpoints" : "none", checkpointEvery_s: 0.12,
            spatialField: denseCollection,
            stabilityEnvelope: true, structuredValidation: true, globalFineGeneration: denseCollection,
            powerGenerationAudit: { everySteps: 1, log: false } },
          diagnostics: denseCollection ? exhaustivePowerDiagnostics(1e-4)
            : exhaustivePowerDiagnostics(1e-4).filter((pack) => pack.id !== "global-fine-publication"),
          acceptance: [...powerAcceptance, ...pinned,
            { id: "power-droplet-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift",
              operator: "at-most", expected: 1e-4 }] }),
      });
  }),
  // The live-occupancy sweep: the droplet family's dual. Container fixed at 256
  // cubed, reservoir swept 100 -> 800 -> 6,400 cells in exact 8x steps at a
  // shared capacity.
  //
  // The pinned rules below are deliberately NOT the droplet family's shape. The
  // droplet pins are family-invariants — the same number at every N, because
  // the fluid is the same fluid — and pinning a live counter there is exactly
  // how that sweep detects domain-shaped work. Here the live counters are the
  // independent variable, so pinning them flat would gate away the signal. What
  // is pinned instead is the container (which must not move) and the two
  // overflow modes, because a fine band that overflows degrades its resident
  // count to the 0xFFFFFFFF sentinel and leaves the pressure solve executing
  // zero iterations while the run still reports PASS — a silently no-op solver
  // would read as a beautifully flat pass.
  //
  // The counters that discriminate the members — `globalFineActiveBricks`,
  // `structuredAirSupportRows`, `pressureRequiredRows` — are collected and
  // reported rather than pinned, because a floor invented before the first
  // capture is a number from intent rather than from a log. Pinning them is the
  // follow-up once the sweep has run.
  ...POWER_FILL_LIQUID_CELLS.map((liquidCells) => {
    const grid = [POWER_FILL_EDGE_CELLS, POWER_FILL_EDGE_CELLS, POWER_FILL_EDGE_CELLS];
    const cells = powerFillReservoirCells(liquidCells);
    const pinned: readonly SceneWebGPUAcceptanceRule[] = [
      { id: "expected-grid", metric: "methods.*.grid", operator: "equal", expected: grid },
      { id: "power-fill-row-arena-fits", metric: "methods.*.info.pressureCapacityOverflow",
        operator: "equal", expected: false },
      // Reads 0 whenever the arena holds, so it gates the overflow lower bound
      // rather than the count; the reserve itself is the ceiling.
      { id: "power-fill-rows-within-reserve", metric: "methods.*.info.pressureRequiredRows",
        operator: "at-most", expected: POWER_FILL_PRESSURE_ROW_CAPACITY },
      { id: "power-fill-fine-residency-within-reserve",
        metric: "methods.*.info.globalFineActiveBricks",
        operator: "at-most", expected: POWER_FILL_FINE_BRICK_CAPACITY },
    ];
    // Always false: the container is 256 cubed for every member, which is 16.8M
    // cells against the 2M ceiling the droplet family established for dense
    // collection. The spatial-field walk and the fine-publication census are
    // both O(domain) and blew the 240 s isolated-runner ceiling at 240 cubed on
    // collection alone. So these lanes collect only what they gate on: no
    // represented field statistics and no fine-publication sample census, but
    // every pinned counter, structured authority, the generation audit and
    // volume drift. The droplet family runs the full collection on the
    // identical solver configuration at 64 and 128 cubed, which is what makes
    // the omission affordable here.
    return suite(`power-fill-${POWER_FILL_EDGE_CELLS}-${liquidCells}`,
      `${liquidCells} liquid cells (${cells.x}x${cells.y}x${cells.z}) in a fixed ${POWER_FILL_EDGE_CELLS}-cubed container: the live-occupancy instrument`,
      { definitionId: `power-fill-${POWER_FILL_EDGE_CELLS}-${liquidCells}` }, {
        default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
          methods: methods(["power-liquids"], { "power-liquids": powerFillOverrides }), timeout_ms: 240_000,
          collect: { fieldStats: "none", spatialField: false,
            stabilityEnvelope: true, structuredValidation: true,
            globalFineGeneration: false, powerGenerationAudit: { everySteps: 1, log: false } },
          diagnostics: [powerDiagnostics[0]],
          acceptance: [...powerAcceptance, ...pinned] }),
        // The measurement lane. Droplet-family divergence lives past step 30, so
        // a correctness read on this discretization wants 80 advances rather
        // than 20; 240 is the same window the droplet lanes gate on and is what
        // makes a fill capture comparable to a droplet one. Cold start pays two
        // O(domain) CPU loops before the first advance at 256 cubed — the
        // footprint budget's triple loop and the tall-cell column walk — so the
        // 240 s isolated-runner ceiling is a real constraint, not a formality.
        // Sample `--lane=fill-<cells> --steps=80` on the benchmark first.
        "runtime-240": lane({ id: "runtime-240", target_s: 0.96, exactSteps: 240,
          maxDt_s: 0.004, oracleSteps: 240,
          methods: methods(["power-liquids"], { "power-liquids": powerFillOverrides }), timeout_ms: 240_000,
          collect: { fieldStats: "none", checkpointEvery_s: 0.12, spatialField: false,
            stabilityEnvelope: true, structuredValidation: true, globalFineGeneration: false,
            powerGenerationAudit: { everySteps: 1, log: false } },
          diagnostics: exhaustivePowerDiagnostics(1e-4)
            .filter((pack) => pack.id !== "global-fine-publication"),
          acceptance: [...powerAcceptance, ...pinned,
            { id: "power-fill-volume-drift", metric: "methods.*.stabilityEnvelope.maximumExactVolumeDrift",
              operator: "at-most", expected: 1e-4 }] }),
      });
  }),
  ...(["rigid-hydrostatic", "rigid-float", "rigid-sink"] as const).map((id) => {
    const target_s = id === "rigid-hydrostatic" ? 0.5 : id === "rigid-float" ? 2 : 1;
    const checkpointEvery_s = id === "rigid-hydrostatic" ? 0.05 : 0.1;
    return suite(id,
      id === "rigid-hydrostatic" ? "Static submerged sphere pressure-buoyancy oracle"
        : id === "rigid-float" ? "Dynamic half-density sphere settling oracle"
          : "Dense sphere bounded-entry and displacement oracle",
      { definitionId: id }, {
        default: lane({ target_s, exactSteps: Math.round(target_s / 0.004),
          maxDt_s: 0.004, oracleSteps: 2,
          methods: methods(["losasso"], { losasso: RIGID_COUPLING_ORACLE_METHOD_PROFILE.overrides }),
          collect: { fieldStats: "checkpoints", checkpointEvery_s, spatialField: true,
            stabilityEnvelope: true, raster: "checkpoints", globalFineGeneration: true,
            structuredValidation: true,
            evidenceCollectors: [{ id: "rigid-coupling", phase: "checkpoint", methods: ["losasso"],
              requires: ["rigid coupling", "compact velocity"], provides: ["rigid coupling"] }] },
          hooks: [{ id: "rigid-coupling-oracle", methods: ["losasso"],
            requires: ["rigid coupling", "checkpoint fields", "front/back raster"],
            parameters: { oracleVersion: 1 } }],
          maximumRepresentedVolumeDrift: 0.005,
          timeout_ms: 240_000 }),
      });
  }),
  ...(["ceiling-slab-drop", "corner-brick-drop", "midair-brick-drop", "midair-corner-drop"] as const).map((id) => suite(id,
    id === "ceiling-slab-drop" ? "Seeded brick flush under the lid free-fall oracle"
      : id === "corner-brick-drop" ? "Seeded brick in a lid/corner seam free-fall oracle"
        : id === "midair-brick-drop" ? "Seeded brick touching no boundary: the zero-contact free-fall control"
          : "Seeded brick on two vertical walls clear of the lid: seam adhesion without ceiling contact",
    { definitionId: id }, {
      default: lane({ target_s: 0.5, oracleSteps: 2,
        // These are standing product-physics oracles, not frozen trajectory
        // comparisons, so all four exercise the Losasso scene profile.
        methods: methods(["losasso"], { losasso: CEILING_DROP_METHOD_PROFILE.overrides }),
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.02, spatialField: true, raster: "checkpoints", globalFineGeneration: true, structuredValidation: true,
          evidenceCollectors: [{ id: "free-fall-contact-attribution", phase: "checkpoint", methods: ["losasso"],
            requires: ["compact velocity", "fine upper surface"], provides: ["free-fall contact attribution"] }] },
        diagnostics: [
          { id: "authoritative-water-raster", methods: ["losasso"], parameters: standardWaterRasterParameters }],
        hooks: [{ id: "free-fall-contact", methods: ["losasso"], requires: ["checkpoint centroid", "ceiling wet cells", "front/back raster", "free-fall contact attribution"],
          parameters: { gravityOracle: true, impactTime_s: 0.29, minimumRunTime_s: 0.3,
            evaluationStart_s: 0.02, preImpactMargin_s: 0.02,
            minimumMeasuredToAnalyticDropRatio: 0.6, maximumMeasuredToAnalyticDropRatio: 1.45,
            maximumDropHeadroom_cells: 0.5, minimumPreImpactCheckpoints: 3, releaseCheckTime_s: 0.2,
            maximumCenterProtrusion_cells: 0.05,
            maximumCeilingWetCellsAfterRelease: 0, maximumCeilingPixelsAfterRelease: 0,
            liquidThreshold: 0.5, ceilingWetLayers: 2,
            initialBrickSize_cells: 8, initialCentroidHalfBrickOffset_cells: 4,
            minimumMeaningfulColumnAmount_cells: 1,
            minimumExpectedSpeed_m_s: 1e-9,
            gravityAxisSelectionThreshold: 0.5,
            attributionDecimalPlaces: 4,
            includeCornerSeams: id === "corner-brick-drop" || id === "midair-corner-drop",
            // The mid-air corner brick measures 0.000 excess in every class;
            // the lid-attached one measures 0.237 from the first checkpoint on.
            seamEvaluationStart_s: 0.02, maximumSeamShortfallExcess: 0.05 } },
          { id: "water-raster-integrity", methods: ["losasso"], requires: ["global fine generation", "front/back raster"],
            parameters: standardWaterRasterParameters }] }),
      ...(id === "ceiling-slab-drop" ? {
        performance: lane({ id: "performance", target_s: 0.024, exactSteps: 6, maxDt_s: 0.004,
          oracleSteps: 6,
          methods: methods(["losasso"], { losasso: CEILING_DROP_METHOD_PROFILE.overrides }),
          collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true },
          diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
        "coarse-baseline-post-impact": lane({ id: "coarse-baseline-post-impact",
          description: "Factor-1 current-phase publication through the ceiling-drop impact",
          target_s: 0.4, exactSteps: 100, maxDt_s: 0.004, oracleSteps: 100,
          methods: methods(["losasso"], { losasso: {
            ...CEILING_DROP_METHOD_PROFILE.overrides, globalFineLevelSetFactor: "1",
          } }),
          collect: { fieldStats: "final", globalFineGeneration: true },
          timeout_ms: 240_000 }),
      } : {}),
    })),
  suite("power-hybrid-deep-ocean", "Deep-interior Bet-4 shipping work gate", { definitionId: "power-hybrid-deep-ocean" }, {
    default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1,
      methods: methods(["power-liquids"], { "power-liquids": {
        ...frozenPowerReferenceOverrides, maximumLeafSize: "32", interfaceRefinementBandCells: 1,
        globalFineLevelSetFactor: "1",
      } }),
      collect: { fieldStats: "none" },
      timeout_ms: 240_000 }),
  }),
  suite("ocean-seiche", "Long gravity wave crossing a wide deep tank", { definitionId: "ocean-seiche", variantId: "gpu-smoke" }, {
    default: lane({ target_s: 6, oracleSteps: 1,
      // 320x96x80 admits leaf 16, not the old Power-era leaf-32 ceiling.
      methods: methods(["losasso"], { losasso: OCEAN_SEICHE_METHOD_PROFILE.overrides }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.5, spatialField: true, stabilityEnvelope: true },
      hooks: [{ id: "ocean-wave-profile", requires: ["checkpoint fields"],
        parameters: { expectedGrid: [320, 96, 80], stationCount: 12, baselineHeightCells: 72, minimumCheckpoints: 3,
          minimumFarHalfDisturbanceWidthRatio: 3.6 } }] }),
    "global-fine-one-step": lane({ id: "global-fine-one-step", target_s: 0.004, oracleSteps: 1,
      methods: methods(["losasso"], { losasso: OCEAN_SEICHE_METHOD_PROFILE.overrides }), collect: { fieldStats: "none", stabilityEnvelope: true, checkpointEvery_s: 0.004 } }),
  }),
] satisfies readonly SceneWebGPUSmokeSuite<SceneWebGPUSmokeId>[];

export const sceneWebGPUSmokeSuites: Readonly<Record<SceneWebGPUSmokeId, SceneWebGPUSmokeSuite<SceneWebGPUSmokeId>>> =
  Object.freeze(Object.fromEntries(suiteList.map((entry) => [entry.sceneId, entry])) as Record<SceneWebGPUSmokeId, SceneWebGPUSmokeSuite<SceneWebGPUSmokeId>>);

export function isSceneWebGPUSmokeId(value: string): value is SceneWebGPUSmokeId {
  return Object.hasOwn(sceneWebGPUSmokeSuites, value);
}

export function getSceneWebGPUSmokeSuite(id: SceneWebGPUSmokeId): SceneWebGPUSmokeSuite<SceneWebGPUSmokeId> {
  return sceneWebGPUSmokeSuites[id];
}

export function getSceneWebGPUSmokeLane(sceneId: SceneWebGPUSmokeId, laneId?: string): SceneWebGPUSmokeLane {
  const suite = getSceneWebGPUSmokeSuite(sceneId);
  const resolvedLaneId = laneId ?? suite.defaultLane;
  const resolved = suite.lanes[resolvedLaneId];
  if (!resolved) throw new Error(`Unknown WebGPU smoke lane ${sceneId}/${resolvedLaneId}; expected ${Object.keys(suite.lanes).join(", ")}`);
  return resolved;
}
