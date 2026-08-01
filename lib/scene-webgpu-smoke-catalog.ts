import { applyGardenPool, GARDEN_DAM_BRICK_SEED_M } from "./garden-scene";
import { cloneScene, defaultScene, type SceneDescription } from "./model";
import type { MethodParamValues } from "./methods";
import { createPaperScenario } from "./paper-scenarios";
import {
  CEILING_DROP_METHOD_PROFILE,
  COARSE_ONLY_POWER_DAM_METHOD_PROFILE,
  LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
  createBrickQuadDamBreakScene,
  createOceanSeicheScene,
  createTwinDamCollisionScene,
  getScenePreset,
} from "./scenes";
import {
  defineSceneWebGPUSmokeSuite,
  type SceneWebGPUAcceptanceRule,
  type SceneWebGPUCollectionPolicy,
  type SceneWebGPUDiagnosticHook,
  type SceneWebGPUDiagnosticPack,
  type SceneWebGPUSmokeLane,
  type SceneWebGPUSmokeMethod,
  type SceneWebGPUSmokeSuite,
  type WebGPUSmokeMethodId,
} from "./scene-webgpu-smoke";

export const sceneWebGPUSmokeIds = [
  "dam-break-ui",
  "settled-tank",
  "settled-tank-ui",
  "dam-break-boxes",
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
  "symmetric-expansion",
  "minimal-power-dam-break-32",
  "minimal-power-dam-break-64",
  "large-power-dam-break",
  "ceiling-slab-drop",
  "corner-brick-drop",
  "midair-brick-drop",
  "midair-corner-drop",
  "ocean-seiche",
] as const;

export type SceneWebGPUSmokeId = typeof sceneWebGPUSmokeIds[number];

const allMethodIds: readonly WebGPUSmokeMethodId[] = ["octree"];

function methods(
  ids: readonly WebGPUSmokeMethodId[] = allMethodIds,
  overrides: Partial<Record<WebGPUSmokeMethodId, Readonly<MethodParamValues>>> = {},
  quality: SceneWebGPUSmokeMethod["quality"] = "balanced",
): SceneWebGPUSmokeMethod[] {
  return ids.map((id) => ({ id, quality, overrides: { ...(overrides[id] ?? {}) } }));
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
  { id: "octree-authority", methods: ["octree"], parameters: {
    expectedGridKind: "octree",
    maximumNeighborRatio: 2,
    maximumTopologyReadbackBytes: 0,
    requirePowerDiagramReady: true,
    requirePowerDiagramAuthoritative: true,
    maximumDescriptorErrors: 0,
    maximumTopologyErrors: 0,
    pressureSolverNameIncludes: "persistent executor",
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
  cpuOracle?: boolean;
  collect?: Partial<SceneWebGPUCollectionPolicy>;
  diagnostics?: readonly SceneWebGPUDiagnosticPack[];
  acceptance?: readonly SceneWebGPUAcceptanceRule[];
  hooks?: readonly SceneWebGPUDiagnosticHook[];
  timeout_ms?: number;
  /** False for source-driven inflow scenes whose represented volume grows. */
  maximumRepresentedVolumeDrift?: number | false;
}

function backendAcceptance(configuredMethods: readonly SceneWebGPUSmokeMethod[]): SceneWebGPUAcceptanceRule[] {
  const ids = new Set(configuredMethods.map(({ id }) => id));
  const rules: SceneWebGPUAcceptanceRule[] = [];
  if (ids.has("octree")) rules.push(
    { id: "octree-grid-kind", metric: "methods.octree.info.gridKind", operator: "equal", expected: "octree" },
    { id: "octree-neighbor-ratio", metric: "methods.octree.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
    { id: "octree-topology-readback", metric: "methods.octree.info.quadtreeTopologyReadbackBytes", operator: "equal", expected: 0 },
    { id: "octree-power-ready", metric: "methods.octree.info.powerDiagramReady", operator: "equal", expected: true },
    { id: "octree-power-authoritative", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
    { id: "octree-power-descriptor-errors", metric: "methods.octree.octreePowerTopologyDiagnostics.descriptor.errorCount", operator: "equal", expected: 0 },
    { id: "octree-power-topology-errors", metric: "methods.octree.octreePowerTopologyDiagnostics.topology.invalidCount", operator: "equal", expected: 0 },
  );
  if (ids.has("quadtree-tall-cell")) rules.push(
    { id: "quadtree-grid-kind", metric: "methods.quadtree-tall-cell.info.gridKind", operator: "equal", expected: "quadtree-tall-cell" },
    { id: "quadtree-neighbor-ratio", metric: "methods.quadtree-tall-cell.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
    { id: "quadtree-has-leaves", metric: "methods.quadtree-tall-cell.info.quadtreeLeafCount", operator: "at-least", expected: 1 },
    { id: "quadtree-has-ghost-faces", metric: "methods.quadtree-tall-cell.info.quadtreeGhostFaceCount", operator: "at-least", expected: 1 },
    { id: "quadtree-fluid-scale", metric: "methods.quadtree-tall-cell.info.quadtreeMaximumFluidScale", operator: "at-most", expected: 100 },
  );
  return rules;
}

function diagnosticAcceptance(
  configuredMethods: readonly SceneWebGPUSmokeMethod[],
  diagnostics: readonly SceneWebGPUDiagnosticPack[],
): SceneWebGPUAcceptanceRule[] {
  const methodIds = new Set(configuredMethods.map(({ id }) => id));
  const pack = (id: SceneWebGPUDiagnosticPack["id"]) => diagnostics.find((entry) => entry.id === id);
  const number = (entry: SceneWebGPUDiagnosticPack, key: string): number => {
    const value = entry.parameters?.[key];
    if (typeof value !== "number" || !Number.isFinite(value)) {
      throw new Error(`${entry.id} requires finite numeric parameter ${key}`);
    }
    return value;
  };
  const rules: SceneWebGPUAcceptanceRule[] = [];
  const equilibrium = pack("equilibrium");
  if (equilibrium && methodIds.has("tall-cell")) rules.push(
    { id: "tall-equilibrium-volume", metric: "methods.tall-cell.info.representedVolumeDrift.abs", operator: "at-most",
      expected: number(equilibrium, "maximumTallCellExactVolumeDrift") },
    { id: "tall-equilibrium-components", metric: "methods.tall-cell.finalSummary.componentCount", operator: "at-most",
      expected: number(equilibrium, "maximumTallCellComponents") },
    { id: "tall-equilibrium-flags", metric: "methods.tall-cell.info.stabilityFlags.length", operator: "at-most",
      expected: number(equilibrium, "maximumTallCellStabilityFlags") },
  );
  if (equilibrium && methodIds.has("quadtree-tall-cell")) rules.push(
    { id: "quadtree-equilibrium-volume", metric: "methods.quadtree-tall-cell.info.representedVolumeDrift.abs", operator: "at-most",
      expected: number(equilibrium, "maximumQuadtreeExactVolumeDrift") },
    { id: "quadtree-equilibrium-components", metric: "methods.quadtree-tall-cell.finalSummary.componentCount", operator: "at-most",
      expected: number(equilibrium, "maximumQuadtreeComponents") },
    { id: "quadtree-equilibrium-speed", metric: "methods.quadtree-tall-cell.velocitySummary.liquidMaximum", operator: "at-most",
      expected: number(equilibrium, "maximumQuadtreeLiquidSpeed_m_s") },
  );
  const compression = pack("deep-compression");
  if (compression) for (const method of ["tall-cell", "quadtree-tall-cell"] as const) {
    if (methodIds.has(method)) rules.push({ id: `${method}-deep-compression`, metric: `methods.${method}.info.compressionRatio`,
      operator: "at-most", expected: number(compression, "maximumCompressionRatioExclusive") });
  }
  const restricted = pack("tall-cell-restricted");
  if (restricted && methodIds.has("tall-cell")) rules.push(
    { id: "tall-restricted-grid-kind", metric: "methods.tall-cell.info.gridKind", operator: "equal",
      expected: restricted.parameters?.expectedGridKind },
    { id: "tall-restricted-columns", metric: "methods.tall-cell.finalTallCellActivity.tallColumns", operator: "at-least",
      expected: number(restricted, "minimumTallColumns") },
    { id: "tall-restricted-ordinary-columns", metric: "methods.tall-cell.finalTallCellActivity.ordinaryColumns", operator: "at-most",
      expected: number(restricted, "maximumOrdinaryColumns") },
    { id: "tall-restricted-volume-gaps", metric: "methods.tall-cell.finalTallVolumeGaps.dryTallWithWetRegularAbove", operator: "at-most",
      expected: number(restricted, "maximumDryTallWithWetRegularAbove") },
  );
  const quadtreeParity = pack("quadtree-dam-parity");
  if (quadtreeParity && methodIds.has("quadtree-tall-cell")) rules.push(
    { id: "quadtree-parity-runtime", metric: "methods.quadtree-tall-cell.info.simulatedTime_s", operator: "at-least",
      expected: number(quadtreeParity, "minimumSimulatedTime_s") },
    { id: "quadtree-parity-rebuild-cadence", metric: "methods.quadtree-tall-cell.info.quadtreeRebuildCadenceSteps", operator: "equal",
      expected: number(quadtreeParity, "expectedRebuildCadenceSteps") },
    { id: "quadtree-parity-blocked-rebuilds", metric: "methods.quadtree-tall-cell.info.quadtreeRebuildBlockedFrames", operator: "at-most",
      expected: number(quadtreeParity, "maximumBlockedRebuildFrames") },
    { id: "quadtree-parity-peak-speed", metric: "methods.quadtree-tall-cell.stabilityEnvelope.peakLiquidSpeed_m_s", operator: "at-most",
      expected: number(quadtreeParity, "maximumPeakLiquidSpeed_m_s") },
    { id: "quadtree-parity-peak-cfl", metric: "methods.quadtree-tall-cell.stabilityEnvelope.peakComponentCfl", operator: "at-most",
      expected: number(quadtreeParity, "maximumPeakComponentCfl") },
    { id: "quadtree-parity-projection-energy", metric: "methods.quadtree-tall-cell.stabilityEnvelope.maximumProjectionEnergyRatio", operator: "at-most",
      expected: number(quadtreeParity, "maximumProjectionEnergyRatio") },
    { id: "quadtree-parity-pressure", metric: "methods.quadtree-tall-cell.stabilityEnvelope.maximumPressureRelativeResidual", operator: "at-most",
      expected: number(quadtreeParity, "maximumPressureRelativeResidual") },
    { id: "quadtree-parity-volume", metric: "methods.quadtree-tall-cell.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most",
      expected: number(quadtreeParity, "maximumExactVolumeDrift") },
    { id: "quadtree-parity-compression", metric: "methods.quadtree-tall-cell.info.compressionRatio", operator: "at-most",
      expected: number(quadtreeParity, "maximumCompressionRatio") },
    { id: "quadtree-parity-connectivity", metric: "methods.quadtree-tall-cell.stabilityEnvelope.minimumDominantComponentFraction", operator: "at-least",
      expected: number(quadtreeParity, "minimumDominantComponentFraction") },
    { id: "quadtree-parity-components", metric: "methods.quadtree-tall-cell.finalSummary.componentCount", operator: "at-most",
      expected: number(quadtreeParity, "maximumFinalComponents") },
    { id: "quadtree-parity-front", metric: "methods.quadtree-tall-cell.info.front_m", operator: "at-least",
      expected: number(quadtreeParity, "minimumFront_m") },
  );
  const fieldParity = pack("cross-method-field-parity");
  if (fieldParity && methodIds.has("tall-cell")) rules.push(
    { id: "tall-parity-grid-kind", metric: "methods.tall-cell.info.gridKind", operator: "equal",
      expected: fieldParity.parameters?.tallCellExpectedGridKind },
    { id: "tall-parity-columns", metric: "methods.tall-cell.finalTallCellActivity.tallColumns", operator: "at-least",
      expected: number(fieldParity, "tallCellMinimumTallColumns") },
    { id: "tall-parity-finite-velocity", metric: "methods.tall-cell.stabilityEnvelope.nonFiniteVelocityCount", operator: "at-most",
      expected: number(fieldParity, "tallCellMaximumNonFiniteVelocitySamples") },
    { id: "tall-parity-projection-energy", metric: "methods.tall-cell.stabilityEnvelope.maximumProjectionEnergyRatio", operator: "at-most",
      expected: number(fieldParity, "tallCellMaximumProjectionEnergyRatio") },
    { id: "tall-parity-cfl", metric: "methods.tall-cell.stabilityEnvelope.peakComponentCfl", operator: "at-most",
      expected: number(fieldParity, "tallCellMaximumPeakComponentCfl") },
    { id: "tall-parity-transient-volume", metric: "methods.tall-cell.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most",
      expected: number(fieldParity, "tallCellMaximumTransientExactVolumeDrift") },
    { id: "tall-parity-front", metric: "methods.tall-cell.info.front_m", operator: "at-least",
      expected: number(fieldParity, "tallCellMinimumFront_m"),
      when: [{ metric: "methods.tall-cell.info.simulatedTime_s", operator: "at-least",
        expected: number(fieldParity, "frontEvaluationStart_s") }] },
  );
  return rules;
}

function lane(options: LaneOptions): SceneWebGPUSmokeLane {
  const configuredMethods = options.methods ?? methods();
  const configuredDiagnostics = [...coreDiagnostics, ...(options.diagnostics ?? [])];
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
    oracle: { enabled: options.cpuOracle ?? true, matchedSteps: options.oracleSteps },
    collect: { ...defaultCollection, ...options.collect },
    diagnostics: configuredDiagnostics,
    acceptance: [
      ...coreAcceptance,
      ...backendAcceptance(configuredMethods),
      ...diagnosticAcceptance(configuredMethods, configuredDiagnostics),
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

const octreePowerOverrides = {
  maximumLeafSize: "32",
  interfaceRefinementBandCells: 3,
  globalFineLevelSetFactor: "4",
} as const;
const largePowerDamOverrides = {
  ...octreePowerOverrides,
  maximumLeafSize: "32",
  interfaceRefinementBandCells: 1,
  globalFineLevelSetMaximumBricks: LARGE_POWER_DAM_FINE_BRICK_CAPACITY,
} as const;

const powerDiagnostics: readonly SceneWebGPUDiagnosticPack[] = [
  { id: "structured-power", methods: ["octree"], parameters: {
    requireDirectStructuredVelocityAuthority: true,
    maximumNeighborRatio: 2,
    requireEveryStepGenerationAuditWhenCollected: true,
  } },
  { id: "exhaustive-power-generation", methods: ["octree"], parameters: {
    minimumStepsWithoutExactStop: 50,
    requireGenerationAuditEveryStep: true,
    requireStabilitySampleEveryStep: true,
    maximumInvalidVolumeSamples: 0,
    maximumNonFiniteVelocitySamples: 0,
    maximumExactVolumeDrift: 0.01,
    maximumPressureRelativeResidual: 1e-4,
  } },
  { id: "global-fine-publication", methods: ["octree"], parameters: {
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
  { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
  { id: "balanced-power-grid", metric: "methods.octree.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
  { id: "power-generation-every-step", metric: "methods.octree.powerGenerationAuditedSteps", operator: "equal", expected: { selector: "methods.octree.steps" } },
  { id: "power-volume-sample-every-step", metric: "methods.octree.stabilityEnvelope.sampledSteps", operator: "equal", expected: { selector: "methods.octree.steps" } },
  { id: "power-volume-samples-valid", metric: "methods.octree.stabilityEnvelope.invalidVolumeSampleCount", operator: "equal", expected: 0 },
  { id: "power-velocities-finite", metric: "methods.octree.stabilityEnvelope.nonFiniteVelocityCount", operator: "equal", expected: 0 },
];

function paperScene(id: "hose-tank" | "dam-break-boxes" | "sphere-jet"): SceneDescription {
  const scene = createPaperScenario(id);
  scene.environment = id === "hose-tank" ? "conservatory" : id === "dam-break-boxes" ? "concrete-gallery" : "night-lab";
  return scene;
}

function gardenSmokeScene(id: "garden-pond" | "garden-dam-break"): SceneDescription {
  const scene = applyGardenPool(cloneScene(defaultScene), id === "garden-dam-break" ? { fillFraction: 0.16 } : {});
  scene.environment = "garden";
  scene.sceneId = `smoke-${id}`;
  scene.rigidBodies = [];
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.fluid.initialCondition = id === "garden-pond" ? "tank-fill" : "dam-break";
  if (id === "garden-dam-break") scene.fluid.initialBrickSeeds_m = [{ ...GARDEN_DAM_BRICK_SEED_M }];
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = id === "garden-pond" ? 1 / 120 : 0.004;
  return scene;
}

function settledTankScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.environment = "default";
  scene.rigidBodies = [];
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.sceneId = "smoke-settled-tank";
  scene.fluid.initialCondition = "tank-fill";
  scene.container.fillFraction = 0.7;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 120;
  return scene;
}

function deepWaterScene(): SceneDescription {
  const scene = cloneScene(defaultScene);
  scene.environment = "research-station";
  scene.rigidBodies = [];
  scene.fluid.surfaceTension_N_m = 0;
  delete scene.fluid.inflow;
  scene.sceneId = "smoke-deep-water";
  scene.container.height_m = 20;
  scene.container.fillFraction = 0.8;
  scene.fluid.initialCondition = "tank-fill";
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 1 / 30;
  return scene;
}

function brickQuadScene(): SceneDescription {
  const scene = createBrickQuadDamBreakScene();
  scene.environment = "default";
  scene.sceneId = "smoke-brick-quad-dam-break";
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

function twinDamScene(): SceneDescription {
  const scene = createTwinDamCollisionScene();
  scene.environment = "default";
  scene.sceneId = "smoke-twin-dam-collision";
  scene.fluid.surfaceTension_N_m = 0;
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.004;
  return scene;
}

function oceanScene(): SceneDescription {
  const scene = createOceanSeicheScene();
  scene.environment = "default";
  scene.sceneId = "smoke-ocean-seiche";
  scene.numerics.fixedDt_s = scene.numerics.maxDt_s = 0.005;
  return scene;
}

function suite<const Id extends SceneWebGPUSmokeId>(
  sceneId: Id,
  description: string,
  createScene: () => SceneDescription,
  lanes: Record<string, SceneWebGPUSmokeLane>,
  defaultLane = "default",
): SceneWebGPUSmokeSuite<Id> {
  return defineSceneWebGPUSmokeSuite({ sceneId, description, createScene, defaultLane, lanes });
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

const damBreakVelocityParityParameters = {
  candidateMethod: "octree",
  referenceMethod: "tall-cell",
  limits: {
    maximumWeightedRelativeL2: 1,
    minimumCosineSimilarity: 0.5,
    minimumEnergyRatio: 0.25,
    maximumEnergyRatio: 4,
    minimumPeakRatio: 0.5,
    maximumPeakRatio: 2,
  },
  minimumWetIntersectionOverUnion: 0.6,
  maximumCentroidDistanceCells: 6,
  checkpointTimeTolerance_s: 0.01,
  minimumSharedVolumeWeight: 1e-4,
  normalizationEpsilon: 1e-30,
  wetThreshold: 0.5,
  referencePeakEpsilon: 1e-9,
  requireCompactPublicationValid: true,
  maximumCompactOverlapCells: 0,
  maximumCompactInvalidRows: 0,
} as const;

const equilibriumDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "equilibrium",
  parameters: {
    maximumTallCellExactVolumeDrift: 0.01,
    maximumTallCellComponents: 1,
    maximumTallCellStabilityFlags: 0,
    maximumQuadtreeExactVolumeDrift: 1e-3,
    maximumQuadtreeComponents: 1,
    maximumQuadtreeLiquidSpeed_m_s: 0.05,
  },
};

const deepCompressionDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "deep-compression",
  parameters: { maximumCompressionRatioExclusive: 0.5 },
};

const tallCellRestrictedDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "tall-cell-restricted",
  methods: ["tall-cell"],
  parameters: {
    expectedGridKind: "restricted-tall-cell",
    minimumTallColumns: 1,
    maximumOrdinaryColumns: 0,
    maximumDryTallWithWetRegularAbove: 0,
    maximumExactVolumeDrift: 0.01,
  },
};

const quadtreeDamParityDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "quadtree-dam-parity",
  methods: ["quadtree-tall-cell"],
  parameters: {
    method: "quadtree-tall-cell",
    referenceMethod: "uniform",
    minimumSimulatedTime_s: 0.2,
    simulatedTimeTolerance_s: 1e-9,
    maximumNeighborRatio: 2,
    minimumLeafCount: 1,
    minimumGhostFaceCount: 1,
    maximumPressureAbsoluteResidual: 1e-5,
    maximumFluidScale: 100,
    expectedRebuildCadenceSteps: 1,
    inlineRebuildCompletionFraction: 0.9,
    maximumBlockedRebuildFrames: 0,
    maximumWallToGpuTimeRatio: 2,
    requireStabilitySampleEveryStep: true,
    maximumNonFiniteVelocitySamples: 0,
    maximumPeakLiquidSpeed_m_s: 5,
    maximumPeakComponentCfl: 1,
    maximumProjectionEnergyRatio: 1.1,
    maximumPressureRelativeResidual: 1e-4,
    maximumExactVolumeDrift: 0.02,
    maximumCompressionRatio: 0.25,
    minimumDominantComponentFraction: 0.995,
    maximumFinalComponents: 10,
    minimumFront_m: -0.005,
    kineticEnergyEvaluationStart_s: 0.5,
    minimumPeakKineticEnergyProxy: 0.4,
    minimumUniformReferencePeakKineticEnergy: 1e-9,
    minimumUniformPeakKineticEnergyRatio: 0.8,
    minimumKineticEnergyRatioToReference: 0.8,
    minimumTallCellWetIntersectionOverUnion: 0.6,
    maximumTallCellCentroidDistanceCells: 6,
    checkpointComparisonStart_s: 1,
    checkpointComparisonStartTolerance_s: 1e-6,
    checkpointTimeTolerance_s: 0.01,
  },
};

const crossMethodFieldParityDiagnostic: SceneWebGPUDiagnosticPack = {
  id: "cross-method-field-parity",
  parameters: {
    tallCellMaximumRepresentedVolumeDrift: 0.02,
    tallCellExpectedGridKind: "restricted-tall-cell",
    tallCellMinimumTallColumns: 1,
    tallCellMaximumNonFiniteVelocitySamples: 0,
    tallCellMaximumProjectionEnergyRatio: 2,
    tallCellMaximumPeakComponentCfl: 32,
    tallCellMaximumTransientExactVolumeDrift: 0.15,
    frontEvaluationStart_s: 1.5,
    tallCellMinimumFront_m: 0.3,
    maximumDryTallWithWetRegularAbove: 0,
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

const suiteList = [
  suite("dam-break-ui", "Exact browser water-box dam-break preset", () => getScenePreset("water-box-dam-break").create(), {
    default: lane({ target_s: 0.2, oracleSteps: 2,
      collect: { fieldStats: "final", evidenceCollectors: [{ id: "collocated-velocity", phase: "terminal",
        requires: ["collocated velocity"], provides: ["collocated velocity", "volume field", "compact velocity"] }] },
      maximumRepresentedVolumeDrift: 0.02,
      diagnostics: [],
      hooks: [{ id: "dam-break-velocity-parity", requires: ["collocated velocity", "compact velocity", "volume field"],
        parameters: damBreakVelocityParityParameters }] }),
    "octree-runtime": lane({ id: "octree-runtime", target_s: 2, exactSteps: 250, maxDt_s: 0.008, oracleSteps: 250, cpuOracle: false,
      methods: methods(["octree"]), collect: { stabilityEnvelope: true, checkpointEvery_s: 0.1, fieldStats: "checkpoints" } }),
    "one-step": lane({ id: "one-step", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { globalFineLevelSetFactor: "4" } }),
      collect: { globalFineGeneration: true, fieldStats: "none" } }),
    "two-step": lane({ id: "two-step", target_s: 0.016, exactSteps: 2, maxDt_s: 0.008, oracleSteps: 2, cpuOracle: false,
      methods: methods(["octree"]), collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints", checkpointEvery_s: 0.008,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: powerDiagnostics, hooks: [{ id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }], timeout_ms: 240_000 }),
    runtime: lane({ id: "runtime", target_s: 1.52, exactSteps: 190, maxDt_s: 0.008, oracleSteps: 190, cpuOracle: false,
      methods: methods(["octree"]), collect: { stabilityEnvelope: true, spatialField: true, fieldStats: "checkpoints", checkpointEvery_s: 0.1,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: powerDiagnostics, hooks: [{ id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }], timeout_ms: 240_000 }),
    performance: lane({ id: "performance", description: "Release-like profiler lane", target_s: 0.496, exactSteps: 62, maxDt_s: 0.008, oracleSteps: 62, cpuOracle: false,
      methods: methods(["octree"]), collect: { fieldStats: "none", performanceProfile: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    throughput: lane({ id: "throughput", description: "Warm command-throughput lane", target_s: 0.496, exactSteps: 62, maxDt_s: 0.008, oracleSteps: 62, cpuOracle: false,
      methods: methods(["octree"]), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
  }),

  suite("settled-tank", "Hydrostatic preservation in a closed level pool", settledTankScene, {
    default: lane({ target_s: 0.1, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
    acceptance: lane({ id: "acceptance", target_s: 0.0667, oracleSteps: 2, cpuOracle: false,
      methods: methods(["octree"]), collect: { stabilityEnvelope: true, fieldStats: "final", powerGenerationAudit: { everySteps: 1, log: false }, boundaryThetaHistogram: true },
      diagnostics: [equilibriumDiagnostic, powerDiagnostics[0]], timeout_ms: 240_000 }),
  }),
  suite("settled-tank-ui", "Exact browser settled-tank preset with rigid bodies", () => getScenePreset("water-box-tank-fill").create(), {
    default: lane({ target_s: 0.2, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
    acceptance: lane({ id: "acceptance", target_s: 0.0667, oracleSteps: 2, cpuOracle: false, methods: methods(["octree"]),
      collect: { stabilityEnvelope: true, fieldStats: "final", powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: [equilibriumDiagnostic], timeout_ms: 240_000 }),
  }),
  suite("dam-break-boxes", "Three-dimensional dam break with immersed boxes", () => paperScene("dam-break-boxes"), {
    default: lane({ target_s: Math.max(createPaperScenario("dam-break-boxes").numerics.maxDt_s * 8, 0.05), oracleSteps: 2,
      diagnostics: [tallCellRestrictedDiagnostic] }),
  }),
  suite("hose-tank", "Fixed cylindrical inflow into a shallow receiving pool", () => paperScene("hose-tank"), {
    default: lane({ target_s: 0.5, oracleSteps: 2, maximumRepresentedVolumeDrift: false,
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance }),
    drift: lane({ id: "drift", target_s: 0.4611111111111111, exactSteps: 166, maxDt_s: 0.002777777777777778, oracleSteps: 166, cpuOracle: false,
      methods: methods(["octree"]), maximumRepresentedVolumeDrift: false,
      collect: { fieldStats: "final", evidenceCollectors: [{ id: "hose-jet-drift", phase: "terminal", methods: ["octree"],
        requires: ["compact velocity"], provides: ["structured velocity", "inflow geometry"] }] },
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance,
      hooks: [{ id: "hose-jet-drift", methods: ["octree"], requires: ["structured velocity", "inflow geometry"], parameters: {
        compareAgainstCoarseCellWidth: true, minimumSampledVelocityCells: 1, minimumOutletAxialSpeed_m_s: 0.05,
        maximumOutletSideSpeed_m_s: 0.1, maximumSideCentroidOffsetCoarseCells: 0.5,
        minimumSampledAirborneBins: 3, minimumAirborneAxialRetentionRatio: 0.9,
        maximumAdjacentAxialSpeedDropRatio: 0.12, maximumAdjacentMomentumFluxDropRatio: 0.25,
        minimumBallisticGravityVelocityRatio: 0.7, maximumBallisticGravityVelocityRatio: 3.1,
        maximumBallisticCenterlineRelativeError: 1.1,
      } }], timeout_ms: 240_000 }),
  }),
  suite("sphere-jet", "Directed inlet jet past a fixed immersed sphere", () => paperScene("sphere-jet"), {
    default: lane({ target_s: 0.5, oracleSteps: 2, maximumRepresentedVolumeDrift: false,
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance }),
  }),
  suite("deep-water", "Extreme vertical aspect ratio", deepWaterScene, {
    default: lane({ target_s: 0.1, oracleSteps: 1, diagnostics: [equilibriumDiagnostic] }),
  }),
  suite("garden-pond", "Hydrostatic rest in an organic terrain pool", () => gardenSmokeScene("garden-pond"), {
    default: lane({ target_s: 0.1, oracleSteps: 2, diagnostics: [equilibriumDiagnostic] }),
  }),
  suite("garden-hose", "Authored terrain pond with continuous hose inflow", () => getScenePreset("garden-hose").create(), {
    default: lane({ target_s: 0.5, oracleSteps: 2, cpuOracle: false, methods: methods(["octree"]),
      maximumRepresentedVolumeDrift: false, collect: { fieldStats: "final" },
      diagnostics: inflowDiagnostics, acceptance: inflowAcceptance, timeout_ms: 240_000 }),
  }),
  suite("garden-dam-break", "Single fluid brick released down terrain into the pond", () => gardenSmokeScene("garden-dam-break"), {
    default: lane({ target_s: 0.2, oracleSteps: 2 }),
    migration: lane({ id: "migration", target_s: 1, oracleSteps: 2, cpuOracle: false, methods: methods(["octree"]),
      collect: { fieldStats: "none", sparsePublication: true },
      hooks: [{ id: "garden-source-brick-migration", methods: ["octree"], requires: ["initial fluid brick stats", "final sparse publication"],
        parameters: { initialCoreBricks: 1, evaluateAfter_s: 1, minimumFinalCoreBricks: 2,
          sourceFluidVoxelsAtEnd: 0, sourceCoreResidencyAtEnd: false,
          requireResidentCountBelowCapacity: true } }] }),
  }),
  suite("brick-quad-dam-break", "Four-brick tank whose release crosses every brick boundary", brickQuadScene, {
    default: lane({ target_s: 1.5, oracleSteps: 2, cpuOracle: false,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.25, spatialField: true, sparsePublication: true },
      hooks: [{ id: "brick-quad-coverage", requires: ["checkpoint fields", "initial fluid brick stats", "final sparse publication"],
        parameters: { expectedBrickGrid: [2, 1, 2], minimumFirstCheckpointColumns: 2,
          minimumInitialResidentBricks: 1, minimumFinalResidentBricks: 2, minimumFinalCoreBricks: 2,
          liquidThreshold: 0.5, requireResidentCountBelowCapacity: true } }] }),
  }),
  suite("twin-dam-collision", "Opposed seeded reservoirs collapsing into an oblique collision", twinDamScene, {
    default: lane({ target_s: 1, oracleSteps: 2, cpuOracle: false, collect: { stabilityEnvelope: true, fieldStats: "checkpoints", checkpointEvery_s: 0.1 } }),
  }),
  suite("hydrostatic-power-two-level", "16-cubed settled leaf-32 power grid", () => getScenePreset("hydrostatic-power-two-level").create(), {
    default: lane({ target_s: 0.2, exactSteps: 50, maxDt_s: 0.004, oracleSteps: 50, cpuOracle: false,
      methods: methods(["octree"], { octree: octreePowerOverrides }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.2, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: exhaustivePowerDiagnostics(1e-4),
      acceptance: [...powerAcceptance,
        { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [16, 16, 16] },
        { id: "hydrostatic-power-volume-drift", metric: "methods.octree.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 1e-4 }] }),
  }),
  suite("hydrostatic-power-large-offset", "32x24x16 cell-cut leaf-32 power grid", () => getScenePreset("hydrostatic-power-large-offset").create(), {
    default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { ...octreePowerOverrides, interfaceRefinementBandCells: 4 } }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: true } },
      diagnostics: [powerDiagnostics[0], powerDiagnostics[2],
        { id: "authoritative-water-raster", methods: ["octree"], parameters: standardWaterRasterParameters }],
      acceptance: [{ id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 24, 16] }],
      hooks: [{ id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
        parameters: standardWaterRasterParameters }] }),
  }),
  suite("minimal-power-dam-break", "Minimal dynamic leaf-32 analytic dam", () => getScenePreset("minimal-power-dam-break").create(), {
    default: lane({ target_s: 2, exactSteps: 500, maxDt_s: 0.004, oracleSteps: 500, cpuOracle: false,
      methods: methods(["octree"], { octree: octreePowerOverrides }), timeout_ms: 240_000,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.1, energyEverySteps: 50, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: [...exhaustivePowerDiagnostics(0.01),
        { id: "authoritative-water-raster", methods: ["octree"], parameters: minimalDamRasterParameters }],
      acceptance: [...powerAcceptance,
        { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [16, 16, 16] },
        { id: "minimal-power-volume-drift", metric: "methods.octree.stabilityEnvelope.maximumExactVolumeDrift", operator: "at-most", expected: 0.01 },
        { id: "minimal-power-connectivity", metric: "methods.octree.stabilityEnvelope.minimumDominantComponentFraction", operator: "at-least", expected: 0.985 },
        { id: "minimal-power-variational-residual", metric: "methods.octree.stabilityEnvelope.maximumProjectedVariationalResidual", operator: "at-most", expected: 3.5e-6 },
        { id: "fine-transport-unavailable", metric: "methods.octree.finalGlobalFineGeneration.transportVelocityUnavailableFraction", operator: "at-most", expected: 0.08 },
        { id: "fine-transport-outside-band", metric: "methods.octree.finalGlobalFineGeneration.transportDepartureOutsideBandFraction", operator: "at-most", expected: 0.005 },
        { id: "fine-transport-committed", metric: "methods.octree.finalGlobalFineGeneration.transportCommitted", operator: "equal", expected: true },
        { id: "fine-transport-processed", metric: "methods.octree.finalGlobalFineGeneration.transportProcessed", operator: "at-least", expected: 1 },
        { id: "fine-transport-unavailable-accounted", metric: "methods.octree.finalGlobalFineGeneration.transportNonfiniteVelocity", operator: "equal",
          expected: { selector: "methods.octree.finalGlobalFineGeneration.transportVelocityUnavailable" } },
        { id: "fine-topology-not-rolled-back", metric: "methods.octree.finalGlobalFineGeneration.topologyRolledBack", operator: "equal", expected: false }],
      hooks: [{ id: "minimal-dam-motion", methods: ["octree"], requires: ["stability envelope", "checkpoint fields", "mechanical energy"],
          parameters: minimalDamMotionParameters },
        { id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
          parameters: minimalDamRasterParameters }] }),
    "two-step": lane({ id: "two-step", target_s: 0.008, exactSteps: 2, maxDt_s: 0.004, oracleSteps: 2, cpuOracle: false,
      methods: methods(["octree"], { octree: octreePowerOverrides }), timeout_ms: 240_000,
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true, stabilityEnvelope: true, structuredValidation: true,
        raster: "checkpoints", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
      diagnostics: [...exhaustivePowerDiagnostics(0.01),
        { id: "authoritative-water-raster", methods: ["octree"], parameters: minimalDamRasterParameters }], acceptance: powerAcceptance,
      hooks: [{ id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
        parameters: minimalDamRasterParameters }] }),
    performance: lane({ id: "performance", target_s: 0.248, exactSteps: 62, maxDt_s: 0.004, oracleSteps: 62, cpuOracle: false,
      methods: methods(["octree"], { octree: octreePowerOverrides }), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    throughput: lane({ id: "throughput", target_s: 0.248, exactSteps: 62, maxDt_s: 0.004, oracleSteps: 62, cpuOracle: false,
      methods: methods(["octree"], { octree: octreePowerOverrides }), collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true }, diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    motion: lane({ id: "motion", target_s: 0.5, oracleSteps: 1, cpuOracle: false, methods: methods(["octree"], { octree: octreePowerOverrides }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.016, spatialField: true, raster: "checkpoints", globalFineGeneration: true },
      hooks: [{ id: "minimal-dam-motion", methods: ["octree"], requires: ["stability envelope", "initial/final raster"],
        parameters: minimalDamMotionParameters }], timeout_ms: 240_000 }),
    "fine-factor-4": lane({ id: "fine-factor-4", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { ...octreePowerOverrides, globalFineLevelSetFactor: "4" } }),
      collect: { fieldStats: "final", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: powerDiagnostics, timeout_ms: 240_000 }),
    "fine-factor-8": lane({ id: "fine-factor-8", target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { ...octreePowerOverrides, globalFineLevelSetFactor: "8" } }),
      collect: { fieldStats: "final", globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } }, diagnostics: powerDiagnostics, timeout_ms: 240_000 }),
  }),
  suite("symmetric-expansion", "Central 2x1x2-brick liquid body with horizontal D4 symmetry",
    () => getScenePreset("symmetric-expansion").create(), {
      default: lane({ target_s: 1, exactSteps: 250, maxDt_s: 0.004, oracleSteps: 250, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }), timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true, raster: "initial-final", globalFineGeneration: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
          { id: "balanced-power-grid", metric: "methods.octree.info.quadtreeMaximumNeighborRatio", operator: "at-most", expected: 2 },
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["octree"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "four-wall contact"],
          parameters: {
            maximumVolumeAbsoluteError: 0,
            maximumVelocityAbsoluteError_m_s: 0,
            maximumPressureAbsoluteError: 0,
            maximumRhsAbsoluteError: 0,
            maximumDiagonalAbsoluteError: 0,
            requireExactTopology: true,
            requireAllWallsReached: true,
            minimumCheckpointCount: 250,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "raster-construction": lane({ id: "raster-construction", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }), timeout_ms: 240_000,
        collect: { fieldStats: "final", raster: "initial-final", globalFineGeneration: true },
        diagnostics: [],
      }),
      "one-step": lane({ id: "one-step", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["octree"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            maximumVolumeAbsoluteError: 0,
            maximumVelocityAbsoluteError_m_s: 0,
            maximumPressureAbsoluteError: 0,
            maximumRhsAbsoluteError: 0,
            maximumDiagonalAbsoluteError: 0,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 1,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "two-step": lane({ id: "two-step", target_s: 0.008, exactSteps: 2,
        maxDt_s: 0.004, oracleSteps: 2, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["octree"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            maximumVolumeAbsoluteError: 0,
            maximumVelocityAbsoluteError_m_s: 0,
            maximumPressureAbsoluteError: 0,
            maximumRhsAbsoluteError: 0,
            maximumDiagonalAbsoluteError: 0,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 2,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "three-step": lane({ id: "three-step", target_s: 0.012, exactSteps: 3,
        maxDt_s: 0.004, oracleSteps: 3, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["octree"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 pressure-operator symmetry", "D4 topology symmetry", "four-wall contact"],
          parameters: {
            maximumVolumeAbsoluteError: 0,
            maximumVelocityAbsoluteError_m_s: 0,
            maximumPressureAbsoluteError: 0,
            maximumRhsAbsoluteError: 0,
            maximumDiagonalAbsoluteError: 0,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 3,
            maximumWallContactStepSpread: 0,
          } }],
      }),
      "fine-factor-4": lane({ id: "fine-factor-4", target_s: 0.004, exactSteps: 1,
        maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: octreePowerOverrides }), timeout_ms: 240_000,
        collect: {
          fieldStats: "checkpoints", checkpointEvery_s: 0.004, spatialField: true,
          structuredValidation: true, raster: "initial-final", globalFineGeneration: true,
          evidenceCollectors: [{ id: "fluid-symmetry", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "compact pressure"],
            provides: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
              "D4 topology symmetry", "four-wall contact"] }],
        },
        diagnostics: [],
        acceptance: [
          { id: "structured-authority", metric: "methods.octree.info.powerDiagramAuthoritative", operator: "equal", expected: true },
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 16, 32] },
        ],
        hooks: [{ id: "fluid-symmetry", methods: ["octree"],
          requires: ["D4 volume symmetry", "D4 velocity symmetry", "D4 pressure symmetry",
            "D4 topology symmetry", "four-wall contact"],
          parameters: {
            maximumVolumeAbsoluteError: 0,
            maximumVelocityAbsoluteError_m_s: 0,
            maximumPressureAbsoluteError: 0,
            maximumRhsAbsoluteError: 0,
            maximumDiagonalAbsoluteError: 0,
            requireExactTopology: true,
            requireAllWallsReached: false,
            minimumCheckpointCount: 1,
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
        maxDt_s: 0.004, oracleSteps: 62, cpuOracle: false,
        methods: methods(["octree"], { octree: octreePowerOverrides }),
        collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true },
        diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
    }),
  suite("minimal-power-dam-break-32", "32-cubed coarse-only analytic mini dam",
    () => getScenePreset("minimal-power-dam-break-32").create(), {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 32, 32] }] }),
      "surface-regression": lane({ id: "surface-regression", target_s: 0.5, exactSteps: 125,
        maxDt_s: 0.004, oracleSteps: 125, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          raster: "checkpoints", checkpointEvery_s: 0.5, globalFineGeneration: true,
          powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [32, 32, 32] },
          { id: "coarse-surface-interface-area",
            metric: "methods.octree.finalGlobalFineGeneration.volumeInterfaceArea",
            operator: "at-most", expected: 1.1 },
          { id: "coarse-surface-active-cubes",
            metric: "methods.octree.finalGlobalFineRaster.activeCubeCount",
            operator: "at-most", expected: 5_200 },
          { id: "coarse-surface-resident-pages",
            metric: "methods.octree.finalGlobalFineGeneration.activePages",
            operator: "at-most", expected: 480 },
          { id: "coarse-surface-no-rejected-generations",
            metric: "methods.octree.finalGlobalFineGeneration.topologyRejectionCount",
            operator: "equal", expected: 0 },
          { id: "coarse-surface-transport-committed",
            metric: "methods.octree.finalGlobalFineGeneration.transportCommitted",
            operator: "equal", expected: true },
          { id: "coarse-surface-topology-not-rolled-back",
            metric: "methods.octree.finalGlobalFineGeneration.topologyRolledBack",
            operator: "equal", expected: false }] }),
    }),
  suite("minimal-power-dam-break-64", "64-cubed coarse-only analytic mini dam",
    () => getScenePreset("minimal-power-dam-break-64").create(), {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: COARSE_ONLY_POWER_DAM_METHOD_PROFILE.overrides }),
        timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [64, 64, 64] }] }),
    }),
  suite("large-power-dam-break", "20x-volume authored dam cold-start and one-step regression",
    () => getScenePreset("large-power-dam-break").create(), {
      default: lane({ target_s: 0.004, exactSteps: 1, maxDt_s: 0.004, oracleSteps: 1, cpuOracle: false,
        methods: methods(["octree"], { octree: largePowerDamOverrides }), timeout_ms: 240_000,
        collect: { fieldStats: "final", spatialField: true, stabilityEnvelope: true, structuredValidation: true,
          globalFineGeneration: true, powerGenerationAudit: { everySteps: 1, log: false } },
        diagnostics: [powerDiagnostics[0], powerDiagnostics[2]],
        acceptance: [...powerAcceptance,
          { id: "expected-grid", metric: "methods.octree.grid", operator: "equal", expected: [64, 20, 64] }] }),
    }),
  ...(["ceiling-slab-drop", "corner-brick-drop", "midair-brick-drop", "midair-corner-drop"] as const).map((id) => suite(id,
    id === "ceiling-slab-drop" ? "Seeded brick flush under the lid free-fall oracle"
      : id === "corner-brick-drop" ? "Seeded brick in a lid/corner seam free-fall oracle"
        : id === "midair-brick-drop" ? "Seeded brick touching no boundary: the zero-contact free-fall control"
          : "Seeded brick on two vertical walls clear of the lid: seam adhesion without ceiling contact",
    () => getScenePreset(id).create(), {
      default: lane({ target_s: 0.5, oracleSteps: 2, cpuOracle: false,
        methods: methods(["octree"], { octree: id === "ceiling-slab-drop"
          ? CEILING_DROP_METHOD_PROFILE.overrides : octreePowerOverrides }),
        collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.02, spatialField: true, raster: "checkpoints", globalFineGeneration: true, structuredValidation: true,
          evidenceCollectors: [{ id: "free-fall-contact-attribution", phase: "checkpoint", methods: ["octree"],
            requires: ["compact velocity", "fine upper surface"], provides: ["free-fall contact attribution"] }] },
        diagnostics: [powerDiagnostics[0],
          { id: "authoritative-water-raster", methods: ["octree"], parameters: standardWaterRasterParameters }],
        hooks: [{ id: "free-fall-contact", methods: ["octree"], requires: ["checkpoint centroid", "ceiling wet cells", "front/back raster", "free-fall contact attribution"],
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
          { id: "water-raster-integrity", methods: ["octree"], requires: ["global fine generation", "front/back raster"],
            parameters: standardWaterRasterParameters }] }),
      ...(id === "ceiling-slab-drop" ? {
        performance: lane({ id: "performance", target_s: 0.024, exactSteps: 6, maxDt_s: 0.004,
          oracleSteps: 6, cpuOracle: false,
          methods: methods(["octree"], { octree: CEILING_DROP_METHOD_PROFILE.overrides }),
          collect: { fieldStats: "none", performanceProfile: true, gpuCommandAudit: true },
          diagnostics: [{ id: "performance" }], timeout_ms: 240_000 }),
        "coarse-baseline-post-impact": lane({ id: "coarse-baseline-post-impact",
          description: "Factor-1 current-phase publication through the ceiling-drop impact",
          target_s: 0.4, exactSteps: 100, maxDt_s: 0.004, oracleSteps: 100, cpuOracle: false,
          methods: methods(["octree"], { octree: {
            ...CEILING_DROP_METHOD_PROFILE.overrides, globalFineLevelSetFactor: "1",
          } }),
          collect: { fieldStats: "final", globalFineGeneration: true },
          diagnostics: [powerDiagnostics[2]], timeout_ms: 240_000 }),
      } : {}),
    })),
  suite("ocean-seiche", "Long gravity wave crossing a wide deep tank", oceanScene, {
    default: lane({ target_s: 6, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { maximumLeafSize: "32" } }),
      collect: { fieldStats: "checkpoints", checkpointEvery_s: 0.5, spatialField: true, stabilityEnvelope: true },
      hooks: [{ id: "ocean-wave-profile", requires: ["checkpoint fields"],
        parameters: { expectedGrid: [320, 96, 80], stationCount: 12, baselineHeightCells: 72, minimumCheckpoints: 3,
          minimumFarHalfDisturbanceWidthRatio: 3.6 } }] }),
    "global-fine-one-step": lane({ id: "global-fine-one-step", target_s: 0.004, oracleSteps: 1, cpuOracle: false,
      methods: methods(["octree"], { octree: { maximumLeafSize: "32" } }), collect: { fieldStats: "none", stabilityEnvelope: true, checkpointEvery_s: 0.004 } }),
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
