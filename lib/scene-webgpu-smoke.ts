import type { SceneDescription } from "./model";
import type { MethodParamValues } from "./methods";
import type { GPUQuality } from "./tall-cell-grid";
import type { DeclarativeDiagnosticRule } from "./scene-diagnostics";

/**
 * Scene-owned input to the generic WebGPU smoke executor.
 *
 * This file deliberately contains data contracts only.  In particular it
 * does not import the smoke runner, WebGPU implementations, or Node APIs, so
 * authored scene suites remain inspectable without initializing Dawn.
 */

export type WebGPUSmokeMethodId = "tall-cell" | "quadtree-tall-cell" | "octree" | "uniform";

export interface SceneWebGPUSmokeMethod {
  readonly id: WebGPUSmokeMethodId;
  readonly quality: GPUQuality;
  readonly overrides: Readonly<MethodParamValues>;
}

export interface SceneWebGPUSmokeStop {
  readonly simulatedTime_s: number;
  /** When present, accepted, encoded, submitted, and completed work must all match. */
  readonly exactSteps?: number;
  /** Optional lane cap, applied without mutating the authored scene document. */
  readonly maxDt_s?: number;
  /** Optional authored repeating outer-step cadence used by cadence probes. */
  readonly dtPattern_s?: readonly number[];
}

export interface SceneWebGPUOraclePolicy {
  readonly enabled: boolean;
  readonly matchedSteps: number;
  readonly maximumCells?: number;
}

export type SceneWebGPURasterPolicy = "none" | "initial-final" | "checkpoints";

export type SceneWebGPUEvidenceCollectorId =
  | "free-fall-contact-attribution"
  | "hose-jet-drift"
  | "collocated-velocity";

export type SceneWebGPUEvidenceCollectionPhase = "checkpoint" | "terminal";
export type SceneWebGPUEvidenceSource = "compact velocity" | "collocated velocity";

/** Scene-declared adapter for evidence that cannot be collected generically. */
export interface SceneWebGPUEvidenceCollector {
  readonly id: SceneWebGPUEvidenceCollectorId;
  readonly phase: SceneWebGPUEvidenceCollectionPhase;
  readonly methods?: readonly WebGPUSmokeMethodId[];
  readonly requires?: readonly SceneWebGPUEvidenceSource[];
  /** Capability labels made available to diagnostic hooks after collection. */
  readonly provides: readonly string[];
}

/** Collection is separate from acceptance so expensive evidence is opt-in. */
export interface SceneWebGPUCollectionPolicy {
  readonly evidenceCollectors: readonly SceneWebGPUEvidenceCollector[];
  readonly fieldStats: "none" | "final" | "checkpoints";
  readonly checkpointEvery_s?: number;
  readonly stabilityEnvelope: boolean;
  readonly spatialField: boolean;
  readonly sparsePublication: boolean;
  readonly raster: SceneWebGPURasterPolicy;
  readonly globalFineGeneration: boolean;
  readonly powerGenerationAudit: false | {
    readonly everySteps: number;
    readonly log: boolean;
  };
  readonly boundaryThetaHistogram: boolean;
  readonly structuredValidation: boolean;
  readonly performanceProfile: boolean;
  readonly gpuCommandAudit: boolean;
  readonly energyEverySteps?: number;
}

export type SceneWebGPUDiagnosticPackId =
  | "core-webgpu-health"
  | "volume-and-topology"
  | "equilibrium"
  | "deep-compression"
  | "inflow-activity"
  | "tall-cell-restricted"
  | "quadtree-dam-parity"
  | "octree-authority"
  | "structured-power"
  | "exhaustive-power-generation"
  | "global-fine-publication"
  | "authoritative-water-raster"
  | "cross-method-field-parity"
  | "settling"
  | "performance";

export interface SceneWebGPUDiagnosticPack {
  readonly id: SceneWebGPUDiagnosticPackId;
  /** Restrict a pack to the methods that can publish its evidence. */
  readonly methods?: readonly WebGPUSmokeMethodId[];
  readonly parameters?: Readonly<Record<string, unknown>>;
}

/** A selector-based rule consumed by the generic diagnostics evaluator. */
export interface SceneWebGPUAcceptanceRule extends DeclarativeDiagnosticRule {
  readonly methods?: readonly WebGPUSmokeMethodId[];
}

export type SceneWebGPUHookId =
  | "brick-quad-coverage"
  | "garden-source-brick-migration"
  | "hose-jet-drift"
  | "ocean-wave-profile"
  | "minimal-dam-motion"
  | "free-fall-contact"
  | "dam-break-velocity-parity"
  | "dam-break-perturbed-cadence"
  | "water-raster-integrity";

/**
 * Hooks identify pure, scene-specific post-processing.  They declare all
 * required evidence here; hook implementations must not inspect solver state
 * or environment variables behind the executor's back.
 */
export interface SceneWebGPUDiagnosticHook {
  readonly id: SceneWebGPUHookId;
  readonly methods?: readonly WebGPUSmokeMethodId[];
  readonly requires: readonly string[];
  readonly parameters?: Readonly<Record<string, unknown>>;
}

export interface SceneWebGPUSmokeLane {
  readonly id: string;
  readonly description: string;
  readonly methods: readonly SceneWebGPUSmokeMethod[];
  readonly stop: SceneWebGPUSmokeStop;
  readonly oracle: SceneWebGPUOraclePolicy;
  readonly collect: SceneWebGPUCollectionPolicy;
  readonly diagnostics: readonly SceneWebGPUDiagnosticPack[];
  readonly acceptance: readonly SceneWebGPUAcceptanceRule[];
  readonly hooks: readonly SceneWebGPUDiagnosticHook[];
  /** Operational hint only; never changes scientific meaning. */
  readonly timeout_ms?: number;
}

export interface SceneWebGPUSmokeSuite<Id extends string = string> {
  readonly sceneId: Id;
  readonly description: string;
  readonly createScene: () => SceneDescription;
  readonly defaultLane: string;
  readonly lanes: Readonly<Record<string, SceneWebGPUSmokeLane>>;
}

/** Recursively freeze declarative suite data while leaving scene factories callable. */
function deepFreeze<T>(value: T): T {
  if (value === null || (typeof value !== "object" && typeof value !== "function") || Object.isFrozen(value)) {
    return value;
  }
  Object.freeze(value);
  for (const child of Object.values(value as Record<string, unknown>)) deepFreeze(child);
  return value;
}

export function defineSceneWebGPUSmokeSuite<const Id extends string>(
  suite: SceneWebGPUSmokeSuite<Id>,
): SceneWebGPUSmokeSuite<Id> {
  if (!(suite.defaultLane in suite.lanes)) {
    throw new Error(`${suite.sceneId} default WebGPU smoke lane ${suite.defaultLane} does not exist`);
  }
  for (const [laneKey, lane] of Object.entries(suite.lanes)) {
    if (lane.id !== laneKey) throw new Error(`${suite.sceneId} lane key ${laneKey} differs from lane id ${lane.id}`);
    if (lane.methods.length === 0) throw new Error(`${suite.sceneId}/${lane.id} has no methods`);
    if (!(lane.stop.simulatedTime_s > 0)) throw new Error(`${suite.sceneId}/${lane.id} has an invalid stop time`);
    if (lane.stop.exactSteps !== undefined && (!Number.isInteger(lane.stop.exactSteps) || lane.stop.exactSteps < 1)) {
      throw new Error(`${suite.sceneId}/${lane.id} has an invalid exact step count`);
    }
    if (!Number.isInteger(lane.oracle.matchedSteps) || lane.oracle.matchedSteps < 1) {
      throw new Error(`${suite.sceneId}/${lane.id} has an invalid oracle step count`);
    }
    const methodIds = new Set(lane.methods.map(({ id }) => id));
    const collectorIds = new Set<SceneWebGPUEvidenceCollectorId>();
    for (const collector of lane.collect.evidenceCollectors) {
      if (collectorIds.has(collector.id)) {
        throw new Error(`${suite.sceneId}/${lane.id} declares duplicate evidence collector ${collector.id}`);
      }
      collectorIds.add(collector.id);
      if (collector.provides.length === 0 || collector.provides.some((label) => label.length === 0)) {
        throw new Error(`${suite.sceneId}/${lane.id} evidence collector ${collector.id} must declare capabilities`);
      }
      if (collector.methods?.some((id) => !methodIds.has(id))) {
        throw new Error(`${suite.sceneId}/${lane.id} evidence collector ${collector.id} targets an undeclared method`);
      }
    }
  }
  return deepFreeze(suite);
}
