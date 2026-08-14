import type { SceneDescription } from "../core/model";
import type {
  SceneWebGPUEvidenceCollector,
  SceneWebGPUEvidenceCollectionPhase,
  SceneWebGPUEvidenceCollectorId,
  WebGPUSmokeMethodId,
} from "./scene-webgpu-smoke";

export interface SceneCheckpointEvidenceContext {
  readonly scene: SceneDescription;
  readonly method: WebGPUSmokeMethodId;
  readonly grid: readonly [number, number, number];
  readonly time_s: number;
  readonly volumeField: Float32Array;
  readonly velocityField?: Float32Array;
  readonly pressureField?: Float32Array;
  readonly pressureRhsField?: Float32Array;
  readonly pressureDiagonalField?: Float32Array;
  readonly pressureSection63DiagonalField?: Float32Array;
  readonly pressureSection63CaseIdField?: Uint32Array;
  readonly pressureInitialResidualField?: Float32Array;
  readonly pressureInitialPreconditionedField?: Float32Array;
  readonly pressureInitialPreconditionedImageField?: Float32Array;
  readonly pressurePreconditionerPreSmoothedField?: Float32Array;
  readonly pressurePreconditionerZeroSmoothedField?: Float32Array;
  readonly pressurePreconditionerFirstOperatorImageField?: Float32Array;
  readonly pressurePreconditionerFirstSmoothedField?: Float32Array;
  readonly pressurePreconditionerInnerResidualField?: Float32Array;
  readonly pressurePreconditionerInnerCorrectionField?: Float32Array;
  readonly pressurePreconditionerPostCorrectedField?: Float32Array;
  readonly topologyField?: Uint32Array;
  /** Authoritative fine-phi upper-interface height, in coarse-cell units, for
   * each coarse x/z column. NaN denotes a column without an upper crossing. */
  readonly fineUpperSurfaceField?: Float32Array;
  readonly rigidCouplingSnapshot?: Readonly<Record<string, unknown>>;
  readonly compactVelocityEvidence?: Readonly<Record<string, unknown>>;
}

export interface SceneEvidenceCollectorImplementation<Id extends SceneWebGPUEvidenceCollectorId = SceneWebGPUEvidenceCollectorId> {
  readonly id: Id;
  readonly phase: SceneWebGPUEvidenceCollectionPhase;
  readonly collect: (context: SceneCheckpointEvidenceContext) => unknown;
}

export type SceneEvidenceCollectorRegistry = Readonly<{
  [Id in SceneWebGPUEvidenceCollectorId]: SceneEvidenceCollectorImplementation<Id>;
}>;

export interface CollectedSceneCheckpointEvidence {
  readonly values: Readonly<Record<string, unknown>>;
  readonly available: readonly string[];
}

/** Dispatch scene-declared collectors without exposing their IDs to the executor. */
export function collectSceneEvidence(
  registry: SceneEvidenceCollectorRegistry,
  declarations: readonly SceneWebGPUEvidenceCollector[],
  phase: SceneWebGPUEvidenceCollectionPhase,
  context: SceneCheckpointEvidenceContext,
): CollectedSceneCheckpointEvidence {
  const values: Record<string, unknown> = {};
  const available = new Set<string>();
  for (const declaration of declarations) {
    if (declaration.phase !== phase) continue;
    if (declaration.methods && !declaration.methods.includes(context.method)) continue;
    const implementation = registry[declaration.id];
    if (!implementation) throw new Error(`missing scene evidence collector implementation: ${declaration.id}`);
    if (implementation.phase !== declaration.phase) {
      throw new Error(`scene evidence collector ${declaration.id} is registered for ${implementation.phase}, not ${declaration.phase}`);
    }
    const value = implementation.collect(context);
    if (value === undefined) throw new Error(`scene evidence collector ${declaration.id} returned no evidence`);
    values[declaration.id] = value;
    for (const capability of declaration.provides) available.add(capability);
  }
  return { values, available: [...available] };
}
