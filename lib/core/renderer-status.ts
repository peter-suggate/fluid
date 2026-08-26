import type {
  SvoLightingVisibilityStatus,
  SvoSilhouetteRefinementStatus,
} from "../svo/svo-render-options";

export type SvoRendererFailureReason =
  | "missing-source"
  | "unsupported-terrain"
  | "unsupported-glass-cutout"
  | "missing-pbr-materials"
  | "missing-lighting-publications"
  | "pipeline-compile-failure"
  | "pipeline-compiling"
  | "frame-rejected";

export interface EffectiveRendererStatus {
  state: "active" | "not-required" | "pending" | "failed";
  failureReason?: SvoRendererFailureReason;
  /** Exact publication-contract cause; never used to select another renderer. */
  detail?: string;
  /** The requested refinement lifecycle is independent and never a fallback selector. */
  silhouetteRefinement?: SvoSilhouetteRefinementStatus;
  /** Requested versus effective shadows/AO/GI path. */
  lightingVisibility?: SvoLightingVisibilityStatus;
  /** Accepted structural leaf census from the live unified SVO publication. */
  terminalCounts?: Readonly<{ voxel: number; planarBoundary: number }>;
}

export interface EffectiveRendererConditions {
  /** False when the authoritative water renderer completely owns this scene. */
  required?: boolean;
  pipelineAvailable: boolean;
  /** A compile is in flight, so an absent pipeline is a rebuild rather than a failure. */
  pipelineCompiling?: boolean;
  /** Exact constructor/compile rejection retained for a fail-closed device. */
  pipelineFailure?: string;
  /** Exact requested-bundle transition while the owning pipeline remains attached. */
  pipelinePending?: string;
  sourceAvailable: boolean;
  terrainSupported: boolean;
  glassSupported?: boolean;
  materialsSupported?: boolean;
  lightingSupported?: boolean;
  svoEncoded: boolean;
  contractFailure?: string;
  silhouetteRefinement?: SvoSilhouetteRefinementStatus;
  lightingVisibility?: SvoLightingVisibilityStatus;
  terminalCounts?: Readonly<{ voxel: number; planarBoundary: number }>;
}

/** Resolve one frame's production renderer without changing simulation state. */
export function resolveEffectiveRendererStatus(
  conditions: EffectiveRendererConditions,
): EffectiveRendererStatus {
  const status = (renderer: Omit<EffectiveRendererStatus, "silhouetteRefinement" | "lightingVisibility" | "terminalCounts">): EffectiveRendererStatus => ({
    ...renderer,
    ...(conditions.silhouetteRefinement ? { silhouetteRefinement: conditions.silhouetteRefinement } : {}),
    ...(conditions.lightingVisibility ? { lightingVisibility: conditions.lightingVisibility } : {}),
    ...(conditions.terminalCounts ? { terminalCounts: conditions.terminalCounts } : {}),
  });
  if (conditions.required === false) return status({ state: "not-required" });
  // An absent pipeline means two very different things. Startup and a primary
  // traversal swap both retire it while the replacement compiles, and reporting
  // that as a compile failure tells the user their renderer broke when it is
  // merely busy.
  if (!conditions.pipelineAvailable) {
    return status(conditions.pipelineCompiling
      ? { state: "pending", failureReason: "pipeline-compiling" }
      : {
          state: "failed",
          failureReason: "pipeline-compile-failure",
          ...(conditions.pipelineFailure ? { detail: conditions.pipelineFailure } : {}),
        });
  }
  if (!conditions.terrainSupported) return status({ state: "failed", failureReason: "unsupported-terrain" });
  if (conditions.glassSupported === false) return status({ state: "failed", failureReason: "unsupported-glass-cutout" });
  if (conditions.materialsSupported === false) return status({ state: "failed", failureReason: "missing-pbr-materials" });
  if (conditions.lightingSupported === false) return status({ state: "failed", failureReason: "missing-lighting-publications" });
  if (!conditions.sourceAvailable) {
    const sourceMissing = !conditions.contractFailure
      || conditions.contractFailure === "live sparse source is not attached";
    if (sourceMissing) return status(conditions.contractFailure
      ? { state: "pending", failureReason: "missing-source", detail: conditions.contractFailure }
      : { state: "pending", failureReason: "missing-source" });
    return status({ state: "failed", failureReason: "frame-rejected", detail: conditions.contractFailure });
  }
  if (!conditions.svoEncoded && conditions.pipelineCompiling) return status({
    state: "pending", failureReason: "pipeline-compiling",
    ...(conditions.pipelinePending ? { detail: conditions.pipelinePending } : {}),
  });
  if (!conditions.svoEncoded && conditions.pipelineFailure) return status({
    state: "failed", failureReason: "pipeline-compile-failure", detail: conditions.pipelineFailure,
  });
  if (!conditions.svoEncoded) return status({
    state: "failed", failureReason: "frame-rejected",
    detail: conditions.contractFailure ?? "live SVO renderer declined the frame",
  });
  return status({ state: "active" });
}
