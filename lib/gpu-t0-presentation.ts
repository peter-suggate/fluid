import type { AdaptiveWaterRenderDiagnostics } from "./webgpu-water-pipeline";

export type InitialRasterSurfaceState =
  | "pending"
  | "gpu-authoritative"
  | "crossing-confirmed"
  | "failed-closed";

export interface InitialRasterPresentationPrerequisites {
  readonly solverAttached: boolean;
  readonly initialSparseAuthorityReady: boolean;
  readonly globalFineAttached: boolean;
  readonly adaptiveSurfaceAttached: boolean;
  readonly surfaceExtractionSubmitted: boolean;
  readonly presentationFenceCompleted: boolean;
  /** Normal diagnostics observes the bounded draw-argument readback. Safe mode does not. */
  readonly diagnosticsRequired: boolean;
  readonly diagnostics?: AdaptiveWaterRenderDiagnostics;
}

export interface InitialRasterPresentationReadiness {
  readonly ready: boolean;
  readonly state: InitialRasterSurfaceState;
  readonly label: string;
}

/**
 * CPU mirror of the paused t=0 presentation gate. Without an opt-in readback,
 * completion relies on the renderer's GPU-only draw-argument transaction: the
 * global crossing latch selects the fine/coarse mesh. Diagnostics mode proves
 * a non-empty current crossing; an adaptive fallback is useful for inspection
 * but is not the paper's t=0 presentation authority and therefore stays locked.
 */
export function initialRasterPresentationReadiness(
  input: InitialRasterPresentationPrerequisites,
): InitialRasterPresentationReadiness {
  if (!input.solverAttached) return { ready: false, state: "pending", label: "Waiting for warmed solver attachment" };
  if (!input.initialSparseAuthorityReady) return { ready: false, state: "pending", label: "Waiting for fenced sparse authority" };
  if (!input.globalFineAttached) return { ready: false, state: "pending", label: "Waiting for global-fine renderer source" };
  if (!input.adaptiveSurfaceAttached) return { ready: false, state: "pending", label: "Waiting for adaptive raster fallback source" };
  if (!input.surfaceExtractionSubmitted) return { ready: false, state: "pending", label: "Waiting for t=0 raster extraction submission" };
  if (!input.presentationFenceCompleted) return { ready: false, state: "pending", label: "Waiting for t=0 presentation fence" };

  const diagnostic = input.diagnostics;
  if (diagnostic) {
    const currentCrossing = diagnostic.globalFineCrossingPublished
      && diagnostic.surfaceGeometrySource === "global-fine-coarse";
    if (diagnostic.vertexCount > 0 && currentCrossing) {
      return {
        ready: true,
        state: "crossing-confirmed",
        label: "WebGPU t=0 ready · global-fine/coarse raster crossing confirmed",
      };
    }
    return {
      ready: false,
      state: "failed-closed",
      label: `t=0 raster publication failed closed: ${diagnostic.surfaceGeometrySource}, ${diagnostic.vertexCount} vertices`,
    };
  }

  if (input.diagnosticsRequired) {
    return { ready: false, state: "pending", label: "Waiting for bounded t=0 raster diagnostics" };
  }
  return {
    ready: true,
    state: "gpu-authoritative",
    label: "WebGPU t=0 ready · GPU raster publication fenced",
  };
}
