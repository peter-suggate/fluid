import type { AdaptiveWaterRenderDiagnostics } from "./webgpu-water-pipeline";

export type InitialRasterSurfaceState =
  | "pending"
  | "gpu-authoritative"
  | "crossing-confirmed"
  | "failed-closed";

export interface InitialRasterPresentationPrerequisites {
  readonly solverAttached: boolean;
  readonly initialSparseAuthorityReady: boolean;
  /** The paper surface path requires a global-fine source. Coarse-only octree
   * mode intentionally has no such allocation and publishes its adaptive
   * coarse surface directly instead. */
  readonly globalFineRequired?: boolean;
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

/** Only the power-octree method has a separately published sparse authority
 * and adaptive raster source that must cross a presentation fence before the
 * transport can unlock. Dense and regular tall-cell methods attach their
 * direct field textures atomically with the warmed solver. */
export function requiresFencedInitialRasterPresentation(methodId: string): boolean {
  return methodId === "octree";
}

/**
 * CPU mirror of the paused t=0 presentation gate. Without an opt-in readback,
 * completion relies on the renderer's GPU-only draw-argument transaction. The
 * paper path requires the global crossing latch; coarse-only mode deliberately
 * has no global-fine allocation and accepts a non-empty adaptive coarse mesh.
 */
export function initialRasterPresentationReadiness(
  input: InitialRasterPresentationPrerequisites,
): InitialRasterPresentationReadiness {
  if (!input.solverAttached) return { ready: false, state: "pending", label: "Waiting for warmed solver attachment" };
  if (!input.initialSparseAuthorityReady) return { ready: false, state: "pending", label: "Waiting for fenced sparse authority" };
  const globalFineRequired = input.globalFineRequired !== false;
  if (globalFineRequired && !input.globalFineAttached) return { ready: false, state: "pending", label: "Waiting for global-fine renderer source" };
  if (!input.adaptiveSurfaceAttached) return { ready: false, state: "pending", label: "Waiting for adaptive raster fallback source" };
  if (!input.surfaceExtractionSubmitted) return { ready: false, state: "pending", label: "Waiting for t=0 raster extraction submission" };
  if (!input.presentationFenceCompleted) return { ready: false, state: "pending", label: "Waiting for t=0 presentation fence" };

  const diagnostic = input.diagnostics;
  if (diagnostic) {
    const currentCrossing = globalFineRequired
      ? diagnostic.globalFineCrossingPublished && diagnostic.surfaceGeometrySource === "global-fine-coarse"
      : diagnostic.surfaceGeometrySource === "adaptive-octree";
    if (diagnostic.vertexCount > 0 && currentCrossing) {
      return {
        ready: true,
        state: "crossing-confirmed",
        label: globalFineRequired
          ? "WebGPU t=0 ready · global-fine/coarse raster crossing confirmed"
          : "WebGPU t=0 ready · coarse-octree raster crossing confirmed",
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
    label: globalFineRequired
      ? "WebGPU t=0 ready · GPU raster publication fenced"
      : "WebGPU t=0 ready · coarse-octree raster publication fenced",
  };
}
