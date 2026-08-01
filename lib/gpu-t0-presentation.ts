import type { WaterRenderDiagnostics } from "./webgpu-water-pipeline";

export type InitialRasterSurfaceState =
  | "pending"
  | "gpu-authoritative"
  | "compact-confirmed"
  | "crossing-confirmed"
  | "failed-closed";

export interface InitialRasterPresentationPrerequisites {
  readonly solverAttached: boolean;
  readonly initialSparseAuthorityReady: boolean;
  readonly globalFineAttached: boolean;
  readonly surfaceSourceAttached: boolean;
  readonly surfaceExtractionSubmitted: boolean;
  readonly presentationFenceCompleted: boolean;
  /** Normal diagnostics observes the bounded draw-argument readback. Safe mode does not. */
  readonly diagnosticsRequired: boolean;
  readonly diagnostics?: WaterRenderDiagnostics;
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
 * The paper path requires the global crossing latch. The retired coarse-only
 * presentation branch has no backing solver authority.
 */
export function initialRasterPresentationReadiness(
  input: InitialRasterPresentationPrerequisites,
): InitialRasterPresentationReadiness {
  if (!input.solverAttached) return { ready: false, state: "pending", label: "Waiting for warmed solver attachment" };
  if (!input.initialSparseAuthorityReady) return { ready: false, state: "pending", label: "Waiting for fenced sparse authority" };
  if (!input.globalFineAttached) return { ready: false, state: "pending", label: "Waiting for global-fine renderer source" };
  if (!input.surfaceSourceAttached) return { ready: false, state: "pending", label: "Waiting for global-fine raster source" };
  if (!input.surfaceExtractionSubmitted) return { ready: false, state: "pending", label: "Waiting for t=0 raster extraction submission" };
  if (!input.presentationFenceCompleted) return { ready: false, state: "pending", label: "Waiting for t=0 presentation fence" };

  const diagnostic = input.diagnostics;
  if (diagnostic) {
    const currentCrossing = diagnostic.globalFineAttached
      && diagnostic.globalFineCrossingPublished
      && diagnostic.surfaceGeometrySource === "global-fine-coarse";
    const currentCompact = !diagnostic.globalFineAttached
      && diagnostic.globalFineAuthorityLatch !== 0
      && diagnostic.surfaceGeometrySource === "compact-coarse";
    if (diagnostic.vertexCount > 0 && currentCrossing) {
      return {
        ready: true,
        state: "crossing-confirmed",
        label: "WebGPU t=0 ready · global-fine/coarse raster crossing confirmed",
      };
    }
    if (diagnostic.vertexCount > 0 && currentCompact) {
      return {
        ready: true,
        state: "compact-confirmed",
        label: "WebGPU t=0 ready · compact-coarse raster publication confirmed",
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
