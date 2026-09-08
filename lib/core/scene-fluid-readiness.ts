import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { ResourceLaneReadiness } from "./resource-readiness";

/** Enabling a solver replaces the dry scene owner, so finish its first upload first. */
export function enableWaterLockReason(
  gpuInfo: Pick<GPUEulerianInfo, "initialRasterSurfaceReady"> | null,
  svo: Pick<ResourceLaneReadiness, "state">,
): string | undefined {
  if (gpuInfo?.initialRasterSurfaceReady !== true || svo.state === "preparing") {
    return "Wait for the scene to finish loading before enabling water.";
  }
  return undefined;
}
