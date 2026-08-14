import type { SimulationMethod } from "./method-contract";
import type { CameraState, SceneDescription, Vec3 } from "./model";
import { planSceneRuntime } from "./scene-runtime";
import { projectToViewport } from "./webgpu-camera";
import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { WaterSurfacePresentationDiagnostics } from "./webgpu-water-pipeline";

export interface ViewportFailureIndicator {
  readonly id: string;
  readonly tone: "rejected" | "stale";
  readonly title: string;
  readonly stage: string;
  readonly detail: string;
  readonly location_m?: Vec3;
  readonly locationLabel?: string;
}

export interface ViewportProjection {
  readonly leftFraction: number;
  readonly topFraction: number;
  readonly visible: boolean;
}

/** Derive one high-signal viewport failure from queue-fenced solver and
 * presentation diagnostics. Pending startup states and mere numeric warnings
 * do not obscure the scene. */
export function viewportFailureIndicator(
  method: SimulationMethod,
  info: GPUEulerianInfo | null | undefined,
  water: WaterSurfacePresentationDiagnostics | null | undefined,
  scene: SceneDescription,
): ViewportFailureIndicator | undefined {
  // Renderer-only sparse scenes may still expose structural octree telemetry,
  // but it is not a fluid publication and must never be interpreted as one.
  // The runtime plan is the authoritative capability boundary shared by every
  // scene; this intentionally avoids preset- or topology-specific exceptions.
  if (!planSceneRuntime(scene).fluidSolver) return undefined;
  if (!info) return undefined;
  // Which stage rejected, whether the drawn mesh is the admitted generation,
  // and where the first invalid sample sits are all statements about one
  // method's transaction. Asking the running method is what keeps this file
  // from reading a second method's publications through the first one's
  // vocabulary — which is what a `gridKind === "octree"` gate did, for both
  // coarse-dynamics backends and for any adaptive method added later.
  return method.viewportFailure?.(info, water ?? undefined, scene);
}

/** Project a world-space diagnostic through the shared camera used by the
 * raster water shaders. Off-screen/behind-camera witnesses stay in the alert
 * text but do not create a misleading marker. */
export function projectViewportFailure(
  position_m: Vec3,
  camera: CameraState,
  width: number,
  height: number,
): ViewportProjection {
  const { leftFraction, topFraction, visible } = projectToViewport(position_m, camera, width, height);
  return { leftFraction, topFraction, visible };
}
