import type { CameraState, SceneDescription, Vec3 } from "./model";
import { paperPipelineStages } from "./paper-pipeline-diagnostics";
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

/** Best bounded spatial witness already present in GPUEulerianInfo. No new
 * readback is requested solely for this viewport marker. */
export function viewportFailureLocation(
  info: GPUEulerianInfo,
  scene: SceneDescription,
): Pick<ViewportFailureIndicator, "location_m" | "locationLabel"> {
  const transport = info.globalFineTransportFirstInvalidVelocityPosition_m;
  if (transport && [transport.x, transport.y, transport.z].every(Number.isFinite)) {
    return {
      location_m: {
        x: transport.x - 0.5 * scene.container.width_m,
        y: transport.y,
        z: transport.z - 0.5 * scene.container.depth_m,
      },
      locationLabel: `first invalid velocity sample ${info.globalFineTransportFirstInvalidVelocityLocalIndex?.toLocaleString() ?? "?"}`,
    };
  }

  return {};
}

/** Derive one high-signal viewport failure from queue-fenced solver and
 * presentation diagnostics. Pending startup states and mere numeric warnings
 * do not obscure the scene. */
export function viewportFailureIndicator(
  info: GPUEulerianInfo | null | undefined,
  water: WaterSurfacePresentationDiagnostics | null | undefined,
  scene: SceneDescription,
): ViewportFailureIndicator | undefined {
  // Renderer-only sparse scenes may still expose structural octree telemetry,
  // but it is not a fluid publication and must never be interpreted as one.
  // The runtime plan is the authoritative capability boundary shared by every
  // scene; this intentionally avoids preset- or topology-specific exceptions.
  if (!planSceneRuntime(scene).fluidSolver) return undefined;
  if (!info || info.gridKind !== "octree") return undefined;
  const stages = paperPipelineStages(info, water);
  const rejected = stages.find((stage) => stage.tone === "rejected" && stage.id !== "raster");
  const raster = stages.find((stage) => stage.id === "raster");
  const retained = water?.surfaceGeometrySource === "retained-previous";
  const empty = water?.surfaceGeometrySource === "empty";
  const staleRaster = raster?.tone === "stale";
  const location = viewportFailureLocation(info, scene);

  if (rejected) {
    const presentation = retained
      ? ` Renderer retained mesh generation ${water?.meshPublicationGeneration ?? "?"}; live generation ${info.globalFineGeneration ?? "?"} was not admitted.`
      : info.globalFineRolledBack
        ? " The rejected generation is not visible; presentation remains on the last admitted mesh."
        : empty
          ? " No water surface from this generation is being drawn."
          : " The rejected generation is not admitted to the visible water surface.";
    return {
      id: `pipeline-${rejected.id}`,
      tone: "rejected",
      title: "WATER UPDATE REJECTED",
      stage: `${rejected.section} · ${rejected.label}`,
      detail: `${rejected.detail}${presentation}`,
      ...location,
    };
  }

  if (retained || staleRaster || empty) {
    return {
      id: retained ? "raster-retained" : empty ? "raster-empty" : "raster-stale",
      tone: retained || empty ? "rejected" : "stale",
      title: empty ? "WATER SURFACE MISSING" : "WATER MESH STALE",
      stage: "render · Fine/coarse raster surface",
      detail: raster?.detail ?? "The displayed water surface is not the current admitted generation.",
      ...location,
    };
  }
  return undefined;
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
