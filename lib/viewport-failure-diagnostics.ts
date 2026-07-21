import { cameraBasis, dot, sub } from "./math";
import type { CameraState, SceneDescription, Vec3 } from "./model";
import { paperPipelineStages } from "./paper-pipeline-diagnostics";
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

function finiteTriple(value: readonly number[] | undefined): value is readonly [number, number, number] {
  return value?.length === 3 && value.every(Number.isFinite);
}

function gridPointToWorld(
  point: readonly [number, number, number],
  info: GPUEulerianInfo,
  scene: SceneDescription,
): Vec3 | undefined {
  if (info.nx <= 0 || info.ny <= 0 || info.nz <= 0) return undefined;
  const { width_m, height_m, depth_m } = scene.container;
  return {
    x: -0.5 * width_m + point[0] * width_m / info.nx,
    y: point[1] * height_m / info.ny,
    z: -0.5 * depth_m + point[2] * depth_m / info.nz,
  };
}

/** Best bounded spatial witness already present in GPUEulerianInfo. No new
 * readback is requested solely for this viewport marker. */
export function viewportFailureLocation(
  info: GPUEulerianInfo,
  scene: SceneDescription,
): Pick<ViewportFailureIndicator, "location_m" | "locationLabel"> {
  const phiFailure = info.globalFineFaceBandPhiFailure;
  if (phiFailure && finiteTriple(phiFailure.centroid)) {
    return {
      location_m: gridPointToWorld(phiFailure.centroid, info, scene),
      locationLabel: `first failed face ${phiFailure.globalFace.toLocaleString()}`,
    };
  }

  const acute = info.globalFineFaceBandAcuteGradingFailure;
  if (acute && acute.rowCell !== 0xffff_ffff && info.nx > 0 && info.ny > 0) {
    const plane = info.nx * info.ny;
    const z = Math.floor(acute.rowCell / plane);
    const remainder = acute.rowCell - z * plane;
    const y = Math.floor(remainder / info.nx);
    const x = remainder - y * info.nx;
    const half = Math.max(1, acute.rowSize) * 0.5;
    return {
      location_m: gridPointToWorld([x + half, y + half, z + half], info, scene),
      locationLabel: `first failed row ${acute.rowCell.toLocaleString()}`,
    };
  }

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

  if (finiteTriple(info.pagedPhiDifferentialMaxCell)) {
    const cell = info.pagedPhiDifferentialMaxCell;
    return {
      location_m: gridPointToWorld([cell[0] + 0.5, cell[1] + 0.5, cell[2] + 0.5], info, scene),
      locationLabel: "largest recorded φ mismatch",
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

/** Project a world-space diagnostic to the same 0.72-tangent camera used by
 * the raster water shaders. Off-screen/behind-camera witnesses stay in the
 * alert text but do not create a misleading marker. */
export function projectViewportFailure(
  position_m: Vec3,
  camera: CameraState,
  width: number,
  height: number,
): ViewportProjection {
  const basis = cameraBasis(camera);
  const relative = sub(position_m, basis.position);
  const depth = dot(relative, basis.forward);
  const aspect = Math.max(1, width) / Math.max(1, height);
  const ndcX = depth > 1e-6 ? dot(relative, basis.right) / (depth * aspect * 0.72) : Number.POSITIVE_INFINITY;
  const ndcY = depth > 1e-6 ? dot(relative, basis.up) / (depth * 0.72) : Number.POSITIVE_INFINITY;
  return {
    leftFraction: 0.5 * (ndcX + 1),
    topFraction: 0.5 * (1 - ndcY),
    visible: depth > 1e-6 && Math.abs(ndcX) <= 1 && Math.abs(ndcY) <= 1,
  };
}
