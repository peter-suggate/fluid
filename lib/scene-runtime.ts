import type { SceneDescription } from "./model";
import type { SvoRenderMode } from "./svo-render-mode";

export interface SceneRuntimeRequest {
  methodId?: string;
  renderMode?: SvoRenderMode;
}

export interface SceneRuntimeContent {
  /** Explicit subsystem ownership. Omission remains enabled for legacy scenes. */
  fluidEnabled: boolean;
  terrain: boolean;
  rigidBodyCount: number;
  dynamicRigidBodyCount: number;
}

export type SceneRuntimeCapability =
  | "static-world"
  | "rigid-dynamics"
  | "fluid-authority"
  | "fluid-rigid-coupling"
  | "water-presentation"
  | "sparse-voxel-presentation";

export type SceneReadinessRequirement = "required" | "not-required";

export interface SceneReadinessGate {
  state: SceneReadinessRequirement;
  requires: readonly SceneRuntimeCapability[];
}

export interface SceneRuntimeReadiness {
  /** Core render resources and the authored static world. */
  renderer: SceneReadinessGate;
  /** Every presentation source selected for this scene. */
  presentation: SceneReadinessGate;
  /** Fenced t=0 fluid authority, when a solver exists. */
  fluidAuthority: SceneReadinessGate;
  /** Permission to advance any time-dependent scene systems. */
  transport: SceneReadinessGate;
}

export interface SceneRuntimePlan {
  content: SceneRuntimeContent;
  capabilities: Readonly<Record<SceneRuntimeCapability, boolean>>;
  readiness: SceneRuntimeReadiness;
  /** Compatibility aliases for the renderer/controller migration. */
  staticWorld: true;
  fluidSolver: boolean;
  rigidCoupling: boolean;
  waterPresentation: boolean;
  sparseVoxelPresentation: boolean;
}

/** Derive runtime ownership from scene content/capabilities, not preset IDs. */
export function planSceneRuntime(
  scene: Pick<SceneDescription, "systems" | "terrain" | "rigidBodies">,
  request: SceneRuntimeRequest = {},
): SceneRuntimePlan {
  // Do not infer this from fillFraction. Legacy scenes that intentionally
  // start empty still own a solver, while an explicit false is authoritative.
  const fluidEnabled = scene.systems?.fluid !== false;
  const dynamicRigidBodyCount = scene.rigidBodies.filter(({ motion }) => motion !== "static").length;
  const rigidDynamics = dynamicRigidBodyCount > 0;
  const rigidCoupling = fluidEnabled && rigidDynamics;
  const sparseVoxelPresentation = request.renderMode === "svo" || request.methodId === "octree" || !fluidEnabled;
  const capabilities: Record<SceneRuntimeCapability, boolean> = {
    "static-world": true,
    "rigid-dynamics": rigidDynamics,
    "fluid-authority": fluidEnabled,
    "fluid-rigid-coupling": rigidCoupling,
    "water-presentation": fluidEnabled,
    "sparse-voxel-presentation": sparseVoxelPresentation,
  };
  const gate = (
    required: boolean,
    requires: readonly SceneRuntimeCapability[],
  ): SceneReadinessGate => ({ state: required ? "required" : "not-required", requires: required ? requires : [] });
  const presentationCapabilities: SceneRuntimeCapability[] = ["static-world"];
  if (capabilities["water-presentation"]) presentationCapabilities.push("water-presentation");
  if (capabilities["sparse-voxel-presentation"]) presentationCapabilities.push("sparse-voxel-presentation");
  const transportCapabilities: SceneRuntimeCapability[] = [];
  if (fluidEnabled) transportCapabilities.push("fluid-authority", "water-presentation");
  if (rigidDynamics) transportCapabilities.push("rigid-dynamics");
  if (rigidCoupling) transportCapabilities.push("fluid-rigid-coupling");

  return {
    content: {
      fluidEnabled,
      terrain: scene.terrain !== undefined,
      rigidBodyCount: scene.rigidBodies.length,
      dynamicRigidBodyCount,
    },
    capabilities,
    readiness: {
      renderer: gate(true, ["static-world"]),
      presentation: gate(true, presentationCapabilities),
      fluidAuthority: gate(fluidEnabled, ["fluid-authority"]),
      transport: gate(transportCapabilities.length > 0, transportCapabilities),
    },
    staticWorld: true,
    fluidSolver: fluidEnabled,
    rigidCoupling,
    waterPresentation: fluidEnabled,
    sparseVoxelPresentation,
  };
}
