import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { EffectiveRendererStatus, GPUStatus } from "./webgpu-renderer";

/**
 * Product capabilities that can become usable independently.
 *
 * This is deliberately not a list of constructors or shader families. UI and
 * controller code care about the capability a resource unlocks, while the
 * resource owner remains free to split or combine its implementation work.
 */
export type ResourceLaneId = "platform" | "fluid" | "svo" | "optional";
export type ResourceLaneState = "idle" | "preparing" | "ready" | "degraded" | "failed" | "unavailable";
export type RuntimeResourceCapability =
  | "renderer"
  | "live-scene"
  | "fluid-authority"
  | "water-presentation"
  | "sparse-voxel-presentation"
  | "optional-tooling";

/**
 * Colocated declaration exported by the resource owner and composed in a
 * static catalog. This mirrors the visualization plugin architecture: copy,
 * gating semantics and capability claims stay beside the code that makes the
 * resource, while orchestration only understands this protocol.
 */
export interface ResourcePluginDefinition {
  readonly id: string;
  readonly lane: ResourceLaneId;
  readonly label: string;
  readonly provides: readonly RuntimeResourceCapability[];
  /** What loses an action when no usable generation exists. */
  readonly blocks: "viewport" | "transport" | "nothing";
  readonly phaseCopy?: Readonly<Record<string, string>>;
}

export interface ResourceActivity {
  id: string;
  pluginId: string;
  lane: ResourceLaneId;
  label: string;
  phase: string;
  completed: number;
  total: number;
  startedAt_ms: number;
  operation?: string;
  /** A usable generation remains attached while this work runs. */
  retainingPrevious: boolean;
}

export interface ResourceLaneReadiness {
  state: ResourceLaneState;
  label: string;
  /** Read and presentation may continue even if replacement work is active. */
  usable: boolean;
  activity?: ResourceActivity;
}

export interface ResourceReadinessSnapshot {
  platform: ResourceLaneReadiness;
  fluid: ResourceLaneReadiness;
  svo: ResourceLaneReadiness;
  optional: ResourceLaneReadiness;
  /** One independently versioned entry per resource-owner plugin. */
  plugins: Readonly<Record<string, ResourcePluginReadiness>>;
  /** The most recently updated lane; presentation only, never a global gate. */
  activeLane?: ResourceLaneId;
}

export interface ResourcePluginReadiness extends ResourceLaneReadiness {
  plugin: ResourcePluginDefinition;
}

export function initialResourceReadiness(): ResourceReadinessSnapshot {
  return {
    platform: { state: "preparing", label: "WebGPU has not started", usable: false },
    fluid: { state: "idle", label: "Fluid resources have not been requested", usable: false },
    svo: { state: "idle", label: "Sparse presentation has not been requested", usable: false },
    optional: { state: "idle", label: "Optional tools load on first use", usable: true },
    plugins: {},
    activeLane: "platform",
  };
}

const sparseStatus = (status: GPUStatus) => /\b(sparse|svo|garden|dry-scene)\b/i.test(status.label);
const platformStatus = (status: Extract<GPUStatus, { state: "initializing" }>) =>
  status.phase === "renderer"
  || status.phase === "water-renderer"
  || /adapter|gpu device|webgpu lease|initializing webgpu|presentation upscale|raster water/i.test(status.label);

function activityLane(status: Extract<GPUStatus, { state: "initializing" }>): ResourceLaneId {
  if (status.resource) return status.resource.lane;
  if (platformStatus(status)) return "platform";
  if (sparseStatus(status)) return "svo";
  return "fluid";
}

const compatibilityPlugin = (status: GPUStatus, lane: ResourceLaneId): ResourcePluginDefinition => ({
  id: `legacy.${lane}`,
  lane,
  label: lane === "platform" ? "WebGPU platform" : lane === "fluid" ? "Fluid runtime" : "Sparse presentation",
  provides: lane === "platform" ? ["renderer"]
    : lane === "fluid" ? ["fluid-authority", "water-presentation"]
    : lane === "svo" ? ["sparse-voxel-presentation"] : ["optional-tooling"],
  blocks: lane === "platform" ? "viewport" : lane === "fluid" ? "transport" : "nothing",
});

function pluginForStatus(status: GPUStatus, lane: ResourceLaneId): ResourcePluginDefinition {
  return status.resource ?? compatibilityPlugin(status, lane);
}

function updatePlugin(
  snapshot: ResourceReadinessSnapshot,
  plugin: ResourcePluginDefinition,
  value: ResourceLaneReadiness,
): ResourceReadinessSnapshot {
  const plugins = { ...snapshot.plugins, [plugin.id]: { ...value, plugin } };
  const states = Object.values(plugins).filter((state) => state.plugin.lane === plugin.lane);
  const preparing = states.filter((state) => state.state === "preparing")
    .sort((left, right) => (right.activity?.startedAt_ms ?? 0) - (left.activity?.startedAt_ms ?? 0))[0];
  const anyUsable = states.some((state) => state.usable);
  const failed = states.find((state) => state.state === "failed" || state.state === "unavailable");
  const ready = states.find((state) => state.state === "ready");
  const aggregate: ResourceLaneReadiness = preparing
    ? { ...preparing, usable: anyUsable }
    : failed && anyUsable
      ? { state: "degraded", label: failed.label, usable: true }
      : failed ?? ready ?? value;
  return {
    ...updateLane(snapshot, plugin.lane, aggregate),
    plugins,
    activeLane: plugin.lane,
  };
}

function updateLane(
  snapshot: ResourceReadinessSnapshot,
  lane: ResourceLaneId,
  value: ResourceLaneReadiness,
): ResourceReadinessSnapshot {
  return { ...snapshot, [lane]: value, activeLane: lane };
}

/**
 * Compatibility reducer for the renderer's former single global status.
 *
 * Producers can arrive in any order: a late SVO compile no longer erases
 * fluid transport readiness, and a solver rebuild no longer makes the device
 * look unavailable. New resource owners should report the lane they unlock;
 * this reducer keeps the existing renderer API migratable one event at a time.
 */
export function reduceGPUResourceStatus(
  snapshot: ResourceReadinessSnapshot,
  status: GPUStatus,
): ResourceReadinessSnapshot {
  if (status.state === "initializing") {
    const lane = activityLane(status);
    const plugin = pluginForStatus(status, lane);
    const previous = snapshot.plugins[plugin.id] ?? snapshot[lane];
    const retainingPrevious = status.retainingPrevious === true || previous.usable;
    const activity: ResourceActivity = {
      id: status.phase ? `${plugin.id}.${status.phase}` : `${plugin.id}.initialization`,
      pluginId: plugin.id,
      lane,
      label: status.label,
      phase: status.phase ?? "planning",
      completed: status.completed ?? 0,
      total: status.total ?? 0,
      startedAt_ms: status.startedAt_ms ?? performance.now(),
      operation: status.operation,
      retainingPrevious,
    };
    return updatePlugin(snapshot, plugin, {
      state: "preparing",
      label: status.label,
      usable: retainingPrevious,
      activity,
    });
  }

  if (status.state === "ready") {
    // During migration an unscoped legacy `ready` meant the whole old global
    // lifecycle was ready. Preserve that meaning without making new scoped
    // plugin events global again.
    if (!status.resource && status.label !== "WebGPU renderer ready" && !sparseStatus(status)) {
      const platform = compatibilityPlugin(status, "platform");
      const withPlatform = updatePlugin(snapshot, platform, { state: "ready", label: status.label, usable: true });
      const fluid = compatibilityPlugin(status, "fluid");
      return updatePlugin(withPlatform, fluid, { state: "ready", label: status.label, usable: true });
    }
    const lane: ResourceLaneId = status.resource?.lane
      ?? (status.label === "WebGPU renderer ready" ? "platform" : sparseStatus(status) ? "svo" : "fluid");
    const plugin = pluginForStatus(status, lane);
    return updatePlugin(snapshot, plugin, { state: "ready", label: status.label, usable: true });
  }

  if (status.state === "blocked") {
    const lane: ResourceLaneId = status.resource?.lane ?? (sparseStatus(status) ? "svo" : "fluid");
    const plugin = pluginForStatus(status, lane);
    return updatePlugin(snapshot, plugin, {
      state: snapshot[lane].usable ? "degraded" : "failed",
      label: status.label,
      usable: snapshot[lane].usable,
    });
  }

  if (status.state === "manual") {
    const plugin = pluginForStatus(status, "platform");
    return updatePlugin(snapshot, plugin, { state: "idle", label: status.label, usable: false });
  }

  if (status.state === "stopping") {
    const plugin = pluginForStatus(status, "platform");
    return updatePlugin(snapshot, plugin, { state: "preparing", label: status.label, usable: false });
  }

  if (status.state === "unavailable" || status.state === "lost") {
    const plugin = pluginForStatus(status, "platform");
    return updatePlugin(snapshot, plugin, { state: "unavailable", label: status.label, usable: false });
  }

  return snapshot;
}

/** Evidence from published resources wins over the order of status messages. */
export function reduceGPUResourceEvidence(
  snapshot: ResourceReadinessSnapshot,
  gpuInfo?: GPUEulerianInfo | null,
  renderer?: EffectiveRendererStatus,
): ResourceReadinessSnapshot {
  let next = snapshot;
  if (gpuInfo?.initialSparseAuthorityReady === true && gpuInfo.initialRasterSurfaceReady === true) {
    const fluid = { state: "ready", label: "Fenced t=0 fluid authority ready", usable: true } as const;
    const plugins = Object.fromEntries(Object.entries(next.plugins).map(([id, state]) => [id,
      state.plugin.provides.includes("fluid-authority") ? { ...fluid, plugin: state.plugin } : state]));
    next = { ...next, fluid, plugins };
  }
  if (renderer?.state === "active") {
    const svo = { state: "ready", label: "Sparse voxel presentation ready", usable: true } as const;
    const plugins = Object.fromEntries(Object.entries(next.plugins).map(([id, state]) => [id,
      state.plugin.provides.includes("sparse-voxel-presentation") ? { ...svo, plugin: state.plugin } : state]));
    next = { ...next, svo, plugins };
  } else if (renderer?.state === "pending") {
    const previous = next.svo;
    next = { ...next, svo: {
      state: "preparing",
      label: renderer.detail ?? (renderer.failureReason === "pipeline-compiling"
        ? "Sparse presentation is compiling; presentation is held fail-closed"
        : "Waiting for the live sparse scene publication"),
      usable: false,
      activity: previous.activity,
    } };
  } else if (renderer?.state === "failed") {
    const plugins = Object.fromEntries(Object.entries(next.plugins).map(([id, state]) => [id,
      state.plugin.provides.includes("sparse-voxel-presentation")
        ? { state: "degraded" as const, label: "Sparse presentation failed closed", usable: false, plugin: state.plugin }
        : state]));
    next = { ...next, plugins, svo: {
      state: "degraded",
      label: renderer.detail
        ? `Sparse presentation failed closed: ${renderer.detail}`
        : "Sparse presentation failed closed; no substitute scene rendered",
      usable: false,
    } };
  }
  return next;
}

export function resourceActivity(snapshot: ResourceReadinessSnapshot): ResourceActivity | undefined {
  const preferred = snapshot.activeLane ? snapshot[snapshot.activeLane].activity : undefined;
  if (preferred) return preferred;
  return (["platform", "fluid", "svo", "optional"] as const)
    .map((lane) => snapshot[lane].activity)
    .find((activity) => activity !== undefined);
}

export function resourceActivities(snapshot: ResourceReadinessSnapshot): readonly ResourceActivity[] {
  return Object.values(snapshot.plugins)
    .flatMap((state) => state.activity ? [state.activity] : [])
    .sort((left, right) => left.startedAt_ms - right.startedAt_ms);
}

export function resourceCapabilityUsable(
  snapshot: ResourceReadinessSnapshot,
  capability: RuntimeResourceCapability,
): boolean {
  const plugins = Object.values(snapshot.plugins);
  const providers = plugins
    .filter((state) => state.plugin.provides.includes(capability));
  if (providers.length > 0) return providers.some((state) => state.usable);
  // Before the first scoped event, preserve the bootstrap state represented by
  // the lane. Once a plugin appears, only its explicit capability claim wins.
  const lane: ResourceLaneId = capability === "renderer" ? "platform"
    : capability === "fluid-authority" || capability === "water-presentation" ? "fluid"
    : capability === "live-scene" || capability === "sparse-voxel-presentation" ? "svo"
    : "optional";
  if (plugins.some((state) => state.plugin.lane === lane)) return false;
  return snapshot[lane].usable;
}

export function resourceInteractionGates(snapshot: ResourceReadinessSnapshot, fluidRequired: boolean) {
  const rendererUsable = resourceCapabilityUsable(snapshot, "renderer");
  const fluidUsable = resourceCapabilityUsable(snapshot, "fluid-authority")
    && resourceCapabilityUsable(snapshot, "water-presentation");
  return {
    /** React editing, camera controls, panels, import/export: never GPU-gated. */
    shellInteractive: true,
    /** Only device acquisition has no image to present. Solver/SVO work is progressive. */
    viewportInteractive: rendererUsable,
    /** Transport depends on authoritative fluid, not unrelated renderer work. */
    transportInteractive: rendererUsable && (!fluidRequired || fluidUsable),
  };
}

export function duplicateResourcePluginIds(definitions: readonly ResourcePluginDefinition[]) {
  const seen = new Set<string>(), duplicates = new Set<string>();
  for (const definition of definitions) {
    if (seen.has(definition.id)) duplicates.add(definition.id);
    seen.add(definition.id);
  }
  return [...duplicates];
}
