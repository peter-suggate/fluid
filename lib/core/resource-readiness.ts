import type { GPUEulerianInfo } from "./webgpu-eulerian";
import type { EffectiveRendererStatus } from "./renderer-status";
import type { GPUStatus } from "./gpu-status";
import type {
  ResourceLaneId,
  ResourceLaneState,
  ResourcePluginDefinition,
  RuntimeResourceCapability,
} from "./resource-plugin";

export type {
  ResourceLaneId,
  ResourceLaneState,
  ResourcePluginDefinition,
  RuntimeResourceCapability,
} from "./resource-plugin";

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

/**
 * Attribution comes from the producer, never from its copy.
 *
 * A status whose owner is missing cannot be placed: guessing a lane would let
 * a rewritten user-facing label move work between capabilities.
 */
/**
 * The owning plugin, checked at run time as well as at compile time.
 *
 * `GPUStatus.resource` is required, so an in-process producer cannot omit it.
 * A status can also arrive from the render worker as a structured clone, and
 * that path is only as sound as the message it was built from — so the lane
 * still refuses to guess rather than falling back to an inference from the
 * label text, which is how rewording a progress string used to re-route a lane.
 */
function statusPlugin(status: GPUStatus): ResourcePluginDefinition {
  if (!status.resource) {
    throw new Error(
      `GPU status "${status.label}" must declare the resource plugin that owns it; `
      + "add `resource: <owner>ResourcePlugin` to the call that publishes it");
  }
  return status.resource;
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
 * Folds one owner's device-lifecycle status into that owner's plugin entry.
 *
 * Producers can arrive in any order: a late SVO compile no longer erases
 * fluid transport readiness, and a solver rebuild no longer makes the device
 * look unavailable. Each event only moves the resource that published it.
 */
export function reduceGPUResourceStatus(
  snapshot: ResourceReadinessSnapshot,
  status: GPUStatus,
): ResourceReadinessSnapshot {
  const plugin = statusPlugin(status);
  const lane = plugin.lane;
  switch (status.state) {
    case "initializing": {
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
    case "ready":
      return updatePlugin(snapshot, plugin, { state: "ready", label: status.label, usable: true });
    case "blocked":
      return updatePlugin(snapshot, plugin, {
        state: snapshot[lane].usable ? "degraded" : "failed",
        label: status.label,
        usable: snapshot[lane].usable,
      });
    case "manual":
      return updatePlugin(snapshot, plugin, { state: "idle", label: status.label, usable: false });
    case "stopping":
      return updatePlugin(snapshot, plugin, { state: "preparing", label: status.label, usable: false });
    case "unavailable":
    case "lost":
      return updatePlugin(snapshot, plugin, { state: "unavailable", label: status.label, usable: false });
  }
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
  if (renderer?.state === "active" || renderer?.state === "not-required") {
    const svo = { state: "ready", label: renderer.state === "active"
      ? "Sparse voxel presentation ready" : "Sparse voxel presentation not required", usable: true } as const;
    const plugins = Object.fromEntries(Object.entries(next.plugins).map(([id, state]) => [id,
      state.plugin.provides.includes("sparse-voxel-presentation") ? { ...svo, plugin: state.plugin } : state]));
    next = { ...next, svo, plugins };
  } else if (renderer?.state === "pending") {
    const previous = next.svo;
    const svo = {
      state: "preparing",
      label: renderer.detail ?? (renderer.failureReason === "pipeline-compiling"
        ? "Sparse presentation is compiling; presentation is held fail-closed"
        : "Waiting for the live sparse scene publication"),
      usable: false,
      activity: previous.activity,
    } as const;
    const plugins = Object.fromEntries(Object.entries(next.plugins).map(([id, state]) => [id,
      state.plugin.provides.includes("sparse-voxel-presentation")
        ? { ...svo, activity: state.activity ?? svo.activity, plugin: state.plugin }
        : state]));
    next = { ...next, svo, plugins };
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

/**
 * How much screen in-flight work may claim, decided by what it takes away.
 *
 * A card is the loudest thing the shell can put over a live viewport, so only
 * work that has actually taken scene interaction earns one. Work that suspends
 * transport states itself beside the controls it suspends, and work that blocks
 * nothing gets a pill. An unrecognised declaration takes the pill: over-claiming
 * would cover a viewport the user can still use.
 */
export type ResourceActivityPresentation = "card" | "transport-inline" | "pill";

export function resourceActivityPresentation(
  plugin?: Pick<ResourcePluginDefinition, "blocks">,
): ResourceActivityPresentation {
  switch (plugin?.blocks) {
    case "viewport": return "card";
    case "transport": return "transport-inline";
    default: return "pill";
  }
}

export function resourceActivitiesFor(
  snapshot: ResourceReadinessSnapshot,
  presentation: ResourceActivityPresentation,
): readonly ResourceActivity[] {
  return resourceActivities(snapshot).filter((activity) =>
    resourceActivityPresentation(snapshot.plugins[activity.pluginId]?.plugin) === presentation);
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

/**
 * What the user may do now, by capability rather than by whether work is running.
 *
 * These are four answers and not one because they stop being true at different
 * moments. A single "the viewport is interactive" bit meant an ordinary SVO
 * recompile — which only replaces the image — took the camera and every editor
 * gizmo with it. Looking at a scene from another angle needs a device to draw
 * with; resolving a ray against the scene needs a complete published generation
 * for the ray to hit. Only the second waits for a rebuild.
 */
export function resourceInteractionGates(snapshot: ResourceReadinessSnapshot, fluidRequired: boolean) {
  const rendererUsable = resourceCapabilityUsable(snapshot, "renderer");
  const scenePresentationUsable = resourceCapabilityUsable(snapshot, "sparse-voxel-presentation");
  const fluidUsable = resourceCapabilityUsable(snapshot, "fluid-authority")
    && resourceCapabilityUsable(snapshot, "water-presentation");
  return {
    /** React editing, panels, import/export: never GPU-gated. */
    shellInteractive: true,
    /** Orbit, pan, zoom and framing move our own camera; they need only a device. */
    cameraInteractive: rendererUsable,
    /** A ray may only resolve against geometry a complete GLOBAL SVO generation published. */
    pickingInteractive: rendererUsable && scenePresentationUsable,
    /** Advancing an invisible/partial scene would break simulation/presentation lockstep. */
    transportInteractive: rendererUsable && scenePresentationUsable && (!fluidRequired || fluidUsable),
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
