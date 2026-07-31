import type { SceneDescription } from "./model";
import type {
  GPUInitializationPhase,
  GPUInitializationTask,
} from "./gpu-initialization";

/**
 * Capabilities describe executable GPU behaviour, not scene names.  A scene
 * contributes facts (terrain, moving bodies, solver kind); shader families
 * state which facts they require.  Keeping that relationship here prevents a
 * preset switch statement from becoming the shader manifest.
 */
export type GPUShaderCapability =
  | "fluid-simulation"
  | "adaptive-topology"
  | "solid-fields"
  | "fine-interface"
  | "distributed-frontier-sort"
  | "diagnostic-overlays"
  | "logical-activity";

export interface GPUShaderCapabilityInputs {
  solver: "uniform" | "quadtree" | "octree";
  fineInterface?: boolean;
  distributedFrontierSort?: boolean;
  diagnosticOverlays?: boolean;
  logicalActivity?: boolean;
}

export interface GPUShaderCapabilityPlan {
  readonly values: ReadonlySet<GPUShaderCapability>;
  has(...capabilities: readonly GPUShaderCapability[]): boolean;
  readonly cacheKey: string;
}

/** Derive shader reachability entirely from content and runtime features. */
export function planGPUShaderCapabilities(
  scene: Pick<SceneDescription, "systems" | "terrain" | "rigidBodies">,
  inputs: GPUShaderCapabilityInputs,
): GPUShaderCapabilityPlan {
  const fluidEnabled = scene.systems?.fluid !== false;
  const values = new Set<GPUShaderCapability>();
  if (fluidEnabled) values.add("fluid-simulation");
  if (fluidEnabled && inputs.solver !== "uniform") values.add("adaptive-topology");
  if (scene.terrain !== undefined || scene.rigidBodies.length > 0) values.add("solid-fields");
  if (fluidEnabled && inputs.fineInterface) values.add("fine-interface");
  if (inputs.distributedFrontierSort) values.add("distributed-frontier-sort");
  if (inputs.diagnosticOverlays) values.add("diagnostic-overlays");
  if (inputs.logicalActivity) values.add("logical-activity");
  const ordered = Object.freeze([...values].sort()) as readonly GPUShaderCapability[];
  const readonlyValues: ReadonlySet<GPUShaderCapability> = new Set(ordered);
  return Object.freeze({
    values: readonlyValues,
    has: (...capabilities: readonly GPUShaderCapability[]) =>
      capabilities.every((capability) => readonlyValues.has(capability)),
    cacheKey: ordered.join("|"),
  });
}

export interface GPUShaderTaskDefinition {
  readonly id: string;
  readonly label: string;
  readonly phase?: GPUInitializationPhase;
  readonly requires?: readonly GPUShaderCapability[];
  readonly dependencies?: readonly string[];
  compile(signal: AbortSignal): void | Promise<void>;
}

/**
 * Turn a declarative shader manifest into initialization tasks.  Unreachable
 * definitions disappear before registration, so progress totals describe the
 * programs that will really be compiled.
 */
export function planGPUShaderTasks(
  capabilities: GPUShaderCapabilityPlan,
  definitions: readonly GPUShaderTaskDefinition[],
): GPUInitializationTask[] {
  const ids = new Set<string>();
  for (const definition of definitions) {
    if (!definition.id) throw new Error("GPU shader task IDs must be non-empty");
    if (ids.has(definition.id)) throw new Error(`Duplicate GPU shader task: ${definition.id}`);
    ids.add(definition.id);
  }
  return definitions
    .filter(({ requires = [] }) => capabilities.has(...requires))
    .map(({ compile, requires: _requires, phase = "solver-pipelines", ...task }) => ({
      ...task,
      phase,
      run: compile,
    }));
}
