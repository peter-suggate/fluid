import type { SimulationMethod } from "./method-contract";
import type { SceneDescription } from "./model";

/**
 * Where core code looks up a simulation method without knowing which ones
 * exist.
 *
 * The renderer needs a method's runtime parameter keys, its solver factories
 * and its resource declaration. Reaching for the method package to get them
 * makes every method — and everything a method reaches — a dependency of the
 * renderer, which is exactly the cycle that welded the uniform, octree and
 * render graphs into one strongly-connected component. The registry inverts
 * that: methods install themselves, core only reads.
 *
 * Installation happens at composition roots (the app shell and the render
 * worker), which import the method package for its side effect. A lookup
 * before installation is a wiring bug in a new entry point, so it throws with
 * the fix rather than silently falling back to some default method and
 * simulating the wrong thing.
 */
/**
 * What one installation declares.
 *
 * `interactive` is a strict subset of `methods`: the picker offers it, while
 * the full set stays reachable for offline comparison lanes that should not
 * appear as a product choice.
 */
export interface SimulationMethodInstallation {
  readonly methods: ReadonlyArray<SimulationMethod>;
  readonly interactive: ReadonlyArray<SimulationMethod>;
  readonly defaultId: string;
}

let installed: SimulationMethodInstallation | undefined;

export function installSimulationMethods(installation: SimulationMethodInstallation): void {
  const byId = new Map(installation.methods.map(method => [method.id, method]));
  if (byId.size !== installation.methods.length) throw new Error("Duplicate simulation method ID");
  if (!byId.has(installation.defaultId)) throw new Error(`Unknown default simulation method ${installation.defaultId}`);
  const interactive = new Set<string>();
  for (const method of installation.interactive) {
    if (byId.get(method.id) !== method) throw new Error(`Interactive method ${method.id} is not the installed definition`);
    if (interactive.has(method.id)) throw new Error(`Duplicate interactive method ${method.id}`);
    interactive.add(method.id);
  }
  for (const method of installation.methods) {
    if (!method.composition || !Object.isFrozen(method.composition)) {
      throw new Error(`Method ${method.id} must supply a resolved feature composition`);
    }
  }
  installed = Object.freeze({
    methods: Object.freeze([...installation.methods]),
    interactive: Object.freeze([...installation.interactive]),
    defaultId: installation.defaultId,
  });
}

function installation(): SimulationMethodInstallation {
  if (!installed) {
    throw new Error(
      "No simulation methods are installed. The entry point must import the method package "
      + "(`import \"@/lib/methods\"`) before constructing a renderer or solver.",
    );
  }
  return installed;
}

export function registeredSimulationMethods(): ReadonlyArray<SimulationMethod> {
  return installation().methods;
}

/** The methods offered as production/experimental choices in the UI. */
export function interactiveSimulationMethods(): ReadonlyArray<SimulationMethod> {
  return installation().interactive;
}

/**
 * The method the application selects when a scene is opened without an
 * explicit method in its URL. Scene profiles may seed method-specific tuning,
 * but they do not replace this product-wide default.
 */
export function defaultMethodId(): string {
  return installation().defaultId;
}

/** The id itself when it names an interactive method, otherwise the default. */
export function interactiveMethodId(id: string): string {
  const { interactive, defaultId } = installation();
  return interactive.some((method) => method.id === id) ? id : defaultId;
}

export function getMethod(id: string): SimulationMethod {
  const { methods, defaultId } = installation();
  const match = methods.find((method) => method.id === id);
  if (match) return match;
  const fallback = methods.find((method) => method.id === defaultId);
  if (!fallback) throw new Error(`Unknown simulation method "${id}" and no default is installed.`);
  return fallback;
}

/**
 * Whether the scene's SVO may descend below the simulation lattice.
 *
 * A dry scene always may. A wet one may only when its method's solver keeps no
 * sparse world, so the renderer's sidecar is the tree being refined and the
 * solver never sees the extra levels. See
 * `SimulationMethod.renderRefinementBelowSolver`.
 */
export function svoRenderRefinementPermitted(
  scene: Pick<SceneDescription, "systems">,
  methodId: string,
): boolean {
  return scene.systems?.fluid === false || getMethod(methodId).renderRefinementBelowSolver === true;
}
