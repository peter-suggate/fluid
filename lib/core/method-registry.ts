import type { SimulationMethod } from "./method-contract";

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
  installed = installation;
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
 * The method a scene runs when it authors no profile of its own.
 *
 * Profiled presets are untouched: `profile?.methodId` wins over this
 * everywhere it is read.
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
