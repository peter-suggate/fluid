import { installSimulationMethods } from "../core/method-registry";
import { registerOctreeCoarseDynamicsLane } from "./octree-shared/octree-coarse-dynamics-lane";
import { registerOctreeDeviceRowCapacityCeiling } from "./octree-shared/octree-pressure-capacity";
import { OctreeLosassoCoarseDynamicsLane } from "./losasso/octree-losasso-lane";
import { losassoMethod } from "./losasso/method";
import { OctreePowerCoarseDynamicsLane } from "./power/octree-power-lane";
import { spgridRowCapacityForBindingLimit } from "./power/webgpu-octree-spgrid-vcycle";
import { structuredVelocityRowCapacityForBindingLimit } from "./power/webgpu-octree-structured-velocity-gpu";
import { powerLiquidsMethod } from "./power/method";
import { uniformMethod } from "./uniform/method";
import type { SimulationMethod } from "../core/method-contract";

/**
 * The method catalog: the one module that names every simulation method.
 *
 * Importing it installs them, so an entry point wires the product up with a
 * single side-effect import and nothing else needs to know the list. Core,
 * the UI and the harness read through `core/method-registry` instead — that
 * asymmetry is the decoupling: the catalog may reach a method, and nothing
 * that a method can reach may reach the catalog.
 */
const simulationMethods: ReadonlyArray<SimulationMethod> = [losassoMethod, powerLiquidsMethod, uniformMethod];

/**
 * The octree engine resolves its coarse dynamics lane by backend id, and this
 * is the only module allowed to name both lanes: neither method package may
 * import a sibling method package, and `octree-shared` may not import either.
 * Installing here is what keeps adding a third backend out of all three.
 */
registerOctreeCoarseDynamicsLane("power2017", (engine) => new OctreePowerCoarseDynamicsLane(engine));
registerOctreeCoarseDynamicsLane("losasso", (engine) => new OctreeLosassoCoarseDynamicsLane(engine));

/**
 * The engine's row arena is sized against the device once, for whichever lane
 * runs, and the two arenas that bind it hardest are the power backend's packed
 * A/B structured-velocity authority and its SPGrid pyramid. `octree-shared`
 * may not import a method to ask them, and making the ceiling lane-specific
 * would raise Losasso's row capacity rather than relocate a symbol, so the
 * catalog installs the same arithmetic for both lanes -- exactly what the
 * engine computed inline before it was allowed to name the power package.
 */
registerOctreeDeviceRowCapacityCeiling(({ dimensions, maximumStorageBindingBytes, plannedRowCapacity }) =>
  Math.min(
    structuredVelocityRowCapacityForBindingLimit(maximumStorageBindingBytes),
    spgridRowCapacityForBindingLimit(dimensions, maximumStorageBindingBytes, plannedRowCapacity),
  ));

installSimulationMethods({
  methods: simulationMethods,
  // Power Liquids is interactive rather than comparison-only because it was
  // reachable from the picker before the split (as the octree method's
  // `coarseBackend` select) and because `interactiveMethodId` silently
  // substitutes the default for a non-interactive id: a `method=power-liquids`
  // link would have hydrated as uniform and simulated something else.
  interactive: [losassoMethod, powerLiquidsMethod, uniformMethod],
  // Uniform is the default for any scene that does not author a method profile.
  defaultId: uniformMethod.id,
});
