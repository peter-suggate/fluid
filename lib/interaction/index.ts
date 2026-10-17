/** Headless public API: safe for editor, worker and Node callers. */
export * from "./contracts";
export { createInteractionCatalog, type InteractionCatalog } from "./catalog";

export { interactionCatalog, interactionRuntimes } from "./installed";
export { SceneInteractionHost, type InteractionHostOptions } from "./runtime/host";
export type { BodyRecipe, PhysicsBehaviorFactory, SensorContact, InteractionFluidDriver, FluidStepRequest } from "./runtime/physics";
export { InteractionInputRouter } from "./runtime/input";
