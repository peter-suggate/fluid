import type { FluidPipelineGraph } from "./fluid-pipeline";

/**
 * Method-id → pipeline-graph registry for the simulation panel.
 *
 * Graphs live beside their encoders (the colocation contract in
 * `fluid-pipeline.ts`), so loading one pulls in the solver module — already
 * paid for methods whose plugin statically imports its solver, and loaded
 * lazily here so the panel never forces a method's GPU code just to ask
 * whether a diagram exists.
 */
const FLUID_PIPELINE_LOADERS: Readonly<Record<string, () => Promise<FluidPipelineGraph>>> = {
  uniform: async () => (await import("./webgpu-uniform-reference")).UNIFORM_FLUID_PIPELINE,
};

export function fluidPipelineSupported(methodId: string): boolean {
  return methodId in FLUID_PIPELINE_LOADERS;
}

export async function loadFluidPipeline(methodId: string): Promise<FluidPipelineGraph | undefined> {
  return FLUID_PIPELINE_LOADERS[methodId]?.();
}
