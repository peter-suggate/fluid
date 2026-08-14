import type { ResourcePluginDefinition } from "./resource-plugin";

/**
 * Device and canvas resources owned by the renderer.
 *
 * The declaration sits apart from the renderer that initializes them because
 * naming this resource must stay cheap: a status producer, a store's bootstrap
 * state or a readiness consumer needs the identity, not the ~10,000 lines of
 * pipeline construction that fulfils it.
 */
export const webGPUPlatformResourcePlugin: ResourcePluginDefinition = Object.freeze({
  id: "platform.webgpu-renderer",
  lane: "platform",
  label: "WebGPU renderer platform",
  provides: ["renderer"] as const,
  blocks: "viewport",
  phaseCopy: {
    planning: "Acquiring the browser GPU and selecting device capabilities.",
    renderer: "Preparing the canvas and minimum presentation resources.",
    "water-renderer": "Compiling rasterized water interfaces and optical compositing.",
  },
});
