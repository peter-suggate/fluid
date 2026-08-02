/**
 * Static composition point for runtime resource plugins.
 *
 * Definitions remain colocated with their owners. This file only fixes
 * deterministic ordering and makes completeness/duplicate checks possible;
 * there is intentionally no mutable register() singleton (hot reload and
 * import order must not change product readiness).
 */
import { octreeMethod } from "./methods/octree";
import type { ResourcePluginDefinition, RuntimeResourceCapability } from "./resource-readiness";
import { liveSvoSceneResourcePlugin } from "./webgpu-live-svo-scene";
import { svoPresentationResourcePlugin } from "./webgpu-svo-dry-scene";
import { webGPUPlatformResourcePlugin } from "./webgpu-renderer";

export const RESOURCE_PLUGIN_CATALOG: readonly ResourcePluginDefinition[] = Object.freeze([
  webGPUPlatformResourcePlugin,
  liveSvoSceneResourcePlugin,
  ...(octreeMethod.resource ? [octreeMethod.resource] : []),
  svoPresentationResourcePlugin,
]);

export function resourcePluginsProviding(
  capability: RuntimeResourceCapability,
  definitions: readonly ResourcePluginDefinition[] = RESOURCE_PLUGIN_CATALOG,
) {
  return definitions.filter((definition) => definition.provides.includes(capability));
}
