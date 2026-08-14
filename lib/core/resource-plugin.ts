/**
 * The resource-plugin protocol: what a capability owner declares about itself.
 *
 * This is the narrow half of `resource-readiness.ts` — pure declaration, no
 * orchestration and no renderer knowledge. It lives on its own so the method
 * contract can name a plugin without dragging the readiness machinery (and,
 * through it, the renderer and the SVO stack) into every method package.
 */

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
