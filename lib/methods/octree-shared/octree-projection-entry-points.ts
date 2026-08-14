/**
 * The projection shader's entry-point set, and who is allowed to skip one.
 *
 * Two consumers walk this list and they must not disagree. The activity
 * instrumenter rewrites every listed entry point to stamp a GPU task counter
 * and requires each one to exist in the module, which is the only thing that
 * keeps the coarse-grid timeline exhaustive as entry points are added. The
 * pipeline builder walks the same list to decide what to compile, and an entry
 * point it compiles but never dispatches is a cold pipeline paid for on every
 * solver construction.
 *
 * Naming an entry point once is the point. Split across two files, a name
 * could reach the instrumenter without reaching the compiler or the reverse,
 * and neither miss surfaces until a scene runs.
 */
export const OCTREE_PROJECTION_ACTIVITY_MODULE_ID = "octree/projection-topology";

export const OCTREE_PROJECTION_BASE_ENTRY_POINTS = [
  "rasterizeSolids", "resetTopology", "refineTopology", "balanceTopology",
  "rasterizeSolidsDelta", "resetTopologyDelta", "refineTopologyDelta", "balanceTopologyDelta",
  "stampFrontierAttempt", "beginFrontier", "classifyFrontierCandidates", "classifyFrontierCandidatesDelta",
  "prefixFrontierCandidateBlocks", "prefixFrontierCandidateBlocksDelta",
  "emitFrontierCandidates", "emitFrontierCandidatesDelta",
  "prepareFrontierDispatch", "sortFrontierCandidatesLocal",
  "classifyFrontierCarry",
  "scanFrontierCarryBlocks", "prefixFrontierCarryBlocks", "mergeFrontierRows", "finalizeFrontier",
  "prepareRowDelta", "classifyRowDelta",
  "finalizeRowDeltaClassification", "scanDirtyRowDeltaBlocks", "prefixDirtyRowDeltaBlocks",
  "scatterDirtyRowDelta", "markRowDeltaRing", "markRowDeltaRingBlocks",
  "scanAffectedRowDeltaBlocks",
  "prefixAffectedRowDeltaBlocks", "compactRowDelta", "publishRowDelta", "publishReusedRowDelta",
  "planLeaves", "scanLeafBlocks", "emitLeaves",
  "classifyTopologyTileSignature", "buildDirtyTileDelta", "buildDirtyFrontierDelta",
] as const;

const OCTREE_PROJECTION_VARIANT_ENTRY_POINTS = [
  "refineTopologyCoarse", "refineTopologyCoarseDelta",
  "balanceTopologyCoarse", "balanceTopologyCoarseDelta",
  "sortFrontierCandidates",
] as const;

export const OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS = Object.freeze([
  // The instrumentation transform requires every listed entry point to exist in
  // the module, which is the exhaustiveness guard.
  ...OCTREE_PROJECTION_BASE_ENTRY_POINTS,
  ...OCTREE_PROJECTION_VARIANT_ENTRY_POINTS,
]);

type OctreeProjectionActivityEntryPoint =
  typeof OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS[number];

const projectionActivityTaskName = (entryPoint: string) => entryPoint
  .replace(/([a-z0-9])([A-Z])/g, "$1-$2").toLowerCase();
const projectionActivityLabel = (entryPoint: string) => entryPoint
  .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
  .replace(/^./, (character) => character.toUpperCase());

export const OCTREE_PROJECTION_ACTIVITY_TASKS = Object.freeze(Object.fromEntries(
  OCTREE_PROJECTION_ACTIVITY_ENTRY_POINTS.map((entryPoint) => {
    const task = projectionActivityTaskName(entryPoint);
    return [entryPoint, Object.freeze({
      task,
      id: `gpu.physics.coarse-grid.${task}`,
      label: `Coarse grid · ${projectionActivityLabel(entryPoint)}`,
      phaseId: "coarse-grid",
    })];
  }),
)) as Readonly<Record<OctreeProjectionActivityEntryPoint, Readonly<{
  task: string; id: string; label: string; phaseId: "coarse-grid";
}>>>;

export interface OctreeProjectionPipelineReachability {
  readonly solidRasterization: boolean;
  readonly localFrontierCandidateSort: boolean;
  readonly cooperativeRowDeltaRing: boolean;
}

/** Compile only entry points reachable from the immutable solver configuration. */
export function octreeProjectionPipelineRequired(
  entryPoint: string,
  config: OctreeProjectionPipelineReachability,
): boolean {
  // These entry points exist only as specialization templates. Every encoded
  // fine refinement selects a target-size pipeline from `refineLevelPipelines`.
  if (entryPoint === "refineTopology" || entryPoint === "refineTopologyDelta") {
    return false;
  }
  if (entryPoint === "rasterizeSolids" || entryPoint === "rasterizeSolidsDelta") {
    return config.solidRasterization;
  }
  if (entryPoint === "sortFrontierCandidatesLocal") {
    return config.localFrontierCandidateSort;
  }
  if (entryPoint === "markRowDeltaRing") {
    return config.cooperativeRowDeltaRing;
  }
  if (entryPoint === "markRowDeltaRingBlocks") {
    return !config.cooperativeRowDeltaRing;
  }
  return true;
}
