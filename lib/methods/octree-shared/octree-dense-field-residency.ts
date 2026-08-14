/**
 * Which dense, domain-sized fields a scene actually has to pay for.
 *
 * The octree is sparse everywhere except a few structures still indexed by
 * finest cell. A scene with no terrain and no rigid bodies has nothing to put
 * in the solid fraction/owner field or the sparse brick world, and allocating
 * them anyway spends `nx*ny*nz*8` bytes holding zeros. The solid plan returns
 * an eight-byte allocation rather than none because a bind group with a
 * missing entry is a validation error while a minimal one is free.
 *
 * The release predicate is the fragile case. The bootstrap dense phi texture
 * may only be dropped once every recurring consumer has taken its handoff to
 * the compact publication; dropping it while one still samples it leaves that
 * consumer reading a destroyed resource, so the predicate is a conjunction of
 * every handoff plus an explicit veto for a consumer that cannot migrate.
 */

export function octreeSparseWorldRequired(
  hasTerrain: boolean,
  rigidBodyCount: number,
): boolean {
  return hasTerrain || rigidBodyCount > 0;
}

interface OctreeDensePhiReleaseState {
  globalFineBootstrapped: boolean;
  coarseProjectionGroupsActive: boolean;
  fineSeedCoarseNative: boolean;
  topologyUsesFineSeedCandidates: boolean;
  compactRendererSourceReady: boolean;
  incompatibleDenseConsumer: boolean;
}

/** All recurring consumers must complete their bind-group handoff before destroy. */
export function octreeDensePhiReleaseReady(state: OctreeDensePhiReleaseState): boolean {
  return state.globalFineBootstrapped
    && state.coarseProjectionGroupsActive
    && state.fineSeedCoarseNative
    && state.topologyUsesFineSeedCandidates
    && state.compactRendererSourceReady
    && !state.incompatibleDenseConsumer;
}

interface OctreeSolidCellAllocationPlan {
  allocatedBytes: number;
  denseBytes: number;
  savedBytes: number;
  hasDenseField: boolean;
}

/** Keep one valid `{ fraction, owner }` binding when a scene has no solids. */
export function planOctreeSolidCellAllocation(
  dims: { nx: number; ny: number; nz: number },
  hasTerrain: boolean,
  rigidBodyCount: number,
): OctreeSolidCellAllocationPlan {
  const denseBytes = Math.max(8, dims.nx * dims.ny * dims.nz * 8);
  const hasDenseField = hasTerrain || rigidBodyCount > 0;
  const allocatedBytes = hasDenseField ? denseBytes : 8;
  return { allocatedBytes, denseBytes, savedBytes: denseBytes - allocatedBytes, hasDenseField };
}
