import type { SliceTopology } from "./slice-topology";

export interface SliceRetirementReceipt {
  readonly generation: number;
  readonly topologyChangedBrickIds: Uint32Array;
  readonly retiredBrickIds: Uint32Array;
  readonly reshapedBrickIds: Uint32Array;
  /** Must stay exact zero: policy cannot retire numerical residue. */
  readonly retiredResidueMassFineCells: number;
  /** Dynamic WDR leaves become releasable only after presentation publication. */
  readonly pendingDynamicReleaseIds: Uint32Array;
}

export interface SliceRetirementAuthority {
  readonly authoredLeafIds: ReadonlySet<number>;
  readonly receipt: SliceRetirementReceipt;
}

export function createSliceRetirementAuthority(topology: SliceTopology): SliceRetirementAuthority {
  return { authoredLeafIds: new Set(topology.bricks.map(brick => brick.id)), receipt: {
    generation: topology.generation, topologyChangedBrickIds: new Uint32Array(),
    retiredBrickIds: new Uint32Array(), reshapedBrickIds: new Uint32Array(),
    retiredResidueMassFineCells: 0, pendingDynamicReleaseIds: new Uint32Array() } };
}

/** CPU markIncrementalActivityPostTopology/finalizeIncrementalActivityMasks. */
export function publishSliceRetirementAuthority(previous: SliceRetirementAuthority,
  before: SliceTopology, after: SliceTopology, beforeDensity: ArrayLike<number>): SliceRetirementAuthority {
  const changed: number[] = [], retired: number[] = [], reshaped: number[] = [], pending: number[] = [];
  let residue = 0;
  const nextByKey = after.brickByKey;
  for (const brick of before.bricks) {
    const next = nextByKey.get(brick.key), wasActive = brick.active !== false,
      isActive = next?.active !== false;
    if (!next || next.resolution !== brick.resolution || wasActive !== isActive) changed.push(brick.id);
    if (wasActive && !isActive) {
      retired.push(brick.id);
      for (const cell of before.cells) if (cell.brickKey === brick.key) {
        residue = Math.fround(residue
          + Math.fround(beforeDensity[cell.id]! * cell.volumeFineCells));
      }
      if (!previous.authoredLeafIds.has(brick.id)) pending.push(brick.id);
    } else if (next && next.resolution !== brick.resolution) reshaped.push(brick.id);
  }
  return { authoredLeafIds: previous.authoredLeafIds, receipt: {
    generation: after.generation,
    topologyChangedBrickIds: Uint32Array.from(changed),
    retiredBrickIds: Uint32Array.from(retired),
    reshapedBrickIds: Uint32Array.from(reshaped),
    retiredResidueMassFineCells: Math.fround(residue),
    pendingDynamicReleaseIds: Uint32Array.from(pending),
  } };
}

export function acknowledgeSliceRetirementReleases(
  authority: SliceRetirementAuthority,
): SliceRetirementAuthority {
  return authority.receipt.pendingDynamicReleaseIds.length === 0 ? authority : {
    authoredLeafIds: authority.authoredLeafIds,
    receipt: { ...authority.receipt, pendingDynamicReleaseIds: new Uint32Array() },
  };
}
