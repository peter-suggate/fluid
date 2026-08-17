/**
 * GPU-owned pressure-membership epoch control.
 *
 * The host encodes the same bootstrap and incremental command sequence every
 * frame. A one-workgroup device planner snapshots the mode for that epoch and
 * publishes zero-count indirect dispatches for the inactive branch. Keeping
 * the mode in the persistent repair header prevents a bootstrap finalizer from
 * accidentally enabling incremental kernels later in the same command buffer.
 */
export const SPARSE_CM12_PRESSURE_REPAIR_HEADER = Object.freeze({
  initialized: 0,
  cellChanges: 1,
  rowChanges: 2,
  fault: 3,
  topologyGeneration: 4,
  epoch: 5,
  mode: 6,
  bootstrapCellIndirect: 7,
  bootstrapRowIndirect: 10,
} as const);

export const SPARSE_CM12_PRESSURE_REPAIR_HEADER_WORDS = 13;

export const SPARSE_CM12_PRESSURE_MEMBERSHIP_MODE = Object.freeze({
  failClosed: 0,
  bootstrap: 1,
  incremental: 2,
} as const);

export const SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_WORDS = 6;
export const SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES =
  4 * SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_WORDS;

