/** Compact pressure bootstrap/fault receipt stored after the immutable worklist. */
export const SPARSE_CM12_PRESSURE_REPAIR_HEADER = Object.freeze({
  cellFirstFault: 0,
  rowFirstFault: 1,
  fault: 2,
  bootstrapCellIndirect: 3,
  bootstrapRowIndirect: 6,
} as const);

export const SPARSE_CM12_PRESSURE_REPAIR_HEADER_WORDS = 9;

export const SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_WORDS = 6;
export const SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_BYTES =
  4 * SPARSE_CM12_PRESSURE_MEMBERSHIP_INDIRECT_WORDS;
