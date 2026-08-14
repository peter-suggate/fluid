import type { FineToCoarseGPUControl } from "../octree-shared/webgpu-octree-fine-to-coarse-levelset";
import type { OctreePowerCoarseLevelSetControl } from "./webgpu-octree-power-coarse-levelset";

export interface Section5RestrictionAudit {
  readonly acceptedRows: number;
  readonly unacceptedRows: number;
  readonly interfaceRows: number;
  readonly correctedRows: number;
  readonly failure?: "restriction-invalid" | "background-invalid" | "row-count-mismatch"
    | "correction-count-mismatch" | "interface-not-covered";
}

/**
 * Audit the two independent level-set authorities from Aanjaneya et al. 2017
 * Section 5 (`docs/papers/aanjaneya-2017-power-liquids.txt`). The fine SPGrid
 * is intentionally a narrow surface band, so rows outside it are owned by the
 * background octree and their fraction is not a failure. The actual contract
 * is that every accepted fine row becomes exactly one coarse correction and
 * that the accepted band is large enough to cover all interface rows.
 */
export function auditSection5FineRestriction(
  restriction: FineToCoarseGPUControl,
  coarse: OctreePowerCoarseLevelSetControl,
): Section5RestrictionAudit {
  const acceptedRows = Math.max(0, restriction.rowCount - restriction.unacceptedRows);
  const result = {
    acceptedRows,
    unacceptedRows: restriction.unacceptedRows,
    interfaceRows: coarse.interfaceRows,
    correctedRows: coarse.correctedRows,
  };
  if (restriction.flags !== 0 || !restriction.valid) {
    return { ...result, failure: "restriction-invalid" };
  }
  if (coarse.flags !== 0 || coarse.valid === 0) {
    return { ...result, failure: "background-invalid" };
  }
  if (restriction.rowCount !== coarse.rowCount) {
    return { ...result, failure: "row-count-mismatch" };
  }
  if (acceptedRows !== coarse.correctedRows) {
    return { ...result, failure: "correction-count-mismatch" };
  }
  if (acceptedRows < coarse.interfaceRows) {
    return { ...result, failure: "interface-not-covered" };
  }
  return result;
}
