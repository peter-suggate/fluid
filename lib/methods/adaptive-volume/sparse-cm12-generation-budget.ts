/** A recoverable reservation refusal; accepted physics remains authoritative. */
export class SparseCM12GenerationBudgetDeferred extends Error {
  constructor(readonly requestedBytes: number, readonly availableBytes: number) {
    super(`CM12 topology generation needs ${requestedBytes} bytes; ${availableBytes} are available`);
    this.name = "SparseCM12GenerationBudgetDeferred";
  }
}

/** Preparation lost its ownership snapshot while accepted physics advanced. */
export class SparseCM12GenerationStale extends Error {
  constructor() { super("CM12 prepared topology is stale; accepted simulation retained"); this.name = "SparseCM12GenerationStale"; }
}

/** Default aggregate reservation for old resident, candidate and preparation.
 * The seed atlas may be tiny: budget the permitted generation envelope, not
 * only three copies of that seed. Measured resident bytes include geometric
 * flux, adjacency, source, moving-solid and presentation scratch allocations.
 */
export function sparseGeometricGenerationByteBudget(input: Readonly<{
  residentBytes: number; residentCells: number; residentLeaves: number;
  maximumCells: number; maximumLeaves: number; maximumBufferBytes: number;
  explicitMaximumBytes?: number;
}>): number {
  if (input.explicitMaximumBytes !== undefined) {
    if (!Number.isSafeInteger(input.explicitMaximumBytes) || input.explicitMaximumBytes < 0) {
      throw new RangeError("topologyGenerationMaximumBytes must be a nonnegative safe integer");
    }
    return input.explicitMaximumBytes;
  }
  const growth = Math.max(1, input.maximumCells / Math.max(1, input.residentCells),
    input.maximumLeaves / Math.max(1, input.residentLeaves));
  const seedCoexistence = 3 * input.residentBytes;
  // maxBufferSize is a conservative aggregate default policy ceiling here,
  // not a claim about physical VRAM. Never shrink the prior seed reservation.
  // Explicit user budgets are returned unchanged above; individual allocations
  // remain subject to the device's actual per-buffer limits.
  const policyCeiling = Math.max(seedCoexistence, input.maximumBufferBytes);
  return Math.ceil(Math.min(seedCoexistence * growth, policyCeiling, Number.MAX_SAFE_INTEGER));
}
