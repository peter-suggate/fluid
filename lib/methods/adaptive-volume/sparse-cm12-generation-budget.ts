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
