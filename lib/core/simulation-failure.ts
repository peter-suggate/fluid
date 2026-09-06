/** Serializable across renderer workers, UI stores and Dawn reports. */
export interface SimulationFailure {
  readonly method: string;
  readonly code: string;
  readonly message: string;
  readonly kernel: string;
  readonly frame: number;
  readonly generation: number;
  readonly ownerId: number;
  readonly operands: readonly number[];
  readonly operandNames?: readonly string[];
  readonly rawWords: readonly number[];
  readonly scene?: string;
  readonly time_s?: number;
}

export class SimulationFailureError extends Error {
  constructor(readonly failure: SimulationFailure) {
    super(`Simulation HALTED: ${failure.code}: ${failure.message}; kernel=${failure.kernel}; frame=${failure.frame}; generation=${failure.generation}; owner=${failure.ownerId}; operands=${failure.operands.join(",")}`);
    this.name = "SimulationFailureError";
  }
}
