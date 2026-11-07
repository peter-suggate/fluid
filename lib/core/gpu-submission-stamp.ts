/**
 * Submission stamp: a host readback is accepted only when the submission that
 * was meant to produce it ran to its last kernel.
 *
 * The host takes a fresh nonzero stamp for each submission (`next`) and writes
 * it into a uniform word with the submission's parameters. The submission's
 * final kernel copies that word into a reserved word of the buffer the host
 * reads back, and the host compares the two (`check`) before trusting any
 * other word. A command buffer that Metal aborted, or one that never ran,
 * leaves the previous submission's stamp (or the zero of a cleared buffer) in
 * the readback, so the check refuses it instead of reading stale or zero data
 * as a result.
 */
export const GPU_SUBMISSION_STAMP_LIMIT = 0xffffffff;

export class GPUSubmissionStamp {
  private issued = 0;
  constructor(readonly label: string) {}
  /** The stamp for the next submission: nonzero, increasing, wrapping to 1 after 2^32 - 1. */
  next(): number {
    this.issued = this.issued >= GPU_SUBMISSION_STAMP_LIMIT ? 1 : this.issued + 1;
    return this.issued;
  }
  /** Throws `<label>: stale GPU readback (stamp <read>, expected <expected>)` unless the readback carries `expected`. */
  check(read: number, expected: number): void {
    if (!Number.isInteger(expected) || expected < 1 || expected > GPU_SUBMISSION_STAMP_LIMIT) throw new Error(`${this.label}: invalid expected GPU submission stamp ${expected}`);
    if (read !== expected) throw new Error(`${this.label}: stale GPU readback (stamp ${read}, expected ${expected})`);
  }
}
