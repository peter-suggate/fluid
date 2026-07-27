/** Exact direct-structured publication audit used by smoke and performance gates.
 *
 * The ABI and validators now live in `lib/structured-authority-audit.ts` so the
 * browser solver's step-coherent snapshot ring shares the identical record
 * layout and decode with the Dawn harness. This module remains the harness's
 * import path. */
export * from "../lib/structured-authority-audit";
