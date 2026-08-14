/**
 * Diagnostic-only pass partitioning for Dawn algorithm attribution.
 *
 * Production keeps compatible kernels in shared compute passes. The opt-in
 * smoke diagnostic closes selected passes at hypothesis boundaries so the
 * pass-local timestamp audit can distinguish serial publication, A2, energy
 * summaries, and topology-transfer work. It never changes dispatch counts,
 * buffers, shader inputs, or GPU/CPU synchronization.
 */
export function octreeAlgorithmDiagnosticsEnabled(
  environment?: Readonly<Record<string, string | undefined>>,
): boolean {
  const resolved = environment
    ?? (typeof process !== "undefined" ? process.env : undefined);
  return resolved?.FLUID_ALGORITHM_DIAGNOSTICS === "1";
}
