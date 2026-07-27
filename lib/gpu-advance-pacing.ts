/**
 * Always-on wall-clock estimate of one GPU physics advance.
 *
 * Physics admission (pre-presentation slack and rolling queue depth) used to
 * read `info.physicsTrace`, which exists only while performance
 * instrumentation is enabled. Toggling the profiler therefore changed the
 * scheduler itself: an uninstrumented session never saw pre-presentation
 * slack and stayed at the bootstrap queue depth, while an instrumented one
 * reordered physics against presentation and ran a measured 1..8 depth. The
 * driver must present the solver with the same procedure regardless of
 * observation, so scheduling consumes this estimator instead; the hardware
 * trace remains telemetry.
 *
 * Sampling uses the fences the renderer already registers per advance:
 * - queue empty at submit → the fence measures the advance itself
 *   (submit-to-completion wall time, the queue-wall observation the trace
 *   system already accepts as its fallback source);
 * - queue busy at submit → completion-to-completion spacing, which under
 *   saturation is exactly the per-advance cost.
 * No GPU work, no readbacks, no extra fences are added.
 */
export class GPUAdvanceWallEstimator {
  private ema_ms: number | undefined;
  private lastCompletionAt_ms: number | undefined;

  /** Rolling estimate of one advance in milliseconds; undefined until sampled. */
  get estimate_ms(): number | undefined { return this.ema_ms; }

  /** Record one advance completion observed by its queue fence. */
  observeCompletion(now_ms: number, submittedAt_ms: number, pendingAtSubmit: number) {
    const sample_ms = pendingAtSubmit > 0 && this.lastCompletionAt_ms !== undefined
      ? now_ms - this.lastCompletionAt_ms
      : now_ms - submittedAt_ms;
    this.lastCompletionAt_ms = now_ms;
    if (!Number.isFinite(sample_ms) || sample_ms <= 0) return;
    this.ema_ms = this.ema_ms === undefined
      ? sample_ms
      : this.ema_ms + 0.25 * (sample_ms - this.ema_ms);
  }

  /** A new solver, scene, or timeline invalidates the old cost model. */
  reset() {
    this.ema_ms = undefined;
    this.lastCompletionAt_ms = undefined;
  }
}
