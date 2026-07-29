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
 * - queue proven empty at submit → the fence measures the advance itself
 *   (submit-to-completion wall time, the queue-wall observation the trace
 *   system already accepts as its fallback source);
 * - uninterrupted target work already pending → completion-to-completion
 *   spacing, which under saturation is exactly the per-advance cost.
 * Observations containing presentation or diagnostic submissions are ignored.
 * No GPU work, no readbacks, no extra fences are added.
 */
export class GPUAdvanceWallEstimator {
  private ema_ms: number | undefined;
  private lastCompletionAt_ms: number | undefined;
  private lastInterveningWorkSequence: number | undefined;

  /** Rolling estimate of one advance in milliseconds; undefined until sampled. */
  get estimate_ms(): number | undefined { return this.ema_ms; }

  /**
   * Record one target-work completion observed by its queue fence.
   *
   * Completion spacing is usable only when no presentation/readback (or other
   * non-target queue work) was submitted between the two target submissions.
   * Submit-to-completion is usable only with positive evidence that the queue
   * was idle. Contaminated observations still advance the comparison anchor,
   * but they never enter the cost EMA.
   */
  observeCompletion(now_ms: number, submittedAt_ms: number, options: {
    readonly pendingTargetWorkAtSubmit: number;
    readonly interveningWorkSequence: number;
    readonly queueWasIdleAtSubmit: boolean;
  }) {
    const previousCompletionAt_ms = this.lastCompletionAt_ms;
    const uninterruptedTargetChain = options.pendingTargetWorkAtSubmit > 0
      && previousCompletionAt_ms !== undefined
      && this.lastInterveningWorkSequence === options.interveningWorkSequence;
    const sample_ms = uninterruptedTargetChain
      ? now_ms - previousCompletionAt_ms
      : options.queueWasIdleAtSubmit ? now_ms - submittedAt_ms : undefined;
    this.lastCompletionAt_ms = now_ms;
    this.lastInterveningWorkSequence = options.interveningWorkSequence;
    if (sample_ms === undefined || !Number.isFinite(sample_ms) || sample_ms <= 0) return;
    this.ema_ms = this.ema_ms === undefined
      ? sample_ms
      : this.ema_ms + 0.25 * (sample_ms - this.ema_ms);
  }

  /** A new solver, scene, or timeline invalidates the old cost model. */
  reset() {
    this.ema_ms = undefined;
    this.lastCompletionAt_ms = undefined;
    this.lastInterveningWorkSequence = undefined;
  }
}
