/**
 * Tracks presentation cadence from a rolling window of frame intervals.
 * Averaging durations before taking their reciprocal avoids the upward bias
 * caused by averaging individual FPS values.
 */
export class SmoothedFrameRate {
  private lastFrameAt_ms: number | undefined;
  private readonly intervals_ms: number[] = [];
  private readonly completionSamples: { completedFrames: number; observedAt_ms: number }[] = [];

  constructor(readonly smoothingFrames = 5) {
    if (!Number.isInteger(smoothingFrames) || smoothingFrames < 1) {
      throw new RangeError("FPS smoothing window must contain at least one frame interval");
    }
  }

  sample(frameAt_ms: number): number | undefined {
    if (!Number.isFinite(frameAt_ms)) return this.framesPerSecond;
    if (this.lastFrameAt_ms !== undefined && frameAt_ms > this.lastFrameAt_ms) {
      this.intervals_ms.push(frameAt_ms - this.lastFrameAt_ms);
      if (this.intervals_ms.length > this.smoothingFrames) this.intervals_ms.shift();
    }
    this.lastFrameAt_ms = frameAt_ms;
    return this.framesPerSecond;
  }

  /**
   * Measure completed-frame throughput from a cumulative GPU completion count.
   *
   * Completion promises can be delivered to JavaScript in batches. Sampling the
   * cumulative counter over a time window avoids reporting a burst of callbacks
   * as thousands of frames per second.
   */
  sampleCompleted(completedFrames: number, observedAt_ms: number, window_ms = 500): number | undefined {
    if (!Number.isFinite(completedFrames) || !Number.isFinite(observedAt_ms) || !Number.isFinite(window_ms) || window_ms <= 0) {
      return this.completedFramesPerSecond;
    }
    const count = Math.max(0, Math.floor(completedFrames));
    const previous = this.completionSamples.at(-1);
    if (previous && (count < previous.completedFrames || observedAt_ms < previous.observedAt_ms)) {
      this.completionSamples.length = 0;
    }
    const latest = this.completionSamples.at(-1);
    if (latest?.observedAt_ms === observedAt_ms) {
      latest.completedFrames = count;
    } else {
      this.completionSamples.push({ completedFrames: count, observedAt_ms });
    }
    const cutoff_ms = observedAt_ms - window_ms;
    while (this.completionSamples.length > 2 && this.completionSamples[1].observedAt_ms <= cutoff_ms) {
      this.completionSamples.shift();
    }
    return this.completedFramesPerSecond;
  }

  reset(): void {
    this.lastFrameAt_ms = undefined;
    this.intervals_ms.length = 0;
    this.completionSamples.length = 0;
  }

  get framesPerSecond(): number | undefined {
    if (this.intervals_ms.length === 0) return undefined;
    const averageFrameTime_ms = this.intervals_ms.reduce((sum, interval) => sum + interval, 0) / this.intervals_ms.length;
    return 1000 / averageFrameTime_ms;
  }

  get completedFramesPerSecond(): number | undefined {
    const first = this.completionSamples[0], last = this.completionSamples.at(-1);
    if (!first || !last || first === last) return undefined;
    const elapsed_ms = last.observedAt_ms - first.observedAt_ms;
    if (elapsed_ms <= 0) return undefined;
    return 1000 * Math.max(0, last.completedFrames - first.completedFrames) / elapsed_ms;
  }
}
