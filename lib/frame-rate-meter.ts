/**
 * Tracks presentation cadence from a rolling window of frame intervals.
 * Averaging durations before taking their reciprocal avoids the upward bias
 * caused by averaging individual FPS values.
 */
export class SmoothedFrameRate {
  private lastFrameAt_ms: number | undefined;
  private readonly intervals_ms: number[] = [];

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

  reset(): void {
    this.lastFrameAt_ms = undefined;
    this.intervals_ms.length = 0;
  }

  get framesPerSecond(): number | undefined {
    if (this.intervals_ms.length === 0) return undefined;
    const averageFrameTime_ms = this.intervals_ms.reduce((sum, interval) => sum + interval, 0) / this.intervals_ms.length;
    return 1000 / averageFrameTime_ms;
  }
}
