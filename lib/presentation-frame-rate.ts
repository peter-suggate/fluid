const DEFAULT_STALE_AFTER_MS = 1_000;

/**
 * Counts completed presentation submissions over short sampling windows.
 * The last frame in one window is retained as the first frame in the next so
 * adjacent samples do not lose the interval across the sampling boundary.
 */
export class PresentationFrameRateTracker {
  private firstFrameAt_ms: number | null = null;
  private lastFrameAt_ms: number | null = null;
  private frameCount = 0;

  constructor(private readonly staleAfter_ms = DEFAULT_STALE_AFTER_MS) {}

  record(timestamp_ms: number) {
    if (!Number.isFinite(timestamp_ms)) return;
    if (this.lastFrameAt_ms !== null && (timestamp_ms < this.lastFrameAt_ms || timestamp_ms - this.lastFrameAt_ms > this.staleAfter_ms)) {
      this.reset();
    }
    if (this.firstFrameAt_ms === null) this.firstFrameAt_ms = timestamp_ms;
    this.lastFrameAt_ms = timestamp_ms;
    this.frameCount += 1;
  }

  sample(now_ms: number): number | null {
    if (!Number.isFinite(now_ms) || this.lastFrameAt_ms === null || now_ms - this.lastFrameAt_ms > this.staleAfter_ms) {
      this.reset();
      return null;
    }
    if (this.firstFrameAt_ms === null || this.frameCount < 2) return null;
    const elapsed_ms = this.lastFrameAt_ms - this.firstFrameAt_ms;
    const frames = this.frameCount - 1;
    this.firstFrameAt_ms = this.lastFrameAt_ms;
    this.frameCount = 1;
    return elapsed_ms > 0 ? frames * 1_000 / elapsed_ms : null;
  }

  reset() {
    this.firstFrameAt_ms = null;
    this.lastFrameAt_ms = null;
    this.frameCount = 0;
  }
}

const presentationFrameRate = new PresentationFrameRateTracker();

export const recordPresentedFrame = (timestamp_ms: number) => presentationFrameRate.record(timestamp_ms);
export const samplePresentationFrameRate = (now_ms: number) => presentationFrameRate.sample(now_ms);
export const resetPresentationFrameRate = () => presentationFrameRate.reset();
