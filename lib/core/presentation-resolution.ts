/** Water retains 136 bytes per render pixel, plus 16 with world reflections.
 * Cap extra supersampling at eight megapixels. Scenes with a playback budget
 * may also shade below native resolution. Every pass and pointer transform
 * uses the reported scale. */
export function boundedPresentationScale(width: number, height: number, requested: number, textureLimit: number,
  playbackPixelBudget?: number): number {
  const nativePixels = Math.max(1, width * height);
  const extraPixelLimit = Math.max(nativePixels, 8 * 1024 * 1024);
  const pixelLimit = Math.min(extraPixelLimit, playbackPixelBudget ?? extraPixelLimit);
  return Math.min(Number.isFinite(requested) ? Math.max(.35, requested) : 1,
    2, Math.sqrt(pixelLimit / nativePixels), textureLimit / width, textureLimit / height);
}

const playbackScales = [1, .88, .77] as const;

/** Adjust only the authored playback budget. Completion counts include the
 * fluid solve, all shading and the final blit. Higher quality needs sustained
 * spare queue time as well as a full display cadence. */
export class PlaybackPixelBudget {
  private base = 0;
  private level = 0;
  private start_ms?: number;
  private startCount = 0;
  private fastWindows = 0;
  private completionLatency_ms = 0;

  observeCompletion(latency_ms: number) {
    this.completionLatency_ms += .08 * (latency_ms - this.completionLatency_ms);
  }

  resolve(budget: number | undefined, now_ms: number, completed: number): number | undefined {
    if (budget === undefined) {
      this.start_ms = undefined;
      this.fastWindows = 0;
      return undefined;
    }
    if (budget !== this.base) {
      this.base = budget;
      this.level = 0;
      this.start_ms = undefined;
    }
    if (this.start_ms === undefined || completed < this.startCount) {
      this.start_ms = now_ms;
      this.startCount = completed;
    } else if (now_ms - this.start_ms >= 1800) {
      const count = completed - this.startCount;
      const fps = 1000 * count / (now_ms - this.start_ms);
      this.fastWindows = fps >= 59.6 && this.completionLatency_ms > 0 && this.completionLatency_ms < 9
        ? this.fastWindows + 1 : 0;
      if (count > 0 && fps < 59.0) this.level = Math.min(playbackScales.length - 1, this.level + 1);
      else if (this.fastWindows >= 6) {
        this.level = Math.max(0, this.level - 1);
        this.fastWindows = 0;
      }
      this.start_ms = now_ms;
      this.startCount = completed;
    }
    // Keep both axes at the same scale. The middle step is roughly 608 by
    // 342 for the pool scene. The solver lattice and editor remain independent.
    const scale = playbackScales[this.level]!;
    return Math.round(budget * scale * scale);
  }
}
