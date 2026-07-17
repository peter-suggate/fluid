export const DEFAULT_TARGET_FPS = 60;
export const MIN_TARGET_FPS = 24;
export const MAX_TARGET_FPS = 120;

/** Keep externally supplied frame rates within the range supported by the UI. */
export function clampTargetFps(value: number): number {
  if (!Number.isFinite(value)) return DEFAULT_TARGET_FPS;
  return Math.min(MAX_TARGET_FPS, Math.max(MIN_TARGET_FPS, Math.round(value)));
}

export function frameInterval_ms(targetFps: number): number {
  return 1000 / clampTargetFps(targetFps);
}

/**
 * requestAnimationFrame timestamps can land a fraction of a millisecond before
 * the nominal deadline. A small tolerance avoids accidentally halving a 60 Hz
 * stream on a 60 Hz display while still pacing 30/60 Hz on faster displays.
 */
export function presentationFrameDue(lastFrameAt_ms: number, now_ms: number, targetFps: number): boolean {
  return !Number.isFinite(lastFrameAt_ms) || now_ms - lastFrameAt_ms + 0.5 >= frameInterval_ms(targetFps);
}

/**
 * Advance the nominal presentation clock rather than resetting it to the
 * latest rAF timestamp. This permits rates such as 24 Hz on a 60 Hz display or
 * 90 Hz on a 120 Hz display by choosing the nearest available callbacks.
 */
export function advancePresentationClock(lastFrameAt_ms: number, now_ms: number, targetFps: number): number {
  const interval = frameInterval_ms(targetFps);
  if (!Number.isFinite(lastFrameAt_ms) || now_ms - lastFrameAt_ms > interval * 2) return now_ms;
  return lastFrameAt_ms + interval;
}
