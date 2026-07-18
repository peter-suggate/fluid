export const PRESENTATION_FPS = 60;

export function frameInterval_ms(): number {
  return 1000 / PRESENTATION_FPS;
}

/**
 * requestAnimationFrame timestamps can land a fraction of a millisecond before
 * the nominal deadline. A small tolerance avoids accidentally halving a 60 Hz
 * stream on a 60 Hz display while still pacing 30/60 Hz on faster displays.
 */
export function presentationFrameDue(lastFrameAt_ms: number, now_ms: number): boolean {
  return !Number.isFinite(lastFrameAt_ms) || now_ms - lastFrameAt_ms + 0.5 >= frameInterval_ms();
}

/**
 * Advance the nominal presentation clock rather than resetting it to the
 * latest rAF timestamp, avoiding drift when callbacks arrive slightly late.
 */
export function advancePresentationClock(lastFrameAt_ms: number, now_ms: number): number {
  const interval = frameInterval_ms();
  if (!Number.isFinite(lastFrameAt_ms) || now_ms - lastFrameAt_ms > interval * 2) return now_ms;
  return lastFrameAt_ms + interval;
}

/**
 * Paused viewports retain their last canvas image and only need another GPU
 * presentation when an input to that image changes. Object.is preserves the
 * reference-identity checks used for immutable store snapshots while also
 * handling scalar inputs such as the canvas size and device pixel ratio.
 */
export function presentationStateChanged(previous: readonly unknown[] | undefined, current: readonly unknown[]): boolean {
  return !previous || previous.length !== current.length || current.some((value, index) => !Object.is(value, previous[index]));
}
