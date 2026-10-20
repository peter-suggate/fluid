/**
 * A pane's presentation frame rate, handed from its render loop to whichever
 * readout is showing that pane.
 *
 * The number is written straight into the element rather than through a store:
 * it changes every frame, and a store write would re-render its subscribers at
 * the presentation rate. The readout lives on the host's transport bar, not in
 * the pane, so the pane cannot hold a ref to it. Instead the bar attaches its
 * element for the focused pane, and gets the latest text immediately on attach
 * so that switching focus never shows a blank.
 */

const NO_FRAME_RATE = "— FPS";

const latest = new Map<string, string>();
const sinks = new Map<string, HTMLElement>();

export function publishFrameRate(paneId: string, fps: number | undefined) {
  const text = fps === undefined ? NO_FRAME_RATE : `${fps.toFixed(1)} FPS`;
  latest.set(paneId, text);
  const sink = sinks.get(paneId);
  if (sink) sink.textContent = text;
}

/** Show `paneId`'s rate in `element` until the returned detach is called. */
export function attachFrameRateReadout(paneId: string, element: HTMLElement): () => void {
  sinks.set(paneId, element);
  element.textContent = latest.get(paneId) ?? NO_FRAME_RATE;
  return () => { if (sinks.get(paneId) === element) sinks.delete(paneId); };
}
