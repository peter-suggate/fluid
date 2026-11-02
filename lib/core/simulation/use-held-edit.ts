import { useEffect, useRef } from "react";
import type { PaneId } from "../session/session";
import { simulation, type EditGesture } from "./controller";

/**
 * A held control — a dial, a slider — as one edit gesture.
 *
 * The document is patched live for preview, and the gesture brackets the hold
 * so undo gets one entry per adjustment, not one per pointer-move. It opens on
 * the first change rather than on pointerdown, so keyboard nudges are
 * bracketed too, and closes on release, blur or unmount: deselecting mid-drag
 * still lands the adjustment.
 *
 * When another gesture is open the hold is refused its own, and its writes
 * join that one instead, which records them when it closes.
 */
export function useHeldEdit(paneId: PaneId, options: { readonly reseed: boolean }) {
  const gesture = useRef<EditGesture | undefined>(undefined);
  const { reseed } = options;
  const end = () => {
    const open = gesture.current;
    gesture.current = undefined;
    open?.commit({ reseed });
  };
  const endRef = useRef(end);
  endRef.current = end;
  useEffect(() => () => endRef.current(), []);
  const begin = (label: string) => {
    if (!gesture.current?.open) gesture.current = simulation.beginEdit(label, paneId);
  };
  return { begin, end };
}
