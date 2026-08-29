"use client";

import { MousePointer2, Orbit } from "lucide-react";
import { toggledViewportMode, VIEWPORT_MODES } from "../lib/core/editor-viewport-mode";
import { useSession } from "../lib/core/session/session-context";

/**
 * The one permanent control in the top-right: look, or edit.
 *
 * It earns a fixed place where the tool strip did not, because it is not a verb
 * — it reports and changes the single fact that decides whether *any* pointer
 * gesture reaches the scene. A reader who clicks and gets nothing has to be
 * able to see why without first knowing to right-click, and a reader watching a
 * solve has to be able to see that their pointer is safe. Neither is a question
 * the contextual ring can answer, since in LOOK the ring is not about anything.
 *
 * One button rather than two: the modes are exclusive and there are exactly
 * two, so a segmented pair would spend twice the width to say the same thing.
 * The icon is the mode you are *in*; the label beside it names it.
 */
export function ViewportModeToggle() {
  const session = useSession();
  const viewportMode = session.ui((state) => state.viewportMode);
  const setViewportMode = session.ui((state) => state.setViewportMode);
  const spec = VIEWPORT_MODES[viewportMode];
  const Icon = viewportMode === "interact" ? MousePointer2 : Orbit;
  return (
    <button
      type="button"
      className="viewport-mode-toggle"
      data-viewport-mode={viewportMode}
      data-testid="viewport-mode-toggle"
      aria-pressed={viewportMode === "interact"}
      title={`${spec.hint} · Tab`}
      onClick={() => setViewportMode(toggledViewportMode(viewportMode))}
    >
      <Icon width={15} height={15} strokeWidth={1.7} aria-hidden />
      <strong>{spec.label}</strong>
      <em>tab</em>
    </button>
  );
}
