"use client";

import type { EditorAction, EditorActionEffect } from "../lib/core/editor-action";
import { performEditorAction } from "../lib/core/editor-action-runtime";
import type { PaneSession } from "../lib/core/session/session";
import { useSession } from "../lib/core/session/session-context";
import { RadialRing } from "./RadialRing";

/**
 * The studio's contextual ring: `RadialRing` bound to the pane's session.
 *
 * The ring itself is host-agnostic — see `RadialRing.tsx` for the geometry, the
 * walk and the keyboard. All that is left here is the binding a host supplies:
 * which store the open menu is read from, and who performs the chosen effect.
 *
 * `perform` is a prop rather than an import so a second host can bring its own
 * performer over its own effects. The 3-D studio takes the default, which is
 * the one runtime that knows about history, arming and the solver; the 2-D
 * advance lab has no solver at all and passes its own.
 */
export function RadialMenu({ perform = performEditorAction }: {
  readonly perform?: (effect: EditorActionEffect, session: PaneSession) => void;
} = {}) {
  const session = useSession();
  const menu = session.ui((state) => state.radialMenu);
  const closeRadialMenu = session.ui((state) => state.closeRadialMenu);
  if (!menu) return null;
  // Closed before performed, and in that order: an effect that opens a panel or
  // arms a gesture must not be undone by a ring still tidying itself away.
  const choose = (action: EditorAction) => {
    if (!action.effect) return;
    closeRadialMenu();
    perform(action.effect, session);
  };
  return <RadialRing menu={menu} onChoose={choose} onClose={closeRadialMenu} />;
}
