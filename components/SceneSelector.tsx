"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { allSceneCards } from "../lib/core/scene-cards";
import {
  browserSceneLibraryStorage,
  readSceneLibrary,
  type SceneLibraryEntry,
} from "../lib/core/scene-library";
import { recordSceneOpen } from "../lib/core/scene-recents";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { ScenePickerPopover } from "./ScenePickerPopover";

/**
 * Which scene this pane is running, changed without leaving the studio.
 *
 * The library is the front door and stays the front door: it is a page of
 * tiles, with saves and renames and chips, for a reader who is *browsing*.
 * This is the other gesture — the reader wants a scene *here*, in this pane,
 * without a route change that unmounts a running viewport and rebuilds it. In
 * compare mode it is the only way to say the thing the mode exists for at its
 * coarsest: two different scenes, one clock.
 *
 * The grid of marks, its search and its two-dimensional arrow keys are
 * `ScenePickerPopover`, shared with every other surface that offers a scene —
 * a reader who learns this list in the studio has learned it everywhere. What
 * is left here is the part that is the studio's alone: which cards exist for
 * this person, and what opening one *means*.
 *
 * Per pane by construction. It reads the session it is mounted under and opens
 * into it, so the same component under pane B's provider chooses pane B's
 * scene — see `docs/ab-compare-handoff.md`.
 *
 * Opening retains this pane's configuration (`retainConfiguration`): the
 * solver, its tuning and the raised instrument are the experiment the reader
 * set up, and a scene swap moves the experiment rather than ending it. The
 * camera is the exception — a scene is framed by its author — which under a
 * linked View reframes both panes.
 */
export function SceneSelector() {
  const router = useRouter();
  const session = useSession();
  const presetId = session.scene((state) => state.presetId);
  const setOpen = session.ui((state) => state.setSceneSelectorOpen);
  const [entries, setEntries] = useState<readonly SceneLibraryEntry[]>([]);

  // Storage is only readable in the browser, and this component mounts exactly
  // when the popover opens — so mounting is the read, and a scene saved since
  // the last time it was raised is in the list without any invalidation.
  useEffect(() => { setEntries(readSceneLibrary(browserSceneLibraryStorage())); }, []);

  const cards = useMemo(() => allSceneCards(entries), [entries]);

  return (
    <ScenePickerPopover
      cards={cards}
      currentId={presetId}
      label="Choose a scene for this pane"
      choose={(card) => {
        // The notice on a refused card is the controller's; a stored document
        // that an older schema wrote fails here rather than becoming a corrupt
        // live scene, and the list stays where it was so the reader can pick
        // another.
        if (!simulation.openSceneCard(card, session.id, { retainConfiguration: true })) return;
        setOpen(false);
        recordSceneOpen(browserSceneLibraryStorage(), card.id, Date.now());
      }}
      close={() => setOpen(false)}
      // The library is still one click away, because this grid has no saves,
      // no renames and no chips — everything a reader does *to* a scene rather
      // than with it lives there.
      browse={{ label: "Browse library…", run: () => { setOpen(false); router.push("/"); } }}
    />
  );
}
