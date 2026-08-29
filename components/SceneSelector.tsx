"use client";

import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { allSceneCards } from "../lib/core/scene-cards";
import type { SceneCard } from "../lib/core/scene-definition";
import {
  browserSceneLibraryStorage,
  readSceneLibrary,
  type SceneLibraryEntry,
} from "../lib/core/scene-library";
import {
  readSceneRecents,
  recentSceneCards,
  recordSceneOpen,
  type RecentSceneOpen,
} from "../lib/core/scene-recents";
import { sceneNameMatch, sceneSearchGroups, sceneSearchOrder } from "../lib/core/scene-search";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";

/**
 * Which scene this pane is running, changed without leaving the studio.
 *
 * The library is the front door and stays the front door: it is a page of
 * tiles, with previews and saves and renames, for a reader who is *browsing*.
 * This is the other gesture — the reader knows which scene they want and wants
 * it *here*, in this pane, without a route change that unmounts a running
 * viewport and rebuilds it. In compare mode it is the only way to say the thing
 * the mode exists for at its coarsest: two different scenes, one clock.
 *
 * Per pane by construction. It reads the session it is mounted under and opens
 * into it, so the same component under pane B's provider chooses pane B's
 * scene — see `docs/ab-compare-handoff.md`. Its search text is local state and
 * deliberately not the shell store's `librarySearch`: two panes typing into one
 * page-level box would filter each other's lists.
 *
 * Opening retains this pane's configuration (`retainConfiguration`): the
 * solver, its tuning and the raised instrument are the experiment the reader
 * set up, and a scene swap moves the experiment rather than ending it. The
 * camera is the exception — a scene is framed by its author — which under a
 * linked View reframes both panes.
 */

/** How many recently opened scenes head the list before anything is typed. */
const RECENT_LIMIT = 5;

export function SceneSelector() {
  const router = useRouter();
  const session = useSession();
  const presetId = session.scene((state) => state.presetId);
  const setOpen = session.ui((state) => state.setSceneSelectorOpen);
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState<readonly SceneLibraryEntry[]>([]);
  const [recents, setRecents] = useState<readonly RecentSceneOpen[]>([]);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);

  // Storage is only readable in the browser, and this component mounts exactly
  // when the popover opens — so mounting is the read, and a scene saved since
  // the last time it was raised is in the list without any invalidation.
  useEffect(() => {
    const storage = browserSceneLibraryStorage();
    setEntries(readSceneLibrary(storage));
    setRecents(readSceneRecents(storage));
  }, []);

  const cards = useMemo(() => allSceneCards(entries), [entries]);
  const groups = useMemo(() => sceneSearchGroups(cards, query, {
    recent: recentSceneCards(recents, cards, RECENT_LIMIT),
  }), [cards, query, recents]);
  const order = useMemo(() => sceneSearchOrder(groups), [groups]);

  // The best answer is always the first row, so typing re-aims the cursor at
  // it: a highlight left on row nine of the previous query would make Enter
  // open a scene the reader is no longer looking at.
  useEffect(() => { setActive(0); }, [query]);

  useEffect(() => {
    listRef.current
      ?.querySelector<HTMLElement>(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, groups]);

  // A press anywhere else puts it down. Registered a frame late so the very
  // press that opened it cannot be the press that closes it, and blind to the
  // toggles themselves so clicking the pane tag twice reads as a toggle rather
  // than as a close followed by a re-open.
  useEffect(() => {
    let stop = () => {};
    const frame = requestAnimationFrame(() => {
      const onPointerDown = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (rootRef.current?.contains(target)) return;
        if (target instanceof Element && target.closest("[data-scene-selector-toggle]")) return;
        setOpen(false);
      };
      window.addEventListener("pointerdown", onPointerDown, true);
      stop = () => window.removeEventListener("pointerdown", onPointerDown, true);
    });
    return () => { cancelAnimationFrame(frame); stop(); };
  }, [setOpen]);

  const choose = (card: SceneCard) => {
    setOpen(false);
    // The notice on a refused card is the controller's; a stored document that
    // an older schema wrote fails here rather than becoming a corrupt live
    // scene, and the list stays where it was so the reader can pick another.
    if (!simulation.openSceneCard(card, session.id, { retainConfiguration: true })) return;
    recordSceneOpen(browserSceneLibraryStorage(), card.id, Date.now());
  };

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") { event.preventDefault(); setOpen(false); return; }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      if (order.length === 0) return;
      const step = event.key === "ArrowDown" ? 1 : -1;
      setActive((current) => (current + step + order.length) % order.length);
      return;
    }
    if (event.key === "Enter") {
      event.preventDefault();
      const card = order[active];
      if (card) choose(card);
    }
  };

  let index = -1;
  return (
    <div
      ref={rootRef}
      className="scene-selector"
      data-testid="scene-selector"
      role="dialog"
      aria-label="Choose a scene for this pane"
      onKeyDown={onKeyDown}
    >
      <label className="scene-selector-search">
        <span className="visually-hidden">Search scenes</span>
        <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.6" /><path d="M10.4 10.4 14 14" /></svg>
        <input
          type="search"
          autoFocus
          value={query}
          placeholder="Search scenes"
          aria-label="Search scenes"
          data-testid="scene-selector-search"
          onChange={(event) => setQuery(event.target.value)}
        />
      </label>
      <div className="scene-selector-list" ref={listRef} role="listbox" aria-label="Scenes">
        {groups.map((group) => (
          <Fragment key={group.shelf}>
            <p className="scene-selector-shelf">{group.shelf}</p>
            {group.cards.map((card) => {
              index += 1;
              const at = index;
              const mark = sceneNameMatch(card, query);
              return (
                <button
                  key={`${group.shelf}/${card.id}`}
                  type="button"
                  role="option"
                  aria-selected={at === active}
                  data-index={at}
                  data-active={at === active}
                  data-current={card.id === presetId}
                  className="scene-selector-row"
                  title={card.blurb}
                  // Pointer move rather than hover CSS: the cursor and the
                  // keyboard have to agree on which row Enter opens, and two
                  // separately drawn highlights would say they do not.
                  onPointerMove={() => setActive(at)}
                  onClick={() => choose(card)}
                >
                  <strong>{mark
                    ? <>{card.name.slice(0, mark[0])}<mark>{card.name.slice(mark[0], mark[1])}</mark>{card.name.slice(mark[1])}</>
                    : card.name}</strong>
                  <small>{card.shelf}</small>
                </button>
              );
            })}
          </Fragment>
        ))}
        {order.length === 0 && <p className="scene-selector-empty">No scene matches that.</p>}
      </div>
      {/* The library is still one click away, because this list has no previews,
          no saves and no renames — everything a reader does *to* a scene rather
          than with it lives there. */}
      <button
        type="button"
        className="scene-selector-browse"
        data-testid="scene-selector-browse"
        onClick={() => { setOpen(false); router.push("/"); }}
      >Browse library…</button>
    </div>
  );
}
