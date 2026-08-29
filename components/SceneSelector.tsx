"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { allSceneCards, cachedSceneCardGlyph, sceneCardGlyph, sceneCardPreview } from "../lib/core/scene-cards";
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
import {
  SCENE_SEARCH_RECENT_SHELF,
  sceneNameMatch,
  sceneSearchGroups,
  sceneSearchOrder,
  sceneSearchRows,
  sceneSearchStep,
  type SceneGridKey,
} from "../lib/core/scene-search";
import { useSession } from "../lib/core/session/session-context";
import { simulation } from "../lib/core/simulation/controller";
import { SceneIsoGlyph } from "./SceneIsoGlyph";

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
 * It shows the library's mark, at the library's proportions, because a name is
 * not enough to recognise a scene by: half this catalog is a room with water in
 * it, and "Dam break", "Twin dam collision" and eight oracles named after
 * figures in a paper are told apart by their *shape*. So the list is a grid of
 * thumbnails and the arrow keys move in two dimensions, while the search box
 * above it still ranks — the reader who knows the name types it and presses
 * Enter, and the reader who does not looks.
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

/**
 * How many recently opened scenes head the grid before anything is typed.
 * Two full rows, so the shelf that matters most is never a ragged one.
 */
const RECENT_LIMIT = 6;

/**
 * Tiles across, fixed rather than measured.
 *
 * The popover is a fixed width and the grid is authored to it, which is what
 * lets the arrow keys know where "up" is without reading layout back out of the
 * DOM. A measured column count would make the keyboard's model of the grid a
 * frame stale exactly when the pane is being resized.
 */
const COLUMNS = 3;

const GRID_KEYS: readonly SceneGridKey[] = ["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"];

/**
 * A tile's mark, built when it is nearly on screen rather than when it is
 * listed.
 *
 * Drawing a card means *building its document* — `sceneCardPreview` runs the
 * scene factory, and the four heaviest in the catalog expand terrain fields and
 * scenery graphs for 40 to 160 ms each. The library can pay that on a page with
 * no viewport on it; this popover opens over a running one, and seventy-odd
 * cards at once would be most of a second of frozen water.
 *
 * So a tile draws its well immediately and its mark when the reader is actually
 * looking near it. `sceneCardGlyph` caches on document identity, so a card built
 * once — here, or on the library page before the reader came in — is free
 * forever after, and that is what the initial state reads: a second opening of
 * the popover paints whole rather than filling in.
 */
function SceneSelectorArt({ card }: { card: SceneCard }) {
  const [drawn, setDrawn] = useState(() => cachedSceneCardGlyph(card) !== undefined);
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (drawn) return;
    const node = ref.current;
    // No observer means no scrolling worth deferring for either — a test DOM,
    // not a reader — so draw rather than stay blank forever.
    if (!node || typeof IntersectionObserver !== "function") { setDrawn(true); return; }
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        observer.disconnect();
        setDrawn(true);
      },
      // A row and a half of margin, so the marks are there by the time a
      // scroll brings them up rather than appearing under the reader's eye.
      { root: node.closest(".scene-selector-list"), rootMargin: "180px" },
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [drawn]);

  const glyph = drawn ? sceneCardGlyph(card) : undefined;
  const scene = glyph ? sceneCardPreview(card) : undefined;
  // The library's own well and wash (`.card-art`), so a scene is the same
  // picture wherever it is offered — and its own dashed frame for a document
  // that will not build, which is a different thing from one not drawn yet.
  return (
    <span className="card-art scene-selector-art" ref={ref}>
      {scene && glyph ? <SceneIsoGlyph scene={scene} glyph={glyph} />
        : drawn ? <span className="card-art-missing" aria-hidden="true" />
        : <span className="scene-selector-art-blank" aria-hidden="true" />}
    </span>
  );
}

function SceneSelectorTile({ card, at, active, current, query, shelf, choose, aim }: {
  card: SceneCard;
  at: number;
  active: boolean;
  current: boolean;
  query: string;
  /** Drawn under the name only where the group heading does not already say it. */
  shelf: boolean;
  choose: (card: SceneCard) => void;
  aim: (at: number) => void;
}) {
  const mark = sceneNameMatch(card, query);
  return (
    <button
      type="button"
      role="option"
      aria-selected={active}
      data-index={at}
      data-active={active}
      data-current={current}
      data-testid={`scene-selector-tile-${card.id}`}
      className="scene-selector-tile"
      title={card.blurb}
      // Pointer move rather than hover CSS: the cursor and the keyboard have to
      // agree on which tile Enter opens, and two separately drawn highlights
      // would say they do not.
      onPointerMove={() => aim(at)}
      onClick={() => choose(card)}
    >
      <SceneSelectorArt card={card} />
      <span className="scene-selector-name">
        <strong>{mark
          ? <>{card.name.slice(0, mark[0])}<mark>{card.name.slice(mark[0], mark[1])}</mark>{card.name.slice(mark[1])}</>
          : card.name}</strong>
        {shelf && <small>{card.shelf}</small>}
      </span>
    </button>
  );
}

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
  const rows = useMemo(() => sceneSearchRows(groups, COLUMNS), [groups]);

  // The best answer is always the first tile, so typing re-aims the cursor at
  // it: a highlight left on the ninth tile of the previous query would make
  // Enter open a scene the reader is no longer looking at.
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
    if (GRID_KEYS.includes(event.key as SceneGridKey)) {
      const key = event.key as SceneGridKey;
      // The search box keeps left and right while there is text to move a caret
      // through: the reader is editing a word, not walking a row. Up and down
      // are always the grid's — a search field has nothing to do with them.
      const caret = event.target instanceof HTMLInputElement && event.target.value.length > 0;
      if (caret && (key === "ArrowLeft" || key === "ArrowRight")) return;
      event.preventDefault();
      setActive((current) => sceneSearchStep(rows, current, key));
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
          <div className="scene-selector-group" role="group" aria-label={group.shelf} key={group.shelf}>
            {/* A shelf heading, not a section: the eyebrow says which family the
                tiles below belong to and then gets out of the way of them. */}
            <p className="scene-selector-shelf">{group.shelf}</p>
            <div className="scene-selector-grid">
              {group.cards.map((card) => {
                index += 1;
                const at = index;
                return (
                  <SceneSelectorTile
                    key={`${group.shelf}/${card.id}`}
                    card={card}
                    at={at}
                    active={at === active}
                    current={card.id === presetId}
                    query={query}
                    // Recent is the one shelf whose cards came from everywhere
                    // else, so it is the one place the shelf still has to be
                    // written on the tile.
                    shelf={group.shelf === SCENE_SEARCH_RECENT_SHELF}
                    choose={choose}
                    aim={setActive}
                  />
                );
              })}
            </div>
          </div>
        ))}
        {order.length === 0 && <p className="scene-selector-empty">No scene matches that.</p>}
      </div>
      {/* The library is still one click away, because this grid has no saves,
          no renames and no chips — everything a reader does *to* a scene rather
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
