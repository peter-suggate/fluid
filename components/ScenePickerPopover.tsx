"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { cachedSceneCardGlyph, sceneCardGlyph, sceneCardPreview } from "../lib/core/scene-cards";
import type { SceneCard } from "../lib/core/scene-definition";
import { browserSceneLibraryStorage } from "../lib/core/scene-library";
import {
  readSceneRecents,
  recentSceneCards,
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
import { SceneIsoGlyph } from "./SceneIsoGlyph";

/**
 * Choosing a scene from a grid of its own marks.
 *
 * This is the gesture, with no opinion about what choosing *does*. It shows the
 * library's mark, at the library's proportions, because a name is not enough to
 * recognise a scene by: half this catalog is a room with water in it, and "Dam
 * break", "Twin dam collision" and eight oracles named after figures in a paper
 * are told apart by their *shape*. So the list is a grid of thumbnails and the
 * arrow keys move in two dimensions, while the search box above it still ranks
 * — the reader who knows the name types it and presses Enter, and the reader
 * who does not looks.
 *
 * It is shared rather than copied because a reader who learns this list in the
 * studio has learned it everywhere the product offers a scene. The studio's
 * copy opens into a running pane (`SceneSelector`); the advance lab's reseeds a
 * CPU slice. Neither is this component's business: it reports a card and lets
 * the caller decide, including deciding to refuse, which is why it never closes
 * itself on a choice.
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
function ScenePickerArt({ card }: { card: SceneCard }) {
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

function ScenePickerTile({ card, at, active, current, query, shelf, choose, aim }: {
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
      <ScenePickerArt card={card} />
      <span className="scene-selector-name">
        <strong>{mark
          ? <>{card.name.slice(0, mark[0])}<mark>{card.name.slice(mark[0], mark[1])}</mark>{card.name.slice(mark[1])}</>
          : card.name}</strong>
        {shelf && <small>{card.shelf}</small>}
      </span>
    </button>
  );
}

export interface ScenePickerPopoverProps {
  /** The cards this surface can actually open, in reading order. */
  readonly cards: readonly SceneCard[];
  /** The scene already running here, marked with a dot rather than a word. */
  readonly currentId?: string;
  readonly label: string;
  /** Positioning is the host's: this component brings the glass, not the anchor. */
  readonly className?: string;
  /**
   * Report the chosen card. The popover does not close itself — a caller that
   * refuses a card (a stored document an older schema wrote) needs the list to
   * stay where it is so the reader can pick another.
   */
  readonly choose: (card: SceneCard) => void;
  readonly close: () => void;
  /** An escape hatch to somewhere with more room, drawn as the footer row. */
  readonly browse?: { readonly label: string; readonly run: () => void };
}

export function ScenePickerPopover({
  cards, currentId, label, className, choose, close, browse,
}: ScenePickerPopoverProps) {
  const [query, setQuery] = useState("");
  const [recents, setRecents] = useState<readonly RecentSceneOpen[]>([]);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const keyboardAim = useRef(true);

  // Storage is only readable in the browser, and this component mounts exactly
  // when the popover opens — so mounting is the read, and a scene opened since
  // the last time it was raised heads the grid without any invalidation.
  useEffect(() => { setRecents(readSceneRecents(browserSceneLibraryStorage())); }, []);

  const groups = useMemo(() => sceneSearchGroups(cards, query, {
    recent: recentSceneCards(recents, cards, RECENT_LIMIT),
  }), [cards, query, recents]);
  const order = useMemo(() => sceneSearchOrder(groups), [groups]);
  const rows = useMemo(() => sceneSearchRows(groups, COLUMNS), [groups]);

  // The best answer is always the first tile, so typing re-aims the cursor at
  // it: a highlight left on the ninth tile of the previous query would make
  // Enter open a scene the reader is no longer looking at.
  useEffect(() => { keyboardAim.current = true; setActive(0); }, [query]);

  useEffect(() => {
    // Hover must not move the tile between pointer press and release. Only
    // keyboard/search navigation scrolls the grid to expose its active item.
    if (!keyboardAim.current) return;
    listRef.current
      ?.querySelector<HTMLElement>(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" });
  }, [active, groups]);

  // A press anywhere else puts it down. Registered a frame late so the very
  // press that opened it cannot be the press that closes it, and blind to the
  // toggles themselves so clicking the anchor twice reads as a toggle rather
  // than as a close followed by a re-open.
  useEffect(() => {
    let stop = () => {};
    const frame = requestAnimationFrame(() => {
      const onPointerDown = (event: PointerEvent) => {
        const target = event.target;
        if (!(target instanceof Node)) return;
        if (rootRef.current?.contains(target)) return;
        if (target instanceof Element && target.closest("[data-scene-selector-toggle]")) return;
        close();
      };
      window.addEventListener("pointerdown", onPointerDown, true);
      stop = () => window.removeEventListener("pointerdown", onPointerDown, true);
    });
    return () => { cancelAnimationFrame(frame); stop(); };
  }, [close]);

  const onKeyDown = (event: React.KeyboardEvent) => {
    if (event.key === "Escape") { event.preventDefault(); close(); return; }
    if (GRID_KEYS.includes(event.key as SceneGridKey)) {
      const key = event.key as SceneGridKey;
      // The search box keeps left and right while there is text to move a caret
      // through: the reader is editing a word, not walking a row. Up and down
      // are always the grid's — a search field has nothing to do with them.
      const caret = event.target instanceof HTMLInputElement && event.target.value.length > 0;
      if (caret && (key === "ArrowLeft" || key === "ArrowRight")) return;
      event.preventDefault();
      keyboardAim.current = true;
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
      className={className ? `scene-selector ${className}` : "scene-selector"}
      data-testid="scene-selector"
      role="dialog"
      aria-label={label}
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
                  <ScenePickerTile
                    key={`${group.shelf}/${card.id}`}
                    card={card}
                    at={at}
                    active={at === active}
                    current={card.id === currentId}
                    query={query}
                    // Recent is the one shelf whose cards came from everywhere
                    // else, so it is the one place the shelf still has to be
                    // written on the tile.
                    shelf={group.shelf === SCENE_SEARCH_RECENT_SHELF}
                    choose={choose}
                    aim={(at) => { keyboardAim.current = false; setActive(at); }}
                  />
                );
              })}
            </div>
          </div>
        ))}
        {order.length === 0 && <p className="scene-selector-empty">No scene matches that.</p>}
      </div>
      {browse && <button
        type="button"
        className="scene-selector-browse"
        data-testid="scene-selector-browse"
        onClick={browse.run}
      >{browse.label}</button>}
    </div>
  );
}
