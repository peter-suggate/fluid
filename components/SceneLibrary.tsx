"use client";

// A second composition root: the library is handed to `AppShell` as children
// from a server component, so it is its own client entry and cannot rely on
// the shell's module having been evaluated first. See `AppShell`.
import "../lib/methods";
import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { simulation } from "../lib/core/simulation/controller";
import { currentScenePageUrl } from "../lib/core/url-state";
import {
  allSceneCards,
  matchSceneCards,
  sceneCardGlyph,
  sceneCardPreview,
  sceneSections,
  starterSceneCards,
  type SceneSection,
} from "../lib/core/scene-cards";
import type { SceneCard } from "../lib/core/scene-definition";
import { sceneResume } from "../lib/core/scene-autosave";
import { browserSceneLibraryStorage, readSceneLibrary, type SceneLibraryEntry } from "../lib/core/scene-library";
import { readSceneRecents, recentSceneCards, recordSceneOpen, type RecentSceneOpen } from "../lib/core/scene-recents";
import { planSceneRuntime } from "../lib/core/scene-runtime";
import { sceneLatticeDimensions } from "../lib/core/scene-lattice";
import { useShellStore } from "../lib/core/stores/shell-store";
import { useSceneStore } from "../lib/core/stores/scene-store";
import { SceneIsoGlyph } from "./SceneIsoGlyph";
import { ThemeSwitch } from "./ThemeSwitch";

/**
 * The front door.
 *
 * The scene library owns the front page. A viewed scene is mounted on /scene,
 * giving the WebGPU viewport a route lifecycle independent of browsing and of
 * Fast Refreshes that only touch this page.
 */

function savedLabel(savedAt_ms: number): string {
  const elapsed_s = Math.max(0, (Date.now() - savedAt_ms) / 1000);
  if (elapsed_s < 60) return "just now";
  if (elapsed_s < 3600) return `${Math.floor(elapsed_s / 60)}m ago`;
  if (elapsed_s < 86_400) return `${Math.floor(elapsed_s / 3600)}h ago`;
  return `${Math.floor(elapsed_s / 86_400)}d ago`;
}

/** The mark a card shows, or nothing when its document will not build. */
function SceneCardArt({ card }: { card: SceneCard }) {
  const scene = sceneCardPreview(card);
  const glyph = sceneCardGlyph(card);
  if (!scene || !glyph) return <span className="card-art-missing" aria-hidden="true" />;
  return <SceneIsoGlyph scene={scene} glyph={glyph} />;
}

/**
 * What a card promises, read off the document rather than authored twice.
 *
 * Every one of these is the kind of thing a description gets wrong six months
 * later; deriving them means a card cannot claim water a scene does not have.
 */
function SceneCardChips({ card }: { card: SceneCard }) {
  const scene = sceneCardPreview(card);
  if (!scene) return <span className="chip" data-tone="warn">unreadable</span>;
  const plan = planSceneRuntime(scene);
  const [nx, ny, nz] = sceneLatticeDimensions(scene);
  return <>
    <span className="chip" data-tone={plan.fluidSolver ? "accent" : undefined}>{plan.fluidSolver ? "water" : "no fluid"}</span>
    <span className="chip">{nx}×{ny}×{nz}</span>
    {plan.content.rigidBodyCount > 0
      && <span className="chip">{plan.content.rigidBodyCount} {plan.content.rigidBodyCount === 1 ? "body" : "bodies"}</span>}
    {card.savedAt_ms !== undefined && <span className="chip">{savedLabel(card.savedAt_ms)}</span>}
  </>;
}

function SceneCardButton({ card, active, open }: { card: SceneCard; active: boolean; open: (card: SceneCard) => void }) {
  return (
    <button
      type="button"
      className="card scene-card"
      data-source={card.source}
      data-active={active || undefined}
      data-testid={`scene-card-${card.id}`}
      aria-current={active || undefined}
      onClick={() => open(card)}
    >
      <span className="card-art"><SceneCardArt card={card} /></span>
      <span className="card-body">
        <strong>{card.name}</strong>
        <small>{card.blurb}</small>
        <span className="chip-row"><SceneCardChips card={card} /></span>
      </span>
    </button>
  );
}

/**
 * Creating, as three real rooms rather than one button.
 *
 * The three starters differ only in how big the world is, which is the one
 * decision a starter can honestly make on the author's behalf — so the choice is
 * *drawn*: every room in the row is projected in one frame, and the hall is
 * wider than the room because it is wider than the room.
 *
 * The size ratio is square-rooted rather than taken straight. At true scale the
 * 0.8 m room is a quarter of the 3.2 m hall and stops reading as a room at all;
 * the root keeps the ordering and the proportions exact — a cube stays a cube —
 * while leaving the smallest one legible. Its metres are on the card's title.
 */
function StarterRow({ open }: { open: (card: SceneCard) => void }) {
  const starters = useMemo(() => starterSceneCards().flatMap((card) => {
    const glyph = sceneCardGlyph(card), scene = sceneCardPreview(card);
    return glyph && scene ? [{ card, glyph, scene }] : [];
  }), []);
  const largest_m = Math.max(...starters.map(({ glyph }) => glyph.size_m));
  return (
    <div className="starter-row">
      {starters.map(({ card, glyph, scene }) => (
        <button
          type="button"
          className="starter"
          key={card.id}
          title={card.blurb}
          data-testid={`starter-${card.id}`}
          onClick={() => open(card)}
        >
          <span className="starter-art">
            <SceneIsoGlyph scene={scene} glyph={glyph} scale={Math.sqrt(glyph.size_m / largest_m)} />
          </span>
          <span className="starter-name">{card.name}</span>
        </button>
      ))}
    </div>
  );
}

/**
 * The map and the page agree on anchor names through these two functions and
 * nothing else; an anchor is an attribute the spy queries, not an element id,
 * so it cannot collide with the URL fragment namespace.
 */
function sectionAnchor(id: SceneSection["id"]): string {
  return `section:${id}`;
}

function shelfAnchor(sectionId: SceneSection["id"], shelf: string): string {
  return `shelf:${sectionId}:${shelf.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "")}`;
}

function SceneSectionBlock({ section, activeId, open }: { section: SceneSection; activeId: string; open: (card: SceneCard) => void }) {
  return (
    <section className="scene-section" data-section={section.id} data-map-anchor={sectionAnchor(section.id)}>
      <header className="section-head">
        <h2>{section.label}</h2>
        <small>{section.blurb}</small>
      </header>
      {section.shelves.map((shelf) => (
        <div key={shelf.shelf} className="scene-shelf" data-map-anchor={shelfAnchor(section.id, shelf.shelf)}>
          {/* One shelf per section reads as a stutter, so its label is only
              drawn when the section actually divides into several. */}
          {section.shelves.length > 1 && <h3>{shelf.shelf}</h3>}
          <div className="scene-grid">
            {shelf.cards.map((card) => <SceneCardButton key={card.id} card={card} active={card.id === activeId} open={open} />)}
          </div>
        </div>
      ))}
    </section>
  );
}

/**
 * The map down the side.
 *
 * The page is a single tall scroll — a hero, three or four sections, and in
 * the last one another eight shelves of oracles — and finding a known shelf
 * meant scrolling past everything above it. The rail names every section and
 * shelf with its card count, marks where the reader currently is, and jumps
 * on click.
 *
 * It is navigation for a page the reader already knows, so it is dense and
 * quiet: text, counts, one accent for "you are here". Hidden while searching —
 * results are one flat grid — and on narrow windows, where it would cost the
 * cards their column.
 */
function LibraryRail({ sections, hasRecent, active, jump }: {
  sections: readonly SceneSection[];
  hasRecent: boolean;
  active: string;
  jump: (anchor: string) => void;
}) {
  const sectionActive = (section: SceneSection) =>
    active === sectionAnchor(section.id) || active.startsWith(`shelf:${section.id}:`);
  return (
    <nav className="library-rail" aria-label="Scene map" data-testid="library-rail">
      <ol>
        {hasRecent && (
          <li>
            <button type="button" className="rail-section" data-active={active === "recent" || undefined} onClick={() => jump("recent")}>
              Recent
            </button>
          </li>
        )}
        {sections.map((section) => (
          <li key={section.id}>
            <button
              type="button"
              className="rail-section"
              data-active={sectionActive(section) || undefined}
              aria-current={sectionActive(section) ? "true" : undefined}
              onClick={() => jump(sectionAnchor(section.id))}
            >
              {section.label}
              <span>{section.shelves.reduce((total, shelf) => total + shelf.cards.length, 0)}</span>
            </button>
            {section.shelves.length > 1 && (
              <ol>
                {section.shelves.map((shelf) => {
                  const anchor = shelfAnchor(section.id, shelf.shelf);
                  return (
                    <li key={shelf.shelf}>
                      <button
                        type="button"
                        className="rail-shelf"
                        data-active={active === anchor || undefined}
                        onClick={() => jump(anchor)}
                      >
                        {shelf.shelf}
                        <span>{shelf.cards.length}</span>
                      </button>
                    </li>
                  );
                })}
              </ol>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}

/**
 * The cards opened most recently, as one compact row.
 *
 * Whoever keeps returning to this page is mostly re-opening the same handful
 * of scenes, and those are the cards the sections bury deepest — a saved
 * variant here, an oracle eight shelves down the last section there. The row
 * is their working set; the full cards below remain the browsing surface, so
 * these carry only the mark, the name and the shelf it came from.
 */
function RecentRow({ cards, activeId, open }: { cards: readonly SceneCard[]; activeId: string; open: (card: SceneCard) => void }) {
  return (
    <section className="scene-section recent-strip" data-section="recent" data-map-anchor="recent">
      <header className="section-head">
        <h2>Recent</h2>
        <small>The scenes you opened last, most recent first.</small>
      </header>
      <div className="recent-row">
        {cards.map((card) => (
          <button
            type="button"
            key={card.id}
            className="card recent-card"
            data-source={card.source}
            data-active={card.id === activeId || undefined}
            data-testid={`recent-${card.id}`}
            onClick={() => open(card)}
          >
            <span className="card-art"><SceneCardArt card={card} /></span>
            <span className="card-body">
              <strong>{card.name}</strong>
              <small>{card.shelf}</small>
            </span>
          </button>
        ))}
      </div>
    </section>
  );
}

export function SceneLibrary() {
  const router = useRouter();
  const search = useShellStore((state) => state.librarySearch);
  const setSearch = useShellStore((state) => state.setLibrarySearch);
  const presetId = useSceneStore((state) => state.presetId);
  const [entries, setEntries] = useState<SceneLibraryEntry[]>([]);
  const [recents, setRecents] = useState<RecentSceneOpen[]>([]);
  const searchRef = useRef<HTMLInputElement | null>(null);
  const scrollBodyRef = useRef<HTMLDivElement | null>(null);
  const [activeAnchor, setActiveAnchor] = useState("");

  // Storage is only readable in the browser. Mounting this page again after a
  // studio visit naturally re-reads anything saved there.
  useEffect(() => {
    let active = true;
    queueMicrotask(() => {
      if (!active) return;
      setEntries(readSceneLibrary(browserSceneLibraryStorage()));
      setRecents(readSceneRecents(browserSceneLibraryStorage()));
    });
    return () => { active = false; };
  }, []);

  const open = (card: SceneCard) => {
    if (!simulation.openSceneCard(card)) return;
    recordSceneOpen(browserSceneLibraryStorage(), card.id, Date.now());
    router.push(currentScenePageUrl());
  };

  const sections = useMemo(() => sceneSections(entries), [entries]);
  const results = useMemo(
    () => search.trim() ? matchSceneCards(allSceneCards(entries), search) : undefined,
    [entries, search],
  );
  const resume = useMemo(() => sceneResume(entries), [entries]);
  const recentCards = useMemo(
    () => recentSceneCards(recents, allSceneCards(entries), 6),
    [entries, recents],
  );

  // The spy: the map's "you are here" is the last anchor above a reading line
  // near the top of the viewport. Anchors are queried from the DOM on each
  // pass rather than held in state, so expanding a section or a storage read
  // adding shelves needs no bookkeeping here.
  const syncActiveAnchor = () => {
    const body = scrollBodyRef.current;
    if (!body) return;
    const line = body.getBoundingClientRect().top + 120;
    let current = "";
    for (const anchor of body.querySelectorAll<HTMLElement>("[data-map-anchor]")) {
      if (anchor.getBoundingClientRect().top <= line || current === "") {
        current = anchor.dataset.mapAnchor ?? current;
      }
    }
    setActiveAnchor(current);
  };

  useEffect(() => {
    const body = scrollBodyRef.current;
    if (!body) return;
    let frame = 0;
    const onScroll = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(syncActiveAnchor);
    };
    body.addEventListener("scroll", onScroll, { passive: true });
    syncActiveAnchor();
    return () => {
      body.removeEventListener("scroll", onScroll);
      cancelAnimationFrame(frame);
    };
  }, []);

  // Instant rather than smooth: inside this fixed layer Chrome accepts a
  // smooth scrollIntoView/scrollTo and then moves nothing, so the jump is a
  // jump.
  const jump = (anchor: string) => {
    scrollBodyRef.current
      ?.querySelector<HTMLElement>(`[data-map-anchor="${CSS.escape(anchor)}"]`)
      ?.scrollIntoView({ block: "start" });
  };

  // Searching from deep in the page would otherwise land the reader in the
  // middle of the results grid, past the count that says what they matched.
  const searching = results !== undefined;
  useEffect(() => {
    if (searching) scrollBodyRef.current?.scrollTo({ top: 0 });
  }, [searching]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editing = target !== null && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.key === "Escape") {
        if (editing && search) { setSearch(""); return; }
        return;
      }
      if (event.key === "/" && !editing) { event.preventDefault(); searchRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [search, setSearch]);

  const resumeScene = resume ? sceneCardPreview(resume.card) : undefined;

  return (
    <main className="scene-library-layer" aria-label="Scene library" data-testid="scene-library-layer">
      <header className="library-bar">
        <span className="library-brand">
          <span className="brand-mark" aria-hidden="true">FL</span>
          <span className="library-wordmark">Fluid Lab</span>
        </span>
        <label className="pill library-search">
          <span className="visually-hidden">Search scenes</span>
          <svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="7" cy="7" r="4.6" /><path d="M10.4 10.4 14 14" /></svg>
          <input
            ref={searchRef}
            type="search"
            value={search}
            placeholder="Search scenes"
            aria-label="Search scenes"
            onChange={(event) => setSearch(event.target.value)}
          />
          <kbd aria-hidden="true">/</kbd>
        </label>
        {/*
          The shape lab is a sibling tool rather than a scene: it draws one
          authored shape on the CPU at a chosen leaf, with that shape's own
          parameters on sliders. It belongs in the bar because the alternative to
          finding it is opening a scene and waiting for a GPU frame to judge a
          form the lab draws in a second.
        */}
        <a className="pill" href="/shape-lab">Shape lab</a>
        <ThemeSwitch />
      </header>

      <div className="library-body panel-scroll" ref={scrollBodyRef}>
        <div className="library-layout" data-rail={!results || undefined}>
          {!results && (
            <LibraryRail
              sections={sections}
              hasRecent={recentCards.length > 0}
              active={activeAnchor}
              jump={jump}
            />
          )}
          <div className="library-page">
            <header className="section-head">
              <h1>Scenes</h1>
              <p>Pick one to explore, or start from an empty room.</p>
            </header>

            {results ? (
              <section className="scene-section" data-section="search">
                <header className="section-head">
                  <h2>{results.length} {results.length === 1 ? "match" : "matches"}</h2>
                  <small>Search reaches every shelf, and the starters too.</small>
                </header>
                <div className="scene-shelf">
                  <div className="scene-grid">
                    {results.map((card) => <SceneCardButton key={card.id} card={card} active={card.id === presetId} open={open} />)}
                  </div>
                </div>
                {results.length === 0 && <p className="library-empty">Nothing here matches “{search}”.</p>}
              </section>
            ) : <>
              <div className="library-hero" data-resume={Boolean(resume) || undefined}>
                {resume && resumeScene && (
                  <button
                    type="button"
                    className="card hero-continue"
                    onClick={() => open(resume.card)}
                    data-testid="resume-scene"
                    data-autosaved={resume.autosaved || undefined}
                  >
                    <span className="card-art">
                      <SceneIsoGlyph scene={resumeScene} glyph={sceneCardGlyph(resume.card)} />
                    </span>
                    <span className="card-body">
                      <span className="eyebrow">Continue</span>
                      <strong>{resume.entry.name}</strong>
                      <small>{resume.autosaved
                        ? `Where you left off, edited ${savedLabel(resume.entry.savedAt_ms)}.`
                        : `Saved ${savedLabel(resume.entry.savedAt_ms)} in this browser.`}</small>
                    </span>
                  </button>
                )}
                <section className="card hero-new" aria-labelledby="hero-new-heading">
                  <div className="hero-new-copy">
                    <svg className="hero-new-plus" viewBox="0 0 24 24" aria-hidden="true">
                      <circle cx="12" cy="12" r="11" /><path d="M12 7v10M7 12h10" />
                    </svg>
                    <h2 id="hero-new-heading">New scene</h2>
                    <small>An empty, lit room. Drop in bodies and props straight away; add water when you want it.</small>
                  </div>
                  <StarterRow open={open} />
                </section>
              </div>
              {recentCards.length > 0 && <RecentRow cards={recentCards} activeId={presetId} open={open} />}
              {sections.map((section) => (
                <SceneSectionBlock key={section.id} section={section} activeId={presetId} open={open} />
              ))}
            </>}
          </div>
        </div>
      </div>
    </main>
  );
}
