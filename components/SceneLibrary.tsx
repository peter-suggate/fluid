"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { simulation } from "@/lib/simulation/controller";
import {
  allSceneCards,
  matchSceneCards,
  sceneCardGlyph,
  sceneCardPreview,
  sceneSections,
  starterSceneCards,
  type SceneSection,
} from "@/lib/scene-cards";
import type { SceneCard } from "@/lib/scene-definition";
import { sceneResume } from "@/lib/scene-autosave";
import { browserSceneLibraryStorage, readSceneLibrary, type SceneLibraryEntry } from "@/lib/scene-library";
import { planSceneRuntime } from "@/lib/scene-runtime";
import { sceneLatticeDimensions } from "@/lib/scene-lattice";
import { useShellStore } from "@/lib/stores/shell-store";
import { useSceneStore } from "@/lib/stores/scene-store";
import { SceneIsoGlyph } from "./SceneIsoGlyph";
import { ThemeSwitch } from "./ThemeSwitch";

/**
 * The front door.
 *
 * A layer over a live viewport rather than a route, so the WebGPU device and
 * every compiled pipeline survive being here — see `lib/stores/shell-store.ts`.
 * The consequence worth naming: a cold load spends its first second or two in
 * the platform lane whatever we show, so showing this instead of a progress bar
 * on an empty canvas turns the wait into browsing, and by the time a scene is
 * chosen the device is usually already up.
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

function SceneCardButton({ card, active }: { card: SceneCard; active: boolean }) {
  return (
    <button
      type="button"
      className="card scene-card"
      data-source={card.source}
      data-active={active || undefined}
      data-testid={`scene-card-${card.id}`}
      aria-current={active || undefined}
      onClick={() => simulation.openSceneCard(card)}
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
function StarterRow() {
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
          onClick={() => simulation.openSceneCard(card)}
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

function SceneSectionBlock({ section, activeId }: { section: SceneSection; activeId: string }) {
  const expandedSections = useShellStore((state) => state.expandedSections);
  const toggleSection = useShellStore((state) => state.toggleSection);
  const expanded = section.disclosed ? expandedSections.includes(section.id) : true;
  const count = section.shelves.reduce((total, shelf) => total + shelf.cards.length, 0);
  return (
    <section className="scene-section" data-section={section.id} data-expanded={expanded}>
      {section.disclosed ? (
        <button
          type="button"
          className="scene-section-disclosure"
          aria-expanded={expanded}
          onClick={() => toggleSection(section.id)}
        >
          <i aria-hidden="true">{expanded ? "−" : "+"}</i>
          <strong>{section.label}</strong>
          <span>{count}</span>
          <small>{section.blurb}</small>
        </button>
      ) : (
        <header className="section-head">
          <h2>{section.label}</h2>
          <small>{section.blurb}</small>
        </header>
      )}
      {expanded && section.shelves.map((shelf) => (
        <div key={shelf.shelf} className="scene-shelf">
          {/* One shelf per section reads as a stutter, so its label is only
              drawn when the section actually divides into several. */}
          {section.shelves.length > 1 && <h3>{shelf.shelf}</h3>}
          <div className="scene-grid">
            {shelf.cards.map((card) => <SceneCardButton key={card.id} card={card} active={card.id === activeId} />)}
          </div>
        </div>
      ))}
    </section>
  );
}

export function SceneLibrary() {
  const view = useShellStore((state) => state.view);
  const studioEntered = useShellStore((state) => state.studioEntered);
  const search = useShellStore((state) => state.librarySearch);
  const setSearch = useShellStore((state) => state.setLibrarySearch);
  const enterStudio = useShellStore((state) => state.enterStudio);
  const presetId = useSceneStore((state) => state.presetId);
  const [entries, setEntries] = useState<SceneLibraryEntry[]>([]);
  const searchRef = useRef<HTMLInputElement | null>(null);

  // Storage is only readable in the browser, and a scene saved in the studio
  // must appear on the way back, so the list is re-read each time this opens.
  useEffect(() => {
    if (view === "library") setEntries(readSceneLibrary(browserSceneLibraryStorage()));
  }, [view]);

  const sections = useMemo(() => sceneSections(entries), [entries]);
  const results = useMemo(
    () => search.trim() ? matchSceneCards(allSceneCards(entries), search) : undefined,
    [entries, search],
  );
  const resume = useMemo(() => sceneResume(entries), [entries]);

  useEffect(() => {
    if (view !== "library") return;
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const editing = target !== null && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
      if (event.key === "Escape") {
        // Nothing to go back to on a cold load: a viewport showing a scene
        // nobody chose is not a place to be returned to.
        if (editing && search) { setSearch(""); return; }
        if (studioEntered) { event.preventDefault(); enterStudio(); }
        return;
      }
      if (event.key === "/" && !editing) { event.preventDefault(); searchRef.current?.focus(); }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [view, studioEntered, enterStudio, search, setSearch]);

  if (view !== "library") return null;

  const resumeScene = resume ? sceneCardPreview(resume.card) : undefined;

  return (
    <div className="scene-library-layer" role="dialog" aria-label="Scene library" data-testid="scene-library-layer">
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
        <ThemeSwitch />
        {studioEntered && (
          <button type="button" className="pill" onClick={enterStudio} data-testid="close-scene-library">
            Back to scene
          </button>
        )}
      </header>

      <div className="library-body panel-scroll">
        <div className="library-page">
          <header className="section-head">
            <h1>Scenes</h1>
            <p>Pick one to explore, or start from an empty room.</p>
          </header>

          {results ? (
            <section className="scene-section" data-section="search">
              <header className="section-head">
                <h2>{results.length} {results.length === 1 ? "match" : "matches"}</h2>
                <small>Search reaches every shelf, including the disclosed ones.</small>
              </header>
              <div className="scene-shelf">
                <div className="scene-grid">
                  {results.map((card) => <SceneCardButton key={card.id} card={card} active={card.id === presetId} />)}
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
                  onClick={() => simulation.openSceneCard(resume.card)}
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
                <StarterRow />
              </section>
            </div>
            {sections.map((section) => (
              <SceneSectionBlock key={section.id} section={section} activeId={presetId} />
            ))}
          </>}
        </div>
      </div>
    </div>
  );
}
