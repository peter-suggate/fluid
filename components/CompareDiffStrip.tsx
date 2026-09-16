"use client";

import { useId, useMemo, useState } from "react";
import {
  compareDiffRows,
  compareOverlayModeDrawable,
  dropCompareOverride,
  moveCompareOverrideToA,
  shellCompareStore,
} from "../lib/core/compare/compare-model";
import {
  COMPARE_ABSENT,
  COMPARE_GROUP_HINTS,
  COMPARE_GROUP_LABELS,
  COMPARE_LINK_GROUPS,
} from "../lib/core/compare/compare-query";
import { COMPARE_ADOPTIONS, SECOND_PANE_ID } from "../lib/core/compare/compare-mode";
import {
  divergenceRows,
  paneClockReading,
  paneStatsFrom,
} from "../lib/core/compare/divergence";
import { sceneOverridesInQuery } from "../lib/core/scene-overrides";
import { findSceneDefinition } from "../lib/core/scenes";
import { PRIMARY_PANE_ID } from "../lib/core/simulation/pane-clock";
import type { PaneSession } from "../lib/core/session/session";
import { simulation } from "../lib/core/simulation/controller";
import { useShellStore } from "../lib/core/stores/shell-store";

/**
 * What differs, on the seam between the panes.
 *
 * The one permanent addition compare mode makes to the chrome, and it is a
 * *readout* rather than a panel: the diff is already the truth of the mode, so
 * the strip only has to draw it. Each row is one key, pane A's value, pane B's,
 * and the two operations that resolve it — drop the override so B falls back to
 * A, or move it to A so both panes adopt it.
 *
 * Deliberately in the instrument's visual language rather than the panel's:
 * translucent, hairline, sentence case. It sits over the water for as long as
 * the mode is open, and a severe box on that seam would read as a dialog to be
 * dismissed rather than a caption on the scene.
 *
 * Closed is its resting state. The strip hangs over the seam the whole time the
 * mode is open, and the seam is where the two images are actually compared —
 * the readout was standing on the one part of the screen the mode exists to
 * show. Closed it says the one thing that has to be legible without asking
 * (how many keys differ, and any warning that the comparison is not lockstep);
 * opening it is the reader saying they want the figures rather than the water.
 */

/**
 * How much of a value a row shows before it elides.
 *
 * Read against `.compare-values` in `globals.css`: the open strip gives each
 * value ~150px of micro type, which is a little more than this many characters,
 * so the clamp here is the one that fires and it fires on the same glyph count
 * whatever the pane width. The whole value is always on the row's `title`.
 */
const VALUE_LIMIT = 30;

/**
 * The `scene` key, spelled the way the reader chose it.
 *
 * Its value is a preset id — `hero-garden-hose`, `dam-break-384` — and the two
 * panes running *different scenes* is the coarsest and most legible thing this
 * strip can report. Reporting it as a slug would make the one row nobody needs
 * to decode the one row everybody has to. The catalog is the same list the
 * selector offers, so the name here is the name that was clicked; an id this
 * build no longer has falls back to itself rather than to nothing.
 */
function sceneName(presetId: string): string {
  return findSceneDefinition(presetId)?.name ?? presetId;
}

function shortValue(raw: string, key: string): string {
  // An absent key is not an empty value: "B carries no override for this" and
  // "B carries the empty string" are different states of the same row.
  if (raw === COMPARE_ABSENT) return "—";
  if (raw === "") return "none";
  const value = key === "scene" ? sceneName(raw) : raw;
  return value.length > VALUE_LIMIT ? `${value.slice(0, VALUE_LIMIT - 1)}…` : value;
}

/**
 * A human name for a managed key.
 *
 * Borrowed from the overrides chip's own classifier rather than restated, so a
 * key the address bar already knows how to name is named the same way on both
 * surfaces — and a key neither of them has a word for falls back to the key,
 * which is at least the thing a reader would search the URL for.
 */
function keyLabels(diff: Readonly<Record<string, string>>): ReadonlyMap<string, string> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(diff)) {
    if (value === COMPARE_ABSENT) continue;
    query.set(key, value);
  }
  const labels = new Map<string, string>();
  for (const override of sceneOverridesInQuery(query.toString(), { hasPresetBaseline: true })) {
    for (const key of override.keys) labels.set(key, override.label);
  }
  // The scene's own key is an *identity* rather than an override, so the
  // classifier deliberately says nothing about it — see `IDENTITY_KEYS`. The
  // strip still has to name the row, and "scene" is the word for it.
  labels.set("scene", "Scene");
  return labels;
}

/** A chevron, drawn for the same reason the padlock is: glyphs on glass. */
function Chevron({ open }: { open: boolean }) {
  return (
    <svg viewBox="0 0 12 12" width="11" height="11" aria-hidden="true" focusable="false">
      <path
        d={open ? "M2.5 7.5 L6 4 L9.5 7.5" : "M2.5 4.5 L6 8 L9.5 4.5"}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.3"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** A padlock, drawn rather than typed: an emoji lock is a colour glyph on glass. */
function Padlock({ linked }: { linked: boolean }) {
  return (
    <svg viewBox="0 0 12 14" width="11" height="13" aria-hidden="true" focusable="false">
      <path
        d={linked ? "M3.5 6 V4 a2.5 2.5 0 0 1 5 0 V6" : "M3.5 6 V4 a2.5 2.5 0 0 1 5 0"}
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <rect x="1.5" y="6" width="9" height="6.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  );
}

/**
 * The divergence oracle: what the two panes each say about the step they are on.
 *
 * Its own component because it is the only part of the strip that changes every
 * frame. The diff above it changes when someone edits pane B — a few times a
 * session — and re-serializing it sixty times a second to redraw four numbers
 * would be the readout costing more than the thing it reads.
 *
 * Every figure comes out of the two panes' own stores and the host clock; there
 * is no readback here and there must never be one. `gpuInfo` is republished as
 * each pane retires an advance (`webgpu-renderer.ts` `retireGPUAdvances`), so
 * subscribing to it is what paces this component — and reading the clocks during
 * that render is what makes the two `t`s per-pane rather than the host minimum
 * both panes' `runtime.simulationTime` carries.
 */
function CompareDivergence({ a, b, identical }: { a: PaneSession; b: PaneSession; identical: boolean }) {
  const infoA = a.diagnostics((state) => state.gpuInfo);
  const infoB = b.diagnostics((state) => state.gpuInfo);
  const frameMsA = a.diagnostics((state) => state.frameMs);
  const frameMsB = b.diagnostics((state) => state.frameMs);
  // Subscribed for the cadence, not for the value: the published clock is the
  // host *minimum* and identical in both panes, so it cannot answer "how far
  // apart are they" — but it moves on every step either pane completes, which
  // is exactly when the clocks read below are worth reading again.
  void a.runtime((state) => state.simulationTime);
  void b.runtime((state) => state.simulationTime);

  const clocks = simulation.paneClocks();
  const rows = divergenceRows(
    paneStatsFrom(infoA, frameMsA),
    paneStatsFrom(infoB, frameMsB),
    {
      a: paneClockReading(clocks, PRIMARY_PANE_ID),
      b: paneClockReading(clocks, SECOND_PANE_ID),
      dtDiffers: simulation.panesDtDiffer(),
      identical,
    },
  );

  return (
    <table className="compare-stats" data-testid="compare-divergence" aria-label="Per-step figures for each pane">
      <thead>
        <tr>
          <th scope="col"><span className="visually-hidden">Figure</span></th>
          <th scope="col">A</th>
          <th scope="col">B</th>
          <th scope="col"><span aria-hidden="true">Δ</span><span className="visually-hidden">difference</span></th>
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.key} data-row={row.key} data-tone={row.tone} title={row.note}>
            <th scope="row">{row.label}{row.unit && <small>{row.unit}</small>}</th>
            <td>{row.a}</td>
            <td>{row.b}</td>
            <td className="compare-stat-delta">{row.delta ?? ""}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export interface CompareDiffStripProps {
  readonly a: PaneSession;
  readonly b: PaneSession;
}

export function CompareDiffStrip({ a, b }: CompareDiffStripProps) {
  // Closed until asked. Component state rather than the compare record: this is
  // whether *this reader* currently wants the figures, not a fact about the
  // experiment, so it has no business in the link two people share. It lives as
  // long as the mode does, which is the span over which the answer holds.
  const [open, setOpen] = useState(false);
  const bodyId = useId();
  const compare = useShellStore((state) => state.compare);
  const setCompareLink = useShellStore((state) => state.setCompareLink);
  // Every value on every row is derived from both panes, so the strip has to
  // subscribe to both. Whole-store reads rather than selectors for the same
  // reason the overrides chip makes them: the rows are a serialization, and a
  // serialization depends on everything.
  const sceneA = a.scene((state) => state.scene);
  const methodA = a.method();
  const uiA = a.ui();
  const sceneB = b.scene((state) => state.scene);
  const methodB = b.method();
  const uiB = b.ui();

  const store = shellCompareStore();
  const rows = useMemo(
    () => compareDiffRows(compare, a, b),
    [compare, a, b, sceneA, methodA, uiA, sceneB, methodB, uiB]);
  const labels = useMemo(() => keyLabels(compare.diff), [compare.diff]);
  const unlinked = COMPARE_LINK_GROUPS.filter((group) => !compare.links[group]);

  // A field view only one method publishes stays legal in a linked Cut: that
  // pane simply draws nothing. Said out loud, because a blank overlay on one
  // side is otherwise indistinguishable from a solver that produced no field.
  const sharedMode = compare.diff.gridMode === undefined && uiA.gridOverlayAxis !== "off"
    ? uiA.gridOverlayMode : undefined;
  const modeMissingIn = sharedMode === undefined ? undefined
    : !compareOverlayModeDrawable(sharedMode, methodB.methodId) ? "B"
      : !compareOverlayModeDrawable(sharedMode, methodA.methodId) ? "A"
        : undefined;

  return (
    <div
      className="compare-diff-strip"
      data-testid="compare-diff-strip"
      data-open={open ? "true" : "false"}
      role="group"
      aria-label="What differs between the panes"
    >
      <header>
        <strong>A <i aria-hidden="true">→</i> B</strong>
        {/* The padlocks are edits to the experiment, so they belong with the
            rows they change rather than on the closed handle, where a stray
            click would unlink a group nobody was looking at. */}
        {open && <div className="compare-padlocks">
          {COMPARE_LINK_GROUPS.map((group) => (
            <button
              key={group}
              type="button"
              data-linked={compare.links[group]}
              aria-pressed={compare.links[group]}
              title={`${COMPARE_GROUP_LABELS[group]} — ${COMPARE_GROUP_HINTS[group]}`}
              onClick={() => setCompareLink(group, !compare.links[group])}
            >
              <Padlock linked={compare.links[group]} />
              <span>{COMPARE_GROUP_LABELS[group]}</span>
            </button>
          ))}
        </div>}
        {/* The count rides inside the control while the strip is closed, so the
            sentence a reader is deciding on is the thing they click rather than
            a caption beside an 11px chevron. */}
        <button
          type="button"
          className="compare-disclosure"
          data-testid="compare-diff-disclosure"
          aria-expanded={open}
          aria-controls={open ? bodyId : undefined}
          title={open
            ? "Hide the diff — the seam is the part of the screen the mode is for"
            : "Show what differs and the per-step figures for each pane"}
          aria-label={open ? "Hide what differs between the panes" : "Show what differs between the panes"}
          onClick={() => setOpen(!open)}
        >
          {!open && <span className="compare-summary" data-diverged={rows.length > 0}>
            {rows.length === 0 ? "identical" : `${rows.length} differ${rows.length === 1 ? "s" : ""}`}
          </span>}
          <Chevron open={open} />
        </button>
      </header>
      {/* Unmounted rather than hidden while closed. `CompareDivergence` renders
          on every completed step; a closed strip that kept it mounted would pay
          the whole readout's cost to draw nothing. */}
      {open && <div className="compare-diff-body" id={bodyId}>
        {rows.length === 0
          ? <p className="compare-identical">identical — edit B to diverge</p>
          : <ul>
            {rows.map((row) => (
              <li key={row.key} data-group={row.group}>
                <span className="compare-key" title={row.key}>{labels.get(row.key) ?? row.key}</span>
                <span className="compare-values">
                  <em title={row.valueA}>{shortValue(row.valueA, row.key)}</em>
                  <i aria-hidden="true">→</i>
                  <strong title={row.valueB}>{shortValue(row.valueB, row.key)}</strong>
                </span>
                <button
                  type="button"
                  className="compare-op"
                  title="Drop this override — B falls back to A"
                  aria-label={`Drop the ${row.key} override`}
                  onClick={() => dropCompareOverride(store, row.key)}
                >×</button>
                <button
                  type="button"
                  className="compare-op"
                  title="Move this override to A — both panes adopt B's value"
                  aria-label={`Move the ${row.key} override to pane A`}
                  onClick={() => moveCompareOverrideToA(store, a, row.key, COMPARE_ADOPTIONS)}
                >⇄</button>
              </li>
            ))}
          </ul>}
        <CompareDivergence a={a} b={b} identical={rows.length === 0} />
        {modeMissingIn && <p className="compare-note" data-tone="info">
          {sharedMode} — n/a in {modeMissingIn}
        </p>}
      </div>}
      {/* Outside the body on purpose: these two say the comparison is not the
          one the reader thinks they are running. A warning that only appears
          once you open the thing is not a warning. */}
      {unlinked.length > 0 && <p className="compare-note" data-tone="warn">
        {unlinked.map((group) => COMPARE_GROUP_LABELS[group]).join(" · ")} unlinked — edits apply to each pane separately
      </p>}
      {simulation.panesDtDiffer() && <p className="compare-note" data-tone="warn">
        dt differs — not lockstep
      </p>}
    </div>
  );
}
