import "../../../methods";
import assert from "node:assert/strict";
import test from "node:test";
import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  SLICE_OVERLAY_VIEWS,
} from "../../../../advance-lab/lenses";
import {
  FRACTION_VIEW_BANDS, fractionBand, fractionBandPaint, fractionReadout,
} from "../../../core/fluid-fraction-view";
import { gridOverlayVisualizations } from "../../../core/grid-overlay-visualizations";
import { createPaneSession } from "../../../core/session/session";
import { SessionProvider } from "../../../core/session/session-context";
import { DEFAULT_GRID_OVERLAY_AXIS } from "../../../core/stores/ui-store";
import type { FieldVisualization } from "../../../core/visualization-registry";
import { VISUALIZATION_FIELDS } from "../../../core/visualization-catalog";
import {
  FieldOverlayRows, FieldViewRows, type FieldOverlayState, type FieldViewState,
} from "../ui";

/**
 * One field row, both hosts — and one definition of what it draws.
 *
 * *"visuals such as the fraction view should be defined in one place and work
 * for both the 2d and 3d."* Before WP8 there were two rows: the studio's
 * `FieldQuickBar` over the visualization registry, and the lab's own `LensRow`
 * and `OverlayRow` over a page-local list, each restating the chevron, the
 * mark, the swatch and the tip. There is one row now, and the three things a
 * host supplies are the entries, the answers, and whether the view is drawn
 * *over* something (so that clicking the lit mark puts it away).
 *
 * So the pins are:
 *
 *   1. the same entries and the same answers render byte-identically whether
 *      the answers came out of the studio's ui store or out of the lab's
 *      `EditorHost.params` — nothing about *which host* reaches the markup;
 *   2. the lab's own declarations go through that row, and its switches are one
 *      per declared entry, lit and offered by the host's reading;
 *   3. the fraction view's bands, readout and legend are
 *      `lib/core/fluid-fraction-view.ts` in both renderers, so the 2-D lab and
 *      the 3-D slice cannot disagree about what a colour means.
 */

function draw(node: ReactNode): string {
  return renderToStaticMarkup(
    <SessionProvider value={createPaneSession("a")}>{node}</SessionProvider>);
}

/** A roster wide enough to exercise the split: marks on the strip, the rest behind the chevron. */
const VIEWS = VISUALIZATION_FIELDS as readonly FieldVisualization[];

test("the same entries and answers render the same row from either host's state", () => {
  const mode = VIEWS.find((view) => view.icon !== undefined && !view.hidden)!.mode;

  // The studio's answers: six ui-store members, as `components/FieldQuickBar.tsx`
  // passes them.
  const session = createPaneSession("b");
  session.ui.getState().setGridOverlayMode(mode as never);
  session.ui.getState().setGridOverlayAxis("y");
  session.ui.getState().setGridOverlaySlice(0.375);
  const ui = session.ui.getState();
  const studio: FieldViewState = {
    mode: ui.gridOverlayMode,
    axis: ui.gridOverlayAxis,
    slice: ui.gridOverlaySlice,
    setMode: ui.setGridOverlayMode as (mode: string) => void,
    setAxis: ui.setGridOverlayAxis,
    setSlice: ui.setGridOverlaySlice,
    defaultAxis: DEFAULT_GRID_OVERLAY_AXIS,
  };

  // The lab's: one `EditorHost.params` record, keyed by the settings a feature
  // control declares. Same answers, arrived at the other way.
  const params: Record<string, string | number> = {
    advanceLens: mode, advanceAxis: "y", advanceSlice: 0.375,
  };
  const lab: FieldViewState = {
    mode: String(params.advanceLens),
    axis: params.advanceAxis as FieldViewState["axis"],
    slice: Number(params.advanceSlice),
    setMode: (next) => { params.advanceLens = next; },
    setAxis: (next) => { params.advanceAxis = next; },
    setSlice: (next) => { params.advanceSlice = next; },
    defaultAxis: DEFAULT_GRID_OVERLAY_AXIS,
  };

  assert.equal(
    draw(<FieldViewRows views={VIEWS} state={lab} volumeCapable />),
    draw(<FieldViewRows views={VIEWS} state={studio} volumeCapable />));
  // And the row really did draw something: the lit mark is the entry the answer
  // names, keyed by the registry's own mode.
  assert.ok(draw(<FieldViewRows views={VIEWS} state={studio} volumeCapable />)
    .includes(`data-testid="field-quick-${mode}"`));
});

test("the lab's declarations render through the same row and the same switches", () => {
  const lens = SLICE_OVERLAY_VIEWS[0]!;
  const rows = draw(<FieldViewRows
    views={SLICE_OVERLAY_VIEWS}
    volumeCapable={false}
    state={{
      mode: lens.mode, axis: "z", slice: 0,
      setMode: () => {}, setAxis: () => {}, setSlice: () => {},
      // The lab's one declared difference: a 2-D cut *is* the picture, so there
      // is no lit mark to put away and no plane to choose.
      dismissable: false, adjustable: false,
    }}
  />);
  assert.ok(rows.includes(`data-testid="field-quick-${lens.mode}"`), "the mark is the entry");
  assert.ok(rows.includes(lens.label), "and its name is the declared label");
  assert.ok(!rows.includes("field-plane"), "no plane chooser where there is one plane");

  // The switches are one per declared entry, lit by the host's reading, and a
  // host that cannot draw one says so rather than hiding it.
  const on = new Set([SLICE_OVERLAY_VIEWS[0]!.mode]);
  const state: FieldOverlayState = {
    enabled: (mode) => on.has(mode),
    offered: (mode) => mode !== SLICE_OVERLAY_VIEWS[1]!.mode,
    unavailable: () => "not under this transport",
    toggle: () => {},
  };
  const switches = draw(<FieldOverlayRows views={SLICE_OVERLAY_VIEWS} state={state} />);
  for (const view of SLICE_OVERLAY_VIEWS) {
    assert.ok(switches.includes(`data-testid="field-overlay-${view.mode}"`),
      `${view.mode} has a switch`);
    assert.ok(switches.includes(view.label), `${view.mode} is named by its declaration`);
  }
  assert.ok(switches.includes("not under this transport"),
    "and the one that cannot be drawn says why");
});

test("the fraction view's bands and legend come from fluid-fraction-view in both renderers", () => {
  const studio = gridOverlayVisualizations
    .find((view) => view.kind === "field" && view.mode === "volume-levelset") as FieldVisualization;
  const labFraction = SLICE_OVERLAY_VIEWS.find((view) => view.mode === "fraction")!;

  for (const view of [studio, labFraction]) {
    assert.equal(view.scalar?.band, fractionBand, "one band test");
    assert.equal(view.scalar?.format, fractionReadout, "one readout");
    assert.equal(view.scalar?.bands, FRACTION_VIEW_BANDS, "one set of bands");
    assert.equal(view.swatch, fractionBandPaint("liquid").swatch, "one chip colour");
  }

  // The lab's legend *is* the band table, in its order, with the declared
  // labels and colours — not a transcription of them.
  assert.deepEqual(
    labFraction.legend?.map((entry) => [entry.swatch, entry.label]),
    FRACTION_VIEW_BANDS.map((paint) => [paint.swatch, paint.label]));
  // The studio's adds two authored lines of its own — the level-set contour and
  // the adaptive lattice, which nothing else draws — and takes its band lines
  // from the same table.
  const bandLabels = new Set(FRACTION_VIEW_BANDS.map((paint) => paint.label));
  assert.ok(studio.legend?.some((entry) => bandLabels.has(entry.label)),
    "the studio's legend reads the same band table");
});
