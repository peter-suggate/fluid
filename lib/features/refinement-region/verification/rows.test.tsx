import "../../../methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  labRegionSpace, type LabRegionDocument,
} from "../../../../advance-lab/lab-region-space";
import {
  refinementRegionFromDrag, studioRegionSpace,
} from "../../../core/editor-refinement-region";
import type { EditorHost } from "../../../core/editor-host";
import { cloneScene, defaultScene, type SceneDescription } from "../../../core/model";
import { DEFAULT_REGION_DRAFT, type RegionDraft } from "../../../core/refinement-regions";
import { EditorHostProvider } from "../../../core/session/host-context";
import { createPaneSession } from "../../../core/session/session";
import { SessionProvider } from "../../../core/session/session-context";
import type { RefinementRegionRecord, RegionSpace } from "../definition";
import { regionDraftCellSize, regionFromDraw, snapRegionBox } from "../policy";
import { RegionOptionRows, RegionRow } from "../ui";

/**
 * One capability, two hosts, one set of rows.
 *
 * This is the assertion the whole plugin exercise is for, and the question that
 * started it — *"i like the UI you have with the enforcement region in the 2d
 * case. a dropdown list allows specifying cell size. why did this not get
 * applied to 3d?"* It was not applied because there were two rows: one in
 * `components/MakeRows.tsx` reading a scene document and one in
 * `advance-lab/SliceToolstrip.tsx` reading two pieces of page state. There is
 * one now, and the only thing either host supplies is a `RegionSpace`.
 *
 * So the pins below are all of the form "handed two different spaces, the same
 * component produces the same thing":
 *
 *   1. the making row, including its cell-size chooser;
 *   2. a selected box's option rows, up to the one difference the interface
 *      actually declares — `RegionSpace.cellEdge_mm`, which only the studio has;
 *   3. the draft a reader sets in that chooser reaching a drawn box, in both
 *      hosts, through each host's own release path.
 */

/** A record both ladders can carry: `2` is on the studio's six and the lab's four. */
const RECORD: RefinementRegionRecord = Object.freeze({
  id: "region-1",
  rule: "minimum-cell-size" as const,
  minimumCellSize_cells: 2,
  min_cells: [8, 8, 8],
  max_cells: [24, 24, 24],
});

const LAB_DOC: LabRegionDocument = { regions: [], nx: 256, ny: 128 };

function scene(): SceneDescription {
  return cloneScene(defaultScene);
}

/** A host that records rather than writes: what a row commits is not the subject here. */
function recordingHost<Doc, Patch>(id: string): EditorHost<Doc, Patch> {
  return {
    id,
    commit: () => {},
    select: () => {},
    arm: () => {},
    notice: () => {},
  };
}

function draw<Doc, Patch>(children: ReactNode): string {
  const session = createPaneSession("a");
  return renderToStaticMarkup(
    <SessionProvider value={session}>
      <EditorHostProvider<Doc, Patch> value={recordingHost<Doc, Patch>("test")}>
        {children}
      </EditorHostProvider>
    </SessionProvider>);
}

test("both hosts render the same making row, chooser and all", () => {
  const studio = draw(createElement(RegionRow, { space: studioRegionSpace, doc: scene() }));
  const lab = draw(createElement(RegionRow, { space: labRegionSpace, doc: LAB_DOC }));
  assert.equal(lab, studio);
  // Not vacuous: the row is the one that carries the chooser, which is the
  // control the 3-D strip did not have before this package existed.
  assert.ok(studio.includes("region-draft"), "the making row carries the cell-size chooser");
  assert.ok(studio.includes("scene-region-row"), "and is the making row itself");
});

test("a full document disables the row on either host, with its own count", () => {
  const full = { ...LAB_DOC, regions: Array.from({ length: labRegionSpace.capacity },
    (_value, index) => ({
      id: `advance-region-${index}`,
      minimumFine: [0, 0] as [number, number],
      maximumFine: [8, 8] as [number, number],
      minimumCellWidth: 2,
    })) };
  const markup = draw(createElement(RegionRow, { space: labRegionSpace, doc: full }));
  assert.ok(markup.includes("disabled"), "a full tail disables the stroke rather than hiding it");
  assert.ok(markup.includes(`All ${labRegionSpace.capacity} refinement boxes are drawn`),
    "and says why before the click rather than after it");
});

/**
 * The one declared difference between the two hosts' option rows.
 *
 * `RegionSpace.cellEdge_mm` is optional precisely so the studio's hints can say
 * "8³ finest cells · 40 mm edge" while the lab's, having no metre scale to
 * quote, say "8³ finest cells" and not a fabricated number. Dropping that one
 * member is therefore the whole of what separates the two row trees — which is
 * a stronger statement than stripping the clause out of the markup afterwards
 * would be, because it is made by construction.
 */
test("both hosts render the same option rows for one record", () => {
  const scaleless: RegionSpace<SceneDescription, Partial<SceneDescription>> =
    { ...studioRegionSpace, cellEdge_mm: undefined };
  const studio = draw(createElement(RegionOptionRows,
    { space: scaleless, doc: scene(), record: RECORD }));
  const lab = draw(createElement(RegionOptionRows,
    { space: labRegionSpace, doc: LAB_DOC, record: RECORD }));
  assert.equal(lab, studio);
  assert.ok(studio.includes("entity-option-minimumCellSize")
    && studio.includes("entity-option-maximumCellSize"),
  "the floor and the ceiling are both rows, not a hand-written pair per host");

  // And the metre scale really is the only thing the studio adds.
  const withScale = draw(createElement(RegionOptionRows,
    { space: studioRegionSpace, doc: scene(), record: RECORD }));
  assert.ok(withScale.includes("mm edge"), "the studio quotes millimetres");
  assert.ok(!lab.includes("mm edge"), "the lab has no metre scale to quote");
});

test("a chosen rung resolves onto each host's own ladder", () => {
  // Nobody has chosen: each host's own default stands, and they differ.
  assert.equal(regionDraftCellSize(studioRegionSpace, DEFAULT_REGION_DRAFT),
    studioRegionSpace.defaultCellSize_cells);
  assert.equal(regionDraftCellSize(labRegionSpace, DEFAULT_REGION_DRAFT),
    labRegionSpace.defaultCellSize_cells);
  assert.notEqual(studioRegionSpace.defaultCellSize_cells, labRegionSpace.defaultCellSize_cells);

  // A rung both ladders have is that rung on both.
  for (const space of [studioRegionSpace, labRegionSpace]) {
    assert.equal(regionDraftCellSize(space, { ...DEFAULT_REGION_DRAFT, cellSize_cells: 4 }), 4);
  }
  // One only the studio has rounds down onto the lab's, rather than handing it
  // a bound it has no leaf for or showing an empty menu.
  const coarse: RegionDraft = { ...DEFAULT_REGION_DRAFT, cellSize_cells: 32 };
  assert.equal(regionDraftCellSize(studioRegionSpace, coarse), 32);
  assert.equal(regionDraftCellSize(labRegionSpace, coarse), 8);
});

test("the draft reaches a box drawn on either host", () => {
  const draft: RegionDraft = { cellSize_cells: 4, rule: "minimum-cell-size", holdAtOneTier: true };

  const drawn = refinementRegionFromDrag(scene(),
    { x: 0.2, y: 0, z: 0.2 }, { x: 0.8, y: 0, z: 0.8 }, { id: "region-1", draft })!;
  assert.equal(drawn.minimumCellSize_cells, 4, "the studio's release reads the draft");
  assert.equal(drawn.maximumCellSize_cells, 4, "and holds it at one tier when asked");

  const record = regionFromDraw(labRegionSpace, LAB_DOC, [10, 10], [70, 50], draft,
    { id: "region-1" })!;
  assert.equal(record.minimumCellSize_cells, 4);
  assert.equal(record.maximumCellSize_cells, 4);

  // Floor only: neither host invents a ceiling.
  const floorOnly: RegionDraft = { ...draft, holdAtOneTier: false };
  assert.equal(refinementRegionFromDrag(scene(),
    { x: 0.2, y: 0, z: 0.2 }, { x: 0.8, y: 0, z: 0.8 },
    { id: "region-1", draft: floorOnly })!.maximumCellSize_cells, undefined);
  assert.equal(regionFromDraw(labRegionSpace, LAB_DOC, [10, 10], [70, 50], floorOnly,
    { id: "region-1" })!.maximumCellSize_cells, undefined);
});

/**
 * A click is a selection, never a draw — on either host.
 *
 * Wave 2 saw a press on the lab commit a one-brick region, and the studio did
 * the same: its press seeds a draft before the pointer has travelled at all.
 * Neither was a transcription bug. `snapRegionBox` guarantees a step of
 * thickness on every axis — a box thinner than the cell it asks for contains
 * none of them — and that guarantee is exactly what turns "no drag" into "one
 * brick". So the two questions are separated: the snap still thickens a box
 * somebody drew, and `regionFromDraw` refuses to name one nobody drew.
 *
 * The third assertion is the one that says why the rule cannot live in the
 * snap: it shows the snap still answering a step-thick box for the degenerate
 * corners, which is the behaviour every drawn box depends on.
 */
test("a press with no drag commits nothing on either host", () => {
  const press = [40, 24];
  assert.equal(
    regionFromDraw(labRegionSpace, LAB_DOC, press, press, DEFAULT_REGION_DRAFT),
    undefined, "the lab's release names no box for a press it did not move");
  // One cell along one axis is still an instruction: a wide thin region, which
  // the snap then makes one step deep. Every axis, not any.
  assert.ok(regionFromDraw(labRegionSpace, LAB_DOC, press, [press[0]! + 1, press[1]!],
    DEFAULT_REGION_DRAFT) !== undefined, "a drag along one axis is a draw");

  // The studio asks the same question before it seeds the height, because the
  // seed is what would hide it: the box handed to the shared rule is already a
  // step tall over a zero footprint.
  const at = { x: -0.02, y: 0.041, z: 0.03 };
  assert.equal(refinementRegionFromDrag(scene(), at, at, {}), undefined,
    "the studio's release names no box for a press it did not move");
  assert.ok(refinementRegionFromDrag(scene(), at, { ...at, x: at.x + 0.08, z: at.z + 0.08 }, {})
    !== undefined, "a travelled drag still draws");

  // And the minimum thickness the click was riding on is untouched.
  const kept = snapRegionBox(press, press, 8, labRegionSpace.lattice(LAB_DOC));
  assert.deepEqual(kept.max.map((high, axis) => high - kept.min[axis]!), [8, 8],
    "the snap still thickens a box that was drawn");
});
