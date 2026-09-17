import "../lib/methods";
import assert from "node:assert/strict";
import test from "node:test";
import { createElement, type ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  EntityDeleteRow,
  EntityOptionRows,
  entityCommitLabel,
  patchCommitter,
} from "../components/EntityOptions";
import type { EditorEntity } from "../lib/core/editor-entity";
import type { EditorCommitOptions, EditorHost } from "../lib/core/editor-host";
import { EditorHostProvider } from "../lib/core/session/host-context";

/**
 * One entity definition, two hosts, one row tree.
 *
 * `EntityOptionRows` was the headline duplication: the advance lab hand-wrote
 * its region strip because this component reached the `simulation` singleton
 * and could not be rendered anywhere else. The seam is only worth anything if
 * the studio's markup is *unchanged* by it and the lab's is *the same markup*,
 * so both halves are pinned here — the first against a capture taken before the
 * component was touched, the second against the first.
 *
 * `ROWS_BEFORE` and `DELETE_BEFORE` were rendered from the pre-change component
 * against the entity below. They are the regression, not decoration: any change
 * to a tag, a testId, a row order or a readout's precision fails here.
 */
const ROWS_BEFORE = "<div class=\"toolstrip-row\"><button type=\"button\" class=\"toolstrip-key\" aria-pressed=\"false\" data-testid=\"entity-option-rule\"><b class=\"toolstrip-tag\">Means</b><span class=\"toolstrip-value\">Bounds</span><span class=\"toolstrip-tip\"><strong>This box means</strong><small>the hint</small></span></button></div><div class=\"toolstrip-row\"><button type=\"button\" class=\"toolstrip-key\" aria-pressed=\"false\" data-testid=\"entity-option-minimumCellSize\"><b class=\"toolstrip-tag\">Min</b><span class=\"toolstrip-value\">8 cubed</span><span class=\"toolstrip-tip\"><strong>Smallest cell</strong><small>8 cells</small></span></button></div><div class=\"toolstrip-row\"><button type=\"button\" class=\"toolstrip-key\" aria-pressed=\"false\" data-testid=\"entity-option-at\"><b class=\"toolstrip-tag\">AT</b><span class=\"toolstrip-value\">0.50 0.25</span><span class=\"toolstrip-tip\"><strong>Position</strong><small>metres</small></span></button></div><div class=\"toolstrip-row\"><button type=\"button\" class=\"toolstrip-key\" aria-pressed=\"false\" data-testid=\"entity-option-radius\"><b class=\"toolstrip-tag\">R</b><span class=\"toolstrip-value\">0.20 m</span><span class=\"toolstrip-tip\"><strong>Radius</strong></span></button></div><div class=\"toolstrip-row\"><button type=\"button\" class=\"toolstrip-key\" aria-pressed=\"false\" data-testid=\"entity-group-place\"><b class=\"toolstrip-tag\">Place</b><span class=\"toolstrip-value\">1 settings</span><span class=\"toolstrip-tip\"><strong>Place</strong><small>coords</small></span></button></div>";
const DELETE_BEFORE = "<div class=\"toolstrip-row is-verb\"><button type=\"button\" class=\"toolstrip-key is-verb tone-danger\" data-testid=\"entity-delete\"><svg xmlns=\"http://www.w3.org/2000/svg\" width=\"14\" height=\"14\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"1.7\" stroke-linecap=\"round\" stroke-linejoin=\"round\" class=\"lucide lucide-trash2 lucide-trash-2\" aria-hidden=\"true\"><path d=\"M10 11v6\"></path><path d=\"M14 11v6\"></path><path d=\"M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6\"></path><path d=\"M3 6h18\"></path><path d=\"M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2\"></path></svg><b class=\"toolstrip-tag\">Delete</b><span class=\"toolstrip-tip\"><strong>Delete REGION-1</strong><small>Takes it out of the document and re-seeds the run. Undoable, and the Delete key does the same.</small></span></button></div>";

interface Commit { readonly kind: "commit" | "commitPatch"; readonly label: string;
  readonly payload: unknown; readonly options?: EditorCommitOptions }

/** The studio's shape: a document and a merge patch over it. */
function studioShapedHost(log: Commit[]): EditorHost<{ doc: true }, { patch: true }> {
  return {
    id: "studio",
    commit: (label, next, options) => log.push({ kind: "commit", label, payload: next, options }),
    commitPatch: (label, patch, options) => log.push({ kind: "commitPatch", label, payload: patch, options }),
    select: () => {}, arm: () => {}, notice: () => {},
  };
}

/**
 * The lab's shape: no merge to express, so `Patch = Doc` and `commitPatch` is
 * simply absent. A row that only knows how to describe a patch must still land
 * on it.
 */
function labShapedHost(log: Commit[]): EditorHost<{ regions: readonly string[] }, { regions: readonly string[] }> {
  return {
    id: "lab",
    commit: (label, next, options) => log.push({ kind: "commit", label, payload: next, options }),
    select: () => {}, arm: () => {}, notice: () => {},
  };
}

/**
 * The same declaration both hosts are handed: two choice groups, a folded
 * position row, a bounded scrub and one folded group — every branch the
 * component has.
 */
function fabricatedEntity<Patch, Doc>(patch: () => Patch, doc: () => Doc): EditorEntity<Patch, Doc> {
  return {
    selection: { kind: "refinement-region", id: "refinement-region-region-1" },
    label: "REGION-1",
    tone: "region",
    frame: { origin_m: { x: 0, y: 0, z: 0 }, basis: [
      { x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }, { x: 0, y: 0, z: 1 }] },
    box: { min: { x: 0, y: 0, z: 0 }, max: { x: 1, y: 1, z: 1 } },
    sizeLabel: "1.00 x 1.00 x 1.00 m",
    handles: [],
    draftSubject: "refinement-region",
    editLabel: () => "Moved",
    choices: [
      { id: "rule", label: "This box means", tag: "Means", value: "minimum-cell-size",
        options: [{ id: "minimum-cell-size", label: "Bounds", hint: "the hint", enabled: true, apply: patch }] },
      { id: "minimumCellSize", label: "Smallest cell", tag: "Min", value: "8",
        options: [1, 2, 4, 8].map((n) => ({
          id: String(n), label: `${n} cubed`, hint: `${n} cells`, enabled: true, apply: patch })) },
    ],
    fields: [
      { id: "at:x", label: "X", tag: "AT", value: 0.5, step: 0.01, unit: "m",
        row: { id: "at", tag: "AT", label: "Position", hint: "metres" }, apply: patch },
      { id: "at:y", label: "Y", value: 0.25, step: 0.01, unit: "m",
        row: { id: "at", tag: "AT", label: "Position", hint: "metres" }, apply: patch },
      { id: "radius", label: "Radius", tag: "R", value: 0.2, step: 0.01, min: 0.01, max: 1, unit: "m", apply: patch },
    ],
    groups: [
      { id: "place", label: "Place", hint: "coords",
        fields: [{ id: "px", label: "X", value: 0, step: 0.01, apply: patch }], summary: "a summary" },
    ],
    summary: "what it means",
    remove: doc,
  } as unknown as EditorEntity<Patch, Doc>;
}

function draw<Doc, Patch>(host: EditorHost<Doc, Patch>, children: ReactNode): string {
  return renderToStaticMarkup(
    <EditorHostProvider<Doc, Patch> value={host}>{children}</EditorHostProvider>);
}

test("the studio's option rows are byte-identical through the host", () => {
  const entity = fabricatedEntity(() => ({ patch: true as const }), () => ({ doc: true as const }));
  const markup = draw(studioShapedHost([]), createElement(EntityOptionRows, { entity }));
  assert.equal(markup, ROWS_BEFORE);
});

test("the studio's delete row is byte-identical through the host", () => {
  const entity = fabricatedEntity(() => ({ patch: true as const }), () => ({ doc: true as const }));
  const markup = draw(studioShapedHost([]), createElement(EntityDeleteRow, { entity }));
  assert.equal(markup, DELETE_BEFORE);
});

test("both hosts render the same rows for one definition", () => {
  const studio = fabricatedEntity(() => ({ patch: true as const }), () => ({ doc: true as const }));
  const lab = fabricatedEntity(
    () => ({ regions: ["region-1"] as readonly string[] }),
    () => ({ regions: [] as readonly string[] }));
  assert.equal(
    draw(labShapedHost([]), createElement(EntityOptionRows, { entity: lab })),
    draw(studioShapedHost([]), createElement(EntityOptionRows, { entity: studio })));
  assert.equal(
    draw(labShapedHost([]), createElement(EntityDeleteRow, { entity: lab })),
    draw(studioShapedHost([]), createElement(EntityDeleteRow, { entity: studio })));
});

test("rows with no host mounted refuse rather than committing into pane A", () => {
  const entity = fabricatedEntity(() => ({ patch: true as const }), () => ({ doc: true as const }));
  assert.throws(() => renderToStaticMarkup(createElement(EntityOptionRows, { entity })),
    /EditorHostProvider/);
  assert.throws(() => renderToStaticMarkup(createElement(EntityDeleteRow, { entity })),
    /EditorHostProvider/);
});

test("a patch reaches commitPatch where there is one, and commit where there is not", () => {
  const studioLog: Commit[] = [];
  patchCommitter(studioShapedHost(studioLog))(
    entityCommitLabel("REGION-1", "Smallest cell"), { patch: true }, { reseed: true });
  assert.deepEqual(studioLog, [{
    kind: "commitPatch", label: "Set REGION-1 Smallest cell",
    payload: { patch: true }, options: { reseed: true },
  }]);

  const labLog: Commit[] = [];
  patchCommitter(labShapedHost(labLog))(
    entityCommitLabel("REGION-1", "Smallest cell"), { regions: ["region-1"] }, { reseed: true });
  assert.deepEqual(labLog, [{
    kind: "commit", label: "Set REGION-1 Smallest cell",
    payload: { regions: ["region-1"] }, options: { reseed: true },
  }]);
});
