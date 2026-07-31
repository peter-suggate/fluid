import assert from "node:assert/strict";
import test from "node:test";
import { fineBandCellVisualizations } from "../lib/fine-band-cell-visualizations";
import {
  assembleDecorations,
  isDecorationVisualization,
  type Visualization,
} from "../lib/visualization-registry";
import { containerDecorationSpace } from "../lib/visualization-decorations";
import {
  VISUALIZATION_CATALOG,
  duplicateVisualizationGroups,
  duplicateVisualizationIds,
  visualizationsForGroups,
} from "../lib/visualization-catalog";
import {
  FLUID_CELL_TRACE_ABI_VERSION,
  FLUID_CELL_TRACE_LAYERS,
  FLUID_CELL_TRACE_LAYER_DEFINITIONS,
  FLUID_CELL_TRACE_STATUS,
  type FluidCellTrace,
  type FluidCellTraceFineProbe,
  type FluidCellTraceNeighbor,
} from "../lib/fluid-cell-trace";

const DIMENSIONS = [16, 16, 16] as const;
const SPACE = containerDecorationSpace([...DIMENSIONS] as [number, number, number], [2, 2, 2]);

function neighbor(overrides: Partial<FluidCellTraceNeighbor> = {}): FluidCellTraceNeighbor {
  return {
    direction: 0, row: 7, leafSize: 4, flags: 1, leafOrigin: [12, 0, 8], pressure: 0,
    present: true, coarser: false, finer: false, boundary: false,
    ...overrides,
  };
}

function probe(overrides: Partial<FluidCellTraceFineProbe> = {}): FluidCellTraceFineProbe {
  return {
    cell: [34, 6, 34], flags: 0, phi: -0.5, seedCell: [34, 6, 34], seedCode: 0, hop: 0,
    resident: true, valid: true, interface: false, negative: true, resolved: true,
    stale: false, missing: false,
    ...overrides,
  };
}

function trace(overrides: Partial<FluidCellTrace> = {}): FluidCellTrace {
  return {
    version: FLUID_CELL_TRACE_ABI_VERSION,
    status: FLUID_CELL_TRACE_STATUS.resolved,
    pixel: [1, 2], requestToken: 1,
    cell: [8, 0, 8], row: 3, leafSize: 4, leafOrigin: [8, 0, 8],
    diagonal: 6, rhs: -1, entryCount: 6, volume: 64, topologyCode: 0, pressure: 0.5,
    fineSamples: 8, fineResolved: 8, fineMaximumHop: 3, fineInterface: 1,
    dimensions: [...DIMENSIONS] as unknown as FluidCellTrace["dimensions"], fineFactor: 4,
    fineProbes: 512, fineMissing: 0, fineStale: 0, fineNegative: 4,
    coarsePhi: 0, coarsePhiMinimum: 0, coarsePhiMaximum: 0, coarsePhiFlags: 0,
    probeMinimumPhi: -1, probeMaximumPhi: 1, probeNearestPhi: 0.1,
    fineProbeRecords: [],
    neighbors: [],
    hits: [], hitIndex: 0, hitOverflow: 0,
    ...overrides,
  };
}

/** Segments one named decoration contributes for a subject. */
function draw(id: string, subject: unknown, emphasis: "hover" | "selected" = "selected") {
  const definition = fineBandCellVisualizations.find((entry) => entry.id === id);
  assert.ok(definition, `no decoration ${id}`);
  const assembled = assembleDecorations({
    definitions: [definition as Visualization], subjects: [subject], space: SPACE, emphasis,
  });
  return assembled;
}

/* ------------------------------------------------------------------------- */
/* Registration                                                               */
/* ------------------------------------------------------------------------- */

test("every fine-band layer has a declaration the toggle strip can find", () => {
  for (const group of ["surface", "patch", "links", "gaps"] as const) {
    assert.equal(visualizationsForGroups([group]).length, 1, `group ${group} is not declared once`);
  }
});

test("the new layers are in the reader's vocabulary and carry definitions", () => {
  for (const group of ["surface", "patch", "links", "gaps"] as const) {
    assert.ok(FLUID_CELL_TRACE_LAYERS.includes(group));
    assert.ok(FLUID_CELL_TRACE_LAYER_DEFINITIONS[group].label.length > 0);
    assert.match(FLUID_CELL_TRACE_LAYER_DEFINITIONS[group].swatch, /^#[0-9a-f]{6}$/);
  }
});

test("the catalog still has unique ids and groups after the additions", () => {
  assert.deepEqual(duplicateVisualizationIds(), []);
  assert.deepEqual(duplicateVisualizationGroups(), []);
  for (const definition of fineBandCellVisualizations) {
    assert.ok(VISUALIZATION_CATALOG.includes(definition));
  }
});

test("a chip's swatch matches the colour its decorator actually draws with", () => {
  // The framework's rule: a legend cannot go on describing a colour the
  // decorator stopped using, because both come from one declaration.
  for (const definition of fineBandCellVisualizations) {
    assert.ok(isDecorationVisualization(definition));
    const group = definition.group as "surface" | "patch" | "links" | "gaps";
    assert.equal(definition.swatch, FLUID_CELL_TRACE_LAYER_DEFINITIONS[group].swatch);
  }
});

/* ------------------------------------------------------------------------- */
/* Hover shows what pinning shows                                             */
/* ------------------------------------------------------------------------- */

test("every fine-band layer draws hovered exactly what it draws pinned", () => {
  // These layers used to draw nothing until a cell was pinned, on the argument
  // that a hover rebuilds as the pointer moves. It does not: the key is the
  // drawn facts, so sweeping across one leaf rebuilds once, and the saving was
  // paid for by making the reader commit to a cell before seeing anything about
  // it. Withholding on hover is now a bug, so it is asserted against.
  const subject = trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: 3 })],
    fineProbeRecords: [
      probe({ interface: true }),
      probe({ hop: 4, seedCell: [30, 6, 34] }),
      // So the gaps layer has something of its own to draw; otherwise the
      // comparison below would pass on an empty picture.
      probe({ cell: [36, 6, 36], resident: false, valid: false, missing: true }),
    ],
  });
  for (const definition of fineBandCellVisualizations) {
    const hovered = draw(definition.id, subject, "hover");
    const pinned = draw(definition.id, subject, "selected");
    assert.ok(pinned.geometry.segmentCount > 0, `${definition.id} drew nothing to compare`);
    assert.equal(hovered.geometry.segmentCount, pinned.geometry.segmentCount, definition.id);
    assert.deepEqual(hovered.entries, pinned.entries, definition.id);
    // The picture is identical; only the assembler's own record of which state
    // it was built in differs, which is what makes a pin transition rebuild.
    assert.equal(hovered.key.replace("hover", "selected"), pinned.key, definition.id);
  }
});

/* ------------------------------------------------------------------------- */
/* Free surface                                                               */
/* ------------------------------------------------------------------------- */

test("a crossing draws the liquid stub and a ring on the dual edge", () => {
  const assembled = draw("pressure-solve/free-surface", trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: 3 })],
  }));
  // One segment for the stub plus a sixteen-facet ring.
  assert.equal(assembled.geometry.segmentCount, 17);
  assert.match(assembled.entries[0].notes[0].detail ?? "", /scaling that face's coefficient 4\.0x/);
  assert.equal(assembled.entries[0].notes[0].evidence, "gathered");
});

test("a row with no coarse record is not accepted by the crossing layer", () => {
  const assembled = draw("pressure-solve/free-surface", trace({
    coarsePhi: -1, coarsePhiFlags: 0, neighbors: [neighbor({ direction: 0, phi: 3 })],
  }));
  assert.equal(assembled.entries.length, 0);
});

test("an inward crossing is drawn but reports no coefficient scale", () => {
  const assembled = draw("pressure-solve/free-surface", trace({
    coarsePhi: 2, coarsePhiFlags: 0b1001, neighbors: [neighbor({ direction: 0, phi: -2 })],
  }));
  assert.ok(assembled.geometry.segmentCount > 0);
  assert.match(assembled.entries[0].notes[0].detail ?? "", /none from the liquid side/);
});

/* ------------------------------------------------------------------------- */
/* Interface patch                                                            */
/* ------------------------------------------------------------------------- */

test("interface probes draw a cross, and an axis crossing also draws its normal ring", () => {
  const axisCode = (1 << 24) | Math.round(0.5 * 0xff_ffff);
  const coincident = draw("fine-band/patch", trace({
    fineProbeRecords: [probe({ interface: true, seedCode: 0 })],
  }));
  // A cross is three bars; with no direction there is no ring to orient.
  assert.equal(coincident.geometry.segmentCount, 3);

  const oriented = draw("fine-band/patch", trace({
    fineProbeRecords: [probe({ interface: true, seedCode: axisCode })],
  }));
  assert.equal(oriented.geometry.segmentCount, 3 + 10);
});

test("non-interface probes contribute nothing to the patch", () => {
  const assembled = draw("fine-band/patch", trace({
    fineProbeRecords: [probe({ interface: false }), probe({ valid: false, interface: true })],
  }));
  assert.equal(assembled.entries.length, 0);
});

/* ------------------------------------------------------------------------- */
/* Closest-point links                                                        */
/* ------------------------------------------------------------------------- */

test("self-seeded samples draw no link, since the link would have zero length", () => {
  const assembled = draw("fine-band/closest-points", trace({
    fineProbeRecords: [probe({ hop: 0 })],
  }));
  assert.equal(assembled.entries.length, 0);
});

test("a hopped sample draws one arrowed link at its closest point", () => {
  const assembled = draw("fine-band/closest-points", trace({
    fineProbeRecords: [probe({ hop: 4, seedCell: [30, 6, 34] })],
  }));
  // A shaft plus its screen-space arrowhead instance.
  assert.equal(assembled.geometry.segmentCount, 2);
  assert.match(assembled.entries[0].notes[0].detail ?? "", /deepest here is 4 fine cells/);
});

test("links are drawn in the fine lattice, not in finest cells", () => {
  // The whole point of DecorationSpace: a sample at fine coordinate 34 with
  // factor four belongs at finest cell 8.5, not at 34.
  const assembled = draw("fine-band/closest-points", trace({
    fineFactor: 4, fineProbeRecords: [probe({ cell: [34, 6, 34], hop: 4, seedCell: [30, 6, 34] })],
  }));
  const x = assembled.geometry.segments[0];
  // Finest cell 8.625 of 16, over a two-metre container centred on x.
  assert.ok(Math.abs(x - (-1 + (34.5 / 64) * 2)) < 1e-6, `unexpected x ${x}`);
});

/* ------------------------------------------------------------------------- */
/* Band gaps                                                                  */
/* ------------------------------------------------------------------------- */

test("gaps draw one dashed brick box, deduplicated across probes inside it", () => {
  const assembled = draw("fine-band/gaps", trace({
    fineProbeRecords: [
      probe({ cell: [32, 4, 32], valid: false, resident: false, missing: true }),
      probe({ cell: [34, 6, 34], valid: false, resident: false, missing: true }),
    ],
  }));
  // Both samples fall in brick (32,4,32); one box of twelve edges.
  assert.equal(assembled.geometry.segmentCount, 12);
  assert.match(assembled.entries[0].notes[0].detail ?? "", /2 recorded probes found no resident page/);
});

test("a stale page is counted and coloured apart from an absent one", () => {
  const assembled = draw("fine-band/gaps", trace({
    fineProbeRecords: [
      probe({ cell: [32, 4, 32], valid: false, resident: false, missing: true }),
      probe({ cell: [40, 4, 32], valid: false, resident: false, stale: true, missing: false }),
    ],
  }));
  assert.equal(assembled.geometry.segmentCount, 24);
  assert.match(assembled.entries[0].notes[0].detail ?? "",
    /1 probes found no page and 1 found one at another generation/);
});

test("a fully resident leaf draws no gaps", () => {
  const assembled = draw("fine-band/gaps", trace({ fineProbeRecords: [probe()] }));
  assert.equal(assembled.entries.length, 0);
});

/* ------------------------------------------------------------------------- */
/* Keying                                                                     */
/* ------------------------------------------------------------------------- */

test("an unchanged picture keys the same, so a frozen cell uploads nothing", () => {
  // The gather re-runs every frame; keying on the drawn facts is what lets the
  // assembler skip the rebuild while the camera orbits a pinned cell.
  const subject = trace({
    coarsePhi: -1, coarsePhiFlags: 0b1001,
    neighbors: [neighbor({ direction: 0, phi: 3 })],
    fineProbeRecords: [probe({ hop: 4, seedCell: [30, 6, 34] })],
  });
  const first = assembleDecorations({
    definitions: fineBandCellVisualizations as Visualization[],
    subjects: [subject], space: SPACE,
  });
  const second = assembleDecorations({
    definitions: fineBandCellVisualizations as Visualization[],
    // A different token and pressure: the same picture by every fact drawn.
    subjects: [{ ...subject, requestToken: 99, pressure: 12 }], space: SPACE,
  });
  assert.equal(first.key, second.key);
});

test("a moved seed changes the key, because the picture changed", () => {
  const base = trace({ fineProbeRecords: [probe({ hop: 4, seedCell: [30, 6, 34] })] });
  const moved = trace({ fineProbeRecords: [probe({ hop: 5, seedCell: [29, 6, 34] })] });
  assert.notEqual(
    draw("fine-band/closest-points", base).key,
    draw("fine-band/closest-points", moved).key,
  );
});
