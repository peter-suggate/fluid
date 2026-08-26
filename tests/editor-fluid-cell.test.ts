import assert from "node:assert/strict";
import test from "node:test";
import type { EditorEntityContext } from "../lib/core/editor-entity";
import { EDITOR_PROBES, targetActionsAt, targetAtRay } from "../lib/core/editor-probe-catalog";
import {
  decodeFluidCellTrace,
  fluidCellLeafBox,
  fluidCellTraceLattice,
  FLUID_CELL_TRACE_HEADER,
  FLUID_CELL_TRACE_HEADER_WORDS,
  FLUID_CELL_TRACE_MAGIC,
  FLUID_CELL_TRACE_STATUS,
  type FluidCellTrace,
} from "../lib/core/fluid-cell-trace";
import { cloneScene, defaultScene } from "../lib/core/model";

/**
 * A trace built through the shipping decode rather than as an object literal.
 *
 * The probe reads seven of the header's forty-eight words, and a literal would
 * assert that they mean what this test thinks they mean rather than what the ABI
 * says. Going through `decodeFluidCellTrace` keeps the fixture honest: if an
 * offset moves, this fixture moves with it.
 */
function traceWith(fields: {
  readonly status?: number;
  readonly cell?: readonly [number, number, number];
  readonly leafOrigin?: readonly [number, number, number];
  readonly leafSize?: number;
  readonly dimensions?: readonly [number, number, number];
}): FluidCellTrace {
  const words = new Uint32Array(FLUID_CELL_TRACE_HEADER_WORDS);
  words[FLUID_CELL_TRACE_HEADER.magic] = FLUID_CELL_TRACE_MAGIC;
  words[FLUID_CELL_TRACE_HEADER.status] = fields.status ?? FLUID_CELL_TRACE_STATUS.resolved;
  words[FLUID_CELL_TRACE_HEADER.leafSize] = fields.leafSize ?? 1;
  const triple = (offset: number, value: readonly [number, number, number]) => {
    words[offset] = value[0]; words[offset + 1] = value[1]; words[offset + 2] = value[2];
  };
  triple(FLUID_CELL_TRACE_HEADER.cell, fields.cell ?? [8, 4, 8]);
  triple(FLUID_CELL_TRACE_HEADER.leafOrigin, fields.leafOrigin ?? [8, 4, 8]);
  triple(FLUID_CELL_TRACE_HEADER.dimensions, fields.dimensions ?? [16, 16, 16]);
  const trace = decodeFluidCellTrace(words);
  assert.ok(trace, "the fixture must decode through the shipping ABI");
  return trace;
}

/** A tank scene, and a context carrying whatever cell publication a case needs. */
function fixture(fields: Parameters<typeof traceWith>[0] = {}, pinned = false) {
  const scene = cloneScene(defaultScene);
  // No authored water, so the fluid-body entity cannot stand in front of the
  // leaf; the case about that ordering sets it up itself.
  scene.fluid = { ...scene.fluid, initialCondition: "tank-fill" };
  scene.container = { ...scene.container, fillFraction: 0 };
  const trace = traceWith(fields);
  const lattice = fluidCellTraceLattice(trace, scene.container);
  assert.ok(lattice);
  const context: EditorEntityContext = {
    scene, bodies: [], pickingAvailable: true,
    fluidCell: { trace, lattice, pinned },
  };
  return { scene, trace, lattice, context };
}

type Lattice = {
  readonly origin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
};

const leafCentre = (lattice: Lattice, leafOrigin: readonly [number, number, number],
  leafSize: number, axis: 0 | 1 | 2) =>
  lattice.origin_m[axis] + (leafOrigin[axis] + 0.5 * leafSize) * lattice.cellSize_m[axis];

/**
 * A ray that enters the named leaf through its −X face, from inside the tank.
 *
 * Inside rather than outside on purpose: a container is lined with solid voxels,
 * so a ray from beyond the glass meets that lining first — which is the correct
 * answer to a different question, and would make every case here a test of the
 * solid-voxel probe.
 */
function rayThroughLeaf(lattice: Lattice, leafOrigin: readonly [number, number, number],
  leafSize: number) {
  const centre = (axis: 0 | 1 | 2) => leafCentre(lattice, leafOrigin, leafSize, axis);
  return {
    origin: { x: centre(0) - 2 * leafSize * lattice.cellSize_m[0], y: centre(1), z: centre(2) },
    direction: { x: 1, y: 0, z: 0 },
  };
}

// The trap the renderer's own coverage source calls out: the publication's frame
// is index space, so metres have to come from somewhere else entirely.
test("the lattice turns index space into metres, and the solver's own frame wins", () => {
  const scene = cloneScene(defaultScene);
  const trace = traceWith({ dimensions: [16, 16, 16] });
  const derived = fluidCellTraceLattice(trace, scene.container);
  assert.deepEqual(derived, {
    origin_m: [-0.5 * scene.container.width_m, 0, -0.5 * scene.container.depth_m],
    cellSize_m: [
      scene.container.width_m / 16, scene.container.height_m / 16, scene.container.depth_m / 16,
    ],
  });

  // Sparse flow can publish a fluid world larger than the tank, so the domain is
  // authoritative wherever there is one — assuming the container would put every
  // leaf of an overflowing scene in the wrong place.
  const domain = { origin_m: [-3, -1, -3] as const, cellSize_m: [0.25, 0.25, 0.25] as const };
  assert.deepEqual(fluidCellTraceLattice(trace, scene.container, domain), domain);

  assert.equal(fluidCellTraceLattice(traceWith({ dimensions: [0, 0, 0] }), scene.container),
    undefined, "a trace with no lattice cannot be placed at all");
});

test("a leaf's box spans leafSize cells from its index-space corner", () => {
  const lattice = { origin_m: [-1, 0, -1] as const, cellSize_m: [0.5, 0.25, 0.5] as const };
  assert.deepEqual(fluidCellLeafBox(lattice, [2, 4, 6], 4), {
    min: [-1 + 2 * 0.5, 0 + 4 * 0.25, -1 + 6 * 0.5],
    max: [-1 + 6 * 0.5, 0 + 8 * 0.25, -1 + 10 * 0.5],
  });
});

test("the probe answers with the published leaf, and names nothing in the document", () => {
  const { lattice, context } = fixture({ leafOrigin: [8, 4, 8], leafSize: 2 });
  const target = targetAtRay(context, rayThroughLeaf(lattice, [8, 4, 8], 2));
  assert.equal(target.kind, "fluid-cell");
  assert.equal(target.id, "8,4,8");
  assert.equal(target.tone, "fluid");
  assert.equal(target.selection, undefined,
    "a cell exists in the solve, not in the document, so a click on it selects nothing");
  assert.equal(target.highlight.kind, "box");
  const box = fluidCellLeafBox(lattice, [8, 4, 8], 2);
  assert.ok(target.highlight.kind === "box");
  assert.deepEqual(
    [target.highlight.box.min.x, target.highlight.box.min.y, target.highlight.box.min.z], [...box.min]);
  assert.deepEqual(
    [target.highlight.box.max.x, target.highlight.box.max.y, target.highlight.box.max.z], [...box.max]);
  // Entered through the face it was aimed at, which is what a placement or a
  // second ray cast from here would rest against.
  assert.deepEqual(target.normal, { x: -1, y: 0, z: 0 });
  assert.ok(targetActionsAt(context, target).some((action) => action.id === "inspect-cell"));
});

// The whole reason the probe re-tests the leaf: it is one frame behind, and a
// slab test is what turns that lag into a decline instead of a wrong answer.
test("a ray that has moved off the published leaf gets a different target", () => {
  const { lattice, context } = fixture({ leafOrigin: [8, 4, 8], leafSize: 1 });
  const away = rayThroughLeaf(lattice, [8, 12, 8], 1);
  assert.notEqual(targetAtRay(context, away).kind, "fluid-cell");
});

test("the probe is silent without a publication, while pinned, and on a miss", () => {
  const { scene, lattice } = fixture();
  const ray = rayThroughLeaf(lattice, [8, 4, 8], 1);
  const bare: EditorEntityContext = { scene, bodies: [], pickingAvailable: true };
  assert.notEqual(targetAtRay(bare, ray).kind, "fluid-cell");

  // A pinned trace describes a pixel the reader chose earlier and has since
  // moved away from, so it says nothing about where the cursor is now.
  assert.notEqual(targetAtRay(fixture({}, true).context, ray).kind, "fluid-cell");

  for (const status of [FLUID_CELL_TRACE_STATUS.pending, FLUID_CELL_TRACE_STATUS.miss,
    FLUID_CELL_TRACE_STATUS.unavailable]) {
    assert.notEqual(targetAtRay(fixture({ status }).context, ray).kind, "fluid-cell",
      `status ${status} names no cell`);
  }
});

// The ordering claim the catalog's own note makes, asserted rather than assumed.
test("a document object in front of the leaf still wins the click", () => {
  const { scene, lattice, context } = fixture({ leafOrigin: [8, 4, 8], leafSize: 1 });
  const withWater: EditorEntityContext = {
    ...context,
    scene: { ...scene, container: { ...scene.container, fillFraction: 0.5 } },
  };
  // Down through the surface at a leaf below it: the authored water box is met
  // first, the solver's leaf some way further in.
  const ray = {
    origin: {
      x: leafCentre(lattice, [8, 4, 8], 1, 0),
      y: 0.78 * scene.container.height_m,
      z: leafCentre(lattice, [8, 4, 8], 1, 2),
    },
    direction: { x: 0, y: -1, z: 0 },
  };
  assert.equal(targetAtRay(withWater, ray).kind, "entity");
  // …and becomes transparent once selected, which is the only route to anything
  // inside the water. Same rule that reaches a body inside the tank.
  assert.equal(
    targetAtRay(withWater, ray, { kind: "fluid-body", id: "fluid-body" }).kind, "fluid-cell");
});

test("the fluid cell ranks behind the entities and in front of the ground", () => {
  const order = EDITOR_PROBES.map((probe) => probe.id);
  assert.ok(order.indexOf("fluid-cell") > order.indexOf("entity"));
  assert.ok(order.indexOf("fluid-cell") < order.indexOf("terrain"));
  assert.equal(EDITOR_PROBES.filter((probe) => probe.fallback).length, 1);
});
