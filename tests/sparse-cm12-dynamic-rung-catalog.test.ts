import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, type SparseAdaptiveMassBrick } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from "../lib/methods/adaptive-volume/sparse-atlas-composite-projection";
import { DYNAMIC_PAGE_RUNGS, DYNAMIC_RUNG_LAYOUTS, DYNAMIC_PAGE_CELL_COUNT,
  DYNAMIC_PAGE_ROW_COUNT, DYNAMIC_PAGE_TERM_COUNT, dynamicRowTermOffset,
  preparedDynamicSeamCatalogue, compileDynamicSeamVariant, dynamicRungCatalogue } from "../lib/methods/adaptive-volume/sparse-cm12-dynamic-rung-catalog";

test("prepared dynamic rung identities do not alias source and candidate storage", () => {
  assert.equal(DYNAMIC_PAGE_CELL_COUNT, 585);
  assert.equal(DYNAMIC_PAGE_ROW_COUNT, 2010);
  assert.equal(DYNAMIC_PAGE_TERM_COUNT, 5550);
  const occupied = new Set<number>();
  for (const layout of DYNAMIC_RUNG_LAYOUTS) {
    for (let row = 0; row < layout.rowCount; row++) {
      const face = row % ((layout.resolution + 1) * layout.resolution ** 2) % (layout.resolution + 1);
      const count = face === 0 || face === layout.resolution ? 5 : 2;
      const first = dynamicRowTermOffset(layout.resolution, row);
      for (let term = first; term < first + count; term++) {
        assert.ok(!occupied.has(term), `aliased term ${term}`); occupied.add(term);
      }
    }
  }
  assert.equal(occupied.size, DYNAMIC_PAGE_TERM_COUNT);
  assert.equal(Math.max(...occupied), DYNAMIC_PAGE_TERM_COUNT - 1);
});

for (const b of [4, 8] as const) test(`B${b} prepared dynamic seams match the authoritative CM12 operator on all six sides`, () => {
  const catalogue = dynamicRungCatalogue(b).preparedDynamicSeamCatalogue();
  for (const variant of catalogue.values()) {
    const axis = Math.floor(variant.side / 2), positive = variant.side % 2 === 1;
    const coordinate: [number, number, number] = [1, 1, 1];
    const otherCoordinate: [number, number, number] = [...coordinate];
    otherCoordinate[axis] += positive ? 1 : -1;
    const brick = (q: [number, number, number], resolution: typeof DYNAMIC_PAGE_RUNGS[number]): SparseAdaptiveMassBrick => ({
      key: q[0] + 3 * (q[1] + 3 * q[2]), coordinate: q, resolution,
      density: new Float64Array(resolution ** 3), gamma: new Float64Array(resolution ** 3).fill(1),
    });
    const own = brick(coordinate, variant.own);
    const bricks = [own]; if (variant.neighbor) bricks.push(brick(otherCoordinate, variant.neighbor));
    const grid = buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([3*b, 3*b, 3*b], bricks, 1, b));
    const plane = b + (positive ? b : 0);
    const rows = grid.gradientRows.filter(row => row.axis === axis && row.centerFine[axis] === plane
      && row.terms.some(term => grid.cells[term.cellId]!.brickKey === own.key));
    assert.equal(variant.rows.length, rows.length);
    for (const compiled of variant.rows) {
      const source = rows.find(row => row.centerFine.every((value, a) => Math.abs(value - b - compiled.center[a]!) < 1e-10));
      assert.ok(source, `${variant.own}/${variant.neighbor}/${variant.side}: missing row`);
      assert.equal(compiled.area, source.area); assert.equal(compiled.distance, source.distance);
      const canonical = (terms: readonly { neighbor: boolean; local: number; coefficient: number }[]) => terms
        .map(t => [Number(t.neighbor), t.local, t.coefficient]).sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
      assert.deepEqual(canonical(compiled.terms), canonical(source.terms.map(term => {
        const cell = grid.cells[term.cellId]!;
        return { neighbor: cell.brickKey !== own.key, local: cell.localIndex, coefficient: term.coefficient };
      })));
    }
    for (const [row, term] of variant.incidence) {
      const source = variant.rows.find(candidate => candidate.row === row);
      assert.ok(source); assert.equal(source.terms[term]!.neighbor, false);
    }
  }
});

test("unprepared and ungraded variants fail closed", () => {
  assert.throws(() => compileDynamicSeamVariant(8, 2, 0), /ungraded/);
  assert.throws(() => compileDynamicSeamVariant(3 as 8, 4, 0), /unprepared/);
});

test("new pages use the coarsest physical rung allowed by 2:1 and authored bounds", async () => {
  const { coarseDynamicAdmissionRung: choose } = await import("../lib/methods/adaptive-volume/sparse-cm12-dynamic-rung-catalog");
  assert.equal(choose([]), 1);
  assert.equal(choose([8, 4]), 1);
  assert.equal(choose([4, 2]), 2);
  assert.equal(choose([2, 1]), 4);
  assert.equal(choose([16, 8, 4]), 1, "macro neighbours use width, not rung number");
  assert.equal(choose([], 4), 4);
  assert.throws(() => choose([1], 1, 2), /conflicts/);
  assert.throws(() => choose([0]), /invalid/);
});

test("construction-time page image reproduces every isolated rung without geometry synthesis", async () => {
  const { prepareDynamicPageImage, DYNAMIC_PAGE_ROW_BASE: rb, DYNAMIC_PAGE_TERM_BASE: tb } =
    await import("../lib/methods/adaptive-volume/sparse-cm12-dynamic-rung-catalog");
  const cellBase = 137, rowBase = 791, termBase = 1703;
  const image = prepareDynamicPageImage(cellBase, rowBase, termBase);
  const f = new Float32Array(image.buffer);
  for (const layout of DYNAMIC_RUNG_LAYOUTS) {
    const brick: SparseAdaptiveMassBrick = { key: 0, coordinate: [0, 0, 0], resolution: layout.resolution,
      density: new Float64Array(layout.cellCount), gamma: new Float64Array(layout.cellCount).fill(1) };
    const grid = buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas([8, 8, 8], [brick]));
    assert.equal(grid.gradientRows.length, layout.rowCount);
    for (let row = 0; row < layout.rowCount; row++) {
      const index = layout.rowOffset + row;
      const center = [0, 1, 2].map(axis => f[rb + (4 + axis) * DYNAMIC_PAGE_ROW_COUNT + index]!);
      const axis = image[rb + DYNAMIC_PAGE_ROW_COUNT + index]! >>> 30;
      const expected = grid.gradientRows.find(row => row.axis === axis && row.centerFine.every((v, a) => v === center[a]));
      assert.ok(expected);
      assert.equal(f[rb + 7 * DYNAMIC_PAGE_ROW_COUNT + index], expected.area);
      assert.equal(f[rb + 2 * DYNAMIC_PAGE_ROW_COUNT + index], expected.distance);
      assert.equal(f[rb + 3 * DYNAMIC_PAGE_ROW_COUNT + index], expected.dualWeight);
      const packed = image[rb + index]!, count = packed >>> 23;
      assert.equal(count, expected.terms.length);
      const first = (packed & 0x7fffff) - termBase;
      for (let term = 0; term < count; term++) {
        assert.equal(image[tb + 2 * (first + term)]! - cellBase - layout.cellOffset, expected.terms[term]!.cellId);
        assert.equal(f[tb + 2 * (first + term) + 1], expected.terms[term]!.coefficient);
      }
    }
  }
  assert.throws(() => prepareDynamicPageImage(0, 0, 0x800000), /address space/);
});
