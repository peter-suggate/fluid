import assert from "node:assert/strict";
import test from "node:test";
import {
  refinementRegionFromDrag,
  refinementRegionFromRecord,
  refinementRegionRecord,
  refinementRegionResizePolicy,
  snapRefinementRegionBox,
  studioRegionSpace,
  withRefinementRegion,
} from "../../../core/editor-refinement-region";
import { cloneScene, defaultScene, type FluidRefinementRegion, type SceneDescription }
  from "../../../core/model";
import { refinementRegionLattice, sceneRefinementRegions } from "../../../core/refinement-regions";
import { sceneContainerBox, type BoxExtent } from "../../../core/editor-entity";
import { BRICK_FINE_CELLS } from "../definition";

/**
 * The 3-D adapter, held to the two things it is allowed to be.
 *
 * It must be **exactly** what it was for every region at or above the brick,
 * because that is every region the studio ships: `DEFAULT_REFINEMENT_REGION_CELL_SIZE`
 * is 8. And it must be a real `RegionSpace` — the same record in and out — or
 * the package below it is generic over nothing.
 *
 * `metreSnapAsItWas` is the arithmetic this file replaced, transcribed. It is
 * the oracle rather than the implementation: a transcription that drifts fails
 * here, which is the point of keeping it.
 */
function metreSnapAsItWas(scene: SceneDescription, box: BoxExtent, cells: number): BoxExtent {
  const limits = sceneContainerBox(scene);
  const { cellSize_m } = refinementRegionLattice(scene);
  const tolerance = 1e-6;
  const min = { ...box.min }, max = { ...box.max };
  (["x", "y", "z"] as const).forEach((axis, index) => {
    const size = cellSize_m[index]! * cells;
    const lo = Math.min(box.min[axis], box.max[axis]);
    const hi = Math.max(box.min[axis], box.max[axis]);
    const snapped = {
      min: limits.min[axis] + Math.floor((lo - limits.min[axis]) / size + tolerance) * size,
      max: limits.min[axis] + Math.ceil((hi - limits.min[axis]) / size - tolerance) * size,
    };
    min[axis] = Math.max(limits.min[axis], snapped.min);
    max[axis] = Math.min(limits.max[axis], Math.max(snapped.max, snapped.min + size));
    if (max[axis] - min[axis] < size) min[axis] = Math.max(limits.min[axis], max[axis] - size);
  });
  return { min, max };
}

function close(actual: BoxExtent, expected: BoxExtent, what: string) {
  for (const axis of ["x", "y", "z"] as const) {
    assert.ok(Math.abs(actual.min[axis] - expected.min[axis]) < 1e-9,
      `${what} min.${axis}: ${actual.min[axis]} vs ${expected.min[axis]}`);
    assert.ok(Math.abs(actual.max[axis] - expected.max[axis]) < 1e-9,
      `${what} max.${axis}: ${actual.max[axis]} vs ${expected.max[axis]}`);
  }
}

const drawn: BoxExtent = {
  min: { x: -0.137, y: 0.041, z: -0.211 },
  max: { x: 0.219, y: 0.283, z: 0.174 },
};

test("at or above the brick the metre snap is exactly what it was", () => {
  const scene = cloneScene(defaultScene);
  for (const cells of [8, 16, 32]) {
    close(snapRefinementRegionBox(scene, drawn, cells), metreSnapAsItWas(scene, drawn, cells),
      `MIN ${cells}`);
  }
});

test("below the brick the snap is the brick, which is the change", () => {
  const scene = cloneScene(defaultScene);
  const brick = metreSnapAsItWas(scene, drawn, BRICK_FINE_CELLS);
  for (const cells of [1, 2, 4]) {
    close(snapRefinementRegionBox(scene, drawn, cells), brick, `MIN ${cells}`);
    // And it is genuinely coarser than the old per-floor-cell answer.
    const asItWas = metreSnapAsItWas(scene, drawn, cells);
    assert.notDeepEqual(
      [asItWas.min.x, asItWas.max.x],
      [brick.min.x, brick.max.x],
      `MIN ${cells} should have moved`);
  }
});

test("the resize handle steps by the brick at the studio default", () => {
  const scene = cloneScene(defaultScene);
  const { cellSize_m } = refinementRegionLattice(scene);
  const region = refinementRegionFromDrag(scene, drawn.min, drawn.max, {})!;
  assert.equal(region.minimumCellSize_cells, 8);
  const step_m = (floor: number) =>
    refinementRegionResizePolicy(scene, { ...region, minimumCellSize_cells: floor }).snap_m ?? [];
  const policy = refinementRegionResizePolicy(scene, region);
  for (let axis = 0; axis < 3; axis += 1) {
    assert.ok(Math.abs((policy.snap_m ?? [])[axis]! - cellSize_m[axis]! * BRICK_FINE_CELLS) < 1e-12,
      `axis ${axis}`);
    assert.equal((policy.minimum_m ?? [])[axis], (policy.snap_m ?? [])[axis]);
  }
  // A region asking for one finest cell still steps by a brick; a coarser floor
  // keeps its own coarser step.
  assert.ok(Math.abs(step_m(1)[0]! - cellSize_m[0]! * BRICK_FINE_CELLS) < 1e-12);
  assert.ok(Math.abs(step_m(32)[0]! - cellSize_m[0]! * 32) < 1e-12);
});

test("a drawn box survives the metres round trip unchanged", () => {
  const scene = cloneScene(defaultScene);
  const region = refinementRegionFromDrag(scene, drawn.min, drawn.max, {})!;
  const again = snapRefinementRegionBox(scene,
    { min: region.min_m, max: region.max_m }, region.minimumCellSize_cells);
  close(again, { min: region.min_m, max: region.max_m }, "re-snap");
});

test("the studio RegionSpace round-trips a document region through the shared record", () => {
  const scene = cloneScene(defaultScene);
  const region = refinementRegionFromDrag(scene, drawn.min, drawn.max,
    { minimumCellSize_cells: 8, maximumCellSize_cells: 16 })!;
  const withRegion = withRefinementRegion(scene, region.id, region);

  const records = studioRegionSpace.list(withRegion);
  assert.equal(records.length, 1);
  const record = records[0]!;
  assert.equal(record.min_cells.length, 3);
  assert.equal(record.minimumCellSize_cells, 8);
  assert.equal(record.maximumCellSize_cells, 16);

  const back = refinementRegionFromRecord(withRegion, record);
  close({ min: back.min_m, max: back.max_m }, { min: region.min_m, max: region.max_m },
    "record round trip");
  assert.equal(back.id, region.id);
  assert.equal(back.rule, region.rule);

  // …and writing that record back produces the same document.
  const rewritten = studioRegionSpace.write(withRegion, record.id, record) as SceneDescription;
  assert.deepEqual(sceneRefinementRegions(rewritten), sceneRefinementRegions(withRegion));

  // `undefined` drops it, and drops the key with the last one.
  const dropped = studioRegionSpace.write(withRegion, record.id, undefined) as SceneDescription;
  assert.equal(dropped.fluid.refinementRegions, undefined);
});

test("the record keeps a region's bounds on the dyadic ladder", () => {
  const scene = cloneScene(defaultScene);
  const malformed = {
    id: "region-1", rule: "minimum-cell-size",
    minimumCellSize_cells: 3, maximumCellSize_cells: 1,
    min_m: drawn.min, max_m: drawn.max,
  } as unknown as FluidRefinementRegion;
  const record = refinementRegionRecord(scene, malformed);
  // 3 rounds down to 2, and a ceiling under the floor is lifted to it.
  assert.equal(record.minimumCellSize_cells, 2);
  assert.equal(record.maximumCellSize_cells, 2);
});

test("the studio space reports the capacity and the ladder the document validates against", () => {
  assert.equal(studioRegionSpace.axes, 3);
  assert.equal(studioRegionSpace.capacity, 8);
  assert.deepEqual([...studioRegionSpace.cellSizes], [1, 2, 4, 8, 16, 32]);
  assert.equal(studioRegionSpace.brick_cells, BRICK_FINE_CELLS);
  const scene = cloneScene(defaultScene);
  assert.equal(studioRegionSpace.nextId(scene), "region-1");
  assert.deepEqual(studioRegionSpace.lattice(scene).dimensions,
    refinementRegionLattice(scene).dimensions);
});
