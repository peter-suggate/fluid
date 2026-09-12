import assert from "node:assert/strict";
import test from "node:test";

import { sceneDocument } from "../../../core/scene-definition";
import { getSceneDefinition } from "../../../core/scenes";
import type { SparseAtlasCompositeCell, SparseAtlasGradientRow,
  SparseAtlasGradientTerm } from "../sparse-atlas-composite-projection";
import { productionSceneSliceSeed } from "./production-scene-slice";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";

for (const sceneId of ["coarse-surface-translation", "water-box-dam-break"] as const) {
  test(`${sceneId} retains one source cell and row for every centre-Z pressure unknown`, () => {
    const definition = getSceneDefinition(sceneId);
    const scene = sceneDocument(definition);
    scene.container.depthBoundary = "symmetry";
    const seed = productionSceneSliceSeed({ ...definition,
      build: () => structuredClone(scene) });
    const slice = createAdvanceSlice(seed);
    const embedding = slice.pressureEmbedding;
    assert.ok(embedding, "a symmetry extrusion must retain its production 3-D graph");
    assert.equal(embedding.mappingFault, undefined);

    const centreZ = seed.viewport.centerCellZ + 0.5;
    assert.equal(embedding.centreCell.length, slice.numericalTopology.cells.length);
    for (const reduced of slice.numericalTopology.cells) {
      const sourceId: number = embedding.centreCell[reduced.id]!;
      assert.ok(sourceId >= 0, `reduced cell ${reduced.id} has no centre-Z source cell`);
      const source: SparseAtlasCompositeCell = embedding.grid.cells[sourceId]!;
      assert.deepEqual(source.minimumFine.slice(0, 2), reduced.minimum);
      assert.deepEqual(source.maximumFine.slice(0, 2), reduced.maximum);
      assert.ok(source.minimumFine[2] <= centreZ && centreZ < source.maximumFine[2]);
      assert.equal(embedding.reducedCell[sourceId], reduced.id);
    }

    assert.equal(embedding.centreRow.length, slice.numericalTopology.rows.length);
    for (const reduced of slice.numericalTopology.rows) {
      const sourceId: number = embedding.centreRow[reduced.id]!;
      assert.ok(sourceId >= 0,
        `reduced row ${reduced.id} has no unique centre-Z source row (${sourceId})`);
      const source: SparseAtlasGradientRow = embedding.grid.gradientRows[sourceId]!;
      assert.equal(source.axis, reduced.axis);
      assert.equal(source.centerFine[0], reduced.center[0]);
      assert.equal(source.centerFine[1], reduced.center[1]);
      assert.equal(embedding.projectedRow[sourceId], reduced.id);
      assert.ok(source.terms.some((term: SparseAtlasGradientTerm) => {
        const reducedCell = embedding.reducedCell[term.cellId]!;
        return reducedCell >= 0 && embedding.centreCell[reducedCell] === term.cellId;
      }), `source row ${sourceId} does not touch its centre-Z source cell`);
    }
  });
}

test("an automatic water-box rerung rebuilds complete centre-Z pressure mappings", () => {
  const definition = getSceneDefinition("water-box-dam-break");
  const scene = sceneDocument(definition);
  scene.container.depthBoundary = "symmetry";
  const seed = productionSceneSliceSeed({ ...definition,
    build: () => structuredClone(scene) });
  const slice = createAdvanceSlice(seed);
  const initialGeneration = slice.topology.accepted.generation;

  advanceSlice(slice);
  advanceSlice(slice);

  assert.ok(slice.topology.accepted.generation > initialGeneration,
    "fixture must publish an automatic candidate generation");
  const embedding = slice.pressureEmbedding;
  assert.ok(embedding);
  assert.equal(embedding.mappingFault, undefined);
  assert.equal([...embedding.centreCell].filter(source => source < 0).length, 0);
  assert.equal([...embedding.centreRow].filter(source => source < 0).length, 0);
  assert.equal(embedding.pressure.length, embedding.grid.cells.length);
  assert.equal(embedding.pressureMember.length, embedding.grid.cells.length);
  assert.equal(slice.fields.fault, null);
});
