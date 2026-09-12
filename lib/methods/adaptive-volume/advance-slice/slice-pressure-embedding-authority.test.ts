import assert from "node:assert/strict";
import test from "node:test";

import { productionSceneSliceSeedById } from "./production-scene-slice";
import { prepareSlicePressureEmbedding } from "./slice-pressure-embedding";
import { advanceSlice, createAdvanceSlice } from "./slice-solver";

test("retained pressure cache reuses an unchanged virtual-extrusion row image", () => {
  const source = productionSceneSliceSeedById("coarse-surface-translation");
  // Static SolidWorld deliberately invalidates every production row. Remove
  // only that capability here to exercise the no-solid reuse branch itself.
  const seed = { ...source, production: undefined,
    boundary: { ...source.boundary, z: "symmetry" as const } };
  const slice = createAdvanceSlice(seed), embedding = slice.pressureEmbedding;
  assert.ok(embedding);
  assert.equal(embedding.mappingFault, undefined);

  const first = prepareSlicePressureEmbedding(embedding, seed,
    slice.numericalTopology, slice.fields);
  const firstAuthority = embedding.pressureAuthority.receipt;
  assert.equal(first.dirtyRowCount, embedding.grid.gradientRows.length);
  assert.ok(firstAuthority.executionGeneration > 0);
  assert.ok(firstAuthority.dirtyRowTiles.length > 0);
  const firstOrder = Uint32Array.from(embedding.pressureAuthority.executionOrder);

  const second = prepareSlicePressureEmbedding(embedding, seed,
    slice.numericalTopology, slice.fields);
  const secondAuthority = embedding.pressureAuthority.receipt;
  assert.equal(second.dirtyRowCount, 0);
  assert.equal(secondAuthority.executionGeneration,
    firstAuthority.executionGeneration + 1);
  assert.equal(secondAuthority.dirtyRowTiles.length, 0);
  assert.deepEqual([...embedding.pressureAuthority.executionOrder], [...firstOrder]);
});

test("automatic rerung preserves retained PCM PCF PEI generations", () => {
  const source = productionSceneSliceSeedById("water-box-dam-break");
  const seed = { ...source, boundary: { ...source.boundary, z: "symmetry" as const } };
  const slice = createAdvanceSlice(seed);
  advanceSlice(slice);
  const beforeTopology = slice.topology.accepted.generation;
  const before = slice.pressureAuthority.receipt;

  advanceSlice(slice);

  assert.ok(slice.topology.accepted.generation > beforeTopology);
  const acceptedInputGeneration = slice.topology.accepted.generation;
  advanceSlice(slice);
  assert.ok(slice.pressureEmbedding);
  assert.equal(slice.pressureEmbedding.mappingFault, undefined);
  const after = slice.pressureAuthority.receipt;
  assert.ok(after.pcmCellGeneration > before.pcmCellGeneration);
  assert.ok(after.pcmRowGeneration > before.pcmRowGeneration);
  assert.ok(after.coefficientGeneration > before.coefficientGeneration);
  assert.ok(after.executionGeneration > before.executionGeneration);
  // Pressure consumes the generation accepted at frame start. A later
  // candidate commit is explicitly next-frame input.
  assert.equal(after.topologyGeneration, acceptedInputGeneration);
});
