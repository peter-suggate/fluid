import assert from "node:assert/strict";
import test from "node:test";
import { compileSparseCM12GenerationTransfer } from "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";
import { transferFixtures, transferSourceRecipe, transferSourceValues, referenceGenerationTransfer } from "./helpers/cm12-generation-transfer-oracle";

for (const fixture of transferFixtures()) test(`${fixture.name}: brute-force transfer oracle conserves physical moments`, () => {
  const source = transferSourceRecipe(fixture.source, fixture.dynamicPage);
  const values = transferSourceValues(fixture.source, source.layout, source.physicalCells, source.physicalRows);
  const plan = compileSparseCM12GenerationTransfer(fixture.source, fixture.target, fixture.air);
  for (const [scalar, face] of [[1, 0], [0, 1]]) {
    const result = referenceGenerationTransfer(fixture.source, fixture.target, values, source.layout,
      source.physicalCells, source.physicalRows, scalar!, face!, fixture.air);
    const density = scalar ? source.layout.densityOtherOffset : source.layout.densityOffset;
    const gamma = scalar ? source.layout.gammaOtherOffset : source.layout.gammaOffset;
    const velocity = scalar ? source.layout.velocityOtherOffset : source.layout.velocityOffset;
    const before = [0, 0, 0, 0, 0], after = [0, 0, 0, 0, 0];
    for (const cell of fixture.source.cells) {
      const id = source.physicalCells[cell.id]!, mass = values[density + id]! * cell.volume;
      before[0]! += mass; before[1]! += values[gamma + id]! * cell.volume;
      for (let axis = 0; axis < 3; axis++) before[axis + 2]! += mass * values[velocity + 4 * id + axis]!;
    }
    for (const cell of fixture.target.cells) {
      const value = result.cells[cell.id]!, mass = value.density * cell.volume;
      after[0]! += mass; after[1]! += value.gamma * cell.volume;
      for (let axis = 0; axis < 3; axis++) after[axis + 2]! += mass * value.velocity[axis]!;
      let cpuPlanMass = 0;
      for (let at = plan.cellOffsets[cell.id]!; at < plan.cellOffsets[cell.id + 1]!; at++) {
        const id = plan.cellSources[at]!;
        if (id !== 0xffffffff) cpuPlanMass += plan.cellVolumes[at]! * values[density + source.physicalCells[id]!]!;
      }
      assert.ok(Math.abs(mass - cpuPlanMass) < 1e-10, "brute force and independent CPU sparse overlap must agree");
    }
    const sourceVolume = fixture.source.cells.reduce((sum, cell) => sum + cell.volume, 0);
    const targetVolume = fixture.target.cells.reduce((sum, cell) => sum + cell.volume, 0);
    before[1]! += targetVolume - sourceVolume; // Newly admitted air has gamma one.
    after.forEach((value, axis) => assert.ok(Math.abs(value - before[axis]!) < 1e-9));
    assert.ok(result.faces.every(Number.isFinite));
    if (fixture.source === fixture.target) {
      for (const row of fixture.target.gradientRows) assert.equal(result.faces[row.id],
        values[(face ? source.layout.faceOtherOffset : source.layout.faceOffset) + source.physicalRows[row.id]!]!);
    }
  }
});

test("signed growth oracle rejects unproven newly admitted air", () => {
  const fixture = transferFixtures().find(value => value.air)!;
  const source = transferSourceRecipe(fixture.source);
  const values = transferSourceValues(fixture.source, source.layout, source.physicalCells, source.physicalRows);
  assert.throws(() => referenceGenerationTransfer(fixture.source, fixture.target, values, source.layout,
    source.physicalCells, source.physicalRows, 0, 0), /missing explicit new-air coverage/);
});
