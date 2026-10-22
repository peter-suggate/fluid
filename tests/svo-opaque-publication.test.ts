import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { SVO_MATERIAL_FLAGS, SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";
import { publishOpaqueSurfaceCapability } from "../lib/svo/features/shading/deferred-specialization";

test("opaque publication reuses untouched page material sets but rechecks painted cells and live tables", () => {
  const base = { ...createEmptyScene(), solidVoxels: [
    { operation: "fill" as const, minimum: [0, 0, 0] as const, maximumExclusive: [1, 1, 1] as const, materialId: 1 },
    { operation: "fill" as const, minimum: [16, 0, 0] as const, maximumExclusive: [17, 1, 1] as const, materialId: 1 },
  ] };
  const table = new Uint32Array(16 * SVO_MATERIAL_RECORD_WORDS), primitives = new Uint32Array();
  for (let id = 1; id < 16; id++) table[id * SVO_MATERIAL_RECORD_WORDS + 23] = SVO_MATERIAL_FLAGS.opaque;
  table[9 * SVO_MATERIAL_RECORD_WORDS + 23] = SVO_MATERIAL_FLAGS.dielectric;
  const world = solidWorldForScene(base);
  assert.equal(publishOpaqueSurfaceCapability(base, table, primitives), true);
  const untouched = world.pages[1]!;
  // An unrelated stroke may reuse the material set, but may not scan its cells.
  Object.defineProperty(untouched.materialId, Symbol.iterator, { configurable: true,
    value: () => { throw new Error("unchanged page material cells were rescanned"); } });
  try {
    const painted = sceneWithSolidStroke(base, [{ operation: "paint", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1], materialId: 9 }]);
    assert.equal(solidWorldForScene(painted).pages[1], untouched);
    assert.notEqual(solidWorldForScene(painted).pages[0], world.pages[0]);
    assert.equal(publishOpaqueSurfaceCapability(painted, table, primitives), false, "new dielectric material disables opaque specialization");
    const opaque = table.slice(); opaque[9 * SVO_MATERIAL_RECORD_WORDS + 23] = SVO_MATERIAL_FLAGS.opaque;
    assert.equal(publishOpaqueSurfaceCapability(painted, opaque, primitives), true, "cached IDs must use the current material table");
    assert.equal(publishOpaqueSurfaceCapability(painted, table, primitives), false);
    const cleared = sceneWithSolidStroke(painted, [{ operation: "clear", minimum: [0, 0, 0], maximumExclusive: [1, 1, 1] }]);
    assert.equal(publishOpaqueSurfaceCapability(cleared, table, primitives), true, "removing the painted cell removes its material from the snapshot");
    assert.equal(publishOpaqueSurfaceCapability(base, table, primitives), true, "undo reuses the original opaque snapshot");
  } finally { Reflect.deleteProperty(untouched.materialId, Symbol.iterator); }
});
