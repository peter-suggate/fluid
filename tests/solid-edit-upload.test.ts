import assert from "node:assert/strict";
import test from "node:test";
import { createEmptyScene } from "../lib/core/empty-scene";
import { sceneWithSolidStroke, solidWorldForScene } from "../lib/core/solid-world";
import { createSparseCM12SolidOccupancyLayout, packSparseCM12SolidOccupancy } from "../lib/methods/adaptive-mass/sparse-cm12-solid-occupancy";
import { prepareSolidEditUpload } from "../lib/methods/adaptive-mass/sparse-cm12-solid-edit-upload";

test("bounded GPU scatter reproduces canonical solids and clears obsolete directory entries", () => {
  const scene = createEmptyScene();
  const before = solidWorldForScene(scene);
  const edited = sceneWithSolidStroke(scene, [{ operation: "fill", minimum: [10, 8, 10], maximumExclusive: [12, 10, 12], materialId: 2 }]);
  const after = solidWorldForScene(edited);
  const layout = createSparseCM12SolidOccupancyLayout({ baseWords: 32, authoredPageCount: Math.max(before.pages.length, after.pages.length) });
  for (const [previous, next] of [[before, after], [after, before]]) {
    const upload = prepareSolidEditUpload(layout, previous!, next!).finish();
    const addresses = Array.from({ length: upload.length / 2 }, (_, index) => upload[2 * index]!);
    assert.equal(new Set(addresses).size, addresses.length, "parallel scatter must have unique destinations");
    const actual = new Uint32Array(layout.totalWords);
    actual.set(packSparseCM12SolidOccupancy(layout, previous!, [0, 0, 0]), layout.baseWords);
    for (let index = 0; index < upload.length; index += 2) actual[upload[index]!] = upload[index + 1]!;
    const expected = packSparseCM12SolidOccupancy(layout, next!, [0, 0, 0]);
    const liveWords = layout.pageBaseWords + next!.pages.length * layout.pageWords;
    assert.deepEqual(actual.slice(layout.baseWords, layout.baseWords + liveWords), expected.slice(0, liveWords));
  }
});

test("private solid scatter separates state addresses, deduplicates writes and rejects excessive work", () => {
  const world = solidWorldForScene(createEmptyScene());
  const layout = createSparseCM12SolidOccupancyLayout({ baseWords: 0, authoredPageCount: world.pages.length });
  const upload = prepareSolidEditUpload(layout, world, world);
  upload.add("state", 8, new Uint32Array([123, 456]));
  upload.add("state", 8, new Uint32Array([789]));
  const words = upload.finish();
  const index = words.findIndex((value, index) => index % 2 === 0 && value === 0x80000008);
  assert.equal(words[index + 1], 789);
  assert.throws(() => prepareSolidEditUpload(layout, world, world, 1), /bounded work budget/);
});
