import assert from "node:assert/strict";
import test from "node:test";
import { NBFLIP_LETTERS_SCENE_ID } from "../lib/core/nbflip-paper-scenes";
import { sceneDocument } from "../lib/core/scene-definition";
import { getSceneDefinition } from "../lib/core/scenes";
import type { UniformFineSlotWords } from "../lib/methods/uniform/uniform-fine-slots";
import { createUniformSolver, withUniformDevice } from "./helpers/uniform-geometric";

// The build's h-slot claim (docs/plans/bulk16-plan-2026-10-28.md, M4b): under
// a coarse start with far fewer h slots than the band asks for, the build
// keeps at 4h the tiles no free slot serves instead of failing. Every block
// an admitted h tile's ring reaches holds a slot, a claim never takes a slot
// a block still needs, and the free stack and the slots in use add up to the
// capacity.
type Detail = { slotState: GPUBuffer; slotClaims: { words: UniformFineSlotWords; capacity: number }; fineSlots: { capacity: number } };

async function readState(device: GPUDevice, detail: Detail): Promise<Uint32Array> {
  const words = detail.slotClaims.words.words, staging = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const encoder = device.createCommandEncoder();encoder.copyBufferToBuffer(detail.slotState, 0, staging, 0, 4 * words);device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);const state = new Uint32Array(staging.getMappedRange().slice(0));staging.unmap();staging.destroy();
  return state;
}

test("the build's slot claim refuses to 4h past its free slots: no fatal, every admitted ring slotted, no slot taken from a block that needs it", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("slot claim", async device => {
    const scene = structuredClone(sceneDocument(getSceneDefinition(NBFLIP_LETTERS_SCENE_ID)));
    scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
    // Eight slots (one layer of the slot grid): the letters' band asks for far more.
    scene.fluid.tiers = { ...scene.fluid.tiers, start: "4h", bricks: 512, fineBlocks: 8 };
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const detail = (solver as unknown as { detailStorage: Detail }).detailStorage, w = detail.slotClaims.words, capacity = detail.fineSlots.capacity;
      assert.ok(capacity >= 8 && capacity < 64, `a few slots: ${capacity}`);
      let last: Uint32Array | undefined, kept = 0;
      for (let step = 1; step <= 24; step++) {
        assert.ok(solver.advanceTo(step / 60, []));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `step ${step}`);
        const state = await readState(device, detail), blocks = w.need;
        const free = state[w.free]!, slots = new Set<number>();
        for (let b = 0; b < blocks; b++) {
          const s = state[w.slot + b]!;
          // The adopted generation's rings hold slots (the claim after the build met nothing new).
          if (state[w.need + b] && b >= 1 && b < blocks - 1) assert.notEqual(s, 0, `step ${step}: block ${b} is in an admitted ring with no slot`);
          if (!s) continue;
          assert.ok(s <= capacity && !slots.has(s), `step ${step}: slot ${s} of block ${b} is past the capacity or given twice`);slots.add(s);
          // A block that held a slot and still needs one keeps it.
          if (last && last[w.slot + b] && state[w.need + b]) { assert.equal(s, last[w.slot + b], `step ${step}: block ${b} moved slot`);kept++; }
        }
        assert.equal(slots.size + free, capacity, `step ${step}: slots in use and free`);
        last = state;
      }
      const info = solver.info;
      assert.equal(info.uniformMixedSlotCapacity, capacity);
      // The free slots ran out: fewer were left than one region's tiles ask for (at most eight blocks, two an axis).
      assert.ok(capacity - (info.uniformMixedSlotPeak ?? 0) < 8, `the claim ran the free stack out: ${info.uniformMixedSlotPeak} of ${capacity}`);
      assert.ok((info.uniformMixedRefusedTileBuilds ?? 0) > 0, "tiles were refused");
      assert.ok((info.uniformMixedFineTiles ?? 0) > 0, "and some admitted");
      assert.ok(kept > 0, "slots were held across frames");
    } finally { solver.destroy(); }
  });
});

// A build asking for more h tiles than the owners hold (fluid.tiers.bricks
// 512 held 4096 against letters' 4219: "a build needs 4219 h tiles", a
// fatal) is cut by priority instead: the tiles below the tile cut stay 4h.
test("the build's tile cut keeps the h tiles at the owners' capacity: no fatal, refusals reported", { skip: !process.env.WEBGPU_NODE_MODULE }, async () => {
  await withUniformDevice("tile cut", async device => {
    const scene = structuredClone(sceneDocument(getSceneDefinition(NBFLIP_LETTERS_SCENE_ID)));
    scene.numerics = { ...scene.numerics, fixedDt_s: 1 / 60, maxDt_s: 1 / 60 };
    // 64 bricks: 512 h tiles; slots enough that only the tile cut binds.
    scene.fluid.tiers = { ...scene.fluid.tiers, start: "4h", bricks: 64, fineBlocks: 256 };
    const solver = await createUniformSolver(device, scene, { volumeControlSeconds: 0 });
    try {
      const detail = (solver as unknown as { detailStorage: Detail }).detailStorage, capacity = detail.fineSlots.capacity;
      let most = 0;
      for (let step = 1; step <= 24; step++) {
        assert.ok(solver.advanceTo(step / 60, []));
        await solver.awaitFrameCompletion();
        assert.equal(solver.info.simulationPipelineError, undefined, `step ${step}`);
        const fine = solver.info.uniformMixedFineTiles ?? 0;
        assert.ok(fine <= 512, `step ${step}: ${fine} h tiles past the 512 the owners hold`);most = Math.max(most, fine);
      }
      const info = solver.info;
      assert.ok((info.uniformMixedRefusedTileBuilds ?? 0) > 0, "tiles were refused");
      assert.ok(most > 256, `the cut serves up to the capacity: ${most}`);
      assert.ok((info.uniformMixedSlotPeak ?? 0) < capacity, `the slots did not bind: ${info.uniformMixedSlotPeak} of ${capacity}`);
    } finally { solver.destroy(); }
  });
});
