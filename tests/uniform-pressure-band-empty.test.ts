import assert from "node:assert/strict";
import test from "node:test";
import { UniformPressureBand } from "../lib/methods/uniform/uniform-pressure-band";
import { UniformColdFrame } from "../lib/methods/uniform/uniform-cold-frame";

test("coarse WARM frames clear band membership without launching a fine solve", () => {
  const coldFrame = new UniformColdFrame();
  coldFrame.begin(1, 0, undefined, false);
  coldFrame.begin(3, 1, new Uint32Array([10, 5, 0, 0, 0, 0]), false);
  assert.equal(coldFrame.ceiling, 1);
  const band = Object.assign(Object.create(UniformPressureBand.prototype), {
    simulation: { capacity: { fineTiles: 512 } }, fields: { coldFrame, warmTiers: {}, freeze: {} }, index: {}, listSlots: 512,
  }) as UniformPressureBand;
  const clears: unknown[][] = [];
  const encoder = { clearBuffer: (...args: unknown[]) => clears.push(args),
    beginComputePass: () => { throw new Error("An empty band must not launch"); } } as unknown as GPUCommandEncoder;
  band.encodePrepare(encoder);
  band.encodeSolve(encoder);
  assert.equal(clears.length, 2, "both receipt header and slot membership must be reset");
  assert.equal(clears[1]![1], band.slotMapOffset);
  for (const mutate of [
    () => { Reflect.get(band, "fields").warmTiers = undefined; },
    () => { Reflect.get(band, "fields").warmTiers = {}; Reflect.get(band, "fields").freeze = undefined; },
    () => { Reflect.get(band, "fields").freeze = {}; coldFrame.trigger(); coldFrame.begin(4, 2, new Uint32Array(6), false); },
  ]) {
    mutate();
    assert.equal(Reflect.get(band, "empty"), false, "fine WARM, unfrozen COLD and triggered frames need the band");
  }
});

test("particles and mid-frame promotions keep the WARM band conservative", () => {
  for (const evidence of [new Uint32Array([10, 5, 0, 0, 1, 0]), new Uint32Array([10, 5, 0, 0, 0, 1])]) {
    const coldFrame = new UniformColdFrame();
    coldFrame.begin(1, 0, undefined, false); coldFrame.begin(3, 1, evidence, false);
    const band = Object.assign(Object.create(UniformPressureBand.prototype), {
      simulation: { capacity: { fineTiles: 512 } }, fields: { coldFrame, warmTiers: {}, freeze: {} },
    });
    assert.equal(Reflect.get(band, "empty"), false);
  }
});
