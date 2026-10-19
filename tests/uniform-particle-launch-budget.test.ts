import assert from "node:assert/strict";
import test from "node:test";
import { UniformColdFrame } from "../lib/methods/uniform/uniform-cold-frame";
import { UniformNarrowBandFlip } from "../lib/methods/uniform/uniform-narrow-band-flip";
import { NARROW_BAND_PARTICLE_GRID, NARROW_BAND_QUIET_PARTICLE_GRID } from "../lib/methods/uniform/uniform-narrow-band-order";

test("particle launch budget follows conservative frame evidence and wakes on triggers", () => {
  const order = { workgroups: 0 };
  const particles = Object.assign(Object.create(UniformNarrowBandFlip.prototype), { order });
  const cold = new UniformColdFrame();
  const check = (expected: number) => {
    particles.beginAdvance(cold);
    assert.equal(order.workgroups, expected);
    assert.equal(Reflect.get(particles, "particleGroups"), expected);
    assert.ok(expected > 0, "a launch must still cover newly arrived GPU work");
  };
  cold.begin(1, 0, undefined, false); check(NARROW_BAND_PARTICLE_GRID);
  cold.begin(3, 1, new Uint32Array([10, 5, 0, 0, 0, 0]), false); check(NARROW_BAND_QUIET_PARTICLE_GRID);
  cold.begin(4, 2, new Uint32Array([10, 5, 0, 0, 0, 1]), false); check(NARROW_BAND_PARTICLE_GRID);
  cold.begin(5, 3, new Uint32Array([10, 5, 0, 0, 1, 0]), false); check(NARROW_BAND_PARTICLE_GRID);
  cold.trigger(); cold.begin(6, 4, new Uint32Array(6), false); check(NARROW_BAND_PARTICLE_GRID);
  cold.begin(8, 6, new Uint32Array([10, 0, 0, 0, 0, 0]), false); check(NARROW_BAND_QUIET_PARTICLE_GRID);
});
