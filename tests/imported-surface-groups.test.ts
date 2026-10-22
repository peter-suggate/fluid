import assert from "node:assert/strict";
import test from "node:test";
import { planImportedSurfaceGroups } from "../lib/svo/features/primary-visibility/imported-surface-groups";

test("native draw groups reserve disjoint full-population segments at every asset LOD", () => {
  const instances = ["oak", "rock", "oak", "fern", "rock", "oak"].map(asset => ({ asset }));
  const levels = {
    oak: [{ first: 0, count: 90 }, { first: 90, count: 30 }, { first: 120, count: 12 }],
    rock: [{ first: 0, count: 24 }, { first: 24, count: 6 }],
    fern: [{ first: 0, count: 48 }],
  };
  const plan = planImportedSurfaceGroups(instances, asset => levels[asset as keyof typeof levels]);
  assert.deepEqual([...plan.firstGroupByAsset], [["oak", 0], ["rock", 3], ["fern", 5]]);
  assert.equal(plan.instanceCapacity, 3 * 3 + 2 * 2 + 1);
  let end = 0;
  for (const group of plan.groups) {
    assert.equal(group.instanceOffset, end, "segments neither overlap nor leave wasted gaps");
    assert.equal(group.capacity, instances.filter(instance => instance.asset === group.asset).length);
    assert.deepEqual({ first: group.first, count: group.count }, levels[group.asset as keyof typeof levels][group.level]);
    assert.equal(plan.groups[plan.firstGroupByAsset.get(group.asset)! + group.level], group);
    end += group.capacity;
  }
  assert.equal(end, plan.instanceCapacity);
  // Every instance may choose the same LOD without changing allocations.
  for (const group of plan.groups) {
    const selected = instances.filter(instance => instance.asset === group.asset);
    assert.ok(group.instanceOffset + selected.length <= group.instanceOffset + group.capacity);
  }
});

test("native group planning handles an empty set and all seven packed LOD slots", () => {
  const empty = planImportedSurfaceGroups([], () => { throw new Error("empty scene should not resolve assets"); });
  assert.equal(empty.groups.length, 0); assert.equal(empty.firstGroupByAsset.size, 0); assert.equal(empty.instanceCapacity, 0);
  const levels = Array.from({ length: 7 }, (_, level) => ({ first: level * 100, count: 99 - level * 9 }));
  const plan = planImportedSurfaceGroups([{ asset: "tree" }, { asset: "tree" }], () => levels);
  assert.equal(plan.groups.length, 7); assert.equal(plan.instanceCapacity, 14);
  assert.equal(plan.groups[6]!.level, 6); assert.equal(plan.groups[6]!.capacity, 2);
});

test("native draw groups reject malformed LOD ranges before uint32 GPU packing", () => {
  const make = (levels: { first: number; count: number }[]) => planImportedSurfaceGroups([{ asset: "tree" }], () => levels);
  assert.throws(() => make([]), /1–7/);
  assert.throws(() => make(Array.from({ length: 8 }, () => ({ first: 0, count: 3 }))), /1–7/);
  for (const bad of [-1, .5, NaN, Infinity, 0x1_0000_0000, Number.MAX_SAFE_INTEGER]) {
    assert.throws(() => make([{ first: bad, count: 3 }]), /invalid index range/);
    assert.throws(() => make([{ first: 0, count: bad }]), /invalid index range/);
  }
});
