import assert from "node:assert/strict";
import test from "node:test";

import { UniformMixedFrame } from "../lib/methods/uniform/uniform-mixed-frame";
import { UNIFORM_MIXED_REACHES, type UniformMixedCache, type UniformMixedHostChange } from "../lib/methods/uniform/uniform-mixed-invalidation";

const CACHES: readonly UniformMixedCache[] = ["cold", "workControl", "dynamics", "heat", "heldAuthority", "splitAuthority", "extension", "geometry",
  "solidRecord", "solidWidths", "workEpoch", "bandWork", "bandCycles", "ownershipWork", "residency", "remapWork"];

test("every mixed frame cache is reached by at least one host change", () => {
  const reached = new Set(Object.values(UNIFORM_MIXED_REACHES).flat());
  assert.deepEqual(CACHES.filter(cache => !reached.has(cache)), []);
  for (const [change, caches] of Object.entries(UNIFORM_MIXED_REACHES)) assert.equal(new Set(caches).size, caches.length, `${change} reaches a cache twice`);
});

test("the edges the scattered invalidations missed are in the table", () => {
  const reaches = (change: UniformMixedHostChange, cache: UniformMixedCache) => UNIFORM_MIXED_REACHES[change].includes(cache);
  // A fluid, value or scene write leaves region heat and the held authority's retained regions stale.
  for (const change of ["fluid", "values", "scene", "solids", "layout"] as const) {
    assert.ok(reaches(change, "heat"), `${change} → heat`);
    assert.ok(reaches(change, "heldAuthority"), `${change} → heldAuthority`);
    assert.ok(reaches(change, "extension") && reaches(change, "geometry") && reaches(change, "cold"), `${change} → extension, geometry, cold`);
  }
  assert.ok(reaches("census", "bandCycles"));
  assert.equal(UNIFORM_MIXED_REACHES.census[0], "workEpoch");
  // Bodies move every presented frame of an interaction scene: their head rebuilds heat and the held authority itself.
  assert.ok(!reaches("bodies", "heat") && !reaches("bodies", "heldAuthority"));
});

/** A frame whose caches record what reached them, behind the real invalidate and setRelayout. */
function recordingFrame() {
  const reached: string[] = [];
  const note = (name: string) => () => { reached.push(name); };
  const frame = Object.assign(Object.create(UniformMixedFrame.prototype), {
    phase: "ready", frameIndex: 7, workEpoch: 0, censused: false, fineCeiling: 64, liquidBand: false, status: {},
    reusableExtension: true, geometryCurrent: true, solidWidthsStale: false,
    cold: { trigger: note("cold") }, surfaceWorkControl: { invalidate: note("workControl") }, dynamics: { invalidate: note("dynamics") },
    heat: { invalidate: note("heat") }, authority: { invalidate: note("heldAuthority") }, split: { authority: { invalidate: note("splitAuthority") } },
    solid: { invalidate: note("solidRecord") },
    band: { capacity: 16, observeWork: note("bandWork"), forgetCycles: note("bandCycles") },
    ownership: { forgetWork: note("ownershipWork"), resetResidency: note("resetResidency") },
    levels: [{ ownership: { observeResidency: note("residency") } }], remap: { forgetWork: note("remapWork"), bindStatus: () => {} },
    reserveFine: () => undefined,
  }) as UniformMixedFrame;
  return { frame, state: frame as unknown as Record<string, unknown>, reached };
}

test("invalidate reaches exactly its row, and a census forgets the band's cycles at its epoch", () => {
  const { frame, state, reached } = recordingFrame();
  frame.invalidate("fluid");
  assert.deepEqual(reached, ["cold", "heat", "heldAuthority", "splitAuthority"]);
  assert.equal(state.reusableExtension, false);
  assert.equal(state.geometryCurrent, false);
  reached.length = 0;
  frame.invalidate("census", 3);
  assert.deepEqual(reached, ["bandWork", "bandCycles", "ownershipWork", "residency", "remapWork"]);
  assert.equal(state.workEpoch, 3);
});

test("an uncounted relayout attach forgets the band's cycles; a counted one keeps its evidence", () => {
  const relayout = {} as Parameters<UniformMixedFrame["setRelayout"]>[0];
  const uncounted = recordingFrame();
  uncounted.frame.setRelayout(relayout);
  assert.deepEqual(uncounted.reached, ["cold", "bandWork", "bandCycles", "ownershipWork", "residency", "remapWork"]);
  assert.equal(uncounted.state.workEpoch, 7);
  const counted = recordingFrame();
  counted.frame.setRelayout(relayout, 8, true);
  assert.deepEqual(counted.reached, ["cold"]);
});
