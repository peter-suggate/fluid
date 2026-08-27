import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  createSparseCM12FramePlanPresentationInitialWords,
  createSparseCM12FramePlanPresentationLayout,
} from "../lib/methods/adaptive-mass/sparse-cm12-frame-plan-presentation";
import {
  createSparseCM12WorldDirectoryLayout,
  createSparseCM12WorldDirectoryWGSL,
} from "../lib/methods/adaptive-mass/sparse-cm12-world-directory";
import { sparseCM12PresentationPageAllocatorWGSL } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const invalid = 0xffff_ffff;

test("presentation pages start in a reclaimable lowest-page-first free list", () => {
  const layout = createSparseCM12FramePlanPresentationLayout({
    baseWords: 64,
    pageCapacity: 8,
    brickCapacity: 8,
    brickFineResolution: 8,
    pageResolution: 8,
    packetIndex: 5,
  });
  const words = createSparseCM12FramePlanPresentationInitialWords(layout, {
    brickPages: [0, 1, invalid, invalid, invalid, invalid, invalid, invalid],
  });
  const allocator = layout.allocatorBaseWords - layout.baseWords;
  const freeList = layout.allocatorFreeListBaseWords - layout.baseWords;

  assert.equal(words[allocator], 2);
  assert.equal(words[allocator + 5], 6);
  assert.deepEqual(Array.from(words.slice(freeList, freeList + 6)),
    [7, 6, 5, 4, 3, 2]);
  assert.equal(words[freeList + words[allocator + 5]! - 1], 2,
    "the first pop must reuse the lowest unallocated physical page");
});

test("world leaves use tombstones and recycle dynamic leaf identities", () => {
  const layout = createSparseCM12WorldDirectoryLayout({
    initialLeaves: 2,
    growthLeaves: 6,
    maximumSpanLog: 3,
  });
  const wgsl = createSparseCM12WorldDirectoryWGSL(layout);

  assert.equal(layout.totalWords, layout.freeListBaseWords + 6);
  assert.match(wgsl, /fn cm12WorldReleaseLeaf\(leaf:u32\)->bool/);
  assert.match(wgsl, /state==0u\|\|state==3u/,
    "allocation must claim either a virgin slot or a tombstone");
  assert.match(wgsl, /CM12_WDR_FREE_LIST\+free-1u/,
    "allocation must pop a reclaimed dynamic leaf before growing high water");
  assert.match(wgsl, /atomicStore\(&topologyArena\[at\+0u\],3u\)/,
    "release must preserve the probe chain with a tombstone");
});

test("retirement follows all-air publication and clears the recorded topology page", () => {
  const presentation = createSparseCM12FramePlanPresentationLayout({
    baseWords: 64,
    pageCapacity: 8,
    brickCapacity: 8,
    brickFineResolution: 8,
    pageResolution: 8,
    packetIndex: 5,
  });
  const world = createSparseCM12WorldDirectoryLayout({
    initialLeaves: 2,
    growthLeaves: 6,
    maximumSpanLog: 3,
  });
  const shader = sparseCM12PresentationPageAllocatorWGSL(
    8, 2, 0, presentation, world,
  );
  assert.match(shader, /if\(true\)\{/,
    "a WDR-backed allocator must always use the signed SparseWorld key path");
  assert.match(shader,
    /key=u32\(coordinate\.x\+1024\)\|\(u32\(coordinate\.y\+512\)<<11u\)\s*\|\(u32\(coordinate\.z\+1024\)<<21u\)/,
  "authored and dynamic WDR leaves must share the signed presentation-key ABI");
  assert.doesNotMatch(shader,
    /key=u32\(coordinate\.x\)\+\d+u\s*\*\(u32\(coordinate\.y\)/,
  "a WDR-backed allocator must not retain the dense atlas-key fallback");
  assert.match(shader,
    /let topologyPage=atomicLoad\(&activity\[activityRecord\+37u\]\)/);
  assert.doesNotMatch(shader, /topologyPage=brick-CM12_WDR_INITIAL_LEAVES/);

  const host = readFileSync(new URL(
    "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
    import.meta.url,
  ), "utf8");
  const stage = host.slice(host.indexOf('stage("presentation-publication"'),
    host.indexOf('stage("presentation-publication"') + 3_000);
  const publish = stage.indexOf("this.encodeFramePlanPresentation");
  const retire = stage.indexOf('dispatch("retireSparseCM12PresentationPages"');
  const compact = stage.indexOf('dispatch("compactSparseCM12PresentationPageDirectory"');
  const commit = stage.indexOf('dispatch("commitSparseCM12FrameControl"');
  assert.ok(publish >= 0 && publish < retire && retire < compact && compact < commit,
    "retirement must follow all-air FPP publication and precede frame acceptance");
});
