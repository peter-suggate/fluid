import assert from "node:assert/strict";
import test from "node:test";

import { planSvoNodeMipPyramid } from "../lib/svo-node-mip-pyramid";
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import {
  SVO_STATIC_SHADOW_FIELD_LAYOUT,
  SvoStaticShadowFieldCache,
  planSvoStaticShadowField,
  svoStaticShadowFieldWGSL,
} from "../lib/svo-static-shadow-field";
import { getScenePreset } from "../lib/scenes";

function inputs(generation = 7, lightRevision = 11) {
  const nodeMips = planSvoNodeMipPyramid({
    generation,
    occupiedPages: [[0, 0, 0], [1, 0, 0]],
    levelCount: 2,
    atlasPages: [4, 1, 1],
  });
  const lights = buildSvoSceneLights(getScenePreset("hose-tank").create(), { revision: lightRevision, maximumRecords: 3 });
  return { nodeMips, lights, plan: planSvoStaticShadowField(nodeMips, lights) };
}

test("static shadow planner shares node-mip slots and costs exactly two words per atlas page", () => {
  const { nodeMips, lights, plan } = inputs();
  assert.equal(plan.sourceGeneration, nodeMips.generation);
  assert.equal(plan.lightRevision, lights.revision);
  assert.deepEqual(plan.pages.map(({ slot }) => slot), nodeMips.pages.map(({ slot }) => slot));
  assert.equal(plan.lightChannels.length, lights.records.length);
  assert.deepEqual(plan.lightChannels.map(({ lightId }) => lightId), lights.records.map(({ lightId }) => lightId));
  assert.equal(plan.allocatedBytes, nodeMips.atlas.capacity * 8);
  assert.equal(plan.residentPayloadBytes, nodeMips.residentPageCount * 8);
  assert.equal(plan.maximumReplacementBytes, plan.allocatedBytes * 2);
  assert.equal(SVO_STATIC_SHADOW_FIELD_LAYOUT.maximumLights, 32);
  assert.equal(SVO_STATIC_SHADOW_FIELD_LAYOUT.bytesPerPage, 8);
});

test("cache keys change exactly with structural or authored-light revisions", () => {
  const first = inputs(7, 11).plan;
  assert.equal(inputs(7, 11).plan.cacheKey, first.cacheKey);
  assert.notEqual(inputs(8, 11).plan.cacheKey, first.cacheKey);
  assert.notEqual(inputs(7, 12).plan.cacheKey, first.cacheKey);

  const { nodeMips, lights } = inputs(7, 11);
  const mutated = lights.packedRecords.slice();
  mutated[0] ^= 1;
  assert.notEqual(planSvoStaticShadowField(nodeMips, { ...lights, packedRecords: mutated }).cacheKey, first.cacheKey,
    "the packed GPU light content participates even if a caller fails to change staticRevision");
});

test("dirty and mixed channels fall back to exact current-frame tracing while certificates take safe fast paths", () => {
  const { plan, lights } = inputs();
  const cache = new SvoStaticShadowFieldCache();
  assert.deepEqual(cache.begin(plan), {
    reused: false,
    dirtyPages: plan.residentPageCount,
    dirtyChannels: plan.residentPageCount * plan.lightChannels.length,
  });
  assert.deepEqual(cache.dirtyWork(2).map(({ slot, lightChannel }) => [slot, lightChannel]), [[0, 0], [0, 1]],
    "bounded builder work is deterministic page-major/light-index order");
  const page = plan.pages[0].key;
  const [clear, blocked, mixed] = lights.records;
  assert.deepEqual(cache.resolve(plan, page, clear.lightId), {
    certificate: "unknown",
    staticAction: "trace-static-current-frame",
    dynamicAction: "trace-current-frame-overlay",
  });
  cache.publishPageCertificates({ page, certificates: [
    { lightId: clear.lightId, certificate: "visible" },
    { lightId: blocked.lightId, certificate: "occluded" },
    { lightId: mixed.lightId, certificate: "mixed" },
  ] });
  assert.equal(cache.dirtyWork().length,
    (plan.residentPageCount - 1) * plan.lightChannels.length,
    "publishing a page removes precisely its completed channels from the work queue");
  assert.deepEqual(cache.resolve(plan, page, clear.lightId), {
    certificate: "visible", staticAction: "skip-static", dynamicAction: "trace-current-frame-overlay",
  });
  assert.deepEqual(cache.resolve(plan, page, blocked.lightId), {
    certificate: "occluded", staticAction: "reject-light", dynamicAction: "skip-dynamic",
  });
  assert.deepEqual(cache.resolve(plan, page, mixed.lightId), {
    certificate: "mixed", staticAction: "trace-static-current-frame", dynamicAction: "trace-current-frame-overlay",
  });
  assert.equal(cache.publish().reason, "dirty-pages");
  assert.equal(cache.telemetry().readyPages, 1);
});

test("page-granular completion publishes atomically and reuses only an exact revision", () => {
  const { plan } = inputs();
  const cache = new SvoStaticShadowFieldCache();
  cache.begin(plan);
  for (const page of plan.pages) cache.publishPageMixed(page.keyString);
  const published = cache.publish();
  assert.equal(published.published, true);
  assert.equal(published.reason, "published");
  assert.deepEqual(cache.begin(plan), { reused: true, dirtyPages: 0, dirtyChannels: 0 });
  assert.equal(cache.publish().reason, "already-visible");
  assert.deepEqual(cache.telemetry(), {
    visibleCacheKey: plan.cacheKey,
    candidateCacheKey: undefined,
    residentPages: plan.residentPageCount,
    readyPages: plan.residentPageCount,
    dirtyPages: 0,
    readyChannels: plan.residentPageCount * plan.lightChannels.length,
    dirtyChannels: 0,
    allocatedBytes: plan.allocatedBytes,
    fallback: "none",
  });
});

test("a changed scene cannot consume stale visible certificates while its candidate is dirty", () => {
  const first = inputs(7, 11).plan;
  const next = inputs(8, 11).plan;
  const cache = new SvoStaticShadowFieldCache();
  cache.begin(first);
  for (const page of first.pages) cache.publishPageMixed(page.keyString);
  assert.equal(cache.publish().published, true);

  cache.begin(next);
  const page = next.pages[0];
  const light = next.lightChannels[0];
  assert.equal(cache.resolve(next, page.keyString, light.lightId).certificate, "unknown",
    "the prior generation must not be interpreted as current");
  assert.equal(cache.telemetry().fallback, "exact-current-frame-trace");
  assert.equal(cache.telemetry().allocatedBytes, first.allocatedBytes + next.allocatedBytes);
});

test("publication detects revision collisions instead of accepting changed content under old stamps", () => {
  const { nodeMips, lights, plan } = inputs();
  const cache = new SvoStaticShadowFieldCache();
  cache.begin(plan);
  for (const page of plan.pages) cache.publishPageMixed(page.keyString);
  assert.equal(cache.publish().published, true);

  const mutated = lights.packedRecords.slice();
  mutated[0] ^= 1;
  const collision = planSvoStaticShadowField(nodeMips, { ...lights, packedRecords: mutated });
  cache.begin(collision);
  for (const page of collision.pages) cache.publishPageMixed(page.keyString);
  assert.equal(cache.publish().reason, "revision-collision");
  assert.equal(cache.resolve(collision, collision.pages[0].keyString, collision.lightChannels[0].lightId).certificate, "mixed",
    "candidate data remains exact and usable even though it cannot replace the visible cache identity");
});

test("fixed WGSL ABI represents four conservative states without scene-specialized constants", () => {
  assert.match(svoStaticShadowFieldWGSL, /struct SvoStaticShadowPage/);
  assert.match(svoStaticShadowFieldWGSL, /visibleMask:u32/);
  assert.match(svoStaticShadowFieldWGSL, /occludedMask:u32/);
  assert.match(svoStaticShadowFieldWGSL, /1u<<lightIndex/);
  assert.doesNotMatch(svoStaticShadowFieldWGSL, /array<[^>]+,\s*\d+>/);
});

test("planner rejects incomplete topology and mismatched per-record light revisions", () => {
  const { lights } = inputs();
  const incomplete = planSvoNodeMipPyramid({ generation: 1, occupiedPages: [[0, 0, 0], [9, 9, 9]], levelCount: 2, capacity: 1 });
  assert.throws(() => planSvoStaticShadowField(incomplete, lights), /complete node-mip topology/);
  assert.throws(() => planSvoStaticShadowField(inputs().nodeMips, {
    ...lights,
    records: lights.records.map((light, index) => index === 0 ? { ...light, revision: lights.revision + 1 } : light),
  }), /does not match publication revision/);
});
