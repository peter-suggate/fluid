/**
 * CPU solid-reach fallback benchmark, excluding asset loading and publication.
 * The default GPU/DMC construction paths bypass this CPU brick filter; these
 * timings are not default scene-build or GPU timings.
 *
 * node --import tsx tools/benchmark-svo-solid-reach.ts
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import { createBackyardPoolPartyScene } from "../lib/core/backyard-pool-party-scene";
import { meshAssetUrl, registerMeshAsset, sceneMeshAssetIds } from "../lib/core/imported-assets/asset-library";
import { decodeMeshVolume } from "../lib/core/imported-assets/mesh-volume";
import { sampleSvoPrimitive } from "../lib/svo/contracts/svo-primitive-abi";
import { buildSvoScenePrimitives, svoScenePrimitiveSolidReach } from "../lib/svo/features/scene-publication/svo-scene-primitives";

const scene = createBackyardPoolPartyScene();
for (const asset of sceneMeshAssetIds(scene)) {
  registerMeshAsset(asset, decodeMeshVolume(gunzipSync(readFileSync(`public${meshAssetUrl(asset)}`))));
}
const build = buildSvoScenePrimitives(scene);
const candidate = svoScenePrimitiveSolidReach(build);
const points = build.metadata.map(entry => {
  const { min, max } = entry.coverageBounds.conservative_m;
  return Array.from({ length: 128 }, (_, i) => ({
    x: min.x + (max.x - min.x) * ((i * 73 % 127) / 126),
    y: min.y + (max.y - min.y) * ((i * 41 % 127) / 126),
    z: min.z + (max.z - min.z) * ((i * 109 % 127) / 126),
  }));
});
const reference = (index: number, p: { x: number; y: number; z: number }) =>
  sampleSvoPrimitive(build.descriptors[index]!, p, ref => build.clusterPackings[ref]).signedDistance_m;
for (let index = 0; index < points.length; index++) for (const point of points[index]!) {
  assert.equal(candidate[index]!.distance_m(point.x, point.y, point.z), reference(index, point));
}
function run(prepared: boolean): number {
  const start = performance.now();
  let checksum = 0;
  for (let index = 0; index < points.length; index++) for (const point of points[index]!) {
    checksum += prepared ? candidate[index]!.distance_m(point.x, point.y, point.z) : reference(index, point);
  }
  assert.ok(Number.isFinite(checksum));
  return performance.now() - start;
}
run(false); run(true);
const before_ms: number[] = [], after_ms: number[] = [];
for (let round = 0; round < 4; round++) {
  for (const prepared of [false, true, true, false]) (prepared ? after_ms : before_ms).push(run(prepared));
}
const median = (values: number[]) => {
  const sorted = [...values].sort((a, b) => a - b);
  return .5 * (sorted[sorted.length / 2 - 1]! + sorted[sorted.length / 2]!);
};
console.log(JSON.stringify({ scene: scene.sceneId, lane: "CPU solid-reach fallback only",
  primitiveCount: build.descriptors.length, queriesPerSample: points.length * 128,
  bitExactParity: true, before_ms, after_ms,
  beforeMedian_ms: median(before_ms), afterMedian_ms: median(after_ms),
  speedup: median(before_ms) / median(after_ms),
}, null, 2));
