import assert from "node:assert/strict";
import test from "node:test";
import { canUseOpaqueRasterLighting } from "../lib/svo/features/shading/deferred-specialization";
import type { SparseVoxelDrySceneData } from "../lib/svo/contracts/scene-publication";
import { buildSvoPrimitiveCandidates } from "../lib/svo/features/scene-publication/svo-primitive-candidates";
import { SvoRasterAo } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
import { rasterSpotLights } from "../lib/svo/features/lighting-visibility/raster-spot-lights";

const sunScene = (): SparseVoxelDrySceneData => {
  const lightRecords = new Uint32Array(28);
  lightRecords[24] = 1;
  new Float32Array(lightRecords.buffer)[5] = 1;
  return { renderRevision: 1, primitiveRecords: new Uint32Array(), primitiveCandidates: buildSvoPrimitiveCandidates([]), ownerBase: 1,
    materialRecords: new Uint32Array(), materialRevision: 1, opaqueSurfaceOnly: true, lightRecords };
};

test("opaque raster specialization falls back for unsupported live publications", () => {
  const options = { coneMode: "raster-ao", globalIllumination: false };
  const scene = sunScene();
  assert.equal(canUseOpaqueRasterLighting(scene, options), true);
  for (const changed of [undefined, { ...scene, opaqueSurfaceOnly: false },
    { ...scene, lightRecords: undefined }, { ...scene, lightRecords: new Uint32Array(56) }]) {
    assert.equal(canUseOpaqueRasterLighting(changed, options), false);
  }
  assert.equal(canUseOpaqueRasterLighting(scene, { ...options, coneMode: "cones" }), false);
  assert.equal(canUseOpaqueRasterLighting(scene, { ...options, globalIllumination: true }), false);
  scene.lightRecords![24] = 2;
  assert.equal(canUseOpaqueRasterLighting(scene, options), false);
  scene.lightRecords![24] = 1;
  new Float32Array(scene.lightRecords!.buffer)[5] = NaN;
  assert.equal(canUseOpaqueRasterLighting(scene, options), false);
});

test("spot maps and shader capability agree on capacity and unsupported projections", () => {
  const scene = sunScene(), options = { coneMode: "raster-ao", globalIllumination: false };
  const lights = new Uint32Array(28 * 6), f = new Float32Array(lights.buffer);
  lights.set(scene.lightRecords!);
  for (let i = 1; i < 6; i++) {
    lights[i * 28 + 24] = 5;
    f.set([0, 4, 0, 20, 0, -2, 0, .8], i * 28);
    f[i * 28 + 20] = .2;
  }
  const four = lights.subarray(0, 28 * 5);
  assert.deepEqual(rasterSpotLights(four).map(s => s.index), [1, 2, 3, 4]);
  assert.deepEqual(rasterSpotLights(four)[0]!.direction, [0, -1, 0]);
  assert.equal(canUseOpaqueRasterLighting({ ...scene, lightRecords: four }, options), true);
  assert.equal(canUseOpaqueRasterLighting({ ...scene, lightRecords: lights }, options), false, "fifth spot keeps exact fallback");
  for (const [word, value] of [[7, -.1], [3, .1], [20, NaN], [5, 0]] as const) {
    const invalid = four.slice(); new Float32Array(invalid.buffer)[28 + word] = value;
    assert.equal(canUseOpaqueRasterLighting({ ...scene, lightRecords: invalid }, options), false);
    assert.deepEqual(rasterSpotLights(invalid).map(s => s.index), [2, 3, 4]);
  }
  const spotOnly = four.slice(28, 56);
  assert.equal(canUseOpaqueRasterLighting({ ...scene, lightRecords: spotOnly }, options), true);
});
