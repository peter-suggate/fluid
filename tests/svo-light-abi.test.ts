import assert from "node:assert/strict";
import test from "node:test";
import { environmentIds } from "../lib/environments";
import { getScenePreset } from "../lib/scenes";
import {
  buildSvoSceneLights,
  canonicalSvoLightRecord,
  packSvoLightRecords,
  SVO_LIGHT_KINDS,
  SVO_LIGHT_MAXIMUM_RECORDS,
  SVO_LIGHT_RECORD_STRIDE_BYTES,
  svoLightWGSL,
} from "../lib/svo-light-abi";

test("112-byte light ABI publishes a stable directional source for every environment", () => {
  assert.equal(SVO_LIGHT_RECORD_STRIDE_BYTES, 112);
  for (const environmentId of environmentIds) {
    const scene = getScenePreset("water-box-dam-break").create();
    scene.environment = environmentId;
    const result = buildSvoSceneLights(scene, { revision: 9 });
    assert.ok(result.records.length >= 1 && result.records.length <= SVO_LIGHT_MAXIMUM_RECORDS);
    assert.equal(result.records[0].kind, "directional");
    assert.equal(result.records[0].lightId, 1);
    assert.equal(result.records[0].revision, 9);
    assert.equal(result.packedRecords.byteLength, result.records.length * 112);
  }
});

test("emissive fixture tags become bounded area lights with stable owner identities", () => {
  const scene = getScenePreset("sphere-jet").create();
  scene.environment = "night-lab";
  const first = buildSvoSceneLights(scene, { revision: 3 });
  const second = buildSvoSceneLights(scene, { revision: 3 });
  assert.deepEqual(first.records, second.records);
  assert.deepEqual(first.omittedFixtureKeys, second.omittedFixtureKeys);
  assert.strictEqual(second, first);
  assert.strictEqual(second.packedRecords, first.packedRecords);
  const advanced = buildSvoSceneLights(scene, { revision: 4 });
  assert.notEqual(advanced.cacheKey, first.cacheKey);
  assert.notStrictEqual(advanced.packedRecords, first.packedRecords);
  const fixtures = first.records.slice(1);
  assert.ok(fixtures.length > 0);
  assert.ok(fixtures.every((light) => light.kind === "rectangleArea" || light.kind === "sphereArea"));
  assert.ok(fixtures.every((light) => light.ownerId >= scene.rigidBodies.length));
  assert.ok(fixtures.every((light) => light.intensity > 0 && light.sourceKey.includes("night-lab")));
});

test("capacity keeps the brightest fixtures deterministically and reports omissions", () => {
  const scene = getScenePreset("sphere-jet").create();
  scene.environment = "night-lab";
  const result = buildSvoSceneLights(scene, { maximumRecords: 3 });
  assert.equal(result.records.length, 3);
  assert.ok(result.omittedFixtureKeys.length > 0);
  assert.equal(result.records[0].kind, "directional");
  assert.deepEqual(result.records.slice(1).map((light) => light.lightId), result.records.slice(1).map((light) => light.lightId).sort((a, b) => a - b));
});

test("packed identity lane carries type, stable ID, owner, and revision", () => {
  const light = canonicalSvoLightRecord({
    lightId: 17, ownerId: 22, revision: 4, kind: "rectangleArea",
    position_m: [1, 2, 3], range_m: 9, direction: [0, -2, 0],
    colorLinear: [0.8, 0.7, 0.5], intensity: 2,
    axisU: [1, 0, 0], halfWidth_m: 0.5, axisV: [0, 0, 1], halfHeight_m: 0.25,
    radius_m: 0, sourceKey: "test/light",
  });
  const packed = packSvoLightRecords([light]);
  assert.deepEqual([...packed.slice(24, 28)], [SVO_LIGHT_KINDS.rectangleArea, 17, 22, 4]);
  const floats = new Float32Array(packed.buffer);
  assert.deepEqual([...floats.slice(0, 4)], [1, 2, 3, 9]);
  assert.deepEqual([...floats.slice(12, 16)], [1, 0, 0, 0.5]);
});

test("garden point fixture packs one-sample identity and conservative emitter endpoint", () => {
  const scene = getScenePreset("garden-svo-lighting").create();
  const point = buildSvoSceneLights(scene).records.find(({ kind }) => kind === "point");
  assert.ok(point);
  assert.equal(point.radius_m, 0.18);
  const packed = packSvoLightRecords([point]);
  const words = new Uint32Array(packed.buffer);
  const floats = new Float32Array(packed.buffer);
  assert.equal(words[24], SVO_LIGHT_KINDS.point);
  assert.ok(Math.abs(floats[20] - 0.18) < 1e-6);
});

test("invalid shape/capacity/identity inputs fail before upload", () => {
  const base = buildSvoSceneLights(getScenePreset("water-box-dam-break").create()).records[0];
  assert.throws(() => canonicalSvoLightRecord({ ...base, lightId: 0 }), /reserved/);
  assert.throws(() => canonicalSvoLightRecord({ ...base, kind: "sphereArea", radius_m: 0 }), /positive/);
  assert.throws(() => packSvoLightRecords([base, base]), /Duplicate/);
  assert.throws(() => buildSvoSceneLights(getScenePreset("water-box-dam-break").create(), { maximumRecords: 0 }), /capacity/);
});

test("WGSL light record is binding-free and exposes authored radiance", () => {
  assert.match(svoLightWGSL, /struct SvoLightRecord/);
  assert.match(svoLightWGSL, /SVO_LIGHT_RECTANGLE_AREA/);
  assert.match(svoLightWGSL, /fn svoLightRadiance/);
  assert.doesNotMatch(svoLightWGSL, /@group|@binding|while\s*\(|loop\s*\{/);
});
