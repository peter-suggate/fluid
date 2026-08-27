import assert from "node:assert/strict";
import test from "node:test";

import { getScenePreset } from "../lib/core/scenes";
import { REC709_LUMINANCE, resolveWaterKeyLight } from "../lib/core/webgpu-lighting";
import { svoSceneLighting } from "../lib/svo/svo-dry-scene-lighting";
import { buildSvoSceneLights, waterKeyDirectionalFromSceneLights } from "../lib/svo/svo-light-abi";
import type { SceneDescription } from "../lib/core/model";

const luminance = (color: readonly number[]) =>
  REC709_LUMINANCE[0] * color[0] + REC709_LUMINANCE[1] * color[1] + REC709_LUMINANCE[2] * color[2];

/** What the renderer passes: the container's middle, in the lights' own frame. */
const receiver = (scene: SceneDescription): [number, number, number] =>
  [0, 0.5 * scene.container.height_m, 0];

function waterKey(scene: SceneDescription) {
  const authored = svoSceneLighting(scene)?.directional;
  const fixture = waterKeyDirectionalFromSceneLights(buildSvoSceneLights(scene).records, receiver(scene));
  return {
    before: resolveWaterKeyLight(authored),
    after: resolveWaterKeyLight(fixture ?? authored),
  };
}

/**
 * The regression guard for every set lit by a sun.
 *
 * The authored directional is a record in the same table the fixtures are in,
 * so it competes rather than being overridden. On a set with no fixture that
 * reaches the water it wins by default, and the water is keyed exactly where it
 * was before the table was consulted at all.
 */
test("a sunlit set keys its water off the authored directional, unchanged", () => {
  // The hillside stands on the stage but is lit like a landscape: it carries no
  // fixture at all, so the competition has one entrant and the sun it authored
  // is the key its water gets.
  for (const id of ["hero-garden-hose", "tall-cells-hillside-dam-break"]) {
    const scene = getScenePreset(id).create();
    const { before, after } = waterKey(scene);

    assert.deepEqual(after.direction, before.direction, `${id}: sunlit key direction must not move`);
    assert.deepEqual(after.radianceLinear, before.radianceLinear, `${id}: sunlit key radiance must not move`);
    assert.equal(after.authored, before.authored);
  }
});

/**
 * The stage, where the gap this exists to close is measurable.
 *
 * The set's illumination is a practical held in the light table; the water's
 * rig is one key resolved from `lighting.directional`, which on this set is a
 * fill deliberately held down *because* the practical exists. Keyed off the
 * fill the water is lit at a fiftieth of the floor beneath it.
 */
test("a spotlit stage keys its water off the practical, not the fill", () => {
  for (const id of ["ocean-seiche", "water-box-dam-break"]) {
    const scene = getScenePreset(id).create();
    assert.equal(scene.environment, "stage", `${id} is expected to stand on the stage`);
    const { before, after } = waterKey(scene);

    assert.ok(
      luminance(after.radianceLinear) > 25 * luminance(before.radianceLinear),
      `${id}: the practical must dominate the fill it was held down against`,
    );
    // The lamp hangs on the container's axis, so the key it supplies points
    // straight up. The fill it replaces is a tenth off vertical, which is why
    // the direction barely moves and the radiance moves by fifty-fold: the gap
    // was never about where the light comes from.
    assert.ok(after.direction[1] > 0.99, `${id}: the key must arrive from the fixture overhead`);
  }
});

/**
 * The reason the reducer restates `dryLightSample` rather than approximating it.
 *
 * A second model of the same fixture is how the lamp lighting the floor and the
 * lamp keying the water come to disagree about its strength. Pinning the spot's
 * arrival against the shader's own terms — base radiance over the larger of one
 * and the squared distance, faded across its range and ramped by the square of
 * its beam — is what makes that drift a failure here rather than a look nobody
 * can explain. The range fade is in this list because it was missing from the
 * first draft of it, and the reducer was right where the expectation was not.
 */
test("the water's key mirrors the dry shader's own spot evaluation", () => {
  const scene = getScenePreset("water-box-dam-break").create();
  const spot = buildSvoSceneLights(scene).records.find((light) => light.kind === "spot");
  assert.ok(spot, "the stage is expected to publish its practical as a spot");

  const target = receiver(scene);
  const offset = spot.position_m.map((axis, index) => axis - target[index]);
  const distanceSquared = offset.reduce((total, axis) => total + axis * axis, 0);
  const distance = Math.sqrt(distanceSquared);
  const towardLight = offset.map((axis) => axis / distance);
  const axisLength = Math.hypot(...spot.direction);
  const alignment = -spot.direction.reduce((total, axis, index) => total + axis * towardLight[index], 0) / axisLength;
  const cosOuter = spot.cone?.cosOuter ?? -1;
  const cosInner = spot.cone?.cosInner ?? 1;
  const beam = Math.min(1, Math.max(0, (alignment - cosOuter) / Math.max(cosInner - cosOuter, 1e-4)));
  const rangeFade = spot.range_m > 0
    ? Math.min(1, Math.max(0, 1 - distance / spot.range_m)) ** 2
    : 1;
  const expected = spot.colorLinear.map((channel) =>
    channel * spot.intensity * rangeFade * beam * beam / Math.max(1, distanceSquared));

  const { after } = waterKey(scene);
  after.radianceLinear.forEach((channel, index) => {
    assert.ok(
      Math.abs(channel - expected[index]) <= 1e-6 * Math.max(1, Math.abs(expected[index])),
      `channel ${index}: water key ${channel} must equal the shader's arrival ${expected[index]}`,
    );
  });
});
