import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoEnvironmentCoverage } from "../lib/svo-scene-coverage";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import { ENVIRONMENT_SCENERY } from "../lib/voxel-scenery";

/**
 * Saturation of a linear-RGB triple, on the same 0..1 scale as HSV chroma.
 *
 * Relative chroma is the right measure for anything the eye can read a hue in,
 * but it divides by the brightest channel, so it explodes on near-black: a
 * monitor bezel at [.030,.034,.040] scores 0.25 on a spread of twelve
 * thousandths that no viewer can see. Surfaces whose *absolute* spread is that
 * small are reported neutral, which is what they look like.
 */
const ABSOLUTE_CHROMA_FLOOR = 0.02;

function saturation(color: readonly [number, number, number]): number {
  const max = Math.max(...color);
  const chroma = max - Math.min(...color);
  if (max <= 0.02 || chroma <= ABSOLUTE_CHROMA_FLOOR) return 0;
  return chroma / max;
}

/**
 * The house style is monochrome: form is read from shading, shadow and ambient
 * occlusion rather than from hue. Emissive sources are deliberately exempt —
 * a warm lamp against a cool panel is the one colour signal we keep, and it is
 * what makes the lighting legible.
 */
const SURFACE_SATURATION_LIMIT = 0.22;

test("every scenery surface stays monochrome so shading carries the form", () => {
  const scene = cloneScene(defaultScene);
  const offenders: string[] = [];
  for (const environmentId of environmentIds) {
    const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
    for (const primitive of environmentProxyPrimitives(catalog)) {
      if (primitive.material.emission > 0) continue;
      const value = saturation(primitive.material.colorLinear);
      if (value > SURFACE_SATURATION_LIMIT) {
        offenders.push(`${primitive.key} ${JSON.stringify(primitive.material.colorLinear)} saturation ${value.toFixed(3)}`);
      }
    }
  }
  assert.deepEqual(offenders, [], `non-emissive scenery must stay near-neutral:\n  ${offenders.join("\n  ")}`);
});

test("scenery is bright enough that shadow reads as shadow", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const catalog = buildEnvironmentProxyCatalog(scene, environmentId);
    const surfaces = environmentProxyPrimitives(catalog).filter(({ material }) => material.emission === 0);
    // A set painted entirely in the basement of the value range has nowhere to
    // fall off to, which is exactly what the old near-black station looked like.
    const brightest = Math.max(...surfaces.map(({ material }) => Math.max(...material.colorLinear)));
    assert.ok(brightest >= 0.45, `${environmentId} has no surface above ${brightest.toFixed(3)}: nothing for light to fall off from`);
  }
});

test("no environment paints its foreground in screen space any more", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const report = buildSvoEnvironmentCoverage(scene, environmentId);
    const painted = report.entries.filter((entry) =>
      entry.category === "procedural-foreground" || entry.visibleOwnership === "raster-only-procedural");
    assert.deepEqual(painted.map(({ key }) => key), [],
      `${environmentId} must build its scenery as geometry, not as an NDC overlay`);
  }
});

test("each environment owns its geometry in its own scenery module", () => {
  for (const environmentId of environmentIds) {
    const scenery = ENVIRONMENT_SCENERY[environmentId];
    assert.ok(scenery, `${environmentId} has a scenery module`);
    assert.equal(scenery.id, environmentId, "scenery module id matches its registry slot");
  }
  assert.equal(Object.keys(ENVIRONMENT_SCENERY).length, environmentIds.length,
    "the registry covers every environment and nothing else");
});
