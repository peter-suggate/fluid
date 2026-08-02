import assert from "node:assert/strict";
import test from "node:test";

import { environmentIds } from "../lib/environments";
import { cloneScene, defaultScene } from "../lib/model";
import { buildSvoEnvironmentCoverage } from "../lib/svo-scene-coverage";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";
import { SCENERY_GRAPHS } from "../lib/scenery-presets";
import { validateSceneryGraph } from "../lib/scenery-graph";

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

test("every visible surface an environment audits is geometry the tracer owns", () => {
  const scene = cloneScene(defaultScene);
  // Environments used to paint a foreground in NDC — botanical framing, a
  // vignette, drifting dust — which parallaxed with nothing, occluded nothing
  // and took no light, so it read as a decal on the lens. The category for it
  // is gone; what remains is that every audited surface names a real owner.
  const owned = new Set(["analytic-primitive", "analytic-rigid-body", "analytic-terrain",
    "thin-glass", "thick-glass", "opaque-proxy-fallback", "not-visible"]);
  for (const environmentId of environmentIds) {
    const report = buildSvoEnvironmentCoverage(scene, environmentId);
    const painted = report.entries.filter((entry) => !owned.has(entry.visibleOwnership));
    assert.deepEqual(painted.map(({ key }) => key), [],
      `${environmentId} must build its scenery as geometry, not as an NDC overlay`);
  }
});

test("every environment describes its geometry as a well-formed scenery graph", () => {
  const scene = cloneScene(defaultScene);
  for (const environmentId of environmentIds) {
    const seed = SCENERY_GRAPHS[environmentId];
    assert.ok(seed, `${environmentId} seeds a scenery graph`);
    assert.deepEqual(validateSceneryGraph(seed(scene)), [],
      `${environmentId} must seed a valid graph: exactly one shell, unique ids, known palettes`);
  }
  assert.equal(Object.keys(SCENERY_GRAPHS).length, environmentIds.length,
    "the registry covers every environment and nothing else");
});
