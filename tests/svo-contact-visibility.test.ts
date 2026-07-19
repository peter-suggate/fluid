import assert from "node:assert/strict";
import test from "node:test";

import {
  SVO_CONTACT_VISIBILITY_CONTRACT,
  svoContactVisibilityDirections,
  svoContactVisibilityRadius_m,
  traceSvoContactVisibility,
} from "../lib/svo-contact-visibility";
import type { SvoVisibilityStepSource } from "../lib/svo-visibility-rays";

const input = {
  surfacePosition_m: [0, 0, 0] as const,
  geometricNormal: [0, 1, 0] as const,
  featureId: 0,
  cellSize_m: [0.02, 0.01, 0.04] as const,
  sceneExtent_m: [4, 2, 6] as const,
};
const counts = { nodeVisits: 1, leafVisits: 1, workItems: 1 } as const;

test("contact radius is finite, cell/scene-scaled, and globally capped", () => {
  assert.equal(svoContactVisibilityRadius_m(input.cellSize_m, input.sceneExtent_m), 0.24);
  assert.equal(svoContactVisibilityRadius_m([0.001, 0.001, 0.001], [4, 2, 6]), 0.06,
    "tiny cells retain a small scene-relative contact reach");
  assert.equal(svoContactVisibilityRadius_m([1, 1, 1], [4, 2, 6]), 0.36,
    "coarse cells cannot turn contact visibility into a long-range ambient shadow");
  assert.throws(() => svoContactVisibilityRadius_m([0, 1, 1], [1, 1, 1]), /positive/);
});

test("the two-sample feature pattern is deterministic, normalized, and hemisphere-only", () => {
  const a = svoContactVisibilityDirections(input.geometricNormal, 0);
  const b = svoContactVisibilityDirections(input.geometricNormal, 0);
  assert.deepEqual(a, b);
  assert.equal(a.length, SVO_CONTACT_VISIBILITY_CONTRACT.sampleCount);
  for (const direction of a) {
    assert.ok(Math.abs(Math.hypot(...direction) - 1) < 1e-12);
    assert.ok(direction[1] > 0);
  }
  assert.notDeepEqual(a, svoContactVisibilityDirections(input.geometricNormal, 1),
    "hard feature identity rotates a stable basis instead of interpolating cube edges");
});

test("open, corner/contact, and closed-wall visibility preserve bounded energy", () => {
  const open: SvoVisibilityStepSource = () => ({ status: "miss", counts });
  assert.deepEqual(traceSvoContactVisibility(input, open).visibility, [1, 1, 1]);

  const corner: SvoVisibilityStepSource = ({ ray }) => ray.direction[0] > 0
    ? { status: "hit", counts, hit: { t_m: ray.tMax_m * 0.5, opaque: true } }
    : { status: "miss", counts };
  assert.deepEqual(traceSvoContactVisibility(input, corner).visibility, [0.5, 0.5, 0.5]);

  const closed: SvoVisibilityStepSource = ({ ray }) => ({
    status: "hit", counts, hit: { t_m: ray.tMax_m * 0.25, opaque: true },
  });
  assert.deepEqual(traceSvoContactVisibility(input, closed).visibility, [0, 0, 0]);
});

test("invalid or exhausted work fails the complete contact estimate closed", () => {
  let calls = 0;
  const invalid: SvoVisibilityStepSource = () => ++calls === 1
    ? { status: "miss", counts }
    : { status: "invalid", counts, reason: "stale publication" };
  assert.deepEqual(traceSvoContactVisibility(input, invalid), {
    status: "invalid", visibility: [0, 0, 0], radius_m: 0.24,
  });

  const exhausted: SvoVisibilityStepSource = ({ remaining }) => {
    assert.deepEqual(remaining, {
      nodeVisits: SVO_CONTACT_VISIBILITY_CONTRACT.maximumNodeVisitsPerSample,
      leafVisits: SVO_CONTACT_VISIBILITY_CONTRACT.maximumLeafVisitsPerSample,
      workItems: SVO_CONTACT_VISIBILITY_CONTRACT.maximumWorkItemsPerSample,
      intersections: SVO_CONTACT_VISIBILITY_CONTRACT.maximumIntersectionsPerSample,
    });
    return { status: "exhausted", counts, reason: "bounded work" };
  };
  assert.deepEqual(traceSvoContactVisibility(input, exhausted).visibility, [0, 0, 0]);
});

test("thin transmission remains bounded and can only reduce indirect diffuse energy", () => {
  let interaction = 0;
  const glass: SvoVisibilityStepSource = ({ ray }) => ++interaction % 2 === 1
    ? { status: "hit", counts, hit: { t_m: ray.tMax_m * 0.25, opaque: false, transmittance: [0.8, 0.6, 0.4] } }
    : { status: "miss", counts };
  const result = traceSvoContactVisibility(input, glass);
  assert.equal(result.status, "resolved");
  assert.deepEqual(result.visibility.map((value) => Number(value.toFixed(6))), [0.8, 0.6, 0.4]);
  assert.ok(result.visibility.every((value) => value >= 0 && value <= 1));
});

