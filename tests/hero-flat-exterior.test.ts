import assert from "node:assert/strict";
import test from "node:test";

import { HERO_GARDEN_VESSEL } from "../lib/hero-garden-scene";
import { pondVesselHeightAt, pondVesselPlanCurve } from "../lib/voxel-scenery/pond-vessel";

test("the hero exterior is one flat plane outside the pond wall", () => {
  assert.deepEqual(HERO_GARDEN_VESSEL.terraces, []);
  const samples = [
    [-0.82, -0.48], [-0.82, 0], [-0.82, 0.48],
    [0, -0.52], [0, 0.52],
    [0.82, -0.48], [0.82, 0], [0.82, 0.48],
  ] as const;
  const curve = pondVesselPlanCurve(HERO_GARDEN_VESSEL, 48);
  const heights = samples.map(([x, z]) => pondVesselHeightAt(HERO_GARDEN_VESSEL, curve, x, z));
  const spread = Math.max(...heights) - Math.min(...heights);
  assert.ok(spread <= 2 * (HERO_GARDEN_VESSEL.relief_m ?? 0) + 1e-9,
    `exterior height varies ${(spread * 1000).toFixed(2)} mm`);
  assert.ok(heights.every((height) => Math.abs(height - HERO_GARDEN_VESSEL.groundHeight_m)
    <= (HERO_GARDEN_VESSEL.relief_m ?? 0) + 1e-9));
});
