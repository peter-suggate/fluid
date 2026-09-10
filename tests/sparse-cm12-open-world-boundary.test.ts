import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.wgsl.ts",
  import.meta.url,
), "utf8");

test("GPU-grown SparseWorld rows use only signed topology and SolidWorld apertures", () => {
  assert.doesNotMatch(shader, /planarFluidBoundary|hasPlanarFluidBoundaries/);
  assert.match(shader, /fn rowOpenFraction\(id:u32\)->f32\{[\s\S]*?return solid\*solidVoxelRowOpenFraction\(id\);/);
  assert.match(shader, /fn rowPressureOpenFraction\(id:u32\)->f32\{[\s\S]*?return solid\*solidVoxelRowOpenFraction\(id\);/);
});
