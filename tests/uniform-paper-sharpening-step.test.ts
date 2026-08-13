import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");

test("Sec. 3.5 sharpening uses the paper's DeltaT = 3 dt at unit authored strength", () => {
  assert.match(shader, /let deltaT=3\.0\*params\.dimsDt\.w\*params\.tuning\.x;let tau=0\.4/);
  assert.doesNotMatch(shader, /let deltaT=2\.0\*min\(/);
});
