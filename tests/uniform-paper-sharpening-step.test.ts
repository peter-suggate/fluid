import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const shader = readFileSync(new URL("../lib/webgpu-uniform-reference.wgsl.ts", import.meta.url), "utf8");

test("Sec. 3.5 sharpening uses the paper's fictitious timestep DeltaT = 3 dt", () => {
  assert.match(shader, /let deltaT=3\.0\*params\.dimsDt\.w;let tau=0\.4/);
  assert.doesNotMatch(shader, /let deltaT=2\.0\*min\(/);
});
