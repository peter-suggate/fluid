import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const resident = readFileSync(new URL(
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident.ts",
  import.meta.url,
), "utf8");

test("candidate rerung synthesis excludes reserved dynamic capacity", () => {
  assert.match(resident,
    /dispatch\("synthesizeCandidateCellPages", this\.worldDirectoryLayout\.initialLeaves\)/);
  assert.doesNotMatch(resident,
    /dispatch\("synthesizeCandidateCellPages", leafCapacity\)/);
  assert.doesNotMatch(resident,
    /dispatchTopology\("synthesizeCandidateCellPages", leafCapacity\)/);
});
