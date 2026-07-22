import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";

const root = new URL("..", import.meta.url).pathname;
const runtimeFiles = readdirSync(join(root, "lib"))
  .filter((name) => (name.startsWith("octree-") || name.startsWith("webgpu-octree"))
    && name.endsWith(".ts"))
  .map((name) => join(root, "lib", name))
  .concat(join(root, "lib", "methods", "octree.ts"));

test("the power-octree runtime names only its 2017 paper authority", () => {
  for (const file of runtimeFiles) {
    const source = readFileSync(file, "utf8");
    assert.doesNotMatch(source, /Ando\s*(?:--|–|-|and|&)\s*Batty|\b2020\b/i,
      `${file} must not attribute power-octree rules to the unrelated 2020 method`);
  }
});
