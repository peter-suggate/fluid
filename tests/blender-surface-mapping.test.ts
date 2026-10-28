import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";

const configuredBlender = process.env.BLENDER_PATH ?? process.env.BLENDER;
const blender = configuredBlender ?? "/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender";
test("Blender FVM mapping preserves affine charts and sRGB-array optical means", {
  skip: !configuredBlender && !existsSync(blender), timeout: 30_000,
}, () => {
  const result = spawnSync(blender, ["-b", "--threads", "1", "--python-exit-code", "1",
    "--python", "tools/blender/test_surface_mapping.py"], { encoding: "utf8", timeout: 25_000 });
  assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
  assert.match(result.stderr, /Ran 5 tests/);
});
