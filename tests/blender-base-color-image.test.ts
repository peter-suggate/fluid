import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import test from "node:test";
const configuredBlender = process.env.BLENDER_PATH ?? process.env.BLENDER;
const blender = configuredBlender ?? "/Applications/Blender 5.2 LTS.app/Contents/MacOS/Blender";
test("Blender base-colour export preserves linear optical values through a real GLB roundtrip", {
  skip: !configuredBlender && !existsSync(blender), timeout: 30_000,
}, () => {
  const result = spawnSync(blender, ["-b", "--threads", "1", "--python-exit-code", "1",
    "--python", "tools/blender/test_base_color_image.py"], { encoding: "utf8", timeout: 25_000 });
  assert.equal(result.status, 0, `${result.error ?? ""}\n${result.stdout}\n${result.stderr}`);
  assert.match(result.stdout, /BASE_COLOR_ROUNDTRIP: linear optical values preserved through GLB/);
});
