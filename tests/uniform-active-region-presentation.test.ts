import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";

test("gamma diffusion leaves sparse presentation authority isolated", () => {
  const host = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
  assert.doesNotMatch(host, /gammaMirror|gammaDiffusionAverageGroup/,
    "the exact three-pass Jacobi schedule needs no mirrored scratch branch");
  assert.doesNotMatch(host,
    /copyTextureToTexture\(\{ texture: this\.gammaA \}, \{ texture: this\.surfaceB \}/,
    "a full-domain gamma copy outside the AABB contours as rectangular fluid sheets");
  assert.match(host,
    /this\.gammaDiffusionGroups = \[[\s\S]*?this\.volumeB, this\.volumeA[\s\S]*?this\.gammaA, this\.gammaB[\s\S]*?this\.volumeA, this\.volumeB[\s\S]*?this\.gammaB, this\.gammaA/,
    "Jacobi diffusion must use only its density/gamma ping-pong pair");
});
