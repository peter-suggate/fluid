import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import test from "node:test";
import * as native from "../lib/methods/adaptive-mass/sparse-cm12-native-surface.wgsl";

// These fingerprints were taken from commit 57b6ae39, not from the replacement.
test("native CM12 surface reconstruction preserves the 7 September implementation", () => {
  assert.equal(createHash("sha256").update(native.NATIVE_PRESENTATION_COARSE_COLUMN_PHI_WGSL.trim()).digest("hex"), "d3099b6904f3cf60f921ad7a454c126608b8c4d180901a2619757b916819b614");
  assert.equal(createHash("sha256").update(native.NATIVE_PRESENTATION_INTERPOLATED_VOLUME_PHI_WGSL.trim()).digest("hex"), "b73d3b536718e2fc761a866561f009afd6f1228cfc398c50b8dd0bd68cfc69e6");
  assert.equal(createHash("sha256").update(native.NATIVE_SURFACE_PROOF_VIRTUAL_COLUMN_PHI_WGSL.trim()).digest("hex"), "df33a30c39357ac872df166116c0e61d69b3cbfe96bdcc41e1f6ad41550e8816");
  assert.equal(createHash("sha256").update(native.NATIVE_SURFACE_PROOF_VIRTUAL_VOLUME_PHI_WGSL.trim()).digest("hex"), "a34e22f8d2670b718237f59cde6d8a7de9c373f7d5ba247c711aec5129c0e0d9");
});

test("default mesh construction and normals preserve the 7 September shaders", () => {
  assert.equal(createHash("sha256").update(readFileSync(new URL("../lib/core/webgpu-water-adaptive-mesh.ts", import.meta.url))).digest("hex"), "ee0e67afd87a10f97e1a41c6f1deda46a1c23645628c05575891e5d8cbe29ffb");
  assert.equal(createHash("sha256").update(readFileSync(new URL("../lib/core/webgpu-water-global-fine-tetra.ts", import.meta.url))).digest("hex"), "2dcdf809ca18ac8eac25f9f4479e59dc6186d5b17d1abc2143e1df1a94618749");
});
