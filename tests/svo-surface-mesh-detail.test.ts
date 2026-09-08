import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { svoFeatureQuery } from "../lib/svo/pipeline/persistence";
import {
  DEFAULT_SVO_RENDER_TUNING,
  SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT,
  SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM,
  normalizeSvoRenderTuning,
  svoRenderTuningKey,
} from "../lib/svo/pipeline/svo-render-tuning";

test("filtered detail ships off and normalises into its bounds", () => {
  assert.equal(DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels, 0);
  assert.ok(SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT > 0 && SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT <= SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM);
  const below = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshLodPixels: -2 });
  const above = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshLodPixels: 1e6 });
  const fractional = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshLodPixels: 0.75 });
  assert.equal(below.surfaceMeshLodPixels, 0);
  assert.equal(above.surfaceMeshLodPixels, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM);
  assert.equal(fractional.surfaceMeshLodPixels, 0.75);
  // The threshold keys frame reuse: two tunings that differ only here are not the same frame.
  assert.notEqual(svoRenderTuningKey(fractional), svoRenderTuningKey(DEFAULT_SVO_RENDER_TUNING));
});

test("the threshold round-trips through the URL and resets from the address", () => {
  const query = new URLSearchParams();
  const state = svoFeatureQuery.read(new URLSearchParams());
  svoFeatureQuery.write(query, { ...state, svoRenderTuning: { ...state.svoRenderTuning, surfaceMeshLodPixels: 1.5 } });
  assert.equal(query.get("svoMeshLodPixels"), "1.5");
  assert.equal(svoFeatureQuery.read(query).svoRenderTuning.surfaceMeshLodPixels, 1.5);
  // An out-of-range address value is rejected back to the shipped default, as every tuning key is.
  assert.equal(svoFeatureQuery.read(new URLSearchParams("svoMeshLodPixels=99")).svoRenderTuning.surfaceMeshLodPixels, 0);
  assert.equal(svoFeatureQuery.read(new URLSearchParams()).svoRenderTuning.surfaceMeshLodPixels, 0);
});

test("the mesh shader extracts every level once and lets the camera pick one per brick", () => {
  for (const culling of [true, false]) {
    const source = svoSurfaceMeshWGSL(3, 1 << 7, culling);
    // Threshold is the spare lod lane, never a compile-time constant.
    assert.match(source, /fn dryMeshLodPixels\(\)->f32\{\s*return max\(dry\.lod\.w,0\.0\)/);
    // One job per exact face layer plus one per coarse level, in prepare and build alike.
    assert.equal(source.match(/6u\*dry\.mapping\.brickSize\+meshLevelCount\(\)-1u/g)?.length, 1);
    assert.equal(source.match(/6u\*n\+meshLevelCount\(\)-1u/g)?.length, 1);
    // Zero threshold selects the exact level everywhere.
    assert.match(source, /if\(threshold<=0\.0\)\{return level==0u;\}/);
    // The cull pass is the selector; without it the vertex stage collapses unselected levels.
    if (culling) {
      assert.match(source, /if\(!meshQuadSelected\(quad\)\)\{return false;\}/);
      assert.doesNotMatch(source, /position=vec4f\(0\.0,0\.0,0\.0,1\.0\)/);
    } else {
      assert.match(source, /if\(!meshQuadSelected\(quad\)\)\{position=vec4f\(0\.0,0\.0,0\.0,1\.0\);\}/);
    }
    // Exact quads merge on material only and read the voxel's baked normal per fragment.
    assert.match(source, /fn meshMergeIdentity\(identity:u32\)->u32\{return sceneIdentityMaterial\(identity\)\|\(SCENE_IDENTITY_NO_NORMAL<<16u\);\}/);
    assert.match(source, /if\(input\.level==0u\)\{[^}]*meshRegionAt\(lattice\)\.identity;/);
    // Coarse boundary faces hide only behind complete coverage.
    assert.match(source, /exposed=!meshNeighbourCovered\(p,cell\);/);
  }
});

test("the full dry-scene bundle validates under naga in every mesh variant", async (t) => {
  const naga = process.env.NAGA ?? "naga";
  if (spawnSync(naga, ["--version"], { encoding: "utf8" }).status !== 0) {
    t.skip("naga is not installed"); return;
  }
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const directory = mkdtempSync(join(tmpdir(), "svo-mesh-detail-"));
  const variants: Array<[string, Parameters<typeof createSvoDrySceneFragmentWGSL>[0], boolean]> = [
    ["scale-1", 1, true], ["scale-half", 0.5, true], ["no-cull", 1, false],
  ];
  for (const [name, scale, culling] of variants) {
    const path = join(directory, `${name}.wgsl`);
    writeFileSync(path, createSvoDrySceneFragmentWGSL(scale, "raster-primary", "bounds", "split", 0, false, true, false, false,
      { surfaceMesh: true, surfaceMeshCulling: culling }));
    const result = spawnSync(naga, [path], { encoding: "utf8" });
    assert.equal(result.status, 0, `${name}: ${(result.stdout + result.stderr).slice(0, 2000)}`);
  }
});
