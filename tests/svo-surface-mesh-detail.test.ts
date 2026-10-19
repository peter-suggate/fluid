import assert from "node:assert/strict";
import test from "node:test";

import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";
import { svoFeatureQuery } from "../lib/svo/pipeline/persistence";
import { DEFAULT_SVO_RENDER_TUNING, SVO_SURFACE_MESH_LOD_PIXELS_DEFAULT, SVO_SURFACE_MESH_LOD_PIXELS_MAXIMUM, normalizeSvoRenderTuning, svoRenderTuningKey } from "../lib/svo/pipeline/svo-render-tuning";

test("filtered detail ships off and normalises into its bounds", () => {
  assert.equal(DEFAULT_SVO_RENDER_TUNING.surfaceMeshFilteringEnabled, false);
  assert.equal(DEFAULT_SVO_RENDER_TUNING.surfaceMeshLodPixels, 1);
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
  assert.equal(svoFeatureQuery.read(new URLSearchParams("svoMeshLodPixels=99")).svoRenderTuning.surfaceMeshLodPixels, 1);
  assert.equal(svoFeatureQuery.read(new URLSearchParams()).svoRenderTuning.surfaceMeshLodPixels, 1);
});

test("the mesh shader extracts every level once and lets the camera pick one per brick", () => {
  for (const culling of [true, false]) {
    const source = svoSurfaceMeshWGSL(3, 1 << 7, culling);
    // Threshold is the spare lod lane, never a compile-time constant.
    assert.match(source, /fn dryMeshLodPixels\(\)->f32\{\s*return select\(0\.0,max\(dry\.lod\.w,0\.0\),dry\.meshFilter\.x>0\.5\)/);
    // One job per exact face layer plus one per coarse level, stated once for
    // the count and emit phases and the schedule alike.
    assert.equal(source.match(/6u\*dry\.mapping\.brickSize\+meshLevelCount\(\)-1u/g)?.length, 1);
    assert.equal(source.match(/meshJobsPerBrick\(\)/g)!.length >= 3, true);
    // Zero threshold selects the exact level everywhere.
    assert.match(source, /if\(dryMeshLodPixels\(\)<=0\.0\)\{return level==0u;\}/);
    // The cull pass consumes per-brick selection; without culling the vertex stage collapses unselected levels.
    if (culling) {
      assert.match(source, /if\(!meshQuadSelected\(quad\)\)\{return false;\}/);
      assert.doesNotMatch(source, /if\(meshQuadDead\(quad\)\|\|!meshQuadSelectedRead\(quad\)\)/);
    } else {
      assert.match(source, /if\(meshQuadDead\(quad\)\|\|!meshQuadSelectedRead\(quad\)\)\{position=vec4f\(0\.0,0\.0,0\.0,1\.0\);\}/);
    }
    // Exact quads merge on material only and read the voxel's baked normal per fragment.
    assert.match(source, /fn meshMergeIdentity\(identity:u32\)->u32\{return sceneIdentityMaterial\(identity\)\|\(SCENE_IDENTITY_NO_NORMAL<<16u\);\}/);
    assert.match(source, /if\(input\.level==0u\)\{[^}]*meshRegionAt\(lattice\)\.identity;/);
    // Coarse boundary faces hide only behind complete coverage.
    assert.match(source, /exposed=!meshNeighbourCovered\(p,cell\);/);
    // A coarse quad's normal is the mean of the voxels facing the way it does:
    // the cell table decides solidity alone, and the identity a mask carries is
    // the exposed face's own mean, so no cell shares one normal across six faces.
    assert.match(source, /cells\[i\]=meshCellMaterial\(payload,vec3u\(i%m,\(i\/m\)%m,i\/\(m\*m\)\),level,n\);/);
    assert.match(source, /if\(exposed\)\{mask\[index\]=meshFaceIdentity\(payload,c,level,n,face\);meshMaskAgreement\[index\]=meshFaceAgreement;\}/);
    assert.match(source, /fn meshFaceIdentity\(payload:u32,cell:vec3u,level:u32,n:u32,face:u32\)->u32/);
    // Only the half-space facing the quad contributes. Store any nonzero mean;
    // the fragment now applies the live agreement threshold.
    assert.match(source, /if\(baked\[axis\]\*towards<=0\.0\)\{continue;\}/);
    assert.match(source, /if\(dot\(sum,sum\)>1e-12\)\{normalWord=svoGBufferPackNormalOct8\(normalize\(sum\)\);\}/);
    // The per-face identity replaced the per-cell one outright.
    assert.doesNotMatch(source, /meshCellIdentity/);
  }
});

test("native sampled surfaces omit duplicate mesh faces but raster shadow maps retain their casters", () => {
  const append = (retainCasters: boolean) => svoSurfaceMeshWGSL(3, 1 << 7, true, true, retainCasters)
    .split("fn meshAppend(")[1]!.split("\n}")[0]!;
  const native = append(false), rasterShadows = append(true);
  // Rejection must precede both count and emit allocations; otherwise the two
  // extraction phases disagree and leave holes or uninitialised arena records.
  assert.ok(native.indexOf("dryMaterial(") < native.indexOf("if(meshPhase==0u)"));
  assert.match(native, /identity\.w&4u/);
  // Retained casters may inspect material after emission to classify the leaf,
  // but must not reject sampled surfaces before the count/emit allocations.
  assert.doesNotMatch(rasterShadows.split("if(meshPhase==0u)")[0]!, /dryMaterial\(/);
  assert.match(rasterShadows, /meshArena\[meshEmitBase\+slot\]=SurfaceQuad/);
  const cull = svoSurfaceMeshWGSL(3, 1 << 7, true, true, true)
    .split("fn surfaceMeshPrimaryCandidates(")[1]!.split("fn surfaceMeshCullDispatch")[0]!;
  assert.match(cull, /dryMaterial\(sceneIdentityMaterial\(quad.identity\)\).identity.w&4u/,
    "geometry compaction excludes shadow proxies from camera candidates");
  for (const culling of [true, false]) {
    const source = svoSurfaceMeshWGSL(3, 1 << 7, culling, true, true);
    const [vertex, fragment] = source.split("@vertex fn surfaceMeshVertex")[1]!.split("@fragment fn surfaceMeshFragment");
    assert.match(vertex!, /if\(\(dryMaterial\(sceneIdentityMaterial\(quad.identity\)\).identity.w&4u\)!=0u\)\{position=vec4f\(0\.0,0\.0,0\.0,1\.0\);\}/,
      "retained proxies collapse before rasterization, including without compaction");
    assert.doesNotMatch(fragment!.split("\n}")[0]!, /discard/,
      "opaque mesh fragments must not retain the proxy rejection discard");
  }
});
