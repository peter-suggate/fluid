import assert from "node:assert/strict";
import test from "node:test";
import { uniformReferenceComputeShader } from "../lib/webgpu-uniform-reference.wgsl";
import { readFileSync } from "node:fs";

test("uniform WGSL carries the exterior-sphere solid through transport and pressure", () => {
  assert.match(uniformReferenceComputeShader, /fn hasSphericalContainer\(\)/);
  assert.match(uniformReferenceComputeShader, /hasRigidBodies\(\)\|\|hasTerrain\(\)\|\|hasSphericalContainer\(\)/);
  assert.match(uniformReferenceComputeShader, /if\(!hasSphericalContainer\(\)\)\{return vec4f\(0\.0,0\.0,0\.0,0\.5\);\}/);
  assert.match(uniformReferenceComputeShader, /fn cellSphericalSolidFraction\(p:vec3i\)/);
  assert.match(uniformReferenceComputeShader, /worldInsideSphericalSolid\(worldCell\(p\)\)/);
  assert.match(uniformReferenceComputeShader, /hasTerrain\(\)\|\|hasSphericalContainer\(\)\|\|nearAnyBody/);
  assert.match(uniformReferenceComputeShader, /sphericalContainerRadius\(\)-length\(world-sphericalContainerCenter\(\)\)/);
});

test("water composite owns exact curved glass and curved fallback exits", () => {
  const source = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(source, /fn sphereHit\(ro:vec3f,rd:vec3f,center:vec3f,radius:f32\)/);
  assert.match(source, /fn sphericalContainerGlassVisible\(\)/);
  assert.match(source, /tankExit=sphereHit\(innerOrigin,inside,sphericalContainerCenter\(\),sphericalContainerRadius\(\)\)/);
  assert.match(source, /if\(u\.cameraTarget\.w>1\.5\)\{return false;\}/,
    "spherical vessels must not emit the rectangular tank's wall-film atlas");
  assert.match(source, /if\(distance\(input\.world,center\)>radius\+0\.001\*cellSize\)\{discard;\}/,
    "surface coverage must be clipped to the same analytic sphere as physics");
});

test("uniform spherical presentation bypasses wall-film reconstruction", () => {
  const source = readFileSync(new URL("../lib/webgpu-uniform-reference.ts", import.meta.url), "utf8");
  assert.match(source, /else if \(sceneHasSphericalContainer\(this\.scene\)\) \{[\s\S]*?copyTextureToTexture\(\{ texture: this\.volumeA \}, \{ texture: this\.surfaceB \}/,
    "spherical presentation must copy conserved density instead of reconstructing a film");
});
