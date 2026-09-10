import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import { packSvoGBufferPixel, unpackSvoGBufferPixel } from "../lib/svo/contracts/svo-gbuffer";
import { svoSurfaceMeshWGSL } from "../lib/svo/features/primary-visibility/svo-surface-mesh";

// The G-buffer has always had two normal slots. Until the voxel mesh's filtered
// detail there was no producer with two normals to put in them, so this pins
// the split end to end: the contract keeps them apart, the mesh fragment is the
// only producer that publishes a face different from what it shades with, and
// the deferred lighting biases its rays along that face while every closure
// keeps the shading normal.

test("the packed contract keeps a geometric normal distinct from a shading normal", () => {
  const packed = packSvoGBufferPixel({
    status: "hit", radianceLinear: [0, 0, 0], depth_m: 3.5,
    geometricNormal: [1, 0, 0], shadingNormal: [0, 0, 1],
    materialId: 7, ownerId: 0xffff, mediumBefore: 0, mediumAfter: 3,
    velocity_m_s: [0, 0, 0], motionKind: 0, motionValid: false,
    fieldSource: 1, localTopologyGeneration: 2, featureId: 0,
  });
  // Two different oct8 words, one per half of packedSurface.x.
  assert.notEqual(packed.packedSurface[0] & 0xffff, packed.packedSurface[0] >>> 16);
  const pixel = unpackSvoGBufferPixel(packed);
  assert.equal(pixel.status, "hit");
  if (pixel.status !== "hit") return;
  // An axis-aligned face survives oct8 exactly, which is what makes the
  // shader's fallback (equal words) an exact statement rather than a tolerance.
  assert.deepEqual(pixel.geometricNormal.map((c) => Math.round(c)), [1, 0, 0]);
  assert.deepEqual(pixel.shadingNormal.map((c) => Math.round(c)), [0, 0, 1]);
});

test("the mesh fragment is the one producer that publishes a face of its own", () => {
  const source = svoSurfaceMeshWGSL(3, 1 << 7, true);
  // The face of the rasterised quad, never the normal the fragment shades with.
  assert.match(source, /dryRasterPrimaryFacedSurface\(hit,input\.normal,camera\[0\],rd,camera\[1\],SVO_GBUFFER_PRODUCER_BRICK\)/);
  // The face -> baked blend and the shading-normal call are untouched by the split.
  assert.match(source, /let shading=dryShadingNormal\(input\.identity,normal\);/);
  assert.match(source, /let hit=DryHit\(t,shading\.normal,/);
  // The background entry has one normal and keeps the plain publisher.
  assert.match(source, /return dryRasterPrimarySurface\(hit,camera\[0\],rd,camera\[1\],producer\);/);
});

test("every other producer publishes one normal twice, bit for bit", async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const source = createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, true, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  // The old signature is the new one with the surface's own normal for a face,
  // so nothing but the mesh can move by so much as an oct8 quantum.
  assert.match(source, /fn dryRasterPrimarySurface\(opaque:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32\)->DryRasterPrimaryOut\{\s*return dryRasterPrimaryFacedSurface\(opaque,opaque\.normal,ro,rd,forward,producer\);/);
  // Exactly one call site passes a face that is not the shading normal.
  const faced = source.match(/dryRasterPrimaryFacedSurface\([^)]*\)/g) ?? [];
  const distinct = faced.filter((call) => !call.includes("opaque,opaque.normal") && !call.startsWith("dryRasterPrimaryFacedSurface(opaque:"));
  assert.equal(distinct.length, 1, distinct.join(" | "));
  assert.match(distinct[0], /hit,input\.normal/);
  // A cleared face word is what every one of them writes, and it reads back as
  // the shading normal itself rather than a decoded approximation of it.
  assert.match(source, /if\(all\(geometricNormal==shadingNormal\)\)\{return 0u;\}/);
  assert.match(source, /fn dryGeometricNormal\(hit:DryHit\)->vec3f\{\s*if\(\(hit\.aux\.y&DRY_OPAQUE_FACE_VALID\)==0u\)\{return hit\.normal;\}/);
});

test("the deferred lighting biases rays along the face and shades with the normal", async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const source = createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, true, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  // The face travels in the identity plane's free metadata bits, because the
  // packed oct8 plane is not bound to the lighting entry at all.
  assert.match(source, /const DRY_OPAQUE_FACE_VALID:u32=134217728u;const DRY_OPAQUE_FACE_SHIFT:u32=28u;const DRY_OPAQUE_FACE_MASK:u32=2013265920u;/);
  // Both entries that rebuild a hit from the split planes: the seam sample and
  // the lighting entry. The reduced composition adds a third, its cached
  // reconstruction entry.
  assert.equal(source.match(/vec3u\(0u,metadata&DRY_OPAQUE_FACE_MASK,0u\)/g)?.length, 2);
  const reduced = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "bounds", "split", 0, false, true, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  assert.equal(reduced.match(/vec3u\(0u,metadata&DRY_OPAQUE_FACE_MASK,0u\)/g)?.length, 3);
  // Shadow rays and the contact hemisphere leave along the face.
  assert.match(source, /dryLightVisibility\(position,geometricNormal,hit\.ownerId,/);
  assert.match(source, /dryContactVisibility\(position,geometricNormal,hit\.featureId,hit\.ownerId\)/);
  // Shading keeps the surface normal: the BRDF, the environment terms and the
  // N.L gate are all still hit.normal.
  assert.match(source, /unifiedLightingInputWithGeometry\(hit\.normal,hit\.normal,-rd,/);
  assert.match(source, /svoEnvironmentDiffuseIrradiance\(dryLighting\.environment,hit\.normal\)/);
  assert.doesNotMatch(source, /dot\(geometricNormal,sample\.towardLight\)<=0\.0\)\{continue;\}let visibility=/);
});

test("the split validates under naga with and without the voxel mesh", async (t) => {
  const naga = process.env.NAGA ?? "naga";
  if (spawnSync(naga, ["--version"], { encoding: "utf8" }).status !== 0) {
    t.skip("naga is not installed"); return;
  }
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const directory = mkdtempSync(join(tmpdir(), "svo-normal-split-"));
  const variants: Array<[string, string]> = [
    ["inline", createSvoDrySceneFragmentWGSL(1, "hybrid", "bounds", "inline")],
    ["split-full", createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, true)],
    ["split-traced", createSvoDrySceneFragmentWGSL(1, "hybrid", "bounds", "split")],
    // The reduced path is where the cone prepass and the silhouette refinement
    // decode the same identity plane the face word now travels in.
    ["split-reduced", createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "bounds", "split", 0, false, true)],
  ];
  for (const [name, code] of variants) {
    const path = join(directory, `${name}.wgsl`);
    writeFileSync(path, code);
    const result = spawnSync(naga, [path], { encoding: "utf8" });
    assert.equal(result.status, 0, `${name}: ${(result.stdout + result.stderr).slice(0, 2000)}`);
  }
});
