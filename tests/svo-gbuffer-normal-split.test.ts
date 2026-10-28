import assert from "node:assert/strict";
import test from "node:test";

import { packSvoGBufferPixel, unpackSvoGBufferPixel } from "../lib/svo/contracts/svo-gbuffer";
import { rasterAoProducerWGSL } from "../lib/svo/features/lighting-visibility/svo-raster-ao";
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
  // Voxel quads and reconstructed triangles retain the plane actually rasterized.
  assert.match(source, /let receiverNormal=input\.geometricNormal;/);
  assert.match(source, /normal=normalize\(boundaryNormal\);/);
  assert.doesNotMatch(source, /receiverNormal=normal;/);
  assert.match(source, /dryRasterPrimaryFacedSurface\(hit,receiverNormal,camera\[0\],rd,camera\[1\],SVO_GBUFFER_PRODUCER_BRICK\)/);
  // The face -> baked blend and the shading-normal call are untouched by the split.
  assert.match(source, /var shading=dryShadingNormal\(input\.identity,normal\);/);
  assert.match(source, /let hit=DryHit\(t,shading\.normal,/);
  // The background entry has one normal and keeps the plain publisher.
  assert.match(source, /return dryRasterPrimarySurface\(hit,camera\[0\],rd,camera\[1\],producer\);/);
});

test("image detail changes the shading normal while each producer retains its geometric face", async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const source = createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  // The plain publisher keeps its original normal as the geometric face.
  // Optional image detail changes only the separate shading field.
  assert.match(source, /fn dryRasterPrimarySurface\(opaque:DryHit,ro:vec3f,rd:vec3f,forward:vec3f,producer:u32\)->DryRasterPrimaryOut\{\s*return dryRasterPrimaryFacedSurface\(opaque,opaque\.normal,ro,rd,forward,producer\);/);
  assert.match(source, /let opaque=opaqueInput;let shadingNormal=dryImageSurfaceNormal\(opaque,ro\+rd\*opaque\.t\);/);
  assert.match(source, /if\(!drySampledMaterial\(hit\.materialId\)\)\{return hit\.normal;\}/);
  // Both voxel quality entries preserve an independent receiver normal.
  // No other producer may introduce another distinct-face call site.
  const faced = source.match(/dryRasterPrimaryFacedSurface\([^)]*\)/g) ?? [];
  const distinct = faced.filter((call) => !call.includes("opaque,opaque.normal") && !call.startsWith("dryRasterPrimaryFacedSurface(opaqueInput:"));
  assert.equal(distinct.length, 2, distinct.join(" | "));
  for (const call of distinct) assert.match(call, /hit,receiverNormal/);
  // A cleared face word is what every one of them writes, and it reads back as
  // the shading normal itself rather than a decoded approximation of it.
  assert.match(source, /if\(all\(geometricNormal==shadingNormal\)\)\{return 0u;\}/);
  assert.match(source, /fn dryGeometricNormal\(hit:DryHit\)->vec3f\{[\s\S]*?if\(\(hit\.aux\.y&DRY_OPAQUE_FACE_VALID\)==0u\)\{return hit\.normal;\}/);
});

test("the deferred lighting biases rays along the face and shades with the normal", async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const source = createSvoDrySceneFragmentWGSL(1, "raster-primary", "bounds", "split", 0, false, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  // The face travels in the identity plane's free metadata bits, because the
  // packed oct8 plane is not bound to the lighting entry at all.
  assert.match(source, /const DRY_OPAQUE_FACE_VALID:u32=134217728u;const DRY_OPAQUE_FACE_SHIFT:u32=28u;const DRY_OPAQUE_FACE_MASK:u32=2013265920u;/);
  // Both entries that rebuild a hit from the split planes: the seam sample and
  // the lighting entry. The reduced composition adds a third, its cached
  // reconstruction entry.
  assert.equal(source.match(/vec3u\(0u,metadata,0u\)/g)?.length, 2);
  const reduced = createSvoDrySceneFragmentWGSL(0.5, "raster-primary", "bounds", "split", 0, false, false, false,
    { surfaceMesh: true, surfaceMeshCulling: true });
  assert.equal(reduced.match(/vec3u\(0u,metadata,0u\)/g)?.length, 5);
  // Shadow rays and the contact hemisphere leave along the face.
  assert.match(source, /dryLightVisibility\(position,geometricNormal,hit\.ownerId,/);
  assert.match(source, /dryContactVisibility\(position,geometricNormal,hit\.featureId,hit\.ownerId\)/);
  // All lobes and the light-facing gate share the resolved optical frame.
  assert.match(source, /surface\.normal=unifiedShadingNormal\(surface\.normal,geometricNormal,-rd\);/);
  assert.match(source, /unifiedLightingInputWithGeometry\(surface\.normal,geometricNormal,-rd,/);
  assert.match(source, /svoEnvironmentDiffuseIrradiance\(dryLighting\.environment,surface\.normal\)/);
  assert.doesNotMatch(source, /dot\(geometricNormal,sample\.towardLight\)<=0\.0\)\{continue;\}let visibility=/);
});


test("continuous receivers retain a full-precision visibility normal", async () => {
  const { createSvoDrySceneFragmentWGSL } = await import("../lib/svo/features/shading/program");
  const source=createSvoDrySceneFragmentWGSL(1,"raster-primary","bounds","split",0,false,false,false,{surfaceMesh:true});
  assert.match(source,/if\(all\(geometricNormal==opaque\.normal\)\)\{opaqueMetadata\|=DRY_OPAQUE_SAME_NORMAL;\}/);
  assert.match(source,/if\(\(hit\.aux\.y&DRY_OPAQUE_SAME_NORMAL\)!=0u\)\{return hit\.normal;\}/);
});


test("reduced raster shadow correction decodes the published triangle plane", () => {
  const source=rasterAoProducerWGSL(false);
  const start=source.indexOf("fn rasterContactAt(");
  const end=source.indexOf("@compute",start);
  const contact=source.slice(start,end);
  assert.match(contact,/shadow=rasterSunVisibility\(position,dryGeometricNormal\(rasterVisibilityHit\(coord,g\)\)\)/);
  // AO keeps its existing continuous receiver-normal contract. This change is
  // only the planar depth comparison against rasterized shadow triangles.
  assert.match(contact,/rasterHorizonVisibility\(pixel,position,normal\)/);
});
