import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import {
  createSvoDryConeMarcherWGSL,
  createSvoDrySceneFragmentWGSL,
  SparseVoxelDrySceneRenderer,
  SVO_DRY_CONE_PREPASS_CONTRACT,
  svoConePrepassSize,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

test("scale 1 preserves the production shader byte-for-byte (fingerprint contract)", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1), svoDrySceneShader,
    "the factory default must return the exact historical string so the bit-exact frame fingerprint reproduces");
  assert.doesNotMatch(svoDrySceneShader, /dryPrepass|@group\(1\)/,
    "the inline path must carry no prepass declarations, bindings, or code");
});

test("reduced scales add the prepass entry and guided upsample while keeping every inline fallback", () => {
  for (const scale of [0.5, 0.25, 0.125] as const) {
    const reduced = createSvoDrySceneFragmentWGSL(scale);
    assert.match(reduced, /@group\(1\) @binding\(0\) var dryPrepassVisibilityKeyTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(1\) var dryPrepassGeometryTexture:texture_2d<f32>/);
    assert.match(reduced, /@group\(1\) @binding\(2\) var dryPrepassIdentityTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(3\) var dryPrepassRadianceTexture:texture_2d<f32>/);
    assert.match(reduced, /@fragment fn dryPrepassMain/);
    assert.match(reduced, /@fragment fn dryPrepassShadeMain/,
      "opaque shading must have a separate reduced-rate entry point so it cannot inflate cone-pass register pressure");
    assert.ok(reduced.includes(createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, fluidCoverage: true })),
      "the reduced variant must embed the identical optimized marcher block");
    assert.match(reduced, /if\(weightSum<0\.05\)\{dryConeFallback=1u;return;\}/,
      "silhouette pixels below the guidance-weight threshold must fall back to exact inline cones");
    // Deliberate cone-banding fix: shadow-cone origins escape the receiver's
    // trilinear support along the geometric normal, and finite emitters clear
    // the march end by one cone-support width before marching.
    assert.match(reduced, /let cone=dryConeVisibility\(ray\.origin_m\+geometricNormal\*coneEscape_m,towardLight,dry\.tuningRays1\.y,coneMax_m,geometricNormal,finiteDistance_m>0\.0\)/,
      "the inline shadow cone must remain the fallback for fallback-band pixels");
    assert.match(reduced, /dryCurrentLightSlot<8u/,
      "every user-shadable light slot must reuse the reduced-rate visibility cache");
    assert.match(reduced, /if\(index<4u\)\{return dryPrepassData0\[index\];\}[^]*if\(index<8u\)\{return dryPrepassData1\[index-4u\];\}[^]*dryPrepassData2\[min\(index-8u,3u\)\]/,
      "AO and all eight light slots must decode from their documented packed planes");
    assert.match(reduced, /if\(lightIndex<3u\)\{visibility0\[1u\+lightIndex\]=packedVisibility;\}[^]*else if\(lightIndex<7u\)\{visibility1\[lightIndex-3u\]=packedVisibility;\}[^]*else\{visibility2\.x=packedVisibility;\}/,
      "the prepass writer and full-resolution reader must agree on every packed light channel");
    assert.match(reduced, /fn dryPrepassQuantize7\(value:f32\)->u32\{return u32\(round\(clamp\(value,0\.0,1\.0\)\*127\.0\)\);\}/);
    assert.match(reduced, /fn dryPrepassPack\([^]*clamp\(data0\.x,0\.0,1\.0\)\*255\.0/);
    assert.match(reduced, /fn dryPrepassUnpack0[^]*\/255\.0[^]*\/127\.0[^]*fn dryPrepassUnpack1[^]*fn dryPrepassUnpack2/,
      "AO must retain eight bits while all eight lights round-trip through seven-bit lanes");
    assert.match(reduced, /let packed=textureLoad\(dryPrepassVisibilityKeyTexture,texel,0\)[^]*accumulated0\+=dryPrepassUnpack0\(packed\)[^]*accumulated1\+=dryPrepassUnpack1\(packed\)[^]*accumulated2\+=dryPrepassUnpack2\(packed\)/,
      "one coherent integer fetch must provide every visibility lane and guide the 2x2 reconstruction");
    assert.match(reduced, /let raw=select\(dryPrepassChannel\(1u\+dryCurrentLightSlot\),0\.0,prepassRigidBlocker\.t<ray\.tMax_m\)/,
      "rigid-body blocker terms stay inline at full resolution on the upsampled shadow path");
    assert.match(reduced, /prepassUnblocked\+=select\(1\.0,0\.0,prepassRigidBlocker\.t<prepassRadius\)/,
      "rigid AO blocker sampling stays inline at full resolution on the upsampled AO path");
    assert.match(reduced, /if\(weight>bestRadianceWeight&&dryPrepassIdentityMatches\(textureLoad\(dryPrepassIdentityTexture,texel,0\)\.x,u32\(round\(geometry\.w\)\),hit\)\)\{bestRadianceWeight=weight;bestRadianceTexel=texel;\}/,
      "radiance reconstruction must reject exact material, owner, feature, field, or motion identity mismatches");
    assert.match(reduced, /accumulated2\+=dryPrepassUnpack2\(packed\)\*weight;[^]*if\(weight>bestRadianceWeight&&dryPrepassIdentityMatches/,
      "depth/normal-guided visibility may cross identity boundaries without authorizing radiance reuse");
    assert.match(reduced, /fn dryPrepassPackIdentity\(hit:DryHit\)->u32\{return \(hit\.materialId&0xffffu\)\|\(\(hit\.ownerId&0xffffu\)<<16u\);\}/);
    assert.match(reduced, /let opaque=DryHit\(geometry\.x,dryPrepassDecodeNormal\(geometry\.yz\),identity&0xffffu,identity>>16u/);
    assert.match(reduced, /return vec4f\(shadeDryOpaque\(opaque,ro,rd\),opaque\.t\)/,
      "the isolated reduced-rate pass must shade the reconstructed coarse hit without another primary trace");
    assert.match(reduced, /if\(hit\.t>=DRY_MISS\)[^]*dryPrepassRadianceState==1u&&hit\.motionKind==DRY_GBUFFER_MOTION_STATIC[^]*let position=ro\+rd\*hit\.t;let surface=dryEvaluateSurfaceMaterial/,
      "exact-matched static radiance must return before full-resolution procedural material evaluation");
  }
});

test("prepass target contract and sizing", () => {
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.visibilityFormat, "rg32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.visibilityTargetCount, 1);
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.geometryFormat, "rgba16float");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.identityFormat, "r32uint");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.radianceFormat, "rgba16float");
  assert.equal(SVO_DRY_CONE_PREPASS_CONTRACT.maximumPrepassLights, 8);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.5), [640, 360]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.25), [320, 180]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 0.125), [160, 90]);
  assert.deepEqual(svoConePrepassSize(1281, 721, 0.5), [641, 361]);
  assert.deepEqual(svoConePrepassSize(1281, 721, 0.125), [160, 90]);
  assert.deepEqual(svoConePrepassSize(1, 1, 0.25), [1, 1], "prepass targets never collapse below 1x1");
  assert.deepEqual(svoConePrepassSize(1, 1, 0.125), [1, 1]);
  assert.deepEqual(svoConePrepassSize(1280, 720, 1), [1280, 720]);
});

function mockSource(): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 8,
    pbrMaterials: { binding: resource, count: 8, strideBytes: SVO_MATERIAL_RECORD_STRIDE_BYTES, revision: 1 },
    structural: {
      control: resource, nodes: resource, leaves: resource, geometry: resource,
      velocity: resource, materialOwners: resource, fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [0, 0, 0], cellSize_m: [.1, .1, .1], dimensionsCells: [16, 16, 16], brickSize: 8, maximumDepth: 1 },
      capacities: { nodes: 8, leaves: 8, geometryVoxels: 4096, velocityVoxels: 4096, materialOwnerVoxels: 4096, fluidLeafStates: 8 },
      strides: { control: 4, node: 32, leaf: 16, geometry: 16, velocity: 16, materialOwner: 4, fluidLeafState: 4 },
      fields: {
        topology: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        staticGeometry: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        materialOwner: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        dynamicSolid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        coarseFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
        fineFluid: { residency: "unavailable", validity: "unavailable", revision: 0 },
      },
      generation: { published: 1, completed: 1 },
    },
  } as unknown as SparseVoxelRenderSource;
}

test("the cone-lighting scale is an optional lighting option that defaults to the inline path", () => {
  const previousBufferUsage = globalThis.GPUBufferUsage, previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 },
    GPUTextureUsage: { TEXTURE_BINDING: 1, RENDER_ATTACHMENT: 2 },
  });
  const writes: Array<{ label?: string }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) { return { label: descriptor.label, destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    queue: {
      writeBuffer(target: { label?: string }) { writes.push({ label: target.label }); },
    },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    renderer.setSource(mockSource(), svoDrySceneFixture);
    assert.equal(renderer.coneLightingScale, 1, "callers without the option keep the historical inline path");
    const paramsWrites = () => writes.filter(({ label }) => label === "Sparse voxel dry scene parameters").length;
    const beforeRepeat = paramsWrites();
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    assert.equal(paramsWrites(), beforeRepeat, "unchanged options (including implicit scale 1) must short-circuit");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5 });
    assert.equal(renderer.coneLightingScale, 0.5);
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.5 });
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true, coneLightingScale: 0.125 });
    assert.equal(renderer.coneLightingScale, 0.125, "the 8x8 option reaches the renderer without normalization loss");
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    assert.equal(renderer.coneLightingScale, 1, "omitting the option returns to the inline path");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("reduced shader variants compile with both entry points on the GPU backend", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU cone-prepass checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    for (const scale of [0.5, 0.25, 0.125] as const) {
      const code = createSvoDrySceneFragmentWGSL(scale);
      const module = device.createShaderModule({ label: `Cone-prepass dry shader validation x${scale}`, code });
      const info = await module.getCompilationInfo();
      assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    }
  } finally { device.destroy(); }
});
