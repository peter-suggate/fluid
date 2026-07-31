import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
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

const rendererSource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");

test("scale 1 preserves the production shader byte-for-byte (fingerprint contract)", () => {
  assert.equal(createSvoDrySceneFragmentWGSL(1), svoDrySceneShader,
    "the factory default must return the exact historical string so the bit-exact frame fingerprint reproduces");
  assert.doesNotMatch(svoDrySceneShader, /dryPrepass|@group\(1\)/,
    "the inline path must carry no prepass declarations, bindings, or code");
  assert.doesNotMatch(svoDrySceneShader, /anyBodyBlockerIgnoring/,
    "the reduced-only blocker specialization must not perturb the scale-1 shader");
});

test("reduced scales add the prepass entry and guided upsample while keeping every inline fallback", () => {
  for (const scale of [0.5, 0.25, 0.125] as const) {
    const reduced = createSvoDrySceneFragmentWGSL(scale);
    assert.match(reduced, /@group\(1\) @binding\(0\) var dryPrepassVisibilityKeyTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(1\) var dryPrepassGeometryTexture:texture_2d<f32>/);
    assert.match(reduced, /@group\(1\) @binding\(2\) var dryPrepassIdentityTexture:texture_2d<u32>/);
    assert.match(reduced, /@group\(1\) @binding\(3\) var dryPrepassRadianceTexture:texture_2d<f32>/);
    assert.match(reduced, /@fragment fn dryPrepassGeometryMain/,
      "the reduced primary trace must be isolated from cone-march register pressure");
    assert.match(reduced, /@fragment fn dryPrepassVisibilityMain/,
      "the cone marcher must consume the freshly traced reduced geometry in its own phase");
    assert.match(reduced, /dryPrepassVisibilityMain[^]*textureLoad\(dryPrepassGeometryTexture,coordinate,0\)[^]*textureLoad\(dryPrepassIdentityTexture,coordinate,0\)[^]*return dryPrepassTraceVisibility/,
      "the cone phase reconstructs the dynamic hit instead of tracing primary visibility again");
    assert.match(reduced, /@fragment fn dryPrepassShadeMain/,
      "opaque shading must have a separate reduced-rate entry point so it cannot inflate cone-pass register pressure");
    assert.ok(reduced.includes(createSvoDryConeMarcherWGSL({ branchlessMorton: true, rangedDirectorySearch: true, fluidCoverage: true, directPageTable: true })),
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
    assert.match(reduced, /let prepassRigidBlocked=anyBodyBlockerIgnoring\(ray\.origin_m,towardLight,ownerId,ray\.tMax_m\);let raw=select\(dryPrepassChannel\(1u\+dryCurrentLightSlot\),0\.0,prepassRigidBlocked\)/,
      "rigid-body blocker terms stay inline at full resolution on the upsampled shadow path");
    assert.match(reduced, /prepassUnblocked\+=select\(1\.0,0\.0,prepassRigidBlocked\)/,
      "rigid AO blocker sampling stays inline at full resolution on the upsampled AO path");
    assert.match(reduced, /fn anyBodyBlockerIgnoring\([^]*if\(bodyHit\(ro,rd,body\)\.t<tMax\)\{return true;\}/,
      "blocker-only paths must early out without carrying the full nearest-hit payload");
    assert.match(reduced, /let identityMatches=dryPrepassIdentityMatches\(textureLoad\(dryPrepassIdentityTexture,texel,0\)\.x,u32\(round\(geometry\.w\)\),hit\)/,
      "radiance reconstruction must reject exact material, owner, feature, field, or motion identity mismatches");
    assert.match(reduced, /accumulated2\+=dryPrepassUnpack2\(packed\)\*weight;[^]*if\(identityMatches\)/,
      "depth/normal-guided visibility may cross identity boundaries without authorizing radiance reuse");
    assert.match(reduced, /if\(bilinear>1e-6&&!identityMatches\)\{linearSafe=0u;\}[^]*if\(bilinear>1e-6&&\(depthWeight<0\.25\|\|normalWeight<0\.25\)\)\{linearSafe=0u;\}/,
      "hardware filtering must be disabled before it can cross an identity, depth, or normal edge");
    assert.match(reduced, /textureSampleLevel\(dryPrepassRadianceTexture,nodeMipSampler,pixel\/max\(uniforms\.viewport\.xy,vec2f\(1\.0\)\),0\.0\)/,
      "the gated-linear mode must use the resident linear sampler without another pass");
    assert.match(reduced, /accumulatedRadiance\+=textureLoad\(dryPrepassRadianceTexture,texel,0\)\*weight;radianceWeightSum\+=weight/,
      "joint-bilateral mode must reuse the visibility guide weights for edge-aware radiance reconstruction");
    assert.match(reduced, /materialPublication\.w&16u\)!=0u\)\{accumulatedGi\+=textureLoad\(dryPrepassRadianceTexture,texel,0\)\*weight;giWeightSum\+=weight/,
      "GLOBAL must reconstruct current-frame environmental GI only from exact-identity neighbours");
    assert.match(reduced, /if\(giWeightSum>=0\.05\)\{dryPrepassGi=accumulatedGi\/giWeightSum;dryPrepassGiState=1u;\}/,
      "insufficient geometry-guided GI support must leave the full-resolution cone fallback active");
    assert.match(reduced, /if\(dryPrepassGiState==1u\)\{return DryGlobalIllumination/,
      "a valid reduced GI surface summary must return before any full-resolution 3D cone taps");
    assert.match(reduced, /let weight=bilinear\*select\(guidedWeight,1\.0,dry\.tuningCounts2\.w==3u\)/,
      "wide relight must aggressively reconstruct shadow factors with unmodified bilinear weights");
    assert.match(reduced, /tuningCounts2\.w!=3u&&dry\.tuningCounts2\.w!=4u&&bestRadianceWeight>0\.0/,
      "both relight modes must bypass every reduced-radiance shortcut");
    assert.match(reduced, /fn dryPrepassPackIdentity\(hit:DryHit\)->u32\{return \(hit\.materialId&0xffffu\)\|\(\(hit\.ownerId&0xffffu\)<<16u\);\}/);
    assert.match(reduced, /let opaque=DryHit\(geometry\.x,dryPrepassDecodeNormal\(geometry\.yz\),identity&0xffffu,identity>>16u/);
    assert.match(reduced, /return vec4f\(shadeDryOpaque\(opaque,ro,rd\),opaque\.t\)/,
      "the isolated reduced-rate pass must shade the reconstructed coarse hit without another primary trace");
    assert.match(reduced, /materialPublication\.w&16u\)!=0u\)\{dryPrepassGiState=0u;let ignoredBodyOwner=select\(DRY_OWNER_NONE,opaque\.ownerId,opaque\.motionKind==DRY_GBUFFER_MOTION_RIGID\);let gi=dryGlobalIllumination\(ro\+rd\*opaque\.t,opaque\.normal,ignoredBodyOwner\);return vec4f\(gi\.radiance,gi\.visibility\)/,
      "GLOBAL's reduced target must store only reusable indirect radiance and occlusion, not a baked material closure");
    assert.match(reduced, /if\(hit\.t>=DRY_MISS\)[^]*dryPrepassRadianceState==1u&&hit\.motionKind==DRY_GBUFFER_MOTION_STATIC[^]*let position=ro\+rd\*hit\.t;let surface=dryEvaluateSurfaceMaterial/,
      "exact-matched static radiance must return before full-resolution procedural material evaluation");
  }
  assert.match(rendererSource, /this\.lightingMode === "gi"[^]*coneRadianceReconstruction !== "wide-relight"[^]*Sparse voxel reduced-rate opaque shading/,
    "GLOBAL must produce the reduced GI target even when material radiance remains full-resolution relight");
  assert.match(rendererSource, /if \(this\.lightingMode === "gi"\)[^]*Sparse voxel reduced-rate environmental GI[^]*conePrepassShadePipeline/,
    "split GLOBAL must evaluate environmental GI after current-frame compact visibility and before deferred lighting");
});

test("automatic relight composition contains the isolated primary and lighting entries", () => {
  const automatic = createSvoDrySceneFragmentWGSL(0.5, "hybrid", "off", "auto-relight");
  assert.match(automatic, /@fragment fn dryVisibilityMain/);
  assert.match(automatic, /@fragment fn dryLightingMain/);
  assert.match(automatic, /@group\(2\) @binding\(0\) var drySplitGeometryWrite/);
  assert.match(automatic, /@group\(2\) @binding\(1\) var drySplitGeometryRead/);
  assert.match(automatic, /@compute @workgroup_size\(8,8\) fn dryPrepassCoherentMain/,
    "2x2 relighting must consume coherent current-frame primary hits without tracing them again");
  assert.match(automatic, /atomicAdd\(&dryPrepassBoundaryQueue\.count,1u\)/,
    "ambiguous 2x2 texels must compact into a boundary queue instead of diverging through primary traversal");
  assert.match(automatic, /@compute @workgroup_size\(64\) fn dryPrepassBoundaryMain[^]*traceOpaqueScene/,
    "only compacted boundary texels may retrace the exact 2x2 centre ray");
  assert.match(rendererSource, /if \(usePrepass && !useSplit\)[^]*if \(useSplit\)[^]*Sparse voxel compact cone visibility[^]*conePrepassCoherentPipeline[^]*conePrepassBoundaryPipeline/,
    "the relight path must run full-resolution primary visibility before the two compact cone kernels");
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
