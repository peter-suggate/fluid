import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import { SVO_FLUID_COVERAGE_LAYOUT } from "../lib/svo-fluid-coverage";
import {
  SparseVoxelDrySceneRenderer,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  SVO_DRY_VISIBILITY_FLAGS,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const drySource = readFileSync(new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url), "utf8");
const worldSource = readFileSync(new URL("../lib/webgpu-octree-sparse-bricks.ts", import.meta.url), "utf8");
const sourceAbi = readFileSync(new URL("../lib/webgpu-voxel-debug.ts", import.meta.url), "utf8");

function source(): SparseVoxelRenderSource {
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

test("lighting quality, shadows, and ambient occlusion write independent visibility flags", () => {
  assert.deepEqual(SVO_DRY_VISIBILITY_FLAGS, {
    exactContact: 1, exactShadow: 2, coneLightingRequested: 4, ambientOcclusion: 8,
    globalIllumination: 16, globalIlluminationOcclusion: 32, globalIlluminationRequested: 64,
  });
  assert.match(drySource, /const coneFallback = this\.lightingMode === "cone" \|\| \(this\.lightingMode === "gi" && !giReady\)/);
  const previousBufferUsage = globalThis.GPUBufferUsage, previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4, MAP_READ: 8 },
    GPUTextureUsage: { TEXTURE_BINDING: 1 },
  });
  const writes: Array<{ label?: string; words: Uint32Array }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) { return { label: descriptor.label, destroy() {} }; },
    createTexture() { return { createView() { return {}; }, destroy() {} }; },
    createSampler() { return {}; },
    queue: {
      writeBuffer(target: { label?: string }, _offset: number, data: ArrayBuffer | ArrayBufferView) {
        const bytes = data instanceof ArrayBuffer ? new Uint8Array(data) : new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        writes.push({ label: target.label, words: new Uint32Array(bytes.slice().buffer) });
      },
    },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    renderer.setLightingMode("cone");
    renderer.setSource(source(), svoDrySceneFixture);
    const params = () => writes.filter(({ label }) => label === "Sparse voxel dry scene parameters");
    const flagWord = (write: { words: Uint32Array }) => write.words[SVO_DRY_SCENE_PARAMS_LAYOUT.materialPublicationWordOffset + 3];
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.exactShadow, SVO_DRY_VISIBILITY_FLAGS.exactShadow);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion, SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested);
    renderer.setLightingOptions({ shadowsEnabled: false, ambientOcclusionEnabled: true });
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.exactShadow, 0);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested);
    renderer.setLightingOptions({ shadowsEnabled: false, ambientOcclusionEnabled: false });
    assert.equal(flagWord(params().at(-1)!) & (SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.exactShadow | SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion), 0);
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: true });
    renderer.setLightingMode("direct");
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.exactShadow, SVO_DRY_VISIBILITY_FLAGS.exactShadow);
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, 0);
    renderer.setLightingMode("cone");
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested);
    renderer.setLightingMode("gi");
    const fallbackFlags = flagWord(params().at(-1)!);
    assert.equal(fallbackFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested,
      SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested, "the probe can diagnose a requested GI atlas that is unavailable");
    assert.equal(fallbackFlags & SVO_DRY_VISIBILITY_FLAGS.globalIllumination, 0);
    renderer.setLightingMode("cone");
    const giSource = source() as unknown as SparseVoxelRenderSource & Record<string, unknown>;
    const plan = { generation: 1, complete: true, pages: [{ key: { generation: 1, level: 0, coordinate: [0, 0, 0] }, slot: 0 }], atlas: { texels: [10, 10, 10] } };
    Object.assign(giSource, {
      nodeMipPyramid: { generation: 1, plan, worldOrigin_m: [0, 0, 0] },
      tetrahedralRadiance: { generation: 1, plan, views: [{}, {}, {}, {}] },
    });
    renderer.setSource(giSource as unknown as SparseVoxelRenderSource, svoDrySceneFixture);
    renderer.setLightingMode("gi");
    const giFlags = flagWord(params().at(-1)!);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIllumination, SVO_DRY_VISIBILITY_FLAGS.globalIllumination);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested,
      SVO_DRY_VISIBILITY_FLAGS.globalIlluminationRequested);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested, SVO_DRY_VISIBILITY_FLAGS.coneLightingRequested);
    assert.equal(giFlags & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion,
      SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion, "AO enables broad GI-cone visibility");
    assert.equal(giFlags & (SVO_DRY_VISIBILITY_FLAGS.exactContact | SVO_DRY_VISIBILITY_FLAGS.ambientOcclusion), 0,
      "GI replaces the standalone AO/contact cones");
    assert.deepEqual([...params().at(-1)!.words.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.tetrahedralRadianceWordOffset + 4)], [1, 1, 0, 0]);
    const giParams = new Float32Array(params().at(-1)!.words.buffer);
    assert.deepEqual([...giParams.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.giLightingWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.giLightingWordOffset + 4)], [1.5, 0.8199999928474426, 0.6499999761581421, 0.8999999761581421]);
    assert.deepEqual([...giParams.slice(SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset,
      SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset + 2)], [1.0499999523162842, 4]);
    renderer.setLightingOptions({ shadowsEnabled: true, ambientOcclusionEnabled: false });
    assert.equal(flagWord(params().at(-1)!) & SVO_DRY_VISIBILITY_FLAGS.globalIlluminationOcclusion, 0,
      "the AO switch remains meaningful in GLOBAL mode");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

test("missing or stale node-mip samples enter exact bounded visibility rather than returning lit", () => {
  const start = svoDrySceneShader.indexOf("fn dryLightVisibility(");
  const end = svoDrySceneShader.indexOf("fn dryContactVisibilityRadius", start);
  const visibility = svoDrySceneShader.slice(start, end);
  // Only a valid cone short-circuits. Water attenuation multiplies that result
  // rather than replacing it, so an invalid cone still falls through to the
  // exact bounded traversal below instead of returning a lit surface.
  assert.match(visibility, /let cone=dryConeVisibility\([^]*if\(cone\.valid!=0u\)\{[^]*let raw=vec3f\(cone\.transmittance\)\*dryFluidTransmittance\(cone\.fluidDepth_m\);return mix\(vec3f\(1\.0\),raw,dry\.tuningRays0\.y\);\}/);
  assert.ok(visibility.indexOf("svoTraceVisibility", visibility.indexOf("dryConeVisibility")) > visibility.indexOf("dryConeVisibility"));
  assert.match(svoDrySceneShader, /fn dryNodeMipReady\(\)->bool\{return dry\.nodeMip\.w!=0u&&dry\.nodeMip\.x!=0u&&dry\.nodeMip\.x==publicationState\[2\]/,
    "cone use is fenced to the matching structural static-geometry revision");
});

test("cone steps reuse a matching mip page and search only on page, LOD, or generation changes", () => {
  const lookupStart = svoDrySceneShader.indexOf("fn dryNodeMipAt(");
  const lookupEnd = svoDrySceneShader.indexOf("struct DryConeVisibility", lookupStart);
  const lookup = svoDrySceneShader.slice(lookupStart, lookupEnd);
  assert.match(svoDrySceneShader, /struct DryNodeMipPageCache\{coordinate:vec3u,level:u32,pageOrigin:vec3u,generation:u32,resident:u32\}/);
  assert.match(lookup, /pageCache:ptr<function,DryNodeMipPageCache>/);
  assert.match(lookup, /generation!=dry\.nodeMip\.x\|\|\(\*pageCache\)\.level!=level\|\|any\(\(\*pageCache\)\.coordinate!=pageCoordinate\)/);
  assert.match(lookup, /\*pageCache=DryNodeMipPageCache\(pageCoordinate,level,vec3u\(0u\),dry\.nodeMip\.x,0u\);let pageIndex=dryNodeMipFind\(level,pageCoordinate\)/,
    "the queried key is cached as non-resident before searching so failed searches are reusable");
  assert.match(lookup, /pageIndex!=0xffffffffu[^]*\*pageCache=DryNodeMipPageCache\(pageCoordinate,level,entry\.pageOrigin,entry\.generation,1u\)/);
  assert.match(lookup, /if\(\(\*pageCache\)\.resident==0u\)\{return DryNodeMipLookup\(SvoNodeMipSample\(0\.0,0\.0,0\.0,0\.0\),1u\);\}/,
    "cached sparse-directory misses sample as transparent without another search");
  assert.match(svoDrySceneShader, /var pageCache=DryNodeMipPageCache\([^]*dryNodeMipAt\([^]*&pageCache\)/);

  const coherentSteps = Array.from({ length: 48 }, (_, step) => ({
    generation: 17,
    level: 0,
    coordinate: [Math.floor((0.75 + step) / 8), 0, 0] as const,
  }));
  let searches = 0;
  let previous: (typeof coherentSteps)[number] | undefined;
  for (const step of coherentSteps) {
    if (!previous || previous.generation !== step.generation || previous.level !== step.level
      || previous.coordinate.some((component, axis) => component !== step.coordinate[axis])) searches += 1;
    previous = step;
  }
  assert.equal(searches, 6, "an axis-aligned 48-step path searches once per crossed 8-voxel page");
  assert.equal(coherentSteps.length - searches, 42);
  assert.equal(1 - searches / coherentSteps.length, 0.875, "the representative path eliminates 87.5% of directory searches");

  const absentPageSteps = Array.from({ length: 12 }, () => ({ generation: 17, level: 0, coordinate: [9, 4, 2] as const }));
  searches = 0;
  let absentPrevious: (typeof absentPageSteps)[number] | undefined;
  for (const step of absentPageSteps) {
    if (!absentPrevious || absentPrevious.generation !== step.generation || absentPrevious.level !== step.level
      || absentPrevious.coordinate.some((component, axis) => component !== step.coordinate[axis])) searches += 1;
    absentPrevious = step;
  }
  assert.equal(searches, 1, "repeated samples in one non-resident sparse page reuse the negative lookup");
});

test("node-mip sampling publishes its own world origin inside the static uniform block", () => {
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipOriginWordOffset, 60);
  // The static-lighting block still ends exactly where it always did. Evolving
  // fluid coverage is appended past it rather than repacking anything below, so
  // every offset a cone-lighting frame reads is unmoved.
  assert.equal((SVO_DRY_SCENE_PARAMS_LAYOUT.nodeMipOriginWordOffset + 4) * 4, 256);
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset * 4, 256,
    "the fluid frame must start where the static block ends");
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.tuningWordOffset,
    SVO_DRY_SCENE_PARAMS_LAYOUT.fluidCoverageWordOffset + SVO_FLUID_COVERAGE_LAYOUT.frameWords,
    "runtime tuning must immediately follow the 12-word fluid frame");
  assert.equal(SVO_DRY_SCENE_PARAMS_LAYOUT.sizeBytes,
    (SVO_DRY_SCENE_PARAMS_LAYOUT.giConesWordOffset + 4) * Uint32Array.BYTES_PER_ELEMENT,
    "the uniform allocation must end after the GI controls");
  assert.match(drySource, /floats\.set\(nodeMip\?\.worldOrigin_m \?\? structural\.domain\.worldOrigin_m, SVO_DRY_SCENE_PARAMS_LAYOUT\.nodeMipOriginWordOffset\)/);
  assert.match(svoDrySceneShader, /virtualVoxel=\(position_m-dry\.nodeMipOrigin\.xyz\)/,
    "topology experiments must not reinterpret an unchanged opacity atlas in the structural tree's coordinate frame");
  assert.match(worldSource, /worldOrigin_m: nodeMipWorldOrigin_m/);
  assert.match(worldSource, /worldExtent_m: staticLightingDomain\.sceneDimensionsCells\.map/);
  assert.match(svoDrySceneShader, /fn dryNodeMipSceneExitDistance\([^]*maximum=minimum\+dry\.nodeMipExtent\.xyz/,
    "GI cones must survey the surrounding authored scene, not stop at the solver tank");
});

test("sparse-brick world exposes, accounts, and retires its optional node-mip capability", () => {
  assert.match(sourceAbi, /nodeMipPyramid\?: import\("\.\/webgpu-svo-node-mip-pyramid"\)\.WebGpuSvoNodeMipVisibleGeneration/);
  assert.match(worldSource, /nodeMipPyramid: this\.nodeMipPyramid\?\.visibleGeneration\(\)/);
  assert.match(worldSource, /nodeMipPyramid: this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0/);
  assert.match(worldSource, /\+ \(this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0\)/);
  assert.match(worldSource, /this\.nodeMipPyramid\?\.destroy\(\)/);
  assert.match(sourceAbi, /tetrahedralRadiance\?: import\("\.\/webgpu-svo-tetrahedral-radiance"\)\.WebGpuSvoTetrahedralRadianceVisibleGeneration/);
  assert.match(worldSource, /tetrahedralRadiance: this\.tetrahedralRadiance\?\.visibleGeneration\(\)/);
  assert.match(worldSource, /page\.certifiedBlack\) tetrahedralRadiance\.certifyBlackPage\(page\.key\)/,
    "certified-black pages must consume no queue writes");
  assert.match(worldSource, /primaryDirectionalLight: \{[^]*towardLightDirection: scene\.lighting\?\.directional\?\.direction[^]*colorLinear: scene\.lighting\?\.directional\?\.colorLinear[^]*intensity: scene\.lighting\?\.directional\?\.intensity/,
    "the published atlas must contain shadowed first-bounce direct exitance as well as authored emission");
  assert.match(worldSource, /catch \{[^]*nodeMipPyramid\?\.destroy\(\);[^]*nodeMipPyramid = undefined;/,
    "failed derived publication must be cleaned up without disturbing canonical world construction");
});

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("production dry shader compiles sampled node-mip atlas and uint directory bindings", {
  skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU cone-lighting checks",
}, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]), adapter = await gpu.requestAdapter({ powerPreference: "high-performance" }); assert.ok(adapter);
  const device = await adapter.requestDevice();
  try {
    const module = device.createShaderModule({ label: "Cone-lighting dry shader validation", code: svoDrySceneShader });
    const info = await module.getCompilationInfo();
    assert.deepEqual(info.messages.filter(({ type }) => type === "error"), []);
    assert.match(svoDrySceneShader, /@binding\(16\) var nodeMipAtlas:texture_3d<f32>/);
    assert.match(svoDrySceneShader, /@binding\(18\) var nodeMipDirectory:texture_2d<u32>/);
    assert.match(svoDrySceneShader, /@binding\(20\) var nodeMipPageTable:texture_3d<u32>/);
  } finally { device.destroy(); }
});
