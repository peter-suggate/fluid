import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { pathToFileURL } from "node:url";

import { SVO_MATERIAL_RECORD_STRIDE_BYTES } from "../lib/svo-material-abi";
import {
  SparseVoxelDrySceneRenderer,
  SVO_DRY_SCENE_PARAMS_LAYOUT,
  SVO_DRY_VISIBILITY_FLAGS,
  svoDrySceneShader,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import { candidateBackedDrySceneFixture } from "./svo-dry-scene-test-fixture";

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
  assert.deepEqual(SVO_DRY_VISIBILITY_FLAGS, { exactContact: 1, exactShadow: 2, coneLightingRequested: 4, ambientOcclusion: 8 });
  assert.match(drySource, /this\.lightingMode === "cone" && \(shadowsEnabled \|\| ambientOcclusionEnabled\) \? SVO_DRY_VISIBILITY_FLAGS\.coneLightingRequested : 0/);
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
    renderer.setSource(source(), candidateBackedDrySceneFixture);
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
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousBufferUsage, GPUTextureUsage: previousTextureUsage });
  }
});

test("missing or stale node-mip samples enter exact bounded visibility rather than returning lit", () => {
  const start = svoDrySceneShader.indexOf("fn dryLightVisibility(");
  const end = svoDrySceneShader.indexOf("fn dryContactVisibilityRadius", start);
  const visibility = svoDrySceneShader.slice(start, end);
  assert.match(visibility, /let cone=dryConeVisibility\([^]*if\(cone\.valid!=0u\)\{[^]*return vec3f\(cone\.transmittance\);\}/);
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

test("sparse-brick world exposes, accounts, and retires its optional node-mip capability", () => {
  assert.match(sourceAbi, /nodeMipPyramid\?: import\("\.\/webgpu-svo-node-mip-pyramid"\)\.WebGpuSvoNodeMipVisibleGeneration/);
  assert.match(worldSource, /nodeMipPyramid: this\.nodeMipPyramid\?\.visibleGeneration\(\)/);
  assert.match(worldSource, /nodeMipPyramid: this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0/);
  assert.match(worldSource, /\+ \(this\.nodeMipPyramid\?\.telemetry\(\)\.allocatedBytes \?\? 0\)/);
  assert.match(worldSource, /this\.nodeMipPyramid\?\.destroy\(\)/);
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
  } finally { device.destroy(); }
});
