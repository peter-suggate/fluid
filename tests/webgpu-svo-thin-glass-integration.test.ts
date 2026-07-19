import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { SVO_SCENE_GLASS_MAXIMUM_PANES } from "../lib/svo-scene-glass";
import { SVO_THIN_GLASS_RECORD_WORDS } from "../lib/svo-thin-glass";
import {
  canEncodeSparseVoxelDryScene,
  SparseVoxelDrySceneRenderer,
  svoDrySceneShader,
  type SparseVoxelDrySceneData,
} from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");

function structuralSource(): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 2,
    materials: resource,
    pbrMaterials: { binding: resource, count: 8, strideBytes: 96, revision: 1 },
    structural: {
      control: resource,
      nodes: resource,
      leaves: resource,
      geometry: resource,
      velocity: resource,
      materialOwners: resource,
      fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [-1, 0, -1], cellSize_m: [0.02, 0.04, 0.03], dimensionsCells: [64, 64, 64], brickSize: 16, maximumDepth: 4 },
      capacities: { nodes: 64, leaves: 32, geometryVoxels: 1024, velocityVoxels: 1024, materialOwnerVoxels: 1024, fluidLeafStates: 32 },
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

test("production scene construction uploads pane records and exposes an explicit lab cutout fallback", () => {
  assert.match(rendererSource, /buildSvoSceneGlass\(scene,\{cellSize_m:sparseSceneSource\?\.structural\?\.domain\.cellSize_m\}\)/);
  assert.match(rendererSource, /glassRecords:sceneGlass\.packedRecords,glassCacheKey:sceneGlass\.cacheKey/);
  assert.match(rendererSource, /const compositorOwnedGlass=sceneGlass\.metadata\.filter\(\(\{role\}\)=>role==="container-pane"\|\|role==="container-top"\)/);
  assert.match(rendererSource, /primaryCompositeOwnedGlassPaneIdBase:compositorOwnedGlass\[0\]\?\.paneId,primaryCompositeOwnedGlassPaneCount:compositorOwnedGlass\.length/,
    "the existing nearest-vessel-pane compositor must retain ownership under camera orbit");
  const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(waterSource, /fn compositeFrontGlass\(color:vec3f,ro:vec3f,rd:vec3f,sceneDepth:f32\)->vec3f/);
  assert.match(waterSource, /return finish\(compositeFrontGlass\(scene\.rgb,ro,rd,scene\.a\),ndc\)/,
    "the post-dry-scene compositor must still render vessel glass when no water interface is present");
  assert.match(rendererSource, /svoGlassSupported=!sceneGlass\.metadata\.some\(\(\{opaqueCutoutKey\}\)=>Boolean\(opaqueCutoutKey\)\)/);
  assert.match(rendererSource, /fallbackReason: "unsupported-glass-cutout"/);
  assert.match(panelSource, /"unsupported-glass-cutout": "authored glazing needs an opaque shell cutout"/);
});

test("pane ABI validation accepts empty gardens and rejects partial or over-capacity uploads", () => {
  const source = structuralSource();
  const base: SparseVoxelDrySceneData = { primitiveRecords: new Uint32Array(16), ownerBase: 1 };
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, glassRecords: new Uint32Array(0) }), true);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, glassRecords: new Uint32Array(SVO_THIN_GLASS_RECORD_WORDS - 1) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, {
    ...base,
    glassRecords: new Uint32Array((SVO_SCENE_GLASS_MAXIMUM_PANES + 1) * SVO_THIN_GLASS_RECORD_WORDS),
  }), false);
});

test("glass upload cache is reused by static revision and destroyed on detach", () => {
  const previousUsage = globalThis.GPUBufferUsage;
  Object.assign(globalThis, { GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 } });
  const created: Array<{ label?: string; destroyed: boolean }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) {
      const buffer = { label: descriptor.label, destroyed: false, destroy() { buffer.destroyed = true; } };
      created.push(buffer);
      return buffer;
    },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer);
    const scene: SparseVoxelDrySceneData = {
      primitiveRecords: new Uint32Array(16), ownerBase: 1,
      glassRecords: new Uint32Array(SVO_THIN_GLASS_RECORD_WORDS), glassCacheKey: "glass:v1",
    };
    renderer.setSource(structuralSource(), scene);
    const firstGlass = created.find(({ label }) => label === "Sparse voxel thin-glass panes");
    assert.ok(firstGlass);
    renderer.setSource(structuralSource(), scene);
    assert.equal(created.filter(({ label }) => label === "Sparse voxel thin-glass panes").length, 1,
      "unchanged static glass must retain its GPU upload");
    assert.equal(firstGlass.destroyed, false);
    renderer.setSource(undefined, undefined);
    assert.equal(firstGlass.destroyed, true, "solver detach must retire the pane buffer before source buffers");
    renderer.destroy();
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage });
  }
});

test("primary pane optics are exact, two-sided, identity preserving, and one-query bounded", () => {
  assert.match(svoDrySceneShader, /fn traceGlass\([^]*svoThinGlassIntersect/);
  assert.match(svoDrySceneShader, /let compositeOwned=skipCompositeOwned&&dry\.terrain\.w>0u&&paneId>=dry\.terrain\.z&&paneId-dry\.terrain\.z<dry\.terrain\.w/);
  assert.match(svoDrySceneShader, /fn shadeThinGlass\([^]*svoThinGlassOptics\(record,glass\.hit,incidentIor\)/);
  assert.match(svoDrySceneShader, /reflected\*optics\.fresnel\+transmitted\*optics\.netTransmittance/);
  assert.match(svoDrySceneShader, /svoThinGlassMaterialId\(record\),svoThinGlassOwnerId\(record\),svoThinGlassPaneId\(record\)/);
  assert.match(svoDrySceneShader, /fn dryThinGlassIncidentIor\(medium:u32\)->f32[^]*medium==DRY_MEDIUM_WATER/,
    "the future fluid-boundary adapter must select water IOR only after coincident exits are resolved");

  const opticsStart = svoDrySceneShader.indexOf("fn shadeThinGlass(");
  const opticsEnd = svoDrySceneShader.indexOf("struct VertexOut", opticsStart);
  const optics = svoDrySceneShader.slice(opticsStart, opticsEnd);
  assert.ok((optics.match(/traceOpaqueScene\(/g) ?? []).length <= 1,
    "the first glass slice permits at most one transmitted scene query");
  assert.match(optics, /shadeDryOpaque\(opaque,ro,rd\)/,
    "a collapsed sheet must reuse the already-resolved collinear opaque hit");
  assert.match(optics, /dryEnvironment\(reflect\(rd,glass\.hit\.geometricNormal\),\.04\)/,
    "reflection must use a bounded environment fallback instead of recursive scene shading");
  assert.doesNotMatch(optics, /DryHit\([^)]*svoThinGlass|shadeThinGlass\([^)]*shadeThinGlass/,
    "a pane is never substituted as opaque or recursively shaded");
});

test("pane shadows use nearest-event bounded transmission rather than opaque substitution", () => {
  assert.match(svoDrySceneShader, /traceGlass\(ray\.origin_m,ray\.direction,tMin_m,bestT,false\)/);
  assert.match(svoDrySceneShader, /dryVisibilityTransmissionStep\([^]*glassTransmission/);
  assert.match(svoDrySceneShader, /SvoVisibilityBudget\(256u,64u,2048u,4u\),true/);
});
