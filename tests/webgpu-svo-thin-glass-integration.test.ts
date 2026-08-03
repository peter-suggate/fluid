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
import { svoDrySceneFixture } from "./svo-dry-scene-test-fixture";

const rendererSource = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
const panelSource = readFileSync(new URL("../components/VisualPanel.tsx", import.meta.url), "utf8");

function structuralSource(): SparseVoxelRenderSource {
  const resource = { buffer: {} as GPUBuffer };
  return {
    materialCount: 2,
    materials: resource,
    pbrMaterials: { binding: resource, count: 8, strideBytes: 96, revision: 1 },
    structural: {
      structure: resource,
      structureOffsetsWords: { control: 0, publication: 64, nodes: 128, leaves: 640 },
      control: resource,
      nodes: resource,
      leaves: resource,
      geometry: resource,
      sceneGeometry: resource,
      velocity: resource,
      materialOwners: resource,
      sceneMaterialOwners: resource,
      fluidLeafStates: resource,
      publication: { state: resource, byteLength: 32 },
      domain: { worldOrigin_m: [-1, 0, -1], cellSize_m: [0.02, 0.04, 0.03], dimensionsCells: [64, 64, 64], brickSize: 16, maximumDepth: 4 },
      capacities: { nodes: 64, leaves: 32, geometryVoxels: 1024, velocityVoxels: 1024, materialOwnerVoxels: 1024, fluidLeafStates: 32 },
      strides: { control: 4, node: 32, leaf: 16, geometry: 16, velocity: 16, materialOwner: 4, fluidLeafState: 4 },
      fields: {
        topology: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
        sceneGeometry: { residency: "all-published-leaves", validity: "published-generation", revision: 1 },
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
  assert.match(rendererSource, /buildSvoSceneGlass\(scene, \{ cellSize_m: source\.structural\?\.domain\.cellSize_m \}\)/);
  assert.match(rendererSource, /glassRecords: sceneGlass\.packedRecords/);
  assert.match(rendererSource, /const compositorOwnedGlass = sceneGlass\.metadata\.filter/);
  assert.match(rendererSource, /primaryCompositeOwnedGlassPaneIdBase: compositorOwnedGlass\[0\]\?\.paneId/,
    "the existing nearest-vessel-pane compositor must retain ownership under camera orbit");
  const waterSource = readFileSync(new URL("../lib/webgpu-water-pipeline.ts", import.meta.url), "utf8");
  assert.match(waterSource, /fn compositeFrontGlass\(color:vec3f,ro:vec3f,rd:vec3f,sceneDepth:f32\)->vec3f/);
  assert.match(waterSource, /return finish\(compositeFrontGlass\(scene\.rgb,ro,rd,scene\.a\),ndc\)/,
    "the post-dry-scene compositor must still render vessel glass when no water interface is present");
  assert.match(rendererSource, /this\.svoGlassSupported = !sceneGlass\.metadata\.some/);
  assert.match(rendererSource, /failureReason: "unsupported-glass-cutout"/);
  assert.match(panelSource, /"unsupported-glass-cutout": "authored glazing needs an opaque shell cutout"/);
});

test("pane ABI validation accepts empty gardens and rejects partial or over-capacity uploads", () => {
  const source = structuralSource();
  const base: SparseVoxelDrySceneData = { ...svoDrySceneFixture, ownerBase: 1 };
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, glassRecords: new Uint32Array(0) }), true);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...base, glassRecords: new Uint32Array(SVO_THIN_GLASS_RECORD_WORDS - 1) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, {
    ...base,
    glassRecords: new Uint32Array((SVO_SCENE_GLASS_MAXIMUM_PANES + 1) * SVO_THIN_GLASS_RECORD_WORDS),
  }), false);
});

test("glass uses one fixed live arena and survives source replacement", () => {
  const previousUsage = globalThis.GPUBufferUsage;
  const previousTextureUsage = globalThis.GPUTextureUsage;
  Object.assign(globalThis, {
    GPUBufferUsage: { UNIFORM: 1, COPY_DST: 2, STORAGE: 4 },
    GPUTextureUsage: { TEXTURE_BINDING: 1, RENDER_ATTACHMENT: 2 },
  });
  const created: Array<{ label?: string; destroyed: boolean }> = [];
  const device = {
    createBuffer(descriptor: { label?: string }) {
      const buffer = { label: descriptor.label, destroyed: false, destroy() { buffer.destroyed = true; } };
      created.push(buffer);
      return buffer;
    },
    createTexture() {
      return { createView() { return {}; }, destroy() {} };
    },
    createSampler() { return {}; },
    queue: { writeBuffer() {} },
  } as unknown as GPUDevice;
  try {
    const renderer = new SparseVoxelDrySceneRenderer(device, {} as GPUBuffer, {} as GPUBuffer, "rgba16float", "canonical");
    const scene: SparseVoxelDrySceneData = {
      ...svoDrySceneFixture, ownerBase: 1,
      glassRecords: new Uint32Array(SVO_THIN_GLASS_RECORD_WORDS), glassCacheKey: "glass:v1",
    };
    renderer.setSource(structuralSource());
    renderer.publishScene(scene);
    const firstGlass = created.find(({ label }) => label === "Live authored scene arena (materials, primitives/BVH, thin glass, terrain)");
    assert.ok(firstGlass);
    renderer.setSource(structuralSource());
    renderer.publishScene(scene);
    assert.equal(created.filter(({ label }) => label === "Live authored scene arena (materials, primitives/BVH, thin glass, terrain)").length, 1,
      "scene updates must retain the fixed-capacity glass arena");
    assert.equal(firstGlass.destroyed, false);
    renderer.setSource(undefined);
    assert.equal(firstGlass.destroyed, false, "source replacement must not churn renderer-owned scene arenas");
    renderer.destroy();
    assert.equal(firstGlass.destroyed, true, "renderer destruction retires the live scene arena");
  } finally {
    Object.assign(globalThis, { GPUBufferUsage: previousUsage, GPUTextureUsage: previousTextureUsage });
  }
});

test("primary pane optics are exact, two-sided, identity preserving, and one-query bounded", () => {
  assert.match(svoDrySceneShader, /fn traceGlass\([^]*svoThinGlassIntersect/);
  assert.match(svoDrySceneShader, /let compositeOwned=skipCompositeOwned&&dry\.terrain\.w>0u&&paneId>=dry\.terrain\.z&&paneId-dry\.terrain\.z<dry\.terrain\.w/);
  assert.match(svoDrySceneShader, /fn shadeThinGlass\([^]*svoThinGlassOptics\(record,glass\.hit,incidentIor\)/);
  assert.match(svoDrySceneShader, /reflected\*optics\.fresnel\+transmitted\*optics\.netTransmittance/);
  assert.match(svoDrySceneShader, /svoThinGlassMaterialId\(record\),svoThinGlassOwnerId\(record\),svoThinGlassPaneId\(record\)/);
  assert.match(svoDrySceneShader, /fn dryThinGlassIncidentIor\(\)->f32\{return 1\.0;\}/,
    "the dry pass must leave fluid-interface optics to the raster water pass");

  const opticsStart = svoDrySceneShader.indexOf("fn shadeThinGlass(");
  const opticsEnd = svoDrySceneShader.indexOf("fn dryThickGlassEmission", opticsStart);
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

test("authored pane shadows use bounded transmission while compositor-owned tank panes are skipped", () => {
  assert.match(svoDrySceneShader, /traceGlass\(ray\.origin_m,ray\.direction,tMin_m,bestT,true\)/,
    "exact visibility must skip the vessel panes already owned by the water compositor");
  assert.match(svoDrySceneShader, /dryVisibilityTransmissionStep\([^]*glassTransmission/);
  assert.match(svoDrySceneShader, /SvoVisibilityBudget\(dry\.tuningCounts1\.w,dry\.tuningCounts2\.x,dry\.tuningCounts2\.y,dry\.tuningCounts2\.z\),true/,
    "pane transmission must share the published bounded visibility tuning");
});
