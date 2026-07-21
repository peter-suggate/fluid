import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  WebGPUSvoFinePhiResources,
  resolveSvoFinePhiOwnerDomain,
  type SvoFinePhiResourceStages,
} from "../lib/webgpu-svo-fine-phi-resources";
import type { SvoFineFluidGpuCapability } from "../lib/webgpu-svo-fine-phi-stager";
import type { SparseSurfaceBandGPUSource } from "../lib/webgpu-sparse-surface-band";
import type { SparseVoxelStructuralRenderSource } from "../lib/webgpu-voxel-debug";
import { getScenePreset } from "../lib/scenes";
import { createTallCellLayout } from "../lib/tall-cell-grid";
import { planSparseSceneDomain } from "../lib/sparse-scene-domain";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";

const fakeBuffer = (label: string) => ({ label, size: 4096, destroy() {} } as unknown as GPUBuffer);

function source(revision = 0, overrides: Partial<SparseSurfaceBandGPUSource> = {}): SparseSurfaceBandGPUSource {
  return {
    mode: "authoritative",
    pageTable: { buffer: fakeBuffer("page table") },
    states: { buffer: fakeBuffer("states") },
    activePages: { buffer: fakeBuffer("active pages") },
    phi: { buffer: fakeBuffer("phi") },
    velocity: { buffer: fakeBuffer("velocity") },
    params: { buffer: fakeBuffer("params") },
    control: { buffer: fakeBuffer("control") },
    coarseLevelSet: {} as GPUTexture,
    coarseVelocity: {} as GPUTexture,
    fineDimensions: [32, 16, 16],
    brickDimensions: [4, 2, 2],
    brickSize: 8,
    refinementFactor: 2,
    pageCapacity: 16,
    revision,
    ...overrides,
  };
}

function stages(calls: string[], capabilityOverrides: Partial<SvoFineFluidGpuCapability> = {}): SvoFinePhiResourceStages {
  const capability = {
    arena: { buffer: fakeBuffer("fine arena") },
    params: { buffer: fakeBuffer("fine params") },
    statusWord: 16,
    acceptedStructuralGenerationWord: 19,
    acceptedFineGenerationWord: 20,
    pageGenerationOffsetWords: 64,
    ownerPageTableOffsetWords: 66,
    payloadOffsetWords: 128,
    paramsWords: new Uint32Array(40),
    publicationMirrorWords: 8,
    coarseFallbackRequired: true,
    directWaterOwnership: false,
    ...capabilityOverrides,
  } as SvoFineFluidGpuCapability;
  return {
    residency: { allocatedBytes: 10, encode: () => calls.push("residency"), destroy: () => calls.push("destroy residency") },
    ownerPages: { allocatedBytes: 20, encode: () => calls.push("owner pages"), destroy: () => calls.push("destroy owner pages") },
    finePhi: {
      allocatedBytes: 30,
      mirrorPublication: () => calls.push("mirror publication"),
      encode: (_encoder, generation) => calls.push(`fine phi ${generation}`),
      capability: () => capability,
      destroy: () => calls.push("destroy fine phi"),
    },
  };
}

function structuralDomain(
  sceneDimensionsCells: readonly [number, number, number],
  originBricks: readonly [number, number, number],
  dimensionsBricks: readonly [number, number, number],
): SparseVoxelStructuralRenderSource {
  return {
    domain: {
      worldOrigin_m: [0, 0, 0], cellSize_m: [0.02, 0.02, 0.02],
      dimensionsCells: sceneDimensionsCells, brickSize: 8, maximumDepth: 6,
    },
    fluidResidency: { domain: { originBricks, dimensionsBricks } },
  } as unknown as SparseVoxelStructuralRenderSource;
}

test("dam-break fine owner domain stays solver-local inside the padded scene SVO", () => {
  const scene = getScenePreset("dam-break-boxes").create();
  const layout = createTallCellLayout(scene, "balanced", 2048);
  const primitives = environmentProxyPrimitives(buildEnvironmentProxyCatalog(scene, scene.environment ?? "default"), true);
  const sceneDomain = planSparseSceneDomain(
    scene, [layout.nx, layout.fineNy, layout.nz], 8,
    primitives.map((primitive) => ({ min: primitive.aabb_m.min, max: primitive.aabb_m.max })),
    { conservativePaddingCells: 1 },
  );
  assert.deepEqual([layout.nx, layout.fineNy, layout.nz], [60, 45, 40]);
  assert.deepEqual(sceneDomain.solverGridOriginCells, [144, 8, 120]);
  const resolved = resolveSvoFinePhiOwnerDomain(structuralDomain(
    sceneDomain.sceneDimensionsCells,
    sceneDomain.solverGridOriginCells.map((value) => value / 8) as [number, number, number],
    [Math.ceil(layout.nx / 8), Math.ceil(layout.fineNy / 8), Math.ceil(layout.nz / 8)],
  ), { fineDimensions: [layout.nx * 2, layout.fineNy * 2, layout.nz * 2], refinementFactor: 2 });
  assert.deepEqual(resolved, {
    ownerDimensionsCells: sceneDomain.sceneDimensionsCells,
    ownerDimensionsBricks: sceneDomain.sceneDimensionsCells.map((value) => Math.ceil(value / 8)),
    sourceDimensionsCells: [60, 45, 40], sourceDimensionsBricks: [8, 6, 5],
    sourceOriginCells: [144, 8, 120], refinementFactor: 2,
  });
});

test("fine owner domain rejects malformed refinement, residency, and structural bounds", () => {
  const valid = structuralDomain([352, 232, 296], [18, 1, 16], [8, 6, 6]);
  assert.throws(() => resolveSvoFinePhiOwnerDomain(valid, {
    fineDimensions: [121, 92, 82], refinementFactor: 2,
  }), /positive multiple/);
  assert.throws(() => resolveSvoFinePhiOwnerDomain(
    structuralDomain([352, 232, 296], [18, 1, 16], [9, 6, 6]),
    { fineDimensions: [122, 92, 82], refinementFactor: 2 },
  ), /residency brick domain/);
  assert.throws(() => resolveSvoFinePhiOwnerDomain(
    structuralDomain([160, 64, 160], [18, 1, 16], [8, 6, 6]),
    { fineDimensions: [122, 92, 82], refinementFactor: 2 },
  ), /exceeds the padded structural scene domain/);
});

test("resource owner encodes residency, owner pages, then fine staging once per generation", () => {
  const calls: string[] = [];
  const initial = source();
  const owner = new WebGPUSvoFinePhiResources(stages(calls), initial);
  const encoder = {} as GPUCommandEncoder;
  assert.equal(owner.allocatedBytes, 60);
  assert.equal(owner.encode(encoder, { ...initial, revision: 0 }), "unpublished");
  assert.deepEqual(calls, ["mirror publication"]);
  assert.equal(owner.encode(encoder, { ...initial, revision: 7 }), "encoded");
  assert.deepEqual(calls, ["mirror publication", "mirror publication", "residency", "owner pages", "fine phi 7"]);
  assert.equal(owner.encode(encoder, { ...initial, revision: 7 }), "unchanged");
  assert.equal(owner.encode(encoder, { ...initial, revision: 8 }), "encoded");
  assert.deepEqual(calls.slice(5), ["mirror publication", "mirror publication", "residency", "owner pages", "fine phi 8"]);
});

test("source replacement and teardown fail closed without publishing stale bindings", () => {
  const calls: string[] = [];
  const initial = source(3);
  const owner = new WebGPUSvoFinePhiResources(stages(calls), initial);
  assert.equal(owner.encode({} as GPUCommandEncoder, { ...initial, phi: { buffer: fakeBuffer("replacement phi") } }), "source-changed");
  assert.deepEqual(calls, []);
  assert.equal(owner.capability()?.coarseFallbackRequired, true);
  assert.equal(owner.capability()?.directWaterOwnership, false);
  owner.destroy();
  owner.destroy();
  assert.deepEqual(calls, ["destroy fine phi", "destroy owner pages", "destroy residency"]);
  assert.equal(owner.capability(), undefined);
  assert.equal(owner.encode({} as GPUCommandEncoder, initial), "destroyed");
});

test("capability validation forbids direct water ownership or loss of coarse fallback", () => {
  const initial = source(1);
  assert.throws(() => new WebGPUSvoFinePhiResources(stages([], {
    directWaterOwnership: true as false,
  }), initial).capability(), /legacy water ownership/);
  assert.throws(() => new WebGPUSvoFinePhiResources(stages([], {
    coarseFallbackRequired: false as true,
  }), initial).capability(), /coarse fallback/);
});

test("production renderer attaches, orders, detaches, and device-loss-cleans the optional chain", () => {
  const renderer = readFileSync(new URL("../lib/webgpu-renderer.ts", import.meta.url), "utf8");
  const resources = readFileSync(new URL("../lib/webgpu-svo-fine-phi-resources.ts", import.meta.url), "utf8");
  assert.match(renderer, /createWebgpuSvoFinePhiResources\(\s*this\.device!, solver\.sparseVoxelSceneSource, solver\.sparseSurfaceBand/);
  assert.match(renderer, /const fineStatus = this\.svoFinePhiResources\.encode\(encoder, fineSource\)/);
  assert.match(resources, /this\.stages\.residency\.encode\(encoder\);\s*this\.stages\.ownerPages\.encode\(encoder\);\s*this\.stages\.finePhi\.encode\(encoder, generation\)/,
    "generation publication order must be consumer -> owner pages -> fine stager");

  const attach = renderer.indexOf("this.attachSvoFinePhiResources(solver)");
  const presentationRebind = renderer.lastIndexOf("this.updateRenderSources(", attach);
  const retirePrevious = renderer.indexOf("this.retireGPUFluid(previous)", attach);
  assert.ok(presentationRebind >= 0 && presentationRebind < attach && attach < retirePrevious,
    "new presentation and fine bindings must replace old bindings before solver retirement");
  const deviceLoss = renderer.indexOf("void device.lost.then");
  const lostDetach = renderer.indexOf("this.detachSvoFinePhiResources()", deviceLoss);
  const lostFluidDestroy = renderer.indexOf("fluid?.destroy()", deviceLoss);
  assert.ok(deviceLoss >= 0 && lostDetach > deviceLoss && lostFluidDestroy > lostDetach);
  assert.match(renderer, /this\.gpuFluidGeneration \+= 1;\s*this\.detachSvoFinePhiResources\(\);\s*try \{ fluid\?\.destroy\(\)/,
    "renderer teardown must invalidate the future capability before solver resources");
  assert.doesNotMatch(resources, /webgpu-svo-dry-scene|RasterWaterPipeline|direct-structural-media/);
});
