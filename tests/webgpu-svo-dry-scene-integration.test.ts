import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

import { DEFAULT_SVO_RENDER_MODE } from "../lib/svo-render-mode";
import { canEncodeSparseVoxelDryScene, type SparseVoxelDrySceneData } from "../lib/webgpu-svo-dry-scene";
import type { SparseVoxelRenderSource } from "../lib/webgpu-voxel-debug";
import type { DrySceneReplacementEncoder } from "../lib/webgpu-water-pipeline";

const rendererUrl = new URL("../lib/webgpu-renderer.ts", import.meta.url);
const waterUrl = new URL("../lib/webgpu-water-pipeline.ts", import.meta.url);
const drySceneUrl = new URL("../lib/webgpu-svo-dry-scene.ts", import.meta.url);
const rendererSource = readFileSync(rendererUrl, "utf8");
const waterSource = readFileSync(waterUrl, "utf8");
const drySceneSource = existsSync(drySceneUrl) ? readFileSync(drySceneUrl, "utf8") : "";

function expectSource(source: string, pattern: RegExp, message: string): void {
  assert.ok(pattern.test(source), message);
}

test("raster remains the default production dry-scene renderer", () => {
  assert.equal(DEFAULT_SVO_RENDER_MODE, "svo");
  expectSource(rendererSource, /svoRenderMode: SvoRenderMode = DEFAULT_SVO_RENDER_MODE/,
    "callers which do not opt in must retain the current raster presentation");
  expectSource(rendererSource, /import \{ DEFAULT_SVO_RENDER_MODE, type SvoRenderMode \} from "\.\/svo-render-mode"/,
    "renderer must consume the canonical SvoRenderMode toggle");
});

test("the water pipeline replacement callback is fail-safe and replaces rather than overlays raster", () => {
  const replacement: DrySceneReplacementEncoder = () => true;
  assert.equal(replacement({} as GPUCommandEncoder, {} as GPUTexture), true);
  expectSource(waterSource, /drySceneReplacement\?\.\(encoder, this\.sceneTexture, timestamps\?\.scene\) \?\? false/,
    "water pipeline must let the replacement explicitly accept or reject a frame");
  expectSource(waterSource, /if \(!sparseSceneEncoded\) \{[^]*label:"Dry scene"/,
    "the analytic pass is encoded only when no replacement accepted the frame");
  assert.doesNotMatch(waterSource, /drySceneOverlay/,
    "a sparse production scene must never be composed after the analytic pass");
  const replacementCall = waterSource.indexOf("const sparseSceneEncoded = drySceneReplacement");
  const rasterFallback = waterSource.indexOf("if (!sparseSceneEncoded)", replacementCall);
  assert.ok(replacementCall >= 0 && rasterFallback > replacementCall);
});

test("the direct renderer exposes a boolean source-aware replacement contract", () => {
  assert.ok(drySceneSource, "lib/webgpu-svo-dry-scene.ts must implement the production dry-scene renderer");
  expectSource(drySceneSource, /export class SparseVoxelDrySceneRenderer/,
    "direct renderer class must be public to the presentation owner");
  expectSource(drySceneSource, /setSource\(source: SparseVoxelSceneRenderSource \| undefined, scene: SparseVoxelDrySceneData \| undefined\)/,
    "the renderer accepts structural arenas, their parent material table, and analytic-owner data together");
  expectSource(drySceneSource, /encode\([^)]*encoder: GPUCommandEncoder[^)]*target: GPUTexture \| GPUTextureView[^)]*timestampWrites\?: TimestampRange[^)]*\): boolean/,
    "encode must match DrySceneReplacementEncoder and report whether it wrote the frame");
  expectSource(drySceneSource, /if \(!this\.pipeline \|\| !this\.bindGroup\) return false/,
    "an absent or unpublished source must trigger the raster fallback");
  expectSource(drySceneSource, /loadOp:\s*"clear"/,
    "a successful replacement owns the complete dry-scene target");
  assert.doesNotMatch(drySceneSource, /SparseVoxelDebugRecord|voxelRecords|brickRecords/,
    "production traversal must not expand or consume debug cube records");
  for (const binding of ["structural.control", "structural.nodes", "structural.leaves", "structural.materialOwners", "structural.publication.state", "source.pbrMaterials!.binding"]) {
    assert.ok(drySceneSource.includes(binding), `direct rendering must bind ${binding}`);
  }
});

test("unavailable structural fields fail over to raster before GPU encoding", () => {
  const source = {
    materialCount: 2,
    pbrMaterials: { binding: { buffer: {} as GPUBuffer }, count: 8, strideBytes: 96, revision: 1 },
    structural: {
      fields: {
        topology: { residency: "all-published-leaves" },
        staticGeometry: { residency: "all-published-leaves" },
        materialOwner: { residency: "all-published-leaves" },
      },
    },
  } as unknown as SparseVoxelRenderSource;
  const scene: SparseVoxelDrySceneData = { primitiveRecords: new Uint32Array(16), ownerBase: 32 };
  assert.equal(canEncodeSparseVoxelDryScene(undefined, scene), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, undefined), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, { ...scene, primitiveRecords: new Uint32Array(0) }), false);
  assert.equal(canEncodeSparseVoxelDryScene(source, scene), true);
  const unavailable = {
    ...source,
    structural: {
      ...source.structural!,
      fields: {
        ...source.structural!.fields,
        staticGeometry: { ...source.structural!.fields.staticGeometry, residency: "unavailable" as const },
      },
    },
  };
  assert.equal(canEncodeSparseVoxelDryScene(unavailable, scene), false);
});

test("renderer attaches and detaches structural sources across solver replacement", () => {
  expectSource(rendererSource, /private svoDryScenePipeline\?: SparseVoxelDrySceneRenderer/,
    "FluidLabRenderer must own the direct renderer lifecycle");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.setSource\(sparseSceneSource,drySceneData\)/,
    "solver attachment must pass the complete source and its analytic-owner data");

  const detach = rendererSource.indexOf("this.svoDryScenePipeline?.setSource(undefined, undefined)");
  const retire = rendererSource.indexOf("this.retireGPUFluid(previous)", detach);
  assert.ok(detach >= 0, "the old structural source must be detached");
  assert.ok(retire > detach, "detach must happen before the solver's GPU buffers are retired");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.destroy\(\)/,
    "renderer teardown must destroy direct-renderer-owned GPU resources");
});

test("SVO is offered to the water pipeline only for smooth production presentation", () => {
  expectSource(rendererSource, /svoRenderMode === "svo" && voxelRenderMode === "smooth"/,
    "inspection modes and the production renderer are separate switches");
  expectSource(rendererSource, /this\.svoDryScenePipeline\?\.encode\(/,
    "SVO mode must offer a replacement encoder");
  expectSource(rendererSource, /this\.waterPipeline\.encode\([^]*drySceneReplacement/s,
    "the replacement callback must target the water pipeline's internal HDR dry-scene attachment");
});

test("raw voxels and brick-grid remain independent clear-and-replace inspection modes", () => {
  assert.match(rendererSource, /this\.voxelInspectionSource = requestedVoxelDebugGeneration >= 0 \? this\.gpuFluid\?\.sparseVoxelRenderSource : undefined/,
    "debug mode materialization must remain gated by inspection visibility");
  assert.match(rendererSource, /this\.voxelDebugPipeline\?\.setSource\(this\.voxelInspectionSource\)/,
    "debug modes consume expanded records only while inspection is visible");
  assert.match(rendererSource, /if \(voxelRenderMode !== "smooth" && this\.voxelDebugDepth\)/);
  assert.match(rendererSource, /mode: voxelRenderMode/);
  assert.match(rendererSource, /colorLoadOp: "clear"/,
    "inspection remains a complete representation switch, not an SVO dry-scene overlay");
});
