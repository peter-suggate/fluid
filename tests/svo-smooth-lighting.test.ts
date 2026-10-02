import assert from "node:assert/strict";
import test from "node:test";
import { defaultScene } from "../lib/core/model";
import { PLANAR_BOUNDARY_PATCH_BYTES } from "../lib/core/planar-boundary";
import { SVO_MATERIAL_RECORD_WORDS } from "../lib/svo/contracts/svo-material-abi";
import { buildSvoPrimitiveCandidates } from "../lib/svo/features/scene-publication/svo-primitive-candidates";
import { RENDER_PIPELINE_NODES, type RenderPipelineContext } from "../lib/svo/pipeline/render-pipeline-graph";
import { DEFAULT_SVO_RENDER_TUNING } from "../lib/svo/pipeline/svo-render-tuning";
import { buildSparseVoxelDrySceneLightingMirrors, SparseVoxelDrySceneRenderer } from "../lib/svo/pipeline/webgpu-svo-dry-scene";

function rendererFixture() {
  // Exercise production selection/publication without allocating GPU resources.
  const renderer = Object.assign(Object.create(SparseVoxelDrySceneRenderer.prototype), {
    shadingPath: "split", coneFanout: true, experiments: { voxelLightCache: false },
    device: { limits: {}, queue: { writeBuffer() {} } },
    lightingOptions: { coneTracingMode: "cones" },
    renderTuning: { ...DEFAULT_SVO_RENDER_TUNING, coneRadianceReconstruction: "full-res-relight" },
    coneScale: .5, conePipelineScale: .5, splitPipelineScale: .5,
    splitPipelineLattice: false, splitPipelineBackdropTerrain: false, splitPipelineRigidBodies: false,
    rigidBodyCountPublished: true, rigidBodyCount: 0,
    source: { structural: {
      fields: { topology: {}, sceneGeometry: {}, materialOwner: {} },
      planarBoundaries: { count: 0, records: { size: 0 }, strideBytes: PLANAR_BOUNDARY_PATCH_BYTES, generation: 1 },
    } },
    pickingFrameToken: 0, latticeGeneration: 0, bindGroup: {},
    invalidateVoxelLightCache() {}, writeScenePrimitiveOverflowPublication() {},
    ensureVoxelLightCache() {}, writeParams() {},
  });
  return renderer;
}

test("smooth surfaces select screen lighting while voxel faces retain the lattice", () => {
  const renderer = rendererFixture();
  for (const scale of [.5, .25, .125]) {
    renderer.scene = { flatVoxelNormals: true };
    assert.equal(renderer.latticeVisibilityRequested(scale, false), true);
    renderer.scene = { flatVoxelNormals: false };
    assert.equal(renderer.latticeVisibilityRequested(scale, false), false);
  }
  renderer.scene = { flatVoxelNormals: true };
  assert.equal(renderer.latticeVisibilityRequested(1, false), false);
  assert.equal(renderer.latticeVisibilityRequested(.5, true), false);
  renderer.lightingOptions.latticeVisibilityEnabled = false;
  assert.equal(renderer.latticeVisibilityRequested(.5, false), false);
});

test("publishing a surface-style change requests a new bundle even with the same backdrop", () => {
  const renderer = rendererFixture();
  const scene = {
    renderRevision: 1, materialRevision: 1,
    primitiveRecords: new Uint32Array(), primitiveCandidates: buildSvoPrimitiveCandidates([]),
    materialRecords: new Uint32Array(2 * SVO_MATERIAL_RECORD_WORDS),
    ...buildSparseVoxelDrySceneLightingMirrors(defaultScene, 1),
  };
  const requested: boolean[] = [];
  renderer.requestSpecializedSplitVariant = () => requested.push(renderer.latticeVisibilityRequested(.5, false));
  for (const flatVoxelNormals of [true, true, false, false, true]) {
    assert.equal(renderer.publishScene({ ...scene, flatVoxelNormals }), true);
  }
  assert.deepEqual(requested, [true, false, true]);
});

test("an old lattice bundle cannot render a newly smooth scene while screen shaders compile", () => {
  const renderer = rendererFixture();
  renderer.scene = { flatVoxelNormals: false };
  renderer.splitPipelineLattice = true;
  renderer.latticeStoreFailure = "old lattice overflow";
  assert.equal(renderer.presentationBundleStatus.state, "compiling");
  renderer.splitPipelineLattice = false;
  assert.equal(renderer.presentationBundleStatus.state, "ready");
  renderer.scene = { flatVoxelNormals: true };
  assert.equal(renderer.presentationBundleStatus.state, "compiling");
  renderer.splitPipelineLattice = true;
  assert.equal(renderer.presentationBundleStatus.state, "failed", "Real lattice faults still fail closed");
});

test("the lighting panel reports screen sampling for smooth surfaces", () => {
  const node = RENDER_PIPELINE_NODES.find(node => node.id === "cone-visibility")!;
  const context = { coneTracingMode: "cones", tuning: { ...DEFAULT_SVO_RENDER_TUNING,
    coneLightingScale: .5, coneRadianceReconstruction: "full-res-relight" } } as RenderPipelineContext;
  assert.match(node.chip({ ...context, smoothSurfaceEnabled: false }), /face lattice/);
  assert.doesNotMatch(node.chip({ ...context, smoothSurfaceEnabled: true }), /face lattice/);
});

test("smooth raster geometry disables independent brick coarsening", () => {
  const detail = RENDER_PIPELINE_NODES.find(node => node.id === "filtered-detail")!;
  const context = { surfaceMeshSelected: true, surfaceMeshActive: true, smoothSurfaceEnabled: true,
    disabledStages: new Set(), tuning: { ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshFilteringEnabled: true } } as unknown as RenderPipelineContext;
  assert.equal(detail.state(context), "unavailable");
  assert.match(detail.chip(context), /native resolution/);
});
