import assert from "node:assert/strict";
import test from "node:test";
import { SVO_MESHING_PLUGINS } from "../lib/svo/features/meshing/plugins";
import { primaryTuningQuery } from "../lib/svo/features/primary-visibility/persistence";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning, resolveSvoSurfaceTuning } from "../lib/svo/pipeline/svo-render-tuning";

test("meshing selections round-trip independently of legacy contour flags", () => {
  for (const plugin of SVO_MESHING_PLUGINS) {
    const tuning = normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING, surfaceMeshing: plugin.id});
    const query = new URLSearchParams();
    primaryTuningQuery.write(query, tuning);
    assert.equal(primaryTuningQuery.read(query).surfaceMeshing, plugin.id);
    assert.equal(tuning.surfaceMeshContours, plugin.id === "contours");
  }
  assert.equal(primaryTuningQuery.read(new URLSearchParams("svoMeshContours=1")).surfaceMeshing,"contours");
  assert.equal(primaryTuningQuery.read(new URLSearchParams("svoMeshContours=1&svoMesher=dual-contouring")).surfaceMeshing,"dual-contouring");
  assert.equal(primaryTuningQuery.read(new URLSearchParams("svoMesher=unknown")).surfaceMeshing,"voxels");
});

test("changing mesher does not reset the simulation allocation", async () => {
  await import("../lib/methods");
  const { gpuSceneSolverKey } = await import("../lib/core/webgpu-renderer");
  const { defaultScene } = await import("../lib/core/model");
  const config = {methodId:"uniform-volume",quality:"balanced" as const,values:{}};
  assert.equal(gpuSceneSolverKey(defaultScene,config),gpuSceneSolverKey({...defaultScene,surfaceStyle:"smooth"},config));
  for (const key of ["svoMeshDualContouring", "svoMeshDualMarchingCubes"]) {
    assert.equal(gpuSceneSolverKey(defaultScene,config),gpuSceneSolverKey(defaultScene,{...config,values:{[key]:true}}));
  }
});

test("a pending producer replacement retains the published meshing mode", async () => {
  const { SparseVoxelDrySceneRenderer } = await import("../lib/svo/pipeline/webgpu-svo-dry-scene");
  // Exercise the production setter without allocating a GPU: only the final
  // uniform upload is stubbed. A pending request must not change extraction.
  const renderer = Object.create(SparseVoxelDrySceneRenderer.prototype);
  renderer.lightingOptions = { coneTracingMode: "cones" };
  renderer.writeBandParams = () => {};
  for (const previous of ["voxels", "contours", "dual-contouring"] as const) {
    renderer.renderTuning = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshing: previous, surfaceMeshContourInflation: .25 });
    const before = renderer.renderTuning;
    const requested = normalizeSvoRenderTuning({ ...before, surfaceMeshing: "dual-marching-cubes" });
    renderer.setRenderTuning(requested, true);
    assert.equal(renderer.renderTuning.surfaceMeshing, previous);
    assert.equal(renderer.renderTuning.surfaceMeshContours, before.surfaceMeshContours);
    assert.equal(renderer.renderTuning.surfaceMeshContourInflation, before.surfaceMeshContourInflation);
    renderer.setRenderTuning(requested, false);
    assert.equal(renderer.renderTuning.surfaceMeshing, "dual-marching-cubes");
  }
});

test("smooth raster selects watertight native geometry without changing the saved mesher", () => {
  for (const plugin of SVO_MESHING_PLUGINS) {
    const saved = normalizeSvoRenderTuning({ ...DEFAULT_SVO_RENDER_TUNING, surfaceMeshing: plugin.id,
      surfaceMeshFilteringEnabled: true, surfaceMeshMaxCoarsening: 3 });
    const smooth = resolveSvoSurfaceTuning(saved, true);
    assert.equal(smooth.surfaceMeshing, "dual-marching-cubes");
    assert.equal(smooth.surfaceMeshContours, false);
    assert.equal(smooth.surfaceMeshFilteringEnabled, false);
    assert.equal(smooth.surfaceMeshMaxCoarsening, 0);
    assert.deepEqual(resolveSvoSurfaceTuning(saved, false), saved);
  }
});
