import assert from "node:assert/strict";
import test from "node:test";
import { SVO_MESHING_PLUGINS } from "../lib/svo/features/meshing/plugins";
import { primaryTuningQuery } from "../lib/svo/features/primary-visibility/persistence";
import { DEFAULT_SVO_RENDER_TUNING, normalizeSvoRenderTuning } from "../lib/svo/pipeline/svo-render-tuning";

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
  const config = {methodId:"sparse-cm12",quality:"balanced" as const,values:{}};
  for (const key of ["svoMeshDualContouring", "svoMeshDualMarchingCubes"]) {
    assert.equal(gpuSceneSolverKey(defaultScene,config),gpuSceneSolverKey(defaultScene,{...config,values:{[key]:true}}));
  }
});
