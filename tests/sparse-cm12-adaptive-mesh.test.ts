import assert from "node:assert/strict";
import test from "node:test";
import { RasterWaterPipeline } from "../lib/core/webgpu-water-pipeline";
import type { GlobalFineLevelSetConsumerSource } from "../lib/core/octree-consumer-sampling";
import { resolveMethodValues } from "../lib/core/method-contract";
import { adaptiveMassMethod, adaptiveMassSolverOptions,
  ADAPTIVE_MASS_RUNTIME_PARAM_KEYS } from "../lib/methods/adaptive-mass/method";
import { SPARSE_CM12_STAGES } from "../lib/methods/adaptive-mass/sparse-cm12-stages";

test("surface mesh ratio defaults to x2 and x4 is available without resetting physics", () => {
  const defaults = resolveMethodValues(adaptiveMassMethod, "balanced", {});
  assert.equal(adaptiveMassSolverOptions(defaults).surfaceMeshRefinement, 2);
  const fine = resolveMethodValues(adaptiveMassMethod, "balanced", { surfaceMeshRefinement: "4" });
  assert.equal(adaptiveMassSolverOptions(fine).surfaceMeshRefinement, 4);
  assert.ok(ADAPTIVE_MASS_RUNTIME_PARAM_KEYS.includes("surfaceMeshRefinement"));
  assert.ok(SPARSE_CM12_STAGES["presentation-publication"].controls.some(control =>
    control.kind === "param-choice" && control.param === "surfaceMeshRefinement"));
});


test("changing the mesh ratio invalidates a retained mesh with unchanged GPU generation", () => {
  const binding = { buffer: {} as GPUBuffer };
  const source: GlobalFineLevelSetConsumerSource = {
    kind: "global-fine-levelset-sampling", metadata: binding, worklist: binding,
    samples: binding, sampleDimensions: [8, 8, 8], brickDimensions: [1, 1, 1],
    brickResolution: 8, samplesPerBrick: 512, pageCapacity: 1, fineFactor: 1,
    fineCellWidth: 1, domainOrigin: [0, 0, 0], generation: 7, surfaceMeshRefinement: 2,
  };
  let writes = 0, rebuilds = 0;
  // Exercise the production retained-mesh decision without allocating a GPU.
  const pipeline = Object.assign(Object.create(RasterWaterPipeline.prototype), {
    globalFineLevelSet: source, extractedRevision: 7, lastExtractionAt_ms: 100,
    causticsValid: true, writeCompactRenderParams: () => { writes++; },
    rebuildBindGroups: () => { rebuilds++; }, ensureGlobalCoarsePipeline: () => {},
  });
  pipeline.setGlobalFineLevelSet({ ...source });
  assert.equal(writes, 0);
  pipeline.setGlobalFineLevelSet({ ...source, surfaceMeshRefinement: 4 });
  assert.equal(writes, 1);
  assert.equal(rebuilds, 0);
  assert.equal(pipeline.extractedRevision, -1);
  assert.equal(pipeline.causticsValid, false);
});
