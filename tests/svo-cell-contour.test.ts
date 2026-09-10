import assert from "node:assert/strict";
import test from "node:test";
import { clipCellContourPolygon, encodeCellContourSupport } from "../lib/svo/features/construction/svo-cell-contour";
import { primaryTuningQuery } from "../lib/svo/features/primary-visibility/persistence";

test("contour offset is outward, reserves rounding slack, and full cubes are absent", () => {
  for (let i = 0; i <= 1000; i++) {
    const support = -3 + 6 * i / 1000;
    const q = encodeCellContourSupport(support, 3);
    const decoded = q === 0 ? 3 : (q * 2 / 255 - 1) * 3;
    assert.ok(decoded >= support - 1e-12);
    if (q) assert.ok(decoded - support >= 6 / 255 - 1e-12);
  }
  assert.equal(encodeCellContourSupport(NaN, 1), 0);
  assert.equal(encodeCellContourSupport(0, 0), 0);
});

test("polygon clipping creates the cap edge and preserves cube-boundary coordinates", () => {
  assert.deepEqual(clipCellContourPolygon([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], [1,0,0], 0),
    [[0,0,0], [.5,0,0], [.5,1,0], [0,1,0]]);
  assert.deepEqual(clipCellContourPolygon([[0,0,0], [1,0,0], [1,1,0], [0,1,0]], [0,0,-1], 0), []);
});

test("contour selection survives URL round trips and is opt in", () => {
  assert.equal(primaryTuningQuery.read(new URLSearchParams()).surfaceMeshContours, false);
  const selection = primaryTuningQuery.read(new URLSearchParams("svoMeshContours=1"));
  assert.equal(selection.surfaceMeshContours, true);
  const url = new URLSearchParams(); primaryTuningQuery.write(url, selection);
  assert.equal(primaryTuningQuery.read(url).surfaceMeshContours, true);
});

test("contour selection does not change the fluid solver allocation key", async () => {
  await import("../lib/methods");
  const { gpuSceneSolverKey } = await import("../lib/core/webgpu-renderer");
  const { defaultScene } = await import("../lib/core/model");
  const config = { methodId: "sparse-cm12", quality: "balanced" as const, values: {} };
  assert.equal(gpuSceneSolverKey(defaultScene, config),
    gpuSceneSolverKey(defaultScene, { ...config, values: { svoMeshContours: true } }));
});


test("contour inflation persists and is clamped to half a cell per side", async () => {
  const { normalizeSvoRenderTuning, DEFAULT_SVO_RENDER_TUNING } = await import("../lib/svo/pipeline/svo-render-tuning");
  assert.equal(primaryTuningQuery.read(new URLSearchParams()).surfaceMeshContourInflation,0);
  const tuning=primaryTuningQuery.read(new URLSearchParams("svoMeshContourInflation=0.25"));
  const url=new URLSearchParams();primaryTuningQuery.write(url,tuning);
  assert.equal(primaryTuningQuery.read(url).surfaceMeshContourInflation,.25);
  assert.equal(normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING,surfaceMeshContourInflation:4}).surfaceMeshContourInflation,.5);
  assert.equal(normalizeSvoRenderTuning({...DEFAULT_SVO_RENDER_TUNING,surfaceMeshContourInflation:-1}).surfaceMeshContourInflation,0);
});
