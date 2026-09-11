import assert from "node:assert/strict";
import test from "node:test";
import { createSolidWorld } from "../lib/core/solid-world";
import { retainedSceneDensity } from "../lib/methods/adaptive-volume/sparse-cm12-retained-scene-density";
import { compileRetainedScenePreparationCache } from "../lib/methods/adaptive-volume/sparse-cm12-retained-preparation-cache";
import { createSparseAdaptiveMassAtlas } from "../lib/methods/adaptive-volume/sparse-brick-atlas";
import { WebGPUSparseCM12Resident } from "../lib/methods/adaptive-volume/webgpu-sparse-cm12-resident";

test("the CPU preparation factory rejects a mismatched frozen cache before allocating resources", async () => {
  const h = .125, dimensions = [8, 8, 8] as const;
  const field = retainedSceneDensity({ generation: 1, transitionWidth: h,
    domain: { lower: [-.5, 0, -.5], upper: [.5, 1, .5] },
    primitives: [{ kind: "quadratic-height", center: [0, h / 2, 0], curvature: [0, 0, 0] }] });
  const world = createSolidWorld(), initial = compileRetainedScenePreparationCache(field, dimensions, h, world);
  const atlas = createSparseAdaptiveMassAtlas(dimensions, [{ key: 0, coordinate: [0, 0, 0], resolution: 1,
    density: new Float64Array([.5]), gamma: new Float64Array([1]) }], 0, 8);
  const input = { atlas, active: new Set([0]), finestCellSize_m: h, solidWorld: world,
    maximumBytes: 64 * 1024 * 1024, topologyPageCapacityMaximum: 0,
    symmetry: { scalar: false, face: false }, limits: { maxComputeWorkgroupsPerDimension: 65535 } as GPUSupportedLimits,
    retainedDensity: field, retainedPreparationCache: { ...initial, unrestrictedMeans: new Float32Array(1) } };
  await assert.rejects(WebGPUSparseCM12Resident.recordPreparedGeneration(input), /numeric field and physical lattice/);
  assert.equal(initial.unrestrictedMeans[0], .5);
});
