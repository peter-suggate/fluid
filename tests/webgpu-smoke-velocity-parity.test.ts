import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { getSceneWebGPUSmokeLane } from "../lib/scene-webgpu-smoke-catalog";
import { compareVelocityFields, rasterizeStructuredCellVelocities,
  velocityParityFailures } from "../tools/webgpu-smoke-velocity-parity";

test("compact structured velocities rasterize over coarse and fine cubic ownership", () => {
  const headers = new Uint32Array(24);
  headers[0] = 0; headers[3] = 2;
  headers[12] = 2; headers[15] = 2;
  const velocities = new Float32Array([
    1, 2, 3, 1,
    -1, -2, -3, 1,
  ]);
  const raster = rasterizeStructuredCellVelocities(headers, velocities, 2, [4, 2, 2]);
  assert.equal(raster.coveredCells, 16);
  assert.equal(raster.overlapCells, 0);
  assert.equal(raster.invalidRows, 0);
  assert.deepEqual(Array.from(raster.field.slice(0, 12)), [
    1, 2, 3, 1, 2, 3, -1, -2, -3, -1, -2, -3,
  ]);
});

test("compact velocity raster fails closed on overlaps and unsolved rows", () => {
  const headers = new Uint32Array(36);
  headers[3] = 1;
  headers[15] = 1;
  headers[24] = 1; headers[27] = 1;
  const velocities = new Float32Array([
    1, 0, 0, 1,
    2, 0, 0, 1,
    3, 0, 0, 0,
  ]);
  const raster = rasterizeStructuredCellVelocities(headers, velocities, 3, [2, 1, 1]);
  assert.equal(raster.coveredCells, 2);
  assert.equal(raster.overlapCells, 1);
  assert.equal(raster.invalidRows, 1);
  assert.ok(Number.isNaN(raster.field[0]));
  assert.ok(Number.isNaN(raster.field[3]));
});

test("the 2.2 second octree/tall-cell command gates final vector-field parity", () => {
  const packageJson = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")) as {
    scripts: Record<string, string>;
  };
  const command = packageJson.scripts["test:webgpu:dam-octree-parity"];
  assert.match(command, /FLUID_METHOD=octree,tall-cell/);
  assert.match(command, /FLUID_TARGET_S=2\.2/);
  const smoke = readFileSync(new URL("../tools/webgpu-smoke-executor.ts", import.meta.url), "utf8");
  const readbacks = readFileSync(new URL("../tools/webgpu-smoke-readbacks.ts", import.meta.url), "utf8");
  const diagnostic = readFileSync(new URL("../lib/scene-field-velocity-parity-diagnostic.ts", import.meta.url), "utf8");
  assert.doesNotMatch(smoke, /phase: "velocity-parity"/,
    "custom parity reporting belongs to the scene hook evaluator");
  assert.match(diagnostic, /compareVelocityFields/);
  assert.match(smoke, /readCompactOctreeVelocityField3D/);
  assert.match(readbacks, /structuredRowVelocities/);
  assert.deepEqual(getSceneWebGPUSmokeLane("dam-break-ui").hooks
    .find(({ id }) => id === "dam-break-velocity-parity")?.parameters?.limits, {
    maximumWeightedRelativeL2: 1,
    minimumCosineSimilarity: 0.5,
    minimumEnergyRatio: 0.25,
    maximumEnergyRatio: 4,
    minimumPeakRatio: 0.5,
    maximumPeakRatio: 2,
  });
});

test("matched liquid vector fields pass the tall-cell parity gate", () => {
  const reference = new Float32Array([1, 0, 0, 0, 2, 0]);
  const candidate = new Float32Array([0.9, 0.1, 0, 0, 1.8, 0.1]);
  const metrics = compareVelocityFields(candidate, reference, [1, 0.5], [1, 1]);
  assert.equal(metrics.comparedCells, 2);
  assert.deepEqual(velocityParityFailures(metrics), []);
});

test("wrong-direction or dissipated waves cannot pass on peak speed alone", () => {
  const reference = new Float32Array([1, 0, 0, 0, 1, 0]);
  const candidate = new Float32Array([-1, 0, 0, 0, 0.1, 0]);
  const metrics = compareVelocityFields(candidate, reference, [1, 1], [1, 1]);
  assert.equal(metrics.candidateToReferencePeakRatio, 1);
  assert.ok(velocityParityFailures(metrics).includes("velocity direction cosine"));
  assert.ok(velocityParityFailures(metrics).includes("velocity relative L2"));
});

test("air-only samples and non-finite vectors fail closed", () => {
  const empty = compareVelocityFields([100, 0, 0], [1, 0, 0], [0], [1]);
  assert.deepEqual(velocityParityFailures(empty), [
    "no shared liquid velocity cells", "velocity direction cosine",
    "velocity energy ratio", "velocity peak ratio",
  ]);
  const invalid = compareVelocityFields([NaN, 0, 0], [1, 0, 0], [1], [1]);
  assert.ok(velocityParityFailures(invalid).length > 0);
});
