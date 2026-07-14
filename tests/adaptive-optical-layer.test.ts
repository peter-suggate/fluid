import assert from "node:assert/strict";
import test from "node:test";
import {
  adaptiveOpticalLayerSettings,
  constructAdaptiveOpticalLayer,
  createAdaptiveOpticalLayerLayout,
  errorToDilationCells,
  fitTallColumnVelocity,
  manhattanDilateSeedBases,
  smoothAdaptiveOpticalBases,
  splitTallCellRigidCoupling,
  summarizeAdaptiveLayer
} from "../lib/adaptive-optical-layer";
import { cloneScene, defaultScene } from "../lib/model";

test("paper parameters scale from the vertical resolution", () => {
  assert.deepEqual(adaptiveOpticalLayerSettings(256), {
    alpha: 0.5,
    minimumDilationCells: 4,
    maximumDilationCells: 32,
    airborneOffsetCells: 8,
    airborneDilationCells: 16,
    smoothingRadius: 4,
    smoothingIterations: 5,
    logicalRegularLayers: 65
  });
});

test("a tall-representable velocity field has zero fitting error", () => {
  const samples = Array.from({ length: 9 }, (_, j) => ({ x: 1 + 0.2 * j, y: -0.4, z: 2 - 0.1 * j }));
  const fit = fitTallColumnVelocity(samples);
  assert.ok(fit.error < 1e-12);
  for (let j = 0; j < samples.length; j += 1) {
    assert.ok(Math.abs(fit.reconstructed[j].x - samples[j].x) < 1e-12);
    assert.ok(Math.abs(fit.reconstructed[j].y - samples[j].y) < 1e-12);
    assert.ok(Math.abs(fit.reconstructed[j].z - samples[j].z) < 1e-12);
  }
});

test("nonlinear motion increases tall-cell fitting error", () => {
  const linear = Array.from({ length: 7 }, (_, j) => ({ x: j, y: 1, z: -j }));
  const disturbed = linear.map((value, j) => j === 3 ? { x: value.x + 4, y: value.y - 2, z: value.z + 1 } : value);
  assert.ok(fitTallColumnVelocity(disturbed).error > fitTallColumnVelocity(linear).error);
});

test("Equation 1 is monotone and respects both dilation clamps", () => {
  const settings = adaptiveOpticalLayerSettings(128);
  const values = [0, 20, 80, 1e9].map((error) => errorToDilationCells(error, 0.1, settings));
  assert.equal(values[0], settings.minimumDilationCells);
  assert.equal(values.at(-1), settings.maximumDilationCells);
  for (let index = 1; index < values.length; index += 1) assert.ok(values[index] >= values[index - 1]);
});

test("variable-radius Manhattan dilation produces the exact diamond profile", () => {
  const seeds = new Float32Array(25); seeds.fill(10); seeds[2 + 5 * 2] = 4;
  const radii = new Float32Array(25); radii[2 + 5 * 2] = 2;
  const bases = manhattanDilateSeedBases(seeds, radii, 5, 5, 10);
  const expected = [
    10, 10, 6, 10, 10,
    10, 6, 5, 6, 10,
    6, 5, 4, 5, 6,
    10, 6, 5, 6, 10,
    10, 10, 6, 10, 10
  ];
  assert.deepEqual([...bases], expected);
});

test("constrained smoothing only thickens the optical layer", () => {
  const raw = new Float32Array([
    8, 8, 8,
    8, 0, 8,
    8, 8, 8
  ]);
  const smoothed = smoothAdaptiveOpticalBases(raw, 3, 3, 1, 5);
  for (let index = 0; index < raw.length; index += 1) assert.ok(smoothed[index] <= raw[index]);
  assert.equal(smoothed[4], 0);
  assert.ok(smoothed[0] < 8);
  const constant = smoothAdaptiveOpticalBases(new Float32Array(16).fill(6), 4, 4, 1, 5);
  assert.ok(constant.every((value) => value === 6));
});

test("rigid tall-cell weights conserve every generalized-force component", () => {
  const contribution = [1, -2, 3, -4, 5, -6];
  const split = splitTallCellRigidCoupling(contribution, 0.25);
  for (let index = 0; index < contribution.length; index += 1) assert.ok(Math.abs(split.top[index] + split.bottom[index] - contribution[index]) < 1e-12);
  assert.deepEqual([...splitTallCellRigidCoupling(contribution, 0).top], contribution);
  assert.deepEqual([...splitTallCellRigidCoupling(contribution, 1).bottom], contribution);
});

test("adaptive layout is independent, representable, and counts active samples", () => {
  const scene = cloneScene(defaultScene);
  const layout = createAdaptiveOpticalLayerLayout(scene, "balanced");
  assert.equal(layout.packedNy, layout.fineNy + 2);
  assert.ok(layout.settings.regularLayers < layout.fineNy);
  assert.ok(layout.columnBases.some((base) => base === 0));
  assert.ok(layout.columnBases.some((base) => base > 0));
  const summary = summarizeAdaptiveLayer(layout.columnBases, layout.columnBases, layout.fineNy);
  assert.ok(summary.activePressureSamples <= layout.packedSampleCount);
  assert.ok(summary.opticalCellCount <= layout.equivalentUniformCellCount);
  assert.ok(summary.tallColumnCount > 0);
  let representedVolume = 0;
  for (let z = 0; z < layout.nz; z += 1) for (let x = 0; x < layout.nx; x += 1) {
    const base = layout.columnBases[x + layout.nx * z];
    representedVolume += layout.initialVolume[x + layout.nx * layout.packedNy * z] * base;
    for (let y = 2; y < layout.packedNy; y += 1) representedVolume += layout.initialVolume[x + layout.nx * (y + layout.packedNy * z)];
  }
  assert.equal(representedVolume, layout.initialVolumeCellSum);
});

test("complete construction oracle handles ground and disconnected surfaces", () => {
  const nx = 9, ny = 48, nz = 9;
  const volume = new Float32Array(nx * ny * nz);
  const velocity = Array.from({ length: volume.length }, () => ({ x: 0, y: 0, z: 0 }));
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  volume[index(1, 0, 1)] = 1;
  volume[index(1, 1, 1)] = 1;
  volume[index(6, 20, 6)] = 1;
  const settings = adaptiveOpticalLayerSettings(ny);
  const result = constructAdaptiveOpticalLayer({ nx, ny, nz, gridWidth: 1, volume, velocity }, settings);
  assert.equal(result.groundSurfaces[1 + nx], 1);
  assert.equal(result.dilations[1 + nx], settings.minimumDilationCells);
  assert.equal(result.groundSurfaces[6 + nx * 6], -1);
  assert.equal(result.rawBases[6 + nx * 6], 20 - settings.airborneDilationCells);
  assert.equal(result.rawBases[7 + nx * 6], 21 - settings.airborneDilationCells);
  for (let i = 0; i < result.bases.length; i += 1) assert.ok(result.bases[i] <= result.rawBases[i]);
});

test("complete construction oracle expands a high-error ground surface to dmax", () => {
  const nx = 19, ny = 64, nz = 19;
  const volume = new Float32Array(nx * ny * nz);
  const velocity = Array.from({ length: volume.length }, () => ({ x: 0, y: 0, z: 0 }));
  const index = (x: number, y: number, z: number) => x + nx * (y + ny * z);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) for (let y = 0; y < 24; y += 1) volume[index(x, y, z)] = 1;
  for (let y = 0; y < 24; y += 1) velocity[index(9, y, 9)] = { x: y % 2 === 0 ? 100 : -100, y: 0, z: 0 };
  const settings = adaptiveOpticalLayerSettings(ny);
  const result = constructAdaptiveOpticalLayer({ nx, ny, nz, gridWidth: 1, volume, velocity }, settings);
  assert.equal(result.dilations[9 + nx * 9], settings.maximumDilationCells);
  assert.equal(result.rawBases[9 + nx * 9], 23 - settings.maximumDilationCells);
});

test("positive-face divergence is the negative adjoint of the forward gradient", () => {
  const pressure = [0.2, -0.4, 1.3, 0.7, -0.1, 0.9];
  const positiveFaces = [0.6, -0.3, 1.1, 0.2, -0.8];
  const h = 0.25;
  let pressureDivergence = 0;
  for (let cell = 0; cell < pressure.length; cell += 1) {
    const right = cell < positiveFaces.length ? positiveFaces[cell] : 0;
    const left = cell > 0 ? positiveFaces[cell - 1] : 0;
    pressureDivergence += pressure[cell] * (right - left) / h;
  }
  let negativeGradientVelocity = 0;
  for (let face = 0; face < positiveFaces.length; face += 1) {
    negativeGradientVelocity -= positiveFaces[face] * (pressure[face + 1] - pressure[face]) / h;
  }
  assert.ok(Math.abs(pressureDivergence - negativeGradientVelocity) < 1e-12);
});
