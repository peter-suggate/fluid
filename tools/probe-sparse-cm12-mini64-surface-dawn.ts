#!/usr/bin/env node
/** Native-Dawn geometry probe for whole-domain and partial min-8 presentation. */
import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import sharp from "sharp";

import "../lib/methods";
import { CM12_PAPER_DT_S } from "../lib/core/cm12-numerics";
import { resolveMethodValues } from "../lib/core/method-contract";
import { parseQueryState } from "../lib/core/url-state";
import { unpackFineLevelSetPackedFlags,
  unpackFineLevelSetPackedPhi } from "../lib/core/fine-levelset-packed-sample";
import { createMinimalPowerDamBreak32Scene,
  createMinimalPowerDamBreak64Scene, createSparseCM12LongDamBreakScene,
  SPARSE_CM12_LONG_DAM_METHOD_PROFILE } from "../lib/core/scenes";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { acquireWebGPUExclusiveLock, releaseWebGPUExclusiveLock } from
  "../lib/harness/webgpu-smoke-isolation";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from
  "../lib/methods/adaptive-mass/method";
import { WebGPUAdaptiveMassSolver } from
  "../lib/methods/adaptive-mass/webgpu-adaptive-mass-solver";

const numericArgument = (name: string, fallback: number): number => {
  const prefix = `--${name}=`;
  const raw = process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length);
  const value = Number(raw ?? fallback);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(`${name} must be a non-negative integer`);
  }
  return value;
};
const stringArgument = (name: string, fallback: string): string => {
  const prefix = `--${name}=`;
  return process.argv.find((value) => value.startsWith(prefix))?.slice(prefix.length)
    ?? fallback;
};
const scenario = stringArgument("scenario", process.env.FLUID_MIN8_SURFACE_SCENARIO ?? "dam");
if (scenario !== "dam" && scenario !== "sphere" && scenario !== "hydrostatic"
  && scenario !== "large-offset" && scenario !== "long-dam") {
  throw new RangeError(
    "scenario must be dam, sphere, hydrostatic, large-offset, or long-dam");
}
const grid = numericArgument("grid", Number(process.env.FLUID_MIN8_SURFACE_GRID ?? 64));
if (grid !== 32 && grid !== 64) throw new RangeError("grid must be 32 or 64");
const region = stringArgument("region", process.env.FLUID_MIN8_SURFACE_REGION ?? "whole");
if (region !== "whole" && region !== "central-x") {
  throw new RangeError("region must be whole or central-x");
}
const steps = numericArgument("steps", scenario === "sphere" ? 0
  : Number(process.env.FLUID_MIN8_SURFACE_STEPS
    ?? process.env.FLUID_MINI64_MIN8_SURFACE_STEPS ?? 7));
const outputPath = resolve(stringArgument("out",
  process.env.FLUID_MINI64_MIN8_SURFACE_OUT
    ?? `artifacts/sparse-cm12-mini${grid}-min8-${region}-${scenario}-surface.json`));
const imagePath = resolve(stringArgument("png",
  process.env.FLUID_MINI64_MIN8_SURFACE_PNG
    ?? `artifacts/sparse-cm12-mini${grid}-min8-${region}-${scenario}-surface.png`));
const dawnModule = process.env.WEBGPU_NODE_MODULE
  ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));

async function readWords(device: GPUDevice, source: GPUBuffer,
  words: number): Promise<Uint32Array> {
  const readback = device.createBuffer({ size: Math.max(4, 4 * words),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, readback, 0, 4 * words);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    return new Uint32Array(readback.getMappedRange()).slice();
  } finally {
    if (readback.mapState === "mapped") readback.unmap();
    readback.destroy();
  }
}

async function readPublishedField(device: GPUDevice,
  solver: WebGPUAdaptiveMassSolver): Promise<Float32Array> {
  const source = solver.globalFineLevelSetSource;
  const { plan } = source;
  assert.equal(plan.fineFactor, 1);
  assert.deepEqual(plan.sampleDimensions,
    [solver.info.nx, solver.info.ny, solver.info.nz]);
  const capacity = plan.maximumResidentBricks;
  const [worklist, metadata, samples] = await Promise.all([
    readWords(device, source.worklist, 7 + capacity),
    readWords(device, source.metadata, 4 * capacity),
    readWords(device, source.samples, plan.payloadCapacityBytes / 4),
  ]);
  const [nx, ny, nz] = plan.sampleDimensions;
  const r = plan.brickResolution;
  const field = new Float32Array(nx * ny * nz).fill(Number.NaN);
  for (let work = 0; work < worklist[1]!; work += 1) {
    const page = worklist[7 + work]!;
    assert.ok(page < capacity);
    const key = metadata[4 * page + 1]!;
    // Sparse CM12 publishes the signed SparseWorld brick key rather than the
    // dense bounded-domain page key used by the older octree publisher.
    const bx = (key & 0x7ff) - 1024;
    const by = ((key >>> 11) & 0x3ff) - 512;
    const bz = ((key >>> 21) & 0x7ff) - 1024;
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const qx = bx * r + local % r;
      const qy = by * r + Math.floor(local / r) % r;
      const qz = bz * r + Math.floor(local / (r * r));
      if (qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz) continue;
      const packed = samples[page * plan.samplesPerBrick + local]!;
      if ((unpackFineLevelSetPackedFlags(packed) & 1) === 0) continue;
      field[qx + nx * (qy + ny * qz)] = unpackFineLevelSetPackedPhi(packed);
    }
  }
  return field;
}

function positiveFacingSurface(field: Float32Array,
  dimensions: readonly [number, number, number], axis: 0 | 1 | 2): {
    readonly values: Float32Array;readonly width: number;readonly height: number;
  } {
  const [nx, ny] = dimensions;
  const tangents = axis === 0 ? [1, 2] as const
    : axis === 1 ? [0, 2] as const : [0, 1] as const;
  const width = dimensions[tangents[0]], height = dimensions[tangents[1]];
  const result = new Float32Array(width * height).fill(Number.NaN);
  const at = (coordinate: readonly [number, number, number]) =>
    field[coordinate[0] + nx * (coordinate[1] + ny * coordinate[2])]!;
  for (let v = 0; v < height; v += 1) for (let u = 0; u < width; u += 1) {
    for (let q = dimensions[axis] - 2; q >= 0; q -= 1) {
      const lowerCoordinate = [0, 0, 0] as [number, number, number];
      lowerCoordinate[axis] = q;lowerCoordinate[tangents[0]] = u;
      lowerCoordinate[tangents[1]] = v;
      const upperCoordinate = [...lowerCoordinate] as [number, number, number];
      upperCoordinate[axis] += 1;
      const lower = at(lowerCoordinate), upper = at(upperCoordinate);
      if (!Number.isFinite(lower) || !Number.isFinite(upper)
        || lower >= 0 || upper < 0) continue;
      const fraction = Math.max(0, Math.min(1, -lower / (upper - lower)));
      result[u + width * v] = q + 0.5 + fraction;
      break;
    }
  }
  return { values: result, width, height };
}

function fieldReceipt(field: Float32Array) {
  let finite = 0, negative = 0, positive = 0, zero = 0;
  let minimum = Number.POSITIVE_INFINITY, maximum = Number.NEGATIVE_INFINITY;
  for (const value of field) {
    if (!Number.isFinite(value)) continue;
    finite += 1;minimum = Math.min(minimum, value);maximum = Math.max(maximum, value);
    if (value < 0) negative += 1;
    else if (value > 0) positive += 1;
    else zero += 1;
  }
  return { finite, negative, positive, zero, minimum, maximum };
}

function heightChangeReceipt(before: Float32Array, after: Float32Array) {
  let samples = 0, beforeSum = 0, afterSum = 0, maximumChangeCells = 0;
  for (let index = 0; index < before.length; index += 1) {
    const first = before[index]!, last = after[index]!;
    if (!Number.isFinite(first) || !Number.isFinite(last)) continue;
    samples += 1;beforeSum += first;afterSum += last;
    maximumChangeCells = Math.max(maximumChangeCells, Math.abs(last - first));
  }
  return { samples, beforeMeanCells: beforeSum / samples,
    afterMeanCells: afterSum / samples,
    meanChangeCells: (afterSum - beforeSum) / samples, maximumChangeCells };
}

function densityHeightReceipt(density: Float32Array, open: Float32Array,
  dimensions: readonly [number, number, number]) {
  const [nx, ny, nz] = dimensions;
  const heights = new Float32Array(nx * nz);
  const floorBrickHeights = new Float32Array(nx * nz);
  let sum = 0;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    let height = 0;
    for (let y = 0; y < ny; y += 1) {
      const at = x + nx * (y + ny * z);
      const fill = density[at]! / Math.max(open[at]!, 1e-6);
      height += fill;
      if (y < 8) floorBrickHeights[x + nx * z] += fill;
    }
    heights[x + nx * z] = height;sum += height;
  }
  return { heights, floorBrickHeights, meanCells: sum / heights.length };
}

function filmVisibilityReceipt(heights: Float32Array,
  visibleSurface: Float32Array, totalHeights: Float32Array,
  nx: number, nz: number) {
  let wetColumns = 0, visibleWetColumns = 0, hiddenWetColumns = 0;
  let hiddenBelowFirstSample = 0, maximumHiddenHeightCells = 0;
  let shortFilmNeighbourPairs = 0, maximumShortFilmNeighbourStepCells = 0;
  const hiddenHeightBuckets = [0, 0, 0, 0];
  for (let index = 0; index < heights.length; index += 1) {
    const height = heights[index]!;
    if (!(height > 1e-3)) continue;
    wetColumns += 1;
    if (Number.isFinite(visibleSurface[index]!)) {
      visibleWetColumns += 1;continue;
    }
    hiddenWetColumns += 1;
    maximumHiddenHeightCells = Math.max(maximumHiddenHeightCells, height);
    if (height <= 0.5 + 1e-4) hiddenBelowFirstSample += 1;
    const bucket = height < 0.125 ? 0 : height < 0.25 ? 1 : height < 0.5 ? 2 : 3;
    hiddenHeightBuckets[bucket] += 1;
  }
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const index = x + nx * z;
    if (!(totalHeights[index]! > 0.5 && totalHeights[index]! <= 8.125)
      || !Number.isFinite(visibleSurface[index]!)) continue;
    for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
      if (x + dx >= nx || z + dz >= nz) continue;
      const other = x + dx + nx * (z + dz);
      if (!(totalHeights[other]! > 0.5 && totalHeights[other]! <= 8.125)
        || !Number.isFinite(visibleSurface[other]!)) continue;
      shortFilmNeighbourPairs += 1;
      maximumShortFilmNeighbourStepCells = Math.max(
        maximumShortFilmNeighbourStepCells,
        Math.abs(visibleSurface[other]! - visibleSurface[index]!),
      );
    }
  }
  const xBricks = Array.from({ length: Math.ceil(nx / 8) }, (_, brickX) => {
    let floorHeight = 0, totalHeight = 0, visible = 0;
    for (let z = 0; z < nz; z += 1) for (let x = 8 * brickX;
      x < Math.min(nx, 8 * (brickX + 1)); x += 1) {
      const index = x + nx * z;
      floorHeight += heights[index]!;totalHeight += totalHeights[index]!;
      visible += Number(Number.isFinite(visibleSurface[index]!));
    }
    const columns = Math.min(8, nx - 8 * brickX) * nz;
    return { brickX, meanFloorHeightCells: floorHeight / columns,
      meanTotalHeightCells: totalHeight / columns, visibleColumns: visible };
  });
  return { wetColumns, visibleWetColumns, hiddenWetColumns,
    hiddenBelowFirstSample, maximumHiddenHeightCells,
    shortFilmNeighbourPairs, maximumShortFilmNeighbourStepCells,
    hiddenHeightBuckets: {
      belowOneEighth: hiddenHeightBuckets[0],
      oneEighthToQuarter: hiddenHeightBuckets[1],
      quarterToHalf: hiddenHeightBuckets[2],
      atLeastHalf: hiddenHeightBuckets[3],
    }, xBricks };
}

function surfaceReceipt(heights: Float32Array, nx: number, nz: number) {
  let columns = 0, neighbourPairs = 0, maximumNeighbourStep = 0;
  let laplacianSamples = 0, absoluteLaplacian = 0;
  let primalSeamSamples = 0, primalSeamAbsoluteSecondDifference = 0;
  let formerDualSeamSamples = 0, formerDualSeamAbsoluteSecondDifference = 0;
  let patchInteriorSamples = 0, patchInteriorAbsoluteSecondDifference = 0;
  const at = (x: number, z: number) => heights[x + nx * z]!;
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const value = at(x, z);
    if (!Number.isFinite(value)) continue;
    columns += 1;
    for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
      if (x + dx >= nx || z + dz >= nz) continue;
      const other = at(x + dx, z + dz);
      if (!Number.isFinite(other)) continue;
      neighbourPairs += 1;
      maximumNeighbourStep = Math.max(maximumNeighbourStep, Math.abs(other - value));
    }
    if (x < 1 || z < 1 || x + 1 >= nx || z + 1 >= nz) continue;
    const neighbours = [at(x - 1, z), at(x + 1, z), at(x, z - 1), at(x, z + 1)];
    if (neighbours.some((sample) => !Number.isFinite(sample))) continue;
    // Exclude true near-vertical fronts; this score is for terraces on the
    // otherwise represented top sheet, not for the dam's physical silhouette.
    if (Math.max(...neighbours.map((sample) => Math.abs(sample - value))) > 3) continue;
    const secondX = Math.abs(neighbours[0]! + neighbours[1]! - 2 * value);
    const secondZ = Math.abs(neighbours[2]! + neighbours[3]! - 2 * value);
    const laplacian = Math.abs(neighbours.reduce((sum, sample) => sum + sample, 0)
      - 4 * value);
    laplacianSamples += 1;absoluteLaplacian += laplacian;
    // A primal coarse cell owns fine samples 0..7. A directional second
    // difference therefore crosses its face when centred at 7 or 0 modulo 8.
    // The former dual-centred reconstruction joined at 3/4; keep that
    // population visible so a phase regression cannot masquerade as a quiet
    // primal seam.
    for (const [coordinate, secondDifference] of [[x, secondX], [z, secondZ]]) {
      const phase = coordinate % 8;
      if (phase === 0 || phase === 7) {
        primalSeamSamples += 1;
        primalSeamAbsoluteSecondDifference += secondDifference;
      } else if (phase === 3 || phase === 4) {
        formerDualSeamSamples += 1;
        formerDualSeamAbsoluteSecondDifference += secondDifference;
      } else {
        patchInteriorSamples += 1;
        patchInteriorAbsoluteSecondDifference += secondDifference;
      }
    }
  }
  const primalSeamMean = primalSeamAbsoluteSecondDifference
    / Math.max(1, primalSeamSamples);
  const formerDualSeamMean = formerDualSeamAbsoluteSecondDifference
    / Math.max(1, formerDualSeamSamples);
  const patchInteriorMean = patchInteriorAbsoluteSecondDifference
    / Math.max(1, patchInteriorSamples);
  return {
    columns, neighbourPairs,
    maximumNeighbourStepCells: maximumNeighbourStep,
    meanAbsoluteLaplacianCells: absoluteLaplacian / Math.max(1, laplacianSamples),
    primalSeamDirectionalSamples: primalSeamSamples,
    primalSeamMeanAbsoluteSecondDifferenceCells: primalSeamMean,
    formerDualSeamDirectionalSamples: formerDualSeamSamples,
    formerDualSeamMeanAbsoluteSecondDifferenceCells: formerDualSeamMean,
    patchInteriorMeanAbsoluteSecondDifferenceCells: patchInteriorMean,
    primalSeamRidgeExcessCells: primalSeamMean - patchInteriorMean,
    primalSeamRidgeRatio: primalSeamMean / Math.max(1e-6, patchInteriorMean),
    formerDualSeamRidgeExcessCells: formerDualSeamMean - patchInteriorMean,
    formerDualSeamRidgeRatio: formerDualSeamMean / Math.max(1e-6, patchInteriorMean),
  };
}

function regionBoundaryReceipt(heights: Float32Array, nx: number, nz: number,
  boundaries: readonly number[]) {
  let samples = 0, stepSum = 0, residualSum = 0;
  let maximumStep = 0, maximumResidual = 0;
  const byBoundary = boundaries.map((boundary) => {
    let boundarySamples = 0, boundaryStepSum = 0, boundaryResidualSum = 0;
    let boundaryMaximumStep = 0, boundaryMaximumResidual = 0;
    for (let z = 0; z < nz; z += 1) {
      if (boundary < 2 || boundary + 1 >= nx) continue;
      const values = [boundary - 2, boundary - 1, boundary, boundary + 1]
        .map((x) => heights[x + nx * z]!);
      if (values.some((value) => !Number.isFinite(value))) continue;
      // A near-vertical dam front is physical geometry, not a refinement seam.
      if (Math.max(...values) - Math.min(...values) > 3) continue;
      const leftSlope = values[1]! - values[0]!;
      const seamStep = values[2]! - values[1]!;
      const rightSlope = values[3]! - values[2]!;
      const residual = Math.abs(seamStep - 0.5 * (leftSlope + rightSlope));
      const step = Math.abs(seamStep);
      boundarySamples += 1;boundaryStepSum += step;boundaryResidualSum += residual;
      boundaryMaximumStep = Math.max(boundaryMaximumStep, step);
      boundaryMaximumResidual = Math.max(boundaryMaximumResidual, residual);
    }
    samples += boundarySamples;stepSum += boundaryStepSum;
    residualSum += boundaryResidualSum;
    maximumStep = Math.max(maximumStep, boundaryMaximumStep);
    maximumResidual = Math.max(maximumResidual, boundaryMaximumResidual);
    return {
      x: boundary, samples: boundarySamples,
      meanStepCells: boundaryStepSum / Math.max(1, boundarySamples),
      maximumStepCells: boundaryMaximumStep,
      meanDetrendedBumpCells: boundaryResidualSum / Math.max(1, boundarySamples),
      maximumDetrendedBumpCells: boundaryMaximumResidual,
    };
  });
  return {
    boundaries: byBoundary, samples,
    meanStepCells: stepSum / Math.max(1, samples),
    maximumStepCells: maximumStep,
    meanDetrendedBumpCells: residualSum / Math.max(1, samples),
    maximumDetrendedBumpCells: maximumResidual,
  };
}

async function writeHeightImage(heights: Float32Array, nx: number, ny: number,
  nz: number): Promise<void> {
  const pixels = Buffer.alloc(nx * nz * 3);
  for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
    const height = heights[x + nx * z]!;
    const value = Number.isFinite(height)
      ? Math.round(255 * Math.max(0, Math.min(1, height / ny))) : 0;
    const at = 3 * (x + nx * (nz - 1 - z));
    pixels[at] = value;pixels[at + 1] = value;pixels[at + 2] = value;
  }
  await mkdir(dirname(imagePath), { recursive: true });
  await sharp(pixels, { raw: { width: nx, height: nz, channels: 3 } })
    .resize(nx * 8, nz * 8, { kernel: "nearest" }).png().toFile(imagePath);
}

await acquireWebGPUExclusiveLock("dawn-probe",
  "tools/probe-sparse-cm12-mini64-surface-dawn.ts");
let device: GPUDevice | undefined;
let solver: WebGPUAdaptiveMassSolver | undefined;
try {
  const dawn = await import(pathToFileURL(dawnModule).href) as {
    create(options: string[]): GPU;globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const gpu = dawn.create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter);
  device = await adapter.requestDevice({
    requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    event.preventDefault();validationErrors.push(event.error.message);
  });
  const largeOffsetUI = scenario === "large-offset" ? parseQueryState(
    "?scene=hydrostatic-power-large-offset&method=adaptive-mass&grid=volume",
  ) : undefined;
  const longDam = scenario === "long-dam";
  const scene = largeOffsetUI?.scene ?? (longDam
    ? createSparseCM12LongDamBreakScene() : grid === 32
    ? createMinimalPowerDamBreak32Scene() : createMinimalPowerDamBreak64Scene());
  if (scenario === "sphere") {
    scene.container.fillFraction = 0;
    scene.fluid.initialCondition = "tank-fill";
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    scene.fluid.initialLiquidVolumes = [{
      shape: "sphere", center_m: { x: 0, y: 0.4, z: 0 }, radius_m: 0.24,
    }];
  } else if (scenario === "hydrostatic") {
    scene.rigidBodies = [];
    // Put the sheet far from the parent-cell midpoint: treating density as a
    // centre sample moves this 8.75-cell waterline by different amounts at B1
    // and B2, while density-integrated height remains exactly invariant.
    scene.container.fillFraction = (Math.floor(grid / 4) + 0.75) / grid;
    scene.fluid.initialCondition = "tank-fill";
    scene.fluid.gravity_m_s2 = { x: 0, y: 0, z: 0 };
    delete scene.fluid.initialDamBreakDimensions_m;
    delete scene.fluid.initialLiquidVolumes;
    delete scene.fluid.initialBrickSeeds_m;
    delete scene.fluid.initialBrickSeedsAdditive;
    delete scene.fluid.inflow;
  }
  scene.duration_s = Math.max(scene.duration_s, steps * CM12_PAPER_DT_S);
  if (!largeOffsetUI) {
    scene.fluid.refinementRegions = [{
      id: `mini${grid}-surface-min8-${region}`,
      rule: "minimum-cell-size",
      minimumCellSize_cells: 8,
      min_m: { x: region === "central-x" ? -0.25 * scene.container.width_m
        : -0.5 * scene.container.width_m, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: region === "central-x" ? 0.25 * scene.container.width_m
        : 0.5 * scene.container.width_m, y: scene.container.height_m,
        z: 0.5 * scene.container.depth_m },
    }];
  }
  const values = resolveMethodValues(adaptiveMassMethod,
    largeOffsetUI?.quality ?? "balanced",
    largeOffsetUI?.overrides[largeOffsetUI.methodId]
      ?? (longDam ? SPARSE_CM12_LONG_DAM_METHOD_PROFILE.overrides : {
      resolutionMode: "adaptive", brickFineResolution: "8",
      presentationPageResolution: "8", timeStep: "paper",
    }));
  if (largeOffsetUI || longDam) {
    const exactSolver = await adaptiveMassMethod.createSolverAsync!(device, scene,
      largeOffsetUI?.quality ?? "balanced", values, undefined, () => {});
    solver = exactSolver as WebGPUAdaptiveMassSolver;
  } else {
    solver = await WebGPUAdaptiveMassSolver.createCompiledTopologyTransport(
      device, scene, "balanced", undefined, adaptiveMassSolverOptions(values), () => {});
  }
  await solver.waitForSimulationReady();
  const resetDimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
  const resetField = await readPublishedField(device, solver);
  const resetPositiveY = positiveFacingSurface(resetField, resetDimensions, 1);
  const [resetActivity, resetPresentationHeader, resetDiagnostic] = scenario
    === "large-offset" ? await Promise.all([solver.readGPUActivityPolicy(),
      solver.readFramePlanPresentationHeaderQA(), solver.readDiagnosticFields()])
      : [undefined, undefined, undefined] as const;
  const resetDensityHeight = resetDiagnostic ? densityHeightReceipt(
    resetDiagnostic.density, resetDiagnostic.solidOpenFraction, resetDimensions)
    : undefined;
  for (let step = 1; step <= steps; step += 1) {
    while (!solver.advanceTo(step * CM12_PAPER_DT_S, [])) await new Promise(setImmediate);
    if (step % 2 === 0) await device.queue.onSubmittedWorkDone();
  }
  await device.queue.onSubmittedWorkDone();
  const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as const;
  const field = await readPublishedField(device, solver);
  const finalDiagnostic = scenario === "large-offset" || longDam
    ? await solver.readDiagnosticFields() : undefined;
  const finalDensityHeight = finalDiagnostic ? densityHeightReceipt(
    finalDiagnostic.density, finalDiagnostic.solidOpenFraction, dimensions)
    : undefined;
  const positiveX = positiveFacingSurface(field, dimensions, 0);
  const positiveY = positiveFacingSurface(field, dimensions, 1);
  const positiveZ = positiveFacingSurface(field, dimensions, 2);
  const surfaces = {
    positiveX: surfaceReceipt(positiveX.values, positiveX.width, positiveX.height),
    positiveY: surfaceReceipt(positiveY.values, positiveY.width, positiveY.height),
    positiveZ: surfaceReceipt(positiveZ.values, positiveZ.width, positiveZ.height),
  };
  const heightChange = heightChangeReceipt(resetPositiveY.values, positiveY.values);
  const filmVisibility = longDam && finalDensityHeight
    ? filmVisibilityReceipt(finalDensityHeight.floorBrickHeights, positiveY.values,
      finalDensityHeight.heights, dimensions[0], dimensions[2])
    : undefined;
  const regionBoundaries = region === "central-x"
    ? [dimensions[0] / 4, 3 * dimensions[0] / 4] : [];
  const boundarySurface = regionBoundaryReceipt(
    positiveY.values, positiveY.width, positiveY.height, regionBoundaries);
  const [presentation, activity] = await Promise.all([
    solver.readPresentationPageAllocatorReceiptQA(), solver.readGPUActivityPolicy(),
  ]);
  const activeResolutions = activity.bricks.filter((brick) => brick.active)
    .map((brick) => brick.acceptedResolution);
  const resolutionHistogram = [...new Set(activeResolutions)].sort((a, b) => a - b)
    .map((resolution) => ({ resolution,
      count: activeResolutions.filter((value) => value === resolution).length }));
  const receipt = {
    probe: longDam ? `sparse-cm12-long-dam-min8-${region}-surface`
      : `sparse-cm12-mini${grid}-min8-${region}-surface`, scenario, steps,
    time_s: steps * CM12_PAPER_DT_S, dimensions,
    field: fieldReceipt(field), surfaces, boundarySurface, heightChange,
    ...(resetDensityHeight && finalDensityHeight ? { densityHeight: {
      beforeMeanCells: resetDensityHeight.meanCells,
      afterMeanCells: finalDensityHeight.meanCells,
      maximumChangeCells: Math.max(...resetDensityHeight.heights.map((height, index) =>
        Math.abs(finalDensityHeight.heights[index]! - height))),
    } } : {}),
    ...(filmVisibility ? { filmVisibility } : {}),
    ...(resetActivity ? { resetSurfaceBricks: resetActivity.bricks.filter((brick) => brick.active
      && brick.coordinate[1] === 1).map((brick) => ({
        coordinate: brick.coordinate,
        resolution: brick.acceptedResolution,
        reasons: brick.reasons,
        meanDensity: brick.meanDensity,
      })) } : {}),
    ...(resetPresentationHeader ? { resetPresentationHeader } : {}),
    resolutionHistogram,
    presentation: {
      generation: solver.globalFineLevelSetSource.generation,
      ...presentation,
    },
    validationErrors, imagePath, outputPath,
  };
  await writeHeightImage(positiveY.values, ...dimensions);
  await mkdir(dirname(outputPath), { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(receipt, null, 2)}\n`);
  process.stdout.write(`${JSON.stringify(receipt, null, 2)}\n`);
  assert.deepEqual(validationErrors, []);
  assert.ok(surfaces.positiveY.columns > 0,
    "the evolved dam did not contain a represented top surface");
  assert.equal(presentation.faultCode, 0,
    "the evolved dam presentation publisher reported a fault");
  if (region === "central-x" && scenario === "hydrostatic" && steps === 0) {
    for (const resolution of [1, 2]) {
      assert.ok(resolutionHistogram.some((entry) => entry.resolution === resolution),
        `the partial min8 scene did not exercise accepted B${resolution}`);
    }
    assert.ok(boundarySurface.samples >= 16,
      "the partial min8 boundary did not intersect enough top-surface columns");
    assert.ok(boundarySurface.maximumDetrendedBumpCells <= 0.05,
      `the partial min8 boundary displaced the planar surface by ${
        boundarySurface.maximumDetrendedBumpCells} cells`);
    assert.ok(surfaces.positiveY.maximumNeighbourStepCells <= 0.125,
      `the planar mini32 surface retained a refinement step of ${
        surfaces.positiveY.maximumNeighbourStepCells} cells`);
    assert.ok(surfaces.positiveY.meanAbsoluteLaplacianCells <= 0.02,
      `the planar mini32 surface retained curvature of ${
        surfaces.positiveY.meanAbsoluteLaplacianCells} cells`);
  }
  if (scenario === "large-offset" && steps === 1) {
    assert.ok(heightChange.samples >= dimensions[0] * dimensions[2] / 2,
      "the large-offset reset/step comparison did not retain enough columns");
    assert.ok(Math.abs(heightChange.beforeMeanCells - 15.25) <= 0.01,
      `the reset waterline was ${heightChange.beforeMeanCells}, expected 15.25 cells`);
    assert.ok(Math.abs(heightChange.meanChangeCells) <= 0.01,
      `the first step moved the mean waterline by ${heightChange.meanChangeCells} cells`);
    assert.ok(heightChange.maximumChangeCells <= 0.02,
      `the first step moved a waterline column by ${heightChange.maximumChangeCells} cells`);
  }
  if (scenario === "long-dam" && steps > 0) {
    assert.ok(filmVisibility);
    assert.equal(filmVisibility.hiddenHeightBuckets.atLeastHalf, 0,
      `coarse cells hid ${filmVisibility.hiddenHeightBuckets.atLeastHalf} floor-film columns `
      + "at least half a fine cell high");
    assert.ok(filmVisibility.xBricks.slice(1).every((brick) =>
      brick.meanTotalHeightCells <= 0.5 || brick.visibleColumns > 0),
    "a non-boundary long-dam brick with representable column mass remained absent");
    assert.ok(filmVisibility.shortFilmNeighbourPairs >= dimensions[2] * 4,
      "the long-dam probe did not retain enough adjacent short-film samples");
    assert.ok(filmVisibility.maximumShortFilmNeighbourStepCells <= 1.1,
      `the reconstructed short film retained a block step of ${
        filmVisibility.maximumShortFilmNeighbourStepCells} cells`);
  }
  if (grid === 64 && region === "whole" && scenario === "dam" && steps === 7) {
    // The former complete-coarse-cell presentation produced a 23.19-cell
    // neighbour jump at this evolved front. The primal patch is about 3-5
    // cells on current Dawn backends; twelve keeps broad physical/numerical
    // headroom while remaining a categorical ridge-regression detector rather
    // than a visual golden image.
    assert.ok(surfaces.positiveY.maximumNeighbourStepCells <= 12,
      `the min8 evolved top surface regained a complete-cell ridge: ${
        surfaces.positiveY.maximumNeighbourStepCells} cells`);
  }
} finally {
  solver?.destroy();device?.destroy();await releaseWebGPUExclusiveLock();
}
