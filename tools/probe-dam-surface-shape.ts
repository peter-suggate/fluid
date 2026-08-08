/**
 * Dawn probe for the shape of the published free surface.
 *
 * The `dam-break-ui` water box at `globalFineLevelSetFactor` 1 publishes its
 * whole surface through the coarse level-set directory, so the geometry the
 * browser draws is recoverable on the host without a renderer: fill each
 * published row's size^3 cells with its phi, then read the topmost sign change
 * per (x, z) column. What comes back is the height field the marching-cubes
 * pass sees, in finest-cell units.
 *
 * The number this exists to produce is `interiorRidgeCells`. A closed-box dam
 * break collapsing in +x has a height profile that only falls, until the front
 * reflects off the far wall and the rise appears AT that wall. An interior
 * slab standing taller than the reservoir behind it is not a wave: it is a
 * front that stopped. `peakCells` is deliberately not the gate -- a real
 * reflected crest is tall too -- and neither is any screen-space raster metric,
 * because the projected surface passed every one of those while the water was
 * heaped in the middle of the tank.
 *
 * `phiDistance` is the second half of the same picture, and the one that names
 * a cause. The refinement ladder coarsens on |phi| read as a distance, so this
 * scores the published phi against the true distance to the zero set. An
 * overstatement of twenty cells is a ladder deciding depth from a constant.
 *
 * Deliberately self-contained: it decodes the directory ABI inline rather than
 * importing the audit library, so the same file runs unchanged in a HEAD
 * worktree that predates that library.
 *
 * Usage:
 *   WEBGPU_NODE_MODULE=$PWD/node_modules/webgpu/index.js \
 *     node --import tsx tools/probe-dam-surface-shape.ts
 *
 * Environment:
 *   FLUID_SCENE            scene definition id (default water-box-dam-break)
 *   FLUID_SAMPLE_TIMES_S   comma-separated sample times
 *   FLUID_MAX_DT           fixed step (default 0.008)
 *   FLUID_MAXIMUM_LEAF_SIZE  bisection handle on the leaf ceiling
 *   FLUID_OCTREE_INTERFACE_BAND  bisection handle on the band reach
 *   FLUID_SURFACE_BAND       live surface half-thickness (0=AUTO)
 *   FLUID_FINEST_SURFACE_CELL  live factor-one cut floor (1 or 2)
 *   FLUID_REFINEMENT_REGION_FLOOR  floor for an aligned region over the far half
 *   FLUID_REFINEMENT_REGION_SCOPE  `far-half` (default) or `full`
 *   FLUID_WALL_BAND          live closed-wall look-ahead (1 through 4)
 *   FLUID_SURFACE_ASCII    1 to print a height map per sample
 *   FLUID_SURFACE_SLICE    1 to print an (x, y) leaf-size slice
 *   FLUID_SURFACE_PHI      1 to print phi in cells on that slice
 *   FLUID_SURFACE_ROWS     1 to print raw row-origin histograms
 *   FLUID_STAGED_OWNER_CENSUS  1 to count dense cells mapped to coarse pressure rows
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

import { octreeMethod } from "../lib/methods/octree";
import type { GPUSolverInstance } from "../lib/methods/types";
import { getScenePreset } from "../lib/scenes";
import { requiredFluidDeviceLimits } from "../lib/webgpu-device-limits";
import { readBufferBinding } from "./webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "./webgpu-smoke-isolation";

const HEADER_WORDS = 8;
const ROW_WORDS = 8;
/** PowerCoarseSampleDirectory flag bits consumed here. */
const FLAG_VALID = 1 << 0;
const FLAG_FINITE = 1 << 3;

const sceneId = process.env.FLUID_SCENE === "dam-break-ui" || !process.env.FLUID_SCENE
  ? "water-box-dam-break" : process.env.FLUID_SCENE!;
const dt = Number(process.env.FLUID_MAX_DT ?? 0.008);
const sampleTimes = (process.env.FLUID_SAMPLE_TIMES_S ?? "0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0")
  .split(",").map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value >= 0)
  .sort((left, right) => left - right);
const printAscii = process.env.FLUID_SURFACE_ASCII === "1";
assert.ok(dt > 0 && sampleTimes.length > 0, "Need a positive step and a sample time");

/**
 * How far the published phi overstates the true distance to the zero set.
 *
 * The refinement predicate coarsens a leaf when `minimumAbsolutePhi` exceeds a
 * protection width in metres, i.e. it reads phi as a distance. That is only
 * sound if phi is redistanced. Under a collapsing column the interface is
 * near-vertical, so a cell can sit one cell HORIZONTALLY from the free surface
 * while its phi -- if it is a depth below the surface directly overhead --
 * reads many cells. Every such cell is one the ladder is entitled to coarsen
 * while it is in fact against the interface.
 *
 * Interface cells are published wet cells with a non-wet 6-neighbour; the true
 * distance is Euclidean to the nearest such cell, in finest-cell units.
 */
function phiDistanceFidelity(
  field: Float32Array, dimensions: readonly [number, number, number], cellWidth: number,
): {
  readonly wetSamples: number;
  readonly maximumOverstatementCells: number;
  readonly cellsOverstatedBeyond2: number;
  readonly cellsOverstatedBeyond4: number;
  readonly worst: readonly { at: readonly [number, number, number]; phiCells: number; trueCells: number }[];
} {
  const [nx, ny, nz] = dimensions;
  const at = (x: number, y: number, z: number) => field[x + nx * (y + ny * z)]!;
  // Outside the lattice is a closed wall, not air: counting it as non-wet
  // would make every boundary cell an interface cell and report distance 0
  // along the tank floor.
  const wet = (x: number, y: number, z: number) => {
    if (x < 0 || y < 0 || z < 0 || x >= nx || y >= ny || z >= nz) return true;
    const value = at(x, y, z);
    return Number.isFinite(value) && value < 0;
  };
  const interfaceCells: Array<[number, number, number]> = [];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (!wet(x, y, z)) continue;
    if (!wet(x + 1, y, z) || !wet(x - 1, y, z) || !wet(x, y + 1, z)
      || !wet(x, y - 1, z) || !wet(x, y, z + 1) || !wet(x, y, z - 1)) {
      interfaceCells.push([x, y, z]);
    }
  }
  let wetSamples = 0, maximumOverstatement = 0, beyond2 = 0, beyond4 = 0;
  const worst: Array<{ at: [number, number, number]; phiCells: number; trueCells: number }> = [];
  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    if (!wet(x, y, z)) continue;
    wetSamples += 1;
    let nearest = Infinity;
    for (const [ix, iy, iz] of interfaceCells) {
      const d = (x - ix) ** 2 + (y - iy) ** 2 + (z - iz) ** 2;
      if (d < nearest) nearest = d;
    }
    const trueCells = Math.sqrt(nearest);
    const phiCells = Math.abs(at(x, y, z)) / cellWidth;
    const overstatement = phiCells - trueCells;
    if (overstatement > maximumOverstatement) maximumOverstatement = overstatement;
    if (overstatement > 2) beyond2 += 1;
    if (overstatement > 4) beyond4 += 1;
    worst.push({ at: [x, y, z], phiCells: Number(phiCells.toFixed(2)), trueCells: Number(trueCells.toFixed(2)) });
  }
  worst.sort((a, b) => (b.phiCells - b.trueCells) - (a.phiCells - a.trueCells));
  return {
    wetSamples,
    maximumOverstatementCells: Number(maximumOverstatement.toFixed(3)),
    cellsOverstatedBeyond2: beyond2, cellsOverstatedBeyond4: beyond4,
    worst: worst.slice(0, 6),
  };
}

function histogram(values: readonly number[]): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const value of values) counts[String(value)] = (counts[String(value)] ?? 0) + 1;
  return counts;
}

function decodeFloat(word: number): number {
  return new Float32Array(new Uint32Array([word >>> 0]).buffer)[0]!;
}

interface SurfaceShape {
  readonly rows: number;
  readonly cellWidth: number;
  readonly wettedColumns: number;
  readonly wetCells: number;
  /** Dense liquid samples not covered by a live pressure row. */
  readonly wetCellsWithoutPressureOwner: number;
  /** Wetted columns whose surface sample lies outside every pressure row. */
  readonly surfaceColumnsWithoutPressureOwner: number;
  /** Dry columns enclosed by wet columns along x or z. */
  readonly enclosedDryColumns: number;
  readonly medianHeightCells: number;
  readonly maximumHeightCells: number;
  readonly minimumHeightCells: number;
  /** Height of the tallest column above the median wetted column. */
  readonly peakCells: number;
  /** Columns standing within half a cell of the tallest one. */
  readonly peakColumns: number;
  readonly peakAt: readonly [number, number];
  /** Largest |h(x,z) - h(neighbour)| over 4-connected wetted columns. */
  readonly maximumNeighborStepCells: number;
  /** Mean wetted height per x slab -- the dam-break profile, in cells. */
  readonly profileX: readonly number[];
  /** Leaf edge of the row holding the surface crossing, per wetted column. */
  readonly surfaceRowSizeHistogram: Readonly<Record<string, number>>;
  /** Leaf edge of the air row directly above each crossing. */
  readonly airRowSizeHistogram: Readonly<Record<string, number>>;
  readonly field: Float32Array;
  readonly sizeField: Uint8Array;
  /**
   * Cells by which the profile rises again after its leftmost value.
   *
   * A closed-box dam break collapsing in +x has a monotonically non-increasing
   * height profile until the front reflects; an interior slab standing taller
   * than the reservoir behind it is the ridge this probe was built to catch.
   */
  readonly interiorRidgeCells: number;
  readonly interiorRidgeAtX: number;
  readonly heights: readonly number[];
}

/**
 * Height field from a published directory.
 *
 * Factor-one Losasso appends the transported dense phi lattice after the
 * compact pressure-row capacity and marks generation bit 30. Rendering uses
 * that dense authority first; fall back to piecewise-constant rows for older
 * or Power publications. The crossing is linearly interpolated between the
 * two straddling cell values.
 */
function surfaceShape(
  words: Uint32Array,
  dimensions: readonly [number, number, number],
): SurfaceShape {
  const [nx, ny, nz] = dimensions;
  const rowCount = words[2] ?? 0;
  const cellWidth = decodeFloat(words[7] ?? 0);
  assert.ok(cellWidth > 0, "Coarse level-set cell width is invalid");
  const field = new Float32Array(nx * ny * nz).fill(Number.NaN);
  /** Leaf edge of the row that painted each cell, so size can be read at the
   * surface rather than averaged over the domain. */
  const sizeField = new Uint8Array(nx * ny * nz);
  let rows = 0;
  const densePublished = ((words[1] ?? 0) & 0x4000_0000) !== 0;
  for (let slot = 0; slot < rowCount; slot += 1) {
    const base = HEADER_WORDS + slot * ROW_WORDS;
    const cellPlusOne = words[base] ?? 0;
    const size = words[base + 1] ?? 0;
    const flags = words[base + 5] ?? 0;
    if (cellPlusOne === 0 || size === 0
      || (flags & (FLAG_VALID | FLAG_FINITE)) !== (FLAG_VALID | FLAG_FINITE)) continue;
    const phi = decodeFloat(words[base + 2] ?? 0);
    if (!Number.isFinite(phi)) continue;
    rows += 1;
    const cell = cellPlusOne - 1;
    const ox = cell % nx;
    const oy = Math.floor(cell / nx) % ny;
    const oz = Math.floor(cell / (nx * ny));
    for (let z = oz; z < Math.min(oz + size, nz); z += 1) {
      for (let y = oy; y < Math.min(oy + size, ny); y += 1) {
        for (let x = ox; x < Math.min(ox + size, nx); x += 1) {
          if (!densePublished) field[x + nx * (y + ny * z)] = phi;
          sizeField[x + nx * (y + ny * z)] = size;
        }
      }
    }
  }
  if (densePublished) {
    const volume = nx * ny * nz;
    const entryCapacity = Math.floor((words.length - HEADER_WORDS) / ROW_WORDS);
    const denseStart = entryCapacity - volume;
    assert.ok(denseStart >= 0, "Dense coarse level-set tail is truncated");
    for (let cell = 0; cell < volume; cell += 1) {
      const base = HEADER_WORDS + ROW_WORDS * (denseStart + cell);
      const flags = words[base + 5] ?? 0;
      const cellPlusOne = words[base] ?? 0;
      const size = words[base + 1] ?? 0;
      const phi = decodeFloat(words[base + 2] ?? 0);
      if (cellPlusOne === cell + 1 && size === 1
        && (flags & (FLAG_VALID | FLAG_FINITE)) === (FLAG_VALID | FLAG_FINITE)
        && Number.isFinite(phi)) field[cell] = phi;
    }
  }

  const heights: number[] = new Array(nx * nz).fill(Number.NaN);
  const surfaceRowSizes: number[] = [];
  const airRowSizes: number[] = [];
  let wetCells = 0;
  let wetCellsWithoutPressureOwner = 0;
  let surfaceColumnsWithoutPressureOwner = 0;
  for (let z = 0; z < nz; z += 1) {
    for (let x = 0; x < nx; x += 1) {
      let topWet = -1;
      for (let y = 0; y < ny; y += 1) {
        const value = field[x + nx * (y + ny * z)]!;
        if (Number.isFinite(value) && value < 0) {
          topWet = y; wetCells += 1;
          if (sizeField[x + nx * (y + ny * z)] === 0) wetCellsWithoutPressureOwner += 1;
        }
      }
      if (topWet < 0) continue;
      const surfaceRowSize = sizeField[x + nx * (topWet + ny * z)]!;
      surfaceRowSizes.push(surfaceRowSize);
      if (surfaceRowSize === 0) surfaceColumnsWithoutPressureOwner += 1;
      if (topWet + 1 < ny) {
        // The cell the front has to advance into. A coarse row here is a
        // coarse air cell in contact with the moving interface.
        airRowSizes.push(sizeField[x + nx * (topWet + 1 + ny * z)]!);
      }
      const inside = field[x + nx * (topWet + ny * z)]!;
      const above = topWet + 1 < ny ? field[x + nx * (topWet + 1 + ny * z)] : undefined;
      // Place the crossing inside the top wet cell when the cell above carries
      // a usable positive value; otherwise the surface is at that cell's top.
      const fraction = above !== undefined && Number.isFinite(above) && above > 0
        ? inside / (inside - above) : 1;
      heights[x + nx * z] = topWet + 0.5 + Math.min(1, Math.max(0, fraction));
    }
  }

  const wetted = heights.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (wetted.length === 0) {
    let finite = 0, negative = 0, minimum = Infinity, maximum = -Infinity;
    for (const value of field) {
      if (!Number.isFinite(value)) continue;
      finite += 1; if (value < 0) negative += 1;
      minimum = Math.min(minimum, value); maximum = Math.max(maximum, value);
    }
    throw new Error(`No wetted column: rows=${rows} rowCount=${rowCount} `
      + `finiteCells=${finite} negativeCells=${negative} phi=[${minimum},${maximum}]`);
  }
  const median = wetted[Math.floor(wetted.length / 2)]!;
  const maximum = wetted[wetted.length - 1]!;
  const minimum = wetted[0]!;
  let peakColumns = 0;
  let peakAt: [number, number] = [0, 0];
  for (let z = 0; z < nz; z += 1) {
    for (let x = 0; x < nx; x += 1) {
      const value = heights[x + nx * z]!;
      if (!Number.isFinite(value)) continue;
      if (value >= maximum - 0.5) peakColumns += 1;
      if (value === maximum) peakAt = [x, z];
    }
  }
  let maximumNeighborStep = 0;
  let enclosedDryColumns = 0;
  for (let z = 0; z < nz; z += 1) {
    for (let x = 0; x < nx; x += 1) {
      const value = heights[x + nx * z]!;
      if (!Number.isFinite(value)) {
        const enclosedX = x > 0 && x + 1 < nx
          && Number.isFinite(heights[x - 1 + nx * z])
          && Number.isFinite(heights[x + 1 + nx * z]);
        const enclosedZ = z > 0 && z + 1 < nz
          && Number.isFinite(heights[x + nx * (z - 1)])
          && Number.isFinite(heights[x + nx * (z + 1)]);
        if (enclosedX || enclosedZ) enclosedDryColumns += 1;
        continue;
      }
      for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
        const qx = x + dx, qz = z + dz;
        if (qx >= nx || qz >= nz) continue;
        const other = heights[qx + nx * qz]!;
        if (!Number.isFinite(other)) continue;
        maximumNeighborStep = Math.max(maximumNeighborStep, Math.abs(value - other));
      }
    }
  }
  const profileX: number[] = [];
  for (let x = 0; x < nx; x += 1) {
    let total = 0, count = 0;
    for (let z = 0; z < nz; z += 1) {
      const value = heights[x + nx * z]!;
      if (Number.isFinite(value)) { total += value; count += 1; }
    }
    profileX.push(count > 0 ? Number((total / count).toFixed(4)) : Number.NaN);
  }
  let interiorRidgeCells = 0;
  let interiorRidgeAtX = -1;
  const leftmost = profileX.find((value) => Number.isFinite(value)) ?? 0;
  for (let x = 0; x < nx; x += 1) {
    const value = profileX[x]!;
    if (!Number.isFinite(value)) continue;
    if (value - leftmost > interiorRidgeCells) {
      interiorRidgeCells = Number((value - leftmost).toFixed(4));
      interiorRidgeAtX = x;
    }
  }

  return {
    rows, cellWidth,
    wettedColumns: wetted.length, wetCells,
    wetCellsWithoutPressureOwner, surfaceColumnsWithoutPressureOwner,
    enclosedDryColumns,
    medianHeightCells: Number(median.toFixed(4)),
    maximumHeightCells: Number(maximum.toFixed(4)),
    minimumHeightCells: Number(minimum.toFixed(4)),
    peakCells: Number((maximum - median).toFixed(4)),
    peakColumns, peakAt,
    maximumNeighborStepCells: Number(maximumNeighborStep.toFixed(4)),
    profileX, interiorRidgeCells, interiorRidgeAtX,
    surfaceRowSizeHistogram: histogram(surfaceRowSizes),
    airRowSizeHistogram: histogram(airRowSizes),
    field, sizeField,
    heights,
  };
}

/**
 * One (x, y) slice of the published rows: leaf size, and which side it is on.
 *
 * Digits are wet rows, letters the dry ones (a=1, b=2, c=4, d=8, e=16); `.` is
 * a cell no row published. Read together with the height profile this answers
 * the question the histograms cannot: whether a coarse row is sitting against
 * the near-vertical collapsing front, where |phi| is large but the interface
 * is one cell away horizontally.
 */
function asciiSizeSlice(
  field: Float32Array, sizeField: Uint8Array,
  dimensions: readonly [number, number, number], z: number,
): string {
  const [nx, ny] = dimensions;
  const wet = "123456789";
  const dry = "abcde";
  const lines: string[] = [];
  for (let y = ny - 1; y >= 0; y -= 1) {
    let line = `${String(y).padStart(2)} `;
    for (let x = 0; x < nx; x += 1) {
      const index = x + nx * (y + ny * z);
      const size = sizeField[index]!;
      const value = field[index]!;
      if (size === 0 || !Number.isFinite(value)) { line += "."; continue; }
      const level = Math.round(Math.log2(size));
      line += value < 0 ? wet[Math.min(level, wet.length - 1)]! : dry[Math.min(level, dry.length - 1)]!;
    }
    lines.push(line);
  }
  return lines.join("\n");
}

function asciiHeightMap(shape: SurfaceShape, nx: number, nz: number): string {
  const ramp = " .:-=+*#%@";
  const lines: string[] = [];
  for (let z = 0; z < nz; z += 1) {
    let line = "";
    for (let x = 0; x < nx; x += 1) {
      const value = shape.heights[x + nx * z]!;
      if (!Number.isFinite(value)) { line += " "; continue; }
      const t = (value - shape.minimumHeightCells)
        / Math.max(1e-6, shape.maximumHeightCells - shape.minimumHeightCells);
      line += ramp[Math.min(ramp.length - 1, Math.max(0, Math.round(t * (ramp.length - 1))))]!;
    }
    lines.push(line);
  }
  return lines.join("\n");
}

await acquireWebGPUExclusiveLock("dawn-benchmark", "tools/probe-dam-surface-shape.ts");
try {
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU;
    globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([
    `backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`,
    ...(process.env.FLUID_WEBGPU_ADAPTER ? [`adapter=${process.env.FLUID_WEBGPU_ADAPTER}`] : []),
  ]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  const requiredFeatures: GPUFeatureName[] = ["subgroups"];
  if (adapter.features.has("timestamp-query")) requiredFeatures.push("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures, requiredLimits: requiredFluidDeviceLimits(adapter.limits),
  });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push(event.error.message);
  });

  const scene = getScenePreset(sceneId).create();
  scene.numerics.fixedDt_s = dt;
  scene.numerics.maxDt_s = dt;
  const refinementRegionFloor = Number(process.env.FLUID_REFINEMENT_REGION_FLOOR ?? 0);
  if (refinementRegionFloor > 0) {
    assert.ok(Number.isSafeInteger(refinementRegionFloor)
      && (refinementRegionFloor & (refinementRegionFloor - 1)) === 0,
      "FLUID_REFINEMENT_REGION_FLOOR must be a positive power of two");
    const nx = Math.round(scene.container.width_m / scene.voxelDomain.finestCellSize_m);
    const ny = Math.round(scene.container.height_m / scene.voxelDomain.finestCellSize_m);
    const nz = Math.round(scene.container.depth_m / scene.voxelDomain.finestCellSize_m);
    const refinementRegionScope = process.env.FLUID_REFINEMENT_REGION_SCOPE ?? "far-half";
    assert.ok(refinementRegionScope === "far-half" || refinementRegionScope === "full",
      "FLUID_REFINEMENT_REGION_SCOPE must be far-half or full");
    const alignedMinX = refinementRegionScope === "full" ? 0
      : Math.floor(nx / (2 * refinementRegionFloor)) * refinementRegionFloor;
    const alignedMaxX = refinementRegionScope === "full" ? nx
      : Math.floor(nx / refinementRegionFloor) * refinementRegionFloor;
    const alignedMaxY = refinementRegionScope === "full" ? ny
      : Math.floor(ny / refinementRegionFloor) * refinementRegionFloor;
    const alignedMaxZ = refinementRegionScope === "full" ? nz
      : Math.floor(nz / refinementRegionFloor) * refinementRegionFloor;
    const h = scene.voxelDomain.finestCellSize_m;
    scene.fluid.refinementRegions = [{
      id: `dawn-${refinementRegionScope}`,
      rule: "minimum-cell-size",
      minimumCellSize_cells: refinementRegionFloor,
      min_m: { x: -0.5 * scene.container.width_m + alignedMinX * h, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: -0.5 * scene.container.width_m + alignedMaxX * h,
        y: alignedMaxY * h,
        z: -0.5 * scene.container.depth_m + alignedMaxZ * h },
    }];
  }
  const solver = await octreeMethod.createSolverAsync!(device, scene, "balanced", {
    ...octreeMethod.presetFor("balanced"),
    globalFineLevelSetFactor: "1",
    secondaryParticles: "off",
    // Bisection handles only: the browser never sets these, and the untouched
    // probe is the shipped preset.
    ...(process.env.FLUID_MAXIMUM_LEAF_SIZE
      ? { maximumLeafSize: process.env.FLUID_MAXIMUM_LEAF_SIZE } : {}),
    ...(process.env.FLUID_OCTREE_INTERFACE_BAND
      ? { interfaceRefinementBandCells: Number(process.env.FLUID_OCTREE_INTERFACE_BAND) } : {}),
    ...(process.env.FLUID_SURFACE_BAND
      ? { interfaceBandCells: Number(process.env.FLUID_SURFACE_BAND) } : {}),
    ...(process.env.FLUID_FINEST_SURFACE_CELL
      ? { finestSurfaceCellSize: Number(process.env.FLUID_FINEST_SURFACE_CELL) } : {}),
    ...(process.env.FLUID_WALL_BAND
      ? { wallBandCells: Number(process.env.FLUID_WALL_BAND) } : {}),
  }, undefined, () => {}) as GPUSolverInstance;
  await device.queue.onSubmittedWorkDone();
  const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as
    [number, number, number];

  const projection = (solver as unknown as {
    octreeProjection?: {
      readSolveDiagnostics(): Promise<void>;
      readTopologyLeafCensus(): Promise<{
        leafCountsBySize: Readonly<Record<string, number>>;
        topologyLeaves: number;
        residentOwnerPages: number;
      }>;
      readCoarseSurfaceTrackerReceipt(): Promise<{
        predictedVolume: number;
        targetVolume: number;
        interfaceCells: number;
        correction: number;
        movingInterfaceCells?: number;
        advancingInterfaceCells?: number;
        retreatingInterfaceCells?: number;
        correctedRegionCells?: number;
        frozenCells?: number;
        interfaceVelocityQueries?: number;
        interfaceVelocityValid?: number;
        interfacePhiMoved?: number;
        maximumInterfaceSpeed?: number;
      } | undefined>;
      losassoExtensionControl?: GPUBuffer;
    };
  }).octreeProjection;
  const projectionRuntime = projection as unknown as {
    interfaceBandCellsEffective?: number;
    surfaceGradingLayersEffective?: number;
    finestSurfaceCellSizeEffective?: number;
    wallBandCellsEffective?: number;
  };
  const stagedVelocity = (projection as unknown as {
    losassoBackend?: { sources?: { velocitySampler?: { stagedVelocity?: GPUBuffer } } };
  }).losassoBackend?.sources?.velocitySampler?.stagedVelocity;
  const runtimeTopologyDials = {
    surfaceBandCells: projectionRuntime.interfaceBandCellsEffective,
    surfaceGradingLayers: projectionRuntime.surfaceGradingLayersEffective,
    finestSurfaceCellSize: projectionRuntime.finestSurfaceCellSizeEffective,
    wallBandCells: projectionRuntime.wallBandCellsEffective,
  };

  const samples: Array<Record<string, unknown>> = [];
  let step = 0;
  for (const target of sampleTimes) {
    const wanted = Math.round(target / dt);
    while (step < wanted) {
      step += 1;
      while (!solver.advanceTo(step * dt, [])) {
        await new Promise((resolve) => setImmediate(resolve));
      }
    }
    await device.queue.onSubmittedWorkDone();
    const source = solver.coarseLevelSetSource;
    assert.ok(source, "factor-1 octree published no coarse level set");
    const bytes = await readBufferBinding(device, source.directory,
      source.directory.size ?? source.directory.buffer.size - (source.directory.offset ?? 0));
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const shape = surfaceShape(words, dimensions);
    const { heights: _heights, field: _field, sizeField: _sizeField, ...summary } = shape;
    if (process.env.FLUID_SURFACE_ROWS === "1") {
      const [nx, ny, nz] = dimensions;
      const rowCount = words[2] ?? 0;
      const byY = new Array<number>(ny).fill(0);
      const byZ = new Array<number>(nz).fill(0);
      const byX = new Array<number>(nx).fill(0);
      const preview: string[] = [];
      for (let slot = 0; slot < rowCount; slot += 1) {
        const base = HEADER_WORDS + slot * ROW_WORDS;
        const cellPlusOne = words[base] ?? 0;
        const size = words[base + 1] ?? 0;
        const flags = words[base + 5] ?? 0;
        if (cellPlusOne === 0 || size === 0) continue;
        const cell = cellPlusOne - 1;
        const ox = cell % nx, oy = Math.floor(cell / nx) % ny, oz = Math.floor(cell / (nx * ny));
        if (ox < nx) byX[ox]! += 1;
        if (oy < ny) byY[oy]! += 1;
        if (oz < nz) byZ[oz]! += 1;
        if (preview.length < 8) {
          preview.push(`cell=${cell} -> (${ox},${oy},${oz}) size=${size} `
            + `flags=0x${flags.toString(16)} phi=${(decodeFloat(words[base + 2] ?? 0) / shape.cellWidth).toFixed(2)}c`);
        }
      }
      console.error(`# t=${(step * dt).toFixed(3)} rowCount=${rowCount} maxCellIndex=${nx * ny * nz - 1}`);
      console.error(`  byX: ${byX.join(",")}`);
      console.error(`  byY: ${byY.join(",")}`);
      console.error(`  byZ: ${byZ.join(",")}`);
      console.error(`  ${preview.join("\n  ")}`);
    }
    // The exact nearest-interface scan is O(cells x interface cells); it is the
    // point of the probe on a 24x18x16 box and untenable on a 64x48x64 one.
    const fidelity = dimensions[0] * dimensions[1] * dimensions[2] <= 50_000
      ? phiDistanceFidelity(shape.field, dimensions, shape.cellWidth) : undefined;
    const census = await projection?.readTopologyLeafCensus();
    const coarseVolume = await projection?.readCoarseSurfaceTrackerReceipt();
    const extensionControl = projection?.losassoExtensionControl;
    const extensionControlBytes = extensionControl
      ? await readBufferBinding(device, { buffer: extensionControl }, extensionControl.size)
      : undefined;
    const extensionBandControl = extensionControlBytes
      ? Array.from(new Uint32Array(extensionControlBytes.buffer,
        extensionControlBytes.byteOffset, extensionControlBytes.byteLength / 4))
      : undefined;
    const extensionBandFaces = extensionBandControl?.[2];
    let stagedOwnerCells: number | undefined;
    let stagedMacValid: number | undefined;
    let stagedRawMacValid: number | undefined;
    let stagedMacInvalid: number | undefined;
    let stagedRawBoundary: number | undefined;
    if (process.env.FLUID_STAGED_OWNER_CENSUS === "1" && stagedVelocity) {
      const staged = new Uint32Array(await readBufferBinding(device, { buffer: stagedVelocity },
        stagedVelocity.size));
      const [nx, ny, nz] = dimensions;
      const mac = (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1);
      const stagedFloats = new Float32Array(staged.buffer, staged.byteOffset, staged.length);
      stagedMacValid = stagedFloats.subarray(0, mac)
        .reduce((count, value) => count + Number(Number.isFinite(value)), 0);
      stagedMacInvalid = mac - stagedMacValid;
      stagedRawMacValid = staged.subarray(mac, 2 * mac)
        .reduce((count, value) => count + Number(value !== 0x7fc0_0000
          && value !== 0x7fc0_0001), 0);
      stagedRawBoundary = staged.subarray(mac, 2 * mac)
        .reduce((count, value) => count + Number(value === 0x7fc0_0001), 0);
      stagedOwnerCells = staged.subarray(2 * mac, 2 * mac + nx * ny * nz)
        .reduce((count, encoded) => count + Number(encoded !== 0), 0);
    }
    const stats = await solver.readStats() as unknown as Record<string, unknown>;
    samples.push({
      t_s: Number((step * dt).toFixed(6)), ...summary,
      phiDistance: fidelity,
      leafCountsBySize: census?.leafCountsBySize,
      topologyLeaves: census?.topologyLeaves,
      residentOwnerPages: census?.residentOwnerPages,
      maximumNeighborDelta: stats.maximumNeighborDelta,
      pressureRequiredRows: stats.pressureRequiredRows,
      pressureRowCapacity: stats.pressureRowCapacity,
      pressureCapacityOverflow: stats.pressureCapacityOverflow ? 1 : 0,
      frontierCapacityOverflow: stats.frontierCapacityOverflow ? 1 : 0,
      // The solve's own verdict. A front that will not advance and a residual
      // that will not fall are the same defect seen from two ends, so the
      // shape and the convergence have to be read from one sample.
      pressureResidual: stats.pressureResidual,
      pressureRelativeResidual: stats.pressureRelativeResidual,
      quadtreePressureIterationsUsed: stats.quadtreePressureIterationsUsed,
      coarseVolume,
      extensionBandControl,
      extensionBandFaces,
      stagedOwnerCells,
      stagedMacValid,
      stagedMacInvalid,
      stagedRawMacValid,
      stagedRawBoundary,
      currentVolume: stats.currentVolume,
      referenceVolume: stats.referenceVolume,
      maximumDivergence: stats.maximumDivergence,
      maximumSpeed: stats.maximumSpeed,
    });
    if (printAscii) {
      console.error(`# t=${(step * dt).toFixed(3)} peak=${shape.peakCells} `
        + `columns=${shape.peakColumns} at=${shape.peakAt.join(",")}`);
      console.error(asciiHeightMap(shape, dimensions[0], dimensions[2]));
    }
    if (process.env.FLUID_SURFACE_PHI === "1") {
      const z = Math.floor(dimensions[2] / 2);
      const [nx, ny] = dimensions;
      console.error(`# t=${(step * dt).toFixed(3)} phi in cells at z=${z}`);
      for (let y = ny - 1; y >= 0; y -= 1) {
        let line = `${String(y).padStart(2)} `;
        for (let x = 0; x < nx; x += 1) {
          const value = shape.field[x + nx * (y + ny * z)]!;
          line += Number.isFinite(value)
            ? String(Math.round(value / shape.cellWidth)).padStart(5) : "    .";
        }
        console.error(line);
      }
    }
    if (process.env.FLUID_SURFACE_SLICE === "1") {
      const z = Math.floor(dimensions[2] / 2);
      console.error(`# t=${(step * dt).toFixed(3)} leaf-size slice at z=${z}`);
      console.error(asciiSizeSlice(shape.field, shape.sizeField, dimensions, z));
    }
  }
  solver.destroy();
  console.log(JSON.stringify({
    phase: "dam-surface-shape", scene: sceneId, dt, dimensions, runtimeTopologyDials,
    refinementRegionFloor: refinementRegionFloor || undefined,
    validationErrors, samples,
  }, null, 1));
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
