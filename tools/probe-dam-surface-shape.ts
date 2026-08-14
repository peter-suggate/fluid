/**
 * Dawn probe for the shape of the published free surface.
 *
 * The `dam-break-ui` water box publishes an Ando-style sparse finest-cell
 * level-set band independently of the pressure octree. The geometry the
 * browser draws is recoverable on the host without a renderer: paint the
 * coarse fallback from the pressure-row directory, overlay every valid sparse
 * fine sample, then read the topmost sign change per (x, z) column. What comes
 * back is the height field the marching-cubes pass sees, in finest-cell units.
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
 *   FLUID_SAMPLE_EVERY_STEP_THROUGH_S  replace sample times with every dt through this time
 *   FLUID_SURFACE_VOLUME_TELEMETRY_ONLY 1 emits only per-sample volume/wall telemetry
 *   FLUID_SURFACE_VOLUME_SUMMARY_ONLY 1 emits jump extrema and key wall-impact samples
 *   FLUID_MAX_DT           fixed step (default 0.008)
 *   FLUID_MAXIMUM_LEAF_SIZE  bisection handle on the leaf ceiling
 *   FLUID_OCTREE_INTERFACE_BAND  bisection handle on the band reach
 *   FLUID_SURFACE_BAND       live surface half-thickness (0=AUTO)
 *   FLUID_FINEST_SURFACE_CELL  live factor-one cut floor (1 or 2)
 *   FLUID_REFINEMENT_REGION_FLOOR  floor for an aligned region over the far half
 *   FLUID_REFINEMENT_REGION_CEILING optional largest cell for that region
 *   FLUID_REFINEMENT_REGION_SCOPE  `far-half` (default) or `full`
 *   FLUID_WALL_BAND          live closed-wall look-ahead (1 through 4)
 *   FLUID_TOPOLOGY_CADENCE_ADVANCES  accepted advances per topology rebuild (1..8)
 *   FLUID_OCTREE_ADAPTIVITY  0 forces the finest topology; 1 is the production default
 *   FLUID_SURFACE_COMPACT  1 reports only the repeatable physics-gate metrics
 *   FLUID_REQUIRE_CONNECTED_SURFACE  1 fails if authoritative leaf rho>.5 has
 *     a face-disconnected liquid component at any requested sample
 *   FLUID_TOPOLOGY_TRANSITION_DIAGNOSTICS  1 adds compact accepted/candidate
 *     authority, graph, and mass handoff controls at each sample time
 *   FLUID_TRANSACTION_ONLY  1 skips surface decoding and audits only the
 *     fail-closed candidate authority/graph/mass/velocity transaction
 *   FLUID_SURFACE_ASCII    1 to print a height map per sample
 *   FLUID_SURFACE_SLICE    1 to print an (x, y) leaf-size slice
 *   FLUID_SURFACE_PHI      1 to print phi in cells on that slice
 *   FLUID_SURFACE_ROWS     1 to print raw row-origin histograms
 *   FLUID_STAGED_OWNER_CENSUS  1 to count dense cells mapped to coarse pressure rows
 *   FLUID_EXTENSION_BAND_CENSUS  1 to count compact velocity faces by extension layer
 */
import assert from "node:assert/strict";
import { fileURLToPath, pathToFileURL } from "node:url";

// Composition root for this entry point: importing the method catalog installs
// the simulation methods and the octree coarse-dynamics lanes, without which
// constructing a solver throws rather than silently running the wrong backend.
import "../lib/methods";
import { losassoMethod } from "../lib/methods/losasso/method";
import type { GPUSolverInstance } from "../lib/core/method-contract";
import { octreeDebugSources } from "../lib/methods/octree-shared/octree-debug-sources";
import {
  unpackFineLevelSetPackedFlags,
  unpackFineLevelSetPackedPhi,
} from "../lib/core/fine-levelset-packed-sample";
import { getScenePreset } from "../lib/core/scenes";
import type { WebGPUFineLevelSetBrickSource } from "../lib/core/levelset-consumer-abi";
import { unpackAdaptivePhiReceipt } from "../lib/methods/losasso/webgpu-octree-losasso-adaptive-phi";
import { adaptiveMassControlLayout, unpackAdaptiveMassReceipt } from
  "../lib/methods/losasso/webgpu-octree-losasso-adaptive-mass";
import { LOSASSO_SURFACE_GRAPH_CONTROL } from
  "../lib/methods/losasso/webgpu-octree-losasso-surface-graph";
import { requiredFluidDeviceLimits } from "../lib/core/webgpu-device-limits";
import { readBufferBinding } from "../lib/harness/webgpu-smoke-readbacks";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "../lib/harness/webgpu-smoke-isolation";

const HEADER_WORDS = 8;
const ROW_WORDS = 8;
/** PowerCoarseSampleDirectory flag bits consumed here. */
const FLAG_VALID = 1 << 0;
const FLAG_FINITE = 1 << 3;

const sceneId = process.env.FLUID_SCENE === "dam-break-ui" || !process.env.FLUID_SCENE
  ? "water-box-dam-break" : process.env.FLUID_SCENE!;
const dt = Number(process.env.FLUID_MAX_DT ?? 0.008);
const requestedSampleTimes = (process.env.FLUID_SAMPLE_TIMES_S ?? "0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0")
  .split(",").map((value) => Number(value.trim()))
  .filter((value) => Number.isFinite(value) && value >= 0)
  .sort((left, right) => left - right);
const sampleEveryStepThrough = Number(process.env.FLUID_SAMPLE_EVERY_STEP_THROUGH_S ?? 0);
const sampleTimes = sampleEveryStepThrough > 0
  ? Array.from({ length: Math.round(sampleEveryStepThrough / dt) },
    (_value, index) => (index + 1) * dt)
  : requestedSampleTimes;
const printAscii = process.env.FLUID_SURFACE_ASCII === "1";
const compact = process.env.FLUID_SURFACE_COMPACT === "1";
const volumeTelemetryOnly = process.env.FLUID_SURFACE_VOLUME_TELEMETRY_ONLY === "1";
const volumeSummaryOnly = process.env.FLUID_SURFACE_VOLUME_SUMMARY_ONLY === "1";
const topologyTransitionDiagnostics =
  process.env.FLUID_TOPOLOGY_TRANSITION_DIAGNOSTICS === "1";
const transactionOnly = process.env.FLUID_TRANSACTION_ONLY === "1";
assert.ok(dt > 0 && sampleTimes.length > 0, "Need a positive step and a sample time");

function assertCandidateVelocityMigration(candidateDiagnostics: {
  candidate: readonly number[];
  velocityMigration: readonly number[];
}, time: number): void {
  const candidate = candidateDiagnostics.candidate;
  const migration = candidateDiagnostics.velocityMigration;
  assert.equal(migration[0], candidate[0], `t=${time}: velocity migration epoch`);
  assert.equal(migration[1], candidate[2], `t=${time}: velocity migration faces`);
  assert.equal((migration[2] ?? 0) + (migration[3] ?? 0), migration[1],
    `t=${time}: velocity migration coverage`);
  assert.equal(migration[4], 0, `t=${time}: velocity migration errors`);
  assert.equal(migration[5], candidate[0], `t=${time}: velocity migration publication`);
  assert.ok((migration[6] ?? 0) >= 2, `t=${time}: velocity migration completion sweeps`);
  assert.ok((migration[7] ?? Infinity) <= (migration[3] ?? 0),
    `t=${time}: bounded fallback-corner face publication`);
}

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

const authorityControlSummary = (words: Uint32Array | undefined) => words
  ? Object.freeze({
    epoch: words[0] ?? 0, rows: words[1] ?? 0, faces: words[2] ?? 0,
    ready: words[3] ?? 0, errors: words[4] ?? 0,
    topologyReused: words[5] ?? 0, faceDirectoryCapacity: words[6] ?? 0,
    reserved: words[7] ?? 0,
  }) : undefined;

const graphControlSummary = (words: Uint32Array | undefined) => words
  ? Object.freeze({
    epoch: words[LOSASSO_SURFACE_GRAPH_CONTROL.epoch] ?? 0,
    leaves: words[LOSASSO_SURFACE_GRAPH_CONTROL.leafCount] ?? 0,
    nodes: words[LOSASSO_SURFACE_GRAPH_CONTROL.nodeCount] ?? 0,
    published: words[LOSASSO_SURFACE_GRAPH_CONTROL.published] ?? 0,
    errors: words[LOSASSO_SURFACE_GRAPH_CONTROL.errors] ?? 0,
    surfaceGeneration: words[LOSASSO_SURFACE_GRAPH_CONTROL.surfaceGeneration] ?? 0,
    velocityGeneration: words[LOSASSO_SURFACE_GRAPH_CONTROL.velocityGeneration] ?? 0,
    constraints: words[LOSASSO_SURFACE_GRAPH_CONTROL.constraintCount] ?? 0,
    missingLookups: words[LOSASSO_SURFACE_GRAPH_CONTROL.missingLookupCount] ?? 0,
    coverageErrors: words[LOSASSO_SURFACE_GRAPH_CONTROL.coverageErrors] ?? 0,
    reciprocalAdjacencyErrors:
      words[LOSASSO_SURFACE_GRAPH_CONTROL.reciprocalAdjacencyErrors] ?? 0,
    leafClosureErrors: words[LOSASSO_SURFACE_GRAPH_CONTROL.leafClosureErrors] ?? 0,
    capacityErrors: words[LOSASSO_SURFACE_GRAPH_CONTROL.capacityErrors] ?? 0,
    pressureRows: words[LOSASSO_SURFACE_GRAPH_CONTROL.pressureRowCount] ?? 0,
    pressureRowMappingErrors:
      words[LOSASSO_SURFACE_GRAPH_CONTROL.pressureRowMappingErrors] ?? 0,
  }) : undefined;

const massControlSummary = (words: Uint32Array | undefined) => words
  ? Object.freeze({
    magic: words[adaptiveMassControlLayout.magic] ?? 0,
    topologyEpoch: words[adaptiveMassControlLayout.topologyEpoch] ?? 0,
    surfaceGeneration: words[adaptiveMassControlLayout.surfaceGeneration] ?? 0,
    leaves: words[adaptiveMassControlLayout.leafCount] ?? 0,
    donors: words[adaptiveMassControlLayout.donorCount] ?? 0,
    transfers: words[adaptiveMassControlLayout.transferCount] ?? 0,
    missingRecipients: words[adaptiveMassControlLayout.missingRecipients] ?? 0,
    valid: words[adaptiveMassControlLayout.valid] ?? 0,
    errors: words[adaptiveMassControlLayout.errorBits] ?? 0,
  }) : undefined;

async function readControlWords(device: GPUDevice, buffer: GPUBuffer | undefined,
  wordCount: number): Promise<Uint32Array | undefined> {
  if (!buffer) return undefined;
  const byteLength = Math.min(buffer.size, 4 * wordCount);
  const bytes = await readBufferBinding(device, { buffer }, byteLength);
  return Uint32Array.from(new Uint32Array(bytes.buffer, bytes.byteOffset,
    bytes.byteLength / 4));
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
  /** Face-connected components of the published negative level set. */
  readonly wetConnectivity: {
    readonly componentCount: number;
    readonly largestComponentCells: number;
    readonly disconnectedCells: number;
    readonly disconnectedCeilingCells: number;
    readonly disconnectedComponents: readonly {
      readonly cells: number;
      readonly minimum: readonly [number, number, number];
      readonly maximum: readonly [number, number, number];
      readonly minimumPhi: number;
      readonly maximumPhi: number;
    }[];
  };
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
  /** Representation-neutral quadrature over the dense cell-centred scalar.
   * Every backend is sampled by the same trilinear reconstruction and fixed
   * 4^3 subcell lattice, so this is suitable for HEAD/adaptive A/B even when
   * their native level-set storage differs. */
  readonly zeroSetQuadrature: {
    readonly subdivisions: number;
    readonly volumeCells: number;
    readonly centerOfMassCells: readonly [number, number, number];
    readonly frontCells: number;
  };
}

function wetConnectivity(
  field: Float32Array,
  dimensions: readonly [number, number, number],
): SurfaceShape["wetConnectivity"] {
  const [nx, ny, nz] = dimensions;
  const visited = new Uint8Array(field.length);
  const componentSizes: number[] = [];
  const componentCeilingCells: number[] = [];
  const componentDetails: Array<{
    cells: number; minimum: [number, number, number]; maximum: [number, number, number];
    minimumPhi: number; maximumPhi: number;
  }> = [];
  const offsets = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  for (let seed = 0; seed < field.length; seed += 1) {
    if (visited[seed] || !(field[seed]! < 0)) continue;
    visited[seed] = 1;
    const queue = [seed];
    let head = 0, cells = 0, ceilingCells = 0;
    const minimum: [number, number, number] = [nx, ny, nz];
    const maximum: [number, number, number] = [-1, -1, -1];
    let minimumPhi = Infinity, maximumPhi = -Infinity;
    while (head < queue.length) {
      const cell = queue[head++]!;
      const x = cell % nx;
      const y = Math.floor(cell / nx) % ny;
      const z = Math.floor(cell / (nx * ny));
      cells += 1;
      minimum[0] = Math.min(minimum[0], x); minimum[1] = Math.min(minimum[1], y);
      minimum[2] = Math.min(minimum[2], z);
      maximum[0] = Math.max(maximum[0], x); maximum[1] = Math.max(maximum[1], y);
      maximum[2] = Math.max(maximum[2], z);
      minimumPhi = Math.min(minimumPhi, field[cell]!);
      maximumPhi = Math.max(maximumPhi, field[cell]!);
      if (y === ny - 1) ceilingCells += 1;
      for (const [dx, dy, dz] of offsets) {
        const qx = x + dx, qy = y + dy, qz = z + dz;
        if (qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz) continue;
        const neighbor = qx + nx * (qy + ny * qz);
        if (visited[neighbor] || !(field[neighbor]! < 0)) continue;
        visited[neighbor] = 1;
        queue.push(neighbor);
      }
    }
    componentSizes.push(cells);
    componentCeilingCells.push(ceilingCells);
    componentDetails.push({ cells, minimum, maximum, minimumPhi, maximumPhi });
  }
  let largestIndex = -1;
  for (let index = 0; index < componentSizes.length; index += 1) {
    if (largestIndex < 0 || componentSizes[index]! > componentSizes[largestIndex]!) {
      largestIndex = index;
    }
  }
  return Object.freeze({
    componentCount: componentSizes.length,
    largestComponentCells: largestIndex < 0 ? 0 : componentSizes[largestIndex]!,
    disconnectedCells: componentSizes.reduce((sum, value, index) =>
      sum + (index === largestIndex ? 0 : value), 0),
    disconnectedCeilingCells: componentCeilingCells.reduce((sum, value, index) =>
      sum + (index === largestIndex ? 0 : value), 0),
    disconnectedComponents: Object.freeze(componentDetails.filter((_value, index) =>
      index !== largestIndex)),
  });
}

interface WetTransitionAttribution {
  readonly newWetCells: number;
  readonly priorFaceReachedNewWetCells: number;
  readonly persistentFaceBridgeNewWetCells: number;
  readonly drainingFaceBridgeNewWetCells: number;
  readonly newCeilingWetCells: number;
  readonly persistentFaceBridgeNewCeilingWetCells: number;
  readonly drainingFaceBridgeNewCeilingWetCells: number;
  readonly priorContourSeedNewCeilingWetCells: number;
  readonly priorContourFaceNewCeilingWetCells: number;
  readonly newCeilingPriorSpanHistogram: Readonly<Record<string, number>>;
}

/** Attribute newly authoritative rho>.5 cells to the preceding face-CFL band.
 * A prior-wet neighbour which becomes dry in the same step is not a persistent
 * bridge: allowing that threshold swap can visibly nucleate a detached sheet. */
function wetTransitionAttribution(previous: Float32Array, current: Float32Array,
  dimensions: readonly [number, number, number],
  previousSpans?: Uint8Array): WetTransitionAttribution {
  const [nx, ny, nz] = dimensions;
  const offsets = [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0],
    [0, 0, -1], [0, 0, 1]] as const;
  let newWetCells = 0, priorFaceReachedNewWetCells = 0;
  let persistentFaceBridgeNewWetCells = 0, drainingFaceBridgeNewWetCells = 0;
  let newCeilingWetCells = 0, persistentFaceBridgeNewCeilingWetCells = 0;
  let drainingFaceBridgeNewCeilingWetCells = 0;
  let priorContourSeedNewCeilingWetCells = 0;
  let priorContourFaceNewCeilingWetCells = 0;
  const newCeilingPriorSpanHistogram: Record<string, number> = {};
  // One fixed-point finest-cell density unit; values this close to rho=.5
  // distinguish equality/cap creep from a genuinely liquid predecessor.
  const contourTolerance = 1 / 65_536;
  for (let cell = 0; cell < current.length; cell += 1) {
    if (!(current[cell]! < 0) || previous[cell]! < 0) continue;
    newWetCells += 1;
    const x = cell % nx, y = Math.floor(cell / nx) % ny;
    const z = Math.floor(cell / (nx * ny));
    const ceiling = y === ny - 1;
    if (ceiling) {
      newCeilingWetCells += 1;
      if (Math.abs(previous[cell]!) <= contourTolerance) {
        priorContourSeedNewCeilingWetCells += 1;
      }
      const priorSpan = previousSpans?.[cell] ?? 0;
      newCeilingPriorSpanHistogram[String(priorSpan)] =
        (newCeilingPriorSpanHistogram[String(priorSpan)] ?? 0) + 1;
    }
    let priorFace = false, persistentFace = false, drainingFace = false;
    let priorContourFace = false;
    for (const [dx, dy, dz] of offsets) {
      const qx = x + dx, qy = y + dy, qz = z + dz;
      if (qx < 0 || qy < 0 || qz < 0 || qx >= nx || qy >= ny || qz >= nz) continue;
      const neighbor = qx + nx * (qy + ny * qz);
      if (Math.abs(previous[neighbor]!) <= contourTolerance) priorContourFace = true;
      if (!(previous[neighbor]! < 0)) continue;
      priorFace = true;
      if (current[neighbor]! < 0) persistentFace = true;
      else drainingFace = true;
    }
    if (priorFace) priorFaceReachedNewWetCells += 1;
    if (persistentFace) persistentFaceBridgeNewWetCells += 1;
    if (drainingFace) drainingFaceBridgeNewWetCells += 1;
    if (ceiling && persistentFace) persistentFaceBridgeNewCeilingWetCells += 1;
    if (ceiling && drainingFace) drainingFaceBridgeNewCeilingWetCells += 1;
    if (ceiling && priorContourFace) priorContourFaceNewCeilingWetCells += 1;
  }
  return Object.freeze({ newWetCells, priorFaceReachedNewWetCells,
    persistentFaceBridgeNewWetCells, drainingFaceBridgeNewWetCells,
    newCeilingWetCells, persistentFaceBridgeNewCeilingWetCells,
    drainingFaceBridgeNewCeilingWetCells, priorContourSeedNewCeilingWetCells,
    priorContourFaceNewCeilingWetCells,
    newCeilingPriorSpanHistogram: Object.freeze(newCeilingPriorSpanHistogram) });
}

function zeroSetQuadrature(
  field: Float32Array,
  dimensions: readonly [number, number, number],
  subdivisions = 4,
): SurfaceShape["zeroSetQuadrature"] {
  const [nx, ny, nz] = dimensions;
  const at = (x: number, y: number, z: number) =>
    field[Math.min(nx - 1, Math.max(0, x))
      + nx * (Math.min(ny - 1, Math.max(0, y))
        + ny * Math.min(nz - 1, Math.max(0, z)))]!;
  const sample = (px: number, py: number, pz: number) => {
    const qx = Math.min(nx - 1, Math.max(0, px - 0.5));
    const qy = Math.min(ny - 1, Math.max(0, py - 0.5));
    const qz = Math.min(nz - 1, Math.max(0, pz - 0.5));
    const x0 = Math.floor(qx), y0 = Math.floor(qy), z0 = Math.floor(qz);
    const x1 = Math.min(nx - 1, x0 + 1), y1 = Math.min(ny - 1, y0 + 1);
    const z1 = Math.min(nz - 1, z0 + 1);
    const tx = qx - x0, ty = qy - y0, tz = qz - z0;
    let value = 0;
    for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
      for (let dx = 0; dx < 2; dx += 1) {
        const corner = at(dx === 0 ? x0 : x1, dy === 0 ? y0 : y1,
          dz === 0 ? z0 : z1);
        if (!Number.isFinite(corner)) return Number.NaN;
        value += corner * (dx === 0 ? 1 - tx : tx) * (dy === 0 ? 1 - ty : ty)
          * (dz === 0 ? 1 - tz : tz);
      }
    }
    return value;
  };
  let wet = 0, sx = 0, sy = 0, sz = 0, front = 0;
  for (let z = 0; z < nz * subdivisions; z += 1) {
    const pz = (z + 0.5) / subdivisions;
    for (let y = 0; y < ny * subdivisions; y += 1) {
      const py = (y + 0.5) / subdivisions;
      for (let x = 0; x < nx * subdivisions; x += 1) {
        const px = (x + 0.5) / subdivisions;
        if (!(sample(px, py, pz) < 0)) continue;
        wet += 1; sx += px; sy += py; sz += pz; front = Math.max(front, px);
      }
    }
  }
  const scale = subdivisions ** 3;
  return Object.freeze({ subdivisions, volumeCells: wet / scale,
    centerOfMassCells: Object.freeze(wet > 0
      ? [sx / wet, sy / wet, sz / wet] as const : [0, 0, 0] as const),
    frontCells: front });
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
  fineField?: Float32Array,
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
  if (fineField) {
    assert.equal(fineField.length, field.length, "Fine surface field dimensions disagree");
    for (let cell = 0; cell < field.length; cell += 1) {
      const value = fineField[cell]!;
      if (Number.isFinite(value)) field[cell] = value;
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
  let runningTrough = Number.POSITIVE_INFINITY;
  // The last x column is the impact wall. A rise there is the desired splash,
  // not a standing interior curtain; wall climb has its own physics gate.
  for (let x = 0; x < nx - 1; x += 1) {
    const value = profileX[x]!;
    if (!Number.isFinite(value)) continue;
    runningTrough = Math.min(runningTrough, value);
    if (value - runningTrough > interiorRidgeCells) {
      interiorRidgeCells = Number((value - runningTrough).toFixed(4));
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
    wetConnectivity: wetConnectivity(field, dimensions),
    heights,
    zeroSetQuadrature: zeroSetQuadrature(field, dimensions),
  };
}

/** Decode the topology-independent factor-one surface pages into their dense
 * logical coordinates. Missing samples deliberately remain NaN: only the
 * coarse directory may classify points outside the authored narrow band. */
async function readFactorOneFineField(
  device: GPUDevice,
  source: WebGPUFineLevelSetBrickSource,
  dimensions: readonly [number, number, number],
): Promise<Float32Array> {
  const { plan } = source;
  assert.equal(plan.fineFactor, 1, "Dam surface-shape probe expects factor-one fine pages");
  assert.deepEqual(plan.sampleDimensions, dimensions,
    "Factor-one fine surface dimensions disagree with the pressure lattice");
  const capacity = plan.maximumResidentBricks;
  const [worklistBytes, metadataBytes, sampleBytes] = await Promise.all([
    readBufferBinding(device, { buffer: source.worklist }, (7 + capacity) * 4),
    readBufferBinding(device, { buffer: source.metadata }, capacity * 16),
    readBufferBinding(device, { buffer: source.samples },
      capacity * plan.samplesPerBrick * 4),
  ]);
  const worklist = new Uint32Array(worklistBytes.buffer, worklistBytes.byteOffset,
    worklistBytes.byteLength / 4);
  const metadata = new Uint32Array(metadataBytes.buffer, metadataBytes.byteOffset,
    metadataBytes.byteLength / 4);
  const samples = new Uint32Array(sampleBytes.buffer, sampleBytes.byteOffset,
    sampleBytes.byteLength / 4);
  assert.equal(worklist[0], source.generation, "Fine worklist generation is stale");
  assert.ok((worklist[3]! & 3) === 3, "Fine worklist is not initialized and published");
  assert.ok(worklist[1]! <= capacity, "Fine worklist exceeds its page capacity");
  const [nx, ny, nz] = dimensions;
  const [bxCount, byCount] = plan.brickDimensions;
  const r = plan.brickResolution;
  const field = new Float32Array(nx * ny * nz).fill(Number.NaN);
  for (let work = 0; work < worklist[1]!; work += 1) {
    const id = worklist[7 + work] ?? 0xffff_ffff;
    assert.ok(id < capacity, `Fine worklist physical page ${id} exceeds capacity`);
    const base = 4 * id;
    assert.equal(metadata[base], id, "Fine metadata physical identity is malformed");
    assert.equal(metadata[base + 2], source.generation, "Fine metadata generation is stale");
    const key = metadata[base + 1]!;
    assert.ok(key < plan.logicalBrickCount, "Fine metadata logical key is outside the domain");
    const bz = Math.floor(key / (bxCount * byCount));
    const remainder = key - bz * bxCount * byCount;
    const by = Math.floor(remainder / bxCount), bx = remainder - by * bxCount;
    for (let local = 0; local < plan.samplesPerBrick; local += 1) {
      const qx = bx * r + local % r;
      const qy = by * r + Math.floor(local / r) % r;
      const qz = bz * r + Math.floor(local / (r * r));
      if (qx >= nx || qy >= ny || qz >= nz) continue;
      const packed = samples[id * plan.samplesPerBrick + local]!;
      if ((unpackFineLevelSetPackedFlags(packed) & FLAG_VALID) === 0) continue;
      const value = unpackFineLevelSetPackedPhi(packed);
      assert.ok(Number.isFinite(value), `Fine phi is non-finite at ${qx},${qy},${qz}`);
      field[qx + nx * (qy + ny * qz)] = value;
    }
  }
  return field;
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

  const scenePreset = getScenePreset(sceneId);
  const scene = scenePreset.create();
  scene.numerics.fixedDt_s = dt;
  scene.numerics.maxDt_s = dt;
  const refinementRegionFloor = Number(process.env.FLUID_REFINEMENT_REGION_FLOOR ?? 0);
  const refinementRegionCeiling = Number(process.env.FLUID_REFINEMENT_REGION_CEILING ?? 0);
  if (refinementRegionFloor > 0) {
    assert.ok(Number.isSafeInteger(refinementRegionFloor)
      && (refinementRegionFloor & (refinementRegionFloor - 1)) === 0,
      "FLUID_REFINEMENT_REGION_FLOOR must be a positive power of two");
    assert.ok(refinementRegionCeiling === 0 || (Number.isSafeInteger(refinementRegionCeiling)
      && (refinementRegionCeiling & (refinementRegionCeiling - 1)) === 0
      && refinementRegionCeiling >= refinementRegionFloor),
    "FLUID_REFINEMENT_REGION_CEILING must be a power of two no smaller than the floor");
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
      ...(refinementRegionCeiling > 0
        ? { maximumCellSize_cells: refinementRegionCeiling } : {}),
      min_m: { x: -0.5 * scene.container.width_m + alignedMinX * h, y: 0,
        z: -0.5 * scene.container.depth_m },
      max_m: { x: -0.5 * scene.container.width_m + alignedMaxX * h,
        y: alignedMaxY * h,
        z: -0.5 * scene.container.depth_m + alignedMaxZ * h },
    }];
  }
  const solverQuality = scenePreset.methodProfile?.quality ?? "balanced";
  const solverValues = {
    ...losassoMethod.presetFor(solverQuality),
    ...scenePreset.methodProfile?.overrides,
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
    ...(process.env.FLUID_TOPOLOGY_CADENCE_ADVANCES
      ? { topologyCadenceAdvances: Number(process.env.FLUID_TOPOLOGY_CADENCE_ADVANCES) } : {}),
    ...(process.env.FLUID_OCTREE_ADAPTIVITY
      ? { octreeAdaptivity: Number(process.env.FLUID_OCTREE_ADAPTIVITY) } : {}),
  };
  const solver = await losassoMethod.createSolverAsync!(device, scene, solverQuality,
    solverValues, undefined, () => {}) as GPUSolverInstance;
  solver.applyRuntimeValues?.(solverValues);
  await device.queue.onSubmittedWorkDone();
  const dimensions = [solver.info.nx, solver.info.ny, solver.info.nz] as
    [number, number, number];
  const debug = octreeDebugSources(solver);

  const projection = (solver as unknown as {
    octreeProjection?: {
      readSolveDiagnostics(): Promise<void>;
      readTopologyLeafCensus(): Promise<{
        leafCountsBySize: Readonly<Record<string, number>>;
        topologyLeaves: number;
        topologyNodes: number;
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
      readAdaptiveNodeReceipt(): Promise<{
        count: number;
        generation: number;
        published: boolean;
        errors: number;
        capacity: number;
        dispatch: readonly [number, number, number];
      } | undefined>;
      readAdaptiveVelocityReceipts(): Promise<readonly number[] | undefined>;
      readLosassoAuthorityDiagnostics(): Promise<Readonly<{
        candidate: readonly number[];
        candidateAdaptiveGraph: readonly number[];
        ownerCandidate: readonly number[];
        frontierControl: readonly number[];
        adaptiveMassControl: readonly number[];
        adaptiveMassReceipts: readonly number[];
        velocityMigration: readonly number[];
      }> | undefined>;
      readPowerFrontierFailure(): Promise<Record<string, unknown>>;
      powerLeafHeaders?: GPUBuffer;
      powerCandidateLeafHeaders?: GPUBuffer;
      losassoExtensionControl?: GPUBuffer;
      losassoBackend?: {
        sources?: { operator?: { control: GPUBuffer } };
        candidateAuthorityControl?: GPUBuffer;
        candidateTopologyCapacities?: Readonly<Record<string, number>>;
        adaptiveSurfaceGraphSources?: {
          accepted: { control: GPUBuffer; leaves: GPUBuffer; nodes: GPUBuffer;
            phi: GPUBuffer; surfaceMass: GPUBuffer; nodalVelocity: GPUBuffer };
          candidate: { control: GPUBuffer; leaves: GPUBuffer; nodes: GPUBuffer;
            phi: GPUBuffer; surfaceMass: GPUBuffer; nodalVelocity: GPUBuffer };
        };
        extensionBand?: { source?: { faceMetrics?: GPUBuffer } };
        adaptivePhiSource?: { receipts: GPUBuffer };
        adaptiveMassSource?: { control: GPUBuffer; receipts: GPUBuffer;
          transferRecords: GPUBuffer; transportAdmission: GPUBuffer };
        adaptiveVelocity?: { candidateStencilControl: GPUBuffer };
      };
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
  let previousMassVisibleVolumeCells: number | undefined;
  let previousMassVisibleVolumeStep: number | undefined;
  let previousReconstructedVisibleVolumeCells: number | undefined;
  let previousReconstructedVisibleVolumeStep: number | undefined;
  let previousAuthoritativeField: Float32Array | undefined;
  let previousAuthoritativeFieldStep: number | undefined;
  let previousAuthoritativeSpanField: Uint8Array | undefined;
  for (const target of sampleTimes) {
    const wanted = Math.round(target / dt);
    while (step < wanted) {
      step += 1;
      try {
        while (!solver.advanceTo(step * dt, [])) {
          await new Promise((resolve) => setImmediate(resolve));
        }
      } catch (error) {
        if (process.env.FLUID_CAPTURE_POWER_FRONTIER_FAILURE === "1" && projection) {
          await device.queue.onSubmittedWorkDone();
          const failure = await projection.readPowerFrontierFailure();
          const frontier = Array.isArray(failure.frontier)
            ? failure.frontier.map(Number) : [];
          const active = frontier[2] ?? 0;
          const inactive = frontier[7] ?? (1 - active);
          const acceptedCount = frontier[active] ?? 0;
          const candidateCount = frontier[inactive] ?? 0;
          const leafSizeHistogram = async (buffer: GPUBuffer | undefined, count: number) => {
            if (!buffer || count <= 0) return {};
            const rows = Math.min(count, Math.floor(buffer.size / 48));
            const bytes = await readBufferBinding(device, { buffer }, rows * 48);
            const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
            const histogram: Record<string, number> = {};
            for (let row = 0; row < rows; row += 1) {
              const span = words[12 * row + 3] ?? 0;
              histogram[String(span)] = (histogram[String(span)] ?? 0) + 1;
            }
            return histogram;
          };
          const [acceptedLeafSizes, inactiveLeafSizes, authority] = await Promise.all([
            leafSizeHistogram(projection.powerLeafHeaders, acceptedCount),
            leafSizeHistogram(projection.powerCandidateLeafHeaders, candidateCount),
            projection.readLosassoAuthorityDiagnostics(),
          ]);
          console.error("[dam-frontier-failure]", JSON.stringify({
            step, t_s: step * dt,
            candidateAuthority: authority?.candidate,
            candidateTopologyCapacities: projection.losassoBackend
              ?.candidateTopologyCapacities,
            candidateGraph: authority?.candidateAdaptiveGraph,
            ownerCandidate: failure.ownerCandidate,
            frontier,
            frontierFailure: failure.frontierFailure,
            frontierPublication: failure.frontierPublication,
            dirtyAuthority: failure.dirtyAuthority,
            dirtyAuthorityState: failure.dirtyAuthorityState,
            rowDelta: failure.rowDelta,
            candidateSchedules: failure.candidateSchedules,
            controlSummary: failure.controlSummary,
            acceptedCount, candidateCount, acceptedLeafSizes, inactiveLeafSizes,
          }));
        }
        throw error;
      }
    }
    await device.queue.onSubmittedWorkDone();
    if (transactionOnly) {
      // Consume/re-arm the same LosassoStepSnapshot ring used by the UI
      // fail-stop path. Sampling authority buffers alone would miss a rejected
      // intermediate candidate that a later topology attempt overwrote.
      await projection?.readSolveDiagnostics();
      const candidateDiagnostics = await projection?.readLosassoAuthorityDiagnostics();
      assert.ok(candidateDiagnostics, `t=${step * dt}: candidate diagnostics absent`);
      const candidate = candidateDiagnostics.candidate;
      const graph = candidateDiagnostics.candidateAdaptiveGraph;
      const mass = candidateDiagnostics.adaptiveMassControl;
      if (candidate[0] === 0) {
        // Epoch zero is an intentional cadence-reuse step. Graph, mass, and
        // migration receipts retain the preceding candidate and therefore do
        // not form a tuple with this empty authority bank. The UI-equivalent
        // step snapshot above remains responsible for rejecting any latched
        // authority error before this diagnostic branch runs.
        assert.deepEqual(candidate.slice(3, 5), [0, 0],
          `t=${step * dt}: absent candidate carried a fatal verdict`);
        samples.push({ t_s: step * dt,
          candidateAuthority: candidate,
          candidateGraph: graph,
          massControl: mass,
          velocityMigration: candidateDiagnostics.velocityMigration });
        continue;
      }
      assertCandidateVelocityMigration(candidateDiagnostics, step * dt);
      assert.equal(candidate[3], 1, `t=${step * dt}: candidate authority publication `
        + `candidate=${candidate.join("/")} graph=${graph.slice(0, 7).join("/")}`);
      assert.equal(candidate[4], 0, `t=${step * dt}: candidate authority errors`);
      assert.equal(graph[0], candidate[0], `t=${step * dt}: candidate graph epoch`);
      assert.equal(graph[3], graph[0], `t=${step * dt}: candidate graph publication`);
      assert.equal(graph[4], 0, `t=${step * dt}: candidate graph errors`);
      assert.equal(graph[6], graph[5], `t=${step * dt}: candidate graph velocity`);
      assert.equal(mass[1], candidate[0], `t=${step * dt}: candidate mass epoch`);
      assert.equal(mass[7], 1, `t=${step * dt}: candidate mass publication`);
      assert.equal(mass[12], 0, `t=${step * dt}: candidate mass errors`);
      samples.push({ t_s: step * dt,
        candidateAuthority: candidate,
        candidateGraph: graph,
        massControl: mass,
        velocityMigration: candidateDiagnostics.velocityMigration });
      continue;
    }
    const fineSource = solver.globalFineLevelSetSource;
    const coarseSource = solver.coarseLevelSetSource;
    const directory = fineSource?.coarsePhiDirectory
      ? { buffer: fineSource.coarsePhiDirectory }
      : coarseSource?.directory;
    assert.ok(directory, "factor-one octree published no coarse fallback directory");
    const directoryBytes = fineSource?.coarsePhiDirectory
      ? 32 + (fineSource.coarsePhiRowCapacity ?? 0) * ROW_WORDS * 4
      : directory.size ?? directory.buffer.size - (directory.offset ?? 0);
    const bytes = await readBufferBinding(device, directory, directoryBytes);
    const words = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
    const fineField = fineSource
      ? await readFactorOneFineField(device, fineSource, dimensions) : undefined;
    const shape = surfaceShape(words, dimensions, fineField);
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
    const adaptiveNodes = await projection?.readAdaptiveNodeReceipt();
    const adaptiveVelocityReceipts = await projection?.readAdaptiveVelocityReceipts();
    const candidateDiagnostics = await projection?.readLosassoAuthorityDiagnostics();
    // The ordinary grading probe may sample immediately after a candidate was
    // committed and its authority bank cleared; only the transaction-only
    // lane promises a live candidate tuple at every requested checkpoint.
    if (transactionOnly && candidateDiagnostics) {
      assertCandidateVelocityMigration(candidateDiagnostics, step * dt);
    }
    const adaptivePhiReceipts = projection?.losassoBackend?.adaptivePhiSource?.receipts;
    const adaptivePhiReceiptBytes = adaptivePhiReceipts
      ? await readBufferBinding(device, { buffer: adaptivePhiReceipts }, adaptivePhiReceipts.size)
      : undefined;
    const adaptivePhiReceipt = adaptivePhiReceiptBytes
      ? unpackAdaptivePhiReceipt(new Uint32Array(adaptivePhiReceiptBytes.buffer,
        adaptivePhiReceiptBytes.byteOffset, adaptivePhiReceiptBytes.byteLength / 4))
      : undefined;
    const adaptiveMassReceipts = projection?.losassoBackend?.adaptiveMassSource?.receipts;
    const adaptiveMassReceiptBytes = adaptiveMassReceipts
      ? await readBufferBinding(device, { buffer: adaptiveMassReceipts }, adaptiveMassReceipts.size)
      : undefined;
    const adaptiveMassReceipt = adaptiveMassReceiptBytes
      ? unpackAdaptiveMassReceipt(new Uint32Array(adaptiveMassReceiptBytes.buffer,
        adaptiveMassReceiptBytes.byteOffset, adaptiveMassReceiptBytes.byteLength / 4))
      : undefined;
    let adaptiveMassDensity: {
      dry: number; airSide: number; liquidSide: number; compressed: number;
      maximum: number; airSideMass_m3: number; compressedMass_m3: number;
      compressionExcessMass_m3: number;
    } | undefined;
    let massSurfaceConnectivity: SurfaceShape["wetConnectivity"] | undefined;
    let massSurfaceQuadrature: SurfaceShape["zeroSetQuadrature"] | undefined;
    let massSurfaceWallProximity: {
      positiveXGapCells: number;
      ceilingGapCellLayers: number;
      wallWetCells: readonly [number, number, number, number, number, number];
    } | undefined;
    let massWetTransition: WetTransitionAttribution | undefined;
    let currentAuthoritativeField: Float32Array | undefined;
    let currentAuthoritativeSpanField: Uint8Array | undefined;
    let ceilingWetLeaves: readonly Record<string, unknown>[] | undefined;
    let ceilingTransferTrace: readonly Record<string, unknown>[] | undefined;
    let reconstructionMismatchNeighborhood: Record<string, unknown> | undefined;
    let adaptiveVelocityProfile: readonly Record<string, number>[] | undefined;
    const acceptedMassGraph = projection?.losassoBackend
      ?.adaptiveSurfaceGraphSources?.accepted;
    if (acceptedMassGraph) {
      const graphControl = await readControlWords(device, acceptedMassGraph.control, 32);
      const leafCount = Math.min(graphControl?.[1] ?? 0,
        Math.floor(acceptedMassGraph.leaves.size / 64),
        Math.floor(acceptedMassGraph.surfaceMass.size / 4));
      if (leafCount > 0) {
        const [leafBytes, massBytes] = await Promise.all([
          readBufferBinding(device, { buffer: acceptedMassGraph.leaves }, leafCount * 64),
          readBufferBinding(device, { buffer: acceptedMassGraph.surfaceMass }, leafCount * 4),
        ]);
        const leaves = new Uint32Array(leafBytes.buffer, leafBytes.byteOffset,
          leafBytes.byteLength / 4);
        const mass = new Float32Array(massBytes.buffer, massBytes.byteOffset,
          massBytes.byteLength / 4);
        let dry = 0, airSide = 0, liquidSide = 0, compressed = 0;
        let maximum = 0, airSideMass_m3 = 0, compressedMass_m3 = 0;
        let compressionExcessMass_m3 = 0;
        let maximumWetY = -1;
        const wetCeilingLeafRows: Array<Record<string, unknown>> = [];
        const wallWetCells: [number, number, number, number, number, number] =
          [0, 0, 0, 0, 0, 0];
        const massPseudoPhi = new Float32Array(dimensions[0] * dimensions[1]
          * dimensions[2]).fill(0.5);
        const massSpanField = new Uint8Array(massPseudoPhi.length);
        for (let leaf = 0; leaf < leafCount; leaf += 1) {
          const origin = [leaves[16 * leaf] ?? 0, leaves[16 * leaf + 1] ?? 0,
            leaves[16 * leaf + 2] ?? 0] as const;
          const span = leaves[16 * leaf + 3] ?? 0;
          const volume = (span * shape.cellWidth) ** 3;
          const value = volume > 0 ? (mass[leaf] ?? 0) / volume : 0;
          if (value > 0.5 && origin[1] + span >= dimensions[1]) {
            wetCeilingLeafRows.push({ leaf, origin, span, rho: value,
              mass_m3: mass[leaf] ?? 0 });
          }
          for (let z = origin[2]; z < Math.min(dimensions[2], origin[2] + span); z += 1) {
            for (let y = origin[1]; y < Math.min(dimensions[1], origin[1] + span); y += 1) {
              for (let x = origin[0]; x < Math.min(dimensions[0], origin[0] + span); x += 1) {
                massPseudoPhi[x + dimensions[0] * (y + dimensions[1] * z)] = 0.5 - value;
                massSpanField[x + dimensions[0] * (y + dimensions[1] * z)] = span;
                if (value > 0.5) {
                  maximumWetY = Math.max(maximumWetY, y);
                  if (x === 0) wallWetCells[0]! += 1;
                  if (x === dimensions[0] - 1) wallWetCells[1]! += 1;
                  if (y === 0) wallWetCells[2]! += 1;
                  if (y === dimensions[1] - 1) wallWetCells[3]! += 1;
                  if (z === 0) wallWetCells[4]! += 1;
                  if (z === dimensions[2] - 1) wallWetCells[5]! += 1;
                }
              }
            }
          }
          maximum = Math.max(maximum, value);
          if (value <= 1e-8) dry += 1;
          else if (value < 0.5) { airSide += 1; airSideMass_m3 += mass[leaf] ?? 0; }
          else if (value <= 1) liquidSide += 1;
          else {
            compressed += 1;
            compressedMass_m3 += mass[leaf] ?? 0;
            compressionExcessMass_m3 += Math.max(0, (mass[leaf] ?? 0) - volume);
          }
        }
        adaptiveMassDensity = { dry, airSide, liquidSide, compressed, maximum,
          airSideMass_m3, compressedMass_m3, compressionExcessMass_m3 };
        massSurfaceConnectivity = wetConnectivity(massPseudoPhi, dimensions);
        ceilingWetLeaves = Object.freeze(wetCeilingLeafRows);
        currentAuthoritativeField = massPseudoPhi;
        currentAuthoritativeSpanField = massSpanField;
        if (previousAuthoritativeField
          && previousAuthoritativeFieldStep !== undefined
          && step - previousAuthoritativeFieldStep === 1) {
          massWetTransition = wetTransitionAttribution(previousAuthoritativeField,
            massPseudoPhi, dimensions, previousAuthoritativeSpanField);
        }
        massSurfaceQuadrature = zeroSetQuadrature(massPseudoPhi, dimensions);
        massSurfaceWallProximity = {
          positiveXGapCells: Math.max(0, dimensions[0] - massSurfaceQuadrature.frontCells),
          ceilingGapCellLayers: maximumWetY < 0
            ? dimensions[1] : dimensions[1] - 1 - maximumWetY,
          wallWetCells: Object.freeze(wallWetCells),
        };
        if (process.env.FLUID_TRACE_CEILING_TRANSFERS === "1"
          && wetCeilingLeafRows.length > 0) {
          const adaptiveMassSource = projection?.losassoBackend?.adaptiveMassSource;
          const massControl = await readControlWords(device, adaptiveMassSource?.control, 32);
          const transferCount = Math.min(massControl?.[5] ?? 0,
            Math.floor((adaptiveMassSource?.transferRecords.size ?? 0) / 16));
          if (adaptiveMassSource && transferCount > 0) {
            const [transferBytes, admissionBytes] = await Promise.all([
              readBufferBinding(device, { buffer: adaptiveMassSource.transferRecords },
                transferCount * 16),
              readBufferBinding(device, { buffer: adaptiveMassSource.transportAdmission },
                leafCount * 4),
            ]);
            const transferWords = new Uint32Array(transferBytes.buffer,
              transferBytes.byteOffset, transferBytes.byteLength / 4);
            const admissionWords = new Uint32Array(admissionBytes.buffer,
              admissionBytes.byteOffset, admissionBytes.byteLength / 4);
            const ceilingSlots = new Set(wetCeilingLeafRows.map((row) => Number(row.leaf)));
            const traced: Array<Record<string, unknown>> = [];
            for (let transfer = 0; transfer < transferCount; transfer += 1) {
              const donor = transferWords[4 * transfer] ?? 0xffff_ffff;
              const recipient = transferWords[4 * transfer + 1] ?? 0xffff_ffff;
              const units = transferWords[4 * transfer + 2] ?? 0;
              const flags = transferWords[4 * transfer + 3] ?? 0;
              if (!ceilingSlots.has(recipient) || units === 0 || (flags & 1) === 0) continue;
              const donorBase = 16 * donor;
              traced.push({ transfer, donor, recipient, units, flags,
                donorOrigin: donor < leafCount ? [leaves[donorBase] ?? 0,
                  leaves[donorBase + 1] ?? 0, leaves[donorBase + 2] ?? 0] : undefined,
                donorSpan: donor < leafCount ? leaves[donorBase + 3] : undefined });
            }
            for (const row of wetCeilingLeafRows) {
              const leaf = Number(row.leaf);
              const admission = admissionWords[leaf] ?? 0;
              row.admissionReach = (admission & 0x8000_0000) !== 0;
              row.admissionRemoteUnits = admission & 0x7fff_ffff;
            }
            ceilingTransferTrace = Object.freeze(traced);
          }
        }
      }
    }
    if (process.env.FLUID_VELOCITY_PROFILE === "1" && acceptedMassGraph) {
      const control = await readControlWords(device, acceptedMassGraph.control, 32);
      const nodeCount = Math.min(control?.[2] ?? 0,
        Math.floor(acceptedMassGraph.nodes.size / 16),
        Math.floor(acceptedMassGraph.phi.size / 8),
        Math.floor(acceptedMassGraph.nodalVelocity.size / 32));
      const [nodeBytes, phiBytes, velocityBytes] = await Promise.all([
        readBufferBinding(device, { buffer: acceptedMassGraph.nodes }, nodeCount * 16),
        readBufferBinding(device, { buffer: acceptedMassGraph.phi }, nodeCount * 8),
        readBufferBinding(device, { buffer: acceptedMassGraph.nodalVelocity }, nodeCount * 32),
      ]);
      const nodes = new Uint32Array(nodeBytes.buffer, nodeBytes.byteOffset,
        nodeBytes.byteLength / 4);
      const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset,
        phiBytes.byteLength / 4);
      const velocityWords = new Uint32Array(velocityBytes.buffer, velocityBytes.byteOffset,
        velocityBytes.byteLength / 4);
      const velocity = new Float32Array(velocityBytes.buffer, velocityBytes.byteOffset,
        velocityBytes.byteLength / 4);
      const bins = Array.from({ length: dimensions[0] + 1 }, () => ({
        count: 0, sumX: 0, minimumX: Infinity, maximumX: -Infinity,
      }));
      for (let node = 0; node < nodeCount; node += 1) {
        if (Math.min(Math.abs(phi[2 * node] ?? Infinity),
          Math.abs(phi[2 * node + 1] ?? Infinity)) > 2 * shape.cellWidth) continue;
        const mask = velocityWords[8 * node + 3] ?? 0;
        const vx = velocity[8 * node] ?? Number.NaN;
        if ((mask & 7) !== 7 || !Number.isFinite(vx)) continue;
        const lattice = nodes[4 * node] ?? 0;
        const x = lattice % (dimensions[0] + 1);
        const bin = bins[x]!;
        bin.count += 1; bin.sumX += vx;
        bin.minimumX = Math.min(bin.minimumX, vx);
        bin.maximumX = Math.max(bin.maximumX, vx);
      }
      adaptiveVelocityProfile = bins.map((bin, x) => ({ x, count: bin.count,
        meanX: bin.count ? Number((bin.sumX / bin.count).toFixed(5)) : Number.NaN,
        minimumX: bin.count ? Number(bin.minimumX.toFixed(5)) : Number.NaN,
        maximumX: bin.count ? Number(bin.maximumX.toFixed(5)) : Number.NaN }));
    }
    if (adaptiveMassReceipt
      && adaptiveMassReceipt.firstReconstructionSignMismatchItem !== 0xffff_ffff
      && projection?.losassoBackend?.adaptiveSurfaceGraphSources) {
      const cell = adaptiveMassReceipt.firstReconstructionSignMismatchItem;
      const nodeDimensions = [dimensions[0] + 1, dimensions[1] + 1,
        dimensions[2] + 1] as const;
      const node = [cell % nodeDimensions[0],
        Math.floor(cell / nodeDimensions[0]) % nodeDimensions[1],
        Math.floor(cell / (nodeDimensions[0] * nodeDimensions[1]))] as const;
      const inspect = async (bank: "accepted" | "candidate") => {
        const graph = projection.losassoBackend!.adaptiveSurfaceGraphSources![bank];
        const control = await readControlWords(device, graph.control, 32);
        const count = Math.min(control?.[1] ?? 0, Math.floor(graph.leaves.size / 64));
        const nodeCount = Math.min(control?.[2] ?? 0, Math.floor(graph.nodes.size / 16),
          Math.floor(graph.phi.size / 8));
        const [leafBytes, massBytes, nodeBytes, phiBytes] = await Promise.all([
          readBufferBinding(device, { buffer: graph.leaves }, count * 64),
          readBufferBinding(device, { buffer: graph.surfaceMass }, count * 4),
          readBufferBinding(device, { buffer: graph.nodes }, nodeCount * 16),
          readBufferBinding(device, { buffer: graph.phi }, nodeCount * 8),
        ]);
        const leaves = new Uint32Array(leafBytes.buffer, leafBytes.byteOffset,
          leafBytes.byteLength / 4);
        const mass = new Float32Array(massBytes.buffer, massBytes.byteOffset,
          massBytes.byteLength / 4);
        const nodes = new Uint32Array(nodeBytes.buffer, nodeBytes.byteOffset,
          nodeBytes.byteLength / 4);
        const phi = new Float32Array(phiBytes.buffer, phiBytes.byteOffset,
          phiBytes.byteLength / 4);
        const incident: Array<Record<string, unknown>> = [];
        for (let leaf = 0; leaf < count; leaf += 1) {
          const base = 16 * leaf, span = leaves[base + 3] ?? 0;
          const origin = [leaves[base] ?? 0, leaves[base + 1] ?? 0,
            leaves[base + 2] ?? 0] as const;
          if (!origin.every((value, axis) => value <= node[axis]
            && node[axis] <= value + span)) continue;
          const volume = (span * shape.cellWidth) ** 3;
          incident.push({ leaf, origin, span, rho: volume > 0 ? (mass[leaf] ?? 0) / volume : 0 });
        }
        const slot = Array.from({ length: nodeCount }, (_, index) => index)
          .find((index) => nodes[4 * index] === cell);
        const rho = incident.reduce((sum, entry) => sum + Number(entry.rho), 0)
          / Math.max(incident.length, 1);
        return { rho, expectedPhi: (0.5 - rho) * shape.cellWidth,
          phi: slot === undefined ? undefined : [phi[2 * slot], phi[2 * slot + 1]],
          incident };
      };
      reconstructionMismatchNeighborhood = { node,
        accepted: await inspect("accepted"), candidate: await inspect("candidate") };
    }
    if (process.env.FLUID_REQUIRE_CONNECTED_SURFACE === "1") {
      assert.ok(massSurfaceConnectivity,
        `t=${(step * dt).toFixed(3)}: authoritative surface-mass census is absent`);
      assert.equal(massSurfaceConnectivity.componentCount, 1,
        `t=${(step * dt).toFixed(3)}: expected exactly one rho>.5 component; `
        + JSON.stringify(massSurfaceConnectivity.disconnectedComponents));
      assert.equal(massSurfaceConnectivity.disconnectedCells, 0,
        `t=${(step * dt).toFixed(3)}: authoritative rho>.5 liquid contains `
        + `${massSurfaceConnectivity.disconnectedCells} cells outside its primary `
        + `face-connected component`);
    }
    const fineTransportControl = await readControlWords(device,
      debug.globalFineTransportControl, 16);
    const fineTopologyControl = await readControlWords(device,
      fineSource?.topologyControl, 16);
    const fineRedistanceControl = await readControlWords(device,
      debug.globalFineRedistanceControl, 24);
    const transitionSources = projection?.losassoBackend;
    const topologyTransition = topologyTransitionDiagnostics ? {
      acceptedAuthority: authorityControlSummary(await readControlWords(device,
        transitionSources?.sources?.operator?.control, 8)),
      candidateAuthority: authorityControlSummary(await readControlWords(device,
        transitionSources?.candidateAuthorityControl, 8)),
      acceptedGraph: graphControlSummary(await readControlWords(device,
        transitionSources?.adaptiveSurfaceGraphSources?.accepted.control, 32)),
      candidateGraph: graphControlSummary(await readControlWords(device,
        transitionSources?.adaptiveSurfaceGraphSources?.candidate.control, 32)),
      mass: massControlSummary(await readControlWords(device,
        transitionSources?.adaptiveMassSource?.control, 32)),
      candidateVelocityStencil: await readControlWords(device,
        transitionSources?.adaptiveVelocity?.candidateStencilControl, 8),
      handoff: adaptiveMassReceipt ? {
        sourceMass_m3: adaptiveMassReceipt.handoffSourceMass_m3,
        targetMass_m3: adaptiveMassReceipt.handoffTargetMass_m3,
        signedDrift_m3: adaptiveMassReceipt.signedHandoffDrift_m3,
        leaves: adaptiveMassReceipt.handoffLeafCount,
        errors: adaptiveMassReceipt.errors,
      } : undefined,
    } : undefined;
    const extensionControl = projection?.losassoExtensionControl;
    const extensionControlBytes = extensionControl
      ? await readBufferBinding(device, { buffer: extensionControl }, extensionControl.size)
      : undefined;
    const extensionBandControl = extensionControlBytes
      ? Array.from(new Uint32Array(extensionControlBytes.buffer,
        extensionControlBytes.byteOffset, extensionControlBytes.byteLength / 4))
      : undefined;
    const extensionBandFaces = extensionBandControl?.[2];
    let extensionBandLayers: number[] | undefined;
    const extensionMetrics = projection?.losassoBackend?.extensionBand?.source?.faceMetrics;
    if (process.env.FLUID_EXTENSION_BAND_CENSUS === "1"
      && extensionMetrics && extensionBandFaces !== undefined) {
      const bytes = await readBufferBinding(device, { buffer: extensionMetrics },
        extensionMetrics.size);
      const metrics = new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
      extensionBandLayers = new Array<number>(8).fill(0);
      for (let face = 0; face < Math.min(extensionBandFaces, metrics.length / 4); face += 1) {
        const layer = metrics[4 * face + 3] ?? 0;
        if (layer < extensionBandLayers.length) extensionBandLayers[layer]! += 1;
      }
    }
    let stagedOwnerCells: number | undefined;
    let stagedOwnerRows: number | undefined;
    let stagedMacValid: number | undefined;
    let stagedRawMacValid: number | undefined;
    let stagedMacInvalid: number | undefined;
    let stagedRawBoundary: number | undefined;
    if (process.env.FLUID_STAGED_OWNER_CENSUS === "1" && stagedVelocity) {
      const stagedBytes = await readBufferBinding(device, { buffer: stagedVelocity },
        stagedVelocity.size);
      const staged = new Uint32Array(stagedBytes.buffer, stagedBytes.byteOffset,
        stagedBytes.byteLength / 4);
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
      stagedOwnerRows = new Set(staged.subarray(2 * mac, 2 * mac + nx * ny * nz)
        .filter((encoded) => encoded !== 0)).size;
    }
    // Refresh the projection receipt at this exact accepted time. Without this
    // readback, readStats() can expose the last checkpoint's cached residual
    // even though the surface directory below is current.
    await projection?.readSolveDiagnostics();
    const stats = await solver.readStats() as unknown as Record<string, unknown>;
    const massVisibleVolumeCells = massSurfaceQuadrature?.volumeCells;
    const massVisibleVolumeDeltaCells = massVisibleVolumeCells === undefined
      || previousMassVisibleVolumeCells === undefined
      ? undefined : massVisibleVolumeCells - previousMassVisibleVolumeCells;
    const massVisibleVolumeJumpFraction = massVisibleVolumeDeltaCells === undefined
      || previousMassVisibleVolumeCells === undefined || previousMassVisibleVolumeCells === 0
      ? undefined : massVisibleVolumeDeltaCells / previousMassVisibleVolumeCells;
    const massVisibleVolumeSampleInterval_s = previousMassVisibleVolumeStep === undefined
      ? undefined : (step - previousMassVisibleVolumeStep) * dt;
    const reconstructedVisibleVolumeCells = adaptiveMassReceipt
      ? adaptiveMassReceipt.reconstructionMeasuredUnits / 65536 : undefined;
    const reconstructedVisibleVolumeDeltaCells = reconstructedVisibleVolumeCells === undefined
      || previousReconstructedVisibleVolumeCells === undefined
      ? undefined : reconstructedVisibleVolumeCells - previousReconstructedVisibleVolumeCells;
    const reconstructedVisibleVolumeJumpFraction = reconstructedVisibleVolumeDeltaCells === undefined
      || previousReconstructedVisibleVolumeCells === undefined
      || previousReconstructedVisibleVolumeCells === 0
      ? undefined : reconstructedVisibleVolumeDeltaCells / previousReconstructedVisibleVolumeCells;
    const reconstructedVisibleVolumeSampleInterval_s =
      previousReconstructedVisibleVolumeStep === undefined
        ? undefined : (step - previousReconstructedVisibleVolumeStep) * dt;
    samples.push({
      t_s: Number((step * dt).toFixed(6)), ...summary,
      phiDistance: fidelity,
      leafCountsBySize: census?.leafCountsBySize,
      topologyLeaves: census?.topologyLeaves,
      topologyNodes: census?.topologyNodes,
      adaptiveNodes,
      adaptiveVelocityReceipts,
      adaptivePhiReceipt,
      adaptiveMassReceipt,
      adaptiveMassDensity,
      adaptiveVelocityProfile,
      massSurfaceConnectivity,
      massSurfaceQuadrature,
      massVisibleVolume_m3: massVisibleVolumeCells === undefined
        ? undefined : massVisibleVolumeCells * shape.cellWidth ** 3,
      massVisibleVolumeDelta_m3: massVisibleVolumeDeltaCells === undefined
        ? undefined : massVisibleVolumeDeltaCells * shape.cellWidth ** 3,
      massVisibleVolumeJumpFraction,
      massVisibleVolumeSampleInterval_s,
      massSurfaceWallProximity,
      massWetTransition,
      ceilingWetLeaves,
      ceilingTransferTrace,
      reconstructedVisibleVolume_m3: reconstructedVisibleVolumeCells === undefined
        ? undefined : reconstructedVisibleVolumeCells * shape.cellWidth ** 3,
      reconstructedVisibleVolumeDelta_m3: reconstructedVisibleVolumeDeltaCells === undefined
        ? undefined : reconstructedVisibleVolumeDeltaCells * shape.cellWidth ** 3,
      reconstructedVisibleVolumeJumpFraction,
      reconstructedVisibleVolumeSampleInterval_s,
      reconstructionMismatchNeighborhood,
      fineTransportControl: fineTransportControl
        ? Array.from(fineTransportControl) : undefined,
      fineTopologyControl: fineTopologyControl
        ? Array.from(fineTopologyControl) : undefined,
      fineRedistanceControl: fineRedistanceControl
        ? Array.from(fineRedistanceControl) : undefined,
      topologyTransition,
      candidateDiagnostics,
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
      extensionBandLayers,
      stagedOwnerCells,
      stagedOwnerRows,
      stagedMacValid,
      stagedMacInvalid,
      stagedRawMacValid,
      stagedRawBoundary,
      currentVolume: stats.currentVolume,
      referenceVolume: stats.referenceVolume,
      maximumDivergence: stats.maximumDivergence,
      maximumSpeed: stats.maximumSpeed,
    });
    if (massVisibleVolumeCells !== undefined) {
      previousMassVisibleVolumeCells = massVisibleVolumeCells;
      previousMassVisibleVolumeStep = step;
    }
    if (reconstructedVisibleVolumeCells !== undefined) {
      previousReconstructedVisibleVolumeCells = reconstructedVisibleVolumeCells;
      previousReconstructedVisibleVolumeStep = step;
    }
    if (currentAuthoritativeField) {
      previousAuthoritativeField = currentAuthoritativeField;
      previousAuthoritativeSpanField = currentAuthoritativeSpanField;
      previousAuthoritativeFieldStep = step;
    }
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
  const telemetrySamples = samples.map((sample) => ({
      t_s: sample.t_s,
      massVisibleVolume_m3: sample.massVisibleVolume_m3,
      massVisibleVolumeDelta_m3: sample.massVisibleVolumeDelta_m3,
      massVisibleVolumeJumpFraction: sample.massVisibleVolumeJumpFraction,
      reconstructedVisibleVolume_m3: sample.reconstructedVisibleVolume_m3,
      reconstructedVisibleVolumeDelta_m3: sample.reconstructedVisibleVolumeDelta_m3,
      reconstructedVisibleVolumeJumpFraction: sample.reconstructedVisibleVolumeJumpFraction,
      publishedVisibleVolume_m3: sample.zeroSetQuadrature === undefined
        ? undefined
        : (sample.zeroSetQuadrature as SurfaceShape["zeroSetQuadrature"]).volumeCells
          * dimensions.reduce(
            (volume, _dimension) => volume * (sample.cellWidth as number), 1),
      wallProximity: sample.massSurfaceWallProximity,
      compressedExcessMass_m3: (sample.adaptiveMassDensity as
        { compressionExcessMass_m3?: number } | undefined)?.compressionExcessMass_m3,
      conservedMass_m3: (sample.adaptiveMassReceipt as ReturnType<
        typeof unpackAdaptiveMassReceipt> | undefined)?.acceptedMass_m3,
      connectivity: sample.massSurfaceConnectivity,
      wetTransition: sample.massWetTransition,
      ceilingWetLeaves: sample.ceilingWetLeaves,
      ceilingTransferTrace: sample.ceilingTransferTrace,
      massErrors: (sample.adaptiveMassReceipt as ReturnType<
        typeof unpackAdaptiveMassReceipt> | undefined)?.errors,
    }));
  const maximumBy = (key: "massVisibleVolumeJumpFraction"
    | "reconstructedVisibleVolumeJumpFraction") => telemetrySamples.reduce(
    (maximum, sample) => Number(sample[key] ?? -Infinity)
      > Number(maximum?.[key] ?? -Infinity) ? sample : maximum,
    undefined as typeof telemetrySamples[number] | undefined);
  const wallVolumeSummary = volumeSummaryOnly ? {
    maximumMassVisibleJump: maximumBy("massVisibleVolumeJumpFraction"),
    maximumReconstructedVisibleJump: maximumBy("reconstructedVisibleVolumeJumpFraction"),
    firstPositiveXWallContact: telemetrySamples.find((sample) =>
      ((sample.wallProximity as { positiveXGapCells?: number } | undefined)
        ?.positiveXGapCells ?? Infinity) <= 0.125),
    keySamples: [0.552, 0.736, 0.8].map((time) => telemetrySamples.find((sample) =>
      Math.abs(Number(sample.t_s) - time) < dt / 2)),
  } : undefined;
  const reportedSamples = transactionOnly ? samples : volumeSummaryOnly ? []
    : volumeTelemetryOnly ? telemetrySamples : compact ? samples.map((sample) => {
    const quadrature = sample.zeroSetQuadrature as SurfaceShape["zeroSetQuadrature"];
    const phiReceipt = sample.adaptivePhiReceipt as ReturnType<
      typeof unpackAdaptivePhiReceipt> | undefined;
    const massReceipt = sample.adaptiveMassReceipt as ReturnType<
      typeof unpackAdaptiveMassReceipt> | undefined;
    const candidateDiagnostics = sample.candidateDiagnostics as Awaited<ReturnType<
      NonNullable<typeof projection>["readLosassoAuthorityDiagnostics"]>>;
    return {
      t_s: sample.t_s,
      wetCells: sample.wetCells,
      wettedColumns: sample.wettedColumns,
      volumeCells: quadrature.volumeCells,
      centerOfMassCells: quadrature.centerOfMassCells,
      frontCells: quadrature.frontCells,
      medianHeightCells: sample.medianHeightCells,
      maximumHeightCells: sample.maximumHeightCells,
      peakCells: sample.peakCells,
      peakColumns: sample.peakColumns,
      peakAt: sample.peakAt,
      maximumNeighborStepCells: sample.maximumNeighborStepCells,
      wetConnectivity: sample.wetConnectivity,
      profileX: sample.profileX,
      interiorRidgeCells: sample.interiorRidgeCells,
      interiorRidgeAtX: sample.interiorRidgeAtX,
      measuredVolume_m3: phiReceipt?.measuredVolume_m3,
      targetVolume_m3: phiReceipt?.targetVolume_m3,
      acceptedAdvanceValid: phiReceipt?.acceptedAdvanceValid,
      conservedMass_m3: massReceipt?.acceptedMass_m3,
      transportDrift_m3: massReceipt?.signedTransportDrift_m3,
      massDonors: massReceipt?.donors,
      massTransfers: massReceipt?.transfers,
      missingMassRecipients: massReceipt?.missingRecipients,
      handoffDrift_m3: massReceipt?.signedHandoffDrift_m3,
      massErrors: massReceipt?.errors,
      reconstructionThreshold: massReceipt?.reconstructionThreshold,
      reconstructionTargetUnits: massReceipt?.reconstructionTargetUnits,
      reconstructionMeasuredUnits: massReceipt?.reconstructionMeasuredUnits,
      reconstructionSignMismatches: massReceipt?.reconstructionSignMismatches,
      adaptiveMassDensity: sample.adaptiveMassDensity,
      adaptiveVelocityProfile: sample.adaptiveVelocityProfile,
      massSurfaceConnectivity: sample.massSurfaceConnectivity,
      massSurfaceQuadrature: sample.massSurfaceQuadrature,
      massVisibleVolume_m3: sample.massVisibleVolume_m3,
      massVisibleVolumeDelta_m3: sample.massVisibleVolumeDelta_m3,
      massVisibleVolumeJumpFraction: sample.massVisibleVolumeJumpFraction,
      massVisibleVolumeSampleInterval_s: sample.massVisibleVolumeSampleInterval_s,
      massSurfaceWallProximity: sample.massSurfaceWallProximity,
      reconstructedVisibleVolume_m3: sample.reconstructedVisibleVolume_m3,
      reconstructedVisibleVolumeDelta_m3: sample.reconstructedVisibleVolumeDelta_m3,
      reconstructedVisibleVolumeJumpFraction: sample.reconstructedVisibleVolumeJumpFraction,
      reconstructedVisibleVolumeSampleInterval_s: sample.reconstructedVisibleVolumeSampleInterval_s,
      reconstructionMismatchNeighborhood: sample.reconstructionMismatchNeighborhood,
      firstReconstructionSignMismatchNode: massReceipt
        && massReceipt.firstReconstructionSignMismatchItem !== 0xffff_ffff
        ? [massReceipt.firstReconstructionSignMismatchItem % (dimensions[0] + 1),
          Math.floor(massReceipt.firstReconstructionSignMismatchItem / (dimensions[0] + 1))
            % (dimensions[1] + 1),
          Math.floor(massReceipt.firstReconstructionSignMismatchItem
            / ((dimensions[0] + 1) * (dimensions[1] + 1)))]
        : undefined,
      leafCountsBySize: sample.leafCountsBySize,
      surfaceRowSizeHistogram: sample.surfaceRowSizeHistogram,
      airRowSizeHistogram: sample.airRowSizeHistogram,
      topologyLeaves: sample.topologyLeaves,
      residentOwnerPages: sample.residentOwnerPages,
      adaptiveVelocityReceipts: sample.adaptiveVelocityReceipts,
      topologyTransition: sample.topologyTransition,
      fineTransportControl: sample.fineTransportControl,
      fineTopologyControl: sample.fineTopologyControl,
      fineRedistanceControl: sample.fineRedistanceControl,
      candidateAuthority: candidateDiagnostics?.candidate,
      candidateGraph: candidateDiagnostics?.candidateAdaptiveGraph,
      ownerCandidate: candidateDiagnostics?.ownerCandidate,
      frontierControl: candidateDiagnostics?.frontierControl,
      massControl: candidateDiagnostics?.adaptiveMassControl,
      velocityMigration: candidateDiagnostics?.velocityMigration,
      maximumSpeed: sample.maximumSpeed,
      pressureResidual: sample.pressureResidual,
      pressureRelativeResidual: sample.pressureRelativeResidual,
    };
  }) : samples;
  console.log(JSON.stringify({
    phase: "dam-surface-shape", scene: sceneId, dt, dimensions, runtimeTopologyDials,
    refinementRegionFloor: refinementRegionFloor || undefined,
    refinementRegionCeiling: refinementRegionCeiling || undefined,
    validationErrors, wallVolumeSummary, samples: reportedSamples,
  }, null, compact ? undefined : 1));
  device.destroy();
} finally {
  await releaseWebGPUExclusiveLock();
}
