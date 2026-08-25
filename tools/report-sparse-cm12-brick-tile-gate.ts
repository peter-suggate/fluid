#!/usr/bin/env node
/**
 * CPU structural/service gate for BTI1.
 *
 * This is intentionally not a GPU performance claim. It compares exact work
 * and serialized bytes for dense arithmetic, the current LOD1+TEI2 access
 * path, and BTI1 before a production shader is migrated.
 */

import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  SPARSE_CM12_BRICK_TILE_IMAGE_FLAG,
  SPARSE_CM12_BRICK_TILE_IMAGE_TILE,
  SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS,
  compileSparseCM12BrickTileImage,
  sparseCM12BrickTileCell,
  sparseCM12BrickTileCellAtFine,
  sparseCM12BrickTileRows,
  validateSparseCM12BrickTileImage,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import {
  createSparseCM12LogicalOwnerDirectory,
  sparseCM12LogicalOwnerCellAtFine,
  type SparseCM12LogicalOwnerRuntime,
} from "../lib/methods/adaptive-mass/sparse-cm12-logical-owner-directory";
import {
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID,
  SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS,
  createSparseCM12TransportExecutionImage,
} from "../lib/methods/adaptive-mass/sparse-cm12-transport-execution-image";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

const args = new Map(process.argv.slice(2).map((argument) => {
  const [key, value = "true"] = argument.replace(/^--/, "").split("=", 2);
  return [key, value] as const;
}));
const lattice = (args.get("lattice") ?? "8,4,4").split(",").map(Number) as
  [number, number, number];
const samples = Math.max(3, Number(args.get("samples") ?? 9));
const repeats = Math.max(1, Number(args.get("repeats") ?? 8));
const outputPath = args.get("out");

for (const value of lattice) if (!Number.isSafeInteger(value) || value <= 0) {
  throw new RangeError("--lattice must contain three positive integers");
}

const dimensions = lattice.map((value) => 8 * value) as [number, number, number];
const bricks: SparseAdaptiveMassBrick[] = [];
for (let z = 0; z < lattice[2]; z += 1)
  for (let y = 0; y < lattice[1]; y += 1)
    for (let x = 0; x < lattice[0]; x += 1) {
      const resolution: SparseBrickResolution = (x + 2 * y + 3 * z) % 3 === 0 ? 4 : 8;
      const count = resolution ** 3;
      bricks.push({ key: sparseBrickKey([x, y, z], lattice), coordinate: [x, y, z],
        resolution, density: new Float64Array(count),
        gamma: new Float64Array(count).fill(1) });
    }
const atlas = createSparseAdaptiveMassAtlas(dimensions, bricks, 1, 8);
const grid = buildSparseAtlasCompositeGrid(atlas);
const bti = compileSparseCM12BrickTileImage(grid);
const validation = validateSparseCM12BrickTileImage(bti, grid);
const logical = createSparseCM12LogicalOwnerDirectory(atlas);
const cellsById = grid.cells;
const runtime: SparseCM12LogicalOwnerRuntime = {
  brickActive: () => true,
  acceptedBrickResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  templateBrickCellRange: (leaf) => {
    const source = atlas.bricks[leaf]!;
    const first = grid.cellBaseByBrick.get(source.key);
    if (first === undefined) throw new Error(`missing cell range for leaf ${leaf}`);
    let count = 0;
    while (first + count < cellsById.length
      && cellsById[first + count]!.brickKey === source.key) count += 1;
    return [first, count];
  },
  cellResolution: (cell) => cellsById[cell]?.brickResolution ?? 1,
  cellOpenVolume: (cell) => cellsById[cell]?.volume ?? 0,
};
const tei = createSparseCM12TransportExecutionImage(atlas, logical, runtime);

const pointCount = dimensions[0] * dimensions[1] * dimensions[2];
const denseOwners = new Uint32Array(pointCount);
denseOwners.fill(0xffff_ffff);
for (const cell of grid.cells)
  for (let z = cell.minimumFine[2]; z < cell.maximumFine[2]; z += 1)
    for (let y = cell.minimumFine[1]; y < cell.maximumFine[1]; y += 1)
      for (let x = cell.minimumFine[0]; x < cell.maximumFine[0]; x += 1) {
        denseOwners[x + dimensions[0] * (y + dimensions[1] * z)] = cell.id;
      }

let sink = 0;
const median = (values: readonly number[]) => [...values]
  .sort((a, b) => a - b)[Math.floor(values.length / 2)]!;
const measure = (operationCount: number, run: () => number) => {
  for (let warmup = 0; warmup < 3; warmup += 1) sink ^= run();
  const values: number[] = [];
  for (let sample = 0; sample < samples; sample += 1) {
    const begin = performance.now();
    let checksum = 0;
    for (let repeat = 0; repeat < repeats; repeat += 1) checksum ^= run();
    const elapsed = performance.now() - begin;
    sink ^= checksum;
    values.push(elapsed * 1e6 / (operationCount * repeats));
  }
  return { medianNanosecondsPerOperation: median(values), samples: values };
};

const denseEnumeration = () => {
  let checksum = 0;
  for (let cell = 0; cell < grid.cells.length; cell += 1) checksum = (checksum + cell) >>> 0;
  return checksum;
};
const btiEnumeration = () => {
  let checksum = 0;
  for (let tile = 0; tile < bti.layout.tileCapacity; tile += 1) {
    const at = bti.layout.tileBaseWords + tile * SPARSE_CM12_BRICK_TILE_IMAGE_TILE_WORDS;
    if ((bti.words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.flags]!
      & SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.active) === 0) continue;
    const first = bti.words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.cellFirst]!;
    const low = bti.words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.validMaskLow]!;
    const high = bti.words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.validMaskHigh]!;
    const strides = bti.words[at + SPARSE_CM12_BRICK_TILE_IMAGE_TILE.strides]!;
    for (let lane = 0; lane < 64; lane += 1) {
      const selected = lane < 32 ? ((low >>> lane) & 1) !== 0
        : ((high >>> (lane - 32)) & 1) !== 0;
      if (!selected) continue;
      const x = lane & 3, y = (lane >>> 2) & 3, z = lane >>> 4;
      checksum = (checksum + first + x + (strides & 0xffff) * y
        + (strides >>> 16) * z) >>> 0;
    }
  }
  return checksum;
};
const teiEnumeration = () => {
  let checksum = 0;
  const packetBase = tei.layout.slotPacketBaseOffsets[0];
  for (let leaf = 0; leaf < atlas.bricks.length; leaf += 1)
    for (let localPacket = 0; localPacket < 8; localPacket += 1) {
      const packet = leaf * 64 + localPacket;
      const at = packetBase + packet * SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_PACKET_WORDS;
      const first = tei.words[at + 1]!;
      if (first === SPARSE_CM12_TRANSPORT_EXECUTION_IMAGE_INVALID) continue;
      const packed = tei.words[at + 2]!, strides = tei.words[at + 3]!;
      const counts = [packed & 31, (packed >>> 5) & 31, (packed >>> 10) & 31];
      for (let lane = 0; lane < 64; lane += 1) {
        const x = lane & 3, y = (lane >>> 2) & 3, z = lane >>> 4;
        if (x >= counts[0]! || y >= counts[1]! || z >= counts[2]!) continue;
        checksum = (checksum + first + x + (strides & 0xffff) * y
          + (strides >>> 16) * z) >>> 0;
      }
    }
  return checksum;
};
const expectedEnumerationChecksum = denseEnumeration();
if (btiEnumeration() !== expectedEnumerationChecksum
  || teiEnumeration() !== expectedEnumerationChecksum) {
  throw new Error("service benchmark enumeration checksums differ");
}

const densePointOwners = () => {
  let checksum = 0;
  for (const owner of denseOwners) checksum = (checksum + owner) >>> 0;
  return checksum;
};
const btiPointOwners = () => {
  let checksum = 0;
  for (let z = 0; z < dimensions[2]; z += 1)
    for (let y = 0; y < dimensions[1]; y += 1)
      for (let x = 0; x < dimensions[0]; x += 1) {
        checksum = (checksum + (sparseCM12BrickTileCellAtFine(bti, [x, y, z])
          ?? 0xffff_ffff)) >>> 0;
      }
  return checksum;
};
const currentPointOwners = () => {
  let checksum = 0;
  for (let z = 0; z < dimensions[2]; z += 1)
    for (let y = 0; y < dimensions[1]; y += 1)
      for (let x = 0; x < dimensions[0]; x += 1) {
        checksum = (checksum + (sparseCM12LogicalOwnerCellAtFine(
          logical, [x, y, z], dimensions, runtime,
        )?.cell ?? 0xffff_ffff)) >>> 0;
      }
  return checksum;
};
const expectedPointChecksum = densePointOwners();
if (btiPointOwners() !== expectedPointChecksum
  || currentPointOwners() !== expectedPointChecksum) {
  throw new Error("service benchmark point-owner checksums differ");
}

const directFaces = () => {
  let checksum = 0;
  for (const row of grid.gradientRows) checksum = (checksum + row.id) >>> 0;
  return checksum;
};
const btiFaces = () => {
  let checksum = 0;
  for (let tile = 0; tile < bti.layout.tileCapacity; tile += 1)
    for (let family = 0; family < 6; family += 1)
      for (let lane = 0; lane < 64; lane += 1)
        for (const row of sparseCM12BrickTileRows(bti, tile, family, lane)) {
          checksum = (checksum + row) >>> 0;
        }
  return checksum;
};
if (btiFaces() !== directFaces()) throw new Error("service benchmark face checksums differ");

const report = {
  schema: "sparse-cm12-brick-tile-gate/v1",
  generatedAt: new Date().toISOString(),
  status: "structural-pass-gpu-service-benchmark-required",
  warning: "CPU nanoseconds are implementation diagnostics, not GPU migration gates.",
  fixture: { lattice, dimensions, leafCount: atlas.bricks.length,
    fineLeafCount: atlas.bricks.filter((value) => value.resolution === 8).length,
    coarseLeafCount: atlas.bricks.filter((value) => value.resolution === 4).length,
    cellCount: grid.cells.length, rowCount: grid.gradientRows.length,
    mixedSeamRowCount: grid.mixedSeamRowCount },
  validation,
  serializedMemory: {
    denseOwnerOracleBytes: denseOwners.byteLength,
    currentLogicalOwnerBytes: logical.layout.totalBytes,
    currentTEI2Bytes: tei.layout.totalBytes,
    currentLOD1PlusTEI2Bytes: logical.layout.totalBytes + tei.layout.totalBytes,
    bti1Bytes: bti.layout.totalBytes,
    comparability: "BTI1 includes face-address masks/exceptions; LOD1+TEI2 does not include IBO/ITR. Do not use this partial byte comparison as a migration gate.",
  },
  services: {
    enumerateCells: {
      operationCount: grid.cells.length,
      denseArithmetic: measure(grid.cells.length, denseEnumeration),
      currentTEI2: measure(grid.cells.length, teiEnumeration),
      bti1: measure(grid.cells.length, btiEnumeration),
    },
    pointOwner: {
      operationCount: pointCount,
      denseDirect: measure(pointCount, densePointOwners),
      currentLOD1: measure(pointCount, currentPointOwners),
      bti1: measure(pointCount, btiPointOwners),
    },
    enumerateFaces: {
      operationCount: grid.gradientRows.length,
      denseDirect: measure(grid.gradientRows.length, directFaces),
      bti1ImplicitInteriorExplicitSeam: measure(grid.gradientRows.length, btiFaces),
      currentITR1: "requires matched GPU shader microbenchmark",
    },
  },
  checksumSink: sink >>> 0,
};

const json = `${JSON.stringify(report, null, 2)}\n`;
if (outputPath) await writeFile(resolve(outputPath), json);
process.stdout.write(json);
