import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import {
  SPARSE_CM12_BRICK_TILE_IMAGE_FLAG,
  SPARSE_CM12_BRICK_TILE_IMAGE_HEADER,
  SPARSE_CM12_BRICK_TILE_IMAGE_INVALID,
  compileSparseCM12BrickTileImage,
  sparseCM12BrickTileCell,
  sparseCM12BrickTileCellAtFine,
  sparseCM12BrickTileRows,
  validateSparseCM12BrickTileImage,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import { createSparseCM12BrickTileImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image.wgsl";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

const brick = (
  coordinate: readonly [number, number, number],
  brickDimensions: readonly [number, number, number],
  resolution: SparseBrickResolution,
  spanBricks = 1,
): SparseAdaptiveMassBrick => ({
  key: sparseBrickKey(coordinate, brickDimensions), coordinate,
  ...(spanBricks === 1 ? {} : { spanBricks }), resolution,
  density: new Float64Array(resolution ** 3),
  gamma: new Float64Array(resolution ** 3).fill(1),
});

test("BTI1 exactly enumerates equal-rung B8 cells, sparse-air faces, and point owners", () => {
  const brickDimensions = [2, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick([0, 0, 0], brickDimensions, 8),
    brick([1, 0, 0], brickDimensions, 8),
  ], 7, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const receipt = validateSparseCM12BrickTileImage(image, grid);

  assert.equal(receipt.cellCount, 1024);
  assert.equal(receipt.activeTileCount, 16);
  // 64 rows join the two leaves. The other 640 are their exposed voxel faces;
  // an authored domain edge is sparse air, not an implicit enclosing wall.
  assert.equal(receipt.explicitFaceRowCount, 704);
  assert.equal(receipt.explicitAddressCollisionCount, 0);
  assert.equal(receipt.implicitInteriorRowCount + receipt.explicitFaceRowCount,
    grid.gradientRows.length);
  assert.equal(image.words[SPARSE_CM12_BRICK_TILE_IMAGE_HEADER.flags],
    SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.complete
      | SPARSE_CM12_BRICK_TILE_IMAGE_FLAG.validated);
  assert.equal(sparseCM12BrickTileCell(image, 0, 0), 0);
  assert.equal(sparseCM12BrickTileCell(image, 0, 63), 3 + 8 * (3 + 8 * 3));
  assert.equal(sparseCM12BrickTileCellAtFine(image, [0, 0, 0]), 0);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [15, 7, 7]), 1023);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [16, 0, 0]), undefined);
});

test("BTI1 keeps a mixed 8:4 seam explicit while interiors remain arithmetic", () => {
  const brickDimensions = [2, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    brick([0, 0, 0], brickDimensions, 8),
    brick([1, 0, 0], brickDimensions, 4),
  ], 3, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const receipt = validateSparseCM12BrickTileImage(image, grid);

  assert.equal(receipt.cellCount, 8 ** 3 + 4 ** 3);
  assert.equal(receipt.activeTileCount, 9);
  assert.ok(receipt.explicitFaceRowCount > 0);
  assert.ok(receipt.memory.bytesPerCell < 20,
    `BTI1 proof image unexpectedly costs ${receipt.memory.bytesPerCell} bytes/cell`);

  const seamRows = new Set<number>();
  for (let tile = 0; tile < image.layout.tileCapacity; tile += 1)
    for (let lane = 0; lane < 64; lane += 1)
      for (const row of sparseCM12BrickTileRows(image, tile, 0, lane)) {
        if (grid.gradientRows[row]!.kind === "mixed-seam") seamRows.add(row);
      }
  assert.equal(seamRows.size, grid.mixedSeamRowCount);
});

test("BTI1 point directory covers a macro leaf without expanding cell topology", () => {
  const brickDimensions = [2, 2, 2] as const;
  const atlas = createSparseAdaptiveMassAtlas([16, 16, 16], [
    brick([0, 0, 0], brickDimensions, 8, 2),
  ], 11, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const receipt = validateSparseCM12BrickTileImage(image, grid);

  assert.equal(receipt.cellCount, 512);
  assert.equal(receipt.activeTileCount, 8);
  assert.equal(image.layout.spatialTileCapacity, 64);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [15, 15, 15]), 511);
});

test("BTI1 handles clipped edge bricks and leaves omitted spatial tiles invalid", () => {
  const brickDimensions = [2, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([14, 8, 8], [
    brick([0, 0, 0], brickDimensions, 8),
    brick([1, 0, 0], brickDimensions, 8),
  ], 5, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const receipt = validateSparseCM12BrickTileImage(image, grid);

  assert.equal(receipt.cellCount, 8 ** 3 + 6 * 8 * 8);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [13, 7, 7]),
    grid.cells.length - 1);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [14, 7, 7]), undefined);
});

test("BTI1 sparse-air face families cover omitted in-domain bricks exactly", () => {
  const brickDimensions = [3, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([24, 8, 8], [
    brick([0, 0, 0], brickDimensions, 8),
    brick([2, 0, 0], brickDimensions, 8),
  ], 2, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const receipt = validateSparseCM12BrickTileImage(image, grid);

  assert.equal(receipt.explicitFaceRowCount, grid.sparseAirRowCount);
  assert.ok(grid.sparseAirRowCount > 0);
  assert.equal(sparseCM12BrickTileCellAtFine(image, [12, 4, 4]), undefined);
  const middleTile = image.layout.spatialOwnerBaseWords + 3;
  assert.equal(image.words[middleTile], SPARSE_CM12_BRICK_TILE_IMAGE_INVALID);
});

test("BTI1 publishes one composable WGSL service ABI", () => {
  const brickDimensions = [1, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    brick([0, 0, 0], brickDimensions, 8),
  ], 1, 8);
  const image = compileSparseCM12BrickTileImage(buildSparseAtlasCompositeGrid(atlas));
  const source = createSparseCM12BrickTileImageWGSL({ layout: image.layout,
    arenaName: "acceptedTopology" });
  for (const service of ["bti1Cell", "bti1PointOwner", "bti1FaceRowCount",
    "bti1FaceRow"] as const) assert.match(source, new RegExp(`fn ${service}\\(`));
  assert.match(source, /acceptedTopology\[at\]/);
  assert.throws(() => createSparseCM12BrickTileImageWGSL({ layout: image.layout,
    arenaName: "invalid-name" }), /identifier/);
});
