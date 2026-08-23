import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12BrickTileImage } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-image";
import {
  SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG,
  SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER,
  compileSparseCM12BrickTileFaceProgram,
  sparseCM12BrickTileFaceProgramSeamPort,
  validateSparseCM12BrickTileFaceProgram,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program";
import { createSparseCM12BrickTileFaceProgramWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-program.wgsl";
import {
  compileSparseCM12BrickTileFaceAddressProgram,
  validateSparseCM12BrickTileFaceAddressCoverage,
} from "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-address-program";
import { createSparseCM12BrickTileFaceAddressWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-brick-tile-face-address-program.wgsl";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";
import {
  createSparseAdaptiveMassAtlas,
  sparseBrickKey,
  type SparseAdaptiveMassBrick,
  type SparseBrickResolution,
} from "../lib/methods/adaptive-mass/sparse-brick-atlas";

const sourceBrick = (
  x: number,
  dimensions: readonly [number, number, number],
  resolution: SparseBrickResolution,
): SparseAdaptiveMassBrick => ({
  key: sparseBrickKey([x, 0, 0], dimensions), coordinate: [x, 0, 0], resolution,
  density: new Float64Array(resolution ** 3),
  gamma: new Float64Array(resolution ** 3).fill(1),
});

test("BFP1 partitions a mixed-rung topology into interior tiles and seam ports", () => {
  const dimensions = [2, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    sourceBrick(0, dimensions, 8), sourceBrick(1, dimensions, 4),
  ], 4, undefined, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const program = compileSparseCM12BrickTileFaceProgram(image, grid);
  const receipt = validateSparseCM12BrickTileFaceProgram(program, image, grid);

  assert.equal(receipt.interiorRowCount + receipt.seamRowCount,
    grid.gradientRows.length);
  assert.equal(receipt.seamRowCount,
    grid.gradientRows.filter((row) => row.kind !== "intra-brick").length);
  assert.equal(receipt.interiorTileCount, 9);
  assert.ok(receipt.invokedLaneCount < image.layout.tileCapacity * 6 * 64 / 4);
  assert.ok(receipt.rowsPerInvokedLane > 2);
  assert.equal(program.words[SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_HEADER.flags],
    SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG.complete
      | SPARSE_CM12_BRICK_TILE_FACE_PROGRAM_FLAG.validated);
});

test("BFP1 seam stream retains equal-rung and sparse-air ownership exactly", () => {
  const dimensions = [3, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([24, 8, 8], [
    sourceBrick(0, dimensions, 8), sourceBrick(2, dimensions, 8),
  ], 2, undefined, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const program = compileSparseCM12BrickTileFaceProgram(image, grid);
  const rows = new Set<number>();
  for (let ordinal = 0; ordinal < program.layout.seamPortCount; ordinal += 1) {
    const port = sparseCM12BrickTileFaceProgramSeamPort(program, ordinal)!;
    rows.add(port.row);
    assert.equal(grid.gradientRows[port.row]!.kind, "sparse-air");
    assert.equal(port.family % 3, grid.gradientRows[port.row]!.axis);
  }
  assert.equal(rows.size, grid.sparseAirRowCount);
});

test("BFP1 publishes compact WGSL execution accessors", () => {
  const dimensions = [1, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([8, 8, 8], [
    sourceBrick(0, dimensions, 8),
  ], 1, undefined, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const program = compileSparseCM12BrickTileFaceProgram(image, grid);
  const source = createSparseCM12BrickTileFaceProgramWGSL({ layout: program.layout,
    arenaName: "faceProgram" });
  assert.match(source, /fn bfp1InteriorTile\(/);
  assert.match(source, /fn bfp1SeamPort\(/);
  assert.match(source, /faceProgram\[at\]/);
});

test("BFA1 all-rung addresses cover the concrete accepted face program", () => {
  const dimensions = [2, 1, 1] as const;
  const atlas = createSparseAdaptiveMassAtlas([16, 8, 8], [
    sourceBrick(0, dimensions, 8), sourceBrick(1, dimensions, 4),
  ], 1, undefined, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const image = compileSparseCM12BrickTileImage(grid);
  const accepted = compileSparseCM12BrickTileFaceProgram(image, grid);
  const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((value) => value.key),
    validDimensions: (_leaf, resolution) => [resolution, resolution, resolution],
    scaleLog2: (_leaf, resolution) => Math.log2(8 / resolution),
    selectedResolution: (leaf) => atlas.bricks[leaf]!.resolution,
  });
  const ibo = compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves: [0, 1] });
  const universal = compileSparseCM12BrickTileFaceAddressProgram({ ibo });
  const coverage = validateSparseCM12BrickTileFaceAddressCoverage(universal, accepted);

  assert.equal(coverage.interiorCovered, accepted.layout.interiorTileCount);
  assert.equal(coverage.seamCovered, accepted.layout.seamPortCount);
  assert.ok(universal.layout.interiorTileCount >= accepted.layout.interiorTileCount);
  assert.ok(universal.layout.seamAddressCount >= accepted.layout.seamPortCount);
  const source = createSparseCM12BrickTileFaceAddressWGSL({ layout: universal.layout });
  assert.match(source, /fn prepareSparseCM12InteriorFaceTiles/);
  assert.match(source, /fn prepareSparseCM12SeamFacePackets/);
  assert.match(source, /fn bfa1Project\(row:u32\).*rowAccepted\(row\)/s);
  assert.doesNotMatch(source, /DFRM|dfrm1|bfa1Selected/);
});
