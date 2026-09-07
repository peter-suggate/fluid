import assert from "node:assert/strict";
import test from "node:test";
import { createSparseAdaptiveMassAtlas, sparseBrickKey, type SparseBrickResolution } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { compileSparseCM12GenerationTransfer } from
  "../lib/methods/adaptive-mass/sparse-cm12-generation-transfer";

function grid(dimensions: readonly [number, number, number], bricks:
  readonly { q: readonly [number, number, number]; span: number; r: SparseBrickResolution }[]) {
  return buildSparseAtlasCompositeGrid(createSparseAdaptiveMassAtlas(dimensions,
    bricks.map(({ q, span, r }) => ({ key: sparseBrickKey(q,
      dimensions.map((n) => Math.ceil(n / 8)) as [number, number, number]),
    coordinate: q, spanBricks: span, resolution: r,
    density: new Float64Array(r ** 3).fill(0.5), gamma: new Float64Array(r ** 3).fill(1) })), 0, 8));
}

test("only explicit new-air allocation can introduce uncovered target volume", () => {
  const source = grid([16,8,8], [{q:[0,0,0],span:1,r:1}]);
  const target = grid([16,8,8], [{q:[0,0,0],span:1,r:1}, {q:[1,0,0],span:1,r:1}]);
  assert.throws(() => compileSparseCM12GenerationTransfer(source,target), /complete source coverage/);
  assert.throws(() => compileSparseCM12GenerationTransfer(source,target,[{
    minimumFine:[8,0,0],maximumExclusiveFine:[12,8,8],
  }]), /complete source coverage/);
  const plan = compileSparseCM12GenerationTransfer(source,target,[{
    minimumFine:[8,0,0],maximumExclusiveFine:[16,8,8],
  }]);
  assert.deepEqual([...plan.cellSources],[0,0xffffffff]);
  assert.deepEqual([...plan.cellVolumes],[512,512]);
  assert.deepEqual([...plan.cellOffsets],[0,1,2]);
});

test("generation transfer preserves coverage and boundary flux through clipped macro split/merge", () => {
  const macro = grid([13, 15, 11], [{ q: [0, 0, 0], span: 2, r: 2 }]);
  const children = grid([13, 15, 11], Array.from({ length: 8 }, (_, i) => ({
    q: [i & 1, (i >>> 1) & 1, i >>> 2] as const, span: 1, r: 4 as const,
  })));
  for (const [source, target] of [[macro, children], [children, macro]]) {
    const plan = compileSparseCM12GenerationTransfer(source!, target!);
    const sourceVolumes = new Float64Array(source!.cells.length);
    for (let cell = 0; cell < target!.cells.length; cell++) {
      let volume = 0;
      for (let at = plan.cellOffsets[cell]!; at < plan.cellOffsets[cell + 1]!; at++) {
        sourceVolumes[plan.cellSources[at]!] += plan.cellVolumes[at]!;
        volume += plan.cellVolumes[at]!;
      }
      assert.equal(volume, target!.cells[cell]!.volume);
    }
    assert.deepEqual(Array.from(sourceVolumes), source!.cells.map((cell) => cell.volume));
    const transferredArea = new Float64Array(source!.gradientRows.length);
    for (const row of target!.gradientRows) {
      let area = 0;
      for (let at = plan.faceOffsets[row.id]!; at < plan.faceOffsets[row.id + 1]!; at++) {
        const before = source!.gradientRows[plan.faceSources[at]!]!;
        assert.equal(before.axis, row.axis);
        assert.equal(before.centerFine[row.axis], row.centerFine[row.axis]);
        transferredArea[before.id] += plan.faceAreas[at]!;
        area += plan.faceAreas[at]!;
      }
      if (row.kind === "sparse-air") assert.equal(area, row.areaFineCells2);
      assert.ok(area <= row.areaFineCells2);
    }
    for (const row of source!.gradientRows) {
      if (row.kind === "sparse-air") assert.equal(transferredArea[row.id], row.areaFineCells2);
    }
  }
});

test("identity generation transfer maps every cell and face once", () => {
  const source = grid([32, 16, 16], [{ q: [0, 0, 0], span: 2, r: 2 },
    ...Array.from({ length: 4 }, (_, i) => ({ q: [2, i % 2, i >>> 1] as const,
      span: 1, r: 2 as const }))]);
  const plan = compileSparseCM12GenerationTransfer(source, source);
  assert.deepEqual(Array.from(plan.cellSources), source.cells.map((cell) => cell.id));
  assert.deepEqual(Array.from(plan.faceSources), source.gradientRows.map((row) => row.id));
  assert.deepEqual(Array.from(plan.faceAreas), source.gradientRows.map((row) => row.areaFineCells2));
});

test("partially occupied macro faces retain every uncovered air patch", () => {
  const source = grid([40,32,32], [{q:[0,0,0],span:4,r:1}, {q:[4,0,0],span:1,r:1}]);
  const face = source.gradientRows.filter(row => row.axis === 0 && row.centerFine[0] === 32);
  assert.equal(face.reduce((sum,row)=>sum+row.areaFineCells2,0),32*32);
  assert.equal(face.filter(row=>row.kind==="sparse-air").reduce((sum,row)=>sum+row.areaFineCells2,0),32*32-8*8);
  const plan = compileSparseCM12GenerationTransfer(source,source);
  assert.deepEqual(Array.from(plan.faceSources),source.gradientRows.map(row=>row.id));
});

test("signed generation geometry preserves full pages beyond both authored boundaries", async () => {
  const { sparseAtlasBrickKey, sparseBrickContainingCoordinate } = await import("../lib/methods/adaptive-mass/sparse-brick-atlas");
  const bricks = [-1,0,1].map(x => {
    const coordinate = [x,0,0] as const;
    return { key:sparseAtlasBrickKey(coordinate,{brickDimensions:[1,1,1],signedCoordinates:true}),
      coordinate,unclipped:x!==0,resolution:2 as const,density:new Float64Array(8).fill(1),gamma:new Float64Array(8).fill(1) };
  });
  const atlas = createSparseAdaptiveMassAtlas([8,8,8],bricks,1,8,true);
  assert.equal(sparseBrickContainingCoordinate(atlas,[-1,0,0])?.key,bricks[0]!.key);
  const source = buildSparseAtlasCompositeGrid(atlas);
  assert.equal(source.cells.reduce((sum,cell)=>sum+cell.volume,0),3*512);
  const plan = compileSparseCM12GenerationTransfer(source,source);
  assert.deepEqual(Array.from(plan.cellSources),source.cells.map(cell=>cell.id));
  assert.deepEqual(Array.from(plan.faceSources),source.gradientRows.map(row=>row.id));
});
