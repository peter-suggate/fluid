import assert from "node:assert/strict";
import test from "node:test";
import { buildSparseAtlasCompositeGrid } from
  "../lib/methods/adaptive-mass/sparse-atlas-composite-projection";
import { createSparseAdaptiveMassAtlas, sparseBrickKey, sparseBrickSpan,
  type SparseAdaptiveMassBrick } from
  "../lib/methods/adaptive-mass/sparse-brick-atlas";
import { compileSparseCM12FactoredAEIPackedTemplateCatalog,
  SPARSE_CM12_PACKED_TEMPLATE_HEADER } from
  "../lib/methods/adaptive-mass/sparse-cm12-factored-aei-packed-template";
import { compileSparseCM12InternedBoundaryOperators } from
  "../lib/methods/adaptive-mass/sparse-cm12-interned-boundary-operators";
import { compileSparseCM12VexIBO1Program,
  materializeSparseCM12VexIBO1DepthZero } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-consumer";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

test("IBO1 all-rung slot refs materialize exact VEX endpoint masks", () => {
  const dimensions = [16, 8, 8] as const, logical = [2, 1, 1] as const;
  const brick = (x: number): SparseAdaptiveMassBrick => ({
    key: sparseBrickKey([x, 0, 0], logical), coordinate: [x, 0, 0], resolution: 8,
    density: new Float64Array(8 ** 3), gamma: new Float64Array(8 ** 3).fill(1),
  });
  const atlas = createSparseAdaptiveMassAtlas(dimensions, [brick(0), brick(1)],
    1, undefined, 8);
  const grid = buildSparseAtlasCompositeGrid(atlas);
  const packed = packSparseCM12ResidentTopologyTemplatesForQA(atlas, grid);
  const validDimensions = (leaf: number, resolution: number) => {
    const source = atlas.bricks[leaf]!;
    const width = atlas.brickFineResolution * sparseBrickSpan(source);
    const scale = width / resolution;
    return ([0, 1, 2] as const).map((axis) => {
      const origin = atlas.brickFineResolution * source.coordinate[axis]!;
      return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
        atlas.dimensions[axis]! - origin)) / scale));
    }) as [number, number, number];
  };
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((value) => value.key),
    validDimensions: (leaf, resolution) => validDimensions(leaf, resolution),
    scaleLog2: (leaf, resolution) => Math.log2(
      atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
    selectedResolution: () => 8,
  });
  const ibo = compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves: [0, 1] });
  const program = compileSparseCM12VexIBO1Program({ catalog, ibo });
  assert.equal(catalog.canonical.every((value) => value.certified), true);
  assert.equal(program.instances.length, catalog.patches.length);
  assert.equal(program.selectedSlotRefsExact, true);

  const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
  const rowBase = packed.words[h.rowBase]!, termBase = packed.words[h.termBase]!;
  const rowTerms = (row: number) => {
    const encoded = packed.words[rowBase + row]!, first = encoded & 0x007f_ffff;
    return Array.from({ length: encoded >>> 23 }, (_, local) =>
      packed.words[termBase + 2 * (first + local)]!);
  };
  const patchRows = (patchId: number) => {
    const patch = catalog.patches[patchId]!;
    return patch.exceptionCount === 0
      ? Array.from({ length: patch.rowCount }, (_, local) => patch.rowFirst + local)
      : [...catalog.exceptionRows.subarray(patch.exceptionFirst,
        patch.exceptionFirst + patch.exceptionCount)];
  };
  const address = (cell: number, descriptor: typeof catalog.canonical[number]) => {
    const ordinal = cell - descriptor.cellFirst, [nx, ny] = descriptor.validDimensions;
    const z = Math.floor(ordinal / (nx * ny)), remain = ordinal - z * nx * ny;
    const y = Math.floor(remain / nx), x = remain - y * nx;
    const axis = Math.max(1, Math.ceil(descriptor.resolution / 4));
    return { packet: descriptor.leafId * 64 + Math.floor(x / 4) + axis
      * (Math.floor(y / 4) + axis * Math.floor(z / 4)),
    lane: (x & 3) + 4 * (y & 3) + 16 * (z & 3) };
  };

  let singletonCompared = 0, ballotsCompared = 0;
  for (const instance of program.instances) {
    const source = catalog.canonical[instance.sourceCanonicalId]!;
    const sourceEnd = source.cellFirst
      + source.validDimensions.reduce((a, b) => a * b, 1);
    const expectedBySource = new Map<number, Set<number>>();
    for (const row of patchRows(instance.patchId)) {
      const terms = rowTerms(row);
      for (const cell of terms) if (cell >= source.cellFirst && cell < sourceEnd) {
        let expected = expectedBySource.get(cell);
        if (!expected) expectedBySource.set(cell, expected = new Set());
        for (const endpoint of terms) expected.add(endpoint);
      }
    }
    const packetUnions = new Map<number, { low: number; high: number; cells: Set<number> }>();
    for (const [cell, expected] of expectedBySource) {
      singletonCompared += 1; const at = address(cell, source);
      const roots = materializeSparseCM12VexIBO1DepthZero(program, instance, at.packet,
        at.lane < 32 ? (1 << at.lane) >>> 0 : 0,
        at.lane >= 32 ? (1 << (at.lane - 32)) >>> 0 : 0, 4);
      assert.deepEqual(new Set(roots.map((root) => root.cell)), expected);
      assert.equal(roots.every((root) => root.cause === 4 && root.depth === 0), true);
      let union = packetUnions.get(at.packet);
      if (!union) packetUnions.set(at.packet,
        union = { low: 0, high: 0, cells: new Set() });
      if (at.lane < 32) union.low = (union.low | (1 << at.lane)) >>> 0;
      else union.high = (union.high | (1 << (at.lane - 32))) >>> 0;
      for (const endpoint of expected) union.cells.add(endpoint);
    }
    for (const [packet, expected] of packetUnions) {
      ballotsCompared += 1;
      const roots = materializeSparseCM12VexIBO1DepthZero(program, instance, packet,
        expected.low, expected.high, 8);
      assert.deepEqual(new Set(roots.map((root) => root.cell)), expected.cells);
      assert.equal(roots.every((root) => root.cause === 8 && root.depth === 0), true);
    }
  }
  assert.ok(singletonCompared > 0); assert.ok(ballotsCompared > 0);
});
