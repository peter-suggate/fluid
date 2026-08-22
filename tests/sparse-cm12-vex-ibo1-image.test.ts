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
import { applySparseCM12VexIBO1Instance, compileSparseCM12VexIBO1Program } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-consumer";
import { applySparseCM12VexIBO1ImageSelected,
  applySparseCM12VexIBO1ImageTemplate, compileSparseCM12VexIBO1Image,
} from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image";
import { createSparseCM12VexIBO1ImageWGSL } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-ibo1-image.wgsl";
import { executeSparseCM12VexDirectPacketRecurrence,
  executeSparseCM12VexPacketFrontierOracle } from
  "../lib/methods/adaptive-mass/sparse-cm12-vex-packet-frontier";
import { packSparseCM12ResidentTopologyTemplatesForQA } from
  "../lib/methods/adaptive-mass/webgpu-sparse-cm12-resident";

const fixture = () => {
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
    const source = atlas.bricks[leaf]!, width = atlas.brickFineResolution
      * sparseBrickSpan(source), scale = width / resolution;
    return ([0, 1, 2] as const).map((axis) => {
      const origin = atlas.brickFineResolution * source.coordinate[axis]!;
      return Math.min(resolution, Math.ceil(Math.max(0, Math.min(width,
        atlas.dimensions[axis]! - origin)) / scale));
    }) as [number, number, number];
  };
  const catalog = compileSparseCM12FactoredAEIPackedTemplateCatalog({
    words: packed.words, brickFineResolution: 8,
    brickKeyByLeafId: atlas.bricks.map((value) => value.key),
    validDimensions, scaleLog2: (leaf, resolution) => Math.log2(
      atlas.brickFineResolution * sparseBrickSpan(atlas.bricks[leaf]!) / resolution),
    selectedResolution: () => 8,
  });
  const ibo = compileSparseCM12InternedBoundaryOperators({ catalog,
    packedWords: packed.words, activeLeaves: [0, 1] });
  const program = compileSparseCM12VexIBO1Program({ catalog, ibo });
  return { packed, catalog, ibo, program,
    image: compileSparseCM12VexIBO1Image(program) };
};

const setLane = (mask: [number, number], lane: number) => {
  mask[lane >>> 5] = (mask[lane >>> 5]! | (1 << (lane & 31))) >>> 0;
};

test("VXI1 packed recipes equal exact selected depth-zero closure and VXP recurrence", () => {
  const { packed, catalog, ibo, program, image } = fixture();
  assert.ok(image.layout.coreTemplateCount > 0);
  assert.equal(image.layout.faceTemplateCount, ibo.templates.length);
  assert.equal(image.words.byteLength, image.layout.totalBytes);
  // Every certified all-rung face recipe survives physical packing exactly,
  // independently of which references happen to be selected in this slot.
  assert.equal(catalog.canonical.every((descriptor) => descriptor.certified), true);
  for (const instance of program.instances) {
    for (const [localPacket, packetProgram] of
      program.templates[instance.templateId]!.packetPrograms) {
      let low = 0, high = 0;
      for (const operation of packetProgram.operations) {
        low = (low | operation.sourceLow) >>> 0;
        high = (high | operation.sourceHigh) >>> 0;
      }
      const stablePacket = 64 * instance.sourceLeaf + localPacket;
      const expected = applySparseCM12VexIBO1Instance(program, instance,
        stablePacket, low, high);
      const actual = applySparseCM12VexIBO1ImageTemplate({ image,
        templateId: image.faceTemplateIds[instance.templateId]!,
        sourceLeaf: instance.sourceLeaf, targetLeaf: instance.targetLeaf,
        sourceStablePacket: stablePacket, sourceLow: low, sourceHigh: high });
      assert.deepEqual(actual, expected);
    }
  }
  const addressByCell = new Map<number, { packetId: number; lane: number }>();
  const cellByAddress = new Map<string, number>();
  const selectedDescriptors = catalog.descriptorIdByLeaf.map((id) => catalog.canonical[id]!);
  for (const descriptor of selectedDescriptors) {
    const [nx, ny, nz] = descriptor.validDimensions;
    const axis = Math.max(1, Math.ceil(descriptor.resolution / 4));
    for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const cell = descriptor.cellFirst + x + nx * (y + ny * z);
        const packetId = 64 * descriptor.leafId + Math.floor(x / 4)
          + axis * (Math.floor(y / 4) + axis * Math.floor(z / 4));
        const lane = (x & 3) + 4 * (y & 3) + 16 * (z & 3);
        addressByCell.set(cell, { packetId, lane });
        cellByAddress.set(`${packetId}/${lane}`, cell);
      }
    }
  }
  const expectedByCell = new Map<number, Set<number>>([...addressByCell]
    .map(([cell]) => [cell, new Set([cell])]));
  const h = SPARSE_CM12_PACKED_TEMPLATE_HEADER;
  const rowBase = packed.words[h.rowBase]!, termBase = packed.words[h.termBase]!;
  for (const row of ibo.expectedStableRows) {
    const encoded = packed.words[rowBase + row]!, first = encoded & 0x007f_ffff;
    const terms = Array.from({ length: encoded >>> 23 }, (_, local) =>
      packed.words[termBase + 2 * (first + local)]!);
    for (const source of terms) {
      const expected = expectedByCell.get(source);
      if (expected) for (const target of terms) expected.add(target);
    }
  }
  const cellsFromMasks = (masks: ReadonlyMap<number, readonly [number, number]>) => {
    const result = new Set<number>();
    for (const [packetId, mask] of masks) for (let lane = 0; lane < 64; lane += 1) {
      if (((mask[lane >>> 5]! >>> (lane & 31)) & 1) === 0) continue;
      const cell = cellByAddress.get(`${packetId}/${lane}`);
      assert.notEqual(cell, undefined, `invalid VXI1 lane ${packetId}/${lane}`);
      result.add(cell!);
    }
    return result;
  };
  const ballots = new Map<number, { mask: [number, number]; expected: Set<number> }>();
  let singletonCompared = 0;
  for (const [cell, expected] of expectedByCell) {
    const address = addressByCell.get(cell)!, low = address.lane < 32
      ? (1 << address.lane) >>> 0 : 0, high = address.lane >= 32
      ? (1 << (address.lane - 32)) >>> 0 : 0;
    assert.deepEqual(cellsFromMasks(applySparseCM12VexIBO1ImageSelected({ image,
      sourceStablePacket: address.packetId, sourceLow: low, sourceHigh: high })), expected);
    singletonCompared += 1;
    if (((cell + address.lane) & 1) !== 0) continue;
    let ballot = ballots.get(address.packetId);
    if (!ballot) ballots.set(address.packetId,
      ballot = { mask: [0, 0], expected: new Set() });
    setLane(ballot.mask, address.lane); for (const target of expected) ballot.expected.add(target);
  }
  assert.ok(singletonCompared >= 1024);
  for (const [sourceStablePacket, ballot] of ballots) {
    assert.deepEqual(cellsFromMasks(applySparseCM12VexIBO1ImageSelected({ image,
      sourceStablePacket, sourceLow: ballot.mask[0], sourceHigh: ballot.mask[1] })),
    ballot.expected);
  }

  // The compiled depth-zero batch feeds the packet recurrence directly. Root
  // cause/stamp/depth-zero and all eight frontier depths match the cell oracle.
  const firstCell = [...expectedByCell.keys()][77]!, source = addressByCell.get(firstCell)!;
  const sourceMask: [number, number] = [0, 0]; setLane(sourceMask, source.lane);
  const roots = applySparseCM12VexIBO1ImageSelected({ image,
    sourceStablePacket: source.packetId, sourceLow: sourceMask[0], sourceHigh: sourceMask[1] });
  const batch = Object.freeze({ generation: 9, topologyGeneration: 3,
    topologySlot: 0 as const, cause: 4, packetMasks: roots });
  const packetCell = (packetId: number, lane: number) =>
    cellByAddress.get(`${packetId}/${lane}`);
  const cellActive = (cell: number) => addressByCell.has(cell);
  const cellOracle = executeSparseCM12VexPacketFrontierOracle({
    cellCapacity: packed.cellCount, packetCapacity: catalog.layout.leafCapacity * 64,
    batches: [batch], packetCell, cellAddress: (cell) => addressByCell.get(cell),
    cellActive, neighbors: (cell) => expectedByCell.get(cell) ?? [], maximumDepth: 8,
  });
  const packetOracle = executeSparseCM12VexDirectPacketRecurrence({
    cellCapacity: packed.cellCount, packetCapacity: catalog.layout.leafCapacity * 64,
    batches: [batch], packetCell, cellActive,
    expandPacketMask: (sourceStablePacket, sourceLow, sourceHigh) =>
      applySparseCM12VexIBO1ImageSelected({ image, sourceStablePacket,
        sourceLow, sourceHigh }), maximumDepth: 8,
  });
  assert.deepEqual(packetOracle.rootStamp, cellOracle.rootStamp);
  assert.deepEqual(packetOracle.rootCause, cellOracle.rootCause);
  assert.deepEqual(packetOracle.blastStamp, cellOracle.blastStamp);
  assert.deepEqual(packetOracle.blastDepth, cellOracle.blastDepth);
  assert.equal([...roots].every(([packetId, mask]) => {
    for (let lane = 0; lane < 64; lane += 1) if (((mask[lane >>> 5]!
      >>> (lane & 31)) & 1) !== 0) {
      const cell = packetCell(packetId, lane)!;
      if (packetOracle.rootStamp[cell] !== 9 || packetOracle.rootCause[cell] !== 4
        || packetOracle.blastDepth[cell] !== 0) return false;
    }
    return true;
  }), true);
});

test("VXI1 WGSL is a bounded packet operator with no semantic fallback", () => {
  const { image } = fixture();
  const source = createSparseCM12VexIBO1ImageWGSL({ layout: image.layout,
    baseWords: 256, iboPrefix: "ibo1", sourcePacketCountFunction: "rootCount",
    sourcePacketFunction: "rootPacket", sourceMaskFunction: "rootMask",
    rootCauseExpression: "4u" });
  assert.match(source, /vxp1RecordRootPacketMask/);
  assert.match(source, /vxp1MergeFrontierTarget/);
  assert.match(source, /ibo1IBOFaceRefCount/);
  assert.match(source, /ibo1IBORef/);
  assert.match(source, /refCount>4u/);
  for (const forbidden of ["RecordRoot(cell", "incidence", "ownerCellAt",
    "globalCell", "worldScan", "rowTerm"]) assert.equal(source.includes(forbidden), false);
});
