import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { codeOnlyWGSL } from "../lib/wgsl-binding-reachability";
import {
  OCTREE_SPGRID_VCYCLE_BINDINGS,
  OCTREE_SPGRID_CAPTURE_CONTROL_WORD,
  SPGRID_CELL_FLAG,
  SPGRID_MAXIMUM_ROW_CAPACITY,
  WebGPUOctreeSPGridVCycle,
  buildSPGridBrickRankDirectory,
  buildSPGridPhysicalPageAdjacency,
  buildSPGridParentGatherCSR,
  buildSPGridPyramidOracle,
  computeSPGridScaledSpectralBounds,
  decodeOctreeSPGridHierarchyCensus,
  octreeSPGridAccurateDispatchGateShader,
  octreeSPGridAccurateOperatorShader,
  octreeSPGridVCycleShader,
  planOctreeSPGridVCycle,
  spgridRowCapacityForBindingLimit,
  lookupSPGridBrickRank,
  prolongSPGrid,
  restrictSPGrid,
  restrictSPGridParentGather,
  solveSPGridBottomLDLT,
  spgridChebyshevRelaxationWeight,
  validateSPGridBrickRankDirectory,
} from "../lib/webgpu-octree-spgrid-vcycle";
import { PassBroker } from "../lib/webgpu-pass-broker";
import { planResolvedPowerRows } from "../lib/webgpu-octree-power-resolved-rows";

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const spgridDelta = (
  buffer: (size: number, usage?: number) => GPUBuffer,
  rowCapacity: number,
) => {
  const controlOffsetWords = 0;
  const newToOldOffsetWords = 16;
  const oldToNewOffsetWords = newToOldOffsetWords + rowCapacity;
  const dirtyRowsOffsetWords = oldToNewOffsetWords + rowCapacity;
  return {
    rows: buffer(4 * (dirtyRowsOffsetWords + rowCapacity)),
    rowCapacity,
    controlOffsetWords,
    newToOldOffsetWords,
    oldToNewOffsetWords,
    dirtyRowsOffsetWords,
  };
};
const spgridSource = (
  buffer: (size: number, usage?: number) => GPUBuffer,
  rowCapacity: number,
  entryBytes: number,
) => {
  const plan = planResolvedPowerRows(rowCapacity, 8);
  const params = buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
  const rows = buffer(2 * plan.arenaBytes, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
  const diagonals = buffer(rowCapacity * 8,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const coefficients = buffer(2 * rowCapacity * 19 * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const rowGeometry = buffer(rowCapacity * 32,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);
  const worksetBuffer = buffer(Math.max(32, entryBytes), GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT);
  const binding = { buffer: worksetBuffer, offset: 0, size: worksetBuffer.size };
  return {
    rowCapacity, plan, params, rows, diagonals, coefficients,
    coefficientBankStrideWords: rowCapacity * 19, rowGeometry,
    topologyMetrics: buffer(rowCapacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    catalogCoefficients: buffer(1_608 * 19 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    worksets: { regularInterior: binding, transitionInterior: binding,
      physicalBoundary: binding, transitionBoundary: binding },
    invalidRows: binding,
    control: buffer(64, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    rowDelta: spgridDelta(buffer, rowCapacity),
    classDispatch: buffer(108, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT),
    classDispatchOffsetBytes: 24,
  };
};

/** Executable model of scanCandidateTransfers: a Hillis-Steele inclusive block
 * scan of the per-fine fan-out with a lane-0 carry, read back as the exclusive
 * prefix. Integer addition is associative, so this must equal the serial running
 * counter the transfer arena is defined by. */
function blockScannedRecordBases(fanOut: readonly number[], chunk: number): { bases: number[]; total: number } {
  const bases = new Array<number>(fanOut.length).fill(0);
  let carry = 0;
  for (let block = 0; block < fanOut.length; block += chunk) {
    const owned = Array.from({ length: chunk }, (_, lane) => fanOut[block + lane] ?? 0);
    const inclusive = owned.slice();
    for (let span = 1; span < chunk; span <<= 1) {
      const staged = inclusive.slice();
      for (let lane = span; lane < chunk; lane += 1) inclusive[lane] = staged[lane] + staged[lane - span];
    }
    for (let lane = 0; lane < chunk && block + lane < fanOut.length; lane += 1) {
      bases[block + lane] = carry + inclusive[lane] - owned[lane];
    }
    carry += inclusive[chunk - 1];
  }
  return { bases, total: carry };
}

// Comments are stripped first, using the same routine the production analyser
// in lib/wgsl-binding-reachability.ts already applies. Without it a comment that
// merely NAMES a bound resource or a callee -- "it sits in uniform control
// flow", say, against the binding whose variable is `control` -- registers as a
// use of that binding, and the audit reports a reachability the shader does not
// have. codeOnlyWGSL preserves offsets and never joins tokens, so every regex
// below still matches the same spans.
function reachableWGSLBindings(rawShader: string, entryPoint: string): number[] {
  const shader = codeOnlyWGSL(rawShader);
  const globals = [...shader.matchAll(/@group\(0\)\s+@binding\((\d+)\)\s+var(?:<[^>]+>)?\s+(\w+)/g)]
    .map((match) => ({ binding: Number(match[1]), name: match[2] }));
  const functions = new Map<string, string>();
  const signature = /\bfn\s+(\w+)\s*\(/g;
  for (let match = signature.exec(shader); match; match = signature.exec(shader)) {
    const open = shader.indexOf("{", signature.lastIndex);
    assert.notEqual(open, -1, `missing WGSL function body ${match[1]}`);
    let depth = 1, cursor = open + 1;
    while (cursor < shader.length && depth > 0) {
      if (shader[cursor] === "{") depth += 1;
      else if (shader[cursor] === "}") depth -= 1;
      cursor += 1;
    }
    assert.equal(depth, 0, `unterminated WGSL function ${match[1]}`);
    functions.set(match[1], shader.slice(open + 1, cursor - 1));
    signature.lastIndex = cursor;
  }
  assert.ok(functions.has(entryPoint), `missing WGSL entry point ${entryPoint}`);
  const visited = new Set<string>(), pending = [entryPoint];
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (visited.has(name)) continue;
    visited.add(name); const body = functions.get(name)!;
    for (const candidate of functions.keys()) {
      if (!visited.has(candidate) && new RegExp(`\\b${candidate}\\s*\\(`).test(body)) pending.push(candidate);
    }
  }
  const reachableBodies = [...visited].map((name) => functions.get(name)!);
  return globals.filter(({ name }) => reachableBodies.some((body) => {
    const locals = new Set([...body.matchAll(/\b(?:let|var)\s+(\w+)/g)].map((match) => match[1]));
    return !locals.has(name) && new RegExp(`\\b${name}\\b`).test(body);
  }))
    .map(({ binding }) => binding).sort((a, b) => a - b);
}

test("native sparse pyramid allocation is bounded by row capacity, not dense domain volume", () => {
  const narrow = planOctreeSPGridVCycle({ dimensions: [64, 48, 32], rowCapacity: 5_000 });
  const wide = planOctreeSPGridVCycle({ dimensions: [1_024, 48, 32], rowCapacity: 5_000 });
  assert.ok(wide.levelStride >= narrow.levelStride);
  assert.equal(narrow.transferStride, 40_000);
  assert.ok(wide.levelCount > narrow.levelCount);
  assert.equal(wide.stateBytes, 26 * wide.totalLevelSlots * 4,
    "first-order MG state must not retain the deleted accurate A2 image");
  assert.equal(wide.pageRecordWords, 28);
  assert.deepEqual(wide.levelOffsets,
    wide.levelCapacities.map((_, level) => wide.levelCapacities.slice(0, level)
      .reduce((sum, capacity) => sum + capacity, 0)));
  assert.ok(wide.levelCapacities.every((capacity, level) =>
    level === 0 || capacity <= wide.levelCapacities[level - 1]));
  assert.equal(wide.transferOffsets.length, wide.levelCount - 1);
  const ui = planOctreeSPGridVCycle({ dimensions: [24, 18, 16], rowCapacity: 6_912 });
  assert.deepEqual(ui.levelCapacities, [16_384, 2_048, 256, 64, 8, 2]);
  assert.ok(ui.stateBytes < 4 * 1024 * 1024,
    "per-level arenas must retain the candidate hash table's 50% load proof without finest-level padding");
  assert.ok(Math.max(...ui.transferCapacities) > ui.levelStride,
    "the UI plan exercises transfer publication beyond the largest level arena");
  assert.ok(ui.stateBytes <= 128 * 1024 * 1024);
  assert.ok(ui.topologyBytes <= 128 * 1024 * 1024);
  assert.equal(wide.dispatchBytes, wide.levelCount * 48 + 20);
  assert.equal(wide.capturePageCount, Math.ceil(wide.rowCapacity / 64));
  assert.equal(wide.capturePageStateBytes, (12 + 4 * wide.capturePageCount) * 4);
  assert.ok(!("cellCount" in wide));
  assert.throws(() => planOctreeSPGridVCycle({ dimensions: [16, 16, 16],
    rowCapacity: SPGRID_MAXIMUM_ROW_CAPACITY + 1 }), /bounded/);
});

test("SPGrid hierarchy census separates live roles, padding, and convergence-gated widths", () => {
  const plan = planOctreeSPGridVCycle({ dimensions: [4, 4, 4], rowCapacity: 8 });
  const flags = new Uint32Array(plan.totalLevelSlots);
  const level0 = plan.levelOffsets[0]!;
  flags.set([SPGRID_CELL_FLAG.active, SPGRID_CELL_FLAG.ghost,
    SPGRID_CELL_FLAG.multigridOnly, SPGRID_CELL_FLAG.active | SPGRID_CELL_FLAG.ghost], level0);
  const dispatch = new Uint32Array(plan.dispatchBytes / 4);
  const gated = new Uint32Array(plan.dispatchBytes / 4);
  dispatch.set([4, 7, 1, 1, 1, 2, 1, 1, 3, 3, 1, 1], 0);
  gated.set([4, 7, 0, 1, 1, 0, 1, 1, 3, 0, 1, 1], 0);
  const census = decodeOctreeSPGridHierarchyCensus(plan, flags, dispatch, gated);
  assert.deepEqual(census.levels[0], {
    level: 0, capacity: plan.levelCapacities[0], occupied: 4,
    active: 2, ghost: 2, multigridOnly: 1, invalidClass: 1,
    publishedCount: 4, publishedTransferCount: 7, publishedPageCount: 3,
    selectedGroups: 1, restrictionGroups: 2, pageGroups: 3,
    gatedSelectedGroups: 0, gatedRestrictionGroups: 0, gatedPageGroups: 0,
  });
  assert.throws(() => decodeOctreeSPGridHierarchyCensus(
    plan, flags.subarray(1), dispatch, gated,
  ), /shorter than the plan/);
});

test("SPGrid derives an aligned row ceiling from the negotiated binding limit", () => {
  const dimensions = [320, 96, 80] as const;
  assert.equal(spgridRowCapacityForBindingLimit(dimensions, 1024 ** 3, 184_832), 131_072);
  assert.equal(spgridRowCapacityForBindingLimit(dimensions, 2 * 1024 ** 3, 369_664), 369_664);
  const limited = planOctreeSPGridVCycle({ dimensions, rowCapacity: 131_072 });
  const overflowing = planOctreeSPGridVCycle({ dimensions, rowCapacity: 131_328 });
  assert.ok(Math.max(limited.stateBytes, limited.topologyBytes) <= 1024 ** 3);
  assert.ok(Math.max(overflowing.stateBytes, overflowing.topologyBytes) > 1024 ** 3);
});

test("degree-2/4 Chebyshev schedules use a safe positive scaled first-order interval", () => {
  // Aanjaneya et al. 2017, Section 4.3
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) requires the first-order
  // V-cycle M1 to be linear, symmetric, and positive definite. This fixture
  // evaluates the same globally synchronized polynomial used by the GPU.
  const matrix = [
    [2, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 2],
  ];
  const bounds = computeSPGridScaledSpectralBounds(matrix.map((row, index) => ({
    diagonal: row[index],
    offDiagonalSum: row.reduce((sum, value, column) =>
      sum + (column === index ? 0 : Math.abs(value)), 0),
  })));
  assert.ok(bounds.lower > 0 && bounds.upper > bounds.lower);
  // Exact D^-1 A eigenvalues for the four-cell Dirichlet chain.
  for (let mode = 1; mode <= 4; mode += 1) {
    const eigenvalue = 1 - Math.cos(mode * Math.PI / 5);
    assert.ok(eigenvalue < bounds.upper);
  }
  // M1 shares L2's boundary conditions without copying L2's adaptive
  // directional graph onto unrelated adjacent SPGrid slots. The retained
  // quantity is the non-negative boundary reaction diag - sum(offdiag), added
  // to a symmetric first-order graph. This concrete graph satisfies Sylvester's
  // criterion (leading minors 1.5, 2, and 3) and exercises unequal reactions.
  const boundaryMatrix = [
    [1.5, -1, 0],
    [-1, 2, -1],
    [0, -1, 2.25],
  ];
  const boundaryApply = (x: readonly number[]) => boundaryMatrix.map((row) => dot(row, x));
  const boundaryA = [1, -2, 0.5], boundaryB = [-0.25, 3, 2];
  assert.ok(Math.abs(dot(boundaryA, boundaryApply(boundaryB))
    - dot(boundaryApply(boundaryA), boundaryB)) < 1e-12);
  assert.ok(dot(boundaryA, boundaryApply(boundaryA)) > 0
    && dot(boundaryB, boundaryApply(boundaryB)) > 0);
  for (const degree of [2, 4] as const) {
    const weights = Array.from({ length: degree }, (_, phase) =>
      spgridChebyshevRelaxationWeight(bounds, degree, phase));
    assert.ok(weights.every((value) => value > 0 && Number.isFinite(value)));
    const apply = (rhs: readonly number[]) => {
      let x = new Array(rhs.length).fill(0);
      for (const weight of weights) {
        const image = matrix.map((row) => dot(row, x));
        x = x.map((value, row) =>
          value + weight * (rhs[row] - image[row]) / matrix[row][row]);
      }
      return x;
    };
    const a = [1, -2, 3, -4], b = [0.5, 2, -1, 3];
    const ma = apply(a), mb = apply(b);
    assert.ok(Math.abs(dot(a, mb) - dot(ma, b)) < 1e-12,
      `degree-${degree} fixed polynomial must remain symmetric`);
    assert.ok(dot(a, ma) > 0 && dot(b, mb) > 0,
      `degree-${degree} fixed polynomial must remain positive definite`);
    const combined = apply(a.map((value, index) => value + 0.25 * b[index]));
    assert.ok(combined.every((value, index) =>
      Math.abs(value - (ma[index] + 0.25 * mb[index])) < 1e-12),
    `degree-${degree} fixed polynomial must remain linear`);
  }
  assert.throws(() => computeSPGridScaledSpectralBounds([
    { diagonal: 1, offDiagonalSum: 1.1 },
  ]), /first-order M-matrix/);
});

test("compact SPGrid level and transfer arenas preserve exact layout and hash headroom", () => {
  for (const dimensions of [
    [2_048, 2_048, 2_048],
    [2_048, 1, 1],
    [16, 16, 16],
    [24, 18, 16],
  ] as const) {
    const rowCapacity = dimensions[0] === 24 ? 6_912 : 5_000;
    const plan = planOctreeSPGridVCycle({ dimensions, rowCapacity });
    let levelPrefix = 0;
    for (let level = 0; level < plan.levelCount; level += 1) {
      assert.equal(plan.levelOffsets[level], levelPrefix);
      const scale = 2 ** level;
      const domainCells = dimensions.map((value) => Math.ceil(value / scale))
        .reduce((product, value) => product * value, 1);
      const worstCells = Math.min(rowCapacity * 8, domainCells);
      assert.ok(plan.levelCapacities[level] >= 2 * worstCells,
        `level ${level} must remain at most half full`);
      assert.equal(plan.levelCapacities[level] & (plan.levelCapacities[level] - 1), 0,
        `level ${level} capacity must remain a power of two`);
      levelPrefix += plan.levelCapacities[level];
    }
    assert.equal(plan.totalLevelSlots, levelPrefix);
    let transferPrefix = 0;
    for (let level = 0; level < plan.levelCount - 1; level += 1) {
      assert.equal(plan.transferOffsets[level], transferPrefix);
      transferPrefix += 4 * plan.transferCapacities[level] + 4 * plan.levelCapacities[level];
    }
    const directoryWords = 16 + 4 * plan.brickCount + plan.totalLevelSlots;
    // Eighteen published column indices per cell, one per stencil channel.
    const stencilNeighbourWords = 18 * plan.totalLevelSlots;
    assert.equal(plan.stencilNeighbourBytes, stencilNeighbourWords * 4,
      "the published column indices must be exactly one word per cell per stencil channel");
    const expectedTopologyWords = 16 + plan.levelCount * rowCapacity + plan.totalLevelSlots
      + 28 * plan.totalLevelSlots + plan.pageDirectoryBytes / 4
      + transferPrefix + directoryWords + stencilNeighbourWords;
    assert.equal(plan.topologyBytes, expectedTopologyWords * 4);
  }
});

test("compact SPGrid addressing and commit dispatch cover every variable arena", () => {
  // The rule was written for the V-cycle alone, so the accurate operator went
  // on violating it in four helpers. The worst was reached from pageSlot, which
  // applyRow calls eighteen times per row and finerAdjoint up to 144 more.
  for (const shader of [octreeSPGridVCycleShader, octreeSPGridAccurateOperatorShader]) {
    assert.doesNotMatch(shader,
      /\(c\*levels\(\)\+l\)\*(?:maxS|s)tride\(\)|l\*transferLevelWords\(\)|rankedSlotsBase\(\)\+l\*(?:maxS|s)tride\(\)/,
      "no shader may retain fixed finest-level padding in state, transfer, or rank addressing");
    assert.match(shader, /fn levelBase\(l:u32\)->u32/);
    assert.match(shader, /fn transferLevelOffset\(l:u32\)->u32/);
    assert.match(shader,
      /fn levelCapacity\(l:u32\)->u32\{let t=levelTable\(l\);return p\.levelCaps\[t\.x\]\[t\.y\];\}/,
      "variable-arena addressing must read the memoized host table, not a per-address loop");
    for (const helper of ["levelBase", "pageLevelOffset", "transferLevelOffset",
      "brickLevelOffset"]) {
      assert.match(shader,
        new RegExp(`fn ${helper}\\(l:u32\\)->u32\\{let t=levelTable\\(l\\);return p\\.\\w+\\[t\\.x\\]\\[t\\.y\\];\\}`),
        `${helper} must read the memoized host table, not a per-address prefix loop`);
    }
    assert.doesNotMatch(shader, /for\(var k=0u;k<l;k\+=1u\)/,
      "no address helper may re-derive a level prefix on every access");
    assert.doesNotMatch(shader, /while\(result<limit\)/,
      "no address helper may re-derive a level capacity on every access");
  }
  assert.match(WebGPUOctreeSPGridVCycle.prototype.constructor.toString(),
    /SPGrid level tables disagree with the allocated plan/,
    "the memoized level tables must fail closed against the exact allocation plan");
  assert.match(WebGPUOctreeSPGridVCycle.prototype.encodeReadySetupCommit.toString(),
    /Math\.max\(this\.plan\.levelStride,this\.plan\.brickCount,\.\.\.this\.plan\.transferCapacities\)/,
    "ready publication must cover the largest validated level, brick, or transfer arena");
});

test("4^3 brick masks rank every occupied bit exactly and reject malformed publication", () => {
  const dimensions = [9, 7, 6] as const;
  const coordinates: [number, number, number][] = [];
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) {
    for (let x = 0; x < dimensions[0]; x += 1) {
      // Exercise every bit in a complete brick and sparse edge bricks.
      if (x < 4 && y < 4 && z < 4 || (x * 17 + y * 11 + z * 5) % 7 === 0) coordinates.push([x, y, z]);
    }
  }
  const cells = coordinates.map((coordinate, index) => ({ coordinate, slot: 10_000 + ((index * 37) % coordinates.length) }));
  const directory = buildSPGridBrickRankDirectory(dimensions, cells);
  validateSPGridBrickRankDirectory(directory);
  const expected = new Map(cells.map(({ coordinate, slot }) => [coordinate.join(","), slot]));
  for (let z = 0; z < dimensions[2]; z += 1) for (let y = 0; y < dimensions[1]; y += 1) {
    for (let x = 0; x < dimensions[0]; x += 1) {
      assert.equal(lookupSPGridBrickRank(directory, [x, y, z]), expected.get(`${x},${y},${z}`), `${x},${y},${z}`);
    }
  }
  assert.equal(lookupSPGridBrickRank(directory, [9, 0, 0]), undefined);
  assert.throws(() => buildSPGridBrickRankDirectory(dimensions, [cells[0], { coordinate: [8, 6, 5], slot: cells[0].slot }]), /Duplicate/);
  assert.throws(() => buildSPGridBrickRankDirectory(dimensions, [{ coordinate: [9, 0, 0], slot: 1 }]), /Malformed/);
  assert.throws(() => validateSPGridBrickRankDirectory({ ...directory, bases: directory.bases.map((base, i) => base + Number(i === 1)) }), /Malformed/);
  assert.throws(() => validateSPGridBrickRankDirectory({ ...directory, slots: directory.slots.slice(1) }), /Malformed/);
  const masks = directory.masks.map((mask) => [...mask] as [number, number]);
  const edgeBrick = masks.length - 1; masks[edgeBrick][1] = (masks[edgeBrick][1] | 0x8000_0000) >>> 0;
  assert.throws(() => validateSPGridBrickRankDirectory({ ...directory, masks }), /Malformed/);
});

test("8x8x4 pressure pages publish exact stable physical 27-neighbour adjacency", () => {
  const pages = buildSPGridPhysicalPageAdjacency([24, 16, 8], [
    [0, 0, 0], [9, 0, 0], [17, 0, 0], [9, 9, 0], [9, 9, 5],
  ]);
  assert.deepEqual(pages.origins, [
    [0, 0, 0], [8, 0, 0], [16, 0, 0], [8, 8, 0], [8, 8, 4],
  ]);
  assert.equal(pages.neighbours.every((record) => record.length === 27), true);
  const centre = pages.neighbours[1];
  assert.equal(centre[13], 1, "the centre adjacency entry is the page's stable physical ID");
  assert.equal(centre[12], 0); assert.equal(centre[14], 2);
  assert.equal(centre[16], 3); assert.equal(centre[25], 4);
  assert.equal(centre[10], -1, "empty logical pages stay invalid rather than aliasing another page");
  assert.throws(() => buildSPGridPhysicalPageAdjacency([8, 8, 4], [[8, 0, 0]]), /Malformed/);
});

test("global M1 smoother consumes the published column index between synchronized phases", () => {
  // This dispatch boundary is part of the Section 4.3 M1 assumption in
  // `docs/papers/aanjaneya-2017-power-liquids.txt`: freezing cross-page halos
  // across multiple phases would not apply the global SPD polynomial above.
  assert.match(octreeSPGridVCycleShader,
    /fn linkCandidatePageNeighbours[\s\S]*pageDirectoryBase\(\)[\s\S]*candidateTopology\[record\+1u\+ordinal\]=physical/,
    "candidate publication must resolve all 27 physical neighbours before commit");
  const smoother = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn relaxChebyshev("),
    octreeSPGridVCycleShader.indexOf("fn aggregateResolve("),
  );
  assert.match(smoother, /applied\(slot,src\)/);
  assert.doesNotMatch(smoother, /pageSlots|workgroupBarrier|\bfind\(|\bdirectoryLookup\(/,
    "one dispatch must evaluate one global Jacobi phase from the published columns");
  const pageSlot = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn pageSlot("),
    octreeSPGridVCycleShader.indexOf("fn originOf("),
  );
  assert.match(pageSlot, /physical=pageNeighbour\(l,page,ordinal\)/);
  assert.doesNotMatch(pageSlot, /directoryLookup\(|topology\[directoryBase\(\)\+2u\+l\]/,
    "physical adjacency removes recurring publication probes from halo staging");
});

test("accurate A2 stages wide direct terms before an ordered row fold", () => {
  // `docs/papers/aanjaneya-2017-power-liquids.txt`, Section 4.3 assumes the
  // accurate A2 matrix is SPD. Omitting rows because a one-dimensional Dawn
  // dispatch exceeds 65,535 workgroups turns p^T A2 p into zero and violates
  // that assumption before CG has taken its first iteration.
  assert.match(octreeSPGridAccurateDispatchGateShader,
    /fn publishAccurateDispatch[\s\S]*min\(65535u,blocks\)[\s\S]*\(blocks\+x-1u\)\/x/,
    "wide accurate stages must fold their exact workgroup count into two dimensions");
  assert.match(octreeSPGridAccurateDispatchGateShader,
    /publishAccurateDispatch\(15u,\(unionRows\*18u\+63u\)\/64u/,
    "every direct A2 term must remain scheduled above the Dawn X-dimension ceiling");
  assert.match(octreeSPGridAccurateDispatchGateShader,
    /publishAccurateDispatch\(18u,\(transitionRows\*144u\+63u\)\/64u/,
    "every fine-adjoint A2 term must remain scheduled above the Dawn X-dimension ceiling");
  assert.match(octreeSPGridAccurateOperatorShader,
    /page=pageFor\(l,q\);\s*if\(page==INVALID\|\|page>=levelCapacity\(l\)\)\{reportAt\(2u,31u,row\);return;\}/,
    "a missing native physical page must fail closed before any page-record access");
  for (const entry of ["applyRegularInterior", "applyTransitionInterior",
    "applyPhysicalBoundary", "applyTransitionBoundary"]) {
    assert.match(octreeSPGridAccurateOperatorShader, new RegExp(`fn ${entry}\\b`));
  }
  assert.match(octreeSPGridAccurateOperatorShader, /section63Coefficients:array<f32>/);
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn coefficientBase\(row:u32\)->u32\{return acceptedBank\(\)\*p\.hierarchy\.w\+row\*19u;/,
    "A2 must read the accepted dynamic Section 6.3 bank");
  assert.match(octreeSPGridAccurateOperatorShader, /fn pageSlot[\s\S]*pageNeighbour/);
  assert.match(octreeSPGridAccurateOperatorShader, /fn finerAdjoint[\s\S]*state\[at\(OWNER,fine,ghost\)\]!=row\+1u/);
  // The compiled operator image carries u32 indices and nothing else. A float
  // stored here and reloaded would end the term expression and cost the fused
  // multiply-add, which POWER_LIQUIDS_ULTIMATE_M1MAX refuted lever 10 measured
  // as a physics change, not a restructuring.
  assert.match(octreeSPGridAccurateOperatorShader,
    /@binding\(13\) var<storage,read_write> operatorRows:array<u32>;/,
    "the compiled operator image must be a u32 index table");
  assert.doesNotMatch(octreeSPGridAccurateOperatorShader,
    /bitcast<f32>\(operatorRows|operatorRows\[[^\]]*\]=bitcast/,
    "no float may round-trip through the operator image");
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn stageDirectTerm[\s\S]{0,900}?let code=operatorRows\[image\+1u\+channel\];[\s\S]{0,200}?term=c\*\(inputVector\[row\]-inputVector\[code\]\);/,
    "the direct term must gather one linear image word instead of chasing pageSlot");
  assert.doesNotMatch(
    octreeSPGridAccurateOperatorShader.slice(
      octreeSPGridAccurateOperatorShader.indexOf("fn stageDirectTerm"),
      octreeSPGridAccurateOperatorShader.indexOf("fn buildOperatorRow")),
    /pageSlot|pageFor\(/,
    "the direct term stage must retain no page-directory chase at all");
  assert.doesNotMatch(octreeSPGridAccurateOperatorShader,
    /ResolvedParams|resolvedRows|neighbourRow|maximumNeighborSlots|atomicAddF32/,
    "production A2 must expose neither the retired row-gather ABI nor scatter accumulation");
  const regularEntry = octreeSPGridAccurateOperatorShader.slice(
    octreeSPGridAccurateOperatorShader.indexOf("fn applyRegularInterior"),
    octreeSPGridAccurateOperatorShader.indexOf("fn applyTransitionInterior"),
  );
  assert.match(regularEntry, /applyRow\(row\)/);
  // RENEGOTIATED with the per-candidate E^T widening (POWER_LIQUIDS_ULTIMATE
  // _M1MAX Part D's "parallel gather, serial fold", priority item 2). The
  // adjoint producer is now `stageAdjointCandidate`, one lane per (row, child,
  // candidate), so it writes ONE term rather than a child's eighteen. The
  // named property is unchanged and is what the pattern still pins: dependent
  // terms are produced independently, then folded in channel order.
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn stageDirectTerm[\s\S]*accurateTerms\[destination\]=term;[\s\S]*fn stageAdjointCandidate[\s\S]*accurateTerms\[destination\]=c\*\(x-inputVector\[other\]\);[\s\S]*fn foldStagedRow[\s\S]*value\+=accurateTerms\[row\*162u\+channel\]/,
    "dependent channel terms must be produced independently then folded in channel order");
  // The fold itself must stay serial, in one lane's registers, in ascending
  // child-then-candidate order. Widening the producer must never become a
  // reassociation of the sum.
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn foldStagedRow[\s\S]*for\(var child=0u;child<8u;child\+=1u\)\{for\(var candidate=0u;candidate<18u;candidate\+=1u\)\{\s*value\+=accurateTerms\[row\*162u\+18u\+child\*18u\+candidate\];/,
    "the E^T fold must keep the serial ascending order the single-lane walk used");
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn finalizeStagedRow\(row:u32\)\{\s*let folded=foldStagedRow\(row\);if\(folded\.valid\)\{outputVector\[row\]=folded\.value;\}/,
    "ordinary A2 application must publish the shared ordered fold unchanged");
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn finalizeStagedResidualRow[\s\S]*let folded=foldStagedRow\(row\)[\s\S]*residualRhs\[row\]-folded\.value[\s\S]*outputVector\[row\]=residual/,
    "the residual specialization must subtract only after the shared ordered A2 fold");
  assert.deepEqual(
    reachableWGSLBindings(octreeSPGridAccurateOperatorShader, "applyAcceptedUnion"),
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "the serial union-row arm must fit the portable ten-storage-buffer limit",
  );
  assert.deepEqual(
    reachableWGSLBindings(
      octreeSPGridAccurateOperatorShader, "applyCompiledAcceptedUnion",
    ),
    [0, 1, 2, 3, 4, 5, 10, 11, 13, 14],
    "compiled ordinary inline A2 reaches exactly eight storage buffers and two uniforms",
  );
  assert.deepEqual(
    reachableWGSLBindings(
      octreeSPGridAccurateOperatorShader, "applyCompiledMergedBand",
    ),
    [0, 1, 2, 3, 4, 5, 9, 10, 11, 13, 14],
    "compiled merged-band inline A2 adds only the metric class buffer: nine storage total",
  );
  const compiledInline = octreeSPGridAccurateOperatorShader.slice(
    octreeSPGridAccurateOperatorShader.indexOf("fn applyCompiledRow"),
    octreeSPGridAccurateOperatorShader.indexOf("// One linear, prefetchable load"),
  );
  assert.doesNotMatch(compiledInline, /pageSlot|pageFor|topology\[|state\[/,
    "compiled inline A2 must contain no topology/state address chase");
  assert.match(compiledInline,
    /for\(var channel=0u;channel<18u;channel\+=1u\)[\s\S]*value\+=c\*\(x-inputVector\[code\]\)[\s\S]*for\(var child=0u;child<8u;child\+=1u\)[\s\S]*for\(var candidate=0u;candidate<18u;candidate\+=1u\)[\s\S]*value\+=c\*\(x-inputVector\[other\]\)/,
    "compiled inline A2 must retain direct-then-child/candidate ascending association");
  // RENEGOTIATED with the same change: the adjoint lane count per row goes from
  // 8 (one per child, each walking eighteen candidates in sequence) to 144.
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn stageAcceptedUnionTerms[\s\S]*unionRow\(item\/18u\)[\s\S]*fn stageMergedBandTerms[\s\S]*workRow\(item\/18u,4u\)[\s\S]*fn stageAcceptedUnionAdjoints[\s\S]*item\/144u[\s\S]*stageAdjointCandidate\(row,\(item%144u\)\/18u,item%18u\)[\s\S]*fn stageMergedBandAdjoints[\s\S]*fn finalizeStagedUnionRows/,
    "the wide stages must expose direct channels and fine adjoint candidates independently");
  // Both publishers of the adjoint record must agree with that lane count.
  assert.match(octreeSPGridAccurateDispatchGateShader,
    /publishAccurateDispatch\(18u,\(transitionRows\*144u\+63u\)\/64u/,
    "the accepted-class gate must launch one lane per (row, child, candidate)");
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn transitionUnionCount[\s\S]*select\(1u,3u,index==1u\)[\s\S]*fn transitionUnionRow[\s\S]*stageAcceptedUnionAdjoints[\s\S]*transitionUnionRow\(rowItem\)/,
    "fine-adjoint workers must be restricted to the two coarse/fine transition classes");
  const merged = octreeSPGridAccurateOperatorShader.slice(
    octreeSPGridAccurateOperatorShader.indexOf("fn applyMergedBand"),
  );
  assert.match(merged, /workRow\(linearLane\(wg,groups,lane\),4u\)/);
  assert.match(merged, /applyRow\(row\)/);
  assert.doesNotMatch(octreeSPGridAccurateOperatorShader,
    /applyAccuratePages|aliasImage|ownerHead|ownerNext|ACCURATE_DIAG|pageSlots/,
    "the cut-over leaves no executable page-image A2 fallback");
});

test("V-cycle lookup is authoritative mask/popcount rank with no hash-probe fallback", () => {
  assert.match(octreeSPGridVCycleShader, /fn directoryLookup[\s\S]*countOneBits[\s\S]*fn find\(l:u32,q:vec3u\)->u32\{return directoryLookup\(l,q,true\);\}/);
  const publishedFind = octreeSPGridVCycleShader.slice(octreeSPGridVCycleShader.indexOf("fn directoryLookup("),
    octreeSPGridVCycleShader.indexOf("fn originOf("));
  assert.doesNotMatch(publishedFind, /probe|hash/);
  assert.match(octreeSPGridVCycleShader,
    /fn markCandidateBrickOccupancy[\s\S]*candidateTopology\[record\]=levelDelta\[deltaAt\(l,0u\)\]/);
  assert.match(octreeSPGridVCycleShader,
    /fn rankCandidateBricks[\s\S]*candidateTopology\[directoryBase\(\)\+2u\+l\]=generation/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /setupDispatch|prepareSetupDispatch/,
    "the redundant per-level indirect setup fabric must stay deleted");
});

test("native pyramid distinguishes active, ghost, and multigrid-only cells", () => {
  const pyramid = buildSPGridPyramidOracle([
    { origin: [0, 0, 0], size: 2 },
    { origin: [2, 0, 0], size: 1 },
  ], 3);
  assert.ok(pyramid.levels[0].flags.some((flag) => (flag & SPGRID_CELL_FLAG.ghost) !== 0));
  assert.ok(pyramid.levels[0].flags.some((flag) => (flag & SPGRID_CELL_FLAG.active) !== 0));
  assert.ok(pyramid.levels[1].flags.some((flag) => (flag & SPGRID_CELL_FLAG.multigridOnly) !== 0));
  assert.ok(pyramid.transfers[0].some((record) => record.weight === 1), "persisting ghost cells copy exactly");
  assert.ok(pyramid.transfers[0].some((record) => record.weight > 0 && record.weight < 1), "finest cells use trilinear records");
  for (const level of pyramid.levels) for (const flags of level.flags) {
    const classes = [SPGRID_CELL_FLAG.active, SPGRID_CELL_FLAG.ghost, SPGRID_CELL_FLAG.multigridOnly]
      .filter((classification) => (flags & classification) !== 0);
    assert.equal(classes.length, 1, "every sparse cell has one exclusive storage class");
  }
});

test("stored trilinear restriction and prolongation are exact adjoints", () => {
  const pyramid = buildSPGridPyramidOracle([
    { origin: [1, 1, 1], size: 1 }, { origin: [3, 1, 1], size: 1 }, { origin: [0, 2, 0], size: 2 },
  ], 3);
  const records = pyramid.transfers[0], fine = pyramid.levels[0].coordinates.map((_, i) => 0.25 + i * 0.7);
  const coarse = pyramid.levels[1].coordinates.map((_, i) => -1.1 + i * 0.31);
  const restricted = restrictSPGrid(fine, records, coarse.length);
  const prolonged = prolongSPGrid(coarse, records, fine.length);
  assert.ok(Math.abs(dot(restricted, coarse) - dot(fine, prolonged)) < 1e-12);
  for (let fineIndex = 0; fineIndex < fine.length; fineIndex += 1) {
    const rowSum = records.filter((record) => record.fine === fineIndex).reduce((sum, record) => sum + record.weight, 0);
    assert.ok(Math.abs(rowSum - 1) < 1e-12, `transfer row ${fineIndex} must preserve constants`);
  }
});

test("parent-owned CSR preserves boundary duplicates and the exact transfer adjoint", () => {
  const pyramid = buildSPGridPyramidOracle([
    { origin: [0, 0, 0], size: 1 }, { origin: [1, 0, 0], size: 1 }, { origin: [0, 1, 0], size: 1 },
  ], 2);
  const records = pyramid.transfers[0], fineCount = pyramid.levels[0].coordinates.length;
  const coarseCount = pyramid.levels[1].coordinates.length;
  const duplicate = records.find((record, index) => records.some((other, otherIndex) => otherIndex !== index
    && other.fine === record.fine && other.coarse === record.coarse));
  assert.ok(duplicate, "clamped boundary interpolation must retain duplicate records");
  const csr = buildSPGridParentGatherCSR(records, fineCount, coarseCount);
  for (let coarse = 0; coarse < coarseCount; coarse += 1) {
    assert.deepEqual(csr.recordIndices.slice(csr.offsets[coarse], csr.offsets[coarse + 1]),
      records.flatMap((record, index) => record.coarse === coarse ? [index] : []),
      "parent ranges retain stable source-record order including duplicates");
  }
  const fine = Array.from({ length: fineCount }, (_, index) => 0.3 + 0.7 * index);
  const coarse = Array.from({ length: coarseCount }, (_, index) => -0.2 + 0.4 * index);
  const scattered = restrictSPGrid(fine, records, coarseCount);
  const gathered = restrictSPGridParentGather(fine, records, csr);
  assert.deepEqual(gathered, scattered);
  const prolonged = prolongSPGrid(coarse, records, fineCount);
  assert.ok(Math.abs(dot(gathered, coarse) - dot(fine, prolonged)) < 1e-12,
    "parent-owned restriction remains the exact transpose of fine-owned prolongation");
});

test("parent-owned restriction has a bounded dispatch and atomic-removal target", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 128, 4_096),
    { dimensions: [16, 16, 16], rowCapacity: 128, finestCellWidth: 1 });
  assert.equal(cycle.diagnostics.restrictionScatterDispatchCount, 0);
  assert.equal(cycle.diagnostics.restrictionAtomicAddUpperBound, 0);
  assert.equal(cycle.diagnostics.parentGatherDispatchCount, cycle.plan.levelCount - 1);
  assert.equal(cycle.diagnostics.parentGatherAtomicAddCount, 0);
  const restriction = octreeSPGridVCycleShader.slice(octreeSPGridVCycleShader.indexOf("fn restrictAndGhostAccumulate"),
    octreeSPGridVCycleShader.indexOf("fn exactBottom"));
  assert.match(restriction,
    /parentHeadBase\(l\)\+coarse[\s\S]*storef\(RHS,l\+1u,coarse,loadf\(RHS,l\+1u,coarse\)\+sum\)/,
    // `docs/papers/aanjaneya-2017-power-liquids.txt`, Section 4.3 assumes L1
    // and L2 share every adaptive pressure variable. A native coarse row's
    // directly seeded RHS must survive E^T accumulation from finer rows.
    "restriction must accumulate into native coarse pressure variables");
  assert.doesNotMatch(restriction, /atomic(?:Add|CompareExchange|Or)|atomicAddF/,
    "one parent owns each restriction sum");
  cycle.destroy();
});

test("fixed LDLT bottom operation is exact, linear, symmetric, and positive", () => {
  const operator = [[4, -1, 0], [-1, 4, -1], [0, -1, 3]];
  const x = [1, -2, 0.5], y = [-0.25, 3, 2];
  const mx = solveSPGridBottomLDLT(operator, x), my = solveSPGridBottomLDLT(operator, y);
  const product = operator.map((row) => dot(row, mx));
  assert.deepEqual(product.map((value, i) => Math.abs(value - x[i]) < 1e-12), [true, true, true]);
  assert.ok(Math.abs(dot(x, my) - dot(y, mx)) < 1e-12);
  assert.ok(dot(x, mx) > 0 && dot(y, my) > 0);
  const sum = solveSPGridBottomLDLT(operator, x.map((value, i) => value + 2 * y[i]));
  assert.ok(sum.every((value, i) => Math.abs(value - mx[i] - 2 * my[i]) < 1e-12));
  assert.throws(() => solveSPGridBottomLDLT([[0]], [1]), /not SPD/);
});

test("GPU correction owns transfers by fine slot and shares one exact adjoint mapping", () => {
  assert.match(octreeSPGridVCycleShader,
    /fn geometricAggregateTransfer\(\)->bool\{return \(p\.solve\.y&2u\)!=0u;\}/,
    "the factor-1 aggregate transfer must have an immutable uniform selector");
  assert.match(octreeSPGridVCycleShader,
    /if\(\(flags&GHOST\)==0u\)\{return select\(8u,1u,geometricAggregateTransfer\(\)\);\}/,
    "factor-1 cells publish one parent record while adaptive cells retain eight");
  assert.match(octreeSPGridVCycleShader,
    /else if\(geometricAggregateTransfer\(\)\)\{[\s\S]*cResolve\(l\+1u,q\/2u\)[\s\S]*cAppendTransfer\(l,base,fine,coarse,1\.0\)/,
    "the aggregate record must be the unit-weight 2x2x2 parent");
  assert.match(octreeSPGridVCycleShader,
    /if\(geometricAggregateTransfer\(\)&&fanOut!=1u\)\{candidateReport\(l\);\}/,
    "candidate setup must reject any aggregate slot outside the total q-to-q/2 mapping");
  assert.match(octreeSPGridVCycleShader,
    /fn coarseRestrictAggregate\(l:u32,lane:u32\)[\s\S]*let coarse=workSlot\(l\+1u,lane\)[\s\S]*for\(var child=0u;child<8u;child\+=1u\)[\s\S]*sum\+=weight\*residual/,
    "the fused aggregate tail must restrict independent parents through arithmetic children");
  assert.equal((octreeSPGridVCycleShader.match(
    /targetCount=topology\[fineCountBase\(l\)\+fine\]/g) ?? []).length, 2,
    "only the generic parallel and fused prolongation branches consume the published fan-out");
  assert.match(octreeSPGridVCycleShader,
    /fn prepareCorrectionDispatches\(\)[\s\S]*local==2u\|\|local==5u\|\|local==9u[\s\S]*value=0u/,
    "convergence zeros every selected-row, transfer, and page record before correction");
  assert.match(octreeSPGridVCycleShader, /clearCorrection[\s\S]*r<rows\(\)&&!stopped\(\)/,
    "post-convergence correction clears must be write-free");
  assert.match(octreeSPGridVCycleShader, /zeroVectors[\s\S]*i>=count\(l\)\|\|stopped\(\)/,
    "post-convergence sparse-level clears must be write-free");
  const bindings = [...octreeSPGridVCycleShader.matchAll(/@group\(0\) @binding\((\d+)\)/g)]
    .map((match) => Number(match[1]));
  assert.equal(new Set(bindings).size, bindings.length,
    "every WGSL binding must have exactly one module-scope declaration");
  assert.match(octreeSPGridVCycleShader, /cAppendTransfer\(l,base\+corner,fine,coarse,parent\.weight\)/);
  assert.match(octreeSPGridVCycleShader, /fn correctionTransfer/);
  // The record index of a fine slot is the exclusive prefix sum of the per-fine
  // fan-out taken in workset order. That property - not any one spelling of the
  // running counter - is what makes E and E^T address one arena. It is pinned in
  // three parts: the narrow scan body publishes a block-scanned exclusive
  // prefix; that block scan provably equals the serial running counter; and the
  // two addressings the GPU actually uses over the resulting arena are exact
  // transposes.
  const scan = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn scanCandidateTransfers("),
    octreeSPGridVCycleShader.indexOf("fn writeCandidateTransfers("));
  assert.match(scan, /owned=candidateTopology\[fineCountBase\(l\)\+fine\]/,
    "the scanned quantity must be the published per-fine fan-out");
  assert.match(scan, /let inclusive=blockInclusiveSum\(lane,owned\);\s*let base=chunkCarry\+inclusive-owned;/,
    "the record base must be the exclusive prefix of the inclusive block scan");
  assert.match(scan, /chunkCarry=chunkCarry\+chunkScan\[CHUNK-1u\]/,
    "each block must carry its whole total into the next block");
  assert.match(scan, /candidateTopology\[fineHeadBase\(l\)\+fine\]=base;/,
    "the exclusive prefix must be published unmodified as that fine slot's record base");
  assert.match(scan, /candidateDispatch\[l\*DISPATCH_WORDS\+1u\]=chunkCarry/,
    "the level's record count must be the final carry");
  const linkChains = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn linkCandidateParentChains("),
    octreeSPGridVCycleShader.indexOf("fn brickOfIndex("));
  assert.match(linkChains, /if\(t<lane\)\{predecessor=block\+t;\}/,
    "a record's chain predecessor is the largest smaller record index sharing its coarse slot");
  // The uniform sweep is load-bearing for speed, not just correctness: the
  // equivalent per-lane first-hit searches measured 0.903 -> 1.134 ms because
  // chunkKey[t] stops being a workgroup-memory broadcast. Pin the shared index.
  assert.match(linkChains, /for\(var t=0u;t<CHUNK;t\+=1u\)/,
    "every lane must read the same chunkKey index so the scan stays a broadcast");
  const adjointLeaves: Array<{ origin: [number, number, number]; size: number }> = [
    { origin: [0, 4, 0], size: 2 }, { origin: [4, 4, 4], size: 4 }];
  for (let x = 0; x < 4; x += 1) for (let y = 0; y < 3; y += 1) adjointLeaves.push({ origin: [x, y, 0], size: 1 });
  const adjointOracle = buildSPGridPyramidOracle(adjointLeaves, 4);
  const arena = adjointOracle.transfers[0];
  const arenaFineCount = adjointOracle.levels[0].coordinates.length;
  const arenaCoarseCount = adjointOracle.levels[1].coordinates.length;
  const fanOut = new Array<number>(arenaFineCount).fill(0);
  for (const record of arena) fanOut[record.fine] += 1;
  assert.ok(fanOut.includes(1) && fanOut.includes(8) && arena.length > 64,
    "the adjoint fixture must span both fan-outs and more than one staged record chunk");
  const scanned = blockScannedRecordBases(fanOut, 256);
  assert.equal(scanned.total, arena.length, "the published record count is the level's whole fan-out");
  // The arena is the fine-major concatenation of the fan-out taken in workset
  // order, so its fine column is exactly the run-length expansion of fanOut and
  // the block-scanned base of every fine slot is the serial running counter.
  assert.deepEqual(arena.map((record) => record.fine),
    fanOut.flatMap((owned, fine) => new Array<number>(owned).fill(fine)),
    "the record arena is the fine-major concatenation of the per-fine fan-out");
  let running = 0;
  for (let fine = 0; fine < arenaFineCount; fine += 1) {
    assert.equal(scanned.bases[fine], running, `fine slot ${fine} owns the serial running record base`);
    running += fanOut[fine];
  }
  const arenaFine = Array.from({ length: arenaFineCount }, (_, index) => 0.25 + 0.7 * index);
  const arenaCoarse = Array.from({ length: arenaCoarseCount }, (_, index) => -1.1 + 0.31 * index);
  // E as prolongAndGhostPropagate reads it: fineHead(fine) + corner.
  const arenaProlonged = arenaFine.map((_, fine) => {
    let value = 0;
    for (let corner = 0; corner < fanOut[fine]; corner += 1) {
      const record = arena[scanned.bases[fine] + corner];
      value += record.weight * arenaCoarse[record.coarse];
    }
    return value;
  });
  // E^T as the restriction reads it: ascending record index restricted to one coarse slot.
  const arenaRestricted = arenaCoarse.map((_, coarse) => {
    let sum = 0;
    for (const record of arena) if (record.coarse === coarse) sum += record.weight * arenaFine[record.fine];
    return sum;
  });
  assert.ok(Math.abs(dot(arenaRestricted, arenaCoarse) - dot(arenaFine, arenaProlonged)) < 1e-12,
    "fineHead+corner prolongation and ascending parent-chain restriction stay exact transposes");
  const correctionTransfer = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn correctionTransfer("),
    octreeSPGridVCycleShader.indexOf("// Record-parallel staging for E^T"),
  );
  assert.match(correctionTransfer,
    /first=topology\[fineHeadBase\(l\)\+fine\][\s\S]*record=first\+corner/);
  assert.doesNotMatch(correctionTransfer, /\bfind\(|\bdirectoryLookup\(/,
    "recurring prolongation must consume setup-published transfer indices directly");
  assert.match(octreeSPGridVCycleShader, /fn restrictAndGhostAccumulate/);
  assert.match(octreeSPGridVCycleShader, /fn prolongAndGhostPropagate/);
  assert.equal(octreeSPGridVCycleShader.match(/correctionTransfer\(l,fine,corner\)/g)?.length, 2,
    "generic wide and fused coarse prolongation consume the same immutable transfer rule");
  const aggregateTarget = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn aggregateResolve("),
    octreeSPGridVCycleShader.indexOf("fn correctionTransfer("),
  );
  assert.match(aggregateTarget,
    /fn aggregateResolve[\s\S]*directoryLookup\(l,min\(q,dims\(l\)-vec3u\(1u\)\),false\)[\s\S]*q=decode\(state\[at\(KEY,l,fine\)\],l\)[\s\S]*aggregateResolve\(l\+1u,q\/2u\)/,
    "factor-1 prolongation must derive its unique parent from the fine coordinate");
  assert.doesNotMatch(aggregateTarget, /transferWord|fineHeadBase|fineCountBase|parentHeadBase/,
    "factor-1 prolongation must not read the transfer arena");
  const aggregateWide = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn restrictAggregateWide("),
    octreeSPGridVCycleShader.indexOf("// Section 4.2 GhostValueAccumulate"),
  );
  assert.match(aggregateWide,
    /childQ=parentQ\*2u\+vec3u\(lane&1u,\(lane>>1u\)&1u,\(lane>>2u\)&1u\)/,
    "wide factor-1 restriction must assign one arithmetic child to each of eight lanes");
  assert.match(aggregateWide,
    /for\(var child=0u;child<8u;child\+=1u\)[\s\S]*sum\+=weight\*restrictResidual\[child\]/,
    "wide factor-1 restriction must fold child bits in deterministic ascending order");
  assert.doesNotMatch(aggregateWide, /transferWord|fineHeadBase|fineCountBase|parentHeadBase/,
    "wide factor-1 restriction must not read the transfer arena");
  const aggregateTail = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn coarseRestrictAggregate("),
    octreeSPGridVCycleShader.indexOf("fn coarseRestrict(", octreeSPGridVCycleShader.indexOf("fn coarseRestrictAggregate(")),
  );
  assert.doesNotMatch(aggregateTail, /transferWord|fineHeadBase|fineCountBase|parentHeadBase|transferCount/,
    "fused factor-1 restriction must not read the transfer arena");
  const prolong = octreeSPGridVCycleShader.slice(octreeSPGridVCycleShader.indexOf("fn prolongAndGhostPropagate"),
    octreeSPGridVCycleShader.indexOf("fn publish", octreeSPGridVCycleShader.indexOf("fn prolongAndGhostPropagate")));
  assert.doesNotMatch(prolong, /atomicAddF/,
    "one fine invocation owns its complete prolongation sum");
  assert.match(prolong,
    /if\(geometricAggregateTransfer\(\)\)\{let coarse=aggregateCorrectionTarget\(l,fine\)/,
    "wide factor-1 prolongation must select the recordless parent path");
  // E^T's matrix entry must be the record's stored weight - the same word E
  // reads - multiplied by the ghost-aware level residual, and folded in
  // ascending record order. Pin those three properties rather than one
  // statement's spelling, and forbid outright the reassociating shapes.
  const restrictBody = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn restrictAndGhostAccumulate"),
    octreeSPGridVCycleShader.indexOf("fn exactBottom"));
  assert.equal(restrictBody.match(/bitcast<f32>\(topology\[transferWord\(l,\w+,2u\)\]\)/g)?.length, 2,
    "only wide and fused generic restriction read the stored record weight");
  assert.doesNotMatch(restrictBody, /bitcast<f32>\(topology\[transferWord\(l,\w+,(?:0u|1u|3u)\)\]\)/,
    "no other transfer word may be reinterpreted as a weight");
  assert.match(restrictBody, /residual=select\(-product,loadf\(RHS,l,fine\)-product,!ghost\)/,
    "restriction restricts the ghost-aware level residual, never a raw value");
  assert.match(restrictBody, /[Ww]eight\w*(?:\[\w+\])?\*\w*[Rr]esidual\w*(?:\[\w+\])?/,
    "each accumulated term is that record's stored weight times its residual");
  assert.doesNotMatch(restrictBody, /subgroup\w*\(|span<</,
    "restriction must fold in its explicit ascending order, never by a reassociating tree");
  assert.match(octreeSPGridVCycleShader, /transfer\.weight\*loadf\(A,l\+1u,transfer\.coarse\)/);
  assert.match(octreeSPGridVCycleShader, /const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u/);
  assert.match(octreeSPGridVCycleShader, /fn cMergeClass/);
  assert.match(octreeSPGridVCycleShader, /fn contactCoord/);
  // Was /fn cInsertOwned/. The ordered owner now claims by key rather than by
  // coordinate (see docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md): cClaimKey takes the
  // staged key directly and folds the ownership claim in, so the pin follows the
  // same responsibility under its new name. What matters to this test is that
  // one function both inserts and takes ownership, which the owner argument
  // pins more precisely than the old name did.
  assert.match(octreeSPGridVCycleShader, /fn cClaimKey\(l:u32,key:u32,flags:u32,encodedOwner:u32\)/);
  assert.match(octreeSPGridVCycleShader, /fn buildCandidateLevelSets\(/);
  assert.match(octreeSPGridVCycleShader, /const XYPP=9u.*const YZPP=17u/s,
    "the production stencil retains all twelve directed octree-edge contacts");
  assert.match(octreeSPGridVCycleShader,
    /const STATE_CHANNELS=26u/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /ACCURATE_DIAG|rebuildCandidateAccurateStencil|ownerHeadBase|ownerNextBase/,
    "the first-order hierarchy must not retain the deleted page-image A2 path");
  assert.match(octreeSPGridVCycleShader,
    /fn publishCandidateSpectralBounds[\s\S]*off>d\*\(1\.0\+1e-4\)[\s\S]*total=max\(total,spectralLane\[i\]\)[\s\S]*total\*=1\.0005[\s\S]*deltaAt\(l,7u\)/,
    "spectral safety must be an exact order-independent maximum over the candidate stencil");
  assert.match(octreeSPGridVCycleShader,
    /if\(stencilDirty\(l\)\)\{state\[at\(SPECTRAL,l,0u\)\]=levelDelta\[deltaAt\(l,7u\)\];\}/,
    "the smoother bound must commit atomically with its stencil generation");
  assert.match(octreeSPGridVCycleShader,
    /fn chebyshevWeight[\s\S]*upper\/30\.0[\s\S]*cos\(/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /\b(?:Galerkin|RAP|tripleProduct)\b/i);
  const applied = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn applied("),
    octreeSPGridVCycleShader.indexOf("fn smoothable", octreeSPGridVCycleShader.indexOf("fn applied(")),
  );
  // Section 4.6 restores the published column-index vector. The recurring
  // correction reads the slot setup already resolved; it does not re-walk the
  // global brick/rank directory once per spoke per lane.
  assert.match(applied, /if\(c==0\.0\)\{continue;\}/,
    "the zero-coefficient early-out guards the neighbour fetch and must be retained");
  assert.match(applied, /neighbourBase\(\)|neighbourAt\(/,
    "non-page correction consumes the setup-published column index for each stencil channel");
  assert.doesNotMatch(applied, /\bfind\(|\bdirectoryLookup\(/,
    "the recurring correction must not re-derive a neighbour slot per spoke");
  assert.match(applied, /reportAt\(OVERFLOW,75u,slot\)/,
    "an unresolvable column must remain fail-closed on the same overflow stage");
  assert.match(octreeSPGridVCycleShader,
    /candidateTopology\[neighbourAt\(k,l,s\)\]=other;cStoref\(XP\+k,l,s,c\)/,
    "a coefficient and the slot it was accumulated against must publish together");
  assert.match(octreeSPGridVCycleShader,
    /fn applyCandidateSkip[\s\S]*levelDelta\[deltaAt\(0u,1u\)\]=DELTA_STENCIL/,
    "unchanged topology must still refresh M1's dynamic finest boundary stencil");
  assert.match(octreeSPGridVCycleShader,
    /fn buildCandidateStencils[\s\S]*i>=selectedCount\(l\)[\s\S]*selectedWorkSlot\(l,i\)[\s\S]*selectedState\(KEY,l,s\)[\s\S]*selectedDirectoryLookup\(l,vec3u\(targetQ\)\)/,
    "a stencil-only refresh must enumerate the accepted topology, never stale inactive candidate slots");
  assert.match(octreeSPGridVCycleShader,
    /fn publishCandidateSpectralBounds[\s\S]*selectedCount\(l\)[\s\S]*selectedWorkSlot\(l,i\)/,
    "the spectral proof must cover the same accepted worklist as a stencil-only rebuild");
  assert.match(octreeSPGridVCycleShader,
    /acceptedOff\+=max\(0\.0,c\)[\s\S]*diagonal=max\(0\.0,acceptedDiagonal-acceptedOff\)/,
    "M1 must import L2's boundary reaction while retaining its symmetric first-order graph");
  assert.match(octreeSPGridVCycleShader,
    /var acceptedFine=\(flags&ACTIVE\)!=0u\s*&&encodedOwner>0u/,
    // `docs/papers/aanjaneya-2017-power-liquids.txt`, Section 4.3 states that
    // L1 and L2 have exactly the same pressure variables. Native adaptive
    // pressure rows exist at every octree level, so their L2 boundary reaction
    // must not be restricted to SPGrid level zero.
    "every native adaptive pressure level must share L2's boundary reaction");
  assert.match(octreeSPGridVCycleShader,
    /if\(!\(diagonal>1e-20\)\|\|!finite\(diagonal\)\)\{diagonal=coefficient;\}/,
    "the exact one-cell bottom solve must retain the first-order grid scale");
  assert.doesNotMatch(octreeSPGridVCycleShader,
    /section63ChannelForDirection\(transform,section63Direction\(k\)\)/,
    "adaptive L2 directions must not be projected onto unrelated adjacent SPGrid slots");
  assert.match(octreeSPGridVCycleShader,
    /if\(\(stencilDirty\(l\)\|\|topologyChanged\)&&i<levelCapacity\(l\)\)\{[\s\S]{0,160}?topology\[neighbourAt\(k,l,i\)\]=candidateTopology\[neighbourAt\(k,l,i\)\]/,
    "the column indices must commit on the union of the gates that publish their coefficients");
  assert.doesNotMatch(octreeSPGridVCycleShader, /select\([^\n]*insert\([^\n]*insertOwned/,
    "WGSL select eagerly evaluates both insertion paths");
  assert.match(octreeSPGridVCycleShader, /fn smoothable/);
  assert.match(octreeSPGridVCycleShader, /fn smoothChebyshevAtoB0/);
  assert.match(octreeSPGridVCycleShader, /fn smoothChebyshevBtoA3/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn smoothPageChebyshev|relaxJacobi/);
  assert.match(octreeSPGridVCycleShader, /let product=applied\(fine,A\)/,
    "restriction must consume the final pre-smoothing stencil without materializing AX or residual");
  assert.doesNotMatch(octreeSPGridVCycleShader,
    /\b(?:AX|RESIDUAL)\b|fn (?:applyA|applyB|jacobiAtoB|jacobiBtoA|formResidual)\b/,
    "the fused correction schedule must not retain legacy materialization channels or kernels");
  assert.match(octreeSPGridVCycleShader, /fn exactBottom/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /while\s*\([^)]*atomicLoad/, "no cross-workgroup spin barrier is permitted");
});

test("every SPGrid auto-layout binds the complete reachable resource ABI", () => {
  assert.equal(OCTREE_SPGRID_CAPTURE_CONTROL_WORD.sourceGeneration, 10,
    "the coupled epoch must consume the explicit topology-source generation word");
  assert.match(octreeSPGridVCycleShader,
    /fn candidateSource\(\)->bool\{return \(p\.solve\.y&1u\)!=0u;\}[\s\S]*fn sourceControlReady\(\)->bool[\s\S]*!candidateSource\(\)[\s\S]*acceptedRows\[0\]==0u[\s\S]*acceptedRows\[0\]==STRUCTURED_CANDIDATE_READY/,
    "candidate and accepted structured controls must use explicit, disjoint source modes");
  assert.match(octreeSPGridVCycleShader,
    /fn sourceGeneration\(\)->u32\{return select\(0u,select\(acceptedRows\[3\],acceptedRows\[4\],candidateSource\(\)\),sourceControlReady\(\)\);\}/,
    "candidate capture must read the candidate epoch rather than its slot count");
  assert.match(octreeSPGridVCycleShader,
    /fn acceptedBank\(\)->u32\{return select\(acceptedRows\[4\],acceptedRows\[5\],candidateSource\(\)\)&1u;\}/,
    "candidate capture must read the candidate bank rather than its epoch");
  assert.match(octreeSPGridVCycleShader,
    /fn geometricAggregateTransfer\(\)->bool\{return \(p\.solve\.y&2u\)!=0u;\}[\s\S]*return select\(8u,1u,geometricAggregateTransfer\(\)\)/,
    "factor-1 aggregate transfer must be orthogonal to the source-authority bit");
  assert.match(octreeSPGridVCycleShader,
    /fn beginL1CapturePlan\(\)[\s\S]*if\(!sourceControlReady\(\)\|\|acceptedRows\[2\]>p\.capacity\.x/,
    "candidate capture setup must accept the explicit candidate-ready magic");
  assert.match(octreeSPGridVCycleShader,
    /capturePages\.sourceGeneration=sourceGeneration\(\)[\s\S]*capturePages\.sourceGeneration!=deltaControl\(7u\)/,
    "candidate capture must publish and validate the topology source generation");
  assert.match(octreeSPGridAccurateDispatchGateShader,
    /classDispatch\[destination\]=select\(0u,worksets\[source\],solveLive&&valid\)/,
    "the accurate owner zeroes every stopped destination class record");
  // The page resolution still happens inline, on purpose: memoizing it on the
  // ordinal is exact but measured as nothing (see the refuted-levers list).
  //
  // RENEGOTIATED for the per-epoch operator image (POWER_LIQUIDS_ULTIMATE_M1MAX
  // Part F1 stage 2, re-scored). The resolution moved from `pageSlot` into
  // `pageSlotCoded`, which returns the report stage instead of raising it, so
  // the once-per-epoch image builder can run the identical resolution without
  // touching the solver control. `pageSlot` is now nothing but that call plus
  // the report. Only the assertion's subject changed: the pin is still "the
  // page is resolved inline, in one function, with no threaded memo".
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn pageSlotCoded\(l:u32,page:u32,origin:vec3u,q:vec3u\)->vec2u\{[\s\S]{0,400}?let physical=pageNeighbour\(l,page,ordinal\);/,
    "pageSlotCoded must resolve the page inline rather than through a threaded memo");
  assert.match(octreeSPGridAccurateOperatorShader,
    /fn pageSlot\(l:u32,page:u32,origin:vec3u,q:vec3u,row:u32\)->u32\{\s*let resolved=pageSlotCoded\(l,page,origin,q\);\s*if\(resolved\.y!=0u\)\{reportAt\(2u,resolved\.y,row\);\}\s*return resolved\.x;\}/,
    "pageSlot must be pageSlotCoded plus its report, so the compiled image and the inline walk cannot diverge");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.beginL1CapturePlan, [0, 3, 6, 13, 14, 18]);
  // The capture plan was split so the per-row Section 6.3 validation runs one
  // workgroup per page instead of one lane of one workgroup. The row scan kept
  // the row bindings and lost the level-delta arena (14), which markDirtyFrom
  // took with it into the reduction; the reduction keeps only what the
  // work-list traversal and the finalization reach.
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.planL1CaptureDelta, [0, 1, 3, 11, 13, 18, 20, 21]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.reduceL1CaptureDelta, [0, 3, 13, 14, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.commitChangedL1, [0, 1, 3, 11, 13, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.finalizeL1CapturePublication, [0, 3, 13, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.detectCandidateGhosts,
    [0, 1, 3, 14, 16, 20, 21, 22],
    "catalog-owned ghost detection stays under the portable storage-binding ceiling");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.buildCandidateLevelDeltas,
    [0, 1, 3, 4, 5, 6, 14, 15, 16, 17],
    "the ordered hierarchy builder stays under the portable storage-binding ceiling");
  assert.equal("buildCandidateLevelSetsAndGhosts" in OCTREE_SPGRID_VCYCLE_BINDINGS, false,
    "the serialized whole-hierarchy candidate builder must not remain in the pipeline ABI");
  assert.equal("resetInvalidBuffers" in OCTREE_SPGRID_VCYCLE_BINDINGS, false,
    "the deleted all-capacity recovery path must not remain in the pipeline ABI");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.clearCorrection, [0, 3, 7, 9],
    "correction clearing observes the solver stop gate before writing output");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.prepareCorrectionDispatches, [0, 3, 6, 7, 19],
    "one singleton owns the zero-x level records and live-row schedule");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.zeroVectors, [0, 4, 5, 6, 7],
    "vector clearing observes the solver stop gate before touching sparse slots");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.seedRhs, [0, 3, 4, 5, 7, 8, 11],
    "native-level RHS seeding reads fixed row geometry");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.seedRhsAndClearCorrection,
    [0, 3, 4, 5, 7, 8, 9, 11],
    "factor-1 initialization is the exact reachable-resource union of clear and seed");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.restrictAndGhostAccumulate, [0, 4, 5, 6, 7],
    "parent-owned restriction reads only compact transfer/state storage");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.publish, [0, 3, 4, 5, 7, 9, 11],
    "native-level correction publication likewise reads fixed row geometry");
  for (const [entryPoint, bindings] of Object.entries(OCTREE_SPGRID_VCYCLE_BINDINGS)) {
    assert.equal(new Set(bindings).size, bindings.length, `${entryPoint} must not bind a resource twice`);
    assert.ok(bindings.filter((binding) => binding !== 0 && binding !== 10).length <= 10,
      `${entryPoint} exceeds the portable ten-storage-buffer stage budget`);
    assert.deepEqual([...bindings].sort((a, b) => a - b), reachableWGSLBindings(octreeSPGridVCycleShader, entryPoint),
      `${entryPoint} TS bind group must equal exact WGSL auto-layout reachability`);
  }
});

test("one correction gates then executes exact indirect records with cached descriptors", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  let passes = 0, dispatches = 0, groups = 0;
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => { groups += 1; return {}; },
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 128, 8 * 512),
    { dimensions: [16, 16, 16], rowCapacity: 128, maximumLevels: 5, finestCellWidth: 1 });
  const encoder = {
    clearBuffer() {}, copyBufferToBuffer() {}, beginComputePass: () => {
      passes += 1; return { setPipeline() {}, setBindGroup() {},
        dispatchWorkgroups() { dispatches += 1; }, dispatchWorkgroupsIndirect() { dispatches += 1; }, end() {} };
    },
  } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(512), correction: buffer(512) };
  const broker = new PassBroker(encoder);
  cycle.encodeSetup(broker, input);
  assert.equal(dispatches, cycle.encodedSetupDispatchCount);
  const setupPasses = passes, before = dispatches;
  cycle.encodeCorrection(broker, input);
  broker.fence("correction complete");
  assert.equal(passes - setupPasses, 2);
  assert.equal(dispatches - before, cycle.encodedCorrectionDispatchCount);
  assert.equal(cycle.encodedCorrectionDispatchCount, 22,
    "two parallel levels plus a barrier-synchronized small tail own the correction schedule");
  assert.equal(cycle.encodedPassTransitionCount, 1);
  assert.equal(cycle.diagnostics.bottomOperation, "exact-single-cell");
  assert.equal(cycle.diagnostics.coarsestDegreesOfFreedom, 1);
  assert.equal(cycle.diagnostics.pageAdjacency, "physical-27");
  assert.equal(cycle.diagnostics.smootherLookup, "published-column-index");
  const firstGroups = groups;
  cycle.encodeCorrection(broker, input);
  broker.fence("correction complete");
  assert.equal(groups, firstGroups, "repeated solves allocate no new bind groups");
  cycle.destroy();
});

test("factor-1 dense correction replaces sparse recurring work and its kill switch preserves fusion", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const encode = (
    geometricAggregateTransfers: boolean,
    denseEnabled = true,
  ) => {
    const device = {
      queue: { writeBuffer() {} },
      createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
      createShaderModule: () => ({}),
      createComputePipeline: ({ label }: { label: string }) =>
        ({ label, getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
    } as unknown as GPUDevice;
    const previous = process.env.FLUID_OCTREE_FACTOR1_DENSE_MG;
    process.env.FLUID_OCTREE_FACTOR1_DENSE_MG = denseEnabled ? "1" : "0";
    let cycle: WebGPUOctreeSPGridVCycle;
    try {
      cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 128, 8 * 512), {
        dimensions: [16, 16, 16], rowCapacity: 128, maximumLevels: 5,
        finestCellWidth: 1, geometricAggregateTransfers,
      });
    } finally {
      if (previous === undefined) delete process.env.FLUID_OCTREE_FACTOR1_DENSE_MG;
      else process.env.FLUID_OCTREE_FACTOR1_DENSE_MG = previous;
    }
    const entries: string[] = [];
    let current = "";
    const pass = {
      setPipeline(pipeline: { label: string }) {
        current = pipeline.label
          .replace("SPGrid V-cycle · ", "")
          .replace("Factor-1 dense M1 · ", "");
      },
      setBindGroup() {},
      dispatchWorkgroups() { entries.push(current); },
      dispatchWorkgroupsIndirect() { entries.push(current); },
      end() {},
    } as unknown as GPUComputePassEncoder;
    const broker = new PassBroker({
      beginComputePass: () => pass,
    } as unknown as GPUCommandEncoder);
    cycle.encodeCorrection(broker, {
      rowCount: buffer(64), solverControl: buffer(64),
      rhs: buffer(512), correction: buffer(512),
    });
    broker.fence("correction complete");
    const result = {
      entries,
      dispatches: cycle.encodedCorrectionDispatchCount,
      initializationDispatches: cycle.diagnostics.correctionInitializationDispatchCount,
    };
    cycle.destroy();
    return result;
  };
  const generic = encode(false);
  const factorOneSparse = encode(true, false);
  const factorOneDense = encode(true);
  assert.equal(generic.dispatches, 22);
  assert.equal(factorOneSparse.dispatches, 21,
    "the kill switch preserves the fused sparse factor-1 fallback");
  assert.equal(factorOneDense.dispatches, 15,
    "degree-two dense factor-1 correction also fuses native publication");
  assert.equal(generic.initializationDispatches, 2);
  assert.equal(factorOneSparse.initializationDispatches, 1);
  assert.equal(factorOneDense.initializationDispatches, 1);
  assert.equal(generic.entries.filter((entry) => entry === "clearCorrection").length, 1);
  assert.equal(generic.entries.filter((entry) => entry === "seedRhs").length, 1);
  assert.equal(generic.entries.includes("seedRhsAndClearCorrection"), false,
    "factor-4/8 fallback retains the original clear/seed ABI and schedule");
  assert.equal(factorOneSparse.entries.includes("clearCorrection"), false);
  assert.equal(factorOneSparse.entries.includes("seedRhs"), false);
  assert.equal(factorOneSparse.entries.filter(
    (entry) => entry === "seedRhsAndClearCorrection",
  ).length, 1);
  assert.equal(factorOneSparse.entries.filter((entry) => entry === "zeroVectors").length,
    generic.entries.filter((entry) => entry === "zeroVectors").length,
    "fusion must not widen or duplicate any sparse-slot clear");
  assert.equal(factorOneDense.entries.includes("zeroVectors"), false);
  assert.equal(factorOneDense.entries.filter(
    (entry) => entry === "initializeDenseCorrection",
  ).length, 1);
  assert.equal(factorOneDense.entries.at(0), "prepareDenseCorrectionDispatches");
  assert.equal(factorOneDense.entries.at(-1), "smoothDenseBtoA0AndPublish");

  const fused = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn seedRhsAndClearCorrection"),
    octreeSPGridVCycleShader.indexOf("fn stencilDirection"),
  );
  assert.match(fused,
    /let r=rowIndex\(g\);if\(r<rows\(\)&&!stopped\(\)\)\{outputCorrection\[r\]=0\.0;seedNativeRhs\(r\);\}/,
    "one stop gate and one accepted-row index must own both independent writes");
});

test("accurate A2 encodes one gate, one wide term stage, and one ordered fold", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  let direct = 0, indirect = 0, groups = 0;
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const base = spgridSource(buffer, 128, 8 * 512);
  const source = { ...base, classDispatch: buffer(48, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE) };
  const device = {
    queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => { groups += 1; return {}; },
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, source,
    { dimensions: [16, 16, 16], rowCapacity: 128, finestCellWidth: 1 });
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { direct += 1; },
    dispatchWorkgroupsIndirect() { indirect += 1; }, end() {} } as unknown as GPUComputePassEncoder;
  const broker = new PassBroker({ beginComputePass: () => pass } as unknown as GPUCommandEncoder);
  const input = buffer(512), output = buffer(512), solverControl = buffer(64);
  cycle.accurateOperator.encode(broker, input, output, solverControl);
  broker.fence("A2 complete");
  assert.equal(cycle.accurateOperator.encodedDispatchCount, 4);
  assert.equal(direct, 1); assert.equal(indirect, 3);
  const mergedWorksets = buffer(4096);
  const mergedDispatch = buffer(84, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE);
  const mergedLayout = buffer(16);
  cycle.accurateOperator.encodeMergedBandWorkset(broker, input, output, solverControl,
    mergedWorksets, mergedDispatch, mergedLayout, 48);
  broker.fence("merged band complete");
  assert.equal(cycle.accurateOperator.encodedMergedBandDispatchCount, 3);
  assert.equal(direct, 1); assert.equal(indirect, 6,
    "the shell union stages direct and adjoint terms before its ordered row fold");
  const residualRhs = buffer(512), residual = buffer(512);
  assert.ok(cycle.accurateOperator.encodeGate);
  assert.ok(cycle.accurateOperator.encodeResidualBody);
  cycle.accurateOperator.encodeGate!(pass, input, residual, solverControl);
  broker.fence("A2 residual gate complete");
  cycle.accurateOperator.encodeResidualBody!(
    broker, input, residualRhs, residual, solverControl,
  );
  broker.fence("A2 residual complete");
  assert.equal(direct, 2); assert.equal(indirect, 9,
    "the fused residual keeps the three A2 body stages and adds no residual dispatch");
  const firstGroups = groups;
  cycle.accurateOperator.encode(broker, input, output, solverControl);
  cycle.accurateOperator.encodeMergedBandWorkset(broker, input, output, solverControl,
    mergedWorksets, mergedDispatch, mergedLayout, 48);
  cycle.accurateOperator.encodeGate!(pass, input, residual, solverControl);
  cycle.accurateOperator.encodeResidualBody!(
    broker, input, residualRhs, residual, solverControl,
  );
  broker.fence("A2 repeated");
  assert.equal(groups, firstGroups, "immutable A2 bind groups must be reused");
  cycle.destroy();
});

test("factor-1 inline accurate A2 defaults on and deletes only ordinary staged body dispatches", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffer = (size: number, usage = 31) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const encode = (factorOne: boolean, mode: "default" | "on" | "off") => {
    let direct = 0, indirect = 0;
    const device = {
      queue: { writeBuffer() {} },
      createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
      createShaderModule: () => ({}),
      createComputePipeline: ({ label }: { label: string }) =>
        ({ label, getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
    } as unknown as GPUDevice;
    const previous = process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2;
    if (mode === "on") process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2 = "1";
    else if (mode === "off") process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2 = "0";
    else delete process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2;
    let cycle: WebGPUOctreeSPGridVCycle;
    try {
      cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 128, 8 * 512), {
        dimensions: [16, 16, 16], rowCapacity: 128, finestCellWidth: 1,
        geometricAggregateTransfers: factorOne,
      });
    } finally {
      if (previous === undefined) delete process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2;
      else process.env.FLUID_OCTREE_FACTOR1_SERIAL_ACCURATE_A2 = previous;
    }
    const pass = {
      setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { direct += 1; },
      dispatchWorkgroupsIndirect() { indirect += 1; }, end() {},
    } as unknown as GPUComputePassEncoder;
    const broker = new PassBroker({
      beginComputePass: () => pass,
    } as unknown as GPUCommandEncoder);
    const input = buffer(512), output = buffer(512), control = buffer(64);
    cycle.accurateOperator.encode(broker, input, output, control);
    broker.fence("ordinary apply complete");
    const ordinary = { direct, indirect };
    const beforeResidual = { direct, indirect };
    cycle.accurateOperator.encodeGate!(pass, input, output, control);
    broker.fence("residual gate complete");
    cycle.accurateOperator.encodeResidualBody!(
      broker, input, buffer(512), output, control,
    );
    broker.fence("residual body complete");
    const result = {
      encoded: cycle.accurateOperator.encodedDispatchCount,
      residualEncoded: cycle.accurateOperator.encodedResidualDispatchCount,
      ordinary,
      residual: {
        direct: direct - beforeResidual.direct,
        indirect: indirect - beforeResidual.indirect,
      },
    };
    cycle.destroy();
    return result;
  };
  assert.deepEqual(encode(true, "default"), {
    encoded: 2, residualEncoded: 4,
    ordinary: { direct: 1, indirect: 1 },
    residual: { direct: 1, indirect: 3 },
  });
  assert.deepEqual(encode(true, "on"), encode(true, "default"),
    "explicit enablement must retain the shipping factor-1 graph");
  assert.equal(encode(false, "on").encoded, 4,
    "factor-4/8 cannot select the factor-1 path");
  assert.equal(encode(true, "off").encoded, 4,
    "the explicit kill switch restores staged ordinary A2");
});

test("factor-1 inline merged-band A2 defaults on with an isolated staged fallback", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffer = (size: number, usage = 31) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const encode = (factorOne: boolean, mode: "default" | "on" | "off") => {
    let indirect = 0;
    const device = {
      queue: { writeBuffer() {} },
      createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
      createShaderModule: () => ({}),
      createComputePipeline: ({ label }: { label: string }) =>
        ({ label, getBindGroupLayout: () => ({}) }),
      createBindGroup: () => ({}),
    } as unknown as GPUDevice;
    const previous = process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2;
    if (mode === "on") process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2 = "1";
    else if (mode === "off") process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2 = "0";
    else delete process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2;
    let cycle: WebGPUOctreeSPGridVCycle;
    try {
      cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 128, 8 * 512), {
        dimensions: [16, 16, 16], rowCapacity: 128, finestCellWidth: 1,
        geometricAggregateTransfers: factorOne,
      });
    } finally {
      if (previous === undefined) delete process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2;
      else process.env.FLUID_OCTREE_FACTOR1_INLINE_MERGED_BAND_A2 = previous;
    }
    const pass = {
      setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {},
      dispatchWorkgroupsIndirect() { indirect += 1; }, end() {},
    } as unknown as GPUComputePassEncoder;
    const broker = new PassBroker({
      beginComputePass: () => pass,
    } as unknown as GPUCommandEncoder);
    cycle.accurateOperator.encodeMergedBandWorkset(
      broker, buffer(512), buffer(512), buffer(64), buffer(4096),
      buffer(84, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE), buffer(16), 48,
    );
    broker.fence("merged-band apply complete");
    const result = {
      encoded: cycle.accurateOperator.encodedMergedBandDispatchCount,
      indirect,
    };
    cycle.destroy();
    return result;
  };
  assert.deepEqual(encode(true, "default"), { encoded: 1, indirect: 1 });
  assert.deepEqual(encode(true, "on"), { encoded: 1, indirect: 1 });
  assert.deepEqual(encode(true, "off"), { encoded: 3, indirect: 3 });
  assert.deepEqual(encode(false, "on"), { encoded: 3, indirect: 3 },
    "factor-4/8 cannot select the factor-1 path");
});

test("compiled-image inline A2 is explicit-on, factor-1-only, and keeps both one-dispatch graphs", () => {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const buffer = (size: number, usage = 31) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const run = (factorOne: boolean, enabled: boolean) => {
    const selected: string[] = [];
    const groupBindings = new Map<string, number[]>();
    const device = {
      queue: { writeBuffer() {} },
      createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
      createShaderModule: () => ({}),
      createComputePipeline: ({ label }: { label: string }) =>
        ({ label, getBindGroupLayout: () => ({}) }),
      createBindGroup: ({ label, entries }: GPUBindGroupDescriptor) => {
        groupBindings.set(String(label), Array.from(entries, (entry) => entry.binding));
        return {};
      },
    } as unknown as GPUDevice;
    const previous = process.env.FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2;
    if (enabled) process.env.FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2 = "1";
    else delete process.env.FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2;
    let cycle: WebGPUOctreeSPGridVCycle;
    try {
      cycle = new WebGPUOctreeSPGridVCycle(
        device, spgridSource(buffer, 128, 8 * 512), {
          dimensions: [16, 16, 16], rowCapacity: 128, finestCellWidth: 1,
          geometricAggregateTransfers: factorOne,
        },
      );
    } finally {
      if (previous === undefined) delete process.env.FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2;
      else process.env.FLUID_OCTREE_FACTOR1_COMPILED_INLINE_A2 = previous;
    }
    const pass = {
      setPipeline(pipeline: { label: string }) { selected.push(pipeline.label); },
      setBindGroup() {}, dispatchWorkgroups() {}, dispatchWorkgroupsIndirect() {}, end() {},
    } as unknown as GPUComputePassEncoder;
    const broker = new PassBroker({
      beginComputePass: () => pass,
    } as unknown as GPUCommandEncoder);
    const input = buffer(512), output = buffer(512), control = buffer(64);
    cycle.accurateOperator.encode(broker, input, output, control);
    cycle.accurateOperator.encodeMergedBandWorkset(
      broker, input, output, control, buffer(4096),
      buffer(84, GPUBufferUsage.INDIRECT | GPUBufferUsage.STORAGE), buffer(16), 48,
    );
    broker.fence("compiled inline applies complete");
    const result = {
      selected,
      ordinary: cycle.accurateOperator.encodedDispatchCount,
      merged: cycle.accurateOperator.encodedMergedBandDispatchCount,
      ordinaryBindings: groupBindings.get(
        "SPGrid Section 6.3 · compiled-image accepted row union bindings",
      ),
      mergedBindings: groupBindings.get(
        "SPGrid Section 6.3 · compiled-image inline merged-band bindings",
      ),
    };
    cycle.destroy();
    return result;
  };
  const control = run(true, false);
  assert.equal(control.selected.some((label) => label.includes("compiled-image")), false,
    "default inline A2 remains the current topology-chase control");
  const compiled = run(true, true);
  assert.equal(compiled.ordinary, 2);
  assert.equal(compiled.merged, 1);
  assert.ok(compiled.selected.includes(
    "SPGrid accurate A2 · compiled-image accepted row union"));
  assert.ok(compiled.selected.includes(
    "SPGrid Section 6.3 · compiled-image inline merged-band rows"));
  assert.deepEqual(compiled.ordinaryBindings, [0, 1, 2, 3, 4, 5, 10, 11, 13, 14]);
  assert.deepEqual(compiled.mergedBindings, [0, 1, 2, 3, 4, 5, 9, 10, 11, 13, 14]);
  assert.equal(run(false, true).selected.some((label) => label.includes("compiled-image")), false,
    "the explicit arm cannot escape factor one");
});

test("resolved-row persistent executor is absent at every production capacity", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  let writes = 0, dispatches = 0, persistentEntries = 0;
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() { writes += 1; } },
    createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: ({ label, entries }: { label?: string; entries: unknown[] }) => {
      if (label === "SPGrid persistent MGPCG bindings") persistentEntries = entries.length; return {};
    },
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 64, 8 * 256),
    { dimensions: [16, 16, 16], rowCapacity: 64, finestCellWidth: 1 });
  assert.equal("persistentMGPCG" in cycle, false);
  assert.equal(dispatches, 0); assert.equal(persistentEntries, 0);
  cycle.destroy();
});

test("large native pyramids expose only the section 6.3 operator", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = { limits: { maxStorageBufferBindingSize: Number.MAX_SAFE_INTEGER, maxBufferSize: Number.MAX_SAFE_INTEGER },
    queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 8_193, 8),
    { dimensions: [16, 16, 16], rowCapacity: 8_193, finestCellWidth: 1 });
  assert.equal("persistentMGPCG" in cycle, false); cycle.destroy();
});

test("setup retires the prior live generation without unconditional full-buffer clears", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const created: Array<{ label?: string; size: number; usage: number; buffer: GPUBuffer }> = [];
  const buffer = (size: number, usage = 31, label?: string) => ({ size, usage, label, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} },
    createBuffer: ({ label, size, usage }: { label?: string; size: number; usage: number }) => {
      const gpuBuffer = buffer(size, usage, label); created.push({ label, size, usage, buffer: gpuBuffer }); return gpuBuffer;
    }, createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 32, 256),
    { dimensions: [8, 8, 8], rowCapacity: 32, finestCellWidth: 1 });
  const events: string[] = []; let dispatches = 0, current = "";
  const encoder = {
    clearBuffer() { throw new Error("warm SPGrid setup must not clear full buffers"); },
    beginComputePass() { events.push("begin"); return {
      setPipeline(pipeline: { label: string }) { current = pipeline.label.replace("SPGrid V-cycle · ", ""); }, setBindGroup() {},
      dispatchWorkgroups() { dispatches += 1; events.push(`direct:${current}`); },
      dispatchWorkgroupsIndirect(source: GPUBuffer, offset: number) {
        dispatches += 1; events.push(`indirect:${current}:${offset}:${String((source as GPUBuffer & { label?: string }).label)}`);
      },
      end() { events.push("end"); } }; },
    copyBufferToBuffer(source: GPUBuffer, sourceOffset: number, destination: GPUBuffer, destinationOffset: number, size: number) {
      assert.equal(sourceOffset, 0); assert.equal(destinationOffset, 0);
      assert.equal(size, cycle.plan.dispatchBytes);
      events.push(`copy:${String((source as GPUBuffer & { label?: string }).label)}->${String((destination as GPUBuffer & { label?: string }).label)}`);
    },
  } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(128), correction: buffer(128) };
  const broker = new PassBroker(encoder);
  cycle.encodeSetup(broker, input);
  assert.equal(dispatches, cycle.encodedSetupDispatchCount,
    "cold setup owns exact page planning plus transactional candidate publication");
  // planL1CaptureDelta is now the page-parallel row scan and
  // reduceL1CaptureDelta the work-list fold that used to be its lane-0 tail, so
  // the capture prologue is three dispatches rather than two.
  assert.deepEqual(events.slice(0, 6), ["begin", "direct:beginL1CapturePlan", "direct:planL1CaptureDelta",
    "direct:reduceL1CaptureDelta", "direct:probeCandidateSkip", "direct:applyCandidateSkip"]);
  assert.ok(events.some((event) => event === "direct:validateCandidateHierarchy"));
  assert.ok(events.some((event) => event === "direct:commitChangedL1"));
  assert.ok(events.some((event) => event === "direct:finalizeL1CapturePublication"));
  assert.ok(events.some((event) => event === "direct:commitCandidateLevels"));
  assert.ok(events.some((event) => event === "direct:finalizeLifecycle"));
  assert.deepEqual(events.slice(-2), ["end",
    "copy:SPGrid worklist counts and published dispatches->SPGrid live indirect dispatches"]);
  const metadata = created.find((entry) => entry.label === "SPGrid worklist counts and published dispatches")!;
  const indirect = created.find((entry) => entry.label === "SPGrid live indirect dispatches")!;
  assert.equal(metadata.usage & GPUBufferUsage.COPY_SRC, GPUBufferUsage.COPY_SRC);
  assert.equal(metadata.usage & GPUBufferUsage.INDIRECT, 0, "writable metadata must never be an indirect source");
  assert.equal(indirect.usage,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT);
  assert.notEqual(indirect.usage & GPUBufferUsage.STORAGE, 0,
    "the separate gate pass must be able to zero future correction records");
  assert.equal(created.some((entry) => entry.label?.includes("topology-reuse setup dispatches")), false,
    "the deleted setup-dispatch buffers must not be allocated");
  const beforeCapture = events.length;
  cycle.encodeCapture(broker);
  broker.fence("capture complete");
  const recurringCapture = events.slice(beforeCapture);
  assert.deepEqual(recurringCapture, ["begin", "direct:beginL1CapturePlan", "direct:planL1CaptureDelta",
    "direct:reduceL1CaptureDelta", "end"],
  "warm capture plans exact changed pages without mutating the published hierarchy");
  cycle.destroy();
});

test("captured L1 deltas rebuild only affected sparse-level suffixes", () => {
  assert.match(octreeSPGridVCycleShader,
    /fn markDirtyFrom\(first:u32,flags:u32,firstRow:u32,rowEnd:u32\)[\s\S]*for\(var l=first;l<levels\(\);l\+=1u\)[\s\S]*deltaAt\(l,2u\)[\s\S]*deltaAt\(l,3u\)/,
    "a changed L1 contribution publishes an affected-row record for its dependent coarse-level suffix");
  // The candidate hierarchy is now a chain of data-parallel dispatches whose
  // ordering is carried by dispatch boundaries, not by statement order in one
  // invocation. Only the two genuinely order-defining phases keep a serial
  // owner, and each owns one level rather than the whole hierarchy.
  const candidatePhases = ["clearCandidateLevels", "buildCandidateLevelSets", "detectCandidateGhosts",
    "insertCandidateGhosts", "buildCandidateLevelDeltas", "countCandidateTransfers",
    "scanCandidateTransfers", "writeCandidateTransfers", "linkCandidateParentChains",
    "markCandidateBrickOccupancy",
    "rankCandidateBricks", "scatterCandidateRankedSlots", "markCandidatePageOccupancy",
    "compactCandidatePages", "linkCandidatePageNeighbours", "buildCandidateStencils",
    "publishCandidateSpectralBounds"];
  for (const phase of candidatePhases) {
    assert.match(octreeSPGridVCycleShader, new RegExp(`@compute @workgroup_size\\(\\d+\\)[\\s\\S]{0,80}fn ${phase}\\(`),
      `${phase} must be its own dispatched phase`);
    assert.ok(phase in OCTREE_SPGRID_VCYCLE_BINDINGS, `${phase} must publish an exact bind group`);
  }
  const candidateOrder = candidatePhases.map((phase) =>
    (WebGPUOctreeSPGridVCycle.prototype as unknown as { encodeSetupCandidate: Function })
      .encodeSetupCandidate.toString().indexOf(`"${phase}"`));
  assert.deepEqual(candidateOrder, [...candidateOrder].sort((a, b) => a - b),
    "the encoded phase order must match the derived dependency order");
  assert.ok(candidateOrder.every((index) => index > 0), "every candidate phase must be encoded");
  for (const parallel of ["clearCandidateLevels", "detectCandidateGhosts", "countCandidateTransfers",
    "writeCandidateTransfers", "markCandidateBrickOccupancy",
    "scatterCandidateRankedSlots", "markCandidatePageOccupancy", "linkCandidatePageNeighbours",
    "buildCandidateStencils"]) {
    assert.match(octreeSPGridVCycleShader, new RegExp(`@compute @workgroup_size\\(64\\)[\\s\\S]{0,80}fn ${parallel}\\(`),
      `${parallel} must iterate a real work domain, not one invocation`);
  }
  assert.doesNotMatch(octreeSPGridVCycleShader,
    /fn rebuildCandidateLevelSetFor|fn rebuildCandidateDirectoryFor|fn rebuildCandidatePageWorksetFor|fn rebuildCandidateStencilFor|fn publishCandidateSpectralBoundFor/,
    "the serialized per-level rebuild helpers must be deleted");
  assert.match(octreeSPGridVCycleShader, /fn rebuildCandidateGhostsFor\(r:u32\)/,
    "the retained ghost predicate must own exactly one row, not a whole level");
  assert.match(octreeSPGridVCycleShader,
    /fn buildCandidateStencils[\s\S]*if\(!stencilDirty\(l\)\|\|i>=selectedCount\(l\)\)\{continue;\}/,
    "rediscretized stencil initialization must scale with live slots, not sparse capacity");
  const levelSetBuild = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn clearCandidateLevels"),
    octreeSPGridVCycleShader.indexOf("fn rebuildCandidateTransferFor"),
  );
  assert.doesNotMatch(levelSetBuild, /for\(var c=0u;c<STATE_CHANNELS/,
    "topology identity reset must not clear solver payload channels capacity-wide");
  assert.match(octreeSPGridVCycleShader,
    /fn transferLive\(l:u32\)->bool\{return l\+1u<levels\(\)&&\(topologyDirty\(l\)\|\|topologyDirty\(l\+1u\)\);\}/,
    "the retained fine endpoint transfer must rebuild with the changed coarse suffix");
  // The predicate is broadcast through `uniformWord` before the early return.
  // That is not indirection for its own sake: the body goes on to take
  // workgroup barriers, so every lane has to agree on whether it returns, and
  // a bare `if(!transferLive(l))` would make that exit non-uniform.
  assert.match(octreeSPGridVCycleShader,
    /fn rebuildCandidateTransferFor[\s\S]*let live=uniformWord\(lane,select\(0u,1u,transferLive\(l\)\)\);\s*if\(live==0u\)\{return;\}/,
    "the ordered transfer owner must consume the shared dirty predicate, uniformly");
  // The row scan is page-parallel: one workgroup per work item, one lane per
  // page row. Pinning the workgroup_id/lane row address keeps a future edit
  // from folding it back into the single-workgroup loop it was split out of,
  // which cost 6.40 ms/advance on 64 of the machine's 98,304 threads.
  //
  // That mapping is only complete because a capture page holds exactly as many
  // rows as the scan has lanes. Widening CAPTURE_PAGE_ROWS without widening the
  // workgroup would silently leave the tail of every page unvalidated, so pin
  // the two together.
  assert.match(octreeSPGridVCycleShader, /const CAPTURE_PAGE_ROWS=64u;/,
    "the page-parallel row scan addresses row begin+lane across a 64-lane workgroup");
  assert.match(octreeSPGridVCycleShader,
    /@workgroup_size\(64\) fn planL1CaptureDelta\(@builtin\(workgroup_id\)[\s\S]*let r=begin\+lane;[\s\S]*validSection63Row\(r\)[\s\S]*deltaOldRow\(r\)[\s\S]*var topologyChanged=true/,
    "compact producer dirty rows directly invalidate their dependent sparse suffix");
  assert.match(octreeSPGridVCycleShader,
    /commitChangedL1[\s\S]*capturePageStamp\(page\)!=generation[\s\S]*capturedGeometry\[r\]=sourceRowGeometry\(r\)/,
    "only exact changed pages are eligible for fixed-geometry commit");
  // The work-list traversal and its finalization moved from planL1CaptureDelta's
  // lane-0 tail into reduceL1CaptureDelta unchanged, so the invariant is now
  // pinned by that name. Membership is still decided by the work list, never by
  // a page stamp: two captures can share one generation, and a stale record
  // must never be mistaken for this capture's.
  assert.match(octreeSPGridVCycleShader,
    /fn reduceL1CaptureDelta[\s\S]*for\(var work=lane;work<workCount;work\+=64u\)[\s\S]*captureWorkUnique\(work,page\)[\s\S]*captureLaneChanged\[lane\][\s\S]*validatedPages=totalChanged[\s\S]*readyGeneration=generation/,
    "capture planning must reduce only compact producer-stamped pages before candidate setup begins");
  assert.match(octreeSPGridVCycleShader,
    /finalizeL1CapturePublication[\s\S]*for\(var work=lane;work<workCount;work\+=64u\)[\s\S]*copyStamp!=generation[\s\S]*totalCopied!=capturePages\.changedPages[\s\S]*publishedGeneration=generation/,
    "the capture publication must fail closed unless copied and changed-page counts agree");
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn validateL1CapturePages|fn finalizeL1CapturePlan/,
    "the recurring all-page compare/reduce kernels must be deleted");
  assert.match(octreeSPGridVCycleShader,
    /fn deltaDirtyCountWord\(\)->u32\{return select\(5u,6u,\(p\.delta\.w&0x80000000u\)!=0u\);\}[\s\S]*fn captureWorkCount\(\)->u32\{return select\(deltaControl\(deltaDirtyCountWord\(\)\),captureExpectedPages\(\),captureBootstrap\(\)\);\}/,
    "only explicit cold bootstrap may enumerate every page");
  const captureTransaction = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn beginL1CapturePlan"),
    octreeSPGridVCycleShader.indexOf("fn clearCorrection"));
  assert.doesNotMatch(captureTransaction, /atomic(?:Add|CompareExchange|Or|Store|Load)/,
    "exclusive page records and deterministic singleton scans replace recurring global counters");
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeCapture.toString(), /copyBufferToBuffer/,
    "cold and warm capture must use the identical exact page transaction, never a capacity-copy fallback");
  const candidateSetup = (WebGPUOctreeSPGridVCycle.prototype as unknown as {
    encodeSetupCandidate: Function;
  }).encodeSetupCandidate.toString();
  const readyCommit = WebGPUOctreeSPGridVCycle.prototype.encodeReadySetupCommit.toString();
  assert.match(candidateSetup,
    /probeCandidateSkip[\s\S]*applyCandidateSkip[\s\S]*commitChangedL1[\s\S]*clearCandidateLevels[\s\S]*buildCandidateLevelSets[\s\S]*buildCandidateLevelDeltas[\s\S]*validateCandidateHierarchy/);
  assert.match(readyCommit,
    /commitChangedL1[\s\S]*finalizeL1CapturePublication[\s\S]*commitCandidateLevels[\s\S]*finalizeLifecycle[\s\S]*publishCommittedInputs/,
    "accepted L1 geometry and dependent levels publish only after the coupled ready gate");
  assert.doesNotMatch(candidateSetup, /dispatchWorkgroupsIndirect|setupIndirect|prepareSetup/,
    "candidate publication must not retain redundant per-level indirect commands");
  assert.match(readyCommit,
    /source\.classDispatch[\s\S]*dispatchWorkgroupsIndirect[\s\S]*ACCURATE_SOURCE_IMAGE_TRANSITION_RECORD_BYTES/,
    "accepted image compilation consumes exact GPU-authored live-row records");
  assert.match(readyCommit, /this\.run\(pass,\s*"finalizeLifecycle"/);
});

test("recurring SPGrid publication paths contain no synchronization atomics", () => {
  assert.match(octreeSPGridVCycleShader, /topology:array<u32>[\s\S]*state:array<u32>[\s\S]*dispatchMeta:array<u32>/);
  const capture = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn beginL1CapturePlan"),
    octreeSPGridVCycleShader.indexOf("fn clearCorrection"));
  const construction = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn cAt("),
    octreeSPGridVCycleShader.indexOf("fn validateCandidateHierarchy"));
  const restriction = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn restrictAndGhostAccumulate"),
    octreeSPGridVCycleShader.indexOf("fn exactBottom"));
  for (const [name, source] of Object.entries({ capture, construction, restriction })) {
    assert.doesNotMatch(source, /\batomic(?:Add|CompareExchangeWeak|Load|Or|Store)\b|atomicAddF/,
      `${name} must be exclusively owned or deterministically reduced`);
  }
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn rebuildHierarchy/,
    "the serial whole-suffix rebuild authority must be deleted");
  assert.match(octreeSPGridVCycleShader, /fn commitCandidateLevelAt[\s\S]*stencilDirty\(l\)[\s\S]*topologyChanged/,
    "candidate publication commits exact topology or stencil payloads only after validation");
  for (const deleted of ["retireSlots", "retireLevelRows", "resetLevelMetadata", "resetTransferMetadata",
    "emitLevelCells", "emitLevelGhostAliases", "buildTransfers", "clearLevelBrickDirectory",
    "markBrickDirectory", "prefixBrickDirectory", "fillBrickDirectory", "validateBrickDirectory",
    "publishBrickDirectory", "buildLevelStencil", "ensureDiagonal", "finalizeIndirect"]) {
    assert.doesNotMatch(octreeSPGridVCycleShader, new RegExp(`fn ${deleted}\\b`),
      `${deleted} backing code must be deleted`);
    assert.equal(deleted in OCTREE_SPGRID_VCYCLE_BINDINGS, false, `${deleted} binding must be deleted`);
  }
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeSetup.toString(),
    /retireSlots|emitLevelCells|buildTransfers|markBrickDirectory|finalizeIndirect|clearBuffer/,
    "host setup retains no legacy staged or full-capacity path");
});

test("failed level publication is terminal and has no full-capacity recovery fallback", () => {
  const candidateValidation = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn validateCandidateHierarchy"),
    octreeSPGridVCycleShader.indexOf("fn commitCandidateLevelAt"),
  );
  assert.match(candidateValidation, /captureReport\(OVERFLOW\)/,
    "an inactive hierarchy reject must poison its coupled candidate receipt");
  assert.doesNotMatch(candidateValidation, /reportAt|atomicOr\(&control/,
    "an inactive hierarchy reject must never contaminate the live MGPCG control");
  assert.match(octreeSPGridVCycleShader, /fn previousValid\(\)->bool/);
  assert.match(octreeSPGridVCycleShader,
    /fn validateCandidateHierarchy[\s\S]*captureReport\(OVERFLOW\)[\s\S]*fn commitCandidateLevelAt[\s\S]*if\(captureFailed\(\)[\s\S]*return/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn resetInvalidBuffers|arrayLength\(&topology\).*arrayLength\(&state\)/s);
  assert.match(octreeSPGridVCycleShader,
    /fn finalizeLifecycle[\s\S]*atomicLoad\(&control\[0\]\)==0u&&!captureFailed\(\)[\s\S]*dispatchMeta\[base\]=1u[\s\S]*Candidate failure leaves the previously published hierarchy byte-for-byte/);
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeSetup.toString(), /clearBuffer/,
    "successful recurring generations must not write full sparse capacity");
});

test("correction covers the shared L1/L2 pressure-row domain and uses live slot dispatches internally", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31, label?: string) => ({ size, usage, label, destroy() {} }) as unknown as GPUBuffer;
  const device = { queue: { writeBuffer() {} },
    createBuffer: ({ label, size, usage }: { label?: string; size: number; usage: number }) => buffer(size, usage, label),
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 32, 256),
    { dimensions: [8, 8, 8], rowCapacity: 32, finestCellWidth: 1 });
  const offsets: number[] = []; const sources = new Set<GPUBuffer>(); let direct = 0;
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() { direct += 1; },
    dispatchWorkgroupsIndirect(source: GPUBuffer, offset: number) { sources.add(source); offsets.push(offset); }, end() {} } as unknown as GPUComputePassEncoder;
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const input = { rowCount: buffer(64), solverControl: buffer(64), rhs: buffer(128), correction: buffer(128) };
  const broker = new PassBroker(encoder);
  cycle.encodeCorrection(broker, input);
  broker.fence("correction complete");
  // Aanjaneya et al. 2017, Section 4.3
  // (`docs/papers/aanjaneya-2017-power-liquids.txt`) assumes that L1 and L2
  // have exactly the same pressure variables. Clear, seed, and publish must
  // therefore cover accepted rows directly: level zero's sparse slot count
  // excludes valid native coarser rows on an adaptive octree.
  assert.equal(direct, 2,
    "only the convergence publisher and coarse tail are direct");
  assert.equal(offsets.length, cycle.encodedCorrectionDispatchCount - direct);
  assert.equal(sources.size, 1, "all correction work must consume the dedicated indirect buffer");
  assert.ok(offsets.includes(8), "level zero consumes its live slot record");
  assert.ok(offsets.every((offset) => offset % 48 === 8 || offset % 48 === 20),
  "kernels consume published slot or parent-slot records");
  assert.equal(offsets.filter((offset) => offset % 48 === 20).length, 2);
  assert.equal(offsets.filter((offset) => offset % 48 === 36).length, 0,
    "no page-local multi-phase smoother may bypass global dispatch ordering");
  cycle.destroy();
});

test("Dawn accepts the native sparse V-cycle shader", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for WGSL validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  device.pushErrorScope("validation");
  const shaderModule = device.createShaderModule({ code: octreeSPGridVCycleShader });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  for (const entryPoint of ["beginL1CapturePlan", "planL1CaptureDelta",
    "commitChangedL1", "finalizeL1CapturePublication",
    "probeCandidateSkip", "applyCandidateSkip", "publishCommittedInputs",
    "clearCandidateLevels", "buildCandidateLevelSets", "detectCandidateGhosts", "insertCandidateGhosts",
    "buildCandidateLevelDeltas", "countCandidateTransfers", "scanCandidateTransfers",
    "writeCandidateTransfers", "linkCandidateParentChains",
    "markCandidateBrickOccupancy", "rankCandidateBricks",
    "scatterCandidateRankedSlots", "markCandidatePageOccupancy", "compactCandidatePages",
    "linkCandidatePageNeighbours", "buildCandidateStencils", "publishCandidateSpectralBounds",
    "validateCandidateHierarchy", "commitCandidateLevels",
    "finalizeLifecycle", "prepareCorrectionDispatches", "clearCorrection", "zeroVectors", "seedRhs",
    "seedRhsAndClearCorrection",
    "restrictAndGhostAccumulate", "coarseVcycleTail", "exactBottom",
    "smoothChebyshevAtoB0", "smoothChebyshevBtoA0",
    "smoothChebyshevAtoB1", "smoothChebyshevBtoA1",
    "smoothChebyshevAtoB2", "smoothChebyshevBtoA2",
    "smoothChebyshevAtoB3", "smoothChebyshevBtoA3",
    "prolongAndGhostPropagate", "publish"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message); device.destroy();
});

test("Dawn accepts the four class-specialized accurate operator and convergence gate", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for accurate operator validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter(); assert.ok(adapter);
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  for (const [code, entryPoints] of [
    [octreeSPGridAccurateOperatorShader, ["applyRegularInterior",
      "applyTransitionInterior", "applyPhysicalBoundary", "applyTransitionBoundary",
      "applyMergedBand", "applyAcceptedUnion",
      "applyCompiledMergedBand", "applyCompiledAcceptedUnion",
      "stageAcceptedUnionTerms", "stageMergedBandTerms",
      "stageAcceptedUnionAdjoints", "stageMergedBandAdjoints",
      "finalizeStagedUnionRows", "buildAccurateOperatorRows"]],
    [octreeSPGridAccurateDispatchGateShader, ["prepareAccurateDispatches"]],
  ] as const) {
    const shaderModule = device.createShaderModule({ code });
    const info = await shaderModule.getCompilationInfo();
    assert.deepEqual(info.messages.filter((message) => message.type === "error")
      .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
    for (const entryPoint of entryPoints) {
      device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
    }
  }
  device.destroy();
});

test("Dawn null compiles the accurate operator module", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for null WGSL compilation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const adapter = await dawn.create(["backend=null"]).requestAdapter();
  assert.ok(adapter);
  const device = await adapter.requestDevice();
  const module = device.createShaderModule({ code: octreeSPGridAccurateOperatorShader });
  const info = await module.getCompilationInfo();
  assert.deepEqual(info.messages.filter((message) => message.type === "error")
    .map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  device.destroy();
});

test("Dawn constructs every production native V-cycle setup bind group", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for production binding validation",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const rowCount = device.createBuffer({ size: 64, usage: storage });
  const control = device.createBuffer({ size: 64, usage: storage });
  const source = spgridSource((size, requestedUsage = storage) =>
    device.createBuffer({ size, usage: requestedUsage }), 4, 32);
  const cycle = new WebGPUOctreeSPGridVCycle(device, source,
    { dimensions: [4, 4, 4], rowCapacity: 4, finestCellWidth: 1 });
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  cycle.encodeSetup(broker, { solverControl: control, rowCount });
  broker.finish();
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  cycle.destroy();
  for (const buffer of [rowCount, control, source.params, source.rows, source.diagonals,
    source.rowGeometry, source.control, source.invalidRows.buffer, source.rowDelta.rows]) buffer.destroy();
  device.destroy();
});

test("Dawn compact-arena setup replaces stale hierarchy and changed capture recopies live L1", {
  skip: !process.env.WEBGPU_NODE_MODULE && "set WEBGPU_NODE_MODULE for GPU lifecycle checks",
}, async () => {
  const dawn = await import(pathToFileURL(process.env.WEBGPU_NODE_MODULE!).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, dawn.globals);
  const nativeGpu = dawn.create([`backend=${process.env.WEBGPU_BACKEND ?? "metal"}`]);
  const adapter = await nativeGpu.requestAdapter();
  assert.ok(adapter); const device = await adapter.requestDevice({ requiredLimits: { maxStorageBuffersPerShaderStage: 10 } });
  const storage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const source = spgridSource((size, requestedUsage = storage) =>
    device.createBuffer({ size, usage: requestedUsage }), 4, 32);
  const geometryWords = new Uint32Array(4 * 4);
  for (let row = 0; row < 4; row += 1) {
    geometryWords[row * 4] = row; geometryWords[row * 4 + 1] = 1;
  }
  const resolvedWords = new Uint32Array(source.plan.arenaBytes / 4).fill(0xffffffff);
  for (let row = 0; row < 4; row += 1) {
    const base = row * source.plan.rowStrideWords;
    resolvedWords[base] = 0; resolvedWords[base + 1] = 0; resolvedWords[base + 2] = 1;
  }
  device.queue.writeBuffer(source.params, 0, new Uint32Array([
    4, source.plan.rowStrideWords, source.plan.maximumNeighborSlots, 1,
  ]));
  device.queue.writeBuffer(source.rowGeometry, 0, geometryWords);
  device.queue.writeBuffer(source.rows, 0, resolvedWords);
  device.queue.writeBuffer(source.diagonals, 0, new Float32Array([2, 2, 2, 2]));
  // The production finest M1 consumes the accepted L2 boundary coefficients,
  // as required by Aanjaneya et al. (2017), Section 4.3. Give this lifecycle
  // fixture a valid identity boundary operator instead of relying on the old
  // synthetic-stencil fallback: one positive diagonal and no coupled faces.
  const acceptedCoefficients = new Float32Array(2 * 4 * 19);
  for (let row = 0; row < 4; row += 1) acceptedCoefficients[row * 19] = 1;
  device.queue.writeBuffer(source.coefficients, 0, acceptedCoefficients);
  const topologyMetrics = new Uint32Array(4 * 4);
  for (let row = 0; row < 4; row += 1) topologyMetrics[row * 4 + 1] = 0x80000000;
  device.queue.writeBuffer(source.topologyMetrics, 0, topologyMetrics);
  const catalogCoefficients = new Float32Array(1_608 * 19);
  catalogCoefficients[0] = 1;
  device.queue.writeBuffer(source.catalogCoefficients, 0, catalogCoefficients);
  device.queue.writeBuffer(source.control, 0,
    new Uint32Array([0, 0xffffffff, 4, 1, 0, 0]));
  const deltaWords = new Uint32Array(source.rowDelta.rows.size / 4);
  deltaWords.set([4, 0, 0, 4, 0, 4, 4, 1, 0x52444c54, 1, 1, 1, 1, 1, 1, 1]);
  deltaWords.set([0, 1, 2, 3], source.rowDelta.dirtyRowsOffsetWords);
  device.queue.writeBuffer(source.rowDelta.rows, 0, deltaWords);
  const cycle = new WebGPUOctreeSPGridVCycle(device, source,
    { dimensions: [4, 4, 4], rowCapacity: 4, finestCellWidth: 1 });
  type Internals = { dispatchMeta: GPUBuffer; indirectDispatch: GPUBuffer;
    state: GPUBuffer; capturedGeometry: GPUBuffer; capturePageState: GPUBuffer;
    levelDelta: GPUBuffer; candidateDispatch: GPUBuffer };
  const internal = cycle as unknown as Internals;
  const publication = new Uint32Array(cycle.plan.dispatchBytes / 4);
  for (let level = 0; level < cycle.plan.levelCount; level += 1) {
    const base = level * 12; publication[base] = 1;
    publication.set([1, 1, 1], base + 2); publication.set([1, 1, 1], base + 5);
  }
  const lifecycle = cycle.plan.levelCount * 12; publication[lifecycle] = 1; publication[lifecycle + 1] = 4;
  device.queue.writeBuffer(internal.dispatchMeta, 0, publication);
  device.queue.writeBuffer(internal.indirectDispatch, 0, publication);
  device.queue.writeBuffer(internal.state, 0, new Uint32Array([0x1234abcd]));
  device.queue.writeBuffer(internal.capturedGeometry, 0, new Uint32Array([99]));
  const countWords = new Uint32Array(16); countWords[0] = 4; countWords[7] = 1;
  const rowCount = device.createBuffer({ size: 64, usage: storage }); device.queue.writeBuffer(rowCount, 0, countWords);
  const control = device.createBuffer({ size: 64, usage: storage });
  const stateRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const publicationRead = device.createBuffer({ size: cycle.plan.dispatchBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const coldCaptureRead = device.createBuffer({ size: cycle.plan.capturePageStateBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const coldControlRead = device.createBuffer({ size: 64,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const coldDeltaRead = device.createBuffer({ size: cycle.plan.levelDeltaBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const coldCandidateDispatchRead = device.createBuffer({ size: cycle.plan.dispatchBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.pushErrorScope("validation"); let encoder = device.createCommandEncoder();
  cycle.encodeSetup(new PassBroker(encoder), { solverControl: control, rowCount });
  encoder.copyBufferToBuffer(internal.state, 0, stateRead, 0, 4);
  encoder.copyBufferToBuffer(internal.dispatchMeta, 0, publicationRead, 0, cycle.plan.dispatchBytes);
  encoder.copyBufferToBuffer(internal.capturePageState, 0, coldCaptureRead, 0,
    cycle.plan.capturePageStateBytes);
  encoder.copyBufferToBuffer(control, 0, coldControlRead, 0, 64);
  encoder.copyBufferToBuffer(internal.levelDelta, 0, coldDeltaRead, 0,
    cycle.plan.levelDeltaBytes);
  encoder.copyBufferToBuffer(internal.candidateDispatch, 0, coldCandidateDispatchRead, 0,
    cycle.plan.dispatchBytes);
  device.queue.submit([encoder.finish()]);
  await Promise.all([stateRead.mapAsync(GPUMapMode.READ), publicationRead.mapAsync(GPUMapMode.READ),
    coldCaptureRead.mapAsync(GPUMapMode.READ), coldControlRead.mapAsync(GPUMapMode.READ),
    coldDeltaRead.mapAsync(GPUMapMode.READ), coldCandidateDispatchRead.mapAsync(GPUMapMode.READ)]);
  let validationError = await device.popErrorScope(); assert.equal(validationError, null, validationError?.message);
  const coldCapture = new Uint32Array(coldCaptureRead.getMappedRange());
  const coldControl = new Uint32Array(coldControlRead.getMappedRange());
  const coldDelta = new Uint32Array(coldDeltaRead.getMappedRange());
  const coldCandidateDispatch = new Uint32Array(coldCandidateDispatchRead.getMappedRange());
  assert.notEqual(new Uint32Array(stateRead.getMappedRange())[0], 0x1234abcd,
    `a dirty cold setup must replace stale state; capture=${JSON.stringify([...coldCapture.slice(0, 12)])} control=${JSON.stringify([...coldControl])} delta=${JSON.stringify([...coldDelta])} candidate=${JSON.stringify([...coldCandidateDispatch])}`);
  const rebuiltPublication = new Uint32Array(publicationRead.getMappedRange());
  assert.equal(rebuiltPublication[lifecycle], 1);
  assert.equal(rebuiltPublication[lifecycle + 1], 4,
    "the compact hierarchy must publish the complete validated L1 row count");
  stateRead.unmap(); publicationRead.unmap(); coldCaptureRead.unmap(); coldControlRead.unmap();
  coldDeltaRead.unmap(); coldCandidateDispatchRead.unmap();

  // The same warm capture dispatches over the producer's dirty rows and
  // replaces the retained header publication.
  device.pushErrorScope("validation");
  deltaWords.fill(0);
  deltaWords.set([4, 4, 4, 0, 0, 1, 1, 2, 0x52444c54, 1, 1, 1, 1, 1, 1, 1]);
  deltaWords.set([1, 2, 3, 4], source.rowDelta.newToOldOffsetWords);
  deltaWords[source.rowDelta.dirtyRowsOffsetWords] = 0;
  device.queue.writeBuffer(source.rowDelta.rows, 0, deltaWords);
  device.queue.writeBuffer(source.control, 0,
    new Uint32Array([0, 0xffffffff, 4, 2, 0, 0]));
  geometryWords[0] = 4; device.queue.writeBuffer(source.rowGeometry, 0, geometryWords);
  const captureRead = device.createBuffer({ size: 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const captureStateRead = device.createBuffer({ size: cycle.plan.capturePageStateBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const controlRead = device.createBuffer({ size: 64, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder = device.createCommandEncoder(); const captureBroker = new PassBroker(encoder);
  cycle.encodeCapture(captureBroker);
  cycle.encodeSetup(captureBroker, { solverControl: control, rowCount });
  captureBroker.fence("capture complete");
  encoder.copyBufferToBuffer(internal.capturedGeometry, 0, captureRead, 0, 16);
  encoder.copyBufferToBuffer(internal.capturePageState, 0, captureStateRead, 0, cycle.plan.capturePageStateBytes);
  encoder.copyBufferToBuffer(control, 0, controlRead, 0, 64);
  device.queue.submit([encoder.finish()]);
  await Promise.all([captureRead.mapAsync(GPUMapMode.READ), captureStateRead.mapAsync(GPUMapMode.READ),
    controlRead.mapAsync(GPUMapMode.READ)]);
  const captureState = new Uint32Array(captureStateRead.getMappedRange());
  const controlState = new Uint32Array(controlRead.getMappedRange());
  assert.equal(captureState[8], 0, `warm capture failed: ${JSON.stringify([...captureState])}`);
  assert.equal(controlState[0], 0, `warm hierarchy failed with control flags ${controlState[0]}`);
  assert.equal(new Uint32Array(captureRead.getMappedRange())[0], 4,
    "an unproven generation must recapture its live L1 header");
  validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  captureRead.unmap(); captureStateRead.unmap(); controlRead.unmap(); cycle.destroy();
  for (const buffer of [rowCount, control, stateRead, publicationRead, coldCaptureRead,
    coldControlRead, coldDeltaRead, coldCandidateDispatchRead, captureRead,
    captureStateRead, controlRead, source.params, source.rows, source.diagonals,
    source.rowGeometry, source.control, source.invalidRows.buffer, source.rowDelta.rows]) buffer.destroy();
  device.destroy();
});
