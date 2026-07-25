import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import {
  OCTREE_SPGRID_VCYCLE_BINDINGS,
  SPGRID_CELL_FLAG,
  SPGRID_MAXIMUM_ROW_CAPACITY,
  WebGPUOctreeSPGridVCycle,
  buildSPGridBrickRankDirectory,
  buildSPGridParentGatherCSR,
  buildSPGridPyramidOracle,
  octreeSPGridPersistentMGPCGShader,
  octreeSPGridVCycleShader,
  planOctreeSPGridVCycle,
  lookupSPGridBrickRank,
  prolongSPGrid,
  restrictSPGrid,
  restrictSPGridParentGather,
  solveSPGridBottomLDLT,
  validateSPGridBrickRankDirectory,
} from "../lib/webgpu-octree-spgrid-vcycle";
import { PassBroker } from "../lib/webgpu-pass-broker";

const dot = (a: readonly number[], b: readonly number[]) => a.reduce((sum, value, index) => sum + value * b[index], 0);
const spgridDelta = (
  buffer: (size: number, usage?: number) => GPUBuffer,
  rowCapacity: number,
) => {
  const controlOffsetWords = 0;
  const newToOldOffsetWords = 16;
  const dirtyRowsOffsetWords = newToOldOffsetWords + rowCapacity;
  return {
    rows: buffer(4 * (dirtyRowsOffsetWords + rowCapacity)),
    rowCapacity,
    controlOffsetWords,
    newToOldOffsetWords,
    dirtyRowsOffsetWords,
  };
};
const spgridSource = (
  buffer: (size: number, usage?: number) => GPUBuffer,
  rowCapacity: number,
  entryBytes: number,
) => {
  return {
    leafHeaders: buffer(48 * rowCapacity),
    leafEntries: buffer(entryBytes),
    rowDelta: spgridDelta(buffer, rowCapacity),
  };
};

function reachableWGSLBindings(shader: string, entryPoint: string): number[] {
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
  assert.equal(narrow.levelStride, wide.levelStride);
  assert.equal(narrow.transferStride, 40_000);
  assert.ok(wide.levelCount > narrow.levelCount);
  assert.equal(wide.stateBytes, 25 * wide.levelCount * wide.levelStride * 4);
  const ui = planOctreeSPGridVCycle({ dimensions: [24, 18, 16], rowCapacity: 6_912 });
  assert.ok(ui.stateBytes <= 128 * 1024 * 1024);
  assert.ok(ui.topologyBytes <= 128 * 1024 * 1024);
  assert.equal(wide.dispatchBytes, wide.levelCount * 32 + 8);
  assert.equal(wide.capturePageCount, Math.ceil(wide.rowCapacity / 64));
  assert.equal(wide.capturePageStateBytes, (12 + 4 * wide.capturePageCount) * 4);
  assert.ok(!("cellCount" in wide));
  assert.throws(() => planOctreeSPGridVCycle({ dimensions: [16, 16, 16],
    rowCapacity: SPGRID_MAXIMUM_ROW_CAPACITY + 1 }), /bounded/);
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

test("V-cycle lookup is authoritative mask/popcount rank with no hash-probe fallback", () => {
  assert.match(octreeSPGridVCycleShader, /fn directoryLookup[\s\S]*countOneBits[\s\S]*fn find\(l:u32,q:vec3u\)->u32\{return directoryLookup\(l,q,true\);\}/);
  const persistentFind = octreeSPGridPersistentMGPCGShader.slice(octreeSPGridPersistentMGPCGShader.indexOf("fn find("),
    octreeSPGridPersistentMGPCGShader.indexOf("fn smoothable"));
  assert.match(persistentFind, /countOneBits/);
  assert.doesNotMatch(persistentFind, /probe|hash/);
  const publishedFind = octreeSPGridVCycleShader.slice(octreeSPGridVCycleShader.indexOf("fn directoryLookup("),
    octreeSPGridVCycleShader.indexOf("fn originOf("));
  assert.doesNotMatch(publishedFind, /probe|hash/);
  assert.doesNotMatch(octreeSPGridPersistentMGPCGShader, /for\(var probe/);
  assert.match(octreeSPGridVCycleShader, /fn rebuildCandidateDirectoryFor[\s\S]*candidateTopology\[record\]=generation/);
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
  assert.match(restriction, /parentHeadBase\(l\)\+coarse[\s\S]*storef\(RHS,l\+1u,coarse,sum\)/);
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
  assert.match(octreeSPGridVCycleShader, /clearCorrection[\s\S]*r<rows\(\)&&!stopped\(\)/,
    "post-convergence correction clears must be write-free");
  assert.match(octreeSPGridVCycleShader, /zeroVectors[\s\S]*i>=count\(l\)\|\|stopped\(\)/,
    "post-convergence sparse-level clears must be write-free");
  const bindings = [...octreeSPGridVCycleShader.matchAll(/@group\(0\) @binding\((\d+)\)/g)]
    .map((match) => Number(match[1]));
  assert.equal(new Set(bindings).size, bindings.length,
    "every WGSL binding must have exactly one module-scope declaration");
  assert.match(octreeSPGridVCycleShader, /cAppendTransfer\(l,fine,coarse,weight\)/);
  assert.match(octreeSPGridVCycleShader, /fn correctionTransfer/);
  assert.match(octreeSPGridVCycleShader,
    /cAppendTransfer[\s\S]*fineHeadBase\(l\)\+fine[\s\S]*fineCountBase\(l\)\+fine/);
  const correctionTransfer = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn correctionTransfer("),
    octreeSPGridVCycleShader.indexOf("fn restrictAndGhostAccumulate"),
  );
  assert.match(correctionTransfer,
    /first=topology\[fineHeadBase\(l\)\+fine\][\s\S]*record=first\+corner/);
  assert.doesNotMatch(correctionTransfer, /\bfind\(|\bdirectoryLookup\(/,
    "recurring prolongation must consume setup-published transfer indices directly");
  assert.match(octreeSPGridVCycleShader, /fn restrictAndGhostAccumulate/);
  assert.match(octreeSPGridVCycleShader, /fn prolongAndGhostPropagate/);
  assert.equal(octreeSPGridVCycleShader.match(/correctionTransfer\(l,fine,corner\)/g)?.length, 1,
    "fine-owned prolongation consumes the same immutable transfer rule used by setup");
  const prolong = octreeSPGridVCycleShader.slice(octreeSPGridVCycleShader.indexOf("fn prolongAndGhostPropagate"),
    octreeSPGridVCycleShader.indexOf("fn publish", octreeSPGridVCycleShader.indexOf("fn prolongAndGhostPropagate")));
  assert.doesNotMatch(prolong, /atomicAddF/,
    "one fine invocation owns its complete prolongation sum");
  assert.match(octreeSPGridVCycleShader, /bitcast<f32>\(topology\[transferWord\(l,record,2u\)\]\)\*residual/);
  assert.match(octreeSPGridVCycleShader, /transfer\.weight\*loadf\(A,l\+1u,transfer\.coarse\)/);
  assert.match(octreeSPGridVCycleShader, /const ACTIVE=1u;const GHOST=2u;const MG_ONLY=4u/);
  assert.match(octreeSPGridVCycleShader, /fn cMergeClass/);
  assert.match(octreeSPGridVCycleShader, /fn contactCoord/);
  assert.match(octreeSPGridVCycleShader, /fn cInsertOwned/);
  assert.match(octreeSPGridVCycleShader, /fn rebuildCandidateLevelSet/);
  assert.match(octreeSPGridVCycleShader, /const XYPP=9u.*const YZPP=17u/s,
    "the production stencil retains all twelve directed octree-edge contacts");
  assert.match(octreeSPGridVCycleShader,
    /const STATE_CHANNELS=25u/);
  const applied = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn applied("),
    octreeSPGridVCycleShader.indexOf("fn smoothable", octreeSPGridVCycleShader.indexOf("fn applied(")),
  );
  assert.match(applied,
    /for\(var k=0u;k<18u;k\+=1u\)[\s\S]*topology\[neighbourAt\(k,l,slot\)\]/,
    "recurring stencil application must gather setup-resolved slots directly");
  assert.doesNotMatch(applied, /\bfind\(|\bneighbour\(|\boffsetNeighbour\(/,
    "recurring stencil application must not resolve coordinates through a directory");
  assert.doesNotMatch(octreeSPGridVCycleShader, /select\([^\n]*insert\([^\n]*insertOwned/,
    "WGSL select eagerly evaluates both insertion paths");
  assert.match(octreeSPGridVCycleShader, /fn smoothable/);
  assert.match(octreeSPGridVCycleShader, /fn smoothAtoB/);
  assert.match(octreeSPGridVCycleShader, /let product=applied\(fine,A\)/,
    "restriction must consume the final pre-smoothing stencil without materializing AX or residual");
  assert.doesNotMatch(octreeSPGridVCycleShader,
    /\b(?:AX|RESIDUAL)\b|fn (?:applyA|applyB|jacobiAtoB|jacobiBtoA|formResidual)\b/,
    "the fused correction schedule must not retain legacy materialization channels or kernels");
  assert.match(octreeSPGridVCycleShader, /fn exactBottom/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /while\s*\([^)]*atomicLoad/, "no cross-workgroup spin barrier is permitted");
});

test("persistent shader preserves host-recorded V-cycle ghost residuals, Jacobi, adjoint transfers, and the uniform gate", () => {
  const gate = octreeSPGridPersistentMGPCGShader.indexOf("Uniform fail-closed gate");
  const firstArithmetic = octreeSPGridPersistentMGPCGShader.indexOf("storePV(PX,row,pressureSeed[row])");
  assert.ok(gate >= 0 && firstArithmetic > gate);
  assert.match(octreeSPGridPersistentMGPCGShader,
    /for\(var row=lid;row<n;row\+=PERSISTENT_LANES\)[\s\S]*h\.entryStart>arrayLength\(&entries\)\|\|h\.entryCount>arrayLength\(&entries\)-h\.entryStart[\s\S]*sync\(\);[\s\S]*if\(!failed\(\)\)/,
    "all lanes synchronize on structural validity before any solver arithmetic");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /const PERSISTENT_LANES=64u;[\s\S]*@compute @workgroup_size\(64\) fn persistentMGPCG/,
    "persistent row and slot strides exactly match the launched workgroup width");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /fn persistentRelaxJacobi[\s\S]*if\(!smoothable\(l,slot\)\)\{storef\(dst,l,slot,loadf\(src,l,slot\)\);return;\}[\s\S]*persistentApplied/);
  assert.match(octreeSPGridPersistentMGPCGShader,
    /fn persistentApplied[\s\S]*topology\[neighbourAt\(k,l,slot\)\]/,
    "the persistent solve must consume the same setup-resolved stencil slots");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /let residualValue=select\(-product,loadf\(RHS,l,fine\)-product,!ghost\)/,
    "ghost residuals retain the host-recorded V-cycle -A*x branch exactly");
  assert.equal(octreeSPGridPersistentMGPCGShader.match(/correctionTransfer\(l,fine,corner\)/g)?.length, 1,
    "prolongation consumes the setup-published transfer rule");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /parentHeadBase\(l\)\+coarse[\s\S]*transferWord\(l,record,2u\)[\s\S]*residualValue/);
  assert.doesNotMatch(octreeSPGridPersistentMGPCGShader,
    /\batomic(?:Add|And|CompareExchangeWeak|Exchange|Load|Max|Min|Or|Store|Sub|Xor)\s*\(|array<atomic|atomicAddF/,
    "one persistent workgroup uses ordinary storage, barriers, and fixed per-lane error records");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /topology:array<u32>[\s\S]*state:array<u32>[\s\S]*dispatchMeta:array<u32>[\s\S]*control:array<u32>/,
    "the deleted staged atomic ABI must not survive in persistent bindings");
  assert.match(octreeSPGridPersistentMGPCGShader, /transfer\.weight\*loadf\(A,l\+1u,transfer\.coarse\)/);
  assert.match(octreeSPGridPersistentMGPCGShader,
    /for\(var iteration=0u;iteration<12u;iteration\+=1u\)\{[\s\S]*persistentStop=select\(0u,1u,stopped\(\)\)[\s\S]*workgroupUniformLoad\(&persistentStop\)[\s\S]*if\(stop!=0u\)\{break;\}[\s\S]*persistentPCGIteration\(lid\)/,
    "the persistent device loop retains the immutable cap and exits on device-published convergence");
  assert.doesNotMatch(octreeSPGridPersistentMGPCGShader, /\bsolveActive\b/,
    "the deleted staged convergence mask must not return");
  assert.match(octreeSPGridPersistentMGPCGShader,
    /fn persistentPCGIteration\(lid:u32\)[\s\S]*if\(!stopped\(\)\)[\s\S]*sync\(\)/,
    "converged iterations must retain only the barrier-uniform synchronization skeleton");
  const bindings = [...octreeSPGridPersistentMGPCGShader.matchAll(/@group\(0\) @binding\((\d+)\)/g)];
  assert.equal(bindings.length, 11); assert.equal(new Set(bindings.map((match) => match[1])).size, 11);
});

test("every SPGrid auto-layout binds the complete reachable resource ABI", () => {
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.beginL1CapturePlan, [0, 3, 6, 13, 14, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.planL1CaptureDelta, [0, 1, 2, 3, 11, 12, 13, 14, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.commitChangedL1, [0, 1, 2, 3, 11, 12, 13, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.finalizeL1CapturePublication, [0, 3, 13, 18]);
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.buildCandidateLevelDeltas,
    [0, 3, 4, 5, 6, 11, 12, 14, 15, 16, 17],
    "the ordered candidate builder remains within the portable ten-storage-buffer limit");
  assert.equal("resetInvalidBuffers" in OCTREE_SPGRID_VCYCLE_BINDINGS, false,
    "the deleted all-capacity recovery path must not remain in the pipeline ABI");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.clearCorrection, [0, 3, 7, 9],
    "correction clearing observes the solver stop gate before writing output");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.zeroVectors, [0, 4, 5, 6, 7],
    "vector clearing observes the solver stop gate before touching sparse slots");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.seedRhs, [0, 1, 3, 4, 5, 7, 8],
    "native-level RHS seeding reads the captured leaf size and must bind headers");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.restrictAndGhostAccumulate, [0, 4, 5, 6, 7],
    "parent-owned restriction reads only compact transfer/state storage");
  assert.deepEqual(OCTREE_SPGRID_VCYCLE_BINDINGS.publish, [0, 1, 3, 4, 5, 7, 9],
    "native-level correction publication likewise reads the captured leaf size");
  for (const [entryPoint, bindings] of Object.entries(OCTREE_SPGRID_VCYCLE_BINDINGS)) {
    assert.equal(new Set(bindings).size, bindings.length, `${entryPoint} must not bind a resource twice`);
    assert.ok(bindings.filter((binding) => binding !== 0).length <= 10,
      `${entryPoint} exceeds the portable ten-storage-buffer stage budget`);
    assert.deepEqual([...bindings].sort((a, b) => a - b), reachableWGSLBindings(octreeSPGridVCycleShader, entryPoint),
      `${entryPoint} TS bind group must equal exact WGSL auto-layout reachability`);
  }
});

test("one correction uses one compute-pass transition, ordered dispatches, and cached descriptors", () => {
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
  assert.equal(passes - setupPasses, 1);
  assert.equal(dispatches - before, cycle.encodedCorrectionDispatchCount);
  assert.equal(cycle.encodedCorrectionDispatchCount, 33,
    "row-local smoothing and residual/restriction fusion removes six dispatches at each non-bottom level");
  assert.equal(cycle.encodedPassTransitionCount, 1);
  assert.equal(cycle.diagnostics.bottomOperation, "exact-single-cell");
  assert.equal(cycle.diagnostics.coarsestDegreesOfFreedom, 1);
  const firstGroups = groups;
  cycle.encodeCorrection(broker, input);
  broker.fence("correction complete");
  assert.equal(groups, firstGroups, "repeated solves allocate no new bind groups");
  cycle.destroy();
});

test("small native pyramids own a real immutable one-dispatch persistent executor", () => {
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
  assert.ok(cycle.persistentMGPCG);
  assert.equal(cycle.persistentMGPCG.maximumRowCapacity, 64,
    "the executor advertises the concrete hierarchy capacity it can address");
  const pass = { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups(x: number, y: number, z: number) {
    assert.deepEqual([x, y, z], [1, 1, 1]); dispatches += 1;
  }, end() {} } as unknown as GPUComputePassEncoder;
  const encoder = { beginComputePass: () => pass } as unknown as GPUCommandEncoder;
  const solve = { leafHeaders: buffer(48 * 64), leafEntries: buffer(8 * 256), rowCount: buffer(64),
    pressureIn: buffer(256), pressureOut: buffer(256), solverControl: buffer(64),
    boundarySmoothingIterations: 8, relativeTolerance: 1e-4 };
  const broker = new PassBroker(encoder);
  const before = writes; cycle.persistentMGPCG.encodeSolve(broker, solve);
  broker.fence("persistent solve complete");
  assert.equal(dispatches, 1); assert.equal(persistentEntries, 11,
    "one uniform plus exactly ten storage buffers is the requested portable ABI ceiling");
  assert.equal(writes, before + 1, "persistent configuration is uploaded exactly once");
  cycle.persistentMGPCG.encodeSolve(broker, solve); broker.fence("persistent solve complete");
  assert.equal(writes, before + 1);
  cycle.destroy();
});

test("large native pyramids expose no small-domain executor to the staged PCG lane", () => {
  Object.assign(globalThis, { GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 } });
  const buffer = (size: number, usage = 31) => ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = { limits: { maxStorageBufferBindingSize: Number.MAX_SAFE_INTEGER, maxBufferSize: Number.MAX_SAFE_INTEGER },
    queue: { writeBuffer() {} }, createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}), createComputePipeline: () => ({ getBindGroupLayout: () => ({}) }), createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, 8_193, 8),
    { dimensions: [16, 16, 16], rowCapacity: 8_193, finestCellWidth: 1 });
  assert.equal(cycle.persistentMGPCG, undefined); cycle.destroy();
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
  assert.deepEqual(events.slice(0, 4), ["begin", "direct:beginL1CapturePlan", "direct:planL1CaptureDelta",
    "direct:buildCandidateLevelDeltas"]);
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
  assert.equal(indirect.usage, GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT);
  assert.equal(indirect.usage & GPUBufferUsage.STORAGE, 0, "indirect arguments must never be writable storage");
  assert.equal(created.some((entry) => entry.label?.includes("topology-reuse setup dispatches")), false,
    "the deleted setup-dispatch buffers must not be allocated");
  const beforeCapture = events.length;
  cycle.encodeCapture(broker);
  broker.fence("capture complete");
  const recurringCapture = events.slice(beforeCapture);
  assert.deepEqual(recurringCapture, ["begin", "direct:beginL1CapturePlan", "direct:planL1CaptureDelta", "end"],
  "warm capture plans exact changed pages without mutating the published hierarchy");
  cycle.destroy();
});

test("captured L1 deltas rebuild only affected sparse-level suffixes", () => {
  assert.match(octreeSPGridVCycleShader,
    /fn markDirtyFrom\(first:u32,flags:u32,firstRow:u32,rowEnd:u32\)[\s\S]*for\(var l=first;l<levels\(\);l\+=1u\)[\s\S]*deltaAt\(l,2u\)[\s\S]*deltaAt\(l,3u\)/,
    "a changed L1 contribution publishes an affected-row record for its dependent coarse-level suffix");
  assert.match(octreeSPGridVCycleShader,
    /fn buildCandidateLevelDeltas\(\)[\s\S]*rebuildCandidateLevelSetFor\(l\)[\s\S]*rebuildCandidateTransferFor\(l\)[\s\S]*rebuildCandidateDirectoryFor\(l\)[\s\S]*rebuildCandidateStencilFor\(l\)/,
    "one singleton preserves the exact dependency order while each helper retains its per-level dirty predicate");
  assert.match(octreeSPGridVCycleShader,
    /rebuildCandidateStencilFor[\s\S]*for\(var i=0u;i<cCount\(l\);i\+=1u\)\{let s=cWorkSlot\(l,i\)/,
    "stencil and resolved-neighbor initialization must scale with live slots, not sparse capacity");
  const levelSetBuild = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn rebuildCandidateLevelSetFor"),
    octreeSPGridVCycleShader.indexOf("fn rebuildCandidateTransferFor"),
  );
  assert.doesNotMatch(levelSetBuild, /for\(var c=0u;c<STATE_CHANNELS/,
    "topology identity reset must not clear solver payload channels capacity-wide");
  assert.match(octreeSPGridVCycleShader,
    /rebuildCandidateTransfer[\s\S]*topologyDirty\(l\)\|\|topologyDirty\(l\+1u\)/,
    "the retained fine endpoint transfer must rebuild with the changed coarse suffix");
  assert.match(octreeSPGridVCycleShader,
    /@workgroup_size\(64\) fn planL1CaptureDelta[\s\S]*deltaOldRow\(r\)[\s\S]*sameL1Topology\(source,captured\)[\s\S]*oldNeighbor!=b\.row[\s\S]*bitcast<u32>\(a\.coefficient\)!=bitcast<u32>\(b\.coefficient\)/,
    "compact producer rows classify topology identity and coefficient-only stencil changes independently");
  assert.match(octreeSPGridVCycleShader,
    /commitChangedL1[\s\S]*capturePageStamp\(page\)!=generation[\s\S]*headers\[r\]=h[\s\S]*entries\[h\.entryStart\+j\]=sourceEntries\[h\.entryStart\+j\]/,
    "only exact changed pages are eligible for L1 commit");
  assert.match(octreeSPGridVCycleShader,
    /planL1CaptureDelta[\s\S]*for\(var work=lane;work<workCount;work\+=64u\)[\s\S]*captureWorkUnique\(work,page\)[\s\S]*captureLaneChanged\[lane\][\s\S]*validatedPages=totalChanged[\s\S]*readyGeneration=generation/,
    "capture planning must reduce only compact producer-stamped pages before candidate setup begins");
  assert.match(octreeSPGridVCycleShader,
    /finalizeL1CapturePublication[\s\S]*for\(var work=lane;work<workCount;work\+=64u\)[\s\S]*copyStamp!=generation[\s\S]*totalCopied!=capturePages\.changedPages[\s\S]*publishedGeneration=generation/,
    "the capture publication must fail closed unless copied and changed-page counts agree");
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn validateL1CapturePages|fn finalizeL1CapturePlan/,
    "the recurring all-page compare/reduce kernels must be deleted");
  assert.match(octreeSPGridVCycleShader,
    /fn captureWorkCount\(\)->u32\{return select\(deltaControl\(5u\),captureExpectedPages\(\),captureBootstrap\(\)\);\}/,
    "only explicit cold bootstrap may enumerate every page");
  const captureTransaction = octreeSPGridVCycleShader.slice(
    octreeSPGridVCycleShader.indexOf("fn beginL1CapturePlan"),
    octreeSPGridVCycleShader.indexOf("fn clearCorrection"));
  assert.doesNotMatch(captureTransaction, /atomic(?:Add|CompareExchange|Or|Store|Load)/,
    "exclusive page records and deterministic singleton scans replace recurring global counters");
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeCapture.toString(), /copyBufferToBuffer/,
    "cold and warm capture must use the identical exact page transaction, never a capacity-copy fallback");
  const setup = WebGPUOctreeSPGridVCycle.prototype.encodeSetup.toString();
  assert.match(setup, /buildCandidateLevelDeltas[\s\S]*validateCandidateHierarchy[\s\S]*commitChangedL1[\s\S]*finalizeL1CapturePublication[\s\S]*commitCandidateLevels/);
  assert.doesNotMatch(setup, /dispatchWorkgroupsIndirect|setupIndirect|prepareSetup/,
    "candidate publication must not retain redundant per-level indirect commands");
  assert.match(setup, /this\.run\(pass,\s*"finalizeLifecycle"/);
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
  assert.match(octreeSPGridVCycleShader, /fn previousValid\(\)->bool/);
  assert.match(octreeSPGridVCycleShader,
    /fn validateCandidateHierarchy[\s\S]*captureReport\(OVERFLOW\)[\s\S]*fn commitCandidateLevelAt[\s\S]*if\(captureFailed\(\)[\s\S]*return/);
  assert.doesNotMatch(octreeSPGridVCycleShader, /fn resetInvalidBuffers|arrayLength\(&topology\).*arrayLength\(&state\)/s);
  assert.match(octreeSPGridVCycleShader,
    /fn finalizeLifecycle[\s\S]*atomicLoad\(&control\[0\]\)==0u&&!captureFailed\(\)[\s\S]*dispatchMeta\[base\]=1u[\s\S]*Candidate failure leaves the previously published hierarchy byte-for-byte/);
  assert.doesNotMatch(WebGPUOctreeSPGridVCycle.prototype.encodeSetup.toString(), /clearBuffer/,
    "successful recurring generations must not write full sparse capacity");
});

test("correction consumes only per-level live slot dispatches", () => {
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
  assert.equal(direct, 3, "only correction clear, RHS seed, and publication use row-capacity dispatches");
  assert.equal(offsets.length, cycle.encodedCorrectionDispatchCount - direct);
  assert.equal(sources.size, 1, "all correction work must consume the dedicated indirect buffer");
  assert.ok(offsets.includes(8), "level zero consumes its live slot record");
  assert.ok(offsets.includes((cycle.plan.levelCount - 1) * 32 + 8), "the bottom level uses its own live slot record");
  assert.ok(offsets.every((offset) => offset % 32 === 8 || offset % 32 === 20),
    "row kernels consume live-slot records and parent-owned restriction consumes live-parent records");
  assert.equal(offsets.filter((offset) => offset % 32 === 20).length, cycle.plan.levelCount - 1);
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
    "buildCandidateLevelDeltas", "validateCandidateHierarchy", "commitCandidateLevels",
    "finalizeLifecycle", "clearCorrection", "zeroVectors", "seedRhs",
    "smoothAtoB", "smoothBtoA", "restrictAndGhostAccumulate", "exactBottom",
    "prolongAndGhostPropagate", "publish"]) {
    device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint } });
  }
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message); device.destroy();
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
  const headers = device.createBuffer({ size: 4 * 48, usage: storage });
  const entries = device.createBuffer({ size: 8, usage: storage });
  const rowCount = device.createBuffer({ size: 64, usage: storage });
  const control = device.createBuffer({ size: 64, usage: storage });
  const rowDelta = spgridDelta((size) => device.createBuffer({ size, usage: storage }), 4);
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: headers, leafEntries: entries, rowDelta },
    { dimensions: [4, 4, 4], rowCapacity: 4, finestCellWidth: 1 });
  device.pushErrorScope("validation");
  const encoder = device.createCommandEncoder();
  const broker = new PassBroker(encoder);
  cycle.encodeSetup(broker, { solverControl: control, rowCount });
  broker.finish();
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  cycle.destroy();
  for (const buffer of [headers, entries, rowCount, control, rowDelta.rows]) buffer.destroy();
  device.destroy();
});

test("Dawn exact-reuse setup preserves prior hierarchy and changed capture recopies live L1", {
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
  const headers = device.createBuffer({ size: 4 * 48, usage: storage });
  const entries = device.createBuffer({ size: 8, usage: storage });
  const sourceWords = new Uint32Array(4 * 12), sourceFloats = new Float32Array(sourceWords.buffer);
  for (let row = 0; row < 4; row += 1) {
    const base = row * 12; sourceWords[base] = row; sourceWords[base + 3] = 1;
    sourceFloats[base + 4] = 2; sourceFloats[base + 5] = -1;
  }
  device.queue.writeBuffer(headers, 0, sourceWords);
  const rowDelta = spgridDelta((size) => device.createBuffer({ size, usage: storage }), 4);
  const deltaWords = new Uint32Array(rowDelta.rows.size / 4);
  deltaWords.set([4, 0, 0, 4, 0, 4, 4, 1, 0x52444c54, 1, 1, 1, 1, 1, 1, 1]);
  deltaWords.set([0, 1, 2, 3], rowDelta.dirtyRowsOffsetWords);
  device.queue.writeBuffer(rowDelta.rows, 0, deltaWords);
  const cycle = new WebGPUOctreeSPGridVCycle(device, { leafHeaders: headers, leafEntries: entries, rowDelta },
    { dimensions: [4, 4, 4], rowCapacity: 4, finestCellWidth: 1 });
  type Internals = { dispatchMeta: GPUBuffer; indirectDispatch: GPUBuffer;
    state: GPUBuffer; capturedHeaders: GPUBuffer };
  const internal = cycle as unknown as Internals;
  const publication = new Uint32Array(cycle.plan.dispatchBytes / 4);
  for (let level = 0; level < cycle.plan.levelCount; level += 1) {
    const base = level * 8; publication[base] = 1;
    publication.set([1, 1, 1], base + 2); publication.set([1, 1, 1], base + 5);
  }
  const lifecycle = cycle.plan.levelCount * 8; publication[lifecycle] = 1; publication[lifecycle + 1] = 4;
  device.queue.writeBuffer(internal.dispatchMeta, 0, publication);
  device.queue.writeBuffer(internal.indirectDispatch, 0, publication);
  device.queue.writeBuffer(internal.state, 0, new Uint32Array([0x1234abcd]));
  device.queue.writeBuffer(internal.capturedHeaders, 0, new Uint32Array([99]));
  const countWords = new Uint32Array(16); countWords[0] = 4; countWords[7] = 1;
  const rowCount = device.createBuffer({ size: 64, usage: storage }); device.queue.writeBuffer(rowCount, 0, countWords);
  const control = device.createBuffer({ size: 64, usage: storage });
  const stateRead = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  const publicationRead = device.createBuffer({ size: cycle.plan.dispatchBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  device.pushErrorScope("validation"); let encoder = device.createCommandEncoder();
  cycle.encodeSetup(new PassBroker(encoder), { solverControl: control, rowCount });
  encoder.copyBufferToBuffer(internal.state, 0, stateRead, 0, 4);
  encoder.copyBufferToBuffer(internal.dispatchMeta, 0, publicationRead, 0, cycle.plan.dispatchBytes);
  device.queue.submit([encoder.finish()]);
  await Promise.all([stateRead.mapAsync(GPUMapMode.READ), publicationRead.mapAsync(GPUMapMode.READ)]);
  let validationError = await device.popErrorScope(); assert.equal(validationError, null, validationError?.message);
  assert.equal(new Uint32Array(stateRead.getMappedRange())[0], 0x1234abcd,
    "zero-work reuse must not retire or rewrite any prior hierarchy slot");
  assert.deepEqual([...new Uint32Array(publicationRead.getMappedRange())], [...publication],
    "reuse must preserve prior counts, correction dispatches, and lifecycle publication exactly");
  stateRead.unmap(); publicationRead.unmap();

  // Remove only the authoritative reuse proof. The same warm capture now
  // dispatches over live rows and replaces the retained header publication.
  device.pushErrorScope("validation");
  countWords[7] = 0; device.queue.writeBuffer(rowCount, 0, countWords);
  sourceWords[0] = 3; device.queue.writeBuffer(headers, 0, sourceWords);
  const captureRead = device.createBuffer({ size: 48, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  encoder = device.createCommandEncoder(); const captureBroker = new PassBroker(encoder);
  cycle.encodeCapture(captureBroker);
  captureBroker.fence("capture complete");
  encoder.copyBufferToBuffer(internal.capturedHeaders, 0, captureRead, 0, 48);
  device.queue.submit([encoder.finish()]);
  await captureRead.mapAsync(GPUMapMode.READ);
  assert.equal(new Uint32Array(captureRead.getMappedRange())[0], 3,
    "an unproven generation must recapture its live L1 header");
  validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
  captureRead.unmap(); cycle.destroy();
  for (const buffer of [headers, entries, rowCount, control, stateRead, publicationRead, captureRead,
    rowDelta.rows]) buffer.destroy();
  device.destroy();
});

test("Dawn accepts the one-workgroup persistent MGPCG shader", {
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
  const shaderModule = device.createShaderModule({ code: octreeSPGridPersistentMGPCGShader });
  const info = await shaderModule.getCompilationInfo();
  const errors = info.messages.filter((message) => message.type === "error");
  assert.deepEqual(errors.map((message) => `${message.lineNum}:${message.linePos} ${message.message}`), []);
  device.createComputePipeline({ layout: "auto", compute: { module: shaderModule, entryPoint: "persistentMGPCG" } });
  const validationError = await device.popErrorScope();
  assert.equal(validationError, null, validationError?.message);
});
