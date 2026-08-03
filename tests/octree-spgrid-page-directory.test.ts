import assert from "node:assert/strict";
import test from "node:test";
import {
  WebGPUOctreeSPGridVCycle,
  octreeSPGridVCycleShader,
  planOctreeSPGridVCycle,
} from "../lib/webgpu-octree-spgrid-vcycle";
import { octreePersistentMGPCGWGSL } from "../lib/webgpu-octree-persistent-mgpcg.wgsl";
import { planResolvedPowerRows } from "../lib/webgpu-octree-power-resolved-rows";

const INVALID = 0xffff_ffff;

/**
 * The dense logical-page directory answers one question — "which physical page
 * holds this corner of the level, if any" — and the answer for "none" has to be
 * INVALID. Zero is the perfectly legal index of physical page 0, so a directory
 * slot left at the buffer's zero-fill sends `finerAdjoint` and
 * `linkCandidatePageNeighbours` into a stranger's page and the persistent MGPCG
 * rejects the whole t=0 publication with ERR_ROW stage 22.
 *
 * The commit path cannot supply that resting value: it copies a directory slot
 * only for pages that are arriving or retiring, so a logical page that has
 * never held a cell is never written at all. Hence the allocation-time
 * authoring these tests pin.
 */

/** The stub used by the allocation tests in webgpu-octree-spgrid-vcycle. */
function recordingDevice() {
  Object.assign(globalThis, {
    GPUBufferUsage: { STORAGE: 1, COPY_DST: 2, COPY_SRC: 4, UNIFORM: 8, INDIRECT: 16 },
  });
  const writes: { target: GPUBuffer; offset: number; words: Uint32Array }[] = [];
  const buffer = (size: number, usage = 31) =>
    ({ size, usage, destroy() {} }) as unknown as GPUBuffer;
  const device = {
    queue: {
      writeBuffer(target: GPUBuffer, offset: number, data: ArrayBufferView) {
        writes.push({ target, offset,
          words: new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4) });
      },
    },
    createBuffer: ({ size, usage }: { size: number; usage: number }) => buffer(size, usage),
    createShaderModule: () => ({}),
    createComputePipeline: ({ label }: { label: string }) => ({ label, getBindGroupLayout: () => ({}) }),
    createBindGroup: () => ({}),
  } as unknown as GPUDevice;
  return { device, buffer, writes };
}

/** The same fixed Section 6.3 source shape the V-cycle allocation tests use. */
function spgridSource(buffer: (size: number, usage?: number) => GPUBuffer, rowCapacity: number) {
  const workset = buffer(8 * 512, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT);
  const binding = { buffer: workset, offset: 0, size: workset.size };
  return {
    rowCapacity,
    plan: planResolvedPowerRows(rowCapacity, 8),
    params: buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST),
    rows: buffer(2 * planResolvedPowerRows(rowCapacity, 8).arenaBytes,
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
    diagonals: buffer(rowCapacity * 8),
    coefficients: buffer(2 * rowCapacity * 19 * 4),
    coefficientBankStrideWords: rowCapacity * 19,
    rowGeometry: buffer(rowCapacity * 32),
    topologyMetrics: buffer(rowCapacity * 16),
    catalogCoefficients: buffer(1_608 * 19 * 4),
    worksets: { regularInterior: binding, transitionInterior: binding,
      physicalBoundary: binding, transitionBoundary: binding },
    invalidRows: binding,
    control: buffer(64),
    rowDelta: {
      rows: buffer(4 * (16 + 3 * rowCapacity)),
      rowCapacity,
      controlOffsetWords: 0,
      newToOldOffsetWords: 16,
      oldToNewOffsetWords: 16 + rowCapacity,
      dirtyRowsOffsetWords: 16 + 2 * rowCapacity,
    },
    classDispatch: buffer(108, GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT),
    classDispatchOffsetBytes: 24,
  };
}

test("the plan publishes the page directory at the offset the shader reads it from", () => {
  for (const [dimensions, rowCapacity] of [
    [[72, 24, 48], 55_040],   // the porcelain pond, where this first bit
    [[16, 16, 16], 3_072],    // the mini power dam
    [[320, 96, 80], 131_072], // the ocean lattice, whose row map dwarfs the rest
  ] as const) {
    const plan = planOctreeSPGridVCycle({ dimensions: [...dimensions], rowCapacity });
    // Verbatim `pageDirectoryBase()`: 16 header words, the per-level row map,
    // the slot worklist, then twenty-eight words per page record.
    const shaderBase = 16 + plan.levelCount * plan.rowCapacity + 29 * plan.totalLevelSlots;
    assert.equal(plan.pageDirectoryOffsetBytes, shaderBase * 4,
      `page directory offset must equal pageDirectoryBase() for ${dimensions.join("x")}`);
    assert.ok(plan.pageDirectoryOffsetBytes + plan.pageDirectoryBytes <= plan.topologyBytes,
      "the directory span must lie inside the topology allocation");
    const logicalPages = Array.from({ length: plan.levelCount }, (_, level) => {
      const d = dimensions.map((value) => Math.ceil(value / 2 ** level));
      return Math.ceil(d[0] / 8) * Math.ceil(d[1] / 8) * Math.ceil(d[2] / 4);
    }).reduce((sum, count) => sum + count, 0);
    assert.equal(plan.pageDirectoryBytes, logicalPages * 4);
  }
});

test("both shader copies still derive that base from the same four terms", () => {
  // If a refactor moves the region, the arithmetic above stops describing it
  // and the INVALID fill lands somewhere harmless-looking instead.
  const source = `${octreeSPGridVCycleShader}\n${octreePersistentMGPCGWGSL({
    maximumIterations: 12,
  })}`;
  for (const chain of [
    /fn rowMapBase\(\)->u32\{return 16u;\}/,
    /fn workBase\(\)->u32\{return rowMapBase\(\)\+levels\(\)\*(?:capacity\(\)|p\.capacity\.x);\}/,
    /fn pageWorkBase\(\)->u32\{return workBase\(\)\+totalLevelSlots\(\);\}/,
    /fn pageDirectoryBase\(\)->u32\{return pageWorkBase\(\)\+(?:PAGE_RECORD_WORDS|28u)\*totalLevelSlots\(\);\}/,
  ]) assert.match(source, chain);
});

test("allocation authors every logical page as absent, in both topology banks", () => {
  const { device, buffer, writes } = recordingDevice();
  const dimensions: readonly [number, number, number] = [72, 24, 48], rowCapacity = 512;
  const cycle = new WebGPUOctreeSPGridVCycle(device, spgridSource(buffer, rowCapacity),
    { dimensions, rowCapacity, finestCellWidth: 0.025, compileHierarchicalExecutor: false });
  const plan = cycle.plan;
  const directoryWrites = writes.filter((write) =>
    write.offset === plan.pageDirectoryOffsetBytes);
  assert.equal(directoryWrites.length, 2,
    "the accepted and candidate directories must both be authored");
  assert.equal(new Set(directoryWrites.map((write) => write.target)).size, 2,
    "the two writes must land in different buffers");
  for (const write of directoryWrites) {
    assert.equal(write.words.length, plan.pageDirectoryBytes / 4);
    assert.ok(write.words.every((word) => word === INVALID),
      "a logical page that has never held a cell must read INVALID, never page 0");
    assert.equal(write.target.size, plan.topologyBytes);
  }
  cycle.destroy();
});
