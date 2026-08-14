import {
  OCTREE_POWER_ROW_DELTA_VALID,
  type OctreePowerRowDeltaSource,
} from "../lib/methods/power/webgpu-octree-power-descriptor";

export interface ColdPowerRowPublication {
  readonly rowCount: GPUBuffer;
  readonly rowDelta: OctreePowerRowDeltaSource;
  destroy(): void;
}

/** Exact t=0 frontier publication: every current row is new, dirty, and affected. */
export function createColdPowerRowPublication(
  device: GPUDevice,
  rowCapacity: number,
  rowCount: number,
  generation: number,
): ColdPowerRowPublication {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(rowCount) || rowCount < 0 || rowCount > rowCapacity
    || !Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
    throw new RangeError("Cold power row publication parameters are invalid");
  }
  const controlOffsetWords = 0;
  const newToOldOffsetWords = 16;
  const oldToNewOffsetWords = newToOldOffsetWords + rowCapacity;
  const dirtyRowsOffsetWords = oldToNewOffsetWords + rowCapacity;
  const affectedRowsOffsetWords = dirtyRowsOffsetWords + rowCapacity;
  const words = new Uint32Array(affectedRowsOffsetWords + rowCapacity);
  const groups = Math.ceil(rowCount / 64);
  const dispatchX = Math.min(groups, 65_535);
  const dispatchY = dispatchX > 0 ? Math.ceil(groups / dispatchX) : 1;
  words.set([
    rowCount, 0, 0, rowCount, 0, rowCount, rowCount, generation,
    OCTREE_POWER_ROW_DELTA_VALID,
    dispatchX, dispatchY, 1,
    dispatchX, dispatchY, 1,
    0,
  ]);
  for (let row = 0; row < rowCount; row += 1) {
    words[newToOldOffsetWords + row] = 0x8000_0000;
    words[dirtyRowsOffsetWords + row] = row;
    words[affectedRowsOffsetWords + row] = row;
  }
  const rowCountBuffer = device.createBuffer({
    label: "Cold power row count",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const rows = device.createBuffer({
    label: "Cold power row delta",
    size: words.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(rowCountBuffer, 0, new Uint32Array([rowCount]));
  device.queue.writeBuffer(rows, 0, words);
  return {
    rowCount: rowCountBuffer,
    rowDelta: {
      rows,
      rowCapacity,
      controlOffsetWords,
      newToOldOffsetWords,
      oldToNewOffsetWords,
      dirtyRowsOffsetWords,
      affectedRowsOffsetWords,
    },
    destroy() {
      rowCountBuffer.destroy();
      rows.destroy();
    },
  };
}

/** Exact unchanged frontier publication: every row carries its prior identity. */
export function createIdentityPowerRowPublication(
  device: GPUDevice,
  rowCapacity: number,
  rowCount: number,
  generation: number,
): ColdPowerRowPublication {
  if (!Number.isSafeInteger(rowCapacity) || rowCapacity < 1
    || !Number.isSafeInteger(rowCount) || rowCount < 0 || rowCount > rowCapacity
    || !Number.isSafeInteger(generation) || generation < 0 || generation > 0xffff_ffff) {
    throw new RangeError("Identity power row publication parameters are invalid");
  }
  const controlOffsetWords = 0;
  const newToOldOffsetWords = 16;
  const oldToNewOffsetWords = newToOldOffsetWords + rowCapacity;
  const dirtyRowsOffsetWords = oldToNewOffsetWords + rowCapacity;
  const affectedRowsOffsetWords = dirtyRowsOffsetWords + rowCapacity;
  const words = new Uint32Array(affectedRowsOffsetWords + rowCapacity);
  words.set([
    rowCount, rowCount, rowCount, 0, 0, 0, 0, generation,
    OCTREE_POWER_ROW_DELTA_VALID,
    0, 1, 1,
    0, 1, 1,
    0,
  ]);
  for (let row = 0; row < rowCount; row += 1) {
    words[newToOldOffsetWords + row] = row + 1;
    words[oldToNewOffsetWords + row] = row + 1;
  }
  const rowCountBuffer = device.createBuffer({
    label: "Identity power row count",
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  const rows = device.createBuffer({
    label: "Identity power row delta",
    size: words.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(rowCountBuffer, 0, new Uint32Array([rowCount]));
  device.queue.writeBuffer(rows, 0, words);
  return {
    rowCount: rowCountBuffer,
    rowDelta: {
      rows,
      rowCapacity,
      controlOffsetWords,
      newToOldOffsetWords,
      oldToNewOffsetWords,
      dirtyRowsOffsetWords,
      affectedRowsOffsetWords,
    },
    destroy() {
      rowCountBuffer.destroy();
      rows.destroy();
    },
  };
}
