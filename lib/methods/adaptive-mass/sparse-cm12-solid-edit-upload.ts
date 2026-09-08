import type { SolidWorld } from "../../core/solid-world";
import { writeSparseCM12SolidOccupancy, type SparseCM12SolidOccupancyLayout } from "./sparse-cm12-solid-occupancy";

/** Private unique-address scatter, bounded independently of total GPU arenas.
 * Last writer wins while preparing; GPU invocations never race overlapping
 * header/directory clears or payload writes. High address bit selects state. */
export function prepareSolidEditUpload(layout: SparseCM12SolidOccupancyLayout,
  previous: SolidWorld, next: SolidWorld, maximumWords = 262144) {
  const writes = new Map<number, number>();
  const set = (address: number, value: number) => {
    if (!writes.has(address) && writes.size >= maximumWords) {
      throw new RangeError("Live solid upload exceeds its bounded work budget; use a smaller edit.");
    }
    writes.set(address, value >>> 0);
  };
  const add = (arena: "topology" | "state", firstWord: number, data: Uint32Array) => {
    if (!Number.isSafeInteger(firstWord) || firstWord < 0 || firstWord + data.length >= 0x80000000) {
      throw new RangeError("Invalid live solid scatter address");
    }
    const flag = arena === "state" ? 0x80000000 : 0;
    for (let index = 0; index < data.length; index++) set((firstWord + index + flag) >>> 0, data[index]!);
  };
  const zeroFirst = layout.baseWords + layout.directoryBaseWords;
  for (let index = 0; index < layout.pageBaseWords - layout.directoryBaseWords; index++) set(zeroFirst + index, 0);
  const queue = { writeBuffer(_destination: GPUBuffer, offset: number, data: GPUAllowSharedBufferSource,
    dataOffset = 0, size?: number): undefined {
    const view = ArrayBuffer.isView(data);
    const bytesPerElement = view && "BYTES_PER_ELEMENT" in data ? Number(data.BYTES_PER_ELEMENT) : 1;
    const buffer = view ? data.buffer : data;
    const first = (view ? data.byteOffset : 0) + dataOffset * bytesPerElement;
    const bytes = size === undefined ? data.byteLength - dataOffset * bytesPerElement : size * bytesPerElement;
    add("topology", offset / 4, new Uint32Array(buffer, first, bytes / 4));
  } };
  writeSparseCM12SolidOccupancy(queue, undefined as unknown as GPUBuffer, layout, next, [0, 0, 0], previous);
  // Region slots are an ordered fixed-capacity program. Clear its unused tail
  // so Undo cannot leave a stale later fill or clear command effective.
  const regions = new Uint32Array(layout.regionCapacity * 8);
  for (const [index, region] of (next.regions ?? []).entries()) regions.set([
    region.operation === "fill" ? 1 : 0, ...region.minimum.map(v => v >>> 0),
    ...region.maximumExclusive.map(v => v >>> 0), region.materialId ?? 1,
  ], index * 8);
  add("topology", layout.baseWords + layout.regionBaseWords, regions);
  return { add, finish() {
    const words = new Uint32Array(writes.size * 2); let at = 0;
    for (const [address, value] of writes) { words[at++] = address; words[at++] = value; }
    return words;
  } };
}
