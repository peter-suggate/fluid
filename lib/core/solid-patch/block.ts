import type { SolidPatchOperation, SolidWorldCoordinate, SolidWorldVoxelPatch } from "../solid-world";
import { SOLID_PATCH_BLOCK_KIND, type VoxelBlock } from "./kind";

/**
 * The payload of a `"block"` patch: a voxel block as a string.
 *
 * A string, because the scene document is cloned and saved as JSON and a typed
 * array does not survive that. Run-length encoded, because an edited region is
 * mostly runs: a 64-wide sphere is a few kilobytes this way and hundreds as
 * boxes.
 *
 * The string is `vb1:` and the base64 of this layout (integers are LEB128):
 *
 *   u8      layout version, 1
 *   varint  size x, y, z
 *   varint  palette length P, then P entries of
 *           u8 fraction, u16 distance + 0x8000, u16 material (little endian),
 *           most used first, ties by ascending value
 *   tokens  until every cell is described, x fastest:
 *           varint 2 (n - 1)      then varint palette index: n cells of it
 *           varint 2 (n - 1) + 1  then n indices of b bits each, packed from
 *                                 the low bit, b = max(1, ceil(log2 P))
 *
 * A palette of whole cells rather than raw channels, because a block is mostly
 * two of them (empty, and solid in one material) and the rest are the surface:
 * a checkerboard then costs a bit a cell, not five bytes.
 */
const PAYLOAD_PREFIX = "vb1:";
const LAYOUT_VERSION = 1;
const PALETTE_ENTRY_BYTES = 5;

/**
 * The largest block a payload may describe: 256 cubed. A decoded block is five
 * bytes a cell whatever its payload's length, so the bound is on the cells.
 */
export const VOXEL_BLOCK_MAXIMUM_CELLS = 1 << 24;

function voxelBlockCellCount(size: SolidWorldCoordinate): number {
  if (size.length !== 3 || !size.every((value) => Number.isSafeInteger(value) && value > 0)) {
    throw new RangeError(`Voxel block size must be three positive integers, received ${size.join(" x ")}`);
  }
  const cells = size[0] * size[1] * size[2];
  if (cells > VOXEL_BLOCK_MAXIMUM_CELLS) {
    throw new RangeError(`Voxel block of ${size.join(" x ")} cells exceeds the ${VOXEL_BLOCK_MAXIMUM_CELLS}-cell limit`);
  }
  return cells;
}

const paletteIndexBits = (paletteLength: number): number => Math.max(1, 32 - Math.clz32(paletteLength - 1));

export function encodeVoxelBlock(block: VoxelBlock): string {
  const cells = voxelBlockCellCount(block.size);
  const { solidFraction, signedDistanceQ8, materialId } = block;
  if (solidFraction.length !== cells || signedDistanceQ8.length !== cells || materialId.length !== cells) {
    throw new RangeError(`Voxel block channels must each hold the ${cells} cells of a ${block.size.join(" x ")} box`);
  }
  // A cell's three channels as one number, 40 bits.
  const cell = (index: number): number => solidFraction[index]!
    + 0x100 * (signedDistanceQ8[index]! + 0x8000) + 0x100_0000 * materialId[index]!;
  const uses = new Map<number, number>();
  for (let index = 0; index < cells;) {
    const value = cell(index);
    let end = index + 1;
    while (end < cells && cell(end) === value) end += 1;
    uses.set(value, (uses.get(value) ?? 0) + end - index);
    index = end;
  }
  const palette = [...uses.entries()].sort((left, right) => right[1] - left[1] || left[0] - right[0])
    .map(([value]) => value);
  const paletteIndex = new Map(palette.map((value, index) => [value, index]));
  const indices = new Uint32Array(cells);
  for (let index = 0, previous = -1, previousIndex = 0; index < cells; index += 1) {
    const value = cell(index);
    if (value !== previous) { previous = value; previousIndex = paletteIndex.get(value)!; }
    indices[index] = previousIndex;
  }

  const out = new ByteWriter();
  out.byte(LAYOUT_VERSION);
  for (const extent of block.size) out.varint(extent);
  out.varint(palette.length);
  for (const value of palette) {
    const distance = Math.floor(value / 0x100) % 0x1_0000, material = Math.floor(value / 0x100_0000);
    out.byte(value % 0x100);
    out.byte(distance & 0xff); out.byte(distance >>> 8);
    out.byte(material & 0xff); out.byte(material >>> 8);
  }
  const bits = paletteIndexBits(palette.length);
  // A run is worth its own token once packed indices would cost more than the
  // two bytes it takes, and the token it interrupts.
  const repeatMinimum = Math.max(2, Math.ceil(16 / bits) + 1);
  let literalStart = 0;
  const literal = (end: number) => {
    if (end === literalStart) return;
    out.varint(2 * (end - literalStart - 1) + 1);
    let pending = 0, pendingBits = 0;
    for (let index = literalStart; index < end; index += 1) {
      pending |= indices[index]! << pendingBits;
      pendingBits += bits;
      for (; pendingBits >= 8; pendingBits -= 8) { out.byte(pending & 0xff); pending >>>= 8; }
    }
    if (pendingBits > 0) out.byte(pending & 0xff);
  };
  for (let index = 0; index < cells;) {
    let end = index + 1;
    while (end < cells && indices[end] === indices[index]) end += 1;
    if (end - index >= repeatMinimum) {
      literal(index);
      out.varint(2 * (end - index - 1));
      out.varint(indices[index]!);
      literalStart = end;
    }
    index = end;
  }
  literal(cells);
  return PAYLOAD_PREFIX + base64FromBytes(out.bytes, out.length);
}

/** Decode a payload for a box of `size`; a payload that does not describe exactly that box is a RangeError. */
export function decodeVoxelBlock(payload: string, size: SolidWorldCoordinate): VoxelBlock {
  const cells = voxelBlockCellCount(size);
  if (typeof payload !== "string" || !payload.startsWith(PAYLOAD_PREFIX)) {
    throw new RangeError(`Voxel block payload must start with "${PAYLOAD_PREFIX}"`);
  }
  const reader = new ByteReader(bytesFromBase64(payload, PAYLOAD_PREFIX.length));
  const version = reader.byte();
  if (version !== LAYOUT_VERSION) throw new RangeError(`Unsupported voxel block layout version ${version}`);
  const encodedSize = [reader.varint(), reader.varint(), reader.varint()];
  if (encodedSize.some((value, axis) => value !== size[axis])) {
    throw new RangeError(`Voxel block payload describes a ${encodedSize.join(" x ")} box, not ${size.join(" x ")}`);
  }
  const paletteLength = reader.varint();
  if (paletteLength < 1 || paletteLength > cells) {
    throw new RangeError(`Voxel block palette of ${paletteLength} entries cannot describe ${cells} cells`);
  }
  const paletteFraction = new Uint8Array(paletteLength), paletteDistance = new Int16Array(paletteLength);
  const paletteMaterial = new Uint16Array(paletteLength);
  for (let index = 0; index < paletteLength; index += 1) {
    paletteFraction[index] = reader.byte();
    paletteDistance[index] = (reader.byte() | reader.byte() << 8) - 0x8000;
    paletteMaterial[index] = reader.byte() | reader.byte() << 8;
  }
  const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells);
  const materialId = new Uint16Array(cells);
  const bits = paletteIndexBits(paletteLength), mask = 2 ** bits - 1;
  const entry = (index: number): number => {
    if (index >= paletteLength) throw new RangeError(`Voxel block payload names palette entry ${index} of ${paletteLength}`);
    return index;
  };
  for (let filled = 0; filled < cells;) {
    const token = reader.varint(), count = Math.floor(token / 2) + 1;
    if (filled + count > cells) throw new RangeError(`Voxel block payload describes more than the ${cells} cells of its box`);
    if (token % 2 === 0) {
      const index = entry(reader.varint());
      solidFraction.fill(paletteFraction[index]!, filled, filled + count);
      signedDistanceQ8.fill(paletteDistance[index]!, filled, filled + count);
      materialId.fill(paletteMaterial[index]!, filled, filled + count);
    } else {
      let pending = 0, pendingBits = 0;
      for (let at = filled; at < filled + count; at += 1) {
        for (; pendingBits < bits; pendingBits += 8) pending |= reader.byte() << pendingBits;
        const index = entry(pending & mask);
        pending >>>= bits; pendingBits -= bits;
        solidFraction[at] = paletteFraction[index]!;
        signedDistanceQ8[at] = paletteDistance[index]!;
        materialId[at] = paletteMaterial[index]!;
      }
    }
    filled += count;
  }
  if (!reader.finished) throw new RangeError("Voxel block payload continues past the last cell of its box");
  return { size: [size[0], size[1], size[2]], solidFraction, signedDistanceQ8, materialId };
}

/**
 * The same cells over a box of a different size, nearest cell to nearest cell.
 *
 * Coverage is a share of a cell and carries over as it is; distance is counted
 * in cells, so it is rescaled by how many new cells an old one spans. A wholly
 * covered cell stays at least the half cell deep that its coverage claims.
 */
export function resampleVoxelBlock(block: VoxelBlock, size: SolidWorldCoordinate): VoxelBlock {
  const cells = voxelBlockCellCount(size), [sx, sy, sz] = block.size;
  const solidFraction = new Uint8Array(cells), signedDistanceQ8 = new Int16Array(cells);
  const materialId = new Uint16Array(cells);
  const scale = Math.cbrt(cells / (sx * sy * sz));
  for (let z = 0, at = 0; z < size[2]; z += 1) {
    const fromZ = Math.min(sz - 1, Math.floor((z + 0.5) * sz / size[2]));
    for (let y = 0; y < size[1]; y += 1) {
      const fromY = Math.min(sy - 1, Math.floor((y + 0.5) * sy / size[1]));
      for (let x = 0; x < size[0]; x += 1, at += 1) {
        const from = Math.min(sx - 1, Math.floor((x + 0.5) * sx / size[0])) + sx * (fromY + sy * fromZ);
        const fraction = block.solidFraction[from]!, distance = block.signedDistanceQ8[from]!;
        const scaled = Math.max(-0x8000, Math.min(0x7fff, Math.round(distance * scale)));
        solidFraction[at] = fraction;
        signedDistanceQ8[at] = fraction === 0 ? distance : fraction === 255 ? Math.min(-128, scaled) : scaled;
        materialId[at] = block.materialId[from]!;
      }
    }
  }
  return { size: [size[0], size[1], size[2]], solidFraction, signedDistanceQ8, materialId };
}

/**
 * The block patch that composes `block` at `minimum`.
 *
 * Cells whose coverage is zero are left alone by every operation, so a block
 * is a stamp of its solid cells rather than an overwrite of its box.
 */
export function voxelBlockPatch(operation: SolidPatchOperation, minimum: SolidWorldCoordinate,
  block: VoxelBlock, materialId?: number): SolidWorldVoxelPatch {
  return {
    operation,
    minimum: [minimum[0], minimum[1], minimum[2]],
    maximumExclusive: [minimum[0] + block.size[0], minimum[1] + block.size[1], minimum[2] + block.size[2]],
    ...(materialId === undefined ? {} : { materialId }),
    kind: SOLID_PATCH_BLOCK_KIND,
    payload: encodeVoxelBlock(block),
  };
}

class ByteWriter {
  bytes = new Uint8Array(1024);
  length = 0;

  byte(value: number): void {
    if (this.length === this.bytes.length) {
      const grown = new Uint8Array(2 * this.bytes.length);
      grown.set(this.bytes);
      this.bytes = grown;
    }
    this.bytes[this.length++] = value;
  }

  varint(value: number): void {
    for (; value >= 0x80; value = Math.floor(value / 0x80)) this.byte(value % 0x80 | 0x80);
    this.byte(value);
  }
}

class ByteReader {
  private position = 0;
  constructor(private readonly bytes: Uint8Array) {}

  get finished(): boolean { return this.position === this.bytes.length; }

  byte(): number {
    if (this.position >= this.bytes.length) throw new RangeError("Voxel block payload ends before the last cell of its box");
    return this.bytes[this.position++]!;
  }

  varint(): number {
    let value = 0;
    // Five groups hold any count or index a block within the cell limit can need.
    for (let scale = 1, group = 0; group < 5; group += 1, scale *= 0x80) {
      const byte = this.byte();
      value += (byte & 0x7f) * scale;
      if (byte < 0x80) return value;
    }
    throw new RangeError("Voxel block payload holds an integer longer than five bytes");
  }
}

const BASE64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_PAD = 61;
const BASE64_CODES = Uint8Array.from(BASE64_ALPHABET, (character) => character.charCodeAt(0));
const BASE64_VALUES = (() => {
  const values = new Int8Array(128).fill(-1);
  BASE64_CODES.forEach((code, value) => { values[code] = value; });
  return values;
})();

// Hand-rolled rather than `btoa`: the same bytes in a window, a worker and Node, with no binary string in between.
function base64FromBytes(bytes: Uint8Array, length: number): string {
  const codes = new Uint8Array(4 * Math.ceil(length / 3));
  for (let from = 0, to = 0; from < length; from += 3, to += 4) {
    const a = bytes[from]!, b = from + 1 < length ? bytes[from + 1]! : 0, c = from + 2 < length ? bytes[from + 2]! : 0;
    codes[to] = BASE64_CODES[a >>> 2]!;
    codes[to + 1] = BASE64_CODES[(a & 3) << 4 | b >>> 4]!;
    codes[to + 2] = from + 1 < length ? BASE64_CODES[(b & 15) << 2 | c >>> 6]! : BASE64_PAD;
    codes[to + 3] = from + 2 < length ? BASE64_CODES[c & 63]! : BASE64_PAD;
  }
  let text = "";
  for (let from = 0; from < codes.length; from += 0x8000) {
    text += String.fromCharCode.apply(null, codes.subarray(from, from + 0x8000) as unknown as number[]);
  }
  return text;
}

function bytesFromBase64(text: string, from: number): Uint8Array {
  const characters = text.length - from;
  const padding = text.endsWith("==") ? 2 : text.endsWith("=") ? 1 : 0;
  const invalid = () => new RangeError("Voxel block payload is not base64");
  if (characters === 0 || characters % 4 !== 0) throw invalid();
  const bytes = new Uint8Array(characters / 4 * 3 - padding);
  const value = (at: number): number => {
    const code = text.charCodeAt(at);
    const sextet = code < 128 ? BASE64_VALUES[code]! : -1;
    if (sextet < 0) {
      // Padding is only ever the final one or two characters.
      if (code === BASE64_PAD && at >= text.length - padding) return 0;
      throw invalid();
    }
    return sextet;
  };
  for (let at = from, to = 0; at < text.length; at += 4, to += 3) {
    const word = value(at) << 18 | value(at + 1) << 12 | value(at + 2) << 6 | value(at + 3);
    bytes[to] = word >>> 16;
    if (to + 1 < bytes.length) bytes[to + 1] = word >>> 8 & 0xff;
    if (to + 2 < bytes.length) bytes[to + 2] = word & 0xff;
  }
  return bytes;
}
