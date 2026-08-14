import { deflateSync } from "node:zlib";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";

import { resolveDisplayGrade, type ResolvedDisplayGrade } from "../lib/core/webgpu-lighting";

/**
 * Write a captured `rgba16float` frame out as a PNG.
 *
 * The headless lanes already read a settled frame back to hash it; this turns
 * that same buffer into something a person can look at. It exists because the
 * browser is not a stable observation post while the scene's generators are
 * being edited — a hot reload can serve a half-written module and the frame goes
 * away — and because an art-direction loop wants the *same* camera every pass,
 * which a hand-driven browser does not give.
 *
 * There is no PNG dependency in this repo and this needs none: PNG's required
 * compressor is zlib, which node ships, and the rest of the container is four
 * chunks and a CRC.
 */

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let c = 0xffffffff;
  for (const byte of bytes) c = CRC_TABLE[(c ^ byte) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, body: Uint8Array): Uint8Array {
  const out = new Uint8Array(body.length + 12);
  const view = new DataView(out.buffer);
  view.setUint32(0, body.length);
  for (let index = 0; index < 4; index += 1) out[4 + index] = type.charCodeAt(index);
  out.set(body, 8);
  view.setUint32(body.length + 8, crc32(out.subarray(4, body.length + 8)));
  return out;
}

function decodeF16(bits: number): number {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;
  if (exponent === 0) return sign * mantissa * 2 ** -24;
  if (exponent === 31) return mantissa ? Number.NaN : sign * Infinity;
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/**
 * The CPU mirror of `unifiedDisplayGrade` in lib/webgpu-lighting.ts.
 *
 * Kept in step by hand rather than shared, because the WGSL side is a string in
 * a shader library and this is the only consumer outside it. If the two drift,
 * an offline frame stops matching what the app shows — which is exactly the
 * thing this file is used to decide — so the curve is transcribed literally.
 */
function displayGrade(value: number, grade: ResolvedDisplayGrade, channel: number): number {
  const balanced = Math.max(0, Number.isFinite(value) ? value : 0) * Math.max(0, grade.whiteBalance[channel]);
  const exposed = balanced * Math.max(0, grade.exposure);
  const mapped = grade.toneCurve === "aces"
    ? Math.min(1, Math.max(0, (exposed * (2.51 * exposed + 0.03)) / (exposed * (2.43 * exposed + 0.59) + 0.14)))
    : exposed / (exposed + 1);
  return Math.round(255 * mapped ** (1 / 2.2));
}

export interface FramePngOptions {
  readonly width: number;
  readonly height: number;
  /** The frame as read back: tightly packed `rgba16float`, row-major, top row first. */
  readonly packedRows: Uint32Array;
  /** Absent takes the neutral grade, which is Reinhard at exposure 1. */
  readonly grade?: ResolvedDisplayGrade;
  /**
   * Region to write, in frame pixels from the top-left. Absent writes the frame.
   *
   * Art direction is done on one object at a time, and a tree that is 90 px
   * across in a 800 px frame cannot be judged from the frame. Rendering large
   * and cutting the object out beats moving the camera, because moving the
   * camera changes the composition being judged — and on this scene it would
   * also re-aim the key, which is derived from the camera's own basis.
   */
  readonly crop?: { readonly x: number; readonly y: number; readonly width: number; readonly height: number };
}

/** Encode and write, creating the containing directory. Returns the path written. */
export function writeFramePng(outPath: string, options: FramePngOptions): string {
  const { width, height, packedRows } = options;
  const grade = options.grade ?? resolveDisplayGrade(undefined);
  const halfWords = new Uint16Array(packedRows.buffer, packedRows.byteOffset, packedRows.length * 2);
  const originX = Math.max(0, Math.min(width - 1, Math.round(options.crop?.x ?? 0)));
  const originY = Math.max(0, Math.min(height - 1, Math.round(options.crop?.y ?? 0)));
  const outWidth = Math.max(1, Math.min(width - originX, Math.round(options.crop?.width ?? width)));
  const outHeight = Math.max(1, Math.min(height - originY, Math.round(options.crop?.height ?? height)));
  // One filter byte per scanline, filter 0: the frames here are smooth and the
  // saving from a predictor is not worth a second pass over the image.
  const raw = new Uint8Array(outHeight * (1 + outWidth * 3));
  for (let y = 0; y < outHeight; y += 1) {
    const rowStart = y * (1 + outWidth * 3);
    raw[rowStart] = 0;
    for (let x = 0; x < outWidth; x += 1) {
      const source = ((y + originY) * width + (x + originX)) * 4;
      const destination = rowStart + 1 + x * 3;
      raw[destination] = displayGrade(decodeF16(halfWords[source]), grade, 0);
      raw[destination + 1] = displayGrade(decodeF16(halfWords[source + 1]), grade, 1);
      raw[destination + 2] = displayGrade(decodeF16(halfWords[source + 2]), grade, 2);
    }
  }
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, outWidth);
  headerView.setUint32(4, outHeight);
  header[8] = 8; // bit depth
  header[9] = 2; // colour type: truecolour
  const png = Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    Buffer.from(chunk("IHDR", header)),
    Buffer.from(chunk("IDAT", deflateSync(raw, { level: 6 }))),
    Buffer.from(chunk("IEND", new Uint8Array(0))),
  ]);
  mkdirSync(path.dirname(outPath), { recursive: true });
  writeFileSync(outPath, png);
  return outPath;
}

/** What the frame carries, in scene-linear radiance, for deciding whether a grade is the problem. */
export function frameRadianceRange(packedRows: Uint32Array): { min: number; max: number; mean: number } {
  const halfWords = new Uint16Array(packedRows.buffer, packedRows.byteOffset, packedRows.length * 2);
  let min = Infinity;
  let max = -Infinity;
  let total = 0;
  let samples = 0;
  for (let index = 0; index + 2 < halfWords.length; index += 4) {
    for (let channel = 0; channel < 3; channel += 1) {
      const value = decodeF16(halfWords[index + channel]);
      if (!Number.isFinite(value)) continue;
      min = Math.min(min, value);
      max = Math.max(max, value);
      total += value;
      samples += 1;
    }
  }
  return { min: samples ? min : 0, max: samples ? max : 0, mean: samples ? total / samples : 0 };
}
