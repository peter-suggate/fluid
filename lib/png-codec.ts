/**
 * The repository's PNG container, in both directions.
 *
 * `tools/write-frame-png.ts` already proved the encoder side needs no
 * dependency: PNG's required compressor is zlib, which node ships, and the rest
 * is four chunks and a CRC. The fidelity gate (`lib/hero-fidelity-score.ts`)
 * needs the *other* direction — a reference plate arrives as a PNG and has to
 * become pixels before it can be scored — so the chunk framing and the CRC
 * table live here once and both directions share them.
 *
 * Scope is deliberately narrow: 8-bit non-interlaced truecolour, with or
 * without alpha, which is what this tree produces and what the plate is. A
 * paletted or 16-bit or interlaced PNG is **refused**, loudly, rather than
 * decoded approximately — a plate that silently decoded as the wrong pixels
 * would poison every score derived from it, and a score nobody can trust is
 * worse than no score.
 */
import { deflateSync, inflateSync } from "node:zlib";

const PNG_SIGNATURE = Object.freeze([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

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

/** One PNG chunk: length, type, body, CRC over type and body. */
export function pngChunk(type: string, body: Uint8Array): Uint8Array {
  const out = new Uint8Array(body.length + 12);
  const view = new DataView(out.buffer);
  view.setUint32(0, body.length);
  for (let index = 0; index < 4; index += 1) out[4 + index] = type.charCodeAt(index);
  out.set(body, 8);
  view.setUint32(body.length + 8, crc32(out.subarray(4, body.length + 8)));
  return out;
}

/** An 8-bit image in tightly packed row-major RGB, top row first. */
export interface RgbImage {
  readonly width: number;
  readonly height: number;
  /** `width * height * 3` bytes, sRGB-encoded. */
  readonly rgb: Uint8Array;
}

/** Encode tightly packed 8-bit RGB as a truecolour PNG. */
export function encodeRgbPng(image: RgbImage): Uint8Array {
  const { width, height, rgb } = image;
  if (rgb.length !== width * height * 3) {
    throw new RangeError(`rgb must be ${width * height * 3} bytes for ${width}x${height}, received ${rgb.length}`);
  }
  // One filter byte per scanline, filter 0. These images are smooth and the
  // saving from a predictor is not worth a second pass.
  const raw = new Uint8Array(height * (1 + width * 3));
  for (let y = 0; y < height; y += 1) {
    const rowStart = y * (1 + width * 3);
    raw[rowStart] = 0;
    raw.set(rgb.subarray(y * width * 3, (y + 1) * width * 3), rowStart + 1);
  }
  const header = new Uint8Array(13);
  const headerView = new DataView(header.buffer);
  headerView.setUint32(0, width);
  headerView.setUint32(4, height);
  header[8] = 8; // bit depth
  header[9] = 2; // colour type: truecolour
  const chunks = [
    Uint8Array.from(PNG_SIGNATURE),
    pngChunk("IHDR", header),
    pngChunk("IDAT", new Uint8Array(deflateSync(raw, { level: 6 }))),
    pngChunk("IEND", new Uint8Array(0)),
  ];
  const total = chunks.reduce((sum, part) => sum + part.length, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of chunks) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

function paethPredictor(a: number, b: number, c: number): number {
  const p = a + b - c;
  const pa = Math.abs(p - a);
  const pb = Math.abs(p - b);
  const pc = Math.abs(p - c);
  if (pa <= pb && pa <= pc) return a;
  if (pb <= pc) return b;
  return c;
}

/**
 * Decode an 8-bit non-interlaced truecolour PNG (colour type 2 or 6) to RGB.
 *
 * Alpha is dropped rather than composited: every consumer here scores opaque
 * frames and plates, and inventing a matte would be a decision the caller did
 * not make.
 */
export function decodeRgbPng(bytes: Uint8Array): RgbImage {
  for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
    if (bytes[index] !== PNG_SIGNATURE[index]) throw new Error("not a PNG: signature mismatch");
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 8;
  let width = 0;
  let height = 0;
  let channels = 0;
  const idatParts: Uint8Array[] = [];
  let sawHeader = false;

  while (offset + 8 <= bytes.length) {
    const length = view.getUint32(offset);
    const type = String.fromCharCode(bytes[offset + 4], bytes[offset + 5], bytes[offset + 6], bytes[offset + 7]);
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = view.getUint32(offset + 8);
      height = view.getUint32(offset + 12);
      const bitDepth = bytes[offset + 16];
      const colourType = bytes[offset + 17];
      const interlace = bytes[offset + 20];
      if (bitDepth !== 8) throw new Error(`unsupported PNG bit depth ${bitDepth}; only 8 is decoded here`);
      if (colourType !== 2 && colourType !== 6) {
        throw new Error(`unsupported PNG colour type ${colourType}; only 2 (RGB) and 6 (RGBA) are decoded here`);
      }
      if (interlace !== 0) throw new Error("interlaced PNGs are not decoded here");
      channels = colourType === 6 ? 4 : 3;
      sawHeader = true;
    } else if (type === "IDAT") {
      idatParts.push(body);
    } else if (type === "IEND") {
      break;
    }
    offset += 12 + length;
  }
  if (!sawHeader) throw new Error("PNG has no IHDR");
  if (idatParts.length === 0) throw new Error("PNG has no IDAT");

  const compressedLength = idatParts.reduce((sum, part) => sum + part.length, 0);
  const compressed = new Uint8Array(compressedLength);
  {
    let cursor = 0;
    for (const part of idatParts) {
      compressed.set(part, cursor);
      cursor += part.length;
    }
  }
  const raw = new Uint8Array(inflateSync(compressed));
  const stride = width * channels;
  if (raw.length < height * (stride + 1)) {
    throw new Error(`PNG IDAT is short: ${raw.length} bytes for ${height} rows of ${stride + 1}`);
  }

  // Unfilter in place into a contiguous scanline buffer, then drop alpha.
  const unfiltered = new Uint8Array(height * stride);
  for (let y = 0; y < height; y += 1) {
    const filter = raw[y * (stride + 1)];
    const source = y * (stride + 1) + 1;
    const destination = y * stride;
    const previous = destination - stride;
    for (let x = 0; x < stride; x += 1) {
      const value = raw[source + x];
      const a = x >= channels ? unfiltered[destination + x - channels] : 0;
      const b = y > 0 ? unfiltered[previous + x] : 0;
      const c = y > 0 && x >= channels ? unfiltered[previous + x - channels] : 0;
      let out: number;
      switch (filter) {
        case 0: out = value; break;
        case 1: out = value + a; break;
        case 2: out = value + b; break;
        case 3: out = value + ((a + b) >> 1); break;
        case 4: out = value + paethPredictor(a, b, c); break;
        default: throw new Error(`unknown PNG filter type ${filter} on row ${y}`);
      }
      unfiltered[destination + x] = out & 0xff;
    }
  }

  if (channels === 3) return { width, height, rgb: unfiltered };
  const rgb = new Uint8Array(width * height * 3);
  for (let pixel = 0; pixel < width * height; pixel += 1) {
    rgb[pixel * 3] = unfiltered[pixel * 4];
    rgb[pixel * 3 + 1] = unfiltered[pixel * 4 + 1];
    rgb[pixel * 3 + 2] = unfiltered[pixel * 4 + 2];
  }
  return { width, height, rgb };
}
