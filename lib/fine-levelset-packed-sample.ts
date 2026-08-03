/**
 * Persistent fine-level-set sample ABI.
 *
 * One B4 brick contains 64 of these words, so its authoritative payload is
 * exactly 256 bytes.  The low half is IEEE binary16 phi.  The high half keeps
 * the three persistent sample-state bits plus the cached interface crossing
 * used by the closest-point transform.  JFA ping-pong identities deliberately
 * remain in transient u32 scratch; they are not part of this page ABI.
 */

export const FINE_LEVELSET_PACKED_SAMPLE_BYTES = 4;
export const FINE_LEVELSET_PACKED_PHI_MASK = 0xffff;

function float32ToFloat16Bits(value: number): number {
  const source = new Float32Array(1);
  const bits = new Uint32Array(source.buffer);
  source[0] = value;
  const word = bits[0]!;
  const sign = (word >>> 16) & 0x8000;
  const exponent = (word >>> 23) & 0xff;
  const fraction = word & 0x7fffff;
  if (exponent === 0xff) return sign | (fraction === 0 ? 0x7c00 : 0x7e00);
  const halfExponent = exponent - 127 + 15;
  if (halfExponent >= 0x1f) return sign | 0x7c00;
  if (halfExponent <= 0) {
    if (halfExponent < -10) return sign;
    const significand = fraction | 0x800000;
    const shift = 14 - halfExponent;
    const rounded = (significand + (1 << (shift - 1)) - 1
      + ((significand >>> shift) & 1)) >>> shift;
    return sign | rounded;
  }
  const roundedFraction = fraction + 0xfff + ((fraction >>> 13) & 1);
  if ((roundedFraction & 0x800000) !== 0) {
    const promoted = halfExponent + 1;
    return sign | (promoted >= 0x1f ? 0x7c00 : promoted << 10);
  }
  return sign | (halfExponent << 10) | (roundedFraction >>> 13);
}

export function float16BitsToFloat32(bits: number): number {
  const word = bits & 0xffff;
  const sign = (word & 0x8000) << 16;
  const exponent = (word >>> 10) & 0x1f;
  const fraction = word & 0x3ff;
  let result: number;
  if (exponent === 0) {
    if (fraction === 0) result = sign;
    else {
      let mantissa = fraction, shift = 0;
      while ((mantissa & 0x400) === 0) { mantissa <<= 1; shift += 1; }
      result = sign | ((113 - shift) << 23) | ((mantissa & 0x3ff) << 13);
    }
  } else if (exponent === 0x1f) {
    result = sign | 0x7f80_0000 | (fraction << 13);
  } else {
    result = sign | ((exponent + 112) << 23) | (fraction << 13);
  }
  const storage = new Uint32Array([result >>> 0]);
  return new Float32Array(storage.buffer)[0]!;
}

export function unpackFineLevelSetPackedFlags(packed: number): number {
  const encoded = packed >>> 16;
  const persistent = (encoded & 3) | ((encoded & 4) << 2);
  if ((encoded & 8) === 0) return persistent >>> 0;
  const direction = (encoded >>> 4) & 7;
  const fraction9 = (encoded >>> 7) & 0x1ff;
  const fraction24 = fraction9 << 15;
  return (persistent | (direction << 29) | (fraction24 << 5)) >>> 0;
}

export function unpackFineLevelSetPackedPhi(packed: number): number {
  return float16BitsToFloat32(packed);
}

/** Persistent flags are VALID, INTERFACE and NEGATIVE.  The former KNOWN and
 * TRIAL bits belonged to the retired serial march.  Nine fraction bits leave
 * a worst-case cached crossing displacement below 0.001 fine cells. */
export function encodeFineLevelSetPackedFlags(flags: number): number {
  const word = flags >>> 0;
  const closest = word >>> 5;
  const present = closest !== 0 ? 1 : 0;
  const direction = (closest >>> 24) & 7;
  const fraction = closest & 0x00ff_ffff;
  const fraction9 = Math.min(0x1ff, (fraction + 0x3fff) >>> 15);
  return ((word & 1) | (word & 2) | ((word >>> 2) & 4)
    | (present << 3) | (direction << 4) | (fraction9 << 7)) >>> 0;
}

export function packFineLevelSetSample(phi: number, flags: number): number {
  return (float32ToFloat16Bits(phi) | (encodeFineLevelSetPackedFlags(flags) << 16)) >>> 0;
}

export function packFineLevelSetSamples(
  phi: ArrayLike<number>,
  flags: ArrayLike<number>,
): Uint32Array {
  if (phi.length !== flags.length) throw new RangeError("Fine packed phi/flags lengths disagree");
  const packed = new Uint32Array(phi.length);
  for (let index = 0; index < packed.length; index += 1) {
    packed[index] = packFineLevelSetSample(Number(phi[index]), Number(flags[index]));
  }
  return packed;
}

/** WGSL helpers for an `array<u32>` named by the caller. */
export function fineLevelSetPackedSampleWGSL(samples: string, writable = false, prefix = "fine"): string {
  return /* wgsl */ `
fn ${prefix}PackedFlags(index:u32)->u32{
 let encoded=${samples}[index]>>16u;
 let persistent=(encoded&3u)|((encoded&4u)<<2u);
 if((encoded&8u)==0u){return persistent;}
 let direction=(encoded>>4u)&7u;let fraction9=(encoded>>7u)&0x1ffu;
 let fraction24=fraction9<<15u;
 return persistent|(direction<<29u)|(fraction24<<5u);
}
fn ${prefix}PackedPhi(index:u32)->f32{return unpack2x16float(${samples}[index]).x;}
fn ${prefix}EncodedFlags(flags:u32)->u32{
 let closest=flags>>5u;let present=select(0u,1u,closest!=0u);
 let direction=(closest>>24u)&7u;let fraction=closest&0x00ffffffu;
 let fraction9=min(0x1ffu,(fraction+0x3fffu)>>15u);
 return (flags&3u)|((flags>>2u)&4u)|(present<<3u)|(direction<<4u)|(fraction9<<7u);
}
fn ${prefix}PackedWord(value:f32,flags:u32)->u32{
 return (pack2x16float(vec2f(value,0.))&0xffffu)|(${prefix}EncodedFlags(flags)<<16u);
}
${writable ? `fn ${prefix}WritePacked(index:u32,value:f32,flags:u32){${samples}[index]=${prefix}PackedWord(value,flags);}
fn ${prefix}WritePackedPhi(index:u32,value:f32){${samples}[index]=${prefix}PackedWord(value,${prefix}PackedFlags(index));}
fn ${prefix}WritePackedFlags(index:u32,flags:u32){${samples}[index]=${prefix}PackedWord(${prefix}PackedPhi(index),flags);}` : ""}
`;
}
