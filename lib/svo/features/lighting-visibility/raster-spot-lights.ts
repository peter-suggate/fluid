import { SVO_LIGHT_KINDS, SVO_LIGHT_RECORD_WORDS } from "../../contracts/svo-light-abi";

export const RASTER_MAX_SPOT_LIGHTS = 4;
export interface RasterSpotLight {
  index: number;
  position: number[];
  direction: number[];
  near: number;
  far: number;
  tanHalfAngle: number;
  radius: number;
}

const lightViews = new WeakMap<Uint32Array, Float32Array>();
export function rasterLightValues(records: Uint32Array): Float32Array {
  let view = lightViews.get(records);
  if (!view) { view = new Float32Array(records.buffer, records.byteOffset, records.length); lightViews.set(records, view); }
  return view;
}

/** Shared allocation-free eligibility predicate for maps and live shading. */
export function rasterSpotLightEligible(records: Uint32Array, f: Float32Array, offset: number): boolean {
  if (records[offset + 24] !== SVO_LIGHT_KINDS.spot) return false;
  for (let i = 0; i < 8; i++) if (!Number.isFinite(f[offset + i])) return false;
  const radius = f[offset + 20]!, cosine = f[offset + 7]!;
  return Number.isFinite(radius) && radius >= 0 && cosine > .05 && cosine < 1
    && f[offset + 3]! > Math.max(.001, radius * 1.01)
    && Math.hypot(f[offset + 4]!, f[offset + 5]!, f[offset + 6]!) >= 1e-6;
}

/** Perspective maps cover forward cones. Wide/invalid cones and excess lights
 * retain exact visibility; the specialization uses this same capability check. */
export function rasterSpotLights(records?: Uint32Array): RasterSpotLight[] {
  if (!records || records.length % SVO_LIGHT_RECORD_WORDS) return [];
  const f = rasterLightValues(records);
  const result: RasterSpotLight[] = [];
  for (let offset = 0; offset < records.length && result.length < RASTER_MAX_SPOT_LIGHTS; offset += SVO_LIGHT_RECORD_WORDS) {
    if (!rasterSpotLightEligible(records, f, offset)) continue;
    const position = Array.from(f.subarray(offset, offset + 3));
    const direction = Array.from(f.subarray(offset + 4, offset + 7));
    const far = f[offset + 3]!, cosine = f[offset + 7]!, radius = f[offset + 20]!;
    const length = Math.hypot(...direction), near = Math.max(.001, radius * 1.01);
    result.push({ index: offset / SVO_LIGHT_RECORD_WORDS, position,
      direction: direction.map(v => v / length), near, far, radius,
      tanHalfAngle: Math.sqrt(1 - cosine * cosine) / cosine });
  }
  return result;
}
