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

/** Perspective maps cover forward cones. Wide/invalid cones and excess lights
 * retain exact visibility; the specialization uses this same capability check. */
export function rasterSpotLights(records?: Uint32Array): RasterSpotLight[] {
  if (!records || records.length % SVO_LIGHT_RECORD_WORDS) return [];
  const f = new Float32Array(records.buffer, records.byteOffset, records.length);
  const result: RasterSpotLight[] = [];
  for (let offset = 0; offset < records.length && result.length < RASTER_MAX_SPOT_LIGHTS; offset += SVO_LIGHT_RECORD_WORDS) {
    if (records[offset + 24] !== SVO_LIGHT_KINDS.spot) continue;
    const position = Array.from(f.subarray(offset, offset + 3));
    const direction = Array.from(f.subarray(offset + 4, offset + 7));
    const far = f[offset + 3]!, cosine = f[offset + 7]!, radius = f[offset + 20]!;
    const length = Math.hypot(...direction), near = Math.max(.001, radius * 1.01);
    if (![...position, ...direction, far, cosine, radius].every(Number.isFinite)
      || length < 1e-6 || cosine <= .05 || cosine >= 1 || radius < 0 || far <= near) continue;
    result.push({ index: offset / SVO_LIGHT_RECORD_WORDS, position,
      direction: direction.map(v => v / length), near, far, radius,
      tanHalfAngle: Math.sqrt(1 - cosine * cosine) / cosine });
  }
  return result;
}
