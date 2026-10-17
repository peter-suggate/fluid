/** A bounded fitted point uses 10 bits per local coordinate, followed by an
 * unchanged f32 field value (DMC) or signs/normal word (DC). Position error is
 * at most 1/2046 of a cell per axis, below the mesh's 1/256-cell output grid. */
export const DUAL_GRID_WORDS_PER_VOXEL = 2;
export const DUAL_GRID_BYTES_PER_VOXEL = DUAL_GRID_WORDS_PER_VOXEL * 4;
export function packDualGridPoint(point: readonly number[]): number {
  if (point.length !== 3 || point.some(v => !Number.isFinite(v) || v < 0 || v > 1)) throw new RangeError("Dual point must lie in its cell");
  const q = point.map(v => Math.round(v * 1023));
  return (q[0]! | q[1]! << 10 | q[2]! << 20) >>> 0;
}
export function unpackDualGridPoint(word: number): [number, number, number] {
  return [word & 1023, word >>> 10 & 1023, word >>> 20 & 1023].map(v => v / 1023) as [number, number, number];
}
export const dualGridPointWGSL = /* wgsl */ `
fn packDualGridPoint(p:vec3f)->u32{let q=vec3u(round(clamp(p,vec3f(0),vec3f(1))*1023.));return q.x|(q.y<<10u)|(q.z<<20u);}
fn unpackDualGridPoint(p:u32)->vec3f{return vec3f(vec3u(p,p>>10u,p>>20u)&vec3u(1023u))/1023.;}
`;
