import { UNIFORM_NEAR_SPAN, uniformNearFrom } from "./uniform-lattice-address";

/** The 4h velocity cache by block: where UniformMixedMomentumCache stores a
 * tile's three +face 4h velocities, and what its readers load.
 *
 * An rgba32float texture of 10 texels an axis for every block of the block
 * grid (umBlockCoord): block b's page has its 8³ tiles at local texels 1..8
 * and one tile around them at 0 and 9. Channel a of a tile's texel is the 4h
 * velocity of that tile's +face along a. A reader resolves the block of its
 * own page once (umCoarseOrigin) and reaches every tap of a 4h stencil at a
 * plain offset from it: a tap reads no table, and neither clamps nor
 * branches.
 *
 * The halo holds what a tap from this page reads past a face
 * (uniformCoarseCacheSource names the tile whose faces it holds):
 * - a stored neighbour: that tile's own faces;
 * - a SOLID neighbour: the tap is held at the page face, so the faces of the
 *   page's own edge tile (the block neighbour rows' rule, diagonals
 *   included): what the box clamp gave at a closed wall;
 * - an AIR neighbour: never held. Nothing is stored there, so zero;
 * - below the page along the channel's own axis, whatever lies there: the
 *   tile itself, whose +face is the low face of the page's first tile. It is
 *   that face's stored value (the low-face table where nothing is stored
 *   below), never a copy of the face above it.
 * A face with no stored cell on either side is zero. The blocks that hold no
 * page (block 0, the last block, a free one) are never written: a reader
 * whose own page is not stored resolves to block 0 or the last block and
 * reads zero. */
export const UNIFORM_COARSE_CACHE_EDGE = 10;
/** The highest 4h sample coordinate, in tiles from the page's low corner less
 * the MAC offset, a reader may take: the f32 below 8, so its upper tap is the
 * halo tile and nothing past it. */
export const UNIFORM_COARSE_CACHE_REACH = "7.9999995";
type Grid = { readonly blockGrid: readonly number[] };
type Triple = [number, number, number];
/** The cache texture's size for a block grid. */
export function uniformCoarseCacheExtent(address: Grid): Triple {
  return [UNIFORM_COARSE_CACHE_EDGE * address.blockGrid[0]!, UNIFORM_COARSE_CACHE_EDGE * address.blockGrid[1]!, UNIFORM_COARSE_CACHE_EDGE * address.blockGrid[2]!];
}
/** Bytes of one cache texture: 16,000 a block of the grid. */
export const uniformCoarseCacheBytes = (address: Grid) => 16 * UNIFORM_COARSE_CACHE_EDGE ** 3 * address.blockGrid[0]! * address.blockGrid[1]! * address.blockGrid[2]!;
/** Workgroups of the cache launch, 5³ lanes each: one lane a texel. */
export function uniformCoarseCacheGroups(address: Grid): Triple {
  return [2 * address.blockGrid[0]!, 2 * address.blockGrid[1]!, 2 * address.blockGrid[2]!];
}

/** For a reader's module, after the lattice address: the texel of the first
 * tile of block b's page. A tile `local` tiles from the page's low corner
 * (-1..8 an axis) is at umCoarseOrigin(b) + local. */
export const uniformCoarseCacheWGSL = /* wgsl */ `
fn umCoarseOrigin(b:u32)->vec3i{return ${UNIFORM_COARSE_CACHE_EDGE}*umBlockCoord(b)+vec3i(1);}
`;

/** umCoarseOrigin on the host, written to out. */
export function uniformCoarseCacheOrigin(blockBits: number, block: number, out: Triple): Triple {
  const m = (1 << blockBits) - 1;
  out[0] = UNIFORM_COARSE_CACHE_EDGE * (block & m) + 1; out[1] = UNIFORM_COARSE_CACHE_EDGE * ((block >> blockBits) & m) + 1; out[2] = UNIFORM_COARSE_CACHE_EDGE * (block >> (2 * blockBits)) + 1;
  return out;
}
/** The block whose 10³ texels hold a texel of the cache (the cache kernel's
 * own decode); it may be a block that holds no page, or one past the pool. */
export function uniformCoarseCacheBlock(blockBits: number, x: number, y: number, z: number): number {
  const e = UNIFORM_COARSE_CACHE_EDGE;
  return Math.floor(x / e) | (Math.floor(y / e) << blockBits) | (Math.floor(z / e) << (2 * blockBits));
}
const near: Triple = [0, 0, 0];
/** The tile whose +face along axis the cache kernel stores in channel axis
 * of the texel `tile` (a tile of home's page or of the ring around it, in
 * tiles), written to out. rows: the block neighbour rows; page: home's page.
 * last: the box's last tile an axis where the box ends inside a page (a tap
 * past it is held there); absent for a box of whole pages. */
export function uniformCoarseCacheSource(rows: ArrayLike<number>, home: number, page: readonly number[], tile: readonly number[], axis: number, out: Triple, last?: readonly number[]): Triple {
  uniformNearFrom(rows, home, page, UNIFORM_NEAR_SPAN.tile, tile, near);
  for (let a = 0; a < 3; a++) out[a] = last ? Math.min(near[a]!, last[a]!) : near[a]!;
  if (tile[axis]! < page[axis]! << 3) out[axis] = tile[axis]!;
  return out;
}
/** umSampleVelocity4In on the host, in f32: the 4h interpolant of component
 * axis at p (cells) from the cache block at origin (uniformCoarseCacheOrigin)
 * of the page `page` its reader resolved. load(x, y, z, axis): a texel's
 * channel. */
export function uniformCoarseCacheSample(load: (x: number, y: number, z: number, axis: number) => number, origin: readonly number[], page: readonly number[], p: readonly number[], axis: number): number {
  const f = Math.fround, reach = f(Number(UNIFORM_COARSE_CACHE_REACH)), base: Triple = [0, 0, 0], t: Triple = [0, 0, 0], v: number[] = [];
  for (let a = 0; a < 3; a++) {
    const local = f(f(p[a]! / 4) - 8 * page[a]!), q = Math.min(Math.max(f(local - (a === axis ? 1 : 0.5)), -1), reach);
    base[a] = Math.floor(q); t[a] = f(q - base[a]!);
  }
  for (let k = 0; k < 8; k++) v.push(load(origin[0]! + base[0] + (k & 1), origin[1]! + base[1] + ((k >> 1) & 1), origin[2]! + base[2] + (k >> 2), axis));
  const mix = (a: number, b: number, w: number) => f(f(f(1 - w) * a) + f(w * b));
  return mix(mix(mix(v[0]!, v[1]!, t[0]), mix(v[2]!, v[3]!, t[0]), t[1]), mix(mix(v[4]!, v[5]!, t[0]), mix(v[6]!, v[7]!, t[0]), t[1]), t[2]);
}
