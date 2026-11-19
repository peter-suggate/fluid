/**
 * The static mask's exact 4h tiles (2026-11-19): the old engine's umSolidGeometryExact on the brick lattice, read once
 * from the shared SolidOccupancyMask image (the scene's SolidWorld, the hit-tested and edited source) and kept beside
 * it, two bits a lattice tile in the directory (brick-abi.ts NB_D_CLOSED, NB_D_INEXACT).
 *
 * **Closed.** A tile in the box whose 64 cells are all mask voxels: a wall to the 4h tier, as the cells past the box
 * are (the old engine's exact walls are such tiles). Its tile test and its cell test (brick-abi.ts resolveBox:
 * nbTileInWallBox, nbInWallC) read it as outside: the coarse level set, the native rows (a closed neighbour across a
 * face is the wall slot, fraction 0.5: umPressureFaceV's voxel-wall face) and the static-cut latches. The
 * full-resolution tier keeps the box (nbInWallBox): its cells see the same voxels through the mask.
 *
 * **Exact.** A closed tile, or an open tile (no mask voxel in its 64 cells) each of whose six neighbours across a face
 * is closed (off the box counts closed) or offers an all-open 4x4 layer to the face: every 4h face patch is then
 * whole (1, the layer open) or a wall (the closed neighbour), so the 4h restriction loses nothing (umCoarseSolidFaceExact
 * with face values 1 and 0.5 only). Stricter than the old test in one way, on purpose: a uniform solid layer of a
 * neighbour that is not closed is inexact here, since only a closed tile is a 4h wall.
 *
 * **Inexact (the static need's support, NB_D_INEXACT).** A tile with a mask voxel among the 6^3 cells of the tile and
 * its one-cell shell (umSolidTile: brick-static-need.wgsl.ts's voxel test) that is not exact. The static need
 * promotes only these (the old solidActive's !umSolidGeometryExact): an exact tile near the solid stays 4h.
 *
 * Terrain and the procedural boundary keep their own GPU support tests (sdTerrainTile, sdFieldTile): this image is
 * the voxels'. An engine with no static mask at the build (or a deferred, empty one) does not bind them: the plain box
 * rule, its kernel text byte-identical. A live edit recomputes the tiles its cells can reach (brickStaticTilesEdit).
 */
import { SOLID_MASK_EMPTY, SOLID_MASK_FULL, solidOccupancyMaskPage, solidOccupancyMaskVoxel } from "../../core/solid-occupancy-mask";

export interface BrickStaticTiles {
  /** Tiles a lattice axis (lattice / 4). */
  readonly tiles: readonly [number, number, number];
  /** Words a bit plane: ceil(tiles / 32), x fastest. */
  readonly words: number;
  /** [closed plane][inexact plane], 2 * words. */
  readonly bits: Uint32Array;
  /** Closed tiles. */
  closed: number;
}

/** The static tiles of a lattice (whole tiles an axis) and a box, from the mask image's words. */
export function brickStaticTiles(lattice: readonly number[], box: readonly number[], mask: ArrayLike<number>): BrickStaticTiles {
  if (lattice.some(n => n % 4 !== 0)) throw new Error(`Static tiles: the lattice ${lattice.join("x")} is not whole 4h tiles`);
  const tiles = lattice.map(n => n / 4) as [number, number, number], count = tiles[0] * tiles[1] * tiles[2];
  const words = Math.ceil(count / 32);
  const t: BrickStaticTiles = { tiles, words, bits: new Uint32Array(2 * words), closed: 0 };
  brickStaticTilesUpdate(t, box, mask, [0, 0, 0], tiles);
  return t;
}

/** Recompute tiles [lo, hi) (tile coordinates, clamped), in place. Returns the word range of each plane touched. */
export function brickStaticTilesUpdate(t: BrickStaticTiles, box: readonly number[], mask: ArrayLike<number>, lo: readonly number[], hi: readonly number[]): { first: number; last: number } {
  const [TX, TY, TZ] = t.tiles, [BX, BY, BZ] = box as [number, number, number];
  const inBox = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < BX && y < BY && z < BZ;
  // A mask voxel (in the box only, as nbStaticVoxel there); shut: a voxel or off the box.
  const voxel = (x: number, y: number, z: number) => inBox(x, y, z) && solidOccupancyMaskVoxel(mask, x, y, z);
  const shut = (x: number, y: number, z: number) => !inBox(x, y, z) || solidOccupancyMaskVoxel(mask, x, y, z);
  const code = (x: number, y: number, z: number) => solidOccupancyMaskPage(mask, x, y, z) & 0xffff;
  const emptyPage = (x: number, y: number, z: number) => !inBox(x, y, z) || code(x, y, z) === SOLID_MASK_EMPTY;
  const fullPage = (x: number, y: number, z: number) => !inBox(x, y, z) || code(x, y, z) >= SOLID_MASK_FULL;
  const tileIn = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && 4 * x + 4 <= BX && 4 * y + 4 <= BY && 4 * z + 4 <= BZ;
  const closedTile = (x: number, y: number, z: number): boolean => {
    if (!tileIn(x, y, z)) return true; // off the box: the box rule's solid
    for (let k = 0; k < 64; k++) if (!voxel(4 * x + (k & 3), 4 * y + ((k >> 2) & 3), 4 * z + (k >> 4))) return false;
    return true;
  };
  const a = lo.map(v => Math.max(0, v)), b = [Math.min(TX, hi[0]!), Math.min(TY, hi[1]!), Math.min(TZ, hi[2]!)];
  let first = Infinity, last = -1;
  const set = (plane: number, i: number, on: boolean) => {
    const w = plane * t.words + (i >>> 5), m = 1 << (i & 31), was = (t.bits[w]! & m) !== 0;
    if (was === on) return;
    t.bits[w] = on ? t.bits[w]! | m : t.bits[w]! & ~m;
    if (plane === 0) t.closed += on ? 1 : -1;
    first = Math.min(first, i >>> 5); last = Math.max(last, i >>> 5);
  };
  for (let z = a[2]!; z < b[2]!; z++) for (let y = a[1]!; y < b[1]!; y++) for (let x = a[0]!; x < b[0]!; x++) {
    const i = x + TX * (y + TY * z), o = [4 * x, 4 * y, 4 * z] as const;
    // The shell's pages (at most two an axis, so its 8 corners name them all): all empty, so no voxel lies near the
    // tile; all full and the tile in the box, so the tile is closed (a page is whole tiles).
    let empty = true, full = tileIn(x, y, z);
    for (let q = 0; q < 8 && (empty || full); q++) {
      const c = [o[0] - 1 + 5 * (q & 1), o[1] - 1 + 5 * ((q >> 1) & 1), o[2] - 1 + 5 * (q >> 2)] as const;
      empty = empty && emptyPage(...c); full = full && fullPage(...c);
    }
    if (full) { set(0, i, true); set(1, i, false); continue; }
    const closed = !empty && tileIn(x, y, z) && closedTile(x, y, z);
    set(0, i, closed);
    if (empty || !tileIn(x, y, z)) { set(1, i, false); continue; }
    let near = false;
    for (let l = 0; l < 216 && !near; l++) near = voxel(o[0] - 1 + (l % 6), o[1] - 1 + (Math.floor(l / 6) % 6), o[2] - 1 + Math.floor(l / 36));
    set(1, i, near && !closed && !brickStaticTileOpenExact(o, shut, closedTile, [x, y, z]));
  }
  return { first, last };
}

/**
 * A live edit's moved cells [lo, hi) (cells, exclusive): a tile's bits read its cells, its one-cell shell and its
 * neighbours' face layers and cells, so the tiles of [lo - 5, hi + 5) cells are recomputed. Returns the word range.
 */
export function brickStaticTilesEdit(t: BrickStaticTiles, box: readonly number[], mask: ArrayLike<number>, lo: readonly number[], hi: readonly number[]) {
  return brickStaticTilesUpdate(t, box, mask, lo.map(v => Math.floor((v - 5) / 4)), hi.map(v => Math.ceil((v + 5) / 4)));
}

/** An open tile's exactness (the module's rule): no shut cell in it, and each face's neighbour closed or its layer open. */
function brickStaticTileOpenExact(o: readonly [number, number, number], shut: (x: number, y: number, z: number) => boolean, closedTile: (x: number, y: number, z: number) => boolean, T: readonly number[]): boolean {
  for (let k = 0; k < 64; k++) if (shut(o[0] + (k & 3), o[1] + ((k >> 2) & 3), o[2] + (k >> 4))) return false;
  for (let f = 0; f < 6; f++) {
    const ax = f >> 1, side = f & 1, n = [...T]; n[ax]! += side ? 1 : -1;
    if (closedTile(n[0]!, n[1]!, n[2]!)) continue;
    const u = (ax + 1) % 3, v = (ax + 2) % 3;
    for (let m = 0; m < 16; m++) {
      const c = [...o]; c[ax]! += side ? 4 : -1; c[u]! += m & 3; c[v]! += m >> 2;
      if (shut(c[0]!, c[1]!, c[2]!)) return false;
    }
  }
  return true;
}
