import { compileUniformStencil, compileUniformBlendMask } from "./uniform-compiled-topology";
import type { FluidRefinementRegion } from "../../core/model";
import { refinementRegionCellBounds, type RefinementRegionLattice } from "../../core/refinement-regions";

export const MIXED_FINE_TILE = 0x80000000;
export const MIXED_CELL_MASK = 0x3fffffff;
/** Owner width of a tile word: h (MIXED_FINE_TILE set) or 4h. */
export function mixedCellWidth(word: number): 1 | 4 {
  return word & MIXED_FINE_TILE ? 1 : 4;
}
type Triple = readonly [number, number, number];
export interface UniformMixedLayout {
  readonly lattice: RefinementRegionLattice;
  readonly tileDimensions: Triple;
  /** Upper bit: h (set) or 4h owner width. Remaining bits: first active cell. */
  readonly tiles: Uint32Array<ArrayBuffer>;
  readonly fineTiles: Uint32Array<ArrayBuffer>;
  readonly coarseTiles: Uint32Array<ArrayBuffer>;
  /** Per tile: the 27-bit h neighbourhood mask with the maximum width in its
   * high bits, then minimum width, detail-ring and compiled vertex recipes. */
  readonly stencils: Uint32Array<ArrayBuffer>;
  /** Minimal fine-neighbor boxes for the exact velocity blending distance. */
  readonly blendMasks: Uint32Array<ArrayBuffer>;
  readonly cellCount: number;
  readonly metadataBytes: number;
  /** Effective bounds in tile coordinates, upper bound exclusive. */
  readonly regions: readonly { id: string; min: Triple; max: Triple }[];
}

/** Manual Uniform ownership: h/4h, ungraded. Tiles outside regions take the
 * background width (h unless a coarse background is requested). */
export function createUniformMixedLayout(
  lattice: RefinementRegionLattice, regions: readonly FluidRefinementRegion[], backgroundWidth: 1 | 4 = 1,
  /** Tiles that must be h, from uniformMixedSolidTiles. Solids override
   * region cell-size bounds: forced tiles are h whatever a region says. */
  forcedFine?: Uint8Array,
): UniformMixedLayout {
  const axes = ["x", "y", "z"] as const;
  for (let a = 0; a < 3; a++) {
    if (!Number.isSafeInteger(lattice.dimensions[a]) || lattice.dimensions[a]! < 4
      || lattice.dimensions[a]! % 4 !== 0 || !Number.isFinite(lattice.cellSize_m[a])
      || lattice.cellSize_m[a]! <= 0 || !Number.isFinite(lattice.origin_m[axes[a]!])) {
      throw new Error("Mixed layout requires finite positive spacing and dimensions divisible by four");
    }
  }
  const dimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
  const count = dimensions[0] * dimensions[1] * dimensions[2];
  if (!Number.isSafeInteger(count) || count * 64 > MIXED_CELL_MASK) throw new Error("Mixed cell index capacity exceeded");
  // Temporary two-bit allowed-scale mask; discarded after topology construction.
  const allowed = new Uint8Array(count).fill(3);
  if (forcedFine && forcedFine.length !== count) throw new Error("Solid promotion mask does not match the tile lattice");
  if (forcedFine) for (let key = 0; key < count; key++) if (forcedFine[key]) allowed[key] = 1;
  const ids = new Set<string>();
  const snapped: { id: string; min: Triple; max: Triple }[] = [];
  for (const r of regions) {
    if (!r.id || ids.has(r.id)) throw new Error(`Duplicate or empty region id: ${r.id}`);
    ids.add(r.id);
    const lo = r.minimumCellSize_cells, hi = r.maximumCellSize_cells ?? Infinity;
    if (r.rule !== "minimum-cell-size" || ![1, 2, 4, 8, 16, 32].includes(lo)
      || (r.maximumCellSize_cells !== undefined && ![1, 2, 4, 8, 16, 32].includes(hi)) || hi < lo) {
      throw new Error(`Invalid cell-size bounds: ${r.id}`);
    }
    for (const axis of axes) {
      if (!Number.isFinite(r.min_m[axis]) || !Number.isFinite(r.max_m[axis]) || r.min_m[axis] >= r.max_m[axis]) {
        throw new Error(`Invalid region bounds: ${r.id}`);
      }
    }
    // Other methods' 8-32 cell regions are valid scene data: on this h/4h
    // lattice they mean the coarsest tier.
    const mask = (lo <= 1 && hi >= 1 ? 1 : 0) | (lo <= 4 && hi >= 4 ? 2 : 0) || 2;
    const bounds = refinementRegionCellBounds(r, lattice);
    if (bounds.min.some((v, a) => v >= bounds.max[a]!)) continue;
    // World-to-cell arithmetic can land a few ULPs past an exact tile edge.
    const tileCoordinate=(v:number)=>{const q=v/4,n=Math.round(q);return Math.abs(q-n)<=8*Number.EPSILON*Math.max(1,Math.abs(q))?n:q;};
    const min = bounds.min.map(v => Math.floor(tileCoordinate(v))) as unknown as Triple;
    const max = bounds.max.map(v => Math.ceil(tileCoordinate(v))) as unknown as Triple;
    for (let z = min[2]; z < max[2]; z++) for (let y = min[1]; y < max[1]; y++) for (let x = min[0]; x < max[0]; x++) {
      const key = x + dimensions[0] * (y + dimensions[1] * z);
      if (forcedFine?.[key]) continue;
      // Overlapping Fine and Coarse boxes: fine wins, as it does for solids.
      // Coarse is a cost hint; refusing here threw mid-advance in the app.
      allowed[key] = allowed[key]! & mask || 1;
    }
    snapped.push({ id: r.id, min, max });
  }
  const widths = Uint8Array.from(allowed, mask => mask === 1 ? 1 : mask === 2 ? 4 : backgroundWidth);
  return packUniformMixedLayout(lattice, widths, snapped);
}

/** Ownership from per-tile widths decided elsewhere (the detail planner's
 * admitted h tiles plus solid promotion), with the snapped authored boxes
 * as metadata. Precedence is the caller's, not region list order. */
export function createUniformMixedLayoutFromWidths(lattice: RefinementRegionLattice, widths: Uint8Array,
  regions: UniformMixedLayout["regions"]): UniformMixedLayout {
  if (lattice.dimensions.some(n => !Number.isSafeInteger(n) || n < 4 || n % 4 !== 0)) throw new Error("Mixed layout requires dimensions divisible by four");
  if (widths.length !== lattice.dimensions.reduce((n, d) => n * d / 4, 1)) throw new Error("Mixed tile widths do not match the lattice");
  return packUniformMixedLayout(lattice, widths, regions);
}

/** Solid promotion. A cut cell has open fraction below one: a voxel of the
 * interior lattice or a terrain column cell below its height. Every tile with
 * a cell within one cell (26-neighbourhood) of a cut cell is solid-coupled;
 * those and their 26 neighbour tiles are forced to h. Fine-owner stencils of
 * one cell (faces, dual cells, phi contact, pressure continuation) therefore
 * never meet solid in a 4h owner or in an interface (seam) tile. The box
 * shell lives in the mask's halo; domain walls are owned by every width. */
export function uniformMixedSolidTiles(dimensions: Triple, mask: Uint32Array, maskHeaderWords: number,
  terrainCells?: Float32Array): { forced: Uint8Array; coupled: Uint8Array; cutCells: number } {
  const [nx, ny, nz] = dimensions, t = dimensions.map(n => n / 4) as unknown as Triple;
  if (dimensions.some(n => !Number.isSafeInteger(n) || n % 4 !== 0)) throw new Error("Solid promotion requires dimensions divisible by four");
  const sx = nx + 2, sy = ny + 2, sz = nz + 2;
  if (mask[1] !== sx || mask[2] !== sy || mask[3] !== sz) throw new Error("Solid promotion mask does not match the lattice");
  const coupled = new Uint8Array(t[0] * t[1] * t[2]);let cutCells = 0;
  const mark = (x: number, y: number, z: number) => {
    cutCells++;
    for (let tz = Math.max(0, (z - 1) >> 2); tz <= Math.min(t[2] - 1, (z + 1) >> 2); tz++)
      for (let ty = Math.max(0, (y - 1) >> 2); ty <= Math.min(t[1] - 1, (y + 1) >> 2); ty++)
        for (let tx = Math.max(0, (x - 1) >> 2); tx <= Math.min(t[0] - 1, (x + 1) >> 2); tx++) coupled[tx + t[0] * (ty + t[1] * tz)] = 1;
  };
  for (let word = 0; word < mask.length - maskHeaderWords; word++) {
    let bits = mask[maskHeaderWords + word]!;
    while (bits) {
      const bit = 31 - Math.clz32(bits & -bits);bits &= bits - 1;
      const index = word * 32 + bit, qx = index % sx, qy = Math.floor(index / sx) % sy, qz = Math.floor(index / (sx * sy));
      const x = qx - 1, y = qy - 1, z = qz - 1;
      if (x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz) mark(x, y, z);
    }
  }
  if (terrainCells) {
    if (terrainCells.length !== nx * nz) throw new Error("Solid promotion terrain does not match the lattice");
    for (let z = 0; z < nz; z++) for (let x = 0; x < nx; x++) {
      const height = terrainCells[x + nx * z]!;
      if (!Number.isFinite(height)) throw new Error("Solid promotion requires finite terrain heights");
      // clamp(height-y,0,1)>0 exactly when y<height: cells 0..top are cut.
      if (!(height > 0)) continue;
      const top = Math.min(ny - 1, Math.ceil(height) - 1);cutCells += top + 1;
      for (let tz = Math.max(0, (z - 1) >> 2); tz <= Math.min(t[2] - 1, (z + 1) >> 2); tz++)
        for (let ty = 0; ty <= Math.min(t[1] - 1, (top + 1) >> 2); ty++)
          for (let tx = Math.max(0, (x - 1) >> 2); tx <= Math.min(t[0] - 1, (x + 1) >> 2); tx++) coupled[tx + t[0] * (ty + t[1] * tz)] = 1;
    }
  }
  const forced = new Uint8Array(coupled.length);
  for (let tz = 0; tz < t[2]; tz++) for (let ty = 0; ty < t[1]; ty++) for (let tx = 0; tx < t[0]; tx++) {
    if (!coupled[tx + t[0] * (ty + t[1] * tz)]) continue;
    for (let z = Math.max(0, tz - 1); z <= Math.min(t[2] - 1, tz + 1); z++) for (let y = Math.max(0, ty - 1); y <= Math.min(t[1] - 1, ty + 1); y++)
      for (let x = Math.max(0, tx - 1); x <= Math.min(t[0] - 1, tx + 1); x++) forced[x + t[0] * (y + t[1] * z)] = 1;
  }
  return { forced, coupled, cutCells };
}

/** Liquid-conditional solid promotion (the census's solidActive and
 * solidPromote on the CPU): a coupled `liquid` tile is active, and it and
 * its 26 neighbours are promoted. Dynamic coarsening
 * holds dry solids at 4h, so only these tiles of `forced` must be h. */
export function uniformMixedLiquidSolidPromotion(tileDimensions: Triple, coupled: Uint8Array, liquid: Uint8Array): Uint8Array {
  const [tx, ty, tz] = tileDimensions, n = tx * ty * tz;
  if (coupled.length !== n || liquid.length !== n) throw new Error("Liquid solid promotion masks do not match the tile lattice");
  const near = (mask: Uint8Array, x: number, y: number, z: number) => {
    for (let k = Math.max(0, z - 1); k <= Math.min(tz - 1, z + 1); k++) for (let j = Math.max(0, y - 1); j <= Math.min(ty - 1, y + 1); j++)
      for (let i = Math.max(0, x - 1); i <= Math.min(tx - 1, x + 1); i++) if (mask[i + tx * (j + ty * k)]) return true;
    return false;
  };
  const active = new Uint8Array(n), promoted = new Uint8Array(n);
  for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) {
    const t = x + tx * (y + ty * z);
    if (coupled[t] && liquid[t]) active[t] = 1;
  }
  for (let z = 0; z < tz; z++) for (let y = 0; y < ty; y++) for (let x = 0; x < tx; x++) if (near(active, x, y, z)) promoted[x + tx * (y + ty * z)] = 1;
  return promoted;
}

/** Loud ownership certificate: no 4h owner may cover a promoted tile. */
export function assertUniformMixedSolidPromotion(layout: UniformMixedLayout, forced: Uint8Array): void {
  if (forced.length !== layout.tiles.length) throw new Error("Solid promotion mask does not match the ownership layout");
  for (let key = 0; key < forced.length; key++) if (forced[key] && mixedCellWidth(layout.tiles[key]!) !== 1) {
    const d = layout.tileDimensions;
    throw new Error(`Mixed ownership places a ${mixedCellWidth(layout.tiles[key]!)}h owner on solid-coupled tile ${key % d[0]},${Math.floor(key / d[0]) % d[1]},${Math.floor(key / (d[0] * d[1]))}`);
  }
}

/** The all-4h layout on the same lattice: every pressure level's ownership.
 * Returns the layout itself when it is already all-4h. */
export function uniformMixedAllCoarseLayout(layout: UniformMixedLayout): UniformMixedLayout {
  if (layout.tiles.every(word => mixedCellWidth(word) === 4)) return layout;
  return packUniformMixedLayout(layout.lattice, new Uint8Array(layout.tiles.length).fill(4), layout.regions);
}

function packUniformMixedLayout(lattice: RefinementRegionLattice, widths: Uint8Array,
  regions: UniformMixedLayout["regions"]): UniformMixedLayout {
  const count = widths.length;
  const dimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
  const tiles = new Uint32Array(count);
  const fine: number[] = [], coarse: number[] = [];
  const fineCount = widths.reduce((n, width) => n + (width === 1 ? 1 : 0), 0);
  const cellCount = widths.reduce((n, width) => n + (4 / width) ** 3, 0);
  let fineBase = 0, coarseBase = fineCount * 64;
  for (let key = 0; key < count; key++) {
    const isFine = widths[key] === 1;
    if (!isFine && widths[key] !== 4) throw new Error("Mixed ownership is h/4h");
    tiles[key] = ((isFine ? fineBase : coarseBase) | (isFine ? MIXED_FINE_TILE : 0)) >>> 0;
    (isFine ? fine : coarse).push(key);
    if (isFine) fineBase += 64; else coarseBase += 1;
  }
  const stencils = mixedStencils(dimensions, widths);
  const blendMasks=Uint32Array.from({length:count},(_,i)=>compileUniformBlendMask(stencils[2*i]!));
  return { lattice, tileDimensions: dimensions, tiles, stencils, blendMasks, fineTiles: Uint32Array.from(fine),
    coarseTiles: Uint32Array.from(coarse), cellCount, metadataBytes: count * 20, regions };
}

/** Second stencil word, bit 0: the tile is in the detail ring, an h tile
 * within three tiles of it (itself included). The h store of a detail field
 * holds the canonical texels of ring tiles only (UniformDetailDomain); a 4h
 * tile outside the ring lives in its base block alone. */
export const UNIFORM_MIXED_DETAIL_RING = 1;
/** Freeze geometric stencil decisions with ownership. Physics stages reuse
 * these masks for arbitrary sample locations, including long departures. */
function mixedStencils(dimensions: Triple, widths: Uint8Array): Uint32Array<ArrayBuffer> {
  const count = widths.length, stencils = new Uint32Array(count * 2);
  for (let key = 0; key < count; key++) {
    const tx = key % dimensions[0], ty = Math.floor(key / dimensions[0]) % dimensions[1], tz = Math.floor(key / (dimensions[0] * dimensions[1]));
    let maximum = widths[key]!, minimum = widths[key]!, valid = 0;
    for (let z = -1; z <= 1; z++) for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) {
      const qx = tx + x, qy = ty + y, qz = tz + z;
      if (qx < 0 || qy < 0 || qz < 0 || qx >= dimensions[0] || qy >= dimensions[1] || qz >= dimensions[2]) continue;
      valid |= 1 << ((x + 1) + 3 * ((y + 1) + 3 * (z + 1)));
      const width = widths[qx + dimensions[0] * (qy + dimensions[1] * qz)]!;
      maximum = Math.max(maximum, width);minimum = Math.min(minimum, width);
      if (width === 1) stencils[2 * key]! |= 1 << ((x + 1) + 3 * ((y + 1) + 3 * (z + 1)));
    }
    stencils[2 * key]! |= maximum << 27;
    let farPositive = 0;
    const position = [tx, ty, tz], stride = [1, dimensions[0], dimensions[0] * dimensions[1]];
    for (let axis = 0; axis < 3; axis++) if (position[axis]! + 2 < dimensions[axis]! && widths[key + 2 * stride[axis]!] === 1) farPositive |= 1 << axis;
    stencils[2 * key + 1]! = (minimum << 27) | (farPositive << 24) | compileUniformStencil(stencils[2 * key]!, valid);
  }
  // The detail ring (UNIFORM_MIXED_DETAIL_RING): bit 0 of the second word,
  // an h tile within three tiles. A tile two tiles out sees the 27-bit h
  // masks of its radius-three neighbourhood (the GPU builder's `mirror`).
  for (let key = 0; key < count; key++) {
    const tx = key % dimensions[0], ty = Math.floor(key / dimensions[0]) % dimensions[1], tz = Math.floor(key / (dimensions[0] * dimensions[1]));
    let near = false;
    for (let z = -2; z <= 2 && !near; z += 2) for (let y = -2; y <= 2 && !near; y += 2) for (let x = -2; x <= 2 && !near; x += 2) {
      const qx = tx + x, qy = ty + y, qz = tz + z;
      if (qx < 0 || qy < 0 || qz < 0 || qx >= dimensions[0] || qy >= dimensions[1] || qz >= dimensions[2]) continue;
      near = (stencils[2 * (qx + dimensions[0] * (qy + dimensions[1] * qz))]! & 0x7ffffff) !== 0;
    }
    if (near) stencils[2 * key + 1]! |= UNIFORM_MIXED_DETAIL_RING;
  }
  return stencils;
}

/** A layout whose tile words were built elsewhere (the GPU layout builder).
 * Only the tile words are stored; worklists and stencils are derived from
 * them on first host access, so an adopted generation costs no per-tile host
 * work unless a host consumer actually reads those arrays. */
/** counts: [h, 4h] tile counts a GPU receipt already holds (no host scan);
 * building the tile lists fails fast if the words disagree with them. */
export function uniformMixedLayoutFromTiles(lattice: RefinementRegionLattice, tiles: Uint32Array<ArrayBuffer>,
  regions: UniformMixedLayout["regions"], counts?: readonly [number, number]): UniformMixedLayout {
  const dimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
  const count = dimensions[0] * dimensions[1] * dimensions[2];
  if (tiles.length !== count) throw new Error("Mixed tile words do not match the lattice");
  let fine = 0, coarse = 0;
  if (counts) {
    [fine, coarse] = counts;
    if (fine + coarse !== count) throw new Error(`Mixed tile counts ${counts} do not cover the lattice`);
  } else for (const word of tiles) { if (mixedCellWidth(word) === 1) fine++; else coarse++; }
  let lists: { fine: Uint32Array<ArrayBuffer>; coarse: Uint32Array<ArrayBuffer> } | undefined;
  let stencils: Uint32Array<ArrayBuffer> | undefined;
  let blendMasks: Uint32Array<ArrayBuffer> | undefined;
  const list = () => {
    if (!lists) {
      lists = { fine: new Uint32Array(fine), coarse: new Uint32Array(coarse) };
      let f = 0, c = 0;
      for (let key = 0; key < count; key++) {
        if (mixedCellWidth(tiles[key]!) === 1) { if (f === fine) throw new Error("Mixed tile words hold more h tiles than their count"); lists.fine[f++] = key; }
        else { if (c === coarse) throw new Error("Mixed tile words hold more 4h tiles than their count"); lists.coarse[c++] = key; }
      }
    }
    return lists;
  };
  return {
    lattice, tileDimensions: dimensions, tiles, regions, metadataBytes: count * 20,
    cellCount: fine * 64 + coarse,
    get fineTiles() { return list().fine; },
    get coarseTiles() { return list().coarse; },
    get stencils() { return stencils ??= mixedStencils(dimensions, Uint8Array.from(tiles, mixedCellWidth)); },
    get blendMasks() { return blendMasks ??= Uint32Array.from({length:count},(_,i)=>compileUniformBlendMask((stencils ??= mixedStencils(dimensions, Uint8Array.from(tiles, mixedCellWidth)))[2*i]!)); },
  };
}
