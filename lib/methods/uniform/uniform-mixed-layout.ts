import type { FluidRefinementRegion } from "../../core/model";
import { refinementRegionCellBounds, type RefinementRegionLattice } from "../../core/refinement-regions";

export const MIXED_FINE_TILE = 0x80000000;
const MIXED_TWO_TILE = 0x40000000;
export const MIXED_CELL_MASK = 0x3fffffff;
export function mixedCellWidth(word: number): 1 | 2 | 4 {
  return word & MIXED_FINE_TILE ? 1 : word & MIXED_TWO_TILE ? 2 : 4;
}
type Triple = readonly [number, number, number];
export interface UniformMixedLayout {
  readonly lattice: RefinementRegionLattice;
  readonly tileDimensions: Triple;
  /** Upper bits: h/2h/4h owner width. Remaining bits: first active cell. */
  readonly tiles: Uint32Array<ArrayBuffer>;
  readonly fineTiles: Uint32Array<ArrayBuffer>;
  readonly coarseTiles: Uint32Array<ArrayBuffer>;
  readonly transitionTiles: Uint32Array<ArrayBuffer>;
  /** Per tile: 27-bit h/2h neighborhood masks; high bits hold maximum/minimum widths. */
  readonly stencils: Uint32Array<ArrayBuffer>;
  readonly cellCount: number;
  readonly metadataBytes: number;
  /** Effective bounds in tile coordinates, upper bound exclusive. */
  readonly regions: readonly { id: string; min: Triple; max: Triple }[];
}

/** Manual Uniform ownership, fine outside regions and strongly graded across faces/edges/corners.
 * An explicit coarse background is available for endpoint/numerical QA fixtures.
 * Disabling grading is reserved for direct 4:1 numerical stress tests. */
export function createUniformMixedLayout(
  lattice: RefinementRegionLattice, regions: readonly FluidRefinementRegion[], stronglyBalanced = true, backgroundWidth: 1 | 4 = 1,
  /** Tiles that must be h, from uniformMixedSolidTiles. Solids override
   * region cell-size bounds: forced tiles are h whatever a region says, and
   * a 4h-only region tile beside one is graded to 2h. */
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
    const mask = (lo <= 1 && hi >= 1 ? 1 : 0) | (lo <= 4 && hi >= 4 ? 2 : 0);
    if (!mask) throw new Error(`Region ${r.id} allows neither 1 nor 4 cell size`);
    const bounds = refinementRegionCellBounds(r, lattice);
    if (bounds.min.some((v, a) => v >= bounds.max[a]!)) throw new Error(`Region ${r.id} does not overlap the lattice`);
    // World-to-cell arithmetic can land a few ULPs past an exact tile edge.
    const tileCoordinate=(v:number)=>{const q=v/4,n=Math.round(q);return Math.abs(q-n)<=8*Number.EPSILON*Math.max(1,Math.abs(q))?n:q;};
    const min = bounds.min.map(v => Math.floor(tileCoordinate(v))) as unknown as Triple;
    const max = bounds.max.map(v => Math.ceil(tileCoordinate(v))) as unknown as Triple;
    for (let z = min[2]; z < max[2]; z++) for (let y = min[1]; y < max[1]; y++) for (let x = min[0]; x < max[0]; x++) {
      const key = x + dimensions[0] * (y + dimensions[1] * z);
      if (forcedFine?.[key]) continue;
      allowed[key] = allowed[key]! & mask;
      if (!allowed[key]) throw new Error(`Conflicting snapped region constraints at tile ${x},${y},${z} while applying ${r.id}`);
    }
    snapped.push({ id: r.id, min, max });
  }
  const widths = Uint8Array.from(allowed, mask => mask === 1 ? 1 : mask === 2 ? 4 : backgroundWidth);
  if (stronglyBalanced) {
    // Include edges and corners: face balance alone does not cover a box stencil.
    for (let key = 0; key < count; key++) if (widths[key] === 1) {
      const t = [key % dimensions[0], Math.floor(key / dimensions[0]) % dimensions[1], Math.floor(key / (dimensions[0] * dimensions[1]))];
      for (let z = -1; z <= 1; z++) for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) {
        const q = [t[0]! + x, t[1]! + y, t[2]! + z];
        if (q.some((v, a) => v < 0 || v >= dimensions[a]!)) continue;
        const neighbor = q[0]! + dimensions[0] * (q[1]! + dimensions[1] * q[2]!);
        if (widths[neighbor] !== 4 || widths[key] !== 1) continue;
        if (allowed[neighbor] !== 2 || forcedFine?.[key]) widths[neighbor] = 2;
        else if (allowed[key] !== 1) widths[key] = 2;
        else throw new Error(`Forced 4h tile ${q} conflicts with 2:1 grading around ${t}`);
      }
    }
  }
  return packUniformMixedLayout(lattice, widths, snapped);
}

/** Solid promotion. A cut cell has open fraction below one: a voxel of the
 * interior lattice or a terrain column cell below its height. Every tile with
 * a cell within one cell (26-neighbourhood) of a cut cell is solid-coupled;
 * those and their 26 neighbour tiles are forced to h. Fine-owner stencils of
 * one cell (faces, dual cells, phi contact, pressure continuation) therefore
 * never meet solid in a 2h/4h owner or in an interface (seam) tile. The box
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

/** Loud ownership certificate: no 2h/4h owner may cover a promoted tile. */
export function assertUniformMixedSolidPromotion(layout: UniformMixedLayout, forced: Uint8Array): void {
  if (forced.length !== layout.tiles.length) throw new Error("Solid promotion mask does not match the ownership layout");
  for (let key = 0; key < forced.length; key++) if (forced[key] && mixedCellWidth(layout.tiles[key]!) !== 1) {
    const d = layout.tileDimensions;
    throw new Error(`Mixed ownership places a ${mixedCellWidth(layout.tiles[key]!)}h owner on solid-coupled tile ${key % d[0]},${Math.floor(key / d[0]) % d[1]},${Math.floor(key / (d[0] * d[1]))}`);
  }
}

/** The two mixed multigrid levels keep existing coarse owners intact while
 * coarsening smaller owners. At 4h the lattice is uniform again. This is
 * pressure hierarchy geometry, not a change to manual enforcement regions. */
export function uniformMixedPressureLevel(layout: UniformMixedLayout, minimumWidth: 2 | 4): UniformMixedLayout {
  if (minimumWidth !== 2 && minimumWidth !== 4) throw new Error("Invalid mixed pressure level width");
  const widths = Uint8Array.from(layout.tiles, word => Math.max(minimumWidth, mixedCellWidth(word)));
  if (widths.every((width, tile) => width === mixedCellWidth(layout.tiles[tile]!))) return layout;
  return packUniformMixedLayout(layout.lattice, widths, layout.regions);
}

function packUniformMixedLayout(lattice: RefinementRegionLattice, widths: Uint8Array,
  regions: UniformMixedLayout["regions"]): UniformMixedLayout {
  const count = widths.length;
  const dimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
  const tiles = new Uint32Array(count);
  const fine: number[] = [], transition: number[] = [], coarse: number[] = [];
  const fineCount = widths.reduce((n, width) => n + (width === 1 ? 1 : 0), 0);
  const cellCount = widths.reduce((n, width) => n + (4 / width) ** 3, 0);
  let fineBase = 0, coarseBase = fineCount * 64;
  for (let key = 0; key < count; key++) {
    const isFine = widths[key] === 1;
    tiles[key] = ((isFine ? fineBase : coarseBase) | (isFine ? MIXED_FINE_TILE : widths[key] === 2 ? MIXED_TWO_TILE : 0)) >>> 0;
    (isFine ? fine : widths[key] === 2 ? transition : coarse).push(key);
    if (isFine) fineBase += 64; else coarseBase += (4 / widths[key]!) ** 3;
  }
  // Freeze geometric stencil decisions with ownership. Physics stages reuse
  // these masks for arbitrary sample locations, including long departures.
  const stencils = new Uint32Array(count * 2);
  for (let key = 0; key < count; key++) {
    const tx = key % dimensions[0], ty = Math.floor(key / dimensions[0]) % dimensions[1], tz = Math.floor(key / (dimensions[0] * dimensions[1]));
    let maximum = widths[key]!, minimum = widths[key]!;
    for (let z = -1; z <= 1; z++) for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) {
      const qx = tx + x, qy = ty + y, qz = tz + z;
      if (qx < 0 || qy < 0 || qz < 0 || qx >= dimensions[0] || qy >= dimensions[1] || qz >= dimensions[2]) continue;
      const width = widths[qx + dimensions[0] * (qy + dimensions[1] * qz)]!;
      maximum = Math.max(maximum, width);minimum = Math.min(minimum, width);
      if (width < 4) stencils[2 * key + (width === 1 ? 0 : 1)]! |= 1 << ((x + 1) + 3 * ((y + 1) + 3 * (z + 1)));
    }
    stencils[2 * key]! |= maximum << 27;
    stencils[2 * key + 1]! |= minimum << 27;
  }
  return { lattice, tileDimensions: dimensions, tiles, stencils, fineTiles: Uint32Array.from(fine),
    coarseTiles: Uint32Array.from(coarse), transitionTiles: Uint32Array.from(transition), cellCount, metadataBytes: count * 16, regions };
}
