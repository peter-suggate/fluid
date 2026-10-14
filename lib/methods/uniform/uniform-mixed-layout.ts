import { compileUniformStencil, compileUniformBlendMask } from "./uniform-compiled-topology";
import type { FluidRefinementRegion } from "../../core/model";
import { SOLID_MASK_EMPTY, SOLID_MASK_FULL, solidOccupancyMaskPage, type SolidOccupancyMask } from "../../core/solid-occupancy-mask";
import { refinementRegionCellBounds, type RefinementRegionLattice } from "../../core/refinement-regions";
import { uniformBoxTileCoord, uniformBoxTileIndex, uniformCellInBox, uniformEachTile, uniformLatticeAddress, uniformLatticeBox, uniformTileInBox, uniformTileKey, uniformTileStored, type UniformLatticeAddress } from "./uniform-lattice-address";

export const MIXED_FINE_TILE = 0x80000000;
export const MIXED_CELL_MASK = 0x3fffffff;
/** Owner width of a tile word: h (MIXED_FINE_TILE set) or 4h. */
export function mixedCellWidth(word: number): 1 | 4 {
  return word & MIXED_FINE_TILE ? 1 : 4;
}
type Triple = readonly [number, number, number];
/** Per-tile arrays here are indexed by tile key (uniformTileKey) and hold
 * zero at a key with no tile; the lists hold tile keys. Per-tile inputs (the
 * widths and solid masks below) are indexed by box index
 * (uniformBoxTileIndex). */
export interface UniformMixedLayout {
  readonly lattice: RefinementRegionLattice;
  /** The lattice's storage: the solver's one address, the same object in
   * every layout of a scene. */
  readonly address: UniformLatticeAddress;
  readonly tileDimensions: Triple;
  /** Upper bit: h (set) or 4h owner width. Remaining bits: first active cell. */
  readonly tiles: Uint32Array<ArrayBuffer>;
  /** The h tiles, then the 4h tiles, each in ascending owner order: by key,
   * and by box index in a layout with no h tile. */
  readonly fineTiles: Uint32Array<ArrayBuffer>;
  readonly coarseTiles: Uint32Array<ArrayBuffer>;
  /** Per tile: the 27-bit h neighbourhood mask with the maximum width in its
   * high bits, then minimum width, detail-ring and compiled vertex recipes. */
  readonly stencils: Uint32Array<ArrayBuffer>;
  /** Minimal fine-neighbor boxes for the exact velocity blending distance. */
  readonly blendMasks: Uint32Array<ArrayBuffer>;
  /** Owners: 64 an h tile and one a 4h tile; in the pressure root's shape the
   * tile keys, an owner index being a tile key there. */
  readonly cellCount: number;
  /** The pressure root's shape (uniformMixedAllCoarseLayout): its owners are
   * the tile keys, each key its own owner, written on the GPU every step
   * (UniformMixedRootOwners). The tile, list and stencil words here are zero
   * and name no tile; umCounts.y is this number, the rows before the wall
   * slots. */
  readonly boxOwners?: number;
  readonly metadataBytes: number;
  /** Effective bounds in tile coordinates, upper bound exclusive. */
  readonly regions: readonly { id: string; min: Triple; max: Triple }[];
}

/** Manual Uniform ownership: h/4h, ungraded. Tiles outside regions take the
 * background width (h unless a coarse background is requested).
 * address: the scene's (the solver's); a lattice on its own takes the box
 * with every page stored. */
export function createUniformMixedLayout(
  lattice: RefinementRegionLattice, regions: readonly FluidRefinementRegion[], backgroundWidth: 1 | 4 = 1,
  /** Tiles that must be h, from uniformMixedSolidTiles. Solids override
   * region cell-size bounds: forced tiles are h whatever a region says. */
  forcedFine?: Uint8Array, address: UniformLatticeAddress = uniformLatticeAddress(lattice.dimensions),
): UniformMixedLayout {
  const axes = ["x", "y", "z"] as const;
  for (let a = 0; a < 3; a++) {
    if (!Number.isSafeInteger(lattice.dimensions[a]) || lattice.dimensions[a]! < 4
      || lattice.dimensions[a]! % 4 !== 0 || !Number.isFinite(lattice.cellSize_m[a])
      || lattice.cellSize_m[a]! <= 0 || !Number.isFinite(lattice.origin_m[axes[a]!])) {
      throw new Error("Mixed layout requires finite positive spacing and dimensions divisible by four");
    }
  }
  if (address.cells.some((n, a) => n !== lattice.dimensions[a])) throw new Error("Mixed layout address does not match the lattice");
  const count = address.boxTiles;
  // Temporary two-bit allowed-scale mask by box index; discarded after topology construction.
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
      const key = uniformBoxTileIndex(address, x, y, z);
      if (forcedFine?.[key]) continue;
      // Overlapping Fine and Coarse boxes: fine wins, as it does for solids.
      // Coarse is a cost hint; refusing here threw mid-advance in the app.
      allowed[key] = allowed[key]! & mask || 1;
    }
    snapped.push({ id: r.id, min, max });
  }
  const widths = Uint8Array.from(allowed, mask => mask === 1 ? 1 : mask === 2 ? 4 : backgroundWidth);
  return packUniformMixedLayout(lattice, address, widths, snapped);
}

/** Ownership from per-tile widths decided elsewhere (the detail planner's
 * admitted h tiles plus solid promotion), by box index, with the snapped
 * authored boxes as metadata. Precedence is the caller's, not region list order. */
export function createUniformMixedLayoutFromWidths(lattice: RefinementRegionLattice, widths: Uint8Array,
  regions: UniformMixedLayout["regions"], address: UniformLatticeAddress = uniformLatticeAddress(lattice.dimensions)): UniformMixedLayout {
  if (lattice.dimensions.some(n => !Number.isSafeInteger(n) || n < 4 || n % 4 !== 0)) throw new Error("Mixed layout requires dimensions divisible by four");
  if (address.cells.some((n, a) => n !== lattice.dimensions[a]) || widths.length !== address.boxTiles) throw new Error("Mixed tile widths do not match the lattice");
  return packUniformMixedLayout(lattice, address, widths, regions);
}

/** Solid promotion. A cut cell has open fraction below one: a voxel of the
 * interior lattice or a terrain column cell below its height. Every tile with
 * a cell within one cell (26-neighbourhood) of a cut cell is solid-coupled;
 * those and their 26 neighbour tiles are forced to h. Fine-owner stencils of
 * one cell (faces, dual cells, phi contact, pressure continuation) therefore
 * never meet solid in a 4h owner or in an interface (seam) tile. A page
 * the lattice walls off cuts nothing; domain walls are owned by every width.
 * The masks returned hold a byte per tile by box index. */
export function uniformMixedSolidTiles(dimensions: Triple, mask: SolidOccupancyMask,
  terrainCells?: Float32Array): UniformMixedSolidTiles {
  if (dimensions.some(n => !Number.isSafeInteger(n) || n % 4 !== 0)) throw new Error("Solid promotion requires dimensions divisible by four");
  const address = uniformLatticeBox(dimensions), [nx, ny, nz] = dimensions, words = mask.words;
  const coupled = new Uint8Array(address.boxTiles);let cutCells = 0;
  const mark = (x: number, y: number, z: number) => {
    cutCells++;
    for (let tz = (z - 1) >> 2; tz <= (z + 1) >> 2; tz++) for (let ty = (y - 1) >> 2; ty <= (y + 1) >> 2; ty++)
      for (let tx = (x - 1) >> 2; tx <= (x + 1) >> 2; tx++) if (uniformTileInBox(address, tx, ty, tz)) coupled[uniformBoxTileIndex(address, tx, ty, tz)] = 1;
  };
  // The box's pages: a full page is every cell of it, a slot is a word a row along x (bit x of word y + 32 z).
  // The mask's directory is by window page, the box's from its origin (an open world's box sits mid-window).
  const o = mask.originCell;
  for (let pz = 0; pz < address.pages[2]; pz++) for (let py = 0; py < address.pages[1]; py++) for (let px = 0; px < address.pages[0]; px++) {
    const code = solidOccupancyMaskPage(words, 32 * px + o[0], 32 * py + o[1], 32 * pz + o[2]) & 0xffff;
    if (code === SOLID_MASK_EMPTY) continue;
    for (let row = 0; row < 1024; row++) {
      let bits = code >= SOLID_MASK_FULL ? 0xffffffff : words[(code << 10) + row]!;
      while (bits) {
        const bit = 31 - Math.clz32(bits & -bits);bits = (bits & (bits - 1)) >>> 0;
        const x = 32 * px + bit, y = 32 * py + (row & 31), z = 32 * pz + (row >> 5);
        if (uniformCellInBox(address, x, y, z)) mark(x, y, z);
      }
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
      for (let tz = (z - 1) >> 2; tz <= (z + 1) >> 2; tz++) for (let ty = 0; ty <= (top + 1) >> 2; ty++)
        for (let tx = (x - 1) >> 2; tx <= (x + 1) >> 2; tx++) if (uniformTileInBox(address, tx, ty, tz)) coupled[uniformBoxTileIndex(address, tx, ty, tz)] = 1;
    }
  }
  const forced = new Uint8Array(coupled.length), p: [number, number, number] = [0, 0, 0];let coupledTiles = 0;
  for (let key = 0; key < address.boxTiles; key++) {
    if (!coupled[key]) continue;
    coupledTiles++;
    uniformBoxTileCoord(address, key, p);
    const tx = p[0], ty = p[1], tz = p[2];
    for (let z = tz - 1; z <= tz + 1; z++) for (let y = ty - 1; y <= ty + 1; y++)
      for (let x = tx - 1; x <= tx + 1; x++) if (uniformTileInBox(address, x, y, z)) forced[uniformBoxTileIndex(address, x, y, z)] = 1;
  }
  return { forced, coupled, cutCells, coupledTiles };
}

/** Solid promotion of one mask: uniformMixedSolidTiles' result. */
export interface UniformMixedSolidTiles {
  forced: Uint8Array;
  coupled: Uint8Array;
  cutCells: number;
  /** Tiles set in `coupled`. */
  coupledTiles: number;
}

/** A solid mask edit as the tile lattice sees it (uniformMixedSolidEdit). */
export interface UniformMixedSolidEdit {
  /** Every tile within one cell (26-neighbourhood) of a voxel whose bit
   * changed, a voxel beside the box included, each once, by box index: the
   * tiles whose coupling the edit can have moved. */
  tiles: number[];
  /** Interior cells that became solid, less those that stopped being solid. */
  cutCells: number;
  /** Take a cell whose bit moved: what SolidOccupancyMask.update tells of an edit. */
  readonly cell: (x: number, y: number, z: number, solid: boolean) => void;
}

/** The tiles a mask edit touches, gathered from the cells the mask reports
 * as it folds the edit (cell). `touched` is the caller's scratch, one byte
 * per tile by box index and zero on entry; it holds one for each tile
 * gathered. */
export function uniformMixedSolidEdit(dimensions: Triple, touched: Uint8Array): UniformMixedSolidEdit {
  const address = uniformLatticeBox(dimensions);
  if (touched.length !== address.boxTiles) throw new Error("Solid edit scratch does not match the tile lattice");
  const tiles: number[] = [];
  const edit: UniformMixedSolidEdit = { tiles, cutCells: 0, cell(x, y, z, solid) {
    // The promotion counts interior cells.
    if (uniformCellInBox(address, x, y, z)) edit.cutCells += solid ? 1 : -1;
    for (let tz = (z - 1) >> 2; tz <= (z + 1) >> 2; tz++) for (let ty = (y - 1) >> 2; ty <= (y + 1) >> 2; ty++)
      for (let tx = (x - 1) >> 2; tx <= (x + 1) >> 2; tx++) {
        if (!uniformTileInBox(address, tx, ty, tz)) continue;
        const key = uniformBoxTileIndex(address, tx, ty, tz);
        if (!touched[key]) { touched[key] = 1;tiles.push(key); }
      }
  } };
  return edit;
}

/** uniformMixedSolidTiles of an edited mask, from the result `held` for the
 * mask before the edit and the tiles the edit touched: the same arrays the
 * lattice scan returns, at the cost of the edit. A tile's coupling follows
 * the cells within one cell of it, so only a touched tile's can have moved;
 * `forced` is the 26-neighbourhood of `coupled`, so only the tiles around a
 * tile whose coupling moved can have. `terrainCells` must be the heights
 * `held` was scanned with. New arrays: `held` is left as it was. */
export function uniformMixedSolidTilesAfterEdit(held: UniformMixedSolidTiles, dimensions: Triple, mask: SolidOccupancyMask,
  edit: UniformMixedSolidEdit, terrainCells?: Float32Array): UniformMixedSolidTiles {
  const address = uniformLatticeBox(dimensions), [nx, ny, nz] = dimensions, p: [number, number, number] = [0, 0, 0];
  const count = address.boxTiles;
  if (held.coupled.length !== count || held.forced.length !== count) throw new Error("Held solid promotion does not match the tile lattice");
  if (terrainCells && terrainCells.length !== nx * nz) throw new Error("Solid promotion terrain does not match the lattice");
  const coupled = held.coupled.slice(), forced = held.forced.slice();let coupledTiles = held.coupledTiles;
  const uncoupled: number[] = [];
  for (const key of edit.tiles) {
    uniformBoxTileCoord(address, key, p);
    const tx = p[0], ty = p[1], tz = p[2];
    // The cells within one cell of the tile, inside the lattice.
    const x0 = Math.max(0, 4 * tx - 1), x1 = Math.min(nx - 1, 4 * tx + 4), y0 = Math.max(0, 4 * ty - 1), y1 = Math.min(ny - 1, 4 * ty + 4);
    const z0 = Math.max(0, 4 * tz - 1), z1 = Math.min(nz - 1, 4 * tz + 4);
    let cut = 0;
    for (let z = z0; z <= z1 && !cut; z++) for (let y = y0; y <= y1 && !cut; y++)
      for (let x = x0; x <= x1 && !cut; x++) if (mask.solid(x, y, z)) cut = 1;
    if (terrainCells) for (let z = z0; z <= z1 && !cut; z++) for (let x = x0; x <= x1 && !cut; x++) {
      const height = terrainCells[x + nx * z]!;
      // Cells 0..top of the column are cut: it couples the tiles up to (top+1)>>2.
      if (height > 0 && ty <= (Math.min(ny - 1, Math.ceil(height) - 1) + 1) >> 2) cut = 1;
    }
    const now = cut ? 1 : 0;
    if (now === coupled[key]) continue;
    coupled[key] = now;coupledTiles += now ? 1 : -1;
    if (!now) { uncoupled.push(key);continue; }
    for (let z = tz - 1; z <= tz + 1; z++) for (let y = ty - 1; y <= ty + 1; y++)
      for (let x = tx - 1; x <= tx + 1; x++) if (uniformTileInBox(address, x, y, z)) forced[uniformBoxTileIndex(address, x, y, z)] = 1;
  }
  // A tile that stopped being coupled leaves its neighbours forced only by another coupled tile.
  for (const key of uncoupled) {
    uniformBoxTileCoord(address, key, p);
    const tx = p[0], ty = p[1], tz = p[2];
    for (let z = tz - 1; z <= tz + 1; z++) for (let y = ty - 1; y <= ty + 1; y++) for (let x = tx - 1; x <= tx + 1; x++) {
      if (!uniformTileInBox(address, x, y, z)) continue;
      let near = 0;
      for (let k = z - 1; k <= z + 1 && !near; k++) for (let j = y - 1; j <= y + 1 && !near; j++)
        for (let i = x - 1; i <= x + 1 && !near; i++) if (uniformTileInBox(address, i, j, k)) near = coupled[uniformBoxTileIndex(address, i, j, k)]!;
      forced[uniformBoxTileIndex(address, x, y, z)] = near;
    }
  }
  return { forced, coupled, cutCells: held.cutCells + edit.cutCells, coupledTiles };
}

/** Loud ownership certificate: no 4h owner may cover a promoted tile. */
export function assertUniformMixedSolidPromotion(layout: UniformMixedLayout, forced: Uint8Array): void {
  const address = layout.address;
  if (forced.length !== address.boxTiles) throw new Error("Solid promotion mask does not match the ownership layout");
  uniformEachTile(address, (key, x, y, z) => {
    if (forced[uniformBoxTileIndex(address, x, y, z)] && mixedCellWidth(layout.tiles[key]!) !== 1)
      throw new Error(`Mixed ownership places a ${mixedCellWidth(layout.tiles[key]!)}h owner on solid-coupled tile ${x},${y},${z}`);
  });
}

/** The all-4h shape on the same lattice: the pressure root's ownership. Only
 * what the page set cannot change is here (the lattice, its address, a row a
 * tile key): which key holds a tile, and each key's stencil, follow
 * the page table and are built on the GPU every step (UniformMixedRootOwners).
 * No host array names a tile. */
export function uniformMixedAllCoarseLayout(layout: UniformMixedLayout): UniformMixedLayout {
  const address = layout.address, count = address.tileKeys;
  return { lattice: layout.lattice, address, tileDimensions: address.tiles, tiles: new Uint32Array(count), stencils: new Uint32Array(2 * count),
    blendMasks: new Uint32Array(count), fineTiles: new Uint32Array(0), coarseTiles: new Uint32Array(0), cellCount: count,
    boxOwners: count, metadataBytes: count * 20, regions: layout.regions };
}

/** widths: a byte per tile by box index; a tile with nothing stored takes no
 * part. Owners are numbered in ascending tile key, the h tiles' first (the
 * GPU builder's order). A layout with no h tile numbers its owners by box
 * index instead: it is the pressure root's ownership, whose rows, wall slots
 * and solid record are one box, and a box index no stored tile has is an
 * owner no tile names. */
function packUniformMixedLayout(lattice: RefinementRegionLattice, address: UniformLatticeAddress, widths: Uint8Array,
  regions: UniformMixedLayout["regions"]): UniformMixedLayout {
  const count = address.tileKeys;
  // A tile word holds its first owner's index in 30 bits and any tile may be h (64 owners): 32,767 pages of 512 tiles.
  if (count * 64 > MIXED_CELL_MASK) throw new Error(`Mixed tile words index owners in 30 bits: ${count} tile keys exceed ${Math.floor(MIXED_CELL_MASK / 64)} (32,767 pages of 512)`);
  const tiles = new Uint32Array(count);
  const fine: number[] = [], coarse: number[] = [];
  let fineCount = 0;
  uniformEachTile(address, (_key, x, y, z) => { if (widths[uniformBoxTileIndex(address, x, y, z)] === 1) fineCount++; });
  const cellCount = fineCount ? 63 * fineCount + address.liveTiles : address.boxTiles;
  let fineBase = 0, coarseBase = fineCount * 64;
  uniformEachTile(address, (key, x, y, z) => {
    const index = uniformBoxTileIndex(address, x, y, z), isFine = widths[index] === 1;
    if (!isFine && widths[index] !== 4) throw new Error("Mixed ownership is h/4h");
    if (!fineCount) { tiles[key] = index;coarse[index] = key;return; }
    tiles[key] = ((isFine ? fineBase : coarseBase) | (isFine ? MIXED_FINE_TILE : 0)) >>> 0;
    (isFine ? fine : coarse).push(key);
    if (isFine) fineBase += 64; else coarseBase += 1;
  });
  const stencils = mixedStencils(address, widths);
  const blendMasks=Uint32Array.from({length:count},(_,i)=>compileUniformBlendMask(stencils[2*i]!));
  // With no h tile, coarse is indexed by box index: filter drops the indices no stored tile has.
  return { lattice, address, tileDimensions: address.tiles, tiles, stencils, blendMasks, fineTiles: Uint32Array.from(fine),
    coarseTiles: Uint32Array.from(fineCount ? coarse : coarse.filter(() => true)), cellCount, metadataBytes: count * 20, regions };
}

/** Second stencil word, bit 0: the tile is in the detail ring, an h tile
 * within three tiles of it (itself included). The h store of a detail field
 * holds the canonical texels of ring tiles only (UniformDetailDomain); a 4h
 * tile outside the ring lives in its base block alone. */
export const UNIFORM_MIXED_DETAIL_RING = 1;
/** Freeze geometric stencil decisions with ownership. Physics stages reuse
 * these masks for arbitrary sample locations, including long departures. */
function mixedStencils(address: UniformLatticeAddress, widths: Uint8Array): Uint32Array<ArrayBuffer> {
  const count = address.tileKeys, stencils = new Uint32Array(count * 2);
  uniformEachTile(address, (key, tx, ty, tz) => {
    const own = widths[uniformBoxTileIndex(address, tx, ty, tz)]!;
    let maximum = own, minimum = own, valid = 0;
    for (let z = -1; z <= 1; z++) for (let y = -1; y <= 1; y++) for (let x = -1; x <= 1; x++) {
      const qx = tx + x, qy = ty + y, qz = tz + z;
      if (!uniformTileStored(address, qx, qy, qz)) continue;
      valid |= 1 << ((x + 1) + 3 * ((y + 1) + 3 * (z + 1)));
      const width = widths[uniformBoxTileIndex(address, qx, qy, qz)]!;
      maximum = Math.max(maximum, width);minimum = Math.min(minimum, width);
      if (width === 1) stencils[2 * key]! |= 1 << ((x + 1) + 3 * ((y + 1) + 3 * (z + 1)));
    }
    stencils[2 * key]! |= maximum << 27;
    let farPositive = 0;
    for (let axis = 0; axis < 3; axis++) {
      const qx = tx + (axis === 0 ? 2 : 0), qy = ty + (axis === 1 ? 2 : 0), qz = tz + (axis === 2 ? 2 : 0);
      if (uniformTileStored(address, qx, qy, qz) && widths[uniformBoxTileIndex(address, qx, qy, qz)] === 1) farPositive |= 1 << axis;
    }
    stencils[2 * key + 1]! = (minimum << 27) | (farPositive << 24) | compileUniformStencil(stencils[2 * key]!, valid);
  });
  // The detail ring (UNIFORM_MIXED_DETAIL_RING): bit 0 of the second word,
  // an h tile within three tiles. A tile two tiles out sees the 27-bit h
  // masks of its radius-three neighbourhood (the GPU builder's `mirror`).
  uniformEachTile(address, (key, tx, ty, tz) => {
    let near = false;
    for (let z = -2; z <= 2 && !near; z += 2) for (let y = -2; y <= 2 && !near; y += 2) for (let x = -2; x <= 2 && !near; x += 2) {
      const qx = tx + x, qy = ty + y, qz = tz + z;
      if (!uniformTileStored(address, qx, qy, qz)) continue;
      near = (stencils[2 * uniformTileKey(address, qx, qy, qz)]! & 0x7ffffff) !== 0;
    }
    if (near) stencils[2 * key + 1]! |= UNIFORM_MIXED_DETAIL_RING;
  });
  return stencils;
}
