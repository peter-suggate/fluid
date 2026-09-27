/**
 * The backdrop ground as voxel terrain on a 2D clipmap of tiles, which the SVO
 * renderer's traversal consults beside the octree.
 *
 * ## What is drawn
 *
 * The set's own terrain, continued: a height field of cubic voxel columns on
 * the set's render lattice, with the set's face presentation (six-axis normals,
 * cell seams). The analytic field (`backdrop-field.ts`) is the source; what a
 * ray meets is its voxelisation, one column per cell, the column's top the
 * field at the column centre rounded to the lattice. Near the set the cell is
 * the set's own, so the seam is voxel against voxel; each tile level outward
 * doubles the cell, so the voxel size grows with distance and the terrain
 * still reaches the haze horizon at a bounded cost.
 *
 * ## Why not octree leaves
 *
 * The ground is a height field. Slices 2-4 of `docs/backdrop-svo-plan.md` put
 * it in the octree as virtual leaves, and every leaf was reached by the generic
 * cursor descending from the root through a domain widened to the whole
 * backdrop: the +6.5 ms (hero) / +15 ms (low orbit). A height field needs no
 * third axis: a 2D walk over tiles that each carry their columns' height
 * interval rejects a tile the ray passes above with one compare, and a column
 * DDA inside the band finds the voxel.
 *
 * ## The clipmap
 *
 * Centred on the footprint's centre snapped to the lattice (c0). Level 0 is a
 * `BACKDROP_TILES_PER_AXIS` square grid over `c0 +- R0`; level `l` is the same
 * grid over `c0 +- R0 2^l`, whose central half belongs to level `l - 1`. R0 is
 * a multiple of 8 cells, so every tile holds whole columns of its level
 * (`columnsPerTile` per axis, the same at every level), level squares fall on
 * the coarser level's column grid, and the grids nest. Levels are added until
 * the ring reaches the haze horizon (`backdropTerrainReach`).
 *
 * Per tile: the exact `[lowest, highest]` column top over its columns outside
 * the footprint (evaluated on the CPU, in f64), and whether it overlaps the
 * footprint. The shader clamps each column top it computes to its tile's
 * interval, so f32 rounding can never put a column outside the bounds the walk
 * trusts. Per level: the highest top over its whole square, finer levels
 * included, so a ray above it skips the level in one step. The bounds depend
 * only on the field and the lattice: the table is built once per scene and a
 * camera orbit rebuilds nothing.
 *
 * ## Where the table lives
 *
 * Past the dense `sceneMaterialOwners` lane, at owner-lane index
 * `voxelCapacity`, which every reader already holds: no new binding and no new
 * parameter word. A world without a backdrop still allocates the header, with
 * zero levels, so the trace reads "no terrain" and returns at once.
 *
 * No DOM or GPU imports: the table and the strings are built in the render worker.
 */
import {
  BACKDROP_WAVE_STRIDE,
  backdropFieldHeight,
  type BackdropField,
} from "./backdrop-field";

/** Tiles per axis per level; the central half of level l > 0 is level l - 1. */
export const BACKDROP_TILES_PER_AXIS = 16;
/** Tiles per level, including the central hole of the rings. */
export const BACKDROP_TILES_PER_LEVEL = BACKDROP_TILES_PER_AXIS * BACKDROP_TILES_PER_AXIS;
/** The most levels the header has room for: 16 is 65,536 times R0. */
export const BACKDROP_TILE_MAXIMUM_LEVELS = 16;
/** Level 0's half-width, in the footprint's larger half-extent (before snapping up to whole tiles of cells). */
export const BACKDROP_TILE_LEVEL_ZERO_FOOTPRINTS = 2;
/** Haze transmittance at the edge of the terrain: past it the ground is the horizon colour to within an 8-bit step. */
export const BACKDROP_HAZE_HORIZON_TRANSMITTANCE = 1 / 256;
/** Tile visits plus column visits one ray may spend. */
export const BACKDROP_TERRAIN_TRACE_BUDGET = 384;

/** Footprint distance out to which the ground is traced: where the haze leaves 1/256 of it. */
export function backdropTerrainReach(field: BackdropField): number {
  return field.description.hazeDistance_m * Math.log(1 / BACKDROP_HAZE_HORIZON_TRANSMITTANCE);
}

/** Word layout of the table. Floats are stored as their f32 bits. */
export const BACKDROP_TERRAIN_TABLE = Object.freeze({
  /** 0 wave count, 1 material id, 2 tile level count, 3 highest column top (f32 bits). */
  countsWord: 0,
  /** footprint minX, minZ, maxX, maxZ. */
  footprintWord: 4,
  /** seam, flat ring, ramp width, valley span. */
  envelopeWord: 8,
  /** valley rise, amplitude, level-0 half-width R0, level-0 cell size h0. */
  shapeWord: 12,
  /** vertical lattice anchor (column tops sit on anchor + k h); stored detail ring count (u32: the leading levels the walk leaves empty); clipmap centre c0 x, z. */
  latticeWord: 16,
  wavesWord: 20,
  /** Content ceiling (see `BackdropContentCeiling`): section word index (0 = none), cells per axis, reserved. */
  ceilingWord: 125,
  /** One word per level: the highest column top over the level's whole square. */
  levelMaximumWord: 128,
  /** Tile records, four words each: lowest top, highest top, overlaps the footprint (1/0), reserved. Empty tiles have lowest > highest. */
  tilesWord: 128 + BACKDROP_TILE_MAXIMUM_LEVELS,
  tileWords: 4,
});

/** The most waves the header has room for. */
export const BACKDROP_TERRAIN_MAXIMUM_WAVES = Math.floor(
  (BACKDROP_TERRAIN_TABLE.ceilingWord - BACKDROP_TERRAIN_TABLE.wavesWord) / BACKDROP_WAVE_STRIDE);

/** Words of the header alone: the table on a world with no backdrop. */
export const BACKDROP_TERRAIN_HEADER_WORDS = BACKDROP_TERRAIN_TABLE.tilesWord;

/** The set's render lattice, which the backdrop's level-0 voxels continue. */
export interface BackdropTerrainLattice {
  /** A lattice corner (any one: only its phase matters). */
  origin_m: readonly [number, number, number];
  /** The set's cubic cell edge. */
  cellSize_m: number;
  /**
   * The first level the walk draws. Finer levels are stored octree voxels
   * (`backdrop-detail.ts`): their tiles are empty, so the walk skips their
   * squares in one step.
   */
  firstLevel?: number;
  /**
   * Snap the centre to `origin + k * centreLattice_m` rather than to the cell,
   * so the stored rings' boundaries fall on octree node boundaries.
   */
  centreLattice_m?: number;
}

export interface BackdropTilePlan {
  /** Clipmap centre c0 (x, z): the footprint's centre snapped to the lattice. Column grids are anchored here. */
  centre: readonly [number, number];
  /** Level 0's half-width R0, a multiple of 8 cells. */
  halfWidth0_m: number;
  /** Level 0's cell h0; level l's is h0 2^l. */
  cellSize0_m: number;
  /** Column tops are `anchor + k h`: the seam snapped to the lattice. */
  anchor_m: number;
  /** Columns per tile axis, at every level. */
  columnsPerTile: number;
  levels: number;
  /** `levels * BACKDROP_TILES_PER_LEVEL` records of `[low, high, overlapsFootprint]`; empty tiles are `[+inf, -inf, 0]`. */
  tiles: Float64Array;
  /** Highest column top over each level's whole square (finer levels included). */
  levelMaximum: Float64Array;
  /** Highest column top anywhere the terrain is traced. */
  highest_m: number;
}

/** Tile edge at level `l`. */
export function backdropTileEdge(plan: Pick<BackdropTilePlan, "halfWidth0_m">, level: number): number {
  return plan.halfWidth0_m * 2 ** level * 2 / BACKDROP_TILES_PER_AXIS;
}

/** Whether a column centre lies inside the footprint, where the set's ground is authoritative. */
function insideFootprint(field: BackdropField, x: number, z: number): boolean {
  const [minX, minZ, maxX, maxZ] = field.seam.footprint_m;
  return x > minX && x < maxX && z > minZ && z < maxZ;
}

/** A column's top before its tile's clamp: the field at the centre, rounded to the level's lattice. */
function columnTop(field: BackdropField, anchor: number, cell: number, x: number, z: number): number {
  return anchor + Math.floor((backdropFieldHeight(field, x, z) - anchor) / cell + 0.5) * cell;
}

/**
 * The clipmap for a compiled field on the set's lattice: layout, exact per-tile
 * column-top bounds and per-level maxima. Pure and deterministic.
 */
export function planBackdropTiles(field: BackdropField, lattice: BackdropTerrainLattice): BackdropTilePlan {
  const h0 = lattice.cellSize_m;
  if (!(h0 > 0) || !Number.isFinite(h0) || !lattice.origin_m.every(Number.isFinite)) {
    throw new RangeError("Backdrop terrain lattice must have a finite origin and a positive cell");
  }
  const [fMinX, fMinZ, fMaxX, fMaxZ] = field.seam.footprint_m;
  const snap = (value: number, origin: number, step = h0) => origin + Math.round((value - origin) / step) * step;
  const centreStep = lattice.centreLattice_m ?? h0;
  const firstLevel = lattice.firstLevel ?? 0;
  if (!(centreStep > 0) || !Number.isInteger(firstLevel) || firstLevel < 0) {
    throw new RangeError("Backdrop tile centre lattice must be positive and the first level a non-negative integer");
  }
  const centre = [snap(0.5 * (fMinX + fMaxX), lattice.origin_m[0], centreStep), snap(0.5 * (fMinZ + fMaxZ), lattice.origin_m[2], centreStep)] as const;
  const anchor_m = snap(field.seam.height_m, lattice.origin_m[1]);
  // Whole tiles of whole cells, covering the footprint from the snapped centre.
  const quarter = BACKDROP_TILES_PER_AXIS / 2;
  const cover = Math.max(fMaxX - centre[0], centre[0] - fMinX, fMaxZ - centre[1], centre[1] - fMinZ);
  const columnsPerTileExact = Math.max(1, Math.ceil(Math.max(
    BACKDROP_TILE_LEVEL_ZERO_FOOTPRINTS * 0.5 * Math.max(fMaxX - fMinX, fMaxZ - fMinZ), cover) / (quarter * h0) - 1e-9));
  // Even, so every ring boundary is a whole number of the next ring's bricks.
  const columnsPerTile = firstLevel > 0 ? columnsPerTileExact + (columnsPerTileExact % 2) : columnsPerTileExact;
  const halfWidth0_m = columnsPerTile * quarter * h0;
  const needed = cover + backdropTerrainReach(field);
  let levels = 1;
  while (halfWidth0_m * 2 ** (levels - 1) < needed || levels <= firstLevel) levels += 1;
  if (levels > BACKDROP_TILE_MAXIMUM_LEVELS) {
    throw new RangeError(`Backdrop terrain needs ${levels} tile levels to reach ${needed.toFixed(0)} m; the table holds ${BACKDROP_TILE_MAXIMUM_LEVELS}`);
  }
  const N = BACKDROP_TILES_PER_AXIS, n = columnsPerTile;
  const tiles = new Float64Array(levels * BACKDROP_TILES_PER_LEVEL * 3);
  const levelMaximum = new Float64Array(levels);
  let running = Number.NEGATIVE_INFINITY;
  for (let level = 0; level < levels; level += 1) {
    const cell = h0 * 2 ** level;
    for (let j = 0; j < N; j += 1) for (let i = 0; i < N; i += 1) {
      const base = ((level * N + j) * N + i) * 3;
      tiles[base] = Number.POSITIVE_INFINITY; tiles[base + 1] = Number.NEGATIVE_INFINITY; tiles[base + 2] = 0;
      // The central half of a ring is the finer level's.
      if (level > 0 && i >= N / 4 && i < N - N / 4 && j >= N / 4 && j < N - N / 4) continue;
      // Stored as octree voxels, not walked.
      if (level < firstLevel) continue;
      let low = Number.POSITIVE_INFINITY, high = Number.NEGATIVE_INFINITY, overlaps = 0;
      for (let b = 0; b < n; b += 1) for (let a = 0; a < n; a += 1) {
        const x = centre[0] + ((i - quarter) * n + a + 0.5) * cell;
        const z = centre[1] + ((j - quarter) * n + b + 0.5) * cell;
        if (insideFootprint(field, x, z)) { overlaps = 1; continue; }
        const top = columnTop(field, anchor_m, cell, x, z);
        low = Math.min(low, top); high = Math.max(high, top);
      }
      tiles[base + 2] = overlaps;
      if (!(low <= high)) continue;
      if (!Number.isFinite(low) || !Number.isFinite(high)) throw new RangeError(`Backdrop tile ${level}/${i},${j} has invalid bounds`);
      tiles[base] = low; tiles[base + 1] = high;
      running = Math.max(running, high);
    }
    levelMaximum[level] = running;
  }
  if (!Number.isFinite(running)) throw new RangeError("Backdrop terrain has no columns outside the footprint");
  return { centre, halfWidth0_m, cellSize0_m: h0, anchor_m, columnsPerTile, levels, tiles, levelMaximum, highest_m: running };
}

/** Content-ceiling columns per axis over the world's xz square. */
export const BACKDROP_CONTENT_CEILING_CELLS = 128;
/** Columns per axis of one ceiling tile: the walk's coarse step (see `BackdropContentCeiling.tiles`). */
export const BACKDROP_CONTENT_CEILING_TILE_CELLS = 8;
/** Ceiling tiles per axis. */
export const BACKDROP_CONTENT_CEILING_TILES = BACKDROP_CONTENT_CEILING_CELLS / BACKDROP_CONTENT_CEILING_TILE_CELLS;
/** Section words before the columns: corner x, corner z, column edge, highest ceiling (f32 bits). The tile maxima follow the columns. */
export const BACKDROP_CONTENT_CEILING_HEADER_WORDS = 4;

/**
 * The top of the world's voxel content, per xz column: what lets a sun cone
 * stop where its ray has risen clear of everything it could still meet.
 *
 * A directional cone marches to the world-box exit, and the box is as tall as
 * the set's tallest content (1.85 m at x10), while most of the world is ground
 * and low scatter. With a fixed step budget the footprint then grows to reach
 * that exit (slice 7's reach floor), spending half the steps in empty sky and
 * blurring every shadow. Each column holds the highest leaf-box top over
 * itself and its eight neighbours (a one-column dilation, so a cone passing
 * beside tall content still sees it), from the octree plan's leaves: every
 * node-mip page lies under a leaf, so no sample above a column's value can
 * read anything. Edits only raise columns (`raiseBackdropContentCeiling`);
 * a removed body leaves its old top, which is conservative. It rides after the
 * terrain table, whose header every dense world allocates, so a world with no
 * backdrop carries it too (`packContentCeilingTable`): the set around a small
 * container — a floor and one lamp on a stem running up out of shot — makes
 * the world box several metres tall over a floor that is almost all empty sky,
 * and every sun cone would otherwise march to that box's lid.
 *
 * The walk is two-level. One tall column (that lamp's stem) holds the highest
 * top, and a ray only clears the ceiling for good once it clears that, so a
 * column-by-column walk crosses the whole world for every floor pixel. Each
 * tile of 8x8 columns therefore also holds its highest column, and the walk
 * steps tile by tile, reading columns only inside a tile the ray may still be
 * under.
 */
export interface BackdropContentCeiling {
  readonly cells: number;
  readonly corner_m: readonly [number, number];
  readonly cell_m: number;
  /** Highest top per column (dilated), -3e38 where nothing stands. */
  readonly tops: Float32Array<ArrayBuffer>;
  /** Highest column per tile of `BACKDROP_CONTENT_CEILING_TILE_CELLS`^2 columns. */
  readonly tiles: Float32Array<ArrayBuffer>;
  highest_m: number;
}

/** An empty ceiling over the world's xz rectangle. */
export function createBackdropContentCeiling(corner_m: readonly [number, number], extent_m: readonly [number, number]): BackdropContentCeiling {
  const cells = BACKDROP_CONTENT_CEILING_CELLS;
  const cell_m = Math.max(extent_m[0], extent_m[1]) / cells;
  if (!(cell_m > 0) || !Number.isFinite(cell_m)) throw new RangeError("Content ceiling needs a positive world extent");
  const tiles = BACKDROP_CONTENT_CEILING_TILES;
  return { cells, corner_m: [corner_m[0], corner_m[1]], cell_m, tops: new Float32Array(cells * cells).fill(-3e38),
    tiles: new Float32Array(tiles * tiles).fill(-3e38), highest_m: -3e38 };
}

/**
 * Raise the columns a box stands over (plus one column around it) to its top.
 * Returns the touched row range `[first, last]` (z rows), or undefined.
 */
export function raiseBackdropContentCeiling(ceiling: BackdropContentCeiling,
  minimum: readonly number[], maximum: readonly number[]): readonly [number, number] | undefined {
  const top = maximum[1]!;
  if (!Number.isFinite(top)) throw new RangeError("Content ceiling box top is not finite");
  const n = ceiling.cells;
  const column = (value: number, axis: 0 | 1) => Math.floor((value - ceiling.corner_m[axis]) / ceiling.cell_m);
  const x0 = Math.max(column(minimum[0]!, 0) - 1, 0), x1 = Math.min(column(maximum[0]!, 0) + 1, n - 1);
  const z0 = Math.max(column(minimum[2]!, 1) - 1, 0), z1 = Math.min(column(maximum[2]!, 1) + 1, n - 1);
  if (x0 > x1 || z0 > z1) return undefined;
  // Rounded up an f32 step so the stored value can only loosen the bound.
  const raised = Math.fround(top + (1e-6 + 2 ** -22 * Math.abs(top)));
  for (let z = z0; z <= z1; z += 1) for (let x = x0; x <= x1; x += 1) {
    const index = z * n + x;
    if (ceiling.tops[index]! < raised) ceiling.tops[index] = raised;
    const tile = Math.floor(z / BACKDROP_CONTENT_CEILING_TILE_CELLS) * BACKDROP_CONTENT_CEILING_TILES
      + Math.floor(x / BACKDROP_CONTENT_CEILING_TILE_CELLS);
    if (ceiling.tiles[tile]! < raised) ceiling.tiles[tile] = raised;
  }
  ceiling.highest_m = Math.max(ceiling.highest_m, raised);
  return [z0, z1];
}

/** The ceiling section's words (header, then the columns row by row, then the tiles row by row). */
export function packBackdropContentCeiling(ceiling: BackdropContentCeiling): Uint32Array<ArrayBuffer> {
  const words = new Uint32Array(BACKDROP_CONTENT_CEILING_HEADER_WORDS + ceiling.tops.length + ceiling.tiles.length);
  const floats = new Float32Array(words.buffer);
  floats.set([ceiling.corner_m[0], ceiling.corner_m[1], ceiling.cell_m, ceiling.highest_m], 0);
  floats.set(ceiling.tops, BACKDROP_CONTENT_CEILING_HEADER_WORDS);
  floats.set(ceiling.tiles, BACKDROP_CONTENT_CEILING_HEADER_WORDS + ceiling.tops.length);
  return words;
}

/** The whole table, ready to write at owner-lane index `voxelCapacity`. */
export function packBackdropTerrainTable(field: BackdropField, materialId: number, plan: BackdropTilePlan,
  ceiling?: BackdropContentCeiling): Uint32Array<ArrayBuffer> {
  if (!Number.isSafeInteger(materialId) || materialId <= 0 || materialId > 0xffff) {
    throw new RangeError("Backdrop material id must be a non-zero 16-bit material");
  }
  if (field.waveCount > BACKDROP_TERRAIN_MAXIMUM_WAVES) {
    throw new RangeError(`Backdrop has ${field.waveCount} waves; the terrain table holds ${BACKDROP_TERRAIN_MAXIMUM_WAVES}`);
  }
  const T = BACKDROP_TERRAIN_TABLE;
  const terrainWords = T.tilesWord + plan.levels * BACKDROP_TILES_PER_LEVEL * T.tileWords;
  const ceilingWords = ceiling ? packBackdropContentCeiling(ceiling) : undefined;
  const words = new Uint32Array(terrainWords + (ceilingWords?.length ?? 0));
  const floats = new Float32Array(words.buffer);
  const { hills, outerRadius_m } = field.description;
  // Widened by an f32 step so rounding can only loosen a bound.
  const down = (value: number) => value - (1e-6 + 2 ** -22 * Math.abs(value));
  const up = (value: number) => value + (1e-6 + 2 ** -22 * Math.abs(value));
  words.set([field.waveCount, materialId, plan.levels], T.countsWord);
  floats[T.countsWord + 3] = up(plan.highest_m);
  floats.set(field.seam.footprint_m, T.footprintWord);
  floats.set([field.seam.height_m, hills.flatRing_m, hills.rampWidth_m, outerRadius_m - hills.flatRing_m], T.envelopeWord);
  floats.set([hills.valleyRise_m, hills.amplitude_m, plan.halfWidth0_m, plan.cellSize0_m], T.shapeWord);
  floats.set([plan.anchor_m, 0, plan.centre[0], plan.centre[1]], T.latticeWord);
  let storedRings = 0;
  while (storedRings < plan.levels && !Number.isFinite(plan.levelMaximum[storedRings]!)) storedRings += 1;
  words[T.latticeWord + 1] = storedRings;
  floats.set(field.waves, T.wavesWord);
  // A level with no walked columns (every stored level) lies under nothing.
  for (let level = 0; level < plan.levels; level += 1) {
    const maximum = plan.levelMaximum[level]!;
    floats[T.levelMaximumWord + level] = Number.isFinite(maximum) ? up(maximum) : -3e38;
  }
  for (let tile = 0; tile < plan.levels * BACKDROP_TILES_PER_LEVEL; tile += 1) {
    const low = plan.tiles[tile * 3]!, high = plan.tiles[tile * 3 + 1]!, overlaps = plan.tiles[tile * 3 + 2]!;
    const base = T.tilesWord + tile * T.tileWords;
    if (low > high) { floats.set([3e38, -3e38, overlaps, 0], base); continue; }
    floats.set([down(low), up(high), overlaps, 0], base);
  }
  if (ceilingWords) {
    words.set(ceilingWords, terrainWords);
    words.set([terrainWords, ceiling!.cells, 0], T.ceilingWord);
  }
  if (!floats.every(Number.isFinite)) throw new RangeError("Backdrop terrain table is not finite");
  return words;
}

/** A table with no terrain (zero levels) carrying only the content ceiling: a dense world without a backdrop. */
export function packContentCeilingTable(ceiling: BackdropContentCeiling): Uint32Array<ArrayBuffer> {
  const T = BACKDROP_TERRAIN_TABLE;
  const ceilingWords = packBackdropContentCeiling(ceiling);
  const words = new Uint32Array(BACKDROP_TERRAIN_HEADER_WORDS + ceilingWords.length);
  words.set(ceilingWords, BACKDROP_TERRAIN_HEADER_WORDS);
  words.set([BACKDROP_TERRAIN_HEADER_WORDS, ceiling.cells, 0], T.ceilingWord);
  if (!new Float32Array(ceilingWords.buffer).every(Number.isFinite)) throw new RangeError("Content ceiling is not finite");
  return words;
}

/** An empty table (zero levels): what a world without a backdrop allocates. */
export function emptyBackdropTerrainTable(): Uint32Array<ArrayBuffer> {
  return new Uint32Array(BACKDROP_TERRAIN_HEADER_WORDS);
}

// ---------------------------------------------------------------------------
// CPU reference of the WGSL walk, for brute-force checks and visit counts.
// ---------------------------------------------------------------------------

/** Which face of a voxel column a ray entered: its x or z side, or its top. */
export type BackdropTerrainFace = "x" | "y" | "z";

export interface BackdropTerrainTraceResult {
  /** First voxel hit, or -1. */
  t: number;
  face: BackdropTerrainFace;
  tiles: number;
  /** Columns visited. */
  steps: number;
  exhausted: boolean;
}

/** The finest level whose square holds (x, z), as the walk looks it up. */
export function backdropTerrainLevelAt(plan: BackdropTilePlan, x: number, z: number): number {
  const m = Math.max(Math.abs(x - plan.centre[0]), Math.abs(z - plan.centre[1]));
  return m <= plan.halfWidth0_m ? 0 : Math.min(plan.levels - 1, Math.ceil(Math.log2(m / plan.halfWidth0_m)));
}

/**
 * The top of the voxel column holding (x, z) at `level`, clamped to its tile's
 * interval exactly as the shader clamps it, or NaN where there is no column
 * (inside the footprint, in an empty tile, or outside the outermost square).
 */
export function backdropTerrainColumnTop(field: BackdropField, plan: BackdropTilePlan, level: number, x: number, z: number): number {
  const N = BACKDROP_TILES_PER_AXIS, cell = plan.cellSize0_m * 2 ** level;
  const ix = Math.floor((x - plan.centre[0]) / cell), iz = Math.floor((z - plan.centre[1]) / cell);
  const cx = plan.centre[0] + (ix + 0.5) * cell, cz = plan.centre[1] + (iz + 0.5) * cell;
  if (insideFootprint(field, cx, cz)) return Number.NaN;
  const i = Math.floor(ix / plan.columnsPerTile) + N / 2, j = Math.floor(iz / plan.columnsPerTile) + N / 2;
  if (i < 0 || j < 0 || i >= N || j >= N) return Number.NaN;
  const base = ((level * N + j) * N + i) * 3;
  const low = plan.tiles[base]!, high = plan.tiles[base + 1]!;
  if (!(low <= high)) return Number.NaN;
  return Math.min(high, Math.max(low, columnTop(field, plan.anchor_m, cell, cx, cz)));
}

/**
 * The walk `backdropTerrainTrace` performs, in f64: the same level lookup,
 * level skip, band clip and column DDA, with the same budget, so its visit
 * counts are the shader's.
 */
export function traceBackdropTerrain(
  field: BackdropField,
  plan: BackdropTilePlan,
  ro: readonly [number, number, number],
  rd: readonly [number, number, number],
  tMin: number,
  tMax: number,
): BackdropTerrainTraceResult {
  const result: BackdropTerrainTraceResult = { t: -1, face: "y", tiles: 0, steps: 0, exhausted: false };
  const N = BACKDROP_TILES_PER_AXIS, n = plan.columnsPerTile;
  const [cx, cz] = plan.centre;
  const squareInterval = (half: number): readonly [number, number] => {
    let enter = Number.NEGATIVE_INFINITY, exit = Number.POSITIVE_INFINITY;
    for (const [o, d, c] of [[ro[0], rd[0], cx], [ro[2], rd[2], cz]] as const) {
      if (Math.abs(d) < 1e-12) { if (o < c - half || o > c + half) return [1, 0]; continue; }
      const a = (c - half - o) / d, b = (c + half - o) / d;
      enter = Math.max(enter, Math.min(a, b)); exit = Math.min(exit, Math.max(a, b));
    }
    return [enter, exit];
  };
  const outer = squareInterval(plan.halfWidth0_m * 2 ** (plan.levels - 1));
  let t = Math.max(tMin, 0, outer[0]);
  const tEnd = Math.min(tMax, outer[1]);
  let work = 0;
  const columns = (a: number, b: number, level: number, ti: number, tj: number, low: number, high: number): number => {
    const cell = plan.cellSize0_m * 2 ** level;
    const first = [(ti - N / 2) * n, (tj - N / 2) * n] as const;
    const probe = a + 1e-5 * (1 + a);
    const index = [
      Math.min(first[0] + n - 1, Math.max(first[0], Math.floor((ro[0] + rd[0] * probe - cx) / cell))),
      Math.min(first[1] + n - 1, Math.max(first[1], Math.floor((ro[2] + rd[2] * probe - cz) / cell))),
    ];
    const dirs = [rd[0], rd[2]] as const, origins = [ro[0], ro[2]] as const, centres = [cx, cz] as const;
    const enter = [0, 0], exit = [0, 0];
    for (let axis = 0; axis < 2; axis += 1) {
      const lo = centres[axis]! + index[axis]! * cell, d = dirs[axis]!;
      if (d > 0) { enter[axis] = (lo - origins[axis]!) / d; exit[axis] = (lo + cell - origins[axis]!) / d; }
      else if (d < 0) { enter[axis] = (lo + cell - origins[axis]!) / d; exit[axis] = (lo - origins[axis]!) / d; }
      else { enter[axis] = Number.NEGATIVE_INFINITY; exit[axis] = Number.POSITIVE_INFINITY; }
    }
    let s = a;
    let face: BackdropTerrainFace = s <= Math.max(enter[0]!, enter[1]!) + 1e-5 * (1 + s) ? (enter[0]! >= enter[1]! ? "x" : "z") : "y";
    while (work < BACKDROP_TERRAIN_TRACE_BUDGET) {
      work += 1; result.steps += 1;
      const columnExit = Math.min(exit[0]!, exit[1]!);
      const x = cx + (index[0]! + 0.5) * cell, z = cz + (index[1]! + 0.5) * cell;
      if (!insideFootprint(field, x, z)) {
        const top = Math.min(high, Math.max(low, columnTop(field, plan.anchor_m, cell, x, z)));
        if (ro[1] + rd[1] * s <= top) { result.face = face; return s; }
        if (rd[1] < 0) {
          const tTop = (top - ro[1]) / rd[1];
          if (tTop <= columnExit) { result.face = "y"; return tTop; }
        }
      }
      if (columnExit >= b) return -1;
      s = columnExit;
      const axis = exit[0]! <= exit[1]! ? 0 : 1;
      index[axis]! += dirs[axis]! > 0 ? 1 : -1;
      exit[axis]! += cell / Math.abs(dirs[axis]!);
      face = axis === 0 ? "x" : "z";
    }
    result.exhausted = true;
    return s;
  };
  while (t < tEnd) {
    if (work >= BACKDROP_TERRAIN_TRACE_BUDGET) { result.exhausted = true; return result; }
    const y = ro[1] + rd[1] * t;
    if (rd[1] >= 0 && y > plan.highest_m) return result;
    const probe = t + 1e-5 * (1 + t);
    const qx = ro[0] + rd[0] * probe - cx, qz = ro[2] + rd[2] * probe - cz;
    const m = Math.max(Math.abs(qx), Math.abs(qz));
    const level = m <= plan.halfWidth0_m ? 0 : Math.min(plan.levels - 1, Math.ceil(Math.log2(m / plan.halfWidth0_m)));
    const half = plan.halfWidth0_m * 2 ** level, edge = 2 * half / N;
    work += 1; result.tiles += 1;
    const [, squareExit] = squareInterval(half);
    if (Math.min(y, ro[1] + rd[1] * Math.min(squareExit, tEnd)) > plan.levelMaximum[level]!) { t = Math.max(squareExit, probe); continue; }
    const i = Math.min(N - 1, Math.max(0, Math.floor((qx + half) / edge)));
    const j = Math.min(N - 1, Math.max(0, Math.floor((qz + half) / edge)));
    const lowX = cx - half + i * edge, lowZ = cz - half + j * edge;
    let tileExit = Number.POSITIVE_INFINITY;
    if (rd[0] > 0) tileExit = Math.min(tileExit, (lowX + edge - ro[0]) / rd[0]);
    else if (rd[0] < 0) tileExit = Math.min(tileExit, (lowX - ro[0]) / rd[0]);
    if (rd[2] > 0) tileExit = Math.min(tileExit, (lowZ + edge - ro[2]) / rd[2]);
    else if (rd[2] < 0) tileExit = Math.min(tileExit, (lowZ - ro[2]) / rd[2]);
    const base = ((level * N + j) * N + i) * 3;
    const low = plan.tiles[base]!, high = plan.tiles[base + 1]!;
    if (low <= high) {
      const [a, b] = band(t, Math.min(tileExit, tEnd), low, high, plan.tiles[base + 2]! === 0);
      if (b >= a) {
        const hit = columns(a, b, level, i, j, low, high);
        if (result.exhausted) return result;
        if (hit >= 0) { result.t = hit; return result; }
      }
    }
    t = Math.max(tileExit, probe);
  }
  return result;

  /**
   * [a, b] clipped to where the ray can meet a column: from where it drops to
   * the highest top, and (unless the tile overlaps the footprint, whose
   * columns are air) up to where it passes under the lowest.
   */
  function band(a: number, b: number, low: number, high: number, clipLow: boolean): readonly [number, number] {
    if (!(b >= a)) return [1, 0];
    if (rd[1] < 0) return [Math.max(a, (high - ro[1]) / rd[1]), clipLow ? Math.min(b, Math.max(a, (low - ro[1]) / rd[1])) : b];
    if (rd[1] > 0) return [a, Math.min(b, (high - ro[1]) / rd[1])];
    return ro[1] > high ? [1, 0] : [a, b];
  }
}

// ---------------------------------------------------------------------------
// WGSL
// ---------------------------------------------------------------------------

export interface BackdropTerrainWGSLOptions {
  /** A `u32` load from the arena holding the table, e.g. `(i) => \`scenePayload[${i}]\``. */
  load: (index: string) => string;
  /** WGSL `u32` expression: the arena word where the table starts. */
  tableBase: string;
}

/**
 * The reader side: the field, the voxel column lookup, the stored-ring lattice and the
 * tiled walk. Self-contained, so any module holding the payload arena can
 * include it.
 */
export function backdropTerrainWGSL(options: BackdropTerrainWGSLOptions): string {
  const T = BACKDROP_TERRAIN_TABLE;
  const N = BACKDROP_TILES_PER_AXIS;
  const { load, tableBase } = options;
  return /* wgsl */ `
fn backdropTableWord(index:u32)->u32{return ${load(`(${tableBase})+index`)};}
fn backdropTableFloat(index:u32)->f32{return bitcast<f32>(backdropTableWord(index));}
// Zero on a world with no backdrop: the header is always allocated.
fn backdropTerrainLevels()->u32{return min(backdropTableWord(${T.countsWord + 2}u),${BACKDROP_TILE_MAXIMUM_LEVELS}u);}
fn backdropTerrainMaterial()->u32{return backdropTableWord(${T.countsWord + 1}u)&0xffffu;}
fn backdropFootprintMinimum()->vec2f{return vec2f(backdropTableFloat(${T.footprintWord}u),backdropTableFloat(${T.footprintWord + 1}u));}
fn backdropFootprintMaximum()->vec2f{return vec2f(backdropTableFloat(${T.footprintWord + 2}u),backdropTableFloat(${T.footprintWord + 3}u));}
fn backdropFootprintDistance(p:vec2f)->f32{return length(p-clamp(p,backdropFootprintMinimum(),backdropFootprintMaximum()));}
fn backdropInsideFootprint(p:vec2f)->bool{return all(p>backdropFootprintMinimum())&&all(p<backdropFootprintMaximum());}
fn backdropTerrainCentre()->vec2f{return vec2f(backdropTableFloat(${T.latticeWord + 2}u),backdropTableFloat(${T.latticeWord + 3}u));}
struct BackdropTerrainSurface{height:f32,gradient:vec2f}
// h(x, z) and its analytic gradient (backdropFieldHeightAndGradient).
fn backdropTerrainSurfaceAt(p:vec2f)->BackdropTerrainSurface{
  let footprintMinimum=vec2f(backdropTableFloat(${T.footprintWord}u),backdropTableFloat(${T.footprintWord + 1}u));
  let footprintMaximum=vec2f(backdropTableFloat(${T.footprintWord + 2}u),backdropTableFloat(${T.footprintWord + 3}u));
  let seam=backdropTableFloat(${T.envelopeWord}u);let flatRing=backdropTableFloat(${T.envelopeWord + 1}u);
  let offset=p-clamp(p,footprintMinimum,footprintMaximum);let d=length(offset);
  if(d<=flatRing){return BackdropTerrainSurface(seam,vec2f(0.0));}
  let rampWidth=backdropTableFloat(${T.envelopeWord + 2}u);let valleySpan=backdropTableFloat(${T.envelopeWord + 3}u);
  let valleyRise=backdropTableFloat(${T.shapeWord}u);let amplitude=backdropTableFloat(${T.shapeWord + 1}u);
  let gradientD=offset/d;
  let tr=clamp((d-flatRing)/rampWidth,0.0,1.0);
  let ramp=tr*tr*(3.0-2.0*tr);let rampSlope=6.0*tr*(1.0-tr)/rampWidth;
  let tv=clamp((d-flatRing)/valleySpan,0.0,1.0);
  let valley=tv*tv*(3.0-2.0*tv);let valleySlope=6.0*tv*(1.0-tv)/valleySpan;
  var n=0.0;var gn=vec2f(0.0);
  let waves=min(backdropTableWord(${T.countsWord}u),${BACKDROP_TERRAIN_MAXIMUM_WAVES}u);
  for(var wave=0u;wave<waves;wave+=1u){
    let base=${T.wavesWord}u+wave*${BACKDROP_WAVE_STRIDE}u;
    let direction=vec2f(backdropTableFloat(base),backdropTableFloat(base+1u));
    let k=backdropTableFloat(base+2u);let a=backdropTableFloat(base+4u);
    let phase=k*dot(direction,p)+backdropTableFloat(base+3u);
    n+=a*sin(phase);gn+=(a*k*cos(phase))*direction;
  }
  let body=valleyRise*valley+amplitude*(0.5+0.5*n);
  let gradient=rampSlope*body*gradientD+ramp*(valleyRise*valleySlope*gradientD+0.5*amplitude*gn);
  return BackdropTerrainSurface(seam+ramp*body,gradient);
}
// The finest level whose square holds the point.
fn backdropTerrainLevelAt(q:vec2f,half0:f32,levels:u32)->u32{
  let m=max(abs(q.x),abs(q.y));
  return select(0u,min(u32(max(ceil(log2(m/half0)),0.0)),levels-1u),m>half0);
}
// Where the stored detail rings are, and how coarse their voxels are.
//
// Ring l is walk level l stored as octree leaves of cell h0 2^l, and a leaf
// owns node-mip pages only at its own level (liveSvoLeafPage): nothing finer
// exists inside it. A cone that samples the pyramid finer than that reads
// empty space and misses the ring's content. So a cone's footprint is floored
// at the stored voxel of the point it samples, the pyramid's own resolution
// there. Stored levels are the leading ones the walk leaves empty (their
// cumulative maximum is -inf); zero rings means no floor at all. The host
// counts them once and writes the count into the header.
struct BackdropStoredLattice{centre:vec2f,half0:f32,cell0:f32,footprintMinimum:vec2f,footprintMaximum:vec2f,rings:u32}
fn backdropStoredLattice()->BackdropStoredLattice{
  let rings=min(backdropTableWord(${T.latticeWord + 1}u),backdropTerrainLevels());
  return BackdropStoredLattice(backdropTerrainCentre(),backdropTableFloat(${T.shapeWord + 2}u),backdropTableFloat(${T.shapeWord + 3}u),
    backdropFootprintMinimum(),backdropFootprintMaximum(),rings);
}
// The stored voxel edge at a point: the ring cell outside the set's footprint
// (the outermost ring's beyond the rings, where nothing is stored), zero
// inside it, where the set's own leaves are the finest level.
fn backdropStoredVoxelWidth(lattice:BackdropStoredLattice,p:vec3f)->f32{
  if(lattice.rings==0u||(all(p.xz>lattice.footprintMinimum)&&all(p.xz<lattice.footprintMaximum))){return 0.0;}
  let q=p.xz-lattice.centre;let m=max(abs(q.x),abs(q.y));
  let ring=select(0u,min(u32(max(ceil(log2(m/lattice.half0)),0.0)),lattice.rings-1u),m>lattice.half0);
  return lattice.cell0*exp2(f32(ring));
}
// Where a rising ray has cleared the content ceiling for good: the end of the
// last ceiling column whose top (plus lift + slope t, the margin a cone's
// sampled footprint needs) the ray is still below, capped at tMax. The walk
// is a 2D DDA over the tiles up to where the ray clears the highest column,
// descending into a tile's columns only where the ray may be under the tile's
// highest one. tMax when there is no ceiling, the ray does not out-climb the
// margin, or the walk runs out; zero when nothing stands anywhere along the ray.
fn backdropContentCeilingColumnsEnd(origin:vec3f,direction:vec3f,t0:f32,t1:f32,lift:f32,slope:f32,tile:vec2i,lastIn:f32)->f32{
  let base=backdropTableWord(${T.ceilingWord}u);let n=i32(backdropTableWord(${T.ceilingWord + 1}u));
  let corner=vec2f(backdropTableFloat(base),backdropTableFloat(base+1u));let cell=backdropTableFloat(base+2u);
  let p=(origin.xz-corner)/cell;let d=direction.xz/cell;let moving=abs(d)>vec2f(1e-12);
  let first=tile*${BACKDROP_CONTENT_CEILING_TILE_CELLS}i;
  // Clamped into the tile: the tile walk already decided the ray is in it.
  var c=clamp(vec2i(floor(p+d*t0)),first,first+vec2i(${BACKDROP_CONTENT_CEILING_TILE_CELLS - 1}));
  let advance=select(vec2i(0),select(vec2i(-1),vec2i(1),d>vec2f(0.0)),moving);
  var last=lastIn;var s0=t0;
  for(var i=0u;i<${2 * BACKDROP_CONTENT_CEILING_TILE_CELLS}u;i+=1u){
    let next=select(vec2f(3.0e38),(vec2f(select(c,c+vec2i(1),d>vec2f(0.0)))-p)/d,moving);
    let s1=clamp(min(next.x,next.y),s0,t1);
    if(origin.y+direction.y*s0-lift-slope*s1<=backdropTableFloat(base+${BACKDROP_CONTENT_CEILING_HEADER_WORDS}u+u32(c.y*n+c.x))){last=s1;}
    if(s1>=t1){break;}
    if(next.x<next.y){c.x+=advance.x;}else{c.y+=advance.y;}
    if(any(c<first)||any(c>=first+vec2i(${BACKDROP_CONTENT_CEILING_TILE_CELLS}))){break;}
    s0=s1;
  }
  return last;
}
fn backdropContentCeilingEnd(origin:vec3f,direction:vec3f,tMax:f32,lift:f32,slope:f32)->f32{
  let base=backdropTableWord(${T.ceilingWord}u);let rise=direction.y-slope;
  if(base==0u||!(rise>0.0)){return tMax;}
  let n=i32(backdropTableWord(${T.ceilingWord + 1}u));let tiles=n/${BACKDROP_CONTENT_CEILING_TILE_CELLS}i;
  let corner=vec2f(backdropTableFloat(base),backdropTableFloat(base+1u));
  let edge=backdropTableFloat(base+2u)*${BACKDROP_CONTENT_CEILING_TILE_CELLS}.0;
  let tileBase=base+${BACKDROP_CONTENT_CEILING_HEADER_WORDS}u+u32(n*n);
  let end=min(tMax,max((backdropTableFloat(base+3u)+lift-origin.y)/rise,0.0));
  let p=(origin.xz-corner)/edge;let d=direction.xz/edge;let moving=abs(d)>vec2f(1e-12);
  var c=vec2i(floor(p));
  if(any(c<vec2i(0))||any(c>=vec2i(tiles))){return tMax;}
  let advance=select(vec2i(0),select(vec2i(-1),vec2i(1),d>vec2f(0.0)),moving);
  let span=select(vec2f(3.0e38),1.0/abs(d),moving);
  var next=select(vec2f(3.0e38),(select(floor(p),floor(p)+1.0,d>vec2f(0.0))-p)/d,moving);
  var t0=0.0;var last=0.0;
  for(var i=0u;i<${2 * BACKDROP_CONTENT_CEILING_TILES + 2}u;i+=1u){
    if(t0>=end||any(c<vec2i(0))||any(c>=vec2i(tiles))){return last;}
    let t1=min(min(next.x,next.y),end);
    if(origin.y+direction.y*t0-lift-slope*t1<=backdropTableFloat(tileBase+u32(c.y*tiles+c.x))){
      last=backdropContentCeilingColumnsEnd(origin,direction,t0,t1,lift,slope,c,last);
    }
    if(next.x<next.y){c.x+=advance.x;next.x+=span.x;}else{c.y+=advance.y;next.y+=span.y;}
    t0=t1;
  }
  return tMax;
}
// The voxel a surface point belongs to, found by stepping a quarter of the
// finest cell back through the face it lies on: (centre x, centre z, cell,
// vertical lattice anchor). Shading reads the voxel's tone and cell seams
// from it, so every face of one voxel shades alike.
fn backdropTerrainVoxelAt(position:vec3f,normal:vec3f)->vec4f{
  let levels=max(backdropTerrainLevels(),1u);
  let centre=backdropTerrainCentre();
  let half0=backdropTableFloat(${T.shapeWord + 2}u);let cell0=backdropTableFloat(${T.shapeWord + 3}u);
  let inside=position.xz-normal.xz*(0.25*cell0);
  let cell=cell0*exp2(f32(backdropTerrainLevelAt(inside-centre,half0,levels)));
  let column=centre+(floor((inside-centre)/cell)+vec2f(0.5))*cell;
  return vec4f(column,cell,backdropTableFloat(${T.latticeWord}u));
}
// [enter, exit] of the ray over an xz rectangle, or an empty interval.
fn backdropRectangleInterval(ro:vec3f,rd:vec3f,low:vec2f,high:vec2f)->vec2f{
  var enter=-3.0e38;var exit=3.0e38;
  for(var axis=0u;axis<2u;axis+=1u){
    let o=select(ro.x,ro.z,axis==1u);let d=select(rd.x,rd.z,axis==1u);
    let lo=select(low.x,low.y,axis==1u);let hi=select(high.x,high.y,axis==1u);
    if(abs(d)<1e-12){if(o<lo||o>hi){return vec2f(1.0,0.0);}continue;}
    let a=(lo-o)/d;let b=(hi-o)/d;enter=max(enter,min(a,b));exit=min(exit,max(a,b));
  }
  return vec2f(enter,exit);
}
// [a, b] clipped to where the ray can meet a column: from where it drops to
// the highest top, and (unless the tile overlaps the footprint, whose columns
// are air) up to where it passes under the lowest. A start already under the
// band stays [a, ...], where the first column answers at once.
fn backdropTerrainBand(ro:vec3f,rd:vec3f,a:f32,b:f32,low:f32,high:f32,clipLow:bool)->vec2f{
  if(!(b>=a)){return vec2f(1.0,0.0);}
  if(rd.y<0.0){return vec2f(max(a,(high-ro.y)/rd.y),select(b,min(b,max(a,(low-ro.y)/rd.y)),clipLow));}
  if(rd.y>0.0){return vec2f(a,min(b,(high-ro.y)/rd.y));}
  return select(vec2f(a,b),vec2f(1.0,0.0),ro.y>high);
}
// face: 0 an x side, 1 the top, 2 a z side.
struct BackdropTerrainTrace{t:f32,face:u32,tiles:u32,steps:u32,exhausted:u32}
var<private> backdropTerrainWork:u32;
// The first voxel column the ray meets over [a, b] inside one tile, or -1: a
// 2D DDA over the level's columns. Each column costs one field evaluation at
// its centre; its top is that height rounded to the level's lattice and
// clamped to the tile's interval. The ray meets a column through the side it
// entered when it is already at or under the top there, or through the top
// when it descends to it before leaving the column.
fn backdropTerrainColumns(ro:vec3f,rd:vec3f,a:f32,b:f32,level:u32,tile:vec2u,low:f32,high:f32,trace:ptr<function,BackdropTerrainTrace>)->f32{
  let centre=backdropTerrainCentre();
  let cell=backdropTableFloat(${T.shapeWord + 3}u)*exp2(f32(level));
  let anchor=backdropTableFloat(${T.latticeWord}u);
  let perTile=i32(round(backdropTableFloat(${T.shapeWord + 2}u)/(backdropTableFloat(${T.shapeWord + 3}u)*${N / 2}.0)));
  let first=(vec2i(tile)-vec2i(${N / 2}))*perTile;
  let probe=a+1e-5*(1.0+a);
  var index=clamp(vec2i(floor((ro.xz+rd.xz*probe-centre)/cell)),first,first+vec2i(perTile-1));
  let lo=centre+vec2f(index)*cell;
  let positive=rd.xz>vec2f(0.0);let still=rd.xz==vec2f(0.0);
  let enter=select(select(lo+vec2f(cell),lo,positive)-ro.xz,vec2f(0.0),still)/select(rd.xz,vec2f(1.0),still);
  var exit=select(select(lo,lo+vec2f(cell),positive)-ro.xz,vec2f(0.0),still)/select(rd.xz,vec2f(1.0),still);
  let entered=select(enter,vec2f(-3.0e38),still);exit=select(exit,vec2f(3.0e38),still);
  let stepT=select(vec2f(cell)/abs(rd.xz),vec2f(0.0),still);
  let stepIndex=select(vec2i(-1),vec2i(1),positive);
  var s=a;
  var face=select(1u,select(2u,0u,entered.x>=entered.y),s<=max(entered.x,entered.y)+1e-5*(1.0+s));
  while(backdropTerrainWork<${BACKDROP_TERRAIN_TRACE_BUDGET}u){
    backdropTerrainWork+=1u;(*trace).steps+=1u;
    let columnExit=min(exit.x,exit.y);
    let column=centre+(vec2f(index)+vec2f(0.5))*cell;
    if(!backdropInsideFootprint(column)){
      let top=clamp(anchor+floor((backdropTerrainSurfaceAt(column).height-anchor)/cell+0.5)*cell,low,high);
      if(ro.y+rd.y*s<=top){(*trace).face=face;return s;}
      if(rd.y<0.0){let tTop=(top-ro.y)/rd.y;if(tTop<=columnExit){(*trace).face=1u;return tTop;}}
    }
    if(columnExit>=b){return -1.0;}
    s=columnExit;
    if(exit.x<=exit.y){index.x+=stepIndex.x;exit.x+=stepT.x;face=0u;}
    else{index.y+=stepIndex.y;exit.y+=stepT.y;face=2u;}
  }
  (*trace).exhausted=1u;
  return s;
}
// The first voxel of the backdrop ground the ray meets in [tMin, tMax], or
// t = -1: the tiled walk (see backdrop-terrain-tiles.ts).
fn backdropTerrainTrace(ro:vec3f,rd:vec3f,tMin:f32,tMax:f32)->BackdropTerrainTrace{
  var trace=BackdropTerrainTrace(-1.0,1u,0u,0u,0u);
  let levels=backdropTerrainLevels();if(levels==0u){return trace;}
  let highest=backdropTableFloat(${T.countsWord + 3}u);
  if(rd.y>=0.0&&ro.y>highest){return trace;}
  let centre=backdropTerrainCentre();
  let half0=backdropTableFloat(${T.shapeWord + 2}u);
  let outerHalf=half0*exp2(f32(levels-1u));
  let outer=backdropRectangleInterval(ro,rd,centre-vec2f(outerHalf),centre+vec2f(outerHalf));
  var t=max(max(tMin,0.0),outer.x);let tEnd=min(tMax,outer.y);
  backdropTerrainWork=0u;
  loop{
    if(!(t<tEnd)){break;}
    if(backdropTerrainWork>=${BACKDROP_TERRAIN_TRACE_BUDGET}u){trace.exhausted=1u;break;}
    let y=ro.y+rd.y*t;
    if(rd.y>=0.0&&y>highest){break;}
    let probe=t+1e-5*(1.0+t);
    let q=ro.xz+rd.xz*probe-centre;
    let level=backdropTerrainLevelAt(q,half0,levels);
    let half=half0*exp2(f32(level));let edge=half*${2 / N};
    backdropTerrainWork+=1u;trace.tiles+=1u;
    // The level's whole square, finer levels included, lies under its maximum:
    // a ray above it at both ends of its chord meets nothing in there.
    let square=backdropRectangleInterval(ro,rd,centre-vec2f(half),centre+vec2f(half));
    if(min(y,ro.y+rd.y*min(square.y,tEnd))>backdropTableFloat(${T.levelMaximumWord}u+level)){t=max(square.y,probe);continue;}
    let tile=vec2u(clamp(floor((q+vec2f(half))/edge),vec2f(0.0),vec2f(${N - 1}.0)));
    let low=centre-vec2f(half)+vec2f(tile)*edge;
    let exits=select(select(vec2f(3.0e38),(low-ro.xz)/rd.xz,rd.xz<vec2f(0.0)),(low+vec2f(edge)-ro.xz)/rd.xz,rd.xz>vec2f(0.0));
    let tileExit=min(exits.x,exits.y);
    let record=${T.tilesWord}u+((level*${N}u+tile.y)*${N}u+tile.x)*${T.tileWords}u;
    let groundLow=backdropTableFloat(record);let groundHigh=backdropTableFloat(record+1u);
    if(groundLow<=groundHigh){
      let band=backdropTerrainBand(ro,rd,t,min(tileExit,tEnd),groundLow,groundHigh,backdropTableFloat(record+2u)==0.0);
      if(band.y>=band.x){
        let hit=backdropTerrainColumns(ro,rd,band.x,band.y,level,tile,groundLow,groundHigh,&trace);
        if(trace.exhausted!=0u){trace.t=hit;return trace;}
        if(hit>=0.0){trace.t=hit;return trace;}
      }
    }
    t=max(tileExit,probe);
  }
  return trace;
}`;
}
