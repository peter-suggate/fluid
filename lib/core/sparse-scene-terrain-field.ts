import type { SparseBrickCoordinate } from "../svo/sparse-brick-octree";
import {
  TERRAIN_CEILING_MARGIN_M,
  terrainCeiling,
  terrainColumnHeightsForLatticeSteps,
  terrainContentStamp,
  type TerrainDescription,
} from "./terrain";

/**
 * The ground, as ordinary voxel coverage.
 *
 * Terrain is the one piece of authored content in this engine that is special
 * everywhere: it is the only render-ABI kind that is not a finite record, and
 * that single fact suppresses the candidate BVH scene-wide
 * (`lib/svo-scene-primitives.ts`), throws in live proxy updates
 * (`lib/webgpu-sparse-scene-proxies.ts`), and carries a bespoke marcher with a
 * CPU mirror that has to agree with it. Worst of all it never reached the
 * octree, so the largest surface in the hero garden was drawn by primary
 * visibility and invisible to every cone, shadow and GI ray that reads the
 * opacity pyramid.
 *
 * This module is the correction, and it is deliberately *not* a new kind. There
 * is no terrain record, no tenth shape number, no new ABI surface: the ground
 * becomes a column-height field the per-voxel rebuild consults alongside its
 * binned candidates, and downstream of the scene lanes nothing can tell terrain
 * from a voxelized box. Everything that made terrain special stays where it is
 * — authoring, collision, the analytic primary — and can be retired later
 * without another format.
 *
 * The lattice is the octree's own finest cell lattice over its own domain, so a
 * voxel's column index is its world cell coordinate and no origin, spacing or
 * transform has to be transmitted or agreed. That is the whole reason the field
 * costs three header words.
 */
export interface SparseSceneTerrainField {
  /** Columns along x and z, at the octree's finest cell size. */
  readonly dimensions: readonly [number, number];
  /** World-space Y of the ground at each column centre, row-major x + nx * z. */
  readonly heights_m: Float32Array<ArrayBuffer>;
  /** Lowest and highest column, for bounds and for brick selection. */
  readonly minimumHeight_m: number;
  readonly maximumHeight_m: number;
  /**
   * The world box the ground occupies inside the domain: the whole footprint,
   * from the domain floor up to the ceiling every marcher already brackets at.
   *
   * This is what a publication dirties when the ground changes, and it is a
   * *region* rather than a record's bounds precisely because there is no record.
   */
  readonly bounds: {
    readonly minimum: readonly [number, number, number];
    readonly maximum: readonly [number, number, number];
  };
}

/** The octree domain a terrain field is baked onto. */
export interface SparseSceneTerrainDomain {
  readonly worldOrigin_m: readonly [number, number, number];
  readonly cellSize_m: readonly [number, number, number];
  /** Finest-level cells along each axis; the field is one column per (x, z). */
  readonly dimensionsCells: readonly [number, number, number];
}

/**
 * The same emptiness test `sceneHasTerrain` applies, without needing a scene.
 *
 * `procedural` is a ground description like any other and has to count: a
 * described vessel with `baseHeight_m === 0` and no features would otherwise
 * get no terrain field at all, and the voxeliser would see a scene with no
 * ground rather than one whose ground it cannot bake.
 */
export function sparseSceneTerrainPresent(terrain: TerrainDescription | undefined): boolean {
  return !!terrain
    && (terrain.baseHeight_m > 0 || terrain.features.length > 0 || !!terrain.grid || !!terrain.procedural);
}

/**
 * The last bake, keyed by what it is a pure function of.
 *
 * One slot, not a table, because the access pattern is a *pair*: the world
 * constructor bakes the field to choose its bricks, and the first publication
 * immediately bakes it again to upload it. Those two calls are adjacent and
 * identical, so a single slot converts the second into a pointer copy — and a
 * second slot would only buy re-entry into an older domain, which is a rebuild
 * and re-bakes everything anyway.
 *
 * It costs nothing in retained memory: `configureTerrainField` holds the same
 * object for the voxelizer's lifetime, so the slot aliases a live array rather
 * than keeping a dead one alive.
 *
 * Keying on {@link terrainContentStamp} rather than object identity is the
 * load-bearing part. The scene is structured-cloned across the render worker
 * boundary, so `terrain` is a *fresh object* on every republication and an
 * identity check misses every time — for a bake that is one procedural
 * evaluation per finest column of the whole domain footprint: ~221k at the
 * shipping leaf, and **14.2 M at environment refinement depth 3**.
 */
let lastTerrainField: {
  readonly key: string;
  readonly field: SparseSceneTerrainField | undefined;
} | undefined;

/** The domain half of the bake's cache key; the terrain half is its content stamp. */
function terrainFieldDomainKey(domain: SparseSceneTerrainDomain): string {
  return `${domain.worldOrigin_m.join(",")}|${domain.cellSize_m.join(",")}|${domain.dimensionsCells.join(",")}`;
}

/**
 * Drop the memoized bake.
 *
 * Only needed by callers that deliberately mutate a `TerrainDescription` in
 * place, which nothing on the production path does — the stamp covers every
 * field the bake reads, so an honest edit produces a new stamp and misses.
 */
export function forgetSparseSceneTerrainField(): void { lastTerrainField = undefined; }

/**
 * Bake the ground onto the octree's finest cell lattice.
 *
 * Undefined when the scene has no ground, which is the encoding for "this
 * voxelizer has no terrain field" all the way down to the shader: a zero
 * dimension in the header block switches the whole path off, so a scene without
 * terrain allocates nothing, writes nothing and evaluates nothing.
 */
export function planSparseSceneTerrainField(
  terrain: TerrainDescription | undefined,
  domain: SparseSceneTerrainDomain,
): SparseSceneTerrainField | undefined {
  const steps = planSparseSceneTerrainFieldSteps(terrain, domain);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}

/**
 * The same bake, offered as slices.
 *
 * The memo is only written once the bake has *returned*, so a build abandoned
 * part-way through the ground can never leave a half-filled field where the
 * next build would find a hit — the cost of that is re-baking from the start,
 * which is the correct trade for a cache whose whole value is that a hit is
 * exactly the field a miss would have produced.
 */
export function* planSparseSceneTerrainFieldSteps(
  terrain: TerrainDescription | undefined,
  domain: SparseSceneTerrainDomain,
): Generator<unknown, SparseSceneTerrainField | undefined, undefined> {
  const key = `${terrainContentStamp(terrain)}|${terrainFieldDomainKey(domain)}`;
  if (lastTerrainField?.key === key) return lastTerrainField.field;
  const field = yield* bakeSparseSceneTerrainFieldSteps(terrain, domain);
  lastTerrainField = { key, field };
  return field;
}

function* bakeSparseSceneTerrainFieldSteps(
  terrain: TerrainDescription | undefined,
  domain: SparseSceneTerrainDomain,
): Generator<unknown, SparseSceneTerrainField | undefined, undefined> {
  if (!sparseSceneTerrainPresent(terrain)) return undefined;
  const [nx, , nz] = domain.dimensionsCells;
  if (!Number.isSafeInteger(nx) || !Number.isSafeInteger(nz) || nx < 1 || nz < 1) {
    throw new RangeError("Terrain field needs a positive integer cell footprint");
  }
  const domainTop_m = domain.worldOrigin_m[1] + domain.dimensionsCells[1] * domain.cellSize_m[1];
  const heights_m = yield* terrainColumnHeightsForLatticeSteps(terrain, {
    originX_m: domain.worldOrigin_m[0], originZ_m: domain.worldOrigin_m[2],
    cellX_m: domain.cellSize_m[0], cellZ_m: domain.cellSize_m[2],
    nx, nz, maximumHeight_m: domainTop_m,
  });
  let minimumHeight_m = Infinity, maximumHeight_m = -Infinity;
  for (const height of heights_m) {
    if (height < minimumHeight_m) minimumHeight_m = height;
    if (height > maximumHeight_m) maximumHeight_m = height;
  }
  if (!Number.isFinite(minimumHeight_m)) { minimumHeight_m = 0; maximumHeight_m = 0; }
  // The ceiling every marcher already brackets at, so the voxelized ground and
  // the analytic one agree about where ground can possibly be. Clamped into the
  // domain because a region outside it dirties nothing and only widens the
  // invalidation dispatch.
  const ceiling_m = Math.min(domainTop_m,
    Math.max(maximumHeight_m + TERRAIN_CEILING_MARGIN_M, terrainCeiling(terrain)));
  return {
    dimensions: [nx, nz],
    heights_m,
    minimumHeight_m,
    maximumHeight_m,
    bounds: {
      minimum: [domain.worldOrigin_m[0], domain.worldOrigin_m[1], domain.worldOrigin_m[2]],
      maximum: [
        domain.worldOrigin_m[0] + nx * domain.cellSize_m[0],
        ceiling_m,
        domain.worldOrigin_m[2] + nz * domain.cellSize_m[2],
      ],
    },
  };
}

/** The ground height at one finest cell column, clamped to the field's edge. */
export function sparseSceneTerrainColumnHeight(field: SparseSceneTerrainField, cellX: number, cellZ: number): number {
  const [nx, nz] = field.dimensions;
  const x = Math.min(nx - 1, Math.max(0, cellX));
  const z = Math.min(nz - 1, Math.max(0, cellZ));
  return field.heights_m[x + nx * z];
}

/** Lowest and highest ground inside one finest-cell box footprint, inclusive. */
export function sparseSceneTerrainColumnRange(
  field: SparseSceneTerrainField,
  firstCellX: number, firstCellZ: number, lastCellX: number, lastCellZ: number,
): { minimum_m: number; maximum_m: number } {
  let minimum_m = Infinity, maximum_m = -Infinity;
  for (let z = firstCellZ; z <= lastCellZ; z += 1) for (let x = firstCellX; x <= lastCellX; x += 1) {
    const height = sparseSceneTerrainColumnHeight(field, x, z);
    if (height < minimum_m) minimum_m = height;
    if (height > maximum_m) maximum_m = height;
  }
  return { minimum_m, maximum_m };
}

/**
 * The finest bricks the ground occupies, for the octree plan.
 *
 * Every brick from the domain floor up to the tallest column inside its own
 * footprint, so the set is the solid the ground actually is rather than the box
 * that contains it — a garden whose ground rises on one side does not claim the
 * sky on the other. The interior bricks are claimed alongside the surface ones
 * deliberately: a coarse mip level averages its eight children, and a hollow
 * ground reads as half-transparent to a cone sampling it two levels up.
 */
export function sparseSceneTerrainBrickCoordinates(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
): SparseBrickCoordinate[] {
  const steps = sparseSceneTerrainBrickCoordinatesSteps(field, domain, brickSize);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}

/**
 * The same solid-column claim, offered a column at a time.
 *
 * Sliced as well as the shelled claim beside it because this is the branch a
 * lane without `FLUID_SVO_GROUND_SHELL` takes, and a build that is
 * interruptible only under one setting of a lever is not interruptible.
 */
export function* sparseSceneTerrainBrickCoordinatesSteps(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
): Generator<unknown, SparseBrickCoordinate[], undefined> {
  const brickDimensions = domain.dimensionsCells.map((cells) => Math.ceil(cells / brickSize));
  const brickEdge = domain.cellSize_m.map((size) => size * brickSize);
  const coordinates: SparseBrickCoordinate[] = [];
  let columns = 0;
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let x = 0; x < brickDimensions[0]; x += 1) {
    if ((columns += 1) % 512 === 0) yield;
    const range = sparseSceneTerrainColumnRange(field,
      x * brickSize, z * brickSize, (x + 1) * brickSize - 1, (z + 1) * brickSize - 1);
    if (!Number.isFinite(range.maximum_m)) continue;
    const top = Math.min(brickDimensions[1] - 1,
      Math.floor((range.maximum_m - domain.worldOrigin_m[1]) / brickEdge[1]));
    for (let y = 0; y <= top; y += 1) coordinates.push({ x, y, z });
  }
  return coordinates;
}

/**
 * Bricks of buried ground a shelled claim keeps below the surface band, or
 * `undefined` for the solid column that shipped.
 *
 * The claim above is a *volume*: on `hero-garden-hose` it is 9 115 finest
 * bricks of a 10 370-page tree at 6.25 mm, 68 167 of 77 044 at 3.125 mm and
 * 527 316 of 591 718 at 1.5625 mm — 88 % of the tree at every rung, growing
 * 7.5x per halving of the cell while the ground's *surface* grows 3.6x. That
 * factor of two per level is the whole reason 0.78 mm is unreachable: over
 * three levels it is 512x against 64x.
 *
 * A shell claims the band the ground boundary passes through — exactly the
 * `surface` half of {@link sparseSceneTerrainBrickCoverage} — plus this many
 * bricks of margin under it, so the cost becomes the ground's area and the
 * per-level growth becomes 4x.
 *
 * **This is not free and the lever is off until it is paid for.** A region with
 * no leaf has no mip page, and a missing page samples as *zero coverage* rather
 * than as "unknown" (`dryNodeMipAt` returns `valid = 1u` for a non-resident
 * page, `lib/webgpu-svo-dry-scene.ts:1826`), so the elided interior reads as
 * air. At the base level a cone never gets there — it is behind an opaque band —
 * but a *parent* page averages its eight children and `childSample` returns
 * `vec4f(0.)` for a child that is not resident
 * (`lib/webgpu-svo-live-derived-builder.ts:394`), so a coarse page straddling
 * the boundary reports a fraction of the opacity it should. A level-`L` page
 * spans `8 * 2^L` finest cells, so a margin of `m` bricks is exact for every
 * level up to `log2(m)` and progressively wrong above it. The margin is the
 * knob that trades pages against how many levels of the pyramid still see
 * solid ground.
 *
 * Measured on `hero-garden-hose` at margin 1, against the same frame with the
 * lever off, the error is **not** the expected light leak: the sky is
 * bit-identical and every lit surface comes out *darker* — lit ground −5.1 %
 * and the shadowed basin −8.9 % at 6.25 mm, −9.0 % and −14.9 % at 3.125 mm. The
 * elided interior does not let the sun through the ground; it removes the
 * ground's own bounce from the coarse pages a grazing cone integrates, so the
 * scene loses indirect light rather than gaining direct.
 *
 * Default `off`, in the same shape as `FLUID_SVO_BRICK_CLAIM` and
 * `FLUID_SVO_SOLVER_CLAIM`; any non-negative integer switches it on with that
 * many buried bricks of margin, and `0` is the bare band.
 */
export function sparseSceneTerrainShellBricks(
  environment: Record<string, string | undefined> | undefined
    = typeof process !== "undefined" ? process.env : undefined,
): number | undefined {
  const raw = environment?.FLUID_SVO_GROUND_SHELL;
  if (raw === undefined || raw === "" || raw === "off") return undefined;
  const value = Number(raw);
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError("FLUID_SVO_GROUND_SHELL must be \"off\" or a non-negative integer brick margin");
  }
  return value;
}

/**
 * The ground's claim as the octree plan should receive it.
 *
 * The solid column of {@link sparseSceneTerrainBrickCoordinates} unless
 * {@link sparseSceneTerrainShellBricks} says otherwise, in which case the
 * column is walked down from its top and stopped once `shellBricks` buried
 * bricks have been taken. The stopping test is
 * {@link sparseSceneTerrainNodeCoverage} rather than a second height
 * comparison, so the shell and the surface/buried split cannot disagree about
 * where the band ends, and the band is contiguous from the top by
 * construction: a column is `outside` above `range.maximum_m`, `surface`
 * between the two extremes and `buried` below `range.minimum_m`.
 */
export function sparseSceneTerrainClaimCoordinates(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
  shellBricks: number | undefined = sparseSceneTerrainShellBricks(),
): SparseBrickCoordinate[] {
  const steps = sparseSceneTerrainClaimCoordinatesSteps(field, domain, brickSize, shellBricks);
  for (;;) { const step = steps.next(); if (step.done) return step.value; }
}

/**
 * The same claim, offered a column of bricks at a time.
 *
 * 66 ms at environment refinement depth 3 on `hero-garden-hose` — small beside
 * the plan, and large beside a frame. A column is the unit because the shell
 * walk is per column and depends on nothing outside it, so a suspended claim is
 * always a whole number of finished columns.
 */
export function* sparseSceneTerrainClaimCoordinatesSteps(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
  shellBricks: number | undefined = sparseSceneTerrainShellBricks(),
): Generator<unknown, SparseBrickCoordinate[], undefined> {
  if (shellBricks === undefined) return yield* sparseSceneTerrainBrickCoordinatesSteps(field, domain, brickSize);
  const brickDimensions = domain.dimensionsCells.map((cells) => Math.ceil(cells / brickSize));
  const brickEdge = domain.cellSize_m.map((size) => size * brickSize);
  const coordinates: SparseBrickCoordinate[] = [];
  let columns = 0;
  for (let z = 0; z < brickDimensions[2]; z += 1) for (let x = 0; x < brickDimensions[0]; x += 1) {
    if ((columns += 1) % 512 === 0) yield;
    const range = sparseSceneTerrainColumnRange(field,
      x * brickSize, z * brickSize, (x + 1) * brickSize - 1, (z + 1) * brickSize - 1);
    if (!Number.isFinite(range.maximum_m)) continue;
    const top = Math.min(brickDimensions[1] - 1,
      Math.floor((range.maximum_m - domain.worldOrigin_m[1]) / brickEdge[1]));
    let buriedKept = 0;
    for (let y = top; y >= 0; y -= 1) {
      const coordinate = { x, y, z };
      if (sparseSceneTerrainNodeCoverage(field, domain, brickSize, 0, 0, coordinate) === "buried") {
        if (buriedKept >= shellBricks) break;
        buriedKept += 1;
      }
      coordinates.push(coordinate);
    }
  }
  return coordinates;
}

/**
 * Where one octree node sits relative to the ground.
 *
 * `surface` is the only class that has to be resolved finely: it is the one
 * whose extent the ground boundary passes through, so its eight children are
 * not all alike. `buried` is uniformly solid and `outside` is uniformly empty,
 * and neither gains anything from a child.
 */
export type SparseSceneTerrainNodeCoverage = "outside" | "surface" | "buried";

/**
 * Column extremes over aligned power-of-two footprints, one level per size.
 *
 * The classification below is called once per octree node the planner visits,
 * and the naive form rescans the node's whole footprint every time — so the same
 * column is read once per level of the tree above it, and a coarse node rescans
 * thousands. A min/max mip makes every query one lookup: measured over the hero
 * garden's whole node set at 3.125 mm, 34 ms against 153 ms.
 *
 * That is not where the plan's time goes today — 500 ms of the 3.125 mm plan is
 * the planner's own Morton prefix sets and the per-node primitive scan, and this
 * removes a rescan that was never the largest term. It is here because the
 * rescan is the term that grows fastest: it is quadratic in the node footprint,
 * so it doubles its share of the plan on every halving of the cell.
 *
 * Cached against the field's identity rather than rebuilt, because that is
 * already how this module treats a ground description: scene documents are
 * replaced rather than edited and a brush stroke returns a new field, so `!==`
 * is exactly "the ground changed" (see `terrainDescription` in
 * `lib/webgpu-octree-sparse-bricks.ts`).
 */
interface TerrainColumnPyramid {
  /** Level `k` covers 2^k by 2^k columns; index `x + dimensions[k][0] * z`. */
  readonly levels: readonly { dimensions: readonly [number, number]; minimum: Float32Array; maximum: Float32Array }[];
}

const terrainColumnPyramids = new WeakMap<SparseSceneTerrainField, TerrainColumnPyramid>();

function terrainColumnPyramid(field: SparseSceneTerrainField): TerrainColumnPyramid {
  const cached = terrainColumnPyramids.get(field);
  if (cached) return cached;
  const [nx, nz] = field.dimensions;
  const levels: { dimensions: readonly [number, number]; minimum: Float32Array; maximum: Float32Array }[] = [{
    dimensions: [nx, nz],
    minimum: Float32Array.from(field.heights_m),
    maximum: Float32Array.from(field.heights_m),
  }];
  while (levels[levels.length - 1].dimensions[0] > 1 || levels[levels.length - 1].dimensions[1] > 1) {
    const source = levels[levels.length - 1];
    const [sx, sz] = source.dimensions;
    const dimensions: readonly [number, number] = [Math.ceil(sx / 2), Math.ceil(sz / 2)];
    const minimum = new Float32Array(dimensions[0] * dimensions[1]);
    const maximum = new Float32Array(dimensions[0] * dimensions[1]);
    for (let z = 0; z < dimensions[1]; z += 1) for (let x = 0; x < dimensions[0]; x += 1) {
      let lo = Infinity, hi = -Infinity;
      // Up to four children: an odd dimension leaves the last block with fewer,
      // which is the same answer clamping to the field's edge would give,
      // because repeating a value cannot move a minimum or a maximum.
      for (let dz = 0; dz < 2; dz += 1) for (let dx = 0; dx < 2; dx += 1) {
        const childX = 2 * x + dx, childZ = 2 * z + dz;
        if (childX >= sx || childZ >= sz) continue;
        lo = Math.min(lo, source.minimum[childX + sx * childZ]);
        hi = Math.max(hi, source.maximum[childX + sx * childZ]);
      }
      minimum[x + dimensions[0] * z] = lo;
      maximum[x + dimensions[0] * z] = hi;
    }
    levels.push({ dimensions, minimum, maximum });
  }
  const pyramid: TerrainColumnPyramid = { levels };
  terrainColumnPyramids.set(field, pyramid);
  return pyramid;
}

/**
 * Column extremes over one aligned 2^power square of columns.
 *
 * Only defined for a footprint whose origin is a multiple of its own size, which
 * every octree node's is: a node at `level` starts at `coordinate * 2^(depth -
 * level) * brickSize` and spans exactly that. Out-of-range blocks clamp to the
 * last one rather than to the edge column, which can only widen the range and so
 * can only turn a `buried` or `outside` answer into `surface` — more refinement,
 * never a gap.
 */
function alignedColumnRange(
  field: SparseSceneTerrainField, power: number, blockX: number, blockZ: number,
): { minimum_m: number; maximum_m: number } {
  const { levels } = terrainColumnPyramid(field);
  const level = levels[Math.min(power, levels.length - 1)];
  const [nx, nz] = level.dimensions;
  const shift = Math.max(0, power - (levels.length - 1));
  const x = Math.min(nx - 1, Math.max(0, blockX >> shift));
  const z = Math.min(nz - 1, Math.max(0, blockZ >> shift));
  return { minimum_m: level.minimum[x + nx * z], maximum_m: level.maximum[x + nx * z] };
}

/**
 * Lowest and highest ground under one octree node's footprint.
 *
 * The same aligned pyramid lookup {@link sparseSceneTerrainNodeCoverage} makes,
 * exposed because the classification is not the only question asked of a node's
 * columns: a planarity test needs the true extremes to know whether a ridge
 * runs between the corners it sampled
 * (`lib/svo-environment-refinement.ts`). Sharing the lookup rather than
 * rebaking a second one is what keeps the pyramid private and the two answers
 * about one node's ground consistent by construction.
 */
export function sparseSceneTerrainNodeColumnRange(
  field: SparseSceneTerrainField,
  brickSize: number,
  level: number,
  maximumDepth: number,
  coordinate: SparseBrickCoordinate,
): { minimum_m: number; maximum_m: number } {
  const cellsPerNode = 2 ** (maximumDepth - level) * brickSize;
  return alignedColumnRange(field, Math.log2(cellsPerNode), coordinate.x, coordinate.z);
}

/**
 * Classify one octree node's extent against the ground.
 *
 * A node at `level` in a tree of depth `maximumDepth` covers
 * `2^(maximumDepth - level)` finest bricks per axis, so its world extent is that
 * scale times the finest brick edge. The classification is the two-sided version
 * of the test `sparseSceneTerrainBrickCoordinates` already applies per column:
 * ground fills every column from the domain floor up to that column's height, so
 * a node is uniformly solid exactly when its top is at or below the *lowest*
 * column it covers, and uniformly empty when its bottom is at or above the
 * *highest* one.
 *
 * The inequalities are deliberately asymmetric about which column extreme they
 * use. Using the maximum for both would call a node buried while part of it sat
 * in open air under a ridge — a solid block hanging in the sky — and using the
 * minimum for both would call a node empty while ground still filled its floor,
 * which is the hole this whole module exists to avoid.
 */
export function sparseSceneTerrainNodeCoverage(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
  level: number,
  maximumDepth: number,
  coordinate: SparseBrickCoordinate,
): SparseSceneTerrainNodeCoverage {
  if (!Number.isInteger(level) || !Number.isInteger(maximumDepth) || level < 0 || level > maximumDepth) {
    throw new RangeError("Terrain node level must be an integer in [0, maximumDepth]");
  }
  if (!Number.isSafeInteger(brickSize) || brickSize < 1 || (brickSize & (brickSize - 1)) !== 0) {
    throw new RangeError("Terrain node brick size must be a positive power of two");
  }
  const cellsPerNode = 2 ** (maximumDepth - level) * brickSize;
  // The footprint is aligned to its own size by construction, so the column
  // pyramid answers it with one lookup instead of a `cellsPerNode^2` rescan.
  const range = alignedColumnRange(field, Math.log2(cellsPerNode), coordinate.x, coordinate.z);
  if (!Number.isFinite(range.maximum_m)) return "outside";
  const floor_m = domain.worldOrigin_m[1] + coordinate.y * cellsPerNode * domain.cellSize_m[1];
  const ceiling_m = floor_m + cellsPerNode * domain.cellSize_m[1];
  if (floor_m >= range.maximum_m) return "outside";
  if (ceiling_m <= range.minimum_m) return "buried";
  return "surface";
}

/**
 * The ground's finest bricks, split by whether the surface passes through them.
 *
 * Same set as {@link sparseSceneTerrainBrickCoordinates} — the union is exactly
 * its return value, and a test asserts that — because the buried interior still
 * has to be *claimed*. What changes downstream is only how deeply it is
 * resolved: the claim gives the planner ancestors to hang a coarse leaf on, and
 * a coarse leaf covering solid ground reports the same full opacity to every
 * cone that a tower of fine ones would. Dropping the interior from the claim
 * instead is what makes the ground translucent, because a region with no leaf
 * has no mip page, and a missing page samples as zero coverage rather than as
 * "unknown" (`lib/webgpu-svo-dry-scene.ts:1826` returns `valid = 1u`).
 */
export function sparseSceneTerrainBrickCoverage(
  field: SparseSceneTerrainField,
  domain: SparseSceneTerrainDomain,
  brickSize: number,
): { surface: SparseBrickCoordinate[]; buried: SparseBrickCoordinate[] } {
  const maximumDepth = 0;
  const surface: SparseBrickCoordinate[] = [];
  const buried: SparseBrickCoordinate[] = [];
  for (const coordinate of sparseSceneTerrainBrickCoordinates(field, domain, brickSize)) {
    // At the finest level a node *is* a brick, so `level === maximumDepth` and
    // the scale is one; the classifier is reused rather than re-derived so the
    // split and the planner's descent can never disagree about a boundary.
    const coverage = sparseSceneTerrainNodeCoverage(field, domain, brickSize, maximumDepth, maximumDepth, coordinate);
    (coverage === "buried" ? buried : surface).push(coordinate);
  }
  return { surface, buried };
}
