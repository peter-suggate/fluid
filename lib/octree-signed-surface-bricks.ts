/**
 * Collision-free structural contract for an octree-native signed surface band.
 *
 * Active 4^3 bricks store base-cell samples on both sides of phi=0. Outside
 * the active band, compact octree tiles carry a saturated liquid/air sign.
 * Keys and tiles are sorted arrays, so the representation has no box-sized
 * page table and no insertion-order-dependent hash collisions.
 */

export type OctreeSurfaceSign = -1 | 1;
export type OctreeSurfaceVec3 = readonly [number, number, number];

export interface OctreeSignedSurfaceBrickOptions {
  readonly brickSize?: 4;
  /** Finest-cell distance retained on each side of phi=0. */
  readonly bandCells?: number;
  /** Capacity multiplier for curvature, motion and fragmentation. */
  readonly capacityHeadroom?: number;
  /** Number of transported/redistance scalar fields stored per brick. */
  readonly scalarFieldCount?: number;
  /** Compact inactive sign-tile records. Defaults to one per brick slot. */
  readonly inactiveTileCapacity?: number;
}

export interface OctreeSignedSurfaceBrickPlan {
  readonly dimensions: OctreeSurfaceVec3;
  readonly logicalBrickDimensions: OctreeSurfaceVec3;
  readonly brickSize: 4;
  readonly samplesPerBrick: 64;
  readonly bandCells: number;
  readonly bandThicknessCells: number;
  readonly interfaceAreaCells: number;
  readonly unpaddedBrickCount: number;
  readonly brickCapacity: number;
  readonly inactiveTileCapacity: number;
  readonly scalarFieldCount: number;
  readonly keyBytes: number;
  readonly stateBytes: number;
  readonly activeListBytes: number;
  readonly inactiveTileBytes: number;
  readonly scalarBytes: number;
  readonly controlBytes: number;
  readonly allocatedBytes: number;
  readonly denseEquivalentBytes: number;
  readonly savedBytes: number;
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < 1) throw new RangeError(`${label} must be a positive integer`);
  return value;
}

function dimensions3(value: OctreeSurfaceVec3): OctreeSurfaceVec3 {
  return value.map((component, axis) => {
    const checked = positiveInteger(component, `Surface dimension ${axis}`);
    if (checked > 1024) throw new RangeError(`Surface dimension ${axis} must fit ten coordinate bits`);
    return checked;
  }) as unknown as OctreeSurfaceVec3;
}

/**
 * Capacity estimate for a surface with the supplied finest-cell area.
 *
 * The area is packed into 4x4 brick cross-sections and the complete signed
 * band is packed into brick layers. This deliberately rounds each direction
 * independently: a nine-cell band needs three brick layers, not 9/4 layers.
 */
export function planOctreeSignedSurfaceBricks(
  dimensionsValue: OctreeSurfaceVec3,
  interfaceAreaCellsValue: number,
  options: OctreeSignedSurfaceBrickOptions = {},
): OctreeSignedSurfaceBrickPlan {
  const dimensions = dimensions3(dimensionsValue);
  const interfaceAreaCells = positiveInteger(interfaceAreaCellsValue, "Interface area");
  const brickSize = options.brickSize ?? 4;
  if (brickSize !== 4) throw new RangeError("Signed surface bricks currently require a brick size of four");
  const bandCells = options.bandCells ?? 4;
  if (!Number.isSafeInteger(bandCells) || bandCells < 1) throw new RangeError("Surface band must be a positive integer");
  const capacityHeadroom = options.capacityHeadroom ?? 1.25;
  if (!Number.isFinite(capacityHeadroom) || capacityHeadroom < 1) throw new RangeError("Surface capacity headroom must be at least one");
  const scalarFieldCount = options.scalarFieldCount ?? 3;
  positiveInteger(scalarFieldCount, "Surface scalar field count");
  const bandThicknessCells = 2 * bandCells + 1;
  const crossSectionBricks = Math.ceil(interfaceAreaCells / (brickSize * brickSize));
  const bandBrickLayers = Math.ceil(bandThicknessCells / brickSize);
  const unpaddedBrickCount = crossSectionBricks * bandBrickLayers;
  const brickCapacity = Math.ceil(unpaddedBrickCount * capacityHeadroom);
  const inactiveTileCapacity = options.inactiveTileCapacity ?? brickCapacity;
  if (!Number.isSafeInteger(inactiveTileCapacity) || inactiveTileCapacity < 0) {
    throw new RangeError("Inactive sign-tile capacity must be a non-negative integer");
  }
  const samplesPerBrick = brickSize ** 3 as 64;
  const keyBytes = brickCapacity * 4;
  const stateBytes = brickCapacity * 4;
  const activeListBytes = brickCapacity * 4;
  // { packed origin, packed size exponent and sign }.
  const inactiveTileBytes = inactiveTileCapacity * 8;
  const scalarBytes = brickCapacity * samplesPerBrick * scalarFieldCount * 4;
  const controlBytes = 256;
  const allocatedBytes = keyBytes + stateBytes + activeListBytes + inactiveTileBytes + scalarBytes + controlBytes;
  const denseEquivalentBytes = dimensions[0] * dimensions[1] * dimensions[2] * scalarFieldCount * 4;
  return {
    dimensions,
    logicalBrickDimensions: dimensions.map((component) => Math.ceil(component / brickSize)) as unknown as OctreeSurfaceVec3,
    brickSize,
    samplesPerBrick,
    bandCells,
    bandThicknessCells,
    interfaceAreaCells,
    unpaddedBrickCount,
    brickCapacity,
    inactiveTileCapacity,
    scalarFieldCount,
    keyBytes,
    stateBytes,
    activeListBytes,
    inactiveTileBytes,
    scalarBytes,
    controlBytes,
    allocatedBytes,
    denseEquivalentBytes,
    savedBytes: denseEquivalentBytes - allocatedBytes,
  };
}

/** Zero is reserved for an empty slot in future GPU adapters. */
export function encodeOctreeSignedSurfaceBrickKey(coord: OctreeSurfaceVec3): number {
  if (coord.some((component) => !Number.isSafeInteger(component) || component < 0 || component >= 1024)) {
    throw new RangeError("Signed surface brick coordinates must fit ten unsigned bits per axis");
  }
  return ((coord[0] | (coord[1] << 10) | (coord[2] << 20)) + 1) >>> 0;
}

export function decodeOctreeSignedSurfaceBrickKey(keyValue: number): OctreeSurfaceVec3 {
  const key = keyValue >>> 0;
  if (key === 0 || key > 0x4000_0000) throw new RangeError("Signed surface brick key is invalid");
  const packed = key - 1;
  return [packed & 1023, (packed >>> 10) & 1023, (packed >>> 20) & 1023];
}

export interface OctreeInactiveSignTile {
  readonly origin: OctreeSurfaceVec3;
  readonly size: number;
  readonly sign: OctreeSurfaceSign;
}

export interface OctreeSignedSurfaceIndex {
  readonly dimensions: OctreeSurfaceVec3;
  readonly brickSize: 4;
  readonly brickKeys: Uint32Array;
  readonly inactiveTiles: readonly OctreeInactiveSignTile[];
  readonly backgroundSign: OctreeSurfaceSign;
}

function contains(tile: OctreeInactiveSignTile, cell: OctreeSurfaceVec3): boolean {
  return cell.every((component, axis) => component >= tile.origin[axis] && component < tile.origin[axis] + tile.size);
}

function tileContainsTile(outer: OctreeInactiveSignTile, inner: OctreeInactiveSignTile): boolean {
  return inner.origin.every((component, axis) => component >= outer.origin[axis]
    && component + inner.size <= outer.origin[axis] + outer.size);
}

function validateTile(tile: OctreeInactiveSignTile, dimensions: OctreeSurfaceVec3): OctreeInactiveSignTile {
  if (!Number.isSafeInteger(tile.size) || tile.size < 1 || (tile.size & (tile.size - 1)) !== 0) {
    throw new RangeError("Inactive sign-tile size must be a positive power of two");
  }
  if (tile.sign !== -1 && tile.sign !== 1) throw new RangeError("Inactive sign-tile sign must be -1 or 1");
  tile.origin.forEach((component, axis) => {
    if (!Number.isSafeInteger(component) || component < 0 || component % tile.size !== 0
      || component + tile.size > dimensions[axis]) {
      throw new RangeError("Inactive sign tile must be aligned and contained by the domain");
    }
  });
  return { origin: [...tile.origin] as unknown as OctreeSurfaceVec3, size: tile.size, sign: tile.sign };
}

/** Build a deterministic, collision-free sparse index. */
export function createOctreeSignedSurfaceIndex(
  dimensionsValue: OctreeSurfaceVec3,
  activeBrickCoords: readonly OctreeSurfaceVec3[],
  inactiveTileValues: readonly OctreeInactiveSignTile[],
  backgroundSign: OctreeSurfaceSign = 1,
): OctreeSignedSurfaceIndex {
  const dimensions = dimensions3(dimensionsValue);
  if (backgroundSign !== -1 && backgroundSign !== 1) throw new RangeError("Surface background sign must be -1 or 1");
  const logical = dimensions.map((component) => Math.ceil(component / 4));
  const keys = [...new Set(activeBrickCoords.map((coord) => {
    coord.forEach((component, axis) => {
      if (!Number.isSafeInteger(component) || component < 0 || component >= logical[axis]) {
        throw new RangeError("Active surface brick lies outside the logical brick domain");
      }
    });
    return encodeOctreeSignedSurfaceBrickKey(coord);
  }))].sort((a, b) => a - b);
  const tiles = inactiveTileValues.map((tile) => validateTile(tile, dimensions)).sort((a, b) =>
    a.size - b.size || encodeOctreeSignedSurfaceBrickKey(a.origin) - encodeOctreeSignedSurfaceBrickKey(b.origin));
  for (let a = 0; a < tiles.length; a += 1) {
    for (let b = a + 1; b < tiles.length; b += 1) {
      const overlap = tiles[a].origin.every((component, axis) => component < tiles[b].origin[axis] + tiles[b].size
        && tiles[b].origin[axis] < component + tiles[a].size);
      if (!overlap) continue;
      if (tiles[a].size === tiles[b].size && tiles[a].origin.every((value, axis) => value === tiles[b].origin[axis])) {
        throw new RangeError("Duplicate inactive sign tiles are ambiguous");
      }
      if (!tileContainsTile(tiles[a], tiles[b]) && !tileContainsTile(tiles[b], tiles[a])) {
        throw new RangeError("Inactive sign tiles may be disjoint or nested, but not partially overlapping");
      }
    }
  }
  return { dimensions, brickSize: 4, brickKeys: Uint32Array.from(keys), inactiveTiles: tiles, backgroundSign };
}

function binarySearch(keys: Uint32Array, key: number): number {
  let low = 0, high = keys.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (keys[middle] < key) low = middle + 1;
    else high = middle;
  }
  return low < keys.length && keys[low] === key ? low : -1;
}

export type OctreeSignedSurfaceLocation =
  | { readonly kind: "active"; readonly brickIndex: number; readonly localIndex: number }
  | { readonly kind: "inactive"; readonly sign: OctreeSurfaceSign; readonly tile?: OctreeInactiveSignTile }
  | { readonly kind: "outside"; readonly sign: 1 };

/** Resolve an exact finest cell without a dense owner/page table. */
export function locateOctreeSignedSurfaceCell(
  index: OctreeSignedSurfaceIndex,
  cell: OctreeSurfaceVec3,
): OctreeSignedSurfaceLocation {
  if (cell.some((component, axis) => !Number.isSafeInteger(component) || component < 0 || component >= index.dimensions[axis])) {
    return { kind: "outside", sign: 1 };
  }
  const brickCoord = cell.map((component) => Math.floor(component / index.brickSize)) as unknown as OctreeSurfaceVec3;
  const brickIndex = binarySearch(index.brickKeys, encodeOctreeSignedSurfaceBrickKey(brickCoord));
  if (brickIndex >= 0) {
    const local = cell.map((component) => component % index.brickSize);
    return { kind: "active", brickIndex, localIndex: local[0] + 4 * (local[1] + 4 * local[2]) };
  }
  const tile = index.inactiveTiles.find((candidate) => contains(candidate, cell));
  return tile ? { kind: "inactive", sign: tile.sign, tile } : { kind: "inactive", sign: index.backgroundSign };
}

export interface OctreeAnalyticSurfaceBandOptions {
  readonly brickSize?: 4;
  readonly bandCells?: number;
}

/** CPU oracle: select only bricks containing a finest-cell sample in |phi| <= band. */
export function selectAnalyticOctreeSignedSurfaceBricks(
  dimensionsValue: OctreeSurfaceVec3,
  signedDistanceCells: (point: readonly [number, number, number]) => number,
  options: OctreeAnalyticSurfaceBandOptions = {},
): readonly OctreeSurfaceVec3[] {
  const dimensions = dimensions3(dimensionsValue);
  if ((options.brickSize ?? 4) !== 4) throw new RangeError("Analytic signed surface selection requires four-cell bricks");
  const bandCells = options.bandCells ?? 4;
  if (!Number.isFinite(bandCells) || bandCells < 0) throw new RangeError("Analytic surface band must be finite and non-negative");
  const logical = dimensions.map((component) => Math.ceil(component / 4));
  const selected: OctreeSurfaceVec3[] = [];
  for (let bz = 0; bz < logical[2]; bz += 1) for (let by = 0; by < logical[1]; by += 1) for (let bx = 0; bx < logical[0]; bx += 1) {
    let active = false;
    for (let z = 4 * bz; z < Math.min(dimensions[2], 4 * bz + 4) && !active; z += 1) {
      for (let y = 4 * by; y < Math.min(dimensions[1], 4 * by + 4) && !active; y += 1) {
        for (let x = 4 * bx; x < Math.min(dimensions[0], 4 * bx + 4); x += 1) {
          const phi = signedDistanceCells([x + 0.5, y + 0.5, z + 0.5]);
          if (!Number.isFinite(phi)) throw new RangeError("Analytic signed distance must be finite");
          if (Math.abs(phi) <= bandCells) { active = true; break; }
        }
      }
    }
    if (active) selected.push([bx, by, bz]);
  }
  return selected;
}
