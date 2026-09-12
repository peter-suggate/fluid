/**
 * Fine-cell point location for the advance slice.
 *
 * An accepted generation tiles the domain with integer, half-open cell bounds,
 * so one fine-cell -> cell-id image answers every owner query with a single
 * array read instead of a scan over every cell. Presentation alone asked the
 * scan `fine cells * cells` times per published lattice, which is the whole
 * cost of a larger slice.
 *
 * The image is a pure restatement of the scan: it is built from the same cell
 * order, keeps the first covering cell exactly as `Array.prototype.find` did,
 * and answers -1 wherever no cell covers the point. Any topology the image
 * cannot represent — fractional or negative bounds, an implausible domain —
 * falls back to the scan, so the lookup is total for hand-built fixtures too.
 */

export interface SliceIndexedCell {
  readonly id: number;
  readonly minimum: readonly [number, number];
  readonly maximum: readonly [number, number];
}

export interface SliceIndexedTopology {
  readonly dimensions?: readonly [number, number];
  readonly cells: readonly SliceIndexedCell[];
}

/** A larger domain than any slice experiment, and still only 64 MiB. */
const MAXIMUM_IMAGE_CELLS = 1 << 24;

interface SliceOwnerImage {
  readonly owner: Int32Array;
  readonly nx: number;
  readonly ny: number;
}

/**
 * Keyed by the topology object. A generation transfer publishes a new object,
 * so a stale image cannot outlive the cell support it was built from.
 */
const images = new WeakMap<SliceIndexedTopology, SliceOwnerImage | null>();

function buildOwnerImage(topology: SliceIndexedTopology): SliceOwnerImage | null {
  let nx = topology.dimensions?.[0] ?? 0, ny = topology.dimensions?.[1] ?? 0;
  for (const cell of topology.cells) {
    if (!Number.isInteger(cell.minimum[0]) || !Number.isInteger(cell.minimum[1])
      || !Number.isInteger(cell.maximum[0]) || !Number.isInteger(cell.maximum[1])
      || cell.minimum[0] < 0 || cell.minimum[1] < 0) return null;
    nx = Math.max(nx, cell.maximum[0]);
    ny = Math.max(ny, cell.maximum[1]);
  }
  if (nx <= 0 || ny <= 0 || nx * ny > MAXIMUM_IMAGE_CELLS) return null;
  const owner = new Int32Array(nx * ny).fill(-1);
  for (const cell of topology.cells) {
    const yEnd = Math.min(ny, cell.maximum[1]), xEnd = Math.min(nx, cell.maximum[0]);
    for (let y = cell.minimum[1]; y < yEnd; y += 1) {
      const row = y * nx;
      // The scan returned the first covering cell in this order; overlapping
      // support, were a fixture ever to carry it, must resolve the same way.
      for (let x = cell.minimum[0]; x < xEnd; x += 1) {
        if (owner[row + x] === -1) owner[row + x] = cell.id;
      }
    }
  }
  return { owner, nx, ny };
}

function ownerImage(topology: SliceIndexedTopology): SliceOwnerImage | null {
  const cached = images.get(topology);
  if (cached !== undefined) return cached;
  const built = buildOwnerImage(topology);
  images.set(topology, built);
  return built;
}

function scanOwnerAt(topology: SliceIndexedTopology, x: number, y: number): number {
  for (const cell of topology.cells) {
    if (x >= cell.minimum[0] && x < cell.maximum[0]
      && y >= cell.minimum[1] && y < cell.maximum[1]) return cell.id;
  }
  return -1;
}

/** The cell owning the point, or -1. Identical to the scan it replaces. */
export function sliceCellOwnerAt(topology: SliceIndexedTopology,
  x: number, y: number): number {
  const image = ownerImage(topology);
  if (!image) return scanOwnerAt(topology, x, y);
  const ix = Math.floor(x), iy = Math.floor(y);
  if (ix < 0 || iy < 0 || ix >= image.nx || iy >= image.ny) return -1;
  return image.owner[iy * image.nx + ix]!;
}

/**
 * The same lookup with the image resolved once. A caller sweeping the whole
 * fine lattice — publishing a display plane, say — should not re-reach the
 * image for every cell it visits.
 */
export function sliceCellOwnerLookup(
  topology: SliceIndexedTopology): (x: number, y: number) => number {
  const image = ownerImage(topology);
  if (!image) return (x, y) => scanOwnerAt(topology, x, y);
  const { owner, nx, ny } = image;
  return (x, y) => {
    const ix = Math.floor(x), iy = Math.floor(y);
    return ix < 0 || iy < 0 || ix >= nx || iy >= ny ? -1 : owner[iy * nx + ix]!;
  };
}

/** The scan itself, so a differential can compare the two without a copy. */
export const sliceCellOwnerByScan = scanOwnerAt;
