/**
 * The lattice the slice is *drawn* on.
 *
 * The solver runs uniformly; the picture must not. A brick carries a rung, and
 * everything the lab draws inside that brick — the cell grid, the volume, the
 * PLIC line, a probed value — is aggregated to it. That is the honest picture
 * of an adaptive solver: coarse blocks where the water is still, single fine
 * cells along the interface, and the two meeting across a 2:1 port.
 *
 * Aggregation runs off a summed-area table so a brick at rung 1 costs the same
 * to read as one at rung 8, and so the Youngs stencil can sample a block-sized
 * neighbourhood without walking it.
 */
import {
  type AdvanceSlice, SLICE_BRICK, SLICE_BX, SLICE_BY, SLICE_NX, SLICE_NY,
  SLICE_RUNGS, plicOffset, sliceCell, youngsNormal,
} from "./slice-solver";

export interface LatticeCell {
  /** Low corner in fine cells, and the edge length in fine cells. */
  readonly x0: number;
  readonly y0: number;
  readonly size: number;
  readonly brick: number;
  readonly volume: number;
  readonly capacity: number;
  readonly fill: number;
  readonly open: boolean;
}

export interface SliceLattice {
  readonly volumeSum: Float64Array;
  readonly capacitySum: Float64Array;
  cells: LatticeCell[];
}

export function createSliceLattice(): SliceLattice {
  const words = (SLICE_NX + 1) * (SLICE_NY + 1);
  return {
    volumeSum: new Float64Array(words),
    capacitySum: new Float64Array(words),
    cells: [],
  };
}

/** Rebuild both summed-area tables. O(cells), once per drawn frame. */
export function buildSliceSums(lattice: SliceLattice, s: AdvanceSlice): void {
  const stride = SLICE_NX + 1;
  const { volumeSum, capacitySum } = lattice;
  for (let y = 1; y <= SLICE_NY; y++) for (let x = 1; x <= SLICE_NX; x++) {
    const i = y * stride + x, cell = sliceCell(x - 1, y - 1);
    volumeSum[i] = s.V[cell] + volumeSum[i - 1] + volumeSum[i - stride]
      - volumeSum[i - stride - 1];
    capacitySum[i] = s.K[cell] + capacitySum[i - 1] + capacitySum[i - stride]
      - capacitySum[i - stride - 1];
  }
}

function boxSum(table: Float64Array, x0: number, y0: number, size: number): number {
  const stride = SLICE_NX + 1;
  const x1 = Math.min(SLICE_NX, x0 + size), y1 = Math.min(SLICE_NY, y0 + size);
  const lx = Math.max(0, x0), ly = Math.max(0, y0);
  if (x1 <= lx || y1 <= ly) return 0;
  return table[y1 * stride + x1] - table[ly * stride + x1]
    - table[y1 * stride + lx] + table[ly * stride + lx];
}

export const latticeVolume = (l: SliceLattice, x: number, y: number, size: number): number =>
  boxSum(l.volumeSum, x, y, size);
export const latticeCapacity = (l: SliceLattice, x: number, y: number, size: number): number =>
  boxSum(l.capacitySum, x, y, size);

/** Fill fraction of one block, or -1 where the block is entirely solid. */
export function latticeFill(
  l: SliceLattice, x: number, y: number, size: number,
): number {
  const capacity = boxSum(l.capacitySum, x, y, size);
  return capacity <= 1e-6 ? -1 : boxSum(l.volumeSum, x, y, size) / capacity;
}

/** Every drawn cell, in brick-major order, at each brick's own rung. */
export function buildSliceLattice(lattice: SliceLattice, s: AdvanceSlice): void {
  const cells: LatticeCell[] = [];
  for (let by = 0; by < SLICE_BY; by++) for (let bx = 0; bx < SLICE_BX; bx++) {
    const brick = by * SLICE_BX + bx;
    const across = SLICE_RUNGS[Math.max(0, s.rung[brick])];
    const size = SLICE_BRICK / across;
    for (let j = 0; j < across; j++) for (let i = 0; i < across; i++) {
      const x0 = bx * SLICE_BRICK + i * size, y0 = by * SLICE_BRICK + j * size;
      const capacity = boxSum(lattice.capacitySum, x0, y0, size);
      const volume = boxSum(lattice.volumeSum, x0, y0, size);
      cells.push({
        x0, y0, size, brick, volume, capacity,
        fill: capacity <= 1e-6 ? 0 : volume / capacity,
        open: capacity > 1e-6,
      });
    }
  }
  lattice.cells = cells;
}

/** The lattice cell containing a point, at whatever rung its brick carries. */
export function latticeCellAt(
  lattice: SliceLattice, s: AdvanceSlice, x: number, y: number,
): LatticeCell | null {
  if (x < 0 || y < 0 || x >= SLICE_NX || y >= SLICE_NY) return null;
  const bx = Math.min(SLICE_BX - 1, (x / SLICE_BRICK) | 0);
  const by = Math.min(SLICE_BY - 1, (y / SLICE_BRICK) | 0);
  const brick = by * SLICE_BX + bx;
  const size = SLICE_BRICK / SLICE_RUNGS[Math.max(0, s.rung[brick])];
  const x0 = bx * SLICE_BRICK + Math.floor((x - bx * SLICE_BRICK) / size) * size;
  const y0 = by * SLICE_BRICK + Math.floor((y - by * SLICE_BRICK) / size) * size;
  const capacity = boxSum(lattice.capacitySum, x0, y0, size);
  const volume = boxSum(lattice.volumeSum, x0, y0, size);
  return {
    x0, y0, size, brick, volume, capacity,
    fill: capacity <= 1e-6 ? 0 : volume / capacity,
    open: capacity > 1e-6,
  };
}

export interface LatticePlane {
  readonly nx: number;
  readonly ny: number;
  readonly offset: number;
}

/**
 * The PLIC plane of a drawn cell.
 *
 * The Youngs stencil samples blocks of this cell's own size rather than the
 * fine field, so the reconstruction is continuous across a brick boundary
 * where the two sides carry different rungs — a line that changed slope at
 * every port would read as the artefact it is not.
 */
export function latticePlane(
  lattice: SliceLattice, cell: LatticeCell,
): LatticePlane | null {
  const fill = cell.fill;
  if (fill <= 1e-3 || fill >= 1 - 1e-3) return null;
  const normal = youngsNormal((dx, dy) => {
    const value = latticeFill(
      lattice, cell.x0 + dx * cell.size, cell.y0 + dy * cell.size, cell.size);
    return value < 0 ? (fill > 0.5 ? 1 : 0) : value;
  });
  if (!normal) return null;
  return {
    nx: normal.nx, ny: normal.ny,
    offset: plicOffset(fill, normal.nx, normal.ny),
  };
}
