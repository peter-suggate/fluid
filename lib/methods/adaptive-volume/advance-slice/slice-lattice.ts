/**
 * Readback view of the accepted two-dimensional sparse authority.
 *
 * Every entry is one compact numerical cell. The lattice does not invent a
 * display rung: coarse cells, partial edge cells and mixed-rung seams are the
 * same records consumed by pressure and geometric transport.
 */
import { type AdvanceSlice, sliceCell } from "./slice-solver";

/**
 * A reconstructed interface, ready to clip against the unit square.
 *
 * `nx, ny` point out of the liquid in the canvas frame (y down) and `offset` is
 * measured from the cell's low corner, not its centre — the conversion happens
 * once, in `buildSliceLattice`, so no consumer has to remember it.
 */
export interface LatticePlane {
  readonly nx: number;
  readonly ny: number;
  readonly offset: number;
}

export interface LatticeCell {
  /** Bounds in the canvas convention: x right, y down, finest-cell units. */
  readonly x0: number;
  readonly y0: number;
  readonly width: number;
  readonly height: number;
  /** Compatibility alias for square, non-edge cells. */
  readonly size: number;
  readonly brick: number;
  readonly topologyCell: number;
  /** Extensive unit-depth quantities, in finest-cell squared units. */
  readonly volume: number;
  readonly capacity: number;
  readonly fill: number;
  readonly open: boolean;
  readonly plane: LatticePlane | null;
}

export interface SliceLattice {
  nx: number;
  ny: number;
  volumeSum: Float64Array;
  capacitySum: Float64Array;
  cells: LatticeCell[];
}

export function createSliceLattice(s?: Pick<AdvanceSlice, "nx" | "ny">): SliceLattice {
  const nx = s?.nx ?? 0, ny = s?.ny ?? 0;
  return {
    nx, ny,
    volumeSum: new Float64Array((nx + 1) * (ny + 1)),
    capacitySum: new Float64Array((nx + 1) * (ny + 1)),
    cells: [],
  };
}

function resize(lattice: SliceLattice, s: Pick<AdvanceSlice, "nx" | "ny">): void {
  if (lattice.nx === s.nx && lattice.ny === s.ny) return;
  lattice.nx = s.nx;
  lattice.ny = s.ny;
  lattice.volumeSum = new Float64Array((s.nx + 1) * (s.ny + 1));
  lattice.capacitySum = new Float64Array((s.nx + 1) * (s.ny + 1));
  lattice.cells = [];
}

/** Dense summed-area readback used by lenses which inspect finest pixels. */
export function buildSliceSums(lattice: SliceLattice, s: AdvanceSlice): void {
  resize(lattice, s);
  const stride = s.nx + 1;
  lattice.volumeSum.fill(0);
  lattice.capacitySum.fill(0);
  for (let y = 1; y <= s.ny; y += 1) for (let x = 1; x <= s.nx; x += 1) {
    const at = y * stride + x, dense = sliceCell(s, x - 1, y - 1);
    lattice.volumeSum[at] = s.V[dense]! + lattice.volumeSum[at - 1]!
      + lattice.volumeSum[at - stride]! - lattice.volumeSum[at - stride - 1]!;
    lattice.capacitySum[at] = s.K[dense]! + lattice.capacitySum[at - 1]!
      + lattice.capacitySum[at - stride]! - lattice.capacitySum[at - stride - 1]!;
  }
}

function boxSum(table: Float64Array, lattice: SliceLattice,
  x0: number, y0: number, width: number, height = width): number {
  const stride = lattice.nx + 1;
  const x1 = Math.min(lattice.nx, x0 + width), y1 = Math.min(lattice.ny, y0 + height);
  const lx = Math.max(0, x0), ly = Math.max(0, y0);
  if (x1 <= lx || y1 <= ly) return 0;
  return table[y1 * stride + x1]! - table[ly * stride + x1]!
    - table[y1 * stride + lx]! + table[ly * stride + lx]!;
}

export const latticeVolume = (l: SliceLattice, x: number, y: number,
  width: number, height = width): number => boxSum(l.volumeSum, l, x, y, width, height);
export const latticeCapacity = (l: SliceLattice, x: number, y: number,
  width: number, height = width): number => boxSum(l.capacitySum, l, x, y, width, height);
export function latticeFill(l: SliceLattice, x: number, y: number,
  width: number, height = width): number {
  const capacity = latticeCapacity(l, x, y, width, height);
  return capacity <= 1e-6 ? -1 : latticeVolume(l, x, y, width, height) / capacity;
}

/** Rebuild from compact accepted cells, in production brick/cell order. */
export function buildSliceLattice(lattice: SliceLattice, s: AdvanceSlice): void {
  resize(lattice, s);
  const topology = s.topology.accepted, fields = s.fields;
  lattice.cells = topology.cells.map(cell => {
    const width = cell.widthsFine[0], height = cell.widthsFine[1];
    const y0 = s.ny - cell.maximumFine[1];
    const area = cell.volumeFineCells;
    const capacity = Math.fround(fields.capacity[cell.id]! * area);
    const volume = Math.fround(fields.density[cell.id]! * area);
    const fill = capacity > 1e-8 ? Math.fround(volume / capacity) : 0;
    const px = fields.interfaceNormal[2 * cell.id]!;
    const py = fields.interfaceNormal[2 * cell.id + 1]!;
    /* Into the drawing's frame, both axes at once.
     *
     * The published record is written about the cell *centre*, with y up; a
     * drawn cell is the unit square with y down. Reflecting y is just negating
     * ny — a reflection about the centre leaves a centre-based offset alone —
     * and moving the origin from the centre to the corner is the half-normal
     * shift the solver's own `sliceLiquidPolygon` applies for the same reason.
     * Skipping it is not a small error: it misplaces the interface by up to
     * half a cell, which reads as a surface drawn on the wrong side of its own
     * row. `offset` below is therefore always ready for `clipUnitSquare`. */
    const ny = -py;
    const plane = px === 0 && py === 0 ? null : {
      nx: px, ny, offset: fields.interfaceOffset[cell.id]! + 0.5 * (px + ny),
    };
    return Object.freeze({
      x0: cell.minimumFine[0], y0, width, height, size: width,
      brick: cell.brickKey, topologyCell: cell.id,
      volume, capacity, fill, open: capacity > 1e-8, plane,
    });
  });
}

/** Accepted compact cell containing a canvas-space point. */
export function latticeCellAt(lattice: SliceLattice, s: AdvanceSlice,
  x: number, y: number): LatticeCell | null {
  if (x < 0 || y < 0 || x >= s.nx || y >= s.ny) return null;
  return lattice.cells.find(cell => x >= cell.x0 && x < cell.x0 + cell.width
    && y >= cell.y0 && y < cell.y0 + cell.height) ?? null;
}

/** PLIC record produced by the numerical reconstruction stage. */
export function latticePlane(_lattice: SliceLattice, cell: LatticeCell): LatticePlane | null {
  return cell.fill > 1e-3 && cell.fill < 1 - 1e-3 ? cell.plane : null;
}
