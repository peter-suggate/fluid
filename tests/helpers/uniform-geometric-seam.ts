import type { UniformMixedLayout } from "../../lib/methods/uniform/uniform-mixed-layout";
import { createUniformMixedLayout, MIXED_CELL_MASK, mixedCellWidth } from "../../lib/methods/uniform/uniform-mixed-layout";

export type Triple = readonly [number, number, number];
export interface CellBox { readonly min: Triple; readonly width: number; readonly capacity: number }
export interface TransportRows {
  readonly cells: readonly CellBox[];
  readonly offsets: Uint32Array;
  readonly donors: Uint32Array;
  readonly weights: Float32Array;
}

/** Small numerical oracle, deliberately not a runtime topology builder. Coordinates
 * are finest-cell units; extensive volumes use the same units cubed. A translated
 * receiver box reduces exactly to trilinear weights on either uniform endpoint.
 * The quadratic overlap search is useful for independence, not GPU scalability.
 */
export function geometricSeamRows(layout: UniformMixedLayout, displacement: (center: Triple) => Triple): TransportRows {
  const cells: CellBox[] = new Array(layout.cellCount);
  const [nx, ny] = layout.tileDimensions;
  layout.tiles.forEach((word, tile) => {
    const width = mixedCellWidth(word);
    const base = word & MIXED_CELL_MASK;
    const corner = [tile % nx * 4, Math.floor(tile / nx) % ny * 4, Math.floor(tile / (nx * ny)) * 4];
    for (let z = 0, i = base; z < 4; z += width) for (let y = 0; y < 4; y += width) for (let x = 0; x < 4; x += width, i++) {
      cells[i] = { min: [corner[0]! + x, corner[1]! + y, corner[2]! + z], width, capacity: width ** 3 };
    }
  });
  const offsets = [0], donors: number[] = [], weights: number[] = [];
  for (const [i, receiver] of cells.entries()) {
    const delta = displacement(receiver.min.map(x => x + receiver.width / 2) as unknown as Triple);
    if (delta.some(x => !Number.isFinite(x))) throw new Error("Nonfinite departure");
    const departure = receiver.min.map((x, axis) => x - delta[axis]!);
    cells.forEach((donor, j) => {
      let overlap = 1;
      for (let a = 0; a < 3; a++) overlap *= Math.max(0,
        Math.min(departure[a]! + receiver.width, donor.min[a]! + donor.width) - Math.max(departure[a]!, donor.min[a]!));
      if (overlap > 0) { donors.push(j); weights.push(overlap); }
    });
    // Dedicated fallback slot, even when the self donor already occurs above.
    donors.push(i); weights.push(0); offsets.push(donors.length);
  }
  const sampled = new Uint8Array(cells.length);
  donors.forEach((j, e) => { if (weights[e]! > 0) sampled[j] = 1; });
  cells.forEach((cell, i) => { if (!sampled[i]) weights[offsets[i + 1]! - 1] = cell.capacity; });
  return { cells, offsets: Uint32Array.from(offsets), donors: Uint32Array.from(donors), weights: Float32Array.from(weights) };
}

/** Native algorithm: three row/column rounds, followed by a donor-conservative
 * gather. Rows target receiver capacity; columns target donor capacity. The last
 * division converts geometric volume weights to fractions of extensive mass.
 */
export function transportSeamReference(rows: TransportRows, volumes: ArrayLike<number>): Float64Array {
  const { cells, offsets, donors } = rows;
  const weights = Float64Array.from(rows.weights);
  if (volumes.length !== cells.length) throw new Error("Volume count mismatch");
  for (let round = 0; round < 3; round++) {
    const sums = new Float64Array(cells.length);
    cells.forEach((cell, i) => {
      let sum = 0;
      for (let e = offsets[i]!; e < offsets[i + 1]!; e++) sum += weights[e]!;
      const scale = cell.capacity / Math.max(sum, 1e-20);
      for (let e = offsets[i]!; e < offsets[i + 1]!; e++) {
        weights[e] = weights[e]! * scale;
        sums[donors[e]!] += weights[e]!;
      }
    });
    for (let e = 0; e < weights.length; e++) weights[e] = weights[e]! * cells[donors[e]!]!.capacity / Math.max(sums[donors[e]!]!, 1e-20);
  }
  return Float64Array.from(cells, (_, i) => {
    let value = 0;
    for (let e = offsets[i]!; e < offsets[i + 1]!; e++) value += weights[e]! * volumes[donors[e]!]! / cells[donors[e]!]!.capacity;
    return value;
  });
}

export function seamLayout(axis: number, mode: "mixed" | "fine" | "coarse" = "mixed") {
  const max = [8, 8, 8]; if (mode === "mixed") max[axis] = 4;
  return createUniformMixedLayout({ dimensions: [8, 8, 8], cellSize_m: [1, 2, .5], origin_m: { x: 0, y: 0, z: 0 } }, mode === "coarse" ? [] : [{
    id: "fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
    min_m: { x: 0, y: 0, z: 0 }, max_m: { x: max[0]!, y: max[1]! * 2, z: max[2]! * .5 },
  }], false, 4);
}

/** Fixed-pattern alternative to the independent intersection oracle above.
 * Owner lookup uses an aligned fragment centre. A row chooses the coarsest
 * sampling grid that resolves EVERY intersected donor, including long traces.
 * Its reserved size depends only on receiver width, never on the trace.
 */
export function fixedGeometricSeamRows(layout: UniformMixedLayout, displacement: (center: Triple) => Triple): TransportRows {
  const { cells } = geometricSeamRows(layout, () => [0, 0, 0]);
  const offsets = [0], donors: number[] = [], weights: number[] = [];
  for (const [i, receiver] of cells.entries()) {
    const delta = displacement(receiver.min.map(x => x + receiver.width / 2) as unknown as Triple);
    if (delta.some(x => !Number.isFinite(x))) throw new Error("Nonfinite departure");
    const departure = receiver.min.map((x, a) => x - delta[a]!);
    let grain = receiver.width;
    for (const donor of cells) if (donor.min.every((x, a) => x < departure[a]! + receiver.width && x + donor.width > departure[a]!)) grain = Math.min(grain, donor.width);
    const ratio = receiver.width / grain, base = departure.map(x => Math.floor(x / grain) * grain);
    const slots = (receiver.width + 1) ** 3;
    const rowDonors = new Array<number>(slots + 1).fill(i), rowWeights = new Array<number>(slots + 1).fill(0);
    for (let z = 0, k = 0; z <= ratio; z++) for (let y = 0; y <= ratio; y++) for (let x = 0; x <= ratio; x++, k++) {
      const corner = [base[0]! + x * grain, base[1]! + y * grain, base[2]! + z * grain];
      let weight = 1;
      for (let a = 0; a < 3; a++) weight *= Math.max(0, Math.min(corner[a]! + grain, departure[a]! + receiver.width) - Math.max(corner[a]!, departure[a]!));
      if (!weight) continue;
      const donor = cells.findIndex(c => corner.every((v, a) => v + grain / 2 >= c.min[a]! && v + grain / 2 < c.min[a]! + c.width));
      if (donor < 0) continue;
      rowDonors[k] = donor; rowWeights[k] = weight;
    }
    donors.push(...rowDonors); weights.push(...rowWeights); offsets.push(donors.length);
  }
  const sampled = new Uint8Array(cells.length);
  donors.forEach((d, e) => { if (weights[e]! > 0) sampled[d] = 1; });
  cells.forEach((c, i) => { if (!sampled[i]) weights[offsets[i + 1]! - 1] = c.capacity; });
  return { cells, offsets: Uint32Array.from(offsets), donors: Uint32Array.from(donors), weights: Float32Array.from(weights) };
}

export function gradedSeamLayout(axis: number) {
  const dimensions = [8, 8, 8]; dimensions[axis] = 16;
  const max = [...dimensions]; max[axis] = 4;
  return createUniformMixedLayout({ dimensions: dimensions as unknown as Triple, cellSize_m: [1, 2, .5], origin_m: { x: 0, y: 0, z: 0 } }, [{
    id: "fine", rule: "minimum-cell-size", minimumCellSize_cells: 1, maximumCellSize_cells: 1,
    min_m: { x: 0, y: 0, z: 0 }, max_m: { x: max[0]!, y: max[1]! * 2, z: max[2]! * .5 },
  }], true, 4);
}
