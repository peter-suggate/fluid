import { integrateRetainedSceneDensity, type RetainedSceneDensity } from "./sparse-cm12-retained-scene-density";

/** Compile native volume averages from the retained physical field. Topology
 * records contain geometry, never point-density samples. The memo is local to
 * an immutable field generation and shared by dormant native rung templates. */
export function compileRetainedNativeIntegrals(field: RetainedSceneDensity,
  topology: Uint32Array, cellCount: number, h: number,
  fine?: { means: Float32Array; dimensions: readonly [number, number, number] }): Float32Array {
  if (!Number.isSafeInteger(cellCount) || cellCount < 0 || !(h > 0) || !Number.isFinite(h)
    || topology.length < 7) throw new Error("Invalid retained native integral catalogue");
  const records = new Float32Array(topology.buffer, topology.byteOffset, topology.length);
  const offset = topology[6]!;
  if (offset < 7 || offset + 8 * cellCount > topology.length) throw new Error("Truncated retained native geometry records");
  if (fine && (fine.dimensions.some(n => !Number.isSafeInteger(n) || n < 1)
    || fine.means.length !== fine.dimensions[0] * fine.dimensions[1] * fine.dimensions[2])) {
    throw new Error("Retained finest moment snapshot does not match its lattice");
  }
  const means = new Float32Array(cellCount);
  const cache = new Map<string, number>();
  for (let cell = 0; cell < cellCount; cell++) {
    const base = offset + 8 * cell;
    const loFine = [0, 1, 2].map(axis => records[base + axis]! - .5 * records[base + 4 + axis]!);
    const hiFine = [0, 1, 2].map(axis => records[base + axis]! + .5 * records[base + 4 + axis]!);
    if (loFine.some((lo, axis) => !Number.isFinite(lo) || !Number.isFinite(hiFine[axis]) || !(hiFine[axis] > lo))) {
      throw new Error(`Invalid retained native geometry at cell ${cell}`);
    }
    if (fine) {
      // A finest-cell moment only restricts to unions of complete support
      // cells. Rounding fractional bounds silently turns a partial integral
      // into a different box (or a zero-volume NaN). Call the physical-field
      // path below when a caller actually needs nonconforming query boxes.
      if (![...loFine, ...hiFine].every(Number.isInteger)) {
        throw new Error(`Retained native cell ${cell} is not aligned to the finest moment lattice`);
      }
      const lo = loFine, hi = hiFine;
      const [nx, ny, nz] = fine.dimensions;
      let amount = 0;
      for (let z = Math.max(0, lo[2]!); z < Math.min(nz, hi[2]!); z++)
        for (let y = Math.max(0, lo[1]!); y < Math.min(ny, hi[1]!); y++)
          for (let x = Math.max(0, lo[0]!); x < Math.min(nx, hi[0]!); x++)
            amount += fine.means[x + nx * (y + ny * z)]!;
      means[cell] = amount / ((hi[0]! - lo[0]!) * (hi[1]! - lo[1]!) * (hi[2]! - lo[2]!));
      continue;
    }
    const lower = [0, 1, 2].map(axis => field.domain.lower[axis]!
      + h * (records[base + axis]! - .5 * records[base + 4 + axis]!)) as [number, number, number];
    const upper = [0, 1, 2].map(axis => field.domain.lower[axis]!
      + h * (records[base + axis]! + .5 * records[base + 4 + axis]!)) as [number, number, number];
    const key = `${lower.join(",")}/${upper.join(",")}`;
    let mean = cache.get(key);
    if (mean === undefined) {
      const receipt = integrateRetainedSceneDensity(field, { lower, upper });
      if (!receipt.toleranceMet) throw new Error(`Retained density integral did not converge for native cell ${cell}: ${receipt.estimatedAbsoluteError} > ${receipt.requestedAbsoluteTolerance}`);
      mean = receipt.mean;
      cache.set(key, mean);
    }
    means[cell] = mean;
  }
  return means;
}
