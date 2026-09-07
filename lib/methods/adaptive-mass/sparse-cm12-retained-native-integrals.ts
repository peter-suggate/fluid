import { integrateRetainedSceneDensity, type RetainedSceneDensity } from "./sparse-cm12-retained-scene-density";

/** Compile native volume averages from the retained physical field. Topology
 * records contain geometry, never point-density samples. The memo is local to
 * an immutable field generation and shared by dormant native rung templates. */
export function compileRetainedNativeIntegrals(field: RetainedSceneDensity,
  topology: Uint32Array, cellCount: number, h: number): Float32Array {
  const records = new Float32Array(topology.buffer, topology.byteOffset, topology.length);
  const offset = topology[6]!;
  const means = new Float32Array(cellCount);
  const cache = new Map<string, number>();
  for (let cell = 0; cell < cellCount; cell++) {
    const base = offset + 8 * cell;
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
