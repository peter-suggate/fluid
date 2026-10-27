import { particleSlabSpill } from "../lib/methods/uniform/uniform-particle-slab.wgsl";

/** Receipt words through the current spill count, followed by the particle bank. */
export const PARTICLE_MASS_HEADER_WORDS = 22;
export const particleMassReadbackBytes = (capacity: number, strideFloats: number) =>
  4 * (PARTICLE_MASS_HEADER_WORDS + capacity * strideFloats);

/** Decode only the current slab extent and spill; older tail records can remain live in memory. */
export function* particleMassRecords(raw: ArrayBuffer, capacity: number, strideFloats: number, merged: boolean) {
  const f = new Float32Array(raw), u = new Uint32Array(raw);
  const extent = Math.min(u[0]!, capacity - particleSlabSpill(capacity));
  const count = Math.min(capacity, extent + u[21]!);
  for (let i = 0; i < count; i++) {
    const at = PARTICLE_MASS_HEADER_WORDS + i * strideFloats;
    if (!(f[at + 3]! > 0)) continue;
    yield { x: f[at]!, y: f[at + 1]!, z: f[at + 2]!, depth: f[at + 7]!,
      mass: !merged ? 1 : strideFloats === 28 ? f[at + 25]! + 1 : (u[at + 11]! & 7) + 1 };
  }
}
