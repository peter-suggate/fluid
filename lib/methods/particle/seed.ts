import { seedLiquidParticles } from "../../core/seed-liquid-particles";

/** APIC adds three affine rows to the shared material particle record. */
export function seedApicParticles(phi: Float32Array, solid: Uint32Array,
  dimensions: readonly [number, number, number], h: readonly [number, number, number], capacity: number) {
  return seedLiquidParticles(phi, solid, dimensions, h, capacity, 20, "APIC");
}
