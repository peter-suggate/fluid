/**
 * Integrated flux represented by a compact power face.
 *
 * `normalVelocity` is already the aperture-weighted face unknown published by
 * the solid-face constraint pass.  The discrete divergence therefore applies
 * the geometric power-face area exactly once and must not multiply by the
 * stored aperture a second time.
 */
export function compactPowerFaceIntegratedFlux(area: number, normalVelocity: number): number {
  return area * normalVelocity;
}

/**
 * Quadratic velocity norm induced by the compact pressure operators.
 *
 * With D = B^T A and G = F L^-1 B, the diagonal metric H satisfying
 * H G = D^T is H = A L F^-1. `inverseDistance` stores L^-1. This is a
 * discrete projection-energy proxy, not a reconstructed cell kinetic energy.
 * Closed faces are constrained rather than projected and contribute zero.
 */
export function compactPowerFaceMetricKineticEnergy(
  area: number,
  inverseDistance: number,
  openFraction: number,
  normalVelocity: number,
): number {
  if (![area, inverseDistance, openFraction, normalVelocity].every(Number.isFinite)
    || area <= 0 || inverseDistance <= 0 || openFraction < 0 || openFraction > 1) return Number.NaN;
  if (openFraction === 0) return 0;
  return 0.5 * area / (openFraction * inverseDistance) * normalVelocity * normalVelocity;
}
