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

export interface CompactMechanicalEnergyDiagnostic {
  readonly gravitationalPotentialEnergyProxy: number;
  readonly reconstructedKineticEnergyProxy: number;
  readonly mechanicalEnergyProxy: number;
  readonly potentialEnergyReleasedProxy: number;
  readonly mechanicalEnergyLossProxy: number;
  readonly mechanicalEnergyRetentionRatio: number;
  readonly releasedPotentialToKineticRatio: number | null;
}

/**
 * Conservation diagnostic for the compact power path.
 *
 * Both inputs are volume-weighted specific-energy integrals (density is
 * omitted), so their sum and every ratio remain physically comparable for a
 * constant-density liquid. A null conversion ratio means gravity has not
 * released a resolvable amount of potential energy yet.
 */
export function compactMechanicalEnergyDiagnostic(
  initialGravitationalPotentialEnergyProxy: number,
  gravitationalPotentialEnergyProxy: number,
  reconstructedKineticEnergyProxy: number,
): CompactMechanicalEnergyDiagnostic {
  if (![initialGravitationalPotentialEnergyProxy, gravitationalPotentialEnergyProxy,
    reconstructedKineticEnergyProxy].every(Number.isFinite)
    || initialGravitationalPotentialEnergyProxy <= 0 || reconstructedKineticEnergyProxy < 0) {
    throw new RangeError("Compact mechanical-energy inputs must be finite, with positive initial potential and non-negative kinetic energy");
  }
  const mechanicalEnergyProxy = gravitationalPotentialEnergyProxy + reconstructedKineticEnergyProxy;
  const potentialEnergyReleasedProxy = initialGravitationalPotentialEnergyProxy - gravitationalPotentialEnergyProxy;
  return {
    gravitationalPotentialEnergyProxy,
    reconstructedKineticEnergyProxy,
    mechanicalEnergyProxy,
    potentialEnergyReleasedProxy,
    mechanicalEnergyLossProxy: initialGravitationalPotentialEnergyProxy - mechanicalEnergyProxy,
    mechanicalEnergyRetentionRatio: mechanicalEnergyProxy / initialGravitationalPotentialEnergyProxy,
    releasedPotentialToKineticRatio: potentialEnergyReleasedProxy > initialGravitationalPotentialEnergyProxy * 1e-12
      ? reconstructedKineticEnergyProxy / potentialEnergyReleasedProxy : null,
  };
}

export interface CompactLiquidVelocityDiagnostic {
  readonly kineticEnergyProxy: number;
  readonly liquidCellCount: number;
  readonly finiteLiquidCellCount: number;
  readonly liquidVolumeCellSum: number;
  readonly finiteLiquidVolumeCellSum: number;
  readonly nonFiniteLiquidComponentCount: number;
  readonly maximumLiquidComponentSpeed_m_s: number;
}

/** Score only represented liquid cells; uncovered air is intentionally NaN. */
export function compactLiquidVelocityDiagnostic(
  velocity: ArrayLike<number>,
  volume: ArrayLike<number>,
  cellVolume_m3: number,
): CompactLiquidVelocityDiagnostic {
  if (velocity.length !== volume.length * 3 || !(cellVolume_m3 > 0) || !Number.isFinite(cellVolume_m3)) {
    throw new RangeError("Compact velocity and occupancy dimensions are inconsistent");
  }
  let kineticEnergyProxy = 0, liquidCellCount = 0, finiteLiquidCellCount = 0;
  let liquidVolumeCellSum = 0, finiteLiquidVolumeCellSum = 0;
  let nonFiniteLiquidComponentCount = 0, maximumLiquidComponentSpeed_m_s = 0;
  for (let cell = 0; cell < volume.length; cell += 1) {
    const alpha = Math.max(0, Math.min(1, Number(volume[cell])));
    if (!(alpha > 1e-4)) continue;
    liquidCellCount += 1; liquidVolumeCellSum += alpha;
    let speedSquared = 0, finiteCell = true;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = Number(velocity[3 * cell + axis]);
      if (!Number.isFinite(value)) { nonFiniteLiquidComponentCount += 1; finiteCell = false; continue; }
      speedSquared += value * value;
      maximumLiquidComponentSpeed_m_s = Math.max(maximumLiquidComponentSpeed_m_s, Math.abs(value));
    }
    if (!finiteCell) continue;
    finiteLiquidCellCount += 1; finiteLiquidVolumeCellSum += alpha;
    kineticEnergyProxy += 0.5 * alpha * speedSquared * cellVolume_m3;
  }
  return { kineticEnergyProxy, liquidCellCount, finiteLiquidCellCount,
    liquidVolumeCellSum, finiteLiquidVolumeCellSum,
    nonFiniteLiquidComponentCount, maximumLiquidComponentSpeed_m_s };
}
