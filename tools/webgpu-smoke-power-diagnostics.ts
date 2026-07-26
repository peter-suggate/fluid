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
 * Conservation diagnostic for the compact structured path.
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
  readonly maximumLiquidComponentCfl: number;
}

/** Score only represented liquid cells. The compact row reconstruction uses
 * an all-NaN vector as its explicit unrepresented-cell sentinel; partially
 * wet air-centre cells in the exact volume field therefore do not masquerade
 * as non-finite accepted row velocities. */
export function compactLiquidVelocityDiagnostic(
  velocity: ArrayLike<number>,
  volume: ArrayLike<number>,
  cellVolume_m3: number,
  componentCellWidths_m: readonly [number, number, number],
  dt_s: number,
): CompactLiquidVelocityDiagnostic {
  if (velocity.length !== volume.length * 3 || !(cellVolume_m3 > 0) || !Number.isFinite(cellVolume_m3)
    || !componentCellWidths_m.every((width) => Number.isFinite(width) && width > 0)
    || !Number.isFinite(dt_s) || dt_s < 0) {
    throw new RangeError("Compact velocity and occupancy dimensions are inconsistent");
  }
  let kineticEnergyProxy = 0, liquidCellCount = 0, finiteLiquidCellCount = 0;
  let liquidVolumeCellSum = 0, finiteLiquidVolumeCellSum = 0;
  let nonFiniteLiquidComponentCount = 0, maximumLiquidComponentSpeed_m_s = 0;
  let maximumLiquidComponentCfl = 0;
  for (let cell = 0; cell < volume.length; cell += 1) {
    const alpha = Math.max(0, Math.min(1, Number(volume[cell])));
    if (!(alpha > 1e-4)) continue;
    const components = [Number(velocity[3 * cell]), Number(velocity[3 * cell + 1]),
      Number(velocity[3 * cell + 2])] as const;
    if (components.every(Number.isNaN)) continue;
    liquidCellCount += 1; liquidVolumeCellSum += alpha;
    let speedSquared = 0, finiteCell = true;
    for (let axis = 0; axis < 3; axis += 1) {
      const value = components[axis]!;
      if (!Number.isFinite(value)) { nonFiniteLiquidComponentCount += 1; finiteCell = false; continue; }
      speedSquared += value * value;
      maximumLiquidComponentSpeed_m_s = Math.max(maximumLiquidComponentSpeed_m_s, Math.abs(value));
      maximumLiquidComponentCfl = Math.max(maximumLiquidComponentCfl,
        Math.abs(value) * dt_s / componentCellWidths_m[axis]!);
    }
    if (!finiteCell) continue;
    finiteLiquidCellCount += 1; finiteLiquidVolumeCellSum += alpha;
    kineticEnergyProxy += 0.5 * alpha * speedSquared * cellVolume_m3;
  }
  return { kineticEnergyProxy, liquidCellCount, finiteLiquidCellCount,
    liquidVolumeCellSum, finiteLiquidVolumeCellSum,
    nonFiniteLiquidComponentCount, maximumLiquidComponentSpeed_m_s, maximumLiquidComponentCfl };
}
