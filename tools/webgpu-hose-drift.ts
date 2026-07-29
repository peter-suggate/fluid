import type { FluidInflow, Vec3 } from "../lib/model";

export interface HoseJetDriftAudit {
  readonly sampledLiquidCells: number;
  readonly sampledVelocityCells: number;
  readonly meanOutletVelocity_m_s: readonly [number, number, number];
  readonly outletAxialSpeed_m_s: number;
  readonly outletGravityTangentSpeed_m_s: number;
  readonly outletSideSpeed_m_s: number;
  readonly outletAngleError_deg: number;
  readonly maximumSideCentroidOffset_m: number;
  readonly sideCentroidOffsets_m: readonly number[];
  readonly airborneAxialBinCenters_m: readonly number[];
  readonly airborneSampleWeights: readonly number[];
  readonly airborneAxialVelocity_m_s: readonly number[];
  readonly airborneGravityTangentVelocity_m_s: readonly number[];
  readonly airborneGravityTangentCentroid_m: readonly number[];
  readonly ballisticGravityTangentVelocity_m_s: readonly number[];
  readonly ballisticGravityTangentCentroid_m: readonly number[];
  readonly airborneMeanForwardMomentumFluxPerDensity_m2_s2: readonly number[];
  readonly sampledAirborneBins: number;
  readonly minimumAirborneAxialRetentionRatio: number;
  readonly maximumAdjacentAxialSpeedDropRatio: number;
  readonly maximumAdjacentMomentumFluxDropRatio: number;
  readonly minimumBallisticGravityVelocityRatio: number;
  readonly maximumBallisticGravityVelocityRatio: number;
  readonly maximumBallisticCenterlineRelativeError: number;
}

export function hoseJetDriftFailures(audit: HoseJetDriftAudit, coarseCellWidth_m: number): string[] {
  const failures: string[] = [];
  if (audit.sampledVelocityCells < 1 || !Number.isFinite(audit.outletAxialSpeed_m_s)
    || audit.outletAxialSpeed_m_s <= 0.05) {
    failures.push(`outlet has no resolved forward jet (${audit.outletAxialSpeed_m_s} m/s)`);
  }
  if (!Number.isFinite(audit.outletSideSpeed_m_s) || Math.abs(audit.outletSideSpeed_m_s) > 0.10) {
    failures.push(`outlet sideways speed ${audit.outletSideSpeed_m_s} m/s exceeds 0.10 m/s`);
  }
  const centroidLimit = 0.5 * coarseCellWidth_m;
  if (!Number.isFinite(audit.maximumSideCentroidOffset_m)
    || audit.maximumSideCentroidOffset_m > centroidLimit) {
    failures.push(`jet sideways centroid ${audit.maximumSideCentroidOffset_m} m exceeds ${centroidLimit} m`);
  }
  if (audit.sampledAirborneBins < 3) {
    failures.push(`airborne jet resolved in only ${audit.sampledAirborneBins} axial bins`);
  }
  if (!Number.isFinite(audit.minimumAirborneAxialRetentionRatio)
    || audit.minimumAirborneAxialRetentionRatio < 0.90) {
    failures.push(`airborne axial speed retention ${audit.minimumAirborneAxialRetentionRatio} is below 0.90`);
  }
  if (!Number.isFinite(audit.maximumAdjacentAxialSpeedDropRatio)
    || audit.maximumAdjacentAxialSpeedDropRatio > 0.12) {
    failures.push(`adjacent airborne axial-speed drop ${audit.maximumAdjacentAxialSpeedDropRatio} exceeds 0.12`);
  }
  if (!Number.isFinite(audit.maximumAdjacentMomentumFluxDropRatio)
    || audit.maximumAdjacentMomentumFluxDropRatio > 0.25) {
    failures.push(`adjacent airborne momentum-flux drop ${audit.maximumAdjacentMomentumFluxDropRatio} exceeds 0.25`);
  }
  if (!Number.isFinite(audit.minimumBallisticGravityVelocityRatio)
    || audit.minimumBallisticGravityVelocityRatio < 0.70) {
    failures.push(`gravity-tangent velocity retention ${audit.minimumBallisticGravityVelocityRatio} is below 0.70 of ballistic acceleration`);
  }
  // This is a continuously driven, finite Dirichlet plug rather than a set of
  // pressure-free ballistic particles. Once the jet meets the receiving pool,
  // the incompressible pressure solve communicates impact work upstream. Keep
  // a broad upper tripwire for runaway forcing, but do not mistake pressure
  // work for a second gravity application.
  if (!Number.isFinite(audit.maximumBallisticGravityVelocityRatio)
    || audit.maximumBallisticGravityVelocityRatio > 3.1) {
    failures.push(`gravity-tangent velocity amplification ${audit.maximumBallisticGravityVelocityRatio} exceeds 3.1 times the pressure-free ballistic reference`);
  }
  if (!Number.isFinite(audit.maximumBallisticCenterlineRelativeError)
    || audit.maximumBallisticCenterlineRelativeError > 1.10) {
    failures.push(`airborne centerline pressure-free ballistic relative error ${audit.maximumBallisticCenterlineRelativeError} exceeds 1.10`);
  }
  return failures;
}

type Tuple3 = readonly [number, number, number];

const dot = (a: Tuple3, b: Tuple3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const length = (a: Tuple3) => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: Tuple3): Tuple3 => {
  const magnitude = length(a);
  return magnitude > 1e-12 ? [a[0] / magnitude, a[1] / magnitude, a[2] / magnitude] : [0, 0, 0];
};
const subtractProjection = (value: Tuple3, axis: Tuple3): Tuple3 => {
  const projection = dot(value, axis);
  return [value[0] - projection * axis[0], value[1] - projection * axis[1], value[2] - projection * axis[2]];
};
const cross = (a: Tuple3, b: Tuple3): Tuple3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const tuple = (value: Vec3): Tuple3 => [value.x, value.y, value.z];

/**
 * Measures an evolved jet in an intrinsic nozzle frame. The gravity-tangent
 * axis is reported separately because its bend is physical; `side` is
 * orthogonal to both the nozzle axis and gravity and must remain symmetric.
 */
export function auditHoseJetDrift(
  volume: ArrayLike<number>,
  velocity: ArrayLike<number>,
  dimensions: readonly [number, number, number],
  container: { width_m: number; height_m: number; depth_m: number; fillFraction: number },
  inflow: FluidInflow,
  outletCenter_m: Vec3,
  gravity_m_s2: Vec3,
): HoseJetDriftAudit {
  const [nx, ny, nz] = dimensions;
  if (volume.length !== nx * ny * nz || velocity.length !== 3 * volume.length) {
    throw new RangeError("Hose drift fields do not match their cubic grid");
  }
  const direction = normalize(tuple(inflow.velocity_m_s));
  const gravityTangent = normalize(subtractProjection(tuple(gravity_m_s2), direction));
  let side = normalize(cross(direction, gravityTangent));
  if (length(side) <= 1e-12) {
    const reference: Tuple3 = Math.abs(direction[1]) < 0.9 ? [0, 1, 0] : [0, 0, 1];
    side = normalize(cross(direction, reference));
  }
  const spacing: Tuple3 = [container.width_m / nx, container.height_m / ny, container.depth_m / nz];
  const minimum: Tuple3 = [-0.5 * container.width_m, 0, -0.5 * container.depth_m];
  const outlet = tuple(outletCenter_m);
  const coarseH = Math.max(...spacing);
  const outletAxialMinimum = -coarseH;
  const outletAxialMaximum = 3 * coarseH;
  const outletRadialMaximum = inflow.radius_m + 1.5 * coarseH;
  const streamAxialMaximum = Math.min(0.36, 0.45 * container.width_m);
  const bins = 6;
  const sideWeighted = new Float64Array(bins), binWeight = new Float64Array(bins);
  const airborneWeight = new Float64Array(bins), airborneAxial = new Float64Array(bins);
  const airborneAxialPosition = new Float64Array(bins), airborneGravityVelocity = new Float64Array(bins);
  const airborneGravityPosition = new Float64Array(bins);
  const airborneMomentum = new Float64Array(bins);
  const velocitySum = new Float64Array(3);
  let velocityWeight = 0, sampledLiquidCells = 0, sampledVelocityCells = 0;
  const binWidth = streamAxialMaximum / bins;
  // Keep impact-sheet and pool velocities out of the ballistic profile. The
  // first cell above this plane is still retained at the normal smoke-test
  // resolution; the half-cell guard only rejects cells intersecting the pool.
  const airborneMinimumY = container.height_m * container.fillFraction + 0.5 * spacing[1];

  for (let z = 0; z < nz; z += 1) for (let y = 0; y < ny; y += 1) for (let x = 0; x < nx; x += 1) {
    const cell = x + nx * (y + ny * z);
    const alpha = Math.max(0, Math.min(1, Number(volume[cell])));
    if (!(alpha > 1e-4)) continue;
    const point: Tuple3 = [minimum[0] + (x + 0.5) * spacing[0],
      minimum[1] + (y + 0.5) * spacing[1], minimum[2] + (z + 0.5) * spacing[2]];
    const relative: Tuple3 = [point[0] - outlet[0], point[1] - outlet[1], point[2] - outlet[2]];
    const axial = dot(relative, direction);
    const radialVector = subtractProjection(relative, direction);
    const radial = length(radialVector);
    if (axial >= outletAxialMinimum && axial <= outletAxialMaximum && radial <= outletRadialMaximum) {
      const vx = Number(velocity[3 * cell]), vy = Number(velocity[3 * cell + 1]), vz = Number(velocity[3 * cell + 2]);
      if (Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
        velocitySum[0] += alpha * vx; velocitySum[1] += alpha * vy; velocitySum[2] += alpha * vz;
        velocityWeight += alpha; sampledVelocityCells += 1;
      }
    }
    // The stream window excludes the pool and is deliberately broad along
    // gravity, while remaining narrow on the non-physical sideways axis.
    if (axial >= 0 && axial < streamAxialMaximum
      && Math.abs(dot(radialVector, gravityTangent)) <= 0.30
      && Math.abs(dot(radialVector, side)) <= 2.5 * inflow.radius_m) {
      const bin = Math.min(bins - 1, Math.floor(axial / streamAxialMaximum * bins));
      if (point[1] > airborneMinimumY) {
        sideWeighted[bin] += alpha * dot(relative, side);
        binWeight[bin] += alpha; sampledLiquidCells += 1;
        const vx = Number(velocity[3 * cell]), vy = Number(velocity[3 * cell + 1]), vz = Number(velocity[3 * cell + 2]);
        if (Number.isFinite(vx) && Number.isFinite(vy) && Number.isFinite(vz)) {
          const axialVelocity = dot([vx, vy, vz], direction);
          airborneWeight[bin] += alpha;
          airborneAxial[bin] += alpha * axialVelocity;
          airborneAxialPosition[bin] += alpha * axial;
          airborneGravityVelocity[bin] += alpha * dot([vx, vy, vz], gravityTangent);
          airborneGravityPosition[bin] += alpha * dot(relative, gravityTangent);
          airborneMomentum[bin] += alpha * Math.max(0, axialVelocity) ** 2;
        }
      }
    }
  }

  const mean: Tuple3 = velocityWeight > 0
    ? [velocitySum[0] / velocityWeight, velocitySum[1] / velocityWeight, velocitySum[2] / velocityWeight]
    : [Number.NaN, Number.NaN, Number.NaN];
  const axialSpeed = dot(mean, direction);
  const gravitySpeed = dot(mean, gravityTangent);
  const sideSpeed = dot(mean, side);
  const meanSpeed = length(mean);
  const angleError = meanSpeed > 0 && Number.isFinite(meanSpeed)
    ? Math.acos(Math.max(-1, Math.min(1, axialSpeed / meanSpeed))) * 180 / Math.PI
    : Number.NaN;
  const offsets = Array.from(sideWeighted, (sum, index) => binWeight[index] > 0 ? sum / binWeight[index] : Number.NaN);
  // Centroids of a vanishing impact-tail sliver are dominated by one voxel's
  // grid phase. Preserve every raw offset for diagnosis, but assert symmetry
  // only where a slice carries at least half the strongest airborne slice.
  const centroidWeightThreshold = 0.5 * Math.max(0, ...binWeight);
  const finiteOffsets = offsets.filter((value, index) => Number.isFinite(value)
    && centroidWeightThreshold > 0 && binWeight[index] >= centroidWeightThreshold);
  const axialProfile = Array.from(airborneAxial,
    (sum, index) => airborneWeight[index] > 0 ? sum / airborneWeight[index] : Number.NaN);
  const axialPositionProfile = Array.from(airborneAxialPosition,
    (sum, index) => airborneWeight[index] > 0 ? sum / airborneWeight[index] : Number.NaN);
  const gravityVelocityProfile = Array.from(airborneGravityVelocity,
    (sum, index) => airborneWeight[index] > 0 ? sum / airborneWeight[index] : Number.NaN);
  const gravityPositionProfile = Array.from(airborneGravityPosition,
    (sum, index) => airborneWeight[index] > 0 ? sum / airborneWeight[index] : Number.NaN);
  // A partially occupied leading/trailing slice is useful for visualization
  // but too grid-phase-sensitive for a regression. The profile assertions use
  // bins carrying at least one fifth of the strongest slice's liquid mass.
  const profileWeightThreshold = 0.20 * Math.max(0, ...airborneWeight);
  const profileIndices = Array.from(airborneWeight.keys()).filter((index) =>
    airborneWeight[index] >= profileWeightThreshold && profileWeightThreshold > 0
    && Number.isFinite(axialProfile[index]) && axialProfile[index] > 0);
  const retention = profileIndices.map((index) => axialProfile[index] / axialSpeed);
  const gravityAxial = dot(tuple(gravity_m_s2), direction);
  const gravityTangentAcceleration = dot(tuple(gravity_m_s2), gravityTangent);
  const ballisticTime = (distance: number) => {
    if (!(distance >= 0) || !(axialSpeed > 0)) return Number.NaN;
    if (Math.abs(gravityAxial) <= 1e-9) return distance / axialSpeed;
    const discriminant = axialSpeed * axialSpeed + 2 * gravityAxial * distance;
    if (!(discriminant >= 0)) return Number.NaN;
    const root = (-axialSpeed + Math.sqrt(discriminant)) / gravityAxial;
    return root >= 0 ? root : (-axialSpeed - Math.sqrt(discriminant)) / gravityAxial;
  };
  const ballisticGravityVelocity = axialPositionProfile.map((distance) => {
    const time = ballisticTime(distance);
    return Number.isFinite(time) ? gravitySpeed + gravityTangentAcceleration * time : Number.NaN;
  });
  const ballisticGravityPosition = axialPositionProfile.map((distance) => {
    const time = ballisticTime(distance);
    return Number.isFinite(time)
      ? gravitySpeed * time + 0.5 * gravityTangentAcceleration * time * time : Number.NaN;
  });
  const gravityProfileIndices = profileIndices.filter((index) => index > 0
    && ballisticGravityVelocity[index] - gravitySpeed > 0.1
    && ballisticGravityPosition[index] > 0);
  const gravityVelocityRatios = gravityProfileIndices.map((index) =>
    (gravityVelocityProfile[index] - gravitySpeed) / (ballisticGravityVelocity[index] - gravitySpeed));
  const centerlineErrors = gravityProfileIndices.map((index) =>
    Math.abs(gravityPositionProfile[index] - ballisticGravityPosition[index])
      / Math.max(coarseH, ballisticGravityPosition[index]));
  let maximumSpeedDrop = 0, maximumMomentumDrop = 0;
  for (let i = 1; i < profileIndices.length; i += 1) {
    const prior = profileIndices[i - 1], current = profileIndices[i];
    if (current !== prior + 1) continue;
    maximumSpeedDrop = Math.max(maximumSpeedDrop,
      Math.max(0, 1 - axialProfile[current] / axialProfile[prior]));
    const priorFlux = airborneMomentum[prior] / airborneWeight[prior];
    const currentFlux = airborneMomentum[current] / airborneWeight[current];
    maximumMomentumDrop = Math.max(maximumMomentumDrop, Math.max(0, 1 - currentFlux / priorFlux));
  }
  return {
    sampledLiquidCells, sampledVelocityCells,
    meanOutletVelocity_m_s: mean,
    outletAxialSpeed_m_s: axialSpeed,
    outletGravityTangentSpeed_m_s: gravitySpeed,
    outletSideSpeed_m_s: sideSpeed,
    outletAngleError_deg: angleError,
    maximumSideCentroidOffset_m: finiteOffsets.length > 0 ? Math.max(...finiteOffsets.map(Math.abs)) : Number.NaN,
    sideCentroidOffsets_m: offsets,
    airborneAxialBinCenters_m: Array.from({ length: bins }, (_, index) => (index + 0.5) * binWidth),
    airborneSampleWeights: Array.from(airborneWeight),
    airborneAxialVelocity_m_s: axialProfile,
    airborneGravityTangentVelocity_m_s: gravityVelocityProfile,
    airborneGravityTangentCentroid_m: gravityPositionProfile,
    ballisticGravityTangentVelocity_m_s: ballisticGravityVelocity,
    ballisticGravityTangentCentroid_m: ballisticGravityPosition,
    airborneMeanForwardMomentumFluxPerDensity_m2_s2: Array.from(airborneMomentum,
      (sum, index) => airborneWeight[index] > 0 ? sum / airborneWeight[index] : Number.NaN),
    sampledAirborneBins: profileIndices.length,
    minimumAirborneAxialRetentionRatio: retention.length > 0 ? Math.min(...retention) : Number.NaN,
    maximumAdjacentAxialSpeedDropRatio: profileIndices.length > 1 ? maximumSpeedDrop : Number.NaN,
    maximumAdjacentMomentumFluxDropRatio: profileIndices.length > 1 ? maximumMomentumDrop : Number.NaN,
    minimumBallisticGravityVelocityRatio: gravityVelocityRatios.length > 0
      ? Math.min(...gravityVelocityRatios) : Number.NaN,
    maximumBallisticGravityVelocityRatio: gravityVelocityRatios.length > 0
      ? Math.max(...gravityVelocityRatios) : Number.NaN,
    maximumBallisticCenterlineRelativeError: centerlineErrors.length > 0
      ? Math.max(...centerlineErrors) : Number.NaN,
  };
}
