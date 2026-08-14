import type {
  SparseAtlasCompositeGrid,
} from "./sparse-atlas-composite-projection";
import type {
  SparseAdaptiveMassAtlas,
  SparseBrickResolution,
} from "./sparse-brick-atlas";

export const SPARSE_ATLAS_TOPOLOGY_CADENCE_STEPS = 4;
export const SPARSE_ATLAS_PROMOTE_SCORE = 160;
export const SPARSE_ATLAS_EMERGENCY_PROMOTE_SCORE = 224;
export const SPARSE_ATLAS_DEMOTE_SCORE = 96;
export const SPARSE_ATLAS_PROMOTE_EPOCHS = 2;
export const SPARSE_ATLAS_DEMOTE_EPOCHS = 8;
export const SPARSE_ATLAS_DETAIL_VETO = 0.08;
export const SPARSE_ATLAS_MIN_FINE_RESIDENCE_STEPS = 64;

export const enum SparseAtlasActivityReason {
  Surface = 1 << 0,
  Deformation = 1 << 1,
  Temporal = 1 << 2,
  FineDetail = 1 << 3,
  PredictedFace = 1 << 4,
  Unknown = 1 << 5,
}

export interface SparseAtlasBrickActivityHistory {
  readonly scoreByte: number;
  readonly reasons: number;
  readonly hotEpochs: number;
  readonly quietEpochs: number;
  readonly meanDensity: number;
  readonly densityMoments: readonly [number, number, number];
  readonly lastTransitionStep: number;
}

export interface SparseAtlasResolutionPolicyState {
  readonly acceptedSteps: number;
  readonly history: ReadonlyMap<number, SparseAtlasBrickActivityHistory>;
  /** Method-level mixed-seam sentinel; removed automatically when its brick retires. */
  readonly pinnedFineBrickKeys: ReadonlySet<number>;
}

export interface SparseAtlasResolutionPolicyReceipt {
  readonly topologyEpoch: boolean;
  readonly measuredBrickCount: number;
  readonly surfaceBrickCount: number;
  readonly hotBrickCount: number;
  readonly quietBrickCount: number;
  readonly promotedBrickCount: number;
  readonly demotedBrickCount: number;
  readonly deferredPromotionCount: number;
  readonly targetFineBrickCount: number;
  readonly targetCoarseBrickCount: number;
  readonly maximumScoreByte: number;
}

export interface SparseAtlasResolutionDecision {
  readonly state: SparseAtlasResolutionPolicyState;
  readonly targetResolutionByBrick: ReadonlyMap<number, SparseBrickResolution>;
  readonly receipt: SparseAtlasResolutionPolicyReceipt;
}

export function initializeSparseAtlasResolutionPolicy(
  atlas: SparseAdaptiveMassAtlas,
): SparseAtlasResolutionPolicyState {
  const history = new Map<number, SparseAtlasBrickActivityHistory>();
  for (const brick of atlas.bricks) {
    history.set(brick.key, {
      scoreByte: 0,
      reasons: SparseAtlasActivityReason.Unknown,
      hotEpochs: 0,
      quietEpochs: 0,
      meanDensity: meanBrickDensity(brick.density),
      densityMoments: brickDensityMoments(brick.density, brick.resolution),
      lastTransitionStep: 0,
    });
  }
  const hasCoarseBrick = atlas.bricks.some((brick) => brick.resolution === 4);
  const pinnedFineBrickKeys = new Set<number>();
  if (hasCoarseBrick) {
    for (const brick of atlas.bricks) {
      if (brick.resolution === 8) pinnedFineBrickKeys.add(brick.key);
    }
  }
  return { acceptedSteps: 0, history, pinnedFineBrickKeys };
}

function meanBrickDensity(values: ArrayLike<number>): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let index = 0; index < values.length; index += 1) sum += values[index];
  return sum / values.length;
}

function brickDensityMoments(
  values: ArrayLike<number>,
  resolution: SparseBrickResolution,
): readonly [number, number, number] {
  const result = [0, 0, 0];
  const inverseCount = 1 / values.length;
  for (let z = 0; z < resolution; z += 1) {
    for (let y = 0; y < resolution; y += 1) {
      for (let x = 0; x < resolution; x += 1) {
        const rho = values[x + resolution * (y + resolution * z)];
        result[0] += rho * (2 * (x + 0.5) / resolution - 1) * inverseCount;
        result[1] += rho * (2 * (y + 0.5) / resolution - 1) * inverseCount;
        result[2] += rho * (2 * (z + 0.5) / resolution - 1) * inverseCount;
      }
    }
  }
  return result as [number, number, number];
}

function fineRestrictionError(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  brickKey: number,
): number {
  const base = grid.cellBaseByBrick.get(brickKey);
  const brick = grid.atlas.directory.get(brickKey);
  if (base === undefined || brick?.resolution !== 8) return 0;
  let maximum = 0;
  for (let coarseZ = 0; coarseZ < 4; coarseZ += 1) {
    for (let coarseY = 0; coarseY < 4; coarseY += 1) {
      for (let coarseX = 0; coarseX < 4; coarseX += 1) {
        let average = 0;
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const local = (2 * coarseX + dx) + 8
              * ((2 * coarseY + dy) + 8 * (2 * coarseZ + dz));
            average += density[base + local] / 8;
          }
        }
        for (let dz = 0; dz < 2; dz += 1) for (let dy = 0; dy < 2; dy += 1) {
          for (let dx = 0; dx < 2; dx += 1) {
            const local = (2 * coarseX + dx) + 8
              * ((2 * coarseY + dy) + 8 * (2 * coarseZ + dz));
            maximum = Math.max(maximum, Math.abs(density[base + local] - average));
          }
        }
      }
    }
  }
  return maximum;
}

function clampScore(value: number): number {
  return Math.max(0, Math.min(255, Math.round(255 * Math.max(0, Math.min(1, value)))));
}

/**
 * Measure accepted compact state and decide the next two-level topology.
 * Surface presence is diagnostic, not an unconditional refinement floor:
 * calm, nearly planar surface bricks are allowed to remain coarse. Activity
 * and demotion change only at an epoch, so presentation frames cannot drive
 * topology churn.
 */
export function planSparseAtlasResolution(
  grid: SparseAtlasCompositeGrid,
  density: ArrayLike<number>,
  velocity: ArrayLike<number>,
  previous: SparseAtlasResolutionPolicyState,
  dt_s: number,
  mode: "adaptive" | "all-fine" | "all-coarse" = "adaptive",
): SparseAtlasResolutionDecision {
  if (density.length !== grid.cells.length || velocity.length !== 3 * grid.cells.length) {
    throw new RangeError("resolution policy fields do not match the composite grid");
  }
  const acceptedSteps = previous.acceptedSteps + 1;
  if (mode !== "adaptive") {
    const fixedResolution: SparseBrickResolution = mode === "all-fine" ? 8 : 4;
    const targets = new Map<number, SparseBrickResolution>();
    let promotedBrickCount = 0, demotedBrickCount = 0;
    for (const brick of grid.atlas.bricks) {
      targets.set(brick.key, fixedResolution);
      if (brick.resolution < fixedResolution) promotedBrickCount += 1;
      if (brick.resolution > fixedResolution) demotedBrickCount += 1;
    }
    return {
      state: { ...previous, acceptedSteps },
      targetResolutionByBrick: targets,
      receipt: {
        topologyEpoch: promotedBrickCount > 0 || demotedBrickCount > 0,
        measuredBrickCount: grid.atlas.bricks.length,
        surfaceBrickCount: 0,
        hotBrickCount: 0,
        quietBrickCount: 0,
        promotedBrickCount,
        demotedBrickCount,
        deferredPromotionCount: 0,
        targetFineBrickCount: fixedResolution === 8 ? grid.atlas.bricks.length : 0,
        targetCoarseBrickCount: fixedResolution === 4 ? grid.atlas.bricks.length : 0,
        maximumScoreByte: 0,
      },
    };
  }
  const topologyEpoch = acceptedSteps % SPARSE_ATLAS_TOPOLOGY_CADENCE_STEPS === 0;
  const surface = new Set<number>();
  const surfaceAxes = new Map<number, number>();
  const predictedSurfaceMotion = new Map<number, number>();
  const deformation = new Map<number, number>();
  const densityIntegral = new Map<number, number>();
  const densityMomentIntegral = new Map<number, [number, number, number]>();
  const volumeIntegral = new Map<number, number>();

  for (const cell of grid.cells) {
    const rho = density[cell.id];
    if (rho > 0.05 && rho < 0.95) surface.add(cell.brickKey);
    densityIntegral.set(cell.brickKey,
      (densityIntegral.get(cell.brickKey) ?? 0) + cell.volume * rho);
    const moments = densityMomentIntegral.get(cell.brickKey) ?? [0, 0, 0];
    for (let axis = 0; axis < 3; axis += 1) {
      const localCoordinate = 2 * (cell.local[axis] + 0.5) / cell.brickResolution - 1;
      moments[axis] += cell.volume * rho * localCoordinate;
    }
    densityMomentIntegral.set(cell.brickKey, moments);
    volumeIntegral.set(cell.brickKey,
      (volumeIntegral.get(cell.brickKey) ?? 0) + cell.volume);
  }

  for (const row of grid.gradientRows) {
    const negativeVelocity = [0, 0, 0];
    const positiveVelocity = [0, 0, 0];
    let negativeDensity = 0;
    let positiveDensity = 0;
    let negativeWeight = 0;
    let positiveWeight = 0;
    for (const term of row.terms) {
      const weight = Math.abs(term.coefficient);
      const isNegative = term.coefficient < 0;
      if (isNegative) {
        negativeWeight += weight;
        negativeDensity += weight * density[term.cellId];
      } else {
        positiveWeight += weight;
        positiveDensity += weight * density[term.cellId];
      }
      const target = isNegative ? negativeVelocity : positiveVelocity;
      target[0] += weight * velocity[3 * term.cellId];
      target[1] += weight * velocity[3 * term.cellId + 1];
      target[2] += weight * velocity[3 * term.cellId + 2];
    }
    const hasNegative = negativeWeight > 0;
    const hasPositive = positiveWeight > 0;
    const negativeWet = hasNegative && negativeDensity / negativeWeight >= 0.5;
    const positiveWet = hasPositive && positiveDensity / positiveWeight >= 0.5;
    const crossesSurface = (hasNegative && hasPositive && negativeWet !== positiveWet)
      || (!hasNegative && positiveWet) || (!hasPositive && negativeWet);
    if (crossesSurface) {
      const negativeNormalVelocity = hasNegative
        ? negativeVelocity[row.axis] / negativeWeight : undefined;
      const positiveNormalVelocity = hasPositive
        ? positiveVelocity[row.axis] / positiveWeight : undefined;
      const faceNormalVelocity = negativeNormalVelocity === undefined
        ? positiveNormalVelocity!
        : positiveNormalVelocity === undefined
          ? negativeNormalVelocity
          : 0.5 * (negativeNormalVelocity + positiveNormalVelocity);
      // Velocity is stored in finest cells/s. One score unit means the
      // interface is predicted to travel a quarter of this composite row's
      // centre distance during the accepted step.
      const motion = dt_s * Math.abs(faceNormalVelocity)
        / Math.max(0.25 * row.distance, 1e-12);
      for (const term of row.terms) {
        const key = grid.cells[term.cellId].brickKey;
        surface.add(key);
        surfaceAxes.set(key, (surfaceAxes.get(key) ?? 0) | (1 << row.axis));
        predictedSurfaceMotion.set(key,
          Math.max(predictedSurfaceMotion.get(key) ?? 0, motion));
      }
    }
    if (!hasNegative || !hasPositive) continue;
    let maximumVariation = 0;
    for (let component = 0; component < 3; component += 1) {
      maximumVariation = Math.max(maximumVariation, dt_s * Math.abs(
        positiveVelocity[component] / positiveWeight
          - negativeVelocity[component] / negativeWeight,
      ) / Math.max(row.distance, 1e-12) / 0.15);
    }
    for (const term of row.terms) {
      const key = grid.cells[term.cellId].brickKey;
      deformation.set(key, Math.max(deformation.get(key) ?? 0, maximumVariation));
    }
  }

  const nextHistory = new Map<number, SparseAtlasBrickActivityHistory>();
  const targets = new Map<number, SparseBrickResolution>();
  const ordinaryPromotionBuckets = new Map<number, number[]>();
  let surfaceBrickCount = 0;
  let hotBrickCount = 0;
  let quietBrickCount = 0;
  let promotedBrickCount = 0;
  let demotedBrickCount = 0;
  let maximumScoreByte = 0;

  for (const brick of [...grid.atlas.bricks].sort((left, right) => left.key - right.key)) {
    const old = previous.history.get(brick.key);
    const meanDensity = (densityIntegral.get(brick.key) ?? 0)
      / Math.max(volumeIntegral.get(brick.key) ?? 1, 1e-30);
    const volume = Math.max(volumeIntegral.get(brick.key) ?? 1, 1e-30);
    const integratedMoments = densityMomentIntegral.get(brick.key) ?? [0, 0, 0];
    const densityMoments = integratedMoments.map((value) => value / volume) as [
      number, number, number,
    ];
    const temporal = old === undefined ? 0 : Math.max(
      Math.abs(meanDensity - old.meanDensity) / 0.05,
      ...densityMoments.map((value, axis) =>
        Math.abs(value - old.densityMoments[axis]) / 0.02),
    );
    const detailError = fineRestrictionError(grid, density, brick.key);
    const isSurface = surface.has(brick.key);
    const surfaceAxisMask = surfaceAxes.get(brick.key) ?? 0;
    const surfaceAxisCount = (surfaceAxisMask & 1 ? 1 : 0)
      + (surfaceAxisMask & 2 ? 1 : 0) + (surfaceAxisMask & 4 ? 1 : 0);
    const surfaceShape = surfaceAxisCount >= 2 ? 1 : 0;
    const predictedMotion = predictedSurfaceMotion.get(brick.key) ?? 0;
    let reasons = 0;
    if (isSurface) reasons |= SparseAtlasActivityReason.Surface;
    if ((deformation.get(brick.key) ?? 0) > 0) reasons |= SparseAtlasActivityReason.Deformation;
    if (temporal > 0) reasons |= SparseAtlasActivityReason.Temporal;
    if (detailError > 0) reasons |= SparseAtlasActivityReason.FineDetail;
    if (predictedMotion > 0) reasons |= SparseAtlasActivityReason.PredictedFace;
    if (!old) reasons |= SparseAtlasActivityReason.Unknown;
    const activity = Math.max(
      deformation.get(brick.key) ?? 0,
      temporal,
      detailError / SPARSE_ATLAS_DETAIL_VETO,
      surfaceShape,
      predictedMotion,
    );
    const scoreByte = clampScore(activity);
    maximumScoreByte = Math.max(maximumScoreByte, scoreByte);
    surfaceBrickCount += isSurface ? 1 : 0;
    const hot = scoreByte >= SPARSE_ATLAS_PROMOTE_SCORE;
    const quiet = scoreByte <= SPARSE_ATLAS_DEMOTE_SCORE
      && detailError <= SPARSE_ATLAS_DETAIL_VETO;
    hotBrickCount += hot ? 1 : 0;
    quietBrickCount += quiet ? 1 : 0;
    const hotEpochs = topologyEpoch
      ? hot ? Math.min(255, (old?.hotEpochs ?? 0) + 1) : 0
      : old?.hotEpochs ?? 0;
    const quietEpochs = topologyEpoch
      ? quiet ? Math.min(255, (old?.quietEpochs ?? 0) + 1) : 0
      : old?.quietEpochs ?? 0;
    let target = brick.resolution;
    if (brick.resolution === 4 && topologyEpoch
      && hotEpochs >= SPARSE_ATLAS_PROMOTE_EPOCHS) {
      const bucket = ordinaryPromotionBuckets.get(scoreByte) ?? [];
      bucket.push(brick.key);
      ordinaryPromotionBuckets.set(scoreByte, bucket);
    } else if (brick.resolution === 8 && topologyEpoch
      && quietEpochs >= SPARSE_ATLAS_DEMOTE_EPOCHS
      && !previous.pinnedFineBrickKeys.has(brick.key)
      && acceptedSteps - (old?.lastTransitionStep ?? 0)
        >= SPARSE_ATLAS_MIN_FINE_RESIDENCE_STEPS) {
      target = 4;
    }
    targets.set(brick.key, target);
    nextHistory.set(brick.key, {
      scoreByte,
      reasons,
      hotEpochs,
      quietEpochs,
      meanDensity,
      densityMoments,
      lastTransitionStep: old?.lastTransitionStep ?? 0,
    });
  }

  const maximumPromotionLeafDelta = Math.max(448,
    Math.min(8 * 448, Math.floor(0.10 * grid.cells.length)));
  let remainingPromotionLeafDelta = maximumPromotionLeafDelta;
  // CPU M1 keeps the pressure authority inside the matched frame budget. This
  // is a capacity control, not a scene special case: roughly one active fine
  // brick per 32 resident bricks, plus the one-brick floor needed to exercise
  // a live 2:1 seam. GPU page-pool work can raise this capacity later.
  const maximumTargetFineBrickCount = Math.max(
    previous.pinnedFineBrickKeys.size,
    Math.max(1, Math.ceil(grid.atlas.bricks.length / 32)),
  );
  let remainingFineBrickSlots = Math.max(0, maximumTargetFineBrickCount
    - grid.atlas.bricks.filter((brick) => targets.get(brick.key) === 8).length);
  let deferredPromotionCount = 0;
  for (const score of [...ordinaryPromotionBuckets.keys()].sort((a, b) => b - a)) {
    const keys = ordinaryPromotionBuckets.get(score)!.sort((a, b) => a - b);
    for (const key of keys) {
      if (remainingPromotionLeafDelta < 448 || remainingFineBrickSlots < 1) {
        deferredPromotionCount += 1;
        continue;
      }
      remainingPromotionLeafDelta -= 448;
      remainingFineBrickSlots -= 1;
      targets.set(key, 8);
    }
  }

  for (const brick of grid.atlas.bricks) {
    const target = targets.get(brick.key) ?? brick.resolution;
    if (target !== brick.resolution) {
      const current = nextHistory.get(brick.key)!;
      nextHistory.set(brick.key, {
        ...current,
        hotEpochs: 0,
        quietEpochs: 0,
        lastTransitionStep: acceptedSteps,
      });
      if (target === 8) promotedBrickCount += 1;
      else demotedBrickCount += 1;
    }
  }

  return {
    state: {
      acceptedSteps,
      history: nextHistory,
      pinnedFineBrickKeys: previous.pinnedFineBrickKeys,
    },
    targetResolutionByBrick: targets,
    receipt: {
      topologyEpoch,
      measuredBrickCount: grid.atlas.bricks.length,
      surfaceBrickCount,
      hotBrickCount,
      quietBrickCount,
      promotedBrickCount,
      demotedBrickCount,
      deferredPromotionCount,
      targetFineBrickCount: grid.atlas.bricks.filter((brick) =>
        targets.get(brick.key) === 8).length,
      targetCoarseBrickCount: grid.atlas.bricks.filter((brick) =>
        targets.get(brick.key) === 4).length,
      maximumScoreByte,
    },
  };
}

export function retainSparseAtlasResolutionPolicy(
  state: SparseAtlasResolutionPolicyState,
  atlas: SparseAdaptiveMassAtlas,
): SparseAtlasResolutionPolicyState {
  const history = new Map<number, SparseAtlasBrickActivityHistory>();
  for (const brick of atlas.bricks) {
    const entry = state.history.get(brick.key);
    if (entry) history.set(brick.key, entry);
  }
  const pinnedFineBrickKeys = new Set<number>();
  for (const key of state.pinnedFineBrickKeys) {
    if (atlas.directory.has(key)) pinnedFineBrickKeys.add(key);
  }
  return { acceptedSteps: state.acceptedSteps, history, pinnedFineBrickKeys };
}
