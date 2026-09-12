import type { SparseCM12ActivityPolicy } from "../features/adaptivity/policy";
import { SPARSE_CM12_ACTIVITY_POLICY } from "../features/adaptivity/policy";
import type { SliceTopology, SliceTopologyBrick, SliceTopologyResolution } from "./slice-topology";

const f = Math.fround;
const ACTIVITY_FIXED = 65_536;
const B = 8;
const VOLUME_ROUNDOFF_RATIO = 9.5367431640625e-7;

/** Current resident sources used for this dimensional port. */
export const SLICE_RESOLUTION_POLICY_SOURCE = Object.freeze({
  /** Hash of the named functions plus coarse-first.wgsl.ts, not unrelated resident code. */
  activitySourcesSha256: "6bc865d859c46f3d41343a0a34ebec4187e5bca45b8958ece4ba1afeab4265b1",
  policySha256: "778e296e35e67a183422dbda7b2c95fdeb352b26b37da9bbf4eec18201fc3129",
  residentFunctions: Object.freeze([
    "measureBrickActivity", "planBrickResolution", "closePlannedResolution",
    "validateCandidateResolution", "scheduleTopologyPreparation",
    "activateSweptFrontierPages", "reserveGeometricTransportFaceSupport",
    "retireUnsupportedEmptyBricks",
  ]),
});

export const enum SliceActivityReason {
  Surface = 1 << 0,
  Deformation = 1 << 1,
  Temporal = 1 << 2,
  FineDetail = 1 << 3,
  PredictedFace = 1 << 4,
  FirstStep = 1 << 5,
  Occupied = 1 << 6,
  VelocityFloor = 1 << 7,
  ThinFluid = 1 << 8,
  CutBoundary = 1 << 9,
  StaticBoundaryB8 = 1 << 10,
  StaticBoundaryB4 = 1 << 11,
  StaticBoundaryB2 = 1 << 12,
  StaticBoundaryB1 = 1 << 13,
  DensitySurface = 1 << 14,
}

export const enum SliceResolutionFault {
  None = 0,
  InvalidResolution = 1 << 0,
  TwoToOne = 1 << 1,
  LeafCapacity = 1 << 2,
  CellCapacity = 1 << 3,
  MissingBacking = 1 << 4,
}

export interface SliceResolutionRegion {
  readonly minimumFine: readonly [number, number];
  readonly maximumFine: readonly [number, number];
  /** Smallest permitted cell width in finest cells. */
  readonly minimumCellWidth: 1 | 2 | 4 | 8 | 16 | 32;
  /** Largest permitted cell width in finest cells. */
  readonly maximumCellWidth?: 1 | 2 | 4 | 8 | 16 | 32;
}

export interface SliceSurfaceProofState {
  readonly generationByTargetResolution: ReadonlyMap<SliceTopologyResolution, number>;
}

export interface SliceBrickActivityHistory {
  readonly scoreByte: number;
  readonly reasons: number;
  readonly hotEpochs: number;
  readonly quietEpochs: number;
  readonly proofEpochs: number;
  readonly meanDensity: number;
  readonly densityMoments: readonly [number, number];
  readonly meanVelocity: readonly [number, number];
  readonly velocityTravel: number;
  readonly supportMask: number;
  readonly sweptSupportMask: number;
  readonly lastTransitionStep: number;
  readonly surfaceProof?: SliceSurfaceProofState;
}

export interface SliceResolutionPolicyState {
  readonly acceptedSteps: number;
  readonly acceptedGeneration: number;
  readonly schedulingCursor: number;
  readonly schedulingCredits?: number;
  readonly history: ReadonlyMap<number, SliceBrickActivityHistory>;
}

export interface SliceResolutionPolicyFields {
  readonly density: ArrayLike<number>;
  readonly capacity: ArrayLike<number>;
  /** Compact interleaved x/y velocity, finest cells/s. */
  readonly cellVelocity: ArrayLike<number>;
  /** Accepted staggered row velocity, used for conservative receiver bounds. */
  readonly faceVelocity?: ArrayLike<number>;
  /** Next-step body acceleration, finest cells/s². */
  readonly accelerationFine?: readonly [number, number];
  /** Reconstructed x/y PLIC normals; required for coarse-first curvature parity. */
  readonly interfaceNormal?: ArrayLike<number>;
}

export interface SliceResolutionPolicyOptions {
  readonly policy?: SparseCM12ActivityPolicy;
  readonly refinementRegions?: readonly SliceResolutionRegion[];
  readonly staticBoundaryFloorByBrick?: ReadonlyMap<number, SliceTopologyResolution>;
  readonly movingRigidBodies?: boolean;
  readonly injectionDemandedBrickKeys?: ReadonlySet<number>;
  readonly frozenBrickKeys?: ReadonlySet<number>;
  readonly maximumLeaves?: number;
  readonly maximumCells?: number;
  /** Mirrors the resident WDR allocator; false is a diagnostic-only override. */
  readonly allocateMissingPages?: boolean;
  /** WDR free-list storage order; production claims from the final element. */
  readonly freeLeafIds?: readonly number[];
}

export interface SliceBrickResolutionReceipt {
  readonly brickKey: number;
  readonly acceptedResolution: SliceTopologyResolution;
  readonly requestedResolution: SliceTopologyResolution;
  readonly scheduledResolution: SliceTopologyResolution;
  readonly acceptedActive: boolean;
  readonly candidateActive: boolean;
  readonly scoreByte: number;
  readonly reasons: number;
  readonly planReasons: number;
  readonly supportMask: number;
  readonly sweptSupportMask: number;
  readonly faultBits: number;
}

export interface SliceResolutionPolicyReceipt {
  readonly topologyEpoch: boolean;
  readonly acceptedGeneration: number;
  readonly candidateGeneration: number;
  readonly measuredBrickCount: number;
  readonly surfaceBrickCount: number;
  readonly occupiedBrickCount: number;
  readonly activatedBrickCount: number;
  readonly allocatedBrickCount: number;
  readonly claimedLeafIds: Uint32Array;
  readonly retiredBrickCount: number;
  readonly promotedBrickCount: number;
  readonly demotedBrickCount: number;
  readonly deferredDemotionCount: number;
  readonly maximumScoreByte: number;
  readonly faultBits: number;
  readonly bricks: readonly SliceBrickResolutionReceipt[];
}

export interface SliceResolutionPolicyDecision {
  readonly candidateBricks: readonly SliceTopologyBrick[];
  readonly state: SliceResolutionPolicyState;
  readonly receipt: SliceResolutionPolicyReceipt;
}

const resolution = (value: number): SliceTopologyResolution => {
  if (value === 1 || value === 2 || value === 4 || value === 8) return value;
  throw new RangeError(`invalid slice resolution ${value}`);
};
const spanOf = (brick: SliceTopologyBrick) => brick.spanBricks ?? 1;
const active = (brick: SliceTopologyBrick) => brick.active !== false;
const widthOf = (brick: SliceTopologyBrick, rung = brick.resolution) =>
  B * spanOf(brick) / rung;
const clampByte = (value: number) => Math.max(0, Math.min(255,
  Math.round(255 * Math.max(0, Math.min(1, value)))));
const popcount2 = (bits: number) => (bits & 1 ? 1 : 0) + (bits & 2 ? 1 : 0);

export function initializeSliceResolutionPolicy(
  topology: SliceTopology,
): SliceResolutionPolicyState {
  const history = new Map<number, SliceBrickActivityHistory>();
  for (const brick of topology.bricks) history.set(brick.key, {
    scoreByte: 0, reasons: SliceActivityReason.FirstStep,
    hotEpochs: 0, quietEpochs: 0, proofEpochs: 0,
    meanDensity: 0, densityMoments: [0, 0], meanVelocity: [0, 0], velocityTravel: 0,
    supportMask: 0, sweptSupportMask: 0, lastTransitionStep: 0,
  });
  return { acceptedSteps: 0, acceptedGeneration: topology.generation,
    schedulingCursor: 0, schedulingCredits: 0, history };
}

function policyThresholds(policy: SparseCM12ActivityPolicy, dt: number,
  cellSize: number): ReadonlyMap<SliceTopologyResolution, number> {
  if (policy.coarseFirst) {
    const finest = f(Math.sqrt(2 * policy.energyThreshold) * dt / cellSize);
    return new Map([[1, f(finest / 8)], [2, f(finest / 4)],
      [4, f(finest / 2)], [8, finest]]);
  }
  return new Map([[1, f(policy.twoTravelCells / 2)],
    [2, f(policy.twoTravelCells)], [4, f(policy.fourTravelCells)],
    [8, f(policy.finestTravelCells)]]);
}

function velocityFloor(travel: number, thresholds: ReadonlyMap<SliceTopologyResolution, number>,
  enabled: boolean): SliceTopologyResolution {
  if (!enabled) return 1;
  for (const rung of [8, 4, 2, 1] as const) if (travel >= thresholds.get(rung)!) return rung;
  return 1;
}

function bounds(brick: SliceTopologyBrick) {
  const lo = [B * brick.coordinate[0], B * brick.coordinate[1]] as const;
  const size = B * spanOf(brick);
  return { lo, hi: [lo[0] + size, lo[1] + size] as const };
}

function overlaps(a0: number, a1: number, b0: number, b1: number) {
  return Math.min(a1, b1) > Math.max(a0, b0);
}

function regionBounds(brick: SliceTopologyBrick, regions: readonly SliceResolutionRegion[]) {
  const box = bounds(brick), nominal = B * spanOf(brick);
  let floor = 1, ceiling = 0;
  for (const region of regions) {
    const intersects = overlaps(box.lo[0], box.hi[0], region.minimumFine[0], region.maximumFine[0])
      && overlaps(box.lo[1], box.hi[1], region.minimumFine[1], region.maximumFine[1]);
    if (intersects) floor = Math.max(floor, region.minimumCellWidth);
    const contained = box.lo[0] >= region.minimumFine[0] && box.lo[1] >= region.minimumFine[1]
      && box.hi[0] <= region.maximumFine[0] && box.hi[1] <= region.maximumFine[1];
    if (contained && region.maximumCellWidth !== undefined) ceiling = ceiling === 0
      ? region.maximumCellWidth : Math.min(ceiling, region.maximumCellWidth);
  }
  const maximumResolution = resolution(Math.max(1, Math.min(B, nominal / floor)));
  const minimumResolution = resolution(ceiling === 0 ? 1
    : Math.max(1, Math.min(B, nominal / ceiling)));
  return { maximumResolution, minimumResolution };
}

function applyRegionBounds(requested: SliceTopologyResolution, brick: SliceTopologyBrick,
  regions: readonly SliceResolutionRegion[]): SliceTopologyResolution {
  const limits = regionBounds(brick, regions);
  return resolution(Math.min(limits.maximumResolution,
    Math.max(limits.minimumResolution, requested)));
}

function ownerAt(bricks: readonly SliceTopologyBrick[], x: number, y: number) {
  return bricks.find(brick => {
    const box = bounds(brick);
    return x >= box.lo[0] && x < box.hi[0] && y >= box.lo[1] && y < box.hi[1];
  });
}

function faceNeighbors(bricks: readonly SliceTopologyBrick[], brick: SliceTopologyBrick) {
  const a = bounds(brick);
  return bricks.filter(other => {
    if (other.key === brick.key) return false;
    const b = bounds(other);
    return (a.hi[0] === b.lo[0] || b.hi[0] === a.lo[0])
        && overlaps(a.lo[1], a.hi[1], b.lo[1], b.hi[1])
      || (a.hi[1] === b.lo[1] || b.hi[1] === a.lo[1])
        && overlaps(a.lo[0], a.hi[0], b.lo[0], b.hi[0]);
  });
}

interface Measurement extends SliceBrickActivityHistory {
  surface: boolean;
  thin: boolean;
  occupied: boolean;
  detail: number;
  deeplyEnclosed: boolean;
  curvatureFloor: SliceTopologyResolution;
}

function measure(topology: SliceTopology, fields: SliceResolutionPolicyFields,
  brick: SliceTopologyBrick, old: SliceBrickActivityHistory | undefined,
  policy: SparseCM12ActivityPolicy, topologyEpoch: boolean, dt: number,
  thresholds: ReadonlyMap<SliceTopologyResolution, number>): Measurement {
  if (!active(brick)) return { scoreByte: 0, reasons: old?.reasons ? old.reasons & 0x3c00 : 0,
    hotEpochs: old?.hotEpochs ?? 0, quietEpochs: old?.quietEpochs ?? 0,
    proofEpochs: old?.proofEpochs ?? 0, meanDensity: 0, densityMoments: [0, 0],
    meanVelocity: [0, 0], velocityTravel: 0, supportMask: 0, sweptSupportMask: 0,
    lastTransitionStep: old?.lastTransitionStep ?? 0, surface: false, thin: false,
    occupied: false, detail: 0, deeplyEnclosed: false, curvatureFloor: 1,
    surfaceProof: old?.surfaceProof };
  const cells = topology.cells.filter(cell => cell.brickKey === brick.key);
  let densitySum = 0, momentX = 0, momentY = 0;
  let deformation = 0, predicted = 0, detail = 0, travel = 0;
  let axes = 0, occupiedCell = false, substantial = false, thin = false;
  let cut = false, densitySurface = false, supportMask = 0, sweptMask = 0;
  let momentumX = 0, momentumY = 0, momentumMass = 0;
  let normalMinX = 1, normalMinY = 1, normalMaxX = -1, normalMaxY = -1;
  const featureDensity = Math.max(policy.residencyDensity, policy.thinFeatureDensity);
  for (const cell of cells) {
    const rho = fields.density[cell.id]!;
    const cap = Math.max(fields.capacity[cell.id]!, 1e-6);
    const fill = rho / cap;
    const ownFixed = Math.round(rho * ACTIVITY_FIXED);
    densitySum += ownFixed;
    momentX += Math.round(rho * (2 * cell.local[0] + 1 - brick.resolution)
      / brick.resolution * ACTIVITY_FIXED);
    momentY += Math.round(rho * (2 * cell.local[1] + 1 - brick.resolution)
      / brick.resolution * ACTIVITY_FIXED);
    cut ||= fields.capacity[cell.id]! < 0.999;
    occupiedCell ||= rho > policy.residencyDensity;
    substantial ||= fill > policy.surfaceDensityMinimum;
    const wet = fill >= 0.5;
    const vx = fields.cellVelocity[2 * cell.id]!, vy = fields.cellVelocity[2 * cell.id + 1]!;
    if (policy.coarseFirst && wet) {
      momentumX = f(momentumX + f(vx * rho));
      momentumY = f(momentumY + f(vy * rho));
      momentumMass = f(momentumMass + rho);
    }
    const begin = topology.incidenceOffsets[cell.id]!, end = topology.incidenceOffsets[cell.id + 1]!;
    // Match the conservative transport's per-subface accepted-volume test.
    // Projection residue below gvRoundoff cannot move represented material;
    // using raw nonzero velocity here would turn it into a whole dry page.
    const sweptMinimum = [0, 0], sweptMaximum = [0, 0];
    const transportRoundoff = VOLUME_ROUNDOFF_RATIO * cap * cell.volumeFineCells;
    for (let at = begin; at < end; at++) {
      const row = topology.rows[topology.incidences[at]!.row]!;
      const dynamic = row as typeof row & { openFraction?: number; solidVelocity?: number };
      const open = dynamic.openFraction ?? 1, wall = dynamic.solidVelocity ?? 0;
      const stored = fields.faceVelocity?.[row.id] ?? (row.axis === 0 ? vx : vy);
      const fluid = open > 1e-6 ? (stored - (1 - open) * wall) / open : wall;
      const endpoint = fluid + dt * (fields.accelerationFine?.[row.axis] ?? 0);
      for (const candidate of [fluid, endpoint]) {
        if (dt * row.areaFineCells * Math.abs(candidate) <= transportRoundoff) continue;
        sweptMinimum[row.axis] = Math.min(sweptMinimum[row.axis]!, candidate);
        sweptMaximum[row.axis] = Math.max(sweptMaximum[row.axis]!, candidate);
      }
    }
    let interfaceCell = false, exposed = 0;
    for (let at = begin; at < end; at++) {
      const row = topology.rows[topology.incidences[at]!.row]!;
      const own = row.terms.find(term => term.cellId === cell.id)!;
      const others = row.terms.filter(term => term.cellId !== cell.id
        && own.coefficient * term.coefficient < 0);
      let sideHasFluid = false;
      for (const term of others) {
        const neighbor = topology.cells[term.cellId]!;
        const neighborFill = fields.density[term.cellId]!
          / Math.max(fields.capacity[term.cellId]!, 1e-6);
        sideHasFluid ||= neighborFill > featureDensity;
        const crosses = (neighborFill >= 0.5) !== wet;
        if (crosses && (!wet || neighbor.brickKey === brick.key || policy.coarseFirst)) {
          interfaceCell = true; densitySurface = true; axes |= 1 << row.axis;
        }
        if (crosses) {
          const liquidId = wet ? cell.id : term.cellId;
          predicted = Math.max(predicted, f(dt * Math.abs(fields.cellVelocity[2 * liquidId + row.axis]!)
            / Math.max(0.25 * row.centerDistanceFine, 1e-12)));
        }
        if (wet && neighborFill >= 0.5) {
          const nvx = fields.cellVelocity[2 * term.cellId]!, nvy = fields.cellVelocity[2 * term.cellId + 1]!;
          deformation = Math.max(deformation, f(dt * Math.max(Math.abs(vx - nvx), Math.abs(vy - nvy))
            / Math.max(0.15 * row.centerDistanceFine, 1e-12)));
        }
      }
      if (row.kind === "sparse-air" && row.boundaryMode !== "closed"
        && others.length === 0 && wet) {
        interfaceCell = true; axes |= 1 << row.axis;
        predicted = Math.max(predicted, f(dt * Math.abs(fields.cellVelocity[2 * cell.id + row.axis]!)
          / Math.max(0.25 * row.centerDistanceFine, 1e-12)));
      }
      if (rho > featureDensity && !sideHasFluid) {
        const side = row.centerFine[row.axis] > cell.centerFine[row.axis] ? 1 : 0;
        exposed |= 1 << (2 * row.axis + side);
      }
    }
    const thickness = Math.max(0, Math.min(1, rho)) * Math.min(...cell.widthsFine);
    const cellThin = fill > featureDensity && thickness < policy.thinFeatureCells
      && (((exposed & 3) === 3) || ((exposed & 12) === 12));
    thin ||= cellThin;
    if (policy.coarseFirst && (interfaceCell || cellThin) && fields.interfaceNormal) {
      const nx = fields.interfaceNormal[2 * cell.id]!, ny = fields.interfaceNormal[2 * cell.id + 1]!;
      if (nx * nx + ny * ny > 0.5) {
        normalMinX = Math.min(normalMinX, nx); normalMaxX = Math.max(normalMaxX, nx);
        normalMinY = Math.min(normalMinY, ny); normalMaxY = Math.max(normalMaxY, ny);
      }
    }
    if ((interfaceCell && wet) || cellThin || (policy.coarseFirst && wet)) {
      travel = Math.max(travel, f(dt * Math.hypot(vx, vy)));
    }
    if (interfaceCell || cellThin || rho !== 0) {
      const box = bounds(brick), cx = cell.centerFine[0], cy = cell.centerFine[1];
      const dxs = cx - 0.5 * cell.widthsFine[0] <= box.lo[0] ? [-1, 0]
        : cx + 0.5 * cell.widthsFine[0] >= box.hi[0] ? [0, 1] : [0];
      const dys = cy - 0.5 * cell.widthsFine[1] <= box.lo[1] ? [-1, 0]
        : cy + 0.5 * cell.widthsFine[1] >= box.hi[1] ? [0, 1] : [0];
      for (const dy of dys) for (const dx of dxs) if (dx || dy) {
        const bit = dx + 1 + 3 * (dy + 1);
        if (interfaceCell || cellThin) supportMask |= 1 << bit;
      }
      if (rho !== 0) {
        sweptMask |= 1 << 4;
        // Page membership follows the actual forward material envelope. Use
        // the donor cell extent, rather than only its centre, so any outward
        // flux from a boundary cell reserves its receiver even when travel is
        // below half a cell. A stationary donor requests no dry page.
        const travelX = [dt * sweptMinimum[0]!, dt * sweptMaximum[0]!];
        const travelY = [dt * sweptMinimum[1]!, dt * sweptMaximum[1]!];
        const cellLoX = cx - 0.5 * cell.widthsFine[0];
        const cellHiX = cx + 0.5 * cell.widthsFine[0];
        const cellLoY = cy - 0.5 * cell.widthsFine[1];
        const cellHiY = cy + 0.5 * cell.widthsFine[1];
        const minDx = cellLoX + Math.min(...travelX) < box.lo[0] ? -1 : 0;
        const maxDx = cellHiX + Math.max(...travelX) > box.hi[0] ? 1 : 0;
        const minDy = cellLoY + Math.min(...travelY) < box.lo[1] ? -1 : 0;
        const maxDy = cellHiY + Math.max(...travelY) > box.hi[1] ? 1 : 0;
        for (let y = minDy; y <= maxDy; y++)
          for (let x = minDx; x <= maxDx; x++) if (x || y) {
            supportMask |= 1 << (x + 1 + 3 * (y + 1));
            sweptMask |= 1 << (x + 1 + 3 * (y + 1));
          }
      }
    }
  }
  if (brick.resolution > 1) {
    const base = topology.cellBaseByBrick.get(brick.key)!;
    for (let y = 0; y < brick.resolution; y += 2) for (let x = 0; x < brick.resolution; x += 2) {
      const values = [0, 1, brick.resolution, brick.resolution + 1]
        .map(offset => Math.round(fields.density[base + x + brick.resolution * y + offset]! * ACTIVITY_FIXED));
      const sum = values.reduce((a, v) => a + v, 0);
      for (const value of values) detail = Math.max(detail,
        f(Math.abs(4 * value - sum) / (4 * ACTIVITY_FIXED)));
    }
  }
  const count = Math.max(1, cells.length);
  const meanDensity = f(densitySum / (count * ACTIVITY_FIXED));
  const moments = [f(momentX / (count * ACTIVITY_FIXED)),
    f(momentY / (count * ACTIVITY_FIXED))] as const;
  const massFine = f(densitySum / ACTIVITY_FIXED * (cells[0]?.volumeFineCells ?? 0));
  const represented = substantial || thin;
  const occupied = occupiedCell && represented && massFine >= policy.residencyMassFineCells;
  const surface = occupied && axes !== 0;
  const shape = popcount2(axes) >= 2 ? 1 : 0;
  const temporal = !policy.coarseFirst && old ? Math.max(
    Math.abs(meanDensity - old.meanDensity) / 0.05,
    Math.abs(moments[0] - old.densityMoments[0]) / 0.02,
    Math.abs(moments[1] - old.densityMoments[1]) / 0.02) : 0;
  const scoredDetail = surface && !thin && shape === 0 ? 0 : detail;
  const dynamic = surface || thin ? Math.max(deformation, temporal) : 0;
  const feature = Math.max(dynamic, predicted, shape, thin ? 1 : 0,
    Math.max(0, scoredDetail / policy.detailTolerance - 1));
  const scoredVelocity = surface || thin ? travel / Math.max(thresholds.get(8)!, 1e-6) : 0;
  const normalDiameter = normalMaxX >= normalMinX && normalMaxY >= normalMinY
    ? Math.hypot(Math.max(0, normalMaxX - normalMinX), Math.max(0, normalMaxY - normalMinY)) : 0;
  let curvatureFloor: SliceTopologyResolution = 1;
  while (curvatureFloor < 8 && normalDiameter / curvatureFloor > policy.curvatureTolerance) {
    curvatureFloor = resolution(curvatureFloor * 2);
  }
  const coarseScore = Math.max(normalDiameter
    / (brick.resolution * Math.max(policy.curvatureTolerance, 0.02)),
  travel / Math.max(thresholds.get(8)!, 1e-6));
  const scoreByte = clampByte(policy.coarseFirst ? coarseScore : Math.max(scoredVelocity, feature));
  let reasons = 0;
  if (surface) reasons |= SliceActivityReason.Surface;
  if (deformation >= policy.emergencyScore) reasons |= SliceActivityReason.Deformation;
  if (temporal > 0) reasons |= SliceActivityReason.Temporal;
  if (detail > policy.detailTolerance) reasons |= SliceActivityReason.FineDetail;
  if (predicted >= policy.emergencyScore) reasons |= SliceActivityReason.PredictedFace;
  if (!old) reasons |= SliceActivityReason.FirstStep;
  if (occupied) reasons |= SliceActivityReason.Occupied;
  if (velocityFloor(travel, thresholds, policy.activitySignals) > 1) reasons |= SliceActivityReason.VelocityFloor;
  if (thin) reasons |= SliceActivityReason.ThinFluid;
  if (cut) reasons |= SliceActivityReason.CutBoundary;
  if (densitySurface) reasons |= SliceActivityReason.DensitySurface;
  if (policy.coarseFirst) reasons |= curvatureFloor << 16;
  const hot = policy.activitySignals && feature >= policy.promoteScore;
  const quiet = !thin && scoreByte / 255 <= policy.demoteScore && scoredDetail <= policy.detailTolerance;
  const hotEpochs = topologyEpoch ? hot ? Math.min(255, (old?.hotEpochs ?? 0) + 1) : 0 : old?.hotEpochs ?? 0;
  const quietEpochs = topologyEpoch ? quiet ? Math.min(255, (old?.quietEpochs ?? 0) + 1) : 0 : old?.quietEpochs ?? 0;
  const ownBox = bounds(brick), neighbors = faceNeighbors(topology.bricks, brick);
  const enclosedSides = [false, false, false, false];
  for (const neighbor of neighbors) {
    const box = bounds(neighbor);
    if (box.hi[0] === ownBox.lo[0]) enclosedSides[0] = true;
    if (box.lo[0] === ownBox.hi[0]) enclosedSides[1] = true;
    if (box.hi[1] === ownBox.lo[1]) enclosedSides[2] = true;
    if (box.lo[1] === ownBox.hi[1]) enclosedSides[3] = true;
  }
  const deeplyEnclosed = occupied && enclosedSides.every(Boolean) && neighbors.every(neighbor => {
    const entry = topology.cells.filter(cell => cell.brickKey === neighbor.key);
    return active(neighbor) && entry.length > 0 && entry.every(cell =>
      fields.density[cell.id]! / Math.max(fields.capacity[cell.id]!, 1e-6) >= 0.5);
  });
  const meanVelocity = [momentumMass > 1e-8 ? f(momentumX / momentumMass) : 0,
    momentumMass > 1e-8 ? f(momentumY / momentumMass) : 0] as const;
  return { scoreByte, reasons, hotEpochs, quietEpochs,
    proofEpochs: old?.proofEpochs ?? 0, meanDensity, densityMoments: moments, meanVelocity,
    velocityTravel: travel, supportMask, sweptSupportMask: sweptMask,
    lastTransitionStep: old?.lastTransitionStep ?? 0, surface, thin, occupied,
    detail, deeplyEnclosed, curvatureFloor, surfaceProof: old?.surfaceProof };
}

function approachTravel(delta: readonly [number, number], sweep: readonly [number, number],
  extent: number): number {
  let enter = 0, leave = 1, approach = 0;
  for (let axis = 0; axis < 2; axis++) {
    const distance = Math.abs(delta[axis]!), motion = sweep[axis]!;
    if (distance >= extent) {
      const closing = Math.sign(delta[axis]!) * motion;
      if (closing <= 0) return 0;
      const axisEnter = (distance - extent) / closing;
      if (axisEnter > enter || approach === 0) approach = closing;
      else if (axisEnter === enter) approach = Math.min(approach, closing);
      enter = Math.max(enter, axisEnter);
      leave = Math.min(leave, (distance + extent) / closing);
    } else if (Math.abs(motion) > 1e-8) {
      leave = Math.min(leave, (extent + Math.sign(motion) * delta[axis]!) / Math.abs(motion));
    }
  }
  return enter < leave ? approach : 0;
}

/** Z-removed form of the resident lifecycle's exact adjacent-page lookup. */
function directionallyDemandedBrickKeys(
  topology: SliceTopology,
  measurements: ReadonlyMap<number, Measurement>,
): ReadonlySet<number> {
  const demanded = new Set<number>();
  for (const source of topology.bricks) {
    if (!active(source)) continue;
    const mask = measurements.get(source.key)?.sweptSupportMask ?? 0;
    const span = spanOf(source);
    for (let bit = 0; bit < 9; bit++) {
      if (bit === 4 || (mask & (1 << bit)) === 0) continue;
      const dx = bit % 3 - 1, dy = Math.floor(bit / 3) - 1;
      const qx = source.coordinate[0] + (dx < 0 ? -1 : dx > 0 ? span : 0);
      const qy = source.coordinate[1] + (dy < 0 ? -1 : dy > 0 ? span : 0);
      const owner = ownerAt(topology.bricks, qx * B + 0.5, qy * B + 0.5);
      if (owner && owner.key !== source.key) demanded.add(owner.key);
    }
  }
  return demanded;
}

function incomingFloor(topology: SliceTopology, brick: SliceTopologyBrick,
  measurements: ReadonlyMap<number, Measurement>, policy: SparseCM12ActivityPolicy): SliceTopologyResolution {
  if (policy.anticipationSeconds <= 0) return 1;
  const box = bounds(brick), center = [0.5 * (box.lo[0] + box.hi[0]),
    0.5 * (box.lo[1] + box.hi[1])] as const;
  const receiver = measurements.get(brick.key)!;
  let required: SliceTopologyResolution = 1;
  for (const source of topology.bricks) {
    if (!active(source) || source.key === brick.key) continue;
    const sb = bounds(source), sx = 0.5 * (sb.lo[0] + sb.hi[0]), sy = 0.5 * (sb.lo[1] + sb.hi[1]);
    if (Math.abs(source.coordinate[0] - brick.coordinate[0]) > policy.anticipationRadiusBricks
      || Math.abs(source.coordinate[1] - brick.coordinate[1]) > policy.anticipationRadiusBricks) continue;
    const m = measurements.get(source.key)!;
    if (!m.occupied || !(m.surface || m.thin)) continue;
    const sweep = [policy.anticipationSeconds * (m.meanVelocity[0] - receiver.meanVelocity[0]),
      policy.anticipationSeconds * (m.meanVelocity[1] - receiver.meanVelocity[1])] as const;
    if (Math.hypot(...sweep) <= 1) continue;
    const delta = [center[0] - sx, center[1] - sy] as const;
    const extent = 0.5 * B * (spanOf(brick) + spanOf(source));
    const approach = approachTravel(delta, sweep, extent);
    if (approach <= 1) continue;
    const gap = Math.hypot(Math.max(Math.abs(delta[0]) - extent, 0),
      Math.max(Math.abs(delta[1]) - extent, 0));
    const demand = B * Math.min(1, approach / Math.max(B, gap + B));
    let rung: SliceTopologyResolution = 1;
    while (rung < 8 && rung < demand) rung = resolution(rung * 2);
    if (rung > required) required = rung;
  }
  return required;
}

function closeTwoToOne(bricks: readonly SliceTopologyBrick[], targets: Map<number, SliceTopologyResolution>,
  regions: readonly SliceResolutionRegion[]): void {
  for (;;) {
    let changed = false;
    for (const brick of bricks) for (const neighbor of faceNeighbors(bricks, brick)) {
      if (neighbor.key < brick.key) continue;
      const own = targets.get(brick.key)!, other = targets.get(neighbor.key)!;
      const ownWidth = widthOf(brick, own), otherWidth = widthOf(neighbor, other);
      if (Math.max(ownWidth, otherWidth) <= 2 * Math.min(ownWidth, otherWidth)) continue;
      const coarse = ownWidth > otherWidth ? brick : neighbor;
      const current = targets.get(coarse.key)!;
      const raised = applyRegionBounds(resolution(Math.min(8, current * 2)), coarse, regions);
      if (raised === current) continue;
      targets.set(coarse.key, raised); changed = true;
    }
    if (!changed) return;
  }
}

function closeRegionGradingCaps(bricks: readonly SliceTopologyBrick[],
  targets: Map<number, SliceTopologyResolution>, regions: readonly SliceResolutionRegion[]): void {
  if (regions.length === 0) return;
  const caps = new Map(bricks.map(brick => [brick.key,
    regionBounds(brick, regions).maximumResolution]));
  for (;;) {
    let changed = false;
    for (const brick of bricks) for (const neighbor of faceNeighbors(bricks, brick)) {
      const ownCap = caps.get(brick.key)!, neighborCap = caps.get(neighbor.key)!;
      const propagated = resolution(Math.max(1, Math.min(8,
        2 * neighborCap * spanOf(brick) / spanOf(neighbor))));
      if (propagated < ownCap) { caps.set(brick.key, propagated); changed = true; }
    }
    if (!changed) break;
  }
  for (const brick of bricks) targets.set(brick.key,
    resolution(Math.min(targets.get(brick.key)!, caps.get(brick.key)!)));
}

/**
 * CPU execution of the resident activity/lifecycle planning transaction with Z
 * removed. Candidate fields remain non-authoritative until the caller runs the
 * generation transfer and commits the topology authority.
 */
export function planSliceResolution(input: {
  readonly topology: SliceTopology;
  readonly fields: SliceResolutionPolicyFields;
  readonly previous: SliceResolutionPolicyState;
  readonly dt: number;
  readonly cellSize: number;
  readonly options?: SliceResolutionPolicyOptions;
}): SliceResolutionPolicyDecision {
  const { topology, fields, previous } = input;
  if (fields.density.length !== topology.cells.length || fields.capacity.length !== topology.cells.length
    || fields.cellVelocity.length !== 2 * topology.cells.length) {
    throw new RangeError("slice resolution fields do not match accepted topology");
  }
  if (!(input.dt > 0) || !(input.cellSize > 0)) throw new RangeError("slice resolution dt/cellSize must be positive");
  const options = input.options ?? {};
  const policy: SparseCM12ActivityPolicy = options.policy ?? SPARSE_CM12_ACTIVITY_POLICY;
  const regions = options.refinementRegions ?? [], acceptedSteps = previous.acceptedSteps + 1;
  const topologyEpoch = acceptedSteps % policy.topologyCadenceSteps === 0;
  const thresholds = policyThresholds(policy, input.dt, input.cellSize);
  const measurements = new Map<number, Measurement>();
  for (const brick of topology.bricks) measurements.set(brick.key,
    measure(topology, fields, brick, previous.history.get(brick.key), policy,
      topologyEpoch, input.dt, thresholds));
  const materialDemand = new Set(directionallyDemandedBrickKeys(topology, measurements));
  for (const key of options.injectionDemandedBrickKeys ?? []) materialDemand.add(key);
  const freeLeafIds = options.freeLeafIds ?? [];
  const freeSet = new Set<number>();
  for (const leaf of freeLeafIds) {
    if (!Number.isSafeInteger(leaf) || leaf < 0 || freeSet.has(leaf)) {
      throw new RangeError(`invalid or duplicate slice WDR free leaf ${leaf}`);
    }
    const existing = topology.bricks.find(brick => brick.id === leaf);
    if (existing && active(existing)) {
      throw new Error(`slice WDR free leaf ${leaf} is still active`);
    }
    freeSet.add(leaf);
  }
  // Released leaves remain in the prior accepted HTP through presentation,
  // but no longer belong to the next candidate graph.
  const workingBricks: SliceTopologyBrick[] = topology.bricks
    .filter(brick => !freeSet.has(brick.id));
  const allocated: SliceTopologyBrick[] = [];
  const allocatedResolution = new Map<number, SliceTopologyResolution>();
  const claimedLeafIds: number[] = [];
  if (options.allocateMissingPages !== false) {
    let nextId = topology.bricks.reduce((maximum, brick) => Math.max(maximum, brick.id), -1) + 1;
    const freeStack = [...freeLeafIds];
    let nextKey = workingBricks.reduce((maximum, brick) => Math.max(maximum, brick.key), -1) + 1;
    const allocate = (qx: number, qy: number, resolution: SliceTopologyResolution) => {
      const fineX = qx * B + 0.5, fineY = qy * B + 0.5;
      if (fineX < 0 || fineY < 0 || fineX >= topology.dimensions[0]
        || fineY >= topology.dimensions[1]) return undefined;
      const existing = ownerAt(workingBricks, fineX, fineY);
      if (existing) return existing;
      const id = freeStack.length ? freeStack.pop()! : nextId++;
      claimedLeafIds.push(id);
      const page: SliceTopologyBrick = { id, key: nextKey++,
        coordinate: [qx, qy], resolution, active: true };
      workingBricks.push(page); allocated.push(page);
      allocatedResolution.set(page.key, resolution);
      measurements.set(page.key, { scoreByte: 0, reasons: 0, hotEpochs: 0,
        quietEpochs: 0, proofEpochs: 0, meanDensity: 0, densityMoments: [0, 0],
        meanVelocity: [0, 0], velocityTravel: 0, supportMask: 0, sweptSupportMask: 0,
        lastTransitionStep: acceptedSteps, surface: false, thin: false, occupied: false,
        detail: 0, deeplyEnclosed: false, curvatureFloor: 1 });
      return page;
    };
    for (const source of [...topology.bricks].sort((a, b) => a.key - b.key)) {
      if (!active(source)) continue;
      const mask = measurements.get(source.key)!.sweptSupportMask;
      for (let bit = 0; bit < 9; bit++) {
        if (bit === 4 || (mask & (1 << bit)) === 0) continue;
        const dx = bit % 3 - 1, dy = Math.floor(bit / 3) - 1, span = spanOf(source);
        const qx = source.coordinate[0] + (dx < 0 ? -1 : dx > 0 ? span : 0);
        const qy = source.coordinate[1] + (dy < 0 ? -1 : dy > 0 ? span : 0);
        const receiver = allocate(qx, qy, 8);
        if (receiver) materialDemand.add(receiver.key);
      }
    }
  }

  const targets = new Map<number, SliceTopologyResolution>();
  const candidateActive = new Map<number, boolean>();
  const planReasons = new Map<number, number>();
  for (const brick of topology.bricks) {
    const current = brick.resolution, m = measurements.get(brick.key)!;
    let requested = current, reason = 32;
    const frozen = policy.freezeTopology || options.frozenBrickKeys?.has(brick.key);
    if (!active(brick) || frozen) {
      targets.set(brick.key, current); candidateActive.set(brick.key, active(brick));
      planReasons.set(brick.key, frozen ? 32 : 128); continue;
    }
    const measuredFloor = velocityFloor(m.velocityTravel, thresholds, policy.activitySignals);
    const staticFloor = options.staticBoundaryFloorByBrick?.get(brick.key) ?? 1;
    const movingBoundaryFloor = options.movingRigidBodies && (m.reasons & SliceActivityReason.CutBoundary)
      ? 4 : 1;
    const enclosed = policy.activitySignals && m.deeplyEnclosed && measuredFloor === 1;
    const surface = m.surface && !enclosed;
    const slowSurface = surface && !m.thin && measuredFloor === 1;
    const next = resolution(Math.max(1, current / 2));
    const touchesLiquid = faceNeighbors(topology.bricks, brick)
      .some(neighbor => active(neighbor) && measurements.get(neighbor.key)!.occupied);
    const pageDemand = options.injectionDemandedBrickKeys?.has(brick.key) === true
      || (touchesLiquid && (!m.occupied || m.meanDensity < policy.surfaceDensityMinimum));
    const required = enclosed ? Math.max(staticFloor, movingBoundaryFloor)
      : Math.max(measuredFloor, surface ? next : 1, m.thin || pageDemand ? 8 : 1,
        staticFloor, movingBoundaryFloor);
    const score = m.scoreByte, emergency = Math.round(255 * policy.emergencyScore);
    if (required > current || (!surface && !enclosed && !slowSurface && score >= emergency)) {
      const urgent = m.thin || pageDemand;
      requested = resolution(urgent ? required : Math.min(8, Math.max(required, 2 * current)));
      reason = pageDemand || (m.reasons & SliceActivityReason.PredictedFace) ? 2
        : m.thin ? 256 : measuredFloor > current ? 64 : 4;
    } else if (topologyEpoch) {
      if (surface && current > 1) {
        const proofGeneration = m.surfaceProof?.generationByTargetResolution.get(next);
        const proofFresh = policy.surfaceCoarseningEnabled && proofGeneration === topology.generation;
        const threshold = 0.5 * (thresholds.get(current)! + thresholds.get(next)!);
        const proofEpochs = proofFresh && required <= next && m.velocityTravel < threshold
          ? Math.min(255, m.proofEpochs + 1) : 0;
        measurements.set(brick.key, { ...m, proofEpochs });
        if (proofEpochs >= Math.max(policy.demoteEpochs, 8)) {
          requested = next; reason = 16;
        }
      } else if (!enclosed && !slowSurface && m.hotEpochs >= policy.promoteEpochs) {
        requested = resolution(Math.min(8, 2 * current)); reason = 8;
      } else if (current > required && (enclosed || slowSurface || m.quietEpochs >= policy.demoteEpochs)
        && !(m.reasons & SliceActivityReason.FineDetail)) {
        requested = resolution(enclosed ? required : Math.max(required, current / 2));
        reason = enclosed ? 2048 : 16;
      }
    }
    if (policy.coarseFirst && policy.forcedSurfaceResolutionForQA === undefined) {
      const geometryFloor = Math.max(m.curvatureFloor, staticFloor, movingBoundaryFloor);
      const incoming = Math.max(geometryFloor, measuredFloor) < 8 && !m.thin
        && !options.injectionDemandedBrickKeys?.has(brick.key) && (surface || pageDemand)
        ? incomingFloor(topology, brick, measurements, policy) : 1;
      // Remote approach retains resolution that local evidence established;
      // it cannot create a finer accepted grid on a calm receiver by itself.
      const incomingRetentionFloor = Math.min(current, incoming);
      const frontierBoundary = [3, 5, 1, 7].some(bit => {
        if (((m.supportMask | m.sweptSupportMask) & (1 << bit)) === 0) return false;
        const dx = bit % 3 - 1, dy = Math.floor(bit / 3) - 1;
        const q = ownerAt(topology.bricks,
          (brick.coordinate[0] + dx) * B + 0.5,
          (brick.coordinate[1] + dy) * B + 0.5);
        return !q || !active(q);
      });
      const movingFrontier = (frontierBoundary || pageDemand) && measuredFloor > 1;
      const airFloor = 1;
      const safetyFloor = m.thin || options.injectionDemandedBrickKeys?.has(brick.key)
        || movingFrontier ? 8 : airFloor;
      const coarseRequired = Math.max(geometryFloor, measuredFloor,
        incomingRetentionFloor, safetyFloor);
      requested = current; reason = 32;
      if (coarseRequired > current) {
        requested = resolution(coarseRequired); reason = 4;
        measurements.set(brick.key, { ...m, proofEpochs: 0 });
      } else if (coarseRequired < current) {
        const next = resolution(Math.max(1, current / 2));
        const proofFresh = !surface || m.surfaceProof?.generationByTargetResolution.get(next) === topology.generation;
        let proofEpochs = proofFresh ? m.proofEpochs : 0;
        if (proofFresh && topologyEpoch) proofEpochs = Math.min(255, proofEpochs + 1);
        measurements.set(brick.key, { ...m, proofEpochs });
        if (proofEpochs >= policy.surfaceQuietEpochs) {
          requested = resolution(Math.max(coarseRequired, current / 2)); reason = 16;
        }
      } else measurements.set(brick.key, { ...m, proofEpochs: 0 });
    }
    requested = applyRegionBounds(requested, brick, regions);
    targets.set(brick.key, requested); candidateActive.set(brick.key, true);
    planReasons.set(brick.key, reason);
  }
  for (const brick of allocated) {
    targets.set(brick.key, allocatedResolution.get(brick.key)!);
    candidateActive.set(brick.key, true);
    planReasons.set(brick.key, 0x80000001);
  }

  // Directional support activates only already allocated sparse leaves. The
  // 3x3 mask is the exact Z-removed form of the resident 3x3x3 mask.
  for (const brick of topology.bricks) if (!active(brick)) {
    const demanded = materialDemand.has(brick.key);
    if (demanded) {
      candidateActive.set(brick.key, true); targets.set(brick.key,
        applyRegionBounds(8, brick, regions)); planReasons.set(brick.key, 0x80000001);
    }
  }

  // Retirement uses exact stored zero, never the residency threshold.
  for (const brick of topology.bricks) if (active(brick)) {
    const m = measurements.get(brick.key)!;
    const cells = topology.cells.filter(cell => cell.brickKey === brick.key);
    const exactEmpty = cells.every(cell => fields.density[cell.id] === 0);
    const movingGeometrySupport = options.movingRigidBodies === true
      && (m.reasons & SliceActivityReason.CutBoundary) !== 0;
    if (exactEmpty && !m.occupied && !materialDemand.has(brick.key)
      && !movingGeometrySupport && !options.injectionDemandedBrickKeys?.has(brick.key)) {
      candidateActive.set(brick.key, false); planReasons.set(brick.key, 0x80000000);
    }
  }

  // Production closes every allocated leaf's planned/accepted rung before
  // candidate membership validation. An inactive neighbour still constrains
  // this pass even though validation later ignores its physical face.
  closeRegionGradingCaps(workingBricks, targets, regions);
  closeTwoToOne(workingBricks, targets, regions);
  let faultBits = SliceResolutionFault.None;
  for (const brick of workingBricks) for (const neighbor of faceNeighbors(workingBricks, brick)) {
    if (!candidateActive.get(brick.key) || !candidateActive.get(neighbor.key)) continue;
    const a = widthOf(brick, targets.get(brick.key)!), b = widthOf(neighbor, targets.get(neighbor.key)!);
    if (Math.max(a, b) > 2 * Math.min(a, b)) faultBits |= SliceResolutionFault.TwoToOne;
  }

  // Production schedules every promotion/lifecycle change urgently, then a
  // rotating key-order window of ordinary demotions.
  const changed = workingBricks.filter(brick => allocated.includes(brick)
    || targets.get(brick.key) !== brick.resolution
    || candidateActive.get(brick.key) !== active(brick));
  const urgent = changed.filter(brick => targets.get(brick.key)! > brick.resolution
    || candidateActive.get(brick.key) !== active(brick)
    || regionBounds(brick, regions).maximumResolution < brick.resolution);
  const ordinary = changed.filter(brick => !urgent.includes(brick) && targets.get(brick.key)! < brick.resolution);
  const ordered = [...workingBricks].sort((a, b) => a.key - b.key);
  const rotated = ordered.slice(previous.schedulingCursor % Math.max(1, ordered.length))
    .concat(ordered.slice(0, previous.schedulingCursor % Math.max(1, ordered.length)));
  const budget = policy.prepareBricksPerFrame;
  // A reflected 2-D orbit has at most two members. Retain only the one spare
  // credit needed for a budget-one window to admit that pair on a later pass.
  const creditBudget = Math.min(budget+1,(previous.schedulingCredits ?? 0)+budget);
  const ordinarySet = new Set(ordinary.map(brick => brick.key));
  const admittedDemotions = new Set<number>();
  const reflected = (brick: SliceTopologyBrick) => {
    const box = bounds(brick), reflectedLoX = topology.dimensions[0] - box.hi[0];
    return workingBricks.find(candidate => {
      const candidateBox = bounds(candidate);
      return candidateBox.lo[0] === reflectedLoX
        && candidateBox.lo[1] === box.lo[1]
        && candidateBox.hi[0] - candidateBox.lo[0] === box.hi[0] - box.lo[0]
        && candidateBox.hi[1] === box.hi[1];
    });
  };
  const considered = new Set<number>();
  for (const brick of rotated) {
    if (!ordinarySet.has(brick.key) || considered.has(brick.key)) continue;
    const mirror = reflected(brick);
    const paired = mirror && mirror.key !== brick.key && ordinarySet.has(mirror.key)
      && mirror.resolution === brick.resolution
      && targets.get(mirror.key) === targets.get(brick.key);
    const orbit = paired ? [brick, mirror] : [brick];
    for (const member of orbit) considered.add(member.key);
    if (admittedDemotions.size + orbit.length > creditBudget) continue;
    for (const member of orbit) admittedDemotions.add(member.key);
  }
  for (const brick of ordinary) if (!admittedDemotions.has(brick.key)) targets.set(brick.key, brick.resolution);

  let candidateBricks: SliceTopologyBrick[] = workingBricks.map(brick => ({ ...brick,
    resolution: targets.get(brick.key)!, active: candidateActive.get(brick.key)! }));
  const leafCount = candidateBricks.filter(active).length;
  const cellCount = candidateBricks.filter(active).reduce((sum, brick) => sum + brick.resolution ** 2, 0);
  if (leafCount > (options.maximumLeaves ?? Infinity)) faultBits |= SliceResolutionFault.LeafCapacity;
  if (cellCount > (options.maximumCells ?? Infinity)) faultBits |= SliceResolutionFault.CellCapacity;
  if (faultBits !== SliceResolutionFault.None) candidateBricks = topology.bricks.map(brick => ({ ...brick }));
  const publishedChange = faultBits === SliceResolutionFault.None && (
    candidateBricks.length !== topology.bricks.length
    || candidateBricks.some(brick => {
      const acceptedBrick = topology.brickByKey.get(brick.key);
      return !acceptedBrick || brick.resolution !== acceptedBrick.resolution
        || active(brick) !== active(acceptedBrick);
    }));

  const nextHistory = new Map<number, SliceBrickActivityHistory>();
  const records: SliceBrickResolutionReceipt[] = [];
  let promoted = 0, demoted = 0, activated = 0, retired = 0, maximumScoreByte = 0;
  for (const brick of workingBricks) {
    const m = measurements.get(brick.key)!;
    const requested = targets.get(brick.key)!;
    const scheduledBrick = candidateBricks.find(candidate => candidate.key === brick.key);
    if (!scheduledBrick) continue;
    const wasAccepted = !allocated.includes(brick) && active(brick);
    const didChange = allocated.includes(brick) || scheduledBrick.resolution !== brick.resolution
      || active(scheduledBrick) !== active(brick);
    const history = didChange ? { ...m, hotEpochs: 0, quietEpochs: 0,
      proofEpochs: 0, lastTransitionStep: acceptedSteps } : m;
    nextHistory.set(brick.key, history); maximumScoreByte = Math.max(maximumScoreByte, m.scoreByte);
    if (scheduledBrick.resolution > brick.resolution) promoted++;
    if (scheduledBrick.resolution < brick.resolution) demoted++;
    if (!wasAccepted && active(scheduledBrick)) activated++;
    if (wasAccepted && !active(scheduledBrick)) retired++;
    records.push({ brickKey: brick.key, acceptedResolution: brick.resolution,
      requestedResolution: requested, scheduledResolution: scheduledBrick.resolution,
      acceptedActive: wasAccepted, candidateActive: active(scheduledBrick),
      scoreByte: m.scoreByte, reasons: m.reasons,
      planReasons: planReasons.get(brick.key)!, supportMask: m.supportMask,
      sweptSupportMask: m.sweptSupportMask, faultBits });
  }
  return { candidateBricks, state: { acceptedSteps,
    acceptedGeneration: topology.generation,
    schedulingCursor: ordered.length ? (previous.schedulingCursor + Math.min(budget, ordinary.length)) % ordered.length : 0,
    schedulingCredits: creditBudget-admittedDemotions.size,
    history: nextHistory }, receipt: { topologyEpoch,
      acceptedGeneration: topology.generation,
      candidateGeneration: publishedChange ? topology.generation + 1 : topology.generation,
      measuredBrickCount: topology.bricks.length,
      surfaceBrickCount: [...measurements.values()].filter(value => value.surface).length,
      occupiedBrickCount: [...measurements.values()].filter(value => value.occupied).length,
      activatedBrickCount: activated, allocatedBrickCount: allocated.length, retiredBrickCount: retired,
      claimedLeafIds: Uint32Array.from(claimedLeafIds),
      promotedBrickCount: promoted, demotedBrickCount: demoted,
      deferredDemotionCount: ordinary.length - admittedDemotions.size,
      maximumScoreByte, faultBits, bricks: records } };
}
