import type { SliceNumericalTopology } from "./slice-stage-numerics";

const f = Math.fround;
const EPSILON_F32 = 1.1920928955078125e-7;

/** CPU image of the production geometric-source ledger's persistent lanes. */
export interface SliceSourceLedger {
  readonly pending: number;
  readonly requested: number;
  readonly emitted: number;
  readonly available: number;
  readonly factor: number;
  readonly eventRequested: number;
  readonly eventEmitted: number;
  readonly fault: number;
  readonly requestedCompensation: number;
  readonly emittedCompensation: number;
  readonly eventBalanceResidual: number;
  readonly eventPendingBefore: number;
  readonly pendingCompensation: number;
  readonly continuousPlannedRate: number;
}

export const EMPTY_SLICE_SOURCE_LEDGER: SliceSourceLedger = Object.freeze({
  pending: 0, requested: 0, emitted: 0,
  available: 0, factor: 0, eventRequested: 0, eventEmitted: 0, fault: 0,
  requestedCompensation: 0, emittedCompensation: 0, pendingCompensation: 0,
  eventBalanceResidual: 0, eventPendingBefore: 0, continuousPlannedRate: 0,
});

export interface SliceDynamicRemapInput {
  readonly topology: SliceNumericalTopology;
  readonly density: ArrayLike<number>;
  readonly capacityBefore: ArrayLike<number>;
  readonly capacityAfter: ArrayLike<number>;
  readonly capacityRate: ArrayLike<number>;
  /** Extensive source area rate, finest-cell squared/s, aligned to cells. */
  readonly sourceRate: ArrayLike<number>;
  readonly dt: number;
  readonly pendingSourceAreaFine: number;
  /** Requested external dose rate. Defaults to the currently planned rate. */
  readonly requestedSourceAreaRateFine?: number;
  readonly sourceAvailableAreaFine?: number;
  readonly sourceFactor?: number;
  readonly continuousPlannedRate?: number;
  readonly ledger?: SliceSourceLedger;
}

export interface SliceDynamicRemapFault {
  readonly stage: "dynamic-input" | "source-ledger";
  readonly index: number;
  readonly observed: number;
  readonly expected: number;
}

export interface SliceDynamicRemapReceipt {
  readonly volumeBefore: number;
  readonly volumeAfterGeometry: number;
  readonly capacityBefore: number;
  readonly capacityAfter: number;
  readonly closingCapacity: number;
  readonly openingCapacity: number;
  readonly excessAfterGeometry: number;
  readonly requestedArea: number;
  readonly plannedArea: number;
  readonly balanceResidual: number;
}

export interface SliceDynamicRemapResult {
  /** Production GCL changes capacity and pressure RHS, never clamps rho here. */
  readonly density: Float32Array;
  readonly capacity: Float32Array;
  readonly capacityRate: Float32Array;
  readonly sourceRate: Float32Array;
  readonly pressureCapacityRate: Float32Array;
  readonly pressureSourceRate: Float32Array;
  readonly ledger: SliceSourceLedger;
  readonly receipt: SliceDynamicRemapReceipt;
  readonly fault: SliceDynamicRemapFault | null;
}

/** FastTwoSum image used by the staged production GPU ledger. */
export function addSliceSourceCompensated(value: number, total: number,
  compensation: number): readonly [number, number] {
  const increment = f(value - compensation);
  const next = f(total + increment);
  const error = Math.abs(total) >= Math.abs(increment)
    ? f(f(total - next) + increment)
    : f(f(increment - next) + total);
  return [next, f(-error)];
}

function sumExtensive(topology: SliceNumericalTopology, scalar: ArrayLike<number>): number {
  let total = 0, correction = 0;
  for (const cell of topology.cells) {
    const value = scalar[cell.id]! * cell.area - correction;
    const next = total + value;
    correction = (next - total) - value;
    total = next;
  }
  return total;
}

/**
 * Freeze moving-capacity and source terms for one outer step.
 *
 * Production does not redistribute or clamp density when a solid closes.
 * `-dC/dt + S` enters the pressure RHS and the projected wall-relative flux
 * supplies the GCL displacement. Keeping rho bitwise here is intentional,
 * including transient rho>K states and their ordinary later repair path.
 */
export function planSliceDynamicRemap(input: SliceDynamicRemapInput): SliceDynamicRemapResult {
  const count = input.topology.cells.length;
  if (!(input.dt > 0) || !Number.isFinite(input.dt)) {
    throw new RangeError("slice dynamic remap dt must be finite and positive");
  }
  const arrays = [input.density, input.capacityBefore, input.capacityAfter,
    input.capacityRate, input.sourceRate];
  if (arrays.some(array => array.length !== count)) {
    throw new RangeError("slice dynamic remap fields must align with compact cells");
  }
  const density = Float32Array.from(input.density);
  const capacity = Float32Array.from(input.capacityAfter);
  const capacityRate = Float32Array.from(input.capacityRate);
  const sourceRate = Float32Array.from(input.sourceRate);
  const pressureCapacityRate = new Float32Array(count);
  const pressureSourceRate = new Float32Array(count);
  let fault: SliceDynamicRemapFault | null = null;
  let closingCapacity = 0, openingCapacity = 0, excess = 0, planned = 0;
  for (const cell of input.topology.cells) {
    const id = cell.id, area = cell.area;
    const before = input.capacityBefore[id]!, after = capacity[id]!, rate = capacityRate[id]!;
    if (![density[id], before, after, rate, sourceRate[id]].every(Number.isFinite)) {
      fault ??= { stage: "dynamic-input", index: id,
        observed: Number.NaN, expected: 0 };
    }
    const delta = (after - before) * area;
    if (delta < 0) closingCapacity -= delta; else openingCapacity += delta;
    excess += Math.max(0, density[id]! - after) * area;
    pressureCapacityRate[id] = f(rate * area);
    pressureSourceRate[id] = sourceRate[id]!;
    planned += sourceRate[id]! * input.dt;
  }

  const prior = input.ledger ?? { ...EMPTY_SLICE_SOURCE_LEDGER,
    pending: input.pendingSourceAreaFine };
  const requestedArea = f((input.requestedSourceAreaRateFine
    ?? sourceRate.reduce((sum, rate) => sum + rate, 0)) * input.dt);
  const [pending, pendingCompensation] = addSliceSourceCompensated(requestedArea,
    prior.pending, prior.pendingCompensation);
  const [requested, requestedCompensation] = addSliceSourceCompensated(requestedArea,
    prior.requested, prior.requestedCompensation);
  const ledger = { ...prior, pending, requested, pendingCompensation,
    requestedCompensation,
    available: f(input.sourceAvailableAreaFine ?? 0),
    factor: f(input.sourceFactor ?? 0),
    eventRequested: requestedArea, eventEmitted: 0, fault: 0,
    eventBalanceResidual: 0, eventPendingBefore: prior.pending,
    continuousPlannedRate: f(input.continuousPlannedRate
      ?? sourceRate.reduce((sum, rate) => f(sum + rate), 0)),
  };
  const volume = sumExtensive(input.topology, density);
  return {
    density, capacity, capacityRate, sourceRate, pressureCapacityRate,
    pressureSourceRate, ledger,
    receipt: { volumeBefore: volume, volumeAfterGeometry: volume,
      capacityBefore: sumExtensive(input.topology, input.capacityBefore),
      capacityAfter: sumExtensive(input.topology, input.capacityAfter),
      closingCapacity, openingCapacity, excessAfterGeometry: excess,
      requestedArea, plannedArea: planned, balanceResidual: 0 },
    fault,
  };
}

export interface SliceSourceCommitResult {
  readonly ledger: SliceSourceLedger;
  readonly emittedArea: number;
  readonly balanceResidual: number;
  readonly fault: SliceDynamicRemapFault | null;
}

/** Commit the already-frozen source rate after one transport microstep. */
export function commitSliceSourceLedger(ledger: SliceSourceLedger,
  _sourceRate: ArrayLike<number>, dt: number): SliceSourceCommitResult {
  if (!(dt > 0) || !Number.isFinite(dt)) {
    throw new RangeError("slice source ledger dt must be finite and positive");
  }
  // geometricSourceCommitMicrostep consumes the frozen two-level reduction
  // stored in ledger lane 13. Re-reducing cells here changes f32 ordering and
  // lets later field mutation alter the transaction after it was planned.
  const emittedArea = f(ledger.continuousPlannedRate * dt);
  const before = ledger.pending;
  const [pending, pendingCompensation] = addSliceSourceCompensated(-emittedArea,
    before, ledger.pendingCompensation);
  const [emitted, emittedCompensation] = addSliceSourceCompensated(emittedArea,
    ledger.emitted, ledger.emittedCompensation);
  const residual = f(f(pending + emittedArea) - before);
  const valid = pending >= -8 * EPSILON_F32 * Math.max(Math.abs(before), Math.abs(emittedArea));
  return { ledger: { ...ledger, pending, emitted, pendingCompensation,
    emittedCompensation, eventEmitted: f(ledger.eventEmitted + emittedArea),
    eventBalanceResidual: residual, fault: valid ? ledger.fault : 2 },
    emittedArea, balanceResidual: residual,
    fault: valid ? null : { stage: "source-ledger", index: -1,
      observed: pending, expected: 0 } };
}
