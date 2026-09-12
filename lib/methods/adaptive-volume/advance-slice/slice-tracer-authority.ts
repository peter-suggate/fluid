import { CM12_DRY_CELL_THRESHOLD, CM12_LIQUID_ISOVALUE } from "../../../core/cm12-numerics";
import { sparseCM12TracerLattice, type SparseCM12TracerLattice } from "../webgpu-sparse-cm12-resident";
import { sliceNumericalOwnerAt, traceSliceEffectiveTransportArrival,
  type SliceNumericalFields, type SliceNumericalTopology } from "./slice-stage-numerics";

const f = Math.fround;

export interface SliceTracerAuthority {
  readonly lattice: SparseCM12TracerLattice;
  /** Literal vec4 state: x, y, centre-slice z, live. */
  readonly state: Float32Array;
  readonly enabled: boolean;
  readonly seedPending: boolean;
  readonly generation: number;
}

export interface SliceTracerReceipt {
  readonly generation: number;
  readonly seeded: boolean;
  readonly count: number;
  readonly liveCount: number;
  readonly retiredCount: number;
}

export function createSliceTracerAuthority(dimensions: readonly [number, number],
  budget?: number): SliceTracerAuthority {
  const lattice = sparseCM12TracerLattice([dimensions[0], dimensions[1], 1], budget);
  return { lattice, state: new Float32Array(4 * lattice.count), enabled: false,
    seedPending: false, generation: 0 };
}

export function setSliceTracersEnabled(authority: SliceTracerAuthority,
  enabled: boolean): SliceTracerAuthority {
  if (authority.enabled === enabled) return authority;
  return { ...authority, enabled, seedPending: enabled };
}

export function reseedSliceTracers(authority: SliceTracerAuthority): SliceTracerAuthority {
  return authority.enabled ? { ...authority, seedPending: true } : authority;
}

function seedPosition(lattice: SparseCM12TracerLattice, index: number): readonly [number, number] {
  const x = index % Math.max(1, lattice.dimensions[0]);
  const y = Math.floor(index / Math.max(1, lattice.dimensions[0]))
    % Math.max(1, lattice.dimensions[1]);
  return [f(lattice.originFine[0] + (x + 0.5) * lattice.spacingFine),
    f(lattice.originFine[1] + (y + 0.5) * lattice.spacingFine)];
}

export function advanceSliceTracerAuthority(authority: SliceTracerAuthority,
  topology: SliceNumericalTopology, fields: SliceNumericalFields,
  dt: number): readonly [SliceTracerAuthority, SliceTracerReceipt] {
  if (!authority.enabled || authority.lattice.count === 0) {
    return [authority, { generation: authority.generation, seeded: false,
      count: authority.lattice.count, liveCount: 0, retiredCount: 0 }];
  }
  const state = authority.state.slice();
  const seeded = authority.seedPending;
  if (seeded) for (let index = 0; index < authority.lattice.count; index += 1) {
    const at = 4 * index, position = seedPosition(authority.lattice, index);
    const cell = sliceNumericalOwnerAt(topology, Math.floor(position[0]) + 0.5,
      Math.floor(position[1]) + 0.5);
    state[at] = position[0]; state[at + 1] = position[1]; state[at + 2] = 0.5;
    state[at + 3] = cell >= 0 && fields.density[cell]! > CM12_LIQUID_ISOVALUE ? 1 : 0;
  }
  let liveCount = 0, retiredCount = 0;
  // Production intentionally advances freshly seeded tracers in the same pass.
  for (let index = 0; index < authority.lattice.count; index += 1) {
    const at = 4 * index;
    if (state[at + 3]! < 0.5) continue;
    const traced = traceSliceEffectiveTransportArrival(topology, fields,
      [state[at]!, state[at + 1]!], dt);
    const cell = sliceNumericalOwnerAt(topology, Math.floor(traced[0]) + 0.5,
      Math.floor(traced[1]) + 0.5);
    state[at] = f(traced[0]); state[at + 1] = f(traced[1]);
    if (cell < 0 || fields.density[cell]! < CM12_DRY_CELL_THRESHOLD) {
      state[at + 3] = 0; retiredCount += 1;
    } else liveCount += 1;
  }
  const next = { ...authority, state, seedPending: false,
    generation: authority.generation + 1 };
  return [next, { generation: next.generation, seeded, count: authority.lattice.count,
    liveCount, retiredCount }];
}

