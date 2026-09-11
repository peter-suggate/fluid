/**
 * Host-side layout and bit-exact reference operations for the Phase-1
 * persistent effective-transport-velocity product.
 *
 * The GPU representation intentionally has no header: accepted template cell
 * ids are already slot-stable execution ordinals, so cell `i` addresses one
 * tightly packed vec4f at byte offset `16 * i`.
 */
export interface SparseCM12EffectiveTransportVelocityLayout {
  readonly cellCapacity: number;
  readonly vectorStrideBytes: 16;
  readonly floatCount: number;
  readonly byteLength: number;
}

const checkedCapacity = (value: number): number => {
  if (!Number.isSafeInteger(value) || value < 1
    || value > Math.floor(0xffff_ffff / 16)) {
    throw new RangeError("effective transport velocity cell capacity is invalid");
  }
  return value;
};

export function createSparseCM12EffectiveTransportVelocityLayout(
  cellCapacity: number,
): SparseCM12EffectiveTransportVelocityLayout {
  const capacity = checkedCapacity(cellCapacity);
  return Object.freeze({
    cellCapacity: capacity,
    vectorStrideBytes: 16 as const,
    floatCount: 4 * capacity,
    byteLength: 16 * capacity,
  });
}

/** Construction seed: copy the exact selector bits for every accepted cell. */
export function seedSparseCM12EffectiveTransportVelocity(
  layout: SparseCM12EffectiveTransportVelocityLayout,
  acceptedCells: Iterable<number>,
  select: (cell: number) => readonly [number, number, number, number],
): Float32Array {
  const values = new Float32Array(layout.floatCount);
  for (const cell of acceptedCells) {
    if (!Number.isSafeInteger(cell) || cell < 0 || cell >= layout.cellCapacity) {
      throw new RangeError(`effective velocity seed cell ${cell} is out of range`);
    }
    values.set(select(cell), 4 * cell);
  }
  return values;
}

/** VEX acceptance owns every cell in the exact blast, including wet roots. */
export function publishSparseCM12VexAcceptedVelocity(
  values: Float32Array,
  cell: number,
  value: readonly [number, number, number, number],
): void {
  const capacity = Math.floor(values.length / 4);
  if (values.length % 4 !== 0 || !Number.isSafeInteger(cell)
    || cell < 0 || cell >= capacity) {
    throw new RangeError(`effective velocity VEX cell ${cell} is out of range`);
  }
  values.set(value, 4 * cell);
}

/**
 * Collocation publishes only wet source values for the next VEX generation.
 * Dry cells deliberately retain their accepted effective value until the next
 * exact VEX blast either overwrites or reuses it.
 */
export function publishSparseCM12CollocatedWetVelocity(
  values: Float32Array,
  cell: number,
  velocity: readonly [number, number, number],
  wet: boolean,
): void {
  if (!wet) return;
  publishSparseCM12VexAcceptedVelocity(values, cell,
    [velocity[0], velocity[1], velocity[2], 1]);
}
