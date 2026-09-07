/** Experimental retained C0 positive density basis, independent of physics cells.
 * Equal-width conforming sparse support is deliberate: hanging faces need a
 * separate constrained trace compiler. GPU ownership and native topology
 * coupling are separate from this immutable algebra module.
 */
export type Point3 = readonly [number, number, number];
export interface BernsteinSupportBox { readonly lower: Point3; readonly width: number }
export interface BernsteinSupport {
  readonly generation: number;
  readonly boxes: readonly BernsteinSupportBox[];
  /** 27 tensor quadratic Bernstein control IDs per support cell, x fastest. */
  readonly cellControls: readonly (readonly number[])[];
  readonly positions: readonly Point3[];
  readonly incidentCells: readonly (readonly number[])[];
  readonly interior: readonly boolean[];
}
export interface PositiveBernsteinField {
  readonly generation: number;
  readonly support: BernsteinSupport;
  readonly controls: readonly number[];
}
const index = (x: number, y: number, z: number) => x + 3 * y + 9 * z;

export function compileBernsteinSupport(boxes: readonly BernsteinSupportBox[], generation = 1): BernsteinSupport {
  if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("Invalid support generation");
  if (!boxes.length) throw new Error("Empty Bernstein support");
  const width = boxes[0].width;
  if (!(width > 0) || !Number.isFinite(width)) throw new Error("Invalid support width");
  const ids = new Map<string, number>(), occupied = new Set<string>();
  const positions: Point3[] = [], incidentCells: number[][] = [], interior: boolean[] = [];
  const cellControls = boxes.map((box, cell) => {
    if (box.width !== width || box.lower.some(q => !Number.isFinite(q) || !Number.isSafeInteger(2 * q / width) || !Number.isInteger(q / width) || !Number.isFinite(q + width) || q + width <= q)) {
      throw new Error("Support requires equal-width aligned conforming cells");
    }
    const cellKey = box.lower.join("/");
    if (occupied.has(cellKey)) throw new Error("Duplicate support cell");
    occupied.add(cellKey);
    const controls = new Uint32Array(27);
    for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
      const integer = [2 * box.lower[0] / width + x, 2 * box.lower[1] / width + y, 2 * box.lower[2] / width + z];
      if (integer.some(v => !Number.isSafeInteger(v))) throw new Error("Support control coordinate exceeds exact integer range");
      const key = integer.join("/");
      let id = ids.get(key);
      if (id === undefined) {
        id = positions.length; ids.set(key, id);
        positions.push(integer.map(q => q * width / 2) as unknown as Point3);
        incidentCells.push([]); interior.push(x === 1 && y === 1 && z === 1);
      }
      controls[index(x, y, z)] = id;
      incidentCells[id].push(cell);
    }
    return Object.freeze(Array.from(controls));
  });
  return Object.freeze({ generation,
    boxes: Object.freeze(boxes.map(b => Object.freeze({ lower: Object.freeze([...b.lower]) as unknown as Point3, width: b.width }))),
    cellControls: Object.freeze(cellControls),
    positions: Object.freeze(positions.map(p => Object.freeze(p))),
    incidentCells: Object.freeze(incidentCells.map(c => Object.freeze(c))),
    interior: Object.freeze(interior) });
}

export function positiveBernsteinField(support: BernsteinSupport, values: ArrayLike<number>, generation = 1): PositiveBernsteinField {
  if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("Invalid field generation");
  if (values.length !== support.positions.length) throw new Error("Wrong control count");
  const controls = Array.from(values);
  if (controls.some(v => !Number.isFinite(v) || v < 0)) throw new Error("Density controls must be finite and nonnegative");
  return Object.freeze({ support, generation, controls: Object.freeze(controls) });
}

const basis = (t: number) => [(1 - t) ** 2, 2 * t * (1 - t), t * t];
export function evaluateBernsteinCell(field: PositiveBernsteinField, cell: number, point: Point3): number {
  const box = field.support.boxes[cell];
  const b = point.map((v, axis) => basis((v - box.lower[axis]) / box.width));
  if (point.some((v, axis) => !Number.isFinite(v) || v < box.lower[axis] || v > box.lower[axis] + box.width)) {
    throw new Error("Query outside support cell");
  }
  const ids = field.support.cellControls[cell];
  let result = 0;
  for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
    result += field.controls[ids[index(x, y, z)]] * b[0][x] * b[1][y] * b[2][z];
  }
  return result;
}

/** Exact polynomial box integration in open space. Solids require actual
 * clipped basis moments; multiplying by scalar open fraction is not exact. */
export function integrateBernstein(field: PositiveBernsteinField, lower: Point3, upper: Point3): number {
  if (lower.some((v, axis) => !Number.isFinite(v) || !Number.isFinite(upper[axis]) || upper[axis] < v)) {
    throw new Error("Invalid integration box");
  }
  let amount = 0;
  for (let cell = 0; cell < field.support.boxes.length; cell++) {
    const box = field.support.boxes[cell], width = box.width;
    const lo = lower.map((v, axis) => Math.max(0, (v - box.lower[axis]) / width));
    const hi = upper.map((v, axis) => Math.min(1, (v - box.lower[axis]) / width));
    if (lo.some((v, axis) => v >= hi[axis])) continue;
    const integrals = lo.map((v, axis) => {
      const span = hi[axis] - v, middle = v + span / 2, variance = span * span / 12;
      return [(1 - middle) ** 2 + variance, 2 * (middle * (1 - middle) - variance), middle * middle + variance]
        .map(q => q * span * width);
    });
    const ids = field.support.cellControls[cell];
    for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
      amount += field.controls[ids[index(x, y, z)]] * integrals[0][x] * integrals[1][y] * integrals[2][z];
    }
  }
  return amount;
}

export function bernsteinCellMeans(field: PositiveBernsteinField): Float64Array {
  return Float64Array.from(field.support.cellControls, ids => ids.reduce((sum, id) => sum + field.controls[id], 0) / 27);
}

/** A conservative positive trace-compatible dynamics correction. All shared
 * boundary controls receive the minimum adjacent cell capacity factor. Each
 * private center control supplies the remaining amount. Proof: the boundary
 * sum for each cell cannot exceed 27*targetMean, all basis functions are
 * nonnegative, and every full-cell basis integral is volume/27.
 *
 * This is only an algebraically feasible correction, not a validated transport
 * model: large corrections can create subcell peaks and alter topology.
 * Never call it for a zero-time change of the physics partition.
 */
export function fitPositiveBernsteinMeans(predicted: PositiveBernsteinField, means: ArrayLike<number>, generation = predicted.generation + 1) {
  if (generation <= predicted.generation) throw new Error("Field update must advance its generation");
  const { support } = predicted;
  if (means.length !== support.boxes.length || Array.from(means).some(v => v < 0 || !Number.isFinite(v))) {
    throw new Error("Target means must be finite and nonnegative");
  }
  const capacity = new Float64Array(means.length).fill(1);
  for (let cell = 0; cell < means.length; cell++) {
    let boundarySum = 0;
    for (const id of support.cellControls[cell]) if (!support.interior[id]) boundarySum += predicted.controls[id];
    if (boundarySum > 0) capacity[cell] = Math.min(1, 27 * means[cell] / boundarySum);
  }
  const controls = predicted.controls.slice();
  for (let id = 0; id < controls.length; id++) if (!support.interior[id]) {
    let factor = 1;
    for (const cell of support.incidentCells[id]) factor = Math.min(factor, capacity[cell]);
    controls[id] *= factor;
  }
  let maximumInteriorChange = 0, maximumTraceChange = 0;
  for (let cell = 0; cell < means.length; cell++) {
    let boundarySum = 0;
    for (const id of support.cellControls[cell]) if (!support.interior[id]) boundarySum += controls[id];
    const interiorId = support.cellControls[cell][13];
    // max removes possible subtraction roundoff only, not a negative mass.
    controls[interiorId] = Math.max(0, 27 * means[cell] - boundarySum);
    maximumInteriorChange = Math.max(maximumInteriorChange, Math.abs(controls[interiorId] - predicted.controls[interiorId]));
  }
  for (let id = 0; id < controls.length; id++) if (!support.interior[id]) {
    maximumTraceChange = Math.max(maximumTraceChange, Math.abs(controls[id] - predicted.controls[id]));
  }
  return { field: positiveBernsteinField(support, controls, generation), maximumInteriorChange, maximumTraceChange };
}

/** Exact de Casteljau support refinement. This changes support generation but
 * preserves q, so its field generation remains the same. All support cells
 * refine together; local hanging-face refinement is deliberately unsupported.
 */
export function refinePositiveBernsteinField(field: PositiveBernsteinField,
  supportGeneration = field.support.generation + 1): PositiveBernsteinField {
  if (supportGeneration <= field.support.generation) throw new Error("Support refinement must advance its generation");
  const boxes: BernsteinSupportBox[] = [];
  for (const box of field.support.boxes) for (let child = 0; child < 8; child++) {
    boxes.push({ lower: box.lower.map((v, axis) => v + ((child >> axis) & 1) * box.width / 2) as unknown as Point3,
      width: box.width / 2 });
  }
  const support = compileBernsteinSupport(boxes, supportGeneration);
  const controls = new Float64Array(support.positions.length), assigned = new Uint8Array(controls.length);
  const matrices = [
    [[1, 0, 0], [0.5, 0.5, 0], [0.25, 0.5, 0.25]],
    [[0.25, 0.5, 0.25], [0, 0.5, 0.5], [0, 0, 1]],
  ];
  for (let parent = 0; parent < field.support.boxes.length; parent++) for (let child = 0; child < 8; child++) {
    const source = field.support.cellControls[parent], target = support.cellControls[8 * parent + child];
    const weights = [0, 1, 2].map(axis => matrices[(child >> axis) & 1]);
    for (let z = 0; z < 3; z++) for (let y = 0; y < 3; y++) for (let x = 0; x < 3; x++) {
      let value = 0;
      for (let k = 0; k < 3; k++) for (let j = 0; j < 3; j++) for (let i = 0; i < 3; i++) {
        value += field.controls[source[index(i, j, k)]] * weights[0][x][i] * weights[1][y][j] * weights[2][z][k];
      }
      const id = target[index(x, y, z)];
      if (assigned[id] && Math.abs(controls[id] - value) > 1e-12 * Math.max(1, value)) {
        throw new Error("Refinement produced incompatible shared traces");
      }
      if (!assigned[id]) { controls[id] = value; assigned[id] = 1; }
    }
  }
  return positiveBernsteinField(support, controls, field.generation);
}
