import {
  CM12_GHOST_FLUID_THETA_MIN,
  CM12_LIQUID_ISOVALUE,
  cm12GhostFluidTheta,
} from "../../../core/cm12-numerics";
import { sliceCellOwnerAt } from "./slice-cell-index";
import { solveSlicePressurePCG } from "./slice-pressure-pcg";

export type SliceAxis = 0 | 1;

export interface SliceNumericalCell {
  readonly id: number;
  /** Production template-cell id retained across accepted generations. */
  readonly stableId?: number;
  readonly minimum: readonly [number, number];
  readonly maximum: readonly [number, number];
  readonly center: readonly [number, number];
  readonly widths: readonly [number, number];
  readonly area: number;
  readonly brickKey?: number;
  readonly refinementRegionScale?: number;
}

export interface SliceNumericalTerm {
  readonly cellId: number;
  readonly coefficient: number;
}

export interface SliceNumericalRow {
  readonly id: number;
  readonly kind: "intra-brick" | "brick-face" | "mixed-seam" | "sparse-air" | "closed-world";
  readonly axis: SliceAxis;
  readonly center: readonly [number, number];
  readonly area: number;
  /** Immutable geometric row area before solid aperture. */
  readonly staticArea?: number;
  readonly distance: number;
  readonly dualWeight: number;
  /** Immutable geometric dual weight before pressure aperture. */
  readonly staticDualWeight?: number;
  readonly terms: readonly SliceNumericalTerm[];
  openFraction?: number;
  /** Moving-solid endpoint apertures; openFraction is their production mean. */
  openFractionBefore?: number;
  openFractionAfter?: number;
  solidVelocity?: number;
  /** Frame-local CM12 unilateral closed-world contact decision. */
  separating?: boolean;
}

export interface SliceNumericalSubface {
  readonly id: number;
  readonly rowId: number;
  readonly axis: SliceAxis;
  readonly center: readonly [number, number];
  readonly area: number;
  readonly negativeCell: number;
  readonly positiveCell: number;
  aperture?: number;
  solidVelocity?: number;
}

export interface SliceNumericalTopology {
  readonly dimensions?: readonly [number, number];
  readonly cells: readonly SliceNumericalCell[];
  readonly rows: readonly SliceNumericalRow[];
  readonly subfaces: readonly SliceNumericalSubface[];
  readonly incidences: readonly (readonly number[])[];
  /** Physical subfaces in resident GV_CELL_FACE entry order. */
  readonly subfaceIncidences?: readonly (readonly {
    readonly subfaceId: number;
    readonly negative: boolean;
  }[])[];
  /** Exact centre-Z SolidWorld Q8 >= 128 characteristic obstruction query. */
  readonly solidVoxelAt?: (x: number, y: number) => boolean;
  readonly solidVoxelFractionAt?: (x: number, y: number) => number;
}

export interface SliceNumericalFields {
  density: Float32Array;
  gamma: Float32Array;
  capacity: Float32Array;
  /** Moving-solid capacity endpoints. capacity aliases the final endpoint. */
  capacityBefore?: Float32Array;
  capacityAfter?: Float32Array;
  /** Mirrors production's rigid-coupling/GSM active gate for this frame. */
  solidMotionActive?: boolean;
  /** Frame controls used by pressure-only moving/source membership quirks. */
  frameDt?: number;
  accelerationFine?: readonly [number, number];
  /** Fraction/s; pressure multiplies by full cell area. */
  capacityRate?: Float32Array;
  /** Extensive unit-depth area/s in finest-cell units. */
  sourceRate?: Float32Array;
  inflowCoverage?: Float32Array;
  cellVelocity: Float32Array;
  faceVelocity: Float32Array;
  pressure: Float32Array;
  pressureRhs: Float32Array;
  pressureDiagonal: Float32Array;
  pressureMember: Uint8Array;
  pressureRowMember?: Uint8Array;
  extensionDepth: Uint8Array;
  interfaceNormal: Float32Array;
  interfaceOffset: Float32Array;
  lowFlux: Float32Array;
  highFlux: Float32Array;
  limitedFlux: Float32Array;
  /** VEX/CM12 departure-stencil bulk certificate in finest-cell units. */
  characteristicClearance?: Float32Array;
  fault: SliceNumericalFault | null;
}

export interface SliceNumericalFault {
  readonly stage: string;
  readonly index: number;
  readonly observed: number;
  readonly expected: number;
}

export interface SlicePressureReceipt {
  readonly iterations: number;
  readonly initialResidual: number;
  readonly residual: number;
  readonly converged: boolean;
}

const f = Math.fround;
const VOLUME_ROUNDOFF_RATIO = 9.5367431640625e-7;
const add = (a: number, b: number): number => f(f(a) + f(b));
const mul = (a: number, b: number): number => f(f(a) * f(b));
const div = (a: number, b: number): number => f(f(a) / f(b));
const physicalRowArea = (row: SliceNumericalRow): number =>
  f((row.staticArea ?? row.area) * (row.openFraction ?? 1));

function cellFill(fields: SliceNumericalFields, cell: number): number {
  return div(fields.density[cell]!, Math.max(fields.capacity[cell]!, 1e-8));
}

/** Exact 2-D reduction of cm12PhysicalSubfaceArea/rowDistance. */
function velocityExtensionNeighborWeight(row: SliceNumericalRow,
  own: SliceNumericalTerm, other: SliceNumericalTerm): number {
  if (own.coefficient * other.coefficient >= 0) return 0;
  const distance = f(row.distance);
  const ownFraction = mul(Math.abs(own.coefficient), distance);
  const otherFraction = mul(Math.abs(other.coefficient), distance);
  const subfaceArea = mul(mul(physicalRowArea(row), ownFraction), otherFraction);
  return div(subfaceArea, Math.max(distance, 1e-9));
}

/** VEX2: one f32 seed publication followed by eight frozen-front transforms. */
export function extendSliceVelocity(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  depthCount = 8,
): void {
  const count = topology.cells.length;
  fields.extensionDepth.fill(255);
  let velocity = Float32Array.from(fields.cellVelocity);
  let known = new Uint8Array(count);
  for (let cell = 0; cell < count; cell += 1) {
    if (fields.density[cell]! > CM12_LIQUID_ISOVALUE) {
      known[cell] = 1;
      fields.extensionDepth[cell] = 0;
    } else {
      velocity[2 * cell] = 0;
      velocity[2 * cell + 1] = 0;
    }
  }
  for (let depth = 1; depth <= depthCount; depth += 1) {
    const nextVelocity = Float32Array.from(velocity);
    const nextKnown = Uint8Array.from(known);
    for (const cell of topology.cells) {
      if (known[cell.id]) continue;
      const sideWeight = new Float32Array(4);
      const sideX = new Float32Array(4);
      const sideY = new Float32Array(4);
      for (const rowId of topology.incidences[cell.id] ?? []) {
        const row = topology.rows[rowId]!;
        if (physicalRowArea(row) <= 1e-8) continue;
        const own = row.terms.find((term) => term.cellId === cell.id);
        if (!own) continue;
        const side = 2 * row.axis + (own.coefficient < 0 ? 1 : 0);
        for (const term of row.terms) {
          if (term.cellId === cell.id || own.coefficient * term.coefficient >= 0
            || !known[term.cellId] || fields.extensionDepth[term.cellId]! >= depth) continue;
          const weight = velocityExtensionNeighborWeight(row, own, term);
          sideWeight[side] = add(sideWeight[side]!, weight);
          sideX[side] = add(sideX[side]!, mul(weight, velocity[2 * term.cellId]!));
          sideY[side] = add(sideY[side]!, mul(weight, velocity[2 * term.cellId + 1]!));
        }
      }
      // WGSL reduces each opposite side with min+max (componentwise) and then
      // adds axes. Preserve that tree even though finite positive values make
      // min+max algebraically equal to addition: NaNs and f32 rounding are ABI.
      const pair = (values: Float32Array, axis: 0 | 1): number => {
        const a = values[2 * axis]!, b = values[2 * axis + 1]!;
        return add(Math.min(a, b), Math.max(a, b));
      };
      const weight = add(pair(sideWeight, 0), pair(sideWeight, 1));
      const x = add(pair(sideX, 0), pair(sideX, 1));
      const y = add(pair(sideY, 0), pair(sideY, 1));
      if (weight > 0) {
        nextVelocity[2 * cell.id] = div(x, weight);
        nextVelocity[2 * cell.id + 1] = div(y, weight);
        nextKnown[cell.id] = 1;
        fields.extensionDepth[cell.id] = depth;
      }
    }
    velocity = nextVelocity;
    known = nextKnown;
  }
  fields.cellVelocity.set(velocity);
}

export function sliceNumericalOwnerAt(topology: SliceNumericalTopology,
  x: number, y: number): number {
  return sliceCellOwnerAt(topology, x, y);
}

const ownerAt = sliceNumericalOwnerAt;

interface FaceSupport {
  readonly velocity: readonly [number, number];
  readonly span: number;
  readonly owner: boolean;
  readonly extended: boolean;
  readonly liquid: boolean;
}

function faceSupportAt(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  x: number, y: number): FaceSupport {
  const cell = ownerAt(topology, Math.floor(x) + 0.5, Math.floor(y) + 0.5);
  if (cell < 0) {
    return { velocity: [0, 0], span: 1, owner: false, extended: false, liquid: false };
  }
  const record = topology.cells[cell]!;
  return { velocity: [fields.cellVelocity[2 * cell]!, fields.cellVelocity[2 * cell + 1]!],
    span: Math.max(1, Math.min(record.widths[0], record.widths[1])), owner: true,
    extended: fields.extensionDepth[cell]! !== 255,
    liquid: fields.density[cell]! > CM12_LIQUID_ISOVALUE };
}

function sampleFaceSupport(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  x: number, y: number, span: number): readonly [number, number] {
  const dimensions = topology.dimensions ?? [
    Math.max(0, ...topology.cells.map(cell => cell.maximum[0])),
    Math.max(0, ...topology.cells.map(cell => cell.maximum[1])),
  ];
  const bx = Math.max(0.5 * span, Math.min(dimensions[0] - 0.5 * span, x));
  const by = Math.max(0.5 * span, Math.min(dimensions[1] - 0.5 * span, y));
  const sx = bx / span - 0.5, sy = by / span - 0.5;
  const lx = Math.floor(sx), ly = Math.floor(sy), tx = f(sx - lx), ty = f(sy - ly);
  let vx = 0, vy = 0;
  for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
    const w = mul(dx ? tx : f(1 - tx), dy ? ty : f(1 - ty));
    if (w === 0) continue;
    const q = faceSupportAt(topology, fields,
      span * (lx + dx + 0.5), span * (ly + dy + 0.5));
    if (!q.owner) continue;
    vx = add(vx, mul(w, q.velocity[0])); vy = add(vy, mul(w, q.velocity[1]));
  }
  return [vx, vy];
}

function rowTouchesLiquid(row: SliceNumericalRow, fields: SliceNumericalFields): boolean {
  return row.terms.some(term => fields.density[term.cellId]! > CM12_LIQUID_ISOVALUE);
}

function rowSourceFluidVelocity(row: SliceNumericalRow, fields: SliceNumericalFields): number {
  const velocity = fields.faceVelocity[row.id]!;
  if (row.kind === "closed-world") return velocity;
  const open = row.openFraction ?? 1, wall = row.solidVelocity ?? 0;
  return open > 1e-6 ? div(velocity - mul(1 - open, wall), open) : wall;
}

/** Production sourceStaggeredCellSample with the omitted Z patch measure. */
function sourceStaggeredCellSample(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, point: readonly [number, number], axis: SliceAxis): readonly [number, number] {
  let cell = ownerAt(topology, Math.floor(point[0]) + 0.5, Math.floor(point[1]) + 0.5);
  if (cell < 0) {
    const q: [number, number] = [Math.floor(point[0]) + 0.5, Math.floor(point[1]) + 0.5];
    q[axis] -= 1; cell = ownerAt(topology, q[0], q[1]);
  }
  if (cell < 0 || fields.capacity[cell]! <= 1e-8) return [0, 0];
  const ownCell = topology.cells[cell]!, tangent = (1 - axis) as SliceAxis;
  const values = new Float32Array(2), weights = new Float32Array(2);
  for (const rowId of topology.incidences[cell] ?? []) {
    const row = topology.rows[rowId]!;
    if (row.axis !== axis || !rowTouchesLiquid(row, fields)) continue;
    if (fields.solidMotionActive && (row.openFraction ?? 1) < 1) continue;
    const own = row.terms.find(term => term.cellId === cell);
    if (!own) continue;
    const side = own.coefficient < 0 ? 1 : 0;
    let patch = 0;
    if (row.terms.length === 1) {
      const length = Math.abs(own.coefficient) * (row.staticDualWeight ?? row.dualWeight);
      const expectedAxis = side === 1 ? ownCell.maximum[axis] : ownCell.minimum[axis];
      if (length !== ownCell.widths[tangent] || row.center[tangent] !== ownCell.center[tangent]
        || row.center[axis] !== expectedAxis) return [0, 0];
      if (point[tangent] >= ownCell.minimum[tangent]
        && point[tangent] < ownCell.maximum[tangent]) patch = length;
    } else {
      for (const term of row.terms) {
        if (own.coefficient * term.coefficient >= 0) continue;
        const other = topology.cells[term.cellId]!;
        const lower = Math.max(ownCell.minimum[tangent], other.minimum[tangent]);
        const upper = Math.min(ownCell.maximum[tangent], other.maximum[tangent]);
        if (point[tangent] >= lower && point[tangent] < upper) patch = add(patch, upper - lower);
      }
    }
    if (patch > 0) {
      values[side] = add(values[side]!, mul(patch, rowSourceFluidVelocity(row, fields)));
      weights[side] = add(weights[side]!, patch);
    }
  }
  const fraction = Math.max(0, Math.min(1,
    (point[axis] - ownCell.minimum[axis]) / ownCell.widths[axis]));
  if (fraction === 0 && weights[0]! > 0) return [div(values[0]!, weights[0]!), 1];
  if (fraction === 1 && weights[1]! > 0) return [div(values[1]!, weights[1]!), 1];
  if (weights[0]! <= 0 || weights[1]! <= 0) return [0, 0];
  const low = div(values[0]!, weights[0]!), high = div(values[1]!, weights[1]!);
  return [add(mul(1 - fraction, low), mul(fraction, high)), 1];
}

function staggeredCoordinates(topology: SliceNumericalTopology,
  position: readonly [number, number], axis: SliceAxis, span: number) {
  const dimensions = topology.dimensions!;
  const offset: [number, number] = [0.5, 0.5]; offset[axis] = 0;
  const bounded: [number, number] = [0, 0];
  for (const a of [0, 1] as const) bounded[a] = Math.max(offset[a] * span,
    Math.min(dimensions[a] - offset[a] * span, position[a]));
  const shifted: [number, number] = [bounded[0] / span - offset[0], bounded[1] / span - offset[1]];
  const lower: [number, number] = [Math.floor(shifted[0]), Math.floor(shifted[1])];
  return { offset, lower, fraction: [f(shifted[0] - lower[0]), f(shifted[1] - lower[1])] as const };
}

function sampleSourceLinear(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  position: readonly [number, number], axis: SliceAxis, span: number): number {
  const c = staggeredCoordinates(topology, position, axis, span);
  let velocity = 0;
  for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
    const weight = mul(dx ? c.fraction[0] : f(1 - c.fraction[0]),
      dy ? c.fraction[1] : f(1 - c.fraction[1]));
    if (weight === 0) continue;
    const point: [number, number] = [span * (c.lower[0] + dx + c.offset[0]),
      span * (c.lower[1] + dy + c.offset[1])];
    const staggered = sourceStaggeredCellSample(topology, fields, point, axis);
    const value = staggered[1] !== 0 ? staggered[0]
      : sampleFaceSupport(topology, fields, point[0], point[1], span)[axis];
    velocity = add(velocity, mul(weight, value));
  }
  return velocity;
}

function uniformStaggeredNode(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  point: readonly [number, number], axis: SliceAxis, span: number): readonly [number, number] {
  const cell = ownerAt(topology, Math.floor(point[0]) + 0.5, Math.floor(point[1]) + 0.5);
  if (cell < 0 || topology.cells[cell]!.widths.some(width => width !== span)
    || fields.capacity[cell] !== 1) return [0, 0];
  const tangent = (1 - axis) as SliceAxis;
  let found = false, value = 0;
  for (const rowId of topology.incidences[cell] ?? []) {
    const row = topology.rows[rowId]!;
    if (row.axis !== axis || row.center[0] !== point[0] || row.center[1] !== point[1]) continue;
    if (row.terms.length !== 2 || (row.openFraction ?? 1) !== 1
      || (row.staticArea ?? row.area) !== span || row.distance !== span
      || !rowTouchesLiquid(row, fields)) return [0, 0];
    let negative = false, positive = false;
    for (const term of row.terms) {
      const endpoint = topology.cells[term.cellId]!;
      if (endpoint.widths.some(width => width !== span) || fields.capacity[term.cellId] !== 1) return [0, 0];
      const expected: [number, number] = [point[0], point[1]];
      if (term.coefficient < 0) { expected[axis] -= 0.5 * span; negative = true; }
      else if (term.coefficient > 0) { expected[axis] += 0.5 * span; positive = true; }
      else return [0, 0];
      if (endpoint.center[0] !== expected[0] || endpoint.center[1] !== expected[1]) return [0, 0];
    }
    if (!negative || !positive || found) return [0, 0];
    found = true; value = fields.faceVelocity[row.id]!;
  }
  void tangent;
  return [value, found ? 1 : 0];
}

function cubicLine(a: number, b: number, c: number, d: number, t: number): number {
  if (t === 0) return b; if (t === 1) return c;
  const inner = add(add(f(a - b), mul(2, f(c - b))), mul(-0.5, f(d - b)));
  const cubic = add(mul(1.5, f(b - c)), mul(0.5, f(d - a)));
  return add(b, mul(t, add(mul(0.5, f(c - a)), mul(t, add(inner, mul(t, cubic))))));
}

function sampleSource(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  position: readonly [number, number], axis: SliceAxis, span: number): number {
  const c = staggeredCoordinates(topology, position, axis, span), dimensions = topology.dimensions!;
  const interpolated = [c.fraction[0] !== 0, c.fraction[1] !== 0] as const;
  const first = [span * (c.lower[0] - 1 + c.offset[0]), span * (c.lower[1] - 1 + c.offset[1])];
  const last = [span * (c.lower[0] + 2 + c.offset[0]), span * (c.lower[1] + 2 + c.offset[1])];
  const lowerBound = [c.offset[0] * span, c.offset[1] * span];
  const upperBound = [dimensions[0] - lowerBound[0]!, dimensions[1] - lowerBound[1]!];
  if ((!interpolated[0] && !interpolated[1])
    || interpolated.some((yes, a) => yes && (first[a]! < lowerBound[a]! || last[a]! > upperBound[a]!))
    || (interpolated[axis] && (first[axis]! <= 0 || last[axis]! >= dimensions[axis]))) {
    return sampleSourceLinear(topology, fields, position, axis, span);
  }
  const values: number[][] = Array.from({ length: 4 }, () => new Array(4).fill(0));
  let coreMin = Number.MAX_VALUE, coreMax = -Number.MAX_VALUE;
  const xCount = interpolated[0] ? 4 : 1, yCount = interpolated[1] ? 4 : 1;
  for (let yi = 0; yi < yCount; yi += 1) {
    const y = interpolated[1] ? yi : 1;
    for (let xi = 0; xi < xCount; xi += 1) {
      const x = interpolated[0] ? xi : 1;
      const point: [number, number] = [span * (c.lower[0] + x - 1 + c.offset[0]),
        span * (c.lower[1] + y - 1 + c.offset[1])];
      const node = uniformStaggeredNode(topology, fields, point, axis, span);
      if (node[1] === 0) return sampleSourceLinear(topology, fields, position, axis, span);
      values[y]![x] = node[0];
      if ((x === 1 || interpolated[0] && x === 2)
        && (y === 1 || interpolated[1] && y === 2)) {
        coreMin = Math.min(coreMin, node[0]); coreMax = Math.max(coreMax, node[0]);
      }
    }
  }
  const lines = new Array(4).fill(0);
  for (let yi = 0; yi < yCount; yi += 1) {
    const y = interpolated[1] ? yi : 1;
    lines[y] = interpolated[0] ? cubicLine(values[y]![0]!, values[y]![1]!,
      values[y]![2]!, values[y]![3]!, c.fraction[0]) : values[y]![1]!;
  }
  const cubic = interpolated[1]
    ? cubicLine(lines[0]!, lines[1]!, lines[2]!, lines[3]!, c.fraction[1]) : lines[1]!;
  return Math.max(coreMin, Math.min(coreMax, cubic));
}

function pointInsideSolid(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  point: readonly [number, number]): boolean {
  const x = Math.floor(point[0]), y = Math.floor(point[1]);
  if (topology.solidVoxelAt?.(x, y)) return true;
  const cell = ownerAt(topology, Math.floor(point[0]) + 0.5, Math.floor(point[1]) + 0.5);
  return cell >= 0 && fields.capacity[cell]! <= 1e-8;
}

function clipBoundarySegment(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  start: readonly [number, number], candidate: readonly [number, number]): readonly [number, number] {
  let low = 0, high = 1, found = false;
  for (let probe = 1; probe <= 8; probe += 1) {
    const t = div(probe, 8);
    const q = [add(start[0], mul(t, f(candidate[0] - start[0]))),
      add(start[1], mul(t, f(candidate[1] - start[1])))] as const;
    if (pointInsideSolid(topology, fields, q)) { low = (probe - 1) / 8; high = t; found = true; break; }
  }
  if (!found) return candidate;
  for (let step = 0; step < 8; step += 1) {
    const mid = mul(0.5, add(low, high));
    const q = [add(start[0], mul(mid, f(candidate[0] - start[0]))),
      add(start[1], mul(mid, f(candidate[1] - start[1])))] as const;
    if (pointInsideSolid(topology, fields, q)) high = mid; else low = mid;
  }
  const t = Math.max(0, f(low - 1e-4));
  return [add(start[0], mul(t, f(candidate[0] - start[0]))),
    add(start[1], mul(t, f(candidate[1] - start[1])))];
}

function traceEffectiveTransportCharacteristic(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, position: readonly [number, number], span: number,
  dt: number, direction: -1 | 1): readonly [number, number] {
  const initial = sampleFaceSupport(topology, fields, position[0], position[1], span);
  const length = f(Math.sqrt(add(mul(div(initial[0], span), div(initial[0], span)),
    mul(div(initial[1], span), div(initial[1], span)))));
  const substeps = Math.max(1, Math.min(16, Math.ceil(mul(length, dt))));
  const subDt = f(dt / substeps), dimensions = topology.dimensions!;
  let traced: readonly [number, number] = position;
  for (let step = 0; step < substeps; step += 1) {
    const first = step === 0 ? initial : sampleFaceSupport(topology, fields, traced[0], traced[1], span);
    const rawMid: [number, number] = [
      Math.max(mul(0.5, span), Math.min(f(dimensions[0] - mul(0.5, span)),
        f(traced[0] + mul(direction, mul(mul(0.5, subDt), first[0]))))),
      Math.max(mul(0.5, span), Math.min(f(dimensions[1] - mul(0.5, span)),
        f(traced[1] + mul(direction, mul(mul(0.5, subDt), first[1])))))];
    const midpoint = clipBoundarySegment(topology, fields, traced, rawMid);
    const middle = sampleFaceSupport(topology, fields, midpoint[0], midpoint[1], span);
    const raw: [number, number] = [
      Math.max(mul(0.5, span), Math.min(f(dimensions[0] - mul(0.5, span)),
        f(traced[0] + mul(direction, mul(subDt, middle[0]))))),
      Math.max(mul(0.5, span), Math.min(f(dimensions[1] - mul(0.5, span)),
        f(traced[1] + mul(direction, mul(subDt, middle[1])))))];
    traced = clipBoundarySegment(topology, fields, traced, raw);
  }
  return traced;
}

function traceFaceDeparture(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  position: readonly [number, number], span: number, dt: number): readonly [number, number] {
  return traceEffectiveTransportCharacteristic(topology, fields, position, span, dt, -1);
}

/** Exact 2-D reduction of traceEffectiveTransportArrival used by resident tracers. */
export function traceSliceEffectiveTransportArrival(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, position: readonly [number, number], dt: number): readonly [number, number] {
  const owner = ownerAt(topology, Math.floor(position[0]) + 0.5,
    Math.floor(position[1]) + 0.5);
  const span = owner < 0 ? 1 : topology.cells[owner]!.widths[0];
  return traceEffectiveTransportCharacteristic(topology, fields, position, span, dt, 1);
}

/**
 * Publish the reduced TRANSPORT_CHARACTERISTIC_CLEARANCE plane. The four
 * bilinear corners are the literal Z-invariant counterpart of the resident
 * eight-corner stencil, including zero-weight support corners.
 */
export function publishSliceTransportCharacteristicClearance(
  topology: SliceNumericalTopology, fields: SliceNumericalFields, dt: number,
  sourceDensity: ArrayLike<number> = fields.density,
  sourceGamma: ArrayLike<number> = fields.gamma,
): Float32Array {
  const result = fields.characteristicClearance?.length === topology.cells.length
    ? fields.characteristicClearance : new Float32Array(topology.cells.length);
  const dimensions = topology.dimensions!;
  for (const receiver of topology.cells) {
    if (fields.capacity[receiver.id]! <= 1e-8) { result[receiver.id] = 0; continue; }
    const span = receiver.widths[0];
    const departure = traceEffectiveTransportCharacteristic(topology, fields,
      receiver.center, span, dt, -1);
    const bx = Math.max(0.5 * span, Math.min(dimensions[0] - 0.5 * span, departure[0]));
    const by = Math.max(0.5 * span, Math.min(dimensions[1] - 0.5 * span, departure[1]));
    const sx = bx / span - 0.5, sy = by / span - 0.5;
    const lx = Math.floor(sx), ly = Math.floor(sy), tx = f(sx - lx), ty = f(sy - ly);
    let visible = 0, minimumWidth = 1e30, valid = true;
    let minX = 1e30, minY = 1e30, maxX = -1e30, maxY = -1e30;
    for (let dy = 0; dy < 2; dy += 1) for (let dx = 0; dx < 2; dx += 1) {
      const weight = mul(dx ? tx : f(1 - tx), dy ? ty : f(1 - ty));
      visible = add(visible, weight);
      const cell = ownerAt(topology, span * (lx + dx + 0.5), span * (ly + dy + 0.5));
      if (cell < 0 || fields.capacity[cell]! <= 1e-8
        || fields.capacity[cell]! < 1 - 1e-6
        || Math.abs(sourceDensity[cell]! - 1) > 0.005
        || Math.abs(sourceGamma[cell]! - 1) > 0.005) { valid = false; continue; }
      const record = topology.cells[cell]!;
      minimumWidth = Math.min(minimumWidth, record.widths[0], record.widths[1]);
      const vx = fields.cellVelocity[2 * cell]!, vy = fields.cellVelocity[2 * cell + 1]!;
      minX = Math.min(minX, vx); minY = Math.min(minY, vy);
      maxX = Math.max(maxX, vx); maxY = Math.max(maxY, vy);
    }
    if (!valid || visible < 0.999999) { result[receiver.id] = 0; continue; }
    const deltaX = f(maxX - minX), deltaY = f(maxY - minY);
    const maximumDelta = f(Math.sqrt(add(mul(deltaX, deltaX), mul(deltaY, deltaY))));
    const clearance = mul(0.02, minimumWidth);
    result[receiver.id] = mul(dt, maximumDelta) <= mul(0.002, minimumWidth)
      ? clearance : 0;
  }
  fields.characteristicClearance = result;
  return result;
}

/** Current pressure predicate used by FSM1 FLIP publication. */
export function slicePressureMembershipPredicate(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, cell: number): boolean {
  let submerged = fields.pressureMember[cell]! !== 0, neighbours = 0;
  if (submerged) for (const rowId of topology.incidences[cell] ?? []) {
    const row = topology.rows[rowId]!;
    if (row.terms.length < 2) { submerged = false; break; }
    for (const term of row.terms) if (term.cellId !== cell) {
      neighbours += 1;
      if (!fields.pressureMember[term.cellId]) { submerged = false; break; }
    }
    if (!submerged) break;
  }
  submerged &&= neighbours > 0;
  return (pressureDensity(topology, fields, cell) >= CM12_LIQUID_ISOVALUE
      || submerged
      || movingPressurePredictedFill(topology, fields, cell)
      || (fields.sourceRate?.[cell] ?? 0) > 0)
    && fields.capacity[cell]! * topology.cells[cell]!.area > 1e-8;
}

/** Exact two-dimensional reduction of production accepted-face preparation. */
export function prepareSliceFaces(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, dt: number): void {
  for (const row of topology.rows) {
    if (physicalRowArea(row) <= 1e-8) { fields.faceVelocity[row.id] = row.solidVelocity ?? 0; continue; }
    let touchesExtended = false, samplingWidth = Number.MAX_VALUE;
    for (const term of row.terms) {
      const support = faceSupportAt(topology, fields,
        topology.cells[term.cellId]!.center[0], topology.cells[term.cellId]!.center[1]);
      touchesExtended ||= support.extended; samplingWidth = Math.min(samplingWidth, support.span);
    }
    if (!touchesExtended) { fields.faceVelocity[row.id] = row.solidVelocity ?? 0; continue; }
    const span = Math.max(1, samplingWidth);
    let regionWidth = span;
    for (const term of row.terms) {
      const cell = topology.cells[term.cellId]!;
      if ((cell.refinementRegionScale ?? 1) > 1) {
        regionWidth = Math.max(regionWidth, Math.min(...cell.widths));
      }
    }
    const departure = traceFaceDeparture(topology, fields, row.center, regionWidth, dt);
    const characteristic = sampleSource(topology, fields, departure, row.axis, span);
    const open = row.openFraction ?? 1;
    fields.faceVelocity[row.id] = add(mul(open, characteristic),
      mul(1 - open, row.solidVelocity ?? 0));
  }
}

/** Production forceFaces reduction for ordinary, closed-world and zero-area rows. */
export function forceSliceFaces(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  dt: number,
  accelerationFine: readonly [number, number],
  inflowVelocityFine: readonly [number, number] = [0, 0],
): void {
  for (const row of topology.rows) {
    const wall = row.solidVelocity ?? 0;
    row.separating = false;
    if (row.kind === "closed-world") {
      const cell = row.terms[0]?.cellId;
      if (cell !== undefined && fields.capacity[cell]! > 1e-8
        && fields.density[cell]! > CM12_LIQUID_ISOVALUE) {
        const length = Math.hypot(accelerationFine[0], accelerationFine[1]);
        const own = row.terms[0]!.coefficient;
        const orientation = own >= 0 ? 1 : -1;
        const predicted = add(fields.cellVelocity[2 * cell + row.axis]!,
          mul(dt, accelerationFine[row.axis]));
        const outwardTravel = f(dt * orientation * (predicted - wall)
          / Math.max(row.distance, 1e-6));
        const deadband = fields.pressureRowMember?.[row.id] ? 5e-5 : 1e-4;
        row.separating = length > 1e-6
          && orientation * accelerationFine[row.axis] > 0.5 * length
          && outwardTravel > deadband;
      }
      fields.faceVelocity[row.id] = row.separating && cell !== undefined
        ? add(fields.cellVelocity[2 * cell + row.axis]!, mul(dt, accelerationFine[row.axis]))
        : f(wall);
    } else if (physicalRowArea(row) <= 1e-8) {
      fields.faceVelocity[row.id] = f(wall);
    } else {
      const open = row.openFraction ?? 1;
      const forced = add(fields.faceVelocity[row.id]!,
        mul(open, mul(dt, accelerationFine[row.axis])));
      const inflow = fields.inflowCoverage?.[row.id] ?? 0;
      fields.faceVelocity[row.id] = add(mul(1 - inflow, forced),
        mul(inflow, inflowVelocityFine[row.axis]));
    }
  }
}

/** Production re-applies the prescribed nozzle disk after pressure. */
export function enforceSliceInflowFaces(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, inflowVelocityFine: readonly [number, number]): void {
  for (const row of topology.rows) {
    if (physicalRowArea(row) <= 1e-8) continue;
    const coverage = fields.inflowCoverage?.[row.id] ?? 0;
    if (coverage <= 0) continue;
    fields.faceVelocity[row.id] = add(mul(1 - coverage, fields.faceVelocity[row.id]!),
      mul(coverage, inflowVelocityFine[row.axis]));
  }
}

function pressureDensity(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  cell: number): number {
  let density = cellFill(fields, cell);
  const dt = fields.frameDt ?? 0, area = topology.cells[cell]!.area;
  const finalCapacity = mul(fields.capacity[cell]!, area);
  const before = mul(fields.capacityBefore?.[cell] ?? fields.capacity[cell]!, area);
  const source = fields.sourceRate?.[cell] ?? 0;
  if (fields.solidMotionActive && finalCapacity < before && fields.density[cell]! > 0) {
    const rate = dt > 0 ? div(f(finalCapacity - before), dt) : 0;
    density = Math.max(density, add(CM12_LIQUID_ISOVALUE,
      Math.min(0.5, div(mul(-rate, dt), Math.max(finalCapacity, 1e-8)))));
  }
  if (source > 0) density = Math.max(density, add(CM12_LIQUID_ISOVALUE,
    Math.min(0.5, div(mul(source, dt), Math.max(finalCapacity, 1e-8)))));
  return f(density);
}

function movingPressurePredictedFill(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, cell: number): boolean {
  if (!fields.solidMotionActive) return false;
  let equation = 0, correction = 0;
  for (const rowId of topology.incidences[cell] ?? []) {
    const row = topology.rows[rowId]!, own = row.terms.find(term => term.cellId === cell);
    if (!own) continue;
    let velocity = fields.faceVelocity[row.id]!;
    if (topology.solidVoxelAt || fields.solidMotionActive) {
      velocity = f(velocity - mul(f(1 - (row.openFraction ?? 1)), row.solidVelocity ?? 0));
    }
    const weight = topology.solidVoxelAt || fields.solidMotionActive
      ? row.staticDualWeight ?? row.dualWeight : pressureDualWeight(row);
    const value = mul(own.coefficient, mul(weight, velocity));
    if (value > 0) {
      let supported = false;
      for (const term of row.terms) {
        if (term.cellId === cell || term.coefficient * own.coefficient >= 0) continue;
        const otherCapacity = mul(fields.capacity[term.cellId]!, topology.cells[term.cellId]!.area);
        const otherVolume = mul(fields.density[term.cellId]!, topology.cells[term.cellId]!.area);
        supported ||= otherVolume > volumeRoundoff(otherCapacity)
          || (fields.sourceRate?.[term.cellId] ?? 0) > 0;
      }
      if (!supported) continue;
    }
    const adjusted = f(value - correction), next = add(equation, adjusted);
    correction = f(f(next - equation) - adjusted); equation = next;
  }
  const predicted = add(mul(fields.density[cell]!, topology.cells[cell]!.area),
    mul(fields.frameDt ?? 0, equation));
  const finalCapacity = mul(fields.capacity[cell]!, topology.cells[cell]!.area);
  return predicted >= f(finalCapacity - volumeRoundoff(finalCapacity));
}

function pressureMembership(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
): void {
  const prior = Uint8Array.from(fields.pressureMember);
  fields.pressureMember.fill(0);
  for (const cell of topology.cells) {
    const fill = pressureDensity(topology, fields, cell.id);
    let submerged = false;
    if (prior[cell.id]) {
      submerged = true; let neighbours = 0;
      for (const rowId of topology.incidences[cell.id] ?? []) {
        const row = topology.rows[rowId]!;
        if (row.terms.length < 2) { submerged = false; break; }
        for (const term of row.terms) {
          if (term.cellId === cell.id) continue;
          neighbours++;
          if (!prior[term.cellId]) { submerged = false; break; }
        }
        if (!submerged) break;
      }
      submerged &&= neighbours > 0;
    }
    let member = fill >= CM12_LIQUID_ISOVALUE || submerged
      || movingPressurePredictedFill(topology, fields, cell.id)
      || (fields.sourceRate?.[cell.id] ?? 0) > 0;
    member &&= fields.capacity[cell.id]! * cell.area > 1e-8;
    fields.pressureMember[cell.id] = member ? 1 : 0;
  }
}

export interface SlicePressureRows {
  active: Uint8Array;
  theta: Float32Array;
}

/** Persistent PCM/PCF row image used to preserve production cache reuse. */
export interface SlicePressureRowCache {
  readonly topologyGeneration: number;
  readonly currentTopologyGeneration: number;
  readonly cellCapacity: number;
  readonly acceptedCellBits: Uint32Array;
  readonly acceptedRowBits: Uint32Array;
  readonly densityBits: Uint32Array;
  readonly capacityBits: Uint32Array;
  readonly normalXBits: Uint32Array;
  readonly normalYBits: Uint32Array;
  readonly rowTheta: Float32Array;
  readonly globalRowInvalidation?: boolean;
}

const scalarWord = (value: number): number =>
  new Uint32Array(new Float32Array([value]).buffer)[0]!;

function cachedPressureCellDirectChanged(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, cache: SlicePressureRowCache, cell: number): boolean {
  const stable = topology.cells[cell]!.stableId ?? cell;
  if (stable >= cache.cellCapacity) return true;
  const word = stable >>> 5, bit = 1 << (stable & 31);
  const memberChanged = (((cache.acceptedCellBits[word] ?? 0) & bit) !== 0)
    !== (fields.pressureMember[cell]! !== 0);
  const closing = fields.solidMotionActive
    && (fields.capacityAfter?.[cell] ?? fields.capacity[cell]!)
      < (fields.capacityBefore?.[cell] ?? fields.capacity[cell]!)
    && fields.density[cell]! > 0;
  return closing || (fields.sourceRate?.[cell] ?? 0) > 0 || memberChanged
    || scalarWord(fields.density[cell]!) !== cache.densityBits[stable]
    || scalarWord(fields.capacity[cell]!) !== cache.capacityBits[stable]
    || scalarWord(fields.interfaceNormal[2 * cell]!) !== cache.normalXBits[stable]
    || scalarWord(fields.interfaceNormal[2 * cell + 1]!) !== cache.normalYBits[stable];
}

function cachedPressureCellChanged(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, cache: SlicePressureRowCache, cell: number): boolean {
  if (cachedPressureCellDirectChanged(topology, fields, cache, cell)) return true;
  const fill = cellFill(fields, cell);
  if (fields.capacity[cell]! < 0.999999 || fill <= 0 || fill >= 1) return false;
  // Interface reconstruction reads one physical neighbour ring. Production
  // propagates FSM1/member dirtiness through that ring before reusing a row.
  for (const rowId of topology.incidences[cell] ?? []) {
    const row = topology.rows[rowId]!, own = row.terms.find(term => term.cellId === cell);
    if (!own) continue;
    for (const term of row.terms) {
      if (term.cellId !== cell && own.coefficient * term.coefficient < 0
        && cachedPressureCellDirectChanged(topology, fields, cache, term.cellId)) return true;
    }
  }
  return false;
}

function pressureDualWeight(row: SliceNumericalRow): number {
  const open = row.kind === "closed-world"
    ? (row.separating ? 1 : 0) : (row.openFraction ?? 1);
  return f((row.staticDualWeight ?? row.dualWeight) * open);
}

function pressureIntegratedColumnHeight(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, x: number): readonly [number, number] {
  const dimensions = topology.dimensions!;
  let y = 0, massHeight = 0, previous = 1, columnOpen = -1;
  let sawOpen = false, sawLiquid = false, sawAir = false;
  while (y < dimensions[1]) {
    if ((topology.solidVoxelFractionAt?.(x, y) ?? 0) >= 1) return [0, 0];
    const owner = ownerAt(topology, x + 0.5, y + 0.5);
    let fill = 0, width = 1;
    if (owner < 0) width = Math.max(1, Math.min(8 - y % 8, dimensions[1] - y));
    else {
      const open = fields.capacity[owner]!;
      if (open <= 1e-6) return [0, 0];
      sawOpen = true; if (columnOpen < 0) columnOpen = open;
      if (Math.abs(f(open - columnOpen)) > 1e-3) return [0, 0];
      fill = Math.max(0, Math.min(1, pressureDensity(topology, fields, owner)));
      const cell = topology.cells[owner]!;
      width = Math.max(1, Math.min(cell.widths[1] - y % cell.widths[1], dimensions[1] - y));
    }
    if (fill > previous + 0.01) return [0, 0];
    previous = fill; sawLiquid ||= fill > 1e-3; sawAir ||= fill < 1 - 1e-3;
    massHeight = add(massHeight, mul(fill, width)); y += width;
  }
  return [massHeight, sawOpen && sawLiquid && sawAir ? 1 : 0];
}

function pressurePlanarColumnHeight(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, row: SliceNumericalRow): readonly [number, number] {
  const nx = topology.dimensions![0], centre = Math.max(0, Math.min(nx - 1, Math.floor(row.center[0])));
  let height = 0, minimum = Number.MAX_VALUE, maximum = -Number.MAX_VALUE, valid = true;
  for (const offset of [0, -1, 1]) {
    const receipt = pressureIntegratedColumnHeight(topology, fields,
      Math.max(0, Math.min(nx - 1, centre + offset)));
    if (offset === 0) height = receipt[0];
    valid &&= receipt[1] > 0.5; minimum = Math.min(minimum, receipt[0]); maximum = Math.max(maximum, receipt[0]);
  }
  valid &&= maximum - minimum <= 0.01;
  return [height, valid ? 1 : 0];
}

export function prepareSlicePressureTopology(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  cache?: SlicePressureRowCache,
): SlicePressureRows {
  pressureMembership(topology, fields);
  const active = new Uint8Array(topology.rows.length);
  const theta = new Float32Array(topology.rows.length);
  const dirtyTiles = new Uint8Array(Math.ceil(topology.rows.length / 64));
  if (!cache || cache.globalRowInvalidation
    || cache.topologyGeneration !== cache.currentTopologyGeneration) dirtyTiles.fill(1);
  else for (const row of topology.rows) {
    if (row.terms.some(term => cachedPressureCellChanged(topology, fields, cache, term.cellId))) {
      dirtyTiles[row.id >>> 6] = 1;
    }
  }
  for (const row of topology.rows) {
    const tileDirty = dirtyTiles[row.id >>> 6] !== 0;
    if (!tileDirty) {
      const cached = cache!;
      const activeWord = cached.acceptedRowBits[row.id >>> 5] ?? 0;
      active[row.id] = (activeWord & (1 << (row.id & 31))) !== 0 ? 1 : 0;
      theta[row.id] = cached.rowTheta[row.id] ?? 0;
      continue;
    }
    let geometricX = 0, geometricY = 0, geometricOffset = 0, geometricWeight = 0;
    if (row.kind !== "closed-world") for (const term of row.terms) {
      const cell = topology.cells[term.cellId]!, nx = fields.interfaceNormal[2 * cell.id]!;
      const ny = fields.interfaceNormal[2 * cell.id + 1]!;
      if (add(mul(nx, nx), mul(ny, ny)) <= 0.5) continue;
      const weight = Math.abs(term.coefficient);
      geometricX = add(geometricX, mul(weight, nx)); geometricY = add(geometricY, mul(weight, ny));
      geometricOffset = add(geometricOffset, mul(weight, add(fields.interfaceOffset[cell.id]!,
        add(mul(nx, f(cell.center[0] - row.center[0])), mul(ny, f(cell.center[1] - row.center[1]))))));
      geometricWeight = add(geometricWeight, weight);
    }
    const geometricLength = f(Math.sqrt(add(mul(geometricX, geometricX), mul(geometricY, geometricY))));
    let geometryValid = geometricWeight > 1e-8 && geometricLength > mul(1e-6, geometricWeight);
    if (geometryValid) for (const term of row.terms) {
      const cell = topology.cells[term.cellId]!;
      const phi = div(f(add(mul(geometricX, f(cell.center[0] - row.center[0])),
        mul(geometricY, f(cell.center[1] - row.center[1]))) - geometricOffset), geometricLength);
      geometryValid &&= fields.capacity[cell.id]! >= 0.999999
        && (fields.pressureMember[cell.id] ? phi <= 0 : phi >= 0);
    }
    let liquidCount = 0, airCount = 0, liquidPhi = 0, liquidWeight = 0, airPhi = 0, airWeight = 0;
    let liquidCenterY = 0, airCenterY = 0, fullPhiGradient = 0, liquidPhiGradient = 0;
    for (const term of row.terms) {
      const cell = topology.cells[term.cellId]!;
      const oldPhi = mul(f(CM12_LIQUID_ISOVALUE - pressureDensity(topology, fields, cell.id)),
        row.kind === "sparse-air" ? 1 : cell.widths[row.axis]);
      const phi = geometryValid ? div(f(add(mul(geometricX, f(cell.center[0] - row.center[0])),
        mul(geometricY, f(cell.center[1] - row.center[1]))) - geometricOffset), geometricLength) : oldPhi;
      const weight = f(Math.abs(term.coefficient));
      const signedPhi = mul(term.coefficient, phi);
      fullPhiGradient = add(fullPhiGradient, signedPhi);
      if (fields.pressureMember[cell.id]) {
        liquidCount += 1; liquidPhi = add(liquidPhi, mul(weight, phi));
        liquidWeight = add(liquidWeight, weight); liquidPhiGradient = add(liquidPhiGradient, signedPhi);
        liquidCenterY = add(liquidCenterY, mul(weight, cell.center[1]));
      } else {
        airCount += 1; airPhi = add(airPhi, mul(weight, phi)); airWeight = add(airWeight, weight);
        airCenterY = add(airCenterY, mul(weight, cell.center[1]));
      }
    }
    if (liquidCount === 0 || pressureDualWeight(row) <= 1e-8) continue;
    if (row.kind === "sparse-air") {
      airPhi = add(airPhi, mul(liquidWeight, 0.5));
      const liquidY = div(liquidCenterY, Math.max(liquidWeight, 1e-9));
      const direction = row.center[1] >= liquidY ? 1 : -1;
      airCenterY = add(airCenterY, mul(liquidWeight, add(liquidY, mul(direction, row.distance))));
      airWeight = add(airWeight, liquidWeight);
    }
    const cut = airCount > 0 || row.kind === "sparse-air";
    let rowTheta = cut ? cm12GhostFluidTheta(
      div(liquidPhi, Math.max(liquidWeight, 1e-9)),
      div(airPhi, Math.max(airWeight, 1e-9)), 1e-12,
    ) : 1;
    const gravity = fields.accelerationFine ?? [0, 0];
    const gravityLength = f(Math.sqrt(add(mul(gravity[0], gravity[0]), mul(gravity[1], gravity[1]))));
    const partialRegion = topology.cells.some(cell => (cell.refinementRegionScale ?? 1) > 1)
      && topology.cells.some(cell => (cell.refinementRegionScale ?? 1) === 1);
    if (cut && row.axis === 1 && gravityLength > 1e-6 && partialRegion
      && gravity[1] < 0 && Math.abs(gravity[0]) <= mul(1e-6, gravityLength)) {
      const height = pressurePlanarColumnHeight(topology, fields, row);
      const liquidY = div(liquidCenterY, Math.max(liquidWeight, 1e-9));
      const airY = div(airCenterY, Math.max(airWeight, 1e-9));
      if (height[1] > 0.5 && airY > liquidY + 1e-6 && height[0] > liquidY && height[0] < airY) {
        rowTheta = Math.max(CM12_GHOST_FLUID_THETA_MIN,
          Math.min(1, div(f(height[0] - liquidY), f(airY - liquidY))));
      }
    }
    if (cut && row.kind === "mixed-seam") {
      let factor = 0;
      if (fullPhiGradient !== 0) factor = liquidPhiGradient === 0
        ? div(1, CM12_GHOST_FLUID_THETA_MIN)
        : Math.max(0, Math.min(div(1, CM12_GHOST_FLUID_THETA_MIN),
          div(fullPhiGradient, liquidPhiGradient)));
      rowTheta = factor > 0 ? div(1, factor) : 0;
    }
    theta[row.id] = f(rowTheta);
    active[row.id] = 1;
  }
  fields.pressureDiagonal.fill(0);
  for (const cell of topology.cells) {
    if (!fields.pressureMember[cell.id]) continue;
    const axes = new Float32Array(2);
    for (const rowId of topology.incidences[cell.id] ?? []) {
      const row = topology.rows[rowId]!;
      if (!active[row.id] || theta[row.id]! <= 0) continue;
      const own = row.terms.find(term => term.cellId === cell.id);
      if (!own) continue;
      axes[row.axis] = add(axes[row.axis]!, div(
        mul(pressureDualWeight(row), mul(own.coefficient, own.coefficient)),
        theta[row.id]!,
      ));
    }
    fields.pressureDiagonal[cell.id] = add(axes[0]!, axes[1]!);
  }
  fields.pressureRowMember = Uint8Array.from(active);
  return { active, theta };
}

function pressureRowGradient(row: SliceNumericalRow, fields: SliceNumericalFields,
  input: Float32Array): number {
  if (row.terms.length === 2
    && row.terms[0]!.coefficient === -row.terms[1]!.coefficient) {
    const a = row.terms[0]!, b = row.terms[1]!;
    const pa = fields.pressureMember[a.cellId] ? input[a.cellId]! : 0;
    const pb = fields.pressureMember[b.cellId] ? input[b.cellId]! : 0;
    return mul(b.coefficient, f(pb - pa));
  }
  let jump = 0;
  for (const term of row.terms) if (fields.pressureMember[term.cellId]) {
    jump = add(jump, mul(term.coefficient, input[term.cellId]!));
  }
  return jump;
}

function applyPressureOperator(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  rows: SlicePressureRows,
  input: Float32Array,
  output: Float32Array,
): void {
  output.fill(0);
  for (const cell of topology.cells) {
    if (!fields.pressureMember[cell.id]) continue;
    const negative = new Float32Array(2), positive = new Float32Array(2);
    for (const rowId of topology.incidences[cell.id] ?? []) {
      const row = topology.rows[rowId]!;
      if (!rows.active[row.id] || rows.theta[row.id]! <= 0) continue;
      const own = row.terms.find(term => term.cellId === cell.id);
      if (!own) continue;
      const contribution = div(mul(pressureDualWeight(row),
        mul(own.coefficient, pressureRowGradient(row, fields, input))), rows.theta[row.id]!);
      if (own.coefficient > 0) negative[row.axis] = add(negative[row.axis]!, contribution);
      else positive[row.axis] = add(positive[row.axis]!, contribution);
    }
    output[cell.id] = add(add(Math.min(negative[0]!, positive[0]!),
      Math.max(negative[0]!, positive[0]!)),
    add(Math.min(negative[1]!, positive[1]!), Math.max(negative[1]!, positive[1]!)));
  }
}

export function assembleSlicePressureRhs(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, rowState: SlicePressureRows): void {
  fields.pressureRhs.fill(0);
  for (const cell of topology.cells) {
    if (!fields.pressureMember[cell.id]) continue;
    const negative = new Float32Array(2), positive = new Float32Array(2);
    for (const rowId of topology.incidences[cell.id] ?? []) {
      const row = topology.rows[rowId]!;
      if (!rowState.active[row.id]) continue;
      const own = row.terms.find(term => term.cellId === cell.id);
      if (!own) continue;
      const fluidVelocity = fields.faceVelocity[row.id]!
        - mul(1 - (row.openFraction ?? 1), row.solidVelocity ?? 0);
      const value = mul(own.coefficient,
        mul(row.staticDualWeight ?? row.dualWeight, fluidVelocity));
      if (own.coefficient > 0) negative[row.axis] = add(negative[row.axis]!, value);
      else positive[row.axis] = add(positive[row.axis]!, value);
    }
    fields.pressureRhs[cell.id] = add(
      add(Math.min(negative[0]!, positive[0]!), Math.max(negative[0]!, positive[0]!)),
      add(Math.min(negative[1]!, positive[1]!), Math.max(negative[1]!, positive[1]!)),
    );
    fields.pressureRhs[cell.id] = add(fields.pressureRhs[cell.id]!,
      -(fields.capacityRate?.[cell.id] ?? 0) * cell.area);
    fields.pressureRhs[cell.id] = add(fields.pressureRhs[cell.id]!,
      fields.sourceRate?.[cell.id] ?? 0);
  }
}

export function solveSlicePressure(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, rowState: SlicePressureRows,
  maximumIterations: number, relativeTolerance = 1e-6,
  executionOrder?: Uint32Array): SlicePressureReceipt {
  const solve = solveSlicePressurePCG({ diagonal: fields.pressureDiagonal,
    rhs: fields.pressureRhs, pressure: fields.pressure, member: fields.pressureMember,
    maximumIterations, relativeTolerance, executionOrder,
    apply: (input, output) => applyPressureOperator(topology, fields, rowState, input, output) });
  fields.pressure.set(solve.pressure);
  return { iterations: solve.iterations,
    initialResidual: Math.sqrt(Math.max(0, solve.initialTrueResidualSquared)),
    residual: Math.sqrt(Math.max(0, solve.finalTrueResidualSquared)),
    converged: solve.converged };
}

export function projectSlicePressureVelocity(topology: SliceNumericalTopology,
  fields: SliceNumericalFields, rowState: SlicePressureRows): void {
  for (const row of topology.rows) {
    if (!rowState.active[row.id] || rowState.theta[row.id]! <= 0) continue;
    const jump = pressureRowGradient(row, fields, fields.pressure);
    const pressureOpen = row.kind === "closed-world"
      ? (row.separating ? 1 : 0) : (row.openFraction ?? 1);
    fields.faceVelocity[row.id] = add(fields.faceVelocity[row.id]!,
      -div(mul(pressureOpen, jump), rowState.theta[row.id]!));
  }
  collocateSliceVelocity(topology, fields);
}

/** Compatibility wrapper for callers that do not need stage snapshots. */
export function projectSliceVelocity(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  maximumIterations: number,
  relativeTolerance = 1e-6,
): SlicePressureReceipt {
  const rowState = prepareSlicePressureTopology(topology, fields);
  assembleSlicePressureRhs(topology, fields, rowState);
  const receipt = solveSlicePressure(topology, fields, rowState,
    maximumIterations, relativeTolerance);
  projectSlicePressureVelocity(topology, fields, rowState);
  return receipt;
}

export function collocateSliceVelocity(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
): void {
  fields.cellVelocity.fill(0);
  const weights = new Float32Array(fields.cellVelocity.length);
  for (const row of topology.rows) for (const term of row.terms) {
    const at = 2 * term.cellId + row.axis;
    const weight = f(Math.abs(term.coefficient) * (row.staticDualWeight ?? row.dualWeight));
    const open = row.openFraction ?? 1;
    const fluidVelocity = open > 1e-6
      ? div(fields.faceVelocity[row.id]! - mul(1 - open, row.solidVelocity ?? 0), open)
      : row.solidVelocity ?? 0;
    fields.cellVelocity[at] = add(fields.cellVelocity[at]!, mul(weight, fluidVelocity));
    weights[at] = add(weights[at]!, weight);
  }
  for (let i = 0; i < fields.cellVelocity.length; i += 1) {
    if (weights[i]! > 0) fields.cellVelocity[i] = div(fields.cellVelocity[i]!, weights[i]!);
  }
}

function boxFraction(normalX: number, normalY: number, offset: number,
  widthX = 1, widthY = 1): number {
  const px = f(Math.abs(normalX) * widthX), py = f(Math.abs(normalY) * widthY);
  const dominant = Math.max(px, py);
  if (dominant <= 1e-20) return offset >= 0 ? 1 : 0;
  const spans = [f(px / dominant), f(py / dominant)].filter(value => value >= 1e-6);
  const total = add(spans[0] ?? 0, spans[1] ?? 0);
  const shifted = add(div(offset, dominant), mul(0.5, total));
  if (shifted <= 0) return 0; if (shifted >= total) return 1;
  const complement = shifted > mul(0.5, total);
  const x = complement ? f(total - shifted) : shifted;
  let fraction: number;
  if (spans.length === 1) fraction = div(x, spans[0]!);
  else {
    const aa = Math.min(spans[0]!, spans[1]!), bb = Math.max(spans[0]!, spans[1]!);
    fraction = x < aa ? mul(mul(0.5, div(x, aa)), div(x, bb))
      : div(f(x - mul(0.5, aa)), bb);
  }
  fraction = Math.max(0, Math.min(1, fraction));
  return complement ? f(1 - fraction) : fraction;
}

function offsetForFill(fill: number, nx: number, ny: number,
  widths: readonly [number, number]): number {
  const projected = [f(Math.abs(nx) * widths[0]), f(Math.abs(ny) * widths[1])] as const;
  const dominant = Math.max(projected[0], projected[1]);
  const radius = mul(0.5, add(projected[0], projected[1]));
  if (fill <= 0) return -radius; if (fill >= 1) return radius;
  if (fill === 0.5 || dominant <= 1e-20) return 0;
  const spans = projected.map(value => f(value / dominant)).filter(value => value >= 1e-6);
  if (spans.length === 1) return mul(mul(f(fill - 0.5), spans[0]!), dominant);
  const complement = fill > 0.5, target = complement ? f(1 - fill) : f(fill);
  const aa = Math.min(spans[0]!, spans[1]!), bb = Math.max(spans[0]!, spans[1]!);
  let shifted = add(mul(target, bb), mul(0.5, aa));
  if (target < div(mul(0.5, aa), bb)) shifted = f(Math.sqrt(mul(mul(2, target), mul(aa, bb))));
  const result = mul(f(shifted - mul(0.5, add(aa, bb))), dominant);
  return complement ? -result : result;
}

function interfaceFromFill(fill: number, gx: number, gy: number,
  widths: readonly [number, number]) {
  const maximum = Math.max(Math.abs(gx), Math.abs(gy));
  if (!(maximum > 1e-20)) return { nx: 0, ny: 0, offset: 0 };
  const sx = div(gx, maximum), sy = div(gy, maximum);
  const length = f(Math.sqrt(add(mul(sx, sx), mul(sy, sy))));
  const nx = div(sx, length), ny = div(sy, length);
  return { nx, ny, offset: offsetForFill(fill, nx, ny, widths) };
}

export function sliceInterfaceCertificateSample(observed:number):number|null{
  return Number.isFinite(observed)&&observed>=-VOLUME_ROUNDOFF_RATIO
    &&observed<=1+VOLUME_ROUNDOFF_RATIO?Math.max(0,Math.min(1,observed)):null;
}

function fitScore(plane: { nx: number; ny: number; offset: number },
  samples: Float32Array): readonly [number, number] {
  let full = 0;
  const sides = new Float32Array(4);
  for (let j = 0; j < 3; j += 1) for (let i = 0; i < 3; i += 1) {
    const displaced = f(plane.offset - add(mul(plane.nx, i - 1), mul(plane.ny, j - 1)));
    const predicted = boxFraction(plane.nx, plane.ny, displaced);
    const difference = f(predicted - samples[3 * j + i]!);
    const error = mul(difference, difference); full = add(full, error);
    if (i <= 1) sides[0] = add(sides[0]!, error); if (i >= 1) sides[1] = add(sides[1]!, error);
    if (j <= 1) sides[2] = add(sides[2]!, error); if (j >= 1) sides[3] = add(sides[3]!, error);
  }
  return [div(full, 9), div(Math.min(...sides), 6)];
}

/** Resident least-squares fallback followed by certified 2-D ELVIRA candidates. */
export function reconstructSliceInterfaces(topology: SliceNumericalTopology,
  fields: SliceNumericalFields): void {
  fields.interfaceNormal.fill(0); fields.interfaceOffset.fill(0);
  for (const cell of topology.cells) {
    const fill = cellFill(fields, cell.id);
    if (fields.capacity[cell.id]! < 0.999999 || !(fill > 0 && fill < 1)) continue;
    let mxx = 0, mxy = 0, myy = 0, bx = 0, by = 0;
    for (const rowId of topology.incidences[cell.id] ?? []) {
      const row = topology.rows[rowId]!, own = row.terms.find(term => term.cellId === cell.id);
      if (!own || physicalRowArea(row) <= 1e-8) continue;
      for (const term of row.terms) {
        if (own.coefficient * term.coefficient >= 0) continue;
        const other = topology.cells[term.cellId]!;
        if (fields.capacity[other.id]! < 0.999999) continue;
        const dx = f(other.center[0] - cell.center[0]), dy = f(other.center[1] - cell.center[1]);
        const tangent = (1 - row.axis) as SliceAxis;
        const overlap = Math.max(0, Math.min(cell.maximum[tangent], other.maximum[tangent])
          - Math.max(cell.minimum[tangent], other.minimum[tangent]));
        const weight = div(overlap, Math.max(add(mul(dx, dx), mul(dy, dy)), 1e-12));
        const delta = f(cellFill(fields, other.id) - fill);
        mxx = add(mxx, mul(weight, mul(dx, dx))); mxy = add(mxy, mul(weight, mul(dx, dy)));
        myy = add(myy, mul(weight, mul(dy, dy)));
        bx = add(bx, mul(weight, mul(dx, delta))); by = add(by, mul(weight, mul(dy, delta)));
      }
    }
    const determinant = f(mul(mxx, myy) - mul(mxy, mxy)), scale = Math.max(mxx, myy);
    if (!(scale > 1e-12 && Math.abs(determinant) > 1e-7 * scale * scale)) continue;
    const gx = div(f(mul(myy, bx) - mul(mxy, by)), determinant);
    const gy = div(f(-mul(mxy, bx) + mul(mxx, by)), determinant);
    let best = interfaceFromFill(fill, -gx, -gy, cell.widths), bestScore: readonly [number, number];
    if (best.nx === 0 && best.ny === 0) continue;

    // The 2-D solver is the production extrusion itself. Certification needs
    // the same complete, open, unit 3x3 stencil before any height candidate.
    const samples = new Float32Array(9);
    let certified = cell.widths[0] === 1 && cell.widths[1] === 1;
    for (let j = 0; certified && j < 3; j += 1) for (let i = 0; i < 3; i += 1) {
      const otherId = ownerAt(topology, cell.center[0] + i - 1, cell.center[1] + j - 1);
      if (otherId < 0) { certified = false; break; }
      const other = topology.cells[otherId]!, observed = cellFill(fields, otherId);
      const certificateSample=sliceInterfaceCertificateSample(observed);
      if (other.widths[0] !== 1 || other.widths[1] !== 1
        || other.center[0] !== cell.center[0] + i - 1
        || other.center[1] !== cell.center[1] + j - 1
        || fields.capacity[otherId]! < 0.999999 || certificateSample===null) {
        certified = false; break;
      }
      // Transport deliberately admits this f32 volume margin. Treating the
      // same state as an invalid ELVIRA sample makes one reflected stencil
      // fall back to LS when its mirror rounded to exact zero or one.
      samples[3 * j + i] = certificateSample;
    }
    if (certified) {
      bestScore = fitScore(best, samples);
      let corner = best, cornerScore = bestScore;
      if (bestScore[0] > 9.094947017729282e-13) {
        for (let direction = 0; direction < 2; direction += 1) {
          const heights = new Float32Array(3);
          for (let column = 0; column < 3; column += 1) for (let at = 0; at < 3; at += 1) {
            const index = direction === 1 ? 3 * at + column : 3 * column + at;
            heights[column] = add(heights[column]!, samples[index]!);
          }
          for (let difference = 0; difference < 3; difference += 1) {
            let slope = f(heights[1]! - heights[0]!);
            if (difference === 1) slope = mul(0.5, f(heights[2]! - heights[0]!));
            if (difference === 2) slope = f(heights[2]! - heights[1]!);
            const integration = direction === 1 ? 1 : 0;
            const orientation = (integration === 0 ? best.nx : best.ny) >= 0 ? 1 : -1;
            const candidate = integration === 0
              ? interfaceFromFill(fill, orientation, -slope, [1, 1])
              : interfaceFromFill(fill, -slope, orientation, [1, 1]);
            const score = fitScore(candidate, samples);
            if (score[0] < bestScore[0]) { best = candidate; bestScore = score; }
            if (score[1] < cornerScore[1]
              || score[1] === cornerScore[1] && score[0] < cornerScore[0]) {
              corner = candidate; cornerScore = score;
            }
          }
        }
        if (bestScore[0] > 9.094947017729282e-13
          && cornerScore[1] <= 9.094947017729282e-13) best = corner;
      }
    }
    fields.interfaceNormal[2 * cell.id] = best.nx;
    fields.interfaceNormal[2 * cell.id + 1] = best.ny;
    fields.interfaceOffset[cell.id] = best.offset;
  }
}

function highFluxForSubface(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  face: SliceNumericalSubface,
  sweep: number,
  low: number,
): number {
  if (sweep === 0) return 0;
  const donorId = sweep > 0 ? face.negativeCell : face.positiveCell;
  if (donorId < 0) return 0;
  const cell = topology.cells[donorId]!;
  const capacity = fields.capacity[donorId]! * cell.area;
  const volume = fields.density[donorId]! * cell.area;
  const observed = Math.max(0, Math.min(capacity, volume));
  if (capacity <= 0 || observed === 0) return 0;
  if (observed === capacity) return sweep;
  const nx = fields.interfaceNormal[2 * donorId]!, ny = fields.interfaceNormal[2 * donorId + 1]!;
  if (nx === 0 && ny === 0) return low;
  const aperture = face.aperture ?? 1;
  if (aperture <= 1e-8) return low;
  const travel = Math.abs(sweep) / Math.max(face.area * aperture, 1e-20);
  if (travel > cell.widths[face.axis]) return Number.NaN;
  const minimum: [number, number] = [-0.5 * cell.widths[0], -0.5 * cell.widths[1]];
  const maximum: [number, number] = [0.5 * cell.widths[0], 0.5 * cell.widths[1]];
  if (face.negativeCell >= 0 && face.positiveCell >= 0) {
    const otherId = donorId === face.negativeCell ? face.positiveCell : face.negativeCell;
    const other = topology.cells[otherId]!;
    for (const axis of [0, 1] as const) {
      minimum[axis] = Math.max(minimum[axis], other.minimum[axis] - cell.center[axis]);
      maximum[axis] = Math.min(maximum[axis], other.maximum[axis] - cell.center[axis]);
    }
  }
  const boundary = face.center[face.axis] - cell.center[face.axis];
  minimum[face.axis] = sweep > 0 ? boundary - travel : boundary;
  maximum[face.axis] = sweep > 0 ? boundary : boundary + travel;
  const widths = [maximum[0] - minimum[0], maximum[1] - minimum[1]] as const;
  if (widths[0] < 0 || widths[1] < 0) return Number.NaN;
  const centre = [0.5 * (minimum[0] + maximum[0]),
    0.5 * (minimum[1] + maximum[1])] as const;
  const offset = f(fields.interfaceOffset[donorId]!
    - add(mul(nx, centre[0]), mul(ny, centre[1])));
  return mul(sweep, boxFraction(nx, ny, offset, widths[0], widths[1]));
}

interface SliceSubfaceEntry { readonly subfaceId: number; readonly negative: boolean }

function cellSubfaces(topology: SliceNumericalTopology, cell: number): readonly SliceSubfaceEntry[] {
  const compiled = topology.subfaceIncidences?.[cell];
  if (compiled) return compiled;
  const result: SliceSubfaceEntry[] = [];
  for (const face of topology.subfaces) {
    if (face.negativeCell === cell) result.push({ subfaceId: face.id, negative: true });
    if (face.positiveCell === cell) result.push({ subfaceId: face.id, negative: false });
  }
  return result;
}

function cellDelta(orientedFlux: number, negative: boolean): number {
  return negative ? f(-orientedFlux) : f(orientedFlux);
}

function volumeRoundoff(capacity: number): number {
  return mul(VOLUME_ROUNDOFF_RATIO, capacity);
}

function volumeValid(volume: number, capacity: number): boolean {
  const margin = volumeRoundoff(capacity);
  return capacity >= 0 && Number.isFinite(capacity)
    && volume >= -margin && volume <= add(capacity, margin);
}

function capacityAt(fields: SliceNumericalFields, topology: SliceNumericalTopology,
  cell: number, numerator: number, denominator: number): number {
  const area = topology.cells[cell]!.area;
  if (!fields.solidMotionActive) return mul(fields.capacity[cell]!, area);
  const before = mul(fields.capacityBefore?.[cell] ?? fields.capacity[cell]!, area);
  const after = mul(fields.capacityAfter?.[cell] ?? fields.capacity[cell]!, area);
  if (numerator <= 0) return before;
  if (numerator >= denominator) return after;
  return add(before, mul(f(after - before), div(numerator, Math.max(1, denominator))));
}

function nextLowerPositiveF32(value: number): number {
  const words = new Uint32Array(1), floats = new Float32Array(words.buffer);
  floats[0] = value;
  if (words[0]! > 0) words[0]!--;
  return floats[0]!;
}

function receiverFor(face: SliceNumericalSubface, flux: number): number {
  return flux >= 0 ? face.positiveCell : face.negativeCell;
}

function donorFor(face: SliceNumericalSubface, sweep: number): number {
  return sweep >= 0 ? face.negativeCell : face.positiveCell;
}

function startingVolume(volumes: Float32Array, fields: SliceNumericalFields,
  cell: number, dtm: number): number {
  return add(volumes[cell]!, mul(dtm, fields.sourceRate?.[cell] ?? 0));
}

/**
 * Literal CPU dispatch image of the resident shared low-flux limiter. The
 * proposal and commit banks are separate so a pass never observes another
 * cell's new receiver factor.
 */
function limitLowFlux(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  volumes: Float32Array, sweeps: Float32Array, dtm: number,
  receiverNumerator: number, denominator: number): boolean {
  const count = topology.cells.length;
  const current = new Float32Array(count);
  const prior = new Float32Array(count);
  const proposed = new Float32Array(count);
  const incoming = new Float32Array(count);
  if (!fields.solidMotionActive) {
    current.fill(1); prior.fill(1);
    for (const cell of topology.cells) {
      let total = 0;
      for (const entry of cellSubfaces(topology, cell.id)) {
        total = add(total, Math.max(0,
          cellDelta(fields.lowFlux[entry.subfaceId]!, entry.negative)));
      }
      incoming[cell.id] = total;
    }
  }
  const maximumPasses = 1024;
  for (let pass = 0; pass < maximumPasses; pass += 1) {
    let invalid = 0, firstInvalid = -1;
    for (const cell of topology.cells) {
      const id = cell.id;
      const volume = startingVolume(volumes, fields, id, dtm);
      const capacity = capacityAt(fields, topology, id, receiverNumerator, denominator);
      let signedDelta = 0, outgoing = 0, sweepWeight = 0;
      for (const entry of cellSubfaces(topology, id)) {
        const face = topology.subfaces[entry.subfaceId]!;
        const original = fields.lowFlux[face.id]!;
        let flux: number;
        if (fields.solidMotionActive) {
          const sweep = sweeps[face.id]!, weight = Math.abs(sweep);
          if (weight === 0) flux = 0;
          else {
            const donor = donorFor(face, sweep), receiver = receiverFor(face, sweep);
            if (donor < 0) flux = 0;
            else {
              const donorDual = current[donor]!, receiverDual = receiver >= 0 ? current[receiver]! : 0;
              const base = sweep >= 0 ? original : f(-original);
              const magnitude = Math.max(0, Math.min(weight,
                add(base, mul(weight, f(donorDual - receiverDual)))));
              flux = sweep >= 0 ? f(magnitude) : f(-magnitude);
            }
          }
          sweepWeight = add(sweepWeight, Math.abs(sweep));
        } else {
          const receiver = receiverFor(face, original);
          const factor = receiver >= 0 ? current[receiver]! : 1;
          flux = mul(original, factor);
          const delta = cellDelta(original, entry.negative);
          if (delta < 0) outgoing = add(outgoing, mul(-delta, factor));
        }
        signedDelta = add(signedDelta, cellDelta(flux, entry.negative));
      }
      const candidate = add(volume, signedDelta);
      if (fields.solidMotionActive) {
        const ready = capacity !== 0 ? volumeValid(candidate, capacity)
          : Math.abs(candidate) <= volumeRoundoff(capacityAt(fields, topology, id, 0, denominator));
        const previous = current[id]!;
        let next = previous;
        const diagonal = mul(2, sweepWeight);
        if (diagonal > 0) {
          const tau = div(1, diagonal);
          next = add(Math.max(0, add(previous, mul(tau, f(candidate - capacity)))),
            Math.min(0, add(previous, mul(tau, candidate))));
        } else if (!ready) {
          fields.fault = { stage: "transport-low", index: id,
            observed: candidate, expected: capacity };
          return false;
        }
        if (!Number.isFinite(next)) {
          fields.fault = { stage: "transport-low", index: id, observed: next, expected: capacity };
          return false;
        }
        proposed[id] = next;
        if (!ready) { invalid++; if (firstInvalid < 0) firstInvalid = id; }
      } else {
        const previous = current[id]!;
        const allowed = volumeValid(volume, capacity) ? Math.max(capacity, volume) : capacity;
        let next = previous;
        if (candidate > allowed && incoming[id]! > 0) {
          next = Math.min(previous, div(Math.max(0, add(f(allowed - volume), outgoing)), incoming[id]!));
          next = Math.min(next, Math.max(0,
            f(previous - div(f(candidate - allowed), incoming[id]!))));
          if (next === previous && previous > 0) next = nextLowerPositiveF32(previous);
        }
        proposed[id] = f(next);
        if (!volumeValid(candidate, capacity)) { invalid++; if (firstInvalid < 0) firstInvalid = id; }
      }
    }
    // A zero-invalid pass certifies the current frozen factor bank. Production
    // deliberately discards the simultaneously calculated proposals.
    if (invalid === 0) break;
    if (pass + 1 >= maximumPasses) {
      fields.fault = { stage: "transport-low", index: firstInvalid,
        observed: pass + 1, expected: invalid };
      return false;
    }
    if (fields.solidMotionActive) {
      const k = f(pass);
      for (const cell of topology.cells) {
        const x = proposed[cell.id]!, old = prior[cell.id]!;
        prior[cell.id] = x;
        current[cell.id] = add(x, mul(div(k, add(k, 3)), f(x - old)));
      }
    } else current.set(proposed);
  }
  for (const face of topology.subfaces) {
    const original = fields.lowFlux[face.id]!;
    if (fields.solidMotionActive) {
      const sweep = sweeps[face.id]!, weight = Math.abs(sweep);
      if (weight === 0) fields.lowFlux[face.id] = 0;
      else {
        const donor = donorFor(face, sweep), receiver = receiverFor(face, sweep);
        if (donor < 0) fields.lowFlux[face.id] = 0;
        else {
          const base = sweep >= 0 ? original : f(-original);
          const magnitude = Math.max(0, Math.min(weight, add(base,
            mul(weight, f(current[donor]! - (receiver >= 0 ? current[receiver]! : 0))))));
          fields.lowFlux[face.id] = sweep >= 0 ? f(magnitude) : f(-magnitude);
        }
      }
    } else {
      const receiver = receiverFor(face, original);
      fields.lowFlux[face.id] = mul(original, receiver >= 0 ? current[receiver]! : 1);
    }
  }
  return true;
}

function closingCellBefore(topology: SliceNumericalTopology, a: number, b: number): boolean {
  if (b < 0) return true;
  const x = topology.cells[a]!.center, y = topology.cells[b]!.center;
  return x[0] !== y[0] ? x[0] < y[0] : x[1] < y[1];
}

function closingFaceBefore(topology: SliceNumericalTopology, a: number, b: number): boolean {
  if (b < 0) return true;
  const af = topology.subfaces[a]!, bf = topology.subfaces[b]!;
  const ar = topology.rows[af.rowId]!, br = topology.rows[bf.rowId]!;
  if (ar.axis !== br.axis) return ar.axis < br.axis;
  if (ar.center[0] !== br.center[0]) return ar.center[0] < br.center[0];
  if (ar.center[1] !== br.center[1]) return ar.center[1] < br.center[1];
  if (af.area !== bf.area) return af.area < bf.area;
  if (af.negativeCell !== bf.negativeCell) {
    if (af.negativeCell < 0) return true;
    if (bf.negativeCell < 0) return false;
    return closingCellBefore(topology, af.negativeCell, bf.negativeCell);
  }
  if (af.positiveCell !== bf.positiveCell) {
    if (af.positiveCell < 0) return true;
    if (bf.positiveCell < 0) return false;
    return closingCellBefore(topology, af.positiveCell, bf.positiveCell);
  }
  return false;
}

function faceOther(face: SliceNumericalSubface, negative: boolean): number {
  return negative ? face.positiveCell : face.negativeCell;
}

function closingFaceBudget(face: SliceNumericalSubface, sweep: number,
  main: number): readonly [number, number, number] {
  const donor = donorFor(face, sweep);
  const maximum = donor < 0 ? 0 : Math.abs(sweep);
  const magnitude = sweep >= 0 ? main : f(-main);
  return [magnitude, f(maximum - magnitude), maximum];
}

function cellCoverageMissing(topology: SliceNumericalTopology, cell: number): boolean {
  const negative = new Float32Array(2), positive = new Float32Array(2);
  for (const entry of cellSubfaces(topology, cell)) {
    const face = topology.subfaces[entry.subfaceId]!;
    if (entry.negative) positive[face.axis] = add(positive[face.axis]!, face.area);
    else negative[face.axis] = add(negative[face.axis]!, face.area);
  }
  const record = topology.cells[cell]!;
  for (const axis of [0, 1] as const) {
    const expected = div(record.area, record.widths[axis]);
    const tolerance = mul(9.5367431640625e-7, expected);
    if (Math.abs(f(negative[axis]! - expected)) > tolerance
      || Math.abs(f(positive[axis]! - expected)) > tolerance) return true;
  }
  return false;
}

function closingOtherAmount(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  volumes: Float32Array, flux: Float32Array, cell: number, parent: number,
  dtm: number): number {
  let delta = 0;
  for (const entry of cellSubfaces(topology, cell)) {
    if (entry.subfaceId === parent) continue;
    delta = add(delta, cellDelta(flux[entry.subfaceId]!, entry.negative));
  }
  return add(startingVolume(volumes, fields, cell, dtm), delta);
}

function closingOrderedAmount(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  volumes: Float32Array, flux: Float32Array, cell: number, parent: number,
  dtm: number): number {
  const base = closingOtherAmount(topology, fields, volumes, flux, cell, parent, dtm);
  if (parent < 0) return base;
  const face = topology.subfaces[parent]!;
  return add(base, cellDelta(flux[parent]!, face.negativeCell === cell));
}

/** Deterministic resident zero-capacity residual tree and terminal face replacement. */
function allocateClosingResidual(topology: SliceNumericalTopology, fields: SliceNumericalFields,
  volumes: Float32Array, sweeps: Float32Array, dtm: number,
  receiverNumerator: number, denominator: number): Int32Array | null {
  const parents = new Int32Array(topology.cells.length).fill(-1);
  if (!fields.solidMotionActive) return parents;
  const closing = topology.cells.filter(cell =>
    capacityAt(fields, topology, cell.id, receiverNumerator, denominator) === 0).map(cell => cell.id);
  const seen = new Set<number>();
  for (const seed of closing) {
    if (seen.has(seed)) continue;
    const members = [seed]; seen.add(seed);
    for (let cursor = 0; cursor < members.length; cursor += 1) {
      const current = members[cursor]!;
      for (const entry of cellSubfaces(topology, current)) {
        const face = topology.subfaces[entry.subfaceId]!, other = faceOther(face, entry.negative);
        if (other < 0 || seen.has(other)
          || capacityAt(fields, topology, other, receiverNumerator, denominator) !== 0) continue;
        const budget = closingFaceBudget(face, sweeps[face.id]!, fields.lowFlux[face.id]!);
        if (Math.min(budget[0], budget[1]) <= 0) continue;
        if (members.length === 128) {
          fields.fault = { stage: "transport-closing", index: seed, observed: 128, expected: 0 };
          return null;
        }
        seen.add(other); members.push(other);
      }
    }
    let leader = members[0]!;
    for (const member of members) if (closingCellBefore(topology, member, leader)) leader = member;
    // The GPU launches each cell, but only this lexicographic leader mutates a component.
    if (seed !== leader) continue;
    let componentBudget = 0, totalResidual = 0, anyResidual = false;
    for (const member of members) {
      const residual = closingOtherAmount(topology, fields, volumes, fields.lowFlux,
        member, -1, dtm);
      const tolerance = volumeRoundoff(capacityAt(fields, topology, member, 0, denominator));
      if (!(Math.abs(residual) <= tolerance)) {
        fields.fault = { stage: "transport-closing", index: member,
          observed: residual, expected: tolerance };
        return null;
      }
      componentBudget = add(componentBudget, tolerance);
      totalResidual = add(totalResidual, residual);
      anyResidual ||= residual !== 0;
    }
    if (!anyResidual) continue;
    const depths = new Int32Array(members.length).fill(-1);
    const processed = new Uint8Array(members.length);
    let rootIndex = -1, rootFace = -1, rootBudget = -1;
    for (let index = 0; index < members.length; index += 1) {
      const member = members[index]!;
      for (const entry of cellSubfaces(topology, member)) {
        const face = topology.subfaces[entry.subfaceId]!, other = faceOther(face, entry.negative);
        if (other >= 0 && (capacityAt(fields, topology, other, receiverNumerator, denominator) <= 0
          || cellCoverageMissing(topology, other))) continue;
        const budget = closingFaceBudget(face, sweeps[face.id]!, fields.lowFlux[face.id]!);
        const signedCorrection = entry.negative ? totalResidual : f(-totalResidual);
        const correction = sweeps[face.id]! >= 0 ? signedCorrection : f(-signedCorrection);
        const available = correction === 0 ? Math.min(budget[0], budget[1])
          : correction >= 0 ? budget[1] : budget[0];
        if (available < componentBudget) continue;
        const preferred = available > rootBudget || available === rootBudget
          && (rootIndex < 0 || closingCellBefore(topology, member, members[rootIndex]!)
            || member === members[rootIndex] && closingFaceBefore(topology, face.id, rootFace));
        if (preferred) { rootIndex = index; rootFace = face.id; rootBudget = available; }
      }
    }
    if (rootIndex < 0) {
      fields.fault = { stage: "transport-closing", index: leader,
        observed: totalResidual, expected: -1 };
      return null;
    }
    parents[members[rootIndex]!] = rootFace; depths[rootIndex] = 0;
    for (let attached = 1; attached < members.length; attached += 1) {
      let chosen = -1, chosenFace = -1, chosenDepth = -1, best = -1;
      for (let index = 0; index < members.length; index += 1) {
        if (depths[index]! >= 0) continue;
        const member = members[index]!;
        for (const entry of cellSubfaces(topology, member)) {
          const face = topology.subfaces[entry.subfaceId]!, other = faceOther(face, entry.negative);
          const parentIndex = members.indexOf(other);
          if (parentIndex < 0 || depths[parentIndex]! < 0) continue;
          const budget = closingFaceBudget(face, sweeps[face.id]!, fields.lowFlux[face.id]!);
          const available = Math.min(budget[0], budget[1]);
          if (available < componentBudget) continue;
          const preferred = available > best || available === best
            && (chosen < 0 || closingCellBefore(topology, member, members[chosen]!)
              || member === members[chosen] && closingFaceBefore(topology, face.id, chosenFace));
          if (preferred) {
            chosen = index; chosenFace = face.id; chosenDepth = depths[parentIndex]! + 1; best = available;
          }
        }
      }
      if (chosen < 0) {
        fields.fault = { stage: "transport-closing", index: leader,
          observed: totalResidual, expected: -2 };
        return null;
      }
      parents[members[chosen]!] = chosenFace; depths[chosen] = chosenDepth;
    }
    for (let completed = 0; completed < members.length; completed += 1) {
      let chosen = -1;
      for (let index = 0; index < members.length; index += 1) {
        if (processed[index]) continue;
        if (chosen < 0 || depths[index]! > depths[chosen]!
          || depths[index] === depths[chosen]
          && closingCellBefore(topology, members[index]!, members[chosen]!)) chosen = index;
      }
      const member = members[chosen]!, faceId = parents[member]!, face = topology.subfaces[faceId]!;
      const base = closingOtherAmount(topology, fields, volumes, fields.lowFlux,
        member, faceId, dtm);
      const replacement = face.negativeCell === member ? base : f(-base);
      const original = fields.lowFlux[faceId]!, sweep = sweeps[faceId]!;
      const magnitude = sweep >= 0 ? replacement : f(-replacement);
      const maximum = closingFaceBudget(face, sweep, original)[2];
      if (!(magnitude >= 0 && magnitude <= maximum
        && Math.abs(f(replacement - original)) <= componentBudget)) {
        fields.fault = { stage: "transport-closing", index: member,
          observed: replacement, expected: maximum };
        return null;
      }
      fields.lowFlux[faceId] = replacement;
      const remaining = closingOrderedAmount(topology, fields, volumes, fields.lowFlux,
        member, faceId, dtm);
      if (remaining !== 0) {
        fields.fault = { stage: "transport-closing", index: member,
          observed: remaining, expected: 0 };
        return null;
      }
      processed[chosen] = 1;
    }
  }
  return parents;
}

/** Production geometric FCT microsteps over one frozen projected face field. */
export interface SliceTransportMicrostepReceipt {
  readonly interfaceNormal: Float32Array;
  readonly interfaceOffset: Float32Array;
  readonly sweep: Float32Array;
  /** Frozen monotone flux before the synchronized receiver limiter. */
  readonly initialLowFlux: Float32Array;
  readonly highFlux: Float32Array;
  /** Low flux after static receiver factors or moving FISTA/closing allocation. */
  readonly lowFlux: Float32Array;
  readonly lowVolume: Float32Array;
  readonly positiveBudget: Float32Array;
  readonly negativeBudget: Float32Array;
  readonly increase: Float32Array;
  readonly decrease: Float32Array;
  readonly limitedFlux: Float32Array;
  readonly nextVolume: Float32Array;
}

export function transportSliceVolume(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  dt: number,
  onMicrostep?: (step: number, dt: number, receipt: SliceTransportMicrostepReceipt) => void,
  onMicrostepCommit?: (step: number, dt: number) => void,
): number {
  const volumes = Float32Array.from(topology.cells,
    cell => mul(fields.density[cell.id]!, cell.area));
  let cfl = 0;
  for (const cell of topology.cells) {
    const oldCapacity = capacityAt(fields, topology, cell.id, 0, 1);
    const finalCapacity = capacityAt(fields, topology, cell.id, 1, 1);
    if (!volumeValid(volumes[cell.id]!, oldCapacity)) {
      fields.fault = { stage: "transport-initialize", index: cell.id,
        observed: volumes[cell.id]!, expected: oldCapacity };
      return 1;
    }
    let cflCapacity = Math.max(oldCapacity, finalCapacity);
    if (oldCapacity > 0 && finalCapacity > 0) cflCapacity = Math.min(oldCapacity, finalCapacity);
    let outgoing = 0, prism = 0, totalRate = 0;
    for (const entry of cellSubfaces(topology, cell.id)) {
      const face = topology.subfaces[entry.subfaceId]!, row = topology.rows[face.rowId]!;
      const aperture = row.openFraction ?? 1, wall = row.solidVelocity ?? 0;
      const stored = fields.faceVelocity[row.id]!;
      const rate = mul(face.area, f(stored - mul(f(1 - aperture), wall)));
      const flow = Math.abs(rate); totalRate = add(totalRate, flow);
      outgoing = add(outgoing, Math.max(0, entry.negative ? rate : f(-rate)));
      if (aperture > 1e-8) prism = Math.max(prism,
        div(flow, mul(mul(face.area, aperture), cell.widths[row.axis])));
    }
    const source = fields.sourceRate?.[cell.id] ?? 0;
    if (cflCapacity === 0) {
      if (add(totalRate, source) > 0) fields.fault = { stage: "transport-plan", index: cell.id,
        observed: add(totalRate, source), expected: cflCapacity };
      continue;
    }
    cfl = Math.max(cfl, mul(dt, Math.max(div(add(outgoing, source), cflCapacity), prism)));
  }
  const steps = Math.max(1, Math.ceil(2 * cfl));
  if (steps > 128) {
    fields.fault = { stage: "transport-plan", index: 0, observed: steps, expected: 128 };
    return steps;
  }
  const dtm = div(dt, steps), sweeps = new Float32Array(topology.subfaces.length);
  fields.lowFlux = new Float32Array(topology.subfaces.length);
  fields.highFlux = new Float32Array(topology.subfaces.length);
  fields.limitedFlux = new Float32Array(topology.subfaces.length);
  for (let step = 0; step < steps; step += 1) {
    reconstructSliceInterfaces(topology, fields);
    const reconstructedNormal = onMicrostep ? Float32Array.from(fields.interfaceNormal) : fields.interfaceNormal;
    const reconstructedOffset = onMicrostep ? Float32Array.from(fields.interfaceOffset) : fields.interfaceOffset;
    for (const face of topology.subfaces) {
      const row = topology.rows[face.rowId]!, aperture = row.openFraction ?? 1;
      const stored = fields.faceVelocity[row.id]!, wall = row.solidVelocity ?? 0;
      const rate = mul(face.area, f(stored - mul(f(1 - aperture), wall)));
      const sweep = mul(rate, dtm); sweeps[face.id] = sweep;
      const n = face.negativeCell, p = face.positiveCell;
      const vn = n >= 0 ? volumes[n]! : 0, vp = p >= 0 ? volumes[p]! : 0;
      const cn = n >= 0 ? capacityAt(fields, topology, n, step, steps) : 0;
      const cp = p >= 0 ? capacityAt(fields, topology, p, step, steps) : 0;
      const observedN = Math.max(0, Math.min(Math.max(0, cn), vn));
      const observedP = Math.max(0, Math.min(Math.max(0, cp), vp));
      const low = sweep > 0 ? (cn > 0 ? mul(sweep, div(observedN, cn)) : 0)
        : sweep < 0 ? (cp > 0 ? mul(sweep, div(observedP, cp)) : 0) : 0;
      const high = highFluxForSubface(topology, fields, face, sweep, low);
      if (!Number.isFinite(high)) {
        fields.fault = { stage: "transport-flux", index: face.id, observed: high, expected: low };
        return steps;
      }
      fields.lowFlux[face.id] = low; fields.highFlux[face.id] = high;
    }
    const initialLowFlux = onMicrostep ? Float32Array.from(fields.lowFlux) : fields.lowFlux;
    if (!limitLowFlux(topology, fields, volumes, sweeps, dtm, step + 1, steps)) return steps;
    const closingParents = allocateClosingResidual(topology, fields, volumes, sweeps, dtm,
      step + 1, steps);
    if (!closingParents) return steps;

    const lowVolume = new Float32Array(topology.cells.length);
    const increase = new Float32Array(topology.cells.length).fill(1);
    const decrease = new Float32Array(topology.cells.length).fill(1);
    const positiveBudgetByCell = new Float32Array(topology.cells.length);
    const negativeBudgetByCell = new Float32Array(topology.cells.length);
    for (const cell of topology.cells) {
      let lowDelta = 0, positiveBudget = 0, negativeBudget = 0;
      for (const entry of cellSubfaces(topology, cell.id)) {
        const face = topology.subfaces[entry.subfaceId]!;
        lowDelta = add(lowDelta, cellDelta(fields.lowFlux[face.id]!, entry.negative));
        const anti = f(fields.highFlux[face.id]! - fields.lowFlux[face.id]!);
        const antiDelta = cellDelta(anti, entry.negative);
        positiveBudget = add(positiveBudget, Math.max(antiDelta, 0));
        negativeBudget = add(negativeBudget, Math.max(-antiDelta, 0));
      }
      const capacity = capacityAt(fields, topology, cell.id, step + 1, steps);
      let low = add(startingVolume(volumes, fields, cell.id, dtm), lowDelta);
      if (fields.solidMotionActive && capacity === 0) low = closingOrderedAmount(
        topology, fields, volumes, fields.lowFlux, cell.id, closingParents[cell.id]!, dtm);
      lowVolume[cell.id] = low;
      positiveBudgetByCell[cell.id] = positiveBudget;
      negativeBudgetByCell[cell.id] = negativeBudget;
      if (!volumeValid(low, capacity)) {
        fields.fault = { stage: "transport-low", index: cell.id, observed: low, expected: capacity };
        return steps;
      }
      const observed = Math.max(0, Math.min(Math.max(0, capacity), low));
      if (positiveBudget > 0) increase[cell.id] = Math.min(1, div(f(capacity - observed), positiveBudget));
      if (negativeBudget > 0) decrease[cell.id] = Math.min(1, div(observed, negativeBudget));
    }
    for (const face of topology.subfaces) {
      const low = fields.lowFlux[face.id]!, anti = f(fields.highFlux[face.id]! - low);
      const n = face.negativeCell, p = face.positiveCell;
      const nInc = n >= 0 ? increase[n]! : 1, nDec = n >= 0 ? decrease[n]! : 1;
      const pInc = p >= 0 ? increase[p]! : 1, pDec = p >= 0 ? decrease[p]! : 1;
      let factor = anti >= 0 ? Math.min(nDec, pInc) : Math.min(nInc, pDec);
      if (fields.solidMotionActive && (n >= 0
        && capacityAt(fields, topology, n, step + 1, steps) === 0
        || p >= 0 && capacityAt(fields, topology, p, step + 1, steps) === 0)) factor = 0;
      fields.limitedFlux[face.id] = add(low, mul(factor, f(fields.highFlux[face.id]! - low)));
    }
    const next = new Float32Array(topology.cells.length);
    for (const cell of topology.cells) {
      let delta = 0;
      for (const entry of cellSubfaces(topology, cell.id)) {
        delta = add(delta, cellDelta(fields.limitedFlux[entry.subfaceId]!, entry.negative));
      }
      const capacity = capacityAt(fields, topology, cell.id, step + 1, steps);
      next[cell.id] = add(startingVolume(volumes, fields, cell.id, dtm), delta);
      if (fields.solidMotionActive && capacity === 0) next[cell.id] = closingOrderedAmount(
        topology, fields, volumes, fields.limitedFlux, cell.id, closingParents[cell.id]!, dtm);
      if (!volumeValid(next[cell.id]!, capacity)) {
        fields.fault = { stage: "transport-commit", index: cell.id,
          observed: next[cell.id]!, expected: capacity };
        return steps;
      }
    }
    volumes.set(next);
    for (const cell of topology.cells) {
      fields.density[cell.id] = div(volumes[cell.id]!, cell.area);
      fields.gamma[cell.id] = 1;
    }
    onMicrostepCommit?.(step, dtm);
    onMicrostep?.(step, dtm, {
      interfaceNormal: reconstructedNormal, interfaceOffset: reconstructedOffset,
      sweep: Float32Array.from(sweeps), initialLowFlux,
      highFlux: Float32Array.from(fields.highFlux), lowFlux: Float32Array.from(fields.lowFlux),
      lowVolume: Float32Array.from(lowVolume), positiveBudget: positiveBudgetByCell,
      negativeBudget: negativeBudgetByCell, increase, decrease,
      limitedFlux: Float32Array.from(fields.limitedFlux), nextVolume: Float32Array.from(next),
    });
  }
  return steps;
}

/** Narrow one-packet seam for raw WGSL differential fixtures. */
export function transportSliceVolumeMicrostep(
  topology: SliceNumericalTopology,
  fields: SliceNumericalFields,
  dt: number,
): SliceTransportMicrostepReceipt {
  let receipt: SliceTransportMicrostepReceipt | undefined;
  const count = transportSliceVolume(topology, fields, dt,
    (_step, _dt, value) => { receipt = value; });
  if (count !== 1 || !receipt) {
    throw new RangeError(`slice validation microstep requires a one-packet CFL, observed ${count}`);
  }
  return receipt;
}
