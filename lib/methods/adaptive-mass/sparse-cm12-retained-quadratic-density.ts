import { retainedAffineRamp, type RetainedAffineRamp } from "./sparse-cm12-retained-affine-density";

/** Declared diffuse quadratic density primitives. The geometry is q=0.5;
 * integral q is deliberately not replaced by enclosed sharp volume. */
export type QuadraticDensityPoint = readonly [number, number, number];
export interface QuadraticDensityBox { readonly lower: QuadraticDensityPoint; readonly upper: QuadraticDensityPoint }
export type QuadraticDensityCoefficients = readonly [number, number, number, number, number, number, number, number, number, number];
export interface RetainedQuadraticDensity {
  readonly kind: "clamped-quadratic";
  readonly generation: number;
  readonly origin: QuadraticDensityPoint;
  readonly scale: QuadraticDensityPoint;
  /** 1,x,y,z,xx,yy,zz,xy,xz,yz in the declared local frame. */
  readonly coefficients: QuadraticDensityCoefficients;
}
type Interval = readonly [number, number];
const bits = new DataView(new ArrayBuffer(8));
function next(x: number, up: boolean): number {
  if (Number.isNaN(x) || x === (up ? Infinity : -Infinity)) return x;
  if (x === 0) return up ? Number.MIN_VALUE : -Number.MIN_VALUE;
  bits.setFloat64(0, x); let word = bits.getBigUint64(0);
  word += ((x > 0) === up) ? 1n : -1n; bits.setBigUint64(0, word);
  return bits.getFloat64(0);
}
const add = (a: Interval, b: Interval): Interval => [next(a[0] + b[0], false), next(a[1] + b[1], true)];
const subtract = (a: Interval, b: Interval): Interval => add(a, [-b[1], -b[0]]);
function multiply(a: Interval, b: Interval): Interval {
  const products = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]];
  return [next(Math.min(...products), false), next(Math.max(...products), true)];
}
const constant = (x: number): Interval => [x, x];
function dividePositive(a: Interval, b: Interval): Interval {
  if (!(b[0] > 0)) throw new Error("Interval denominator must be positive");
  return multiply(a, [next(1 / b[1], false), next(1 / b[0], true)]);
}
function square(a: Interval): Interval {
  return [a[0] <= 0 && a[1] >= 0 ? 0 : next(Math.min(a[0] ** 2, a[1] ** 2), false),
    next(Math.max(a[0] ** 2, a[1] ** 2), true)];
}
const clamp = (v: number) => Math.max(0, Math.min(1, v));
export function retainedQuadraticDensity(input: {
  origin: QuadraticDensityPoint; scale: QuadraticDensityPoint; coefficients: QuadraticDensityCoefficients; generation?: number;
}): RetainedQuadraticDensity {
  const generation = input.generation ?? 1;
  if (!Number.isSafeInteger(generation) || generation < 1 || input.coefficients.length !== 10 || [...input.origin, ...input.scale, ...input.coefficients].some(v => !Number.isFinite(v))
    || input.scale.some(v => v <= 0)) throw new Error("Invalid declared quadratic density");
  return Object.freeze({ kind: "clamped-quadratic", generation,
    origin: Object.freeze([...input.origin]) as unknown as QuadraticDensityPoint,
    scale: Object.freeze([...input.scale]) as unknown as QuadraticDensityPoint,
    coefficients: Object.freeze([...input.coefficients]) as unknown as QuadraticDensityCoefficients });
}
/** q=clamp(.5+(R²-r²)/(2 R w),0,1). w sets the slope at r=R;
 * unlike an affine ramp, the two radial saturation distances are asymmetric. */
export function retainedSphereDensity(center: QuadraticDensityPoint, radius: number, transitionWidth: number, generation = 1) {
  if (!(radius > 0) || !(transitionWidth > 0) || !Number.isFinite(radius + transitionWidth)) throw new Error("Invalid density sphere");
  const ratio = radius / (2 * transitionWidth);
  return retainedQuadraticDensity({ origin: center, scale: [radius, radius, radius], generation,
    coefficients: [0.5 + ratio, 0, 0, 0, -ratio, -ratio, -ratio, 0, 0, 0] });
}
export function evaluateRetainedQuadraticDensity(field: RetainedQuadraticDensity, point: QuadraticDensityPoint) {
  if (point.some(v => !Number.isFinite(v))) throw new Error("Nonfinite density point");
  const [x, y, z] = point.map((v, i) => (v - field.origin[i]) / field.scale[i]), c = field.coefficients;
  return clamp(c[0] + c[1] * x + c[2] * y + c[3] * z + c[4] * x * x + c[5] * y * y + c[6] * z * z
    + c[7] * x * y + c[8] * x * z + c[9] * y * z);
}
/** Gradient inside the unsaturated ramp; zero in saturated bulk. A clamp
 * boundary has two one-sided derivatives, so no unique normal is claimed. */
export function gradientRetainedQuadraticDensity(field: RetainedQuadraticDensity, point: QuadraticDensityPoint): QuadraticDensityPoint {
  const value = evaluateRetainedQuadraticDensity(field, point);
  if (value === 0 || value === 1) return [0, 0, 0];
  const [x, y, z] = point.map((v, i) => (v - field.origin[i]) / field.scale[i]), c = field.coefficients;
  return [(c[1] + 2 * c[4] * x + c[7] * y + c[8] * z) / field.scale[0],
    (c[2] + 2 * c[5] * y + c[7] * x + c[9] * z) / field.scale[1],
    (c[3] + 2 * c[6] * z + c[8] * x + c[9] * y) / field.scale[2]];
}
interface Node { readonly box: QuadraticDensityBox; readonly bound: Interval; readonly axis: number }
function node(field: RetainedQuadraticDensity, box: QuadraticDensityBox): Node {
  const c = field.coefficients, coordinates: Interval[] = [], means: Interval[] = [], second: Interval[] = [];
  let volume: Interval = [1, 1];
  for (let axis = 0; axis < 3; axis++) {
    const lo = box.lower[axis], hi = box.upper[axis];
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || !(hi > lo)) throw new Error("Invalid quadratic integration box");
    const a = dividePositive(subtract(constant(lo), constant(field.origin[axis])), constant(field.scale[axis]));
    const b = dividePositive(subtract(constant(hi), constant(field.origin[axis])), constant(field.scale[axis]));
    coordinates.push([a[0], b[1]]);
    const middle = multiply(add(a, b), constant(0.5)); means.push(middle);
    second.push(add(square(middle), dividePositive(square(subtract(b, a)), constant(12))));
    volume = multiply(volume, subtract(constant(hi), constant(lo)));
  }
  const pairs = [[0, 1], [0, 2], [1, 2]];
  const rangeBasis: Interval[] = [[1, 1], ...coordinates, ...coordinates.map(square), ...pairs.map(([a, b]) => multiply(coordinates[a], coordinates[b]))];
  const meanBasis: Interval[] = [[1, 1], ...means, ...second, ...pairs.map(([a, b]) => multiply(means[a], means[b]))];
  let range: Interval = [0, 0], mean: Interval = [0, 0];
  for (let i = 0; i < 10; i++) if (c[i] !== 0) {
    range = add(range, multiply(constant(c[i]), rangeBasis[i]));
    mean = add(mean, multiply(constant(c[i]), meanBasis[i]));
  }
  const [low, high] = range;
  let bound: Interval;
  if (high <= 0) bound = [0, 0];
  else if (low >= 1) bound = [1, 1];
  else if (low >= 0 && high <= 1) bound = [clamp(mean[0]), clamp(mean[1])];
  else if (low < 0 && high <= 1) {
    const chord = multiply(constant(high), dividePositive(subtract(mean, constant(low)), subtract(constant(high), constant(low))));
    bound = [clamp(mean[0]), clamp(chord[1])];
  } else if (low >= 0) {
    const chord = add(constant(low), multiply(subtract(constant(1), constant(low)),
      dividePositive(subtract(mean, constant(low)), subtract(constant(high), constant(low)))));
    bound = [clamp(chord[0]), clamp(mean[1])];
  } else {
    const lower = dividePositive(mean, constant(high));
    const upper = dividePositive(subtract(mean, constant(low)), subtract(constant(1), constant(low)));
    bound = [clamp(lower[0]), clamp(upper[1])];
  }
  const weighted = multiply(bound, volume);
  if (!weighted.every(Number.isFinite) || weighted[1] < weighted[0]) throw new Error("Quadratic interval overflow or inconsistent enclosure");
  const scores = coordinates.map((r, axis) => {
    let derivative = Math.abs(c[1 + axis]) + 2 * Math.abs(c[4 + axis]) * Math.max(Math.abs(r[0]), Math.abs(r[1]));
    pairs.forEach(([a, b], k) => { if (a === axis || b === axis) {
      const other = coordinates[a === axis ? b : a]; derivative += Math.abs(c[7 + k]) * Math.max(Math.abs(other[0]), Math.abs(other[1]));
    } });
    return (r[1] - r[0]) * derivative;
  });
  return { box, bound: [Math.max(0, weighted[0]), weighted[1]], axis: scores.indexOf(Math.max(...scores)) };
}
export interface QuadraticDensityIntegralReceipt {
  readonly lower: number; readonly upper: number; readonly estimate: number;
  readonly toleranceMet: boolean; readonly leaves: number; readonly evaluatedBoxes: number;
}
/** Certified for the declared binary64 polynomial through outward-rounded
 * interval operations. A budget exhaustion returns its unresolved enclosure.
 * This is a CPU oracle; it is not a GPU or runtime performance claim. */
export function integrateRetainedQuadraticDensity(field: RetainedQuadraticDensity, box: QuadraticDensityBox,
  options: { absoluteTolerance?: number; maximumLeaves?: number } = {}): QuadraticDensityIntegralReceipt {
  const tolerance = options.absoluteTolerance ?? 1e-5, maximum = options.maximumLeaves ?? 8192;
  if (!Number.isFinite(tolerance) || tolerance <= 0 || !Number.isSafeInteger(maximum) || maximum < 1) throw new Error("Invalid quadratic integration budget");
  const heap: Node[] = [];
  const error = (n: Node) => n.bound[1] - n.bound[0];
  const push = (n: Node) => {
    let at = heap.length; heap.push(n);
    while (at > 0) { const parent = (at - 1) >> 1; if (error(heap[parent]) >= error(n)) break;
      heap[at] = heap[parent]; at = parent; } heap[at] = n;
  };
  const pop = () => {
    const result = heap[0], last = heap.pop()!;
    if (heap.length) { let at = 0; heap[0] = last;
      for (;;) { let child = 2 * at + 1; if (child >= heap.length) break;
        if (child + 1 < heap.length && error(heap[child + 1]) > error(heap[child])) child++;
        if (error(last) >= error(heap[child])) break; heap[at] = heap[child]; at = child; }
      heap[at] = last; }
    return result;
  };
  const first = node(field, box); push(first);
  let lower = first.bound[0], upper = first.bound[1], evaluatedBoxes = 1;
  while (upper - lower > tolerance && heap.length < maximum) {
    const parent = pop(), axis = parent.axis;
    const middle = parent.box.lower[axis] + (parent.box.upper[axis] - parent.box.lower[axis]) / 2;
    if (!(middle > parent.box.lower[axis] && middle < parent.box.upper[axis])) { push(parent); break; }
    const leftUpper = [...parent.box.upper], rightLower = [...parent.box.lower]; leftUpper[axis] = middle; rightLower[axis] = middle;
    const left = node(field, { lower: parent.box.lower, upper: leftUpper as unknown as QuadraticDensityPoint });
    const right = node(field, { lower: rightLower as unknown as QuadraticDensityPoint, upper: parent.box.upper });
    lower = next(next(next(lower - parent.bound[0], false) + left.bound[0], false) + right.bound[0], false);
    upper = next(next(next(upper - parent.bound[1], true) + left.bound[1], true) + right.bound[1], true);
    push(left); push(right); evaluatedBoxes += 2;
  }
  return Object.freeze({ lower: Math.max(0, lower), upper, estimate: lower + (upper - lower) / 2,
    toleranceMet: next(upper - lower, true) <= tolerance, leaves: heap.length, evaluatedBoxes });
}


/** Initialization takes declared density geometry, never ambiguous cell means.
 * Each result owns only numeric coefficients/parameters; no authored callback
 * remains. All primitives represent diffuse q in [0,1]. A pool is a plane
 * clipped by the caller's physical/open domain, not an implicit side-wall CSG.
 */
export type DeclaredRetainedDensityPrimitive =
  | { readonly kind: "pool"; readonly height: number; readonly transitionWidth: number; readonly generation?: number }
  | { readonly kind: "plane"; readonly origin: QuadraticDensityPoint; readonly normal: QuadraticDensityPoint;
      readonly offset: number; readonly transitionWidth: number; readonly generation?: number }
  | { readonly kind: "sphere"; readonly center: QuadraticDensityPoint; readonly radius: number;
      readonly transitionWidth: number; readonly generation?: number }
  | { readonly kind: "quadratic"; readonly origin: QuadraticDensityPoint; readonly scale: QuadraticDensityPoint;
      readonly coefficients: QuadraticDensityCoefficients; readonly generation?: number };
export function initializeRetainedDensityPrimitive(primitive: DeclaredRetainedDensityPrimitive): RetainedAffineRamp | RetainedQuadraticDensity {
  switch (primitive.kind) {
    case "pool": return retainedAffineRamp({ origin: [0, 0, 0], normal: [0, 1, 0], offset: primitive.height,
      transitionWidth: primitive.transitionWidth, generation: primitive.generation });
    case "plane": return retainedAffineRamp(primitive);
    case "sphere": return retainedSphereDensity(primitive.center, primitive.radius, primitive.transitionWidth, primitive.generation);
    case "quadratic": return retainedQuadraticDensity(primitive);
  }
}
