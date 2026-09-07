/** Retained physical-coordinate ramp/crease density algebra. Support or physics
 * splits never change transition width or refit the accepted field. Integrals
 * clip a normalized unit cube, avoiding divided-difference cancellation for
 * nearly axis-aligned planes and anisotropic physical query boxes.
 */
export type AffineDensityPoint = readonly [number, number, number];
export interface AffineDensityBox { readonly lower: AffineDensityPoint; readonly upper: AffineDensityPoint }
export interface RetainedAffineRamp {
  readonly kind: "clamped-affine";
  readonly generation: number;
  readonly origin: AffineDensityPoint;
  readonly normal: AffineDensityPoint;
  readonly offset: number;
  readonly transitionWidth: number;
}
export interface RetainedAffineFeature {
  readonly kind: "minimum" | "maximum";
  readonly generation: number;
  readonly branches: readonly [RetainedAffineRamp, RetainedAffineRamp];
}
export type RetainedAffineDensity = RetainedAffineRamp | RetainedAffineFeature;
type V = [number, number, number];
type Plane = readonly [number, number, number, number];
type Polyhedron = V[][];
const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const sub = (a: readonly number[], b: readonly number[]): V => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: V, b: V): V => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
function generationValid(generation: number) {
  if (!Number.isSafeInteger(generation) || generation < 1) throw new Error("Invalid retained density generation");
}
function boxWidths(box: AffineDensityBox): V {
  const widths = sub(box.upper, box.lower);
  if ([...box.lower, ...box.upper].some(v => !Number.isFinite(v)) || widths.some(w => !Number.isFinite(w) || w <= 0)) {
    throw new Error("Expected a finite positive-volume density box");
  }
  return widths;
}
export function retainedAffineRamp(input: {
  origin: AffineDensityPoint; normal: AffineDensityPoint; offset: number; transitionWidth: number; generation?: number;
}): RetainedAffineRamp {
  const generation = input.generation ?? 1; generationValid(generation);
  const length = Math.hypot(...input.normal);
  if ([...input.origin, ...input.normal, input.offset, input.transitionWidth].some(v => !Number.isFinite(v))
    || !(length > 0) || !Number.isFinite(length) || !(input.transitionWidth > 0)) throw new Error("Invalid retained affine ramp");
  return Object.freeze({ kind: "clamped-affine", generation,
    origin: Object.freeze([...input.origin]) as unknown as AffineDensityPoint,
    normal: Object.freeze(input.normal.map(v => v / length)) as unknown as AffineDensityPoint,
    offset: input.offset, transitionWidth: input.transitionWidth });
}
export function retainedAffineFeature(kind: "minimum" | "maximum",
  branches: readonly [RetainedAffineRamp, RetainedAffineRamp], generation = Math.max(...branches.map(b => b.generation))): RetainedAffineFeature {
  generationValid(generation);
  if (kind !== "minimum" && kind !== "maximum") throw new Error("Unsupported feature operator");
  return Object.freeze({ kind, generation, branches: Object.freeze([...branches]) as unknown as RetainedAffineFeature["branches"] });
}
function rawRamp(field: RetainedAffineRamp, point: AffineDensityPoint) {
  return 0.5 + (field.offset - dot(field.normal, sub(point, field.origin))) / field.transitionWidth;
}
export function evaluateRetainedAffineDensity(field: RetainedAffineDensity, point: AffineDensityPoint): number {
  if (point.some(v => !Number.isFinite(v))) throw new Error("Nonfinite density query");
  if (field.kind === "clamped-affine") return Math.min(1, Math.max(0, rawRamp(field, point)));
  const a = evaluateRetainedAffineDensity(field.branches[0], point), b = evaluateRetainedAffineDensity(field.branches[1], point);
  return field.kind === "minimum" ? Math.min(a, b) : Math.max(a, b);
}
function cube(): Polyhedron {
  return [
    [[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]],
    [[1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
    [[0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1]],
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    [[0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]],
  ];
}
const planeAt = (plane: Plane, point: V) => plane[0] + plane[1] * point[0] + plane[2] * point[1] + plane[3] * point[2];
const negate = (plane: Plane): Plane => [-plane[0], -plane[1], -plane[2], -plane[3]];
function unique(points: V[]): V[] {
  const seen = new Set<string>();
  return points.filter(p => { const key = p.join("/"); if (seen.has(key)) return false; seen.add(key); return true; });
}
/** Convex intersection with plane(x)>=0; no small coefficient is discarded. */
function clip(polyhedron: Polyhedron, plane: Plane): Polyhedron {
  const result: Polyhedron = [], cap: V[] = [];
  for (const face of polyhedron) {
    const out: V[] = [];
    for (let i = 0; i < face.length; i++) {
      const a = face[i], b = face[(i + 1) % face.length], da = planeAt(plane, a), db = planeAt(plane, b);
      if (da >= 0) out.push(a);
      if ((da < 0 && db >= 0) || (da >= 0 && db < 0)) {
        const t = da / (da - db);
        const point = a.map((v, axis) => v + t * (b[axis] - v)) as V;
        out.push(point); cap.push(point);
      }
    }
    const polygon = unique(out);
    if (polygon.length >= 3) result.push(polygon);
  }
  const polygon = unique(cap);
  if (polygon.length >= 3) {
    const normal = plane.slice(1).map(Math.abs);
    const drop = normal.indexOf(Math.max(...normal));
    const axes = [0, 1, 2].filter(axis => axis !== drop);
    const center = polygon.reduce((s, p) => s.map((v, axis) => v + p[axis] / polygon.length) as V, [0, 0, 0] as V);
    polygon.sort((a, b) => Math.atan2(a[axes[1]] - center[axes[1]], a[axes[0]] - center[axes[0]])
      - Math.atan2(b[axes[1]] - center[axes[1]], b[axes[0]] - center[axes[0]]));
    result.push(polygon);
  }
  return result;
}
/** Sum positive tetrahedra from an interior point. Affine vertex averaging is
 * exact on each tetrahedron, without subtracting large positive-part moments. */
function moment(polyhedron: Polyhedron, plane: Plane): number {
  const points = polyhedron.flat();
  if (!points.length) return 0;
  const center = points.reduce((s, p) => s.map((v, axis) => v + p[axis] / points.length) as V, [0, 0, 0] as V);
  let result = 0;
  for (const face of polyhedron) for (let i = 1; i + 1 < face.length; i++) {
    const a = face[0], b = face[i], c = face[i + 1];
    const volume = Math.abs(dot(sub(a, center), cross(sub(b, center), sub(c, center)))) / 6;
    result += volume * (planeAt(plane, center) + planeAt(plane, a) + planeAt(plane, b) + planeAt(plane, c)) / 4;
  }
  return result;
}
function normalizedPlane(field: RetainedAffineRamp, box: AffineDensityBox): Plane {
  const widths = boxWidths(box);
  const plane: Plane = [rawRamp(field, box.lower), ...field.normal.map((v, axis) => -v * widths[axis] / field.transitionWidth) as V];
  if (plane.some(v => !Number.isFinite(v))) throw new Error("Ramp/query scale exceeds representable integration range");
  return plane;
}
function clampedMoment(polyhedron: Polyhedron, plane: Plane): number {
  const values = polyhedron.flat().map(point => planeAt(plane, point));
  if (!values.length || Math.max(...values) <= 0) return 0;
  if (Math.min(...values) >= 1) return moment(polyhedron, [1, 0, 0, 0]);
  if (Math.min(...values) >= 0 && Math.max(...values) <= 1) return moment(polyhedron, plane);
  const upper: Plane = [1 - plane[0], -plane[1], -plane[2], -plane[3]];
  const transition = clip(clip(polyhedron, plane), upper);
  const filled = clip(polyhedron, negate(upper));
  return moment(transition, plane) + moment(filled, [1, 0, 0, 0]);
}
export function meanRetainedAffineDensity(field: RetainedAffineDensity, box: AffineDensityBox): number {
  const base = cube();
  let value: number;
  if (field.kind === "clamped-affine") value = clampedMoment(base, normalizedPlane(field, box));
  else {
    const a = normalizedPlane(field.branches[0], box), b = normalizedPlane(field.branches[1], box);
    let selection = a.map((v, i) => v - b[i]) as unknown as Plane;
    if (selection.every(v => v === 0)) return clampedMoment(base, a);
    if (field.kind === "minimum") selection = negate(selection);
    value = clampedMoment(clip(base, selection), a) + clampedMoment(clip(base, negate(selection)), b);
  }
  if (!Number.isFinite(value) || value < -1e-12 || value > 1 + 1e-12) throw new Error("Invalid retained ramp integration receipt");
  return Math.min(1, Math.max(0, value));
}
export function integrateRetainedAffineDensity(field: RetainedAffineDensity, box: AffineDensityBox): number {
  const widths = boxWidths(box), volume = widths[0] * widths[1] * widths[2];
  if (!Number.isFinite(volume) || !(volume > 0)) throw new Error("Density box volume is outside representable range");
  return meanRetainedAffineDensity(field, box) * volume;
}
export function splitRetainedAffineDensity(field: RetainedAffineDensity, box: AffineDensityBox) {
  const widths = boxWidths(box), middle = box.lower.map((v, axis) => v + widths[axis] / 2);
  return Array.from({ length: 8 }, (_, child) => {
    const lower = box.lower.map((v, axis) => child & (1 << axis) ? middle[axis] : v) as unknown as AffineDensityPoint;
    const upper = box.upper.map((v, axis) => child & (1 << axis) ? v : middle[axis]) as unknown as AffineDensityPoint;
    const childBox = Object.freeze({ lower: Object.freeze(lower), upper: Object.freeze(upper) });
    return Object.freeze({ field, box: childBox, mean: meanRetainedAffineDensity(field, childBox) });
  });
}
