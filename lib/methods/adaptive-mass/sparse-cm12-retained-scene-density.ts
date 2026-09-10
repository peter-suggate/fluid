import type { SceneDescription } from "../../core/model";
import { initialFluidSeedBrickCoordinates, sceneDamBreakBox } from "../../core/initial-fluid";
import { sceneLatticeDimensions } from "../../core/scene-lattice-dimensions";

export type RetainedScenePoint = readonly [number, number, number];
export interface RetainedSceneBox { readonly lower: RetainedScenePoint; readonly upper: RetainedScenePoint }
export type RetainedScenePrimitive =
  | { readonly kind: "box"; readonly lower: RetainedScenePoint; readonly upper: RetainedScenePoint }
  | { readonly kind: "ellipsoid"; readonly center: RetainedScenePoint; readonly radii: RetainedScenePoint }
  | { readonly kind: "quadratic-height"; readonly center: RetainedScenePoint; readonly curvature: RetainedScenePoint };

/** An immutable numeric field, independent of native-cell ownership. The
 * density is clamp(.5 - min(primitivePhi)/transitionWidth, 0, 1), restricted
 * to the domain. Its .5 surface is the authored surface; its amount is the
 * diffuse density integral, deliberately not the sharp enclosed volume. */
export interface RetainedSceneDensity {
  readonly generation: number;
  readonly transitionWidth: number;
  readonly domain: RetainedSceneBox;
  readonly primitives: readonly RetainedScenePrimitive[];
  /** The admitted initial support is a union of complete physical cells.
   * Its upper bound is derived from origin + dimensions*f32(cellSize), not
   * independently rounded authored endpoints. Primitive coefficients and
   * authored wall attachment still refer to domain above. */
  readonly supportLattice?: Readonly<{ dimensions: RetainedScenePoint; cellSize: number }>;
}

export const RETAINED_SCENE_HEADER_FLOATS = 16;
export const RETAINED_SCENE_PRIMITIVE_FLOATS = 16;

/** Production's retained fine-coordinate transform is scalar and isotropic.
 * Validate the whole initial lattice before atlas seeding and resident
 * allocation; individual clipped leaves need not span whole domain axes.
 * Rounding or the minimum lattice dimension can produce anisotropic realized
 * scene cells, which require a vector transform throughout the solver. Taking
 * their minimum here would silently change the authored physical geometry.
 * Raw physical-box integration is independent of this adoption precondition. */
export function assertRetainedSceneIsotropicLattice(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number): void {
  const h = Math.fround(cellSize);
  if (!(h > 0) || !Number.isFinite(h)
    || dimensions.some(n => !Number.isSafeInteger(n) || n < 1)) {
    throw new Error("Invalid retained physical lattice");
  }
  if (field.supportLattice && (field.supportLattice.cellSize !== h
    || field.supportLattice.dimensions.some((n, axis) => n !== dimensions[axis]))) {
    throw new Error("Retained support lattice cannot change during physical field adoption");
  }
  for (let axis = 0; axis < 3; axis++) {
    const extent = field.domain.upper[axis] - field.domain.lower[axis];
    const realized = dimensions[axis] * h;
    // Both domain endpoints and GPU h have already been rounded to f32.
    const tolerance = 8 * 2 ** -23 * Math.max(extent, realized, h);
    if (!Number.isFinite(extent) || !Number.isFinite(realized) || !(extent > 0)
      || Math.abs(extent - realized) > tolerance) {
      throw new Error(`Retained density requires an isotropic realized lattice: axis ${axis} has extent ${extent}, ${dimensions[axis]} cells at ${h} cover ${realized}`);
    }
  }
}

const supportBoxes = new WeakMap<RetainedSceneDensity, RetainedSceneBox>();
/** Hard physical support. Authored domain endpoints remain the authority for
 * deciding whether a liquid box face attaches to a domain wall. */
export function retainedSceneSupportBox(field: RetainedSceneDensity): RetainedSceneBox {
  if (!field.supportLattice) return field.domain;
  let box = supportBoxes.get(field);
  if (!box) {
    const lattice = field.supportLattice;
    box = Object.freeze({ lower: field.domain.lower,
      upper: Object.freeze(field.domain.lower.map((lo, axis) => lo + lattice.dimensions[axis] * lattice.cellSize)) as unknown as RetainedScenePoint });
    supportBoxes.set(field, box);
  }
  return box;
}

/** Declare the same support geometry used by native capacity and GPU sample
 * coordinates. This changes no authored liquid primitive or density value;
 * it prevents f32 endpoint roundoff from clipping fictitious boundary slivers. */
export function bindRetainedSceneSupportLattice(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number): RetainedSceneDensity {
  assertRetainedSceneIsotropicLattice(field, dimensions, cellSize);
  if (field.supportLattice) return field;
  return retainedSceneDensity({ ...field, supportLattice: { dimensions, cellSize } });
}

const clamp = (v: number) => Math.max(0, Math.min(1, v));
const finite = (v: number) => {
  const f = Math.fround(v);
  if (!Number.isFinite(f)) throw new Error("Retained scene coefficient must be finite float32");
  return f;
};
const point = (v: RetainedScenePoint): RetainedScenePoint => Object.freeze(v.map(finite)) as unknown as RetainedScenePoint;

export function retainedSceneDensity(input: RetainedSceneDensity): RetainedSceneDensity {
  if (!Number.isSafeInteger(input.generation) || input.generation < 1 || input.generation > 0xff_ffff
    || !(input.transitionWidth > 0)) throw new Error("Invalid retained scene field generation or width");
  const domain = Object.freeze({ lower: point(input.domain.lower), upper: point(input.domain.upper) });
  if (domain.lower.some((v, i) => v >= domain.upper[i])) throw new Error("Invalid retained scene domain");
  const primitives = input.primitives.map((p): RetainedScenePrimitive => {
    if (p.kind === "box") {
      const lower = point(p.lower), upper = point(p.upper);
      if (lower.some((v, i) => v >= upper[i])) throw new Error("Invalid retained liquid box");
      return Object.freeze({ kind: p.kind, lower, upper });
    }
    if (p.kind === "ellipsoid") {
      const radii = point(p.radii);
      if (radii.some(v => !(v > 0))) throw new Error("Invalid retained liquid ellipsoid");
      return Object.freeze({ kind: p.kind, center: point(p.center), radii });
    }
    if (p.kind !== "quadratic-height") throw new Error("Unsupported retained liquid primitive");
    return Object.freeze({ kind: p.kind, center: point(p.center), curvature: point(p.curvature) });
  });
  const transitionWidth = finite(input.transitionWidth);
  if (!(transitionWidth > 0)) throw new Error("Retained transition width underflows float32");
  const supportLattice = input.supportLattice ? Object.freeze({
    dimensions: Object.freeze([...input.supportLattice.dimensions]) as unknown as RetainedScenePoint,
    cellSize: finite(input.supportLattice.cellSize),
  }) : undefined;
  const result = Object.freeze({ generation: input.generation, transitionWidth, domain, primitives: Object.freeze(primitives),
    ...(supportLattice ? { supportLattice } : {}) });
  if (supportLattice) assertRetainedSceneIsotropicLattice(result, supportLattice.dimensions, supportLattice.cellSize);
  return result;
}

/** Compile source authoring once, never infer a surface from cell means.
 * Unsupported authoring returns null so callers cannot silently omit a source.
 * Solids/open-domain coupling remains the caller's responsibility. */
export function compileRetainedSceneDensity(scene: SceneDescription,
  options: { generation?: number; transitionWidth?: number } = {}): RetainedSceneDensity | null {
  if (scene.fluid.initialHeightField?.kind === "cosine"
    || scene.fluid.initialLiquidVolumes?.some(v => v.shape !== "box" && v.shape !== "sphere")) return null;
  const c = scene.container;
  const lower: RetainedScenePoint = [-c.width_m / 2, 0, -c.depth_m / 2];
  const upper: RetainedScenePoint = [c.width_m / 2, c.height_m, c.depth_m / 2];
  const primitives: RetainedScenePrimitive[] = [];
  if (scene.systems?.fluid !== false) {
    if (!scene.fluid.initialBrickSeeds_m || scene.fluid.initialBrickSeedsAdditive) {
      const height = scene.fluid.initialHeightField;
      if (height?.kind === "quadratic") primitives.push({ kind: "quadratic-height",
        center: [height.center_m.x, height.baseHeight_m, height.center_m.z],
        curvature: [height.curvatureX_mInv, 0, height.curvatureZ_mInv] });
      else if (scene.fluid.initialCondition === "tank-fill") {
        if (c.fillFraction > 0) primitives.push({ kind: "quadratic-height", center: [0, c.fillFraction * c.height_m, 0], curvature: [0, 0, 0] });
      } else {
        const box = sceneDamBreakBox(scene);
        if (box.max.x > box.min.x && box.max.y > box.min.y && box.max.z > box.min.z) primitives.push({ kind: "box",
          lower: [lower[0] + box.min.x * c.width_m, box.min.y * c.height_m, lower[2] + box.min.z * c.depth_m],
          upper: [lower[0] + box.max.x * c.width_m, box.max.y * c.height_m, lower[2] + box.max.z * c.depth_m] });
      }
    }
    if (scene.fluid.initialBrickSeeds_m?.length) {
      const dimensions = sceneLatticeDimensions(scene), h = [c.width_m / dimensions[0], c.height_m / dimensions[1], c.depth_m / dimensions[2]];
      for (const brick of initialFluidSeedBrickCoordinates(scene, dimensions, 8)) primitives.push({ kind: "box",
        lower: brick.map((b, axis) => lower[axis] + 8 * b * h[axis]) as unknown as RetainedScenePoint,
        upper: brick.map((b, axis) => Math.min(upper[axis], lower[axis] + 8 * (b + 1) * h[axis])) as unknown as RetainedScenePoint });
    }
    for (const volume of scene.fluid.initialLiquidVolumes ?? []) {
      if (volume.shape === "box") primitives.push({ kind: "box", lower: [volume.min_m.x, volume.min_m.y, volume.min_m.z], upper: [volume.max_m.x, volume.max_m.y, volume.max_m.z] });
      else if (volume.shape === "sphere") primitives.push({ kind: "ellipsoid", center: [volume.center_m.x, volume.center_m.y, volume.center_m.z], radii: [volume.radius_m, volume.radius_m, volume.radius_m] });
    }
  }
  return retainedSceneDensity({ generation: options.generation ?? 1,
    transitionWidth: options.transitionWidth ?? scene.voxelDomain.finestCellSize_m,
    domain: { lower, upper }, primitives });
}

function boxPhi(field: RetainedSceneDensity, p: Extract<RetainedScenePrimitive, { kind: "box" }>, x: RetainedScenePoint) {
  let phi = -1e30;
  for (let axis = 0; axis < 3; axis++) {
    // A tank wall bounds support, not a liquid/air transition inside water.
    if (p.lower[axis] > field.domain.lower[axis]) phi = Math.max(phi, p.lower[axis] - x[axis]);
    if (p.upper[axis] < field.domain.upper[axis]) phi = Math.max(phi, x[axis] - p.upper[axis]);
  }
  return phi;
}
export function evaluateRetainedScenePhi(field: RetainedSceneDensity, x: RetainedScenePoint): number {
  if (x.some(v => !Number.isFinite(v))) throw new Error("Nonfinite retained scene query");
  let phi = 1e30;
  for (const p of field.primitives) {
    let value: number;
    if (p.kind === "box") value = boxPhi(field, p, x);
    else if (p.kind === "ellipsoid") value = .5 * Math.min(...p.radii)
      * (p.radii.reduce((sum, r, axis) => sum + ((x[axis] - p.center[axis]) / r) ** 2, 0) - 1);
    else value = x[1] - p.center[1] - p.curvature[0] * (x[0] - p.center[0]) ** 2 - p.curvature[2] * (x[2] - p.center[2]) ** 2;
    phi = Math.min(phi, value);
  }
  const support = retainedSceneSupportBox(field); let outside = 0;
  for (let axis = 0; axis < 3; axis++) outside = Math.max(outside, support.lower[axis] - x[axis], x[axis] - support.upper[axis]);
  return outside > 0 ? Math.max(outside, phi) : phi;
}
export function evaluateRetainedSceneDensity(field: RetainedSceneDensity, x: RetainedScenePoint): number {
  const support = retainedSceneSupportBox(field);
  if (x.some((v, axis) => v < support.lower[axis] || v > support.upper[axis])) return 0;
  return clamp(.5 - evaluateRetainedScenePhi(field, x) / field.transitionWidth);
}

/** Float-only ABI, so the production resident arena can append it directly.
 * Header: version,count,generation,width; lower at4, upper at8, world origin
 * at12. Primitive: kind at0 (1 box,2 ellipsoid,3 height), first vector at4,
 * second at8. The origin converts solver metres to authored world metres.
 * Optional support dimensions at7/11/15 derive the hard support upper bound
 * with the admitted GPU h; zeros preserve raw authored physical-box queries. */
export function packRetainedSceneDensity(field: RetainedSceneDensity): Float32Array {
  const result = new Float32Array(RETAINED_SCENE_HEADER_FLOATS + RETAINED_SCENE_PRIMITIVE_FLOATS * field.primitives.length);
  result.set([1, field.primitives.length, field.generation, field.transitionWidth]);
  result.set(field.domain.lower, 4); result.set(field.domain.upper, 8); result.set(field.domain.lower, 12);
  if (field.supportLattice) field.supportLattice.dimensions.forEach((n, axis) => { result[7 + 4 * axis] = n; });
  field.primitives.forEach((p, i) => {
    const start = RETAINED_SCENE_HEADER_FLOATS + i * RETAINED_SCENE_PRIMITIVE_FLOATS;
    result[start] = p.kind === "box" ? 1 : p.kind === "ellipsoid" ? 2 : 3;
    result.set(p.kind === "box" ? p.lower : p.center, start + 4);
    result.set(p.kind === "box" ? p.upper : p.kind === "ellipsoid" ? p.radii : p.curvature, start + 8);
  });
  return result;
}

type Polynomial = readonly [number, number, number]; // 1,t,t²; t = y - query midpoint.
interface Segment { readonly low: number; readonly high: number; readonly polynomial: Polynomial }
const value = (p: Polynomial, x: number) => p[0] + x * (p[1] + x * p[2]);
function roots(p: Polynomial, low: number, high: number, result: number[]) {
  const [c, b, a] = p;
  if (a === 0) { if (b !== 0) { const r = -c / b; if (r > low && r < high) result.push(r); } return; }
  const d = b * b - 4 * a * c;
  if (!(d > 0)) return;
  const q = -.5 * (b + (b < 0 ? -1 : 1) * Math.sqrt(d));
  for (const r of [q / a, c / q]) if (r > low && r < high) result.push(r);
}
function intersections(a: Polynomial, b: Polynomial, low: number, high: number, result: number[]) {
  roots([a[0] - b[0], a[1] - b[1], a[2] - b[2]], low, high, result);
}
function sortedBreaks(values: number[]) {
  values.sort((a, b) => a - b);
  return values.filter((v, i) => i === 0 || v > values[i - 1]);
}
function clampedMinimum(polynomials: Polynomial[], low: number, high: number): Segment[] {
  const breaks = [low, high], zero: Polynomial = [0, 0, 0], one: Polynomial = [1, 0, 0];
  for (let i = 0; i < polynomials.length; i++) {
    intersections(polynomials[i], zero, low, high, breaks); intersections(polynomials[i], one, low, high, breaks);
    for (let j = 0; j < i; j++) intersections(polynomials[i], polynomials[j], low, high, breaks);
  }
  const positions = sortedBreaks(breaks), result: Segment[] = [];
  for (let i = 1; i < positions.length; i++) {
    const a = positions[i - 1], b = positions[i], middle = (a + b) / 2;
    let p = polynomials[0];
    for (let j = 1; j < polynomials.length; j++) if (value(polynomials[j], middle) < value(p, middle)) p = polynomials[j];
    const v = value(p, middle); p = v <= 0 ? zero : v >= 1 ? one : p;
    result.push({ low: a, high: b, polynomial: p });
  }
  return result;
}
function primitiveProfile(field: RetainedSceneDensity, p: RetainedScenePrimitive,
  x: number, z: number, middle: number, low: number, high: number): Segment[] {
  const w = field.transitionWidth;
  if (p.kind === "quadratic-height") return clampedMinimum([[
    .5 + (p.center[1] + p.curvature[0] * (x - p.center[0]) ** 2 + p.curvature[2] * (z - p.center[2]) ** 2 - middle) / w,
    -1 / w, 0]], low, high);
  if (p.kind === "ellipsoid") {
    const k = Math.min(...p.radii) / (2 * w), dy = middle - p.center[1], iy2 = 1 / p.radii[1] ** 2;
    return clampedMinimum([[.5 + k * (1 - ((x - p.center[0]) / p.radii[0]) ** 2 - ((z - p.center[2]) / p.radii[2]) ** 2 - dy * dy * iy2),
      -2 * k * dy * iy2, -k * iy2]], low, high);
  }
  const terms: Polynomial[] = [[1, 0, 0]];
  for (const axis of [0, 2]) {
    const coordinate = axis === 0 ? x : z;
    if (p.lower[axis] > field.domain.lower[axis]) terms.push([.5 + (coordinate - p.lower[axis]) / w, 0, 0]);
    if (p.upper[axis] < field.domain.upper[axis]) terms.push([.5 + (p.upper[axis] - coordinate) / w, 0, 0]);
  }
  if (p.lower[1] > field.domain.lower[1]) terms.push([.5 + (middle - p.lower[1]) / w, 1 / w, 0]);
  if (p.upper[1] < field.domain.upper[1]) terms.push([.5 + (p.upper[1] - middle) / w, -1 / w, 0]);
  return clampedMinimum(terms, low, high);
}
function polynomialIntegral(p: Polynomial, low: number, high: number) {
  const middle = (low + high) / 2, width = high - low;
  return width * (value(p, middle) + p[2] * width * width / 12);
}

/** Exact (up to binary64 roundoff) integral of the union's piecewise quadratic
 * vertical density profile. All clamp and branch-switch roots are included. */
export function integrateRetainedSceneVertical(field: RetainedSceneDensity,
  x: number, z: number, lowerY: number, upperY: number): number {
  if (![x, z, lowerY, upperY].every(Number.isFinite) || upperY < lowerY) throw new Error("Invalid retained vertical query");
  const support = retainedSceneSupportBox(field);
  if (x < support.lower[0] || x > support.upper[0] || z < support.lower[2] || z > support.upper[2]) return 0;
  const y0 = Math.max(lowerY, support.lower[1]), y1 = Math.min(upperY, support.upper[1]);
  if (!(y1 > y0) || !field.primitives.length) return 0;
  const middle = (y0 + y1) / 2, low = y0 - middle, high = y1 - middle;
  const profiles = field.primitives.map(p => primitiveProfile(field, p, x, z, middle, low, high));
  if (profiles.length === 1) return Math.max(0, Math.min(y1 - y0,
    profiles[0].reduce((sum, s) => sum + polynomialIntegral(s.polynomial, s.low, s.high), 0)));
  const positions = sortedBreaks(profiles.flatMap(profile => profile.flatMap(s => [s.low, s.high])));
  const indices = profiles.map(() => 0);
  let integral = 0;
  for (let i = 1; i < positions.length; i++) {
    const a = positions[i - 1], b = positions[i], midpoint = (a + b) / 2;
    const polynomials = profiles.map((profile, p) => {
      while (indices[p] + 1 < profile.length && profile[indices[p]].high <= midpoint) indices[p]++;
      return profile[indices[p]].polynomial;
    });
    const cuts = [a, b];
    for (let p = 0; p < polynomials.length; p++) for (let q = 0; q < p; q++) intersections(polynomials[p], polynomials[q], a, b, cuts);
    const sorted = sortedBreaks(cuts);
    for (let j = 1; j < sorted.length; j++) {
      const l = sorted[j - 1], h = sorted[j], m = (l + h) / 2;
      let selected = polynomials[0];
      for (let p = 1; p < polynomials.length; p++) if (value(polynomials[p], m) > value(selected, m)) selected = polynomials[p];
      integral += polynomialIntegral(selected, l, h);
    }
  }
  return Math.max(0, Math.min(y1 - y0, integral));
}

function squaredRange(low: number, high: number, center: number): readonly [number, number] {
  const a = low - center, b = high - center;
  return [a <= 0 && b >= 0 ? 0 : Math.min(a * a, b * b), Math.max(a * a, b * b)];
}
function densityRange(field: RetainedSceneDensity, box: RetainedSceneBox): readonly [number, number] {
  let low = 0, high = 0;
  for (const p of field.primitives) {
    let phiLow: number, phiHigh: number;
    if (p.kind === "box") {
      phiLow = -1e30; phiHigh = -1e30;
      for (let axis = 0; axis < 3; axis++) {
        if (p.lower[axis] > field.domain.lower[axis]) { phiLow = Math.max(phiLow, p.lower[axis] - box.upper[axis]); phiHigh = Math.max(phiHigh, p.lower[axis] - box.lower[axis]); }
        if (p.upper[axis] < field.domain.upper[axis]) { phiLow = Math.max(phiLow, box.lower[axis] - p.upper[axis]); phiHigh = Math.max(phiHigh, box.upper[axis] - p.upper[axis]); }
      }
    } else if (p.kind === "ellipsoid") {
      let lo = -1, hi = -1;
      for (let axis = 0; axis < 3; axis++) { const range = squaredRange(box.lower[axis], box.upper[axis], p.center[axis]); lo += range[0] / p.radii[axis] ** 2; hi += range[1] / p.radii[axis] ** 2; }
      const scale = Math.min(...p.radii) / 2; phiLow = scale * lo; phiHigh = scale * hi;
    } else {
      let hLow = p.center[1], hHigh = p.center[1];
      for (const axis of [0, 2]) {
        const range = squaredRange(box.lower[axis], box.upper[axis], p.center[axis]);
        hLow += p.curvature[axis] * range[p.curvature[axis] >= 0 ? 0 : 1];
        hHigh += p.curvature[axis] * range[p.curvature[axis] >= 0 ? 1 : 0];
      }
      phiLow = box.lower[1] - hHigh; phiHigh = box.upper[1] - hLow;
    }
    low = Math.max(low, clamp(.5 - phiHigh / field.transitionWidth));
    high = Math.max(high, clamp(.5 - phiLow / field.transitionWidth));
  }
  return [low, high];
}

/** Algebraic density bounds for a physical box, including hard domain support.
 * These permit exact constant-region fast paths; they are not outward-rounded
 * certified interval arithmetic for arbitrary binary64 inputs. */
export function retainedSceneDensityRange(field: RetainedSceneDensity, query: RetainedSceneBox): readonly [number, number] {
  if ([...query.lower, ...query.upper].some(v => !Number.isFinite(v))
    || query.lower.some((v, axis) => v >= query.upper[axis])) throw new Error("Invalid retained density range box");
  const support = retainedSceneSupportBox(field);
  const lower = query.lower.map((v, axis) => Math.max(v, support.lower[axis])) as unknown as RetainedScenePoint;
  const upper = query.upper.map((v, axis) => Math.min(v, support.upper[axis])) as unknown as RetainedScenePoint;
  if (lower.some((v, axis) => v >= upper[axis])) return [0, 0];
  const [low, high] = densityRange(field, { lower, upper });
  return [query.lower.some((v, axis) => v < support.lower[axis])
    || query.upper.some((v, axis) => v > support.upper[axis]) ? 0 : low, high];
}

const GAUSS5 = [[-.906179845938664, .2369268850561891], [-.5384693101056831, .4786286704993665], [0, .5688888888888889], [.5384693101056831, .4786286704993665], [.906179845938664, .2369268850561891]] as const;
const GAUSS9 = [[-.9681602395076261, .08127438836157441], [-.8360311073266358, .1806481606948574], [-.6133714327005904, .2606106964029354], [-.3242534234038089, .3123470770400029], [0, .3302393550012598], [.3242534234038089, .3123470770400029], [.6133714327005904, .2606106964029354], [.8360311073266358, .1806481606948574], [.9681602395076261, .08127438836157441]] as const;
export interface RetainedSceneIntegralReceipt {
  readonly amount: number;
  readonly mean: number;
  /** Safety multiplier on two-rule disagreement, not an interval certificate. */
  readonly estimatedAbsoluteError: number;
  readonly requestedAbsoluteTolerance: number;
  readonly toleranceMet: boolean;
  readonly rectangles: number;
  readonly verticalEvaluations: number;
}
interface Rectangle { readonly box: RetainedSceneBox; readonly amount: number; readonly error: number }

/** Initialization/topology integral, with an exact y profile and adaptive
 * tensor Gauss 5/9 in x,z. Full/dry boxes are exact; compact primitive bounds
 * seed the partition so quadrature cannot skip an isolated body. The receipt
 * reports numerical convergence, not a certified interval enclosure. */
export function integrateRetainedSceneDensity(field: RetainedSceneDensity, query: RetainedSceneBox,
  options: { absoluteTolerance?: number; maximumRectangles?: number } = {}): RetainedSceneIntegralReceipt {
  const widths = query.upper.map((v, axis) => v - query.lower[axis]);
  if ([...query.lower, ...query.upper].some(v => !Number.isFinite(v)) || widths.some(v => !(v > 0))) throw new Error("Invalid retained scene integration box");
  const volume = widths[0] * widths[1] * widths[2];
  const tolerance = options.absoluteTolerance ?? Math.max(1e-15, volume * 2e-7), maximum = options.maximumRectangles ?? 8192;
  if (!(tolerance > 0) || !Number.isFinite(tolerance) || !Number.isSafeInteger(maximum) || maximum < 1) throw new Error("Invalid retained integration budget");
  const support = retainedSceneSupportBox(field);
  const lower = query.lower.map((v, axis) => Math.max(v, support.lower[axis])) as unknown as RetainedScenePoint;
  const upper = query.upper.map((v, axis) => Math.min(v, support.upper[axis])) as unknown as RetainedScenePoint;
  let evaluations = 0;
  const receipt = (amount: number, error: number, rectangles: number): RetainedSceneIntegralReceipt => Object.freeze({ amount, mean: amount / volume,
    estimatedAbsoluteError: error, requestedAbsoluteTolerance: tolerance, toleranceMet: error <= tolerance, rectangles, verticalEvaluations: evaluations });
  if (lower.some((v, axis) => v >= upper[axis]) || !field.primitives.length) return receipt(0, 0, 0);
  const clipped = { lower, upper };
  const wholeRange = densityRange(field, clipped);
  const clippedVolume = (upper[0] - lower[0]) * (upper[1] - lower[1]) * (upper[2] - lower[2]);
  if (wholeRange[1] === 0 || wholeRange[0] === 1) return receipt(clippedVolume * wholeRange[0], 0, 1);
  // This is query compilation only: retain the original immutable field while
  // excluding provably dry branches from the integration work for this box.
  field = { ...field, primitives: field.primitives.filter(p => densityRange({ ...field, primitives: [p] }, clipped)[1] > 0) };
  if (field.primitives.length === 1) {
    const p = field.primitives[0];
    if (p.kind === "quadratic-height" && p.curvature[0] === 0 && p.curvature[2] === 0) {
      evaluations++;
      return receipt(integrateRetainedSceneVertical(field, lower[0], lower[2], lower[1], upper[1])
        * (upper[0] - lower[0]) * (upper[2] - lower[2]), 0, 1);
    }
    if (p.kind === "ellipsoid") {
      const ratio = field.transitionWidth / Math.min(...p.radii), outer = Math.sqrt(1 + ratio);
      let fraction = 1;
      for (let axis = 0; axis < 3; axis++) {
        const lo = p.center[axis] - outer * p.radii[axis], hi = p.center[axis] + outer * p.radii[axis];
        if (lower[axis] <= lo && upper[axis] >= hi) continue;
        if ((lower[axis] === p.center[axis] && upper[axis] >= hi) || (upper[axis] === p.center[axis] && lower[axis] <= lo)) fraction *= .5;
        else { fraction = 0; break; }
      }
      if (fraction > 0) {
        const upperLog = 2.5 * Math.log1p(ratio), lowerLog = ratio < 1 ? 2.5 * Math.log1p(-ratio) : -Infinity;
        const difference = ratio < 1 ? Math.exp(lowerLog) * Math.expm1(upperLog - lowerLog) : Math.exp(upperLog);
        return receipt(fraction * 4 * Math.PI * p.radii[0] * p.radii[1] * p.radii[2] * difference / (15 * ratio), 0, 1);
      }
    }
  }
  const make = (box: RetainedSceneBox): Rectangle => {
    const volume = (box.upper[0] - box.lower[0]) * (box.upper[1] - box.lower[1]) * (box.upper[2] - box.lower[2]);
    const range = densityRange(field, box);
    if (range[1] === 0 || range[0] === 1) return { box, amount: volume * range[0], error: 0 };
    const mx = (box.lower[0] + box.upper[0]) / 2, mz = (box.lower[2] + box.upper[2]) / 2;
    const hx = (box.upper[0] - box.lower[0]) / 2, hz = (box.upper[2] - box.lower[2]) / 2;
    const quadrature = (rule: typeof GAUSS5 | typeof GAUSS9) => {
      let amount = 0;
      for (const [x, wx] of rule) for (const [z, wz] of rule) {
        amount += wx * wz * integrateRetainedSceneVertical(field, mx + hx * x, mz + hz * z, box.lower[1], box.upper[1]); evaluations++;
      }
      return amount * hx * hz;
    };
    const coarse = quadrature(GAUSS5), fine = quadrature(GAUSS9);
    let error = 8 * Math.abs(fine - coarse) + volume * 2e-14;
    if (fine === 0 && coarse === 0) error = volume * range[1];
    else if (Math.abs(fine - volume) < volume * 1e-14 && Math.abs(coarse - volume) < volume * 1e-14) error = volume * (1 - range[0]);
    return { box, amount: Math.max(0, Math.min(volume, fine)), error };
  };
  const cuts = ([0, 2] as const).map(axis => {
    const positions = [lower[axis], upper[axis]];
    for (const p of field.primitives) {
      let events: number[];
      if (p.kind === "ellipsoid") {
        const nearestY2 = squaredRange(lower[1], upper[1], p.center[1])[0] / p.radii[1] ** 2;
        const extent = p.radii[axis] * Math.sqrt(Math.max(0, 1 + field.transitionWidth / Math.min(...p.radii) - nearestY2));
        events = [p.center[axis] - extent, p.center[axis], p.center[axis] + extent];
      } else if (p.kind === "box") events = [p.lower[axis] - field.transitionWidth / 2, p.lower[axis], p.lower[axis] + field.transitionWidth / 2,
        p.upper[axis] - field.transitionWidth / 2, p.upper[axis], p.upper[axis] + field.transitionWidth / 2];
      else events = [p.center[axis]];
      for (const event of events) if (event > lower[axis] && event < upper[axis]) positions.push(event);
    }
    return sortedBreaks(positions);
  });
  const nodes: Rectangle[] = [];
  if ((cuts[0].length - 1) * (cuts[1].length - 1) > maximum) {
    const only = make(clipped);
    // A budget too small even for compact-support partitioning cannot accept
    // agreement between rules that might both have missed the same component.
    return receipt(only.amount, Math.max(only.error, clippedVolume), 1);
  }
  for (let x = 1; x < cuts[0].length; x++) for (let z = 1; z < cuts[1].length; z++) nodes.push(make({
    lower: [cuts[0][x - 1], lower[1], cuts[1][z - 1]], upper: [cuts[0][x], upper[1], cuts[1][z]] }));
  let amount = nodes.reduce((sum, node) => sum + node.amount, 0), error = nodes.reduce((sum, node) => sum + node.error, 0);
  while (error > tolerance && nodes.length + 3 <= maximum) {
    let worst = 0;
    for (let i = 1; i < nodes.length; i++) if (nodes[i].error > nodes[worst].error) worst = i;
    const parent = nodes[worst], b = parent.box, mx = (b.lower[0] + b.upper[0]) / 2, mz = (b.lower[2] + b.upper[2]) / 2;
    if (!(mx > b.lower[0] && mx < b.upper[0] && mz > b.lower[2] && mz < b.upper[2])) break;
    const children: Rectangle[] = [];
    for (let z = 0; z < 2; z++) for (let x = 0; x < 2; x++) children.push(make({
      lower: [x ? mx : b.lower[0], b.lower[1], z ? mz : b.lower[2]], upper: [x ? b.upper[0] : mx, b.upper[1], z ? b.upper[2] : mz] }));
    amount += children.reduce((sum, n) => sum + n.amount, 0) - parent.amount;
    error = Math.max(0, error + children.reduce((sum, n) => sum + n.error, 0) - parent.error);
    nodes[worst] = children[0]; nodes.push(...children.slice(1));
  }
  return receipt(Math.max(0, Math.min(volume, amount)), error, nodes.length);
}

export interface RetainedSceneFineMeansReceipt {
  readonly cells: number;
  readonly integratedCells: number;
  readonly constantCells: number;
  readonly reflectedCells: number;
  readonly verticalEvaluations: number;
  readonly estimatedAbsoluteError: number;
  readonly maximumEstimatedMeanError: number;
}

/** Compile physical finest-cell moments once. Interval full/dry blocks avoid
 * marching the empty domain; numeric reflection symmetries avoid duplicate
 * boundary integrals. Native partitions subsequently restrict these moments.
 * An unresolved quadrature receipt is an error, never an accepted seed mean.
 * The caller owns the returned snapshot and must retain its field generation. */
export function compileRetainedSceneFineMeans(field: RetainedSceneDensity,
  dimensions: readonly [number, number, number], cellSize: number,
  options: { absoluteMeanTolerance?: number; maximumRectangles?: number;
    onReceipt?: (receipt: RetainedSceneFineMeansReceipt) => void } = {}): Float32Array {
  const h = finite(cellSize), meanTolerance = options.absoluteMeanTolerance ?? 2e-7;
  if (!(h > 0) || !(meanTolerance > 0) || !Number.isFinite(meanTolerance)
    || dimensions.some(v => !Number.isSafeInteger(v) || v < 1)) throw new Error("Invalid retained finest-cell lattice");
  const count = dimensions[0] * dimensions[1] * dimensions[2];
  if (!Number.isSafeInteger(count) || count > 0x1000_0000) throw new Error("Retained finest-cell lattice exceeds storage budget");
  if (field.supportLattice) assertRetainedSceneIsotropicLattice(field, dimensions, h);
  const support = retainedSceneSupportBox(field);
  const means = new Float32Array(count), origin = support.lower;
  const reflected = dimensions.map((n, axis) => {
    if (Math.abs(n * h - (support.upper[axis] - origin[axis])) > n * h * 1e-12) return false;
    const center = (origin[axis] + support.upper[axis]) / 2;
    return field.primitives.every(p => {
      if (p.kind === "ellipsoid") return p.center[axis] === center;
      if (p.kind === "box") return p.lower[axis] + p.upper[axis] === 2 * center;
      return axis !== 1 && (p.curvature[axis] === 0 || p.center[axis] === center);
    });
  });
  const work = dimensions.map((n, axis) => reflected[axis] ? Math.ceil(n / 2) : n);
  const index = (x: number, y: number, z: number) => x + dimensions[0] * (y + dimensions[1] * z);
  let integratedCells = 0, constantCells = 0, verticalEvaluations = 0, error = 0, maxMeanError = 0;
  const visit = (lo: readonly number[], hi: readonly number[]) => {
    const box: RetainedSceneBox = { lower: lo.map((v, axis) => origin[axis] + h * v) as unknown as RetainedScenePoint,
      upper: hi.map((v, axis) => origin[axis] + h * v) as unknown as RetainedScenePoint };
    const outside = box.lower.some((v, axis) => v >= support.upper[axis])
      || box.upper.some((v, axis) => v <= support.lower[axis]);
    const inside = box.lower.every((v, axis) => v >= support.lower[axis])
      && box.upper.every((v, axis) => v <= support.upper[axis]);
    const range = outside ? [0, 0] : densityRange(field, box);
    if (range[1] === 0 || (inside && range[0] === 1)) {
      const value = range[1] === 0 ? 0 : 1;
      for (let z = lo[2]; z < hi[2]; z++) for (let y = lo[1]; y < hi[1]; y++) means.fill(value, index(lo[0], y, z), index(hi[0] - 1, y, z) + 1);
      constantCells += (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]); return;
    }
    if (hi.every((v, axis) => v - lo[axis] === 1)) {
      const integral = integrateRetainedSceneDensity(field, box, { absoluteTolerance: h ** 3 * meanTolerance,
        maximumRectangles: options.maximumRectangles });
      if (!integral.toleranceMet) throw new Error(`Retained scene integral did not converge at ${lo.join(",")}: estimated mean error ${integral.estimatedAbsoluteError / h ** 3}`);
      means[index(lo[0], lo[1], lo[2])] = integral.mean;
      integratedCells++; verticalEvaluations += integral.verticalEvaluations;
      error += integral.estimatedAbsoluteError; maxMeanError = Math.max(maxMeanError, integral.estimatedAbsoluteError / h ** 3); return;
    }
    const sizes = hi.map((v, axis) => v - lo[axis]);
    // Prefer Y on ties: horizontal fills then terminate in whole x/z slabs.
    let axis = 1;
    for (const candidate of [0, 2]) if (sizes[candidate] > sizes[axis]) axis = candidate;
    const middle = Math.floor((lo[axis] + hi[axis]) / 2), leftHi = [...hi], rightLo = [...lo];
    leftHi[axis] = middle; rightLo[axis] = middle;
    visit(lo, leftHi); visit(rightLo, hi);
  };
  visit([0, 0, 0], work);
  let reflectedCells = 0;
  for (let z = 0; z < dimensions[2]; z++) for (let y = 0; y < dimensions[1]; y++) for (let x = 0; x < dimensions[0]; x++) {
    const coordinates = [x, y, z].map((v, axis) => reflected[axis] ? Math.min(v, dimensions[axis] - 1 - v) : v);
    if (coordinates[0] !== x || coordinates[1] !== y || coordinates[2] !== z) {
      means[index(x, y, z)] = means[index(coordinates[0], coordinates[1], coordinates[2])]; reflectedCells++;
    }
  }
  const multiplier = reflected.reduce((value, reflect) => value * (reflect ? 2 : 1), 1);
  options.onReceipt?.(Object.freeze({ cells: count, integratedCells, constantCells, reflectedCells, verticalEvaluations,
    estimatedAbsoluteError: error * multiplier, maximumEstimatedMeanError: maxMeanError }));
  return means;
}
