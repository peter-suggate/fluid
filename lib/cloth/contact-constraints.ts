import {closestSegmentPoints, closestTrianglePoint, connectedRestFaces, restSegmentTriangleDistance, sampleClothCollider, type ClothCollider, type ClothContactReport, type ClothReaction} from './contact';
import {sampleClippedRodCollider} from './clipped-contact';
import {ClothContactBVH} from './contact-bvh';
import {convexClothDistance, supportClippedRod} from './rod-convex';
import {clothRodClippedCapsule, clothRodLineRoots, clothRodPlaneContains, clothRodSegmentClipNormals, type ClothRodCapsuleGeometry} from './rod-geometry';
import {cross3, dot3, norm3, point3, scale3, sub3, transportDirector, type PreparedClothTopology} from './topology';
import type {ClothVector} from './types';
import {sceneShape} from '../core/scene-shape';

/** Contact inside the implicit minimization.
 *
 * Every contact found by the reference's continuous detection (the same
 * conservative-advancement CCD, thickness and exclusions as
 * resolveClothContacts) becomes a one-sided constraint g_k(x) >= 0 on the
 * end-of-step positions, enforced by an augmented Lagrangian term
 *   psi_k = rho_k/2 max(0, lambda_k/rho_k - g_k)^2
 * added to the incremental potential, with lambda_k <- max(0, lambda_k - rho_k g_k)
 * after each inner solve. Every constraint is linear in the positions,
 *   g_k = n_k . sum_j w_kj x_j - b_k,
 * with n and w first taken where the CCD found the impact (self contact: vertex
 * minus face point; edge-edge: the two closest points; colliders: the tangent
 * plane), then re-linearized at the current end-of-step positions every outer
 * iteration (sequential linearization of the true distance minus thickness >= 0):
 * colliders take the tangent plane there, self pairs the closest points there,
 * the normal kept on the side the pair was detected from.
 * Coulomb friction is the lagged smoothed form of Li et al. 2020 (IPC): with the
 * normal force lambda and tangent plane of the previous outer iteration,
 *   mu lambda f0(|P_t (sum_j w_kj (x_j - x_j^old) - d_obstacle)|),
 * f0 C1-smoothed below eps = FRICTION_SLIP_SPEED dt.
 * Outer iterations re-detect from the old to the current end positions and stop
 * when no new pair appears and, in metres, every KKT residual
 * |min(g, lambda/rho)| (penetration, or force acting across a gap), every true
 * penetration (collider or self) and every lagged-friction residual
 * mu |lambda - lambda_friction|/rho is below the solver tolerance. (The inner
 * solve fixes g only to that tolerance, so lambda is known to rho times it.) Velocity is then (x - x_old)/dt of these
 * positions, so nothing is projected after the solve.
 *
 * Round, uncut rods are capsules here: each rod vertex is a sphere of the rod
 * radius (the c: collider pairs and rv:/rp:/rq: self pairs), and a segment
 * interior meets colliders (rc:), faces (rf:) and other segments or shell
 * edges (rs:, rx:) where its closest point is away from both ends. The gap
 * stays linear in the positions (the centreline point, by its weights); the
 * friction slip is that of the rod's material point, rodContactDOF's
 * work-conjugate Jacobian: the endpoint couple and the segment's material
 * twist roll it, so friction acts on the twist too (RodContactGeometry).
 * Same-rod pairs that overlap in the rest shape are one continuous yarn and
 * never contact.
 * An elliptical section (RodEllipse) is a rounded convex core: its gap is the
 * cores' GJK distance minus the rounding radii, its vertices give way to the
 * segments' end caps (ec:, ev:, eq:, ep:, keyed 2s + end), its gap row is the
 * material point's Jacobian transposed onto the normal (the envelope theorem),
 * so a contact off the centreline also loads the twist.
 * A cut rod end (rodStartClipNormals, rodEndClipNormals) is the exact clipped
 * volume of rendering and picking (clothRodClippedCapsule): its end vertex
 * has no sphere or cap; a cut cap (the clipped volume near that end) takes its
 * pairs under the cap's keys, and the segment's interior pairs see the whole
 * clipped volume. A clipped volume has no rounding, so a penetrating
 * configuration is measured by the separating axis (rodFeatureDistance).
 * Against a sampled solid an elliptical or cut feature is its analytic volume
 * (rendering's clipped capsule; an uncut cap the capsule beyond its vertex)
 * under the field's Lipschitz bound (sampleClippedRodCollider), but that
 * bound does not certify near-tangent gaps yet, so elliptical sections and
 * cut ends against sampled or rigid solids, and dynamic rigid colliders
 * (whose velocity responds within the step) are not modelled here: such
 * steps keep the sequential post-projection, and inSolveContactUnsupported()
 * says why. */
export interface ClothContactConstraint {
  readonly key: string;
  readonly kind: 'collider' | 'self';
  readonly vertices: readonly number[];
  weights: number[];
  normal: [number, number, number];
  offset: number;
  readonly colliderIndex: number;
  /** Collider surface displacement over the step at the contact. */
  obstacle: [number, number, number];
  readonly friction: number;
  /** Collision radius of the cloth vertex (collider contacts). */
  readonly radius: number;
  lambda: number;
  frictionLambda: number;
  rho: number;
  /** Hessian-model coefficients at the last gradient evaluation. */
  normalCoefficient: number;
  frictionCoefficient: number;
  /** Rod contacts: shape and rolling Jacobian (undefined for shell pairs). */
  readonly rod?: RodContactGeometry;
}

/** A rod side of a contact: the segment whose material point touches. Its
 * rotation (endpoint couple, material twist) moves that point tangentially.
 * sign is +1 for the pair's first feature, -1 for its second. */
interface RodContactSide {
  readonly segment: number; readonly a: number; readonly b: number; readonly sign: 1 | -1; readonly radius: number;
  /** The material point's offset from the centreline at the last
   * linearization: -sign radius n (round), the ellipse's support (elliptical). */
  arm: ClothVector;
  tau: ClothVector;
  /** The gap's twist row, sign n.tau (zero for a round section). */
  twistRow: number;
}
export type RodContactShape = 'point-collider' | 'segment-collider' | 'point-face' | 'segment-face' | 'point-segment' | 'segment-segment';
/** A feature of an elliptical rod pair, rebuilt at each configuration: a
 * segment end's cap (the ellipsoid of that segment at the vertex), a segment's
 * swept body, a ball at a vertex, a shell face or edge of the given radius;
 * or a cut end's cap (the clipped volume near that end vertex). */
export type RodFeatureSpec = {readonly kind: 'cap' | 'cut'; readonly segment: number; readonly vertex: number} | {readonly kind: 'segment'; readonly segment: number}
  | {readonly kind: 'point'; readonly vertex: number; readonly radius: number} | {readonly kind: 'face'; readonly vertices: readonly number[]; readonly radius: number}
  | {readonly kind: 'edge'; readonly vertices: readonly number[]; readonly radius: number};
export interface RodContactGeometry {
  readonly shape: RodContactShape;
  /** c.vertices holds the first feature's `first` vertices, then the second
   * feature's `second`, then rolling endpoints of weight zero. */
  readonly first: number;
  readonly second: number;
  readonly sides: readonly RodContactSide[];
  /** Row-major 3x3 friction-slip Jacobian per entry of c.vertices: w_j I plus
   * each rod side's endpoint couple; the twist column of a side is its tau. */
  readonly jacobian: Float64Array;
  /** The gap's rows J_j^T n per entry of c.vertices (w_j n for round sections):
   * g = sum_j rows_j . x_j + sum_sides twistRow theta - offset. */
  readonly rows: Float64Array;
  /** A round self pair's thickness (the sum of its radii). */
  readonly thickness: number;
  /** Elliptical pairs: the two features whose GJK distance is the gap (the
   * second undefined against a collider). */
  readonly features?: readonly [RodFeatureSpec, RodFeatureSpec | undefined];
}
const ROD_INTERIOR = 1e-6;
const closestPointSegment = (p: ClothVector, a: ClothVector, b: ClothVector) => {const e = sub3(b, a), ee = dot3(e, e), t = ee > 1e-30 ? Math.max(0, Math.min(1, dot3(sub3(p, a), e)/ee)) : 0; return {t, point: lerp3(a, b, t)};};
/** Closest points of a segment and a triangle: segment parameter s and the
 * triangle's barycentric weights (a crossing segment returns distance zero). */
function closestSegmentTriangle(a: ClothVector, b: ClothVector, x: ClothVector, y: ClothVector, z: ClothVector): {s: number; weights: ClothVector; pa: ClothVector; pb: ClothVector} {
  const normal = cross3(sub3(y, x), sub3(z, x)), da = dot3(sub3(a, x), normal), db = dot3(sub3(b, x), normal);
  if (da*db <= 0 && Math.abs(da - db) > 1e-30) {const s = da/(da - db), p = lerp3(a, b, s), q = closestTrianglePoint(p, x, y, z); if (norm3(sub3(p, q.point)) < 1e-12) return {s, weights: q.weights, pa: p, pb: q.point};}
  let best = {s: 0, weights: [1, 0, 0] as ClothVector, pa: a, pb: x}, distance = Infinity;
  const consider = (s: number, weights: ClothVector, pa: ClothVector, pb: ClothVector) => {const d = norm3(sub3(pa, pb)); if (d < distance) {distance = d; best = {s, weights, pa, pb};}};
  for (const [s, p] of [[0, a], [1, b]] as const) {const q = closestTrianglePoint(p, x, y, z); consider(s, q.weights, p, q.point);}
  const corners = [x, y, z];
  for (let i = 0; i < 3; i++) {const q = closestSegmentPoints(a, b, corners[i]!, corners[(i + 1) % 3]!), w: [number, number, number] = [0, 0, 0]; w[i] = 1 - q.t; w[(i + 1) % 3] = q.t; consider(q.s, w, q.a, q.b);}
  return best;
}
const perpendicular = (e: ClothVector): ClothVector => {const f = cross3(e, Math.abs(e[0]) < .8*norm3(e) ? [1, 0, 0] : [0, 1, 0]), l = norm3(f); return l > 0 ? scale3(f, 1/l) : [0, 1, 0];};
/** Closest features of a self pair of the given shape (points: the first
 * feature's, then the second's): the pair's weights, the separation from the
 * second to the first, a normal for a zero separation, and whether the
 * closest points lie where this shape (not a vertex or edge pair) owns them. */
export function rodSelfClosest(shape: RodContactShape, q: readonly ClothVector[]): {weights: number[]; delta: ClothVector; fallback: ClothVector; owned: boolean} {
  if (shape === 'point-face') {const c = closestTrianglePoint(q[0]!, q[1]!, q[2]!, q[3]!); return {weights: [1, -c.weights[0], -c.weights[1], -c.weights[2]], delta: sub3(q[0]!, c.point), fallback: cross3(sub3(q[2]!, q[1]!), sub3(q[3]!, q[1]!)), owned: true};}
  if (shape === 'segment-face') {const c = closestSegmentTriangle(q[0]!, q[1]!, q[2]!, q[3]!, q[4]!); return {weights: [1 - c.s, c.s, -c.weights[0], -c.weights[1], -c.weights[2]], delta: sub3(c.pa, c.pb), fallback: cross3(sub3(q[3]!, q[2]!), sub3(q[4]!, q[2]!)), owned: c.s > ROD_INTERIOR && c.s < 1 - ROD_INTERIOR && c.weights.every(w => w > ROD_INTERIOR)};}
  if (shape === 'point-segment') {const c = closestPointSegment(q[0]!, q[1]!, q[2]!); return {weights: [1, -(1 - c.t), -c.t], delta: sub3(q[0]!, c.point), fallback: perpendicular(sub3(q[2]!, q[1]!)), owned: true};}
  if (shape === 'segment-segment') {const c = closestSegmentPoints(q[0]!, q[1]!, q[2]!, q[3]!); return {weights: [1 - c.s, c.s, -(1 - c.t), -c.t], delta: sub3(c.a, c.b), fallback: cross3(sub3(q[1]!, q[0]!), sub3(q[3]!, q[2]!)), owned: [c.s, c.t].every(v => v > ROD_INTERIOR && v < 1 - ROD_INTERIOR)};}
  throw new Error(`Rod self contact shape ${shape} is not a self pair`);
}
const unit = (delta: ClothVector, fallback: ClothVector): ClothVector => {const l = norm3(delta), f = norm3(fallback); return l > 1e-12 ? scale3(delta, 1/l) : f > 1e-12 ? scale3(fallback, 1/f) : [0, 1, 0];};

/** An elliptical rod segment's cross-section at a configuration
 * (clothRodEllipsoidAxes): material directors d1, d2 of radii A and B, the
 * tangent t of the caps' axial radius sqrt(AB). rho is the rounding radius,
 * half the ellipsoid's smallest curvature radius c^2/a: E = E' + ball(rho)
 * with E' smooth and strictly convex, so the signed distance of two rounded
 * bodies is the GJK distance of their cores minus the radii, also through a
 * penetration shallower than the radii (the round rod's centreline distance
 * minus thickness is the case E' = a point). */
export interface RodEllipse {readonly d1: ClothVector; readonly d2: ClothVector; readonly t: ClothVector; readonly radii: ClothVector; readonly rho: number}
/** The support offset of the full ellipsoid in direction d (clothRodEllipsoidSupportOffset). */
export function rodEllipseSupport(e: RodEllipse, d: ClothVector): ClothVector {
  const q0 = dot3(e.d1, d)*e.radii[0], q1 = dot3(e.d2, d)*e.radii[1], q2 = dot3(e.t, d)*e.radii[2], l = Math.hypot(q0, q1, q2);
  if (!(l > 0)) return [0, 0, 0];
  const a = e.radii[0]*q0/l, b = e.radii[1]*q1/l, c = e.radii[2]*q2/l;
  return [a*e.d1[0] + b*e.d2[0] + c*e.t[0], a*e.d1[1] + b*e.d2[1] + c*e.t[1], a*e.d1[2] + b*e.d2[2] + c*e.t[2]];
}
/** The ellipse frame of a segment from its end positions, its start-of-step
 * reference frame (director, tangent: the solver's parallel transport) and its
 * material twist; undefined for a round section. */
export function rodEllipseFrame(radiusA: number, radiusB: number, a: ClothVector, b: ClothVector, director: ClothVector, tangent: ClothVector, twist: number): RodEllipse {
  const e = sub3(b, a), length = norm3(e);
  if (!(length > 0)) throw new Error('Cloth elliptical rod segment has zero length in contact');
  const t = scale3(e, 1/length), d = transportDirector(director, tangent, t), q = cross3(t, d), c = Math.cos(twist), s = Math.sin(twist);
  const d1: ClothVector = [c*d[0] + s*q[0], c*d[1] + s*q[1], c*d[2] + s*q[2]], radii: ClothVector = [radiusA, radiusB, Math.sqrt(radiusA*radiusB)];
  const sorted = [...radii].sort((x, y) => y - x);
  return {d1, d2: cross3(t, d1), t, radii, rho: .5*sorted[2]!*sorted[2]!/sorted[0]!};
}
/** A convex contact feature: the hull of its core points (a vertex, a segment,
 * a triangle) plus an ellipse frame or a ball of the given radius, or a
 * clipped rod volume (clip: its whole surface, the core points its anchors). */
export interface RodFeature {readonly core: readonly ClothVector[]; readonly ellipse?: RodEllipse; readonly radius: number; readonly clip?: ClothRodCapsuleGeometry;
  /** An elliptical feature's analytic volume for sampled fields: its segment's capsule, a cap's beyond its vertex. */
  readonly shape?: ClothRodCapsuleGeometry}
const featureRho = (f: RodFeature) => f.clip ? 0 : f.ellipse ? f.ellipse.rho : f.radius;
/** The offset from a feature's core point to its surface point of outward normal n
 * (a clipped volume's core point is its surface point). */
const featureSurface = (f: RodFeature, n: ClothVector): ClothVector => f.clip ? [0, 0, 0] : f.ellipse ? rodEllipseSupport(f.ellipse, n) : scale3(n, f.radius);
/** The feature's full support (its surface point farthest along d). */
function featureSupport(f: RodFeature, d: ClothVector): ClothVector {
  const c = featureCoreSupport(f, d), rho = featureRho(f), l = norm3(d);
  return rho > 0 && l > 0 ? [c[0] + rho*d[0]/l, c[1] + rho*d[1]/l, c[2] + rho*d[2]/l] : c;
}
/** Support of the feature's core plus its strictly convex part (E', or nothing for a ball). */
function featureCoreSupport(f: RodFeature, d: ClothVector): ClothVector {
  if (f.clip) return supportClippedRod(f.clip, d);
  let best = f.core[0]!, value = dot3(best, d);
  for (let k = 1; k < f.core.length; k++) {const v = dot3(f.core[k]!, d); if (v > value) {value = v; best = f.core[k]!;}}
  if (!f.ellipse) return best;
  const s = rodEllipseSupport(f.ellipse, d), l = norm3(d), r = f.ellipse.rho/l;
  return [best[0] + s[0] - r*d[0], best[1] + s[1] - r*d[1], best[2] + s[2] - r*d[2]];
}
/** The closest points of two features: the signed distance (negative through
 * a penetration shallower than the rounding radii), the normal from b to a,
 * and the core points (on the segment or triangle) the material points hang
 * from. Cores within GJK tolerance of each other are a penetration deeper than
 * the rounding of rounded features: an explicit error. */
export function rodFeatureDistance(a: RodFeature, b: RodFeature, initial: ClothVector): {gap: number; normal: ClothVector; ca: ClothVector; cb: ClothVector} {
  const face = cutFaceDistance(a, b);
  if (face && face.gap >= 0) return face;
  const general = convexFeatureDistance(a, b, face?.normal ?? initial);
  return face && !(general.gap > face.gap) ? face : general;
}
/** A cut face's plane against the other feature: the cut volume lies behind
 * it, so the signed distance is at least the other feature's height over the
 * plane at its lowest point, and equals it when apart with that point over
 * the face. Exact, and unlike GJK's closest pair or the separating axis's
 * support on two parallel flat faces (any point of their overlap), the same
 * support point as a cut end on a plane. The largest such bound, or none. */
function cutFaceDistance(a: RodFeature, b: RodFeature): {gap: number; normal: ClothVector; ca: ClothVector; cb: ClothVector} | undefined {
  let best: {gap: number; normal: ClothVector; ca: ClothVector; cb: ClothVector} | undefined;
  for (const [own, other, sign] of [[a, b, -1], [b, a, 1]] as const) for (const plane of own.clip?.planes ?? []) {
    // m points out of own's face; the normal from b to a is -m when own is a, m when own is b.
    const m = plane.normal, p = featureSupport(other, scale3(m, -1)), gap = dot3(m, p) - plane.offset_m;
    if (!Number.isFinite(gap) || (best && best.gap >= gap)) continue;
    const q = sub3(p, scale3(m, gap)), roots = clothRodLineRoots(own.clip!, q, m);
    if (!clothRodPlaneContains(own.clip!.planes, q, 1e-12) || !(Math.min(...roots) <= 0 && Math.max(...roots) >= 0)) continue;
    const otherCore = sub3(p, featureSurface(other, scale3(m, -1)));
    best = sign < 0 ? {gap, normal: scale3(m, -1), ca: q, cb: otherCore} : {gap, normal: m, ca: otherCore, cb: q};
  }
  return best;
}
function convexFeatureDistance(a: RodFeature, b: RodFeature, initial: ClothVector): {gap: number; normal: ClothVector; ca: ClothVector; cb: ClothVector} {
  const result = convexClothDistance(d => featureCoreSupport(a, d), d => featureCoreSupport(b, d), norm3(initial) > 0 ? initial : [1, 0, 0], 1e-11);
  // Where GJK's simplex degenerates on flat faces (a box's face, a cut face:
  // a support repeats short of tolerance) the distance is the separating
  // axis's, and so is a sharp clipped volume's through any overlap.
  if (!result.converged || ((a.clip || b.clip) && !(result.distance_m > 1e-11))) return separatingAxis(a, b, result.normal);
  if (!(result.distance_m > 1e-11)) throw new Error('Cloth elliptical rod contact penetrates deeper than its rounding radius');
  const n = result.normal, ca = sub3(result.a, a.ellipse ? sub3(rodEllipseSupport(a.ellipse, scale3(n, -1)), scale3(n, -a.ellipse.rho)) : [0, 0, 0]);
  const cb = sub3(result.b, b.ellipse ? sub3(rodEllipseSupport(b.ellipse, n), scale3(n, b.ellipse.rho)) : [0, 0, 0]);
  return {gap: result.distance_m - featureRho(a) - featureRho(b), normal: n, ca, cb};
}
/** The signed distance of two features by the separating axis: the largest
 * separation s(n) = n.(S_a(-n) - S_b(n)) over unit n (n from b to a), the
 * distance when apart and minus the penetration depth when overlapping,
 * exact for sharp volumes (a clipped rod) at any depth. Local ascent on the
 * sphere along the separation's gradient (the supports' difference, projected)
 * from the given normal, the cores' offset and the axes; by the envelope
 * theorem the gap's derivative is n.(da - db) at the support points. */
function separatingAxis(a: RodFeature, b: RodFeature, initial: ClothVector): {gap: number; normal: ClothVector; ca: ClothVector; cb: ClothVector} {
  const at = (direction: ClothVector) => {
    const n = scale3(direction, 1/norm3(direction)), pa = featureSupport(a, scale3(n, -1)), pb = featureSupport(b, n);
    return {n, pa, pb, s: dot3(n, sub3(pa, pb))};
  };
  const mean = (f: RodFeature) => scale3(f.core.reduce((sum, p) => add3(sum, p), [0, 0, 0] as ClothVector), 1/f.core.length);
  const starts: ClothVector[] = [initial, sub3(mean(a), mean(b)), [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]];
  let best = starts.filter(d => norm3(d) > 0).map(at).reduce((x, y) => y.s > x.s ? y : x);
  // The angular step's scale: the larger feature's extent.
  const extent = (f: RodFeature) => Math.max(...[0, 1, 2].map(k => {const e: ClothVector = [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0]; return dot3(e, sub3(featureSupport(f, e), featureSupport(f, scale3(e, -1))));}));
  const size = Math.max(extent(a), extent(b), 1e-12);
  for (let iteration = 0; iteration < 400; iteration++) {
    const d = sub3(best.pa, best.pb), g = sub3(d, scale3(best.n, dot3(best.n, d))), length = norm3(g);
    if (!(length > 1e-14)) break;
    let step = Math.min(.5, length/size), improved = false;
    for (let k = 0; k < 60 && step > 1e-15; k++, step *= .5) {
      const next = at(add3(best.n, scale3(g, step/length)));
      if (next.s > best.s) {best = next; improved = true; break;}
    }
    if (!improved) break;
  }
  return {gap: best.s, normal: best.n, ca: sub3(best.pa, featureSurface(a, scale3(best.n, -1))), cb: sub3(best.pb, featureSurface(b, best.n))};
}
/** The material point offset of a feature facing direction -sign n (the
 * first feature faces the second along -n): the ellipse's support, or the
 * ball's radius along the normal. */
const featureArm = (f: RodFeature, n: ClothVector, sign: 1 | -1): ClothVector => f.ellipse ? scale3(rodEllipseSupport(f.ellipse, n), -sign) : scale3(n, -sign*f.radius);

/** Slip speed below which lagged Coulomb friction is smoothed (IPC eps_v). */
export const FRICTION_SLIP_SPEED_M_S = 1e-3;
/** Outer contact iterations per step before the step reports nonconvergence. */
export const MAXIMUM_CONTACT_ITERATIONS = 32;
/** Contact multipliers of a state's last step, keyed by constraint, so the CPU
 * and GPU solves warm-start the same augmented Lagrangian. */
const warmStarts = new WeakMap<object, ReadonlyMap<string, number>>();
export const clothContactWarmStart = (state: object): ReadonlyMap<string, number> | undefined => warmStarts.get(state);
export function rememberClothContactMultipliers(state: object, set: ClothContactSet): void {warmStarts.set(state, set.multipliers());}
/** rho_k = PENALTY_RATIO times the constraint's effective diagonal stiffness. */
const PENALTY_RATIO = 10;

export function inSolveContactUnsupported(topology: PreparedClothTopology, colliders: readonly ClothCollider[]): string | undefined {
  const solid = colliders.some(c => c.kind === 'sampled' || c.kind === 'rigid');
  for (let r = 0; r < topology.rods.length; r++) {
    const m = topology.materials[topology.rods[r]!.material]!;
    // An elliptical section or a cut end meets planes, spheres, boxes, faces
    // and rods by the convex distance; a sampled field's bound on its analytic
    // volume does not converge near touch, and a rigid solid has none yet.
    if ((m.rodRadiusB_m ?? m.rodRadius_m) !== m.rodRadius_m && solid) return 'elliptical rods against sampled or rigid solids keep the sequential projection';
    if (clothRodCut(topology, r) && solid) return 'clipped rod end caps against sampled or rigid solids keep the sequential projection';
  }
  for (const c of colliders) if (c.kind === 'rigid' && !c.body.held && c.body.description.motion !== 'static') return 'a dynamic rigid collider responds within the step; it keeps the sequential projection';
  return undefined;
}

/** Whether rod r has a cut end (a clipping plane on its first or last segment). */
export function clothRodCut(topology: PreparedClothTopology, r: number): boolean {
  return [topology.data.rodStartClipNormals, topology.data.rodEndClipNormals].some(normals => normals !== undefined && norm3(point3(normals, r)) > 0);
}
function colliderVelocity(collider: ClothCollider, point: ClothVector): ClothVector {
  if (collider.kind === 'sampled') return collider.velocityAt(point);
  if (collider.kind !== 'rigid') return collider.velocity_m_s ?? [0, 0, 0];
  const b = collider.body, omega: ClothVector = [b.angularVelocity_rad_s.x, b.angularVelocity_rad_s.y, b.angularVelocity_rad_s.z], r = sub3(point, [b.position_m.x, b.position_m.y, b.position_m.z]), w = cross3(omega, r);
  return [b.linearVelocity_m_s.x + w[0], b.linearVelocity_m_s.y + w[1], b.linearVelocity_m_s.z + w[2]];
}
const colliderDistance = (c: ClothCollider, p: ClothVector, beforeEnd_s: number) => c.kind === 'sampled' && c.distanceAt ? c.distanceAt(p, beforeEnd_s) : sampleClothCollider(c, p, beforeEnd_s).distance_m;
const lerp3 = (a: ClothVector, b: ClothVector, t: number): [number, number, number] => [a[0] + t*(b[0] - a[0]), a[1] + t*(b[1] - a[1]), a[2] + t*(b[2] - a[2])];
const add3 = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
/** The most a collider's distance can change over the step for a point that
 * travels the given length (the CCD's step bound). */
function sweepBound(c: ClothCollider, travel: number, dt: number): number {
  const rotation = c.kind === 'rigid' ? Math.hypot(c.body.angularVelocity_rad_s.x, c.body.angularVelocity_rad_s.y, c.body.angularVelocity_rad_s.z)*sceneShape(c.body.description.shape).boundingRadius_m(c.body.description.dimensions_m) : 0;
  return (travel + dt*(c.kind === 'sampled' ? c.maximumSpeed_m_s : norm3(c.velocity_m_s ?? [0, 0, 0]) + rotation))*(c.kind === 'sampled' ? (c.spatialLipschitz ?? 2) : 1);
}
/** Conservative advancement of a step's sweep: whether the distance from the
 * solid reaches the tolerance. Each sample gives the distance and a lower
 * bound to advance on; an unresolved march counts as entering. */
function sweepEnters(at: (time: number) => {readonly distance: number; readonly bound: number}, speed: number, tolerance: number): boolean {
  for (let iteration = 0, time = 0; iteration < 256; iteration++) {
    const {distance, bound} = at(time);
    if (distance <= tolerance) return true;
    if (time >= 1 || speed === 0) return false;
    if (!(bound > 0)) return true;
    time = Math.min(1, time + Math.max(1e-9, .9*bound/speed));
  }
  return true;
}
/** The crossing test: whether a point's path through the step, start to end
 * with the collider swept alongside, enters the solid (distance <= 0 at the
 * tolerance). Unlike the nearest surface's normal, it does not turn over at a
 * cavity's medial plane, where the nearest wall becomes the opposite one. */
export function pathEntersCollider(c: ClothCollider, start: ClothVector, end: ClothVector, dt: number, tolerance: number): boolean {
  return sweepEnters(time => {const distance = colliderDistance(c, lerp3(start, end, time), (1 - time)*dt); return {distance, bound: distance};}, sweepBound(c, norm3(sub3(end, start)), dt), tolerance);
}
/** The same test for a rod segment's centreline, start [a, b] to end [a, b]. */
function segmentEntersCollider(c: ClothCollider, oa: ClothVector, ob: ClothVector, na: ClothVector, nb: ClothVector, dt: number, tolerance: number): boolean {
  const reach = Math.max(norm3(sub3(na, oa)), norm3(sub3(nb, ob)));
  return sweepEnters(time => {const q = sampleClippedRodCollider({a: lerp3(oa, na, time), b: lerp3(ob, nb, time), radius_m: 0, planes: []}, c, (1 - time)*dt, sampleClothCollider); return {distance: q.distance_m, bound: q.lowerBound_m};}, sweepBound(c, reach, dt), tolerance);
}

export interface ShellEdge {readonly vertices: readonly [number, number]; readonly r: number; readonly min: number[]; readonly max: number[]; restLocalFaces?: Set<number>}
interface ShellCache {faceBounds: Float64Array; faceTree?: ClothContactBVH; edges: ShellEdge[]; edgeBounds: Float64Array; edgeTree?: ClothContactBVH; vertexLocalFaces: Map<number, Set<number>>}
const shellCaches = new WeakMap<PreparedClothTopology, ShellCache>();

/** Collision radius of a cloth vertex: half the shell thickness, or the rod radius. */
export function clothVertexContactRadius(topology: PreparedClothTopology, i: number): number {
  const m = topology.materials[topology.vertexMaterial[i]!]!;
  return topology.incidentTriangles[i]!.length ? .5*m.thickness_m : Math.max(m.rodRadius_m, m.rodRadiusB_m ?? m.rodRadius_m);
}
const vertexRadius = clothVertexContactRadius;
/** The shell's edges and per-vertex rest exclusion patches, in the order and
 * orientation the self-contact keys and weights use. */
export function clothShellCache(topology: PreparedClothTopology): ShellCache {
  let cache = shellCaches.get(topology);
  if (!cache) {
    const edgeMap = new Map<string, readonly [number, number]>();
    for (const f of topology.triangles) for (let j = 0; j < 3; j++) {const a = f.vertices[j]!, b = f.vertices[(j + 1) % 3]!; edgeMap.set(`${Math.min(a, b)}:${Math.max(a, b)}`, [a, b]);}
    const edges = [...edgeMap.values()].map(vertices => ({vertices, r: Math.max(...vertices.map(v => vertexRadius(topology, v))), min: [0, 0, 0], max: [0, 0, 0]}));
    cache = {faceBounds: new Float64Array(6*topology.triangles.length), edges, edgeBounds: new Float64Array(6*edges.length), vertexLocalFaces: new Map()};
    shellCaches.set(topology, cache);
  }
  return cache;
}
/** Faces a vertex never contacts: its connected rest patch within its support. */
export function clothVertexLocalFaces(topology: PreparedClothTopology, i: number): Set<number> {
  const cache = clothShellCache(topology);
  let local = cache.vertexLocalFaces.get(i);
  if (!local) {const p = point3(topology.data.restPositions_m, i); local = connectedRestFaces(topology, topology.incidentTriangles[i]!, vertexRadius(topology, i), t => norm3(sub3(p, closestTrianglePoint(p, t[0]!, t[1]!, t[2]!).point))); cache.vertexLocalFaces.set(i, local);}
  return local;
}
/** resolveClothContacts' edge-edge rest exclusion (ea is the lower index). */
export function clothLocalShellEdges(topology: PreparedClothTopology, ea: ShellEdge, eb: ShellEdge): boolean {
  if (ea.vertices.some(v => eb.vertices.some(w => topology.incidentTriangles[w]!.some(f => clothVertexLocalFaces(topology, v).has(f))))) return true;
  const rest = topology.data.restPositions_m, ra = ea.vertices.map(v => point3(rest, v)), rb = eb.vertices.map(v => point3(rest, v)), nearest = closestSegmentPoints(ra[0]!, ra[1]!, rb[0]!, rb[1]!);
  if (norm3(sub3(nearest.a, nearest.b)) > ea.r + eb.r + 1e-12) return false;
  ea.restLocalFaces ??= connectedRestFaces(topology, ea.vertices.flatMap(v => topology.incidentTriangles[v]!), ea.r, t => restSegmentTriangleDistance(ra[0]!, ra[1]!, t));
  return eb.vertices.some(v => topology.incidentTriangles[v]!.some(f => ea.restLocalFaces!.has(f)));
}

export class ClothContactSet {
  readonly constraints: ClothContactConstraint[] = [];
  private readonly byKey = new Map<string, ClothContactConstraint>();
  private readonly free: Uint8Array;
  /** Friction slip regularization (m per step). */
  readonly epsilon: number;
  /** Rod segments by global segment index, each rod vertex's owning segment
   * (the one it starts, or the rod's last), rod neighbours, free twist. */
  private readonly rodSegments: {readonly rod: number; readonly a: number; readonly b: number; readonly r: number; readonly ellipse?: readonly [number, number];
    /** Cut ends' material-frame plane normals; the radii; the farthest the
     * clipped volume reaches from the centreline's ends (r uncut). */
    readonly clip?: {readonly start?: ClothVector; readonly end?: ClothVector}; readonly radii: readonly [number, number]; readonly reach: number}[] = [];
  /** Rod end vertices of cut ends: no sphere or cap there. */
  private readonly cutEnds = new Set<number>();
  private readonly owner: Int32Array;
  private readonly rodNeighbours = new Map<number, number[]>();
  private readonly twistFree: Uint8Array;
  private rodBounds = new Float64Array(0);
  private rodTree?: ClothContactBVH;
  /** delta: the penalty's smoothing width (the solver tolerance, metres).
   * warmStart: multipliers of the previous step's pairs, by key.
   * rodTwist: the live and start-of-step material twist (rod friction rolls it),
   * and the start-of-step reference frames (an elliptical section's directors). */
  constructor(private readonly topology: PreparedClothTopology, private readonly old: Float64Array, private readonly dt: number, private readonly colliders: readonly ClothCollider[], private readonly selfCollision: boolean, readonly delta: number, private readonly warmStart?: ReadonlyMap<string, number>, private readonly rodTwist?: {readonly current: ArrayLike<number>; readonly old: ArrayLike<number>; readonly directors?: ArrayLike<number>; readonly tangents?: ArrayLike<number>}) {
    const unsupported = inSolveContactUnsupported(topology, colliders);
    if (unsupported) throw new Error(`In-solve cloth contact: ${unsupported}`);
    this.free = Uint8Array.from({length: topology.vertexCount}, (_, i) => (topology.kinematicFixed[i] ?? 0) & 1 ? 0 : 1);
    this.epsilon = FRICTION_SLIP_SPEED_M_S*dt;
    this.owner = new Int32Array(topology.vertexCount).fill(-1);
    const segments = topology.rods.reduce((total, rod) => total + rod.restLengths_m.length, 0);
    this.twistFree = new Uint8Array(segments);
    for (let r = 0; r < topology.rods.length; r++) {
      const rod = topology.rods[r]!, count = rod.restLengths_m.length, m = topology.materials[rod.material]!, radiusB = m.rodRadiusB_m ?? m.rodRadius_m;
      const ellipse = radiusB !== m.rodRadius_m ? [m.rodRadius_m, radiusB] as const : undefined;
      if ((ellipse || clothRodCut(topology, r)) && !(rodTwist?.directors && rodTwist.tangents)) throw new Error('In-solve elliptical or cut rod contact needs the rod reference frames');
      for (let j = 0; j < count; j++) {
        const a = rod.vertices[j]!, b = rod.vertices[j + 1]!, segment = rod.segmentOffset + j, cut = clothRodSegmentClipNormals(topology.data, r, j), radius = Math.max(m.rodRadius_m, radiusB);
        const clip = cut.start || cut.end ? cut : undefined;
        if (cut.start) this.cutEnds.add(a);
        if (cut.end) this.cutEnds.add(b);
        // A cut rim lies within radius/|axial| of its end vertex (axial: the plane normal's tangent part).
        const reach = Math.max(radius, ...[cut.start, cut.end].filter(n => n !== undefined).map(n => radius*norm3(n!)/Math.abs(n![2])));
        this.rodSegments[segment] = {rod: r, a, b, r: radius, ellipse, clip, radii: [m.rodRadius_m, radiusB], reach};
        this.twistFree[segment] = ((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2 ? 0 : 1;
        for (const [v, w] of [[a, b], [b, a]] as const) {const list = this.rodNeighbours.get(v) ?? []; if (!list.includes(w)) list.push(w); this.rodNeighbours.set(v, list);}
      }
      for (let k = 0; k < rod.vertices.length; k++) if (this.owner[rod.vertices[k]!]! < 0) this.owner[rod.vertices[k]!] = rod.segmentOffset + Math.min(k, count - 1);
    }
    if (this.rodSegments.length !== segments || this.rodSegments.some(g => !g)) throw new Error('Cloth rod segments must cover their offsets contiguously');
  }
  /** A vertex that is a rod sphere (on a rod, not on the shell). */
  private rodVertex(i: number): boolean {return this.owner[i]! >= 0 && this.topology.incidentTriangles[i]!.length === 0;}
  private side(segment: number, sign: 1 | -1): RodContactSide {const g = this.rodSegments[segment]!; return {segment, a: g.a, b: g.b, sign, radius: g.r, arm: [0, 0, 0], tau: [0, 0, 0], twistRow: 0};}
  /** Whether a segment's section is elliptical. */
  private elliptical(segment: number): boolean {return this.rodSegments[segment]!.ellipse !== undefined;}
  /** Whether a segment's pairs are convex features (an elliptical section or a cut end). */
  private featured(segment: number): boolean {const g = this.rodSegments[segment]!; return g.ellipse !== undefined || g.clip !== undefined;}
  /** Segment s's ellipse frame with its ends at the given positions and twist. */
  private ellipseAt(segment: number, x: ArrayLike<number>, twist: number): RodEllipse {
    const g = this.rodSegments[segment]!, frames = this.rodTwist!;
    return rodEllipseFrame(g.ellipse![0], g.ellipse![1], point3(x, g.a), point3(x, g.b), point3(frames.directors!, segment), point3(frames.tangents!, segment), twist);
  }
  /** The live twist of a segment, and its value a fraction `time` from the step's start. */
  private twistAt(segment: number, time = 1): number {const t = this.rodTwist!; return t.old[segment]! + time*(t.current[segment]! - t.old[segment]!);}
  /** Adds a rod pair: its rolling endpoints join the vertices at weight zero,
   * and its gap rows and slip Jacobian are linearized at x, the gap keeping its
   * value there (c.offset and the weights give it, as for a shell pair). */
  private addRod(c: Omit<ClothContactConstraint, 'lambda' | 'frictionLambda' | 'rho' | 'normalCoefficient' | 'frictionCoefficient' | 'rod'>, shape: RodContactShape, first: number, second: number, sides: RodContactSide[], x: ArrayLike<number>, features?: readonly [RodFeatureSpec, RodFeatureSpec | undefined]): boolean {
    const vertices = [...c.vertices], weights = [...c.weights];
    for (const side of sides) for (const v of [side.a, side.b]) if (!vertices.includes(v)) {vertices.push(v); weights.push(0);}
    let gap = -c.offset;
    for (let j = 0; j < vertices.length; j++) gap += weights[j]!*dot3(c.normal, point3(x, vertices[j]!));
    // Each pair owns its sides: their arms and twist rows follow its own normal
    // (a detection loop passes one side list to every face near a vertex).
    if (!this.add({...c, vertices, weights, rod: {shape, first, second, sides: sides.map(side => ({...side, arm: [...side.arm], tau: [...side.tau]})), jacobian: new Float64Array(9*vertices.length), rows: new Float64Array(3*vertices.length), thickness: c.kind === 'self' ? c.offset : 0, features}})) return false;
    this.linearizeRolling(this.byKey.get(c.key)!, x, gap);
    return true;
  }
  /** J_j = w_j I + sum over rod sides of sign (+C at b, -C at a), where
   * C v = ((t x v) x arm)/L: the endpoint couple of rodContactDOF, with the
   * material point at arm from the centreline (-sign radius n on a round
   * section, the ellipse's support facing the other feature on an elliptical
   * one); the twist moves it by tau = t x arm per radian. The gap's rows are
   * J_j^T n and its twist rows sign n.tau: the material point's motion along
   * n (by the envelope theorem, the distance's own derivative). On a round
   * section n.C = 0 = n.tau, so the rows are w_j n. The offset keeps the gap
   * at x equal to `gap`. */
  private linearizeRolling(c: ClothContactConstraint, x: ArrayLike<number>, gap: number): void {
    const g = c.rod!, J = g.jacobian, n = c.normal;
    J.fill(0);
    for (let j = 0; j < c.vertices.length; j++) J[9*j] = J[9*j + 4] = J[9*j + 8] = c.weights[j]!;
    let shaped = false;
    for (const side of g.sides) {
      const e = sub3(point3(x, side.b), point3(x, side.a)), length = norm3(e);
      if (!(length > 0)) throw new Error(`Cloth rod segment ${side.segment} has zero length in contact`);
      const t = scale3(e, 1/length), ia = c.vertices.indexOf(side.a), ib = c.vertices.indexOf(side.b);
      // The side's feature (the first faces -n, the second +n): a cut segment's
      // material point is its clipped volume's support, hung from the centreline
      // point of the side's weights; an elliptical one the ellipse's support.
      const spec = g.features?.[side.sign > 0 ? 0 : 1], clipped = (spec?.kind === 'segment' || spec?.kind === 'cut') && this.rodSegments[side.segment]!.clip !== undefined;
      let arm: ClothVector;
      if (clipped) {
        // A clipped volume's material point: its support facing the other feature.
        const f = this.feature(spec!, v => point3(x, v), 1), surface = f.clip ? supportClippedRod(f.clip, scale3(n, -side.sign)) : f.core[0]!, anchor: [number, number, number] = [0, 0, 0];
        for (const [j, v] of [[ia, side.a], [ib, side.b]] as const) for (let q = 0; q < 3; q++) anchor[q]! += side.sign*c.weights[j]!*x[3*v + q]!;
        arm = sub3(surface, anchor);
      } else arm = this.elliptical(side.segment) && spec?.kind !== 'point' ? scale3(rodEllipseSupport(this.ellipseAt(side.segment, x, this.twistAt(side.segment)), n), -side.sign) : scale3(n, -side.sign*side.radius);
      const ta = dot3(t, arm), round = !clipped && !(this.elliptical(side.segment) && spec?.kind !== 'point');
      shaped ||= !round;
      side.arm = arm; side.tau = cross3(t, arm);
      side.twistRow = round ? 0 : side.sign*dot3(n, side.tau);
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {const C = side.sign*((p === q ? ta : 0) - t[p]!*arm[q]!)/length; J[9*ib + 3*p + q]! += C; J[9*ia + 3*p + q]! -= C;}
    }
    let offset = -gap;
    for (let j = 0; j < c.vertices.length; j++) {
      for (let q = 0; q < 3; q++) {
        // A round pair keeps w_j n exactly (n.C vanishes but for rounding).
        g.rows[3*j + q] = shaped ? n[0]*J[9*j + q]! + n[1]*J[9*j + 3 + q]! + n[2]*J[9*j + 6 + q]! : c.weights[j]!*n[q]!;
        offset += g.rows[3*j + q]!*x[3*c.vertices[j]! + q]!;
      }
    }
    for (const side of g.sides) if (side.twistRow) offset += side.twistRow*this.twistAt(side.segment);
    c.offset = offset;
  }

  private add(c: Omit<ClothContactConstraint, 'lambda' | 'frictionLambda' | 'rho' | 'normalCoefficient' | 'frictionCoefficient' | 'rod'> & {rod?: RodContactGeometry}): boolean {
    if (this.byKey.has(c.key) || !c.vertices.some((v, j) => this.free[v] && (!c.rod || c.weights[j] !== 0))) return false;
    const lambda = this.warmStart?.get(c.key) ?? 0;
    const constraint: ClothContactConstraint = {...c, lambda, frictionLambda: lambda, rho: 0, normalCoefficient: 0, frictionCoefficient: 0};
    this.byKey.set(c.key, constraint); this.constraints.push(constraint);
    return true;
  }

  /** A feature with its vertices at `at` and its twist a fraction `time` from the step's start. */
  private feature(spec: RodFeatureSpec, at: (v: number) => ClothVector, time: number): RodFeature {
    if (spec.kind === 'point' || spec.kind === 'face' || spec.kind === 'edge') return {core: spec.kind === 'point' ? [at(spec.vertex)] : spec.vertices.map(at), radius: spec.radius};
    const g = this.rodSegments[spec.segment]!, a = at(g.a), b = at(g.b);
    // An uncut end's cap of a cut segment is its whole ellipsoid (clippedFeature checks the cut misses it).
    if (g.clip && spec.kind !== 'cap') return this.clippedFeature(spec, a, b, time, at);
    if (spec.kind === 'cut') throw new Error(`Cloth rod segment ${spec.segment} has no cut end`);
    const ellipse = g.ellipse ? rodEllipseFrame(g.ellipse[0], g.ellipse[1], a, b, point3(this.rodTwist!.directors!, spec.segment), point3(this.rodTwist!.tangents!, spec.segment), this.twistAt(spec.segment, time)) : undefined;
    // The capsule rendering draws, and for a cap its part beyond the vertex (the tube is the segment's):
    // together they partition the segment's volume.
    const shape = ellipse ? {a, b, radius_m: g.ellipse![0], radiusB_m: g.ellipse![1], director1: ellipse.d1, director2: ellipse.d2,
      planes: spec.kind === 'cap' ? [spec.vertex === g.a ? {normal: ellipse.t, offset_m: dot3(ellipse.t, a)} : {normal: scale3(ellipse.t, -1), offset_m: -dot3(ellipse.t, b)}] : []} : undefined;
    return {core: spec.kind === 'cap' ? [at(spec.vertex)] : [a, b], ellipse, radius: g.r, shape};
  }
  /** A cut segment's feature: its whole clipped volume ('segment'), or a cut
   * cap: that volume from the rim's farthest reach back along the centreline
   * (radius |sin|/|cos| of the plane's tilt from the tangent) to the cut end,
   * which holds the cut face, its rim and the tube beside it. The other end's
   * sphere or ellipsoid cap stays whole only where the cut does not reach it
   * (else an explicit error), so every feature is material of the volume. */
  private clippedFeature(spec: RodFeatureSpec & {segment: number}, a: ClothVector, b: ClothVector, time: number, at: (v: number) => ClothVector): RodFeature {
    const g = this.rodSegments[spec.segment]!, cut = g.clip!, frames = this.rodTwist!;
    const frame = rodEllipseFrame(g.radii[0], g.radii[1], a, b, point3(frames.directors!, spec.segment), point3(frames.tangents!, spec.segment), this.twistAt(spec.segment, time));
    const full = clothRodClippedCapsule({a, b, t: frame.t, d1: frame.d1, d2: frame.d2}, g.radii[0], g.radii[1], cut.start, cut.end), length = norm3(sub3(b, a));
    for (let k = 0; k < full.planes.length; k++) {
      // The plane of one cut end against the other end's sphere or cap.
      const plane = full.planes[k]!, startPlane = k === 0 && cut.start !== undefined, local = startPlane ? cut.start! : cut.end!, vertex = startPlane ? g.b : g.a, other = startPlane ? b : a;
      const rod = this.topology.rods[g.rod]!, shared = vertex !== rod.vertices[0] && vertex !== rod.vertices[rod.vertices.length - 1];
      // A round rod's shared vertex sphere is its neighbour's cap too (rendering's union).
      if ((startPlane ? cut.end : cut.start) || (shared && !g.ellipse)) continue;
      if (dot3(plane.normal, add3(other, rodEllipseSupport(frame, plane.normal))) > plane.offset_m) throw new Error(`Cloth rod segment ${spec.segment}'s cut reaches its uncut end's cap (local normal ${local.join(', ')})`);
    }
    if (spec.kind === 'segment') return {core: [a, b], radius: g.r, clip: full};
    if (spec.kind !== 'cut' || !('vertex' in spec)) throw new Error(`Cloth rod segment ${spec.segment}'s cut end has no ${spec.kind} feature`);
    const start = spec.vertex === g.a, local = start ? cut.start : cut.end;
    if (!local) throw new Error(`Cloth rod vertex ${spec.vertex} is not a cut end of segment ${spec.segment}`);
    const axial = Math.abs(local[2])/norm3(local), back = Math.min(length, g.r*Math.sqrt(Math.max(0, 1 - axial*axial))/axial);
    return {core: [at(spec.vertex)], radius: g.r, clip: start ? {...full, b: add3(a, scale3(frame.t, back))} : {...full, a: sub3(b, scale3(frame.t, back))}};
  }
  /** The centreline a feature's crossing test follows: its core, but a cut
   * end's vertex lies on its face, so there the point min(r, L/2) inside along
   * the centreline (inside the volume for any cut pointing out of its segment). */
  private insidePath(spec: RodFeatureSpec, at: (v: number) => ClothVector): [ClothVector, ClothVector] {
    if (spec.kind === 'point') return [at(spec.vertex), at(spec.vertex)];
    if (spec.kind === 'face' || spec.kind === 'edge') throw new Error('Cloth shell features have no rod centreline');
    const g = this.rodSegments[spec.segment]!, a = at(g.a), b = at(g.b), inward = Math.min(.5, g.r/norm3(sub3(b, a)));
    const start = g.clip?.start ? lerp3(a, b, inward) : a, end = g.clip?.end ? lerp3(b, a, inward) : b;
    if (spec.kind === 'segment') return [start, end];
    const point = spec.vertex === g.a ? start : end;
    return [point, point];
  }
  /** A feature's weights for a point of its core, and whether the point lies
   * where this feature owns it (a segment's or face's interior). */
  private featureWeights(spec: RodFeatureSpec, f: RodFeature, c: ClothVector): {weights: number[]; interior: boolean} {
    if (f.core.length === 1) return {weights: [1], interior: true};
    if (f.core.length === 2) {const t = closestPointSegment(c, f.core[0]!, f.core[1]!).t; return {weights: [1 - t, t], interior: t > ROD_INTERIOR && t < 1 - ROD_INTERIOR};}
    const w = closestTrianglePoint(c, f.core[0]!, f.core[1]!, f.core[2]!).weights;
    return {weights: [w[0], w[1], w[2]], interior: w.every(v => v > ROD_INTERIOR)};
  }
  /** The vertices a feature moves with (its core, and for a cap its segment's other end). */
  private featureReach(spec: RodFeatureSpec): readonly number[] {
    if (spec.kind === 'point') return [spec.vertex];
    if (spec.kind === 'face' || spec.kind === 'edge') return spec.vertices;
    const g = this.rodSegments[spec.segment]!; return [g.a, g.b];
  }
  /** The most a feature's surface moves over the step: its vertices' travel,
   * and for an ellipse its largest radius times a bound on the frame's turn
   * (twice the tangent's turn, the transport, plus the twist's). */
  private featureTravel(spec: RodFeatureSpec, x: ArrayLike<number>): number {
    let travel = 0;
    for (const v of this.featureReach(spec)) travel = Math.max(travel, norm3(sub3(point3(x, v), point3(this.old, v))));
    if (spec.kind !== 'cap' && spec.kind !== 'cut' && spec.kind !== 'segment') return travel;
    const g = this.rodSegments[spec.segment]!;
    if (!g.ellipse && !g.clip) return travel;
    const t0 = sub3(point3(this.old, g.b), point3(this.old, g.a)), t1 = sub3(point3(x, g.b), point3(x, g.a)), turn = Math.atan2(norm3(cross3(t0, t1)), dot3(t0, t1));
    return travel + g.reach*(2*turn + Math.abs(this.twistAt(spec.segment) - this.twistAt(spec.segment, 0)));
  }
  /** Closest points of an elliptical pair at `at` and twist `time`: the
   * weights (first feature's, then the second's negated), the normal from the
   * second to the first, the signed gap, whether both features own their
   * closest points, and the core points' separation along n (the weight-form
   * gap plus the gap's thickness). */
  private ellipticClosest(features: readonly [RodFeatureSpec, RodFeatureSpec], at: (v: number) => ClothVector, time: number, initial: ClothVector): {weights: number[]; normal: ClothVector; gap: number; owned: boolean; separation: number} {
    const fa = this.feature(features[0], at, time), fb = this.feature(features[1], at, time), q = rodFeatureDistance(fa, fb, initial);
    const wa = this.featureWeights(features[0], fa, q.ca), wb = this.featureWeights(features[1], fb, q.cb);
    // rodSelfClosest's ownership: a cap or point owns every pair, a segment interior only interior points of both.
    const owned = features[0].kind !== 'segment' || (wa.interior && wb.interior);
    return {weights: [...wa.weights, ...wb.weights.map(w => -w)], normal: q.normal, gap: q.gap, owned, separation: dot3(q.normal, sub3(q.ca, q.cb))};
  }
  /** An elliptical feature against a collider at `beforeEnd` before the step's
   * end: the signed gap, the normal from the collider and the core point. */
  private ellipticCollider(f: RodFeature, collider: ClothCollider, beforeEnd: number, initial?: ClothVector): {gap: number; normal: ClothVector; core: ClothVector; point: ClothVector} {
    if (collider.kind === 'plane') {
      // A clipped volume's lowest point is its support (any depth); its core point is that surface point.
      const n = scale3(collider.normal, 1/norm3(collider.normal)), core = f.clip ? supportClippedRod(f.clip, scale3(n, -1)) : f.core.reduce((best, p) => dot3(p, n) < dot3(best, n) ? p : best), point = sub3(core, featureSurface(f, n));
      return {gap: sampleClothCollider(collider, point, beforeEnd).distance_m, normal: n, core, point};
    }
    if (collider.kind === 'sampled') {
      // The field's least distance over the feature's analytic volume, by its Lipschitz bound;
      // the core point hangs the surface point from the centreline (or is it, for a cut volume).
      const shape = f.clip ?? f.shape;
      if (!shape) throw new Error('In-solve cloth contact: a rod feature without an analytic volume against a sampled solid');
      const q = sampleClippedRodCollider(shape, collider, beforeEnd, sampleClothCollider);
      return {gap: q.distance_m, normal: q.normal, core: f.clip ? q.point : add3(q.point, featureSurface(f, q.normal)), point: q.point};
    }
    const shift = scale3(collider.velocity_m_s ?? [0, 0, 0], -beforeEnd);
    let solid: RodFeature, face: {gap: number; normal: ClothVector; core: ClothVector; point: ClothVector} | undefined;
    if (collider.kind === 'sphere') solid = {core: [add3(collider.center_m, shift)], radius: collider.radius_m};
    else if (collider.kind === 'box') {
      const lo = add3(collider.minimum_m, shift), hi = add3(collider.maximum_m, shift);
      // A face's plane: the box lies in its inner half-space, so the signed distance is at
      // least the plane's, and equals it when apart with the feature's lowest point along the
      // face normal over the face. Exact, and unlike GJK's closest pair on two parallel flat
      // faces (any point of their overlap), the same support point as on a plane. Overlapping,
      // the face is the answer when the general distance from its normal finds nothing larger.
      for (let axis = 0; axis < 3; axis++) for (const side of [-1, 1] as const) {
        const n: ClothVector = [axis === 0 ? side : 0, axis === 1 ? side : 0, axis === 2 ? side : 0];
        const core = f.clip ? supportClippedRod(f.clip, scale3(n, -1)) : f.core.reduce((best, p) => dot3(p, n) < dot3(best, n) ? p : best), point = sub3(core, featureSurface(f, n));
        const gap = side*(point[axis]! - (side > 0 ? hi : lo)[axis]!);
        if (!Number.isFinite(gap) || (face && face.gap >= gap)) continue;
        if ([0, 1, 2].every(k => k === axis || (point[k]! >= lo[k]! && point[k]! <= hi[k]!))) face = {gap, normal: n, core, point};
      }
      if (face && face.gap >= 0) return face;
      const core: ClothVector[] = []; for (let k = 0; k < 8; k++) core.push([k & 1 ? hi[0] : lo[0], k & 2 ? hi[1] : lo[1], k & 4 ? hi[2] : lo[2]]); solid = {core, radius: 0};
    }
    else throw new Error(`In-solve cloth contact: rod features against ${collider.kind} colliders are not modelled by the convex distance`);
    const q = rodFeatureDistance(f, solid, face?.normal ?? initial ?? sub3(f.core[0]!, solid.core[0]!));
    if (face && !(q.gap > face.gap)) return face;
    return {gap: q.gap, normal: q.normal, core: q.ca, point: sub3(q.ca, featureSurface(f, q.normal))};
  }
  /** Elliptical segment ends and interiors against each collider (ec: cap, rc:
   * interior), cut ends (ec: cut cap) and cut segments' interiors, and round
   * segment interiors against boxes: conservative
   * advancement on the GJK gap, a cap's pair at touch, an interior's where the
   * closest core point is inside the segment (never against a plane, whose
   * closest point is an end). A round segment's ends are its vertices' pairs. */
  private detectEllipticColliders(x: ArrayLike<number>): number {
    const {colliders, dt, topology} = this;
    let added = 0;
    for (let segment = 0; segment < this.rodSegments.length; segment++) {
      const g = this.rodSegments[segment]!;
      if ((!this.featured(segment) && !colliders.some(c => c.kind === 'box')) || (!this.free[g.a] && !this.free[g.b])) continue;
      const specs: [string, RodFeatureSpec][] = [];
      // A cut end is its cut cap; a round rod's uncut end is its vertex's sphere (detect).
      for (const [end, vertex] of [[0, g.a], [1, g.b]] as const) {
        if (end ? g.clip?.end : g.clip?.start) specs.push([`ec:${2*segment + end}`, {kind: 'cut', segment, vertex}]);
        else if (g.ellipse) specs.push([`ec:${2*segment + end}`, {kind: 'cap', segment, vertex}]);
      }
      specs.push([`rc:${segment}`, {kind: 'segment', segment}]);
      for (const [prefix, spec] of specs) {
        if ((spec.kind === 'cap' || spec.kind === 'cut') && !this.free[spec.vertex]) continue;
        const reach = this.featureTravel(spec, x);
        for (let ci = 0; ci < colliders.length; ci++) {
          const collider = colliders[ci]!, key = `${prefix}:${ci}`;
          if (this.byKey.has(key) || (spec.kind === 'segment' && collider.kind === 'plane') || (!this.featured(segment) && collider.kind !== 'box')) continue;
          const travel = sweepBound(collider, reach, dt), tolerance = Math.max(1e-10, g.r*1e-6);
          let time = 0, hit: ReturnType<ClothContactSet['ellipticCollider']> | undefined, f: RodFeature | undefined;
          for (let iteration = 0; iteration < 160; iteration++) {
            f = this.feature(spec, v => lerp3(point3(this.old, v), point3(x, v), time), time);
            const q = this.ellipticCollider(f, collider, (1 - time)*dt);
            if (q.gap <= tolerance) {hit = q; break;}
            if (iteration === 0 && q.gap - travel > tolerance) break;
            if (time >= 1 || travel === 0) break;
            time = Math.min(1, time + Math.max(1e-9, .9*q.gap/travel));
          }
          if (!hit || !f) continue;
          const w = this.featureWeights(spec, f, hit.core);
          if (spec.kind === 'segment' && !w.interior) continue;
          const vertices = spec.kind === 'cap' || spec.kind === 'cut' ? [spec.vertex] : [g.a, g.b], carried = scale3(colliderVelocity(collider, hit.point), (1 - time)*dt);
          // The weight-form gap: the core point's motion from the touch along n, less the collider's.
          const offset = dot3(hit.normal, add3(hit.core, carried)) - hit.gap;
          const constraint = {key, kind: 'collider' as const, vertices, weights: w.weights, normal: [hit.normal[0], hit.normal[1], hit.normal[2]] as [number, number, number], offset, colliderIndex: ci, obstacle: scale3(colliderVelocity(collider, hit.point), dt), friction: collider.friction ?? topology.materials[topology.rods[g.rod]!.material]!.friction, radius: g.r};
          if (this.addRod(constraint, spec.kind === 'segment' ? 'segment-collider' : 'point-collider', vertices.length, 0, [this.side(segment, 1)], x, [spec, undefined])) added++;
        }
      }
    }
    return added;
  }
  /** Conservative advancement of an elliptical self pair from the old to the
   * given end positions: the pair at first touch if its features own their
   * closest points, with the weight-form offset that gives the touch's gap. */
  private sweepEllipticPair(features: readonly [RodFeatureSpec, RodFeatureSpec], x: ArrayLike<number>): {weights: number[]; normal: ClothVector; offset: number} | undefined {
    const travel = this.featureTravel(features[0], x) + this.featureTravel(features[1], x), tolerance = 1e-9;
    let time = 0, initial: ClothVector = [1, 0, 0];
    for (let iteration = 0; iteration < 100; iteration++) {
      const q = this.ellipticClosest(features, v => lerp3(point3(this.old, v), point3(x, v), time), time, initial);
      if (q.gap <= tolerance) return q.owned ? {weights: q.weights, normal: q.normal, offset: q.separation - q.gap} : undefined;
      if (time >= 1 || travel === 0) return undefined;
      initial = q.normal;
      time = Math.min(1, time + Math.max(1e-10, .9*q.gap/travel));
    }
    return undefined;
  }

  /** The reference CCD from the old to the given end positions. Pairs already
   * held keep their frozen geometry and multiplier; returns new pairs. */
  detect(x: ArrayLike<number>): number {
    const {topology, old, dt, colliders} = this;
    let added = 0;
    for (let i = 0; i < topology.vertexCount; i++) {
      const rodVertex = this.rodVertex(i);
      if (!this.free[i] || (topology.incidentTriangles[i]!.length === 0 && !rodVertex)) continue;
      // An elliptical rod's vertex is its segments' end caps, a cut end its cut cap (detectEllipticColliders).
      if (rodVertex && (this.elliptical(this.owner[i]!) || this.cutEnds.has(i))) continue;
      const start = point3(old, i), next = point3(x, i), delta = sub3(next, start), r = vertexRadius(topology, i);
      for (let ci = 0; ci < colliders.length; ci++) {
        const collider = colliders[ci]!, key = `c:${i}:${ci}`;
        if (this.byKey.has(key)) continue;
        const speed = sweepBound(collider, norm3(delta), dt);
        const tolerance = Math.max(1e-10, r*1e-6);
        let time = 0, hit = false, clear = false, hitNormal: ClothVector = [0, 1, 0], hitDistance = 0;
        for (let iteration = 0; iteration < 80; iteration++) {
          const at = lerp3(start, next, time), distance = colliderDistance(collider, at, (1 - time)*dt) - r;
          if (distance <= tolerance) {hit = true; hitNormal = sampleClothCollider(collider, at, (1 - time)*dt).normal; hitDistance = distance; break;}
          if (iteration === 0 && distance - speed > tolerance) {clear = true; break;}
          if (time >= 1 || speed === 0) break;
          time = Math.min(1, time + Math.max(1e-9, .9*distance/speed));
        }
        if (clear) continue;
        const end = sampleClothCollider(collider, next, 0);
        if (!hit && end.distance_m >= r) continue;
        // The tangent plane at the impact, carried to the collider's end pose.
        const impact = lerp3(start, next, time), normal = hit ? hitNormal : end.normal, carried = scale3(colliderVelocity(collider, impact), (1 - time)*dt);
        const plane = hit ? [impact[0] + carried[0], impact[1] + carried[1], impact[2] + carried[2]] as ClothVector : next;
        const gap = hit ? hitDistance : end.distance_m - r;
        const constraint = {key, kind: 'collider' as const, vertices: [i], weights: [1], normal: [normal[0], normal[1], normal[2]] as [number, number, number], offset: dot3(normal, plane) - gap, colliderIndex: ci, obstacle: scale3(colliderVelocity(collider, next), dt), friction: collider.friction ?? topology.materials[topology.vertexMaterial[i]!]!.friction, radius: r};
        if (rodVertex ? this.addRod(constraint, 'point-collider', 1, 0, [this.side(this.owner[i]!, 1)], x) : this.add(constraint)) added++;
      }
    }
    added += this.detectRodColliders(x) + this.detectEllipticColliders(x);
    if (this.selfCollision) {added += this.detectSelf(x); added += this.detectRodSelf(x);}
    return added;
  }

  /** A rod segment's interior against each collider (rc:segment:collider):
   * conservative advancement on the capsule's distance from the old to the
   * given end positions, kept where the closest centreline point lies inside
   * the segment (its ends are the rod-vertex spheres' pairs). */
  private detectRodColliders(x: ArrayLike<number>): number {
    const {topology, old, dt, colliders} = this;
    let added = 0;
    for (let segment = 0; segment < this.rodSegments.length; segment++) {
      const {a, b, r, rod} = this.rodSegments[segment]!;
      if (this.featured(segment) || (!this.free[a] && !this.free[b])) continue;
      const oa = point3(old, a), ob = point3(old, b), na = point3(x, a), nb = point3(x, b), reach = Math.max(norm3(sub3(na, oa)), norm3(sub3(nb, ob)));
      const shape = (time: number) => ({a: lerp3(oa, na, time), b: lerp3(ob, nb, time), radius_m: r, planes: []});
      for (let ci = 0; ci < colliders.length; ci++) {
        const collider = colliders[ci]!, key = `rc:${segment}:${ci}`;
        // A box meets the segment by the convex distance (detectEllipticColliders).
        if (this.byKey.has(key) || collider.kind === 'box') continue;
        const travel = sweepBound(collider, reach, dt);
        const tolerance = Math.max(1e-10, r*1e-6);
        let time = 0, hit: ReturnType<typeof sampleClippedRodCollider> | undefined;
        for (let iteration = 0; iteration < 160; iteration++) {
          const q = sampleClippedRodCollider(shape(time), collider, (1 - time)*dt, sampleClothCollider);
          if (q.distance_m <= tolerance) {hit = q; break;}
          if (iteration === 0 && q.lowerBound_m - travel > tolerance) break;
          if (time >= 1 || travel === 0) break;
          if (!(q.lowerBound_m > 0)) throw new Error(`Rod segment ${segment} collider distance failed to separate`);
          time = Math.min(1, time + Math.max(1e-9, .9*q.lowerBound_m/travel));
        }
        if (!hit) continue;
        const at = shape(time), t = closestPointSegment(add3(hit.point, scale3(hit.normal, r)), at.a, at.b).t;
        if (!(t > ROD_INTERIOR && t < 1 - ROD_INTERIOR)) continue;
        const normal = hit.normal, carried = scale3(colliderVelocity(collider, hit.point), (1 - time)*dt), plane = add3(lerp3(at.a, at.b, t), carried);
        const constraint = {key, kind: 'collider' as const, vertices: [a, b], weights: [1 - t, t], normal: [normal[0], normal[1], normal[2]] as [number, number, number], offset: dot3(normal, plane) - hit.distance_m, colliderIndex: ci, obstacle: scale3(colliderVelocity(collider, lerp3(na, nb, t)), dt), friction: collider.friction ?? topology.materials[topology.rods[rod]!.material]!.friction, radius: r};
        if (this.addRod(constraint, 'segment-collider', 2, 0, [this.side(segment, 1)], x)) added++;
      }
    }
    return added;
  }

  /** Conservative advancement of a self pair from the old to the given end
   * positions; the pair at first touch if this shape owns its closest points. */
  private sweepRodPair(shape: RodContactShape, ids: readonly number[], first: number, thickness: number, x: ArrayLike<number>): {weights: number[]; normal: ClothVector} | undefined {
    const start = ids.map(i => point3(this.old, i)), next = ids.map(i => point3(x, i)), moved = start.map((q, j) => norm3(sub3(next[j]!, q)));
    const travel = Math.max(...moved.slice(0, first)) + Math.max(...moved.slice(first));
    let time = 0;
    for (let iteration = 0; iteration < 100; iteration++) {
      const q = rodSelfClosest(shape, start.map((o, j) => lerp3(o, next[j]!, time))), gap = norm3(q.delta);
      if (gap <= thickness + 1e-9) return q.owned ? {weights: q.weights, normal: unit(q.delta, q.fallback)} : undefined;
      if (time >= 1 || travel === 0) return undefined;
      time = Math.min(1, time + Math.max(1e-10, .9*(gap - thickness)/travel));
    }
    return undefined;
  }

  /** Rod self contact: rod-vertex spheres against faces (rv:), shell edges
   * (rq:) and rod segments (rp:, also every shell vertex against rod
   * segments), and segment interiors against faces (rf:), rod segments (rs:)
   * and shell edges (rx:). Excluded: pairs sharing a vertex or a triangle
   * (the sequential path's exclusions), and same-rod pairs overlapping in the
   * rest shape. Runs after detectSelf, whose face and edge trees it reuses. */
  private detectRodSelf(x: ArrayLike<number>): number {
    const {topology, old} = this, segments = this.rodSegments, cache = clothShellCache(topology), rest = topology.data.restPositions_m;
    if (!segments.length) return 0;
    if (this.rodBounds.length !== 6*segments.length) {this.rodBounds = new Float64Array(6*segments.length); this.rodTree = undefined;}
    for (let s = 0; s < segments.length; s++) {
      const {a, b, reach: r} = segments[s]!;
      for (let axis = 0; axis < 3; axis++) {
        this.rodBounds[6*s + axis] = Math.min(old[3*a + axis]!, old[3*b + axis]!, x[3*a + axis]!, x[3*b + axis]!) - r;
        this.rodBounds[6*s + 3 + axis] = Math.max(old[3*a + axis]!, old[3*b + axis]!, x[3*a + axis]!, x[3*b + axis]!) + r;
      }
    }
    if (this.rodTree) this.rodTree.refit(); else this.rodTree = new ClothContactBVH(this.rodBounds);
    const rodTree = this.rodTree, faceTree = topology.triangles.length ? cache.faceTree : undefined, edgeTree = cache.edges.length ? cache.edgeTree : undefined;
    // Feature sets share a vertex, or a vertex of one lies on a triangle of the other's.
    const touches = (near: readonly number[], others: readonly number[]) => others.some(v => near.includes(v) || near.some(u => topology.incidentTriangles[u]!.some(f => topology.triangles[f]!.vertices.includes(v))));
    const friction = (ids: readonly number[]) => Math.min(...ids.map(v => topology.materials[topology.vertexMaterial[v]!]!.friction));
    const box = (ids: readonly number[], pad: number) => [[0, 1, 2].map(k => Math.min(...ids.flatMap(i => [old[3*i + k]!, x[3*i + k]!])) - pad), [0, 1, 2].map(k => Math.max(...ids.flatMap(i => [old[3*i + k]!, x[3*i + k]!])) + pad)] as const;
    const restGap = (shape: RodContactShape, ids: readonly number[]) => norm3(rodSelfClosest(shape, ids.map(i => point3(rest, i))).delta);
    const sameRod = (vertexOrSegment: {vertex?: number; segment?: number}, segment: number) => {const r = vertexOrSegment.segment !== undefined ? segments[vertexOrSegment.segment]!.rod : this.owner[vertexOrSegment.vertex!]! >= 0 ? segments[this.owner[vertexOrSegment.vertex!]!]!.rod : -1; return r === segments[segment]!.rod;};
    let added = 0;
    const pair = (key: string, shape: RodContactShape, ids: number[], first: number, thickness: number, radius: number, sides: RodContactSide[], features?: readonly [RodFeatureSpec, RodFeatureSpec]) => {
      if (this.byKey.has(key)) return;
      const hit: {weights: number[]; normal: ClothVector; offset?: number} | undefined = features ? this.sweepEllipticPair(features, x) : this.sweepRodPair(shape, ids, first, thickness, x);
      if (hit && this.addRod({key, kind: 'self', vertices: ids, weights: hit.weights, normal: [hit.normal[0], hit.normal[1], hit.normal[2]], offset: hit.offset ?? thickness, colliderIndex: -1, obstacle: [0, 0, 0], friction: friction(ids), radius}, shape, first, ids.length - first, sides, x, features)) added++;
    };
    const segmentSpec = (s: number): RodFeatureSpec => ({kind: 'segment', segment: s});
    // An elliptical rod's vertices are its segments' end caps, and a cut end
    // its cut cap: each cap against faces (ev:), shell edges (eq:) and rod
    // segments (ep:), with the exclusions of the rod-vertex sphere it replaces.
    for (let s = 0; s < segments.length; s++) {
      const g = segments[s]!;
      if (!this.featured(s)) continue;
      for (const [end, v] of [[0, g.a], [1, g.b]] as const) {
        const cut = end ? g.clip?.end : g.clip?.start;
        if (!this.rodVertex(v) || (!cut && !g.ellipse)) continue;
        const cap: RodFeatureSpec = {kind: cut ? 'cut' : 'cap', segment: s, vertex: v}, k = 2*s + end, [min, max] = box([v], g.reach), near = [v, ...(this.rodNeighbours.get(v) ?? [])], own = [this.side(s, 1)];
        if (faceTree) for (const f of faceTree.query(min, max)) {const face = topology.triangles[f]!, r = .5*topology.materials[face.material]!.thickness_m; if (!touches(near, face.vertices)) pair(`ev:${k}:${f}`, 'point-face', [v, ...face.vertices], 1, g.r + r, g.r, own, [cap, {kind: 'face', vertices: face.vertices, radius: r}]);}
        if (edgeTree) for (const e of edgeTree.query(min, max)) {const edge = cache.edges[e]!; if (!touches(near, edge.vertices)) pair(`eq:${k}:${e}`, 'point-segment', [v, ...edge.vertices], 1, g.r + edge.r, g.r, [this.side(s, 1)], [cap, {kind: 'edge', vertices: edge.vertices, radius: edge.r}]);}
        for (const u of rodTree.query(min, max)) {
          const h = segments[u]!;
          if (h.a === v || h.b === v) continue;
          if (sameRod({segment: s}, u) && restGap('point-segment', [v, h.a, h.b]) <= g.r + h.r + 1e-12) continue;
          pair(`ep:${k}:${u}`, 'point-segment', [v, h.a, h.b], 1, g.r + h.r, g.r, [this.side(s, 1), this.side(u, -1)], [cap, segmentSpec(u)]);
        }
      }
    }
    for (let i = 0; i < topology.vertexCount; i++) {
      const rodVertex = this.rodVertex(i), ri = vertexRadius(topology, i), [min, max] = box([i], ri);
      if (rodVertex && (this.elliptical(this.owner[i]!) || this.cutEnds.has(i))) continue;
      if (rodVertex) {
        const near = [i, ...(this.rodNeighbours.get(i) ?? [])], own = [this.side(this.owner[i]!, 1)];
        if (faceTree) for (const f of faceTree.query(min, max)) {const face = topology.triangles[f]!; if (!touches(near, face.vertices)) pair(`rv:${i}:${f}`, 'point-face', [i, ...face.vertices], 1, ri + .5*topology.materials[face.material]!.thickness_m, ri, own);}
        if (edgeTree) for (const e of edgeTree.query(min, max)) {const edge = cache.edges[e]!; if (!touches(near, edge.vertices)) pair(`rq:${i}:${e}`, 'point-segment', [i, ...edge.vertices], 1, ri + edge.r, ri, [this.side(this.owner[i]!, 1)]);}
      }
      if (!rodVertex && topology.incidentTriangles[i]!.length === 0) continue;
      for (const s of rodTree.query(min, max)) {
        const g = segments[s]!;
        if (g.a === i || g.b === i || (!rodVertex && touches([i], [g.a, g.b]))) continue;
        if (sameRod({vertex: i}, s) && restGap('point-segment', [i, g.a, g.b]) <= ri + g.r + 1e-12) continue;
        pair(`rp:${i}:${s}`, 'point-segment', [i, g.a, g.b], 1, ri + g.r, ri, rodVertex ? [this.side(this.owner[i]!, 1), this.side(s, -1)] : [this.side(s, -1)], this.featured(s) ? [{kind: 'point', vertex: i, radius: ri}, segmentSpec(s)] : undefined);
      }
    }
    for (let s = 0; s < segments.length; s++) {
      const g = segments[s]!, ids = [g.a, g.b], min = this.rodBounds.subarray(6*s, 6*s + 3), max = this.rodBounds.subarray(6*s + 3, 6*s + 6);
      if (faceTree) for (const f of faceTree.query([...min], [...max])) {const face = topology.triangles[f]!, r = .5*topology.materials[face.material]!.thickness_m; if (!touches(ids, face.vertices)) pair(`rf:${s}:${f}`, 'segment-face', [...ids, ...face.vertices], 2, g.r + r, g.r, [this.side(s, 1)], this.featured(s) ? [segmentSpec(s), {kind: 'face', vertices: face.vertices, radius: r}] : undefined);}
      if (edgeTree) for (const e of edgeTree.query([...min], [...max])) {const edge = cache.edges[e]!; if (!touches(ids, edge.vertices)) pair(`rx:${s}:${e}`, 'segment-segment', [...ids, ...edge.vertices], 2, g.r + edge.r, g.r, [this.side(s, 1)], this.featured(s) ? [segmentSpec(s), {kind: 'edge', vertices: edge.vertices, radius: edge.r}] : undefined);}
      for (const u of rodTree.query([...min], [...max])) {
        if (u <= s) continue;
        const h = segments[u]!;
        if (touches(ids, [h.a, h.b])) continue;
        if (g.rod === h.rod && restGap('segment-segment', [...ids, h.a, h.b]) <= g.r + h.r + 1e-12) continue;
        pair(`rs:${s}:${u}`, 'segment-segment', [...ids, h.a, h.b], 2, g.r + h.r, g.r, [this.side(s, 1), this.side(u, -1)], this.featured(s) || this.featured(u) ? [segmentSpec(s), segmentSpec(u)] : undefined);
      }
    }
    return added;
  }

  private detectSelf(x: ArrayLike<number>): number {
    const {topology, old} = this, cache = clothShellCache(topology);
    let added = 0;
    for (let f = 0; f < topology.triangles.length; f++) {
      const face = topology.triangles[f]!, r = .5*topology.materials[face.material]!.thickness_m;
      for (let a = 0; a < 3; a++) {
        let lo = Infinity, hi = -Infinity;
        for (const v of face.vertices) {lo = Math.min(lo, old[3*v + a]!, x[3*v + a]!); hi = Math.max(hi, old[3*v + a]!, x[3*v + a]!);}
        cache.faceBounds[6*f + a] = lo - r; cache.faceBounds[6*f + a + 3] = hi + r;
      }
    }
    if (cache.faceTree) cache.faceTree.refit(); else cache.faceTree = new ClothContactBVH(cache.faceBounds);
    const shares = (i: number, face: readonly number[]) => face.includes(i) || face.some(v => topology.incidentTriangles[i]!.some(g => topology.triangles[g]!.vertices.includes(v)));
    for (let i = 0; i < topology.vertexCount; i++) {
      if (topology.incidentTriangles[i]!.length === 0) continue;
      const ri = vertexRadius(topology, i), p0 = point3(old, i), p1 = point3(x, i);
      const min = [0, 1, 2].map(a => Math.min(p0[a]!, p1[a]!) - ri), max = [0, 1, 2].map(a => Math.max(p0[a]!, p1[a]!) + ri);
      for (const faceIndex of cache.faceTree.query(min, max)) {
        const face = topology.triangles[faceIndex]!, ids = face.vertices, key = `v:${i}:${faceIndex}`;
        if (this.byKey.has(key) || shares(i, ids) || clothVertexLocalFaces(topology, i).has(faceIndex)) continue;
        const triangleOld = ids.map(v => point3(old, v)), triangleNew = ids.map(v => point3(x, v)), distance = ri + .5*topology.materials[face.material]!.thickness_m;
        if ([0, 1, 2].some(a => Math.max(p0[a]!, p1[a]!) + distance < Math.min(...triangleOld.map(q => q[a]!), ...triangleNew.map(q => q[a]!)) || Math.min(p0[a]!, p1[a]!) - distance > Math.max(...triangleOld.map(q => q[a]!), ...triangleNew.map(q => q[a]!)))) continue;
        const bound = norm3(sub3(p1, p0)) + Math.max(...triangleOld.map((q, j) => norm3(sub3(triangleNew[j]!, q))));
        let time = 0, found = false, normal: ClothVector = [0, 1, 0], weights: ClothVector = [1, 0, 0];
        for (let iteration = 0; iteration < 100; iteration++) {
          const v = lerp3(p0, p1, time), [a, b, c] = triangleOld.map((q, j) => lerp3(q, triangleNew[j]!, time)) as unknown as [ClothVector, ClothVector, ClothVector];
          const closest = closestTrianglePoint(v, a, b, c), delta = sub3(v, closest.point), gap = norm3(delta);
          if (gap <= distance + 1e-9) {found = true; weights = closest.weights; const n = cross3(sub3(b, a), sub3(c, a)); normal = gap > 1e-12 ? scale3(delta, 1/gap) : scale3(n, 1/Math.max(1e-20, norm3(n))); break;}
          if (time >= 1 || bound === 0) break;
          time = Math.min(1, time + Math.max(1e-10, .9*(gap - distance)/bound));
        }
        if (!found) continue;
        const friction = Math.min(...[i, ...ids].map(v => topology.materials[topology.vertexMaterial[v]!]!.friction));
        if (this.add({key, kind: 'self', vertices: [i, ...ids], weights: [1, -weights[0], -weights[1], -weights[2]], normal: [normal[0], normal[1], normal[2]], offset: distance, colliderIndex: -1, obstacle: [0, 0, 0], friction, radius: ri})) added++;
      }
    }
    // Edge-edge: two long edges crossing away from their endpoints.
    const edges = cache.edges;
    for (let e = 0; e < edges.length; e++) {
      const edge = edges[e]!, [a, b] = edge.vertices;
      for (let axis = 0; axis < 3; axis++) {
        edge.min[axis] = Math.min(old[3*a + axis]!, old[3*b + axis]!, x[3*a + axis]!, x[3*b + axis]!) - edge.r;
        edge.max[axis] = Math.max(old[3*a + axis]!, old[3*b + axis]!, x[3*a + axis]!, x[3*b + axis]!) + edge.r;
      }
      cache.edgeBounds.set(edge.min, 6*e); cache.edgeBounds.set(edge.max, 6*e + 3);
    }
    if (cache.edgeTree) cache.edgeTree.refit(); else cache.edgeTree = new ClothContactBVH(cache.edgeBounds);
    for (let ai = 0; ai < edges.length; ai++) for (const bi of cache.edgeTree.query(edges[ai]!.min, edges[ai]!.max)) {
      if (bi <= ai) continue;
      const ea = edges[ai]!, eb = edges[bi]!, ids = [...ea.vertices, ...eb.vertices], key = `e:${ai}:${bi}`;
      if (this.byKey.has(key) || new Set(ids).size < 4) continue;
      if (ea.vertices.some(p => eb.vertices.some(q => topology.incidentTriangles[p]!.some(f => topology.triangles[f]!.vertices.includes(q))))) continue;
      if (clothLocalShellEdges(topology, ea, eb)) continue;
      const start = ids.map(i => point3(old, i)), next = ids.map(i => point3(x, i)), thickness = ea.r + eb.r;
      const travel = Math.max(norm3(sub3(next[0]!, start[0]!)), norm3(sub3(next[1]!, start[1]!))) + Math.max(norm3(sub3(next[2]!, start[2]!)), norm3(sub3(next[3]!, start[3]!)));
      let time = 0, found = false, normal: ClothVector = [0, 1, 0], s = 0, t = 0;
      for (let it = 0; it < 100; it++) {
        const [p, q, u, v] = start.map((o, j) => lerp3(o, next[j]!, time)) as unknown as [ClothVector, ClothVector, ClothVector, ClothVector];
        const closest = closestSegmentPoints(p, q, u, v), delta = sub3(closest.a, closest.b), gap = norm3(delta);
        if (gap <= thickness + 1e-9) {
          found = true; s = closest.s; t = closest.t;
          const fallback = cross3(sub3(q, p), sub3(v, u));
          normal = gap > 1e-12 ? scale3(delta, 1/gap) : norm3(fallback) > 1e-12 ? scale3(fallback, 1/norm3(fallback)) : [0, 1, 0];
          break;
        }
        if (time >= 1 || travel === 0) break;
        time = Math.min(1, time + Math.max(1e-10, .9*(gap - thickness)/travel));
      }
      if (!found) continue;
      const friction = Math.min(...ids.map(i => topology.materials[topology.vertexMaterial[i]!]!.friction));
      if (this.add({key, kind: 'self', vertices: ids, weights: [1 - s, s, -(1 - t), -t], normal: [normal[0], normal[1], normal[2]], offset: thickness, colliderIndex: -1, obstacle: [0, 0, 0], friction, radius: ea.r})) added++;
    }
    return added;
  }

  /** Every pair re-linearized at x (end-of-step pose): collider tangent planes,
   * and the closest points of self pairs with their normal kept on the side of
   * detection. Returns the largest true penetration (thickness minus distance). */
  relinearize(x: ArrayLike<number>): number {
    let penetration = 0;
    for (const c of this.constraints) {
      if (c.rod) {penetration = Math.max(penetration, this.relinearizeRod(c, x)); continue;}
      if (c.kind === 'self') {penetration = Math.max(penetration, this.relinearizeSelf(c, x)); continue;}
      const i = c.vertices[0]!, p = point3(x, i), collider = this.colliders[c.colliderIndex]!, sample = sampleClothCollider(collider, p, 0);
      // Crossed when the step's path enters the solid. Not when the nearest normal turns: past a
      // cavity's medial plane that is the opposite wall, with the path in free space throughout.
      if (pathEntersCollider(collider, point3(this.old, i), p, this.dt, Math.max(1e-10, c.radius*1e-6))) throw new Error(`Cloth vertex ${i} crossed collider ${c.colliderIndex} within the step`);
      c.normal = [sample.normal[0], sample.normal[1], sample.normal[2]];
      c.offset = dot3(sample.normal, p) - (sample.distance_m - c.radius);
      c.obstacle = scale3(colliderVelocity(collider, p), this.dt);
      penetration = Math.max(penetration, c.radius - sample.distance_m);
    }
    return penetration;
  }

  private relinearizeSelf(c: ClothContactConstraint, x: ArrayLike<number>): number {
    const [p, a, b, d] = c.vertices.map(v => point3(x, v)) as unknown as [ClothVector, ClothVector, ClothVector, ClothVector];
    let delta: ClothVector, fallback: ClothVector;
    if (c.key.startsWith('v:')) {
      const closest = closestTrianglePoint(p, a, b, d);
      delta = sub3(p, closest.point); fallback = cross3(sub3(b, a), sub3(d, a));
      c.weights = [1, -closest.weights[0], -closest.weights[1], -closest.weights[2]];
    } else {
      const closest = closestSegmentPoints(p, a, b, d);
      delta = sub3(closest.a, closest.b); fallback = cross3(sub3(a, p), sub3(d, b));
      c.weights = [1 - closest.s, closest.s, -(1 - closest.t), -closest.t];
    }
    const length = norm3(delta), f = norm3(fallback);
    let normal = length > 1e-12 ? scale3(delta, 1/length) : f > 1e-12 ? scale3(fallback, 1/f) : c.normal;
    if (dot3(normal, c.normal) < 0) normal = scale3(normal, -1);
    c.normal = [normal[0], normal[1], normal[2]];
    return c.offset - dot3(normal, delta);
  }

  /** A rod pair at x: weights and normal of its closest points (the normal
   * kept on the side of detection), its rolling Jacobian; returns the true
   * penetration. With apply false it only measures. */
  private relinearizeRod(c: ClothContactConstraint, x: ArrayLike<number>, apply = true): number {
    const g = c.rod!, q = c.vertices.slice(0, g.first + g.second).map(v => point3(x, v));
    if (g.features) {
      // An elliptical pair: the GJK gap of its features at x, the normal from the second (or the collider).
      const live = (v: number) => point3(x, v), [spec, other] = g.features;
      return this.relinearizeFeatures(c, x, live, spec, other, apply);
    }
    if (g.shape === 'point-collider' || g.shape === 'segment-collider') {
      const collider = this.colliders[c.colliderIndex]!;
      const sample = g.shape === 'point-collider' ? (() => {const s = sampleClothCollider(collider, q[0]!, 0); return {distance_m: s.distance_m - c.radius, normal: s.normal, point: sub3(q[0]!, scale3(s.normal, c.radius))};})() : sampleClippedRodCollider({a: q[0]!, b: q[1]!, radius_m: c.radius, planes: []}, collider, 0, sampleClothCollider);
      if (!apply) return -sample.distance_m;
      // Crossed when the step's path (the rod vertex, or the segment's centreline) enters the solid.
      const [a, b] = [c.vertices[0]!, c.vertices[g.shape === 'segment-collider' ? 1 : 0]!], tolerance = Math.max(1e-10, c.radius*1e-6);
      if (g.shape === 'point-collider' ? pathEntersCollider(collider, point3(this.old, a), q[0]!, this.dt, tolerance) : segmentEntersCollider(collider, point3(this.old, a), point3(this.old, b), q[0]!, q[1]!, this.dt, tolerance)) throw new Error(`Cloth rod contact ${c.key} crossed collider ${c.colliderIndex} within the step`);
      if (g.shape === 'segment-collider') {const t = closestPointSegment(add3(sample.point, scale3(sample.normal, c.radius)), q[0]!, q[1]!).t; c.weights[0] = 1 - t; c.weights[1] = t;}
      c.normal = [sample.normal[0], sample.normal[1], sample.normal[2]];
      c.obstacle = scale3(colliderVelocity(collider, sample.point), this.dt);
      this.linearizeRolling(c, x, sample.distance_m);
      return -sample.distance_m;
    }
    const closest = rodSelfClosest(g.shape, q);
    if (!apply) return g.thickness - norm3(closest.delta);
    let normal = unit(closest.delta, closest.fallback);
    if (norm3(closest.delta) <= 1e-12 && norm3(closest.fallback) <= 1e-12) normal = c.normal;
    if (dot3(normal, c.normal) < 0) normal = scale3(normal, -1);
    for (let j = 0; j < closest.weights.length; j++) c.weights[j] = closest.weights[j]!;
    c.normal = [normal[0], normal[1], normal[2]];
    const gap = dot3(normal, closest.delta) - g.thickness;
    this.linearizeRolling(c, x, gap);
    return -gap;
  }
  /** A feature pair at x: the convex distance of its features (the collider's
   * for one feature), its weights, normal and rolling Jacobian; returns the
   * true penetration. With apply false it only measures. */
  private relinearizeFeatures(c: ClothContactConstraint, x: ArrayLike<number>, live: (v: number) => ClothVector, spec: RodFeatureSpec, other: RodFeatureSpec | undefined, apply = true): number {
    if (!other) {
      const collider = this.colliders[c.colliderIndex]!, f = this.feature(spec, live, 1), hit = this.ellipticCollider(f, collider, 0, c.normal);
      if (!apply) return -hit.gap;
      const tolerance = Math.max(1e-10, c.radius*1e-6), [oa, ob] = this.insidePath(spec, v => point3(this.old, v)), [na, nb] = this.insidePath(spec, live);
      if (spec.kind === 'cap' || spec.kind === 'cut' ? pathEntersCollider(collider, oa, na, this.dt, tolerance) : segmentEntersCollider(collider, oa, ob, na, nb, this.dt, tolerance)) throw new Error(`Cloth rod contact ${c.key} crossed collider ${c.colliderIndex} within the step`);
      this.featureWeights(spec, f, hit.core).weights.forEach((w, j) => {c.weights[j] = w;});
      c.normal = [hit.normal[0], hit.normal[1], hit.normal[2]];
      c.obstacle = scale3(colliderVelocity(collider, hit.point), this.dt);
      this.linearizeRolling(c, x, hit.gap);
      return -hit.gap;
    }
    const pairAt = this.ellipticClosest([spec, other], live, 1, c.normal);
    if (!apply) return -pairAt.gap;
    pairAt.weights.forEach((w, j) => {c.weights[j] = w;});
    c.normal = [pairAt.normal[0], pairAt.normal[1], pairAt.normal[2]];
    this.linearizeRolling(c, x, pairAt.gap);
    return -pairAt.gap;
  }
  /** The rod material point's slip over the step (rolling included), in the tangent plane. */
  private rodSlip(c: ClothContactConstraint, x: ArrayLike<number>): [number, number, number] {
    const g = c.rod!, J = g.jacobian, u: [number, number, number] = [-c.obstacle[0], -c.obstacle[1], -c.obstacle[2]];
    for (let j = 0; j < c.vertices.length; j++) {
      const v = c.vertices[j]!, d0 = x[3*v]! - this.old[3*v]!, d1 = x[3*v + 1]! - this.old[3*v + 1]!, d2 = x[3*v + 2]! - this.old[3*v + 2]!;
      for (let p = 0; p < 3; p++) u[p] += J[9*j + 3*p]!*d0 + J[9*j + 3*p + 1]!*d1 + J[9*j + 3*p + 2]!*d2;
    }
    for (const side of g.sides) {
      if (!this.rodTwist) throw new Error('In-solve rod contact friction needs the rod twist state');
      const turn = side.sign*(this.rodTwist.current[side.segment]! - this.rodTwist.old[side.segment]!);
      for (let p = 0; p < 3; p++) u[p] += turn*side.tau[p]!;
    }
    const un = dot3(u, c.normal);
    return [u[0] - un*c.normal[0], u[1] - un*c.normal[1], u[2] - un*c.normal[2]];
  }
  /** (J_j^T P J_j)_ab, P the tangent projector. */
  private rodTangentBlock(c: ClothContactConstraint, j: number, a: number, b: number): number {
    const J = c.rod!.jacobian, n = c.normal;
    let full = 0, na = 0, nb = 0;
    for (let p = 0; p < 3; p++) {full += J[9*j + 3*p + a]!*J[9*j + 3*p + b]!; na += n[p]!*J[9*j + 3*p + a]!; nb += n[p]!*J[9*j + 3*p + b]!;}
    return full - na*nb;
  }

  gap(c: ClothContactConstraint, x: ArrayLike<number>): number {
    let g = -c.offset;
    if (c.rod) {
      const rows = c.rod.rows;
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!; g += rows[3*j]!*x[3*v]! + rows[3*j + 1]!*x[3*v + 1]! + rows[3*j + 2]!*x[3*v + 2]!;}
      for (const side of c.rod.sides) if (side.twistRow) g += side.twistRow*this.twistAt(side.segment);
      return g;
    }
    for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; g += w*(c.normal[0]*x[3*v]! + c.normal[1]*x[3*v+1]! + c.normal[2]*x[3*v+2]!);}
    return g;
  }
  private slip(c: ClothContactConstraint, x: ArrayLike<number>): [number, number, number] {
    const u: [number, number, number] = [-c.obstacle[0], -c.obstacle[1], -c.obstacle[2]];
    for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; for (let a = 0; a < 3; a++) u[a] += w*(x[3*v + a]! - this.old[3*v + a]!);}
    const un = dot3(u, c.normal);
    return [u[0] - un*c.normal[0], u[1] - un*c.normal[1], u[2] - un*c.normal[2]];
  }
  private f0(y: number): number {const e = this.epsilon; return y < e ? -y*y*y/(3*e*e) + y*y/e + e/3 : y;}
  private f1OverY(y: number): number {const e = this.epsilon; return y < e ? 2/e - y/(e*e) : 1/y;}

  /** Penalties for new constraints from the coordinate stiffness (3n) the
   * solver evaluates without contact: rho = ratio / sum_j w_j^2 / (n.S_j n). */
  assignPenalties(stiffness: ArrayLike<number>): void {
    for (const c of this.constraints) {
      if (c.rho > 0) continue;
      let compliance = 0;
      for (let j = 0; j < c.vertices.length; j++) {
        const v = c.vertices[j]!; if (!this.free[v]) continue;
        // The row's squared length over the stiffness along it (w_j^2 / n.S n for w_j n).
        const r0 = c.rod ? c.rod.rows[3*j]! : c.weights[j]!*c.normal[0], r1 = c.rod ? c.rod.rows[3*j + 1]! : c.weights[j]!*c.normal[1], r2 = c.rod ? c.rod.rows[3*j + 2]! : c.weights[j]!*c.normal[2], rr = r0*r0 + r1*r1 + r2*r2;
        if (!(rr > 0)) continue;
        const d = (r0*r0*stiffness[3*v]! + r1*r1*stiffness[3*v+1]! + r2*r2*stiffness[3*v+2]!)/rr;
        compliance += rr/d;
      }
      if (!(compliance > 0 && Number.isFinite(compliance))) throw new Error(`Cloth contact ${c.key} has no free stiffness`);
      c.rho = PENALTY_RATIO/compliance;
    }
  }

  energy(x: ArrayLike<number>): number {
    let energy = 0;
    for (const c of this.constraints) {
      energy += this.penalty(c, c.lambda/c.rho - this.gap(c, x));
      if (c.friction > 0 && c.frictionLambda > 0) energy += c.friction*c.frictionLambda*this.f0(norm3(c.rod ? this.rodSlip(c, x) : this.slip(c, x)));
    }
    return energy;
  }
  /** Adds the gradient to `gradient` and the diagonal of the Hessian model to
   * `diagonal` (both 3n), and stores each constraint's Hessian coefficients.
   * Rod friction also adds its twist gradient and curvature (per segment);
   * the model leaves the twist-position coupling out, as the rod energy's. */
  accumulate(x: ArrayLike<number>, gradient: Float64Array, diagonal: Float64Array, twistGradient?: Float64Array, twistDiagonal?: Float64Array): void {
    for (const c of this.constraints) {
      const m = c.lambda/c.rho - this.gap(c, x), force = this.force(c, m);
      c.normalCoefficient = this.curvature(c, m);
      let slip: [number, number, number] = [0, 0, 0], f = 0;
      if (c.friction > 0 && c.frictionLambda > 0) {slip = c.rod ? this.rodSlip(c, x) : this.slip(c, x); f = c.friction*c.frictionLambda*this.f1OverY(norm3(slip));}
      c.frictionCoefficient = f;
      if (c.rod) {
        const J = c.rod.jacobian;
        const rows = c.rod.rows;
        for (let j = 0; j < c.vertices.length; j++) {
          const v = c.vertices[j]!;
          if (!this.free[v]) continue;
          for (let a = 0; a < 3; a++) {
            gradient[3*v + a]! += -force*rows[3*j + a]! + f*(J[9*j + a]!*slip[0] + J[9*j + 3 + a]!*slip[1] + J[9*j + 6 + a]!*slip[2]);
            diagonal[3*v + a]! += c.normalCoefficient*rows[3*j + a]!**2 + f*this.rodTangentBlock(c, j, a, a);
          }
        }
        for (const side of c.rod.sides) {
          if (!this.twistFree[side.segment] || (!f && !side.twistRow)) continue;
          if (!twistGradient || !twistDiagonal) throw new Error('In-solve rod contact needs the twist gradient');
          const tn = dot3(side.tau, c.normal);
          // The normal force's torque on an elliptical section, then the rolling friction.
          twistGradient[side.segment]! += -force*side.twistRow + f*side.sign*dot3(side.tau, slip);
          twistDiagonal[side.segment]! += c.normalCoefficient*side.twistRow**2 + f*(dot3(side.tau, side.tau) - tn*tn);
        }
        continue;
      }
      for (let j = 0; j < c.vertices.length; j++) {
        const v = c.vertices[j]!, w = c.weights[j]!;
        if (!this.free[v]) continue;
        for (let a = 0; a < 3; a++) {
          gradient[3*v + a]! += w*(-force*c.normal[a]! + f*slip[a]!);
          diagonal[3*v + a]! += w*w*(c.normalCoefficient*c.normal[a]!**2 + f*(1 - c.normal[a]!**2));
        }
      }
    }
  }
  // The augmented-Lagrangian penalty of m = lambda/rho - g: rho m^2/2 for
  // m >= delta, zero below -delta, its curvature ramping linearly in between
  // (C2), with delta the solver tolerance. A max(0, m)^2 penalty has a Hessian
  // jump at contact; iterates hovering there (|g| ~ tolerance) alternate between
  // a model that aims through the contact and one that blocks a separating
  // pair, and Newton stalls. The smoothed force acts across gaps below delta,
  // which the tolerance already admits.
  private penalty(c: ClothContactConstraint, m: number): number {
    const d = this.delta;
    return m <= -d ? 0 : m < d ? c.rho*(m + d)**3/(12*d) : .5*c.rho*m*m + c.rho*d*d/6;
  }
  private force(c: ClothContactConstraint, m: number): number {
    const d = this.delta;
    return m <= -d ? 0 : m < d ? c.rho*(m + d)**2/(4*d) : c.rho*m;
  }
  private curvature(c: ClothContactConstraint, m: number): number {
    const d = this.delta;
    return m <= -d ? 0 : m < d ? c.rho*(m + d)/(2*d) : c.rho;
  }
  /** output += (sum_k rho_k a_k a_k^T + friction tangent terms) input. With
   * twistOffset, input and output also carry the rod twist from that index,
   * and rod friction couples it to the positions (the rolling slip). */
  multiply(input: ArrayLike<number>, output: Float64Array, twistOffset = -1): void {
    for (const c of this.constraints) {
      if (!c.normalCoefficient && !c.frictionCoefficient) continue;
      if (c.rod) {this.multiplyRod(c, input, output, twistOffset); continue;}
      const s = [0, 0, 0];
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) s[a]! += w*input[3*v + a]!;}
      const sn = s[0]!*c.normal[0] + s[1]!*c.normal[1] + s[2]!*c.normal[2];
      const t = [0, 1, 2].map(a => c.normalCoefficient*sn*c.normal[a]! + c.frictionCoefficient*(s[a]! - sn*c.normal[a]!));
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) output[3*v + a]! += w*t[a]!;}
    }
  }
  /** Rod pair: normal rows by the gap rows (and an elliptical section's twist
   * rows), friction rows by the slip Jacobian. */
  private multiplyRod(c: ClothContactConstraint, input: ArrayLike<number>, output: Float64Array, twistOffset: number): void {
    const J = c.rod!.jacobian, rows = c.rod!.rows, n = c.normal, s = [0, 0, 0], sides = twistOffset >= 0 ? c.rod!.sides.filter(side => this.twistFree[side.segment] && (c.frictionCoefficient || side.twistRow)) : [];
    let sn = 0;
    for (let j = 0; j < c.vertices.length; j++) {
      const v = c.vertices[j]!; if (!this.free[v]) continue;
      const i0 = input[3*v]!, i1 = input[3*v + 1]!, i2 = input[3*v + 2]!;
      sn += rows[3*j]!*i0 + rows[3*j + 1]!*i1 + rows[3*j + 2]!*i2;
      for (let p = 0; p < 3; p++) s[p]! += J[9*j + 3*p]!*i0 + J[9*j + 3*p + 1]!*i1 + J[9*j + 3*p + 2]!*i2;
    }
    // The rolling slip of a twist change (rodSlip's turn), and its gap row.
    for (const side of sides) {const turn = side.sign*input[twistOffset + side.segment]!; sn += side.twistRow*input[twistOffset + side.segment]!; for (let p = 0; p < 3; p++) s[p]! += turn*side.tau[p]!;}
    const st = s[0]!*n[0] + s[1]!*n[1] + s[2]!*n[2], t = [0, 1, 2].map(p => c.frictionCoefficient*(s[p]! - st*n[p]!));
    for (const side of sides) output[twistOffset + side.segment]! += c.normalCoefficient*sn*side.twistRow + side.sign*(side.tau[0]*t[0]! + side.tau[1]*t[1]! + side.tau[2]*t[2]!);
    for (let j = 0; j < c.vertices.length; j++) {
      const v = c.vertices[j]!; if (!this.free[v]) continue;
      for (let a = 0; a < 3; a++) output[3*v + a]! += c.normalCoefficient*sn*rows[3*j + a]! + J[9*j + a]!*t[0]! + J[9*j + 3 + a]!*t[1]! + J[9*j + 6 + a]!*t[2]!;
    }
  }
  /** Adds each free vertex's diagonal 3x3 contact block to blocks (9n). */
  addDiagonalBlocks(blocks: Float64Array): void {
    for (const c of this.constraints) {
      if (!c.normalCoefficient && !c.frictionCoefficient) continue;
      if (c.rod) {
        const rows = c.rod.rows;
        for (let j = 0; j < c.vertices.length; j++) {
          const v = c.vertices[j]!;
          if (!this.free[v]) continue;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) blocks[9*v + 3*a + b]! += c.normalCoefficient*rows[3*j + a]!*rows[3*j + b]! + c.frictionCoefficient*this.rodTangentBlock(c, j, a, b);
        }
        continue;
      }
      for (let j = 0; j < c.vertices.length; j++) {
        const v = c.vertices[j]!, w2 = c.weights[j]!**2;
        if (!this.free[v]) continue;
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) blocks[9*v + 3*a + b]! += w2*(c.normalCoefficient*c.normal[a]!*c.normal[b]! + c.frictionCoefficient*((a === b ? 1 : 0) - c.normal[a]!*c.normal[b]!));
      }
    }
  }

  /** Multiplier update after an inner solve. Returns the largest KKT residual
   * |lambda_new - lambda|/rho = |min(g, lambda/rho)| and lagged-friction
   * residual mu |lambda_new - lambda_friction|/rho, in metres. */
  updateMultipliers(x: ArrayLike<number>): {kkt_m: number; friction_m: number} {
    let kkt = 0, friction = 0;
    for (const c of this.constraints) {
      const next = this.force(c, c.lambda/c.rho - this.gap(c, x));
      kkt = Math.max(kkt, Math.abs(next - c.lambda)/c.rho);
      if (c.friction > 0) friction = Math.max(friction, c.friction*Math.abs(next - c.frictionLambda)/c.rho);
      c.lambda = next;
    }
    return {kkt_m: kkt, friction_m: friction};
  }
  /** Multipliers of the pairs carrying force, to warm-start the next step. */
  multipliers(): Map<string, number> {return new Map(this.constraints.filter(c => c.lambda > 0).map(c => [c.key, c.lambda]));}
  lagFriction(): void {for (const c of this.constraints) c.frictionLambda = c.lambda;}
  /** Raise rho on constraints whose KKT residual |min(g, lambda/rho)| did not
   * fall fast enough: a penetration, or a force still pushing across a gap
   * (redundant pairs at one contact share their force non-uniquely, and a small
   * rho releases its share only by rho g per outer iteration). */
  stiffen(x: ArrayLike<number>, tolerance: number): void {
    for (const c of this.constraints) if (Math.abs(Math.min(this.gap(c, x), c.lambda/c.rho)) > tolerance) c.rho *= 10;
  }

  /** Report in the reference's shape: active contacts, the largest remaining
   * penetration, and the body reactions of the converged contact forces. */
  report(x: ArrayLike<number>, outerIterations: number, residual_m: number): ClothContactReport {
    let contacts = 0, selfContacts = 0, penetration = 0;
    const reactions = new Map<string, ClothReaction>();
    for (const c of this.constraints) {
      const g = this.gap(c, x);
      penetration = Math.max(penetration, -g, c.rod ? this.relinearizeRod(c, x, false) : c.kind === 'collider' ? c.radius - sampleClothCollider(this.colliders[c.colliderIndex]!, point3(x, c.vertices[0]!), 0).distance_m : 0);
      if (!(c.lambda > 0)) continue;
      if (c.kind === 'self') {selfContacts++; continue;}
      contacts++;
      const collider = this.colliders[c.colliderIndex]!;
      if (!collider.bodyId) continue;
      // Force on the cloth vertex from this constraint; the body takes -F dt.
      const slip = c.friction > 0 && c.frictionLambda > 0 ? (c.rod ? this.rodSlip(c, x) : this.slip(c, x)) : [0, 0, 0] as [number, number, number], f = c.friction > 0 && c.frictionLambda > 0 ? c.friction*c.frictionLambda*this.f1OverY(norm3(slip)) : 0;
      const normalForce = this.force(c, c.lambda/c.rho - g), force = [0, 1, 2].map(a => normalForce*c.normal[a]! - f*slip[a]!) as [number, number, number];
      // A rod's force acts at its material point on the surface, facing the collider.
      const at = c.rod ? add3(c.vertices.reduce<ClothVector>((p, v, j) => add3(p, scale3(point3(x, v), c.weights[j]!)), [0, 0, 0]), c.rod.sides[0]!.arm) : point3(x, c.vertices[0]!);
      const impulse = scale3(force, -this.dt), arm = sub3(at, collider.centreOfMass_m ?? [0, 0, 0]), torque = cross3(arm, impulse);
      let reaction = reactions.get(collider.bodyId);
      if (!reaction) {reaction = {bodyId: collider.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0]}; reactions.set(collider.bodyId, reaction);}
      for (let a = 0; a < 3; a++) {reaction.linearImpulse_N_s[a]! += impulse[a]!; reaction.angularImpulse_N_m_s[a]! += torque[a]!;}
    }
    return {iterations: outerIterations, residual_m, contacts, selfContacts, maximumPenetration_m: penetration, reactions: [...reactions.values()]};
  }
}
