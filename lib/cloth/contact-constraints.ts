import {closestSegmentPoints, closestTrianglePoint, connectedRestFaces, restSegmentTriangleDistance, sampleClothCollider, type ClothCollider, type ClothContactReport, type ClothReaction} from './contact';
import {sampleClippedRodCollider} from './clipped-contact';
import {ClothContactBVH} from './contact-bvh';
import {cross3, dot3, norm3, point3, scale3, sub3, type PreparedClothTopology} from './topology';
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
 * Clipped caps, elliptical sections and dynamic rigid colliders (whose
 * velocity responds within the step) are not modelled here: such steps keep the
 * sequential post-projection, and inSolveContactUnsupported() says why. */
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
interface RodContactSide {readonly segment: number; readonly a: number; readonly b: number; readonly sign: 1 | -1; readonly radius: number; tau: ClothVector}
type RodContactShape = 'point-collider' | 'segment-collider' | 'point-face' | 'segment-face' | 'point-segment' | 'segment-segment';
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
function rodSelfClosest(shape: RodContactShape, q: readonly ClothVector[]): {weights: number[]; delta: ClothVector; fallback: ClothVector; owned: boolean} {
  if (shape === 'point-face') {const c = closestTrianglePoint(q[0]!, q[1]!, q[2]!, q[3]!); return {weights: [1, -c.weights[0], -c.weights[1], -c.weights[2]], delta: sub3(q[0]!, c.point), fallback: cross3(sub3(q[2]!, q[1]!), sub3(q[3]!, q[1]!)), owned: true};}
  if (shape === 'segment-face') {const c = closestSegmentTriangle(q[0]!, q[1]!, q[2]!, q[3]!, q[4]!); return {weights: [1 - c.s, c.s, -c.weights[0], -c.weights[1], -c.weights[2]], delta: sub3(c.pa, c.pb), fallback: cross3(sub3(q[3]!, q[2]!), sub3(q[4]!, q[2]!)), owned: c.s > ROD_INTERIOR && c.s < 1 - ROD_INTERIOR && c.weights.every(w => w > ROD_INTERIOR)};}
  if (shape === 'point-segment') {const c = closestPointSegment(q[0]!, q[1]!, q[2]!); return {weights: [1, -(1 - c.t), -c.t], delta: sub3(q[0]!, c.point), fallback: perpendicular(sub3(q[2]!, q[1]!)), owned: true};}
  if (shape === 'segment-segment') {const c = closestSegmentPoints(q[0]!, q[1]!, q[2]!, q[3]!); return {weights: [1 - c.s, c.s, -(1 - c.t), -c.t], delta: sub3(c.a, c.b), fallback: cross3(sub3(q[1]!, q[0]!), sub3(q[3]!, q[2]!)), owned: [c.s, c.t].every(v => v > ROD_INTERIOR && v < 1 - ROD_INTERIOR)};}
  throw new Error(`Rod self contact shape ${shape} is not a self pair`);
}
const unit = (delta: ClothVector, fallback: ClothVector): ClothVector => {const l = norm3(delta), f = norm3(fallback); return l > 1e-12 ? scale3(delta, 1/l) : f > 1e-12 ? scale3(fallback, 1/f) : [0, 1, 0];};

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
  for (let r = 0; r < topology.rods.length; r++) {
    const m = topology.materials[topology.rods[r]!.material]!;
    if ((m.rodRadiusB_m ?? m.rodRadius_m) !== m.rodRadius_m) return 'elliptical rod contact keeps the sequential projection';
    for (const normals of [topology.data.rodStartClipNormals, topology.data.rodEndClipNormals]) if (normals && norm3(point3(normals, r)) > 0) return 'clipped rod end caps keep the sequential projection';
  }
  for (const c of colliders) if (c.kind === 'rigid' && !c.body.held && c.body.description.motion !== 'static') return 'a dynamic rigid collider responds within the step; it keeps the sequential projection';
  return undefined;
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
  private readonly rodSegments: {readonly rod: number; readonly a: number; readonly b: number; readonly r: number}[] = [];
  private readonly owner: Int32Array;
  private readonly rodNeighbours = new Map<number, number[]>();
  private readonly twistFree: Uint8Array;
  private rodBounds = new Float64Array(0);
  private rodTree?: ClothContactBVH;
  /** delta: the penalty's smoothing width (the solver tolerance, metres).
   * warmStart: multipliers of the previous step's pairs, by key.
   * rodTwist: the live and start-of-step material twist (rod friction rolls it). */
  constructor(private readonly topology: PreparedClothTopology, private readonly old: Float64Array, private readonly dt: number, private readonly colliders: readonly ClothCollider[], private readonly selfCollision: boolean, readonly delta: number, private readonly warmStart?: ReadonlyMap<string, number>, private readonly rodTwist?: {readonly current: ArrayLike<number>; readonly old: ArrayLike<number>}) {
    const unsupported = inSolveContactUnsupported(topology, colliders);
    if (unsupported) throw new Error(`In-solve cloth contact: ${unsupported}`);
    this.free = Uint8Array.from({length: topology.vertexCount}, (_, i) => (topology.kinematicFixed[i] ?? 0) & 1 ? 0 : 1);
    this.epsilon = FRICTION_SLIP_SPEED_M_S*dt;
    this.owner = new Int32Array(topology.vertexCount).fill(-1);
    const segments = topology.rods.reduce((total, rod) => total + rod.restLengths_m.length, 0);
    this.twistFree = new Uint8Array(segments);
    for (let r = 0; r < topology.rods.length; r++) {
      const rod = topology.rods[r]!, count = rod.restLengths_m.length, radius = topology.materials[rod.material]!.rodRadius_m;
      for (let j = 0; j < count; j++) {
        const a = rod.vertices[j]!, b = rod.vertices[j + 1]!, segment = rod.segmentOffset + j;
        this.rodSegments[segment] = {rod: r, a, b, r: radius};
        this.twistFree[segment] = ((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2 ? 0 : 1;
        for (const [v, w] of [[a, b], [b, a]] as const) {const list = this.rodNeighbours.get(v) ?? []; if (!list.includes(w)) list.push(w); this.rodNeighbours.set(v, list);}
      }
      for (let k = 0; k < rod.vertices.length; k++) if (this.owner[rod.vertices[k]!]! < 0) this.owner[rod.vertices[k]!] = rod.segmentOffset + Math.min(k, count - 1);
    }
    if (this.rodSegments.length !== segments || this.rodSegments.some(g => !g)) throw new Error('Cloth rod segments must cover their offsets contiguously');
  }
  /** A vertex that is a rod sphere (on a rod, not on the shell). */
  private rodVertex(i: number): boolean {return this.owner[i]! >= 0 && this.topology.incidentTriangles[i]!.length === 0;}
  private side(segment: number, sign: 1 | -1): RodContactSide {const g = this.rodSegments[segment]!; return {segment, a: g.a, b: g.b, sign, radius: g.r, tau: [0, 0, 0]};}
  /** Adds a rod pair: its rolling endpoints join the vertices at weight zero,
   * and its slip Jacobian is linearized at x. */
  private addRod(c: Omit<ClothContactConstraint, 'lambda' | 'frictionLambda' | 'rho' | 'normalCoefficient' | 'frictionCoefficient' | 'rod'>, shape: RodContactShape, first: number, second: number, sides: RodContactSide[], x: ArrayLike<number>): boolean {
    const vertices = [...c.vertices], weights = [...c.weights];
    for (const side of sides) for (const v of [side.a, side.b]) if (!vertices.includes(v)) {vertices.push(v); weights.push(0);}
    if (!this.add({...c, vertices, weights, rod: {shape, first, second, sides, jacobian: new Float64Array(9*vertices.length)}})) return false;
    this.linearizeRolling(this.byKey.get(c.key)!, x);
    return true;
  }
  /** J_j = w_j I + sum over rod sides of sign (+C at b, -C at a), where
   * C v = ((t x v) x arm)/L: the endpoint couple of rodContactDOF, with the
   * material point at arm = -sign radius n from the centreline; the twist
   * moves it by tau = t x arm per radian. n.C = 0, so the gap is unchanged. */
  private linearizeRolling(c: ClothContactConstraint, x: ArrayLike<number>): void {
    const g = c.rod!, J = g.jacobian;
    J.fill(0);
    for (let j = 0; j < c.vertices.length; j++) J[9*j] = J[9*j + 4] = J[9*j + 8] = c.weights[j]!;
    for (const side of g.sides) {
      const e = sub3(point3(x, side.b), point3(x, side.a)), length = norm3(e);
      if (!(length > 0)) throw new Error(`Cloth rod segment ${side.segment} has zero length in contact`);
      const t = scale3(e, 1/length), arm = scale3(c.normal, -side.sign*side.radius), ta = dot3(t, arm), ia = c.vertices.indexOf(side.a), ib = c.vertices.indexOf(side.b);
      side.tau = cross3(t, arm);
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {const C = side.sign*((p === q ? ta : 0) - t[p]!*arm[q]!)/length; J[9*ib + 3*p + q]! += C; J[9*ia + 3*p + q]! -= C;}
    }
  }

  private add(c: Omit<ClothContactConstraint, 'lambda' | 'frictionLambda' | 'rho' | 'normalCoefficient' | 'frictionCoefficient' | 'rod'> & {rod?: RodContactGeometry}): boolean {
    if (this.byKey.has(c.key) || !c.vertices.some((v, j) => this.free[v] && (!c.rod || c.weights[j] !== 0))) return false;
    const lambda = this.warmStart?.get(c.key) ?? 0;
    const constraint: ClothContactConstraint = {...c, lambda, frictionLambda: lambda, rho: 0, normalCoefficient: 0, frictionCoefficient: 0};
    this.byKey.set(c.key, constraint); this.constraints.push(constraint);
    return true;
  }

  /** The reference CCD from the old to the given end positions. Pairs already
   * held keep their frozen geometry and multiplier; returns new pairs. */
  detect(x: ArrayLike<number>): number {
    const {topology, old, dt, colliders} = this;
    let added = 0;
    for (let i = 0; i < topology.vertexCount; i++) {
      const rodVertex = this.rodVertex(i);
      if (!this.free[i] || (topology.incidentTriangles[i]!.length === 0 && !rodVertex)) continue;
      const start = point3(old, i), next = point3(x, i), delta = sub3(next, start), r = vertexRadius(topology, i);
      for (let ci = 0; ci < colliders.length; ci++) {
        const collider = colliders[ci]!, key = `c:${i}:${ci}`;
        if (this.byKey.has(key)) continue;
        const rotationSpeed = collider.kind === 'rigid' ? Math.hypot(collider.body.angularVelocity_rad_s.x, collider.body.angularVelocity_rad_s.y, collider.body.angularVelocity_rad_s.z)*sceneShape(collider.body.description.shape).boundingRadius_m(collider.body.description.dimensions_m) : 0;
        const speed = (norm3(delta) + dt*(collider.kind === 'sampled' ? collider.maximumSpeed_m_s : norm3(collider.velocity_m_s ?? [0, 0, 0]) + rotationSpeed))*(collider.kind === 'sampled' ? (collider.spatialLipschitz ?? 2) : 1);
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
    added += this.detectRodColliders(x);
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
      if (!this.free[a] && !this.free[b]) continue;
      const oa = point3(old, a), ob = point3(old, b), na = point3(x, a), nb = point3(x, b), reach = Math.max(norm3(sub3(na, oa)), norm3(sub3(nb, ob)));
      const shape = (time: number) => ({a: lerp3(oa, na, time), b: lerp3(ob, nb, time), radius_m: r, planes: []});
      for (let ci = 0; ci < colliders.length; ci++) {
        const collider = colliders[ci]!, key = `rc:${segment}:${ci}`;
        if (this.byKey.has(key)) continue;
        const rotationSpeed = collider.kind === 'rigid' ? Math.hypot(collider.body.angularVelocity_rad_s.x, collider.body.angularVelocity_rad_s.y, collider.body.angularVelocity_rad_s.z)*sceneShape(collider.body.description.shape).boundingRadius_m(collider.body.description.dimensions_m) : 0;
        const travel = (reach + dt*(collider.kind === 'sampled' ? collider.maximumSpeed_m_s : norm3(collider.velocity_m_s ?? [0, 0, 0]) + rotationSpeed))*(collider.kind === 'sampled' ? (collider.spatialLipschitz ?? 2) : 1);
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
      const {a, b, r} = segments[s]!;
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
    const pair = (key: string, shape: RodContactShape, ids: number[], first: number, thickness: number, radius: number, sides: RodContactSide[]) => {
      if (this.byKey.has(key)) return;
      const hit = this.sweepRodPair(shape, ids, first, thickness, x);
      if (hit && this.addRod({key, kind: 'self', vertices: ids, weights: hit.weights, normal: [hit.normal[0], hit.normal[1], hit.normal[2]], offset: thickness, colliderIndex: -1, obstacle: [0, 0, 0], friction: friction(ids), radius}, shape, first, ids.length - first, sides, x)) added++;
    };
    for (let i = 0; i < topology.vertexCount; i++) {
      const rodVertex = this.rodVertex(i), ri = vertexRadius(topology, i), [min, max] = box([i], ri);
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
        pair(`rp:${i}:${s}`, 'point-segment', [i, g.a, g.b], 1, ri + g.r, ri, rodVertex ? [this.side(this.owner[i]!, 1), this.side(s, -1)] : [this.side(s, -1)]);
      }
    }
    for (let s = 0; s < segments.length; s++) {
      const g = segments[s]!, ids = [g.a, g.b], min = this.rodBounds.subarray(6*s, 6*s + 3), max = this.rodBounds.subarray(6*s + 3, 6*s + 6);
      if (faceTree) for (const f of faceTree.query([...min], [...max])) {const face = topology.triangles[f]!; if (!touches(ids, face.vertices)) pair(`rf:${s}:${f}`, 'segment-face', [...ids, ...face.vertices], 2, g.r + .5*topology.materials[face.material]!.thickness_m, g.r, [this.side(s, 1)]);}
      if (edgeTree) for (const e of edgeTree.query([...min], [...max])) {const edge = cache.edges[e]!; if (!touches(ids, edge.vertices)) pair(`rx:${s}:${e}`, 'segment-segment', [...ids, ...edge.vertices], 2, g.r + edge.r, g.r, [this.side(s, 1)]);}
      for (const u of rodTree.query([...min], [...max])) {
        if (u <= s) continue;
        const h = segments[u]!;
        if (touches(ids, [h.a, h.b])) continue;
        if (g.rod === h.rod && restGap('segment-segment', [...ids, h.a, h.b]) <= g.r + h.r + 1e-12) continue;
        pair(`rs:${s}:${u}`, 'segment-segment', [...ids, h.a, h.b], 2, g.r + h.r, g.r, [this.side(s, 1), this.side(u, -1)]);
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
      if (!(dot3(sample.normal, c.normal) > 0)) throw new Error(`Cloth vertex ${i} crossed collider ${c.colliderIndex} within the step`);
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
    if (g.shape === 'point-collider' || g.shape === 'segment-collider') {
      const collider = this.colliders[c.colliderIndex]!;
      const sample = g.shape === 'point-collider' ? (() => {const s = sampleClothCollider(collider, q[0]!, 0); return {distance_m: s.distance_m - c.radius, normal: s.normal, point: sub3(q[0]!, scale3(s.normal, c.radius))};})() : sampleClippedRodCollider({a: q[0]!, b: q[1]!, radius_m: c.radius, planes: []}, collider, 0, sampleClothCollider);
      if (!apply) return -sample.distance_m;
      if (!(dot3(sample.normal, c.normal) > 0)) throw new Error(`Cloth rod contact ${c.key} crossed collider ${c.colliderIndex} within the step`);
      if (g.shape === 'segment-collider') {const t = closestPointSegment(add3(sample.point, scale3(sample.normal, c.radius)), q[0]!, q[1]!).t; c.weights[0] = 1 - t; c.weights[1] = t;}
      const centre = lerp3(q[0]!, q[g.shape === 'segment-collider' ? 1 : 0]!, g.shape === 'segment-collider' ? c.weights[1]! : 0);
      c.normal = [sample.normal[0], sample.normal[1], sample.normal[2]];
      c.offset = dot3(sample.normal, centre) - sample.distance_m;
      c.obstacle = scale3(colliderVelocity(collider, sample.point), this.dt);
      this.linearizeRolling(c, x);
      return -sample.distance_m;
    }
    const closest = rodSelfClosest(g.shape, q);
    if (!apply) return c.offset - norm3(closest.delta);
    let normal = unit(closest.delta, closest.fallback);
    if (norm3(closest.delta) <= 1e-12 && norm3(closest.fallback) <= 1e-12) normal = c.normal;
    if (dot3(normal, c.normal) < 0) normal = scale3(normal, -1);
    for (let j = 0; j < closest.weights.length; j++) c.weights[j] = closest.weights[j]!;
    c.normal = [normal[0], normal[1], normal[2]];
    this.linearizeRolling(c, x);
    return c.offset - dot3(normal, closest.delta);
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
        const d = c.normal[0]**2*stiffness[3*v]! + c.normal[1]**2*stiffness[3*v+1]! + c.normal[2]**2*stiffness[3*v+2]!;
        compliance += c.weights[j]!**2/d;
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
        for (let j = 0; j < c.vertices.length; j++) {
          const v = c.vertices[j]!, w = c.weights[j]!;
          if (!this.free[v]) continue;
          for (let a = 0; a < 3; a++) {
            gradient[3*v + a]! += -force*w*c.normal[a]! + f*(J[9*j + a]!*slip[0] + J[9*j + 3 + a]!*slip[1] + J[9*j + 6 + a]!*slip[2]);
            diagonal[3*v + a]! += c.normalCoefficient*w*w*c.normal[a]!**2 + f*this.rodTangentBlock(c, j, a, a);
          }
        }
        if (f) for (const side of c.rod.sides) {
          if (!this.twistFree[side.segment]) continue;
          if (!twistGradient || !twistDiagonal) throw new Error('In-solve rod contact friction needs the twist gradient');
          const tn = dot3(side.tau, c.normal);
          twistGradient[side.segment]! += f*side.sign*dot3(side.tau, slip);
          twistDiagonal[side.segment]! += f*(dot3(side.tau, side.tau) - tn*tn);
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
  /** output += (sum_k rho_k a_k a_k^T + friction tangent terms) input. */
  multiply(input: ArrayLike<number>, output: Float64Array): void {
    for (const c of this.constraints) {
      if (!c.normalCoefficient && !c.frictionCoefficient) continue;
      if (c.rod) {this.multiplyRod(c, input, output); continue;}
      const s = [0, 0, 0];
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) s[a]! += w*input[3*v + a]!;}
      const sn = s[0]!*c.normal[0] + s[1]!*c.normal[1] + s[2]!*c.normal[2];
      const t = [0, 1, 2].map(a => c.normalCoefficient*sn*c.normal[a]! + c.frictionCoefficient*(s[a]! - sn*c.normal[a]!));
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) output[3*v + a]! += w*t[a]!;}
    }
  }
  /** Rod pair: normal rows by the weights, friction rows by the slip Jacobian. */
  private multiplyRod(c: ClothContactConstraint, input: ArrayLike<number>, output: Float64Array): void {
    const J = c.rod!.jacobian, n = c.normal, s = [0, 0, 0];
    let sn = 0;
    for (let j = 0; j < c.vertices.length; j++) {
      const v = c.vertices[j]!; if (!this.free[v]) continue;
      const i0 = input[3*v]!, i1 = input[3*v + 1]!, i2 = input[3*v + 2]!;
      sn += c.weights[j]!*(n[0]*i0 + n[1]*i1 + n[2]*i2);
      for (let p = 0; p < 3; p++) s[p]! += J[9*j + 3*p]!*i0 + J[9*j + 3*p + 1]!*i1 + J[9*j + 3*p + 2]!*i2;
    }
    const st = s[0]!*n[0] + s[1]!*n[1] + s[2]!*n[2], t = [0, 1, 2].map(p => c.frictionCoefficient*(s[p]! - st*n[p]!));
    for (let j = 0; j < c.vertices.length; j++) {
      const v = c.vertices[j]!; if (!this.free[v]) continue;
      for (let a = 0; a < 3; a++) output[3*v + a]! += c.weights[j]!*c.normalCoefficient*sn*n[a]! + J[9*j + a]!*t[0]! + J[9*j + 3 + a]!*t[1]! + J[9*j + 6 + a]!*t[2]!;
    }
  }
  /** Adds each free vertex's diagonal 3x3 contact block to blocks (9n). */
  addDiagonalBlocks(blocks: Float64Array): void {
    for (const c of this.constraints) {
      if (!c.normalCoefficient && !c.frictionCoefficient) continue;
      if (c.rod) {
        for (let j = 0; j < c.vertices.length; j++) {
          const v = c.vertices[j]!, w2 = c.weights[j]!**2;
          if (!this.free[v]) continue;
          for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) blocks[9*v + 3*a + b]! += w2*c.normalCoefficient*c.normal[a]!*c.normal[b]! + c.frictionCoefficient*this.rodTangentBlock(c, j, a, b);
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
      const at = c.rod ? sub3(c.vertices.reduce<ClothVector>((p, v, j) => add3(p, scale3(point3(x, v), c.weights[j]!)), [0, 0, 0]), scale3(c.normal, c.radius)) : point3(x, c.vertices[0]!);
      const impulse = scale3(force, -this.dt), arm = sub3(at, collider.centreOfMass_m ?? [0, 0, 0]), torque = cross3(arm, impulse);
      let reaction = reactions.get(collider.bodyId);
      if (!reaction) {reaction = {bodyId: collider.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0]}; reactions.set(collider.bodyId, reaction);}
      for (let a = 0; a < 3; a++) {reaction.linearImpulse_N_s[a]! += impulse[a]!; reaction.angularImpulse_N_m_s[a]! += torque[a]!;}
    }
    return {iterations: outerIterations, residual_m, contacts, selfContacts, maximumPenetration_m: penetration, reactions: [...reactions.values()]};
  }
}
