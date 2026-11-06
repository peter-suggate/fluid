import {closestSegmentPoints, closestTrianglePoint, connectedRestFaces, restSegmentTriangleDistance, sampleClothCollider, type ClothCollider, type ClothContactReport, type ClothReaction} from './contact';
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
 * with n and w frozen where the CCD found the impact (self contact: vertex
 * minus face point; edge-edge: the two closest points), or, for colliders, the
 * collider's tangent plane re-linearized at the current end-of-step position
 * every outer iteration (sequential linearization of phi(x) - r >= 0).
 * Coulomb friction is the lagged smoothed form of Li et al. 2020 (IPC): with the
 * normal force lambda and tangent plane of the previous outer iteration,
 *   mu lambda f0(|P_t (sum_j w_kj (x_j - x_j^old) - d_obstacle)|),
 * f0 C1-smoothed below eps = FRICTION_SLIP_SPEED dt.
 * Outer iterations re-detect from the old to the current end positions and stop
 * when no new pair appears and, in metres, every KKT residual
 * |min(g, lambda/rho)| (penetration, or force acting across a gap), every true
 * collider penetration and every lagged-friction residual
 * mu |lambda - lambda_friction|/rho is below the solver tolerance. (The inner
 * solve fixes g only to that tolerance, so lambda is known to rho times it.) Velocity is then (x - x_old)/dt of these
 * positions, so nothing is projected after the solve.
 *
 * Rods (capsules, clipped caps, convex pairs) and dynamic rigid colliders (whose
 * velocity responds within the step) are not modelled here: such steps keep the
 * sequential post-projection, and inSolveContactUnsupported() says why. */
export interface ClothContactConstraint {
  readonly key: string;
  readonly kind: 'collider' | 'self';
  readonly vertices: readonly number[];
  readonly weights: readonly number[];
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
}

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
  if (topology.rods.length) return 'rod contact (capsules, clipped caps) keeps the sequential projection';
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
  /** delta: the penalty's smoothing width (the solver tolerance, metres).
   * warmStart: multipliers of the previous step's pairs, by key. */
  constructor(private readonly topology: PreparedClothTopology, private readonly old: Float64Array, private readonly dt: number, private readonly colliders: readonly ClothCollider[], private readonly selfCollision: boolean, readonly delta: number, private readonly warmStart?: ReadonlyMap<string, number>) {
    const unsupported = inSolveContactUnsupported(topology, colliders);
    if (unsupported) throw new Error(`In-solve cloth contact: ${unsupported}`);
    this.free = Uint8Array.from({length: topology.vertexCount}, (_, i) => (topology.kinematicFixed[i] ?? 0) & 1 ? 0 : 1);
    this.epsilon = FRICTION_SLIP_SPEED_M_S*dt;
  }

  private add(c: Omit<ClothContactConstraint, 'lambda' | 'frictionLambda' | 'rho' | 'normalCoefficient' | 'frictionCoefficient'>): boolean {
    if (this.byKey.has(c.key) || !c.vertices.some(v => this.free[v])) return false;
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
      if (!this.free[i] || topology.incidentTriangles[i]!.length === 0) continue;
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
        if (this.add({key, kind: 'collider', vertices: [i], weights: [1], normal: [normal[0], normal[1], normal[2]], offset: dot3(normal, plane) - gap, colliderIndex: ci, obstacle: scale3(colliderVelocity(collider, next), dt), friction: collider.friction ?? topology.materials[topology.vertexMaterial[i]!]!.friction, radius: r})) added++;
      }
    }
    if (this.selfCollision) added += this.detectSelf(x);
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

  /** Collider tangent planes re-linearized at x (end-of-step pose). Returns
   * the largest true collider penetration r - phi(x) found. */
  relinearize(x: ArrayLike<number>): number {
    let penetration = 0;
    for (const c of this.constraints) {
      if (c.kind !== 'collider') continue;
      const i = c.vertices[0]!, p = point3(x, i), collider = this.colliders[c.colliderIndex]!, sample = sampleClothCollider(collider, p, 0);
      if (!(dot3(sample.normal, c.normal) > 0)) throw new Error(`Cloth vertex ${i} crossed collider ${c.colliderIndex} within the step`);
      c.normal = [sample.normal[0], sample.normal[1], sample.normal[2]];
      c.offset = dot3(sample.normal, p) - (sample.distance_m - c.radius);
      c.obstacle = scale3(colliderVelocity(collider, p), this.dt);
      penetration = Math.max(penetration, c.radius - sample.distance_m);
    }
    return penetration;
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
      if (c.friction > 0 && c.frictionLambda > 0) energy += c.friction*c.frictionLambda*this.f0(norm3(this.slip(c, x)));
    }
    return energy;
  }
  /** Adds the gradient to `gradient` and the diagonal of the Hessian model to
   * `diagonal` (both 3n), and stores each constraint's Hessian coefficients. */
  accumulate(x: ArrayLike<number>, gradient: Float64Array, diagonal: Float64Array): void {
    for (const c of this.constraints) {
      const m = c.lambda/c.rho - this.gap(c, x), force = this.force(c, m);
      c.normalCoefficient = this.curvature(c, m);
      let slip: [number, number, number] = [0, 0, 0], f = 0;
      if (c.friction > 0 && c.frictionLambda > 0) {slip = this.slip(c, x); f = c.friction*c.frictionLambda*this.f1OverY(norm3(slip));}
      c.frictionCoefficient = f;
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
      const s = [0, 0, 0];
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) s[a]! += w*input[3*v + a]!;}
      const sn = s[0]!*c.normal[0] + s[1]!*c.normal[1] + s[2]!*c.normal[2];
      const t = [0, 1, 2].map(a => c.normalCoefficient*sn*c.normal[a]! + c.frictionCoefficient*(s[a]! - sn*c.normal[a]!));
      for (let j = 0; j < c.vertices.length; j++) {const v = c.vertices[j]!, w = c.weights[j]!; if (this.free[v]) for (let a = 0; a < 3; a++) output[3*v + a]! += w*t[a]!;}
    }
  }
  /** Adds each free vertex's diagonal 3x3 contact block to blocks (9n). */
  addDiagonalBlocks(blocks: Float64Array): void {
    for (const c of this.constraints) {
      if (!c.normalCoefficient && !c.frictionCoefficient) continue;
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
      penetration = Math.max(penetration, -g, c.kind === 'collider' ? c.radius - sampleClothCollider(this.colliders[c.colliderIndex]!, point3(x, c.vertices[0]!), 0).distance_m : 0);
      if (!(c.lambda > 0)) continue;
      if (c.kind === 'self') {selfContacts++; continue;}
      contacts++;
      const collider = this.colliders[c.colliderIndex]!;
      if (!collider.bodyId) continue;
      // Force on the cloth vertex from this constraint; the body takes -F dt.
      const slip = c.friction > 0 && c.frictionLambda > 0 ? this.slip(c, x) : [0, 0, 0] as [number, number, number], f = c.friction > 0 && c.frictionLambda > 0 ? c.friction*c.frictionLambda*this.f1OverY(norm3(slip)) : 0;
      const normalForce = this.force(c, c.lambda/c.rho - g), force = [0, 1, 2].map(a => normalForce*c.normal[a]! - f*slip[a]!) as [number, number, number];
      const impulse = scale3(force, -this.dt), arm = sub3(point3(x, c.vertices[0]!), collider.centreOfMass_m ?? [0, 0, 0]), torque = cross3(arm, impulse);
      let reaction = reactions.get(collider.bodyId);
      if (!reaction) {reaction = {bodyId: collider.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0]}; reactions.set(collider.bodyId, reaction);}
      for (let a = 0; a < 3; a++) {reaction.linearImpulse_N_s[a]! += impulse[a]!; reaction.angularImpulse_N_m_s[a]! += torque[a]!;}
    }
    return {iterations: outerIterations, residual_m, contacts, selfContacts, maximumPenetration_m: penetration, reactions: [...reactions.values()]};
  }
}
