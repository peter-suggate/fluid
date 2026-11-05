import {cohesionPairValue, type ClothCohesionPair} from './cohesion';
import {bendingPhi, finishClothMechanicalStep, hingeCoordinateDerivatives, hingeEnergy, membraneCoordinateDerivatives, membraneEnergy, rodEnergy, rodStretchCoordinateDerivatives, triangleStrain, type ClothStepOptions, type ClothStepReport} from './dynamics';
import {bendingRigidity, membraneModulus, rodRigidity, rodSegmentInertia_kg_m2} from './material';
import {hingeAngle, type PreparedClothTopology} from './topology';
import type {ClothState} from './types';

/** Global Newton solve of the same implicit-Euler incremental potential that
 * stepCloth minimizes coordinate by coordinate (lib/cloth/dynamics.ts):
 *   inertia + finite attachments + cohesion + membrane + hinge + rod energies
 *   (with their Kelvin-Voigt terms) + twist inertia and attachment.
 * The gradient and the convergence measure are the coordinate solver's own:
 * analytic membrane, hinge and rod-stretch derivatives, central differences
 * for rod bending and twist, and max |g_k / max(m/dt^2, d2E/dx_k^2)| (twist
 * scaled by rod radius) against the same tolerance.
 *
 * Each Newton iteration assembles a positive-semidefinite model of the
 * Hessian (exact membrane material stiffness, membrane geometric stiffness
 * with its 2x2 stress projected onto its nonnegative eigenvalues, Gauss-Newton
 * hinge and rod-stretch stiffness, exact cohesion stiffness, central-difference
 * rod bending diagonal), solves it with 3x3 block-Jacobi preconditioned CG,
 * solves each rod's twist through its exact tridiagonal Hessian, and takes an
 * Armijo backtracking step on the energy. The Hessian model only shapes the
 * step: acceptance is decided by the energy, convergence by the true gradient.
 * Not modelled (each costs Newton iterations, not correctness): twist-position
 * coupling and rod bending between neighbouring vertices. */
export interface ClothNewtonReport extends ClothStepReport {
  readonly newtonIterations: number;
  readonly linearIterations: number;
}

interface BlockPattern {
  readonly rowStart: Int32Array;
  readonly columns: Int32Array;
  readonly diagonal: Int32Array;
}

const patternCache = new WeakMap<PreparedClothTopology, BlockPattern>();

function blockPattern(topology: PreparedClothTopology, cohesion: readonly ClothCohesionPair[]): BlockPattern {
  const cached = cohesion.length ? undefined : patternCache.get(topology);
  if (cached) return cached;
  const n = topology.vertexCount, neighbors = Array.from({length: n}, (_, i) => new Set<number>([i]));
  const connect = (vertices: readonly number[]) => {for (const a of vertices) for (const b of vertices) neighbors[a]!.add(b);};
  for (const face of topology.triangles) connect(face.vertices);
  for (const hinge of topology.hinges) connect(hinge.vertices);
  for (const rod of topology.rods) for (let j = 0; j + 1 < rod.vertices.length; j++) connect([rod.vertices[j]!, rod.vertices[j + 1]!]);
  for (const pair of cohesion) connect(pair.vertices);
  const rowStart = new Int32Array(n + 1), sorted = neighbors.map(set => [...set].sort((a, b) => a - b));
  for (let i = 0; i < n; i++) rowStart[i + 1] = rowStart[i]! + sorted[i]!.length;
  const columns = new Int32Array(rowStart[n]!), diagonal = new Int32Array(n);
  for (let i = 0; i < n; i++) {columns.set(sorted[i]!, rowStart[i]!); diagonal[i] = rowStart[i]! + sorted[i]!.indexOf(i);}
  const pattern = {rowStart, columns, diagonal};
  if (!cohesion.length) patternCache.set(topology, pattern);
  return pattern;
}

function blockIndex(pattern: BlockPattern, row: number, column: number): number {
  for (let b = pattern.rowStart[row]!, end = pattern.rowStart[row + 1]!; b < end; b++) if (pattern.columns[b] === column) return b;
  throw new Error(`Cloth Newton Hessian has no block (${row}, ${column})`);
}

/** Gradient of the hinge's dihedral angle with respect to its 12 coordinates,
 * the same chain rule as hingeCoordinateGradients without the energy factor. */
function dihedralGradient(P: ArrayLike<number>, vertices: readonly [number, number, number, number], out: Float64Array): number {
  const [ia, ib, ic, id] = vertices;
  const e0 = P[3*ib]! - P[3*ia]!, e1 = P[3*ib+1]! - P[3*ia+1]!, e2 = P[3*ib+2]! - P[3*ia+2]!;
  const u0 = P[3*ic]! - P[3*ia]!, u1 = P[3*ic+1]! - P[3*ia+1]!, u2 = P[3*ic+2]! - P[3*ia+2]!;
  const v0 = P[3*id]! - P[3*ia]!, v1 = P[3*id+1]! - P[3*ia+1]!, v2 = P[3*id+2]! - P[3*ia+2]!;
  const n0 = e1*u2 - e2*u1, n1 = e2*u0 - e0*u2, n2 = e0*u1 - e1*u0, m0 = v1*e2 - v2*e1, m1 = v2*e0 - v0*e2, m2 = v0*e1 - v1*e0;
  const l = Math.hypot(e0, e1, e2), il = 1/l, ll = l*l, t0 = e0*il, t1 = e1*il, t2 = e2*il;
  const p0 = n1*m2 - n2*m1, p1 = n2*m0 - n0*m2, p2 = n0*m1 - n1*m0, x = n0*m0 + n1*m1 + n2*m2, y = t0*p0 + t1*p1 + t2*p2;
  const denominator = x*x + y*y;
  for (let local = 0; local < 4; local++) {
    const sE = local === 0 ? -1 : local === 1 ? 1 : 0, sU = local === 0 ? -1 : local === 2 ? 1 : 0, sV = local === 0 ? -1 : local === 3 ? 1 : 0;
    for (let axis = 0; axis < 3; axis++) {
      const k0 = axis === 0 ? 1 : 0, k1 = axis === 1 ? 1 : 0, k2 = axis === 2 ? 1 : 0;
      const de0 = k0*sE, de1 = k1*sE, de2 = k2*sE, du0 = k0*sU, du1 = k1*sU, du2 = k2*sU, dv0 = k0*sV, dv1 = k1*sV, dv2 = k2*sV;
      const dn0 = (de1*u2 - de2*u1) + (e1*du2 - e2*du1), dn1 = (de2*u0 - de0*u2) + (e2*du0 - e0*du2), dn2 = (de0*u1 - de1*u0) + (e0*du1 - e1*du0);
      const dm0 = (dv1*e2 - dv2*e1) + (v1*de2 - v2*de1), dm1 = (dv2*e0 - dv0*e2) + (v2*de0 - v0*de2), dm2 = (dv0*e1 - dv1*e0) + (v0*de1 - v1*de0);
      const dl = (e0*de0 + e1*de1 + e2*de2) / l, s = dl / ll;
      const td0 = de0*il - e0*s, td1 = de1*il - e1*s, td2 = de2*il - e2*s;
      const q0 = (dn1*m2 - dn2*m1) + (n1*dm2 - n2*dm1), q1 = (dn2*m0 - dn0*m2) + (n2*dm0 - n0*dm2), q2 = (dn0*m1 - dn1*m0) + (n0*dm1 - n1*dm0);
      const dx = (dn0*m0 + dn1*m1 + dn2*m2) + (n0*dm0 + n1*dm1 + n2*dm2), dy = (td0*p0 + td1*p1 + td2*p2) + (t0*q0 + t1*q1 + t2*q2);
      out[3*local + axis] = (x*dy - y*dx) / denominator;
    }
  }
  return Math.atan2(y, x);
}

function invert3(m: Float64Array, offset: number, out: Float64Array, target: number): boolean {
  const a = m[offset]!, b = m[offset+1]!, c = m[offset+2]!, d = m[offset+3]!, e = m[offset+4]!, f = m[offset+5]!, g = m[offset+6]!, h = m[offset+7]!, k = m[offset+8]!;
  const A = e*k - f*h, B = f*g - d*k, C = d*h - e*g, det = a*A + b*B + c*C;
  if (!(Math.abs(det) > 0 && Number.isFinite(det))) return false;
  const s = 1/det;
  out[target] = A*s; out[target+1] = (c*h - b*k)*s; out[target+2] = (b*f - c*e)*s;
  out[target+3] = B*s; out[target+4] = (a*k - c*g)*s; out[target+5] = (c*d - a*f)*s;
  out[target+6] = C*s; out[target+7] = (b*g - a*h)*s; out[target+8] = (a*e - b*d)*s;
  return true;
}

/** Same setup, energy, tolerance and acceptance as stepCloth; a global solve
 * instead of coordinate sweeps. options.iterations caps Newton iterations. */
export function solveClothIncrement(topology: PreparedClothTopology, state: ClothState, dt: number, options: ClothStepOptions = {}, initialGuess?: {readonly positions_m: ArrayLike<number>; readonly twist_rad: ArrayLike<number>}): ClothNewtonReport {
  if (!(Number.isFinite(dt) && dt > 0)) throw new Error('Cloth timestep must be positive and finite');
  const n = topology.vertexCount, x = state.positions_m, old = x.slice(), oldTwist = state.twist_rad.slice(), predicted = x.slice(), mass = new Float64Array(n);
  const gravity = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
  for (let i = 0; i < n; i++) {
    mass[i] = state.dryMass_kg[i]! + topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3 * state.waterVolume_m3[i]!;
    if (!(mass[i]! > 0 && Number.isFinite(mass[i]))) throw new Error(`Invalid cloth vertex mass ${i}`);
    for (let axis = 0; axis < 3; axis++) {
      const k = 3*i + axis;
      predicted[k] = old[k]! + dt*state.velocities_m_s[k]! + dt*dt*(gravity[axis]! + (options.externalForces_N?.[k] ?? 0)/mass[i]!);
      x[k] = (topology.kinematicFixed[i] ?? 0) & 1 ? target[k]! : initialGuess?.positions_m[k] ?? predicted[k]!;
    }
  }
  const twist = state.twist_rad, S = twist.length, twistFree = new Uint8Array(S), twistInertia = new Float64Array(S), twistRadius = new Float64Array(S), twistRod = new Int32Array(S), twistLocal = new Int32Array(S);
  for (let r = 0; r < topology.rods.length; r++) {
    const rod = topology.rods[r]!, m = topology.materials[rod.material]!;
    for (let j = 0; j < rod.restLengths_m.length; j++) {
      const s = rod.segmentOffset + j, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
      twistRod[s] = r; twistLocal[s] = j; twistRadius[s] = Math.max(m.rodRadius_m, m.rodRadiusB_m ?? m.rodRadius_m);
      twistFree[s] = ((topology.kinematicFixed[a] ?? 0) & 2) || ((topology.kinematicFixed[b] ?? 0) & 2) ? 0 : 1;
      twistInertia[s] = rodSegmentInertia_kg_m2(topology, state, r, j);
    }
  }
  for (let s = 0; s < S; s++) twist[s] = twistFree[s] ? initialGuess?.twist_rad[s] ?? oldTwist[s]! + dt*state.angularVelocity_rad_s[s]! : oldTwist[s]!;
  const maximumNewton = options.iterations ?? 12, tolerance = options.tolerance_m ?? 1e-9;
  if (!Number.isInteger(maximumNewton) || maximumNewton < 1) throw new Error('Cloth iterations must be a positive integer');
  const cohesive = options.cohesionPairs ?? [], incidentCohesion: number[][] = Array.from({length: n}, () => []);
  for (let j = 0; j < cohesive.length; j++) {const pair = cohesive[j]!; if (pair.vertices.length !== pair.weights.length || !Number.isFinite(pair.force_N) || pair.force_N < 0 || !pair.weights.every(Number.isFinite) || Math.abs(pair.weights.reduce((a, b) => a + b, 0) - (pair.targetPosition_m ? 1 : 0)) > 1e-10 || (pair.targetPosition_m && !pair.targetPosition_m.every(Number.isFinite)) || (pair.bodyId && (!pair.targetPosition_m || !pair.bodyCenter_m?.every(Number.isFinite)))) throw new Error('Invalid cloth cohesion pair'); for (const i of new Set(pair.vertices)) {if (!Number.isSafeInteger(i) || i < 0 || i >= n) throw new Error('Invalid cohesion vertex'); incidentCohesion[i]!.push(j);}}

  const free = Uint8Array.from({length: n}, (_, i) => (topology.kinematicFixed[i] ?? 0) & 1 ? 0 : 1);
  const bendingVertex = Uint8Array.from({length: n}, (_, i) => topology.incidentRods[i]!.some(r => topology.rods[r]!.vertices.length > 2) ? 1 : 0);
  const inertia = Float64Array.from(mass, m => m/(dt*dt));
  const twistEnergy = (s: number) => {
    const r = twistRod[s]!, j = twistLocal[s]!;
    return .5*topology.attachmentTwist_N_m[s]!*(twist[s]! - state.attachmentTwistTarget_rad[s]!)**2 + rodEnergy(topology, state, r, undefined, old, oldTwist, dt, j) + .5*twistInertia[s]!/(dt*dt)*(twist[s]! - oldTwist[s]! - dt*state.angularVelocity_rad_s[s]!)**2;
  };
  const localRodBending = (i: number) => topology.incidentRods[i]!.reduce((sum, r) => sum + rodEnergy(topology, state, r, i, old, oldTwist, dt, undefined, true), 0);
  const totalEnergy = () => {
    let energy = 0;
    for (let i = 0; i < n; i++) for (let a = 0; a < 3; a++) {const k = 3*i + a; energy += .5*inertia[i]!*(x[k]! - predicted[k]!)**2 + .5*topology.attachmentTranslation_N_m[i]!*(x[k]! - target[k]!)**2;}
    for (const pair of cohesive) energy += cohesionPairValue(pair, x).energy_J;
    for (let f = 0; f < topology.triangles.length; f++) energy += membraneEnergy(topology, state, f, old, dt);
    for (let h = 0; h < topology.hinges.length; h++) energy += hingeEnergy(topology, state, h, old, dt);
    for (let r = 0; r < topology.rods.length; r++) energy += rodEnergy(topology, state, r, undefined, old, oldTwist, dt);
    for (let s = 0; s < S; s++) if (twistFree[s]) energy += .5*topology.attachmentTwist_N_m[s]!*(twist[s]! - state.attachmentTwistTarget_rad[s]!)**2 + .5*twistInertia[s]!/(dt*dt)*(twist[s]! - oldTwist[s]! - dt*state.angularVelocity_rad_s[s]!)**2;
    return energy;
  };

  // Gradient, the coordinate solver's own diagonal, and the bending diagonal
  // the Hessian model needs (rod bending has no assembled block).
  const gradient = new Float64Array(3*n), rodDiagonal = new Float64Array(3*n), twistGradient = new Float64Array(S), twistDiagonal = new Float64Array(S), twistSecond = new Float64Array(S);
  const evaluate = (): number => {
    let measure = 0;
    for (let i = 0; i < n; i++) {
      if (!free[i]) {gradient[3*i] = gradient[3*i+1] = gradient[3*i+2] = 0; continue;}
      const h = Math.max(1e-9, Math.cbrt(state.restControlVolume_m3[i]!)*1e-4);
      for (let axis = 0; axis < 3; axis++) {
        const k = 3*i + axis, value = x[k]!;
        let g = inertia[i]!*(value - predicted[k]!) + topology.attachmentTranslation_N_m[i]!*(value - target[k]!), stiffness = inertia[i]! + topology.attachmentTranslation_N_m[i]!;
        for (const f of topology.incidentTriangles[i]!) {const d = membraneCoordinateDerivatives(topology, state, f, i, axis, old, dt); g += d.gradient; stiffness += d.diagonal;}
        for (const hinge of topology.incidentHinges[i]!) {const d = hingeCoordinateDerivatives(topology, state, hinge, i, axis, old, dt); g += d.gradient; stiffness += d.diagonal;}
        for (const r of topology.incidentRods[i]!) {const d = rodStretchCoordinateDerivatives(topology, state, r, i, axis, old, dt); g += d.gradient; stiffness += d.diagonal;}
        rodDiagonal[k] = 0;
        if (bendingVertex[i]) {
          const e = localRodBending(i); x[k] = value + h; const plus = localRodBending(i); x[k] = value - h; const minus = localRodBending(i); x[k] = value;
          const second = (plus + minus - 2*e)/(h*h);
          g += (plus - minus)/(2*h); stiffness += second; rodDiagonal[k] = Math.max(0, second);
        }
        for (const j of incidentCohesion[i]!) {const d = cohesionPairValue(cohesive[j]!, x, i, axis); g += d.gradient_N; stiffness += d.diagonal_N_m;}
        if (!Number.isFinite(g)) throw new Error(`Cloth topology degenerated at vertex ${i}`);
        gradient[k] = g;
        measure = Math.max(measure, Math.abs(g/Math.max(inertia[i]!, stiffness)));
      }
    }
    for (let s = 0; s < S; s++) {
      if (!twistFree[s]) {twistGradient[s] = 0; twistDiagonal[s] = 1; continue;}
      const theta = twist[s]!, h = 1e-5, e = twistEnergy(s); twist[s] = theta + h; const ep = twistEnergy(s); twist[s] = theta - h; const em = twistEnergy(s); twist[s] = theta;
      const g = (ep - em)/(2*h), diagonal = Math.max(twistInertia[s]!/(dt*dt), (ep + em - 2*e)/(h*h));
      if (!Number.isFinite(g) || !Number.isFinite(diagonal)) throw new Error(`Cloth rod twist degenerated at segment ${s}`);
      twistGradient[s] = g; twistDiagonal[s] = diagonal; twistSecond[s] = (ep + em - 2*e)/(h*h);
      measure = Math.max(measure, Math.abs(g/diagonal)*twistRadius[s]!);
    }
    return measure;
  };

  const pattern = blockPattern(topology, cohesive), values = new Float64Array(9*pattern.columns.length), preconditioner = new Float64Array(9*n);
  const addBlock = (i: number, j: number, block: ArrayLike<number>, scale = 1) => {const b = 9*blockIndex(pattern, i, j); for (let q = 0; q < 9; q++) values[b + q]! += scale*block[q]!;};
  const block = new Float64Array(9), angleGradient = new Float64Array(12);
  // The positive-semidefinite Hessian model. (The true Hessian, indefinite
  // where the sheet is compressed or folded, was measured as a first attempt
  // per iteration with this as fallback: no fewer iterations, slower steps.)
  const assemble = () => {
    values.fill(0);
    for (let i = 0; i < n; i++) {const b = 9*pattern.diagonal[i]!, d = inertia[i]! + topology.attachmentTranslation_N_m[i]!; values[b] += d + rodDiagonal[3*i]!; values[b + 4] += d + rodDiagonal[3*i+1]!; values[b + 8] += d + rodDiagonal[3*i+2]!;}
    for (const face of topology.triangles) {
      const m = topology.materials[face.material]!, nu = m.poissonRatio, gamma = (m.membraneDamping_s ?? m.mechanicalDamping_s)/dt;
      const e = triangleStrain(face, x), e0 = triangleStrain(face, old), fraction = face.vertices.reduce((sum, i) => sum + state.solidFraction[i]!, 0)/3, factor = face.restArea_m2*membraneModulus(m, fraction);
      const u = [0, 0, 0], v = [0, 0, 0];
      for (let j = 0; j < 3; j++) for (let a = 0; a < 3; a++) {u[a]! += x[3*face.vertices[j]! + a]!*face.ru[j]!; v[a]! += x[3*face.vertices[j]! + a]!*face.rv[j]!;}
      const ex = e[0] + gamma*(e[0] - e0[0]), ey = e[1] + gamma*(e[1] - e0[1]), es = e[2] + gamma*(e[2] - e0[2]);
      // Stress tensor (sx ss; ss sy), projected onto its nonnegative eigenvalues.
      let sx = ex + nu*ey, sy = ey + nu*ex, ss = .5*(1 - nu)*es;
      {
        const mean = .5*(sx + sy), radius = Math.hypot(.5*(sx - sy), ss), l1 = mean + radius, l2 = mean - radius;
        if (l2 < 0) {
          if (l1 <= 0) sx = sy = ss = 0;
          else {const cx = radius > 0 ? .5*(sx - sy)/radius : 1, cs = radius > 0 ? ss/radius : 0, c2 = .5*(1 + cx), s2 = .5*(1 - cx), sc = .5*cs; sx = l1*c2; sy = l1*s2; ss = l1*sc;}
        }
      }
      const material = factor*(1 + gamma), G = [0, 0, 0].map(() => new Float64Array(9));
      for (let j = 0; j < 3; j++) for (let a = 0; a < 3; a++) {G[j]![a] = face.ru[j]!*u[a]!; G[j]![3 + a] = face.rv[j]!*v[a]!; G[j]![6 + a] = face.rv[j]!*u[a]! + face.ru[j]!*v[a]!;}
      for (let j = 0; j < 3; j++) for (let l = 0; l < 3; l++) {
        const Gj = G[j]!, Gl = G[l]!, geometric = factor*(sx*face.ru[j]!*face.ru[l]! + sy*face.rv[j]!*face.rv[l]! + ss*(face.ru[j]!*face.rv[l]! + face.rv[j]!*face.ru[l]!));
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) {
          const p0 = Gj[a]!, p1 = Gj[3 + a]!, p2 = Gj[6 + a]!, q0 = Gl[b]!, q1 = Gl[3 + b]!, q2 = Gl[6 + b]!;
          block[3*a + b] = material*(p0*q0 + p1*q1 + nu*(p0*q1 + p1*q0) + .5*(1 - nu)*p2*q2) + (a === b ? geometric : 0);
        }
        addBlock(face.vertices[j]!, face.vertices[l]!, block);
      }
    }
    for (const hinge of topology.hinges) {
      const m = topology.materials[topology.triangles[hinge.faces[0]]!.material]!, gamma = (m.bendingDamping_s ?? m.mechanicalDamping_s)/dt;
      const angle = dihedralGradient(x, hinge.vertices, angleGradient);
      if (!angleGradient.every(Number.isFinite)) throw new Error('Degenerate cloth hinge');
      const phi = bendingPhi(angle, m.bendingMode), strain = phi.value - bendingPhi(hinge.restAngle_rad, m.bendingMode).value + gamma*(phi.value - bendingPhi(hingeAngle(old, hinge.vertices), m.bendingMode).value);
      const fraction = (state.solidFraction[hinge.vertices[0]]! + state.solidFraction[hinge.vertices[1]]!)/2;
      const coefficient = bendingRigidity(m, fraction)*hinge.shapeFactor*Math.max(0, (1 + gamma)*phi.first*phi.first + strain*phi.second);
      if (!(coefficient > 0)) continue;
      for (let p = 0; p < 4; p++) for (let q = 0; q < 4; q++) {
        for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) block[3*a + b] = coefficient*angleGradient[3*p + a]!*angleGradient[3*q + b]!;
        addBlock(hinge.vertices[p], hinge.vertices[q], block);
      }
    }
    for (const rod of topology.rods) {
      const m = topology.materials[rod.material]!;
      for (let j = 0; j < rod.restLengths_m.length; j++) {
        const a = rod.vertices[j]!, b = rod.vertices[j + 1]!, edge = [0, 1, 2].map(k => x[3*b + k]! - x[3*a + k]!), l = Math.hypot(...edge), oldLength = Math.hypot(...[0, 1, 2].map(k => old[3*b + k]! - old[3*a + k]!));
        const gamma = (m.rodStretchDamping_s ?? m.mechanicalDamping_s)/dt*(m.attachmentMultiplier !== undefined ? rod.restLengths_m[j]!/oldLength : 1);
        const factor = rodRigidity(m, m.attachmentMultiplier !== undefined ? state.solidFraction[a]! : (state.solidFraction[a]! + state.solidFraction[b]!)/2).stretch_N/rod.restLengths_m[j]!;
        const strain = l - rod.restLengths_m[j]! + gamma*(l - oldLength), along = factor*(1 + gamma), across = factor*Math.max(0, strain)/l;
        for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) {const t = edge[p]!*edge[q]!/(l*l); block[3*p + q] = along*t + across*((p === q ? 1 : 0) - t);}
        addBlock(a, a, block); addBlock(b, b, block); addBlock(a, b, block, -1); addBlock(b, a, block, -1);
      }
    }
    for (const pair of cohesive) {
      if (!(pair.force_N > 0)) continue;
      const delta = [0, 0, 0];
      for (let q = 0; q < pair.vertices.length; q++) for (let a = 0; a < 3; a++) delta[a]! += pair.weights[q]!*x[3*pair.vertices[q]! + a]!;
      if (pair.targetPosition_m) for (let a = 0; a < 3; a++) delta[a]! -= pair.targetPosition_m[a]!;
      const distance = Math.max(1e-14, Math.hypot(...delta)), weight = new Map<number, number>();
      for (let q = 0; q < pair.vertices.length; q++) weight.set(pair.vertices[q]!, (weight.get(pair.vertices[q]!) ?? 0) + pair.weights[q]!);
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) block[3*p + q] = pair.force_N/distance*((p === q ? 1 : 0) - delta[p]!*delta[q]!/(distance*distance));
      for (const [i, wi] of weight) for (const [j, wj] of weight) addBlock(i, j, block, wi*wj);
    }
    for (let i = 0; i < n; i++) if (free[i] && !invert3(values, 9*pattern.diagonal[i]!, preconditioner, 9*i)) throw new Error(`Singular cloth Newton block at vertex ${i}`);
  };

  const multiply = (input: Float64Array, output: Float64Array) => {
    for (let i = 0; i < n; i++) {
      let y0 = 0, y1 = 0, y2 = 0;
      if (free[i]) for (let b = pattern.rowStart[i]!, end = pattern.rowStart[i + 1]!; b < end; b++) {
        const c = 3*pattern.columns[b]!, o = 9*b, v0 = input[c]!, v1 = input[c+1]!, v2 = input[c+2]!;
        y0 += values[o]!*v0 + values[o+1]!*v1 + values[o+2]!*v2; y1 += values[o+3]!*v0 + values[o+4]!*v1 + values[o+5]!*v2; y2 += values[o+6]!*v0 + values[o+7]!*v1 + values[o+8]!*v2;
      }
      output[3*i] = y0; output[3*i+1] = y1; output[3*i+2] = y2;
    }
  };
  const precondition = (input: Float64Array, output: Float64Array) => {
    for (let i = 0; i < n; i++) {
      const o = 9*i, v0 = input[3*i]!, v1 = input[3*i+1]!, v2 = input[3*i+2]!;
      if (!free[i]) {output[3*i] = output[3*i+1] = output[3*i+2] = 0; continue;}
      output[3*i] = preconditioner[o]!*v0 + preconditioner[o+1]!*v1 + preconditioner[o+2]!*v2; output[3*i+1] = preconditioner[o+3]!*v0 + preconditioner[o+4]!*v1 + preconditioner[o+5]!*v2; output[3*i+2] = preconditioner[o+6]!*v0 + preconditioner[o+7]!*v1 + preconditioner[o+8]!*v2;
    }
  };
  const dot = (a: Float64Array, b: Float64Array) => {let s = 0; for (let k = 0; k < a.length; k++) s += a[k]!*b[k]!; return s;};
  const step = new Float64Array(3*n), residual = new Float64Array(3*n), search = new Float64Array(3*n), product = new Float64Array(3*n), preconditioned = new Float64Array(3*n), twistStep = new Float64Array(S);
  const maximumLinear = Math.max(50, 10*n);
  // Preconditioned CG on H p = -g; Eisenstat-Walker forcing keeps early
  // Newton steps cheap and the final ones accurate.
  const solveLinear = (forcing: number): number => {
    step.fill(0);
    for (let k = 0; k < 3*n; k++) residual[k] = -gradient[k]!;
    precondition(residual, preconditioned); search.set(preconditioned);
    let rz = dot(residual, preconditioned);
    const stop = forcing*forcing*rz;
    let performed = 0;
    for (; performed < maximumLinear && rz > stop && rz > 0; performed++) {
      multiply(search, product);
      const curvature = dot(search, product);
      if (!(curvature > 0)) {if (performed === 0) step.set(search); break;}
      const alpha = rz/curvature;
      for (let k = 0; k < 3*n; k++) {step[k]! += alpha*search[k]!; residual[k]! -= alpha*product[k]!;}
      precondition(residual, preconditioned);
      const next = dot(residual, preconditioned), beta = next/rz; rz = next;
      for (let k = 0; k < 3*n; k++) search[k] = preconditioned[k]! + beta*search[k]!;
    }
    solveTwist();
    return performed;
  };

  // Twist couples only to its rod neighbours, through the joint they share.
  // That Hessian is tridiagonal along each rod: the exact second derivative
  // on the diagonal, the mixed central difference of the shared joint off it,
  // solved directly. A diagonal-only twist step overshoots this Laplacian-like
  // coupling by up to 2x and stalls Newton at a linear rate.
  const twistCoupling = new Float64Array(S), pivot = new Float64Array(S);
  const solveTwist = () => {
    twistStep.fill(0);
    for (const rod of topology.rods) {
      const count = rod.restLengths_m.length, base = rod.segmentOffset;
      for (let j = 0; j + 1 < count; j++) {
        const s = base + j;
        twistCoupling[s] = 0;
        if (!twistFree[s] || !twistFree[s + 1]) continue;
        const h = 1e-4, a = twist[s]!, b = twist[s + 1]!, E = () => rodEnergy(topology, state, twistRod[s]!, undefined, old, oldTwist, dt, j);
        twist[s] = a + h; twist[s + 1] = b + h; const pp = E(); twist[s + 1] = b - h; const pm = E();
        twist[s] = a - h; const mm = E(); twist[s + 1] = b + h; const mp = E(); twist[s] = a; twist[s + 1] = b;
        twistCoupling[s] = (pp - pm - mp + mm)/(4*h*h);
      }
      // Thomas elimination; a nonpositive pivot (twist locally unstable)
      // leaves that rod on the diagonal step.
      let positive = true;
      for (let j = 0; j < count && positive; j++) {
        const s = base + j;
        if (!twistFree[s]) {pivot[s] = 1; twistStep[s] = 0; continue;}
        const coupling = j > 0 ? twistCoupling[s - 1]! : 0, previous = j > 0 && twistFree[s - 1] ? pivot[s - 1]! : 1;
        pivot[s] = twistSecond[s]! - coupling*coupling/previous;
        twistStep[s] = -twistGradient[s]! - (j > 0 ? coupling*twistStep[s - 1]!/previous : 0);
        positive = pivot[s]! > 0 && Number.isFinite(pivot[s]);
      }
      if (positive) for (let j = count - 1; j >= 0; j--) {
        const s = base + j;
        if (!twistFree[s]) continue;
        twistStep[s] = (twistStep[s]! - (j + 1 < count ? twistCoupling[s]!*twistStep[s + 1]! : 0))/pivot[s]!;
      }
      else for (let j = 0; j < count; j++) {const s = base + j; twistStep[s] = twistFree[s] ? -twistGradient[s]!/twistDiagonal[s]! : 0;}
    }
  };
  const startX = new Float64Array(3*n), startTwist = new Float64Array(S);
  // One Newton iteration: assemble, solve, Armijo backtracking on the energy.
  const iterate = (forcing: number): boolean => {
    assemble();
    linear += solveLinear(forcing);
    let slope = dot(gradient, step) + dot(twistGradient, twistStep);
    if (!(slope < 0)) {
      // Roundoff in CG; the preconditioned gradient is always a descent direction.
      precondition(gradient, step); for (let k = 0; k < 3*n; k++) step[k] = -step[k]!;
      slope = dot(gradient, step) + dot(twistGradient, twistStep);
    }
    startX.set(x); startTwist.set(twist);
    const energy = totalEnergy(), slack = 64*Number.EPSILON*Math.abs(energy);
    for (let line = 0, alpha = 1; line < 40; line++, alpha *= .5) {
      for (let k = 0; k < 3*n; k++) x[k] = startX[k]! + alpha*step[k]!;
      for (let s = 0; s < S; s++) twist[s] = startTwist[s]! + alpha*twistStep[s]!;
      const trial = totalEnergy();
      if (Number.isFinite(trial) && trial <= energy + 1e-4*alpha*slope + slack) return true;
    }
    x.set(startX); twist.set(startTwist);
    return false;
  };
  let measure = evaluate(), newton = 0, linear = 0, previousNorm = Infinity;
  while (measure >= tolerance && newton < maximumNewton) {
    newton++;
    const norm = Math.sqrt(dot(gradient, gradient) + dot(twistGradient, twistGradient));
    const forcing = Math.min(.1, Math.max(1e-6, Number.isFinite(previousNorm) ? norm/previousNorm : .1));
    previousNorm = norm;
    if (!iterate(forcing)) break;
    measure = evaluate();
  }
  return {...finishClothMechanicalStep(topology, state, old, oldTwist, dt, options, {iterations: newton, converged: measure < tolerance, maximumUpdate_m: measure}), newtonIterations: newton, linearIterations: linear};
}
