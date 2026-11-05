import {cohesionPairValue,cohesionSolidReactions,type ClothCohesionPair,type ClothCohesionReaction} from './cohesion';
import { bendingRigidity, membraneModulus, rodRigidity, rodSegmentInertia_kg_m2 } from "./material";
import { cross3, dot3, hingeAngle, norm3, point3, scale3, sub3, transportDirector, updateClothControlVolumes, type ClothRod, type ClothTriangle, type PreparedClothTopology } from "./topology";
import type { ClothState, ClothVector } from "./types";
import { resolveClothContacts, type ClothCollider, type ClothContactReport } from "./contact";

const wrapAngle = (a: number) => Math.atan2(Math.sin(a), Math.cos(a));
export function bendingPhi(angle:number,mode: "theta"|"tan-half"|"sin-half" = "theta") {
  if(mode==="tan-half") {const t=Math.tan(angle/2);return {value:2*t,first:1+t*t,second:t*(1+t*t)};}
  if(mode==="sin-half") return {value:Math.sin(angle/2),first:.5*Math.cos(angle/2),second:-.25*Math.sin(angle/2)};
  return {value:angle,first:1,second:0};
}

export function triangleStrain(face: ClothTriangle, positions: ArrayLike<number>): ClothVector {
  const u: [number, number, number] = [0, 0, 0], v: [number, number, number] = [0, 0, 0];
  for (let j = 0; j < 3; j++) for (let axis = 0; axis < 3; axis++) {
    u[axis]! += positions[3 * face.vertices[j]! + axis]! * face.ru[j]!;
    v[axis]! += positions[3 * face.vertices[j]! + axis]! * face.rv[j]!;
  }
  return [.5 * (dot3(u, u) - 1), .5 * (dot3(v, v) - 1), dot3(u, v)];
}

export function membraneEnergy(topology: PreparedClothTopology, state: ClothState, f: number, prior?: ArrayLike<number>, dt = 1): number {
  const face = topology.triangles[f]!, m = topology.materials[face.material]!, strain = triangleStrain(face, state.positions_m);
  const fraction = face.vertices.reduce((sum, i) => sum + state.solidFraction[i]!, 0) / 3;
  const modulus = membraneModulus(m, fraction), nu = m.poissonRatio;
  const quadratic = (e: ClothVector) => e[0] ** 2 + e[1] ** 2 + 2 * nu * e[0] * e[1] + .5 * (1 - nu) * e[2] ** 2;
  let energy = quadratic(strain);
  if (prior && (m.membraneDamping_s??m.mechanicalDamping_s) > 0) energy += (m.membraneDamping_s??m.mechanicalDamping_s) / dt * quadratic(sub3(strain, triangleStrain(face, prior)));
  return .5 * face.restArea_m2 * modulus * energy;
}

/** Exact first/second derivative of Green-strain plane-stress energy with
 * respect to one Cartesian coordinate; includes Kelvin-Voigt dissipation. */
export function membraneCoordinateDerivatives(topology: PreparedClothTopology, state: ClothState, f: number, vertex: number, axis: number, prior?: ArrayLike<number>, dt = 1): {gradient: number; diagonal: number} {
  const face = topology.triangles[f]!, j = face.vertices.indexOf(vertex);
  if (j < 0) return {gradient: 0, diagonal: 0};
  const m = topology.materials[face.material]!, nu = m.poissonRatio, e = triangleStrain(face, state.positions_m), old = prior ? triangleStrain(face, prior) : e;
  const gamma = prior ? (m.membraneDamping_s??m.mechanicalDamping_s) / dt : 0, ru = face.ru[j]!, rv = face.rv[j]!;
  let u = 0, v = 0;
  for (let k = 0; k < 3; k++) {u += state.positions_m[3 * face.vertices[k]! + axis]! * face.ru[k]!; v += state.positions_m[3 * face.vertices[k]! + axis]! * face.rv[k]!;}
  const a = ru * u, b = rv * v, c = ru * v + rv * u;
  const ex = e[0] + gamma * (e[0] - old[0]), ey = e[1] + gamma * (e[1] - old[1]), es = e[2] + gamma * (e[2] - old[2]);
  const sx = ex + nu * ey, sy = ey + nu * ex, ss = .5 * (1 - nu) * es;
  const fraction = face.vertices.reduce((sum, i) => sum + state.solidFraction[i]!, 0) / 3, factor = face.restArea_m2 * membraneModulus(m, fraction);
  return {gradient: factor * (sx * a + sy * b + ss * c), diagonal: factor * ((1 + gamma) * (a * a + b * b + 2 * nu * a * b + .5 * (1 - nu) * c * c) + sx * ru * ru + sy * rv * rv + 2 * ss * ru * rv)};
}

export function hingeEnergy(topology: PreparedClothTopology, state: ClothState, index: number, prior?: ArrayLike<number>, dt = 1): number {
  const hinge = topology.hinges[index]!, face = topology.triangles[hinge.faces[0]]!, m = topology.materials[face.material]!;
  const fraction = (state.solidFraction[hinge.vertices[0]]! + state.solidFraction[hinge.vertices[1]]!) / 2;
  let angle: number;
  try { angle = hingeAngle(state.positions_m, hinge.vertices); } catch { return Infinity; }
  const phi = bendingPhi(angle,m.bendingMode).value, d = phi - bendingPhi(hinge.restAngle_rad,m.bendingMode).value;
  let value = d * d;
  if (prior && (m.bendingDamping_s??m.mechanicalDamping_s) > 0) value += (m.bendingDamping_s??m.mechanicalDamping_s) / dt * (phi - bendingPhi(hingeAngle(prior, hinge.vertices),m.bendingMode).value) ** 2;
  return .5 * bendingRigidity(m, fraction) * hinge.shapeFactor * value;
}

export function hingeCoordinateDerivatives(topology: PreparedClothTopology, state: ClothState, index: number, vertex: number, axis: number, prior?: ArrayLike<number>, dt = 1): {gradient:number;diagonal:number} {
  const hinge = topology.hinges[index]!, local = hinge.vertices.indexOf(vertex);
  if (local < 0) return {gradient:0,diagonal:0};
  const [a,b,c,d] = hinge.vertices.map(i=>point3(state.positions_m,i)) as unknown as [ClothVector,ClothVector,ClothVector,ClothVector];
  const unit: [number,number,number] = [0,0,0]; unit[axis] = 1;
  const e = sub3(b,a), u = sub3(c,a), v = sub3(d,a), de = scale3(unit, local===0?-1:local===1?1:0), du = scale3(unit, local===0?-1:local===2?1:0), dv = scale3(unit, local===0?-1:local===3?1:0);
  const add = (p:ClothVector,q:ClothVector):ClothVector => [p[0]+q[0],p[1]+q[1],p[2]+q[2]];
  const n = cross3(e,u), m = cross3(v,e), dn = add(cross3(de,u),cross3(e,du)), dm = add(cross3(dv,e),cross3(v,de));
  const l = norm3(e), dl = dot3(e,de)/l, ddl = (dot3(de,de)-dl*dl)/l;
  const t = scale3(e,1/l), td = sub3(scale3(de,1/l),scale3(e,dl/(l*l))), tdd = add(scale3(de,-2*dl/(l*l)),scale3(e,2*dl*dl/(l*l*l)-ddl/(l*l)));
  const product = cross3(n,m), productD = add(cross3(dn,m),cross3(n,dm)), productDD = scale3(cross3(dn,dm),2);
  const x = dot3(n,m), dx = dot3(dn,m)+dot3(n,dm), ddx = 2*dot3(dn,dm), y = dot3(t,product), dy = dot3(td,product)+dot3(t,productD), ddy = dot3(tdd,product)+2*dot3(td,productD)+dot3(t,productDD);
  const denominator=x*x+y*y, numerator=x*dy-y*dx, angle=Math.atan2(y,x), derivative=numerator/denominator, second=((x*ddy-y*ddx)*denominator-numerator*2*(x*dx+y*dy))/(denominator*denominator);
  const material=topology.materials[topology.triangles[hinge.faces[0]]!.material]!, gamma=prior?(material.bendingDamping_s??material.mechanicalDamping_s)/dt:0;
  const phi=bendingPhi(angle,material.bendingMode),strain=phi.value-bendingPhi(hinge.restAngle_rad,material.bendingMode).value+gamma*(prior?phi.value-bendingPhi(hingeAngle(prior,hinge.vertices),material.bendingMode).value:0);
  const fraction=(state.solidFraction[hinge.vertices[0]]!+state.solidFraction[hinge.vertices[1]]!)/2, factor=bendingRigidity(material,fraction)*hinge.shapeFactor;
  return {gradient:factor*strain*phi.first*derivative,diagonal:factor*((1+gamma)*phi.first*phi.first*derivative*derivative+strain*(phi.first*second+phi.second*derivative*derivative))};
}

/** Every (vertex, axis) gradient of one hinge's bending energy, evaluated
 * with the same arithmetic as hingeCoordinateDerivatives(...).gradient but
 * sharing the hinge's angle, strain and rigidity across its 12 coordinates. */
export function hingeCoordinateGradients(topology: PreparedClothTopology, state: ClothState, index: number, prior: ArrayLike<number> | undefined, dt: number, out: Float64Array, offset = 0): void {
  const hinge = topology.hinges[index]!, P = state.positions_m, [ia, ib, ic, id] = hinge.vertices;
  const e0 = P[3*ib]! - P[3*ia]!, e1 = P[3*ib+1]! - P[3*ia+1]!, e2 = P[3*ib+2]! - P[3*ia+2]!;
  const u0 = P[3*ic]! - P[3*ia]!, u1 = P[3*ic+1]! - P[3*ia+1]!, u2 = P[3*ic+2]! - P[3*ia+2]!;
  const v0 = P[3*id]! - P[3*ia]!, v1 = P[3*id+1]! - P[3*ia+1]!, v2 = P[3*id+2]! - P[3*ia+2]!;
  const n0 = e1*u2 - e2*u1, n1 = e2*u0 - e0*u2, n2 = e0*u1 - e1*u0, m0 = v1*e2 - v2*e1, m1 = v2*e0 - v0*e2, m2 = v0*e1 - v1*e0;
  const l = Math.hypot(e0, e1, e2), il = 1/l, ll = l*l, t0 = e0*il, t1 = e1*il, t2 = e2*il;
  const p0 = n1*m2 - n2*m1, p1 = n2*m0 - n0*m2, p2 = n0*m1 - n1*m0, x = n0*m0 + n1*m1 + n2*m2, y = t0*p0 + t1*p1 + t2*p2;
  const denominator = x*x + y*y, angle = Math.atan2(y, x);
  const material = topology.materials[topology.triangles[hinge.faces[0]]!.material]!, gamma = prior ? (material.bendingDamping_s ?? material.mechanicalDamping_s) / dt : 0;
  const phi = bendingPhi(angle, material.bendingMode), strain = phi.value - bendingPhi(hinge.restAngle_rad, material.bendingMode).value + gamma * (prior ? phi.value - bendingPhi(hingeAngle(prior, hinge.vertices), material.bendingMode).value : 0);
  const fraction = (state.solidFraction[hinge.vertices[0]]! + state.solidFraction[hinge.vertices[1]]!) / 2, coefficient = bendingRigidity(material, fraction) * hinge.shapeFactor * strain * phi.first;
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
      out[offset + 3*local + axis] = coefficient * ((x*dy - y*dx) / denominator);
    }
  }
}

function rodFrame(rod: ClothRod, segment: number, state: ClothState, positions = state.positions_m, twists = state.twist_rad) {
  const global = rod.segmentOffset + segment, edge = sub3(point3(positions, rod.vertices[segment + 1]!), point3(positions, rod.vertices[segment]!)), length = norm3(edge);
  if (length < 1e-14) return undefined;
  const tangent = scale3(edge, 1 / length), director = transportDirector(point3(state.rodDirectors, global), point3(state.rodTangents, global), tangent), other = cross3(tangent, director);
  const theta = twists[global]!, c = Math.cos(theta), s = Math.sin(theta);
  const m1: ClothVector = [c * director[0] + s * other[0], c * director[1] + s * other[1], c * director[2] + s * other[2]];
  return { length, tangent, director, m1, m2: cross3(tangent, m1) };
}

function rodJoint(rod: ClothRod, j: number, state: ClothState, positions = state.positions_m, twists = state.twist_rad) {
  const a = rodFrame(rod, j - 1, state, positions, twists), b = rodFrame(rod, j, state, positions, twists);
  if (!a || !b || 1 + dot3(a.tangent, b.tangent) < 1e-10) return undefined;
  const kb = scale3(cross3(a.tangent, b.tangent), 2 / (1 + dot3(a.tangent, b.tangent)));
  const curvature: readonly [number,number,number,number] = [dot3(kb,a.m2),-dot3(kb,a.m1),dot3(kb,b.m2),-dot3(kb,b.m1)];
  const transported = transportDirector(a.director, a.tangent, b.tangent);
  const rawReferenceTwist = Math.atan2(dot3(b.tangent, cross3(transported, b.director)), dot3(transported, b.director)),previousReference=state.rodReferenceTwist_rad[rod.segmentOffset+j]!,referenceTwist=previousReference+wrapAngle(rawReferenceTwist-previousReference);
  return { curvature, referenceTwist, twist: twists[rod.segmentOffset + j]! - twists[rod.segmentOffset + j - 1]! + referenceTwist };
}

/** Discrete elastic rod stretch, material-frame bending and twist energies. */
const rodLocalCache = new WeakMap<ClothRod, Map<number, readonly number[]>>();
function rodLocalVertices(rod: ClothRod, vertex: number): readonly number[] {
  let map = rodLocalCache.get(rod);
  if (!map) { map = new Map<number, number[]>(); rod.vertices.forEach((v, i) => map!.set(v, [...(map!.get(v) ?? []), i])); rodLocalCache.set(rod, map); }
  return map.get(vertex) ?? [];
}
export function rodEnergy(topology: PreparedClothTopology, state: ClothState, index: number, vertex?: number, prior?: Float64Array, priorTwist?: Float64Array, dt = 1, twistSegment?: number, skipStretch = false): number {
  const rod = topology.rods[index]!, material = topology.materials[rod.material]!;
  const localVertices = vertex === undefined ? undefined : rodLocalVertices(rod, vertex);
  const segments = skipStretch || twistSegment !== undefined ? [] : localVertices ? [...new Set(localVertices.flatMap(i => [i - 1, i]))].filter(j => j >= 0 && j < rod.restLengths_m.length) : rod.restLengths_m.keys();
  let energy = 0;
  for (const j of segments) {
    const fraction = material.attachmentMultiplier!==undefined?state.solidFraction[rod.vertices[j]!]!:(state.solidFraction[rod.vertices[j]!]! + state.solidFraction[rod.vertices[j + 1]!]!) / 2, rigidity = rodRigidity(material, fraction);
    const length = norm3(sub3(point3(state.positions_m, rod.vertices[j + 1]!), point3(state.positions_m, rod.vertices[j]!))); if (length < 1e-14) return Infinity;
    const rest = rod.restLengths_m[j]!, strain = length / rest - 1;
    energy += .5 * rigidity.stretch_N * rest * strain ** 2;
    if (prior && (material.rodStretchDamping_s??material.mechanicalDamping_s)) {
      const oldLength = norm3(sub3(point3(prior, rod.vertices[j + 1]!), point3(prior, rod.vertices[j]!)));
      energy += .5 * rigidity.stretch_N / (material.attachmentMultiplier!==undefined?oldLength:rest) * (material.rodStretchDamping_s??material.mechanicalDamping_s) / dt * (length - oldLength) ** 2;
    }
  }
  const joints = twistSegment !== undefined ? [twistSegment, twistSegment + 1] : localVertices ? [...new Set(localVertices.flatMap(i => [i - 1, i, i + 1]))] : Array.from({length: rod.restLengths_m.length - 1}, (_, i) => i + 1);
  for (const j of joints) {
    if (j < 1 || j >= rod.restLengths_m.length) continue;
    const fraction = material.attachmentMultiplier!==undefined?state.solidFraction[rod.vertices[j]!]!:(state.solidFraction[rod.vertices[j - 1]!]! + state.solidFraction[rod.vertices[j]!]! + state.solidFraction[rod.vertices[j + 1]!]!) / 3;
    const rigidity = rodRigidity(material, fraction), joint = rodJoint(rod, j, state); if (!joint) return Infinity;
    const length = (rod.restLengths_m[j - 1]! + rod.restLengths_m[j]!) / 2;
    const bending=(reference:ArrayLike<number>,offset=0)=>joint.curvature.reduce((sum,value,k)=>sum+(k%2?rigidity.bend2_N_m2:rigidity.bend1_N_m2)*(value-reference[offset+k]!)**2,0);
    energy += .25 / length * bending(rod.restCurvature,4*(j-1)) + .5/length*rigidity.twist_N_m2*(joint.twist-rod.restTwist[j-1]!)**2;
    if (prior && priorTwist) {
      const old = rodJoint(rod, j, state, prior, priorTwist)!;
      energy += .25/(length*dt)*(material.rodBendingDamping_s??material.mechanicalDamping_s)*bending(old.curvature) + .5/(length*dt)*(material.rodTwistDamping_s??material.mechanicalDamping_s)*rigidity.twist_N_m2*(joint.twist-old.twist)**2;
    }
  }
  return energy;
}

export function rodStretchCoordinateDerivatives(topology:PreparedClothTopology,state:ClothState,index:number,vertex:number,axis:number,prior:Float64Array,dt:number) {
  const rod=topology.rods[index]!,material=topology.materials[rod.material]!;
  let gradient=0,diagonal=0;
  const segments=[...new Set(rodLocalVertices(rod,vertex).flatMap(i=>[i-1,i]))].filter(j=>j>=0&&j<rod.restLengths_m.length);
  for(const j of segments) {
    const a=rod.vertices[j]!,b=rod.vertices[j+1]!,edge=sub3(point3(state.positions_m,b),point3(state.positions_m,a)),l=norm3(edge),old=norm3(sub3(point3(prior,b),point3(prior,a))),d=edge[axis]!/l,sign=vertex===b?1:-1;
    const gamma=(material.rodStretchDamping_s??material.mechanicalDamping_s)/dt*(material.attachmentMultiplier!==undefined?rod.restLengths_m[j]!/old:1);
    const factor=rodRigidity(material,material.attachmentMultiplier!==undefined?state.solidFraction[a]!:(state.solidFraction[a]!+state.solidFraction[b]!)/2).stretch_N/rod.restLengths_m[j]!,strain=l-rod.restLengths_m[j]!+gamma*(l-old);
    gradient+=factor*strain*d*sign;diagonal+=factor*((1+gamma)*d*d+strain/l*(1-d*d));
  }
  return {gradient,diagonal};
}

export function clothElasticEnergy(topology: PreparedClothTopology, state: ClothState): number {
  let energy = 0;
  for (let i = 0; i < topology.triangles.length; i++) energy += membraneEnergy(topology, state, i);
  for (let i = 0; i < topology.hinges.length; i++) energy += hingeEnergy(topology, state, i);
  for (let i = 0; i < topology.rods.length; i++) energy += rodEnergy(topology, state, i);
  return energy;
}

export interface ClothStepOptions {
  /** Accepted native-liquid-gated source capillary bridges. */
  readonly cohesionPairs?:readonly ClothCohesionPair[];
  readonly gravity_m_s2?: ClothVector;
  readonly externalForces_N?: ArrayLike<number>;
  readonly supportPositions_m?: ArrayLike<number>;
  readonly colliders?: readonly ClothCollider[];
  readonly selfCollision?: boolean;
  readonly iterations?: number;
  readonly tolerance_m?: number;
  /** Source-compatible fixed material control volumes when false. */
  readonly varyingSolidFraction?: boolean;
}
export interface ClothStepReport {
  readonly cohesionEnergy_J?:number;
  readonly cohesionReactions?:readonly ClothCohesionReaction[];
  readonly gpuIterations?: number;
  readonly refinementIterations?: number;
  readonly iterations: number;
  readonly converged: boolean;
  readonly maximumUpdate_m: number;
  readonly elasticEnergy_J: number;
  readonly kineticEnergy_J: number;
  readonly contacts: ClothContactReport;
}

/** Float64 reference implicit Euler. Coordinate minimization evaluates the
 * physical energy, including nonlinear strain; it is not a spring proxy.
 * Finite-difference local derivatives deliberately favor auditability over
 * throughput. Production GPU kernels must be checked against this oracle. */
export function nonFluidClothPredictor(topology:PreparedClothTopology,state:ClothState,dt:number,options:ClothStepOptions={}):Float64Array {
  if(!(Number.isFinite(dt)&&dt>0))throw new Error('Cloth predictor timestep must be positive and finite');
  const x=state.positions_m,old=x.slice(),twist=state.twist_rad.slice(),target=options.supportPositions_m??topology.data.restPositions_m,gravity=options.gravity_m_s2??[0,-9.81,0],result=state.velocities_m_s.slice(),cohesive=options.cohesionPairs??[],incident:number[][]=Array.from({length:topology.vertexCount},()=>[]);
  for(let j=0;j<cohesive.length;j++)for(const i of new Set(cohesive[j]!.vertices))incident[i]!.push(j);
  const hingeGradient=new Float64Array(12*topology.hinges.length);for(let h=0;h<topology.hinges.length;h++)hingeCoordinateGradients(topology,state,h,old,dt,hingeGradient,12*h);
  for(let i=0;i<topology.vertexCount;i++){
    const mass=state.dryMass_kg[i]!+topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3*state.waterVolume_m3[i]!;
    if(!(mass>0&&Number.isFinite(mass)))throw new Error('Invalid cloth predictor mass');
    const localBending=()=>topology.incidentRods[i]!.reduce((sum,r)=>sum+rodEnergy(topology,state,r,i,old,twist,dt,undefined,true),0),h=Math.max(1e-9,Math.cbrt(state.restControlVolume_m3[i]!)*1e-4);
    for(let axis=0;axis<3;axis++){
      const k=3*i+axis;if((topology.kinematicFixed[i]??0)&1){result[k]=(target[k]!-old[k]!)/dt;continue;}
      let gradient=topology.attachmentTranslation_N_m[i]!*(x[k]!-target[k]!);
      for(const f of topology.incidentTriangles[i]!)gradient+=membraneCoordinateDerivatives(topology,state,f,i,axis,old,dt).gradient;
      for(const f of topology.incidentHinges[i]!)gradient+=hingeGradient[12*f+3*topology.hinges[f]!.vertices.indexOf(i)+axis]!;
      for(const r of topology.incidentRods[i]!)gradient+=rodStretchCoordinateDerivatives(topology,state,r,i,axis,old,dt).gradient;
      if(topology.incidentRods[i]!.some(r=>topology.rods[r]!.vertices.length>2)){
        try{x[k]=old[k]!+h;const plus=localBending();x[k]=old[k]!-h;gradient+=(plus-localBending())/(2*h);}finally{x[k]=old[k]!;}
      }
      for(const j of incident[i]!)gradient+=cohesionPairValue(cohesive[j]!,x,i,axis).gradient_N;
      result[k]!+=dt*(gravity[axis]!+((options.externalForces_N?.[k]??0)-gradient)/mass);
      if(!Number.isFinite(result[k]))throw new Error(`Nonfinite cloth predictor at vertex ${i}`);
    }
  }
  return result;
}

export function stepCloth(topology: PreparedClothTopology, state: ClothState, dt: number, options: ClothStepOptions = {}, initialGuess?:{readonly positions_m:ArrayLike<number>;readonly twist_rad:ArrayLike<number>}): ClothStepReport {
  if (!(Number.isFinite(dt) && dt > 0)) throw new Error("Cloth timestep must be positive and finite");
  const n = topology.vertexCount, x = state.positions_m, old = x.slice(), oldTwist = state.twist_rad.slice(), predicted = x.slice(), mass = new Float64Array(n);
  const gravity = options.gravity_m_s2 ?? [0, -9.81, 0], target = options.supportPositions_m ?? topology.data.restPositions_m;
  for (let i = 0; i < n; i++) {
    mass[i] = state.dryMass_kg[i]! + topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3 * state.waterVolume_m3[i]!;
    if (!(mass[i]! > 0 && Number.isFinite(mass[i]))) throw new Error(`Invalid cloth vertex mass ${i}`);
    for (let axis = 0; axis < 3; axis++) {
      const k = 3 * i + axis;
      predicted[k] = old[k]! + dt * state.velocities_m_s[k]! + dt * dt * (gravity[axis]! + (options.externalForces_N?.[k] ?? 0) / mass[i]!);
      x[k] = (topology.kinematicFixed[i] ?? 0) & 1 ? target[k]! : initialGuess?.positions_m[k] ?? predicted[k]!;
    }
  }
  const priorTwist = oldTwist.slice();
  for (let s = 0; s < state.twist_rad.length; s++) state.twist_rad[s] = initialGuess?.twist_rad[s] ?? oldTwist[s]! + dt * state.angularVelocity_rad_s[s]!;
  const iterations = options.iterations ?? 12, tolerance = options.tolerance_m ?? 1e-9;
  if (!Number.isInteger(iterations) || iterations < 1) throw new Error("Cloth iterations must be a positive integer");
  let maxUpdate = Infinity, performed = 0;
  const cohesive=options.cohesionPairs??[],incidentCohesion:number[][]=Array.from({length:n},()=>[]);
  for(let j=0;j<cohesive.length;j++){const pair=cohesive[j]!;if(pair.vertices.length!==pair.weights.length||!Number.isFinite(pair.force_N)||pair.force_N<0||!pair.weights.every(Number.isFinite)||Math.abs(pair.weights.reduce((a,b)=>a+b,0)-(pair.targetPosition_m?1:0))>1e-10||(pair.targetPosition_m&&!pair.targetPosition_m.every(Number.isFinite))||(pair.bodyId&&(!pair.targetPosition_m||!pair.bodyCenter_m?.every(Number.isFinite))))throw new Error('Invalid cloth cohesion pair');for(const i of new Set(pair.vertices)){if(!Number.isSafeInteger(i)||i<0||i>=n)throw new Error('Invalid cohesion vertex');incidentCohesion[i]!.push(j);}}
  const localEnergy = (i: number) => {
    let energy = 0;
    for(let a=0;a<3;a++)energy+=.5*topology.attachmentTranslation_N_m[i]!*(x[3*i+a]!-target[3*i+a]!)**2;
    for(const j of incidentCohesion[i]!)energy+=cohesionPairValue(cohesive[j]!,x).energy_J;
    for (let a = 0; a < 3; a++) energy += .5 * mass[i]! / (dt * dt) * (x[3 * i + a]! - predicted[3 * i + a]!) ** 2;
    for (const f of topology.incidentTriangles[i]!) energy += membraneEnergy(topology, state, f, old, dt);
    for (const h of topology.incidentHinges[i]!) energy += hingeEnergy(topology, state, h, old, dt);
    for (const r of topology.incidentRods[i]!) energy += rodEnergy(topology, state, r, i, old, oldTwist, dt);
    return energy;
  };
  const localRodBending = (i:number) => topology.incidentRods[i]!.reduce((sum,r)=>sum+rodEnergy(topology,state,r,i,old,oldTwist,dt,undefined,true),0);
  for (let iteration = 0; iteration < iterations; iteration++) {
    maxUpdate = 0; performed++;
    for (let visit = 0; visit < n; visit++) {
      const i = iteration % 2 ? n - 1 - visit : visit;
      if ((topology.kinematicFixed[i] ?? 0) & 1) continue;
      const scale = Math.cbrt(state.restControlVolume_m3[i]!), h = Math.max(1e-9, scale * 1e-4);
      for (let axis = 0; axis < 3; axis++) {
        const k = 3 * i + axis, value = x[k]!, e = localEnergy(i);
        if (!Number.isFinite(e)) throw new Error(`Cloth topology degenerated at vertex ${i}`);
        let gradient = mass[i]!/(dt*dt)*(value-predicted[k]!), stiffness = mass[i]!/(dt*dt);
        gradient+=topology.attachmentTranslation_N_m[i]!*(value-target[k]!);stiffness+=topology.attachmentTranslation_N_m[i]!;
        for (const f of topology.incidentTriangles[i]!) {const d = membraneCoordinateDerivatives(topology, state, f, i, axis, old, dt); gradient += d.gradient; stiffness += d.diagonal;}
        for (const h of topology.incidentHinges[i]!) {const d = hingeCoordinateDerivatives(topology,state,h,i,axis,old,dt); gradient += d.gradient; stiffness += d.diagonal;}
        for (const r of topology.incidentRods[i]!) {const d = rodStretchCoordinateDerivatives(topology,state,r,i,axis,old,dt); gradient += d.gradient; stiffness += d.diagonal;}
        if (topology.incidentRods[i]!.some(r=>topology.rods[r]!.vertices.length>2)) {
          const e = localRodBending(i); x[k]=value+h; const plus=localRodBending(i);x[k]=value-h;const minus=localRodBending(i);x[k]=value;
          gradient+=(plus-minus)/(2*h);stiffness+=(plus+minus-2*e)/(h*h);
        }
        for(const j of incidentCohesion[i]!){const d=cohesionPairValue(cohesive[j]!,x,i,axis);gradient+=d.gradient_N;stiffness+=d.diagonal_N_m;}
        const diagonal = Math.max(mass[i]! / (dt * dt), stiffness);
        maxUpdate = Math.max(maxUpdate, Math.abs(gradient / diagonal));
        let delta = Math.max(-scale, Math.min(scale, -gradient / diagonal));
        if (!Number.isFinite(delta)) delta = 0;
        for (let backtrack = 0; backtrack < 12; backtrack++) {
          x[k] = value + delta;
          if (localEnergy(i) <= e + 1e-18) break;
          x[k] = value; delta *= .5;
        }
        maxUpdate = Math.max(maxUpdate, Math.abs(x[k]! - value));
      }
    }
    // Twist is a real rotational degree of freedom, not inferred from stretch.
    for (let r = 0; r < topology.rods.length; r++) {
      const rod = topology.rods[r]!, m = topology.materials[rod.material]!;
      for (let j = 0; j < rod.restLengths_m.length; j++) {
        const s = rod.segmentOffset + j, a = rod.vertices[j]!, b = rod.vertices[j + 1]!;
        if (((topology.kinematicFixed[a] ?? 0) & 2) || ((topology.kinematicFixed[b] ?? 0) & 2)) { state.twist_rad[s] = oldTwist[s]!; continue; }
        const inertia = rodSegmentInertia_kg_m2(topology,state,r,j);
        const energy = () => .5*topology.attachmentTwist_N_m[s]!*(state.twist_rad[s]!-state.attachmentTwistTarget_rad[s]!)**2 + rodEnergy(topology, state, r, undefined, old, oldTwist, dt, j) + .5 * inertia / (dt * dt) * (state.twist_rad[s]! - priorTwist[s]! - dt * state.angularVelocity_rad_s[s]!) ** 2;
        const theta = state.twist_rad[s]!, h = 1e-5, e = energy(); state.twist_rad[s] = theta + h; const ep = energy(); state.twist_rad[s] = theta - h; const em = energy(); state.twist_rad[s] = theta;
        let delta = -(ep - em) / (2 * h) / Math.max(inertia / (dt * dt), (ep + em - 2 * e) / (h * h));
        if (!Number.isFinite(delta)) delta = 0;
        maxUpdate = Math.max(maxUpdate, Math.abs(delta) * Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m));
        delta = Math.max(-.25, Math.min(.25, delta));
        for (let line = 0; line < 12; line++) { state.twist_rad[s] = theta + delta; if (energy() <= e + 1e-20) break; state.twist_rad[s] = theta; delta *= .5; }
        maxUpdate = Math.max(maxUpdate, Math.abs(state.twist_rad[s]! - theta) * Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m));
      }
    }
    if (maxUpdate < tolerance) break;
  }
  return finishClothMechanicalStep(topology, state, old, oldTwist, dt, options, {iterations: performed, converged: maxUpdate < tolerance, maximumUpdate_m: maxUpdate});
}

/** Shared acceptance path for Float64 and GPU solvers: physical velocities,
 * CCD/reactions, reference-frame transport and conserved material volume. */
export function finishClothMechanicalStep(topology: PreparedClothTopology, state: ClothState, old: Float64Array, oldTwist: Float64Array, dt:number, options:ClothStepOptions, solve: Pick<ClothStepReport,"iterations"|"converged"|"maximumUpdate_m">):ClothStepReport {
  const x = state.positions_m, n = topology.vertexCount;
  const mass = Float64Array.from(state.dryMass_kg, (m,i)=>m+topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3*state.waterVolume_m3[i]!);
  for (let k = 0; k < x.length; k++) state.velocities_m_s[k] = (x[k]! - old[k]!) / dt;
  for (let s = 0; s < state.twist_rad.length; s++) state.angularVelocity_rad_s[s] = (state.twist_rad[s]! - oldTwist[s]!) / dt;
  const cohesionReactions=cohesionSolidReactions(options.cohesionPairs??[],x,dt);
  let contacts = resolveClothContacts(topology, state, old, dt, options.colliders ?? [], options.selfCollision ?? false);
  if(cohesionReactions.some(r=>r.bodyId)){
    const reactions=new Map(contacts.reactions.map(r=>[r.bodyId,{...r,linearImpulse_N_s:[...r.linearImpulse_N_s] as [number,number,number],angularImpulse_N_m_s:[...r.angularImpulse_N_m_s] as [number,number,number]}]));
    for(const r of cohesionReactions)if(r.bodyId){const current=reactions.get(r.bodyId)??{bodyId:r.bodyId,linearImpulse_N_s:[0,0,0] as [number,number,number],angularImpulse_N_m_s:[0,0,0] as [number,number,number]};for(let a=0;a<3;a++){current.linearImpulse_N_s[a]!+=r.linearImpulse_N_s[a]!;current.angularImpulse_N_m_s[a]!+=r.angularImpulse_N_m_s[a]!;}reactions.set(r.bodyId,current);}
    contacts={...contacts,reactions:[...reactions.values()]};
  }
  for(const rod of topology.rods)for(let j=1;j<rod.restLengths_m.length;j++)state.rodReferenceTwist_rad[rod.segmentOffset+j]=rodJoint(rod,j,state)!.referenceTwist;
  for (const rod of topology.rods) for (let j = 0; j < rod.restLengths_m.length; j++) {
    const frame = rodFrame(rod, j, state)!; state.rodDirectors.set(frame.director, 3 * (rod.segmentOffset + j)); state.rodTangents.set(frame.tangent, 3 * (rod.segmentOffset + j));
  }
  if(options.varyingSolidFraction===false){
    state.controlVolume_m3.set(state.restControlVolume_m3);
    for(let i=0;i<n;i++)state.solidFraction[i]=state.solidVolume_m3[i]!/state.restControlVolume_m3[i]!;
  }else updateClothControlVolumes(topology,state);
  state.time_s += dt;
  let kinetic = 0;
  for (let i = 0; i < n; i++) kinetic += .5 * mass[i]! * dot3(point3(state.velocities_m_s, i), point3(state.velocities_m_s, i));
  if (!x.every(Number.isFinite) || !state.velocities_m_s.every(Number.isFinite)) throw new Error("Nonfinite cloth mechanical state");
  return { ...solve, cohesionReactions, cohesionEnergy_J:(options.cohesionPairs??[]).reduce((sum,pair)=>sum+cohesionPairValue(pair,state.positions_m).energy_J,0), elasticEnergy_J: clothElasticEnergy(topology, state), kineticEnergy_J: kinetic, contacts };
}
