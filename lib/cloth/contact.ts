import {applySolidBoxCuts} from '../core/solid-cuts';
import { cross3, dot3, norm3, point3, scale3, sub3, type PreparedClothTopology } from "./topology";
import type { ClothState, ClothVector } from "./types";
import { quaternionInverseRotate, quaternionMultiply, quaternionRotate, type RigidBodyState } from "../core/rigid-body";
import { sceneShape } from "../core/scene-shape";
import { ClothContactBVH } from "./contact-bvh";
import { clothRodCapsuleGeometry, type ClothRodCapsuleGeometry } from './rod-geometry';
import { supportClippedRod, convexClothDistance } from './rod-convex';
import { sampleClippedRodCollider } from './clipped-contact';
import { rodContactDOF } from './rod-contact-dof';
interface ClothCollisionEdge {
    vertices: readonly [
        number,
        number
    ];
    r: number;
    min: number[];
    max: number[];
    rod?: number;
    segment?: number;
    clipped?: boolean;
    elliptical?:boolean;
    restLocalFaces?:Set<number>;
}
const contactTrees = new WeakMap<PreparedClothTopology, {
    faceBounds: Float64Array;
    vertexLocalFaces?:Map<number,Set<number>>;
    faceTree?: ClothContactBVH;
    edges?: ClothCollisionEdge[];
    edgeBounds?: Float64Array;
    edgeTree?: ClothContactBVH;
}>();
interface ColliderBase {
    readonly bodyId?: string;
    readonly friction?: number;
    readonly velocity_m_s?: ClothVector;
    readonly centreOfMass_m?: ClothVector;
}
export type ClothCollider = (ColliderBase & {
    readonly kind: "plane";
    readonly normal: ClothVector;
    readonly offset_m: number;
}) | (ColliderBase & {
    readonly kind: "sphere";
    readonly center_m: ClothVector;
    readonly radius_m: number;
}) | (ColliderBase & {
    readonly kind: "box";
    readonly minimum_m: ClothVector;
    readonly maximum_m: ClothVector;
}) | (ColliderBase & {
    readonly kind: "sampled";
    readonly sample: (p: ClothVector, beforeEnd_s: number) => {
        distance_m: number;
        normal: ClothVector;
    };
    readonly velocityAt: (p: ClothVector) => ClothVector;
    readonly maximumSpeed_m_s: number;
    /** Bound on the spatial gradient of the sampled signed field. */
    readonly spatialLipschitz?:number;
    readonly distanceLowerBound?:(minimum:ClothVector,maximum:ClothVector,beforeEnd_s:number)=>number;
    /** The same signed distance as sample(), without its normal. */
    readonly distanceAt?:(p:ClothVector,beforeEnd_s:number)=>number;
}) | (ColliderBase & {
    readonly kind: "rigid";
    readonly body: RigidBodyState;
});
export interface ClothReaction {
    readonly bodyId: string;
    readonly linearImpulse_N_s: [
        number,
        number,
        number
    ];
    readonly angularImpulse_N_m_s: [
        number,
        number,
        number
    ];
}
export interface ClothContactReport {
    readonly iterations?: number;
    readonly residual_m?: number;
    readonly contacts: number;
    readonly selfContacts: number;
    readonly maximumPenetration_m: number;
    readonly reactions: readonly ClothReaction[];
}
const add3 = (a: ClothVector, b: ClothVector): [
    number,
    number,
    number
] => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const lerp3 = (a: ClothVector, b: ClothVector, t: number): [
    number,
    number,
    number
] => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])];
/** Collider geometry describes its end-of-step pose; prescribed velocity
 * reconstructs its swept pose. Box3D remains the dynamic-body authority. */
function clothColliderDistance(c: ClothCollider, p: ClothVector, beforeEnd_s: number): number {
    return c.kind === "sampled" && c.distanceAt ? c.distanceAt(p, beforeEnd_s) : sampleClothCollider(c, p, beforeEnd_s).distance_m;
}
export function sampleClothCollider(c: ClothCollider, p: ClothVector, beforeEnd_s = 0): {
    distance_m: number;
    normal: ClothVector;
} {
    if (c.kind === "sampled")
        return c.sample(p, beforeEnd_s);
    if (c.kind === "rigid") {
        const body = c.body, shape = sceneShape(body.description.shape), omega = body.angularVelocity_rad_s, speed = Math.hypot(omega.x, omega.y, omega.z), half = -.5 * speed * beforeEnd_s;
        const s = speed > 0 ? Math.sin(half) / speed : 0;
        const orientation = quaternionMultiply({ w: Math.cos(half), x: omega.x * s, y: omega.y * s, z: omega.z * s }, body.orientation);
        const local = quaternionInverseRotate(orientation, { x: p[0] - body.position_m.x + beforeEnd_s * body.linearVelocity_m_s.x, y: p[1] - body.position_m.y + beforeEnd_s * body.linearVelocity_m_s.y, z: p[2] - body.position_m.z + beforeEnd_s * body.linearVelocity_m_s.z });
        const dimensions = body.description.dimensions_m, h = Math.max(1e-8, shape.boundingRadius_m(dimensions) * 1e-6), sign = body.description.fluidBoundary === "interior" ? -1 : 1;
        const signedDistance=(p:{x:number;y:number;z:number})=>applySolidBoxCuts(sign*shape.distance_m(dimensions,p),body.description.cuts,p);
        const distance_m = signedDistance(local), gradient = { x: 0, y: 0, z: 0 };
        for (const axis of ["x", "y", "z"] as const)
            gradient[axis] = (signedDistance({ ...local, [axis]: local[axis] + h }) - signedDistance({ ...local, [axis]: local[axis] - h })) / (2 * h);
        const world = quaternionRotate(orientation, gradient), length = Math.hypot(world.x, world.y, world.z);
        return { distance_m, normal: length > 1e-12 ? [world.x / length, world.y / length, world.z / length] : [0, 1, 0] };
    }
    const q = add3(p, scale3(c.velocity_m_s ?? [0, 0, 0], beforeEnd_s));
    if (c.kind === "plane") {
        const length = norm3(c.normal);
        if (!(length > 0))
            throw new Error("Invalid cloth collision plane");
        const normal = scale3(c.normal, 1 / length);
        return { distance_m: dot3(q, normal) - c.offset_m, normal };
    }
    if (c.kind === "sphere") {
        const delta = sub3(q, c.center_m), length = norm3(delta);
        return { distance_m: length - c.radius_m, normal: length > 1e-14 ? scale3(delta, 1 / length) : [0, 1, 0] };
    }
    if (c.kind === "box") {
        const closest: ClothVector = [0, 1, 2].map(a => Math.max(c.minimum_m[a]!, Math.min(c.maximum_m[a]!, q[a]!))) as unknown as ClothVector;
        const outside = sub3(q, closest), distance = norm3(outside);
        if (distance > 1e-14)
            return { distance_m: distance, normal: scale3(outside, 1 / distance) };
        let penetration = Infinity, axis = 0, sign = 1;
        for (let a = 0; a < 3; a++)
            for (const s of [-1, 1]) {
                const gap = s < 0 ? q[a]! - c.minimum_m[a]! : c.maximum_m[a]! - q[a]!;
                if (gap < penetration) {
                    penetration = gap;
                    axis = a;
                    sign = s;
                }
            }
        const normal: [
            number,
            number,
            number
        ] = [0, 0, 0];
        normal[axis] = sign;
        return { distance_m: -penetration, normal };
    }
    throw new Error(`Unsupported cloth collider ${(c as {
        kind: unknown;
    }).kind}`);
}
export function clothCollidersForRigidBodies(bodies: readonly RigidBodyState[]): ClothCollider[] {
    return bodies.map(body => ({ kind: "rigid", body, bodyId: body.description.id, friction: body.description.friction,
        velocity_m_s: [body.linearVelocity_m_s.x, body.linearVelocity_m_s.y, body.linearVelocity_m_s.z], centreOfMass_m: [body.position_m.x, body.position_m.y, body.position_m.z] }));
}
function colliderVelocity(collider: ClothCollider, point: ClothVector): ClothVector {
    if (collider.kind === "sampled")
        return collider.velocityAt(point);
    if (collider.kind !== "rigid")
        return collider.velocity_m_s ?? [0, 0, 0];
    const b = collider.body, omega: ClothVector = [b.angularVelocity_rad_s.x, b.angularVelocity_rad_s.y, b.angularVelocity_rad_s.z];
    return add3([b.linearVelocity_m_s.x, b.linearVelocity_m_s.y, b.linearVelocity_m_s.z], cross3(omega, sub3(point, [b.position_m.x, b.position_m.y, b.position_m.z])));
}
function colliderInverseEffectiveMass(collider: ClothCollider, point: ClothVector, direction: ClothVector): number {
    if (collider.kind !== "rigid" || collider.body.held || collider.body.description.motion === "static")
        return 0;
    const b = collider.body, arm = sub3(point, [b.position_m.x, b.position_m.y, b.position_m.z]);
    const torque = cross3(arm, direction), local = quaternionInverseRotate(b.orientation, { x: torque[0], y: torque[1], z: torque[2] });
    const acceleration = quaternionRotate(b.orientation, { x: local.x * b.inverseInertiaBody_kg_m2.x, y: local.y * b.inverseInertiaBody_kg_m2.y, z: local.z * b.inverseInertiaBody_kg_m2.z });
    return b.inverseMass_kg + dot3(cross3([acceleration.x, acceleration.y, acceleration.z], arm), direction);
}
export function closestTrianglePoint(p: ClothVector, a: ClothVector, b: ClothVector, c: ClothVector): {
    point: ClothVector;
    weights: ClothVector;
} {
    const ab = sub3(b, a), ac = sub3(c, a), ap = sub3(p, a), d1 = dot3(ab, ap), d2 = dot3(ac, ap);
    if (d1 <= 0 && d2 <= 0)
        return { point: a, weights: [1, 0, 0] };
    const bp = sub3(p, b), d3 = dot3(ab, bp), d4 = dot3(ac, bp);
    if (d3 >= 0 && d4 <= d3)
        return { point: b, weights: [0, 1, 0] };
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) {
        const v = d1 / (d1 - d3);
        return { point: add3(a, scale3(ab, v)), weights: [1 - v, v, 0] };
    }
    const cp = sub3(p, c), d5 = dot3(ab, cp), d6 = dot3(ac, cp);
    if (d6 >= 0 && d5 <= d6)
        return { point: c, weights: [0, 0, 1] };
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) {
        const w = d2 / (d2 - d6);
        return { point: add3(a, scale3(ac, w)), weights: [1 - w, 0, w] };
    }
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
        const w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return { point: add3(b, scale3(sub3(c, b), w)), weights: [0, 1 - w, w] };
    }
    const denominator = va + vb + vc;
    if (Math.abs(denominator) < 1e-30)
        return { point: a, weights: [1, 0, 0] };
    const v = vb / denominator, w = vc / denominator;
    return { point: add3(a, add3(scale3(ab, v), scale3(ac, w))), weights: [1 - v - w, v, w] };
}
export function closestSegmentPoints(a: ClothVector, b: ClothVector, c: ClothVector, d: ClothVector) {
    const u = sub3(b, a), v = sub3(d, c), w = sub3(a, c), aa = dot3(u, u), bb = dot3(u, v), cc = dot3(v, v), dd = dot3(u, w), ee = dot3(v, w);
    const clamp = (x: number) => Math.max(0, Math.min(1, x));
    let s = aa > 1e-30 ? clamp((bb * ee - cc * dd) / Math.max(1e-30, aa * cc - bb * bb)) : 0;
    let t = cc > 1e-30 ? (bb * s + ee) / cc : 0;
    if (t < 0) {
        t = 0;
        s = aa > 1e-30 ? clamp(-dd / aa) : 0;
    }
    else if (t > 1) {
        t = 1;
        s = aa > 1e-30 ? clamp((bb - dd) / aa) : 0;
    }
    return { s, t, a: lerp3(a, b, s), b: lerp3(c, d, t) };
}
const restFaceAdjacency=new WeakMap<PreparedClothTopology,readonly number[][]>();
/** A thick shell is one continuous volume near each material point. Exclude
 * only its connected rest patch inside the feature's collision support. A
 * nearby disconnected layer remains a collision candidate. This is independent
 * of mesh spacing and never shrinks the authored physical thickness. */
function connectedRestFaces(topology:PreparedClothTopology,seeds:readonly number[],radius:number,distance:(triangle:readonly ClothVector[])=>number):Set<number>{
 let adjacency=restFaceAdjacency.get(topology);if(!adjacency){const rows:number[][]=Array.from({length:topology.triangles.length},()=>[]);for(const hinge of topology.hinges){rows[hinge.faces[0]]!.push(hinge.faces[1]);rows[hinge.faces[1]]!.push(hinge.faces[0]);}adjacency=rows;restFaceAdjacency.set(topology,rows);}
 const visited=new Set<number>(),local=new Set<number>(),pending=[...seeds];
 for(let cursor=0;cursor<pending.length;cursor++){const f=pending[cursor]!;if(visited.has(f))continue;visited.add(f);const face=topology.triangles[f]!,support=radius+.5*topology.materials[face.material]!.thickness_m;
  if(distance(face.vertices.map(v=>point3(topology.data.restPositions_m,v)))>support+1e-12)continue;
  local.add(f);for(const neighbor of adjacency[f]!)if(!visited.has(neighbor))pending.push(neighbor);
 }return local;
}
function restSegmentTriangleDistance(a:ClothVector,b:ClothVector,triangle:readonly ClothVector[]):number{
 const [x,y,z]=triangle as readonly[ClothVector,ClothVector,ClothVector],normal=cross3(sub3(y,x),sub3(z,x)),da=dot3(sub3(a,x),normal),db=dot3(sub3(b,x),normal);
 if(da*db<=0&&Math.abs(da-db)>1e-30){const p=lerp3(a,b,da/(da-db));if(norm3(sub3(p,closestTrianglePoint(p,x,y,z).point))<1e-12)return 0;}
 let best=Math.min(norm3(sub3(a,closestTrianglePoint(a,x,y,z).point)),norm3(sub3(b,closestTrianglePoint(b,x,y,z).point)));
 for(let i=0;i<3;i++){const q=closestSegmentPoints(a,b,triangle[i]!,triangle[(i+1)%3]!);best=Math.min(best,norm3(sub3(q.a,q.b)));}return best;
}
type BodyResponses = Map<string, {
    linear: ClothVector;
    angular: ClothVector;
}>;
/** Sequential contact constraints are iterated to a physical displacement and
 * velocity tolerance. Failure rolls back the enclosing mechanical step. */
export function resolveClothContacts(topology: PreparedClothTopology, state: ClothState, oldPositions: Float64Array, dt: number, colliders: readonly ClothCollider[], selfCollision: boolean): ClothContactReport {
    const bodyResponses: BodyResponses = new Map(), reactions = new Map<string, ClothReaction>();
    let contacts = 0, selfContacts = 0, maximumPenetration_m = 0;
    for (let sweep = 0; sweep < 64; sweep++) {
        const before = state.positions_m.slice(), velocity = state.velocities_m_s.slice(), twist = state.twist_rad.slice(), angular = state.angularVelocity_rad_s.slice();
        const report = resolveClothContactsSweep(topology, state, sweep ? before : oldPositions, sweep ? 0 : dt, colliders, selfCollision, bodyResponses);
        contacts += report.contacts;
        selfContacts += report.selfContacts;
        maximumPenetration_m = Math.max(maximumPenetration_m, report.maximumPenetration_m);
        for (const r of report.reactions) {
            let sum = reactions.get(r.bodyId);
            if (!sum) {
                sum = { bodyId: r.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0] };
                reactions.set(r.bodyId, sum);
            }
            for (let a = 0; a < 3; a++) {
                sum.linearImpulse_N_s[a]! += r.linearImpulse_N_s[a]!;
                sum.angularImpulse_N_m_s[a]! += r.angularImpulse_N_m_s[a]!;
            }
        }
        let residual = 0;
        for (let i = 0; i < before.length; i++)
            residual = Math.max(residual, Math.abs(state.positions_m[i]! - before[i]!), dt * Math.abs(state.velocities_m_s[i]! - velocity[i]!));
        for (const rod of topology.rods) {
            const m=topology.materials[rod.material]!,radius=Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m);
            for (let j = 0; j < rod.restLengths_m.length; j++) {
                const s = rod.segmentOffset + j;
                residual = Math.max(residual, radius * Math.abs(state.twist_rad[s]! - twist[s]!), dt * radius * Math.abs(state.angularVelocity_rad_s[s]! - angular[s]!));
            }
        }
        if (residual < 1e-8)
            return { iterations: sweep + 1, residual_m: residual, contacts, selfContacts, maximumPenetration_m, reactions: [...reactions.values()] };
    }
    throw new Error('Cloth contact constraints did not converge within 64 sweeps');
}
function resolveClothContactsSweep(topology: PreparedClothTopology, state: ClothState, oldPositions: Float64Array, dt: number, colliders: readonly ClothCollider[], selfCollision: boolean, bodyResponses: BodyResponses): ClothContactReport {
    const p = state.positions_m, velocity = state.velocities_m_s, reactions = new Map<string, ClothReaction>();
    // Sequential contacts share the body's accumulated velocity response, while
    // its swept collision pose remains the accepted Box3D geometry for this step.
    const surfaceVelocity = (collider: ClothCollider, point: ClothVector): ClothVector => {
        const base = colliderVelocity(collider, point), response = collider.bodyId ? bodyResponses.get(collider.bodyId) : undefined;
        if (!response)
            return base;
        return add3(base, add3(response.linear, cross3(response.angular, sub3(point, collider.centreOfMass_m ?? [0, 0, 0]))));
    };
    let contacts = 0, selfContacts = 0, maximumPenetration = 0;
    const mass = (i: number) => state.dryMass_kg[i]! + state.waterVolume_m3[i]! * topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3;
    const radius = (i: number) => {
        const m = topology.materials[topology.vertexMaterial[i]!]!;
        return topology.incidentTriangles[i]!.length ? .5 * m.thickness_m : Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m);
    };
    const inverseMass = (i: number) => ((topology.kinematicFixed[i] ?? 0) & 1) ? 0 : 1 / mass(i);
    const rodEdges: ClothCollisionEdge[] = [], clippedVertices = new Set<number>(), clipMap = new Map<string, ClothCollisionEdge>();
    for (let r = 0; r < topology.rods.length; r++) {
        const rod = topology.rods[r]!;
        for (let segment = 0; segment < rod.vertices.length - 1; segment++) {
            const start = segment === 0 && norm3(point3(topology.data.rodStartClipNormals ?? [0, 0, 0], topology.data.rodStartClipNormals ? r : 0)) > 0;
            const end = segment === rod.vertices.length - 2 && norm3(point3(topology.data.rodEndClipNormals ?? [0, 0, 0], topology.data.rodEndClipNormals ? r : 0)) > 0;
            const vertices = [rod.vertices[segment]!, rod.vertices[segment + 1]!] as const;
            const m=topology.materials[rod.material]!;
            const edge: ClothCollisionEdge = { vertices, r: Math.max(m.rodRadius_m,m.rodRadiusB_m??m.rodRadius_m), elliptical:(m.rodRadiusB_m??m.rodRadius_m)!==m.rodRadius_m, min: [0, 0, 0], max: [0, 0, 0], rod: r, segment, clipped: !!(start || end) };
            rodEdges.push(edge);
            clipMap.set(`${Math.min(...vertices)}:${Math.max(...vertices)}`, edge);
            if (start)
                clippedVertices.add(vertices[0]);
            if (end)
                clippedVertices.add(vertices[1]);
        }
    }
    const scratch = p.slice(), oldContactTwist = Float64Array.from(state.twist_rad, (theta, i) => theta - dt * state.angularVelocity_rad_s[i]!), contactTwist = state.twist_rad.slice(), frames = { rodTwist_rad: contactTwist, rodDirectors: state.rodDirectors, rodTangents: state.rodTangents };
    const edgeShape = (edge: ClothCollisionEdge, time: number): ClothRodCapsuleGeometry => {
        const [a, b] = edge.vertices;
        const x = lerp3(point3(oldPositions, a), point3(p, a), time), y = lerp3(point3(oldPositions, b), point3(p, b), time);
        if (edge.rod===undefined)
            return { a: x, b: y, radius_m: edge.r, planes: [] };
        scratch.set(x, 3 * a);
        scratch.set(y, 3 * b);
        const s = topology.rods[edge.rod!]!.segmentOffset + edge.segment!;
        contactTwist[s] = oldContactTwist[s]! + time * (state.twist_rad[s]! - oldContactTwist[s]!);
        return clothRodCapsuleGeometry(topology, topology.data, scratch, edge.rod!, edge.segment!, frames);
    };
    const edgeWeights = (edge: ClothCollisionEdge, point: ClothVector, time: number) => { const a = lerp3(point3(oldPositions, edge.vertices[0]), point3(p, edge.vertices[0]), time), b = lerp3(point3(oldPositions, edge.vertices[1]), point3(p, edge.vertices[1]), time), e = sub3(b, a), t = Math.max(0, Math.min(1, dot3(sub3(point, a), e) / dot3(e, e))); return [1 - t, t]; };
    const edgeTravel = (edge: ClothCollisionEdge) => { const [a, b] = edge.vertices, u = sub3(point3(oldPositions, b), point3(oldPositions, a)), v = sub3(point3(p, b), point3(p, a)), length = Math.min(norm3(u), norm3(v)), travel = Math.max(norm3(sub3(point3(p, a), point3(oldPositions, a))), norm3(sub3(point3(p, b), point3(oldPositions, b)))); const spin = edge.clipped||edge.elliptical ? Math.abs(state.twist_rad[topology.rods[edge.rod!]!.segmentOffset + edge.segment!]! - oldContactTwist[topology.rods[edge.rod!]!.segmentOffset + edge.segment!]!) * edge.r : 0; return travel * (1 + 4 * edge.r / Math.max(length, 1e-12)) + spin; };
    for (let i = 0; i < topology.vertexCount; i++) {
        if (!inverseMass(i) || topology.incidentTriangles[i]!.length === 0)
            continue;
        for (const collider of colliders) {
            const old = point3(oldPositions, i), next = point3(p, i), delta = sub3(next, old), r = radius(i);
            const rotationSpeed = collider.kind === "rigid" ? Math.hypot(collider.body.angularVelocity_rad_s.x, collider.body.angularVelocity_rad_s.y, collider.body.angularVelocity_rad_s.z) * sceneShape(collider.body.description.shape).boundingRadius_m(collider.body.description.dimensions_m) : 0;
            const speed = (norm3(delta) + dt * (collider.kind === "sampled" ? collider.maximumSpeed_m_s : norm3(collider.velocity_m_s ?? [0, 0, 0]) + rotationSpeed))*(collider.kind==="sampled"?(collider.spatialLipschitz??2):1);
            let time = 0, hit = false, hitNormal: ClothVector = [0, 1, 0];
            const tolerance = Math.max(1e-10, r * 1e-6);
            let clear = false;
            for (let iteration = 0; iteration < 80; iteration++) {
                const at = lerp3(old, next, time), distance = clothColliderDistance(collider, at, (1 - time) * dt) - r;
                if (distance <= tolerance) {
                    hit = true;
                    hitNormal = sampleClothCollider(collider, at, (1 - time) * dt).normal;
                    break;
                }
                // speed bounds the clearance change over the whole sweep, so a
                // start clearance above it cannot reach the surface or end inside.
                if (iteration === 0 && distance - speed > tolerance) {
                    clear = true;
                    break;
                }
                if (time >= 1 || speed === 0)
                    break;
                time = Math.min(1, time + Math.max(1e-9, .9 * distance / speed));
            }
            if (clear)
                continue;
            const endDistance = clothColliderDistance(collider, next, 0), penetration = Math.max(0, r - endDistance);
            maximumPenetration = Math.max(maximumPenetration, penetration);
            if (!hit && penetration === 0)
                continue;
            contacts++;
            const normal = hit ? hitNormal : sampleClothCollider(collider, next).normal;
            // Retain tangential travel while removing travel through the obstacle.
            const impact = lerp3(old, next, time), obstacleRemaining = scale3(colliderVelocity(collider, impact), (1 - time) * dt), relativeRemaining = sub3(sub3(next, impact), obstacleRemaining);
            let corrected = add3(impact, add3(obstacleRemaining, sub3(relativeRemaining, scale3(normal, Math.min(0, dot3(relativeRemaining, normal))))));
            for (let projection = 0; projection < 4; projection++) {
                const sample = sampleClothCollider(collider, corrected);
                if (sample.distance_m >= r)
                    break;
                corrected = add3(corrected, scale3(sample.normal, r - sample.distance_m + 1e-10));
            }
            p.set(corrected, 3 * i);
            const before = point3(velocity, i), obstacleVelocity = surfaceVelocity(collider, corrected), relative = sub3(before, obstacleVelocity), normalSpeed = dot3(relative, normal);
            let after = before;
            if (normalSpeed < 0) {
                const tangent = sub3(relative, scale3(normal, normalSpeed)), tangentSpeed = norm3(tangent), friction = collider.friction ?? topology.materials[topology.vertexMaterial[i]!]!.friction;
                const invMass = inverseMass(i), normalImpulse = -normalSpeed / (invMass + colliderInverseEffectiveMass(collider, corrected, normal));
                const tangentDirection = tangentSpeed > 0 ? scale3(tangent, 1 / tangentSpeed) : [0, 0, 0] as ClothVector;
                const tangentImpulse = tangentSpeed > 0 ? Math.min(friction * normalImpulse, tangentSpeed / (invMass + colliderInverseEffectiveMass(collider, corrected, tangentDirection))) : 0;
                after = add3(before, scale3(sub3(scale3(normal, normalImpulse), scale3(tangentDirection, tangentImpulse)), invMass));
            }
            velocity.set(after, 3 * i);
            if (collider.bodyId) {
                let reaction = reactions.get(collider.bodyId);
                if (!reaction) {
                    reaction = { bodyId: collider.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0] };
                    reactions.set(collider.bodyId, reaction);
                }
                const impulse = scale3(sub3(before, after), mass(i)), torque = cross3(sub3(corrected, collider.centreOfMass_m ?? [0, 0, 0]), impulse);
                for (let axis = 0; axis < 3; axis++) {
                    reaction.linearImpulse_N_s[axis]! += impulse[axis]!;
                    reaction.angularImpulse_N_m_s[axis]! += torque[axis]!;
                }
                if (collider.kind === "rigid" && !collider.body.held && collider.body.description.motion !== "static") {
                    const b = collider.body, current = bodyResponses.get(collider.bodyId) ?? { linear: [0, 0, 0] as ClothVector, angular: [0, 0, 0] as ClothVector };
                    const local = quaternionInverseRotate(b.orientation, { x: torque[0], y: torque[1], z: torque[2] }), angular = quaternionRotate(b.orientation, { x: local.x * b.inverseInertiaBody_kg_m2.x, y: local.y * b.inverseInertiaBody_kg_m2.y, z: local.z * b.inverseInertiaBody_kg_m2.z });
                    bodyResponses.set(collider.bodyId, { linear: add3(current.linear, scale3(impulse, b.inverseMass_kg)), angular: add3(current.angular, [angular.x, angular.y, angular.z]) });
                }
            }
        }
    }
    // Cut end caps use their exact convex volume; the removed endpoint sphere
    // never participates in collision. Witness impulses distribute to the rod DOFs.
    for (const edge of rodEdges)
        for (const collider of colliders) {
            for (let contactIteration = 0; contactIteration < 32; contactIteration++) {
                const bodySpeed = collider.kind === 'sampled' ? collider.maximumSpeed_m_s : norm3(collider.velocity_m_s ?? [0, 0, 0]) + (collider.kind === 'rigid' ? Math.hypot(collider.body.angularVelocity_rad_s.x, collider.body.angularVelocity_rad_s.y, collider.body.angularVelocity_rad_s.z) * sceneShape(collider.body.description.shape).boundingRadius_m(collider.body.description.dimensions_m) : 0), travel = (edgeTravel(edge) + dt * bodySpeed)*(collider.kind==="sampled"?(collider.spatialLipschitz??2):1);
                let time = 0, hit: ReturnType<typeof sampleClippedRodCollider> | undefined;
                if (contactIteration) {
                    const q = sampleClippedRodCollider(edgeShape(edge, 1), collider, 0, sampleClothCollider);
                    if (q.distance_m >= -1e-9)
                        break;
                    hit = q;
                    time = 1;
                }
                for (let iteration = 0; !hit && iteration < 160; iteration++) {
                    const q = sampleClippedRodCollider(edgeShape(edge, time), collider, (1 - time) * dt, sampleClothCollider);
                    if (q.distance_m <= 1e-8) {
                        hit = q;
                        break;
                    }
                    if (time >= 1 || travel === 0)
                        break;
                    if (q.lowerBound_m <= 0)
                        throw new Error('Clipped yarn collider distance failed to separate');
                    time = Math.min(1, time + .9 * q.lowerBound_m / travel);
                    if (iteration === 159)
                        throw new Error('Clipped yarn collider CCD did not converge');
                }
                if (!hit)
                    break;
                const normal = hit.normal, weights = edgeWeights(edge, hit.point, time), ids = edge.vertices;
                for (const i of ids)
                    scratch.set(lerp3(point3(oldPositions, i), point3(p, i), time), 3 * i);
                const dof = rodContactDOF(topology, state, edge.rod!, edge.segment!, hit.point, scratch), inv = dof.inverseMass(normal);
                if (!inv)
                    break;
                const movement: ClothVector = [0, 1, 2].map(a => ids.reduce((sum, i, j) => sum + weights[j]! * (p[3 * i + a]! - oldPositions[3 * i + a]!), 0) * (1 - time)) as unknown as ClothVector;
                const obstacle = scale3(colliderVelocity(collider, hit.point), (1 - time) * dt), penetration = Math.max(0, -hit.distance_m - dot3(sub3(movement, obstacle), normal));
                maximumPenetration = Math.max(maximumPenetration, penetration);
                contacts++;
                dof.applyCorrection(scale3(normal, penetration / inv));
                const relative: ClothVector = sub3(dof.velocity(), surfaceVelocity(collider, hit.point)), closing = dot3(relative, normal);
                if (closing >= 0)
                    continue;
                const impulseN = -closing / (inv + colliderInverseEffectiveMass(collider, hit.point, normal)), tangent = sub3(relative, scale3(normal, closing)), speed = norm3(tangent), direction = scale3(tangent, speed ? 1 / speed : 0), friction = collider.friction ?? topology.materials[topology.rods[edge.rod!]!.material]!.friction, impulseT = speed ? Math.min(friction * impulseN, speed / (dof.inverseMass(direction) + colliderInverseEffectiveMass(collider, hit.point, direction))) : 0, impulse = sub3(scale3(normal, impulseN), scale3(direction, impulseT));
                dof.applyImpulse(impulse);
                if (collider.bodyId) {
                    const opposite = scale3(impulse, -1), torque = cross3(sub3(hit.point, collider.centreOfMass_m ?? [0, 0, 0]), opposite);
                    let reaction = reactions.get(collider.bodyId);
                    if (!reaction) {
                        reaction = { bodyId: collider.bodyId, linearImpulse_N_s: [0, 0, 0], angularImpulse_N_m_s: [0, 0, 0] };
                        reactions.set(collider.bodyId, reaction);
                    }
                    for (let a = 0; a < 3; a++) {
                        reaction.linearImpulse_N_s[a]! += opposite[a]!;
                        reaction.angularImpulse_N_m_s[a]! += torque[a]!;
                    }
                    if (collider.kind === 'rigid' && !collider.body.held && collider.body.description.motion !== 'static') {
                        const b = collider.body, current = bodyResponses.get(collider.bodyId) ?? { linear: [0, 0, 0] as ClothVector, angular: [0, 0, 0] as ClothVector }, local = quaternionInverseRotate(b.orientation, { x: torque[0], y: torque[1], z: torque[2] }), angular = quaternionRotate(b.orientation, { x: local.x * b.inverseInertiaBody_kg_m2.x, y: local.y * b.inverseInertiaBody_kg_m2.y, z: local.z * b.inverseInertiaBody_kg_m2.z });
                        bodyResponses.set(collider.bodyId, { linear: add3(current.linear, scale3(opposite, b.inverseMass_kg)), angular: add3(current.angular, [angular.x, angular.y, angular.z]) });
                    }
                }
            }
            if (sampleClippedRodCollider(edgeShape(edge, 1), collider, 0, sampleClothCollider).distance_m < -1e-8 && edge.vertices.some(i => inverseMass(i) > 0))
                throw new Error('Rod rigid contact projection did not converge');
        }
    const contactDOF = (ids: readonly number[], weights: readonly number[], point: ClothVector, time: number, edge?: ClothCollisionEdge) => {
        if (edge?.rod !== undefined) {
            for (const i of edge.vertices)
                scratch.set(lerp3(point3(oldPositions, i), point3(p, i), time), 3 * i);
            return rodContactDOF(topology, state, edge.rod, edge.segment!, point, scratch);
        }
        const inv = ids.reduce((sum, i, j) => sum + weights[j]! ** 2 * inverseMass(i), 0), apply = (impulse: ClothVector, positions: boolean) => { const values = positions ? p : velocity; for (let j = 0; j < ids.length; j++)
            for (let a = 0; a < 3; a++)
                values[3 * ids[j]! + a]! += weights[j]! * inverseMass(ids[j]!) * impulse[a]!; };
        return { inverseMass: (_direction: ClothVector) => inv, velocity: () => [0, 1, 2].map(a => ids.reduce((sum, i, j) => sum + weights[j]! * velocity[3 * i + a]!, 0)) as unknown as ClothVector, applyImpulse: (j: ClothVector) => apply(j, false), applyCorrection: (j: ClothVector) => apply(j, true) };
    };
    if (selfCollision) {
        // Reference CCD: conservative advancement bounds the relative travel of a
        // vertex and all three triangle vertices. No sampled endpoint-only test.
        let cache = contactTrees.get(topology);
        if (!cache) {
            cache = { faceBounds: new Float64Array(topology.triangles.length * 6) };
            contactTrees.set(topology, cache);
        }
        const faceBounds = cache.faceBounds;
        for (let f = 0; f < topology.triangles.length; f++) {
            const face = topology.triangles[f]!, r = .5 * topology.materials[face.material]!.thickness_m;
            for (let a = 0; a < 3; a++) {
                let lo = Infinity, hi = -Infinity;
                for (const v of face.vertices) {
                    lo = Math.min(lo, oldPositions[3 * v + a]!, p[3 * v + a]!);
                    hi = Math.max(hi, oldPositions[3 * v + a]!, p[3 * v + a]!);
                }
                faceBounds[6 * f + a] = lo - r;
                faceBounds[6 * f + a + 3] = hi + r;
            }
        }
        if (cache.faceTree)
            cache.faceTree.refit();
        else
            cache.faceTree = new ClothContactBVH(faceBounds);
        const faceTree = cache.faceTree;
        cache.vertexLocalFaces??=new Map();
        const localVertexFace=(i:number,f:number)=>{let local=cache!.vertexLocalFaces!.get(i);if(!local){const p=point3(topology.data.restPositions_m,i);local=connectedRestFaces(topology,topology.incidentTriangles[i]!,radius(i),triangle=>norm3(sub3(p,closestTrianglePoint(p,...triangle as [ClothVector,ClothVector,ClothVector]).point)));cache!.vertexLocalFaces!.set(i,local);}return local.has(f);};
        const nearbyFaces = (i: number) => {
            const r = radius(i), min = [0, 1, 2].map(a => Math.min(oldPositions[3 * i + a]!, p[3 * i + a]!) - r), max = [0, 1, 2].map(a => Math.max(oldPositions[3 * i + a]!, p[3 * i + a]!) + r);
            return faceTree.query(min, max);
        };
        for (let i = 0; i < topology.vertexCount; i++)
            for (const faceIndex of nearbyFaces(i)) {
                if (topology.incidentTriangles[i]!.length === 0)
                    continue;
                const face = topology.triangles[faceIndex]!;
                if (face.vertices.includes(i) || face.vertices.some(v => topology.incidentTriangles[i]!.some(f => topology.triangles[f]!.vertices.includes(v))))
                    continue;
                if(localVertexFace(i,faceIndex))continue;
                const ids = face.vertices, p0 = point3(oldPositions, i), p1 = point3(p, i), triangleOld = ids.map(v => point3(oldPositions, v)), triangleNew = ids.map(v => point3(p, v));
                const distance = radius(i) + .5 * topology.materials[face.material]!.thickness_m;
                // Swept AABB rejects distant pairs before the iterative CCD query.
                if ([0, 1, 2].some(a => Math.max(p0[a]!, p1[a]!) + distance < Math.min(...triangleOld.map(q => q[a]!), ...triangleNew.map(q => q[a]!)) || Math.min(p0[a]!, p1[a]!) - distance > Math.max(...triangleOld.map(q => q[a]!), ...triangleNew.map(q => q[a]!))))
                    continue;
                const bound = norm3(sub3(p1, p0)) + Math.max(...triangleOld.map((q, j) => norm3(sub3(triangleNew[j]!, q))));
                let time = 0, found = false, normal: ClothVector = [0, 1, 0], weights: ClothVector = [1, 0, 0];
                for (let iteration = 0; iteration < 100; iteration++) {
                    const v = lerp3(p0, p1, time), [a, b, c] = triangleOld.map((q, j) => lerp3(q, triangleNew[j]!, time)) as unknown as [
                        ClothVector,
                        ClothVector,
                        ClothVector
                    ];
                    const closest = closestTrianglePoint(v, a, b, c), delta = sub3(v, closest.point), gap = norm3(delta);
                    if (gap <= distance + 1e-9) {
                        found = true;
                        weights = closest.weights;
                        const n = cross3(sub3(b, a), sub3(c, a));
                        normal = gap > 1e-12 ? scale3(delta, 1 / gap) : scale3(n, 1 / Math.max(1e-20, norm3(n)));
                        break;
                    }
                    if (time >= 1 || bound === 0)
                        break;
                    time = Math.min(1, time + Math.max(1e-10, .9 * (gap - distance) / bound));
                }
                if (!found)
                    continue;
                const inv = inverseMass(i) + ids.reduce((sum, v, j) => sum + weights[j]! ** 2 * inverseMass(v), 0);
                if (!inv)
                    continue;
                const facePoint: ClothVector = [0, 1, 2].map(a => ids.reduce((sum, v, j) => sum + weights[j]! * p[3 * v + a]!, 0)) as unknown as ClothVector;
                const penetration = Math.max(0, distance - dot3(sub3(point3(p, i), facePoint), normal));
                if (!penetration)
                    continue;
                maximumPenetration = Math.max(maximumPenetration, penetration);
                selfContacts++;
                const correction = penetration / inv;
                for (let axis = 0; axis < 3; axis++) {
                    p[3 * i + axis]! += inverseMass(i) * correction * normal[axis]!;
                    for (let j = 0; j < 3; j++)
                        p[3 * ids[j]! + axis]! -= weights[j]! * inverseMass(ids[j]!) * correction * normal[axis]!;
                }
                const faceVelocity: ClothVector = [0, 1, 2].map(a => ids.reduce((sum, v, j) => sum + weights[j]! * velocity[3 * v + a]!, 0)) as unknown as ClothVector;
                const relative = sub3(point3(velocity, i), faceVelocity), closing = dot3(relative, normal);
                if (closing < 0) {
                    const tangent = sub3(relative, scale3(normal, closing)), speed = norm3(tangent), friction = Math.min(...[i, ...ids].map(v => topology.materials[topology.vertexMaterial[v]!]!.friction));
                    const impulse = add3(scale3(normal, -closing / inv), scale3(tangent, speed ? -Math.min(speed, friction * (-closing)) / (speed * inv) : 0));
                    for (let axis = 0; axis < 3; axis++) {
                        velocity[3 * i + axis]! += inverseMass(i) * impulse[axis]!;
                        for (let j = 0; j < 3; j++)
                            velocity[3 * ids[j]! + axis]! -= weights[j]! * inverseMass(ids[j]!) * impulse[axis]!;
                    }
                }
            }
        // Vertex/face queries alone miss two long edges crossing away from their
        // endpoints. This same finite-radius segment path handles rod/rod and
        // rod/cloth-edge contact. Sorted swept bounds avoid all-pairs far queries.
        if (!cache.edges) {
            const edgeMap = new Map<string, readonly [
                number,
                number
            ]>();
            const addEdge = (a: number, b: number) => edgeMap.set(`${Math.min(a, b)}:${Math.max(a, b)}`, [a, b]);
            for (const f of topology.triangles)
                for (let j = 0; j < 3; j++)
                    addEdge(f.vertices[j]!, f.vertices[(j + 1) % 3]!);
            for (const rod of topology.rods)
                for (let j = 0; j < rod.vertices.length - 1; j++)
                    addEdge(rod.vertices[j]!, rod.vertices[j + 1]!);
            cache.edges = [...edgeMap.values()].map(vertices => (clipMap.get(`${Math.min(...vertices)}:${Math.max(...vertices)}`) ?? { vertices, r: Math.max(...vertices.map(radius)), min: [0, 0, 0], max: [0, 0, 0] }));
            cache.edgeBounds = new Float64Array(cache.edges.length * 6);
        }
        const edges = cache.edges, edgeBounds = cache.edgeBounds!;
        for (let i = 0; i < edges.length; i++) {
            const edge = edges[i]!, [a, b] = edge.vertices;
            const shape = edgeShape(edge, 1), pad = edge.r + (edge.clipped ? Math.max(norm3(sub3(shape.a, point3(p, a))), norm3(sub3(shape.b, point3(p, b)))) : 0);
            for (let axis = 0; axis < 3; axis++) {
                edge.min[axis] = Math.min(oldPositions[3 * a + axis]!, oldPositions[3 * b + axis]!, p[3 * a + axis]!, p[3 * b + axis]!) - pad;
                edge.max[axis] = Math.max(oldPositions[3 * a + axis]!, oldPositions[3 * b + axis]!, p[3 * a + axis]!, p[3 * b + axis]!) + pad;
            }
            edgeBounds.set(edge.min, 6 * i);
            edgeBounds.set(edge.max, 6 * i + 3);
        }
        if (cache.edgeTree)
            cache.edgeTree.refit();
        else
            cache.edgeTree = new ClothContactBVH(edgeBounds);
        const edgeTree = cache.edgeTree;
        const localShellEdges=(a:ClothCollisionEdge,b:ClothCollisionEdge)=>{
            if(a.rod!==undefined||b.rod!==undefined)return false;
            if(a.vertices.some(v=>b.vertices.some(w=>topology.incidentTriangles[w]!.some(f=>cache!.vertexLocalFaces!.get(v)?.has(f)))))return true;
            const oldA=a.vertices.map(v=>point3(topology.data.restPositions_m,v)),oldB=b.vertices.map(v=>point3(topology.data.restPositions_m,v)),nearest=closestSegmentPoints(oldA[0]!,oldA[1]!,oldB[0]!,oldB[1]!);
            if(norm3(sub3(nearest.a,nearest.b))>a.r+b.r+1e-12)return false;
            if(!a.restLocalFaces){const [x,y]=a.vertices.map(v=>point3(topology.data.restPositions_m,v));a.restLocalFaces=connectedRestFaces(topology,a.vertices.flatMap(v=>topology.incidentTriangles[v]!),a.r,triangle=>restSegmentTriangleDistance(x!,y!,triangle));}
            return b.vertices.some(v=>topology.incidentTriangles[v]!.some(f=>a.restLocalFaces!.has(f)));
        };
        for (let ai = 0; ai < edges.length; ai++)
            for (const bi of edgeTree.query(edges[ai]!.min, edges[ai]!.max)) {
                if (bi <= ai)
                    continue;
                const ea = edges[ai]!, eb = edges[bi]!, ids = [...ea.vertices, ...eb.vertices];
                if (ea.clipped || eb.clipped || ea.elliptical || eb.elliptical)
                    continue;
                if (new Set(ids).size < 4 || [1, 2].some(a => ea.max[a]! < eb.min[a]! || eb.max[a]! < ea.min[a]!))
                    continue;
                if (ea.vertices.some(a => eb.vertices.some(b => topology.incidentTriangles[a]!.some(f => topology.triangles[f]!.vertices.includes(b)))))
                    continue;
                if(localShellEdges(ea,eb))continue;
                const old = ids.map(i => point3(oldPositions, i)), next = ids.map(i => point3(p, i)), thickness = ea.r + eb.r;
                const travel = Math.max(norm3(sub3(next[0]!, old[0]!)), norm3(sub3(next[1]!, old[1]!))) + Math.max(norm3(sub3(next[2]!, old[2]!)), norm3(sub3(next[3]!, old[3]!)));
                let time = 0, found = false, normal: ClothVector = [0, 1, 0], s = 0, t = 0;
                for (let it = 0; it < 100; it++) {
                    const [a, b, c, d] = old.map((q, j) => lerp3(q, next[j]!, time)) as unknown as [
                        ClothVector,
                        ClothVector,
                        ClothVector,
                        ClothVector
                    ];
                    const q = closestSegmentPoints(a, b, c, d), delta = sub3(q.a, q.b), gap = norm3(delta);
                    if (gap <= thickness + 1e-9) {
                        found = true;
                        s = q.s;
                        t = q.t;
                        const fallback = cross3(sub3(b, a), sub3(d, c));
                        normal = gap > 1e-12 ? scale3(delta, 1 / gap) : norm3(fallback) > 1e-12 ? scale3(fallback, 1 / norm3(fallback)) : [0, 1, 0];
                        break;
                    }
                    if (time >= 1 || travel === 0)
                        break;
                    time = Math.min(1, time + Math.max(1e-10, .9 * (gap - thickness) / travel));
                }
                if (!found)
                    continue;
                const weights = [1 - s, s, -(1 - t), -t];
                const qa = lerp3(lerp3(old[0]!, next[0]!, time), lerp3(old[1]!, next[1]!, time), s), qb = lerp3(lerp3(old[2]!, next[2]!, time), lerp3(old[3]!, next[3]!, time), t), da = contactDOF(ea.vertices, [1 - s, s], sub3(qa, scale3(normal, ea.r)), time, ea), db = contactDOF(eb.vertices, [1 - t, t], add3(qb, scale3(normal, eb.r)), time, eb), inv = da.inverseMass(normal) + db.inverseMass(normal);
                if (!inv)
                    continue;
                const separation: ClothVector = [0, 1, 2].map(a => ids.reduce((sum, i, j) => sum + weights[j]! * p[3 * i + a]!, 0)) as unknown as ClothVector;
                const penetration = Math.max(0, thickness - dot3(separation, normal));
                if (!penetration)
                    continue;
                maximumPenetration = Math.max(maximumPenetration, penetration);
                selfContacts++;
                da.applyCorrection(scale3(normal, penetration / inv));
                db.applyCorrection(scale3(normal, -penetration / inv));
                const relative = sub3(da.velocity(), db.velocity()), normalSpeed = dot3(relative, normal);
                if (normalSpeed < 0) {
                    const tangent = sub3(relative, scale3(normal, normalSpeed)), speed = norm3(tangent), direction = scale3(tangent, speed ? 1 / speed : 0), mu = Math.min(...ids.map(i => topology.materials[topology.vertexMaterial[i]!]!.friction)), normalImpulse = -normalSpeed / inv, tangentImpulse = speed ? Math.min(mu * normalImpulse, speed / (da.inverseMass(direction) + db.inverseMass(direction))) : 0, impulse = sub3(scale3(normal, normalImpulse), scale3(direction, tangentImpulse));
                    da.applyImpulse(impulse);
                    db.applyImpulse(scale3(impulse, -1));
                }
            }
        const resolveConvexPair = (ids: readonly number[], weightsAt: (a: ClothVector, b: ClothVector, t: number) => number[], supportsAt: (t: number) => readonly [
            (d: ClothVector) => ClothVector,
            (d: ClothVector) => ClothVector
        ], travel: number, firstEdge: ClothCollisionEdge, secondEdge?: ClothCollisionEdge) => {
            let time = 0, hit: ReturnType<typeof convexClothDistance> | undefined;
            for (let iteration = 0; iteration < 160; iteration++) {
                const [a, b] = supportsAt(time), q = convexClothDistance(a, b);
                if (q.distance_m <= 1e-8) {
                    hit = q;
                    break;
                }
                if (time >= 1 || travel === 0)
                    break;
                if (q.lowerBound_m <= 0)
                    throw new Error(`Clipped yarn self-contact distance failed to separate (${q.distance_m}, ${q.lowerBound_m}, converged=${q.converged})`);
                time = Math.min(1, time + .9 * q.lowerBound_m / travel);
                if (iteration === 159)
                    throw new Error('Clipped yarn self-contact CCD did not converge');
            }
            if (!hit)
                return;
            const weights = weightsAt(hit.a, hit.b, time), normal = hit.normal, da = contactDOF(ids.slice(0, 2), weights.slice(0, 2), hit.a, time, firstEdge), db = contactDOF(ids.slice(2), weights.slice(2).map(w => -w), hit.b, time, secondEdge), inv = da.inverseMass(normal) + db.inverseMass(normal);
            if (!inv)
                return;
            const [a, b] = supportsAt(1), penetration = Math.max(0, -dot3(sub3(a(scale3(normal, -1)), b(normal)), normal));
            if (!penetration)
                return;
            maximumPenetration = Math.max(maximumPenetration, penetration);
            selfContacts++;
            da.applyCorrection(scale3(normal, penetration / inv));
            db.applyCorrection(scale3(normal, -penetration / inv));
            const relative = sub3(da.velocity(), db.velocity()), closing = dot3(relative, normal);
            if (closing >= 0)
                return;
            const tangent = sub3(relative, scale3(normal, closing)), speed = norm3(tangent), direction = scale3(tangent, speed ? 1 / speed : 0), friction = Math.min(...ids.map(i => topology.materials[topology.vertexMaterial[i]!]!.friction)), normalImpulse = -closing / inv, tangentImpulse = speed ? Math.min(friction * normalImpulse, speed / (da.inverseMass(direction) + db.inverseMass(direction))) : 0, impulse = sub3(scale3(normal, normalImpulse), scale3(direction, tangentImpulse));
            da.applyImpulse(impulse);
            db.applyImpulse(scale3(impulse, -1));
        };
        for (let ai = 0; ai < edges.length; ai++)
            for (const bi of edgeTree.query(edges[ai]!.min, edges[ai]!.max)) {
                if (bi <= ai)
                    continue;
                const a = edges[ai]!, b = edges[bi]!, ids = [...a.vertices, ...b.vertices];
                if (!(a.clipped || b.clipped || a.elliptical || b.elliptical) || new Set(ids).size < 4)
                    continue;
                if (a.vertices.some(i => b.vertices.some(j => topology.incidentTriangles[i]!.some(f => topology.triangles[f]!.vertices.includes(j)))))
                    continue;
                resolveConvexPair(ids, (x, y, t) => [...edgeWeights(a, x, t), ...edgeWeights(b, y, t).map(v => -v)], t => { const sa = edgeShape(a, t), sb = edgeShape(b, t); return [d => supportClippedRod(sa, d), d => supportClippedRod(sb, d)]; }, edgeTravel(a) + edgeTravel(b), a, b);
            }
        // A finite rod or cut cap can hit a triangle interior away from every edge.
        for (const edge of edges.filter(e => e.rod!==undefined))
            for (const f of faceTree.query(edge.min, edge.max)) {
                const face = topology.triangles[f]!;
                if (face.vertices.some(i => edge.vertices.includes(i)||edge.vertices.some(v=>topology.incidentTriangles[v]!.some(f=>topology.triangles[f]!.vertices.includes(i)))))
                    continue;
                const radius = .5 * topology.materials[face.material]!.thickness_m, triangle = (t: number) => face.vertices.map(i => lerp3(point3(oldPositions, i), point3(p, i), t));
                const travel = edgeTravel(edge) + Math.max(...face.vertices.map(i => norm3(sub3(point3(p, i), point3(oldPositions, i)))));
                resolveConvexPair([...edge.vertices, ...face.vertices], (a, b, t) => { const q = triangle(t); return [...edgeWeights(edge, a, t), ...closestTrianglePoint(b, q[0]!, q[1]!, q[2]!).weights.map(w => -w)]; }, t => { const shape = edgeShape(edge, t), q = triangle(t); return [d => supportClippedRod(shape, d), d => { const vertex = q.reduce((a, b) => dot3(a, d) >= dot3(b, d) ? a : b); return add3(vertex, scale3(d, radius / norm3(d))); }]; }, travel, edge);
            }
    }
    return { contacts, selfContacts, maximumPenetration_m: maximumPenetration, reactions: [...reactions.values()] };
}
