import {rodSegmentInertia_kg_m2} from './material';
import type { ClothState, ClothVector } from './types';
import { cross3, dot3, norm3, point3, scale3, sub3, type PreparedClothTopology } from './topology';
const add = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
export interface RodContactDOF {
    readonly segment: number;
    readonly vertices: readonly [
        number,
        number
    ];
    inverseMass(direction: ClothVector): number;
    velocity(): ClothVector;
    applyImpulse(impulse: ClothVector): void;
    applyCorrection(correction: ClothVector): void;
}
/** Work-conjugate Jacobian of a material point on a DER segment. The endpoint
 * force couple carries the transverse moment; the independent material angle
 * carries the axial torque. Thus a radial friction impulse conserves both
 * linear and angular momentum, including its equal/opposite rigid reaction. */
export function rodContactDOF(topology: PreparedClothTopology, state: ClothState, rodIndex: number, localSegment: number, point: ClothVector, positions: ArrayLike<number> = state.positions_m): RodContactDOF {
    const rod = topology.rods[rodIndex]!, material = topology.materials[rod.material]!, a = rod.vertices[localSegment]!, b = rod.vertices[localSegment + 1]!, segment = rod.segmentOffset + localSegment, x = point3(positions, a), y = point3(positions, b), e = sub3(y, x), length = norm3(e), tangent = scale3(e, 1 / length), t = Math.max(0, Math.min(1, dot3(sub3(point, x), e) / (length * length))), center = add(x, scale3(e, t)), arm = sub3(point, center);
    const inv = (i: number) => ((topology.kinematicFixed[i] ?? 0) & 1) ? 0 : 1 / (state.dryMass_kg[i]! + state.waterVolume_m3[i]! * topology.materials[topology.vertexMaterial[i]!]!.waterDensity_kg_m3);
    const inertia = rodSegmentInertia_kg_m2(topology,state,rodIndex,localSegment), invTwist = ((topology.kinematicFixed[a] ?? 0) | (topology.kinematicFixed[b] ?? 0)) & 2 ? 0 : 1 / inertia;
    const jacobian = (impulse: ClothVector) => { const torque = cross3(arm, impulse), spin = dot3(tangent, torque), couple = scale3(cross3(tangent, torque), 1 / length); return { a: add(scale3(impulse, 1 - t), couple), b: sub3(scale3(impulse, t), couple), spin }; };
    const apply = (impulse: ClothVector, position: boolean) => { const j = jacobian(impulse), array = position ? state.positions_m : state.velocities_m_s; for (let k = 0; k < 3; k++) {
        array[3 * a + k]! += inv(a) * j.a[k]!;
        array[3 * b + k]! += inv(b) * j.b[k]!;
    } if (position)
        state.twist_rad[segment]! += invTwist * j.spin;
    else
        state.angularVelocity_rad_s[segment]! += invTwist * j.spin; };
    return { segment, vertices: [a, b], inverseMass: d => { const j = jacobian(d); return inv(a) * dot3(j.a, j.a) + inv(b) * dot3(j.b, j.b) + invTwist * j.spin * j.spin; }, velocity: () => { const va = point3(state.velocities_m_s, a), vb = point3(state.velocities_m_s, b), omega = add(scale3(cross3(tangent, sub3(vb, va)), 1 / length), scale3(tangent, state.angularVelocity_rad_s[segment]!)); return add(add(scale3(va, 1 - t), scale3(vb, t)), cross3(omega, arm)); }, applyImpulse: j => apply(j, false), applyCorrection: j => apply(j, true) };
}
