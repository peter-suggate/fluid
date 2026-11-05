import type { ClothCollider } from "./contact";
import { sourceFieldDistance } from "./source-fields";
import { sourceFieldBounds, type SourceQuaternion } from "./source-motion";
import type { WetClothSourceField } from "./source-types";
import type { ClothVector } from "./types";
import { cross3, norm3, scale3, sub3 } from "./topology";
function quaternion(f: WetClothSourceField): SourceQuaternion {
    const a = f.axis ?? [0, 1, 0], angle = f.angle_rad ?? 0, n = norm3(a), s = n ? Math.sin(angle / 2) / n : 0;
    return [a[0] * s, a[1] * s, a[2] * s, Math.cos(angle / 2)];
}
function delta(a: SourceQuaternion, b: SourceQuaternion): SourceQuaternion {
    const q: SourceQuaternion = [-b[3] * a[0] + b[0] * a[3] - b[1] * a[2] + b[2] * a[1], -b[3] * a[1] + b[0] * a[2] + b[1] * a[3] - b[2] * a[0], -b[3] * a[2] - b[0] * a[1] + b[1] * a[0] + b[2] * a[3], b[3] * a[3] + b[0] * a[0] + b[1] * a[1] + b[2] * a[2]];
    return q[3] < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q;
}
/** Same authored source SDFs as fluid seeding, swept between accepted script
 * poses. Primitive motion is rigid; intersection roots share child velocities. */
export function sourceClothColliders(before: readonly WetClothSourceField[], after: readonly WetClothSourceField[], dt: number,excludedRootIndices:readonly number[]=[]): ClothCollider[] {
    const motions = after.map(f => {
        const old = before.find(v => v.index === f.index)!;
        const a = quaternion(old), b = quaternion(f), rotation = delta(a, b), n = Math.hypot(rotation[0], rotation[1], rotation[2]);
        const omega = n > 1e-14 ? scale3([rotation[0], rotation[1], rotation[2]], 2 * Math.atan2(n, rotation[3]) / (dt * n)) : [0, 0, 0] as ClothVector;
        const linear = scale3(sub3(f.center_m ?? [0, 0, 0], old.center_m ?? [0, 0, 0]), 1 / dt);
        const bounds = f.kind === "intersect" || f.kind === "file" ? undefined : sourceFieldBounds(after, f.index);
        return { f, old, a, b, omega, linear, radius: bounds ? .5 * norm3(sub3(bounds.max, bounds.min)) : 0 };
    });
    const atTime = (t: number): WetClothSourceField[] => motions.map(({ f, old, a, b }) => {
        if (f.kind === "intersect" || f.kind === "file")
            return f;
        let cosine = a.reduce((sum, v, i) => sum + v * b[i]!, 0), sign = 1;
        if (cosine < 0) {
            cosine = -cosine;
            sign = -1;
        }
        const angle = Math.acos(Math.min(1, cosine)), sine = Math.sin(angle);
        const w0 = sine > 1e-9 ? Math.sin((1 - t) * angle) / sine : 1 - t, w1 = sine > 1e-9 ? Math.sin(t * angle) / sine : t;
        const q = a.map((v, i) => v * w0 + sign * b[i]! * w1), n = Math.hypot(q[0]!, q[1]!, q[2]!);
        return { ...f, center_m: f.center_m!.map((v, i) => old.center_m![i]! * (1 - t) + v * t) as unknown as ClothVector,
            axis: n > 1e-12 ? [q[0]! / n, q[1]! / n, q[2]! / n] : [0, 1, 0], angle_rad: 2 * Math.atan2(n, q[3]!) };
    });
    return after.filter(f => f.root && f.usage === "solid"&&!excludedRootIndices.includes(f.index)).map(root => {
        const descendants = new Set<number>();
        const visit = (id: number) => { if (descendants.has(id))
            return; descendants.add(id); for (const child of after.find(f => f.index === id)?.children ?? [])
            visit(child); };
        visit(root.index);
        const primitives = motions.filter(m => descendants.has(m.f.index) && m.f.kind !== "intersect");
        const maximumSpeed = Math.max(0, ...primitives.map(m => norm3(m.linear) + norm3(m.omega) * m.radius));
        return { kind: "sampled", maximumSpeed_m_s: maximumSpeed, spatialLipschitz:1,
            sample: (p: ClothVector, beforeEnd_s: number) => {
                const fields = beforeEnd_s === 0 ? after : atTime(Math.max(0, Math.min(1, 1 - beforeEnd_s / dt)));
                const distance_m = sourceFieldDistance(fields, root.index, p), h = 1e-7;
                const gradient = [0, 1, 2].map(i => { const a = [...p] as [
                    number,
                    number,
                    number
                ], b = [...p] as [
                    number,
                    number,
                    number
                ]; a[i]! += h; b[i]! -= h; return (sourceFieldDistance(fields, root.index, a) - sourceFieldDistance(fields, root.index, b)) / (2 * h); }) as [
                    number,
                    number,
                    number
                ];
                const length = norm3(gradient);
                return { distance_m, normal: length > 1e-12 ? scale3(gradient, 1 / length) : [0, 1, 0] as ClothVector };
            },
            velocityAt: (p: ClothVector) => {
                const m = primitives.reduce((best, next) => Math.abs(sourceFieldDistance(after, next.f.index, p)) < Math.abs(sourceFieldDistance(after, best.f.index, p)) ? next : best);
                const angular = cross3(m.omega, sub3(p, m.f.center_m ?? [0, 0, 0]));
                return [m.linear[0] + angular[0], m.linear[1] + angular[1], m.linear[2] + angular[2]];
            } } as ClothCollider;
    });
}
