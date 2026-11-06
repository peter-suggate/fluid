// Source script semantics ported from libWetCloth (2018), Fei, Batty, Grinspun, Zheng.
// This file is subject to MPL-2.0; see public/assets/wetcloth/LICENSE.txt.
import type { ClothSupportScript, ClothTopologyData, ClothVector } from './types';
import type { WetClothSourceField } from './source-types';
import { sourceMeshBounds } from './source-mesh';

export type SourceQuaternion = readonly [number, number, number, number]; // xyzw
const identity: SourceQuaternion = [0, 0, 0, 1];
const add = (a: ClothVector, b: ClothVector): ClothVector => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a: ClothVector, b: ClothVector): ClothVector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const conjugate = (q: SourceQuaternion): SourceQuaternion => [-q[0], -q[1], -q[2], q[3]];
const multiply = (a: SourceQuaternion, b: SourceQuaternion): SourceQuaternion => [
  a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
  a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
  a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
  a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
];
const rotate = (q: SourceQuaternion, v: ClothVector): ClothVector => multiply(multiply(q, [...v, 0]), conjugate(q)).slice(0, 3) as unknown as ClothVector;
function quaternion(axis: ClothVector, angle: number): SourceQuaternion {
  if (angle === 0) return identity;
  const n = Math.hypot(...axis);
  if (!(n > 0)) throw new Error('Script rotation needs a nonzero axis');
  const s = Math.sin(angle / 2) / n;
  return [axis[0] * s, axis[1] * s, axis[2] * s, Math.cos(angle / 2)];
}
function axisAngle(q: SourceQuaternion): { axis: ClothVector; angle_rad: number } {
  const n = Math.hypot(q[0], q[1], q[2]);
  return n < 1e-15 ? {axis: [0, 1, 0], angle_rad: 0} : {axis: [q[0] / n, q[1] / n, q[2] / n], angle_rad: 2 * Math.atan2(n, q[3])};
}

/** Unit-displacement velocity, including original cubic acceleration ramps. */
export function sourceCubicVelocity(script: ClothSupportScript, t: number): number {
  const t0 = script.start_s, t1 = script.end_s;
  const ta = t0 + (script.easeIn_s ?? (t1 - t0) / 3), tb = t1 - (script.easeOut_s ?? (t1 - t0) / 3);
  if (!(t1 > t0) || ta > tb || ta < t0 || tb > t1) throw new Error('Invalid source script easing');
  if (script.easing && script.easing !== 'cubic') throw new Error('Source runtime supports cubic scripts only');
  if (t < t0 || t > t1) return 0;
  const peak = 2 / (t1 - t0 + tb - ta);
  if (t < ta) { const u = (t - t0) / (ta - t0); return peak * u * u * (3 - 2 * u); }
  if (t > tb) { const u = (t1 - t) / (t1 - tb); return peak * u * u * (3 - 2 * u); }
  return peak;
}

export interface SourceBounds { readonly min: ClothVector; readonly max: ClothVector }
/** Original primitive bounds, including rounded corners; useful for group pivots. */
export function sourceFieldBounds(fields: readonly WetClothSourceField[], index: number, visited = new Set<number>()): SourceBounds {
  const f = fields.find(f => f.index === index);
  if (!f || visited.has(index)) throw new Error('Invalid source bounds graph');
  if (f.kind === 'file') return sourceMeshBounds(f);
  if (f.kind === 'intersect') {
    const next = new Set(visited).add(index), b = f.children!.map(i => sourceFieldBounds(fields, i, next));
    return {min: [0, 1, 2].map(i => Math.max(...b.map(v => v.min[i]!))) as unknown as ClothVector,
      max: [0, 1, 2].map(i => Math.min(...b.map(v => v.max[i]!))) as unknown as ClothVector};
  }
  const c = f.center_m!, q = quaternion(f.axis ?? [0, 1, 0], f.angle_rad ?? 0);
  let e: ClothVector;
  if (f.kind === 'sphere') e = [f.radius_m!, f.radius_m!, f.radius_m!];
  else if (f.kind === 'box') {
    const r = f.roundingRadius_m ?? 0, ex = f.halfExtents_m!;
    const basis = [[ex[0], 0, 0], [0, ex[1], 0], [0, 0, ex[2]]].map(v => rotate(q, v as unknown as ClothVector));
    e = [0, 1, 2].map(i => basis.reduce((s, b) => s + Math.abs(b[i]!), r)) as unknown as ClothVector;
  } else {
    const r = f.roundingRadius_m ?? 0, axis = rotate(q, [0, 1, 0]);
    e = axis.map(a => Math.abs(a) * (f.halfLength_m! + r) + (f.radius_m! + r) * Math.sqrt(Math.max(0, 1 - a * a))) as unknown as ClothVector;
  }
  return {min: sub(c, e), max: add(c, e)};
}

interface GroupPose { center: ClothVector; rotation: SourceQuaternion }
export interface SourceMotionCheckpoint {
  readonly positions: readonly number[]; readonly velocities: readonly number[];
  readonly fields: readonly WetClothSourceField[]; readonly poses: readonly (readonly [number, GroupPose])[];
  readonly increments: readonly (readonly [number, SourceQuaternion])[]; readonly time_s: number;
}
/** Stateful source integration: original scripts evaluate velocity at t+dt.
 * Supports and SDFs consume the same script increment. Call once per substep. */
export class SourceMotionRuntime {
  readonly supportPositions_m: number[];
  readonly supportVelocities_m_s: number[];
  fields: readonly WetClothSourceField[];
  private readonly poses = new Map<number, GroupPose>();
  private readonly descendants = new Map<number, Set<number>>();
  private nextTime = 0;
  private increments = new Map<number, SourceQuaternion>();
  constructor(private readonly scripts: readonly ClothSupportScript[], private readonly topology: ClothTopologyData,
    fields: readonly WetClothSourceField[] = [], defaultCenter_m: ClothVector = [0, 0, 0]) {
    this.supportPositions_m = [...(topology.initialSupportPositions_m ?? topology.initialPositions_m ?? topology.restPositions_m)];
    this.supportVelocities_m_s = new Array(topology.restPositions_m.length).fill(0);
    this.fields = structuredClone(fields);
    const groups = new Set([...scripts.map(s => s.group), ...(topology.groups ?? []), ...fields.filter(f => f.root).map(f => f.group)]);
    for (const group of groups) {
      const roots = fields.filter(f => f.root && f.group === group);
      const bounds = roots.map(f => sourceFieldBounds(fields, f.index));
      const center: ClothVector = bounds.length ? [0, 1, 2].map(i => (Math.min(...bounds.map(b => b.min[i]!)) + Math.max(...bounds.map(b => b.max[i]!))) / 2) as unknown as ClothVector : defaultCenter_m;
      this.poses.set(group, {center, rotation: identity});
      const ids = new Set<number>();
      const visit = (f: WetClothSourceField) => { if (ids.has(f.index)) return; ids.add(f.index); for (const i of f.children ?? []) visit(fields.find(v => v.index === i)!); };
      roots.forEach(visit); this.descendants.set(group, ids);
    }
  }
  checkpoint(): SourceMotionCheckpoint {
    // The support arrays are flat numbers: slice copies them (structuredClone
    // serializes element by element, ~9 ms per array on a 32k-vertex cloth).
    return {positions: this.supportPositions_m.slice(), velocities: this.supportVelocities_m_s.slice(),
      ...structuredClone({fields: this.fields, poses: [...this.poses], increments: [...this.increments]}), time_s: this.nextTime};
  }
  restore(snapshot: SourceMotionCheckpoint): void {
    if (snapshot.positions.length !== this.supportPositions_m.length || snapshot.velocities.length !== this.supportVelocities_m_s.length) throw new Error('Incompatible source motion checkpoint');
    for (let i = 0; i < snapshot.positions.length; i++) this.supportPositions_m[i] = snapshot.positions[i]!;
    for (let i = 0; i < snapshot.velocities.length; i++) this.supportVelocities_m_s[i] = snapshot.velocities[i]!;
    this.fields = structuredClone(snapshot.fields);
    this.poses.clear(); for (const [id, pose] of structuredClone(snapshot.poses)) this.poses.set(id, pose);
    this.increments = new Map(snapshot.increments); this.nextTime = snapshot.time_s;
  }
  /** Transport fixed twist directors with this world-space rotation increment. */
  groupRotationIncrement(group: number): SourceQuaternion { return this.increments.get(group) ?? identity; }
  step(dt_s: number, currentTime_s = this.nextTime): void {
    if (!(dt_s > 0) || !Number.isFinite(dt_s) || Math.abs(currentTime_s - this.nextTime) > 1e-9) throw new Error('Source motion must advance sequentially');
    const previous = new Map([...this.poses].map(([id, pose]) => [id, {...pose}]));
    for (const script of this.scripts) {
      // The reference does not partly execute a substep crossing an endpoint.
      if (currentTime_s < script.start_s || currentTime_s + dt_s > script.end_s) continue;
      const amount = sourceCubicVelocity(script, currentTime_s + dt_s) * dt_s;
      const pose = this.poses.get(script.group)!;
      const oldCenter = pose.center, oldRotation = pose.rotation;
      let increment = identity;
      if (script.kind === 'translate') pose.center = add(pose.center, script.vector.map(v => v * amount) as unknown as ClothVector);
      else {
        increment = quaternion(script.vector, script.angle_rad! * amount);
        pose.rotation = script.global ? multiply(increment, pose.rotation) : multiply(pose.rotation, increment);
        if (script.origin_m) pose.center = add(script.origin_m, rotate(multiply(pose.rotation, conjugate(oldRotation)), sub(pose.center, script.origin_m)));
      }
      const shift = sub(pose.center, oldCenter), ids = this.descendants.get(script.group)!;
      this.fields = this.fields.map(f => {
        if (!ids.has(f.index) || f.kind === 'intersect') return f;
        const original = quaternion(f.axis ?? [0, 1, 0], f.angle_rad ?? 0);
        const rotation = script.global ? multiply(increment, original) : multiply(original, increment);
        return {...f, center_m: add(f.center_m!, shift), ...axisAngle(rotation)};
      });
    }
    this.increments = new Map([...this.poses].map(([id, pose]) => [id, multiply(pose.rotation, conjugate(previous.get(id)!.rotation))]));
    this.supportVelocities_m_s.fill(0);
    for (let i = 0; i < this.supportPositions_m.length / 3; i++) {
      if (!((this.topology.fixed?.[i] ?? 0) & 1)) continue;
      const group = this.topology.groups?.[i] ?? 0, before = previous.get(group), after = this.poses.get(group);
      if (!before || !after) continue;
      const old = this.supportPositions_m.slice(i * 3, i * 3 + 3) as unknown as ClothVector;
      const next = add(after.center, rotate(multiply(after.rotation, conjugate(before.rotation)), sub(old, before.center)));
      for (let j = 0; j < 3; j++) { this.supportPositions_m[i * 3 + j] = next[j]!; this.supportVelocities_m_s[i * 3 + j] = (next[j]! - old[j]!) / dt_s; }
    }
    this.nextTime = currentTime_s + dt_s;
  }
}
