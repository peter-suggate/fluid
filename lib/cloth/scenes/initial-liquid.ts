import type { InitialLiquidVolume, ScheduledLiquidDrop, Vec3 } from '../../core/model';
import type { ClothVector } from '../types';
import type { WetClothSourceField, WetClothSourceScript } from '../source-types';
import { sourceFieldBounds } from '../source-motion';

const vector = (a: ClothVector): Vec3 => ({x: a[0], y: a[1], z: a[2]});
const equalVelocity = (a: ClothVector, b: ClothVector) => a.every((v, i) => Math.abs(v - b[i]!) < 1e-12);

/** Exact subset of the existing fluid volume vocabulary. Unsupported sources
 * remain explicit requirements for the source SDF emitter; never substitute
 * their bounding box or claim their scene is initially dry. */
export function sourceInitialLiquid(fields: readonly WetClothSourceField[], scripts: readonly WetClothSourceScript[] = []): {
  readonly volumes: InitialLiquidVolume[];
  readonly scheduledDrops: ScheduledLiquidDrop[];
  readonly scheduledFieldIndices: readonly number[];
  readonly velocity_m_s: Vec3;
  readonly initializedFieldIndices: readonly number[];
  readonly needsSourceEmitter: boolean;
} {
  const candidates: {field: WetClothSourceField; volume: InitialLiquidVolume; velocity: ClothVector}[] = [];
  let needsSourceEmitter = false;
  const scheduled: {fieldIndex: number; drop: ScheduledLiquidDrop}[] = [];
  const movingGroups = new Set(scripts.map(s => s.group));
  for (const f of fields.filter(f => f.root && f.usage === 'source')) {
    if (f.kind === 'intersect' || f.inside) { needsSourceEmitter = true; continue; }
    const initial = f.emission?.find(d => d.start_s === 0 && d.end_s === 0);
    for (const d of f.emission ?? []) {
      if (d.start_s === 0 && d.end_s === 0) continue;
      if (f.kind === 'sphere' && d.start_s > 0 && d.start_s === d.end_s && d.velocity_m_s.every(v => v === 0) && !movingGroups.has(f.group))
        scheduled.push({fieldIndex: f.index, drop: {time_s: d.start_s, volume: {shape: 'sphere', center_m: vector(f.center_m!), radius_m: f.radius_m!}}});
      else needsSourceEmitter = true;
    }
    if (!initial) { if (!f.emission?.length) needsSourceEmitter = true; continue; }
    let volume: InitialLiquidVolume | undefined;
    if (f.kind === 'sphere') volume = {shape: 'sphere', center_m: vector(f.center_m!), radius_m: f.radius_m!};
    else if (f.kind === 'box' && f.roundingRadius_m === 0 && (f.angle_rad ?? 0) === 0) {
      const b = sourceFieldBounds(fields, f.index);
      volume = {shape: 'box', min_m: vector(b.min), max_m: vector(b.max)};
    } else if (f.kind === 'cylinder' && f.roundingRadius_m === 0) {
      const a = f.axis!, norm = Math.hypot(...a), angle = f.angle_rad ?? 0, c = Math.cos(angle), s = Math.sin(angle);
      const [x, y, z] = norm ? a.map(v => v / norm) : [0, 1, 0];
      // Rotate source Y onto the only cylinder axis the current liquid schema supports, Z.
      const axis = [x! * y! * (1 - c) - z! * s, c + y! * y! * (1 - c), z! * y! * (1 - c) + x! * s];
      if (Math.abs(axis[0]!) < 1e-12 && Math.abs(axis[1]!) < 1e-12 && Math.abs(Math.abs(axis[2]!) - 1) < 1e-12)
        volume = {shape: 'cylinder', center_m: vector(f.center_m!), radius_m: f.radius_m!, halfHeight_m: f.halfLength_m!};
    }
    if (volume) candidates.push({field: f, volume, velocity: initial.velocity_m_s});
    else needsSourceEmitter = true;
  }
  const uniqueScheduled = scheduled.filter(a => scheduled.filter(b => b.drop.time_s === a.drop.time_s).length === 1);
  if (uniqueScheduled.length !== scheduled.length) needsSourceEmitter = true;
  uniqueScheduled.sort((a,b) => a.drop.time_s-b.drop.time_s);
  const scheduledResult = {scheduledDrops: uniqueScheduled.map(s => s.drop), scheduledFieldIndices: uniqueScheduled.map(s => s.fieldIndex)};
  const velocity: ClothVector = candidates[0]?.velocity ?? [0, 0, 0];
  // Existing NB-FLIP uniformInitialLiquidVelocity is global. It is incorrect to
  // initialize two differently moving bodies with that one velocity.
  if (candidates.some(c => !equalVelocity(c.velocity, velocity))) return {...scheduledResult, volumes: [], velocity_m_s: vector([0, 0, 0]), initializedFieldIndices: [], needsSourceEmitter: true};
  return {...scheduledResult, volumes: candidates.map(c => c.volume), velocity_m_s: vector(velocity),
    initializedFieldIndices: candidates.map(c => c.field.index), needsSourceEmitter};
}
