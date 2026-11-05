// Source shape formulae ported from libWetCloth (2018), Fei, Batty, Grinspun, Zheng.
// This file is subject to MPL-2.0; see public/assets/wetcloth/LICENSE.txt.
import type { WetClothSourceField } from './source-types';
import type { ClothVector } from './types';
import { sourceMeshDistance } from './source-mesh';

/** Reference CPU geometry for source parity and seeding, in SI. The rounded
 * cylinder is aligned with Y, matching libWetCloth DistanceFields.cpp. */
export function sourceFieldDistance(fields: readonly WetClothSourceField[], index: number, point: ClothVector,
  external?: (field: WetClothSourceField, point: ClothVector) => number): number {
  const visiting = new Set<number>();
  const evaluate = (id: number): number => {
    const f = fields.find(v => v.index === id);
    if (!f || visiting.has(id)) throw new Error(`Invalid or cyclic source field ${id}`);
    visiting.add(id);
    let distance: number;
    if (f.kind === 'intersect') {
      if (!f.children?.length) throw new Error('Empty source intersection');
      distance = Math.max(...f.children.map(evaluate));
    } else if (f.kind === 'file') {
      distance = (external ?? sourceMeshDistance)(f, point);
    } else {
      const center = f.center_m ?? [0, 0, 0];
      let x = point[0] - center[0], y = point[1] - center[1], z = point[2] - center[2];
      const angle = -(f.angle_rad ?? 0);
      if (angle !== 0) {
        const axis = f.axis ?? [0, 0, 0], n = Math.hypot(...axis);
        if (!(n > 0)) throw new Error('Rotated source needs a nonzero axis');
        const [a, b, c] = axis.map(v => v / n), cosine = Math.cos(angle), sine = Math.sin(angle);
        const dot = a! * x + b! * y + c! * z;
        [x, y, z] = [x * cosine + (b! * z - c! * y) * sine + a! * dot * (1 - cosine),
          y * cosine + (c! * x - a! * z) * sine + b! * dot * (1 - cosine),
          z * cosine + (a! * y - b! * x) * sine + c! * dot * (1 - cosine)];
      }
      switch (f.kind) {
        case 'sphere': distance = Math.hypot(x, y, z) - f.radius_m!; break;
        case 'box': {
          const e = f.halfExtents_m!, r = f.roundingRadius_m ?? 0;
          const q = [Math.abs(x) - e[0], Math.abs(y) - e[1], Math.abs(z) - e[2]];
          distance = Math.min(Math.max(...q), 0) + Math.hypot(...q.map(v => Math.max(v, 0))) - r;
          break;
        }
        case 'cylinder': {
          const r = f.roundingRadius_m ?? 0;
          const radial = Math.hypot(x, z) - f.radius_m! + r, axial = Math.abs(y) - f.halfLength_m!;
          distance = Math.min(Math.max(radial, axial), 0) + Math.hypot(Math.max(radial, 0), Math.max(axial, 0)) - r;
          break;
        }
        default: throw new Error(`Unsupported source field kind: ${String(f.kind)}`);
      }
      if (f.inside) distance = -distance;
    }
    visiting.delete(id);
    if (!Number.isFinite(distance)) throw new Error(`Non-finite source field ${id}`);
    return distance;
  };
  return evaluate(index);
}

export function sourceTerminatesPoint(fields: readonly WetClothSourceField[], point: ClothVector): boolean {
  return fields.some(f => f.root && f.usage === 'terminator' && sourceFieldDistance(fields, f.index, point) < 0);
}

export interface SourceEmissionEvent {
  readonly fieldIndex: number;
  readonly kind: 'instant' | 'interval';
  readonly start_s: number;
  readonly end_s: number;
  readonly velocity_m_s: ClothVector;
}

/** Deterministic scheduling only; the fluid solver must sample vacant source
 * volume and carry this velocity, never replace it with a decorative emitter.
 * Intersections inherit the first active child's duration/velocity upstream. */
export class SourceEmissionScheduler {
  private lastEnd = 0;
  constructor(private readonly fields: readonly WetClothSourceField[]) {}
  advance(start_s: number, end_s: number): readonly SourceEmissionEvent[] {
    if (![start_s, end_s].every(Number.isFinite) || start_s < this.lastEnd || end_s <= start_s) throw new Error('Source schedule requires increasing nonoverlapping windows');
    const events: SourceEmissionEvent[] = [];
    const durations = (f: WetClothSourceField, seen = new Set<number>()): NonNullable<WetClothSourceField['emission']> => {
      if (seen.has(f.index)) throw new Error('Cyclic source schedule');
      if (f.kind !== 'intersect') return f.emission ?? [];
      const next = new Set(seen).add(f.index);
      return (f.children ?? []).flatMap(id => {
        const child = this.fields.find(v => v.index === id);
        if (!child) throw new Error(`Missing source child ${id}`);
        return durations(child, next);
      });
    };
    for (const field of this.fields.filter(f => f.root && f.usage === 'source')) {
      const all = durations(field);
      if (all.some(d => !Number.isFinite(d.start_s) || !Number.isFinite(d.end_s) || d.end_s < d.start_s)) throw new Error('Invalid source duration');
      const times = new Set(all.filter(d => d.start_s === d.end_s && d.start_s >= start_s && d.start_s < end_s).map(d => d.start_s));
      for (const t of times) {
        const d = all.find(v => v.start_s <= t && v.end_s >= t)!;
        events.push({fieldIndex: field.index, kind: 'instant', start_s: t, end_s: t, velocity_m_s: d.velocity_m_s});
      }
      const boundaries = [...new Set([start_s, end_s, ...all.flatMap(d => [d.start_s, d.end_s]).filter(t => t > start_s && t < end_s)])].sort((a, b) => a - b);
      for (let i = 0; i < boundaries.length - 1; i++) {
        const a = boundaries[i]!, b = boundaries[i + 1]!, mid = (a + b) / 2;
        const d = all.find(v => v.start_s <= mid && v.end_s >= mid);
        if (d) events.push({fieldIndex: field.index, kind: 'interval', start_s: a, end_s: b, velocity_m_s: d.velocity_m_s});
      }
    }
    this.lastEnd = end_s;
    return events.sort((a, b) => a.start_s - b.start_s || a.fieldIndex - b.fieldIndex);
  }
  /** Restore only alongside fluid simulation state. */
  restore(time_s: number): void {
    if (!Number.isFinite(time_s) || time_s < 0) throw new Error('Invalid source schedule time');
    this.lastEnd = time_s;
  }
}
