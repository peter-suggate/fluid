/** CPU reference map compiler for the prescribed-flow lab and Dawn probes.
 * Compose continuous piecewise affine shears on convex pieces. Each shear has
 * determinant one. Split at its knot planes BEFORE transforming any vertices.
 * Liquid signed distance is a transported affine attribute, never refitted.
 */
export type V3 = [number, number, number];
export interface Vertex { p: V3; liquid: number }
type Poly = Vertex[][];
export interface Shear { axis: number; dependent: number; amplitude: number; knots: number }
export interface MapSpec { size: number; shift: V3; shears: Shear[]; cornerOnly?: boolean }
export interface Overlap { donor: number; receiver: number; vertices: Vertex[] }
export interface CompiledMap {
  overlaps: Overlap[];
  donorMappedVolumes: number[];
  initialLiquid: number[];
  pieceCount: number;
  tetrahedronCount: number;
  maximumPiecesPerCell: number;
  candidateCount: number;
  compileMs: number;
}
const add = (a: V3, b: V3): V3 => a.map((x, k) => x + b[k]!) as V3;
const sub = (a: V3, b: V3): V3 => a.map((x, k) => x - b[k]!) as V3;
const mul = (a: V3, s: number): V3 => a.map(x => x * s) as V3;
const dot = (a: V3, b: V3) => a.reduce((s, x, k) => s + x * b[k]!, 0);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
export const tetraVolume = (v: Vertex[]) => Math.abs(dot(sub(v[1]!.p, v[0]!.p),
  cross(sub(v[2]!.p, v[0]!.p), sub(v[3]!.p, v[0]!.p)))) / 6;
const mean = (vs: Vertex[]): Vertex => ({
  p: mul(vs.reduce((s, v) => add(s, v.p), [0, 0, 0] as V3), 1 / vs.length),
  liquid: vs.reduce((s, v) => s + v.liquid, 0) / vs.length,
});
const mix = (a: Vertex, b: Vertex, t: number): Vertex => ({
  p: add(a.p, mul(sub(b.p, a.p), t)), liquid: a.liquid + t * (b.liquid - a.liquid),
});
const unique = (vs: Vertex[]) => vs.filter((v, i) => !vs.slice(0, i)
  .some(w => Math.hypot(...sub(v.p, w.p)) < 1e-11));

function clip(poly: Poly, axis: number, bound: number, direction: number): Poly {
  const out: Poly = [], cap: Vertex[] = [];
  // Inverse shears can put an existing face a few f64 ulps across its original
  // knot plane. Exact-sign classification followed by vertex deduplication
  // would then add a second, full-area cap over that face. Classify only this
  // arithmetic uncertainty as coplanar. This is a local geometric predicate,
  // not a volume correction; both halves use the same predicate and original
  // vertex, and raw donor/receiver ledgers still audit the resulting geometry.
  const signed = (v: Vertex) => {
    const d = direction * (v.p[axis]! - bound);
    const roundoff = 32 * Number.EPSILON * Math.max(1, Math.abs(bound), Math.abs(v.p[axis]!));
    return Math.abs(d) <= roundoff ? 0 : d;
  };
  for (const face of poly) {
    const result: Vertex[] = [];
    let prior = face.at(-1)!;
    let a = signed(prior);
    for (const next of face) {
      const b = signed(next);
      if ((a >= 0) !== (b >= 0)) {
        const v = mix(prior, next, a / (a - b));
        v.p[axis] = bound;
        result.push(v); cap.push(v);
      }
      if (b >= 0) result.push(next);
      prior = next; a = b;
    }
    const faceOut = unique(result);
    if (faceOut.length >= 3) out.push(faceOut);
  }
  const vertices = unique(cap);
  if (vertices.length >= 3) {
    const c = mean(vertices).p, u = (axis + 1) % 3, v = (axis + 2) % 3;
    vertices.sort((a, b) => Math.atan2(a.p[v]! - c[v]!, a.p[u]! - c[u]!)
      - Math.atan2(b.p[v]! - c[v]!, b.p[u]! - c[u]!));
    out.push(vertices);
  }
  return out;
}

function cube(origin: V3, normal: V3, offset: number, full: boolean): Poly {
  const vs: Vertex[] = Array.from({ length: 8 }, (_, k) => {
    const p = add(origin, [k & 1, (k >> 1) & 1, (k >> 2) & 1]);
    return { p, liquid: full ? 1 : offset - dot(normal, p) };
  });
  return [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4],
    [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]].map(f => f.map(i => vs[i]!));
}

function tetrahedralize(poly: Poly): Vertex[][] {
  if (poly.length < 4) return [];
  const center = mean(unique(poly.flat())), tets: Vertex[][] = [];
  for (const face of poly) for (let i = 1; i + 1 < face.length; i++) {
    const tet = [center, face[0]!, face[i]!, face[i + 1]!];
    // Exact zero pieces are boundary geometry, not lost liquid volume.
    if (tetraVolume(tet) > 0) tets.push(tet);
  }
  return tets;
}

export function shearValue(y: number, shear: Shear, size: number): number {
  const scaled = y * shear.knots / size;
  const i = Math.floor(scaled), t = scaled - i;
  const sample = (k: number) => shear.amplitude * Math.sin(2 * Math.PI
    * (((k % shear.knots) + shear.knots) % shear.knots) / shear.knots);
  return (1 - t) * sample(i) + t * sample(i + 1);
}
export function mapPoint(p: V3, spec: MapSpec): V3 {
  const q = [...p] as V3;
  for (const s of spec.shears) q[s.axis]! += shearValue(q[s.dependent]!, s, spec.size);
  return add(q, spec.shift);
}

function applyShear(poly: Poly, shear: Shear, size: number): Poly[] {
  if (shear.axis === shear.dependent) throw new Error("A volume-preserving shear needs distinct axes");
  const coords = poly.flat().map(v => v.p[shear.dependent]!);
  const lo = Math.min(...coords), hi = Math.max(...coords), step = size / shear.knots;
  let remainder = poly;
  const pieces: Poly[] = [];
  for (let k = Math.floor(lo / step) + 1; k * step < hi; k++) {
    const bound = k * step;
    pieces.push(clip(remainder, shear.dependent, bound, -1));
    remainder = clip(remainder, shear.dependent, bound, 1);
  }
  pieces.push(remainder);
  return pieces.filter(p => p.length >= 4).map(p => p.map(face => face.map(v => {
    const q = [...v.p] as V3;
    q[shear.axis]! += shearValue(q[shear.dependent]!, shear, size);
    return { p: q, liquid: v.liquid };
  })));
}

/** Independent analytic integral, inclusion-exclusion CDF in f64. */
export function boxFraction(normal: V3, offset: number, lower: V3, upper: V3): number {
  const widths = sub(upper, lower), center = mul(add(lower, upper), 0.5);
  const spans = normal.map((v, k) => Math.abs(v) * widths[k]!).filter(v => v > 1e-14);
  const target = offset - dot(normal, center) + 0.5 * spans.reduce((a, b) => a + b, 0);
  if (!spans.length) return Number(target >= 0);
  if (target <= 0) return 0;
  if (target >= spans.reduce((a, b) => a + b, 0)) return 1;
  let numerator = 0;
  for (let mask = 0; mask < 1 << spans.length; mask++) {
    let shifted = target, sign = 1;
    for (let k = 0; k < spans.length; k++) if (mask & (1 << k)) {
      shifted -= spans[k]!; sign = -sign;
    }
    numerator += sign * Math.max(0, shifted) ** spans.length;
  }
  return numerator / ((spans.length === 3 ? 6 : spans.length === 2 ? 2 : 1)
    * spans.reduce((a, b) => a * b, 1));
}

// Stable CDF for volume-constrained reconstruction. Unlike the independent
// inclusion-exclusion oracle above, this avoids subtracting nearly equal
// cubic powers when an evolving plane is almost axis-aligned.
function reconstructionFraction(normal: V3, offset: number): number {
  const spans = normal.map(Math.abs).filter(v => v > 0).sort((a, b) => a - b);
  const total = spans.reduce((s, v) => s + v, 0), shifted = offset + total * .5;
  if (shifted <= 0) return 0; if (shifted >= total) return 1;
  const complement = shifted > total * .5, x = complement ? total - shifted : shifted;
  const [a, b, c] = spans as [number, number, number];
  let fraction: number;
  if (spans.length === 1) fraction = x / a;
  else if (spans.length === 2) fraction = x < a ? .5 * (x / a) * (x / b) : (x - .5 * a) / b;
  else {
    const primitive = (q: number) => q <= 0 ? 0 : q < a ? q * (q / a) * (q / b) / 6
      : q <= b ? (.5 * q * (q - a) + a * a / 6) / b
        : q < a + b ? q - .5 * (a + b) + (a + b - q) * ((a + b - q) / a) * ((a + b - q) / b) / 6
          : q - .5 * (a + b);
    fraction = (primitive(x) - primitive(x - c)) / c;
  }
  return complement ? 1 - fraction : fraction;
}
export const cellOrigin = (i: number, n: number): V3 => [i % n, Math.floor(i / n) % n, Math.floor(i / n / n)];
const wrap = (i: number, n: number) => ((i % n) + n) % n;
const cellIndex = (p: V3, n: number) => wrap(p[0], n) + n * (wrap(p[1], n) + n * wrap(p[2], n));

export function compileMap(spec: MapSpec, full = false, cellVolumes?: ArrayLike<number>): CompiledMap {
  const start = performance.now(), n = spec.size;
  if (cellVolumes && cellVolumes.length !== n ** 3) throw new Error("Remap volume/grid size mismatch");
  const normal: V3 = [2, -1, 3], offset = 2.15 * n;
  const overlaps: Overlap[] = [], donorMappedVolumes: number[] = [], initialLiquid: number[] = [];
  let pieceCount = 0, tetrahedronCount = 0, maximumPiecesPerCell = 0, candidateCount = 0;
  for (let donor = 0; donor < n ** 3; donor++) {
    const origin = cellOrigin(donor, n);
    let cellNormal = normal, cellOffset = offset, cellFull = full;
    if (cellVolumes) {
      const value = cellVolumes[donor]!;
      if (!Number.isFinite(value) || value < -1e-6 || value > 1 + 1e-6) {
        throw new Error(`Invalid source volume at cell ${donor}: ${value}`);
      }
      // Authority retains raw f32 volumes. Geometric observation of excursions
      // within the explicit roundoff band is bounded; its discrepancy remains
      // visible in the donor ledger and cumulative liquid-volume check.
      const fill = Math.max(0, Math.min(1, value));
      const gradient = [0, 1, 2].map(axis => {
        const lo = [...origin] as V3, hi = [...origin] as V3;
        lo[axis]!--; hi[axis]!++;
        return cellVolumes[cellIndex(lo, n)]! - cellVolumes[cellIndex(hi, n)]!;
      }) as V3;
      const length = Math.hypot(...gradient);
      cellNormal = length > 1e-12 ? mul(gradient, 1 / length) : [1, 0, 0];
      const center = add(origin, [0.5, 0.5, 0.5]);
      const radius = cellNormal.reduce((s, v) => s + Math.abs(v) * 0.5, 0);
      let lo = -radius, hi = radius;
      for (let iteration = 0; iteration < 44; iteration++) {
        const middle = (lo + hi) * 0.5;
        if (reconstructionFraction(cellNormal, middle) < fill) lo = middle;
        else hi = middle;
      }
      cellOffset = dot(cellNormal, center) + (fill === 0 ? -radius - 1 : (lo + hi) * 0.5);
      cellFull = fill === 1;
      initialLiquid.push(value);
    } else initialLiquid.push(full ? 1 : boxFraction(normal, offset, origin, add(origin, [1, 1, 1])));
    const initial = cube(origin, cellNormal, cellOffset, cellFull);
    let tets: Vertex[][];
    if (spec.cornerOnly) {
      // Freudenthal tetrahedra share triangulations with neighboring cells.
      const vertices = unique(initial.flat());
      const at = (q: V3) => vertices.find(v => v.p.every((x, k) => x === origin[k]! + q[k]!))!;
      tets = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]].map(order => {
        const p: V3 = [0, 0, 0], tet = [at([...p])];
        for (const axis of order) { p[axis] = 1; tet.push(at([...p])); }
        return tet.map(v => ({ p: mapPoint(v.p, spec), liquid: v.liquid }));
      });
      pieceCount++; maximumPiecesPerCell = Math.max(maximumPiecesPerCell, 1);
    } else {
      let pieces = [initial];
      for (const s of spec.shears) {
        pieces = pieces.flatMap(p => applyShear(p, s, n));
        if (pieces.length > 256) throw new Error(`Map fragment budget exceeded at donor ${donor}`);
      }
      pieceCount += pieces.length; maximumPiecesPerCell = Math.max(maximumPiecesPerCell, pieces.length);
      tets = pieces.flatMap(p => tetrahedralize(p).map(tet => tet.map(v => ({
        p: add(v.p, spec.shift), liquid: v.liquid,
      }))));
    }
    tetrahedronCount += tets.length;
    donorMappedVolumes.push(tets.reduce((sum, tet) => sum + tetraVolume(tet), 0));
    for (const tet of tets) {
      const lower = [0, 1, 2].map(k => Math.floor(Math.min(...tet.map(v => v.p[k]!)))) as V3;
      const upper = [0, 1, 2].map(k => Math.ceil(Math.max(...tet.map(v => v.p[k]!)))) as V3;
      for (let z = lower[2]; z < upper[2]; z++) for (let y = lower[1]; y < upper[1]; y++)
        for (let x = lower[0]; x < upper[0]; x++) {
          const cell: V3 = [x, y, z]; candidateCount++;
          overlaps.push({ donor, receiver: cellIndex(cell, n), vertices: tet.map(v => ({
            p: sub(v.p, cell), liquid: v.liquid,
          })) });
          if (overlaps.length > 2_000_000) throw new Error("Overlap work-list budget exceeded");
        }
    }
  }
  overlaps.sort((a, b) => a.receiver - b.receiver || a.donor - b.donor);
  return { overlaps, donorMappedVolumes, initialLiquid, pieceCount, tetrahedronCount,
    maximumPiecesPerCell, candidateCount, compileMs: performance.now() - start };
}

export function translatedReference(spec: MapSpec, full: boolean): number[] {
  if (spec.shears.length) throw new Error("Translation oracle does not support deformation");
  const n = spec.size, normal: V3 = [2, -1, 3], offset = 2.15 * n;
  return Array.from({ length: n ** 3 }, (_, i) => {
    if (full) return 1;
    const lo = sub(cellOrigin(i, n), spec.shift), hi = add(lo, [1, 1, 1]);
    let volume = 0;
    for (let z = Math.floor(lo[2]); z < Math.ceil(hi[2]); z++)
      for (let y = Math.floor(lo[1]); y < Math.ceil(hi[1]); y++)
        for (let x = Math.floor(lo[0]); x < Math.ceil(hi[0]); x++) {
          const source: V3 = [x, y, z], a = lo.map((v, k) => Math.max(v, source[k]!)) as V3;
          const b = hi.map((v, k) => Math.min(v, source[k]! + 1)) as V3;
          const delta = source.map(v => wrap(v, n) - v) as V3;
          const amount = sub(b, a).reduce((s, v) => s * v, 1);
          volume += amount * boxFraction(normal, offset, add(a, delta), add(b, delta));
        }
    return volume;
  });
}

/** Independent inverse-map midpoint quadrature. It never consumes compiled
 * pieces, tetrahedra, liquid attributes or overlap lists. The reference is the
 * authored periodic halfspace pulled back through the analytic inverse shear
 * composition. This is an approximate shape oracle; resolution differences
 * are reported and must not be described as exact volume integration.
 */
export function inverseMapReference(spec: MapSpec, samplesPerAxis: number): number[] {
  const n = spec.size, samples = samplesPerAxis, invSamples = 1 / samples;
  const inverse = [...spec.shears].reverse().map(s => ({ ...s,
    values: Array.from({ length: s.knots + 1 }, (_, k) =>
      s.amplitude * Math.sin(2 * Math.PI * (k % s.knots) / s.knots)),
  }));
  const wrapCoordinate = (v: number) => v - n * Math.floor(v / n);
  const result: number[] = [];
  for (let cell = 0; cell < n ** 3; cell++) {
    const origin = cellOrigin(cell, n);
    let liquid = 0;
    for (let z = 0; z < samples; z++) for (let y = 0; y < samples; y++)
      for (let x = 0; x < samples; x++) {
        const p = [origin[0] + (x + 0.5) * invSamples - spec.shift[0],
          origin[1] + (y + 0.5) * invSamples - spec.shift[1],
          origin[2] + (z + 0.5) * invSamples - spec.shift[2]];
        for (const s of inverse) {
          const a = wrapCoordinate(p[s.dependent]!) * s.knots / n;
          const k = Math.floor(a), t = a - k;
          p[s.axis]! -= (1 - t) * s.values[k]! + t * s.values[k + 1]!;
        }
        if (2 * wrapCoordinate(p[0]!) - wrapCoordinate(p[1]!)
          + 3 * wrapCoordinate(p[2]!) <= 2.15 * n) liquid++;
      }
    result.push(liquid / samples ** 3);
  }
  return result;
}
