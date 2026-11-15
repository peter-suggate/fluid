/** The NB-FLIP port referee's rules (docs/plans/nbflip-port-2026-11-15.md §4, amended §4.5), shared by
 * tools/nbflip-gates.ts and tests/nbflip-gate-rules.test.ts.
 *
 * - Two-sided scalar and distance rows: |c - mean(refs)| <= max(floor, 3S), S the references' range
 *   (distance rows: the largest pairwise distance).
 * - One-sided drift rows (§4.5 (1)): |c| <= |mean(refs)| + max(floor, 3S).
 * - Volume-coupled rows (§4.5 (2)): the candidate passes inside the band around the references, or
 *   inside the band around the references with each run's own measured volume loss restored, or
 *   anywhere between them: the segment ref + alpha (restored - ref), alpha in [0, 1], each point's
 *   tolerance interpolated between the two ends' (the interval rule of seep §4.1's G6 centroid).
 * - restoreBox: a box-scene sample with its run's measured loss (expected - total) added back as a
 *   uniform depth over its wet footprint (§4.5 defines it). */
import type { BoxGeometry, BoxObservation } from "./nbflip-observables";

export const mean = (v: readonly number[]) => v.reduce((a, b) => a + b, 0) / v.length;
export const spread = (v: readonly number[]) => Math.max(...v) - Math.min(...v);
export const l1 = (a: readonly number[], b: readonly number[]) => a.reduce((s, v, i) => s + Math.abs(v - b[i]!), 0);
export const rms = (a: readonly number[], b: readonly number[]) => Math.sqrt(mean(a.map((v, i) => (v - b[i]!) ** 2)));
export const normalise = (v: readonly number[]) => { const t = v.reduce((a, b) => a + b, 0); return v.map(x => t > 0 ? x / t : 0); };

/** One checkpoint of a gate. `ref` is the reference mean (distance rows: 0; band rows: the measured end). */
export interface Row { at: string; ref: number | null; S: number; candidate: number | null; tol: number; pass: boolean; margin: number; restored?: number | null; restoredS?: number; restoredTol?: number;
  /** Band rows: the margin under the strict union of the two bands (alpha 0 or 1 only), for the report. */
  unionMargin?: number }
export interface Gate { id: string; observable: string; rule: string; rows: Row[]; pass: boolean; informational?: boolean }
/** A floor: `rel` of |ref| or `abs`, whichever is larger. */
export interface Floor { rel?: number; abs?: number }
export const floorOf = (fl: Floor, ref: number) => Math.max((fl.rel ?? 0) * Math.abs(ref), fl.abs ?? 0);
export const ruleText = (fl: Floor, unit = "") => `max(${[fl.rel !== undefined ? `${100 * fl.rel}%` : "", fl.abs !== undefined ? `${fl.abs}${unit}` : ""].filter(Boolean).join(" | ")}, 3S)`;
const known = (v: readonly (number | null)[]) => v.filter((x): x is number => x !== null && Number.isFinite(x));
/** A row whose references are missing on some runs: passes only when every side is missing. */
function missingRow(at: string, refs: (number | null)[], candidate: number | null): Row | undefined {
  const k = known(refs);
  if (k.length === refs.length) return undefined;
  const pass = k.length === 0 && candidate === null;
  return { at, ref: k.length ? mean(k) : null, S: k.length ? spread(k) : 0, candidate, tol: 0, pass, margin: pass ? 0 : Infinity };
}
export interface ScalarPoint { at: string; refs: (number | null)[]; candidate: number | null }

/** A scalar gate: per checkpoint, the references' values and the candidate's. A null on both sides passes; on one side fails. */
export function scalarGate(id: string, observable: string, fl: Floor, unit: string, points: ScalarPoint[], informational = false): Gate {
  const rows = points.map(p => {
    const missing = missingRow(p.at, p.refs, p.candidate); if (missing) return missing;
    const v = p.refs as number[], ref = mean(v), S = spread(v), tol = Math.max(floorOf(fl, ref), 3 * S);
    if (p.candidate === null || !Number.isFinite(p.candidate)) return { at: p.at, ref, S, candidate: p.candidate, tol, pass: false, margin: Infinity };
    const d = Math.abs(p.candidate - ref);
    return { at: p.at, ref, S, candidate: p.candidate, tol, pass: d <= tol, margin: tol > 0 ? d / tol : d > 0 ? Infinity : 0 };
  });
  return { id, observable, rule: `|c - ref| <= ${ruleText(fl, unit)}`, rows, pass: rows.every(r => r.pass), informational };
}

/** §4.5 (1): a one-sided drift gate. Passes when |c| <= |mean(refs)| + max(floor, 3S); `ref` prints |mean(refs)|. */
export function driftGate(id: string, observable: string, fl: Floor, points: ScalarPoint[], informational = false): Gate {
  const rows = points.map(p => {
    const missing = missingRow(p.at, p.refs, p.candidate); if (missing) return missing;
    const v = p.refs as number[], ref = Math.abs(mean(v)), S = spread(v), tol = Math.max(floorOf(fl, ref), 3 * S);
    if (p.candidate === null || !Number.isFinite(p.candidate)) return { at: p.at, ref, S, candidate: p.candidate, tol, pass: false, margin: Infinity };
    const over = Math.max(0, Math.abs(p.candidate) - ref);
    return { at: p.at, ref, S, candidate: p.candidate, tol, pass: over <= tol, margin: tol > 0 ? over / tol : over > 0 ? Infinity : 0 };
  });
  return { id, observable, rule: `|c| <= |ref| + ${ruleText(fl)}`, rows, pass: rows.every(r => r.pass), informational };
}

/** The alphas at which a band segment is evaluated: 0, 1/200, ..., 1 (both ends exact). */
export const BAND_ALPHAS = Array.from({ length: 201 }, (_, i) => i / 200);
/** The smallest d(alpha) / tol(alpha) along the segment, tol interpolated linearly between the ends, and the
 * smaller of the two ends' (the strict union of the two bands). */
function segmentMargin(d: (alpha: number) => number, tolRef: number, tolRestored: number) {
  const at = (a: number) => { const dist = d(a), tol = tolRef + a * (tolRestored - tolRef); return tol > 0 ? dist / tol : dist > 0 ? Infinity : 0; };
  let best = Infinity;
  for (const a of BAND_ALPHAS) best = Math.min(best, at(a));
  return { margin: best, unionMargin: Math.min(at(0), at(1)) };
}
export interface BandScalarPoint { at: string; refs: (number | null)[]; restored: (number | null)[]; candidate: number | null }
/** §4.5 (2), scalar: the candidate within the band between the references (their own S and floor) and the
 * references with their measured loss restored (the restored values' S and floor). */
export function bandScalarGate(id: string, observable: string, fl: Floor, unit: string, points: BandScalarPoint[], informational = false): Gate {
  const rows = points.map(p => {
    const missing = missingRow(p.at, [...p.refs, ...p.restored], p.candidate); if (missing) return missing;
    const v = p.refs as number[], w = p.restored as number[];
    const ref = mean(v), S = spread(v), tol = Math.max(floorOf(fl, ref), 3 * S);
    const restored = mean(w), restoredS = spread(w), restoredTol = Math.max(floorOf(fl, restored), 3 * restoredS);
    const base = { at: p.at, ref, S, candidate: p.candidate, tol, restored, restoredS, restoredTol };
    if (p.candidate === null || !Number.isFinite(p.candidate)) return { ...base, pass: false, margin: Infinity };
    const c = p.candidate, { margin, unionMargin } = segmentMargin(a => Math.abs(c - (ref + a * (restored - ref))), tol, restoredTol);
    return { ...base, pass: margin <= 1, margin, unionMargin };
  });
  return { id, observable, rule: `c within ${ruleText(fl, unit)} of [ref, ref restored]`, rows, pass: rows.every(r => r.pass), informational };
}
export interface BandProfilePoint { at: string; refs: number[][]; restored: number[][]; candidate: number[] }
const pairwiseMax = (v: number[][], distance: (a: number[], b: number[]) => number) => Math.max(0, ...v.flatMap((a, i) => v.slice(i + 1).map(b => distance(a, b))));
const meanProfile = (v: number[][]) => v[0]!.map((_, i) => mean(v.map(r => r[i]!)));
/** §4.5 (2), profile: d(c, m_ref + alpha (m_restored - m_ref)) within the tolerance interpolated between
 * max(floor, 3 S_ref) and max(floor, 3 S_restored), for some alpha in [0, 1]. `ref` prints d(c, m_ref),
 * `restored` d(c, m_restored). */
export function bandDistanceGate(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number,
  points: BandProfilePoint[], informational = false): Gate {
  const rows = points.map(p => {
    const m = meanProfile(p.refs), mr = meanProfile(p.restored);
    const S = pairwiseMax(p.refs, distance), restoredS = pairwiseMax(p.restored, distance);
    const tol = Math.max(floor, 3 * S), restoredTol = Math.max(floor, 3 * restoredS);
    const { margin, unionMargin } = segmentMargin(a => distance(p.candidate, m.map((v, i) => v + a * (mr[i]! - v))), tol, restoredTol);
    return { at: p.at, ref: distance(p.candidate, m), S, candidate: distance(p.candidate, m), tol, restored: distance(p.candidate, mr), restoredS, restoredTol, pass: margin <= 1, margin, unionMargin };
  });
  return { id, observable, rule: `d(c, [ref, ref restored]) <= max(${floor}${unit}, 3S)`, rows, pass: rows.every(r => r.pass), informational };
}
/** A distance gate: per checkpoint, d(candidate, mean of references) within max(floor, 3 x the largest pairwise reference distance). */
export function distanceGate(id: string, observable: string, floor: number, unit: string, distance: (a: number[], b: number[]) => number,
  points: { at: string; refs: number[][]; candidate: number[] }[], informational = false): Gate {
  const rows = points.map(p => {
    const m = meanProfile(p.refs), S = pairwiseMax(p.refs, distance);
    const d = distance(p.candidate, m), tol = Math.max(floor, 3 * S);
    return { at: p.at, ref: 0, S, candidate: d, tol, pass: d <= tol, margin: d / tol };
  });
  return { id, observable, rule: `d(c, ref) <= max(${floor}${unit}, 3S), S the largest pairwise d between references`, rows, pass: rows.every(r => r.pass), informational };
}

/** Litres per cell of a box geometry. */
export const cellLitres = (g: Pick<BoxGeometry, "cell_m">) => 1000 * g.cell_m ** 3;
/** Each (x, z) bin's footprint in cells (the last bin on an axis takes the remainder), x fastest. */
export function binAreas(g: Pick<BoxGeometry, "nx" | "nz" | "binCells" | "bins">) {
  const [bx, bz] = g.bins, w = (n: number, e: number, i: number) => Math.min(e, n - i * e);
  return Array.from({ length: bx * bz }, (_, b) => w(g.nx, g.binCells[0], b % bx) * w(g.nz, g.binCells[1], Math.floor(b / bx)));
}
/** A box run's mean liquid depth per (x, z) bin (cells). */
export const binDepths = (g: Pick<BoxGeometry, "nx" | "nz" | "cell_m" | "binCells" | "bins">, litres: readonly number[]) => {
  const A = binAreas(g), L = cellLitres(g); return litres.map((v, b) => v / (L * A[b]!));
};
/** A bin is in the wet footprint when its mean depth is at least WET_DEPTH_CELLS. */
export const WET_DEPTH_CELLS = 1;
export interface RestoredBox {
  /** The run's measured loss (expected - total, litres; 0 when it gained) and the depth added over the footprint (cells). */
  lost_L: number; depth_cells: number; wetBins: number;
  bins_L: number[]; binDepth_cells: number[]; profileDepth_cells: number[]; binTops_cells: number[]; profileTop_cells: number[];
  columnDepth_cells: number | null; columnTop_cells: number | null;
  pe_J: number; energy_J: number; centroidY_cells: number | null;
}
/**
 * §4.5's restoration of a box sample: its own measured loss, lost = expected_L - total_L (none when
 * negative), added back as one uniform depth delta over its wet footprint, the bins of mean depth
 * d_b >= 1 cell: delta = lost / (litres per cell x sum of the wet bins' footprints). Each wet bin's
 * litres grow by delta x footprint, its mean depth and highest wet top by delta. The added layer
 * sits on each wet bin's mean surface, so it carries potential energy rho g h^4 A_b delta (d_b +
 * delta / 2) and raises the centroid accordingly. The x profiles are recomputed from the restored
 * bins (the observer's definition: litres in the x strip over strip width x nz). The back-wall
 * column (dam) gains delta when its own mean depth is at least 1 cell. KE is unchanged: the
 * restored liquid is at rest.
 */
export function restoreBox(g: BoxGeometry, s: BoxObservation): RestoredBox {
  const L = cellLitres(g), A = binAreas(g), [bx, bz] = g.bins;
  const depth = binDepths(g, s.bins_L), wet = depth.map(d => d >= WET_DEPTH_CELLS);
  const area = A.reduce((t, a, b) => t + (wet[b] ? a : 0), 0);
  const lost_L = Math.max(0, s.expected_L - s.total_L);
  const delta = area > 0 ? lost_L / (L * area) : 0;
  const bins_L = s.bins_L.map((v, b) => wet[b] ? v + delta * A[b]! * L : v);
  const binDepth_cells = depth.map((d, b) => wet[b] ? d + delta : d);
  const binTops_cells = s.binTops_cells.map((t, b) => wet[b] ? t + delta : t);
  const profileDepth_cells = Array.from({ length: bx }, (_, i) => {
    let litres = 0; for (let j = 0; j < bz; j++) litres += bins_L[i + bx * j]!;
    return litres / L / (Math.min(g.binCells[0], g.nx - i * g.binCells[0]) * g.nz);
  });
  const profileTop_cells = Array.from({ length: bx }, (_, i) => { let t = 0; for (let j = 0; j < bz; j++) t = Math.max(t, binTops_cells[i + bx * j]!); return t; });
  // The layer's cell-volume moment: sum over wet bins of A_b delta (d_b + delta / 2) (cells^4).
  const moment = depth.reduce((t, d, b) => t + (wet[b] ? A[b]! * delta * (d + delta / 2) : 0), 0);
  const added = delta * area;
  const dPE = g.density_kg_m3 * g.gravity_m_s2 * g.cell_m ** 4 * moment;
  const cells = s.total_L / L, cy = s.centroid?.[1] ?? null;
  const columnWet = s.columnDepth_cells !== null && s.columnDepth_cells >= WET_DEPTH_CELLS;
  return {
    lost_L, depth_cells: delta, wetBins: wet.filter(Boolean).length,
    bins_L, binDepth_cells, profileDepth_cells, binTops_cells, profileTop_cells,
    columnDepth_cells: s.columnDepth_cells === null ? null : s.columnDepth_cells + (columnWet ? delta : 0),
    columnTop_cells: s.columnTop_cells === null ? null : s.columnTop_cells + (columnWet ? delta : 0),
    pe_J: s.pe_J + dPE, energy_J: s.energy_J + dPE,
    centroidY_cells: cy === null ? null : (cells * cy + moment) / (cells + added),
  };
}
