import { COEFFICIENT_COUNT, meanBasis, polynomial, volume, validateBox,
  type Box, type Frame, type Polynomial } from "./field";

export interface Donor { readonly id: number; readonly box: Box }
export interface CompiledStencil {
  readonly generation: number;
  readonly frame: Frame;
  readonly ids: Uint32Array;
  /** Coefficient-major map from current donor means to polynomial coefficients. */
  readonly weights: Float64Array;
  readonly rankRatio: number;
}

/** Geometry-only compilation, including each donor's second volume moments.
 * Constrain the home mean exactly; fit the other nine modes in weighted LS.
 * Reorthogonalized QR avoids the condition-number squaring of normal equations.
 * This is an oracle compiler, not an adapter to the production topology ABI.
 */
export function compileStencil(donors: readonly Donor[], homeId: number,
  frame: Frame, generation: number): CompiledStencil {
  if (!Number.isInteger(generation) || generation < 0) throw new Error("invalid generation");
  if (donors.length < COEFFICIENT_COUNT) throw new Error("insufficient reconstruction support");
  const ids = new Set<number>();
  for (const d of donors) {
    validateBox(d.box);
    if (!Number.isInteger(d.id) || d.id < 0 || d.id > 0xffff_fffe || ids.has(d.id)) {
      throw new Error("invalid or duplicate donor ordinal");
    }
    ids.add(d.id);
  }
  const home = donors.findIndex(d => d.id === homeId);
  if (home < 0) throw new Error("home donor is absent");
  const basis = donors.map(d => meanBasis(frame, d.box));
  const indices = donors.map((_, i) => i).filter(i => i !== home);
  const factors = indices.map(i => Math.sqrt(volume(donors[i].box) / volume(donors[home].box)));
  const columns = Array.from({ length: 9 }, (_, k) => Float64Array.from(indices,
    (i, row) => factors[row] * (basis[i][k + 1] - basis[home][k + 1])));
  const r = Array.from({ length: 9 }, () => new Float64Array(9));
  const q: Float64Array[] = [];
  const inputNorm = Math.max(...columns.map(v => Math.hypot(...v)));
  if (!(inputNorm > 0) || !Number.isFinite(inputNorm)) {
    throw new Error("rank-deficient reconstruction support");
  }
  for (let k = 0; k < 9; k++) {
    const v = columns[k].slice();
    for (let pass = 0; pass < 2; pass++) for (let j = 0; j < k; j++) {
      let dot = 0; for (let at = 0; at < v.length; at++) dot += q[j][at] * v[at];
      r[j][k] += dot;
      for (let at = 0; at < v.length; at++) v[at] -= dot * q[j][at];
    }
    r[k][k] = Math.hypot(...v);
    if (r[k][k] < inputNorm * 1e-10) throw new Error("rank-deficient reconstruction support");
    q.push(Float64Array.from(v, x => x / r[k][k]));
  }
  const n = donors.length, weights = new Float64Array(10 * n);
  for (let at = 0; at < indices.length; at++) {
    const solution = new Float64Array(9);
    for (let k = 8; k >= 0; k--) {
      let value = q[k][at];
      for (let j = k + 1; j < 9; j++) value -= r[k][j] * solution[j];
      solution[k] = value / r[k][k];
    }
    for (let k = 0; k < 9; k++) {
      const value = solution[k] * factors[at];
      weights[(k + 1) * n + indices[at]] = value;
      weights[(k + 1) * n + home] -= value;
    }
  }
  weights[home] = 1;
  for (let at = 0; at < n; at++) for (let k = 1; k < 10; k++) {
    weights[at] -= basis[home][k] * weights[k * n + at];
  }
  const diagonal = r.map((row, k) => row[k]);
  return { generation, frame, ids: Uint32Array.from(donors, d => d.id), weights,
    rankRatio: Math.min(...diagonal) / Math.max(...diagonal) };
}

/** Steady-state apply: one read per compiled donor; no geometric lookup or fit. */
export function applyStencil(packet: CompiledStencil, generation: number,
  readDensity: (id: number) => number, float32 = false): Polynomial {
  if (generation !== packet.generation) throw new Error("stale reconstruction packet");
  const n = packet.ids.length, values = Float64Array.from(packet.ids, id => {
    const value = readDensity(id);
    if (!Number.isFinite(value)) throw new Error("non-finite donor density");
    return float32 ? Math.fround(value) : value;
  });
  const coefficients = new Array<number>(10).fill(0);
  for (let k = 0; k < 10; k++) for (let i = 0; i < n; i++) {
    const w = packet.weights[k * n + i];
    coefficients[k] = float32
      ? Math.fround(coefficients[k] + Math.fround(Math.fround(w) * values[i]))
      : coefficients[k] + w * values[i];
  }
  return polynomial(packet.frame, coefficients);
}

export function packetCost(packet: CompiledStencil) {
  const donors = packet.ids.length;
  return { donors, densityReadsPerApply: donors, multiplyAddsPerApply: 10 * donors,
    expandedFloat32WeightsBytes: 4 * packet.weights.length,
    donorOrdinalBytes: packet.ids.byteLength,
    note: "excludes headers, frame, reverse dependencies and output; intern geometry maps before GPU rollout" };
}
