import type { DensityNativeGeometry, DensityNativeVec3 } from "./sparse-cm12-density-native-geometry";

/** Nonoverlapping retained ownership boxes, independent of the physics grid.
 * Controls use x + 3*y + 9*z, degree two Bernstein polynomials per axis. */
export interface DensityBernsteinSupportBox {
  readonly id: number;
  readonly lower: DensityNativeVec3;
  readonly upper: DensityNativeVec3;
}
export interface DensitySupportCoupling {
  readonly topologyGeneration: number;
  readonly boundaryGeneration: number;
  readonly supportGeneration: number;
  readonly domain: "full-native-boxes";
  /** Collision-free ordered geometry identity; independent of field values. */
  readonly supportGeometryKey: string;
  readonly cellIds: Uint32Array;
  readonly offsets: Uint32Array;
  readonly supportIds: Uint32Array;
  /** Input support ordinals for direct GPU control addressing. */
  readonly supportIndices: Uint32Array;
  /** Intersection-major physical integrals: B0/B1/B2 for x, then y, then z. */
  readonly axisMoments: Float64Array;
  readonly inverseCellVolumes: Float64Array;
  readonly cellVolumes: Float64Array;
  readonly coveredVolumes: Float64Array;
  readonly receipt: Readonly<{
    supportBoxes: number; nativeCells: number; intersections: number;
    bvhNodes: number; boxTests: number; incompleteCells: number;
    maximumUncoveredFraction: number; supportIdentityUtf16Bytes: number; couplingBytes: number;
  }>;
}
/** Canonical exact finite-number serialization includes ordering and native IDs.
 * Unlike a short hash this identity has no collision-based acceptance path.
 * -0 and +0 intentionally identify the same physical coordinate. */
export function densitySupportGeometryKey(supports: readonly DensityBernsteinSupportBox[]): string {
  return JSON.stringify(supports.map(box => {
    if (!Number.isInteger(box.id) || box.id < 0 || box.id > 0xffff_fffe
      || box.lower.length !== 3 || box.upper.length !== 3
      || box.lower.some((v, k) => !Number.isFinite(v) || !Number.isFinite(box.upper[k]) || v >= box.upper[k]!)) {
      throw new Error("invalid density support identity");
    }
    return [box.id, ...box.lower, ...box.upper];
  }));
}
/** Call against the actual uploaded support descriptors, not a reused epoch
 * number, before binding a coupling to another retained field. */
export function assertDensitySupportCouplingSupport(coupling: DensitySupportCoupling,
  supports: readonly DensityBernsteinSupportBox[], supportGeneration: number): void {
  if (supportGeneration !== coupling.supportGeneration
    || densitySupportGeometryKey(supports) !== coupling.supportGeometryKey) {
    throw new Error("retained density coupling support identity mismatch");
  }
}
interface Bounds { lower: DensityNativeVec3; upper: DensityNativeVec3 }
interface Node extends Bounds { left?: Node; right?: Node; support?: number }
const vec = (v: number[]) => v as unknown as DensityNativeVec3;
function intersection(a: Bounds, b: Bounds): Bounds | undefined {
  const lower = vec(a.lower.map((v, k) => Math.max(v, b.lower[k]!)));
  const upper = vec(a.upper.map((v, k) => Math.min(v, b.upper[k]!)));
  return lower.every((v, k) => v < upper[k]!) ? { lower, upper } : undefined;
}
const volume = (box: Bounds) => box.lower.reduce((v, lo, k) => v * (box.upper[k]! - lo), 1);
function validGeneration(value: number): boolean { return Number.isSafeInteger(value) && value >= 0; }
/** Exact integrals of B0=(1-u)^2, B1=2u(1-u), B2=u^2.
 * Midpoint form avoids cancellation from subtracting near-equal primitives. */
function axisIntegrals(lo: number, hi: number): readonly [number, number, number] {
  const width = hi - lo, m = (lo + hi) / 2, variance = width * width / 12;
  return [width * ((1 - m) ** 2 + variance), width * (2 * m * (1 - m) - 2 * variance), width * (m * m + variance)];
}
/** Geometry-only sparse coupling compilation. BVH storage scales with retained
 * boxes, never logical empty extent or the finest volume of a macro cell.
 * This integrates full native boxes. Solid-clipped integration requires an
 * independent open-domain tensor moment compiler and is not claimed here. */
export function compileDensitySupportCoupling(options: Readonly<{
  geometry: DensityNativeGeometry;
  supports: readonly DensityBernsteinSupportBox[];
  topologyGeneration: number;
  boundaryGeneration: number;
  supportGeneration: number;
  requireCoverage?: boolean;
  requireFullyOpen?: boolean;
  coverageTolerance?: number;
}>): DensitySupportCoupling {
  const { geometry, supports } = options;
  const supportGeometryKey = densitySupportGeometryKey(supports);
  if (geometry.topologyGeneration !== options.topologyGeneration || geometry.boundaryGeneration !== options.boundaryGeneration || !validGeneration(options.supportGeneration)) throw new Error("stale density coupling geometry");
  const tolerance = options.coverageTolerance ?? 1e-12;
  if (!Number.isFinite(tolerance) || tolerance < 0 || tolerance > 1e-6) throw new Error("invalid density coverage tolerance");
  const ids = new Set<number>();
  for (const box of supports) {
    if (!Number.isInteger(box.id) || box.id < 0 || box.id > 0xffff_fffe || ids.has(box.id)) throw new Error("invalid or duplicate density support id");
    ids.add(box.id);
    if (box.lower.length !== 3 || box.upper.length !== 3 || box.lower.some((v, k) => !Number.isFinite(v) || !Number.isFinite(box.upper[k]) || v >= box.upper[k]!)) throw new Error("invalid density support bounds");
    if (!(volume(box) > 0) || !Number.isFinite(volume(box))) throw new Error("invalid density support volume");
  }
  let bvhNodes = 0, boxTests = 0;
  function build(indices: number[]): Node | undefined {
    if (!indices.length) return undefined;
    bvhNodes++;
    const lower = [Infinity, Infinity, Infinity], upper = [-Infinity, -Infinity, -Infinity];
    for (const i of indices) for (let k = 0; k < 3; k++) {
      lower[k] = Math.min(lower[k]!, supports[i]!.lower[k]!);
      upper[k] = Math.max(upper[k]!, supports[i]!.upper[k]!);
    }
    const bounds = { lower: vec(lower), upper: vec(upper) };
    if (indices.length === 1) return { ...bounds, support: indices[0]! };
    let axis = 0;
    for (let k = 1; k < 3; k++) if (upper[k]! - lower[k]! > upper[axis]! - lower[axis]!) axis = k;
    indices.sort((a, b) => (supports[a]!.lower[axis]! + supports[a]!.upper[axis]!) / 2 - (supports[b]!.lower[axis]! + supports[b]!.upper[axis]!) / 2 || supports[a]!.id - supports[b]!.id);
    const mid = Math.floor(indices.length / 2);
    return { ...bounds, left: build(indices.slice(0, mid)), right: build(indices.slice(mid)) };
  }
  const root = build(supports.map((_, i) => i));
  function query(box: Bounds, visit: (i: number, overlap: Bounds) => void, node = root): void {
    if (!node) return;
    boxTests++;
    const overlap = intersection(box, node);
    if (!overlap) return;
    if (node.support !== undefined) { visit(node.support, overlap); return; }
    if (node.left) query(box, visit, node.left);
    if (node.right) query(box, visit, node.right);
  }
  // Support ownership is checked independently of the queried physics cells.
  supports.forEach((box, i) => query(box, j => {
    if (j !== i) throw new Error(`overlapping retained density supports ${box.id} and ${supports[j]!.id}`);
  }));
  const offsets = new Uint32Array(geometry.cells.length + 1), supportIds: number[] = [], supportIndices: number[] = [], moments: number[] = [];
  const cellVolumes = Float64Array.from(geometry.cells, c => c.volume), coveredVolumes = new Float64Array(geometry.cells.length);
  let incompleteCells = 0, maximumUncoveredFraction = 0;
  geometry.cells.forEach((cell, ordinal) => {
    if (options.requireFullyOpen && (!cell.openMoments || Math.abs(cell.openMoments.volume - cell.volume) > tolerance * cell.volume)) throw new Error("density box coupling requires certified fully-open cells");
    const hits: { index: number; box: Bounds }[] = [];
    query(cell, (index, box) => hits.push({ index, box }));
    hits.sort((a, b) => supports[a.index]!.id - supports[b.index]!.id);
    let covered = 0, compensation = 0;
    for (const hit of hits) {
      const support = supports[hit.index]!, overlapVolume = volume(hit.box);
      const adjusted = overlapVolume - compensation, next = covered + adjusted;
      compensation = (next - covered) - adjusted; covered = next;
      const integrals = support.lower.map((lo, axis) => {
        const span = support.upper[axis]! - lo;
        return axisIntegrals((hit.box.lower[axis]! - lo) / span, (hit.box.upper[axis]! - lo) / span).map(v => v * span);
      });
      supportIds.push(support.id); supportIndices.push(hit.index);
      moments.push(...integrals[0]!, ...integrals[1]!, ...integrals[2]!);
    }
    if (supportIds.length > 0xffff_ffff) throw new Error("density coupling exceeds u32 address space");
    offsets[ordinal + 1] = supportIds.length;
    coveredVolumes[ordinal] = covered;
    const uncovered = Math.max(0, (cell.volume - covered) / cell.volume);
    maximumUncoveredFraction = Math.max(maximumUncoveredFraction, uncovered);
    if (uncovered > tolerance) {
      incompleteCells++;
      if (options.requireCoverage !== false) throw new Error(`retained density support does not cover native cell ${cell.id}`);
    }
  });
  const cellIds = Uint32Array.from(geometry.cells, cell => cell.id), packedIds = Uint32Array.from(supportIds), packedIndices = Uint32Array.from(supportIndices), axisMoments = Float64Array.from(moments);
  const inverseCellVolumes = Float64Array.from(cellVolumes, v => 1 / v);
  return Object.freeze({ topologyGeneration: options.topologyGeneration, boundaryGeneration: options.boundaryGeneration, supportGeneration: options.supportGeneration,
    domain: "full-native-boxes" as const, supportGeometryKey, cellIds, offsets, supportIds: packedIds, supportIndices: packedIndices, axisMoments, inverseCellVolumes, cellVolumes, coveredVolumes,
    receipt: Object.freeze({ supportBoxes: supports.length, nativeCells: geometry.cells.length, intersections: packedIds.length, bvhNodes, boxTests,
      incompleteCells, maximumUncoveredFraction, supportIdentityUtf16Bytes: 2 * supportGeometryKey.length, couplingBytes: 2 * supportGeometryKey.length + cellIds.byteLength + offsets.byteLength + packedIds.byteLength + packedIndices.byteLength + axisMoments.byteLength + inverseCellVolumes.byteLength + cellVolumes.byteLength + coveredVolumes.byteLength }) });
}
/** Apply precompiled integration only. The accepted controls may evolve, while
 * a physics-only repartition compiles new coupling against identical controls. */
export function applyDensitySupportCoupling(coupling: DensitySupportCoupling, options: Readonly<{
  topologyGeneration: number; boundaryGeneration: number; supportGeneration: number;
  supportGeometryKey: string;
  controls: (supportId: number) => ArrayLike<number>;
}>): Float64Array {
  if (coupling.topologyGeneration !== options.topologyGeneration || coupling.boundaryGeneration !== options.boundaryGeneration || coupling.supportGeneration !== options.supportGeneration) throw new Error("stale retained density coupling");
  if (options.supportGeometryKey !== coupling.supportGeometryKey) throw new Error("retained density coupling support identity mismatch");
  const means = new Float64Array(coupling.cellIds.length);
  for (let cell = 0; cell < means.length; cell++) {
    let sum = 0, compensation = 0;
    for (let at = coupling.offsets[cell]!; at < coupling.offsets[cell + 1]!; at++) {
      const controls = options.controls(coupling.supportIds[at]!);
      if (controls.length !== 27) throw new Error("expected 27 Bernstein controls");
      for (let k = 0; k < 27; k++) {
        const value = controls[k]!;
        if (!Number.isFinite(value)) throw new Error("non-finite retained density control");
        const x = k % 3, y = Math.floor(k / 3) % 3, z = Math.floor(k / 9), base = 9 * at;
        const weight = coupling.axisMoments[base + x]! * coupling.axisMoments[base + 3 + y]! * coupling.axisMoments[base + 6 + z]! * coupling.inverseCellVolumes[cell]!;
        const adjusted = value * weight - compensation, next = sum + adjusted;
        compensation = (next - sum) - adjusted; sum = next;
      }
    }
    means[cell] = sum;
  }
  return means;
}
