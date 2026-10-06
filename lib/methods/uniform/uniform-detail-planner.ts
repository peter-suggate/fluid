/**
 * Uniform Geometric detail planner (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md).
 *
 * Pure CPU planning at 4h-tile granularity (one tile = 4³ h cells = one H
 * cell): request → requested tile mask (world bounds snapped outward to
 * tiles exactly as createUniformMixedLayout snaps regions), admission
 * (manual first, whole requests, against a hard patch capacity; automatic
 * tiles by rank against the soft budget measured after support closure),
 * certified support closure (26-neighbour dilation of admitted h tiles by
 * `supportRadiusTiles`), the patch-rounded residency set, hysteresis
 * (promote at the request mask, retire only outside the 1/retireRatio
 * retention box for retireSteps accepted steps) and the automatic churn
 * limit (distinct patches whose automatic membership changes per step).
 *
 * plan() never mutates: a candidate that is not accepted leaves the planner
 * at its last accepted state; commit() adopts one. Every iteration is in
 * world-key order, so equal inputs give identical plans.
 */
import type { RefinementRegionLattice } from "../../core/refinement-regions";
import { UNIFORM_DETAIL_POLICY, type UniformDetailPolicy } from "./uniform-detail-policy";
import type { DetailBounds, DetailRequest, DetailRequestSource, DetailSuppression } from "./uniform-detail-requests";

type Triple = readonly [number, number, number];
type AutomaticSource = "focus" | "activity";

/** Per admitted tile: why it is h. */
export const UNIFORM_DETAIL_REASON = Object.freeze({ region: 1, full: 2, focus: 4, activity: 8 } satisfies Record<DetailRequestSource, number>);
const MANUAL: ReadonlySet<DetailRequestSource> = new Set(["region", "full"]);

/** World box → tile box, upper bound exclusive; undefined when it covers no cell. The
 * arithmetic of refinementRegionCellBounds + createUniformMixedLayout, so a Fine region
 * names the same tiles on every path. */
export function uniformDetailTileBounds(lattice: RefinementRegionLattice, bounds: DetailBounds): { min: Triple; max: Triple } | undefined {
  const axes = ["x", "y", "z"] as const, cells = (v: number, a: number) =>
    Math.max(0, Math.min(lattice.dimensions[a]!, (v - lattice.origin_m[axes[a]!]) / lattice.cellSize_m[a]!));
  const lo = axes.map((k, a) => cells(bounds.min[k], a)), hi = axes.map((k, a) => cells(bounds.max[k], a));
  if (lo.some((v, a) => v >= hi[a]!)) return undefined;
  // World-to-cell arithmetic can land a few ULPs past an exact tile edge.
  const tile = (v: number) => { const q = v / 4, n = Math.round(q); return Math.abs(q - n) <= 8 * Number.EPSILON * Math.max(1, Math.abs(q)) ? n : q; };
  return { min: lo.map(v => Math.floor(tile(v))) as unknown as Triple, max: hi.map(v => Math.ceil(tile(v))) as unknown as Triple };
}

export interface UniformDetailPlanInput {
  readonly requests: readonly DetailRequest[];
  readonly suppressions?: readonly DetailSuppression[];
  /** Automatic sources enabled this step; a member whose sources are all disabled drops at once. */
  readonly automaticSources?: ReadonlySet<AutomaticSource>;
  readonly acceptedStep: number;
  /** Soft budget for automatic requests, % of all tiles, counted after support closure. */
  readonly budgetPercent: number;
  /** Hard limit in resident patches (the pool's byte budget). Absent: the whole domain. */
  readonly capacityPatches?: number;
}

export interface UniformDetailPlanDiagnostics {
  readonly tiles: number;
  /** Tiles some live request names (automatic ones after suppression). */
  readonly requestedTiles: number;
  /** Admitted h tiles (manual ∪ automatic). */
  readonly admittedTiles: number;
  readonly manualTiles: number;
  readonly automaticTiles: number;
  /** Certified support outside admitted tiles; may leave authored bounds. */
  readonly supportTiles: number;
  /** Manual tiles whose whole request was deferred by capacity. */
  readonly deferredTiles: number;
  /** Automatic candidates the soft budget clipped. */
  readonly budgetClippedTiles: number;
  /** Automatic candidates/retirements held back by the churn limit. */
  readonly churnDeferredTiles: number;
  /** Automatic candidates the hard capacity refused. */
  readonly capacityDeferredTiles: number;
  readonly budgetTiles: number;
  /** Closure tiles the automatic set adds beyond the manual closure. */
  readonly automaticCostTiles: number;
  readonly churnPatches: number;
  readonly churnLimitPatches: number;
  readonly patchCells: number;
  readonly domainPatches: number;
  readonly residentPatches: number;
  /** residentPatches × patchCells³. */
  readonly allocatedCells: number;
  /** Admitted h cells (admittedTiles × 64). */
  readonly activeCells: number;
  readonly supportCells: number;
  /** Allocated cells in partly occupied patches holding neither admitted nor support tiles (incl. past the domain edge). */
  readonly wastedCells: number;
  /** Admitted tiles by reason (a tile may carry several). */
  readonly reasons: Readonly<Record<DetailRequestSource, number>>;
}

export interface UniformDetailPlan {
  readonly step: number;
  readonly tileDimensions: Triple;
  readonly fine: Uint8Array;
  readonly support: Uint8Array;
  readonly reasons: Uint8Array;
  /** Resident world patch keys, ascending. */
  readonly patches: Uint32Array;
  /** Manual request ids deferred whole by capacity. */
  readonly deferred: readonly string[];
  readonly diagnostics: UniformDetailPlanDiagnostics;
  /** The automatic state commit() adopts. */
  readonly next: { readonly members: Uint8Array; readonly sources: Uint8Array; readonly low: Uint8Array };
}

export class UniformDetailPlanner {
  readonly tileDimensions: Triple;
  readonly tiles: number;
  readonly tilesPerPatch: number;
  readonly patchDimensions: Triple;
  readonly domainPatches: number;
  private members: Uint8Array;
  private sources: Uint8Array;
  private low: Uint8Array;
  /** Members counting toward retirement, kept by commit(); `settled` is read every frame. */
  private counting = 0;
  private acceptedStep = -Infinity;
  constructor(readonly lattice: RefinementRegionLattice, readonly policy: UniformDetailPolicy = UNIFORM_DETAIL_POLICY) {
    if (lattice.dimensions.some(n => !Number.isSafeInteger(n) || n < 4 || n % 4 !== 0)) throw new Error("Detail planning requires dimensions divisible by four");
    if (policy.patchCells !== 16 && policy.patchCells !== 32) throw new Error(`Detail patches are 16³ or 32³ h cells, not ${policy.patchCells}`);
    this.tileDimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
    this.tiles = this.tileDimensions[0] * this.tileDimensions[1] * this.tileDimensions[2];
    this.tilesPerPatch = policy.patchCells / 4;
    this.patchDimensions = this.tileDimensions.map(n => Math.ceil(n / this.tilesPerPatch)) as unknown as Triple;
    this.domainPatches = this.patchDimensions[0] * this.patchDimensions[1] * this.patchDimensions[2];
    this.members = new Uint8Array(this.tiles);this.sources = new Uint8Array(this.tiles);this.low = new Uint8Array(this.tiles);
  }
  /** No automatic member is counting toward retirement: an unchanged input replans to the same set. */
  get settled(): boolean { return this.counting === 0; }
  get automaticTiles(): number { let n = 0; for (const m of this.members) n += m; return n; }
  patchKeyOfTile(t: number): number {
    const [tx, ty] = this.tileDimensions, x = t % tx, y = Math.floor(t / tx) % ty, z = Math.floor(t / (tx * ty)), p = this.tilesPerPatch;
    return Math.floor(x / p) + this.patchDimensions[0] * (Math.floor(y / p) + this.patchDimensions[1] * Math.floor(z / p));
  }
  /** Tile keys a world box covers, ascending. */
  tilesOf(bounds: DetailBounds): number[] {
    const box = uniformDetailTileBounds(this.lattice, bounds);if (!box) return [];
    const [tx, ty] = this.tileDimensions, out: number[] = [];
    for (let z = box.min[2]; z < box.max[2]; z++) for (let y = box.min[1]; y < box.max[1]; y++) for (let x = box.min[0]; x < box.max[0]; x++) out.push(x + tx * (y + ty * z));
    return out;
  }

  plan(input: UniformDetailPlanInput): UniformDetailPlan {
    const n = this.tiles, policy = this.policy, [tx, ty, tz] = this.tileDimensions, radius = Math.max(0, Math.round(policy.supportRadiusTiles));
    const capacity = Math.max(0, Math.floor(input.capacityPatches ?? this.domainPatches));
    const budgetTiles = Math.floor(Math.min(100, Math.max(0, input.budgetPercent)) / 100 * n);
    const churnLimit = Math.max(1, Math.ceil(policy.churnFraction * this.domainPatches));
    const patchOf = new Uint32Array(n);for (let t = 0; t < n; t++) patchOf[t] = this.patchKeyOfTile(t);
    // Closure tracker: coverage counts, closure size and per-patch residency.
    const cover = new Uint32Array(n), patchCount = new Uint32Array(this.domainPatches);
    let closure = 0, resident = 0;
    const visit = (t: number, delta: 1 | -1) => {
      const x = t % tx, y = Math.floor(t / tx) % ty, z = Math.floor(t / (tx * ty));
      for (let k = Math.max(0, z - radius); k <= Math.min(tz - 1, z + radius); k++) for (let j = Math.max(0, y - radius); j <= Math.min(ty - 1, y + radius); j++)
        for (let i = Math.max(0, x - radius); i <= Math.min(tx - 1, x + radius); i++) {
          const q = i + tx * (j + ty * k), p = patchOf[q]!;
          if (delta > 0) { if (cover[q]++ === 0) { closure++; if (patchCount[p]++ === 0) resident++; } }
          else if (--cover[q] === 0) { closure--; if (--patchCount[p] === 0) resident--; }
        }
    };
    const fine = new Uint8Array(n), reasons = new Uint8Array(n), requested = new Uint8Array(n);
    // Manual requests: whole requests, priority first, ids break ties; Full and Fine regions.
    const manual = input.requests.filter(r => MANUAL.has(r.source)).sort((a, b) => b.priority - a.priority || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    const deferred: string[] = [];let deferredTiles = 0;
    for (const request of manual) {
      const tiles = this.tilesOf(request.bounds_m), added: number[] = [];
      for (const t of tiles) { requested[t] = 1; if (!fine[t]) { fine[t] = 1; added.push(t); visit(t, 1); } }
      if (resident > capacity) {
        for (const t of added) { fine[t] = 0; visit(t, -1); }
        deferred.push(request.id);deferredTiles += added.length;continue;
      }
      for (const t of tiles) reasons[t]! |= UNIFORM_DETAIL_REASON[request.source];
    }
    const manualTiles = fine.reduce((s, v) => s + v, 0), manualClosure = closure;
    // Automatic: suppression (explicit Coarse), promotion masks, retention boxes and rank.
    const enabled = [...(input.automaticSources ?? [])].reduce((bits, s) => bits | UNIFORM_DETAIL_REASON[s], 0);
    const suppressed = new Uint8Array(n);
    for (const s of input.suppressions ?? []) for (const t of this.tilesOf(s.bounds_m)) suppressed[t] = 1;
    const candidate = new Uint8Array(n), retain = new Uint8Array(n), priority = new Float64Array(n).fill(-Infinity), score = new Float64Array(n);
    const centre = (t: number, a: number) => this.lattice.origin_m[(["x", "y", "z"] as const)[a]!] + (([t % tx, Math.floor(t / tx) % ty, Math.floor(t / (tx * ty))][a]! + 0.5) * 4) * this.lattice.cellSize_m[a]!;
    for (const request of input.requests) {
      if (MANUAL.has(request.source)) continue;
      const bit = UNIFORM_DETAIL_REASON[request.source];if (!(enabled & bit)) continue;
      const b = request.bounds_m, mid = [(b.min.x + b.max.x) / 2, (b.min.y + b.max.y) / 2, (b.min.z + b.max.z) / 2];
      const half = [(b.max.x - b.min.x) / 2, (b.max.y - b.min.y) / 2, (b.max.z - b.min.z) / 2];
      const tiles = this.tilesOf(b);
      for (const t of tiles) {
        if (suppressed[t] || fine[t]) continue;
        requested[t] = 1;candidate[t]! |= bit;
        let d = 0;for (let a = 0; a < 3; a++) d = Math.max(d, Math.abs(centre(t, a) - mid[a]!) / Math.max(half[a]!, 1e-30));
        const s = 1 / Math.max(d, 1e-9);
        if (request.priority > priority[t]! || (request.priority === priority[t] && s > score[t]!)) { priority[t] = request.priority; score[t] = s; }
      }
      const r = 1 / policy.retireRatio, outer = { min: { x: mid[0]! - r * half[0]!, y: mid[1]! - r * half[1]!, z: mid[2]! - r * half[2]! },
        max: { x: mid[0]! + r * half[0]!, y: mid[1]! + r * half[1]!, z: mid[2]! + r * half[2]! } };
      for (const t of this.tilesOf(outer)) if (!suppressed[t]) retain[t]! |= bit;
    }
    const members = this.members.slice(), sources = this.sources.slice(), low = this.low.slice(), advanced = input.acceptedStep > this.acceptedStep;
    const touched = new Uint8Array(this.domainPatches);let churn = 0, churnDeferred = 0;
    const churnAllows = (t: number) => touched[patchOf[t]!] !== 0 || churn < churnLimit;
    const spendChurn = (t: number) => { if (!touched[patchOf[t]!]) { touched[patchOf[t]!] = 1; churn++; } };
    // Immediate drops (user actions, not churn): sources switched off, Coarse-suppressed, now manual.
    for (let t = 0; t < n; t++) {
      if (!members[t]) continue;
      sources[t]! &= enabled;
      if (!sources[t] || suppressed[t] || fine[t]) { members[t] = 0;sources[t] = 0;low[t] = 0;continue; }
      sources[t]! |= candidate[t]!;
      // Only an accepted step counts toward retirement; a paused replan does not.
      low[t] = retain[t] ? 0 : advanced ? Math.min(255, low[t]! + 1) : low[t]!;
      visit(t, 1);
    }
    const rank = (a: number, b: number) => priority[b]! - priority[a]! || score[b]! - score[a]! || a - b;
    const retire = (t: number) => { members[t] = 0;sources[t] = 0;low[t] = 0;visit(t, -1);spendChurn(t); };
    // Hysteresis retirement, churn-limited, world-key order.
    for (let t = 0; t < n; t++) if (members[t] && low[t]! >= policy.retireSteps) { if (churnAllows(t)) retire(t); else churnDeferred++; }
    // Hard capacity, then the soft budget: evict the lowest-ranked members.
    const ranked = () => { const list: number[] = [];for (let t = 0; t < n; t++) if (members[t]) list.push(t);return list.sort(rank); };
    if (resident > capacity) for (const t of ranked().reverse()) { if (resident <= capacity) break;members[t] = 0;sources[t] = 0;low[t] = 0;visit(t, -1);spendChurn(t); }
    if (closure - manualClosure > budgetTiles) for (const t of ranked().reverse()) {
      if (closure - manualClosure <= budgetTiles) break;
      if (churnAllows(t)) retire(t); else churnDeferred++;
    }
    // Predictive promotion by rank, within budget, capacity and churn.
    let budgetClipped = 0, capacityDeferred = 0;
    const promotions: number[] = [];for (let t = 0; t < n; t++) if (candidate[t] && !members[t]) promotions.push(t);
    for (const t of promotions.sort(rank)) {
      visit(t, 1);
      if (resident > capacity) { visit(t, -1);capacityDeferred++;continue; }
      if (closure - manualClosure > budgetTiles) { visit(t, -1);budgetClipped++;continue; }
      if (!churnAllows(t)) { visit(t, -1);churnDeferred++;continue; }
      spendChurn(t);members[t] = 1;sources[t] = candidate[t]!;low[t] = 0;
    }
    let automaticTiles = 0;
    for (let t = 0; t < n; t++) if (members[t]) { fine[t] = 1;reasons[t]! |= sources[t]!;automaticTiles++; }
    const support = new Uint8Array(n), patches: number[] = [];let supportTiles = 0;
    for (let t = 0; t < n; t++) if (cover[t] && !fine[t]) { support[t] = 1;supportTiles++; }
    for (let p = 0; p < this.domainPatches; p++) if (patchCount[p]) patches.push(p);
    const count = (bit: number) => reasons.reduce((s, r) => s + (r & bit ? 1 : 0), 0);
    const admittedTiles = manualTiles + automaticTiles, patchVolume = policy.patchCells ** 3;
    return {
      step: input.acceptedStep, tileDimensions: this.tileDimensions, fine, support, reasons, patches: Uint32Array.from(patches), deferred,
      next: { members, sources, low },
      diagnostics: {
        tiles: n, requestedTiles: requested.reduce((s, v) => s + v, 0), admittedTiles, manualTiles, automaticTiles, supportTiles,
        deferredTiles, budgetClippedTiles: budgetClipped, churnDeferredTiles: churnDeferred, capacityDeferredTiles: capacityDeferred,
        budgetTiles, automaticCostTiles: closure - manualClosure, churnPatches: churn, churnLimitPatches: churnLimit,
        patchCells: policy.patchCells, domainPatches: this.domainPatches, residentPatches: resident,
        allocatedCells: resident * patchVolume, activeCells: admittedTiles * 64, supportCells: supportTiles * 64,
        wastedCells: resident * patchVolume - closure * 64,
        reasons: { region: count(UNIFORM_DETAIL_REASON.region), full: count(UNIFORM_DETAIL_REASON.full),
          focus: count(UNIFORM_DETAIL_REASON.focus), activity: count(UNIFORM_DETAIL_REASON.activity) },
      },
    };
  }
  /** Adopt an accepted plan's automatic state. Plans must be committed in accepted-step order. */
  commit(plan: UniformDetailPlan): void {
    if (plan.step < this.acceptedStep) throw new Error(`Detail plan for step ${plan.step} is older than accepted step ${this.acceptedStep}`);
    if (plan.next.members.length !== this.tiles) throw new Error("Detail plan does not match the planner's tile lattice");
    this.members = plan.next.members.slice();this.sources = plan.next.sources.slice();this.low = plan.next.low.slice();this.acceptedStep = plan.step;
    let counting = 0;for (const v of this.low) if (v) counting++;this.counting = counting;
  }
}

type ManualSource = "region" | "full";
/** A manual-only request list (Requested: Fine regions; Full: the domain) held
 * as tiles and moved by the difference of two lists. `fine` and the
 * diagnostics are plan()'s for the same requests with no automatic source
 * and no patch capacity, at the cost of the tiles of the boxes that left and
 * entered (times the support stencil), never of the lattice. */
export class UniformDetailRequestTiles {
  readonly tileDimensions: Triple;
  readonly tiles: number;
  /** 1 where a request holds the tile. */
  readonly fine: Uint8Array;
  /** Per tile: UNIFORM_DETAIL_REASON bits of the sources holding it. */
  readonly reasons: Uint8Array;
  private readonly holds: Partial<Record<ManualSource, Uint16Array>> = {};
  /** Admitted tiles within the support radius of each tile; resident tiles per patch. */
  private readonly cover: Uint16Array;
  private readonly patchCount: Uint32Array;
  private readonly radius: number;
  private readonly tilesPerPatch: number;
  private readonly patchDimensions: Triple;
  private boxes = new Map<string, { source: ManualSource; min: Triple; max: Triple; count: number }>();
  private admitted = 0;
  private closure = 0;
  private resident = 0;
  private held: Record<ManualSource, number> = { region: 0, full: 0 };
  constructor(readonly lattice: RefinementRegionLattice, readonly policy: UniformDetailPolicy = UNIFORM_DETAIL_POLICY) {
    if (lattice.dimensions.some(n => !Number.isSafeInteger(n) || n < 4 || n % 4 !== 0)) throw new Error("Detail planning requires dimensions divisible by four");
    this.tileDimensions = lattice.dimensions.map(n => n / 4) as unknown as Triple;
    this.tiles = this.tileDimensions[0] * this.tileDimensions[1] * this.tileDimensions[2];
    this.tilesPerPatch = policy.patchCells / 4;this.radius = Math.max(0, Math.round(policy.supportRadiusTiles));
    this.patchDimensions = this.tileDimensions.map(n => Math.ceil(n / this.tilesPerPatch)) as unknown as Triple;
    this.fine = new Uint8Array(this.tiles);this.reasons = new Uint8Array(this.tiles);this.cover = new Uint16Array(this.tiles);
    this.patchCount = new Uint32Array(this.patchDimensions[0] * this.patchDimensions[1] * this.patchDimensions[2]);
  }
  private visit(t: number, delta: 1 | -1): void {
    const [tx, ty, tz] = this.tileDimensions, r = this.radius, p = this.tilesPerPatch, [px, py] = this.patchDimensions, cover = this.cover, patchCount = this.patchCount;
    const x = t % tx, y = Math.floor(t / tx) % ty, z = Math.floor(t / (tx * ty));
    for (let k = Math.max(0, z - r); k <= Math.min(tz - 1, z + r); k++) for (let j = Math.max(0, y - r); j <= Math.min(ty - 1, y + r); j++)
      for (let i = Math.max(0, x - r); i <= Math.min(tx - 1, x + r); i++) {
        const q = i + tx * (j + ty * k), patch = Math.floor(i / p) + px * (Math.floor(j / p) + py * Math.floor(k / p));
        if (delta > 0) { if (cover[q]!++ === 0) { this.closure++; if (patchCount[patch]!++ === 0) this.resident++; } }
        else if (--cover[q]! === 0) { this.closure--; if (--patchCount[patch]! === 0) this.resident--; }
      }
  }
  /** Make the set these requests' tiles (manual sources only). Returns the
   * tiles that entered and the tiles that left, disjoint: boxes are added
   * before the ones no longer asked are taken away, so a tile both lists
   * hold never flips. */
  set(requests: readonly DetailRequest[]): { entered: number[]; left: number[] } {
    if (requests.length > 0xffff) throw new Error(`Detail requests: ${requests.length} boxes exceed the per-tile hold count`);
    const next: typeof this.boxes = new Map(), entered: number[] = [], left: number[] = [], [tx, ty] = this.tileDimensions, fine = this.fine, reasons = this.reasons;
    for (const request of requests) {
      if (!MANUAL.has(request.source)) throw new Error(`Detail request ${request.id} is not a manual request`);
      const box = uniformDetailTileBounds(this.lattice, request.bounds_m);if (!box) continue;
      const key = `${request.source}:${box.min.join(",")}:${box.max.join(",")}`, same = next.get(key);
      if (same) same.count++; else next.set(key, { source: request.source as ManualSource, min: box.min, max: box.max, count: 1 });
    }
    const each = (b: { min: Triple; max: Triple }, f: (t: number) => void) => {
      for (let z = b.min[2]; z < b.max[2]; z++) for (let y = b.min[1]; y < b.max[1]; y++) for (let x = b.min[0]; x < b.max[0]; x++) f(x + tx * (y + ty * z));
    };
    for (const [key, b] of next) {
      const add = b.count - (this.boxes.get(key)?.count ?? 0);if (add <= 0) continue;
      const holds = this.holds[b.source] ??= new Uint16Array(this.tiles), bit = UNIFORM_DETAIL_REASON[b.source];
      each(b, t => {
        const before = holds[t]!;holds[t] = before + add;if (before) return;
        this.held[b.source]++;const other = reasons[t]!;reasons[t] = other | bit;
        if (!other) { fine[t] = 1;this.admitted++;entered.push(t);this.visit(t, 1); }
      });
    }
    for (const [key, b] of this.boxes) {
      const drop = b.count - (next.get(key)?.count ?? 0);if (drop <= 0) continue;
      const holds = this.holds[b.source]!, bit = UNIFORM_DETAIL_REASON[b.source];
      each(b, t => {
        if ((holds[t] = holds[t]! - drop)) return;
        this.held[b.source]--;
        if (!(reasons[t] = reasons[t]! & ~bit)) { fine[t] = 0;this.admitted--;left.push(t);this.visit(t, -1); }
      });
    }
    this.boxes = next;
    return { entered, left };
  }
  get admittedTiles(): number { return this.admitted; }
  diagnostics(budgetPercent: number): UniformDetailPlanDiagnostics {
    const n = this.tiles, policy = this.policy, domainPatches = this.patchCount.length, patchVolume = policy.patchCells ** 3, support = this.closure - this.admitted;
    return {
      tiles: n, requestedTiles: this.admitted, admittedTiles: this.admitted, manualTiles: this.admitted, automaticTiles: 0, supportTiles: support,
      deferredTiles: 0, budgetClippedTiles: 0, churnDeferredTiles: 0, capacityDeferredTiles: 0,
      budgetTiles: Math.floor(Math.min(100, Math.max(0, budgetPercent)) / 100 * n), automaticCostTiles: 0, churnPatches: 0,
      churnLimitPatches: Math.max(1, Math.ceil(policy.churnFraction * domainPatches)),
      patchCells: policy.patchCells, domainPatches, residentPatches: this.resident,
      allocatedCells: this.resident * patchVolume, activeCells: this.admitted * 64, supportCells: support * 64,
      wastedCells: this.resident * patchVolume - this.closure * 64,
      reasons: { region: this.held.region, full: this.held.full, focus: 0, activity: 0 },
    };
  }
}
