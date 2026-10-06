/**
 * Uniform Geometric detail requests: authored regions, the whole-domain Full
 * request and transient automatic sources (focus, activity) merged into one
 * list the planner admits (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md).
 *
 * Precedence is decided here and in the planner, never by list order:
 *  - Full wins: one whole-domain request; regions add nothing to it.
 *  - Explicit Fine wins over explicit Coarse (the planner admits manual h
 *    first; a Coarse box is never an h request).
 *  - Explicit Coarse suppresses automatic (focus/activity) requests.
 *  - Numerical support of admitted h tiles may extend beyond authored bounds;
 *    the planner reports it separately from requested/admitted detail.
 *
 * Legacy min/max region records are read through the existing allowed-tier
 * semantics of createUniformMixedLayout's two-bit mask: allowed widths are
 * {h if lo <= 1 <= hi} ∪ {4h if lo <= 4 <= hi}, an empty set meaning 4h.
 *  - lo = 1, hi in {1, 2}         → Fine (explicit h request).
 *  - lo = 1, hi >= 4 or absent    → Automatic: both widths allowed, nothing
 *    enforced. A lower bound of 1 with an unbounded ceiling is NOT an h
 *    request; it neither requests nor suppresses detail.
 *  - lo >= 2 (2, 4, 8, 16, 32)    → Coarse (explicit 4h; other methods' 8-32
 *    cell regions mean the coarsest tier on this h/4h lattice).
 */
import type { FluidRefinementRegion, Vec3 } from "../../core/model";
import type { RefinementRegionLattice } from "../../core/refinement-regions";
import { UNIFORM_DETAIL_POLICY, type UniformDetailPolicy, type UniformDetailSettings } from "./uniform-detail-policy";

export type DetailRequestSource = "region" | "focus" | "activity" | "full";
export type DetailBounds = { readonly min: Vec3; readonly max: Vec3 };
export type DetailRequest = {
  id: string;
  bounds_m: DetailBounds;
  targetSpacing: "h";
  source: DetailRequestSource;
  priority: number;
  /** Live through this accepted step, inclusive; absent never expires. */
  expiresAfterStep?: number;
};
/** An explicit Coarse box: never an h request; masks automatic requests. */
export type DetailSuppression = { readonly id: string; readonly bounds_m: DetailBounds };
export type UniformRegionDetailTier = "fine" | "coarse" | "automatic";

/** Transient per-pane focus: the orbit/interaction target, never the camera eye. Not persisted. */
export interface DetailFocus { readonly revision: number; readonly position_m: Vec3 }

const CELL_SIZES = [1, 2, 4, 8, 16, 32];
/** Validate a region exactly as createUniformMixedLayout does and name its tier. */
export function uniformRegionDetailTier(region: FluidRefinementRegion): UniformRegionDetailTier {
  const lo = region.minimumCellSize_cells, hi = region.maximumCellSize_cells ?? Infinity;
  if (region.rule !== "minimum-cell-size" || !CELL_SIZES.includes(lo)
    || (region.maximumCellSize_cells !== undefined && !CELL_SIZES.includes(hi)) || hi < lo) {
    throw new Error(`Invalid cell-size bounds: ${region.id}`);
  }
  for (const axis of ["x", "y", "z"] as const) {
    if (!Number.isFinite(region.min_m[axis]) || !Number.isFinite(region.max_m[axis]) || region.min_m[axis] >= region.max_m[axis]) {
      throw new Error(`Invalid region bounds: ${region.id}`);
    }
  }
  const mask = (lo <= 1 && hi >= 1 ? 1 : 0) | (lo <= 4 && hi >= 4 ? 2 : 0) || 2;
  return mask === 1 ? "fine" : mask === 2 ? "coarse" : "automatic";
}

/** The lattice's physical extent. */
export function uniformDetailDomain(lattice: RefinementRegionLattice): DetailBounds {
  const o = lattice.origin_m, [nx, ny, nz] = lattice.dimensions, [hx, hy, hz] = lattice.cellSize_m;
  return { min: { ...o }, max: { x: o.x + nx * hx, y: o.y + ny * hy, z: o.z + nz * hz } };
}

/** Requests live at an accepted step (expiry is by accepted step, inclusive). */
export function liveDetailRequests<T extends Pick<DetailRequest, "expiresAfterStep">>(requests: readonly T[], acceptedStep: number): T[] {
  return requests.filter(r => r.expiresAfterStep === undefined || acceptedStep <= r.expiresAfterStep);
}

/** The focus request: a cube of half-extent focusRadiusPercent of the domain's longest extent, clipped to the domain. */
export function focusDetailRequest(focus: DetailFocus, domain: DetailBounds, radiusPercent: number,
  policy: UniformDetailPolicy = UNIFORM_DETAIL_POLICY): DetailRequest | undefined {
  const p = focus.position_m;
  if (![p.x, p.y, p.z].every(Number.isFinite)) return undefined;
  const r = uniformDetailFocusRadius_m(domain, radiusPercent);
  const clip = (v: number, a: "x" | "y" | "z") => Math.min(domain.max[a], Math.max(domain.min[a], v));
  const min = { x: clip(p.x - r, "x"), y: clip(p.y - r, "y"), z: clip(p.z - r, "z") };
  const max = { x: clip(p.x + r, "x"), y: clip(p.y + r, "y"), z: clip(p.z + r, "z") };
  if (min.x >= max.x || min.y >= max.y || min.z >= max.z) return undefined;
  return { id: "focus", bounds_m: { min, max }, targetSpacing: "h", source: "focus", priority: policy.priority.focus };
}
export function uniformDetailFocusRadius_m(domain: DetailBounds, radiusPercent: number): number {
  const extent = Math.max(domain.max.x - domain.min.x, domain.max.y - domain.min.y, domain.max.z - domain.min.z);
  return extent * Math.min(100, Math.max(0, radiusPercent)) / 100;
}

export interface DetailRequestSet {
  readonly requests: DetailRequest[];
  readonly suppressions: DetailSuppression[];
  /** Automatic sources the planner may admit this step (empty outside Dynamic). */
  readonly automaticSources: ReadonlySet<"focus" | "activity">;
  /** Regions whose bounds enforce nothing (lo = 1, unbounded ceiling). */
  readonly unconstrainedRegions: string[];
}

/** Merge policy, authored regions and transient sources into one request list. */
export function buildUniformDetailRequests(input: {
  readonly settings: UniformDetailSettings;
  readonly regions: readonly FluidRefinementRegion[];
  readonly domain: DetailBounds;
  readonly acceptedStep: number;
  readonly focus?: DetailFocus;
  /** Activity requests from an estimator (not yet produced by the existing solver). */
  readonly activity?: readonly DetailRequest[];
  readonly policy?: UniformDetailPolicy;
}): DetailRequestSet {
  const policy = input.policy ?? UNIFORM_DETAIL_POLICY, ids = new Set<string>();
  const requests: DetailRequest[] = [], suppressions: DetailSuppression[] = [], unconstrainedRegions: string[] = [];
  for (const region of input.regions) {
    if (!region.id || ids.has(region.id)) throw new Error(`Duplicate or empty region id: ${region.id}`);
    ids.add(region.id);
    const tier = uniformRegionDetailTier(region), bounds_m = { min: { ...region.min_m }, max: { ...region.max_m } };
    if (tier === "fine") requests.push({ id: `region:${region.id}`, bounds_m, targetSpacing: "h", source: "region", priority: policy.priority.region });
    else if (tier === "coarse") suppressions.push({ id: `region:${region.id}`, bounds_m });
    else unconstrainedRegions.push(region.id);
  }
  const { settings } = input, automaticSources = new Set<"focus" | "activity">();
  if (settings.policy === "full") {
    // Full wins: Fine boxes add nothing and Coarse boxes suppress nothing manual.
    return { requests: [{ id: "full", bounds_m: { min: { ...input.domain.min }, max: { ...input.domain.max } }, targetSpacing: "h", source: "full", priority: policy.priority.full }],
      suppressions: [], automaticSources, unconstrainedRegions };
  }
  if (settings.policy === "dynamic") {
    if (settings.nearFocus) {
      automaticSources.add("focus");
      const focus = input.focus && focusDetailRequest(input.focus, input.domain, settings.focusRadiusPercent, policy);
      if (focus) requests.push(focus);
    }
    if (settings.activity) {
      automaticSources.add("activity");
      requests.push(...(input.activity ?? []).filter(r => r.source === "activity"));
    }
  }
  return { requests: liveDetailRequests(requests, input.acceptedStep), suppressions, automaticSources, unconstrainedRegions };
}

/** What the accepted generation answered, for the UI's pending-change indicator. */
export function uniformDetailRequestKey(policy: string, regions: readonly FluidRefinementRegion[] | undefined): string {
  return `${policy}|${(regions ?? []).map(r => `${r.id}:${r.minimumCellSize_cells}:${r.maximumCellSize_cells ?? "-"}:${[r.min_m.x, r.min_m.y, r.min_m.z, r.max_m.x, r.max_m.y, r.max_m.z].join(",")}`).join(";")}`;
}
