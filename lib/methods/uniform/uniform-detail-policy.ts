/**
 * Uniform Geometric detail policy (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md,
 * "Dynamic transactions and temporal reuse" and "Production and UI contract").
 *
 * One typed place for the detail controls' vocabulary, the estimator
 * thresholds and the hysteresis/churn constants. Every number here is a
 * documented STARTING value for the first test build, not a validated
 * physics constant; tune them here, not at call sites.
 */
import type { MethodParamValues } from "../../core/method-contract";
import type { UniformDetailCriterion } from "./uniform-stage-grids";

export type UniformDetailPolicyMode = "requested" | "dynamic" | "full";
export const UNIFORM_DETAIL_POLICY_MODES: readonly UniformDetailPolicyMode[] = ["requested", "dynamic", "full"];

/** The product default: H = 4h base with explicitly requested h detail only. */
export const UNIFORM_DETAIL_APP_DEFAULT_POLICY: UniformDetailPolicyMode = "requested";
/**
 * The declared (lane) default. Verification lanes and probes resolve declared
 * defaults (MethodProfile.appDefaults' contract), and until the compact 4h-first
 * engine lands "dynamic" maps to the GPU surface census those lanes were
 * written against. The app opens on UNIFORM_DETAIL_APP_DEFAULT_POLICY through
 * `uniformVolumeMethod.appDefaults`; an explicit saved value always wins.
 */
export const UNIFORM_DETAIL_DECLARED_POLICY: UniformDetailPolicyMode = "dynamic";

/**
 * What the shape criterion measures on an h tile against the trilinear phi of
 * its eight 4h corners, and so what `detailShapeTolerance` (in h) bounds:
 * - "value": the largest |phi − trilinear| over the tile's h vertices within
 *   2h of the surface, with held corners read as distances
 *   (uniformDetailHeldDistance). It counts a wrong distance beside a surface
 *   that is in the right place.
 * - "displacement": how far the surface itself moves at 4h. At every zero
 *   crossing of the tile's h edges, the distance to the corners' zero set
 *   (|trilinear| over its gradient, the stored corner values: those are the
 *   surface 4h would carry). On a clean distance field it asks for about half
 *   the tiles "value" does at the same tolerance.
 * Either way a tolerance of 0 keeps every surface tile at h, and a 4h tile
 * refines on the second difference of its 4h lattice (unchanged).
 */
export type UniformDetailShapeMetric = "value" | "displacement";
export const UNIFORM_DETAIL_SHAPE_METRICS: readonly UniformDetailShapeMetric[] = ["value", "displacement"];

/** Runtime detail controls, all `update: "runtime"` (never rebuild or reseed). */
export const UNIFORM_DETAIL_PARAM_KEYS = Object.freeze([
  "detailPolicy", "detailNearFocus", "detailActivity",
  "detailBudgetPercent", "detailFocusRadiusPercent", "detailSensitivity",
  "detailSurface", "detailSurfaceDistance", "detailShape", "detailShapeTolerance", "detailShapeMetric", "detailThin", "detailThinThickness",
  "detailStrain", "detailStrainThreshold", "detailRotation", "detailRotationThreshold",
  "detailImpact", "detailImpactTravel", "detailApproach", "detailApproachSteps",
  "detailBulk", "detailMarginTiles", "detailHoldSteps", "detailSolidContact",
] as const);
export type UniformDetailParamKey = typeof UNIFORM_DETAIL_PARAM_KEYS[number];

/** Near focus is off (Peter, 2026-10-03): it holds h around the orbit target
 * whether or not liquid is there. The toggle stays for tuning.
 *
 * The importance criteria (UNIFORM_DETAIL_CRITERIA) declare the census the
 * Dawn lanes were written against: every surface tile at h (shape at
 * tolerance 0) plus the impact rule, nothing held, no budget. The app opens
 * on UNIFORM_DETAIL_APP_IMPORTANCE instead. */
export const UNIFORM_DETAIL_CONTROL_DEFAULTS = Object.freeze({
  detailPolicy: UNIFORM_DETAIL_DECLARED_POLICY as UniformDetailPolicyMode,
  detailNearFocus: "off", detailActivity: "on",
  detailBudgetPercent: 100, detailFocusRadiusPercent: 20, detailSensitivity: 0.5,
  detailSurface: "off", detailSurfaceDistance: 0, detailShape: "on", detailShapeTolerance: 0, detailShapeMetric: "value" as UniformDetailShapeMetric, detailThin: "off", detailThinThickness: 4,
  detailStrain: "off", detailStrainThreshold: 0.1, detailRotation: "off", detailRotationThreshold: 0.1,
  detailImpact: "on", detailImpactTravel: 1, detailApproach: "off", detailApproachSteps: 2,
  detailBulk: "off", detailMarginTiles: 0, detailHoldSteps: 0,
  detailSolidContact: "on",
});
/** What Dynamic opens on in the app (Peter, 2026-10-03: allow 4h at the
 * surface, fix what that exposes later): a smooth surface runs 4h, and the
 * surface criteria ask for h where 4h loses it. Strain and rotation stay off
 * until tuned; the importance layer scores them either way. */
export const UNIFORM_DETAIL_APP_IMPORTANCE = Object.freeze({
  detailShapeTolerance: 0.5, detailThin: "on", detailApproach: "on", detailHoldSteps: 8,
});
/** Allowed range of each numeric control: [min, max, step, digits]. */
export const UNIFORM_DETAIL_RANGES = Object.freeze({
  detailBudgetPercent: [0, 100, 1, 0], detailFocusRadiusPercent: [1, 100, 1, 0], detailSensitivity: [0, 1, 0.05, 2],
  detailSurfaceDistance: [0, 3, 1, 0],
  detailShapeTolerance: [0, 2, 0.05, 2], detailThinThickness: [1, 8, 0.5, 1],
  detailStrainThreshold: [0.01, 2, 0.01, 2], detailRotationThreshold: [0.01, 2, 0.01, 2],
  detailImpactTravel: [0, 64, 0.5, 1], detailApproachSteps: [1, 8, 0.5, 1],
  detailMarginTiles: [0, 8, 1, 0], detailHoldSteps: [0, 32, 1, 0],
} as const satisfies Partial<Record<UniformDetailParamKey, readonly [number, number, number, number]>>);

/** Capacity C against the h tiles `need` a census's last build asked for.
 * It grows, between frames, to (1 + headroom) need + floorTiles as soon as
 * (1 + grow) need would not fit: a build defers only when its need outruns
 * (1 + grow) times the last receipt's, which is two or three frames old.
 * hold: C never returns, and keeps the host's bound whatever the receipts
 * count (Dynamic every tile, a request its host count). A change of C
 * re-creates every buffer it sizes and binds every group again, 4 to 7 ms of
 * host time in the frame it lands on (NB-FLIP letters, 256x192x128), so the
 * storage the attach already had to hold is kept: memory for a frame time no
 * capacity event interrupts. Without hold it returns when the peak of the
 * last windowBuilds receipts, held the same way, is under C / returnRatio,
 * and none of them grew it. */
export interface UniformDetailCapacityRule {
  readonly grow: number;
  readonly headroom: number;
  readonly floorTiles: number;
  readonly windowBuilds: number;
  readonly returnRatio: number;
  readonly hold: boolean;
}
/** QA only (probes, before they build a solver). hold: false follows the
 * receipts down again. double: a growth is at least twice the held capacity.
 * grow < 0: growth only after a deferred build. lagFrames: a receipt is used
 * this many frames late. */
export interface UniformDetailCapacityRuleForQA extends UniformDetailCapacityRule { readonly double: boolean; readonly lagFrames: number }
let qaCapacityRule: Partial<UniformDetailCapacityRuleForQA> | undefined;
export function setUniformDetailCapacityRuleForQA(rule?: Partial<UniformDetailCapacityRuleForQA>): void { qaCapacityRule = rule; }
export function uniformDetailCapacityRule(): UniformDetailCapacityRuleForQA {
  return { ...UNIFORM_DETAIL_POLICY.capacity, double: false, lagFrames: 0, ...qaCapacityRule };
}

/** Starting constants of the planner and pool. */
export interface UniformDetailPolicy {
  /** Ownership/request granularity: one 4³ h tile = one H cell. */
  readonly tileCells: 4;
  /** Patch allocation unit in h cells per axis (32 or 16). */
  readonly patchCells: 16 | 32;
  /** Certified numerical support around admitted h tiles, in tiles (26-neighbour dilation). */
  readonly supportRadiusTiles: number;
  /** Automatic retirement: only below retireRatio × the promotion threshold ... */
  readonly retireRatio: number;
  /** ... for this many consecutive accepted steps. */
  readonly retireSteps: number;
  /** Automatic membership changes per accepted step, as a fraction of the domain's patch count. */
  readonly churnFraction: number;
  /** Pool growth factor (geometric). */
  readonly poolGrowth: number;
  /** h-tile capacity under a census (a layout whose need the liquid moves:
   * Dynamic, or requests with solid contact or bodies), from its builds'
   * receipts (WebGPUUniformReferenceSolver.followMixedCapacity). */
  readonly capacity: UniformDetailCapacityRule;
  /** Activity estimator base thresholds at sensitivity 0.5 (scaled by uniformDetailSensitivityScale). */
  readonly strainThreshold: number;
  readonly curvatureThreshold: number;
  /** Impending solid contact horizon, in accepted steps; a high-priority request. */
  readonly contactHorizonSteps: number;
  /** Request priorities: higher wins the budget and capacity first. */
  readonly priority: { readonly full: number; readonly region: number; readonly contact: number; readonly activity: number; readonly focus: number };
}

export const UNIFORM_DETAIL_POLICY: UniformDetailPolicy = Object.freeze({
  tileCells: 4,
  patchCells: 32,
  supportRadiusTiles: 1,
  retireRatio: 0.7,
  retireSteps: 8,
  churnFraction: 0.05,
  poolGrowth: 2,
  capacity: Object.freeze({ grow: 0.5, headroom: 1, floorTiles: 256, windowBuilds: 30, returnRatio: 1.5, hold: true }),
  strainThreshold: 0.1,
  curvatureThreshold: 0.5,
  contactHorizonSteps: 2,
  priority: Object.freeze({ full: 1000, region: 900, contact: 300, activity: 200, focus: 100 }),
});

/** `2^(2 - 4 s)`: s = 0.5 keeps the base thresholds; s = 1 divides them by 4 (more detail), s = 0 multiplies by 4. */
export function uniformDetailSensitivityScale(sensitivity: number): number {
  const s = Number.isFinite(sensitivity) ? Math.min(1, Math.max(0, sensitivity)) : 0.5;
  return 2 ** (2 - 4 * s);
}
/** Activity estimator thresholds at a sensitivity: dimensionless strain dt·|sym ∇u|, curvature H·|κ|. */
export function uniformDetailEstimatorThresholds(sensitivity: number, policy: UniformDetailPolicy = UNIFORM_DETAIL_POLICY) {
  const scale = uniformDetailSensitivityScale(sensitivity);
  return { strain: policy.strainThreshold * scale, curvature: policy.curvatureThreshold * scale };
}

/** Resolved runtime detail settings, from normalized method values. */
export interface UniformDetailSettings {
  /** Filter automatic detail by distance to a phi-crossing tile; never seed detail. */
  readonly surfaceOnly: boolean;
  /** Tile-neighbour distance from an actual surface crossing (0..3). */
  readonly surfaceDistance: number;
  readonly policy: UniformDetailPolicyMode;
  readonly nearFocus: boolean;
  readonly activity: boolean;
  readonly budgetPercent: number;
  readonly focusRadiusPercent: number;
  readonly sensitivity: number;
  /** Dynamic's importance criteria: which may require h, and each one's threshold. */
  readonly criteria: Readonly<Record<UniformDetailCriterion, boolean>>;
  /** Surface error, in h, a tile may take at 4h. 0: every surface tile is h. */
  readonly shapeTolerance: number;
  /** What that error is measured as (UniformDetailShapeMetric). */
  readonly shapeMetric: UniformDetailShapeMetric;
  /** Liquid or air thinner than this across the surface, in h, is h. */
  readonly thinThickness: number;
  /** dt·‖sym ∇u‖ and dt·‖curl u‖ at which a tile is h. */
  readonly strainThreshold: number;
  readonly rotationThreshold: number;
  /** Travel, in h per step, into or up a closed wall or solid at which a surface tile is h. */
  readonly impactTravel: number;
  /** A surface that meets a wall, a solid or another surface within this many steps is h. */
  readonly approachSteps: number;
  /** Strain and rotation also judge liquid tiles with no surface in them. */
  readonly bulk: boolean;
  /** Tiles of h kept around every tile a criterion requires. */
  readonly marginTiles: number;
  /** Steps a tile stays h after its last trigger (and while a score stays above retireRatio). */
  readonly holdSteps: number;
  /** Static solids ask for their contact tiles (and those tiles' neighbours)
   * at h. Off, under Requested, a solid asks for nothing: its tiles run as cut
   * 4h owners (uniformDetailCoarseSolids). */
  readonly solidContact: boolean;
}
export function uniformDetailPolicyMode(value: unknown): UniformDetailPolicyMode | undefined {
  return UNIFORM_DETAIL_POLICY_MODES.includes(value as UniformDetailPolicyMode) ? value as UniformDetailPolicyMode : undefined;
}
export function uniformDetailSettings(values: MethodParamValues): UniformDetailSettings {
  const number = (key: UniformDetailParamKey, lo: number, hi: number) => {
    const n = Number(values[key]);
    return Number.isFinite(n) ? Math.min(hi, Math.max(lo, n)) : UNIFORM_DETAIL_CONTROL_DEFAULTS[key] as number;
  };
  const ranged = (key: keyof typeof UNIFORM_DETAIL_RANGES) => number(key, UNIFORM_DETAIL_RANGES[key][0], UNIFORM_DETAIL_RANGES[key][1]);
  const on = (key: UniformDetailParamKey) => (values[key] ?? UNIFORM_DETAIL_CONTROL_DEFAULTS[key]) === "on";
  const shapeMetric = (values.detailShapeMetric ?? UNIFORM_DETAIL_CONTROL_DEFAULTS.detailShapeMetric) as UniformDetailShapeMetric;
  if (!UNIFORM_DETAIL_SHAPE_METRICS.includes(shapeMetric)) throw new Error(`Uniform detail shape metric must be one of ${UNIFORM_DETAIL_SHAPE_METRICS.join(", ")}: ${String(values.detailShapeMetric)}`);
  return {
    surfaceOnly: on("detailSurface"),
    surfaceDistance: Math.round(ranged("detailSurfaceDistance")),
    policy: uniformDetailPolicyMode(values.detailPolicy) ?? UNIFORM_DETAIL_DECLARED_POLICY,
    nearFocus: values.detailNearFocus === "on",
    activity: values.detailActivity !== "off",
    budgetPercent: ranged("detailBudgetPercent"),
    focusRadiusPercent: ranged("detailFocusRadiusPercent"),
    sensitivity: ranged("detailSensitivity"),
    criteria: { shape: on("detailShape"), thin: on("detailThin"), strain: on("detailStrain"), rotation: on("detailRotation"), impact: on("detailImpact"), approach: on("detailApproach") },
    shapeTolerance: ranged("detailShapeTolerance"),
    shapeMetric,
    thinThickness: ranged("detailThinThickness"),
    strainThreshold: ranged("detailStrainThreshold"),
    rotationThreshold: ranged("detailRotationThreshold"),
    impactTravel: ranged("detailImpactTravel"),
    approachSteps: ranged("detailApproachSteps"),
    bulk: on("detailBulk"),
    marginTiles: Math.round(ranged("detailMarginTiles")),
    holdSteps: Math.round(ranged("detailHoldSteps")),
    solidContact: on("detailSolidContact"),
  };
}
/** The parameter that switches each criterion, and the one that holds its threshold. */
export const UNIFORM_DETAIL_CRITERION_PARAMS = Object.freeze({
  shape: { toggle: "detailShape", threshold: "detailShapeTolerance" },
  thin: { toggle: "detailThin", threshold: "detailThinThickness" },
  strain: { toggle: "detailStrain", threshold: "detailStrainThreshold" },
  rotation: { toggle: "detailRotation", threshold: "detailRotationThreshold" },
  impact: { toggle: "detailImpact", threshold: "detailImpactTravel" },
  approach: { toggle: "detailApproach", threshold: "detailApproachSteps" },
} as const satisfies Record<UniformDetailCriterion, { toggle: UniformDetailParamKey; threshold: keyof typeof UNIFORM_DETAIL_RANGES }>);
const UNIFORM_DETAIL_NO_CRITERIA: Readonly<Record<UniformDetailCriterion, boolean>> = Object.freeze({ shape: false, thin: false, strain: false, rotation: false, impact: false, approach: false });
/** Whether these settings can ask for an h tile at all: any policy but
 * Requested, a Fine region, a rigid body, or the solid-contact request over
 * solids. Without one the layout is all 4h, no census runs and the h-tile
 * capacity is zero. */
export function uniformDetailRequests(settings: UniformDetailSettings, state: { fineRegions: boolean; solids: boolean; bodies: number }): boolean {
  return settings.policy !== "requested" || state.fineRegions || state.bodies > 0 || (settings.solidContact && state.solids);
}
/**
 * The census's importance policy at these settings. Sensitivity is one dial
 * over every threshold (uniformDetailSensitivityScale): a "less than" measure
 * (strain, rotation, travel, shape error) scales its threshold, a "more than"
 * one (thickness, horizon) divides it. A budget of 100% clips nothing.
 */
export function uniformDetailImportance(settings: UniformDetailSettings, tiles: number) {
  const scale = uniformDetailSensitivityScale(settings.sensitivity);
  // Requested and Full are Dynamic with the importance criteria off: the
  // census builds from requests only (Fine regions as builder statics, the
  // solid-contact request, bodies). No criterion, hold, budget or source rule.
  if (settings.policy !== "dynamic") return {
    surfaceOnly: false, surfaceDistance: 0, criteria: UNIFORM_DETAIL_NO_CRITERIA, shapeTolerance: 0, shapeMetric: "value" as UniformDetailShapeMetric, thinThickness: 0, strainThreshold: 0, rotationThreshold: 0,
    impactTravel: 0, approachSteps: 0, bulk: false, holdSteps: 0, retireRatio: UNIFORM_DETAIL_POLICY.retireRatio,
    budgetTiles: undefined as number | undefined, sources: false,
  };
  return {
    surfaceOnly: settings.surfaceOnly,
    surfaceDistance: settings.surfaceDistance,
    criteria: settings.criteria,
    shapeTolerance: settings.shapeTolerance * scale,
    shapeMetric: settings.shapeMetric,
    thinThickness: Math.min(UNIFORM_DETAIL_RANGES.detailThinThickness[1], settings.thinThickness / scale),
    strainThreshold: settings.strainThreshold * scale,
    rotationThreshold: settings.rotationThreshold * scale,
    impactTravel: settings.impactTravel * scale,
    approachSteps: settings.approachSteps / scale,
    bulk: settings.bulk,
    holdSteps: settings.holdSteps,
    retireRatio: UNIFORM_DETAIL_POLICY.retireRatio,
    budgetTiles: settings.budgetPercent >= 100 ? undefined : Math.floor(tiles * settings.budgetPercent / 100),
    sources: true,
  };
}
export type UniformDetailImportance = ReturnType<typeof uniformDetailImportance>;
/** Whether a surface tile may run at 4h. Requested holds one wherever no Fine
 * region covers it; Dynamic unless shape at tolerance 0 holds every one at h.
 * Such a surface rebuilds its 4h distance by travel, not every step: the
 * per-step rebuild is a linear instability of a resting 4h surface (a flat
 * all-4h pond at a fractional waterline grew 25% a step from float noise to
 * 1 m/s in 60 steps; uniform-coarse-solid-rest-dawn). */
export function uniformDetailCoarseSurface(settings: UniformDetailSettings): boolean {
  return settings.policy === "requested" || (settings.policy === "dynamic" && !(settings.criteria.shape && settings.shapeTolerance === 0));
}
/** Whether the census' shape criterion measures 4h value error (Dynamic,
 * shape at a tolerance above 0, the "value" metric): redistance then
 * publishes, for the wide vertices the travel gate holds, what a rebuild
 * would have changed. The "displacement" metric reads the stored corners. */
export function uniformDetailHeldDistance(settings: UniformDetailSettings): boolean {
  return settings.policy === "dynamic" && settings.criteria.shape && settings.shapeTolerance > 0 && settings.shapeMetric === "value";
}
/** Whether static solids run on cut 4h owners: Requested with no solid-contact
 * request. Dynamic's census and Full keep their solid tiles at h (the h band
 * has no cut Neumann face yet), and bodies are h in every policy. */
export function uniformDetailCoarseSolids(settings: UniformDetailSettings): boolean {
  return settings.policy === "requested" && !settings.solidContact;
}
/** Settings back to parameter values; a partial-values caller merges over these. */
export function uniformDetailValues(settings: UniformDetailSettings): MethodParamValues {
  const flag = (on: boolean) => on ? "on" : "off", c = settings.criteria;
  return { detailPolicy: settings.policy, detailNearFocus: flag(settings.nearFocus), detailActivity: flag(settings.activity),
    detailBudgetPercent: settings.budgetPercent, detailFocusRadiusPercent: settings.focusRadiusPercent, detailSensitivity: settings.sensitivity,
    detailSurface: flag(settings.surfaceOnly), detailSurfaceDistance: settings.surfaceDistance, detailShape: flag(c.shape), detailShapeTolerance: settings.shapeTolerance, detailShapeMetric: settings.shapeMetric, detailThin: flag(c.thin), detailThinThickness: settings.thinThickness,
    detailStrain: flag(c.strain), detailStrainThreshold: settings.strainThreshold, detailRotation: flag(c.rotation), detailRotationThreshold: settings.rotationThreshold,
    detailImpact: flag(c.impact), detailImpactTravel: settings.impactTravel, detailApproach: flag(c.approach), detailApproachSteps: settings.approachSteps,
    detailBulk: flag(settings.bulk), detailMarginTiles: settings.marginTiles, detailHoldSteps: settings.holdSteps,
    detailSolidContact: flag(settings.solidContact) } satisfies Record<UniformDetailParamKey, string | number>;
}
export function sameUniformDetailSettings(a: UniformDetailSettings, b: UniformDetailSettings): boolean {
  // Flat values in a fixed key order: equal settings serialize alike.
  return JSON.stringify(uniformDetailValues(a)) === JSON.stringify(uniformDetailValues(b));
}

/**
 * Retire the old `coarsening` control (regions | dynamic). Neither old value
 * is permission to enable automatic detail: every old configuration opens
 * Requested unless `detailPolicy` itself was saved. Authored regions are
 * scene data and are untouched. Returns the overrides with `coarsening`
 * removed and, when one was present, the notice to show once.
 */
export function migrateUniformDetailOverrides(overrides: MethodParamValues): { overrides: MethodParamValues; notice?: string } {
  if (!Object.hasOwn(overrides, "coarsening")) return { overrides };
  const { coarsening, ...rest } = overrides;
  if (uniformDetailPolicyMode(rest.detailPolicy)) return { overrides: rest };
  const old = coarsening === "regions" ? "Regions" : coarsening === "dynamic" ? "Dynamic" : String(coarsening);
  return { overrides: rest, notice: `Simulation detail: the retired “Coarsening: ${old}” setting now opens as Requested detail (4h base, Fine regions in h). Authored regions are kept.` };
}
