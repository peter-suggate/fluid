import { UNIFORM_PARAMS } from "./parameters";
import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";
import { UNIFORM_DETAIL_CONTROL_DEFAULTS, UNIFORM_DETAIL_PARAM_KEYS, UNIFORM_DETAIL_RANGES } from "./uniform-detail-policy";

// Sharpening runs at the paper dose (strength 1). Its band (default 2.1h) is
// one uniform the admission test and the orphan relay both read.
export const UNIFORM_GEOMETRIC_SHARPENING_STRENGTH = 1;
export const UNIFORM_GEOMETRIC_SHARPENING_DISTANCE = 2.1;
/** Sharpening sweeps ping-pong V through its scratch: the count is even, so the last sweep writes V.
 * Zero sweeps, or a zero band, skips the stage as its Off switch does. */
export const UNIFORM_GEOMETRIC_SHARPENING_SWEEPS = 8;
export const UNIFORM_GEOMETRIC_SHARPENING_MAX_SWEEPS = 16;
export const uniformGeometricSharpeningSweeps = (value: unknown): number => {
  const n = Number(value);
  return Number.isFinite(n) ? 2 * Math.round(Math.min(UNIFORM_GEOMETRIC_SHARPENING_MAX_SWEEPS, Math.max(0, n)) / 2) : UNIFORM_GEOMETRIC_SHARPENING_SWEEPS;
};
/** Secant Newton rounds of the total surface-volume shift; zero skips the stage as its Off switch does. */
export const UNIFORM_GEOMETRIC_SURFACE_VOLUME_ROUNDS = 2;
export const UNIFORM_GEOMETRIC_SURFACE_VOLUME_MAX_ROUNDS = 6;
export const uniformGeometricSurfaceVolumeRounds = (value: unknown): number => {
  const n = Number(value);
  return Number.isFinite(n) ? Math.round(Math.min(UNIFORM_GEOMETRIC_SURFACE_VOLUME_MAX_ROUNDS, Math.max(0, n))) : UNIFORM_GEOMETRIC_SURFACE_VOLUME_ROUNDS;
};
const omitted = new Set(["gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing", "activeRegion", "pressureCycleBudget", "pressureBudgetHeadroom"]);
// Retired controls: hardwired in the adapter. Unlike `omitted`, a saved value is
// ignored like any other undeclared key, so old configurations still load.
const retired = new Set(["sharpeningStrength", "velocityTransport", "liquidOnlyVelocityAdvection"]);
const params: MethodParamSpec[] = UNIFORM_PARAMS.filter(p => !omitted.has(p.key) && !retired.has(p.key)).map(p => {
  // Two sweeps: the front only needs to carry the band one cell per step, and the
  // hierarchy fill covers what it does not reach (docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.md
  // measured eight within 0.01 cell of sixteen; Peter set two 2026-09-19). The paper method keeps sixteen.
  if (p.key === "extensionFrontSweeps" && p.kind === "number") return { ...p, default: 2 };
  if (p.key === "densitySharpening" && p.kind === "select") return { ...p,
    label: "Volume sharpening", options: [{ value: "on", label: "On" }, { value: "off", label: "Off" }],
    hint: "Conservatively redistribute V toward the vertex level set without moving the surface." };
  if (p.key === "sharpeningDistance" && p.kind === "number") return { ...p,
    label: "Sharpening band", default: UNIFORM_GEOMETRIC_SHARPENING_DISTANCE, min: 0,
    hint: "Admission band of volume sharpening, in cells of each owner's own width: an owner whose centre phi lies within it gives and takes V; outside it, dilute orphan V is relayed toward the surface. Narrower bands visit fewer tiles. Zero skips sharpening: a zero band would make every owner on or outside the surface an orphan." };
  return p;
});
params.push({kind:"number",key:"sharpeningSweeps",label:"Sharpening sweeps",default:UNIFORM_GEOMETRIC_SHARPENING_SWEEPS,tier:"fine",update:"runtime",
  min:0,max:UNIFORM_GEOMETRIC_SHARPENING_MAX_SWEEPS,step:2,digits:0,unit:"sweeps",
  hint:"Propose/limit/commit face-transfer sweeps per step, three launches each. Each sweep moves V at most one owner, so fewer sweeps sharpen less per step. Even only: the sweeps ping-pong V through a scratch field. Zero skips sharpening, its dust clearing included."});
params.push({kind:"number",key:"surfaceVolumeRounds",label:"Surface volume rounds",default:UNIFORM_GEOMETRIC_SURFACE_VOLUME_ROUNDS,tier:"fine",update:"runtime",
  min:0,max:UNIFORM_GEOMETRIC_SURFACE_VOLUME_MAX_ROUNDS,step:1,digits:0,unit:"rounds",
  hint:"Secant Newton rounds of the total surface-volume shift, each a measure, reduce and solve over the band. A converged solve skips the remaining rounds' work but not their launches. Zero skips the shift, band build included, as switching it off does."});
params.push({kind:"number",key:"volumeDustThreshold",label:"Volume dust floor",default:1e-3,tier:"fine",update:"runtime",
  min:0,max:1e-3,step:1e-7,digits:7,unit:"cell volumes",
  hint:"Discard |V| below this outside the 4h surface band, plus ULP-scale negatives. Positive deposits near the surface are preserved. The 1e-3 default removes far-air residue that keeps transport tiles active; the diagnostics report discarded mass. Zero is off and stores the untreated sum bit for bit."});
params.push({kind:"number",key:"orphanDustThreshold",label:"Orphan dust floor",default:0.01,tier:"fine",update:"runtime",
  min:0,max:0.05,step:0.001,digits:3,unit:"cell volumes",
  hint:"Extra cleanup once per step for dilute volume beyond the 4h surface band. Preserves clusters holding a quarter cell or containing a cell above 5%, cut cells, and surface neighbours. Requires the regular dust floor. Discarded mass is reported; total surface correction matches remaining V and does not replace it. Zero disables extra cleanup."});

// Splash survival: cubic advection and the ghost drain are on by default and
// stay switchable for comparison in the app.
/** The long-form tooltips, shared by the parameter and its SIM panel switch. */
export const UNIFORM_GEOMETRIC_SPLASH_HINTS = Object.freeze({
  phiCubicAdvection: [
    "Moves the level set with a cubic resample instead of a trilinear one, so curved surfaces -- drops, sheet rims, a splash crown -- stop shrinking a little on every step they move.",
    "Why: each step, every vertex traces back along the velocity (RK2) and reads phi at its departure point. A signed distance field is convex outside a convex body, and trilinear interpolation overestimates a convex function, so each resample pushes the zero contour inward by up to h²/4r. Flat pools (r infinite) do not notice; small, fast, curved features erode fastest, which is why only splashes look dissipative.",
    "How: a vertex whose trilinear read is within 2 cells of the surface re-reads phi with a separable Catmull-Rom cubic over the 4×4×4 vertices around its departure point. Catmull-Rom reproduces quadratics exactly, so the curvature bias drops from second to third order. The result is clamped to the lowest and highest of the 8 vertices enclosing the departure point, so it adds no new extrema: no overshoot or ringing at a crease or a thin sheet.",
    "Scope and cost: 64 taps instead of 8, on band vertices only; deep liquid and far air keep the trilinear read. Frame time within run-to-run noise on the crown splash.",
    "Measured (cm12-figure-6, frame 90, this toggle alone): V inside phi 80,425 → 84,409 of about 94,000 cell volumes; V stranded in the relay band 7,303 → 4,794. No effect on a drop at rest: it does not move, so nothing is resampled. The resting shrink comes from redistancing.",
    "Off: the trilinear resample.",
  ].join("\n\n"),
  phiDrain: [
    "Removes ghost liquid: level set that still says liquid where no volume is left behind it. The surface stops drawing water that has already gone, and V and phi agree about where the liquid is.",
    "Why: V (the conservative per-cell volume) decides how much liquid there is; phi only decides where it is, and phi is merely advected. It can keep a liquid region after V has left it: a sheet whose V was sharpened into the pool, a drop whose V was relayed away, a sliver trailing a fast front. Nothing else in the step removes it except the global total-volume shift, which takes the difference off every surface at once, drops and pool alike.",
    "How: during phi advection, a vertex whose advected phi is below half a cell (liquid, or just outside the surface) checks the 4×4×4 cells around its departure point, using the start-of-step V. If none holds more than 5% of a cell, phi rises by half a cell this step, capped at +h/2, just outside the surface. A ghost two cells deep is air after four steps and settles at +h/2 on the fifth; redistancing retires unsupported positive plateaus to the 4h band edge only after checking that no zero crossing remains within that band.",
    "Safety: any cell with real V nearby leaves the vertex alone, so a genuine surface -- even a film holding 5% of a cell -- is never drained, and resting drops and pools are untouched (every liquid vertex has V beneath it).",
    "Scope and cost: a 64-cell read, only for band vertices below h/2. Off also stops that retirement, which keeps far-air tiles fine after a splash.",
    "Off: ghost phi stays until the total-volume shift or a merge absorbs it.",
  ].join("\n\n"),
  phiPreserveSurface: [
    "Stops redistancing from moving the surface, so drops and thin sheets keep their level set instead of shrinking a little every step. Experimental: off by default.",
    "Why: after advection, redistancing rebuilds every band vertex as the distance to the trilinear zero set. For a convex body -- a drop, a sheet rim -- that zero set lies inside the true surface, so each rebuild moves the surface inward, at rest and in flight. Large pools do not notice; a drop of radius 2 cells loses about half its phi volume in half a second of free fall while its V is untouched, and the difference becomes invisible water.",
    "How: CM11b sec. 3.4 -- do not modify phi next to the surface. An h vertex with a face neighbour of the opposite sign keeps its advected value; every other band vertex is redistanced as before. 4h vertices keep the native rule.",
    "Measured (thin-liquid ladders, free fall at 0.5 s, phi volume as a share of V): drop r = 2 cells 46% → 91%, r = 3 73% → 97%; sheets 3-4 cells thick about 80% → 90%; free and wall films 2-4 cells thick 93% → 97%. One-cell sheets and r = 1 drops are below what the vertex lattice can hold and are not rescued.",
    "Watch for: surface noise on large calm pools, which the skipped rebuild no longer smooths. It removes work (no Newton search on those vertices).",
    "Off: every band vertex is redistanced.",
  ].join("\n\n"),
});
params.push(
  {kind:"select",key:"phiCubicAdvection",label:"Cubic phi advection",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiCubicAdvection},
  {kind:"select",key:"phiDrain",label:"Drain ghost phi",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiDrain},
  {kind:"select",key:"phiPreserveSurface",label:"Preserve surface vertices",default:"off",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiPreserveSurface},
);

params.push({kind:"select",key:"totalSurfaceVolume",label:"Total surface volume",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],
  hint:"Shift the existing surface uniformly along its normals to match total conservative V. Bounded to one cell per step; no regional correction or phi seeding. One global constraint across all liquid bodies. Includes continuous inflow."});

params.push({kind:"select",key:"surfaceDeficitBalancing",label:"Surface-deficit balancing",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Preserve overfill expansion and balance it globally with contraction in underfilled pressure-liquid cells, weighted by the existing surface-fill estimate. Reduces persistent sloshing; capped to the available deficit per step."});

// Simulation detail (docs/plans/uniform-4h-first-implementation-handoff-2026-10-03.md):
// runtime-only, so a policy change or a moved focus never rebuilds or reseeds.
// The Simulation detail control (components/UniformCoarseControl.tsx) owns them.
// The retired `coarsening` control (regions|dynamic) is ignored like any
// undeclared key; persistence migrates it (migrateUniformDetailOverrides).
params.push(
  {kind:"select",key:"detailPolicy",label:"Simulation detail",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailPolicy,tier:"coarse",update:"runtime",dedicated:true,
    options:[{value:"requested",label:"Requested"},{value:"dynamic",label:"Dynamic"},{value:"full",label:"Full"}],
    hint:"Requested: a 4h base everywhere, with h only in Fine regions you draw. Dynamic: Fine regions plus automatic detail at active water (this build: the surface-driven census), and near the focus when that is switched on. Full: h detail across the whole domain through the same planner."},
  {kind:"select",key:"detailNearFocus",label:"Detail near focus",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailNearFocus,tier:"coarse",update:"runtime",dedicated:true,
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:"Dynamic only: hold h detail around the orbit target (not the camera eye), with or without liquid there. Off by default."},
  {kind:"select",key:"detailActivity",label:"Detail for activity",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailActivity,tier:"coarse",update:"runtime",dedicated:true,
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:"Reserved for host-side activity requests; Dynamic's importance criteria below decide h on the GPU whatever this says."},
  {kind:"number",key:"detailBudgetPercent",label:"Automatic detail budget",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailBudgetPercent,tier:"coarse",update:"runtime",dedicated:true,
    min:0,max:100,step:1,digits:0,unit:"%",
    hint:"Most of the domain Dynamic's criteria may hold at h, closure and travel included. Over it, the lowest-scoring tiles are dropped first (from the frame after). Fine regions, solid contact and Full are never clipped. 100 clips nothing."},
  {kind:"number",key:"detailFocusRadiusPercent",label:"Focus radius",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailFocusRadiusPercent,tier:"coarse",update:"runtime",dedicated:true,
    min:1,max:100,step:1,digits:0,unit:"% of extent",
    hint:"Half-width of the focus request, as a share of the domain's longest physical extent."},
  {kind:"number",key:"detailSensitivity",label:"Detail sensitivity",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailSensitivity,tier:"coarse",update:"runtime",dedicated:true,
    min:0,max:1,step:0.05,digits:2,unit:"",
    hint:"One dial over every importance threshold, scaled by 2^(2 − 4 × sensitivity): 0.5 leaves them as set, 1 asks for h four times sooner, 0 four times later."},
);
// Dynamic's importance criteria (UNIFORM_DETAIL_CRITERIA): each scores a tile
// as its measure over its threshold, and a score of 1 requires the tile at h.
// The declared defaults are the census the lanes were written against; the
// app's are UNIFORM_DETAIL_APP_IMPORTANCE.
const detailToggle=(key:keyof typeof UNIFORM_DETAIL_CONTROL_DEFAULTS,label:string,hint:string):MethodParamSpec=>({kind:"select",key,label,default:UNIFORM_DETAIL_CONTROL_DEFAULTS[key] as string,tier:"coarse",update:"runtime",dedicated:true,
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],hint});
const detailNumber=(key:keyof typeof UNIFORM_DETAIL_RANGES,label:string,unit:string,hint:string):MethodParamSpec=>{const [min,max,step,digits]=UNIFORM_DETAIL_RANGES[key];
  return {kind:"number",key,label,default:UNIFORM_DETAIL_CONTROL_DEFAULTS[key],tier:"coarse",update:"runtime",dedicated:true,min,max,step,digits,unit,hint};};
params.push(
  detailToggle("detailSurface","Surface","Only tiles within the Surface distance of a surface-crossing tile are eligible for automatic detail. Filters the existing selection, including its margin and hold; never adds fine tiles. Explicit requests and required solid contact remain."),
  detailNumber("detailSurfaceDistance","Surface distance","tiles","Allowed distance from a surface-crossing tile. 0: crossing tiles only; 1–3: that many neighbouring tile layers, including diagonals. Filters existing refinement; does not request it."),
  detailToggle("detailShape","Detail for surface shape","A surface tile is h where 4h corners cannot carry its shape."),
  detailNumber("detailShapeTolerance","Shape tolerance","h","Surface error a tile may take at 4h: an h tile coarsens when trilinear phi from its 4h corners stays within it; a 4h tile refines when its second difference predicts twice that. 0 keeps every surface tile at h."),
  {kind:"select",key:"detailShapeMetric",label:"Shape metric",default:UNIFORM_DETAIL_CONTROL_DEFAULTS.detailShapeMetric,tier:"coarse",update:"runtime",dedicated:true,
    options:[{value:"value",label:"Value error"},{value:"displacement",label:"Surface displacement"}],
    hint:"What the shape tolerance bounds on an h tile. Value error: the largest gap between phi and the trilinear phi of its 4h corners within 2h of the surface. Surface displacement: how far the surface itself moves when the tile is carried by its 4h corners; it asks for about half the tiles at the same tolerance."},
  detailToggle("detailThin","Detail for thin features","A surface tile is h where the liquid, or the air gap, across it is thin: sheets, films, jets, necks and closing gaps that a 4h owner cannot pressurise."),
  detailNumber("detailThinThickness","Thin thickness","h","Liquid or air thinner than this across the surface is h. Measured as twice the deepest phi on either side within a tile of the surface."),
  detailToggle("detailStrain","Detail for deformation","A tile is h where the flow stretches or shears fast: dt·‖sym ∇u‖ across its neighbours."),
  detailNumber("detailStrainThreshold","Deformation threshold","/step","Deformation per step (dt·‖sym ∇u‖) at which a tile is h. 0.1: a tile-sized blob changes shape by a tenth in a step."),
  detailToggle("detailRotation","Detail for rotation","A tile is h where the flow turns fast: dt·‖curl u‖ across its neighbours. Vortices and rolling fronts that 4h smears."),
  detailNumber("detailRotationThreshold","Rotation threshold","rad/step","Rotation per step (dt·‖curl u‖) at which a tile is h."),
  detailToggle("detailImpact","Detail for wall impact","Surface liquid a wall or solid redirects is h: moving into a closed wall or solid it reaches within one 4h cell (impact), or up a closed side wall faster than along it (lift)."),
  detailNumber("detailImpactTravel","Impact travel","cells/step","Travel toward (or up) the wall at which the impact rule fires. At 4h such a sheet is thinner than an owner can pressurise, so it piles up instead of climbing. Free fast fronts are unaffected."),
  detailToggle("detailApproach","Detail for approaching contact","A surface tile is h before it meets a wall, a solid or another surface: the gap ahead of it closes within the horizon."),
  detailNumber("detailApproachSteps","Approach horizon","steps","A surface that closes its gap within this many steps is h. Refining a fast front early has held the long-dam toe short of the wall: watch the toe."),
  detailToggle("detailBulk","Detail in bulk liquid","Deformation and rotation also judge liquid tiles with no surface in them. Off: only surface tiles."),
  detailNumber("detailMarginTiles","Detail margin","tiles","Tiles of h kept around every tile a criterion requires, on top of the surface's travel over the step."),
  detailNumber("detailHoldSteps","Detail hold","steps","Steps a tile stays h after its last trigger, and for as long as a score stays above 0.7. 0 follows the criteria step by step."),
  detailToggle("detailSolidContact","Detail at solids","Static solids hold the tiles they cut, and those tiles' neighbours, at h. Off (Requested only): a solid asks for no detail and its tiles run as cut 4h owners; Fine regions still apply. Dynamic and Full keep solid contact at h, and rigid bodies are always h."),
);

/** Shared by the studio and scene harnesses. */
export const UNIFORM_GEOMETRIC_PARAMS: readonly MethodParamSpec[] = Object.freeze(params);
/** Controls the 2D native port does not carry: GPU mixed-ownership scheduling and presentation. */
const gpuOnlyKeys = new Set<string>(["orphanDustThreshold", "sharpeningDistance", "sharpeningSweeps", "surfaceVolumeRounds", "phiPreserveSurface", ...UNIFORM_DETAIL_PARAM_KEYS]);
export const UNIFORM_GEOMETRIC_NATIVE_PARAMS = Object.freeze(params.filter(p => !gpuOnlyKeys.has(p.key) && !p.key.startsWith("coarsening")));
export const UNIFORM_GEOMETRIC_DEFAULTS: Readonly<MethodParamValues> = Object.freeze(
  Object.fromEntries(params.map(p => [p.key, p.default])),
);
/** Removed experimental controls never survive a saved configuration reload. */
export function resolveUniformGeometricValues(values: MethodParamValues = {}): MethodParamValues {
  // An omitted key is a control this method fixes, not one it forgot: dropping
  // such an override quietly let callers believe they had changed the solve
  // (pressureCycleBudget above all) while the adapter hard-coded it back. Keys
  // this method never declared are still ignored -- that is the reload contract.
  for (const key of Object.keys(values)) {
    if (!omitted.has(key)) continue;
    throw new Error(`Uniform Geometric fixes "${key}" and cannot take it as an override; `
      + `see uniformGeometricSolverOptions. Drop the override or change the method.`);
  }
  return Object.fromEntries(params.map(spec => {
    const raw = values[spec.key];
    const numeric = spec.kind === "number" ? numberValue(values, params, spec.key) : 0;
    const value = spec.kind === "number" ? (spec.step === 1 ? Math.round(numeric) : numeric)
      : spec.options.some(option => option.value === raw) ? raw! : spec.default;
    return [spec.key, value];
  }));
}
