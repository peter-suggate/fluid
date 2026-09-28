import { UNIFORM_PARAMS } from "./parameters";
import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";

// Sharpening runs at the paper dose (strength 1) over the 2.1h band; the orphan
// relay and the sharpening admission band both assume that distance.
export const UNIFORM_GEOMETRIC_SHARPENING_STRENGTH = 1;
export const UNIFORM_GEOMETRIC_SHARPENING_DISTANCE = 2.1;
const omitted = new Set(["gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing", "activeRegion", "pressureCycleBudget", "pressureBudgetHeadroom"]);
// Retired controls: hardwired in the adapter. Unlike `omitted`, a saved value is
// ignored like any other undeclared key, so old configurations still load.
const retired = new Set(["sharpeningStrength", "sharpeningDistance", "velocityTransport", "liquidOnlyVelocityAdvection"]);
const params: MethodParamSpec[] = UNIFORM_PARAMS.filter(p => !omitted.has(p.key) && !retired.has(p.key)).map(p => {
  // Two sweeps: the front only needs to carry the band one cell per step, and the
  // hierarchy fill covers what it does not reach (docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.md
  // measured eight within 0.01 cell of sixteen; Peter set two 2026-09-19). The paper method keeps sixteen.
  if (p.key === "extensionFrontSweeps" && p.kind === "number") return { ...p, default: 2 };
  if (p.key === "densitySharpening" && p.kind === "select") return { ...p,
    label: "Volume sharpening", options: [{ value: "on", label: "On" }, { value: "off", label: "Off" }],
    hint: "Conservatively redistribute V toward the vertex level set without moving the surface." };
  return p;
});
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
});
params.push(
  {kind:"select",key:"phiCubicAdvection",label:"Cubic phi advection",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiCubicAdvection},
  {kind:"select",key:"phiDrain",label:"Drain ghost phi",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiDrain},
);

params.push({kind:"select",key:"totalSurfaceVolume",label:"Total surface volume",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],
  hint:"Shift the existing surface uniformly along its normals to match total conservative V. Bounded to one cell per step; no regional correction or phi seeding. One global constraint across all liquid bodies. Includes continuous inflow."});

params.push({kind:"select",key:"surfaceDeficitBalancing",label:"Surface-deficit balancing",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Preserve overfill expansion and balance it globally with contraction in underfilled pressure-liquid cells, weighted by the existing surface-fill estimate. Reduces persistent sloshing; capped to the available deficit per step."});

params.push({kind:"select",key:"coarsening",label:"Coarsening",default:"dynamic",tier:"coarse",update:"runtime",
  options:[{value:"regions",label:"Regions"},{value:"dynamic",label:"Dynamic"}],
  hint:"Regions: authored refinement regions choose coarse tiles. Dynamic: after every frame, only the tiles the surface can occupy during the next step (the RK2 departure boxes of their points hold the current surface) are fine; everything else coarsens to 4h. Authored regions can then only force fine. See docs/plans/uniform-dynamic-coarsening.md."});
params.push({kind:"number",key:"coarseningBoundaryTravel",label:"Boundary impact travel",default:1,tier:"coarse",update:"runtime",
  min:0,max:64,step:0.5,digits:1,unit:"cells/step",
  hint:"Surface liquid a wall or solid redirects is fine whatever its speed: liquid moving at least this many fine cells per step toward a closed wall or solid it reaches within one 4h cell (impact), or up a closed side wall faster than along it (lift). At 4h such a sheet is thinner than an owner can pressurise, so it piles up instead of climbing. Free fast fronts are unaffected. 0 disables."});

/** Shared by the studio and scene harnesses. */
export const UNIFORM_GEOMETRIC_PARAMS: readonly MethodParamSpec[] = Object.freeze(params);
/** Controls the 2D native port does not carry: GPU mixed-ownership scheduling and presentation. */
const gpuOnlyKeys = new Set(["orphanDustThreshold", "coarsening"]);
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
