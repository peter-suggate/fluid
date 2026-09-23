import { UNIFORM_PARAMS } from "./parameters";
import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";

const omitted = new Set(["gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing", "activeRegion", "pressureCycleBudget", "pressureBudgetHeadroom"]);
const params: MethodParamSpec[] = UNIFORM_PARAMS.filter(p => !omitted.has(p.key)).map(p => {
  if (p.key === "velocityTransport" && p.kind === "select") return { ...p, default: "semi-lagrangian", update: "solver" as const };
  // Two sweeps: the front only needs to carry the band one cell per step, and the
  // hierarchy fill covers what it does not reach (docs/benchmarks/uniform-extension-front-sweeps-2026-09-19.md
  // measured eight within 0.01 cell of sixteen; Peter set two 2026-09-19). The paper method keeps sixteen.
  if (p.key === "extensionFrontSweeps" && p.kind === "number") return { ...p, default: 2 };
  if (p.key === "densitySharpening" && p.kind === "select") return { ...p,
    label: "Volume sharpening", options: [{ value: "on", label: "On" }, { value: "off", label: "Off" }],
    hint: "Conservatively redistribute V toward the vertex level set without moving the surface." };
  if (p.key === "sharpeningStrength" && p.kind === "number") return { ...p, min: 0, max: 1,
    hint: "Fraction of available donor volume transferred toward phi capacity each sweep." };
  if (p.key === "sharpeningDistance") return { ...p, label: "Sharpening band",
    hint: "Only cells within this distance of phi=0 participate in local volume return." };
  return p;
});
params.push({kind:"select",key:"pageSize",label:"Page size",default:"32",tier:"coarse",update:"solver",
  options:[{value:"16",label:"16³"},{value:"32",label:"32³"}],
  hint:"Cells along each domain page edge. Changing this rebuilds the solver and resets the simulation to time zero."});
params.push({kind:"select",key:"redistance",label:"Level-set redistancing",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],hint:"Reconstruct metric distance near phi=0 after transport. Disable to isolate contour drift."});
params.push({kind:"select",key:"sharpeningWorkMap",label:"Sharpening work map",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Skip sharpening work in 4×4×4 tiles with no cell in the admission band. Identical result to the dense schedule; Off retains the dense control for comparison."});
params.push({kind:"number",key:"volumeDustThreshold",label:"Volume dust floor",default:1e-3,tier:"fine",update:"runtime",
  min:0,max:1e-3,step:1e-7,digits:7,unit:"cell volumes",
  hint:"Discard |V| below this wherever transport or sharpening writes V, ULP-scale negatives included. The 1e-3 default removes residue that keeps transport tiles active; the diagnostics report discarded mass. Zero is off and stores the untreated sum bit for bit."});
params.push({kind:"select",key:"twoLevelVelocity",label:"Two-level velocity sampler",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Outside the fine tile map, sample velocity from the 4h face table the extension hierarchy publishes instead of the finest lattice, and let the extension, advection and projection skip the far-air tiles. Off is the all-fine dense control."});
params.push({kind:"number",key:"twoLevelFineReach",label:"Fine reach",default:2,tier:"fine",update:"runtime",
  min:0,max:8,step:1,digits:0,unit:"tiles",
  hint:"Chebyshev dilation of the seed tiles (liquid above the dust floor, solids, sources, or a vertex inside the 4h band). It must cover a step's backward trace plus the stencils that read beyond it."});
params.push({kind:"select",key:"twoLevelExtension",label:"Extension work",default:"tiles",tier:"fine",update:"runtime",
  options:[{value:"tiles",label:"Shell tiles"},{value:"dense",label:"Dense"}],
  hint:"Run the extension's finest passes — authority aside — only in the shell tiles, the fine set dilated by one more tile. Only meaningful with the two-level sampler on, since the shrunk fine field is exactly what the 4h sampler replaces. Dense retains the full-lattice schedule for comparison."});
params.push({kind:"number",key:"twoLevelShellReach",label:"Shell reach",default:1,tier:"fine",update:"runtime",
  min:0,max:8,step:1,digits:0,unit:"tiles",
  hint:"Extra 4h tiles the extension's working set adds past the fine set. It must cover every stencil the extension's finest passes read beyond a fine tile; raise it if far-air velocity looks wrong near the fine boundary."});
params.push({kind:"select",key:"twoLevelAdvection",label:"Advection work",default:"tiles",tier:"fine",update:"runtime",
  options:[{value:"tiles",label:"Fine tiles"},{value:"dense",label:"Dense"}],
  hint:"Outside the fine tiles, velocity advection and the projection take their far-air arm directly: no backward traces, no face data, no pressure taps. Those cells have no pressure row and no liquid neighbour, so the projection overwrote the advected value anyway. Only meaningful with the two-level sampler on."});
params.push({kind:"select",key:"transportWorkMap",label:"Transport work",default:"tiles",tier:"fine",update:"runtime",
  options:[{value:"tiles",label:"Live tiles"},{value:"dense",label:"Dense"}],
  hint:"Run every pass of the conservative volume transport only on the 4h tiles that can hold or receive liquid this step. Donor columns normalise over the receivers that sample them, so this is exactly conservative while every cell outside the set holds V=0 — which is what the dust floor guarantees. Dense retains the full-lattice schedule, and is forced when the dust floor or the two-level sampler is off."});
params.push({kind:"number",key:"transportReach",label:"Transport margin",default:1,tier:"fine",update:"runtime",
  min:0,max:8,step:1,digits:0,unit:"tiles",
  hint:"Extra 4h tiles added to the reach this step's own measured maximum displacement requires. The classify measures that displacement one dispatch before the dilation reads it, so the set already tracks the flow; this is only headroom. Zero is the exact predicate."});
params.push({kind:"select",key:"volumePressureRows",label:"Volume pressure rows",default:"off",tier:"fine",update:"runtime",
  options:[{value:"abandoned",label:"Abandoned cells"},{value:"all",label:"All"},{value:"off",label:"Off"}],
  hint:"Let a cell holding at least half its open capacity in V own a pressure row even where the level set reads dry. Without it a film thinner than half a cell has no pressure, its volume stacks against the far wall with nothing to push it back, and once no cell centre is liquid the solve stops. Abandoned cells grants the row only where no face neighbour is phi-liquid; All also lets V move the free surface beside phi-liquid, which roughens every surface. Off is the phi-only control."});
// phi/V agreement (docs/uniform-geometric-phi-volume-agreement-handoff.md): three
// independent stages, all off until they have been judged in the app.
params.push({kind:"select",key:"volumeCompaction",label:"Volume compaction",default:"off",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Let sharpening pour V from a liquid cell into a deeper liquid neighbour at any depth, not only inside the 2.1h band. The level set deletes entrained air but V keeps the void, and nothing else refills it: the dam break's deep interior settles a third full while the displaced volume piles on the surface."});
params.push({kind:"select",key:"phiSeedFromVolume",label:"Seed phi from volume",default:"off",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Where no cell centre nearby is phi-liquid but the cells around a vertex average over a quarter full, write V's implied depth into phi. Keeps a film thinner than half a cell alive: the seeded cells own ordinary pressure rows and render. Never fires beside an existing phi surface."});
params.push({kind:"select",key:"phiAgreement",label:"Phi follows volume",default:"off",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Move band phi along its normal by a slow, smooth residual: V minus phi's fill, gathered over the 8x8x8 cells around each vertex with tent weights. Cancels phi's transport drift without reading per-cell V as geometry. Wants compaction on, or the residual has the wrong sign."});
params.push({kind:"number",key:"phiAgreementGain",label:"Agreement gain",default:0.05,tier:"fine",update:"runtime",
  min:0,max:1,step:0.01,digits:2,unit:"cells / residual",
  hint:"Cells of normal shift per unit of patch residual. 0.25 roughened the dam break threefold; 0.05 stays inside baseline noise."});
params.push({kind:"number",key:"phiAgreementClamp",label:"Agreement clamp",default:0.02,tier:"fine",update:"runtime",
  min:0,max:0.5,step:0.005,digits:3,unit:"cells / step",
  hint:"Largest shift in one step. At gain 0.05 the dam break does not care (0.01 to 0.05 all read the same roughness); the thin film does: 0.002 cannot keep up with its erosion, 0.02 and 0.05 hold phi within 6-14% of V."});

// Splash survival (docs/uniform-geometric-splash-dissipation-plan.md): independent
// stages for comparing in the app. Cubic advection, the ghost drain and airborne
// momentum ship on; the rest are off. 3D only -- not in the native contract.
/** The long-form tooltips, shared by the parameter and its SIM panel switch. */
export const UNIFORM_GEOMETRIC_SPLASH_HINTS = Object.freeze({
  phiCubicAdvection: [
    "Moves the level set with a cubic resample instead of a trilinear one, so curved surfaces -- drops, sheet rims, a splash crown -- stop shrinking a little on every step they move.",
    "Why: each step, every vertex traces back along the velocity (RK2) and reads phi at its departure point. A signed distance field is convex outside a convex body, and trilinear interpolation overestimates a convex function, so each resample pushes the zero contour inward by up to h²/4r. Flat pools (r infinite) do not notice; small, fast, curved features erode fastest, which is why only splashes look dissipative.",
    "How: a vertex whose trilinear read is within 2 cells of the surface re-reads phi with a separable Catmull-Rom cubic over the 4×4×4 vertices around its departure point. Catmull-Rom reproduces quadratics exactly, so the curvature bias drops from second to third order. The result is clamped to the lowest and highest of the 8 vertices enclosing the departure point, so it adds no new extrema: no overshoot or ringing at a crease or a thin sheet.",
    "Scope and cost: 64 taps instead of 8, on band vertices only; deep liquid and far air keep the trilinear read. A far-air vertex the lean path would skip is still advected when it traces back into a two-level shell tile, so airborne drops get the cubic read too. Frame time within run-to-run noise on the crown splash.",
    "Measured (cm12-figure-6, frame 90, this toggle alone): V inside phi 80,425 → 84,409 of about 94,000 cell volumes; V stranded in the relay band 7,303 → 4,794. No effect on a drop at rest: it does not move, so nothing is resampled. The resting shrink comes from redistancing (see Redistance surface).",
    "Off: the trilinear resample.",
  ].join("\n\n"),
  phiDrain: [
    "Removes ghost liquid: level set that still says liquid where no volume is left behind it. The surface stops drawing water that has already gone, and V and phi agree about where the liquid is.",
    "Why: V (the conservative per-cell volume) decides how much liquid there is; phi only decides where it is, and phi is merely advected. It can keep a liquid region after V has left it: a sheet whose V was sharpened into the pool, a drop whose V was relayed away, a sliver trailing a fast front. Nothing else in the step removes it except the global total-volume shift, which takes the difference off every surface at once, drops and pool alike.",
    "How: during phi advection, a vertex whose advected phi is below half a cell (liquid, or just outside the surface) checks the 4×4×4 cells around its departure point, using the start-of-step V. If none holds more than 5% of a cell, phi rises by half a cell this step, capped at +h/2, just outside the surface. A ghost two cells deep is air after four steps and settles at +h/2 on the fifth; redistancing then rebuilds a proper distance around what remains.",
    "Safety: any cell with real V nearby leaves the vertex alone, so a genuine surface -- even a film holding 5% of a cell -- is never drained, and resting drops and pools are untouched (every liquid vertex has V beneath it).",
    "Scope and cost: a 64-cell read, only for band vertices below h/2.",
    "Off: ghost phi stays until the total-volume shift or a merge absorbs it.",
  ].join("\n\n"),
  airborneMomentum: [
    "Liquid volume flying through the air keeps its own velocity and falls under gravity, instead of hanging in place or being dragged along by the pool's surface velocity. Spray and drops that phi has lost keep a ballistic path, land where they should, and are picked up again by the surface they hit.",
    "Why: only cells phi calls liquid own pressure rows and velocity. V more than about a cell from phi's surface has neither. Gravity is gated on surface occupancy (phi under 2 cells), the projection zeroes faces between two rowless cells, and velocity extension then overwrites those faces with the nearest phi-liquid's velocity. So airborne V floats, or slides with the pool, until sharpening pours it into the nearest body.",
    "Which cells: a cell is airborne when it holds more than 5% of a cell of V (the same liquid threshold the ghost-phi drain uses), its centre phi is more than 1.5 cells outside the surface (it has no pressure row and is not already an extension source), and the 5×5×5 box around it (5×5 in 2D) lies inside the domain and is fully open, with no solid or wall within two cells.",
    "What changes for them: (1) Extension: their authority is raised just above the 0.5 liquid isovalue, so their faces are extension sources, copied outward rather than overwritten, and they count as momentum donors in advection. (2) Gravity: applied to their vertical faces without the 2-cell occupancy gate. (3) Projection: their predicted face velocity is kept rather than zeroed. No pressure acts on them, which is free flight.",
    "The 5% floor is deliberate: transport and sharpening leave a thin V tail above every surface, about 0.5% of a cell 1.5 cells up over a resting pool and 0.001% a cell higher. With the dust floor alone that tail qualified and fell at g, putting 0.65 m/s into a still pool's air. Thinner V near a real drop still moves with it, as an extension target of the drop's faces.",
    "The wall and solid clearance is deliberate: ballistic faces on a film touching a wall are not divergence free, and an earlier version (keepab) tore contact films.",
    "Cost: a 125-cell open-fraction check per candidate cell, skipped when the scene has no solids. One run on the crown splash: 51.4 ms against 48.4 ms (+6%).",
    "Measured (cm12-figure-6, frame 90, this toggle alone): V inside phi 80,425 → 90,782 of about 94,000 cell volumes; relay band 7,303 → 1,191; far orphan V 6,061 → 2,123. The largest single gain of the splash toggles. With the dust floor in place of the 5% floor it reached 92,003, but it also set a still pool's air moving.",
    "Off: airborne V has no velocity of its own.",
  ].join("\n\n"),
});
const splash: MethodParamSpec[] = [
  {kind:"select",key:"redistanceSurface",label:"Redistance surface",default:"rebuild",tier:"fine",update:"runtime",
    options:[{value:"rebuild",label:"Rebuild"},{value:"preserve",label:"Preserve"},{value:"sparse",label:"Preserve, every 10th"}],
    hint:"Rebuild re-measures every band vertex against the trilinear contour each step, which moves a curved surface inward by up to h²/4r a step even at rest. Preserve leaves every vertex of a cell the surface crosses at its advected value and clamps the surface's edge neighbours to one cell (CM11b Sec. 3.4). Every 10th also redistances only one step in ten, as that paper does."},
  {kind:"select",key:"phiCubicAdvection",label:"Cubic phi advection",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiCubicAdvection},
  {kind:"select",key:"orphanVolume",label:"Orphan volume",default:"relay",tier:"fine",update:"runtime",
    options:[{value:"relay",label:"Relay"},{value:"local",label:"Local"},{value:"compact",label:"Compact"}],
    hint:"What sharpening does with V that phi has lost. Relay pours it down phi's gradient into the nearest surface within 2.1h, across empty air if need be. Local lets an air cell receive only if it already holds V or sits within a cell of the surface, so V cannot jump a gap into another body. Compact also gathers V far from any surface up its own gradient, into full cells."},
  {kind:"select",key:"orphanVolumeRender",label:"Show orphan volume",default:"off",tier:"fine",update:"runtime",
    options:[{value:"off",label:"Off"},{value:"density",label:"Density"},{value:"spheres",label:"Spheres"}],
    hint:"Publish V more than 1.5 cells from any phi surface, which phi alone draws as nothing. Density is CM12 Sec. 3.8's rho/gamma amplification over 3³ cells; Spheres draws each 3³ cluster as a sphere of its own volume. Presentation only: no solver stage reads it."},
  {kind:"select",key:"isolatedBodyVolume",label:"Isolated body volume",default:"off",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:"A vertex within a cell of the surface whose 12³ window holds a whole body -- nothing in the window's outer shell -- moves phi along its normal by that body's own V minus fill, at most a quarter cell a step. A pool never qualifies, so it cannot pump. Replaces Follow V while on."},
  {kind:"select",key:"phiSeedCells",label:"Seed phi from V cells",default:"off",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:"Where no cell centre nearby is phi-liquid, write the implied depth of the fullest incident cell holding over half its capacity, read at the vertex's departure point. Unlike the averaged seed this fires for a single full cell, so a compacted drop owns a pressure row."},
  {kind:"select",key:"phiDrain",label:"Drain ghost phi",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.phiDrain},
  {kind:"select",key:"airborneMomentum",label:"Airborne momentum",default:"on",tier:"fine",update:"runtime",
    options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
    hint:UNIFORM_GEOMETRIC_SPLASH_HINTS.airborneMomentum},
];
params.push(...splash);
export const UNIFORM_GEOMETRIC_SPLASH_KEYS: ReadonlySet<string> = new Set(splash.map(p => p.key));

params.push({kind:"select",key:"totalSurfaceVolume",label:"Total surface volume",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],
  hint:"Shift the existing surface uniformly along its normals to match total conservative V. Bounded to one cell per step; no regional correction or phi seeding. One global constraint across all liquid bodies. Includes continuous inflow."});

params.push({kind:"select",key:"surfaceDeficitBalancing",label:"Surface-deficit balancing",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Preserve overfill expansion and balance it globally with contraction in underfilled pressure-liquid cells, weighted by the existing surface-fill estimate. Reduces persistent sloshing; capped to the available deficit per step."});

/** Shared by the studio and scene harnesses. */
export const UNIFORM_GEOMETRIC_PARAMS: readonly MethodParamSpec[] = Object.freeze(params);
/** Storage is a WebGPU implementation choice, excluded from the native contract. */
export const UNIFORM_GEOMETRIC_NATIVE_PARAMS = Object.freeze(params.filter(p => p.key !== "volumeStorage" && p.key !== "pageSize" && !UNIFORM_GEOMETRIC_SPLASH_KEYS.has(p.key)));
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
