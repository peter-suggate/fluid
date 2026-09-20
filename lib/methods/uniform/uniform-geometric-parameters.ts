import { UNIFORM_PARAMS } from "./parameters";
import { numberValue, type MethodParamSpec, type MethodParamValues } from "../../core/method-contract";

const omitted = new Set(["gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing"]);
const params: MethodParamSpec[] = UNIFORM_PARAMS.filter(p => !omitted.has(p.key)).map(p => {
  if (p.key === "velocityTransport" && p.kind === "select") return { ...p, default: "semi-lagrangian" };
  // The solve window. Same key, same constructor-level update kind and the
  // same "Work box" readout as the paper method's active region; what differs
  // is that the geometric predicate is liquid OR near-surface phi rather than
  // V alone, and that every geometric kernel — the vertex phi lattice
  // included — is dispatched from its origin. Measured on the tall-air fixture
  // (docs/research/uniform-geometric-tall-air-2026-09-19/solve-window-report.md):
  // -23% on a 4x-taller domain, +1.1% on the small one.
  if (p.key === "activeRegion" && p.kind === "select") return { ...p,
    label: "Solve window", default: "on",
    options: [{ value: "on", label: "Liquid window" }, { value: "off", label: "Whole domain" }],
    hint: "Run every kernel, every pressure-multigrid pass and every extension-hierarchy pass on the box holding the liquid, the near-surface band and this step's sources, padded by the largest reach any stage uses and aligned to the 4h tile lattice. A domain that is mostly empty air then costs what its liquid costs. Needs the volume dust floor above zero, which is what makes V exactly zero outside the box. Whole domain is the dense control." };
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
// The CM11a hierarchy planned on the window rather than the domain. It is the
// planner cliff this pays for, not the launches: a tall, mostly empty domain
// falls off lockstep coarsening into semi-coarsening, and the plan it gets has
// half again the levels and twice the passes of the same liquid in a small
// domain. A lattice sized to the liquid gets the small domain's plan back.
// Live, because the instance cache makes it live.
params.push({kind:"select",key:"pressureWindow",label:"Pressure lattice",default:"window",tier:"fine",update:"runtime",
  options:[{value:"window",label:"Liquid window"},{value:"domain",label:"Whole domain"}],
  hint:"Plan the CM11a pressure hierarchy for a lattice that holds the solve window instead of the whole domain, with the origin in simulation cells and the halo outside it treated as far air. Needs the solve window to be on; with it off this has no effect. Whole domain is the control, and is what a violated or freshly reset step falls back to."});
params.push({kind:"select",key:"redistance",label:"Level-set redistancing",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],hint:"Reconstruct metric distance near phi=0 after transport. Disable to isolate contour drift."});
params.push({kind:"select",key:"sharpeningWorkMap",label:"Sharpening work map",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],
  hint:"Skip sharpening work in 4×4×4 tiles with no cell in the admission band. Identical result to the dense schedule; Off retains the dense control for comparison."});
params.push({kind:"number",key:"volumeDustThreshold",label:"Volume dust floor",default:1e-6,tier:"fine",update:"runtime",
  min:0,max:1e-3,step:1e-7,digits:7,unit:"cell volumes",
  hint:"Discard |V| below this wherever transport or sharpening writes V, ULP-scale negatives included. On figure 7 the residue is four fifths of the nonzero cells and a ten-millionth of the mass, and it keeps every tile it touches live. Zero is off and stores the untreated sum bit for bit."});
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

params.push({kind:"select",key:"totalSurfaceVolume",label:"Total surface volume",default:"on",tier:"coarse",update:"runtime",
  options:[{value:"off",label:"Off"},{value:"on",label:"On"}],
  hint:"Shift the existing surface uniformly along its normals to match total conservative V. Bounded to one cell per step; no regional correction or phi seeding. One global constraint across all liquid bodies. Source frames are skipped."});

/** Shared by the studio, advance-lab, native Rust generator and scene harnesses. */
export const UNIFORM_GEOMETRIC_PARAMS: readonly MethodParamSpec[] = Object.freeze(params);
export const UNIFORM_GEOMETRIC_DEFAULTS: Readonly<MethodParamValues> = Object.freeze(
  Object.fromEntries(params.map(p => [p.key, p.default])),
);
/** Removed experimental controls never survive a saved configuration reload. */
export function resolveUniformGeometricValues(values: MethodParamValues = {}): MethodParamValues {
  return Object.fromEntries(params.map(spec => {
    const raw = values[spec.key];
    const numeric = spec.kind === "number" ? numberValue(values, params, spec.key) : 0;
    const value = spec.kind === "number" ? (spec.step === 1 ? Math.round(numeric) : numeric)
      : spec.options.some(option => option.value === raw) ? raw! : spec.default;
    return [spec.key, value];
  }));
}
