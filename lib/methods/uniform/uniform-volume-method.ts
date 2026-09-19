import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformMethod, uniformReferenceSolverOptions } from "./method";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";
import type { SimulationMethod, MethodParamSpec, MethodParamValues } from "../../core/method-contract";

const omitted = new Set(["activeRegion", "gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing"]);
const params: MethodParamSpec[] = uniformMethod.params.filter(p => !omitted.has(p.key)).map(p => {
  if (p.key === "velocityTransport" && p.kind === "select") return { ...p, default: "semi-lagrangian" };
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
params.push({kind:"select",key:"redistance",label:"Level-set redistancing",default:"on",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],hint:"Reconstruct metric distance near phi=0 after transport. Disable to isolate contour drift."});
params.push({kind:"select",key:"liquidCapacityBalancing",label:"Liquid capacity balancing",default:"off",tier:"fine",update:"runtime",
  options:[{value:"on",label:"On"},{value:"off",label:"Off"}],hint:"Balance transport weights against open cell capacity. Disabling skips the balancing rounds while retaining conservative volume gathering."});
params.push({kind:"number",key:"liquidCapacityBalancingRounds",label:"Balancing rounds",default:64,tier:"fine",update:"runtime",
  min:1,max:64,step:1,digits:0,unit:"rounds",hint:"Maximum liquid capacity balancing rounds per step."});
params.push({kind:"number",key:"liquidCapacityBalancingTolerance",label:"Balancing error tolerance",default:0.1,tier:"fine",update:"runtime",
  min:0,max:100,step:0.01,digits:2,unit:"%",hint:"Stop when maximum receiver overfill relative to open cell capacity is at or below this percentage."});
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
const point = "simulation.uniform-volume.algorithms";
const choices = params.filter(p => p.kind === "select");
const algorithmFeature = parameterVariantFeature(point, choices, ["simulation.dense-grid"]);
const resolveComposition = (values: MethodParamValues = {}) => composeFeatures({
  features: [{ id: "simulation.uniform-volume.host", provides: ["simulation.dense-grid", "simulation.vertex-phi", "simulation.conservative-volume"] }, algorithmFeature],
  selections: parameterVariantSelections(point, choices, values),
});

/** Shares dense device services, never adaptive topology or sparse arenas. */
export const uniformVolumeMethod: SimulationMethod = {
  ...uniformMethod,
  id: "uniform-volume",
  label: "Uniform Geometric",
  shortLabel: "Uniform Geometric",
  badge: "UNIFORM GEOMETRIC",
  supportedFieldModes: [...uniformMethod.supportedFieldModes!, "volume-levelset"],
  description: "All-fine vertex level set and conservative liquid volume.",
  detail: "Dense specialization of the geometric volume method: RK2 vertex phi, balanced conservative volume transport, phi-only surface geometry and conservative V-only sharpening. Intended for small scenes.",
  resource: { ...uniformMethod.resource!, id: "fluid.uniform-volume", label: "Uniform Geometric fluid" },
  params,
  composition: resolveComposition(),
  resolveComposition,
  runtimeParamKeys: params.filter(p=>p.update==="runtime").map(p=>p.key),
  pipelineGraph: async () => (await import("./uniform-volume-pipeline")).UNIFORM_VOLUME_PIPELINE,
  harness: async () => ({ ...(await import("./harness")).uniformHarnessPlugin, methodId: "uniform-volume" }),
  createSolverAsync: (device, scene, quality, values, loads, progress, signal) =>
    WebGPUUniformReferenceSolver.createAsync(device, scene, quality, loads, {
      ...uniformReferenceSolverOptions(values, scene), geometricVolume: true,
      velocityTransport: values.velocityTransport === "maccormack" ? "maccormack" : "semi-lagrangian",
      geometricRedistance: values.redistance !== "off",
      liquidCapacityBalancing: values.liquidCapacityBalancing === "on",
      liquidCapacityBalancingRounds: Number(values.liquidCapacityBalancingRounds ?? 64),
      liquidCapacityBalancingTolerance: Number(values.liquidCapacityBalancingTolerance ?? 0.1),
      geometricTileWork: values.sharpeningWorkMap !== "off",
      volumeDustThreshold: Number(values.volumeDustThreshold ?? 1e-6),
      twoLevelVelocity: values.twoLevelVelocity === "on",
      twoLevelFineReach: Number(values.twoLevelFineReach ?? 2),
      twoLevelExtensionTiles: values.twoLevelExtension !== "dense",
      twoLevelShellReach: Number(values.twoLevelShellReach ?? 1),
      twoLevelAdvectionTiles: values.twoLevelAdvection !== "dense",
      activeRegion: false, gammaDiffusionIterations: 0, densityPostProcessing: false,
      solidExcessCorrection: false,
    }, progress, signal),
};
