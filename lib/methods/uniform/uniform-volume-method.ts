import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformMethod, uniformReferenceSolverOptions } from "./method";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";
import type { SimulationMethod, MethodParamSpec, MethodParamValues } from "../../core/method-contract";

const omitted = new Set(["activeRegion", "gammaDiffusion", "gammaDiffusionIterations", "sharpeningMassCorrection", "solidExcessCorrection", "densityPostProcessing"]);
const params: MethodParamSpec[] = uniformMethod.params.filter(p => !omitted.has(p.key)).map(p => {
  if (p.key === "velocityTransport" && p.kind === "select") return { ...p, default: "semi-lagrangian" };
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
      activeRegion: false, gammaDiffusionIterations: 0, densityPostProcessing: false,
      solidExcessCorrection: false,
    }, progress, signal),
};
