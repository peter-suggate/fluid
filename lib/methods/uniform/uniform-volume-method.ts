import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { UNIFORM_GEOMETRIC_PARAMS as params, resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformMethod } from "./method";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";
import type { SimulationMethod, MethodParamValues } from "../../core/method-contract";

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
  supportedFieldModes: [...uniformMethod.supportedFieldModes!, "volume-levelset", "fine-tiles", "solve-window"],
  description: "All-fine vertex level set and conservative liquid volume.",
  detail: "Dense specialization of the geometric volume method: RK2 vertex phi, balanced conservative volume transport, phi-only surface geometry and conservative V-only sharpening. Intended for small scenes.",
  resource: { ...uniformMethod.resource!, id: "fluid.uniform-volume", label: "Uniform Geometric fluid" },
  params,
  normalizeValues: resolveUniformGeometricValues,
  composition: resolveComposition(),
  resolveComposition,
  runtimeParamKeys: params.filter(p=>p.update==="runtime").map(p=>p.key),
  pipelineGraph: async () => (await import("./uniform-volume-pipeline")).UNIFORM_VOLUME_PIPELINE,
  harness: async () => ({ ...(await import("./harness")).uniformHarnessPlugin, methodId: "uniform-volume" }),
  createSolverAsync: (device, scene, quality, values, loads, progress, signal) =>
    WebGPUUniformReferenceSolver.createAsync(device, scene, quality, loads,
      uniformGeometricSolverOptions(values, scene), progress, signal),
};
