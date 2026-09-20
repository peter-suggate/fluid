import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { UNIFORM_GEOMETRIC_PARAMS as params, resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformMethod } from "./method";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";
import type { SimulationMethod, MethodParamValues } from "../../core/method-contract";

const point = "simulation.uniform-volume.algorithms";
const choices = params.filter(p => p.kind === "select");
const algorithmFeature = parameterVariantFeature(point, choices, ["simulation.page-domain"]);
const resolveComposition = (values: MethodParamValues = {}) => composeFeatures({
  features: [{ id: "simulation.uniform-volume.host", provides: ["simulation.page-domain", "simulation.vertex-phi", "simulation.conservative-volume"] }, algorithmFeature],
  selections: parameterVariantSelections(point, choices, values),
});

/** Fixed-resolution page-domain method; dense backing is a migration adapter. */
export const uniformVolumeMethod: SimulationMethod = {
  ...uniformMethod,
  id: "uniform-volume",
  label: "Uniform Geometric",
  shortLabel: "Uniform Geometric",
  badge: "UNIFORM GEOMETRIC",
  supportedFieldModes: [...uniformMethod.supportedFieldModes!.filter(mode=>mode!=="solve-window"), "volume-levelset", "fine-tiles"],
  description: "All-fine vertex level set and conservative liquid volume.",
  detail: "Fixed-resolution page-domain migration: page-owned cell and vertex execution, conservative volume transport and phi surface geometry. Persistent fields and pressure currently retain dense backing while their sparse replacements are implemented.",
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
