import { uniformGeometricSolverOptions } from "./uniform-geometric-options";
import { UNIFORM_GEOMETRIC_PARAMS as params, resolveUniformGeometricValues } from "./uniform-geometric-parameters";
import { composeFeatures } from "../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../core/method-parameter-variants";
import { uniformMethod } from "./method";
import { WebGPUUniformReferenceSolver } from "./webgpu-uniform-reference";
import type { SimulationMethod, MethodParamValues } from "../../core/method-contract";
import { migrateUniformDetailOverrides, UNIFORM_DETAIL_APP_DEFAULT_POLICY, UNIFORM_DETAIL_APP_IMPORTANCE } from "./uniform-detail-policy";

const point = "simulation.uniform-volume.algorithms";
const choices = params.filter(p => p.kind === "select");
const algorithmFeature = parameterVariantFeature(point, choices, ["simulation.page-domain"]);
const resolveComposition = (values: MethodParamValues = {}) => composeFeatures({
  features: [{ id: "simulation.uniform-volume.host", provides: ["simulation.page-domain", "simulation.vertex-phi", "simulation.conservative-volume"] }, algorithmFeature],
  selections: parameterVariantSelections(point, choices, values),
});

/** One ownership-driven geometric pipeline for fine and mixed layouts. */
export const uniformVolumeMethod: SimulationMethod = {
  ...uniformMethod,
  id: "uniform-volume",
  label: "Uniform Geometric",
  shortLabel: "Uniform Geometric",
  badge: "UNIFORM GEOMETRIC",
  supportedFieldModes: [...uniformMethod.supportedFieldModes!.filter(mode=>mode!=="solve-window"), "volume-levelset", "fine-tiles"],
  // The mixed-ownership frame has no page catalogue, dispatch window or
  // released-face record, and the geometric method has no particles.
  capabilities: { ...uniformMethod.capabilities, visualLayers: { hidden: ["pages", "window", "release", "particles"] } },
  description: "Vertex level set and conservative liquid volume with live simulation detail.",
  detail: "One coupled simulation on a 4h base: Requested detail runs h inside drawn Fine regions and at solid contact, Dynamic follows the surface, Full runs h everywhere; direct h/4h interfaces and conservative live remapping.",
  resource: { ...uniformMethod.resource!, id: "fluid.uniform-volume", label: "Uniform Geometric fluid" },
  params,
  // One 1/60 s advance per 60 Hz frame: presentations follow the solver
  // one-to-one, and the paper's 1/30 s step capped them at 30 FPS.
  // The app opens on Requested detail (4h base, Fine regions in h); lanes
  // and tools resolve the declared default (Dynamic, the GPU census, every
  // surface tile at h). Dynamic in the app allows a smooth surface at 4h.
  appDefaults: { timeStep: "sixtieth", detailPolicy: UNIFORM_DETAIL_APP_DEFAULT_POLICY, ...UNIFORM_DETAIL_APP_IMPORTANCE },
  normalizeValues: resolveUniformGeometricValues,
  migrateOverrides: migrateUniformDetailOverrides,
  composition: resolveComposition(),
  resolveComposition,
  // The uniform solver has no sparse world; the scene is drawn from the
  // renderer's sidecar, which can spend depth the solver never sees.
  renderRefinementBelowSolver: true,
  runtimeParamKeys: params.filter(p=>p.update==="runtime").map(p=>p.key),
  pipelineGraph: async () => (await import("./uniform-volume-pipeline")).UNIFORM_VOLUME_PIPELINE,
  harness: async () => ({ ...(await import("./harness")).uniformHarnessPlugin, methodId: "uniform-volume" }),
  createSolverAsync: (device, scene, quality, values, loads, progress, signal) =>
    WebGPUUniformReferenceSolver.createAsync(device, scene, quality, loads,
      uniformGeometricSolverOptions(values, scene), progress, signal),
};
