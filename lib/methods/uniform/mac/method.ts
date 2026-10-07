import { composeFeatures } from "../../../framework/composition";
import { parameterVariantFeature, parameterVariantSelections } from "../../../core/method-parameter-variants";
import type { SimulationMethod } from "../../../core/method-contract";
import { MAC_PARAMS, macOptions } from "./parameters";
import { MAC_PIPELINE } from "./pipeline";
import { UniformMacSolver } from "./solver";

/** Numerical references (independent implementation; no third-party source copied):
 * Bridson & Müller-Fischer, SIGGRAPH 2007 course notes:
 * https://www.cs.ubc.ca/~rbridson/fluidsimulation/fluids_notes.pdf
 * Batty's minimal MAC/ghost-fluid reference (its markers are replaced here by vertex phi):
 * https://github.com/christopherbatty/Fluid3D
 * Selle et al., An Unconditionally Stable MacCormack Method, JSC 35 (2008):
 * https://physbam.stanford.edu/papers/stanford2006-09.pdf
 * This baseline is intentionally nonconservative. Surface volume and energy
 * are diagnostics, never targets of a global correction or velocity damping.
 */

const point = "simulation.uniform-mac.algorithms";
const choices = MAC_PARAMS.filter(p => p.kind === "select");
const algorithms = parameterVariantFeature(point, choices, ["simulation.dense-grid"]);
const resolveComposition: SimulationMethod["resolveComposition"] = values => composeFeatures({
  features: [{ id: "simulation.uniform-mac.host", provides: ["simulation.dense-grid", "simulation.vertex-phi", "simulation.pressure-projection", "simulation.surface-publication"] }, algorithms],
  selections: parameterVariantSelections(point, choices, values),
});

export const uniformMacMethod: SimulationMethod = {
  id: "uniform-mac", label: "Uniform MAC baseline", shortLabel: "Uniform MAC", badge: "UNIFORM MAC",
  description: "Conventional uniform MAC / level-set liquid baseline with small CFL-limited substeps.",
  detail: "Bridson-style incompressible MAC projection, ghost-fluid pressure and bounded MacCormack transport. Fixed voxel solids with free-slip walls, gravity, explicit viscosity and surface tension. Level-set volume drift is measured without correction; rigid bodies and inflows are not supported.",
  backend: "webgpu", renderRefinementBelowSolver: true,
  composition: resolveComposition({}), resolveComposition,
  params: MAC_PARAMS, runtimeParamKeys: MAC_PARAMS.map(p => p.key),
  supportedFieldModes: ["structure", "density", "cfl", "speed", "phi", "volume-levelset"],
  qualityLabels: { balanced: "Residual-controlled PCG", high: "Residual-controlled PCG", ultra: "Residual-controlled PCG" },
  resource: { id: "fluid.uniform-mac", lane: "fluid", label: "Uniform MAC baseline", provides: ["fluid-authority", "water-presentation"], blocks: "transport" },
  effectiveStep_s: (_scene, values) => macOptions(values).maxStep,
  pressureMapping: "MAC ghost-fluid Poisson, diagonal PCG, fresh absolute divergence residual acceptance.",
  presetFor: () => ({}), pipelineGraph: async () => MAC_PIPELINE,
  diagnosticRows: info => [
    { id: "mac-grid", label: "Uniform MAC grid", value: info ? `${info.nx} × ${info.ny} × ${info.nz}` : "initializing", unit: info ? `${(info.allocatedBytes / 1048576).toFixed(1)} MiB` : undefined },
    { id: "mac-pressure", label: "Pressure residual", value: info?.pressureResidual?.toExponential(2) ?? "—", unit: "s⁻¹ · fresh b − Ap", tone: info?.pressureSolveConverged ? "good" : "neutral" },
    { id: "mac-iterations", label: "Pressure iterations", value: String(info?.pressureIterationsExecuted ?? "—"), unit: "diagonal PCG" },
    { id: "mac-volume", label: "Level-set volume drift", value: info?.volumeDrift === undefined ? "—" : `${(100 * info.volumeDrift).toFixed(3)}%`, unit: "uncorrected · derived surface volume" },
    { id: "mac-energy", label: "Kinetic energy", value: (info as UniformMacSolver["info"] | undefined)?.macKineticEnergy_J?.toExponential(3) ?? "—", unit: "J · cell-centred estimate" },
  ],
  harness: async () => ({ methodId: "uniform-mac", lane: { compactAdaptivePublication: false, silentFailureTripwires: false, stagedTextureComparison: false, structuredGenerationAudit: false, nativeTerminalReceipt: false, separateFineLevelSetBand: () => false }, environmentVariables: [], applyEnvironmentOverrides: () => {} }),
  createSolverAsync: (device, scene, quality, values, _loads, progress, signal) => UniformMacSolver.createAsync(device, scene, quality, values, progress, signal),
};
