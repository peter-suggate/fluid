import { composeFeatures } from "../../framework/composition";
import type { SimulationMethod } from "../../core/method-contract";
import { SPH_PARAMS } from "./parameters";
import { SPH_PIPELINE } from "./pipeline";
import { SphSolver, type SphInfo } from "./solver";
const composition = composeFeatures({ features: [{ id: "simulation.particle-sph", provides:
  ["simulation.particle-transport", "simulation.surface-publication"] }] });
export const sphMethod: SimulationMethod = {
  id: "particle-sph", label: "Traditional SPH", shortLabel: "SPH", badge: "SPH",
  description: "Simple weakly compressible particles with explicit pressure and viscosity forces.",
  detail: "Experimental traditional SPH: poly6 density, spiky pressure, viscosity and surface tension, with GPU neighbor bins and the shared water surface. Existing scenes are supported; rigid bodies and inflows are ignored. Fixed voxel solids collide with particles. Live detail and solid edits require a reset.",
  backend: "webgpu", composition, resolveComposition: () => composition, renderRefinementBelowSolver: true,
  capabilities: { volumeRendering: true, adoptsRigidRosterShape: true },
  params: SPH_PARAMS, runtimeParamKeys: SPH_PARAMS.filter(p => p.update === "runtime").map(p => p.key),
  supportedFieldModes: ["structure", "density", "cfl", "speed", "phi", "volume-levelset", "particles"],
  qualityLabels: { balanced: "SPH / scene particle spacing", high: "SPH / scene particle spacing", ultra: "SPH / scene particle spacing" },
  resource: { id: "fluid.particle-sph", lane: "fluid", label: "Traditional SPH fluid", provides: ["fluid-authority", "water-presentation"], blocks: "transport" },
  pressureMapping: "Explicit p = c² max(ρ − ρ₀, 0); compressible particles, no pressure iterations.",
  presetFor: () => ({}), pipelineGraph: async () => SPH_PIPELINE,
  diagnosticRows: info => {
    const sph = info as SphInfo | undefined;
    return [
      { id: "sph-count", label: "Material particles", value: sph?.sphParticleCount?.toLocaleString() ?? "—", unit: "SPH" },
      { id: "sph-compression", label: "Maximum density compression", value: sph?.sphMaxCompression === undefined ? "—" : (100 * sph.sphMaxCompression).toFixed(2), unit: "% above rest density" },
      { id: "sph-dt", label: "Last substep", value: info?.lastDt_s === undefined ? "—" : (1000 * info.lastDt_s).toFixed(3), unit: "ms · GPU stability limited" },
      { id: "sph-substeps", label: "Substeps per advance", value: info?.lastSubsteps?.toString() ?? "—", unit: "explicit" },
      { id: "sph-material", label: "Material balance error", value: sph?.sphMaterialDrift === undefined ? "—" : (100 * sph.sphMaterialDrift).toFixed(5), unit: "% · includes measured outflow" },
      { id: "sph-surface", label: "Reconstructed volume drift", value: info?.volumeDrift === undefined ? "—" : (100 * info.volumeDrift).toFixed(3), unit: "% · surface estimate" },
      { id: "sph-memory", label: "SPH allocation", value: info ? (info.allocatedBytes / 1048576).toFixed(1) : "—", unit: "MiB" },
      { id: "sph-scope", label: "Scene compatibility", value: "Bodies and inflows ignored", unit: "experimental" },
    ];
  },
  harness: async () => ({ methodId: "particle-sph", lane: { compactAdaptivePublication: false, silentFailureTripwires: false,
    stagedTextureComparison: false, structuredGenerationAudit: false, nativeTerminalReceipt: false, separateFineLevelSetBand: () => false }, environmentVariables: [], applyEnvironmentOverrides: () => {} }),
  createSolverAsync: (device, scene, quality, values, _loads, progress, signal) => SphSolver.createAsync(device, scene, quality, values, progress, signal),
};
