import { composeFeatures } from "../../framework/composition";
import type { SimulationMethod } from "../../core/method-contract";
import { MacGridSolver } from "../mac-shared/solver";
import { APIC_PARAMS, apicOptions, validateApicScene } from "./parameters";
import { APIC_PHASES, APIC_PIPELINE } from "./pipeline";
import { ApicTransport, type ApicInfo } from "./transport";
import type { MacMethodConfiguration } from "../mac-shared/transport";

export const APIC_CONFIGURATION: MacMethodConfiguration = {
  id: "particle-apic", label: "APIC", phases: APIC_PHASES, validateScene: validateApicScene,
  createTransport: ApicTransport.create, sceneWallMode: true, resolveOptions: apicOptions, liveSceneUniforms: true,
  // Metal measurements: cheaper encoding than indirect PCG, with identical acceptance criteria.
  directPressure: true,
  multigridPressure: true, advanceStepCapacity: 64,
  advanceCellStepBudget: 64 * 32 ** 3,
  substepCapacityFactor: 2,
  commandBufferSubsteps: 4,
  continuationSubsteps: 4,
};

const composition = composeFeatures({ features: [{ id: "simulation.particle-apic", provides:
  ["simulation.dense-grid", "simulation.particle-transport", "simulation.pressure-projection", "simulation.surface-publication"] }] });
export const apicMethod: SimulationMethod = {
  id: "particle-apic", label: "APIC particles", shortLabel: "APIC", badge: "APIC",
  description: "APIC, PIC or PIC/FLIP particles with incompressible MAC projection and a reconstructed liquid surface.",
  detail: "Experimental fixed-grid particles: eight particles per initially wet cell, quadratic transfers, GPU binning and ghost-fluid pressure. Choose affine APIC, smooth PIC or blended PIC/FLIP. Gravity, viscosity, surface tension and fixed voxel solids. Inflows, rigid bodies and live simulation detail are not yet supported.",
  backend: "webgpu", renderRefinementBelowSolver: true, composition, resolveComposition: () => composition,
  params: APIC_PARAMS, runtimeParamKeys: APIC_PARAMS.map(param => param.key),
  supportedFieldModes: ["structure", "density", "cfl", "speed", "phi", "volume-levelset", "particles"],
  qualityLabels: { balanced: "APIC / residual-controlled pressure", high: "APIC / residual-controlled pressure", ultra: "APIC / residual-controlled pressure" },
  resource: { id: "fluid.particle-apic", lane: "fluid", label: "APIC particle fluid", provides: ["fluid-authority", "water-presentation"], blocks: "transport" },
  pressureMapping: "MAC ghost-fluid Poisson with Galerkin multigrid-preconditioned CG and fresh divergence-residual acceptance; particles own material transport.",
  presetFor: () => ({}), pipelineGraph: async () => APIC_PIPELINE,
  diagnosticRows: info => {
    const particle = info as ApicInfo | undefined;
    return [
      { id: "apic-mode", label: "Particle transfer", value: particle?.apicTransferMode === 1 ? "PIC" : particle?.apicTransferMode === 2 ? "PIC/FLIP" : "APIC",
        unit: particle?.apicTransferMode === 2 ? `${Math.round(100 * (particle.apicFlipRatio ?? 0.95))}% FLIP` : particle?.apicTransferMode === 1 ? "smooth" : "affine" },
      { id: "apic-count", label: "Material particles", value: particle?.apicParticleCount?.toLocaleString() ?? "—", unit: "particles" },
      { id: "apic-memory", label: "APIC allocation", value: info ? (info.allocatedBytes / 1048576).toFixed(1) : "—", unit: "MiB" },
      { id: "apic-pressure", label: "Pressure residual", value: info?.pressureResidual?.toExponential(2) ?? "—", unit: "s⁻¹", tone: info?.pressureSolveConverged ? "good" : "neutral" },
      { id: "apic-pressure-target", label: "Pressure target", value: info?.pressureResidualTarget?.toExponential(2) ?? "—", unit: "s⁻¹ · absolute + relative" },
      { id: "apic-pressure-work", label: "Pressure iterations", value: info?.pressureIterationsExecuted?.toFixed(0) ?? "—", unit: "MG-PCG · last substep" },
      { id: "apic-dt", label: "Last accepted substep", value: info?.lastDt_s === undefined ? "—" : (1000 * info.lastDt_s).toFixed(3), unit: "ms · GPU stability limited" },
      { id: "apic-transfer-speed", label: "Transfer speed bound", value: particle?.apicAffineSpeedBound_m_s?.toFixed(3) ?? "—", unit: "m/s · includes affine gradients" },
      { id: "apic-material", label: "Material balance error", value: particle?.apicMaterialDrift === undefined ? "—" : (100 * particle.apicMaterialDrift).toFixed(5), unit: "% · includes measured outflow" },
      { id: "apic-surface", label: "Reconstructed volume drift", value: info?.volumeDrift === undefined ? "—" : (100 * info.volumeDrift).toFixed(3), unit: "% · surface estimate" },
      { id: "apic-energy", label: "Particle kinetic energy", value: particle?.apicKineticEnergy_J?.toExponential(3) ?? "—", unit: "J" },
    ];
  },
  harness: async () => ({ methodId: "particle-apic", lane: { compactAdaptivePublication: false, silentFailureTripwires: false,
    stagedTextureComparison: false, structuredGenerationAudit: false, nativeTerminalReceipt: false, separateFineLevelSetBand: () => false }, environmentVariables: [], applyEnvironmentOverrides: () => {} }),
  createSolverAsync: (device, scene, quality, values, _loads, progress, signal) =>
    MacGridSolver.createAsync(device, scene, quality, values, progress, signal, APIC_CONFIGURATION),
};
