import { WebGPUOctreeEulerianSolver } from "../octree-shared/webgpu-octree-eulerian";
import { VISUALIZATION_FIELDS } from "../../core/visualization-catalog";
import type { MethodParamSpec, MethodParamValues, SimulationMethod } from "../../core/method-contract";
import type { GPUQuality } from "../../core/gpu-quality";
import type { SceneDescription } from "../../core/model";
import { resolveOctreeCoarseDynamics } from "../octree-shared/octree-coarse-backend";
import {
  OCTREE_RESOURCE_PHASE_COPY,
  OCTREE_STRUCTURAL_PARAMS,
  octreeSolverOptions,
} from "../octree-shared/octree-method-surface";
import { OctreeTechniqueOverlayPipeline } from "../octree-shared/webgpu-octree-technique-overlay";
import { OctreeTechniqueAuditOverlayPipeline } from "../octree-shared/webgpu-octree-technique-audit-overlay";
import {
  octreeConfigurationReadouts,
  octreeDiagnosticRowsFor,
  octreeViewportFailure,
} from "../octree-shared/octree-diagnostics";

/**
 * The frozen Aanjaneya et al. 2017 reference lane rebuilds its candidate
 * topology on every accepted advance, so cadence is not a dial — a skipped
 * rebuild would leave the power diagram describing a surface that has already
 * moved. It is pinned here rather than exposed as a parameter whose only legal
 * value is one.
 */
const POWER_TOPOLOGY_CADENCE_ADVANCES = 1;

const params: MethodParamSpec[] = [
  // Section 5 evolves a separate uniformly high-resolution SPGrid narrow band
  // alongside the coarse octree phi, and four is the paper's canonical
  // benchmark rung — which is why it is the default here rather than the
  // Losasso default of one. Factor one is still offered: it drops the separate
  // band and tracks the surface on the octree itself, which is the
  // configuration the deep-interior hybrid row census runs, and removing it
  // would take a shipping gate's scene with it.
  { kind: "select", key: "globalFineLevelSetFactor", label: "Subcell surface", default: "4", tier: "coarse", options: [{ value: "1", label: "Coarse octree surface" }, { value: "4", label: "4× subcell surface · benchmark" }, { value: "8", label: "8× subcell surface · experimental" }], hint: "Resolution multiple of the separate sparse SPGrid narrow band this method evolves alongside the coarse octree. Factor one omits that band and tracks the surface on the octree cells themselves." },
  ...OCTREE_STRUCTURAL_PARAMS,
];

/** Canonical browser/native construction options for the Power Liquids method. */
export const powerLiquidsSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) =>
  octreeSolverOptions(params, resolveOctreeCoarseDynamics({
    backend: "power2017",
    topologyCadenceAdvances: POWER_TOPOLOGY_CADENCE_ADVANCES,
  }), values);

export const powerLiquidsMethod: SimulationMethod = {
  id: "power-liquids",
  label: "Power liquids (2017)",
  shortLabel: "Power",
  badge: "POWER LIQUIDS",
  description: "Frozen Aanjaneya et al. 2017 reference: power-diagram faces at T-junctions over a sparse paged grid.",
  detail: "The paper's power-cell pipeline — power-diagram face areas at octree T-junctions, an SPGrid paged pyramid, the Section 4.3 hybrid preconditioner, and a separate uniformly high-resolution narrow band for the surface. Accepts seam maintenance only; it exists as the reproducible comparison against Losasso and the dense uniform reference.",
  backend: "webgpu",
  resource: {
    id: "fluid.power-liquids",
    lane: "fluid",
    label: "Power liquids fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: OCTREE_RESOURCE_PHASE_COPY,
  },
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  // Declared here as well as on Losasso rather than inherited from an "octree
  // family": `gpu-t0-presentation.ts` and `TransportBar.tsx` gate the fenced
  // t=0 handshake on this flag, and a lane that omitted it would unlock the
  // transport before its sparse authority existed.
  capabilities: { volumeRendering: true, fencedInitialPresentation: true },
  supportedFieldModes: VISUALIZATION_FIELDS
    .filter((field) => field.mode !== "density")
    .map((field) => field.mode),
  params,
  pressureMapping: "The frozen Power 2017 backend runs its persistent Section 4.3 hybrid-preconditioned solver over power-diagram rows.",
  // No live dials: every slider on the performance strip drives Losasso
  // coarse-dynamics machinery — the resident MGPCG, its first-order V-cycle,
  // the Section 5 axis-face extension, the candidate topology epoch — and this
  // backend runs none of it. Declaring none is why the strip renders nothing
  // here instead of offering controls that would move nothing at all.
  diagnosticRows: octreeDiagnosticRowsFor("power2017"),
  configurationReadouts: octreeConfigurationReadouts,
  viewportFailure: octreeViewportFailure,
  overlayPipelines: {
    "technique-overlay": (device, format, uniformBuffer) =>
      new OctreeTechniqueOverlayPipeline(device, format, uniformBuffer),
    "technique-audit-overlay": (device, format, uniformBuffer) =>
      new OctreeTechniqueAuditOverlayPipeline(device, format, uniformBuffer),
  },
  presetFor: () => ({
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    surfaceRefinementGradingLayers: 1,
    globalFineLevelSetFactor: "4",
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUOctreeEulerianSolver(device, scene, quality, onRigidLoads, powerLiquidsSolverOptions(scene, quality, values)),
  harness: async () => (await import("./harness")).powerLiquidsHarnessPlugin,
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUOctreeEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, powerLiquidsSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "planning" || phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
