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
import {
  OCTREE_RUNTIME_DIALS,
  OCTREE_RUNTIME_DIAL_KEYS,
  octreeRuntimeDialReadouts,
} from "../octree-shared/octree-runtime-dials";
import { OctreeTechniqueOverlayPipeline } from "../octree-shared/webgpu-octree-technique-overlay";
import { OctreeTechniqueAuditOverlayPipeline } from "../octree-shared/webgpu-octree-technique-audit-overlay";
import {
  octreeConfigurationReadouts,
  octreeDiagnosticRowsFor,
  octreeViewportFailure,
} from "../octree-shared/octree-diagnostics";

/**
 * The live accuracy/frame-time dials, as ordinary method parameters.
 *
 * They are declared from one schema (`octree-shared/octree-runtime-dials.ts`)
 * so the performance strip, the solver, and the URL state cannot disagree
 * about a range or a default. `update: "runtime"` is what keeps a drag off the
 * structural fingerprint: the controller applies the value to the attached
 * solver instead of resetting to t=0.
 */
const runtimeDialParams: MethodParamSpec[] = OCTREE_RUNTIME_DIALS.map((dial) => ({
  kind: "number",
  key: dial.key,
  label: dial.label,
  unit: dial.unit,
  min: dial.min,
  max: dial.max,
  step: dial.step,
  digits: dial.digits,
  default: dial.default,
  tier: "fine",
  update: "runtime",
  hint: dial.hint,
}));

const params: MethodParamSpec[] = [
  { kind: "select", key: "losassoVelocityExtension", label: "Losasso extrapolation", default: "fixed-jacobi", tier: "fine", options: [{ value: "fixed-jacobi", label: "Fixed Jacobi · default" }, { value: "causal-front", label: "Causal layer front" }], hint: "Construction-time A/B control for Section 5 air velocity extension. Causal-front publishes one graph layer per sweep from already-valid inner layers." },
  { kind: "select", key: "globalFineLevelSetFactor", label: "Surface tracking", default: "1", tier: "coarse", options: [{ value: "1", label: "Adaptive finest surface · default" }, { value: "4", label: "4× subcell surface" }, { value: "8", label: "8× subcell surface · experimental" }], hint: "Ando-style factor 1 remeshes the octree itself: every leaf cut by the moving interface stays at the finest pressure/level-set tier, while pure liquid and air grade rapidly to coarse cells away from it. Fixed-point mass is handed conservatively between old and new leaves. Factors 4/8 instead opt into a separate sparse subcell interface band." },
  ...OCTREE_STRUCTURAL_PARAMS,
  { kind: "number", key: "topologyCadenceAdvances", label: "Topology cadence", unit: "advances", min: 1, max: 8, step: 1, digits: 0, default: 8, tier: "fine", hint: "Losasso candidate epochs cover eight accepted advances by default. Each skipped rebuild is represented spatially by an extra dilation ring, keeping the moving interface inside the resident band." },
  ...runtimeDialParams,
];

/** Canonical browser/native construction options for the Losasso method. */
export const losassoSolverOptions = (scene: SceneDescription, quality: GPUQuality, values: MethodParamValues) =>
  octreeSolverOptions(params, resolveOctreeCoarseDynamics({
    backend: "losasso",
    topologyCadenceAdvances: values.topologyCadenceAdvances,
    topologyDisplacementRingsPerAdvance: values.topologyDisplacementRingsPerAdvance,
    losassoVelocityExtension: values.losassoVelocityExtension,
  }), values);

export const losassoMethod: SimulationMethod = {
  id: "losasso",
  label: "Losasso adaptive octree",
  shortLabel: "Losasso",
  badge: "LOSASSO",
  description: "GPU-resident adaptive liquid simulation on a Losasso 2004 octree with a uniformly-fine surface shell.",
  detail: "Axis-aligned face velocities on a strictly 2:1-balanced octree, a uniformly-fine shell around phi=0, resident warm-started MGPCG, and Section 5 air velocity extension. Surface tracking runs on the octree itself at factor one, or on a separate sparse subcell band at factors 4/8.",
  backend: "webgpu",
  resource: {
    id: "fluid.losasso-octree",
    lane: "fluid",
    label: "Losasso octree fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: OCTREE_RESOURCE_PHASE_COPY,
  },
  qualityLabels: { balanced: "paper defaults", high: "paper defaults", ultra: "paper defaults" },
  showQualityControl: false,
  // The octree publishes every catalog field bar one: the technique overlays
  // read its compact debug source, and the generic dense-grid views read the
  // diagnostic textures it materializes on demand. `density` is the exception
  // — it is the uniform method's transported rho, and this method tracks a
  // level set instead, so the view would have nothing to read.
  capabilities: { volumeRendering: true, fencedInitialPresentation: true },
  supportedFieldModes: VISUALIZATION_FIELDS
    .filter((field) => field.mode !== "density")
    .map((field) => field.mode),
  params,
  pressureMapping: "Losasso uses the wide, warm-started V-cycle-preconditioned MGPCG authority over the accepted octree rows.",
  runtimeParamKeys: OCTREE_RUNTIME_DIAL_KEYS,
  // Same table the params above are declared from, so the strip, the solver and
  // the URL state cannot disagree about a range, a default or a readout.
  dials: {
    label: "LOSASSO COARSE BAND · LIVE",
    specs: OCTREE_RUNTIME_DIALS,
    readouts: octreeRuntimeDialReadouts,
  },
  // Every card, readout and alert below reports a transaction only this method
  // runs — a frontier publication, a global-fine generation, a §4.3 executor's
  // convergence. Contributed rather than selected by a panel testing
  // `gridKind === "octree"`: both adaptive methods pass that test, so it never
  // said which of these publications a solver actually makes.
  diagnosticRows: octreeDiagnosticRowsFor("losasso"),
  configurationReadouts: octreeConfigurationReadouts,
  viewportFailure: octreeViewportFailure,
  // The technique overlays draw this method's own compact topology, faces and
  // tetrahedra, against constants generated with the power catalog. They are
  // registered rather than imported so the renderer composes them by key.
  overlayPipelines: {
    "technique-overlay": (device, format, uniformBuffer) =>
      new OctreeTechniqueOverlayPipeline(device, format, uniformBuffer),
    "technique-audit-overlay": (device, format, uniformBuffer) =>
      new OctreeTechniqueAuditOverlayPipeline(device, format, uniformBuffer),
  },
  presetFor: () => ({
    losassoVelocityExtension: "fixed-jacobi",
    maximumLeafSize: "32",
    interfaceRefinementBandCells: 4,
    surfaceRefinementGradingLayers: 1,
    globalFineLevelSetFactor: "1",
    topologyCadenceAdvances: 8,
  }),
  createSolver: (device, scene, quality, values, onRigidLoads) => new WebGPUOctreeEulerianSolver(device, scene, quality, onRigidLoads, losassoSolverOptions(scene, quality, values)),
  harness: async () => (await import("./harness")).losassoHarnessPlugin,
  createSolverAsync: (device, scene, quality, values, onRigidLoads, onProgress, signal) => WebGPUOctreeEulerianSolver.createAsync(
    device, scene, quality, onRigidLoads, losassoSolverOptions(scene, quality, values),
    (label, completed, total, phase, taskId) => onProgress({ phase: phase === "planning" || phase === "adaptive-topology" || phase === "secondary-particles" || phase === "allocation" || phase === "warmup" ? phase : "solver-pipelines", taskId, label, completed, total }),
    signal,
  )
};
