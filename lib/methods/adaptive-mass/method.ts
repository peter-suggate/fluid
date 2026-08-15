import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "./adaptive-mass-frame-pipeline";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";

export type AdaptiveMassSeamAxis = "x" | "y" | "z";
export type AdaptiveMassFineSide = "negative" | "positive";
export type AdaptiveMassResolutionMode = "adaptive" | "all-fine" | "all-coarse";

/** Initial sparse-resolution split consumed by the interactive solver factory. */
export interface AdaptiveMassSolverOptions {
  readonly resolutionMode: AdaptiveMassResolutionMode;
  readonly seamAxis: AdaptiveMassSeamAxis;
  readonly fineSide: AdaptiveMassFineSide;
  readonly fineTileResolution: 8;
  readonly coarseTileResolution: 4;
  /** Omitted only by direct diagnostic constructors, which retain scene-step behavior. */
  readonly timeStep?: "paper" | "scene";
}

const params: MethodParamSpec[] = [
  {
    kind: "select",
    key: "resolutionMode",
    label: "Resolution policy",
    default: "adaptive",
    tier: "coarse",
    options: [
      { value: "adaptive", label: "Adaptive · 4³ / 8³" },
      { value: "all-fine", label: "All fine · 8³" },
      { value: "all-coarse", label: "All coarse · 4³" },
    ],
    hint: "Fixed modes keep every resident and newly activated world tile at one rung. They provide matched-resolution parity lanes against fine or reduced Uniform CM12.",
  },
  {
    kind: "select",
    key: "seamAxis",
    label: "Fine seed axis",
    default: "x",
    tier: "coarse",
    options: [
      { value: "x", label: "X axis" },
      { value: "y", label: "Y axis" },
      { value: "z", label: "Z axis" },
    ],
    hint: "Chooses the axis used to place the retained 8³ seam-acceptance seed. Changing it rebuilds the sparse solver.",
  },
  {
    kind: "select",
    key: "fineSide",
    label: "Fine seed side",
    default: "negative",
    tier: "coarse",
    options: [
      { value: "negative", label: "Negative side · 8³" },
      { value: "positive", label: "Positive side · 8³" },
    ],
    hint: "Places one deterministic 8³ seed on this side of each large quiescent component; activity-driven promotion will grow the fine region. Changing it rebuilds the solver.",
  },
  {
    kind: "select",
    key: "timeStep",
    label: "Time step",
    default: "paper",
    tier: "coarse",
    options: [
      { value: "paper", label: "Paper · 1/30 s large steps" },
      { value: "scene", label: "Scene · authored maxDt" },
    ],
    update: "runtime",
    hint: "Sparse CM12 defaults to the same exact 1/30 s operating step as Uniform CM12. Scene mode is retained for matched-step validation and explicit timestep overrides.",
  },
];

/**
 * The controls a live Sparse CM12 solver adopts.
 *
 * Only the clock. The resolution policy and the seam seed decide how the atlas
 * was packed — a different rung or a different fine side is a different world —
 * so they stay structural. `timeStep` is consulted at the top of every
 * `advanceTo` and nowhere else, and it is also what the transport bar's step
 * slider releases: leaving it structural made a nudge of that slider rebuild
 * the sparse world to arrive at an identical one.
 */
export const ADAPTIVE_MASS_RUNTIME_PARAM_KEYS = Object.freeze(["timeStep"] as const);

const seamAxis = (value: unknown): AdaptiveMassSeamAxis =>
  value === "y" || value === "z" ? value : "x";

const fineSide = (value: unknown): AdaptiveMassFineSide =>
  value === "positive" ? value : "negative";

const resolutionMode = (value: unknown): AdaptiveMassResolutionMode =>
  value === "all-fine" || value === "all-coarse" ? value : "adaptive";

export function adaptiveMassSolverOptions(
  values: MethodParamValues,
): AdaptiveMassSolverOptions {
  return {
    resolutionMode: resolutionMode(values.resolutionMode),
    seamAxis: seamAxis(values.seamAxis),
    fineSide: fineSide(values.fineSide),
    fineTileResolution: 8,
    coarseTileResolution: 4,
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
  };
}

export const adaptiveMassMethod: SimulationMethod = {
  id: "adaptive-mass",
  label: "Sparse CM12",
  shortLabel: "Sparse CM12",
  badge: "SPARSE CM12",
  description: "Sparse 4³/8³ world-brick expansion of the uniform CM12 mass-conserving method.",
  detail: "Sparse CM12 maps any authored scene into a fixed-world-space GPU brick atlas and couples 4³ and 8³ neighbours through shared conservative transport and a global composite pressure solve. Compact receiver support, evolving fields, pressure iterations, projection, diagnostics fields, and presentation publication remain GPU-resident between submissions; construction-time topology packing is the only host-sized preparation.",
  backend: "webgpu",
  resource: {
    id: "fluid.sparse-cm12",
    lane: "fluid",
    label: "Sparse CM12 fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Planning the scene's sparse 4³/8³ tile atlas.",
      "adaptive-topology": "Building resident tiles, neighbours, and conservative seam ports.",
      allocation: "Allocating compact fluid authority and water presentation resources.",
      warmup: "Uploading and fencing the initial Sparse CM12 state.",
      attach: "Attaching the warmed adaptive solver atomically.",
    },
  },
  qualityLabels: {
    balanced: "Sparse fixed 4³/8³",
    high: "Sparse fixed 4³/8³",
    ultra: "Sparse fixed 4³/8³",
  },
  showQualityControl: false,
  // `adoptsRigidRosterShape` for the opposite reason to Uniform CM12's: this
  // method has no rigid system at all — `_onRigidLoads` is ignored, `advanceTo`
  // voids its bodies, and nothing under lib/methods/adaptive-mass so much as
  // names `scene.rigidBodies` — so the roster sizes none of its allocations.
  // Keeping the first body in the solver key restarted the scene to buy an
  // identical solver, and a body that the sparse water cannot see is no less
  // invisible after a rebuild than before one.
  capabilities: { volumeRendering: true, adoptsRigidRosterShape: true },
  supportedFieldModes: ["structure", "resolution", "density", "cfl", "speed", "phi", "pressure"],
  params,
  runtimeParamKeys: ADAPTIVE_MASS_RUNTIME_PARAM_KEYS,
  pipelineGraph: async () => ADAPTIVE_MASS_FLUID_PIPELINE,
  pressureMapping: "Every live Sparse CM12 step solves one globally coupled composite pressure system over regular faces and conservative 2:1 seam ports using matrix-free Jacobi-PCG.",
  normalizeValues: (values) => ({
    ...values,
    resolutionMode: resolutionMode(values.resolutionMode),
    seamAxis: seamAxis(values.seamAxis),
    fineSide: fineSide(values.fineSide),
    timeStep: values.timeStep === "scene" ? "scene" : "paper",
  }),
  effectiveStep_s: (_scene, values) =>
    values.timeStep !== "scene" ? CM12_PAPER_DT_S : undefined,
  presetFor: () => ({
    resolutionMode: "adaptive",
    seamAxis: "x",
    fineSide: "negative",
    timeStep: "paper",
  }),
  diagnosticRows: adaptiveMassDiagnosticRows,
  createSolverAsync: (
    device,
    scene,
    quality,
    values,
    onRigidLoads,
    onProgress,
    signal,
  ) => WebGPUAdaptiveMassSolver.createAsync(
    device,
    scene,
    quality,
    onRigidLoads,
    adaptiveMassSolverOptions(values),
    onProgress,
    signal,
  ),
};
