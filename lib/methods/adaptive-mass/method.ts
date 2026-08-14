import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { ADAPTIVE_MASS_FLUID_PIPELINE } from "./adaptive-mass-frame-pipeline";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";

export type AdaptiveMassSeamAxis = "x" | "y" | "z";
export type AdaptiveMassFineSide = "negative" | "positive";

/** Initial sparse-resolution split consumed by the interactive solver factory. */
export interface AdaptiveMassSolverOptions {
  readonly seamAxis: AdaptiveMassSeamAxis;
  readonly fineSide: AdaptiveMassFineSide;
  readonly fineTileResolution: 8;
  readonly coarseTileResolution: 4;
}

const params: MethodParamSpec[] = [
  {
    kind: "select",
    key: "seamAxis",
    label: "Resolution split axis",
    default: "x",
    tier: "coarse",
    options: [
      { value: "x", label: "X axis" },
      { value: "y", label: "Y axis" },
      { value: "z", label: "Z axis" },
    ],
    hint: "Chooses the initial 4³/8³ resolution split. Changing it rebuilds the sparse solver.",
  },
  {
    kind: "select",
    key: "fineSide",
    label: "Fine half",
    default: "negative",
    tier: "coarse",
    options: [
      { value: "negative", label: "Negative side · 8³" },
      { value: "positive", label: "Positive side · 8³" },
    ],
    hint: "Prefers 8³ bricks on this side of the initial split and 4³ bricks on the other. Changing it rebuilds the sparse solver.",
  },
];

const seamAxis = (value: unknown): AdaptiveMassSeamAxis =>
  value === "y" || value === "z" ? value : "x";

const fineSide = (value: unknown): AdaptiveMassFineSide =>
  value === "positive" ? value : "negative";

export function adaptiveMassSolverOptions(
  values: MethodParamValues,
): AdaptiveMassSolverOptions {
  return {
    seamAxis: seamAxis(values.seamAxis),
    fineSide: fineSide(values.fineSide),
    fineTileResolution: 8,
    coarseTileResolution: 4,
  };
}

export const adaptiveMassMethod: SimulationMethod = {
  id: "adaptive-mass",
  label: "Sparse CM12",
  shortLabel: "Sparse CM12",
  badge: "SPARSE CM12",
  description: "Sparse 4³/8³ world-brick expansion of the uniform CM12 mass-conserving method.",
  detail: "Sparse CM12 maps any authored scene into a fixed-world-space brick atlas, retains no dry bricks, and couples 4³ and 8³ neighbours through shared conservative transport and a global composite pressure solve. This milestone runs the sparse physics authority on CPU and publishes its fields through WebGPU; the final zero-empty-work GPU page pool and camera/activity-driven resolution policy are not yet claimed.",
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
  capabilities: { volumeRendering: true },
  supportedFieldModes: ["structure", "resolution", "density", "cfl", "speed", "phi", "pressure"],
  params,
  pipelineGraph: async () => ADAPTIVE_MASS_FLUID_PIPELINE,
  pressureMapping: "Every live Sparse CM12 step solves one globally coupled composite pressure system over regular faces and conservative 2:1 seam ports using matrix-free Jacobi-PCG.",
  normalizeValues: (values) => ({
    ...values,
    seamAxis: seamAxis(values.seamAxis),
    fineSide: fineSide(values.fineSide),
  }),
  presetFor: () => ({
    seamAxis: "x",
    fineSide: "negative",
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
