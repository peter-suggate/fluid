import type {
  MethodParamSpec,
  MethodParamValues,
  SimulationMethod,
} from "../../core/method-contract";
import { adaptiveMassDiagnosticRows } from "./adaptive-mass-diagnostics";
import { WebGPUAdaptiveMassSolver } from "./webgpu-adaptive-mass-solver";

export type AdaptiveMassSeamAxis = "x" | "y" | "z";
export type AdaptiveMassFineSide = "negative" | "positive";

/** Frozen M1 topology controls consumed by the interactive solver factory. */
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
    label: "M1 seam axis",
    default: "x",
    tier: "coarse",
    options: [
      { value: "x", label: "X axis" },
      { value: "y", label: "Y axis" },
      { value: "z", label: "Z axis" },
    ],
    hint: "Structural acceptance orientation for 4³/8³ neighbours. Changing it rebuilds the sparse solver.",
  },
  {
    kind: "select",
    key: "fineSide",
    label: "M1 fine side",
    default: "negative",
    tier: "coarse",
    options: [
      { value: "negative", label: "Negative side · 8³" },
      { value: "positive", label: "Positive side · 8³" },
    ],
    hint: "Places the 8³ tile on one side of the frozen seam and the 4³ tile on the other. Changing it rebuilds the sparse solver.",
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
  label: "Adaptive mass (Experimental M1)",
  shortLabel: "Adaptive M1",
  badge: "ADAPTIVE M1",
  description: "Experimental sparse 4³/8³ world-tile implementation of the uniform mass-conserving method.",
  detail: "M1 maps an arbitrary authored scene into a sparse fixed-world-space brick atlas, retains no dry bricks, and couples 4³ and 8³ neighbours through the shared conservative transport algebra. The live step is still a CPU prescribed-translation oracle with transient receiver support; zero-empty-work GPU execution, forces, and the general pressure solve are not yet claimed.",
  backend: "webgpu",
  resource: {
    id: "fluid.adaptive-mass-m1",
    lane: "fluid",
    label: "Adaptive mass M1 fluid authority",
    provides: ["fluid-authority", "water-presentation"],
    blocks: "transport",
    phaseCopy: {
      planning: "Planning the scene's sparse 4³/8³ tile atlas.",
      "adaptive-topology": "Building resident tiles, neighbours, and conservative seam ports.",
      allocation: "Allocating compact fluid authority and water presentation resources.",
      warmup: "Uploading and fencing the experimental M1 state.",
      attach: "Attaching the warmed adaptive solver atomically.",
    },
  },
  qualityLabels: {
    balanced: "M1 fixed 4³/8³",
    high: "M1 fixed 4³/8³",
    ultra: "M1 fixed 4³/8³",
  },
  showQualityControl: false,
  capabilities: { volumeRendering: true },
  supportedFieldModes: ["structure", "resolution", "phi"],
  params,
  pressureMapping: "M1 validates the globally coupled composite pressure operator over regular faces and conservative 2:1 seam ports; the interactive prescribed-translation step does not yet execute that solve.",
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
