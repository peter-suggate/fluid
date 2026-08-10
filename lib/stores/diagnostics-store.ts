import { create } from "zustand";
import type { EffectiveRendererStatus, GPUStatus } from "../webgpu-renderer";
import type { GPUEulerianInfo } from "../webgpu-eulerian";
import type { EulerianDiagnostics, EulerianRenderState } from "../eulerian-solver";
import type { RigidBodyState, RigidStepDiagnostics } from "../rigid-body";
import type { GPURigidBodyPose } from "../webgpu-rigid-body";
import type { CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { MetricSample } from "../model";
import type { WaterSurfacePresentationDiagnostics } from "../webgpu-water-pipeline";
import type { PerformanceTrace } from "../performance-trace";
import {
  initialResourceReadiness,
  reduceGPUResourceEvidence,
  reduceGPUResourceStatus,
  type ResourceReadinessSnapshot,
} from "../resource-readiness";

export interface PerformanceReport {
  methodId: string;
  context: string;
  capturedAt_ms: number;
  cpu?: PerformanceTrace;
  physics?: PerformanceTrace;
  presentation?: PerformanceTrace;
  presentationStages?: PerformanceTrace;
}

export const emptyPerformanceReport: PerformanceReport = {
  methodId: "",
  context: "",
  capturedAt_ms: 0,
};

export const emptyCoupling: CouplingDiagnostics = {
  displacedVolume_m3: 0,
  bodyImpulse_N_s: { x: 0, y: 0, z: 0 },
  fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 },
  momentumClosureError_N_s: 0,
  coupledBodyCount: 0,
};

/**
 * Read-only outputs of the running simulation. Performance data is stored as
 * complete lane traces; individual phases are never copied into flat fields.
 */
interface DiagnosticsStore {
  /**
   * The rigid roster as *commanded*: authored poses, edits, and whatever the
   * last drag left behind. This is what the solver is handed each frame, so it
   * must never be written back from the run — see `bodyPoses` for where the
   * bodies actually are.
   */
  bodies: RigidBodyState[];
  /**
   * Where the drawn frame put each body, by body id.
   *
   * Once a run starts the solver owns rigid motion and never reports back, so a
   * settled crate can be a metre from where `bodies` says it is. Everything the
   * user points at or sees drawn around a body — the hover chip, the selection
   * gizmo, the grab that opens a throw — reads through here so the editor and
   * the image agree. Empty until the renderer publishes its first readback, and
   * bodies absent from it fall back to the commanded pose.
   */
  bodyPoses: Readonly<Record<string, GPURigidBodyPose>>;
  rigidState: RigidStepDiagnostics | null;
  fluidState: EulerianDiagnostics | null;
  fluidRenderState: EulerianRenderState | null;
  couplingState: CouplingDiagnostics;
  gpuStatus: GPUStatus;
  /** Capability-oriented readiness. Unlike gpuStatus, independent work cannot overwrite another lane. */
  resourceReadiness: ResourceReadinessSnapshot;
  gpuInfo: GPUEulerianInfo | null;
  effectiveRendererStatus: EffectiveRendererStatus;
  waterSurfacePresentation: WaterSurfacePresentationDiagnostics | null;
  frameMs: number;
  resolution: string;
  samples: MetricSample[];
  performanceReport: PerformanceReport;
  performanceReports: PerformanceReport[];
  set: (patch: Partial<DiagnosticsStore>) => void;
  pushPerformanceReport: (report: PerformanceReport, sample?: MetricSample) => void;
}

/**
 * The rigid roster as the last frame drew it.
 *
 * `bodies` is what the host has *told* the solver; `bodyPoses` is what the
 * solver did with it. Anything answering a question about the picture — what is
 * under the cursor, where a gizmo goes, where a grab starts — wants this. The
 * frame itself still gets `bodies`, because that is the channel it commands
 * through. On the CPU backend the two are the same object: nothing publishes
 * poses there, and the host roster is already authoritative.
 */
export function drawnBodies(): RigidBodyState[] {
  const state = useDiagnosticsStore.getState();
  return mergeDrawnPoses(state.bodies, state.bodyPoses);
}

/** The same merge for React renders, which hold both halves already. */
export function mergeDrawnPoses(
  bodies: readonly RigidBodyState[],
  bodyPoses: Readonly<Record<string, GPURigidBodyPose>>,
): RigidBodyState[] {
  return bodies.map((body) => {
    const pose = bodyPoses[body.description.id];
    return pose ? { ...body, position_m: pose.position_m, orientation: pose.orientation } : body;
  });
}

export const useDiagnosticsStore = create<DiagnosticsStore>((set) => ({
  bodies: [],
  bodyPoses: {},
  rigidState: null,
  fluidState: null,
  fluidRenderState: null,
  couplingState: emptyCoupling,
  gpuStatus: { state: "initializing", label: "Initializing WebGPU" },
  resourceReadiness: initialResourceReadiness(),
  gpuInfo: null,
  effectiveRendererStatus: {
    state: "pending",
    failureReason: "missing-source",
    silhouetteRefinement: {
      state: "compiling",
      detail: "Waiting for the sparse presentation pipeline",
    },
  },
  waterSurfacePresentation: null,
  frameMs: 0,
  resolution: "—",
  samples: [],
  performanceReport: emptyPerformanceReport,
  performanceReports: [],
  set: (patch) => set((state) => {
    let resourceReadiness = state.resourceReadiness;
    if (patch.gpuStatus) resourceReadiness = reduceGPUResourceStatus(resourceReadiness, patch.gpuStatus);
    if ("gpuInfo" in patch || "effectiveRendererStatus" in patch) {
      resourceReadiness = reduceGPUResourceEvidence(
        resourceReadiness,
        patch.gpuInfo ?? state.gpuInfo,
        patch.effectiveRendererStatus ?? state.effectiveRendererStatus,
      );
    }
    return { ...patch, resourceReadiness };
  }),
  pushPerformanceReport: (report, sample) => set((state) => ({
    performanceReport: report,
    performanceReports: [...state.performanceReports.slice(-239), report],
    samples: sample ? [...state.samples.slice(-79), sample] : state.samples,
  })),
}));
