import { create } from "zustand";
import type { EffectiveRendererStatus, GPUStatus } from "../webgpu-renderer";
import type { GPUEulerianInfo } from "../webgpu-eulerian";
import type { EulerianDiagnostics, EulerianRenderState } from "../eulerian-solver";
import type { RigidBodyState, RigidStepDiagnostics } from "../rigid-body";
import type { CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { MetricSample } from "../model";
import type { WaterSurfacePresentationDiagnostics } from "../webgpu-water-pipeline";
import type { PerformanceTrace } from "../performance-trace";

export interface PerformanceReport {
  methodId: string;
  context: string;
  capturedAt_ms: number;
  cpu?: PerformanceTrace;
  physics?: PerformanceTrace;
  presentation?: PerformanceTrace;
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
  bodies: RigidBodyState[];
  rigidState: RigidStepDiagnostics | null;
  fluidState: EulerianDiagnostics | null;
  fluidRenderState: EulerianRenderState | null;
  couplingState: CouplingDiagnostics;
  gpuStatus: GPUStatus;
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

export const useDiagnosticsStore = create<DiagnosticsStore>((set) => ({
  bodies: [],
  rigidState: null,
  fluidState: null,
  fluidRenderState: null,
  couplingState: emptyCoupling,
  gpuStatus: { state: "initializing", label: "Initializing WebGPU" },
  gpuInfo: null,
  effectiveRendererStatus: {
    requestedMode: "svo",
    effectiveMode: "raster",
    fallbackReason: "missing-source",
  },
  waterSurfacePresentation: null,
  frameMs: 0,
  resolution: "—",
  samples: [],
  performanceReport: emptyPerformanceReport,
  performanceReports: [],
  set: (patch) => set(patch),
  pushPerformanceReport: (report, sample) => set((state) => ({
    performanceReport: report,
    performanceReports: [...state.performanceReports.slice(-239), report],
    samples: sample ? [...state.samples.slice(-79), sample] : state.samples,
  })),
}));
