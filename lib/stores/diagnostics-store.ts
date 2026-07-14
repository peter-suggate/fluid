import { create } from "zustand";
import type { GPUStatus } from "../webgpu-renderer";
import type { GPUEulerianInfo } from "../webgpu-eulerian";
import type { EulerianDiagnostics, EulerianRenderState } from "../eulerian-solver";
import type { RigidBodyState, RigidStepDiagnostics } from "../rigid-body";
import type { CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { MetricSample } from "../model";

export interface PerformanceSnapshot {
  cpuSimulation_ms: number;
  cpuFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  gpuAdvection_ms: number;
  gpuLayerConstruction_ms: number;
  gpuPressure_ms: number;
  gpuProjection_ms: number;
  gpuRigid_ms: number;
  gpuDiagnostics_ms: number;
  gpuOverhead_ms: number;
  gpuRender_ms: number;
}

export const emptyPerformance: PerformanceSnapshot = { cpuSimulation_ms: 0, cpuFrame_ms: 0, cpuPhysicsSubmit_ms: 0, cpuDataUpload_ms: 0, cpuRenderEncode_ms: 0, gpuLayerConstruction_ms: 0, gpuAdvection_ms: 0, gpuPressure_ms: 0, gpuProjection_ms: 0, gpuRigid_ms: 0, gpuDiagnostics_ms: 0, gpuOverhead_ms: 0, gpuRender_ms: 0 };

export const emptyCoupling: CouplingDiagnostics = { displacedVolume_m3: 0, bodyImpulse_N_s: { x: 0, y: 0, z: 0 }, fluidReactionImpulse_N_s: { x: 0, y: 0, z: 0 }, momentumClosureError_N_s: 0, coupledBodyCount: 0 };

/**
 * Read-only outputs of the running simulation, published by the controller at
 * step/readback cadence. Panels subscribe selectively so per-frame churn does
 * not re-render the whole shell.
 */
interface DiagnosticsStore {
  bodies: RigidBodyState[];
  rigidState: RigidStepDiagnostics | null;
  fluidState: EulerianDiagnostics | null;
  fluidRenderState: EulerianRenderState | null;
  couplingState: CouplingDiagnostics;
  gpuStatus: GPUStatus;
  gpuInfo: GPUEulerianInfo | null;
  frameMs: number;
  resolution: string;
  samples: MetricSample[];
  performanceSnapshot: PerformanceSnapshot;
  performanceHistory: PerformanceSnapshot[];
  set: (patch: Partial<DiagnosticsStore>) => void;
  pushPerformance: (snapshot: PerformanceSnapshot, sample: MetricSample) => void;
}

export const useDiagnosticsStore = create<DiagnosticsStore>((set) => ({
  bodies: [],
  rigidState: null,
  fluidState: null,
  fluidRenderState: null,
  couplingState: emptyCoupling,
  gpuStatus: { state: "initializing", label: "Initializing WebGPU" },
  gpuInfo: null,
  frameMs: 0,
  resolution: "—",
  samples: [],
  performanceSnapshot: emptyPerformance,
  performanceHistory: [],
  set: (patch) => set(patch),
  pushPerformance: (snapshot, sample) => set((state) => ({
    performanceSnapshot: snapshot,
    performanceHistory: [...state.performanceHistory.slice(-119), snapshot],
    samples: [...state.samples.slice(-79), sample]
  }))
}));
