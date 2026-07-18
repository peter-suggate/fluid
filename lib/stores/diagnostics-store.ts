import { create } from "zustand";
import type { GPUStatus, WaterRenderMode } from "../webgpu-renderer";
import type { GPUEulerianInfo, GPUPhysicsStageId } from "../webgpu-eulerian";
import type { EulerianDiagnostics, EulerianRenderState } from "../eulerian-solver";
import type { RigidBodyState, RigidStepDiagnostics } from "../rigid-body";
import type { CouplingDiagnostics } from "../fluid-rigid-coupling";
import type { MetricSample } from "../model";

export interface PerformanceSnapshot {
  methodId: string;
  waterRenderMode: WaterRenderMode;
  gpuPhysicsTimingAvailable: boolean;
  gpuRenderTimestampSupported: boolean;
  gpuRenderTimingAvailable: boolean;
  cpuSimulation_ms: number;
  cpuFrame_ms: number;
  cpuPhysicsSubmit_ms: number;
  cpuDataUpload_ms: number;
  cpuRenderEncode_ms: number;
  adaptiveRebuildWall_ms: number;
  adaptiveRebuildPending: boolean;
  adaptiveInlineTopology: boolean;
  adaptiveRebuildBlockedFrames: number;
  adaptiveRebuildCompletedCount: number;
  adaptiveGPUConstructionKernel_ms: number;
  adaptiveGPUSparsePack_ms: number;
  adaptiveCPUTopologyPack_ms: number;
  adaptiveCPURedistance_ms: number;
  adaptiveCPUQuadtreeDecode_ms: number;
  adaptiveCPUTallGrid_ms: number;
  adaptiveCPUVariationalAssembly_ms: number;
  adaptiveCPUSystemPack_ms: number;
  adaptiveCPUICFactorization_ms: number;
  adaptiveCPUResourceUpload_ms: number;
  gpuActiveStages: GPUPhysicsStageId[];
  gpuPreparation_ms: number;
  gpuAdvection_ms: number;
  gpuLayerConstruction_ms: number;
  gpuConditioning_ms: number;
  gpuRemeshing_ms: number;
  gpuPressure_ms: number;
  gpuProjection_ms: number;
  gpuExtrapolation_ms: number;
  gpuMaterialization_ms: number;
  gpuSurfaceUpdate_ms: number;
  gpuRigid_ms: number;
  gpuSpraySimulation_ms: number;
  gpuDiagnostics_ms: number;
  gpuOverhead_ms: number;
  gpuRender_ms: number;
  gpuSurfaceExtraction_ms: number;
  gpuDryScene_ms: number;
  gpuInterfaces_ms: number;
  gpuSprayRender_ms: number;
  gpuOpticalComposite_ms: number;
  gpuUpscale_ms: number;
}

export const emptyPerformance: PerformanceSnapshot = { methodId: "", waterRenderMode: "rasterized", gpuPhysicsTimingAvailable: false, gpuRenderTimestampSupported: false, gpuRenderTimingAvailable: false, cpuSimulation_ms: 0, cpuFrame_ms: 0, cpuPhysicsSubmit_ms: 0, cpuDataUpload_ms: 0, cpuRenderEncode_ms: 0, adaptiveRebuildWall_ms: 0, adaptiveRebuildPending: false, adaptiveInlineTopology: false, adaptiveRebuildBlockedFrames: 0, adaptiveRebuildCompletedCount: 0, adaptiveGPUConstructionKernel_ms: 0, adaptiveGPUSparsePack_ms: 0, adaptiveCPUTopologyPack_ms: 0, adaptiveCPURedistance_ms: 0, adaptiveCPUQuadtreeDecode_ms: 0, adaptiveCPUTallGrid_ms: 0, adaptiveCPUVariationalAssembly_ms: 0, adaptiveCPUSystemPack_ms: 0, adaptiveCPUICFactorization_ms: 0, adaptiveCPUResourceUpload_ms: 0, gpuActiveStages: [], gpuPreparation_ms: 0, gpuLayerConstruction_ms: 0, gpuAdvection_ms: 0, gpuConditioning_ms: 0, gpuRemeshing_ms: 0, gpuPressure_ms: 0, gpuProjection_ms: 0, gpuExtrapolation_ms: 0, gpuMaterialization_ms: 0, gpuSurfaceUpdate_ms: 0, gpuRigid_ms: 0, gpuSpraySimulation_ms: 0, gpuDiagnostics_ms: 0, gpuOverhead_ms: 0, gpuRender_ms: 0, gpuSurfaceExtraction_ms: 0, gpuDryScene_ms: 0, gpuInterfaces_ms: 0, gpuSprayRender_ms: 0, gpuOpticalComposite_ms: 0, gpuUpscale_ms: 0 };

/** Sum only measurements that were actually produced by timestamp queries. */
export function measuredGPUTime_ms(sample: PerformanceSnapshot) {
  const physics = sample.gpuPhysicsTimingAvailable
    ? sample.gpuPreparation_ms + sample.gpuLayerConstruction_ms + sample.gpuAdvection_ms
      + sample.gpuConditioning_ms + sample.gpuRemeshing_ms + sample.gpuPressure_ms
      + sample.gpuProjection_ms + sample.gpuExtrapolation_ms + sample.gpuMaterialization_ms
      + sample.gpuSurfaceUpdate_ms + sample.gpuRigid_ms + sample.gpuSpraySimulation_ms
      + sample.gpuDiagnostics_ms + sample.gpuOverhead_ms
    : 0;
  return physics + (sample.gpuRenderTimingAvailable ? sample.gpuRender_ms : 0);
}

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
  pushPerformance: (snapshot: PerformanceSnapshot, sample?: MetricSample) => void;
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
    samples: sample ? [...state.samples.slice(-79), sample] : state.samples
  }))
}));
