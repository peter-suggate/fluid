import type { SceneDescription } from "./model";
import type { RigidBodyState } from "./rigid-body";
import type { GPUQuality } from "./tall-cell-grid";
import { WebGPUEulerianSolver, type GPURigidLoad, type WebGPUEulerianSolverOptions } from "./webgpu-eulerian";

/**
 * Independently selectable implementation of Narita and Kanai's adaptive
 * optical-layer construction. It shares the verified packed projection and
 * remapping operators with the fixed tall-cell solver, but owns a distinct
 * layout, per-step layer planner, GPU state, diagnostics, and mode identity.
 */
export class WebGPUAdaptiveOpticalLayerSolver extends WebGPUEulerianSolver {
  constructor(device: GPUDevice, scene: SceneDescription, quality: GPUQuality, onRigidLoads?: (loads: GPURigidLoad[]) => void, options: Omit<WebGPUEulerianSolverOptions, "adaptiveOpticalLayer"> = {}) {
    super(device, scene, quality, onRigidLoads, { ...options, adaptiveOpticalLayer: true });
  }

  override advanceTo(time_s: number, bodies: RigidBodyState[] = []) {
    return super.advanceTo(time_s, bodies);
  }
}
