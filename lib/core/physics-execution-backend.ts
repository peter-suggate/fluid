import type { MethodParamSpec, MethodParamValues } from "./method-contract";

/** Where a method executes fluid physics. Rendering remains WebGPU in both cases. */
export type PhysicsExecutionBackend = "gpu" | "cpu";

export const PHYSICS_EXECUTION_BACKEND_KEY = "physicsExecutionBackend";

/**
 * Method settings factory for the full 3D CPU cutover.
 *
 * Returning no setting until the caller has a real CPU world prevents the
 * generic method UI from advertising a backend that cannot run the scene.
 */
export function physicsExecutionBackendParams(
  cpuWorldAvailable: boolean,
): readonly MethodParamSpec[] {
  if (!cpuWorldAvailable) return [];
  return [{
    kind: "select",
    key: PHYSICS_EXECUTION_BACKEND_KEY,
    label: "Physics backend",
    default: "gpu",
    tier: "coarse",
    options: [
      { value: "gpu", label: "GPU · default" },
      { value: "cpu", label: "CPU · Rust/Wasm" },
    ],
    hint: "Choose where fluid physics runs. Both choices use the GPU renderer.",
  }];
}

export function resolvePhysicsExecutionBackend(
  values: MethodParamValues,
): PhysicsExecutionBackend {
  return values[PHYSICS_EXECUTION_BACKEND_KEY] === "cpu" ? "cpu" : "gpu";
}
