import type { FluidWasmModule } from "../../lib/physics-wasm/module";
export function loadFluidWasmForNode(repository?: string, options?: {
  artifact?: "scalar" | "simd" | "threaded";
  threadCount?: number;
}): Promise<FluidWasmModule>;
