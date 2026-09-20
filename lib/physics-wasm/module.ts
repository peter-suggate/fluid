export interface FluidWorldBinding {
  advance(commandSequence: number, dt_s: number): string;
  snapshot(viewMask: number): Uint8Array;
  apply_command(commandJson: string): string;
  receipt(): string;
  free(): void;
}

export interface FluidWorldConstructor {
  from_scene(sceneJson: string, optionsJson: string): FluidWorldBinding;
}

export interface FluidWasmModule {
  default(input?: RequestInfo | URL | Response | BufferSource | WebAssembly.Module
    | { module_or_path: RequestInfo | URL | Response | BufferSource | WebAssembly.Module }): Promise<unknown>;
  readonly FluidWorld: FluidWorldConstructor;
  readonly initThreadPool?: (threadCount: number) => Promise<void>;
  /** Transitional cross-language oracle; excluded from production stepping. */
  readonly run_stage?: (stage: string, graphJson: string, fieldsJson: string,
    optionsJson: string) => string;
  /** Whole-scene uniform migration oracle; not interactive world selection. */
  readonly run_uniform_geometric_scene?: (requestJson: string) => string;
  readonly run_pressure?: (diagonal: Float32Array, rhs: Float32Array,
    pressure: Float32Array, member: Uint8Array, graphJson: string, optionsJson: string) => string;
}

export type FluidWasmModuleLoader = (url: string) => Promise<FluidWasmModule>;

// Keep generated glue in `public/` as a runtime artifact. Vite otherwise
// classifies the import as source code, appends `?import`, and rejects the
// public file before the browser can load its relative Wasm/Rayon modules.
const nativeDynamicImport = new Function(
  "specifier", "return import(specifier)",
) as (specifier: string) => Promise<unknown>;

export const importFluidWasmModule: FluidWasmModuleLoader = async (url) => {
  const loaded = await nativeDynamicImport(url) as FluidWasmModule;
  if (typeof loaded.default !== "function" || typeof loaded.FluidWorld?.from_scene !== "function") {
    throw new TypeError("Generated fluid Wasm module does not expose init and FluidWorld.from_scene");
  }
  return loaded;
};
