/**
 * Kernel sources and placements of the brick engine (brick-abi.md §6).
 *
 * A lane exports BrickKernel objects; E's record() places them in the
 * schedule. Bindings are group 0, in order, by name: `cfg` (a 16-byte
 * uniform vec4u baked per dispatch: x the ring slot, y the parity, z and w
 * the placement's own), the engine's buffers (`set`, `dir`, `args`, `vel`,
 * `phi`, `pres`, `cut`, `parts`) and lane-owned buffers by name.
 */

export type BrickBindingKind = "rw" | "ro" | "uniform";
export interface BrickBinding { readonly name: string; readonly kind: BrickBindingKind; readonly type?: string }

export interface BrickKernel {
  readonly label: string;
  readonly entry: string;
  /** WGSL; starts with brickAbiWGSL(...) and declares its bindings with declareBindings(). */
  readonly code: string;
  readonly bindings: readonly BrickBinding[];
}

/** A dispatch: a fixed workgroup count, or indirect from an `args` word (fold rule, brick-abi.md §5.5). */
export type BrickDispatch = { readonly direct: readonly [number, number, number] } | { readonly indirect: number };

export interface BrickPlacement {
  readonly kernel: BrickKernel;
  readonly dispatch: BrickDispatch;
  /** cfg.z and cfg.w (u32). */
  readonly z?: number;
  readonly w?: number;
}

/** The engine's own buffer names; any other binding name is a lane-owned buffer. */
export const ENGINE_BUFFERS = ["set", "dir", "args", "vel", "phi", "pres", "cut", "parts", "xfer", "census"] as const;
export type EngineBufferName = typeof ENGINE_BUFFERS[number];

/** The at-most-8 storage buffers a kernel rule (brick-abi.md §6). */
export const MAX_STORAGE_BINDINGS = 8;

/** WGSL identifiers of engine buffers whose names WGSL reserves (`set`). */
export const WGSL_NAMES: Readonly<Record<string, string>> = { set: "nbSet" };
export const wgslName = (name: string) => WGSL_NAMES[name] ?? name;

/** WGSL declarations of a binding list (group 0, binding = index). */
export function declareBindings(bindings: readonly BrickBinding[]): string {
  return bindings.map((b, i) => {
    const n = wgslName(b.name);
    switch (b.kind) {
      case "rw": return `@group(0) @binding(${i}) var<storage,read_write> ${n}:${b.type ?? "array<u32>"};`;
      case "ro": return `@group(0) @binding(${i}) var<storage,read> ${n}:${b.type ?? "array<u32>"};`;
      case "uniform": return `@group(0) @binding(${i}) var<uniform> ${n}:${b.type ?? "vec4u"};`;
    }
  }).join("\n");
}

export function bindGroupLayoutEntries(bindings: readonly BrickBinding[]): GPUBindGroupLayoutEntry[] {
  const storage = bindings.filter(b => b.kind !== "uniform").length;
  if (storage > MAX_STORAGE_BINDINGS) throw new Error(`Brick engine: a kernel binds ${storage} storage buffers, over ${MAX_STORAGE_BINDINGS}`);
  return bindings.map((b, i): GPUBindGroupLayoutEntry => {
    const base = { binding: i, visibility: 4 /* GPUShaderStage.COMPUTE */ };
    switch (b.kind) {
      case "rw": return { ...base, buffer: { type: "storage" } };
      case "ro": return { ...base, buffer: { type: "read-only-storage" } };
      case "uniform": return { ...base, buffer: { type: "uniform" } };
    }
  });
}

/** The cfg binding every kernel declares first. */
export const CFG: BrickBinding = { name: "cfg", kind: "uniform", type: "vec4u" };
