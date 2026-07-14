import type { SceneDescription } from "../model";
import type { GPUQuality } from "../tall-cell-grid";
import type { GPUEulerianInfo, GPURigidLoad } from "../webgpu-eulerian";
import type { RigidBodyState } from "../rigid-body";

/**
 * Method plugin contract.
 *
 * A simulation method owns three things:
 *  1. its identity and descriptive copy (picker labels, viewport badge, the
 *     technical summary shown in diagnostics),
 *  2. a declarative parameter schema — common numerics the method consumes
 *     plus any method-specific knobs — which the UI renders generically, and
 *  3. a solver factory that receives the scene and the resolved parameter
 *     values and returns a GPU solver honouring the shared solver interface.
 *
 * Adding a new method means adding one module under lib/methods/ and listing
 * it in the registry; no UI or renderer changes are required.
 */

export type MethodParamValue = number | string | boolean;
export type MethodParamValues = Record<string, MethodParamValue>;

interface ParamBase {
  key: string;
  label: string;
  /** One-line explanation shown under the control. */
  hint?: string;
  /** Coarse controls are always visible; fine controls sit behind "Advanced". */
  tier: "coarse" | "fine";
}

export interface NumberParamSpec extends ParamBase {
  kind: "number";
  unit: string;
  min: number;
  max: number;
  step: number;
  digits?: number;
  default: number;
}

export interface SelectParamSpec extends ParamBase {
  kind: "select";
  options: ReadonlyArray<{ value: string; label: string }>;
  default: string;
}

export type MethodParamSpec = NumberParamSpec | SelectParamSpec;

/** Minimal interface the renderer needs from a GPU solver. */
export interface GPUSolverInstance {
  readonly info: GPUEulerianInfo;
  readonly volumeTexture: GPUTexture;
  readonly columnBaseTexture: GPUTexture;
  advanceTo(time_s: number, bodies: RigidBodyState[]): boolean;
  readStats(): Promise<GPUEulerianInfo>;
  destroy(): void;
}

export interface SimulationMethod {
  id: string;
  /** Full name shown in the method picker. */
  label: string;
  /** Short name for segmented controls and the top bar. */
  shortLabel: string;
  /** Upper-case badge shown over the viewport. */
  badge: string;
  /** One-sentence summary for the picker. */
  description: string;
  /** Technical summary for the diagnostics panel. */
  detail: string;
  /** Where the authoritative fluid state lives. */
  backend: "webgpu" | "cpu";
  /** Per-quality flavour text for the quality selector. */
  qualityLabels: Record<GPUQuality, string>;
  /**
   * Method-specific parameters. Common parameters (resolution, time step,
   * pressure solve effort) live in the scene numerics and are declared once
   * in the common schema, not here.
   */
  params: ReadonlyArray<MethodParamSpec>;
  /**
   * How this method interprets the common "pressure solve effort" iteration
   * budget (scene.numerics.pressureMaxIterations). Purely descriptive; the
   * mapping itself happens in createSolver.
   */
  pressureMapping: string;
  /**
   * Effective parameter values implied by a quality preset. The UI shows
   * these as the baseline; user overrides are stored sparsely on top and
   * merged via resolveMethodValues before reaching createSolver.
   */
  presetFor(quality: GPUQuality): MethodParamValues;
  /** WebGPU methods create a solver; the CPU reference method does not. */
  createSolver?(
    device: GPUDevice,
    scene: SceneDescription,
    quality: GPUQuality,
    values: MethodParamValues,
    onRigidLoads?: (loads: GPURigidLoad[]) => void
  ): GPUSolverInstance;
}

export function resolveMethodValues(method: SimulationMethod, quality: GPUQuality, overrides: MethodParamValues): MethodParamValues {
  const defaults = Object.fromEntries(method.params.map((spec) => [spec.key, spec.default]));
  return { ...defaults, ...method.presetFor(quality), ...overrides };
}

export function numberValue(values: MethodParamValues, spec: ReadonlyArray<MethodParamSpec>, key: string): number {
  const declared = spec.find((candidate) => candidate.key === key);
  const raw = values[key];
  if (typeof raw === "number" && Number.isFinite(raw)) {
    if (declared?.kind === "number") return Math.min(declared.max, Math.max(declared.min, raw));
    return raw;
  }
  return declared?.kind === "number" ? declared.default : 0;
}
