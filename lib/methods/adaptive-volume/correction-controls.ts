import type { FluidStageControl } from "../../core/fluid-pipeline";
import type { MethodParamSpec, MethodParamValues } from "../../core/method-contract";

/** Runtime experiments; defaults reproduce the existing Sparse CM12 advance. */
export interface SparseCM12CorrectionControls {
  readonly massConservationEnabled?: boolean;
  readonly massConservationStrength?: number;
  readonly gammaConditioningEnabled?: boolean;
  readonly gammaConditioningStrength?: number;
  readonly gammaDiffusionStrength?: number;
  readonly gammaDiffusionIterations?: number;
  readonly sharpeningTau?: number;
  readonly densityCapacityRepairEnabled?: boolean;
  readonly densityCapacityRepairStrength?: number;
  readonly densityCapacityRepairIterations?: number;
  readonly volumeCorrectionEnabled?: boolean;
  readonly volumeCorrectionStrength?: number;
  readonly volumeCorrectionCap?: number;
}

const toggle = (key: string, label: string, hint: string): MethodParamSpec => ({
  kind: "select", key, label, hint, default: "on", tier: "fine", update: "runtime",
  options: [{ value: "on", label: "On" }, { value: "off", label: "Off" }],
});
const dial = (key: string, label: string, value: number, min: number, max: number,
  step: number, unit: string, hint: string): MethodParamSpec => ({
  kind: "number", key, label, default: value, min, max, step, unit, hint,
  digits: step >= 1 ? 0 : 2, tier: "fine", update: "runtime",
});

export const CORRECTION_PARAMS: MethodParamSpec[] = [
  toggle("massConservation", "Mass conservation", "Blend between ordinary backward transport and CM12's conservative transport. Below full strength, mass and momentum conservation are deliberately relaxed."),
  dial("massConservationStrength", "Conservation strength", 1, 0, 1, 0.05, "×",
    "Zero uses backward interpolation; one applies the full column clamp and forward mass return. Full correction is the conservation endpoint."),
  toggle("gammaConditioning", "Persistent gamma conditioning", "Use cumulative gamma to reduce artificial concentration and dilution during transport."),
  dial("gammaConditioningStrength", "Gamma conditioning strength", 1, 0, 2, 0.05, "×",
    "Scales cumulative gamma's departure from one before its existing bounds. Zero ignores gamma history; one is the default; two exaggerates its response."),
  dial("gammaDiffusionStrength", "Diffusion strength", 1, 0, 1, 0.05, "× / pass",
    "Scales paired density and gamma transfers. One is the full stable per-pass dose; increase iterations for stronger diffusion."),
  dial("gammaDiffusionIterations", "Diffusion iterations", 1, 1, 8, 1, "passes",
    "Repeated immutable-snapshot diffusion passes. One reproduces the current solver; additional passes strengthen smoothing without increasing the per-pass dose."),
  dial("sharpeningTau", "Sharpening contrast threshold", 0.4, 0.05, 1, 0.05, "density",
    "The paper's tau. Sharpening fades as neighbour density contrast approaches this threshold; raising it lets sharpening act on steeper interfaces."),
  toggle("densityCapacityRepair", "Density capacity repair", "Relay excess density into neighbouring cells independently of sharpening. Turning it off retains excess mass for pressure-based recovery."),
  dial("densityCapacityRepairStrength", "Excess redistribution", 1, 0, 1, 0.05, "× / pass",
    "Fraction of excess mass relayed per pass, with paired debits and credits. Increase passes for a stronger or farther-reaching repair."),
  dial("densityCapacityRepairIterations", "Capacity repair passes", 8, 1, 16, 1, "passes",
    "Maximum neighbour-to-neighbour relay rounds. Eight is the current default; more rounds allow excess to reach more distant free capacity."),
  toggle("volumeCorrection", "Pressure volume recovery", "Add bounded expansion to the pressure RHS where density exceeds available volume. The incompressibility solve remains active when this correction is off."),
  dial("volumeCorrectionStrength", "Volume recovery strength", 1, 0, 4, 0.05, "×",
    "Scales the paper's lambda = 0.5 before limiting the expansion rate. One is the current default; zero removes the expansion source."),
  dial("volumeCorrectionCap", "Volume recovery cap", 1, 0, 4, 0.05, "eta",
    "The paper's eta limit on expansion. One is the default; higher values allow faster recovery. The existing one-cell-per-step rate bound still applies."),
];

export function correctionValues(values: MethodParamValues): MethodParamValues {
  return Object.fromEntries(CORRECTION_PARAMS.map(spec => {
    const raw = values[spec.key];
    if (spec.kind === "select") return [spec.key, raw === "off" ? "off" : "on"];
    const value = typeof raw === "number" && Number.isFinite(raw) ? raw : Number(spec.default);
    const bounded = Math.min(spec.max!, Math.max(spec.min!, value));
    return [spec.key, spec.step === 1 ? Math.round(bounded) : bounded];
  }));
}

export function correctionOptions(values: MethodParamValues): SparseCM12CorrectionControls {
  const normalized = correctionValues(values);
  return Object.fromEntries(CORRECTION_PARAMS.map(spec => spec.kind === "select"
    ? [`${spec.key}Enabled`, normalized[spec.key] !== "off"]
    : [spec.key, normalized[spec.key]]));
}

/** One normalization path for direct constructors and live UI updates. */
export function normalizedCorrections(options: SparseCM12CorrectionControls = {}) {
  const values = Object.fromEntries(CORRECTION_PARAMS.map(spec => [spec.key,
    spec.kind === "select"
      ? options[`${spec.key}Enabled` as keyof SparseCM12CorrectionControls] === false ? "off" : "on"
      : options[spec.key as keyof SparseCM12CorrectionControls] ?? spec.default]));
  return correctionOptions(values) as Required<SparseCM12CorrectionControls>;
}

/** Stage controls reuse parameter bounds and explanations, including persistence. */
export function correctionStageControl(key: string, toggleKey?: string): FluidStageControl {
  const spec = CORRECTION_PARAMS.find(param => param.key === key);
  if (!spec) throw new Error(`Unknown CM12 correction control: ${key}`);
  if (spec.kind === "select") return {
    kind: "param-choice", param: key, label: spec.label, hint: spec.hint, options: spec.options,
  };
  return {
    kind: "param-range", param: key, label: spec.label, hint: spec.hint,
    min: spec.min!, max: spec.max!, step: spec.step!, digits: spec.digits, unit: spec.unit,
    enabled: toggleKey ? context => context.values[toggleKey] !== "off" : undefined,
  };
}
