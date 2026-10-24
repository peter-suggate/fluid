/**
 * The Tune menu's dials (components/UniformCoarseControl.tsx): three writers
 * over NB-FLIP's own params, which stay what the solver reads. A dial runs
 * 0..1 and 0.5 is the app's defaults exactly. Each param it drives moves from
 * its low end to its default over 0..0.5 and on to its high end over 0.5..1:
 * geometrically where all three are positive (tolerances, times, counts),
 * linearly where an end is 0 (Hold and Margin stay 0 below the middle and
 * ramp above it), then rounded to the param's step. The dial's position is a
 * param of its own (NARROW_BAND_TUNE_CONTROLS), kept for the slider to show;
 * a dial whose params the full controls moved off its curve reads Custom.
 */
import type { MethodParamValue, MethodParamValues } from "../../core/method-contract";
import { UNIFORM_DETAIL_RANGES } from "./uniform-detail-policy";
import {
  NARROW_BAND_ACTIVITY_CONTROLS, NARROW_BAND_PARTICLE_CAP_CONTROL as CAP, NARROW_BAND_SPARSE_CONTROL as SPARSE,
  NARROW_BAND_SPARSE_NEIGHBOURS_CONTROL as NEIGHBOURS, NARROW_BAND_TUNE_CONTROLS, narrowBandSurfaceAdvection, type NarrowBandSurfaceAdvection,
} from "./uniform-narrow-band-controls";

export type NarrowBandTuneDial = "surface" | "splash" | "persistence";
export const NARROW_BAND_TUNE_DIALS: readonly NarrowBandTuneDial[] = ["surface", "splash", "persistence"];
/** The param that holds each dial's position. */
export const NARROW_BAND_TUNE_KEYS = Object.freeze({
  surface: NARROW_BAND_TUNE_CONTROLS[0].key, splash: NARROW_BAND_TUNE_CONTROLS[1].key, persistence: NARROW_BAND_TUNE_CONTROLS[2].key,
} as const satisfies Record<NarrowBandTuneDial, string>);

/** One driven param: its value at 0, 0.5 (today's default) and 1, and the step it rounds to. */
interface NarrowBandTuneCurve { readonly key: string; readonly low: number; readonly middle: number; readonly high: number; readonly step: number }
const FADE = NARROW_BAND_ACTIVITY_CONTROLS[1];
const curve = (key: string, low: number, middle: number, high: number, step: number): NarrowBandTuneCurve => ({ key, low, middle, high, step });
const detail = (key: keyof typeof UNIFORM_DETAIL_RANGES, low: number, middle: number, high: number) => curve(key, low, middle, high, UNIFORM_DETAIL_RANGES[key][2]);
export const NARROW_BAND_TUNE_CURVES: Readonly<Record<NarrowBandTuneDial, readonly NarrowBandTuneCurve[]>> = Object.freeze({
  surface: [detail("detailShapeTolerance", 1, 0.5, 0.25), detail("detailThinThickness", 2, 4, 6)],
  splash: [detail("detailImpactTravel", 2, 1, 0.5), detail("detailApproachSteps", 1, 2, 4), curve(CAP.key, 10, 20, 40, CAP.step)],
  persistence: [
    curve(FADE.key, 0.25, 0.5, 1, FADE.step), curve(SPARSE.key, 0.5, 2, 6, SPARSE.step), curve(NEIGHBOURS.key, 32, 16, 8, NEIGHBOURS.step),
    detail("detailHoldSteps", 0, 0, 8), detail("detailMarginTiles", 0, 0, 1),
  ],
});
/** Surface accuracy's advection level: Fast in its bottom quarter, Best in its top one. */
const advection = (position: number): NarrowBandSurfaceAdvection => position < 0.25 ? "fast" : position > 0.75 ? "best" : "balanced";

const decimals = (step: number) => (String(step).split(".")[1] ?? "").length;
const clampPosition = (position: number) => Number.isFinite(position) ? Math.min(1, Math.max(0, position)) : 0.5;
function curveValue(c: NarrowBandTuneCurve, position: number): number {
  const t = clampPosition(position), lower = t <= 0.5, u = lower ? 2 * t : 2 * t - 1;
  const a = lower ? c.low : c.middle, b = lower ? c.middle : c.high;
  const v = a > 0 && b > 0 ? a * (b / a) ** u : a + (b - a) * u;
  return Number((Math.round(v / c.step) * c.step).toFixed(decimals(c.step)));
}

/** The params a dial writes at a position (not its own position param). */
export function narrowBandTuneValues(dial: NarrowBandTuneDial, position: number): MethodParamValues {
  const values: MethodParamValues = Object.fromEntries(NARROW_BAND_TUNE_CURVES[dial].map(c => [c.key, curveValue(c, position)]));
  if (dial === "surface") values.surfaceAdvection = advection(clampPosition(position));
  return values;
}
const same = (key: string, a: MethodParamValue | undefined, b: MethodParamValue) =>
  typeof b === "number" ? Math.abs(Number(a) - b) < 1e-6 : key === "surfaceAdvection" ? narrowBandSurfaceAdvection(a) === b : a === b;
/** Whether resolved values sit on a dial's curve at a position. */
export function narrowBandTuneOnCurve(dial: NarrowBandTuneDial, position: number, values: MethodParamValues): boolean {
  return Object.entries(narrowBandTuneValues(dial, position)).every(([key, want]) => same(key, values[key], want));
}
/**
 * Where a dial shows for resolved values: its stored position while the
 * params match it, else the slider stop nearest it that they do match (a
 * saved run from before the dials, say), else the stored position as Custom.
 */
export function narrowBandTunePosition(dial: NarrowBandTuneDial, values: MethodParamValues): { position: number; custom: boolean } {
  const stored = clampPosition(Number(values[NARROW_BAND_TUNE_KEYS[dial]] ?? 0.5));
  if (narrowBandTuneOnCurve(dial, stored, values)) return { position: stored, custom: false };
  const step = NARROW_BAND_TUNE_CONTROLS[0].step, stops = Math.round(1 / step);
  let best: number | undefined;
  for (let i = 0; i <= stops; i++) {
    const t = Number((i * step).toFixed(decimals(step)));
    if ((best === undefined || Math.abs(t - stored) < Math.abs(best - stored)) && narrowBandTuneOnCurve(dial, t, values)) best = t;
  }
  return best === undefined ? { position: stored, custom: true } : { position: best, custom: false };
}
