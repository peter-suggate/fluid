import { CM12_PAPER_DT_S } from "../../core/cm12-numerics";

/** Simulation step used by every example in CM12 Sec. 4. */
export const UNIFORM_PAPER_DT_S = CM12_PAPER_DT_S;

/** Half the paper step: the app's default, one advance per 60 Hz refresh. */
export const UNIFORM_SIXTIETH_DT_S = 1 / 60;

/** The fixed advance a time-step mode pins, or undefined for the scene's own maxDt. */
export function uniformFixedStep_s(timeStep: unknown): number | undefined {
  return timeStep === "scene" ? undefined : timeStep === "sixtieth" ? UNIFORM_SIXTIETH_DT_S : UNIFORM_PAPER_DT_S;
}

/** A fixed-step mode only admits complete advances of its step. */
export function uniformFixedAdvanceReady(requestedTime_s: number, currentTime_s: number, step_s: number): boolean {
  return requestedTime_s - currentTime_s >= step_s - 1e-9;
}
