import {
  CM12_SHARPENING_DISTANCE_CELLS,
  CM12_SHARPENING_TRACE_STEPS,
} from "../../core/cm12-numerics";

/**
 * Sharpening bounds, shared by the panel spec and the solver.
 *
 * These live in a leaf module rather than in the resident solver because the
 * method's parameter definitions declare the same bounds and are themselves
 * imported by the solver: reading the constants out of the resident would
 * close an import cycle and leave `ALGORITHM_PARAMS` uninitialized at the
 * point the stage registry builds its controls.
 */

/** Shared CM12 Algorithm 2 return distance; longer traces remain an explicit setting. */
export const SPARSE_CM12_SHARPENING_DISTANCE_CELLS = CM12_SHARPENING_DISTANCE_CELLS;
export const SPARSE_CM12_SHARPENING_TRACE_STEPS = CM12_SHARPENING_TRACE_STEPS;
export const SPARSE_CM12_SHARPENING_STRENGTH = 1;

/** Kept inside the paper's own D range; the panel spec declares the same bounds. */
export const sparseCM12SharpeningDistance = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(3.1, Math.max(0.1, value))
    : SPARSE_CM12_SHARPENING_DISTANCE_CELLS;

export const sparseCM12SharpeningTraceSteps = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(16, Math.max(1, Math.round(value)))
    : SPARSE_CM12_SHARPENING_TRACE_STEPS;

/**
 * Every live volume-sharpening reader clamps the dose to [0,1]
 * (`resident-volume.wgsl.ts` lines 1081, 1155 and 1285), so the panel exposes
 * that range. The wider host bound is retained for diagnostic constructors
 * that predate the volume path.
 */
export const sparseCM12SharpeningStrength = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(4, Math.max(0, value))
    : SPARSE_CM12_SHARPENING_STRENGTH;

/** One sweep updates both ping-pong distance banks. Zero disables far return. */
export const ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS = 8;
export const ADAPTIVE_VOLUME_RETURN_ROUNDS = 4;
export const sparseCM12DistanceSweeps = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(16, Math.max(0, Math.round(value)))
    : ADAPTIVE_VOLUME_RETURN_PROPAGATION_PAIRS;
export const sparseCM12ReturnPasses = (value: unknown): number =>
  typeof value === "number" && Number.isFinite(value)
    ? Math.min(16, Math.max(0, Math.round(value)))
    : ADAPTIVE_VOLUME_RETURN_ROUNDS;
