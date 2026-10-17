/** Backend-neutral CM11a schedule and tolerances. */
export const UNIFORM_CM11A_FULL_CYCLES = 3;
export const UNIFORM_CM11A_V_CYCLES = 4;
// Retain the validated six pre/post sweeps. Full-depth bounds and the
// finest-level acceptance gate protect deeper hierarchies from divergence.
export const UNIFORM_CM11A_PRE_SWEEPS = 6;
export const UNIFORM_CM11A_POST_SWEEPS = 6;
export const UNIFORM_CM11A_PHI_PRESERVATION_LEVELS = 2;
// TallCells reports 1e-4 s^-1 as its GPU/single-precision absolute L-infinity
// tolerance; 1e-8 belongs to its double-precision CPU comparison. CM11a
// itself fixes the cycle schedule but does not prescribe a residual tolerance.
export const UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE = 1e-4;
// Absolute L-infinity divergence left in an accepted solve. At 20 s^-1 one
// cycle was accepted and a resting pool compressed hydrostatically -- V
// 1.1-1.4 by depth, ~10% of the voxel-bath settled tank within 0.5 s -- which
// the total-surface-volume constraint then painted onto drained walls as
// phi-only sheets. 5 s^-1 (three cycles) holds that tank to ~1%.
export const UNIFORM_PRESSURE_RESIDUAL_TOLERANCE = 5;
// The h band starts from p=0 (and the coarse root did, before it warm-started
// from the last presented pressure), so its initial residual IS the RHS. An
// absolute bound alone accepts that zero iterate whenever the unprojected
// divergence is below it: a resting pool's gravity is g dt/h (3.3 s^-1 at
// h=0.1, dt=1/30), so a coarse or small-step scene published p=0, fell
// g dt for a frame and rang into corner jets. A checkpoint must also reduce
// the initial residual tenfold; the floor is the GPU single-precision
// tolerance above, so an already-divergence-free field still accepts.
export const UNIFORM_PRESSURE_RELATIVE_REDUCTION = 0.1;
// Seed the warm start at the solver's absolute single-precision accuracy.
// A timestep-scaled 0.3*dt bound accepted 0.005 s^-1 at 60 Hz: the 8 x 4 m
// voxel pool started with 0.32 mm/s currents, which displaced its flat surface
// at h/4h seams and grew into 0.8 m/s edge flow within a second. The one-time
// kick must establish hydrostatic pressure before transport sees that error;
// later frames retain their normal absolute/relative stopping policy.
export const UNIFORM_PRESSURE_KICK_RESIDUAL_TOLERANCE = UNIFORM_CM11A_COARSE_RESIDUAL_TOLERANCE;

export interface UniformCM11aSchedule {
  readonly fullCycles: number;
  readonly vCycles: number;
  readonly preSweeps: number;
  readonly postSweeps: number;
  readonly residualTolerance?: number;
}

export const DEFAULT_UNIFORM_CM11A_SCHEDULE: UniformCM11aSchedule = Object.freeze({
  fullCycles: UNIFORM_CM11A_FULL_CYCLES,
  vCycles: UNIFORM_CM11A_V_CYCLES,
  preSweeps: UNIFORM_CM11A_PRE_SWEEPS,
  postSweeps: UNIFORM_CM11A_POST_SWEEPS,
  residualTolerance: UNIFORM_PRESSURE_RESIDUAL_TOLERANCE,
});
