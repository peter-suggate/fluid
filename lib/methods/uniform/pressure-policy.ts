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
// The start-up kick's absolute bound (UniformMixedFrame.kick), per second of
// step: 0.3 dt s^-1. Every frame warm-starts from the kick's pressure, so
// what it leaves is what a resting pool starts with: an accepted h-equivalent
// divergence r is a face speed r h, a surface travel r h dt = 0.3 h dt^2 in
// frame 1 (8e-5 h at 1/60, 1.3e-3 h at 1/15) against gravity's own g dt^2/2.
// At the frame tolerance above (one cycle, tenfold) the 128x96x128 pool kept
// 0.066 m/s at 1/15 and spiked to 2 m/s; at this bound the kick leaves it
// 6e-5 m/s at 1/60 and 2.5e-4 m/s at 1/15, in three cycles of the seven-slot
// envelope, and it then rests. The bound stays above the single-precision
// floor (1e-4 s^-1) down to dt = 1/3000.
export const UNIFORM_PRESSURE_KICK_RESIDUAL_PER_STEP = 0.3;

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
