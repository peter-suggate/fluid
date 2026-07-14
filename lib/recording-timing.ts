export const SIMULATION_VIDEO_FRAME_RATE = 30;
export const SIMULATION_VIDEO_FRAME_DURATION_S = 1 / SIMULATION_VIDEO_FRAME_RATE;

/** Number of fixed-rate video samples whose simulation-time boundaries have
 * been crossed by the latest rendered state. */
export function simulationFramesDue(currentSimulation_s: number, nextFrameSimulation_s: number): number {
  if (!Number.isFinite(currentSimulation_s) || !Number.isFinite(nextFrameSimulation_s)) return 0;
  if (currentSimulation_s + 1e-9 < nextFrameSimulation_s) return 0;
  return Math.floor((currentSimulation_s - nextFrameSimulation_s + 1e-9) / SIMULATION_VIDEO_FRAME_DURATION_S) + 1;
}

/**
 * Maps a wall-clock-paced canvas recording back onto the simulation clock.
 *
 * A solver that needs 12 seconds to calculate 3 seconds of physics should be
 * replayed at 4x. The resulting 3-second presentation then preserves SI time:
 * one second in the video represents one simulated second.
 */
export function realTimePlaybackRate(recordedDuration_s: number, simulationDuration_s: number): number {
  if (!Number.isFinite(recordedDuration_s) || !Number.isFinite(simulationDuration_s)) return 1;
  if (recordedDuration_s <= 0 || simulationDuration_s <= 0) return 1;
  return recordedDuration_s / simulationDuration_s;
}

/** MediaRecorder WebM files commonly expose `Infinity` as their metadata
 * duration. Keep the measured capture clock in that case instead of allowing
 * the player to collapse the real-time correction back to 1x. */
export function sourceDurationForPlayback(reportedDuration_s: number, measuredDuration_s: number): number {
  if (Number.isFinite(reportedDuration_s) && reportedDuration_s > 0) return reportedDuration_s;
  if (Number.isFinite(measuredDuration_s) && measuredDuration_s > 0) return measuredDuration_s;
  return 0;
}
